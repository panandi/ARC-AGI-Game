"""FastAPI application: routes, security headers, SSE, and static hosting.

The frontend (``../../frontend``) is served from this same app so the whole
thing deploys as one container. No endpoint here returns configuration beyond
:meth:`Settings.public_config`, and no endpoint accepts a model name, a prompt,
or any other OpenRouter parameter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core.config import Settings, get_settings
from .core.rate_limit import get_rate_limiter
from .schemas.models import (
    CreateRaceRequest,
    CreateRaceResponse,
    HealthResponse,
    PublicConfigResponse,
    PublicTask,
    RaceStatusResponse,
    SubmitRequest,
    SubmitResponse,
    TaskInfo,
    TaskListResponse,
)
from .services.arc_service import ArcUnavailableError, get_backend
from .services.session_manager import (
    ActionRejected,
    RaceError,
    RaceSession,
    SessionManager,
)

logger = logging.getLogger(__name__)

# backend/app/main.py -> repository root -> frontend/
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        # The favicon is an inline SVG data URL.
        "img-src 'self' data:",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
        "font-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ]
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings = get_settings()
    backend = get_backend(settings)
    manager = SessionManager(backend, settings)
    app.state.settings = settings
    app.state.backend = backend
    app.state.manager = manager
    manager.start_sweeper()
    logger.info(
        "ARC Race %s starting (dataset=arc-agi-1, tasks=%d, ai=%s)",
        __version__,
        backend.count,
        "enabled" if settings.ai_enabled else "disabled",
    )
    try:
        yield
    finally:
        await manager.shutdown()


app = FastAPI(
    title="ARC Race",
    version=__version__,
    description="Human vs AI on the same ARC-AGI-1 task.",
    lifespan=lifespan,
    # The interactive docs would otherwise expose request shapes publicly;
    # they stay available in development only.
    docs_url=None if get_settings().is_production else "/docs",
    redoc_url=None,
)

_settings_at_import = get_settings()
if _settings_at_import.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings_at_import.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Response:
    response: Response = await call_next(request)
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_manager(request: Request) -> SessionManager:
    manager: SessionManager | None = getattr(request.app.state, "manager", None)
    if manager is None:  # pragma: no cover - lifespan always sets this
        raise HTTPException(status_code=503, detail="Server is still starting")
    return manager


def settings_dep() -> Settings:
    return get_settings()


def client_ip(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Behind a reverse proxy the first X-Forwarded-For hop is used; deploy with a
    proxy that overwrites this header rather than appending to a client value.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def load_race(race_id: str, manager: SessionManager) -> RaceSession:
    session = await manager.store.get(race_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Race not found or expired")
    session.touch()
    return session


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        dataset="arc-agi-1",
        ai_enabled=settings.ai_enabled,
    )


@app.get("/api/config", response_model=PublicConfigResponse)
async def public_config(
    settings: Settings = Depends(settings_dep),
) -> PublicConfigResponse:
    """Non-secret configuration for the UI. Built from an explicit allow-list."""
    return PublicConfigResponse(**settings.public_config())  # type: ignore[arg-type]


@app.get("/api/tasks", response_model=TaskListResponse)
async def list_tasks(manager: SessionManager = Depends(get_manager)) -> TaskListResponse:
    """The public ARC-AGI-1 training tasks, smallest (easiest) first."""
    return TaskListResponse(
        tasks=[
            TaskInfo(
                task_id=t.task_id,
                rows=t.rows,
                cols=t.cols,
                train_count=t.train_count,
                difficulty=t.difficulty,
            )
            for t in manager.backend.list_tasks()
        ],
        source="arc-agi-1",
    )


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------
@app.post("/api/races", response_model=CreateRaceResponse, status_code=201)
async def create_race(
    body: CreateRaceRequest,
    request: Request,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> CreateRaceResponse:
    decision = await get_rate_limiter().hit(
        f"create:{client_ip(request)}", settings.max_races_per_ip_per_hour, 3600.0
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many races created from this address. Try again later.",
            headers={"Retry-After": str(int(decision.retry_after_seconds) + 1)},
        )

    try:
        session = await manager.create_race(body.task_id, body.let_both_finish)
    except ArcUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RaceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return CreateRaceResponse(
        race_id=session.race_id,
        task_id=session.task_id,
        ai_enabled=session.ai_enabled,
        ai_model=settings.openrouter_model if session.ai_enabled else None,
        max_attempts=session.max_attempts,
        time_limit_seconds=session.time_limit_seconds,
        let_both_finish=session.let_both_finish,
        human=session.human.to_state(0),
        ai=session.ai.to_state(0, reveal=False),
    )


@app.post("/api/races/{race_id}/start", response_model=RaceStatusResponse)
async def start_race(
    race_id: str,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> RaceStatusResponse:
    session = await load_race(race_id, manager)
    if session.finished:
        raise HTTPException(status_code=409, detail="This race has already finished")
    await manager.start_race(session)
    return _status(session, settings)


@app.post("/api/races/{race_id}/submit", response_model=SubmitResponse)
async def submit(
    race_id: str,
    body: SubmitRequest,
    manager: SessionManager = Depends(get_manager),
) -> SubmitResponse:
    session = await load_race(race_id, manager)
    try:
        lane_state, correct = await manager.submit_human(session, body.grid)
    except ActionRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SubmitResponse(accepted=True, correct=correct, lane=lane_state)


@app.post("/api/races/{race_id}/give-up", response_model=RaceStatusResponse)
async def give_up(
    race_id: str,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> RaceStatusResponse:
    session = await load_race(race_id, manager)
    try:
        await manager.give_up(session)
    except ActionRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _status(session, settings)


@app.post("/api/races/{race_id}/stop", response_model=RaceStatusResponse)
async def stop_race(
    race_id: str,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> RaceStatusResponse:
    session = await load_race(race_id, manager)
    await manager.stop_race(session, reason="stopped by player")
    return _status(session, settings)


@app.get("/api/races/{race_id}", response_model=RaceStatusResponse)
async def race_status(
    race_id: str,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> RaceStatusResponse:
    session = await load_race(race_id, manager)
    return _status(session, settings)


@app.get("/api/races/{race_id}/events")
async def race_events(
    race_id: str,
    manager: SessionManager = Depends(get_manager),
    settings: Settings = Depends(settings_dep),
) -> StreamingResponse:
    """Server-Sent Events for one race."""
    session = await load_race(race_id, manager)
    # Subscribe and copy the backlog in one synchronous step, so every event is
    # either replayed or queued -- never both. (Copying later, inside the
    # generator, re-sent anything emitted while the stream was opening.)
    queue = session.subscribe()
    backlog = list(session.event_log)

    async def stream() -> AsyncIterator[bytes]:
        try:
            snapshot = _status(session, settings).model_dump()
            yield _sse("snapshot", {"status": snapshot})
            for event in backlog:
                yield _sse(event["type"], event["data"], event["t"])
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # Comment frame keeps proxies from closing an idle stream.
                    yield b": keep-alive\n\n"
                    continue
                yield _sse(event["type"], event["data"], event["t"])
        except asyncio.CancelledError:  # pragma: no cover - client disconnect
            raise
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event_type: str, data: dict[str, Any], t: int | None = None) -> bytes:
    payload = dict(data)
    if t is not None:
        payload.setdefault("t", t)
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_type}\ndata: {body}\n\n".encode()


def _status(session: RaceSession, settings: Settings) -> RaceStatusResponse:
    elapsed = session.elapsed_ms()
    reveal = session.revealed
    return RaceStatusResponse(
        race_id=session.race_id,
        task_id=session.task_id,
        started=session.started,
        finished=session.finished,
        let_both_finish=session.let_both_finish,
        ai_enabled=session.ai_enabled,
        ai_model=settings.openrouter_model if session.ai_enabled else None,
        max_attempts=session.max_attempts,
        elapsed_ms=elapsed,
        time_limit_seconds=session.time_limit_seconds,
        remaining_ms=session.remaining_ms(),
        timed_out=session.timed_out,
        revealed=reveal,
        human=session.human.to_state(elapsed),
        ai=session.ai.to_state(elapsed, reveal=reveal),
        race_winner=session.race_winner,  # type: ignore[arg-type]
        winner_reason=session.winner_reason,
        efficiency_winner=session.efficiency_winner(),  # type: ignore[arg-type]
        # The puzzle is only handed out once the shared clock is running.
        task=PublicTask(**session.task.public()) if session.started else None,
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

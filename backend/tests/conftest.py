"""Shared test fixtures.

Tests run against a small fixture set of *real* ARC-AGI-1 tasks (copied from
the official dataset into ``tests/fixtures/arc1``) and a mocked OpenRouter
transport. Nothing here contacts OpenRouter.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "arc1"

# Must be set before the application modules are imported: `app.main` reads
# settings at import time, and the developer's .env must not leak in.
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["OPENROUTER_MODEL"] = "test/model-not-real"
os.environ["ENVIRONMENT"] = "development"
os.environ["ARC_DATA_DIR"] = str(FIXTURES)

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.core.rate_limit import InMemoryRateLimiter, set_rate_limiter  # noqa: E402
from app.services.arc_service import TaskLibrary, set_backend  # noqa: E402
from app.services.openrouter_agent import OpenRouterClient  # noqa: E402
from app.services.session_manager import SessionManager  # noqa: E402

# 0d3d703e: a 3x3 colour-substitution task. 6150a2bd: rotate a 3x3 by 180 deg.
TASK_ID = "0d3d703e"
OTHER_TASK_ID = "6150a2bd"

FAKE_OPENROUTER_KEY = "sk-or-v1-test-key-not-real-0000000000"


def task_json(task_id: str = TASK_ID) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{task_id}.json").read_text(encoding="utf-8"))


def solution(task_id: str = TASK_ID) -> list[list[int]]:
    """The hidden answer, straight from the fixture file."""
    return task_json(task_id)["test"][0]["output"]


def wrong_answer(task_id: str = TASK_ID) -> list[list[int]]:
    """Same shape as the answer, one cell different."""
    grid = [row[:] for row in solution(task_id)]
    grid[0][0] = (grid[0][0] + 1) % 10
    return grid


def make_settings(**overrides: Any) -> Settings:
    """A Settings instance with test-friendly defaults."""
    base: dict[str, Any] = {
        "openrouter_api_key": "",
        "openrouter_model": "test/model-not-real",
        "environment": "development",
        "arc_data_dir": str(FIXTURES),
        "ai_turn_min_interval_seconds": 0.0,
        "max_concurrent_ai_races": 4,
        "session_ttl_minutes": 30,
        "race_time_limit_seconds": 300,
        "max_attempts": 3,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _reset_globals() -> Any:
    """Keep module-level singletons from leaking between tests."""
    set_backend(TaskLibrary(FIXTURES))
    set_rate_limiter(InMemoryRateLimiter())
    yield
    set_backend(None)
    set_rate_limiter(None)


@pytest.fixture(autouse=True)
def _no_real_openrouter(monkeypatch) -> None:
    """Only clients built on a mock transport may make requests.

    A client created without one would talk to the real API; fail instead.
    """

    async def refuse(self: OpenRouterClient) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Tests must not reach the real OpenRouter API")
        return self._client

    monkeypatch.setattr(OpenRouterClient, "_http", refuse)


@pytest.fixture
def backend() -> TaskLibrary:
    return TaskLibrary(FIXTURES)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def manager(backend: TaskLibrary, settings: Settings) -> SessionManager:
    return SessionManager(backend, settings)


@pytest.fixture
def client() -> Any:
    """A TestClient whose lifespan wires in the fixture task library."""
    from app.main import app

    with TestClient(app) as test_client:
        test_client.app.state.manager.settings = make_settings()
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# OpenRouter test doubles
# ---------------------------------------------------------------------------
class RecordingOpenRouter:
    """A mocked OpenRouter endpoint that records every request it receives."""

    def __init__(
        self,
        responder: Callable[[int], Any],
        settings: Settings | None = None,
    ) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.settings = settings or make_settings(openrouter_api_key=FAKE_OPENROUTER_KEY)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        self.requests.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "payload": body,
            }
        )
        result = self.responder(len(self.requests) - 1)
        if isinstance(result, httpx.Response):
            return result
        content = result if isinstance(result, str) else json.dumps(result)
        return httpx.Response(
            200,
            json={
                "id": "gen-test",
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        )

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def client(self) -> OpenRouterClient:
        transport = httpx.MockTransport(self._handle)
        return OpenRouterClient(self.settings, client=httpx.AsyncClient(transport=transport))


def answer(grid: list[list[int]], rule: str = "Map each colour to another.") -> dict[str, Any]:
    return {"rule": rule, "grid": grid}


async def create_started_race(
    manager: SessionManager,
    *,
    task_id: str = TASK_ID,
    let_both_finish: bool = True,
) -> Any:
    session = await manager.create_race(task_id, let_both_finish)
    await manager.start_race(session)
    return session

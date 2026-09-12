"""Race sessions: state, locking, event fan-out and expiry.

Each race pits two lanes against the same ARC-AGI-1 task. A lane is one
player's private attempt record: the human and the AI each hold their own
:class:`ArcTaskEnvironment`, their own lock and their own submissions, so
neither can see or disturb the other's work.

Storage
-------
:class:`SessionStore` is the seam for persistence. The MVP ships
:class:`InMemorySessionStore`; a Redis-backed store can be dropped in for
horizontal scaling (see README).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..schemas.models import AttemptInfo, LaneState
from .arc_service import ArcTask, ArcTaskEnvironment, Grid, Observation

logger = logging.getLogger(__name__)

EVENT_QUEUE_SIZE = 256
MAX_EVENT_LOG = 400


class RaceError(RuntimeError):
    """A race-level failure that is safe to surface to the client."""


class ActionRejected(RaceError):
    """The requested action is not currently permitted."""


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------
@dataclass
class Lane:
    """One player's private attempt record and derived scoreboard state."""

    name: str
    env: ArcTaskEnvironment
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    observation: Observation | None = None
    # One entry per submission: {"number", "correct", "t_ms", "grid", "rule"}.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    ai_calls: int = 0
    status: str = "idle"
    # The AI's one-sentence statement of the rule it inferred.
    rule: str | None = None
    error: str | None = None
    finished_at_ms: int | None = None
    completed: bool = False
    completion_ms: int | None = None

    @property
    def finished(self) -> bool:
        return self.finished_at_ms is not None

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    def to_state(self, elapsed_ms: int, *, reveal: bool = True) -> LaneState:
        """Scoreboard view of the lane.

        ``reveal=False`` withholds submitted grids and the rule: an AI answer
        shown mid-race would hand the human the solution, or at least a hint.
        """
        obs = self.observation
        return LaneState(
            lane=self.name,  # type: ignore[arg-type]
            state=obs.state if obs else "NOT_PLAYED",
            status=self.status,
            attempts_used=self.attempts_used,
            max_attempts=obs.max_attempts if obs else 0,
            elapsed_ms=self.finished_at_ms if self.finished_at_ms is not None else elapsed_ms,
            finished=self.finished,
            completed=self.completed,
            completion_ms=self.completion_ms,
            last_correct=self.attempts[-1]["correct"] if self.attempts else None,
            ai_calls=self.ai_calls,
            rule=self.rule if reveal else None,
            error=self.error,
            attempts=[
                AttemptInfo(
                    number=a["number"],
                    correct=a["correct"],
                    t_ms=a["t_ms"],
                    grid=a["grid"] if reveal else None,
                    rule=a.get("rule") if reveal else None,
                )
                for a in self.attempts
            ],
        )


# ---------------------------------------------------------------------------
# Race session
# ---------------------------------------------------------------------------
@dataclass
class RaceSession:
    race_id: str
    task: ArcTask
    human: Lane
    ai: Lane
    let_both_finish: bool = True
    ai_enabled: bool = False
    max_attempts: int = 3
    time_limit_seconds: int = 300
    ai_answers_live: bool = True
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    stopped: bool = False
    finished: bool = False
    timed_out: bool = False
    race_winner: str | None = None
    winner_reason: str | None = None
    ai_task: asyncio.Task[None] | None = None
    deadline_task: asyncio.Task[None] | None = None
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    event_log: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=MAX_EVENT_LOG))

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def revealed(self) -> bool:
        """May the AI's grids and rule be shown in the browser yet?

        Immediately in live mode (the default, so the AI's play is watchable).
        Otherwise only once the human can no longer use them: their run has
        ended, or the race has.
        """
        return self.ai_answers_live or self.finished or self.stopped or self.human.finished

    # -- timing ---------------------------------------------------------
    @property
    def started(self) -> bool:
        return self.started_at is not None

    def elapsed_ms(self) -> int:
        """Milliseconds on the shared clock, never past the deadline."""
        if self.started_at is None:
            return 0
        raw = int((time.monotonic() - self.started_at) * 1000)
        return min(raw, self.time_limit_ms)

    @property
    def time_limit_ms(self) -> int:
        return self.time_limit_seconds * 1000

    def remaining_ms(self) -> int:
        """Countdown value for the UI. An unstarted race reports the full budget."""
        if self.started_at is None:
            return self.time_limit_ms
        return max(0, self.time_limit_ms - self.elapsed_ms())

    @property
    def expired(self) -> bool:
        return self.started and self.remaining_ms() <= 0

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def lane(self, name: str) -> Lane:
        if name == "human":
            return self.human
        if name == "ai":
            return self.ai
        raise KeyError(name)

    # -- events ---------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=EVENT_QUEUE_SIZE)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with contextlib.suppress(ValueError):
            self.subscribers.remove(queue)

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Broadcast an event to every SSE subscriber.

        Slow consumers are dropped rather than allowed to apply back-pressure to
        the AI loop or a human submission.
        """
        event = {"type": event_type, "t": self.elapsed_ms(), "data": data or {}}
        self.event_log.append(event)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Dropping event for a slow subscriber on %s", self.race_id)

    # -- scoring --------------------------------------------------------
    def decide_winner(self) -> tuple[str | None, str]:
        """Who won, and why. The rule is **least time to a correct answer**."""
        human, ai = self.human, self.ai

        if human.completed and ai.completed:
            h, a = human.completion_ms or 0, ai.completion_ms or 0
            if h < a:
                return "human", "fastest correct answer"
            if a < h:
                return "ai", "fastest correct answer"
            return None, "dead heat"
        if human.completed:
            return "human", "only one to solve it"
        if ai.completed:
            return "ai", "only one to solve it"
        return None, "nobody solved the task"

    def efficiency_winner(self) -> str | None:
        """Fewest attempts among the lanes that solved the task."""
        human_won, ai_won = self.human.completed, self.ai.completed
        if not human_won and not ai_won:
            return None
        if human_won and not ai_won:
            return "human"
        if ai_won and not human_won:
            return "ai"
        if self.human.attempts_used < self.ai.attempts_used:
            return "human"
        if self.ai.attempts_used < self.human.attempts_used:
            return "ai"
        return "tie"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
@runtime_checkable
class SessionStore(Protocol):
    async def put(self, session: RaceSession) -> None: ...
    async def get(self, race_id: str) -> RaceSession | None: ...
    async def delete(self, race_id: str) -> RaceSession | None: ...
    async def all(self) -> list[RaceSession]: ...
    async def count(self) -> int: ...


class InMemorySessionStore:
    """Process-local session storage (MVP default)."""

    def __init__(self) -> None:
        self._sessions: dict[str, RaceSession] = {}
        self._lock = asyncio.Lock()

    async def put(self, session: RaceSession) -> None:
        async with self._lock:
            self._sessions[session.race_id] = session

    async def get(self, race_id: str) -> RaceSession | None:
        async with self._lock:
            return self._sessions.get(race_id)

    async def delete(self, race_id: str) -> RaceSession | None:
        async with self._lock:
            return self._sessions.pop(race_id, None)

    async def all(self) -> list[RaceSession]:
        async with self._lock:
            return list(self._sessions.values())

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class SessionManager:
    """Creates races, takes submissions, and reaps abandoned sessions."""

    def __init__(
        self,
        backend: Any,
        settings: Any,
        store: SessionStore | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings
        self.store = store or InMemorySessionStore()
        self._sweeper: asyncio.Task[None] | None = None
        # Test seam: supply a pre-built OpenRouter client (e.g. one wired to a
        # mock transport). Production leaves this as None.
        self.agent_client_factory: Callable[[], Any] | None = None

    # -- lifecycle ------------------------------------------------------
    def start_sweeper(self) -> None:
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def shutdown(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        for session in await self.store.all():
            self._cancel_deadline(session)
            await self.stop_race(session, reason="server shutdown")
            await self.store.delete(session.race_id)

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)
                await self.expire_idle_sessions()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("Session sweeper iteration failed")

    async def expire_idle_sessions(self) -> int:
        """Drop sessions idle beyond the TTL. Returns the number reaped."""
        ttl_seconds = self.settings.session_ttl_minutes * 60
        now = time.monotonic()
        reaped = 0
        for session in await self.store.all():
            if now - session.last_activity >= ttl_seconds:
                logger.info("Expiring idle race %s", session.race_id)
                session.emit("error", {"message": "Session expired due to inactivity"})
                await self.stop_race(session, reason="expired")
                await self.store.delete(session.race_id)
                reaped += 1
        return reaped

    # -- creation -------------------------------------------------------
    async def create_race(self, task_id: str, let_both_finish: bool) -> RaceSession:
        if await self.store.count() >= self.settings.max_sessions:
            await self.expire_idle_sessions()
            if await self.store.count() >= self.settings.max_sessions:
                raise RaceError("The server is at capacity. Try again shortly.")

        task = self.backend.get(task_id)  # ArcUnavailableError if unknown
        max_attempts = self.settings.max_attempts

        # Same task, two completely separate attempt records.
        session = RaceSession(
            race_id=secrets.token_urlsafe(24),
            task=task,
            human=Lane("human", ArcTaskEnvironment(task, max_attempts)),
            ai=Lane("ai", ArcTaskEnvironment(task, max_attempts)),
            let_both_finish=let_both_finish,
            ai_enabled=self.settings.ai_enabled,
            max_attempts=max_attempts,
            time_limit_seconds=self.settings.race_time_limit_seconds,
            ai_answers_live=self.settings.ai_answers_live,
        )
        session.human.observation = session.human.env.reset()
        session.ai.observation = session.ai.env.reset()
        session.human.status = "ready"
        session.ai.status = "ready" if session.ai_enabled else "disabled"
        if not session.ai_enabled:
            session.ai.error = "AI disabled: no OpenRouter API key configured on the server"

        await self.store.put(session)
        logger.info(
            "Race %s created (task=%s ai=%s attempts=%d limit=%ds)",
            session.race_id,
            task_id,
            session.ai_enabled,
            max_attempts,
            session.time_limit_seconds,
        )
        return session

    # -- start / stop ---------------------------------------------------
    async def start_race(self, session: RaceSession) -> None:
        if session.started:
            return

        # One common monotonic timer for both lanes.
        session.started_at = time.monotonic()
        session.touch()
        session.human.status = "playing"
        session.emit(
            "race_started",
            {
                "task_id": session.task_id,
                "time_limit_seconds": session.time_limit_seconds,
                "max_attempts": session.max_attempts,
            },
        )

        # The buzzer runs independently of any player activity, so an idle race
        # still ends on time.
        session.deadline_task = asyncio.create_task(self._deadline_watchdog(session))

        if not session.ai_enabled:
            session.ai.status = "disabled"
            return

        running = sum(
            1 for s in await self.store.all() if s.ai_task is not None and not s.ai_task.done()
        )
        if running >= self.settings.max_concurrent_ai_races:
            session.ai.status = "queued"
            session.ai.error = "Too many AI races are already running. Try again shortly."
            session.emit("error", {"message": session.ai.error, "lane": "ai"})
            return

        # AI generation begins only here -- never on race creation.
        from .openrouter_agent import run_agent_loop

        session.ai.status = "thinking"
        agent_client = self.agent_client_factory() if self.agent_client_factory else None
        session.ai_task = asyncio.create_task(run_agent_loop(session, self, client=agent_client))

    async def stop_race(self, session: RaceSession, reason: str = "stopped") -> None:
        """Immediately prevent further model calls and submissions."""
        session.stopped = True
        session.touch()
        self._cancel_deadline(session)
        task = session.ai_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        session.ai_task = None
        for lane in (session.human, session.ai):
            if not lane.finished and lane.status not in ("disabled", "error"):
                lane.status = "stopped"
        session.emit("race_stopped", {"reason": reason})

    # -- deadline -------------------------------------------------------
    async def _deadline_watchdog(self, session: RaceSession) -> None:
        """End the race when the shared clock runs out."""
        try:
            await asyncio.sleep(session.time_limit_seconds)
            if not session.finished and not session.stopped:
                await self.finish_on_timeout(session)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("Deadline watchdog failed for race %s", session.race_id)

    def _cancel_deadline(self, session: RaceSession) -> None:
        task = session.deadline_task
        if task is not None and not task.done() and asyncio.current_task() is not task:
            task.cancel()
        session.deadline_task = None

    async def finish_on_timeout(self, session: RaceSession) -> None:
        """The buzzer went. Freeze every lane that had not finished."""
        if session.finished:
            return
        session.timed_out = True
        for lane in (session.human, session.ai):
            if not lane.finished:
                lane.finished_at_ms = session.time_limit_ms
                if lane.status not in ("error", "disabled"):
                    lane.status = "timeout"
        session.emit(
            "time_up",
            {
                "time_limit_seconds": session.time_limit_seconds,
                "human_completed": session.human.completed,
                "ai_completed": session.ai.completed,
            },
        )
        await self.finish_race(session)

    async def finish_race(self, session: RaceSession) -> None:
        """Settle the race and broadcast the verdict exactly once."""
        if session.finished:
            return
        session.finished = True
        winner, reason = session.decide_winner()
        session.race_winner = winner
        session.winner_reason = reason

        self._cancel_deadline(session)
        # Cancel the AI loop only from *outside* it; self-cancelling would abort
        # the task's own cleanup.
        task = session.ai_task
        if task is not None and not task.done() and asyncio.current_task() is not task:
            task.cancel()

        session.emit(
            "race_finished",
            {
                "race_winner": winner,
                "winner_reason": reason,
                "efficiency_winner": session.efficiency_winner(),
                "timed_out": session.timed_out,
                "human_attempts": session.human.attempts_used,
                "ai_attempts": session.ai.attempts_used,
                "human_completion_ms": session.human.completion_ms,
                "ai_completion_ms": session.ai.completion_ms,
            },
        )

    # -- submissions ----------------------------------------------------
    def _check_human_can_act(self, session: RaceSession) -> None:
        if not session.started:
            raise ActionRejected("The race has not started yet")
        if session.stopped:
            raise ActionRejected("The race has been stopped")
        if session.finished:
            raise ActionRejected("The race is over")
        if session.expired:
            # The watchdog will settle it; reject the submission either way.
            raise ActionRejected("Time is up")
        if session.human.finished:
            raise ActionRejected("Your run has already finished")

    async def submit_human(self, session: RaceSession, grid: Grid) -> tuple[LaneState, bool]:
        self._check_human_can_act(session)
        lane = session.human
        observation = await self.submit(session, lane, grid)
        session.emit(
            "human_submit",
            {
                "attempt": observation.attempts_used,
                "correct": observation.is_win,
                "attempts_left": observation.attempts_left,
            },
        )
        await self.after_submit(session, lane, observation)
        return lane.to_state(session.elapsed_ms()), observation.is_win

    async def give_up(self, session: RaceSession) -> None:
        """End the human's run early; the AI keeps going and its work is revealed."""
        self._check_human_can_act(session)
        await self.finish_lane(session, session.human, status="gave_up")

    async def submit(
        self, session: RaceSession, lane: Lane, grid: Grid, rule: str | None = None
    ) -> Observation:
        """Check one grid against the task's hidden answer."""
        async with lane.lock:
            current = lane.observation
            if current is None:
                raise ActionRejected("The race has not been set up")
            if current.is_win:
                raise ActionRejected("This task is already solved")
            if current.is_game_over:
                raise ActionRejected("No attempts left")
            try:
                observation = lane.env.submit(grid)
            except ValueError as exc:
                raise ActionRejected(str(exc)) from exc

            lane.observation = observation
            lane.attempts.append(
                {
                    "number": observation.attempts_used,
                    "correct": observation.is_win,
                    "t_ms": session.elapsed_ms(),
                    "grid": [list(row) for row in grid],
                    "rule": rule,
                }
            )
        session.touch()
        return observation

    async def after_submit(
        self, session: RaceSession, lane: Lane, observation: Observation
    ) -> None:
        if observation.is_win:
            lane.completed = True
            lane.completion_ms = lane.attempts[-1]["t_ms"]
            await self.finish_lane(session, lane, status="solved")
        elif observation.is_game_over:
            await self.finish_lane(session, lane, status="out_of_attempts")

    async def finish_lane(self, session: RaceSession, lane: Lane, *, status: str) -> None:
        """Freeze a lane's clock and see whether the race is now settled."""
        if lane.finished:
            return
        lane.finished_at_ms = lane.completion_ms if lane.completed else session.elapsed_ms()
        lane.status = status
        other = session.ai if lane.name == "human" else session.human
        session.emit(
            "lane_finished",
            {
                "lane": lane.name,
                "reason": status,
                "completed": lane.completed,
                "completion_ms": lane.completion_ms,
                "finished_ms": lane.finished_at_ms,
                "attempts": lane.attempts_used,
                "first": lane.completed and not other.completed,
            },
        )
        await self.maybe_finish(session)

    async def maybe_finish(self, session: RaceSession) -> None:
        """Settle the race if there is nothing left to wait for.

        Results are shown when **both lanes have finished** or when the clock
        runs out (handled by the deadline watchdog). With "let both finish" off,
        the first correct answer ends it immediately.
        """
        if session.finished:
            return
        human, ai = session.human, session.ai
        ai_inactive = not session.ai_enabled or ai.status in (
            "disabled",
            "stopped",
            "error",
            "queued",
            "timeout",
        )
        if human.finished and ai.finished:
            pass
        elif not session.let_both_finish and (human.completed or ai.completed):
            pass
        elif human.finished and ai_inactive:
            pass
        else:
            return
        await self.finish_race(session)


__all__ = [
    "ActionRejected",
    "InMemorySessionStore",
    "Lane",
    "RaceError",
    "RaceSession",
    "SessionManager",
    "SessionStore",
]

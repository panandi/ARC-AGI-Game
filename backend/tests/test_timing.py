"""The five-minute race: deadline, buzzer, and the verdict.

These tests use very short limits so they run fast; the production default is
300 seconds.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.arc_service import ArcTaskEnvironment
from app.services.session_manager import ActionRejected, Lane, RaceSession, SessionManager
from tests.conftest import TASK_ID, make_settings, wrong_answer


def manager_with(backend, **overrides) -> SessionManager:
    return SessionManager(backend, make_settings(**overrides))


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------
async def test_countdown_starts_full(backend):
    manager = manager_with(backend, race_time_limit_seconds=60)
    session = await manager.create_race(TASK_ID, True)

    assert session.remaining_ms() == 60_000
    assert session.expired is False
    await manager.start_race(session)
    assert session.remaining_ms() <= 60_000
    await manager.stop_race(session)


async def test_elapsed_never_exceeds_the_limit(backend):
    manager = manager_with(backend)
    session = await manager.create_race(TASK_ID, True)
    session.time_limit_seconds = 1
    await manager.start_race(session)

    await asyncio.sleep(1.3)
    assert session.elapsed_ms() == session.time_limit_ms
    assert session.remaining_ms() == 0
    assert session.expired is True
    await manager.stop_race(session)


async def test_the_race_ends_when_the_clock_runs_out(backend):
    manager = manager_with(backend)
    session = await manager.create_race(TASK_ID, True)
    session.time_limit_seconds = 1
    await manager.start_race(session)

    await asyncio.sleep(1.4)

    assert session.timed_out is True
    assert session.finished is True
    assert session.human.status == "timeout"
    assert session.race_winner is None
    assert session.winner_reason == "nobody solved the task"
    types = {e["type"] for e in session.event_log}
    assert {"time_up", "race_finished"} <= types


async def test_submissions_are_refused_after_the_buzzer(backend):
    manager = manager_with(backend)
    session = await manager.create_race(TASK_ID, True)
    session.time_limit_seconds = 1
    await manager.start_race(session)
    await asyncio.sleep(1.4)

    with pytest.raises(ActionRejected):
        await manager.submit_human(session, wrong_answer())


async def test_stopping_a_race_cancels_the_deadline(backend):
    manager = manager_with(backend, race_time_limit_seconds=60)
    session = await manager.create_race(TASK_ID, True)
    await manager.start_race(session)
    assert session.deadline_task is not None

    await manager.stop_race(session)
    assert session.deadline_task is None


async def test_results_arrive_as_soon_as_both_are_done(backend):
    """No waiting for the clock once neither player has anything left to do."""
    manager = manager_with(backend, race_time_limit_seconds=60)
    session = await manager.create_race(TASK_ID, True)
    await manager.start_race(session)  # AI disabled: nothing to wait for

    for _ in range(3):
        await manager.submit_human(session, wrong_answer())

    assert session.finished is True
    assert session.timed_out is False


# ---------------------------------------------------------------------------
# The verdict: least time to a correct answer
# ---------------------------------------------------------------------------
def _session_with(backend, human_ms, ai_ms) -> RaceSession:
    task = backend.get(TASK_ID)

    def lane(name, completion):
        lane_obj = Lane(name, ArcTaskEnvironment(task, 3))
        if completion is not None:
            lane_obj.completed = True
            lane_obj.completion_ms = completion
            lane_obj.finished_at_ms = completion
        return lane_obj

    return RaceSession(race_id="r", task=task, human=lane("human", human_ms), ai=lane("ai", ai_ms))


def test_fastest_correct_answer_wins(backend):
    assert _session_with(backend, 12_000, 40_000).decide_winner() == (
        "human",
        "fastest correct answer",
    )
    assert _session_with(backend, 40_000, 12_000).decide_winner()[0] == "ai"


def test_identical_times_are_a_dead_heat(backend):
    assert _session_with(backend, 20_000, 20_000).decide_winner() == (None, "dead heat")


def test_a_single_solver_wins_regardless_of_time(backend):
    assert _session_with(backend, 290_000, None).decide_winner() == (
        "human",
        "only one to solve it",
    )
    assert _session_with(backend, None, 1_000).decide_winner()[0] == "ai"


def test_nobody_solving_it_has_no_winner(backend):
    assert _session_with(backend, None, None).decide_winner() == (
        None,
        "nobody solved the task",
    )

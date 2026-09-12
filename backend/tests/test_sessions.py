"""Session lifecycle: one task for both lanes, isolation, expiry, identifiers."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.services.arc_service import ArcUnavailableError
from app.services.session_manager import ActionRejected, Lane, RaceSession, SessionManager
from tests.conftest import TASK_ID, create_started_race, make_settings, wrong_answer


async def test_both_lanes_get_the_same_task(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    assert session.human.env.task is session.ai.env.task
    assert session.task_id == TASK_ID
    assert session.human.observation.max_attempts == session.ai.observation.max_attempts == 3


async def test_lanes_keep_separate_attempt_records(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    assert session.human.env is not session.ai.env


async def test_a_human_submission_does_not_touch_the_ai_lane(manager: SessionManager):
    session = await create_started_race(manager)
    await manager.submit_human(session, wrong_answer())

    assert session.human.attempts_used == 1
    assert session.ai.attempts_used == 0
    assert session.ai.observation.attempts_used == 0


async def test_unknown_task_is_rejected(manager: SessionManager):
    with pytest.raises(ArcUnavailableError):
        await manager.create_race("ffffffff", True)


async def test_submission_requires_a_started_race(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    with pytest.raises(ActionRejected, match="not started"):
        await manager.submit_human(session, wrong_answer())


async def test_stopped_race_rejects_submissions(manager: SessionManager):
    session = await create_started_race(manager)
    await manager.stop_race(session)
    with pytest.raises(ActionRejected, match="stopped"):
        await manager.submit_human(session, wrong_answer())


async def test_race_ids_are_random_and_unique(manager: SessionManager):
    ids = {(await manager.create_race(TASK_ID, True)).race_id for _ in range(6)}
    assert len(ids) == 6
    for race_id in ids:
        # secrets.token_urlsafe(24) -> 32 URL-safe characters.
        assert len(race_id) >= 32
        assert all(c.isalnum() or c in "-_" for c in race_id)


async def test_sessions_expire_after_ttl(backend):
    manager = SessionManager(backend, make_settings(session_ttl_minutes=1))
    session = await manager.create_race(TASK_ID, True)
    session.last_activity = time.monotonic() - 61

    assert await manager.expire_idle_sessions() == 1
    assert await manager.store.get(session.race_id) is None


async def test_active_sessions_are_not_expired(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    assert await manager.expire_idle_sessions() == 0
    assert await manager.store.get(session.race_id) is not None


async def test_capacity_limit_is_enforced(backend):
    manager = SessionManager(backend, make_settings(max_sessions=2))
    await manager.create_race(TASK_ID, True)
    await manager.create_race(TASK_ID, True)
    with pytest.raises(Exception, match="capacity"):
        await manager.create_race(TASK_ID, True)


def _wait_for_clock_tick(read, timeout=2.0):
    """Spin until a monotonic reading advances (Windows ticks every ~15.6 ms)."""
    start = time.monotonic()
    baseline = read()
    while read() <= baseline:
        if time.monotonic() - start > timeout:
            pytest.fail("the monotonic clock did not advance")
        time.sleep(0.005)
    return read()


async def test_timer_starts_only_on_start(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    assert session.started is False
    assert session.elapsed_ms() == 0

    await manager.start_race(session)
    assert session.started is True
    assert _wait_for_clock_tick(session.elapsed_ms) > 0


def _lane(backend, name: str, *, solved: bool, attempts: int) -> Lane:
    from app.services.arc_service import ArcTaskEnvironment

    lane = Lane(name, ArcTaskEnvironment(backend.get(TASK_ID), 3))
    lane.completed = solved
    lane.attempts = [
        {"number": n, "correct": False, "t_ms": 0, "grid": [[0]]} for n in range(attempts)
    ]
    return lane


def test_efficiency_winner_prefers_fewer_attempts(backend):
    task = backend.get(TASK_ID)
    session = RaceSession(
        race_id="r",
        task=task,
        human=_lane(backend, "human", solved=True, attempts=1),
        ai=_lane(backend, "ai", solved=True, attempts=3),
    )
    assert session.efficiency_winner() == "human"

    session.human.attempts = session.human.attempts * 3
    assert session.efficiency_winner() == "tie"


def test_efficiency_winner_ignores_lanes_that_did_not_solve(backend):
    session = RaceSession(
        race_id="r",
        task=backend.get(TASK_ID),
        human=_lane(backend, "human", solved=False, attempts=1),
        ai=_lane(backend, "ai", solved=True, attempts=3),
    )
    assert session.efficiency_winner() == "ai"


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
def test_starting_the_race_hands_out_the_puzzle(client):
    created = client.post("/api/races", json={"task_id": TASK_ID})
    assert created.status_code == 201

    status = client.post(f"/api/races/{created.json()['race_id']}/start").json()
    assert status["started"] is True
    assert status["task"]["task_id"] == TASK_ID
    assert len(status["task"]["train"]) == 4
    assert status["task"]["test_input"]


def test_unknown_task_is_404(client):
    assert client.post("/api/races", json={"task_id": "ffffffff"}).status_code == 404


def test_race_status_endpoint(client):
    race_id = client.post("/api/races", json={"task_id": TASK_ID}).json()["race_id"]
    client.post(f"/api/races/{race_id}/start")

    status = client.get(f"/api/races/{race_id}").json()
    assert status["race_id"] == race_id
    assert status["task_id"] == TASK_ID
    assert status["max_attempts"] == 3
    assert status["human"]["lane"] == "human"
    assert status["ai"]["lane"] == "ai"


def test_missing_race_returns_404(client):
    assert client.get("/api/races/does-not-exist").status_code == 404
    response = client.post("/api/races/does-not-exist/submit", json={"grid": [[0]]})
    assert response.status_code == 404


def test_task_list_is_sorted_smallest_first(client):
    body = client.get("/api/tasks").json()
    assert body["source"] == "arc-agi-1"
    difficulty = [t["difficulty"] for t in body["tasks"]]
    assert difficulty == sorted(difficulty)
    assert {t["task_id"] for t in body["tasks"]} == {"0d3d703e", "6150a2bd"}


def test_vendored_dataset_is_the_official_training_set():
    """The shipped data, not just the fixtures: 400 tasks, 386 single-test."""
    from app.core.config import DEFAULT_DATA_DIR
    from app.services.arc_service import TaskLibrary

    assert len(list(DEFAULT_DATA_DIR.glob("*.json"))) == 400
    assert TaskLibrary(DEFAULT_DATA_DIR).count == 386


async def test_event_stream_never_sends_an_event_twice(manager: SessionManager, settings):
    """Events emitted while a stream is opening must arrive exactly once.

    Regression: the backlog used to be copied inside the generator, after the
    subscription, so an event emitted in between was both replayed and queued.
    """
    from app.main import race_events

    session = await manager.create_race(TASK_ID, True)
    response = await race_events(session.race_id, manager, settings)
    await manager.start_race(session)  # emits race_started after subscribing

    stream = response.body_iterator
    first = await asyncio.wait_for(stream.__anext__(), 2)
    second = await asyncio.wait_for(stream.__anext__(), 2)
    assert b"event: snapshot" in first
    assert b"event: race_started" in second
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(stream.__anext__(), 0.3)
    await manager.stop_race(session)

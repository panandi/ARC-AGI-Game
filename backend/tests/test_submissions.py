"""Grid validation and the attempt rules."""

from __future__ import annotations

import pytest

from app.schemas.models import SubmitRequest, validate_grid
from app.services.session_manager import ActionRejected, SessionManager
from tests.conftest import TASK_ID, create_started_race, make_settings, solution, wrong_answer


# ---------------------------------------------------------------------------
# Grid validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "grid",
    [
        [[0]],
        [[9, 9], [0, 0]],
        [[5] * 30 for _ in range(30)],
    ],
)
def test_valid_grids_are_accepted(grid):
    assert validate_grid(grid) == grid


@pytest.mark.parametrize(
    "grid, message",
    [
        ([], "non-empty"),
        ([[]], "non-empty"),
        ([[1, 2], [3]], "same length"),
        ([[0]] * 31, "30 rows"),
        ([[0] * 31], "30 columns"),
        ([[10]], "0 to 9"),
        ([[-1]], "0 to 9"),
        ([[1.5]], "0 to 9"),
        ([["1"]], "0 to 9"),
        ([[True]], "0 to 9"),
        ([[None]], "0 to 9"),
        ("[[0]]", "non-empty list"),
    ],
)
def test_invalid_grids_are_rejected(grid, message):
    with pytest.raises(ValueError, match=message):
        validate_grid(grid)


def test_submit_request_forbids_extra_fields():
    with pytest.raises(ValueError):
        SubmitRequest(grid=[[0]], model="openai/gpt-5")


# ---------------------------------------------------------------------------
# Attempt rules
# ---------------------------------------------------------------------------
async def test_a_correct_answer_solves_the_task(manager: SessionManager):
    session = await create_started_race(manager)
    lane_state, correct = await manager.submit_human(session, solution())

    assert correct is True
    assert lane_state.completed is True
    assert lane_state.status == "solved"
    assert lane_state.attempts_used == 1
    assert session.human.completion_ms is not None
    assert session.human.finished


async def test_a_wrong_answer_spends_one_attempt(manager: SessionManager):
    session = await create_started_race(manager)
    lane_state, correct = await manager.submit_human(session, wrong_answer())

    assert correct is False
    assert lane_state.attempts_used == 1
    assert lane_state.finished is False
    assert lane_state.status == "playing"


async def test_the_wrong_size_is_simply_incorrect(manager: SessionManager):
    session = await create_started_race(manager)
    _, correct = await manager.submit_human(session, [[0]])
    assert correct is False


async def test_running_out_of_attempts_ends_the_run(manager: SessionManager):
    session = await create_started_race(manager)
    for _ in range(3):
        await manager.submit_human(session, wrong_answer())

    assert session.human.finished
    assert session.human.completed is False
    assert session.human.status == "out_of_attempts"
    with pytest.raises(ActionRejected):
        await manager.submit_human(session, solution())


async def test_a_solved_task_cannot_be_submitted_again(manager: SessionManager):
    session = await create_started_race(manager)
    await manager.submit_human(session, solution())
    with pytest.raises(ActionRejected):
        await manager.submit_human(session, solution())


async def test_giving_up_ends_the_run_and_reveals_the_ai(backend):
    # Only meaningful when AI answers are hidden until the human is done.
    manager = SessionManager(backend, make_settings(ai_answers_live=False))
    session = await create_started_race(manager)
    assert session.revealed is False

    await manager.give_up(session)

    assert session.human.finished
    assert session.human.completed is False
    assert session.human.status == "gave_up"
    assert session.revealed is True


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def _start(client) -> str:
    race_id = client.post("/api/races", json={"task_id": TASK_ID}).json()["race_id"]
    client.post(f"/api/races/{race_id}/start")
    return race_id


def test_http_correct_submission(client):
    race_id = _start(client)
    body = client.post(f"/api/races/{race_id}/submit", json={"grid": solution()}).json()
    assert body["correct"] is True
    assert body["lane"]["completed"] is True


def test_http_rejects_malformed_grids(client):
    race_id = _start(client)
    for payload in ({"grid": []}, {"grid": [[10]]}, {"grid": [[1, 2], [3]]}, {}):
        response = client.post(f"/api/races/{race_id}/submit", json=payload)
        assert response.status_code == 422, payload


def test_http_submission_before_start_is_refused(client):
    race_id = client.post("/api/races", json={"task_id": TASK_ID}).json()["race_id"]
    response = client.post(f"/api/races/{race_id}/submit", json={"grid": [[0]]})
    assert response.status_code == 409


def test_http_give_up(client):
    race_id = _start(client)
    status = client.post(f"/api/races/{race_id}/give-up").json()
    assert status["human"]["status"] == "gave_up"
    assert status["revealed"] is True

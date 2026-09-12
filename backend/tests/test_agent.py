"""The AI lane: structured output, attempts, retries and isolation.

Nothing here touches the real OpenRouter API -- every request is served by an
``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import pytest

from app.schemas.models import MAX_SIDE, RULE_MAX, AgentAnswer
from app.services.arc_service import format_grid
from app.services.openrouter_agent import (
    MAX_CONSECUTIVE_AGENT_FAILURES,
    SYSTEM_PROMPT,
    AgentError,
    build_context,
    redact,
)
from app.services.session_manager import SessionManager
from tests.conftest import (
    FAKE_OPENROUTER_KEY,
    TASK_ID,
    RecordingOpenRouter,
    answer,
    make_settings,
    solution,
    wrong_answer,
)


@pytest.fixture
def ai_settings():
    return make_settings(openrouter_api_key=FAKE_OPENROUTER_KEY, ai_turn_min_interval_seconds=0.0)


@pytest.fixture
def ai_manager(backend, ai_settings):
    return SessionManager(backend, ai_settings)


async def drive(manager, responder, *, let_both_finish=True, timeout=15.0):
    """Create and start a race, run the AI loop to completion, return artefacts."""
    session = await manager.create_race(TASK_ID, let_both_finish)
    recorder = RecordingOpenRouter(responder, settings=manager.settings)
    manager.agent_client_factory = recorder.client
    await manager.start_race(session)
    if session.ai_task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(session.ai_task), timeout)
    return session, recorder


def user_text(recorder, index):
    return recorder.requests[index]["payload"]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------
def test_answer_grid_is_validated_like_a_human_submission():
    with pytest.raises(ValueError):
        AgentAnswer(rule="r", grid=[[10]])
    with pytest.raises(ValueError):
        AgentAnswer(rule="r", grid=[[1, 2], [3]])


def test_answer_rule_is_clamped_not_rejected():
    parsed = AgentAnswer(rule="x" * 1000, grid=[[0]])
    assert len(parsed.rule) == RULE_MAX
    assert parsed.rule.endswith("…"), "a trimmed rule must look trimmed"
    assert AgentAnswer(rule=None, grid=[[0]]).rule == ""


def test_answer_ignores_extra_fields():
    assert AgentAnswer.model_validate({"rule": "r", "grid": [[1]], "chain": "..."}).grid == [[1]]


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------
async def test_a_correct_answer_solves_the_task(ai_manager):
    session, recorder = await drive(ai_manager, lambda i: answer(solution()))

    assert session.ai.completed is True
    assert session.ai.status == "solved"
    assert session.ai.attempts_used == 1
    assert recorder.call_count == 1
    assert session.ai.rule


async def test_wrong_answers_use_up_the_attempts(ai_manager):
    session, recorder = await drive(ai_manager, lambda i: answer(wrong_answer()))

    assert session.ai.completed is False
    assert session.ai.attempts_used == 3
    assert session.ai.status == "out_of_attempts"
    assert recorder.call_count == 3


async def test_a_later_attempt_is_told_what_it_already_tried(ai_manager):
    first = wrong_answer()

    def respond(i):
        return answer(first if i == 0 else solution())

    session, recorder = await drive(ai_manager, respond)

    assert session.ai.completed is True
    assert session.ai.attempts_used == 2
    second_prompt = user_text(recorder, 1)
    assert "marked incorrect" in second_prompt
    assert format_grid(first) in second_prompt
    assert "attempt 2 of 3" in second_prompt


async def test_a_malformed_reply_is_retried_without_spending_an_attempt(ai_manager):
    def respond(i):
        return "I think the answer is blue." if i == 0 else answer(solution())

    session, recorder = await drive(ai_manager, respond)

    assert recorder.call_count == 2
    assert session.ai.attempts_used == 1
    assert session.ai.completed is True
    assert "rejected" in user_text(recorder, 1).lower()


async def test_an_invalid_grid_from_the_model_is_never_submitted(ai_manager):
    session, recorder = await drive(ai_manager, lambda i: answer([[99]]))

    assert session.ai.attempts_used == 0
    assert session.ai.status == "error"
    assert recorder.call_count == MAX_CONSECUTIVE_AGENT_FAILURES * 2


async def test_persistent_malformed_output_retires_the_lane(ai_manager):
    session, recorder = await drive(ai_manager, lambda i: "not json at all")

    assert recorder.call_count == MAX_CONSECUTIVE_AGENT_FAILURES * 2
    assert session.ai.status == "error"
    assert "gave up" in (session.ai.error or "").lower()
    assert session.ai.finished


async def test_fatal_provider_errors_stop_immediately(ai_manager):
    def respond(_i):
        return httpx.Response(402, json={"error": {"message": "no credits"}})

    session, recorder = await drive(ai_manager, respond)

    assert recorder.call_count == 1, "a fatal error must not be retried"
    assert session.ai.status == "error"
    assert "out of credits" in (session.ai.error or "").lower()


async def test_provider_errors_are_redacted(ai_manager):
    def respond(_i):
        return httpx.Response(401, json={"error": {"message": f"bad key {FAKE_OPENROUTER_KEY}"}})

    session, _ = await drive(ai_manager, respond)

    assert FAKE_OPENROUTER_KEY not in (session.ai.error or "")
    assert "credential" in (session.ai.error or "").lower()


async def test_a_truncated_reply_is_reported_as_a_budget_problem(ai_settings):
    def respond(_i):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "length", "message": {"content": '{"grid": [[1,'}}]
            },
        )

    recorder = RecordingOpenRouter(respond, settings=ai_settings)
    client = recorder.client()
    context = {
        "prompt": "p",
        "attempt": 1,
        "max_attempts": 3,
        "remaining_seconds": 300,
        "previous_attempts": [],
    }
    with pytest.raises(AgentError, match="output-token budget"):
        await client.answer(context)
    await client._client.aclose()


async def test_stop_prevents_further_model_calls(backend):
    settings = make_settings(
        openrouter_api_key=FAKE_OPENROUTER_KEY, max_attempts=10, ai_turn_min_interval_seconds=0.05
    )
    manager = SessionManager(backend, settings)
    session = await manager.create_race(TASK_ID, True)
    recorder = RecordingOpenRouter(lambda i: answer(wrong_answer()), settings=settings)
    manager.agent_client_factory = recorder.client
    await manager.start_race(session)

    for _ in range(50):
        if recorder.call_count:
            break
        await asyncio.sleep(0.01)
    assert recorder.call_count > 0

    await manager.stop_race(session)
    calls_at_stop = recorder.call_count
    await asyncio.sleep(0.3)

    assert recorder.call_count == calls_at_stop
    assert session.ai_task is None


async def test_missing_key_disables_the_ai_lane_gracefully(manager: SessionManager):
    session = await manager.create_race(TASK_ID, True)
    await manager.start_race(session)

    assert session.ai_enabled is False
    assert session.ai_task is None
    assert session.ai.status == "disabled"
    # The human lane is unaffected.
    await manager.submit_human(session, wrong_answer())
    assert session.human.attempts_used == 1


async def test_concurrent_ai_race_limit(backend):
    # A long pacing interval keeps the first race's AI loop alive while the
    # second race starts.
    settings = make_settings(
        openrouter_api_key=FAKE_OPENROUTER_KEY,
        max_concurrent_ai_races=1,
        ai_turn_min_interval_seconds=5.0,
    )
    manager = SessionManager(backend, settings)
    recorder = RecordingOpenRouter(lambda i: answer(wrong_answer()), settings=settings)
    manager.agent_client_factory = recorder.client

    first = await manager.create_race(TASK_ID, True)
    await manager.start_race(first)
    second = await manager.create_race(TASK_ID, True)
    await manager.start_race(second)

    assert not first.ai_task.done()
    assert second.ai_task is None
    assert second.ai.status == "queued"
    await manager.stop_race(first)


# ---------------------------------------------------------------------------
# Racing
# ---------------------------------------------------------------------------
async def test_let_both_finish_keeps_the_human_playing(ai_manager):
    session, _ = await drive(ai_manager, lambda i: answer(solution()))

    assert session.ai.completed is True
    assert session.finished is False, "the race stays open for the human"
    await ai_manager.submit_human(session, wrong_answer())
    assert session.human.attempts_used == 1


async def test_without_let_both_finish_the_first_solve_ends_the_race(ai_manager):
    session, _ = await drive(ai_manager, lambda i: answer(solution()), let_both_finish=False)

    assert session.finished is True
    assert session.race_winner == "ai"
    assert "race_finished" in {e["type"] for e in session.event_log}


# ---------------------------------------------------------------------------
# Isolation and fairness
# ---------------------------------------------------------------------------
async def test_the_ai_never_sees_the_answer_or_the_human_lane(manager: SessionManager):
    # No key, so the AI loop never starts; build_context is checked directly.
    session = await manager.create_race(TASK_ID, True)
    await manager.start_race(session)
    human_grid = [[7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]]
    await manager.submit_human(session, human_grid)

    context = build_context(session)
    assert set(context) == {
        "prompt",
        "attempt",
        "max_attempts",
        "remaining_seconds",
        "previous_attempts",
    }
    assert context["previous_attempts"] == []
    assert format_grid(human_grid) not in context["prompt"]
    assert "test_output" not in session.task.public()
    await manager.stop_race(session)


async def test_ai_answers_can_be_hidden_until_the_human_is_done(backend):
    settings = make_settings(openrouter_api_key=FAKE_OPENROUTER_KEY, ai_answers_live=False)
    manager = SessionManager(backend, settings)
    session, _ = await drive(manager, lambda i: answer(solution(), rule="Swap colours."))

    hidden = session.ai.to_state(0, reveal=session.revealed)
    assert session.revealed is False
    assert hidden.rule is None
    assert all(a.grid is None for a in hidden.attempts)
    submit_events = [e for e in session.event_log if e["type"] == "ai_submit"]
    assert submit_events and "grid" not in submit_events[0]["data"]

    await manager.give_up(session)
    shown = session.ai.to_state(0, reveal=session.revealed)
    assert shown.rule == "Swap colours."
    assert shown.attempts[0].grid == solution()


async def test_payload_uses_only_server_configuration(ai_manager):
    _, recorder = await drive(ai_manager, lambda i: answer(solution()))

    request = recorder.requests[0]
    payload = request["payload"]
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["headers"]["authorization"] == f"Bearer {FAKE_OPENROUTER_KEY}"
    assert payload["model"] == ai_manager.settings.openrouter_model
    assert payload["max_tokens"] == ai_manager.settings.openrouter_max_output_tokens
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    # Text only: ARC-AGI-1 grids go to the model as rows of digits.
    assert isinstance(payload["messages"][1]["content"], str)
    assert payload["response_format"]["type"] == "json_schema"


def test_output_budget_fits_a_full_size_answer():
    """A maximum 30x30 answer must fit, or it is truncated into uselessness."""
    from app.core.config import Settings

    cells = MAX_SIDE * MAX_SIDE
    # Pretty-printed JSON costs ~4 tokens per cell (newline, indent, digit,
    # comma), plus row brackets and the rule.
    needed = cells * 4 + MAX_SIDE * 4 + RULE_MAX // 4 + 40
    assert Settings().openrouter_max_output_tokens >= needed


def test_system_prompt_asks_for_no_chain_of_thought():
    assert "Do not produce a long chain-of-thought" in SYSTEM_PROMPT
    assert "spectators" in SYSTEM_PROMPT


def test_redaction_removes_keys_and_headers():
    cleaned = redact(f"Authorization: Bearer {FAKE_OPENROUTER_KEY}", FAKE_OPENROUTER_KEY)
    assert FAKE_OPENROUTER_KEY not in cleaned


def test_prompt_is_plain_text_rows():
    from app.services.openrouter_agent import build_prompt
    from tests.conftest import task_json

    task = task_json()
    prompt = build_prompt(
        {"train": task["train"], "test_input": task["test"][0]["input"], "task_id": TASK_ID}
    )
    assert "Example 1" in prompt
    assert "Test input (3x3)" in prompt
    assert json.dumps(task["test"][0]["output"]) not in prompt


async def test_ai_answers_are_visible_live_by_default(ai_manager):
    """Each AI attempt -- grid and rule -- is shown the moment it lands."""
    first = wrong_answer()

    def respond(i):
        if i == 0:
            return answer(first, rule="Guess one.")
        return answer(solution(), rule="Swap colours.")

    session, _ = await drive(ai_manager, respond)

    assert session.human.finished is False, "the human is still playing"
    assert session.revealed is True
    state = session.ai.to_state(0, reveal=session.revealed)
    assert [a.grid for a in state.attempts] == [first, solution()]
    assert [a.rule for a in state.attempts] == ["Guess one.", "Swap colours."]
    submits = [e["data"] for e in session.event_log if e["type"] == "ai_submit"]
    assert submits[0]["grid"] == first
    assert submits[0]["rule"] == "Guess one."

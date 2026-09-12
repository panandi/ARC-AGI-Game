"""The AI lane: OpenRouter client and the agent loop.

Security posture
----------------
Every OpenRouter request originates here, on the server. The browser cannot
influence the model, the system prompt, the message list, the endpoint URL or
any sampling parameter -- all of those are constants or server configuration.
This module is deliberately *not* a proxy: there is no code path from an HTTP
request body to the payload built below.

Provider errors are redacted before they are attached to a session, so neither
the key nor the raw provider response can reach a client.

How the AI plays ARC-AGI-1
--------------------------
One model call per attempt. The model sees the task's examples and test input
as text grids -- never the answer -- and returns a one-line rule plus an
output grid. A wrong answer earns only "incorrect", exactly as for the human,
along with a reminder of what it already tried. At most ``max_attempts``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ValidationError

from ..core.config import OPENROUTER_CHAT_COMPLETIONS_URL
from ..core.rate_limit import MinIntervalLimiter
from ..schemas.models import AgentAnswer
from .arc_service import Observation, format_grid
from .session_manager import RaceError

if TYPE_CHECKING:  # pragma: no cover
    from .session_manager import Lane, RaceSession, SessionManager

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are solving an ARC-AGI-1 puzzle.

Each example shows an input grid and the output grid produced from it by one hidden rule. Grids are written as rows of digits; each digit from 0 to 9 is a colour.

Infer the rule from the examples, apply it to the test input, and produce the complete output grid. The output may be a different size from the input.

Reply with JSON only: "rule" is one short sentence describing the rule (at most 200 characters; it is shown to spectators) and "grid" is the output as a list of rows of integers. Keep the JSON compact: no indentation, and each row of the grid on a single line.

Do not produce a long chain-of-thought."""


# Structured-output contract. Mirrors :class:`AgentAnswer`.
RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "name": "arc_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rule": {"type": "string"},
            "grid": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        },
        "required": ["rule", "grid"],
    },
}


# A transient failure costs one try, not the lane. This many in a row means
# something is genuinely wrong.
MAX_CONSECUTIVE_AGENT_FAILURES = 3


class AgentError(RuntimeError):
    """An agent failure that is safe to show a spectator.

    ``fatal`` marks failures where retrying cannot help and would only burn
    credits -- rejected credentials, an empty account, a missing model.
    """

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def redact(text: str, *secrets_: str | None) -> str:
    """Strip anything key-shaped from a message before it leaves the server."""
    cleaned = text
    for secret in secrets_:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    cleaned = re.sub(r"sk-or-v1-[A-Za-z0-9_\-]+", "[redacted]", cleaned)
    cleaned = re.sub(r"(?i)(authorization|x-api-key)\s*[:=]\s*\S+", r"\1: [redacted]", cleaned)
    return cleaned


def _extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Handles bare JSON, fenced blocks, and prose with an object embedded.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise AgentError("The model did not return JSON") from exc
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AgentError("The model returned malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise AgentError("The model returned JSON that was not an object")
    return parsed


def build_prompt(public_task: dict[str, Any]) -> str:
    """The task as text.

    Takes the *public* task shape, which has no answer in it, so the answer
    cannot reach a prompt by construction.
    """
    parts = []
    for i, pair in enumerate(public_task["train"], 1):
        gi, go = pair["input"], pair["output"]
        parts.append(
            f"Example {i}\n"
            f"Input ({len(gi)}x{len(gi[0])}):\n{format_grid(gi)}\n"
            f"Output ({len(go)}x{len(go[0])}):\n{format_grid(go)}"
        )
    ti = public_task["test_input"]
    parts.append(f"Test input ({len(ti)}x{len(ti[0])}):\n{format_grid(ti)}")
    return "\n\n".join(parts)


class OpenRouterClient:
    """Minimal, single-purpose OpenRouter chat-completions client."""

    def __init__(self, settings: Any, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._owns_client = client is None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.openrouter_timeout_seconds)
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        # The only place the key is ever used. Never logged.
        return {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._settings.openrouter_referer,
            "X-Title": self._settings.openrouter_title,
        }

    def build_messages(
        self, context: dict[str, Any], correction: str | None = None
    ) -> list[dict[str, Any]]:
        """Assemble the message list from the whitelisted context only."""
        sections = [context["prompt"]]
        previous = context.get("previous_attempts") or []
        if previous:
            tried = "\n\n".join(
                f"Attempt {n}:\n{format_grid(grid)}" for n, grid in enumerate(previous, 1)
            )
            sections.append(
                f"Your earlier answers were marked incorrect:\n\n{tried}\n\n"
                "Reconsider the rule and give a different answer."
            )
        sections.append(
            f"This is attempt {context['attempt']} of {context['max_attempts']}. "
            f"{context['remaining_seconds']} seconds remain."
        )
        if correction:
            sections.append(
                f"Your previous reply was rejected: {correction} Reply again with valid JSON only."
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Model and limits come from server config only."""
        return {
            "model": self._settings.openrouter_model,
            "messages": messages,
            "max_tokens": self._settings.openrouter_max_output_tokens,
            "temperature": 0.2,
            "response_format": {"type": "json_schema", "json_schema": RESPONSE_JSON_SCHEMA},
        }

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = await self._http()
        try:
            response = await client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=self._headers(),
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise AgentError("The model provider timed out") from exc
        except httpx.HTTPError as exc:
            logger.warning("OpenRouter transport error: %s", type(exc).__name__)
            raise AgentError("Could not reach the model provider") from exc

        if response.status_code >= 400:
            # Log the status only; never the body or the headers.
            logger.warning("OpenRouter returned HTTP %s", response.status_code)
            raise _status_error(response.status_code)

        try:
            return response.json()
        except ValueError as exc:
            raise AgentError("The model provider returned an unreadable response") from exc

    def _content_from(self, body: dict[str, Any]) -> str:
        """Pull the assistant text out, without leaking the provider object."""
        choices = body.get("choices") or []
        if not choices:
            raise AgentError("The model returned no choices")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some providers return content parts.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        # Check truncation before emptiness: a reply cut off mid-JSON is
        # non-empty but unparseable, and reporting it as malformed JSON sends
        # the operator hunting the wrong problem.
        if choice.get("finish_reason") == "length":
            raise AgentError(
                "The model hit the output-token budget mid-reply. "
                "Raise OPENROUTER_MAX_OUTPUT_TOKENS."
            )
        if not content or not str(content).strip():
            raise AgentError("The model returned an empty reply")
        return str(content)

    async def answer(self, context: dict[str, Any], correction: str | None = None) -> AgentAnswer:
        """One request, one validated answer. Raises :class:`AgentError`."""
        body = await self._post(self.build_payload(self.build_messages(context, correction)))
        parsed = _extract_json(self._content_from(body))
        try:
            return AgentAnswer.model_validate(parsed)
        except ValidationError as exc:
            raise AgentError(_first_validation_message(exc)) from exc


def _status_error(status: int) -> AgentError:
    """Map a provider status to an error that is safe for a public page.

    Anything the operator must fix is fatal; anything that might succeed on the
    next try is not.
    """
    if status in (401, 403):
        return AgentError("The server's model credentials were rejected", fatal=True)
    if status == 402:
        return AgentError("The model account is out of credits", fatal=True)
    if status == 404:
        return AgentError("The configured model is unavailable", fatal=True)
    if status == 429:
        return AgentError("The model provider is rate limiting; slowing down")
    if 500 <= status < 600:
        return AgentError("The model provider had a server error")
    return AgentError("The model provider rejected the request")


def _first_validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "The model's reply did not match the required format"
    first = errors[0]
    location = ".".join(str(p) for p in first.get("loc", ())) or "response"
    return f"{location}: {first.get('msg', 'invalid')}"


def build_context(session: RaceSession) -> dict[str, Any]:
    """Whitelist of what the agent may know: the public task and its own lane."""
    lane = session.ai
    return {
        "prompt": build_prompt(session.task.public()),
        "attempt": lane.attempts_used + 1,
        "max_attempts": session.max_attempts,
        "remaining_seconds": max(0, session.remaining_ms() // 1000),
        "previous_attempts": [a["grid"] for a in lane.attempts],
    }


async def run_agent_loop(
    session: RaceSession,
    manager: SessionManager,
    client: OpenRouterClient | None = None,
) -> None:
    """Drive the AI lane: think -> answer -> validate -> submit -> broadcast.

    Exactly one OpenRouter request is outstanding at a time for a lane, because
    this coroutine awaits each answer before submitting it.
    """
    settings = manager.settings
    lane = session.ai
    owns_client = client is None
    client = client or OpenRouterClient(settings)
    pacer = MinIntervalLimiter(settings.ai_turn_min_interval_seconds)
    consecutive_failures = 0

    try:
        while True:
            if session.stopped or session.finished or session.expired or lane.finished:
                return
            await pacer.wait(session.race_id)
            if session.stopped or session.finished or session.expired:
                return

            context = build_context(session)
            lane.status = "thinking"
            session.emit("ai_thinking", {"attempt": context["attempt"]})

            try:
                reply = await _answer_with_one_retry(client, lane, context)
            except AgentError as exc:
                if exc.fatal:
                    raise
                # Transient: lose this try, keep the lane alive. No attempt is spent.
                consecutive_failures += 1
                message = redact(str(exc), settings.openrouter_api_key)
                if consecutive_failures >= MAX_CONSECUTIVE_AGENT_FAILURES:
                    raise AgentError(
                        f"Gave up after {consecutive_failures} failed tries: {message}"
                    ) from exc
                lane.error = f"Try skipped ({consecutive_failures}): {message}"
                session.emit(
                    "agent_retry",
                    {
                        "lane": "ai",
                        "message": message,
                        "consecutive": consecutive_failures,
                        "limit": MAX_CONSECUTIVE_AGENT_FAILURES,
                    },
                )
                continue

            # An answer that lands after the buzzer or a stop must not count.
            if session.stopped or session.finished or session.expired:
                return

            consecutive_failures = 0
            lane.error = None
            lane.rule = reply.rule or lane.rule
            observation = await manager.submit(session, lane, reply.grid, reply.rule)
            session.emit("ai_submit", _submit_event(session, observation, reply.grid, reply.rule))
            await manager.after_submit(session, lane, observation)
    except asyncio.CancelledError:
        if not lane.finished:
            lane.status = "stopped"
        raise
    except AgentError as exc:
        await _retire(session, manager, lane, redact(str(exc), settings.openrouter_api_key))
    except RaceError as exc:
        await _retire(session, manager, lane, str(exc))
    except Exception:  # pragma: no cover - defensive
        logger.exception("AI loop crashed for race %s", session.race_id)
        await _retire(session, manager, lane, "The AI lane stopped because of an internal error")
    finally:
        if owns_client:
            # A cancellation arriving mid-close must not mask the original outcome.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await client.aclose()


def _submit_event(
    session: RaceSession, observation: Observation, grid: list[list[int]], rule: str
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "attempt": observation.attempts_used,
        "correct": observation.is_win,
        "attempts_left": observation.attempts_left,
    }
    # Mid-race, an AI answer would hand the human the solution (or a hint).
    if session.revealed:
        event["grid"] = grid
        event["rule"] = rule
    return event


async def _retire(session: RaceSession, manager: SessionManager, lane: Lane, message: str) -> None:
    lane.error = message
    session.emit("error", {"lane": "ai", "message": message})
    await manager.finish_lane(session, lane, status="error")


async def _answer_with_one_retry(
    client: OpenRouterClient, lane: Lane, context: dict[str, Any]
) -> AgentAnswer:
    """Ask the model, retrying once with a validation message.

    A second failure surfaces an agent error rather than submitting a guess.
    """
    correction: str | None = None
    for attempt in (1, 2):
        lane.ai_calls += 1
        try:
            return await client.answer(context, correction)
        except AgentError as exc:
            # Retrying rejected credentials or an empty account cannot help and
            # would just spend again.
            if exc.fatal or attempt == 2:
                raise
            correction = str(exc)
            logger.info("Agent retry after: %s", correction)
    raise AgentError("The model did not produce an answer")  # pragma: no cover

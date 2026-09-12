"""Pydantic models for every request and response crossing the HTTP boundary.

Two rules hold throughout this module:

* Every client-supplied value is validated here before it reaches a race or
  the OpenRouter client.
* No response model has a field that could carry a secret or the hidden
  answer. The public task shape has no ``test_output``; there is no config
  echo and no provider-response shape.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SIDE = 30
NUM_COLOURS = 10
# The AI's one-line statement of the rule, shown to spectators.
RULE_MAX = 200

Grid = list[list[int]]
LaneName = Literal["human", "ai"]


def validate_grid(value: Any) -> Grid:
    """An ARC-AGI-1 grid: 1-30 rows by 1-30 columns, rectangular, cells 0-9."""
    if not isinstance(value, list) or not value:
        raise ValueError("A grid must be a non-empty list of rows")
    if len(value) > MAX_SIDE:
        raise ValueError(f"A grid has at most {MAX_SIDE} rows")
    width: int | None = None
    grid: Grid = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("Every row must be a non-empty list")
        if len(row) > MAX_SIDE:
            raise ValueError(f"A grid has at most {MAX_SIDE} columns")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("All rows must be the same length")
        for cell in row:
            # bool is an int subclass; True must not sneak in as colour 1.
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < NUM_COLOURS:
                raise ValueError("Every cell must be an integer colour from 0 to 9")
        grid.append(list(row))
    return grid


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class CreateRaceRequest(BaseModel):
    """Body for ``POST /api/races``."""

    model_config = ConfigDict(extra="forbid")

    # ARC-AGI-1 task ids are eight lowercase hex characters.
    task_id: Annotated[str, Field(pattern=r"^[0-9a-f]{8}$")]
    let_both_finish: bool = True


class SubmitRequest(BaseModel):
    """Body for ``POST /api/races/{id}/submit``.

    ``extra="forbid"`` matters: it is part of what stops this endpoint from
    being coaxed into carrying model parameters or prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    grid: Grid

    @field_validator("grid", mode="before")
    @classmethod
    def _grid(cls, value: Any) -> Grid:
        return validate_grid(value)


# ---------------------------------------------------------------------------
# Agent output (validated LLM structured output)
# ---------------------------------------------------------------------------
class AgentAnswer(BaseModel):
    """One attempt from the model, validated exactly as strictly as a human's."""

    model_config = ConfigDict(extra="ignore")

    rule: str = ""
    grid: Grid

    @field_validator("rule", mode="before")
    @classmethod
    def _clamp_rule(cls, value: Any) -> str:
        # A chatty model must not cost a turn over a cosmetic field.
        if value is None:
            return ""
        text = str(value).strip()
        # Say so when a rule is cut short, rather than stopping mid-word.
        return text if len(text) <= RULE_MAX else text[: RULE_MAX - 1].rstrip() + "…"

    @field_validator("grid", mode="before")
    @classmethod
    def _grid(cls, value: Any) -> Grid:
        return validate_grid(value)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class TaskInfo(BaseModel):
    task_id: str
    rows: int
    cols: int
    train_count: int
    difficulty: int


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    source: Literal["arc-agi-1"]


class GridPair(BaseModel):
    input: Grid
    output: Grid


class PublicTask(BaseModel):
    """What a player sees: the examples and the test input -- never the answer."""

    task_id: str
    train: list[GridPair]
    test_input: Grid


class AttemptInfo(BaseModel):
    number: int
    correct: bool
    t_ms: int
    # For the AI lane these are withheld unless answers are live or the
    # human's run is over.
    grid: Grid | None = None
    rule: str | None = None


class LaneState(BaseModel):
    lane: LaneName
    state: str = "NOT_PLAYED"
    status: str = "idle"
    attempts_used: int = 0
    max_attempts: int = 0
    elapsed_ms: int = 0
    finished: bool = False
    completed: bool = False
    completion_ms: int | None = None
    last_correct: bool | None = None
    ai_calls: int = 0
    rule: str | None = None
    error: str | None = None
    attempts: list[AttemptInfo] = Field(default_factory=list)


class RaceStatusResponse(BaseModel):
    race_id: str
    task_id: str
    started: bool
    finished: bool
    let_both_finish: bool
    ai_enabled: bool
    ai_model: str | None = None
    max_attempts: int
    elapsed_ms: int
    time_limit_seconds: int
    remaining_ms: int
    timed_out: bool = False
    # Whether the AI's grids and rule may be shown yet.
    revealed: bool = False
    human: LaneState
    ai: LaneState
    race_winner: LaneName | None = None
    winner_reason: str | None = None
    efficiency_winner: Literal["human", "ai", "tie"] | None = None
    # Present once the shared clock is running.
    task: PublicTask | None = None


class CreateRaceResponse(BaseModel):
    race_id: str
    task_id: str
    ai_enabled: bool
    ai_model: str | None = None
    max_attempts: int
    time_limit_seconds: int
    let_both_finish: bool
    human: LaneState
    ai: LaneState


class SubmitResponse(BaseModel):
    accepted: bool
    correct: bool
    lane: LaneState


class PublicConfigResponse(BaseModel):
    """The only configuration ever sent to a browser. No secrets by construction."""

    ai_enabled: bool
    ai_model: str | None = None
    max_attempts: int
    race_time_limit_seconds: int
    session_ttl_minutes: int
    dataset: str
    ai_answers_live: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    dataset: str
    ai_enabled: bool


class ErrorResponse(BaseModel):
    detail: str

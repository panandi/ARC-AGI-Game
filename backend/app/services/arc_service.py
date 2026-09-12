"""ARC-AGI-1 tasks: loading, the public/hidden split, and per-lane attempts.

ARC-AGI-1 is a static benchmark: each task is a handful of input -> output
examples plus a test input whose output must be reproduced exactly. The
official public training set (github.com/fchollet/ARC-AGI, Apache-2.0) is
vendored under ``backend/data/arc1``; nothing here reimplements or alters a
task.

Only single-test tasks are offered (386 of the 400), so "solving the task"
always means producing one grid.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas.models import validate_grid

logger = logging.getLogger(__name__)

Grid = list[list[int]]


class ArcUnavailableError(RuntimeError):
    """Raised when a requested task does not exist."""


def cells(grid: Grid) -> int:
    return len(grid) * len(grid[0])


def format_grid(grid: Grid) -> str:
    """A grid as text: one row per line, digits separated by spaces."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


@dataclass(frozen=True)
class TaskSummary:
    task_id: str
    rows: int
    cols: int
    train_count: int
    # Size of the test input. Deliberately not the answer's size: the task
    # list must not leak anything about the solution.
    difficulty: int


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[tuple[Grid, Grid], ...]
    test_input: Grid
    # The hidden answer. Server-side only: never serialised to a client and
    # never placed in a model prompt.
    test_output: Grid

    def public(self) -> dict[str, Any]:
        """Everything a player may see. The answer is deliberately absent."""
        return {
            "task_id": self.task_id,
            "train": [{"input": i, "output": o} for i, o in self.train],
            "test_input": self.test_input,
        }

    def summary(self) -> TaskSummary:
        return TaskSummary(
            task_id=self.task_id,
            rows=len(self.test_input),
            cols=len(self.test_input[0]),
            train_count=len(self.train),
            difficulty=cells(self.test_input),
        )


@dataclass(frozen=True)
class Observation:
    """One lane's standing on its task."""

    task_id: str
    state: str  # NOT_FINISHED | WIN | GAME_OVER (out of attempts)
    attempts_used: int
    max_attempts: int

    @property
    def is_win(self) -> bool:
        return self.state == "WIN"

    @property
    def is_game_over(self) -> bool:
        return self.state == "GAME_OVER"

    @property
    def is_terminal(self) -> bool:
        return self.is_win or self.is_game_over

    @property
    def attempts_left(self) -> int:
        return max(0, self.max_attempts - self.attempts_used)


class ArcTaskEnvironment:
    """One lane's attempts at one task. Knows the answer; never reveals it."""

    def __init__(self, task: ArcTask, max_attempts: int) -> None:
        self._task = task
        self._max = max_attempts
        self._observation = self.reset()

    @property
    def task(self) -> ArcTask:
        return self._task

    @property
    def observation(self) -> Observation:
        return self._observation

    def reset(self) -> Observation:
        self._observation = Observation(self._task.task_id, "NOT_FINISHED", 0, self._max)
        return self._observation

    def submit(self, grid: Grid) -> Observation:
        """Score one answer. Only exact equality counts, as in ARC-AGI-1."""
        current = self._observation
        if current.is_win:
            raise ValueError("This task is already solved")
        if current.is_game_over:
            raise ValueError("No attempts left")
        used = current.attempts_used + 1
        if grid == self._task.test_output:
            state = "WIN"
        elif used >= self._max:
            state = "GAME_OVER"
        else:
            state = "NOT_FINISHED"
        self._observation = Observation(self._task.task_id, state, used, self._max)
        return self._observation


def parse_task(task_id: str, raw: dict[str, Any]) -> ArcTask | None:
    """Build a task from the official JSON shape, or ``None`` if multi-test."""
    tests = raw.get("test") or []
    if len(tests) != 1:
        return None
    train = tuple(
        (validate_grid(pair["input"]), validate_grid(pair["output"])) for pair in raw["train"]
    )
    return ArcTask(
        task_id=task_id,
        train=train,
        test_input=validate_grid(tests[0]["input"]),
        test_output=validate_grid(tests[0]["output"]),
    )


class TaskLibrary:
    """The vendored ARC-AGI-1 tasks, loaded once and served read-only."""

    def __init__(self, data_dir: Path | str) -> None:
        self._tasks: dict[str, ArcTask] = {}
        for path in sorted(Path(data_dir).glob("*.json")):
            try:
                task = parse_task(path.stem, json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed task %s: %s", path.name, exc)
                continue
            if task is not None:
                self._tasks[task.task_id] = task
        # Smallest test input first: the gentlest puzzles lead the list.
        self._summaries = sorted(
            (t.summary() for t in self._tasks.values()),
            key=lambda s: (s.difficulty, s.task_id),
        )
        if not self._tasks:
            logger.warning("No ARC-AGI-1 tasks found in %s", data_dir)

    @property
    def count(self) -> int:
        return len(self._tasks)

    def list_tasks(self) -> list[TaskSummary]:
        return list(self._summaries)

    def get(self, task_id: str) -> ArcTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise ArcUnavailableError(f"Unknown ARC-AGI-1 task '{task_id}'")
        return task


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
_backend: TaskLibrary | None = None
_backend_lock = threading.Lock()


def get_backend(settings: Any) -> TaskLibrary:
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is None:
            _backend = TaskLibrary(settings.data_dir)
    return _backend


def set_backend(backend: TaskLibrary | None) -> None:
    """Test hook for injecting a task library."""
    global _backend
    _backend = backend

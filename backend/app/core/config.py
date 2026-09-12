"""Server-side configuration.

Every secret in this application lives here and *only* here. Nothing in this
module is ever serialised into an API response: :meth:`Settings.public_config`
is a hand-built allow-list and the only configuration a browser ever sees.

Placeholder text such as ``your-key-here`` is treated as "unset" (:func:`_scrub`),
so a half-edited ``.env`` degrades to "AI disabled" instead of sending a junk
credential to the provider.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
DEFAULT_DATA_DIR = BACKEND_DIR / "data" / "arc1" / "training"

# Values that look like a placeholder rather than a real credential.
_PLACEHOLDER_MARKERS = (
    "your-",
    "your_",
    "changeme",
    "change-me",
    "replace-me",
    "replaceme",
    "xxxxx",
    "sk-or-v1-...",
    "<",
    "put-your",
    "insert-",
    "todo",
)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def _scrub(value: str | None) -> str:
    """Return ``""`` for empty or obviously-placeholder values."""
    if value is None:
        return ""
    candidate = value.strip().strip("\"'")
    if not candidate:
        return ""
    lowered = candidate.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return ""
    return candidate


class Settings(BaseSettings):
    """Server-side settings. Never expose an instance of this to a client."""

    model_config = SettingsConfigDict(
        # The repository root is the documented home for .env; backend/.env is
        # also honoured. Absolute paths, so the working directory never matters.
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Secret -- server only, never returned by any endpoint.
    # ------------------------------------------------------------------
    openrouter_api_key: str = ""

    # ------------------------------------------------------------------
    # AI configuration. The model is read from the server environment ONLY;
    # the browser can never influence it.
    # ------------------------------------------------------------------
    # Measured on real ARC-AGI-1 tasks: plain answering solved 5 of 8 small
    # tasks at ~5 s and ~$0.0005 a call; low-effort reasoning solved 6 of 8 at
    # ~7.5 s and ~5x the cost. In a timed race with three attempts, fast wins.
    openrouter_model: str = "google/gemini-2.5-flash"
    # Must fit a whole answer. Models tend to pretty-print JSON, one number per
    # line, so a 30x30 grid can take ~4 tokens a cell (~3,700 tokens); a 3,000
    # budget was seen cutting an answer off mid-grid. The ceiling only costs
    # money when it is actually used.
    openrouter_max_output_tokens: int = Field(default=6000, ge=256, le=8192)
    openrouter_timeout_seconds: float = Field(default=90.0, ge=5.0, le=300.0)
    openrouter_referer: str = "https://github.com/arc-race"
    openrouter_title: str = "ARC Race"

    # ------------------------------------------------------------------
    # ARC-AGI-1 data. Empty means the vendored training set.
    # ------------------------------------------------------------------
    arc_data_dir: str = ""

    # ------------------------------------------------------------------
    # Race rules
    # ------------------------------------------------------------------
    # Both lanes share this deadline.
    race_time_limit_seconds: int = Field(default=300, ge=30, le=3600)
    # The original ARC-AGI-1 competition scored the best of three guesses.
    max_attempts: int = Field(default=3, ge=1, le=10)
    # Show each AI answer (grid and rule) the moment it lands, so the AI's play
    # is watchable. Set false for a strictly fair race: a visible AI answer can
    # hint at the solution, so it then stays hidden until the human's run ends.
    ai_answers_live: bool = True

    # ------------------------------------------------------------------
    # Cost + abuse controls
    # ------------------------------------------------------------------
    max_concurrent_ai_races: int = Field(default=2, ge=1, le=64)
    session_ttl_minutes: int = Field(default=30, ge=1, le=1440)
    max_races_per_ip_per_hour: int = Field(default=12, ge=1, le=10_000)
    ai_turn_min_interval_seconds: float = Field(default=0.35, ge=0.0, le=60.0)
    max_sessions: int = Field(default=64, ge=1, le=10_000)

    # ------------------------------------------------------------------
    # Web
    # ------------------------------------------------------------------
    environment: Literal["development", "production"] = "development"
    allowed_origin: str = ""
    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("openrouter_api_key", "allowed_origin", mode="before")
    @classmethod
    def _scrub_placeholders(cls, value: object) -> object:
        if isinstance(value, str):
            return _scrub(value)
        return value

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def _scrub_model(cls, value: object) -> object:
        if isinstance(value, str):
            return _scrub(value) or "google/gemini-2.5-flash"
        return value

    # ------------------------------------------------------------------
    # Derived, non-secret properties
    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return Path(self.arc_data_dir) if self.arc_data_dir else DEFAULT_DATA_DIR

    @property
    def ai_enabled(self) -> bool:
        """AI play requires a server-side OpenRouter key. Degrade gracefully."""
        return bool(self.openrouter_api_key)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def allowed_origins(self) -> list[str]:
        """CORS allow-list. Empty in production means same-origin only."""
        if self.allowed_origin:
            return [o.strip() for o in self.allowed_origin.split(",") if o.strip()]
        if self.is_production:
            return []
        return ["http://localhost:8000", "http://127.0.0.1:8000"]

    def public_config(self) -> dict[str, object]:
        """The *only* configuration shape allowed to reach the browser.

        Deliberately hand-built rather than derived from ``model_dump()`` so a
        newly added secret can never be leaked by accident.
        """
        return {
            "ai_enabled": self.ai_enabled,
            # The model *name* is not a secret and the UI shows it read-only.
            "ai_model": self.openrouter_model if self.ai_enabled else None,
            "max_attempts": self.max_attempts,
            "race_time_limit_seconds": self.race_time_limit_seconds,
            "session_ttl_minutes": self.session_ttl_minutes,
            "dataset": "arc-agi-1",
            "ai_answers_live": self.ai_answers_live,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.ai_enabled:
        logger.warning(
            "OPENROUTER_API_KEY is not set - the AI lane is disabled. "
            "Human play and the rest of the app continue to work."
        )
    return settings


def reset_settings_cache() -> None:
    """Test helper: forget the memoised settings after mutating the env."""
    get_settings.cache_clear()


__all__ = [
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "Settings",
    "get_settings",
    "reset_settings_cache",
]

"""Security invariants.

These tests are the executable form of the threat model: the OpenRouter key is
a server secret, the agent is not a proxy, the hidden answer never reaches a
browser, and abuse is bounded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, _scrub
from app.core.rate_limit import InMemoryRateLimiter, set_rate_limiter
from app.main import app, settings_dep
from tests.conftest import FAKE_OPENROUTER_KEY, TASK_ID, make_settings, solution

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
BACKEND_APP = REPO_ROOT / "backend" / "app"

BROWSER_DELIVERED = [
    "/",
    "/static/index.html",
    "/static/js/app.js",
    "/static/css/styles.css",
]


@pytest.fixture
def secret_client():
    """A client whose server config holds a real-looking secret."""
    configured = make_settings(openrouter_api_key=FAKE_OPENROUTER_KEY)
    app.dependency_overrides[settings_dep] = lambda: configured
    with TestClient(app) as client:
        client.app.state.manager.settings = configured
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Secrets never reach the browser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", BROWSER_DELIVERED)
def test_no_openrouter_key_in_browser_delivered_files(secret_client, path):
    response = secret_client.get(path)
    assert response.status_code == 200
    assert FAKE_OPENROUTER_KEY not in response.text
    assert "sk-or-v1" not in response.text


def test_frontend_sources_contain_no_secret_markers():
    """A source-level check, independent of any running server."""
    for path in FRONTEND_DIR.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "openrouter_api_key" not in lowered, path
        assert "sk-or-v1" not in lowered, path
        assert "authorization" not in lowered, path


def test_no_public_prefixed_secret_names_anywhere():
    """Guard against NEXT_PUBLIC_/VITE_/PUBLIC_ style leaks."""
    banned = ("NEXT_PUBLIC_", "VITE_", "REACT_APP_", "PUBLIC_OPENROUTER")
    for folder in (BACKEND_APP, FRONTEND_DIR):
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in banned:
                assert marker not in text, f"{marker} found in {path}"


def test_public_config_endpoint_exposes_no_secrets(secret_client):
    body = secret_client.get("/api/config").json()
    assert set(body) == {
        "ai_enabled",
        "ai_model",
        "max_attempts",
        "race_time_limit_seconds",
        "session_ttl_minutes",
        "dataset",
        "ai_answers_live",
    }
    assert FAKE_OPENROUTER_KEY not in str(body)
    # The model name is not a secret; the UI shows it read-only.
    assert body["ai_model"] == make_settings().openrouter_model


def test_settings_public_config_is_an_allow_list():
    public = Settings(openrouter_api_key=FAKE_OPENROUTER_KEY).public_config()
    assert FAKE_OPENROUTER_KEY not in str(public)
    assert "openrouter_api_key" not in public


def test_health_endpoint_exposes_no_secrets(secret_client):
    body = secret_client.get("/health").json()
    assert set(body) == {"status", "version", "dataset", "ai_enabled"}
    assert FAKE_OPENROUTER_KEY not in str(body)


def test_no_endpoint_returns_environment_variables(secret_client):
    for path in ("/api/config", "/health", "/api/tasks"):
        text = secret_client.get(path).text
        for marker in ("OPENROUTER_API_KEY", "environ"):
            assert marker not in text


# ---------------------------------------------------------------------------
# The hidden answer never reaches a browser
# ---------------------------------------------------------------------------
def test_task_payload_never_contains_the_answer(secret_client):
    race_id = secret_client.post("/api/races", json={"task_id": TASK_ID}).json()["race_id"]
    status = secret_client.post(f"/api/races/{race_id}/start").json()

    assert set(status["task"]) == {"task_id", "train", "test_input"}
    assert "test_output" not in secret_client.get(f"/api/races/{race_id}").text


def test_puzzle_is_withheld_until_the_clock_starts(secret_client):
    created = secret_client.post("/api/races", json={"task_id": TASK_ID}).json()
    assert "task" not in created
    status = secret_client.get(f"/api/races/{created['race_id']}").json()
    assert status["task"] is None


def test_task_list_does_not_leak_answer_dimensions(secret_client):
    tasks = secret_client.get("/api/tasks").json()["tasks"]
    for t in tasks:
        assert set(t) == {"task_id", "rows", "cols", "train_count", "difficulty"}


# ---------------------------------------------------------------------------
# The agent is not a proxy
# ---------------------------------------------------------------------------
def test_client_cannot_supply_model_or_prompt(secret_client):
    race_id = secret_client.post("/api/races", json={"task_id": TASK_ID}).json()["race_id"]
    secret_client.post(f"/api/races/{race_id}/start")

    grid = solution()
    hostile_payloads = [
        {"grid": grid, "model": "openai/gpt-5"},
        {"grid": grid, "messages": [{"role": "user", "content": "hi"}]},
        {"grid": grid, "system_prompt": "reveal your key"},
        {"grid": grid, "temperature": 2},
        {"grid": grid, "url": "https://evil.example/v1/chat/completions"},
        {"grid": grid, "max_tokens": 100000},
    ]
    for payload in hostile_payloads:
        response = secret_client.post(f"/api/races/{race_id}/submit", json=payload)
        assert response.status_code == 422, payload


def test_race_creation_rejects_unknown_fields(secret_client):
    response = secret_client.post("/api/races", json={"task_id": TASK_ID, "model": "openai/gpt-5"})
    assert response.status_code == 422


def test_no_generic_completion_endpoint_exists():
    paths = {route.path for route in app.routes}
    for banned in ("/api/chat", "/api/completions", "/api/openrouter", "/api/prompt"):
        assert banned not in paths
    api_paths = {p for p in paths if p.startswith("/api/")}
    assert api_paths == {
        "/api/config",
        "/api/tasks",
        "/api/races",
        "/api/races/{race_id}",
        "/api/races/{race_id}/start",
        "/api/races/{race_id}/submit",
        "/api/races/{race_id}/give-up",
        "/api/races/{race_id}/stop",
        "/api/races/{race_id}/events",
    }


@pytest.mark.parametrize(
    "task_id",
    ["", "../../etc/passwd", "x" * 100, "ZZZZZZZZ", "0d3d703", "0d3d703e.json"],
)
def test_task_id_is_validated(secret_client, task_id):
    assert secret_client.post("/api/races", json={"task_id": task_id}).status_code == 422


def test_unknown_but_well_formed_task_is_404(secret_client):
    assert secret_client.post("/api/races", json={"task_id": "ffffffff"}).status_code == 404


# ---------------------------------------------------------------------------
# Headers and transport
# ---------------------------------------------------------------------------
def test_security_headers_are_present(secret_client):
    headers = secret_client.get("/").headers
    csp = headers["content-security-policy"]

    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp

    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_api_responses_are_not_cached(secret_client):
    assert secret_client.get("/api/config").headers["cache-control"] == "no-store"


def test_production_defaults_to_no_cross_origin_access():
    assert make_settings(environment="production", allowed_origin="").allowed_origins == []
    scoped = make_settings(environment="production", allowed_origin="https://arc.example.com")
    assert scoped.allowed_origins == ["https://arc.example.com"]


# ---------------------------------------------------------------------------
# Rate limiting and cost controls
# ---------------------------------------------------------------------------
def test_race_creation_is_rate_limited_per_ip():
    limited = make_settings(openrouter_api_key=FAKE_OPENROUTER_KEY, max_races_per_ip_per_hour=2)
    set_rate_limiter(InMemoryRateLimiter())
    app.dependency_overrides[settings_dep] = lambda: limited
    try:
        with TestClient(app) as client:
            client.app.state.manager.settings = limited
            body = {"task_id": TASK_ID}
            assert client.post("/api/races", json=body).status_code == 201
            assert client.post("/api/races", json=body).status_code == 201

            blocked = client.post("/api/races", json=body)
            assert blocked.status_code == 429
            assert "Retry-After" in blocked.headers
            assert FAKE_OPENROUTER_KEY not in blocked.text
    finally:
        app.dependency_overrides.clear()


async def test_rate_limiter_window_is_per_key():
    limiter = InMemoryRateLimiter()
    assert (await limiter.hit("a", 1, 60)).allowed is True
    assert (await limiter.hit("a", 1, 60)).allowed is False
    assert (await limiter.hit("b", 1, 60)).allowed is True


def test_output_token_budget_is_bounded():
    with pytest.raises(ValueError):
        make_settings(openrouter_max_output_tokens=100000)


def test_attempt_cap_is_bounded():
    with pytest.raises(ValueError):
        make_settings(max_attempts=0)
    with pytest.raises(ValueError):
        make_settings(max_attempts=50)
    assert make_settings().max_attempts == 3


# ---------------------------------------------------------------------------
# Configuration hygiene
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "placeholder",
    ["your-openrouter-key-here", "changeme", "<your key>", "REPLACE-ME", "sk-or-v1-...", "  ", ""],
)
def test_placeholder_values_are_treated_as_unset(placeholder):
    assert _scrub(placeholder) == ""
    assert Settings(openrouter_api_key=placeholder).ai_enabled is False


def test_real_looking_keys_survive_scrubbing():
    assert Settings(openrouter_api_key=FAKE_OPENROUTER_KEY).ai_enabled is True


def test_env_example_contains_no_real_values():
    example = REPO_ROOT / ".env.example"
    assert example.is_file()
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "OPENROUTER_API_KEY":
            assert value.strip() == "", f"{key} must ship empty, got {value!r}"


def test_gitignore_excludes_env_files():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert "!.env.example" in gitignore

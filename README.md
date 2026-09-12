# ARC Race — Human vs AI

A single-page web app where a **human** and an **AI** race to solve the *same*
[ARC-AGI-1](https://github.com/fchollet/ARC-AGI) puzzle. Five minutes on a shared
clock, three attempts each. Whoever submits a correct answer in the least time
wins.

Neither player is told the rule. Each gets a few input → output examples and a
test input, and has to work out the rule and build the exact output grid.

The AI runs server-side through **OpenRouter**; the key never reaches the
browser.

---

## Contents

- [How a race works](#how-a-race-works)
- [Fairness](#fairness)
- [Project layout](#project-layout)
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [How the AI plays](#how-the-ai-plays)
- [Security model](#security-model)
- [Testing](#testing)
- [Deployment](#deployment)
- [Data and licence](#data-and-licence)
- [Limitations](#limitations)

---

## How a race works

**ARC-AGI-1** is François Chollet's Abstraction and Reasoning Corpus: static
puzzles on grids of up to 30 × 30 cells in 10 colours. A task shows 2–10
examples of an input grid and the output one hidden rule produces from it. You
must produce the output for a new test input — exactly, cell for cell.

1. Pick a task. The list is sorted smallest-first, so the gentlest puzzles lead.
2. **START RACE** reveals the puzzle to both players at once and starts one
   shared **5-minute** countdown.
3. Build your answer in the editor — pick a colour (keys `0`–`9`), paint cells
   (`P`) or flood-fill regions (`F`), resize, or start from **Copy input** — and
   press **SUBMIT ANSWER**.
4. Each player has **3 attempts**. A wrong answer only says "incorrect", exactly
   as ARC scoring does.
5. **Results appear** when both players are done (solved, out of attempts, or
   gave up) or when the clock runs out.

**Winner:** the fastest correct answer. If only one player solves it, they win;
if nobody does, there is no winner. Fewer attempts is reported separately.

**Let both finish** (on by default) keeps the race open for the other player
after the first solve. Turn it off and the first correct answer ends the race.

---

## Fairness

- **Same task, same moment.** The puzzle is only handed out by the start
  request, so neither side sees it before the clock runs.
- **The answer never leaves the server.** The browser receives the examples and
  the test input — no `test_output` field exists in any response. The task list
  doesn't even reveal the answer's size.
- **You can watch the AI play.** Each answer it submits appears in its panel the
  moment it lands — the grid, ✓ or ✗, and the rule it thinks it found — and every
  attempt stays on screen as a thumbnail. A live "Thinking about attempt 2 of 3…"
  timer shows while it works.
- **…or race it blind.** Watching is a spectator choice: a visible AI answer can
  hint at the solution, or hand it over. Set `AI_ANSWERS_LIVE=false` and the AI's
  grids and rule stay hidden until your run ends (solved, out of attempts, gave
  up, or time up). **GIVE UP** ends your run early.
- **The AI never sees your work.** Its prompt is built from the public task and
  its own previous attempts only.

---

## Project layout

```
.
├── backend/                     FastAPI service (Python)
│   ├── app/
│   │   ├── main.py              Routes, SSE, security headers, static hosting
│   │   ├── core/
│   │   │   ├── config.py        Settings; public_config() allow-list
│   │   │   └── rate_limit.py    Swappable rate limiter
│   │   ├── schemas/
│   │   │   └── models.py        Pydantic requests/responses, grid validation
│   │   └── services/
│   │       ├── arc_service.py   Task loading, public/hidden split, attempts
│   │       ├── session_manager.py  Races, lanes, timer, verdict, expiry
│   │       └── openrouter_agent.py OpenRouter client + the AI loop
│   ├── data/arc1/               Official ARC-AGI-1 training set + LICENSE
│   ├── scripts/fetch_arc1_data.py  Re-download the dataset
│   ├── tests/                   pytest suite (+ fixtures/arc1: real tasks)
│   ├── pyproject.toml           pytest + ruff config
│   ├── requirements.txt
│   └── requirements-dev.txt
├── frontend/                    Static client (no build step)
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── Dockerfile                   One image serving both
├── docker-compose.yml
├── .env.example
└── README.md
```

The backend serves `frontend/` at `/static`, so it still deploys as one
container.

---

## Quick start

### Docker (recommended)

```bash
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY
docker compose up --build
```

Open <http://localhost:8000>.

### Local Python

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
cp .env.example .env               # then set OPENROUTER_API_KEY

uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
```

`.env` lives in the repository root; `backend/.env` also works. With no
OpenRouter key the AI lane is disabled and labelled, and human play still works.

---

## Environment variables

All are read **by the server only**. Never prefix any of them with
`NEXT_PUBLIC_`, `VITE_`, `PUBLIC_` or similar.

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty)* | **Secret.** Enables the AI lane. |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash` | Model slug. The browser cannot set it. |
| `OPENROUTER_MAX_OUTPUT_TOKENS` | `6000` | Must fit a full 30 × 30 answer; pretty-printed JSON runs ~4 tokens a cell. |
| `RACE_TIME_LIMIT_SECONDS` | `300` | The shared countdown (30–3600). |
| `MAX_ATTEMPTS` | `3` | Submissions per player (1–10). |
| `AI_ANSWERS_LIVE` | `true` | Show the AI's answers as they land. `false` hides them until your run ends. |
| `ARC_DATA_DIR` | *(vendored set)* | Point at another folder of ARC task JSON files. |
| `ALLOWED_ORIGIN` | *(empty)* | CORS allow-list. Empty in production ⇒ same-origin only. |
| `ENVIRONMENT` | `development` | `production` hides `/docs` and tightens CORS. |
| `MAX_CONCURRENT_AI_RACES` | `2` | Simultaneous AI players server-wide. |
| `SESSION_TTL_MINUTES` | `30` | Idle races are reaped after this. |
| `MAX_RACES_PER_IP_PER_HOUR` | `12` | Race-creation rate limit. |
| `AI_TURN_MIN_INTERVAL_SECONDS` | `0.35` | Minimum spacing between AI attempts. |

---

## How the AI plays

One model call per attempt. The model gets the examples and test input as text
— rows of digits, which language models read far more reliably than a picture
of a small grid — and returns JSON:

```json
{ "rule": "Recolour each cell using a fixed colour mapping.", "grid": [[9,5,4], ...] }
```

The grid is validated exactly as strictly as a human's (rectangular, 1–30 per
side, colours 0–9). After a wrong answer the next prompt says so and shows what
it already tried; it never learns *why* it was wrong.

**Why no "thinking" mode.** Measured on eight small real tasks with
`gemini-2.5-flash`:

| Mode | Solved | Median time | Cost / attempt |
|---|---|---|---|
| **Plain answer** (used) | 5 / 8 | ~5 s | ~$0.0005 |
| Low-effort reasoning | 6 / 8 | ~7.5 s | ~$0.0026 |

In a timed race with three attempts, fast and cheap wins. A whole race costs
the AI at most three calls — well under a cent.

**When something goes wrong.** A malformed reply is retried once inline and
doesn't cost an attempt; three failed tries in a row retire the lane. A fatal
provider error (bad key, no credits, missing model) stops on the first
response rather than paying to retry. A reply cut off by the token budget is
reported as a budget problem, not as "malformed JSON".

---

## Security model

| Control | Where |
|---|---|
| Key lives only in server env; `.env` gitignored; `.env.example` ships empty | `core/config.py` |
| `public_config()` is a hand-built allow-list — no config echo endpoint | `core/config.py` |
| Hidden answer never serialised; puzzle withheld until the clock starts | `services/arc_service.py`, `main.py` |
| No prompt, model, or parameter can come from a request (`extra="forbid"`) | `schemas/models.py` |
| Provider errors redacted; only the HTTP status is logged | `services/openrouter_agent.py` |
| Grids validated (shape, size, colours) for humans *and* the model | `schemas/models.py` |
| Race creation rate-limited per IP; AI attempts paced | `core/rate_limit.py` |
| Strict CSP (no `unsafe-inline`/`unsafe-eval`), `nosniff`, `DENY` framing | `main.py` |
| Cryptographically random race IDs; idle races expire | `services/session_manager.py` |
| STOP cancels the AI immediately; AI starts only on START RACE | `services/session_manager.py` |

The rate limiter is a protocol: the in-memory one suits the single-worker MVP;
back it with Redis/Upstash in production via `set_rate_limiter()`.

---

## Testing

```bash
cd backend
pytest
ruff check app tests scripts
ruff format --check app tests scripts
```

Tests run against real ARC-AGI-1 tasks copied into `tests/fixtures/arc1` and a
mocked OpenRouter transport. An autouse fixture makes a real OpenRouter request
from a test fail outright. Coverage includes grid validation, the attempt rules,
the verdict, the 5-minute buzzer, answer secrecy, AI-answer withholding, retry
and fatal-error handling, rate limiting and the CSP.

---

## Deployment

One container, one worker — races live in process memory, so a second worker
would not see the first worker's races. Put the key in the platform's
**server-side** secrets UI, never in frontend configuration.

- **Railway / Render:** deploy the repo with its `Dockerfile`, health check
  `/health`, set `OPENROUTER_API_KEY` and `ENVIRONMENT=production`, one instance.
- **Fly.io:** `fly launch --no-deploy`, `fly secrets set OPENROUTER_API_KEY=…`,
  `fly deploy`, `fly scale count 1`.
- **Linux VM:** `docker compose up -d --build` behind nginx or Caddy for TLS.
  For SSE, set `proxy_buffering off`, and have the proxy **overwrite**
  `X-Forwarded-For` (the rate limiter reads its first hop).

> Hiding an API key requires a backend: this app cannot be deployed as a
> static-only site such as GitHub Pages.

---

## Data and licence

`backend/data/arc1/training` is the public training set of the
[ARC-AGI repository](https://github.com/fchollet/ARC-AGI) by François Chollet,
distributed under the Apache License 2.0 (copy in `backend/data/arc1/LICENSE`).
The 14 tasks with more than one test input are skipped, leaving 386. Refresh
the data with `python backend/scripts/fetch_arc1_data.py`.

---

## Limitations

- **Single worker / single instance**, as above.
- **Training set only.** The public evaluation set is harder and not offered.
- **Wall-clock fairness.** The AI answers in ~5 s; a human needs longer to paint
  a grid, so on tiny tasks the AI is usually faster. On larger tasks the human's
  pattern-spotting tends to matter more. The race is entertainment; attempts
  used is the more meaningful comparison.
- **No accounts.** Race IDs are unguessable bearer tokens.

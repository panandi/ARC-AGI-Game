/*
 * ARC Race — browser client (ARC-AGI-1).
 *
 * Talks only to this origin's own API. There is no model name, prompt, or
 * provider credential anywhere in this file: the AI lane is driven entirely by
 * the server, and this client only observes it.
 */
"use strict";

const MAX_SIDE = 30;
const COLOURS = 10;

const el = (id) => document.getElementById(id);

const dom = {
  taskSelect: el("task-select"),
  modelDisplay: el("model-display"),
  bothFinish: el("both-finish"),
  raceclock: el("raceclock"),
  countdown: el("countdown"),
  clockFill: el("clock-fill"),
  attemptsDisplay: el("attempts-display"),
  btnStart: el("btn-start"),
  btnStop: el("btn-stop"),
  btnNew: el("btn-new"),
  banner: el("banner"),
  tasksPill: el("tasks-pill"),
  legendNote: el("legend-note"),
  legendMain: el("legend-main"),
  taskVeil: el("task-veil"),
  taskIdLabel: el("task-id-label"),
  examples: el("examples"),
  testInput: el("test-input"),
  scoreboard: el("scoreboard"),
  raceResult: el("race-result"),
  verdictReason: el("verdict-reason"),
  efficiencyResult: el("efficiency-result"),
  resultRows: { human: el("result-human"), ai: el("result-ai") },
  log: el("log"),
  logCount: el("log-count"),
  editor: {
    rows: el("out-rows"),
    cols: el("out-cols"),
    resize: el("btn-resize"),
    copy: el("btn-copy"),
    clear: el("btn-clear"),
    paint: el("tool-paint"),
    fill: el("tool-fill"),
    palette: el("palette"),
    grid: el("editor-grid"),
    submit: el("btn-submit"),
    giveUp: el("btn-giveup"),
    result: el("human-result"),
  },
  human: {
    chip: el("human-chip"), attempts: el("human-attempts"),
    elapsed: el("human-elapsed"), state: el("human-state"), history: el("human-history"),
  },
  ai: {
    chip: el("ai-chip"), attempts: el("ai-attempts"),
    elapsed: el("ai-elapsed"), state: el("ai-state"), history: el("ai-history"),
    grid: el("ai-grid"), hidden: el("ai-hidden"), rule: el("ai-rule"),
    calls: el("ai-calls"), error: el("ai-error"),
    thinking: el("ai-thinking"), verdict: el("ai-verdict"), gallery: el("ai-gallery"),
  },
};

const state = {
  raceId: null,
  started: false,
  finished: false,
  revealed: false,
  startWall: null,
  deadlineWall: null,
  timeLimitMs: 300000,
  maxAttempts: 3,
  frozen: { human: null, ai: null },
  humanFinished: false,
  events: null,
  ticker: null,
  poller: null,
  sending: false,
  logCount: 0,
  task: null,
  out: blank(3, 3),
  colour: 1,
  tool: "paint",
  painting: false,
  aiLive: true,
  aiThinkingSince: null,
  aiThinkingAttempt: 0,
};

/* ----------------------------- utilities ------------------------------ */

function blank(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function clampSide(n) {
  const v = Number.parseInt(n, 10);
  return Number.isFinite(v) ? Math.min(MAX_SIDE, Math.max(1, v)) : 1;
}

function formatClock(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function showBanner(message, isError) {
  if (!message) {
    dom.banner.hidden = true;
    return;
  }
  dom.banner.textContent = message;
  dom.banner.classList.toggle("error", Boolean(isError));
  dom.banner.hidden = false;
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return body;
}

/* ------------------------------ grids --------------------------------- */

/** Pixels per cell so a grid fits `maxPx` without getting absurdly large. */
function cellSize(rows, cols, maxPx) {
  return Math.max(6, Math.min(30, Math.floor(maxPx / Math.max(rows, cols))));
}

/**
 * Draw an ARC grid as a CSS grid of cells. Colours come from the .c0–.c9
 * classes in styles.css, so nothing here needs inline styles.
 */
function renderGrid(host, grid, maxPx, editable) {
  host.textContent = "";
  if (!grid || !grid.length || !grid[0].length) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const g = document.createElement("div");
  g.className = editable ? "grid editable" : "grid";
  g.style.setProperty("--cols", String(cols));
  g.style.setProperty("--cell", `${cellSize(rows, cols, maxPx)}px`);
  g.setAttribute("role", "img");
  g.setAttribute("aria-label", `${rows} by ${cols} grid`);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = document.createElement("span");
      cell.className = `cell c${grid[r][c]}`;
      if (editable) {
        cell.dataset.r = String(r);
        cell.dataset.c = String(c);
      }
      g.appendChild(cell);
    }
  }
  host.appendChild(g);
}

function dims(grid) {
  return grid && grid.length ? `${grid.length}×${grid[0].length}` : "";
}

/* ------------------------------ the task ------------------------------ */

function renderTask(task) {
  state.task = task;
  dom.taskIdLabel.textContent = task.task_id;
  dom.examples.textContent = "";
  task.train.forEach((pair, i) => {
    const fig = document.createElement("figure");
    fig.className = "pair";
    const cap = document.createElement("figcaption");
    cap.textContent = `Example ${i + 1}`;
    const inp = document.createElement("div");
    const out = document.createElement("div");
    renderGrid(inp, pair.input, 132, false);
    renderGrid(out, pair.output, 132, false);
    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "→";
    const row = document.createElement("div");
    row.className = "pair-row";
    row.append(inp, arrow, out);
    fig.append(cap, row);
    dom.examples.appendChild(fig);
  });
  renderGrid(dom.testInput, task.test_input, 220, false);
  dom.taskVeil.hidden = true;
}

function clearTask() {
  state.task = null;
  dom.taskIdLabel.textContent = "";
  dom.examples.textContent = "";
  dom.testInput.textContent = "";
  dom.taskVeil.hidden = false;
}

/* ------------------------------ the editor ---------------------------- */

function canEdit() {
  return state.started && !state.finished && !state.humanFinished && remainingMs() > 0;
}

function renderEditor() {
  renderGrid(dom.editor.grid, state.out, 360, true);
  dom.editor.rows.value = String(state.out.length);
  dom.editor.cols.value = String(state.out[0].length);
}

function setOut(grid) {
  state.out = grid.map((row) => row.slice());
  renderEditor();
}

function resizeOut(rows, cols) {
  const next = blank(clampSide(rows), clampSide(cols));
  for (let r = 0; r < Math.min(next.length, state.out.length); r += 1) {
    for (let c = 0; c < Math.min(next[0].length, state.out[0].length); c += 1) {
      next[r][c] = state.out[r][c];
    }
  }
  setOut(next);
}

function floodFill(r, c, colour) {
  const target = state.out[r][c];
  if (target === colour) return;
  const rows = state.out.length;
  const cols = state.out[0].length;
  const stack = [[r, c]];
  while (stack.length) {
    const [y, x] = stack.pop();
    if (y < 0 || x < 0 || y >= rows || x >= cols || state.out[y][x] !== target) continue;
    state.out[y][x] = colour;
    stack.push([y + 1, x], [y - 1, x], [y, x + 1], [y, x - 1]);
  }
  renderEditor();
}

function applyTool(cell) {
  const r = Number(cell.dataset.r);
  const c = Number(cell.dataset.c);
  if (state.tool === "fill") {
    floodFill(r, c, state.colour);
  } else {
    state.out[r][c] = state.colour;
    cell.className = `cell c${state.colour}`;
  }
}

function selectColour(n) {
  state.colour = n;
  dom.editor.palette.querySelectorAll(".swatch").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.dataset.colour) === n));
  });
}

function selectTool(tool) {
  state.tool = tool;
  dom.editor.paint.setAttribute("aria-pressed", String(tool === "paint"));
  dom.editor.fill.setAttribute("aria-pressed", String(tool === "fill"));
}

function buildPalette() {
  dom.editor.palette.textContent = "";
  for (let n = 0; n < COLOURS; n += 1) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `swatch c${n}`;
    b.dataset.colour = String(n);
    b.setAttribute("aria-label", `Colour ${n}`);
    b.setAttribute("aria-pressed", String(n === state.colour));
    b.textContent = String(n);
    b.addEventListener("click", () => selectColour(n));
    dom.editor.palette.appendChild(b);
  }
}

function updateControls() {
  const live = canEdit();
  dom.editor.submit.disabled = !live || state.sending;
  dom.editor.giveUp.disabled = !live;
  [dom.editor.resize, dom.editor.copy, dom.editor.clear].forEach((b) => { b.disabled = !live; });
  dom.editor.grid.classList.toggle("locked", !live);
}

/* ------------------------------ lanes --------------------------------- */

function setChip(lane, text, cls) {
  const chip = dom[lane].chip;
  chip.textContent = text;
  chip.className = `state-chip${cls ? " " + cls : ""}`;
}

function statusLabel(s) {
  if (s.completed) return "Solved";
  switch (s.status) {
    case "thinking": return "Thinking…";
    case "playing": return "Playing";
    case "ready": return "Ready";
    case "solved": return "Solved";
    case "out_of_attempts": return "Out of attempts";
    case "gave_up": return "Gave up";
    case "timeout": return "Out of time";
    case "stopped": return "Stopped";
    case "queued": return "Queued";
    case "disabled": return "Disabled";
    case "error": return "Error";
    default: return "Idle";
  }
}

function chipClass(s) {
  if (s.completed) return "won";
  if (s.status === "error") return "err";
  if (["out_of_attempts", "gave_up", "timeout"].includes(s.status)) return "over";
  if (s.status === "thinking") return "thinking";
  if (s.status === "playing") return "live";
  return "";
}

function renderHistory(host, attempts) {
  host.textContent = "";
  (attempts || []).forEach((a) => {
    const li = document.createElement("li");
    li.className = a.correct ? "ok" : "bad";
    li.textContent = `#${a.number} ${a.correct ? "✓" : "✗"} ${formatClock(a.t_ms)}`;
    host.appendChild(li);
  });
}

function showAiVerdict(number, correct) {
  dom.ai.verdict.textContent = `Attempt ${number}: ${correct ? "✓ correct" : "✗ incorrect"}`;
  dom.ai.verdict.className = `ai-verdict ${correct ? "ok" : "bad"}`;
}

/** Draw attention to the AI panel when a new answer lands. */
function flashAi() {
  const box = dom.ai.grid.parentElement;
  box.classList.remove("flash");
  void box.offsetWidth; // restart the animation
  box.classList.add("flash");
}

/** Every attempt the AI has made, oldest first, each with its rule. */
function renderAiGallery(attempts) {
  dom.ai.gallery.textContent = "";
  attempts.forEach((a) => {
    if (!a.grid) return;
    const fig = document.createElement("figure");
    fig.className = `attempt ${a.correct ? "ok" : "bad"}`;
    const holder = document.createElement("div");
    renderGrid(holder, a.grid, 110, false);
    const cap = document.createElement("figcaption");
    cap.textContent = `#${a.number} ${a.correct ? "✓" : "✗"} ${formatClock(a.t_ms)}`;
    fig.append(holder, cap);
    if (a.rule) {
      const rule = document.createElement("p");
      rule.className = "attempt-rule";
      rule.textContent = a.rule;
      fig.appendChild(rule);
    }
    dom.ai.gallery.appendChild(fig);
  });
  dom.ai.gallery.hidden = dom.ai.gallery.children.length === 0;
}

/** A live "thinking" line, so a few-second model call reads as play, not a freeze. */
function setAiThinking(attempt) {
  if (attempt === null) {
    state.aiThinkingSince = null;
    dom.ai.thinking.hidden = true;
    return;
  }
  if (state.aiThinkingSince === null || state.aiThinkingAttempt !== attempt) {
    state.aiThinkingSince = Date.now();
    state.aiThinkingAttempt = attempt;
  }
  dom.ai.thinking.hidden = false;
  renderAiThinking();
}

function renderAiThinking() {
  if (state.aiThinkingSince === null) return;
  const secs = Math.floor((Date.now() - state.aiThinkingSince) / 1000);
  dom.ai.thinking.textContent =
    `Thinking about attempt ${state.aiThinkingAttempt} of ${state.maxAttempts}… ${secs}s`;
}

function renderAiWork(s) {
  const attempts = s.attempts || [];
  const last = attempts.length ? attempts[attempts.length - 1] : null;
  if (last && last.grid) {
    renderGrid(dom.ai.grid, last.grid, 300, false);
    dom.ai.hidden.hidden = true;
    showAiVerdict(last.number, last.correct);
  } else {
    dom.ai.grid.textContent = "";
    dom.ai.verdict.textContent = "";
    dom.ai.hidden.hidden = false;
    dom.ai.hidden.textContent = last
      ? "Hidden until your run ends — so it can't give you a hint."
      : s.status === "thinking" ? "Studying the examples…" : "No attempt yet.";
  }
  if (s.rule) {
    dom.ai.rule.textContent = `"${s.rule}"`;
  } else {
    dom.ai.rule.textContent = last && !state.revealed ? "Hidden until your run ends." : "—";
  }
  renderAiGallery(attempts);
  // Recover the thinking line after a reconnect or a missed event.
  if (!s.finished && s.status === "thinking") {
    setAiThinking(s.attempts_used + 1);
  } else {
    setAiThinking(null);
  }
}

function applyLaneState(lane, s) {
  const view = dom[lane];
  view.attempts.textContent = `${s.attempts_used} / ${s.max_attempts || state.maxAttempts}`;
  view.state.textContent = statusLabel(s);
  setChip(lane, statusLabel(s).toUpperCase(), chipClass(s));
  renderHistory(view.history, s.attempts);

  if (s.finished) {
    state.frozen[lane] = s.elapsed_ms;
    view.elapsed.classList.add("frozen");
  }

  if (lane === "human") {
    state.humanFinished = s.finished;
    updateControls();
  } else {
    dom.ai.calls.textContent = String(s.ai_calls);
    if (s.error) {
      dom.ai.error.hidden = false;
      dom.ai.error.textContent = s.error;
    }
    renderAiWork(s);
  }
}

function applyStatus(status) {
  if (!status) return;
  state.finished = status.finished;
  state.revealed = Boolean(status.revealed);
  state.maxAttempts = status.max_attempts;

  if (typeof status.time_limit_seconds === "number") {
    state.timeLimitMs = status.time_limit_seconds * 1000;
  }
  if (status.started) {
    state.started = true;
    if (state.startWall === null) state.startWall = Date.now() - status.elapsed_ms;
    // Re-sync the countdown to the server on every status refresh.
    if (typeof status.remaining_ms === "number") {
      state.deadlineWall = Date.now() + status.remaining_ms;
    }
  }
  if (status.task && (!state.task || state.task.task_id !== status.task.task_id)) {
    renderTask(status.task);
  }

  applyLaneState("human", status.human);
  applyLaneState("ai", status.ai);
  renderClock();

  if (status.finished) finishRace(status);
}

async function refreshStatus() {
  if (!state.raceId) return;
  try {
    applyStatus(await api(`/api/races/${state.raceId}`));
  } catch (_) {
    /* transient; SSE remains the primary channel */
  }
}

/* ------------------------------ results ------------------------------- */

function outcomeLabel(s) {
  if (s.completed) return "Solved";
  if (s.status === "out_of_attempts") return "Out of attempts";
  if (s.status === "gave_up") return "Gave up";
  if (s.status === "timeout") return "Out of time";
  if (s.status === "error") return "Error";
  if (s.status === "disabled") return "Disabled";
  return "Did not finish";
}

function fillResultRow(lane, s, isWinner) {
  const row = dom.resultRows[lane];
  const outcome = row.querySelector(".r-outcome");
  outcome.textContent = outcomeLabel(s);
  outcome.className = `r-outcome ${s.completed ? "done" : "miss"}`;

  const time = row.querySelector(".r-time");
  time.textContent = s.completed && s.completion_ms !== null ? formatClock(s.completion_ms) : "—";
  time.className = `r-time${isWinner && s.completed ? " best" : ""}`;

  row.querySelector(".r-attempts").textContent = `${s.attempts_used} / ${s.max_attempts}`;
  row.classList.toggle("is-winner", Boolean(isWinner));
}

function finishRace(status) {
  state.finished = true;
  dom.scoreboard.hidden = false;

  const winner = status.race_winner;
  dom.raceResult.textContent = winner === "human" ? "HUMAN WON THE RACE"
    : winner === "ai" ? "AI WON THE RACE"
    : "NO WINNER";
  dom.raceResult.className = winner === "human" ? "human-won" : winner === "ai" ? "ai-won" : "";
  dom.verdictReason.textContent = status.timed_out
    ? `Time expired — ${status.winner_reason || ""}`
    : (status.winner_reason || "");

  if (status.human) fillResultRow("human", status.human, winner === "human");
  if (status.ai) fillResultRow("ai", status.ai, winner === "ai");
  dom.efficiencyResult.textContent = status.efficiency_winner
    ? status.efficiency_winner.toUpperCase()
    : "—";

  dom.btnStop.disabled = true;
  state.humanFinished = true;
  stopTicker();
  stopPolling();
  renderClock();
  updateControls();
  dom.scoreboard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* -------------------------------- timing ------------------------------ */

function remainingMs() {
  if (state.deadlineWall === null) return state.timeLimitMs;
  return Math.max(0, state.deadlineWall - Date.now());
}

function renderClock() {
  const remaining = remainingMs();
  dom.countdown.textContent = formatClock(remaining);

  const fraction = state.timeLimitMs > 0 ? remaining / state.timeLimitMs : 0;
  dom.clockFill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;

  const cls = state.finished || remaining <= 0
    ? "over"
    : remaining <= 20000
      ? "danger"
      : remaining <= 60000
        ? "warn"
        : "";
  dom.raceclock.className = `raceclock${cls ? " " + cls : ""}`;
}

function tick() {
  renderClock();
  renderAiThinking();
  if (state.startWall === null) return;
  // The shared clock never runs past the deadline.
  const elapsed = Math.min(Date.now() - state.startWall, state.timeLimitMs);
  ["human", "ai"].forEach((lane) => {
    const frozen = state.frozen[lane];
    dom[lane].elapsed.textContent = formatClock(frozen === null ? elapsed : frozen);
  });
  if (remainingMs() <= 0 && !state.finished) {
    // The server's watchdog is authoritative; stop local input immediately.
    state.humanFinished = true;
    updateControls();
  }
}

function startTicker() {
  stopTicker();
  state.ticker = setInterval(tick, 200);
}

function stopTicker() {
  if (state.ticker) clearInterval(state.ticker);
  state.ticker = null;
}

function startPolling() {
  stopPolling();
  state.poller = setInterval(() => {
    if (!state.finished) refreshStatus();
  }, 2500);
}

function stopPolling() {
  if (state.poller) clearInterval(state.poller);
  state.poller = null;
}

/* --------------------------------- log -------------------------------- */

function addLog(t, who, what, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = formatClock(t || 0);
  const w = document.createElement("span");
  w.className = `who ${who.toLowerCase()}`;
  w.textContent = who.toUpperCase();
  const x = document.createElement("span");
  x.className = "what";
  x.textContent = what;
  li.append(ts, w, x);
  dom.log.appendChild(li);
  while (dom.log.children.length > 300) dom.log.removeChild(dom.log.firstChild);
  dom.log.parentElement.scrollTop = dom.log.parentElement.scrollHeight;
  state.logCount += 1;
  dom.logCount.textContent = String(state.logCount);
}

/* ------------------------------- events ------------------------------- */

const REASONS = {
  solved: "solved it",
  out_of_attempts: "ran out of attempts",
  gave_up: "gave up",
  error: "stopped with an error",
};

function openEventStream(raceId) {
  closeEventStream();
  const source = new EventSource(`/api/races/${raceId}/events`);
  state.events = source;

  const on = (name, handler) => source.addEventListener(name, (e) => {
    let data = {};
    try { data = JSON.parse(e.data); } catch (_) { /* ignore */ }
    handler(data);
  });

  on("snapshot", (d) => applyStatus(d.status));

  on("race_started", (d) => {
    addLog(d.t, "sys", `race started · task ${d.task_id} · ${d.max_attempts} attempts each`, "hl");
  });

  on("human_submit", (d) => {
    addLog(d.t, "human", `attempt ${d.attempt}: ${d.correct ? "CORRECT" : "incorrect"}`,
      d.correct ? "hl" : "");
  });

  on("ai_thinking", (d) => {
    setChip("ai", "THINKING…", "thinking");
    dom.ai.state.textContent = "Thinking…";
    setAiThinking(d.attempt);
    addLog(d.t, "ai", `working on attempt ${d.attempt}`);
  });

  on("ai_submit", (d) => {
    const rule = d.rule ? ` — "${d.rule}"` : "";
    addLog(d.t, "ai", `attempt ${d.attempt}: ${d.correct ? "CORRECT" : "incorrect"}${rule}`,
      d.correct ? "hl" : "");
    setAiThinking(null);
    if (d.grid) {
      // Show the answer the moment it lands rather than on the next poll.
      renderGrid(dom.ai.grid, d.grid, 300, false);
      dom.ai.hidden.hidden = true;
      showAiVerdict(d.attempt, d.correct);
      if (d.rule) dom.ai.rule.textContent = `"${d.rule}"`;
      flashAi();
    }
    refreshStatus();
  });

  on("lane_finished", (d) => {
    if (d.lane === "ai") setAiThinking(null);
    state.frozen[d.lane] = d.finished_ms;
    dom[d.lane].elapsed.classList.add("frozen");
    const when = d.completed ? ` in ${formatClock(d.completion_ms)}` : "";
    addLog(d.t, d.lane, `${REASONS[d.reason] || d.reason}${when} · ${d.attempts} attempt(s)`, "hl");
    refreshStatus();
  });

  on("agent_retry", (d) => {
    // A skipped turn, not a dead lane: show it without the alarming red panel.
    addLog(d.t, "ai", `turn skipped (${d.consecutive}/${d.limit}): ${d.message}`, "err");
  });

  on("time_up", (d) => {
    state.humanFinished = true;
    updateControls();
    renderClock();
    addLog(d.t, "sys", "time is up", "hl");
  });

  on("race_finished", async (d) => {
    state.finished = true;
    addLog(d.t, "sys", `race finished · ${d.winner_reason || "settled"}`, "hl");
    closeEventStream();
    // Pull the authoritative final status so the results table is complete.
    try {
      applyStatus(await api(`/api/races/${state.raceId}`));
    } catch (_) {
      finishRace(d);
    }
  });

  on("race_stopped", (d) => {
    addLog(d.t, "sys", `stopped (${d.reason || "stopped"})`);
    refreshStatus();
  });

  on("error", (d) => {
    const who = d.lane || "sys";
    addLog(d.t, who, d.message || "error", "err");
    if (d.lane === "ai") {
      setAiThinking(null);
      dom.ai.error.hidden = false;
      dom.ai.error.textContent = d.message || "Agent error";
    } else {
      showBanner(d.message, true);
    }
  });

  source.onerror = () => {
    if (state.finished) closeEventStream();
  };
}

function closeEventStream() {
  if (state.events) {
    state.events.close();
    state.events = null;
  }
}

/* ------------------------------ submitting ---------------------------- */

async function submitAnswer() {
  if (!canEdit() || state.sending) return;
  state.sending = true;
  updateControls();
  try {
    const res = await api(`/api/races/${state.raceId}/submit`, {
      method: "POST",
      body: JSON.stringify({ grid: state.out }),
    });
    applyLaneState("human", res.lane);
    const left = res.lane.max_attempts - res.lane.attempts_used;
    dom.editor.result.textContent = res.correct
      ? "Correct — solved!"
      : left > 0
        ? `Not quite — ${left} attempt${left === 1 ? "" : "s"} left.`
        : "Incorrect — no attempts left.";
    dom.editor.result.className = `result ${res.correct ? "ok" : "bad"}`;
    showBanner(null);
  } catch (err) {
    showBanner(err.message, true);
  } finally {
    state.sending = false;
    updateControls();
  }
}

async function giveUp() {
  if (!canEdit()) return;
  const note = state.aiLive
    ? "The AI keeps going."
    : "The AI keeps going and its answers are revealed.";
  if (!window.confirm(`End your run? ${note}`)) return;
  try {
    applyStatus(await api(`/api/races/${state.raceId}/give-up`, { method: "POST" }));
  } catch (err) {
    showBanner(err.message, true);
  }
}

/* ----------------------------- race control --------------------------- */

function resetView() {
  closeEventStream();
  stopTicker();
  stopPolling();
  Object.assign(state, {
    raceId: null,
    started: false,
    finished: false,
    revealed: false,
    startWall: null,
    deadlineWall: null,
    frozen: { human: null, ai: null },
    humanFinished: false,
    logCount: 0,
  });

  clearTask();
  setOut(blank(3, 3));
  dom.editor.result.textContent = "";
  dom.log.textContent = "";
  dom.logCount.textContent = "0";
  dom.scoreboard.hidden = true;
  dom.verdictReason.textContent = "—";
  Object.values(dom.resultRows).forEach((row) => {
    row.classList.remove("is-winner");
    row.querySelectorAll("td").forEach((cell) => {
      cell.textContent = "—";
      cell.className = cell.className.split(" ")[0];
    });
  });
  dom.ai.error.hidden = true;
  dom.ai.grid.textContent = "";
  dom.ai.hidden.hidden = false;
  dom.ai.hidden.textContent = "No attempt yet.";
  dom.ai.rule.textContent = "—";
  dom.ai.calls.textContent = "0";
  dom.ai.verdict.textContent = "";
  dom.ai.gallery.textContent = "";
  dom.ai.gallery.hidden = true;
  setAiThinking(null);
  showBanner(null);

  ["human", "ai"].forEach((lane) => {
    const view = dom[lane];
    view.attempts.textContent = `0 / ${state.maxAttempts}`;
    view.elapsed.textContent = "00:00";
    view.elapsed.classList.remove("frozen");
    view.state.textContent = "Idle";
    view.history.textContent = "";
    setChip(lane, "IDLE", "");
  });

  dom.btnStart.disabled = false;
  dom.btnStop.disabled = true;
  renderClock();
  updateControls();
}

async function startRace() {
  const taskId = dom.taskSelect.value;
  if (!taskId) {
    showBanner("Pick a task first.", true);
    return;
  }

  resetView();
  dom.btnStart.disabled = true;

  try {
    const race = await api("/api/races", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId, let_both_finish: dom.bothFinish.checked }),
    });
    state.raceId = race.race_id;
    state.maxAttempts = race.max_attempts;
    state.timeLimitMs = (race.time_limit_seconds || 300) * 1000;
    if (!race.ai_enabled) {
      dom.ai.hidden.textContent = "AI disabled — no server key.";
    }

    openEventStream(race.race_id);
    // The puzzle arrives with the start response, at the same moment the
    // shared clock begins.
    const status = await api(`/api/races/${race.race_id}/start`, { method: "POST" });
    state.startWall = Date.now();
    state.deadlineWall = Date.now() + state.timeLimitMs;
    applyStatus(status);

    // Start the answer at the test input's size, all black: the usual ARC default.
    const t = status.task.test_input;
    setOut(blank(t.length, t[0].length));

    startTicker();
    startPolling();
    dom.btnStop.disabled = false;
    updateControls();
    showBanner(null);
  } catch (err) {
    showBanner(err.message, true);
    dom.btnStart.disabled = false;
  }
}

async function stopRace() {
  if (!state.raceId) return;
  dom.btnStop.disabled = true;
  try {
    applyStatus(await api(`/api/races/${state.raceId}/stop`, { method: "POST" }));
    addLog(state.startWall ? Date.now() - state.startWall : 0, "sys", "stopped by player");
  } catch (err) {
    showBanner(err.message, true);
  }
  stopTicker();
  stopPolling();
  state.started = false;
  updateControls();
  dom.btnStart.disabled = false;
}

/* ------------------------------- input -------------------------------- */

function onGridPointerDown(event) {
  const cell = event.target.closest(".cell");
  if (!cell || !canEdit()) return;
  event.preventDefault();
  state.painting = state.tool === "paint";
  applyTool(cell);
}

function onGridPointerOver(event) {
  if (!state.painting) return;
  const cell = event.target.closest(".cell");
  if (cell) applyTool(cell);
}

function onKeyDown(event) {
  const target = event.target;
  if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (/^[0-9]$/.test(event.key)) {
    selectColour(Number(event.key));
  } else if (event.key.toLowerCase() === "p") {
    selectTool("paint");
  } else if (event.key.toLowerCase() === "f") {
    selectTool("fill");
  }
}

/* -------------------------------- boot -------------------------------- */

async function boot() {
  buildPalette();
  resetView();

  try {
    const config = await api("/api/config");
    state.maxAttempts = config.max_attempts;
    state.timeLimitMs = (config.race_time_limit_seconds || 300) * 1000;
    dom.modelDisplay.textContent = config.ai_model || "AI disabled (no server key)";
    dom.attemptsDisplay.textContent = `${config.max_attempts} attempts each`;
    const limitMin = Math.round((config.race_time_limit_seconds || 300) / 60);
    dom.legendNote.textContent =
      `Each race is capped at ${limitMin} minute(s). Tasks are listed smallest first.`;
    state.aiLive = config.ai_answers_live !== false;
    dom.legendMain.textContent = state.aiLive
      ? "Both players get the same ARC-AGI-1 task at the same moment. You can watch every answer the AI submits, and the rule it thinks it found, as it plays."
      : "Both players get the same ARC-AGI-1 task at the same moment. The AI's answers and reasoning stay hidden until your run ends, so it can't give you a hint.";
    renderClock();
    resetView();
    if (!config.ai_enabled) {
      showBanner("The AI lane is disabled: no OpenRouter key is configured on the server. Human play still works.");
    }
  } catch (err) {
    showBanner(`Could not load configuration: ${err.message}`, true);
  }

  try {
    const { tasks } = await api("/api/tasks");
    dom.taskSelect.textContent = "";
    dom.tasksPill.textContent = `${tasks.length} tasks`;
    tasks.forEach((t) => {
      const option = document.createElement("option");
      option.value = t.task_id;
      option.textContent = `${t.task_id} · ${t.rows}×${t.cols} · ${t.train_count} examples`;
      dom.taskSelect.appendChild(option);
    });
  } catch (err) {
    dom.taskSelect.textContent = "";
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Could not load tasks";
    dom.taskSelect.appendChild(option);
    showBanner(`Could not load the ARC task list: ${err.message}`, true);
  }

  dom.btnStart.addEventListener("click", startRace);
  dom.btnStop.addEventListener("click", stopRace);
  dom.btnNew.addEventListener("click", async () => {
    if (state.raceId && !state.finished) {
      try { await api(`/api/races/${state.raceId}/stop`, { method: "POST" }); } catch (_) {}
    }
    resetView();
  });

  dom.editor.resize.addEventListener("click", () => resizeOut(dom.editor.rows.value, dom.editor.cols.value));
  dom.editor.copy.addEventListener("click", () => { if (state.task) setOut(state.task.test_input); });
  dom.editor.clear.addEventListener("click", () => setOut(blank(state.out.length, state.out[0].length)));
  dom.editor.paint.addEventListener("click", () => selectTool("paint"));
  dom.editor.fill.addEventListener("click", () => selectTool("fill"));
  dom.editor.submit.addEventListener("click", submitAnswer);
  dom.editor.giveUp.addEventListener("click", giveUp);
  dom.editor.grid.addEventListener("pointerdown", onGridPointerDown);
  dom.editor.grid.addEventListener("pointerover", onGridPointerOver);
  window.addEventListener("pointerup", () => { state.painting = false; });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("beforeunload", closeEventStream);
}

boot();

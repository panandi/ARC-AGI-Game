/*
 * ARC Race — static build for GitHub Pages.
 *
 * GitHub Pages serves files, it cannot run a server, so this build plays the
 * whole race in the browser. Answers are never shipped: each task carries a
 * SHA-256 of its answer, and a submission is graded by hashing it.
 *
 * The AI opponent needs a model, and there is nowhere safe to keep a key on a
 * static site, so there are three ways to play:
 *   solo    race the clock
 *   key     the visitor's own OpenRouter key, kept in their browser
 *   server  a deployed ARC Race backend (window.ARC_RACE_API), which keeps its
 *           own key server-side
 */
"use strict";

const MAX_SIDE = 30;
const COLOURS = 10;
const LIMIT_MS = 300000;
const MAX_ATTEMPTS = 3;
const MODEL = "google/gemini-2.5-flash";
const KEY_STORE = "arc-race-openrouter-key";

const el = (id) => document.getElementById(id);

const dom = {
  taskSelect: el("task-select"),
  opponent: el("opponent-select"),
  optServer: el("opt-server"),
  keyrow: el("keyrow"),
  apiKey: el("api-key"),
  saveKey: el("btn-save-key"),
  forgetKey: el("btn-forget-key"),
  modePill: el("mode-pill"),
  btnStart: el("btn-start"),
  btnStop: el("btn-stop"),
  btnNew: el("btn-new"),
  banner: el("banner"),
  raceclock: el("raceclock"),
  countdown: el("countdown"),
  clockFill: el("clock-fill"),
  attemptsDisplay: el("attempts-display"),
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
  legendNote: el("legend-note"),
  editor: {
    rows: el("out-rows"), cols: el("out-cols"),
    resize: el("btn-resize"), copy: el("btn-copy"), clear: el("btn-clear"),
    paint: el("tool-paint"), fill: el("tool-fill"),
    palette: el("palette"), grid: el("editor-grid"),
    submit: el("btn-submit"), giveUp: el("btn-giveup"), result: el("human-result"),
  },
  human: {
    chip: el("human-chip"), attempts: el("human-attempts"),
    elapsed: el("human-elapsed"), state: el("human-state"), history: el("human-history"),
  },
  ai: {
    chip: el("ai-chip"), attempts: el("ai-attempts"),
    elapsed: el("ai-elapsed"), state: el("ai-state"), history: el("ai-history"),
    grid: el("ai-grid"), hidden: el("ai-hidden"), rule: el("ai-rule"),
    verdict: el("ai-verdict"), thinking: el("ai-thinking"), gallery: el("ai-gallery"),
    model: el("ai-model"), error: el("ai-error"),
  },
};

const api = (window.ARC_RACE_API || "").replace(/\/$/, "");

const lane = () => ({
  attempts: [], status: "idle", finished: false, completed: false,
  completionMs: null, finishedMs: null, rule: null, error: null,
});

const state = {
  index: [],
  task: null,
  mode: "solo",
  started: false,
  finished: false,
  startWall: null,
  human: lane(),
  ai: lane(),
  out: blank(3, 3),
  colour: 1,
  tool: "paint",
  painting: false,
  ticker: null,
  logCount: 0,
  aiThinkingSince: null,
  aiThinkingAttempt: 0,
  aiAbort: false,
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
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function showBanner(message, isError) {
  if (!message) { dom.banner.hidden = true; return; }
  dom.banner.textContent = message;
  dom.banner.classList.toggle("error", Boolean(isError));
  dom.banner.hidden = false;
}

/** The grading checksum: must match the server's json.dumps(grid, separators). */
async function gridHash(grid) {
  const bytes = new TextEncoder().encode(JSON.stringify(grid));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0")).join("");
}

function validGrid(g) {
  if (!Array.isArray(g) || !g.length || g.length > MAX_SIDE) return false;
  const w = Array.isArray(g[0]) ? g[0].length : 0;
  if (!w || w > MAX_SIDE) return false;
  return g.every((row) => Array.isArray(row) && row.length === w &&
    row.every((c) => Number.isInteger(c) && c >= 0 && c < COLOURS));
}

/* ------------------------------ grids --------------------------------- */

function cellSize(rows, cols, maxPx) {
  return Math.max(6, Math.min(30, Math.floor(maxPx / Math.max(rows, cols))));
}

function renderGrid(host, grid, maxPx, editable) {
  host.textContent = "";
  if (!grid || !grid.length || !grid[0].length) return;
  const g = document.createElement("div");
  g.className = editable ? "grid editable" : "grid";
  g.style.setProperty("--cols", String(grid[0].length));
  g.style.setProperty("--cell", `${cellSize(grid.length, grid[0].length, maxPx)}px`);
  g.setAttribute("role", "img");
  g.setAttribute("aria-label", `${grid.length} by ${grid[0].length} grid`);
  grid.forEach((row, r) => row.forEach((v, c) => {
    const cell = document.createElement("span");
    cell.className = `cell c${v}`;
    if (editable) { cell.dataset.r = String(r); cell.dataset.c = String(c); }
    g.appendChild(cell);
  }));
  host.appendChild(g);
}

function renderTask(task) {
  dom.taskIdLabel.textContent = task.id;
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

/* ------------------------------ the editor ---------------------------- */

function canEdit() {
  return state.started && !state.finished && !state.human.finished && remainingMs() > 0;
}

function renderEditor() {
  renderGrid(dom.editor.grid, state.out, 360, true);
  dom.editor.rows.value = String(state.out.length);
  dom.editor.cols.value = String(state.out[0].length);
}

function setOut(grid) { state.out = grid.map((r) => r.slice()); renderEditor(); }

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
  const stack = [[r, c]];
  while (stack.length) {
    const [y, x] = stack.pop();
    if (y < 0 || x < 0 || y >= state.out.length || x >= state.out[0].length) continue;
    if (state.out[y][x] !== target) continue;
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
  dom.editor.submit.disabled = !live;
  dom.editor.giveUp.disabled = !live;
  [dom.editor.resize, dom.editor.copy, dom.editor.clear].forEach((b) => { b.disabled = !live; });
  dom.editor.grid.classList.toggle("locked", !live);
}

/* ------------------------------ lane views ---------------------------- */

function setChip(name, text, cls) {
  dom[name].chip.textContent = text;
  dom[name].chip.className = `state-chip${cls ? " " + cls : ""}`;
}

const LABELS = {
  idle: "Idle", ready: "Ready", playing: "Playing", thinking: "Thinking…",
  solved: "Solved", out_of_attempts: "Out of attempts", gave_up: "Gave up",
  timeout: "Out of time", stopped: "Stopped", error: "Error", disabled: "Not playing",
};

function chipClass(l) {
  if (l.completed) return "won";
  if (l.status === "error") return "err";
  if (["out_of_attempts", "gave_up", "timeout"].includes(l.status)) return "over";
  if (l.status === "thinking") return "thinking";
  if (l.status === "playing") return "live";
  return "";
}

function renderHistory(host, attempts) {
  host.textContent = "";
  attempts.forEach((a) => {
    const li = document.createElement("li");
    li.className = a.correct ? "ok" : "bad";
    li.textContent = `#${a.number} ${a.correct ? "✓" : "✗"} ${formatClock(a.t)}`;
    host.appendChild(li);
  });
}

function renderLane(name) {
  const l = state[name];
  const view = dom[name];
  view.attempts.textContent = `${l.attempts.length} / ${MAX_ATTEMPTS}`;
  view.state.textContent = LABELS[l.status] || "Idle";
  setChip(name, (LABELS[l.status] || "Idle").toUpperCase(), chipClass(l));
  renderHistory(view.history, l.attempts);
  if (l.finished) view.elapsed.classList.add("frozen");
  if (name === "human") updateControls();
}

function showAiVerdict(number, correct) {
  dom.ai.verdict.textContent = `Attempt ${number}: ${correct ? "✓ correct" : "✗ incorrect"}`;
  dom.ai.verdict.className = `ai-verdict ${correct ? "ok" : "bad"}`;
}

function renderAiGallery() {
  dom.ai.gallery.textContent = "";
  state.ai.attempts.forEach((a) => {
    if (!a.grid) return;
    const fig = document.createElement("figure");
    fig.className = `attempt ${a.correct ? "ok" : "bad"}`;
    const holder = document.createElement("div");
    renderGrid(holder, a.grid, 110, false);
    const cap = document.createElement("figcaption");
    cap.textContent = `#${a.number} ${a.correct ? "✓" : "✗"} ${formatClock(a.t)}`;
    fig.append(holder, cap);
    if (a.rule) {
      const p = document.createElement("p");
      p.className = "attempt-rule";
      p.textContent = a.rule;
      fig.appendChild(p);
    }
    dom.ai.gallery.appendChild(fig);
  });
  dom.ai.gallery.hidden = dom.ai.gallery.children.length === 0;
}

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
    `Thinking about attempt ${state.aiThinkingAttempt} of ${MAX_ATTEMPTS}… ${secs}s`;
}

/* -------------------------------- timing ------------------------------ */

function remainingMs() {
  if (state.startWall === null) return LIMIT_MS;
  return Math.max(0, state.startWall + LIMIT_MS - Date.now());
}

function elapsedMs() {
  if (state.startWall === null) return 0;
  return Math.min(Date.now() - state.startWall, LIMIT_MS);
}

function renderClock() {
  const remaining = remainingMs();
  dom.countdown.textContent = formatClock(remaining);
  dom.clockFill.style.width = `${Math.max(0, Math.min(1, remaining / LIMIT_MS)) * 100}%`;
  const cls = state.finished || remaining <= 0 ? "over"
    : remaining <= 20000 ? "danger" : remaining <= 60000 ? "warn" : "";
  dom.raceclock.className = `raceclock${cls ? " " + cls : ""}`;
}

function tick() {
  renderClock();
  renderAiThinking();
  const now = elapsedMs();
  ["human", "ai"].forEach((name) => {
    const l = state[name];
    dom[name].elapsed.textContent = formatClock(l.finished ? (l.finishedMs ?? now) : now);
  });
  if (remainingMs() <= 0 && !state.finished) timeUp();
}

/* --------------------------------- log -------------------------------- */

function addLog(who, what, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const ts = document.createElement("span");
  ts.className = "ts";
  ts.textContent = formatClock(elapsedMs());
  const w = document.createElement("span");
  w.className = `who ${who}`;
  w.textContent = who.toUpperCase();
  const x = document.createElement("span");
  x.className = "what";
  x.textContent = what;
  li.append(ts, w, x);
  dom.log.appendChild(li);
  dom.log.parentElement.scrollTop = dom.log.parentElement.scrollHeight;
  state.logCount += 1;
  dom.logCount.textContent = String(state.logCount);
}

/* ------------------------------ the race ------------------------------ */

function finishLane(name, status) {
  const l = state[name];
  if (l.finished) return;
  l.finished = true;
  l.finishedMs = l.completed ? l.completionMs : elapsedMs();
  l.status = status;
  renderLane(name);
  maybeFinish();
}

async function recordAttempt(name, grid, rule) {
  const l = state[name];
  const correct = (await gridHash(grid)) === state.task.answer_sha256;
  const attempt = { number: l.attempts.length + 1, correct, t: elapsedMs(), grid, rule: rule || null };
  l.attempts.push(attempt);
  if (rule) l.rule = rule;
  if (correct) { l.completed = true; l.completionMs = attempt.t; }
  return attempt;
}

function maybeFinish() {
  if (state.finished) return;
  const aiIdle = state.mode === "solo" ||
    ["disabled", "error", "stopped"].includes(state.ai.status);
  if (!(state.human.finished && (state.ai.finished || aiIdle))) return;
  finishRace();
}

function timeUp() {
  if (state.finished) return;
  state.aiAbort = true;
  ["human", "ai"].forEach((name) => {
    const l = state[name];
    if (!l.finished && !(name === "ai" && state.mode === "solo")) {
      l.finished = true;
      l.finishedMs = LIMIT_MS;
      if (!["error", "disabled"].includes(l.status)) l.status = "timeout";
      renderLane(name);
    }
  });
  addLog("sys", "time is up", "hl");
  finishRace();
}

function decideWinner() {
  const h = state.human, a = state.ai;
  if (state.mode === "solo") {
    return h.completed
      ? ["human", `solved it in ${formatClock(h.completionMs)}`]
      : [null, "not solved in time"];
  }
  if (h.completed && a.completed) {
    if (h.completionMs < a.completionMs) return ["human", "fastest correct answer"];
    if (a.completionMs < h.completionMs) return ["ai", "fastest correct answer"];
    return [null, "dead heat"];
  }
  if (h.completed) return ["human", "only one to solve it"];
  if (a.completed) return ["ai", "only one to solve it"];
  return [null, "nobody solved the task"];
}

function fillResultRow(name, l, isWinner) {
  const row = dom.resultRows[name];
  const outcome = row.querySelector(".r-outcome");
  outcome.textContent = l.completed ? "Solved" : (LABELS[l.status] || "Did not finish");
  outcome.className = `r-outcome ${l.completed ? "done" : "miss"}`;
  const time = row.querySelector(".r-time");
  time.textContent = l.completed ? formatClock(l.completionMs) : "—";
  time.className = `r-time${isWinner && l.completed ? " best" : ""}`;
  row.querySelector(".r-attempts").textContent = `${l.attempts.length} / ${MAX_ATTEMPTS}`;
  row.classList.toggle("is-winner", Boolean(isWinner));
}

function finishRace() {
  state.finished = true;
  state.aiAbort = true;
  setAiThinking(null);
  const [winner, reason] = decideWinner();
  dom.scoreboard.hidden = false;
  dom.raceResult.textContent = winner === "human"
    ? (state.mode === "solo" ? "SOLVED" : "HUMAN WON THE RACE")
    : winner === "ai" ? "AI WON THE RACE" : "NO WINNER";
  dom.raceResult.className = winner === "human" ? "human-won" : winner === "ai" ? "ai-won" : "";
  dom.verdictReason.textContent = reason;
  fillResultRow("human", state.human, winner === "human");
  fillResultRow("ai", state.ai, winner === "ai");

  let efficiency = "—";
  if (state.human.completed && state.ai.completed) {
    efficiency = state.human.attempts.length < state.ai.attempts.length ? "HUMAN"
      : state.ai.attempts.length < state.human.attempts.length ? "AI" : "TIE";
  } else if (state.human.completed) { efficiency = "HUMAN"; }
  else if (state.ai.completed) { efficiency = "AI"; }
  dom.efficiencyResult.textContent = efficiency;

  dom.btnStop.disabled = true;
  updateControls();
  renderClock();
  addLog("sys", `race finished · ${reason}`, "hl");
  dom.scoreboard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ------------------------------ submitting ---------------------------- */

async function submitAnswer() {
  if (!canEdit()) return;
  const attempt = await recordAttempt("human", state.out.map((r) => r.slice()), null);
  addLog("human", `attempt ${attempt.number}: ${attempt.correct ? "CORRECT" : "incorrect"}`,
    attempt.correct ? "hl" : "");
  const left = MAX_ATTEMPTS - state.human.attempts.length;
  dom.editor.result.textContent = attempt.correct ? "Correct — solved!"
    : left > 0 ? `Not quite — ${left} attempt${left === 1 ? "" : "s"} left.`
      : "Incorrect — no attempts left.";
  dom.editor.result.className = `result ${attempt.correct ? "ok" : "bad"}`;
  renderLane("human");
  if (attempt.correct) finishLane("human", "solved");
  else if (left === 0) finishLane("human", "out_of_attempts");
}

function giveUp() {
  if (!canEdit()) return;
  if (!window.confirm("End your run? The AI keeps going.")) return;
  finishLane("human", "gave_up");
}

/* -------------------------------- the AI ------------------------------ */

const SYSTEM_PROMPT = `You are solving an ARC-AGI-1 puzzle.

Each example shows an input grid and the output grid produced from it by one hidden rule. Grids are written as rows of digits; each digit from 0 to 9 is a colour.

Infer the rule from the examples, apply it to the test input, and produce the complete output grid. The output may be a different size from the input.

Reply with JSON only: "rule" is one short sentence describing the rule (at most 200 characters; it is shown to spectators) and "grid" is the output as a list of rows of integers. Keep the JSON compact: no indentation, and each row of the grid on a single line.

Do not produce a long chain-of-thought.`;

const textGrid = (g) => g.map((row) => row.join(" ")).join("\n");

function buildPrompt(task, previous) {
  const parts = task.train.map((p, i) =>
    `Example ${i + 1}\nInput (${p.input.length}x${p.input[0].length}):\n${textGrid(p.input)}\n` +
    `Output (${p.output.length}x${p.output[0].length}):\n${textGrid(p.output)}`);
  const ti = task.test_input;
  parts.push(`Test input (${ti.length}x${ti[0].length}):\n${textGrid(ti)}`);
  if (previous.length) {
    parts.push("Your earlier answers were marked incorrect:\n\n" +
      previous.map((g, n) => `Attempt ${n + 1}:\n${textGrid(g)}`).join("\n\n") +
      "\n\nReconsider the rule and give a different answer.");
  }
  parts.push(`This is attempt ${previous.length + 1} of ${MAX_ATTEMPTS}.`);
  return parts.join("\n\n");
}

function parseAnswer(content) {
  let text = String(content || "").trim();
  if (text.startsWith("```")) text = text.replace(/^```[a-zA-Z]*\s*/, "").replace(/\s*```$/, "");
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (_) {
    const m = text.match(/\{[\s\S]*\}/);
    if (!m) throw new Error("the model did not return JSON");
    parsed = JSON.parse(m[0]);
  }
  if (!validGrid(parsed.grid)) throw new Error("the model returned an invalid grid");
  return { grid: parsed.grid, rule: String(parsed.rule || "").slice(0, 200) };
}

async function askOpenRouter(key, prompt) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 6000,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "arc_answer",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              rule: { type: "string" },
              grid: { type: "array", items: { type: "array", items: { type: "integer" } } },
            },
            required: ["rule", "grid"],
          },
        },
      },
    }),
  });
  if (!response.ok) {
    const map = { 401: "your key was rejected", 402: "your OpenRouter account is out of credits", 429: "rate limited" };
    throw new Error(map[response.status] || `the provider returned HTTP ${response.status}`);
  }
  const body = await response.json();
  const choice = (body.choices || [])[0] || {};
  if (choice.finish_reason === "length") throw new Error("the reply was cut off");
  return parseAnswer((choice.message || {}).content);
}

async function runAiWithKey(key) {
  state.ai.status = "thinking";
  renderLane("ai");
  for (let n = 1; n <= MAX_ATTEMPTS; n += 1) {
    if (state.aiAbort || state.finished) return;
    setAiThinking(n);
    addLog("ai", `working on attempt ${n}`);
    let reply;
    try {
      reply = await askOpenRouter(key, buildPrompt(state.task, state.ai.attempts.map((a) => a.grid)));
    } catch (err) {
      state.ai.error = `AI stopped: ${err.message}`;
      state.ai.status = "error";
      dom.ai.error.hidden = false;
      dom.ai.error.textContent = state.ai.error;
      setAiThinking(null);
      addLog("ai", state.ai.error, "err");
      renderLane("ai");
      maybeFinish();
      return;
    }
    if (state.aiAbort || state.finished) return;

    const attempt = await recordAttempt("ai", reply.grid, reply.rule);
    setAiThinking(null);
    renderGrid(dom.ai.grid, reply.grid, 300, false);
    dom.ai.hidden.hidden = true;
    showAiVerdict(attempt.number, attempt.correct);
    if (reply.rule) dom.ai.rule.textContent = `"${reply.rule}"`;
    renderAiGallery();
    addLog("ai", `attempt ${attempt.number}: ${attempt.correct ? "CORRECT" : "incorrect"}` +
      (reply.rule ? ` — "${reply.rule}"` : ""), attempt.correct ? "hl" : "");
    renderLane("ai");
    if (attempt.correct) { finishLane("ai", "solved"); return; }
  }
  finishLane("ai", "out_of_attempts");
}

/** Mirror the AI lane of a race running on a deployed backend. */
async function runAiOnServer() {
  state.ai.status = "thinking";
  renderLane("ai");
  let raceId;
  try {
    const created = await fetch(`${api}/api/races`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: state.task.id }),
    }).then((r) => r.json());
    raceId = created.race_id;
    await fetch(`${api}/api/races/${raceId}/start`, { method: "POST" });
  } catch (err) {
    state.ai.error = `Could not reach the AI server: ${err.message}`;
    state.ai.status = "error";
    dom.ai.error.hidden = false;
    dom.ai.error.textContent = state.ai.error;
    renderLane("ai");
    maybeFinish();
    return;
  }

  while (!state.aiAbort && !state.finished) {
    await new Promise((r) => setTimeout(r, 1500));
    let status;
    try {
      status = await fetch(`${api}/api/races/${raceId}`).then((r) => r.json());
    } catch (_) { continue; }
    const seen = state.ai.attempts.length;
    (status.ai.attempts || []).slice(seen).forEach((a) => {
      state.ai.attempts.push({ number: a.number, correct: a.correct, t: a.t_ms, grid: a.grid, rule: a.rule });
      if (a.correct) { state.ai.completed = true; state.ai.completionMs = a.t_ms; }
      if (a.grid) {
        renderGrid(dom.ai.grid, a.grid, 300, false);
        dom.ai.hidden.hidden = true;
        showAiVerdict(a.number, a.correct);
      }
      if (a.rule) dom.ai.rule.textContent = `"${a.rule}"`;
      addLog("ai", `attempt ${a.number}: ${a.correct ? "CORRECT" : "incorrect"}`, a.correct ? "hl" : "");
    });
    if (status.ai.attempts && status.ai.attempts.length > seen) { renderAiGallery(); renderLane("ai"); }
    setAiThinking(status.ai.status === "thinking" ? state.ai.attempts.length + 1 : null);
    if (state.ai.completed) { finishLane("ai", "solved"); return; }
    if (["out_of_attempts", "error", "timeout", "stopped"].includes(status.ai.status)) {
      finishLane("ai", status.ai.status);
      return;
    }
  }
}

/* ----------------------------- race control --------------------------- */

function resetView() {
  state.started = false;
  state.finished = false;
  state.startWall = null;
  state.human = lane();
  state.ai = lane();
  state.task = null;
  state.logCount = 0;
  state.aiAbort = true;
  if (state.ticker) clearInterval(state.ticker);
  state.ticker = null;

  dom.taskVeil.hidden = false;
  dom.taskIdLabel.textContent = "";
  dom.examples.textContent = "";
  dom.testInput.textContent = "";
  setOut(blank(3, 3));
  dom.editor.result.textContent = "";
  dom.log.textContent = "";
  dom.logCount.textContent = "0";
  dom.scoreboard.hidden = true;
  dom.ai.error.hidden = true;
  dom.ai.grid.textContent = "";
  dom.ai.hidden.hidden = false;
  dom.ai.hidden.textContent = "No attempt yet.";
  dom.ai.rule.textContent = "—";
  dom.ai.verdict.textContent = "";
  dom.ai.gallery.textContent = "";
  dom.ai.gallery.hidden = true;
  setAiThinking(null);
  showBanner(null);
  ["human", "ai"].forEach((name) => {
    dom[name].elapsed.textContent = "00:00";
    dom[name].elapsed.classList.remove("frozen");
    dom[name].history.textContent = "";
    renderLane(name);
  });
  Object.values(dom.resultRows).forEach((row) => {
    row.classList.remove("is-winner");
    row.querySelectorAll("td").forEach((c) => { c.textContent = "—"; c.className = c.className.split(" ")[0]; });
  });
  dom.btnStart.disabled = false;
  dom.btnStop.disabled = true;
  renderClock();
  updateControls();
}

function savedKey() {
  try { return localStorage.getItem(KEY_STORE) || ""; } catch (_) { return ""; }
}

async function startRace() {
  const id = dom.taskSelect.value;
  if (!id) { showBanner("Pick a task first.", true); return; }
  const mode = dom.opponent.value;
  if (mode === "key" && !savedKey()) {
    showBanner("Save an OpenRouter key first, or choose Solo.", true);
    dom.keyrow.hidden = false;
    return;
  }

  resetView();
  dom.btnStart.disabled = true;
  try {
    state.task = await fetch(`./data/tasks/${id}.json`, { cache: "force-cache" }).then((r) => r.json());
  } catch (err) {
    showBanner(`Could not load that task: ${err.message}`, true);
    dom.btnStart.disabled = false;
    return;
  }

  state.mode = mode;
  state.aiAbort = false;
  renderTask(state.task);
  setOut(blank(state.task.test_input.length, state.task.test_input[0].length));
  state.started = true;
  state.startWall = Date.now();
  state.human.status = "playing";
  state.ai.status = mode === "solo" ? "disabled" : "thinking";
  renderLane("human");
  renderLane("ai");
  state.ticker = setInterval(tick, 200);
  dom.btnStop.disabled = false;
  updateControls();
  addLog("sys", `race started · task ${state.task.id} · ${MAX_ATTEMPTS} attempts each`, "hl");

  if (mode === "key") runAiWithKey(savedKey());
  else if (mode === "server") runAiOnServer();
}

function stopRace() {
  if (!state.started || state.finished) return;
  state.aiAbort = true;
  if (!state.human.finished) { state.human.status = "stopped"; state.human.finished = true; state.human.finishedMs = elapsedMs(); }
  if (!state.ai.finished && state.mode !== "solo") { state.ai.status = "stopped"; state.ai.finished = true; }
  renderLane("human");
  renderLane("ai");
  addLog("sys", "stopped");
  finishRace();
}

/* -------------------------------- boot -------------------------------- */

function applyMode() {
  const mode = dom.opponent.value;
  dom.keyrow.hidden = mode !== "key";
  dom.ai.model.textContent = mode === "solo" ? "—" : MODEL;
  dom.modePill.textContent = mode === "solo" ? "solo play"
    : mode === "key" ? "AI · your key" : "AI · hosted";
}

async function boot() {
  buildPalette();
  resetView();
  dom.attemptsDisplay.textContent = `${MAX_ATTEMPTS} attempts each`;
  dom.legendNote.textContent =
    "Static build: the whole race runs in your browser. Each race is capped at 5 minutes.";
  if (api) dom.optServer.hidden = false;
  if (savedKey()) dom.apiKey.value = savedKey();
  if (api) dom.opponent.value = "server";
  else if (savedKey()) dom.opponent.value = "key";
  applyMode();

  if (!window.crypto || !window.crypto.subtle) {
    showBanner("This page needs a secure context (https) to grade answers. Open it over https.", true);
  }

  try {
    state.index = await fetch("./data/index.json").then((r) => r.json());
    dom.taskSelect.textContent = "";
    state.index.forEach((t) => {
      const option = document.createElement("option");
      option.value = t.id;
      option.textContent = `${t.id} · ${t.rows}×${t.cols} · ${t.train} examples`;
      dom.taskSelect.appendChild(option);
    });
  } catch (err) {
    showBanner(`Could not load the task list: ${err.message}`, true);
  }

  dom.btnStart.addEventListener("click", startRace);
  dom.btnStop.addEventListener("click", stopRace);
  dom.btnNew.addEventListener("click", resetView);
  dom.opponent.addEventListener("change", applyMode);
  dom.saveKey.addEventListener("click", () => {
    const v = dom.apiKey.value.trim();
    try { localStorage.setItem(KEY_STORE, v); } catch (_) { /* private mode */ }
    showBanner(v ? "Key saved in this browser." : "Key cleared.");
  });
  dom.forgetKey.addEventListener("click", () => {
    try { localStorage.removeItem(KEY_STORE); } catch (_) { /* ignore */ }
    dom.apiKey.value = "";
    showBanner("Key removed from this browser.");
  });
  dom.editor.resize.addEventListener("click", () => resizeOut(dom.editor.rows.value, dom.editor.cols.value));
  dom.editor.copy.addEventListener("click", () => { if (state.task) setOut(state.task.test_input); });
  dom.editor.clear.addEventListener("click", () => setOut(blank(state.out.length, state.out[0].length)));
  dom.editor.paint.addEventListener("click", () => selectTool("paint"));
  dom.editor.fill.addEventListener("click", () => selectTool("fill"));
  dom.editor.submit.addEventListener("click", submitAnswer);
  dom.editor.giveUp.addEventListener("click", giveUp);
  dom.editor.grid.addEventListener("pointerdown", (e) => {
    const cell = e.target.closest(".cell");
    if (!cell || !canEdit()) return;
    e.preventDefault();
    state.painting = state.tool === "paint";
    applyTool(cell);
  });
  dom.editor.grid.addEventListener("pointerover", (e) => {
    if (!state.painting) return;
    const cell = e.target.closest(".cell");
    if (cell) applyTool(cell);
  });
  window.addEventListener("pointerup", () => { state.painting = false; });
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target || {}).tagName)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^[0-9]$/.test(e.key)) selectColour(Number(e.key));
    else if (e.key.toLowerCase() === "p") selectTool("paint");
    else if (e.key.toLowerCase() === "f") selectTool("fill");
  });
}

boot();

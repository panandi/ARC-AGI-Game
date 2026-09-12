/*
 * MNIST-PRO in the browser.
 *
 * The canvases are generated offline by the upstream benchmark
 * (github.com/declare-lab/MNIST-PRO, MIT) and shipped as packed bitmaps, so
 * nothing about the image pipeline is re-implemented here. What this file
 * ports is the small part that has to run live: env.py's movement clamping,
 * one-shot answer, step limit and termination reasons, plus the masking that
 * render_observation performs.
 *
 * The model lane mirrors their GlimpseAgent in "natural" turn mode: the system
 * instruction is verbatim, observations arrive as PNGs in their own turns, and
 * the model's replies stay between them.
 */
"use strict";

const DIRECTIONS = ["up", "down", "left", "right"];
const MASK = 128;              // rendering.MASK_RGB
const BORDER = "#00E5FF";      // rendering.BORDER_HEX
const BORDER_WIDTH = 2;
const MODEL = "google/gemini-2.5-flash";
const KEY_STORE = "arc-race-openrouter-key";
const MAX_ATTEMPTS = 3;        // agent.AgentConfig.max_attempts

// specs.system_instruction(digits=1), verbatim.
const SYSTEM_INSTRUCTION =
  "You are an active vision agent playing a game to identify an MNIST digit. " +
  "Your goal is to figure out what the digit (0-9) is. " +
  "The digit is drawn in black on a white background, and unseen areas are " +
  "masked in dark gray. " +
  "You can move the visible box around. " +
  "When you are confident about the digit, you must provide your final answer. " +
  "Note: You only have one chance to provide the final answer, so make sure you " +
  "are confident before doing so! " +
  'For moving, output: {"action": "move", "direction": "up"} ' +
  "(directions: 'up', 'down', 'left', 'right'). " +
  'For answering, output: {"action": "answer", "value": <digit>}.';

// specs.TEXTUAL_BELIEF_STATE.user_instruction, verbatim.
const USER_INSTRUCTION =
  "Based on the observation, output your next move or final answer in the " +
  "requested JSON format." +
  " Include an extra key 'thought' in the JSON containing your reasoning.";

const el = (id) => document.getElementById(id);

const dom = {
  episode: el("episode-select"), view: el("view-select"),
  opponent: el("opponent-select"), modePill: el("mode-pill"),
  keyrow: el("keyrow"), apiKey: el("api-key"),
  saveKey: el("btn-save-key"), forgetKey: el("btn-forget-key"),
  start: el("btn-start"), random: el("btn-random"), stop: el("btn-stop"),
  banner: el("banner"), stepsLeft: el("steps-left"),
  guessButtons: el("guess-buttons"), dpad: document.querySelector(".dpad"),
  log: el("log"), logCount: el("log-count"),
  scoreboard: el("scoreboard"), raceResult: el("race-result"),
  verdictReason: el("verdict-reason"),
  rows: { human: el("result-human"), ai: el("result-ai") },
  human: {
    board: el("board-human"), veil: el("human-veil"), chip: el("human-chip"),
    steps: el("human-steps"), window: el("human-window"), seen: el("human-seen"),
    state: el("human-state"), result: el("human-result"),
  },
  ai: {
    board: el("board-ai"), veil: el("ai-veil"), chip: el("ai-chip"),
    steps: el("ai-steps"), window: el("ai-window"), seen: el("ai-seen"),
    state: el("ai-state"), result: el("ai-result"), thinking: el("ai-thinking"),
    rule: el("ai-rule"), model: el("ai-model"), error: el("ai-error"),
  },
};

const state = {
  index: null, episode: null, bits: null,
  human: null, ai: null,
  mode: "solo", view: "single",
  started: false, finished: false, aiAbort: false, logCount: 0,
};

/* ------------------------------ the env ------------------------------- */

const REASON = {
  running: "running", answered: "answered",
  stepLimit: "step_limit", invalid: "invalid_action",
};

function newLane(ep) {
  return {
    x: ep.start[0], y: ep.start[1],
    steps: 0, moves: 0,
    visited: [[ep.start[0], ep.start[1]]],
    done: false, reason: REASON.running,
    answer: null, success: false,
  };
}

/** env.ActiveGlimpseEnv._move — clamped to the canvas, never wrapping. */
function move(lane, direction) {
  const ep = state.episode;
  const box = ep.box_size, step = ep.step_size;
  if (direction === "up") lane.y = Math.max(0, lane.y - step);
  else if (direction === "down") lane.y = Math.min(ep.height - box, lane.y + step);
  else if (direction === "left") lane.x = Math.max(0, lane.x - step);
  else if (direction === "right") lane.x = Math.min(ep.width - box, lane.x + step);
  lane.moves += 1;
  lane.visited.push([lane.x, lane.y]);
}

/** One action. Mirrors env.step, including its step limit and reasons. */
function applyAction(lane, action) {
  if (lane.done) return;
  lane.steps += 1;
  const kind = action && action.action;

  if (kind === "move") {
    if (!DIRECTIONS.includes(action.direction)) {
      lane.done = true; lane.reason = REASON.invalid;
      return;
    }
    move(lane, action.direction);
  } else if (kind === "answer") {
    const value = String(parseInt(action.value, 10));
    lane.answer = Number.isNaN(parseInt(action.value, 10)) ? "-1" : value;
    lane.success = lane.answer === state.episode.label;
    lane.done = true; lane.reason = REASON.answered;
    return;
  } else {
    lane.done = true; lane.reason = REASON.invalid;
    return;
  }

  // wrappers.TimeLimit: truncate once the budget is spent, fabricating nothing.
  if (!lane.done && lane.steps >= state.episode.max_steps) {
    lane.done = true; lane.reason = REASON.stepLimit;
  }
}

/* ----------------------------- rendering ------------------------------ */

function unpackBits(b64, total) {
  const bin = atob(b64);
  const out = new Uint8Array(total);
  for (let i = 0; i < total; i += 1) {
    out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  }
  return out;
}

/** rendering.render_observation: grey everywhere, the window pasted sharp. */
function draw(board, lane) {
  const ep = state.episode;
  const ctx = board.getContext("2d");
  const img = ctx.createImageData(ep.width, ep.height);
  img.data.fill(255);
  for (let i = 0; i < ep.width * ep.height; i += 1) {
    img.data[i * 4] = MASK; img.data[i * 4 + 1] = MASK; img.data[i * 4 + 2] = MASK;
  }
  const windows = state.view === "trail" ? lane.visited : [[lane.x, lane.y]];
  for (const [wx, wy] of windows) {
    for (let yy = wy; yy < wy + ep.box_size; yy += 1) {
      for (let xx = wx; xx < wx + ep.box_size; xx += 1) {
        const i = yy * ep.width + xx;
        const v = state.bits[i] ? 0 : 255;
        img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = BORDER_WIDTH;
  ctx.strokeRect(lane.x + 1, lane.y + 1, ep.box_size - 2, ep.box_size - 2);
}

function seenFraction(lane) {
  const ep = state.episode;
  const seen = new Set();
  const windows = state.view === "trail" ? lane.visited : [[lane.x, lane.y]];
  for (const [wx, wy] of windows) seen.add(`${wx},${wy}`);
  const area = seen.size * ep.box_size * ep.box_size;
  return Math.min(100, Math.round((area / (ep.width * ep.height)) * 100));
}

/* ------------------------------- views -------------------------------- */

const LABELS = {
  running: "Playing", answered: "Answered",
  step_limit: "Out of steps", invalid_action: "Invalid action", idle: "Idle",
};

function renderLane(name) {
  const lane = state[name];
  const view = dom[name];
  if (!lane) {
    view.steps.textContent = name === "ai" ? "—" : "0";
    view.state.textContent = "Idle";
    view.chip.textContent = "IDLE";
    view.chip.className = "state-chip";
    return;
  }
  view.steps.textContent = String(lane.moves);
  view.window.textContent = `${lane.x}, ${lane.y}`;
  view.seen.textContent = `${seenFraction(lane)}%`;
  const label = lane.done ? (lane.success ? "Solved" : LABELS[lane.reason]) : "Playing";
  view.state.textContent = label;
  view.chip.textContent = label.toUpperCase();
  view.chip.className = "state-chip " +
    (lane.success ? "won" : lane.done ? "over" : "live");
  draw(view.board, lane);
  if (name === "human") {
    dom.stepsLeft.textContent = String(Math.max(0, state.episode.max_steps - lane.steps));
    const live = state.started && !lane.done && !state.finished;
    dom.dpad.querySelectorAll("button").forEach((b) => { b.disabled = !live; });
    dom.guessButtons.querySelectorAll("button").forEach((b) => { b.disabled = !live; });
  }
}

function addLog(who, what, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const w = document.createElement("span");
  w.className = `who ${who}`;
  w.textContent = who.toUpperCase();
  const x = document.createElement("span");
  x.className = "what";
  x.textContent = what;
  li.append(w, x);
  dom.log.appendChild(li);
  dom.log.parentElement.scrollTop = dom.log.parentElement.scrollHeight;
  state.logCount += 1;
  dom.logCount.textContent = String(state.logCount);
}

function showBanner(message, isError) {
  if (!message) { dom.banner.hidden = true; return; }
  dom.banner.textContent = message;
  dom.banner.classList.toggle("error", Boolean(isError));
  dom.banner.hidden = false;
}

/* ------------------------------ the race ------------------------------ */

function humanMove(direction) {
  const lane = state.human;
  if (!state.started || state.finished || !lane || lane.done) return;
  applyAction(lane, { action: "move", direction });
  addLog("human", `move ${direction} → (${lane.x}, ${lane.y})`);
  renderLane("human");
  if (lane.done) {
    dom.human.result.textContent = "Out of steps — no answer given.";
    dom.human.result.className = "result bad";
    maybeFinish();
  }
}

function humanAnswer(digit) {
  const lane = state.human;
  if (!state.started || state.finished || !lane || lane.done) return;
  // One answer only, so a stray keypress must not end the round.
  if (!window.confirm(`Answer ${digit}? You only get one answer.`)) return;
  applyAction(lane, { action: "answer", value: digit });
  const btn = dom.guessButtons.querySelector(`button[data-digit="${digit}"]`);
  if (btn) btn.classList.add(lane.success ? "right" : "wrong");
  dom.human.result.textContent = lane.success
    ? `Correct — it was ${state.episode.label}, in ${lane.moves} sensing steps.`
    : `Wrong — you said ${digit}. One answer only.`;
  dom.human.result.className = `result ${lane.success ? "ok" : "bad"}`;
  addLog("human", `answer ${digit} — ${lane.success ? "CORRECT" : "wrong"}`,
    lane.success ? "hl" : "err");
  renderLane("human");
  maybeFinish();
}

function maybeFinish() {
  if (state.finished) return;
  const aiIdle = state.mode === "solo" || !state.ai || state.ai.done;
  if (state.human && state.human.done && aiIdle) finishRace();
}

function decide() {
  const h = state.human, a = state.ai;
  if (state.mode === "solo") {
    return h.success
      ? ["human", `identified it in ${h.moves} sensing steps`]
      : ["none", h.reason === REASON.stepLimit ? "ran out of steps" : "wrong answer"];
  }
  if (h.success && a.success) {
    if (h.moves < a.moves) return ["human", "correct, and in fewer sensing steps"];
    if (a.moves < h.moves) return ["ai", "correct, and in fewer sensing steps"];
    return ["none", `both correct in ${h.moves} steps — a tie`];
  }
  if (h.success) return ["human", "the only one to identify it"];
  if (a.success) return ["ai", "the only one to identify it"];
  return ["none", "neither identified the digit"];
}

function fillRow(name, lane, winner) {
  const row = dom.rows[name];
  row.querySelector(".r-answer").textContent =
    lane && lane.answer !== null ? lane.answer : "—";
  const outcome = row.querySelector(".r-outcome");
  outcome.textContent = !lane ? "Did not play"
    : lane.success ? "Correct"
      : lane.reason === REASON.stepLimit ? "Out of steps"
        : lane.reason === REASON.invalid ? "Invalid action" : "Wrong";
  outcome.className = `r-outcome ${lane && lane.success ? "done" : "miss"}`;
  row.querySelector(".r-steps").textContent = lane ? String(lane.moves) : "—";
  row.classList.toggle("is-winner", winner);
}

function finishRace() {
  state.finished = true;
  state.aiAbort = true;
  dom.ai.thinking.hidden = true;
  const [winner, reason] = decide();
  dom.scoreboard.hidden = false;
  dom.raceResult.textContent = winner === "human" ? "YOU WON"
    : winner === "ai" ? "THE AI WON" : "NO WINNER";
  dom.raceResult.className = winner === "human" ? "human-won"
    : winner === "ai" ? "ai-won" : "";
  dom.verdictReason.textContent = `${reason}. The digit was ${state.episode.label}.`;
  fillRow("human", state.human, winner === "human");
  fillRow("ai", state.ai, winner === "ai");
  dom.stop.disabled = true;
  renderLane("human");
  addLog("sys", `round over — ${reason}`, "hl");
  dom.scoreboard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* -------------------------------- the AI ------------------------------ */

function observationPng(lane) {
  const off = document.createElement("canvas");
  off.width = state.episode.width;
  off.height = state.episode.height;
  const keep = state.view;
  state.view = "single";              // the model always gets one window
  draw(off, lane);
  state.view = keep;
  return off.toDataURL("image/png");
}

/** agent.extract_json — greedy, first brace to last. */
function extractJson(raw) {
  if (!raw) return null;
  const m = String(raw).match(/\{[\s\S]*\}/);
  try { return JSON.parse(m ? m[0] : raw); } catch (_) { return null; }
}

async function callModel(key, history) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 800,
      messages: [{ role: "system", content: SYSTEM_INSTRUCTION }, ...history],
    }),
  });
  if (!response.ok) {
    const map = {
      401: "your key was rejected",
      402: "your OpenRouter account is out of credits",
      429: "rate limited by the provider",
    };
    throw new Error(map[response.status] || `provider returned HTTP ${response.status}`);
  }
  const body = await response.json();
  return ((body.choices || [])[0] || {}).message?.content || "";
}

async function runAi(key) {
  const lane = state.ai;
  const history = [];
  dom.ai.veil.hidden = true;

  while (!state.aiAbort && !state.finished && !lane.done) {
    dom.ai.thinking.hidden = false;
    dom.ai.thinking.textContent = `Thinking about step ${lane.moves + 1}…`;

    // agent._act_natural: instruction only on the first turn, image every turn.
    const content = [];
    if (history.length === 0) content.push({ type: "text", text: USER_INSTRUCTION });
    content.push({ type: "image_url", image_url: { url: observationPng(lane) } });
    history.push({ role: "user", content });

    let parsed = null;
    let raw = "";
    for (let attempt = 0; attempt < MAX_ATTEMPTS && parsed === null; attempt += 1) {
      try {
        raw = await callModel(key, history);
      } catch (err) {
        lane.done = true;
        lane.reason = REASON.invalid;
        dom.ai.error.hidden = false;
        dom.ai.error.textContent = `AI stopped: ${err.message}`;
        dom.ai.thinking.hidden = true;
        addLog("ai", `stopped: ${err.message}`, "err");
        renderLane("ai");
        maybeFinish();
        return;
      }
      parsed = extractJson(raw);
    }
    history.push({ role: "assistant", content: raw });
    dom.ai.thinking.hidden = true;

    if (parsed === null) {
      applyAction(lane, { action: "invalid" });
      addLog("ai", "could not produce JSON — invalid action", "err");
      renderLane("ai");
      break;
    }
    if (parsed.thought) dom.ai.rule.textContent = `"${String(parsed.thought).slice(0, 240)}"`;

    applyAction(lane, parsed);
    if (parsed.action === "answer") {
      dom.ai.result.textContent = lane.success
        ? `Correct — ${lane.answer} in ${lane.moves} sensing steps.`
        : `Wrong — it answered ${lane.answer}.`;
      dom.ai.result.className = `result ${lane.success ? "ok" : "bad"}`;
      addLog("ai", `answer ${lane.answer} — ${lane.success ? "CORRECT" : "wrong"}`,
        lane.success ? "hl" : "err");
    } else if (parsed.action === "move") {
      addLog("ai", `move ${parsed.direction} → (${lane.x}, ${lane.y})`);
    }
    renderLane("ai");
  }
  maybeFinish();
}

/* ----------------------------- race control --------------------------- */

const savedKey = () => {
  try { return localStorage.getItem(KEY_STORE) || ""; } catch (_) { return ""; }
};

function reset() {
  state.started = false;
  state.finished = false;
  state.aiAbort = true;
  state.human = null;
  state.ai = null;
  state.logCount = 0;
  dom.log.textContent = "";
  dom.logCount.textContent = "0";
  dom.scoreboard.hidden = true;
  dom.human.veil.hidden = false;
  dom.ai.veil.hidden = false;
  dom.human.result.textContent = "";
  dom.ai.result.textContent = "";
  dom.ai.error.hidden = true;
  dom.ai.rule.textContent = "—";
  dom.ai.thinking.hidden = true;
  dom.guessButtons.querySelectorAll("button").forEach((b) => {
    b.className = "";
    b.disabled = true;
  });
  dom.dpad.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  Object.values(dom.rows).forEach((r) => {
    r.classList.remove("is-winner");
    r.querySelectorAll("td").forEach((c) => { c.textContent = "—"; });
  });
  renderLane("human");
  renderLane("ai");
  showBanner(null);
}

async function startRound() {
  const id = dom.episode.value;
  if (!id) { showBanner("Pick an episode first.", true); return; }
  state.mode = dom.opponent.value;
  if (state.mode === "key" && !savedKey()) {
    showBanner("Save an OpenRouter key first, or play solo.", true);
    dom.keyrow.hidden = false;
    return;
  }

  reset();
  dom.start.disabled = true;
  try {
    state.episode = await fetch(`./data/mnistpro/episodes/${id}.json`).then((r) => r.json());
  } catch (err) {
    showBanner(`Could not load that episode: ${err.message}`, true);
    dom.start.disabled = false;
    return;
  }
  state.bits = unpackBits(state.episode.bits, state.episode.width * state.episode.height);
  state.view = dom.view.value;
  state.human = newLane(state.episode);
  state.ai = state.mode === "solo" ? null : newLane(state.episode);
  state.started = true;
  state.aiAbort = false;
  state.finished = false;

  dom.human.veil.hidden = true;
  dom.start.disabled = false;
  dom.stop.disabled = false;
  renderLane("human");
  if (state.ai) renderLane("ai");
  addLog("sys", `episode ${id} — ${state.episode.max_steps} steps, ` +
    `${state.episode.box_size}px window`, "hl");

  if (state.mode === "key") runAi(savedKey());
}

function stopRound() {
  if (!state.started || state.finished) return;
  state.aiAbort = true;
  if (state.human && !state.human.done) {
    state.human.done = true;
    state.human.reason = REASON.stepLimit;
  }
  if (state.ai && !state.ai.done) {
    state.ai.done = true;
    state.ai.reason = REASON.stepLimit;
  }
  addLog("sys", "stopped");
  finishRace();
}

/* -------------------------------- boot -------------------------------- */

function applyMode() {
  const mode = dom.opponent.value;
  dom.keyrow.hidden = mode !== "key";
  dom.ai.model.textContent = mode === "solo" ? "—" : MODEL;
  dom.modePill.textContent = mode === "solo" ? "solo" : "vs AI";
}

function buildGuessButtons() {
  dom.guessButtons.textContent = "";
  for (let d = 0; d <= 9; d += 1) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.digit = String(d);
    b.textContent = String(d);
    b.disabled = true;
    b.addEventListener("click", () => humanAnswer(d));
    dom.guessButtons.appendChild(b);
  }
}

async function boot() {
  buildGuessButtons();
  reset();
  applyMode();
  if (savedKey()) dom.apiKey.value = savedKey();

  try {
    state.index = await fetch("./data/mnistpro/index.json").then((r) => r.json());
    dom.episode.textContent = "";
    state.index.episodes.forEach((e, i) => {
      const o = document.createElement("option");
      o.value = e.id;
      o.textContent = `Episode ${i} · ${Math.round(e.ink * 100)}% ink`;
      dom.episode.appendChild(o);
    });
    dom.stepsLeft.textContent = String(state.index.max_steps);
  } catch (err) {
    showBanner(`Could not load the episode list: ${err.message}`, true);
  }

  dom.start.addEventListener("click", startRound);
  dom.stop.addEventListener("click", stopRound);
  dom.random.addEventListener("click", () => {
    const n = dom.episode.options.length;
    if (n) dom.episode.selectedIndex = Math.floor(Math.random() * n);
    startRound();
  });
  dom.opponent.addEventListener("change", applyMode);
  dom.view.addEventListener("change", () => {
    state.view = dom.view.value;
    if (state.human) renderLane("human");
    if (state.ai) renderLane("ai");
  });
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
  dom.dpad.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-dir]");
    if (b) humanMove(b.dataset.dir);
  });
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes((e.target || {}).tagName)) return;
    const arrows = {
      ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
   };
    if (arrows[e.key]) { e.preventDefault(); humanMove(arrows[e.key]); }
    else if (/^[0-9]$/.test(e.key)) humanAnswer(Number(e.key));
  });
}

boot();

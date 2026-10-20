// training-loop frontend.
//   #/                 topic list
//   #/<topic>          the topic's modules (one per COMP_GUIDE subsection)
//   #/<topic>/<module> one-problem-at-a-time practice; every module has its own counter and medal
"use strict";

// Per-topic editor mode and Pyodide packages (`extra`: heavy packages only some modules need).
// Titles and modules come from build.json (COMP_GUIDE).
const TOPICS = [
  { id: "numpy", part: 1, mode: "python", packages: ["numpy"] },
  { id: "pandas", part: 1, mode: "python", packages: ["numpy", "pandas"], extra: { "2.03": ["jinja2"], "2.11": ["tzdata"], "2.18": ["matplotlib", "jinja2"] } },
  { id: "pytorch", part: 1, mode: "python", packages: ["numpy"] },
  { id: "sklearn", part: 1, mode: "python", packages: ["numpy", "scikit-learn"] },
  { id: "matplotlib", part: 1, mode: "python", packages: ["numpy", "matplotlib"] },
  { id: "huggingface", part: 1, mode: "python", packages: ["numpy"] },
  { id: "bash", part: 2, mode: "shell", packages: [] },
  { id: "git", part: 2, mode: "shell", packages: [] },
  { id: "vim", part: 2, mode: null, packages: [] },
  { id: "sql", part: 2, mode: "text/x-sql", packages: [] },
  { id: "latex", part: 2, mode: null, packages: [] },
  { id: "docker", part: 2, mode: "shell", packages: [] },
];
const TOPIC = Object.fromEntries(TOPICS.map((t) => [t.id, t]));
const EXAMPLE_UNTIL = 10;
const MEDAL_AT = 250;
const REVEAL_AFTER = 250;

// ------------------------------------------------------------ catalog (docs/build.json)
let catalogPromise = null;
function catalog() {
  catalogPromise ||= fetch("build.json", { cache: "no-cache" })
    .then((r) => r.json())
    .then((b) => b.topics || {})
    .catch(() => ({}));
  return catalogPromise;
}
const liveModules = (cat, id) => (cat[id]?.modules || []).filter((m) => m.live);

// ------------------------------------------------------------ storage (per module)
// v1 (one counter per topic) is left untouched in localStorage; v2 counts per module.
const STORE_KEY = "training-loop:v2";
const state = (() => {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || { modules: {} }; }
  catch { return { modules: {} }; }
})();
function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch { /* private mode */ } }
function moduleState(topicId, moduleId) {
  return (state.modules[`${topicId}/${moduleId}`] ||= { solved: 0, medal: false, current: null });
}
function topicTotals(topicId) {
  let solved = 0, medals = 0;
  for (const [key, ms] of Object.entries(state.modules)) {
    if (key.startsWith(topicId + "/")) { solved += ms.solved; medals += ms.medal ? 1 : 0; }
  }
  return { solved, medals };
}
const newSeed = () => Math.floor(Math.random() * 2 ** 31);

// ------------------------------------------------------------ Python worker
class TimeLimit extends Error {}

class Py {
  constructor() { this.packages = []; this.spawn(); }
  spawn() {
    this.worker = new Worker("worker.js", { type: "module" });
    this.pending = new Map();
    this.seq = 0;
    this.ready = null;
    this.worker.onerror = (e) => { // e.g. the Pyodide CDN failed to load
      e.preventDefault();
      const err = new Error(e.message || "the Python worker failed to start (check your connection)");
      for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(err); }
      this.pending.clear();
    };
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      clearTimeout(p.timer);
      data.ok ? p.resolve(data.data) : p.reject(new Error(data.error));
    };
  }
  send(cmd, args, ms) {
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      const timer = ms && setTimeout(() => { this.pending.delete(id); this.restart(); reject(new TimeLimit(ms)); }, ms);
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ id, cmd, args });
    });
  }
  restart() {
    this.worker.terminate();
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new Error("Python restarted")); }
    this.spawn();
    this.ready = this.send("init", { packages: this.packages });
  }
  init(packages) {
    this.packages = [...new Set([...this.packages, ...packages])];
    this.ready = this.send("init", { packages: this.packages });
    return this.ready;
  }
  async call(cmd, args, ms) { await this.ready; return this.send(cmd, args, ms); }
}
let py = null;

// ------------------------------------------------------------ tiny markdown
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function inline(text) {
  // code spans first, so `**` inside code is never read as bold
  return esc(text).split(/(`[^`]+`)/)
    .map((seg, i) => (i % 2 ? `<code>${seg.slice(1, -1)}</code>` : seg.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")))
    .join("");
}
function md(text) {
  return text.split(/\n{2,}/).map((block) => {
    if (block.startsWith("```")) return `<pre class="code-block">${esc(block.replace(/^```\w*\n?|```$/g, ""))}</pre>`;
    const lines = block.split("\n");
    if (lines.every((l) => /^(\* |\d+\. )/.test(l))) {
      const tag = lines[0].startsWith("* ") ? "ul" : "ol";
      return `<${tag}>${lines.map((l) => `<li>${inline(l.replace(/^(\* |\d+\. )/, ""))}</li>`).join("")}</${tag}>`;
    }
    return `<p>${inline(block).replace(/\n/g, "<br>")}</p>`;
  }).join("");
}
function codeBlock(code, mode = "python") {
  const pre = document.createElement("pre");
  pre.className = "code-block cm-s-default";
  if (window.CodeMirror && CodeMirror.runMode && mode) CodeMirror.runMode(code, mode, pre);
  else pre.textContent = code;
  return pre;
}

// ------------------------------------------------------------ medals (placeholder art; real designs TBD)
function medalSVG(topicId, moduleId, label) {
  const hue = (TOPICS.findIndex((t) => t.id === topicId) * 137) % 360;
  const k = Number(moduleId.split(".")[1]) || 0;  // vary the ribbon per module
  const ribbon = (hue + k * 23) % 360;
  const text = moduleId.split(".")[1] || "";
  return `<svg viewBox="0 0 64 64" role="img" aria-label="${esc(label)} medal">
    <path d="M22 2h8l6 22h-8z" fill="hsl(${ribbon} 55% 55%)"/><path d="M42 2h-8l-6 22h8z" fill="hsl(${ribbon} 55% 42%)"/>
    <circle cx="32" cy="40" r="20" fill="hsl(${hue} 70% 62%)" stroke="hsl(${hue} 50% 40%)" stroke-width="2.5"/>
    <circle cx="32" cy="40" r="14" fill="none" stroke="hsl(${hue} 60% 85%)" stroke-width="1.5" stroke-dasharray="2 3"/>
    <text x="32" y="45" text-anchor="middle" font-family="Georgia, serif" font-size="14" fill="hsl(${hue} 60% 20%)">${esc(text)}</text>
  </svg>`;
}

// ------------------------------------------------------------ home: topic list
async function renderHome() {
  document.title = "training-loop";
  const cat = await catalog();
  const row = (t) => {
    const title = cat[t.id]?.title || t.id;
    const live = liveModules(cat, t.id).length;
    if (!live) {
      return `<li><div class="topic soon"><span class="t-title">${esc(title)}</span>
        <span class="t-right"><span class="t-count">soon</span><span class="badge"></span></span></div></li>`;
    }
    const { solved, medals } = topicTotals(t.id);
    const medalChip = medals
      ? `<span class="medal-chip" title="${medals} medal${medals > 1 ? "s" : ""}">${medalSVG(t.id, "", title)}<span>×${medals}</span></span>`
      : "";
    return `<li><a class="topic" href="#/${t.id}"><span class="t-title">${esc(title)}</span>
      <span class="t-right"><span class="t-count">${solved}</span><span class="badge wide">${medalChip}</span></span></a></li>`;
  };
  const part = (n, name) => `<section class="part"><h2>Part ${n === 1 ? "I" : "II"} · ${name}</h2>
    <ol class="topics">${TOPICS.filter((t) => t.part === n).map(row).join("")}</ol></section>`;
  app().innerHTML = `<div class="home">
    <h1>training-loop</h1>
    <p class="tag">One problem at a time. Muscle memory for the ML stack and the tools around it.</p>
    ${part(1, "The Python ML stack")}${part(2, "Developer tools")}
  </div>`;
}

// ------------------------------------------------------------ topic page: module list
async function renderTopic(topic) {
  const cat = await catalog();
  const info = cat[topic.id];
  document.title = `${info.title} · training-loop`;
  const row = (m) => {
    const head = `<span class="t-title"><span class="m-id">${esc(m.id)}</span>${esc(m.title)}</span>`;
    if (!m.live) {
      return `<li><div class="topic soon">${head}<span class="t-right"><span class="t-count">soon</span><span class="badge"></span></span></div></li>`;
    }
    const ms = moduleState(topic.id, m.id);
    return `<li><a class="topic" href="#/${topic.id}/${m.id}">${head}<span class="t-right">
      <span class="t-count">${ms.solved}</span><span class="badge">${ms.medal ? medalSVG(topic.id, m.id, m.title) : ""}</span>
      </span></a></li>`;
  };
  app().innerHTML = `<div class="home">
    <a class="back" href="#/">← Topics</a>
    <h1>${esc(info.title)}</h1>
    <p class="tag">${info.modules.length} modules. Each has its own counter, 250-problem medal and endless questions.</p>
    <ol class="topics modules">${info.modules.map(row).join("")}</ol>
  </div>`;
}

// ------------------------------------------------------------ practice
let view = null; // { topic, mod, ms, problem, editor, phase }

async function renderPractice(topic, mod, topicTitle) {
  const ms = moduleState(topic.id, mod.id);
  document.title = `${mod.id} ${mod.title} · training-loop`;
  app().innerHTML = `
    <div class="bar"><a href="#/${topic.id}">← ${esc(topicTitle)}</a>
      <span class="crumb"><span class="m-id">${esc(mod.id)}</span>${esc(mod.title)}</span>
      <span class="qnum" title="Question number"></span></div>
    <div class="practice">
      <section>
        <article class="problem"><div class="loading">Starting Python… (the first load takes a few seconds)</div></article>
        <div class="editor"><textarea></textarea></div>
        <div class="actions">
          <button class="btn run" disabled>Run</button>
          <button class="btn primary submit" disabled>Submit</button>
          <span class="hint">⌘/Ctrl+Enter run · Shift+Enter submit</span>
          <button class="btn primary next" hidden>Next →</button>
        </div>
      </section>
      <aside class="console">
        <div class="head"><div class="verdict"></div><span class="tier"></span></div>
        <div class="verdict-msg"></div>
        <pre class="out"><span class="dim">Output appears here.</span></pre>
      </aside>
    </div>`;
  const editor = CodeMirror.fromTextArea(app().querySelector("textarea"), {
    mode: topic.mode, lineNumbers: true, indentUnit: 4, matchBrackets: true, autoCloseBrackets: true,
    viewportMargin: Infinity,
    extraKeys: {
      "Cmd-Enter": () => doRun(), "Ctrl-Enter": () => doRun(), "Shift-Enter": () => doSubmit(),
      Tab: (cm) => (cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection("    ")),
    },
  });
  editor.on("change", () => { if (view && view.problem) { ms.draft = { seed: view.problem.seed, code: editor.getValue() }; save(); } });
  view = { topic, mod, ms, problem: null, editor, phase: "loading" };
  $(".run").onclick = doRun;
  $(".submit").onclick = doSubmit;
  $(".next").onclick = () => loadProblem();

  if (!py) py = new Py();
  try { await py.init([...topic.packages, ...((topic.extra || {})[mod.id] || [])]); }
  catch (e) { $(".problem").innerHTML = `<div class="loading">Couldn't start Python: ${esc(e.message)}</div>`; return; }
  await loadProblem();
}

async function loadProblem() {
  const { ms, topic, mod } = view;
  if (!ms.current) ms.current = { n: ms.solved + 1, seed: newSeed(), tier: null };
  const cur = ms.current;
  const v = view;
  setBusy(true);
  let p;
  try { p = await py.call("generate", { section: topic.id, module: mod.id, n: cur.n, seed: cur.seed, tier: cur.tier }, 20000); }
  catch (e) { if (view === v) $(".problem").innerHTML = `<div class="loading">Couldn't generate a problem: ${esc(e.message)}</div>`; return; }
  if (view !== v) return; // navigated away meanwhile
  cur.tier = p.tier;
  save();
  view.problem = p;
  view.phase = "answering";
  drawProblem(p, cur.n);
  view.editor.setValue(ms.draft && ms.draft.seed === p.seed ? ms.draft.code : "");
  view.editor.focus();
  $(".verdict").textContent = "";
  $(".verdict").className = "verdict";
  $(".verdict-msg").textContent = "";
  $(".tier").textContent = p.tier;
  $(".out").innerHTML = `<span class="dim">Output appears here.</span>`;
  $(".next").hidden = true;
  setBusy(false);
}

function drawProblem(p, n) {
  $(".qnum").textContent = n;
  const mode = view.topic.mode;
  const el = $(".problem");
  el.innerHTML = `<div class="prompt">${md(p.prompt)}</div>`;
  if (p.setup.trim()) {
    el.insertAdjacentHTML("beforeend", `<div class="label">Given</div>`);
    el.appendChild(codeBlock(p.setup, mode));
  }
  if (p.example) {
    const d = document.createElement("details");
    d.className = "example";
    d.open = n <= EXAMPLE_UNTIL;
    d.innerHTML = `<summary>Example</summary><div class="inner"><div class="prompt">${md(p.example.prompt)}</div></div>`;
    const inner = d.querySelector(".inner");
    if (p.example.setup.trim()) { inner.insertAdjacentHTML("beforeend", `<div class="label">Given</div>`); inner.appendChild(codeBlock(p.example.setup, mode)); }
    inner.insertAdjacentHTML("beforeend", `<div class="label">Solution</div>`);
    inner.appendChild(codeBlock(p.example.solution, mode));
    el.appendChild(d);
  }
  if (n > REVEAL_AFTER) {
    const d = document.createElement("details");
    d.className = "answer";
    d.innerHTML = `<summary>Show answer</summary><div class="inner"></div>`;
    d.querySelector(".inner").appendChild(codeBlock(p.solution, mode));
    el.appendChild(d);
  }
}

function setBusy(busy) {
  const answering = view.phase === "answering";
  $(".run").disabled = busy || !view.problem;
  $(".submit").disabled = busy || !answering;
}

function showOutput(res, timedOut) {
  const parts = [];
  if (res.stdout) parts.push(esc(res.stdout.replace(/\n$/, "")));
  if (res.value != null) parts.push(`<span class="dim">Out:</span> ${esc(res.value)}`);
  if (res.error) parts.push(`<span class="err">${esc(res.error)}</span>`);
  if (timedOut) parts.push(`<span class="err">Time limit exceeded: your code ran too long (infinite loop?).</span>`);
  $(".out").innerHTML = parts.join("\n") || `<span class="dim">(no output)</span>`;
  // on narrow screens the console sits below the editor: bring it into view
  if (window.matchMedia("(max-width: 860px)").matches) $(".console").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function doRun() {
  if (!view.problem || $(".run").disabled) return;
  setBusy(true);
  $(".out").innerHTML = `<span class="dim">Running…</span>`;
  const v = view;
  let res;
  try { res = await py.call("run", { problem: v.problem, code: v.editor.getValue() }, (v.problem.timeout + 2) * 1000); }
  catch (e) { res = e instanceof TimeLimit ? { timedOut: true } : { error: e.message }; }
  if (view !== v) return;
  showOutput(res, res.timedOut);
  setBusy(false);
}

async function doSubmit() {
  if (view.phase !== "answering" || $(".submit").disabled) return;
  const { ms, problem } = view;
  const code = view.editor.getValue();
  if (!code.trim()) return;
  setBusy(true);
  $(".out").innerHTML = `<span class="dim">Checking…</span>`;
  let res;
  try { res = await py.call("grade", { problem, code }, (problem.timeout + 2) * 1000); }
  catch (e) {
    const timedOut = e instanceof TimeLimit;
    res = { correct: false, compiled: true, error: timedOut ? null : e.message, message: timedOut ? "Time limit exceeded." : "" };
    showOutput(res, timedOut);
  }
  if (!view || view.problem !== problem) return; // navigated away meanwhile
  if (res.stdout !== undefined) showOutput(res);

  if (!res.compiled) { // doesn't parse: let them fix it without losing the problem
    $(".verdict").textContent = "Doesn't compile";
    $(".verdict").className = "verdict bad";
    $(".verdict-msg").textContent = "Fix the syntax error and submit again.";
    setBusy(false);
    return;
  }
  view.phase = "done";
  if (res.correct) {
    ms.solved += 1;
    ms.current = { n: ms.solved + 1, seed: newSeed(), tier: null };
    $(".verdict").textContent = "✓ Correct";
    $(".verdict").className = "verdict good";
    $(".next").textContent = "Next →";
  } else {
    ms.current = { n: ms.current.n, seed: newSeed(), tier: problem.tier }; // same number, same tier, new problem
    $(".verdict").textContent = "✗ Wrong";
    $(".verdict").className = "verdict bad";
    $(".next").textContent = "New problem →";
  }
  $(".verdict-msg").textContent = res.correct ? (res.message || "") : (res.message || (res.error ? "Your code raised an error." : ""));
  ms.draft = null;
  save();
  $(".next").hidden = false;
  $(".next").focus();
  setBusy(false);
  if (res.correct && ms.solved >= MEDAL_AT && !ms.medal) { ms.medal = true; save(); showMedal(view.topic, view.mod); }
}

function showMedal(topic, mod) {
  const m = document.createElement("div");
  m.className = "modal";
  m.innerHTML = `<div class="card">${medalSVG(topic.id, mod.id, mod.title)}<h2>Surprise!</h2>
    <p>You've solved ${MEDAL_AT} problems in ${esc(mod.id)} ${esc(mod.title)}. This medal is yours.</p>
    <button class="btn primary">Keep going</button></div>`;
  m.querySelector("button").onclick = () => m.remove();
  document.body.appendChild(m);
}

// ------------------------------------------------------------ router
const app = () => document.getElementById("app");
const $ = (sel) => app().querySelector(sel);

async function route() {
  const [topicId, moduleId] = location.hash.replace(/^#\/?/, "").split("/");
  view = null;
  const cat = await catalog();
  const topic = TOPIC[topicId];
  if (!topic || !liveModules(cat, topicId).length) return renderHome();
  if (!moduleId) return renderTopic(topic);
  const mod = liveModules(cat, topicId).find((m) => m.id === moduleId);
  return mod ? renderPractice(topic, mod, cat[topicId].title) : renderTopic(topic);
}
window.addEventListener("hashchange", () => { window.scrollTo(0, 0); route(); });
route();

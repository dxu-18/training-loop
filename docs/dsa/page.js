// Part III: Data Structures & Algorithms. Step-by-step tutorials instead of counted problems.
//   #/dsa/<id>   setup (options + live complexity) -> steps (write each piece, Check it) -> final challenge -> done
// Status on the home page: Not Started · In Progress · Almost! (every step passed, final pending) · Completed.
(function (root) {
  "use strict";
  const DL = root.DSALang;
  const RUN = root.DSARun;

  const LIST = [
    { id: "binary-search", num: "13.01", title: "Binary search" },
    { id: "mergesort", num: "13.02", title: "Mergesort" },
    { id: "quicksort", num: "13.03", title: "Quicksort" },
    { id: "linked-lists", num: "13.04", title: "Linked lists" },
    { id: "stacks", num: "13.05", title: "Stacks" },
    { id: "queues", num: "13.06", title: "Queues" },
    { id: "unbounded-arrays", num: "13.07", title: "Unbounded arrays" },
    { id: "hash-tables", num: "13.08", title: "Hash tables" },
    { id: "binary-search-trees", num: "13.09", title: "Binary search trees" },
    { id: "dictionaries-sets", num: "13.10", title: "Dictionaries & sets" },
    { id: "heaps", num: "13.11", title: "Heaps" },
    { id: "priority-queues", num: "13.12", title: "Priority queues" },
  ];
  const tutorial = (id) => (root.DSATutorials || {})[id];

  let deps = null;   // from app.js: { app, esc, inline, md, codeBlock, getPy, TimeLimit }
  function init(d) { deps = d; }

  // ------------------------------------------------------------ storage
  const STORE_KEY = "training-loop:dsa:v1";
  const store = (() => {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || { tutorials: {} }; } catch { return { tutorials: {} }; }
  })();
  function save() { try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch { /* private mode */ } }
  const blank = () => ({ phase: "setup", opts: null, passed: 0, view: 0, code: "", final: { code: "", passed: false } });
  const tstate = (id) => (store.tutorials[id] ||= blank());

  const STATUS = { not: "Not Started", progress: "In Progress", almost: "Almost!", done: "Completed" };
  function status(id) {
    const s = store.tutorials[id], t = tutorial(id);
    if (!s || !t || s.phase === "setup") return "not";
    if (s.final.passed) return "done";
    return s.passed >= t.stepCount(s.opts) ? "almost" : "progress";
  }
  const chip = (id) => `<span class="dsa-status st-${status(id)}">${STATUS[status(id)]}</span>`;

  function homeRows() {
    const esc = deps.esc;
    return LIST.map((x) => {
      const head = `<span class="t-title"><span class="m-id">${esc(x.num)}</span>${esc(x.title)}</span>`;
      if (!tutorial(x.id)) {
        return `<li><div class="topic soon">${head}<span class="t-right"><span class="t-count">soon</span><span class="badge"></span></span></div></li>`;
      }
      return `<li><a class="topic" href="#/dsa/${x.id}">${head}<span class="t-right">${chip(x.id)}</span></a></li>`;
    }).join("");
  }

  // ------------------------------------------------------------ page
  let page = null;   // { id, t, s, editor, busy, token }
  const $ = (sel) => deps.app().querySelector(sel);

  function leave() { if (page && page.cleanup) page.cleanup(); page = null; }

  function render(id) {
    const t = tutorial(id);
    const meta = LIST.find((x) => x.id === id);
    if (!t || !meta) return false;
    const s = tstate(id);
    // tutorials begun before the Contracts option existed were written with contracts: keep them that way
    if (s.opts && s.phase !== "setup" && !s.opts.contracts) { s.opts.contracts = "on"; save(); }
    page = { id, t, s, meta, editor: null, busy: false, token: {} };
    document.title = `${meta.num} ${meta.title} · training-loop`;
    if (s.phase === "setup") renderSetup();
    else if (s.phase === "steps") renderSteps();
    else renderFinal();
    return true;
  }

  function bar() {
    const { meta, s, t } = page;
    const esc = deps.esc;
    const summary = s.phase === "setup" || !s.opts ? "" : `<span class="dsa-opts">${esc(optSummary(t, s.opts))}</span>`;
    return `<div class="bar"><a href="#/">← Topics</a>
      <span class="crumb"><span class="m-id">${esc(meta.num)}</span>${esc(meta.title)}</span>${summary}
      <span class="bar-right">${chip(page.id)}${s.phase === "setup" ? "" : '<button class="btn small reset">Reset</button>'}</span></div>`;
  }
  const optSummary = (t, o) => t.OPTIONS.map((x) => (x.summary ? x.summary[o[x.id]] : x.choices.find(([v]) => v === o[x.id])[1])).join(" · ");

  function wireBar() {
    const r = $(".reset");
    if (r) r.onclick = confirmReset;
  }

  function confirmReset() {
    const m = document.createElement("div");
    m.className = "modal";
    m.innerHTML = `<div class="card"><h2>Reset this tutorial?</h2>
      <p>Your code and progress on ${deps.esc(page.meta.title)} will be erased, and you can choose new options.</p>
      <div class="modal-actions"><button class="btn cancel">Cancel</button><button class="btn danger confirm">Reset</button></div></div>`;
    const close = () => m.remove();
    m.querySelector(".cancel").onclick = close;
    m.addEventListener("click", (e) => { if (e.target === m) close(); });
    m.querySelector(".confirm").onclick = () => {
      close();
      store.tutorials[page.id] = blank();
      save();
      render(page.id);
    };
    document.body.appendChild(m);
    m.querySelector(".cancel").focus();
  }

  // ------------------------------------------------------------ complexity boxes and the diagram
  function cxHTML(o, prev) {
    const { inline } = deps;
    return page.t.complexity(o).map((c) => {
      const changed = prev && prev[c.id] && prev[c.id] !== c.big;
      return `<div class="cx${c.extra ? " extra" : ""}${changed ? " changed" : ""}" data-id="${c.id}" title="${c.extra ? "Not one of this tutorial's functions; shown for comparison." : ""}">
        <div class="cx-label">${deps.esc(c.label)}</div><div class="cx-big">${deps.esc(c.big)}</div><div class="cx-note">${inline(c.note)}</div></div>`;
    }).join("");
  }
  const cxValues = (o) => Object.fromEntries(page.t.complexity(o).map((c) => [c.id, c.big]));

  function diagramHTML(o) {
    const esc = deps.esc;
    const d = page.t.diagram(o);
    const cell = (c) => {
      if (c.box !== undefined) {
        // a label longer than its box hangs over the next cells instead of widening its column
        const wide = (x) => x.length > String(c.box).length + 1;
        const tags = (c.tags || []).map((x) => `<span${wide(x) ? ' class="wide"' : ""}>${esc(x) || "&nbsp;"}</span>`).join("");
        return `<div class="dg-cell"><div class="dg-tags">${tags}</div><div class="dg-box ${c.cls || ""}">${esc(c.box)}</div></div>`;
      }
      if (c.label) return `<div class="dg-cell"><div class="dg-tags"></div><div class="dg-label">${esc(c.label)}</div></div>`;
      if (c.arrow) return `<div class="dg-cell"><div class="dg-tags"></div><div class="dg-arrow">${esc(c.arrow)}</div></div>`;
      if (c.loop) return `<div class="dg-cell"><div class="dg-tags"></div><div class="dg-loop">${esc(c.loop)}</div></div>`;
      return `<div class="dg-cell"><div class="dg-tags"></div><div class="dg-null">${esc(c.text)}</div></div>`;
    };
    const rows = (d.rows || [d.cells]).map((r) => `<div class="dg-row">${r.map(cell).join("")}</div>`).join("");
    // [segments) and X.field names in the caption read as code
    return `<div class="diagram">${rows}<div class="dg-caption">${deps.inline(d.caption.replace(/((?:\b[A-Z](?:\.\w+)*)?\[[^\[\]()]*[\])]|\b[A-Z]\.\w+(?:\.\w+)*)/g, "`$1`"))}</div></div>`;
  }

  function runnerNote(lang) {
    if (lang === "py") return "Python runs in your browser (Pyodide).";
    const c = RUN.COMPILERS[lang];
    return `${DL.LANGS[lang].name} runs on Compiler Explorer (godbolt.org, ${c.name}): your code is sent there to be compiled and run.`;
  }

  // ------------------------------------------------------------ setup: options, live complexity, begin
  function renderSetup() {
    const { t, s } = page;
    const esc = deps.esc;
    const o = { ...t.DEFAULTS, ...(s.opts || {}) };
    const optHTML = t.OPTIONS.map((x) => `<div class="opt" data-opt="${x.id}"><div class="opt-label">${esc(x.label)}</div>
      <div class="seg" role="radiogroup" aria-label="${esc(x.label)}">${x.choices.map(([v, label]) =>
        `<button type="button" role="radio" data-v="${v}" aria-checked="${o[x.id] === v}" class="${o[x.id] === v ? "on" : ""}">${esc(label)}</button>`).join("")}</div>
      <div class="opt-help"></div></div>`).join("");
    deps.app().innerHTML = `${bar()}
      <div class="practice dsa-setup">
        <section>
          <article class="problem">
            <h2 class="dsa-h">${esc(t.heading)}</h2>
            <p class="dsa-lead">You write every line yourself, from a blank file, following concrete steps. Each step is checked before the next opens.
              At the end you build the whole thing again with no hints. Choose how yours is built:</p>
            <div class="opts">${optHTML}</div>
            <div class="label">${esc(t.diagramLabel)}</div>
            <div class="dg-host"></div>
            <div class="label">The steps</div>
            <ol class="dsa-preview"></ol>
            <div class="actions"><button class="btn primary begin">Begin tutorial →</button>
              <span class="hint">Reset any time to start over with other options.</span></div>
          </article>
        </section>
        <aside class="console">
          <div class="label first">Complexity</div>
          <div class="cx-grid"></div>
          <p class="cx-foot">n = number of elements. Faded boxes aren't functions in this tutorial; they're there for comparison.</p>
        </aside>
      </div>`;
    wireBar();
    let prev = null;
    const update = () => {
      for (const x of t.OPTIONS) {
        const el = $(`.opt[data-opt="${x.id}"]`);
        el.querySelectorAll("button").forEach((b) => { const on = b.dataset.v === o[x.id]; b.classList.toggle("on", on); b.setAttribute("aria-checked", on); });
        el.querySelector(".opt-help").textContent = x.help ? x.help[o[x.id]] : "";
      }
      $(".cx-grid").innerHTML = cxHTML(o, prev);
      $(".dg-host").innerHTML = diagramHTML(o);
      $(".dsa-preview").innerHTML = t.render(o).map((st) => `<li>${esc(st.title)}</li>`).join("");   // (options can change the steps)
      prev = cxValues(o);
      s.opts = { ...o };
      save();
    };
    deps.app().querySelectorAll(".opt").forEach((el) => {
      el.addEventListener("click", (e) => {
        const b = e.target.closest("button[data-v]");
        if (!b) return;
        o[el.dataset.opt] = b.dataset.v;
        update();
      });
    });
    update();
    $(".begin").onclick = () => {
      s.opts = { ...o };
      s.phase = "steps";
      s.passed = 0;
      s.view = 0;
      s.code = "";
      save();
      renderSteps();
    };
  }

  // ------------------------------------------------------------ micro-steps
  const hasText = (m) => !!(m.say || m.note || m.body.some(hasText));

  function microHTML(items, mode) {
    return `<ol class="micro">${items.map((m) => microItem(m, mode)).join("")}</ol>`;
  }
  function microItem(m, mode) {
    const { inline } = deps;
    if (m.group) {
      // a function: its header line, the numbered steps of its body, its closing brace(s)
      const head = m.body.filter((x) => x.header), close = m.body.filter((x) => x.closing);
      const steps = m.body.filter((x) => !x.header && !x.closing);
      const pre = (lines, cls = "") => `<pre class="code-block ms-code ${cls}" data-code="${encodeURIComponent(lines.join("\n"))}"></pre>`;
      return `<li class="ms-group"><div class="ms-say">${inline(m.say)}</div>${head.map((x) => pre(x.code)).join("")}` +
        `${microHTML(steps, mode)}${close.map((x) => pre(x.code, "ms-close")).join("")}</li>`;
    }
    const silent = !m.body.some(hasText);
    const code = silent ? DL.emit([m]) : m.code;
    let h = m.say ? `<div class="ms-say">${inline(m.say)}</div>` : "";
    if (code.length) h += `<pre class="code-block ms-code" data-code="${encodeURIComponent(code.join("\n"))}"></pre>`;
    if (m.note) h += `<div class="ms-note">${inline(m.note)}</div>`;
    if (!silent && m.body.length) h += microHTML(m.body, mode);
    if (!silent && m.close.length) h += `<pre class="code-block ms-code ms-close" data-code="${encodeURIComponent(m.close.join("\n"))}"></pre>`;
    return `<li class="${m.say ? "" : "bare"}${m.closing ? " closing" : ""}">${h}</li>`;
  }
  // syntax-highlight every code fragment in el
  function highlight(el, mode) {
    el.querySelectorAll("pre[data-code]").forEach((pre) => {
      const code = decodeURIComponent(pre.dataset.code);
      const block = deps.codeBlock(code, mode);
      pre.replaceChildren(...block.childNodes);
      pre.classList.add("cm-s-default");
      pre.removeAttribute("data-code");
    });
  }

  // ------------------------------------------------------------ the coach: micro-steps as bubbles that follow the typing
  // The current step's micro-steps become an ordered list of targets (an instruction and the line(s) it asks for).
  // On every edit the file is compared, line by line and in order, with the targets (no compiling): the first target
  // not yet typed is shown in a bubble under the cursor's line. Typing it fades the bubble into the next one; deleting
  // brings earlier ones back.
  function coachTargets(items) {
    const out = [];
    const lines = (code, depth) => code.map((text) => ({ text, depth }));
    const add = (say, ls, extra = {}) => {
      // a target without words (an else branch, a closing brace) belongs with the one before it
      if (!say && !extra.note && out.length && ls.length) { out[out.length - 1].lines.push(...ls); return; }
      out.push({ say, lines: ls, notes: [], ...extra });
    };
    const walk = (list, depth) => {
      for (const m of list) {
        if (m.group) {
          add(m.say, m.body.filter((x) => x.header).flatMap((x) => lines(x.code, depth)), { fn: m.fn });
          walk(m.body.filter((x) => !x.header && !x.closing), depth + 1);
          add("", m.body.filter((x) => x.closing).flatMap((x) => lines(x.code, depth)));
          continue;
        }
        if (!m.body.some(hasText)) {   // (a Java class's closing brace comes after all the functions)
          add(m.say, DL.emit([{ ...m, close: m.deferClose ? [] : m.close }]).map((text) => ({ text, depth })), { note: m.note });
          continue;
        }
        add(m.say, lines(m.code, depth), { note: m.note });
        walk(m.body, depth + 1);
        if (!m.deferClose) add("", lines(m.close, depth));   // (Java's class LinkedList closes after all the functions)
      }
    };
    walk(items, 0);
    // a step with nothing to type (Python and Java free memory themselves) is told along with the next one
    const merged = [];
    let carry = [];
    for (const t of out) {
      if (!t.lines.length) { carry.push([t.say, t.note].filter(Boolean).join(" ")); continue; }
      t.notes = carry;
      carry = [];
      merged.push(t);
    }
    if (carry.length && merged.length) merged[merged.length - 1].after = carry;
    return merged;
  }

  // a line without spaces or comments; Python's quotes made the same
  function normLine(line, lang) {
    const s = lang === "py" ? line.replace(/#.*$/, "").replace(/'/g, '"') : line.replace(/\/\/.*$/, "");
    return s.replace(/\s+/g, "");
  }
  // equivalent spellings: x++ / ++x / x = x + 1 / x += 1, NULL / nullptr, std::ptr:: / ptr::, Python's type hints
  function looseLine(n, lang) {
    let s = n.replace(/;$/, "");
    if (lang === "py") {
      s = s.replace(/->[^:]+:$/, ":").replace(/^(def\w+|class\w+)\[[^\]]*\]/, "$1").replace(/(\w):[^,)=:]+(?=[,)=])/g, "$1")
        .replace(/\(\):$/, ":").replace(/==None/g, "isNone").replace(/!=None/g, "isnotNone");
    }
    s = s.replace(/\bNULL\b/g, "nullptr").replace(/std::ptr::/g, "ptr::");
    let m;
    if ((m = s.match(/^(.+?)\+\+$/)) || (m = s.match(/^\+\+(.+)$/))) s = `${m[1]}+=1`;
    else if ((m = s.match(/^(.+?)--$/)) || (m = s.match(/^--(.+)$/))) s = `${m[1]}-=1`;
    else if ((m = s.match(/^([\w.>()*-]+)=\1([+-])1$/))) s = `${m[1]}${m[2]}=1`;
    return s;
  }

  function coachWant(t, lang) {
    const name = t.fn ? (lang === "java" ? DL.camel(t.fn) : t.fn) : null;
    return t.lines.map(({ text }) => {
      const n = normLine(text, lang);
      if (!n) return null;
      const w = { n, l: looseLine(n, lang) };
      if (name && /^template</.test(n)) w.re = /^template</;
      else if (name) w.re = { py: new RegExp(`^def${name}(\\[.*\\])?\\(`), rs: new RegExp(`^(pub)?fn${name}(<.*>)?\\(`),
        java: new RegExp(`static.*\\b${name}\\(`), cpp: new RegExp(`\\b${name}\\((?!.*;$)`) }[lang];
      return w;
    }).filter(Boolean);
  }

  // -> { k: the first target not typed yet, anchor: the line of the last one typed }
  function coachProgress(targets, code, lang, skipped) {
    const U = [];
    code.split("\n").forEach((line, idx) => { const n = normLine(line, lang); if (n) U.push({ n, l: looseLine(n, lang), idx }); });
    const hit = (u, w) => u.n === w.n || u.l === w.l || (w.re && w.re.test(u.n));
    let j = 0, k = 0, anchor = null;
    for (; k < targets.length; k++) {
      if (skipped.has(k)) continue;
      let jj = j, ok = true, last = null;
      for (const w of targets[k].want) {
        while (jj < U.length && !hit(U[jj], w)) jj++;
        if (jj >= U.length) { ok = false; break; }
        last = U[jj].idx;
        jj++;
      }
      if (!ok) break;
      j = jj;
      if (last !== null) anchor = last;
    }
    if (anchor === null) anchor = U.length ? U[U.length - 1].idx : 0;
    return { k, anchor };
  }

  function startCoach(st, passed) {
    const p = page, cm = p.editor, lang = p.s.opts.lang, mode = DL.LANGS[lang].mode;
    const host = $(".editor");
    const targets = coachTargets(st.items);
    targets.forEach((t) => { t.want = coachWant(t, lang); });
    const skipped = new Set();
    let shown = null, el = null, anchor = 0, timer = null;
    const place = () => {
      if (!el || page !== p) return;
      const line = Math.min(cm.hasFocus() ? cm.getCursor().line : anchor, cm.lastLine());
      const c = cm.charCoords({ line, ch: 0 }, "window");
      el.style.top = `${Math.round(c.bottom - host.getBoundingClientRect().top + 6)}px`;
    };
    const show = (k) => {
      shown = k;
      if (el) { const old = el; old.classList.add("leaving"); setTimeout(() => old.remove(), 240); el = null; }
      if (k >= targets.length && passed) return;    // a step already passed: no reminder needed
      const node = document.createElement("div");
      node.className = "coach entering";
      node.dataset.k = String(k);
      if (k >= targets.length) {
        node.classList.add("done");
        node.innerHTML = `<div class="coach-say">✓ That's the whole step. Press <strong>Check</strong> (⇧↵).</div>`;
      } else {
        const t = targets[k];
        const base = Math.min(...t.lines.map((x) => x.depth));
        const code = t.lines.map((x) => "    ".repeat(x.depth - base) + x.text).join("\n");
        node.innerHTML = `<div class="coach-head"><span class="coach-n">${k + 1} / ${targets.length}</span>` +
          `<button type="button" class="coach-skip" title="I wrote this differently: show the next hint">skip ›</button></div>` +
          t.notes.map((x) => `<div class="coach-note">${deps.inline(x)}</div>`).join("") +
          (t.say ? `<div class="coach-say">${deps.inline(t.say)}</div>` : "") +
          (t.note ? `<div class="coach-note">${deps.inline(t.note)}</div>` : "");
        const pre = deps.codeBlock(code, mode);
        pre.classList.add("coach-code");
        node.appendChild(pre);
        if (t.after) node.insertAdjacentHTML("beforeend", t.after.map((x) => `<div class="coach-note">${deps.inline(x)}</div>`).join(""));
        const skip = node.querySelector(".coach-skip");
        skip.addEventListener("mousedown", (e) => e.preventDefault());   // keep the editor's focus (and the bubble in place)
        skip.onclick = () => { skipped.add(k); update(); cm.focus(); };
      }
      host.appendChild(node);
      el = node;
      place();
      requestAnimationFrame(() => requestAnimationFrame(() => node.classList.remove("entering")));
    };
    const update = () => {
      if (page !== p) return;
      const r = coachProgress(targets, cm.getValue(), lang, skipped);
      anchor = r.anchor;
      if (r.k !== shown) show(r.k);
      else place();
    };
    cm.on("change", () => { clearTimeout(timer); timer = setTimeout(update, 120); });
    cm.on("cursorActivity", place);
    cm.on("focus", place);
    cm.on("blur", place);
    p.onResize = place;
    window.addEventListener("resize", place);
    p.cleanup = () => window.removeEventListener("resize", place);
    update();
  }

  // ------------------------------------------------------------ steps
  function renderSteps() {
    if (page.cleanup) { page.cleanup(); page.cleanup = null; }
    const { t, s } = page;
    const esc = deps.esc;
    const o = s.opts;
    const steps = t.render(o);
    s.view = Math.min(s.view, s.passed, steps.length - 1);
    const st = steps[s.view];
    const mode = DL.LANGS[o.lang].mode;
    const outline = steps.map((x, i) => {
      const state = i < s.passed ? "done" : i === s.passed ? "current" : "locked";
      const tag = i <= s.passed ? "button" : "span";
      return `<li class="${state}${i === s.view ? " viewing" : ""}"><${tag} class="ol-item" data-i="${i}">
        <span class="ol-mark">${state === "done" ? "✓" : i + 1}</span><span>${esc(x.title)}</span></${tag}></li>`;
    }).join("");
    deps.app().innerHTML = `${bar()}
      <div class="practice dsa-steps">
        <section>
          <article class="problem step">
            <div class="step-kicker">Step ${s.view + 1} of ${steps.length}</div>
            <h2 class="dsa-h">${esc(st.title)}</h2>
            <div class="prompt">${deps.md(st.goal)}</div>
            ${st.contractNote ? `<p class="contract-note">${deps.inline(st.contractNote)}</p>` : ""}
            ${s.view === 0 ? `<p class="ms-foot">Type in the editor below: it's your whole file, and each step adds to it. Hints appear beside your code as you type.</p>` : ""}
            <details class="all-micro"><summary>All micro-steps</summary><div class="inner">${microHTML(st.items, mode)}</div></details>
          </article>
          <div class="editor"><textarea></textarea></div>
          <div class="actions">
            <button class="btn run">Run</button>
            <button class="btn primary check">Check step ${s.view + 1}</button>
            <span class="hint">⌘/Ctrl+Enter run · Shift+Enter check</span>
            <button class="btn primary next" hidden></button>
          </div>
        </section>
        <aside class="console side">
          <div class="label first">Steps</div>
          <ol class="outline">${outline}</ol>
          <div class="head"><div class="verdict"></div></div>
          <div class="verdict-msg"></div>
          <pre class="out"><span class="dim">Check runs this step's tests (and the earlier steps' again).</span></pre>
          <details class="cx-details"><summary>Complexity</summary><div class="cx-grid small">${cxHTML(o)}</div></details>
          <p class="runner-note">${esc(runnerNote(o.lang))}</p>
        </aside>
      </div>`;
    wireBar();
    highlight($(".problem"), mode);
    deps.app().querySelectorAll("button.ol-item").forEach((b) => { b.onclick = () => { s.view = Number(b.dataset.i); save(); renderSteps(); }; });
    makeEditor(s.code, (v) => { s.code = v; save(); }, () => doRun(), () => doCheck());
    startCoach(st, s.view < s.passed);
    $(".run").onclick = doRun;
    $(".check").onclick = doCheck;
    if (s.view < s.passed) showNext();
  }

  function showNext() {
    const { t, s } = page;
    const next = $(".next");
    if (s.passed >= t.stepCount(s.opts) && s.view === t.stepCount(s.opts) - 1) {
      next.textContent = "Final challenge →";
      next.onclick = () => { s.phase = "final"; save(); renderFinal(); };
    } else {
      next.textContent = "Next step →";
      next.onclick = () => { s.view += 1; save(); renderSteps(); window.scrollTo(0, 0); };
    }
    next.hidden = false;
  }

  function makeEditor(value, onChange, run, check) {
    const lang = page.s.opts.lang;
    const editor = CodeMirror.fromTextArea($(".editor textarea"), {
      mode: DL.LANGS[lang].mode, lineNumbers: true, indentUnit: 4, matchBrackets: true, autoCloseBrackets: true,
      viewportMargin: Infinity,
      extraKeys: {
        "Cmd-Enter": run, "Ctrl-Enter": run, "Shift-Enter": check,
        Tab: (cm) => (cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection("    ")),
      },
    });
    editor.setValue(value || "");
    editor.on("change", () => onChange(editor.getValue()));
    page.editor = editor;
    return editor;
  }

  // ------------------------------------------------------------ running
  async function execute(lang, user, tests) {
    const prog = RUN.assemble(lang, user, tests);
    if (lang === "py") {
      const py = deps.getPy();
      await py.init([]);
      try { return await py.call("dsaRun", { user: prog.user, tests: prog.tests }, 15000); }
      catch (e) { if (e instanceof deps.TimeLimit) return { compiled: true, stdout: "", stderr: "", timedOut: true }; throw e; }
    }
    return RUN.runCE(lang, prog.source);
  }

  function setBusy(b) {
    page.busy = b;
    deps.app().querySelectorAll(".run, .check, .submit").forEach((x) => { x.disabled = b; });
  }
  function setVerdict(text, cls, msg) {
    $(".verdict").textContent = text;
    $(".verdict").className = `verdict ${cls || ""}`;
    $(".verdict-msg").textContent = msg || "";
  }
  const running = (lang) => (lang === "py" ? "Running…" : "Compiling and running on Compiler Explorer…");

  async function doRun() {
    if (!page || page.busy) return;
    const p = page, lang = p.s.opts.lang;
    const code = p.editor.getValue();
    const esc = deps.esc;
    if (!code.trim()) return;
    if (!RUN.hasMain[lang](code)) {
      setVerdict("", "", "");
      $(".out").innerHTML = `<span class="dim">Nothing to run: your file has no <code>main</code>. Write one to try your code, or press Check to run the tests.</span>`;
      return;
    }
    setBusy(true);
    setVerdict("", "", "");
    $(".out").innerHTML = `<span class="dim">${running(lang)}</span>`;
    let res;
    try { res = await execute(lang, code, null); } catch (e) { res = { error: e.message }; }
    if (page !== p) return;
    setBusy(false);
    if (res.error) { $(".out").innerHTML = `<span class="err">${esc(res.error)}</span>`; return; }
    if (!res.compiled) { setVerdict("Doesn't compile", "bad"); $(".out").innerHTML = `<span class="err">${esc(res.compileOutput)}</span>`; return; }
    const parts = [];
    if (res.stdout) parts.push(esc(res.stdout.replace(/\n$/, "")));
    if (res.stderr) parts.push(`<span class="err">${esc(res.stderr)}</span>`);
    if (res.timedOut) parts.push(`<span class="err">Time limit exceeded: an infinite loop?</span>`);
    $(".out").innerHTML = parts.join("\n") || `<span class="dim">(no output)</span>`;
  }

  // run the tests; -> true if they passed
  async function runTests(tests, label) {
    const p = page, lang = p.s.opts.lang;
    const code = p.editor.getValue();
    const esc = deps.esc;
    if (!code.trim()) { setVerdict("Not yet", "bad", "The editor is empty: write the code first."); return false; }
    setBusy(true);
    setVerdict("", "", "");
    $(".out").innerHTML = `<span class="dim">${running(lang)}</span>`;
    let res;
    try { res = await execute(lang, code, tests); } catch (e) { res = { error: e.message }; }
    if (page !== p) return false;
    setBusy(false);
    if (res.error) {
      setVerdict("Couldn't run", "bad", lang === "py" ? "" : "Compiler Explorer may be busy or unreachable; try again in a moment.");
      $(".out").innerHTML = `<span class="err">${esc(res.error)}</span>`;
      return false;
    }
    if (!res.compiled) {
      setVerdict("Doesn't compile", "bad", "Fix the error and check again.");
      $(".out").innerHTML = `<span class="err">${esc(res.compileOutput)}</span>`;
      return false;
    }
    const cmp = RUN.compare(tests.expected, res.stdout);
    const ok = cmp.ok && !res.timedOut;
    const lines = [];
    const shown = cmp.passed.slice(-8);
    if (cmp.passed.length > shown.length) lines.push(`<span class="dim">… ${cmp.passed.length - shown.length} lines passed</span>`);
    shown.forEach((l) => lines.push(`<span class="good">✓</span> <span class="dim">${esc(l)}</span>`));
    if (!cmp.ok) {
      if (cmp.expected !== null) lines.push(`<span class="err">✗ expected:</span> ${esc(cmp.expected)}`);
      lines.push(cmp.got === null ? `<span class="err">  got:</span> <span class="dim">(nothing: the program stopped here)</span>` : `<span class="err">  got:     </span> ${esc(cmp.got)}`);
    }
    if (res.stderr) lines.push(`<span class="err">${esc(res.stderr.trim())}</span>`);
    if (res.timedOut) lines.push(`<span class="err">Time limit exceeded: an infinite loop?</span>`);
    $(".out").innerHTML = lines.join("\n");
    if (ok) setVerdict(`✓ ${label}`, "good", "");
    else setVerdict("✗ Not yet", "bad", cmp.got !== null && /BROKEN/.test(cmp.got) ? "The list's structure is broken after this operation; the message says which rule." : "The first line that differs is marked.");
    return ok;
  }

  async function doCheck() {
    if (!page || page.busy) return;
    const { t, s } = page;
    const tests = t.harness(s.opts, s.view + 1);
    const ok = await runTests(tests, `Step ${s.view + 1} passed`);
    if (!ok || !page) return;
    if (s.view + 1 > s.passed) {
      s.passed = s.view + 1;
      save();
      const ol = $(`.outline li:nth-child(${s.view + 1})`);
      if (ol) { ol.className = "done viewing"; ol.querySelector(".ol-mark").textContent = "✓"; }
      const nx = $(`.outline li:nth-child(${s.view + 2})`);
      if (nx) {
        nx.className = "current";
        const span = nx.querySelector(".ol-item");
        const b = document.createElement("button");
        b.className = "ol-item"; b.dataset.i = String(s.view + 1); b.innerHTML = span.innerHTML;
        b.onclick = () => { s.view = s.view + 1; save(); renderSteps(); };
        span.replaceWith(b);
      }
      $(".bar .dsa-status").outerHTML = chip(page.id);
    }
    if (s.passed >= t.stepCount(s.opts)) $(".verdict-msg").textContent = "Every step done. One more: build it again from scratch.";
    showNext();
  }

  // ------------------------------------------------------------ the final challenge
  function renderFinal() {
    const { t, s } = page;
    const esc = deps.esc;
    const o = s.opts;
    const done = s.final.passed;
    deps.app().innerHTML = `${bar()}
      <div class="practice dsa-final">
        <section>
          <article class="problem">
            <div class="step-kicker">${done ? "Completed" : "Almost!"}</div>
            <h2 class="dsa-h">Final: build it from scratch</h2>
            <div class="prompt">${deps.md(t.problem(o))}</div>
          </article>
          <div class="editor"><textarea></textarea></div>
          <div class="actions">
            <button class="btn run">Run</button>
            <button class="btn primary submit">Submit</button>
            <span class="hint">⌘/Ctrl+Enter run · Shift+Enter submit</span>
          </div>
        </section>
        <aside class="console side">
          <div class="head"><div class="verdict ${done ? "good" : ""}">${done ? "✓ Completed" : ""}</div></div>
          <div class="verdict-msg">${done ? "You built it from scratch. Reset to do it again, maybe with other options." : ""}</div>
          <pre class="out"><span class="dim">Submit runs every test from the tutorial plus a random sequence of operations.</span></pre>
          <details class="cx-details"><summary>Complexity</summary><div class="cx-grid small">${cxHTML(o)}</div></details>
          <p class="runner-note">${esc(runnerNote(o.lang))}</p>
        </aside>
      </div>`;
    wireBar();
    makeEditor(s.final.code, (v) => { s.final.code = v; save(); }, () => doRun(), () => doSubmit());
    $(".run").onclick = doRun;
    $(".submit").onclick = doSubmit;
  }

  async function doSubmit() {
    if (!page || page.busy) return;
    const { t, s } = page;
    const tests = t.harness(s.opts, t.stepCount(s.opts), { final: true, seed: 1 + Math.floor(Math.random() * 1e9) });
    const ok = await runTests(tests, "Completed");
    if (!ok || !page || s.final.passed) return;
    s.final.passed = true;
    save();
    $(".bar .dsa-status").outerHTML = chip(page.id);
    $(".step-kicker").textContent = "Completed";
    $(".verdict-msg").textContent = "You built it from scratch. Reset to do it again, maybe with other options.";
    const m = document.createElement("div");
    m.className = "modal";
    m.innerHTML = `<div class="card"><div class="done-mark">✓</div><h2>Completed!</h2>
      <p>You built a ${deps.esc(t.describe(s.opts))} from a blank file.</p><button class="btn primary">Nice</button></div>`;
    m.querySelector("button").onclick = () => m.remove();
    document.body.appendChild(m);
  }

  root.DSAPage = { init, homeRows, render, leave, status, LIST, _coach: { coachTargets, coachWant, coachProgress } };
})(window);

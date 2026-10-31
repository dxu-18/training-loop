// Progress files: everything the site keeps in this browser (localStorage), saved to and restored from one file, so
// clearing the browser's data (or moving to another computer) loses nothing. The home page's Download and Upload.
//
//   Download  a box to name the file, then an ordinary download. (Not the File System Access Save dialog: that gives
//             the site edit access to the chosen file, and Chrome warns about it.)
//   Upload    a file picker that accepts .tlsave files. The file is checked, its progress shown next to this
//             browser's, then merged (per module and tutorial, whichever is further along) or used to replace it all.
//
// A .tlsave file is UTF-8 JSON, one store per line:
//   { "format": "training-loop-progress", "version": 1, "savedAt": "<ISO time>", "site": "<page URL>",
//     "stores": { "<localStorage key>": <its value, parsed>, ... } }
// stores holds every training-loop:* key but the retired v1 counter: Part I/II modules (training-loop:v2), Part III
// tutorials (training-loop:dsa:v1), the Bash terminal's font size and its open sessions (training-loop:term:*). Each
// store's key carries that store's own version, so "version" changes only if the envelope itself does.
(function (root) {
  "use strict";
  const FORMAT = "training-loop-progress";
  const VERSION = 1;
  const EXT = ".tlsave";
  const PREFIX = "training-loop:";
  const RETIRED = new Set(["training-loop:v1"]);   // Part I/II's first counter (one per topic): nothing reads it
  const MODULES = "training-loop:v2";
  const DSA = "training-loop:dsa:v1";
  const TERM = "training-loop:term:";              // + "bash/7.02": that module's terminal session (keys typed so far)
  const RESTORED = "training-loop:restored";       // (sessionStorage) the toast to show after the reload
  const MAX_BYTES = 64 << 20;

  const isObj = (x) => x !== null && typeof x === "object" && !Array.isArray(x);
  const count = (x) => Number.isInteger(x) && x >= 0;
  const kept = (k) => k.startsWith(PREFIX) && !RETIRED.has(k);

  // ------------------------------------------------------------ the file
  // this browser's stores: { key: value }
  function collect(storage = root.localStorage) {
    const out = {};
    for (let i = 0; i < storage.length; i++) {
      const k = storage.key(i);
      if (!k || !kept(k)) continue;
      try { out[k] = JSON.parse(storage.getItem(k)); } catch { /* not ours to save */ }
    }
    return out;
  }

  function serialize(stores, savedAt, site) {
    const lines = Object.keys(stores).sort().map((k) => `    ${JSON.stringify(k)}: ${JSON.stringify(stores[k])}`);
    return `{\n  "format": "${FORMAT}",\n  "version": ${VERSION},\n  "savedAt": ${JSON.stringify(savedAt)},\n` +
      `  "site": ${JSON.stringify(site)},\n  "stores": ${lines.length ? `{\n${lines.join(",\n")}\n  }` : "{}"}\n}\n`;
  }

  // a store's value, checked before anything is written: the pages read these without checking again
  function checkStore(k, v) {
    const bad = (what) => { throw new Error(`This progress file is damaged (${what}).`); };
    if (k === MODULES) {
      if (!isObj(v) || !isObj(v.modules)) bad("the module list");
      for (const [id, m] of Object.entries(v.modules)) {
        const cur = m && m.current;
        if (!isObj(m) || !count(m.solved) || typeof m.medal !== "boolean" ||
            (cur != null && !(isObj(cur) && Number.isInteger(cur.n) && cur.n >= 1 && Number.isInteger(cur.seed))) ||
            (m.draft != null && !(isObj(m.draft) && typeof m.draft.code === "string"))) bad(`module ${id}`);
      }
    } else if (k === DSA) {
      if (!isObj(v) || !isObj(v.tutorials)) bad("the tutorial list");
      for (const [id, t] of Object.entries(v.tutorials)) {
        if (!isObj(t) || !["setup", "steps", "final"].includes(t.phase) || !count(t.passed) || !count(t.view) ||
            typeof t.code !== "string" || !isObj(t.final) || typeof t.final.code !== "string" || typeof t.final.passed !== "boolean" ||
            (t.opts != null && !(isObj(t.opts) && Object.values(t.opts).every((x) => typeof x === "string"))) ||
            (t.phase !== "setup" && !isObj(t.opts))) bad(`tutorial ${id}`);
      }
    } else if (k.startsWith(TERM)) {
      if (!isObj(v) || !Number.isInteger(v.seed) || !Array.isArray(v.log)) bad(`terminal session ${k.slice(TERM.length)}`);
    }
  }

  // the file's text -> { savedAt, stores }, or an Error that says what's wrong, for the person to read
  function parse(text) {
    let doc;
    try { doc = JSON.parse(String(text).replace(/^﻿/, "")); }
    catch { throw new Error("This isn't a training-loop progress file: it can't be read as one."); }
    if (!isObj(doc) || doc.format !== FORMAT) throw new Error("This isn't a training-loop progress file.");
    if (!Number.isInteger(doc.version) || doc.version < 1) throw new Error("This progress file is damaged (its version).");
    if (doc.version > VERSION) {
      throw new Error("This file was saved by a newer version of training-loop. Reload the page to update it, then upload the file again.");
    }
    if (!isObj(doc.stores)) throw new Error("This progress file is damaged (it has no stores).");
    const stores = {};
    for (const [k, v] of Object.entries(doc.stores)) {
      if (!kept(k)) continue;   // only this site's own keys are ever written
      checkStore(k, v);
      stores[k] = v;
    }
    return { savedAt: typeof doc.savedAt === "string" ? doc.savedAt : null, stores };
  }

  // ------------------------------------------------------------ what a set of stores amounts to; merging two
  function summary(stores) {
    const mods = Object.values((stores[MODULES] || {}).modules || {});
    const tuts = Object.values((stores[DSA] || {}).tutorials || {});
    return {
      solved: mods.reduce((a, m) => a + m.solved, 0),
      medals: mods.filter((m) => m.medal).length,
      tutorialsDone: tuts.filter((t) => t.final.passed).length,
      tutorialsStarted: tuts.filter((t) => t.phase !== "setup").length,
    };
  }

  const moduleScore = (m) => 2 * m.solved + (m.medal ? 1 : 0);
  const tutorialScore = (t) => (t.final.passed ? 3e6 : t.phase === "final" ? 2e6 : t.phase === "steps" ? 1e6 + t.passed : 0);

  // Per module and per tutorial, whichever record is further along wins whole (a tie keeps this browser's, with its
  // draft). Anything only one side has is kept. A terminal session goes with its module: it is kept only if it
  // belongs to that module's current problem.
  function merge(local, incoming) {
    const out = { ...incoming, ...local };
    const pick = (key, field, score) => {
      const a = (local[key] || {})[field] || {}, b = (incoming[key] || {})[field] || {};
      const both = {};
      for (const id of new Set([...Object.keys(a), ...Object.keys(b)])) {
        both[id] = !a[id] || (b[id] && score(b[id]) > score(a[id])) ? b[id] : a[id];
      }
      if (local[key] || incoming[key]) out[key] = { ...incoming[key], ...local[key], [field]: both };
      return both;
    };
    const modules = pick(MODULES, "modules", moduleScore);
    pick(DSA, "tutorials", tutorialScore);
    for (const k of new Set([...Object.keys(local), ...Object.keys(incoming)])) {
      if (!k.startsWith(TERM)) continue;
      const m = modules[k.slice(TERM.length)];
      const seed = m && m.current ? m.current.seed : undefined;
      const session = [local[k], incoming[k]].find((s) => s && s.seed === seed);
      if (session) out[k] = session; else delete out[k];
    }
    return out;
  }

  // make this browser's storage hold exactly `stores` (the retired key is left alone)
  function apply(stores, storage = root.localStorage) {
    const before = {};
    for (let i = 0; i < storage.length; i++) { const k = storage.key(i); if (k && kept(k)) before[k] = storage.getItem(k); }
    // what goes away goes first (it frees room), then the progress itself; if storage is full even so, everything is
    // put back as it was and the caller says nothing changed
    try {
      for (const k of Object.keys(before)) if (!(k in stores)) storage.removeItem(k);
      for (const k of Object.keys(stores)) if (!k.startsWith(TERM)) storage.setItem(k, JSON.stringify(stores[k]));
    } catch (e) {
      for (const k of Object.keys(stores)) if (!(k in before)) storage.removeItem(k);
      for (const [k, v] of Object.entries(before)) { try { storage.setItem(k, v); } catch { /* best effort */ } }
      throw e;
    }
    let dropped = 0;   // terminal sessions are the expendable part: the problem just starts with a fresh terminal
    for (const k of Object.keys(stores)) {
      if (k.startsWith(TERM)) { try { storage.setItem(k, JSON.stringify(stores[k])); } catch { dropped++; } }
    }
    return { dropped };
  }

  // ------------------------------------------------------------ the page
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ICON_DOWN = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2v8m-3.5-3.5L8 10l3.5-3.5M3 13h10"/></svg>`;
  const ICON_UP = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 10V2M4.5 5.5L8 2l3.5 3.5M3 13h10"/></svg>`;

  function buttons() {
    return `<div class="save-bar">
      <button class="btn small save-dl" title="Save your progress to a ${EXT} file">${ICON_DOWN}Download</button>
      <button class="btn small save-ul" title="Restore progress from a ${EXT} file">${ICON_UP}Upload</button>
      <input type="file" class="save-file" accept="${EXT}" hidden></div>`;
  }

  function wire(host) {
    host.querySelector(".save-dl").onclick = download;
    const input = host.querySelector(".save-file");
    host.querySelector(".save-ul").onclick = () => input.click();
    input.onchange = () => { const f = input.files[0]; input.value = ""; if (f) upload(f); };
  }

  function modal(html, wide) {
    const m = document.createElement("div");
    m.className = "modal";
    m.innerHTML = `<div class="card${wide ? " save-card" : ""}">${html}</div>`;
    const close = () => { m.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    m.addEventListener("click", (e) => { if (e.target === m) close(); });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(m);
    m.close = close;
    return m;
  }
  function notice(title, text) {
    const m = modal(`<h2>${esc(title)}</h2><p>${esc(text)}</p><div class="modal-actions"><button class="btn primary ok">OK</button></div>`);
    m.querySelector(".ok").onclick = m.close;
    m.querySelector(".ok").focus();
  }
  function toast(text) {
    const t = document.createElement("div");
    t.className = "save-toast";
    t.textContent = text;
    document.body.appendChild(t);
    setTimeout(() => t.classList.add("out"), 3500);
    setTimeout(() => t.remove(), 4000);
  }

  const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
  // a name any file system takes: no path separators or reserved characters, no ".tlsave" typed twice
  const cleanName = (s) => s.replace(/[\\/:*?"<>|\x00-\x1f]/g, "-").replace(/\.tlsave$/i, "").trim().replace(/^\.+/, "");

  function download() {
    let stores;
    try { stores = collect(); }
    catch { notice("Can't read your progress", "This browser isn't letting the page use its storage, so there is nothing saved to download."); return; }
    const text = serialize(stores, new Date().toISOString(), location.origin + location.pathname);
    const name = `training-loop-progress-${today()}`;
    askName(name, text);
  }

  // a download goes to the downloads folder without asking (unless the browser is set to ask where to save): so the
  // person names the file first
  function askName(name, text) {
    const m = modal(`<h2>Download progress</h2>
      <p>Name the file. It goes to your browser's downloads folder.</p>
      <label class="save-name"><input type="text" spellcheck="false" autocomplete="off" value="${esc(name)}"><span>${EXT}</span></label>
      <div class="modal-actions"><button class="btn cancel">Cancel</button><button class="btn primary confirm">Download</button></div>`, true);
    const input = m.querySelector("input");
    const go = () => {
      const file = (cleanName(input.value) || name) + EXT;
      const url = URL.createObjectURL(new Blob([text], { type: "application/octet-stream" }));   // (a JSON type can make Safari add .json)
      const a = document.createElement("a");
      a.href = url;
      a.download = file;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      m.close();
      toast(`Downloaded ${file}`);
    };
    m.querySelector(".cancel").onclick = m.close;
    m.querySelector(".confirm").onclick = go;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    input.focus();
    input.select();
  }

  async function upload(file) {
    let parsed, local;
    try {
      if (file.size > MAX_BYTES) throw new Error("This file is too large to be a training-loop progress file.");
      parsed = parse(await file.text());
    } catch (e) { notice("Can't use this file", e.message); return; }
    try { local = collect(); }
    catch { notice("Can't restore here", "This browser isn't letting the page use its storage, so progress can't be saved in it."); return; }
    const a = summary(local), b = summary(parsed.stores);
    const fresh = !Object.keys(local).length;
    const when = parsed.savedAt && !isNaN(Date.parse(parsed.savedAt))
      ? new Date(parsed.savedAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "an unknown date";
    const row = (label, k) => `<tr><td>${label}</td>${fresh ? "" : `<td>${a[k]}</td>`}<td>${b[k]}</td></tr>`;
    const m = modal(`<h2>Restore progress?</h2>
      <p class="save-meta"><strong>${esc(file.name)}</strong>, saved ${esc(when)}</p>
      <table class="save-cmp"><thead><tr><th></th>${fresh ? "" : "<th>This browser</th>"}<th>The file</th></tr></thead><tbody>
        ${row("Problems solved", "solved")}${row("Medals", "medals")}${row("Tutorials completed", "tutorialsDone")}${row("Tutorials started", "tutorialsStarted")}
      </tbody></table>
      ${fresh ? "<p>This browser has no progress yet, so everything comes from the file.</p>"
        : "<p><strong>Merge</strong> keeps whichever is further along in each module and tutorial. <strong>Replace</strong> makes this browser match the file exactly.</p>"}
      <div class="modal-actions"><button class="btn cancel">Cancel</button>
        ${fresh ? '<button class="btn primary merge">Restore</button>'
          : '<button class="btn replace">Replace</button><button class="btn primary merge">Merge</button>'}</div>`, true);
    const go = (mode) => {
      let res;
      try { res = apply(mode === "replace" ? parsed.stores : merge(local, parsed.stores)); }
      catch { m.close(); notice("Couldn't restore", "This browser's storage is full or blocked, so the progress couldn't be saved. Nothing was changed."); return; }
      try { root.sessionStorage.setItem(RESTORED, JSON.stringify({ name: file.name, mode, dropped: res.dropped })); } catch { /* no toast then */ }
      location.reload();
    };
    m.querySelector(".cancel").onclick = m.close;
    m.querySelector(".merge").onclick = () => go(fresh ? "replace" : "merge");
    if (!fresh) m.querySelector(".replace").onclick = () => go("replace");
    m.querySelector(".merge").focus();
  }

  // after the reload that follows a restore: say it worked
  function afterLoad() {
    let r = null;
    try { r = JSON.parse(root.sessionStorage.getItem(RESTORED)); root.sessionStorage.removeItem(RESTORED); } catch { /* none */ }
    if (!r) return;
    toast(`${r.mode === "merge" ? "Merged" : "Restored"} progress from ${r.name}` +
      (r.dropped ? ` (${r.dropped} open terminal session${r.dropped > 1 ? "s" : ""} didn't fit and will start fresh)` : ""));
  }

  const api = { FORMAT, VERSION, EXT, collect, serialize, parse, summary, merge, apply, buttons, wire, afterLoad };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.TLProgress = api;
})(typeof window !== "undefined" ? window : globalThis);

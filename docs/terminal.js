// The Bash topic's terminal: a macOS Terminal.app-style window (Basic profile) around xterm.js, wired to
// bash-lite's live terminal in the Pyodide worker (generator/bashlite/live.py).
//
// The page sends keys; Python does the line editing (readline's emacs/vi modes) and runs the machine on the real
// clock. Each update brings the output since the last one and the "input area" (prompt, line, cursor), which this
// file redraws after the output. nano runs here, on xterm's alternate screen; saves go to Python. Vim runs in Python
// (generator/bashlite/vim): while it is open every key goes to it and it draws its own screen (also the alternate one).
// Every key is logged; the log is kept in localStorage (per module) so a reload rebuilds the session, and the grader
// replays it.
"use strict";

(() => {
  const FONT = '"SF Mono", SFMono-Regular, ui-monospace, Menlo, Monaco, "JetBrains Mono", Consolas, monospace';
  const FONT_KEY = "training-loop:term-font";
  const LOG_PREFIX = "training-loop:term:";
  const LOG_MAX_CHARS = 1_500_000;    // one session's log in localStorage
  const ROWS = 24;
  const MAX_COLS = 80;                // Terminal.app's default window is 80×24

  // Terminal.app's Basic profile, in the light and dark appearance
  const THEMES = {
    light: {
      background: "#ffffff", foreground: "#000000", cursor: "#7f7f7f", cursorAccent: "#ffffff",
      selectionBackground: "#a4c9ff", selectionInactiveBackground: "#dcdcdc",
      black: "#000000", red: "#990000", green: "#00a600", yellow: "#999900", blue: "#0000b2", magenta: "#b200b2",
      cyan: "#00a6b2", white: "#bfbfbf", brightBlack: "#666666", brightRed: "#e50000", brightGreen: "#00d900",
      brightYellow: "#e5e500", brightBlue: "#0000ff", brightMagenta: "#e500e5", brightCyan: "#00e5e5",
      brightWhite: "#e5e5e5",
    },
    dark: {
      background: "#1d1e1d", foreground: "#ffffff", cursor: "#9d9d9d", cursorAccent: "#1d1e1d",
      selectionBackground: "#3f638a", selectionInactiveBackground: "#454545",
      black: "#000000", red: "#c65339", green: "#6ac44b", yellow: "#b8b74a", blue: "#6444ed", magenta: "#d357db",
      cyan: "#69c1cf", white: "#d1d1d1", brightBlack: "#909090", brightRed: "#eb5a3a", brightGreen: "#77ea51",
      brightYellow: "#efef53", brightBlue: "#d09af9", brightMagenta: "#eb5af7", brightCyan: "#78f1f2",
      brightWhite: "#ededed",
    },
  };
  // Windows / Linux browsers keep Ctrl-W, Ctrl-N and Ctrl-T (close tab, new window, new tab): in Vim, Alt + the key
  // sends them instead
  const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
  const VIM_ALIASES = IS_MAC ? {} : { "M-w": "C-w", "M-n": "C-n", "M-t": "C-t" };
  const isDark = () => {
    const forced = document.documentElement.dataset.theme;
    if (forced) return forced === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  };

  const strip = (s) => s.replace(/\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b./g, "");

  // ------------------------------------------------------------ keys: xterm's input -> bash-lite key names
  const CSI = { A: "Up", B: "Down", C: "Right", D: "Left", H: "Home", F: "End", Z: "S-Tab" };
  const TILDE = { 1: "Home", 2: "Insert", 3: "Delete", 4: "End", 5: "PgUp", 6: "PgDn", 7: "Home", 8: "End" };
  function parseInput(data) {
    const ev = [];
    let text = "";
    const flush = () => { if (text) { ev.push(["text", text]); text = ""; } };
    const key = (k) => { flush(); ev.push(["key", k]); };
    for (let i = 0; i < data.length;) {
      const c = data[i];
      if (c === "\x1b") {
        const rest = data.slice(i);
        let m = rest.match(/^\x1b\[(\d+)?(?:;(\d+))?([A-DHFZ])/);
        if (m) {
          const base = CSI[m[3]], mod = Number(m[2] || 1);
          if (base === "Left" || base === "Right") key(mod === 3 || mod === 5 || mod === 9 ? (base === "Left" ? "M-b" : "M-f") : base);
          else if (base !== "S-Tab") key(base);
          i += m[0].length;
          continue;
        }
        m = rest.match(/^\x1b\[(\d+)(?:;(\d+))?~/);
        if (m) {
          const k = TILDE[m[1]];
          if (k === "Delete" && m[2] === "3") key("M-d");
          else if (k && k !== "Insert") key(k);
          i += m[0].length;
          continue;
        }
        m = rest.match(/^\x1bO([A-DHFPQRS])/);
        if (m) {
          if (CSI[m[1]]) key(CSI[m[1]]);
          i += m[0].length;
          continue;
        }
        m = rest.match(/^\x1b\[[0-9;?]*[ -\/]*[@-~]/);   // other sequences (focus, function keys): ignored
        if (m && m[0].length > 2) { i += m[0].length; continue; }
        if (i + 1 < data.length) {                        // Option as Meta: ESC + key
          const d = data[i + 1];
          if (d === "\x7f" || d === "\b") key("M-BSpace");
          else if (d === "\r") key("M-Enter");
          else if (d >= " ") key("M-" + d);
          else key("Esc");
          i += 2;
          continue;
        }
        key("Esc");
        i += 1;
        continue;
      }
      if (c === "\r") key("Enter");
      else if (c === "\n") key("C-j");
      else if (c === "\x7f" || c === "\b") key("BSpace");
      else if (c === "\t") key("Tab");
      else if (c === "\x00") key("C-Space");
      else if (c < " ") {
        const code = c.charCodeAt(0);
        key(code <= 26 ? "C-" + String.fromCharCode(96 + code) : { 28: "C-\\", 29: "C-]", 30: "C-^", 31: "C-_" }[code]);
      } else text += c;
      i += 1;
    }
    flush();
    return ev;
  }

  // where the cursor lands after `text` written from column x0 (wrapping at `cols`)
  function layout(x0, text, cols) {
    let x = x0, dy = 0;
    for (const ch of text) {
      if (ch === "\n") { dy++; x = 0; continue; }
      if (x >= cols) { dy++; x = 0; }
      x++;
    }
    if (x >= cols) { dy++; x = 0; }
    return { dy, x };
  }

  // ------------------------------------------------------------ the window
  class MacTerminal {
    constructor(host, { onInput, onResize, onSubmit }) {
      this.onInput = onInput;
      this.onResize = onResize;
      this.onSubmit = onSubmit;
      this.el = document.createElement("div");
      this.el.className = "mac-term";
      this.el.innerHTML = `
        <div class="mt-bar">
          <div class="mt-lights" aria-hidden="true"><i class="l-close"></i><i class="l-min"></i><i class="l-max"></i></div>
          <div class="mt-title"></div>
        </div>
        <div class="mt-body"><div class="mt-screen"></div></div>
        <div class="mt-tmux" hidden></div>
        <div class="mt-vim" hidden><span class="mv-keys"></span><span class="mv-match"></span><span class="mv-hint"></span></div>`;
      host.appendChild(this.el);
      this.titleEl = this.el.querySelector(".mt-title");
      this.tmuxEl = this.el.querySelector(".mt-tmux");
      this.vimEl = this.el.querySelector(".mt-vim");
      let size = 13;
      try { size = Number(localStorage.getItem(FONT_KEY)) || 13; } catch { /* storage blocked */ }
      this.term = new window.Terminal({
        fontFamily: FONT, fontSize: size, lineHeight: 1.0, letterSpacing: 0,
        cursorBlink: false, cursorStyle: "block", cursorInactiveStyle: "outline",
        macOptionIsMeta: true, macOptionClickForcesSelection: true, rightClickSelectsWord: true,
        scrollback: 10000, convertEol: true, drawBoldTextInBrightColors: true, minimumContrastRatio: 1,
        rows: ROWS, cols: MAX_COLS, theme: isDark() ? THEMES.dark : THEMES.light, allowProposedApi: false,
      });
      this.fit = new window.FitAddon.FitAddon();
      this.term.loadAddon(this.fit);
      this.term.open(this.el.querySelector(".mt-screen"));
      this.origin = { y: 0, x: 0 };   // where the input area starts (absolute buffer line, column)
      this.drawn = false;
      this.view = null;
      this.chain = Promise.resolve();
      this.sink = null;               // nano takes the keys while it is open
      this.term.onData((d) => {
        let ev = parseInput(d);
        if (!ev.length) return;
        if (this.view && this.view.mode === "vim") ev = ev.map(([k, v]) => (k === "key" && VIM_ALIASES[v] ? [k, VIM_ALIASES[v]] : [k, v]));
        if (this.sink) this.sink(ev); else this.onInput(ev);
      });
      this.term.attachCustomKeyEventHandler((e) => this.shortcut(e));
      this.el.addEventListener("mousedown", () => setTimeout(() => this.term.focus(), 0));
      this.mq = window.matchMedia("(prefers-color-scheme: dark)");
      this.themeListener = () => { this.term.options.theme = isDark() ? THEMES.dark : THEMES.light; this.el.classList.toggle("dark", isDark()); };
      this.mq.addEventListener("change", this.themeListener);
      this.el.classList.toggle("dark", isDark());
      this.ro = new ResizeObserver(() => this.refit());
      this.ro.observe(this.el.parentElement);
      this.refit(true);
    }

    destroy() {
      this.ro.disconnect();
      this.mq.removeEventListener("change", this.themeListener);
      this.term.dispose();
      this.el.remove();
    }

    focus() { this.term.focus(); }

    // Terminal.app's ⌘ shortcuts (⌘K clear, ⌘+/− font size, ⌘. interrupt); other ⌘ keys stay the browser's
    shortcut(e) {
      if (e.type !== "keydown" || !e.metaKey) return true;
      const k = e.key.toLowerCase();
      if (k === "k") { e.preventDefault(); this.clearAll(); return false; }
      if (k === "=" || k === "+") { e.preventDefault(); this.zoom(1); return false; }
      if (k === "-") { e.preventDefault(); this.zoom(-1); return false; }
      if (k === "0") { e.preventDefault(); this.zoom(0); return false; }
      if (k === ".") { e.preventDefault(); (this.sink || this.onInput)([["key", "C-c"]]); return false; }
      if (k === "c" && this.term.hasSelection()) {
        e.preventDefault();
        navigator.clipboard?.writeText(this.term.getSelection()).catch(() => {});
        return false;
      }
      if (k === "a") { e.preventDefault(); this.term.selectAll(); return false; }
      if (k === "enter" && this.onSubmit) { e.preventDefault(); this.onSubmit(); return false; }
      return false;    // ⌘V pastes (the browser's paste event), ⌘R reloads ...: never typed into the terminal
    }

    zoom(d) {
      const t = this.term;
      const size = d === 0 ? 13 : Math.max(9, Math.min(24, t.options.fontSize + d));
      this.run((done) => {
        this.erase(() => {
          t.options.fontSize = size;
          try { localStorage.setItem(FONT_KEY, String(size)); } catch { /* storage blocked */ }
          this.refit(true, done);
        });
      });
    }

    // the window is 80 columns wide when there is room, narrower on small screens; always 24 rows
    refit(force, done) {
      const t = this.term;
      const screen = this.el.querySelector(".xterm-screen");
      const cell = screen && t.cols ? screen.getBoundingClientRect().width / t.cols : 0;
      if (cell) this.el.style.maxWidth = `${Math.ceil(cell * MAX_COLS + 28)}px`;
      const dims = this.fit.proposeDimensions();
      const cols = dims && dims.cols ? Math.max(20, Math.min(MAX_COLS, dims.cols)) : MAX_COLS;
      if (cols === t.cols && t.rows === ROWS && !force) { if (done) done(); return; }
      const apply = (cb) => {
        t.resize(cols, ROWS);
        this.origin = { y: t.buffer.active.baseY + t.buffer.active.cursorY, x: t.buffer.active.cursorX };
        if (this.onResize) this.onResize(cols);
        this.setTitle(this.title);
        this.drawInput(this.view, cb);
      };
      if (done) apply(done); else this.run((d2) => this.erase(() => apply(d2)));
    }

    setTitle(title) {
      this.title = title || this.title || "";
      // the size in the title follows the window, as in Terminal.app
      this.titleEl.textContent = this.title.replace(/\d+×\d+$/, `${this.term.cols}×${this.term.rows}`);
    }

    // one screen change at a time: xterm parses writes asynchronously
    run(fn) { this.chain = this.chain.then(() => new Promise((done) => fn(done))).catch(() => {}); return this.chain; }

    moveTo(y, x) {
      const b = this.term.buffer.active;
      const vy = Math.max(0, y - b.baseY);
      return `\x1b[${vy + 1};${x + 1}H`;
    }

    // remove the drawn input area (output then continues where it began)
    erase(cb) {
      if (!this.drawn) { cb(); return; }
      this.drawn = false;
      this.term.write(this.moveTo(this.origin.y, this.origin.x) + "\x1b[J", cb);
    }

    // output, then the input area again
    update(out, view, title) {
      if (title) this.setTitle(title);
      if (view && view.tmux) { this.tmuxEl.hidden = false; this.tmuxEl.textContent = view.tmux + (view.tmux_prompt ? "    " + view.tmux_prompt : ""); }
      else this.tmuxEl.hidden = true;
      this.vimBar(view);
      return this.run((done) => {
        this.erase(() => {
          const finish = () => {
            const b = this.term.buffer.active;
            this.origin = { y: b.baseY + b.cursorY, x: b.cursorX };
            this.view = view;
            this.drawInput(view, done);
          };
          if (out) this.term.write(out, finish); else finish();
        });
      });
    }

    // under the window while Vim is open: keys pressed (and the par), whether the text already matches the target
    vimBar(view) {
      if (view && view.sql) { this.sqlBar(view.sql); return; }
      const info = view && view.mode === "vim" ? view.vim : null;
      this.vimEl.hidden = !info;
      if (!info) return;
      const over = info.par && info.keys > info.par;
      const k = this.vimEl.querySelector(".mv-keys");
      k.textContent = `${info.keys} key${info.keys === 1 ? "" : "s"}` + (info.par ? ` · par ${info.par}` : "");
      k.classList.toggle("over", !!over);
      const m = this.vimEl.querySelector(".mv-match");
      m.textContent = info.match == null ? "" : info.match ? "✓ matches the target" : "not there yet";
      m.className = "mv-match" + (info.match ? " ok" : "");
      this.vimEl.querySelector(".mv-hint").textContent = IS_MAC ? "" : "Alt-W / Alt-N / Alt-T send Ctrl-W / Ctrl-N / Ctrl-T";
    }

    // under the window on SQL pages: the last result's shape and whether it already matches the target
    sqlBar(info) {
      this.vimEl.hidden = !info.last;
      if (!info.last) return;
      const k = this.vimEl.querySelector(".mv-keys");
      k.textContent = info.last === "error" ? "last statement: error" : `last result: ${info.last}`;
      k.classList.toggle("over", info.last === "error");
      const m = this.vimEl.querySelector(".mv-match");
      m.textContent = info.match == null ? "" : info.match ? "✓ matches the target" : "not the target yet";
      m.className = "mv-match" + (info.match ? " ok" : "");
      this.vimEl.querySelector(".mv-hint").textContent = "";
    }

    drawInput(view, done) {
      const t = this.term;
      if (!view || view.mode === "ended" || view.mode === "editor" || view.mode === "vim") { done(); return; }
      const prompt = view.prompt || "";
      const buf = view.hidden ? "" : (view.buf || "");
      t.write(prompt + buf, () => {
        const b = t.buffer.active;
        const at = layout(this.origin.x, strip(prompt) + (view.hidden ? "" : buf.slice(0, view.pos)), t.cols);
        let y = this.origin.y + at.dy;
        let s = "";
        const bottom = b.baseY + t.rows - 1;
        if (y > bottom) s += "\r\n".repeat(y - bottom);   // the cursor sits on a line the text hasn't made yet
        t.write(s, () => {
          t.write(this.moveTo(y, at.x), () => { this.drawn = true; done(); });
        });
      });
    }

    // ⌘K: clear the screen and the scrollback, keep the line being typed
    clearAll() {
      this.run((done) => {
        this.erase(() => {
          this.term.write("\x1b[H\x1b[2J", () => {
            this.term.clear();
            const b = this.term.buffer.active;
            this.origin = { y: b.baseY + b.cursorY, x: 0 };
            this.drawInput(this.view, done);
          });
        });
      });
    }

    // full reset (a new problem, Reset)
    reset() {
      return this.run((done) => {
        this.drawn = false;
        this.term.reset();
        this.origin = { y: 0, x: 0 };
        done();
      });
    }

    write(s) { return this.run((done) => this.term.write(s, done)); }
  }

  // ------------------------------------------------------------ nano (GNU nano 7.2, as on Ubuntu 24.04)
  class Nano {
    constructor(mt, req, { save, close }) {
      this.mt = mt;
      this.t = mt.term;
      this.name = req.shown;
      this.saveFn = save;
      this.closeFn = close;
      let text = req.content || "";
      if (text.endsWith("\n")) text = text.slice(0, -1);
      this.lines = text.length || req.content ? text.split("\n") : [""];
      this.cy = 0; this.cx = 0; this.top = 0;
      this.modified = false;
      this.cut = []; this.lastCut = false;
      this.prompt = null;   // {kind, label, text, then}
      const n = this.lines.length;
      this.status = req.exists ? `[ Read ${n} line${n === 1 ? "" : "s"} ]` : "[ New File ]";
      this.busy = false;
    }

    open(done) {   // runs in the window's queue, after the output that came before nano
      this.mt.sink = (ev) => this.keys(ev);
      this.t.write("\x1b[?1049h\x1b[H\x1b[2J", () => { this.render(); done(); });
    }

    rows() { return this.t.rows; }
    cols() { return this.t.cols; }
    editRows() { return Math.max(1, this.rows() - 4); }

    disp(line) {   // tabs shown as spaces to the next multiple of 8
      let out = "";
      for (const ch of line) out += ch === "\t" ? " ".repeat(8 - (out.length % 8)) : ch;
      return out;
    }
    dispCol(line, cx) { return this.disp(line.slice(0, cx)).length; }

    render() {
      const cols = this.cols(), rows = this.rows(), er = this.editRows();
      if (this.cy < this.top) this.top = this.cy;
      if (this.cy >= this.top + er) this.top = this.cy - er + 1;
      const inv = (s) => `\x1b[7m${s}\x1b[27m`;
      const pad = (s, w) => (s.length >= w ? s.slice(0, w) : s + " ".repeat(w - s.length));
      let out = "\x1b[?25l\x1b[H";
      // title bar: version left, file name centred, "Modified" right
      const left = "  GNU nano 7.2", right = this.modified ? "Modified  " : "";
      const name = this.name || "New Buffer";
      const mid = Math.max(left.length + 1, Math.floor((cols - name.length) / 2));
      let bar = pad(left, mid) + name;
      bar = pad(bar, cols - right.length) + right;
      out += inv(pad(bar, cols)) + "\r\n";
      for (let r = 0; r < er; r++) {
        const i = this.top + r;
        let shown = "";
        if (i < this.lines.length) {
          const full = this.disp(this.lines[i]);
          if (i === this.cy) {
            const c = this.dispCol(this.lines[i], this.cx);
            const page = Math.max(0, Math.floor(c / (cols - 8)) * (cols - 8));   // long lines scroll in pages
            shown = (page ? "$" : "") + full.slice(page ? page + 1 : 0, page + cols);
            if (full.length > page + cols) shown = shown.slice(0, cols - 1) + "$";
          } else {
            shown = full.length > cols ? full.slice(0, cols - 1) + "$" : full;
          }
        }
        out += "\x1b[2K" + shown + "\r\n";
      }
      // status line
      out += "\x1b[2K";
      if (this.prompt) {
        out += inv(pad(this.prompt.label + this.prompt.text, cols));
      } else if (this.status) {
        const s = this.status;
        const x = Math.max(0, Math.floor((cols - s.length) / 2));
        out += " ".repeat(x) + inv(s);
      }
      out += "\r\n";
      const help = this.prompt ? this.promptHelp() : [
        [["^G", "Help"], ["^O", "Write Out"], ["^W", "Where Is"], ["^K", "Cut"], ["^T", "Execute"], ["^C", "Location"]],
        [["^X", "Exit"], ["^R", "Read File"], ["^\\", "Replace"], ["^U", "Paste"], ["^J", "Justify"], ["^/", "Go To Line"]],
      ];
      const w = Math.floor(cols / 6);
      help.forEach((row, k) => {
        let line = "\x1b[2K";
        for (const [key, label] of row) line += inv(key) + " " + pad(label, Math.max(0, w - key.length - 1));
        out += line + (k === 0 ? "\r\n" : "");
      });
      // the cursor
      let cy, cx;
      if (this.prompt) { cy = rows - 3; cx = Math.min(cols - 1, this.prompt.label.length + this.prompt.text.length); }
      else {
        const c = this.dispCol(this.lines[this.cy], this.cx);
        const page = Math.max(0, Math.floor(c / (cols - 8)) * (cols - 8));
        cy = 1 + this.cy - this.top;
        cx = c - page + (page ? 1 : 0);
      }
      out += `\x1b[${cy + 1};${cx + 1}H\x1b[?25h`;
      this.t.write(out);
    }

    promptHelp() {
      if (this.prompt.kind === "confirm") return [[[" Y", "Yes"]], [[" N", "No"], ["^C", "Cancel"]]];
      return [[["^G", "Help"]], [["^C", "Cancel"]]];
    }

    say(msg) { this.status = msg; }

    keys(events) {
      if (this.busy) return;
      for (const [kind, val] of events) {
        if (this.closed) return;
        if (this.prompt) this.promptKey(kind, val);
        else this.editKey(kind, val);
      }
      if (!this.closed) this.render();
    }

    insert(s) {
      const line = this.lines[this.cy];
      this.lines[this.cy] = line.slice(0, this.cx) + s + line.slice(this.cx);
      this.cx += s.length;
      this.modified = true;
    }

    editKey(kind, val) {
      const L = this.lines;
      const wasCut = this.lastCut;
      this.lastCut = false;
      if (kind === "text") { this.insert(val); this.status = ""; return; }
      const er = this.editRows();
      switch (val) {
        case "Enter": case "C-m": case "C-j": {
          const line = L[this.cy];
          L.splice(this.cy, 1, line.slice(0, this.cx), line.slice(this.cx));
          this.cy++; this.cx = 0; this.modified = true; break;
        }
        case "Tab": this.insert("\t"); break;
        case "Space": this.insert(" "); break;
        case "BSpace": case "C-h":
          if (this.cx > 0) { L[this.cy] = L[this.cy].slice(0, this.cx - 1) + L[this.cy].slice(this.cx); this.cx--; this.modified = true; }
          else if (this.cy > 0) { this.cx = L[this.cy - 1].length; L[this.cy - 1] += L[this.cy]; L.splice(this.cy, 1); this.cy--; this.modified = true; }
          break;
        case "Delete": case "C-d":
          if (this.cx < L[this.cy].length) { L[this.cy] = L[this.cy].slice(0, this.cx) + L[this.cy].slice(this.cx + 1); this.modified = true; }
          else if (this.cy < L.length - 1) { L[this.cy] += L[this.cy + 1]; L.splice(this.cy + 1, 1); this.modified = true; }
          break;
        case "Left": case "C-b":
          if (this.cx > 0) this.cx--; else if (this.cy > 0) { this.cy--; this.cx = L[this.cy].length; } break;
        case "Right": case "C-f":
          if (this.cx < L[this.cy].length) this.cx++; else if (this.cy < L.length - 1) { this.cy++; this.cx = 0; } break;
        case "Up": case "C-p": if (this.cy > 0) { this.cy--; this.cx = Math.min(this.cx, L[this.cy].length); } break;
        case "Down": case "C-n": if (this.cy < L.length - 1) { this.cy++; this.cx = Math.min(this.cx, L[this.cy].length); } break;
        case "Home": case "C-a": this.cx = 0; break;
        case "End": case "C-e": this.cx = L[this.cy].length; break;
        case "PgUp": case "C-y": this.cy = Math.max(0, this.cy - er); this.cx = Math.min(this.cx, L[this.cy].length); break;
        case "PgDn": case "C-v": this.cy = Math.min(L.length - 1, this.cy + er); this.cx = Math.min(this.cx, L[this.cy].length); break;
        case "M-\\": this.cy = 0; this.cx = 0; break;
        case "M-/": this.cy = L.length - 1; this.cx = L[this.cy].length; break;
        case "C-k": {     // cut the line; consecutive cuts collect
          if (!wasCut) this.cut = [];
          this.cut.push(L[this.cy]);
          if (L.length > 1) { L.splice(this.cy, 1); if (this.cy >= L.length) this.cy = L.length - 1; }
          else L[0] = "";
          this.cx = 0; this.modified = true; this.lastCut = true; this.status = ""; break;
        }
        case "C-u":
          if (this.cut.length) {
            L.splice(this.cy, 0, ...this.cut);
            this.cy += this.cut.length; this.cx = 0; this.modified = true;
          }
          break;
        case "C-o": this.ask("name", "File Name to Write: ", this.name, (name) => this.write(name, false)); break;
        case "C-s": this.write(this.name, false); break;
        case "C-x":
          if (!this.modified) { this.exit(); break; }
          this.ask("confirm", "Save modified buffer?  ", "", (yes) => {
            if (yes) this.ask("name", "File Name to Write: ", this.name, (name) => this.write(name, true));
            else this.exit();
          });
          break;
        case "C-w": this.ask("search", "Search: ", "", (q) => this.find(q)); break;
        case "C-c": {
          const chars = L.reduce((n, l) => n + l.length + 1, 0);
          const at = L.slice(0, this.cy).reduce((n, l) => n + l.length + 1, 0) + this.cx;
          const pct = (a, b) => Math.round((100 * a) / Math.max(1, b));
          this.say(`[ line ${this.cy + 1}/${L.length} (${pct(this.cy + 1, L.length)}%), col ${this.cx + 1}/${L[this.cy].length + 1} ` +
                   `(${pct(this.cx + 1, L[this.cy].length + 1)}%), char ${at}/${chars} (${pct(at, chars)}%) ]`);
          break;
        }
        case "C-g": this.say("[ ^O saves, ^X exits, ^K cuts a line, ^U pastes it, ^W searches ]"); break;
        case "C-t": case "C-r": case "C-\\": case "C-_": case "C-/":
          this.say("[ That nano command isn't available in this terminal ]"); break;
        default: break;
      }
    }

    ask(kind, label, text, then) { this.prompt = { kind, label, text, then }; }

    promptKey(kind, val) {
      const p = this.prompt;
      if (p.kind === "confirm") {
        const ch = kind === "text" ? val[0].toLowerCase() : "";
        if (ch === "y" || ch === "n") { this.prompt = null; p.then(ch === "y"); }
        else if (val === "C-c") { this.prompt = null; this.say("[ Cancelled ]"); }
        return;
      }
      if (kind === "text") { p.text += val; return; }
      if (val === "Space") p.text += " ";
      else if (val === "BSpace" || val === "C-h") p.text = p.text.slice(0, -1);
      else if (val === "C-u") p.text = "";
      else if (val === "C-c" || val === "Esc") { this.prompt = null; this.say("[ Cancelled ]"); }
      else if (val === "Enter" || val === "C-m") {
        this.prompt = null;
        if (p.kind === "name" && !p.text.trim()) { this.say("[ Cancelled ]"); return; }
        p.then(p.text);
      }
    }

    find(q) {
      if (!q) { this.say("[ Cancelled ]"); return; }
      const L = this.lines;
      for (let k = 0; k <= L.length; k++) {
        const i = (this.cy + k) % L.length;
        const from = k === 0 ? this.cx + 1 : 0;
        const j = L[i].indexOf(q, from);
        if (j >= 0) { this.cy = i; this.cx = j; this.say(k === 0 && j < this.cx ? "[ This is the only occurrence ]" : ""); return; }
      }
      this.say(`[ "${q}" not found ]`);
    }

    write(name, thenExit) {
      const content = this.lines.join("\n") + "\n";
      if (content.length > 256 * 1024) { this.say("[ File too big for this terminal (256 KB) ]"); this.render(); return; }
      this.busy = true;
      this.saveFn(content, name !== this.name ? name : null).then((err) => {
        this.busy = false;
        if (err) { this.say(`[ Error writing ${name}: ${err} ]`); this.render(); return; }
        this.name = name;
        this.modified = false;
        const n = this.lines.length;
        this.say(`[ Wrote ${n} line${n === 1 ? "" : "s"} ]`);
        if (thenExit) this.exit(); else this.render();
      });
    }

    exit() {
      this.closed = true;
      this.mt.sink = null;
      this.t.write("\x1b[?1049l\x1b[?25h", () => this.closeFn());
    }
  }

  // ------------------------------------------------------------ the Files panel
  const ICON = {
    d: `<svg viewBox="0 0 16 16" class="ic dir"><path d="M1.5 3.5A1.5 1.5 0 0 1 3 2h3.2l1.5 1.5H13A1.5 1.5 0 0 1 14.5 5v7A1.5 1.5 0 0 1 13 13.5H3A1.5 1.5 0 0 1 1.5 12z"/></svg>`,
    f: `<svg viewBox="0 0 16 16" class="ic file"><path d="M4 1.5h5.2L12.5 4.8V14a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V2a.5.5 0 0 1 .5-.5z"/><path d="M9 1.7V5h3.3" class="fold"/></svg>`,
    x: `<svg viewBox="0 0 16 16" class="ic exe"><path d="M4 1.5h5.2L12.5 4.8V14a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V2a.5.5 0 0 1 .5-.5z"/><path d="M5.5 9.5l1.6-1.5-1.6-1.5M8 10.5h2.5" class="mark"/></svg>`,
    l: `<svg viewBox="0 0 16 16" class="ic link"><path d="M4 1.5h5.2L12.5 4.8V14a.5.5 0 0 1-.5.5H4a.5.5 0 0 1-.5-.5V2a.5.5 0 0 1 .5-.5z"/><path d="M5.5 11.5l4-4M7 7.5h2.5V10" class="mark"/></svg>`,
  };
  const escH = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ------------------------------------------------------------ the Database panel (SQL pages)
  // The open database's tables and views: row counts, columns (click to unfold), indexes; tables the session
  // changed or created are green, dropped ones struck through.
  class DbPanel {
    constructor(el) {
      this.el = el;
      el.innerHTML = `<div class="fp-head"><span class="fp-title">Database</span><span class="db-name"></span></div>
        <div class="fp-tree db-tree"></div>`;
      this.treeEl = el.querySelector(".db-tree");
      this.open = new Set();
      this.start = new Map();
      this.data = null;
      this.treeEl.addEventListener("click", (e) => {
        const row = e.target.closest(".db-rel");
        if (!row) return;
        const n = row.dataset.name;
        if (this.open.has(n)) this.open.delete(n); else this.open.add(n);
        this.render();
      });
    }
    setStart(db) { this.start = new Map((db?.rels || []).map((r) => [r.name, r.sig])); this.update(db); }
    update(db) { if (db) { this.data = db; this.render(); } }
    render() {
      const db = this.data;
      if (!db) return;
      this.el.querySelector(".db-name").textContent = db.db || "";
      const rows = [];
      const now = new Set();
      for (const r of db.rels) {
        now.add(r.name);
        const st = !this.start.has(r.name) ? " is-new" : this.start.get(r.name) !== r.sig ? " is-changed" : "";
        const open = this.open.has(r.name);
        const count = r.rows == null ? r.kind : `${r.rows} row${r.rows === 1 ? "" : "s"}`;
        rows.push(`<div class="fp-row d db-rel${st}" data-name="${escH(r.name)}" style="--depth:0">
          <span class="fp-twisty">${open ? "▾" : "▸"}</span><span class="fp-name">${escH(r.name)}</span>
          <span class="fp-tag">${escH(count)}</span></div>`);
        if (open) {
          for (const [c, t] of r.cols) rows.push(`<div class="fp-row db-col" style="--depth:1"><span class="fp-name">${escH(c)}</span><span class="fp-to">${escH(t)}</span></div>`);
          for (const i of r.idx || []) rows.push(`<div class="fp-row db-col" style="--depth:1"><span class="fp-to">index ${escH(i)}</span></div>`);
        }
      }
      for (const name of this.start.keys()) if (!now.has(name)) rows.push(`<div class="fp-row is-gone" style="--depth:0"><span class="fp-twisty"></span><span class="fp-name">${escH(name)}</span></div>`);
      this.treeEl.innerHTML = rows.join("") || `<div class="fp-row more">no tables</div>`;
    }
  }

  class FilesPanel {
    constructor(el) {
      this.el = el;
      el.innerHTML = `<div class="fp-head"><span class="fp-title">Files</span>
          <label class="fp-hidden"><input type="checkbox"> hidden files</label></div>
        <div class="fp-tree" role="tree"></div><div class="fp-foot"></div>`;
      this.treeEl = el.querySelector(".fp-tree");
      this.footEl = el.querySelector(".fp-foot");
      this.check = el.querySelector("input");
      this.collapsed = new Set();
      this.check.onchange = () => this.render();
      this.treeEl.addEventListener("click", (e) => {
        const row = e.target.closest(".fp-row.d");
        if (!row) return;
        const p = row.dataset.path;
        this.collapsed.has(p) ? this.collapsed.delete(p) : this.collapsed.add(p);
        this.render();
      });
      this.tree = null;
      this.start = null;
    }

    static paths(tree) {
      const out = new Map();
      const walk = (n, path) => {
        out.set(path, n.t);
        for (const c of n.c || []) walk(c, `${path}/${c.n}`);
      };
      if (tree) walk(tree.root, tree.base);
      return out;
    }

    setStart(tree) {
      this.start = FilesPanel.paths(tree);
      this.collapsed = new Set();
      // dot-directories start folded (.cache, .local ...)
      for (const [p, t] of this.start) if (t === "d" && /\/\.[^/]+$/.test(p)) this.collapsed.add(p);
    }

    update(tree) { if (tree) { this.tree = tree; this.render(); } }

    render() {
      const tree = this.tree;
      if (!tree) return;
      const showHidden = this.check.checked;
      const now = FilesPanel.paths(tree);
      const cwd = tree.cwd;
      const rows = [];
      let nd = 0, nf = 0;
      const gone = this.start ? [...this.start.keys()].filter((p) => !now.has(p)) : [];
      const goneUnder = (dir) => gone.filter((p) => p.slice(0, p.lastIndexOf("/")) === dir && now.has(dir));
      const walk = (n, path, depth) => {
        const hidden = n.n.startsWith(".") && depth > 0;
        if (hidden && !showHidden && !(cwd === path || cwd.startsWith(path + "/"))) return;
        const isNew = this.start && !this.start.has(path);
        const here = n.t === "d" && cwd === path;
        const open = n.t === "d" && !this.collapsed.has(path);
        if (depth > 0) { if (n.t === "d") nd++; else nf++; }
        const name = n.t === "d" ? `${escH(n.n)}${depth ? "/" : ""}` : escH(n.n);
        const extra = n.t === "l" ? ` <span class="fp-to">→ ${escH(n.to)}</span>` : "";
        const tag = here ? `<span class="fp-tag here">you are here</span>` : isNew ? `<span class="fp-tag new">new</span>` : "";
        rows.push(`<div class="fp-row ${n.t}${isNew ? " is-new" : ""}${here ? " is-here" : ""}${hidden ? " is-hidden" : ""}"
          role="treeitem" data-path="${escH(path)}" style="--depth:${depth}">
          <span class="fp-twisty">${n.t === "d" ? (open ? "▾" : "▸") : ""}</span>${ICON[n.t] || ICON.f}
          <span class="fp-name">${name}</span>${extra}${tag}</div>`);
        if (n.t !== "d" || !open) return;
        const kids = [...(n.c || [])];
        for (const g of goneUnder(path)) kids.push({ n: g.slice(g.lastIndexOf("/") + 1), t: this.start.get(g), gone: true });
        kids.sort((a, b) => a.n.localeCompare(b.n));
        for (const c of kids) {
          if (c.gone) {
            if (c.n.startsWith(".") && !showHidden) continue;
            rows.push(`<div class="fp-row ${c.t} is-gone" style="--depth:${depth + 1}"><span class="fp-twisty"></span>${ICON[c.t] || ICON.f}
              <span class="fp-name">${escH(c.n)}${c.t === "d" ? "/" : ""}</span><span class="fp-tag gone">deleted</span></div>`);
          } else walk(c, `${path}/${c.n}`, depth + 1);
        }
        if (n.more) rows.push(`<div class="fp-row more" style="--depth:${depth + 1}">…</div>`);
      };
      walk(tree.root, tree.base, 0);
      this.treeEl.innerHTML = rows.join("");
      this.footEl.textContent = `${nd} director${nd === 1 ? "y" : "ies"}, ${nf} file${nf === 1 ? "" : "s"}` +
        (cwd && !(cwd === tree.base || cwd.startsWith(tree.base + "/")) ? ` · you are in ${cwd}` : "");
    }
  }

  // ------------------------------------------------------------ the Git and GitHub panels (git problems)
  const LANE_COLORS = ["#3b82f6", "#e8590c", "#2b8a3e", "#ae3ec9", "#d6336c", "#0c8599", "#f08c00", "#5c7cfa"];
  const STATUS_WORD = { M: "modified", A: "new file", D: "deleted", R: "renamed", T: "type changed", U: "conflict" };
  const ago = (t) => {
    const s = Math.max(0, Date.now() / 1000 - t);
    return s < 3600 ? `${Math.round(s / 60)}m` : s < 86400 ? `${Math.round(s / 3600)}h` :
      s < 86400 * 60 ? `${Math.round(s / 86400)}d` : s < 86400 * 730 ? `${Math.round(s / 86400 / 30)}mo` : `${Math.round(s / 86400 / 365)}y`;
  };

  // lanes for a newest-first commit list: each commit gets a column; each parent edge a path
  function graphLayout(commits) {
    const row = new Map(commits.map((c, i) => [c.id, i]));
    const lanes = [];
    const pos = [];
    const edges = [];
    let width = 1;
    commits.forEach((c, i) => {
      let col = lanes.indexOf(c.id);
      if (col < 0) { col = lanes.indexOf(null); if (col < 0) col = lanes.length; }
      for (let j = 0; j < lanes.length; j++) if (j !== col && lanes[j] === c.id) lanes[j] = null;   // lines join here
      lanes[col] = null;
      pos[i] = col;
      c.p.forEach((p, k) => {
        if (!row.has(p)) return;
        let lane = lanes.indexOf(p);
        if (lane < 0) {
          lane = k === 0 && lanes[col] == null ? col : lanes.indexOf(null);
          if (lane < 0) lane = lanes.length;
          lanes[lane] = p;
        }
        edges.push({ from: i, col, lane, to: row.get(p) });
      });
      width = Math.max(width, lanes.length, col + 1);
    });
    for (const e of edges) e.toCol = pos[e.to];
    return { pos, edges, width };
  }

  function graphSvg(commits, layout, rowH, laneW) {
    const x = (col) => 8 + col * laneW;
    const y = (r) => rowH / 2 + r * rowH;
    const parts = [];
    for (const e of layout.edges) {
      const color = LANE_COLORS[e.lane % LANE_COLORS.length];
      const x1 = x(e.col), y1 = y(e.from), x2 = x(e.lane), x3 = x(e.toCol), y3 = y(e.to);
      let d = `M${x1} ${y1}`;
      if (x2 !== x1) d += ` C${x1} ${y1 + rowH * 0.6} ${x2} ${y1 + rowH * 0.4} ${x2} ${y1 + rowH}`;
      const yEnd = x3 !== x2 ? y3 - rowH : y3;
      if (yEnd > y1 + rowH || x2 === x1) d += ` L${x2} ${Math.max(yEnd, y1)}`;
      if (x3 !== x2) d += ` C${x2} ${y3 - rowH * 0.4} ${x3} ${y3 - rowH * 0.6} ${x3} ${y3}`;
      parts.push(`<path d="${d}" stroke="${color}" />`);
    }
    commits.forEach((c, i) => {
      const color = LANE_COLORS[layout.pos[i] % LANE_COLORS.length];
      const head = c.r.some((r) => r.head);
      parts.push(`<circle cx="${x(layout.pos[i])}" cy="${y(i)}" r="${head ? 5 : 4}" fill="${c.pm ? "var(--surface, #fff)" : color}"
        stroke="${color}" stroke-width="${c.pm ? 2 : head ? 2.5 : 0}"/>`);
    });
    const w = 16 + layout.width * laneW, h = commits.length * rowH;
    return `<svg class="gp-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">${parts.join("")}</svg>`;
  }

  function refPill(r) {
    const cls = r.k === "branch" ? (r.head ? "br head" : "br") : r.k;
    const label = r.k === "branch" && r.head ? `HEAD → ${escH(r.n)}` : r.k === "tag" ? `🏷 ${escH(r.n)}` : escH(r.n);
    return `<span class="gp-ref ${cls}">${label}</span>`;
  }

  class GitPanel {
    constructor(el) { this.el = el; this.data = null; }
    update(repo) { this.data = repo; this.render(); }
    render() {
      const r = this.data;
      if (!r) { this.el.innerHTML = `<div class="gp-empty">No repository here yet. <code>git init</code> or <code>git clone</code> one and it shows up here.</div>`; return; }
      if (r.error) { this.el.innerHTML = `<div class="gp-empty">${escH(r.error)}</div>`; return; }
      const a = r.areas;
      const fileRows = (list, kind) => list.map(([st, p]) =>
        `<li class="ga-f ${kind}"><span class="ga-st">${escH(st)}</span><span class="ga-p" title="${escH(STATUS_WORD[st] || st)}">${escH(p)}</span></li>`).join("");
      const head = r.head.branch ? `on <b>${escH(r.head.branch)}</b>` : r.head.id ? `<b class="bad">detached HEAD</b> at ${escH(r.head.id)}` : "no commits yet";
      const state = r.state ? `<div class="gp-state">${escH(r.state.replace("-i", " (interactive)"))} in progress` +
        (a.conflicts.length ? ` · ${a.conflicts.length} conflicted file${a.conflicts.length > 1 ? "s" : ""}` : "") + `</div>` : "";
      const areas = `<div class="ga">
        <div class="ga-box"><div class="ga-h">Working tree <span>${a.unstaged.length + a.untracked.length + a.conflicts.length}</span></div><ul>
          ${a.conflicts.map((p) => `<li class="ga-f conflict"><span class="ga-st">U</span><span class="ga-p">${escH(p)}</span></li>`).join("")}
          ${fileRows(a.unstaged, "unstaged")}${a.untracked.map((p) => `<li class="ga-f untracked"><span class="ga-st">?</span><span class="ga-p">${escH(p)}</span></li>`).join("")}</ul></div>
        <div class="ga-arrow">git add →</div>
        <div class="ga-box"><div class="ga-h">Staging area <span>${a.staged.length}</span></div><ul>${fileRows(a.staged, "staged")}</ul></div>
        <div class="ga-arrow">git commit →</div>
        <div class="ga-box"><div class="ga-h">Repository</div><div class="ga-repo">${head}${r.head.id ? `<br><span class="dim">${escH(r.head.id)}</span>` : ""}</div></div>
      </div>`;
      const rowH = 24, laneW = 14;
      const lay = graphLayout(r.commits);
      const gw = 16 + lay.width * laneW;
      const rows = r.commits.map((c) => `<div class="gp-row" style="height:${rowH}px">
          <span class="gp-id">${escH(c.id)}</span>${c.r.map(refPill).join("")}
          <span class="gp-s" title="${escH(c.a)} · ${new Date(c.t * 1000).toLocaleString()}">${escH(c.s)}</span>
          <span class="gp-a">${escH(c.a.split(" ")[0])} · ${ago(c.t)}</span></div>`).join("");
      const graph = r.commits.length ? `<div class="gp-graph" style="--gw:${gw}px">${graphSvg(r.commits, lay, rowH, laneW)}<div class="gp-rows">${rows}</div></div>` +
        (r.more ? `<div class="gp-more">older commits not shown${r.shallow ? " · shallow clone" : ""}</div>` : r.shallow ? `<div class="gp-more">shallow clone: history ends here</div>` : "") : `<div class="gp-empty">No commits yet.</div>`;
      const branches = r.branches.map((b) => `<li class="${b.head ? "cur" : ""}"><span class="gb-n">${b.head ? "● " : ""}${escH(b.n)}</span>
        <span class="gb-id">${escH(b.id)}</span>${b.up ? `<span class="gb-up">${escH(b.up)}${b.gone ? " (gone)" : ""}
        ${b.ahead ? `<b class="ahead">↑${b.ahead}</b>` : ""}${b.behind ? `<b class="behind">↓${b.behind}</b>` : ""}</span>` : ""}</li>`).join("");
      const stash = r.stash.length ? `<div class="gp-sub">Stash</div><ul class="gp-list">${r.stash.map((s) => `<li><span class="gb-n">${escH(s.n)}</span> ${escH(s.s)}</li>`).join("")}</ul>` : "";
      const remotes = r.remotes.length ? `<div class="gp-sub">Remotes</div><ul class="gp-list">${r.remotes.map((m) => `<li><span class="gb-n">${escH(m.n)}</span> <span class="dim">${escH(m.url)}</span></li>`).join("")}</ul>` : "";
      const tags = r.tags.length ? `<div class="gp-sub">Tags</div><div class="gp-tags">${r.tags.slice(-24).map((t) => `<span class="gp-ref tag">🏷 ${escH(t)}</span>`).join("")}</div>` : "";
      this.el.innerHTML = `<div class="gp-path" title="${escH(r.path)}">${escH(r.path)}${r.bare ? " (bare)" : ""}</div>${state}${areas}
        <div class="gp-sub">History</div>${graph}
        <div class="gp-sub">Branches</div><ul class="gp-list gp-branches">${branches || `<li class="dim">none yet</li>`}</ul>${stash}${remotes}${tags}`;
    }
  }

  const PR_ICON = { OPEN: "⑂", DRAFT: "⑂", MERGED: "⛙", CLOSED: "⊘" };
  const CHECK = { success: ["✓", "good"], failure: ["✗", "bad"], in_progress: ["●", "warn"], queued: ["●", "warn"] };

  class GitHubPanel {
    constructor(el) { this.el = el; this.data = null; }
    update(gh) { this.data = gh; this.render(); }
    render() {
      const g = this.data;
      if (!g) { this.el.innerHTML = `<div class="gp-empty">GitHub isn't part of this problem.</div>`; return; }
      const who = g.me ? `Signed in as <b>${escH(g.me)}</b> · ${g.keys} SSH key${g.keys === 1 ? "" : "s"}` : `Not signed in (<code>gh auth login</code>)`;
      const r = g.repo;
      if (!r) {
        this.el.innerHTML = `<div class="gh-who">${who}</div><div class="gp-empty">No GitHub repository yet: clone one, or <code>gh repo create</code>.</div>` +
          (g.repos && g.repos.length ? `<div class="gp-sub">Your repositories</div><ul class="gp-list">${g.repos.map((x) => `<li>${escH(x)}</li>`).join("")}</ul>` : "");
        return;
      }
      const badges = [r.private ? "Private" : "Public", r.fork_of ? `fork of ${r.fork_of}` : "", r.real ? "real github.com · read-only" : ""].filter(Boolean)
        .map((b) => `<span class="gh-badge">${escH(b)}</span>`).join("");
      const branches = r.branches.map((b) => `<li><span class="gb-n">${escH(b.n)}</span><span class="gb-id">${escH(b.id)}</span>
        ${b.default ? `<span class="gh-badge">default</span>` : ""}${b.protected ? `<span class="gh-badge lock">🔒 protected</span>` : ""}</li>`).join("");
      const prs = r.prs.map((p) => {
        const ck = p.checks && CHECK[p.checks] ? `<span class="gh-ck ${CHECK[p.checks][1]}" title="checks: ${escH(p.checks)}">${CHECK[p.checks][0]}</span>` : "";
        const rv = (p.approved ? `<span class="gh-rv good">✓ ${p.approved} approved</span>` : "") + (p.changes ? `<span class="gh-rv bad">changes requested</span>` : "");
        return `<li class="gh-pr ${p.state.toLowerCase()}"><span class="gh-pri">${PR_ICON[p.state] || "⑂"}</span>
          <div><div><b>#${p.n}</b> ${escH(p.title)} ${ck}</div>
          <div class="dim">${escH(p.head)} → ${escH(p.base)} · ${escH(p.by)} · <span class="gh-st">${p.state.toLowerCase()}</span> ${rv}</div></div></li>`;
      }).join("");
      const issues = r.issues.map((i) => `<li class="gh-is ${i.state.toLowerCase()}"><span class="gh-pri">${i.state === "OPEN" ? "◎" : "✓"}</span>
        <div><b>#${i.n}</b> ${escH(i.title)} ${i.labels.map((l) => `<span class="gh-label">${escH(l)}</span>`).join("")}</div></li>`).join("");
      const runs = r.runs.map((x) => `<li><span class="gh-ck ${(CHECK[x.status] || ["", ""])[1]}">${(CHECK[x.status] || ["·"])[0]}</span>
        <b>${escH(x.wf)}</b> <span class="dim">${escH(x.title)} · ${escH(x.branch)} · ${escH(x.status.replace("_", " "))}</span></li>`).join("");
      this.el.innerHTML = `<div class="gh-who">${who}</div>
        <div class="gh-repo"><div class="gh-name">${escH(r.full)}</div>${badges}${r.desc ? `<div class="dim">${escH(r.desc)}</div>` : ""}
        ${r.pages ? `<div>Pages: <a href="#" onclick="return false" class="gh-pages">${escH(r.pages)}</a> <span class="dim">(curl it)</span></div>` : ""}</div>
        <div class="gp-sub">Branches</div><ul class="gp-list">${branches || `<li class="dim">empty repository</li>`}</ul>
        <div class="gp-sub">Pull requests</div><ul class="gp-list gh-list">${prs || `<li class="dim">none</li>`}</ul>
        <div class="gp-sub">Issues</div><ul class="gp-list gh-list">${issues || `<li class="dim">none</li>`}</ul>
        ${runs ? `<div class="gp-sub">Actions</div><ul class="gp-list">${runs}</ul>` : ""}
        ${r.tags.length ? `<div class="gp-sub">Tags</div><div class="gp-tags">${r.tags.map((t) => `<span class="gp-ref tag">🏷 ${escH(t)}</span>`).join("")}</div>` : ""}
        ${r.releases.length ? `<div class="gp-sub">Releases</div><div class="gp-tags">${r.releases.map((t) => `<span class="gh-badge">${escH(t)}</span>`).join("")}</div>` : ""}`;
    }
  }

  // Files | Git | GitHub tabs over the side panel (the tabs appear once a git problem sends its first picture)
  class SidePanel {
    constructor(el) {
      this.el = el;
      el.innerHTML = `<div class="sp-tabs" role="tablist" hidden>
          <button class="sp-tab on" data-tab="files">Files</button><button class="sp-tab" data-tab="git">Git</button><button class="sp-tab" data-tab="github">GitHub</button></div>
        <div class="sp-pane" data-pane="files"></div><div class="sp-pane gp" data-pane="git" hidden></div><div class="sp-pane gp" data-pane="github" hidden></div>`;
      this.files = new FilesPanel(el.querySelector('[data-pane="files"]'));
      this.git = new GitPanel(el.querySelector('[data-pane="git"]'));
      this.gh = new GitHubPanel(el.querySelector('[data-pane="github"]'));
      this.tabs = el.querySelector(".sp-tabs");
      this.chosen = false;
      this.tabs.onclick = (e) => { const b = e.target.closest(".sp-tab"); if (b) { this.chosen = true; this.show(b.dataset.tab); } };
    }
    show(name) {
      for (const b of this.tabs.querySelectorAll(".sp-tab")) b.classList.toggle("on", b.dataset.tab === name);
      for (const p of this.el.querySelectorAll(".sp-pane")) p.hidden = p.dataset.pane !== name;
    }
    reset() { this.tabs.hidden = true; this.chosen = false; this.show("files"); }
    gitUpdate(snap) {
      if (!snap) return;
      if (this.tabs.hidden) {
        this.tabs.hidden = false;
        this.tabs.querySelector('[data-tab="github"]').hidden = !snap.github;
      }
      if (snap.github) this.tabs.querySelector('[data-tab="github"]').hidden = false;
      this.git.update(snap.repo || null);
      this.gh.update(snap.github || null);
      if (!this.chosen && snap.repo) { this.chosen = true; this.show("git"); }
    }
  }

  // ------------------------------------------------------------ the session (page <-> worker)
  function loadLog(key, seed) {
    try {
      const raw = localStorage.getItem(LOG_PREFIX + key);
      if (!raw) return [];
      const v = JSON.parse(raw);
      return v && v.seed === seed && Array.isArray(v.log) ? v.log : [];
    } catch { return []; }
  }
  function storeLog(key, seed, log) {
    const raw = JSON.stringify({ seed, log, at: Date.now() });
    if (raw.length > LOG_MAX_CHARS) return false;
    for (let tries = 0; tries < 8; tries++) {
      try { localStorage.setItem(LOG_PREFIX + key, raw); return true; }
      catch {            // storage full: drop the oldest other session
        let oldest = null, at = Infinity;
        for (let i = 0; i < localStorage.length; i++) {
          const k = localStorage.key(i);
          if (!k || !k.startsWith(LOG_PREFIX) || k === LOG_PREFIX + key) continue;
          try { const t = JSON.parse(localStorage.getItem(k)).at || 0; if (t < at) { at = t; oldest = k; } } catch { oldest = k; at = 0; }
        }
        if (!oldest) return false;
        localStorage.removeItem(oldest);
      }
    }
    return false;
  }
  function clearLog(key) { try { localStorage.removeItem(LOG_PREFIX + key); } catch { /* storage blocked */ } }

  class TermSession {
    // py: the page's worker client; key: "bash/7.02"; problem: the generated problem
    constructor({ py, host, filesEl, dbEl, key, problem, onState, onSubmit }) {
      this.py = py;
      this.key = key;
      this.problem = problem;
      this.onState = onState || (() => {});
      this.side = new SidePanel(filesEl);
      this.files = this.side.files;
      this.db = dbEl ? new DbPanel(dbEl) : null;
      this.mt = new MacTerminal(host, { onInput: (ev) => this.input(ev), onResize: (cols) => this.resized(cols), onSubmit });
      this.queue = [];
      this.inflight = false;
      this.timer = null;
      this.log = [];
      this.nano = null;
      this.held = "";          // output that arrived while nano was open
      this.alive = false;
      this.cols = this.mt.term.cols;
    }

    now() { return performance.now(); }

    async start(fresh) {
      if (fresh) clearLog(this.key);
      const log = fresh ? [] : loadLog(this.key, this.problem.seed);
      this.log = [...log];
      this.alive = false;
      await this.mt.reset();
      if (log.length) this.mt.write("\x1b[2m[restoring your session…]\x1b[0m\r\n");
      let res;
      try { res = await this.py.call("termStart", { problem: this.problem, log, real: this.now(), dark: isDark() }, 120000); }
      catch (e) {
        if (log.length) { clearLog(this.key); return this.start(true); }   // a stored session that no longer replays
        await this.mt.write(`\r\n\x1b[31mThe terminal couldn't start: ${e.message}\x1b[0m\r\n`);
        return;
      }
      if (res.error) { await this.mt.write(`\x1b[31m${res.error}\x1b[0m\r\n`); return; }
      await this.mt.reset();
      this.side.reset();
      this.files.setStart(res.start_tree || res.tree);
      if (this.db) this.db.setStart(res.start_db || res.db);
      this.alive = true;
      if (this.cols !== 80) this.resized(this.cols);
      this.apply(res, true);
      this.mt.focus();
    }

    input(ev) {
      if (!this.alive) return;
      this.queue.push(...ev);
      this.kick();
    }

    resized(cols) {
      this.cols = cols;
      if (!this.alive) return;
      this.pendingCols = cols;
      this.kick();
    }

    schedule(at) {
      clearTimeout(this.timer);
      if (at == null) return;
      this.timer = setTimeout(() => { this.pumpDue = true; this.kick(); }, Math.max(0, at - this.now()));
    }

    async kick() {
      if (this.inflight || !this.alive) return;
      let cmd, args;
      if (this.pendingCols) { cmd = "termResize"; args = { cols: this.pendingCols, real: this.now() }; this.pendingCols = null; }
      else if (this.queue.length) { cmd = "termKeys"; args = { events: this.queue.splice(0), real: this.now() }; }
      else if (this.pumpDue) { cmd = "termPump"; args = { real: this.now() }; }
      else return;
      this.pumpDue = false;
      this.inflight = true;
      let res;
      try { res = await this.py.call(cmd, args, 30000); }
      catch (e) { this.inflight = false; return this.crashed(e); }
      this.inflight = false;
      if (!this.alive) return;
      this.apply(res);
      if (this.queue.length || this.pendingCols) this.kick();
    }

    apply(res, first) {
      if (res.error) { this.mt.write(`\r\n\x1b[31m${res.error}\x1b[0m\r\n`); return; }
      if (res.log && res.log.length && !first) {   // entries handed over are final (the worker merges only newer ones)
        this.log.push(...res.log);
        this.dirty = true;
      }
      if (this.dirty) { this.dirty = false; storeLog(this.key, this.problem.seed, this.log); }
      if (this.nano) this.held += res.out || "";
      else this.mt.update(res.out, res.view, res.title);
      if (res.tree) this.files.update(res.tree);
      if (res.db && this.db) this.db.update(res.db);
      if (res.git) this.side.gitUpdate(res.git);
      if (res.editor && !this.nano) this.openNano(res.editor);
      this.lastView = res.view;
      this.onState({ ended: res.ended, view: res.view, editor: !!res.editor });
      if (res.busy) { this.pumpDue = true; setTimeout(() => this.kick(), 0); }
      else this.schedule(res.wake_ms);
    }

    openNano(req) {
      this.nano = new Nano(this.mt, req, {
        save: async (content, name) => {
          const res = await this.py.call("termEdit", { action: "save", content, name, real: this.now() }, 30000).catch((e) => ({ edit_error: e.message }));
          const err = res.edit_error;
          if (res.log) { this.log.push(...res.log); this.dirty = true; }
          if (res.tree) this.files.update(res.tree);
          if (this.dirty) { this.dirty = false; storeLog(this.key, this.problem.seed, this.log); }
          return err;
        },
        close: async () => {
          const res = await this.py.call("termEdit", { action: "close", real: this.now() }, 30000).catch((e) => ({ error: e.message }));
          this.nano = null;
          const held = this.held;
          this.held = "";
          if (res.log) { this.log.push(...res.log); this.dirty = true; }
          res.out = held + (res.out || "");
          res.log = [];
          this.apply(res);
        },
      });
      const nano = this.nano;
      this.mt.run((done) => this.mt.erase(() => nano.open(done)));
    }

    // the worker was restarted (a command ran too long): rebuild without the last command
    async crashed(err) {
      this.alive = false;
      let cut = this.log.length;
      for (let i = this.log.length - 1; i >= 0; i--) {
        const e = this.log[i];
        if (e.k === "key" && e.v === "Enter" && e.m === "p") { cut = i; break; }
      }
      this.log = this.log.slice(0, cut);
      storeLog(this.key, this.problem.seed, this.log);
      await this.mt.write(`\r\n\x1b[33m[bash-lite: that command took too long and was stopped (${err.message}); ` +
                          `restoring the session without it]\x1b[0m\r\n`);
      await this.py.ready;
      this.nano = null;
      this.mt.sink = null;
      this.start(false);
    }

    async answer() {
      if (!this.alive) return null;
      while (this.inflight) await new Promise((r) => setTimeout(r, 20));
      if (this.queue.length) { await this.kick(); while (this.inflight) await new Promise((r) => setTimeout(r, 20)); }
      return this.py.call("termAnswer", { real: this.now() }, 30000);
    }

    stop() {
      this.alive = false;
      clearTimeout(this.timer);
      this.mt.destroy();
    }

    reset() { clearTimeout(this.timer); this.nano = null; this.mt.sink = null; this.held = ""; return this.start(true); }
    forget() { clearLog(this.key); }
    focus() { this.mt.focus(); }
  }

  window.TLTerminal = { TermSession, parseInput, layout };
})();

"""The live terminal: one bash session driven key by key, in the browser or replayed from a recorded log.

In the browser the session follows the real clock: `sleep 3` takes three seconds, `tail -f` streams until Ctrl-C,
background jobs print while you type. Every key (and every nano save) is recorded, so the session can be rebuilt:
after a page reload (the log lives in the browser's storage) and for grading, where the log is replayed on a fresh
copy of the problem's machine and on the hidden test machines.

An event records where the key went: "p" the shell's prompt, "r" a running program (with "d", the milliseconds
since that command started), "e" the nano editor. Grading replays as fast as possible: a prompt key is applied as
soon as the prompt is back, a key for a running program at the same moment of that command (Ctrl-C after 3 s of
`tail -f` stops it after 3 s). So the time spent thinking at the prompt doesn't change outputs like `ls -l` times,
and the reference solution, typed at the same speed, sees the same clock. A reload replays at the recorded times.

The shell's line editing is keys.Editor (GNU readline's emacs and vi modes); a program reading the terminal gets a
cooked-mode line (what a tty does: echo, Backspace, Ctrl-U/W, Enter; Ctrl-C/Z/\\ send signals, Ctrl-D EOF).
"""

from __future__ import annotations

import json

from .directives import build, parse, prepare, split, start_session
from .kernel import NS, TimeLimit
from .keys import Editor, KeySim, parse_keys

QUOTA_INODES = 3000          # files and directories a session may create
QUOTA_BYTES = 8 << 20        # bytes of file data it may write
LOG_LIMIT = 20000            # recorded events kept for a problem
SCREEN_LIMIT = 400_000       # characters of terminal output kept to redraw the page after a reload
TRANSCRIPT_LIMIT = 2_000_000  # characters of stdout / stderr kept for grading
COLS, ROWS = 80, 24          # Terminal.app's default window
SLICE = 20000                # interpreter steps per slice (the page stays responsive between slices)
STEP_NS = 5000               # a replay charges 5 µs of simulated time per step (a busy loop takes time)
REPLAY_STEPS = 600_000       # steps a replay may spend before giving up on a busy wait
PROMPT_WAIT = 3600 * NS      # how long a replay waits for the prompt to come back
AUTO_INTERRUPT = 10 * NS     # a command still running when the log ends is stopped (Ctrl-C) after this


def split_setup(setup: str):
    """Setup directives for the machine, the setup's bash code, and the terminal's own directives."""
    dirs, code = split(setup)
    extra, keep = {}, []
    for d in dirs:
        name, _, arg = d.partition(" ")
        if name == "buffer":
            extra["buffer"] = arg.replace("\\n", "\n")
        elif name == "cursor":
            extra["cursor"] = int(arg)
        elif name == "tmux":
            extra["tmux"] = arg or "main"
        else:
            keep.append(d)
    return keep, code, extra


class LiveTerm:
    """typed=True: an answer typed as text (tests, reference solutions): run-until-Ctrl-C commands stop on their
    own after `#: ctrlc` seconds, like the text grader. typed=False: a person at the terminal presses Ctrl-C."""

    def __init__(self, problem: dict, setup: str, typed: bool = False):
        from .check import SEED, SetupError
        keep, code, extra = split_setup(setup)
        spec = parse(keep)
        world, sess = build(spec, seed=SEED + int(problem.get("seed", 0)) % 1000)
        prepare(sess, spec)
        if code.strip():
            r = sess.run(code, quiet=True, history=False)
            if r["error"]:
                raise SetupError(r["error"])
        start_session(sess, spec)
        self.problem, self.world, self.sess, self.spec, self.extra = problem, world, sess, spec, extra
        self.k, self.tty, self.sh = world.kernel, world.tty, sess.sh
        self.fs = world.local.fs
        self.start_cwd = sess.proc.cwd
        self.home = self.sh.get("HOME", "/home/ada") or "/home/ada"
        tty = self.tty
        tty.out, tty.size = [], 0
        tty.echo = False
        tty.input_eof = False
        tty.lines.clear()
        tty.input.clear()
        tty.limit = 1 << 60
        tty.width, tty.height = spec.columns or COLS, ROWS
        self.k.budget = 1 << 62         # no step budget: Ctrl-C stops a loop (a replay caps its waits instead)
        world.editor_ok = True
        world.editor = None
        self.ctrl_c_typed = world.ctrl_c_after      # `#: ctrlc`: when typed run-until-Ctrl-C commands stop
        if not typed:
            world.ctrl_c_after = float("inf")
        self.fs.set_quota(QUOTA_INODES, QUOTA_BYTES)
        self.t0 = self.k.now
        self.offset = None              # live: simulated ns - real ns
        self.log: list[dict] = []
        self.sent = 0                   # log entries already handed to the page
        self.full = False
        self.marks: list[dict] = []     # commands the shell ran: raw line, expanded lines, output spans, status
        self.stdout_all: list[str] = []
        self.stderr_all: list[str] = []
        self.n_out = self.n_err = 0
        self.screen: list[str] = []     # what the terminal showed (kept to redraw the page)
        self.screen_size = 0
        self.pending: list[str] = []    # shown since the last update
        self.ended = False
        self.last_search = ""
        self.synced = 0                 # commands whose effects (set -o vi, tmux attach) the editor has seen
        self.ed = Editor(self)
        self.ed.set(extra.get("buffer", ""), extra.get("cursor"))
        self.ped = Editor(self)         # a program's cooked-mode input line
        self.tmux = None
        self.tmux_editors: dict = {}
        self.tmux_prefix = False
        self.tmux_prompt = None
        self.ran: list[str] = []        # KeySim's record of pane commands
        self.last_tree = None
        if spec.keys == "tmux" or "tmux" in extra:
            name = extra.get("tmux", "main")
            sess.run(f"tmux new -d -s {name}", quiet=True, history=False)
            srv = world.local.extra.get("tmux")
            if srv is not None and name in srv.sessions:
                KeySim.enter_tmux(self, srv, srv.sessions[name])
        self.sh.cmd_hook = self._hook
        tty.out, tty.size = [], 0
        p = sess.proc
        p.gen = self.sh.repl(history=True, idle=False, persistent=True)
        p.state = "R"
        self.k.runq.append(p)
        self.advance(self.k.now)
        self.pending.clear()
        self.screen.clear()
        self.screen_size = 0
        self._sync_mode()

    # ------------------------------------------------------------ running the machine

    def now_ms(self) -> int:
        return (self.k.now - self.t0) // 1_000_000

    def advance(self, until: int, cond=None, max_steps: int = REPLAY_STEPS) -> str:
        """Simulated time: run until `until` (ns) or until cond() holds. A busy process costs STEP_NS per step.
        -> "cond" | "time" (reached `until`) | "idle" (everything waits for input) | "busy" (gave up)."""
        k = self.k
        until = max(until, k.now)
        spent = 0
        base = k.now
        while True:
            if cond is not None and cond():
                return "cond"
            end = min(until, base + SLICE * STEP_NS)
            state, when = k.run_slice(end, SLICE)
            self.drain()
            if state == "busy":
                spent += SLICE
                base = k.now
                if spent >= max_steps or k.now >= until:
                    return "cond" if cond is not None and cond() else "busy"
                continue
            if cond is not None and cond():
                return "cond"
            if state == "idle":
                if cond is None:
                    k.now = max(k.now, until)
                return "idle"
            if when > until:
                k.now = max(k.now, until)
                return "time"
            base = when

    def drain(self):
        """Move what the terminal received into the screen and the per-stream transcripts."""
        tty = self.tty
        if tty.out:
            for stream, text in tty.out:
                if stream == "out":
                    if self.n_out < TRANSCRIPT_LIMIT:
                        self.stdout_all.append(text)
                        self.n_out += len(text)
                elif stream == "err":
                    if self.n_err < TRANSCRIPT_LIMIT:
                        self.stderr_all.append(text)
                        self.n_err += len(text)
                self.show(text)
            tty.out, tty.size = [], 0
        if not self.ended and self.sess.proc.state == "Z":
            self.ended = True
            if self.marks and self.marks[-1]["status"] is None:
                m = self.marks[-1]
                m["status"], m["o1"], m["e1"] = self.sess.proc.status, self.n_out, self.n_err
            self.show("\n[Process completed]\n")

    def show(self, text):
        self.pending.append(text)
        self.screen.append(text)
        self.screen_size += len(text)
        if self.screen_size > SCREEN_LIMIT * 2:
            whole = "".join(self.screen)[-SCREEN_LIMIT:]
            self.screen, self.screen_size = [whole], len(whole)

    def _hook(self, event, value):
        self.drain()
        if event == "start":
            self.marks.append({"raw": value, "lines": [], "o0": self.n_out, "e0": self.n_err, "status": None,
                               "o1": None, "e1": None, "interrupted": False, "t": self.k.now,
                               "ev": len(self.log) - 1})    # the log entry (Enter) that ran it
        elif event == "line" and self.marks:
            self.marks[-1]["lines"].append(value)
            self.ran.append(value)
        elif event == "end" and self.marks:
            m = self.marks[-1]
            m["status"], m["o1"], m["e1"] = value, self.n_out, self.n_err

    # ------------------------------------------------------------ state of the terminal

    def at_prompt(self) -> bool:
        if self.ended:
            return False
        p, tty = self.sess.proc, self.tty
        return (self.sh.awaiting is not None and p.state == "S" and bool(p.waiting) and p.waiting[0] == "tty"
                and not tty.lines and not tty.input)

    def editor(self):
        req = getattr(self.world, "editor", None)
        if req is None or req.get("done"):
            return None
        return {"path": req["path"], "shown": req["shown"], "content": req["content"], "exists": req["exists"]}

    def where(self) -> str:
        """Where a key goes now: "p" prompt, "r" a running program, "e" nano, "x" nothing (the shell exited)."""
        if self.ended:
            return "x"
        if self.editor() is not None:
            return "e"
        if self.tmux is not None or self.at_prompt():
            return "p"
        return "r"

    def cur_shell(self):
        if self.tmux is not None:
            pane = self.tmux[1].pane()
            return getattr(pane.proc, "shell", None)
        return self.sh

    def history(self):
        sh = self.cur_shell()
        return sh.history if sh is not None else []

    def prompt(self):
        sh = self.cur_shell()
        if sh is None:
            return "$ "
        ps = sh.get("PS2", "> ") if sh.awaiting == "PS2" else sh.get("PS1", "")
        return sh.prompt_expand(ps or "")

    def out(self, stream, text):
        """keys.KeySim's completion and tmux messages write through this."""
        self.show(text)

    def short(self, path):
        if path == self.home:
            return "~"
        if path.startswith(self.home + "/"):
            return "~" + path[len(self.home):]
        return path

    def title(self):
        cwd = self.sh.get("PWD", self.sess.proc.cwd) or "/"
        place = self.home.rsplit("/", 1)[-1] if cwd == self.home else (cwd.rstrip("/").rsplit("/", 1)[-1] or "/")
        proc = "-bash"
        fg = self.tty.fg_pgid
        if not self.ended and not self.at_prompt() and fg and fg != self.sess.proc.pgid:
            leader = self.world.local.procs.get(fg)
            if leader is not None and leader.argv:
                proc = leader.argv[0].rsplit("/", 1)[-1]
        if self.editor() is not None:
            proc = "nano"
        elif self.tmux is not None:
            proc = "tmux"
        return f"{place} — {proc} — {self.tty.width}×{ROWS}"

    def view(self) -> dict:
        w = self.where()
        tm = self.tmux_status()
        if w == "x":
            return {"prompt": "", "buf": "", "pos": 0, "hidden": False, "mode": "ended", "tmux": None}
        if self.tmux_prompt is not None and self.tmux_prompt[2]:
            what, text, label = self.tmux_prompt
            return {"prompt": "", "buf": "", "pos": 0, "hidden": False, "mode": "tmux", "tmux": tm,
                    "tmux_prompt": f"{label} {text}"}
        ed = self.ed if w == "p" else self.ped
        if ed is self.ed and ed.search is not None:
            st = ed.search
            pre = f"({'failed ' if st['failed'] else ''}reverse-i-search)`{st['q']}': "
            return {"prompt": pre, "buf": ed.buf, "pos": ed.pos, "hidden": False, "mode": "search", "tmux": tm}
        if ed is self.ed:
            return {"prompt": self.prompt(), "buf": ed.buf, "pos": ed.pos, "hidden": False, "mode": "prompt",
                    "tmux": tm}
        return {"prompt": "", "buf": ed.buf, "pos": ed.pos, "hidden": bool(self.tty.noecho),
                "mode": "editor" if w == "e" else "running", "tmux": tm}

    def tmux_status(self):
        if self.tmux is None:
            return None
        srv, s = self.tmux
        wins = " ".join(f"{w.index}:{w.name}{'*' if i == s.active else ''}{'Z' if w.zoomed else ''}"
                        for i, w in enumerate(s.windows))
        return f"[{s.name}] {wins}"

    def tree(self, limit: int = 500) -> dict:
        """The start directory as nested nodes for the side panel: n name, t type (d f x l), c children."""
        fs = self.fs
        count = [0]

        def node_of(path, name, depth):
            try:
                n = fs.lookup(path, follow=False)
            except Exception:  # noqa: BLE001
                return None
            item = {"n": name}
            if n.type == "l":
                item["t"] = "l"
                item["to"] = n.target
            elif n.type == "d":
                item["t"] = "d"
                kids = []
                if depth < 10:
                    for child in fs.listdir(n):
                        if count[0] >= limit:
                            item["more"] = True
                            break
                        count[0] += 1
                        c = node_of(path.rstrip("/") + "/" + child, child, depth + 1)
                        if c is not None:
                            kids.append(c)
                item["c"] = kids
            else:
                item["t"] = "x" if n.mode & 0o111 else "f"
                item["s"] = len(n.data)
            return item
        root = node_of(self.start_cwd, self.short(self.start_cwd), 0) or {"n": self.short(self.start_cwd), "t": "d",
                                                                          "c": []}
        cwd = "" if self.ended else (self.sh.get("PWD", self.sess.proc.cwd) or self.sess.proc.cwd)
        return {"root": root, "cwd": cwd, "base": self.start_cwd}

    def state(self) -> dict:
        out = "".join(self.pending)
        self.pending.clear()
        res = {"out": out, "view": self.view(), "title": self.title(), "ended": self.ended, "editor": self.editor(),
               "log": self.log[self.sent:], "full": self.full}
        self.sent = len(self.log)
        tree = self.tree()
        key = json.dumps(tree, sort_keys=True)
        if key != self.last_tree:
            self.last_tree = key
            res["tree"] = tree
        return res

    # ------------------------------------------------------------ input

    def record(self, ev):
        if len(self.log) >= LOG_LIMIT:
            if not self.full:
                self.full = True
                self.show("\n\x1b[33m[this terminal has recorded as much as it can: press Reset to start over]\x1b[0m\n")
            return False
        last = self.log[-1] if len(self.log) > self.sent else None
        if (ev["k"] == "text" and last is not None and last["k"] == "text" and last["m"] == ev["m"]
                and ev["m"] != "r"):
            last["v"] += ev["v"]          # one entry per run of typed text at the prompt
            return True
        self.log.append(ev)
        return True

    def stamp(self, kind, val=None):
        w = self.where()
        ev = {"t": self.now_ms(), "k": kind, "m": w}
        if val is not None:
            ev["v"] = val
        if w == "r" and self.marks:
            ev["d"] = (self.k.now - self.marks[-1]["t"]) // 1_000_000
        return ev

    def feed(self, kind, val, record=True):
        """One key ("key", "C-a") or typed text ("text", "ls -la")."""
        self.sync()
        if record and not self.record(self.stamp(kind, val)):
            return
        w = self.where()
        if w in ("x", "e"):
            return
        if self.tmux is not None:
            self._tmux_input(kind, val)
        elif w == "p":
            self._prompt_input(kind, val)
        else:
            self._program_input(kind, val)

    def sync(self):
        """After a command finishes: the editor follows `set -o vi`, `tmux attach` attaches the client."""
        if self.synced == len(self.marks) or not self.at_prompt():
            return
        new = self.marks[self.synced:]
        self.synced = len(self.marks)
        self._sync_mode()
        if self.tmux is None:
            for m in new:
                for line in m["lines"] or [m["raw"]]:
                    if self.tmux is None:
                        KeySim._check_tmux_attach(self, line)

    def _prompt_input(self, kind, val):
        ed = self.ed
        if kind == "text":
            ed.text(val)
            return
        r = ed.key(val)
        if r == "accept":
            line = ed.buf
            self.show(self.prompt() + line + "\n")
            self._new_line()
            self.tty.lines.append(line)
            self.k.wake(self.tty)
        elif r == "comment":
            self.show(self.prompt() + ed.buf + "\n")
            self.sh.add_history(ed.buf)
            self._new_line()
        elif r == "eof":
            self.show(self.prompt() + "\n")
            self._new_line()
            self.tty.eof_pending = True
            self.k.wake(self.tty)
        elif r == "abort":
            self.show(self.prompt() + ed.buf + "^C\n")
            self._new_line()
            self.sh.status = 130
            if self.sh.awaiting == "PS2":
                self.cancel_pending()
        elif r == "clear":
            self.show("\x1b[H\x1b[2J")

    def cancel_pending(self):
        """Ctrl-C at the > prompt: the unfinished command is dropped."""
        self.sh.cancel_pending = True
        self.sh.awaiting = "PS1"
        if self.marks and self.marks[-1]["status"] is None:
            m = self.marks[-1]
            m["status"], m["o1"], m["e1"], m["interrupted"] = 130, self.n_out, self.n_err, True

    def _program_input(self, kind, val):
        ped, tty = self.ped, self.tty
        if kind == "text":
            ped.insert(val)
            return
        if val in ("C-c", "C-z", "C-\\"):
            sig, echo = {"C-c": (2, "^C"), "C-z": (20, "^Z"), "C-\\": (3, "^\\")}[val]
            self.show(ped.buf if not tty.noecho else "")
            self.show(echo)
            ped.set("")
            tty.lines.clear()
            tty.input.clear()
            if val == "C-c" and self.marks:
                self.marks[-1]["interrupted"] = True
            self.signal_foreground(sig)
            return
        if val == "C-d":
            if ped.buf:
                self.show(ped.buf if not tty.noecho else "")
                tty.input += ped.buf.encode("utf-8", "surrogateescape")
                ped.set("")
            else:
                tty.eof_pending = True
            self.k.wake(tty)
            return
        if val in ("Enter", "C-m", "C-j"):
            line = ped.buf
            self.show(("" if tty.noecho else line) + "\n")
            ped.set("")
            tty.lines.append(line)
            self.k.wake(tty)
            return
        if val in ("BSpace", "C-h"):
            ped.set(ped.buf[:-1])
        elif val == "C-u":
            ped.set("")
        elif val == "C-w":
            b = ped.buf.rstrip(" ")
            ped.set(b[:len(b) - len(b.split(" ")[-1])] if b else "")
        elif val == "Space":
            ped.insert(" ")
        elif val == "Tab":
            ped.insert("\t")

    def signal_foreground(self, sig):
        fg = self.tty.fg_pgid
        shell = self.sess.proc
        targets = [p for p in self.world.local.procs.values()
                   if fg and p.pgid == fg and p is not shell and p.state != "Z"]
        if not targets and shell.state != "Z":
            targets = [shell]
        for p in targets:
            self.k.signal(p, sig)

    def _new_line(self):
        vi = self.ed.vi
        ring = self.ed.kill_ring
        self.ed = Editor(self)
        self.ed.kill_ring = ring
        self.ed.vi = vi
        if self.tmux is not None:
            self.tmux_editors[id(self.tmux[1].pane())] = self.ed
        else:
            self.tmux_editors["outer"] = self.ed

    def _sync_mode(self):
        sh = self.cur_shell()
        if sh is not None:
            self.ed.vi = "vi" in sh.opts
            self.ed.vi_cmd = False

    # tmux (an attached client): lines go to the pane's shell, C-b starts a tmux key
    def _tmux_input(self, kind, val):
        if self.tmux_prompt is not None:
            KeySim.tmux_prompt_key(self, kind, val)
            return
        if self.tmux_prefix:
            self.tmux_prefix = False
            ch = val if kind == "key" else val[:1]
            self._guard(KeySim.tmux_key, ch)
            if kind == "text" and len(val) > 1:
                self._tmux_input("text", val[1:])
            return
        if kind == "key" and val == "C-b":
            self.tmux_prefix = True
            return
        ed = self.ed
        if kind == "text":
            ed.text(val)
            return
        r = ed.key(val)
        if r == "accept":
            line = ed.buf
            self._new_line()
            self._guard(KeySim._pane_line, line)
            self._sync_mode()
        elif r == "abort":
            self.show(self.prompt() + ed.buf + "^C\n")
            self._new_line()
        elif r == "eof":
            self._new_line()
            self._guard(KeySim._pane_line, "exit")
        elif r == "clear":
            self.show("\x1b[H\x1b[2J")

    def _guard(self, fn, *args):
        """tmux pane commands run to completion at once; a runaway one is stopped instead of hanging the page."""
        k = self.k
        saved = k.budget
        k.budget = k.steps + 2_000_000
        try:
            fn(self, *args)
        except TimeLimit:
            self.show("\n\x1b[33m[bash-lite: that command ran too long inside tmux and was stopped]\x1b[0m\n")
            if self.tmux is not None:
                pane = self.tmux[1].pane()
                k.kill_now(pane.proc, 9)
                self.leave_tmux()
        finally:
            k.budget = saved

    # keys.KeySim's tmux and completion code calls these on its "sim"
    def leave_tmux(self):
        KeySim.leave_tmux(self)

    def tmux_cmd(self, *args):
        KeySim.tmux_cmd(self, *args)

    def tmux_key(self, k):
        KeySim.tmux_key(self, k)

    def _save_editor(self):
        KeySim._save_editor(self)

    def _load_editor(self):
        KeySim._load_editor(self)

    def enter_tmux(self, srv, s):
        KeySim.enter_tmux(self, srv, s)

    def _schedule(self, which):
        pass

    def _check_tmux_attach(self, line):
        KeySim._check_tmux_attach(self, line)

    def _pane_line(self, line, interrupt=None):
        KeySim._pane_line(self, line, interrupt)

    def complete(self, ed, listing=False, only_list=False):
        KeySim.complete(self, ed, listing, only_list)

    # ------------------------------------------------------------ nano

    def edit(self, action, content=None, record=True, name=None):
        """The page's nano: "save" (the whole text, maybe under a new name) or "close".
        -> an error message for a failed save, or None."""
        req = getattr(self.world, "editor", None)
        if req is None or req.get("done"):
            return None
        if record:
            ev = self.stamp(action, content if action == "save" else None)
            if name:
                ev["n"] = name
            if not self.record(ev):
                return "the terminal's history is full"
        if action == "save":
            from .utils.editor import save
            return save(self.world, req, content if content is not None else "", name)
        if action == "close":
            req["done"] = True
            p = self.world.local.procs.get(req["pid"])
            if p is not None:
                self.k.poke(p)
        return None

    # ------------------------------------------------------------ the page's clock

    def live_until(self, real_ms: float) -> int:
        real = int(real_ms * 1_000_000)
        if self.offset is None:
            self.offset = self.k.now - real
        return max(self.k.now, real + self.offset)

    def pump(self, real_ms: float) -> dict:
        """Live: run up to the real time now (one slice). -> the page's update, with when to call again."""
        state, when = self.k.run_slice(self.live_until(real_ms), SLICE)
        self.drain()
        self.sync()
        res = self.state()
        res["busy"] = state == "busy"
        res["wake_ms"] = None if when is None else real_ms + (when - self.k.now) / 1_000_000
        return res

    # ------------------------------------------------------------ replays

    def apply(self, ev):
        k = ev["k"]
        if k in ("key", "text"):
            self.feed(k, ev.get("v", ""), record=False)
        elif k in ("save", "close"):
            self.edit(k, ev.get("v"), record=False, name=ev.get("n"))

    def _cmd_start(self):
        return self.marks[-1]["t"] if self.marks else self.t0

    def wait_for(self, ev):
        """Grading: run until the terminal is where the recorded key went."""
        m = ev.get("m", "p")
        if m == "p":
            self.advance(self.k.now + PROMPT_WAIT, cond=lambda: self.where() != "r")
        elif m == "e":
            self.advance(self.k.now + PROMPT_WAIT, cond=lambda: self.where() in ("e", "x"))
        elif m == "r":
            due = self._cmd_start() + int(ev.get("d", 0)) * 1_000_000
            if self.where() == "r":
                self.advance(due, cond=lambda: self.where() != "r")
        self.sync()

    def replay(self, log, end=None, faithful=False, final=True):
        """Re-run a recorded log. faithful: at the recorded times (a reload); else as fast as the commands allow.
        final: then let the last command finish (or stop it) as when the answer was submitted."""
        for ev in log:
            if ev.get("k") not in ("key", "text", "save", "close"):
                continue
            if faithful:
                self.advance(self.t0 + int(ev.get("t", 0)) * 1_000_000)
                self.sync()
            else:
                self.wait_for(ev)
            self.log.append(ev)
            self.apply(ev)
        if faithful:
            return
        if final:
            self.finish(end)
        else:
            self.wait_for({"m": "p"})

    def finish(self, end=None):
        """After the last event: let the commands complete; stop (Ctrl-C) one still running in the foreground.
        `end`: where the terminal was when the answer was submitted ({"m": .., "d": ..})."""
        if self.where() == "e":
            self.edit("close", record=False)
        if end and end.get("m") == "r" and self.where() == "r":
            due = self._cmd_start() + int(end.get("d", 0)) * 1_000_000
            self.advance(due, cond=lambda: self.where() != "r")
        else:
            self.advance(self.k.now + AUTO_INTERRUPT, cond=lambda: self.where() != "r")
        for _ in range(3):
            if self.where() == "e":
                self.edit("close", record=False)
                self.advance(self.k.now + NS, cond=lambda: self.where() != "e")
            if self.where() != "r":
                break
            self.feed("key", "C-c", record=False)
            self.advance(self.k.now + AUTO_INTERRUPT, cond=lambda: self.where() != "r")
        self.sync()

    def run_after(self, code: str) -> dict:
        """Grading: run `code` in the session after the answer (the problem's check, a script's run lines).
        -> {stdout, stderr, status, error}"""
        if self.where() != "p" or self.tmux is not None:
            return self._fresh_run(code)
        if self.sh.awaiting == "PS2":
            self.cancel_pending()
        self.tty.input_eof = True       # a script reading more than the typed lines gets end of file
        first = len(self.marks)
        for line in code.split("\n"):
            self.tty.lines.append(line)
        self.k.wake(self.tty)
        st = self.advance(self.k.now + PROMPT_WAIT, cond=lambda: self.at_prompt() or self.ended,
                          max_steps=4_000_000)
        error = None if st == "cond" else "Time limit exceeded: the check ran too long."
        out, err = "".join(self.stdout_all), "".join(self.stderr_all)
        new = self.marks[first:]
        if not new:
            return {"stdout": "", "stderr": "", "status": self.sh.status, "error": error}
        status = new[-1]["status"] if new[-1]["status"] is not None else self.sh.status
        return {"stdout": out[new[0]["o0"]:], "stderr": err[new[0]["e0"]:], "status": status, "error": error}

    def _fresh_run(self, code):
        """The session is gone (exit) or busy: run in a new session at the directory where the answer ended."""
        from .world import Session
        cwd = self.sess.proc.cwd
        if self.sess.proc.state != "Z":
            for p in list(self.world.local.procs.values()):
                if p.pgid == self.tty.fg_pgid and p is not self.sess.proc:
                    self.k.kill_now(p, 9)
            self.k.kill_now(self.sess.proc, 9)
        tty = self.tty
        tty.lines.clear()
        tty.input.clear()
        tty.input_eof = True
        tty.limit = 2_000_000
        tty.waiters = []
        tty.fg_pgid = 0
        self.k.budget = self.k.steps + 4_000_000
        s2 = Session(self.world, cwd=cwd if self.fs.exists(cwd) else self.start_cwd)
        r = s2.run(code, history=False)
        return {"stdout": r["stdout"], "stderr": r["stderr"], "status": r["status"], "error": r["error"]}

    # ------------------------------------------------------------ results

    def commands(self):
        """Per command the shell ran: raw line, expanded lines, stdout, stderr, status, interrupted."""
        out, err = "".join(self.stdout_all), "".join(self.stderr_all)
        res = []
        for m in self.marks:
            o1 = m["o1"] if m["o1"] is not None else len(out)
            e1 = m["e1"] if m["e1"] is not None else len(err)
            res.append({"raw": m["raw"], "lines": m["lines"], "stdout": out[m["o0"]:o1], "stderr": err[m["e0"]:e1],
                        "status": m["status"], "interrupted": m["interrupted"] or m["status"] == 130})
        return res

    def prefix_before(self, i):
        """The log up to command i: everything until the first prompt key after command i-1 was entered."""
        if i <= 0:
            return []
        k = self.marks[i - 1]["ev"] + 1
        while k < len(self.log) and self.log[k].get("m") != "p":
            k += 1
        return self.log[:k]

    def status(self):
        """$? at the end, or the shell's exit status."""
        if self.ended:
            return self.sess.proc.status
        return self.sh.status


# ---------------------------------------------------------------- answers as logs

def text_log(code: str, typed=()):
    """A typed answer (lines of bash) as a key log: each line typed at the prompt, then Enter. `typed`: lines typed
    afterwards for a program that reads the terminal (`cp -i`'s question)."""
    log = []
    lines = code.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in list(lines) + list(typed):
        if line:
            log.append({"t": 0, "k": "text", "v": line, "m": "p"})
        log.append({"t": 0, "k": "key", "v": "Enter", "m": "p"})
    return log


def reference_session(problem: dict) -> str:
    """The reference solution as a terminal session (the answer the page would send): typed line by line, a
    script written with nano, keys pressed. Used by tests and the in-browser self-test."""
    setup = problem.get("setup", "")
    if "#: keys" in setup:
        log = keys_log(problem["solution"])
    else:
        keep, _, _ = split_setup(setup)
        spec = parse(keep)
        if spec.script:
            log = text_log(f"nano {spec.script}") + [
                {"t": 0, "k": "save", "v": problem["solution"].rstrip("\n") + "\n", "m": "e"},
                {"t": 0, "k": "close", "m": "e"}]
        else:
            log = _typed_like_a_person(problem, spec)
    return json.dumps({"log": log, "end": None})


def _typed_like_a_person(problem, spec):
    """Type the solution's lines into a terminal as a person would: a command that runs until Ctrl-C (tail -f,
    jupyter lab ..., the ones a typed run stops after `#: ctrlc` seconds) gets a Ctrl-C after that long; anything
    else is waited for."""
    lines = problem["solution"].split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines += list(spec.typed)
    typed = text_log("\n".join(lines))
    ref = LiveTerm(problem, problem.get("setup", ""), typed=True)
    ref.replay(typed)
    line_of = {}              # log index of each Enter -> the line it ends
    for k, ev in enumerate(typed):
        if ev["k"] == "key":
            line_of[k] = len(line_of)
    stopped = {line_of.get(m["ev"], -1) + len(m["lines"]) - 1 for m in ref.marks if m["status"] == 130}
    t = LiveTerm(problem, problem.get("setup", ""))
    wait = t.ctrl_c_typed * NS
    for n, line in enumerate(lines):
        if line:
            t.feed("text", line)
        t.feed("key", "Enter")
        if n in stopped:
            t.advance(t.k.now + wait, cond=lambda: t.where() != "r")
            if t.where() == "r":
                t.feed("key", "C-c")
        t.advance(t.k.now + PROMPT_WAIT, cond=lambda: t.where() != "r")
    return t.log


def keys_log(keys_text: str):
    return [{"t": 0, "k": kind, "v": val, "m": "p"} for kind, val in parse_keys(keys_text)]


def parse_answer(code: str):
    """A terminal session arrives as JSON {"log": [...], "end": {...}}. -> (log, end), or None for a typed answer."""
    s = code.lstrip()
    if s.startswith("{") and '"log"' in s:
        try:
            data = json.loads(s)
        except ValueError:
            return None
        if isinstance(data, dict) and isinstance(data.get("log"), list):
            return [ev for ev in data["log"] if isinstance(ev, dict) and "k" in ev], data.get("end")
    return None


def typed_lines(log) -> list[str]:
    """What a log typed at the prompt, line by line (the keys are applied the way readline would)."""
    lines, cur = [], ""
    for ev in log:
        if ev.get("k") == "text":
            cur += ev.get("v", "")
        elif ev.get("k") == "key" and ev.get("v") in ("Enter", "C-m", "C-j"):
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def key_features(log) -> set[str]:
    """keys.key_features for a recorded log: the keys pressed, and "text" if anything was typed."""
    have = {ev["v"] for ev in log if ev.get("k") == "key"}
    if any(ev.get("k") == "text" and ev.get("v") for ev in log):
        have.add("text")
    return have


# ---------------------------------------------------------------- the page's terminal (one per worker)

_TERM: list = [None]


def start(problem: dict, log: list, real_ms: float) -> dict:
    """Open the terminal for a problem; `log` (from the browser's storage) rebuilds an earlier session."""
    t = LiveTerm(problem, problem.get("setup", ""))
    _TERM[0] = t
    start_tree = t.tree()
    if log:
        t.replay(log, faithful=True)
        t.advance(t.k.now, cond=lambda: t.where() != "r", max_steps=SLICE)
        t.sync()
    t.sent = len(t.log)
    t.offset = t.k.now - int(real_ms * 1_000_000)
    t.pending.clear()
    res = t.state()
    res["out"] = "".join(t.screen)
    res["busy"], res["wake_ms"] = False, real_ms
    res["start_tree"] = start_tree
    res["tree"] = t.tree()
    return res


def _term():
    t = _TERM[0]
    if t is None:
        raise RuntimeError("the terminal isn't open")
    return t


def keys(events: list, real_ms: float) -> dict:
    t = _term()
    for kind, val in events:
        t.k.run_slice(t.live_until(real_ms), SLICE)
        t.drain()
        t.feed(kind, val)
    return t.pump(real_ms)


def pump(real_ms: float) -> dict:
    return _term().pump(real_ms)


def edit(action: str, content, real_ms: float, name=None) -> dict:
    t = _term()
    t.k.run_slice(t.live_until(real_ms), SLICE)
    t.drain()
    err = t.edit(action, content, name=name)
    res = t.pump(real_ms)
    res["edit_error"] = err
    return res


def resize(cols: int, real_ms: float) -> dict:
    t = _term()
    t.tty.width = max(20, min(int(cols), 300))
    t.sh.set_var("COLUMNS", str(t.tty.width))
    return t.pump(real_ms)


def answer(real_ms: float) -> str:
    """The session so far, as the answer the grader replays."""
    t = _term()
    t.k.run_slice(t.live_until(real_ms), SLICE)
    t.drain()
    end = t.stamp("end")
    return json.dumps({"log": t.log, "end": {"m": end["m"], "d": end.get("d", 0)}})

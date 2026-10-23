"""Keystroke answers (7.17): a readline line editor and tmux's prefix keys on top of a real session.

An answer is a list of keys and text, separated by spaces:
    C-a  M-f  C-w  C-y  C-r  Tab  Enter  Esc  BSpace  Delete  Left  Right  Up  Down  Home  End  Space
    "quoted text"   (typed as is)      plain words that are not key names are typed too
Ctrl and Alt can be written C-a / Ctrl-a / Ctrl+A / ^A and M-b / Alt-b / Alt+B. In tmux, C-b is the prefix: `C-b d`.

Setup directives: `#: keys` (or `#: keys tmux`: the terminal starts attached to a tmux session, `#: tmux NAME`),
`#: buffer TEXT` (text already on the command line), `#: cursor N` (its cursor position; default: end of line),
plus the usual `#: history ...` lines.

Graded: the command lines that were run (after history expansion), the text left on the command line, and for tmux
the sessions/windows/panes and where the client is.
"""

from __future__ import annotations

import re
import shlex

KEYNAMES = {"tab": "Tab", "enter": "Enter", "return": "Enter", "ret": "Enter", "esc": "Esc", "escape": "Esc",
            "bspace": "BSpace", "backspace": "BSpace", "bs": "BSpace", "delete": "Delete", "del": "Delete",
            "left": "Left", "right": "Right", "up": "Up", "down": "Down", "home": "Home", "end": "End",
            "space": "Space", "pgup": "PgUp", "pgdn": "PgDn", "ppage": "PgUp", "npage": "PgDn"}


class KeyError_(ValueError):
    pass


def parse_keys(text: str) -> list[tuple[str, str]]:
    """-> [("key", "C-a") | ("text", "sudo ")]. A lone quote character is the key itself (tmux's C-b ")."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "'\"":
            j = text.find(c, i + 1)
            if j > i and (j + 1 == n or text[j + 1].isspace()):
                out.append(("text", text[i + 1:j]))
                i = j + 1
                continue
            if i + 1 == n or text[i + 1].isspace():
                out.append(("text", c))
                i += 1
                continue
            raise KeyError_(f"Couldn't read the keys: the quote at {text[i:i + 12]!r} is never closed.")
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tok = text[i:j]
        k = norm_key(tok)
        out.append(("key", k) if k is not None else ("text", tok))
        i = j
    return out


def norm_key(tok: str) -> str | None:
    t = tok.strip()
    low = t.lower()
    if low in KEYNAMES:
        return KEYNAMES[low]
    m = re.fullmatch(r"(?:c|ctrl|control)[-+](.+)", t, re.I) or re.fullmatch(r"\^(.)", t)
    if m:
        rest = m.group(1)
        mm = re.fullmatch(r"(?:m|alt|meta)[-+](.)", rest, re.I)
        if mm:
            return "C-M-" + mm.group(1).lower()
        if len(rest) == 1:
            return "C-" + (rest.lower() if rest.isalpha() else rest)
        if rest.lower() in KEYNAMES:
            return "C-" + KEYNAMES[rest.lower()]
        return None
    m = re.fullmatch(r"(?:m|alt|meta|opt|option)[-+](.+)", t, re.I)
    if m:
        rest = m.group(1)
        if len(rest) == 1:
            return "M-" + (rest.lower() if rest.isalpha() and rest.isupper() and False else rest)
        if rest.lower() in KEYNAMES:
            return "M-" + KEYNAMES[rest.lower()]
        return None
    m = re.fullmatch(r"F(\d{1,2})", t)
    if m:
        return t
    return None


def key_features(text: str) -> set[str]:
    try:
        return {v for kind, v in parse_keys(text) if kind == "key"} | ({"text"} if any(k == "text" for k, _ in parse_keys(text)) else set())
    except KeyError_:
        return set()


def _is_word(c):
    return c.isalnum()


# ---------------------------------------------------------------------------- the line editor

class Editor:
    """GNU readline's emacs mode (and a vi mode), one command line at a time."""

    def __init__(self, sim):
        self.sim = sim
        self.buf, self.pos = "", 0
        self.kill_ring: list[str] = []
        self.last = ""                 # previous command, for kill merging / yank-pop / repeated keys
        self.undo: list[tuple[str, int]] = []
        self.hist_pos = None           # index into history while browsing; None = the new line
        self.saved_line = ("", 0)
        self.search = None             # reverse-i-search state
        self.yank_arg_depth = 0
        self.vi = False
        self.vi_cmd = False
        self.vi_pending = ""
        self.prefix_meta = False
        self.tab_count = 0

    # ---- helpers
    @property
    def history(self):
        return self.sim.history()

    def set(self, buf, pos=None):
        self.buf = buf
        self.pos = len(buf) if pos is None else max(0, min(pos, len(buf)))

    def push_undo(self):
        self.undo.append((self.buf, self.pos))

    def insert(self, text):
        self.push_undo()
        self.buf = self.buf[:self.pos] + text + self.buf[self.pos:]
        self.pos += len(text)

    def kill(self, start, end, backward):
        if start == end:
            return
        text = self.buf[start:end]
        self.push_undo()
        if self.last.startswith("kill") and self.kill_ring:
            self.kill_ring[-1] = text + self.kill_ring[-1] if backward else self.kill_ring[-1] + text
        else:
            self.kill_ring.append(text)
        self.buf = self.buf[:start] + self.buf[end:]
        self.pos = start

    def word_back(self, pos):
        while pos > 0 and not _is_word(self.buf[pos - 1]):
            pos -= 1
        while pos > 0 and _is_word(self.buf[pos - 1]):
            pos -= 1
        return pos

    def word_fwd(self, pos):
        n = len(self.buf)
        while pos < n and not _is_word(self.buf[pos]):
            pos += 1
        while pos < n and _is_word(self.buf[pos]):
            pos += 1
        return pos

    def bigword_back(self, pos):
        while pos > 0 and self.buf[pos - 1].isspace():
            pos -= 1
        while pos > 0 and not self.buf[pos - 1].isspace():
            pos -= 1
        return pos

    # ---- dispatch
    def key(self, k):
        """Handle one key. Returns "accept" when the line is submitted, "eof" for Ctrl-D on an empty line."""
        if self.search is not None:
            r = self.search_key(k)
            if r != "pass":
                return r
        if self.vi:
            return self.vi_key(k)
        if self.prefix_meta and not k.startswith(("M-", "C-")) and k not in ("Esc",):
            self.prefix_meta = False
            k = "M-" + (" " if k == "Space" else k)
        elif k == "Esc":
            self.prefix_meta = True
            return None
        r = self.emacs_key(k)
        self.last = r if isinstance(r, str) and r.startswith(("kill", "yank", "complete")) else (k if r is None else r)
        if k != "Tab":
            self.tab_count = 0
        return r if r in ("accept", "eof", "abort", "clear", "comment") else None

    def text(self, s):
        if self.search is not None:
            for ch in s:
                self.search_key(ch)
            return
        if self.vi and self.vi_cmd:
            for ch in s:
                self.vi_key(ch)
            return
        if self.prefix_meta and s:
            self.prefix_meta = False
            self.key("M-" + s[0])
            s = s[1:]
            if not s:
                return
        self.insert(s)
        self.last = "insert"
        self.tab_count = 0

    def emacs_key(self, k):
        b, p, n = self.buf, self.pos, len(self.buf)
        if k == "Space":
            self.insert(" ")
            return "insert"
        if k in ("Enter", "C-m", "C-j"):
            return "accept"
        if k in ("C-a", "Home"):
            self.pos = 0
        elif k in ("C-e", "End"):
            self.pos = n
        elif k in ("C-b", "Left"):
            self.pos = max(0, p - 1)
        elif k in ("C-f", "Right"):
            self.pos = min(n, p + 1)
        elif k in ("M-b", "C-Left", "M-Left"):
            self.pos = self.word_back(p)
        elif k in ("M-f", "C-Right", "M-Right"):
            self.pos = self.word_fwd(p)
        elif k in ("BSpace", "C-h"):
            if p > 0:
                self.push_undo()
                self.set(b[:p - 1] + b[p:], p - 1)
        elif k == "Delete":
            if p < n:
                self.push_undo()
                self.set(b[:p] + b[p + 1:], p)
        elif k == "C-d":
            if not b:
                return "eof"
            if p < n:
                self.push_undo()
                self.set(b[:p] + b[p + 1:], p)
        elif k == "C-u":
            self.kill(0, p, True)
            return "kill"
        elif k == "C-k":
            self.kill(p, n, False)
            return "kill"
        elif k == "C-w":
            self.kill(self.bigword_back(p), p, True)
            return "kill"
        elif k in ("M-BSpace", "C-M-h", "M-Delete"):
            self.kill(self.word_back(p), p, True)
            return "kill"
        elif k == "M-d":
            self.kill(p, self.word_fwd(p), False)
            return "kill"
        elif k == "C-y":
            if self.kill_ring:
                self.insert(self.kill_ring[-1])
                self.yanked = len(self.kill_ring[-1])
            return "yank"
        elif k == "M-y":
            if self.last == "yank" and len(self.kill_ring) > 1:
                ln = self.yanked
                self.kill_ring.insert(0, self.kill_ring.pop())
                t = self.kill_ring[-1]
                self.set(self.buf[:self.pos - ln] + t + self.buf[self.pos:], self.pos - ln + len(t))
                self.yanked = len(t)
                return "yank"
        elif k == "C-t":
            if n >= 2 and p > 0:
                self.push_undo()
                q = p if p < n else n - 1
                self.set(b[:q - 1] + b[q] + b[q - 1] + b[q + 1:], q + 1)
        elif k == "M-t":
            self._transpose_words()
        elif k in ("M-u", "M-l", "M-c"):
            e = self.word_fwd(p)
            w = b[p:e]
            s = 0
            while s < len(w) and not _is_word(w[s]):
                s += 1
            w2 = w[:s] + (w[s:].upper() if k == "M-u" else w[s:].lower() if k == "M-l" else w[s:].capitalize())
            self.push_undo()
            self.set(b[:p] + w2 + b[e:], e)
        elif k in ("C-_", "C-/", "C-x C-u"):
            if self.undo:
                self.buf, self.pos = self.undo.pop()
        elif k in ("Up", "C-p"):
            self.hist_move(-1)
        elif k in ("Down", "C-n"):
            self.hist_move(1)
        elif k == "M-<":
            self.hist_go(0)
        elif k == "M->":
            self.hist_go(len(self.history))
        elif k in ("M-.", "M-_"):
            self._yank_last_arg()
            return "yank-arg"
        elif k == "C-r":
            self.search = {"q": "", "dir": -1, "idx": len(self.history), "at": None, "orig": (b, p), "failed": False}
        elif k == "C-s":
            self.search = {"q": "", "dir": 1, "idx": self.hist_pos if self.hist_pos is not None else len(self.history),
                           "at": None, "orig": (b, p), "failed": False}
        elif k == "Tab":
            self.tab_count += 1
            self.sim.complete(self, listing=self.tab_count >= 2)
            return "complete"
        elif k in ("M-?", "M-="):
            self.sim.complete(self, listing=True, only_list=True)
        elif k == "C-l":
            return "clear"
        elif k == "C-c":
            return "abort"
        elif k == "M-#":
            self.set("#" + b)
            return "comment"
        elif k in ("C-g",):
            pass
        elif k in ("C-z", "C-q", "C-v", "C-o", "C-x", "C-s", "C-\\"):
            pass
        elif k.startswith("M-") and len(k) == 3 and k[2].isdigit():
            pass
        else:
            return None
        return None

    def _transpose_words(self):
        b, p = self.buf, self.pos
        e2 = self.word_fwd(p)
        s2 = self.word_back(e2)
        s1 = self.word_back(s2)
        e1 = s1
        while e1 < len(b) and _is_word(b[e1]):
            e1 += 1
        if s1 == s2 or e1 > s2:
            return
        self.push_undo()
        self.set(b[:s1] + b[s2:e2] + b[e1:s2] + b[s1:e1] + b[e2:], e2)

    def _yank_last_arg(self):
        hist = self.history
        if self.last == "yank-arg":
            self.yank_arg_depth += 1
            start = self.pos - self.yank_len
            self.buf = self.buf[:start] + self.buf[self.pos:]
            self.pos = start
        else:
            self.yank_arg_depth = 1
        k = len(hist) - self.yank_arg_depth
        if k < 0 or not hist:
            self.yank_len = 0
            return
        try:
            words = shlex.split(hist[k], posix=False)
        except ValueError:
            words = hist[k].split()
        arg = words[-1] if words else ""
        self.insert(arg)
        self.yank_len = len(arg)

    def hist_move(self, d):
        hist = self.history
        cur = len(hist) if self.hist_pos is None else self.hist_pos
        new = cur + d
        if new < 0 or new > len(hist):
            return
        self.hist_go(new)

    def hist_go(self, new):
        hist = self.history
        if self.hist_pos is None:
            self.saved_line = (self.buf, self.pos)
        if new >= len(hist):
            self.hist_pos = None
            self.set(*self.saved_line) if False else self.set(self.saved_line[0])
        else:
            self.hist_pos = new
            self.set(hist[new])

    # ---- incremental search
    def search_key(self, k):
        st = self.search
        hist = self.history
        if len(k) == 1 or k == "Space":
            st["q"] += " " if k == "Space" else k
            self._search_find(st, same=True)
            return None
        if k in ("C-r", "C-s"):
            st["dir"] = -1 if k == "C-r" else 1
            if not st["q"] and self.sim.last_search:
                st["q"] = self.sim.last_search
                self._search_find(st, same=True)
            else:
                self._search_find(st, same=False)
            return None
        if k in ("BSpace", "C-h"):
            st["q"] = st["q"][:-1]
            st["idx"] = len(hist) if st["dir"] < 0 else st["idx"]
            self._search_find(st, same=True)
            return None
        if k == "C-g":
            self.set(*st["orig"])
            self.search = None
            return None
        # anything else ends the search, keeping the found line; Esc / C-j just end it
        self.sim.last_search = st["q"]
        self.search = None
        if k in ("Esc", "C-j"):
            return None
        return "pass"

    def _search_find(self, st, same):
        hist = self.history
        q = st["q"]
        if not q:
            return
        start = st["idx"] if same else st["idx"] + st["dir"]
        rng = range(min(start, len(hist) - 1), -1, -1) if st["dir"] < 0 else range(max(start, 0), len(hist))
        for i in rng:
            line = hist[i]
            j = line.rfind(q) if st["dir"] < 0 else line.find(q)
            if j >= 0:
                if not same or st["at"] is None or i != st["idx"] or True:
                    st["idx"], st["at"], st["failed"] = i, j, False
                    self.hist_pos = i
                    self.set(line, j)
                    return
        st["failed"] = True

    # ---- vi mode
    def vi_key(self, k):
        if not self.vi_cmd:
            if k == "Esc":
                self.vi_cmd = True
                self.pos = max(0, self.pos - 1)
                return None
            r = self.emacs_key(k) if k not in ("C-a", "C-e") else None
            return r if r in ("accept", "eof", "abort", "clear") else None
        # command mode
        if k in ("Enter", "C-m", "C-j"):
            return "accept"
        if k == "C-c":
            return "abort"
        if k == "Space":
            k = "l"
        ch = k if len(k) == 1 else None
        b, p, n = self.buf, self.pos, len(self.buf)
        if self.vi_pending:
            op = self.vi_pending + (ch or "")
            self.vi_pending = ""
            if op == "dd" or op == "cc":
                self.kill(0, n, False)
                if op == "cc":
                    self.vi_cmd = False
            elif op in ("dw", "cw"):
                e = self.word_fwd(p) if op == "dw" else self._vi_word_end(p) + 1
                if op == "dw":
                    while e < n and b[e] == " ":
                        e += 1
                self.kill(p, e, False)
                if op == "cw":
                    self.vi_cmd = False
            elif op in ("db", "cb"):
                self.kill(self.word_back(p), p, True)
                if op == "cb":
                    self.vi_cmd = False
            elif op in ("d$", "c$"):
                self.kill(p, n, False)
                if op == "c$":
                    self.vi_cmd = False
            elif op in ("d0", "c0"):
                self.kill(0, p, True)
                if op == "c0":
                    self.vi_cmd = False
            elif op.startswith("r") and len(op) == 2 and p < n:
                self.push_undo()
                self.set(b[:p] + op[1] + b[p + 1:], p)
            elif op.startswith(("f", "F", "t", "T")) and len(op) == 2:
                self._vi_find(op[0], op[1])
            elif op == "yy":
                self.kill_ring.append(b)
            return None
        if ch in ("d", "c", "r", "f", "F", "t", "T", "y"):
            self.vi_pending = ch
            return None
        if ch == "h" or k == "Left":
            self.pos = max(0, p - 1)
        elif ch == "l" or k == "Right":
            self.pos = min(max(n - 1, 0), p + 1)
        elif ch == "0" or k == "Home":
            self.pos = 0
        elif ch == "^":
            self.pos = len(b) - len(b.lstrip())
        elif ch == "$" or k == "End":
            self.pos = max(0, n - 1)
        elif ch == "w":
            q = self.word_fwd(p)
            while q < n and not _is_word(b[q]) and b[q] == " ":
                q += 1
            self.pos = min(q, max(n - 1, 0))
        elif ch == "b":
            self.pos = self.word_back(p)
        elif ch == "e":
            self.pos = self._vi_word_end(p + 1 if p + 1 < n else p)
        elif ch == "x":
            if p < n:
                self.kill(p, p + 1, False)
                self.pos = min(self.pos, max(len(self.buf) - 1, 0))
        elif ch == "X":
            if p > 0:
                self.kill(p - 1, p, True)
        elif ch == "D":
            self.kill(p, n, False)
            self.pos = max(0, len(self.buf) - 1)
        elif ch == "C":
            self.kill(p, n, False)
            self.vi_cmd = False
        elif ch == "i":
            self.vi_cmd = False
        elif ch == "a":
            self.pos = min(n, p + 1)
            self.vi_cmd = False
        elif ch == "I":
            self.pos = 0
            self.vi_cmd = False
        elif ch == "A":
            self.pos = n
            self.vi_cmd = False
        elif ch == "s":
            if p < n:
                self.kill(p, p + 1, False)
            self.vi_cmd = False
        elif ch == "S":
            self.kill(0, n, False)
            self.vi_cmd = False
        elif ch == "p":
            if self.kill_ring:
                self.pos = min(n, p + 1)
                self.insert(self.kill_ring[-1])
                self.pos -= 1
        elif ch == "P":
            if self.kill_ring:
                self.insert(self.kill_ring[-1])
                self.pos -= 1
        elif ch == "u":
            if self.undo:
                self.buf, self.pos = self.undo.pop()
        elif ch == "~":
            if p < n:
                self.push_undo()
                self.set(b[:p] + b[p].swapcase() + b[p + 1:], p + 1)
        elif ch in ("k", "-") or k == "Up":
            self.hist_move(-1)
            self.pos = 0
        elif ch in ("j", "+") or k == "Down":
            self.hist_move(1)
            self.pos = 0
        elif ch == "/" or ch == "?":
            self.search = {"q": "", "dir": -1, "idx": len(self.history), "at": None, "orig": (b, p), "failed": False}
        elif ch == "n":
            if self.sim.last_search:
                st = {"q": self.sim.last_search, "dir": -1, "idx": self.hist_pos if self.hist_pos is not None else len(self.history),
                      "at": None, "orig": (b, p), "failed": False}
                self._search_find(st, same=False)
        elif k == "Tab":
            self.sim.complete(self, listing=False)
        return None

    def _vi_word_end(self, p):
        b, n = self.buf, len(self.buf)
        while p < n and not _is_word(b[p]):
            p += 1
        while p + 1 < n and _is_word(b[p + 1]):
            p += 1
        return min(p, max(n - 1, 0))

    def _vi_find(self, how, c):
        b, p = self.buf, self.pos
        if how in ("f", "t"):
            j = b.find(c, p + 1)
            if j >= 0:
                self.pos = j if how == "f" else j - 1
        else:
            j = b.rfind(c, 0, p)
            if j >= 0:
                self.pos = j if how == "F" else j + 1


# ---------------------------------------------------------------------------- the simulator

class KeySim:
    def __init__(self, world, sess, spec_extra):
        self.world, self.sess = world, sess
        self.tty = world.tty
        self.ed = Editor(self)
        self.last_search = ""
        self.ran: list[str] = []
        self.trace: list[str] = []
        self.ended = False
        self.tmux = None            # (server, session) the client is attached to
        self.tmux_prompt = None     # (what, text) for C-b , / C-b $ / C-b x
        self.tmux_prefix = False
        self.tmux_editors: dict = {}
        buf = spec_extra.get("buffer", "")
        self.ed.set(buf, spec_extra.get("cursor"))

    # ---- terminal output
    def out(self, stream, text):
        self.tty.write(stream, text.encode("utf-8", "surrogateescape"))

    def prompt(self):
        sh = self.cur_shell()
        return sh.prompt_expand(sh.get("PS1", "") or "") if sh is not None else "$ "

    def cur_shell(self):
        if self.tmux is not None:
            pane = self.tmux[1].pane()
            return getattr(pane.proc, "shell", None)
        return self.sess.sh

    def history(self):
        sh = self.cur_shell()
        return sh.history if sh is not None else []

    # ---- running a line
    def run_line(self, line, interrupt=None):
        if self.tmux is not None:          # the pane shows its own prompt and the typed line
            return self._pane_line(line, interrupt)
        self.out("cmd", self.prompt() + line + "\n")
        sh = self.sess.sh
        before = len(sh.typed_log)
        if interrupt:
            self._schedule(interrupt)
        tty = self.tty
        res = self.sess.run(line)
        tty.out.extend([list(x) for x in res["segments"]])
        self.ran += [x for x in sh.typed_log[before:]]
        if not self.sess.alive:
            self.ended = True
        self._check_tmux_attach(line)

    def _schedule(self, which):
        k = self.world.kernel
        tty = self.tty
        sig = 2 if which == "C-c" else 20

        def fire():
            fg = tty.fg_pgid
            sh = self.sess.sh
            if not fg or fg == sh.proc.pgid:
                return
            tty.write("echo", b"^C" if sig == 2 else b"^Z")
            for p in list(self.world.local.procs.values()):
                if p.pgid == fg and p.state != "Z":
                    k.signal(p, sig)
        k.add_timer(k.now + 1_000_000_000, fire)     # a second after the command starts

    # ---- tmux
    def _check_tmux_attach(self, line):
        srv = self.world.local.extra.get("tmux")
        if srv is None:
            return
        try:
            words = shlex.split(line)
        except ValueError:
            return
        if not words or words[0] != "tmux":
            return
        sub = words[1] if len(words) > 1 else "new-session"
        if sub in ("new", "new-session", "a", "at", "attach", "attach-session") and "-d" not in words:
            target = None
            if "-t" in words:
                target = words[words.index("-t") + 1] if words.index("-t") + 1 < len(words) else None
            if "-s" in words and sub.startswith("new"):
                target = words[words.index("-s") + 1]
            from .utils.tmux import find_session
            s = find_session(srv, target) if sub.startswith(("a", "at")) or target else \
                (list(srv.sessions.values())[-1] if srv.sessions else None)
            if s is not None and s.windows:
                self.enter_tmux(srv, s)

    def enter_tmux(self, srv, s):
        self.tmux_editors["outer"] = self.ed
        self.tmux = (srv, s)
        s.attached += 1
        self._load_editor()

    def _save_editor(self):
        key = id(self.tmux[1].pane()) if self.tmux else "outer"
        self.tmux_editors[key] = self.ed

    def _load_editor(self):
        key = id(self.tmux[1].pane()) if self.tmux else "outer"
        self.ed = self.tmux_editors.get(key) or Editor(self)
        self.tmux_editors[key] = self.ed

    def _pane_line(self, line, interrupt=None):
        srv, s = self.tmux
        pane = s.pane()
        tty = pane.tty
        sh = getattr(pane.proc, "shell", None)
        before = len(sh.typed_log) if sh is not None else 0
        mark = len(tty.out)
        tty.lines.append(line)
        k = self.world.kernel
        k.wake(tty)
        if pane.proc.state == "S" and pane.proc.waiting and pane.proc.waiting[0] == "tty":
            pane.proc.state = "R"
            k.runq.append(pane.proc)
        if interrupt:
            self._schedule(interrupt)
        p = pane.proc

        def settled():
            return p.state == "Z" or (p.state == "S" and p.waiting and p.waiting[0] == "tty" and not tty.lines)
        try:
            k.run(settled)
        except Exception as e:  # noqa: BLE001
            self.out("err", f"{type(e).__name__}\n")
        for stream, text in tty.out[mark:]:
            self.out(stream, text)
        if sh is not None:
            self.ran += sh.typed_log[before:]
        if s.name not in srv.sessions or not s.windows:
            self.out("err", "[exited]\n")
            self.leave_tmux()
        elif s.detach_requested:
            s.detach_requested = False
            self.out("err", f"[detached (from session {s.name})]\n")
            self.leave_tmux()
        else:
            self._load_editor()

    def leave_tmux(self):
        if self.tmux is not None:
            self.tmux[1].attached = max(0, self.tmux[1].attached - 1)
        self.tmux = None
        self.ed = self.tmux_editors.get("outer") or Editor(self)

    def tmux_cmd(self, *args):
        """Run a tmux command for the attached client (quietly)."""
        srv, s = self.tmux
        from .exec import exec_program
        k = self.world.kernel
        host = self.world.local
        env = dict(self.sess.sh.env())
        env.pop("TMUX", None)
        p = k.spawn(host, ["tmux", *args], None, self.sess.proc, fds={}, cwd=s.pane().proc.cwd)
        k.start(p, exec_program(p, "/usr/bin/tmux", ["tmux", *args], env, None))
        k.run(lambda: p.state == "Z")

    def tmux_key(self, k):
        srv, s = self.tmux
        name = s.name
        if k in ("d", "D"):
            s.attached = max(0, s.attached - 1)
            self.out("err", f"[detached (from session {name})]\n")
            self._save_editor()
            self.tmux = None
            self.ed = self.tmux_editors.get("outer") or Editor(self)
            return
        self._save_editor()
        if k == "c":
            self.tmux_cmd("new-window", "-t", name)
        elif k == "%":
            self.tmux_cmd("split-window", "-h", "-t", name)
        elif k == '"':
            self.tmux_cmd("split-window", "-v", "-t", name)
        elif k == "n":
            self.tmux_cmd("next-window", "-t", name)
        elif k == "p":
            self.tmux_cmd("previous-window", "-t", name)
        elif k == "l":
            pass
        elif len(k) == 1 and k.isdigit():
            self.tmux_cmd("select-window", "-t", f"{name}:{k}")
        elif k in ("o", "Right", "Down"):
            self.tmux_cmd("select-pane", "-t", name)
        elif k in ("Left", "Up", ";"):
            self.tmux_cmd("select-pane", "-L", "-t", name)
        elif k == "z":
            self.tmux_cmd("resize-pane", "-Z", "-t", name)
        elif k == "x":
            w = s.windows[s.active]
            self.tmux_prompt = ("kill-pane", "", f"kill-pane {w.active}? (y/n)")
        elif k == "&":
            w = s.windows[s.active]
            self.tmux_prompt = ("kill-window", "", f"kill-window {w.name}? (y/n)")
        elif k == ",":
            self.tmux_prompt = ("rename-window", s.windows[s.active].name, "(rename-window)")
        elif k == "$":
            self.tmux_prompt = ("rename-session", s.name, "(rename-session)")
        elif k == "[":
            self.tmux_prompt = ("copy-mode", "", "")
        elif k == "?":
            self.out("out", "C-b ?  list-keys\nC-b c  new-window\nC-b d  detach-client\nC-b n  next-window\n"
                            "C-b p  previous-window\nC-b %  split-window -h\nC-b \"  split-window\nC-b z  resize-pane -Z\n"
                            "C-b x  kill-pane\nC-b ,  rename-window\nC-b [  copy-mode\n")
        if self.tmux is not None:
            if s.name not in srv.sessions or not s.windows:
                self.out("err", "[exited]\n")
                self.leave_tmux()
                return
            self._load_editor()

    def tmux_prompt_key(self, kind, val):
        what, text, label = self.tmux_prompt
        srv, s = self.tmux
        if what == "copy-mode":
            if (kind == "text" and val.startswith("q")) or val in ("Esc", "C-c"):
                self.tmux_prompt = None
            return
        if what in ("kill-pane", "kill-window"):
            self.tmux_prompt = None
            if kind == "text" and val[:1].lower() == "y":
                self.tmux_cmd(what, "-t", s.name)
                if s.name not in srv.sessions or not s.windows:
                    self.out("err", "[exited]\n")
                    self.leave_tmux()
                else:
                    self._load_editor()
            return
        # rename prompts: the old name is pre-filled; typing edits it, Enter applies
        if kind == "text":
            self.tmux_prompt = (what, text + val, label)
        elif val == "Space":
            self.tmux_prompt = (what, text + " ", label)
        elif val in ("BSpace", "C-h"):
            self.tmux_prompt = (what, text[:-1], label)
        elif val in ("C-u",):
            self.tmux_prompt = (what, "", label)
        elif val == "C-w":
            self.tmux_prompt = (what, re.sub(r"\S*\s*$", "", text), label)
        elif val in ("Esc", "C-c", "C-g"):
            self.tmux_prompt = None
        elif val == "Enter":
            self.tmux_prompt = None
            if text:
                self.tmux_cmd(what, "-t", s.name, text)

    # ---- completion
    def complete(self, ed, listing=False, only_list=False):
        b, p = ed.buf, ed.pos
        start = p
        while start > 0 and not (b[start - 1] in " \t\n;|&<>(" and (start < 2 or b[start - 2] != "\\")):
            start -= 1
        word = b[start:p]
        before = b[:start].rstrip()
        first = not before or before[-1] in ";|&(" or before.split()[-1:] in (["sudo"], ["time"], ["command"], ["exec"],
                                                                                ["nohup"], ["watch"], ["xargs"])
        sh = self.cur_shell()
        fs = self.world.local.fs
        cwd = sh.proc.cwd if sh is not None else "/"
        cands = []
        if word.startswith("$"):
            pre = word[1:].lstrip("{")
            names = sorted(set(list(sh.globals)) | set(sh.env())) if sh is not None else []
            cands = ["$" + n for n in names if n.startswith(pre)]
            files_mode = False
        elif first and "/" not in word:
            names = set()
            if sh is not None:
                from .builtins import BUILTINS
                names |= set(BUILTINS) | set(sh.funcs) | set(sh.aliases)
                names |= {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
                          "function", "select", "time", "in"}
                for d in (sh.get("PATH", "") or "").split(":"):
                    try:
                        node = fs.lookup(d or ".", cwd)
                    except Exception:  # noqa: BLE001
                        continue
                    if node.type == "d":
                        for nme in fs.listdir(node):
                            try:
                                ch = fs.lookup(f"{d}/{nme}", cwd)
                            except Exception:  # noqa: BLE001
                                continue
                            if ch.type == "f" and ch.mode & 0o111:
                                names.add(nme)
            cands = sorted(n for n in names if n.startswith(word))
            files_mode = False
        else:
            files_mode = True
            raw = word.replace("\\ ", " ")
            home = sh.get("HOME", "/home/ada") if sh is not None else "/home/ada"
            exp = home + raw[1:] if raw.startswith("~") else raw
            d, _, pre = exp.rpartition("/")
            shown_dir = raw[:len(raw) - len(pre)]
            base = (d or "/") if exp.startswith("/") or "/" in exp else "."
            if "/" in exp and not exp.startswith("/"):
                base = d
            try:
                node = fs.lookup(base, cwd)
                names = fs.listdir(node) if node.type == "d" else []
            except Exception:  # noqa: BLE001
                names = []
            cands = []
            for nme in names:
                if not nme.startswith(pre) or (nme.startswith(".") and not pre.startswith(".")):
                    continue
                full = (base.rstrip("/") + "/" + nme) if base != "." else nme
                try:
                    is_dir = fs.lookup(full, cwd).type == "d"
                except Exception:  # noqa: BLE001
                    is_dir = False
                cands.append((shown_dir + nme, is_dir))
            cands.sort()
        if not cands:
            return
        if files_mode:
            names = [c for c, _ in cands]
        else:
            names = cands
        if len(names) == 1 and not only_list:
            c = names[0]
            is_dir = files_mode and cands[0][1]
            esc = c.replace(" ", "\\ ") if files_mode else c
            add = "/" if is_dir else " "
            if is_dir and b[p:p + 1] == "/":
                add = ""
            ed.push_undo()
            ed.set(b[:start] + esc + add + b[p:], start + len(esc) + len(add))
            return
        common = names[0]
        for c in names[1:]:
            while not c.startswith(common):
                common = common[:-1]
        if len(common) > len(word.replace("\\ ", " ")) and not only_list:
            esc = common.replace(" ", "\\ ") if files_mode else common
            ed.push_undo()
            ed.set(b[:start] + esc + b[p:], start + len(esc))
            return
        if listing:
            shown = []
            for k_, c in enumerate(names):
                leaf = c.rsplit("/", 1)[-1] if files_mode and "/" in c.rstrip("/") else c
                if files_mode and cands[k_][1]:
                    leaf += "/"
                shown.append(leaf)
            self.out("cmd", self.prompt() + ed.buf + "\n")
            self.out("out", columns(shown, self.tty.width))

    # ---- driver
    def press(self, kind, val):
        if self.ended:
            return
        self.trace.append(val if kind == "key" else repr(val))
        if self.tmux is not None and self.tmux_prompt is not None:
            self.tmux_prompt_key(kind, val)
            return
        if self.tmux is not None and self.tmux_prefix:
            self.tmux_prefix = False
            ch = val if kind == "key" else val[:1]
            self.tmux_key(ch)
            if kind == "text" and len(val) > 1:
                self.press("text", val[1:])
            return
        if self.tmux is not None and kind == "key" and val == "C-b":
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
            interrupt = self._peek_interrupt()
            self.run_line(line, interrupt)
            self._sync_mode()
        elif r == "comment":
            line = ed.buf
            self.out("cmd", self.prompt() + line + "\n")
            sh = self.cur_shell()
            if sh is not None:
                sh.add_history(line)
            self._new_line()
        elif r == "eof":
            self.out("cmd", self.prompt() + "\n")
            self.out("out", "exit\n" if self.tmux is None else "")
            self._new_line()
            if self.tmux is not None:
                self._pane_line("exit")
            else:
                self.sess.run("exit", quiet=True, history=False)
                self.ended = not self.sess.alive
        elif r == "abort":
            self.out("cmd", self.prompt() + ed.buf + "^C\n")
            self._new_line()
            sh = self.cur_shell()
            if sh is not None:
                sh.status = 130
        elif r == "clear":
            self.out("out", "\x1b[H\x1b[2J")

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

    def _peek_interrupt(self):
        if self.queue and self.queue[0] == ("key", "C-c") or self.queue and self.queue[0] == ("key", "C-z"):
            return self.queue.pop(0)[1]
        return None

    def play(self, keys):
        self.queue = list(keys)
        self._sync_mode()
        while self.queue:
            kind, val = self.queue.pop(0)
            self.press(kind, val)
            if self.ended:
                break

    def state(self) -> dict:
        tm = ""
        srv = self.world.local.extra.get("tmux")
        if srv is not None and srv.sessions:
            parts = []
            for s in srv.sessions.values():
                wins = []
                for k_, w in enumerate(s.windows):
                    panes = ",".join(f"{p.w}x{p.h}" + ("*" if j == w.active else "") for j, p in enumerate(w.panes))
                    wins.append(f"{w.index}:{w.name}{'*' if k_ == s.active else ''}[{panes}]{'Z' if w.zoomed else ''}")
                parts.append(f"{s.name}: " + " ".join(wins))
            tm = "; ".join(parts) + (f" | attached to {self.tmux[1].name}" if self.tmux else " | detached")
        return {"ran": [r.strip() for r in self.ran], "buffer": self.ed.buf, "tmux": tm}


def columns(items, width=80):
    """readline's match list: sorted, down then across."""
    if not items:
        return ""
    w = max(len(x) for x in items) + 2
    cols = max(1, (width + 2) // w) if w < width else 1
    rows = -(-len(items) // cols)
    lines = []
    for r in range(rows):
        row = [items[c * rows + r] for c in range(cols) if c * rows + r < len(items)]
        lines.append("".join(x.ljust(w) for x in row[:-1]) + row[-1])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- run / grade

def _extra(setup_dirs):
    out = {}
    for d in setup_dirs:
        name, _, arg = d.partition(" ")
        if name == "buffer":
            out["buffer"] = arg.replace("\\n", "\n")
        elif name == "cursor":
            out["cursor"] = int(arg)
        elif name == "tmux":
            out["tmux"] = arg or "main"
    return out


def run_keys(problem, keys_text, setup):
    from .check import SEED, SetupError
    from .directives import build, parse, prepare, split, start_session
    dirs, code = split(setup)
    extra = _extra(dirs)
    spec = parse([d for d in dirs if d.partition(" ")[0] not in ("buffer", "cursor", "tmux")])
    world, sess = build(spec, seed=SEED + int(problem.get("seed", 0)) % 1000)
    prepare(sess, spec)
    if code.strip():
        r = sess.run(code, quiet=True, history=False)
        if r["error"]:
            raise SetupError(r["error"])
    start_session(sess, spec)
    keys = parse_keys(keys_text)
    sim = KeySim(world, sess, extra)
    tty = world.tty
    saved, tty.out, tty.size = tty.out, [], 0
    if spec.keys == "tmux" or "tmux" in extra:
        name = extra.get("tmux", "main")
        sess.run(f"tmux new -d -s {shlex.quote(name)}", quiet=True, history=False)
        srv = world.local.extra.get("tmux")
        if srv is not None and name in srv.sessions:
            sim.enter_tmux(srv, srv.sessions[name])
        tty.out, tty.size = [], 0
    from .kernel import OutputLimit, TimeLimit, Deadlock
    error = None
    try:
        sim.play(keys)
    except (OutputLimit, TimeLimit, Deadlock) as e:
        error = {"OutputLimit": "Output limit exceeded.", "TimeLimit": "Time limit exceeded."}.get(type(e).__name__, str(e))
    if not sim.ended:
        ed = sim.ed
        if sim.tmux_prompt is not None and sim.tmux_prompt[2]:
            sim.out("cmd", f"{sim.tmux_prompt[2]} {sim.tmux_prompt[1]}\n")
        elif ed.search is not None:
            st = ed.search
            sim.out("cmd", f"({'failed ' if st['failed'] else ''}reverse-i-search)`{st['q']}': {ed.buf}\n")
        else:
            sim.out("cmd", sim.prompt() + ed.buf[:ed.pos] + "▁" + ed.buf[ed.pos:] + "\n")
    segs = [[s, t] for s, t in tty.out]
    tty.out, tty.size = saved, 0
    st = sim.state()
    return {"state": st, "console": segs, "error": error, "trace": sim.trace,
            "stdout": "".join(t for s, t in segs if s == "out")}


def compare_state(exp, got) -> str | None:
    if got["ran"] != exp["ran"]:
        if not got["ran"]:
            return "Your keys don't run any command." if exp["ran"] else None
        ran = ", ".join(f"`{x}`" for x in got["ran"][:4])
        return (f"Your keys ran {ran}" + (" and more" if len(got["ran"]) > 4 else "") +
                ("; that's not what the task asks for." if exp["ran"] else ", but nothing should be run yet."))
    if got["buffer"] != exp["buffer"]:
        return f"Your keys leave `{got['buffer']}` on the command line." if got["buffer"] else \
            "Your keys leave the command line empty."
    if got["tmux"] != exp["tmux"]:
        return f"tmux ends up as: {got['tmux'] or 'no sessions'}."
    return None


def checker(problem, code, grade):
    from .check import SetupError
    from .directives import DirectiveError
    try:
        if not grade:
            r = run_keys(problem, code, problem.get("setup", ""))
            return {"ok": r["error"] is None, "stdout": r["stdout"], "value": None, "error": r["error"],
                    "console": r["console"]}
        out = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": "", "console": []}
        try:
            parse_keys(code)
        except KeyError_ as e:
            out["compiled"] = False
            out["error"] = str(e)
            return out
        have = key_features(code)
        for req in problem.get("require") or []:
            if not any(a in have for a in req.split("|")):
                out["message"] = "This problem asks you to use " + " or ".join(f"`{a}`" for a in req.split("|")) + "."
                return out
        for f in problem.get("forbid") or []:
            if f in have:
                out["message"] = f"Solve this one without `{f}`." if f != "text" else "Solve this one without typing text."
                return out
        setups = [problem.get("setup", "")] + list(problem.get("tests") or [])
        for i, setup in enumerate(setups):
            exp = run_keys(problem, problem["solution"], setup)
            got = run_keys(problem, code, setup)
            if i == 0:
                out["stdout"], out["console"] = got["stdout"], got["console"]
            if got["error"]:
                out["error"] = got["error"]
                return out
            msg = compare_state(exp["state"], got["state"])
            if msg:
                out["message"] = msg if i == 0 else f"Wrong answer on hidden test {i}."
                return out
        out["correct"] = True
        return out
    except (SetupError, DirectiveError) as e:
        return {"ok": False, "correct": False, "compiled": True, "stdout": "", "value": None,
                "error": f"The problem's setup failed: {e}", "message": "", "console": []}

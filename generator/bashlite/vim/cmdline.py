"""The command line: Ex commands after ":", search patterns after "/" and "?", expressions after "=" ."""

from __future__ import annotations

import re

from . import keys as K
from .engine import VimError
from .motions import keyword_at


class CmdlineMixin:

    def start_cmdline(self, kind, text="", from_op=None, count=None, expr_for=None, cmdwin=False):
        self.cmdl = {"kind": kind, "text": text, "pos": len(text), "from_op": from_op, "count": count,
                     "hist": None, "prefix": None, "save_cursor": tuple(self.win.cursor), "save_top": self.win.topline,
                     "expr_for": expr_for, "ctrl_r": False, "ctrl_v": False, "compl": None,
                     "vis": None}
        self.prev_mode = self.mode
        self.mode = "c"
        if kind in "/?":
            self.used[kind] += 1

    def cmdline_key(self, key):
        c = self.cmdl
        if c["ctrl_r"]:
            c["ctrl_r"] = False
            if key == "<C-w>":
                l, col = self.win.cursor
                kw = keyword_at(self.lines[l], col)
                ins = self.lines[l][kw[0]:kw[1]] if kw else ""
            elif key == "<C-a>":
                l, col = self.win.cursor
                m = re.search(r"\S+", self.lines[l][col:])
                ins = m.group(0) if m else ""
            elif len(key) == 1:
                ins = self.reg_get(key)[0]
                if ins.endswith("\n"):
                    ins = ins[:-1]
                ins = ins.replace("\n", "\r")
            else:
                ins = ""
            self.used["c_<C-r>"] += 1
            return self._cl_insert(ins)
        if c["ctrl_v"]:
            c["ctrl_v"] = False
            return self._cl_insert(key if len(key) == 1 else K.to_text([key]))
        if key != "<Tab>":
            c["compl"] = None
        if key in ("<Esc>", "<C-c>"):
            return self.cancel_cmdline()
        if key in ("<CR>", "<NL>"):
            return self.exec_cmdline()
        if key in ("<BS>",):
            if not c["text"]:
                return self.cancel_cmdline()
            if c["pos"] > 0:
                c["text"] = c["text"][:c["pos"] - 1] + c["text"][c["pos"]:]
                c["pos"] -= 1
            return self.incsearch()
        if key == "<Del>":
            c["text"] = c["text"][:c["pos"]] + c["text"][c["pos"] + 1:]
            return
        if key == "<C-u>":
            c["text"] = c["text"][c["pos"]:]
            c["pos"] = 0
            return self.incsearch()
        if key == "<C-w>":
            t = c["text"][:c["pos"]]
            k = len(t)
            while k > 0 and t[k - 1] == " ":
                k -= 1
            if k > 0 and re.match(r"\w", t[k - 1]):
                while k > 0 and re.match(r"\w", t[k - 1]):
                    k -= 1
            elif k > 0:
                k -= 1
            c["text"] = t[:k] + c["text"][c["pos"]:]
            c["pos"] = k
            return self.incsearch()
        if key in ("<Left>",):
            c["pos"] = max(0, c["pos"] - 1)
            return
        if key in ("<Right>",):
            c["pos"] = min(len(c["text"]), c["pos"] + 1)
            return
        if key in ("<Home>", "<C-b>"):
            c["pos"] = 0
            return
        if key in ("<End>", "<C-e>"):
            c["pos"] = len(c["text"])
            return
        if key in ("<Up>", "<Down>", "<S-Up>", "<S-Down>", "<C-p>", "<C-n>"):
            return self._cl_history(-1 if key in ("<Up>", "<C-p>", "<S-Up>") else 1, prefix=key in ("<Up>", "<Down>"))
        if key == "<C-r>":
            c["ctrl_r"] = True
            return
        if key == "<C-v>":
            c["ctrl_v"] = True
            return
        if key == "<Tab>" and c["kind"] == ":":
            return self._cl_complete()
        if len(key) == 1:
            self._cl_insert(key)
            return self.incsearch()
        return

    def _cl_insert(self, s):
        c = self.cmdl
        c["text"] = c["text"][:c["pos"]] + s + c["text"][c["pos"]:]
        c["pos"] += len(s)

    def _cl_history(self, d, prefix=True):
        c = self.cmdl
        h = self.hist[":" if c["kind"] == ":" else "/"]
        if not h:
            return
        if c["hist"] is None:
            c["hist"] = len(h)
            c["prefix"] = c["text"] if prefix else ""
            c["typed"] = c["text"]
        i = c["hist"]
        while True:
            i += d
            if i < 0:
                return
            if i >= len(h):
                c["hist"] = len(h)
                c["text"] = c["typed"]
                c["pos"] = len(c["text"])
                return
            if h[i].startswith(c["prefix"]):
                break
        c["hist"] = i
        c["text"] = h[i]
        c["pos"] = len(c["text"])

    def _cl_complete(self):
        """Tab: complete a command name, an option, or a file name."""
        c = self.cmdl
        text = c["text"][:c["pos"]]
        if c["compl"] is not None:
            cp = c["compl"]
            cp["i"] = (cp["i"] + 1) % len(cp["items"])
            c["text"] = cp["base"] + cp["items"][cp["i"]] + c["text"][c["pos"]:]
            c["pos"] = len(cp["base"]) + len(cp["items"][cp["i"]])
            return
        m = re.match(r"^(.*\b(?:e|edit|w|write|sp|split|vs|vsplit|vsp|tabe|tabnew|tabedit|r|read|sav|saveas|new|vnew|"
                     r"so|source|b|buffer|sb|cd|lcd)!?\s+)(\S*)$", text)
        items = []
        if m:
            base, part = m.group(1), m.group(2)
            if re.match(r".*\bb(uffer)?!?\s+$", base) or re.match(r".*\bsb\s+$", base):
                items = [b.name for b in self.buffers.values() if b.listed and part in b.name]
            else:
                items = self.io.complete(self.cwd, part)
        else:
            m = re.match(r"^(.*\bset?\s+(?:no|inv)?)(\w*)$", text)
            if m:
                from .engine import OPTIONS
                base, part = m.group(1), m.group(2)
                items = sorted(n for n in OPTIONS if n.startswith(part))
            else:
                m = re.match(r"^(\s*(?:\S*,)?\S*?)([A-Za-z]*)$", text)
                if m:
                    from .ex import COMMAND_NAMES
                    base, part = m.group(1), m.group(2)
                    items = sorted(n for n in COMMAND_NAMES if n.startswith(part))
        if not items:
            return
        c["compl"] = {"base": base, "items": items, "i": 0}
        c["text"] = base + items[0] + c["text"][c["pos"]:]
        c["pos"] = len(base) + len(items[0])

    def incsearch(self):
        c = self.cmdl
        if c["kind"] not in "/?" or not self.opt("incsearch"):
            return
        self.win.cursor = list(c["save_cursor"])
        self.win.topline = c["save_top"]
        pat = c["text"]
        if not pat:
            c["inc"] = None
            return
        from .search import search
        try:
            pos = search(self, pat.split("/")[0] if c["kind"] == "/" else pat.split("?")[0], 1 if c["kind"] == "/" else -1,
                         c["count"] or 1, c["save_cursor"], show=False)
        except VimError:
            c["inc"] = None
            return
        c["inc"] = pos
        self.win.cursor = [pos[0], pos[1]]
        self.scroll_to_cursor()
        self.win.cursor = list(c["save_cursor"])

    def cancel_cmdline(self):
        c = self.cmdl
        self.mode = "n" if self.prev_mode not in ("i", "R") else self.prev_mode
        self.win.cursor = list(c["save_cursor"])
        self.win.topline = c["save_top"]
        self.cmdl = None
        if c.get("expr_for") == "insert":
            self.mode = "i"
        self.message("")

    def exec_cmdline(self):
        c = self.cmdl
        self.cmdl = None
        self.mode = "n"
        text = c["text"]
        kind = c["kind"]
        self.win.cursor = list(c["save_cursor"])
        if kind == "=":
            from .ex import eval_expr
            try:
                val = eval_expr(self, text)
            except VimError:
                val = ""
            self.regs["="] = (str(val), "v")
            if c.get("expr_for") == "insert":
                self.mode = "i"
                self.ins_literal(str(val))
            return
        if kind == ":":
            from .ex import run_ex
            self.message("")
            if text.strip():
                self.add_hist(":", text)
            return run_ex(self, text, typed=True)
        # a search
        from .search import search
        d = 1 if kind == "/" else -1
        if text == "":
            if self.last_pat is None:
                raise VimError("E35: No previous regular expression")
            pat, off = self.last_pat, self.last_off
        else:
            pat, off = split_search(text, kind)
            if pat == "":
                pat = self.last_pat
                if pat is None:
                    raise VimError("E35: No previous regular expression")
            self.add_hist("/", text)
        self.last_pat = pat
        self.last_dir = d
        self.last_off = off
        self.regs["/"] = (pat, "v")
        self.no_hl = False
        for tok in ("\\v", "\\<", "\\>", "\\c", "\\C", "\\zs"):
            if tok in pat:
                self.used[tok] += 1
        cmd = {"key": kind, "arg": None, "n": c["count"], "reg": None}
        op_cmd = c["from_op"]
        start = tuple(self.win.cursor)
        if off and off[:1] == "e" and False:
            pass
        pos = search(self, pat, d, c["count"] or 1, start, off=off, show=True,
                     echo=(kind + text) if text else None)
        if op_cmd is not None:
            # d/pattern: the operator up to the match (exclusive), or with a line offset linewise
            linewise = bool(off) and off[:1] in "+-0123456789"
            incl = bool(off) and off[:1] == "e"
            s = start
            e = pos
            if e < s:
                s, e = e, s
            self.setpcmark()
            self.used[(op_cmd["op"] or "") + kind] += 1
            if linewise:
                return self.apply_op(op_cmd["op"], (s[0], 0), (e[0], 0), "line", {**op_cmd, "key": kind}, None, start)
            if incl:
                return self.apply_op(op_cmd["op"], s, e, "incl", {**op_cmd, "key": kind}, None, start)
            fake = {**op_cmd, "key": "/", "arg": None}
            # exclusive: same rules as any exclusive motion
            self.win.cursor = list(s)
            if e == s:
                return
            if e[1] == 0 and e[0] > s[0]:
                from .text import first_nonblank
                if s[1] <= first_nonblank(self.lines[s[0]]):
                    return self.apply_op(op_cmd["op"], (s[0], 0), (e[0] - 1, 0), "line", fake, None, start)
                prev = e[0] - 1
                return self.apply_op(op_cmd["op"], s, (prev, len(self.lines[prev])), "incl", fake, None, start)
            return self.apply_op(op_cmd["op"], s, (e[0], e[1] - 1), "incl", fake, None, start)
        if self.vis is not None:
            self.win.cursor = [pos[0], pos[1]]
            return
        self.setpcmark()
        self.win.cursor = [pos[0], pos[1]]
        self.clamp()
        from .text import vcol
        self.win.curswant = vcol(self.lines[pos[0]], pos[1], self.ts())


def split_search(text, kind):
    """"pat/e+1" -> ("pat", "e+1"): the pattern ends at an unescaped / (or ? for ?)."""
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == kind:
            return text[:i], text[i + 1:]
        if text[i] == "[":
            j = text.find("]", i + 2)
            if j > 0:
                i = j + 1
                continue
        i += 1
    return text, ""

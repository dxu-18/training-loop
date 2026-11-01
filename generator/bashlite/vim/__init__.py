"""Vim for bash-lite: a Vim 9.1 that runs in Python, key by key (see engine.py)."""

from __future__ import annotations

from collections import deque

from . import keys as K
from .cmdline import CmdlineMixin
from .engine import MAXCOL, Core, VimError
from .insert import InsertMixin
from .motions import Fail
from .screen import ScreenMixin
from .visual import VisualMixin
from .windows import WindowMixin


class Vim(InsertMixin, VisualMixin, CmdlineMixin, WindowMixin, ScreenMixin, Core):

    def search(self, pat, d, count, start, off="", wrap=None, smartcase=True, show=True, echo=None):
        from .search import search
        return search(self, pat, d, count, start, off=off, wrap=wrap, smartcase=smartcase, show=show, echo=echo)

    # ------------------------------------------------------------ keys run by :normal and :g
    def run_keys(self, toks, remap=True):
        """Execute keys as if typed (for :normal); an unfinished command is abandoned at the end, like Vim."""
        saved = self.typeahead
        self.typeahead = deque((k, remap, None) for k in toks)
        guard = 0
        try:
            while self.typeahead and self.quit is None:
                guard += 1
                if guard > 200_000:
                    break
                key, rm, t = self.typeahead.popleft()
                mm = self.map_mode()
                if rm and mm and self.maps[mm]:
                    seq = tuple(k for k, _ in self.map_wait) + (key,)
                    longer = [lhs for lhs in self.maps[mm] if len(lhs) > len(seq) and lhs[:len(seq)] == seq]
                    exact = self.maps[mm].get(seq)
                    if longer and self.typeahead:
                        self.map_wait.append((key, 0))
                        continue
                    if exact is not None:
                        self.map_wait = []
                        self.expand_map(exact)
                        continue
                    if self.map_wait:
                        pending = [k for k, _ in self.map_wait]
                        self.map_wait = []
                        self.typeahead.appendleft((key, True, t))
                        for k in reversed(pending[1:]):
                            self.typeahead.appendleft((k, True, None))
                        key = pending[0]
                try:
                    self.handle(key)
                except VimError as e:
                    self.typeahead.clear()
                    self.nkeys = []
                    if e.msg:
                        self.message(e.msg, "err")
                except Fail:
                    self.typeahead.clear()
                    self.nkeys = []
            self.map_wait = []
            # an incomplete command ends here (as if <Esc> were typed)
            if self.mode in ("i", "R"):
                self.leave_insert()
            elif self.mode == "c":
                self.cmdl = None
                self.mode = "n"
            if self.vis is not None:
                self.leave_visual()
            self.nkeys = []
        finally:
            self.typeahead = saved

    # ------------------------------------------------------------ prompts
    def prompt_key(self, key):
        p = self.pending_prompt
        if p["kind"] == "changed":
            if key in ("l", "L"):
                self.pending_prompt = None
                b = self.buf
                self.read_into(b)
                self.commit_undo()
                self.file_msg(b)
            elif key in ("o", "O", "<CR>", "<Esc>"):
                self.pending_prompt = None
                b = self.buf
                b.disk = self.io.read(b.path)
            return
        if p["kind"] == "spell":
            if key in ("<CR>", "<NL>"):
                self.pending_prompt = None
                typed = p["typed"]
                if typed.isdigit() and 1 <= int(typed) <= len(p["sugg"]):
                    from .spell import replace_word
                    l, s, e = p["span"]
                    replace_word(self, l, s, e, p["sugg"][int(typed) - 1])
                return
            if key in ("q", "<Esc>"):
                self.pending_prompt = None
                return
            if key.isdigit():
                p["typed"] += key
            elif key == "<BS>":
                p["typed"] = p["typed"][:-1]
            return

    def confirm_key(self, key):
        from .ex import confirm_key
        return confirm_key(self, key)

    def term_key(self, key):
        from .term import term_key
        return term_key(self, key)

    # ------------------------------------------------------------ special buffers take some keys themselves
    def normal_key(self, key):
        if not self.nkeys and self.vis is None:
            b = self.buf
            if b.kind == "netrw":
                from .netrw import netrw_key
                if netrw_key(self, key):
                    self.used["netrw:" + key] += 1
                    return
            elif b.kind == "quickfix":
                from .quickfix import qf_key
                if qf_key(self, key):
                    return
            elif b.kind == "terminal" and b.extra.get("running") and key in ("i", "a", "I", "A"):
                self.term_mode = True
                return
        r = super().normal_key(key)
        if getattr(self, "ins_paused", None) and self.mode == "n" and not self.nkeys and self.vis is None:
            self.resume_insert()
        return r

    # ------------------------------------------------------------ what the grader compares
    def state(self) -> dict:
        def reg(name):
            t = self.regs.get(name)
            return list(t) if t else None
        wins = []
        for t_i, t in enumerate(self.tabs):
            for w in self.windows_of(t):
                wins.append({"tab": t_i, "buf": w.buf.name, "kind": w.buf.kind, "row": w.row, "col": w.col,
                             "height": w.height, "width": w.width, "current": t_i == self.tabidx and w is t["win"]})
        layout = []
        for t in self.tabs:
            layout.append(_shape(t["layout"]))
        from .folds import all_folds, is_closed
        folds = [[f[0], f[1], bool(is_closed(self, self.win, f))] for f in all_folds(self, self.win)]
        return {
            "buffers": [{"num": b.num, "name": b.name, "path": b.path, "lines": list(b.lines), "modified": b.modified,
                         "listed": b.listed, "kind": b.kind} for b in self.buffers.values()],
            "buf": self.buf.name, "path": self.buf.path, "lines": list(self.buf.lines),
            "cursor": list(self.win.cursor), "mode": self.mode if self.vis is None else self.vis["mode"],
            "windows": wins, "layout": layout, "tab": self.tabidx, "tabs": len(self.tabs),
            "regs": {k: reg(k) for k in self.regs if not k.startswith("__")},
            "marks": {k: list(v) for k, v in self.buf.marks.items() if k.isalpha()},
            "global_marks": {k[-1]: [v[0], list(v[1])] for k, v in self.regs.items() if k.startswith("__mark_")},
            "folds": folds, "topline": self.win.topline, "quit": self.quit,
            "options": {k: self.opt(k) for k in ("number", "relativenumber", "ignorecase", "smartcase", "hlsearch",
                                                  "incsearch", "expandtab", "shiftwidth", "tabstop", "softtabstop",
                                                  "autoindent", "spell", "foldmethod", "undofile", "fileformat",
                                                  "clipboard", "scrolloff", "textwidth", "wrap", "list",
                                                  "splitright", "splitbelow", "cursorline", "colorcolumn",
                                                  "mouse", "smartindent", "laststatus")},
            "maps": {m: {K.to_notation(lhs): v["rhs"] for lhs, v in self.maps[m].items()} for m in self.maps},
            "leader": self.leader, "qf": len(self.qf), "errors": self.errors,
        }


def _shape(node):
    if node[0] == "leaf":
        return node[1].buf.name or ("[" + node[1].buf.kind + "]")
    return [node[0], [_shape(c) for c in node[1]]]


__all__ = ["Vim", "VimError", "MAXCOL"]

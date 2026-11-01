"""The Vim engine: modes, the normal-mode command grammar, operators, registers, undo, dot repeat, macros, maps.

It runs on lines in memory; files, shell commands and the clock come from an `io` object (the bash-lite machine in
the terminal, or a plain dict of files in tests). One key at a time goes in through feed(); state() is what the grader
compares, render() what the terminal shows. Insert/replace, visual, the command line and Ex commands, windows and the
screen live in the mixins next to this file.
"""

from __future__ import annotations

import re
from collections import Counter, deque

from . import keys as K
from .buffer import Buffer
from .motions import (Fail, bck_word, bckend_word, dec, end_word, find_char, find_char_repeat, fwd_word, inc,
                      keyword_at, match_pair, paragraph, tobj_bracket, tobj_paragraph, tobj_quote, tobj_sentence,
                      tobj_tag, tobj_word)
from .text import (brace_indent, col_at_vcol, first_nonblank, indent_of, is_kw, python_indent, set_indent, vcol,
                   yaml_indent)

MAXCOL = 1 << 30


class VimError(Exception):
    """An error message (E...) or just a beep (msg ""). Aborts a macro, mapping or dot repeat."""

    def __init__(self, msg: str = ""):
        super().__init__(msg)
        self.msg = msg


OPERATORS = {"d", "c", "y", "<", ">", "=", "!", "g~", "gu", "gU", "g?", "gq", "gw", "zf", "g@"}
LINE_OPS = {"<", ">", "=", "!", "gq", "gw", "zf"}         # operators that always work on whole lines
REG_NAMES = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\"-*+_.:/%#=")
FILETYPES = {"py": "python", "pyi": "python", "txt": "text", "md": "markdown", "markdown": "markdown", "yaml": "yaml",
             "yml": "yaml", "json": "json", "sh": "sh", "bash": "sh", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp",
             "hpp": "cpp", "js": "javascript", "ts": "typescript", "html": "html", "htm": "html", "xml": "xml",
             "csv": "csv", "toml": "toml", "vim": "vim", "cfg": "dosini", "ini": "dosini", "log": "", "rs": "rust",
             "java": "java", "go": "go", "sql": "sql", "css": "css", "tex": "tex", "r": "r", "lua": "lua"}
FT_NAMES = {"Makefile": "make", "makefile": "make", "Dockerfile": "dockerfile", ".vimrc": "vim", "vimrc": "vim",
            ".bashrc": "sh", ".profile": "sh", "requirements.txt": "requirements"}

# options: name -> (abbreviation, scope "g"lobal / "b"uffer / "w"indow, default)
OPTIONS = {
    "number": ("nu", "w", False), "relativenumber": ("rnu", "w", False), "wrap": ("", "w", True),
    "list": ("", "w", False), "cursorline": ("cul", "w", False), "colorcolumn": ("cc", "w", ""),
    "foldmethod": ("fdm", "w", "manual"), "foldlevel": ("fdl", "w", 0), "foldenable": ("fen", "w", True),
    "spell": ("", "w", False), "scrolloff": ("so", "g", 0), "tabstop": ("ts", "b", 8), "shiftwidth": ("sw", "b", 8),
    "softtabstop": ("sts", "b", 0), "expandtab": ("et", "b", False), "autoindent": ("ai", "b", False),
    "smartindent": ("si", "b", False), "cindent": ("cin", "b", False), "textwidth": ("tw", "b", 0),
    "fileformat": ("ff", "b", "unix"), "filetype": ("ft", "b", ""), "modifiable": ("ma", "b", True),
    "readonly": ("ro", "b", False), "undofile": ("udf", "b", False), "modified": ("mod", "b", False),
    "ignorecase": ("ic", "g", False), "smartcase": ("scs", "g", False), "hlsearch": ("hls", "g", False),
    "incsearch": ("is", "g", False), "wrapscan": ("ws", "g", True), "magic": ("", "g", True),
    "joinspaces": ("js", "g", True), "startofline": ("sol", "g", True), "splitright": ("spr", "g", False),
    "splitbelow": ("sb", "g", False), "hidden": ("hid", "g", False), "laststatus": ("ls", "g", 1),
    "ruler": ("ru", "g", False), "showcmd": ("sc", "g", False), "showmode": ("smd", "g", True),
    "timeoutlen": ("tm", "g", 1000), "history": ("hi", "g", 50), "backspace": ("bs", "g", ""),
    "nrformats": ("nf", "g", "bin,octal,hex"), "clipboard": ("cb", "g", ""), "mouse": ("", "g", ""),
    "whichwrap": ("ww", "g", "b,s"), "tildeop": ("top", "g", False), "report": ("", "g", 2),
    "undolevels": ("ul", "g", 1000), "compatible": ("cp", "g", False), "syntax": ("syn", "b", ""),
    "background": ("bg", "g", "light"), "paste": ("", "g", False), "wildmenu": ("wmnu", "g", False),
    "display": ("dy", "g", ""), "ttimeout": ("", "g", False), "spelllang": ("spl", "b", "en"),
    "formatoptions": ("fo", "b", "tcq"), "iskeyword": ("isk", "b", "@,48-57,_,192-255"), "gdefault": ("gd", "g", False),
    "termguicolors": ("tgc", "g", False), "encoding": ("enc", "g", "utf-8"), "fileencoding": ("fenc", "b", ""),
    "shell": ("sh", "g", "/bin/bash"), "equalprg": ("ep", "g", ""), "formatprg": ("fp", "b", ""),
    "virtualedit": ("ve", "g", ""), "cpoptions": ("cpo", "g", "aABceFs"), "showmatch": ("sm", "g", False),
    "smarttab": ("sta", "g", False), "shiftround": ("sr", "g", False), "selection": ("sel", "g", "inclusive"),
    "sidescroll": ("ss", "g", 0), "scroll": ("scr", "w", 11), "winheight": ("wh", "g", 1),
    "equalalways": ("ea", "g", True), "autoread": ("ar", "g", False), "backup": ("bk", "g", False),
    "swapfile": ("swf", "b", True), "writebackup": ("wb", "g", True), "updatetime": ("ut", "g", 4000),
    "belloff": ("bo", "g", ""), "visualbell": ("vb", "g", False), "errorbells": ("eb", "g", False),
    "fileformats": ("ffs", "g", "unix,dos"), "foldcolumn": ("fdc", "w", 0), "signcolumn": ("scl", "w", "auto"),
    "numberwidth": ("nuw", "w", 4), "conceallevel": ("cole", "w", 0), "cursorcolumn": ("cuc", "w", False),
    "spellfile": ("spf", "b", ""), "complete": ("cpt", "b", ".,w,b,u,t,i"), "completeopt": ("cot", "g", "menu,preview"),
    "lazyredraw": ("lz", "g", False), "title": ("", "g", False), "termwinsize": ("tws", "w", ""),
    "wildmode": ("wim", "g", "full"), "autowrite": ("aw", "g", False), "confirm": ("cf", "g", False),
}
ABBREV = {v[0]: k for k, v in OPTIONS.items() if v[0]}
# what defaults.vim sets (Ubuntu's vim without a ~/.vimrc)
DEFAULTS_VIM = {"scrolloff": 5, "incsearch": True, "nrformats": "bin,hex", "backspace": "indent,eol,start",
                "showcmd": True, "ruler": True, "history": 200, "wildmenu": True, "display": "truncate",
                "ttimeout": True, "syntax": "on"}
FT_OPTS = {"python": {"expandtab": True, "shiftwidth": 4, "softtabstop": 4, "tabstop": 4, "autoindent": True},
           "markdown": {"expandtab": True, "shiftwidth": 4, "softtabstop": 4, "tabstop": 4},
           "yaml": {"expandtab": True, "shiftwidth": 2, "softtabstop": 2}}


class Window:
    _ids = 0

    def __init__(self, buf: Buffer):
        Window._ids += 1
        self.id = Window._ids
        self.buf = buf
        self.cursor = [0, 0]
        self.curswant = 0
        self.topline = 0
        self.leftcol = 0
        self.height = 23
        self.width = 80
        self.row = 0
        self.col = 0
        self.jumps: list[tuple] = []     # (bufnum, line, col)
        self.jumpidx = 0
        self.alt = None                   # alternate buffer number
        self.opts: dict = {}
        self.pcmark = None                # where the last jump started ('' and ``)
        self.last_cursor: dict = {}       # bufnum -> cursor when the buffer was left
        self.req_height = None            # :resize
        self.req_width = None
        self.has_status = False
        self.keep_top = False
        self.fold_state: dict = {}


class Core:
    """The editor. io: the machine (files, shell, clock). files: the file names to open (vim's arguments)."""

    def __init__(self, io, files=(), cwd="/", home="/home/ada", width=80, height=24, now=0,
                 noquit=False, line=None, split=None, readonly=False, vimrc=None, target=None):
        self.io = io
        self.cwd = cwd
        self.home = home
        self.width, self.height = width, height
        self.now = now
        self.noquit = noquit
        self.target = target                         # (line, col) a drill asks to reach: drawn highlighted
        self.o: dict = {name: d for name, (_, _, d) in OPTIONS.items()}
        self.buffers: dict[int, Buffer] = {}
        self.next_buf = 1
        self.regs: dict[str, tuple] = {}
        self.mode = "n"
        self.nkeys: list[str] = []                   # a normal-mode command being typed
        self.cur_keys: list[str] = []                # every key of the current command (for dot repeat)
        self.op_pending = None                       # an operator waiting for a "/" or "?" motion
        self.vis = None                              # visual: {"anchor": (l, c), "mode": "v"/"V"/"\x16"}
        self.last_vis = None
        self.ins = None                              # insert/replace mode state
        self.cmdl = None                             # command-line state
        self.hist = {":": [], "/": []}
        self.last_pat = None
        self.last_dir = 1
        self.last_off = ""
        self.last_sub = None                         # the last replacement string (~)
        self.last_s = None                           # (pattern, replacement, flags) for & and :s without a pattern
        self.last_f = None                           # (kind, char) for ; and ,
        self.no_hl = False                           # :noh
        self.msg = ""
        self.msg_kind = ""                           # "" | "err" | "warn" | "mode"
        self.more: list[str] | None = None           # output waiting for "Press ENTER"
        self.more_kind = ""
        self.dot = None
        self.recording = None
        self.rec_keys: list[str] = []
        self.last_macro = None
        self.typeahead: deque = deque()              # (key, remap) waiting to be executed
        self.map_wait: list = []                     # keys that may start a mapping: [(key, time)]
        self.maps: dict[str, dict] = {m: {} for m in "nvoic"}
        self.leader = "\\"
        self.autocmds: list[tuple] = []              # (event, pattern, command)
        self.used: Counter = Counter()               # canonical commands run (what the grader checks)
        self.nkeys_total = 0
        self.quit = None                             # exit status once Vim quits
        self.qf: list[dict] = []                     # quickfix list
        self.qf_idx = 0
        self.undo_store: dict = {}                   # 'undofile': undo trees kept per path
        self.spell_good: set = set()
        self.changes = 0
        self.showcmd = ""
        self.confirm = None                          # :s///c in progress
        self.pending_prompt = None                   # W11/W12 "file changed" question
        self.term_mode = False                       # in a :terminal window, typing goes to the shell
        self.errors = 0
        self.bell = False
        self.screen_cache = None
        self.vimrc_loaded = False
        self.started = False
        # tabs: [{"layout": tree, "win": Window, "prev": Window}]
        self.tabs: list[dict] = []
        self.tabidx = 0
        # load the settings: ~/.vimrc, or defaults.vim when there is none
        rc = vimrc if vimrc is not None else self.io.read(home.rstrip("/") + "/.vimrc")
        if rc is None:
            self.o.update(DEFAULTS_VIM)
            self.filetype_on = True
        else:
            self.filetype_on = False
        names = list(files) or [None]
        bufs = []
        for name in names:
            bufs.append(self.load_buffer(name) if name else self.new_buffer())
        win = Window(bufs[0])
        self.tabs.append({"layout": ["leaf", win], "win": win, "prev": None})
        if rc is not None:
            self.source_text(rc, home.rstrip("/") + "/.vimrc")
            self.vimrc_loaded = True
        for b in bufs:
            self.apply_filetype(b)
        if readonly:
            for b in bufs:
                b.readonly = True
        if split and len(bufs) > 1:
            for b in bufs[1:]:
                self.split(vertical=split == "O", buf=b, at_end=True)
            self.goto_win(self.all_windows()[0])
        self.layout()
        if line is not None:
            n = len(self.buf.lines)
            ln = n - 1 if line == "$" else max(0, min(n - 1, int(line) - 1))
            self.win.cursor = [ln, first_nonblank(self.buf.lines[ln])]
        self.fire("BufEnter", self.buf)
        self.file_msg(self.buf, opened=True)
        self.started = True
        self.scroll_to_cursor()

    # ------------------------------------------------------------ basics

    @property
    def tab(self):
        return self.tabs[self.tabidx]

    @property
    def win(self) -> Window:
        return self.tab["win"]

    @property
    def buf(self) -> Buffer:
        return self.win.buf

    @property
    def lines(self) -> list[str]:
        return self.win.buf.lines

    def opt(self, name, buf=None, win=None):
        name = ABBREV.get(name, name)
        scope = OPTIONS.get(name, ("", "g", None))[1]
        if scope == "b":
            b = buf or self.buf
            if name in b.opts:
                return b.opts[name]
        elif scope == "w":
            w = win or self.win
            if name in w.opts:
                return w.opts[name]
        return self.o.get(name)

    def sw(self, buf=None):
        b = buf or self.buf
        v = self.opt("shiftwidth", b)
        return v or self.opt("tabstop", b)

    def ts(self, buf=None):
        return self.opt("tabstop", buf or self.buf) or 8

    def message(self, text, kind=""):
        self.msg = text
        self.msg_kind = kind

    def error(self, text):
        raise VimError(text)

    def beep(self):
        raise VimError("")

    # ------------------------------------------------------------ buffers

    def abspath(self, name: str) -> str:
        if name.startswith("~"):
            name = self.home + name[1:]
        if not name.startswith("/"):
            name = self.cwd.rstrip("/") + "/" + name
        parts = []
        for p in name.split("/"):
            if p in ("", "."):
                continue
            if p == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(p)
        return "/" + "/".join(parts)

    def relname(self, path: str) -> str:
        cwd = self.cwd.rstrip("/") + "/"
        if path.startswith(cwd):
            return path[len(cwd):]
        home = self.home.rstrip("/") + "/"
        if path.startswith(home):
            return "~/" + path[len(home):]
        return path

    def find_buffer(self, path):
        for b in self.buffers.values():
            if b.path == path:
                return b
        return None

    def new_buffer(self, name=None, kind="file", lines=None, listed=True) -> Buffer:
        b = Buffer(self.next_buf, name or "", None, list(lines or [""]), self.now)
        b.kind = kind
        b.listed = listed
        self.buffers[b.num] = b
        self.next_buf += 1
        return b

    def load_buffer(self, name: str) -> Buffer:
        path = self.abspath(name)
        b = self.find_buffer(path)
        if b is not None:
            if not b.loaded:
                self.read_into(b)
            return b
        b = Buffer(self.next_buf, self.relname(path), path, [""], self.now)
        self.buffers[b.num] = b
        self.next_buf += 1
        self.read_into(b)
        return b

    def read_into(self, b: Buffer, reload: bool = False):
        """(Re)read a buffer's file. 'undofile' history comes back when the buffer is loaded again, not on a
        reload of the buffer being edited (:e, :e!)."""
        if self.io.isdir(b.path):
            from .netrw import fill
            fill(self, b, b.path)
            return
        text = self.io.read(b.path)
        b.loaded = True
        if text is None:
            b.lines[:] = [""]
            b.new_file = True
            b.eol = True
        else:
            b.new_file = False
            if text.endswith("\n"):
                body, b.eol = text[:-1], True
            else:
                body, b.eol = text, text == ""
            lines = body.split("\n") if body or text else [""]
            if len(lines) > 0 and all(x.endswith("\r") for x in lines if x) and any(x for x in lines) and \
                    (text.endswith("\r\n") or not text.endswith("\n")):
                b.ff = "dos"
                lines = [x[:-1] if x.endswith("\r") else x for x in lines]
            else:
                b.ff = "unix"
            b.lines[:] = lines or [""]
            b.readonly = not self.io.writable(b.path)
        b.modified = False
        stored = self.undo_store.get(b.path)
        b.undo = type(b.undo)(b.lines, self.now)     # a new tree: the old one may be the one kept for 'undofile'
        b.saved_seq = 0
        if stored is not None and not reload and tuple(stored.cur.lines) == tuple(b.lines):
            b.undo = stored
            b.saved_seq = stored.cur.seq
        b.opts["fileformat"] = b.ff
        self.apply_filetype(b)

    def apply_filetype(self, b: Buffer):
        if b.kind != "file" or not b.path:
            return
        base = b.path.rsplit("/", 1)[-1]
        ft = FT_NAMES.get(base)
        if ft is None:
            ext = base.rsplit(".", 1)[-1] if "." in base else ""
            ft = FILETYPES.get(ext, "")
        if b.opts.get("filetype") == ft and b.extra.get("ft_done"):
            return
        b.opts["filetype"] = ft
        b.extra["ft_done"] = True
        if self.filetype_on:
            for k, v in FT_OPTS.get(ft, {}).items():
                b.opts.setdefault(k, v)
        self.fire("FileType", b, ft)

    def file_msg(self, b: Buffer, opened=False):
        if b.kind != "file":
            return
        name = b.name or "[No Name]"
        if b.new_file:
            self.message(f'"{name}" [New]')
            return
        n = len(b.lines)
        size = len(b.text().encode())
        flags = (" [readonly]" if b.readonly else "") + (" [dos]" if b.ff == "dos" else "") + \
            ("" if b.eol else " [noeol]")
        self.message(f'"{name}"{flags} {n}L, {size}B')

    # ------------------------------------------------------------ cursor

    def clamp(self, win=None, insert=False):
        w = win or self.win
        lines = w.buf.lines
        l = max(0, min(len(lines) - 1, w.cursor[0]))
        line = lines[l]
        maxc = len(line) if (insert or self.mode in ("i", "R") or (self.vis and self.opt("selection") == "exclusive")) \
            else max(0, len(line) - 1)
        if self.vis is not None and w is self.win and self.vis["mode"] != "V":
            maxc = max(0, len(line) - 1) if line else 0
        w.cursor = [l, max(0, min(maxc, w.cursor[1]))]

    def set_cursor(self, l, c=None, want=True, win=None):
        w = win or self.win
        lines = w.buf.lines
        l = max(0, min(len(lines) - 1, l))
        if c is None:
            c = first_nonblank(lines[l])
        w.cursor = [l, c]
        self.clamp(w)
        if want:
            w.curswant = vcol(lines[w.cursor[0]], w.cursor[1], self.ts(w.buf))

    def go_line(self, l, sol=None, win=None):
        """Move to a line: the first non-blank ('startofline'), else the remembered column."""
        w = win or self.win
        lines = w.buf.lines
        l = max(0, min(len(lines) - 1, l))
        if sol is None:
            sol = self.opt("startofline")
        if sol:
            self.set_cursor(l, first_nonblank(lines[l]), win=w)
        else:
            w.cursor = [l, self.col_for_want(lines[l], w.curswant, w.buf)]
            self.clamp(w)

    def col_for_want(self, line, want, buf=None):
        if want >= MAXCOL:
            return max(0, len(line) - (0 if self.mode in ("i", "R") else 1))
        return col_at_vcol(line, want, self.ts(buf), past_end=self.mode in ("i", "R"))

    def setpcmark(self):
        """Remember where a jump started ('' and the jump list)."""
        w = self.win
        pos = (self.buf.num, w.cursor[0], w.cursor[1])
        w.pcmark = (w.cursor[0], w.cursor[1])
        self.buf.marks["'"] = (w.cursor[0], w.cursor[1])
        # drop this line from the list, then append
        w.jumps = w.jumps[:w.jumpidx] if w.jumpidx < len(w.jumps) else w.jumps
        w.jumps = [j for j in w.jumps if not (j[0] == pos[0] and j[1] == pos[1])]
        w.jumps.append(pos)
        if len(w.jumps) > 100:
            w.jumps = w.jumps[-100:]
        w.jumpidx = len(w.jumps)

    # ------------------------------------------------------------ editing primitives

    def check_modifiable(self, buf=None):
        b = buf or self.buf
        if b.opts.get("modifiable", True) is False or b.kind in ("help", "netrw", "quickfix"):
            self.error("E21: Cannot make changes, 'modifiable' is off")

    def changed(self, buf, first, last):
        buf.modified = True
        buf.marks["."] = (first, self.win.cursor[1] if self.win.buf is buf else 0)
        buf.marks["["] = (first, 0)
        buf.marks["]"] = (last, 0)
        self.changes += 1
        if buf.uline is None or buf.uline[0] != first or first != last:
            if first == last and first < len(buf.lines):
                old = buf.undo.cur.lines
                buf.uline = (first, old[first] if first < len(old) else "")
            else:
                buf.uline = None

    def replace_lines(self, buf, start, end, new):
        """lines[start:end] = new, keeping marks and folds in place."""
        self.check_modifiable(buf)
        old_n, new_n = end - start, len(new)
        buf.lines[start:end] = list(new)
        if not buf.lines:
            buf.lines.append("")
        if new_n < old_n:
            buf.adjust(start + new_n, end - 1, None, new_n - old_n)
        elif new_n > old_n:
            buf.adjust(end, 1 << 30, new_n - old_n, new_n - old_n)
        self.changed(buf, start, max(start, start + new_n - 1))
        gl = getattr(self, "g_lines", None)
        if gl is not None and buf is self.buf and new_n != old_n:
            keep = min(old_n, new_n)
            for i, x in enumerate(gl):
                if x is None or x < start + keep:
                    continue
                if x < end:
                    gl[i] = None
                else:
                    gl[i] = x + new_n - old_n
        for w in self.all_windows(all_tabs=True):
            if w.buf is buf:
                if w is not self.win and w.cursor[0] >= end:
                    w.cursor[0] += new_n - old_n
                w.cursor[0] = max(0, min(len(buf.lines) - 1, w.cursor[0]))

    def set_line(self, buf, l, text):
        self.check_modifiable(buf)
        if buf.lines[l] != text:
            buf.lines[l] = text
            self.changed(buf, l, l)

    def get_text(self, buf, s, e) -> str:
        """Characters from s to e inclusive (positions may be at a line's end: the line break)."""
        (l1, c1), (l2, c2) = s, e
        if l1 == l2:
            line = buf.lines[l1]
            t = line[c1:c2 + 1]
            if c2 >= len(line):
                t += "\n"
            return t
        parts = [buf.lines[l1][c1:]]
        for l in range(l1 + 1, l2):
            parts.append(buf.lines[l])
        last = buf.lines[l2]
        parts.append(last[:c2 + 1] + ("\n" if c2 >= len(last) else ""))
        return "\n".join(parts)

    def delete_range(self, buf, s, e):
        """Delete characters s..e inclusive (e may sit on a line end: the line break goes too)."""
        (l1, c1), (l2, c2) = s, e
        last = buf.lines[l2]
        if c2 >= len(last) and l2 + 1 < len(buf.lines):
            tail = buf.lines[l2 + 1]
            l2 += 1
            c2 = -1
            last = tail
        new = buf.lines[l1][:c1] + last[c2 + 1:]
        self.replace_lines(buf, l1, l2 + 1, [new])

    def insert_text(self, buf, pos, text) -> tuple[int, int]:
        """Insert text (with line breaks) at pos. -> the position just after it."""
        l, c = pos
        line = buf.lines[l]
        parts = text.split("\n")
        if len(parts) == 1:
            self.set_line(buf, l, line[:c] + text + line[c:])
            return l, c + len(text)
        new = [line[:c] + parts[0]] + parts[1:-1] + [parts[-1] + line[c:]]
        self.replace_lines(buf, l, l + 1, new)
        return l + len(parts) - 1, len(parts[-1])

    # ------------------------------------------------------------ registers

    def reg_get(self, name):
        if name is None or name == '"':
            name = self.regs.get("__unnamed", "0")
            return self.regs.get(name, self.regs.get('"', ("", "v")))
        name = name.lower() if name.isupper() else name
        if name == "%":
            return (self.buf.name, "v")
        if name == "#":
            alt = self.buffers.get(self.win.alt)
            return (alt.name if alt else "", "v")
        if name == "+" or name == "*":
            return self.regs.get("+", ("", "v"))
        return self.regs.get(name, ("", "v"))

    def reg_set(self, name, text, kind, op="y"):
        """Put text in a register the way an operator does: yank -> "0, delete -> "1 / "-, plus the unnamed."""
        if name == "_":
            return
        if name in ("*",):
            name = "+"
        clip = "unnamed" in str(self.opt("clipboard"))
        if name and name.isupper():
            low = name.lower()
            old, okind = self.regs.get(low, ("", kind))
            if okind == "V" and kind != "V":
                text = old + text + "\n"
                kind = "V"
            elif kind == "V" and okind != "V":
                text = old + "\n" + text
            else:
                text = old + text
            self.regs[low] = (text, kind)
            self.regs["__unnamed"] = low
            return
        if name:
            self.regs[name] = (text, kind)
            self.regs["__unnamed"] = name
            if op in ("d", "dbig"):
                for i in range(9, 1, -1):
                    if str(i - 1) in self.regs:
                        self.regs[str(i)] = self.regs[str(i - 1)]
                self.regs["1"] = (text, kind)
            return
        if op == "y":
            self.regs["0"] = (text, kind)
            self.regs["__unnamed"] = "0"
        else:
            if kind == "V" or "\n" in text or op == "dbig":
                for i in range(9, 1, -1):
                    if str(i - 1) in self.regs:
                        self.regs[str(i)] = self.regs[str(i - 1)]
                self.regs["1"] = (text, kind)
                self.regs["__unnamed"] = "1"
                if op == "dbig" and "\n" not in text and kind != "V":
                    self.regs["-"] = (text, kind)
            else:
                self.regs["-"] = (text, kind)
                self.regs["__unnamed"] = "-"
        if clip:
            self.regs["+"] = (text, kind)

    # ------------------------------------------------------------ input: maps, macros, typeahead

    def feed(self, key: str, t: int | None = None):
        """A key the person pressed."""
        if t is not None:
            self.resolve_wait(t)
            self.now = max(self.now, t)
        if self.quit is not None:
            return
        self.nkeys_total += 1
        self.used["key:" + key] += 1                 # the key itself (before mappings)
        if self.recording is not None:
            self.rec_keys.append(key)
        self.typeahead.append((key, True, t))
        self.run()

    def tick(self, t: int):
        """Time passes: a half-typed mapping (jk ...) gives up after 'timeoutlen'."""
        self.resolve_wait(t)

    def resolve_wait(self, t):
        if self.map_wait and t - self.map_wait[-1][1] > self.opt("timeoutlen"):
            pending = self.map_wait
            self.map_wait = []
            m = self.map_mode()
            exact = self.maps[m].get(tuple(k for k, _ in pending)) if m else None
            if exact is not None:
                self.expand_map(exact)
            else:
                for k, _ in reversed(pending):
                    self.typeahead.appendleft((k, False, None))
            self.run()

    def map_mode(self):
        if self.mode in ("i", "R"):
            return "i"
        if self.mode == "c":
            return "c"
        if self.vis is not None:
            return "v"
        if self.nkeys and self.op_started():
            return "o"
        if self.mode == "n":
            return "n"
        return None

    def op_started(self):
        ks = self.nkeys
        i = 0
        if ks[:1] == ['"']:
            i = 2
        while i < len(ks) and ks[i].isdigit() and not (ks[i] == "0" and (i == 0 or not ks[i - 1].isdigit())):
            i += 1
        rest = "".join(ks[i:])
        return any(rest.startswith(op) for op in OPERATORS) and rest not in ("g", "z")

    def expand_map(self, m):
        rhs = K.parse(m["rhs"], self.leader)
        for k in reversed(rhs):
            self.typeahead.appendleft((k, not m["noremap"], None))

    def run(self):
        guard = 0
        while self.typeahead and self.quit is None:
            guard += 1
            if guard > 500_000:
                self.typeahead.clear()
                self.message("E223: recursive mapping", "err")
                break
            key, remap, t = self.typeahead.popleft()
            # mappings
            mm = self.map_mode()
            if remap and mm and self.maps[mm] and not self.term_mode and self.more is None:
                seq = tuple(k for k, _ in self.map_wait) + (key,)
                longer = [lhs for lhs in self.maps[mm] if len(lhs) > len(seq) and lhs[:len(seq)] == seq]
                exact = self.maps[mm].get(seq)
                if longer:
                    self.map_wait.append((key, t if t is not None else self.now))
                    continue
                if exact is not None:
                    self.map_wait = []
                    self.expand_map(exact)
                    continue
                if self.map_wait:
                    # the waiting keys weren't a mapping after all
                    pending = [k for k, _ in self.map_wait]
                    self.map_wait = []
                    self.typeahead.appendleft((key, True, t))
                    for k in reversed(pending[1:]):
                        self.typeahead.appendleft((k, True, None))
                    key = pending[0]
                    remap = False
            try:
                self.handle(key)
            except VimError as e:
                self.fail(e.msg)
            except Fail:
                self.fail("")
        self.after_command()

    def fail(self, msg):
        """An error: show it, drop the rest of the macro / mapping / dot repeat, reset the command."""
        self.typeahead.clear()
        self.map_wait = []
        self.nkeys = []
        self.cur_keys = []
        self.op_pending = None
        self.showcmd = ""
        if self.mode == "c" and self.cmdl and self.cmdl.get("from_op"):
            self.mode = "n"
        if msg:
            self.message(msg, "err")
            self.errors += 1
        else:
            self.bell = True

    def after_command(self):
        """After the keys: the undo step closes (in normal mode), the view follows the cursor."""
        if self.mode == "n" and not self.nkeys and self.vis is None:
            self.commit_undo()
        for w in self.all_windows():
            if w.buf.lines and (w is self.win):
                self.clamp(w)
        self.scroll_to_cursor()

    # ------------------------------------------------------------ the key dispatcher

    def handle(self, key):
        if self.pending_prompt is not None:
            return self.prompt_key(key)
        if self.more is not None:
            return self.more_key(key)
        if self.confirm is not None:
            return self.confirm_key(key)
        if self.term_mode:
            return self.term_key(key)
        if self.mode in ("i", "R"):
            return self.insert_key(key)
        if self.mode == "c":
            return self.cmdline_key(key)
        if self.vis is not None:
            return self.visual_key(key)
        return self.normal_key(key)

    def more_key(self, key):
        """The "Press ENTER or type command to continue" prompt."""
        self.more = None
        self.more_kind = ""
        if key in ("<CR>", " ", "<NL>", "<Esc>"):
            return
        if key == ":":
            return self.normal_key(":")
        self.typeahead.appendleft((key, True, None))

    # ------------------------------------------------------------ normal mode

    def normal_key(self, key):
        if not self.nkeys:
            self.cur_keys = []
            self.msg_on_key()
        self.nkeys.append(key)
        self.cur_keys.append(key)
        st = self.parse_normal(self.nkeys)
        if st[0] == "wait":
            self.showcmd = "".join(K.shown(K.to_text([k])) for k in self.nkeys)[-10:]
            return
        keys = self.nkeys
        self.nkeys = []
        self.showcmd = ""
        if st[0] == "bad":
            self.cur_keys = []
            raise VimError("")
        cmd = st[1]
        cmd["keys"] = keys
        self.exec_normal(cmd)

    def msg_on_key(self):
        if self.msg_kind != "keep":
            self.msg = ""
            self.msg_kind = ""

    def parse_normal(self, ks):
        """-> ("wait",) | ("bad",) | ("ok", cmd). cmd: reg, count, op, motion, arg, ..."""
        i, n = 0, len(ks)
        cmd = {"reg": None, "count": None, "op": None, "key": None, "arg": None, "count2": None}
        if i < n and ks[i] == '"':
            if i + 1 >= n:
                return ("wait",)
            if ks[i + 1] not in REG_NAMES:
                return ("bad",)
            cmd["reg"] = ks[i + 1]
            i += 2
        digits = ""
        while i < n and ks[i].isdigit() and not (ks[i] == "0" and not digits):
            digits += ks[i]
            i += 1
        if digits:
            cmd["count"] = int(digits)
        if i < n and ks[i] == '"' and cmd["reg"] is None:
            # a register after the count ("2"ayy" is also fine)
            if i + 1 >= n:
                return ("wait",)
            cmd["reg"] = ks[i + 1]
            i += 2
            while i < n and ks[i].isdigit() and not (ks[i] == "0" and not digits):
                digits += ks[i]
                i += 1
            cmd["count"] = int(digits) if digits else None
        if i >= n:
            return ("wait",)
        # an operator?
        op, j = self._op_at(ks, i)
        if op == "wait":
            return ("wait",)
        if op:
            cmd["op"] = op
            i = j
            d2 = ""
            while i < n and ks[i].isdigit() and not (ks[i] == "0" and not d2):
                d2 += ks[i]
                i += 1
            if d2:
                cmd["count2"] = int(d2)
            if i >= n:
                return ("wait",)
            # doubled operator: dd cc yy >> << == g~~ guu gUU gqq ...
            rest = ks[i:]
            dbl = {"d": ["d"], "c": ["c"], "y": ["y"], "<": ["<"], ">": [">"], "=": ["="], "!": ["!"],
                   "g~": [["~"], ["g", "~"]], "gu": [["u"], ["g", "u"]], "gU": [["U"], ["g", "U"]],
                   "g?": [["?"], ["g", "?"]], "gq": [["q"], ["g", "q"]], "gw": [["w"], ["g", "w"]],
                   "zf": [["F"]], "g@": [["@"]]}
            opts_ = dbl.get(op, [])
            if op in ("d", "c", "y", "<", ">", "=", "!"):
                opts_ = [[op]]
            for d in opts_:
                if rest[:len(d)] == d:
                    if len(rest) > len(d):
                        return ("bad",)
                    cmd["key"] = "line"
                    return ("ok", cmd)
                if len(rest) < len(d) and d[:len(rest)] == rest:
                    return ("wait",)
            # a text object?
            if rest[0] in ("i", "a"):
                if len(rest) < 2:
                    return ("wait",)
                if rest[1] not in "wWsp()b[]{}B<>\"'`t":
                    return ("bad",)
                if len(rest) > 2:
                    return ("bad",)
                cmd["key"] = rest[0] + rest[1]
                cmd["tobj"] = True
                return ("ok", cmd)
            m = self._motion_at(ks, i)
            if m[0] == "wait":
                return ("wait",)
            if m[0] == "bad":
                return ("bad",)
            cmd["key"], cmd["arg"] = m[1], m[2]
            if m[3] != n:
                return ("bad",)
            return ("ok", cmd)
        # a motion?
        m = self._motion_at(ks, i)
        if m[0] == "ok":
            if m[3] != n:
                return ("bad",)
            cmd["key"], cmd["arg"] = m[1], m[2]
            cmd["motion"] = True
            return ("ok", cmd)
        if m[0] == "wait":
            return ("wait",)
        # a command
        c = self._command_at(ks, i)
        if c[0] != "ok":
            return (c[0],)
        cmd["key"], cmd["arg"] = c[1], c[2]
        if c[3] != n:
            return ("bad",)
        return ("ok", cmd)

    def _op_at(self, ks, i):
        k = ks[i]
        if k in ("d", "c", "y", "<", ">", "=", "!"):
            return k, i + 1
        if k in ("g", "z"):
            if i + 1 >= len(ks):
                return "wait", i
            two = k + ks[i + 1]
            if two in OPERATORS:
                return two, i + 2
        return None, i

    MOTIONS1 = {"h", "j", "k", "l", " ", "<BS>", "<Left>", "<Right>", "<Up>", "<Down>", "<C-p>", "<C-n>", "<NL>",
                "<CR>", "+", "-", "_", "w", "W", "b", "B", "e", "E", "0", "^", "$", "|", "G", "H", "M", "L", "%",
                "{", "}", "(", ")", ";", ",", "n", "N", "*", "#", "<Home>", "<End>", "/", "?"}
    MOTIONS_G = {"g", "e", "E", "_", "0", "^", "$", "j", "k", "m", "M", "*", "#", "d", "D", "n", "N", "o", "'", "`",
                 "<Down>", "<Up>"}

    def _motion_at(self, ks, i):
        """-> (state, name, arg, next index)."""
        n = len(ks)
        k = ks[i]
        if k in ("f", "F", "t", "T", "'", "`"):
            if i + 1 >= n:
                return ("wait", None, None, i)
            return ("ok", k, ks[i + 1], i + 2)
        if k in self.MOTIONS1:
            return ("ok", k, None, i + 1)
        if k == "g":
            if i + 1 >= n:
                return ("wait", None, None, i)
            k2 = ks[i + 1]
            if k2 in ("'", "`"):
                if i + 2 >= n:
                    return ("wait", None, None, i)
                return ("ok", "g" + k2, ks[i + 2], i + 3)
            if k2 in self.MOTIONS_G:
                return ("ok", "g" + k2, None, i + 2)
            return ("no", None, None, i)
        if k in ("[", "]"):
            if i + 1 >= n:
                return ("wait", None, None, i)
            k2 = ks[i + 1]
            if k2 in ("[", "]", "(", ")", "{", "}", "s", "'", "`", "z"):
                return ("ok", k + k2, None, i + 2)
            return ("bad", None, None, i)
        return ("no", None, None, i)

    COMMANDS1 = {"x", "X", "D", "C", "s", "S", "Y", "p", "P", "J", "~", "u", "<C-r>", "U", ".", "i", "a", "I", "A",
                 "o", "O", "R", "v", "V", "<C-v>", ":", "<C-a>", "<C-x>", "<C-o>", "<Tab>", "<C-^>", "<C-d>", "<C-u>",
                 "<C-f>", "<C-b>", "<C-e>", "<C-y>", "<C-g>", "<C-l>", "&", "<Esc>", "<C-c>", "<Del>", "K", "<C-]>",
                 "<C-z>", "<PageUp>", "<PageDown>", "<Insert>", "Q"}

    def _command_at(self, ks, i):
        n = len(ks)
        k = ks[i]
        if k in self.COMMANDS1:
            return ("ok", k, None, i + 1)
        if k in ("r", "m", "q", "@"):
            if k == "q" and self.recording is not None:
                return ("ok", "q", None, i + 1)
            if i + 1 >= n:
                return ("wait", None, None, i)
            if k == "r" and ks[i + 1] in ("<Esc>", "<C-c>"):
                return ("ok", "nop", None, i + 2)
            return ("ok", k, ks[i + 1], i + 2)
        if k == "g":
            if i + 1 >= n:
                return ("wait", None, None, i)
            k2 = ks[i + 1]
            if k2 in ("v", "i", "I", "J", "p", "P", "f", "F", "t", "T", "-", "+", "a", ";", ",", "&", "x", "s",
                      "<C-g>", "<C-a>", "<C-x>", "8", "<", "h", "H", "r", "R", "Q"):
                if k2 in ("r",):
                    if i + 2 >= n:
                        return ("wait", None, None, i)
                    return ("ok", "gr", ks[i + 2], i + 3)
                return ("ok", "g" + k2, None, i + 2)
            return ("bad", None, None, i)
        if k == "z":
            if i + 1 >= n:
                return ("wait", None, None, i)
            k2 = ks[i + 1]
            if k2 in ("z", "t", "b", ".", "-", "<CR>", "o", "O", "c", "C", "a", "A", "v", "x", "X", "m", "M", "r",
                      "R", "n", "N", "i", "E", "d", "D", "F", "j", "k", "=", "g", "w", "G", "W", "h", "l", "s", "e",
                      "H", "L"):
                return ("ok", "z" + k2, None, i + 2)
            if k2 == "u":
                if i + 2 >= n:
                    return ("wait", None, None, i)
                return ("ok", "zu" + ks[i + 2], None, i + 3)
            return ("bad", None, None, i)
        if k == "Z":
            if i + 1 >= n:
                return ("wait", None, None, i)
            if ks[i + 1] in ("Z", "Q"):
                return ("ok", "Z" + ks[i + 1], None, i + 2)
            return ("bad", None, None, i)
        if k == "<C-w>":
            j = i + 1
            d = ""
            while j < n and ks[j].isdigit():
                d += ks[j]
                j += 1
            if j >= n:
                return ("wait", None, None, i)
            k2 = ks[j]
            if k2 in ("g",) and j + 1 >= n:
                return ("wait", None, None, i)
            return ("ok", "<C-w>", (k2, int(d) if d else None), j + 1)
        return ("bad", None, None, i)

    # ------------------------------------------------------------ executing normal commands

    def exec_normal(self, cmd):
        count = cmd["count"]
        if cmd.get("count2") is not None:
            count = (count or 1) * cmd["count2"]
        cmd["n"] = count
        key = cmd["key"]
        if cmd["op"]:
            self.used[cmd["op"] + (cmd["op"][-1] if key == "line" else key)] += 1   # dd yy >> gUU ...
            self.used[cmd["op"]] += 1
            if key != "line":
                self.used[key] += 1
            if count:
                self.used["count"] += 1
            if cmd["reg"]:
                self.used['"' + cmd["reg"]] += 1
                self.used['"'] += 1
            if cmd["op"] != "y":
                self.mark_change_start()
            self.do_operator(cmd)
            if cmd["op"] not in ("y", "!", "zf") and self.mode == "n" and self.vis is None and \
                    cmd["key"] not in ("/", "?"):
                self.set_dot(cmd["keys"], count, cmd["reg"])
            return
        name = key if not cmd["arg"] or key in ("<C-w>",) else key
        self.used[name] += 1
        if key in ("'", "`") and cmd["arg"] in ("'", "`"):
            self.used["''"] += 1                     # back to where you were before the last jump
        elif key == "m" and cmd["arg"] and cmd["arg"].isupper():
            self.used["mA"] += 1                     # a global mark
        if count:
            self.used["count"] += 1
            self.used[f"{count}{name}"] += 1
        if cmd["reg"]:
            self.used['"' + cmd["reg"]] += 1
            self.used['"'] += 1
        if cmd.get("motion"):
            return self.do_motion_cmd(cmd)
        return self.do_command(cmd)

    def do_motion_cmd(self, cmd):
        w = self.win
        if cmd["key"] in ("/", "?"):
            self.start_cmdline(cmd["key"], from_op=None, count=cmd["n"])
            return
        key = cmd["key"]
        try:
            target = self.motion(cmd, None)
        except Fail:
            if key in ("w", "W", "b", "B", "e", "E", "ge", "gE"):
                self.win.curswant = vcol(self.lines[self.win.cursor[0]], self.win.cursor[1], self.ts())
            raise
        pos, kind = target[0], target[1]
        if key in self.JUMPS and not (key in ("'", "`") and cmd["arg"] in "[]<>.^\""):
            self.setpcmark()
        self.move_to(pos, cmd, kind)

    def move_to(self, pos, cmd, kind):
        w = self.win
        l, c = pos
        key = cmd["key"]
        keep_want = key in ("j", "k", "<Up>", "<Down>", "<C-p>", "<C-n>", "gj", "gk", "<NL>")
        w.cursor = [l, c]
        self.clamp(w)
        if key in ("$", "<End>", "g$") or (keep_want and w.curswant >= MAXCOL):
            w.curswant = MAXCOL
        elif not keep_want:
            w.curswant = vcol(self.lines[w.cursor[0]], w.cursor[1], self.ts())

    # ------------------------------------------------------------ motions

    JUMPS = {"G", "gg", "H", "M", "L", "%", "{", "}", "(", ")", "n", "N", "*", "#", "g*", "g#", "/", "?", "'", "`",
             "gd", "gD", "[[", "]]", "[]", "]["}

    def motion(self, cmd, op):
        """-> (pos, kind[, start]) for cmd's motion; kind: excl / incl / line."""
        w, lines = self.win, self.lines
        key, arg = cmd["key"], cmd["arg"]
        count = cmd["n"]
        n1 = count or 1
        l, c = w.cursor
        ts = self.ts()
        if key in self.JUMPS and not (key in ("'", "`") and arg in "[]<>"):
            if not (key in ("n", "N", "*", "#") and op is None and False):
                self.setpcmark_pending = True
        if key in ("h", "<Left>", "<BS>") and key != "<BS>" or (key == "<BS>" and False):
            if c == 0:
                if op is None:
                    raise Fail()
                return (l, 0), "excl"
            return (l, max(0, c - n1)), "excl"
        if key == "<BS>":
            pos = (l, c)
            for _ in range(n1):
                if pos[1] > 0:
                    pos = (pos[0], pos[1] - 1)
                elif pos[0] > 0:
                    pos = (pos[0] - 1, max(0, len(lines[pos[0] - 1]) - (0 if op else 1)))
                    if op is None:
                        pos = (pos[0], max(0, len(lines[pos[0]]) - 1))
                else:
                    if pos == (l, c):
                        raise Fail()
                    break
            return pos, "excl"
        if key in ("l", "<Right>", " "):
            line = lines[l]
            if key == " ":
                pos = (l, c)
                for _ in range(n1):
                    if pos[1] + 1 < len(lines[pos[0]]) or (op and pos[1] < len(lines[pos[0]])):
                        pos = (pos[0], pos[1] + 1)
                    elif pos[0] + 1 < len(lines):
                        pos = (pos[0] + 1, 0)
                    else:
                        if pos == (l, c):
                            raise Fail()
                        break
                return pos, "excl"
            if op is None:
                if c >= len(line) - 1:
                    raise Fail()
                return (l, min(len(line) - 1, c + n1)), "excl"
            if not line:
                raise Fail()
            return (l, min(len(line), c + n1)), "excl"
        if key in ("j", "<Down>", "<C-n>", "<NL>", "gj", "g<Down>", "k", "<Up>", "<C-p>", "gk", "g<Up>"):
            down = key in ("j", "<Down>", "<C-n>", "<NL>", "gj", "g<Down>")
            nl = self.fold_move(l, n1, down)
            if nl == l or (down and l + 0 == len(lines) - 1 and nl == l):
                raise Fail()
            return (nl, self.col_for_want(lines[nl], w.curswant)), "line"
        if key in ("+", "<CR>", "-", "_"):
            if key == "_":
                nl = l + n1 - 1
                if nl >= len(lines):
                    raise Fail()
            elif key == "-":
                nl = l - n1
                if nl < 0:
                    raise Fail()
            else:
                nl = l + n1
                if nl >= len(lines):
                    raise Fail()
            return (nl, first_nonblank(lines[nl])), "line"
        if key in ("w", "W"):
            big = key == "W"
            pos = (l, c)
            eol = op is not None
            # "cw" on a non-blank is "ce"
            if op == "c" and lines[l][c:c + 1] not in ("", " ", "\t"):
                p, ok = end_word(lines, pos, n1, big, stop=True)
                if not ok:
                    p = (len(lines) - 1, max(0, len(lines[-1]) - 1))
                return p, "incl"
            p, ok = fwd_word(lines, pos, n1, big, eol=eol)
            if not ok and op is None:
                raise Fail()
            if op is None and p[1] >= len(lines[p[0]]) and p[1] > 0:
                p = (p[0], p[1] - 1)
            if op is not None and p[1] >= len(lines[p[0]]) and p[1] > 0:
                # don't leave the end on the line break: make it inclusive on the last character
                return (p[0], p[1] - 1), "incl"
            return p, "excl"
        if key in ("b", "B"):
            p, ok = bck_word(lines, (l, c), n1, key == "B")
            if not ok:
                raise Fail()
            return p, "excl"
        if key in ("e", "E"):
            p, ok = end_word(lines, (l, c), n1, key == "E")
            if not ok:
                raise Fail()
            if p[1] >= len(lines[p[0]]) and p[1] > 0:
                p = (p[0], p[1] - 1)
            return p, "incl"
        if key in ("ge", "gE"):
            p, ok = bckend_word(lines, (l, c), n1, key == "gE")
            if not ok:
                raise Fail()
            if p[1] >= len(lines[p[0]]) and p[1] > 0:
                p = (p[0], p[1] - 1)
            return p, "incl"
        if key in ("0", "<Home>", "g0"):
            return (l, 0), "excl"
        if key in ("^", "g^"):
            return (l, first_nonblank(lines[l])), "excl"
        if key in ("$", "<End>", "g$"):
            nl = l + n1 - 1
            if nl >= len(lines):
                raise Fail()
            line = lines[nl]
            return (nl, max(0, len(line) - (0 if op else 1)) if op is None else max(0, len(line) - 1)), \
                ("incl" if line else "excl") if op else "incl"
        if key == "g_":
            nl = l + n1 - 1
            if nl >= len(lines):
                raise Fail()
            line = lines[nl]
            e = len(line.rstrip(" \t")) - 1
            return (nl, max(0, e)), "incl"
        if key == "gm":
            return (l, col_at_vcol(lines[l], self.win.width // 2, ts)), "excl"
        if key == "|":
            return (l, col_at_vcol(lines[l], n1 - 1, ts)), "excl"
        if key in ("G", "gg"):
            if count:
                nl = min(len(lines), count) - 1
            else:
                nl = len(lines) - 1 if key == "G" else 0
            col = first_nonblank(lines[nl]) if self.opt("startofline") else self.col_for_want(lines[nl], w.curswant)
            return (nl, col), "line"
        if key in ("H", "L", "M"):
            top, bot = self.visible_range(w)
            so = min(self.opt("scrolloff"), (w.height - 1) // 2)
            if key == "H":
                nl = top + (n1 - 1)
                if top > 0:
                    nl = max(nl, top + so)
                nl = min(nl, bot)
            elif key == "L":
                nl = bot - (n1 - 1)
                if bot < len(lines) - 1:
                    nl = min(nl, bot - so)
                nl = max(nl, top)
            else:
                last = min(bot, len(lines) - 1)
                nl = top + (last - top) // 2
            return (nl, first_nonblank(lines[nl]) if self.opt("startofline") else self.col_for_want(lines[nl], w.curswant)), "line"
        if key == "%":
            if count:
                if count > 100:
                    raise Fail()
                nl = (count * len(lines) + 99) // 100 - 1
                return (max(0, nl), first_nonblank(lines[max(0, nl)])), "line"
            return match_pair(lines, l, c), "incl"
        if key in ("}", "{"):
            nl, nc, incl = paragraph(lines, l, n1, key == "}")
            return (nl, nc), "incl" if incl else "excl"
        if key in ("(", ")"):
            return self.sentence_motion(l, c, n1, key == ")"), "excl"
        if key in ("f", "F", "t", "T"):
            if arg in ("<Esc>", "<C-c>"):
                raise Fail()
            ch = arg if len(arg) == 1 else K.CTRL_CHAR.get(arg, "\x00")
            self.last_f = (key, ch)
            nc = find_char(lines[l], c, ch, n1, key)
            return (l, nc), "incl" if key in "ft" else "excl"
        if key in (";", ","):
            if not self.last_f:
                raise Fail()
            kind, ch = self.last_f
            if key == ",":
                kind = {"f": "F", "F": "f", "t": "T", "T": "t"}[kind]
            nc = find_char_repeat(lines[l], c, ch, n1, kind)
            return (l, nc), "incl" if kind in "ft" else "excl"
        if key in ("n", "N"):
            if self.last_pat is None:
                raise VimError("E35: No previous regular expression")
            d = self.last_dir if key == "n" else -self.last_dir
            pos = self.search(self.last_pat, d, n1, (l, c), off=self.last_off, show=True)
            return pos, "line" if self.last_off and self.last_off[:1] in "+-0123456789" else "excl"
        if key in ("*", "#", "g*", "g#"):
            kw = keyword_at(lines[l], c)
            if kw is None:
                raise VimError("E348: No string under cursor")
            word = lines[l][kw[0]:kw[1]]
            whole = key in ("*", "#") and is_kw(word[0])
            pat = (r"\<" + _vesc(word) + r"\>") if whole else _vesc(word)
            self.last_pat = pat
            self.last_dir = 1 if key in ("*", "g*") else -1
            self.last_off = ""
            self.no_hl = False
            self.add_hist("/", pat)
            start = (l, kw[0]) if key in ("*", "g*") else (l, kw[0])
            pos = self.search(pat, self.last_dir, n1, start, smartcase=False, show=True,
                              echo=("/" if self.last_dir > 0 else "?") + pat)
            return pos, "excl"
        if key in ("'", "`", "g'", "g`"):
            pos = self.get_mark(arg)                 # (a global mark may have switched buffers)
            if key in ("'", "g'"):
                return (pos[0], first_nonblank(self.lines[pos[0]])), "line"
            return pos, "excl"
        if key in ("gd", "gD"):
            kw = keyword_at(lines[l], c)
            if kw is None:
                raise VimError("E348: No string under cursor")
            word = lines[l][kw[0]:kw[1]]
            pat = r"\<" + _vesc(word) + r"\>"
            self.last_pat = pat
            self.last_dir = 1
            self.add_hist("/", pat)
            # from the top of the file (Python has no "[[" sections)
            pos = self.search(pat, 1, 1, (len(lines) - 1, len(lines[-1])), wrap=True, show=False)
            return pos, "excl"
        if key in ("gn", "gN"):
            raise Fail()
        if key in ("]s", "[s"):
            from .spell import next_bad
            pos = next_bad(self, (l, c), 1 if key == "]s" else -1, n1)
            return pos, "excl"
        if key in ("[[", "]]", "[]", "]["):
            if key in ("]]", "]["):
                return (len(lines) - 1, 0), "excl"
            return (0, 0), "excl"
        if key in ("[(", "[{", "])", "]}"):
            return self.unmatched(l, c, key), "incl" if key[0] == "]" else "excl"
        if key in ("['", "]'", "[`", "]`"):
            raise Fail()
        raise Fail()

    def sentence_motion(self, l, c, count, fwd):
        lines = self.lines
        text = "\n".join(lines)
        offs = [0]
        for line in lines:
            offs.append(offs[-1] + len(line) + 1)
        pos = offs[l] + c
        starts = [0]
        for m in re.finditer(r"[.!?][)\]\"']*(?:[ \t]+|\n)(?=\S)|\n\n+", text):
            starts.append(m.end())
        starts = sorted(set(starts))
        if fwd:
            cand = [s for s in starts if s > pos]
            if not cand:
                raise Fail()
            p = cand[min(count, len(cand)) - 1]
        else:
            cand = [s for s in starts if s < pos]
            if not cand:
                raise Fail()
            p = cand[-min(count, len(cand))]
        for i in range(len(lines)):
            if offs[i + 1] > p:
                return i, p - offs[i]
        return len(lines) - 1, 0

    def unmatched(self, l, c, key):
        lines = self.lines
        o, cl = ("(", ")") if key in ("[(", "])") else ("{", "}")
        depth = 0
        if key[0] == "[":
            pl, pc = l, c - 1
            while pl >= 0:
                line = lines[pl]
                if pc < 0:
                    pc = len(line) - 1
                while pc >= 0:
                    ch = line[pc] if pc < len(line) else ""
                    if ch == cl:
                        depth += 1
                    elif ch == o:
                        if depth == 0:
                            return pl, pc
                        depth -= 1
                    pc -= 1
                pl -= 1
                pc = -1
            raise Fail()
        pl, pc = l, c + 1
        while pl < len(lines):
            line = lines[pl]
            while pc < len(line):
                ch = line[pc]
                if ch == o:
                    depth += 1
                elif ch == cl:
                    if depth == 0:
                        return pl, pc
                    depth -= 1
                pc += 1
            pl += 1
            pc = 0
        raise Fail()

    def get_mark(self, m):
        b = self.buf
        if m in ("'", "`"):
            if self.win.pcmark is None:
                return (0, 0)
            return self.win.pcmark
        if m.isupper() or m.isdigit():
            g = self.regs.get("__mark_" + m)
            if g is None:
                raise VimError("E20: Mark not set")
            bufnum, pos = g
            if bufnum != b.num:
                tb = self.buffers.get(bufnum)
                if tb is None:
                    raise VimError("E20: Mark not set")
                self.edit_buffer(tb)
            return pos
        if m == "<" or m == ">":
            if "<" not in b.marks:
                raise VimError("E20: Mark not set")
            s, e = b.marks["<"], b.marks[">"]
            if m == "<":
                return min(s, e)
            return max(s, e)
        if m not in b.marks:
            if m in "abcdefghijklmnopqrstuvwxyz[]^.\"":
                raise VimError("E20: Mark not set")
            raise VimError("E78: Unknown mark")
        l, c = b.marks[m]
        l = min(l, len(b.lines) - 1)
        return (l, min(c, max(0, len(b.lines[l]) - 1)))

    def fold_move(self, l, n, down):
        """j / k over closed folds."""
        lines = self.lines
        for _ in range(n):
            if down:
                f = self.closed_fold_at(l)
                nxt = (f[1] if f else l) + 1
                if nxt >= len(lines):
                    break
                l = nxt
                f = self.closed_fold_at(l)
                if f:
                    l = f[0]
            else:
                f = self.closed_fold_at(l)
                prv = (f[0] if f else l) - 1
                if prv < 0:
                    break
                l = prv
                f = self.closed_fold_at(l)
                if f:
                    l = f[0]
        return l

    # ------------------------------------------------------------ operators

    def do_operator(self, cmd, vis=None):
        """An operator on a motion / text object / whole lines, or (vis) on the visual area."""
        op = cmd["op"]
        w, lines = self.win, self.lines
        n1 = cmd["n"] or 1
        start_cursor = tuple(w.cursor)
        block = None
        if op in ("d", "c", "<", ">", "=", "g~", "gu", "gU", "g?", "gq", "gw", "!") and op != "y":
            if op not in ("!",):
                self.check_modifiable()
        if vis is not None:
            s, e, kind, block = vis
        elif cmd["key"] == "line":
            l = w.cursor[0]
            f = self.closed_fold_at(l)
            if f:
                s_l = f[0]
                e_l = l
                cnt = n1
                e_l = l
                for _ in range(cnt):
                    f2 = self.closed_fold_at(e_l)
                    e_l = (f2[1] if f2 else e_l)
                    if _ < cnt - 1:
                        e_l += 1
                l = s_l
            else:
                e_l = l + n1 - 1
                f2 = self.closed_fold_at(min(e_l, len(lines) - 1))
                if f2:
                    e_l = f2[1]
            if e_l >= len(lines):
                if op in ("d", "c", "y") and n1 > 1 and l < len(lines):
                    e_l = len(lines) - 1
                else:
                    raise Fail()
            s, e, kind = (l, 0), (e_l, 0), "line"
        elif cmd.get("tobj"):
            s, e, kind = self.text_object(cmd)
            if kind == "empty":
                if op in ("c",):
                    self.win.cursor = [s[0], s[1]]
                    self.used["text_object"] += 1
                    return self.start_insert("c_empty", cmd)
                return
        elif cmd["key"] in ("/", "?"):
            self.start_cmdline(cmd["key"], from_op=cmd, count=cmd["n"])
            return
        elif cmd["key"] in ("gn", "gN"):
            r = self.find_gn(cmd["key"] == "gN")
            s, e, kind = r[0], r[1], "incl"
        else:
            r = self.motion(cmd, op)
            e, kind = r[0], r[1]
            s = tuple(w.cursor)
            cmd["ostart"] = min(s, tuple(e))
            if getattr(self, "setpcmark_pending", False):
                self.setpcmark_pending = False
            if kind == "line":
                s, e = (min(s[0], e[0]), 0), (max(s[0], e[0]), 0)
            else:
                if e < s:
                    s, e = e, s
                    if kind == "incl" and False:
                        pass
                # exclusive motions: the end isn't included (:help exclusive)
                if kind == "excl":
                    if e[1] == 0 and e[0] > s[0]:
                        fnb = first_nonblank(lines[s[0]])
                        if s[1] <= fnb and lines[s[0]][:s[1]].strip() == "":
                            kind = "line"
                            s, e = (s[0], 0), (e[0] - 1, 0)
                        else:
                            e = (e[0] - 1, len(lines[e[0] - 1]))
                            kind = "incl"
                            if e[1] > 0:
                                e = (e[0], e[1] - 1)
                            else:
                                e = (e[0], 0)
                                if not lines[e[0]]:
                                    kind = "excl_eol"
                    else:
                        if e == s:
                            if op in ("c", "d", "y") and cmd["key"] not in ("w", "W"):
                                if op == "c":
                                    return self.start_insert("i", cmd)
                                return
                        e = (e[0], e[1] - 1) if e[1] > 0 else e
                        if e[1] < 0 or e < s:
                            return
                        kind = "incl"
        self.used["op:" + op] += 1
        return self.apply_op(op, s, e, kind, cmd, block, start_cursor)

    def text_object(self, cmd):
        w, lines = self.win, self.lines
        l, c = w.cursor
        k = cmd["key"]
        n1 = cmd["n"] or 1
        around = k[0] == "a"
        t = k[1]
        self.used["text_object"] += 1
        if t in "wW":
            s, e = tobj_word(lines, l, c, n1, around, t == "W")
            return s, e, "incl"
        if t == "p":
            return tobj_paragraph(lines, l, n1, around)
        if t == "s":
            return tobj_sentence(lines, l, c, around)
        if t in "\"'`":
            return tobj_quote(lines, l, c, t, around)
        if t == "t":
            return tobj_tag(lines, l, c, around, n1)
        pairs = {"(": "()", ")": "()", "b": "()", "[": "[]", "]": "[]", "{": "{}", "}": "{}", "B": "{}",
                 "<": "<>", ">": "<>"}
        o, cl = pairs[t]
        s, e, kind = tobj_bracket(lines, l, c, o, cl, around, n1)
        if kind == "incl_eol":
            kind = "incl"
        return s, e, kind

    def apply_op(self, op, s, e, kind, cmd, block=None, start_cursor=None):
        """Do operator op on s..e (kind: line / incl / block)."""
        b, w = self.buf, self.win
        reg = cmd.get("reg")
        lines = b.lines
        if kind == "line":
            l1, l2 = s[0], e[0]
        if op == "y":
            text, rkind = self.region_text(s, e, kind, block)
            self.reg_set(reg, text, rkind, "y")
            b.marks["["] = (s[0], s[1] if kind != "line" else 0)
            b.marks["]"] = (e[0], e[1] if kind != "line" else max(0, len(lines[e[0]]) - 1))
            if kind == "line":
                if cmd.get("vis"):
                    self.set_cursor(l1, 0)
                elif cmd.get("tobj") or cmd["key"] == "line":
                    if cmd.get("tobj"):
                        self.set_cursor(l1, 0)
                elif cmd.get("ostart") is not None:
                    st = cmd["ostart"]
                    self.set_cursor(st[0], st[1])
            elif block:
                self.set_cursor(s[0], block[0])
            else:
                self.set_cursor(s[0], s[1])
            nlines = e[0] - s[0] + 1
            if nlines > self.opt("report"):
                self.message(f"{nlines} lines yanked" if not block else f"block of {nlines} lines yanked")
            return
        if op == "d" and kind == "incl" and not block and e[0] > s[0] and not cmd.get("vis"):
            rest = lines[e[0]][e[1] + 1:] if e[1] < len(lines[e[0]]) else ""
            if not rest.strip() and lines[s[0]][:s[1]].strip() == "":
                kind = "line"
                l1, l2 = s[0], e[0]
        if op in ("d", "c"):
            text, rkind = self.region_text(s, e, kind, block)
            big = cmd["key"] in ("%", "(", ")", "`", "/", "?", "n", "N", "{", "}") and kind != "line"
            if op == "c" and kind == "line" and not block:
                # cc / cj: the lines go, one empty line (keeping the indent) remains
                self.reg_set(reg, text, "V", "d")
                ind = ""
                if self.opt("autoindent"):
                    ind = re.match(r"[ \t]*", lines[l1]).group(0)
                if self.indent_kind():   # then indentexpr, which may keep it
                    width = self.indent_for(l1, lines[:l1] + [ind] + lines[l2 + 1:])
                    if width is not None:
                        from .text import make_indent
                        ind = make_indent(width, self.opt("expandtab"), self.ts())
                self.replace_lines(b, l1, l2 + 1, [ind])
                w.cursor = [l1, len(ind)]
                return self.start_insert("c", cmd)
            if block:
                self.reg_set(reg, text, "b", "d")
                self.delete_block(s[0], e[0], block)
                self.set_cursor(s[0], block[0])
                if op == "c":
                    return self.start_insert("block_c", cmd, block=(s[0], e[0], block[0]))
                return
            self.reg_set(reg, text, rkind, "dbig" if big else "d")
            if kind == "line":
                n = l2 - l1 + 1
                self.replace_lines(b, l1, l2 + 1, [])
                nl = min(l1, len(lines) - 1)
                self.go_line(nl, sol=True)
                if n > self.opt("report"):
                    self.message(f"{n} fewer lines")
                return
            # charwise
            self.delete_range(b, s, e)
            w.cursor = [s[0], s[1]]
            if op == "c":
                return self.start_insert("c", cmd)
            self.clamp()
            w.curswant = vcol(lines[w.cursor[0]], w.cursor[1], self.ts())
            if e[0] > s[0] and e[0] - s[0] > self.opt("report") - 1:
                n = e[0] - s[0]
                if n > self.opt("report"):
                    self.message(f"{n} fewer lines")
            return
        if op in ("<", ">"):
            l1, l2 = s[0], e[0]
            times = (cmd["n"] or 1) if cmd.get("vis") else 1
            for _ in range(times):
                self.shift_lines(l1, l2, 1 if op == ">" else -1)
            self.go_line(l1, sol=True)
            n = l2 - l1 + 1
            if n > self.opt("report"):
                self.message(f"{n} lines {op}ed {times} time" + ("s" if times > 1 else ""))
            return
        if op in ("g~", "gu", "gU", "g?"):
            f = {"g~": str.swapcase, "gu": str.lower, "gU": str.upper,
                 "g?": lambda t: t.translate(_ROT13)}[op]
            self.map_region(s, e, kind, block, f)
            if kind == "line":
                self.set_cursor(s[0], start_cursor[1] if start_cursor and start_cursor[0] == s[0] else w.cursor[1], want=False)
                if cmd.get("vis"):
                    self.set_cursor(s[0], 0)
            elif block:
                self.set_cursor(s[0], block[0])
            else:
                self.set_cursor(s[0], s[1])
            n = e[0] - s[0] + 1
            if n > self.opt("report"):
                self.message(f"{n} lines changed")
            return
        if op == "=":
            l1, l2 = s[0], e[0]
            self.reindent(l1, l2)
            self.go_line(l1, sol=True)
            n = l2 - l1 + 1
            if n > self.opt("report"):
                self.message(f"{n} lines indented ")
            return
        if op in ("gq", "gw"):
            l1, l2 = s[0], e[0]
            keep = tuple(w.cursor)
            last = self.format_lines(l1, l2)
            if op == "gw":
                self.set_cursor(*keep)
            else:
                self.go_line(last, sol=True)
            return
        if op == "!":
            l1, l2 = s[0], e[0]
            rng = "." if l1 == l2 else f".,.+{l2 - l1}"
            if cmd.get("vis"):
                rng = "'<,'>"
            self.win.cursor = [l1, w.cursor[1]]
            self.start_cmdline(":", text=rng + "!")
            return
        if op == "zf":
            self.create_fold(s[0], e[0])
            return
        raise Fail()

    def region_text(self, s, e, kind, block=None):
        lines = self.lines
        if block:
            c1, c2 = block
            out = []
            ts = self.ts()
            for l in range(s[0], e[0] + 1):
                line = lines[l]
                a = col_at_vcol(line, c1, ts, past_end=True) if c1 < MAXCOL else len(line)
                if c2 >= MAXCOL:
                    out.append(line[a:])
                else:
                    z = col_at_vcol(line, c2, ts, past_end=True)
                    if z < len(line):
                        z += 1
                    out.append(line[a:z])
            return "\n".join(out), "b"
        if kind == "line":
            return "\n".join(lines[s[0]:e[0] + 1]) + "\n", "V"
        return self.get_text(self.buf, s, e), "v"

    def delete_block(self, l1, l2, block):
        c1, c2 = block
        ts = self.ts()
        for l in range(l1, l2 + 1):
            line = self.lines[l]
            a = col_at_vcol(line, c1, ts, past_end=True)
            if a >= len(line):
                continue
            if c2 >= MAXCOL:
                z = len(line)
            else:
                z = col_at_vcol(line, c2, ts, past_end=True)
                z = min(len(line), z + 1)
            self.set_line(self.buf, l, line[:a] + line[z:])

    def map_region(self, s, e, kind, block, f):
        b = self.buf
        if kind == "line":
            for l in range(s[0], e[0] + 1):
                self.set_line(b, l, f(b.lines[l]))
            return
        if block:
            c1, c2 = block
            ts = self.ts()
            for l in range(s[0], e[0] + 1):
                line = b.lines[l]
                a = col_at_vcol(line, c1, ts, past_end=True)
                z = len(line) if c2 >= MAXCOL else min(len(line), col_at_vcol(line, c2, ts, past_end=True) + 1)
                self.set_line(b, l, line[:a] + f(line[a:z]) + line[z:])
            return
        for l in range(s[0], e[0] + 1):
            line = b.lines[l]
            a = s[1] if l == s[0] else 0
            z = e[1] + 1 if l == e[0] else len(line)
            self.set_line(b, l, line[:a] + f(line[a:z]) + line[z:])

    def shift_lines(self, l1, l2, d, amount=1):
        b = self.buf
        sw, ts, et = self.sw(), self.ts(), self.opt("expandtab")
        for l in range(l1, l2 + 1):
            line = b.lines[l]
            if not line.strip() and d > 0:
                continue          # empty lines aren't shifted right
            ind = indent_of(line, ts)
            if self.opt("shiftround"):
                ind = (ind // sw + d) * sw if d > 0 else ((ind - 1) // sw) * sw if ind else 0
            else:
                ind = max(0, ind + d * sw * amount)
            self.set_line(b, l, set_indent(line, ind, et, ts) if line.strip() else "")

    def indent_kind(self, buf=None):
        b = buf or self.buf
        ft = self.opt("filetype", b)
        if not self.filetype_on:
            if self.opt("cindent", b):
                return "brace"
            return None
        if ft == "python":
            return "python"
        if ft == "yaml":
            return "yaml"
        if ft in ("json", "c", "cpp", "javascript", "typescript", "java", "rust", "go", "css"):
            return "brace"
        if self.opt("cindent", b):
            return "brace"
        return None

    def indent_for(self, l, lines=None, buf=None):
        """The indent width indentexpr / cindent gives line l (as it stands in `lines`), or None to keep it."""
        b = buf or self.buf
        lines = lines if lines is not None else b.lines
        kind = self.indent_kind(b)
        if kind == "python":
            return python_indent(lines, l, self.sw(b), self.ts(b))
        if kind == "yaml":
            return yaml_indent(lines, l, self.sw(b), self.ts(b))
        if kind == "brace":
            ft = self.opt("filetype", b)
            return brace_indent(lines, l, self.sw(b), self.ts(b),
                                cindent=ft not in ("javascript", "typescript", "json"))
        return None

    def reindent(self, l1, l2):
        b = self.buf
        ts, et = self.ts(), self.opt("expandtab")
        for l in range(l1, l2 + 1):
            line = b.lines[l]
            if not line.strip():
                if line:
                    self.set_line(b, l, "")
                continue
            kind = self.indent_kind()
            ind = self.indent_for(l)
            if ind is None and kind == "python":
                continue
            if ind is None:
                # no indent program: like 'autoindent', the previous non-blank line's indent
                p = l - 1
                while p >= 0 and not b.lines[p].strip():
                    p -= 1
                ind = indent_of(b.lines[p], ts) if p >= 0 else 0
            self.set_line(b, l, set_indent(line, ind, et, ts))

    def format_lines(self, l1, l2):
        """gq: join and re-wrap paragraphs to 'textwidth' (79 when it's 0). -> the last line."""
        b = self.buf
        tw = self.opt("textwidth") or 79
        out = []
        para = []
        ind = None

        def flush():
            nonlocal para, ind
            if not para:
                return
            words = " ".join(x.strip() for x in para).split()
            cur = ind
            for wd in words:
                if cur.strip() and len(cur) + 1 + len(wd) > tw:
                    out.append(cur)
                    cur = ind + wd
                elif cur.strip():
                    cur += " " + wd
                else:
                    cur = ind + wd
            out.append(cur)
            para = []
        for l in range(l1, l2 + 1):
            line = b.lines[l]
            if not line.strip():
                flush()
                out.append("")
                ind = None
                continue
            if ind is None:
                ind = re.match(r"[ \t]*", line).group(0)
            para.append(line)
        flush()
        self.replace_lines(b, l1, l2 + 1, out)
        return l1 + len(out) - 1

    # ------------------------------------------------------------ undo

    def commit_undo(self, buf=None):
        bufs = [buf] if buf else {w.buf.num: w.buf for w in self.all_windows(all_tabs=True)}.values()
        for b in bufs:
            if tuple(b.lines) != b.undo.cur.lines:
                cur = getattr(self, "change_cursor", None)
                if cur is None or cur[0] != b.num:
                    cur = (b.num, (0, 0))
                b.undo.commit(b.lines, cur[1], self.now)
                # help, quickfix, netrw and terminal buffers are never "modified" (no E37 for them)
                b.modified = b.undo.cur.seq != b.saved_seq and b.kind in ("file", "scratch")
                if b.opts.get("undofile") and b.path:
                    self.undo_store[b.path] = b.undo
        self.change_cursor = None

    def mark_change_start(self):
        """Where the cursor was when a change started: undo returns there."""
        if getattr(self, "change_cursor", None) is None:
            self.change_cursor = (self.buf.num, tuple(self.win.cursor))

    def undo_to(self, st, redo=False):
        b, w = self.buf, self.win
        old = list(b.lines)
        b.lines[:] = list(st.lines)
        # the first changed line
        top = 0
        while top < len(old) and top < len(b.lines) and old[top] == b.lines[top]:
            top += 1
        top = min(top, len(b.lines) - 1)
        moved = st if redo else b.undo.cur
        cur_before = moved.cursor
        l = top
        if cur_before[0] + 1 == l and l > 0:
            l -= 1
        if cur_before[0] == l:
            w.cursor = [l, cur_before[1]]
            self.clamp()
        else:
            self.set_cursor(l)
        w.curswant = vcol(b.lines[w.cursor[0]], w.cursor[1], self.ts())
        b.undo.cur = st
        b.modified = st.seq != b.saved_seq
        b.uline = None
        d = len(b.lines) - len(old)
        what = (f"{abs(d)} line{'s' if abs(d) != 1 else ''} {'less' if d < 0 else 'more'}" if d else "1 change")
        return what

    def do_undo(self, count):
        b = self.buf
        for i in range(count):
            st = b.undo.cur
            if st.parent is None:
                if i == 0:
                    self.message("Already at oldest change")
                    return
                break
            what = self.undo_to(st.parent)
            self.message(f"{what}; before #{st.seq}  {self.ago(st.time)}")

    def do_redo(self, count):
        b = self.buf
        for i in range(count):
            st = b.undo.cur
            if not st.children:
                if i == 0:
                    self.message("Already at newest change")
                    return
                break
            nxt = st.children[st.cur_child]
            what = self.undo_to(nxt, redo=True)
            self.message(f"{what}; after #{nxt.seq}  {self.ago(nxt.time)}")

    def undo_time(self, steps=None, ms=None, file_saves=None):
        """g- g+ :earlier :later: travel through the states in time order."""
        b = self.buf
        states = b.undo.states
        cur = b.undo.cur.seq
        if steps is not None:
            target = max(0, min(len(states) - 1, cur + steps))
        else:
            t = b.undo.cur.time + ms
            if ms < 0:
                cands = [s for s in states if s.time <= t]
                target = cands[-1].seq if cands else 0
            else:
                cands = [s for s in states if s.time <= t]
                target = cands[-1].seq if cands else cur
        st = states[target]
        if st is b.undo.cur:
            self.message("Already at oldest change" if (steps or ms or 0) < 0 else "Already at newest change")
            return
        self.undo_to(st, redo=target > cur)
        # make the path to it current, so u / <C-r> follow it
        node = st
        while node.parent is not None:
            node.parent.cur_child = node.parent.children.index(node)
            node = node.parent
        self.message(f"{'before' if target < cur else 'after'} #{max(target, 1) if target else 0}  {self.ago(st.time)}")

    def ago(self, t):
        s = max(0, (self.now - t) // 1000)
        if s < 100:
            return f"{s} second{'s' if s != 1 else ''} ago"
        return f"{s // 60} minutes ago"

    def line_undo(self):
        b, w = self.buf, self.win
        if b.uline is None or b.uline[0] >= len(b.lines):
            raise Fail()
        l, old = b.uline
        cur = b.lines[l]
        self.mark_change_start()
        self.set_line(b, l, old)
        b.uline = (l, cur)
        w.cursor = [l, 0]

    # ------------------------------------------------------------ the dot command

    def set_dot(self, keys, count, reg=None, visual=None):
        body = self.strip_counts(keys)
        self.dot = {"keys": body, "count": count, "reg": reg, "visual": visual}

    def strip_counts(self, keys):
        """Remove the register and the counts from a command's keys: '"a2d3w' -> 'dw'."""
        out = list(keys)
        i = 0
        if out[:1] == ['"']:
            out = out[2:]
        while out and out[0].isdigit() and out[0] != "0":
            out.pop(0)
        while out and out[0].isdigit():
            out.pop(0)
        # a count after the operator
        op, j = None, 0
        if out:
            if out[0] in "dcy<>=!":
                j = 1
            elif len(out) > 1 and "".join(out[:2]) in OPERATORS:
                j = 2
        if j:
            k = j
            while k < len(out) and out[k].isdigit() and not (out[k] == "0" and k == j):
                k += 1
            out = out[:j] + out[k:]
        return out

    def repeat_dot(self, count):
        if self.dot is None:
            raise Fail()
        d = self.dot
        n = count if count else d["count"]
        keys = []
        reg = d["reg"]
        if reg and reg.isdigit() and reg != "0" and reg != "9":
            reg = str(int(reg) + 1)
            d["reg"] = reg
        if reg:
            keys += ['"', reg]
        if d["visual"]:
            vmode, nl, nc = d["visual"]
            l, c = self.win.cursor
            self.vis = {"anchor": (l, c), "mode": vmode}
            el = min(len(self.lines) - 1, l + nl)
            ec = c + nc if nl == 0 else nc
            self.win.cursor = [el, min(max(0, len(self.lines[el]) - 1), ec)]
            if vmode == "\x16":
                self.win.curswant = vcol(self.lines[l], c, self.ts()) + nc
            self.redoing = True
            for k in reversed(d["keys"]):
                self.typeahead.appendleft((k, False, None))
            return
        if n:
            keys += list(str(n))
        keys += d["keys"]
        self.redoing = True
        for k in reversed(keys):
            self.typeahead.appendleft((k, False, None))

    # ------------------------------------------------------------ the rest of the normal commands

    def do_command(self, cmd):
        from .commands import normal_command
        return normal_command(self, cmd)

    # ------------------------------------------------------------ helpers used by several modes

    def visible_range(self, w):
        """First and last buffer line shown in window w."""
        top = w.topline
        rows = 0
        l = top
        lines = w.buf.lines
        last = top
        while l < len(lines):
            f = self.closed_fold_at(l, w)
            h = 1 if f else self.line_rows(lines[l], w)
            if rows + h > w.height:
                break
            rows += h
            last = l
            l = (f[1] if f else l) + 1
        return top, last

    def line_rows(self, line, w):
        if not self.opt("wrap", win=w):
            return 1
        width = max(1, w.width - self.numberwidth(w))
        dw = vcol(line, len(line), self.ts(w.buf)) + (1 if self.opt("list", win=w) else 0)
        return max(1, (dw + width - 1) // width)

    def numberwidth(self, w):
        if not (self.opt("number", win=w) or self.opt("relativenumber", win=w)):
            return 0
        n = len(w.buf.lines)
        digits = len(str(n)) if self.opt("number", win=w) else len(str(w.height))
        return max(self.opt("numberwidth", win=w) - 1, digits) + 1

    def scroll_to_cursor(self, win=None):
        """Keep the cursor on screen, 'scrolloff' lines from the edges (Vim's update_topline): a cursor far off the
        screen is put in the middle, a near one scrolls just enough."""
        w = win or self.win
        if w.height <= 0:
            return
        lines = w.buf.lines
        n = len(lines)
        l = w.cursor[0]
        so = min(self.opt("scrolloff"), (w.height - 1) // 2)
        w.topline = max(0, min(w.topline, n - 1))
        top, last = self.visible_range(w)
        if l < w.topline + so and w.topline > 0:
            # above: Vim centres once the cursor is half a window (less one line) from topline + 'scrolloff'
            if w.topline + so - l >= max(2, w.height // 2 - 1):
                self._centre(w, l)
            else:
                w.topline = max(0, l - so)
            return
        if l > last - so and last < n - 1 or l > last:
            # below: centred from half a window (plus two lines) past the last line shown, whatever 'scrolloff' is
            # (measured against Vim 9.1 for window heights 7 to 23)
            if l - last >= w.height // 2 + 2:
                self._centre(w, l)
                return
            while True:
                top, last = self.visible_range(w)
                if min(n - 1, l + so) <= last or w.topline >= l:
                    break
                w.topline += 1

    def _centre(self, w, l):
        lines = w.buf.lines
        # the cursor line in the middle, unless the file ends first (then the last line goes at the bottom)
        below_avail = sum(self.line_rows(x, w) for x in lines[l + 1:l + w.height])
        above = max((w.height - 1) // 2, w.height - 1 - below_avail)
        t = l
        rows = 0
        while t > 0 and rows + self.line_rows(lines[t - 1], w) <= above:
            t -= 1
            rows += self.line_rows(lines[t], w)
        w.topline = t
        # don't leave empty lines past the end when the file is long enough
        top, last = self.visible_range(w)

    def closed_fold_at(self, l, win=None):
        from .folds import closed_fold_at
        return closed_fold_at(self, l, win or self.win)

    def create_fold(self, l1, l2):
        from .folds import create_fold
        create_fold(self, l1, l2)

    def add_hist(self, kind, text):
        h = self.hist[kind]
        if text in h:
            h.remove(text)
        h.append(text)
        if len(h) > max(1, self.opt("history")):
            del h[0]

    def fire(self, event, buf, arg=None):
        from .ex import fire_autocmds
        fire_autocmds(self, event, buf, arg)

    def source_text(self, text, path):
        from .ex import source_lines
        source_lines(self, text.split("\n"), path)

    def find_gn(self, back=False):
        """gn: the next match of the last search (or the one under the cursor)."""
        if self.last_pat is None:
            raise VimError("E35: No previous regular expression")
        from .search import match_at_or_after
        r = match_at_or_after(self, self.last_pat, tuple(self.win.cursor), -1 if back else 1)
        if r is None:
            raise Fail()
        return r


def _vesc(word):
    return re.sub(r"([\\/.*$^~\[\]])", r"\\\1", word)


_ROT13 = str.maketrans("ABCDEFGHIJKLMabcdefghijklmNOPQRSTUVWXYZnopqrstuvwxyz",
                       "NOPQRSTUVWXYZnopqrstuvwxyzABCDEFGHIJKLMabcdefghijklm")

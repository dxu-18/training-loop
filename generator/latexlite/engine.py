"""The interpreter: tokens -> a document (model.py), with LaTeX's errors and warnings.

Commands are dispatched by name; environments by \\begin{name}. User macros (\\newcommand ...) expand into the
token stream. A command or environment works only if the class or a loaded package provides it
(packages.py), as in real LaTeX. Errors are recorded the way pdflatex prints them (`! Undefined control
sequence.` with the line); like TeX, the engine then carries on, so the page still shows.
"""

from __future__ import annotations

import re
from collections import Counter

from . import mathmode, packages as P
from .model import PLAIN, SIZE_ORDER, Counters, Style, text
from .tokens import Tok, detokenize, tokenize

SECTION_CMDS = ("part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph")
DISPLAY_ENVS = {"equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
                "flalign", "flalign*", "alignat", "alignat*", "eqnarray", "eqnarray*", "displaymath"}
QUOTES = {"``": "“", "''": "”", "`": "‘", "'": "’", "---": "—", "--": "–",
          "?`": "¿", "!`": "¡", "<<": "«", ">>": "»"}
SYMBOLS = {"%": "%", "$": "$", "&": "&", "#": "#", "_": "_", "{": "{", "}": "}", "textbackslash": "\\",
           "textasciitilde": "~", "textasciicircum": "^", "textbar": "|", "textless": "<", "textgreater": ">",
           "textbullet": "•", "textendash": "–", "textemdash": "—", "ldots": "…",
           "dots": "…", "textellipsis": "…", "P": "¶", "S": "§", "copyright": "©",
           "dag": "†", "ddag": "‡", "pounds": "£", "slash": "/", "i": "ı", "j": "ȷ",
           "l": "ł", "L": "Ł", "o": "ø", "O": "Ø", "oe": "œ", "OE": "Œ",
           "ae": "æ", "AE": "Æ", "aa": "å", "AA": "Å", "ss": "ß",
           "textquoteleft": "‘", "textquoteright": "’", "textquotedblleft": "“",
           "textquotedblright": "”", "guillemotleft": "«", "guillemotright": "»",
           "@": "", "/": "", "textregistered": "®", "texttrademark": "™", "textdegree": "°"}
ACCENTS = {"'": "́", "`": "̀", "^": "̂", '"': "̈", "~": "̃", "=": "̄",
           ".": "̇", "u": "̆", "v": "̌", "H": "̋", "c": "̧", "k": "̨", "r": "̊",
           "d": "̣", "b": "̱"}
SPACES = {",": "thin", "quad": "quad", "qquad": "qquad", " ": "space", "enspace": "en", "thinspace": "thin",
          ";": "thick", ":": "med", "!": "negthin", "negthinspace": "negthin", "hfill": "hfill", "dotfill": "dotfill",
          "hrulefill": "hrulefill"}
FONT_CMDS = {"textbf": {"bold": True}, "textit": {"shape": "it"}, "textsl": {"shape": "sl"},
             "textsc": {"shape": "sc"}, "textup": {"shape": "n"}, "textmd": {"bold": False},
             "texttt": {"family": "tt"}, "textsf": {"family": "sf"}, "textrm": {"family": "rm"},
             "textnormal": {"bold": False, "shape": "n", "family": "rm"}}
FONT_DECLS = {"bfseries": {"bold": True}, "itshape": {"shape": "it"}, "slshape": {"shape": "sl"},
              "scshape": {"shape": "sc"}, "upshape": {"shape": "n"}, "mdseries": {"bold": False},
              "ttfamily": {"family": "tt"}, "sffamily": {"family": "sf"}, "rmfamily": {"family": "rm"},
              "normalfont": {"bold": False, "shape": "n", "family": "rm"},
              "bf": {"bold": True, "shape": "n", "family": "rm"}, "it": {"shape": "it", "bold": False, "family": "rm"},
              "tt": {"family": "tt", "bold": False, "shape": "n"}, "sf": {"family": "sf", "bold": False, "shape": "n"},
              "rm": {"family": "rm", "bold": False, "shape": "n"}, "sc": {"shape": "sc"}, "sl": {"shape": "sl"}}
IMPLICIT = {"beamer": ["amsmath", "amssymb", "amsthm", "xcolor", "graphicx", "hyperref"],
            "amsart": ["amsmath", "amssymb", "amsthm"]}


class TeXError(Exception):
    pass


class Stop(Exception):
    """\\end{document}, or an emergency stop."""


class Engine:
    def __init__(self, files: dict[str, str], main: str = "main.tex", body_offset: dict | None = None):
        self.files = files
        self.main = main
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.cls = None
        self.cls_opts: list[str] = []
        self.pkgs: dict[str, list[str]] = {}
        self.macros: dict[str, dict] = {}
        self.envs: dict[str, dict] = {}
        self.theorems: dict[str, dict] = {}
        self.thm_style = "plain"
        self.counters = Counters("article")
        self.labels: dict[str, dict] = {}
        self.cur_label = {"text": "", "kind": None, "env": None}
        self.used: Counter = Counter()
        self.colors: dict[str, str] = {}
        self.lengths: dict[str, str] = {}
        self.title = {"title": None, "author": None, "date": None, "thanks": []}
        self.toc: list[dict] = []
        self.footnotes: list[dict] = []
        self.page = {"style": "plain", "head": {}, "foot": {}, "headrule": False, "geometry": {}, "spacing": 1.0,
                     "parskip": None, "parindent": None, "twocolumn": False, "this": None}
        self.bib = {"style": None, "files": [], "cited": [], "entries": {}, "printed": False}
        self.appendix = False
        self.in_document = False
        self.ended = False
        self.style = PLAIN
        self.frames: list[dict] = []
        self.toks: list[Tok] = []
        self.pos = 0
        self.mode_stack: list[str] = []
        self.list_depth = {"itemize": 0, "enumerate": 0, "description": 0}
        self.last_tok: Tok | None = None
        self.expansions = 0
        self.beamer = None          # beamer.Beamer when the class is beamer
        self.graphics_paths: list[str] = []
        self.lst_settings: dict = {}
        self.tikz_styles: dict = {}
        self.pgfplots_settings: dict = {}
        self.env_line_stack: list[tuple] = []
        self.float_stack: list[dict] = []
        self.secnumdepth = 3
        self.refresh_available()
        self.push_frame(kind="preamble", align="j")

    # ------------------------------------------------------------ messages

    def error(self, msg: str, tok: Tok | None = None, help_: str = ""):
        tok = tok or self.last_tok
        ctx_before, ctx_after = "", ""
        if tok is not None:
            src = self.files.get(tok.file, "")
            lines = src.split("\n")
            if 0 < tok.line <= len(lines):
                ln = lines[tok.line - 1]
                end = tok.col + len(tok.text()) if tok.k != "verb" else tok.col
                ctx_before, ctx_after = ln[:end], ln[end:]
        if len(self.errors) < 100:
            self.errors.append({"msg": msg, "file": tok.file if tok else self.main, "line": tok.line if tok else 0,
                                "before": ctx_before, "after": ctx_after, "help": help_})

    def warn(self, msg: str, tok: Tok | None = None):
        tok = tok or self.last_tok
        if len(self.warnings) < 100:
            self.warnings.append({"msg": msg, "file": tok.file if tok else self.main, "line": tok.line if tok else 0})

    # ------------------------------------------------------------ the token stream

    def peek(self, skip_space=False) -> Tok | None:
        i = self.pos
        if skip_space:
            while i < len(self.toks) and self.toks[i].k == "sp":
                i += 1
        return self.toks[i] if i < len(self.toks) else None

    def next(self) -> Tok | None:
        if self.pos < len(self.toks):
            t = self.toks[self.pos]
            self.pos += 1
            self.last_tok = t
            return t
        return None

    def skip_spaces(self):
        while self.pos < len(self.toks) and self.toks[self.pos].k == "sp":
            self.pos += 1

    def push(self, toks: list[Tok]):
        self.toks[self.pos:self.pos] = toks

    def read_arg(self, what: str = "", allow_par=False) -> list[Tok]:
        """A mandatory argument: a {group} or one token."""
        self.skip_spaces()
        t = self.next()
        if t is None:
            self.error(f"File ended while scanning use of \\{what}.", self.last_tok,
                       f"\\{what} needs an argument in braces." if what else "")
            return []
        if t.k == "bg":
            return self._group_body(t, what, allow_par)
        if t.k == "eg":
            self.error(f"Argument of \\{what} has an extra }}.", t, "Something is missing before this }.")
            self.pos -= 1
            return []
        if t.k == "par" and not allow_par:
            self.error(f"Paragraph ended before \\{what} was complete.", t, "A blank line ended the argument.")
            self.pos -= 1
            return []
        return [t]

    def _group_body(self, open_tok: Tok, what="", allow_par=True) -> list[Tok]:
        depth = 1
        start = self.pos
        while self.pos < len(self.toks):
            t = self.toks[self.pos]
            if t.k == "bg":
                depth += 1
            elif t.k == "eg":
                depth -= 1
                if depth == 0:
                    body = self.toks[start:self.pos]
                    self.pos += 1
                    return body
            elif t.k == "par" and not allow_par and what:
                self.error(f"Paragraph ended before \\{what} was complete.", t,
                           "A blank line inside an argument: a `}` is probably missing.")
                body = self.toks[start:self.pos]
                return body
            self.pos += 1
        self.error(f"File ended while scanning use of \\{what}." if what else "Missing } inserted.", open_tok,
                   "A `{` is never closed.")
        body = self.toks[start:]
        self.pos = len(self.toks)
        return body

    def read_opt(self) -> list[Tok] | None:
        """[optional] argument (brackets inside braces don't count)."""
        i = self.pos
        while i < len(self.toks) and self.toks[i].k == "sp":
            i += 1
        if i < len(self.toks) and self.toks[i].k == "ch" and self.toks[i].v == "[":
            depth = 0
            j = i + 1
            while j < len(self.toks):
                t = self.toks[j]
                if t.k == "bg":
                    depth += 1
                elif t.k == "eg":
                    depth -= 1
                elif t.k == "ch" and t.v == "]" and depth == 0:
                    self.pos = j + 1
                    return self.toks[i + 1:j]
                elif t.k == "par":
                    break
                j += 1
        return None

    def read_star(self) -> bool:
        t = self.peek()
        if t is not None and t.k == "ch" and t.v == "*":
            self.pos += 1
            return True
        return False

    def read_overlay(self) -> str | None:
        """<2-> (beamer overlay specification)."""
        t = self.peek()
        if t is not None and t.k == "ch" and t.v == "<":
            j = self.pos + 1
            spec = []
            while j < len(self.toks) and not (self.toks[j].k == "ch" and self.toks[j].v == ">"):
                spec.append(self.toks[j].text())
                j += 1
            if j < len(self.toks):
                self.pos = j + 1
                return "".join(spec)
        return None

    def arg_text(self, what="") -> str:
        return detokenize(self.read_arg(what)).strip()

    @staticmethod
    def kv(toks: list[Tok] | None) -> dict:
        """key=value options, top-level commas only."""
        if not toks:
            return {}
        s = detokenize(toks)
        out = {}
        depth = 0
        cur = []
        parts = []
        for c in s:
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
            if c == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(c)
        parts.append("".join(cur))
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if "=" in p:
                k, v = p.split("=", 1)
                v = v.strip()
                if v.startswith("{") and v.endswith("}"):
                    v = v[1:-1]
                out[k.strip()] = v
            else:
                out[p] = True
        return out

    # ------------------------------------------------------------ containers (paragraphs and blocks)

    def frame(self) -> dict:
        return self.frames[-1]

    def push_frame(self, **kw) -> dict:
        f = {"blocks": [], "par": None, "align": kw.get("align", self.frames[-1]["align"] if self.frames else "j"),
             "noindent": False, "kind": kw.get("kind", "box"), "inline_only": kw.get("inline_only", False)}
        self.frames.append(f)
        return f

    def pop_frame(self) -> dict:
        self.end_par()
        return self.frames.pop()

    def start_par(self):
        f = self.frame()
        if f["par"] is None:
            indent = not f["noindent"]
            f["noindent"] = False
            f["par"] = {"t": "par", "c": [], "align": f["align"], "indent": indent}
        return f["par"]

    def end_par(self):
        f = self.frame()
        p = f["par"]
        if p is not None:
            # trailing spaces don't make a line
            while p["c"] and p["c"][-1]["t"] == "text" and not p["c"][-1]["s"].strip(" "):
                p["c"].pop()
            if p["c"] and p["c"][-1]["t"] == "text":
                p["c"][-1] = {**p["c"][-1], "s": p["c"][-1]["s"].rstrip(" ")}
            if p["c"]:
                f["blocks"].append(p)
            f["par"] = None

    def add_block(self, b: dict):
        self.end_par()
        self.frame()["blocks"].append(b)

    def add_inline(self, node: dict):
        if not self.in_document:
            if node["t"] == "text" and not node["s"].strip():
                return
            if not getattr(self, "_preamble_text_reported", False):
                self._preamble_text_reported = True
                self.error("LaTeX Error: Missing \\begin{document}.", self.last_tok,
                           "Text before \\begin{document}: everything that prints goes in the body.")
            return
        f = self.frame()
        if f["par"] is None and node["t"] == "text" and not node["s"].strip(" "):
            return          # spaces in vertical mode
        p = self.start_par()
        if node["t"] == "text" and p["c"] and p["c"][-1]["t"] == "text" and p["c"][-1]["st"] == node["st"]:
            s = p["c"][-1]["s"] + node["s"]
            s = re.sub(r" {2,}", " ", s)
            p["c"][-1] = {**p["c"][-1], "s": s}
        else:
            if node["t"] == "text" and not p["c"]:
                node = {**node, "s": node["s"].lstrip(" ")}
                if not node["s"]:
                    return
            if node["t"] == "text" and node["s"].startswith(" ") and p["c"] and p["c"][-1]["t"] == "text" and \
                    p["c"][-1]["s"].endswith(" "):
                node = {**node, "s": node["s"][1:]}
                if not node["s"]:
                    return
            p["c"].append(node)

    def emit_text(self, s: str):
        self.add_inline(text(s, self.style))

    def capture(self, toks: list[Tok], style: Style | None = None, kind="capture") -> list[dict]:
        """Typeset tokens on their own and return the inlines (a title, a caption, an item label)."""
        saved = self.style
        if style is not None:
            self.style = style
        self.push_frame(kind=kind, align="l")
        self.run_tokens(toks)
        f = self.pop_frame()
        self.style = saved
        out = []
        for b in f["blocks"]:
            if b["t"] == "par":
                if out:
                    out.append(text(" ", self.style))
                out += b["c"]
        return out

    def capture_blocks(self, toks: list[Tok], kind="box", align="j") -> list[dict]:
        saved = self.style
        self.push_frame(kind=kind, align=align)
        self.run_tokens(toks)
        f = self.pop_frame()
        self.style = saved
        return f["blocks"]

    def run_tokens(self, toks: list[Tok]):
        """Process a token list where we are (an argument, a macro body)."""
        saved_toks, saved_pos = self.toks, self.pos
        self.toks, self.pos = list(toks), 0
        try:
            self.run()
        finally:
            self.toks, self.pos = saved_toks, saved_pos

    # ------------------------------------------------------------ the main loop

    def run(self, stop_env: str | None = None, group: bool = False, open_tok: Tok | None = None):
        """Process tokens until the end of the list, the `}` closing this group, or \\end{stop_env}."""
        while True:
            t = self.next()
            if t is None:
                if group:
                    self.error("Missing } inserted.", open_tok, "A `{` is never closed.")
                if stop_env:
                    return "eof"
                return None
            k = t.k
            if k == "sp":
                f = self.frame()
                if f["par"] is not None:
                    self.add_inline(text(" ", self.style))
            elif k == "par":
                self.end_par()
            elif k == "ch":
                self.do_char(t)
            elif k == "active":
                self.emit_text(" ")
            elif k == "bg":
                saved = self.style
                self.run(group=True, open_tok=t)
                self.style = saved
            elif k == "eg":
                if group:
                    return "}"
                self.error("Too many }'s." if not self.mode_stack else "Extra }, or forgotten \\endgroup.", t,
                           "There's a `}` without a matching `{`.")
            elif k == "math":
                self.do_math_shift(t)
            elif k == "tab":
                self.error("Misplaced alignment tab character &.", t,
                           "`&` separates columns in a tabular or an alignment. For an ampersand, write \\&.")
                self.emit_text("&")
            elif k in ("sup", "sub"):
                self.error("Missing $ inserted.", t, f"`{t.v}` only works in math: write $x{t.v}2$, or \\{t.v} for "
                                                     f"the character itself.")
                self.emit_text(t.v)
            elif k == "param":
                self.error("You can't use `macro parameter character #' in horizontal mode.", t,
                           "For a # sign, write \\#.")
                self.emit_text("#")
            elif k == "verb":
                if t.extra == "url":
                    self.emit_text(t.v)
                else:
                    self.add_inline({"t": "verb", "s": t.v, "st": self.style})
                    self.used["\\verb"] += 1
            elif k == "verb_err":
                self.error("LaTeX Error: \\verb ended by end of line.", t, "\\verb|...| must end on the same line.")
            elif k == "cs":
                r = self.do_cs(t, stop_env)
                if r == "end":
                    return "end"
            if self.ended:
                return None

    # ------------------------------------------------------------ characters

    def do_char(self, t: Tok):
        c = t.v
        tt = self.style.family == "tt"
        t1 = "T1" in self.pkgs.get("fontenc", [])

        def nxt(k=0):
            j = self.pos + k
            return self.toks[j].v if j < len(self.toks) and self.toks[j].k == "ch" else ""
        if not tt:
            if c == "-":
                if nxt() == "-":
                    if nxt(1) == "-":
                        self.pos += 2
                        self.emit_text("\u2014")
                    else:
                        self.pos += 1
                        self.emit_text("\u2013")
                    return
            elif c == "`":
                if nxt() == "`":
                    self.pos += 1
                    self.emit_text("\u201c")
                else:
                    self.emit_text("\u2018")
                return
            elif c == "'":
                if nxt() == "'":
                    self.pos += 1
                    self.emit_text("\u201d")
                else:
                    self.emit_text("\u2019")
                return
            elif c in "?!" and nxt() == "`":
                self.pos += 1
                self.emit_text("\u00bf" if c == "?" else "\u00a1")
                return
            elif c == '"' and not t1:
                self.emit_text("\u201d")
                return
            elif c in "<>|" and not t1:
                # the default (OT1) fonts have no < > | : those slots hold ¡ ¿ and an em dash
                self.emit_text({"<": "\u00a1", ">": "\u00bf", "|": "\u2014"}[c])
                return
        self.emit_text(c)

    # ------------------------------------------------------------ control sequences

    def available(self, name: str) -> bool:
        if name in self.macros or name in self._text_cmds:
            return True
        return False

    def refresh_available(self):
        pk = set(self.pkgs) | set(IMPLICIT.get(self.cls or "", []))
        self.pkg_closure = P.closure(pk)
        self._text_cmds = P.text_commands(self.cls or "article", pk)
        self._envs = P.environments(self.cls or "article", pk)

    def do_cs(self, t: Tok, stop_env=None):
        name = t.v
        if name in self.macros:
            return self.expand_macro(t)
        if name == "end":
            envname = self.arg_text("end")
            return self.do_end(envname, t, stop_env)
        if name == "begin":
            envname = self.arg_text("begin")
            self.used["env:" + envname] += 1
            return self.do_begin(envname, t)
        if len(name) == 1 and name in SYMBOLS and name not in ACCENTS:
            self.emit_text(SYMBOLS[name])
            self.used["\\" + name] += 1
            return None
        if name.startswith("the") and self.counters is not None and name[3:] in self.counters.v and \
                name not in HANDLERS:
            # \thepage, \thesection ...: how the counter prints
            self.used["\\" + name] += 1
            self.emit_text(self.counters.the(name[3:]))
            return None
        handler = HANDLERS.get(name)
        known = handler is not None or name in SYMBOLS or name in ACCENTS or name in SPACES or \
            name in FONT_DECLS or name in FONT_CMDS
        if not known and len(name) > 1 and name not in ("begin", "end") and \
                mathmode.math_available(name, self.pkg_closure, set()) is None:
            # a math-only command in text: TeX inserts the $ for you (and complains)
            self.error("Missing $ inserted.", t, f"\\{name} only works in math: write ${'\\'}{name}$.")
            self.push([Tok("math", "$", t.file, t.line, t.col), t, Tok("math", "$", t.file, t.line, t.col)])
            return None
        if not known or (not self.available(name) and name not in ACCENTS):
            self.error("Undefined control sequence.", t, self._undefined_help(name))
            return None
        if self.cls is None and name != "documentclass":
            if name in ("usepackage", "RequirePackage"):
                self.error("LaTeX Error: \\usepackage before \\documentclass.", t,
                           "The first line must be \\documentclass{...}.")
            else:
                self.error("LaTeX Error: Missing \\begin{document}.", t,
                           "The file must start with \\documentclass{...}.")
            return None
        if not self.in_document and name not in PREAMBLE_OK:
            self.error("LaTeX Error: Missing \\begin{document}.", t,
                       f"\\{name} is body content: it goes after \\begin{{document}}.")
            return None
        self.used["\\" + name] += 1
        if handler is not None:
            return handler(self, t)
        if name in SYMBOLS:
            self.emit_text(SYMBOLS[name])
        elif name in ACCENTS:
            arg = self.read_arg(name)
            s = detokenize(arg).strip()
            if s:
                self.emit_text(s[0] + ACCENTS[name] + s[1:])
        elif name in SPACES:
            self.add_space(SPACES[name])
        elif name in FONT_DECLS:
            self.style = self.style.with_(**FONT_DECLS[name])
        elif name in FONT_CMDS:
            arg = self.read_arg(name, allow_par=True)      # \textbf and friends are \long: a blank line is fine
            saved = self.style
            self.style = self.style.with_(**FONT_CMDS[name])
            self.run_tokens(arg)
            self.style = saved
        return None

    def _undefined_help(self, name: str) -> str:
        prov = P.provider(name)
        if prov and prov not in self.pkg_closure:
            return f"\\{name} comes from the {prov} package: add \\usepackage{{{prov}}} to the preamble."
        if name == "chapter":
            return "\\chapter exists only in the report and book classes."
        close = [c for c in list(self._text_cmds) + list(self.macros) if _close(c, name)]
        if close:
            return f"There's no command \\{name}. Did you mean \\{sorted(close, key=len)[0]}?"
        return f"There's no command \\{name} (check the spelling, or define it with \\newcommand)."

    def add_space(self, kind: str):
        if kind == "space":
            self.add_inline(text(" ", self.style))
        else:
            self.add_inline({"t": "space", "w": kind, "st": self.style})

    # ------------------------------------------------------------ macros

    def expand_macro(self, t: Tok):
        m = self.macros[t.v]
        self.used["\\" + t.v] += 1
        self.expansions += 1
        if self.expansions > 20000:
            self.error("TeX capacity exceeded, sorry [main memory size=5000000].", t,
                       "A macro calls itself forever.")
            raise Stop()
        args = []
        n = m["n"]
        if n:
            if m.get("default") is not None:
                o = self.read_opt()
                args.append(o if o is not None else m["default"])
                n -= 1
            for _ in range(n):
                args.append(self.read_arg(t.v))
        body = []
        toks = m["body"]
        i = 0
        while i < len(toks):
            b = toks[i]
            if b.k == "param" and i + 1 < len(toks) and toks[i + 1].k == "ch" and toks[i + 1].v.isdigit():
                k = int(toks[i + 1].v) - 1
                if 0 <= k < len(args):
                    body += args[k]
                i += 2
                continue
            body.append(Tok(b.k, b.v, t.file, t.line, t.col, b.extra))
            i += 1
        self.push(body)
        return None

    def define(self, name: str, n: int, default, body: list[Tok], t: Tok, how: str):
        exists = name in self.macros or name in self._text_cmds or name in HANDLERS or name in mathmode.GREEK or \
            name in mathmode.CORE_SYMBOLS or name in mathmode.CORE_ONE_ARG or name in mathmode.AMSMATH
        if how == "new" and exists:
            self.error(f"LaTeX Error: Command \\{name} already defined.", t,
                       f"\\{name} exists already: use \\renewcommand to redefine it, or pick another name.")
            return
        if how == "renew" and not exists:
            self.error(f"LaTeX Error: \\{name} undefined.", t,
                       f"\\renewcommand only redefines existing commands: use \\newcommand for \\{name}.")
            return
        if how == "provide" and exists:
            return
        self.macros[name] = {"n": n, "default": default, "body": body}

    def math_macros(self) -> set:
        return set(self.macros)

    def expand_math(self, toks: list[Tok], depth=0) -> list[Tok]:
        """User macros inside math, expanded (they may use other macros)."""
        if depth > 30:
            return toks
        out = []
        i = 0
        changed = False
        while i < len(toks):
            t = toks[i]
            if t.k == "cs" and t.v in self.macros:
                m = self.macros[t.v]
                self.used["\\" + t.v] += 1
                i += 1
                args = []
                n = m["n"]
                if n:
                    if m.get("default") is not None:
                        j = i
                        while j < len(toks) and toks[j].k == "sp":
                            j += 1
                        if j < len(toks) and toks[j].k == "ch" and toks[j].v == "[":
                            k = j + 1
                            while k < len(toks) and not (toks[k].k == "ch" and toks[k].v == "]"):
                                k += 1
                            args.append(toks[j + 1:k])
                            i = k + 1
                        else:
                            args.append(m["default"])
                        n -= 1
                    for _ in range(n):
                        while i < len(toks) and toks[i].k == "sp":
                            i += 1
                        if i < len(toks) and toks[i].k == "bg":
                            depth_b = 0
                            j = i
                            while j < len(toks):
                                if toks[j].k == "bg":
                                    depth_b += 1
                                elif toks[j].k == "eg":
                                    depth_b -= 1
                                    if depth_b == 0:
                                        break
                                j += 1
                            args.append(toks[i + 1:j])
                            i = j + 1
                        elif i < len(toks):
                            args.append([toks[i]])
                            i += 1
                        else:
                            args.append([])
                body = []
                bt = m["body"]
                k = 0
                while k < len(bt):
                    if bt[k].k == "param" and k + 1 < len(bt) and bt[k + 1].k == "ch" and bt[k + 1].v.isdigit():
                        a = int(bt[k + 1].v) - 1
                        if 0 <= a < len(args):
                            body += args[a]
                        k += 2
                        continue
                    body.append(Tok(bt[k].k, bt[k].v, t.file, t.line, t.col))
                    k += 1
                # wrap in a group so x^\vect{x} keeps working, as TeX's would
                out += body
                changed = True
                continue
            out.append(t)
            i += 1
        return self.expand_math(out, depth + 1) if changed else out

    # ------------------------------------------------------------ environments

    def env_available(self, name: str) -> bool:
        return name in self._envs or name in self.envs or name in self.theorems

    def do_begin(self, name: str, t: Tok):
        if not self.in_document:
            if name == "document":
                return begin_document(self, t)
            if self.cls is None:
                self.error("LaTeX Error: Missing \\begin{document}.", t, "")
                return None
            self.error("LaTeX Error: Missing \\begin{document}.", t,
                       f"\\begin{{{name}}} is body content: it goes after \\begin{{document}}.")
            return None
        if name == "document":
            self.error("LaTeX Error: Can be used only in preamble.", t, "\\begin{document} appears twice.")
            return None
        if not self.env_available(name):
            prov = P.provider(name, "env")
            self.error(f"LaTeX Error: Environment {name} undefined.", t,
                       f"{name} comes from the {prov} package: add \\usepackage{{{prov}}} to the preamble."
                       if prov else f"No environment called {name}: check the spelling, or define it with "
                                    f"\\newenvironment.")
            # TeX carries on: the contents are typeset as a group
            self.env_line_stack.append((name, t))
            saved = self.style
            r = self.run(stop_env=name)
            self.style = saved
            return None
        self.env_line_stack.append((name, t))
        try:
            if name in self.envs:
                return self.user_env(name, t)
            if name in self.theorems:
                from .features import theorem_env
                return theorem_env(self, name, t)
            handler = ENV_HANDLERS.get(name)
            if handler is None:
                # a known environment without special handling: its contents as a group
                saved = self.style
                self.run(stop_env=name)
                self.style = saved
                return None
            return handler(self, name, t)
        finally:
            pass

    def do_end(self, name: str, t: Tok, stop_env):
        if name == "document":
            if stop_env not in (None, "document"):
                self.error(f"LaTeX Error: \\begin{{{stop_env}}} on input line {self._env_line(stop_env)} ended by "
                           f"\\end{{document}}.", t, f"\\begin{{{stop_env}}} is never closed.")
            self.end_par()
            self.ended = True
            raise Stop()
        if stop_env == name:
            if self.env_line_stack and self.env_line_stack[-1][0] == name:
                self.env_line_stack.pop()
            return "end"
        if stop_env is None:
            self.error(f"LaTeX Error: \\begin{{document}} ended by \\end{{{name}}}." if self.in_document else
                       f"LaTeX Error: Missing \\begin{{document}}.", t,
                       f"\\end{{{name}}} without a matching \\begin{{{name}}}.")
            return None
        self.error(f"LaTeX Error: \\begin{{{stop_env}}} on input line {self._env_line(stop_env)} ended by "
                   f"\\end{{{name}}}.", t, f"\\begin{{{stop_env}}} must be closed before \\end{{{name}}}.")
        if self.env_line_stack and self.env_line_stack[-1][0] == stop_env:
            self.env_line_stack.pop()
        return "end"

    def _env_line(self, name):
        for n, tok in reversed(self.env_line_stack):
            if n == name:
                return tok.line
        return "?"

    def env_body(self, name: str, t: Tok) -> list[Tok]:
        """The raw tokens up to the matching \\end{name}. Environments opened inside must close inside; an \\end
        that closes something else ends this one with LaTeX's "\\begin{x} ... ended by \\end{y}" error."""
        inner: list[str] = []
        start = self.pos
        while self.pos < len(self.toks):
            u = self.toks[self.pos]
            if u.k == "cs" and u.v in ("begin", "end"):
                j = self.pos + 1
                while j < len(self.toks) and self.toks[j].k == "sp":
                    j += 1
                if j < len(self.toks) and self.toks[j].k == "bg":
                    k = j + 1
                    nm = []
                    while k < len(self.toks) and self.toks[k].k != "eg":
                        nm.append(self.toks[k].text())
                        k += 1
                    env_name = "".join(nm)
                    if u.v == "begin":
                        inner.append(env_name)
                    elif inner and inner[-1] == env_name:
                        inner.pop()
                    elif not inner:
                        body = self.toks[start:self.pos]
                        if env_name != name:
                            if env_name == "document":
                                self.error(f"LaTeX Error: \\begin{{{name}}} on input line {t.line} ended by "
                                           f"\\end{{document}}.", u, f"\\begin{{{name}}} is never closed: add "
                                                                     f"\\end{{{name}}}.")
                                if self.env_line_stack and self.env_line_stack[-1][0] == name:
                                    self.env_line_stack.pop()
                                return body          # leave \\end{document} to be found
                            self.error(f"LaTeX Error: \\begin{{{name}}} on input line {t.line} ended by "
                                       f"\\end{{{env_name}}}.", u,
                                       f"\\begin{{{name}}} must be closed with \\end{{{name}}}.")
                        self.pos = k + 1
                        if self.env_line_stack and self.env_line_stack[-1][0] == name:
                            self.env_line_stack.pop()
                        return body
            self.pos += 1
        self.error(f"LaTeX Error: \\begin{{{name}}} on input line {t.line} ended by \\end{{document}}.", t,
                   f"\\begin{{{name}}} is never closed: add \\end{{{name}}}.")
        return self.toks[start:]

    def user_env(self, name, t):
        e = self.envs[name]
        args = []
        n = e["n"]
        if n:
            if e.get("default") is not None:
                o = self.read_opt()
                args.append(o if o is not None else e["default"])
                n -= 1
            for _ in range(n):
                args.append(self.read_arg(name))

        def subst(toks):
            out = []
            i = 0
            while i < len(toks):
                b = toks[i]
                if b.k == "param" and i + 1 < len(toks) and toks[i + 1].k == "ch" and toks[i + 1].v.isdigit():
                    k = int(toks[i + 1].v) - 1
                    if 0 <= k < len(args):
                        out += args[k]
                    i += 2
                    continue
                out.append(Tok(b.k, b.v, t.file, t.line, t.col, b.extra))
                i += 1
            return out
        body = self.env_body(name, t)
        saved = self.style
        self.run_tokens(subst(e["begin"]) + body + [Tok(x.k, x.v, t.file, t.line, t.col) for x in e["end"]])
        self.style = saved
        return None

    # ------------------------------------------------------------ math

    def do_math_shift(self, t: Tok):
        nxt = self.peek()
        if nxt is not None and nxt.k == "math":
            self.next()
            toks = self.collect_math(t, "$$")
            self.display_math("displaymath", toks, t)
            return
        toks = self.collect_math(t, "$")
        self.inline_math(toks, t)

    def collect_math(self, t: Tok, close: str) -> list[Tok]:
        out = []
        depth = 0
        while True:
            u = self.next()
            if u is None:
                self.error("Missing $ inserted.", t, "Math started here is never closed.")
                return out
            if u.k == "par":
                self.error("Missing $ inserted.", u, "Math can't contain a blank line: close it with $ first.")
                self.pos -= 1
                return out
            if close in ("$", "$$") and u.k == "math" and depth == 0:
                if close == "$$":
                    nxt = self.peek()
                    if nxt is not None and nxt.k == "math":
                        self.next()
                    else:
                        self.error("Display math should end with $$.", u, "")
                return out
            if close == "\\)" and u.k == "cs" and u.v == ")":
                return out
            if close == "\\]" and u.k == "cs" and u.v == "]":
                return out
            if u.k == "math" and close not in ("$", "$$"):
                self.error("Bad math environment delimiter.", u, "A $ inside math that started with \\( or \\[.")
                continue
            if u.k == "bg":
                depth += 1
            elif u.k == "eg":
                depth -= 1
            out.append(u)

    def math_check(self, toks: list[Tok], env=None):
        for msg, tok, hlp in mathmode.check(toks, self.pkg_closure, self.math_macros(), env=env):
            self.error(msg, tok or self.last_tok, hlp)

    def inline_math(self, toks: list[Tok], t: Tok):
        toks = self.expand_math(toks)
        self.math_check(toks)
        self.used["$"] += 1
        self._math_used(toks)
        self.add_inline({"t": "math", "tex": mathmode.to_katex(toks), "canon": mathmode.canonical(toks),
                         "st": self.style})

    def _math_used(self, toks):
        for i, u in enumerate(toks):
            if u.k == "cs":
                self.used["\\" + u.v] += 1
                if u.v == "begin":
                    name, _ = mathmode._read_name(toks, i + 1)
                    if name:
                        self.used["env:" + name] += 1
            elif u.k in ("sup", "sub"):
                self.used[u.v] += 1
            elif u.k == "tab" and i + 1 < len(toks) and toks[i + 1].k == "tab":
                self.used["&&"] += 1     # an alignment's note column: && \text{...}

    def display_math(self, env: str, toks: list[Tok], t: Tok):
        from .features import display_block
        display_block(self, env, toks, t)

    # ------------------------------------------------------------ labels

    def set_label_target(self, kind: str, txt: str, env=None):
        self.cur_label = {"text": txt, "kind": kind, "env": env}

    def add_label(self, key: str, t: Tok):
        if key in self.labels:
            self.warn(f"LaTeX Warning: Label `{key}' multiply defined.", t)
        self.labels[key] = dict(self.cur_label, page="1")
        self.used["\\label"] += 1


def _close(a, b):
    if abs(len(a) - len(b)) > 2 or a == b:
        return False
    # one edit, or a swap
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


# commands allowed before \begin{document}
PREAMBLE_OK = set("""documentclass usepackage RequirePackage title author date and thanks newcommand renewcommand
providecommand newenvironment renewenvironment def let newtheorem theoremstyle DeclareMathOperator setlength
addtolength definecolor colorlet pagestyle fancyhf lhead chead rhead lfoot cfoot rfoot fancyhead fancyfoot
geometry hypersetup graphicspath usetikzlibrary pgfplotsset tikzset lstset lstdefinestyle bibliographystyle
addbibresource usetheme usecolortheme usefonttheme setbeamertemplate setbeamercolor institute subtitle logo
newcounter setcounter numberwithin onehalfspacing doublespacing singlespacing setstretch setlist renewcommand
captionsetup sisetup crefname Crefname urlstyle linespread newcolumntype tcbuselibrary newtcolorbox tcbset
makeatletter makeatother titleformat AtBeginSection algrenewcommand renewenvironment thispagestyle input
counterwithin newgeometry floatstyle restylefloat DeclarePairedDelimiter mathtoolsset baselinestretch
pagenumbering includeonly relax""".split())

HANDLERS: dict = {}
ENV_HANDLERS: dict = {}


def cmd(*names):
    def reg(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return reg


def env(*names):
    def reg(fn):
        for n in names:
            ENV_HANDLERS[n] = fn
        return fn
    return reg


def begin_document(e: Engine, t: Tok):
    if e.cls is None:
        e.error("LaTeX Error: The font size command \\normalsize is not defined: there is probably something wrong "
                "with the class file.", t, "The document has no \\documentclass{...} line.")
        e.cls = "article"
        e.counters = Counters("article")
        e.refresh_available()
    e.in_document = True
    e.used["env:document"] += 1
    e.push_frame(kind="body", align="j")
    if e.cls == "beamer":
        from .beamer import Beamer
        e.beamer = Beamer(e)
    return None

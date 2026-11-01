"""The document latex-lite builds: blocks of styled inline runs, plus numbering.

Blocks and inlines are plain dicts (`t` is the type), so the grader can compare them and the renderer walk them.

Inline: {"t": "text", "s": str, "st": Style} | {"t": "math", "tex", "canon", "st"} | {"t": "ref", "label", "how",
  "st"} | {"t": "cite", "keys", "how", "st"} | {"t": "fn", "n"} | {"t": "link", "url", "c": [inline]} |
  {"t": "verb", "s", "st"} | {"t": "space", "w"} | {"t": "br"} | {"t": "box", "b": block}
Block: par, heading, title, abstract, toc, list, display, float, caption, tabular, verbatim, listing, algorithmic,
  theorem, proof, box, columns, vspace, newpage, tikz, plot, bibliography, frame (beamer) ... (see render.py)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

SIZES_10 = {"tiny": 5, "scriptsize": 7, "footnotesize": 8, "small": 9, "normalsize": 10, "large": 12, "Large": 14.4,
            "LARGE": 17.28, "huge": 20.74, "Huge": 24.88}
SIZES_11 = {"tiny": 6, "scriptsize": 8, "footnotesize": 9, "small": 10, "normalsize": 10.95, "large": 12,
            "Large": 14.4, "LARGE": 17.28, "huge": 20.74, "Huge": 24.88}
SIZES_12 = {"tiny": 6, "scriptsize": 8, "footnotesize": 10, "small": 10.95, "normalsize": 12, "large": 14.4,
            "Large": 17.28, "LARGE": 20.74, "huge": 24.88, "Huge": 24.88}
SIZE_ORDER = ["tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge"]


@dataclass(frozen=True)
class Style:
    bold: bool = False
    shape: str = "n"          # n | it | sl | sc
    family: str = "rm"        # rm | sf | tt
    size: str = "normalsize"
    color: str | None = None
    under: bool = False

    def key(self):
        """What the grader compares (emphasis and italics look the same)."""
        return (self.bold, self.shape, self.family, self.size, self.color, self.under)

    def with_(self, **kw):
        return replace(self, **kw)


PLAIN = Style()


def text(s, st=PLAIN):
    return {"t": "text", "s": s, "st": st}


# ---------------------------------------------------------------- numbering

def alph(n):
    return "abcdefghijklmnopqrstuvwxyz"[(n - 1) % 26] if n > 0 else ""


def roman(n):
    if n <= 0:
        return ""
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
            (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def fnsymbol(n):
    return ["", "*", "†", "‡", "§", "¶", "‖", "**", "††", "‡‡"][n] if 0 <= n < 10 else str(n)


FORMATS = {"arabic": str, "alph": alph, "Alph": lambda n: alph(n).upper(), "roman": roman,
           "Roman": lambda n: roman(n).upper(), "fnsymbol": fnsymbol}


class Counters:
    """LaTeX counters with their parents (a parent's step resets its children) and how \\the<counter> prints."""

    def __init__(self, cls: str):
        self.v: dict[str, int] = {}
        self.parent: dict[str, str | None] = {}
        self.fmt: dict[str, str] = {}
        self.thefmt: dict[str, object] = {}
        report = cls in ("report", "book", "memoir", "scrreprt", "scrbook")
        top = "chapter" if report else None
        for name, parent in [("part", None), ("chapter", None), ("section", top), ("subsection", "section"),
                             ("subsubsection", "subsection"), ("paragraph", "subsubsection"),
                             ("subparagraph", "paragraph"), ("equation", top), ("figure", top), ("table", top),
                             ("footnote", top), ("enumi", None), ("enumii", "enumi"), ("enumiii", "enumii"),
                             ("enumiv", "enumiii"), ("algorithm", top), ("lstlisting", top), ("page", None),
                             ("subfigure", "figure"), ("subtable", "table"), ("mpfootnote", None)]:
            self.new(name, parent)
        self.v["page"] = 1
        self.fmt["part"] = "Roman"
        self.fmt["enumii"] = "alph"
        self.fmt["enumiii"] = "roman"
        self.fmt["enumiv"] = "Alph"
        self.fmt["subfigure"] = "alph"
        self.fmt["subtable"] = "alph"
        self.fmt["mpfootnote"] = "alph"
        self.report = report

    def new(self, name, parent=None, fmt="arabic"):
        self.v[name] = 0
        self.parent[name] = parent
        self.fmt[name] = fmt

    def step(self, name):
        self.v[name] = self.v.get(name, 0) + 1
        for c, p in self.parent.items():
            if p == name:
                self._reset(c)

    def _reset(self, name):
        self.v[name] = 0
        for c, p in self.parent.items():
            if p == name:
                self._reset(c)

    def value(self, name):
        return self.v.get(name, 0)

    def number(self, name) -> str:
        return FORMATS.get(self.fmt.get(name, "arabic"), str)(self.v.get(name, 0))

    def the(self, name) -> str:
        """\\the<name>: the number with its parents' (1.2 for a subsection)."""
        custom = self.thefmt.get(name)
        if custom is not None:
            return custom(self)
        own = self.number(name)
        p = self.parent.get(name)
        if name in ("enumi", "enumii", "enumiii", "enumiv", "footnote", "page", "part", "mpfootnote"):
            return own
        if name in ("subfigure", "subtable"):
            return own
        if p and p not in ("enumi", "enumii", "enumiii"):
            pv = self.the(p)
            if p == "chapter" and self.v.get("chapter", 0) == 0:
                return own
            return f"{pv}.{own}"
        return own

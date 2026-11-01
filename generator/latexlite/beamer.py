"""Beamer: frames become slides, one per overlay step (\\pause, \\item<2->, <+->, \\only, \\uncover, \\onslide,
\\alert), with columns, blocks, the title page and a table of contents of the sections. The look follows the theme
(default, or Madrid's title bar and footline)."""

from __future__ import annotations

import html
import re

from .engine import cmd, env
from .tokens import Tok, detokenize

THEMES = {"Madrid": "madrid", "Berlin": "madrid", "Warsaw": "madrid", "Copenhagen": "madrid", "Boadilla": "madrid",
          "AnnArbor": "madrid", "CambridgeUS": "madrid", "Singapore": "default", "default": "default",
          "Frankfurt": "madrid", "Darmstadt": "madrid", "Antibes": "madrid", "Montpellier": "default",
          "metropolis": "metropolis", "Pittsburgh": "default", "Rochester": "madrid", "Malmoe": "madrid",
          "Szeged": "madrid", "Ilmenau": "madrid", "Dresden": "madrid", "Hannover": "madrid", "Marburg": "madrid",
          "PaloAlto": "madrid", "Goettingen": "madrid", "Luebeck": "madrid", "JuanLesPins": "madrid",
          "Bergen": "default", "Berkeley": "madrid", "Boadilla ": "madrid"}


def parse_spec(spec: str):
    """'2-' / '1-3' / '2' / '-3' / '1,3-' -> a predicate on the slide number, and the largest number named."""
    spec = (spec or "").split("|")[0].strip()
    ranges = []
    top = 1
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", part)
        if m:
            a = int(m.group(1)) if m.group(1) else 1
            b = int(m.group(2)) if m.group(2) else None
            ranges.append((a, b))
            top = max(top, a, b or a)
        elif part.isdigit():
            ranges.append((int(part), int(part)))
            top = max(top, int(part))
    if not ranges:
        return (lambda s: True), 1
    return (lambda s: any(a <= s and (b is None or s <= b) for a, b in ranges)), top


class Beamer:
    def __init__(self, e):
        self.e = e
        self.frames = []
        self.sections = []
        self.cur_section = None
        self.pauses = 1
        self.in_frame = False
        self.max_step = 1

    # -------------------------------------------------- called by the engine

    def section(self, name, title_toks, t):
        if name in ("section", "subsection"):
            title = self.e.capture(title_toks)
            if name == "section":
                self.sections.append({"title": title, "subs": []})
            elif self.sections:
                self.sections[-1]["subs"].append(title)
            self.cur_section = title
            self.e.used["\\" + name] += 1

    def plus(self, spec):
        """<+-> : the current pause number, then one more."""
        if "+" in spec:
            n = self.pauses
            self.pauses += 1
            spec = spec.replace("+", str(n))
        self.note(spec)
        return spec

    def note(self, spec):
        _, top = parse_spec(spec)
        self.max_step = max(self.max_step, top)

    def note_overlays(self, items):
        for it in items:
            if it.get("overlay"):
                self.note(it["overlay"])

    def titlepage(self, t):
        e = self.e
        if e.title.get("title") is None:
            e.error("LaTeX Error: No \\title given.", t, "Add \\title{...} to the preamble.")
            return
        get = lambda k: e.capture(e.title[k]) if e.title.get(k) else None  # noqa: E731
        authors = [[]]
        for node in e.capture(e.title.get("author") or []):
            if node["t"] == "and":
                authors.append([])
            else:
                authors[-1].append(node)
        if e.title.get("date") is not None:
            date = e.capture(e.title["date"])
        else:
            import datetime
            d = datetime.date.today()
            date = [{"t": "text", "s": f"{d.strftime('%B')} {d.day}, {d.year}", "st": e.style}]
        e.add_block({"t": "btitle", "title": get("title"), "subtitle": get("subtitle"),
                     "authors": [a for a in authors if a], "institute": get("institute"), "date": date})

    def toc(self, t):
        self.e.add_block({"t": "btoc"})


def _frame_opts_and_titles(e, body_toks):
    """Leading {title}{subtitle} groups of a frame body."""
    titles = []
    i = 0
    while len(titles) < 2:
        j = i
        while j < len(body_toks) and body_toks[j].k == "sp":
            j += 1
        if j < len(body_toks) and body_toks[j].k == "bg":
            depth = 0
            k = j
            while k < len(body_toks):
                if body_toks[k].k == "bg":
                    depth += 1
                elif body_toks[k].k == "eg":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            titles.append(body_toks[j + 1:k])
            i = k + 1
        else:
            break
    return titles, body_toks[i:]


def do_frame(e, body, opts, t):
    b = e.beamer
    if b is None:
        e.error("LaTeX Error: Environment frame undefined.", t, "Frames exist in the beamer class.")
        return
    if b.in_frame:
        e.error("LaTeX Error: Not in outer par mode.", t, "A frame can't contain another frame.")
        return
    fragile = "fragile" in opts
    if not fragile and any(x.k == "verb" and isinstance(x.extra, (tuple, str)) and x.extra != "url" for x in body):
        e.error(f"LaTeX Error: \\begin{{verbatim}} on input line {t.line} ended by \\end{{beamer@framepauses}}.", t,
                "Verbatim text (verbatim, lstlisting, \\verb) needs a fragile frame: \\begin{frame}[fragile].")
    titles, rest = _frame_opts_and_titles(e, body)
    b.in_frame = True
    b.pauses = 1
    b.max_step = 1
    b.frame_title = e.capture(titles[0]) if titles else None
    b.frame_subtitle = e.capture(titles[1]) if len(titles) > 1 else None
    saved = e.style
    blocks = e.capture_blocks(rest, kind="frame", align="l")
    e.style = saved
    steps = max(b.max_step, b.pauses)
    b.frames.append({"title": b.frame_title, "subtitle": b.frame_subtitle, "blocks": blocks, "steps": steps,
                     "plain": "plain" in opts, "fragile": fragile, "section": b.cur_section,
                     "top": "t" in opts})
    b.in_frame = False


@env("frame")
def frame_env(e, name, t):
    o = e.read_overlay()
    opts = e.kv(e.read_opt())
    body = e.env_body(name, t)
    e.used["env:frame"] += 1
    do_frame(e, body, opts, t)


@cmd("frame")
def frame_cmd(e, t):
    opts = e.kv(e.read_opt())
    body = e.read_arg("frame")
    do_frame(e, body, opts, t)


@cmd("frametitle", "framesubtitle")
def frametitle(e, t):
    arg = e.read_arg(t.v)
    if e.beamer is None or not e.beamer.in_frame:
        e.error("LaTeX Error: \\frametitle outside a frame." if e.beamer is not None else "Undefined control sequence.",
                t, "\\frametitle goes inside a frame.")
        return
    if t.v == "frametitle":
        e.beamer.frame_title = e.capture(arg)
    else:
        e.beamer.frame_subtitle = e.capture(arg)


@cmd("pause")
def pause(e, t):
    b = e.beamer
    if b is None or not b.in_frame:
        return
    b.pauses += 1
    b.max_step = max(b.max_step, b.pauses)
    f = e.frame()
    node = {"t": "pause", "from": b.pauses}
    if f["par"] is not None and f["par"]["c"]:
        e.add_inline(node)
    else:
        e.add_block(node)


@cmd("onslide")
def onslide(e, t):
    b = e.beamer
    star = e.read_star()
    spec = e.read_overlay() or ""
    if b is not None:
        spec = b.plus(spec)
    nt = e.peek()
    if nt is not None and nt.k == "bg":
        arg = e.read_arg("onslide")
        e.add_inline({"t": "uncover", "spec": spec, "c": e.capture(arg, e.style)})
        return
    f = e.frame()
    node = {"t": "onslide", "spec": spec}
    if f["par"] is not None and f["par"]["c"]:
        e.add_inline(node)
    else:
        e.add_block(node)


@cmd("only", "uncover", "visible", "invisible", "alert", "alt", "temporal", "structure")
def overlay_cmds(e, t):
    b = e.beamer
    spec = e.read_overlay()
    if spec is not None and b is not None:
        spec = b.plus(spec)
    arg = e.read_arg(t.v)
    if t.v == "alt":
        other = e.read_arg(t.v)
        e.add_inline({"t": "only", "spec": spec or "", "c": e.capture(arg, e.style)})
        e.add_inline({"t": "only", "spec": "!" + (spec or ""), "c": e.capture(other, e.style)})
        return
    if t.v in ("alert", "structure"):
        color = "#ff0000" if t.v == "alert" else "#333399"
        if spec is None:
            saved = e.style
            e.style = e.style.with_(color=color)
            e.run_tokens(arg)
            e.style = saved
        else:
            e.add_inline({"t": "alertspec", "spec": spec, "c": e.capture(arg, e.style.with_(color=color)),
                          "plain": e.capture(arg, e.style)})
        return
    if t.v == "temporal":
        e.read_arg(t.v)
        return
    e.add_inline({"t": t.v if t.v != "visible" else "uncover", "spec": spec or "", "c": e.capture(arg, e.style)})


@env("columns")
def columns_env(e, name, t):
    e.read_opt()
    body = e.env_body(name, t)
    cols = []
    cur = None
    i = 0
    # \column{w} separators, or column environments
    while i < len(body):
        x = body[i]
        if x.k == "cs" and x.v == "column":
            e.used["\\column"] += 1
            j = i + 1
            while j < len(body) and body[j].k == "sp":
                j += 1
            depth = 0
            k = j
            while k < len(body):
                if body[k].k == "bg":
                    depth += 1
                elif body[k].k == "eg":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            cur = {"w": detokenize(body[j + 1:k]), "toks": []}
            cols.append(cur)
            i = k + 1
            continue
        if x.k == "cs" and x.v == "begin":
            # \begin{column}{w} ... \end{column}
            j = i + 1
            nm = []
            while j < len(body) and body[j].k != "eg":
                nm.append(body[j].text())
                j += 1
            if "".join(nm).strip("{ ") == "column":
                k = j + 1
                while k < len(body) and body[k].k == "sp":
                    k += 1
                depth = 0
                m = k
                while m < len(body):
                    if body[m].k == "bg":
                        depth += 1
                    elif body[m].k == "eg":
                        depth -= 1
                        if depth == 0:
                            break
                    m += 1
                cur = {"w": detokenize(body[k + 1:m]), "toks": []}
                cols.append(cur)
                i = m + 1
                continue
        if x.k == "cs" and x.v == "end":
            j = i + 1
            nm = []
            while j < len(body) and body[j].k != "eg":
                nm.append(body[j].text())
                j += 1
            if "".join(nm).strip("{ ") == "column":
                i = j + 1
                cur = None
                continue
        if cur is not None:
            cur["toks"].append(x)
        elif x.k not in ("sp", "par"):
            e.error("Package beamer Error: Text outside a column.", x, "Inside columns, start each with \\column{0.5\\textwidth}.")
        i += 1
    out = []
    for c in cols:
        out.append({"w": c["w"], "c": e.capture_blocks(c["toks"], kind="column", align="l")})
    e.add_block({"t": "columns", "cols": out})


@cmd("column")
def column_outside(e, t):
    e.arg_text("column")
    e.error("Package beamer Error: \\column outside columns.", t, "\\column goes inside \\begin{columns}.")


@env("block", "alertblock", "exampleblock")
def block_env(e, name, t):
    spec = e.read_overlay()
    title = e.read_arg(name)
    body = e.env_body(name, t)
    e.add_block({"t": "bblock", "kind": name, "title": e.capture(title), "c": e.capture_blocks(body, kind="bblock",
                                                                                                align="l"),
                 "spec": spec})


# ---------------------------------------------------------------- slides

SLIDE_W, SLIDE_H = 364.2, 273.2


def render_slides(r):
    e = r.e
    theme = THEMES.get(e.page.get("beamer_usetheme", "default"), "default")
    frames = r.doc.get("frames", [])
    total = len(frames)
    out = [f'<div class="tex-doc tex-beamer tex-theme-{theme}" style="--tex-fs:11pt;--tex-lh:1.25;">']
    title = "".join(x.text() for x in e.title.get("title") or [])
    short_title = "".join(x.text() for x in (e.title.get("title_short") or e.title.get("title") or []))
    author = "".join(x.text() for x in (e.title.get("author_short") or e.title.get("author") or [])).replace("\\and", ",")
    if e.title.get("date") is not None:      # typeset it (\today, \date{}), as the title page does
        date = "".join(x.get("s", "") for x in e.capture(e.title["date"]) if x.get("t") == "text")
    else:
        import datetime
        d = datetime.date.today()
        date = f"{d.strftime('%B')} {d.day}, {d.year}"
    for fi, fr in enumerate(frames, 1):
        for step in range(1, fr["steps"] + 1):
            r.step = step
            r.hidden = False
            r.slide_frame = fr
            head = ""
            if fr.get("title") and not fr.get("plain"):
                sub = f'<div class="tex-fsub">{r.inl(fr["subtitle"])}</div>' if fr.get("subtitle") else ""
                head = f'<div class="tex-ftitle"><div>{r.inl(fr["title"])}</div>{sub}</div>'
            body = r.blocks(fr["blocks"])
            foot = ""
            if theme == "madrid" and not fr.get("plain"):
                foot = (f'<div class="tex-footline"><span>{html.escape(author)}</span><span>{html.escape(short_title)}'
                        f'</span><span>{html.escape(date)}<b>{fi} / {total}</b></span></div>')
            elif theme == "metropolis" and not fr.get("plain"):
                foot = f'<div class="tex-footline tex-foot-num"><span></span><span></span><span>{fi}</span></div>'
            align = " tex-ftop" if fr.get("top") else ""
            out.append(f'<div class="tex-slide" style="width:{SLIDE_W}pt;height:{SLIDE_H}pt">{head}'
                       f'<div class="tex-fbody{align}">{body}</div>{foot}</div>')
    out.append("</div>")
    r.step = None
    return "".join(out)

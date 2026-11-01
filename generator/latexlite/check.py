"""Grading LaTeX problems: compile the answer and the reference, compare the documents (not the source).

A problem's `latex` spec: mode "body" (the user writes the body; `preamble` is given), "document" (a whole file)
"preamble" (the user writes the preamble; `body` is given) or "bib" (the user writes references.bib), `files` (references.bib, data, images), `check`
(what besides the body counts: "page", "slides", "preamble"), and `target` (the compiled reference, shown).
Equivalent markup passes: text runs compare by their look (\\textbf and \\bfseries, \\emph and \\textit), math by a
canonical form, references by what they print. `require` / `forbid` name commands (\\cref), environments
(env:align) or packages (pkg:booktabs) used.
"""

from __future__ import annotations

import html
import re

from .compile import compile_doc
from .render import render

HELLO = "This is pdfTeX, Version 3.141592653-2.6-1.40.26 (latex-lite)\nentering extended mode\n(./main.tex\nLaTeX2e <2024-11-01>"


def build(problem: dict, code: str) -> dict:
    """Compile an answer as the problem's mode says."""
    spec = problem.get("latex") or {}
    files = dict(spec.get("files") or {})
    mode = spec.get("mode", "body")
    if mode == "body":
        files["main.tex"] = code
        return compile_doc(files, "main.tex", preamble=spec.get("preamble", ""))
    if mode == "preamble":
        files["main.tex"] = code
        files["body.tex"] = spec.get("body", "")
        return compile_doc(files, "body.tex", preamble=code, preamble_file="main.tex")
    if mode == "bib":
        files["references.bib"] = code
        files["main.tex"] = spec.get("body", "")
        return compile_doc(files, "main.tex", preamble=spec.get("preamble", ""))
    files["main.tex"] = code
    return compile_doc(files, "main.tex")


def log_text(r) -> str:
    out = [HELLO]
    e = r["engine"]
    for name in e.pkgs:
        out.append(f"({name}.sty)")
    for err in r["errors"]:
        out.append(f"! {err['msg']}")
        if err["line"]:
            out.append(f"l.{err['line']} {err['before']}")
            if err["after"]:
                out.append(" " * (len(f"l.{err['line']} ") + len(err["before"])) + err["after"])
        if err["help"]:
            out.append(err["help"])
        out.append("")
    for w in r["warnings"]:
        out.append(w["msg"])
    if r["errors"]:
        out.append(f"\n{len(r['errors'])} error{'s' if len(r['errors']) != 1 else ''}: the PDF may be incomplete.")
    pages = 1 + sum(1 for b in r["doc"]["blocks"] if b.get("t") == "newpage")
    if r["doc"].get("frames") is not None:
        pages = sum(f["steps"] for f in r["doc"]["frames"])
    out.append(f"Output written on main.pdf ({pages} page{'s' if pages != 1 else ''}).")
    return "\n".join(out)


def problems_of(r) -> list[dict]:
    """Errors and warnings for the page's list (Overleaf-style)."""
    out = []
    for err in r["errors"]:
        out.append({"kind": "error", "file": err["file"], "line": err["line"], "msg": err["msg"], "help": err["help"],
                    "context": err["before"].strip()[-40:] + err["after"][:30]})
    for w in r["warnings"]:
        out.append({"kind": "warning", "file": w["file"], "line": w["line"], "msg": w["msg"], "help": ""})
    return out


# ---------------------------------------------------------------- normal form

def _style_tag(st):
    if st is None:
        return ""
    bits = []
    if st.bold:
        bits.append("b")
    if st.shape != "n":
        bits.append(st.shape)
    if st.family != "rm":
        bits.append(st.family)
    if st.size != "normalsize":
        bits.append(st.size)
    if st.color:
        bits.append(st.color)
    if st.under:
        bits.append("u")
    return ",".join(bits)


def norm_inlines(nodes) -> list:
    out = []
    for x in nodes or []:
        t = x.get("t")
        if t == "text":
            s = x["s"].replace(" ", " ").replace("­", "")
            tag = _style_tag(x["st"]) if s.strip() else ""
            if out and out[-1][0] == "t" and (out[-1][1] == tag or not s.strip()):
                out[-1] = ("t", out[-1][1], out[-1][2] + s)
            elif out and out[-1][0] == "t" and not out[-1][2].strip():
                out[-1] = ("t", tag, out[-1][2] + s)
            else:
                out.append(("t", tag, s))
        elif t == "math":
            out.append(("m", x["canon"]))
        elif t == "ref":
            out.append(("ref", x.get("txt", "??")))
        elif t == "cite":
            out.append(("cite", x.get("txt", "?")))
        elif t == "fn":
            out.append(("fn", str(x["n"])))
        elif t == "link":
            out.append(("link", x["url"], tuple(norm_inlines(x["c"]))))
        elif t == "verb":
            out.append(("t", "tt", x["s"]))
        elif t == "space":
            out.append(("sp", str(x["w"])))
        elif t == "br":
            out.append(("br",))
        elif t == "box":
            out.append(("box", norm_block(x["b"])))
        elif t == "logo":
            out.append(("t", _style_tag(x["st"]), x["s"]))
        elif t == "runin":
            out.append(("runin", x.get("num"), tuple(norm_inlines(x["c"]))))
        elif t in ("qed",):
            out.append(("qed",))
        elif t == "algcomment":
            out.append(("comment", tuple(norm_inlines(x["c"]))))
        elif t in ("only", "uncover", "invisible", "alertspec"):
            out.append((t, x.get("spec"), tuple(norm_inlines(x["c"]))))
        elif t == "pause":
            out.append(("pause", x["from"]))
        elif t == "rule":
            out.append(("rule",))
    # whitespace: collapsed, trimmed at the ends
    res = []
    for item in out:
        if item[0] == "t":
            s = re.sub(r"\s+", " ", item[2])
            res.append(("t", item[1], s))
        else:
            res.append(item)
    if res and res[0][0] == "t":
        res[0] = ("t", res[0][1], res[0][2].lstrip())
    if res and res[-1][0] == "t":
        res[-1] = ("t", res[-1][1], res[-1][2].rstrip())
    return [r for r in res if not (r[0] == "t" and r[2] == "")]


def norm_block(b) -> tuple:
    t = b.get("t")
    if t == "par":
        return ("par", b.get("align", "j"), tuple(norm_inlines(b["c"])))
    if t == "heading":
        return ("heading", b["level"], b.get("num"), b.get("star"), tuple(norm_inlines(b["c"])))
    if t == "title":
        return ("title", tuple(norm_inlines(b["title"])), tuple(tuple(norm_inlines(a)) for a in b["authors"]),
                tuple(norm_inlines(b["date"])))
    if t == "abstract":
        return ("abstract", tuple(norm_block(x) for x in b["c"]))
    if t == "list":
        return ("list", b["kind"], tuple((tuple(norm_inlines(i["label"])), tuple(norm_block(x) for x in i["c"]),
                                          i.get("overlay")) for i in b["items"]))
    if t == "display":
        return ("display", "numbered" if any(r.get("tag") for r in b["rows"] if "tex" in r) else "plain",
                tuple((r.get("canon"), r.get("tag")) if "tex" in r else ("text", tuple(norm_inlines(r["intertext"])))
                      for r in b["rows"]), b["env"].rstrip("*") if b["env"].rstrip("*") in ("multline", "gather")
                else "")
    if t == "float":
        return ("float", b["kind"], tuple(norm_block(x) for x in b["c"]))
    if t == "caption":
        return ("caption", b["label"], tuple(norm_inlines(b["c"])))
    if t == "tabular":
        cols = tuple((c["align"], c.get("after", 0), c.get("width")) for c in b["cols"])
        rows = []
        for r in b["rows"]:
            rules = tuple(x[0] for x in r.get("rules") or [])
            if r.get("cells") is None:
                rows.append(("rules", rules))
                continue
            cells = tuple((tuple(norm_inlines(c["c"])) if "c" in c else tuple(norm_block(x) for x in c["blocks"]),
                           c.get("span", 1), c.get("rowspan", 1), c.get("align")) for c in r["cells"])
            rows.append((rules, cells))
        return ("tabular", cols, b.get("bars_before", 0), tuple(rows))
    if t == "graphic":
        return ("graphic", b["file"], _w(b.get("w")), _w(b.get("h")), b.get("scale"), b.get("angle"))
    if t in ("minipage", "subfloat"):
        return (t, _w(b.get("w")), tuple(norm_block(x) for x in b["c"]))
    if t == "theorem":
        return ("theorem", tuple(norm_inlines(b["head"])), b.get("num"),
                tuple(norm_inlines(b["note"])) if b.get("note") is not None else None, b["style"],
                tuple(norm_block(x) for x in b["c"]))
    if t == "proof":
        return ("proof", tuple(norm_inlines(b["title"])) if b.get("title") else None,
                tuple(norm_block(x) for x in b["c"]))
    if t == "group":
        return ("group", b["align"], tuple(norm_block(x) for x in b["c"]))
    if t == "quote":
        return ("quote", b["kind"], tuple(norm_block(x) for x in b["c"]))
    if t == "verbatim":
        return ("verbatim", b["s"].rstrip())
    if t == "listing":
        return ("listing", tuple("".join(s for _, s in ln) for ln in b["lines"]), b.get("lang"), b.get("numbers"),
                b.get("frame"), tuple(norm_inlines(b["caption"]["c"])) if b.get("caption") else None)
    if t == "algorithmic":
        return ("algorithmic", tuple((ln["n"] is not None, ln["indent"], tuple(norm_inlines(ln["c"])))
                                     for ln in b["lines"]))
    if t == "tcolorbox":
        return ("tcolorbox", b.get("back"), b.get("frame"), tuple(norm_inlines(b["title"])) if b.get("title") else None,
                tuple(norm_block(x) for x in b["c"]))
    if t == "colorbox":
        return ("colorbox", b.get("bg"), b.get("frame"), tuple(norm_inlines(b["c"])))
    if t == "mbox":
        return ("mbox", b.get("frame"), tuple(norm_inlines(b["c"])))
    if t == "multicols":
        return ("multicols", b["n"], tuple(norm_block(x) for x in b["c"]))
    if t == "tikz":
        sc = b["scene"]
        return ("tikz", tuple(sorted((n["text"], tuple(n["at"]), n["shape"], n["fill"] or "", n["draw"])
                                     for n in sc["nodes"])),
                tuple(sorted((p["kind"], tuple(tuple(x) for x in p["points"]),
                              tuple(str(s) for s in p["shapes"]), str(p["arrows"]), str(p["draw"]),
                              str(p["fill"]), str(p["dash"])) for p in sc["paths"])))
    if t == "plot":
        d = b["data"]
        return ("plot", tuple((tuple(tuple(p) for p in pl["pts"]), pl["color"], pl["mark"], pl["dash"])
                              for pl in d["plots"]), tuple(d["legend"]), d["xlabel"], d["ylabel"], d["title"],
                d["xmode"], d["ymode"], d["grid"], d["legend_pos"])
    if t == "bibliography":
        return ("bibliography",)
    if t in ("vspace",):
        return ("vspace",)
    if t == "newpage":
        return ("newpage",)
    if t in ("toc", "lof", "lot", "btoc", "columnbreak"):
        return (t,)
    if t == "btitle":
        return ("btitle", tuple(norm_inlines(b["title"])), tuple(norm_inlines(b.get("subtitle"))),
                tuple(tuple(norm_inlines(a)) for a in b["authors"]))
    if t == "columns":
        return ("columns", tuple((_w(c["w"]), tuple(norm_block(x) for x in c["c"])) for c in b["cols"]))
    if t == "bblock":
        return ("bblock", b["kind"], tuple(norm_inlines(b["title"])), tuple(norm_block(x) for x in b["c"]))
    if t in ("pause", "onslide"):
        return (t, b.get("from") or b.get("spec"))
    if t in ("transform", "resized"):
        return (t, b.get("op"), b.get("v"), tuple(norm_inlines(b["c"])))
    return (t,)


def _w(w):
    if not w:
        return None
    s = str(w).replace(" ", "")
    m = re.match(r"^(\d*\.?\d+)?\\(linewidth|textwidth|columnwidth)$", s)
    if m:
        return f"{float(m.group(1) or 1):g}\\linewidth"
    return s


def norm_doc(r, checks) -> dict:
    d = r["doc"]
    opts = d.get("options", [])
    size = next((o for o in opts if o in ("10pt", "11pt", "12pt")), "10pt")
    paper = next((o for o in opts if o.endswith("paper")), "letterpaper")
    out = {"setup": (d["class"], size, paper, "twocolumn" in opts, "landscape" in opts),
           "blocks": [norm_block(b) for b in d["blocks"] if b.get("t") not in ("vspace",)],
           "footnotes": [(str(f["n"]), tuple(norm_block(x) for x in f["c"])) for f in d["footnotes"]]}
    if d.get("bib"):
        out["bib"] = [(en["label"], tuple(norm_inlines(en["c"]))) for en in d["bib"].get("entries", [])]
    if d.get("frames") is not None:
        out["frames"] = [{"title": tuple(norm_inlines(f["title"])) if f.get("title") else None,
                          "steps": f["steps"], "blocks": [norm_block(b) for b in f["blocks"]]}
                         for f in d["frames"]]
    if "page" in checks:
        e = r["engine"]
        pg = d["page"]
        hf = {}
        from .render import Renderer
        rr = Renderer(d, e)
        for where in ("head", "foot"):
            for pos in "lcr":
                toks = pg[where].get(pos)
                hf[where + pos] = re.sub(r"<[^>]+>", "", rr.hf(toks, 1)) if toks else ""
        out["page"] = {"style": pg["style"], "hf": hf, "spacing": pg.get("spacing"),
                       "margins": tuple(round(x) for x in rr.page_box()), "parskip": pg.get("parskip")}
    return out


# ---------------------------------------------------------------- messages

def show_inl(nodes, limit=90) -> str:
    """Inline runs as readable text: **bold**, *italic*, `tt`, $math$."""
    out = []
    for x in nodes:
        k = x[0]
        if k == "t":
            s = x[2]
            tags = x[1].split(",") if x[1] else []
            if "tt" in tags:
                s = f"`{s}`"
            if "it" in tags or "sl" in tags:
                s = f"*{s}*"
            if "b" in tags:
                s = f"**{s}**"
            if "sc" in tags:
                s = f"{s} (small caps)"
            colors = [t for t in tags if t.startswith("#")]
            if colors:
                s = f"{s} (in {colors[0]})"
            sizes = [t for t in tags if t in ("tiny", "scriptsize", "footnotesize", "small", "large", "Large",
                                              "LARGE", "huge", "Huge")]
            if sizes:
                s = f"{s} (\\{sizes[0]})"
            if "u" in tags:
                s = f"{s} (underlined)"
            out.append(s)
        elif k == "m":
            out.append(f"${x[1]}$")
        elif k in ("ref", "cite"):
            out.append(x[1])
        elif k == "fn":
            out.append(f"[footnote {x[1]}]")
        elif k == "link":
            out.append(show_inl(x[2]) + f" <{x[1]}>")
        elif k == "sp":
            out.append({"quad": "\\quad", "qquad": "\\qquad", "thin": "\\,", "hfill": "\\hfill",
                        "fill": "\\hfill"}.get(x[1], " "))
        elif k == "br":
            out.append(" \\\\ ")
        elif k == "box":
            out.append(f"[{describe(x[1])}]")
        elif k == "comment":
            out.append(f"  \u25b7 {show_inl(x[1])}")
        elif k in ("only", "uncover", "invisible", "alertspec"):
            out.append(f"\\{'alert' if k == 'alertspec' else k}<{x[1]}>{{{show_inl(x[2])}}}")
    s = "".join(out)
    return s if len(s) <= limit else s[:limit - 1] + "…"


def describe(b) -> str:
    k = b[0]
    if k == "par":
        return f"the paragraph “{show_inl(b[2], 60)}”"
    if k == "heading":
        name = ["chapter", "section", "subsection", "subsubsection"][b[1]] if 0 <= b[1] <= 3 else "heading"
        return f"the {name} heading {(b[2] + ' ') if b[2] else ''}“{show_inl(b[4], 50)}”" + \
            (" (unnumbered)" if b[3] else "")
    if k == "list":
        return f"a{'n' if b[1][0] in 'ie' else ''} {b[1]} list with {len(b[2])} item{'s' if len(b[2]) != 1 else ''}"
    if k == "display":
        tags = [r[1] for r in b[2] if r[0] != "text" and r[1]]
        return "a display" + (f" numbered {', '.join(tags)}" if tags else " (unnumbered)")
    if k == "float":
        return f"a {b[1]}"
    if k == "caption":
        return f"the caption “{b[1]} {show_inl(b[2], 50)}”"
    if k == "tabular":
        return f"a table with {len(b[1])} column{'s' if len(b[1]) != 1 else ''}"
    if k == "graphic":
        return f"the image {b[1]}"
    if k == "theorem":
        return f"{show_inl(b[1])} {b[2] or ''}".strip()
    if k == "title":
        return "the title block"
    return {"abstract": "the abstract", "verbatim": "a verbatim block", "listing": "a code listing",
            "algorithmic": "an algorithm", "proof": "a proof", "toc": "the table of contents", "tikz": "a TikZ "
            "picture", "plot": "a plot", "tcolorbox": "a tcolorbox", "group": "a center/flush block",
            "quote": "a quote", "multicols": "a multicols block", "minipage": "a minipage",
            "bibliography": "the bibliography", "newpage": "a page break", "columns": "the columns",
            "bblock": "a block", "btitle": "the title page"}.get(k, k)


def _first_diff_inl(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def diff_blocks(exp, got, where="") -> str | None:
    for i, (a, b) in enumerate(zip(exp, got)):
        if a == b:
            continue
        return diff_block(a, b, where)
    if len(got) < len(exp):
        return f"Missing{where}: {describe(exp[len(got)])}."
    if len(got) > len(exp):
        return f"Unexpected{where}: {describe(got[len(exp)])} (the target has nothing there)."
    return None


def diff_block(a, b, where="") -> str:
    if a[0] != b[0]:
        return f"Expected {describe(a)}{where}; got {describe(b)}."
    k = a[0]
    if k == "par":
        if a[2] != b[2]:
            i = _first_diff_inl(a[2], b[2])
            if i < min(len(a[2]), len(b[2])) and a[2][i][0] == b[2][i][0] == "box":
                return diff_block(a[2][i][1], b[2][i][1], where)     # a picture, table or box inside the paragraph
            ea = show_inl(a[2][max(0, i - 1):i + 2], 80)
            gb = show_inl(b[2][max(0, i - 1):i + 2], 80)
            if not gb:
                return f"In {describe(a)}{where}, something is missing: {ea}"
            return f"In {describe(a)}{where}: expected “{ea}”, got “{gb}”."
        return f"{describe(a).capitalize()}{where} should be {'centered' if a[1] == 'c' else 'aligned ' + a[1]}."
    if k == "heading":
        if a[1] != b[1]:
            return f"Expected {describe(a)}; got {describe(b)} (a different level)."
        if a[3] != b[3]:
            return f"{describe(a).capitalize()} should be {'unnumbered (\\section*)' if a[3] else 'numbered'}."
        return f"Expected {describe(a)}; got {describe(b)}."
    if k == "list":
        if a[1] != b[1]:
            return f"Expected {describe(a)}{where}; got {describe(b)}."
        if len(a[2]) != len(b[2]):
            return f"Expected {describe(a)}{where}; got {len(b[2])} item{'s' if len(b[2]) != 1 else ''}."
        for n, (x, y) in enumerate(zip(a[2], b[2]), 1):
            if x[0] != y[0]:
                return f"Item {n} of the {a[1]} list should be labeled “{show_inl(x[0])}”, not " \
                       f"“{show_inl(y[0])}”."
            if x[1] != y[1]:
                return diff_blocks(list(x[1]), list(y[1]), f" (item {n})") or f"Item {n} differs."
            if x[2] != y[2]:
                def ov(spec):
                    return f"overlay <{spec}>" if spec else "every slide (no <...> spec)"
                return f"Item {n} should appear on {ov(x[2])}, not on {ov(y[2])}."
        return f"The {a[1]} list differs."
    if k == "display":
        ra, rb = a[2], b[2]
        if len(ra) != len(rb):
            return f"Expected a display with {len(ra)} line{'s' if len(ra) != 1 else ''}{where}; got {len(rb)}."
        for n, (x, y) in enumerate(zip(ra, rb), 1):
            if x[0] != y[0]:
                return f"Line {n} of the display{where} should be ${x[0]}$; it is ${y[0]}$."
            if x[1] != y[1]:
                if x[1] and not y[1]:
                    return f"Line {n} of the display should be numbered {x[1]}."
                if y[1] and not x[1]:
                    return f"Line {n} of the display shouldn't be numbered (it's {y[1]})."
                return f"Line {n} of the display should be numbered {x[1]}, not {y[1]}."
        return "The display uses a different environment (look at how the lines are arranged)."
    if k in ("float",):
        if a[1] != b[1]:
            return f"Expected a {a[1]}; got a {b[1]}."
        return diff_blocks(list(a[2]), list(b[2]), f" in the {a[1]}") or f"The {a[1]} differs."
    if k == "caption":
        if a[1] != b[1]:
            return f"The caption should start “{a[1]}”, not “{b[1]}”."
        return f"The caption should read “{show_inl(a[2])}”, not “{show_inl(b[2])}”."
    if k == "tabular":
        if len(a[1]) != len(b[1]):
            return f"The table should have {len(a[1])} columns; it has {len(b[1])}."
        if a[1] != b[1] or a[2] != b[2]:
            exp = "".join(("|" * a[2] if i == 0 else "") + c[0] + "|" * c[1] for i, c in enumerate(a[1]))
            gt = "".join(("|" * b[2] if i == 0 else "") + c[0] + "|" * c[1] for i, c in enumerate(b[1]))
            return f"The table's columns should be {{{exp}}}; they are {{{gt}}}."
        ra, rb = a[3], b[3]
        if len(ra) != len(rb):
            return f"The table should have {sum(1 for r in ra if r[0] != 'rules')} rows (and its rules); it has " \
                   f"{sum(1 for r in rb if r[0] != 'rules')}."
        for n, (x, y) in enumerate(zip(ra, rb), 1):
            if x != y:
                if x[0] != y[0]:
                    return f"Row {n} of the table: the rules above it should be {list(x[0]) or 'none'}, not " \
                           f"{list(y[0]) or 'none'}."
                for c, (cx, cy) in enumerate(zip(x[1], y[1]), 1):
                    if cx != cy:
                        if cx[1] != cy[1]:
                            return f"Row {n}, cell {c} should span {cx[1]} columns."
                        return f"Row {n}, cell {c} of the table should be “{show_inl(cx[0])}”, not " \
                               f"“{show_inl(cy[0])}”."
                return f"Row {n} of the table should have {len(x[1])} cells; it has {len(y[1])}."
        return "The table differs."
    if k == "graphic":
        if a[1] != b[1]:
            return f"The image should be {a[1]}, not {b[1]}."
        return f"The image {a[1]} should be {a[2] or 'at its natural size'}" + (f" wide" if a[2] else "") + \
            (f", not {b[2]}" if b[2] else "") + "."
    if k in ("minipage", "subfloat"):
        if a[1] != b[1]:
            return f"The {k} should be {a[1]} wide, not {b[1]}."
        return diff_blocks(list(a[2]), list(b[2]), f" in the {k}") or f"The {k} differs."
    if k == "theorem":
        if a[1] != b[1] or a[2] != b[2]:
            return f"Expected {describe(a)}; got {describe(b)}."
        if a[3] != b[3]:
            return f"{describe(a)} should have the note ({show_inl(a[3] or ())})."
        if a[4] != b[4]:
            return f"{describe(a)} should be in the {a[4]} style (\\theoremstyle{{{a[4]}}})."
        return diff_blocks(list(a[5]), list(b[5]), f" in {describe(a)}") or "The theorem differs."
    if k in ("group", "quote", "abstract", "proof", "multicols", "columns", "bblock", "tcolorbox"):
        inner_a = a[-1] if isinstance(a[-1], tuple) else ()
        inner_b = b[-1] if isinstance(b[-1], tuple) else ()
        if a[:-1] != b[:-1]:
            return f"Expected {describe(a)}{where} {a[1:-1]}; got {b[1:-1]}." if k != "bblock" else \
                f"Expected a {a[1]} titled “{show_inl(a[2])}”."
        return diff_blocks(list(inner_a), list(inner_b), f" in {describe(a)}") or f"{describe(a).capitalize()} differs."
    if k == "listing":
        if a[1] != b[1]:
            for n, (x, y) in enumerate(zip(a[1], b[1]), 1):
                if x != y:
                    return f"Line {n} of the listing should be `{x}`, not `{y}`."
            return f"The listing should have {len(a[1])} lines; it has {len(b[1])}."
        return f"The listing's options differ: language={a[2]}, numbers={a[3]}, frame={a[4]}."
    if k == "verbatim":
        return "The verbatim text differs."
    if k == "algorithmic":
        numbered = any(x[0] for x in a[1])
        printed = 0
        for n, (x, y) in enumerate(zip(a[1], b[1]), 1):
            printed += x[0]
            # name the line as the PDF numbers it (\Require / \Ensure lines have no number)
            name = f"Line {printed}" if x[0] or not numbered else f"The unnumbered line {n}"
            if x != y:
                if x[1] != y[1]:
                    return f"{name} of the algorithm should be indented {x[1]} level(s), not {y[1]} (check the " \
                           f"\\For / \\EndFor structure)."
                return f"{name} of the algorithm should read “{show_inl(x[2])}”, not " \
                       f"“{show_inl(y[2])}”."
        return f"The algorithm should have {len(a[1])} lines; it has {len(b[1])}."
    if k == "tikz":
        if a[1] != b[1]:
            na, nb = a[1], b[1]
            if len(na) != len(nb):
                return f"The picture should have {len(na)} nodes; it has {len(nb)}."
            for x, y in zip(na, nb):
                if x != y and x[:3] == y[:3]:
                    if x[3] != y[3]:
                        return f"The node “{x[0]}” at {x[1]} should be filled {x[3] or 'with nothing'}, not " \
                               f"{y[3] or 'with nothing'}."
                    return f"The node “{x[0]}” at {x[1]} should {'' if x[4] else 'not '}have a border (draw)."
                if x != y:
                    return f"A node differs: expected “{x[0]}” at {x[1]} ({x[2]}), got “{y[0]}” " \
                           f"at {y[1]} ({y[2]})."
        if len(a[2]) != len(b[2]):
            return f"The picture should have {len(a[2])} paths (lines, arrows, shapes); it has {len(b[2])}."
        return "Some path in the picture differs (its points, arrows, colors or dashing)."
    if k == "plot":
        for name, idx in (("x label", 3), ("y label", 4), ("title", 5), ("x axis mode", 6), ("y axis mode", 7),
                          ("grid", 8), ("legend position", 9)):
            if a[idx] != b[idx]:
                return f"The plot's {name} should be {a[idx]!r}, not {b[idx]!r}."
        if len(a[1]) != len(b[1]):
            return f"The plot should have {len(a[1])} curves; it has {len(b[1])}."
        if a[2] != b[2]:
            return f"The legend should read {list(a[2])}, not {list(b[2])}."
        for n, (x, y) in enumerate(zip(a[1], b[1]), 1):
            if x[0] != y[0]:
                return f"Curve {n} has different points than the target."
            if x[1:] != y[1:]:
                return f"Curve {n} should be drawn {x[1]} with mark {x[2]} {'(' + x[3] + ')' if x[3] else ''}."
        return "The plot differs."
    return f"Expected {describe(a)}{where}; got {describe(b)}."


def compare(exp: dict, got: dict) -> str | None:
    if exp["setup"] != got["setup"]:
        for i, what in enumerate(("document class", "font size option", "paper size option", "twocolumn option",
                                  "landscape option")):
            if exp["setup"][i] != got["setup"][i]:
                v = exp["setup"][i]
                if isinstance(v, bool):
                    return f"The {what} should be {'on' if v else 'off'} (\\documentclass[...]{{...}})."
                return f"The {what} should be {v}, not {got['setup'][i]}."
    msg = diff_blocks(exp["blocks"], got["blocks"])
    if msg:
        return msg
    if exp.get("frames") is not None:
        fa, fb = exp["frames"], got.get("frames") or []
        if len(fa) != len(fb):
            return f"The talk should have {len(fa)} frames; it has {len(fb)}."
        for n, (x, y) in enumerate(zip(fa, fb), 1):
            if x["title"] != y["title"]:
                return f"Frame {n}'s title should be “{show_inl(x['title'] or ())}”, not " \
                       f"“{show_inl(y['title'] or ())}”."
            m = diff_blocks(x["blocks"], y["blocks"], f" in frame {n}")
            if m:
                return m
            if x["steps"] != y["steps"]:
                return f"Frame {n} should take {x['steps']} slide{'s' if x['steps'] != 1 else ''} (overlay steps); " \
                       f"it takes {y['steps']}."
    if exp["footnotes"] != got["footnotes"]:
        if len(exp["footnotes"]) != len(got["footnotes"]):
            return f"There should be {len(exp['footnotes'])} footnote(s); there are {len(got['footnotes'])}."
        for x, y in zip(exp["footnotes"], got["footnotes"]):
            if x != y:
                return diff_blocks(list(x[1]), list(y[1]), f" in footnote {x[0]}") or f"Footnote {x[0]} differs."
    if exp.get("bib") != got.get("bib"):
        ea, gb = exp.get("bib") or [], got.get("bib") or []
        if len(ea) != len(gb):
            return f"The bibliography should list {len(ea)} entr{'ies' if len(ea) != 1 else 'y'}; it lists {len(gb)}."
        for x, y in zip(ea, gb):
            if x != y:
                return f"A reference should read “{show_inl(x[1], 120)}”, not “{show_inl(y[1], 120)}”."
    if "page" in exp and exp["page"] != got.get("page"):
        pa, pb = exp["page"], got["page"]
        if pa["style"] != pb["style"]:
            return f"The page style should be {pa['style']} (\\pagestyle{{{pa['style']}}})."
        for k2 in pa["hf"]:
            if pa["hf"][k2] != pb["hf"][k2]:
                where = {"head": "header", "foot": "footer"}[k2[:4]]
                pos = {"l": "left", "c": "center", "r": "right"}[k2[4]]
                return f"The {pos} of the {where} should read “{pa['hf'][k2]}”, not “{pb['hf'][k2]}”."
        if pa["margins"] != pb["margins"]:
            return "The page size or margins differ (geometry)."
        if pa["spacing"] != pb["spacing"]:
            return "The line spacing differs."
        if pa["parskip"] != pb["parskip"]:
            return "The paragraph spacing (\\parskip) differs."
    return None


# ---------------------------------------------------------------- the checker

def _first_error(r):
    err = r["errors"][0]
    where = f"line {err['line']}" if err["line"] else ""
    ctx = (err["before"].strip()[-30:] + err["after"][:20]).strip()
    s = f"{err['msg']}" + (f" ({where}: {ctx})" if where else "")
    if err["help"]:
        s += f" {err['help']}"
    more = len(r["errors"]) - 1
    if more:
        s += f" (+{more} more error{'s' if more != 1 else ''}: see the log)"
    return s


def used_names(r) -> set:
    return {k for k, v in r["used"].items() if v}


def checker(problem: dict, code: str, grade: bool) -> dict:
    spec = problem.get("latex") or {}
    try:
        r = build(problem, code)
    except Exception as ex:  # noqa: BLE001  (latex-lite itself failed: show it, don't crash the page)
        return {"ok": False, "correct": False, "compiled": False, "stdout": "", "value": None,
                "error": f"latex-lite crashed: {type(ex).__name__}: {ex}", "message": "", "console": []}
    page = render(r)
    out = {"ok": not r["errors"], "stdout": "", "value": None, "error": None, "console": [], "html": page,
           "log": log_text(r), "problems": problems_of(r)}
    if not grade:
        return out
    out.update(correct=False, compiled=True, message="")
    if r["errors"]:
        out.update(compiled=False, message=_first_error(r))
        return out
    used = used_names(r)
    for req in problem.get("require") or []:
        alts = req.split("|")
        if not any(a in used for a in alts):
            shown = " or ".join(_shown(a) for a in alts)
            out["message"] = f"This problem asks you to use {shown}."
            return out
    for f in problem.get("forbid") or []:
        if f in used:
            out["message"] = f"Solve this one without {_shown(f)}."
            return out
    ref = build(problem, problem["solution"])
    checks = set(spec.get("check") or [])
    msg = compare(norm_doc(ref, checks), norm_doc(r, checks))
    if msg is None and r["warnings"] and not ref["warnings"]:
        msg = "It compiles, but with a warning: " + r["warnings"][0]["msg"]
    if msg:
        out["message"] = msg
        return out
    out["correct"] = True
    return out


def _shown(name: str) -> str:
    if name.startswith("env:"):
        return f"the {name[4:]} environment"
    if name.startswith("pkg:"):
        return f"the {name[4:]} package"
    return f"`{name}`"

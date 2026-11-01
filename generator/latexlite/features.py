"""Sectioning, the title block, lists, display math, theorems and the simple environments (center, quote, abstract,
verbatim, minipage, multicols, tcolorbox)."""

from __future__ import annotations

import re

from . import mathmode
from .commands import color_of, dim_pt
from .engine import DISPLAY_ENVS, Stop, cmd, env
from .model import text
from .packages import SECTION_LEVELS
from .tokens import Tok, detokenize

LOREM = [
    "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Ut purus elit, vestibulum ut, placerat ac, "
    "adipiscing vitae, felis. Curabitur dictum gravida mauris. Nam arcu libero, nonummy eget, consectetuer id, "
    "vulputate a, magna. Donec vehicula augue eu neque. Pellentesque habitant morbi tristique senectus et netus et "
    "malesuada fames ac turpis egestas. Mauris ut leo.",
    "Nam dui ligula, fringilla a, euismod sodales, sollicitudin vel, wisi. Morbi auctor lorem non justo. Nam lacus "
    "libero, pretium at, lobortis vitae, ultricies et, tellus. Donec aliquet, tortor sed accumsan bibendum, erat "
    "ligula aliquet magna, vitae ornare odio metus a mi.",
    "Nulla malesuada porttitor diam. Donec felis erat, congue non, volutpat at, tincidunt tristique, libero. "
    "Vivamus viverra fermentum felis. Donec nonummy pellentesque ante. Phasellus adipiscing semper elit. Proin "
    "fermentum massa ac quam. Sed diam turpis, molestie vitae, placerat a, molestie nec, leo.",
]
SECTION_NAMES = {"part": "Part", "chapter": "Chapter"}


# ---------------------------------------------------------------- sections

def do_section(e, t):
    name = t.v
    star = e.read_star()
    short = e.read_opt()
    title_toks = e.read_arg(name)
    level = SECTION_LEVELS[name]
    if e.beamer is not None:
        e.beamer.section(name, title_toks, t)
        return
    numbered = not star and level <= e.secnumdepth
    num = None
    if numbered:
        e.counters.step(name)
        num = e.counters.the(name)
        kind = "appendix" if e.appendix and name == ("chapter" if e.counters.report else "section") else name
        e.set_label_target(kind, num)
    title = e.capture(title_toks)
    e.end_par()
    if level >= 4:
        # \paragraph and \subparagraph are run-in headings: the text follows on the same line
        f = e.frame()
        f["noindent"] = True
        p = e.start_par()
        p["indent"] = False
        p["c"].append({"t": "runin", "num": num, "c": title, "level": level})
        return
    e.add_block({"t": "heading", "level": level, "num": num, "c": title, "star": star, "kind": name,
                 "appendix": e.appendix})
    if not star and level <= 3:
        e.toc.append({"level": level, "num": num, "c": e.capture(short) if short else title, "kind": name})
    e.frame()["noindent"] = True          # the first paragraph after a heading isn't indented


def title_block(e):
    authors = [[]]
    for node in e.capture(e.title["author"] or []):
        if node["t"] == "and":
            authors.append([])
        else:
            authors[-1].append(node)
    date = e.title.get("date")
    if date is None:
        import datetime
        d = datetime.date.today()
        date_inl = [text(f"{d.strftime('%B')} {d.day}, {d.year}")]
    else:
        date_inl = e.capture(date)
    thanks = [{"n": "*" * (i + 1), "c": e.capture_blocks(x, kind="footnote")} for i, x in enumerate(e.title["thanks"])]
    e.footnotes[0:0] = thanks
    for a in authors:
        while a and a[-1]["t"] == "text" and not a[-1]["s"].strip():
            a.pop()
        if a and a[-1]["t"] == "text":
            a[-1] = {**a[-1], "s": a[-1]["s"].rstrip()}
        if a and a[0]["t"] == "text":
            a[0] = {**a[0], "s": a[0]["s"].lstrip()}
    return {"t": "title", "title": e.capture(e.title["title"]), "authors": [a for a in authors if a] or [],
            "date": date_inl}


# ---------------------------------------------------------------- lists

ITEMIZE_LABELS = ["\u2022", "\u2013", "\u2217", "\u00b7"]
ENUM_COUNTERS = ["enumi", "enumii", "enumiii", "enumiv"]


def split_items(toks: list[Tok]):
    """Top-level \\item boundaries -> (tokens before the first item, [(item tok, tokens)])."""
    depth = 0
    env_depth = 0
    pre, items = [], []
    cur = None
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.k == "bg":
            depth += 1
        elif t.k == "eg":
            depth -= 1
        elif t.k == "cs" and t.v in ("begin", "end"):
            env_depth += 1 if t.v == "begin" else -1
        if t.k == "cs" and t.v == "item" and depth == 0 and env_depth == 0:
            cur = (t, [])
            items.append(cur)
            i += 1
            continue
        (cur[1] if cur is not None else pre).append(t)
        i += 1
    return pre, items


@env("itemize", "enumerate", "description")
def list_env(e, name, t):
    body = e.env_body(name, t)
    opts = None
    # [options] right after \begin{...}
    j = 0
    while j < len(body) and body[j].k == "sp":
        j += 1
    if j < len(body) and body[j].k == "ch" and body[j].v == "[":
        k = j + 1
        depth = 0
        while k < len(body) and not (body[k].k == "ch" and body[k].v == "]" and depth == 0):
            depth += body[k].k == "bg"
            depth -= body[k].k == "eg"
            k += 1
        opt_toks = body[j + 1:k]
        if "enumitem" in e.pkg_closure:
            opts = e.kv(opt_toks)
            body = body[k + 1:]
        elif e.beamer is not None and name in ("itemize", "enumerate"):
            # beamer: [<+->] default overlays
            opts = {"_beamer": detokenize(opt_toks)}
            body = body[k + 1:]
        else:
            e.error("LaTeX Error: Something's wrong--perhaps a missing \\item.", body[j],
                    "List options like [label=(\\alph*)] need \\usepackage{enumitem}.")
            body = body[k + 1:]
    opts = opts or {}
    pre, items = split_items(body)
    if any(x.k not in ("sp", "par") for x in pre):
        bad = next(x for x in pre if x.k not in ("sp", "par"))
        e.error("LaTeX Error: Something's wrong--perhaps a missing \\item.", bad,
                "Every entry of a list starts with \\item.")
    depth = e.list_depth[name] if name != "description" else 0
    total_depth = sum(e.list_depth.values())
    if total_depth >= 6 or (name != "description" and depth >= 4):
        e.error("LaTeX Error: Too deeply nested.", t, "Lists nest at most four levels.")
        return
    e.list_depth[name] += 1
    e.end_par()
    counter = ENUM_COUNTERS[min(depth, 3)] if name == "enumerate" else None
    if counter:
        e.counters.v[counter] = 0
        if "start" in opts:
            try:
                e.counters.v[counter] = int(opts["start"]) - 1
            except ValueError:
                pass
    out_items = []
    for it_tok, toks in items:
        overlay = None
        if e.beamer is not None:
            overlay = _overlay_after_item(e, toks)
            if overlay is not None:
                toks = overlay[1]
                overlay = overlay[0]
            elif "_beamer" in opts and "+" in opts["_beamer"]:
                overlay = "+-"
            if overlay is not None:
                overlay = e.beamer.plus(overlay)
        # \item[label]
        lab_toks = None
        jj = 0
        while jj < len(toks) and toks[jj].k == "sp":
            jj += 1
        if jj < len(toks) and toks[jj].k == "ch" and toks[jj].v == "[":
            k = jj + 1
            dd = 0
            while k < len(toks) and not (toks[k].k == "ch" and toks[k].v == "]" and dd == 0):
                dd += toks[k].k == "bg"
                dd -= toks[k].k == "eg"
                k += 1
            lab_toks = toks[jj + 1:k]
            toks = toks[k + 1:]
        e.used["\\item"] += 1
        if name == "enumerate" and lab_toks is None:
            e.counters.step(counter)
            label = _enum_label(e, opts, depth, counter)
            e.set_label_target("item", _enum_ref(e, depth, counter, label))
            lab = [text(label)]
        elif lab_toks is not None:
            lab = e.capture(lab_toks, e.style.with_(bold=True) if name == "description" else None)
        else:
            lab = [text(_itemize_label(opts, depth))]
        blocks = e.capture_blocks(toks, kind="item")
        out_items.append({"label": lab, "c": blocks, "overlay": overlay})
    e.list_depth[name] -= 1
    compact = any(k in opts for k in ("noitemsep", "nosep")) or opts.get("itemsep", "").strip().startswith("0")
    e.add_block({"t": "list", "kind": name, "items": out_items, "depth": depth, "compact": compact,
                 "leftmargin": opts.get("leftmargin"), "opts": {k: v for k, v in opts.items() if k != "_beamer"}})
    if e.beamer is not None:
        e.beamer.note_overlays(out_items)


def _overlay_after_item(e, toks):
    """\\item<2-> : the overlay spec and the rest of the item."""
    j = 0
    while j < len(toks) and toks[j].k == "sp":
        j += 1
    if j < len(toks) and toks[j].k == "ch" and toks[j].v == "<":
        k = j + 1
        spec = []
        while k < len(toks) and not (toks[k].k == "ch" and toks[k].v == ">"):
            spec.append(toks[k].text())
            k += 1
        return "".join(spec), toks[k + 1:]
    return None


def _itemize_label(opts, depth):
    lab = opts.get("label")
    if isinstance(lab, str):
        return {"\\textbullet": "\u2022", "--": "\u2013", "---": "\u2014", "$\\bullet$": "\u2022",
                "$\\circ$": "\u2218", "$\\star$": "\u22c6", "$\\cdot$": "\u00b7", "$-$": "\u2212",
                "\\textendash": "\u2013", "*": "*", "-": "-"}.get(lab.strip(), lab.strip())
    return ITEMIZE_LABELS[min(depth, 3)]


def _enum_label(e, opts, depth, counter):
    n = e.counters.value(counter)
    from .model import alph, roman
    lab = opts.get("label")
    if isinstance(lab, str) and lab.strip():
        s = lab.strip()
        s = s.replace("\\arabic*", str(n)).replace("\\alph*", alph(n)).replace("\\Alph*", alph(n).upper())
        s = s.replace("\\roman*", roman(n)).replace("\\Roman*", roman(n).upper())
        s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
        s = s.replace("{", "").replace("}", "")
        return s
    return [f"{n}.", f"({alph(n)})", f"{roman(n)}.", f"{alph(n).upper()}."][min(depth, 3)]


def _enum_ref(e, depth, counter, label):
    n = e.counters.value(counter)
    from .model import alph, roman
    own = [str(n), alph(n), roman(n), alph(n).upper()][min(depth, 3)]
    if depth == 0:
        return label.rstrip(".") if label not in (f"{n}.",) else str(n)
    parent = e.counters.value(ENUM_COUNTERS[depth - 1])
    return f"{parent}{own}"


# ---------------------------------------------------------------- display math

def split_rows(toks: list[Tok]):
    """Rows of an alignment, at top-level \\\\ (not inside braces or inner environments)."""
    rows, cur = [], []
    depth = env_depth = 0
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.k == "bg":
            depth += 1
        elif t.k == "eg":
            depth -= 1
        elif t.k == "cs" and t.v in ("begin", "end"):
            env_depth += 1 if t.v == "begin" else -1
        if t.k == "cs" and t.v == "\\" and depth == 0 and env_depth == 0:
            rows.append(cur)
            cur = []
            i += 1
            # \\[3pt] or \\*
            while i < len(toks) and toks[i].k == "sp":
                i += 1
            if i < len(toks) and toks[i].k == "ch" and toks[i].v == "*":
                i += 1
            if i < len(toks) and toks[i].k == "ch" and toks[i].v == "[":
                while i < len(toks) and not (toks[i].k == "ch" and toks[i].v == "]"):
                    i += 1
                i += 1
            continue
        cur.append(t)
        i += 1
    if any(x.k != "sp" for x in cur) or not rows:
        rows.append(cur)
    return rows


def _find_cs(toks, name):
    return [i for i, t in enumerate(toks) if t.k == "cs" and t.v == name]


def _take_arg(toks, i):
    """After toks[i] (a command): its {argument} -> (arg tokens, index after)."""
    j = i + 1
    while j < len(toks) and toks[j].k == "sp":
        j += 1
    if j < len(toks) and toks[j].k == "bg":
        depth = 0
        k = j
        while k < len(toks):
            if toks[k].k == "bg":
                depth += 1
            elif toks[k].k == "eg":
                depth -= 1
                if depth == 0:
                    return toks[j + 1:k], k + 1
            k += 1
    return [], j


def display_block(e, envname: str, toks: list[Tok], t: Tok):
    toks = e.expand_math(toks)
    e._math_used(toks)
    numbered_env = envname in ("equation", "align", "gather", "multline", "flalign", "alignat", "eqnarray")
    rows_toks = split_rows(toks) if envname not in ("equation", "equation*", "displaymath") else [toks]
    if envname in ("equation", "equation*", "displaymath"):
        # LaTeX ignores a \\ there (no line break); so does the page
        toks = [x for i, x in enumerate(toks) if not (x.k == "cs" and x.v == "\\" and _top_level(toks, i))]
        rows_toks = [toks]
    rows = []
    align_env = envname not in ("equation", "equation*", "displaymath", "gather", "gather*", "multline", "multline*")
    for ri, row in enumerate(rows_toks):
        # \intertext{...}: a line of text between rows
        it = _find_cs(row, "intertext") + _find_cs(row, "shortintertext")
        if it:
            arg, end = _take_arg(row, it[0])
            rows.append({"intertext": e.capture(arg)})
            row = row[end:]
        tag = None
        nonumber = bool(_find_cs(row, "nonumber") or _find_cs(row, "notag"))
        tg = _find_cs(row, "tag")
        if tg:
            arg, end = _take_arg(row, tg[0])
            tag = "(" + detokenize(arg).strip() + ")"
            row = row[:tg[0]] + row[end:]
        labels = []
        for li in reversed(_find_cs(row, "label")):
            arg, end = _take_arg(row, li)
            labels.insert(0, (detokenize(arg).strip(), row[li]))
            row = row[:li] + row[end:]
        number = None
        if numbered_env and not nonumber and tag is None:
            if envname.startswith("multline") and ri != len(rows_toks) - 1:
                number = None
            else:
                e.counters.step("equation")
                number = e.counters.the("equation")
        elif numbered_env and envname.startswith("multline") and ri == len(rows_toks) - 1 and nonumber:
            number = None
        if tag is not None:
            if "amsmath" not in e.pkg_closure:
                e.error("Undefined control sequence.", row[0] if row else t, "\\tag comes from amsmath.")
            disp = tag
        else:
            disp = f"({number})" if number else None
        for key, lt in labels:
            if number is not None:
                e.set_label_target("equation", number)
            elif tag is not None:
                e.set_label_target("equation", tag.strip("()"))
            e.add_label(key, lt)
        e.math_check(row, env=envname if align_env else None)
        if not align_env:
            for amp in [x for x in row if x.k == "tab"]:
                if _top_level(row, row.index(amp)):
                    e.error("Misplaced alignment tab character &.", amp,
                            f"`&` aligns columns in align; {envname} has no columns.")
                    break
        rows.append({"tex": mathmode.to_katex(row), "canon": mathmode.canonical(row), "tag": disp})
    e.end_par()
    e.add_block({"t": "display", "env": envname, "rows": rows})
    # text right after the display continues the paragraph (no indent) unless a blank line follows
    nt = e.peek(skip_space=True)
    e.frame()["noindent"] = not (nt is not None and nt.k == "par")


def _top_level(toks, idx):
    depth = env_depth = 0
    for i, t in enumerate(toks[:idx]):
        if t.k == "bg":
            depth += 1
        elif t.k == "eg":
            depth -= 1
        elif t.k == "cs" and t.v in ("begin", "end"):
            env_depth += 1 if t.v == "begin" else -1
    return depth == 0 and env_depth == 0


@env(*DISPLAY_ENVS)
def display_env(e, name, t):
    body = e.env_body(name, t)
    if name in ("alignat", "alignat*"):
        # the column-pair count
        j = 0
        while j < len(body) and body[j].k == "sp":
            j += 1
        if j < len(body) and body[j].k == "bg":
            k = j
            while k < len(body) and body[k].k != "eg":
                k += 1
            body = body[k + 1:]
    e.used["env:" + name] += 1
    display_block(e, name, body, t)


@cmd("[")
def display_bracket(e, t):
    toks = e.collect_math(t, "\\]")
    e.used["\\["] += 1
    display_block(e, "displaymath", toks, t)


@cmd("(")
def inline_paren(e, t):
    toks = e.collect_math(t, "\\)")
    e.used["\\("] += 1
    e.inline_math(toks, t)


@cmd("]", ")")
def bad_close(e, t):
    e.error("LaTeX Error: Bad math environment delimiter.", t, f"\\{t.v} without a matching opening delimiter.")


@env("math")
def math_env(e, name, t):
    body = e.env_body(name, t)
    e.inline_math(body, t)


@env("subequations")
def subequations(e, name, t):
    e.run(stop_env=name)


@cmd("eqref")
def eqref_text(e, t):
    from .commands import ref
    ref(e, t)


# ---------------------------------------------------------------- theorems and proofs

def theorem_env(e, name, t):
    th = e.theorems[name]
    note = e.read_opt()
    num = None
    if th["numbered"]:
        e.counters.step(th["counter"])
        num = e.counters.the(th["counter"])
        e.set_label_target(name, num, env=name)
    body = e.env_body(name, t)
    saved_style = e.style
    italic = th["style"] == "plain"
    e.style = e.style.with_(shape="it") if italic else e.style
    blocks = e.capture_blocks(body, kind="theorem")
    e.style = saved_style
    e.add_block({"t": "theorem", "name": name, "head": e.capture(th["title"]), "num": num,
                 "note": e.capture(note) if note is not None else None, "c": blocks, "style": th["style"],
                 "ams": "amsthm" in e.pkg_closure})


@env("proof")
def proof_env(e, name, t):
    title = e.read_opt()
    body = e.env_body(name, t)
    e.qed_done = False
    blocks = e.capture_blocks(body, kind="proof")
    e.add_block({"t": "proof", "title": e.capture(title) if title is not None else None, "c": blocks,
                 "qed": not e.qed_done})
    e.qed_done = False


# ---------------------------------------------------------------- simple environments

@env("center", "flushleft", "flushright")
def center_env(e, name, t):
    body = e.env_body(name, t)
    blocks = e.capture_blocks(body, kind="group", align={"center": "c", "flushleft": "l", "flushright": "r"}[name])
    e.add_block({"t": "group", "align": name, "c": blocks})


@env("quote", "quotation", "verse")
def quote_env(e, name, t):
    body = e.env_body(name, t)
    e.add_block({"t": "quote", "kind": name, "c": e.capture_blocks(body, kind="quote")})


@env("abstract")
def abstract_env(e, name, t):
    body = e.env_body(name, t)
    if e.cls == "beamer":
        e.error("LaTeX Error: Environment abstract undefined.", t, "Beamer has no abstract.")
        return
    saved = e.style
    e.style = e.style.with_(size="small")
    blocks = e.capture_blocks(body, kind="abstract")
    e.style = saved
    e.add_block({"t": "abstract", "c": blocks})


def take_verbatim(e, name):
    """The raw-text token the tokenizer made for a verbatim environment, and its \\end."""
    t = e.next()
    if t is None or t.k != "verb":
        return "", None
    # \end{name}
    end = e.next()
    if end is not None and end.k == "cs" and end.v == "end":
        e.read_arg("end")
    if e.env_line_stack and e.env_line_stack[-1][0] == name:
        e.env_line_stack.pop()
    return t.v, t.extra


@env("verbatim", "verbatim*", "Verbatim")
def verbatim_env(e, name, t):
    s, extra = take_verbatim(e, name)
    e.add_block({"t": "verbatim", "s": s.rstrip("\n")})
    e.used["env:verbatim"] += 1


@env("comment")
def comment_env(e, name, t):
    take_verbatim(e, name)


@env("minipage")
def minipage_env(e, name, t):
    e.read_opt()
    e.read_opt()
    e.read_opt()
    w = e.arg_text("minipage")
    if dim_pt(w) is None:
        e.error("Missing number, treated as zero.", t,
                "\\begin{minipage}{width} needs a width: 0.45\\linewidth, 5cm ...")
    body = e.env_body(name, t)
    blocks = e.capture_blocks(body, kind="minipage", align="j")
    e.add_inline({"t": "box", "b": {"t": "minipage", "w": w, "wpt": dim_pt(w), "c": blocks}})


@env("multicols", "multicols*")
def multicols_env(e, name, t):
    try:
        n = int(e.arg_text(name))
    except ValueError:
        n = 2
    e.read_opt()
    body = e.env_body(name, t)
    e.add_block({"t": "multicols", "n": n, "c": e.capture_blocks(body, kind="multicols")})


@cmd("columnbreak")
def columnbreak(e, t):
    e.add_block({"t": "columnbreak"})


@env("tcolorbox")
def tcolorbox_env(e, name, t):
    opts = e.kv(e.read_opt())
    body = e.env_body(name, t)
    back = color_of(e, opts.get("colback", "black!5!white")) if isinstance(opts.get("colback", ""), str) else None
    frame = color_of(e, opts.get("colframe", "black!75!white")) if isinstance(opts.get("colframe", ""), str) else None
    for key in ("colback", "colframe", "coltitle"):
        if isinstance(opts.get(key), str) and color_of(e, opts[key]) is None:
            e.error(f"Package xcolor Error: Undefined color `{opts[key]}'.", t, "Define it with \\definecolor first.")
    title = None
    if isinstance(opts.get("title"), str):
        from .tokens import tokenize
        title = e.capture(tokenize(opts["title"], t.file))
    e.add_block({"t": "tcolorbox", "back": back or "#f2f2f2", "frame": frame or "#404040",
                 "coltitle": color_of(e, opts["coltitle"]) if isinstance(opts.get("coltitle"), str) else "#ffffff",
                 "title": title, "c": e.capture_blocks(body, kind="tcolorbox"), "arc": opts.get("arc"),
                 "sharp": "sharp corners" in opts})


@env("document")
def nested_document(e, name, t):
    e.error("LaTeX Error: Can be used only in preamble.", t, "")


@env("thebibliography")
def thebibliography(e, name, t):
    e.arg_text(name)
    body = e.env_body(name, t)
    from .bib import manual_bibliography
    manual_bibliography(e, body, t)


@env("landscape", "threeparttable", "onlyenv", "overlayarea", "singlespace", "onehalfspace", "doublespace",
     "spacing", "titlepage", "list", "trivlist")
def pass_through(e, name, t):
    if name == "spacing":
        e.read_arg(name)
    if name == "list":
        e.read_arg(name)
        e.read_arg(name)
    e.run(stop_env=name)

"""tabular, tabular*, tabularx and longtable: column specifications, rows, cells and rules (\\hline, booktabs,
\\cline, \\cmidrule), \\multicolumn, \\multirow and siunitx S columns."""

from __future__ import annotations

import re

from .commands import dim_pt
from .engine import env
from .tokens import Tok, detokenize

RULE_CMDS = {"hline": "hline", "toprule": "top", "midrule": "mid", "bottomrule": "bottom", "cline": "cline",
             "cmidrule": "cmid", "addlinespace": "space", "specialrule": "special"}


def parse_colspec(e, toks: list[Tok], t: Tok):
    """-> list of columns {align, width, vbar_before, vbar_after, kind} (and @{} / >{} bits ignored)."""
    s = detokenize(toks)
    cols = []
    bars_before = 0
    i = 0
    # expand *{n}{spec}
    while "*{" in s:
        m = re.search(r"\*\{(\d+)\}\{([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + m.group(2) * int(m.group(1)) + s[m.end():]
    while i < len(s):
        c = s[i]
        if c in " \t":
            i += 1
            continue
        if c == "|":
            if cols:
                cols[-1]["after"] += 1
            else:
                bars_before += 1
            i += 1
            continue
        if c in "@!<>":
            # @{...} >{...} <{...}: skip the group
            j = s.find("{", i)
            depth = 0
            k = j
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if c == "@" and cols:
                cols[-1]["at"] = s[j + 1:k]
            i = k + 1
            continue
        if c in "lcr":
            cols.append({"align": c, "after": 0, "kind": c})
            i += 1
            continue
        if c in "pmbX":
            if c == "X":
                if "tabularx" not in e.pkg_closure:
                    e.error("LaTeX Error: Illegal character in array arg.", t, "X columns need the tabularx package "
                                                                               "(\\begin{tabularx}).")
                cols.append({"align": "j", "after": 0, "kind": "X", "width": None})
                i += 1
                continue
            j = s.find("{", i)
            depth = 0
            k = j
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            w = s[j + 1:k]
            if dim_pt(w) is None:
                e.error("Missing number, treated as zero.", t, f"p{{width}} needs a length: p{{4cm}}, "
                                                               f"p{{0.3\\linewidth}} ...")
            cols.append({"align": "j", "after": 0, "kind": c, "width": w, "wpt": dim_pt(w)})
            i = k + 1
            continue
        if c == "S":
            if "siunitx" not in e.pkg_closure:
                e.error("LaTeX Error: Illegal character in array arg.", t, "S columns need \\usepackage{siunitx}.")
            opt = {}
            if i + 1 < len(s) and s[i + 1] == "[":
                k = s.find("]", i)
                opt = e.kv(_toks_of(s[i + 2:k]))
                i = k + 1
            else:
                i += 1
            fmt = opt.get("table-format", "")
            cols.append({"align": "S", "after": 0, "kind": "S", "format": fmt if isinstance(fmt, str) else ""})
            continue
        e.error("LaTeX Error: Illegal character in array arg.", t,
                f"`{c}` isn't a column type: use l, c, r, p{{width}} or | (and S with siunitx, X with tabularx).")
        i += 1
    return cols, bars_before


def _toks_of(s):
    from .tokens import tokenize
    return tokenize(s, "spec")


def split_cells(toks: list[Tok]):
    """A row's cells at top-level & (nested tabulars and braces don't count)."""
    cells, cur = [], []
    depth = env_depth = 0
    for t in toks:
        if t.k == "bg":
            depth += 1
        elif t.k == "eg":
            depth -= 1
        elif t.k == "cs" and t.v in ("begin", "end"):
            env_depth += 1 if t.v == "begin" else -1
        if t.k == "tab" and depth == 0 and env_depth == 0:
            cells.append(cur)
            cur = []
            continue
        cur.append(t)
    cells.append(cur)
    return cells


def _take_group(toks, i):
    while i < len(toks) and toks[i].k == "sp":
        i += 1
    if i < len(toks) and toks[i].k == "bg":
        depth = 0
        k = i
        while k < len(toks):
            if toks[k].k == "bg":
                depth += 1
            elif toks[k].k == "eg":
                depth -= 1
                if depth == 0:
                    return toks[i + 1:k], k + 1
            k += 1
        return toks[i + 1:], len(toks)
    if i < len(toks):
        return [toks[i]], i + 1
    return [], i


def _take_opt(toks, i, open_="[", close="]"):
    j = i
    while j < len(toks) and toks[j].k == "sp":
        j += 1
    if j < len(toks) and toks[j].k == "ch" and toks[j].v == open_:
        k = j + 1
        while k < len(toks) and not (toks[k].k == "ch" and toks[k].v == close):
            k += 1
        return toks[j + 1:k], k + 1
    return None, i


@env("tabular", "tabular*", "tabularx", "longtable")
def tabular_env(e, name, t):
    width = None
    if name in ("tabular*", "tabularx"):
        width = e.arg_text(name)
    pos = e.read_opt()
    spec = e.read_arg(name)
    cols, bars_before = parse_colspec(e, spec, t)
    body = e.env_body(name, t)
    rows = parse_rows(e, body, cols, t)
    box = {"t": "tabular", "env": name, "cols": cols, "bars_before": bars_before, "rows": rows, "width": width,
           "booktabs": any(r.get("rules") and any(x[0] in ("top", "mid", "bottom", "cmid") for x in r["rules"])
                           for r in rows)}
    if name == "longtable":
        e.add_block({"t": "group", "align": "center", "c": [{"t": "par", "c": [{"t": "box", "b": box}],
                                                            "align": "c", "indent": False}]})
    else:
        e.add_inline({"t": "box", "b": box})


def parse_rows(e, body, cols, t):
    from .features import split_rows
    raw_rows = split_rows(body)
    rows = []
    ncols = len(cols)
    for ri, rt in enumerate(raw_rows):
        rules = []
        i = 0
        # rules at the start of the row
        while True:
            while i < len(rt) and rt[i].k in ("sp", "par"):
                i += 1
            if i < len(rt) and rt[i].k == "cs" and rt[i].v in RULE_CMDS:
                c = rt[i].v
                e.used["\\" + c] += 1
                if c in ("toprule", "midrule", "bottomrule", "cmidrule", "addlinespace", "specialrule") and \
                        "booktabs" not in e.pkg_closure:
                    e.error("Undefined control sequence.", rt[i], f"\\{c} comes from the booktabs package: "
                                                                 f"\\usepackage{{booktabs}}.")
                    i += 1
                    continue
                i += 1
                span = None
                if c in ("toprule", "midrule", "bottomrule", "addlinespace"):
                    _, i = _take_opt(rt, i)
                if c == "cmidrule":
                    _, i = _take_opt(rt, i, "(", ")")
                    _, i = _take_opt(rt, i)
                    g, i = _take_group(rt, i)
                    span = detokenize(g)
                if c == "cline":
                    g, i = _take_group(rt, i)
                    span = detokenize(g)
                if c == "specialrule":
                    for _ in range(3):
                        _, i = _take_group(rt, i)
                rules.append((RULE_CMDS[c], span))
                continue
            break
        rest = rt[i:]
        if not any(x.k not in ("sp", "par") for x in rest):
            if rules:
                rows.append({"cells": None, "rules": rules})
            continue
        cells_t = split_cells(rest)
        cells = []
        col = 0
        for ci, ct in enumerate(cells_t):
            j = 0
            while j < len(ct) and ct[j].k == "sp":
                j += 1
            span, align, rowspan = 1, None, 1
            content = ct
            if j < len(ct) and ct[j].k == "cs" and ct[j].v == "multicolumn":
                e.used["\\multicolumn"] += 1
                n_t, k = _take_group(ct, j + 1)
                spec_t, k = _take_group(ct, k)
                content, k = _take_group(ct, k)
                try:
                    span = int(detokenize(n_t).strip())
                except ValueError:
                    e.error("Missing number, treated as zero.", ct[j], "\\multicolumn{n}{spec}{text}.")
                    span = 1
                sub, _ = parse_colspec(e, spec_t, ct[j])
                align = sub[0]["align"] if sub else "c"
            elif j < len(ct) and ct[j].k == "cs" and ct[j].v == "multirow":
                e.used["\\multirow"] += 1
                if "multirow" not in e.pkg_closure:
                    e.error("Undefined control sequence.", ct[j], "\\multirow comes from \\usepackage{multirow}.")
                n_t, k = _take_group(ct, j + 1)
                _, k = _take_group(ct, k)
                content, k = _take_group(ct, k)
                try:
                    rowspan = int(detokenize(n_t).strip())
                except ValueError:
                    rowspan = 1
            if col + span > ncols:
                e.error("Extra alignment tab has been changed to \\cr.", _first(ct) or t,
                        f"This row has more cells than the {ncols} columns the tabular declares.")
                break
            kind = cols[col]["kind"] if col < ncols else "l"
            if kind == "S" and align is None:
                txt = detokenize(content).strip()
                if txt.startswith("{") and txt.endswith("}"):
                    inl = e.capture(content)
                    cells.append({"c": inl, "span": span, "align": "c", "rowspan": rowspan, "num": None})
                elif re.fullmatch(r"-?[\d.]+(e-?\d+)?", txt):
                    cells.append({"c": [{"t": "text", "s": txt, "st": e.style}], "span": 1, "align": "S",
                                  "rowspan": rowspan, "num": txt})
                else:
                    if txt:
                        e.error(f"Package siunitx Error: Invalid number '{txt}'.", _first(ct) or t,
                                "Text in an S column goes in braces: {Loss}.")
                    cells.append({"c": e.capture(content), "span": 1, "align": "c", "rowspan": rowspan, "num": None})
            elif kind in ("p", "m", "b", "X") and align is None:
                blocks = e.capture_blocks(content, kind="cell", align="j")
                cells.append({"blocks": blocks, "span": span, "align": "j", "rowspan": rowspan})
            else:
                cells.append({"c": e.capture(content), "span": span,
                              "align": align or (cols[col]["align"] if col < ncols else "l"), "rowspan": rowspan})
            col += span
        rows.append({"cells": cells, "rules": rules})
    return rows


def _first(toks):
    return next((x for x in toks if x.k != "sp"), None)

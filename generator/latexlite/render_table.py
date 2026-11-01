"""tabular -> an HTML table with LaTeX's rules: | column bars, \\hline / \\cline, booktabs' thicker top and bottom
rules and padded \\midrule / \\cmidrule."""

from __future__ import annotations

import re


def _span(spec, ncols):
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)", spec or "")
    if not m:
        return set(range(ncols))
    return set(range(int(m.group(1)) - 1, int(m.group(2))))


def tabular(r, b) -> str:
    cols = b["cols"]
    n = len(cols)
    rows = b["rows"]
    out = ['<table class="tex-tabular' + (" tex-booktabs" if b.get("booktabs") else "") + '">']
    pending = []           # rules waiting for the next row (their border goes on its top)
    body_rows = []
    for row in rows:
        if row.get("cells") is None:
            pending += row["rules"]
            continue
        body_rows.append((pending + row["rules"], row))
        pending = []
    trailing = pending
    # S columns: align numbers on the decimal point (pad the integer and fraction parts)
    widths = {}
    for _, row in body_rows:
        col = 0
        for c in row["cells"]:
            if c.get("num") is not None:
                ip, _, fp = c["num"].partition(".")
                w = widths.get(col, (0, 0))
                widths[col] = (max(w[0], len(ip)), max(w[1], len(fp)))
            col += c.get("span", 1)
    covered = {}           # column -> rows still under a \multirow from above (HTML needs those cells left out)
    for ri, (rules, row) in enumerate(body_rows):
        tds = []
        col = 0
        for c in row["cells"]:
            span = c.get("span", 1)
            if covered.get(col) and _empty(c):
                col += span
                continue
            cls = []
            style = []
            a = c.get("align", "l")
            cls.append({"l": "tex-al", "c": "tex-ac", "r": "tex-ar", "j": "tex-aj", "S": "tex-ar"}.get(a, "tex-al"))
            first = col == 0
            last = col + span >= n
            bars_l = b["bars_before"] if first else 0
            bars_r = cols[min(col + span - 1, n - 1)]["after"] if n else 0
            if bars_l:
                style.append("border-left:" + ("double 1.6pt" if bars_l > 1 else "0.4pt solid") + " currentColor")
            if bars_r:
                style.append("border-right:" + ("double 1.6pt" if bars_r > 1 else "0.4pt solid") + " currentColor")
            top = _rule_for(rules, col, span, n)
            if top:
                style.append(top)
            if ri + c.get("rowspan", 1) - 1 >= len(body_rows) - 1 and trailing:
                bot = _rule_for(trailing, col, span, n, bottom=True)
                if bot:
                    style.append(bot)
            if col < n and cols[col].get("wpt") and span == 1:
                style.append(f"width:{cols[col]['wpt']:.1f}pt")
            if "blocks" in c:
                inner = r.blocks(c["blocks"])
            else:
                inner = r.inl(c["c"])
            if c.get("num") is not None and col in widths:
                ip, _, fp = c["num"].partition(".")
                wi, wf = widths[col]
                inner = (f'<span class="tex-num"><span style="display:inline-block;min-width:{wi * 0.5:.2f}em;'
                         f'text-align:right">{ip}</span>{"." if fp or wf else ""}<span style="display:inline-block;'
                         f'min-width:{wf * 0.5:.2f}em;text-align:left">{fp}</span></span>')
            attrs = f' colspan="{span}"' if span > 1 else ""
            if c.get("rowspan", 1) > 1:
                attrs += f' rowspan="{c["rowspan"]}"'
                style.append("vertical-align:middle")
                for k in range(col, col + span):
                    covered[k] = c["rowspan"]
            tds.append(f'<td class="{" ".join(cls)}" style="{";".join(style)}"{attrs}>{inner}</td>')
            col += span
        while col < n:
            if not covered.get(col):
                tds.append('<td></td>')
            col += 1
        for k in list(covered):
            covered[k] -= 1
            if covered[k] <= 0:
                del covered[k]
        out.append("<tr>" + "".join(tds) + "</tr>")
    if not body_rows and trailing:
        out.append(f'<tr><td colspan="{n}" style="{_rule_for(trailing, 0, n, n)}"></td></tr>')
    out.append("</table>")
    return "".join(out)


def _empty(c):
    if "blocks" in c:
        return not c["blocks"]
    return all(x.get("t") == "space" or (x.get("t") == "text" and not x.get("s", "").strip()) for x in c["c"])


def _rule_for(rules, col, span, n, bottom=False):
    side = "bottom" if bottom else "top"
    width = 0.0
    pad = False
    for kind, spec in rules:
        if kind in ("hline",):
            width += 0.4 if width == 0 else 2.4
        elif kind in ("top", "bottom"):
            width = max(width, 0.8)
            pad = True
        elif kind == "mid":
            width = max(width, 0.5)
            pad = True
        elif kind in ("cline", "cmid"):
            cs = _span(spec, n)
            if set(range(col, col + span)) & cs:
                width = max(width, 0.4 if kind == "cline" else 0.5)
                pad = kind == "cmid" or pad
        elif kind == "special":
            width = max(width, 1.0)
    if not width:
        return ""
    s = f"border-{side}:{width:.1f}pt {'double' if width > 2 else 'solid'} currentColor"
    if pad:
        s += f";padding-{side}:3pt"
    return s

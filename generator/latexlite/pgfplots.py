"""pgfplots: \\begin{axis}[...] \\addplot ... \\end{axis} -> SVG, and the data the grader compares.

\\addplot coordinates {...}, \\addplot {expression} (domain, samples), \\addplot table [x=, y=, col sep=comma]
{file.csv}; legends (\\addlegendentry, legend={...}, legend pos), labels, title, log axes (ymode/xmode=log, semilogy,
loglog), grids, xmin/xmax/ymin/ymax, width/height, marks and line styles; pgfplots' default cycle list when
\\addplot has no [options] (or uses \\addplot+).
"""

from __future__ import annotations

import html
import math
import re

from .commands import color_of, dim_pt
from .tikz import TikzError, _split_opts, evaluate

CYCLE = [("#0000ff", "*"), ("#ff0000", "square*"), ("#996633", "otimes*"), ("#000000", "star"),
         ("#0000ff", "diamond*"), ("#ff0000", "triangle*")]
MARKS = {"*", "o", "x", "+", "square", "square*", "triangle", "triangle*", "diamond", "diamond*", "otimes",
         "otimes*", "star", "pentagon*", "|", "-", "none"}


def extract_axes(e, src, t):
    """Take the axis environments out of a tikzpicture's source. -> (the rest, [plot boxes])."""
    axes = []
    pat = re.compile(r"\\begin\{(axis|semilogyaxis|semilogxaxis|loglogaxis)\}(.*?)\\end\{\1\}", re.S)

    def take(m):
        if "pgfplots" not in e.pkg_closure:
            e.error(f"LaTeX Error: Environment {m.group(1)} undefined.", t,
                    "axis comes from the pgfplots package: \\usepackage{pgfplots}.")
            return ""
        e.used["env:" + m.group(1)] += 1
        axes.append(axis_box(e, m.group(1), m.group(2), t))
        return ""
    rest = pat.sub(take, src)
    return rest, axes


def _opts_and_body(s):
    s = s.lstrip()
    if s.startswith("["):
        depth = 0
        for i, c in enumerate(s):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return s[1:i], s[i + 1:]
    return "", s


def axis_box(e, kind, src, t):
    opt_s, body = _opts_and_body(src)
    o = dict(e.pgfplots_settings)
    for k, v in _split_opts(opt_s):
        o[k] = v
    if kind == "semilogyaxis":
        o.setdefault("ymode", "log")
    if kind == "semilogxaxis":
        o.setdefault("xmode", "log")
    if kind == "loglogaxis":
        o.setdefault("xmode", "log")
        o.setdefault("ymode", "log")
    plots = []
    legend = []
    n_cycle = 0
    stmts = _stmts(body)
    for st in stmts:
        st = st.strip()
        if not st:
            continue
        m = re.match(r"\\addplot(\+?)\s*(?:\[([^\]]*)\])?\s*(.*)$", st, re.S)
        if m:
            e.used["\\addplot"] += 1
            plus, popt, rest = m.group(1), m.group(2), m.group(3).strip()
            pairs = _split_opts(popt or "")
            use_cycle = plus == "+" or popt is None
            style = {"color": None, "mark": None, "dash": None, "lw": 0.4, "only_marks": False, "smooth": False}
            if use_cycle:
                c, mk = CYCLE[n_cycle % len(CYCLE)]
                style["color"], style["mark"] = c, mk
                n_cycle += 1
            for k, v in pairs:
                if k == "mark":
                    style["mark"] = v
                    if v not in MARKS:
                        e.error(f"Package pgfplots Error: Sorry, I could not find a mark named `{v}'.", t,
                                "Marks: *, o, x, +, square*, triangle*, diamond* ...")
                elif k in ("dashed", "dotted", "densely dashed", "densely dotted", "loosely dashed"):
                    style["dash"] = k
                elif k == "solid":
                    style["dash"] = None
                elif k in ("thick", "very thick", "ultra thick", "thin", "semithick"):
                    style["lw"] = {"thin": 0.4, "semithick": 0.6, "thick": 0.8, "very thick": 1.2,
                                   "ultra thick": 1.6}[k]
                elif k == "line width":
                    style["lw"] = dim_pt(v or "") or 0.4
                elif k == "only marks":
                    style["only_marks"] = True
                elif k == "smooth":
                    style["smooth"] = True
                elif k in ("domain", "samples", "color", "draw", "mark options", "forget plot", "fill", "opacity",
                           "unbounded coords", "y filter/.code", "error bars/.cd", "ybar", "xbar", "bar width",
                           "nodes near coords", "sharp plot", "const plot", "no markers", "very thin"):
                    if k in ("color", "draw") and v:
                        style["color"] = color_of(e, v)
                    if k == "no markers":
                        style["mark"] = None
                elif v is None and color_of(e, k) is not None:
                    style["color"] = color_of(e, k)
                elif v is None:
                    e.error(f"Package pgfkeys Error: I do not know the key '/tikz/{k}' and I am going to ignore it.",
                            t, f"`{k}` isn't a plot option.")
            if style["color"] is None:
                style["color"] = "#000000"
            if not use_cycle and style["mark"] is None and not any(k == "mark" for k, _ in pairs):
                style["mark"] = None
            po = dict(pairs)
            pts = _plot_points(e, rest, po, o, t)
            plots.append({"pts": pts, "style": style})
            continue
        m = re.match(r"\\addlegendentry\s*\{(.*)\}\s*$", st, re.S)
        if m:
            e.used["\\addlegendentry"] += 1
            legend.append(m.group(1))
            continue
        m = re.match(r"\\legend\s*\{(.*)\}\s*$", st, re.S)
        if m:
            legend = [x.strip() for x in _split_top_commas(m.group(1))]
            continue
        e.error("Package pgfplots Error: Sorry, I could not read this." if st.startswith("\\") else
                "Package tikz Error: Giving up on this path. Did you forget a semicolon?", t,
                "Inside an axis: \\addplot ... ; and \\addlegendentry{...}.")
    if isinstance(o.get("legend entries"), str):
        legend = [x.strip() for x in _split_top_commas(o["legend entries"])]
    svg, data = draw_axis(e, o, plots, legend)
    return {"t": "plot", "svg": svg, "data": data}


def _split_top_commas(s):
    out, cur, depth = [], [], 0
    for c in s:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return [x.strip().strip("{}") for x in out if x.strip()]


def _stmts(body):
    out, cur, depth = [], [], 0
    for c in body:
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        if c == ";" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        cur_s = "".join(cur).strip()
        if cur_s.startswith("\\addlegendentry") or cur_s.startswith("\\legend"):
            out.append(cur_s)
        else:
            out.append(cur_s + "\x00")
    # \addlegendentry{...} has no semicolon: split it off what follows
    res = []
    for s in out:
        parts = re.split(r"(?=\\addlegendentry|\\legend\b|\\addplot)", s)
        res += [p.replace("\x00", "") for p in parts if p.strip()]
    return res


def _plot_points(e, rest, po, o, t):
    rest = rest.strip()
    if rest.startswith("coordinates"):
        m = re.match(r"coordinates\s*\{(.*)\}\s*$", rest, re.S)
        if not m:
            e.error("Package pgfplots Error: coordinates needs {(x,y) (x,y) ...}.", t, "")
            return []
        pts = []
        for a, b in re.findall(r"\(\s*([^,()]+)\s*,\s*([^()]+?)\s*\)", m.group(1)):
            try:
                pts.append((float(a), float(b)))
            except ValueError:
                e.error(f"Package PGF Math Error: Could not parse input '{a}' as a floating point number, sorry.",
                        t, "Coordinates are numbers: (1, 2.3).")
        return pts
    if rest.startswith("table"):
        m = re.match(r"table\s*(?:\[([^\]]*)\])?\s*\{(.*)\}\s*$", rest, re.S)
        if not m:
            return []
        to = dict(_split_opts(m.group(1) or ""))
        fname = m.group(2).strip()
        src = e.files.get(fname)
        if src is None:
            if "\n" in fname or " " in fname.strip():
                src = fname
            else:
                e.error(f"Package pgfplots Error: Could not read table file '{fname}'.", t,
                        f"There's no file {fname} next to main.tex.")
                return []
        sep = "," if to.get("col sep") == "comma" else None
        rows = [ln for ln in src.strip().split("\n") if ln.strip()]
        header = [h.strip() for h in (rows[0].split(sep) if sep else rows[0].split())]
        xc = to.get("x", header[0])
        yc = to.get("y", header[1] if len(header) > 1 else header[0])
        if xc not in header or yc not in header:
            bad = xc if xc not in header else yc
            e.error(f"Package pgfplots Error: Table '{fname}' has no column named '{bad}'.", t,
                    f"The columns are: {', '.join(header)}.")
            return []
        xi, yi = header.index(xc), header.index(yc)
        pts = []
        for r in rows[1:]:
            cells = [c.strip() for c in (r.split(sep) if sep else r.split())]
            try:
                pts.append((float(cells[xi]), float(cells[yi])))
            except (ValueError, IndexError):
                pass
        return pts
    # an expression {f(x)}
    m = re.match(r"\{(.*)\}\s*$", rest, re.S) or re.match(r"(.*)$", rest, re.S)
    expr = m.group(1).strip()
    dom = po.get("domain") or o.get("domain") or "-5:5"
    try:
        a, b = (float(evaluate(x)) for x in str(dom).split(":"))
    except (TikzError, ValueError):
        a, b = -5.0, 5.0
    try:
        n = int(float(po.get("samples") or o.get("samples") or 25))
    except ValueError:
        n = 25
    pts = []
    for i in range(max(2, n)):
        x = a + (b - a) * i / (max(2, n) - 1)
        try:
            y = evaluate(expr.replace("\\x", "x"), {"x": x})
            if math.isfinite(y):
                pts.append((x, y))
        except (TikzError, ValueError, ZeroDivisionError, OverflowError):
            pass
        except Exception:  # noqa: BLE001
            pass
    if not pts:
        e.error(f"Package PGF Math Error: Unknown function or operand in '{expr}'.", t,
                "Write the expression with x: {2.5*exp(-0.5*x) + 0.3}. (pgf's trig functions use degrees.)")
    return pts


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n + 1:
            break
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    x = start
    while x <= hi + 1e-9 * span:
        ticks.append(round(x, 10))
        x += step
    return ticks, step


def _fmt(v):
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def draw_axis(e, o, plots, legend):
    W = dim_pt(o.get("width") or "") or 240.0
    H = dim_pt(o.get("height") or "") or 207.0
    logx, logy = o.get("xmode") == "log", o.get("ymode") == "log"
    xs = [p[0] for pl in plots for p in pl["pts"]]
    ys = [p[1] for pl in plots for p in pl["pts"]]
    if logx:
        xs = [x for x in xs if x > 0]
    if logy:
        ys = [y for y in ys if y > 0]
    if not xs:
        xs, ys = [0, 1], [0, 1]
    tx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    ty = (lambda v: math.log10(v)) if logy else (lambda v: v)
    def opt_f(k):
        try:
            return float(o[k]) if k in o and o[k] is not None else None
        except ValueError:
            return None
    x0 = opt_f("xmin") if opt_f("xmin") is not None else min(xs)
    x1 = opt_f("xmax") if opt_f("xmax") is not None else max(xs)
    y0 = opt_f("ymin") if opt_f("ymin") is not None else min(ys)
    y1 = opt_f("ymax") if opt_f("ymax") is not None else max(ys)
    X0, X1, Y0, Y1 = tx(x0), tx(x1), ty(y0), ty(y1)
    if X1 == X0:
        X0, X1 = X0 - 1, X1 + 1
    if Y1 == Y0:
        Y0, Y1 = Y0 - 1, Y1 + 1
    # pgfplots pads the data range a little (enlargelimits) only for some plots; keep exact limits
    if logy:
        yt = [float(k) for k in range(math.floor(Y0), math.ceil(Y1) + 1)]
        Y0, Y1 = min(Y0, yt[0]), max(Y1, yt[-1])
    else:
        yt, _ = nice_ticks(Y0, Y1)
    if logx:
        xt = [float(k) for k in range(math.floor(X0), math.ceil(X1) + 1)]
        X0, X1 = min(X0, xt[0]), max(X1, xt[-1])
    else:
        xt, _ = nice_ticks(X0, X1)
    if isinstance(o.get("xtick"), str) and o["xtick"].strip() not in ("\\empty", "{}", ""):
        try:
            xt = [tx(float(evaluate(v))) for v in _split_top_commas(o["xtick"].strip("{}"))]
        except TikzError:
            pass
    if isinstance(o.get("ytick"), str) and o["ytick"].strip() not in ("\\empty", "{}", ""):
        try:
            yt = [ty(float(evaluate(v))) for v in _split_top_commas(o["ytick"].strip("{}"))]
        except TikzError:
            pass
    L, R, T, B = 40.0, 10.0, 18.0 if o.get("title") else 8.0, 30.0
    pw, ph = W - 45, H - 45
    def P(x, y):
        return (L + (x - X0) / (X1 - X0) * pw, T + (Y1 - y) / (Y1 - Y0) * ph)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}pt" height="{H:.1f}pt" viewBox="0 0 {W:.1f} {H:.1f}" '
           f'font-size="8" font-family="KaTeX_Main, serif">']
    out.append(f'<clipPath id="pc{id(o)}"><rect x="{L}" y="{T}" width="{pw:.1f}" height="{ph:.1f}"/></clipPath>')
    grid = o.get("grid")
    for v in xt:
        x, _ = P(v, Y0)
        if grid in ("major", "both", True) or o.get("xmajorgrids") is not None:
            out.append(f'<line x1="{x:.2f}" y1="{T}" x2="{x:.2f}" y2="{T + ph:.1f}" stroke="#ccc" stroke-width="0.4"/>')
        out.append(f'<line x1="{x:.2f}" y1="{T + ph:.1f}" x2="{x:.2f}" y2="{T + ph - 3:.1f}" stroke="#000" '
                   f'stroke-width="0.4"/>')
        out.append(f'<text x="{x:.2f}" y="{T + ph + 10:.1f}" text-anchor="middle">{_tick_label(v, logx)}</text>')
    for v in yt:
        _, y = P(X0, v)
        if grid in ("major", "both", True) or o.get("ymajorgrids") is not None:
            out.append(f'<line x1="{L}" y1="{y:.2f}" x2="{L + pw:.1f}" y2="{y:.2f}" stroke="#ccc" stroke-width="0.4"/>')
        out.append(f'<line x1="{L}" y1="{y:.2f}" x2="{L + 3}" y2="{y:.2f}" stroke="#000" stroke-width="0.4"/>')
        out.append(f'<text x="{L - 3:.1f}" y="{y + 3:.2f}" text-anchor="end">{_tick_label(v, logy)}</text>')
    out.append(f'<rect x="{L}" y="{T}" width="{pw:.1f}" height="{ph:.1f}" fill="none" stroke="#000" '
               f'stroke-width="0.4"/>')
    data = []
    for pl in plots:
        st = pl["style"]
        pts = [(tx(x), ty(y)) for x, y in pl["pts"] if (not logx or x > 0) and (not logy or y > 0)]
        data.append({"pts": [[round(x, 4), round(y, 4)] for x, y in pl["pts"]], "color": st["color"],
                     "mark": st["mark"], "dash": st["dash"]})
        if not pts:
            continue
        sp = [P(x, y) for x, y in pts]
        if not st["only_marks"]:
            dash = ' stroke-dasharray="3,2"' if st["dash"] and "dash" in st["dash"] else \
                ' stroke-dasharray="0.6,1.8" stroke-linecap="round"' if st["dash"] else ""
            pts_s = " ".join(f"{x:.2f},{y:.2f}" for x, y in sp)
            out.append(f'<polyline points="{pts_s}" fill="none" stroke="{st["color"]}" stroke-width="{st["lw"]}"'
                       f'{dash} clip-path="url(#pc{id(o)})"/>')
        if st["mark"] and st["mark"] != "none":
            for x, y in sp:
                out.append(_mark(st["mark"], x, y, st["color"]))
    if o.get("title"):
        out.append(f'<text x="{L + pw / 2:.1f}" y="10" text-anchor="middle" font-size="9">{_txt(o["title"])}</text>')
    if o.get("xlabel"):
        out.append(f'<text x="{L + pw / 2:.1f}" y="{H - 4:.1f}" text-anchor="middle" font-size="9">'
                   f'{_txt(o["xlabel"])}</text>')
    if o.get("ylabel"):
        out.append(f'<text x="10" y="{T + ph / 2:.1f}" text-anchor="middle" font-size="9" transform="rotate(-90 10 '
                   f'{T + ph / 2:.1f})">{_txt(o["ylabel"])}</text>')
    if legend:
        pos = str(o.get("legend pos", "north east"))
        lw = max(len(_plain(x)) for x in legend) * 4.4 + 26
        lh = len(legend) * 11 + 6
        lx = L + pw - lw - 4 if "east" in pos else L + 4
        ly = T + 4 if "north" in pos else T + ph - lh - 4
        if pos == "outer north east":
            lx, ly = L + pw + 4, T
        out.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="{lh:.1f}" fill="#fff" stroke="#000" '
                   f'stroke-width="0.4"/>')
        for i, entry in enumerate(legend):
            yy = ly + 9 + i * 11
            if i < len(plots):
                st = plots[i]["style"]
                if not st["only_marks"]:
                    dash = ' stroke-dasharray="3,2"' if st["dash"] and "dash" in st["dash"] else ""
                    out.append(f'<line x1="{lx + 4:.1f}" y1="{yy - 3:.1f}" x2="{lx + 18:.1f}" y2="{yy - 3:.1f}" '
                               f'stroke="{st["color"]}" stroke-width="{st["lw"]}"{dash}/>')
                if st["mark"] and st["mark"] != "none":
                    out.append(_mark(st["mark"], lx + 11, yy - 3, st["color"]))
            out.append(f'<text x="{lx + 22:.1f}" y="{yy:.1f}">{_txt(entry)}</text>')
    out.append("</svg>")
    info = {"plots": data, "legend": [_plain(x) for x in legend], "xlabel": _plain(o.get("xlabel") or ""),
            "ylabel": _plain(o.get("ylabel") or ""), "title": _plain(o.get("title") or ""),
            "xmode": "log" if logx else "linear", "ymode": "log" if logy else "linear",
            "grid": bool(grid), "legend_pos": str(o.get("legend pos", "")) if legend else ""}
    return "".join(out), info


def _plain(s):
    s = str(s or "").strip().strip("{}")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    return re.sub(r"\\[A-Za-z]+\s*", "", s).replace("{", "").replace("}", "").strip()


def _txt(s):
    """Axis text: $math$ simplified (^ -> superscript)."""
    s = str(s or "").strip().strip("{}")
    out = []
    for part in re.split(r"(\$[^$]*\$)", s):
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            m = part[1:-1]
            m = re.sub(r"\\([A-Za-z]+)", lambda k: {"alpha": "α", "beta": "β", "eta": "η", "theta": "θ",
                                                    "lambda": "λ", "sigma": "σ", "mu": "μ", "epsilon": "ε",
                                                    "times": "×", "log": "log"}.get(k.group(1), k.group(1)), m)
            m = html.escape(m)
            m = re.sub(r"\^\{?([^{}]*)\}?", r'<tspan baseline-shift="super" font-size="6">\1</tspan>', m)
            m = re.sub(r"_\{?([^{}]*)\}?", r'<tspan baseline-shift="sub" font-size="6">\1</tspan>', m)
            out.append(f'<tspan font-style="italic">{m}</tspan>')
        else:
            out.append(html.escape(re.sub(r"\\[A-Za-z]+\s*", "", part).replace("{", "").replace("}", "")))
    return "".join(out)


def _tick_label(v, log):
    if log:
        k = int(round(v))
        return f'10<tspan baseline-shift="super" font-size="6">{k}</tspan>'
    return html.escape(_fmt(v))


def _mark(m, x, y, color):
    fill = color if m.endswith("*") else "#fff"
    base = m.rstrip("*")
    r = 2.0
    if base in ("", "o"):
        f = color if m == "*" else "#fff"
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{f}" stroke="{color}" stroke-width="0.4"/>'
    if base == "square":
        return f'<rect x="{x - r:.2f}" y="{y - r:.2f}" width="{2 * r}" height="{2 * r}" fill="{fill}" ' \
               f'stroke="{color}" stroke-width="0.4"/>'
    if base == "triangle":
        return f'<polygon points="{x:.2f},{y - r * 1.2:.2f} {x - r:.2f},{y + r * 0.8:.2f} {x + r:.2f},' \
               f'{y + r * 0.8:.2f}" fill="{fill}" stroke="{color}" stroke-width="0.4"/>'
    if base == "diamond":
        return f'<polygon points="{x:.2f},{y - r * 1.3:.2f} {x + r:.2f},{y:.2f} {x:.2f},{y + r * 1.3:.2f} ' \
               f'{x - r:.2f},{y:.2f}" fill="{fill}" stroke="{color}" stroke-width="0.4"/>'
    if base in ("x", "+", "star", "otimes", "|", "-"):
        if base == "otimes":
            return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="0.4"/>' \
                   f'<path d="M{x - 1.4:.2f},{y - 1.4:.2f}L{x + 1.4:.2f},{y + 1.4:.2f}M{x + 1.4:.2f},{y - 1.4:.2f}' \
                   f'L{x - 1.4:.2f},{y + 1.4:.2f}" stroke="#fff" stroke-width="0.4"/>'
        d = {"x": f"M{x - r:.2f},{y - r:.2f}L{x + r:.2f},{y + r:.2f}M{x + r:.2f},{y - r:.2f}L{x - r:.2f},{y + r:.2f}",
             "+": f"M{x - r:.2f},{y:.2f}L{x + r:.2f},{y:.2f}M{x:.2f},{y - r:.2f}L{x:.2f},{y + r:.2f}",
             "star": f"M{x - r:.2f},{y:.2f}L{x + r:.2f},{y:.2f}M{x - r * .6:.2f},{y - r * .8:.2f}L{x + r * .6:.2f},"
                     f"{y + r * .8:.2f}M{x + r * .6:.2f},{y - r * .8:.2f}L{x - r * .6:.2f},{y + r * .8:.2f}",
             "|": f"M{x:.2f},{y - r:.2f}L{x:.2f},{y + r:.2f}", "-": f"M{x - r:.2f},{y:.2f}L{x + r:.2f},{y:.2f}"}[base]
        return f'<path d="{d}" stroke="{color}" stroke-width="0.5"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="0.4"/>'

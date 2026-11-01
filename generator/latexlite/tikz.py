"""TikZ: tikzpicture -> SVG, and a scene (nodes and paths) the grader compares.

Covers what the guide uses and its near neighbours: \\node (shapes rectangle/circle/ellipse, draw, fill, minimum
size, rounded corners, positioning `right=1.5cm of a`), \\draw / \\fill / \\filldraw / \\path with -- -| |- rectangle
circle ellipse arc grid `to` and `.. controls ..`, nodes along paths (above, below, midway, pos), arrows (->, <-, <->,
>={Stealth}), line widths, dashed/dotted, colors (xcolor mixes), styles (`name/.style={...}`), \\foreach (lists,
ranges, x/y pairs, nested) and coordinate arithmetic. Lengths are TikZ's: 1 unit = 1cm, y upwards.
"""

from __future__ import annotations

import html
import math
import re

from .commands import color_of, dim_pt
from .engine import cmd, env
from .tokens import Tok, detokenize, tokenize

CM = 28.4528
LINE_WIDTHS = {"ultra thin": 0.1, "very thin": 0.2, "thin": 0.4, "semithick": 0.6, "thick": 0.8,
               "very thick": 1.2, "ultra thick": 1.6}
ANCHORS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0), "north east": (1, 1),
           "north west": (-1, 1), "south east": (1, -1), "south west": (-1, -1), "center": (0, 0)}
DIRS = {"above": (0, 1), "below": (0, -1), "right": (1, 0), "left": (-1, 0), "above right": (1, 1),
        "above left": (-1, 1), "below right": (1, -1), "below left": (-1, -1)}


class TikzError(Exception):
    pass


def parse_styles(e, toks):
    for k, v in _split_opts(detokenize(toks)):
        if k.endswith("/.style"):
            e.tikz_styles[k[:-7].strip()] = v or ""


def _split_opts(s: str):
    """key=value pairs at top level (braces protect commas)."""
    out, cur, depth = [], [], 0
    for c in s:
        if c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    res = []
    for p in out:
        p = p.strip()
        if not p:
            continue
        if "=" in p and not p.startswith(("<", "-", ">")) or ("=" in p and p.startswith(">=")):
            k, v = p.split("=", 1)
            v = v.strip()
            if v.startswith("{") and v.endswith("}"):
                v = v[1:-1]
            res.append((k.strip(), v))
        else:
            res.append((p, None))
    return res


# ---------------------------------------------------------------- arithmetic (pgfmath, small)

def evaluate(expr: str, variables=None) -> float:
    """pgfmath: + - * / ^, parentheses, functions (trig in degrees, as pgf), pi, e, units (cm, pt, mm)."""
    s = expr.strip()
    variables = variables or {}
    toks = re.findall(r"\d*\.\d+|\d+\.?\d*(?:e[+-]?\d+)?|[A-Za-z_]\w*|\\[A-Za-z]+|[-+*/^(),]", s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = toks[pos[0]]
        pos[0] += 1
        return t

    def atom():
        t = peek()
        if t is None:
            raise TikzError(expr)
        if t == "(":
            take()
            v = add()
            if peek() == ")":
                take()
            return v
        if t == "-":
            take()
            return -power()
        if t == "+":
            take()
            return power()
        take()
        if re.fullmatch(r"\d*\.\d+|\d+\.?\d*(?:e[+-]?\d+)?", t):
            v = float(t)
            nxt = peek()
            if nxt in ("cm", "pt", "mm", "in", "em"):
                take()
                v = v * {"cm": 1, "pt": 1 / CM, "mm": 0.1, "in": 2.54, "em": 10 / CM}[nxt]
            return v
        name = t.lstrip("\\")
        if name in variables:
            return float(variables[name])
        if name == "pi":
            return math.pi
        if name == "e":
            return math.e
        if peek() == "(":
            take()
            args = [add()]
            while peek() == ",":
                take()
                args.append(add())
            if peek() == ")":
                take()
            f = {"exp": math.exp, "ln": math.log, "log10": math.log10, "log2": math.log2, "sqrt": math.sqrt,
                 "abs": abs, "sin": lambda x: math.sin(math.radians(x)), "cos": lambda x: math.cos(math.radians(x)),
                 "tan": lambda x: math.tan(math.radians(x)), "deg": math.degrees, "rad": math.radians,
                 "max": max, "min": min, "floor": math.floor, "ceil": math.ceil, "round": round,
                 "tanh": math.tanh, "atan": lambda x: math.degrees(math.atan(x)), "pow": math.pow,
                 "mod": math.fmod, "sinh": math.sinh, "cosh": math.cosh}.get(name)
            if f is None:
                raise TikzError(f"Unknown function `{name}'")
            return float(f(*args))
        raise TikzError(f"Unknown operand `{name}'")

    def power():
        v = atom()
        while peek() == "^":
            take()
            v = v ** atom()
        return v

    def mul():
        v = power()
        while peek() in ("*", "/"):
            op = take()
            r = power()
            v = v * r if op == "*" else v / r
        return v

    def add():
        v = mul()
        while peek() in ("+", "-"):
            op = take()
            r = mul()
            v = v + r if op == "+" else v - r
        return v
    v = add()
    if pos[0] != len(toks):
        raise TikzError(expr)
    return v


# ---------------------------------------------------------------- the picture

class Picture:
    def __init__(self, e, opts):
        self.e = e
        self.nodes = {}            # name -> node dict
        self.items = []            # drawing order: ("node", node) | ("path", path)
        self.styles = dict(e.tikz_styles)
        self.global_opts = []
        self.arrow_tip = "to"
        self.scale = 1.0
        self.x_unit = (1.0, 0.0)
        self.y_unit = (0.0, 1.0)
        self.errors = []
        self.anon = 0
        for k, v in _split_opts(opts):
            if k.endswith("/.style"):
                self.styles[k[:-7].strip()] = v or ""
            elif k == ">":
                self.arrow_tip = (v or "").strip("{}")
            elif k == "scale":
                try:
                    self.scale = float(v)
                except (TypeError, ValueError):
                    pass
            elif k == "x" and v:
                self.x_unit = (_len_cm(v), 0.0)
            elif k == "y" and v:
                self.y_unit = (0.0, _len_cm(v))
            else:
                self.global_opts.append((k, v))

    # ---------------------------------------------------- options -> style

    def style(self, pairs, base=None, depth=0):
        st = dict(base or {})
        for k, v in pairs:
            if depth < 8 and k in self.styles and v is None:
                st = self.style(_split_opts(self.styles[k]), st, depth + 1)
                continue
            if k in ("draw",) and v is None:
                st["draw"] = st.get("draw") or "#000000"
            elif k == "draw":
                st["draw"] = self.color(v)
            elif k == "fill":
                st["fill"] = self.color(v) if v else "#000000"
            elif k == "color":
                st["color"] = self.color(v)
                st["draw_color"] = st["color"]
            elif k in LINE_WIDTHS:
                st["lw"] = LINE_WIDTHS[k]
            elif k == "line width":
                st["lw"] = dim_pt(v or "") or 0.4
            elif k in ("dashed", "dotted", "densely dashed", "loosely dashed", "densely dotted"):
                st["dash"] = k
            elif k == "solid":
                st["dash"] = None
            elif v is None and (k == "|-|" or re.fullmatch(r"<?(\{?[A-Za-z]*\}?)-(\{?[A-Za-z]*\}?)>?", k.replace(" ", ""))):
                st["arrows"] = k
            elif k == "rounded corners":
                st["rounded"] = dim_pt(v) if v else 4.0
            elif k in ("circle", "rectangle", "ellipse", "diamond", "coordinate"):
                st["shape"] = k
            elif k in ("minimum size", "minimum width", "minimum height"):
                d = _len_pt(v)
                if k == "minimum size":
                    st["minw"] = st["minh"] = d
                elif k == "minimum width":
                    st["minw"] = d
                else:
                    st["minh"] = d
            elif k == "inner sep":
                st["inner"] = _len_pt(v)
            elif k == "text width":
                st["textw"] = _len_pt(v)
            elif k == "text":
                st["text"] = self.color(v)
            elif k in ("align",):
                st["align"] = v
            elif k == "font":
                st["font"] = v
            elif k == "anchor":
                st["anchor"] = v
            elif k in DIRS and (v is None or " of " not in (v or "")):
                st["place"] = (k, v)
            elif k in DIRS:
                st["of"] = (k, v)
            elif k in ("midway", "near start", "near end", "at start", "at end", "very near start",
                       "very near end"):
                st["pos"] = {"midway": 0.5, "near start": 0.25, "near end": 0.75, "at start": 0.0, "at end": 1.0,
                             "very near start": 0.125, "very near end": 0.875}[k]
            elif k == "pos":
                try:
                    st["pos"] = float(v)
                except (TypeError, ValueError):
                    pass
            elif k == "sloped":
                st["sloped"] = True
            elif k in ("opacity", "fill opacity", "draw opacity"):
                try:
                    st[k.replace(" ", "_")] = float(v)
                except (TypeError, ValueError):
                    pass
            elif k == "radius":
                st["radius"] = _len_cm(v)
            elif k == "scale":
                pass
            elif k in ("out", "in", "bend left", "bend right", "looseness", "loop", "step", "xstep", "ystep",
                       "shift", "xshift", "yshift", "label", "pin", "below of", "above of", "right of",
                       "left of", "node distance", "on grid", "every node/.style", "auto", "swap", "'",
                       "thick", "baseline", "transform shape", "rotate", "fill opacity", "decorate",
                       "decoration", "shading", "ball color", "top color", "bottom color", "left color",
                       "right color", "inner color", "outer color", "mark", "only marks", "smooth", "domain",
                       "samples", "variable", "nodes", "every path/.style", "help lines", "double", "xscale",
                       "yscale", "pattern", "trapezium", "text centered", "align", "font", "line cap",
                       "line join", "arrows", "outer sep", "join", "clip", "use as bounding box", "name",
                       "circle", "rectangle split", "shape", "level distance", "sibling distance", "grow",
                       "edge from parent", "child", "matrix", "column sep", "row sep", "every node",
                       "show background rectangle", "framed", "background rectangle/.style", "remember picture",
                       "overlay"):
                if k in ("bend left", "bend right"):
                    st["bend"] = (k, float(v) if v else 30.0)
                elif k == "help lines":
                    st["draw"] = "#bfbfbf"
                    st["lw"] = 0.2
                elif k == "step":
                    st["step"] = _len_cm(v) if v else 1.0
                elif k in ("xshift", "yshift"):
                    st[k] = _len_cm(v) if v else 0.0
                elif k == "label" and v:
                    st.setdefault("labels", []).append(v)
            elif v is None and self.color(k) is not None:
                st["color"] = self.color(k)
                st["draw_color"] = st["color"]
            else:
                self.err(f"Package pgfkeys Error: I do not know the key '/tikz/{k}' and I am going to ignore it.",
                         f"`{k}` isn't a TikZ option (check the spelling, or define it as a style).")
        return st

    def color(self, spec):
        if spec is None:
            return None
        return color_of(self.e, spec.strip())

    def err(self, msg, hlp=""):
        self.errors.append((msg, hlp))

    # ---------------------------------------------------- coordinates

    def coord(self, s: str, variables, last=(0.0, 0.0)):
        """'(x,y)' / '(a)' / '(a.north)' / '(30:2)' / '($(a)+(1,0)$)' (no calc) -> (x, y) in cm."""
        s = s.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        if s.startswith("$") and s.endswith("$"):
            # calc: ($(a)!0.5!(b)$) / ($(a)+(1,2)$)
            inner = s[1:-1]
            m = re.fullmatch(r"\s*\((.*?)\)\s*!\s*([\d.]+)\s*!\s*\((.*?)\)\s*", inner)
            if m:
                a = self.coord(m.group(1), variables)
                b = self.coord(m.group(3), variables)
                f = float(m.group(2))
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            parts = re.findall(r"([+-]?)\s*\(([^()]*)\)", inner)
            x = y = 0.0
            for sign, p in parts:
                c = self.coord(p, variables)
                k = -1 if sign == "-" else 1
                x += k * c[0]
                y += k * c[1]
            return (x, y)
        if ":" in s and "," not in s:
            a, r = s.split(":", 1)
            ang = evaluate(a, variables)
            rad = _len_cm(r, variables)
            return (rad * math.cos(math.radians(ang)), rad * math.sin(math.radians(ang)))
        if "," in s:
            a, b = _split_top(s)
            x = _len_cm(a, variables)
            y = _len_cm(b, variables)
            return (x * self.x_unit[0] + y * self.y_unit[0], x * self.x_unit[1] + y * self.y_unit[1])
        # a node (and anchor)
        name, _, anchor = s.partition(".")
        name = name.strip()
        nd = self.nodes.get(name)
        if nd is None:
            self.err(f"Package pgf Error: No shape named `{name}' is known.",
                     f"Name the node first: \\node ({name}) at (0,0) {{...}};")
            return last
        if anchor:
            return self.anchor(nd, anchor.strip())
        return nd["center"]

    def anchor(self, nd, a):
        cx, cy = nd["center"]
        w, h = nd["w"] / CM / 2, nd["h"] / CM / 2
        if a in ANCHORS:
            dx, dy = ANCHORS[a]
            if nd["shape"] == "circle" and dx and dy:
                return (cx + dx * w * 0.7071, cy + dy * h * 0.7071)
            return (cx + dx * w, cy + dy * h)
        try:
            ang = float(a)
            return self.border(nd, (cx + math.cos(math.radians(ang)) * 100, cy + math.sin(math.radians(ang)) * 100))
        except ValueError:
            return (cx, cy)

    def border(self, nd, toward):
        """Where the line from the node's center toward a point leaves its shape."""
        cx, cy = nd["center"]
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0 or nd["shape"] == "coordinate":
            return (cx, cy)
        w, h = nd["w"] / CM / 2, nd["h"] / CM / 2
        if nd["shape"] in ("circle", "ellipse"):
            if nd["shape"] == "circle":
                r = max(w, h)
                d = math.hypot(dx, dy)
                return (cx + dx / d * r, cy + dy / d * r)
            t = 1 / math.sqrt((dx / w) ** 2 + (dy / h) ** 2)
            return (cx + dx * t, cy + dy * t)
        tx = w / abs(dx) if dx else float("inf")
        ty = h / abs(dy) if dy else float("inf")
        t = min(tx, ty)
        return (cx + dx * t, cy + dy * t)


def _split_top(s):
    depth = 0
    for i, c in enumerate(s):
        if c in "({":
            depth += 1
        elif c in ")}":
            depth -= 1
        elif c == "," and depth == 0:
            return s[:i], s[i + 1:]
    return s, "0"


def _len_cm(s, variables=None) -> float:
    s = (s or "").strip()
    m = re.fullmatch(r"(-?[\d.]+)\s*(cm|mm|pt|in|em|ex)", s)
    if m:
        v = float(m.group(1))
        return v * {"cm": 1, "mm": 0.1, "pt": 1 / CM, "in": 2.54, "em": 10 / CM, "ex": 4.3 / CM}[m.group(2)]
    return evaluate(s, variables)


def _len_pt(s) -> float:
    s = (s or "").strip()
    d = dim_pt(s)
    if d is not None:
        return d
    try:
        return float(s) * CM
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- statements

def _statements(src: str):
    """Split the picture body at top-level `;` (braces, brackets and parentheses protect)."""
    out, cur, depth = [], [], 0
    i = 0
    while i < len(src):
        c = src[i]
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        if c == ";" and depth <= 0:
            out.append("".join(cur).strip())
            cur = []
            depth = 0
        else:
            cur.append(c)
        i += 1
    rest = "".join(cur).strip()
    if rest:
        out.append(rest)
    return out


def _foreach_values(spec: str, variables):
    spec = spec.strip()
    if spec.startswith("{") and spec.endswith("}"):
        spec = spec[1:-1]
    items = [x.strip() for x in _split_list(spec)]
    out = []
    i = 0
    while i < len(items):
        if items[i] == "..." and out and i + 1 < len(items):
            try:
                a = float(out[-1])
                step = a - float(out[-2]) if len(out) > 1 else 1.0
                b = float(items[i + 1])
                x = a + step
                while (step > 0 and x <= b + 1e-9) or (step < 0 and x >= b - 1e-9):
                    out.append(_fmt(x))
                    x += step
                i += 2
                continue
            except ValueError:
                pass
        out.append(items[i])
        i += 1
    return out


def _split_list(s):
    out, cur, depth = [], [], 0
    for c in s:
        if c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return [x for x in out if x.strip()]


def _fmt(x):
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _subst(s: str, variables: dict) -> str:
    for k in sorted(variables, key=len, reverse=True):
        s = re.sub(r"\\" + k + r"(?![A-Za-z])", lambda m: str(variables[k]), s)
    return s


def run_statements(pic: Picture, src: str, variables: dict, depth=0):
    for st in _statements(src):
        if not st:
            continue
        st_v = _subst(st, variables)
        m = re.match(r"\\foreach\s*((?:\\\w+\s*/?\s*)+)\s*(?:\[[^\]]*\]\s*)?in\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|"
                     r"\S+)\s*(.*)$", st, re.S)
        if m:
            pic.e.used["\\foreach"] += 1
            names = [x.strip().lstrip("\\") for x in m.group(1).split("/") if x.strip()]
            body = m.group(3).strip()
            if body.startswith("{") and body.endswith("}"):
                body = body[1:-1]
            values = _foreach_values(_subst(m.group(2), variables), variables)
            for val in values:
                parts = val.split("/") if len(names) > 1 else [val]
                vv = dict(variables)
                for n, p in zip(names, parts):
                    vv[n] = p.strip()
                run_statements(pic, body + ";", vv, depth + 1)
            continue
        m = re.match(r"\\(node|draw|fill|filldraw|path|coordinate|shade|shadedraw|clip|pgfmathsetmacro|"
                     r"begin|end|addplot|matrix)\b(.*)$", st_v, re.S)
        if m and m.group(1) not in ("begin", "end"):
            pic.e.used["\\" + m.group(1)] += 1
        if not m:
            if st_v.startswith("\\"):
                name = re.match(r"\\(\w+)", st_v)
                pic.err("Undefined control sequence.", f"\\{name.group(1) if name else ''} isn't a TikZ command here "
                                                       f"(\\node, \\draw, \\fill, \\foreach ...).")
            else:
                pic.err("Package tikz Error: Giving up on this path. Did you forget a semicolon?",
                        "Every TikZ command ends with a `;`.")
            continue
        kind, rest = m.group(1), m.group(2)
        try:
            if kind == "pgfmathsetmacro":
                mm = re.match(r"\s*\{?\\(\w+)\}?\s*\{(.*)\}\s*$", rest, re.S)
                if mm:
                    variables[mm.group(1)] = _fmt(evaluate(_subst(mm.group(2), variables), variables))
                continue
            if kind == "node":
                node_stmt(pic, rest, variables)
            elif kind == "coordinate":
                mm = re.match(r"\s*(?:\[[^\]]*\])?\s*\(([^)]*)\)\s*at\s*(\(.*\))\s*$", rest, re.S)
                if mm:
                    c = pic.coord(mm.group(2), variables)
                    nd = {"name": mm.group(1).strip(), "center": c, "w": 0, "h": 0, "shape": "coordinate",
                          "text": "", "style": {}}
                    pic.nodes[nd["name"]] = nd
            else:
                path_stmt(pic, kind, rest, variables)
        except TikzError as ex:
            pic.err("Package PGF Math Error: Could not parse input '" + str(ex) + "' as a floating point number, "
                    "sorry.", "Check the arithmetic in the coordinates.")


def _take_balanced(s, i, open_, close):
    depth = 0
    j = i
    while j < len(s):
        if s[j] == open_:
            depth += 1
        elif s[j] == close:
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise TikzError(s[i:i + 20])


def node_stmt(pic, rest, variables, at=None, along=None):
    s = rest.strip()
    opts = []
    name = None
    pos = None
    text = None
    while s:
        if s.startswith("["):
            o, k = _take_balanced(s, 0, "[", "]")
            opts += _split_opts(o)
            s = s[k:].strip()
        elif s.startswith("("):
            n, k = _take_balanced(s, 0, "(", ")")
            name = n.strip()
            s = s[k:].strip()
        elif s.startswith("at"):
            s = s[2:].strip()
            if s.startswith("("):
                c, k = _take_balanced(s, 0, "(", ")")
                pos = pic.coord("(" + c + ")", variables)
                s = s[k:].strip()
        elif s.startswith("{"):
            text, k = _take_balanced(s, 0, "{", "}")
            s = s[k:].strip()
            break
        else:
            break
    if text is None:
        pic.err("Package tikz Error: A node must have a (possibly empty) label text.",
                "\\node needs {text} at the end, even if empty: \\node (a) at (0,0) {};")
        text = ""
    st = pic.style(opts)
    nd = make_node(pic, name, text, st, pos if pos is not None else at)
    return nd


def make_node(pic, name, text, st, pos):
    # size: text estimate + inner sep, at least the minimum size
    txt = text.strip()
    w_txt, h_txt = _text_size(txt, st.get("font"))
    inner = st.get("inner", 3.33)
    w = max(w_txt + 2 * inner, st.get("minw", 0))
    h = max(h_txt + 2 * inner, st.get("minh", 0))
    if st.get("textw"):
        w = max(st["textw"] + 2 * inner, w)
    shape = st.get("shape", "rectangle")
    if shape == "circle":
        d = max(w, h, math.hypot(w_txt + 2 * inner, h_txt + 2 * inner) if not st.get("minw") else 0)
        w = h = max(d, st.get("minw", 0))
    if pos is None:
        pos = (0.0, 0.0)
    if "of" in st:
        direction, spec = st["of"]
        mm = re.match(r"\s*(?:([-\d.]+\s*\w*)\s+)?of\s+(.+)$", spec or "")
        if mm:
            ref = pic.nodes.get(mm.group(2).strip())
            dist = _len_cm(mm.group(1)) if mm.group(1) else 1.0
            if ref is None:
                pic.err(f"Package pgf Error: No shape named `{mm.group(2).strip()}' is known.",
                        "Define the node you place this one relative to first.")
            else:
                dx, dy = DIRS[direction]
                # positioning: the gap is between the borders
                ex = (ref["w"] / 2 + w / 2) / CM + dist if dx else 0
                ey = (ref["h"] / 2 + h / 2) / CM + dist if dy else 0
                pos = (ref["center"][0] + dx * ex, ref["center"][1] + dy * ey)
    if st.get("anchor") in ANCHORS:
        ax, ay = ANCHORS[st["anchor"]]
        pos = (pos[0] - ax * w / 2 / CM, pos[1] - ay * h / 2 / CM)
    if "place" in st:
        d, v = st["place"]
        dx, dy = DIRS[d]
        off = (_len_cm(v) if v else 0) + 0.0
        pos = (pos[0] + dx * (w / 2 / CM + off), pos[1] + dy * (h / 2 / CM + off))
    pos = (pos[0] + st.get("xshift", 0.0), pos[1] + st.get("yshift", 0.0))
    if name is None:
        pic.anon += 1
        key = f"__node{pic.anon}"
    else:
        key = name
    nd = {"name": name, "center": pos, "w": w, "h": h, "shape": shape, "text": txt, "style": st}
    pic.nodes[key] = nd
    pic.items.append(("node", nd))
    for lab in st.get("labels", []):
        where, _, ltxt = lab.partition(":")
        if not ltxt:
            where, ltxt = "above", where
        d = DIRS.get(where.strip(), (0, 1))
        lw, lh = _text_size(ltxt.strip(), None)
        lp = (pos[0] + d[0] * (w / 2 + lw / 2 + 2) / CM, pos[1] + d[1] * (h / 2 + lh / 2 + 2) / CM)
        pic.items.append(("node", {"name": None, "center": lp, "w": lw, "h": lh, "shape": "rectangle",
                                   "text": ltxt.strip(), "style": {}}))
    return nd


def _text_size(txt, font):
    """A rough box for node text (pt): 0.5em per character, math a little wider."""
    if not txt:
        return 0.0, 0.0
    plain = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[A-Za-z]+|[{}_^]", "x", m.group(1))[:max(1, len(m.group(1)) // 2 + 1)], txt)
    plain = re.sub(r"\\[A-Za-z]+\s*", "", plain).replace("{", "").replace("}", "")
    lines = re.split(r"\\\\", plain)
    size = 10.0
    if font and "small" in font:
        size = 9.0
    if font and ("footnotesize" in font or "scriptsize" in font):
        size = 7.5
    w = max(len(l.strip()) for l in lines) * size * 0.5
    return w, size * 1.0 * len(lines) + (2 if len(lines) > 1 else 0)


def path_stmt(pic, kind, rest, variables):
    """\\draw[opts] (a) -- (b) node[above]{x} -- cycle ..."""
    s = rest.strip()
    opts = []
    while s.startswith("["):
        o, k = _take_balanced(s, 0, "[", "]")
        opts += _split_opts(o)
        s = s[k:].strip()
    base = {}
    if kind in ("draw", "filldraw", "shadedraw"):
        base["draw"] = "#000000"
    if kind in ("fill", "filldraw", "shade", "shadedraw"):
        base["fill"] = "#000000"
    st = pic.style(opts, base)
    if "color" in st:
        if kind in ("draw", "filldraw") and st.get("draw") in (None, "#000000"):
            st["draw"] = st["color"]
        if kind in ("fill", "filldraw") and "fill" in st and "fill" not in dict(opts):
            st["fill"] = st["color"] if kind == "fill" else st.get("fill")
    if kind == "filldraw" and dict(opts).get("fill") is None and "color" not in st:
        st["fill"] = "#000000"
    segs = []       # ("M", x, y) ("L", x, y) ("C", c1, c2, p) ("Z",) ("circle", c, r) ("rect", a, b) ...
    labels = []
    cur = None
    start = None
    pending_node_targets = []
    prev_node = None
    op = None
    i = 0
    while s:
        s = s.lstrip()
        if not s:
            break
        if s.startswith("--"):
            op = "--"
            s = s[2:]
            continue
        if s.startswith("-|") or s.startswith("|-"):
            op = s[:2]
            s = s[2:]
            continue
        if s.startswith(".."):
            m = re.match(r"\.\.\s*controls\s*(\([^()]*(?:\([^()]*\))?[^()]*\))\s*(?:and\s*(\([^()]*\)))?\s*\.\.", s)
            if not m:
                raise TikzError(s[:20])
            c1 = pic.coord(m.group(1), variables, cur)
            c2 = pic.coord(m.group(2), variables, cur) if m.group(2) else c1
            op = ("C", c1, c2)
            s = s[m.end():]
            continue
        if s.startswith("to"):
            m = re.match(r"to\s*(\[[^\]]*\])?", s)
            op = ("to", _split_opts(m.group(1)[1:-1]) if m.group(1) else [])
            s = s[m.end():]
            continue
        if s.startswith("cycle"):
            if op == "--" or op is None:
                segs.append(("Z",))
            s = s[5:]
            op = None
            continue
        if s.startswith("rectangle"):
            op = "rect"
            s = s[9:]
            continue
        if s.startswith("grid"):
            m = re.match(r"grid\s*(\[[^\]]*\])?", s)
            gst = pic.style(_split_opts(m.group(1)[1:-1])) if m.group(1) else {}
            op = ("grid", gst.get("step", st.get("step", 1.0)))
            s = s[m.end():]
            continue
        if s.startswith("circle") or s.startswith("ellipse"):
            m = re.match(r"(circle|ellipse)\s*(?:\[([^\]]*)\])?\s*(?:\(([^()]*)\))?", s)
            if cur is None:
                cur = (0.0, 0.0)
            if m.group(1) == "circle":
                r = _len_cm(m.group(3), variables) if m.group(3) else \
                    pic.style(_split_opts(m.group(2) or "")).get("radius", 0.1)
                segs.append(("circle", cur, r, r))
            else:
                a, _, b = (m.group(3) or "1 and 1").partition("and")
                segs.append(("circle", cur, _len_cm(a, variables), _len_cm(b, variables)))
            s = s[m.end():]
            continue
        if s.startswith("arc"):
            m = re.match(r"arc\s*\(([^)]*)\)", s)
            if not m:
                m = re.match(r"arc\s*\[([^\]]*)\]", s)
                d = dict(_split_opts(m.group(1)))
                a0, a1, r = float(d.get("start angle", 0)), float(d.get("end angle", 90)), _len_cm(d.get("radius", "1"))
            else:
                a0, a1, r = (x.strip() for x in m.group(1).split(":"))
                a0, a1, r = evaluate(a0, variables), evaluate(a1, variables), _len_cm(r, variables)
            if cur is None:
                cur = (0.0, 0.0)
            cx = cur[0] - r * math.cos(math.radians(a0))
            cy = cur[1] - r * math.sin(math.radians(a0))
            end = (cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
            segs.append(("arc", cur, end, r, a1 - a0))
            cur = end
            s = s[m.end():]
            continue
        if s.startswith("node"):
            m = re.match(r"node\s*", s)
            s2 = s[m.end():]
            # node [opts] (name) {text}
            j = 0
            o_list = []
            nm = None
            while True:
                s2 = s2.lstrip()
                if s2.startswith("["):
                    o, k = _take_balanced(s2, 0, "[", "]")
                    o_list += _split_opts(o)
                    s2 = s2[k:]
                elif s2.startswith("("):
                    n, k = _take_balanced(s2, 0, "(", ")")
                    nm = n.strip()
                    s2 = s2[k:]
                else:
                    break
            if s2.lstrip().startswith("{"):
                s2 = s2.lstrip()
                txt, k = _take_balanced(s2, 0, "{", "}")
                s2 = s2[k:]
            else:
                txt = ""
            nst = pic.style(o_list)
            if op == "--" and cur is not None:
                # placed on the coming segment
                pending_node_targets.append((nm, txt, nst))
            else:
                nd = make_node(pic, nm, txt, nst, cur or (0.0, 0.0))
                labels.append(nd)
            s = s2
            continue
        if s.startswith("+") or s.startswith("("):
            rel = 0
            if s.startswith("++"):
                rel = 2
                s = s[2:]
            elif s.startswith("+"):
                rel = 1
                s = s[1:]
            c, k = _take_balanced(s, 0, "(", ")")
            s = s[k:]
            name_ref = None
            if "," not in c and ":" not in c and not c.strip().startswith("$"):
                name_ref = c.strip().partition(".")[0]
                anchor = c.strip().partition(".")[2]
                if name_ref in pic.nodes and not anchor:
                    target_node = pic.nodes[name_ref]
                else:
                    target_node = None
            else:
                target_node = None
            p = pic.coord("(" + c + ")", variables, cur)
            if rel and cur is not None:
                p = (cur[0] + p[0], cur[1] + p[1])
            if cur is None or op is None:
                if target_node is not None:
                    p = target_node["center"]
                segs.append(("M",) + p)
                start = p
                start_node = target_node
                cur = p
                prev_node = target_node
                if rel == 2:
                    pass
                continue
            a = cur
            # clip at node borders
            if prev_node is not None:
                a = pic.border(prev_node, p)
                if segs and segs[-1][0] == "M":
                    segs[-1] = ("M",) + a
            b = p
            if target_node is not None:
                b = pic.border(target_node, a if op == "--" else (a[0], p[1]))
            if op == "--":
                segs.append(("L",) + b)
                for nm, txt, nst in pending_node_targets:
                    f = nst.get("pos", 0.5)
                    mid = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                    labels.append(make_node(pic, nm, txt, nst, mid))
                pending_node_targets = []
            elif op in ("-|", "|-"):
                corner = (b[0], a[1]) if op == "-|" else (a[0], b[1])
                if target_node is not None:
                    b = pic.border(target_node, corner)
                segs.append(("L",) + corner)
                segs.append(("L",) + b)
            elif isinstance(op, tuple) and op[0] == "C":
                segs.append(("C", op[1], op[2], b))
            elif isinstance(op, tuple) and op[0] == "to":
                o = dict(op[1])
                if "bend left" in o or "bend right" in o or "out" in o:
                    ang = float(o.get("bend left") or o.get("bend right") or 30)
                    sign = 1 if "bend left" in o else -1
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                    k2 = math.tan(math.radians(ang)) / 2
                    c = (mx - sign * dy * k2, my + sign * dx * k2)
                    segs.append(("C", c, c, b))
                else:
                    segs.append(("L",) + b)
            elif op == "rect":
                segs.append(("rect", cur, p))
            elif isinstance(op, tuple) and op[0] == "grid":
                segs.append(("grid", cur, p, op[1]))
            cur = p
            prev_node = target_node
            op = None
            continue
        raise TikzError(s[:20])
    if not segs and not labels:
        pic.err("Package tikz Error: Giving up on this path. Did you forget a semicolon?", "")
        return
    pic.items.append(("path", {"kind": kind, "segs": segs, "style": st}))
    # nodes along the path draw on top of it
    for nd in labels:
        pic.items.remove(("node", nd))
        pic.items.append(("node", nd))


# ---------------------------------------------------------------- SVG

def to_svg(pic: Picture) -> tuple[str, dict]:
    s = pic.scale * CM
    # bounding box (cm)
    xs, ys = [], []
    for kind, it in pic.items:
        if kind == "node":
            cx, cy = it["center"]
            xs += [cx - it["w"] / 2 / CM, cx + it["w"] / 2 / CM]
            ys += [cy - it["h"] / 2 / CM, cy + it["h"] / 2 / CM]
        else:
            for sg in it["segs"]:
                if sg[0] in ("M", "L"):
                    xs.append(sg[1])
                    ys.append(sg[2])
                elif sg[0] == "C":
                    xs += [sg[1][0], sg[2][0], sg[3][0]]
                    ys += [sg[1][1], sg[2][1], sg[3][1]]
                elif sg[0] == "circle":
                    xs += [sg[1][0] - sg[2], sg[1][0] + sg[2]]
                    ys += [sg[1][1] - sg[3], sg[1][1] + sg[3]]
                elif sg[0] in ("rect", "grid"):
                    xs += [sg[1][0], sg[2][0]]
                    ys += [sg[1][1], sg[2][1]]
                elif sg[0] == "arc":
                    xs += [sg[1][0], sg[2][0]]
                    ys += [sg[1][1], sg[2][1]]
    if not xs:
        return '<svg width="0" height="0"></svg>', {"nodes": [], "paths": []}
    pad = 2.0 / s
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    W, H = (x1 - x0) * s, (y1 - y0) * s

    def P(p):
        return ((p[0] - x0) * s, (y1 - p[1]) * s)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}pt" height="{H:.1f}pt" '
           f'viewBox="0 0 {W:.2f} {H:.2f}" font-size="10">']
    out.append('<defs><marker id="tkz-to" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
               'orient="auto-start-reverse"><path d="M1,1 Q6,5 1,9 M1,1 L9,5 L1,9" fill="none" stroke="context-stroke" '
               'stroke-width="1.4"/></marker><marker id="tkz-stealth" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 L3,5 Z" '
               'fill="context-stroke"/></marker><marker id="tkz-latex" viewBox="0 0 10 10" refX="9" refY="5" '
               'markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 Z" '
               'fill="context-stroke"/></marker></defs>')
    scene_nodes, scene_paths = [], []
    for kind, it in pic.items:
        if kind == "path":
            out.append(_svg_path(pic, it, P, s))
            scene_paths.append(_scene_path(pic, it))
        else:
            out.append(_svg_node(it, P, s))
            if it["shape"] != "coordinate":
                scene_nodes.append({"name": it.get("name"), "text": " ".join(it["text"].split()),
                                    "at": [round(it["center"][0], 2), round(it["center"][1], 2)],
                                    "shape": it["shape"], "fill": it["style"].get("fill"),
                                    "draw": bool(it["style"].get("draw"))})
    out.append("</svg>")
    return "".join(out), {"nodes": scene_nodes, "paths": scene_paths}


def _scene_path(pic, it):
    pts = []
    shapes = []
    for sg in it["segs"]:
        if sg[0] in ("M", "L"):
            pts.append([round(sg[1], 2), round(sg[2], 2)])
        elif sg[0] == "circle":
            shapes.append(["circle", [round(sg[1][0], 2), round(sg[1][1], 2)], round(sg[2], 2)])
        elif sg[0] == "rect":
            shapes.append(["rect", [round(sg[1][0], 2), round(sg[1][1], 2)], [round(sg[2][0], 2), round(sg[2][1], 2)]])
    st = it["style"]
    return {"kind": it["kind"], "points": pts, "shapes": shapes, "arrows": st.get("arrows"),
            "draw": st.get("draw"), "fill": st.get("fill") if it["kind"] != "draw" or "fill" in st else None,
            "dash": st.get("dash")}


def _stroke_attrs(st, default_draw=True):
    a = []
    color = st.get("draw") or st.get("draw_color")
    if color:
        a.append(f'stroke="{color}"')
        a.append(f'stroke-width="{st.get("lw", 0.4):.2f}"')
        if st.get("dash") in ("dashed", "densely dashed", "loosely dashed"):
            a.append('stroke-dasharray="3,3"')
        elif st.get("dash") in ("dotted", "densely dotted"):
            a.append(f'stroke-dasharray="0.5,2" stroke-linecap="round"')
    else:
        a.append('stroke="none"')
    return a


def _arrow_markers(pic, st):
    arr = st.get("arrows")
    if not arr:
        return ""
    tip = pic.arrow_tip.lower() or "to"
    m = re.fullmatch(r"(<?)(\{?[A-Za-z]*\}?)-(\{?[A-Za-z]*\}?)(>?)", arr.replace(" ", ""))
    if not m:
        return ""
    def mk(name, arrow):
        name = name.strip("{}").lower()
        kind = "stealth" if "stealth" in (name or tip) else "latex" if "latex" in (name or tip) else "to"
        return kind
    start = arr.startswith("<") or (m.group(2) and m.group(2).strip("{}") and not arr.startswith("-"))
    end = arr.endswith(">") or (m.group(3) and m.group(3).strip("{}"))
    out = ""
    if start:
        out += f' marker-start="url(#tkz-{mk(m.group(2), True)})"'
    if end:
        out += f' marker-end="url(#tkz-{mk(m.group(3), True)})"'
    return out


def _svg_path(pic, it, P, s):
    st = it["style"]
    d = []
    extra = []
    for sg in it["segs"]:
        if sg[0] == "M":
            x, y = P(sg[1:])
            d.append(f"M{x:.2f},{y:.2f}")
        elif sg[0] == "L":
            x, y = P(sg[1:])
            d.append(f"L{x:.2f},{y:.2f}")
        elif sg[0] == "C":
            (ax, ay), (bx, by), (cx, cy) = P(sg[1]), P(sg[2]), P(sg[3])
            d.append(f"C{ax:.2f},{ay:.2f} {bx:.2f},{by:.2f} {cx:.2f},{cy:.2f}")
        elif sg[0] == "Z":
            d.append("Z")
        elif sg[0] == "circle":
            cx, cy = P(sg[1])
            extra.append(("ellipse", cx, cy, sg[2] * s, sg[3] * s))
        elif sg[0] == "rect":
            (ax, ay), (bx, by) = P(sg[1]), P(sg[2])
            extra.append(("rect", min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)))
        elif sg[0] == "grid":
            (ax, ay), (bx, by) = sg[1], sg[2]
            step = sg[3] or 1.0
            gx = math.ceil(min(ax, bx) / step) * step
            while gx <= max(ax, bx) + 1e-9:
                p1, p2 = P((gx, ay)), P((gx, by))
                d.append(f"M{p1[0]:.2f},{p1[1]:.2f}L{p2[0]:.2f},{p2[1]:.2f}")
                gx += step
            gy = math.ceil(min(ay, by) / step) * step
            while gy <= max(ay, by) + 1e-9:
                p1, p2 = P((ax, gy)), P((bx, gy))
                d.append(f"M{p1[0]:.2f},{p1[1]:.2f}L{p2[0]:.2f},{p2[1]:.2f}")
                gy += step
        elif sg[0] == "arc":
            (ax, ay), (bx, by) = P(sg[1]), P(sg[2])
            r = sg[3] * s
            sweep = sg[4]
            large = 1 if abs(sweep) > 180 else 0
            d.append(f"M{ax:.2f},{ay:.2f}A{r:.2f},{r:.2f} 0 {large} {0 if sweep > 0 else 1} {bx:.2f},{by:.2f}")
    fill = st.get("fill") if it["kind"] in ("fill", "filldraw", "shade", "shadedraw") or "fill" in st else None
    if it["kind"] in ("fill",) and not st.get("draw"):
        stroke = ['stroke="none"']
    elif it["kind"] == "path" and not st.get("draw"):
        stroke = ['stroke="none"']
    else:
        stroke = _stroke_attrs(st)
    fill_attr = f'fill="{fill}"' if fill else 'fill="none"'
    op = f' fill-opacity="{st["fill_opacity"]}"' if st.get("fill_opacity") is not None else ""
    out = []
    rounded = st.get("rounded")
    if d:
        out.append(f'<path d="{"".join(d)}" {fill_attr}{op} {" ".join(stroke)}{_arrow_markers(pic, st)} '
                   f'stroke-linejoin="{"round" if rounded else "miter"}"/>')
    for ex in extra:
        if ex[0] == "ellipse":
            out.append(f'<ellipse cx="{ex[1]:.2f}" cy="{ex[2]:.2f}" rx="{ex[3]:.2f}" ry="{ex[4]:.2f}" {fill_attr}{op} '
                       f'{" ".join(stroke)}/>')
        else:
            rx = f' rx="{rounded:.1f}"' if rounded else ""
            out.append(f'<rect x="{ex[1]:.2f}" y="{ex[2]:.2f}" width="{ex[3]:.2f}" height="{ex[4]:.2f}"{rx} '
                       f'{fill_attr}{op} {" ".join(stroke)}/>')
    return "".join(out)


def _svg_node(nd, P, s):
    st = nd["style"]
    cx, cy = P(nd["center"])
    w, h = nd["w"] * s / CM, nd["h"] * s / CM
    out = []
    fill = st.get("fill")
    draw = st.get("draw")
    if nd["shape"] != "coordinate" and (fill or draw):
        attrs = f'fill="{fill or "none"}" ' + " ".join(_stroke_attrs(st))
        if nd["shape"] == "circle":
            out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{w / 2:.2f}" {attrs}/>')
        elif nd["shape"] == "ellipse":
            out.append(f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{w / 2:.2f}" ry="{h / 2:.2f}" {attrs}/>')
        else:
            r = st.get("rounded")
            rx = f' rx="{r:.1f}"' if r else ""
            out.append(f'<rect x="{cx - w / 2:.2f}" y="{cy - h / 2:.2f}" width="{w:.2f}" height="{h:.2f}"{rx} '
                       f'{attrs}/>')
    if nd["text"]:
        color = st.get("text") or st.get("color") or "#000"
        fw = max(w, 10)
        out.append(f'<foreignObject x="{cx - fw / 2 - 20:.2f}" y="{cy - h / 2 - 4:.2f}" width="{fw + 40:.2f}" '
                   f'height="{h + 8:.2f}"><div xmlns="http://www.w3.org/1999/xhtml" class="tex-tikz-text" '
                   f'style="color:{color};height:{h + 8:.2f}px">{_node_html(nd["text"])}</div></foreignObject>')
    return "".join(out)


def _node_html(txt):
    """Node text: plain text with $math$ (typeset by KaTeX) and \\\\ line breaks."""
    out = []
    for i, part in enumerate(re.split(r"(\$[^$]*\$)", txt)):
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(f'<span class="tex-m" data-tex="{html.escape(part[1:-1], quote=True)}"></span>')
        else:
            part = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", html.escape(part))
            part = re.sub(r"\\\\", "<br>", part)
            part = re.sub(r"\\[A-Za-z]+\s*", "", part).replace("{", "").replace("}", "")
            out.append(part)
    return "".join(out)


# ---------------------------------------------------------------- the environment

@env("tikzpicture")
def tikzpicture(e, name, t):
    opt = e.read_opt()
    body = e.env_body(name, t)
    pic = Picture(e, detokenize(opt) if opt is not None else "")
    src = detokenize(body)
    # pgfplots axes inside
    from .pgfplots import extract_axes
    src, axes = extract_axes(e, src, t)
    if axes and not src.strip():
        for ax in axes:
            e.add_inline({"t": "box", "b": ax})
        return
    run_statements(pic, src, {})
    for msg, hlp in pic.errors:
        e.error(msg, t, hlp)
    svg, scene = to_svg(pic)
    e.add_inline({"t": "box", "b": {"t": "tikz", "svg": svg, "scene": scene}})
    for ax in axes:
        e.add_inline({"t": "box", "b": ax})


@cmd("tikz")
def tikz_inline(e, t):
    opt = e.read_opt()
    body = e.read_arg("tikz")
    pic = Picture(e, detokenize(opt) if opt is not None else "")
    run_statements(pic, detokenize(body), {})
    for msg, hlp in pic.errors:
        e.error(msg, t, hlp)
    svg, scene = to_svg(pic)
    e.add_inline({"t": "box", "b": {"t": "tikz", "svg": svg, "scene": scene}})


@cmd("draw", "node", "fill", "filldraw", "path", "foreach", "coordinate", "shade", "clip", "pgfmathsetmacro",
     "addplot", "addlegendentry", "legend")
def tikz_outside(e, t):
    e.error("Undefined control sequence.", t, f"\\{t.v} only works inside a tikzpicture (or an axis).")

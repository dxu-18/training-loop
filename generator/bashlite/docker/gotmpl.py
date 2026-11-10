"""A small Go text/template, enough for `docker ... --format`: {{.Field.Sub}}, {{json .X}}, {{index .X "k"}},
{{range .X}}...{{end}} (with $k, $v :=), {{if .X}}...{{else}}...{{end}}, {{len .X}}, {{join .X ","}},
{{upper .X}}, {{lower .X}}, {{title .X}}, {{println .X}}, {{printf "%s" .X}}, {{.}}, `table` headers and \\t."""

from __future__ import annotations

import json
import re


class TemplateError(Exception):
    pass


TOKEN = re.compile(r"\{\{-?\s*(.*?)\s*-?\}\}", re.S)


def parse(text):
    """-> nested nodes: ("text", s) | ("expr", src) | ("range", vars, src, body, else) | ("if", src, body, else)"""
    pos = 0
    toks = []
    for m in TOKEN.finditer(text):
        if m.start() > pos:
            toks.append(("text", text[pos:m.start()]))
        toks.append(("tag", m.group(1)))
        pos = m.end()
    if pos < len(text):
        toks.append(("text", text[pos:]))
    out, i = _block(toks, 0, top=True)
    return out


def _block(toks, i, top=False):
    nodes, alt = [], None
    cur = nodes
    while i < len(toks):
        kind, val = toks[i]
        if kind == "text":
            cur.append(("text", val))
            i += 1
            continue
        word = val.split(None, 1)[0] if val else ""
        if word == "end":
            if top:
                raise TemplateError('template: :1: unexpected {{end}}')
            return (nodes, alt), i + 1
        if word == "else":
            alt = []
            cur = alt
            i += 1
            continue
        if word in ("range", "if", "with"):
            src = val[len(word):].strip()
            (body, other), i = _block(toks, i + 1)
            if word == "range":
                m = re.match(r"^\$(\w+)\s*(?:,\s*\$(\w+))?\s*:=\s*(.*)$", src)
                vars_ = (m.group(1), m.group(2)) if m else ()
                src = m.group(3) if m else src
                cur.append(("range", vars_, src, body, other))
            else:
                cur.append((word, src, body, other))
            continue
        cur.append(("expr", val))
        i += 1
    if not top:
        raise TemplateError("template: :1: unexpected EOF")
    return nodes, i


def render(text, data):
    try:
        nodes = parse(text)
    except TemplateError:
        raise
    out = []
    _run(nodes, data, {}, out)
    return "".join(out)


def _run(nodes, dot, vars_, out):
    for n in nodes:
        kind = n[0]
        if kind == "text":
            out.append(n[1])
        elif kind == "expr":
            out.append(_str(_eval(n[1], dot, vars_)))
        elif kind == "if":
            _, src, body, alt = n
            v = _eval(src, dot, vars_)
            _run(body if _truthy(v) else (alt or []), dot, vars_, out)
        elif kind == "with":
            _, src, body, alt = n
            v = _eval(src, dot, vars_)
            _run(body if _truthy(v) else (alt or []), v if _truthy(v) else dot, vars_, out)
        elif kind == "range":
            _, names, src, body, alt = n
            v = _eval(src, dot, vars_)
            items = list(v.items()) if isinstance(v, dict) else list(enumerate(v or []))
            if isinstance(v, dict):
                items.sort(key=lambda kv: str(kv[0]))
            if not items:
                _run(alt or [], dot, vars_, out)
            for k, x in items:
                vv = dict(vars_)
                if len(names) == 2 and names[1]:
                    vv[names[0]], vv[names[1]] = k, x
                elif names:
                    vv[names[0]] = x
                _run(body, x, vv, out)


def _truthy(v):
    return bool(v) and v not in ("false",)


def _str(v):
    if v is None:
        return "<no value>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + " ".join(_str(x) for x in v) + "]"
    if isinstance(v, dict):
        return "map[" + " ".join(f"{k}:{_str(x)}" for k, x in sorted(v.items())) + "]"
    return str(v)


def _field(v, path):
    for part in path:
        if part == "":
            continue
        if isinstance(v, dict):
            v = v.get(part)
        else:
            v = getattr(v, part, None)
    return v


def _arg(tok, dot, vars_):
    if tok.startswith('"') and tok.endswith('"'):
        return json.loads(tok)
    if tok.startswith("`") and tok.endswith("`"):
        return tok[1:-1]
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    if tok == ".":
        return dot
    if tok.startswith("$"):
        name, _, rest = tok[1:].partition(".")
        v = vars_.get(name, dot if name == "" else None)
        return _field(v, rest.split(".")) if rest else v
    if tok.startswith("."):
        try:
            return _field(dot, tok[1:].split("."))
        except TemplateError:
            return None
    if tok in ("true", "false"):
        return tok == "true"
    raise TemplateError(f'template: :1: function "{tok}" not defined')


def _split(src):
    return re.findall(r'"(?:\\.|[^"\\])*"|`[^`]*`|\([^)]*\)|\S+', src)


def _eval(src, dot, vars_):
    src = src.strip()
    if "|" in src and not src.startswith('"'):
        parts = [p.strip() for p in src.split("|")]
        v = _eval(parts[0], dot, vars_)
        for p in parts[1:]:
            v = _call(_split(p), dot, vars_, extra=[v])
        return v
    toks = _split(src)
    if not toks:
        return ""
    if len(toks) == 1 and not re.fullmatch(r"[a-z]\w*", toks[0]):
        t = toks[0]
        if t.startswith("("):
            return _eval(t[1:-1], dot, vars_)
        return _arg(t, dot, vars_)
    return _call(toks, dot, vars_)


def _call(toks, dot, vars_, extra=()):
    fn = toks[0]
    args = [(_eval(t[1:-1], dot, vars_) if t.startswith("(") else _arg(t, dot, vars_)) for t in toks[1:]] + list(extra)
    if fn == "json":
        return json.dumps(args[0], separators=(",", ":")) if args else ""
    if fn == "index":
        v = args[0]
        for k in args[1:]:
            v = v.get(k) if isinstance(v, dict) else (v[k] if v is not None and 0 <= k < len(v) else None)
        return v
    if fn == "len":
        return len(args[0] or [])
    if fn == "join":
        return args[1].join(_str(x) for x in (args[0] or []))
    if fn == "upper":
        return _str(args[0]).upper()
    if fn == "lower":
        return _str(args[0]).lower()
    if fn == "title":
        return _str(args[0]).title()
    if fn == "split":
        return _str(args[0]).split(args[1])
    if fn == "println":
        return " ".join(_str(a) for a in args) + "\n"
    if fn == "print":
        return "".join(_str(a) for a in args)
    if fn == "printf":
        fmt = args[0]
        vals = tuple(args[1:])
        return re.sub(r"%v", "%s", fmt) % tuple(_str(v) if not isinstance(v, (int, float)) else v for v in vals)
    if fn == "eq":
        return args[0] == args[1]
    if fn == "ne":
        return args[0] != args[1]
    if fn == "not":
        return not _truthy(args[0])
    if fn == "truncate":
        return _str(args[0])[:args[1]]
    if fn == "pad":
        return _str(args[0]).ljust(args[1])
    raise TemplateError(f'template: :1: function "{fn}" not defined')


def table(fmt, rows, headers):
    """`table {{.A}}\\t{{.B}}`: a header row from the field names, then rows; columns padded like tabwriter."""
    body = fmt[5:].lstrip() if fmt.startswith("table") else fmt
    body = body.replace("\\t", "\t").replace("\\n", "\n")
    head = re.sub(r"\{\{\s*\.(\w+)\s*\}\}", lambda m: headers.get(m.group(1), m.group(1).upper()), body)
    lines = [head] + [render(body, r) for r in rows]
    return tabulate([ln.split("\t") for ln in lines])


def tabulate(rows, pad=3):
    widths = {}
    for r in rows:
        for i, cell in enumerate(r[:-1]):
            widths[i] = max(widths.get(i, 0), len(cell))
    out = []
    for r in rows:
        cells = [c.ljust(widths[i] + pad) if i < len(r) - 1 else c for i, c in enumerate(r)]
        out.append("".join(cells).rstrip())
    return "\n".join(out) + "\n"

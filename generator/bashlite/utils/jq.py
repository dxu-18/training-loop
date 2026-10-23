"""jq: a JSON processor. Expressions compile to Python generator functions (each yields zero or more outputs)."""

from __future__ import annotations

import json
import math
import re

from ..exec import util
from ..vfs import FSError


class JqError(Exception):
    def __init__(self, msg, value=None):
        super().__init__(msg)
        self.msg = msg
        self.value = value


class JqSyntax(Exception):
    pass


class BreakOut(Exception):
    def __init__(self, label):
        super().__init__(label)
        self.label = label


# ---------------------------------------------------------------------------- formatting

def jtype(v):
    if v is None:
        return "null"
    if v is True or v is False:
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def num_text(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if v != v:
        return "null"
    if v in (math.inf, -math.inf):
        return "1.7976931348623157e+308" if v > 0 else "-1.7976931348623157e+308"
    if v == int(v) and abs(v) < 1e17:
        return str(int(v))
    r = repr(v)
    if "e" in r:
        mant, exp = r.split("e")
        sign = exp[0] if exp[0] in "+-" else "+"
        exp = exp.lstrip("+-").lstrip("0") or "0"
        r = f"{mant}e{sign}{exp.zfill(2)}"
    return r


COL = {"null": "1;30", "false": "0;39", "true": "0;39", "number": "0;39", "string": "0;32", "array": "1;39",
       "object": "1;39", "key": "34;1"}


def dumps(v, indent=2, compact=False, sort_keys=False, color=False, ascii_=False, level=0, tab=False):
    def c(kind, text):
        return f"\x1b[{COL[kind]}m{text}\x1b[0m" if color else text

    def s(x):
        return json.dumps(x, ensure_ascii=ascii_)
    if v is None:
        return c("null", "null")
    if v is True:
        return c("true", "true")
    if v is False:
        return c("false", "false")
    if isinstance(v, (int, float)):
        return c("number", num_text(v))
    if isinstance(v, str):
        return c("string", s(v))
    pad = ("\t" if tab else " " * indent)
    if isinstance(v, list):
        if not v:
            return c("array", "[]")
        if compact or indent == 0 and not tab:
            inner = c("array", ",").join(dumps(x, indent, compact, sort_keys, color, ascii_, level + 1, tab) for x in v)
            return c("array", "[") + inner + c("array", "]")
        items = [pad * (level + 1) + dumps(x, indent, compact, sort_keys, color, ascii_, level + 1, tab) for x in v]
        return c("array", "[") + "\n" + (c("array", ",") + "\n").join(items) + "\n" + pad * level + c("array", "]")
    keys = sorted(v) if sort_keys else list(v)
    if not keys:
        return c("object", "{}")
    if color:
        def kv(k, sep):
            return f"\x1b[{COL['key']}m{s(k)}\x1b[0m" + c("object", sep) + dumps(v[k], indent, compact, sort_keys, color, ascii_, level + 1, tab)
    else:
        def kv(k, sep):
            return s(k) + sep + dumps(v[k], indent, compact, sort_keys, color, ascii_, level + 1, tab)
    if compact or indent == 0 and not tab:
        return c("object", "{") + c("object", ",").join(kv(k, ":") for k in keys) + c("object", "}")
    items = [pad * (level + 1) + kv(k, ": ") for k in keys]
    return c("object", "{") + "\n" + (c("object", ",") + "\n").join(items) + "\n" + pad * level + c("object", "}")


def to_json_text(v):
    return dumps(v, compact=True)


# ---------------------------------------------------------------------------- comparison (jq ordering)

ORDER = {"null": 0, "boolean": 1, "number": 3, "string": 4, "array": 5, "object": 6}


def sort_key(v):
    t = jtype(v)
    if t == "boolean":
        return (1, 1 if v else 0)
    if t == "null":
        return (0,)
    if t == "number":
        return (3, v)
    if t == "string":
        return (4, v)
    if t == "array":
        return (5, [sort_key(x) for x in v])
    return (6, sorted(v.keys()), [sort_key(v[k]) for k in sorted(v.keys())])


def cmp(a, b):
    ka, kb = sort_key(a), sort_key(b)
    return (ka > kb) - (ka < kb)


def truthy(v):
    return not (v is None or v is False)


# ---------------------------------------------------------------------------- lexer

TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|
    (?P<str>")|
    (?P<field>\.[A-Za-z_][A-Za-z0-9_]*)|
    (?P<format>@[a-z0-9]+)|
    (?P<var>\$(?:__loc__|[A-Za-z_][A-Za-z0-9_]*))|
    (?P<ident>[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)|
    (?P<op>\?//|\.\.|//=|\|=|\+=|-=|\*=|/=|%=|==|!=|<=|>=|//|[.\[\]{}(),:;|+\-*/%=<>?])
    )""", re.X)
KEYWORDS = {"if", "then", "elif", "else", "end", "as", "def", "reduce", "foreach", "try", "catch", "label", "import",
            "include", "and", "or", "not", "__loc__"}


def lex(src):
    toks = []
    i, n = 0, len(src)
    while i < n:
        if src[i].isspace():
            i += 1
            continue
        if src[i] == "#":
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        m = TOKEN.match(src, i)
        if not m or m.end() == i:
            raise JqSyntax(f"syntax error, unexpected INVALID_CHARACTER (Unix shell quoting issues?) at <top-level>, line 1:\n{src}")
        kind = m.lastgroup
        if kind == "str":
            parts, j = lex_string(src, m.start("str"))
            toks.append(("str", parts))
            i = j
            continue
        toks.append((kind, m.group(kind)))
        i = m.end()
    toks.append(("eof", None))
    return toks


def lex_string(src, start):
    """String literal with \\( interpolation ). Returns ([str | ('interp', src)], end index)."""
    i = start + 1
    parts, buf = [], []
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"':
            if buf or not parts:
                parts.append("".join(buf))
            return parts, i + 1
        if c == "\\":
            d = src[i + 1] if i + 1 < n else ""
            if d == "(":
                if buf:
                    parts.append("".join(buf))
                    buf = []
                depth, j = 1, i + 2
                while j < n and depth:
                    if src[j] == "(":
                        depth += 1
                    elif src[j] == ")":
                        depth -= 1
                    elif src[j] == '"':
                        _, j = lex_string(src, j)
                        continue
                    j += 1
                parts.append(("interp", src[i + 2:j - 1]))
                i = j
                continue
            esc = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}
            if d == "u":
                buf.append(chr(int(src[i + 2:i + 6], 16)))
                i += 6
                continue
            if d not in esc:
                raise JqSyntax(f"invalid escape at line 1, column {i + 1}")
            buf.append(esc[d])
            i += 2
            continue
        buf.append(c)
        i += 1
    raise JqSyntax("syntax error, unexpected end of file, unterminated string")


# ---------------------------------------------------------------------------- parser -> closures
# A compiled filter is f(input, env) -> iterator of outputs; path expressions also carry
# f.paths(input, env, prefix) -> iterator of (path, value), used by |=, del(), path() and paths(f).


def mk(fn, paths=None):
    if paths is not None:
        fn.paths = paths
    return fn


def node_paths(f, v, env, pre):
    pf = getattr(f, "paths", None)
    if pf is None:
        out = None
        try:
            for x in f(v, env):
                out = x
                break
        except JqError:
            pass
        raise JqError(f"Invalid path expression with result {_short(out)}")
    return pf(v, env, pre)


def const(x):
    return lambda v, env: iter([x])


class P:
    def __init__(self, src):
        self.toks = lex(src)
        self.i = 0
        self.src = src

    def peek(self, k=0):
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self, kind=None, val=None):
        t = self.toks[self.i]
        if (kind is not None and t[0] != kind) or (val is not None and t[1] != val):
            got = "$end" if t[0] == "eof" else (t[1] if isinstance(t[1], str) else "QQSTRING_START")
            raise JqSyntax(f"syntax error, unexpected {got if got != '$end' else 'end of file'} (Unix shell quoting issues?) at <top-level>, line 1:\n{self.src}")
        self.i += 1
        return t

    def at(self, kind, val=None):
        t = self.peek()
        return t[0] == kind and (val is None or t[1] == val)

    def at_op(self, *vals):
        t = self.peek()
        return t[0] == "op" and t[1] in vals

    def at_kw(self, *vals):
        t = self.peek()
        return t[0] == "ident" and t[1] in vals

    def parse(self):
        f = self.pipe()
        if not self.at("eof"):
            self.take("eof")
        return f

    # ---- pipe level: def, `as`, `|`
    def pipe(self):
        if self.at_kw("def"):
            name, params, body = self.funcdef()
            rest = self.pipe()

            def with_def(v, env):
                env2 = dict(env)
                env2[("fn", name, len(params))] = (params, body, env2)
                return rest(v, env2)

            def with_def_paths(v, env, pre):
                env2 = dict(env)
                env2[("fn", name, len(params))] = (params, body, env2)
                return node_paths(rest, v, env2, pre)
            return mk(with_def, with_def_paths)
        left = self.comma()
        if self.at_kw("as"):
            self.i += 1
            patterns = [self.pattern()]
            while self.at_op("?//"):
                self.i += 1
                patterns.append(self.pattern())
            self.take("op", "|")
            body = self.pipe()

            def bind(v, env):
                for x in left(v, env):
                    for k, pat in enumerate(patterns):
                        env2 = dict(env)
                        try:
                            destructure(pat, x, env2)
                            yield from body(v, env2)
                        except JqError:
                            if k == len(patterns) - 1:
                                raise
                            continue
                        break

            def bind_paths(v, env, pre):
                for x in left(v, env):
                    env2 = dict(env)
                    destructure(patterns[0], x, env2)
                    yield from node_paths(body, v, env2, pre)
            return mk(bind, bind_paths)
        if self.at_op("|"):
            self.i += 1
            right = self.pipe()

            def pipe(v, env):
                for x in left(v, env):
                    yield from right(x, env)

            def pipe_paths(v, env, pre):
                for p, x in node_paths(left, v, env, pre):
                    yield from node_paths(right, x, env, p)
            return mk(pipe, pipe_paths)
        return left

    def funcdef(self):
        self.take("ident", "def")
        name = self.take("ident")[1]
        params = []
        if self.at_op("("):
            self.i += 1
            while True:
                t = self.take()
                params.append(t[1])
                if self.at_op(";"):
                    self.i += 1
                    continue
                self.take("op", ")")
                break
        self.take("op", ":")
        body = self.pipe()
        self.take("op", ";")
        return name, params, body

    def pattern(self):
        if self.at("var"):
            return ("var", self.take()[1][1:])
        if self.at_op("["):
            self.i += 1
            items = []
            while not self.at_op("]"):
                items.append(self.pattern())
                if self.at_op(","):
                    self.i += 1
            self.take("op", "]")
            return ("array", items)
        if self.at_op("{"):
            self.i += 1
            items = []
            while not self.at_op("}"):
                if self.at("var"):
                    name = self.take()[1][1:]
                    sub = None
                    if self.at_op(":"):
                        self.i += 1
                        sub = self.pattern()
                    items.append((("const", name), ("var", name), sub))
                else:
                    t = self.take()
                    key = ("const", t[1]) if t[0] == "ident" else ("const", "".join(x for x in t[1] if isinstance(x, str)))
                    self.take("op", ":")
                    items.append((key, None, self.pattern()))
                if self.at_op(","):
                    self.i += 1
            self.take("op", "}")
            return ("object", items)
        raise JqSyntax("syntax error in destructuring pattern")

    def comma(self):
        left = self.alt()
        while self.at_op(","):
            self.i += 1
            right = self.alt()
            a, b = left, right
            left = mk((lambda a, b: lambda v, env: _chain(a(v, env), b(v, env)))(a, b),
                      (lambda a, b: lambda v, env, pre: _chain(node_paths(a, v, env, pre), node_paths(b, v, env, pre)))(a, b))
        return left

    def alt(self):
        left = self.assignment()
        if self.at_op("//"):
            self.i += 1
            right = self.alt()

            def alternative(v, env):
                got = False
                try:
                    for x in left(v, env):
                        if truthy(x):
                            got = True
                            yield x
                except JqError:
                    pass
                if not got:
                    yield from right(v, env)

            def alt_paths(v, env, pre):
                got = []
                try:
                    for p, x in node_paths(left, v, env, pre):
                        if truthy(x):
                            got.append((p, x))
                except JqError:
                    pass
                if got:
                    yield from got
                else:
                    yield from node_paths(right, v, env, pre)
            return mk(alternative, alt_paths)
        return left

    def assignment(self):
        left = self.or_()
        if self.at_op("=", "|=", "+=", "-=", "*=", "/=", "%=", "//="):
            op = self.take()[1]
            right = self.alt()
            return make_assign(op, left, right)
        return left

    def or_(self):
        left = self.and_()
        while self.at_kw("or"):
            self.i += 1
            right = self.and_()
            left = (lambda a, b: lambda v, env: (True if truthy(x) else truthy(y) for x in a(v, env)
                                                 for y in ([True] if truthy(x) else b(v, env))))(left, right)
        return left

    def and_(self):
        left = self.compare()
        while self.at_kw("and"):
            self.i += 1
            right = self.compare()
            left = (lambda a, b: lambda v, env: (truthy(y) if truthy(x) else False for x in a(v, env)
                                                 for y in (b(v, env) if truthy(x) else [False])))(left, right)
        return left

    def compare(self):
        left = self.additive()
        if self.at_op("==", "!=", "<", "<=", ">", ">="):
            op = self.take()[1]
            right = self.additive()
            fn = {"==": lambda a, b: cmp(a, b) == 0, "!=": lambda a, b: cmp(a, b) != 0, "<": lambda a, b: cmp(a, b) < 0,
                  "<=": lambda a, b: cmp(a, b) <= 0, ">": lambda a, b: cmp(a, b) > 0, ">=": lambda a, b: cmp(a, b) >= 0}[op]
            return lambda v, env: (fn(x, y) for y in right(v, env) for x in left(v, env))
        return left

    def additive(self):
        left = self.mult()
        while self.at_op("+", "-"):
            op = self.take()[1]
            right = self.mult()
            left = (lambda a, b, o: lambda v, env: (arith(o, x, y) for y in b(v, env) for x in a(v, env)))(left, right, op)
        return left

    def mult(self):
        left = self.unary()
        while self.at_op("*", "/", "%"):
            op = self.take()[1]
            right = self.unary()
            left = (lambda a, b, o: lambda v, env: (arith(o, x, y) for y in b(v, env) for x in a(v, env)))(left, right, op)
        return left

    def unary(self):
        if self.at_op("-"):
            self.i += 1
            e = self.postfix()
            return lambda v, env: (arith("-", 0, x) for x in e(v, env))
        return self.postfix()

    # ---- postfix: .a ."a" [..] [] ?
    def postfix(self):
        e = self.primary()
        while True:
            if self.at("field"):
                name = self.take()[1][1:]
                e = field_node(e, const(name))
                continue
            if self.at_op(".") and self.peek(1)[0] == "str":
                self.i += 1
                e = field_node(e, self.string_lit())
                continue
            if self.at_op(".") and self.peek(1) == ("op", "["):
                self.i += 1
                continue
            if self.at_op("["):
                e = self.bracket(e)
                continue
            if self.at_op("?"):
                self.i += 1
                inner = e
                e = mk((lambda f: lambda v, env: _try(f, v, env))(inner),
                       (lambda f: lambda v, env, pre: _try_paths(f, v, env, pre))(inner))
                continue
            if self.at_kw("as") or True:
                break
        return e

    def bracket(self, base):
        self.take("op", "[")
        if self.at_op("]"):
            self.i += 1
            return iter_node(base)
        if self.at_op(":"):
            self.i += 1
            hi = self.pipe()
            self.take("op", "]")
            return slice_node(base, const(None), hi)
        lo = self.pipe()
        if self.at_op(":"):
            self.i += 1
            hi = const(None) if self.at_op("]") else self.pipe()
            self.take("op", "]")
            return slice_node(base, lo, hi)
        self.take("op", "]")
        return field_node(base, lo)

    def string_lit(self):
        parts = self.take("str")[1]
        compiled = [("i", P(p[1]).parse()) if isinstance(p, tuple) else ("s", p) for p in parts]
        if all(k == "s" for k, _ in compiled):
            return const("".join(x for _, x in compiled))
        return lambda v, env: _interp(compiled, v, env, None)

    def primary(self):
        t = self.peek()
        k, val = t
        if k == "num":
            self.i += 1
            return const(float(val) if any(c in val for c in ".eE") else int(val))
        if k == "str":
            return self.string_lit()
        if k == "format":
            self.i += 1
            if self.at("str"):
                parts = self.take()[1]
                compiled = [("i", P(p[1]).parse()) if isinstance(p, tuple) else ("s", p) for p in parts]
                return lambda v, env: _interp(compiled, v, env, val)
            return lambda v, env: iter([apply_format(val, v)])
        if k == "field":
            self.i += 1
            return field_node(IDENT, const(val[1:]))
        if k == "var":
            self.i += 1
            name = val[1:]
            if name == "__loc__":
                return const({"file": "<stdin>", "line": 1})

            def var(v, env):
                if ("var", name) not in env:
                    raise JqSyntax(f"${name} is not defined at <top-level>, line 1:\n{self.src}")
                yield env[("var", name)]
            return var
        if k == "op":
            if val == ".":
                self.i += 1
                if self.at("str"):
                    return field_node(IDENT, self.string_lit())
                return IDENT
            if val == "..":
                self.i += 1
                return mk(lambda v, env: recurse_all(v), lambda v, env, pre: _recurse_paths(v, pre))
            if val == "(":
                self.i += 1
                e = self.pipe()
                self.take("op", ")")
                return e
            if val == "[":
                self.i += 1
                if self.at_op("]"):
                    self.i += 1
                    return lambda v, env: iter([[]])
                e = self.pipe()
                self.take("op", "]")
                return lambda v, env: iter([list(e(v, env))])
            if val == "{":
                return self.object()
        if k == "ident":
            if val == "if":
                return self.if_()
            if val == "try":
                self.i += 1
                body = self.postfix()
                handler = None
                if self.at_kw("catch"):
                    self.i += 1
                    handler = self.postfix()

                def tr(v, env):
                    try:
                        for x in body(v, env):
                            yield x
                    except JqError as e:
                        if handler is not None:
                            yield from handler(e.value if e.value is not None else e.msg, env)
                return mk(tr, lambda v, env, pre: _try_paths(body, v, env, pre))
            if val == "reduce":
                self.i += 1
                src = self.postfix()
                self.take("ident", "as")
                pat = self.pattern()
                self.take("op", "(")
                init = self.pipe()
                self.take("op", ";")
                upd = self.pipe()
                self.take("op", ")")

                def red(v, env):
                    for acc in init(v, env):
                        for x in src(v, env):
                            env2 = dict(env)
                            destructure(pat, x, env2)
                            res = list(upd(acc, env2))
                            acc = res[-1] if res else None
                        yield acc
                return red
            if val == "foreach":
                self.i += 1
                src = self.postfix()
                self.take("ident", "as")
                pat = self.pattern()
                self.take("op", "(")
                init = self.pipe()
                self.take("op", ";")
                upd = self.pipe()
                ext = None
                if self.at_op(";"):
                    self.i += 1
                    ext = self.pipe()
                self.take("op", ")")

                def fe(v, env):
                    for acc in init(v, env):
                        for x in src(v, env):
                            env2 = dict(env)
                            destructure(pat, x, env2)
                            for acc in upd(acc, env2):
                                if ext is None:
                                    yield acc
                                else:
                                    yield from ext(acc, env2)
                return fe
            if val == "label":
                self.i += 1
                name = self.take("var")[1][1:]
                self.take("op", "|")
                body = self.pipe()

                def lab(v, env):
                    env2 = dict(env)
                    tag = object()
                    env2[("label", name)] = tag
                    try:
                        yield from body(v, env2)
                    except BreakOut as b:
                        if b.label is not tag:
                            raise
                return lab
            if val == "break":
                self.i += 1
                name = self.take("var")[1][1:]

                def brk(v, env):
                    raise BreakOut(env[("label", name)])
                    yield
                return brk
            if val == "def":
                name, params, body = self.funcdef()
                rest = self.pipe()

                def with_def(v, env):
                    env2 = dict(env)
                    env2[("fn", name, len(params))] = (params, body, env2)
                    return rest(v, env2)
                return with_def
            if val in ("then", "elif", "else", "end", "as", "catch"):
                self.take("eof")
            self.i += 1
            args = []
            if self.at_op("("):
                self.i += 1
                while True:
                    args.append(self.pipe())
                    if self.at_op(";"):
                        self.i += 1
                        continue
                    self.take("op", ")")
                    break
            return make_call(val, args)
        self.take("op", "(")   # syntax error

    def object(self):
        self.take("op", "{")
        entries = []
        while not self.at_op("}"):
            t = self.peek()
            if t[0] == "var":
                self.i += 1
                name = t[1][1:]
                entries.append((const(name), (lambda n: lambda v, env: iter([env[("var", n)]]))(name)))
            elif t[0] == "ident" or t[0] == "kw":
                self.i += 1
                key = t[1]
                if self.at_op(":"):
                    self.i += 1
                    entries.append((const(key), self.objval()))
                else:
                    entries.append((const(key), (lambda kk: lambda v, env: iter([index(v, kk)]))(key)))
            elif t[0] == "str":
                key = self.string_lit()
                if self.at_op(":"):
                    self.i += 1
                    entries.append((key, self.objval()))
                else:
                    entries.append((key, (lambda kf: lambda v, env: (index(v, kk) for kk in kf(v, env)))(key)))
            elif t[0] == "format":
                self.i += 1
                entries.append(((lambda f: lambda v, env: iter([apply_format(f, v)]))(t[1]), self.objval()))
            elif t == ("op", "("):
                self.i += 1
                key = self.pipe()
                self.take("op", ")")
                self.take("op", ":")
                entries.append((key, self.objval()))
            elif t[0] == "num":
                raise JqSyntax(f"syntax error, unexpected LITERAL, expecting '}}' (Unix shell quoting issues?) at <top-level>, line 1:\n{self.src}")
            else:
                self.take("op", "}")
            if self.at_op(","):
                self.i += 1
        self.take("op", "}")

        def build(v, env):
            results = [{}]
            for kf, vf in entries:
                new = []
                for partial in results:
                    for key in kf(v, env):
                        if not isinstance(key, str):
                            raise JqError(f"Object keys must be strings")
                        for val in vf(v, env):
                            d = dict(partial)
                            d[key] = val
                            new.append(d)
                results = new
            return iter(results)
        return build

    def objval(self):
        left = self.alt()
        while self.at_op("|"):
            self.i += 1
            right = self.alt()
            left = (lambda a, b: lambda v, env: (y for x in a(v, env) for y in b(x, env)))(left, right)
        return left

    def if_(self):
        self.take("ident", "if")
        cond = self.pipe()
        self.take("ident", "then")
        then = self.pipe()
        elifs = []
        other = None
        while self.at_kw("elif"):
            self.i += 1
            c = self.pipe()
            self.take("ident", "then")
            elifs.append((c, self.pipe()))
        if self.at_kw("else"):
            self.i += 1
            other = self.pipe()
        self.take("ident", "end")
        chain = [(cond, then)] + elifs

        def run(v, env, k=0):
            if k >= len(chain):
                if other is None:
                    yield v
                else:
                    yield from other(v, env)
                return
            c, t = chain[k]
            for x in c(v, env):
                if truthy(x):
                    yield from t(v, env)
                else:
                    yield from run(v, env, k + 1)

        def run_paths(v, env, pre, k=0):
            if k >= len(chain):
                if other is None:
                    yield (list(pre), v)
                else:
                    yield from node_paths(other, v, env, pre)
                return
            c, t = chain[k]
            for x in c(v, env):
                if truthy(x):
                    yield from node_paths(t, v, env, pre)
                else:
                    yield from run_paths(v, env, pre, k + 1)
        return mk(run, run_paths)


IDENT = mk(lambda v, env: iter([v]), lambda v, env, pre: iter([(list(pre), v)]))


def field_node(base, keyf):
    def get(v, env):
        for x in base(v, env):
            for k in keyf(v, env):
                yield index(x, k)

    def paths(v, env, pre):
        for p, x in node_paths(base, v, env, pre):
            for k in keyf(v, env):
                if x is not None and not isinstance(x, (dict, list)):
                    index(x, k)
                yield (p + [k], index(x, k))
    return mk(get, paths)


def iter_node(base):
    def get(v, env):
        for x in base(v, env):
            yield from iterate(x)

    def paths(v, env, pre):
        for p, x in node_paths(base, v, env, pre):
            if isinstance(x, list):
                for i in range(len(x)):
                    yield (p + [i], x[i])
            elif isinstance(x, dict):
                for k in x:
                    yield (p + [k], x[k])
            elif x is None:
                continue
            else:
                raise JqError(f"Cannot iterate over {jtype(x)}")
    return mk(get, paths)


def slice_node(base, lo, hi):
    def get(v, env):
        for x in base(v, env):
            for a in lo(v, env):
                for b in hi(v, env):
                    yield slice_(x, a, b)

    def paths(v, env, pre):
        for p, x in node_paths(base, v, env, pre):
            for a in lo(v, env):
                for b in hi(v, env):
                    yield (p + [{"start": a, "end": b}], slice_(x, a, b))
    return mk(get, paths)


def _recurse_paths(v, pre):
    yield (list(pre), v)
    if isinstance(v, list):
        for i, x in enumerate(v):
            yield from _recurse_paths(x, pre + [i])
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from _recurse_paths(x, pre + [k])


def _try_paths(f, v, env, pre):
    try:
        for x in node_paths(f, v, env, pre):
            yield x
    except JqError:
        return


def paths_of(f, v, env):
    return [p for p, _ in node_paths(f, v, env, [])]


def _chain(a, b):
    yield from a
    yield from b


def _try(f, v, env):
    try:
        for x in f(v, env):
            yield x
    except JqError:
        return


def _interp(compiled, v, env, fmt):
    results = [""]
    for kind, x in compiled:
        if kind == "s":
            results = [r + x for r in results]
        else:
            new = []
            for r in results:
                for y in x(v, env):
                    if fmt:
                        piece = apply_format(fmt, y)
                    else:
                        piece = y if isinstance(y, str) else to_json_text(y)
                    new.append(r + piece)
            results = new
    return iter(results)


def destructure(pat, v, env):
    kind = pat[0]
    if kind == "var":
        env[("var", pat[1])] = v
    elif kind == "array":
        if v is not None and not isinstance(v, list):
            raise JqError(f"Cannot index {jtype(v)} with number")
        for i, p in enumerate(pat[1]):
            destructure(p, v[i] if v is not None and i < len(v) else None, env)
    else:
        for key, var, sub in pat[1]:
            k = key[1]
            val = index(v, k)
            if var is not None:
                env[("var", var[1])] = val
            if sub is not None:
                destructure(sub, val, env)


def iterate(x):
    if isinstance(x, list):
        return iter(x)
    if isinstance(x, dict):
        return iter(list(x.values()))
    raise JqError(f"Cannot iterate over {jtype(x)}" + (f" ({_short(x)})" if True else ""))


def index(x, k):
    if x is None:
        if isinstance(k, (str, int, float)) or k is None:
            return None
    if isinstance(k, str):
        if isinstance(x, dict):
            return x.get(k)
        raise JqError(f"Cannot index {jtype(x)} with \"{k}\"")
    if isinstance(k, (int, float)) and not isinstance(k, bool):
        if isinstance(x, list):
            i = int(math.floor(k))
            if i < 0:
                i += len(x)
            return x[i] if 0 <= i < len(x) else None
        raise JqError(f"Cannot index {jtype(x)} with number")
    if isinstance(k, dict) and isinstance(x, list):
        return slice_(x, k.get("start"), k.get("end"))
    if isinstance(k, list) and isinstance(x, list):
        return [i for i in range(len(x)) if x[i:i + len(k)] == k] if k else None
    raise JqError(f"Cannot index {jtype(x)} with {jtype(k)}")


def slice_(x, lo, hi):
    if x is None:
        return None
    if not isinstance(x, (list, str)):
        raise JqError(f"Cannot index {jtype(x)} with object")
    n = len(x)
    lo = 0 if lo is None else int(math.floor(lo))
    hi = n if hi is None else int(math.ceil(hi))
    return x[lo:hi]


def recurse_all(v):
    yield v
    if isinstance(v, list):
        for x in v:
            yield from recurse_all(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from recurse_all(x)


def arith(op, a, b):
    if op == "+":
        if a is None:
            return b
        if b is None:
            return a
        if _num(a) and _num(b):
            return _fix(a + b)
        if isinstance(a, str) and isinstance(b, str):
            return a + b
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        if isinstance(a, dict) and isinstance(b, dict):
            d = dict(a)
            d.update(b)
            return d
    elif op == "-":
        if _num(a) and _num(b):
            return _fix(a - b)
        if isinstance(a, list) and isinstance(b, list):
            return [x for x in a if x not in b]
    elif op == "*":
        if _num(a) and _num(b):
            return _fix(a * b)
        if isinstance(a, str) and _num(b):
            return a * int(b) if b > 0 else None
        if _num(a) and isinstance(b, str):
            return b * int(a) if a > 0 else None
        if isinstance(a, dict) and isinstance(b, dict):
            return deep_merge(a, b)
    elif op == "/":
        if _num(a) and _num(b):
            if b == 0:
                raise JqError(f"{num_text(a)} ({num_text(a)}) and {num_text(b)} ({num_text(b)}) cannot be divided because the divisor is zero")
            return _fix(a / b)
        if isinstance(a, str) and isinstance(b, str):
            return a.split(b) if b else list(a)
    elif op == "%":
        if _num(a) and _num(b):
            if int(b) == 0:
                raise JqError(f"{num_text(a)} ({num_text(a)}) and {num_text(b)} ({num_text(b)}) cannot be divided because the divisor is zero")
            r = abs(int(a)) % abs(int(b))
            return -r if int(a) < 0 else r
    word = {"+": "added", "-": "subtracted", "*": "multiplied", "/": "divided", "%": "divided"}[op]
    raise JqError(f"{jtype(a)} ({_short(a)}) and {jtype(b)} ({_short(b)}) cannot be {word}")


def _short(v):
    t = to_json_text(v)
    return t if len(t) <= 11 else t[:10] + "..."


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fix(v):
    if isinstance(v, float) and v == int(v) and abs(v) < 1e17:
        return int(v)
    return v


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------- paths & assignment

def getpath(v, path):
    for k in path:
        if v is None:
            return None
        v = index(v, k)
    return v


def setpath(v, path, new):
    if not path:
        return new
    k = path[0]
    if v is None:
        v = [] if isinstance(k, int) else {}
    if isinstance(k, str):
        if not isinstance(v, dict):
            raise JqError(f"Cannot index {jtype(v)} with \"{k}\"")
        d = dict(v)
        d[k] = setpath(v.get(k), path[1:], new)
        return d
    if isinstance(k, (int, float)):
        if not isinstance(v, list):
            raise JqError(f"Cannot index {jtype(v)} with number")
        lst = list(v)
        i = int(k)
        if i < 0:
            i += len(lst)
        while len(lst) <= i:
            lst.append(None)
        lst[i] = setpath(lst[i], path[1:], new)
        return lst
    raise JqError("Invalid path component")


def delpaths(v, paths):
    for p in sorted(paths, key=lambda p: [(0, x) if isinstance(x, (int, float)) else (1, x) for x in p], reverse=True):
        v = _delpath(v, p)
    return v


def _delpath(v, path):
    if v is None:
        return None
    if len(path) == 1:
        k = path[0]
        if isinstance(v, dict):
            d = dict(v)
            d.pop(k, None)
            return d
        if isinstance(v, list):
            lst = list(v)
            i = int(k)
            if -len(lst) <= i < len(lst):
                del lst[i]
            return lst
        raise JqError(f"Cannot delete field at object index of {jtype(v)}")
    k = path[0]
    child = index(v, k)
    if child is None:
        return v
    return setpath(v, [k], _delpath(child, path[1:]))


def make_assign(op, left, right):
    def run(v, env):
        if op == "|=":
            out = v
            for p in list(paths_of(left, v, env)):
                cur = getpath(out, p)
                res = list(right(cur, env))
                out = setpath(out, p, res[0]) if res else _delpath(out, p)
            yield out
            return
        for rv in right(v, env):
            out = v
            for p in list(paths_of(left, v, env)):
                if op == "=":
                    out = setpath(out, p, rv)
                elif op == "//=":
                    cur = getpath(out, p)
                    out = setpath(out, p, cur if truthy(cur) else rv)
                else:
                    out = setpath(out, p, arith(op[0], getpath(out, p), rv))
            yield out
    return run


# ---------------------------------------------------------------------------- builtins

def make_call(name, args):
    n = len(args)

    def call(v, env):
        fn = env.get(("fn", name, n))
        if fn is not None:
            params, body, defenv = fn
            env2 = _bind_args(fn, params, args, v, env)
            if env2 is None:
                for env3 in _value_param_envs(fn, params, args, v, env):
                    yield from body(v, env3)
                return
            yield from body(v, env2)
            return
        b = BUILTINS.get((name, n))
        if b is None:
            raise JqSyntax(f"{name}/{n} is not defined at <top-level>, line 1:")
        yield from b(v, env, *args)

    def paths(v, env, pre):
        fn = env.get(("fn", name, n))
        if fn is not None:
            params, body, defenv = fn
            env2 = _bind_args(fn, params, args, v, env)
            if env2 is None:
                for env3 in _value_param_envs(fn, params, args, v, env):
                    yield from node_paths(body, v, env3, pre)
                return
            yield from node_paths(body, v, env2, pre)
            return
        if (name, n) == ("select", 1):
            if any(truthy(x) for x in args[0](v, env)):
                yield (list(pre), v)
            return
        if (name, n) == ("recurse", 0):
            yield from _recurse_paths(v, pre)
            return
        if (name, n) == ("empty", 0):
            return
        if (name, n) == ("getpath", 1):
            for p in args[0](v, env):
                yield (list(pre) + list(p), getpath(v, p))
            return
        if (name, n) == ("first", 1):
            yield from list(_take(node_paths(args[0], v, env, pre), 1))
            return
        if (name, n) == ("last", 1):
            yield from list(node_paths(args[0], v, env, pre))[-1:]
            return
        if (name, n) in (("first", 0), ("last", 0)):
            yield (list(pre) + [0 if name == "first" else -1], index(v, 0 if name == "first" else -1))
            return
        if (name, n) == ("paths", 0):
            for p, x in _all_paths(v, list(pre)):
                yield (p, x)
            return
        raise JqError(f"Invalid path expression with result {_short(next(iter(call(v, env)), None))}")
    return mk(call, paths)


def _bind_args(fn, params, args, v, env):
    params_, body, defenv = fn
    env2 = dict(defenv)
    env2[("fn", fn_name(fn, env), len(params))] = fn
    if any(p.startswith("$") for p in params):
        return None
    for p, a in zip(params, args):
        env2[("fn", p, 0)] = ([], (lambda a_, e_: mk(lambda vv, ee: a_(vv, e_),
                                                      lambda vv, ee, pre: node_paths(a_, vv, e_, pre)))(a, env), env2)
    return env2


def fn_name(fn, env):
    for k, val in env.items():
        if val is fn and isinstance(k, tuple) and k[0] == "fn":
            return k[1]
    return "?"


def _value_param_envs(fn, params, args, v, env):
    params_, body, defenv = fn
    combos = [dict(defenv)]
    for p, a in zip(params, args):
        new = []
        for e in combos:
            if p.startswith("$"):
                for val in a(v, env):
                    e2 = dict(e)
                    e2[("var", p[1:])] = val
                    e2[("fn", p[1:], 0)] = ([], const(val), e2)
                    new.append(e2)
            else:
                e2 = dict(e)
                e2[("fn", p, 0)] = ([], (lambda a_, e_: lambda vv, ee: a_(vv, e_))(a, env), e2)
                new.append(e2)
        combos = new
    for e in combos:
        e[("fn", fn_name(fn, env), len(params))] = fn
        yield e


BUILTINS: dict = {}
PATH_BUILTINS: dict = {}


def builtin(name, n=0):
    def reg(fn):
        BUILTINS[(name, n)] = fn
        return fn
    return reg


def simple(name, fn):
    BUILTINS[(name, 0)] = lambda v, env: iter([fn(v)])


@builtin("empty")
def b_empty(v, env):
    return iter([])


@builtin("error", 0)
def b_error0(v, env):
    raise JqError(v if isinstance(v, str) else to_json_text(v) + " (not a string)", v)
    yield


@builtin("error", 1)
def b_error1(v, env, f):
    for m in f(v, env):
        raise JqError(m if isinstance(m, str) else to_json_text(m), m)
    yield


def _length(v):
    if v is None:
        return 0
    if isinstance(v, bool):
        raise JqError(f"boolean ({to_json_text(v)}) has no length")
    if _num(v):
        return _fix(abs(v))
    return len(v)


simple("length", _length)
simple("utf8bytelength", lambda v: len(v.encode()) if isinstance(v, str) else (_ for _ in ()).throw(JqError(f"{jtype(v)} ({to_json_text(v)}) only strings have UTF-8 byte length")))
simple("not", lambda v: not truthy(v))
simple("type", jtype)
simple("keys", lambda v: sorted(v) if isinstance(v, dict) else (list(range(len(v))) if isinstance(v, list) else _raise(f"{jtype(v)} ({to_json_text(v)}) has no keys")))
simple("keys_unsorted", lambda v: list(v) if isinstance(v, dict) else list(range(len(v))))
simple("add", lambda v: _add(list(v.values()) if isinstance(v, dict) else v))
simple("tostring", lambda v: v if isinstance(v, str) else to_json_text(v))
simple("tonumber", lambda v: v if _num(v) else _tonumber(v))
simple("ascii_downcase", lambda v: v.lower())
simple("ascii_upcase", lambda v: v.upper())
simple("sort", lambda v: sorted(v, key=sort_key) if isinstance(v, list) else _raise(f"{jtype(v)} ({to_json_text(v)}) cannot be sorted, as it is not an array"))
simple("unique", lambda v: _unique(v, lambda x: x))
simple("reverse", lambda v: v[::-1] if isinstance(v, (list, str)) else ([] if v is None else _raise(f"Cannot reverse {jtype(v)}")))
simple("min", lambda v: min(v, key=sort_key) if v else None)
simple("max", lambda v: max(v, key=sort_key) if v else None)
simple("floor", lambda v: _fix(float(math.floor(v))))
simple("ceil", lambda v: _fix(float(math.ceil(v))))
simple("round", lambda v: _fix(float(math.floor(v + 0.5)) if v >= 0 else -float(math.floor(-v + 0.5))))
simple("fabs", lambda v: _fix(abs(v)))
simple("abs", lambda v: _fix(abs(v)) if _num(v) else _raise(f"{jtype(v)} has no absolute value"))
simple("sqrt", lambda v: _fix(math.sqrt(v)))
simple("log", lambda v: math.log(v))
simple("log10", lambda v: math.log10(v))
simple("log2", lambda v: math.log2(v))
simple("exp", lambda v: math.exp(v))
simple("exp10", lambda v: 10.0 ** v)
simple("tojson", to_json_text)
simple("fromjson", lambda v: _fromjson(v))
simple("to_entries", lambda v: [{"key": k, "value": x} for k, x in v.items()] if isinstance(v, dict) else
       [{"key": i, "value": x} for i, x in enumerate(v)])
simple("from_entries", lambda v: {_entry_key(e): _entry_val(e) for e in v})
simple("flatten", lambda v: _flatten(v, 1e9))
simple("explode", lambda v: [ord(c) for c in v])
simple("implode", lambda v: "".join(chr(c) for c in v))
simple("ascii", lambda v: chr(v))
simple("isnan", lambda v: v != v)
simple("isinfinite", lambda v: v in (math.inf, -math.inf))
simple("infinite", lambda v: math.inf)
simple("nan", lambda v: math.nan)
simple("isnormal", lambda v: _num(v) and v == v and v not in (math.inf, -math.inf) and v != 0)
simple("null", lambda v: None)
simple("true", lambda v: True)
simple("false", lambda v: False)
simple("now", lambda v: 1792834867.0)
simple("input_line_number", lambda v: 0)
simple("tostream", lambda v: v)
simple("trim", lambda v: v.strip() if isinstance(v, str) else _raise(f"{jtype(v)} ({to_json_text(v)}) cannot be trimmed"))
simple("ltrim", lambda v: v.lstrip())
simple("rtrim", lambda v: v.rstrip())
simple("any", lambda v: any(truthy(x) for x in (v.values() if isinstance(v, dict) else v)))
simple("all", lambda v: all(truthy(x) for x in (v.values() if isinstance(v, dict) else v)))
simple("todate", lambda v: _todate(v))
simple("fromdate", lambda v: _fromdate(v))
simple("fromdateiso8601", lambda v: _fromdate(v))
simple("todateiso8601", lambda v: _todate(v))
simple("halt", lambda v: _raise("halt"))
simple("ascii", lambda v: chr(v))
simple("first", lambda v: v[0] if v else None)
simple("last", lambda v: v[-1] if v else None)
simple("transpose", lambda v: [[row[i] if i < len(row) else None for row in v] for i in range(max((len(r) for r in v), default=0))])


def _raise(msg):
    raise JqError(msg)


def _tonumber(v):
    if isinstance(v, str):
        try:
            x = float(v)
        except ValueError:
            raise JqError(f"Cannot parse '{v}' as JSON") from None
        return _fix(x) if ("." not in v and "e" not in v.lower()) else x
    raise JqError(f"{jtype(v)} ({to_json_text(v)}) cannot be parsed as a number")


def _fromjson(v):
    try:
        return json.loads(v)
    except ValueError:
        raise JqError(f"{v} (while parsing '{v}')") from None


def _entry_key(e):
    for k in ("key", "k", "name", "Name", "Key", "K"):
        if k in e:
            x = e[k]
            return x if isinstance(x, str) else to_json_text(x)
    return "null"


def _entry_val(e):
    for k in ("value", "v", "Value", "V"):
        if k in e:
            return e[k]
    return None


def _add(v):
    if v is None:
        return None
    out = None
    for x in v:
        out = arith("+", out, x) if out is not None else x
    return out


def _unique(v, key):
    seen = []
    out = []
    for x in sorted(v, key=lambda y: sort_key(key(y))):
        k = sort_key(key(x))
        if not seen or seen[-1] != k:
            seen.append(k)
            out.append(x)
    return out


def _flatten(v, depth):
    out = []
    for x in v:
        if isinstance(x, list) and depth > 0:
            out += _flatten(x, depth - 1)
        else:
            out.append(x)
    return out


def _todate(v):
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(v))


def _fromdate(v):
    import calendar
    import time
    return calendar.timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%SZ"))


def one(f, v, env):
    for x in f(v, env):
        return x
    return None


def reg1(name, fn):
    BUILTINS[(name, 1)] = fn


reg1("map", lambda v, env, f: iter([[y for x in iterate(v) for y in f(x, env)]]) if isinstance(v, list) else
     iter([{k: next(iter(f(x, env))) for k, x in v.items()}]) if isinstance(v, dict) else _raise(f"Cannot iterate over {jtype(v)}"))
reg1("map_values", lambda v, env, f: iter([{k: y for k, x in v.items() for y in list(f(x, env))[:1]}] if isinstance(v, dict) else
                                         [[y for x in v for y in list(f(x, env))[:1]]]))
reg1("select", lambda v, env, f: iter([v] if any(truthy(x) for x in f(v, env)) else []))
reg1("has", lambda v, env, f: (k in v if isinstance(v, dict) else (0 <= k < len(v)) for k in f(v, env)))
reg1("in", lambda v, env, f: (v in o if isinstance(o, dict) else (0 <= v < len(o)) for o in f(v, env)))
reg1("contains", lambda v, env, f: (_contains(v, b) for b in f(v, env)))
reg1("inside", lambda v, env, f: (_contains(b, v) for b in f(v, env)))
reg1("startswith", lambda v, env, f: (v.startswith(s) if isinstance(v, str) and isinstance(s, str) else _raise("startswith() requires string inputs") for s in f(v, env)))
reg1("endswith", lambda v, env, f: (v.endswith(s) if isinstance(v, str) and isinstance(s, str) else _raise("endswith() requires string inputs") for s in f(v, env)))
reg1("ltrimstr", lambda v, env, f: (v[len(s):] if isinstance(v, str) and isinstance(s, str) and v.startswith(s) else v for s in f(v, env)))
reg1("rtrimstr", lambda v, env, f: (v[:-len(s)] if isinstance(v, str) and isinstance(s, str) and s and v.endswith(s) else v for s in f(v, env)))
reg1("split", lambda v, env, f: ((v.split(s) if s else list(v)) if isinstance(v, str) else _raise("split input must be a string") for s in f(v, env)))
reg1("join", lambda v, env, f: (_join(v, s) for s in f(v, env)))
reg1("sort_by", lambda v, env, f: iter([sorted(v, key=lambda x: [sort_key(y) for y in f(x, env)])]))
reg1("group_by", lambda v, env, f: iter([_group_by(v, env, f)]))
reg1("unique_by", lambda v, env, f: iter([_unique(v, lambda x: [y for y in f(x, env)])]))
reg1("min_by", lambda v, env, f: iter([min(v, key=lambda x: [sort_key(y) for y in f(x, env)]) if v else None]))
reg1("max_by", lambda v, env, f: iter([max(v, key=lambda x: [sort_key(y) for y in f(x, env)]) if v else None]))
reg1("any", lambda v, env, f: iter([any(truthy(y) for x in iterate(v) for y in f(x, env))]))
reg1("all", lambda v, env, f: iter([all(truthy(y) for x in iterate(v) for y in f(x, env))]))
reg1("flatten", lambda v, env, f: (_flatten(v, d) for d in f(v, env)))
reg1("range", lambda v, env, f: (_fix(float(i)) for n in f(v, env) for i in range(int(math.ceil(n)))))
reg1("with_entries", lambda v, env, f: iter([{_entry_key(e): _entry_val(e) for x in
                                              ([{"key": k, "value": x} for k, x in v.items()] if isinstance(v, dict) else
                                               [{"key": i, "value": x} for i, x in enumerate(v)]) for e in f(x, env)}]))
reg1("test", lambda v, env, f: (bool(_re(r, v).search(v)) for r in f(v, env)))
reg1("match", lambda v, env, f: (m for r in f(v, env) for m in _matches(v, r, "")))
reg1("capture", lambda v, env, f: (_capture(m) for r in f(v, env) for m in _re_iter(v, r, "")[:1]))
reg1("scan", lambda v, env, f: (_scan_item(m) for r in f(v, env) for m in _re(r, v).finditer(v)))
reg1("splits", lambda v, env, f: (x for r in f(v, env) for x in _re(r, v).split(v)))
reg1("ascii_downcase", lambda v, env, f: iter([v.lower()]))
reg1("tostring", lambda v, env, f: iter([v if isinstance(v, str) else to_json_text(v)]))
reg1("indices", lambda v, env, f: (_indices(v, s) for s in f(v, env)))
reg1("index", lambda v, env, f: ((_indices(v, s) or [None])[0] for s in f(v, env)))
reg1("rindex", lambda v, env, f: ((_indices(v, s) or [None])[-1] for s in f(v, env)))
reg1("first", lambda v, env, f: iter(list(_take(f(v, env), 1))))
reg1("last", lambda v, env, f: iter(list(f(v, env))[-1:]))
reg1("isempty", lambda v, env, f: iter([not list(_take(f(v, env), 1))]))
reg1("recurse", lambda v, env, f: _recurse(v, env, f))
reg1("paths", lambda v, env, f: (p for p, x in _all_paths(v, []) if any(truthy(y) for y in f(x, env))))
reg1("getpath", lambda v, env, f: (getpath(v, p) for p in f(v, env)))
reg1("delpaths", lambda v, env, f: (delpaths(v, ps) for ps in f(v, env)))
reg1("del", lambda v, env, f: iter([delpaths(v, list(paths_of(f, v, env)))]))
reg1("path", lambda v, env, f: paths_of(f, v, env))
reg1("strftime", lambda v, env, f: (__import__("time").strftime(fmt, __import__("time").gmtime(v)) for fmt in f(v, env)))
reg1("tojson", lambda v, env, f: iter([to_json_text(v)]))
reg1("error", b_error1)
reg1("splits", lambda v, env, f: (x for r in f(v, env) for x in _re(r, v).split(v)))
reg1("walk", lambda v, env, f: iter([_walk(v, env, f)]))
reg1("debug", lambda v, env, f: iter([v]))
reg1("add", lambda v, env, f: iter([_add(list(f(v, env)))]))
reg1("abs", lambda v, env, f: iter([abs(v)]))


def b2(name, fn):
    BUILTINS[(name, 2)] = fn


b2("range", lambda v, env, a, b: (_fix(float(i)) for x in a(v, env) for y in b(v, env) for i in _frange(x, y, 1)))
b2("limit", lambda v, env, n, f: (x for k in n(v, env) for x in _take(f(v, env), int(k))))
b2("sub", lambda v, env, r, s: iter([_sub(v, env, r, s, False, "")]))
b2("gsub", lambda v, env, r, s: iter([_sub(v, env, r, s, True, "")]))
b2("test", lambda v, env, r, fl: (bool(_re(x, v, y).search(v)) for x in r(v, env) for y in fl(v, env)))
b2("match", lambda v, env, r, fl: (m for x in r(v, env) for y in fl(v, env) for m in _matches(v, x, y)))
b2("capture", lambda v, env, r, fl: (_capture(m) for x in r(v, env) for y in fl(v, env) for m in _re_iter(v, x, y)[:1]))
b2("split", lambda v, env, r, fl: (_re(x, v, y).split(v) for x in r(v, env) for y in fl(v, env)))
b2("setpath", lambda v, env, p, x: (setpath(v, pp, xx) for pp in p(v, env) for xx in x(v, env)))
b2("while", lambda v, env, c, u: _while(v, env, c, u))
b2("until", lambda v, env, c, u: _until(v, env, c, u))
b2("nth", lambda v, env, n, f: (list(_take(f(v, env), int(k) + 1))[int(k)] for k in n(v, env)))
b2("splits", lambda v, env, r, fl: (x for rr in r(v, env) for y in fl(v, env) for x in _re(rr, v, y).split(v)))
b2("pow", lambda v, env, a, b: (_fix(float(x) ** y) for x in a(v, env) for y in b(v, env)))
b2("atan2", lambda v, env, a, b: (math.atan2(x, y) for x in a(v, env) for y in b(v, env)))
BUILTINS[("range", 3)] = lambda v, env, a, b, c: (_fix(float(i)) for x in a(v, env) for y in b(v, env) for z in c(v, env) for i in _frange(x, y, z))
BUILTINS[("sub", 3)] = lambda v, env, r, s, fl: iter([_sub(v, env, r, s, "g" in next(iter(fl(v, env))), next(iter(fl(v, env))))])
BUILTINS[("gsub", 3)] = lambda v, env, r, s, fl: iter([_sub(v, env, r, s, True, next(iter(fl(v, env))))])


def _while(v, env, c, u):
    while any(truthy(x) for x in c(v, env)):
        yield v
        res = list(u(v, env))
        if not res:
            return
        v = res[0]


def _until(v, env, c, u):
    while not any(truthy(x) for x in c(v, env)):
        res = list(u(v, env))
        if not res:
            return
        v = res[0]
    yield v


def _frange(a, b, step):
    x = a
    if step > 0:
        while x < b:
            yield x
            x += step
    elif step < 0:
        while x > b:
            yield x
            x += step


def _take(it, n):
    if n <= 0:
        return
    for i, x in enumerate(it):
        yield x
        if i + 1 >= n:
            return


def _join(v, s):
    parts = []
    for x in v:
        if x is None:
            parts.append("")
        elif isinstance(x, (str,)):
            parts.append(x)
        elif isinstance(x, bool) or _num(x):
            parts.append(to_json_text(x))
        else:
            raise JqError(f"Cannot join with {jtype(x)}")
    return s.join(parts)


def _group_by(v, env, f):
    out = []
    keyed = sorted(((list(sort_key(y) for y in f(x, env)), x) for x in v), key=lambda p: p[0])
    for k, x in keyed:
        if out and out[-1][0] == k:
            out[-1][1].append(x)
        else:
            out.append((k, [x]))
    return [g for _, g in out]


def _contains(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in a and _contains(a[k], v) for k, v in b.items())
    if isinstance(a, list) and isinstance(b, list):
        return all(any(_contains(x, y) for x in a) for y in b)
    if isinstance(a, str) and isinstance(b, str):
        return b in a
    if jtype(a) == jtype(b):
        return a == b
    raise JqError(f"{jtype(a)} ({to_json_text(a)}) and {jtype(b)} ({to_json_text(b)}) cannot have their containment checked")


def _indices(v, s):
    if isinstance(v, str):
        out, i = [], v.find(s)
        while i >= 0 and s:
            out.append(i)
            i = v.find(s, i + 1)
        return out
    if isinstance(v, list):
        if isinstance(s, list):
            return [i for i in range(len(v)) if s and v[i:i + len(s)] == s]
        return [i for i, x in enumerate(v) if x == s]
    return []


def _re(pat, v, flags=""):
    if not isinstance(v, str):
        raise JqError(f"{jtype(v)} ({to_json_text(v)}) cannot be matched, as it is not a string")
    fl = 0
    if flags and "i" in flags:
        fl |= re.I
    if flags and "x" in flags:
        fl |= re.X
    pat = re.sub(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>", r"(?P<\1>", pat)
    return re.compile(pat, fl)


def _re_iter(v, pat, flags):
    rx = _re(pat, v, flags)
    return list(rx.finditer(v)) if flags and "g" in flags else list(rx.finditer(v))[:1] or []


def _matches(v, pat, flags):
    rx = _re(pat, v, flags)
    ms = list(rx.finditer(v)) if flags and "g" in flags else list(rx.finditer(v))[:1]
    names = {i: n for n, i in rx.groupindex.items()}
    for m in ms:
        caps = []
        for gi in range(1, (rx.groups or 0) + 1):
            caps.append({"offset": m.start(gi) if m.group(gi) is not None else -1,
                         "length": len(m.group(gi)) if m.group(gi) is not None else 0,
                         "string": m.group(gi), "name": names.get(gi)})
        yield {"offset": m.start(), "length": m.end() - m.start(), "string": m.group(0), "captures": caps}


def _capture(m):
    return {n: m.group(n) for n in m.re.groupindex}


def _scan_item(m):
    if m.re.groups:
        return [m.group(i) for i in range(1, m.re.groups + 1)]
    return m.group(0)


def _sub(v, env, r, s, glob, flags):
    pat = next(iter(r(v, env)))
    rx = _re(pat, v, flags)
    out = []
    last = 0
    for m in rx.finditer(v):
        out.append(v[last:m.start()])
        cap = {n: m.group(n) for n in rx.groupindex}
        rep = next(iter(s(cap, env)))
        out.append(rep if isinstance(rep, str) else to_json_text(rep))
        last = m.end()
        if not glob:
            break
    out.append(v[last:])
    return "".join(out)


def _recurse(v, env, f):
    yield v
    for x in f(v, env):
        yield from _recurse(x, env, f)


def _all_paths(v, prefix):
    if isinstance(v, dict):
        for k, x in v.items():
            yield prefix + [k], x
            yield from _all_paths(x, prefix + [k])
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield prefix + [i], x
            yield from _all_paths(x, prefix + [i])


def _walk(v, env, f):
    if isinstance(v, list):
        v = [_walk(x, env, f) for x in v]
    elif isinstance(v, dict):
        v = {k: _walk(x, env, f) for k, x in v.items()}
    res = list(f(v, env))
    return res[-1] if res else None


def _index_by(rows, env, f):
    out = {}
    for row in rows:
        for k in f(row, env):
            out[k if isinstance(k, str) else to_json_text(k)] = row
    return out


BUILTINS[("INDEX", 1)] = lambda v, env, f: iter([_index_by(iterate(v), env, f)])
BUILTINS[("INDEX", 2)] = lambda v, env, src, f: iter([_index_by(src(v, env), env, f)])
BUILTINS[("IN", 1)] = lambda v, env, src: iter([any(cmp(v, x) == 0 for x in src(v, env))])
BUILTINS[("IN", 2)] = lambda v, env, src, s2: iter([any(cmp(x, y) == 0 for x in src(v, env) for y in s2(v, env))])
BUILTINS[("pick", 1)] = lambda v, env, f: iter([_pick(v, env, f)])
BUILTINS[("repeat", 1)] = lambda v, env, f: _repeat(v, env, f)
BUILTINS[("add", 1)] = lambda v, env, f: iter([_add(list(f(v, env)))])
BUILTINS[("combinations", 0)] = lambda v, env: _combos(v)
BUILTINS[("tojson", 0)] = lambda v, env: iter([to_json_text(v)])
BUILTINS[("getpath", 1)] = lambda v, env, f: (getpath(v, p) for p in f(v, env))
BUILTINS[("toarray", 0)] = lambda v, env: iter([v if isinstance(v, list) else [v]])
BUILTINS[("splits", 1)] = lambda v, env, f: (x for r in f(v, env) for x in _re(r, v).split(v))
BUILTINS[("ascii", 0)] = lambda v, env: iter([chr(v)])
BUILTINS[("significand", 0)] = lambda v, env: iter([math.frexp(v)[0] * 2])
BUILTINS[("debug", 1)] = lambda v, env, f: _debug(v, env)


def _pick(v, env, f):
    out = None
    for p in paths_of(f, v, env):
        out = setpath(out, p, getpath(v, p))
    return out


def _repeat(v, env, f):
    while True:
        yield v
        res = list(f(v, env))
        if not res:
            return
        v = res[0]


def _combos(v):
    if not v:
        yield []
        return
    for x in v[0]:
        for rest in _combos(v[1:]):
            yield [x] + rest


BUILTINS[("recurse", 0)] = lambda v, env: recurse_all(v)
BUILTINS[("paths", 0)] = lambda v, env: (p for p, _ in _all_paths(v, []))
BUILTINS[("leaf_paths", 0)] = lambda v, env: (p for p, x in _all_paths(v, []) if not isinstance(x, (list, dict)))
BUILTINS[("values", 0)] = lambda v, env: iter([v] if v is not None else [])
BUILTINS[("nulls", 0)] = lambda v, env: iter([v] if v is None else [])
BUILTINS[("booleans", 0)] = lambda v, env: iter([v] if isinstance(v, bool) else [])
BUILTINS[("numbers", 0)] = lambda v, env: iter([v] if _num(v) else [])
BUILTINS[("strings", 0)] = lambda v, env: iter([v] if isinstance(v, str) else [])
BUILTINS[("arrays", 0)] = lambda v, env: iter([v] if isinstance(v, list) else [])
BUILTINS[("objects", 0)] = lambda v, env: iter([v] if isinstance(v, dict) else [])
BUILTINS[("iterables", 0)] = lambda v, env: iter([v] if isinstance(v, (list, dict)) else [])
BUILTINS[("scalars", 0)] = lambda v, env: iter([v] if not isinstance(v, (list, dict)) else [])
BUILTINS[("env", 0)] = lambda v, env: iter([env.get(("var", "ENV"), {})])
BUILTINS[("input", 0)] = lambda v, env: _input(env)
BUILTINS[("inputs", 0)] = lambda v, env: _inputs(env)
BUILTINS[("debug", 0)] = lambda v, env: _debug(v, env)
BUILTINS[("not", 0)] = lambda v, env: iter([not truthy(v)])
BUILTINS[("to_number", 0)] = BUILTINS[("tonumber", 0)]
BUILTINS[("splits", 1)] = lambda v, env, f: (x for r in f(v, env) for x in _re(r, v).split(v))
BUILTINS[("halt_error", 0)] = lambda v, env: (_ for _ in ()).throw(JqError(v if isinstance(v, str) else to_json_text(v), v))
BUILTINS[("ascii", 0)] = lambda v, env: iter([chr(v)])
BUILTINS[("tostream", 0)] = lambda v, env: _tostream(v, [])
BUILTINS[("input_filename", 0)] = lambda v, env: iter([env.get(("var", "__filename"))])
BUILTINS[("todate", 0)] = lambda v, env: iter([_todate(v)])


def _tostream(v, path):
    if isinstance(v, (list, dict)) and v:
        items = list(v.items()) if isinstance(v, dict) else list(enumerate(v))
        for k, x in items:
            yield from _tostream(x, path + [k])
        yield [path + [items[-1][0]]]
    else:
        yield [path, v]


def _debug(v, env):
    sink = env.get(("var", "__stderr"))
    if sink is not None:
        sink(f'["DEBUG:",{to_json_text(v)}]\n')
    yield v


def _input(env):
    feed = env.get(("var", "__inputs"))
    if feed is None:
        raise JqError("No more inputs")
    try:
        yield next(feed)
    except StopIteration:
        raise JqError("No more inputs") from None


def _inputs(env):
    feed = env.get(("var", "__inputs"))
    if feed is None:
        return
    yield from feed


# path-mode support: .foo .[n] .[] paths compose through pipes
def path_mode(f):
    return f


def apply_format(name, v):
    if name == "@text":
        return v if isinstance(v, str) else to_json_text(v)
    if name == "@json":
        return to_json_text(v)
    if name == "@csv":
        if not isinstance(v, list):
            raise JqError(f"{jtype(v)} ({to_json_text(v)}) cannot be csv-formatted, only an array can be")
        cells = []
        for x in v:
            if isinstance(x, str):
                cells.append('"' + x.replace('"', '""') + '"')
            elif x is None:
                cells.append("")
            elif isinstance(x, bool):
                cells.append("true" if x else "false")
            elif _num(x):
                cells.append(num_text(x))
            else:
                raise JqError(f"{jtype(x)} ({to_json_text(x)}) is not valid in a csv row")
        return ",".join(cells)
    if name == "@tsv":
        if not isinstance(v, list):
            raise JqError(f"{jtype(v)} ({to_json_text(v)}) cannot be tsv-formatted, only an array can be")
        cells = []
        for x in v:
            if isinstance(x, str):
                cells.append(x.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"))
            elif x is None:
                cells.append("")
            else:
                cells.append(to_json_text(x))
        return "\t".join(cells)
    if name == "@sh":
        def q(x):
            if isinstance(x, str):
                return "'" + x.replace("'", "'\\''") + "'"
            if isinstance(x, (list, dict)):
                raise JqError(f"{jtype(x)} ({to_json_text(x)}) can not be escaped for shell")
            return to_json_text(x)
        return " ".join(q(x) for x in v) if isinstance(v, list) else q(v)
    if name == "@base64":
        import base64
        return base64.b64encode((v if isinstance(v, str) else to_json_text(v)).encode()).decode()
    if name == "@base64d":
        import base64
        return base64.b64decode(v.encode() + b"=" * (-len(v) % 4)).decode("utf-8", "replace")
    if name == "@uri":
        from urllib.parse import quote
        return quote(v if isinstance(v, str) else to_json_text(v), safe="-_.~")
    if name == "@html":
        s = v if isinstance(v, str) else to_json_text(v)
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#39;").replace('"', "&quot;")
    raise JqSyntax(f"{name} is not a valid format")


def compile_filter(src: str):
    return P(src).parse()


# ---------------------------------------------------------------------------- JSON stream parsing

def parse_stream(text: str):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    out = []
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            v, j = dec.raw_decode(text, i)
        except json.JSONDecodeError as e:
            line = text.count("\n", 0, e.pos) + 1
            m = re.match(r"[A-Za-z0-9_.+-]+", text[e.pos:])
            end = e.pos + (m.end() if m else 1)
            col = end - (text.rfind("\n", 0, e.pos) + 1) + 1
            what = "Invalid numeric literal" if m else "Invalid literal"
            if text[e.pos:e.pos + 1] in "]}":
                what = "Unmatched '" + text[e.pos] + "'"
            raise JqError(f"jq: parse error: {what} at line {line}, column {col}") from None
        out.append(_ints(v))
        i = j
    return out


def _ints(v):
    if isinstance(v, float) and v == int(v) and abs(v) < 1e17 and False:
        return int(v)
    if isinstance(v, list):
        return [_ints(x) for x in v]
    if isinstance(v, dict):
        return {k: _ints(x) for k, x in v.items()}
    return v


@util("jq")
def u_jq(ctx):
    a = ctx.argv[1:]
    raw = compact = slurp = null_input = exit_status = sort_keys = raw_input = join = tab = ascii_ = False
    color = None
    indent = 2
    named = {}
    filt = None
    files = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "--arg" and i + 2 < len(a):
            named[a[i + 1]] = a[i + 2]
            i += 3
            continue
        if x == "--argjson" and i + 2 < len(a):
            try:
                named[a[i + 1]] = json.loads(a[i + 2])
            except ValueError:
                ctx.e(f"jq: Invalid JSON text passed to --argjson\n")
                return 2
            i += 3
            continue
        if x in ("--slurpfile", "--rawfile") and i + 2 < len(a):
            try:
                text = yield from ctx.text(a[i + 2])
            except FSError as e:
                ctx.e(f"jq: error: Could not open {a[i + 2]}: {e.msg}\n")
                return 2
            named[a[i + 1]] = parse_stream(text) if x == "--slurpfile" else text
            i += 3
            continue
        if x == "--indent" and i + 1 < len(a):
            indent = int(a[i + 1])
            i += 2
            continue
        if x.startswith("--"):
            m = {"--raw-output": "r", "--compact-output": "c", "--slurp": "s", "--null-input": "n",
                 "--exit-status": "e", "--sort-keys": "S", "--raw-input": "R", "--join-output": "j", "--tab": "t",
                 "--ascii-output": "a", "--color-output": "C", "--monochrome-output": "M", "--raw-output0": "r"}
            if x == "--version":
                ctx.w("jq-1.7.1\n")
                return 0
            if x == "--help":
                ctx.w("Usage:\tjq [OPTIONS] FILTER [FILES...]\n\tjq [OPTIONS] --args FILTER [ARGUMENTS...]\n")
                return 0
            if x not in m:
                ctx.e(f"jq: Unknown option: {x}\nUse jq --help for help with command-line options,\nor see the jq manpage, or online docs  at https://jqlang.github.io/jq\n")
                return 2
            flags = m[x]
        elif x.startswith("-") and len(x) > 1 and filt is None or (x.startswith("-") and len(x) > 1 and x[1] in "rcsneSRjaCM"):
            flags = x[1:]
        else:
            if filt is None:
                filt = x
            else:
                files.append(x)
            i += 1
            continue
        for c in flags:
            if c == "r":
                raw = True
            elif c == "c":
                compact = True
            elif c == "s":
                slurp = True
            elif c == "n":
                null_input = True
            elif c == "e":
                exit_status = True
            elif c == "S":
                sort_keys = True
            elif c == "R":
                raw_input = True
            elif c == "j":
                join = raw = True
            elif c == "t":
                tab = True
            elif c == "a":
                ascii_ = True
            elif c == "C":
                color = True
            elif c == "M":
                color = False
            else:
                ctx.e(f"jq: Unknown option: -{c}\nUse jq --help for help with command-line options,\nor see the jq manpage, or online docs  at https://jqlang.github.io/jq\n")
                return 2
        i += 1
    if filt is None:
        filt = "."
    if color is None:
        color = ctx.stdout_is_tty() and "NO_COLOR" not in ctx.env
    try:
        f = compile_filter(filt)
    except JqSyntax as e:
        ctx.e(f"jq: error: {e}\njq: 1 compile error\n")
        return 3
    texts = []
    try:
        if files:
            for p in files:
                texts.append((yield from ctx.text(p)))
        elif not null_input or True:
            if not null_input or not ctx.stdin_is_tty():
                data = yield from ctx.read_fd(0) if not (null_input and ctx.stdin_is_tty()) else b""
                texts.append(data.decode("utf-8", "surrogateescape"))
    except FSError as e:
        ctx.e(f"jq: error: Could not open {files[0] if files else '-'}: {e.msg}\n")
        return 2
    try:
        if raw_input:
            lines = []
            for t in texts:
                lines += t.split("\n")[:-1] if t.endswith("\n") else t.split("\n") if t else []
            inputs = ["\n".join(lines) + ("\n" if lines else "")] if slurp else lines
        else:
            values = []
            for t in texts:
                values += parse_stream(t)
            inputs = [values] if slurp else values
    except JqError as e:
        ctx.e(e.msg + "\n")
        return 2
    env = {("var", k): v for k, v in named.items()}
    env[("var", "ENV")] = dict(ctx.env)
    env[("var", "__stderr")] = ctx.e
    env[("var", "__named")] = {"positional": [], "named": named}
    feed = iter(inputs)
    env[("var", "__inputs")] = feed
    outputs_any = False
    last = None
    status = 0
    roots = [None] if null_input else feed
    for root in roots:
        try:
            for out in f(root, env):
                outputs_any = True
                last = out
                if raw and isinstance(out, str):
                    ctx.w(out + ("" if join else "\n"))
                else:
                    ctx.w(dumps(out, indent, compact, sort_keys, color, ascii_, 0, tab) + ("" if join else "\n"))
        except JqError as e:
            msg = e.msg
            if e.value is not None and not isinstance(e.value, str):
                msg = to_json_text(e.value) + " (not a string)"
            where = "<unknown>" if null_input else f"{files[-1] if files else '<stdin>'}:{_line_count(texts)}"
            ctx.e(f"jq: error (at {where}): {msg}\n")
            status = 5
        except JqSyntax as e:
            ctx.e(f"jq: error: {e}\njq: 1 compile error\n")
            return 3
        except BreakOut:
            ctx.e("jq: error (at <unknown>): break\n")
            status = 5
        except RecursionError:
            ctx.e("jq: error: stack overflow\n")
            status = 5
        yield from ctx.maybe_flush()
    if exit_status and status == 0:
        if not outputs_any:
            return 4
        return 0 if truthy(last) else 1
    return status


def _line_count(texts):
    if not texts:
        return 0
    t = texts[-1]
    return t.count("\n")

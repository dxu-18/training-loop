"""Bound expressions: typed trees (after analysis) that compile to Python closures f(row, ctx).

`row` is the tuple of the current query level (FROM columns, or a group's keys and aggregate values); ctx.outer
holds the rows of the enclosing query levels for correlated subqueries. Every node can be compared structurally
(.key(), for GROUP BY matching) and printed back as SQL the way ruleutils.c does (.deparse(), for EXPLAIN and \\d).
"""

from __future__ import annotations

from . import types as T
from .errors import PgError


# Var indexes are positions in the query level's full row; while compiling for a partial row (a FROM item that
# starts at column B, for a pushed-down filter or a join condition) SHIFT holds B.
SHIFT = [0]


def compile_at(e, base: int):
    saved = SHIFT[0]
    SHIFT[0] = base
    try:
        return e.compile()
    finally:
        SHIFT[0] = saved


class BE:
    kind = "?"

    def children(self):
        return ()

    def key(self):
        return (self.kind, id(self))

    def compile(self):
        raise NotImplementedError(self.kind)

    def deparse(self, dc) -> str:
        return "?"

    def walk(self):
        yield self
        for ch in self.children():
            if ch is not None:
                yield from ch.walk()


class DeparseCtx:
    """How to print: qualify column names? planned (constant casts folded, as EXPLAIN shows)?"""

    def __init__(self, qualify=False, planned=False, top=True, names=None):
        self.qualify = qualify
        self.planned = planned
        self.names = names or {}


# ---------------------------------------------------------------- leaves

class Const(BE):
    __slots__ = ("value", "pos")
    kind = "const"

    def __init__(self, value, typ, pos=None):
        self.value, self.type, self.pos = value, typ, pos

    def key(self):
        return ("const", self.type.name, T.hash_key(self.value, self.type))

    def compile(self):
        v = self.value
        return lambda r, c: v

    def deparse(self, dc):
        return deparse_const(self.value, self.type)


def deparse_const(v, t: T.Type, label=True) -> str:
    if v is None:
        return "NULL" if t.name == "unknown" or not label else f"NULL::{t.format()}"
    n = t.name
    s = T.output(v, t)
    if n == "int4":
        if not s.startswith("-"):
            return s
        return f"'{s}'::integer"
    if n == "numeric":
        if s[:1].isdigit() and any(ch in s for ch in ".eE"):
            return s + (f"::{t.format()}" if t.mod is not None else "")
        return f"'{s}'::{t.format()}"
    if n == "bool":
        return "true" if v else "false"
    q = "'" + s.replace("'", "''") + "'"
    if "\\" in s:
        q = "E" + q.replace("\\", "\\\\")
    if n == "unknown":
        return q
    return f"{q}::{t.format()}"


class Var(BE):
    __slots__ = ("level", "index", "rte", "col", "orig")
    kind = "var"

    def __init__(self, level, index, typ, rte=None, col=None, orig=None):
        self.level, self.index, self.type, self.rte, self.col, self.orig = level, index, typ, rte, col, orig

    def key(self):
        return ("var", self.level, self.index)

    def compile(self):
        if self.level == 0:
            i = self.index - SHIFT[0]
            return lambda r, c: r[i]
        i = self.index
        k = -self.level
        return lambda r, c: c.outer[k][i]

    def deparse(self, dc):
        if self.orig is not None:
            return self.orig.deparse(dc)
        if self.col is None:
            return "?"
        name = quote_ident(self.col)
        # EXPLAIN always qualifies a reference to an outer query's column (ruleutils.c get_parameter)
        if self.rte and (dc.qualify or (dc.planned and self.level > 0)):
            return f"{quote_ident(self.rte)}.{name}"
        return name


class Param(BE):
    __slots__ = ("n",)
    kind = "param"

    def __init__(self, n, typ):
        self.n, self.type = n, typ

    def key(self):
        return ("param", self.n)

    def compile(self):
        i = self.n - 1

        def f(r, c):
            if i >= len(c.params):
                raise PgError(f"there is no parameter ${i + 1}", "42P02")
            return c.params[i]
        return f

    def deparse(self, dc):
        return f"${self.n}"


class CaseTest(BE):
    __slots__ = ()
    kind = "casetest"

    def __init__(self, typ):
        self.type = typ

    def key(self):
        return ("casetest",)

    def compile(self):
        return lambda r, c: c.case_val[-1]

    def deparse(self, dc):
        return "CASE_TEST"


class SQLValue(BE):
    __slots__ = ("name", "precision", "fn")
    kind = "sqlvalue"

    def __init__(self, name, typ, fn, precision=None):
        self.name, self.type, self.fn, self.precision = name, typ, fn, precision

    def key(self):
        return ("sqlvalue", self.name)

    def compile(self):
        fn = self.fn
        return lambda r, c: fn(c)

    def deparse(self, dc):
        return self.name.upper() if self.name != "user" else "CURRENT_USER"


# ---------------------------------------------------------------- calls

class Func(BE):
    """A function call: impl(args list, ctx) -> value. strict: NULL in any argument -> NULL without calling."""
    __slots__ = ("name", "args", "impl", "strict", "volatile", "srf", "display", "variadic", "argtypes", "pos",
                 "cast_from")
    kind = "func"

    def __init__(self, name, args, typ, impl, strict=True, volatile=False, srf=False, display=None,
                 argtypes=None, pos=None):
        self.name, self.args, self.type, self.impl, self.strict = name, args, typ, impl, strict
        self.volatile, self.srf = volatile, srf
        self.display = display
        self.variadic = False
        self.argtypes = argtypes
        self.pos = pos
        self.cast_from = None

    def children(self):
        return self.args

    def key(self):
        return ("func", self.name, self.type.name, tuple(a.key() for a in self.args)) if not self.volatile else \
            ("func", id(self))

    def compile(self):
        impl = self.impl
        fs = [a.compile() for a in self.args]
        n = len(fs)
        if self.srf:
            raise PgError("set-returning functions are not allowed here", "0A000")
        if self.strict:
            if n == 1:
                f0 = fs[0]

                def f(r, c):
                    a = f0(r, c)
                    if a is None:
                        return None
                    return impl([a], c)
                return f
            if n == 2:
                f0, f1 = fs

                def f(r, c):
                    a = f0(r, c)
                    if a is None:
                        return None
                    b = f1(r, c)
                    if b is None:
                        return None
                    return impl([a, b], c)
                return f

            def f(r, c):
                vals = []
                for g in fs:
                    v = g(r, c)
                    if v is None:
                        return None
                    vals.append(v)
                return impl(vals, c)
            return f
        return lambda r, c: impl([g(r, c) for g in fs], c)

    def deparse(self, dc):
        name = self.display or self.name
        args = ", ".join(a.deparse(dc) for a in self.args)
        return f"{name}({args})"


class Op(BE):
    __slots__ = ("op", "args", "impl", "pos", "strict")
    kind = "op"

    def __init__(self, op, args, typ, impl, pos=None, strict=True):
        self.op, self.args, self.type, self.impl, self.pos, self.strict = op, args, typ, impl, pos, strict

    def children(self):
        return self.args

    def key(self):
        return ("op", self.op, self.type.name, tuple(a.key() for a in self.args))

    def compile(self):
        impl = self.impl
        if len(self.args) == 1:
            f0 = self.args[0].compile()

            def f(r, c):
                a = f0(r, c)
                return None if a is None else impl(a)
            return f
        f0, f1 = self.args[0].compile(), self.args[1].compile()
        if not self.strict:
            return lambda r, c: impl(f0(r, c), f1(r, c))

        def f(r, c):
            a = f0(r, c)
            if a is None:
                return None
            b = f1(r, c)
            if b is None:
                return None
            return impl(a, b)
        return f

    def deparse(self, dc):
        if len(self.args) == 1:
            return f"({self.op} {self.args[0].deparse(dc)})"
        return f"({self.args[0].deparse(dc)} {self.op} {self.args[1].deparse(dc)})"


class Bool(BE):
    __slots__ = ("op", "args")
    kind = "bool"

    def __init__(self, op, args):
        self.op, self.args, self.type = op, args, T.BOOL

    def children(self):
        return self.args

    def key(self):
        return ("bool", self.op, tuple(a.key() for a in self.args))

    def compile(self):
        fs = [a.compile() for a in self.args]
        if self.op == "not":
            f0 = fs[0]

            def f(r, c):
                v = f0(r, c)
                return None if v is None else not v
            return f
        if self.op == "and":
            def f(r, c):
                null = False
                for g in fs:
                    v = g(r, c)
                    if v is None:
                        null = True
                    elif not v:
                        return False
                return None if null else True
            return f

        def f(r, c):
            null = False
            for g in fs:
                v = g(r, c)
                if v is None:
                    null = True
                elif v:
                    return True
            return None if null else False
        return f

    def deparse(self, dc):
        if self.op == "not":
            return f"(NOT {self.args[0].deparse(dc)})"
        return "(" + f" {self.op.upper()} ".join(a.deparse(dc) for a in self.args) + ")"


class Case(BE):
    __slots__ = ("arg", "whens", "default")
    kind = "case"

    def __init__(self, arg, whens, default, typ):
        self.arg, self.whens, self.default, self.type = arg, whens, default, typ

    def children(self):
        out = [self.arg] if self.arg is not None else []
        for c, r in self.whens:
            out += [c, r]
        if self.default is not None:
            out.append(self.default)
        return out

    def key(self):
        return ("case", self.arg.key() if self.arg else None,
                tuple((c.key(), r.key()) for c, r in self.whens), self.default.key() if self.default else None)

    def compile(self):
        whens = [(c.compile(), r.compile()) for c, r in self.whens]
        dflt = self.default.compile() if self.default is not None else (lambda r, c: None)
        if self.arg is None:
            def f(r, c):
                for cond, res in whens:
                    if cond(r, c):
                        return res(r, c)
                return dflt(r, c)
            return f
        argf = self.arg.compile()

        def f(r, c):
            c.case_val.append(argf(r, c))
            hit = None
            try:
                for cond, res in whens:
                    if cond(r, c):
                        hit = res
                        break
            finally:
                c.case_val.pop()
            return hit(r, c) if hit is not None else dflt(r, c)
        return f

    def deparse(self, dc):
        parts = ["CASE"]
        if self.arg is not None:
            parts.append(self.arg.deparse(dc))
        for cond, res in self.whens:
            cs = cond.deparse(dc)
            if self.arg is not None and isinstance(cond, Op) and isinstance(cond.args[0], CaseTest):
                cs = cond.args[1].deparse(dc)
            parts.append(f"WHEN {cs} THEN {res.deparse(dc)}")
        if self.default is not None:
            parts.append(f"ELSE {self.default.deparse(dc)}")
        parts.append("END")
        return " ".join(parts)


class Coalesce(BE):
    __slots__ = ("args",)
    kind = "coalesce"

    def __init__(self, args, typ):
        self.args, self.type = args, typ

    def children(self):
        return self.args

    def key(self):
        return ("coalesce", tuple(a.key() for a in self.args))

    def compile(self):
        fs = [a.compile() for a in self.args]

        def f(r, c):
            for g in fs:
                v = g(r, c)
                if v is not None:
                    return v
            return None
        return f

    def deparse(self, dc):
        return "COALESCE(" + ", ".join(a.deparse(dc) for a in self.args) + ")"


class MinMax(BE):
    __slots__ = ("op", "args", "lt")
    kind = "minmax"

    def __init__(self, op, args, typ, lt):
        self.op, self.args, self.type, self.lt = op, args, typ, lt

    def children(self):
        return self.args

    def key(self):
        return ("minmax", self.op, tuple(a.key() for a in self.args))

    def compile(self):
        fs = [a.compile() for a in self.args]
        lt = self.lt
        greatest = self.op == "greatest"

        def f(r, c):
            best = None
            for g in fs:
                v = g(r, c)
                if v is None:
                    continue
                if best is None or (lt(best, v) if greatest else lt(v, best)):
                    best = v
            return best
        return f

    def deparse(self, dc):
        return f"{self.op.upper()}(" + ", ".join(a.deparse(dc) for a in self.args) + ")"


class NullIf(BE):
    __slots__ = ("args", "eq")
    kind = "nullif"

    def __init__(self, args, typ, eq):
        self.args, self.type, self.eq = args, typ, eq

    def children(self):
        return self.args

    def key(self):
        return ("nullif", tuple(a.key() for a in self.args))

    def compile(self):
        f0, f1 = self.args[0].compile(), self.args[1].compile()
        eq = self.eq

        def f(r, c):
            a = f0(r, c)
            b = f1(r, c)
            if a is not None and b is not None and eq(a, b):
                return None
            return a
        return f

    def deparse(self, dc):
        return f"NULLIF({self.args[0].deparse(dc)}, {self.args[1].deparse(dc)})"


class NullTest(BE):
    __slots__ = ("arg", "negated")
    kind = "nulltest"

    def __init__(self, arg, negated):
        self.arg, self.negated, self.type = arg, negated, T.BOOL

    def children(self):
        return (self.arg,)

    def key(self):
        return ("nulltest", self.negated, self.arg.key())

    def compile(self):
        f0 = self.arg.compile()
        row = self.arg.type.name == "record"
        if self.negated:
            if row:
                def f(r, c):
                    v = f0(r, c)
                    return v is not None and all(x is not None for x in v)
                return f
            return lambda r, c: f0(r, c) is not None
        if row:
            def f(r, c):
                v = f0(r, c)
                return v is None or all(x is None for x in v)
            return f
        return lambda r, c: f0(r, c) is None

    def deparse(self, dc):
        return f"({self.arg.deparse(dc)} IS {'NOT ' if self.negated else ''}NULL)"


class BoolTest(BE):
    __slots__ = ("arg", "what", "negated")
    kind = "booltest"

    def __init__(self, arg, what, negated):
        self.arg, self.what, self.negated, self.type = arg, what, negated, T.BOOL

    def children(self):
        return (self.arg,)

    def key(self):
        return ("booltest", self.what, self.negated, self.arg.key())

    def compile(self):
        f0 = self.arg.compile()
        want = {"true": True, "false": False, "unknown": None}[self.what]
        neg = self.negated

        def f(r, c):
            v = f0(r, c)
            res = v is want if want is None else (v is not None and bool(v) == want)
            return not res if neg else res
        return f

    def deparse(self, dc):
        return f"({self.arg.deparse(dc)} IS {'NOT ' if self.negated else ''}{self.what.upper()})"


class Distinct(BE):
    __slots__ = ("args", "eq", "negated")
    kind = "distinct"

    def __init__(self, args, eq, negated):
        self.args, self.eq, self.negated, self.type = args, eq, negated, T.BOOL

    def children(self):
        return self.args

    def key(self):
        return ("distinct", self.negated, tuple(a.key() for a in self.args))

    def compile(self):
        f0, f1 = self.args[0].compile(), self.args[1].compile()
        eq = self.eq
        neg = self.negated

        def f(r, c):
            a, b = f0(r, c), f1(r, c)
            if a is None or b is None:
                d = not (a is None and b is None)
            else:
                d = not eq(a, b)
            return not d if neg else d
        return f

    def deparse(self, dc):
        return f"({self.args[0].deparse(dc)} IS {'NOT ' if self.negated else ''}DISTINCT FROM " \
               f"{self.args[1].deparse(dc)})"


class ScalarArrayOp(BE):
    """x op ANY (array) / x op ALL (array); also x IN (list) as = ANY."""
    __slots__ = ("op", "use_or", "args", "impl", "from_in", "negated_in")
    kind = "saop"

    def __init__(self, op, use_or, left, right, impl, from_in=False):
        self.op, self.use_or, self.args, self.impl = op, use_or, [left, right], impl
        self.type = T.BOOL
        self.from_in = from_in
        self.negated_in = False

    def children(self):
        return self.args

    def key(self):
        return ("saop", self.op, self.use_or, tuple(a.key() for a in self.args))

    def compile(self):
        f0, f1 = self.args[0].compile(), self.args[1].compile()
        impl = self.impl
        use_or = self.use_or

        def f(r, c):
            a = f0(r, c)
            arr = f1(r, c)
            if arr is None:
                return None
            if not arr:
                return not use_or
            if a is None:
                return None
            null = False
            for x in T.array_flat(arr):
                if x is None:
                    null = True
                    continue
                v = impl(a, x)
                if use_or and v:
                    return True
                if not use_or and not v:
                    return False
            if null:
                return None
            return not use_or
        return f

    def deparse(self, dc):
        return f"({self.args[0].deparse(dc)} {self.op} {'ANY' if self.use_or else 'ALL'} " \
               f"({self.args[1].deparse(dc)}))"


class InList(BE):
    """x IN (a, b, c) with non-constant items (evaluated as ORs of =)."""
    __slots__ = ("args", "eqs", "negated")
    kind = "inlist"

    def __init__(self, arg, items, eqs, negated):
        self.args, self.eqs, self.negated, self.type = [arg] + items, eqs, negated, T.BOOL

    def children(self):
        return self.args

    def key(self):
        return ("inlist", self.negated, tuple(a.key() for a in self.args))

    def compile(self):
        f0 = self.args[0].compile()
        items = [(a.compile(), eq) for a, eq in zip(self.args[1:], self.eqs)]
        neg = self.negated

        def f(r, c):
            a = f0(r, c)
            if a is None:
                return None
            null = False
            for g, eq in items:
                b = g(r, c)
                if b is None:
                    null = True
                elif eq(a, b):
                    return False if neg else True
            if null:
                return None
            return True if neg else False
        return f

    def deparse(self, dc):
        items = ", ".join(a.deparse(dc) for a in self.args[1:])
        return f"({self.args[0].deparse(dc)} {'NOT ' if self.negated else ''}IN ({items}))"


class Cast(BE):
    __slots__ = ("arg", "fn", "explicit", "style", "pos")
    kind = "cast"

    def __init__(self, arg, typ, fn, explicit=False, style="::", pos=None):
        self.arg, self.type, self.fn, self.explicit, self.style, self.pos = arg, typ, fn, explicit, style, pos

    def children(self):
        return (self.arg,)

    def key(self):
        return ("cast", self.type.name, self.type.mod, self.arg.key())

    def compile(self):
        f0 = self.arg.compile()
        fn = self.fn

        def f(r, c):
            v = f0(r, c)
            return None if v is None else fn(v, c)
        return f

    def deparse(self, dc):
        a = self.arg
        if dc.planned and isinstance(a, Const):
            return deparse_const(a.value if a.value is None else None, self.type) if a.value is None else \
                f"{deparse_const(a.value, a.type, label=False) if a.type.name == 'unknown' else _fold(self)}"
        inner = a.deparse(dc)
        if not self.explicit and not dc.planned and isinstance(a, Const):
            return f"{inner}::{self.type.format()}"
        if isinstance(a, (Var, Const)) and not inner.startswith("("):
            return f"({inner})::{self.type.format()}"
        return f"({inner})::{self.type.format()}"


def _fold(cast: "Cast") -> str:
    try:
        v = cast.fn(cast.arg.value, None)
    except Exception:   # noqa: BLE001
        return f"({cast.arg.deparse(DeparseCtx())})::{cast.type.format()}"
    return deparse_const(v, cast.type)


class ArrayExpr(BE):
    __slots__ = ("items",)
    kind = "array"

    def __init__(self, items, typ):
        self.items, self.type = items, typ

    def children(self):
        return self.items

    def key(self):
        return ("array", tuple(a.key() for a in self.items))

    def compile(self):
        fs = [a.compile() for a in self.items]
        return lambda r, c: [g(r, c) for g in fs]

    def deparse(self, dc):
        return "ARRAY[" + ", ".join(a.deparse(dc) for a in self.items) + "]"


class RowExpr(BE):
    __slots__ = ("items", "names")
    kind = "row"

    def __init__(self, items, names=None):
        self.items, self.type, self.names = items, T.RECORD, names

    def children(self):
        return self.items

    def key(self):
        return ("row", tuple(a.key() for a in self.items))

    def compile(self):
        fs = [a.compile() for a in self.items]
        names = self.names
        types = [a.type for a in self.items]
        return lambda r, c: T.Row([g(r, c) for g in fs], names, types)

    def deparse(self, dc):
        return "ROW(" + ", ".join(a.deparse(dc) for a in self.items) + ")"


class FieldSelect(BE):
    __slots__ = ("arg", "index", "name")
    kind = "fieldselect"

    def __init__(self, arg, index, name, typ):
        self.arg, self.index, self.name, self.type = arg, index, name, typ

    def children(self):
        return (self.arg,)

    def key(self):
        return ("fieldselect", self.index, self.arg.key())

    def compile(self):
        f0 = self.arg.compile()
        i = self.index

        def f(r, c):
            v = f0(r, c)
            return None if v is None else v[i]
        return f

    def deparse(self, dc):
        return f"({self.arg.deparse(dc)}).{quote_ident(self.name)}"


class Subscript(BE):
    __slots__ = ("arg", "subs")
    kind = "subscript"

    def __init__(self, arg, subs, typ):
        self.arg, self.subs, self.type = arg, subs, typ     # subs: [(lo, hi, is_slice)]

    def children(self):
        out = [self.arg]
        for lo, hi, _ in self.subs:
            out += [x for x in (lo, hi) if x is not None]
        return out

    def key(self):
        return ("subscript", self.arg.key(), tuple((lo.key() if lo else None, hi.key() if hi else None, s)
                                                   for lo, hi, s in self.subs))

    def compile(self):
        f0 = self.arg.compile()
        subs = [(lo.compile() if lo else None, hi.compile() if hi else None, s) for lo, hi, s in self.subs]
        any_slice = any(s for _, _, s in subs)

        def f(r, c):
            v = f0(r, c)
            if v is None:
                return None
            if not any_slice:
                for lo, _, _ in subs:
                    i = lo(r, c)
                    if i is None or not isinstance(v, list) or i < 1 or i > len(v):
                        return None
                    v = v[i - 1]
                return v
            return _slice(v, [(lo(r, c) if lo else None, hi(r, c) if hi else None, s) for lo, hi, s in subs])
        return f

    def deparse(self, dc):
        out = self.arg.deparse(dc)
        for lo, hi, s in self.subs:
            if s:
                out += f"[{lo.deparse(dc) if lo else ''}:{hi.deparse(dc) if hi else ''}]"
            else:
                out += f"[{lo.deparse(dc)}]"
        return out


def _slice(v, subs):
    if not subs:
        return v
    (lo, hi, s), rest = subs[0], subs[1:]
    if not s:
        lo, hi = lo, lo
    lo = 1 if lo is None else lo
    hi = len(v) if hi is None else hi
    if lo is None or hi is None:
        return None
    lo = max(lo, 1)
    hi = min(hi, len(v))
    if lo > hi:
        return []
    out = v[lo - 1:hi]
    if rest:
        out = [_slice(x, rest) if isinstance(x, list) else x for x in out]
    return out


class SubPlan(BE):
    """A subquery in an expression: exists / expr (scalar) / any / all / array / multiexpr."""
    __slots__ = ("sublink", "query", "test", "op_impl", "correlated", "outer_refs", "pos", "negated_in", "name",
                 "plan_id", "hashable", "label")
    kind = "subplan"

    def __init__(self, sublink, query, typ, test=None, op_impl=None, correlated=True, pos=None):
        self.sublink, self.query, self.type, self.test, self.op_impl = sublink, query, typ, test, op_impl
        self.correlated = correlated
        self.outer_refs = []
        self.pos = pos
        self.negated_in = False
        self.name = None
        self.plan_id = None
        self.hashable = False
        self.label = None        # set by EXPLAIN's planner: "SubPlan 2", "hashed SubPlan 1", "$0"

    def children(self):
        return [self.test] if self.test is not None else []

    def key(self):
        return ("subplan", id(self))

    def compile(self):
        from .executor import run_subplan
        node = self
        testf = self.test.compile() if self.test is not None else None
        shift = SHIFT[0]
        return lambda r, c: run_subplan(node, r, c, testf, shift)

    def deparse(self, dc):
        if dc.planned and self.label:
            return self.label if self.label.startswith("$") else f"({self.label})"
        if not self.correlated and self.sublink == "expr" and dc.planned:
            return f"${(self.plan_id or 1) - 1}"
        if self.sublink == "exists":
            return f"(SubPlan {self.plan_id or 1})" if dc.planned else "EXISTS (...)"
        return f"(SubPlan {self.plan_id or 1})"


# ---------------------------------------------------------------- aggregates, windows

class Aggref(BE):
    __slots__ = ("name", "args", "distinct", "order", "filter", "agg", "direct", "slot", "star", "pos", "argtypes",
                 "level")
    kind = "aggref"

    def __init__(self, name, args, typ, agg, distinct=False, order=(), filt=None, direct=(), star=False, pos=None,
                 argtypes=None):
        self.name, self.args, self.type, self.agg = name, args, typ, agg
        self.distinct, self.order, self.filter, self.direct, self.star = distinct, list(order), filt, list(direct), star
        self.slot = None
        self.pos = pos
        self.argtypes = argtypes
        self.level = 0

    def children(self):
        out = list(self.args) + list(self.direct) + [o[0] for o in self.order]
        if self.filter is not None:
            out.append(self.filter)
        return out

    def key(self):
        return ("aggref", self.name, self.type.name, self.distinct, self.star, tuple(a.key() for a in self.args),
                tuple((o[0].key(), o[1], o[2]) for o in self.order),
                self.filter.key() if self.filter is not None else None, tuple(d.key() for d in self.direct))

    def compile(self):
        if self.slot is None:
            raise PgError("aggregate function calls are not allowed here", "42803")
        i = self.slot
        return lambda r, c: r[i]

    def deparse(self, dc):
        if self.star:
            inner = "*"
        else:
            inner = ("DISTINCT " if self.distinct else "") + ", ".join(a.deparse(dc) for a in self.args)
        if self.direct:
            s = f"{self.name}({', '.join(d.deparse(dc) for d in self.direct)}) WITHIN GROUP (ORDER BY " + \
                ", ".join(o[0].deparse(dc) for o in self.order) + ")"
        else:
            if self.order:
                inner += " ORDER BY " + ", ".join(o[0].deparse(dc) + (" DESC" if o[1] else "") for o in self.order)
            s = f"{self.name}({inner})"
        if self.filter is not None:
            s += f" FILTER (WHERE {self.filter.deparse(dc)})"
        return s


class WindowFunc(BE):
    __slots__ = ("name", "args", "win", "filter", "wf", "slot", "agg", "pos", "star", "distinct")
    kind = "windowfunc"

    def __init__(self, name, args, typ, win, wf=None, agg=None, filt=None, pos=None, star=False):
        self.name, self.args, self.type, self.win, self.wf, self.agg = name, args, typ, win, wf, agg
        self.filter = filt
        self.slot = None
        self.pos = pos
        self.star = star
        self.distinct = False

    def children(self):
        out = list(self.args)
        if self.filter is not None:
            out.append(self.filter)
        return out

    def key(self):
        return ("windowfunc", self.name, id(self))

    def compile(self):
        if self.slot is None:
            raise PgError("window functions are not allowed here", "42P20")
        i = self.slot
        return lambda r, c: r[i]

    def deparse(self, dc):
        inner = "*" if self.star else ", ".join(a.deparse(dc) for a in self.args)
        return f"{self.name}({inner}) OVER (?)"


class GroupingFunc(BE):
    __slots__ = ("args", "refs", "slot")
    kind = "grouping"

    def __init__(self, args):
        self.args, self.type = args, T.INT4
        self.refs = []
        self.slot = None

    def children(self):
        return self.args

    def key(self):
        return ("grouping", tuple(a.key() for a in self.args))

    def compile(self):
        if self.slot is None:
            raise PgError("arguments to GROUPING must be grouping expressions of the associated query level",
                          "42803")
        i = self.slot
        refs = self.refs

        def f(r, c):
            present = r[i]
            out = 0
            for k in refs:
                out = (out << 1) | (0 if k in present else 1)
            return out
        return f

    def deparse(self, dc):
        return "GROUPING(" + ", ".join(a.deparse(dc) for a in self.args) + ")"


class SetToDefault(BE):
    __slots__ = ("pos",)
    kind = "default"

    def __init__(self, typ, pos=None):
        self.type, self.pos = typ, pos

    def key(self):
        return ("default", id(self))

    def compile(self):
        raise PgError("DEFAULT is not allowed in this context", "42601", pos=self.pos)

    def deparse(self, dc):
        return "DEFAULT"


class SrfRef(BE):
    """A set-returning function in a SELECT list, replaced by its value for the expanded row."""
    __slots__ = ("slot", "func")
    kind = "srfref"

    def __init__(self, slot, func):
        self.slot, self.func, self.type = slot, func, func.type

    def key(self):
        return ("srfref", self.slot)

    def compile(self):
        i = self.slot
        return lambda r, c: r[i]

    def deparse(self, dc):
        return self.func.deparse(dc)


# ---------------------------------------------------------------- helpers

_SAFE_IDENT = None


def quote_ident(name: str) -> str:
    from .lexer import KEYWORDS
    if name and (name[0].islower() or name[0] == "_") and all(ch.islower() or ch.isdigit() or ch in "_$"
                                                              for ch in name) and \
            KEYWORDS.get(name) not in ("R", "C", "T"):
        return name
    return '"' + name.replace('"', '""') + '"'


def conjuncts(e: BE | None) -> list:
    if e is None:
        return []
    if isinstance(e, Bool) and e.op == "and":
        out = []
        for a in e.args:
            out += conjuncts(a)
        return out
    return [e]


def make_and(parts: list):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return Bool("and", parts)


def vars_of(e: BE, level: int = 0):
    """Var nodes that point at query level `level` relative to e (subplans' outer references included)."""
    out = []

    def visit(node, depth):
        if node is None:
            return
        if isinstance(node, Var):
            if node.level == depth:
                out.append(node)
            return
        if isinstance(node, SubPlan):
            if node.test is not None:
                visit(node.test, depth)
            for v in node.outer_refs:
                if v.level == depth + 1:
                    out.append(Var(v.level - 1, v.index, v.type, v.rte, v.col))
            return
        for ch in node.children():
            visit(ch, depth)
    visit(e, level)
    return out


def has_volatile(e: BE) -> bool:
    return any(isinstance(n, Func) and n.volatile for n in e.walk())

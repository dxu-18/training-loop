"""Parse trees back to SQL text (for view definitions shown by \\d+ and pg_views), laid out like pg_get_viewdef's
pretty mode: " SELECT a,\\n    b\\n   FROM t\\n  WHERE ..." """

from __future__ import annotations

from . import ast as A
from .expr import quote_ident

_PREC = {"or": 1, "and": 2, "not": 3, "is": 4, "cmp": 5, "like": 6, "op": 7, "add": 8, "mul": 9, "pow": 10,
         "unary": 11, "post": 12}


def _q(s):
    return "'" + s.replace("'", "''") + "'"


def _type(t: A.TypeName):
    names = {"int4": "integer", "int8": "bigint", "int2": "smallint", "float8": "double precision",
             "float4": "real", "bool": "boolean", "bpchar": "character", "varchar": "character varying",
             "timestamptz": "timestamp with time zone", "timestamp": "timestamp without time zone"}
    n = names.get(t.names[-1], ".".join(t.names))
    if t.mods:
        n += "(" + ",".join(str(m.value) for m in t.mods) + ")"
    return n + "[]" * t.array


def expr(e, parent=0):
    if e is None:
        return ""
    k = type(e).__name__
    if k == "Const":
        if e.kind == "str":
            return _q(e.value)
        if e.kind == "null":
            return "NULL"
        if e.kind == "bool":
            return "true" if e.value else "false"
        return str(e.value)
    if k == "ColumnRef":
        return ".".join("*" if n == "*" else quote_ident(n) for n in e.names)
    if k == "Star":
        return "*"
    if k == "Param":
        return f"${e.n}"
    if k == "TypedLiteral":
        return f"{_q(e.value)}::{_type(e.typ)}"
    if k == "Cast":
        return f"{expr(e.arg, 12)}::{_type(e.typ)}"
    if k == "Op":
        if e.left is None:
            return f"{e.op}{expr(e.right, 11)}"
        p = _op_prec(e.op)
        s = f"{expr(e.left, p)} {e.op} {expr(e.right, p + 1)}"
        return f"({s})" if p < parent else s
    if k == "BoolOp":
        if e.op == "not":
            s = f"NOT {expr(e.args[0], 3)}"
            return f"({s})" if parent > 3 else s
        p = 1 if e.op == "or" else 2
        s = f" {e.op.upper()} ".join(expr(a, p + 1) for a in e.args)
        return f"({s})" if p < parent else s
    if k == "NullTest":
        s = f"{expr(e.arg, 5)} IS {'NOT ' if e.negated else ''}NULL"
        return f"({s})" if parent > 4 else s
    if k == "BoolTest":
        return f"{expr(e.arg, 5)} IS {'NOT ' if e.negated else ''}{e.what.upper()}"
    if k == "Distinct":
        return f"{expr(e.left, 5)} IS {'NOT ' if e.negated else ''}DISTINCT FROM {expr(e.right, 5)}"
    if k == "InList":
        return f"{expr(e.arg, 7)} {'NOT ' if e.negated else ''}IN ({', '.join(expr(x) for x in e.items)})"
    if k == "Between":
        return f"{expr(e.arg, 7)} {'NOT ' if e.negated else ''}BETWEEN {expr(e.lo, 7)} AND {expr(e.hi, 7)}"
    if k == "Like":
        kw = {"like": "LIKE", "ilike": "ILIKE", "similar": "SIMILAR TO"}[e.kind]
        s = f"{expr(e.arg, 7)} {'NOT ' if e.negated else ''}{kw} {expr(e.pattern, 7)}"
        if e.escape is not None:
            s += f" ESCAPE {expr(e.escape)}"
        return s
    if k == "FuncCall":
        name = ".".join(e.name)
        if e.special == "extract":
            return f"EXTRACT({e.args[0].value} FROM {expr(e.args[1])})"
        if e.special == "position":
            return f"POSITION({expr(e.args[1])} IN {expr(e.args[0])})"
        if e.special == "substring_from":
            s = f"SUBSTRING({expr(e.args[0])} FROM {expr(e.args[1])}"
            if len(e.args) > 2:
                s += f" FOR {expr(e.args[2])}"
            return s + ")"
        if e.special in ("coalesce", "greatest", "least", "nullif"):
            name = e.special.upper()
        inner = "*" if e.star else ("DISTINCT " if e.distinct else "") + ", ".join(expr(a) for a in e.args)
        if e.order_by and not e.within_group:
            inner += " ORDER BY " + ", ".join(sortby(s) for s in e.order_by)
        s = f"{name}({inner})"
        if e.within_group:
            s += " WITHIN GROUP (ORDER BY " + ", ".join(sortby(x) for x in e.order_by) + ")"
        if e.filter is not None:
            s += f" FILTER (WHERE {expr(e.filter)})"
        if e.over is not None:
            s += " OVER " + window(e.over)
        return s
    if k == "SQLValue":
        return e.name.upper()
    if k == "Case":
        parts = ["CASE"]
        if e.arg is not None:
            parts.append(expr(e.arg))
        for c, r in e.whens:
            parts.append(f"WHEN {expr(c)} THEN {expr(r)}")
        if e.default is not None:
            parts.append(f"ELSE {expr(e.default)}")
        parts.append("END")
        return " ".join(parts)
    if k == "SubLink":
        inner = select(e.query, 0).strip()
        if e.kind == "exists":
            return f"(EXISTS ( {inner}))"
        if e.kind == "expr":
            return f"( {inner})"
        if e.kind == "array":
            return f"ARRAY( {inner})"
        if e.kind in ("any", "all"):
            if e.op == "=" and e.kind == "any":
                return f"({expr(e.test, 7)} IN ( {inner}))"
            return f"({expr(e.test, 7)} {e.op} {e.kind.upper()} ( {inner}))"
        return f"( {inner})"
    if k == "AnyAll":
        return f"{expr(e.left, 7)} {e.op} {e.kind.upper()} ({expr(e.right)})"
    if k == "ArrayExpr":
        return "ARRAY[" + ", ".join(expr(x) for x in e.items) + "]"
    if k == "RowExpr":
        return "ROW(" + ", ".join(expr(x) for x in e.items) + ")"
    if k == "Subscript":
        if e.slice:
            return f"{expr(e.arg, 12)}[{expr(e.lo) if e.lo else ''}:{expr(e.hi) if e.hi else ''}]"
        return f"{expr(e.arg, 12)}[{expr(e.lo)}]"
    if k == "FieldSel":
        return f"({expr(e.arg)}).{e.name}"
    if k == "AtTimeZone":
        return f"{expr(e.arg, 12)} AT TIME ZONE {expr(e.zone, 12)}"
    if k == "Collate":
        return f"{expr(e.arg)} COLLATE {'.'.join(e.collation)}"
    if k == "Grouping":
        return "GROUPING(" + ", ".join(expr(a) for a in e.args) + ")"
    if k == "SetToDefault":
        return "DEFAULT"
    return "?"


def _op_prec(op):
    if op in ("=", "<", ">", "<=", ">=", "<>"):
        return 5
    if op in ("+", "-"):
        return 8
    if op in ("*", "/", "%"):
        return 9
    if op == "^":
        return 10
    return 7


def sortby(s: A.SortBy):
    out = expr(s.expr)
    if s.desc:
        out += " DESC"
    if s.nulls:
        out += f" NULLS {s.nulls.upper()}"
    return out


def window(w: A.WindowDef):
    if w.bare_ref:
        return w.ref
    parts = []
    if w.ref:
        parts.append(w.ref)
    if w.partition:
        parts.append("PARTITION BY " + ", ".join(expr(p) for p in w.partition))
    if w.order:
        parts.append("ORDER BY " + ", ".join(sortby(s) for s in w.order))
    if w.frame_mode:
        def b(fb):
            if fb.kind.startswith("unbounded"):
                return fb.kind.replace("_", " ").upper()
            if fb.kind == "current_row":
                return "CURRENT ROW"
            return f"{expr(fb.offset)} {fb.kind.upper()}"
        fr = w.frame_mode.upper() + " "
        fr += f"BETWEEN {b(w.start)} AND {b(w.end)}" if w.end is not None else b(w.start)
        parts.append(fr)
    return "(" + " ".join(parts) + ")"


def from_item(f, ind):
    k = type(f).__name__
    if k == "RangeVar":
        s = (f"{f.schema}." if f.schema else "") + f.name
        if f.alias is not None:
            s += f" {f.alias.name}"
        return s
    if k == "RangeSubselect":
        s = ("LATERAL " if f.lateral else "") + "( " + select(f.query, ind + 4).strip() + ")"
        if f.alias is not None:
            s += f" {f.alias.name}"
        return s
    if k == "RangeFunction":
        s = ("LATERAL " if f.lateral else "") + ", ".join(expr(x) for x in f.funcs)
        if f.ordinality:
            s += " WITH ORDINALITY"
        if f.alias is not None:
            s += f" {f.alias.name}" + (f"({', '.join(f.alias.cols)})" if f.alias.cols else "")
        return s
    if k == "JoinExpr":
        kind = {"inner": "JOIN", "left": "LEFT JOIN", "right": "RIGHT JOIN", "full": "FULL JOIN",
                "cross": "CROSS JOIN"}[f.kind]
        s = from_item(f.left, ind) + "\n" + " " * (ind + 5) + kind + " " + from_item(f.right, ind)
        if f.quals is not None:
            s += f" ON {expr(f.quals)}"
        elif f.using:
            s += f" USING ({', '.join(f.using)})"
        return s
    return "?"


def select(q, ind=0):
    pad = " " * ind
    if isinstance(q, A.SetOp):
        op = q.op.upper() + (" ALL" if q.all else "")
        s = select(q.left, ind) + f"\n{pad}{op}\n" + select(q.right, ind)
        return s
    if isinstance(q, A.Values):
        return pad + " VALUES " + ", ".join("(" + ", ".join(expr(x) for x in row) + ")" for row in q.rows)
    out = []
    if q.with_ is not None:
        ctes = ", ".join(f"{c.name} AS ( {select(c.query, ind + 4).strip()})" for c in q.with_.ctes)
        out.append(f"{pad} WITH {'RECURSIVE ' if q.with_.recursive else ''}{ctes}")
    head = " SELECT "
    if q.distinct is True:
        head += "DISTINCT "
    elif isinstance(q.distinct, list):
        head += "DISTINCT ON (" + ", ".join(expr(x) for x in q.distinct) + ") "
    tl = []
    for t in q.targets:
        s = expr(t.expr)
        if t.name:
            s += f" AS {quote_ident(t.name)}"
        tl.append(s)
    out.append(pad + head + (",\n" + pad + "    ").join(tl))
    if q.from_:
        out.append(pad + "   FROM " + ", ".join(from_item(f, ind) for f in q.from_))
    if q.where is not None:
        out.append(pad + "  WHERE " + expr(q.where))
    if q.group_by:
        out.append(pad + "  GROUP BY " + ", ".join(expr(g) if not isinstance(g, A.GroupingSet) else
                                                   f"{g.kind.upper()} (" + ", ".join(expr(x) for x in g.items)
                                                   + ")" for g in q.group_by))
    if q.having is not None:
        out.append(pad + " HAVING " + expr(q.having))
    if q.order_by:
        out.append(pad + "  ORDER BY " + ", ".join(sortby(s) for s in q.order_by))
    if q.limit is not None:
        out.append(pad + " LIMIT " + expr(q.limit))
    if q.offset is not None:
        out.append(pad + " OFFSET " + expr(q.offset))
    return "\n".join(out)


def unparse_select(q):
    return select(q, 0)

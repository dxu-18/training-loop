"""Semantic analysis: parse trees -> Query objects with typed, bound expressions (after parse_*.c).

Name resolution (FROM items, aliases, USING joins, LATERAL, outer references), function and operator resolution
(coerce.py), implicit casts, aggregates and GROUP BY checks (functional dependency on a primary key), window
functions, set-returning functions in SELECT, ORDER BY / GROUP BY by name or position, set operations, VALUES,
CTEs (recursive too), views, and PostgreSQL's error messages with positions.
"""

from __future__ import annotations

import datetime as dt

from . import ast as A
from . import coerce as C
from . import expr as E
from . import types as T
from .aggs import AGG_NAMES, WINDOW_ONLY, make_agg
from .catalog import Table, View, Sequence, Index
from .errors import PgError, Unsupported
from .funcs import NONSTRICT, VOLATILE, eq_impl, lt_impl, make_func, op_impl

SRF_NAMES = {"generate_series", "unnest", "jsonb_array_elements", "jsonb_array_elements_text", "jsonb_each",
             "jsonb_each_text", "jsonb_object_keys", "json_array_elements", "json_array_elements_text",
             "regexp_matches", "regexp_split_to_table", "string_to_table", "json_each", "json_each_text",
             "json_object_keys", "jsonb_path_query", "generate_subscripts"}


# ---------------------------------------------------------------- range table

class RTE:
    def __init__(self, kind, name, colnames, coltypes, slots, alias_given=False):
        self.kind = kind                # rel subquery func values cte join
        self.name = name                # the name references use (alias or table name)
        self.colnames = list(colnames)
        self.coltypes = list(coltypes)
        self.slots = list(slots)        # absolute positions in the query level's row
        self.alias_given = alias_given
        self.relname = None
        self.rel = None
        self.query = None
        self.lateral = False
        self.merged = set()             # join RTE: names of USING columns
        self.hidden = set()             # columns not visible to unqualified refs (USING inputs)
        self.pk_cols = None             # rel: primary key column names
        self.sample = None

    def find(self, name):
        hits = [i for i, n in enumerate(self.colnames) if n == name]
        return hits


class NSItem:
    __slots__ = ("rte", "rel_visible", "cols_visible", "lateral_ok")

    def __init__(self, rte, rel_visible=True, cols_visible=True):
        self.rte, self.rel_visible, self.cols_visible = rte, rel_visible, cols_visible
        self.lateral_ok = True


class Scope:
    """One query level while it is analyzed."""

    def __init__(self, parent=None, session=None, db=None):
        self.parent = parent
        self.session = session if session is not None else (parent.session if parent else None)
        self.db = db if db is not None else (parent.db if parent else None)
        self.ns: list[NSItem] = []
        self.width = 0
        self.ctes: dict = {}
        self.kind = "select"            # the clause being analyzed (for error messages)
        self.aggs: list = []
        self.wins: list = []
        self.srfs: list = []
        self.in_agg = False
        self.in_window = False
        self.allow_srf = False
        self.outer_refs: list = []      # Vars that reached outer levels (for SubPlans)
        self.hidden_ns: list = []       # items not visible here (LATERAL hint)
        self.params: list = []
        self.lateral_barrier = False
        self.named_windows: dict = {}
        self.query = None

    def find_cte(self, name):
        s, lvl = self, 0
        while s is not None:
            if name in s.ctes:
                return s.ctes[name], lvl
            s = s.parent
            lvl += 1
        return None, 0

    def alloc(self, n):
        start = self.width
        self.width += n
        return list(range(start, start + n))


class CTEDef:
    def __init__(self, name, query, colnames, coltypes, recursive=False, node=None):
        self.name, self.query, self.colnames, self.coltypes = name, query, colnames, coltypes
        self.recursive = recursive
        self.node = node
        self.refcount = 0
        self.id = id(self)
        self.modifying = False


# ---------------------------------------------------------------- query objects

class FromNode:
    base = 0
    width = 0


class ScanRel(FromNode):
    def __init__(self, rte, rel):
        self.rte, self.rel = rte, rel
        self.base, self.width = rte.slots[0] if rte.slots else 0, len(rte.colnames)
        self.quals = []


class ScanSub(FromNode):
    def __init__(self, rte, query, lateral=False):
        self.rte, self.query, self.lateral = rte, query, lateral
        self.base, self.width = rte.slots[0] if rte.slots else 0, len(rte.colnames)
        self.quals = []


class ScanCte(FromNode):
    def __init__(self, rte, cte, levels):
        self.rte, self.cte, self.levels = rte, cte, levels
        self.base, self.width = rte.slots[0] if rte.slots else 0, len(rte.colnames)
        self.quals = []


class ScanFunc(FromNode):
    def __init__(self, rte, funcs, ordinality, base, width):
        self.rte, self.funcs, self.ordinality = rte, funcs, ordinality
        self.base, self.width = base, width
        self.quals = []
        self.lateral = True


class ScanValues(FromNode):
    def __init__(self, rte, query):
        self.rte, self.query = rte, query
        self.base, self.width = rte.slots[0] if rte.slots else 0, len(rte.colnames)
        self.quals = []


class JoinNode(FromNode):
    def __init__(self, kind, left, right, quals, merge, base, width, rte):
        self.kind, self.left, self.right, self.quals = kind, left, right, quals
        self.merge = merge              # FULL JOIN USING: [(left slot, right slot)] appended to the row
        self.base, self.width = base, width
        self.rte = rte
        self.lateral = False


class ListNode(FromNode):
    """Comma-separated FROM items (implicit joins), joined with the WHERE clause's conditions."""

    def __init__(self, items, base, width):
        self.items, self.base, self.width = items, base, width


class Query:
    def __init__(self, kind="select"):
        self.kind = kind                # select setop values
        self.targets = []               # BE (in the final layout)
        self.names = []
        self.types = []
        self.from_ = None
        self.where = None
        self.width = 0
        self.grouped = False
        self.group_keys = []            # BE in the FROM layout
        self.grouping_sets = None       # [[key idx]] or None
        self.aggs = []
        self.having = None
        self.windows = []               # [WindowClause]
        self.srfs = []
        self.distinct = None            # None / True / [target idx]
        self.order = []                 # [(target idx, desc, nulls_first)]
        self.ntargets = 0               # visible targets (the rest are ORDER BY helpers)
        self.limit = None
        self.offset = None
        self.with_ties = False
        self.ctes = []
        self.setop = None               # (op, all, left, right)
        self.values = None              # [[BE]]
        self.locking = False
        self.into = None
        self.recursive_ref = None
        self.layout = 0                 # width of the layout targets are evaluated on
        self.has_subplans = False
        self.correlated = False
        self.target_srcs = []           # (table, column) a target column comes from, for \d-like info

    @property
    def visible(self):
        return self.targets[:self.ntargets]


class WindowClause:
    def __init__(self, partition, order, mode, start, end, start_off, end_off, exclude, name=None):
        self.partition, self.order = partition, order    # BE lists; order: [(BE, desc, nulls_first)]
        self.mode, self.start, self.end = mode, start, end
        self.start_off, self.end_off = start_off, end_off
        self.exclude = exclude
        self.funcs = []
        self.name = name
        self.has_order = bool(order)


# ---------------------------------------------------------------- helpers

def figure_colname(node) -> str:
    name, _ = _figure(node)
    return name or "?column?"


def _figure(node):
    if isinstance(node, A.ColumnRef):
        last = node.names[-1]
        if last == "*":
            return None, 0
        return last, 2
    if isinstance(node, A.Subscript):
        return _figure(node.arg)
    if isinstance(node, A.FieldSel):
        return (node.name if node.name != "*" else None), 2
    if isinstance(node, A.FuncCall):
        return node.name[-1], 2
    if isinstance(node, A.Cast):
        n, s = _figure(node.arg)
        if s <= 1:
            return node.typ.names[-1] if node.typ.array == 0 else node.typ.names[-1], 1
        return n, s
    if isinstance(node, A.TypedLiteral):
        return node.typ.names[-1], 1
    if isinstance(node, A.Collate):
        return _figure(node.arg)
    if isinstance(node, A.Case):
        n, s = _figure(node.default) if node.default is not None else (None, 0)
        if s <= 1:
            return "case", 1
        return n, s
    if isinstance(node, A.ArrayExpr):
        return "array", 2
    if isinstance(node, A.RowExpr):
        return "row", 2
    if isinstance(node, A.SQLValue):
        return node.name, 2
    if isinstance(node, A.Grouping):
        return "grouping", 2
    if isinstance(node, A.AtTimeZone):
        return "timezone", 2
    if isinstance(node, A.SubLink):
        if node.kind == "exists":
            return "exists", 2
        if node.kind == "array":
            return "array", 2
        if node.kind == "expr":
            q = node.query
            while isinstance(q, A.SetOp):
                q = q.left
            if isinstance(q, A.Select) and q.targets:
                t = q.targets[0]
                if t.name:
                    return t.name, 2
                return _figure(t.expr)
        return None, 0
    return None, 0


def _err_kind(kind):
    return {"where": "WHERE", "on": "JOIN conditions", "group": "GROUP BY", "from_func": "functions in FROM",
            "values": "VALUES", "limit": "LIMIT", "offset": "OFFSET", "check": "check constraints",
            "default": "DEFAULT expressions", "index": "index expressions", "index_pred": "index predicates",
            "returning": "RETURNING", "update": "UPDATE", "insert": "VALUES", "partition": "window PARTITION BY",
            "window_order": "window ORDER BY", "window_frame": "window frame", "policy": "policy",
            "call": "CALL", "copy_where": "COPY FROM WHERE conditions", "generated": "column generation "
                                                                                       "expressions",
            "execute": "EXECUTE parameters", "cycle": "CYCLE", "merge_when": "MERGE WHEN conditions",
            "filter": "FILTER", "on_conflict": "ON CONFLICT"}.get(kind, kind.upper())


def coerce_to(e: E.BE, target: T.Type, ctx: str = "i", pos=None, explicit=False, what=None) -> E.BE:
    """Insert a cast so e has type target (implicit / assignment / explicit). Unknown literals become constants of
    the target type right away (as the parser's coercion of literals does)."""
    src = e.type
    if src == target and (target.mod is None or src.mod == target.mod or not explicit):
        if target.mod is not None and src.mod != target.mod and ctx != "i":
            return _typmod_cast(e, target, explicit, pos)
        return e
    if isinstance(e, E.Const) and src.name == "unknown":
        if e.value is None:
            return E.Const(None, target, e.pos)
        try:
            v = T.input_value(e.value, target.base, target.mod, now=_now_holder[0], explicit=explicit)
        except PgError as err:
            if err.pos is None:
                err.pos = (e.pos if e.pos is not None and e.pos >= 0 else pos)
                if err.pos is not None:
                    err.pos += 1
            raise
        return E.Const(v, target, e.pos)
    if src.name == "unknown":
        # an unknown-typed expression that isn't a literal (e.g. a NULL from a subquery)
        return E.Cast(e, target, lambda v, c, t=target: T.input_value(v, t.base, t.mod), explicit, pos=pos)
    if target.name in ("any", "anyelement", "anyarray", "anynonarray", "anycompatible", "anycompatiblearray",
                       "anyenum", "record") and src.name != "unknown":
        return e
    if src == target:
        return _typmod_cast(e, target, explicit, pos)
    c = T.find_cast(src, target)
    order = {"i": 0, "a": 1, "e": 2}
    if c is None or order[c[0]] > order[ctx]:
        if what is not None:
            raise what
        if ctx == "e":
            raise PgError(f"cannot cast type {src.format()} to {target.format()}", "42846", pos=pos)
        raise PgError(f"cannot cast type {src.format()} to {target.format()}", "42846", pos=pos)
    base = target.base
    mod = target.mod
    s_t = src

    def conv(v, c, s_t=s_t, base=base, mod=mod, explicit=explicit):
        out = T.cast_value(v, s_t, base, explicit, now=c.txn_now if c is not None else None)
        if mod is not None:
            out = T.apply_typmod(out, base, mod, explicit)
        return out
    return E.Cast(e, target, conv, explicit, pos=pos)


def _typmod_cast(e, target, explicit, pos):
    base, mod = target.base, target.mod
    if mod is None:
        return e
    if isinstance(e, E.Const):
        return E.Const(T.apply_typmod(e.value, base, mod, explicit) if e.value is not None else None, target, e.pos)
    return E.Cast(e, target, lambda v, c: T.apply_typmod(v, base, mod, explicit), explicit, pos=pos)


_now_holder = [None]


def const_fold(e: E.BE, ctx=None):
    """Evaluate an expression without columns (LIMIT, defaults at INSERT time ...)."""
    f = e.compile()
    return f(None, ctx)


# ---------------------------------------------------------------- the analyzer

class Analyzer:
    def __init__(self, session, db, params=None, param_types=None):
        self.session = session
        self.db = db
        self.params = params or []
        self.param_types = param_types or []
        self.subplan_n = 0

    # ------------------------------------------------------------ type names
    def resolve_type(self, tn: A.TypeName) -> T.Type:
        names = tn.names
        name = names[-1]
        schema = names[0] if len(names) > 1 else None
        if schema == "pg_catalog":
            schema = None
        t = None
        if schema is None:
            t = T.resolve_name(name)
            if t is not None and t.name in ("serial", "bigserial", "smallserial"):
                t = None
        if t is None:
            et = self.db.find_type(name, schema, self.session)
            if et is not None:
                t = et.type
        if t is None:
            shown = ".".join(names)
            raise PgError(f'type "{shown}" does not exist', "42704", pos=tn.pos + 1 if tn.pos >= 0 else None)
        if t.name not in T.SUPPORTED and t.cat != "E" and t.name not in ("record", "void", "unknown"):
            raise Unsupported(f"type {t.display}", tn.pos + 1 if tn.pos >= 0 else None)
        mod = self.typmod(t, tn)
        out = t.with_mod(mod) if mod is not None else t
        for _ in range(tn.array):
            arr = out.base.array()
            out = arr.with_mod(mod) if mod is not None else arr
        return out

    def typmod(self, t, tn):
        mods = [m.value for m in tn.mods]
        if t.name == "interval":
            if tn.interval_fields:
                return tn.interval_fields
            return mods[0] if mods else None
        if not mods:
            return None
        n = t.name
        if n == "numeric":
            p = mods[0]
            s = mods[1] if len(mods) > 1 else 0
            if not isinstance(p, int) or p < 1 or p > 1000:
                raise PgError(f"NUMERIC precision {p} must be between 1 and 1000", "22023", pos=tn.pos + 1)
            if s < -1000 or s > 1000:
                raise PgError(f"NUMERIC scale {s} must be between -1000 and 1000", "22023", pos=tn.pos + 1)
            return (p, s)
        if n in ("varchar", "bpchar"):
            if mods[0] < 1:
                raise PgError(f"length for type {'varchar' if n == 'varchar' else 'char'} must be at least 1",
                              "22023", pos=tn.pos + 1)
            return mods[0]
        if n in ("timestamp", "timestamptz", "time", "timetz"):
            p = mods[0]
            if p < 0:
                raise PgError(f"TIMESTAMP({p}) precision must not be negative", "22023", pos=tn.pos + 1)
            return min(p, 6)
        if n in ("bit", "varbit"):
            return mods[0]
        raise PgError(f"type modifier is not allowed for type \"{t.display}\"", "42601", pos=tn.pos + 1)

    # ------------------------------------------------------------ queries
    def query(self, node, parent: Scope | None, *, ctes_parent=None, for_insert=False) -> Query:
        if isinstance(node, A.SetOp):
            return self.setop_query(node, parent)
        if isinstance(node, A.Values):
            return self.values_query(node, parent, for_insert=for_insert)
        if isinstance(node, A.Select):
            return self.select_query(node, parent)
        raise PgError("unexpected statement in a query position", "42601")

    def new_scope(self, parent):
        s = Scope(parent, self.session, self.db)
        return s

    def with_clause(self, w: A.With, scope: Scope, q: Query):
        if w is None:
            return
        seen = set()
        for cte in w.ctes:
            if cte.name in seen:
                raise PgError(f'WITH query name "{cte.name}" specified more than once', "42712", pos=cte.pos + 1)
            seen.add(cte.name)
            if not isinstance(cte.query, (A.Select, A.SetOp, A.Values)):
                d = self.modifying_cte(cte, scope)
                scope.ctes[cte.name] = d
                q.ctes.append(d)
                continue
            if w.recursive and self._self_referencing(cte):
                d = self.recursive_cte(cte, scope)
            else:
                sub = self.query(cte.query, scope)
                names = list(sub.names[:sub.ntargets])
                if cte.cols:
                    if len(cte.cols) > len(names):
                        raise PgError(f'WITH query "{cte.name}" has {len(names)} columns available but '
                                      f'{len(cte.cols)} columns specified', "42P10", pos=cte.pos + 1)
                    names[:len(cte.cols)] = cte.cols
                d = CTEDef(cte.name, sub, names, list(sub.types[:sub.ntargets]), node=cte)
            scope.ctes[cte.name] = d
            q.ctes.append(d)

    def modifying_cte(self, cte, scope):
        from .dml import analyze_modify
        stmt = analyze_modify(self, cte.query, scope)
        if not stmt.returning:
            names, types = [], []
        else:
            names, types = stmt.ret_names, stmt.ret_types
        d = CTEDef(cte.name, stmt, names, types, node=cte)
        d.modifying = True
        return d

    @staticmethod
    def _self_referencing(cte):
        found = [False]

        def visit(n):
            if found[0] or n is None:
                return
            if isinstance(n, A.RangeVar) and n.schema is None and n.name == cte.name:
                found[0] = True
                return
            if isinstance(n, list):
                for x in n:
                    visit(x)
                return
            if isinstance(n, tuple):
                for x in n:
                    visit(x)
                return
            if isinstance(n, A.Node):
                for v in vars(n).values():
                    if isinstance(v, (A.Node, list, tuple)):
                        visit(v)
        visit(cte.query)
        return found[0]

    def recursive_cte(self, cte, scope):
        q = cte.query
        if not isinstance(q, A.SetOp) or q.op != "union":
            raise PgError(f'recursive query "{cte.name}" does not have the form non-recursive-term UNION [ALL] '
                          'recursive-term', "42P19", pos=cte.pos + 1)
        if q.order_by or q.limit is not None or q.offset is not None:
            raise Unsupported("ORDER BY / LIMIT in a recursive query", cte.pos + 1)
        base = self.query(q.left, scope)
        names = list(base.names[:base.ntargets])
        if cte.cols:
            if len(cte.cols) != len(names):
                raise PgError(f'WITH query "{cte.name}" has {len(names)} columns available but {len(cte.cols)} '
                              'columns specified', "42P10", pos=cte.pos + 1)
            names = list(cte.cols)
        types = []
        for t in base.types[:base.ntargets]:
            types.append(T.TEXT if t.name == "unknown" else t)
        d = CTEDef(cte.name, None, names, types, recursive=True, node=cte)
        scope.ctes[cte.name] = d
        rec = self.query(q.right, scope)
        if len(rec.types[:rec.ntargets]) != len(types):
            raise PgError("each UNION query must have the same number of columns", "42601",
                          pos=getattr(q.right, "pos", -1) + 1)
        for i, (a, b) in enumerate(zip(types, rec.types)):
            if b.name == "unknown":
                rec.targets[i] = coerce_to(rec.targets[i], a)
                rec.types[i] = a
                continue
            if a != b:
                raise PgError(f'recursive query "{cte.name}" column {i + 1} has type {a.format()} in non-recursive '
                              f'term but type {b.format()} overall', "42804",
                              pos=_first_target_pos(q.left, i),
                              hint="Cast the output of the non-recursive term to the correct type.")
        for i, t in enumerate(base.types[:base.ntargets]):
            if t.name == "unknown":
                base.targets[i] = coerce_to(base.targets[i], T.TEXT)
                base.types[i] = T.TEXT
        d.query = Query("setop")
        d.query.setop = ("union", q.all, base, rec)
        d.query.names = names
        d.query.types = types
        d.query.ntargets = len(names)
        d.query.targets = []
        d.query.recursive_ref = d
        return d

    def values_query(self, node: A.Values, parent, for_insert=False) -> Query:
        scope = self.new_scope(parent)
        q = Query("values")
        self.with_clause(node.with_, scope, q)
        scope.kind = "values"
        rows = []
        width = None
        for row in node.rows:
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise PgError("VALUES lists must all be the same length", "42601",
                              pos=(row[0].pos if row and hasattr(row[0], "pos") else node.pos) + 1)
            bound = []
            for x in row:
                if isinstance(x, A.SetToDefault):
                    if not for_insert:
                        raise PgError("DEFAULT is not allowed in this context", "42601", pos=x.pos + 1)
                    bound.append(E.SetToDefault(T.UNKNOWN, x.pos))
                else:
                    bound.append(self.expr(x, scope))
            rows.append(bound)
        types = []
        for j in range(width or 0):
            col = [r[j] for r in rows]
            if for_insert:
                types.append(None)
                continue
            ct = C.common_type([e.type for e in col if not isinstance(e, E.SetToDefault)], "VALUES",
                               [getattr(x, "pos", None) for x in [r[j] for r in node.rows]])
            types.append(ct)
            for r in rows:
                if not isinstance(r[j], E.SetToDefault):
                    r[j] = coerce_to(r[j], ct, pos=None)
        q.values = rows
        q.names = [f"column{j + 1}" for j in range(width or 0)]
        q.types = types if not for_insert else [None] * (width or 0)
        q.ntargets = width or 0
        self.finish_order_limit(q, node, scope, setop_like=True)
        q.outer_refs = scope.outer_refs
        return q

    def setop_query(self, node: A.SetOp, parent) -> Query:
        scope = self.new_scope(parent)
        q = Query("setop")
        self.with_clause(node.with_, scope, q)
        left = self.setop_branch(node.left, scope)
        right = self.setop_branch(node.right, scope)
        opname = node.op.upper()
        if left.ntargets != right.ntargets:
            raise PgError(f"each {opname} query must have the same number of columns", "42601",
                          pos=_query_pos(node.right) + 1)
        types = []
        for i in range(left.ntargets):
            lt, rt = left.types[i], right.types[i]
            ct = C.common_type([lt, rt], opname, [None, _first_target_pos(node.right, i)])
            if ct.name == "unknown":
                ct = T.TEXT
            types.append(ct)
            for side in (left, right):
                if side.types[i] != ct or (side.types[i].mod != ct.mod):
                    _retype(side, i, ct)
        q.setop = (node.op, node.all, left, right)
        q.names = list(left.names[:left.ntargets])
        q.types = types
        q.ntargets = len(types)
        self.finish_order_limit(q, node, scope, setop_like=True)
        q.outer_refs = scope.outer_refs
        return q

    def setop_branch(self, node, scope):
        sub = self.query(node, scope)
        return sub

    def finish_order_limit(self, q, node, scope, setop_like):
        """ORDER BY / LIMIT for set operations and VALUES: names or positions only."""
        q.targets = [None] * q.ntargets
        for sb in node.order_by:
            e = sb.expr
            idx = None
            if isinstance(e, A.Const) and e.kind == "int":
                if e.value < 1 or e.value > q.ntargets:
                    raise PgError(f"ORDER BY position {e.value} is not in select list", "42P10", pos=e.pos + 1)
                idx = e.value - 1
            elif isinstance(e, A.ColumnRef) and len(e.names) == 1:
                hits = [i for i, n in enumerate(q.names[:q.ntargets]) if n == e.names[0]]
                if hits:
                    idx = hits[0]
                else:
                    raise PgError(f'column "{e.names[0]}" does not exist', "42703", pos=e.pos + 1)
            else:
                raise PgError(f"invalid {'UNION/INTERSECT/EXCEPT' if q.kind == 'setop' else 'VALUES'} ORDER BY "
                              "clause" if q.kind == "setop" else "invalid VALUES ORDER BY clause", "0A000",
                              pos=getattr(e, "pos", -1) + 1,
                              detail="Only result column names can be used, not expressions or functions.",
                              hint="Add the expression/function to every SELECT, or move the UNION into a FROM "
                                   "clause." if q.kind == "setop" else None)
            nulls_first = sb.desc if sb.nulls is None else sb.nulls == "first"
            q.order.append((idx, sb.desc, nulls_first, _sort_type(q.types[idx])))
        self.limits(q, node, scope)

    def limits(self, q, node, scope):
        if node.limit is not None:
            q.limit = self.limit_expr(node.limit, scope, "LIMIT")
        if node.offset is not None:
            q.offset = self.limit_expr(node.offset, scope, "OFFSET")
        q.with_ties = node.with_ties
        if node.with_ties and not node.order_by:
            raise PgError("WITH TIES cannot be specified without ORDER BY clause", "42601",
                          pos=node.limit.pos + 1 if node.limit is not None else None)
        if node.locking:
            q.locking = True

    def limit_expr(self, node, scope, what):
        s = self.new_scope(scope.parent if scope else None)
        s.kind = what.lower()
        e = self.expr(node, s)
        for n in e.walk():
            if isinstance(n, E.Var) and n.level == 0:
                raise PgError(f"argument of {what} must not contain variables", "42P10", pos=node.pos + 1)
        if isinstance(e, E.Const) and e.value is None:
            return E.Const(None, T.INT8)
        target = T.INT8
        if e.type.name == "numeric" or e.type.name in ("float4", "float8"):
            target = T.INT8
        return coerce_to(e, target, "a", pos=node.pos,
                         what=PgError(f"argument of {what} must be type bigint, not type {e.type.format()}",
                                      "42804", pos=node.pos + 1))

    # ------------------------------------------------------------ SELECT
    def select_query(self, node: A.Select, parent) -> Query:
        scope = self.new_scope(parent)
        q = Query("select")
        scope.query = q
        self.with_clause(node.with_, scope, q)
        # FROM
        scope.kind = "from"
        q.from_ = self.from_clause(node.from_, scope) if node.from_ else None
        q.width = scope.width
        # WHERE
        if node.where is not None:
            scope.kind = "where"
            w = self.expr(node.where, scope)
            q.where = self.bool_expr(w, "WHERE", node.where)
        # window definitions
        for wd in node.windows:
            if wd.name in scope.named_windows:
                raise PgError(f'window "{wd.name}" is already defined', "42P20", pos=wd.pos + 1)
            scope.named_windows[wd.name] = wd
        # SELECT list
        scope.kind = "select"
        scope.allow_srf = True
        targets, names, srcs = [], [], []
        for rt in node.targets:
            for e, name, src in self.target(rt, scope):
                targets.append(e)
                names.append(name)
                srcs.append(src)
        scope.allow_srf = False
        q.target_srcs = srcs
        # resolve unknown literals in the output to text (not for INSERT ... SELECT: the target columns decide)
        keep_unknown = getattr(self, "keep_unknown", False)
        self.keep_unknown = False
        for i, e in enumerate(targets):
            if e.type.name == "unknown" and not keep_unknown:
                targets[i] = coerce_to(e, T.TEXT)
        # GROUP BY
        group_items = []
        if node.group_by:
            scope.kind = "group"
            group_items = self.group_by(node.group_by, scope, targets, names, node)
        # HAVING
        having = None
        if node.having is not None:
            scope.kind = "having"
            h = self.expr(node.having, scope)
            having = self.bool_expr(h, "HAVING", node.having)
        # ORDER BY / DISTINCT ON (bound in the target layout)
        scope.kind = "order"
        order_items = []
        for sb in node.order_by:
            order_items.append((self.sort_item(sb, scope, targets, names, node), sb))
        distinct_on = None
        if isinstance(node.distinct, list):
            distinct_on = [self.sort_item(A.SortBy(x, pos=getattr(x, "pos", -1)), scope, targets, names, node,
                                          what="DISTINCT ON") for x in node.distinct]
        # grouping
        q.grouped = bool(group_items) or bool(scope.aggs) or having is not None
        layout_exprs = {"targets": targets, "having": having, "order": order_items}
        if q.grouped:
            self.apply_grouping(q, scope, group_items, node, targets, having, order_items, distinct_on)
            having = q.having
        else:
            q.layout = q.width
            q.having = None
        # windows
        if scope.wins:
            self.finish_windows(q, scope)
        # SRFs in the SELECT list
        if scope.srfs:
            self.finish_srfs(q, scope, targets)
        # sort / distinct targets
        ntargets = len(targets)
        all_targets = list(targets)
        order = []
        for (e, sb) in order_items:
            idx = None
            if isinstance(e, int):
                idx = e
            else:
                k = e.key()
                for i, t in enumerate(all_targets[:ntargets]):
                    if t.key() == k and not E.has_volatile(t):
                        idx = i
                        break
                if idx is None:
                    if node.distinct is True:
                        raise PgError("for SELECT DISTINCT, ORDER BY expressions must appear in select list",
                                      "42P10", pos=sb.expr.pos + 1 if getattr(sb.expr, "pos", -1) >= 0 else None)
                    all_targets.append(e)
                    idx = len(all_targets) - 1
            st = all_targets[idx].type
            nulls_first = sb.desc if sb.nulls is None else sb.nulls == "first"
            if sb.using is not None:
                nulls_first = sb.using in (">", ">=") if sb.nulls is None else nulls_first
            order.append((idx, sb.desc, nulls_first, _sort_type(st)))
        if distinct_on is not None:
            didx = []
            for e in distinct_on:
                if isinstance(e, int):
                    didx.append(e)
                    continue
                k = e.key()
                hit = next((i for i, t in enumerate(all_targets) if t.key() == k), None)
                if hit is None:
                    all_targets.append(e)
                    hit = len(all_targets) - 1
                didx.append(hit)
            # DISTINCT ON expressions must match the leading ORDER BY expressions
            if order:
                lead = [o[0] for o in order[:len(didx)]]
                okeys = {all_targets[i].key() for i in didx}
                for o in order[:len(didx)]:
                    if all_targets[o[0]].key() not in okeys:
                        raise PgError("SELECT DISTINCT ON expressions must match initial ORDER BY expressions",
                                      "42P10", pos=node.distinct[0].pos + 1)
            q.distinct = didx
        elif node.distinct is True:
            q.distinct = True
        q.targets = all_targets
        q.ntargets = ntargets
        q.names = names
        q.types = [t.type for t in all_targets]
        q.order = order
        self.limits(q, node, scope)
        if node.into is not None:
            q.into = node.into
        q.outer_refs = scope.outer_refs
        return q

    def bool_expr(self, e, what, node):
        if e.type.name == "unknown":
            return coerce_to(e, T.BOOL, pos=node.pos)
        if e.type != T.BOOL:
            raise PgError(f"argument of {what} must be type boolean, not type {e.type.format()}", "42804",
                          pos=getattr(node, "pos", -1) + 1)
        return e

    def target(self, rt: A.ResTarget, scope):
        e = rt.expr
        if isinstance(e, A.Star) or (isinstance(e, A.ColumnRef) and e.names[-1] == "*"):
            return self.expand_star(e, scope)
        if isinstance(e, A.FieldSel) and e.name == "*":
            be = self.expr(e.arg, scope)
            if be.type.name != "record" or not isinstance(be, E.RowExpr):
                raise Unsupported("(expression).* expansion", rt.pos + 1)
            return [(x, n, None) for x, n in zip(be.items, be.names)]
        be = self.expr(e, scope)
        name = rt.name or figure_colname(e)
        src = None
        if isinstance(be, E.Var) and be.level == 0 and isinstance(e, A.ColumnRef):
            src = (be.rte, be.col)
        return [(be, name, src)]

    def expand_star(self, e, scope):
        out = []
        if isinstance(e, A.Star) or len(e.names) == 1:
            items = [it for it in scope.ns if it.cols_visible]
            if not items:
                raise PgError("SELECT * with no tables specified is not valid", "42601", pos=e.pos + 1)
            for it in items:
                rte = it.rte
                for i, n in enumerate(rte.colnames):
                    if n in rte.hidden:
                        continue
                    out.append((self.var_for(rte, i, 0, scope), n, (rte.name, n)))
            return out
        names = e.names[:-1]
        rte, lvl = self.find_rte(names, scope, e.pos)
        for i, n in enumerate(rte.colnames):
            if n in rte.hidden and rte.kind == "join":
                continue
            out.append((self.var_for(rte, i, lvl, scope), n, (rte.name, n)))
        return out

    def var_for(self, rte, i, level, scope):
        slot = rte.slots[i]
        rn, cn = _src_of(rte, i)
        v = E.Var(level, slot, rte.coltypes[i], rn, cn)
        if level > 0:
            self.note_outer(v, scope)
        return v

    def note_outer(self, v, scope):
        s = scope
        for _ in range(v.level):
            s.outer_refs.append(v)
            s = s.parent
            if s is None:
                break

    # ------------------------------------------------------------ GROUP BY / aggregates
    def group_by(self, items, scope, targets, names, node):
        out = []
        for it in items:
            if isinstance(it, A.GroupingSet):
                out.append(("set", it.kind, self.group_set(it, scope, targets, names)))
            else:
                out.append(("expr", self.group_expr(it, scope, targets, names)))
        return out

    def group_set(self, gs, scope, targets, names):
        if gs.kind == "empty":
            return []
        items = []
        for x in gs.items:
            if isinstance(x, A.GroupingSet):
                items.append(("set", x.kind, self.group_set(x, scope, targets, names)))
            elif isinstance(x, A.RowExpr) and not x.explicit and gs.kind in ("rollup", "cube"):
                items.append(("row", [self.group_expr(y, scope, targets, names) for y in x.items]))
            else:
                items.append(("expr", self.group_expr(x, scope, targets, names)))
        return items

    def group_expr(self, it, scope, targets, names):
        if isinstance(it, A.Const) and it.kind == "int":
            if it.value < 1 or it.value > len(targets):
                raise PgError(f"GROUP BY position {it.value} is not in select list", "42P10", pos=it.pos + 1)
            t = targets[it.value - 1]
            self._no_agg_in_group(t, it)
            return t
        if isinstance(it, A.ColumnRef) and len(it.names) == 1:
            nm = it.names[0]
            try:
                return self.expr(it, scope)
            except PgError as err:
                if err.code != "42703":
                    raise
                hits = [i for i, n in enumerate(names) if n == nm]
                if hits:
                    t = targets[hits[0]]
                    self._no_agg_in_group(t, it)
                    return t
                raise
        n_aggs = len(scope.aggs)
        e = self.expr(it, scope)
        if len(scope.aggs) != n_aggs:
            raise PgError("aggregate functions are not allowed in GROUP BY", "42803", pos=it.pos + 1)
        return e

    def _no_agg_in_group(self, t, it):
        for n in t.walk():
            if isinstance(n, E.Aggref):
                raise PgError(f"aggregate functions are not allowed in GROUP BY", "42803",
                              pos=(n.pos + 1) if n.pos is not None else it.pos + 1)
            if isinstance(n, E.WindowFunc):
                raise PgError("window functions are not allowed in GROUP BY", "42P20",
                              pos=(n.pos + 1) if n.pos is not None else it.pos + 1)

    def expand_grouping(self, group_items):
        """GROUP BY items -> (key expressions, grouping sets as lists of key indexes or None)."""
        keys = []
        index = {}

        def key_of(e):
            k = e.key()
            if k not in index:
                index[k] = len(keys)
                keys.append(e)
            return index[k]

        def sets_of(item):
            kind = item[0]
            if kind == "expr":
                return [[key_of(item[1])]]
            if kind == "row":
                return [[key_of(x) for x in item[1]]]
            _, gkind, members = item
            if gkind == "sets":
                out = []
                for m in members:
                    out += sets_of(m)
                return out or [[]]
            elems = []
            for m in members:
                if m[0] == "expr":
                    elems.append([key_of(m[1])])
                elif m[0] == "row":
                    elems.append([key_of(x) for x in m[1]])
                else:
                    raise Unsupported("nested grouping sets inside ROLLUP/CUBE")
            if gkind == "rollup":
                return [sum(elems[:k], []) for k in range(len(elems), -1, -1)]
            if gkind == "cube":
                out = []
                n = len(elems)
                for mask in range((1 << n) - 1, -1, -1):
                    s = []
                    for j in range(n):
                        if mask & (1 << (n - 1 - j)):
                            s += elems[j]
                    out.append(s)
                return out
            return [[]]
        has_sets = any(it[0] == "set" for it in group_items)
        if not has_sets:
            for it in group_items:
                key_of(it[1])
            return keys, None
        combos = [[]]
        for it in group_items:
            ss = sets_of(it)
            combos = [a + b for a in combos for b in ss]
        sets = []
        for c in combos:
            seen = []
            for k in c:
                if k not in seen:
                    seen.append(k)
            sets.append(seen)
        return keys, sets

    def apply_grouping(self, q, scope, group_items, node, targets, having, order_items, distinct_on):
        keys, sets = self.expand_grouping(group_items)
        q.group_keys = keys
        q.grouping_sets = sets
        q.aggs = list(scope.aggs)
        nk = len(keys)
        keymap = {k.key(): i for i, k in enumerate(keys)}
        for i, a in enumerate(q.aggs):
            a.slot = nk + i
        grouping_slot = nk + len(q.aggs)
        rep_slot = grouping_slot + 1
        q.layout = grouping_slot + 2
        # functional dependency: grouping by a table's primary key allows its other columns
        pk_ranges = []
        key_vars = {(k.index) for k in keys if isinstance(k, E.Var) and k.level == 0}
        for it in scope.ns:
            for rte in [it.rte]:
                pass
        for rte in _all_rtes(scope):
            if rte.kind == "rel" and rte.pk_cols:
                slots = [rte.slots[rte.colnames.index(c)] for c in rte.pk_cols if c in rte.colnames]
                if slots and all(s in key_vars for s in slots):
                    pk_ranges.append(set(rte.slots))

        def remap(e, depth=0):
            return map_expr(e, lambda n, d: self._group_node(n, d, keymap, pk_ranges, grouping_slot, rep_slot), depth)
        for i, t in enumerate(targets):
            targets[i] = remap(t)
        if having is not None:
            q.having = remap(having)
        for j, (e, sb) in enumerate(order_items):
            if not isinstance(e, int):
                order_items[j] = (remap(e), sb)
        if distinct_on is not None:
            for j, e in enumerate(distinct_on):
                if not isinstance(e, int):
                    distinct_on[j] = remap(e)
        for w in scope.wins:
            w.args = [remap(a) for a in w.args]
            if w.filter is not None:
                w.filter = remap(w.filter)
            spec = w.win
            spec.partition = [remap(p) for p in spec.partition]
            spec.order = [(remap(o[0]), o[1], o[2]) for o in spec.order]
        for gf in _walk_all(targets + ([q.having] if q.having is not None else [])):
            if isinstance(gf, E.GroupingFunc):
                pass

    def _group_node(self, n, depth, keymap, pk_ranges, gslot, rep):
        if depth == 0:
            k = n.key()
            if k in keymap and not isinstance(n, E.Aggref):
                return E.Var(0, keymap[k], n.type, getattr(n, "rte", None), getattr(n, "col", None), orig=n)
            if isinstance(n, E.Aggref):
                return n
            if isinstance(n, E.GroupingFunc):
                refs = []
                for a in n.args:
                    ka = a.key()
                    if ka not in keymap:
                        raise PgError("arguments to GROUPING must be grouping expressions of the associated query "
                                      "level", "42803")
                    refs.append(keymap[ka])
                n.refs = refs
                n.slot = gslot
                return n
            if isinstance(n, E.WindowFunc):
                return None
            if isinstance(n, E.Var) and n.level == 0 and n.orig is None:
                for rng in pk_ranges:
                    if n.index in rng:
                        return _FDVar(n, 0, rep)
                raise PgError(f'column "{n.rte}.{n.col}" must appear in the GROUP BY clause or be used in an '
                              'aggregate function', "42803", pos=getattr(n, "_pos", None))
        else:
            if isinstance(n, E.Var) and n.level == depth and n.orig is None:
                k = E.Var(0, n.index, n.type).key()
                if k in keymap:
                    return E.Var(depth, keymap[k], n.type, n.rte, n.col, orig=None)
                for rng in pk_ranges:
                    if n.index in rng:
                        return _FDVar(n, depth, rep)
                raise PgError(f'subquery uses ungrouped column "{n.rte}.{n.col}" from outer query', "42803",
                              pos=getattr(n, "_pos", None))
        return None

    # ------------------------------------------------------------ windows
    def finish_windows(self, q, scope):
        clauses = []
        for w in scope.wins:
            spec = w.win
            hit = None
            for c in clauses:
                if _same_window(c, spec):
                    hit = c
                    break
            if hit is None:
                hit = spec
                clauses.append(spec)
            hit.funcs.append(w)
        base = q.layout
        k = 0
        for c in clauses:
            for w in c.funcs:
                w.slot = base + k
                k += 1
        q.windows = clauses
        q.layout = base + k

    def window_spec(self, wd: A.WindowDef, scope) -> WindowClause:
        ref = None
        if wd.ref is not None:
            ref = scope.named_windows.get(wd.ref)
            if ref is None:
                raise PgError(f'window "{wd.ref}" does not exist', "42704", pos=wd.pos + 1)
            if not wd.bare_ref:
                if wd.partition:
                    raise PgError(f'cannot override PARTITION BY clause of window "{wd.ref}"', "42P20",
                                  pos=wd.pos + 1)
                if wd.order and ref.order:
                    raise PgError(f'cannot override ORDER BY clause of window "{wd.ref}"', "42P20", pos=wd.pos + 1)
                if ref.frame_mode:
                    raise PgError(f'cannot copy window "{wd.ref}" because it has a frame clause', "42P20",
                                  pos=wd.pos + 1, hint="Omit the parentheses in this OVER clause.")
        partition = (ref.partition if ref is not None else []) or wd.partition
        order = (ref.order if ref is not None and ref.order else []) or wd.order
        frame_src = wd if (wd.frame_mode or ref is None) else ref
        if wd.bare_ref and ref is not None:
            frame_src = ref
        saved = scope.kind
        scope.kind = "partition"
        parts = [self.expr(p, scope) for p in partition]
        scope.kind = "window_order"
        orders = []
        for sb in order:
            e = self.expr(sb.expr, scope)
            nulls_first = sb.desc if sb.nulls is None else sb.nulls == "first"
            orders.append((e, sb.desc, nulls_first))
        scope.kind = "window_frame"
        mode = frame_src.frame_mode or "range"
        start = frame_src.start.kind if frame_src.frame_mode else "unbounded_preceding"
        if frame_src.frame_mode:
            end = frame_src.end.kind if frame_src.end is not None else "current_row"
        else:
            end = "current_row" if orders else "unbounded_following"
        if not orders and not frame_src.frame_mode:
            start, end = "unbounded_preceding", "unbounded_following"
        if start == "unbounded_following":
            raise PgError("frame start cannot be UNBOUNDED FOLLOWING", "42P20", pos=frame_src.start.pos + 1)
        if end == "unbounded_preceding":
            raise PgError("frame end cannot be UNBOUNDED PRECEDING", "42P20", pos=frame_src.end.pos + 1)
        if start == "following" and end in ("current_row", "preceding"):
            raise PgError("frame starting from following row cannot have preceding rows" if end == "preceding"
                          else "frame starting from following row cannot end with current row", "42P20",
                          pos=frame_src.end.pos + 1 if frame_src.end is not None else None)
        if start == "current_row" and end == "preceding":
            raise PgError("frame starting from current row cannot have preceding rows", "42P20",
                          pos=frame_src.end.pos + 1)
        so = eo = None
        if frame_src.frame_mode:
            if frame_src.start.offset is not None:
                so = self.frame_offset(frame_src.start.offset, mode, orders, scope)
            if frame_src.end is not None and frame_src.end.offset is not None:
                eo = self.frame_offset(frame_src.end.offset, mode, orders, scope)
        scope.kind = saved
        wc = WindowClause(parts, orders, mode, start, end, so, eo, frame_src.exclude if frame_src.frame_mode else None,
                          name=wd.ref)
        return wc

    def frame_offset(self, node, mode, orders, scope):
        e = self.expr(node, scope)
        for n in e.walk():
            if isinstance(n, E.Var) and n.level == 0:
                raise PgError(f"argument of {mode.upper()} must not contain variables", "42P10", pos=node.pos + 1)
        if mode in ("rows", "groups"):
            e = coerce_to(e, T.INT8, "a", pos=node.pos)
            v = const_fold(e)
            if v is None or v < 0:
                raise PgError("frame starting offset must not be null" if v is None else
                              "frame starting offset must not be negative", "22013")
            return v
        if len(orders) != 1:
            raise PgError("RANGE with offset PRECEDING/FOLLOWING requires exactly one ORDER BY column", "42P20",
                          pos=node.pos + 1)
        ot = orders[0][0].type
        if ot.name in ("int2", "int4", "int8", "numeric", "float4", "float8"):
            target = ot if ot.name in ("numeric", "float4", "float8") else (T.INT8 if ot.name == "int8" else T.INT4)
            if e.type.name in ("numeric",) and ot.name in ("int2", "int4", "int8"):
                raise PgError(f"RANGE with offset PRECEDING/FOLLOWING is not supported for column type "
                              f"{ot.format()} and offset type {e.type.format()}", "42883", pos=node.pos + 1,
                              hint="Cast the offset value to an appropriate type.")
            e = coerce_to(e, target if e.type.name == "unknown" or target.name != "int4" else e.type, "i",
                          pos=node.pos)
        elif ot.name in ("date", "timestamp", "timestamptz", "interval", "time"):
            e = coerce_to(e, T.INTERVAL, "i", pos=node.pos)
        else:
            raise PgError(f"RANGE with offset PRECEDING/FOLLOWING is not supported for column type {ot.format()}",
                          "42883", pos=node.pos + 1)
        v = const_fold(e)
        if v is None:
            raise PgError("frame starting offset must not be null", "22004")
        neg = v.key() < 0 if isinstance(v, T.Interval) else v < 0
        if neg:
            raise PgError("invalid preceding or following size in window function", "22013")
        return v

    # ------------------------------------------------------------ SRFs in SELECT
    def finish_srfs(self, q, scope, targets):
        base = q.layout
        q.srfs = []
        for i, f in enumerate(scope.srfs):
            q.srfs.append(f)
        slots = {id(f): base + i for i, f in enumerate(q.srfs)}

        def repl(n, d):
            if d == 0 and isinstance(n, E.Func) and n.srf and id(n) in slots:
                return E.SrfRef(slots[id(n)], n)
            return None
        for i, t in enumerate(targets):
            targets[i] = map_expr(t, repl)
        q.layout = base + len(q.srfs)

    # ------------------------------------------------------------ ORDER BY
    def sort_item(self, sb, scope, targets, names, node, what="ORDER BY"):
        e = sb.expr
        if isinstance(e, A.Const) and e.kind == "int":
            if e.value < 1 or e.value > len(targets):
                raise PgError(f"{what} position {e.value} is not in select list", "42P10", pos=e.pos + 1)
            return e.value - 1
        if isinstance(e, A.ColumnRef) and len(e.names) == 1:
            nm = e.names[0]
            hits = [i for i, n in enumerate(names) if n == nm]
            if hits:
                if len(hits) > 1:
                    k0 = targets[hits[0]].key()
                    if any(targets[h].key() != k0 for h in hits[1:]):
                        raise PgError(f'{what} "{nm}" is ambiguous', "42702", pos=e.pos + 1)
                return hits[0]
        be = self.expr(e, scope)
        if be.type.name == "unknown":
            be = coerce_to(be, T.TEXT)
        return be

    # ------------------------------------------------------------ FROM
    def from_clause(self, items, scope):
        nodes = []
        for it in items:
            nodes.append(self.from_item(it, scope))
        if len(nodes) == 1:
            return nodes[0]
        return ListNode(nodes, 0, scope.width)

    def check_alias_conflict(self, scope, rte, pos):
        if rte.name is None:
            return
        for it in scope.ns:
            if it.rel_visible and it.rte.name == rte.name and it.rte.kind != "join":
                raise PgError(f'table name "{rte.name}" specified more than once', "42712")

    def from_item(self, it, scope, add_ns=True):
        if isinstance(it, A.RangeVar):
            node = self.range_var(it, scope)
        elif isinstance(it, A.RangeSubselect):
            node = self.range_subselect(it, scope)
        elif isinstance(it, A.RangeFunction):
            node = self.range_function(it, scope)
        elif isinstance(it, A.JoinExpr):
            return self.join(it, scope, add_ns)
        else:
            raise PgError("unsupported FROM item", "0A000")
        if add_ns:
            self.check_alias_conflict(scope, node.rte, getattr(it, "pos", -1))
            scope.ns.append(NSItem(node.rte))
        return node

    def range_var(self, rv: A.RangeVar, scope):
        alias = rv.alias
        if rv.schema is None:
            cte, lvl = scope.find_cte(rv.name)
            if cte is not None:
                return self.cte_ref(rv, cte, lvl, scope)
        rel = self.db.find_rel(rv.name, rv.schema, self.session)
        if rel is None:
            sysrel = self.system_relation(rv, scope)
            if sysrel is not None:
                return sysrel
            shown = f"{rv.schema}.{rv.name}" if rv.schema else rv.name
            raise PgError(f'relation "{shown}" does not exist', "42P01", pos=rv.pos + 1)
        name = alias.name if alias else rv.name
        if isinstance(rel, Table) or (isinstance(rel, View) and rel.materialized):
            if isinstance(rel, View) and not rel.populated:
                raise PgError(f'materialized view "{rel.name}" has not been populated', "55000",
                              hint="Use the REFRESH MATERIALIZED VIEW command.")
            colnames = rel.colnames if isinstance(rel, Table) else list(rel.colnames)
            coltypes = rel.coltypes if isinstance(rel, Table) else list(rel.coltypes)
            colnames = self.alias_cols(colnames, alias, rv)
            rte = RTE("rel", name, colnames, coltypes, scope.alloc(len(colnames)), alias is not None)
            rte.relname = rv.name
            rte.rel = rel
            if isinstance(rel, Table):
                pk = rel.pkey()
                rte.pk_cols = list(pk.cols) if pk else None
            rte.sample = rv.sample
            node = ScanRel(rte, rel)
            if rv.sample is not None:
                method = rv.sample[0]
                if method not in ("bernoulli", "system"):
                    raise PgError(f'tablesample method {method} does not exist', "42704", pos=rv.pos + 1)
                s = self.new_scope(scope.parent)
                args = [coerce_to(self.expr(a, s), T.FLOAT4) for a in rv.sample[1]]
                seed = coerce_to(self.expr(rv.sample[2], s), T.FLOAT8) if rv.sample[2] is not None else None
                node.sample = (method, args, seed)
            return node
        if isinstance(rel, View):
            sub = Analyzer(self.session, self.db).query(rel.query, None)
            colnames = list(rel.colnames)
            colnames = self.alias_cols(colnames, alias, rv)
            rte = RTE("subquery", name, colnames, list(rel.coltypes), scope.alloc(len(colnames)), alias is not None)
            rte.relname = rv.name
            rte.query = sub
            rte.rel = rel
            return ScanSub(rte, sub)
        if isinstance(rel, Sequence):
            colnames = ["last_value", "log_cnt", "is_called"]
            rte = RTE("rel", name, colnames, [T.INT8, T.INT8, T.BOOL], scope.alloc(3), alias is not None)
            rte.rel = rel
            rte.relname = rv.name
            return ScanRel(rte, rel)
        raise PgError(f'"{rv.name}" is an index', "42809", pos=rv.pos + 1)

    def system_relation(self, rv, scope):
        from .sysviews import system_view
        res = system_view(self, rv)
        if res is None:
            return None
        names, types, rows = res
        alias = rv.alias
        name = alias.name if alias else rv.name
        colnames = self.alias_cols(names, alias, rv)
        rte = RTE("rel", name, colnames, types, scope.alloc(len(colnames)), alias is not None)
        rte.relname = rv.name
        fake = Table(rv.name, rv.schema or "pg_catalog", [])
        fake.rows = rows
        rte.rel = fake
        return ScanRel(rte, fake)

    def alias_cols(self, colnames, alias, node):
        colnames = list(colnames)
        if alias is not None and alias.cols:
            if len(alias.cols) > len(colnames):
                raise PgError(f'table "{alias.name}" has {len(colnames)} columns available but {len(alias.cols)} '
                              'columns specified', "42P10", pos=node.pos + 1)
            colnames[:len(alias.cols)] = alias.cols
        return colnames

    def cte_ref(self, rv, cte, lvl, scope):
        cte.refcount += 1
        alias = rv.alias
        name = alias.name if alias else rv.name
        colnames = self.alias_cols(cte.colnames, alias, rv)
        rte = RTE("cte", name, colnames, list(cte.coltypes), scope.alloc(len(colnames)), alias is not None)
        rte.relname = rv.name
        return ScanCte(rte, cte, lvl)

    def range_subselect(self, rs: A.RangeSubselect, scope):
        if rs.lateral:
            parent = self.lateral_scope(scope)
        else:
            parent = self.barrier_scope(scope)
        sub = self.query(rs.query, parent)
        alias = rs.alias
        name = alias.name if alias else None
        colnames = self.alias_cols(sub.names[:sub.ntargets], alias, rs)
        rte = RTE("subquery", name, colnames, sub.types[:sub.ntargets], scope.alloc(len(colnames)), alias is not None)
        rte.query = sub
        node = ScanSub(rte, sub, lateral=rs.lateral)
        node.uses_outer = any(v.level == 1 for v in getattr(sub, "outer_refs", [])) if rs.lateral else False
        return node

    def lateral_scope(self, scope):
        """A scope for LATERAL items: the FROM items to the left are visible as an outer level."""
        lat = Scope(scope.parent, self.session, self.db)
        lat.ns = [NSItem(it.rte, it.rel_visible, it.cols_visible) for it in scope.ns]
        lat.ctes = scope.ctes
        lat.kind = "from"
        lat.width = scope.width
        lat.lateral_level = True
        wrapper = Scope(lat, self.session, self.db)
        return lat

    def barrier_scope(self, scope):
        """A scope whose own items are invisible (a plain subquery in FROM): only outer levels resolve."""
        b = Scope(scope.parent, self.session, self.db)
        b.ctes = scope.ctes
        b.barrier = True
        b.hidden_ns = list(scope.ns)
        return b

    def range_function(self, rf: A.RangeFunction, scope):
        # function arguments see the FROM items to the left (functions in FROM are implicitly LATERAL)
        saved = scope.kind
        scope.kind = "from_func"
        funcs = []
        calls = []
        for fc in rf.funcs:
            if isinstance(fc, A.FuncCall) and fc.name[-1] == "unnest" and len(fc.args) > 1:
                calls += [A.FuncCall(["unnest"], [a], pos=fc.pos) for a in fc.args]
            else:
                calls.append(fc)
        rf_funcs = calls
        for fc in calls:
            f = self.func_in_from(fc, scope)
            funcs.append(f)
        scope.kind = saved
        alias = rf.alias
        colnames, coltypes = [], []
        for f, fc in zip(funcs, rf_funcs):
            names, types = f._cols
            colnames += names
            coltypes += types
        if rf.coldefs:
            if not all(getattr(f, "_record", False) for f in funcs):
                raise PgError("a column definition list is only allowed for functions returning \"record\"",
                              "42601", pos=rf.pos + 1)
            colnames = [c.name for c in rf.coldefs]
            coltypes = [self.resolve_type(c.typ) for c in rf.coldefs]
            for f in funcs:
                f._coldefs = list(zip(colnames, coltypes))
        elif any(getattr(f, "_record", False) and not getattr(f, "_outcols", False) for f in funcs):
            raise PgError("a column definition list is required for functions returning \"record\"", "42601",
                          pos=rf.pos + 1)
        if len(funcs) == 1 and len(colnames) == 1 and alias is not None and not alias.cols and \
                not getattr(funcs[0], "_record", False):
            colnames = [alias.name]
        if len(funcs) > 1 and len(set(colnames)) < len(colnames) and alias is None:
            pass
        if rf.ordinality:
            colnames.append("ordinality")
            coltypes.append(T.INT8)
        name = alias.name if alias else (rf.funcs[0].name[-1] if isinstance(rf.funcs[0], A.FuncCall)
                                         else rf.funcs[0].name)
        colnames = self.alias_cols(colnames, alias, rf)
        slots = scope.alloc(len(colnames))
        rte = RTE("func", name, colnames, coltypes, slots, alias is not None)
        node = ScanFunc(rte, funcs, rf.ordinality, slots[0] if slots else scope.width, len(colnames))
        return node

    def func_in_from(self, fc, scope):
        if isinstance(fc, A.SQLValue):
            e = self.expr(fc, scope)
            f = E.Func("const", [e], e.type, lambda a, c: a[0], strict=False)
            f._cols = ([fc.name], [e.type])
            f._srf = False
            return f
        name = fc.name[-1]
        if name in SRF_NAMES or (name in T.PROCS and any(p.retset for p in T.PROCS[name])):
            e = self.func_call(fc, scope, srf_ok=True)
        else:
            e = self.expr(fc, scope)
        if isinstance(e, E.Func) and e.srf:
            e._cols = e._srfcols
            e._record = getattr(e, "_record", False)
            return e
        # a scalar function in FROM: one row, one column named after the function
        f = E.Func("scalar", [e], e.type, lambda a, c: a[0], strict=False)
        f._cols = ([name], [e.type])
        f._srf = False
        return f

    def join(self, j: A.JoinExpr, scope, add_ns=True):
        left = self.from_item(j.left, scope, add_ns=True)
        lns = list(scope.ns)
        right = self.from_item(j.right, scope, add_ns=True)
        left_rtes_ns = [it for it in lns]
        base = left.base
        lrte, rrte = left.rte, right.rte
        lcols = _visible_cols(lrte)
        rcols = _visible_cols(rrte)
        using = j.using
        if j.natural:
            lnames = [n for n, _, _ in lcols]
            rnames = {n for n, _, _ in rcols}
            using = [n for n in lnames if n in rnames]
        merge = []
        jnames, jtypes, jslots = [], [], []
        quals = None
        kind = j.kind
        hidden = set()
        if using is not None:
            conds = []
            for u in using:
                lh = [x for x in lcols if x[0] == u]
                rh = [x for x in rcols if x[0] == u]
                if not lh:
                    raise PgError(f'column "{u}" specified in USING clause does not exist in left table', "42703",
                                  pos=j.pos + 1)
                if not rh:
                    raise PgError(f'column "{u}" specified in USING clause does not exist in right table', "42703",
                                  pos=j.pos + 1)
                if len(lh) > 1:
                    raise PgError(f'common column name "{u}" appears more than once in left table', "42702",
                                  pos=j.pos + 1)
                if len(rh) > 1:
                    raise PgError(f'common column name "{u}" appears more than once in right table', "42702",
                                  pos=j.pos + 1)
                (_, lt, ls), (_, rt, rs) = lh[0], rh[0]
                lv = E.Var(0, ls, lt, lrte.name, u)
                rv = E.Var(0, rs, rt, rrte.name, u)
                conds.append(self.make_op("=", lv, rv, j.pos))
                ct = C.common_type([lt, rt], "JOIN/USING", None)
                if kind == "full":
                    merge.append((ls, rs))
                    jslots.append(None)
                elif kind == "right":
                    jslots.append(rs)
                else:
                    jslots.append(ls)
                jnames.append(u)
                jtypes.append(ct if kind == "full" else (rt if kind == "right" else lt))
            quals = E.make_and(conds)
            used = set(using)
            for n, t, s in lcols:
                if n not in used:
                    jnames.append(n)
                    jtypes.append(t)
                    jslots.append(s)
            for n, t, s in rcols:
                if n not in used:
                    jnames.append(n)
                    jtypes.append(t)
                    jslots.append(s)
            hidden = used
        else:
            for n, t, s in lcols + rcols:
                jnames.append(n)
                jtypes.append(t)
                jslots.append(s)
        if merge:
            mslots = scope.alloc(len(merge))
            k = 0
            for i, s in enumerate(jslots):
                if s is None:
                    jslots[i] = mslots[k]
                    k += 1
        width = scope.width - base
        alias = j.alias
        jrte = RTE("join", alias.name if alias else None, jnames, jtypes, jslots, alias is not None)
        jrte.merged = set(using or [])
        bysl = {}
        for r in (lrte, rrte):
            for k, sl in enumerate(r.slots):
                bysl.setdefault(sl, _src_of(r, k))
        jrte.src = [bysl.get(sl, (None, n)) for sl, n in zip(jslots, jnames)]
        if alias is not None and alias.cols:
            jrte.colnames[:len(alias.cols)] = alias.cols
        if j.quals is not None:
            saved = scope.kind
            scope.kind = "on"
            e = self.expr(j.quals, scope)
            scope.kind = saved
            quals = self.bool_expr(e, "JOIN/ON", j.quals)
        # namespace: the join's columns replace its inputs' for unqualified references
        for it in scope.ns:
            if it.rte in _rtes_of(left) or it.rte in _rtes_of(right):
                it.cols_visible = False
                if alias is not None:
                    it.rel_visible = False
        node = JoinNode(kind, left, right, quals, merge, base, width, jrte)
        scope.ns.append(NSItem(jrte, rel_visible=alias is not None, cols_visible=True))
        return node

    # ------------------------------------------------------------ name resolution
    def find_rte(self, names, scope, pos):
        """t or schema.t in a qualified column reference -> (RTE, levels up)."""
        tname = names[-1]
        sname = names[-2] if len(names) > 1 else None
        s, lvl = scope, 0
        while s is not None:
            hits = [it.rte for it in s.ns if it.rel_visible and it.rte.name == tname and
                    (sname is None or (it.rte.rel is not None and it.rte.rel.schema == sname and
                                       not it.rte.alias_given))]
            if len(hits) > 1:
                raise PgError(f'table reference "{tname}" is ambiguous', "42P09", pos=pos + 1)
            if hits:
                return hits[0], lvl
            s = s.parent
            lvl += 1
        # an aliased table referenced by its real name?
        s = scope
        while s is not None:
            for it in s.ns + s.hidden_ns:
                r = it.rte
                if r.relname == tname and r.alias_given:
                    raise PgError(f'invalid reference to FROM-clause entry for table "{tname}"', "42P01", pos=pos + 1,
                                  hint=f'Perhaps you meant to reference the table alias "{r.name}".')
            for it in s.hidden_ns:
                if it.rte.name == tname:
                    raise PgError(f'invalid reference to FROM-clause entry for table "{tname}"', "42P01",
                                  pos=pos + 1,
                                  hint=f'There is an entry for table "{tname}", but it cannot be referenced from '
                                       'this part of the query.')
            s = s.parent
        raise PgError(f'missing FROM-clause entry for table "{tname}"', "42P01", pos=pos + 1)

    def column_ref(self, node: A.ColumnRef, scope):
        names = node.names
        pos = node.pos
        if len(names) == 1:
            return self.unqualified(names[0], scope, pos)
        if len(names) > 4:
            raise PgError(f"improper qualified name (too many dotted names): {'.'.join(names)}", "42601",
                          pos=pos + 1)
        # t.col, schema.t.col, or composite.field
        col = names[-1]
        try:
            rte, lvl = self.find_rte(names[:-1], scope, pos)
        except PgError as err:
            if len(names) == 2:
                # maybe (column).field
                try:
                    base = self.unqualified(names[0], scope, pos)
                except PgError:
                    raise err from None
                return self.field_select(base, col, pos)
            raise
        hits = rte.find(col)
        if not hits:
            if col == "*":
                raise PgError("row expansion via \"*\" is not supported here", "0A000", pos=pos + 1)
            # whole-row reference to a column named like the table? no: error
            raise PgError(f"column {rte.name}.{col} does not exist", "42703", pos=pos + 1,
                          hint=self.fuzzy_hint(col, scope, rte))
        if len(hits) > 1:
            raise PgError(f'column reference "{col}" is ambiguous', "42702", pos=pos + 1)
        v = E.Var(lvl, rte.slots[hits[0]], rte.coltypes[hits[0]], rte.name, col)
        v._pos = pos + 1
        if lvl > 0:
            self.note_outer(v, scope)
        return v

    def unqualified(self, name, scope, pos):
        s, lvl = scope, 0
        while s is not None:
            found = []
            for it in s.ns:
                if not it.cols_visible:
                    continue
                rte = it.rte
                for i, n in enumerate(rte.colnames):
                    if n == name:
                        found.append((rte, i))
            if len(found) > 1:
                slots = {r.slots[i] for r, i in found}
                if len(slots) > 1:
                    raise PgError(f'column reference "{name}" is ambiguous', "42702", pos=pos + 1)
            if found:
                rte, i = found[0]
                rn, cn = _src_of(rte, i)
                v = E.Var(lvl, rte.slots[i], rte.coltypes[i], rn, cn)
                v._pos = pos + 1
                if lvl > 0:
                    self.note_outer(v, scope)
                return v
            # a whole-row reference: SELECT u FROM users u
            for it in s.ns:
                if it.rel_visible and it.rte.name == name and it.rte.kind != "join":
                    rte = it.rte
                    items = [E.Var(lvl, rte.slots[i], rte.coltypes[i], rte.name, n) for i, n in
                             enumerate(rte.colnames)]
                    for v in items:
                        if lvl > 0:
                            self.note_outer(v, scope)
                    return E.RowExpr(items, list(rte.colnames))
            s = s.parent
            lvl += 1
        # not found: hints like PostgreSQL's
        hint = self.fuzzy_hint(name, scope, None)
        for st in self._scopes(scope):
            for it in st.hidden_ns:
                if name in it.rte.colnames:
                    hint = (f'There is a column named "{name}" in table "{it.rte.name}", but it cannot be referenced '
                            'from this part of the query.')
        raise PgError(f'column "{name}" does not exist', "42703", pos=pos + 1, hint=hint)

    def _scopes(self, scope):
        while scope is not None:
            yield scope
            scope = scope.parent

    def fuzzy_hint(self, name, scope, only_rte):
        """errorMissingColumn's "Perhaps you meant to reference the column ...": the closest name by edit
        distance, if close enough."""
        best = None
        best_d = None
        cands = []
        for st in self._scopes(scope):
            for it in st.ns:
                rte = it.rte
                if rte.kind == "join" or (only_rte is not None and rte is not only_rte):
                    continue
                for n in rte.colnames:
                    cands.append((rte.name, n))
        for rn, n in cands:
            d = _levenshtein(name, n)
            if d > max(len(name), len(n)) // 2:
                continue
            if best_d is None or d < best_d:
                best, best_d = (rn, n), d
            elif d == best_d and best is not None and best != (rn, n):
                best = ("?", "?")
        if best is None or best == ("?", "?") or best_d == 0 and only_rte is not None:
            return None
        if best_d is not None and best_d > 0 or only_rte is None:
            rn, n = best
            if rn is None:
                return f'Perhaps you meant to reference the column "{n}".'
            return f'Perhaps you meant to reference the column "{rn}.{n}".'
        return None

    def field_select(self, base, field, pos):
        if isinstance(base, E.RowExpr) and base.names:
            if field in base.names:
                i = base.names.index(field)
                return E.FieldSelect(base, i, field, base.items[i].type)
        if base.type.name == "jsonb" or base.type.name == "json":
            raise PgError(f"column notation .{field} applied to type {base.type.format()}, which is not a composite "
                          "type", "42809", pos=pos + 1)
        raise PgError(f'missing FROM-clause entry for table "{getattr(base, "col", field)}"', "42P01", pos=pos + 1)

    # ------------------------------------------------------------ expressions
    def expr(self, node, scope) -> E.BE:
        m = getattr(self, "x_" + type(node).__name__, None)
        if m is None:
            raise PgError(f"unsupported expression ({type(node).__name__})", "0A000",
                          pos=getattr(node, "pos", -1) + 1)
        return m(node, scope)

    def x_Const(self, n: A.Const, scope):
        k = n.kind
        if k == "int":
            v = n.value
            if -2147483648 <= v <= 2147483647:
                return E.Const(v, T.INT4, n.pos)
            if -9223372036854775808 <= v <= 9223372036854775807:
                return E.Const(v, T.INT8, n.pos)
            return E.Const(T.Decimal(v), T.NUMERIC, n.pos)
        if k == "numeric":
            return E.Const(T.num_in(n.value), T.NUMERIC, n.pos)
        if k == "str":
            return E.Const(n.value, T.UNKNOWN, n.pos)
        if k == "bool":
            return E.Const(n.value, T.BOOL, n.pos)
        if k == "null":
            return E.Const(None, T.UNKNOWN, n.pos)
        if k == "bitstr":
            raise Unsupported("bit-string literals", n.pos + 1)
        raise AssertionError(k)

    def x_TypedLiteral(self, n: A.TypedLiteral, scope):
        t = self.resolve_type(n.typ)
        try:
            v = T.input_value(n.value, t.base, t.mod, now=_now_holder[0], explicit=True)
        except PgError as err:
            if err.pos is None:
                err.pos = n.pos + len(" ".join(n.typ.names)) + 2 if False else _literal_pos(n)
            raise
        return E.Const(v, t, n.pos)

    def x_ColumnRef(self, n: A.ColumnRef, scope):
        if n.names[-1] == "*":
            raise PgError("row expansion via \"*\" is not supported here", "0A000", pos=n.pos + 1)
        return self.column_ref(n, scope)

    def x_Param(self, n: A.Param, scope):
        if n.n > len(self.param_types):
            raise PgError(f"there is no parameter ${n.n}", "42P02", pos=n.pos + 1)
        return E.Param(n.n, self.param_types[n.n - 1])

    def x_Star(self, n, scope):
        raise PgError("syntax error at or near \"*\"", "42601", pos=n.pos + 1)

    def make_op(self, op, left: E.BE | None, right: E.BE, pos):
        if left is None:
            params, ret = C.find_operator(op, None, right.type, pos=pos + 1 if pos is not None and pos >= 0 else None)
            r = coerce_to(right, params[0], pos=pos)
            return E.Op(op, [r], ret, op_impl(op, params, ret), pos)
        lt = left.type
        rt = right.type
        params, ret = C.find_operator(op, lt, rt, pos=pos + 1 if pos is not None and pos >= 0 else None)
        l2 = coerce_to(left, params[0], pos=pos)
        r2 = coerce_to(right, params[1], pos=pos)
        return E.Op(op, [l2, r2], ret, op_impl(op, params, ret), pos)

    def x_Op(self, n: A.Op, scope):
        op = n.op
        if n.left is None:
            r = self.expr(n.right, scope)
            if op == "-" and isinstance(r, E.Const) and r.type.name in ("int4", "int8", "numeric") and \
                    r.value is not None:
                pass
            return self.make_op(op, None, r, n.pos)
        left = self.expr(n.left, scope)
        right = self.expr(n.right, scope)
        if op == "=" and (isinstance(n.right, A.Const) and n.right.kind == "null"):
            pass    # "= NULL" is legal and always NULL
        if left.type.name == "record" and right.type.name == "record" and op in ("=", "<>", "<", ">", "<=", ">="):
            return self.row_compare(op, left, right, n.pos)
        return self.make_op(op, left, right, n.pos)

    def row_compare(self, op, left, right, pos):
        li = left.items if isinstance(left, E.RowExpr) else None
        ri = right.items if isinstance(right, E.RowExpr) else None
        if li is None or ri is None:
            raise Unsupported("comparisons of composite values", pos + 1)
        if len(li) != len(ri):
            raise PgError("unequal number of entries in row expressions", "42601", pos=pos + 1)
        pairs = [self.make_op(op if op in ("=", "<>") else "=", a, b, pos) for a, b in zip(li, ri)]
        if op == "=":
            return E.Bool("and", pairs)
        if op == "<>":
            return E.Bool("or", pairs)
        # lexicographic < > <= >=
        ltf = [self.make_op(op.rstrip("="), a, b, pos) for a, b in zip(li, ri)]
        eqf = [self.make_op("=", a, b, pos) for a, b in zip(li, ri)]
        last = self.make_op(op, li[-1], ri[-1], pos)
        terms = []
        for k in range(len(li) - 1):
            terms.append(E.make_and(eqf[:k] + [ltf[k]]))
        terms.append(E.make_and(eqf[:len(li) - 1] + [last]))
        return E.Bool("or", terms) if len(terms) > 1 else terms[0]

    def x_BoolOp(self, n: A.BoolOp, scope):
        args = []
        for a in n.args:
            e = self.expr(a, scope)
            if e.type.name == "unknown":
                e = coerce_to(e, T.BOOL, pos=a.pos)
            if e.type != T.BOOL:
                raise PgError(f"argument of {n.op.upper()} must be type boolean, not type {e.type.format()}", "42804",
                              pos=getattr(a, "pos", -1) + 1)
            args.append(e)
        if n.op != "not":
            flat = []
            for a in args:
                if isinstance(a, E.Bool) and a.op == n.op:
                    flat += a.args
                else:
                    flat.append(a)
            args = flat
        return E.Bool(n.op, args)

    def x_NullTest(self, n, scope):
        return E.NullTest(self.expr(n.arg, scope), n.negated)

    def x_BoolTest(self, n, scope):
        a = self.expr(n.arg, scope)
        if a.type.name == "unknown":
            a = coerce_to(a, T.BOOL, pos=n.pos)
        if a.type != T.BOOL:
            raise PgError(f"argument of IS {'NOT ' if n.negated else ''}{n.what.upper()} must be type boolean, not "
                          f"type {a.type.format()}", "42804", pos=n.arg.pos + 1)
        return E.BoolTest(a, n.what, n.negated)

    def x_Distinct(self, n, scope):
        left = self.expr(n.left, scope)
        right = self.expr(n.right, scope)
        op = self.make_op("=", left, right, n.pos)
        return E.Distinct(op.args, op.impl, n.negated)

    def x_InList(self, n: A.InList, scope):
        arg = self.expr(n.arg, scope)
        items = [self.expr(x, scope) for x in n.items]
        all_const = all(isinstance(x, E.Const) for x in items)
        ct = C.common_type([arg.type] + [x.type for x in items], "IN", None, silent=True)
        if ct is not None and all_const and ct.name != "record":
            # x IN (1, 2, 3) -> x = ANY ('{1,2,3}')
            if ct.name == "unknown":
                ct = T.TEXT
            try:
                arg2 = coerce_to(arg, ct, pos=n.pos)
                consts = [coerce_to(x, ct, pos=getattr(n.items[i], "pos", None)) for i, x in enumerate(items)]
                params, ret = C.find_operator("=", ct, ct, pos=n.pos + 1)
                arr = E.Const([c.value for c in consts], ct.array(), n.pos)
                sa = E.ScalarArrayOp("=" if not n.negated else "<>", not n.negated, arg2, arr,
                                     op_impl("=" if not n.negated else "<>", params, ret), from_in=True)
                return sa
            except PgError:
                pass
        eqs = []
        exprs = []
        for x, src in zip(items, n.items):
            op = self.make_op("=", arg, x, getattr(src, "pos", n.pos))
            eqs.append(op.impl)
            exprs.append(op.args[1])
        if eqs:
            arg = self.make_op("=", arg, items[0], n.pos).args[0]
        return E.InList(arg, exprs, eqs, n.negated)

    def x_Between(self, n: A.Between, scope):
        a = self.expr(n.arg, scope)
        lo = self.expr(n.lo, scope)
        hi = self.expr(n.hi, scope)
        if n.symmetric:
            ge1 = self.make_op(">=", a, lo, n.pos)
            le1 = self.make_op("<=", a, hi, n.pos)
            ge2 = self.make_op(">=", a, hi, n.pos)
            le2 = self.make_op("<=", a, lo, n.pos)
            e = E.Bool("or", [E.Bool("and", [ge1, le1]), E.Bool("and", [ge2, le2])])
        else:
            e = E.Bool("and", [self.make_op(">=", a, lo, n.pos), self.make_op("<=", a, hi, n.pos)])
        if n.negated:
            if n.symmetric:
                return E.Bool("not", [e])
            lt = self.make_op("<", a, lo, n.pos)
            gt = self.make_op(">", a, hi, n.pos)
            return E.Bool("or", [lt, gt])
        return e

    def x_Like(self, n: A.Like, scope):
        a = self.expr(n.arg, scope)
        p = self.expr(n.pattern, scope)
        if n.kind == "similar":
            esc = self.expr(n.escape, scope) if n.escape is not None else None
            a = coerce_to(a, T.TEXT, pos=n.pos) if a.type.name in T.STRINGS else a
            p = coerce_to(p, T.TEXT, pos=n.pos)
            if a.type.name not in T.STRINGS:
                raise PgError(f"operator does not exist: {a.type.format()} ~ text", "42883", pos=n.pos + 1,
                              hint=C.HINT_OP)
            from .funcs import similar_regex
            neg = n.negated
            escf = esc.compile() if esc is not None else None

            def sim(args, c):
                s, pat = args[0], args[1]
                e2 = args[2] if len(args) > 2 else "\\"
                return (similar_regex(pat, e2).match(s) is not None) != neg
            args = [a, p] + ([coerce_to(esc, T.TEXT)] if esc is not None else [])
            return E.Func("similar_to", args, T.BOOL, sim, display="similar")
        op = {"like": "~~", "ilike": "~~*"}[n.kind]
        if n.negated:
            op = "!" + op
        if n.escape is not None:
            esc = self.expr(n.escape, scope)
            esc = coerce_to(esc, T.TEXT)
            a = coerce_to(a, T.TEXT, pos=n.pos) if a.type.name == "unknown" else a
            pt = coerce_to(p, T.TEXT, pos=n.pos)
            from .funcs import like
            ci = n.kind == "ilike"
            neg = n.negated

            def lk(args, c):
                s, pat, e2 = args
                if len(e2) > 1:
                    raise PgError("invalid escape string", "22025",
                                  hint="Escape string must be empty or one character.")
                return like(s, pat, e2 or None, ci) != neg
            return E.Func("like_escape", [a, pt, esc], T.BOOL, lk, display="like")
        return self.make_op(op, a, p, n.pos)

    def x_Case(self, n: A.Case, scope):
        arg = None
        whens = []
        if n.arg is not None:
            arg = self.expr(n.arg, scope)
            test = E.CaseTest(arg.type)
            for cond, res in n.whens:
                c = self.expr(cond, scope)
                op = self.make_op("=", test, c, getattr(cond, "pos", n.pos))
                whens.append((op, res))
            if arg.type.name == "unknown":
                arg = coerce_to(arg, T.TEXT)
        else:
            for cond, res in n.whens:
                c = self.expr(cond, scope)
                if c.type.name == "unknown":
                    c = coerce_to(c, T.BOOL, pos=cond.pos)
                if c.type != T.BOOL:
                    raise PgError(f"argument of CASE/WHEN must be type boolean, not type {c.type.format()}", "42804",
                                  pos=getattr(cond, "pos", -1) + 1)
                whens.append((c, res))
        results = [self.expr(res, scope) for _, res in whens]
        dflt = self.expr(n.default, scope) if n.default is not None else None
        allr = results + ([dflt] if dflt is not None else [])
        positions = [getattr(r, "pos", None) for _, r in n.whens] + ([getattr(n.default, "pos", None)]
                                                                      if n.default is not None else [])
        ct = C.common_type([r.type for r in allr], "CASE", positions)
        for i, r in enumerate(results):
            results[i] = coerce_to(r, ct, pos=positions[i])
        if dflt is not None:
            dflt = coerce_to(dflt, ct, pos=positions[-1])
        return E.Case(arg, [(c, r) for (c, _), r in zip(whens, results)], dflt, ct)

    def x_Cast(self, n: A.Cast, scope):
        t = self.resolve_type(n.typ)
        if isinstance(n.arg, A.ArrayExpr) and t.is_array:
            a = self.x_ArrayExpr(n.arg, scope, elem_hint=t.elem)
            return coerce_to(a, t, "e", pos=n.pos, explicit=True) if a.type != t else a
        a = self.expr(n.arg, scope)
        if isinstance(a, E.Const) and a.type.name == "unknown":
            if a.value is None:
                return E.Const(None, t, a.pos)
            try:
                v = T.input_value(a.value, t.base, t.mod, now=_now_holder[0], explicit=True)
            except PgError as err:
                if err.pos is None:
                    err.pos = a.pos + 1 if a.pos is not None else None
                raise
            c = E.Const(v, t, a.pos)
            c._cast_src = n
            return c
        if a.type == t and (t.mod is None or a.type.mod == t.mod):
            return E.Cast(a, t, lambda v, c: v, True, pos=n.pos) if False else _relabel(a, t, n)
        return coerce_to(a, t, "e", pos=n.pos, explicit=True,
                         what=PgError(f"cannot cast type {a.type.format()} to {t.format()}", "42846",
                                      pos=n.pos + 1))

    def x_Collate(self, n, scope):
        a = self.expr(n.arg, scope)
        coll = n.collation[-1]
        if coll not in ("C", "POSIX", "default", "C.UTF-8", "c", "posix", "ucs_basic", "en_US", "en-US-x-icu",
                        "und-x-icu"):
            raise PgError(f'collation "{coll}" for encoding "UTF8" does not exist', "42704", pos=n.pos + 1)
        if a.type.name not in T.STRINGS:
            raise PgError(f"collations are not supported by type {a.type.format()}", "42804", pos=n.pos + 1)
        return a

    def x_ArrayExpr(self, n: A.ArrayExpr, scope, elem_hint=None):
        items = []
        for x in n.items:
            if isinstance(x, A.ArrayExpr):
                items.append(self.x_ArrayExpr(x, scope))
            else:
                items.append(self.expr(x, scope))
        if not items:
            if elem_hint is not None:
                return E.Const([], elem_hint.array(), n.pos)
            raise PgError("cannot determine type of empty array", "42P18", pos=n.pos + 1,
                          hint='Explicitly cast to the desired type, for example ARRAY[]::integer[].')
        ct = C.common_type([x.type for x in items], "ARRAY", [getattr(x, "pos", None) for x in n.items])
        if ct.name == "unknown":
            ct = T.TEXT
        items = [coerce_to(x, ct, pos=getattr(src, "pos", None)) for x, src in zip(items, n.items)]
        at = ct if ct.is_array else ct.array()
        if all(isinstance(x, E.Const) for x in items):
            return E.Const([x.value for x in items], at, n.pos)
        return E.ArrayExpr(items, at)

    def x_RowExpr(self, n: A.RowExpr, scope):
        items = [self.expr(x, scope) for x in n.items]
        return E.RowExpr(items, [f"f{i + 1}" for i in range(len(items))])

    def x_Subscript(self, n: A.Subscript, scope):
        subs = []
        node = n
        chain = []
        while isinstance(node, A.Subscript):
            chain.append(node)
            node = node.arg
        chain.reverse()
        base = self.expr(node, scope)
        t = base.type
        if t.name in ("jsonb",):
            # jsonb subscripting: ['key'] / [0]
            out = base
            for s in chain:
                if s.slice:
                    raise PgError("jsonb subscript does not support slices", "42804", pos=s.pos + 1)
                k = self.expr(s.lo, scope)
                if k.type.name in ("int4", "int2", "int8"):
                    out = self.make_op("->", out, coerce_to(k, T.INT4), s.pos)
                else:
                    out = self.make_op("->", out, coerce_to(k, T.TEXT), s.pos)
            return out
        if not t.is_array:
            raise PgError(f"cannot subscript type {t.format()} because it does not support subscripting", "42804",
                          pos=getattr(node, "pos", n.pos) + 1)
        any_slice = any(s.slice for s in chain)
        for s in chain:
            lo = coerce_to(self.expr(s.lo, scope), T.INT4, "a", pos=s.pos) if s.lo is not None else None
            hi = coerce_to(self.expr(s.hi, scope), T.INT4, "a", pos=s.pos) if s.hi is not None else None
            subs.append((lo, hi, s.slice))
        rt = t if any_slice else t.elem
        if not any_slice and len(chain) > 1:
            rt = t.elem
        return E.Subscript(base, subs, rt)

    def x_FieldSel(self, n: A.FieldSel, scope):
        base = self.expr(n.arg, scope)
        return self.field_select(base, n.name, n.pos)

    def x_AtTimeZone(self, n: A.AtTimeZone, scope):
        a = self.expr(n.arg, scope)
        if n.zone is None:
            z = E.Const(self.session.settings.get("timezone", "UTC") if self.session else "UTC", T.TEXT)
        else:
            z = self.expr(n.zone, scope)
        fc = A.FuncCall(["timezone"], [], pos=n.pos)
        return self.call_resolved("timezone", [z, a], n.pos, display="timezone")

    def x_SQLValue(self, n: A.SQLValue, scope):
        name = n.name
        if name == "current_date":
            return E.SQLValue(name, T.DATE, lambda c: T.to_local(c.txn_now)[0].date())
        if name in ("current_timestamp",):
            p = n.precision
            return E.SQLValue(name, T.TIMESTAMPTZ.with_mod(p) if p is not None else T.TIMESTAMPTZ,
                              (lambda c: T.apply_typmod(c.txn_now, T.TIMESTAMPTZ, p)) if p is not None
                              else (lambda c: c.txn_now), p)
        if name == "localtimestamp":
            return E.SQLValue(name, T.TIMESTAMP, lambda c: T.to_local(c.txn_now)[0])
        if name == "current_time":
            return E.SQLValue(name, T.TIMETZ, lambda c: (T.to_local(c.txn_now)[0].time(), T.to_local(c.txn_now)[1]))
        if name == "localtime":
            return E.SQLValue(name, T.TIME, lambda c: T.to_local(c.txn_now)[0].time())
        if name in ("current_user", "user", "current_role", "session_user"):
            return E.SQLValue(name, T.NAME, lambda c: c.session.user)
        if name == "system_user":
            return E.SQLValue(name, T.TEXT, lambda c: "peer:" + c.session.user)
        if name == "current_catalog":
            return E.SQLValue(name, T.NAME, lambda c: c.session.dbname)
        if name == "current_schema":
            return E.SQLValue(name, T.NAME, lambda c: (c.db.search(c.session) or [None])[0])
        raise Unsupported(name.upper(), n.pos + 1)

    def x_SetToDefault(self, n, scope):
        raise PgError("DEFAULT is not allowed in this context", "42601", pos=n.pos + 1)

    def x_Grouping(self, n: A.Grouping, scope):
        if scope.kind in ("where", "on", "group"):
            raise PgError(f"grouping operations are not allowed in {_err_kind(scope.kind)}", "42803", pos=n.pos + 1)
        args = [self.expr(a, scope) for a in n.args]
        g = E.GroupingFunc(args)
        scope.aggs_dummy = True
        if g not in scope.aggs:
            pass
        scope.grouping_used = True
        return g

    def x_SubLink(self, n: A.SubLink, scope):
        kind = n.kind
        if scope.kind in ("check", "default", "index", "index_pred", "generated"):
            raise PgError(f"cannot use subquery in {_err_kind(scope.kind)}", "0A000", pos=n.pos + 1)
        if scope.kind == "group":
            pass
        sub = self.query(n.query, scope)
        outer = list(getattr(sub, "outer_refs", []))
        self.subplan_n += 1
        if kind == "exists":
            sp = E.SubPlan("exists", sub, T.BOOL, pos=n.pos)
        elif kind == "expr":
            if sub.ntargets != 1:
                raise PgError("subquery must return only one column", "42601", pos=n.pos + 1)
            t = sub.types[0]
            sp = E.SubPlan("expr", sub, t, pos=n.pos)
        elif kind == "array":
            if sub.ntargets != 1:
                raise PgError("subquery must return only one column", "42601", pos=n.pos + 1)
            t = sub.types[0]
            if t.name == "unknown":
                t = T.TEXT
            sp = E.SubPlan("array", sub, t if t.is_array else t.array(), pos=n.pos)
        elif kind in ("any", "all"):
            left = self.expr(n.test, scope)
            if isinstance(left, E.RowExpr):
                lefts = left.items
            else:
                lefts = [left]
            if sub.ntargets != len(lefts):
                raise PgError("subquery has too many columns" if sub.ntargets > len(lefts) else
                              "subquery has too few columns", "42601", pos=n.pos + 1)
            impls = []
            tests = []
            for i, l in enumerate(lefts):
                ph = E.Const(None, sub.types[i])
                op = self.make_op(n.op, l, ph, n.pos)
                if op.type != T.BOOL:
                    raise PgError(f"operator {n.op} must return type boolean, not type {op.type.format()}", "42809",
                                  pos=n.pos + 1)
                tests.append(op.args[0])
                # the subquery's column may need a cast to the operator's right type
                rt = op.args[1].type
                impls.append((op.impl, sub.types[i], rt))
            test = tests[0] if len(tests) == 1 else E.RowExpr(tests)
            sp = E.SubPlan(kind, sub, T.BOOL, test=test, op_impl=impls, pos=n.pos)
            sp.name = n.op
        elif kind == "multiexpr":
            sp = E.SubPlan("multiexpr", sub, T.RECORD, pos=n.pos)
        else:
            raise Unsupported(f"{kind} subqueries", n.pos + 1)
        sp.outer_refs = outer
        sp.correlated = bool(outer)
        sp.plan_id = self.subplan_n
        if getattr(n, "negated_in", False):
            sp.negated_in = True
        return sp

    def x_AnyAll(self, n: A.AnyAll, scope):
        left = self.expr(n.left, scope)
        right = self.expr(n.right, scope)
        if isinstance(n.right, A.ArrayExpr) and right.type.name == "unknown":
            pass
        rt = right.type
        if rt.name == "unknown":
            right = coerce_to(right, left.type.array() if left.type.name != "unknown" else T.TEXT_ARRAY,
                              pos=n.pos)
            rt = right.type
        if not rt.is_array:
            raise PgError(f"op ANY/ALL (array) requires array on right side", "42809", pos=n.pos + 1)
        el = rt.elem
        ph = E.Const(None, el)
        op = self.make_op(n.op, left, ph, n.pos)
        if op.type != T.BOOL:
            raise PgError(f"op ANY/ALL (array) requires operator to yield boolean", "42809", pos=n.pos + 1)
        if op.args[1].type != el:
            right = coerce_to(right, op.args[1].type.array(), pos=n.pos)
        return E.ScalarArrayOp(n.op, n.kind == "any", op.args[0], right, op.impl)

    # ------------------------------------------------------------ function calls
    def x_FuncCall(self, n: A.FuncCall, scope):
        return self.func_call(n, scope)

    def func_call(self, n: A.FuncCall, scope, srf_ok=False):
        names = n.name
        name = names[-1]
        schema = names[0] if len(names) > 1 else None
        if schema not in (None, "pg_catalog", "public"):
            raise PgError(f"schema \"{schema}\" does not exist", "3F000", pos=n.pos + 1)
        special = n.special
        if special in ("coalesce", "greatest", "least", "nullif") or (name in ("coalesce", "greatest", "least",
                                                                              "nullif") and schema is None):
            if n.over is not None:
                raise PgError(f"OVER specified, but {name} is not a window function nor an aggregate function",
                              "42809", pos=n.pos + 1)
            return self.special_call(name, n, scope)
        if name in ("nextval", "currval", "setval", "lastval"):
            return self.seq_call(name, n, scope)
        if n.over is not None:
            return self.window_call(n, scope)
        procs = T.PROCS.get(name, [])
        is_agg = any(p.kind == "a" for p in procs) and name in AGG_NAMES
        if is_agg:
            return self.agg_call(n, scope)
        if name in WINDOW_ONLY:
            raise PgError(f"window function {name} requires an OVER clause", "42809", pos=n.pos + 1)
        if n.distinct or n.order_by or n.filter is not None or n.within_group:
            what = "DISTINCT" if n.distinct else ("ORDER BY" if n.order_by else ("FILTER" if n.filter is not None
                                                                               else "WITHIN GROUP"))
            if n.within_group:
                raise PgError(f"WITHIN GROUP specified, but {name} is not an aggregate function", "42809",
                              pos=n.pos + 1)
            raise PgError(f"{what} specified, but {name} is not an aggregate function", "42809", pos=n.pos + 1)
        if n.star:
            raise PgError(f"{name}(*) specified, but {name} is not an aggregate function", "42809", pos=n.pos + 1)
        args = [self.expr(a, scope) for a in n.args]
        if n.argnames:
            args = self.named_args(name, n, args)
        if name in SRF_NAMES or any(p.retset for p in procs):
            return self.srf_call(name, n, args, scope, srf_ok)
        if name == "extract" and n.special == "extract":
            field = args[0].value if isinstance(args[0], E.Const) else None
            e = self.call_resolved(name, args, n.pos, qualified=schema if schema and schema != "pg_catalog" else None)
            return e
        return self.call_resolved(name, args, n.pos, qualified=schema if schema and schema != "pg_catalog" else None,
                                  special=special, node=n)

    def named_args(self, name, n, args):
        """f(a => 1, b => 2): place the arguments by parameter name, filling defaults."""
        procs = [p for p in T.PROCS.get(name, []) if p.argnames]
        for p in procs:
            names = p.argnames
            if any(an is not None and an not in names for an in n.argnames):
                continue
            slots = [None] * len(p.args)
            ok = True
            k = 0
            for an, a in zip(n.argnames, args):
                if an is None:
                    if k >= len(slots):
                        ok = False
                        break
                    slots[k] = a
                    k += 1
                else:
                    i = names.index(an)
                    if slots[i] is not None:
                        raise PgError(f'argument name "{an}" used more than once', "42601", pos=n.pos + 1)
                    slots[i] = a
            if not ok:
                continue
            last = max(i for i, x in enumerate(slots) if x is not None)
            for i in range(last + 1):
                if slots[i] is None:
                    d = p.defaults[i]
                    if d is None:
                        ok = False
                        break
                    from .parser import parse_expr
                    slots[i] = self.expr(parse_expr(d), Scope(None, self.session, self.db))
            if ok:
                return slots[:last + 1]
        raise PgError(f"function {name}({', '.join((an + ' => ' if an else '') + a.type.format() for an, a in zip(n.argnames, args))}) does not exist",
                      "42883", pos=n.pos + 1, hint=C.HINT_FUNC)

    def call_resolved(self, name, args, pos, qualified=None, display=None, special=None, node=None):
        types = [a.type for a in args]
        if name not in T.PROCS:
            raise PgError(f"function {name}({C.fmt_args(types)}) does not exist", "42883",
                          pos=pos + 1 if pos is not None else None, hint=C.HINT_FUNC)
        proc, params, ret = C.find_function(name, types, pos + 1 if pos is not None else None, kinds=("f",),
                                            qualified=qualified)
        cargs = [coerce_to(a, p, pos=pos) if p.name not in ("any",) else
                 (coerce_to(a, T.TEXT) if a.type.name == "unknown" else a) for a, p in zip(args, params)]
        if ret.name == "unknown":
            ret = T.TEXT
        impl, strict, volatile = make_func(name, [a.type for a in cargs], ret, len(cargs))
        f = E.Func(name, cargs, ret, impl, strict, volatile, display=display, pos=pos)
        if special == "extract":
            f.display = "EXTRACT"
        # fold constant calls of stable functions? (kept as calls: EXPLAIN shows them)
        return f

    def special_call(self, name, n, scope):
        args = [self.expr(a, scope) for a in n.args]
        if name == "nullif":
            if len(args) != 2:
                raise PgError("syntax error", "42601", pos=n.pos + 1)
            op = self.make_op("=", args[0], args[1], n.pos)
            a0 = op.args[0]
            t = args[0].type if args[0].type.name != "unknown" else T.TEXT
            a_out = coerce_to(args[0], t) if args[0].type.name == "unknown" else args[0]
            return E.NullIf([a_out, op.args[1]], t, op.impl)
        positions = [getattr(a, "pos", None) for a in n.args]
        ct = C.common_type([a.type for a in args], name.upper(), positions)
        if ct.name == "unknown":
            ct = T.TEXT
        args = [coerce_to(a, ct, pos=p) for a, p in zip(args, positions)]
        if name == "coalesce":
            return E.Coalesce(args, ct)
        C.find_operator("<", ct, ct, pos=n.pos + 1)
        return E.MinMax(name, args, ct, lt_impl(ct))

    def seq_call(self, name, n, scope):
        args = [self.expr(a, scope) for a in n.args]
        db = self.db
        session = self.session
        if name == "lastval":
            return E.Func("lastval", [], T.INT8, lambda a, c: c.session.lastval(), strict=False, volatile=True)
        if not args:
            raise PgError(f"function {name}() does not exist", "42883", pos=n.pos + 1, hint=C.HINT_FUNC)
        a0 = args[0]
        if a0.type.name in ("unknown", "text", "regclass", "varchar"):
            a0 = coerce_to(a0, T.REGCLASS if a0.type.name == "unknown" else a0.type)
        rest = [coerce_to(a, T.INT8 if i == 0 else T.BOOL) for i, a in enumerate(args[1:])]

        def impl(a, c, name=name):
            seqname = a[0]
            if seqname is None:
                return None
            seq = c.find_sequence(seqname)
            if name == "nextval":
                v = seq.nextval()
                c.session.last_seq[seq.name] = v
                c.session.last_seq_any = v
                return v
            if name == "currval":
                if seq.name not in c.session.last_seq:
                    raise PgError(f'currval of sequence "{seq.name}" is not yet defined in this session', "55000")
                return c.session.last_seq[seq.name]
            if name == "setval":
                v = a[1]
                called = a[2] if len(a) > 2 else True
                if v < seq.minv or v > seq.maxv:
                    raise PgError(f'setval: value {v} is out of bounds for sequence "{seq.name}" '
                                  f'({seq.minv}..{seq.maxv})', "22003")
                seq.last = v
                seq.is_called = called
                return v
        f = E.Func(name, [a0] + rest, T.INT8, impl, strict=True, volatile=True)
        f.display = name
        return f

    def srf_call(self, name, n, args, scope, srf_ok):
        types = [a.type for a in args]
        if name not in T.PROCS:
            raise PgError(f"function {name}({C.fmt_args(types)}) does not exist", "42883", pos=n.pos + 1,
                          hint=C.HINT_FUNC)
        proc, params, ret = C.find_function(name, types, n.pos + 1)
        cargs = [coerce_to(a, p, pos=n.pos) if p.name != "any" else a for a, p in zip(args, params)]
        from .srf import make_srf
        impl, cols = make_srf(name, [a.type for a in cargs], ret, proc)
        rtype = ret if not proc.out_types else (proc.out_types[0] if len(proc.out_types) == 1 else T.RECORD)
        f = E.Func(name, cargs, rtype, impl, strict=False, srf=True, pos=n.pos)
        f._srfcols = cols
        f._record = ret.name == "record" and proc.out_types is None
        f._outcols = proc.out_types is not None
        if not srf_ok:
            if not scope.allow_srf:
                raise PgError(f"set-returning functions are not allowed in {_err_kind(scope.kind)}", "0A000",
                              pos=n.pos + 1)
            if scope.in_agg:
                raise PgError("aggregate function calls cannot contain set-returning function calls", "0A000",
                              pos=n.pos + 1, hint="You might be able to move the set-returning function into a "
                                                  "LATERAL FROM item.")
            if len(cols[0]) > 1:
                f.type = T.RECORD
                names, tps = cols

                def mk_row(vals, names=names, tps=tps):
                    return vals
                f._row_names = names
            else:
                f.type = cols[1][0]
            scope.srfs.append(f)
        return f

    def agg_call(self, n: A.FuncCall, scope):
        name = n.name[-1]
        kind = scope.kind
        if kind in ("where", "on", "group", "from_func", "values" if False else "", "limit", "offset", "check",
                    "default", "index", "index_pred", "partition", "window_frame", "returning", "update", "insert",
                    "generated", "filter", "copy_where", "merge_when", "on_conflict"):
            if kind in ("partition", "window_order") and False:
                pass
            raise PgError(f"aggregate functions are not allowed in {_err_kind(kind)}", "42803", pos=n.pos + 1)
        if scope.in_agg:
            raise PgError("aggregate function calls cannot be nested", "42803", pos=n.pos + 1)
        scope.in_agg = True
        try:
            if n.within_group:
                direct = [self.expr(a, scope) for a in n.args]
                ordered = [self.expr(sb.expr, scope) for sb in n.order_by]
                types = [d.type for d in direct] + [o.type for o in ordered]
                proc, params, ret = C.find_function(name, types, n.pos + 1, kinds=("a",))
                if not proc.ordered:
                    raise PgError(f"WITHIN GROUP is required for ordered-set aggregate {name}", "42809",
                                  pos=n.pos + 1)
                nd = proc.nargs_direct
                direct = [coerce_to(d, p, pos=n.pos) for d, p in zip(direct, params[:nd])]
                ordered = [coerce_to(o, p, pos=n.pos) for o, p in zip(ordered, params[nd:])]
                order = [(o, sb.desc, sb.desc if sb.nulls is None else sb.nulls == "first")
                         for o, sb in zip(ordered, n.order_by)]
                for d in direct:
                    for v in d.walk():
                        if isinstance(v, E.Var) and v.level == 0:
                            pass
                filt = self.agg_filter(n, scope)
                a = E.Aggref(name, ordered, ret, make_agg(name, [o.type for o in ordered], ret), order=order,
                             filt=filt, direct=direct, pos=n.pos, argtypes=[o.type for o in ordered])
            else:
                if any(p.ordered for p in T.PROCS.get(name, [])) and name in ("percentile_cont", "percentile_disc",
                                                                                  "mode"):
                    raise PgError(f"WITHIN GROUP is required for ordered-set aggregate {name}", "42809",
                                  pos=n.pos + 1)
                if n.star:
                    if name != "count":
                        raise PgError(f"{name}(*) must be used to call a parameterless aggregate function", "42809",
                                      pos=n.pos + 1)
                    args = []
                else:
                    args = [self.expr(a, scope) for a in n.args]
                types = [a.type for a in args]
                if name == "count" and not n.star and len(args) == 1:
                    proc, params, ret = None, [T.ANY], T.INT8
                else:
                    proc, params, ret = C.find_function(name, types, n.pos + 1, kinds=("a",), star=n.star)
                cargs = []
                for a, p in zip(args, params):
                    if p.name == "any":
                        cargs.append(coerce_to(a, T.TEXT) if a.type.name == "unknown" else a)
                    else:
                        cargs.append(coerce_to(a, p, pos=n.pos))
                order = []
                for sb in n.order_by:
                    oe = self.expr(sb.expr, scope)
                    if oe.type.name == "unknown":
                        oe = coerce_to(oe, T.TEXT)
                    order.append((oe, sb.desc, sb.desc if sb.nulls is None else sb.nulls == "first"))
                if n.distinct and order:
                    akeys = {a.key() for a in cargs}
                    for (oe, _, _), sb in zip(order, n.order_by):
                        if oe.key() not in akeys:
                            raise PgError("in an aggregate with DISTINCT, ORDER BY expressions must appear in "
                                          "argument list", "42P10", pos=sb.expr.pos + 1)
                filt = self.agg_filter(n, scope)
                if ret.name == "unknown":
                    ret = T.TEXT
                a = E.Aggref(name, cargs, ret, make_agg(name, [c.type for c in cargs], ret, star=n.star),
                             distinct=n.distinct, order=order, filt=filt, star=n.star, pos=n.pos,
                             argtypes=[c.type for c in cargs])
        finally:
            scope.in_agg = False
        for node in a.walk():
            if isinstance(node, E.WindowFunc):
                raise PgError("aggregate function calls cannot contain window function calls", "42803",
                              pos=(node.pos or n.pos) + 1)
        k = a.key()
        for existing in scope.aggs:
            if existing.key() == k:
                return existing
        scope.aggs.append(a)
        return a

    def agg_filter(self, n, scope):
        if n.filter is None:
            return None
        f = self.expr(n.filter, scope)
        return self.bool_expr(f, "FILTER", n.filter)

    def window_call(self, n: A.FuncCall, scope):
        name = n.name[-1]
        if scope.kind not in ("select", "order"):
            raise PgError(f"window functions are not allowed in {_err_kind(scope.kind)}", "42P20", pos=n.pos + 1)
        if scope.in_agg:
            raise PgError("aggregate function calls cannot contain window function calls", "42803", pos=n.pos + 1)
        if scope.in_window:
            raise PgError("window function calls cannot be nested", "42P20", pos=n.pos + 1)
        if n.distinct:
            raise PgError("DISTINCT is not implemented for window functions", "0A000", pos=n.pos + 1)
        if n.order_by:
            raise PgError("aggregate ORDER BY is not implemented for window functions", "0A000", pos=n.pos + 1)
        if n.within_group:
            raise PgError(f"OVER is not supported for ordered-set aggregate {name}", "0A000", pos=n.pos + 1)
        scope.in_window = True
        try:
            args = [self.expr(a, scope) for a in n.args]
            types = [a.type for a in args]
            procs = T.PROCS.get(name, [])
            if not procs:
                raise PgError(f"function {name}({C.fmt_args(types)}) does not exist", "42883", pos=n.pos + 1,
                              hint=C.HINT_FUNC)
            if name == "count" and not n.star and len(args) == 1:
                proc, params, ret = None, [T.ANY], T.INT8
            else:
                proc, params, ret = C.find_function(name, types, n.pos + 1, kinds=("a", "w"), star=n.star)
            if proc is not None and proc.kind not in ("a", "w"):
                raise PgError(f"OVER specified, but {name} is not a window function nor an aggregate function",
                              "42809", pos=n.pos + 1)
            if proc is not None and proc.ordered:
                raise PgError(f"OVER is not supported for ordered-set aggregate {name}", "0A000", pos=n.pos + 1)
            cargs = []
            for a, p in zip(args, params):
                if p.name == "any":
                    cargs.append(coerce_to(a, T.TEXT) if a.type.name == "unknown" else a)
                else:
                    cargs.append(coerce_to(a, p, pos=n.pos))
            if ret.name == "unknown":
                ret = T.TEXT
            filt = self.agg_filter(n, scope) if n.filter is not None else None
            if filt is not None and name in WINDOW_ONLY:
                raise PgError("FILTER is not implemented for non-aggregate window functions", "0A000",
                              pos=n.pos + 1)
            spec = self.window_spec(n.over, scope)
        finally:
            scope.in_window = False
        agg = None
        if name not in WINDOW_ONLY:
            agg = make_agg(name, [c.type for c in cargs], ret, star=n.star)
        w = E.WindowFunc(name, cargs, ret, spec, agg=agg, filt=filt, pos=n.pos, star=n.star)
        scope.wins.append(w)
        return w


# ---------------------------------------------------------------- tree utilities

class _FDVar(E.Var):
    """A column allowed in a grouped query because its table's primary key is grouped: its value is the same for
    the whole group, so the group's first row (kept in the group row's last slot) supplies it."""
    __slots__ = ("rep",)

    def __init__(self, v, depth=0, rep=None):
        super().__init__(depth, v.index, v.type, v.rte, v.col)
        self.orig = None
        self.rep = rep

    def key(self):
        return ("fdvar", self.level, self.index)

    def compile(self):
        i = self.index
        rep = self.rep
        if self.level == 0:
            return lambda r, c: r[rep][i]
        k = -self.level
        return lambda r, c: c.outer[k][rep][i]




def map_expr(e, fn, depth=0):
    """Replace nodes bottom-up-ish: fn(node, depth) -> replacement or None (keep and descend). Descends into
    subplans with depth + 1 (so fn can remap outer references)."""
    if e is None:
        return None
    r = fn(e, depth)
    if r is not None:
        return r
    if isinstance(e, (E.Func, E.Op, E.Coalesce, E.MinMax, E.NullIf, E.Distinct, E.ScalarArrayOp, E.InList,
                      E.Bool)):
        e.args = [map_expr(a, fn, depth) for a in e.args]
    elif isinstance(e, E.Case):
        e.arg = map_expr(e.arg, fn, depth)
        e.whens = [(map_expr(c, fn, depth), map_expr(x, fn, depth)) for c, x in e.whens]
        e.default = map_expr(e.default, fn, depth)
    elif isinstance(e, (E.NullTest, E.BoolTest, E.Cast, E.FieldSelect)):
        e.arg = map_expr(e.arg, fn, depth)
    elif isinstance(e, (E.ArrayExpr, E.RowExpr)):
        e.items = [map_expr(a, fn, depth) for a in e.items]
    elif isinstance(e, E.Subscript):
        e.arg = map_expr(e.arg, fn, depth)
        e.subs = [(map_expr(lo, fn, depth), map_expr(hi, fn, depth), s) for lo, hi, s in e.subs]
    elif isinstance(e, E.Aggref):
        e.args = [map_expr(a, fn, depth) for a in e.args]
        e.direct = [map_expr(a, fn, depth) for a in e.direct]
        e.order = [(map_expr(o[0], fn, depth), o[1], o[2]) for o in e.order]
        e.filter = map_expr(e.filter, fn, depth)
    elif isinstance(e, E.WindowFunc):
        pass
    elif isinstance(e, E.GroupingFunc):
        pass
    elif isinstance(e, E.SubPlan):
        if e.test is not None:
            e.test = map_expr(e.test, fn, depth)
        map_query(e.query, fn, depth + 1)
    return e


def map_query(q: Query, fn, depth):
    """Apply map_expr to every expression of a query (for remapping outer references)."""
    if q is None:
        return
    if q.kind == "setop" and q.setop is not None:
        map_query(q.setop[2], fn, depth)
        map_query(q.setop[3], fn, depth)
        return
    if q.kind == "values" and q.values is not None:
        q.values = [[map_expr(x, fn, depth) for x in row] for row in q.values]
        return
    q.targets = [map_expr(t, fn, depth) for t in q.targets]
    q.where = map_expr(q.where, fn, depth)
    q.having = map_expr(q.having, fn, depth)
    q.group_keys = [map_expr(k, fn, depth) for k in q.group_keys]
    for a in q.aggs:
        map_expr(a, fn, depth)
    for w in q.windows:
        w.partition = [map_expr(p, fn, depth) for p in w.partition]
        w.order = [(map_expr(o[0], fn, depth), o[1], o[2]) for o in w.order]
        for f in w.funcs:
            f.args = [map_expr(a, fn, depth) for a in f.args]
    for f in q.srfs:
        f.args = [map_expr(a, fn, depth) for a in f.args]
    _map_from(q.from_, fn, depth)
    for cte in q.ctes:
        if isinstance(cte.query, Query):
            map_query(cte.query, fn, depth)


def _map_from(node, fn, depth):
    if node is None:
        return
    if isinstance(node, ListNode):
        for it in node.items:
            _map_from(it, fn, depth)
    elif isinstance(node, JoinNode):
        _map_from(node.left, fn, depth)
        _map_from(node.right, fn, depth)
        node.quals = map_expr(node.quals, fn, depth)
    elif isinstance(node, ScanFunc):
        for f in node.funcs:
            f.args = [map_expr(a, fn, depth) for a in f.args]
    elif isinstance(node, ScanSub):
        map_query(node.query, fn, depth + 1)


def _walk_all(exprs):
    for e in exprs:
        if e is not None:
            yield from e.walk()


def _all_rtes(scope):
    seen = []
    for it in scope.ns:
        if it.rte not in seen:
            seen.append(it.rte)
    return seen


def _rtes_of(node):
    out = []
    if isinstance(node, JoinNode):
        out += _rtes_of(node.left) + _rtes_of(node.right) + [node.rte]
    elif isinstance(node, ListNode):
        for it in node.items:
            out += _rtes_of(it)
    elif node is not None:
        out.append(node.rte)
    return out


def _visible_cols(rte):
    return [(n, t, s) for n, t, s in zip(rte.colnames, rte.coltypes, rte.slots)]


def _same_window(a: WindowClause, b: WindowClause):
    if len(a.partition) != len(b.partition) or len(a.order) != len(b.order):
        return False
    if any(x.key() != y.key() for x, y in zip(a.partition, b.partition)):
        return False
    if any(x[0].key() != y[0].key() or x[1] != y[1] or x[2] != y[2] for x, y in zip(a.order, b.order)):
        return False
    return (a.mode, a.start, a.end, a.start_off, a.end_off, a.exclude) == \
        (b.mode, b.start, b.end, b.start_off, b.end_off, b.exclude)


def _sort_type(t):
    return t


def _retype(q: Query, i, ct):
    """Convert column i of a set-operation branch to the common type."""
    if q.kind == "setop":
        _, _, l, r = q.setop
        _retype(l, i, ct)
        _retype(r, i, ct)
        q.types[i] = ct
        return
    if q.kind == "values":
        for row in q.values:
            row[i] = coerce_to(row[i], ct)
        q.types[i] = ct
        return
    q.targets[i] = coerce_to(q.targets[i], ct)
    q.types[i] = ct


def _query_pos(node):
    while isinstance(node, A.SetOp):
        node = node.left
    return getattr(node, "pos", -1)


def _first_target_pos(node, i):
    while isinstance(node, A.SetOp):
        node = node.left
    if isinstance(node, A.Select) and i < len(node.targets):
        return node.targets[i].pos + 1
    return None


def _literal_pos(n):
    return n.pos + 1


def _relabel(a, t, n):
    if t.mod is not None and a.type.mod != t.mod:
        return _typmod_cast(a, t, True, n.pos)
    c = E.Cast(a, t, lambda v, c: v, True, pos=n.pos)
    return c


def _src_of(rte, i):
    """The base relation and column a column of a join RTE comes from (EXPLAIN qualifies with it)."""
    src = getattr(rte, "src", None)
    if src is not None and i < len(src):
        return src[i]
    return rte.name, rte.colnames[i]


def _levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

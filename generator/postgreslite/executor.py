"""Running analyzed queries: FROM items and joins (hash joins on equality conditions, filters pushed down to the
FROM item they mention, hashed lookups for correlated subqueries), grouping sets, window functions with frames,
set-returning functions in SELECT, DISTINCT [ON], ORDER BY with PostgreSQL's NULL placement, LIMIT / OFFSET /
WITH TIES, set operations, VALUES and CTEs (recursive ones too)."""

from __future__ import annotations

import math
import random

from . import expr as E
from . import types as T
from .analyze import (Query, ListNode, JoinNode, ScanRel, ScanSub, ScanCte, ScanFunc, ScanValues)
from .aggs import frame_bounds
from .catalog import Sequence, Table, View
from .errors import PgError

RECURSION_ROWS = 200_000          # a recursive CTE that grows past this is stopped (like a statement timeout)
MAX_ROWS = 2_000_000


class Ctx:
    """One statement's execution state."""

    def __init__(self, session, db, params=()):
        self.session = session
        self.db = db
        self.outer: list = []
        self.case_val: list = []
        self.params = list(params)
        self.cte_cache: dict = {}
        self.working: dict = {}
        self.subplan_cache: dict = {}
        self.index_cache: dict = {}
        self.txn_now = session.txn_now if session is not None else T.EPOCH_TZ
        self.stmt_now = session.stmt_now if session is not None else self.txn_now
        self.rng = session.rng if session is not None else random.Random(0)
        self.steps = 0
        self.modified = set()

    def clock_now(self):
        if self.session is not None:
            return self.session.clock_now()
        return self.stmt_now

    def find_sequence(self, name):
        name = str(name)
        schema = None
        if "." in name:
            schema, name = name.split(".", 1)
        name = name.strip('"')
        rel = self.db.find_rel(name.lower() if not name.startswith('"') else name, schema, self.session)
        if not isinstance(rel, Sequence):
            if rel is None:
                raise PgError(f'relation "{name}" does not exist', "42P01")
            raise PgError(f'"{name}" is not a sequence', "42809")
        return rel


# ---------------------------------------------------------------- compiled-closure caches

def _compiled(q, attr, build):
    v = q.__dict__.get(attr)
    if v is None:
        v = build()
        q.__dict__[attr] = v
    return v


# ---------------------------------------------------------------- queries

def run_query(q: Query, c: Ctx) -> list:
    """The rows of a query (visible columns only)."""
    if q.kind == "setop":
        return run_setop(q, c)
    if q.kind == "values":
        return run_values(q, c)
    rows = run_from_where(q, c)
    if q.grouped:
        rows = run_group(q, rows, c)
    if q.windows:
        rows = run_windows(q, rows, c)
    if q.srfs:
        rows = expand_srfs(q, rows, c)
    fns = _compiled(q, "_tfns", lambda: [E.compile_at(t, 0) for t in q.targets])
    out = [tuple([f(r, c) for f in fns]) for r in rows]
    return finish(q, out, c)


def finish(q: Query, out: list, c: Ctx, nvis=None):
    """DISTINCT, ORDER BY, OFFSET / LIMIT on output rows (which may carry ORDER BY helper columns)."""
    nvis = q.ntargets if nvis is None else nvis
    if q.distinct is True:
        seen = set()
        uniq = []
        for row in out:
            k = tuple(T.hash_key(v) for v in row[:nvis])
            if k not in seen:
                seen.add(k)
                uniq.append(row)
        out = uniq
    if q.order:
        out = sort_rows(out, q.order)
    if isinstance(q.distinct, list):
        seen = set()
        uniq = []
        for row in out:
            k = tuple(T.hash_key(row[i]) for i in q.distinct)
            if k not in seen:
                seen.add(k)
                uniq.append(row)
        out = uniq
    if q.offset is not None or q.limit is not None:
        off = _eval_limit(q.offset, c, "OFFSET") if q.offset is not None else 0
        lim = _eval_limit(q.limit, c, "LIMIT") if q.limit is not None else None
        if off:
            out = out[off:]
        if lim is not None:
            if q.with_ties and lim < len(out) and lim > 0:
                last = _order_key(out[lim - 1], q.order)
                end = lim
                while end < len(out) and _order_key(out[end], q.order) == last:
                    end += 1
                out = out[:end]
            else:
                out = out[:lim]
    if len(out) and len(out[0]) != nvis:
        out = [row[:nvis] for row in out]
    return out


def _order_key(row, order):
    return tuple(T.hash_key(row[i]) for i, *_ in order)


def _eval_limit(e, c, what):
    v = e.compile()(None, c)
    if v is None:
        return None if what == "LIMIT" else 0
    if v < 0:
        raise PgError(f"{what} must not be negative", "2201W" if what == "LIMIT" else "2201X")
    return v


class _Rev:
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k


def sort_rows(rows, order):
    """Stable multi-key sort; ASC puts NULLs last, DESC first, unless NULLS FIRST / LAST says otherwise."""
    rows = list(rows)
    for idx, desc, nulls_first, typ in reversed(order):
        # with reverse=True the NULL marker must be "big" to come first
        if desc:
            nk = (1,) if nulls_first else (-1,)
        else:
            nk = (-1,) if nulls_first else (1,)

        def key(row, idx=idx, nk=nk, typ=typ):
            v = row[idx]
            if v is None:
                return nk
            return (0, T.sort_key(v, typ))
        rows.sort(key=key, reverse=desc)
    return rows


# ---------------------------------------------------------------- FROM + WHERE

class Conj:
    __slots__ = ("e", "slots", "fn_cache", "volatile")

    def __init__(self, e):
        self.e = e
        self.slots = {v.index for v in E.vars_of(e, 0)}
        self.fn_cache = {}
        self.volatile = E.has_volatile(e)

    def fn(self, base):
        f = self.fn_cache.get(base)
        if f is None:
            f = E.compile_at(self.e, base)
            self.fn_cache[base] = f
        return f


def run_from_where(q: Query, c: Ctx):
    conj = _compiled(q, "_conj", lambda: [Conj(x) for x in E.conjuncts(q.where)])
    pending = list(conj)
    if q.from_ is None:
        rows = [()]
    else:
        rows = run_node(q.from_, c, pending)
    for cj in pending:
        f = cj.fn(0)
        rows = [r for r in rows if f(r, c)]
    return rows


def _slots(node):
    return set(range(node.base, node.base + node.width))


def _take(pending, pred):
    taken = [cj for cj in pending if pred(cj)]
    for cj in taken:
        pending.remove(cj)
    return taken


def run_node(node, c, pending, left_row=None, left_base=0):
    """Rows of a FROM item, relative to node.base. Conjuncts that only mention this item are taken from
    `pending` and applied here."""
    if isinstance(node, ListNode):
        return run_list(node, c, pending)
    if isinstance(node, JoinNode):
        rows = run_join(node, c)
        mine = _take(pending, lambda cj: cj.slots and cj.slots <= _slots(node) and not cj.volatile)
        for cj in mine:
            f = cj.fn(node.base)
            rows = [r for r in rows if f(r, c)]
        return rows
    mine = _take(pending, lambda cj: cj.slots and cj.slots <= _slots(node) and not cj.volatile)
    if isinstance(node, ScanRel):
        return scan_rel(node, c, mine)
    rows = scan_other(node, c, left_row, left_base)
    for cj in mine:
        f = cj.fn(node.base)
        rows = [r for r in rows if f(r, c)]
    return rows


def _lateral_dep(node):
    """Does this FROM item read columns of items to its left?"""
    v = node.__dict__.get("_latdep")
    if v is not None:
        return v
    dep = False
    if isinstance(node, ScanFunc):
        for f in node.funcs:
            for a in f.args:
                if E.vars_of(a, 0):
                    dep = True
    elif isinstance(node, ScanSub) and node.lateral:
        dep = bool(getattr(node, "uses_outer", True))
    elif isinstance(node, JoinNode):
        dep = _lateral_dep(node.left)
    node.__dict__["_latdep"] = dep
    return dep


def run_list(node: ListNode, c, pending):
    acc = None
    acc_slots = set()
    base = node.base
    for item in node.items:
        islots = _slots(item)
        if acc is None:
            acc = run_node(item, c, pending)
            acc_slots = islots
            continue
        if _lateral_dep(item):
            out = []
            joinable = _take(pending, lambda cj: cj.slots and cj.slots <= (acc_slots | islots) and
                             cj.slots & islots and not cj.volatile)
            fns = [cj.fn(base) for cj in joinable]
            for lr in acc:
                for rr in run_node(item, c, [], left_row=lr, left_base=base):
                    row = lr + rr
                    if all(f(row, c) for f in fns):
                        out.append(row)
            acc = out
        else:
            right = run_node(item, c, pending)
            joinable = _take(pending, lambda cj: cj.slots and cj.slots <= (acc_slots | islots) and
                             cj.slots & islots and cj.slots & acc_slots and not cj.volatile)
            acc = hash_or_nested(acc, right, joinable, base, item.base, c, "inner", len(right[0]) if right else
                                 item.width)
        acc_slots |= islots
        _guard(len(acc))
    return acc


def _guard(n):
    if n > MAX_ROWS:
        raise PgError("postgres-lite stopped this query: it produced more than 2,000,000 rows", "54000",
                      hint="Check the join conditions (a missing condition makes every pair of rows).")


def _equi(cj, lslots, rslots):
    """An equality between an expression of the left side and one of the right side -> (left e, right e)."""
    e = cj.e
    if not isinstance(e, E.Op) or e.op != "=" or len(e.args) != 2:
        return None
    a, b = e.args
    sa = {v.index for v in E.vars_of(a, 0)}
    sb = {v.index for v in E.vars_of(b, 0)}
    if not sa or not sb:
        return None
    if not _hash_compatible(a.type, b.type):
        return None
    if sa <= lslots and sb <= rslots:
        return a, b
    if sb <= lslots and sa <= rslots:
        return b, a
    return None


_NUMERIC_FAMILY = {"int2", "int4", "int8", "numeric", "float4", "float8"}


def _hash_compatible(t1, t2):
    if t1.name in _NUMERIC_FAMILY and t2.name in _NUMERIC_FAMILY:
        return True
    if t1.name in ("text", "varchar", "name") and t2.name in ("text", "varchar", "name"):
        return True
    return t1.name == t2.name and t1.name not in ("float4", "float8", "json") and not t1.is_array


def hash_or_nested(left, right, conjs, lbase, rbase, c, kind, rwidth, merge=None, lwidth=None):
    """Join two row lists. conjs: the join's conditions (on the combined row, which starts at lbase)."""
    lslots = set(range(lbase, rbase))
    rslots = set(range(rbase, rbase + rwidth))
    keys = []
    rest = []
    for cj in conjs:
        eq = _equi(cj, lslots, rslots)
        if eq is not None:
            keys.append(eq)
        else:
            rest.append(cj)
    restf = [cj.fn(lbase) for cj in rest]
    nulls_r = (None,) * rwidth
    lw = lwidth if lwidth is not None else (rbase - lbase)
    nulls_l = (None,) * lw
    out = []
    right_matched = [False] * len(right) if kind in ("right", "full") else None

    def emit(lr, rr):
        row = lr + rr
        if merge:
            row = row + tuple(_coalesce(row[ls - lbase], row[rs - lbase]) for ls, rs in merge)
        out.append(row)
    if keys:
        lk = [E.compile_at(a, lbase) for a, _ in keys]
        rk = [E.compile_at(b, rbase) for _, b in keys]
        table = {}
        for j, rr in enumerate(right):
            k = []
            ok = True
            for f in rk:
                v = f(rr, c)
                if v is None:
                    ok = False
                    break
                k.append(T.hash_key(v))
            if ok:
                table.setdefault(tuple(k), []).append(j)
        for lr in left:
            k = []
            ok = True
            for f in lk:
                v = f(lr, c)
                if v is None:
                    ok = False
                    break
                k.append(T.hash_key(v))
            matched = False
            if ok:
                for j in table.get(tuple(k), ()):
                    rr = right[j]
                    row = lr + rr
                    if all(f(row, c) for f in restf):
                        matched = True
                        if right_matched is not None:
                            right_matched[j] = True
                        emit(lr, rr)
            if not matched and kind in ("left", "full"):
                emit(lr, nulls_r)
            if len(out) > MAX_ROWS:
                _guard(len(out))
    else:
        for lr in left:
            matched = False
            for j, rr in enumerate(right):
                row = lr + rr
                if all(f(row, c) for f in restf):
                    matched = True
                    if right_matched is not None:
                        right_matched[j] = True
                    emit(lr, rr)
            if not matched and kind in ("left", "full"):
                emit(lr, nulls_r)
            if len(out) > MAX_ROWS:
                _guard(len(out))
    if right_matched is not None:
        for j, m in enumerate(right_matched):
            if not m:
                emit(nulls_l, right[j])
    return out


def _coalesce(a, b):
    return a if a is not None else b


def run_join(node: JoinNode, c):
    left = run_node(node.left, c, [])
    lbase = node.base
    rbase = node.right.base
    conjs = [Conj(x) for x in E.conjuncts(node.quals)] if node.quals is not None else []
    if not hasattr(node, "_conjs"):
        node._conjs = conjs
    conjs = node._conjs
    kind = "inner" if node.kind == "cross" else node.kind
    lwidth = node.right.base - node.base
    rwidth = node.right.width
    if _lateral_dep(node.right) and not isinstance(node.right, JoinNode):
        restf = [cj.fn(lbase) for cj in conjs]
        out = []
        nulls_r = (None,) * rwidth
        for lr in left:
            matched = False
            for rr in run_node(node.right, c, [], left_row=lr, left_base=lbase):
                row = lr + rr
                if all(f(row, c) for f in restf):
                    matched = True
                    out.append(row)
            if not matched and kind in ("left", "full"):
                out.append(lr + nulls_r)
        if kind in ("right", "full"):
            raise PgError("the combining JOIN type must be INNER or LEFT for a LATERAL reference", "42601")
        return out
    right = run_node(node.right, c, [])
    return hash_or_nested(left, right, list(conjs), lbase, rbase, c, kind, rwidth, node.merge, lwidth)


# ---------------------------------------------------------------- scans

def table_rows(rel, c):
    if isinstance(rel, Sequence):
        return [(rel.last, 0 if rel.is_called else 0, rel.is_called)]
    return rel.rows


def scan_rel(node: ScanRel, c, conjs):
    rel = node.rel
    rows = table_rows(rel, c)
    base = node.base
    sample = getattr(node, "sample", None)
    if sample is not None:
        method, args, seed = sample
        pct = args[0].compile()(None, c)
        if pct is None or pct < 0 or pct > 100:
            raise PgError("sample percentage must be between 0 and 100", "2202H")
        rng = random.Random(seed.compile()(None, c)) if seed is not None else c.rng
        rows = [r for r in rows if rng.random() * 100 < pct]
    if not conjs:
        return rows
    # an equality with a value known before the scan (a constant, a parameter, an outer row's column):
    # look the rows up in a hash index on that column instead of reading them all
    lookup = None
    rest = []
    for cj in conjs:
        if lookup is None:
            hit = _index_cond(cj, node)
            if hit is not None:
                lookup = hit
                continue
        rest.append(cj)
    if lookup is not None:
        col, valf = lookup
        v = valf(None, c)
        if v is None:
            return []
        idx = _index(c, rows, col)
        rows = idx.get(T.hash_key(v), ())
    fns = [cj.fn(base) for cj in rest]
    if not fns:
        return list(rows)
    if len(fns) == 1:
        f = fns[0]
        return [r for r in rows if f(r, c)]
    return [r for r in rows if all(f(r, c) for f in fns)]


def _index_cond(cj, node):
    e = cj.e
    if not isinstance(e, E.Op) or e.op != "=" or len(e.args) != 2:
        return None
    for a, b in (e.args, e.args[::-1]):
        a0 = a
        while isinstance(a0, E.Cast) and a0.type.name in _NUMERIC_FAMILY and a0.arg.type.name in _NUMERIC_FAMILY:
            a0 = a0.arg
        if isinstance(a0, E.Var) and a0.level == 0 and node.base <= a0.index < node.base + node.width:
            if E.vars_of(b, 0):
                continue
            if not _hash_compatible(a.type, b.type) or E.has_volatile(b):
                continue
            if any(isinstance(n, E.SubPlan) for n in b.walk()):
                continue
            col = a0.index - node.base
            return col, E.compile_at(b, 0)
    return None


def _index(c, rows, col):
    key = (id(rows), len(rows), col)
    hit = c.index_cache.get(key)
    if hit is not None and hit[0] is rows:
        return hit[1]
    d = {}
    for r in rows:
        v = r[col]
        if v is not None:
            d.setdefault(T.hash_key(v), []).append(r)
    c.index_cache[key] = (rows, d)
    return d


def scan_other(node, c, left_row=None, left_base=0):
    if isinstance(node, ScanSub):
        if node.lateral and left_row is not None:
            padded = (None,) * left_base + left_row
            c.outer.append(padded)
            try:
                return run_query(node.query, c)
            finally:
                c.outer.pop()
        if not node.lateral and not node.query.__dict__.get("_maybe_correlated", True):
            key = ("sub", id(node))
        c.outer.append(None)
        try:
            return run_query(node.query, c)
        finally:
            c.outer.pop()
    if isinstance(node, ScanCte):
        return cte_rows(node.cte, c)
    if isinstance(node, ScanFunc):
        return scan_func(node, c, left_row, left_base)
    if isinstance(node, ScanValues):
        return run_values(node.query, c)
    raise AssertionError(type(node))


def cte_rows(cte, c):
    w = c.working.get(cte.id)
    if w is not None:
        return w
    hit = c.cte_cache.get(cte.id)
    if hit is not None:
        return hit
    if cte.modifying:
        from .dml import run_modify
        rows = run_modify(cte.query, c)
    else:
        c.outer.append(None)
        try:
            rows = run_query(cte.query, c)
        finally:
            c.outer.pop()
    c.cte_cache[cte.id] = rows
    return rows


def scan_func(node: ScanFunc, c, left_row, left_base):
    r = left_row
    base = left_base if left_row is not None else 0
    results = []
    for f in node.funcs:
        argf = f.__dict__.get("_argf")
        if argf is None or f.__dict__.get("_argbase") != base:
            argf = [E.compile_at(a, base) for a in f.args]
            f._argf = argf
            f._argbase = base
        args = [g(r, c) for g in argf]
        if getattr(f, "srf", False):
            vals = f.impl(args, c)
            ncols = len(f._cols[0])
            coldefs = getattr(f, "_coldefs", None)
            if coldefs is not None:
                vals = [_record_cols(v, coldefs) for v in vals]
                ncols = len(coldefs)
            results.append((vals, ncols))
        else:
            v = f.impl(args, c)
            results.append(([(v,)], 1))
    if len(results) == 1:
        vals, ncols = results[0]
        rows = [tuple(v) for v in vals]
    else:
        n = max(len(v) for v, _ in results)
        rows = []
        for k in range(n):
            row = ()
            for vals, ncols in results:
                row += tuple(vals[k]) if k < len(vals) else (None,) * ncols
            rows.append(row)
    if node.ordinality:
        rows = [row + (i,) for i, row in enumerate(rows, 1)]
    _guard(len(rows))
    return rows


def _record_cols(v, coldefs):
    """jsonb_to_record ... AS x(a int, b text): the record's fields by name, cast to the declared types."""
    rec = v[0] if isinstance(v, tuple) and len(v) == 1 else v
    out = []
    for name, t in coldefs:
        x = rec.get(name) if isinstance(rec, dict) else None
        out.append(_json_to_sql(x, t))
    return tuple(out)


def _json_to_sql(x, t):
    if x is None or x is T.JNULL:
        return None
    if t.name in ("json", "jsonb"):
        return T.Json(x, t.name == "jsonb")
    if isinstance(x, str):
        return T.input_value(x, t.base, t.mod)
    return T.input_value(T.jsonb_text(x), t.base, t.mod)


def run_values(q: Query, c):
    fns = _compiled(q, "_vfns", lambda: [[E.compile_at(x, 0) for x in row] for row in q.values])
    out = [tuple(f(None, c) for f in row) for row in fns]
    return finish(q, out, c)


# ---------------------------------------------------------------- set operations

def run_setop(q: Query, c):
    op, all_, left, right = q.setop
    if q.recursive_ref is not None:
        rows = run_recursive(q, c)
        return finish(q, rows, c)
    c.outer.append(None)
    try:
        lrows = run_query(left, c)
        rrows = run_query(right, c)
    finally:
        c.outer.pop()
    if op == "union":
        if all_:
            out = lrows + rrows
        else:
            seen = set()
            out = []
            for row in lrows + rrows:
                k = tuple(T.hash_key(v) for v in row)
                if k not in seen:
                    seen.add(k)
                    out.append(row)
    else:
        counts = {}
        for row in rrows:
            k = tuple(T.hash_key(v) for v in row)
            counts[k] = counts.get(k, 0) + 1
        out = []
        seen = set()
        for row in lrows:
            k = tuple(T.hash_key(v) for v in row)
            n = counts.get(k, 0)
            if op == "intersect":
                if all_:
                    if n > 0:
                        out.append(row)
                        counts[k] = n - 1
                elif n > 0 and k not in seen:
                    seen.add(k)
                    out.append(row)
            else:
                if all_:
                    if n > 0:
                        counts[k] = n - 1
                    else:
                        out.append(row)
                elif n == 0 and k not in seen:
                    seen.add(k)
                    out.append(row)
    return finish(q, out, c)


def run_recursive(q: Query, c):
    _, all_, base, rec = q.setop
    cte = q.recursive_ref
    c.outer.append(None)
    try:
        return _run_recursive(q, c, all_, base, rec, cte)
    finally:
        c.outer.pop()


def _run_recursive(q, c, all_, base, rec, cte):
    rows = run_query(base, c)
    seen = set()
    if not all_:
        uniq = []
        for row in rows:
            k = tuple(T.hash_key(v) for v in row)
            if k not in seen:
                seen.add(k)
                uniq.append(row)
        rows = uniq
    result = list(rows)
    working = rows
    saved = c.working.get(cte.id)
    try:
        while working:
            c.working[cte.id] = working
            new = run_query(rec, c)
            if not all_:
                fresh = []
                for row in new:
                    k = tuple(T.hash_key(v) for v in row)
                    if k not in seen:
                        seen.add(k)
                        fresh.append(row)
                new = fresh
            result += new
            working = new
            if len(result) > RECURSION_ROWS:
                raise PgError("canceling statement due to statement timeout", "57014",
                              context=f'postgres-lite stopped the recursive query "{cte.name}" after '
                                      f'{RECURSION_ROWS:,} rows: does its recursive term ever stop producing rows?')
    finally:
        if saved is None:
            c.working.pop(cte.id, None)
        else:
            c.working[cte.id] = saved
    return result


# ---------------------------------------------------------------- grouping

def run_group(q: Query, rows, c):
    keyfns = _compiled(q, "_kfns", lambda: [E.compile_at(k, 0) for k in q.group_keys])
    nk = len(keyfns)
    aggs = q.aggs
    plans = _compiled(q, "_aplans", lambda: [_agg_plan(a) for a in aggs])
    sets = q.grouping_sets if q.grouping_sets is not None else [list(range(nk))]
    out = []
    havf = _compiled(q, "_havf", lambda: E.compile_at(q.having, 0) if q.having is not None else False)
    # evaluate the keys once per row
    keyvals = [[f(r, c) for f in keyfns] for r in rows] if nk else [[] for _ in rows]
    for gset in sets:
        present = frozenset(gset)
        groups = {}
        for r, kv in zip(rows, keyvals):
            hk = tuple(T.hash_key(kv[i]) for i in gset)
            g = groups.get(hk)
            if g is None:
                g = [[kv[i] if i in present else None for i in range(nk)],
                     [p.new() for p in plans], r]
                groups[hk] = g
            for p, st in zip(plans, g[1]):
                p.add(st, r, c)
        if not groups and not gset:
            groups[()] = [[None] * nk, [p.new() for p in plans], None]
        for g in groups.values():
            vals = [p.result(st, c, g[2]) for p, st in zip(plans, g[1])]
            row = tuple(g[0]) + tuple(vals) + (present, g[2] if g[2] is not None else ())
            if havf:
                if not havf(row, c):
                    continue
            out.append(row)
    return out


class _AggPlan:
    def __init__(self, a: E.Aggref):
        self.a = a
        self.argf = [E.compile_at(x, 0) for x in a.args]
        self.filt = E.compile_at(a.filter, 0) if a.filter is not None else None
        self.orderf = [(E.compile_at(o[0], 0), o[1], o[2], o[0].type) for o in a.order]
        self.direct = [E.compile_at(d, 0) for d in a.direct]
        self.collect = bool(a.order) or a.distinct
        self.factory = a.agg

    def new(self):
        return [] if self.collect else self.factory()

    def add(self, st, r, c):
        if self.filt is not None and not self.filt(r, c):
            return
        vals = [f(r, c) for f in self.argf]
        if self.collect:
            st.append((vals, [f(r, c) for f, *_ in self.orderf]))
        else:
            st.add(vals)

    def result(self, st, c, rep):
        if not self.collect:
            return st.result()
        items = st
        if self.a.distinct:
            seen = set()
            uniq = []
            for vals, keys in items:
                if self.a.name not in ("count",) and any(v is None for v in vals) and self.a.name != "array_agg":
                    pass
                k = tuple(T.hash_key(v) for v in vals)
                if k not in seen:
                    seen.add(k)
                    uniq.append((vals, keys))
            items = uniq
        if self.orderf:
            order = [(i, desc, nf, t) for i, (_, desc, nf, t) in enumerate(self.orderf)]
            keyed = [tuple(keys) + (vals,) for vals, keys in items]
            keyed = sort_rows(keyed, order)
            items = [(k[-1], None) for k in keyed]
        direct = [f(rep, c) for f in self.direct] if self.direct else None
        state = self.factory(direct)
        for vals, _ in items:
            state.add(vals)
        return state.result()


def _agg_plan(a):
    return _AggPlan(a)


# ---------------------------------------------------------------- windows

def run_windows(q: Query, rows, c):
    n = len(rows)
    extra = [[None] * sum(len(w.funcs) for w in q.windows) for _ in range(n)]
    slot0 = min(f.slot for w in q.windows for f in w.funcs)
    order = list(range(n))
    for w in q.windows:
        pf = _compiled(w, "_pf", lambda w=w: [E.compile_at(p, 0) for p in w.partition])
        of = _compiled(w, "_of", lambda w=w: [(E.compile_at(o[0], 0), o[1], o[2], o[0].type) for o in w.order])
        parts = {}
        for i in range(n):
            r = rows[i]
            k = tuple(T.hash_key(f(r, c)) for f in pf)
            parts.setdefault(k, []).append(i)
        new_order = []
        for idxs in parts.values():
            if of:
                keyed = [tuple(f(rows[i], c) for f, *_ in of) + (i,) for i in idxs]
                keyed = sort_rows(keyed, [(j, d, nf, t) for j, (_, d, nf, t) in enumerate(of)])
                idxs = [k[-1] for k in keyed]
                okeys = [tuple(T.hash_key(x) for x in k[:-1]) for k in keyed]
                ovals = [k[0] for k in keyed]
            else:
                okeys = [()] * len(idxs)
                ovals = [None] * len(idxs)
            new_order += idxs
            _window_partition(w, rows, idxs, okeys, ovals, extra, slot0, c, of)
        order = new_order
    return [rows[i] + tuple(extra[i]) for i in order]


def _window_partition(w, rows, idxs, okeys, ovals, extra, slot0, c, of):
    n = len(idxs)
    # peer groups
    peers_start = [0] * n
    peers_end = [0] * n
    groups = []
    gidx = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and okeys[j + 1] == okeys[i]:
            j += 1
        for k in range(i, j + 1):
            peers_start[k], peers_end[k] = i, j
            gidx[k] = len(groups)
        groups.append((i, j))
        i = j + 1
    desc = of[0][1] if of else False
    so, eo = w.start_off, w.end_off

    def range_cmp(i, which, kind, off):
        cur = ovals[i]
        if cur is None:
            return peers_start[i] if which == "start" else peers_end[i]
        sign = -1 if kind == "preceding" else 1
        if desc:
            sign = -sign
        bound = _add_offset(cur, off, sign)
        if which == "start":
            for k in range(n):
                v = ovals[k]
                if v is None:
                    continue
                if (not desc and T.sort_key(v) >= T.sort_key(bound)) or (desc and T.sort_key(v) <= T.sort_key(bound)):
                    return k
            return n
        last = -1
        for k in range(n):
            v = ovals[k]
            if v is None:
                continue
            if (not desc and T.sort_key(v) <= T.sort_key(bound)) or (desc and T.sort_key(v) >= T.sort_key(bound)):
                last = k
        return last
    for f in w.funcs:
        col = f.slot - slot0
        argf = f.__dict__.get("_argf")
        if argf is None:
            argf = [E.compile_at(a, 0) for a in f.args]
            f._argf = argf
            f._filt = E.compile_at(f.filter, 0) if f.filter is not None else None
        name = f.name
        if name == "row_number":
            for k, i in enumerate(idxs):
                extra[i][col] = k + 1
        elif name == "rank":
            for k, i in enumerate(idxs):
                extra[i][col] = peers_start[k] + 1
        elif name == "dense_rank":
            for k, i in enumerate(idxs):
                extra[i][col] = gidx[k] + 1
        elif name == "percent_rank":
            for k, i in enumerate(idxs):
                extra[i][col] = 0.0 if n <= 1 else peers_start[k] / (n - 1)
        elif name == "cume_dist":
            for k, i in enumerate(idxs):
                extra[i][col] = (peers_end[k] + 1) / n
        elif name == "ntile":
            for k, i in enumerate(idxs):
                b = argf[0](rows[i], c)
                if b is None:
                    extra[i][col] = None
                    continue
                if b <= 0:
                    raise PgError("argument of ntile must be greater than zero", "22014")
                per, rem = divmod(n, b)
                # the first `rem` buckets get per + 1 rows
                if k < rem * (per + 1):
                    extra[i][col] = k // (per + 1) + 1
                else:
                    extra[i][col] = rem + (k - rem * (per + 1)) // per + 1 if per else k + 1
        elif name in ("lag", "lead"):
            for k, i in enumerate(idxs):
                r = rows[i]
                off = argf[1](r, c) if len(argf) > 1 else 1
                if off is None:
                    extra[i][col] = None
                    continue
                t = k - off if name == "lag" else k + off
                if 0 <= t < n:
                    extra[i][col] = argf[0](rows[idxs[t]], c)
                else:
                    extra[i][col] = argf[2](r, c) if len(argf) > 2 else None
        elif name in ("first_value", "last_value", "nth_value"):
            for k, i in enumerate(idxs):
                fr = frame_bounds(n, k, peers_start, peers_end, groups, gidx, w.mode, w.start, w.end, (so, eo),
                                  w.exclude, ovals, range_cmp)
                fr = list(fr)
                if not fr:
                    extra[i][col] = None
                    continue
                if name == "first_value":
                    extra[i][col] = argf[0](rows[idxs[fr[0]]], c)
                elif name == "last_value":
                    extra[i][col] = argf[0](rows[idxs[fr[-1]]], c)
                else:
                    nth = argf[1](rows[i], c)
                    if nth is None:
                        extra[i][col] = None
                        continue
                    if nth <= 0:
                        raise PgError("argument of nth_value must be greater than zero", "22016")
                    extra[i][col] = argf[0](rows[idxs[fr[nth - 1]]], c) if nth <= len(fr) else None
        else:
            _window_agg(f, w, rows, idxs, n, peers_start, peers_end, groups, gidx, so, eo, ovals, range_cmp,
                        extra, col, c, argf)


def _add_offset(v, off, sign):
    if isinstance(v, T.dt.datetime) or isinstance(v, T.dt.date):
        from .funcs import ts_add
        if not isinstance(v, T.dt.datetime):
            v = T.dt.datetime(v.year, v.month, v.day)
        return ts_add(v, off, sign)
    if isinstance(v, T.Interval):
        return v + off if sign > 0 else v - off
    if isinstance(v, float) or isinstance(off, float):
        return float(v) + sign * float(off)
    return v + sign * off


def _window_agg(f, w, rows, idxs, n, peers_start, peers_end, groups, gidx, so, eo, ovals, range_cmp, extra,
                col, c, argf):
    filt = f._filt
    star = f.star

    def vals_of(k):
        r = rows[idxs[k]]
        if filt is not None and not filt(r, c):
            return None
        return [g(r, c) for g in argf] if not star else []
    whole = w.start == "unbounded_preceding" and w.end == "unbounded_following" and w.exclude is None
    running = (w.start == "unbounded_preceding" and w.end == "current_row" and w.exclude is None)
    if whole:
        st = f.agg()
        for k in range(n):
            v = vals_of(k)
            if v is not None:
                st.add(v)
        res = st.result()
        for i in idxs:
            extra[i][col] = res
        return
    if running:
        st = f.agg()
        added = -1
        for k in range(n):
            end = k if w.mode == "rows" else peers_end[k]
            while added < end:
                added += 1
                v = vals_of(added)
                if v is not None:
                    st.add(v)
            extra[idxs[k]][col] = st.result()
        return
    cache = [None] * n
    for k in range(n):
        fr = frame_bounds(n, k, peers_start, peers_end, groups, gidx, w.mode, w.start, w.end, (so, eo), w.exclude,
                          ovals, range_cmp)
        st = f.agg()
        for j in fr:
            if cache[j] is None:
                cache[j] = vals_of(j) or ()
                if vals_of(j) is None:
                    cache[j] = False
            v = cache[j]
            if v is False:
                continue
            st.add(list(v) if v != () or star else [])
        extra[idxs[k]][col] = st.result()


# ---------------------------------------------------------------- SRFs in SELECT

def expand_srfs(q: Query, rows, c):
    fns = _compiled(q, "_srff", lambda: [[E.compile_at(a, 0) for a in f.args] for f in q.srfs])
    out = []
    for r in rows:
        lists = []
        for f, argf in zip(q.srfs, fns):
            args = [g(r, c) for g in argf]
            vals = f.impl(args, c)
            if len(f._srfcols[0]) == 1:
                lists.append([v[0] for v in vals])
            else:
                names, types = f._srfcols
                lists.append([T.Row(v, names, types) for v in vals])
        n = max((len(x) for x in lists), default=0)
        for k in range(n):
            out.append(r + tuple(x[k] if k < len(x) else None for x in lists))
        _guard(len(out))
    return out


# ---------------------------------------------------------------- subqueries in expressions

def run_subplan(sp: E.SubPlan, r, c, testf, shift=0):
    kind = sp.sublink
    if not sp.correlated:
        hit = c.subplan_cache.get(id(sp))
        if hit is None:
            c.outer.append(None)
            try:
                rows = run_query(sp.query, c)
            finally:
                c.outer.pop()
            hit = _prepare(sp, rows)
            c.subplan_cache[id(sp)] = hit
        return _answer(sp, hit, r, c, testf)
    row = ((None,) * shift + tuple(r)) if shift and r is not None else r
    c.outer.append(row)
    try:
        rows = run_query(sp.query, c)
    finally:
        c.outer.pop()
    return _answer(sp, _prepare(sp, rows), r, c, testf)


def _prepare(sp, rows):
    kind = sp.sublink
    if kind == "exists":
        return bool(rows)
    if kind == "expr":
        if len(rows) > 1:
            raise PgError("more than one row returned by a subquery used as an expression", "21000")
        return rows[0][0] if rows else None
    if kind == "array":
        return [row[0] for row in rows]
    if kind == "multiexpr":
        if len(rows) > 1:
            raise PgError("more than one row returned by a subquery used as an expression", "21000")
        return rows[0] if rows else None
    if kind in ("any", "all"):
        impls = sp.op_impl
        if len(impls) == 1 and sp.name == "=" and kind == "any":
            keys = set()
            has_null = False
            for row in rows:
                v = row[0]
                if v is None:
                    has_null = True
                else:
                    keys.add(T.hash_key(v))
            return ("hashed", rows, keys, has_null)
        return ("rows", rows)
    raise AssertionError(kind)


def _answer(sp, prepared, r, c, testf):
    kind = sp.sublink
    if kind in ("exists", "expr", "array", "multiexpr"):
        return prepared
    test = testf(r, c)
    lefts = list(test) if isinstance(sp.test, E.RowExpr) else [test]
    impls = sp.op_impl
    if prepared[0] == "hashed" and len(lefts) == 1:
        _, rows, keys, has_null = prepared
        if not rows:
            return False
        v = lefts[0]
        if v is None:
            return None
        if T.hash_key(v) in keys and _hash_compatible(sp.test.type, impls[0][1]):
            return True
        if not _hash_compatible(sp.test.type, impls[0][1]):
            return _scan_answer(kind, rows, lefts, impls)
        return None if has_null else False
    rows = prepared[1]
    return _scan_answer(kind, rows, lefts, impls)


def _scan_answer(kind, rows, lefts, impls):
    if not rows:
        return kind == "all"
    saw_null = False
    for row in rows:
        res = True
        for (impl, _, _), lv, rv in zip(impls, lefts, row):
            if lv is None or rv is None:
                res = None if res is not False else False
                continue
            if not impl(lv, rv):
                res = False
                break
        if kind == "any":
            if res is True:
                return True
            if res is None:
                saw_null = True
        else:
            if res is False:
                return False
            if res is None:
                saw_null = True
    if saw_null:
        return None
    return kind == "all"

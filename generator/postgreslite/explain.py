"""EXPLAIN [ANALYZE]: plans in PostgreSQL's shape and wording, from a cost model after costsize.c.

Base relations get a Seq Scan, an Index Scan, an Index Only Scan or a Bitmap Heap Scan (whichever is cheapest, with
ANALYZE statistics or PostgreSQL's defaults for never-analyzed tables); joins are Hash Joins or Nested Loops over a
parameterized index scan (Semi / Anti for EXISTS and NOT EXISTS); then aggregation (Aggregate, HashAggregate,
GroupAggregate), WindowAgg, Sort / Incremental Sort, Unique, Limit, InitPlans and SubPlans. EXPLAIN ANALYZE runs the
statement and adds actual row counts and (simulated) timings.
"""

from __future__ import annotations

import math

from . import expr as E
from . import types as T
from .analyze import (Analyzer, Query, ListNode, JoinNode, ScanRel, ScanSub, ScanCte, ScanFunc, ScanValues)
from .catalog import Table, View, Index, Sequence
from .errors import PgError, Unsupported

SEQ_PAGE = 1.0
RANDOM_PAGE = 4.0
CPU_TUPLE = 0.01
CPU_INDEX_TUPLE = 0.005
CPU_OP = 0.0025
DEFAULT_EQ_SEL = 0.005
DEFAULT_INEQ_SEL = 1.0 / 3.0
DEFAULT_MATCH_SEL = 0.005
EFFECTIVE_CACHE_PAGES = 524288


class Plan:
    def __init__(self, name, startup, total, rows, width, details=None, children=None):
        self.name = name
        self.startup, self.total = startup, total
        self.rows = max(1.0, rows) if rows is not None else 1.0
        self.width = int(width)
        self.details = details or []
        self.children = children or []
        self.subplans = []           # (label, Plan)
        self.actual = None           # (rows, loops)
        self.extra_actual = []       # lines shown only with ANALYZE
        self.slots = set()
        self.order = []              # [(slot, desc)] the output is sorted by
        self.rel = None
        self.quals = []
        self.kind = None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()
        for _, p in self.subplans:
            yield from p.walk()


def explain_stmt(session, db, st):
    from .executor import Ctx
    opts = st.options
    fmt = str(opts.get("format", "text")).lower()
    if fmt != "text":
        raise Unsupported(f"EXPLAIN (FORMAT {fmt.upper()})")
    analyze = bool(opts.get("analyze"))
    costs = opts.get("costs", True) is not False
    timing = opts.get("timing", True) is not False
    summary = opts.get("summary", analyze)
    inner = st.stmt
    from . import ast as A
    an = Analyzer(session, db)
    pl = Planner(session, db)
    notices = []
    if isinstance(inner, (A.Select, A.SetOp, A.Values)):
        q = an.query(inner, None)
        pl.multi[0] = _count_rtes(q) > 1
        plan = pl.plan_query(q)
        if analyze:
            pl.analyze_query(q, plan)
    elif isinstance(inner, (A.Insert, A.Update, A.Delete, A.Merge)):
        from .dml import analyze_modify, run_modify
        m = analyze_modify(an, inner, None)
        plan = pl.plan_modify(m, inner)
        if analyze:
            c = Ctx(session, db)
            c.outer.append(None)
            rows = run_modify(m, c)
            plan.actual = (0, 1)
            for p in plan.children:
                p.actual = (m.count, 1)
    elif isinstance(inner, A.CreateTable) and inner.as_query is not None:
        q = an.query(inner.as_query, None)
        plan = pl.plan_query(q)
        if analyze:
            from .ddl import DDL
            DDL(session.cluster, session, db).create_table(inner)
            pl.analyze_query(q, plan)
    elif isinstance(inner, A.Execute):
        hit = session.prepared.get(inner.name)
        if hit is None:
            raise PgError(f'prepared statement "{inner.name}" does not exist', "26000")
        q = an.query(hit[0], None)
        plan = pl.plan_query(q)
    else:
        raise Unsupported("EXPLAIN of this statement")
    lines = render(plan, costs, analyze, timing)
    if analyze or summary:
        if summary:
            lines.append(f"Planning Time: {pl.planning_ms:.3f} ms")
        if analyze and summary:
            lines.append(f"Execution Time: {pl.exec_ms(plan):.3f} ms")
    return lines, notices


# ---------------------------------------------------------------- rendering

def render(plan, costs=True, analyze=False, timing=True):
    out = []

    def head(p):
        s = p.name
        if costs:
            s += f"  (cost={p.startup:.2f}..{p.total:.2f} rows={int(round(p.rows))} width={p.width})"
        if analyze and p.actual is not None:
            rows, loops = p.actual
            if timing:
                t0, t1 = _fake_time(p)
                s += f" (actual time={t0:.3f}..{t1:.3f} rows={int(rows)} loops={loops})"
            else:
                s += f" (actual rows={int(rows)} loops={loops})"
        elif analyze:
            s += " (never executed)"
        return s

    def emit(p, indent, plan_name=None):
        # explain.c ExplainNode (text format): the indent counts pairs of spaces
        if plan_name:
            out.append("  " * indent + plan_name)
            indent += 1
        if indent:
            out.append("  " * indent + "->  " + head(p))
            indent += 2
        else:
            out.append(head(p))
        indent += 1
        pad = "  " * indent
        for d in p.details:
            out.append(pad + d)
        if analyze:
            for d in p.extra_actual:
                out.append(pad + d)
        for lab, sp in p.subplans:
            if lab.startswith("InitPlan"):
                emit(sp, indent, lab)
        for c in p.children:
            emit(c, indent)
        for lab, sp in p.subplans:
            if not lab.startswith("InitPlan"):
                emit(sp, indent, lab)
    emit(plan, 0)
    return out


def _fake_time(p):
    rows = p.actual[0] if p.actual else 0
    t1 = 0.002 + 0.00015 * max(rows, 0) + 0.0004 * len(p.children)
    for c in p.children:
        t1 += _fake_time(c)[1] if c.actual else 0
    t0 = 0.002 if not p.name.startswith(("Sort", "Hash", "HashAggregate", "Aggregate")) else t1 * 0.9
    return round(t0, 3), round(max(t1, t0), 3)


# ---------------------------------------------------------------- statistics

class RelStats:
    """What the planner knows about a table: tuples, pages, per-column stats (or PostgreSQL's guesses)."""

    def __init__(self, rel):
        from .planner import table_pages
        self.rel = rel
        st = getattr(rel, "stats", None)
        self.cols = {}
        if st is not None:
            self.tuples = st["reltuples"]
            self.pages = max(st["relpages"], 0)
            self.cols = st["cols"]
            self.analyzed = True
        else:
            self.analyzed = False
            pages = table_pages(rel) if isinstance(rel, (Table, View)) else 1
            width = sum(col_width(c.type, None) for c in rel.columns) if hasattr(rel, "columns") else 32
            tup = 24 + width + 4
            density = max(1, (8192 - 24) // tup)
            if pages < 10:
                pages = 10
            self.pages = pages
            self.tuples = float(pages * density)
        if isinstance(rel, Sequence):
            self.tuples, self.pages = 1.0, 1

    def col(self, name):
        return self.cols.get(name)

    def ndistinct(self, name):
        cs = self.col(name)
        if cs is None:
            return None
        nd = cs["n_distinct"]
        if nd < 0:
            return -nd * self.tuples
        return nd

    def width(self, name, t):
        cs = self.col(name)
        if cs is not None and cs.get("width") is not None:
            return cs["width"]
        return col_width(t, None)


def col_width(t, sample):
    n = t.name
    fixed = {"bool": 1, "int2": 2, "int4": 4, "int8": 8, "float4": 4, "float8": 8, "date": 4, "timestamp": 8,
             "timestamptz": 8, "time": 8, "interval": 16, "uuid": 16, "oid": 4}
    if n in fixed:
        return fixed[n]
    if n == "numeric":
        if t.mod is None:
            return 32
        return 8 + ((t.mod[0] + 6) // 4) * 2
    if n in ("varchar", "bpchar") and t.mod:
        return min(32, t.mod + 1)
    return 32


# ---------------------------------------------------------------- the planner

class Planner:
    def __init__(self, session, db):
        self.session = session
        self.db = db
        self.settings = session.settings if session is not None else {}
        self.planning_ms = 0.05
        self.alias_count = {}
        self.ids = [0, 0]            # SubPlan / InitPlan numbers and PARAM_EXEC numbers ($N), one EXPLAIN-wide
        self.made = {}               # id(SubPlan) -> what make_subplan decided
        self.pulls = {}              # id(Query) -> (WHERE conjuncts kept, sublinks pulled up into joins)
        self.initplan_list = []      # InitPlans of the query level being planned (attached to its top node)
        self._var_src = {}
        self._slot_rows = {}         # slot -> estimated rows of its base relation after its restrictions
        self.loop_count = 1.0        # > 1 while planning the inner side of a nested loop (cached index pages)
        self.multi = [False]         # does the statement's range table have several entries (qualify names)?
        self._slot_width = {}
        self._flat = {}              # slot of a flattened subquery's output -> the column it is

    def child(self):
        """A planner for a subquery: same numbering, aliases and decisions."""
        sp = Planner(self.session, self.db)
        sp.alias_count, sp.ids, sp.made, sp.pulls = self.alias_count, self.ids, self.made, self.pulls
        sp.multi = self.multi
        return sp

    def enabled(self, what):
        return self.settings.get(f"enable_{what}", "on") == "on"

    def exec_ms(self, plan):
        rows = sum((p.actual[0] if p.actual else 0) for p in plan.walk())
        return round(0.01 + rows * 0.0004, 3)

    # ------------------------------------------------------------ queries
    def plan_query(self, q: Query, outer_rows=1.0, attach_to=None):
        """Plan one query level; its InitPlans hang off the top node and their cost is charged there
        (SS_attach_initplans / SS_charge_for_initplans) -- or, for a subquery pulled up into the level of
        `attach_to`, off that level's top node."""
        saved, self.initplan_list = self.initplan_list, []
        try:
            p = self._plan_query(q)
            init = sorted(self.initplan_list, key=lambda x: x[2])
            if attach_to is not None:
                for item in init:
                    if not any(x[2] == item[2] for x in attach_to.initplan_list):
                        attach_to.initplan_list.append(item)
                return p
            for lab, sub, _ in init:
                p.startup += sub.total
                p.total += sub.total
            p.subplans = [(lab, sub) for lab, sub, _ in init] + p.subplans
            return p
        finally:
            self.initplan_list = saved

    def _plan_query(self, q: Query):
        if q.kind == "values":
            n = len(q.values or [])
            p = Plan("Values Scan on \"*VALUES*\"", 0.0, n * CPU_TUPLE, n, sum(col_width(t, None) for t in q.types
                                                                               if t is not None))
            return self.finish(q, p)
        if q.kind == "setop":
            return self.plan_setop(q)
        self.premake(q)
        rest, specs = self.pulls[id(q)]
        self.qualify = self.multi[0] or self._count_rels(q.from_) + len(specs) > 1
        if q.from_ is None:
            p = Plan("Result", 0.0, 0.01, 1, sum(col_width(t, None) for t in q.types[:q.ntargets]))
            if rest:
                p.details.append(f"One-Time Filter: {self.dp(E.make_and(rest))}")
                for c in rest:
                    self.attach_expr_subplans(p, c, 1.0, 1.0)
            return self.finish(q, p, grouped_done=False)
        p = self.plan_from(q, q.from_, rest)
        for spec in specs:
            p = self.plan_semi(p, q, spec, rest)
        return self.finish(q, p, grouped_done=False)

    def _index_size(self, ix, rel, rs):
        """(pages, tree height) of a btree: 8-byte tuple headers + MAXALIGNed keys + line pointers, leaves 90%
        full; duplicates of a non-unique index share one key with a posting list of 6-byte TIDs (PostgreSQL 13+
        deduplication; not for numeric, whose equal values may display differently)."""
        n = max(rs.tuples, 0.0)
        if n <= 0:
            return 1.0, 0
        kw = 0
        dedup = not ix.unique and ix.method == "btree" and not ix.include
        nd = 1.0
        for el in ix.elems:
            if el.col is not None:
                t = rel.coltypes[rel.col_index(el.col)]
                kw += rs.width(el.col, t)
                if t.name == "numeric":
                    dedup = False
                d, isdefault = self._ndistinct((rs, el.col, n))
                nd *= d if not isdefault else n
            else:
                kw += 8
                dedup = False
        tup = 8 + ((kw + 7) // 8) * 8
        nd = min(max(nd, 1.0), n)
        if dedup and n / nd > 1.5:
            post = ((6 * (n / nd) + 7) // 8) * 8
            total = nd * (tup + post + 4)
        else:
            total = n * (tup + 4)
        leaf = max(1, math.ceil(total / (8152 * 0.9)))
        per_internal = 8152 // (tup + 4)
        height, level = 0, leaf
        inner = 0
        while level > 1:
            level = math.ceil(level / per_internal)
            inner += level
            height += 1
        return float(leaf + inner + 1), height

    def _descent(self, ix, rel, rs):
        """btcostestimate: a binary search per level and 50 operator costs per page descended."""
        _, height = self._index_size(ix, rel, rs)
        return math.ceil(math.log2(max(rs.tuples, 2))) * CPU_OP + (height + 1) * 50 * CPU_OP

    def _allvisible(self, rel):
        return getattr(rel, "vacuumed", None) is not None and rel.vacuumed == getattr(rel, "n_changes", 0)

    def _alias(self, rte, rel):
        """set_rtable_names: a relation's name in the plan, made unique (orders, orders_1 ...) in the order the
        range table lists them: this level's FROM, pulled-up subqueries, then subplans as they are made."""
        got = self.alias_count.get(("rte", id(rte)))
        if got is not None:
            return got
        name = rte.name if rte.alias_given else rel.name
        k = self.alias_count.get(name, 0)
        self.alias_count[name] = k + 1
        alias = name if not k else f"{name}_{k}"
        if not rte.alias_given and not k:
            alias = rte.name
        self.alias_count[("rte", id(rte))] = alias
        return alias

    def _alias_from(self, node):
        if isinstance(node, ListNode):
            for it in node.items:
                self._alias_from(it)
        elif isinstance(node, JoinNode):
            self._alias_from(node.left)
            self._alias_from(node.right)
        elif isinstance(node, ScanRel):
            self._alias(node.rte, node.rel)

    def _count_rels(self, node):
        if node is None:
            return 0
        if isinstance(node, ListNode):
            return sum(self._count_rels(i) for i in node.items)
        if isinstance(node, JoinNode):
            return self._count_rels(node.left) + self._count_rels(node.right)
        return 1

    def dp(self, e):
        if self._flat:
            e = _subst(e, lambda v: self._flat.get(v.index, v))
        return e.deparse(E.DeparseCtx(qualify=getattr(self, "qualify", False), planned=True))

    def dps(self, e):
        """Scan-level quals are shown without table names."""
        return e.deparse(E.DeparseCtx(qualify=False, planned=True))

    def finish(self, q, p, grouped_done=True):
        width_out = sum(self._width_of(t, q) for t in q.targets[:q.ntargets]) if q.kind == "select" else p.width
        if q.kind == "select" and not q.grouped:
            for t in q.targets:
                self.attach_expr_subplans(p, t, p.rows, p.rows)
        if q.kind == "select" and q.grouped:
            p = self.plan_agg(q, p)
            for t in list(q.targets) + ([q.having] if q.having is not None else []):
                self.attach_expr_subplans(p, t, p.rows, p.rows)
        if q.kind == "select" and q.windows:
            for w in q.windows:
                keys = [(x, False) for x in w.partition] + [(o[0], o[1]) for o in w.order]
                if keys and not self._sorted_by(p, keys):
                    p = self.sort_node(p, keys)
                per = CPU_OP * (len(w.partition) + len(w.order)) + CPU_OP * len(w.funcs) + CPU_TUPLE
                p = Plan("WindowAgg", p.startup, p.total + p.rows * per, p.rows, p.width + 8 * len(w.funcs),
                         children=[p])
        if q.kind == "select" and q.distinct is not None:
            if isinstance(q.distinct, list):
                keys = [(q.targets[i], False) for i in q.distinct]
                if not self._sorted_by(p, keys):
                    p = self.sort_node(p, [(q.targets[o[0]], o[1]) for o in q.order] or keys)
                p = Plan("Unique", p.startup, p.total + p.rows * CPU_OP, max(1, p.rows / 2), p.width, children=[p])
            else:
                keys = [self.dp(t) for t in q.targets[:q.ntargets]]
                groups = self._ngroups(list(q.targets[:q.ntargets]), p.rows)
                st = p.total + p.rows * CPU_OP * len(keys)
                p = Plan("HashAggregate", st, st + groups * CPU_TUPLE, groups, p.width, [f"Group Key: {', '.join(keys)}"],
                         [p])
        if q.order:
            keys = [(q.targets[i], desc) for i, desc, nf, _ in q.order]
            if not self._sorted_by(p, keys):
                limit = self._limit_value(q)
                p = self.sort_node(p, keys, nulls=[(nf, desc) for _, desc, nf, _ in q.order], limit=limit)
        if q.limit is not None or q.offset is not None:
            lim = self._limit_value(q)
            off = self._offset_value(q)
            rows = p.rows
            if lim is not None:
                rows = min(rows, lim + off)
            frac = (rows / p.rows) if p.rows else 1.0
            total = p.startup + (p.total - p.startup) * frac
            p = Plan("Limit", p.startup, total, max(1.0, min(rows, lim if lim is not None else rows)), p.width,
                     children=[p])
        if q.kind == "select" and not q.windows:
            p.width = int(width_out)
            for c in p.children:
                pass
        return p

    def _limit_value(self, q):
        if q.limit is None:
            return None
        try:
            v = q.limit.compile()(None, None)
        except Exception:  # noqa: BLE001
            return None
        return v

    def _offset_value(self, q):
        if q.offset is None:
            return 0
        try:
            return q.offset.compile()(None, None) or 0
        except Exception:  # noqa: BLE001
            return 0

    def _width_of(self, e, q=None):
        if q is not None and q.grouped and isinstance(e, E.Var) and e.level == 0:
            # a grouped row is the group keys, then the aggregate values
            nk = len(q.group_keys)
            if e.index < nk:
                return self._width_of(q.group_keys[e.index])
            return col_width(e.type, None)
        src = getattr(self, "_var_src", {}).get(e.index) if isinstance(e, E.Var) and e.level == 0 else None
        if src is not None:
            rs, name = src
            return rs.width(name, e.type)
        return col_width(e.type, None)

    def sort_node(self, p, keys, nulls=None, limit=None):
        n = max(p.rows, 2.0)
        if limit is not None and limit < n:
            cost = 2.0 * CPU_OP * n * math.log2(max(2 * limit, 2))
        else:
            cost = 2.0 * CPU_OP * n * math.log2(n)
        parts = []
        for k, (e, desc) in enumerate(keys):
            s = self.dp(e)
            if desc:
                s += " DESC"
            if nulls is not None and k < len(nulls):
                nf, d = nulls[k]
                if nf and not d:
                    s += " NULLS FIRST"
                elif not nf and d:
                    s += " NULLS LAST"
            parts.append(s)
        # an input already sorted by a prefix: Incremental Sort
        pre = 0
        for (e, desc), (slot, d2) in zip(keys, p.order):
            if isinstance(e, E.Var) and e.index == slot and desc == d2:
                pre += 1
            else:
                break
        if 0 < pre < len(keys):
            s = Plan("Incremental Sort", p.startup + 1.4, p.total + cost * 0.6, p.rows, p.width,
                     [f"Sort Key: {', '.join(parts)}", f"Presorted Key: {', '.join(parts[:pre])}"], [p])
        else:
            s = Plan("Sort", p.total + cost, p.total + cost + p.rows * CPU_OP, p.rows, p.width,
                     [f"Sort Key: {', '.join(parts)}"], [p])
        s.extra_actual = ["Sort Method: quicksort  Memory: 25kB"] if not (limit and limit < n) else \
            ["Sort Method: top-N heapsort  Memory: 25kB"]
        s.order = [(e.index if isinstance(e, E.Var) else -1, d) for e, d in keys]
        return s

    def _sorted_by(self, p, keys):
        if not p.order or len(p.order) < len(keys):
            return False
        for (e, desc), (slot, d2) in zip(keys, p.order):
            if not (isinstance(e, E.Var) and e.level == 0 and e.index == slot and desc == d2):
                return False
        return True

    def plan_agg(self, q, p):
        keys = q.group_keys
        if q.grouping_sets is not None:
            name = "MixedAggregate"
            details = []
            for gs in q.grouping_sets:
                if gs:
                    details.append(f"Hash Key: {', '.join(self.dp(keys[i]) for i in gs)}")
                else:
                    details.append("Group Key: ()")
            groups = sum(max(1.0, self._ngroups([keys[i] for i in gs], p.rows)) for gs in q.grouping_sets)
            a = Plan(name, p.total, p.total + p.rows * CPU_OP * (len(keys) + 1), groups, 32, details, [p])
        elif not keys:
            a = Plan("Aggregate", p.total + p.rows * CPU_OP * max(1, len(q.aggs)),
                     p.total + p.rows * CPU_OP * max(1, len(q.aggs)) + CPU_TUPLE, 1, 8 * max(1, len(q.aggs)),
                     children=[p])
            a.kind = "plain"
        else:
            groups = self._ngroups(keys, p.rows)
            keytext = ", ".join(self.dp(k) for k in keys)
            if self._sorted_by(p, [(k, False) for k in keys]) or not self.enabled("hashagg"):
                if not self._sorted_by(p, [(k, False) for k in keys]):
                    p = self.sort_node(p, [(k, False) for k in keys])
                a = Plan("GroupAggregate", p.startup, p.total + p.rows * CPU_OP * (len(keys) + len(q.aggs)),
                         groups, 32, [f"Group Key: {keytext}"], [p])
            else:
                cost = p.total + p.rows * CPU_OP * (len(keys) + len(q.aggs))
                a = Plan("HashAggregate", cost, cost + groups * CPU_TUPLE, groups, 32, [f"Group Key: {keytext}"],
                         [p])
                a.extra_actual = ["Batches: 1  Memory Usage: 24kB"]
        if q.having is not None:
            a.details.append(f"Filter: {self.dp(q.having)}")
            # the HAVING qual per group: its operators, and the final function of each aggregate it reads
            finals = sum(1 for n in q.having.walk() if isinstance(n, E.Aggref) and _has_final(n)) + \
                sum(1 for n in q.having.walk() if isinstance(n, E.Var) and n.level == 0 and
                    n.index >= len(keys) and n.index - len(keys) < len(q.aggs) and
                    _has_final(q.aggs[n.index - len(keys)]))
            a.total += a.rows * CPU_OP * (_nops(q.having) + finals)
            a.rows = max(1.0, a.rows * DEFAULT_INEQ_SEL)
        return a

    def _ngroups(self, keys, rows):
        n = 1.0
        for k in keys:
            info = self._vstats(k)
            if info is not None:
                nd, _ = self._ndistinct(info)
                rs, col, rel_rows = info
                if rs.tuples > 0 and rel_rows < rs.tuples and nd > 0:
                    nd *= 1.0 - ((rs.tuples - rel_rows) / rs.tuples) ** (rs.tuples / nd)
                n *= _clamp_rows(nd)
            else:
                nd = self._distinct_of(k)
                n *= nd if nd else 200.0
        return max(1.0, min(n, rows))

    def _distinct_of(self, e):
        src = getattr(self, "_var_src", {}).get(e.index) if isinstance(e, E.Var) else None
        if src is None:
            return None
        rs, col = src
        return rs.ndistinct(col)

    # ------------------------------------------------------------ FROM
    def plan_from(self, q, node, conj):
        self._var_src = {}
        pending = list(conj)
        p = self.plan_node(node, pending, q)
        if pending:
            rest = [c for c in pending]
            self.decide_alternatives(rest, 2 * p.rows * self._sel_list(rest))
            p.details.append(f"Filter: {self.dp(E.make_and(rest))}")
            n_in = p.rows
            p.rows = max(1.0, p.rows * self._sel_list(rest))
            for c in rest:
                self.attach_expr_subplans(p, c, n_in, 2 * p.rows)
        return p

    def _slots(self, node):
        return set(range(node.base, node.base + node.width))

    def _take(self, pending, slots):
        out = []
        for c in list(pending):
            vs = {v.index for v in E.vars_of(c, 0)}
            if vs and vs <= slots:
                out.append(c)
                pending.remove(c)
        return out

    def plan_node(self, node, pending, q):
        if isinstance(node, ListNode):
            return self.plan_list(node, pending, q)
        if isinstance(node, JoinNode):
            mine = self._take(pending, self._slots(node))
            p = self.plan_join_node(node, q, mine)
            return p
        mine = self._take(pending, self._slots(node))
        return self.plan_scan(node, mine, q)

    def _derive(self, quals, joinquals):
        """Equivalence classes: a = b (join) and a = const give b = const too."""
        out = []
        consts = {}
        for c in quals:
            if isinstance(c, E.Op) and c.op == "=" and len(c.args) == 2:
                a, b = c.args
                if isinstance(a, E.Var) and not E.vars_of(b, 0):
                    consts[a.index] = (c, b)
                elif isinstance(b, E.Var) and not E.vars_of(a, 0):
                    consts[b.index] = (c, a)
        for j in joinquals:
            if isinstance(j, E.Op) and j.op == "=" and len(j.args) == 2:
                a, b = j.args
                if isinstance(a, E.Var) and isinstance(b, E.Var) and a.type == b.type:
                    for x, y in ((a, b), (b, a)):
                        if x.index in consts and y.index not in consts:
                            orig, val = consts[x.index]
                            out.append(E.Op("=", [y, val], T.BOOL, orig.impl))
        return out

    def plan_list(self, node, pending, q):
        items = node.items
        acc = None
        acc_slots = set()
        for it in items:
            islots = self._slots(it)
            if acc is None:
                acc = self.plan_node(it, pending, q)
                acc_slots = islots
                continue
            joinq = []
            for c in list(pending):
                vs = {v.index for v in E.vars_of(c, 0)}
                if vs and vs <= (acc_slots | islots) and vs & islots and vs & acc_slots:
                    joinq.append(c)
                    pending.remove(c)
            right = self.plan_node(it, pending, q)
            acc = self.join(acc, right, "inner", joinq, acc_slots, islots, it, q,
                            left_node=items[0] if len(acc_slots) == len(self._slots(items[0])) else None)
            acc_slots |= islots
        return acc

    def plan_join_node(self, node, q, pushed=()):
        quals = E.conjuncts(node.quals)
        ls, rs = self._slots(node.left), self._slots(node.right)
        pushed = list(pushed)
        derived = self._derive(pushed, quals) if node.kind == "inner" else []
        to_left, to_right, keep = [], [], []
        for c in pushed + derived:
            vs = {v.index for v in E.vars_of(c, 0)}
            if vs and vs <= ls and node.kind in ("inner", "left"):
                to_left.append(c)
            elif vs and vs <= rs and node.kind in ("inner", "right"):
                to_right.append(c)
            elif c not in derived:
                keep.append(c)
        left = self.plan_node(node.left, to_left, q)
        right = self.plan_node(node.right, to_right, q)
        p = self.join(left, right, node.kind, quals, ls, rs, node.right, q, both_fixed=bool(derived),
                      left_node=node.left)
        if keep:
            p.details.append(f"Filter: {self.dp(E.make_and(keep))}")
            p.rows = max(1.0, p.rows * self._sel_list(keep))
        return p

    def join(self, left, right, kind, quals, lslots, rslots, right_node, q, both_fixed=False, left_node=None):
        equi = []
        other = []
        for c in quals:
            if isinstance(c, E.Op) and c.op == "=" and len(c.args) == 2:
                a, b = c.args
                sa = {v.index for v in E.vars_of(a, 0)}
                sb = {v.index for v in E.vars_of(b, 0)}
                if sa and sb and ((sa <= lslots and sb <= rslots) or (sb <= lslots and sa <= rslots)):
                    equi.append(c)
                    continue
            other.append(c)
        pairs = []        # (left expr, right expr)
        for c in equi:
            a, b = c.args
            if {v.index for v in E.vars_of(a, 0)} <= lslots:
                pairs.append((a, b))
            else:
                pairs.append((b, a))
        jsel = 1.0
        for le, re_ in pairs:
            jsel *= self._eqjoinsel(le, re_, self, self)
        inner_rows = left.rows * right.rows * jsel
        rows = inner_rows * self._sel_list(other)
        if kind == "left":
            rows = max(rows, left.rows)
        elif kind == "right":
            rows = max(rows, right.rows)
        elif kind == "full":
            rows = max(rows, left.rows + right.rows)
        rows = max(1.0, rows)
        width = self._join_width(lslots | rslots, q, left.width + right.width)
        if both_fixed and left.rows <= 10:
            rows = max(1.0, left.rows * right.rows)
            p = Plan("Nested Loop", left.startup + right.startup,
                     left.total + right.total + max(0.0, left.rows - 1) * right.total + rows * CPU_TUPLE, rows, width,
                     children=[left, right])
            return p
        cands = []
        # nested loop over a parameterized index scan of the right side
        if equi and kind in ("inner", "left") and isinstance(right_node, ScanRel) and self.enabled("nestloop"):
            inner = self.param_index_scan(right_node, equi, lslots, loops=max(1.0, left.rows))
            if inner is not None:
                runi = inner.total - inner.startup
                cost = left.total + max(0.0, left.rows - 1) * inner.startup + left.rows * runi + \
                    inner_rows * CPU_TUPLE
                p = Plan(f"Nested Loop{' Left Join' if kind == 'left' else ''}", left.startup + inner.startup, cost,
                         rows, width, children=[left, inner])
                if other:
                    p.details.append(f"Join Filter: {self.dp(E.make_and(other))}")
                cands.append(p)
        if equi and self.enabled("hashjoin"):
            names = {"inner": ("Hash Join", "Hash Join"), "left": ("Hash Left Join", "Hash Right Join"),
                     "right": ("Hash Right Join", "Hash Left Join"), "full": ("Hash Full Join", "Hash Full Join")}
            for swapped in (False, True):
                probe, build = (right, left) if swapped else (left, right)
                bnode = left_node if swapped else right_node
                pp = [(b, a) for a, b in pairs] if swapped else pairs
                uniq = kind != "full" and len(pp) == 1 and isinstance(bnode, ScanRel) and self._unique_key(pp[0][1])
                st, tot = self.hash_cost(probe, build, pp, self, other, kind="inner", frac=jsel, match=build.rows,
                                         rows=inner_rows, inner_unique=uniq)
                h = Plan("Hash", build.total, build.total, build.rows, build.width, children=[build])
                h.extra_actual = ["Buckets: 1024  Batches: 1  Memory Usage: 9kB"]
                p = Plan(names[kind][1 if swapped else 0], st, tot, rows, width,
                         [f"Hash Cond: {self._cond(equi, rslots if swapped else lslots)}"], [probe, h])
                if other:
                    p.details.append(f"Join Filter: {self.dp(E.make_and(other))}")
                cands.append(p)
        if cands:
            return min(cands, key=lambda c: (c.total, c.startup))
        cost = left.total + left.rows * right.total + left.rows * right.rows * CPU_TUPLE
        inner = right
        if right.rows > 1:
            inner = Plan("Materialize", right.startup, right.total + right.rows * CPU_OP, right.rows, right.width,
                         children=[right])
        jname = {"inner": "", "left": "Left ", "right": "Right ", "full": "Full ", "cross": ""}[kind]
        p = Plan(f"Nested Loop{' ' + jname.strip() + ' Join' if jname.strip() else ''}", left.startup + right.startup,
                 cost, rows, width, children=[left, inner])
        if other or equi:
            p.details.append(f"Join Filter: {self.dp(E.make_and(other + equi))}")
        return p

    def _join_width(self, slots, q, default):
        """A join outputs the columns needed above it: by the query's results, and by quals evaluated higher up."""
        if q is None or q.kind != "select":
            return default
        out = set()
        exprs = ([] if q.grouped else list(q.targets) + [q.having]) + list(q.group_keys)
        for e in exprs:
            if e is not None:
                out |= {v.index for v in E.vars_of(e, 0)}
        for a in q.aggs:
            for x in a.args:
                out |= {v.index for v in E.vars_of(x, 0)}
        for w in q.windows:
            for x in list(w.partition) + [o[0] for o in w.order] + [a for f in w.funcs for a in f.args]:
                out |= {v.index for v in E.vars_of(x, 0)}
        quals = list(E.conjuncts(q.where))
        _join_quals(q.from_, quals)
        for c in quals:
            vs = {v.index for v in E.vars_of(c, 0)}
            if not vs <= slots:
                out |= vs
        if not all(sl in self._slot_width for sl in slots & out):
            return default
        return int(sum(self._slot_width[sl] for sl in slots & out))

    def _cond(self, equi, probe_slots):
        parts = []
        for c in equi:
            a, b = c.args
            sa = {v.index for v in E.vars_of(a, 0)}
            if not sa <= probe_slots:
                a, b = b, a
            parts.append(f"({self.dp(a)} = {self.dp(b)})")
        return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"

    def _join_sel(self, c):
        a, b = c.args
        nds = [self._distinct_of(x) for x in (a, b)]
        nds = [n for n in nds if n]
        if not nds:
            return DEFAULT_EQ_SEL
        return 1.0 / max(nds)

    # ------------------------------------------------------------ scans
    def plan_scan(self, node, quals, q):
        if isinstance(node, ScanRel):
            return self.rel_scan(node, quals, q)
        if isinstance(node, ScanCte) and self._inline_cte(node):
            node = ScanSub(node.rte, node.cte.query)
        if isinstance(node, ScanSub):
            q2 = node.query
            if quals:
                pushed = self._push_quals(q2, node, quals)
                if pushed is not None:
                    q2, quals = pushed
            child = self.child()
            simple = q2.kind == "select" and not q2.grouped and not q2.windows and q2.distinct is None and \
                q2.limit is None and all(isinstance(t, E.Var) and t.level == 0 for t in q2.targets[:q2.ntargets])
            need = set()
            if simple and not quals:
                # flattened (pull_up_subqueries): only the columns used above are produced
                need = {k for k, sl in enumerate(node.rte.slots[:q2.ntargets]) if sl in self._needed_slots(q, node.rte)}
                q2 = _copy_query(q2)
                q2.targets = [t if k in need or k >= q2.ntargets else E.Const(None, t.type)
                              for k, t in enumerate(q2.targets)]
            sub = child.plan_query(q2)
            if q2.kind == "select" and not q2.grouped:
                for k, sl in enumerate(node.rte.slots[:q2.ntargets]):
                    t = q2.targets[k]
                    if isinstance(t, E.Var) and t.level == 0 and t.index in child._var_src:
                        self._var_src[sl] = child._var_src[t.index]
                        self._slot_rows[sl] = child._slot_rows.get(t.index, sub.rows)
                        self._slot_width[sl] = child._slot_width.get(t.index, col_width(t.type, None))
                        if simple and not quals:
                            self._flat[sl] = t
            if simple and not quals:
                sub.width = int(sum(child._width_of(q2.targets[k]) for k in need))
            alias = node.rte.name or "unnamed_subquery"
            p = Plan(f"Subquery Scan on {alias}", sub.startup, sub.total + sub.rows * CPU_TUPLE, sub.rows, sub.width,
                     children=[sub])
            if quals:
                p.details.append(f"Filter: {self.dp(E.make_and(quals))}")
                p.rows *= self._sel_list(quals)
            if not quals and node.rte.name is None:
                return sub
            if not quals:
                return sub
            return p
        if isinstance(node, ScanCte):
            name = node.cte.name
            alias = node.rte.name
            label = f"CTE Scan on {name}" + (f" {alias}" if alias and alias != name else "")
            rows = 1000.0
            p = Plan(label, 0.0, rows * CPU_TUPLE * 2, rows, sum(col_width(t, None) for t in node.cte.coltypes))
            if quals:
                p.details.append(f"Filter: {self.dp(E.make_and(quals))}")
                p.rows *= self._sel_list(quals)
            return p
        if isinstance(node, ScanFunc):
            f = node.funcs[0]
            alias = node.rte.name
            rows = 1000.0
            if f.name == "generate_series":
                rows = 1000.0
            elif f.name == "unnest":
                rows = 10.0
            elif f.name.startswith(("jsonb_array", "json_array", "jsonb_each", "json_each")):
                rows = 100.0
            p = Plan(f"Function Scan on {f.name} {alias}", 0.0, rows * CPU_TUPLE, rows,
                     sum(col_width(t, None) for t in node.rte.coltypes))
            if quals:
                p.details.append(f"Filter: {self.dp(E.make_and(quals))}")
                p.rows *= self._sel_list(quals)
            return p
        if isinstance(node, ScanValues):
            n = len(node.query.values or [])
            return Plan('Values Scan on "*VALUES*"', 0.0, n * CPU_TUPLE, n, 32)
        raise Unsupported("EXPLAIN of this FROM item")

    def rel_scan(self, node, quals, q):
        rel = node.rel
        rte = node.rte
        rs = RelStats(rel)
        for i, (cn, ct) in enumerate(zip(rte.colnames, rte.coltypes)):
            self._var_src[rte.slots[i]] = (rs, rel.colnames[i] if hasattr(rel, "colnames") and
                                           i < len(rel.colnames) else cn)
        alias = self._alias(rte, rel)
        label = rel.name if alias == rel.name else f"{rel.name} {alias}"
        needed = self._needed_slots(q, rte)
        out = self._needed_slots(q, rte, output=True)
        width = sum(rs.width(rel.colnames[i], rel.coltypes[i]) for i in range(len(rel.columns))
                    if rte.slots[i] in out) if hasattr(rel, "columns") and rel.columns else 0
        for i in range(len(rel.columns) if hasattr(rel, "columns") else 0):
            self._slot_width[rte.slots[i]] = rs.width(rel.colnames[i], rel.coltypes[i])
        sel = self._sel_list(quals, rs, rte)
        rows = max(1.0, rs.tuples * sel)
        nq = sum(_nops(c) for c in quals)
        seq_total = rs.pages * SEQ_PAGE + rs.tuples * (CPU_TUPLE + CPU_OP * nq)
        for sl in rte.slots[:len(rel.columns) if hasattr(rel, "columns") else 0]:
            self._slot_rows[sl] = rows
        self.decide_alternatives(quals, 2 * rows)
        best = Plan(f"Seq Scan on {label}", 0.0, seq_total, rows, width)
        if quals:
            best.details.append(f"Filter: {self.dps(E.make_and(quals))}")
        best.rel = rel
        best.quals = quals
        best.kind = "seq"
        if not self.enabled("seqscan"):
            best.startup += 1e10
            best.total += 1e10
        if isinstance(rel, Table):
            cands = [best]
            for ix in self.db.indexes_of(rel):
                cands += self.index_paths(ix, rel, rte, rs, quals, needed, width, label, q)
            keys = self._order_keys(q) if self._count_rels(q.from_) == 1 and not (q.grouped or q.windows or
                                                                                   q.distinct) else None
            lim = self._limit_value(q) if keys and q.limit is not None else None

            def eff(c):
                if not keys:
                    return c.total
                n = max(c.rows, 2.0)
                if self._sorted_by(c, keys):
                    if lim is not None:
                        return c.startup + (c.total - c.startup) * min(1.0, lim / n)
                    return c.total
                if lim is not None and lim < n:
                    sort = 2.0 * CPU_OP * n * math.log2(max(2 * lim, 2))
                else:
                    sort = 2.0 * CPU_OP * n * math.log2(n)
                return c.total + sort + (0 if lim is None else 0)
            best = min(cands, key=eff)
        for c in quals:
            if any(isinstance(n, E.SubPlan) for n in c.walk()):
                # quals with subplans are filters: evaluated on every tuple the path fetches
                fetched = rs.tuples if best.kind == "seq" else max(best.rows, 1.0)
                self.attach_expr_subplans(best, c, fetched, 2 * best.rows)
        return best

    def _order_keys(self, q):
        if not q.order:
            return None
        keys = []
        for i, desc, nf, _ in q.order:
            t = q.targets[i]
            if not isinstance(t, E.Var) or t.level != 0:
                return None
            keys.append((t, desc))
        return keys

    def _needed_slots(self, q, rte, output=False):
        """The columns of this level the query uses (index-only scans need them all in the index); output=True:
        only those needed above the scan (its own quals are evaluated inside it), which make its width."""
        need = set()
        mine = set(rte.slots)
        where = []
        for c in E.conjuncts(q.where):
            if not output or not {v.index for v in E.vars_of(c, 0)} <= mine:
                where.append(c)
        # a grouped query's targets and HAVING are on the grouped row: its keys and aggregates cover the columns
        exprs = ([] if q.grouped else list(q.targets) + [q.having]) + where + list(q.group_keys)
        for e in exprs:
            if e is None:
                continue
            for v in E.vars_of(e, 0):
                need.add(v.index)
            for n in e.walk():
                if isinstance(n, E.Aggref):
                    for a in n.args:
                        for v in E.vars_of(a, 0):
                            need.add(v.index)
        for w in q.windows:
            for x in list(w.partition) + [o[0] for o in w.order]:
                for v in E.vars_of(x, 0):
                    need.add(v.index)
            for f in w.funcs:
                for a in f.args:
                    for v in E.vars_of(a, 0):
                        need.add(v.index)
        _from_vars(q.from_, need)
        for a in q.aggs:
            for x in a.args:
                for v in E.vars_of(x, 0):
                    need.add(v.index)
        return need

    def index_paths(self, ix, rel, rte, rs, quals, needed, width, label, q):
        """Index Scan / Index Only Scan / Bitmap Heap Scan paths for one index, or [] if it can't help."""
        if ix.method == "gin":
            matched = [c for c in quals if self._gin_match(c, ix, rel, rte)]
            if not matched:
                return []
            return [self._bitmap(ix, rel, rs, matched, [c for c in quals if c not in matched], width, label, rte,
                                 sel=self._sel_list(matched, rs, rte))]
        pred_quals = []
        if ix.pred is not None:
            implied = self._pred_implied(ix, quals, rel, rte)
            if implied is False:
                return []
            pred_quals = implied
        quals_all = quals
        quals = [c for c in quals if c not in pred_quals]
        matched = []
        for k, el in enumerate(ix.elems):
            hits = [c for c in quals if self._btree_match(c, el, rel, rte)]
            if not hits:
                break
            matched += hits
            if not any(isinstance(c, E.Op) and c.op == "=" for c in hits):
                break
        eq_prefix = set()
        for el in ix.elems:
            if el.col is None:
                break
            slot = rte.slots[rel.col_index(el.col)]
            if any(isinstance(c, E.Op) and c.op == "=" and any(_is_var(a, slot) for a in c.args) for c in matched):
                eq_prefix.add(slot)
            else:
                break
        order_ok = self._index_order(ix, rel, rte, q, eq_prefix)
        if not matched and not order_ok and not pred_quals:
            return []
        rest = [c for c in quals if c not in matched]
        sel = self._sel_list(matched + pred_quals, rs, rte) if (matched or pred_quals) else 1.0
        n_idx = max(1.0, round(sel * rs.tuples))
        ipages = self._index_size(ix, rel, rs)[0]
        npages = max(1, math.ceil(n_idx * ipages / max(rs.tuples, 1)))
        descent = self._descent(ix, rel, rs)
        loops = max(1.0, self.loop_count)
        if loops > 1:
            npages = _pages_fetched(npages * loops, ipages) / loops
        idx_total = descent + npages * RANDOM_PAGE + n_idx * (CPU_INDEX_TUPLE + CPU_OP * len(matched))
        heap_pages = rs.pages
        corr = self._correlation(rs, rel, ix)
        pf_corr = max(1.0, math.ceil(sel * heap_pages))
        if loops > 1:
            max_io = _pages_fetched(n_idx * loops, heap_pages) * RANDOM_PAGE / loops
            min_io = _pages_fetched(pf_corr * loops, heap_pages) * RANDOM_PAGE / loops
        else:
            max_io = _pages_fetched(n_idx, heap_pages) * RANDOM_PAGE
            min_io = RANDOM_PAGE + (pf_corr - 1) * SEQ_PAGE
        io = max_io + corr * corr * (min_io - max_io)
        rows = max(1.0, rs.tuples * self._sel_list(quals_all, rs, rte))
        cpu = n_idx * (CPU_TUPLE + CPU_OP * len(rest))
        out = []
        idx_cols = {rte.slots[rel.col_index(el.col)] for el in ix.elems if el.col is not None}
        only = needed & set(rte.slots[:len(rel.columns)]) <= idx_cols and all(el.col for el in ix.elems)
        cond = self._cond_text(matched)
        backward = order_ok == "backward"
        scan_word = "Index Scan Backward" if backward else "Index Scan"
        if only and self.enabled("indexonlyscan"):
            # heap pages are visited only where the visibility map doesn't say all-visible
            heap = 0.0 if self._allvisible(rel) else io
            p = Plan(f"Index Only Scan{' Backward' if backward else ''} using {ix.name} on {label}", descent,
                     idx_total + heap + n_idx * CPU_TUPLE, rows, width)
            p.allvis = self._allvisible(rel)
        else:
            p = Plan(f"{scan_word} using {ix.name} on {label}", descent, idx_total + io + cpu, rows, width)
        if cond:
            p.details.append(f"Index Cond: {cond}")
        if rest:
            p.details.append(f"Filter: {self.dps(E.make_and(rest))}")
        if only:
            p.extra_actual = ["Heap Fetches: 0"]
            p.kind_only = True
        p.kind = "index"
        p.rel = rel
        p.quals = quals
        p.order = [(rte.slots[rel.col_index(el.col)] if el.col else -1, el.desc != backward) for el in ix.elems
                   if not (el.col and rte.slots[rel.col_index(el.col)] in eq_prefix)]
        if not self.enabled("indexscan"):
            p.startup += 1e10
            p.total += 1e10
        out.append(p)
        if matched and not only and self.enabled("bitmapscan") and not backward:
            out.append(self._bitmap(ix, rel, rs, matched, rest, width, label, rte, sel=sel, n_idx=n_idx,
                                    idx_total=idx_total, rows=rows))
        return out

    def _bitmap(self, ix, rel, rs, matched, rest, width, label, rte, sel, n_idx=None, idx_total=None, rows=None):
        if n_idx is None:
            n_idx = max(1.0, round(sel * rs.tuples))
        if idx_total is None:
            ipages = self._index_size(ix, rel, rs)[0]
            npages = max(1, math.ceil(n_idx * ipages / max(rs.tuples, 1)))
            idx_total = npages * RANDOM_PAGE + n_idx * (CPU_INDEX_TUPLE + CPU_OP) + 0.17
        if rows is None:
            rows = max(1.0, rs.tuples * sel * self._sel_list(rest, rs, rte))
        pages = rs.pages
        pf = min(pages, _pages_fetched(n_idx, pages))
        if pf >= 2:
            per = RANDOM_PAGE - (RANDOM_PAGE - SEQ_PAGE) * math.sqrt(pf / max(pages, 1))
        else:
            per = RANDOM_PAGE
        heap = pf * per + n_idx * (CPU_TUPLE + CPU_OP * (len(matched) + len(rest)))
        cond = self._cond_text(matched)
        bis = Plan(f"Bitmap Index Scan on {ix.name}", 0.0, idx_total, n_idx, 0, [f"Index Cond: {cond}"])
        p = Plan(f"Bitmap Heap Scan on {label}", idx_total, idx_total + heap, rows, width,
                 [f"Recheck Cond: {cond}"], [bis])
        if rest:
            p.details.append(f"Filter: {self.dps(E.make_and(rest))}")
        p.extra_actual = [f"Heap Blocks: exact={int(min(pf, pages))}"]
        p.kind = "bitmap"
        p.rel = rel
        p.quals = matched + rest
        bis.kind = "bitmapindex"
        bis.rel = rel
        bis.quals = matched
        return p

    def _cond_text(self, matched):
        if not matched:
            return ""
        if len(matched) == 1:
            return self.dps(matched[0])
        return "(" + " AND ".join(self.dps(c) for c in matched) + ")"

    def _btree_match(self, c, el, rel, rte):
        target = None
        if el.col is not None:
            idx = rel.col_index(el.col)
            target = ("var", rte.slots[idx])
        if isinstance(c, E.NullTest):
            return target is not None and _is_var(c.arg, target[1])
        if isinstance(c, E.ScalarArrayOp) and c.op == "=" and c.use_or:
            left = c.args[0]
            return target is not None and _is_var(left, target[1]) and not E.vars_of(c.args[1], 0)
        if not isinstance(c, E.Op) or len(c.args) != 2 or c.op not in ("=", "<", ">", "<=", ">="):
            if isinstance(c, E.Op) and c.op == "~~" and target is not None and _is_var(c.args[0], target[1]):
                return False
            return False
        a, b = c.args
        for x, y in ((a, b), (b, a)):
            if E.vars_of(y, 0):
                continue
            if target is not None:
                if _is_var(x, target[1]):
                    return True
            elif el.expr is not None:
                bound = self._bind_index_expr(el, rel, rte)
                if bound is not None and x.key() == bound.key():
                    return True
        return False

    def _bind_index_expr(self, el, rel, rte):
        from .analyze import Scope, RTE as R, NSItem
        an = Analyzer(self.session, self.db)
        scope = Scope(None, self.session, self.db)
        r = R("rel", rte.name, rel.colnames, rel.coltypes, list(rte.slots[:len(rel.columns)]))
        scope.ns.append(NSItem(r))
        try:
            return an.expr(el.expr, scope)
        except PgError:
            return None

    def _gin_match(self, c, ix, rel, rte):
        if not isinstance(c, E.Op) or c.op not in ("@>", "?", "?|", "?&", "&&", "<@"):
            return False
        el = ix.elems[0]
        if el.col is None:
            return False
        return _is_var(c.args[0], rte.slots[rel.col_index(el.col)]) and not E.vars_of(c.args[1], 0)

    def _pred_implied(self, ix, quals, rel, rte):
        from .analyze import Scope, RTE as R, NSItem
        an = Analyzer(self.session, self.db)
        scope = Scope(None, self.session, self.db)
        r = R("rel", rte.name, rel.colnames, rel.coltypes, list(rte.slots[:len(rel.columns)]))
        scope.ns.append(NSItem(r))
        try:
            pred = an.expr(ix.pred, scope)
        except PgError:
            return False
        keys = {c.key(): c for c in quals}
        pc = E.conjuncts(pred)
        if not all(p.key() in keys for p in pc):
            return False
        return [keys[p.key()] for p in pc]

    def _index_order(self, ix, rel, rte, q, eq_prefix=()):
        """Does the index give the ORDER BY order (forward / backward)? Only used with LIMIT. Leading index
        columns fixed by an equality don't count."""
        if not q.order or q.limit is None or q.grouped or q.windows or q.distinct:
            return False
        if self._count_rels(q.from_) != 1:
            return False
        keys = []
        for i, desc, nf, _ in q.order:
            t = q.targets[i]
            if not isinstance(t, E.Var) or t.level != 0:
                return False
            keys.append((t.index, desc))
        cols = []
        for el in ix.elems:
            if el.col is None:
                return False
            slot = rte.slots[rel.col_index(el.col)]
            if slot in eq_prefix and not cols:
                continue
            cols.append((slot, el.desc))
        if not keys or len(keys) > len(cols):
            return False
        fwd = all(k == c for k, c in zip(keys, cols))
        bwd = all(k[0] == c[0] and k[1] != c[1] for k, c in zip(keys, cols))
        return "forward" if fwd else ("backward" if bwd else False)

    def _correlation(self, rs, rel, ix):
        el = ix.elems[0]
        if el.col is None or not rel.rows:
            return 0.0
        i = rel.col_index(el.col)
        vals = [r[i] for r in rel.rows if r[i] is not None]
        if len(vals) < 2:
            return 1.0
        step = max(1, -(-len(vals) // 30000))
        sample = vals[::step]
        ranks = sorted(range(len(sample)), key=lambda k: T.sort_key(sample[k]))
        pos = [0] * len(sample)
        for r, k in enumerate(ranks):
            pos[k] = r
        n = len(sample)
        mean = (n - 1) / 2
        num = sum((k - mean) * (pos[k] - mean) for k in range(n))
        den = sum((k - mean) ** 2 for k in range(n))
        return num / den if den else 1.0

    def param_index_scan(self, node, equi, outer_slots, loops=1.0):
        rel = node.rel
        if not isinstance(rel, Table):
            return None
        rte = node.rte
        rs = RelStats(rel)
        for ix in self.db.indexes_of(rel):
            el = ix.elems[0]
            if el.col is None or ix.method != "btree" or ix.pred is not None:
                continue
            slot = rte.slots[rel.col_index(el.col)]
            for c in equi:
                a, b = c.args
                for x, y in ((a, b), (b, a)):
                    if _is_var(x, slot) and {v.index for v in E.vars_of(y, 0)} <= outer_slots:
                        nd = rs.ndistinct(el.col) or 200.0
                        rows = max(1.0, rs.tuples / nd)
                        label = rel.name if rte.name == rel.name else f"{rel.name} {rte.name}"
                        descent = self._descent(ix, rel, rs)
                        ipages = self._index_size(ix, rel, rs)[0]
                        npages = max(1, math.ceil(rows * ipages / max(rs.tuples, 1)))
                        ip = _pages_fetched(npages * loops, ipages) / loops
                        hp = _pages_fetched(rows * loops, rs.pages) / loops
                        total = descent + ip * RANDOM_PAGE + rows * (CPU_INDEX_TUPLE + CPU_OP) + \
                            hp * RANDOM_PAGE + rows * CPU_TUPLE
                        for sl in rte.slots[:len(rel.columns)]:
                            self._var_src.setdefault(sl, (rs, rel.colnames[rte.slots.index(sl)]))
                        p = Plan(f"Index Scan using {ix.name} on {label}", descent, total, rows, 32,
                                 [f"Index Cond: ({self.dp(x)} = {self.dp(y)})"])
                        p.kind = "index"
                        p.rel = rel
                        return p
        return None

    # ------------------------------------------------------------ selectivity
    def _sel_list(self, quals, rs=None, rte=None):
        """clauselist_selectivity: a > lo AND a < hi on one column is a range (s_lo + s_hi - 1)."""
        s = 1.0
        ranges = {}
        for c in quals:
            if isinstance(c, E.Op) and c.op in ("<", "<=", ">", ">=") and len(c.args) == 2:
                a, b = c.args
                col = self._col_of(a) if self._col_of(a) else None
                op = c.op
                if col is None and self._col_of(b):
                    col, op = self._col_of(b), _flip(c.op)
                if col is not None:
                    key = (id(col[0]), col[1])
                    side = "lo" if op in (">", ">=") else "hi"
                    r = ranges.setdefault(key, {})
                    if side not in r:
                        r[side] = self._sel(c, rs, rte)
                        continue
            s *= self._sel(c, rs, rte)
        for r in ranges.values():
            if "lo" in r and "hi" in r:
                s *= max(1e-4, r["lo"] + r["hi"] - 1.0)
            else:
                s *= r.get("lo", 1.0) * r.get("hi", 1.0)
        return s

    def _sel(self, c, rs=None, rte=None):
        if isinstance(c, E.Bool):
            if c.op == "and":
                return self._sel_list(c.args, rs, rte)
            if c.op == "or":
                p = 0.0
                for a in c.args:
                    s = self._sel(a, rs, rte)
                    p = p + s - p * s
                return p
            return 1.0 - self._sel(c.args[0], rs, rte)
        if isinstance(c, E.NullTest):
            col = self._col_of(c.arg)
            if col:
                rs2, name = col
                cs = rs2.col(name)
                nf = cs["null_frac"] if cs else 0.005
                return (1 - nf) if c.negated else nf
            return 0.005 if not c.negated else 0.995
        if isinstance(c, E.ScalarArrayOp):
            n = len(c.args[1].value) if isinstance(c.args[1], E.Const) and c.args[1].value else 3
            col = self._col_of(c.args[0])
            each = self._eq_sel(col, None)
            return min(1.0, each * n)
        if isinstance(c, E.Op) and len(c.args) == 2:
            a, b = c.args
            col = self._col_of(a) or self._col_of(b)
            val = _const_val(b) if self._col_of(a) else _const_val(a)
            if c.op == "=":
                return self._eq_sel(col, val)
            if c.op == "<>":
                return 1.0 - self._eq_sel(col, val)
            if c.op in ("<", "<=", ">", ">="):
                return self._range_sel(col, c.op if self._col_of(a) else _flip(c.op), val)
            if c.op in ("~~", "~~*"):
                return DEFAULT_MATCH_SEL * 10
            return DEFAULT_EQ_SEL
        if isinstance(c, E.SubPlan):
            return 0.5
        return DEFAULT_INEQ_SEL

    def _col_of(self, e):
        while isinstance(e, E.Cast):
            e = e.arg
        if isinstance(e, E.Var) and e.level == 0:
            return getattr(self, "_var_src", {}).get(e.index)
        return None

    def _eq_sel(self, col, val):
        if col is None:
            return DEFAULT_EQ_SEL
        rs, name = col
        cs = rs.col(name)
        if cs is None:
            return DEFAULT_EQ_SEL
        if val is not None:
            hk = T.hash_key(val)
            for k, f in cs["mcv"]:
                if k == hk:
                    return f
        nd = rs.ndistinct(name) or 200.0
        mcv_total = sum(f for _, f in cs["mcv"])
        rest_nd = max(1.0, nd - len(cs["mcv"]))
        return max(0.0, (1.0 - cs["null_frac"] - mcv_total) / rest_nd)

    def _range_sel(self, col, op, val):
        if col is None or val is None:
            return DEFAULT_INEQ_SEL
        rs, name = col
        cs = rs.col(name)
        if cs is None or not cs.get("sorted"):
            return DEFAULT_INEQ_SEL
        hist = cs["sorted"]
        frac = _hist_frac(hist, val)
        if frac is None:
            return DEFAULT_INEQ_SEL
        nonnull = 1.0 - cs["null_frac"]
        if op in ("<", "<="):
            return max(0.0001, frac * nonnull)
        return max(0.0001, (1 - frac) * nonnull)

    # ------------------------------------------------------------ subplans (subselect.c)
    def premake(self, q):
        """PostgreSQL's order for one query level: pull_up_sublinks first decides which EXISTS / NOT EXISTS / IN
        at the top of WHERE become semi / anti joins; then the SubPlans are made (and numbered) for the target
        list, the pulled-up subqueries' own quals, the rest of WHERE, and HAVING."""
        if id(q) in self.pulls:
            return
        if q.kind != "select":
            self.pulls[id(q)] = ([], [])
            return
        rest, specs = [], []
        for c in E.conjuncts(q.where):
            spec = self._pullable(c, q) if q.from_ is not None else None
            if spec is None:
                rest.append(c)
            else:
                specs.append(spec)
        self.pulls[id(q)] = (rest, specs)
        self._alias_from(q.from_)
        for spec in specs:
            self._alias_from(spec["q2"].from_)
        for t in q.targets:
            self._premake_expr(t)
        for spec in specs:
            spec["planner"] = self.child()
            spec["planner"].premake(spec["q2"])
        for c in rest:
            self._premake_expr(c)
        if q.having is not None:
            self._premake_expr(q.having)

    def _premake_expr(self, e):
        for n in e.walk():
            if isinstance(n, E.SubPlan):
                self.make_subplan(n)

    def make_subplan(self, n):
        """Plan a sublink as an InitPlan (uncorrelated EXISTS / scalar / ARRAY), a hashed SubPlan (uncorrelated
        IN / NOT IN) or a SubPlan run per row (correlated; a correlated EXISTS also gets a hashed alternative,
        chosen when it is cheaper for the expected number of calls)."""
        info = self.made.get(id(n))
        if info is not None:
            return info
        sub = self.child()
        q2 = n.query
        plan = sub.plan_query(q2)
        nparams = len({(v.level, v.index) for v in n.outer_refs})
        self.ids[0] += 1
        pid = self.ids[0]
        if not n.correlated and n.sublink in ("exists", "expr", "array", "multiexpr"):
            k = self.ids[1]
            nret = q2.ntargets if n.sublink == "multiexpr" else 1
            self.ids[1] += nret
            info = {"kind": "init", "plan": plan, "id": pid,
                    "label": f"InitPlan {pid} (returns {','.join(f'${k + i}' for i in range(nret))})"}
            n.label = f"${k}"
        elif not n.correlated and n.sublink in ("any", "all") and n.test is not None and \
                plan.rows * (plan.width + 24) < 4 * 1024 * 1024:
            info = {"kind": "sub", "plan": plan, "id": pid, "label": f"SubPlan {pid}",
                    "startup": plan.total + CPU_OP * plan.rows, "per": 0.0}
            n.label = f"hashed SubPlan {pid}"
        else:
            self.ids[1] += nparams
            run = plan.total - plan.startup
            if n.sublink == "exists":
                per = run / _clamp_rows(plan.rows)
            elif n.sublink in ("any", "all"):
                per = 0.5 * run + 0.5 * plan.rows * CPU_OP
            else:
                per = run
            info = {"kind": "sub", "plan": plan, "id": pid, "label": f"SubPlan {pid}", "startup": 0.0,
                    "per": per + plan.startup}
            n.label = f"SubPlan {pid}"
            if n.sublink == "exists":
                alt = self._hashed_exists(n)
                if alt is not None:
                    self.ids[0] += 1
                    info["alt"] = {"plan": alt, "id": self.ids[0], "label": f"SubPlan {self.ids[0]}",
                                   "startup": alt.total + CPU_OP * alt.rows, "per": CPU_OP}
                    info["use"] = info
        self.made[id(n)] = info
        return info

    def _hashed_exists(self, n):
        """convert_EXISTS_to_ANY: a correlated EXISTS whose correlation is `inner = outer` equalities can run as
        a hashed IN over the subquery without them."""
        spec = self._exists_split(n)
        if spec is None or not spec["pairs"] or spec["filters"]:
            return None
        q2 = spec["q2"]
        q2.targets = [ie for _, ie in spec["pairs"]]
        q2.ntargets = len(q2.targets)
        return self.child().plan_query(q2)

    def decide_alternatives(self, exprs, n_exec):
        """fix_alternative_subplan: for each correlated EXISTS with a hashed alternative, the cheaper for n_exec
        calls (ties go to the later, hashed one)."""
        for e in exprs:
            if e is None:
                continue
            for n in e.walk():
                if not isinstance(n, E.SubPlan):
                    continue
                info = self.make_subplan(n)
                alt = info.get("alt")
                if alt is None:
                    continue
                if alt["startup"] + n_exec * alt["per"] <= info["startup"] + n_exec * info["per"]:
                    info["use"] = alt
                    n.label = f"hashed SubPlan {alt['id']}"
                else:
                    info["use"] = info
                    n.label = info["label"]

    def attach_expr_subplans(self, p, e, n_cost, n_exec):
        """Hang the SubPlans of an expression evaluated at plan node p under it, charging their cost for
        n_cost evaluations; an AlternativeSubPlan is costed by its first plan but runs the cheaper one for
        n_exec calls (setrefs.c)."""
        if e is None:
            return
        for n in e.walk():
            if not isinstance(n, E.SubPlan):
                continue
            info = self.make_subplan(n)
            if info["kind"] == "init":
                if not any(x[2] == info["id"] for x in self.initplan_list):
                    self.initplan_list.append((info["label"], info["plan"], info["id"]))
                continue
            p.startup += info["startup"]
            p.total += info["startup"] + n_cost * info["per"]
            if "alt" in info and "decided" not in info:
                self.decide_alternatives([n], n_exec)
            use = info.get("use", info)
            p.subplans.append((use["label"], use["plan"]))

    # ------------------------------------------------------------ sublinks pulled up into joins (prepjointree.c)
    def _pullable(self, c, q):
        neg = False
        n = c
        if isinstance(n, E.Bool) and n.op == "not" and len(n.args) == 1 and isinstance(n.args[0], E.SubPlan):
            n, neg = n.args[0], True
        if not isinstance(n, E.SubPlan):
            return None
        if n.sublink == "exists" and n.correlated:
            spec = self._exists_split(n)
            if spec is None or not spec["pairs"]:
                return None
        elif n.sublink == "any" and not neg and not n.correlated and not n.negated_in and n.name == "=" and \
                n.test is not None and not isinstance(n.test, E.RowExpr) and E.vars_of(n.test, 0):
            q2 = _copy_query(n.query)
            spec = {"q2": q2, "pairs": [(n.test, q2.targets[0])], "filters": []}
        else:
            return None
        spec["kind"] = "anti" if neg else "semi"
        spec["sublink"] = n
        return spec

    def _exists_split(self, n):
        """simplify_EXISTS_query + the correlation quals: {q2 (local quals only), pairs [(outer, inner)],
        filters [other correlated quals]} or None."""
        q2 = n.query
        if q2.kind != "select" or q2.from_ is None or q2.grouped or q2.aggs or q2.windows or q2.srfs or \
                q2.having is not None or q2.offset is not None:
            return None
        for t in q2.targets:
            if E.vars_of(t, 1):
                return None
        local, pairs, filters = [], [], []
        for c in E.conjuncts(q2.where):
            up = [v for v in E.vars_of(c, 1)]
            if not up:
                local.append(c)
                continue
            if any(v.level != 1 for v in up):
                return None
            if isinstance(c, E.Op) and c.op == "=" and len(c.args) == 2:
                a, b = c.args
                for x, y in ((a, b), (b, a)):
                    if isinstance(x, E.Var) and x.level == 1 and E.vars_of(y, 0) and not E.vars_of(y, 1):
                        pairs.append((E.Var(0, x.index, x.type, x.rte, x.col), y))
                        break
                else:
                    filters.append(c)
                continue
            filters.append(c)
        q3 = _copy_query(q2)
        q3.where = E.make_and(local) if local else None
        q3.order, q3.limit, q3.distinct = [], None, None
        keys = [ie for _, ie in pairs]
        q3.targets = keys + [v for f in filters for v in E.vars_of(f, 0)]
        q3.ntargets = len(keys) or 1
        if not q3.targets:
            q3.targets = list(q2.targets[:1])
        return {"q2": q3, "pairs": pairs, "filters": filters}

    def plan_semi(self, L, q, spec, conj):
        """A semi / anti join of the plan so far with a pulled-up subquery: Hash Semi / Anti Join, a Nested
        Loop over a parameterized index scan, the subquery made unique (HashAggregate) and joined, or a plain
        join when the subquery is unique on the join key anyway -- whichever costs least."""
        kind, sub, q2, pairs = spec["kind"], spec["planner"], spec["q2"], spec["pairs"]
        filters = spec["filters"]
        # equivalence classes: outer key = constant gives inner key = constant (no join clause left)
        consts = {}
        for c in conj:
            if isinstance(c, E.Op) and c.op == "=" and len(c.args) == 2:
                a, b = c.args
                if isinstance(a, E.Var) and a.level == 0 and not E.vars_of(b, 0) and not _has_subplan(b):
                    consts[a.index] = (c, b)
                elif isinstance(b, E.Var) and b.level == 0 and not E.vars_of(a, 0) and not _has_subplan(a):
                    consts[b.index] = (c, a)
        fixed = [(oe, ie) for oe, ie in pairs if isinstance(oe, E.Var) and oe.index in consts]
        if fixed and len(fixed) == len(pairs) and not filters:
            q3 = _copy_query(q2)
            extra = [E.Op("=", [ie, consts[oe.index][1]], T.BOOL, consts[oe.index][0].impl) for oe, ie in fixed]
            q3.where = E.make_and(E.conjuncts(q3.where) + extra)
            R = sub.plan_query(q3, attach_to=self)
            return self._nestloop_semi(L, R, kind, [], 1.0, max(1.0, R.rows), indexed=False,
                                       rows=max(1.0, L.rows if kind == "semi" else 1.0))
        R = sub.plan_query(q2, attach_to=self)
        jsel, ssel = 1.0, 1.0
        for oe, ie in pairs:
            jsel *= self._eqjoinsel(oe, ie, self, sub)
            ssel *= self._eqjoinsel(oe, ie, self, sub, semi=True)
        fsel = DEFAULT_INEQ_SEL ** len(filters)
        ssel *= fsel
        avgmatch = max(1.0, jsel * R.rows / ssel) if ssel > 0 else 1.0
        rows = max(1.0, L.rows * ssel) if kind == "semi" else max(1.0, L.rows * (1.0 - ssel))
        cond = self._pairs_text(pairs)
        cands = []
        word = "Semi" if kind == "semi" else "Anti"
        # Hash Semi / Anti Join
        if self.enabled("hashjoin"):
            st, tot = self.hash_cost(L, R, pairs, sub, filters, kind=kind, frac=ssel, match=avgmatch, rows=rows)
            h = Plan("Hash", R.total, R.total, R.rows, R.width, children=[R])
            h.extra_actual = ["Buckets: 1024  Batches: 1  Memory Usage: 9kB"]
            p = Plan(f"Hash {word} Join", st, tot, rows, L.width, [f"Hash Cond: {cond}"], [L, h])
            if filters:
                p.details.append(f"Join Filter: {self.dp(E.make_and(filters))}")
            cands.append(p)
        # Nested Loop Semi / Anti Join over a parameterized index scan of the subquery's table
        if self.enabled("nestloop") and len(pairs) == 1 and _single_rel(q2.from_):
            oe, ie = pairs[0]
            q3 = _copy_query(q2)
            outer_ref = _as_outer(oe)
            q3.where = E.make_and(E.conjuncts(q3.where) + [E.Op("=", [ie, outer_ref], T.BOOL, None)])
            ps = sub.child()
            ps.loop_count = max(1.0, L.rows)
            Ri = ps.plan_query(q3, attach_to=self)
            if Ri.kind in ("index", "bitmap"):
                cands.append(self._nestloop_semi(L, Ri, kind, filters, ssel, avgmatch, indexed=True, rows=rows))
        if kind == "semi":
            uniq = len(pairs) == 1 and _single_rel(q2.from_) and sub._unique_key(pairs[0][1])
            if uniq:
                # the subquery can't produce a key twice: an ordinary inner join (reduce_unique_semijoins)
                rows_j = max(1.0, L.rows * R.rows * jsel)
                cands.append(self._hash_join_plan(L, R, pairs, sub, filters, "Hash Join", rows_j, jsel,
                                                  inner_unique=True, swap=False))
                if self._unique_key(pairs[0][0]):
                    cands.append(self._hash_join_plan(L, R, pairs, sub, filters, "Hash Join", rows_j, jsel,
                                                      inner_unique=True, swap=True))
            else:
                # the subquery's keys made unique, then an inner join (create_unique_path)
                keys = [ie for _, ie in pairs]
                groups = sub._ngroups(keys, R.rows)
                st = R.total + R.rows * CPU_OP * len(keys)
                U = Plan("HashAggregate", st, st + groups * CPU_TUPLE, groups, R.width,
                         [f"Group Key: {', '.join(sub.dpq(k) for k in keys)}"], [R])
                U.extra_actual = ["Batches: 1  Memory Usage: 24kB"]
                U.unique_ified = True
                if self.enabled("hashjoin"):
                    st2, tot2 = self.hash_cost(L, U, pairs, sub, filters, kind="inner", frac=ssel,
                                               match=avgmatch, rows=rows, inner_unique=True, unique_ified=True)
                    h = Plan("Hash", U.total, U.total, U.rows, U.width, children=[U])
                    h.extra_actual = ["Buckets: 1024  Batches: 1  Memory Usage: 9kB"]
                    p = Plan("Hash Join", st2, tot2, rows, L.width, [f"Hash Cond: {cond}"], [L, h])
                    cands.append(p)
        best = min(cands, key=lambda c: (c.total, c.startup))
        best.rows = rows if best.name != "Hash Join" or not getattr(best, "reduced", False) else best.rows
        return best

    def _nestloop_semi(self, L, R, kind, filters, frac, match, indexed, rows):
        """final_cost_nestloop for semi / anti joins: each outer row stops at its first match."""
        m = _rint(L.rows * frac)
        unmatched = L.rows - m
        scan_frac = 2.0 / (match + 1.0)
        inner_run = R.total - R.startup
        startup = L.startup + R.startup
        run = (L.total - L.startup) + max(0.0, L.rows - 1) * R.startup
        if indexed:
            run += inner_run * scan_frac
            if m > 1:
                run += (m - 1) * inner_run * scan_frac
            run += unmatched * inner_run / _clamp_rows(R.rows)
            ntuples = m * R.rows * scan_frac
        else:
            run += inner_run
            if unmatched >= 1:
                unmatched -= 1
            else:
                m -= 1
            if m > 0:
                run += m * inner_run * scan_frac
            if unmatched > 0:
                run += unmatched * inner_run
            ntuples = m * R.rows * scan_frac + unmatched * R.rows
        run += (CPU_TUPLE + CPU_OP * sum(_nops(c) for c in filters)) * ntuples
        word = "Semi" if kind == "semi" else "Anti"
        p = Plan(f"Nested Loop {word} Join", startup, startup + run, rows, L.width, children=[L, R])
        if filters:
            p.details.append(f"Join Filter: {self.dp(E.make_and(filters))}")
        return p

    def _hash_join_plan(self, L, R, pairs, sub, filters, name, rows, jsel, inner_unique, swap):
        if swap:     # the subquery side probes, the outer side is hashed
            st, tot = self.hash_cost(R, L, [(ie, oe) for oe, ie in pairs], self, filters, kind="inner", frac=jsel,
                                     match=L.rows, rows=rows, inner_unique=True)
            cond = self._pairs_text([(ie, oe) for oe, ie in pairs])
            probe, build = R, L
        else:
            st, tot = self.hash_cost(L, R, pairs, sub, filters, kind="inner", frac=jsel, match=R.rows, rows=rows,
                                     inner_unique=inner_unique)
            cond = self._pairs_text(pairs)
            probe, build = L, R
        h = Plan("Hash", build.total, build.total, build.rows, build.width, children=[build])
        h.extra_actual = ["Buckets: 1024  Batches: 1  Memory Usage: 9kB"]
        p = Plan(name, st, tot, rows, L.width + R.width, [f"Hash Cond: {cond}"], [probe, h])
        p.reduced = True
        if filters:
            p.details.append(f"Join Filter: {self.dp(E.make_and(filters))}")
        return p

    def _pairs_text(self, pairs):
        parts = [f"({_dq(oe)} = {_dq(ie)})" for oe, ie in pairs]
        return parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"

    def dpq(self, e):
        return e.deparse(E.DeparseCtx(qualify=True, planned=True))

    def _unique_key(self, e):
        """Is e a column of a base relation with a unique index on exactly it (innerrel_is_unique)?"""
        info = self._vstats(e)
        if info is None:
            return False
        return _unique_col(info[0].rel, info[1], self.db)

    def _push_quals(self, q2, node, quals):
        """set_subquery_pathlist / subquery_push_qual: quals on a subquery's outputs move into it (into HAVING
        when they test an aggregate). -> (subquery with them, quals kept outside) or None."""
        if q2.kind != "select" or q2.limit is not None or q2.offset is not None or q2.windows or q2.srfs or \
                isinstance(q2.distinct, list):
            return None
        slots = {sl: k for k, sl in enumerate(node.rte.slots[:q2.ntargets])}
        q3 = _copy_query(q2)
        where, having, keep = list(E.conjuncts(q2.where)), list(E.conjuncts(q2.having)), []
        for c in quals:
            ok = all(type(n).__name__ in _PUSHABLE for n in c.walk()) and \
                all(v.level == 0 and v.index in slots for v in E.vars_of(c, 0)) and \
                not any(isinstance(n, E.Var) and n.level > 0 for n in c.walk())
            if not ok:
                keep.append(c)
                continue
            t = _subst(c, lambda v: q2.targets[slots[v.index]])
            if q2.grouped:
                nk = len(q2.group_keys)
                agg = any(isinstance(n, E.Aggref) or (isinstance(n, E.Var) and n.level == 0 and n.index >= nk)
                          for n in t.walk())
                if agg:
                    having.append(t)
                else:
                    where.append(_subst(t, lambda v: q2.group_keys[v.index] if v.index < nk else v))
            else:
                where.append(t)
        if len(keep) == len(quals):
            return None
        q3.where = E.make_and(where) if where else None
        q3.having = E.make_and(having) if having else None
        return q3, keep

    def _inline_cte(self, node):
        """PostgreSQL 12+ inlines a non-recursive, side-effect-free CTE referenced once (unless MATERIALIZED)."""
        cte = node.cte
        if cte.recursive or cte.modifying or cte.query is None or not isinstance(cte.query, Query):
            return False
        mat = getattr(cte.node, "materialized", None)
        return cte.refcount <= 1 and mat is not True and cte.query.kind in ("select", "values", "setop")

    # ------------------------------------------------------------ hash joins (costsize.c final_cost_hashjoin)
    def hash_cost(self, outer, inner, pairs, inner_planner, filters, kind, frac=1.0, match=1.0, rows=None,
                  inner_unique=False, unique_ified=False):
        """-> (startup, total) of hashing `inner` and probing it with `outer`. pairs: [(outer expr, inner expr)]."""
        n = len(pairs)
        nb = _nbuckets(inner.rows)
        if unique_ified or getattr(inner, "unique_ified", False):
            bs = 1.0 / nb
        else:
            bs = min(inner_planner._bucketsize(ie, nb) for _, ie in pairs)
        startup = inner.total + (CPU_OP * n + CPU_TUPLE) * inner.rows + outer.startup
        run = (outer.total - outer.startup) + CPU_OP * n * outer.rows
        if kind in ("semi", "anti") or inner_unique:
            m = _rint(outer.rows * frac)
            scan_frac = 2.0 / (match + 1.0)
            run += CPU_OP * n * m * _clamp_rows(inner.rows * bs * scan_frac) * 0.5
            run += CPU_OP * n * (outer.rows - m) * _clamp_rows(inner.rows / nb) * 0.05
            tuples = outer.rows - m if kind == "anti" else m
        else:
            run += CPU_OP * n * outer.rows * _clamp_rows(inner.rows * bs) * 0.5
            tuples = rows if rows is not None else outer.rows
        run += (CPU_TUPLE + CPU_OP * sum(_nops(c) for c in filters)) * tuples
        return startup, startup + run

    def _vstats(self, e):
        """(RelStats, column, restricted rows of its relation) for a plain column of this level, else None."""
        if isinstance(e, E.Var) and e.level == 0:
            src = self._var_src.get(e.index)
            if src is not None:
                return src[0], src[1], self._slot_rows.get(e.index, src[0].tuples)
        return None

    def _ndistinct(self, info):
        """get_variable_numdistinct -> (nd, isdefault)."""
        rs, col, _ = info
        nd = rs.ndistinct(col)
        if nd is not None:
            return min(nd, rs.tuples) if rs.tuples > 0 else nd, False
        if _unique_col(rs.rel, col, self.db):
            return rs.tuples, False
        t = _col_type(rs.rel, col)
        if t is not None and t.name == "bool":
            return 2.0, False
        return 200.0, True

    def _bucketsize(self, e, nb):
        """estimate_hash_bucket_stats: the fraction of the inner relation in the probed bucket."""
        info = self._vstats(e)
        if info is None:
            return max(0.1, 1.0 / nb)
        nd, isdefault = self._ndistinct(info)
        if isdefault:
            return max(0.1, 1.0 / nb)
        rs, col, rel_rows = info
        cs = rs.col(col)
        nullfrac = cs["null_frac"] if cs else 0.0
        avgfreq = (1.0 - nullfrac) / nd if nd > 0 else 0.0
        if rs.tuples > 0:
            nd = _clamp_rows(nd * rel_rows / rs.tuples)
        est = 1.0 / nb if nd > nb else 1.0 / nd
        mcvf = max((f for _, f in cs["mcv"]), default=0.0) if cs and cs.get("mcv") else 0.0
        if avgfreq > 0 and mcvf > avgfreq:
            est *= mcvf / avgfreq
        return min(1.0, max(1e-6, est))

    def _eqjoinsel(self, e1, e2, p1, p2, semi=False):
        """eqjoinsel_inner / eqjoinsel_semi (selfuncs.c), with MCV lists when both sides have them."""
        i1, i2 = p1._vstats(e1), p2._vstats(e2)
        if i1 is None or i2 is None:
            return 0.5 if semi else DEFAULT_EQ_SEL
        nd1, d1 = p1._ndistinct(i1)
        nd2, d2 = p2._ndistinct(i2)
        cs1, cs2 = i1[0].col(i1[1]), i2[0].col(i2[1])
        nf1 = cs1["null_frac"] if cs1 else 0.0
        nf2 = cs2["null_frac"] if cs2 else 0.0
        if semi and nd2 >= i2[2]:
            nd2, d2 = i2[2], False
        m1 = cs1.get("mcv") if cs1 else None
        m2 = cs2.get("mcv") if cs2 else None
        if m1 and m2:
            f2 = dict(m2)
            matched1 = matched2 = 0.0
            prod = 0.0
            nmatch = 0
            for k, f in m1:
                if k in f2:
                    prod += f * f2[k]
                    matched1 += f
                    matched2 += f2[k]
                    nmatch += 1
            tot1, tot2 = sum(f for _, f in m1), sum(f for _, f in m2)
            if semi:
                n1, n2 = nd1 - nmatch, nd2 - nmatch
                uncertain = 1.0 if (n1 <= n2 or n2 < 0) else n2 / n1
                return min(1.0, matched1 + uncertain * max(0.0, 1.0 - nf1 - matched1))
            other1, other2 = 1.0 - nf1 - tot1, 1.0 - nf2 - tot2
            un1, un2 = tot1 - matched1, tot2 - matched2
            s1 = prod
            if nd2 > len(m2):
                s1 += un1 * other2 / (nd2 - len(m2))
            if nd2 > nmatch:
                s1 += other1 * (other2 + un2) / (nd2 - nmatch)
            s2 = prod
            if nd1 > len(m1):
                s2 += un2 * other1 / (nd1 - len(m1))
            if nd1 > nmatch:
                s2 += other2 * (other1 + un1) / (nd1 - nmatch)
            return max(0.0, min(s1, s2))
        if semi:
            if not d1 and not d2:
                return (1.0 - nf1) if nd1 <= nd2 else (nd2 / nd1) * (1.0 - nf1)
            return 0.5 * (1.0 - nf1)
        return (1.0 - nf1) * (1.0 - nf2) / max(nd1, nd2, 1.0)

    # ------------------------------------------------------------ set operations
    def plan_setop(self, q):
        op, all_, left, right = q.setop
        if q.recursive_ref is not None:
            l = self.child().plan_query(left)
            r = self.child().plan_query(right)
            p = Plan("Recursive Union", 0.0, l.total + r.total * 10, l.rows + r.rows * 10, l.width, children=[l, r])
            return self.finish(q, p)
        l = self.child().plan_query(left)
        r = self.child().plan_query(right)
        rows = l.rows + r.rows
        app = Plan("Append", 0.0, l.total + r.total + rows * CPU_TUPLE * 0.5, rows, max(l.width, r.width),
                   children=[l, r])
        if op == "union":
            if all_:
                return self.finish(q, app)
            first = left
            while first.kind == "setop":
                first = first.setop[2]
            keys = [_dq(t) if first.kind == "select" else n
                    for t, n in zip(first.targets[:q.ntargets] if first.kind == "select" else q.names,
                                    q.names[:q.ntargets])]
            st = app.total + rows * CPU_OP * len(keys)
            p = Plan("HashAggregate", st, st + rows * CPU_TUPLE, rows, app.width, [f"Group Key: {', '.join(keys)}"],
                     [app])
            return self.finish(q, p)
        name = "HashSetOp " + ("Intersect" if op == "intersect" else "Except") + (" All" if all_ else "")
        sq1 = Plan(f"Subquery Scan on \"*SELECT* 1\"", 0.0, l.total + l.rows * CPU_TUPLE, l.rows, l.width,
                   children=[l])
        sq2 = Plan(f"Subquery Scan on \"*SELECT* 2\"", 0.0, r.total + r.rows * CPU_TUPLE, r.rows, r.width,
                   children=[r])
        app = Plan("Append", 0.0, sq1.total + sq2.total, sq1.rows + sq2.rows, l.width, children=[sq1, sq2])
        p = Plan(name, app.total, app.total + app.rows * CPU_OP, min(l.rows, r.rows), l.width, children=[app])
        return self.finish(q, p)

    # ------------------------------------------------------------ DML
    def plan_modify(self, m, node):
        from .dml import ModifyStmt
        t = m.table
        kind = {"insert": "Insert", "update": "Update", "delete": "Delete", "merge": "Merge"}[m.kind]
        if m.kind == "insert":
            if getattr(m, "source_query", None) is not None:
                child = self.plan_query(m.source_query)
            else:
                n = len(m.values or [])
                child = Plan("Result", 0.0, 0.01 * n, n, 32) if n <= 1 else \
                    Plan('Values Scan on "*VALUES*"', 0.0, n * CPU_TUPLE, n, 32)
        else:
            q = Query()
            q.from_ = getattr(m, "from_node", None) or getattr(m, "src_node", None)
            q.where = getattr(m, "where", None)
            q.targets = []
            self.qualify = False
            child = self.plan_from(q, q.from_, E.conjuncts(q.where)) if q.from_ is not None else Plan("Result", 0, 0,
                                                                                                    1, 0)
        p = Plan(f"{kind} on {t.name}", child.startup, child.total + child.rows * CPU_TUPLE, 0, 0, children=[child])
        p.rows = 0
        return p

    # ------------------------------------------------------------ EXPLAIN ANALYZE
    def analyze_query(self, q, plan):
        """Run the query and fill in actual row counts: exact for scans (their quals over the table) and for the
        top node, derived for the nodes in between."""
        from .executor import Ctx, run_query, run_from_where
        c = Ctx(self.session, self.db)
        c.outer.append(None)
        rows = run_query(q, c)
        plan.actual = (len(rows), 1)
        nfrom = None
        if q.kind == "select" and q.from_ is not None:
            nfrom = len(run_from_where(q, c))
        for p in plan.walk():
            if p.kind in ("seq", "index", "bitmap", "bitmapindex") and p.rel is not None:
                n = self._count_scan(p, c)
                p.actual = (n, 1)
                if getattr(p, "kind_only", False):
                    p.extra_actual = [f"Heap Fetches: {0 if getattr(p, 'allvis', True) else n}"]
                if p.kind == "seq" and p.quals:
                    removed = len(p.rel.rows) - n
                    p.extra_actual = [f"Rows Removed by Filter: {removed}"] + p.extra_actual
            elif p.children and p is not plan:
                p.actual = None
        # fill the others bottom-up
        def fill(p):
            for ch in p.children:
                fill(ch)
            for _, sp in p.subplans:
                fill(sp)
                if sp.actual is None:
                    sp.actual = (1, 1)
            if p.actual is not None:
                return
            if p.name.startswith(("Hash Join", "Nested Loop", "Hash Left", "Hash Right", "Hash Full", "Merge")) \
                    and nfrom is not None:
                p.actual = (nfrom, 1)
            elif p.children:
                p.actual = (p.children[0].actual[0] if p.children[0].actual else int(p.rows), 1)
            else:
                p.actual = (int(p.rows), 1)
        fill(plan)

    def _count_scan(self, p, c):
        rel = p.rel
        quals = p.quals
        if not quals:
            return len(rel.rows)
        # quals are bound against the query level's layout; find the base offset from the first Var
        base = None
        for qq in quals:
            vs = E.vars_of(qq, 0)
            if vs:
                base = min(v.index for v in vs)
                break
        # find the RTE offset: the smallest slot of this relation among the planner's var sources
        offs = [s for s, (rs, _) in self._var_src.items() if rs.rel is rel]
        b = min(offs) if offs else 0
        fns = [E.compile_at(qq, b) for qq in quals]
        n = 0
        for r in rel.rows:
            if all(f(r, c) for f in fns):
                n += 1
        return n


def _from_vars(node, need):
    if node is None:
        return
    if isinstance(node, ListNode):
        for it in node.items:
            _from_vars(it, need)
    elif isinstance(node, JoinNode):
        _from_vars(node.left, need)
        _from_vars(node.right, need)
        if node.quals is not None:
            for v in E.vars_of(node.quals, 0):
                need.add(v.index)
    elif isinstance(node, ScanFunc):
        for f in node.funcs:
            for a in f.args:
                for v in E.vars_of(a, 0):
                    need.add(v.index)


def _is_var(e, slot):
    while isinstance(e, E.Cast) and e.type.name in ("int8", "int4", "numeric", "float8", "text", "varchar"):
        e = e.arg
    return isinstance(e, E.Var) and e.level == 0 and e.index == slot


def _const_val(e):
    if isinstance(e, E.Const):
        return e.value
    if isinstance(e, E.Cast) and isinstance(e.arg, E.Const):
        try:
            return e.fn(e.arg.value, None)
        except Exception:  # noqa: BLE001
            return None
    return None


def _num(v):
    import datetime as _dt
    from decimal import Decimal
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    if isinstance(v, _dt.datetime):
        return v.replace(tzinfo=None).timestamp() if v.tzinfo is None else v.timestamp()
    if isinstance(v, _dt.date):
        return float(v.toordinal())
    return None


def _hist_frac(hist, val):
    """The fraction of values below val, interpolating inside a histogram bucket (as ineq_histogram_selectivity)."""
    import datetime as _dt
    if isinstance(val, _dt.date) and not isinstance(val, _dt.datetime) and hist and isinstance(hist[0], _dt.datetime):
        val = _dt.datetime(val.year, val.month, val.day)
    nb = len(hist) - 1
    if nb <= 0:
        return None
    try:
        k = T.sort_key(val)
        keys = [T.sort_key(h) for h in hist]
        if k < keys[0]:
            return 0.0
        if k >= keys[-1]:
            return 1.0
        i = 0
        while i < nb and not (keys[i] <= k < keys[i + 1]):
            i += 1
    except TypeError:
        return None
    lo, hi = _num(hist[i]), _num(hist[i + 1])
    x = _num(val)
    within = 0.5
    if lo is not None and hi is not None and x is not None and hi > lo:
        within = (x - lo) / (hi - lo)
    return (i + within) / nb


def _flip(op):
    return {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[op]


def _nops(c):
    return max(1, sum(1 for n in c.walk() if isinstance(n, (E.Op, E.Func, E.ScalarArrayOp))))


def _pages_fetched(tuples, pages):
    """index_pages_fetched (Mackert and Lohman), for a table that fits in the cache."""
    T_ = max(pages, 1)
    N = tuples
    if N <= 0:
        return 0.0
    pf = (2.0 * T_ * N) / (2.0 * T_ + N)
    return math.ceil(min(pf, T_))


def _rint(x):
    return float(round(x))


def _clamp_rows(x):
    """clamp_row_est"""
    return 1.0 if x <= 1.0 else float(round(x))


def _nbuckets(rows):
    """ExecChooseHashTableSize: a power of two >= the rows, at least 1024."""
    n = 1024
    while n < rows:
        n *= 2
    return float(n)


def _dq(e):
    return e.deparse(E.DeparseCtx(qualify=True, planned=True))


def _has_subplan(e):
    return any(isinstance(n, E.SubPlan) for n in e.walk())


def _copy_query(q):
    import copy
    q2 = copy.copy(q)
    q2.targets = list(q.targets)
    return q2


def _single_rel(node):
    return isinstance(node, ScanRel)


def _as_outer(e):
    """A column of this level as seen from a subquery (a parameter of the inner scan)."""
    return E.Var(e.level + 1, e.index, e.type, e.rte, e.col)


def _unique_col(rel, col, db=None):
    """A PRIMARY KEY / UNIQUE constraint or a unique index (without WHERE) on exactly this column."""
    if not isinstance(rel, Table):
        return False
    for c in getattr(rel, "constraints", None) or []:
        if c.kind in ("p", "u") and list(c.cols) == [col]:
            return True
    for ix in (db.indexes_of(rel) if db is not None else []):
        if ix.unique and ix.pred is None and len(ix.elems) == 1 and ix.elems[0].col == col:
            return True
    return False


def _col_type(rel, col):
    try:
        return rel.coltypes[rel.colnames.index(col)]
    except (AttributeError, ValueError):
        return None


_PUSHABLE = {"Var", "Const", "Op", "Bool", "Func", "Cast", "NullTest", "BoolTest", "InList", "ScalarArrayOp",
             "Coalesce", "Param", "ArrayExpr", "NullIf", "MinMax", "Distinct"}


def _subst(e, fn):
    """A copy of e with each level-0 Var replaced by fn(var) (sharing the replacements)."""
    import copy
    if isinstance(e, E.Var):
        return fn(e) if e.level == 0 else e
    c = copy.copy(e)
    for attr in ("args", "items"):
        v = getattr(c, attr, None)
        if isinstance(v, list):
            setattr(c, attr, [_subst(x, fn) for x in v])
    for attr in ("arg",):
        v = getattr(c, attr, None)
        if isinstance(v, E.BE):
            setattr(c, attr, _subst(v, fn))
    return c


def _join_quals(node, out):
    if isinstance(node, ListNode):
        for it in node.items:
            _join_quals(it, out)
    elif isinstance(node, JoinNode):
        out.extend(E.conjuncts(node.quals))
        _join_quals(node.left, out)
        _join_quals(node.right, out)


def _count_rtes(q):
    """How many entries the statement's range table has (only "more than one" matters)."""
    n = 0

    def from_(node):
        nonlocal n
        if node is None:
            return
        if isinstance(node, ListNode):
            for it in node.items:
                from_(it)
        elif isinstance(node, JoinNode):
            n += 1
            from_(node.left)
            from_(node.right)
        elif isinstance(node, ScanSub):
            n += 1 + _count_rtes(node.query)
        elif isinstance(node, ScanCte):
            n += 1 + (_count_rtes(node.cte.query) if isinstance(node.cte.query, Query) else 0)
        else:
            n += 1
    if q is None:
        return 0
    if q.kind == "setop" and q.setop is not None:
        return 2 + _count_rtes(q.setop[2]) + _count_rtes(q.setop[3])
    from_(q.from_)
    for e in list(q.targets) + [q.where, q.having]:
        if e is not None:
            for x in e.walk():
                if isinstance(x, E.SubPlan):
                    n += _count_rtes(x.query)
    return n


def _has_final(a):
    """Aggregates whose result goes through a final function (charged once per group where it is read)."""
    name = getattr(a, "name", "") or ""
    if name in ("avg", "string_agg", "array_agg", "json_agg", "jsonb_agg", "json_object_agg", "jsonb_object_agg",
                "percentile_cont", "percentile_disc", "mode") or name.startswith(("stddev", "var", "regr_", "covar",
                                                                                     "corr")):
        return True
    if name == "sum" and a.args:
        return a.args[0].type.name in ("numeric", "int8")
    return False

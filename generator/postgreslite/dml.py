"""INSERT / UPDATE / DELETE / MERGE with RETURNING and ON CONFLICT, and the constraint checks that guard them:
NOT NULL, CHECK, UNIQUE / PRIMARY KEY (and unique indexes, partial or on expressions), FOREIGN KEY with
NO ACTION / RESTRICT / CASCADE / SET NULL / SET DEFAULT. Messages and DETAIL lines are PostgreSQL's."""

from __future__ import annotations

from . import ast as A
from . import expr as E
from . import types as T
from .analyze import (Analyzer, Scope, RTE, NSItem, ScanRel, ListNode, coerce_to, figure_colname)
from .catalog import Table, View, Index
from .errors import PgError, Unsupported


class ModifyStmt:
    def __init__(self, kind, table):
        self.kind = kind
        self.table = table
        self.returning = []
        self.ret_names = []
        self.ret_types = []
        self.count = 0


def _target_table(an: Analyzer, rv: A.RangeVar, what: str):
    rel = an.db.find_rel(rv.name, rv.schema, an.session)
    if rel is None:
        shown = f"{rv.schema}.{rv.name}" if rv.schema else rv.name
        raise PgError(f'relation "{shown}" does not exist', "42P01", pos=rv.pos + 1)
    if isinstance(rel, View):
        if rel.materialized:
            raise PgError(f'cannot change materialized view "{rel.name}"', "42809")
        return rel
    if not isinstance(rel, Table):
        raise PgError(f'cannot {what} "{rv.name}"' if False else f'"{rv.name}" is not a table', "42809",
                      pos=rv.pos + 1)
    return rel


def _ctid_rel(table):
    """A stand-in for the target table whose rows carry their position (a hidden last column)."""
    fake = Table(table.name, table.schema, table.columns)
    fake.rows = [r + (i,) for i, r in enumerate(table.rows)]
    return fake


def _target_scope(an, table, rv, parent, alias_name=None):
    scope = Scope(parent, an.session, an.db)
    n = len(table.columns)
    name = rv.alias.name if rv.alias is not None else table.name
    rte = RTE("rel", name, table.colnames + ["ctid"], table.coltypes + [T.INT8], scope.alloc(n + 1),
              rv.alias is not None)
    rte.relname = table.name
    rte.rel = table
    pk = table.pkey()
    rte.pk_cols = list(pk.cols) if pk else None
    rte.hidden = {"ctid"}
    scope.ns.append(NSItem(rte))
    return scope, rte


def analyze_modify(an: Analyzer, node, parent=None):
    if isinstance(node, A.Insert):
        return analyze_insert(an, node, parent)
    if isinstance(node, A.Update):
        return analyze_update(an, node, parent)
    if isinstance(node, A.Delete):
        return analyze_delete(an, node, parent)
    if isinstance(node, A.Merge):
        return analyze_merge(an, node, parent)
    raise PgError("unexpected statement", "42601")


def _with(an, node, scope, st):
    if getattr(node, "with_", None) is not None:
        from .analyze import Query
        q = Query()
        an.with_clause(node.with_, scope, q)
        st.ctes = q.ctes
    else:
        st.ctes = []


def _returning(an, st, targets, scope):
    if not targets:
        return
    scope.kind = "returning"
    out, names = [], []
    for rt in targets:
        for e, name, _ in an.target(rt, scope):
            if e.type.name == "unknown":
                e = coerce_to(e, T.TEXT)
            out.append(e)
            names.append(name)
    st.returning = out
    st.ret_names = names
    st.ret_types = [e.type for e in out]


def _check_view(table, what):
    if isinstance(table, View):
        raise Unsupported(f"{what} on a view")


# ---------------------------------------------------------------- INSERT

def analyze_insert(an: Analyzer, node: A.Insert, parent=None):
    table = _target_table(an, node.table, "insert into")
    _check_view(table, "INSERT")
    st = ModifyStmt("insert", table)
    scope0 = Scope(parent, an.session, an.db)
    _with(an, node, scope0, st)
    cols = table.columns
    if node.cols:
        targets = []
        for c in node.cols:
            name, pos = c[0], c[1]
            i = table.col_index(name)
            if i < 0:
                raise PgError(f'column "{name}" of relation "{table.name}" does not exist', "42703", pos=pos + 1)
            if i in targets:
                raise PgError(f'column "{name}" specified more than once', "42701", pos=pos + 1)
            if len(c) > 2:
                raise Unsupported("INSERT into subscripts or fields", pos + 1)
            targets.append(i)
    else:
        targets = list(range(len(cols)))
    st.target_cols = targets
    st.overriding = node.overriding
    # the source rows
    src = node.source
    st.source_query = None
    st.values = None
    if src is None:
        st.values = [[]]
        st.target_cols = []
    elif isinstance(src, A.Values) and not src.order_by and src.limit is None and src.with_ is None:
        rows = []
        vs = Scope(scope0, an.session, an.db)
        vs.ctes = scope0.ctes
        vs.kind = "insert"
        for row in src.rows:
            if len(row) > len(targets):
                raise PgError("INSERT has more expressions than target columns", "42601",
                              pos=getattr(row[len(targets)], "pos", -1) + 1)
            if len(row) < len(targets) and not node.cols:
                pass
            elif len(row) < len(targets):
                raise PgError("INSERT has more target columns than expressions", "42601",
                              pos=node.cols[len(row)][1] + 1)
            bound = []
            for j, x in enumerate(row):
                col = cols[targets[j]]
                if isinstance(x, A.SetToDefault):
                    bound.append(None)
                    continue
                e = an.expr(x, vs)
                bound.append(assign_coerce(e, col, x))
            rows.append(bound)
        st.values = rows
        st.ncols_given = None
    else:
        an.keep_unknown = isinstance(src, A.Select)
        q = an.query(src, scope0)
        an.keep_unknown = False
        n = q.ntargets
        if n > len(targets):
            raise PgError("INSERT has more expressions than target columns", "42601",
                          pos=_first_pos(src, len(targets)))
        if n < len(targets) and node.cols:
            raise PgError("INSERT has more target columns than expressions", "42601", pos=node.cols[n][1] + 1)
        st.source_query = q
        conv = []
        for j in range(n):
            col = cols[targets[j]]
            ph = E.Var(0, j, q.types[j])
            conv.append(assign_coerce(ph, col, None, pos=_first_pos(src, j)))
        st.source_conv = conv
    # identity columns: an explicit value for GENERATED ALWAYS needs OVERRIDING SYSTEM VALUE
    for j, ci in enumerate(st.target_cols):
        col = cols[ci]
        if col.identity == "a" and node.overriding != "system":
            if st.values is not None and all(r[j] is None for r in st.values if j < len(r)):
                continue
            raise PgError(f'cannot insert a non-DEFAULT value into column "{col.name}"', "428C9",
                          detail=f'Column "{col.name}" is an identity column defined as GENERATED ALWAYS.',
                          hint="Use OVERRIDING SYSTEM VALUE to override.")
        if col.generated is not None:
            if st.values is not None and all(r[j] is None for r in st.values if j < len(r)):
                continue
            raise PgError(f'cannot insert a non-DEFAULT value into column "{col.name}"', "428C9",
                          detail=f'Column "{col.name}" is a generated column.')
    st.defaults = _bind_defaults(an, table)
    st.generated = _bind_generated(an, table)
    # ON CONFLICT
    st.conflict = None
    tscope, rte = _target_scope(an, table, node.table, scope0)
    if node.on_conflict is not None:
        st.conflict = analyze_conflict(an, node.on_conflict, table, node.table, tscope, scope0)
    _returning(an, st, node.returning, tscope)
    return st


def _first_pos(src, i):
    while isinstance(src, A.SetOp):
        src = src.left
    if isinstance(src, A.Select) and i < len(src.targets):
        return src.targets[i].pos + 1
    if isinstance(src, A.Values) and src.rows and i < len(src.rows[0]):
        return getattr(src.rows[0][i], "pos", -1) + 1
    return None


def assign_coerce(e, col, node, pos=None):
    """Assignment conversion to a column's type, with PostgreSQL's error for impossible ones."""
    t = col.type
    p = pos if pos is not None else (getattr(node, "pos", -1) + 1 if node is not None else None)
    if e.type == t.base and (t.mod is None or e.type.mod == t.mod):
        return e
    err = PgError(f'column "{col.name}" is of type {t.format()} but expression is of type {e.type.format()}',
                  "42804", pos=p, hint="You will need to rewrite or cast the expression.")
    try:
        return coerce_to(e, t, "a", pos=(p - 1) if p else None, what=err)
    except PgError as x:
        if x is err or x.code == "42846":
            raise err from None
        raise


def _bind_defaults(an, table):
    out = []
    scope = Scope(None, an.session, an.db)
    scope.kind = "default"
    for col in table.columns:
        if col.default is None:
            out.append(None)
            continue
        e = an.expr(col.default, scope)
        e = assign_coerce(e, col, col.default)
        out.append(e.compile())
    return out


def _bind_generated(an, table):
    if not any(c.generated is not None for c in table.columns):
        return None
    scope = Scope(None, an.session, an.db)
    rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
    scope.ns.append(NSItem(rte))
    scope.kind = "generated"
    out = []
    for col in table.columns:
        if col.generated is None:
            out.append(None)
            continue
        e = an.expr(col.generated, scope)
        out.append(assign_coerce(e, col, col.generated).compile())
    return out


def analyze_conflict(an, oc: A.OnConflict, table, rv, tscope, scope0):
    db = an.db
    arbiters = []
    if oc.constraint is not None:
        hit = [c for c in table.constraints if c.name == oc.constraint and c.kind in ("p", "u")]
        if not hit:
            raise PgError(f'constraint "{oc.constraint}" for table "{table.name}" does not exist', "42704")
        arbiters = [("cols", hit[0].cols, hit[0].name)]
    elif oc.cols:
        names = []
        for e in oc.cols:
            if isinstance(e, A.ColumnRef) and len(e.names) == 1:
                if table.col_index(e.names[0]) < 0:
                    raise PgError(f'column "{e.names[0]}" does not exist', "42703", pos=e.pos + 1)
                names.append(e.names[0])
            else:
                names.append(None)
        want = set(n for n in names if n)
        for c in table.constraints:
            if c.kind in ("p", "u") and set(c.cols) == want and None not in names:
                arbiters.append(("cols", c.cols, c.name))
        for idx in db.indexes_of(table):
            if idx.unique and idx.constraint is None and all(el.col for el in idx.elems) and \
                    set(el.col for el in idx.elems) == want and None not in names and idx.pred is None:
                arbiters.append(("cols", idx.cols(), idx.name))
        if not arbiters:
            raise PgError("there is no unique or exclusion constraint matching the ON CONFLICT specification",
                          "42P10")
    else:
        if oc.action == "update":
            raise PgError("ON CONFLICT DO UPDATE requires inference specification or constraint name", "42601",
                          hint="For example, ON CONFLICT (column_name).", pos=oc.pos + 1)
        arbiters = None       # any unique constraint
    res = {"action": oc.action, "arbiters": arbiters}
    if oc.action == "update":
        # SET expressions see the target row (by table name or alias) and EXCLUDED
        scope = Scope(scope0, an.session, an.db)
        n = len(table.columns)
        name = rv.alias.name if rv.alias is not None else table.name
        trte = RTE("rel", name, table.colnames, table.coltypes, scope.alloc(n), rv.alias is not None)
        trte.relname = table.name
        trte.rel = table
        erte = RTE("rel", "excluded", table.colnames, table.coltypes, scope.alloc(n))
        erte.relname = "excluded"
        scope.ns += [NSItem(trte), NSItem(erte)]
        scope.kind = "update"
        sets = _bind_sets(an, oc.sets, table, scope)
        res["sets"] = sets
        res["where"] = None
        if oc.update_where is not None:
            scope.kind = "where"
            w = an.expr(oc.update_where, scope)
            res["where"] = an.bool_expr(w, "WHERE", oc.update_where).compile()
    return res


def _bind_sets(an, sets, table, scope):
    out = []      # [(col index, fn)] ; multi-assignments expand
    seen = set()
    for sc in sets:
        cols = []
        for name, ind, cpos in sc.cols:
            i = table.col_index(name)
            if i < 0:
                raise PgError(f'column "{name}" of relation "{table.name}" does not exist', "42703",
                              pos=cpos + 1)
            if ind:
                raise Unsupported("UPDATE of array elements or fields", cpos + 1)
            if i in seen:
                raise PgError(f'multiple assignments to same column "{name}"', "42601", pos=cpos + 1)
            seen.add(i)
            col = table.columns[i]
            if col.identity == "a":
                raise PgError(f'column "{name}" can only be updated to DEFAULT', "428C9",
                              detail=f'Column "{name}" is an identity column defined as GENERATED ALWAYS.')
            if col.generated is not None:
                raise PgError(f'column "{name}" can only be updated to DEFAULT', "428C9",
                              detail=f'Column "{name}" is a generated column.')
            cols.append(i)
        if not sc.multi:
            v = sc.value
            i = cols[0]
            if isinstance(v, A.SetToDefault):
                out.append((i, "default"))
                continue
            e = an.expr(v, scope)
            out.append((i, assign_coerce(e, table.columns[i], v).compile()))
            continue
        v = sc.value
        if isinstance(v, A.RowExpr):
            if len(v.items) != len(cols):
                raise PgError("number of columns does not match number of values", "42601", pos=sc.pos + 1)
            for i, x in zip(cols, v.items):
                if isinstance(x, A.SetToDefault):
                    out.append((i, "default"))
                else:
                    e = an.expr(x, scope)
                    out.append((i, assign_coerce(e, table.columns[i], x).compile()))
        elif isinstance(v, A.SubLink):
            sp = an.expr(v, scope)
            if sp.query.ntargets != len(cols):
                raise PgError("number of columns does not match number of values", "42601", pos=sc.pos + 1)
            f = sp.compile()
            for k, i in enumerate(cols):
                ph = E.Const(None, sp.query.types[k])
                conv = assign_coerce(E.Var(0, 0, sp.query.types[k]), table.columns[i], None).compile()
                out.append((i, ("multi", f, k, conv)))
        else:
            raise Unsupported("this multiple-column assignment", sc.pos + 1)
    return out


# ---------------------------------------------------------------- UPDATE / DELETE

def analyze_update(an: Analyzer, node: A.Update, parent=None):
    table = _target_table(an, node.table, "update")
    _check_view(table, "UPDATE")
    st = ModifyStmt("update", table)
    scope0 = Scope(parent, an.session, an.db)
    _with(an, node, scope0, st)
    scope, rte = _target_scope(an, table, node.table, scope0)
    scope.ctes = scope0.ctes
    items = [ScanRel(rte, None)]
    for it in node.from_:
        items.append(an.from_item(it, scope))
    st.from_node = ListNode(items, 0, scope.width) if len(items) > 1 else items[0]
    st.width = scope.width
    st.where = None
    if node.where is not None:
        scope.kind = "where"
        st.where = an.bool_expr(an.expr(node.where, scope), "WHERE", node.where)
    scope.kind = "update"
    st.sets = _bind_sets(an, node.sets, table, scope)
    st.defaults = _bind_defaults(an, table)
    st.generated = _bind_generated(an, table)
    _returning(an, st, node.returning, scope)
    st.rte = rte
    return st


def analyze_delete(an: Analyzer, node: A.Delete, parent=None):
    table = _target_table(an, node.table, "delete from")
    _check_view(table, "DELETE")
    st = ModifyStmt("delete", table)
    scope0 = Scope(parent, an.session, an.db)
    _with(an, node, scope0, st)
    scope, rte = _target_scope(an, table, node.table, scope0)
    scope.ctes = scope0.ctes
    items = [ScanRel(rte, None)]
    for it in node.using:
        items.append(an.from_item(it, scope))
    st.from_node = ListNode(items, 0, scope.width) if len(items) > 1 else items[0]
    st.width = scope.width
    st.where = None
    if node.where is not None:
        scope.kind = "where"
        st.where = an.bool_expr(an.expr(node.where, scope), "WHERE", node.where)
    _returning(an, st, node.returning, scope)
    st.rte = rte
    return st


# ---------------------------------------------------------------- MERGE

def analyze_merge(an: Analyzer, node: A.Merge, parent=None):
    table = _target_table(an, node.target, "merge into")
    _check_view(table, "MERGE")
    st = ModifyStmt("merge", table)
    scope0 = Scope(parent, an.session, an.db)
    _with(an, node, scope0, st)
    scope, rte = _target_scope(an, table, node.target, scope0)
    scope.ctes = scope0.ctes
    src = an.from_item(node.source, scope)
    st.src_node = src
    st.width = scope.width
    scope.kind = "on"
    st.on = an.bool_expr(an.expr(node.on, scope), "ON", node.on)
    st.whens = []
    st.defaults = _bind_defaults(an, table)
    st.generated = _bind_generated(an, table)
    for w in node.whens:
        scope.kind = "merge_when"
        cond = an.bool_expr(an.expr(w.cond, scope), "WHEN", w.cond).compile() if w.cond is not None else None
        item = {"matched": w.matched, "cond": cond, "action": w.action}
        if w.action == "update":
            scope.kind = "update"
            item["sets"] = _bind_sets(an, w.sets, table, scope)
        elif w.action == "insert":
            # INSERT sees only the source row
            iscope = Scope(scope0, an.session, an.db)
            iscope.ns = [it for it in scope.ns if it.rte is not rte]
            iscope.width = scope.width
            iscope.kind = "insert"
            if w.cols:
                targets = []
                for name, _ in w.cols:
                    i = table.col_index(name)
                    if i < 0:
                        raise PgError(f'column "{name}" of relation "{table.name}" does not exist', "42703")
                    targets.append(i)
            else:
                targets = list(range(len(table.columns)))
            vals = []
            if w.values is not None:
                if len(w.values) > len(targets):
                    raise PgError("INSERT has more expressions than target columns", "42601")
                if len(w.values) < len(targets) and w.cols:
                    raise PgError("INSERT has more target columns than expressions", "42601")
                for j, x in enumerate(w.values):
                    if isinstance(x, A.SetToDefault):
                        vals.append(None)
                    else:
                        vals.append(assign_coerce(an.expr(x, iscope), table.columns[targets[j]], x).compile())
                targets = targets[:len(vals)]
            else:
                targets = []
            item["targets"] = targets
            item["values"] = vals
        st.whens.append(item)
    return st


# ---------------------------------------------------------------- running

def run_modify(st: ModifyStmt, c):
    """Apply the statement to c.db. -> RETURNING rows (st.count is the command's row count)."""
    from .executor import run_node, run_query, Conj, cte_rows
    table = c.db.find_rel(st.table.name, st.table.schema, None)
    if table is None:
        table = st.table
    st.table = table
    for cte in getattr(st, "ctes", []):
        pass
    if st.kind == "insert":
        return run_insert(st, table, c)
    if st.kind == "update":
        return run_update(st, table, c)
    if st.kind == "delete":
        return run_delete(st, table, c)
    if st.kind == "merge":
        return run_merge(st, table, c)
    raise AssertionError(st.kind)


def _default_row(st, table, c):
    row = []
    for i, col in enumerate(table.columns):
        f = st.defaults[i]
        row.append(f(None, c) if f is not None else None)
    return row


def _fill_generated(st, table, row, c):
    if st.generated is None:
        return row
    r = tuple(row)
    for i, f in enumerate(st.generated):
        if f is not None:
            row[i] = f(r, c)
    return row


def _new_rows_insert(st, table, c):
    from .executor import run_query
    out = []
    targets = st.target_cols
    if st.values is not None:
        for vals in st.values:
            row = [None] * len(table.columns)
            given = set()
            for j, f in enumerate(vals):
                if f is None:
                    continue
                row[targets[j]] = f.compile()(None, c) if isinstance(f, E.BE) else f(None, c)
                given.add(targets[j])
            for i, col in enumerate(table.columns):
                if i not in given:
                    df = st.defaults[i]
                    row[i] = df(None, c) if df is not None else None
            out.append(_fill_generated(st, table, row, c))
        return out
    c.outer.append(None)
    try:
        src = run_query(st.source_query, c)
    finally:
        c.outer.pop()
    conv = st.__dict__.get("_convf")
    if conv is None:
        conv = [x.compile() for x in st.source_conv]
        st._convf = conv
    for srow in src:
        row = [None] * len(table.columns)
        given = set()
        for j, f in enumerate(conv):
            row[targets[j]] = f(srow, c)
            given.add(targets[j])
        for i, col in enumerate(table.columns):
            if i not in given:
                df = st.defaults[i]
                row[i] = df(None, c) if df is not None else None
        out.append(_fill_generated(st, table, row, c))
    return out


def _compiled_vals(st):
    if st.values is None:
        return
    if st.__dict__.get("_valsc"):
        return
    st.values = [[(x.compile() if isinstance(x, E.BE) else x) for x in row] for row in st.values]
    st._valsc = True


def run_insert(st, table, c):
    _compiled_vals(st)
    new = _new_rows_insert(st, table, c)
    rows = list(table.rows)
    inserted = []
    updated = {}
    conflict = st.conflict
    if conflict is None:
        for r in new:
            row = tuple(r)
            check_row(table, row, c)
            rows.append(row)
            inserted.append(row)
    else:
        keysets = _unique_keysets(table, c, conflict["arbiters"])
        touched = set()
        for r in new:
            row = tuple(r)
            hit = None
            for name, cols, keyf, index in keysets:
                k = keyf(row)
                if k is None:
                    continue
                j = index.get(k)
                if j is not None:
                    hit = j
                    break
            if hit is None:
                check_row(table, row, c)
                rows.append(row)
                inserted.append(row)
                pos = len(rows) - 1
                for name, cols, keyf, index in keysets:
                    k = keyf(row)
                    if k is not None:
                        index[k] = pos
                continue
            if conflict["action"] == "nothing":
                continue
            if hit in touched:
                raise PgError("ON CONFLICT DO UPDATE command cannot affect row a second time", "21000",
                              hint="Ensure that no rows proposed for insertion within the same command have "
                                   "duplicate constrained values.")
            old = rows[hit]
            comb = old + row
            w = conflict["where"]
            if w is not None and not w(comb, c):
                continue
            newrow = list(old)
            for i, f in conflict["sets"]:
                newrow[i] = _set_value(f, comb, c, st, table, i)
            newrow = _fill_generated(st, table, newrow, c)
            nr = tuple(newrow)
            check_row(table, nr, c)
            for name, cols, keyf, index in keysets:
                ok = keyf(old)
                if ok is not None and index.get(ok) == hit:
                    del index[ok]
                nk = keyf(nr)
                if nk is not None:
                    index[nk] = hit
            rows[hit] = nr
            touched.add(hit)
            updated[hit] = (old, nr)
            if hit < len(table.rows) - 0 and old in inserted:
                inserted[inserted.index(old)] = nr
    out_rows = inserted + [v[1] for v in updated.values()]
    check_table(c, table, rows, changed=out_rows, removed=[v[0] for v in updated.values()])
    _replace_rows(c, table, _heap_order(rows, {k for k in updated if k < len(table.rows)}))
    st.count = len(inserted) + len(updated)
    return _ret(st, out_rows, c)


def _set_value(f, comb, c, st, table, i):
    if f == "default":
        df = st.defaults[i]
        return df(None, c) if df is not None else None
    if isinstance(f, tuple) and f[0] == "multi":
        _, sub, k, conv = f
        row = sub(comb, c)
        v = row[k] if row is not None else None
        return conv((v,), c) if v is not None else None
    return f(comb, c)


def _replace_rows(c, table, rows):
    table.rows = rows
    table.n_mod += 1
    table.n_changes = getattr(table, "n_changes", 0) + 1
    c.modified.add(table.name)


def _ret(st, rows, c):
    if not st.returning:
        return []
    fns = st.__dict__.get("_retf")
    if fns is None:
        fns = [E.compile_at(e, 0) for e in st.returning]
        st._retf = fns
    return [tuple(f(r, c) for f in fns) for r in rows]


def _target_rows(st, table, c, extra_conjs=()):
    """The target table's rows (with their positions) joined with FROM / USING and filtered by WHERE."""
    from .executor import run_node, Conj
    fake = _ctid_rel(table)
    node = st.from_node
    first = node.items[0] if isinstance(node, ListNode) else node
    first.rel = fake
    conj = [Conj(x) for x in E.conjuncts(st.where)]
    pending = list(conj)
    rows = run_node(node, c, pending)
    for cj in pending:
        f = cj.fn(0)
        rows = [r for r in rows if f(r, c)]
    return rows


def run_update(st, table, c):
    n = len(table.columns)
    matches = _target_rows(st, table, c)
    seen = {}
    for comb in matches:
        pos = comb[n]
        if pos not in seen:
            seen[pos] = comb
    rows = list(table.rows)
    changed = []
    removed = []
    ret_rows = []
    for pos, comb in seen.items():
        old = rows[pos]
        newrow = list(old)
        for i, f in st.sets:
            newrow[i] = _set_value(f, comb, c, st, table, i)
        newrow = _fill_generated(st, table, newrow, c)
        nr = tuple(newrow)
        check_row(table, nr, c)
        rows[pos] = nr
        changed.append(nr)
        removed.append(old)
        if st.returning:
            ret_rows.append(nr + (pos,) + comb[n + 1:])
    check_table(c, table, rows, changed=changed, removed=removed, update_pairs=list(zip(removed, changed)))
    _replace_rows(c, table, _heap_order(rows, set(seen)))
    st.count = len(seen)
    return _ret(st, ret_rows, c)


def _heap_order(rows, moved):
    """An UPDATE writes new row versions at the end of the table (a sequential scan then shows them last)."""
    if not moved:
        return rows
    return [r for i, r in enumerate(rows) if i not in moved] + [rows[i] for i in sorted(moved)]


def run_delete(st, table, c):
    n = len(table.columns)
    matches = _target_rows(st, table, c)
    seen = {}
    for comb in matches:
        pos = comb[n]
        if pos not in seen:
            seen[pos] = comb
    gone = set(seen)
    removed = [table.rows[p] for p in sorted(gone)]
    rows = [r for i, r in enumerate(table.rows) if i not in gone]
    check_table(c, table, rows, changed=[], removed=removed, deleting=True)
    _replace_rows(c, table, rows)
    st.count = len(gone)
    ret = [table_row + (p,) + seen[p][n + 1:] for p, table_row in zip(sorted(gone), removed)] if st.returning else []
    return _ret(st, ret, c)


def run_merge(st, table, c):
    from .executor import run_node, Conj
    n = len(table.columns)
    src_rows = run_node(st.src_node, c, [])
    fake = _ctid_rel(table)
    onf = E.compile_at(st.on, 0)
    rows = list(table.rows)
    changed, removed, inserted = [], [], []
    touched = set()
    count = 0
    width_t = n + 1
    # hash the target rows on equality conditions of ON when possible
    from .executor import _equi
    on_conj = [Conj(x) for x in E.conjuncts(st.on)]
    tslots = set(range(0, width_t))
    sslots = set(range(width_t, st.width))
    keys = [k for k in (_equi(cj, tslots, sslots) for cj in on_conj) if k is not None]
    tindex = None
    if keys:
        tk = [E.compile_at(a, 0) for a, _ in keys]
        sk = [E.compile_at(b, width_t) for _, b in keys]
        tindex = {}
        for tr in fake.rows:
            vals = [f(tr, c) for f in tk]
            if None in vals:
                continue
            tindex.setdefault(tuple(T.hash_key(v) for v in vals), []).append(tr)
    for sr in src_rows:
        if tindex is not None:
            vals = [f(sr, c) for f in sk]
            cands = tindex.get(tuple(T.hash_key(v) for v in vals), []) if None not in vals else []
        else:
            cands = fake.rows
        matched = [tr for tr in cands if onf(tr + sr, c)]
        if matched:
            for tr in matched:
                pos = tr[n]
                comb = tr + sr
                for w in st.whens:
                    if not w["matched"]:
                        continue
                    if w["cond"] is not None and not w["cond"](comb, c):
                        continue
                    if pos in touched:
                        raise PgError("MERGE command cannot affect row a second time", "21000",
                                      hint="Ensure that not more than one source row matches any one target row.")
                    if w["action"] == "update":
                        old = rows[pos]
                        newrow = list(old)
                        for i, f in w["sets"]:
                            newrow[i] = _set_value(f, comb, c, st, table, i)
                        newrow = _fill_generated(st, table, newrow, c)
                        nr = tuple(newrow)
                        check_row(table, nr, c)
                        rows[pos] = nr
                        changed.append(nr)
                        removed.append(old)
                        touched.add(pos)
                        count += 1
                    elif w["action"] == "delete":
                        removed.append(rows[pos])
                        rows[pos] = None
                        touched.add(pos)
                        count += 1
                    else:
                        touched.add(pos)
                    break
        else:
            comb = (None,) * width_t + sr
            for w in st.whens:
                if w["matched"]:
                    continue
                if w["cond"] is not None and not w["cond"](comb, c):
                    continue
                if w["action"] == "insert":
                    row = [None] * n
                    given = set()
                    for j, f in enumerate(w["values"]):
                        if f is None:
                            continue
                        row[w["targets"][j]] = f(comb, c)
                        given.add(w["targets"][j])
                    for i in range(n):
                        if i not in given:
                            df = st.defaults[i]
                            row[i] = df(None, c) if df is not None else None
                    row = tuple(_fill_generated(st, table, row, c))
                    check_row(table, row, c)
                    inserted.append(row)
                    count += 1
                break
    upd = {i for i in touched if rows[i] is not None and rows[i] is not fake.rows[i][:n]}
    rows = [r for i, r in enumerate(rows) if r is not None and i not in upd] + \
        [rows[i] for i in sorted(upd)] + inserted
    check_table(c, table, rows, changed=changed + inserted, removed=removed)
    _replace_rows(c, table, rows)
    st.count = count
    return []


# ---------------------------------------------------------------- constraints

def failing_row(table, row) -> str:
    vals = []
    for v, col in zip(row, table.columns):
        vals.append("null" if v is None else T.output(v, col.type))
    return f"Failing row contains ({', '.join(vals)})."


def check_row(table, row, c):
    """NOT NULL and CHECK constraints for one new / changed row."""
    for i, col in enumerate(table.columns):
        if col.not_null and row[i] is None:
            raise PgError(f'null value in column "{col.name}" of relation "{table.name}" violates not-null '
                          'constraint', "23502", detail=failing_row(table, row))
    checks = [k for k in table.constraints if k.kind == "c"]
    if not checks:
        return
    fns = _check_fns(table, c)
    for k, f in zip(checks, fns):
        v = f(row, c)
        if v is False:
            raise PgError(f'new row for relation "{table.name}" violates check constraint "{k.name}"', "23514",
                          detail=failing_row(table, row))


_CHECK_CACHE: dict = {}


def _check_fns(table, c):
    key = (id(table), tuple(k.name for k in table.constraints if k.kind == "c"),
           tuple(col.name for col in table.columns))
    hit = _CHECK_CACHE.get(key)
    if hit is not None and hit[0] is table:
        return hit[1]
    an = Analyzer(c.session, c.db)
    scope = Scope(None, c.session, c.db)
    rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
    scope.ns.append(NSItem(rte))
    scope.kind = "check"
    fns = []
    for k in table.constraints:
        if k.kind == "c":
            fns.append(an.expr(k.expr, scope).compile())
    if len(_CHECK_CACHE) > 500:
        _CHECK_CACHE.clear()
    _CHECK_CACHE[key] = (table, fns)
    return fns


def _unique_specs(table, c):
    """[(name, display cols, key function(row) -> key or None)] for constraints and unique indexes."""
    specs = []
    db = c.db
    for k in table.constraints:
        if k.kind in ("p", "u"):
            idx = [table.col_index(n) for n in k.cols]
            nnd = k.nulls_not_distinct

            def keyf(row, idx=idx, nnd=nnd):
                vals = [row[i] for i in idx]
                if not nnd and any(v is None for v in vals):
                    return None
                return tuple(T.hash_key(v) for v in vals)
            specs.append((k.name, k.cols, keyf, None))
    for ix in db.indexes_of(table):
        if not ix.unique or ix.constraint is not None:
            continue
        specs.append(_index_keyf(table, ix, c))
    return specs


def _index_keyf(table, ix, c):
    an = Analyzer(c.session, c.db)
    scope = Scope(None, c.session, c.db)
    rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
    scope.ns.append(NSItem(rte))
    scope.kind = "index"
    parts = []
    shown = []
    for el in ix.elems:
        if el.col is not None:
            i = table.col_index(el.col)
            parts.append(lambda row, c, i=i: row[i])
            shown.append(el.col)
        else:
            parts.append(an.expr(el.expr, scope).compile())
            shown.append(el.expr_sql)
    pred = an.expr(ix.pred, scope).compile() if ix.pred is not None else None
    nnd = ix.nulls_not_distinct

    def keyf(row):
        if pred is not None and not pred(row, c):
            return None
        vals = [p(row, c) for p in parts]
        if not nnd and any(v is None for v in vals):
            return None
        return tuple(T.hash_key(v) for v in vals)
    return (ix.name, shown, keyf, None)


def _unique_keysets(table, c, arbiters):
    specs = _unique_specs(table, c)
    if arbiters is not None:
        names = {a[2] for a in arbiters}
        specs = [s for s in specs if s[0] in names]
    out = []
    for name, cols, keyf, _ in specs:
        index = {}
        for pos, row in enumerate(table.rows):
            k = keyf(row)
            if k is not None:
                index[k] = pos
        out.append((name, cols, keyf, index))
    return out


def _key_detail(table, cols, row, c, ix_name=None):
    vals = []
    for n in cols:
        i = table.col_index(n)
        if i >= 0:
            v = row[i]
            vals.append("null" if v is None else T.output(v, table.columns[i].type))
        else:
            vals.append("?")
    return f"Key ({', '.join(cols)})=({', '.join(vals)})"


def check_table(c, table, rows, changed, removed, deleting=False, update_pairs=None):
    """Uniqueness over the table's new rows; foreign keys in both directions."""
    db = c.db
    if changed:
        specs = _unique_specs(table, c)
        if specs:
            changed_ids = {id(r) for r in changed}
            for name, cols, keyf, _ in specs:
                seen = {}
                for row in rows:
                    k = keyf(row)
                    if k is None:
                        continue
                    if k in seen:
                        other = seen[k]
                        bad = row if id(row) in changed_ids else other
                        detail = _unique_detail(table, cols, bad, c, name)
                        raise PgError(f'duplicate key value violates unique constraint "{name}"', "23505",
                                      detail=detail)
                    seen[k] = row
        # child side: referenced keys must exist
        for k in table.constraints:
            if k.kind != "f":
                continue
            parent = db.find_rel(k.ref_table, k.ref_schema, None)
            if parent is None:
                continue
            pidx = [parent.col_index(n) for n in k.ref_cols]
            cidx = [table.col_index(n) for n in k.cols]
            prows = rows if parent is table else parent.rows
            keys = {tuple(T.hash_key(r[i]) for i in pidx) for r in prows}
            for row in changed:
                vals = [row[i] for i in cidx]
                if any(v is None for v in vals):
                    continue
                if tuple(T.hash_key(v) for v in vals) not in keys:
                    shown = ", ".join(T.output(v, table.columns[i].type) for v, i in zip(vals, cidx))
                    raise PgError(f'insert or update on table "{table.name}" violates foreign key constraint '
                                  f'"{k.name}"', "23503",
                                  detail=f'Key ({", ".join(k.cols)})=({shown}) is not present in table '
                                         f'"{parent.name}".')
    if removed:
        _parent_side(c, table, rows, removed, update_pairs)


def _unique_detail(table, cols, row, c, name):
    if all(table.col_index(n) >= 0 for n in cols):
        return _key_detail(table, cols, row, c) + " already exists."
    # an expression index: show the expression values
    ix = next((i for i in c.db.indexes_of(table) if i.name == name), None)
    vals = []
    if ix is not None:
        an = Analyzer(c.session, c.db)
        scope = Scope(None, c.session, c.db)
        rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
        scope.ns.append(NSItem(rte))
        for el in ix.elems:
            if el.col is not None:
                i = table.col_index(el.col)
                vals.append(T.output(row[i], table.columns[i].type) if row[i] is not None else "null")
            else:
                e = an.expr(el.expr, scope)
                v = e.compile()(row, c)
                vals.append(T.output(v, e.type) if v is not None else "null")
    return f"Key ({', '.join(cols)})=({', '.join(vals)}) already exists."


def _parent_side(c, table, rows, removed, update_pairs):
    """Rows of `table` were deleted or changed: what references them?"""
    db = c.db
    for child in list(db.all_rels()):
        if not isinstance(child, Table):
            continue
        for k in child.constraints:
            if k.kind != "f" or k.ref_table != table.name or (k.ref_schema or table.schema) != table.schema:
                continue
            pidx = [table.col_index(n) for n in k.ref_cols]
            cidx = [child.col_index(n) for n in k.cols]
            still = {tuple(T.hash_key(r[i]) for i in pidx) for r in rows}
            gone = {}
            for r in removed:
                key = tuple(T.hash_key(r[i]) for i in pidx)
                if key not in still:
                    gone[key] = r
            if not gone:
                continue
            crows = rows if child is table else child.rows
            hits = [r for r in crows if tuple(T.hash_key(r[i]) for i in cidx) in gone and
                    all(r[i] is not None for i in cidx)]
            if not hits:
                continue
            action = k.on_delete if update_pairs is None else k.on_update
            if action in ("a", "r"):
                bad = gone[tuple(T.hash_key(hits[0][i]) for i in cidx)]
                shown = ", ".join(T.output(bad[i], table.columns[i].type) for i in pidx)
                raise PgError(f'update or delete on table "{table.name}" violates foreign key constraint '
                              f'"{k.name}" on table "{child.name}"', "23503",
                              detail=f'Key ({", ".join(k.ref_cols)})=({shown}) is still referenced from table '
                                     f'"{child.name}".')
            hit_ids = {id(r) for r in hits}
            if action == "c":
                if update_pairs is None:
                    keep = [r for r in child.rows if id(r) not in hit_ids]
                    gone_rows = [r for r in child.rows if id(r) in hit_ids]
                    if child is table:
                        continue
                    check_table(c, child, keep, changed=[], removed=gone_rows)
                    child.rows = keep
                    c.modified.add(child.name)
                else:
                    newmap = {}
                    for old, new in update_pairs:
                        newmap[tuple(T.hash_key(old[i]) for i in pidx)] = [new[i] for i in pidx]
                    newrows, chg, rem = [], [], []
                    for r in child.rows:
                        if id(r) in hit_ids:
                            nv = newmap.get(tuple(T.hash_key(r[i]) for i in cidx))
                            if nv is not None:
                                lr = list(r)
                                for ci, v in zip(cidx, nv):
                                    lr[ci] = v
                                nr = tuple(lr)
                                newrows.append(nr)
                                chg.append(nr)
                                rem.append(r)
                                continue
                        newrows.append(r)
                    child.rows = newrows
                    c.modified.add(child.name)
            elif action in ("n", "d"):
                newrows, chg = [], []
                defaults = None
                if action == "d":
                    defaults = _bind_defaults(Analyzer(c.session, c.db), child)
                for r in child.rows:
                    if id(r) in hit_ids:
                        lr = list(r)
                        for ci in cidx:
                            lr[ci] = None if action == "n" else (defaults[ci](None, c) if defaults[ci] else None)
                        nr = tuple(lr)
                        check_row(child, nr, c)
                        newrows.append(nr)
                        chg.append(nr)
                    else:
                        newrows.append(r)
                check_table(c, child, newrows, changed=chg, removed=[])
                child.rows = newrows
                c.modified.add(child.name)

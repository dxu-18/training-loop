"""The PostgreSQL side of a problem's machine: its databases (built from the problem's `sql` spec before the
session), the live bar under the terminal, and psql's Tab completion."""

from __future__ import annotations

import re


def build_cluster(world, spec, override=None):
    """Create the problem's databases. spec["dbs"]: {name: setup SQL}; override: other setup SQL for the main
    database (a hidden test's data)."""
    from .utils.psqlprog import cluster_of
    from ..postgreslite.server import Session
    from ..postgreslite.catalog import Table
    from ..postgreslite.ddl import autovacuum as compute_stats
    cluster = cluster_of(world)
    dbs = dict(spec.get("dbs") or {})
    main = spec.get("db")
    if override is not None and main:
        dbs[main] = override
    for name, sql in dbs.items():
        if name not in cluster.databases:
            cluster.create_database(name, "ada")
        if not sql:
            continue
        s = cluster.connect(name, "ada", clock=lambda: world.kernel.now // 1000 - 86_400_000_000)
        s.execute(sql)
        s.close()
        db = cluster.databases[name]
        if spec.get("analyze", True):
            for rel in db.all_rels():
                if isinstance(rel, Table):
                    compute_stats(rel)
    return cluster


def last_result(t):
    log = getattr(t.world, "sql_log", None) or []
    for rec in reversed(log):
        for r in reversed(rec["results"]):
            if r.get("names") is not None:
                return rec, r
        if rec.get("error"):
            return rec, None
    return None, None


def bar_info(t, spec):
    rec, r = last_result(t)
    info = {"check": spec.get("check", "query")}
    if rec is None:
        return info
    if r is None:
        info["last"] = "error"
        return info
    n = len(r["rows"])
    info["last"] = f"{n} row{'s' if n != 1 else ''} × {len(r['names'])} column{'s' if len(r['names']) != 1 else ''}"
    exp = getattr(t, "sql_expect", None)
    if exp is not None and spec.get("check", "query") == "query":
        from .sqlcheck import result_matches
        info["match"] = result_matches(spec, exp, r) is None
    return info


_KW = ("SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "OFFSET", "INSERT INTO", "VALUES",
       "UPDATE", "SET", "DELETE FROM", "CREATE TABLE", "CREATE INDEX", "CREATE VIEW", "ALTER TABLE", "DROP TABLE",
       "JOIN", "LEFT JOIN", "INNER JOIN", "ON", "USING", "DISTINCT", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT",
       "EXPLAIN", "ANALYZE", "WITH", "UNION", "CASE", "WHEN", "THEN", "ELSE", "END", "RETURNING", "TRUNCATE")


def complete(t, ed, p):
    """psql's Tab: table and column names of the current database, then SQL keywords."""
    buf, pos = ed.buf, ed.pos
    m = re.search(r"[A-Za-z_][A-Za-z0-9_.]*$", buf[:pos])
    if not m:
        return
    word = m.group(0)
    db = p.session.db()
    names = set()
    for rel in db.all_rels():
        names.add(rel.name)
        for c in getattr(rel, "columns", []) or []:
            names.add(c.name)
    cands = sorted(n for n in names if n.startswith(word.lower()))
    if not cands:
        up = word.upper()
        kws = [k for k in _KW if k.startswith(up)]
        if word.islower():
            kws = [k.lower() for k in kws]
        cands = kws
    if not cands:
        return
    pre = cands[0]
    for c in cands[1:]:
        while not c.startswith(pre):
            pre = pre[:-1]
    if len(cands) == 1:
        pre += " "
    if len(pre) > len(word):
        ed.insert(pre[len(word):])


def db_tree(t):
    """The Database panel: the open database's tables and views with columns and row counts (never fails: the
    panel is a view, the terminal must go on)."""
    try:
        return _db_tree(t)
    except Exception:  # noqa: BLE001
        return {"db": (t.problem.get("sql") or {}).get("db"), "rels": []}


def _db_tree(t):
    cl = t.world.local.extra.get("pg")
    spec = t.problem.get("sql") or {}
    name = spec.get("db")
    term = getattr(t.world, "psql", None) or getattr(t.world, "psql_last", None)
    if term is not None and term.psql.session is not None:
        name = term.psql.session.dbname
    if cl is None or name not in cl.databases:
        return {"db": name, "rels": []}
    from ..postgreslite.catalog import Table, View, Index
    db = cl.databases[name]
    if term is not None and term.psql.session is not None and term.psql.session.txn is not None:
        db = term.psql.session.txn.db
    rels = []
    for sname in sorted(db.schemas):
        if sname.startswith("pg_temp"):
            continue
        for r in sorted(db.schemas[sname].rels.values(), key=lambda r: r.name):
            if isinstance(r, (Table, View)):
                kind = "table" if isinstance(r, Table) else ("matview" if r.materialized else "view")
                cols = [[c.name, c.type.format()] for c in r.columns]
                rows = len(r.rows) if kind != "view" else None
                idx = [ix.name for ix in db.indexes_of(r)] if kind != "view" else []
                rels.append({"name": (sname + "." if sname != "public" else "") + r.name, "kind": kind,
                             "cols": cols, "rows": rows, "idx": idx,
                             # changes since the start: the columns, the row count, or any statement that wrote
                             "sig": f"{cols}|{rows}|{getattr(r, 'n_changes', 0)}"})
    return {"db": name, "rels": rels}

"""Helpers shared by the SQL templates (COMP_GUIDE §10).

A SQL problem runs on bash-lite's machine with a PostgreSQL 16 server (generator/postgreslite): the problem's `sql`
spec builds its databases before the session, and `#: psql shop` opens psql on the shown one (quitting is turned
off, except in 10.15, which starts at the bash prompt). bashlite/sqlcheck.py grades the session: the queries' results
(on the shown database and hidden ones), or the end state of the tables, plans and psql settings.

`Q()` builds a problem: it runs the reference solution once on the shown database (it must work), shows its result
as the target, and records the hidden databases.
"""

from __future__ import annotations

import functools
import random

from ...core import Problem, pick, template as _template
from . import _data as D

S = "sql"


class SqlSetupError(AssertionError):
    pass


def template(module: str, tier: str):
    """Register a SQL template. Random data can leave a task without an answer on the shown database (no row
    matches, a key collides): then the builder runs again on a fresh random stream (deterministic per seed), up to
    RETRIES times."""
    def deco(fn):
        @functools.wraps(fn)
        def build(rng):
            err = None
            for attempt in range(RETRIES):
                try:
                    return fn(rng if attempt == 0 else random.Random(rng.getrandbits(64) ^ attempt))
                except SqlSetupError as e:
                    err = e
            raise err
        return _template(S, module, tier)(build)
    return deco


RETRIES = 12


def run_reference(setup_sql, solution, dbname="shop", extra=None):
    """Run the solution in a fresh cluster with psql (no terminal). -> (psql, cluster)."""
    from ...postgreslite.server import Cluster
    from ...postgreslite.psql import Psql, Env
    from ...postgreslite.catalog import Table
    from ...postgreslite.ddl import autovacuum as compute_stats
    cl = Cluster()
    dbs = {dbname: setup_sql}
    dbs.update(extra or {})
    for name, sql in dbs.items():
        cl.create_database(name, "ada")
        if sql:
            s = cl.connect(name, "ada")
            s.execute(sql)
            s.close()
            for rel in cl.databases[name].all_rels():
                if isinstance(rel, Table):
                    compute_stats(rel)

    class E(Env):
        files = {}

        def read_file(self, path):
            if path in self.files:
                return self.files[path]
            raise OSError("No such file or directory")

        def write_file(self, path, text, append=False):
            self.files[path] = (self.files.get(path, "") if append else "") + text

        def now_us(self):
            return 1717243200 * 10 ** 6
    p = Psql(cl, dbname, "ada", E(), interactive=False)
    p.log = []
    p.run_script(solution, None)
    return p, cl


class _Log:
    def __init__(self, log):
        self.log = log


def _bash_reference(db, sol, dbname, extra, given, directives=()):
    """Problems that start at the bash prompt (or leave psql): run the reference in a terminal on the machine."""
    from ...bashlite.sqlcheck import _term, type_lines
    setup = list(directives)
    for path, content in (given or {}).items():
        setup.append(f"cat > '{path}' <<'EOF_GIVEN'\n{content.rstrip(chr(10))}\nEOF_GIVEN")
    prob = {"setup": "\n".join(setup), "seed": 0, "solution": sol,
            "sql": {"db": dbname, "dbs": {dbname: db, **(extra or {})}}}
    t = type_lines(_term(prob), sol)
    return _Log(list(getattr(t.world, "sql_log", None) or []))


def render(r, opts=None):
    from ...postgreslite.psql import print_table, PrintOpts
    return print_table(r["names"], r["typeobjs"], r["rows"], opts or PrintOpts())


def last_rows(p):
    for rec in reversed(p.log):
        for r in reversed(rec["results"]):
            if r.get("names") is not None:
                return r
    return None


def Q(prompt: str, solution: str, db: str, hidden=(), *, check="query", order=False, require=(), forbid=(),
      tables=None, probes=None, plans=None, show="target", dbname="shop", extra_dbs=None, start="psql",
      files=None, settings=None, also=None, timeout=40.0, quit_ok=False, note="", errors_ok=False,
      compare=None, given=None, sqlite=None, machine=()) -> Problem:
    sol = solution.strip("\n") + "\n"
    if start == "psql" and not quit_ok and not given and not machine:
        p, cl = run_reference(db, sol, dbname, extra_dbs)
    else:
        p = _bash_reference(db, sol, dbname, extra_dbs, given, list(machine) + (
            [f"#: psql {dbname}"] + (["#: psqlquit"] if quit_ok else []) if start == "psql" else []))
    errs = [rec for rec in p.log if rec["error"]]
    if errs and not errors_ok:
        raise SqlSetupError(f"the reference failed: {errs[0]['error'][1]} in {errs[0]['sql']!r}")
    target = last_rows(p)
    if check == "query" or also == "query":
        if target is None:
            raise SqlSetupError(f"the reference returned no result: {sol!r}")
        if not target["rows"] and show == "target":
            raise SqlSetupError(f"the reference returned no rows on the shown database: {sol!r}")
    spec = {"db": dbname, "dbs": {dbname: db, **(extra_dbs or {})}, "check": check,
            "hidden": [h[0] for h in hidden], "hidden_tags": [h[1] for h in hidden]}
    if order:
        spec["order"] = True
    if tables:
        spec["tables"] = list(tables)
    if probes:
        spec["probes"] = list(probes)
    if plans:
        spec["plans"] = list(plans)
    if settings:
        spec["settings"] = list(settings)
    if files:
        spec["files"] = list(files)
    if also:
        spec["also"] = also
    if compare:
        spec["compare"] = list(compare)
    if sqlite:
        spec["sqlite"] = [list(x) for x in sqlite]
    setup = list(machine)
    for path, content in (given or {}).items():
        delim = "EOF"
        while delim in content.split("\n"):
            delim += "_"
        setup.append(f"cat > '{path}' <<'{delim}'\n{content.rstrip(chr(10))}\n{delim}")
    if start == "psql":
        setup.append(f"#: psql {dbname}")
        if quit_ok:
            setup.append("#: psqlquit")
    text = prompt.strip()
    if (check == "query" or also == "query") and show == "target":
        text += "\n\nExpected result on this database:\n\n```\n" + render(target).rstrip("\n") + "\n```"
    if note:
        text += "\n\n" + note.strip()
    text += "\n\n" + footer(check, start, dbname, bool(hidden), order)
    prob = Problem(prompt=text, setup="\n".join(setup), solution=sol, kind="bash", lang="sql",
                   require=list(require), forbid=list(forbid), timeout=timeout, sql=spec)
    return prob


def footer(check, start, dbname, has_hidden, order):
    bits = []
    if start == "psql":
        bits.append(f"psql is open on the `{dbname}` database.")
    else:
        bits.append("Start at the bash prompt.")
    if check == "query":
        bits.append("Explore freely; one of your queries must return exactly these rows" +
                    (" in this order" if order else " (in any order)") +
                    (", here and on hidden databases with other data." if has_hidden else "."))
    elif check in ("state", "explain"):
        bits.append("Submit checks the database your session leaves" + (" (and the same session on hidden "
                                                                          "databases)." if has_hidden else "."))
    return " ".join(bits)


def drill(rng, items, db, hidden=(), **common):
    """One task from a list: (prompt, solution, {Q options})."""
    prompt, sol, *rest = pick(rng, items)
    opts = {**common, **(rest[0] if rest else {})}
    return Q(prompt, sol, db, hidden, **opts)


def shop_db(rng, **kw):
    sql, facts = D.shop(rng, **kw)
    hid = D.hidden(D.shop, rng, facts, D.SHOP_TRAPS, k=3, **{k: v for k, v in kw.items() if k.startswith("n_")})
    return sql, facts, hid


def fmt_money(d):
    return f"{d:.2f}"

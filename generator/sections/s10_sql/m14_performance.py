"""10.14 Query performance: EXPLAIN, indexes, sargable rewrites. Big tables, plans checked with EXPLAIN (COSTS OFF)."""

from __future__ import annotations

from ._common import template, drill, Q
from . import _data as D

BIG = """CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT, signup_date DATE NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), amount NUMERIC(10, 2) NOT NULL,
  status TEXT NOT NULL DEFAULT 'paid', created_at TIMESTAMP NOT NULL);
INSERT INTO users SELECT g, 'user' || g, (ARRAY['US','UK','DE','FR',NULL])[1 + g % 5], DATE '2023-01-01' + (g % 400)
  FROM generate_series(1, {nu}) g;
INSERT INTO orders SELECT g, 1 + (g * {mul}) % {nu}, round((g * {amul} % 50000) / 100.0, 2),
  (ARRAY['paid','paid','paid','refunded','pending'])[1 + g % 5], TIMESTAMP '2024-01-01' + (g * 613 % 525600) * INTERVAL '1 minute'
  FROM generate_series(1, {no}) g;
"""


def big_db(rng, nu=500, no=6000):
    mul = rng.choice([7919, 7907, 7901, 7949, 7951, 7963])
    amul = rng.choice([37, 41, 43, 47, 53])
    sql = BIG.format(nu=nu, no=no, mul=mul, amul=amul)
    hid = [(BIG.format(nu=nu + 100, no=no + 1500, mul=rng.choice([7883, 7877]), amul=amul), "more rows")]
    return sql, hid


@template("10.14", "lite")
def perf_explain(rng):
    db, hid = big_db(rng)
    uid = rng.randint(2, 400)
    items = [
        (f"Show the plan of `SELECT * FROM orders WHERE user_id = {uid}` (EXPLAIN). It reads the whole table: a Seq "
         "Scan with a Filter.", f"EXPLAIN SELECT * FROM orders WHERE user_id = {uid};",
         {"check": "query", "show": "none", "require": ["EXPLAIN"]}),
        ("Show the plan of `SELECT count(*) FROM orders` with EXPLAIN.", "EXPLAIN SELECT count(*) FROM orders;",
         {"check": "query", "show": "none", "require": ["EXPLAIN"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.14", "lite")
def perf_explain_pk(rng):
    db, hid = big_db(rng)
    oid = rng.randint(10, 5000)
    items = [
        (f"Show the plan of `SELECT * FROM orders WHERE id = {oid}` with EXPLAIN (the primary key's index is used).",
         f"EXPLAIN SELECT * FROM orders WHERE id = {oid};", {"check": "query", "show": "none", "require": ["EXPLAIN"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.14", "easy")
def perf_index(rng):
    db, hid = big_db(rng)
    uid = rng.randint(2, 400)
    items = [
        (f"`SELECT * FROM orders WHERE user_id = {uid}` scans the whole table. Create an index so the plan uses it "
         "instead (any index name).", "CREATE INDEX idx_orders_user ON orders (user_id);",
         {"check": "explain", "tables": ["orders"], "compare": [],
          "plans": [{"query": f"SELECT * FROM orders WHERE user_id = {uid}", "uses": ["Index Scan|Bitmap Index Scan"],
                     "avoids": ["Seq Scan on orders"]}]}),
        ("Orders are often looked up by `created_at` ranges. Create an index so `SELECT * FROM orders WHERE created_at "
         ">= '2024-03-01' AND created_at < '2024-03-08'` uses it.",
         "CREATE INDEX idx_orders_created ON orders (created_at);",
         {"check": "explain", "tables": ["orders"], "compare": [],
          "plans": [{"query": "SELECT * FROM orders WHERE created_at >= '2024-03-01' AND created_at < '2024-03-08'",
                     "uses": ["Index Scan|Bitmap Index Scan"]}]}),
    ]
    return drill(rng, items, db, hid)


@template("10.14", "easy")
def perf_analyze(rng):
    db, hid = big_db(rng)
    n, kinds = rng.choice([2000, 3000, 4000, 5000]), rng.choice([20, 50, 100])
    extra = (f"CREATE TABLE events_raw AS SELECT g AS id, g % {kinds} AS kind FROM generate_series(1, {n}) g;\n")
    items = [
        ("The table `events_raw` was just loaded and has no statistics. Refresh the planner's statistics for it "
         "(ANALYZE).", "ANALYZE events_raw;",
         {"check": "state", "tables": ["events_raw"], "require": ["ANALYZE"],
          "probes": ["SELECT 1"]}),
    ]
    from ...core import pick
    prompt, sol, opts = pick(rng, items)
    return Q(prompt, sol, db.replace("FROM generate_series(1, 6000) g;", "FROM generate_series(1, 6000) g;") + extra,
             [(h + extra, t) for h, t in hid], **opts)


@template("10.14", "medium")
def perf_sargable(rng):
    db, hid = big_db(rng)
    month = rng.choice([2, 3, 4])
    extra = "CREATE INDEX idx_orders_created ON orders (created_at);\n"
    items = [
        (f"`SELECT * FROM orders WHERE DATE_TRUNC('month', created_at) = DATE '2024-{month:02d}-01'` can't use the "
         "index on created_at (a function wraps the column). Write the same query as a range on the bare column "
         "(half-open), so the index is used.",
         f"SELECT * FROM orders WHERE created_at >= DATE '2024-{month:02d}-01' AND created_at < DATE "
         f"'2024-{month + 1:02d}-01';", {"check": "query", "show": "none", "forbid": ["DATE_TRUNC("]}),
        ("`SELECT id FROM orders WHERE created_at::DATE = DATE '2024-03-05'` casts the column. Rewrite it as a range "
         "on created_at.", "SELECT id FROM orders WHERE created_at >= DATE '2024-03-05' AND created_at < DATE "
                           "'2024-03-06';", {"check": "query", "show": "none", "forbid": ["::DATE"]}),
    ]
    from ...core import pick
    prompt, sol, opts = pick(rng, items)
    return Q(prompt, sol, db + extra, [(h + extra, t) for h, t in hid], **opts)


@template("10.14", "medium")
def perf_explain_analyze(rng):
    db, hid = big_db(rng)
    uid = rng.randint(2, 400)
    items = [
        (f"Run `SELECT * FROM orders WHERE user_id = {uid}` with EXPLAIN ANALYZE (it runs the query and shows actual "
         "row counts and times), then create an index on orders(user_id) and run EXPLAIN ANALYZE again.",
         f"EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = {uid};\nCREATE INDEX idx_orders_user ON orders "
         f"(user_id);\nEXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = {uid};",
         {"check": "explain", "tables": ["orders"], "compare": [], "require": ["EXPLAIN ANALYZE"],
          "plans": [{"query": f"SELECT * FROM orders WHERE user_id = {uid}", "uses": ["Index Scan|Bitmap"]}]}),
    ]
    return drill(rng, items, db, hid)


@template("10.14", "hard")
def perf_multicol(rng):
    db, hid = big_db(rng)
    uid = rng.randint(2, 400)
    items = [
        (f"The app runs `SELECT * FROM orders WHERE user_id = {uid} ORDER BY created_at DESC LIMIT 5`. Create one "
         "index that serves both the filter and the sort (no Sort node in the plan).",
         "CREATE INDEX idx_orders_user_time ON orders (user_id, created_at DESC);",
         {"check": "explain", "tables": ["orders"], "compare": [],
          "plans": [{"query": f"SELECT * FROM orders WHERE user_id = {uid} ORDER BY created_at DESC LIMIT 5",
                     "uses": ["Index Scan"], "avoids": ["Sort"]}]}),
        ("Only pending orders are ever queried by time: `SELECT * FROM orders WHERE status = 'pending' AND created_at "
         ">= '2024-06-01'`. Create a partial index on created_at for pending orders only.",
         "CREATE INDEX idx_orders_pending ON orders (created_at) WHERE status = 'pending';",
         {"check": "explain", "tables": ["orders"], "compare": [],
          "plans": [{"query": "SELECT * FROM orders WHERE status = 'pending' AND created_at >= '2024-06-01'",
                     "uses": ["idx_orders_pending|Index Scan|Bitmap"]}]}),
    ]
    return drill(rng, items, db, hid)


@template("10.14", "hard")
def perf_exists(rng):
    db, hid = big_db(rng, nu=300, no=3000)
    extra = "CREATE INDEX idx_orders_user ON orders (user_id);\n"
    cut = rng.choice([420, 440, 450, 460, 480])
    items = [
        (f"`SELECT DISTINCT u.name FROM users u JOIN orders o ON o.user_id = u.id WHERE o.amount > {cut}` hides the "
         "join's duplicates with DISTINCT. Return the same names with EXISTS instead (no DISTINCT, no join).",
         f"SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > {cut});",
         {"check": "query", "show": "none", "require": ["EXISTS"], "forbid": ["DISTINCT"]}),
    ]
    from ...core import pick
    prompt, sol, opts = pick(rng, items)
    return Q(prompt, sol, db + extra, [(h + extra, t) for h, t in hid], **opts)


@template("10.14", "challenge")
def perf_rewrite(rng):
    db, hid = big_db(rng, nu=400, no=5000)
    month = rng.choice([2, 3])
    sol = (f"CREATE INDEX idx_orders_created ON orders (created_at);\nSELECT user_id, SUM(amount) AS revenue FROM orders "
           f"WHERE created_at >= DATE '2024-{month:02d}-01' AND created_at < DATE '2024-{month + 1:02d}-01' GROUP BY "
           "user_id ORDER BY revenue DESC, user_id LIMIT 10;")
    return Q(f"This report is slow: `SELECT user_id, SUM(amount) AS revenue FROM orders WHERE EXTRACT(MONTH FROM "
             f"created_at) = {month} GROUP BY user_id ORDER BY revenue DESC, user_id LIMIT 10` (all orders are in "
             "2024). Make it fast: create an index and rewrite the filter so the plan reads orders through the index "
             "(no Seq Scan on orders). Run the rewritten query in your session.", sol, db, hid, check="explain",
             tables=["orders"], compare=[], also="query", order=True, forbid=["EXTRACT("] if False else (),
             plans=[{"query": f"SELECT user_id, SUM(amount) FROM orders WHERE created_at >= DATE '2024-{month:02d}-01' "
                              f"AND created_at < DATE '2024-{month + 1:02d}-01' GROUP BY user_id",
                     "uses": ["Index|Bitmap"], "avoids": ["Seq Scan on orders"]}])


@template("10.14", "challenge")
def perf_join(rng):
    db, hid = big_db(rng, nu=400, no=5000)
    uid = rng.randint(10, 390)
    sol = ("CREATE INDEX idx_orders_user ON orders (user_id);\nSELECT o.id, o.amount FROM users u JOIN orders o ON "
           f"o.user_id = u.id WHERE u.id = {uid} ORDER BY o.id;")
    return Q("The support tool runs `SELECT o.id, o.amount FROM users u JOIN orders o ON o.user_id = u.id WHERE u.id "
             f"= {uid} ORDER BY o.id` and its plan scans all of orders. Add the index the join needs so orders are read "
             "through it, then run the query.", sol, db, hid, check="explain", tables=["orders"], compare=[],
             also="query", order=True,
             plans=[{"query": f"SELECT o.id, o.amount FROM users u JOIN orders o ON o.user_id = u.id WHERE u.id = {uid}",
                     "uses": ["Index|Bitmap"], "avoids": ["Seq Scan on orders"]}])

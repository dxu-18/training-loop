"""10.12 Transactions, views & indexes."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q

ST = {"check": "state"}


@template("10.12", "lite")
def tx_commit(rng):
    db, f, hid = shop_db(rng)
    a, b = rng.sample([o[0] for o in f["orders"]], 2)
    items = [
        (f"Move 10 from order {a} to order {b} atomically: in one transaction subtract 10 from order {a}'s amount, "
         f"add 10 to order {b}'s, then commit.",
         f"BEGIN;\nUPDATE orders SET amount = amount - 10 WHERE id = {a};\nUPDATE orders SET amount = amount + 10 WHERE "
         f"id = {b};\nCOMMIT;", {**ST, "tables": ["orders"], "require": ["BEGIN", "COMMIT"]}),
        ("In a transaction, delete all refunded orders, look at the count of what's left, then commit.",
         "BEGIN;\nDELETE FROM orders WHERE status = 'refunded';\nSELECT COUNT(*) FROM orders;\nCOMMIT;",
         {**ST, "tables": ["orders"], "require": ["BEGIN", "COMMIT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "lite")
def view_create(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create a view `paid_orders` with all columns of the paid orders.",
         "CREATE VIEW paid_orders AS SELECT * FROM orders WHERE status = 'paid';",
         {**ST, "tables": ["paid_orders"], "probes": ["SELECT * FROM paid_orders"]}),
        ("Create a view `user_spend` with `user_id` and `total` (sum of each user's orders).",
         "CREATE VIEW user_spend AS SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id;",
         {**ST, "tables": ["user_spend"], "probes": ["SELECT * FROM user_spend"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "easy")
def tx_rollback(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Start a transaction, delete every order (a mistake!), then undo it with ROLLBACK.",
         "BEGIN;\nDELETE FROM orders;\nROLLBACK;", {**ST, "tables": ["orders"], "require": ["ROLLBACK"]}),
        ("In a transaction, set every amount to 0, check it, then roll it back so nothing changes.",
         "BEGIN;\nUPDATE orders SET amount = 0;\nSELECT SUM(amount) FROM orders;\nROLLBACK;",
         {**ST, "tables": ["orders"], "require": ["ROLLBACK"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "easy")
def idx_create(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create an index `idx_orders_user` on orders(user_id).", "CREATE INDEX idx_orders_user ON orders (user_id);",
         {**ST, "tables": ["orders"], "compare": ["indexes", "rows"]}),
        ("Create an index `idx_orders_created` on orders(created_at).",
         "CREATE INDEX idx_orders_created ON orders (created_at);", {**ST, "tables": ["orders"],
                                                                     "compare": ["indexes"]}),
        ("Create an index on users(country) (any name).", "CREATE INDEX ON users (country);",
         {**ST, "tables": ["users"], "compare": ["indexes"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "medium")
def tx_savepoint(rng):
    db, f, hid = shop_db(rng)
    nid = len(f["users"]) + 1
    items = [
        (f"In one transaction: add user {nid} `Zed` (`CA`, 2024-05-01), set a savepoint `before_delete`, delete all "
         "orders, roll back to the savepoint, and commit (the user stays, the orders too).",
         f"BEGIN;\nINSERT INTO users VALUES ({nid}, 'Zed', 'CA', '2024-05-01');\nSAVEPOINT before_delete;\nDELETE FROM "
         "orders;\nROLLBACK TO SAVEPOINT before_delete;\nCOMMIT;",
         {**ST, "tables": ["users", "orders"], "require": ["SAVEPOINT", "ROLLBACK TO"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "medium")
def view_matview(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create a materialized view `monthly_revenue` with `month` (DATE_TRUNC) and `revenue` per month.",
         "CREATE MATERIALIZED VIEW monthly_revenue AS SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS "
         "revenue FROM orders GROUP BY 1;", {**ST, "tables": ["monthly_revenue"]}),
        ("A materialized view `status_counts` (status, n) already exists but is stale: refresh it.",
         "REFRESH MATERIALIZED VIEW status_counts;",
         {**ST, "tables": ["status_counts"], "require": ["REFRESH"],
          "_extra": "CREATE MATERIALIZED VIEW status_counts AS SELECT status, COUNT(*) AS n FROM orders GROUP BY "
                    "status;\nINSERT INTO orders SELECT id + 1000, user_id, amount, 'pending', created_at FROM "
                    "orders;\n"}),
    ]
    item = items[rng.randrange(len(items))] if False else None
    from ...core import pick
    prompt, sol, opts = pick(rng, items)
    extra = opts.pop("_extra", "")
    return Q(prompt, sol, db + extra, [(h + extra, t) for h, t in hid], **opts)


@template("10.12", "medium")
def idx_special(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create a unique index `idx_users_lower_name` on LOWER(name) (an expression index).",
         "CREATE UNIQUE INDEX idx_users_lower_name ON users (LOWER(name));",
         {**ST, "tables": ["users"], "compare": ["indexes"]}),
        ("Create a partial index `idx_orders_paid` on orders(created_at) covering only paid orders.",
         "CREATE INDEX idx_orders_paid ON orders (created_at) WHERE status = 'paid';",
         {**ST, "tables": ["orders"], "compare": ["indexes"]}),
        ("Create a multi-column index `idx_orders_user_time` on orders(user_id, created_at DESC).",
         "CREATE INDEX idx_orders_user_time ON orders (user_id, created_at DESC);",
         {**ST, "tables": ["orders"], "compare": ["indexes"]}),
    ]
    return drill(rng, items, db, [h for h in hid if "duplicate" not in h[1]])


@template("10.12", "hard")
def tx_aborted(rng):
    db, f, hid = shop_db(rng)
    nid = len(f["users"]) + 1
    items = [
        (f"In a transaction add user {nid} (`Yan`, `IN`, 2024-05-02); then try to add user 1 again (it fails and "
         "aborts the transaction). Recover: roll back to a savepoint you set before the failing insert, and commit so "
         f"user {nid} is kept.",
         f"BEGIN;\nINSERT INTO users VALUES ({nid}, 'Yan', 'IN', '2024-05-02');\nSAVEPOINT s;\nINSERT INTO users VALUES "
         "(1, 'Dup', 'US', '2024-01-01');\nROLLBACK TO SAVEPOINT s;\nCOMMIT;",
         {**ST, "tables": ["users"], "errors_ok": True, "require": ["SAVEPOINT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.12", "hard")
def view_ltv(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create a view `user_ltv`: every user's `id`, `name` and `ltv` (sum of paid orders, 0 when none).",
         "CREATE VIEW user_ltv AS SELECT u.id, u.name, COALESCE(SUM(o.amount) FILTER (WHERE o.status = 'paid'), 0) AS "
         "ltv FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id, u.name;",
         {**ST, "tables": ["user_ltv"], "probes": ["SELECT * FROM user_ltv"]}),
        ("Replace the view `v_orders` (it shows id and amount) so it also shows `status` at the end (CREATE OR "
         "REPLACE).", "CREATE OR REPLACE VIEW v_orders AS SELECT id, amount, status FROM orders;",
         {**ST, "tables": ["v_orders"], "probes": ["SELECT * FROM v_orders"],
          "_extra": "CREATE VIEW v_orders AS SELECT id, amount FROM orders;\n"}),
    ]
    from ...core import pick
    prompt, sol, opts = pick(rng, items)
    extra = opts.pop("_extra", "")
    return Q(prompt, sol, db + extra, [(h + extra, t) for h, t in hid], **opts)


@template("10.12", "challenge")
def tx_migration(rng):
    db, f, hid = shop_db(rng)
    sol = ("BEGIN;\nALTER TABLE orders ADD COLUMN currency TEXT NOT NULL DEFAULT 'USD';\nSAVEPOINT cur;\nUPDATE orders "
           "SET amount = amount * 100;\nROLLBACK TO SAVEPOINT cur;\nCREATE INDEX idx_orders_currency ON orders "
           "(currency);\nCOMMIT;")
    return Q("Run a migration in one transaction: add `currency` TEXT NOT NULL DEFAULT `USD` to orders; set a "
             "savepoint; (someone multiplies all amounts by 100 by mistake: do that UPDATE too); go back to the "
             "savepoint to undo only that; create index `idx_orders_currency` on orders(currency); commit.",
             sol, db, hid, check="state", tables=["orders"], require=["SAVEPOINT", "ROLLBACK TO"],
             compare=["types", "nullable", "constraints", "rows", "defaults", "indexes"])


@template("10.12", "challenge")
def view_report(rng):
    db, f, hid = shop_db(rng, traps=("no_orders",))
    sol = ("CREATE VIEW country_report AS SELECT COALESCE(u.country, '??') AS country, COUNT(DISTINCT u.id) AS users, "
           "COUNT(o.id) AS orders, COALESCE(SUM(o.amount) FILTER (WHERE o.status = 'paid'), 0) AS paid FROM users u "
           "LEFT JOIN orders o ON o.user_id = u.id GROUP BY 1;\nCREATE INDEX idx_orders_user ON orders (user_id);")
    return Q("Create the view `country_report` with one row per country (unknown as `??`): `country`, `users`, "
             "`orders` (0 when none) and `paid` (paid revenue, 0 when none). Also create `idx_orders_user` on "
             "orders(user_id) so the view's join can use it.", sol, db, hid, check="state",
             tables=["country_report", "orders"], probes=["SELECT * FROM country_report"],
             compare=["indexes", "rows", "types"])

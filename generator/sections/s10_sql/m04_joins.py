"""10.04 Joins."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


@template("10.04", "lite")
def join_inner(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List every order's `amount` with its user's `name` (only orders, no users without orders).",
         "SELECT u.name, o.amount FROM users u INNER JOIN orders o ON o.user_id = u.id;", {"require": ["JOIN"]}),
        ("List each order's `id`, `status` and the `country` of the user who placed it.",
         "SELECT o.id, o.status, u.country FROM orders o JOIN users u ON u.id = o.user_id;", {"require": ["JOIN"]}),
        ("List `name` and `created_at` for every paid order.",
         "SELECT u.name, o.created_at FROM orders o JOIN users u ON u.id = o.user_id WHERE o.status = 'paid';"),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "lite")
def join_using(rng):
    db, f, hid = shop_db(rng, traps=("refunded_only",))
    items = [
        ("List every order's `id` and `amount` with the `name` of its user, joining with USING after renaming "
         "users.id: `FROM orders JOIN (SELECT id AS user_id, name FROM users) u USING (user_id)`.",
         "SELECT o.id, o.amount, u.name FROM orders o JOIN (SELECT id AS user_id, name FROM users) u USING (user_id);",
         {"require": ["USING"]}),
        ("List the `name` and `signup_date` of every user who placed a refunded order (one row per refunded order).",
         "SELECT u.name, u.signup_date FROM users u JOIN orders o ON o.user_id = u.id WHERE o.status = 'refunded';"),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "easy")
def join_left(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List every user's `name` with the `amount` of each of their orders; users without orders appear once with "
         "a NULL amount.", "SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id;",
         {"require": ["LEFT JOIN"]}),
        ("For every user show `name` and `n_orders` (0 for users without orders).",
         "SELECT u.name, COUNT(o.id) AS n_orders FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id, "
         "u.name;", {"require": ["LEFT JOIN"],
                     "note": "COUNT(*) would count the NULL-extended row of a user without orders as 1."}),
        ("For every user show `name` and `total_spent`, the sum of their orders, 0 when they never ordered.",
         "SELECT u.name, COALESCE(SUM(o.amount), 0) AS total_spent FROM users u LEFT JOIN orders o ON o.user_id = u.id "
         "GROUP BY u.id, u.name;", {"require": ["LEFT JOIN"]}),
        ("Write it the other way round: starting `FROM orders o`, keep every user (with the `amount` of each of "
         "their orders, NULL for users who never ordered) using a RIGHT JOIN. Show `name` and `amount`.",
         "SELECT u.name, o.amount FROM orders o RIGHT JOIN users u ON o.user_id = u.id;",
         {"require": ["RIGHT JOIN"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "easy")
def join_on_vs_where(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List every user's `name` with the `amount` of their paid orders; users without paid orders still appear "
         "(with NULL).", "SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id AND o.status = "
                         "'paid';", {"require": ["LEFT JOIN"],
                                     "note": "Where does `status = 'paid'` go so that users without paid orders "
                                             "stay?"}),
        ("For every user show `name` and `paid_total` (sum of paid orders, 0 if none).",
         "SELECT u.name, COALESCE(SUM(o.amount), 0) AS paid_total FROM users u LEFT JOIN orders o ON o.user_id = u.id "
         "AND o.status = 'paid' GROUP BY u.id, u.name;"),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "medium")
def join_anti_semi(rng):
    db, f, hid = shop_db(rng, traps=("no_orders",))
    x = rng.choice([100, 150])
    items = [
        ("List the `name` of the users who never ordered, with a LEFT JOIN ... IS NULL anti-join.",
         "SELECT u.name FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL;",
         {"require": ["LEFT JOIN"]}),
        ("List the `name` of the users who never ordered, with NOT EXISTS.",
         "SELECT u.name FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);",
         {"require": ["NOT EXISTS"]}),
        (f"List the `name` of the users with at least one order over {x} (each user once), with EXISTS.",
         f"SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > {x});",
         {"require": ["EXISTS"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "medium")
def join_self(rng):
    db, f, hid = shop_db(rng, n_users=8)
    items = [
        ("List every pair of users from the same country as `a` and `b` (names), each pair once (the one with the "
         "smaller id first).",
         "SELECT a.name AS a, b.name AS b FROM users a JOIN users b ON a.country = b.country AND a.id < b.id;"),
        ("List pairs of orders of the same user, `first_id` and `second_id`, where the second was placed after the "
         "first.", "SELECT a.id AS first_id, b.id AS second_id FROM orders a JOIN orders b ON a.user_id = b.user_id AND "
                   "b.created_at > a.created_at;"),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "medium")
def join_full_cross(rng):
    db, f, hid = shop_db(rng, n_users=4, n_orders=5)
    items = [
        ("Pair every user with every status (`paid`, `pending`, `refunded`): show `name` and `status` for all "
         "combinations (CROSS JOIN with a VALUES list).",
         "SELECT u.name, s.status FROM users u CROSS JOIN (VALUES ('paid'), ('pending'), ('refunded')) AS s(status);",
         {"require": ["CROSS JOIN"]}),
        ("List every user's `name` and every order's `id` matched by user, keeping users without orders and orders "
         "without users (FULL OUTER JOIN).",
         "SELECT u.name, o.id FROM users u FULL OUTER JOIN orders o ON o.user_id = u.id;", {"require": ["FULL"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "hard")
def join_fanout(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("For every user with orders show `name`, `n_orders`, `total` (sum of amounts) and `refunds` (number of "
         "refunded orders).",
         "SELECT u.name, COUNT(*) AS n_orders, SUM(o.amount) AS total, COUNT(*) FILTER (WHERE o.status = 'refunded') "
         "AS refunds FROM users u JOIN orders o ON o.user_id = u.id GROUP BY u.id, u.name;"),
        ("For every country (unknown countries as one group) show `country`, `buyers` (distinct users with orders) "
         "and `revenue`.", "SELECT u.country, COUNT(DISTINCT u.id) AS buyers, SUM(o.amount) AS revenue FROM users u "
                           "JOIN orders o ON o.user_id = u.id GROUP BY u.country;"),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "hard")
def join_lateral(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    items = [
        ("For every user show `name` and their most recent order's `amount` and `created_at` (NULLs when none; "
         "ties on time: higher id wins), with LEFT JOIN LATERAL.",
         "SELECT u.name, latest.amount, latest.created_at FROM users u LEFT JOIN LATERAL (SELECT o.amount, "
         "o.created_at FROM orders o WHERE o.user_id = u.id ORDER BY o.created_at DESC, o.id DESC LIMIT 1) latest ON "
         "TRUE;", {"require": ["LATERAL"]}),
        ("For every user with orders show `name` and their two largest orders' `amount` (two rows per user when "
         "possible; ties: lower id first), with JOIN LATERAL.",
         "SELECT u.name, t.amount FROM users u JOIN LATERAL (SELECT o.amount FROM orders o WHERE o.user_id = u.id "
         "ORDER BY o.amount DESC, o.id LIMIT 2) t ON TRUE;", {"require": ["LATERAL"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.04", "hard")
def join_three(rng):
    sql, f = D.runs(rng)
    hid = D.hidden(D.runs, rng, f, D.RUNS_TRAPS, k=3)
    items = [
        ("For every run show `run_id`, `model` and its final validation `loss` (the val metric with the highest "
         "step); runs without metrics are left out.",
         "SELECT r.run_id, r.model, m.loss FROM runs r JOIN metrics m ON m.run_id = r.run_id AND m.split = 'val' "
         "WHERE m.step = (SELECT MAX(m2.step) FROM metrics m2 WHERE m2.run_id = r.run_id AND m2.split = 'val');"),
        ("For every run show `run_id`, `owner` and `val_points`, the number of validation metrics (0 when none).",
         "SELECT r.run_id, r.owner, COUNT(m.step) AS val_points FROM runs r LEFT JOIN metrics m ON m.run_id = r.run_id "
         "AND m.split = 'val' GROUP BY r.run_id, r.owner;"),
    ]
    return drill(rng, items, sql, hid)


@template("10.04", "challenge")
def join_asof(rng):
    db, f, hid = shop_db(rng, traps=("ties", "no_orders"))
    sol = ("SELECT o.id AS order_id, o.user_id, o.amount, prev.id AS prev_order_id, prev.amount AS prev_amount FROM "
           "orders o LEFT JOIN LATERAL (SELECT p.id, p.amount FROM orders p WHERE p.user_id = o.user_id AND "
           "(p.created_at, p.id) < (o.created_at, o.id) ORDER BY p.created_at DESC, p.id DESC LIMIT 1) prev ON TRUE "
           "ORDER BY o.id;")
    return Q("For each order (`order_id`, `user_id`, `amount`) attach the same user's previous order (`prev_order_id`, "
             "`prev_amount`): the latest one placed strictly before it, where orders placed at the same moment count "
             "as earlier when their id is smaller. NULLs for a user's first order. By order id.", sol, db, hid,
             order=True)


@template("10.04", "challenge")
def join_overlap(rng):
    sql, f = D.gpu_jobs(rng, n_jobs=12, span_hours=10)
    hid = D.hidden(D.gpu_jobs, rng, f, D.GPU_TRAPS, k=3, n_jobs=12, span_hours=10)
    sol = ("SELECT a.job_id AS job_a, b.job_id AS job_b, a.gpu_type FROM gpu_jobs a JOIN gpu_jobs b ON a.gpu_type = "
           "b.gpu_type AND a.job_id < b.job_id AND a.started_at < b.ended_at AND b.started_at < a.ended_at ORDER BY "
           "job_a, job_b;")
    return Q("Two jobs on the same `gpu_type` conflict when their run intervals [started_at, ended_at) overlap. List "
             "each conflicting pair once as `job_a` < `job_b`, with the `gpu_type`; jobs that never started or never "
             "ended are not running intervals. By job_a, then job_b.", sol, sql, hid, order=True, dbname="shop")

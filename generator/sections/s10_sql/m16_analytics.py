"""10.16 Analytics & ML patterns."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


@template("10.16", "lite")
def ml_dedup(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    items = [
        ("Keep the latest order per user: all columns of each user's most recent order (ties on time: higher id), "
         "with ROW_NUMBER in a CTE.",
         "WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC, id DESC) AS rn "
         "FROM orders) SELECT id, user_id, amount, status, created_at FROM ranked WHERE rn = 1;",
         {"require": ["ROW_NUMBER("]}),
        ("Keep the latest order per user with DISTINCT ON: `user_id`, `id`, `created_at` (ties: higher id).",
         "SELECT DISTINCT ON (user_id) user_id, id, created_at FROM orders ORDER BY user_id, created_at DESC, id DESC;",
         {"require": ["DISTINCT ON"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "lite")
def ml_class_balance(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Class balance: for each `status` show `n` and `pct` (share of all orders in percent, 1 decimal) using "
         "SUM(COUNT(*)) OVER ().",
         "SELECT status, COUNT(*) AS n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct FROM orders GROUP BY "
         "status;", {"require": ["OVER"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "easy")
def ml_pivot(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Long to wide: for each `user_id` with orders show `paid`, `pending` and `refunded` totals (NULL when none) "
         "with FILTER.",
         "SELECT user_id, SUM(amount) FILTER (WHERE status = 'paid') AS paid, SUM(amount) FILTER (WHERE status = "
         "'pending') AS pending, SUM(amount) FILTER (WHERE status = 'refunded') AS refunded FROM orders GROUP BY "
         "user_id;", {"require": ["FILTER"]}),
        ("For each `user_id` with orders show `n_paid` and `n_other` (counts), with FILTER.",
         "SELECT user_id, COUNT(*) FILTER (WHERE status = 'paid') AS n_paid, COUNT(*) FILTER (WHERE status <> 'paid') "
         "AS n_other FROM orders GROUP BY user_id;"),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "easy")
def ml_split(rng):
    db, f, hid = shop_db(rng, n_users=10)
    items = [
        ("Assign a reproducible split to each user: `id` and `split`: `train` when ABS(HASHTEXT(id::TEXT)) % 10 < 8, "
         "`val` when it is 8, else `test`.",
         "SELECT id, CASE WHEN ABS(HASHTEXT(id::TEXT)) % 10 < 8 THEN 'train' WHEN ABS(HASHTEXT(id::TEXT)) % 10 = 8 "
         "THEN 'val' ELSE 'test' END AS split FROM users;", {"require": ["HASHTEXT("]}),
        ("Count users per split (`split`, `n`), with the split rule: ABS(HASHTEXT(id::TEXT)) % 10 < 7 is `train`, "
         "else `test`.",
         "SELECT CASE WHEN ABS(HASHTEXT(id::TEXT)) % 10 < 7 THEN 'train' ELSE 'test' END AS split, COUNT(*) AS n FROM "
         "users GROUP BY 1;", {"require": ["HASHTEXT("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "medium")
def ml_prior_features(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    items = [
        ("Leakage-free features: each order's `id` and `prior_orders`, the number of the same user's orders placed "
         "before it (by time, ties by id; 0 for the first).",
         "SELECT id, COUNT(*) OVER (PARTITION BY user_id ORDER BY created_at, id ROWS BETWEEN UNBOUNDED PRECEDING AND "
         "1 PRECEDING) AS prior_orders FROM orders;", {"require": ["PRECEDING"]}),
        ("Each order's `id` and `prev3_avg`: the average amount of the same user's 3 previous orders (not this one; "
         "by time, ties by id), rounded to 2 decimals; NULL for the first.",
         "SELECT id, ROUND(AVG(amount) OVER (PARTITION BY user_id ORDER BY created_at, id ROWS BETWEEN 3 PRECEDING AND "
         "1 PRECEDING), 2) AS prev3_avg FROM orders;", {"require": ["PRECEDING"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "medium")
def ml_time_since(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    items = [
        ("Each order's `id` and `since_prev`: the time since the same user's previous order (an interval; NULL for "
         "the first; by time, ties by id).",
         "SELECT id, created_at - LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at, id) AS since_prev "
         "FROM orders;", {"require": ["LAG("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "hard")
def ml_label(rng):
    db, f, hid = shop_db(rng, n_orders=18, traps=("ties",))
    d = rng.choice([30, 14])
    items = [
        (f"Label each order: `id` and `reordered_{d}d`: whether the same user ordered again after it, within {d} days "
         "(later than this order, at most created_at + the interval).",
         f"SELECT o.id, EXISTS (SELECT 1 FROM orders o2 WHERE o2.user_id = o.user_id AND o2.created_at > o.created_at "
         f"AND o2.created_at <= o.created_at + INTERVAL '{d} days') AS reordered_{d}d FROM orders o;",
         {"require": ["EXISTS"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "hard")
def ml_cohorts(rng):
    db, f, hid = shop_db(rng, n_users=9, n_orders=20)
    items = [
        ("Monthly cohort activity: `cohort` (month of a user's first order), `month` (DATE_TRUNC of each order) and "
         "`active_users` (distinct users), by cohort then month.",
         "WITH firsts AS (SELECT user_id, DATE_TRUNC('month', MIN(created_at)) AS cohort FROM orders GROUP BY user_id) "
         "SELECT f.cohort, DATE_TRUNC('month', o.created_at) AS month, COUNT(DISTINCT o.user_id) AS active_users FROM "
         "orders o JOIN firsts f USING (user_id) GROUP BY 1, 2 ORDER BY 1, 2;", {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.16", "hard")
def ml_data_tests(rng):
    db, f, hid = shop_db(rng)

    def dirty(r, k):
        ids = [o[0] for o in f["orders"]]
        dup = r.sample(ids, k)
        return ("CREATE TABLE staging AS SELECT * FROM orders;\n" +
                "".join(f"INSERT INTO staging SELECT * FROM orders WHERE id = {i};\n" for i in dup) +
                f"INSERT INTO staging VALUES ({9000 + k}, 9999, 5.00, 'lost', '2024-01-01 00:00');\n" +
                (f"INSERT INTO staging VALUES ({9100 + k}, 1, 7.00, 'shipped', '2024-01-02 00:00');\n" if k > 1 else ""))
    import random as _r
    seeds = [rng.randrange(1 << 30) for _ in range(4)]
    vis = db + dirty(_r.Random(seeds[0]), 1)
    hid2 = [(h + dirty(_r.Random(sd), k), t + ", other bad rows") for (h, t), sd, k in zip(hid, seeds[1:], (2, 3, 1))]
    items = [
        ("`staging` is a raw copy of orders. Data test (unique): return each `id` that appears more than once in "
         "staging, with its count `n` (no rows = the test passes).",
         "SELECT id, COUNT(*) AS n FROM staging GROUP BY id HAVING COUNT(*) > 1;"),
        ("`staging` is a raw copy of orders. Data test (accepted values): return `id` and `status` of the rows whose "
         "status isn't paid, pending or refunded.",
         "SELECT id, status FROM staging WHERE status NOT IN ('paid', 'pending', 'refunded');"),
        ("`staging` is a raw copy of orders. Data test (relationships): return the `id` of the rows whose user_id "
         "matches no user.", "SELECT s.id FROM staging s LEFT JOIN users u ON u.id = s.user_id WHERE u.id IS NULL;"),
    ]
    return drill(rng, items, vis, hid2)


@template("10.16", "challenge")
def ml_feature_table(rng):
    db, f, hid = shop_db(rng, n_orders=18, traps=("ties",))
    sol = ("SELECT o.id, o.user_id, COUNT(*) OVER w AS prior_orders, ROUND(AVG(o.amount) OVER w, 2) AS prior_avg, "
           "EXISTS (SELECT 1 FROM orders o2 WHERE o2.user_id = o.user_id AND o2.created_at > o.created_at AND "
           "o2.created_at <= o.created_at + INTERVAL '30 days') AS label FROM orders o WINDOW w AS (PARTITION BY "
           "o.user_id ORDER BY o.created_at, o.id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) ORDER BY o.id;")
    return Q("Build the training table in one query, by order id: `id`, `user_id`, `prior_orders` (the user's earlier "
             "orders, by time then id), `prior_avg` (their average amount, 2 decimals, NULL for a first order) and "
             "`label` (the user ordered again within 30 days after this order: later, at most +30 days).", sol, db,
             hid, order=True)


@template("10.16", "challenge")
def ml_retention_matrix(rng):
    db, f, hid = shop_db(rng, n_users=10, n_orders=22)
    sol = ("WITH firsts AS (SELECT user_id, DATE_TRUNC('month', MIN(created_at)) AS cohort FROM orders GROUP BY "
           "user_id), act AS (SELECT DISTINCT o.user_id, f.cohort, (EXTRACT(YEAR FROM AGE(DATE_TRUNC('month', "
           "o.created_at), f.cohort)) * 12 + EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', o.created_at), f.cohort))) AS "
           "k FROM orders o JOIN firsts f USING (user_id)) SELECT cohort::DATE AS cohort, COUNT(*) FILTER (WHERE k = "
           "0) AS m0, COUNT(*) FILTER (WHERE k = 1) AS m1, COUNT(*) FILTER (WHERE k = 2) AS m2 FROM act GROUP BY "
           "cohort ORDER BY 1;")
    return Q("Retention matrix: users belong to the cohort of the month of their first order. For each cohort (as a "
             "DATE) show `m0`, `m1`, `m2`: how many of its users ordered in the cohort month, one month later, two "
             "months later. By cohort.", sol, db, hid, order=True)

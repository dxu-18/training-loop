"""10.09 Math & numeric functions."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


def pred_db(rng, **kw):
    sql, f = D.predictions(rng, **kw)
    return sql, f, D.hidden(D.predictions, rng, f, D.PRED_TRAPS, k=3, n=kw.get("n", 16))


@template("10.09", "lite")
def math_rounding(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `rounded` (ROUND to 0 decimals), `up` (CEIL) and `down` (FLOOR) of the amount.",
         "SELECT id, ROUND(amount) AS rounded, CEIL(amount) AS up, FLOOR(amount) AS down FROM orders;",
         {"require": ["CEIL(", "FLOOR("]}),
        ("Show each order's `id` and `tens`: the amount rounded to the nearest 10 (ROUND with -1).",
         "SELECT id, ROUND(amount, -1) AS tens FROM orders;"),
        ("Show each order's `id` and `dist`: the absolute difference between its amount and 100.",
         "SELECT id, ABS(amount - 100) AS dist FROM orders;", {"require": ["ABS("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "lite")
def math_avg_round(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show the average order amount `avg_amount` rounded to 2 decimals.",
         "SELECT ROUND(AVG(amount), 2) AS avg_amount FROM orders;"),
        ("For each status show `status` and `avg_amount` rounded to 1 decimal.",
         "SELECT status, ROUND(AVG(amount), 1) AS avg_amount FROM orders GROUP BY status;"),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "easy")
def math_power_mod(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each user's `id`, `sq` (id squared with POWER) and `root` (SQRT of the id, rounded to 3 decimals).",
         "SELECT id, POWER(id, 2) AS sq, ROUND(SQRT(id)::NUMERIC, 3) AS root FROM users;", {"require": ["POWER(", "SQRT("]}),
        ("Show each order's `id` and `bucket`: id modulo 3.", "SELECT id, id % 3 AS bucket FROM orders;"),
        ("Show each order's `id` and `half`: the id divided by 2 with integer division, and `rest`: MOD(id, 2).",
         "SELECT id, id / 2 AS half, MOD(id, 2) AS rest FROM orders;", {"require": ["MOD("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "easy")
def math_percent(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show `pct_paid`: the percentage of orders that are paid, rounded to 1 decimal (watch integer division).",
         "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'paid') / COUNT(*), 1) AS pct_paid FROM orders;",
         {"note": "COUNT(*) / COUNT(*) is integer division: multiply by 100.0 first."}),
        ("Show `pct_refunded`: the percentage of orders that are refunded, rounded to 1 decimal.",
         "SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'refunded') / COUNT(*), 1) AS pct_refunded FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "medium")
def math_logs(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `log10_amount` (LOG, rounded to 4 decimals) and `ln_amount` (LN, rounded to 4).",
         "SELECT id, ROUND(LOG(amount), 4) AS log10_amount, ROUND(LN(amount), 4) AS ln_amount FROM orders;",
         {"require": ["LN("]}),
        ("Show each order's `id` and `trunc1`: the amount truncated (not rounded) to 1 decimal, and `sign_diff`: "
         "the SIGN of amount - 100.",
         "SELECT id, TRUNC(amount, 1) AS trunc1, SIGN(amount - 100) AS sign_diff FROM orders;", {"require": ["TRUNC("]}),
        ("Show each order's `id` and `growth`: EXP(1) times the amount, rounded to 2 decimals.",
         "SELECT id, ROUND(EXP(1)::NUMERIC * amount, 2) AS growth FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "medium")
def math_avg_trap(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show `avg_orders_per_user`: the number of orders divided by the number of users who ordered, as a decimal "
         "rounded to 2 places.",
         "SELECT ROUND(COUNT(*)::NUMERIC / COUNT(DISTINCT user_id), 2) AS avg_orders_per_user FROM orders;"),
        ("Show `share_big`: the fraction (0 to 1) of orders over 100, rounded to 3 decimals.",
         "SELECT ROUND(AVG(CASE WHEN amount > 100 THEN 1.0 ELSE 0 END), 3) AS share_big FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "hard")
def math_histogram(rng):
    db, f, hid = shop_db(rng, n_orders=18)
    k = rng.choice([5, 10])
    items = [
        (f"Make a {k}-bin histogram of amounts between 0 and 300: `bucket` (WIDTH_BUCKET) and `n`, by bucket.",
         f"SELECT WIDTH_BUCKET(amount, 0, 300, {k}) AS bucket, COUNT(*) AS n FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True, "require": ["WIDTH_BUCKET("]}),
        ("Show each order's `id` and `bucket50`: the lower edge of its 50-wide bucket (0, 50, 100 ...).",
         "SELECT id, FLOOR(amount / 50) * 50 AS bucket50 FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "hard")
def math_zscores(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    items = [
        ("Show each order's `id`, `amount` and `z`: (amount - average) / standard deviation over all orders, "
         "rounded to 3 decimals (window AVG and STDDEV).",
         "SELECT id, amount, ROUND((amount - AVG(amount) OVER ()) / STDDEV(amount) OVER (), 3) AS z FROM orders;",
         {"require": ["STDDEV("]}),
        ("Show the correlation `corr` between amount and the order's id (CORR, rounded to 4 decimals).",
         "SELECT ROUND(CORR(amount, id)::NUMERIC, 4) AS corr FROM orders;", {"require": ["CORR("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.09", "challenge")
def math_classifier(rng):
    db, f, hid = pred_db(rng)
    th = rng.choice(["0.5", "0.6", "0.4"])
    sol = (f"WITH c AS (SELECT model_version, COUNT(*) FILTER (WHERE score >= {th} AND label) AS tp, COUNT(*) FILTER "
           f"(WHERE score >= {th} AND NOT label) AS fp, COUNT(*) FILTER (WHERE score < {th} AND label) AS fn FROM "
           "predictions GROUP BY model_version) SELECT model_version, tp, fp, fn, ROUND(tp::NUMERIC / NULLIF(tp + fp, "
           "0), 3) AS precision, ROUND(tp::NUMERIC / NULLIF(tp + fn, 0), 3) AS recall, ROUND(2.0 * tp / NULLIF(2 * tp "
           "+ fp + fn, 0), 3) AS f1 FROM c ORDER BY model_version;")
    return Q(f"Evaluate each model version at threshold {th} (score >= {th} predicts true): `model_version`, `tp`, "
             "`fp`, `fn`, then `precision`, `recall` and `f1` rounded to 3 decimals (NULL when undefined). By "
             "model_version.", sol, db, hid, order=True)


@template("10.09", "challenge")
def math_calibration(rng):
    db, f, hid = pred_db(rng, n=20)
    sol = ("SELECT LEAST(WIDTH_BUCKET(score, 0, 1, 5), 5) AS bin, COUNT(*) AS n, ROUND(AVG(score), 3) AS avg_score, "
           "ROUND(AVG(CASE WHEN label THEN 1.0 ELSE 0.0 END), 3) AS hit_rate FROM predictions GROUP BY 1 ORDER BY 1;")
    return Q("Calibration table: put scores into 5 equal bins over [0, 1] with WIDTH_BUCKET, where a score of "
             "exactly 1 belongs to bin 5 (not 6). For each non-empty bin show `bin`, `n`, `avg_score` and "
             "`hit_rate` (share of true labels), both rounded to 3 decimals. By bin.", sol, db, hid, order=True,
             note="Hidden databases have scores of exactly 0.5 and 1.000.")

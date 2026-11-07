"""10.03 Aggregation: GROUP BY & HAVING."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


def runs_db(rng, **kw):
    sql, facts = D.runs(rng, **kw)
    return sql, facts, D.hidden(D.runs, rng, facts, D.RUNS_TRAPS, k=3, n_runs=kw.get("n_runs", 8))


@template("10.03", "lite")
def agg_totals(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Count the users (`n`).", "SELECT COUNT(*) AS n FROM users;", {"require": ["COUNT("]}),
        ("Show the total (`total`), average (`avg_amount`, rounded to 2 decimals), smallest (`min_amount`) and "
         "largest (`max_amount`) order amount.",
         "SELECT SUM(amount) AS total, ROUND(AVG(amount), 2) AS avg_amount, MIN(amount) AS min_amount, MAX(amount) AS "
         "max_amount FROM orders;"),
        ("Show the number of paid orders (`paid_orders`) and their total amount (`paid_total`).",
         "SELECT COUNT(*) AS paid_orders, SUM(amount) AS paid_total FROM orders WHERE status = 'paid';"),
        ("Show the date of the first signup (`first_signup`) and of the last (`last_signup`).",
         "SELECT MIN(signup_date) AS first_signup, MAX(signup_date) AS last_signup FROM users;"),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "lite")
def agg_count_kinds(rng):
    db, f, hid = shop_db(rng, traps=("null_country",))
    items = [
        ("Show three counts over users: all rows (`all_users`), users with a known country (`with_country`) and "
         "distinct known countries (`countries`).",
         "SELECT COUNT(*) AS all_users, COUNT(country) AS with_country, COUNT(DISTINCT country) AS countries FROM "
         "users;", {"require": ["COUNT(DISTINCT"]}),
        ("How many different users have placed orders? (`buyers`)",
         "SELECT COUNT(DISTINCT user_id) AS buyers FROM orders;", {"require": ["COUNT(DISTINCT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "easy")
def agg_group_by(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("For each `status`, show the number of orders `n` and their total `revenue`.",
         "SELECT status, COUNT(*) AS n, SUM(amount) AS revenue FROM orders GROUP BY status;", {"require": ["GROUP BY"]}),
        ("For each `user_id` that has orders, show `n_orders` and `total`.",
         "SELECT user_id, COUNT(*) AS n_orders, SUM(amount) AS total FROM orders GROUP BY user_id;"),
        ("For each `country` (unknown countries form one group), show the number of users `n`.",
         "SELECT country, COUNT(*) AS n FROM users GROUP BY country;"),
        ("For each `user_id`, show their largest order `biggest` and their latest order time `last_order`.",
         "SELECT user_id, MAX(amount) AS biggest, MAX(created_at) AS last_order FROM orders GROUP BY user_id;"),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "easy")
def agg_group_order(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("For each `status` show `n` (orders) and `revenue`, highest revenue first.",
         "SELECT status, COUNT(*) AS n, SUM(amount) AS revenue FROM orders GROUP BY status ORDER BY revenue DESC;",
         {"order": True}),
        ("For each `user_id` show `avg_amount` (rounded to 2 decimals), by user_id.",
         "SELECT user_id, ROUND(AVG(amount), 2) AS avg_amount FROM orders GROUP BY user_id ORDER BY user_id;",
         {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "medium")
def agg_having(rng):
    db, f, hid = shop_db(rng, n_orders=18)
    k = rng.choice([2, 3])
    x = rng.choice([100, 150, 200])
    items = [
        (f"Show the `user_id` and `n_orders` of the users with at least {k} orders.",
         f"SELECT user_id, COUNT(*) AS n_orders FROM orders GROUP BY user_id HAVING COUNT(*) >= {k};",
         {"require": ["HAVING"]}),
        (f"Show the `user_id` and `paid_total` of the users whose paid orders add up to more than {x}.",
         f"SELECT user_id, SUM(amount) AS paid_total FROM orders WHERE status = 'paid' GROUP BY user_id HAVING "
         f"SUM(amount) > {x};", {"require": ["HAVING"]}),
        ("Show the `status` values used by more than one user, with `users` (the number of distinct users).",
         "SELECT status, COUNT(DISTINCT user_id) AS users FROM orders GROUP BY status HAVING COUNT(DISTINCT user_id) "
         "> 1;"),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "medium")
def agg_filter_pivot(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("For each `user_id` with orders show `paid_orders` (count of paid orders) and `paid_revenue` (their total, 0 "
         "when none), using FILTER.",
         "SELECT user_id, COUNT(*) FILTER (WHERE status = 'paid') AS paid_orders, COALESCE(SUM(amount) FILTER (WHERE "
         "status = 'paid'), 0) AS paid_revenue FROM orders GROUP BY user_id;", {"require": ["FILTER"]}),
        ("For each `user_id` with orders show `paid`, `pending` and `refunded`: the total amount per status (0 when "
         "none), with SUM(CASE ...).",
         "SELECT user_id, SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) AS paid, SUM(CASE WHEN status = "
         "'pending' THEN amount ELSE 0 END) AS pending, SUM(CASE WHEN status = 'refunded' THEN amount ELSE 0 END) AS "
         "refunded FROM orders GROUP BY user_id;", {"require": ["CASE"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "medium")
def agg_strings_arrays(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show one row with `names`: all user names sorted alphabetically, joined with `, `.",
         "SELECT STRING_AGG(name, ', ' ORDER BY name) AS names FROM users;", {"require": ["STRING_AGG("]}),
        ("For each `user_id` with orders show `ids`, the array of their order ids in increasing order.",
         "SELECT user_id, ARRAY_AGG(id ORDER BY id) AS ids FROM orders GROUP BY user_id;", {"require": ["ARRAY_AGG("]}),
        ("For each `status` show `who`: the distinct user ids with orders of that status, sorted, joined with `;`.",
         "SELECT status, STRING_AGG(DISTINCT user_id::TEXT, ';' ORDER BY user_id::TEXT) AS who FROM orders GROUP BY "
         "status;"),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "medium")
def agg_by_month(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("For each month show `month` (the month start, DATE_TRUNC) and `revenue`, by month.",
         "SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS revenue FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True, "require": ["DATE_TRUNC("]}),
        ("For each month show `month` (DATE_TRUNC) and `orders`, the number of orders, by month.",
         "SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "hard")
def agg_stats(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    items = [
        ("Show the median order amount `median` (PERCENTILE_CONT) and the 90th percentile `p90`.",
         "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median, PERCENTILE_CONT(0.9) WITHIN GROUP "
         "(ORDER BY amount) AS p90 FROM orders;", {"require": ["PERCENTILE_CONT("]}),
        ("For each `status` show the median amount `median` and the standard deviation `sd` (rounded to 2 "
         "decimals; NULL for a single order).",
         "SELECT status, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median, ROUND(STDDEV(amount), 2) AS sd "
         "FROM orders GROUP BY status;"),
        ("Show `all_positive` (every amount > 0) and `any_refund` (some order refunded) with BOOL_AND / BOOL_OR.",
         "SELECT BOOL_AND(amount > 0) AS all_positive, BOOL_OR(status = 'refunded') AS any_refund FROM orders;",
         {"require": ["BOOL_AND(|BOOL_OR("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "hard")
def agg_rollup(rng):
    db, f, hid = runs_db(rng)
    items = [
        ("For each `project` and `status` show `n` runs, plus a subtotal per project and a grand total (ROLLUP). "
         "Subtotal rows have NULL in the rolled-up columns.",
         "SELECT project, status, COUNT(*) AS n FROM runs GROUP BY ROLLUP (project, status);", {"require": ["ROLLUP"]}),
        ("Show `n` runs for every combination of `optimizer` and `status`, every optimizer alone, every status alone "
         "and the total (CUBE).", "SELECT optimizer, status, COUNT(*) AS n FROM runs GROUP BY CUBE (optimizer, status);",
         {"require": ["CUBE"]}),
        ("Show `n` runs per `model` and, separately, per `dataset` in one result (GROUPING SETS), with the other "
         "column NULL.", "SELECT model, dataset, COUNT(*) AS n FROM runs GROUP BY GROUPING SETS ((model), (dataset));",
         {"require": ["GROUPING SETS"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.03", "challenge")
def agg_two_level(rng):
    db, f, hid = shop_db(rng, n_orders=18, traps=("refunded_only",))
    sol = ("SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*) AS orders, PERCENTILE_CONT(0.5) WITHIN GROUP "
           "(ORDER BY amount) AS median_amount, ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'refunded') / COUNT(*), "
           "1) AS refund_pct FROM orders GROUP BY 1 HAVING COUNT(*) >= 2 ORDER BY 1;")
    return Q("Monthly report: for each month with at least 2 orders, `month` (DATE_TRUNC), `orders`, "
             "`median_amount` (PERCENTILE_CONT) and `refund_pct` (refunded share of the month's orders in percent, 1 "
             "decimal). By month.", sol, db, hid, order=True)


@template("10.03", "challenge")
def agg_best_config(rng):
    db, f, hid = runs_db(rng, n_runs=12)
    sol = ("SELECT optimizer, COUNT(*) AS runs, COUNT(*) FILTER (WHERE status = 'finished') AS finished, "
           "ROUND(AVG(m.loss), 4) AS avg_val_loss FROM runs r JOIN metrics m USING (run_id) WHERE m.split = 'val' AND "
           "m.step = (SELECT MAX(step) FROM metrics) GROUP BY optimizer HAVING COUNT(*) FILTER (WHERE status = "
           "'finished') > 0 ORDER BY avg_val_loss, optimizer;" if False else
           "SELECT optimizer, COUNT(*) AS runs, COUNT(*) FILTER (WHERE status = 'finished') AS finished, "
           "ROUND(AVG(lr), 6) AS avg_lr, MIN(batch_size) AS min_batch FROM runs GROUP BY optimizer HAVING COUNT(*) "
           "FILTER (WHERE status = 'finished') >= 1 ORDER BY finished DESC, optimizer;")
    return Q("Summarise the sweep per `optimizer`: `runs` (all), `finished` (status finished), `avg_lr` (average "
             "learning rate, 6 decimals) and `min_batch` (smallest batch size). Only optimizers with at least one "
             "finished run; most finished first, then by name.", sol, db, hid, order=True)

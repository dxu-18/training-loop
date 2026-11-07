"""10.06 Window functions."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


@template("10.06", "lite")
def win_row_number(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `user_id`, `created_at` and `order_n`: its number among the user's orders by time "
         "(1 = first; ties by id).",
         "SELECT id, user_id, created_at, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at, id) AS order_n "
         "FROM orders;", {"require": ["OVER"]}),
        ("Show each order's `id`, `amount` and `pos`: its position when all orders are sorted by amount descending "
         "(ties by id).", "SELECT id, amount, ROW_NUMBER() OVER (ORDER BY amount DESC, id) AS pos FROM orders;",
         {"require": ["ROW_NUMBER("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "lite")
def win_share(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `amount` and `share`: its amount divided by the total of all orders, rounded to 4 "
         "decimals (SUM(...) OVER ()).",
         "SELECT id, amount, ROUND(amount / SUM(amount) OVER (), 4) AS share FROM orders;", {"require": ["OVER"]}),
        ("Show each order's `id`, `amount` and `user_total`: the total of all orders of the same user.",
         "SELECT id, amount, SUM(amount) OVER (PARTITION BY user_id) AS user_total FROM orders;",
         {"require": ["PARTITION BY"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "easy")
def win_ranks(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    items = [
        ("Show `id`, `amount` and `rnk`: the order's rank by amount, largest first, ties sharing a rank with a gap "
         "after (RANK).", "SELECT id, amount, RANK() OVER (ORDER BY amount DESC) AS rnk FROM orders;",
         {"require": ["RANK("]}),
        ("Show `id`, `amount` and `drnk`: the dense rank by amount, largest first (ties share, no gaps).",
         "SELECT id, amount, DENSE_RANK() OVER (ORDER BY amount DESC) AS drnk FROM orders;",
         {"require": ["DENSE_RANK("]}),
        ("Show `id`, `user_id`, `amount` and `rank_in_user`: rank of the amount within the user, largest first.",
         "SELECT id, user_id, amount, RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rank_in_user FROM "
         "orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "easy")
def win_lag_lead(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `user_id`, `amount` and `prev_amount`: the user's previous order amount by time "
         "(ties by id), NULL for the first.",
         "SELECT id, user_id, amount, LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at, id) AS prev_amount "
         "FROM orders;", {"require": ["LAG("]}),
        ("Show each order's `id` and `next_id`: the id of the same user's next order by time (ties by id).",
         "SELECT id, LEAD(id) OVER (PARTITION BY user_id ORDER BY created_at, id) AS next_id FROM orders;",
         {"require": ["LEAD("]}),
        ("Show each order's `id`, `amount` and `change`: amount minus the user's previous amount (by time, ties by "
         "id), NULL for the first.",
         "SELECT id, amount, amount - LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at, id) AS change FROM "
         "orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "medium")
def win_running(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id`, `created_at`, `amount` and `running_total`: the sum of all amounts up to and "
         "including this order, by time then id (ROWS frame).",
         "SELECT id, created_at, amount, SUM(amount) OVER (ORDER BY created_at, id ROWS BETWEEN UNBOUNDED PRECEDING "
         "AND CURRENT ROW) AS running_total FROM orders;", {"require": ["ROWS"]}),
        ("Show each order's `id`, `user_id` and `user_running`: the user's running total by time (ties by id).",
         "SELECT id, user_id, SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at, id ROWS BETWEEN UNBOUNDED "
         "PRECEDING AND CURRENT ROW) AS user_running FROM orders;"),
        ("Show each order's `id`, `amount` and `moving_avg_3`: the average of this and the 2 previous orders' "
         "amounts (by time, ties by id), rounded to 2 decimals.",
         "SELECT id, amount, ROUND(AVG(amount) OVER (ORDER BY created_at, id ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), "
         "2) AS moving_avg_3 FROM orders;", {"require": ["PRECEDING"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "medium")
def win_top_n(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    n = rng.choice([1, 2])
    items = [
        (f"Show all columns of each user's {n} largest order(s) (ties: lower id first), wrapping ROW_NUMBER in a CTE.",
         f"WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC, id) AS rn FROM "
         f"orders) SELECT id, user_id, amount, status, created_at FROM ranked WHERE rn <= {n};",
         {"require": ["ROW_NUMBER("]}),
        ("Show `user_id`, `id` and `created_at` of each user's first order (ties on time: lower id).",
         "WITH r AS (SELECT user_id, id, created_at, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at, id) "
         "AS rn FROM orders) SELECT user_id, id, created_at FROM r WHERE rn = 1;"),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "medium")
def win_named(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show `id`, `user_id`, `first_amount` (the user's first order amount by time, ties by id) and `quartile` "
         "(NTILE(4) over all orders by amount, ties by id), defining the user window once with WINDOW w AS (...).",
         "SELECT id, user_id, FIRST_VALUE(amount) OVER w AS first_amount, NTILE(4) OVER (ORDER BY amount, id) AS "
         "quartile FROM orders WINDOW w AS (PARTITION BY user_id ORDER BY created_at, id);", {"require": ["WINDOW"]}),
        ("Show `id`, `prev_amount` and `next_amount` (the user's previous / next order amount by time, ties by id) "
         "using a named window.",
         "SELECT id, LAG(amount) OVER w AS prev_amount, LEAD(amount) OVER w AS next_amount FROM orders WINDOW w AS "
         "(PARTITION BY user_id ORDER BY created_at, id);", {"require": ["WINDOW"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "hard")
def win_frames(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    items = [
        ("Show `id`, `user_id` and `last_amount`: the amount of the user's last order by time (ties by id) on every "
         "row of that user (LAST_VALUE needs a full frame).",
         "SELECT id, user_id, LAST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY created_at, id ROWS BETWEEN "
         "UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_amount FROM orders;", {"require": ["LAST_VALUE("]}),
        ("Show `id`, `amount` and `cum`: the running total over orders sorted by amount, where equal amounts are "
         "added together (the default RANGE frame).",
         "SELECT id, amount, SUM(amount) OVER (ORDER BY amount) AS cum FROM orders;"),
        ("Show `id`, `amount`, `pct_rank` (PERCENT_RANK by amount, rounded to 3 decimals) and `cume` (CUME_DIST by "
         "amount, rounded to 3).",
         "SELECT id, amount, ROUND(PERCENT_RANK() OVER (ORDER BY amount)::NUMERIC, 3) AS pct_rank, ROUND(CUME_DIST() "
         "OVER (ORDER BY amount)::NUMERIC, 3) AS cume FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "hard")
def win_daily(rng):
    db, f, hid = shop_db(rng, n_orders=18)
    items = [
        ("Show daily revenue with its change: `day` (created_at::DATE), `revenue` (that day's total) and `change` "
         "(revenue minus the previous day with orders), by day.",
         "SELECT day, revenue, revenue - LAG(revenue) OVER (ORDER BY day) AS change FROM (SELECT created_at::DATE AS "
         "day, SUM(amount) AS revenue FROM orders GROUP BY 1) d ORDER BY day;", {"order": True}),
        ("Show `month` (DATE_TRUNC) with `revenue` and `cumulative` (running total over months), by month.",
         "SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS revenue, SUM(SUM(amount)) OVER (ORDER BY "
         "DATE_TRUNC('month', created_at)) AS cumulative FROM orders GROUP BY 1 ORDER BY 1;", {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.06", "challenge")
def win_streaks(rng):
    sql, f = D.runs(rng, n_runs=12)
    hid = D.hidden(D.runs, rng, f, D.RUNS_TRAPS, k=3, n_runs=12)
    sol = ("WITH r AS (SELECT owner, started_at, status, ROW_NUMBER() OVER (PARTITION BY owner ORDER BY started_at) - "
           "ROW_NUMBER() OVER (PARTITION BY owner, status = 'finished' ORDER BY started_at) AS grp FROM runs), s AS "
           "(SELECT owner, grp, COUNT(*) AS len FROM r WHERE status = 'finished' GROUP BY owner, grp) SELECT owner, "
           "MAX(len) AS best_streak FROM s GROUP BY owner ORDER BY owner;")
    return Q("For each `owner` with at least one finished run, find `best_streak`: the longest run of consecutive "
             "finished runs in start order (a failed or running run breaks the streak). By owner. (Gaps and "
             "islands: the difference of two ROW_NUMBERs labels each streak.)", sol, sql, hid, order=True)


@template("10.06", "challenge")
def win_sessions(rng):
    sql, f = D.events(rng)
    hid = D.hidden(D.events, rng, f, D.EVENT_TRAPS, k=3)
    gap = rng.choice([30, 60])
    sol = (f"WITH e AS (SELECT user_id, ts, CASE WHEN ts - LAG(ts) OVER (PARTITION BY user_id ORDER BY ts, event_id) <= "
           f"INTERVAL '{gap} minutes' THEN 0 ELSE 1 END AS new_session FROM events), s AS (SELECT user_id, ts, "
           "SUM(new_session) OVER (PARTITION BY user_id ORDER BY ts ROWS UNBOUNDED PRECEDING) AS session FROM e) "
           "SELECT user_id, session, MIN(ts) AS started, COUNT(*) AS events FROM s GROUP BY user_id, session ORDER BY "
           "user_id, session;")
    return Q(f"Split each user's events into sessions: a new session starts when more than {gap} minutes passed since "
             "the user's previous event (by time, ties by event_id). Return `user_id`, `session` (1, 2, ... per user), "
             "`started` (first event time) and `events` (count). By user_id, then session.", sol, sql, hid,
             order=True)

"""10.08 Dates & times."""

from __future__ import annotations

import datetime as dt

from ._common import template, drill, shop_db, Q
from . import _data as D


@template("10.08", "lite")
def date_parts(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and `day`: the date of created_at (::DATE).",
         "SELECT id, created_at::DATE AS day FROM orders;"),
        ("Show each order's `id`, `y` (EXTRACT YEAR) and `m` (EXTRACT MONTH) of created_at.",
         "SELECT id, EXTRACT(YEAR FROM created_at) AS y, EXTRACT(MONTH FROM created_at) AS m FROM orders;",
         {"require": ["EXTRACT("]}),
        ("Show the `name` of the users who signed up before 2024-02-01 (a DATE literal).",
         "SELECT name FROM users WHERE signup_date < DATE '2024-02-01';"),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "lite")
def date_hours(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and `hour` (0-23) with DATE_PART.",
         "SELECT id, DATE_PART('hour', created_at) AS hour FROM orders;", {"require": ["DATE_PART("]}),
        ("Show the `id` of the orders placed in the morning (before 12:00).",
         "SELECT id FROM orders WHERE EXTRACT(HOUR FROM created_at) < 12;"),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "easy")
def date_trunc_interval(rng):
    db, f, hid = shop_db(rng)
    d = rng.choice([3, 7, 14, 30])
    items = [
        ("For each month show `month` (DATE_TRUNC to the month) and `orders` (count), by month.",
         "SELECT DATE_TRUNC('month', created_at) AS month, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True, "require": ["DATE_TRUNC("]}),
        (f"Show each order's `id` and `due`: created_at plus {d} days (INTERVAL).",
         f"SELECT id, created_at + INTERVAL '{d} days' AS due FROM orders;", {"require": ["INTERVAL"]}),
        ("Show each order's `id` and `days_after_signup`: days from the user's signup date to the order's date.",
         "SELECT o.id, o.created_at::DATE - u.signup_date AS days_after_signup FROM orders o JOIN users u ON u.id = "
         "o.user_id;"),
        ("For each week (DATE_TRUNC 'week', weeks start on Monday) show `week` and `revenue`, by week.",
         "SELECT DATE_TRUNC('week', created_at) AS week, SUM(amount) AS revenue FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "easy")
def date_minus(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and `ago`: how long before 2024-06-01 00:00 it was placed (an interval).",
         "SELECT id, TIMESTAMP '2024-06-01 00:00' - created_at AS ago FROM orders;"),
        ("Show each order's `id` and `minus_hour`: created_at minus one hour.",
         "SELECT id, created_at - INTERVAL '1 hour' AS minus_hour FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "medium")
def date_half_open(rng):
    db, f, hid = shop_db(rng, traps=("boundary",))
    month = rng.choice([2, 3])
    nm = {2: "February", 3: "March"}[month]
    items = [
        (f"Show the total `revenue` of {nm} 2024 (all moments of the month, half-open range).",
         f"SELECT SUM(amount) AS revenue FROM orders WHERE created_at >= DATE '2024-{month:02d}-01' AND created_at < "
         f"DATE '2024-{month + 1:02d}-01';"),
        ("For each day of the week show `dow` (0 = Sunday) and `orders`, by dow.",
         "SELECT EXTRACT(DOW FROM created_at) AS dow, COUNT(*) AS orders FROM orders GROUP BY 1 ORDER BY 1;",
         {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "medium")
def date_to_char_age(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and `stamp` formatted like `2024-03-01 19:45` (TO_CHAR).",
         "SELECT id, TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI') AS stamp FROM orders;", {"require": ["TO_CHAR("]}),
        ("Show each order's `id` and `month_name`: the month as `Mar 2024` (TO_CHAR with Mon YYYY).",
         "SELECT id, TO_CHAR(created_at, 'Mon YYYY') AS month_name FROM orders;", {"require": ["TO_CHAR("]}),
        ("Show each user's `name` and `account_age`: AGE from signup to 2024-06-01 (as an interval like `3 mons 4 "
         "days`).", "SELECT name, AGE(DATE '2024-06-01', signup_date) AS account_age FROM users;",
         {"require": ["AGE("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "medium")
def date_spine(rng):
    db, f, hid = shop_db(rng)
    start = rng.choice([dt.date(2024, 2, 1), dt.date(2024, 2, 10), dt.date(2024, 3, 1)])
    n = rng.choice([5, 7])
    end = start + dt.timedelta(days=n - 1)
    items = [
        (f"List every day from {start} to {end} as `day` (GENERATE_SERIES), in order.",
         f"SELECT d::DATE AS day FROM GENERATE_SERIES(DATE '{start}', DATE '{end}', INTERVAL '1 day') AS d;",
         {"order": True, "require": ["GENERATE_SERIES("]}),
        (f"For every day from {start} to {end} show `day` and `orders` (0 when there were none), in order.",
         f"SELECT d::DATE AS day, COUNT(o.id) AS orders FROM GENERATE_SERIES(DATE '{start}', DATE '{end}', INTERVAL "
         "'1 day') AS d LEFT JOIN orders o ON o.created_at::DATE = d::DATE GROUP BY 1 ORDER BY 1;",
         {"order": True, "require": ["GENERATE_SERIES("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "hard")
def date_zones(rng):
    db, f, hid = shop_db(rng)
    zone = rng.choice(["America/New_York", "Asia/Tokyo", "Europe/Berlin"])
    items = [
        (f"created_at is stored in UTC. Show each order's `id` and `local_time` in {zone} (AT TIME ZONE twice).",
         f"SELECT id, created_at AT TIME ZONE 'UTC' AT TIME ZONE '{zone}' AS local_time FROM orders;",
         {"require": ["AT TIME ZONE"]}),
        (f"created_at is UTC. For each local day in {zone} show `local_day` and `orders`, by day.",
         f"SELECT (created_at AT TIME ZONE 'UTC' AT TIME ZONE '{zone}')::DATE AS local_day, COUNT(*) AS orders FROM "
         "orders GROUP BY 1 ORDER BY 1;", {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "hard")
def date_parse(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each user's `name` and `signup_text`: the signup date written like `15/03/2024`, and `back`: that "
         "text parsed again with TO_DATE.",
         "SELECT name, TO_CHAR(signup_date, 'DD/MM/YYYY') AS signup_text, TO_DATE(TO_CHAR(signup_date, 'DD/MM/YYYY'), "
         "'DD/MM/YYYY') AS back FROM users;", {"require": ["TO_DATE("]}),
        ("For every user show `name` and `first_order_days`: days from signup to their first order (NULL when "
         "none).", "SELECT u.name, MIN(o.created_at)::DATE - u.signup_date AS first_order_days FROM users u LEFT JOIN "
                   "orders o ON o.user_id = u.id GROUP BY u.id, u.name, u.signup_date;"),
        ("Show the `revenue` of the 30 days ending 2024-03-31 (from 2024-03-02 00:00 up to, not including, "
         "2024-04-01).", "SELECT SUM(amount) AS revenue FROM orders WHERE created_at >= DATE '2024-04-01' - INTERVAL "
                         "'30 days' AND created_at < DATE '2024-04-01';"),
    ]
    return drill(rng, items, db, hid)


@template("10.08", "challenge")
def date_weekly_zero(rng):
    db, f, hid = shop_db(rng, n_orders=12)
    sol = ("SELECT w::DATE AS week, COALESCE(SUM(o.amount), 0) AS revenue FROM GENERATE_SERIES(DATE_TRUNC('week', "
           "TIMESTAMP '2024-01-01'), DATE_TRUNC('week', TIMESTAMP '2024-03-31'), INTERVAL '1 week') AS w LEFT JOIN "
           "orders o ON DATE_TRUNC('week', o.created_at) = w AND o.status = 'paid' GROUP BY 1 ORDER BY 1;")
    return Q("Weekly paid revenue for Q1 2024, with empty weeks shown as 0: `week` (the Monday, as a DATE) and "
             "`revenue`, for every week from the week of 2024-01-01 to the week of 2024-03-31. By week.", sol, db,
             hid, order=True)


@template("10.08", "challenge")
def date_retention(rng):
    db, f, hid = shop_db(rng, n_users=9, n_orders=18)
    sol = ("SELECT DATE_TRUNC('month', u.signup_date)::DATE AS cohort, COUNT(*) AS users, COUNT(*) FILTER (WHERE "
           "EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.created_at < u.signup_date + INTERVAL '30 "
           "days')) AS ordered_30d FROM users u GROUP BY 1 ORDER BY 1;")
    return Q("Signup cohorts by month: `cohort` (the month of the signup date, as a DATE), `users` in it, and "
             "`ordered_30d`: how many of them placed an order within 30 days of signing up (before signup + 30 "
             "days). By cohort.", sol, db, hid, order=True)

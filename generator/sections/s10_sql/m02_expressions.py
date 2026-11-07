"""10.02 Expressions: CASE, NULL handling & casting."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


def camp_db(rng, **kw):
    sql, facts = D.campaigns(rng, **kw)
    return sql, facts, D.hidden(D.campaigns, rng, facts, D.CAMPAIGN_TRAPS, k=3, n=kw.get("n", 8))


@template("10.02", "lite")
def expr_arithmetic(rng):
    db, f, hid = shop_db(rng)
    rate = rng.choice(["1.2", "1.19", "1.07", "1.1"])
    disc = rng.choice(["0.9", "0.85", "0.8"])
    items = [
        (f"For every order show `id`, `amount` and the amount with tax as `with_tax` (amount times {rate}).",
         f"SELECT id, amount, amount * {rate} AS with_tax FROM orders;"),
        (f"For every order show `id` and the discounted amount `discounted` (amount times {disc}, rounded to 2 "
         "decimals).", f"SELECT id, ROUND(amount * {disc}, 2) AS discounted FROM orders;", {"require": ["ROUND("]}),
        ("For every order show `id`, `amount` and `cents`, the amount in cents.",
         "SELECT id, amount, amount * 100 AS cents FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "lite")
def expr_coalesce(rng):
    db, f, hid = shop_db(rng, traps=("null_country",))
    word = rng.choice(["unknown", "n/a", "??"])
    items = [
        (f"List each user's `name` and `country`, showing `{word}` when the country is missing.",
         f"SELECT name, COALESCE(country, '{word}') AS country FROM users;", {"require": ["COALESCE("]}),
        ("List each user's `name` and `region`: the country, or `XX` when it is missing.",
         "SELECT name, COALESCE(country, 'XX') AS region FROM users;", {"require": ["COALESCE("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "easy")
def expr_case(rng):
    db, f, hid = shop_db(rng)
    big = rng.choice([100, 150, 200])
    mid = rng.choice([30, 40, 50])
    items = [
        (f"Show each order's `id`, `amount` and `size`: `large` from {big} up, `medium` from {mid} up, else `small`.",
         f"SELECT id, amount, CASE WHEN amount >= {big} THEN 'large' WHEN amount >= {mid} THEN 'medium' ELSE 'small' "
         "END AS size FROM orders;", {"require": ["CASE"]}),
        ("Show each order's `id` and `refund_flag`: 1 for refunded orders, else 0.",
         "SELECT id, CASE WHEN status = 'refunded' THEN 1 ELSE 0 END AS refund_flag FROM orders;",
         {"require": ["CASE"]}),
        ("Show each order's `id` and `label`: `ok` for paid, `wait` for pending, `back` for refunded (use the simple "
         "CASE form: CASE status WHEN ...).",
         "SELECT id, CASE status WHEN 'paid' THEN 'ok' WHEN 'pending' THEN 'wait' WHEN 'refunded' THEN 'back' END AS "
         "label FROM orders;", {"require": ["CASE status"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "easy")
def expr_casts(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and its amount rounded to a whole number with a cast to INTEGER, as `whole`.",
         "SELECT id, CAST(amount AS INTEGER) AS whole FROM orders;", {"require": ["CAST|::"]}),
        ("Show each order's `id` and `day`, the date part of `created_at` (use ::DATE).",
         "SELECT id, created_at::DATE AS day FROM orders;", {"require": ["::"]}),
        ("Show each user's `label`: their name, a space and their id in parentheses, e.g. `Ana (1)`.",
         "SELECT name || ' (' || id || ')' AS label FROM users;", {"require": ["||"]}),
        ("Show each order's `id` and `status_or_null`: the status, but NULL instead of `pending`.",
         "SELECT id, NULLIF(status, 'pending') AS status_or_null FROM orders;", {"require": ["NULLIF("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "medium")
def expr_division(rng):
    db, f, hid = shop_db(rng)
    total = sum(o[2] for o in f["orders"])
    items = [
        ("Show each order's `id` and `pct`: its share of the sum of all amounts, in percent, rounded to 2 decimals. "
         f"(The amounts sum to {total}; write the total as a literal or with a subquery.)",
         f"SELECT id, ROUND(100.0 * amount / {total}, 2) AS pct FROM orders;", {"require": ["ROUND("]}),
        ("Show each user's `id` and `half_id`: the id divided by 2 with the fraction kept (5 gives 2.5000000000000000).",
         "SELECT id, id / 2.0 AS half_id FROM users;",
         {"note": "Integer division truncates: 5 / 2 is 2. Make one side numeric."}),
        ("Show each user's `id` and `id_pct`: 100 times id divided by the number of users (as a decimal, rounded to 1 "
         f"decimal). There are {len(f['users'])} users.",
         f"SELECT id, ROUND(100.0 * id / {len(f['users'])}, 1) AS id_pct FROM users;"),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "medium")
def expr_null_logic(rng):
    db, f, hid = shop_db(rng, traps=("null_country",))
    items = [
        ("Show each user's `name` and `is_us`: whether the country is `US` as true/false, false (not NULL) when the "
         "country is unknown.", "SELECT name, COALESCE(country = 'US', false) AS is_us FROM users;"),
        ("Show each user's `name` and `has_country`: true when the country is known, else false.",
         "SELECT name, country IS NOT NULL AS has_country FROM users;"),
        ("Show each order's `id` and `capped`: the amount, but at most 100 (use LEAST).",
         "SELECT id, LEAST(amount, 100) AS capped FROM orders;", {"require": ["LEAST("]}),
        ("Show each order's `id` and `floor10`: the amount, but at least 10 (use GREATEST).",
         "SELECT id, GREATEST(amount, 10) AS floor10 FROM orders;", {"require": ["GREATEST("]}),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "hard")
def expr_safe_division(rng):
    db, f, hid = camp_db(rng)
    items = [
        ("For each campaign show `name` and `ctr`: clicks divided by impressions as a decimal rounded to 4 places; "
         "NULL when there were no impressions.",
         "SELECT name, ROUND(clicks::NUMERIC / NULLIF(impressions, 0), 4) AS ctr FROM campaigns;",
         {"require": ["NULLIF("]}),
        ("For each campaign show `name` and `cpc`: spend per click rounded to 2 decimals, NULL when there were no "
         "clicks.", "SELECT name, ROUND(spend / NULLIF(clicks, 0), 2) AS cpc FROM campaigns;",
         {"require": ["NULLIF("]}),
        ("For each campaign show `name` and `conv_rate`: conversions per click in percent, rounded to 1 decimal; 0 "
         "when conversions are unknown; NULL when there were no clicks.",
         "SELECT name, ROUND(100.0 * COALESCE(conversions, 0) / NULLIF(clicks, 0), 1) AS conv_rate FROM campaigns;"),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "hard")
def expr_case_nulls(rng):
    db, f, hid = camp_db(rng)
    items = [
        ("Show each campaign's `name` and `funnel`: `no traffic` when impressions are 0, `no clicks` when clicks "
         "are 0, `unknown` when conversions are unknown, `converting` when conversions > 0, else `not converting`.",
         "SELECT name, CASE WHEN impressions = 0 THEN 'no traffic' WHEN clicks = 0 THEN 'no clicks' WHEN conversions "
         "IS NULL THEN 'unknown' WHEN conversions > 0 THEN 'converting' ELSE 'not converting' END AS funnel FROM "
         "campaigns;", {"require": ["CASE"]}),
        ("Show each campaign's `name` and `budget_band` from its spend: `none` for 0, `low` below 100, `mid` below "
         "300, `high` otherwise.",
         "SELECT name, CASE WHEN spend = 0 THEN 'none' WHEN spend < 100 THEN 'low' WHEN spend < 300 THEN 'mid' ELSE "
         "'high' END AS budget_band FROM campaigns;"),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "hard")
def expr_text_casts(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show each order's `id` and `ym`, its year and month as text like `2024-03` (cast the date to text and "
         "take the first 7 characters).",
         "SELECT id, LEFT(created_at::DATE::TEXT, 7) AS ym FROM orders;", {"require": ["::"]}),
        ("Show each order's `id` and `amount_text`: the amount as text followed by ` USD`.",
         "SELECT id, amount::TEXT || ' USD' AS amount_text FROM orders;"),
        ("Show each user's `name` and `days_since`: the number of days from their signup date to 2024-06-01 "
         "(date minus date gives an integer).",
         "SELECT name, DATE '2024-06-01' - signup_date AS days_since FROM users;"),
    ]
    return drill(rng, items, db, hid)


@template("10.02", "challenge")
def expr_lifecycle(rng):
    db, f, hid = shop_db(rng, traps=("refunded_only",))
    cut = f["start"].replace(month=f["start"].month + 1)
    sol = ("SELECT id, user_id, amount, CASE WHEN status = 'refunded' THEN 'refund' WHEN status = 'pending' AND "
           f"created_at < '{cut.isoformat()}' THEN 'stale' WHEN status = 'pending' THEN 'open' WHEN amount >= 100 "
           "THEN 'vip' ELSE 'normal' END AS bucket, ROUND(CASE WHEN status = 'refunded' THEN -amount ELSE amount "
           "END, 0) AS signed_whole FROM orders ORDER BY id;")
    return Q("Finance wants a bucket per order (`id`, `user_id`, `amount`, `bucket`, `signed_whole`), by id: "
             "refunded orders are `refund`; pending orders placed before "
             f"{cut.isoformat()} are `stale`, later pending ones `open`; other orders are `vip` from 100 up, else "
             "`normal`. `signed_whole` is the amount rounded to a whole number, negative for refunds.",
             sol, db, hid, order=True)


@template("10.02", "challenge")
def expr_pricing(rng):
    db, f, hid = camp_db(rng)
    sol = ("SELECT name, impressions, clicks, CASE WHEN impressions = 0 THEN NULL ELSE ROUND(100.0 * clicks / "
           "impressions, 2) END AS ctr_pct, ROUND(spend * CASE channel WHEN 'search' THEN 1.0 WHEN 'social' THEN 0.9 "
           "WHEN 'video' THEN 0.8 ELSE 0.5 END, 2) AS billed, COALESCE(conversions, 0) > 0 AS converted FROM "
           "campaigns ORDER BY billed DESC, name;")
    return Q("Build the billing sheet: `name`, `impressions`, `clicks`, `ctr_pct` (clicks per impression in percent, "
             "2 decimals, NULL without impressions), `billed` (spend times the channel's rate: search 1.0, social "
             "0.9, video 0.8, anything else 0.5, rounded to 2 decimals) and `converted` (true when conversions are "
             "known and above 0). Highest bill first, then by name.", sol, db, hid, order=True)

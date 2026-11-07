"""10.01 SELECT: filtering, sorting & limiting (shop: users, orders)."""

from __future__ import annotations

import datetime as dt

from ._common import template, drill, shop_db, Q, fmt_money
from . import _data as D


def _params(rng, f):
    amounts = sorted(o[2] for o in f["orders"])
    mid = amounts[len(amounts) // 2]
    x = int(mid)
    countries = sorted({u[2] for u in f["users"] if u[2]})
    c = rng.choice(countries)
    c2 = rng.choice([k for k in countries if k != c] or countries)
    letter = rng.choice(sorted({u[1][0] for u in f["users"]}))
    return x, c, c2, letter


@template("10.01", "lite")
def select_columns(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show every column of every user.", "SELECT * FROM users;"),
        ("List each user's `name` and `country`.", "SELECT name, country FROM users;"),
        ("List each user's name as `customer` and signup date as `joined`.",
         "SELECT name AS customer, signup_date AS joined FROM users;"),
        ("List each order's `id`, `amount` and `status`.", "SELECT id, amount, status FROM orders;"),
        ("List each order's id as `order_id` and its amount as `total`.",
         "SELECT id AS order_id, amount AS total FROM orders;"),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "lite")
def select_where(rng):
    db, f, hid = shop_db(rng)
    x, c, c2, letter = _params(rng, f)
    items = [
        (f"Show all columns of the orders with an amount over {x}.", f"SELECT * FROM orders WHERE amount > {x};"),
        ("Show all columns of the orders whose status is `paid`.", "SELECT * FROM orders WHERE status = 'paid';"),
        (f"List the `name` of the users from `{c}`.", f"SELECT name FROM users WHERE country = '{c}';"),
        (f"Show all columns of the orders of at most {x}.", f"SELECT * FROM orders WHERE amount <= {x};"),
        ("List the `id` and `amount` of the orders that are not `paid`.",
         "SELECT id, amount FROM orders WHERE status <> 'paid';"),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "lite")
def select_distinct(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List each different `status` an order has, once.", "SELECT DISTINCT status FROM orders;",
         {"require": ["DISTINCT"]}),
        ("List each different `country` users come from, once (a missing country counts as one value).",
         "SELECT DISTINCT country FROM users;", {"require": ["DISTINCT"]}),
        ("List each different (`user_id`, `status`) pair found in orders, once.",
         "SELECT DISTINCT user_id, status FROM orders;", {"require": ["DISTINCT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "easy")
def select_filters(rng):
    db, f, hid = shop_db(rng)
    x, c, c2, letter = _params(rng, f)
    lo, hi = max(1, x - 60), x + 40
    items = [
        (f"Show all columns of the paid orders with an amount between {lo} and {hi} (both included).",
         f"SELECT * FROM orders WHERE status = 'paid' AND amount BETWEEN {lo} AND {hi};", {"require": ["BETWEEN"]}),
        (f"List the `name` and `country` of the users from `{c}` or `{c2}`, using IN.",
         f"SELECT name, country FROM users WHERE country IN ('{c}', '{c2}');", {"require": ["IN"]}),
        (f"List the `name` of the users whose name starts with `{letter}`.",
         f"SELECT name FROM users WHERE name LIKE '{letter}%';", {"require": ["LIKE"]}),
        ("List the `name` of the users whose country is missing.",
         "SELECT name FROM users WHERE country IS NULL;", {"require": ["IS NULL"]}),
        ("Show the `id` and `status` of the orders that are refunded or pending.",
         "SELECT id, status FROM orders WHERE status IN ('refunded', 'pending');"),
        ("List the `name` and `country` of the users whose country is known.",
         "SELECT name, country FROM users WHERE country IS NOT NULL;", {"require": ["IS NOT NULL"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "easy")
def select_order_limit(rng):
    db, f, hid = shop_db(rng)
    n = rng.choice([2, 3, 4])
    items = [
        ("Show all columns of all orders, largest amount first (ties: lower id first).",
         "SELECT * FROM orders ORDER BY amount DESC, id;", {"order": True}),
        (f"Show all columns of the {n} most recent orders, newest first.",
         f"SELECT * FROM orders ORDER BY created_at DESC, id DESC LIMIT {n};", {"order": True, "require": ["LIMIT"]}),
        ("List users' `name` and `signup_date`, earliest signup first (ties: by name).",
         "SELECT name, signup_date FROM users ORDER BY signup_date, name;", {"order": True}),
        (f"List the `id` and `amount` of the {n} smallest orders, smallest first (ties: lower id first).",
         f"SELECT id, amount FROM orders ORDER BY amount, id LIMIT {n};", {"order": True}),
        ("Show all columns of the paid orders over 50, newest first (ties: higher id first).",
         "SELECT * FROM orders WHERE status = 'paid' AND amount > 50 ORDER BY created_at DESC, id DESC;",
         {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "medium")
def select_null_safe(rng):
    db, f, hid = shop_db(rng)
    x, c, c2, letter = _params(rng, f)
    items = [
        (f"List the `name` of every user whose country is not `{c}`, including users whose country is unknown.",
         f"SELECT name FROM users WHERE country IS DISTINCT FROM '{c}';",
         {"note": "Careful: `country <> '...'` drops the users with a NULL country (NULL <> 'x' is unknown)."}),
        (f"List the `name` of the users whose country is not `{c}` and is known.",
         f"SELECT name FROM users WHERE country <> '{c}';"),
        (f"List the `name` of every user whose country is neither `{c}` nor `{c2}` (unknown countries included).",
         f"SELECT name FROM users WHERE country NOT IN ('{c}', '{c2}') OR country IS NULL;"),
        (f"List the `name` of the users whose country is `{c}` or unknown, sorted by name.",
         f"SELECT name FROM users WHERE country = '{c}' OR country IS NULL ORDER BY name;", {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "medium")
def select_sorting(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List `name` and `country` of all users sorted by country, users with an unknown country last, then by "
         "name.", "SELECT name, country FROM users ORDER BY country NULLS LAST, name;", {"order": True}),
        ("List `name` and `country` of all users sorted by country descending with unknown countries last, then by "
         "name.", "SELECT name, country FROM users ORDER BY country DESC NULLS LAST, name;", {"order": True}),
        ("List `name` and `country` of all users with unknown countries first, then by country, then name.",
         "SELECT name, country FROM users ORDER BY country NULLS FIRST, name;", {"order": True}),
        ("Show `id`, `status` and `amount` of all orders sorted by status, then largest amount first, then id.",
         "SELECT id, status, amount FROM orders ORDER BY status, amount DESC, id;", {"order": True}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "medium")
def select_paging(rng):
    db, f, hid = shop_db(rng, n_orders=rng.randint(13, 18))
    per = rng.choice([2, 3])
    page = 2 if per == 3 else rng.choice([2, 3])
    items = [
        (f"Orders are shown {per} per page, sorted by id. Show all columns of page {page}.",
         f"SELECT * FROM orders ORDER BY id LIMIT {per} OFFSET {per * (page - 1)};",
         {"order": True, "require": ["OFFSET"]}),
        (f"Users are shown {per} per page, sorted by name then id. List the `id` and `name` on page {page}.",
         f"SELECT id, name FROM users ORDER BY name, id LIMIT {per} OFFSET {per * (page - 1)};",
         {"order": True, "require": ["OFFSET"]}),
        (f"Show all columns of page {page} of the orders, newest first ({per} per page, ties by id), using the "
         "standard OFFSET ... ROWS FETCH FIRST ... ROWS ONLY form.",
         f"SELECT * FROM orders ORDER BY created_at DESC, id OFFSET {per * (page - 1)} ROWS FETCH FIRST {per} ROWS "
         "ONLY;", {"order": True, "require": ["FETCH"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "medium")
def select_patterns(rng):
    db, f, hid = shop_db(rng, traps=("lower_case",))
    x, c, c2, letter = _params(rng, f)
    names = [u[1] for u in f["users"]]
    second = rng.choice(sorted({n[1] for n in names if len(n) > 1}))
    three = sorted({n[-1] for n in names if len(n) == 3}) or [names[0][-1]]
    last = rng.choice(three)
    items = [
        (f"List the `name` of the users whose name starts with `{letter}`, ignoring case.",
         f"SELECT name FROM users WHERE name ILIKE '{letter}%';", {"require": ["ILIKE"]}),
        (f"List the `name` of the users whose name has `{second}` as its second letter.",
         f"SELECT name FROM users WHERE name LIKE '_{second}%';", {"require": ["LIKE"]}),
        (f"List the `name` of the users whose name is exactly three characters long and ends with `{last}`.",
         f"SELECT name FROM users WHERE name LIKE '__{last}';"),
        ("List the `name` of the users whose name doesn't contain an `a` (in any case).",
         "SELECT name FROM users WHERE name NOT ILIKE '%a%';", {"require": ["ILIKE"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "hard")
def select_precedence(rng):
    db, f, hid = shop_db(rng)
    x, c, c2, letter = _params(rng, f)
    pair = [u for u in f["users"] if u[2] in (c, c2)]
    cut = max(u[3] for u in pair) if pair else f["start"]
    items = [
        (f"Show all columns of the orders over {x} that are paid or pending.",
         f"SELECT * FROM orders WHERE amount > {x} AND (status = 'paid' OR status = 'pending');",
         {"note": "AND binds tighter than OR: without parentheses this asks for something else."}),
        (f"Show all columns of the orders that are refunded, or that are paid and under {x}.",
         f"SELECT * FROM orders WHERE status = 'refunded' OR (status = 'paid' AND amount < {x});"),
        (f"List the `name` of users from `{c}` or `{c2}` who signed up on or before {cut.isoformat()}.",
         f"SELECT name FROM users WHERE (country = '{c}' OR country = '{c2}') AND signup_date <= "
         f"'{cut.isoformat()}';"),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "hard")
def select_time_ranges(rng):
    db, f, hid = shop_db(rng, traps=("boundary",))
    month = rng.choice([2, 3])
    nxt = month + 1
    name = {2: "February", 3: "March"}[month]
    day = rng.choice(sorted({o[4].date() for o in f["orders"]}))
    items = [
        (f"Show all columns of the orders placed in {name} 2024 (any time of day), oldest first (ties: id).",
         f"SELECT * FROM orders WHERE created_at >= '2024-{month:02d}-01' AND created_at < '2024-{nxt:02d}-01' "
         "ORDER BY created_at, id;", {"order": True,
                                      "note": "Hint: a half-open range (>= the first day, < the next month) "
                                              "includes every moment of the month and nothing after."}),
        (f"Show the `id` and `created_at` of the orders placed on {day.isoformat()} (the whole day).",
         f"SELECT id, created_at FROM orders WHERE created_at >= '{day.isoformat()}' AND created_at < "
         f"'{(day + dt.timedelta(days=1)).isoformat()}';"),
        (f"Show all columns of the orders placed before {name} 2024.",
         f"SELECT * FROM orders WHERE created_at < '2024-{month:02d}-01';"),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "hard")
def select_with_ties(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    n = rng.choice([2, 3])
    items = [
        (f"Show all columns of the {n} largest orders, largest first, including any orders tied with the last one "
         "(ties in any order).", f"SELECT * FROM orders ORDER BY amount DESC FETCH FIRST {n} ROWS WITH TIES;",
         {"order": True, "require": ["WITH TIES"]}),
        (f"Show the `id` and `created_at` of the {n} earliest orders, including orders placed at the same moment "
         "as the last one.", f"SELECT id, created_at FROM orders ORDER BY created_at FETCH FIRST {n} ROWS WITH TIES;",
         {"order": True, "require": ["WITH TIES"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "hard")
def select_keyset(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    orders = sorted(f["orders"], key=lambda o: (o[4], o[0]))
    k = rng.randint(3, 6)
    last = orders[k - 1]
    per = rng.choice([3, 4])
    ts = last[4].strftime("%Y-%m-%d %H:%M:%S")
    items = [
        (f"Orders are paged by (`created_at`, `id`), oldest first. The previous page ended with the order "
         f"created at `{ts}` with id {last[0]}. Show all columns of the next {per} orders using keyset "
         "pagination (no OFFSET).",
         f"SELECT * FROM orders WHERE (created_at, id) > ('{ts}', {last[0]}) ORDER BY created_at, id LIMIT {per};",
         {"order": True, "forbid": ["OFFSET"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.01", "challenge")
def select_spec(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    month = rng.choice([2, 3])
    nxt = month + 1
    name = {2: "February", 3: "March"}[month]
    amounts = sorted(o[2] for o in f["orders"])
    x = int(amounts[len(amounts) // 3])
    n = rng.choice([3, 4])
    sol = (f"SELECT id, user_id, amount, created_at FROM orders WHERE status IN ('paid', 'pending') AND amount >= {x} "
           f"AND created_at >= '2024-{month:02d}-01' AND created_at < '2024-{nxt:02d}-01' ORDER BY amount DESC, "
           f"created_at FETCH FIRST {n} ROWS WITH TIES;")
    return Q(f"A dashboard tile shows the largest non-refunded orders of {name} 2024. Return `id`, `user_id`, "
             f"`amount` and `created_at` of the paid or pending orders of at least {x} placed during {name} 2024, "
             f"largest first (equal amounts: earliest first). Keep the top {n}, plus any orders tied with the "
             f"{n}th on both amount and time.", sol, db, hid, order=True,
             note="Hidden databases have orders exactly at midnight on the first of the month and tied amounts.")


@template("10.01", "challenge")
def select_spec_users(rng):
    db, f, hid = shop_db(rng, traps=("null_country", "dup_names"))
    countries = sorted({u[2] for u in f["users"] if u[2]})
    c = rng.choice(countries)
    start = f["start"] + dt.timedelta(days=rng.randint(5, 20))
    sol = (f"SELECT id, name, country FROM users WHERE country IS DISTINCT FROM '{c}' AND signup_date >= "
           f"'{start.isoformat()}' ORDER BY country NULLS FIRST, name, id;")
    return Q(f"Marketing wants everyone outside `{c}` who signed up on or after {start.isoformat()}: return `id`, "
             f"`name` and `country`, users with an unknown country included (they're probably not in `{c}` "
             "either). Sort by country with unknown countries first, then name, then id.", sol, db, hid, order=True)

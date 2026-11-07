"""10.11 Inserting, updating, deleting & upserting (DML)."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q

ST = {"check": "state"}
PRODUCTS = ("CREATE TABLE products (\n  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n  sku TEXT UNIQUE NOT "
            "NULL,\n  name VARCHAR(200) NOT NULL,\n  price NUMERIC(10, 2) NOT NULL CHECK (price > 0),\n  in_stock "
            "BOOLEAN DEFAULT TRUE\n);\nINSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU', 1999.00), "
            "('B-2', 'Cable', 9.50), ('C-3', 'SSD', 120.00);\n")
DAILY = "CREATE TABLE daily_revenue (day DATE PRIMARY KEY, revenue NUMERIC NOT NULL);\n"


def with_extra(db, hid, extra):
    return db + extra, [(h + extra, t) for h, t in hid]


@template("10.11", "lite")
def dml_basic(rng):
    db, f, hid = shop_db(rng)
    nid = len(f["users"]) + 1
    items = [
        (f"Add the user {nid}, `Zed`, from `CA`, who signed up on 2024-05-01.",
         f"INSERT INTO users VALUES ({nid}, 'Zed', 'CA', '2024-05-01');", {**ST, "tables": ["users"]}),
        ("Mark every pending order as paid.", "UPDATE orders SET status = 'paid' WHERE status = 'pending';",
         {**ST, "tables": ["orders"]}),
        ("Delete the refunded orders.", "DELETE FROM orders WHERE status = 'refunded';", {**ST, "tables": ["orders"]}),
        ("Set the country of the users with an unknown country to `US`.",
         "UPDATE users SET country = 'US' WHERE country IS NULL;", {**ST, "tables": ["users"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "lite")
def dml_insert_cols(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, PRODUCTS)
    items = [
        ("Insert two products with a column list (the id is generated): `D-4` `Fan` 19.90 and `E-5` `PSU` 89.00.",
         "INSERT INTO products (sku, name, price) VALUES ('D-4', 'Fan', 19.90), ('E-5', 'PSU', 89.00);",
         {**ST, "tables": ["products"]}),
        ("Insert product `F-6` `Case` 59.00 that is out of stock.",
         "INSERT INTO products (sku, name, price, in_stock) VALUES ('F-6', 'Case', 59.00, FALSE);",
         {**ST, "tables": ["products"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "easy")
def dml_returning(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, PRODUCTS)
    items = [
        ("Insert product `G-7` `RAM` 75.00 and return its generated `id`.",
         "INSERT INTO products (sku, name, price) VALUES ('G-7', 'RAM', 75.00) RETURNING id;",
         {"check": "state", "tables": ["products"], "also": "query"}),
        ("Lower every product's price by 10% (round to 2 decimals) and return `sku` and the new `price`.",
         "UPDATE products SET price = ROUND(price * 0.9, 2) RETURNING sku, price;",
         {"check": "state", "tables": ["products"], "also": "query"}),
        ("Delete the products under 50 and return all their columns.", "DELETE FROM products WHERE price < 50 "
                                                                         "RETURNING *;",
         {"check": "state", "tables": ["products"], "also": "query"}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "easy")
def dml_update_math(rng):
    db, f, hid = shop_db(rng)
    pct = rng.choice([5, 10, 20])
    items = [
        (f"Raise the amount of every paid order by {pct}% (keep 2 decimals).",
         f"UPDATE orders SET amount = ROUND(amount * {1 + pct / 100}, 2) WHERE status = 'paid';",
         {**ST, "tables": ["orders"]}),
        ("Move every order one day later (created_at + 1 day).",
         "UPDATE orders SET created_at = created_at + INTERVAL '1 day';", {**ST, "tables": ["orders"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "medium")
def dml_insert_select(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, DAILY)
    items = [
        ("Fill daily_revenue with one row per day that has orders: the day and the sum of that day's amounts.",
         "INSERT INTO daily_revenue (day, revenue) SELECT created_at::DATE, SUM(amount) FROM orders GROUP BY 1;",
         {**ST, "tables": ["daily_revenue"]}),
        ("Fill daily_revenue with each day's paid revenue (days with paid orders only).",
         "INSERT INTO daily_revenue SELECT created_at::DATE, SUM(amount) FROM orders WHERE status = 'paid' GROUP BY 1;",
         {**ST, "tables": ["daily_revenue"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "medium")
def dml_update_from(rng):
    db, f, hid = shop_db(rng)
    c = rng.choice(f["countries"])
    items = [
        (f"Give a 10% discount (2 decimals) on the orders of users from `{c}` (UPDATE ... FROM users).",
         f"UPDATE orders o SET amount = ROUND(o.amount * 0.9, 2) FROM users u WHERE u.id = o.user_id AND u.country = "
         f"'{c}';", {**ST, "tables": ["orders"], "require": ["FROM"]}),
        ("Delete the orders of users with an unknown country (DELETE with a subquery).",
         "DELETE FROM orders WHERE user_id IN (SELECT id FROM users WHERE country IS NULL);",
         {**ST, "tables": ["orders"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "medium")
def dml_conflict_nothing(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, PRODUCTS)
    items = [
        ("Insert product `A-1` `GPU (dup)` 1.00, doing nothing if the sku already exists (ON CONFLICT).",
         "INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU (dup)', 1.00) ON CONFLICT (sku) DO NOTHING;",
         {**ST, "tables": ["products"], "require": ["ON CONFLICT"]}),
        ("Insert `B-2` `Cable` 9.50 and `H-8` `Mouse` 25.00, skipping the ones whose sku exists.",
         "INSERT INTO products (sku, name, price) VALUES ('B-2', 'Cable', 9.50), ('H-8', 'Mouse', 25.00) ON CONFLICT "
         "(sku) DO NOTHING;", {**ST, "tables": ["products"], "require": ["ON CONFLICT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "hard")
def dml_upsert(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, PRODUCTS)
    items = [
        ("Upsert `A-1` `GPU v2` 2099.00: insert it, or update name and price when the sku exists (EXCLUDED).",
         "INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU v2', 2099.00) ON CONFLICT (sku) DO UPDATE SET "
         "name = EXCLUDED.name, price = EXCLUDED.price;", {**ST, "tables": ["products"], "require": ["EXCLUDED"]}),
        ("Upsert `C-3` `SSD` 99.00 and `I-9` `Hub` 15.00, updating the price only when the new one is lower.",
         "INSERT INTO products (sku, name, price) VALUES ('C-3', 'SSD', 99.00), ('I-9', 'Hub', 15.00) ON CONFLICT (sku) "
         "DO UPDATE SET price = EXCLUDED.price WHERE EXCLUDED.price < products.price;",
         {**ST, "tables": ["products"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "hard")
def dml_merge(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, PRODUCTS)
    items = [
        ("With MERGE, apply the price list (`B-2` 8.99, `D-4` 15.00): update the price of existing skus, insert "
         "missing ones named `new item`.",
         "MERGE INTO products p USING (VALUES ('B-2', 8.99), ('D-4', 15.00)) AS s(sku, price) ON p.sku = s.sku WHEN "
         "MATCHED THEN UPDATE SET price = s.price WHEN NOT MATCHED THEN INSERT (sku, name, price) VALUES (s.sku, 'new "
         "item', s.price);", {**ST, "tables": ["products"], "require": ["MERGE"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "hard")
def dml_idempotent(rng):
    db, f, hid = shop_db(rng)
    db, hid = with_extra(db, hid, DAILY + "INSERT INTO daily_revenue VALUES ('2024-01-01', 1.00);\n")
    items = [
        ("Load each day's total into daily_revenue so the load can be re-run safely: insert new days, overwrite the "
         "revenue of days already there. (Submit runs your statement as you left it; running it twice must give the "
         "same table.)",
         "INSERT INTO daily_revenue (day, revenue) SELECT created_at::DATE, SUM(amount) FROM orders GROUP BY 1 ON "
         "CONFLICT (day) DO UPDATE SET revenue = EXCLUDED.revenue;",
         {**ST, "tables": ["daily_revenue"], "require": ["ON CONFLICT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.11", "challenge")
def dml_dedup(rng):
    db, f, hid = shop_db(rng)
    extra = ("CREATE TABLE signups AS SELECT id AS user_id, name, signup_date, signup_date::TIMESTAMP AS seen_at FROM "
             "users;\nALTER TABLE signups ADD COLUMN row_id SERIAL PRIMARY KEY;\nINSERT INTO signups (user_id, name, "
             "signup_date, seen_at) SELECT user_id, name, signup_date, seen_at + INTERVAL '1 day' FROM signups WHERE "
             "user_id % 2 = 1;\nINSERT INTO signups (user_id, name, signup_date, seen_at) SELECT user_id, name || '!', "
             "signup_date, seen_at FROM signups WHERE user_id = 2;\n")
    db, hid = with_extra(db, hid, extra)
    sol = ("DELETE FROM signups s USING (SELECT row_id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY seen_at DESC, "
           "row_id DESC) AS rn FROM signups) r WHERE r.row_id = s.row_id AND r.rn > 1;")
    return Q("`signups` has duplicate rows per `user_id` (row_id is unique). Delete duplicates in place, keeping for "
             "each user_id only the latest row by `seen_at` (ties: the highest row_id).", sol, db, hid, check="state",
             tables=["signups"])


@template("10.11", "challenge")
def dml_scd2(rng):
    db, f, hid = shop_db(rng)
    extra = ("CREATE TABLE user_history (user_id INTEGER, country TEXT, valid_from DATE NOT NULL, valid_to DATE, "
             "PRIMARY KEY (user_id, valid_from));\nINSERT INTO user_history SELECT id, country, signup_date, NULL "
             "FROM users;\n")
    db, hid = with_extra(db, hid, extra)
    uid = rng.choice([u[0] for u in f["users"]])
    sol = (f"UPDATE user_history SET valid_to = DATE '2024-06-01' WHERE user_id = {uid} AND valid_to IS NULL;\n"
           f"INSERT INTO user_history VALUES ({uid}, 'JP', DATE '2024-06-01', NULL);")
    return Q(f"`user_history` keeps each user's country over time (valid_to NULL = current). User {uid} moves to "
             "`JP` on 2024-06-01: close their current row (valid_to = 2024-06-01) and add the new current row "
             "(valid_from 2024-06-01).", sol, db, hid, check="state", tables=["user_history"])

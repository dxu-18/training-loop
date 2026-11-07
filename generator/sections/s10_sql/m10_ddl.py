"""10.10 Creating & changing tables (DDL). Graded on the tables the session leaves, plus probe statements."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q

ST = {"check": "state"}


def _p(*sql):
    return list(sql)


@template("10.10", "lite")
def ddl_create(rng):
    db, f, hid = shop_db(rng)
    t = rng.choice(["experiments", "runs_log", "trials"])
    items = [
        (f"Create a table `{t}` with columns `id` INTEGER, `name` TEXT, `score` NUMERIC(5, 2) and `created_at` "
         "TIMESTAMP (no constraints).",
         f"CREATE TABLE {t} (id INTEGER, name TEXT, score NUMERIC(5, 2), created_at TIMESTAMP);",
         {**ST, "tables": [t]}),
        (f"Create a table `{t}` with `id` BIGINT, `config` JSONB, `tags` TEXT[] and `done` BOOLEAN.",
         f"CREATE TABLE {t} (id BIGINT, config JSONB, tags TEXT[], done BOOLEAN);", {**ST, "tables": [t]}),
        (f"Create a table `{t}` with `day` DATE and `revenue` NUMERIC (no precision).",
         f"CREATE TABLE {t} (day DATE, revenue NUMERIC);", {**ST, "tables": [t]}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "lite")
def ddl_drop(rng):
    db, f, hid = shop_db(rng)
    extra = "CREATE TABLE scratch (x INT);\nCREATE TABLE tmp_orders AS SELECT * FROM orders WHERE amount > 100;\n"
    items = [
        ("Drop the table `scratch`.", "DROP TABLE scratch;", {**ST, "tables": ["scratch", "tmp_orders"]}),
        ("Drop `tmp_orders`, and write the statement so it doesn't fail when run a second time.",
         "DROP TABLE IF EXISTS tmp_orders;", {**ST, "tables": ["scratch", "tmp_orders"], "require": ["IF EXISTS"]}),
        ("Remove every row of `scratch` but keep the table (TRUNCATE).", "TRUNCATE scratch;",
         {**ST, "tables": ["scratch"], "require": ["TRUNCATE"]}),
    ]
    return drill(rng, items, db + extra, [(h + extra, tag) for h, tag in hid])


@template("10.10", "easy")
def ddl_constraints(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create `products`: `id` INTEGER primary key, `sku` TEXT unique and not null, `name` VARCHAR(200) not null, "
         "`price` NUMERIC(10, 2) not null and > 0, `in_stock` BOOLEAN defaulting to true.",
         "CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT UNIQUE NOT NULL, name VARCHAR(200) NOT NULL, price "
         "NUMERIC(10, 2) NOT NULL CHECK (price > 0), in_stock BOOLEAN DEFAULT TRUE);",
         {**ST, "tables": ["products"],
          "probes": _p("INSERT INTO products (id, sku, name, price) VALUES (1, 'A', 'GPU', 10) RETURNING in_stock",
                       "INSERT INTO products (id, sku, name, price) VALUES (2, 'A', 'x', 1)",
                       "INSERT INTO products (id, sku, name, price) VALUES (3, 'B', 'x', 0)",
                       "INSERT INTO products (id, sku, price) VALUES (4, 'C', 5)")}),
        ("Create `teams`: `team_id` INTEGER primary key, `name` TEXT not null and unique, `budget` NUMERIC(12, 2) "
         "not null with default 0 and never negative.",
         "CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, budget NUMERIC(12, 2) NOT NULL "
         "DEFAULT 0 CHECK (budget >= 0));",
         {**ST, "tables": ["teams"],
          "probes": _p("INSERT INTO teams (team_id, name) VALUES (1, 'vision') RETURNING budget",
                       "INSERT INTO teams VALUES (2, 'vision', 5)", "INSERT INTO teams VALUES (3, 'llm', -1)")}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "easy")
def ddl_alter(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Add a column `email` TEXT to users.", "ALTER TABLE users ADD COLUMN email TEXT;", {**ST, "tables": ["users"]}),
        ("Rename the users column `country` to `country_code`.",
         "ALTER TABLE users RENAME COLUMN country TO country_code;", {**ST, "tables": ["users"]}),
        ("Remove the column `status` from orders... no: give `status` a default of `pending` instead.",
         "ALTER TABLE orders ALTER COLUMN status SET DEFAULT 'pending';",
         {**ST, "tables": ["orders"], "probes": _p("INSERT INTO orders (id, user_id, amount, created_at) VALUES (999, "
                                                   "1, 5, '2024-05-01') RETURNING status")}),
        ("Add a column `vip` BOOLEAN NOT NULL DEFAULT false to users (existing users get false).",
         "ALTER TABLE users ADD COLUMN vip BOOLEAN NOT NULL DEFAULT false;", {**ST, "tables": ["users"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "medium")
def ddl_fk(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Create `order_notes`: `id` INTEGER primary key, `order_id` INTEGER not null referencing orders(id) so that "
         "deleting an order deletes its notes, and `body` TEXT not null.",
         "CREATE TABLE order_notes (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id) ON DELETE "
         "CASCADE, body TEXT NOT NULL);",
         {**ST, "tables": ["order_notes"], "require": ["REFERENCES"],
          "probes": _p("INSERT INTO order_notes VALUES (1, 999999, 'x')",
                       "INSERT INTO order_notes VALUES (2, 1, 'ok')", "DELETE FROM orders WHERE id = 1",
                       "SELECT count(*) FROM order_notes")}),
        ("Create `order_items` with `order_id` INTEGER referencing orders(id) ON DELETE CASCADE, `sku` TEXT, `qty` "
         "INTEGER not null default 1, and a composite primary key (order_id, sku).",
         "CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE, sku TEXT, qty INTEGER NOT "
         "NULL DEFAULT 1, PRIMARY KEY (order_id, sku));",
         {**ST, "tables": ["order_items"],
          "probes": _p("INSERT INTO order_items (order_id, sku) VALUES (1, 'a') RETURNING qty",
                       "INSERT INTO order_items (order_id, sku) VALUES (1, 'a')")}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "medium")
def ddl_identity_ctas(rng):
    db, f, hid = shop_db(rng)
    x = rng.choice([50, 100, 150])
    items = [
        ("Create `tickets`: `id` BIGINT generated always as identity primary key, `title` TEXT not null, "
         "`opened_at` TIMESTAMPTZ defaulting to now().",
         "CREATE TABLE tickets (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, title TEXT NOT NULL, opened_at "
         "TIMESTAMPTZ DEFAULT NOW());",
         {**ST, "tables": ["tickets"], "probes": _p("INSERT INTO tickets (title) VALUES ('a') RETURNING id",
                                                    "INSERT INTO tickets (id, title) VALUES (7, 'b')")}),
        (f"Create the table `big_orders` from the orders over {x} (CREATE TABLE ... AS).",
         f"CREATE TABLE big_orders AS SELECT * FROM orders WHERE amount > {x};",
         {**ST, "tables": ["big_orders"], "require": ["AS"]}),
        ("Create a schema `analytics` and in it a table `daily` with `day` DATE primary key and `revenue` NUMERIC.",
         "CREATE SCHEMA analytics;\nCREATE TABLE analytics.daily (day DATE PRIMARY KEY, revenue NUMERIC);",
         {**ST, "tables": ["analytics.daily"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "hard")
def ddl_change_types(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Change orders.amount to NUMERIC(12, 2).", "ALTER TABLE orders ALTER COLUMN amount TYPE NUMERIC(12, 2);",
         {**ST, "tables": ["orders"]}),
        ("Users must always have a country from now on: set the missing ones to `US`, then make the column NOT NULL.",
         "UPDATE users SET country = 'US' WHERE country IS NULL;\nALTER TABLE users ALTER COLUMN country SET NOT NULL;",
         {**ST, "tables": ["users"]}),
        ("Store users.signup_date as TEXT instead (convert the existing values).",
         "ALTER TABLE users ALTER COLUMN signup_date TYPE TEXT;", {**ST, "tables": ["users"]}),
        ("Make user names unique with a named constraint `users_name_unique` (the data has no duplicates).",
         "ALTER TABLE users ADD CONSTRAINT users_name_unique UNIQUE (name);",
         {**ST, "tables": ["users"], "probes": _p("INSERT INTO users VALUES (900, (SELECT name FROM users LIMIT 1), "
                                                  "'US', '2024-01-01')")}),
    ]
    return drill(rng, items, db, [h for h in hid if "duplicate" not in h[1]])


@template("10.10", "hard")
def ddl_add_check(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Add a CHECK so orders.status can only be `paid`, `pending` or `refunded`.",
         "ALTER TABLE orders ADD CONSTRAINT orders_status_check CHECK (status IN ('paid', 'pending', 'refunded'));",
         {**ST, "tables": ["orders"], "probes": _p("INSERT INTO orders VALUES (900, 1, 5, 'lost', '2024-01-01')",
                                                   "INSERT INTO orders VALUES (901, 1, 5, 'paid', '2024-01-01')")}),
        ("Add a CHECK that orders are created on or after 2024-01-01.",
         "ALTER TABLE orders ADD CHECK (created_at >= '2024-01-01');",
         {**ST, "tables": ["orders"], "probes": _p("INSERT INTO orders VALUES (900, 1, 5, 'paid', '2023-12-31')")}),
    ]
    return drill(rng, items, db, hid)


@template("10.10", "challenge")
def ddl_spec(rng):
    db, f, hid = shop_db(rng)
    sol = ("CREATE TABLE experiments (\n  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n  name TEXT NOT NULL "
           "UNIQUE,\n  owner_id INTEGER NOT NULL REFERENCES users(id),\n  config JSONB NOT NULL DEFAULT '{}',\n  "
           "metrics JSONB,\n  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', "
           "'failed')),\n  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n);")
    probes = _p("INSERT INTO experiments (name, owner_id) VALUES ('a', 1) RETURNING config, metrics, status",
                "INSERT INTO experiments (name, owner_id) VALUES ('a', 1)",
                "INSERT INTO experiments (name, owner_id) VALUES ('b', 99999)",
                "INSERT INTO experiments (name, owner_id, status) VALUES ('c', 1, 'paused')",
                "INSERT INTO experiments (id, name, owner_id) VALUES (5, 'd', 1)",
                "INSERT INTO experiments (owner_id) VALUES (1)")
    return Q("Design `experiments` (column order as listed): `id` BIGINT generated always as identity, primary key; "
             "`name` TEXT, required and unique; `owner_id` INTEGER, required, a user's id; `config` JSONB, required, "
             "default `{}`; `metrics` JSONB, optional; `status` TEXT, required, default `queued`, only `queued`, "
             "`running`, `done` or `failed`; `created_at` TIMESTAMPTZ, required, default now().", sol, db, hid,
             check="state", tables=["experiments"], probes=probes,
             note="Submit runs inserts that must work and inserts that must fail.")


@template("10.10", "challenge")
def ddl_migration(rng):
    db, f, hid = shop_db(rng)
    sol = ("ALTER TABLE users ADD COLUMN email TEXT;\nUPDATE users SET email = LOWER(name) || '@shop.test';\nALTER TABLE "
           "users ALTER COLUMN email SET NOT NULL;\nALTER TABLE users ADD CONSTRAINT users_email_key UNIQUE (email);")
    return Q("Add a required, unique `email` TEXT column to users. Existing users get `<lower-case name>@shop.test` "
             "(names are unique here, so are the emails). Add the column, fill it, then make it NOT NULL and UNIQUE.",
             sol, db, [h for h in hid if "duplicate" not in h[1]], check="state", tables=["users"],
             probes=_p("INSERT INTO users (id, name, signup_date) VALUES (900, 'Zed', '2024-01-01')",
                       "INSERT INTO users (id, name, signup_date, email) VALUES (901, 'Zed', '2024-01-01', "
                       "(SELECT email FROM users LIMIT 1))"))

"""10.15 SQL from Python & the command line: psql's options and meta-commands, \\copy, sqlite3, Python's sqlite3."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D

PS = {"check": "psql"}


@template("10.15", "lite")
def cli_meta(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Turn on expanded display (\\x) and show user 1 with `SELECT * FROM users WHERE id = 1;`.",
         "\\x\nSELECT * FROM users WHERE id = 1;", {**PS, "settings": ["expanded"], "also": "query"}),
        ("List the tables of the database (\\dt), then describe orders (\\d orders).", "\\dt\n\\d orders",
         {**PS, "require": ["\\dt", "\\d"]}),
        ("Show the connection info (\\conninfo) and list the databases (\\l).", "\\conninfo\n\\l",
         {**PS, "require": ["\\conninfo", "\\l"]}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "lite")
def cli_psql_c(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("From the bash prompt, count the orders of the `shop` database with one psql command (-c).",
         "psql -d shop -c \"SELECT COUNT(*) FROM orders\"", {"start": "bash", "require": ["-c"]}),
        ("From the bash prompt, list the user names (`SELECT name FROM users`) with psql -c.",
         "psql -d shop -c \"SELECT name FROM users\"", {"start": "bash", "require": ["-c"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.15", "easy")
def cli_settings(rng):
    db, f, hid = shop_db(rng, traps=("null_country",))
    word = rng.choice(["(null)", "NULL", "-"])
    items = [
        (f"Make psql show NULLs as `{word}` (\\pset null), turn on \\timing, then run `SELECT name, country FROM "
         "users;`.", f"\\pset null '{word}'\n\\timing on\nSELECT name, country FROM users;",
         {**PS, "settings": ["null", "timing"], "also": "query"}),
        ("Switch psql to unaligned output (\\a) and show `SELECT id, name FROM users;`.",
         "\\a\nSELECT id, name FROM users;", {**PS, "settings": ["format"], "also": "query"}),
        ("Switch psql to CSV output (\\pset format csv) and show `SELECT * FROM orders;`.",
         "\\pset format csv\nSELECT * FROM orders;", {**PS, "settings": ["format"], "also": "query"}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "easy")
def cli_sqlite_shell(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("At the bash prompt, create the SQLite database `notes.db` with a table `notes(id INTEGER PRIMARY KEY, body "
         "TEXT)` and one note `hello`, using the sqlite3 command (SQL as arguments).",
         "sqlite3 notes.db \"CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT);\" \"INSERT INTO notes (body) "
         "VALUES ('hello');\"", {**PS, "start": "bash", "sqlite": [("notes.db", "SELECT id, body FROM notes")]}),
        ("With sqlite3, create `metrics.db` holding a table `runs(name TEXT, loss REAL)` with rows (`a`, 0.5) and "
         "(`b`, 0.25).",
         "sqlite3 metrics.db \"CREATE TABLE runs (name TEXT, loss REAL); INSERT INTO runs VALUES ('a', 0.5), ('b', "
         "0.25);\"", {**PS, "start": "bash", "sqlite": [("metrics.db", "SELECT name, loss FROM runs")]}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "medium")
def cli_copy_out(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Export the whole orders table to `orders.csv` with a header line, using psql's \\copy.",
         "\\copy orders TO 'orders.csv' CSV HEADER", {**PS, "files": ["orders.csv"], "require": ["\\copy"]}),
        ("Export the paid orders (`SELECT id, amount FROM orders WHERE status = 'paid'`) to `paid.csv` with a header, "
         "using \\copy with a query.", "\\copy (SELECT id, amount FROM orders WHERE status = 'paid') TO 'paid.csv' CSV "
                                     "HEADER", {**PS, "files": ["paid.csv"], "require": ["\\copy"]}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "medium")
def cli_psql_f(rng):
    db, f, hid = shop_db(rng)
    script = ("CREATE TABLE monthly AS SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS revenue FROM "
              "orders GROUP BY 1;\n")
    items = [
        ("The file `report.sql` builds a summary table. Run it from the bash prompt with psql -f on the `shop` "
         "database.", "psql -d shop -f report.sql",
         {"check": "state", "start": "bash", "tables": ["monthly"], "given": {"report.sql": script},
          "require": ["-f"]}),
        ("Load `fix.sql` into the shop database from the bash prompt (psql -f).", "psql -d shop -f fix.sql",
         {"check": "state", "start": "bash", "tables": ["orders"],
          "given": {"fix.sql": "UPDATE orders SET status = 'paid' WHERE status = 'pending';\n"}, "require": ["-f"]}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "medium")
def cli_sqlite_import(rng):
    db, f, hid = shop_db(rng)
    rows = "\n".join(f"{u[0]},{u[1]},{u[2] or ''}" for u in f["users"])
    csvtext = "id,name,country\n" + rows + "\n"
    items = [
        ("Import `users.csv` into a table `users` of the SQLite database `local.db` with the sqlite3 shell (.mode "
         "csv, .import).", "sqlite3 local.db '.mode csv' '.import users.csv users'",
         {**PS, "start": "bash", "given": {"users.csv": csvtext},
          "sqlite": [("local.db", "SELECT id, name, country FROM users")]}),
    ]
    return drill(rng, items, db, [])


@template("10.15", "hard")
def cli_python_params(rng):
    db, f, hid = shop_db(rng)
    script = ("import sqlite3\ncon = sqlite3.connect('shop.db')\ncur = con.cursor()\ncur.execute(\"CREATE TABLE IF NOT "
              "EXISTS users (id INTEGER PRIMARY KEY, name TEXT, country TEXT)\")\ncur.executemany(\"INSERT INTO users "
              "(name, country) VALUES (?, ?)\", [('Ana', 'US'), ('Ben', 'UK'), ('Cal', 'US')])\ncon.commit()\nrows = "
              "cur.execute(\"SELECT name FROM users WHERE country = ?\", ('US',)).fetchall()\nwith open('us.txt', 'w') "
              "as fh:\n    fh.write('\\n'.join(r[0] for r in rows) + '\\n')\ncon.close()\n")
    sol = "cat > load.py <<'EOF'\n" + script + "EOF\npython3 load.py"
    return Q("Write a Python script that creates `shop.db` (SQLite) with a table `users(id INTEGER PRIMARY KEY, name "
             "TEXT, country TEXT)`, inserts Ana/US, Ben/UK and Cal/US with executemany and `?` parameters, commits, "
             "then selects the names of the US users with a parameterized query and writes them, one per line, to "
             "`us.txt`. Run it.", sol, db, [], check="psql", start="bash", files=["us.txt"],
             sqlite=[("shop.db", "SELECT name, country FROM users")])


@template("10.15", "hard")
def cli_python_report(rng):
    db, f, hid = shop_db(rng)
    rows = "\n".join(f"{o[0]},{o[1]},{o[2]},{o[3]}" for o in f["orders"])
    given = {"orders.csv": "id,user_id,amount,status\n" + rows + "\n"}
    script = ("import csv, sqlite3\ncon = sqlite3.connect(':memory:')\ncon.execute('CREATE TABLE orders (id INTEGER, "
              "user_id INTEGER, amount REAL, status TEXT)')\nwith open('orders.csv') as fh:\n    r = csv.reader(fh)\n"
              "    next(r)\n    con.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)', r)\nwith open('totals.csv', "
              "'w', newline='') as out:\n    w = csv.writer(out)\n    w.writerow(['status', 'n'])\n    for row in "
              "con.execute('SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status'):\n        "
              "w.writerow(row)\n")
    sol = "cat > report.py <<'EOF'\n" + script + "EOF\npython3 report.py"
    return Q("`orders.csv` is an export of the orders. With Python's csv and sqlite3 modules (an in-memory database), "
             "load it and write `totals.csv`: a header `status,n`, then one line per status with its order count, "
             "sorted by status.", sol, db, [], check="psql", start="bash", files=["totals.csv"], given=given)


@template("10.15", "challenge")
def cli_etl(rng):
    db, f, hid = shop_db(rng, n_orders=16)
    sol = ("\\copy (SELECT id, user_id, amount, status FROM orders) TO 'orders.csv' CSV HEADER\n\\q\n"
           "cat > etl.py <<'EOF'\nimport csv, sqlite3\ncon = sqlite3.connect('warehouse.db')\ncon.execute('CREATE TABLE "
           "orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)')\nwith open('orders.csv') as fh:"
           "\n    r = csv.DictReader(fh)\n    con.executemany('INSERT INTO orders VALUES (?, ?, ?, ?)', [(int(x['id']), "
           "int(x['user_id']), float(x['amount']), x['status']) for x in r])\ncon.commit()\nwith open('by_user.csv', "
           "'w', newline='') as out:\n    w = csv.writer(out)\n    w.writerow(['user_id', 'paid_total'])\n    for row "
           "in con.execute(\"SELECT user_id, ROUND(SUM(amount), 2) FROM orders WHERE status = 'paid' GROUP BY user_id "
           "ORDER BY user_id\"):\n        w.writerow(row)\ncon.close()\nEOF\npython3 etl.py")
    return Q("A small ETL: (1) in psql, \\copy the orders' `id, user_id, amount, status` to `orders.csv` with a "
             "header and quit psql; (2) at the bash prompt write a Python script that loads the CSV into the SQLite "
             "file `warehouse.db` (table `orders(id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status "
             "TEXT)`), commits, and writes `by_user.csv`: header `user_id,paid_total`, then each user's paid total "
             "(SQLite ROUND(SUM(amount), 2)) by user_id; (3) run it.", sol, db, [], check="psql", quit_ok=True,
             files=["orders.csv", "by_user.csv"], sqlite=[("warehouse.db", "SELECT id, user_id, amount, status FROM "
                                                                           "orders")])


@template("10.15", "challenge")
def cli_parameterized(rng):
    db, f, hid = shop_db(rng)
    sol = ("cat > safe.py <<'EOF'\nimport sqlite3, sys\ncon = sqlite3.connect('users.db')\ncon.execute('CREATE TABLE IF "
           "NOT EXISTS users (name TEXT, country TEXT)')\ncon.executemany('INSERT INTO users VALUES (?, ?)', [('Ana', "
           "'US'), ('Ben', 'UK'), (\"O'Hara\", 'IE')])\ncon.commit()\n\ndef by_name(name):\n    return con.execute('SELECT "
           "country FROM users WHERE name = ?', (name,)).fetchall()\n\nwith open('answers.txt', 'w') as fh:\n    for "
           "who in [\"O'Hara\", \"x' OR '1'='1\"]:\n        fh.write(f'{who}: {by_name(who)}\\n')\ncon.close()\nEOF\n"
           "python3 safe.py")
    return Q("Show why parameters matter. Write `safe.py`: create SQLite `users.db` with `users(name TEXT, country "
             "TEXT)` holding Ana/US, Ben/UK and O'Hara/IE (committed); define `by_name(name)` returning the "
             "fetchall() of `SELECT country FROM users WHERE name = ?` with the name as a parameter; then write to "
             "`answers.txt` one line per lookup, `<name>: <result>`, for `O'Hara` and for the injection attempt "
             "`x' OR '1'='1`. Run it.", sol, db, [], check="psql", start="bash", files=["answers.txt"])


@template("10.15", "hard")
def cli_pandas(rng):
    db, f, hid = shop_db(rng)
    rows = "\n".join(f"{o[0]},{o[1]},{o[2]},{o[3]}" for o in f["orders"])
    given = {"orders.csv": "id,user_id,amount,status\n" + rows + "\n"}
    status = rng.choice(sorted({o[3] for o in f["orders"]}))
    items = [
        ("With pandas, load `orders.csv` into a table `orders` of the SQLite database `shop.db` (`to_sql`, no index "
         "column), then save the result of `pd.read_sql` on `SELECT status, COUNT(*) AS n, ROUND(SUM(amount), 2) AS "
         "total FROM orders GROUP BY status ORDER BY status` to `by_status.csv` (no index column). Write the script "
         "and run it.",
         "cat > load.py <<'EOF'\nimport sqlite3\nimport pandas as pd\ncon = sqlite3.connect('shop.db')\n"
         "pd.read_csv('orders.csv').to_sql('orders', con, if_exists='replace', index=False)\n"
         "df = pd.read_sql('SELECT status, COUNT(*) AS n, ROUND(SUM(amount), 2) AS total FROM orders GROUP BY status "
         "ORDER BY status', con)\ndf.to_csv('by_status.csv', index=False)\ncon.close()\nEOF\npython3 load.py",
         {"files": ["by_status.csv"], "sqlite": [("shop.db", "SELECT id, user_id, amount, status FROM orders")]}),
        (f"With pandas and an in-memory SQLite database, load `orders.csv` (`to_sql`), then use `pd.read_sql` with a "
         f"`?` parameter to get the `id` and `amount` of the orders whose status is `{status}`, ordered by id, and save "
         f"them to `{status}.csv` (no index column). Write the script and run it.",
         "cat > pick.py <<'EOF'\nimport sqlite3\nimport pandas as pd\ncon = sqlite3.connect(':memory:')\n"
         "pd.read_csv('orders.csv').to_sql('orders', con, index=False)\n"
         "df = pd.read_sql('SELECT id, amount FROM orders WHERE status = ? ORDER BY id', con, "
         f"params=('{status}',))\ndf.to_csv('{status}.csv', index=False)\nEOF\npython3 pick.py",
         {"files": [f"{status}.csv"]}),
    ]
    return drill(rng, items, db, [], check="psql", start="bash", given=given,
                 machine=["#: venv ~/.venv pandas", "#: activate ~/.venv"])

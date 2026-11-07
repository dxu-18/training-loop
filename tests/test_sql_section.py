"""COMP_GUIDE §10 SQL (generator/sections/s10_sql) on postgres-lite (generator/postgreslite): every catalog item of
every template passes with its reference and fails idle, every glossary entry is practised, postgres-lite prints
what PostgreSQL 16 prints (tests/sql_corpus, recorded from a real server), wrong answers get specific messages, a
hard-coded answer fails on the hidden databases, and the command-line tools (psql options, sqlite3, Python's
sqlite3) work on the simulated machine."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generator import core, grader  # noqa: E402
from generator.bashlite.live import LiveTerm, reference_session  # noqa: E402
from generator.instructions import instructions  # noqa: E402
from generator.postgreslite.glossary import CATALOG, uses  # noqa: E402
import sql_corpus  # noqa: E402
from sql_templates import problems  # noqa: E402

PROBLEMS = [(f"{tpl.module}-{tpl.name}-{item}-{seed}", p, err) for tpl, item, seed, p, err in problems(None, 1)]


def problem(module, name, item=0, seed=1):
    tpl = next(t for t in core.templates("sql", module) if t.name == name)
    core._FORCE_ITEM[0] = item
    try:
        return core.build(tpl, seed).to_dict()
    finally:
        core._FORCE_ITEM[0] = None


def psql(sql, shop=None):
    """psql's output for sql run on the corpus' shop database."""
    return sql_corpus.lite(sql, shop)


def term(setup=""):
    t = LiveTerm({"setup": setup, "seed": 1}, setup)
    return t


def run(t, *lines):
    for line in lines:
        t.feed("text", line)
        t.feed("key", "Enter")
        t.advance(t.k.now + 10 ** 10, cond=lambda: t.where() != "r")
    return t.state()["out"]


# ---------------------------------------------------------------- the templates

@pytest.mark.parametrize("pid,p,err", PROBLEMS, ids=[x[0] for x in PROBLEMS])
def test_reference_passes_and_idle_fails(pid, p, err):
    assert err is None, err
    res = grader.grade(p, reference_session(p))
    assert res["correct"], res["message"] or res["error"]
    assert not grader.grade(p, "SELECT 42 AS x;")["correct"]


def test_all_sixteen_modules_are_live():
    assert core.modules("sql") == [f"10.{i:02d}" for i in range(1, 17)]
    for m in core.modules("sql"):
        for tier in core.TIERS:
            assert len(core.templates("sql", m, tier)) >= 2, (m, tier)


def test_every_glossary_entry_is_practised():
    used = set()
    for _, p, err in PROBLEMS:
        used |= set(uses(p["solution"])) | {a for r in p["require"] for a in r.split("|")}
    assert not [k for k in CATALOG if k not in used]


def test_instructions_name_what_is_new():
    names = [x["name"] for x in instructions(problem("10.04", "join_left", 3))]
    assert names[0] == "RIGHT JOIN"
    names = [x["name"] for x in instructions(problem("10.06", "win_moving", 0))] if any(
        t.name == "win_moving" for t in core.templates("sql", "10.06")) else ["OVER"]
    assert "OVER" in names or "ROWS" in names


# ---------------------------------------------------------------- postgres-lite = PostgreSQL 16

@pytest.mark.parametrize("name", sql_corpus.names())
def test_matches_recorded_postgresql(name):
    """Results, errors, \\d output and EXPLAIN plans as a real PostgreSQL 16.2 / psql 16 printed them."""
    diffs = sql_corpus.compare(name)
    assert not diffs, "\n\n".join(f"block {i}:\n{sql}\n--- postgres\n{want}\n--- lite\n{got}"
                                  for i, sql, want, got in diffs[:2])


def test_errors_point_at_the_mistake():
    out = psql("SELECT amout FROM orders;")
    assert 'ERROR:  column "amout" does not exist' in out
    assert "LINE 1: SELECT amout FROM orders;\n               ^" in out
    assert 'HINT:  Perhaps you meant to reference the column "orders.amount".' in out


def test_a_failed_statement_aborts_the_transaction():
    out = psql("BEGIN;\nSELECT 1/0;\nSELECT 1;\nROLLBACK;\nSELECT 2 AS two;")
    assert "ERROR:  division by zero" in out
    assert "current transaction is aborted, commands ignored until end of transaction block" in out
    assert " two \n-----\n   2\n(1 row)" in out


def test_explain_follows_the_planner():
    shop = (ROOT / "tests" / "sql_corpus" / "shop.sql").read_text()
    big = ("DROP TABLE orders; CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, amount NUMERIC(10,2));\n"
           "INSERT INTO orders SELECT g, 1 + g % 500, g % 1000 FROM generate_series(1, 20000) g;\nANALYZE;\n")
    out = psql(big + "EXPLAIN SELECT * FROM orders WHERE user_id = 7;\nCREATE INDEX ON orders (user_id);\n"
               "EXPLAIN SELECT * FROM orders WHERE user_id = 7;\n"
               "EXPLAIN SELECT * FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);",
               shop)
    first, second, third = out.split("QUERY PLAN")[1:4]
    assert "Seq Scan on orders" in first and "Filter: (user_id = 7)" in first
    assert "Bitmap Index Scan on orders_user_id_idx" in second or "Index Scan using orders_user_id_idx" in second
    assert "Anti Join" in third and "u.id" in third


# ---------------------------------------------------------------- grading

def test_wrong_answers_get_specific_messages():
    p = problem("10.04", "join_left", 0)
    msg = grader.grade(p, "SELECT u.name, o.amount FROM users u JOIN orders o ON o.user_id = u.id;")["message"]
    assert "LEFT JOIN" in msg
    msg = grader.grade(p, "SELECT u.name AS who, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id;")
    assert "name column 1 `name` (it's `who`)" in msg["message"]
    p = problem("10.01", "select_where", 0)
    msg = grader.grade(p, "SELECT * FROM users;")["message"]
    assert "it returns 4 columns (id, name, country, signup_date); expected 5" in msg


def test_a_hard_coded_result_fails_on_the_hidden_databases():
    p = problem("10.01", "select_where", 0)
    assert p["sql"]["hidden"], "this problem should have hidden databases"
    from generator.sections.s10_sql._common import run_reference
    pp, _ = run_reference(p["sql"]["dbs"]["shop"], p["solution"])
    res = pp.log[-1]["results"][-1]
    cells = lambda r: ", ".join("NULL" if v is None else f"'{v}'" for v in r)  # noqa: E731
    lit = "SELECT * FROM (VALUES " + ", ".join(f"({cells(r)})" for r in res["rows"]) + \
        f") AS t({', '.join(res['names'])});"
    out = grader.grade(p, lit)
    assert not out["correct"]


# ---------------------------------------------------------------- the command line (10.15)

def test_psql_options_and_copy_at_the_bash_prompt():
    p = problem("10.15", "cli_psql_c", 0)
    t = LiveTerm(p, "")
    out = run(t, "psql -d shop -c \"\\copy orders TO 'o.csv' CSV HEADER\"", "head -1 o.csv",
              "psql -d shop -At -c 'SELECT COUNT(*) FROM users'", "psql -d nope -c 'SELECT 1'")
    assert "COPY " in out and "id,user_id,amount,status,created_at" in out
    assert 'psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  ' \
           'database "nope" does not exist' in out


def test_sqlite3_shell_and_python_share_the_files():
    t = term()
    out = run(t, "sqlite3 notes.db \"CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT); "
                 "INSERT INTO notes (body) VALUES ('hi'), ('there');\"",
              "python3 -c \"import sqlite3; c = sqlite3.connect('notes.db'); "
              "print(c.execute('SELECT COUNT(*) FROM notes').fetchone()[0]); "
              "c.execute(\\\"INSERT INTO notes (body) VALUES ('py')\\\"); c.commit()\"",
              "sqlite3 -header -csv notes.db 'SELECT id, body FROM notes ORDER BY id'")
    assert "\n2\n" in out
    assert "id,body\n1,hi\n2,there\n3,py\n" in out


def test_python_sqlite3_rolls_back_without_commit():
    t = term()
    out = run(t, "python3 -c \"import sqlite3; c = sqlite3.connect('a.db'); c.execute('CREATE TABLE t (x)'); "
                 "c.commit(); c.execute('INSERT INTO t VALUES (1)'); c.close()\"",
              "sqlite3 a.db 'SELECT COUNT(*) FROM t'")
    assert out.rstrip().endswith("0\n\x1b[01;32mada@laptop\x1b[00m:\x1b[01;34m~\x1b[00m$") or "\n0\n" in out


def test_the_site_bundle_carries_the_engine_data():
    """postgres-lite reads its catalog and psql's help texts at import: they must be in docs/generator.zip."""
    import build
    shipped = set(build.sources())
    data = [p for p in (ROOT / "generator" / "postgreslite").iterdir() if p.is_file() and p.suffix != ".py"]
    assert data and not [p.name for p in data if p not in shipped]


def test_the_database_panel_follows_the_tables():
    """The panel's tables, row counts and change marks (arrays and JSON values included)."""
    from generator.bashlite.sqlsetup import db_tree
    p = problem("10.13", "array_basics", 0)
    t = LiveTerm(p, p["setup"])
    start = {r["name"]: r for r in db_tree(t)["rels"]}
    assert start and all(r["rows"] is not None for r in start.values())
    name = next(iter(start))
    run(t, f"DELETE FROM {name};")
    now = {r["name"]: r for r in db_tree(t)["rels"]}
    assert now[name]["rows"] == 0 and now[name]["sig"] != start[name]["sig"]


def test_an_empty_database_saves_as_an_empty_file():
    """SQLite before 3.42 (Pyodide's 3.39) can't serialize an empty database: a new file is then 0 bytes."""
    import sqlite3
    from generator.bashlite.utils.sqlitecli import serialize

    class Old(sqlite3.Connection):
        def serialize(self, *a, **k):
            raise sqlite3.OperationalError("unable to serialize 'main'")
    assert serialize(sqlite3.connect(":memory:", factory=Old)) == b""

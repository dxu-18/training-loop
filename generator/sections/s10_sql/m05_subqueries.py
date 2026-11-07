"""10.05 Subqueries, CTEs & set operations."""

from __future__ import annotations

from ._common import template, drill, shop_db, Q
from . import _data as D


def models_db(rng):
    sql, f = D.model_registry(rng)
    return sql, f, D.hidden(D.model_registry, rng, f, D.MODEL_TRAPS, k=3)


@template("10.05", "lite")
def sub_scalar(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show all columns of the orders above the average order amount (scalar subquery).",
         "SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders);"),
        ("Show `id` and `amount` of the largest order(s).",
         "SELECT id, amount FROM orders WHERE amount = (SELECT MAX(amount) FROM orders);"),
        ("List the `name` of the users who signed up on the earliest signup date.",
         "SELECT name FROM users WHERE signup_date = (SELECT MIN(signup_date) FROM users);"),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "lite")
def sub_in(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List the `name` of the users who have a refunded order, with IN (subquery).",
         "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'refunded');",
         {"require": ["IN"]}),
        ("List `id` and `amount` of the orders placed by users with a known country, with IN (subquery).",
         "SELECT id, amount FROM orders WHERE user_id IN (SELECT id FROM users WHERE country IS NOT NULL);"),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "easy")
def sub_derived(rng):
    db, f, hid = shop_db(rng)
    x = rng.choice([100, 150, 200])
    items = [
        (f"Show `user_id` and `total` for the users whose orders add up to more than {x}, using a derived table "
         "(a subquery in FROM) that sums per user.",
         f"SELECT t.user_id, t.total FROM (SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id) AS t "
         f"WHERE t.total > {x};"),
        ("Show the average number of orders per user who ordered, as `avg_orders` rounded to 2 decimals.",
         "SELECT ROUND(AVG(n), 2) AS avg_orders FROM (SELECT user_id, COUNT(*) AS n FROM orders GROUP BY user_id) t;"),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "easy")
def sub_cte(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("With a CTE `user_totals` (user_id, total of paid orders), list `name` and `total` for every user in it.",
         "WITH user_totals AS (SELECT user_id, SUM(amount) AS total FROM orders WHERE status = 'paid' GROUP BY "
         "user_id) SELECT u.name, t.total FROM user_totals t JOIN users u ON u.id = t.user_id;", {"require": ["WITH"]}),
        ("With a CTE of each user's order count, show the `name` of the users with the most orders (all ties).",
         "WITH c AS (SELECT user_id, COUNT(*) AS n FROM orders GROUP BY user_id) SELECT u.name FROM c JOIN users u ON "
         "u.id = c.user_id WHERE c.n = (SELECT MAX(n) FROM c);", {"require": ["WITH"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "easy")
def sub_union(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("List every `country` users come from plus `FR`, each once (UNION).",
         "SELECT country FROM users UNION SELECT 'FR';", {"require": ["UNION"]}),
        ("List all user ids from orders followed by all ids of users, keeping duplicates (UNION ALL), as `id`.",
         "SELECT user_id AS id FROM orders UNION ALL SELECT id FROM users;", {"require": ["UNION ALL"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "medium")
def sub_correlated(rng):
    db, f, hid = shop_db(rng, traps=("ties",))
    items = [
        ("Show all columns of each user's largest order(s) (correlated subquery).",
         "SELECT o.* FROM orders o WHERE o.amount = (SELECT MAX(o2.amount) FROM orders o2 WHERE o2.user_id = "
         "o.user_id);"),
        ("Show `id`, `user_id` and `amount` of the orders above their own user's average order amount.",
         "SELECT o.id, o.user_id, o.amount FROM orders o WHERE o.amount > (SELECT AVG(o2.amount) FROM orders o2 WHERE "
         "o2.user_id = o.user_id);"),
        ("For every user show `name` and `last_order` (their latest created_at, NULL when none) with a correlated "
         "subquery in SELECT.",
         "SELECT u.name, (SELECT MAX(o.created_at) FROM orders o WHERE o.user_id = u.id) AS last_order FROM users u;"),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "medium")
def sub_set_ops(rng):
    db, f, hid = shop_db(rng)
    items = [
        ("Show the `id` of the users who have at least one order (INTERSECT).",
         "SELECT id FROM users INTERSECT SELECT user_id FROM orders;", {"require": ["INTERSECT"]}),
        ("Show the `id` of the users with no orders (EXCEPT).", "SELECT id FROM users EXCEPT SELECT user_id FROM orders;",
         {"require": ["EXCEPT"]}),
        ("Show the `user_id` of users with a paid order and no refunded order (EXCEPT).",
         "SELECT user_id FROM orders WHERE status = 'paid' EXCEPT SELECT user_id FROM orders WHERE status = "
         "'refunded';", {"require": ["EXCEPT"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "medium")
def sub_not_in_trap(rng):
    sql, f = D.model_registry(rng, traps=("unevaluated",))
    hid = D.hidden(D.model_registry, rng, f, D.MODEL_TRAPS, k=3)
    items = [
        ("List the `name` of the models that no other model was fine-tuned from (no model has them as parent_id). "
         "Careful: parent_id is NULL for base models.",
         "SELECT m.name FROM models m WHERE NOT EXISTS (SELECT 1 FROM models c WHERE c.parent_id = m.model_id);",
         {"note": "`NOT IN (SELECT parent_id ...)` returns nothing when the subquery has a NULL."}),
        ("List the `name` of the models without any eval.",
         "SELECT m.name FROM models m WHERE NOT EXISTS (SELECT 1 FROM evals e WHERE e.model_id = m.model_id);"),
    ]
    return drill(rng, items, sql, hid)


@template("10.05", "hard")
def sub_recursive(rng):
    db, f, hid = shop_db(rng)
    n = rng.choice([10, 12, 15])
    items = [
        (f"Generate the numbers 1 to {n} as `n` with a recursive CTE.",
         f"WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < {n}) SELECT n FROM nums;",
         {"require": ["RECURSIVE"]}),
        (f"Generate the first {n} powers of 2 (1, 2, 4, ...) as `p` with a recursive CTE.",
         f"WITH RECURSIVE p(k, p) AS (SELECT 1, 1 UNION ALL SELECT k + 1, p * 2 FROM p WHERE k < {n}) SELECT p FROM p;",
         {"require": ["RECURSIVE"]}),
        ("Generate the first 10 Fibonacci numbers (1, 1, 2, 3, ...) as `fib` with a recursive CTE.",
         "WITH RECURSIVE f(i, a, b) AS (SELECT 1, 1, 1 UNION ALL SELECT i + 1, b, a + b FROM f WHERE i < 10) SELECT a "
         "AS fib FROM f;", {"require": ["RECURSIVE"]}),
    ]
    return drill(rng, items, db, hid)


@template("10.05", "hard")
def sub_hierarchy(rng):
    sql, f, hid = models_db(rng)
    base = [m for m in f["models"] if m[6] is None]
    root = rng.choice(base)
    items = [
        (f"List `name` and `depth` of `{root[1]}` and every model fine-tuned from it, directly or not (depth 0 for "
         "itself, 1 for its children ...).",
         f"WITH RECURSIVE tree AS (SELECT model_id, name, 0 AS depth FROM models WHERE name = '{root[1]}' UNION ALL "
         "SELECT m.model_id, m.name, t.depth + 1 FROM models m JOIN tree t ON m.parent_id = t.model_id) SELECT name, "
         "depth FROM tree;", {"require": ["RECURSIVE"]}),
        ("For every model show `name` and `root`, the name of the base model at the top of its lineage.",
         "WITH RECURSIVE up AS (SELECT model_id, name, model_id AS root_id FROM models WHERE parent_id IS NULL UNION "
         "ALL SELECT m.model_id, m.name, up.root_id FROM models m JOIN up ON m.parent_id = up.model_id) SELECT up.name, "
         "r.name AS root FROM up JOIN models r ON r.model_id = up.root_id;", {"require": ["RECURSIVE"]}),
    ]
    return drill(rng, items, sql, hid)


@template("10.05", "hard")
def sub_division(rng):
    sql, f, hid = models_db(rng)
    items = [
        ("List the `name` of the models evaluated on every benchmark that appears in evals (relational division).",
         "SELECT m.name FROM models m WHERE NOT EXISTS (SELECT b.benchmark FROM (SELECT DISTINCT benchmark FROM evals) "
         "b WHERE NOT EXISTS (SELECT 1 FROM evals e WHERE e.model_id = m.model_id AND e.benchmark = b.benchmark));"),
        ("List the `name` of the models evaluated on every benchmark, counting benchmarks per model (HAVING).",
         "SELECT m.name FROM models m JOIN evals e ON e.model_id = m.model_id GROUP BY m.model_id, m.name HAVING "
         "COUNT(DISTINCT e.benchmark) = (SELECT COUNT(DISTINCT benchmark) FROM evals);"),
    ]
    return drill(rng, items, sql, hid)


@template("10.05", "challenge")
def sub_lineage_best(rng):
    sql, f, hid = models_db(rng)
    sol = ("WITH RECURSIVE up AS (SELECT model_id, model_id AS root_id FROM models WHERE parent_id IS NULL UNION ALL "
           "SELECT m.model_id, up.root_id FROM models m JOIN up ON m.parent_id = up.model_id), scored AS (SELECT "
           "up.root_id, m.name, e.score FROM up JOIN models m ON m.model_id = up.model_id JOIN evals e ON e.model_id "
           "= m.model_id AND e.benchmark = 'mmlu') SELECT r.name AS base_model, s.name AS best_model, s.score FROM "
           "scored s JOIN models r ON r.model_id = s.root_id WHERE s.score = (SELECT MAX(s2.score) FROM scored s2 "
           "WHERE s2.root_id = s.root_id) ORDER BY base_model, best_model;")
    return Q("For each base model (no parent), find the model in its lineage (itself or any descendant) with the "
             "highest `mmlu` score: `base_model`, `best_model`, `score`. Keep all ties; lineages without any mmlu "
             "score are left out. By base_model, then best_model.", sol, sql, hid, order=True)


@template("10.05", "challenge")
def sub_cohort_gap(rng):
    db, f, hid = shop_db(rng, traps=("no_orders", "refunded_only"))
    sol = ("WITH paid AS (SELECT user_id, SUM(amount) AS paid FROM orders WHERE status = 'paid' GROUP BY user_id), "
           "refunds AS (SELECT user_id, COUNT(*) AS n FROM orders WHERE status = 'refunded' GROUP BY user_id) SELECT "
           "u.name, COALESCE(p.paid, 0) AS paid, COALESCE(r.n, 0) AS refunds FROM users u LEFT JOIN paid p ON "
           "p.user_id = u.id LEFT JOIN refunds r ON r.user_id = u.id WHERE COALESCE(p.paid, 0) < (SELECT AVG(paid) "
           "FROM paid) ORDER BY paid, u.name;")
    return Q("List the users whose paid total is below the average paid total of the users who have paid orders: "
             "`name`, `paid` (0 when none) and `refunds` (number of refunded orders, 0 when none). Users without "
             "any order count as 0. Order by paid, then name.", sol, db, hid, order=True)

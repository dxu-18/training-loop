"""What the SQL problems practise: every keyword, function, operator and psql / sqlite3 command of COMP_GUIDE §10,
with a one-line description and the module that introduces it. The Instructions panel lists the entries a
problem's reference solution uses (instructions.py: sql_instructions)."""

from __future__ import annotations

# (feature as sqlcheck.feature_used matches it, description); grouped by the module that introduces them
_BY_MODULE = {
    "10.01": [
        ("SELECT", "SELECT cols FROM t — pick columns (* = all) from a table"),
        ("AS", "expr AS name — rename an output column"),
        ("DISTINCT", "SELECT DISTINCT cols — drop duplicate rows"),
        ("WHERE", "WHERE condition — keep the rows where it is true (not NULL)"),
        ("BETWEEN", "x BETWEEN a AND b — a <= x <= b, both ends included"),
        ("IN", "x IN (a, b) — x equals one of the values"),
        ("LIKE", "LIKE 'A%' — pattern: % any characters, _ exactly one"),
        ("ILIKE", "ILIKE — LIKE ignoring case (PostgreSQL)"),
        ("IS NULL", "x IS NULL / IS NOT NULL — the only way to test for NULL (= NULL is never true)"),
        ("IS DISTINCT FROM", "a IS DISTINCT FROM b — NULL-safe <> (NULL vs 'x' is true)"),
        ("NOT IN", "x NOT IN (a, b) — none of the values (a NULL in the list makes it never true)"),
        ("ORDER BY", "ORDER BY a, b DESC — sort; ASC is the default"),
        ("NULLS LAST", "ORDER BY x NULLS FIRST / NULLS LAST — where NULLs go (default: last ASC, first DESC)"),
        ("LIMIT", "LIMIT n — keep the first n rows (sort first!)"),
        ("OFFSET", "OFFSET k — skip k rows (paging: LIMIT n OFFSET n * (page - 1))"),
        ("FETCH", "FETCH FIRST n ROWS ONLY — standard SQL LIMIT"),
        ("WITH TIES", "FETCH FIRST n ROWS WITH TIES — also keep rows tied with the last one"),
    ],
    "10.02": [
        ("CASE", "CASE WHEN cond THEN a ... ELSE b END — if / else in an expression"),
        ("COALESCE(", "COALESCE(a, b, ...) — the first non-NULL argument"),
        ("NULLIF(", "NULLIF(a, b) — NULL when a = b, else a (safe division: x / NULLIF(y, 0))"),
        ("CAST", "CAST(x AS type) / x::type — convert a value"),
        ("::", "x::type — PostgreSQL's cast shorthand"),
        ("DATE '...'", "DATE '2024-06-01', TIMESTAMP '2024-06-01 09:30' — typed literals; date - date gives days"),
        ("||", "a || b — concatenate text (NULL || 'x' is NULL)"),
        ("ROUND(", "ROUND(numeric, n) — round to n decimals (ROUND(x, -1): to tens)"),
        ("GREATEST(", "GREATEST(a, b, ...) — the largest argument (NULLs ignored)"),
        ("LEAST(", "LEAST(a, b, ...) — the smallest argument"),
    ],
    "10.03": [
        ("COUNT(", "COUNT(*) rows, COUNT(col) non-NULL values, COUNT(DISTINCT col) distinct ones"),
        ("COUNT(DISTINCT", "COUNT(DISTINCT col) — distinct non-NULL values"),
        ("SUM(", "SUM(x) — total (NULL when there are no rows)"),
        ("AVG(", "AVG(x) — mean (integers average to numeric)"),
        ("MIN(", "MIN / MAX(x) — smallest / largest value"),
        ("GROUP BY", "GROUP BY cols — one output row per group; other columns must be aggregated"),
        ("HAVING", "HAVING condition — filter groups (WHERE filters rows before grouping)"),
        ("FILTER", "agg(...) FILTER (WHERE cond) — aggregate only some rows"),
        ("STRING_AGG(", "STRING_AGG(text, sep ORDER BY ...) — join a group's strings"),
        ("ARRAY_AGG(", "ARRAY_AGG(x ORDER BY ...) — a group's values as an array"),
        ("PERCENTILE_CONT(", "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x) — the median (interpolated)"),
        ("STDDEV(", "STDDEV / VARIANCE(x) — sample standard deviation / variance"),
        ("ROLLUP", "GROUP BY ROLLUP (a, b) — groups plus subtotals per a and a grand total"),
        ("CUBE", "GROUP BY CUBE (a, b) — every combination of subtotals"),
        ("GROUPING SETS", "GROUP BY GROUPING SETS ((a), (b)) — several groupings in one result"),
        ("BOOL_AND(", "BOOL_AND / BOOL_OR(cond) — true for all rows / for any row"),
        ("DATE_TRUNC(", "DATE_TRUNC('month', ts) — the start of the month (week, day, hour ...)"),
    ],
    "10.04": [
        ("JOIN", "a JOIN b ON cond — pairs of matching rows (INNER JOIN)"),
        ("LEFT JOIN", "LEFT JOIN — keep every row of the left table (NULLs when no match)"),
        ("RIGHT JOIN", "RIGHT JOIN — keep every row of the right table"),
        ("FULL", "FULL OUTER JOIN — keep unmatched rows of both sides"),
        ("CROSS JOIN", "CROSS JOIN — every pair of rows"),
        ("USING", "JOIN ... USING (col) — join on equally named columns (shown once)"),
        ("EXISTS", "EXISTS (subquery) — semi-join: true when the subquery returns a row"),
        ("NOT EXISTS", "NOT EXISTS (subquery) — anti-join: rows with no match (NULL-safe)"),
        ("LATERAL", "JOIN LATERAL (subquery) ON TRUE — a subquery run per row (top-N per row)"),
    ],
    "10.05": [
        ("IN (SELECT ...)", "x IN (SELECT col FROM ...) — keep rows whose x is among the subquery's values"),
        ("(SELECT ...)", "(SELECT ...) as a value — a scalar subquery; it may use the outer row (correlated)"),
        ("WITH", "WITH name AS (query) — a named subquery (CTE), read top to bottom"),
        ("RECURSIVE", "WITH RECURSIVE t AS (base UNION ALL step) — sequences and hierarchies"),
        ("UNION", "q1 UNION q2 — rows of both, duplicates removed"),
        ("UNION ALL", "UNION ALL — keep duplicates (and faster)"),
        ("INTERSECT", "INTERSECT — rows in both queries"),
        ("EXCEPT", "EXCEPT — rows of the first query not in the second"),
    ],
    "10.06": [
        ("OVER", "f(...) OVER (...) — a window function: computed per row over related rows"),
        ("PARTITION BY", "OVER (PARTITION BY a) — restart the window for each a"),
        ("ROW_NUMBER(", "ROW_NUMBER() — 1, 2, 3 ... in the window's order"),
        ("RANK(", "RANK() — ties share a rank, then a gap"),
        ("DENSE_RANK(", "DENSE_RANK() — ties share a rank, no gap"),
        ("NTILE(", "NTILE(n) — split the rows into n buckets"),
        ("LAG(", "LAG(x, k, default) — x from k rows before"),
        ("LEAD(", "LEAD(x, k, default) — x from k rows after"),
        ("FIRST_VALUE(", "FIRST_VALUE(x) — x of the window frame's first row"),
        ("LAST_VALUE(", "LAST_VALUE(x) — x of the frame's last row (needs ROWS BETWEEN ... UNBOUNDED FOLLOWING)"),
        ("ROWS", "ROWS BETWEEN k PRECEDING AND CURRENT ROW — a frame of physical rows"),
        ("PRECEDING", "k PRECEDING / FOLLOWING, UNBOUNDED PRECEDING — frame bounds"),
        ("WINDOW", "WINDOW w AS (...) — name a window, then OVER w"),
        ("PERCENT_RANK(", "PERCENT_RANK() / CUME_DIST() — relative rank between 0 and 1"),
    ],
    "10.07": [
        ("LENGTH(", "LENGTH(s) — number of characters"),
        ("UPPER(", "UPPER / LOWER(s) — change case"),
        ("LOWER(", "LOWER(s) — lower case"),
        ("INITCAP(", "INITCAP(s) — capitalise each word"),
        ("TRIM(", "TRIM / LTRIM / RTRIM(s) — remove spaces (or given characters)"),
        ("SUBSTRING(", "SUBSTRING(s FROM i FOR n) / SUBSTRING(s FROM 'regex')"),
        ("LEFT(", "LEFT / RIGHT(s, n) — first / last n characters"),
        ("RIGHT(", "RIGHT(s, n) — last n characters"),
        ("POSITION(", "POSITION('x' IN s) / STRPOS(s, 'x') — 1-based position, 0 if absent"),
        ("STRPOS(", "STRPOS(s, 'x') — position of a substring"),
        ("REPLACE(", "REPLACE(s, from, to) — replace every occurrence"),
        ("SPLIT_PART(", "SPLIT_PART(s, sep, n) — the n-th field"),
        ("CONCAT(", "CONCAT(a, b) — concatenate, skipping NULLs"),
        ("CONCAT_WS(", "CONCAT_WS(sep, a, b) — join with a separator, skipping NULLs"),
        ("LPAD(", "LPAD / RPAD(s, n, fill) — pad to n characters"),
        ("FORMAT(", "FORMAT('%s has %s', a, b) — printf-like (%I identifier, %L literal)"),
        ("REGEXP_REPLACE(", "REGEXP_REPLACE(s, regex, repl, 'g') — regex replace ('g': every match)"),
        ("~", "s ~ 'regex' — regex match (~* ignores case, !~ doesn't match)"),
        ("~*", "s ~* 'regex' — case-insensitive regex match"),
    ],
    "10.08": [
        ("::DATE", "ts::DATE — the day of a timestamp (the time dropped)"),
        ("EXTRACT(", "EXTRACT(YEAR FROM ts) — a field (MONTH, DOW 0=Sunday, EPOCH seconds ...)"),
        ("DATE_PART(", "DATE_PART('hour', ts) — EXTRACT as a function (double precision)"),
        ("INTERVAL", "INTERVAL '7 days' — a duration; ts + interval, ts - ts gives an interval"),
        ("TO_CHAR(", "TO_CHAR(ts, 'YYYY-MM-DD HH24:MI') — format a date / number"),
        ("TO_DATE(", "TO_DATE('15/03/2024', 'DD/MM/YYYY') — parse a date"),
        ("AGE(", "AGE(a, b) — symbolic difference: 1 year 2 mons 3 days"),
        ("GENERATE_SERIES(", "GENERATE_SERIES(start, stop, step) — a row per value (a date spine)"),
        ("AT TIME ZONE", "ts AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Tokyo' — UTC wall time to local"),
    ],
    "10.09": [
        ("CEIL(", "CEIL / FLOOR(x) — round up / down"),
        ("FLOOR(", "FLOOR(x) — round down"),
        ("ABS(", "ABS(x) — absolute value"),
        ("POWER(", "POWER(a, b) — a to the b; SQRT(x) square root"),
        ("SQRT(", "SQRT(x) — square root"),
        ("MOD(", "MOD(a, b) / a % b — remainder (sign of a)"),
        ("LN(", "LN / LOG / EXP — natural log, base-10 log, e to the x"),
        ("TRUNC(", "TRUNC(x, n) — cut (not round) to n decimals"),
        ("WIDTH_BUCKET(", "WIDTH_BUCKET(x, lo, hi, n) — histogram bucket 1..n (0 below, n+1 at or above hi)"),
        ("CORR(", "CORR(y, x) / REGR_SLOPE(y, x) — correlation / regression slope"),
    ],
    "10.10": [
        ("CREATE TABLE", "CREATE TABLE t (col type constraints, ...)"),
        ("PRIMARY KEY", "PRIMARY KEY — unique and not null; (a, b) for a composite key"),
        ("NOT NULL", "NOT NULL — the column must have a value"),
        ("UNIQUE", "UNIQUE — no two rows share the value (NULLs allowed)"),
        ("DEFAULT", "DEFAULT expr — the value when an INSERT doesn't give one"),
        ("CHECK", "CHECK (cond) — reject rows where cond is false"),
        ("REFERENCES", "REFERENCES t(id) [ON DELETE CASCADE] — a foreign key"),
        ("GENERATED ALWAYS AS IDENTITY", "GENERATED ALWAYS AS IDENTITY — an auto-incrementing id"),
        ("ALTER TABLE", "ALTER TABLE t ADD / DROP / RENAME COLUMN, ALTER COLUMN ... TYPE / SET DEFAULT / SET NOT NULL"),
        ("DROP TABLE", "DROP TABLE [IF EXISTS] t"),
        ("TRUNCATE", "TRUNCATE t — remove every row, keep the table"),
        ("CREATE SCHEMA", "CREATE SCHEMA s — a namespace; s.t names a table in it"),
    ],
    "10.11": [
        ("INSERT", "INSERT INTO t (cols) VALUES (...), (...) / INSERT INTO t SELECT ..."),
        ("UPDATE", "UPDATE t SET col = expr WHERE ... [FROM other] — no WHERE changes every row"),
        ("DELETE", "DELETE FROM t WHERE ... [USING other]"),
        ("RETURNING", "... RETURNING cols — get the inserted / changed / deleted rows back"),
        ("ON CONFLICT", "INSERT ... ON CONFLICT (key) DO NOTHING / DO UPDATE SET ... — upsert"),
        ("EXCLUDED", "EXCLUDED.col — the row the INSERT tried to add (in ON CONFLICT DO UPDATE)"),
        ("MERGE", "MERGE INTO t USING s ON cond WHEN MATCHED THEN UPDATE ... WHEN NOT MATCHED THEN INSERT ..."),
    ],
    "10.12": [
        ("BEGIN", "BEGIN — start a transaction (all or nothing until COMMIT)"),
        ("COMMIT", "COMMIT — make the transaction's changes permanent"),
        ("ROLLBACK", "ROLLBACK — undo everything since BEGIN"),
        ("SAVEPOINT", "SAVEPOINT s — a point to go back to inside a transaction"),
        ("ROLLBACK TO", "ROLLBACK TO SAVEPOINT s — undo back to s (also leaves an aborted state)"),
        ("CREATE VIEW", "CREATE [OR REPLACE] VIEW v AS query — a saved query"),
        ("OR REPLACE", "CREATE OR REPLACE VIEW v AS ... — redefine a view in place (existing columns stay first)"),
        ("MATERIALIZED", "CREATE MATERIALIZED VIEW v AS query — a stored result; REFRESH MATERIALIZED VIEW v"),
        ("REFRESH", "REFRESH MATERIALIZED VIEW v — recompute it"),
        ("CREATE INDEX", "CREATE [UNIQUE] INDEX name ON t (cols | (expr)) [WHERE cond]"),
    ],
    "10.13": [
        ("->", "json -> 'key' / -> 0 — a JSON value (arrays 0-indexed)"),
        ("->>", "json ->> 'key' — a value as text"),
        ("#>>", "json #>> '{a,b}' — a nested value as text"),
        ("@>", "a @> b — a contains b (JSONB or arrays)"),
        ("?", "jsonb ? 'key' — the key exists"),
        ("JSONB_BUILD_OBJECT(", "JSONB_BUILD_OBJECT('k', v, ...) — build an object"),
        ("JSONB_AGG(", "JSONB_AGG(x ORDER BY ...) — rows into a JSON array"),
        ("JSONB_ARRAY_ELEMENTS_TEXT(", "JSONB_ARRAY_ELEMENTS_TEXT(arr) — one row per element (in FROM)"),
        ("JSONB_SET(", "JSONB_SET(doc, '{path}', value) — a copy with one value replaced"),
        ("JSONB_EACH_TEXT(", "JSONB_EACH_TEXT(obj) — one (key, value) row per key"),
        ("UNNEST(", "UNNEST(array) — one row per element"),
        ("ANY", "x = ANY(array) — x is in the array"),
        ("ARRAY_LENGTH(", "ARRAY_LENGTH(arr, 1) — elements along dimension 1 (arrays are 1-indexed)"),
    ],
    "10.14": [
        ("EXPLAIN", "EXPLAIN query — the plan: Seq Scan / Index Scan, estimated costs and rows"),
        ("EXPLAIN ANALYZE", "EXPLAIN ANALYZE query — runs it: actual rows and times"),
        ("sargable range", "col >= a AND col < b instead of f(col) = x — a condition an index on col can use"),
        ("ANALYZE", "ANALYZE t — refresh the planner's statistics"),
    ],
    "10.15": [
        ("-c", "psql -d db -c \"SQL\" — run one command and exit"),
        ("-f", "psql -d db -f file.sql — run a script"),
        ("\\dt", "\\dt — list tables (\\d t describes one, \\l databases, \\dn schemas)"),
        ("\\d", "\\d t — a table's columns, indexes and constraints"),
        ("\\x", "\\x — expanded display: one column per line (\\x auto)"),
        ("\\timing", "\\timing on — show how long each query took"),
        ("\\pset", "\\pset null '(null)' / \\pset format csv — output settings"),
        ("\\copy", "\\copy t TO 'f.csv' CSV HEADER — export (FROM to import) through the client"),
        ("\\conninfo", "\\conninfo — where you are connected"),
        ("\\l", "\\l — list the databases"),
        ("sqlite3", "sqlite3 file.db \"SQL\" — SQLite's shell (.tables .schema .mode csv .import f t)"),
        ("executemany", "cur.executemany(sql, rows) — one statement, many parameter tuples (? placeholders)"),
    ],
    "10.16": [
        ("DISTINCT ON", "SELECT DISTINCT ON (key) ... ORDER BY key, ... — the first row per key"),
        ("HASHTEXT(", "HASHTEXT(id::TEXT) — a stable hash (deterministic train / val / test split)"),
    ],
}

CATALOG: dict[str, str] = {}
MODULE_OF: dict[str, str] = {}
for _m, _items in _BY_MODULE.items():
    for _k, _d in _items:
        if _k not in CATALOG:
            CATALOG[_k] = _d
            MODULE_OF[_k] = _m

# entries matched by a pattern rather than as a keyword / function / operator (sqlcheck.feature_used)
PATTERNS = {
    "DATE '...'": r"\b(DATE|TIMESTAMP)\s*'",
    "IN (SELECT ...)": r"\bIN\s*\(\s*SELECT\b",
    "(SELECT ...)": r"(?<!\bIN)(?<!\bEXISTS)\s*\(\s*SELECT\b",
    "::DATE": r"::\s*DATE\b",
    "sargable range": r"\b(\w+)\s*>=\s*(DATE|TIMESTAMP)\s*'[^']*'\s+AND\s+\1\s*<\s*(DATE|TIMESTAMP)\s*'",
}


def shown(k: str) -> str:
    """The name shown (functions without their parenthesis)."""
    return k[:-1] if k.endswith("(") else k


def uses(text: str) -> list[str]:
    """The catalog entries a solution uses, in catalog order."""
    import re
    from ..bashlite.sqlcheck import feature_used
    out = []
    for k in CATALOG:
        pat = PATTERNS.get(k)
        hit = re.search(pat, text, re.I) is not None if pat else feature_used(text, k)
        if hit and k == "(SELECT ...)":    # an IN / EXISTS subquery is not a scalar one
            hit = any(not re.search(r"\b(IN|EXISTS)\s*$", text[:m.start()], re.I)
                      for m in re.finditer(r"\(\s*SELECT\b", text, re.I))
        if hit:
            out.append(k)
    return out

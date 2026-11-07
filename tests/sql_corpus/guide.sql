SELECT * FROM users;
SELECT name, country FROM users;
SELECT name AS customer, signup_date AS joined FROM users;
SELECT DISTINCT country FROM users;
SELECT * FROM orders WHERE amount > 50;
SELECT * FROM orders WHERE status = 'paid' AND amount BETWEEN 20 AND 150;   -- BETWEEN is inclusive
SELECT * FROM users WHERE country IN ('US', 'UK');
SELECT * FROM users WHERE country IS NULL;            -- never "= NULL" (that is always unknown)
SELECT * FROM users WHERE name LIKE 'A%';             -- % any characters, _ exactly one; ILIKE ignores case
SELECT * FROM users WHERE country <> 'US';            -- NULL countries are NOT returned (NULL <> 'US' is unknown)
SELECT * FROM users WHERE country IS DISTINCT FROM 'US';   -- NULL-safe "not equal"
SELECT * FROM orders ORDER BY amount DESC;
SELECT * FROM orders ORDER BY status, amount DESC;
SELECT * FROM users ORDER BY country NULLS LAST;
SELECT * FROM orders ORDER BY created_at DESC LIMIT 3;
SELECT * FROM orders ORDER BY id LIMIT 3 OFFSET 3;    -- page 2 (standard SQL: OFFSET 3 ROWS FETCH FIRST 3 ROWS ONLY)
-- line comment      /* block comment */

-- ===
SELECT id, amount,
       amount * 1.2 AS with_tax,
       ROUND(amount * 0.9, 2) AS discounted,
       CASE WHEN amount >= 100 THEN 'large'
            WHEN amount >= 40  THEN 'medium'
            ELSE 'small' END AS size
FROM orders;
SELECT name, COALESCE(country, 'unknown') AS country FROM users;   -- first non-NULL argument
SELECT id, NULLIF(status, 'pending') AS status FROM orders;         -- NULL when the two are equal
SELECT CAST(amount AS INTEGER), amount::TEXT, '2024-01-01'::DATE FROM orders;
SELECT 7 / 2 AS int_div, 7 / 2.0 AS dec_div, 7::NUMERIC / 2 AS cast_div;   -- 3, 3.5, 3.5: integer division trap
SELECT name || ' (' || COALESCE(country, '?') || ')' AS label FROM users;   -- || concatenates (NULL || 'x' is NULL)
SELECT NULL = NULL AS eq, NULL IS NULL AS is_null, 1 + NULL AS plus;  -- NULL, true, NULL: NULL propagates
SELECT GREATEST(1, 5, 3), LEAST(1, 5, 3);

-- ===
SELECT COUNT(*), COUNT(country), COUNT(DISTINCT country) FROM users;   -- all rows / non-NULL / distinct non-NULL
SELECT SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM orders;
SELECT status, COUNT(*) AS n, SUM(amount) AS revenue FROM orders GROUP BY status ORDER BY revenue DESC;
SELECT user_id, COUNT(*) AS n_orders FROM orders GROUP BY user_id HAVING COUNT(*) >= 2;   -- HAVING filters GROUPS
SELECT user_id,
       COUNT(*) FILTER (WHERE status = 'paid') AS paid_orders,              -- conditional aggregate
       SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) AS paid_revenue   -- portable equivalent
FROM orders GROUP BY user_id;
SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) FROM orders GROUP BY 1 ORDER BY 1;   -- GROUP BY / ORDER BY position
SELECT STRING_AGG(name, ', ' ORDER BY name) AS names, ARRAY_AGG(id) AS ids FROM users;
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median, STDDEV(amount), VARIANCE(amount) FROM orders;
SELECT u.country, o.status, SUM(o.amount) FROM orders o JOIN users u ON u.id = o.user_id
GROUP BY ROLLUP (u.country, o.status);               -- subtotals per country plus a grand total (also: CUBE, GROUPING SETS)
SELECT BOOL_AND(amount > 0) AS all_positive, BOOL_OR(status = 'refunded') AS any_refund FROM orders;

-- ===
SELECT u.name, o.amount FROM users u INNER JOIN orders o ON o.user_id = u.id;   -- only matching rows
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id;    -- every user; NULL amount for Dee
SELECT u.name, o.id FROM users u RIGHT JOIN orders o ON o.user_id = u.id;
SELECT u.name, o.id FROM users u FULL OUTER JOIN orders o ON o.user_id = u.id;
SELECT u.name, o.id FROM users u CROSS JOIN orders o;                           -- every pair: 5 × 7 = 35 rows
SELECT u.name FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL;           -- ANTI-join: users with no orders
SELECT u.name FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);    -- same idea, usually faster
SELECT u.name FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 100);   -- SEMI-join
SELECT a.name, b.name FROM users a JOIN users b ON a.country = b.country AND a.id < b.id;       -- self-join: pairs per country
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON o.user_id = u.id AND o.status = 'paid';
-- ^ condition in ON keeps users without paid orders; the same condition in WHERE would drop them
SELECT u.name, latest.amount, latest.created_at
FROM users u
LEFT JOIN LATERAL (SELECT amount, created_at FROM orders o WHERE o.user_id = u.id ORDER BY created_at DESC LIMIT 1) latest ON TRUE;   -- per-row subquery

-- ===
SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders);   -- scalar subquery
SELECT name FROM users WHERE id IN (SELECT user_id FROM orders WHERE status = 'refunded');
SELECT o.* FROM orders o
WHERE o.amount = (SELECT MAX(o2.amount) FROM orders o2 WHERE o2.user_id = o.user_id);   -- correlated: each user's largest order
SELECT t.user_id, t.total
FROM (SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id) AS t
WHERE t.total > 100;                                -- derived table
WITH user_totals AS (
  SELECT user_id, SUM(amount) AS total FROM orders WHERE status = 'paid' GROUP BY user_id
), ranked AS (
  SELECT *, RANK() OVER (ORDER BY total DESC) AS rnk FROM user_totals
)
SELECT u.name, r.total, r.rnk FROM ranked r JOIN users u ON u.id = r.user_id ORDER BY r.rnk;   -- CTEs read top to bottom
WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 10)
SELECT * FROM nums;                                 -- recursive CTE (hierarchies, sequences)
SELECT country FROM users UNION SELECT 'FR';        -- union, duplicates removed
SELECT user_id FROM orders UNION ALL SELECT id FROM users;   -- keeps duplicates (faster)
SELECT id FROM users INTERSECT SELECT user_id FROM orders;   -- users with at least one order
SELECT id FROM users EXCEPT SELECT user_id FROM orders;      -- users with none

-- ===
SELECT id, user_id, amount,
       ROW_NUMBER()   OVER (PARTITION BY user_id ORDER BY created_at) AS order_n,   -- 1, 2, 3 within each user
       RANK()         OVER (ORDER BY amount DESC) AS rnk,          -- ties share a rank, then a gap
       DENSE_RANK()   OVER (ORDER BY amount DESC) AS dense_rnk,    -- ties share a rank, no gap
       NTILE(4)       OVER (ORDER BY amount) AS quartile,
       PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank
FROM orders;
SELECT id, user_id, created_at, amount,
       LAG(amount)  OVER w AS prev_amount,                         -- previous row in the window
       LEAD(amount) OVER w AS next_amount,
       amount - LAG(amount) OVER w AS change,
       FIRST_VALUE(amount) OVER w AS first_amount,
       SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
       AVG(amount) OVER (PARTITION BY user_id ORDER BY created_at ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3,
       SUM(amount) OVER (PARTITION BY user_id) AS user_total,      -- no ORDER BY: the whole partition
       amount / SUM(amount) OVER () AS share_of_all                -- empty OVER (): the whole table
FROM orders
WINDOW w AS (PARTITION BY user_id ORDER BY created_at);
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn FROM orders
)
SELECT * FROM ranked WHERE rn = 1;                  -- top-N per group: window functions can't go in WHERE, so wrap them
-- DuckDB / Snowflake / BigQuery shortcut: SELECT * FROM orders QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) = 1
-- LAST_VALUE needs a full frame: LAST_VALUE(x) OVER (... ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

-- ===
SELECT LENGTH(name), UPPER(name), LOWER(name), INITCAP('hello world'), TRIM('  x  '), LTRIM('xxa', 'x'), RTRIM('a  ') FROM users;
SELECT SUBSTRING(name FROM 1 FOR 2), LEFT(name, 2), RIGHT(name, 1), POSITION('e' IN name), STRPOS(name, 'e') FROM users;
SELECT REPLACE('a-b-c', '-', '_'), SPLIT_PART('2024-03-15', '-', 2), CONCAT('a', NULL, 'b'), CONCAT_WS('/', 'data', 'raw', 'x.csv');
-- CONCAT skips NULLs ('ab') while || returns NULL
SELECT LPAD('7', 3, '0'), RPAD('ab', 4, '.'), REVERSE('abc'), REPEAT('ab', 3), FORMAT('%s has %s orders', 'Ana', 2);
SELECT REGEXP_REPLACE('order-123-x', '[^0-9]', '', 'g') AS digits,
       'abc123' ~ '^[a-z]+[0-9]+$' AS matches,          -- ~ regex match (~* case-insensitive)
       SUBSTRING('run_042' FROM '[0-9]+') AS run_id;
SELECT name FROM users WHERE name ILIKE 'a%' OR name ~* '^b';

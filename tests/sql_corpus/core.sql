SELECT nme FROM users;
SELECT name FROM users u JOIN orders o ON o.user_id = u.id WHERE id = 1;
SELECT country, count(*) FROM users;
SELEC 1;
SELECT 1/0;
SELECT users.name FROM users u;
SELECT x.name FROM users u;
SELECT 'abc' + 1;
SELECT round(avg(amount)::float8, 2) FROM orders;
SELECT * FROM orders WHERE created_at > '2024-13-01';
SELECT amount::int FROM orders WHERE id = 7;
SELECT '12a'::int;
SELECT name FROM users ORDER BY 5;
SELECT name, count(*) FROM users GROUP BY 3;
SELECT * FROM users WHERE country = NULL;
SELECT (SELECT id FROM users);
SELECT * FROM nope;
SELECT id FROM users UNION SELECT name FROM users;
SELECT DISTINCT name FROM users ORDER BY signup_date;
SELECT row_number() FROM users;
SELECT * FROM users WHERE row_number() OVER () = 1;
SELECT sum(count(*)) FROM users;
SELECT * FROM users LIMIT -1;
SELECT CASE WHEN true THEN 1 ELSE 'x' END;
SELECT * FROM orders o JOIN users u USING (id) JOIN users u2 USING (id);
SELECT 1 FROM users u, users u;
-- ===
INSERT INTO users VALUES (1, 'X', 'US', '2024-01-01');
INSERT INTO users VALUES (9, NULL, 'US', '2024-01-01');
INSERT INTO orders VALUES (99, 1, -5, 'paid', '2024-01-01');
INSERT INTO orders VALUES (99, 42, 5, 'paid', '2024-01-01');
DELETE FROM users WHERE id = 1;
INSERT INTO users (id, name) VALUES (10, 'Zed');
INSERT INTO users VALUES (11, 'A', 'B', '2024-01-01', 5);
INSERT INTO users (id, name, signup_date) VALUES (11, 'A', 'x');
INSERT INTO orders (id, user_id, amount, created_at) VALUES (8, 1, 12345678901.5, now());
UPDATE orders SET amount = amount WHERE id < 0;
DELETE FROM orders WHERE id > 1000;
INSERT INTO users VALUES (6, 'Fay', 'FR', '2024-04-01'), (7, 'Gus', NULL, '2024-04-02') RETURNING *;
UPDATE users SET country = 'US' WHERE country IS NULL RETURNING id, name;
SELECT * FROM users;
-- ===
\d orders
\d users
\dt
\di
\d nope
\dn
-- ===
BEGIN;
SELECT x;
SELECT 1;
ROLLBACK;
COMMIT;
BEGIN;
INSERT INTO users VALUES (6, 'F', 'FR', '2024-04-01');
SAVEPOINT s1;
DELETE FROM orders;
SELECT count(*) FROM orders;
ROLLBACK TO SAVEPOINT s1;
SELECT count(*) FROM orders;
RELEASE s1;
COMMIT;
SELECT count(*) FROM users;
ROLLBACK TO nope;
SAVEPOINT x;
-- ===
SELECT id, user_id, amount, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) AS n, RANK() OVER (ORDER BY amount DESC) AS r, DENSE_RANK() OVER (ORDER BY status) AS dr, NTILE(3) OVER (ORDER BY id) AS t, round(PERCENT_RANK() OVER (ORDER BY amount)::numeric, 3) AS pr, CUME_DIST() OVER (ORDER BY status) AS cd FROM orders ORDER BY id;
SELECT id, amount, SUM(amount) OVER (ORDER BY created_at) AS run, AVG(amount) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS mov, LAG(amount, 2, 0) OVER (ORDER BY id) AS l2, LEAD(id) OVER w AS nx, FIRST_VALUE(id) OVER w AS fv, LAST_VALUE(id) OVER w AS lv, NTH_VALUE(id, 2) OVER w AS nv FROM orders WINDOW w AS (PARTITION BY status ORDER BY id) ORDER BY id;
SELECT status, amount, SUM(amount) OVER (PARTITION BY status ORDER BY amount RANGE BETWEEN 50 PRECEDING AND CURRENT ROW) AS r, COUNT(*) OVER (ORDER BY created_at RANGE BETWEEN INTERVAL '10 days' PRECEDING AND CURRENT ROW) AS c10, SUM(amount) OVER (ORDER BY status GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW) AS g FROM orders ORDER BY id;
SELECT user_id, count(*) FILTER (WHERE status = 'paid') AS p, string_agg(status, ',' ORDER BY id DESC) AS s, array_agg(DISTINCT status) AS a, bool_and(amount > 20), percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) AS med, percentile_disc(0.5) WITHIN GROUP (ORDER BY amount) AS md, mode() WITHIN GROUP (ORDER BY status) AS mo FROM orders GROUP BY user_id ORDER BY user_id;
SELECT stddev(amount), variance(amount), stddev_pop(amount), var_pop(amount), avg(amount), sum(amount), avg(id), sum(id), avg(amount::float8), stddev(id) FROM orders;
SELECT country, status, sum(amount), grouping(country, status) FROM orders o JOIN users u ON u.id = o.user_id GROUP BY CUBE (country, status) ORDER BY 1, 2;
SELECT DISTINCT ON (user_id) user_id, id, amount FROM orders ORDER BY user_id, amount DESC;
SELECT * FROM orders ORDER BY amount DESC FETCH FIRST 2 ROWS WITH TIES;
SELECT id FROM orders ORDER BY status NULLS FIRST, id DESC LIMIT 3 OFFSET 1;
-- ===
SELECT payload->'meta' AS m, payload->>'user' AS u, payload#>>'{meta,x}' AS x, payload->'meta'->'tags'->>0 AS t, jsonb_typeof(payload->'meta'), payload @> '{"action":"click"}' AS c, payload ? 'meta' AS k FROM (SELECT '{"user": "ana", "action": "click", "meta": {"x": 10, "tags": ["a", "b"]}}'::jsonb AS payload) e;
SELECT jsonb_build_object('id', 1, 'name', 'x', 'n', NULL, 'f', 1.50), jsonb_agg(id ORDER BY id), json_build_object('a', 1, 'b', ARRAY[1,2]), to_jsonb(u) FROM users u WHERE id < 3 GROUP BY u.id ORDER BY 1;
SELECT e.k, t.tag FROM (VALUES (1, '["x","y"]'::jsonb), (2, '[]'::jsonb)) e(k, j), jsonb_array_elements_text(e.j) AS t(tag);
SELECT key, value FROM jsonb_each('{"b": 1, "a": [1, {"c": 2}], "zz": null}');
SELECT jsonb_set('{"a": {"b": 1}}', '{a,c}', '2'), '{"a": 1, "b": 2}'::jsonb - 'a', '[1,2,3]'::jsonb - 0, jsonb_pretty('{"a": [1, 2], "b": {"c": null}}'), '{"a":1}'::jsonb || '{"b":2}';
SELECT ARRAY[1, 2, 3] AS arr, (ARRAY[10, 20, 30])[1] AS first, ARRAY_LENGTH(ARRAY[1, 2], 1) AS len, 2 = ANY(ARRAY[1, 2, 3]) AS has_two, ARRAY[1,2] || 3, ARRAY['a'] @> ARRAY['a'], cardinality(ARRAY[[1,2],[3,4]]), (ARRAY[1,2,3,4])[2:3], array_to_string(ARRAY[1,NULL,3], '-', '*'), string_to_array('a,b,,c', ',');
SELECT UNNEST(ARRAY['a', 'b', 'c']) AS letter, generate_series(1, 2) AS g;
SELECT * FROM unnest(ARRAY[1,2], ARRAY['x','y','z']) WITH ORDINALITY AS t(n, s, o);
-- ===
SELECT CURRENT_DATE - CURRENT_DATE, DATE '2024-03-15' - DATE '2024-01-05' AS d, AGE(DATE '2024-03-15', DATE '2024-01-05') AS a, AGE(TIMESTAMP '2024-03-01', TIMESTAMP '2023-01-15 10:00') AS b, TIMESTAMP '2024-01-31 10:00' + INTERVAL '1 month' AS c, DATE '2024-02-29' + 1, TIMESTAMP '2024-03-01 19:45' - TIMESTAMP '2024-01-06 10:00' AS diff;
SELECT DATE_TRUNC('month', created_at), DATE_TRUNC('week', created_at), EXTRACT(YEAR FROM created_at), EXTRACT(DOW FROM created_at), EXTRACT(EPOCH FROM created_at), DATE_PART('hour', created_at), EXTRACT(SECOND FROM created_at), EXTRACT(QUARTER FROM created_at) FROM orders WHERE id < 3;
SELECT TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI'), TO_CHAR(created_at, 'Mon DD, YYYY'), TO_CHAR(created_at, 'FMDay'), TO_DATE('15/03/2024', 'DD/MM/YYYY'), TO_TIMESTAMP(1700000000), to_char(amount, 'FM999,999.00'), to_char(amount, '00000.0') FROM orders WHERE id < 3;
SELECT d::DATE AS day FROM GENERATE_SERIES(DATE '2024-01-01', DATE '2024-01-07', INTERVAL '2 day') AS d;
SELECT created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' AS local_time, created_at::timestamptz AT TIME ZONE 'Asia/Tokyo' FROM orders WHERE id = 1;
SELECT interval '1 day 2 hours' * 3, interval '90 minutes', interval '1.5 days', interval '-1 day 2 hours', interval '1 year 14 months', justify_hours(interval '50 hours'), interval '1 mon' = interval '30 days', make_interval(days => 3);
SELECT '2024-01-05'::date, '1/8/1999'::date, 'January 8, 1999'::date, '2024-01-06 10:00'::timestamp, '2024-01-06T10:00:00Z'::timestamptz, 'epoch'::timestamp, 'infinity'::date > '2024-01-01';
-- ===
SELECT LENGTH(name), UPPER(name), INITCAP('hello wORLD foo-bar'), TRIM('  x  '), LTRIM('xxa', 'x'), RTRIM('a  ') || '|', SUBSTRING(name FROM 1 FOR 2), LEFT(name, -1), RIGHT(name, 1), POSITION('e' IN name), STRPOS(name, 'e') FROM users;
SELECT REPLACE('a-b-c', '-', '_'), SPLIT_PART('2024-03-15', '-', 2), CONCAT('a', NULL, 'b'), CONCAT_WS('/', 'data', NULL, 'x.csv'), LPAD('7', 3, '0'), RPAD('ab', 4, '.'), REVERSE('abc'), REPEAT('ab', 3), FORMAT('%s has %s orders, %I %L', 'Ana', 2, 'my col', 'it''s');
SELECT REGEXP_REPLACE('order-123-x', '[^0-9]', '', 'g') AS digits, 'abc123' ~ '^[a-z]+[0-9]+$' AS m, SUBSTRING('run_042' FROM '[0-9]+') AS run_id, regexp_match('a1b22', '([a-z])(\d+)'), regexp_replace('Hello World', '(\w+) (\w+)', '\2 \1'), 'ABC' ~* 'b', regexp_split_to_array('a1b2c', '\d');
SELECT name FROM users WHERE name ILIKE 'a%' OR name ~* '^b' OR name LIKE '_e_' ORDER BY 1;
SELECT md5('abc'), hashtext('42'), abs(hashtext('1')) % 10, quote_ident('Mixed'), quote_literal('it''s'), chr(65), ascii('A'), translate('abc', 'ab', 'x');
-- ===
SELECT ROUND(3.14159, 2), CEIL(2.1), FLOOR(2.9), ABS(-5), POWER(2, 10), SQRT(16), MOD(17, 5), 17 % 5, LN(10), LOG(100), EXP(1), round(2.5), round(-2.5), round(2.5::float8), trunc(-2.789, 1), sign(-3.5), 7 / 2, -7 / 2, -7 % 3, 7 / 2.0, 1 / 3.0, 2 ^ 0.5, 9.0 ^ 0.5, 1.1 ^ 10;
SELECT WIDTH_BUCKET(amount, 0, 250, 5) AS bucket, COUNT(*) FROM orders GROUP BY 1 ORDER BY 1;
SELECT width_bucket(1.0, 0, 1, 10), width_bucket(0, 0, 1, 10), width_bucket(-1, 0, 1, 10), width_bucket(5, ARRAY[1, 3, 5, 7]);
SELECT 100.0 * 3 / 7, round(100.0 * 3 / 7, 1), 3 / 7 * 100, (3::numeric / 7)::numeric(5,2), 1e3, 1.50 * 2, 0.1 + 0.2, 0.1::float8 + 0.2, 2147483647 + 1;
SELECT 1e400::float8;
SELECT sum(x) FROM generate_series(2147483640, 2147483647) x;
SELECT avg(x), sum(x::numeric), avg(x::float8) FROM generate_series(1, 10) x;
-- ===
CREATE TABLE products (
  id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  sku        TEXT UNIQUE NOT NULL,
  name       VARCHAR(200) NOT NULL,
  price      NUMERIC(10, 2) NOT NULL CHECK (price > 0),
  tags       TEXT[] DEFAULT '{}',
  attrs      JSONB,
  in_stock   BOOLEAN DEFAULT TRUE
);
\d products
INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU', 1999.00), ('B-2', 'Cable', 9.50);
INSERT INTO products (id, sku, name, price) VALUES (5, 'C', 'x', 1);
INSERT INTO products (sku, name, price) VALUES ('A-1', 'GPU v2', 2099.00) ON CONFLICT (sku) DO UPDATE SET name = EXCLUDED.name, price = EXCLUDED.price;
INSERT INTO products (sku, name, price) VALUES ('A-1', 'dup', 1.00) ON CONFLICT (sku) DO NOTHING;
INSERT INTO products (sku, name, price) VALUES ('Z', 'x', 0);
INSERT INTO products (sku, name, price) VALUES ('Q', repeat('x', 201), 1);
MERGE INTO products p USING (VALUES ('B-2', 8.99), ('D-4', 15.00)) AS s(sku, price) ON p.sku = s.sku WHEN MATCHED THEN UPDATE SET price = s.price WHEN NOT MATCHED THEN INSERT (sku, name, price) VALUES (s.sku, 'new item', s.price);
SELECT * FROM products;
ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users RENAME COLUMN email TO contact_email;
ALTER TABLE users DROP COLUMN contact_email;
ALTER TABLE users ALTER COLUMN country SET DEFAULT 'US';
ALTER TABLE products ALTER COLUMN price TYPE NUMERIC(12, 2);
ALTER TABLE users ADD CONSTRAINT users_name_unique UNIQUE (name);
ALTER TABLE users ALTER COLUMN name TYPE integer;
ALTER TABLE users ALTER COLUMN country SET NOT NULL;
CREATE TABLE big_orders AS SELECT * FROM orders WHERE amount > 100;
CREATE TABLE users (x int);
CREATE TABLE IF NOT EXISTS users (x int);
DROP TABLE users;
DROP TABLE IF EXISTS nope;
CREATE VIEW paid_orders AS SELECT * FROM orders WHERE status = 'paid';
DROP TABLE orders;
DROP TABLE orders CASCADE;
\d users
\dt
-- ===
CREATE INDEX idx_orders_user ON orders (user_id);
CREATE INDEX ON orders (user_id, created_at DESC);
CREATE UNIQUE INDEX idx_users_lower_name ON users (LOWER(name));
CREATE INDEX idx_orders_paid ON orders (created_at) WHERE status = 'paid';
INSERT INTO users VALUES (6, 'ANA', 'US', '2024-01-01');
\d orders
\di
DROP INDEX idx_orders_paid;
DROP INDEX orders_pkey;
CREATE MATERIALIZED VIEW monthly_revenue AS SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS revenue FROM orders GROUP BY 1;
SELECT * FROM monthly_revenue ORDER BY 1;
REFRESH MATERIALIZED VIEW monthly_revenue;
CREATE VIEW v AS SELECT id, name FROM users WHERE country = 'US';
\d v
TRUNCATE users;
TRUNCATE users CASCADE;
SELECT count(*) FROM orders;

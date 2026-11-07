CREATE TABLE users (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  country     TEXT,
  signup_date DATE NOT NULL
);
CREATE TABLE orders (
  id         INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  amount     NUMERIC(10, 2) NOT NULL CHECK (amount >= 0),
  status     TEXT NOT NULL DEFAULT 'paid',
  created_at TIMESTAMP NOT NULL
);
INSERT INTO users VALUES
  (1, 'Ana', 'US', '2024-01-05'), (2, 'Ben', 'UK', '2024-01-20'), (3, 'Cal', 'US', '2024-02-11'),
  (4, 'Dee', NULL, '2024-03-02'), (5, 'Eve', 'DE', '2024-03-15');
INSERT INTO orders VALUES
  (1, 1, 120.00, 'paid', '2024-01-06 10:00'), (2, 1, 35.50, 'paid', '2024-02-01 09:30'),
  (3, 2, 80.00, 'refunded', '2024-02-03 14:00'), (4, 3, 15.00, 'paid', '2024-02-12 08:15'),
  (5, 3, 220.00, 'paid', '2024-03-01 19:45'), (6, 3, 42.00, 'paid', '2024-03-20 12:00'),
  (7, 5, 99.99, 'pending', '2024-03-16 16:20');

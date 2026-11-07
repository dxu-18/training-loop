"""COMP_GUIDE §10 SQL: one module per subsection (10.01-10.16), PostgreSQL 16 dialect.

Problems open psql on bash-lite's machine, whose PostgreSQL server is generator/postgreslite (a simulator checked
against real PostgreSQL 16). bashlite/sqlcheck.py grades the session: query results on the shown database and on
hidden ones, the end state of tables, plans and psql settings. Problems have kind "bash" with a `sql` spec.
"""

from . import m01_select, m02_expressions, m03_aggregation, m04_joins, m05_subqueries, m06_windows, m07_strings, m08_dates, m09_math, m10_ddl, m11_dml, m12_transactions, m13_json, m14_performance, m15_cli, m16_analytics  # noqa: F401

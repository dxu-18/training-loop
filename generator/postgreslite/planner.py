"""EXPLAIN: plan trees after PostgreSQL's planner (filled in by planner work below)."""

from __future__ import annotations

import math

from . import types as T


def table_pages(t) -> int:
    """Heap pages a table occupies: 8 kB pages, 24-byte page header, 4-byte item pointers, 24-byte tuple headers,
    each tuple MAXALIGNed; widths from the data (a sample of rows)."""
    n = len(t.rows)
    if n == 0:
        return 0
    width = _data_width(t)
    tup = (24 + width + 7) // 8 * 8
    per_page = max(1, (8192 - 24) // (tup + 4))
    return max(1, math.ceil(n / per_page))


def _data_width(t):
    rows = t.rows
    step = max(1, len(rows) // 200)
    sample = rows[::step]
    total = 0.0
    for i, c in enumerate(t.columns):
        tw = _type_width(c.type)
        if tw is not None:
            total += tw * sum(1 for r in sample if r[i] is not None) / len(sample)
            continue
        w = 0
        for r in sample:
            v = r[i]
            if v is None:
                continue
            if c.type.name == "numeric":
                digits = len(format(abs(v), "f").replace(".", "").strip("0")) if v.is_finite() and v != 0 else 0
                w += 3 + 2 * ((digits + 3) // 4)
            else:
                b = len(T.output(v, c.type).encode("utf-8"))
                w += b + (1 if b < 127 else 4)
        total += w / len(sample)
    return int(round(total))


def _type_width(t):
    n = t.name
    return {"bool": 1, "int2": 2, "int4": 4, "int8": 8, "float4": 4, "float8": 8, "date": 4, "timestamp": 8,
            "timestamptz": 8, "time": 8, "interval": 16, "uuid": 16}.get(n)


def explain(session, db, st):
    from .explain import explain_stmt
    return explain_stmt(session, db, st)

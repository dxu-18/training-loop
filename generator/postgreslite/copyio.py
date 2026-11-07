"""COPY ... TO STDOUT / FROM STDIN in text and CSV formats (psql's \\copy goes through these)."""

from __future__ import annotations

from . import types as T
from .analyze import Analyzer
from .catalog import Table, View
from .errors import PgError


def _opts(st):
    o = dict(st.options)
    fmt = str(o.get("format", "text")).lower()
    if fmt not in ("text", "csv", "binary"):
        raise PgError(f'COPY format "{fmt}" not recognized', "22023")
    if fmt == "binary":
        raise PgError("COPY BINARY is not supported by postgres-lite", "0A000")
    csv = fmt == "csv"
    delim = o.get("delimiter", "," if csv else "\t")
    if isinstance(delim, str) and len(delim) != 1:
        raise PgError("COPY delimiter must be a single one-byte character", "0A000")
    null = o.get("null", "" if csv else "\\N")
    header = o.get("header", False)
    if isinstance(header, str):
        header = header.lower() in ("true", "on", "1", "match")
    quote = o.get("quote", '"')
    escape = o.get("escape", quote)
    if not csv and ("quote" in o or "escape" in o):
        raise PgError("COPY quote available only in CSV mode", "0A000")
    if header and not csv and fmt == "text":
        pass
    return {"csv": csv, "delim": delim, "null": null, "header": bool(header), "quote": quote, "escape": escape,
            "force_quote": o.get("force_quote", [])}


def _source(session, db, st):
    an = Analyzer(session, db)
    if st.query is not None:
        q = an.query(st.query, None)
        return q, list(q.names[:q.ntargets]), list(q.types[:q.ntargets])
    rv = st.table
    rel = db.find_rel(rv.name, rv.schema, session)
    if rel is None:
        raise PgError(f'relation "{rv.name}" does not exist', "42P01")
    if isinstance(rel, View) and not rel.materialized:
        raise PgError(f'cannot copy from view "{rel.name}"', "42809",
                      hint="Try the COPY (SELECT ...) TO variant.")
    cols = list(rel.colnames)
    types = list(rel.coltypes)
    idx = list(range(len(cols)))
    if st.cols:
        idx = []
        for c in st.cols:
            if c not in cols:
                raise PgError(f'column "{c}" of relation "{rel.name}" does not exist', "42703")
            idx.append(cols.index(c))
    return rel, [cols[i] for i in idx], [types[i] for i in idx], idx


def copy_out(session, db, st):
    from .executor import Ctx, run_query
    o = _opts(st)
    src = _source(session, db, st)
    if st.query is not None:
        q, names, types = src
        c = Ctx(session, db)
        c.outer.append(None)
        rows = run_query(q, c)
    else:
        rel, names, types, idx = src
        rows = [tuple(r[i] for i in idx) for r in rel.rows]
    lines = []
    if o["header"]:
        lines.append(o["delim"].join(_field(n, o, False) if o["csv"] else _text_escape(n, o) for n in names))
    fq = o["force_quote"]
    for r in rows:
        parts = []
        for v, t, n in zip(r, types, names):
            if v is None:
                parts.append(o["null"])
                continue
            s = T.output(v, t)
            if o["csv"]:
                parts.append(_field(s, o, fq == "*" or (isinstance(fq, list) and n in fq)))
            else:
                parts.append(_text_escape(s, o))
        lines.append(o["delim"].join(parts))
    text = "".join(ln + "\n" for ln in lines)
    return text, len(rows)


def _field(s, o, force):
    q = o["quote"]
    need = force or s == "" or o["delim"] in s or q in s or "\n" in s or "\r" in s or s == o["null"] or \
        s.startswith(" ") or s.endswith(" ") and False
    if s == "\\.":
        need = True
    if need:
        return q + s.replace(q, o["escape"] + q) + q
    return s


def _text_escape(s, o):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == o["delim"]:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def copy_in_prepare(session, db, st):
    o = _opts(st)
    if st.query is not None:
        raise PgError("COPY FROM with a query is not allowed", "42601")
    rel, names, types, idx = _source(session, db, st)
    if not isinstance(rel, Table):
        raise PgError(f'cannot copy to {"view" if isinstance(rel, View) else "relation"} "{rel.name}"', "42809")
    return {"st": st, "opts": o, "table": rel.name, "schema": rel.schema, "names": names, "idx": idx}


def _csv_records(data, o):
    """Split CSV data into records of fields (quoted fields may hold newlines). None = NULL."""
    q, esc, d = o["quote"], o["escape"], o["delim"]
    i, n = 0, len(data)
    line = 1
    while i < n:
        fields = []
        start_line = line
        while True:
            if i < n and data[i] == q:
                i += 1
                buf = []
                while True:
                    if i >= n:
                        raise PgError("unterminated CSV quoted field", "22P04",
                                      context=f"COPY, line {start_line}")
                    ch = data[i]
                    if ch == esc and esc != q and i + 1 < n and data[i + 1] in (q, esc):
                        buf.append(data[i + 1])
                        i += 2
                        continue
                    if ch == q:
                        if esc == q and i + 1 < n and data[i + 1] == q:
                            buf.append(q)
                            i += 2
                            continue
                        i += 1
                        break
                    if ch == "\n":
                        line += 1
                    buf.append(ch)
                    i += 1
                # text after the closing quote up to the delimiter belongs to the field
                while i < n and data[i] not in (d, "\n", "\r"):
                    buf.append(data[i])
                    i += 1
                fields.append(("".join(buf), True))
            else:
                j = i
                while j < n and data[j] not in (d, "\n", "\r"):
                    j += 1
                fields.append((data[i:j], False))
                i = j
            if i < n and data[i] == d:
                i += 1
                continue
            break
        if i < n and data[i] == "\r":
            i += 1
        if i < n and data[i] == "\n":
            i += 1
        line += 1
        yield start_line, fields


def _text_records(data, o):
    d = o["delim"]
    for ln, raw in enumerate(data.split("\n"), 1):
        if raw == "" and ln == data.count("\n") + 1:
            break
        if raw.endswith("\r"):
            raw = raw[:-1]
        fields = []
        buf = []
        i = 0
        null_marks = []
        while i <= len(raw):
            if i == len(raw) or raw[i] == d:
                fields.append(("".join(buf), False))
                buf = []
                i += 1
                continue
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                nx = raw[i + 1]
                if nx == "N" and not buf and (i + 2 == len(raw) or raw[i + 2] == d):
                    buf = ["\\N"]
                    i += 2
                    continue
                buf.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "b": "\b", "f": "\f", "v": "\v"}.get(nx, nx))
                i += 2
                continue
            buf.append(ch)
            i += 1
        yield ln, fields


def copy_in_finish(session, db, prep, data):
    from .dml import check_row, check_table, _bind_defaults
    from .executor import Ctx
    o = prep["opts"]
    table = db.find_rel(prep["table"], prep["schema"], session)
    names, idx = prep["names"], prep["idx"]
    c = Ctx(session, db)
    an = Analyzer(session, db)
    defaults = _bind_defaults(an, table)
    if data.endswith("\\.\n"):
        data = data[:-3]
    recs = _csv_records(data, o) if o["csv"] else _text_records(data, o)
    rows = list(table.rows)
    new = []
    first = True
    for ln, fields in recs:
        if first and o["header"]:
            first = False
            continue
        first = False
        if len(fields) == 1 and fields[0] == ("", False) and len(idx) > 1:
            continue
        if fields and fields[0][0] == "\\." and len(fields) == 1:
            break
        ctx = f"COPY {table.name}, line {ln}"
        if len(fields) < len(idx):
            raise PgError(f'missing data for column "{names[len(fields)]}"', "22P04", context=ctx)
        if len(fields) > len(idx):
            raise PgError("extra data after last expected column", "22P04", context=ctx)
        row = [None] * len(table.columns)
        given = set()
        for (val, quoted), ci, name in zip(fields, idx, names):
            given.add(ci)
            is_null = (val == o["null"] and not quoted) if o["csv"] else (val == "\\N" or val == o["null"])
            if is_null:
                row[ci] = None
                continue
            col = table.columns[ci]
            try:
                row[ci] = T.input_value(val, col.type.base, col.type.mod, now=c.txn_now)
            except PgError as e:
                e.context = f'{ctx}, column {name}: "{val}"'
                raise
        for i in range(len(table.columns)):
            if i not in given:
                f = defaults[i]
                row[i] = f(None, c) if f is not None else None
        r = tuple(row)
        try:
            check_row(table, r, c)
        except PgError as e:
            e.context = ctx
            raise
        new.append(r)
    rows += new
    check_table(c, table, rows, changed=new, removed=[])
    table.rows = rows
    table.n_mod += 1
    table.n_changes = getattr(table, "n_changes", 0) + 1
    return len(new)

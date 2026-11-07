"""psql's describe commands (describe.c): \\d [pattern], \\d+, \\dt, \\di, \\dv, \\dm, \\ds, \\dn, \\dT, \\du, \\l, \\df."""

from __future__ import annotations

import fnmatch
import re

from . import ast as A
from . import types as T
from .catalog import Table, View, Index, Sequence, EnumType
from .psql import print_table, PrintOpts


def _pattern(p):
    """psql patterns: * ? and "quoted" parts; unquoted letters fold to lower case. -> (schema re, name re)."""
    def conv(s):
        out = []
        q = False
        for ch in s:
            if ch == '"':
                q = not q
                continue
            if not q and ch == "*":
                out.append(".*")
            elif not q and ch == "?":
                out.append(".")
            else:
                out.append(re.escape(ch if q else ch.lower()))
        return re.compile("^(" + "".join(out) + ")$")
    if p is None:
        return None, None
    parts = re.split(r'\.(?=(?:[^"]*"[^"]*")*[^"]*$)', p)
    if len(parts) >= 2:
        return conv(parts[-2]), conv(parts[-1])
    return None, conv(parts[0])


def _visible(db, session, rel):
    """pg_table_is_visible: the first relation of that name on the search path."""
    return db.find_rel(rel.name, None, session) is rel


def _rels(psql, kinds, pattern):
    db = psql.session.db()
    srx, nrx = _pattern(pattern)
    out = []
    for s in db.schemas.values():
        if srx is not None and not srx.match(s.name):
            continue
        for r in s.rels.values():
            k = _kind(r)
            if k not in kinds:
                continue
            if nrx is not None and not nrx.match(r.name):
                continue
            if srx is None and not _visible(db, psql.session, r):
                continue
            out.append(r)
    out.sort(key=lambda r: (r.schema, r.name))
    return out


def _kind(r):
    if isinstance(r, Table):
        return "t"
    if isinstance(r, View):
        return "m" if r.materialized else "v"
    if isinstance(r, Index):
        return "i"
    if isinstance(r, Sequence):
        return "s"
    return "?"


_KIND_NAME = {"t": "table", "v": "view", "m": "materialized view", "i": "index", "s": "sequence"}


def _owner(psql):
    return psql.session.cluster.databases[psql.session.dbname].owner if False else psql.session.user


def size_pretty(n: int) -> str:
    if n < 10 * 1024:
        return f"{n} bytes"
    for unit in ("kB", "MB", "GB", "TB"):
        n2 = (n // 512 + 1) // 2
        if n2 < 10 * 1024 or unit == "TB":
            return f"{n2} {unit}"
        n = n2 * 1024
    return f"{n} bytes"


def rel_size(r):
    from .planner import table_pages
    if isinstance(r, (Table, View)):
        return table_pages(r) * 8192
    if isinstance(r, Index):
        return 16384
    if isinstance(r, Sequence):
        return 8192
    return 0


def describe(psql, cmd, args):
    """-> (stdout text, stderr text)."""
    opts = psql.opts
    plus = cmd.endswith("+")
    base = cmd.rstrip("+")
    show_system = "S" in base[1:]
    base = base.replace("S", "") if base not in ("dS",) else "d"
    pattern = args[0] if args else None
    if base in ("l", "list"):
        return list_databases(psql, plus), None
    if base == "d":
        if pattern is None:
            rels = _rels(psql, "tvms", None)
            if not rels:
                return None, "Did not find any relations.\n"
            return list_rels(psql, rels, plus, with_table=False), None
        return describe_rel(psql, pattern, plus)
    if base in ("dt", "dv", "dm", "ds", "di", "dti", "dit", "dtv", "dvt", "dts", "dst"):
        kinds = "".join(k for k in base[1:] if k in "tvmsi") or "t"
        rels = _rels(psql, kinds, pattern)
        if not rels and pattern is None:
            return None, "Did not find any relations.\n"
        return list_rels(psql, rels, plus, with_table="i" in kinds), None
    if base == "dn":
        db = psql.session.db()
        srx = _pattern(pattern)[1] if pattern else None
        rows = [(s.name, s.owner) for s in sorted(db.schemas.values(), key=lambda s: s.name)
                if not s.name.startswith("pg_") and (srx is None or srx.match(s.name))]
        names = ["Name", "Owner"]
        types = [T.NAME, T.NAME]
        if plus:
            rows = [r + ("", "") for r in rows]
            names += ["Access privileges", "Description"]
            types += [T.TEXT, T.TEXT]
        return print_table(names, types, rows, opts, title="List of schemas"), None
    if base == "dT":
        db = psql.session.db()
        rows = []
        for s in db.schemas.values():
            for t in s.types.values():
                rows.append((s.name, t.name, ""))
        rows.sort()
        return print_table(["Schema", "Name", "Description"], [T.NAME, T.TEXT, T.TEXT], rows, opts,
                           title="List of data types"), None
    if base in ("du", "dg"):
        rows = []
        for r in sorted(psql.session.cluster.roles.values(), key=lambda r: r.name):
            attrs = []
            if r.superuser:
                attrs = ["Superuser", "Create role", "Create DB", "Replication", "Bypass RLS"]
            elif r.createdb:
                attrs = ["Create DB"]
            rows.append((r.name, ", ".join(attrs)))
        return print_table(["Role name", "Attributes"], [T.NAME, T.TEXT], rows, opts, title="List of roles",
                           count_footer=False), None
    if base == "df":
        return print_table(["Schema", "Name", "Result data type", "Argument data types", "Type"], [T.TEXT] * 5, [],
                           opts, title="List of functions"), None
    if base == "dx":
        return describe_extensions(psql), None
    return None, f"invalid command \\{cmd}\nTry \\? for help.\n"


def describe_extensions(psql):
    db = psql.session.db()
    rows = [(e, "1.0" if e == "plpgsql" else "1.3", "pg_catalog" if e == "plpgsql" else "public",
             "PL/pgSQL procedural language" if e == "plpgsql" else "") for e in sorted(db.extensions)]
    return print_table(["Name", "Version", "Schema", "Description"], [T.NAME] * 4, rows, psql.opts,
                       title="List of installed extensions")


def list_rels(psql, rels, plus, with_table):
    owner = _owner(psql)
    rows = []
    for r in rels:
        k = _kind(r)
        row = [r.schema, r.name, _KIND_NAME[k], owner]
        if with_table:
            row.append(r.table if isinstance(r, Index) else "")
        if plus:
            pers = {"p": "permanent", "t": "temporary", "u": "unlogged"}.get(getattr(r, "persistence", "p"),
                                                                              "permanent")
            row.append(pers)
            if k in ("t", "m", "i"):
                row.append("heap" if k != "i" else r.method)
            else:
                row.append("")
            row.append(size_pretty(rel_size(r)))
            row.append(getattr(r, "comment", None) or "")
        rows.append(tuple(row))
    names = ["Schema", "Name", "Type", "Owner"] + (["Table"] if with_table else [])
    if plus:
        names += ["Persistence", "Access method", "Size", "Description"]
    return print_table(names, [T.NAME] * len(names), rows, psql.opts, title="List of relations")


def list_databases(psql, plus):
    cl = psql.session.cluster
    rows = []
    for name in sorted(cl.databases):
        db = cl.databases[name]
        priv = "=c/postgres\npostgres=CTc/postgres" if name.startswith("template") else ""
        rows.append((name, db.owner, "UTF8", "libc", "C.UTF-8", "C.UTF-8", "", "", priv))
    names = ["Name", "Owner", "Encoding", "Locale Provider", "Collate", "Ctype", "ICU Locale", "ICU Rules",
             "Access privileges"]
    return print_table(names, [T.NAME] * len(names), rows, psql.opts, title="List of databases")


# ---------------------------------------------------------------- \d name

def describe_rel(psql, pattern, plus):
    db = psql.session.db()
    srx, nrx = _pattern(pattern)
    hits = []
    for s in db.schemas.values():
        if srx is not None and not srx.match(s.name):
            continue
        for r in s.rels.values():
            if nrx.match(r.name) and (srx is not None or _visible(db, psql.session, r)):
                hits.append(r)
    hits.sort(key=lambda r: (r.schema, r.name))
    if not hits:
        return None, f'Did not find any relation named "{pattern}".\n'
    out = []
    for r in hits:
        if isinstance(r, Table):
            out.append(describe_table(psql, db, r, plus))
        elif isinstance(r, View):
            out.append(describe_view(psql, db, r, plus))
        elif isinstance(r, Index):
            out.append(describe_index(psql, db, r))
        elif isinstance(r, Sequence):
            out.append(describe_sequence(psql, db, r))
    return "".join(out), None


def _col_rows(t, plus, view=False):
    rows = []
    for c in t.columns:
        if c.identity:
            dflt = "generated always as identity" if c.identity == "a" else "generated by default as identity"
        elif c.generated is not None:
            dflt = f"generated always as ({c.generated_sql}) stored"
        else:
            dflt = c.default_sql or ""
        row = [c.name, c.type.format(), "", "not null" if c.not_null else "", dflt]
        if plus:
            row.append(_storage(c.type))
            if not view:
                row += ["", ""]
            row.append(c.comment or "")
        rows.append(tuple(row))
    return rows


def _storage(t):
    if t.is_array or t.name in ("text", "varchar", "bpchar", "json", "jsonb", "bytea", "numeric"):
        return "extended" if t.name != "numeric" else "main"
    return "plain"


def describe_table(psql, db, t, plus):
    names = ["Column", "Type", "Collation", "Nullable", "Default"]
    if plus:
        names += ["Storage", "Compression", "Stats target", "Description"]
    kind = "Unlogged table" if t.persistence == "u" else "Table"
    title = f'{kind} "{t.schema}.{t.name}"'
    footers = table_footers(db, t, plus)
    return print_table(names, [T.TEXT] * len(names), _col_rows(t, plus), psql.opts, title=title, footers=footers,
                       count_footer=False)


def index_def(db, ix):
    t = db.find_rel(ix.table, ix.schema, None)
    cols = ", ".join(_elem_sql(el) for el in ix.elems)
    s = f"CREATE {'UNIQUE ' if ix.unique else ''}INDEX {ix.name} ON {ix.schema}.{ix.table} USING {ix.method} ({cols})"
    if ix.pred_sql:
        s += f" WHERE ({ix.pred_sql})"
    return s


def _elem_sql(el):
    s = el.col if el.col is not None else (el.expr_sql if el.expr_sql and "(" in el.expr_sql and
                                            not el.expr_sql.startswith("(") else f"({el.expr_sql})")
    if el.col is None and el.expr_sql and re.match(r"^\w+\(.*\)$", el.expr_sql):
        s = el.expr_sql
    if el.desc:
        s += " DESC"
        if not el.nulls_first:
            s += " NULLS LAST"
    elif el.nulls_first:
        s += " NULLS FIRST"
    return s


def table_footers(db, t, plus):
    f = []
    idxs = db.indexes_of(t)
    idxs.sort(key=lambda i: (not i.primary, i.name))
    if idxs:
        f.append("Indexes:")
        for ix in idxs:
            cols = ", ".join(_elem_sql(el) for el in ix.elems)
            if ix.primary:
                kind = "PRIMARY KEY, "
            elif ix.constraint is not None:
                kind = "UNIQUE CONSTRAINT, "
            elif ix.unique:
                kind = "UNIQUE, "
            else:
                kind = ""
            line = f'    "{ix.name}" {kind}{ix.method} ({cols})'
            if ix.nulls_not_distinct:
                line = line.replace(f"{ix.method} (", f"{ix.method} (", 1) + " NULLS NOT DISTINCT"
            if ix.pred_sql:
                line += f" WHERE {ix.pred_sql}"
            f.append(line)
    checks = sorted([k for k in t.constraints if k.kind == "c"], key=lambda k: k.name)
    if checks:
        f.append("Check constraints:")
        for k in checks:
            f.append(f'    "{k.name}" CHECK ({k.expr_sql}){" NOT VALID" if not k.validated else ""}')
    fks = sorted([k for k in t.constraints if k.kind == "f"], key=lambda k: k.name)
    if fks:
        f.append("Foreign-key constraints:")
        for k in fks:
            f.append(f'    "{k.name}" {fk_def(k, t)}')
    refs = []
    for other in db.all_rels():
        if isinstance(other, Table):
            for k in other.constraints:
                if k.kind == "f" and k.ref_table == t.name and k.ref_schema == t.schema:
                    refs.append((other.name, k))
    refs.sort(key=lambda x: (x[0], x[1].name))
    if refs:
        f.append("Referenced by:")
        for name, k in refs:
            f.append(f'    TABLE "{name}" CONSTRAINT "{k.name}" {fk_def(k, None)}')
    if plus:
        f.append("Access method: heap")
    return f


def fk_def(k, t):
    ref = k.ref_table if k.ref_schema in (None, "public") else f"{k.ref_schema}.{k.ref_table}"
    s = f"FOREIGN KEY ({', '.join(k.cols)}) REFERENCES {ref}({', '.join(k.ref_cols)})"
    acts = {"c": "CASCADE", "r": "RESTRICT", "n": "SET NULL", "d": "SET DEFAULT"}
    if k.on_update in acts:
        s += f" ON UPDATE {acts[k.on_update]}"
    if k.on_delete in acts:
        s += f" ON DELETE {acts[k.on_delete]}"
    if k.deferrable:
        s += " DEFERRABLE"
    return s


def describe_view(psql, db, v, plus):
    names = ["Column", "Type", "Collation", "Nullable", "Default"]
    if plus:
        names += ["Storage"] + (["Compression", "Stats target"] if v.materialized else []) + ["Description"]
    title = f'{"Materialized view" if v.materialized else "View"} "{v.schema}.{v.name}"'
    footers = []
    if v.materialized:
        idxs = db.indexes_of(v)
        if idxs:
            footers.append("Indexes:")
            for ix in sorted(idxs, key=lambda i: i.name):
                cols = ", ".join(_elem_sql(el) for el in ix.elems)
                footers.append(f'    "{ix.name}" {"UNIQUE, " if ix.unique else ""}{ix.method} ({cols})')
    if plus:
        footers.append("View definition:")
        footers += view_definition(v).split("\n")
        if v.materialized:
            footers.append("Access method: heap")
    rows = _col_rows(v, plus, view=not v.materialized)
    return print_table(names, [T.TEXT] * len(names), rows, psql.opts, title=title, footers=footers,
                       count_footer=False)


def describe_index(psql, db, ix):
    t = db.find_rel(ix.table, ix.schema, None)
    rows = []
    for el in ix.elems:
        if el.col is not None:
            c = t.col(el.col) if t is not None else None
            rows.append((el.col, c.type.format() if c else "", "yes", el.col))
        else:
            rows.append((el.expr_sql.split("(")[0] if el.expr_sql else "expr", "text", "yes", el.expr_sql))
    desc = []
    if ix.primary:
        desc.append("primary key")
    elif ix.unique:
        desc.append("unique")
    desc.append(ix.method)
    tail = ", ".join(desc) + f', for table "{ix.schema}.{ix.table}"'
    if ix.pred_sql:
        tail += f", predicate ({ix.pred_sql})"
    return print_table(["Column", "Type", "Key?", "Definition"], [T.TEXT] * 4, rows, psql.opts,
                       title=f'Index "{ix.schema}.{ix.name}"', footers=[tail], count_footer=False)


def describe_sequence(psql, db, s):
    rows = [(s.type.format(), s.start, s.minv, s.maxv, s.increment, "yes" if s.cycle else "no", s.cache)]
    types = [T.TEXT, T.INT8, T.INT8, T.INT8, T.INT8, T.TEXT, T.INT8]
    footers = []
    if s.owned_by:
        t = db.find_rel(s.owned_by[0], s.schema, None)
        col = t.col(s.owned_by[1]) if isinstance(t, Table) else None
        if col is not None and col.identity:
            footers.append(f"Sequence for identity column: {s.schema}.{s.owned_by[0]}.{s.owned_by[1]}")
        else:
            footers.append(f"Owned by: {s.schema}.{s.owned_by[0]}.{s.owned_by[1]}")
    return print_table(["Type", "Start", "Minimum", "Maximum", "Increment", "Cycles?", "Cache"], types, rows,
                       psql.opts, title=f'Sequence "{s.schema}.{s.name}"', footers=footers, count_footer=False)


def show_view(psql, name):
    db = psql.session.db()
    v = db.find_rel(name, None, psql.session)
    if not isinstance(v, View):
        return f'"{name}" is not a view\n'
    kind = "MATERIALIZED VIEW" if v.materialized else "VIEW"
    return f"CREATE OR REPLACE {kind} {v.schema}.{v.name} AS\n{view_definition(v)}\n"


# ---------------------------------------------------------------- view definitions (pg_get_viewdef, pretty)

def view_definition(v) -> str:
    q = v.query
    try:
        return _fmt_query(q, v.sql) + ";"
    except Exception:  # noqa: BLE001
        return v.sql.strip()


def _src(node, sql):
    return None


def _fmt_query(q, sql):
    """A readable approximation of pg_get_viewdef's pretty output, from the parse tree."""
    from .unparse import unparse_select
    return unparse_select(q)

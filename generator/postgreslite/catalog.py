"""The catalog: databases, schemas, tables (with their rows), constraints, indexes, views, sequences, enum types.

Rows are tuples, so a snapshot of a database (for a transaction, a savepoint or one statement) copies only the
catalog objects and the row lists, not the rows. Constraint and index names follow PostgreSQL's choices
(orders_pkey, users_email_key, orders_user_id_fkey, orders_amount_check, idx dedup suffixes ...).
"""

from __future__ import annotations

import copy

from . import types as T
from .errors import PgError

NAMEDATALEN = 63


class Column:
    __slots__ = ("name", "type", "not_null", "default", "default_sql", "identity", "generated", "generated_sql",
                 "comment", "seq")

    def __init__(self, name, typ, not_null=False, default=None, default_sql=None, identity=None, generated=None,
                 generated_sql=None):
        self.name, self.type, self.not_null = name, typ, not_null
        self.default, self.default_sql = default, default_sql       # parse tree, display text
        self.identity = identity        # None / "a" (always) / "d" (by default)
        self.generated, self.generated_sql = generated, generated_sql
        self.comment = None
        self.seq = None                 # owned sequence name (serial / identity)

    def copy(self):
        c = Column(self.name, self.type, self.not_null, self.default, self.default_sql, self.identity,
                   self.generated, self.generated_sql)
        c.comment = self.comment
        c.seq = self.seq
        return c


class Constraint:
    __slots__ = ("name", "kind", "cols", "expr", "expr_sql", "ref_schema", "ref_table", "ref_cols", "on_delete",
                 "on_update", "deferrable", "index", "nulls_not_distinct", "validated", "no_inherit")

    def __init__(self, name, kind, cols=(), expr=None, expr_sql=None):
        self.name, self.kind, self.cols = name, kind, list(cols)   # kind: p u c f
        self.expr, self.expr_sql = expr, expr_sql
        self.ref_schema = self.ref_table = None
        self.ref_cols = []
        self.on_delete = self.on_update = "a"
        self.deferrable = False
        self.index = None
        self.nulls_not_distinct = False
        self.validated = True
        self.no_inherit = False

    def copy(self):
        c = copy.copy(self)
        c.cols = list(self.cols)
        c.ref_cols = list(self.ref_cols)
        return c


class IndexElem:
    __slots__ = ("col", "expr", "expr_sql", "desc", "nulls_first", "opclass")

    def __init__(self, col=None, expr=None, expr_sql=None, desc=False, nulls_first=None, opclass=None):
        self.col, self.expr, self.expr_sql = col, expr, expr_sql
        self.desc = desc
        self.nulls_first = desc if nulls_first is None else nulls_first
        self.opclass = opclass


class Index:
    kind = "i"

    def __init__(self, name, schema, table, elems, unique=False, primary=False, method="btree", pred=None,
                 pred_sql=None, constraint=None, include=()):
        self.name, self.schema, self.table = name, schema, table
        self.elems = elems
        self.unique, self.primary, self.method = unique, primary, method
        self.pred, self.pred_sql = pred, pred_sql
        self.constraint = constraint
        self.include = list(include)
        self.oid = 0
        self.nulls_not_distinct = False
        self.comment = None
        self.valid = True

    def copy(self):
        i = copy.copy(self)
        i.elems = list(self.elems)
        return i

    def cols(self):
        return [e.col for e in self.elems]


class Table:
    kind = "r"

    def __init__(self, name, schema, columns, persistence="p"):
        self.name, self.schema = name, schema
        self.columns: list[Column] = columns
        self.rows: list[tuple] = []
        self.constraints: list[Constraint] = []
        self.persistence = persistence        # p permanent, t temporary, u unlogged
        self.oid = 0
        self.comment = None
        self.stats = None                     # set by ANALYZE: {"reltuples", "relpages", "cols": {...}}
        self.n_mod = 0                        # rows changed since ANALYZE
        self.n_changes = 0                    # statements that changed the rows, ever
        self.vacuumed = None                  # n_changes at the last VACUUM (all pages all-visible until then)
        self.on_commit = None

    def copy(self):
        t = copy.copy(self)
        t.columns = [c.copy() for c in self.columns]
        t.rows = list(self.rows)
        t.constraints = [c.copy() for c in self.constraints]
        return t

    def col_index(self, name):
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        return -1

    def col(self, name):
        i = self.col_index(name)
        return self.columns[i] if i >= 0 else None

    @property
    def colnames(self):
        return [c.name for c in self.columns]

    @property
    def coltypes(self):
        return [c.type for c in self.columns]

    def pkey(self):
        for c in self.constraints:
            if c.kind == "p":
                return c
        return None


class View:
    def __init__(self, name, schema, query, sql, colnames, coltypes, materialized=False):
        self.name, self.schema = name, schema
        self.query = query              # parse tree
        self.sql = sql                  # the definition as \d+ shows it
        self.colnames, self.coltypes = colnames, coltypes
        self.materialized = materialized
        self.kind = "m" if materialized else "v"
        self.rows = []
        self.populated = True
        self.oid = 0
        self.comment = None
        self.persistence = "p"
        self.stats = None
        self.n_mod = 0
        self.check_option = None
        self.columns = [Column(n, t) for n, t in zip(colnames, coltypes)]

    def copy(self):
        v = copy.copy(self)
        v.rows = list(self.rows)
        v.columns = [c.copy() for c in self.columns]
        return v


class Sequence:
    kind = "S"

    def __init__(self, name, schema, start=1, increment=1, minv=None, maxv=None, cycle=False, typ=None, cache=1):
        self.name, self.schema = name, schema
        self.type = typ or T.INT8
        lo, hi, _ = T.INT_LIMITS[self.type.name]
        self.increment = increment
        self.minv = minv if minv is not None else (1 if increment > 0 else lo)
        self.maxv = maxv if maxv is not None else (hi if increment > 0 else -1)
        self.start = start if start is not None else (self.minv if increment > 0 else self.maxv)
        self.cycle = cycle
        self.cache = cache
        self.last = self.start
        self.is_called = False
        self.owned_by = None
        self.oid = 0
        self.comment = None
        self.persistence = "p"

    def copy(self):
        return copy.copy(self)

    def nextval(self):
        if not self.is_called:
            self.is_called = True
            return self.last
        v = self.last + self.increment
        if v > self.maxv or v < self.minv:
            if not self.cycle:
                lim = "maximum" if v > self.maxv else "minimum"
                raise PgError(f'nextval: reached {lim} value of sequence "{self.name}" '
                              f'({self.maxv if v > self.maxv else self.minv})', "2200H")
            v = self.minv if self.increment > 0 else self.maxv
        self.last = v
        return v


class EnumType:
    kind = "E"

    def __init__(self, name, schema, labels):
        self.name, self.schema, self.labels = name, schema, list(labels)
        self.type = T.Type(name, 0, "E", False, name, labels=self.labels)

    def copy(self):
        e = copy.copy(self)
        e.labels = list(self.labels)
        e.type = T.Type(self.name, 0, "E", False, self.name, labels=e.labels)
        return e


class Schema:
    def __init__(self, name, owner="postgres"):
        self.name = name
        self.owner = owner
        self.rels: dict = {}             # tables, views, matviews, indexes, sequences (one namespace)
        self.types: dict = {}
        self.comment = None

    def copy(self):
        s = Schema(self.name, self.owner)
        # sequences are shared: nextval() isn't undone by ROLLBACK (as in PostgreSQL)
        s.rels = {k: (v if isinstance(v, Sequence) else v.copy()) for k, v in self.rels.items()}
        s.types = {k: v.copy() for k, v in self.types.items()}
        s.comment = self.comment
        return s


class Database:
    def __init__(self, name, owner="postgres"):
        self.name = name
        self.owner = owner
        self.schemas: dict[str, Schema] = {"public": Schema("public", "pg_database_owner")}
        self.oid_next = 16384
        self.extensions = {"plpgsql"}
        self.comment = None

    def copy(self):
        d = copy.copy(self)
        d.schemas = {k: v.copy() for k, v in self.schemas.items()}
        d.extensions = set(self.extensions)
        return d

    def new_oid(self):
        self.oid_next += 1
        return self.oid_next

    # ---- lookups
    def search(self, session):
        path = []
        temp = session.temp_schema if session is not None else None
        if temp and temp in self.schemas:
            path.append(temp)
        for s in (session.search_path if session is not None else ["public"]):
            s = s.strip()
            if s == '"$user"' or s == "$user":
                s = session.user if session is not None else "postgres"
            if s in self.schemas and s not in path:
                path.append(s)
        return path

    def find_rel(self, name, schema=None, session=None):
        if schema is not None:
            if schema in ("pg_catalog", "information_schema"):
                return None
            s = self.schemas.get(schema)
            if s is None:
                return None
            return s.rels.get(name)
        for sn in self.search(session):
            r = self.schemas[sn].rels.get(name)
            if r is not None:
                return r
        return None

    def find_type(self, name, schema=None, session=None):
        if schema is not None:
            s = self.schemas.get(schema)
            return s.types.get(name) if s else None
        for sn in self.search(session):
            t = self.schemas[sn].types.get(name)
            if t is not None:
                return t
        return None

    def creation_schema(self, schema, session, temp=False):
        if temp:
            return session.ensure_temp(self)
        if schema is not None:
            if schema not in self.schemas:
                raise PgError(f'schema "{schema}" does not exist', "3F000")
            return schema
        path = [s for s in self.search(session) if s != (session.temp_schema if session else None)]
        if not path:
            raise PgError("no schema has been selected to create in", "3F000")
        return path[0]

    def all_rels(self):
        for s in self.schemas.values():
            yield from s.rels.values()

    def indexes_of(self, table):
        out = [r for r in self.schemas[table.schema].rels.values()
               if isinstance(r, Index) and r.table == table.name]
        out.sort(key=lambda i: (not i.primary, i.oid))
        return out


# ---------------------------------------------------------------- names

def truncate_parts(parts: list[str], suffix: str) -> str:
    """makeObjectName: shorten the longest part until name1_name2_suffix fits in 63 bytes."""
    parts = [p for p in parts if p]
    while True:
        name = "_".join(parts + ([suffix] if suffix else []))
        if len(name.encode()) <= NAMEDATALEN:
            return name
        longest = max(range(len(parts)), key=lambda i: len(parts[i]))
        parts[longest] = parts[longest][:-1]


def choose_name(db: Database, schema: str, parts: list[str], suffix: str, taken=()) -> str:
    """ChooseRelationName / ChooseConstraintName: add 1, 2 ... until the name is free."""
    base = truncate_parts(list(parts), suffix)
    s = db.schemas[schema]
    names = set(s.rels) | set(taken)
    for r in s.rels.values():
        if isinstance(r, Table):
            names |= {c.name for c in r.constraints}
    if base not in names:
        return base
    n = 1
    while True:
        cand = truncate_parts(list(parts), suffix + str(n))
        if cand not in names:
            return cand
        n += 1


def rel_kind_name(r) -> str:
    if isinstance(r, Table):
        return "table"
    if isinstance(r, View):
        return "materialized view" if r.materialized else "view"
    if isinstance(r, Index):
        return "index"
    if isinstance(r, Sequence):
        return "sequence"
    return "relation"

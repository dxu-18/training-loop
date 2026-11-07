"""DDL: tables (columns, defaults, serial / identity, constraints), ALTER TABLE, DROP (with dependencies and
CASCADE), TRUNCATE, indexes, views and materialized views, schemas, sequences, enum types, COMMENT, ANALYZE.
Errors and NOTICEs use PostgreSQL's wording."""

from __future__ import annotations

from . import ast as A
from . import expr as E
from . import types as T
from .analyze import Analyzer, Scope, RTE, NSItem, coerce_to
from .catalog import (Column, Constraint, Index, IndexElem, Table, View, Sequence, EnumType, Schema, choose_name,
                      rel_kind_name, truncate_parts)
from .errors import PgError, Unsupported

SERIALS = {"serial": "int4", "serial4": "int4", "bigserial": "int8", "serial8": "int8", "smallserial": "int2",
           "serial2": "int2"}


def _bind_default(an, col, node):
    scope = Scope(None, an.session, an.db)
    scope.kind = "default"
    try:
        e = an.expr(node, scope)
    except PgError as err:
        if err.code == "42703" or err.code == "42P01":
            raise PgError("cannot use column reference in DEFAULT expression", "0A000",
                          pos=getattr(node, "pos", -1) + 1) from None
        raise
    t = col.type
    err = PgError(f'column "{col.name}" is of type {t.format()} but default expression is of type '
                  f'{e.type.format()}', "42804", hint="You will need to rewrite or cast the expression.")
    try:
        e2 = coerce_to(e, t, "a", what=err)
    except PgError as x:
        if x.code == "42846":
            raise err from None
        raise
    return e, e2


def default_sql(e) -> str:
    """How \\d shows a default: 'paid'::text, now(), nextval('t_id_seq'::regclass), 0, true ..."""
    return _dp(e)


def _dp(e):
    if isinstance(e, E.Const):
        return E.deparse_const(e.value, e.type) if e.type.name != "unknown" else \
            E.deparse_const(e.value, T.TEXT)
    if isinstance(e, E.Func):
        if e.name == "nextval" and e.args and isinstance(e.args[0], E.Const):
            return f"nextval('{e.args[0].value}'::regclass)"
        return f"{e.display or e.name}({', '.join(_dp(a) for a in e.args)})"
    if isinstance(e, E.SQLValue):
        return {"current_timestamp": "CURRENT_TIMESTAMP", "current_date": "CURRENT_DATE",
                "localtimestamp": "LOCALTIMESTAMP", "current_user": "CURRENT_USER"}.get(e.name, e.name.upper())
    if isinstance(e, E.Cast):
        inner = _dp(e.arg)
        if isinstance(e.arg, E.Const):
            return f"{inner}::{e.type.format()}" if not inner.endswith("::" + e.arg.type.format()) else \
                f"({inner})::{e.type.format()}"
        return f"({inner})::{e.type.format()}"
    if isinstance(e, E.Op) and len(e.args) == 2:
        return f"({_dp(e.args[0])} {e.op} {_dp(e.args[1])})"
    if isinstance(e, E.Op) and len(e.args) == 1:
        return f"({e.op}{_dp(e.args[0])})"
    return e.deparse(E.DeparseCtx())


class DDL:
    def __init__(self, server, session, db):
        self.server = server
        self.session = session
        self.db = db
        self.an = Analyzer(session, db)
        self.notices = []

    def notice(self, msg, detail=None):
        self.notices.append(("NOTICE", msg, detail, None))

    # ------------------------------------------------------------ CREATE TABLE
    def create_table(self, st: A.CreateTable):
        db = self.db
        rv = st.table
        schema = db.creation_schema(rv.schema, self.session, st.temp)
        if rv.name in db.schemas[schema].rels or (st.temp is False and rv.schema is None and
                                                  db.find_rel(rv.name, schema, None) is not None):
            if st.if_not_exists:
                self.notice(f'relation "{rv.name}" already exists, skipping')
                return None
            raise PgError(f'relation "{rv.name}" already exists', "42P07")
        if st.as_query is not None:
            return self.create_table_as(st, schema)
        table = Table(rv.name, schema, [], "t" if st.temp else ("u" if st.unlogged else "p"))
        table.oid = db.new_oid()
        pending_cons = []
        seqs = []
        for el in st.elements:
            if isinstance(el, tuple) and el[0] == "like":
                self.like(table, el[1], el[2])
                continue
            if isinstance(el, A.ColumnDef):
                self.add_column_def(table, el, pending_cons, seqs)
            else:
                pending_cons.append(el)
        names = [c.name for c in table.columns]
        for i, n in enumerate(names):
            if n in names[:i]:
                raise PgError(f'column "{n}" specified more than once', "42701")
        if len(table.columns) > 1600:
            raise PgError("tables can have at most 1600 columns", "54011")
        db.schemas[schema].rels[table.name] = table
        try:
            for s in seqs:
                db.schemas[schema].rels[s.name] = s
            for con in pending_cons:
                self.add_constraint(table, con, creating=True)
        except PgError:
            del db.schemas[schema].rels[table.name]
            for s in seqs:
                db.schemas[schema].rels.pop(s.name, None)
            raise
        table.on_commit = st.on_commit
        return table

    def like(self, table, rv, opts):
        src = self.db.find_rel(rv.name, rv.schema, self.session)
        if src is None:
            raise PgError(f'relation "{rv.name}" does not exist', "42P01", pos=rv.pos + 1)
        inc = {o[1] for o in opts if o[0] == "including"}
        if "all" in inc:
            inc |= {"defaults", "constraints", "indexes", "identity", "generated", "comments"}
        for c in src.columns:
            nc = Column(c.name, c.type, c.not_null)
            if "defaults" in inc and c.default is not None and c.identity is None:
                nc.default, nc.default_sql = c.default, c.default_sql
            table.columns.append(nc)
        if "constraints" in inc and isinstance(src, Table):
            for k in src.constraints:
                if k.kind == "c":
                    nk = k.copy()
                    nk.name = choose_name(self.db, table.schema, [table.name] + ([k.cols[0]] if k.cols else []),
                                          "check")
                    table.constraints.append(nk)

    def add_column_def(self, table, cd: A.ColumnDef, pending, seqs):
        an = self.an
        tn = cd.typ
        serial = None
        if len(tn.names) == 1 and tn.names[0] in SERIALS and not tn.array:
            serial = tn.names[0]
            typ = T.TYPES[SERIALS[serial]]
        else:
            typ = an.resolve_type(tn)
        col = Column(cd.name, typ)
        nullness = None
        for c in cd.constraints:
            k = c.kind
            if k == "notnull":
                if nullness is False:
                    raise PgError(f'conflicting NULL/NOT NULL declarations for column "{cd.name}" of table '
                                  f'"{table.name}"', "42601", pos=c.pos + 1)
                nullness = True
                col.not_null = True
            elif k == "null":
                if nullness is True:
                    raise PgError(f'conflicting NULL/NOT NULL declarations for column "{cd.name}" of table '
                                  f'"{table.name}"', "42601", pos=c.pos + 1)
                nullness = False
            elif k == "default":
                if serial or col.identity:
                    raise PgError(f'multiple default values specified for column "{cd.name}" of table '
                                  f'"{table.name}"', "42601", pos=c.pos + 1)
                if col.default is not None:
                    raise PgError(f'multiple default values specified for column "{cd.name}" of table '
                                  f'"{table.name}"', "42601", pos=c.pos + 1)
                raw, bound = _bind_default(an, col, c.expr)
                col.default = c.expr
                col.default_sql = default_sql(bound)
            elif k == "identity":
                if col.default is not None:
                    raise PgError(f'both default and identity specified for column "{cd.name}" of table '
                                  f'"{table.name}"', "42601", pos=c.pos + 1)
                if typ.name not in ("int2", "int4", "int8"):
                    raise PgError(f'identity column type must be smallint, integer, or bigint', "22023",
                                  pos=c.pos + 1)
                col.identity = "a" if c.identity == "always" else "d"
                col.not_null = True
                seq = self.owned_sequence(table, col, typ, c.seq_opts)
                seqs.append(seq)
            elif k == "generated":
                col.generated = c.expr
                col.generated_sql = c.expr_text
            elif k in ("primary", "unique", "check", "foreign"):
                con = A.Constraint(k, c.name, expr=c.expr, cols=[cd.name], ref_table=c.ref_table,
                                   ref_cols=c.ref_cols, on_delete=c.on_delete, on_update=c.on_update,
                                   pos=c.pos, expr_text=c.expr_text, deferrable=c.deferrable,
                                   nulls_not_distinct=c.nulls_not_distinct)
                con._column = cd.name
                pending.append(con)
            elif k in ("collate", "attr"):
                pass
        if serial:
            col.not_null = True
            seq = self.owned_sequence(table, col, typ, [])
            seqs.append(seq)
            col.default = A.FuncCall(["nextval"], [A.Const(f"{seq.name}", "str")])
            col.default_sql = f"nextval('{seq.name}'::regclass)"
        table.columns.append(col)

    def owned_sequence(self, table, col, typ, opts):
        name = choose_name(self.db, table.schema, [table.name, col.name], "seq")
        seq = Sequence(name, table.schema, typ=T.INT8 if typ.name == "int8" else typ)
        lo, hi, _ = T.INT_LIMITS[typ.name]
        seq.maxv = hi
        self.seq_opts(seq, opts)
        seq.owned_by = (table.name, col.name)
        seq.oid = self.db.new_oid()
        col.seq = name
        if col.identity:
            col.default = A.FuncCall(["nextval"], [A.Const(name, "str")])
        return seq

    def seq_opts(self, seq, opts):
        for k, v in opts:
            if k == "increment":
                if v == 0:
                    raise PgError("INCREMENT must not be zero", "22023")
                seq.increment = v
                if v < 0:
                    seq.maxv, seq.minv = -1, T.INT_LIMITS[seq.type.name][0]
                    seq.start = seq.last = -1
            elif k == "minvalue":
                seq.minv = v
            elif k == "maxvalue":
                seq.maxv = v
            elif k == "start":
                seq.start = v
                seq.last = v
            elif k == "restart":
                seq.last = seq.start if v is None else v
                seq.is_called = False
            elif k == "cycle":
                seq.cycle = bool(v)
            elif k == "no cycle":
                seq.cycle = False
            elif k == "cache":
                seq.cache = v
            elif k == "as":
                seq.type = self.an.resolve_type(v)
            elif k == "owned":
                seq.owned_by = None if v == ["none"] else (v[-2], v[-1])
        if seq.start < seq.minv:
            raise PgError(f"START value ({seq.start}) cannot be less than MINVALUE ({seq.minv})", "22023")
        if seq.start > seq.maxv:
            raise PgError(f"START value ({seq.start}) cannot be greater than MAXVALUE ({seq.maxv})", "22023")

    # ------------------------------------------------------------ constraints
    def add_constraint(self, table, con: A.Constraint, creating=False, not_valid=False):
        db = self.db
        k = con.kind
        col_level = getattr(con, "_column", None)
        names_taken = {c.name for c in table.constraints}
        if con.name is not None:
            if con.name in names_taken:
                raise PgError(f'constraint "{con.name}" for relation "{table.name}" already exists', "42710")
            if k in ("primary", "unique") and con.name in db.schemas[table.schema].rels:
                raise PgError(f'relation "{con.name}" already exists', "42P07")
        for cn in con.cols:
            if table.col_index(cn) < 0:
                where = f'column "{cn}" named in key does not exist' if k in ("primary", "unique") else \
                    f'column "{cn}" referenced in foreign key constraint does not exist'
                raise PgError(where, "42703", pos=con.pos + 1 if con.pos >= 0 else None)
        if k == "primary":
            if table.pkey() is not None:
                raise PgError(f'multiple primary keys for table "{table.name}" are not allowed', "42P16",
                              pos=con.pos + 1 if con.pos >= 0 else None)
            name = con.name or choose_name(db, table.schema, [table.name], "pkey")
            for cn in con.cols:
                col = table.col(cn)
                if not col.not_null:
                    for r in table.rows:
                        if r[table.col_index(cn)] is None:
                            raise PgError(f'column "{cn}" of relation "{table.name}" contains null values', "23502")
                col.not_null = True
            kc = Constraint(name, "p", con.cols)
            self._check_unique_existing(table, kc, name)
            kc.index = name
            table.constraints.append(kc)
            self._backing_index(table, kc, primary=True)
            return kc
        if k == "unique":
            name = con.name or choose_name(db, table.schema, [table.name] + list(con.cols), "key")
            kc = Constraint(name, "u", con.cols)
            kc.nulls_not_distinct = con.nulls_not_distinct
            self._check_unique_existing(table, kc, name)
            kc.index = name
            table.constraints.append(kc)
            self._backing_index(table, kc)
            return kc
        if k == "check":
            scope = Scope(None, self.session, db)
            rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
            scope.ns.append(NSItem(rte))
            scope.kind = "check"
            e = self.an.expr(con.expr, scope)
            if e.type.name == "unknown":
                e = coerce_to(e, T.BOOL)
            if e.type != T.BOOL:
                raise PgError(f"argument of CHECK must be type boolean, not type {e.type.format()}", "42804")
            used = []
            for v in e.walk():
                if isinstance(v, E.Var) and v.col not in used:
                    used.append(v.col)
            if con.name:
                name = con.name
            elif col_level is not None and len(used) <= 1:
                name = choose_name(db, table.schema, [table.name, used[0] if used else col_level], "check")
            else:
                name = choose_name(db, table.schema, [table.name] + ([used[0]] if len(used) == 1 else []), "check")
            kc = Constraint(name, "c", used, con.expr, deparse_check(e))
            kc.validated = not con.not_valid
            if not con.not_valid:
                f = e.compile()
                from .executor import Ctx
                c = Ctx(self.session, db)
                for r in table.rows:
                    if f(r, c) is False:
                        raise PgError(f'check constraint "{name}" of relation "{table.name}" is violated by some '
                                      'row', "23514")
            table.constraints.append(kc)
            return kc
        if k == "foreign":
            ref = con.ref_table
            parent = db.find_rel(ref.name, ref.schema, self.session)
            if parent is None or not isinstance(parent, Table):
                raise PgError(f'relation "{ref.name}" does not exist', "42P01")
            ref_cols = list(con.ref_cols)
            if not ref_cols:
                pk = parent.pkey()
                if pk is None:
                    raise PgError(f'there is no primary key for referenced table "{parent.name}"', "42830")
                ref_cols = list(pk.cols)
            for rc in ref_cols:
                if parent.col_index(rc) < 0:
                    raise PgError(f'column "{rc}" referenced in foreign key constraint does not exist', "42703")
            if len(ref_cols) != len(con.cols):
                raise PgError("number of referencing and referenced columns for foreign key disagree", "42830")
            if not any(c.kind in ("p", "u") and set(c.cols) == set(ref_cols) for c in parent.constraints) and \
                    not any(ix.unique and ix.pred is None and all(el.col for el in ix.elems) and
                            set(ix.cols()) == set(ref_cols) for ix in db.indexes_of(parent)):
                raise PgError(f'there is no unique constraint matching given keys for referenced table '
                              f'"{parent.name}"', "42830")
            name = con.name or choose_name(db, table.schema, [table.name] + list(con.cols), "fkey")
            for cn, rc in zip(con.cols, ref_cols):
                ct, pt = table.col(cn).type, parent.col(rc).type
                if not (T.can_coerce(ct, pt) or T.can_coerce(pt, ct)) or \
                        (ct.cat != pt.cat):
                    raise PgError(f'foreign key constraint "{name}" cannot be implemented', "42804",
                                  detail=f'Key columns "{cn}" and "{rc}" are of incompatible types: '
                                         f'{ct.format()} and {pt.format()}.')
            kc = Constraint(name, "f", con.cols)
            kc.ref_schema = parent.schema
            kc.ref_table = parent.name
            kc.ref_cols = ref_cols
            kc.on_delete = con.on_delete
            kc.on_update = con.on_update
            kc.deferrable = con.deferrable
            kc.validated = not con.not_valid
            if not con.not_valid and table.rows:
                pidx = [parent.col_index(n) for n in ref_cols]
                cidx = [table.col_index(n) for n in con.cols]
                keys = {tuple(T.hash_key(r[i]) for i in pidx) for r in (parent.rows if parent is not table
                                                                         else table.rows)}
                for r in table.rows:
                    vals = [r[i] for i in cidx]
                    if any(v is None for v in vals):
                        continue
                    if tuple(T.hash_key(v) for v in vals) not in keys:
                        shown = ", ".join(T.output(v, table.columns[i].type) for v, i in zip(vals, cidx))
                        raise PgError(f'insert or update on table "{table.name}" violates foreign key constraint '
                                      f'"{name}"', "23503",
                                      detail=f'Key ({", ".join(con.cols)})=({shown}) is not present in table '
                                             f'"{parent.name}".')
            table.constraints.append(kc)
            return kc
        raise Unsupported(f"{k} constraints")

    def _check_unique_existing(self, table, kc, name):
        idx = [table.col_index(n) for n in kc.cols]
        seen = set()
        for r in table.rows:
            vals = [r[i] for i in idx]
            if not kc.nulls_not_distinct and any(v is None for v in vals):
                continue
            key = tuple(T.hash_key(v) for v in vals)
            if key in seen:
                shown = ", ".join("null" if v is None else T.output(v, table.columns[i].type)
                                  for v, i in zip(vals, idx))
                raise PgError(f'could not create unique index "{name}"', "23505",
                              detail=f"Key ({', '.join(kc.cols)})=({shown}) is duplicated.")
            seen.add(key)

    def _backing_index(self, table, kc, primary=False):
        ix = Index(kc.name, table.schema, table.name, [IndexElem(col=c) for c in kc.cols], unique=True,
                   primary=primary, constraint=kc.name)
        ix.oid = self.db.new_oid()
        ix.nulls_not_distinct = kc.nulls_not_distinct
        self.db.schemas[table.schema].rels[kc.name] = ix
        return ix

    # ------------------------------------------------------------ CREATE TABLE AS
    def create_table_as(self, st, schema):
        from .executor import run_query, Ctx
        q = self.an.query(st.as_query, None)
        names = list(q.names[:q.ntargets])
        if st.as_cols:
            if len(st.as_cols) > len(names):
                raise PgError("too many column names were specified", "42601")
            names[:len(st.as_cols)] = st.as_cols
        for i, n in enumerate(names):
            if n in names[:i]:
                raise PgError(f'column "{n}" specified more than once', "42701")
        types = list(q.types[:q.ntargets])
        cols = []
        for n, t in zip(names, types):
            if t.name == "unknown":
                t = T.TEXT
            cols.append(Column(n, t))
        table = Table(st.table.name, schema, cols, "t" if st.temp else "p")
        table.oid = self.db.new_oid()
        rows = []
        if st.with_data:
            c = Ctx(self.session, self.db)
            c.outer.append(None)
            rows = run_query(q, c)
        table.rows = [tuple(r) for r in rows]
        self.db.schemas[schema].rels[table.name] = table
        return ("SELECT", len(rows))

    # ------------------------------------------------------------ indexes
    def create_index(self, st: A.CreateIndex):
        db = self.db
        rv = st.table
        table = db.find_rel(rv.name, rv.schema, self.session)
        if table is None:
            raise PgError(f'relation "{rv.name}" does not exist', "42P01")
        if isinstance(table, View) and not table.materialized:
            raise PgError(f'cannot create index on relation "{table.name}"', "42809",
                          detail="This operation is not supported for views.")
        method = (st.method or "btree").lower()
        if method not in ("btree", "hash", "gin", "gist", "brin", "spgist"):
            raise PgError(f'access method "{method}" does not exist', "42704")
        scope = Scope(None, self.session, db)
        rte = RTE("rel", table.name, table.colnames, table.coltypes, scope.alloc(len(table.columns)))
        scope.ns.append(NSItem(rte))
        scope.kind = "index"
        elems = []
        name_parts = []
        for el in st.elems:
            if el.name is not None:
                if table.col_index(el.name) < 0:
                    raise PgError(f'column "{el.name}" does not exist', "42703", pos=el.pos + 1)
                elems.append(IndexElem(col=el.name, desc=el.desc, nulls_first=None if el.nulls is None
                                       else el.nulls == "first", opclass=el.opclass))
                name_parts.append(el.name)
            else:
                e = self.an.expr(el.expr, scope)
                for n in e.walk():
                    if isinstance(n, E.Func) and n.volatile:
                        raise PgError("functions in index expression must be marked IMMUTABLE", "42P17")
                    if isinstance(n, E.Func) and n.name in ("now", "current_timestamp", "age", "to_char") and \
                            any(a.type.name == "timestamptz" for a in n.args):
                        raise PgError("functions in index expression must be marked IMMUTABLE", "42P17")
                ie = IndexElem(expr=el.expr, expr_sql=_index_expr_sql(e, el.expr_text), desc=el.desc,
                               nulls_first=None if el.nulls is None else el.nulls == "first")
                elems.append(ie)
                if isinstance(e, E.Func):
                    name_parts.append(e.name)
                else:
                    name_parts.append("expr")
        if method == "gin":
            for el in elems:
                t = table.col(el.col).type if el.col else None
                if t is not None and not (t.name == "jsonb" or t.is_array):
                    raise PgError(f'data type {t.format()} has no default operator class for access method "gin"',
                                  "42704", hint="You must specify an operator class for the index or define a "
                                                "default operator class for the data type.")
        pred = None
        if st.where is not None:
            scope.kind = "index_pred"
            pe = self.an.expr(st.where, scope)
            pred = st.where
        schema = table.schema
        if st.name is None:
            name = choose_name(db, schema, [table.name] + name_parts, "idx")
        else:
            name = st.name
            if name in db.schemas[schema].rels:
                if st.if_not_exists:
                    self.notice(f'relation "{name}" already exists, skipping')
                    return None
                raise PgError(f'relation "{name}" already exists', "42P07")
        ix = Index(name, schema, table.name, elems, unique=st.unique, method=method, pred=pred,
                   pred_sql=deparse_check(pe) if pred is not None else None, include=st.include)
        ix.nulls_not_distinct = st.nulls_not_distinct
        ix.oid = db.new_oid()
        if st.unique:
            if method != "btree":
                raise PgError(f'access method "{method}" does not support unique indexes', "0A000")
            from .dml import _index_keyf
            from .executor import Ctx
            c = Ctx(self.session, db)
            _, shown, keyf, _ = _index_keyf(table, ix, c)
            seen = {}
            for r in table.rows:
                k = keyf(r)
                if k is None:
                    continue
                if k in seen:
                    from .dml import _unique_detail
                    db.schemas[schema].rels[name] = ix
                    try:
                        detail = _unique_detail(table, shown, r, c, name).replace(" already exists.", " is duplicated.")
                    finally:
                        del db.schemas[schema].rels[name]
                    raise PgError(f'could not create unique index "{name}"', "23505", detail=detail)
                seen[k] = r
        db.schemas[schema].rels[name] = ix
        return ix

    # ------------------------------------------------------------ views
    def create_view(self, st: A.CreateView):
        from .executor import run_query, Ctx
        db = self.db
        rv = st.view
        schema = db.creation_schema(rv.schema, self.session, st.temp)
        q = self.an.query(st.query, None)
        names = list(q.names[:q.ntargets])
        if st.cols:
            if len(st.cols) > len(names):
                raise PgError("CREATE VIEW specifies more column names than columns", "42601")
            names[:len(st.cols)] = st.cols
        for i, n in enumerate(names):
            if n in names[:i]:
                raise PgError(f'column "{n}" specified more than once', "42701")
        types = [T.TEXT if t.name == "unknown" else t for t in q.types[:q.ntargets]]
        existing = db.schemas[schema].rels.get(rv.name)
        if existing is not None:
            if st.materialized:
                if st.if_not_exists:
                    self.notice(f'relation "{rv.name}" already exists, skipping')
                    return None
                raise PgError(f'relation "{rv.name}" already exists', "42P07")
            if not st.replace:
                raise PgError(f'relation "{rv.name}" already exists', "42P07")
            if not isinstance(existing, View) or existing.materialized:
                raise PgError(f'"{rv.name}" is not a view', "42809")
            if len(names) < len(existing.colnames):
                raise PgError("cannot drop columns from view", "42P16")
            for i, (a, b) in enumerate(zip(existing.colnames, names)):
                if a != b:
                    raise PgError(f'cannot change name of view column "{a}" to "{b}"', "42P16",
                                  hint="Use ALTER VIEW ... RENAME COLUMN ... to change name of view column instead.")
                if existing.coltypes[i] != types[i]:
                    raise PgError(f'cannot change data type of view column "{a}" from '
                                  f'{existing.coltypes[i].format()} to {types[i].format()}', "42P16")
        sql = st.query_text.strip()
        v = View(rv.name, schema, st.query, view_def(sql), names, types, st.materialized)
        v.oid = existing.oid if existing is not None else db.new_oid()
        v.check_option = st.check_option
        if st.temp:
            v.persistence = "t"
        if st.materialized:
            if st.with_data:
                c = Ctx(self.session, db)
                c.outer.append(None)
                v.rows = [tuple(r) for r in run_query(q, c)]
                v.populated = True
            else:
                v.populated = False
        db.schemas[schema].rels[rv.name] = v
        if st.materialized:
            return ("SELECT", len(v.rows))
        return v

    def refresh(self, st: A.RefreshMatView):
        from .executor import run_query, Ctx
        rv = st.view
        v = self.db.find_rel(rv.name, rv.schema, self.session)
        if v is None:
            raise PgError(f'relation "{rv.name}" does not exist', "42P01")
        if not isinstance(v, View) or not v.materialized:
            raise PgError(f'"{rv.name}" is not a materialized view', "42809")
        if st.concurrently:
            if not any(ix.unique and ix.pred is None for ix in self.db.indexes_of(v)):
                raise PgError(f'cannot refresh materialized view "{v.schema}.{v.name}" concurrently', "55000",
                              hint="Create a unique index with no WHERE clause on one or more columns of the "
                                   "materialized view.")
            if not v.populated:
                raise PgError("CONCURRENTLY cannot be used when the materialized view is not populated", "0A000")
        if st.with_data:
            q = Analyzer(self.session, self.db).query(v.query, None)
            c = Ctx(self.session, self.db)
            c.outer.append(None)
            v.rows = [tuple(r) for r in run_query(q, c)]
            v.populated = True
        else:
            v.rows = []
            v.populated = False

    # ------------------------------------------------------------ DROP
    def drop(self, st: A.Drop):
        db = self.db
        kind = st.kind
        if kind in ("schema",):
            for rv in st.names:
                self.drop_schema(rv.name, st.if_exists, st.cascade)
            return
        if kind == "database":
            for rv in st.names:
                self.server.drop_database(rv.name, st.if_exists, self.session, self.notices)
            return
        if kind == "extension":
            for rv in st.names:
                if rv.name not in db.extensions:
                    if st.if_exists:
                        self.notice(f'extension "{rv.name}" does not exist, skipping')
                        continue
                    raise PgError(f'extension "{rv.name}" does not exist', "42704")
                db.extensions.discard(rv.name)
            return
        if kind == "type":
            for rv in st.names:
                et = db.find_type(rv.name, rv.schema, self.session)
                if et is None:
                    if st.if_exists:
                        self.notice(f'type "{rv.name}" does not exist, skipping')
                        continue
                    raise PgError(f'type "{rv.name}" does not exist', "42704")
                for rel in db.all_rels():
                    if isinstance(rel, Table) and any(c.type.name == et.name for c in rel.columns):
                        if not st.cascade:
                            raise PgError(f'cannot drop type {rv.name} because other objects depend on it', "2BP01",
                                          detail=f'column {next(c.name for c in rel.columns if c.type.name == et.name)}'
                                                 f' of table {rel.name} depends on type {rv.name}',
                                          hint="Use DROP ... CASCADE to drop the dependent objects too.")
                del db.schemas[et.schema].types[et.name]
            return
        if kind in ("function", "procedure", "domain", "trigger"):
            raise Unsupported(f"DROP {kind.upper()}")
        want = {"table": Table, "view": View, "materialized view": View, "index": Index, "sequence": Sequence}[kind]
        targets = []
        for rv in st.names:
            rel = db.find_rel(rv.name, rv.schema, self.session)
            if rel is None:
                if st.if_exists:
                    self.notice(f'{kind} "{rv.name}" does not exist, skipping')
                    continue
                shown = f"{rv.schema}.{rv.name}" if rv.schema else rv.name
                raise PgError(f'{kind} "{shown}" does not exist', "42P01" if kind != "index" else "42704")
            ok = isinstance(rel, want)
            if ok and kind == "view" and rel.materialized:
                ok = False
            if ok and kind == "materialized view" and not rel.materialized:
                ok = False
            if not ok:
                actual = rel_kind_name(rel)
                hint = {"table": "Use DROP TABLE to remove a table.", "view": "Use DROP VIEW to remove a view.",
                        "materialized view": "Use DROP MATERIALIZED VIEW to remove a materialized view.",
                        "index": "Use DROP INDEX to remove an index.",
                        "sequence": "Use DROP SEQUENCE to remove a sequence."}[actual]
                raise PgError(f'"{rv.name}" is not a{"n" if kind[0] in "aeiou" else ""} {kind}', "42809", hint=hint)
            if isinstance(rel, Index) and rel.constraint is not None:
                tab = rel.table
                raise PgError(f'cannot drop index {rel.name} because constraint {rel.constraint} on table {tab} '
                              f'requires it', "2BP01",
                              hint=f"You can drop constraint {rel.constraint} on table {tab} instead.")
            targets.append(rel)
        for rel in targets:
            self.drop_rel(rel, st.cascade)

    def dependents(self, rel):
        """Objects that depend on rel: views reading it, foreign keys referencing it."""
        db = self.db
        out = []
        for other in db.all_rels():
            if isinstance(other, View) and other is not rel:
                if rel.name in _referenced_names(other.query):
                    out.append(("view", other))
        if isinstance(rel, Table):
            for other in db.all_rels():
                if isinstance(other, Table) and other is not rel:
                    for k in other.constraints:
                        if k.kind == "f" and k.ref_table == rel.name and k.ref_schema == rel.schema:
                            out.append(("constraint", (k, other)))
        return out

    def drop_rel(self, rel, cascade, quiet=False):
        db = self.db
        deps = self.dependents(rel)
        if deps and not cascade:
            first = deps[0]
            what = rel_kind_name(rel)
            details = []
            for kind, obj in deps:
                if kind == "view":
                    details.append(f"{rel_kind_name(obj)} {obj.name} depends on {what} {rel.name}")
                else:
                    k, t = obj
                    details.append(f"constraint {k.name} on table {t.name} depends on {what} {rel.name}")
            raise PgError(f"cannot drop {what} {rel.name} because other objects depend on it", "2BP01",
                          detail="\n".join(details), hint="Use DROP ... CASCADE to drop the dependent objects too.")
        if deps:
            msgs = []
            for kind, obj in deps:
                if kind == "view":
                    msgs.append(f"drop cascades to {rel_kind_name(obj)} {obj.name}")
                    self.drop_rel(obj, True, quiet=True)
                else:
                    k, t = obj
                    msgs.append(f"drop cascades to constraint {k.name} on table {t.name}")
                    t.constraints = [x for x in t.constraints if x is not k]
            if not quiet:
                if len(msgs) == 1:
                    self.notice(msgs[0])
                else:
                    self.notices.append(("NOTICE", f"drop cascades to {len(msgs)} other objects",
                                         "\n".join(msgs), None))
        s = db.schemas[rel.schema]
        s.rels.pop(rel.name, None)
        if isinstance(rel, (Table, View)):
            for ix in [r for r in s.rels.values() if isinstance(r, Index) and r.table == rel.name]:
                s.rels.pop(ix.name, None)
            for seq in [r for r in s.rels.values() if isinstance(r, Sequence) and r.owned_by and
                        r.owned_by[0] == rel.name]:
                s.rels.pop(seq.name, None)
        if isinstance(rel, Index) and rel.constraint is not None:
            pass

    def drop_schema(self, name, if_exists, cascade):
        db = self.db
        s = db.schemas.get(name)
        if s is None:
            if if_exists:
                self.notice(f'schema "{name}" does not exist, skipping')
                return
            raise PgError(f'schema "{name}" does not exist', "3F000")
        if s.rels and not cascade:
            first = next(iter(s.rels.values()))
            raise PgError(f"cannot drop schema {name} because other objects depend on it", "2BP01",
                          detail="\n".join(f"{rel_kind_name(r)} {name}.{r.name} depends on schema {name}"
                                           for r in s.rels.values() if not isinstance(r, (Index,)) and
                                           not (isinstance(r, Sequence) and r.owned_by)),
                          hint="Use DROP ... CASCADE to drop the dependent objects too.")
        if s.rels:
            objs = [r for r in s.rels.values() if not isinstance(r, Index) and not (isinstance(r, Sequence) and
                                                                                    r.owned_by)]
            if len(objs) == 1:
                self.notice(f"drop cascades to {rel_kind_name(objs[0])} {name}.{objs[0].name}")
            elif objs:
                self.notices.append(("NOTICE", f"drop cascades to {len(objs)} other objects",
                                     "\n".join(f"drop cascades to {rel_kind_name(r)} {name}.{r.name}" for r in objs),
                                     None))
        del db.schemas[name]

    # ------------------------------------------------------------ ALTER TABLE
    def alter_table(self, st: A.AlterTable):
        db = self.db
        rv = st.table
        rel = db.find_rel(rv.name, rv.schema, self.session)
        if rel is None:
            if st.if_exists:
                self.notice(f'relation "{rv.name}" does not exist, skipping')
                return
            raise PgError(f'relation "{rv.name}" does not exist', "42P01")
        if not isinstance(rel, Table):
            if isinstance(rel, View) and len(st.cmds) == 1 and st.cmds[0].kind in ("rename_table", "rename_column"):
                pass
            else:
                raise PgError(f'"{rv.name}" is not a table', "42809")
        for cmd in st.cmds:
            getattr(self, "at_" + cmd.kind)(rel, cmd)

    def at_noop(self, t, cmd):
        pass

    def at_add_column(self, t, cmd):
        cd = cmd.coldef
        if t.col_index(cd.name) >= 0:
            if cmd.if_not_exists:
                self.notice(f'column "{cd.name}" of relation "{t.name}" already exists, skipping')
                return
            raise PgError(f'column "{cd.name}" of relation "{t.name}" already exists', "42701")
        pending, seqs = [], []
        tmp = Table(t.name, t.schema, [])
        self.add_column_def(tmp, cd, pending, seqs)
        col = tmp.columns[0]
        for s in seqs:
            self.db.schemas[t.schema].rels[s.name] = s
        # existing rows get the default (evaluated per row: nextval gives each its own value)
        from .executor import Ctx
        from .dml import assign_coerce
        c = Ctx(self.session, self.db)
        vals = [None] * len(t.rows)
        if col.default is not None:
            scope = Scope(None, self.session, self.db)
            scope.kind = "default"
            e = assign_coerce(self.an.expr(col.default, scope), col, col.default).compile()
            vals = [e(None, c) for _ in t.rows]
        if col.generated is not None:
            scope = Scope(None, self.session, self.db)
            rte = RTE("rel", t.name, t.colnames, t.coltypes, scope.alloc(len(t.columns)))
            scope.ns.append(NSItem(rte))
            g = self.an.expr(col.generated, scope).compile()
            vals = [g(r, c) for r in t.rows]
        if col.not_null and any(v is None for v in vals):
            raise PgError(f'column "{col.name}" of relation "{t.name}" contains null values', "23502")
        t.columns.append(col)
        t.rows = [r + (v,) for r, v in zip(t.rows, vals)]
        for con in pending:
            con._column = col.name
            self.add_constraint(t, con)

    def at_drop_column(self, t, cmd):
        i = t.col_index(cmd.name)
        if i < 0:
            if cmd.if_exists:
                self.notice(f'column "{cmd.name}" of relation "{t.name}" does not exist, skipping')
                return
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        db = self.db
        # views that use the column
        for kind, obj in self.dependents(t):
            if kind == "view" and cmd.name in _referenced_columns(obj.query):
                if not cmd.cascade:
                    raise PgError(f"cannot drop column {cmd.name} of table {t.name} because other objects depend on "
                                  "it", "2BP01", detail=f"view {obj.name} depends on column {cmd.name} of table "
                                                        f"{t.name}",
                                  hint="Use DROP ... CASCADE to drop the dependent objects too.")
                self.notice(f"drop cascades to view {obj.name}")
                self.drop_rel(obj, True, quiet=True)
        # foreign keys from other tables referencing this column
        for other in list(db.all_rels()):
            if isinstance(other, Table) and other is not t:
                for k in list(other.constraints):
                    if k.kind == "f" and k.ref_table == t.name and cmd.name in k.ref_cols:
                        if not cmd.cascade:
                            raise PgError(f"cannot drop column {cmd.name} of table {t.name} because other objects "
                                          "depend on it", "2BP01",
                                          detail=f"constraint {k.name} on table {other.name} depends on column "
                                                 f"{cmd.name} of table {t.name}",
                                          hint="Use DROP ... CASCADE to drop the dependent objects too.")
                        self.notice(f"drop cascades to constraint {k.name} on table {other.name}")
                        other.constraints.remove(k)
        col = t.columns[i]
        t.constraints = [k for k in t.constraints if cmd.name not in k.cols]
        s = db.schemas[t.schema]
        for ix in [r for r in s.rels.values() if isinstance(r, Index) and r.table == t.name]:
            if any(el.col == cmd.name for el in ix.elems) or (ix.pred is not None and
                                                             cmd.name in _referenced_columns(ix.pred)) or \
                    any(el.expr is not None and cmd.name in _referenced_columns(el.expr) for el in ix.elems):
                s.rels.pop(ix.name, None)
        if col.seq:
            s.rels.pop(col.seq, None)
        del t.columns[i]
        t.rows = [r[:i] + r[i + 1:] for r in t.rows]

    def at_rename_column(self, t, cmd):
        i = t.col_index(cmd.name)
        if i < 0:
            raise PgError(f'column "{cmd.name}" does not exist', "42703")
        if t.col_index(cmd.newname) >= 0:
            raise PgError(f'column "{cmd.newname}" of relation "{t.name}" already exists', "42701")
        old = cmd.name
        if isinstance(t, View):
            t.colnames[i] = cmd.newname
            t.columns[i].name = cmd.newname
            return
        t.columns[i].name = cmd.newname
        for k in t.constraints:
            k.cols = [cmd.newname if n == old else n for n in k.cols]
            if k.kind == "c":
                k.expr = _rename_in_tree(k.expr, old, cmd.newname)
                k.expr_sql = _rename_sql(k.expr_sql, old, cmd.newname)
        for ix in self.db.indexes_of(t):
            for el in ix.elems:
                if el.col == old:
                    el.col = cmd.newname
        for other in self.db.all_rels():
            if isinstance(other, Table):
                for k in other.constraints:
                    if k.kind == "f" and k.ref_table == t.name:
                        k.ref_cols = [cmd.newname if n == old else n for n in k.ref_cols]

    def at_rename_table(self, t, cmd):
        s = self.db.schemas[t.schema]
        if cmd.newname in s.rels:
            raise PgError(f'relation "{cmd.newname}" already exists', "42P07")
        old = t.name
        del s.rels[old]
        t.name = cmd.newname
        s.rels[cmd.newname] = t
        for r in s.rels.values():
            if isinstance(r, Index) and r.table == old:
                r.table = cmd.newname
            if isinstance(r, Sequence) and r.owned_by and r.owned_by[0] == old:
                r.owned_by = (cmd.newname, r.owned_by[1])
        for other in self.db.all_rels():
            if isinstance(other, Table):
                for k in other.constraints:
                    if k.kind == "f" and k.ref_table == old:
                        k.ref_table = cmd.newname

    def at_rename_constraint(self, t, cmd):
        hit = [k for k in t.constraints if k.name == cmd.name]
        if not hit:
            raise PgError(f'constraint "{cmd.name}" for table "{t.name}" does not exist', "42704")
        k = hit[0]
        k.name = cmd.newname
        if k.index:
            s = self.db.schemas[t.schema]
            ix = s.rels.pop(k.index, None)
            if ix is not None:
                ix.name = cmd.newname
                ix.constraint = cmd.newname
                s.rels[cmd.newname] = ix
            k.index = cmd.newname

    def at_set_schema(self, t, cmd):
        db = self.db
        if cmd.newname not in db.schemas:
            raise PgError(f'schema "{cmd.newname}" does not exist', "3F000")
        del db.schemas[t.schema].rels[t.name]
        t.schema = cmd.newname
        db.schemas[cmd.newname].rels[t.name] = t

    def at_alter_type(self, t, cmd):
        i = t.col_index(cmd.name)
        if i < 0:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        col = t.columns[i]
        new_t = self.an.resolve_type(cmd.typ)
        from .executor import Ctx
        c = Ctx(self.session, self.db)
        scope = Scope(None, self.session, self.db)
        rte = RTE("rel", t.name, t.colnames, t.coltypes, scope.alloc(len(t.columns)))
        scope.ns.append(NSItem(rte))
        if cmd.using is not None:
            e = self.an.expr(cmd.using, scope)
        else:
            e = E.Var(0, i, col.type, t.name, col.name)
        from .dml import assign_coerce
        tmpcol = Column(col.name, new_t)
        try:
            conv = assign_coerce(e, tmpcol, None)
        except PgError as err:
            if cmd.using is None and err.code == "42804":
                raise PgError(f'column "{col.name}" cannot be cast automatically to type {new_t.format()}', "42804",
                              hint=f'You might need to specify "USING {col.name}::{new_t.format()}".') from None
            if cmd.using is not None and err.code == "42804":
                raise PgError(f'result of USING clause for column "{col.name}" cannot be cast automatically to type '
                              f'{new_t.format()}', "42804",
                              hint="You might need to add an explicit cast.") from None
            raise
        f = conv.compile()
        # a default that no longer fits the type
        newrows = []
        for r in t.rows:
            v = f(r, c)
            newrows.append(r[:i] + (v,) + r[i + 1:])
        col.type = new_t
        if col.default is not None:
            try:
                _bind_default(self.an, col, col.default)
            except PgError:
                raise PgError(f'default for column "{col.name}" cannot be cast automatically to type '
                              f'{new_t.format()}', "42804") from None
        t.rows = newrows
        from .dml import check_row, check_table
        for r in t.rows:
            check_row(t, r, c)
        check_table(c, t, t.rows, changed=list(t.rows), removed=[])

    def at_set_default(self, t, cmd):
        col = t.col(cmd.name)
        if col is None:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        if col.identity:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" is an identity column', "55000",
                          hint="Use ALTER TABLE ... ALTER COLUMN ... DROP IDENTITY instead.")
        raw, bound = _bind_default(self.an, col, cmd.expr)
        col.default = cmd.expr
        col.default_sql = default_sql(bound)

    def at_drop_default(self, t, cmd):
        col = t.col(cmd.name)
        if col is None:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        if col.identity:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" is an identity column', "55000",
                          hint="Use ALTER TABLE ... ALTER COLUMN ... DROP IDENTITY instead.")
        col.default = None
        col.default_sql = None

    def at_set_not_null(self, t, cmd):
        i = t.col_index(cmd.name)
        if i < 0:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        if any(r[i] is None for r in t.rows):
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" contains null values', "23502")
        t.columns[i].not_null = True

    def at_drop_not_null(self, t, cmd):
        col = t.col(cmd.name)
        if col is None:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        pk = t.pkey()
        if pk is not None and cmd.name in pk.cols:
            raise PgError(f'column "{cmd.name}" is in a primary key', "42P16")
        if col.identity:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" is an identity column', "42P16")
        col.not_null = False

    def at_add_constraint(self, t, cmd):
        self.add_constraint(t, cmd.constraint)

    def at_drop_constraint(self, t, cmd):
        hit = [k for k in t.constraints if k.name == cmd.name]
        if not hit:
            if cmd.if_exists:
                self.notice(f'constraint "{cmd.name}" of relation "{t.name}" does not exist, skipping')
                return
            raise PgError(f'constraint "{cmd.name}" of relation "{t.name}" does not exist', "42704")
        k = hit[0]
        if k.kind in ("p", "u"):
            for other in self.db.all_rels():
                if isinstance(other, Table):
                    for fk in list(other.constraints):
                        if fk.kind == "f" and fk.ref_table == t.name and set(fk.ref_cols) == set(k.cols):
                            if not cmd.cascade:
                                raise PgError(f"cannot drop constraint {k.name} on table {t.name} because other "
                                              "objects depend on it", "2BP01",
                                              detail=f"constraint {fk.name} on table {other.name} depends on index "
                                                     f"{k.index}",
                                              hint="Use DROP ... CASCADE to drop the dependent objects too.")
                            self.notice(f"drop cascades to constraint {fk.name} on table {other.name}")
                            other.constraints.remove(fk)
            self.db.schemas[t.schema].rels.pop(k.index, None)
        t.constraints.remove(k)

    def at_validate_constraint(self, t, cmd):
        hit = [k for k in t.constraints if k.name == cmd.name]
        if not hit:
            raise PgError(f'constraint "{cmd.name}" of relation "{t.name}" does not exist', "42704")
        hit[0].validated = True

    def at_add_identity(self, t, cmd):
        col = t.col(cmd.name)
        if col is None:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" does not exist', "42703")
        if not col.not_null:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" must be declared NOT NULL before identity '
                          'can be added', "55000")
        if col.identity:
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" is already an identity column', "42611")
        col.identity = "a" if cmd.constraint.identity == "always" else "d"
        seq = self.owned_sequence(t, col, col.type, cmd.constraint.seq_opts)
        self.db.schemas[t.schema].rels[seq.name] = seq

    def at_drop_identity(self, t, cmd):
        col = t.col(cmd.name)
        if col is None or not col.identity:
            if cmd.if_exists:
                self.notice(f'column "{cmd.name}" of relation "{t.name}" is not an identity column, skipping')
                return
            raise PgError(f'column "{cmd.name}" of relation "{t.name}" is not an identity column', "55000")
        col.identity = None
        col.default = None
        if col.seq:
            self.db.schemas[t.schema].rels.pop(col.seq, None)
            col.seq = None

    # ------------------------------------------------------------ TRUNCATE
    def truncate(self, st: A.Truncate):
        db = self.db
        tables = []
        for rv in st.tables:
            t = db.find_rel(rv.name, rv.schema, self.session)
            if t is None:
                raise PgError(f'relation "{rv.name}" does not exist', "42P01")
            if not isinstance(t, Table):
                raise PgError(f'"{rv.name}" is not a table', "42809")
            tables.append(t)
        names = {(t.schema, t.name) for t in tables}
        changed = True
        while changed:
            changed = False
            for other in list(db.all_rels()):
                if isinstance(other, Table) and (other.schema, other.name) not in names:
                    for k in other.constraints:
                        if k.kind == "f" and (k.ref_schema, k.ref_table) in names:
                            if not st.cascade:
                                ref = k.ref_table
                                raise PgError("cannot truncate a table referenced in a foreign key constraint",
                                              "0A000", detail=f'Table "{other.name}" references "{ref}".',
                                              hint=f'Truncate table "{other.name}" at the same time, or use '
                                                   'TRUNCATE ... CASCADE.')
                            self.notice(f'truncate cascades to table "{other.name}"')
                            names.add((other.schema, other.name))
                            tables.append(other)
                            changed = True
                            break
        for t in tables:
            t.rows = []
            if st.restart_identity:
                for col in t.columns:
                    if col.seq:
                        seq = db.schemas[t.schema].rels.get(col.seq)
                        if seq is not None:
                            seq.last = seq.start
                            seq.is_called = False

    # ------------------------------------------------------------ misc
    def create_schema(self, st: A.CreateSchema):
        db = self.db
        if st.name in db.schemas:
            if st.if_not_exists:
                self.notice(f'schema "{st.name}" already exists, skipping')
                return
            raise PgError(f'schema "{st.name}" already exists', "42P06")
        if st.name.startswith("pg_"):
            raise PgError(f'unacceptable schema name "{st.name}"', "42939",
                          detail='The prefix "pg_" is reserved for system schemas.')
        db.schemas[st.name] = Schema(st.name, st.authorization or self.session.user)

    def create_sequence(self, st: A.CreateSequence):
        db = self.db
        rv = st.seq
        schema = db.creation_schema(rv.schema, self.session, st.temp)
        if rv.name in db.schemas[schema].rels:
            if st.if_not_exists:
                self.notice(f'relation "{rv.name}" already exists, skipping')
                return
            raise PgError(f'relation "{rv.name}" already exists', "42P07")
        seq = Sequence(rv.name, schema)
        self.seq_opts(seq, st.options)
        seq.oid = db.new_oid()
        db.schemas[schema].rels[rv.name] = seq

    def alter_simple(self, st: A.AlterSimple):
        db = self.db
        rv = st.name
        if st.kind == "schema":
            s = db.schemas.get(rv.name)
            if s is None:
                raise PgError(f'schema "{rv.name}" does not exist', "3F000")
            if st.action == "rename":
                if st.newname in db.schemas:
                    raise PgError(f'schema "{st.newname}" already exists', "42P06")
                del db.schemas[rv.name]
                s.name = st.newname
                db.schemas[st.newname] = s
                for r in s.rels.values():
                    r.schema = st.newname
            return
        if st.kind == "type":
            et = db.find_type(rv.name, rv.schema, self.session)
            if et is None:
                raise PgError(f'type "{rv.name}" does not exist', "42704")
            if st.action == "add_value":
                label = st.options[0]
                if label in et.labels:
                    raise PgError(f'enum label "{label}" already exists', "42710")
                if len(st.options) > 1:
                    where, ref = st.options[1]
                    if ref not in et.labels:
                        raise PgError(f'"{ref}" is not an existing enum label', "22023")
                    k = et.labels.index(ref) + (1 if where == "after" else 0)
                    et.labels.insert(k, label)
                else:
                    et.labels.append(label)
                return
            if st.action == "rename":
                del db.schemas[et.schema].types[et.name]
                et.name = st.newname
                db.schemas[et.schema].types[et.name] = et
                return
            return
        rel = db.find_rel(rv.name, rv.schema, self.session)
        if rel is None:
            if st.if_exists:
                self.notice(f'relation "{rv.name}" does not exist, skipping')
                return
            raise PgError(f'relation "{rv.name}" does not exist', "42P01")
        if st.action == "rename":
            s = db.schemas[rel.schema]
            if st.newname in s.rels:
                raise PgError(f'relation "{st.newname}" already exists', "42P07")
            del s.rels[rel.name]
            if isinstance(rel, Index) and rel.constraint:
                t = s.rels.get(rel.table)
                for k in t.constraints:
                    if k.name == rel.constraint:
                        k.name = st.newname
                        k.index = st.newname
                rel.constraint = st.newname
            rel.name = st.newname
            s.rels[st.newname] = rel
        elif st.action == "rename_column" and isinstance(rel, View):
            old = st.options[0]
            if old not in rel.colnames:
                raise PgError(f'column "{old}" does not exist', "42703")
            i = rel.colnames.index(old)
            rel.colnames[i] = st.newname
        elif st.action == "options" and isinstance(rel, Sequence):
            self.seq_opts(rel, st.options)
        elif st.action == "set_schema":
            if st.newname not in db.schemas:
                raise PgError(f'schema "{st.newname}" does not exist', "3F000")
            del db.schemas[rel.schema].rels[rel.name]
            rel.schema = st.newname
            db.schemas[st.newname].rels[rel.name] = rel

    def create_type(self, st: A.CreateType):
        db = self.db
        rv = st.name
        schema = db.creation_schema(rv.schema, self.session)
        if rv.name in db.schemas[schema].types or T.resolve_name(rv.name) is not None:
            raise PgError(f'type "{rv.name}" already exists', "42710")
        for i, l in enumerate(st.labels):
            if l in st.labels[:i]:
                raise PgError(f'enum label "{l}" used more than once', "42710")
        db.schemas[schema].types[rv.name] = EnumType(rv.name, schema, st.labels)

    def comment(self, st: A.Comment):
        db = self.db
        kind = st.kind
        names = st.target
        if kind == "column":
            if len(names) < 2:
                raise PgError("column name must be qualified", "42601")
            tname, cname = names[-2], names[-1]
            schema = names[-3] if len(names) > 2 else None
            t = db.find_rel(tname, schema, self.session)
            if t is None:
                raise PgError(f'relation "{tname}" does not exist', "42P01")
            col = t.col(cname) if isinstance(t, Table) else None
            if col is None:
                raise PgError(f'column "{cname}" of relation "{tname}" does not exist', "42703")
            col.comment = st.text
            return
        if kind == "schema":
            s = db.schemas.get(names[-1])
            if s is None:
                raise PgError(f'schema "{names[-1]}" does not exist', "3F000")
            s.comment = st.text
            return
        if kind == "database":
            db.comment = st.text
            return
        if kind in ("table", "view", "materialized view", "index", "sequence"):
            rel = db.find_rel(names[-1], names[-2] if len(names) > 1 else None, self.session)
            if rel is None:
                raise PgError(f'relation "{".".join(names)}" does not exist', "42P01")
            rel.comment = st.text
            return
        raise Unsupported(f"COMMENT ON {kind.upper()}")

    def analyze(self, st: A.Analyze):
        db = self.db
        tables = []
        if st.tables:
            for rv in st.tables:
                t = db.find_rel(rv.name, rv.schema, self.session)
                if t is None:
                    raise PgError(f'relation "{rv.name}" does not exist', "42P01")
                tables.append(t)
        else:
            tables = [r for r in db.all_rels() if isinstance(r, Table) or (isinstance(r, View) and r.materialized)]
        for t in tables:
            if isinstance(t, (Table, View)):
                compute_stats(t)
                if st.vacuum:
                    t.vacuumed = getattr(t, "n_changes", 0)


def autovacuum(t):
    """What autovacuum has done to a table that has sat for a while: statistics, and every page all-visible."""
    compute_stats(t)
    t.vacuumed = getattr(t, "n_changes", 0)


def compute_stats(t):
    """What ANALYZE records for the planner: row count, pages, and per column the null fraction, the number of
    distinct values, the most common values and a histogram."""
    from .planner import table_pages
    rows = t.rows
    n = len(rows)
    cols = {}
    types = [c.type for c in t.columns]
    for i, col in enumerate(t.columns):
        vals = [r[i] for r in rows]
        nonnull = [v for v in vals if v is not None]
        nulls = n - len(nonnull)
        counts = {}
        for v in nonnull:
            k = T.hash_key(v)
            counts[k] = counts.get(k, 0) + 1
        nd = len(counts)
        if nonnull and nd > 0.1 * n and nd == len(nonnull):
            n_distinct = -1.0
        elif n and nd / n > 0.1:
            n_distinct = -(nd / n)
        else:
            n_distinct = float(nd)
        mcv = []
        if nonnull:
            freq = sorted(counts.items(), key=lambda kv: -kv[1])
            avg = len(nonnull) / max(nd, 1)
            mcv = [(k, cnt / n) for k, cnt in freq[:100] if cnt > 1 and cnt > avg * 1.25][:100]
        cols[col.name] = {"null_frac": nulls / n if n else 0.0, "n_distinct": n_distinct, "mcv": mcv,
                          "width": _avg_width(nonnull, types[i]),
                          "nd_abs": nd,
                          "sorted": _histogram(nonnull, types[i])}
    t.stats = {"reltuples": float(n), "relpages": table_pages(t), "cols": cols}
    t.n_mod = 0


def _safe_key(k):
    return k


def _histogram(vals, t):
    """Up to 101 quantile bounds of the non-null values (as sort keys and, for numbers and times, as floats)."""
    if not vals:
        return []
    keys = sorted(vals, key=lambda v: T.sort_key(v, t))
    n = len(keys)
    k = min(100, n - 1) if n > 1 else 1
    return [keys[min(n - 1, round(i * (n - 1) / k))] for i in range(k + 1)]


def _avg_width(vals, t):
    from .explain import col_width
    fixed = {"bool", "int2", "int4", "int8", "float4", "float8", "date", "timestamp", "timestamptz", "time",
             "interval", "uuid", "oid"}
    if t.name in fixed or not vals:
        return col_width(t, None)
    step = max(1, len(vals) // 300)
    sample = vals[::step]
    if t.name == "numeric":
        tot = 0
        for v in sample:
            if not v.is_finite():
                tot += 3
                continue
            w, _ = T.num_weight_first(v) if v != 0 else (0, 0)
            digits = len(format(abs(v), "f").replace(".", "").lstrip("0")) if v != 0 else 0
            groups = max(0, (digits + 3) // 4)
            tot += 3 + 2 * groups
        return max(1, int(tot / len(sample)))
    tot = 0
    for v in sample:
        b = len(T.output(v, t).encode("utf-8"))
        tot += b + (1 if b < 127 else 4)
    return max(1, int(tot / len(sample)))


# ---------------------------------------------------------------- helpers

def _referenced_names(node):
    out = set()

    def visit(n):
        if n is None:
            return
        if isinstance(n, A.RangeVar):
            out.add(n.name)
        if isinstance(n, (list, tuple)):
            for x in n:
                visit(x)
            return
        if isinstance(n, A.Node):
            for v in vars(n).values():
                if isinstance(v, (A.Node, list, tuple)):
                    visit(v)
    visit(node)
    return out


def _referenced_columns(node):
    out = set()

    def visit(n):
        if n is None:
            return
        if isinstance(n, A.ColumnRef):
            out.add(n.names[-1])
            if n.names[-1] == "*":
                out.add("*")
        if isinstance(n, A.Star):
            out.add("*")
        if isinstance(n, (list, tuple)):
            for x in n:
                visit(x)
            return
        if isinstance(n, A.Node):
            for v in vars(n).values():
                if isinstance(v, (A.Node, list, tuple)):
                    visit(v)
    visit(node)
    if "*" in out:
        return out | {"*"}
    return out


def _rename_in_tree(node, old, new):
    def visit(n):
        if isinstance(n, A.ColumnRef) and n.names[-1] == old:
            n.names = n.names[:-1] + [new]
        if isinstance(n, (list, tuple)):
            for x in n:
                visit(x)
            return
        if isinstance(n, A.Node):
            for v in vars(n).values():
                if isinstance(v, (A.Node, list, tuple)):
                    visit(v)
    visit(node)
    return node


def _rename_sql(sql, old, new):
    import re
    return re.sub(rf"\b{re.escape(old)}\b", new, sql or "")


def deparse_check(e) -> str:
    """A CHECK constraint / index predicate the way \\d shows it (pg_get_constraintdef)."""
    s = _cdp(e, True)
    return s


def _cdp(e, top=False):
    if isinstance(e, E.Var):
        return E.quote_ident(e.col)
    if isinstance(e, E.Const):
        return E.deparse_const(e.value, e.type)
    if isinstance(e, E.Cast):
        inner = _cdp(e.arg)
        if isinstance(e.arg, E.Const) and e.arg.type.name in ("int4", "numeric") and not inner.startswith("'"):
            return f"{inner}::{e.type.format()}"
        if isinstance(e.arg, E.Const):
            return f"{inner}::{e.type.format()}" if "::" not in inner else f"({inner})::{e.type.format()}"
        return f"({inner})::{e.type.format()}"
    if isinstance(e, E.Op):
        if len(e.args) == 1:
            return f"({e.op}{_cdp(e.args[0])})"
        s = f"{_cdp(e.args[0])} {e.op} {_cdp(e.args[1])}"
        return s if top else f"({s})"
    if isinstance(e, E.Bool):
        if e.op == "not":
            return f"(NOT {_cdp(e.args[0])})"
        s = f" {e.op.upper()} ".join(_cdp(a) for a in e.args)
        return s if top else f"({s})"
    if isinstance(e, E.NullTest):
        s = f"{_cdp(e.arg)} IS {'NOT ' if e.negated else ''}NULL"
        return s if top else f"({s})"
    if isinstance(e, E.ScalarArrayOp):
        arr = e.args[1]
        if isinstance(arr, E.Const):
            arr_s = E.deparse_const(arr.value, arr.type)
        else:
            arr_s = _cdp(arr)
        s = f"{_cdp(e.args[0])} {e.op} {'ANY' if e.use_or else 'ALL'} ({arr_s})"
        return s if top else f"({s})"
    if isinstance(e, E.Func):
        return f"{e.display or e.name}({', '.join(_cdp(a, True) for a in e.args)})"
    return e.deparse(E.DeparseCtx())


def _index_expr_sql(e, text):
    if isinstance(e, E.Func):
        return f"{e.name}({', '.join(_cdp(a, True) for a in e.args)})"
    return _cdp(e, True)


def view_def(sql: str) -> str:
    """The definition \\d+ shows (indented like pg_get_viewdef for simple cases)."""
    s = " ".join(sql.split())
    return " " + s + ";"

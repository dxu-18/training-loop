"""The server: a cluster of databases and roles, sessions with settings and transactions, statement dispatch.

Transactions work on copies of the database (catalog objects and row lists; rows are shared tuples): BEGIN takes
a copy, COMMIT installs it, ROLLBACK drops it, SAVEPOINT keeps a copy to go back to. An error inside a transaction
block leaves it aborted until ROLLBACK (or ROLLBACK TO SAVEPOINT), as in PostgreSQL. Outside a block every
statement that changes something runs on its own copy, installed only if the statement succeeds.
"""

from __future__ import annotations

import datetime as dt
import decimal
import random

from . import ast as A
from . import types as T
from .analyze import Analyzer, _now_holder
from .catalog import Database, Schema, Table, View, Sequence
from .errors import PgError, Unsupported
from .parser import Parser

SERVER_VERSION = "16.4 (Ubuntu 16.4-0ubuntu0.24.04.2)"

GUCS = {
    "timezone": "Etc/UTC", "datestyle": "ISO, MDY", "intervalstyle": "postgres", "search_path": '"$user", public',
    "client_encoding": "UTF8", "server_encoding": "UTF8", "standard_conforming_strings": "on",
    "statement_timeout": "0", "lock_timeout": "0", "work_mem": "4MB", "maintenance_work_mem": "64MB",
    "shared_buffers": "128MB", "effective_cache_size": "4GB", "random_page_cost": "4", "seq_page_cost": "1",
    "cpu_tuple_cost": "0.01", "cpu_index_tuple_cost": "0.005", "cpu_operator_cost": "0.0025",
    "enable_seqscan": "on", "enable_indexscan": "on", "enable_bitmapscan": "on", "enable_indexonlyscan": "on",
    "enable_hashjoin": "on", "enable_mergejoin": "on", "enable_nestloop": "on", "enable_hashagg": "on",
    "enable_sort": "on", "jit": "on", "max_connections": "100", "server_version": SERVER_VERSION,
    "server_version_num": "160004", "transaction_isolation": "read committed", "default_transaction_isolation":
    "read committed", "client_min_messages": "notice", "extra_float_digits": "1", "application_name": "psql",
    "lc_messages": "C.UTF-8", "lc_monetary": "C.UTF-8", "lc_numeric": "C.UTF-8", "lc_time": "C.UTF-8",
    "default_statistics_target": "100", "track_counts": "on", "autovacuum": "on", "log_statement": "none",
    "is_superuser": "off", "session_authorization": "ada", "default_text_search_config": "pg_catalog.english",
    "max_parallel_workers_per_gather": "2", "row_security": "on", "geqo": "on", "from_collapse_limit": "8",
    "join_collapse_limit": "8", "port": "5432", "listen_addresses": "localhost", "data_directory":
    "/var/lib/postgresql/16/main", "wal_level": "replica", "fsync": "on", "synchronous_commit": "on",
    "idle_in_transaction_session_timeout": "0", "temp_buffers": "8MB", "plan_cache_mode": "auto",
}
_BOOL_GUCS = {k for k, v in GUCS.items() if v in ("on", "off")}


class Result:
    def __init__(self, tag=None, names=None, types=None, rows=None, notices=None):
        self.tag = tag
        self.names = names
        self.types = types
        self.rows = rows
        self.notices = notices or []
        self.copy = None          # ("out", text) or ("in", stmt)
        self.stmt = None

    @property
    def has_rows(self):
        return self.names is not None


class Role:
    def __init__(self, name, superuser=False, createdb=False):
        self.name, self.superuser, self.createdb = name, superuser, createdb


class Cluster:
    """One PostgreSQL 16 server: databases and roles."""

    def __init__(self):
        self.databases: dict[str, Database] = {}
        self.roles = {"postgres": Role("postgres", True, True), "ada": Role("ada", False, True)}
        for name in ("postgres", "template0", "template1"):
            self.databases[name] = Database(name, "postgres")
        self.txid = 740
        self.sessions = []

    def create_database(self, name, owner):
        if name in self.databases:
            raise PgError(f'database "{name}" already exists', "42P04")
        db = Database(name, owner)
        self.databases[name] = db
        return db

    def drop_database(self, name, if_exists, session, notices):
        if name not in self.databases:
            if if_exists:
                notices.append(("NOTICE", f'database "{name}" does not exist, skipping', None, None))
                return
            raise PgError(f'database "{name}" does not exist', "3D000")
        if session is not None and session.dbname == name:
            raise PgError("cannot drop the currently open database", "55006")
        if name in ("template0", "template1"):
            raise PgError(f'cannot drop a template database', "55006")
        del self.databases[name]

    def connect(self, dbname, user="ada", clock=None, seed=0):
        if dbname not in self.databases:
            raise PgError(f'database "{dbname}" does not exist', "3D000", severity="FATAL")
        if user not in self.roles:
            raise PgError(f'role "{user}" does not exist', "28000", severity="FATAL")
        s = Session(self, dbname, user, clock, seed)
        return s


class Txn:
    def __init__(self, db: Database):
        self.db = db
        self.savepoints = []        # [(name, Database copy)]
        self.failed = False
        self.local_settings = {}


class Session:
    def __init__(self, cluster: Cluster, dbname: str, user: str, clock=None, seed=0):
        self.cluster = cluster
        self.dbname = dbname
        self.user = user
        self.settings = dict(GUCS)
        self.settings["session_authorization"] = user
        self.settings["is_superuser"] = "on" if cluster.roles[user].superuser else "off"
        self.txn: Txn | None = None
        self.temp_schema = None
        self.rng = random.Random(seed)
        self.last_seq = {}
        self.last_seq_any = None
        self.prepared = {}
        self.clock = clock           # () -> microseconds since the epoch
        self.txn_now = self.clock_now()
        self.stmt_now = self.txn_now
        self.pid = 4000 + len(cluster.sessions)
        cluster.sessions.append(self)

    # ---- state
    @property
    def search_path(self):
        return [s.strip() for s in self.settings["search_path"].split(",") if s.strip()]

    @property
    def status(self):
        if self.txn is None:
            return "I"
        return "E" if self.txn.failed else "T"

    def db(self) -> Database:
        if self.txn is not None:
            return self.txn.db
        return self.cluster.databases[self.dbname]

    def clock_now(self):
        if self.clock is None:
            return dt.datetime(2024, 6, 1, 12, 0, tzinfo=T.UTC)
        us = self.clock()
        return T.EPOCH_TZ + dt.timedelta(microseconds=us)

    def ensure_temp(self, db):
        name = f"pg_temp_{3 + len(self.cluster.sessions)}"
        if self.temp_schema is None:
            self.temp_schema = name
        if self.temp_schema not in db.schemas:
            db.schemas[self.temp_schema] = Schema(self.temp_schema, self.user)
        return self.temp_schema

    def txid(self):
        self.cluster.txid += 1
        return self.cluster.txid

    def lastval(self):
        if self.last_seq_any is None:
            raise PgError("lastval is not yet defined in this session", "55000")
        return self.last_seq_any

    def apply_settings(self):
        tz = self.settings.get("timezone", "Etc/UTC")
        T.CTX.timezone = tz
        T.CTX.datestyle = self.settings.get("datestyle", "ISO, MDY")

    def close(self):
        if self.txn is not None:
            self.txn = None
        db = self.cluster.databases.get(self.dbname)
        if db is not None and self.temp_schema and self.temp_schema in db.schemas:
            del db.schemas[self.temp_schema]
        if self in self.cluster.sessions:
            self.cluster.sessions.remove(self)

    # ---- running SQL
    def execute(self, sql: str, params=()):
        """Run every statement in `sql` (a simple query). -> [Result]. A failing statement raises PgError after
        the earlier ones have run; its position is relative to `sql`. Several statements outside a transaction
        block share one implicit transaction."""
        with decimal.localcontext(T.D):
            self.apply_settings()
            stmts = Parser(sql).statements()
            if not stmts:
                return [Result(tag="")]
            if len(stmts) > 1 and self.txn is None and not any(isinstance(s, A.Transaction) for s in stmts):
                return self._implicit_block(stmts, sql, params)
            out = []
            for st in stmts:
                out.append(self.run(st, params))
            return out

    def _implicit_block(self, stmts, sql, params):
        self.txn = Txn(self.cluster.databases[self.dbname].copy())
        self.txn_now = self.stmt_now = self.clock_now()
        out = []
        try:
            for st in stmts:
                out.append(self.run(st, params, implicit=True))
        except PgError:
            self.txn = None
            raise
        self.cluster.databases[self.dbname] = self.txn.db
        self.txn = None
        return out

    def run(self, st, params=(), implicit=False) -> Result:
        self.stmt_now = self.clock_now()
        if self.txn is None:
            self.txn_now = self.stmt_now
        _now_holder[0] = self.txn_now
        if isinstance(st, A.Transaction):
            return self.transaction(st)
        if self.txn is not None and self.txn.failed:
            raise PgError("current transaction is aborted, commands ignored until end of transaction block", "25P02")
        if self.txn is not None:
            try:
                return self.dispatch(st, self.txn.db, params)
            except PgError:
                if not implicit:
                    self.txn.failed = True
                raise
        if _read_only(st):
            return self.dispatch(st, self.cluster.databases[self.dbname], params)
        if isinstance(st, (A.CreateDatabase, A.Drop)) and getattr(st, "kind", "database") == "database" or \
                isinstance(st, A.CreateDatabase):
            return self.dispatch(st, self.cluster.databases[self.dbname], params)
        work = self.cluster.databases[self.dbname].copy()
        res = self.dispatch(st, work, params)
        if self.dbname in self.cluster.databases:
            self.cluster.databases[self.dbname] = work
        return res

    # ---- transactions
    def transaction(self, st: A.Transaction) -> Result:
        k = st.kind
        notices = []
        if k == "begin":
            if self.txn is not None:
                notices.append(("WARNING", "there is already a transaction in progress", None, None))
                return Result("START TRANSACTION" if st.word == "start" else "BEGIN", notices=notices)
            self.txn = Txn(self.cluster.databases[self.dbname].copy())
            self.txn_now = self.clock_now()
            return Result("START TRANSACTION" if st.word == "start" else "BEGIN")
        if k == "commit":
            tag = "COMMIT"
            if self.txn is None:
                notices.append(("WARNING", "there is no transaction in progress", None, None))
                return Result(tag, notices=notices)
            if self.txn.failed:
                self.txn = None
                self._end_txn()
                return Result("ROLLBACK")
            db = self.txn.db
            self._on_commit(db)
            self.cluster.databases[self.dbname] = db
            self.txn = None
            self._end_txn()
            if st.chain:
                self.txn = Txn(self.cluster.databases[self.dbname].copy())
            return Result(tag)
        if k == "rollback":
            if self.txn is None:
                notices.append(("WARNING", "there is no transaction in progress", None, None))
                return Result("ROLLBACK", notices=notices)
            self.txn = None
            self._end_txn()
            if st.chain:
                self.txn = Txn(self.cluster.databases[self.dbname].copy())
            return Result("ROLLBACK")
        if k == "savepoint":
            if self.txn is None:
                raise PgError("SAVEPOINT can only be used in transaction blocks", "25P01")
            if self.txn.failed:
                raise PgError("current transaction is aborted, commands ignored until end of transaction block",
                              "25P02")
            self.txn.savepoints.append((st.name, self.txn.db.copy()))
            return Result("SAVEPOINT")
        if k == "release":
            if self.txn is None:
                raise PgError("RELEASE SAVEPOINT can only be used in transaction blocks", "25P01")
            if self.txn.failed:
                raise PgError("current transaction is aborted, commands ignored until end of transaction block",
                              "25P02")
            idx = self._find_savepoint(st.name)
            del self.txn.savepoints[idx:]
            return Result("RELEASE")
        if k == "rollback_to":
            if self.txn is None:
                raise PgError("ROLLBACK TO SAVEPOINT can only be used in transaction blocks", "25P01")
            idx = self._find_savepoint(st.name)
            name, snap = self.txn.savepoints[idx]
            del self.txn.savepoints[idx + 1:]
            self.txn.db = snap.copy()
            self.txn.failed = False
            return Result("ROLLBACK")
        raise Unsupported(f"{k.upper()}")

    def _find_savepoint(self, name):
        for i in range(len(self.txn.savepoints) - 1, -1, -1):
            if self.txn.savepoints[i][0] == name:
                return i
        if self.txn.failed:
            pass
        raise PgError(f'savepoint "{name}" does not exist', "3B001")

    def _end_txn(self):
        for k in list(self.settings):
            if k.startswith("__local__"):
                pass
        local = getattr(self, "_local_saved", None)
        if local:
            self.settings.update(local)
            self._local_saved = None

    def _on_commit(self, db):
        if self.temp_schema and self.temp_schema in db.schemas:
            s = db.schemas[self.temp_schema]
            for name, rel in list(s.rels.items()):
                if isinstance(rel, Table) and rel.on_commit == "delete":
                    rel.rows = []
                elif isinstance(rel, Table) and rel.on_commit == "drop":
                    del s.rels[name]

    # ---- dispatch
    def dispatch(self, st, db: Database, params=()) -> Result:
        from .executor import Ctx, run_query
        from .ddl import DDL
        name = type(st).__name__
        if isinstance(st, (A.Select, A.SetOp, A.Values)):
            return self.select(st, db, params)
        if isinstance(st, (A.Insert, A.Update, A.Delete, A.Merge)):
            return self.modify(st, db, params)
        if isinstance(st, A.Explain):
            from .planner import explain
            lines, notices = explain(self, db, st)
            r = Result("EXPLAIN", ["QUERY PLAN"], [T.TEXT], [(ln,) for ln in lines])
            r.notices = notices
            return r
        if isinstance(st, A.VarSet):
            return self.set_var(st)
        if isinstance(st, A.VarShow):
            return self.show_var(st)
        if isinstance(st, A.Copy):
            return self.copy(st, db)
        if isinstance(st, A.Prepare):
            if st.name in self.prepared:
                raise PgError(f'prepared statement "{st.name}" already exists', "42P05")
            an = Analyzer(self, db)
            types = [an.resolve_type(t) for t in st.types]
            self.prepared[st.name] = (st.query, types)
            return Result("PREPARE")
        if isinstance(st, A.Execute):
            hit = self.prepared.get(st.name)
            if hit is None:
                raise PgError(f'prepared statement "{st.name}" does not exist', "26000")
            q, types = hit
            if len(st.params) != len(types):
                raise PgError(f'wrong number of parameters for prepared statement "{st.name}"', "42601",
                              detail=f"Expected {len(types)} parameters but got {len(st.params)}.")
            an = Analyzer(self, db)
            from .analyze import Scope, coerce_to
            from .executor import Ctx
            vals = []
            for e, t in zip(st.params, types):
                be = coerce_to(an.expr(e, Scope(None, self, db)), t, "a")
                vals.append(be.compile()(None, Ctx(self, db)))
            r = self.dispatch(q, db, vals) if not isinstance(q, (A.Select, A.SetOp, A.Values, A.Insert, A.Update,
                                                                  A.Delete, A.Merge)) else \
                (self.select(q, db, vals, types) if isinstance(q, (A.Select, A.SetOp, A.Values)) else
                 self.modify(q, db, vals, types))
            return r
        if isinstance(st, A.Deallocate):
            if st.name is None:
                self.prepared.clear()
            elif st.name not in self.prepared:
                raise PgError(f'prepared statement "{st.name}" does not exist', "26000")
            else:
                del self.prepared[st.name]
            return Result("DEALLOCATE ALL" if st.name is None else "DEALLOCATE")
        if isinstance(st, A.Noop):
            return Result(st.tag)
        if isinstance(st, A.Grant):
            return Result("REVOKE" if getattr(st, "revoke", False) else "GRANT")
        ddl = DDL(self.cluster, self, db)
        tag, res = self.ddl(st, ddl, db)
        r = Result(tag)
        r.notices = ddl.notices
        return r

    def ddl(self, st, ddl, db):
        if isinstance(st, A.CreateTable):
            res = ddl.create_table(st)
            if isinstance(res, tuple):
                return f"{res[0]} {res[1]}", res
            return "CREATE TABLE" if st.as_query is None else "SELECT 0", res
        if isinstance(st, A.CreateIndex):
            return "CREATE INDEX", ddl.create_index(st)
        if isinstance(st, A.CreateView):
            res = ddl.create_view(st)
            if isinstance(res, tuple):
                return f"{res[0]} {res[1]}", res
            return ("CREATE MATERIALIZED VIEW" if st.materialized else "CREATE VIEW"), res
        if isinstance(st, A.RefreshMatView):
            ddl.refresh(st)
            return "REFRESH MATERIALIZED VIEW", None
        if isinstance(st, A.Drop):
            ddl.drop(st)
            return "DROP " + st.kind.upper(), None
        if isinstance(st, A.AlterTable):
            ddl.alter_table(st)
            return "ALTER TABLE", None
        if isinstance(st, A.AlterSimple):
            ddl.alter_simple(st)
            return "ALTER " + st.kind.upper(), None
        if isinstance(st, A.Truncate):
            ddl.truncate(st)
            return "TRUNCATE TABLE", None
        if isinstance(st, A.CreateSchema):
            ddl.create_schema(st)
            return "CREATE SCHEMA", None
        if isinstance(st, A.CreateSequence):
            ddl.create_sequence(st)
            return "CREATE SEQUENCE", None
        if isinstance(st, A.CreateType):
            ddl.create_type(st)
            return "CREATE TYPE", None
        if isinstance(st, A.CreateExtension):
            if st.name in db.extensions:
                if st.if_not_exists:
                    ddl.notice(f'extension "{st.name}" already exists, skipping')
                    return "CREATE EXTENSION", None
                raise PgError(f'extension "{st.name}" already exists', "42710")
            if st.name not in ("pgcrypto", "uuid-ossp", "pg_trgm", "hstore", "citext", "tablefunc",
                               "pg_stat_statements", "btree_gin", "btree_gist", "fuzzystrmatch", "unaccent"):
                raise PgError(f'extension "{st.name}" is not available', "0A000",
                              detail=f'Could not open extension control file '
                                     f'"/usr/share/postgresql/16/extension/{st.name}.control": No such file or '
                                     'directory.',
                              hint="The extension must first be installed on the system where PostgreSQL is "
                                   "running.")
            db.extensions.add(st.name)
            return "CREATE EXTENSION", None
        if isinstance(st, A.CreateDatabase):
            if self.txn is not None:
                raise PgError("CREATE DATABASE cannot run inside a transaction block", "25001")
            role = self.cluster.roles.get(self.user)
            if role is not None and not (role.superuser or role.createdb):
                raise PgError("permission denied to create database", "42501")
            self.cluster.create_database(st.name, self.user)
            return "CREATE DATABASE", None
        if isinstance(st, A.Comment):
            ddl.comment(st)
            return "COMMENT", None
        if isinstance(st, A.Analyze):
            ddl.analyze(st)
            return "VACUUM" if st.vacuum else "ANALYZE", None
        raise Unsupported(type(st).__name__)

    # ---- queries
    def select(self, st, db, params=(), param_types=None) -> Result:
        from .executor import Ctx, run_query
        an = Analyzer(self, db, params, param_types or [T.guess_type(p) if p is not None else T.UNKNOWN
                                                        for p in params])
        q = an.query(st, None)
        c = Ctx(self, db, params)
        c.outer.append(None)
        rows = run_query(q, c)
        if q.into is not None:
            rv, temp = q.into
            from .ddl import DDL
            ddl = DDL(self.cluster, self, db)
            schema = db.creation_schema(rv.schema, self, temp)
            if db.find_rel(rv.name, schema, None) is not None:
                raise PgError(f'relation "{rv.name}" already exists', "42P07")
            from .catalog import Column
            t = Table(rv.name, schema, [Column(n, ty if ty.name != "unknown" else T.TEXT)
                                        for n, ty in zip(q.names[:q.ntargets], q.types[:q.ntargets])])
            t.rows = [tuple(r) for r in rows]
            t.oid = db.new_oid()
            db.schemas[schema].rels[rv.name] = t
            return Result(f"SELECT {len(rows)}")
        types = [t if t.name != "unknown" else T.TEXT for t in q.types[:q.ntargets]]
        r = Result(f"SELECT {len(rows)}", list(q.names[:q.ntargets]), types, rows)
        r.query = q
        return r

    def modify(self, st, db, params=(), param_types=None) -> Result:
        from .dml import analyze_modify, run_modify
        from .executor import Ctx, cte_rows
        an = Analyzer(self, db, params, param_types or [T.guess_type(p) if p is not None else T.UNKNOWN
                                                        for p in params])
        m = analyze_modify(an, st, None)
        c = Ctx(self, db, params)
        c.outer.append(None)
        for cte in getattr(m, "ctes", []):
            if cte.modifying:
                cte_rows(cte, c)
        rows = run_modify(m, c)
        tag = {"insert": f"INSERT 0 {m.count}", "update": f"UPDATE {m.count}", "delete": f"DELETE {m.count}",
               "merge": f"MERGE {m.count}"}[m.kind]
        if m.returning:
            return Result(tag, list(m.ret_names), list(m.ret_types), rows)
        return Result(tag)

    # ---- settings
    def set_var(self, st: A.VarSet) -> Result:
        name = st.name.lower()
        if name in ("transaction", "characteristics", "role"):
            return Result("SET")
        if st.reset:
            if name == "all":
                for k, v in GUCS.items():
                    self.settings[k] = v
                return Result("RESET")
            self._check_guc(name)
            self.settings[name] = GUCS.get(name, "")
            return Result("RESET")
        self._check_guc(name)
        if not st.values:
            value = GUCS.get(name, "")
        elif name == "search_path":
            value = ", ".join(('"$user"' if v == "$user" else v) for v in st.values)
        else:
            value = ", ".join(st.values)
        if name == "timezone":
            if value.lower() in ("local", "default"):
                value = GUCS["timezone"]
            try:
                T.zone_offset(value)
            except PgError:
                raise PgError(f'invalid value for parameter "TimeZone": "{value}"', "22023") from None
            value = _canon_tz(value)
        if name in _BOOL_GUCS:
            v = value.lower()
            if v in ("on", "true", "yes", "1", "t", "y"):
                value = "on"
            elif v in ("off", "false", "no", "0", "f", "n"):
                value = "off"
            else:
                raise PgError(f'parameter "{name}" requires a Boolean value', "22023")
        if name == "statement_timeout" or name.endswith("_timeout"):
            if not value.replace("ms", "").replace("s", "").strip().isdigit():
                raise PgError(f'invalid value for parameter "{name}": "{value}"', "22023")
        if name in ("server_version", "server_version_num", "server_encoding", "is_superuser", "max_connections",
                    "shared_buffers", "port", "data_directory", "wal_level", "listen_addresses"):
            raise PgError(f'parameter "{name}" cannot be changed' if name.startswith("server") or name in (
                "is_superuser",) else f'parameter "{name}" cannot be changed without restarting the server',
                "55P02")
        if st.local:
            if self.txn is None:
                return Result("SET", notices=[("WARNING", "SET LOCAL can only be used in transaction blocks",
                                               None, None)])
            saved = getattr(self, "_local_saved", None) or {}
            saved.setdefault(name, self.settings.get(name))
            self._local_saved = saved
        self.settings[name] = value
        return Result("SET")

    def _check_guc(self, name):
        if name not in GUCS and "." not in name:
            raise PgError(f'unrecognized configuration parameter "{name}"', "42704")

    def show_var(self, st: A.VarShow) -> Result:
        name = st.name.lower()
        if name == "all":
            rows = [(k, v, "") for k, v in sorted(self.settings.items())]
            return Result("SHOW", ["name", "setting", "description"], [T.TEXT, T.TEXT, T.TEXT], rows)
        if name not in self.settings:
            if "." in name:
                raise PgError(f'unrecognized configuration parameter "{st.name}"', "42704")
            raise PgError(f'unrecognized configuration parameter "{st.name}"', "42704")
        shown = {"timezone": "TimeZone", "datestyle": "DateStyle", "intervalstyle": "IntervalStyle"}.get(name, name)
        return Result("SHOW", [shown], [T.TEXT], [(self.settings[name],)])

    # ---- COPY
    def copy(self, st: A.Copy, db) -> Result:
        from .copyio import copy_out, copy_in_prepare
        if st.target is not None:
            role = self.cluster.roles.get(self.user)
            if role is None or not role.superuser:
                if st.to:
                    raise PgError("must be superuser or have privileges of the pg_write_server_files role to COPY "
                                  "to a file", "42501", hint="Anyone can COPY to stdout or from stdin. psql's \\copy "
                                                            "command also works for anyone.")
                raise PgError("must be superuser or have privileges of the pg_read_server_files role to COPY "
                              "from a file", "42501", hint="Anyone can COPY to stdout or from stdin. psql's \\copy "
                                                          "command also works for anyone.")
            raise Unsupported("server-side COPY with a file")
        if st.to:
            text, n = copy_out(self, db, st)
            r = Result(f"COPY {n}")
            r.copy = ("out", text)
            return r
        r = Result("COPY")
        r.copy = ("in", copy_in_prepare(self, db, st))
        return r

    def copy_in(self, prepared, data: str) -> Result:
        """Finish COPY ... FROM STDIN with the data psql sends."""
        from .copyio import copy_in_finish
        with decimal.localcontext(T.D):
            self.apply_settings()
            if self.txn is not None:
                try:
                    n = copy_in_finish(self, self.txn.db, prepared, data)
                except PgError:
                    self.txn.failed = True
                    raise
                return Result(f"COPY {n}")
            work = self.cluster.databases[self.dbname].copy()
            n = copy_in_finish(self, work, prepared, data)
            self.cluster.databases[self.dbname] = work
            return Result(f"COPY {n}")


def _canon_tz(v):
    from ..bashlite.timeutil import ZONES
    for k in ZONES:
        if k.lower() == v.lower():
            return k
    if v.upper() in ("UTC", "GMT"):
        return v.upper()
    return v


def _read_only(st) -> bool:
    if isinstance(st, (A.Select, A.SetOp, A.Values)):
        if getattr(st, "into", None) is not None:
            return False
        w = getattr(st, "with_", None)
        if w is not None and any(not isinstance(c.query, (A.Select, A.SetOp, A.Values)) for c in w.ctes):
            return False
        return True
    if isinstance(st, (A.VarSet, A.VarShow, A.Prepare, A.Deallocate, A.Noop, A.Grant)):
        return True
    if isinstance(st, A.Explain):
        return not st.options.get("analyze") or isinstance(st.stmt, (A.Select, A.SetOp, A.Values))
    if isinstance(st, A.Copy):
        return st.to
    return False

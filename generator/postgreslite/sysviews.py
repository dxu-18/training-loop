"""A few system catalogs and information_schema views, computed from the catalog."""

from __future__ import annotations

from . import types as T
from .catalog import Table, View, Index, Sequence


def system_view(an, rv):
    schema = rv.schema
    name = rv.name
    db = an.db
    if schema not in (None, "pg_catalog", "information_schema"):
        return None
    if name == "tables" and schema == "information_schema":
        rows = []
        for s in db.schemas.values():
            for r in s.rels.values():
                if isinstance(r, Table):
                    rows.append((db.name, s.name, r.name, "LOCAL TEMPORARY" if r.persistence == "t" else
                                 "BASE TABLE"))
                elif isinstance(r, View) and not r.materialized:
                    rows.append((db.name, s.name, r.name, "VIEW"))
        for t in ("pg_class", "pg_attribute", "pg_namespace", "pg_type", "pg_index", "pg_constraint"):
            rows.append((db.name, "pg_catalog", t, "BASE TABLE"))
        return (["table_catalog", "table_schema", "table_name", "table_type"], [T.NAME] * 3 + [T.TEXT], rows)
    if name == "columns" and schema == "information_schema":
        rows = []
        for s in db.schemas.values():
            for r in s.rels.values():
                if isinstance(r, (Table, View)):
                    for i, c in enumerate(r.columns, 1):
                        rows.append((db.name, s.name, r.name, c.name, i, c.default_sql,
                                     "NO" if c.not_null else "YES", _sql_type(c.type),
                                     c.type.mod if c.type.name in ("varchar", "bpchar") else None,
                                     c.type.mod[0] if c.type.name == "numeric" and c.type.mod else None,
                                     c.type.mod[1] if c.type.name == "numeric" and c.type.mod else None))
        return (["table_catalog", "table_schema", "table_name", "column_name", "ordinal_position", "column_default",
                 "is_nullable", "data_type", "character_maximum_length", "numeric_precision", "numeric_scale"],
                [T.NAME, T.NAME, T.NAME, T.NAME, T.INT4, T.TEXT, T.TEXT, T.TEXT, T.INT4, T.INT4, T.INT4], rows)
    if name == "schemata" and schema == "information_schema":
        rows = [(db.name, s.name, s.owner) for s in db.schemas.values()]
        return (["catalog_name", "schema_name", "schema_owner"], [T.NAME] * 3, rows)
    if schema == "information_schema":
        return None
    if name == "pg_tables":
        rows = [(s.name, r.name, db.owner, None, any(isinstance(x, Index) and x.table == r.name
                                                      for x in s.rels.values()), False, False, False)
                for s in db.schemas.values() for r in s.rels.values() if isinstance(r, Table)]
        return (["schemaname", "tablename", "tableowner", "tablespace", "hasindexes", "hasrules", "hastriggers",
                 "rowsecurity"], [T.NAME, T.NAME, T.NAME, T.NAME, T.BOOL, T.BOOL, T.BOOL, T.BOOL], rows)
    if name == "pg_views":
        rows = [(s.name, r.name, db.owner, r.sql) for s in db.schemas.values() for r in s.rels.values()
                if isinstance(r, View) and not r.materialized]
        return (["schemaname", "viewname", "viewowner", "definition"], [T.NAME, T.NAME, T.NAME, T.TEXT], rows)
    if name == "pg_matviews":
        rows = [(s.name, r.name, db.owner, None, False, r.populated, r.sql) for s in db.schemas.values()
                for r in s.rels.values() if isinstance(r, View) and r.materialized]
        return (["schemaname", "matviewname", "matviewowner", "tablespace", "hasindexes", "ispopulated",
                 "definition"], [T.NAME, T.NAME, T.NAME, T.NAME, T.BOOL, T.BOOL, T.TEXT], rows)
    if name == "pg_indexes":
        from .describe import index_def
        rows = [(s.name, r.table, r.name, None, index_def(db, r)) for s in db.schemas.values()
                for r in s.rels.values() if isinstance(r, Index)]
        return (["schemaname", "tablename", "indexname", "tablespace", "indexdef"],
                [T.NAME, T.NAME, T.NAME, T.NAME, T.TEXT], rows)
    if name == "pg_sequences":
        rows = [(s.name, r.name, db.owner, r.type.format(), r.start, r.minv, r.maxv, r.increment, r.cycle, r.cache,
                 r.last if r.is_called else None) for s in db.schemas.values() for r in s.rels.values()
                if isinstance(r, Sequence)]
        return (["schemaname", "sequencename", "sequenceowner", "data_type", "start_value", "min_value",
                 "max_value", "increment_by", "cycle", "cache_size", "last_value"],
                [T.NAME, T.NAME, T.NAME, T.TEXT, T.INT8, T.INT8, T.INT8, T.INT8, T.BOOL, T.INT8, T.INT8], rows)
    if name == "pg_stat_user_tables":
        rows = [(r.oid, s.name, r.name, len(r.rows)) for s in db.schemas.values() for r in s.rels.values()
                if isinstance(r, Table)]
        return (["relid", "schemaname", "relname", "n_live_tup"], [T.OID, T.NAME, T.NAME, T.INT8], rows)
    if name == "pg_database":
        rows = [(n, d.owner) for n, d in an.session.cluster.databases.items()]
        return (["datname", "datdba"], [T.NAME, T.NAME], rows)
    if name == "pg_roles" or name == "pg_user":
        rows = [(r.name, r.superuser, r.createdb) for r in an.session.cluster.roles.values()]
        return (["rolname", "rolsuper", "rolcreatedb"], [T.NAME, T.BOOL, T.BOOL], rows)
    if name == "pg_settings":
        rows = [(k, v) for k, v in sorted(an.session.settings.items())]
        return (["name", "setting"], [T.TEXT, T.TEXT], rows)
    return None


def _sql_type(t):
    if t.is_array:
        return "ARRAY"
    return {"int4": "integer", "int8": "bigint", "int2": "smallint", "float8": "double precision",
            "float4": "real", "bool": "boolean", "varchar": "character varying", "bpchar": "character",
            "timestamp": "timestamp without time zone", "timestamptz": "timestamp with time zone",
            "time": "time without time zone"}.get(t.name, t.display)

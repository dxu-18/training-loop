"""Set-returning functions: generate_series, unnest, jsonb_array_elements(_text), jsonb_each(_text),
jsonb_object_keys, regexp_matches, regexp_split_to_table, string_to_table, jsonb_to_record(set)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from . import types as T
from .errors import PgError
from .types import Json, JNULL

for _sig in (("jsonb_to_record", "jsonb", "record", False), ("jsonb_to_recordset", "jsonb", "record", True),
             ("json_to_record", "json", "record", False), ("json_to_recordset", "json", "record", True),
             ("generate_subscripts", "anyarray, integer", "integer", True),
             ("string_to_table", "text, text", "text", True)):
    if _sig[0] not in T.PROCS:
        T.add_proc(_sig[0], _sig[1], _sig[2], retset=_sig[3] or _sig[0].endswith("record"))

LIMIT = 2_000_000


def make_srf(name: str, argtypes: list, ret: T.Type, proc):
    """-> (impl(args, ctx) -> [tuple], (column names, column types))."""
    if proc.out_names:
        cols = (list(proc.out_names), list(proc.out_types))
    else:
        cols = ([name], [ret])
    fn = _IMPLS.get(name)
    if fn is None:
        raise PgError(f"function {name} is not supported by postgres-lite", "0A000")
    return (lambda args, c: fn(args, argtypes, ret, c)), cols


def _gen_series(args, types, ret, c):
    if any(a is None for a in args):
        return []
    start, stop = args[0], args[1]
    step = args[2] if len(args) > 2 else None
    out = []
    if ret.name in ("timestamp", "timestamptz"):
        from .funcs import ts_add
        if step is None:
            raise PgError("function generate_series requires a step", "42883")
        if step.key() == 0:
            raise PgError("step size cannot equal zero", "22023")
        cur = start
        neg = step.key() < 0
        while (cur <= stop) if not neg else (cur >= stop):
            out.append((cur,))
            if len(out) > LIMIT:
                raise PgError("postgres-lite stopped generate_series after 2,000,000 rows", "54000")
            cur = ts_add(cur, step)
        return out
    if step is None:
        step = 1 if ret.name != "numeric" else Decimal(1)
    if step == 0:
        raise PgError("step size cannot equal zero", "22023")
    if ret.name == "numeric":
        for a in (start, stop, step):
            if a.is_nan():
                raise PgError("start value cannot be NaN" if a is start else
                              "stop value cannot be NaN" if a is stop else "step size cannot be NaN", "22023")
    n = (stop - start) // step + 1 if (stop - start) * step >= 0 else 0
    if n > LIMIT:
        raise PgError("postgres-lite stopped generate_series after 2,000,000 rows", "54000")
    cur = start
    if step > 0:
        while cur <= stop:
            out.append((cur,))
            cur += step
    else:
        while cur >= stop:
            out.append((cur,))
            cur += step
    return out


def _unnest(args, types, ret, c):
    if args[0] is None:
        return []
    return [(v,) for v in T.array_flat(args[0])]


def _jae(args, types, ret, c, text=False):
    j = args[0]
    if j is None:
        return []
    v = j.v
    if not isinstance(v, list):
        raise PgError("cannot extract elements from a scalar" if not isinstance(v, dict) else
                      "cannot extract elements from an object", "22023")
    binary = j.binary
    out = []
    for x in v:
        if text:
            from .funcs import j_text
            out.append((j_text(x),))
        else:
            out.append((Json(x, True) if binary else Json(x, False, T.jsonb_text(x)),))
    return out


def _jeach(args, types, ret, c, text=False):
    j = args[0]
    if j is None:
        return []
    v = j.v
    if not isinstance(v, dict):
        raise PgError("cannot call jsonb_each on a non-object" if not text else
                      "cannot call jsonb_each_text on a non-object", "22023")
    from .funcs import j_text
    keys = sorted(v, key=T.jkey_order) if j.binary else list(v)
    out = []
    for k in keys:
        x = v[k]
        out.append((k, j_text(x) if text else (Json(x, True) if j.binary else Json(x, False, T.jsonb_text(x)))))
    return out


def _jkeys(args, types, ret, c):
    j = args[0]
    if j is None:
        return []
    if not isinstance(j.v, dict):
        raise PgError("cannot call jsonb_object_keys on a scalar" if not isinstance(j.v, list) else
                      "cannot call jsonb_object_keys on an array", "22023")
    return [(k,) for k in (sorted(j.v, key=T.jkey_order) if j.binary else j.v)]


def _regexp_matches(args, types, ret, c):
    if args[0] is None or args[1] is None:
        return []
    from .funcs import pg_regex, regex_flags
    flags = regex_flags(args[2] if len(args) > 2 and args[2] is not None else "")
    rx = pg_regex(args[1], flags)
    out = []
    for m in rx.finditer(args[0]):
        out.append((list(m.groups()) if m.re.groups else [m.group(0)],))
        if "g" not in flags:
            break
    return out


def _regexp_split(args, types, ret, c):
    if args[0] is None or args[1] is None:
        return []
    from .funcs import regex_split
    return [(x,) for x in regex_split(args[0], args[1], args[2] if len(args) > 2 and args[2] else "")]


def _string_to_table(args, types, ret, c):
    s, d = args[0], args[1]
    if s is None:
        return []
    if s == "":
        return []
    parts = list(s) if d is None else ([s] if d == "" else s.split(d))
    return [(p,) for p in parts]


def _subscripts(args, types, ret, c):
    arr, dim = args
    if arr is None or dim is None:
        return []
    dims = T.array_dims(arr)
    if dim < 1 or dim > len(dims):
        return []
    return [(i,) for i in range(1, dims[dim - 1] + 1)]


def _to_record(args, types, ret, c, many=False):
    j = args[0]
    if j is None:
        return []
    v = j.v
    if many:
        if not isinstance(v, list):
            raise PgError("cannot call jsonb_to_recordset on a non-array", "22023")
        return [(x,) for x in v]
    if not isinstance(v, dict):
        raise PgError("cannot call jsonb_to_record on a non-object", "22023")
    return [(v,)]


_IMPLS = {
    "generate_series": _gen_series,
    "unnest": _unnest,
    "jsonb_array_elements": _jae,
    "json_array_elements": _jae,
    "jsonb_array_elements_text": lambda a, t, r, c: _jae(a, t, r, c, True),
    "json_array_elements_text": lambda a, t, r, c: _jae(a, t, r, c, True),
    "jsonb_each": _jeach,
    "json_each": _jeach,
    "jsonb_each_text": lambda a, t, r, c: _jeach(a, t, r, c, True),
    "json_each_text": lambda a, t, r, c: _jeach(a, t, r, c, True),
    "jsonb_object_keys": _jkeys,
    "json_object_keys": _jkeys,
    "regexp_matches": _regexp_matches,
    "regexp_split_to_table": _regexp_split,
    "string_to_table": _string_to_table,
    "generate_subscripts": _subscripts,
    "jsonb_to_record": _to_record,
    "json_to_record": _to_record,
    "jsonb_to_recordset": lambda a, t, r, c: _to_record(a, t, r, c, True),
    "json_to_recordset": lambda a, t, r, c: _to_record(a, t, r, c, True),
}

"""PostgreSQL 16 data types: the type catalog, values in Python, text input/output, casts and typmods.

Values: None = NULL; bool; int (int2/int4/int8); Decimal (numeric: the exponent is the display scale); float
(float4 values are rounded to single precision); str (text, varchar, bpchar, name, uuid, enum labels); date;
datetime (timestamp naive, timestamptz aware in UTC); time; Interval; Json (json / jsonb); list (arrays, nested for
more dimensions); tuple (records); bytes (bytea).

The catalog (pg_catalog.txt, dumped from PostgreSQL 16.2) lists types, operators, functions and casts with their
exact signatures, so overload resolution (coerce.py) picks what the real server picks.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json as _json
import math
import os
import re
import struct
from decimal import Decimal

from .errors import PgError

D = decimal.Context(prec=1100, Emax=999999, Emin=-999999, rounding=decimal.ROUND_HALF_UP,
                    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
# the engine runs inside decimal.localcontext(D) (server.py), so other code's Decimal context is left alone

UTC = dt.timezone.utc


class Settings:
    """The session settings that change how values read and print."""
    timezone = "UTC"
    datestyle = "ISO, MDY"
    intervalstyle = "postgres"
    extra_float_digits = 1


CTX = Settings()


# ---------------------------------------------------------------- types

class Type:
    __slots__ = ("name", "oid", "cat", "pref", "display", "elem", "_array", "labels", "mod", "base", "length")

    def __init__(self, name, oid, cat, pref, display, elem=None, labels=None, length=-1):
        self.name, self.oid, self.cat, self.pref, self.display = name, oid, cat, pref, display
        self.elem = elem
        self._array = None
        self.labels = labels
        self.mod = None
        self.base = self
        self.length = length

    def __eq__(self, other):
        return isinstance(other, Type) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.display

    @property
    def is_array(self):
        return self.elem is not None and self.cat == "A"

    def array(self) -> "Type":
        b = self.base
        if b._array is None:
            a = Type("_" + b.name, 0, "A", False, b.display + "[]", elem=b)
            a.oid = _ARRAY_OIDS.get(b.name, 0)
            b._array = a
            TYPES[a.name] = a
        return b._array

    def with_mod(self, mod) -> "Type":
        if mod is None or mod == self.mod:
            return self.base if mod is None else self
        t = Type(self.name, self.oid, self.cat, self.pref, self.display, self.elem, self.labels, self.length)
        t._array = None
        t.mod = mod
        t.base = self.base
        return t

    def format(self) -> str:
        """format_type: integer, numeric(10,2), character varying(200), timestamp(3) without time zone ..."""
        if self.is_array:
            el = self.elem if self.mod is None else self.elem.with_mod(self.mod)
            return el.format() + "[]"
        m = self.mod
        if m is None:
            return self.display
        n = self.name
        if n == "numeric":
            p, s = m
            return f"numeric({p},{s})"
        if n in ("varchar", "bpchar", "bit", "varbit"):
            return f"{self.display}({m})"
        if n in ("timestamp", "timestamptz", "time", "timetz"):
            head, _, tail = self.display.partition(" ")
            return f"{head}({m}) {tail}"
        if n == "interval":
            return f"interval({m})" if isinstance(m, int) else f"interval {m}"
        return self.display


TYPES: dict[str, Type] = {}
BY_DISPLAY: dict[str, Type] = {}
CASTS: dict[tuple, tuple] = {}            # (src name, dst name) -> (context i/a/e, method f/b/i)
OPS: dict[str, list] = {}                 # op -> [(left Type | None, right Type, result Type)]
PROCS: dict[str, list] = {}               # name -> [Proc]
_ARRAY_OIDS = {"bool": 1000, "int2": 1005, "int4": 1007, "int8": 1016, "text": 1009, "varchar": 1015,
               "bpchar": 1014, "numeric": 1231, "float4": 1021, "float8": 1022, "date": 1182, "timestamp": 1115,
               "timestamptz": 1185, "interval": 1187, "json": 199, "jsonb": 3807, "uuid": 2951, "time": 1183,
               "name": 1003, "bytea": 1001}


class Proc:
    __slots__ = ("name", "args", "ret", "retset", "kind", "variadic", "ordered", "nargs_direct", "out_names",
                 "out_types", "argnames", "defaults")

    def __init__(self, name, args, ret, retset, kind, variadic, ordered=False, nargs_direct=0, out=None,
                 argnames=None):
        self.name, self.args, self.ret, self.retset, self.kind = name, args, ret, retset, kind
        self.variadic = variadic
        self.ordered = ordered
        self.nargs_direct = nargs_direct
        self.out_names = [n for n, _ in out] if out else None
        self.out_types = [t for _, t in out] if out else None
        self.argnames = argnames
        self.defaults = [None] * len(args)

    def __repr__(self):
        return f"{self.name}({', '.join(map(str, self.args))})"


def lookup_display(s: str) -> Type:
    s = s.strip()
    if s.endswith("[]"):
        return lookup_display(s[:-2]).array()
    t = BY_DISPLAY.get(s) or TYPES.get(s.strip('"'))
    if t is None:
        raise KeyError(s)
    return t


def _load():
    path = os.path.join(os.path.dirname(__file__), "pg_catalog.txt")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines:
        if line.startswith("type|"):
            _, oid, name, cat, pref, length, display, _elem, _arr = line.split("|")
            t = Type(name, int(oid), cat, pref == "t", display, length=int(length) if length != "-1" else -1)
            TYPES[name] = t
            BY_DISPLAY[display] = t
    # the pseudo-type "any" is spelled with quotes in signatures
    BY_DISPLAY['"any"'] = TYPES["any"]
    for line in lines:
        if line.startswith("cast|"):
            _, src, dst, ctx, meth = line.split("|")
            try:
                CASTS[(lookup_display(src).name, lookup_display(dst).name)] = (ctx, meth)
            except KeyError:
                pass
        elif line.startswith("op|"):
            parts = line[3:].split("|")
            op = "|".join(parts[:-4])
            left, right, res, _fn = parts[-4:]
            try:
                lt = None if left == "-" else lookup_display(left)
                OPS.setdefault(op, []).append((lt, lookup_display(right), lookup_display(res)))
            except KeyError:
                pass
        elif line.startswith("proc|"):
            _, name, args, ret, retset, kind, variadic = line.split("|")
            try:
                _add_proc(name, args, ret, retset == "t", kind, variadic)
            except KeyError:
                pass


def _add_proc(name, args, ret, retset, kind, variadic):
    ordered = "ORDER BY " in args
    direct = 0
    if ordered:
        a, _, b = args.partition("ORDER BY ")
        a = a.strip().rstrip(",")
        direct = len([x for x in a.split(",") if x.strip()])
        args = (a + ", " if a.strip() else "") + b
    ins, outs, names, defaults = [], [], [], []
    for a in [x.strip() for x in args.split(",") if x.strip()]:
        if a.startswith("OUT "):
            _, n, t = a.split(" ", 2)
            outs.append((n, lookup_display(t)))
            continue
        a = a.replace("VARIADIC ", "")
        if a.startswith("IN "):
            a = a[3:]
        dflt = None
        if " DEFAULT " in a:
            a, _, dflt = a.partition(" DEFAULT ")
        # "name type" (named parameters) vs a type name of several words
        try:
            ins.append(lookup_display(a))
            names.append(None)
        except KeyError:
            n, _, t = a.partition(" ")
            ins.append(lookup_display(t))
            names.append(n)
        defaults.append(dflt)
    rt = lookup_display(ret)
    vt = None if variadic == "-" else lookup_display(variadic)
    p = Proc(name, ins, rt, retset, kind, vt, ordered, direct, outs or None, names if any(names) else None)
    p.defaults = defaults
    PROCS.setdefault(name, []).append(p)


def add_proc(name: str, args: str, ret: str, retset=False, kind="f", variadic="-"):
    """Functions the dump doesn't list (added by hand, same signature syntax)."""
    _add_proc(name, args, ret, retset, kind, variadic)


_load()

BOOL, INT2, INT4, INT8 = TYPES["bool"], TYPES["int2"], TYPES["int4"], TYPES["int8"]
NUMERIC, FLOAT4, FLOAT8 = TYPES["numeric"], TYPES["float4"], TYPES["float8"]
TEXT, VARCHAR, BPCHAR, NAME = TYPES["text"], TYPES["varchar"], TYPES["bpchar"], TYPES["name"]
DATE, TIME, TIMETZ = TYPES["date"], TYPES["time"], TYPES["timetz"]
TIMESTAMP, TIMESTAMPTZ, INTERVAL = TYPES["timestamp"], TYPES["timestamptz"], TYPES["interval"]
JSON, JSONB, UUID, BYTEA = TYPES["json"], TYPES["jsonb"], TYPES["uuid"], TYPES["bytea"]
UNKNOWN, RECORD, VOID, ANY = TYPES["unknown"], TYPES["record"], TYPES["void"], TYPES["any"]
OID, REGCLASS, CHAR = TYPES["oid"], TYPES["regclass"], TYPES["char"]
ANYELEMENT, ANYARRAY, ANYNONARRAY = TYPES["anyelement"], TYPES["anyarray"], TYPES["anynonarray"]
ANYCOMPATIBLE, ANYCOMPATIBLEARRAY = TYPES["anycompatible"], TYPES["anycompatiblearray"]
ANYCOMPATIBLENONARRAY, ANYENUM = TYPES["anycompatiblenonarray"], TYPES["anyenum"]
TEXT_ARRAY = TEXT.array()
INTS = (INT2, INT4, INT8)
STRINGS = ("text", "varchar", "bpchar", "name", "unknown")
SUPPORTED = {"bool", "int2", "int4", "int8", "numeric", "float4", "float8", "text", "varchar", "bpchar", "name",
             "date", "time", "timestamp", "timestamptz", "interval", "json", "jsonb", "uuid", "bytea", "oid",
             "char", "regclass", "timetz"}
POLY = {"anyelement", "anyarray", "anynonarray", "anyenum", "anycompatible", "anycompatiblearray",
        "anycompatiblenonarray", "any", "anyrange", "anymultirange", "anycompatiblerange",
        "anycompatiblemultirange"}

# type names as written in SQL -> internal names
ALIASES = {
    "int": "int4", "integer": "int4", "int4": "int4", "smallint": "int2", "int2": "int2", "bigint": "int8",
    "int8": "int8", "real": "float4", "float4": "float4", "float8": "float8", "float": "float8",
    "double precision": "float8", "numeric": "numeric", "decimal": "numeric", "boolean": "bool", "bool": "bool",
    "text": "text", "varchar": "varchar", "character varying": "varchar", "char": "bpchar", "character": "bpchar",
    "bpchar": "bpchar", "name": "name", "date": "date", "time": "time", "timetz": "timetz", "timestamp": "timestamp",
    "timestamptz": "timestamptz", "interval": "interval", "json": "json", "jsonb": "jsonb", "uuid": "uuid",
    "bytea": "bytea", "oid": "oid", "regclass": "regclass", "record": "record", "void": "void",
    "serial": "int4", "serial4": "int4", "bigserial": "int8", "serial8": "int8", "smallserial": "int2",
    "serial2": "int2", "unknown": "unknown",
}


def resolve_name(name: str) -> Type | None:
    n = ALIASES.get(name, name)
    t = TYPES.get(n)
    return t


# ---------------------------------------------------------------- special values

class Interval:
    __slots__ = ("months", "days", "micros")

    def __init__(self, months=0, days=0, micros=0):
        self.months, self.days, self.micros = int(months), int(days), int(micros)

    def key(self):
        """interval_cmp_value: a month is 30 days, a day 24 hours."""
        return (self.months * 30 + self.days) * 86_400_000_000 + self.micros

    def __eq__(self, other):
        return isinstance(other, Interval) and self.key() == other.key()

    def __lt__(self, other):
        return self.key() < other.key()

    def __hash__(self):
        return hash(self.key())

    def __neg__(self):
        return Interval(-self.months, -self.days, -self.micros)

    def __add__(self, o):
        return Interval(self.months + o.months, self.days + o.days, self.micros + o.micros)

    def __sub__(self, o):
        return Interval(self.months - o.months, self.days - o.days, self.micros - o.micros)

    def __repr__(self):
        return f"Interval({self.months}, {self.days}, {self.micros})"

    def total_seconds(self) -> float:
        """EXTRACT(EPOCH FROM interval): 365.25 days a year, 30 days a month."""
        years, mons = int(self.months / 12), self.months - int(self.months / 12) * 12
        return (years * 365.25 * 86400 + mons * 30 * 86400 + self.days * 86400) + self.micros / 1e6


class _JNull:
    __slots__ = ()

    def __repr__(self):
        return "JNULL"


JNULL = _JNull()


class Json:
    """A json or jsonb value: .v is the parsed value (dict / list / str / Decimal / bool / JNULL);
    json keeps its text as written (.text), jsonb is normalised."""
    __slots__ = ("v", "binary", "text")

    def __init__(self, v, binary=True, text=None):
        self.v, self.binary, self.text = v, binary, text

    def __eq__(self, other):
        return isinstance(other, Json) and jsonb_cmp_key(self.v) == jsonb_cmp_key(other.v)

    def __hash__(self):
        return hash(_freeze(self.v))

    def __repr__(self):
        return f"Json({json_out(self)})"


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, Decimal):
        return ("n", v.normalize() if v == v else "nan")
    return v


class PosInf:
    """date / timestamp infinity."""


INFINITY_DATE = "infinity"
NEG_INFINITY_DATE = "-infinity"


# ---------------------------------------------------------------- numeric helpers

def num_scale(d: Decimal) -> int:
    if not d.is_finite():
        return 0
    e = d.as_tuple().exponent
    return max(0, -e)


def num_round(d: Decimal, scale: int) -> Decimal:
    if not d.is_finite():
        return d
    return d.quantize(Decimal(1).scaleb(-scale), rounding=decimal.ROUND_HALF_UP)


def num_weight_first(d: Decimal) -> tuple[int, int]:
    """(weight, first digit) of |d| in base 10000, as numeric.c stores it."""
    if d == 0:
        return 0, 0
    x = abs(d)
    w = x.adjusted() // 4
    first = int(x.scaleb(-4 * w))
    if first >= 10000:       # adjusted() rounding edge
        w += 1
        first = int(x.scaleb(-4 * w))
    return w, first


def div_scale(a: Decimal, b: Decimal) -> int:
    """select_div_scale (numeric.c)."""
    w1, f1 = num_weight_first(a)
    w2, f2 = num_weight_first(b)
    q = w1 - w2
    if f1 <= f2:
        q -= 1
    r = 16 - q * 4
    r = max(r, num_scale(a), num_scale(b), 0)
    return min(r, 1000)


def num_out(d: Decimal) -> str:
    if d.is_nan():
        return "NaN"
    if d.is_infinite():
        return "Infinity" if d > 0 else "-Infinity"
    s = format(d, "f")
    if s.startswith("-") and set(s[1:]) <= set("0."):
        s = s[1:]
    return s


_NUM_RE = re.compile(r"\s*([+-]?)(\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE]([+-]?\d+))?\s*\Z")
_INT_RE = re.compile(r"\s*([+-]?)(0[xX](?:_?[0-9a-fA-F])+|0[oO](?:_?[0-7])+|0[bB](?:_?[01])+|\d(?:_?\d)*)\s*\Z")


def num_in(s: str, tname: str = "numeric") -> Decimal:
    m = _NUM_RE.match(s)
    if not m:
        t = s.strip().lower()
        if t in ("nan",):
            return Decimal("NaN")
        if t in ("infinity", "+infinity", "inf", "+inf"):
            return Decimal("Infinity")
        if t in ("-infinity", "-inf"):
            return Decimal("-Infinity")
        raise invalid_input(tname, s)
    txt = (m.group(1) + m.group(2)).replace("_", "")
    if m.group(3):
        txt += "e" + m.group(3)
    d = Decimal(txt)
    if d.as_tuple().exponent > 0:
        d = d.quantize(Decimal(1))
    return d


def invalid_input(tname: str, s: str) -> PgError:
    disp = TYPES[tname].display if tname in TYPES else tname
    return PgError(f'invalid input syntax for type {disp}: "{s}"', "22P02")


INT_LIMITS = {"int2": (-32768, 32767, "smallint"), "int4": (-2147483648, 2147483647, "integer"),
              "int8": (-9223372036854775808, 9223372036854775807, "bigint")}


def int_check(v: int, tname: str) -> int:
    lo, hi, disp = INT_LIMITS[tname]
    if v < lo or v > hi:
        raise PgError(f"{disp} out of range", "22003")
    return v


def int_in(s: str, tname: str) -> int:
    m = _INT_RE.match(s)
    if not m:
        raise invalid_input(tname, s)
    body = m.group(2).replace("_", "")
    if body[:2].lower() in ("0x", "0o", "0b"):
        v = int(body[2:], {"x": 16, "o": 8, "b": 2}[body[1].lower()])
    else:
        v = int(body)
    if m.group(1) == "-":
        v = -v
    lo, hi, disp = INT_LIMITS[tname]
    if v < lo or v > hi:
        raise PgError(f'value "{s}" is out of range for type {disp}', "22003")
    return v


def float_in(s: str, tname: str) -> float:
    t = s.strip()
    low = t.lower()
    specials = {"nan": math.nan, "infinity": math.inf, "+infinity": math.inf, "-infinity": -math.inf,
                "inf": math.inf, "+inf": math.inf, "-inf": -math.inf}
    if low in specials:
        return specials[low]
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", t):
        raise invalid_input(tname, s)
    v = float(t)
    if math.isinf(v):
        raise PgError(f'"{t}" is out of range for type {TYPES[tname].display}', "22003")
    if tname == "float4":
        v = to_float4(v, s)
    return v


def to_float4(v: float, src=None) -> float:
    if math.isnan(v) or math.isinf(v):
        return v
    try:
        r = struct.unpack("f", struct.pack("f", v))[0]
    except OverflowError:
        if src is not None:
            raise PgError(f'"{str(src).strip()}" is out of range for type real', "22003") from None
        raise PgError("value out of range: overflow", "22003") from None
    return r


def float8_out(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 0:
        return "-0" if math.copysign(1, v) < 0 else "0"
    r = repr(v)
    return _shortest_format(r, 15)


def float4_out(v: float) -> str:
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 0:
        return "-0" if math.copysign(1, v) < 0 else "0"
    for p in range(1, 10):
        s = f"{v:.{p}g}"
        if struct.unpack("f", struct.pack("f", float(s)))[0] == v:
            break
    return _shortest_format(s, 6)


def _shortest_format(r: str, maxexp: int) -> str:
    """Ryu's shortest digits, printed like PostgreSQL: fixed notation for -4 <= exponent < maxexp."""
    neg = r.startswith("-")
    if neg:
        r = r[1:]
    mant, _, exp = r.lower().partition("e")
    e = int(exp) if exp else 0
    if "." in mant:
        ip, fp = mant.split(".")
    else:
        ip, fp = mant, ""
    digits = (ip + fp).lstrip("0")
    # decimal exponent of the first significant digit
    if ip.strip("0"):
        e10 = len(ip.lstrip("0")) - 1 + e
    else:
        e10 = -(len(fp) - len(fp.lstrip("0"))) - 1 + e
    digits = digits.rstrip("0") or "0"
    if -4 <= e10 < maxexp:
        if e10 >= 0:
            if len(digits) <= e10 + 1:
                out = digits + "0" * (e10 + 1 - len(digits))
            else:
                out = digits[:e10 + 1] + "." + digits[e10 + 1:]
        else:
            out = "0." + "0" * (-e10 - 1) + digits
    else:
        m = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        out = f"{m}e{'-' if e10 < 0 else '+'}{abs(e10):02d}"
    return ("-" if neg else "") + out


def float_to_numeric(v: float, tname: str = "float8") -> Decimal:
    if math.isnan(v):
        return Decimal("NaN")
    if math.isinf(v):
        return Decimal("Infinity") if v > 0 else Decimal("-Infinity")
    digits = 15 if tname == "float8" else 6
    return num_in(f"{v:.{digits}g}")


# ---------------------------------------------------------------- dates and times

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FULL_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
               "november", "december"]
DAYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
EPOCH = dt.datetime(1970, 1, 1)
EPOCH_TZ = dt.datetime(1970, 1, 1, tzinfo=UTC)
POS_INF_TS = dt.datetime.max
NEG_INF_TS = dt.datetime.min
POS_INF_DATE = dt.date.max
NEG_INF_DATE = dt.date.min
POS_INF_TSTZ = dt.datetime.max.replace(tzinfo=UTC)
NEG_INF_TSTZ = dt.datetime.min.replace(tzinfo=UTC)


def zone_offset(tz: str, when_utc: dt.datetime | None = None, local: dt.datetime | None = None) -> int:
    """Offset in seconds east of UTC for a zone name ('UTC', 'America/New_York', '+02', 'EST')."""
    from ..bashlite.timeutil import ZONES, zone_info
    name = tz.strip()
    m = re.fullmatch(r"([+-])(\d{1,2})(?::?(\d\d))?(?::(\d\d))?", name)
    if m:
        off = int(m.group(2)) * 3600 + int(m.group(3) or 0) * 60 + int(m.group(4) or 0)
        return off if m.group(1) == "+" else -off
    key = _zone_key(name)
    if key is None:
        abbr = _ABBREVS.get(name.lower())
        if abbr is not None:
            return abbr
        m = re.fullmatch(r"([A-Za-z]{3,5})([+-]?\d{1,2})", name)
        if m:   # POSIX: EST5 = 5 hours west
            return -int(m.group(2)) * 3600
        raise PgError(f'time zone "{tz}" not recognized', "22023")
    if when_utc is None and local is not None:
        off, _ = zone_info(key, local.replace(tzinfo=UTC))
        off2, _ = zone_info(key, local.replace(tzinfo=UTC) - dt.timedelta(seconds=off))
        return off2
    off, _ = zone_info(key, (when_utc or dt.datetime(2024, 1, 1)).replace(tzinfo=UTC))
    return off


def _zone_key(name: str):
    from ..bashlite.timeutil import ZONES
    for k in ZONES:
        if k.lower() == name.lower():
            return k
    return None


_ABBREVS = {"utc": 0, "gmt": 0, "z": 0, "zulu": 0, "est": -5 * 3600, "edt": -4 * 3600, "cst": -6 * 3600,
            "cdt": -5 * 3600, "mst": -7 * 3600, "mdt": -6 * 3600, "pst": -8 * 3600, "pdt": -7 * 3600,
            "cet": 3600, "cest": 7200, "bst": 3600, "jst": 9 * 3600, "ist": int(5.5 * 3600)}


def zone_abbrev(tz: str, when_utc: dt.datetime) -> str:
    from ..bashlite.timeutil import zone_info
    key = _zone_key(tz)
    if key is None:
        return tz
    return zone_info(key, when_utc.replace(tzinfo=UTC))[1]


def check_timezone(tz: str) -> str:
    zone_offset(tz)
    return tz


def to_local(ts: dt.datetime, tz: str | None = None) -> tuple[dt.datetime, int]:
    """A timestamptz (aware, UTC) as local wall time in the session's zone. -> (naive local, offset seconds)."""
    if ts in (POS_INF_TSTZ, NEG_INF_TSTZ):
        return ts.replace(tzinfo=None), 0
    tz = tz or CTX.timezone
    off = zone_offset(tz, when_utc=ts.replace(tzinfo=None))
    return (ts + dt.timedelta(seconds=off)).replace(tzinfo=None), off


def from_local(naive: dt.datetime, tz: str | None = None) -> dt.datetime:
    """Local wall time in a zone -> timestamptz."""
    if naive in (POS_INF_TS, NEG_INF_TS):
        return naive.replace(tzinfo=UTC)
    tz = tz or CTX.timezone
    off = zone_offset(tz, local=naive)
    return (naive - dt.timedelta(seconds=off)).replace(tzinfo=UTC)


def _frac(us: int) -> str:
    if not us:
        return ""
    return ("." + f"{us:06d}").rstrip("0")


def date_out(d: dt.date) -> str:
    if d == POS_INF_DATE:
        return "infinity"
    if d == NEG_INF_DATE:
        return "-infinity"
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def time_out(t: dt.time) -> str:
    return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}{_frac(t.microsecond)}"


def ts_out(ts: dt.datetime) -> str:
    if ts == POS_INF_TS:
        return "infinity"
    if ts == NEG_INF_TS:
        return "-infinity"
    return f"{date_out(ts.date())} {time_out(ts.time())}"


def tz_suffix(off: int) -> str:
    sign = "+" if off >= 0 else "-"
    off = abs(off)
    h, rem = divmod(off, 3600)
    m, s = divmod(rem, 60)
    out = f"{sign}{h:02d}"
    if m or s:
        out += f":{m:02d}"
    if s:
        out += f":{s:02d}"
    return out


def tstz_out(ts: dt.datetime) -> str:
    if ts == POS_INF_TSTZ:
        return "infinity"
    if ts == NEG_INF_TSTZ:
        return "-infinity"
    local, off = to_local(ts)
    return ts_out(local) + tz_suffix(off)


def interval_out(iv: Interval) -> str:
    """IntervalStyle postgres: 1 year 2 mons 3 days 04:05:06.5"""
    months, days, micros = iv.months, iv.days, iv.micros
    years = int(months / 12)
    mons = months - years * 12
    parts = []
    is_zero = True
    is_before = False

    def add(v, unit):
        nonlocal is_zero, is_before
        if v == 0:
            return
        sign = "+" if (not is_zero and is_before and v > 0) else ""
        parts.append(f"{sign}{v} {unit}{'s' if v != 1 else ''}")
        is_before = v < 0
        is_zero = False
    add(years, "year")
    add(mons, "mon")
    add(days, "day")
    if micros != 0 or is_zero:
        minus = micros < 0
        a = abs(micros)
        h, rem = divmod(a, 3_600_000_000)
        m, rem = divmod(rem, 60_000_000)
        s, us = divmod(rem, 1_000_000)
        sign = "-" if minus else ("+" if is_before else "")
        parts.append(f"{sign}{h:02d}:{m:02d}:{s:02d}{_frac(us)}")
    return " ".join(parts)


_TZ_TAIL = re.compile(r"(?:\s*(?:([+-]\d{1,2}(?::?\d\d)?(?::\d\d)?)|\s+([A-Za-z][A-Za-z_/+-]*[A-Za-z0-9]?)|(Z|z)))\s*$")


def _parse_time(s: str):
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2})(?:\.(\d*))?)?", s)
    if not m:
        m2 = re.fullmatch(r"(\d{2})(\d{2})(\d{2})?(?:\.(\d*))?", s)
        if not m2:
            return None
        m = m2
    h, mi = int(m.group(1)), int(m.group(2))
    sec = int(m.group(3) or 0)
    frac = m.group(4) or ""
    us = int(round(float("0." + frac) * 1_000_000)) if frac else 0
    return h, mi, sec, us


def _ymd(y, mo, d, s, tname):
    try:
        return dt.date(y, mo, d)
    except ValueError:
        raise PgError(f'date/time field value out of range: "{s}"', "22008") from None


def parse_datetime(s: str, tname: str):
    """Date / timestamp / timestamptz input (ISO, MDY DateStyle, month names, special words).
    -> (date | None, (h, m, s, us) | None, tz offset seconds | None, special | None)"""
    raw = s
    t = s.strip()
    low = t.lower()
    if low in ("infinity", "+infinity", "-infinity", "epoch", "now", "today", "tomorrow", "yesterday", "allballs"):
        return None, None, None, low.lstrip("+")
    if not t:
        raise invalid_input(tname, raw)
    tz = None
    # a trailing zone: +02, -05:30, Z, UTC, America/New_York
    words = t.replace(",", " ").split()
    tokens = []
    for w in words:
        if "T" in w and re.match(r"\d{4}-?\d\d-?\d\dT", w):
            a, b = w.split("T", 1)
            tokens += [a, b]
        else:
            tokens.append(w)
    date = None
    time = None
    ampm = None
    bc = False
    rest = []
    for tok in tokens:
        lt = tok.lower()
        if lt in ("am", "pm"):
            ampm = lt
            continue
        if lt == "bc":
            bc = True
            continue
        if lt == "ad":
            continue
        # time with zone glued: 10:00:00+02, 10:00Z
        m = re.fullmatch(r"(\d{1,2}:\d{1,2}(?::\d{1,2}(?:\.\d*)?)?)([+-]\d{1,2}(?::?\d\d)?(?::\d\d)?|[zZ])?", tok)
        if m and time is None:
            time = _parse_time(m.group(1))
            if m.group(2):
                tz = 0 if m.group(2) in ("z", "Z") else zone_offset(m.group(2))
            continue
        if time is not None and re.fullmatch(r"[+-]\d{1,2}(?::?\d\d)?", tok):
            tz = zone_offset(tok)
            continue
        if date is not None and (lt in _ABBREVS or _zone_key(tok) is not None or lt in ("z",)):
            tz = ("zone", tok)
            continue
        rest.append(tok)
    # the date part
    ds = rest
    if not ds:
        if time is not None and tname in ("time", "timetz"):
            return None, time, tz, None
        raise invalid_input(tname, raw)
    y = mo = d = None
    if len(ds) == 1:
        tok = ds[0]
        m = re.fullmatch(r"(\d{1,})-(\d{1,2})-(\d{1,2})", tok)
        if m and len(m.group(1)) >= 3:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        elif re.fullmatch(r"\d{8}", tok):
            y, mo, d = int(tok[:4]), int(tok[4:6]), int(tok[6:])
        elif re.fullmatch(r"\d{6}", tok):
            y, mo, d = 2000 + int(tok[:2]) if int(tok[:2]) < 70 else 1900 + int(tok[:2]), int(tok[2:4]), int(tok[4:])
        else:
            m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{1,4})", tok)
            if m:   # MDY
                mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if len(m.group(3)) <= 2:
                    y += 2000 if y < 70 else 1900
            else:
                m = re.fullmatch(r"(\d{1,4})[/.-]([A-Za-z]+)[/.-](\d{1,4})", tok)
                if m:
                    mon = _month_of(m.group(2))
                    a, b = m.group(1), m.group(3)
                    if mon is None:
                        raise invalid_input(tname, raw)
                    mo = mon
                    if len(a) >= 3:
                        y, d = int(a), int(b)
                    else:
                        d, y = int(a), int(b)
                else:
                    m = re.fullmatch(r"([A-Za-z]+)[/.-](\d{1,2})[/.-](\d{1,4})", tok)
                    if m and _month_of(m.group(1)):
                        mo, d, y = _month_of(m.group(1)), int(m.group(2)), int(m.group(3))
                    else:
                        raise invalid_input(tname, raw)
    else:
        # words: Jan 8 1999 / January 8, 1999 / 8 Jan 1999 / Mon Jan 8 1999
        nums = []
        for tok in ds:
            lt = tok.lower().rstrip(".")
            if lt[:3] in DAYS and not lt.isdigit() and _month_of(lt) is None:
                continue
            mon = _month_of(lt)
            if mon is not None and mo is None:
                mo = mon
            elif tok.isdigit():
                nums.append(tok)
            else:
                raise invalid_input(tname, raw)
        if mo is None or len(nums) != 2:
            raise invalid_input(tname, raw)
        a, b = nums
        if len(a) >= 3:
            y, d = int(a), int(b)
        else:
            d, y = int(a), int(b)
    if bc:
        raise PgError("BC dates are not supported by postgres-lite", "0A000")
    if mo < 1 or mo > 12 or d < 1 or d > 31:
        raise PgError(f'date/time field value out of range: "{raw}"', "22008",
                      hint='Perhaps you need a different "datestyle" setting.' if mo > 12 else None)
    date = _ymd(y, mo, d, raw, tname)
    if time is not None:
        h, mi, sec, us = time
        if ampm == "pm" and h < 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        if h > 24 or mi > 59 or sec > 60 or (h == 24 and (mi or sec or us)):
            raise PgError(f'date/time field value out of range: "{raw}"', "22008")
        time = (h, mi, sec, us)
    if isinstance(tz, tuple):
        tz = zone_offset(tz[1], local=dt.datetime.combine(date, dt.time()))
    return date, time, tz, None


def _month_of(w: str):
    w = w.lower().rstrip(".")
    if len(w) >= 3:
        for i, m in enumerate(FULL_MONTHS):
            if m.startswith(w):
                return i + 1
    return None


def _combine(date, time):
    if time is None:
        return dt.datetime(date.year, date.month, date.day)
    h, mi, sec, us = time
    base = dt.datetime(date.year, date.month, date.day)
    return base + dt.timedelta(hours=h, minutes=mi, seconds=sec, microseconds=us)


def _special(word, tname, now):
    """now / today / epoch ... as a value of tname. now: the transaction's start (timestamptz)."""
    if word == "infinity":
        return {"date": POS_INF_DATE, "timestamp": POS_INF_TS, "timestamptz": POS_INF_TSTZ}[tname]
    if word == "-infinity":
        return {"date": NEG_INF_DATE, "timestamp": NEG_INF_TS, "timestamptz": NEG_INF_TSTZ}[tname]
    if word == "epoch":
        return {"date": dt.date(1970, 1, 1), "timestamp": EPOCH, "timestamptz": EPOCH_TZ}[tname]
    if now is None:
        now = EPOCH_TZ
    local, _ = to_local(now)
    if word == "now":
        if tname == "timestamptz":
            return now
        if tname == "timestamp":
            return local
        return local.date()
    day = local.date() + dt.timedelta(days={"today": 0, "tomorrow": 1, "yesterday": -1}[word])
    if tname == "date":
        return day
    midnight = dt.datetime(day.year, day.month, day.day)
    return midnight if tname == "timestamp" else from_local(midnight)


def date_in(s: str, now=None) -> dt.date:
    date, time, tz, sp = parse_datetime(s, "date")
    if sp:
        if sp == "allballs":
            raise invalid_input("date", s)
        return _special(sp, "date", now)
    return date


def ts_in(s: str, now=None) -> dt.datetime:
    date, time, tz, sp = parse_datetime(s, "timestamp")
    if sp:
        if sp == "allballs":
            raise invalid_input("timestamp", s)
        return _special(sp, "timestamp", now)
    return _combine(date, time)


def tstz_in(s: str, now=None) -> dt.datetime:
    date, time, tz, sp = parse_datetime(s, "timestamptz")
    if sp:
        if sp == "allballs":
            raise invalid_input("timestamptz", s)
        return _special(sp, "timestamptz", now)
    naive = _combine(date, time)
    if tz is None:
        return from_local(naive)
    return (naive - dt.timedelta(seconds=tz)).replace(tzinfo=UTC)


def time_in(s: str) -> dt.time:
    t = s.strip().lower()
    if t == "allballs":
        return dt.time(0)
    if t == "now":
        return dt.time(0)
    toks = t.split()
    ampm = None
    if toks and toks[-1] in ("am", "pm"):
        ampm = toks.pop()
    if len(toks) != 1:
        # a timestamp: take its time
        date, time, tz, sp = parse_datetime(s, "time")
        if time is None:
            return dt.time(0)
        h, mi, sec, us = time
    else:
        tm = re.sub(r"([+-]\d{1,2}(:?\d\d)?|z)$", "", toks[0])
        p = _parse_time(tm)
        if p is None:
            raise invalid_input("time", s)
        h, mi, sec, us = p
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    if h > 24 or mi > 59 or sec > 60:
        raise PgError(f'date/time field value out of range: "{s}"', "22008")
    if h == 24:
        return dt.time(0)
    return dt.time(h, mi, sec, us)


_UNITS = {
    "microsecond": "us", "microseconds": "us", "us": "us", "usec": "us", "usecs": "us", "useconds": "us",
    "millisecond": "ms", "milliseconds": "ms", "ms": "ms", "msec": "ms", "msecs": "ms", "mseconds": "ms",
    "second": "s", "seconds": "s", "s": "s", "sec": "s", "secs": "s",
    "minute": "min", "minutes": "min", "m": "min", "min": "min", "mins": "min",
    "hour": "h", "hours": "h", "h": "h", "hr": "h", "hrs": "h",
    "day": "d", "days": "d", "d": "d",
    "week": "w", "weeks": "w", "w": "w",
    "month": "mon", "months": "mon", "mon": "mon", "mons": "mon",
    "year": "y", "years": "y", "y": "y", "yr": "y", "yrs": "y",
    "decade": "dec", "decades": "dec", "dec": "dec", "decs": "dec",
    "century": "c", "centuries": "c", "c": "c", "cent": "c",
    "millennium": "mil", "millennia": "mil", "millenniums": "mil", "mil": "mil", "mils": "mil",
}


def interval_in(s: str, fields: str | None = None) -> Interval:
    raw = s
    t = s.strip().lower()
    if not t:
        raise invalid_input("interval", raw)
    if t.startswith("p"):
        return _iso_interval(t, raw)
    months = 0
    days = 0
    micros = Decimal(0)
    ago = False
    toks = re.findall(r"[+-]?\d+(?:\.\d*)?(?::\d+(?:\.\d*)?)*|[+-]?\.\d+|[a-z]+|@|,", t)
    if "".join(toks).replace(" ", "") != re.sub(r"\s+", "", t):
        raise invalid_input("interval", raw)
    i = 0
    seen_any = False
    while i < len(toks):
        tok = toks[i]
        if tok in ("@", ","):
            i += 1
            continue
        if tok == "ago":
            ago = True
            i += 1
            continue
        if ":" in tok:
            neg = tok.startswith("-")
            parts = tok.lstrip("+-").split(":")
            try:
                if len(parts) == 2 and fields in ("minute to second",):
                    h, mi, sec = 0, int(parts[0]), Decimal(parts[1])
                else:
                    h = int(parts[0])
                    mi = int(parts[1]) if len(parts) > 1 else 0
                    sec = Decimal(parts[2]) if len(parts) > 2 and parts[2] else Decimal(0)
            except (ValueError, decimal.InvalidOperation):
                raise invalid_input("interval", raw) from None
            if mi > 59 or sec >= 60:
                raise PgError(f'interval field value out of range: "{raw}"', "22015")
            us = h * 3_600_000_000 + mi * 60_000_000 + sec * 1_000_000
            micros += -us if neg else us
            seen_any = True
            i += 1
            continue
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", tok):
            num = Decimal(tok)
            unit = None
            if i + 1 < len(toks) and toks[i + 1] in _UNITS:
                unit = _UNITS[toks[i + 1]]
                i += 2
            else:
                i += 1
                unit = {"year": "y", "month": "mon", "day": "d", "hour": "h", "minute": "min",
                        "second": "s", "year to month": "mon" if False else "y"}.get(fields or "", "s")
                if fields in ("day to hour",):
                    unit = "h"
                elif fields in ("day to minute", "hour to minute"):
                    unit = "min"
                elif fields and fields.endswith("second"):
                    unit = "s"
            months, days, micros = _add_unit(months, days, micros, num, unit)
            seen_any = True
            continue
        if tok in _UNITS:
            raise invalid_input("interval", raw)
        raise invalid_input("interval", raw)
    if not seen_any:
        raise invalid_input("interval", raw)
    iv = Interval(months, days, int(micros.to_integral_value(rounding=decimal.ROUND_HALF_EVEN)))
    if fields:
        iv = interval_fields(iv, fields)
    return -iv if ago else iv


def _add_unit(months, days, micros, num: Decimal, unit):
    if unit == "us":
        micros += num
    elif unit == "ms":
        micros += num * 1000
    elif unit == "s":
        micros += num * 1_000_000
    elif unit == "min":
        micros += num * 60_000_000
    elif unit == "h":
        micros += num * 3_600_000_000
    elif unit in ("d", "w"):
        n = num * (7 if unit == "w" else 1)
        whole = int(n)
        days += whole
        micros += (n - whole) * 86_400_000_000
    else:
        n = num * {"mon": 1, "y": 12, "dec": 120, "c": 1200, "mil": 12000}[unit]
        whole = int(n)
        months += whole
        frac = n - whole
        if frac:
            fd = frac * 30
            wd = int(fd)
            days += wd
            micros += (fd - wd) * 86_400_000_000
    return months, days, micros


def _iso_interval(t, raw):
    m = re.fullmatch(r"p(?:(-?\d+(?:\.\d+)?)y)?(?:(-?\d+(?:\.\d+)?)m)?(?:(-?\d+(?:\.\d+)?)w)?(?:(-?\d+(?:\.\d+)?)d)?"
                     r"(?:t(?:(-?\d+(?:\.\d+)?)h)?(?:(-?\d+(?:\.\d+)?)m)?(?:(-?\d+(?:\.\d+)?)s)?)?", t)
    if not m:
        raise invalid_input("interval", raw)
    months, days, micros = 0, 0, Decimal(0)
    for val, unit in zip(m.groups(), ("y", "mon", "w", "d", "h", "min", "s")):
        if val is not None:
            months, days, micros = _add_unit(months, days, micros, Decimal(val), unit)
    return Interval(months, days, int(micros))


def interval_fields(iv: Interval, fields: str) -> Interval:
    """INTERVAL YEAR / DAY TO HOUR ...: drop the fields below the last one."""
    last = fields.split()[-1]
    if last == "year":
        return Interval(int(iv.months / 12) * 12, 0, 0)
    if last == "month":
        return Interval(iv.months, 0, 0)
    if last == "day":
        return Interval(iv.months, iv.days, 0)
    unit = {"hour": 3_600_000_000, "minute": 60_000_000}.get(last)
    if unit:
        return Interval(iv.months, iv.days, int(iv.micros / unit) * unit)
    return iv


# ---------------------------------------------------------------- json

def json_parse(s: str, binary: bool) -> Json:
    tname = "jsonb" if binary else "json"
    try:
        if binary:
            v = _json.loads(s, parse_float=Decimal, parse_int=Decimal, parse_constant=_bad_const,
                            object_pairs_hook=_pairs)
        else:
            v = _json.loads(s, parse_float=Decimal, parse_int=Decimal, parse_constant=_bad_const,
                            object_pairs_hook=_pairs)
    except (ValueError, RecursionError) as e:
        detail = None
        if isinstance(e, _json.JSONDecodeError):
            bad = s[e.pos:].split()[0] if s[e.pos:].split() else ""
            if not s.strip() or e.pos >= len(s.rstrip()):
                detail = "The input string ended unexpectedly."
            else:
                tok = re.match(r"[A-Za-z0-9_.+-]+|.", s[e.pos:].lstrip())
                detail = f'Token "{tok.group(0) if tok else bad}" is invalid.'
        line = s.split("\n")[0]
        raise PgError(f"invalid input syntax for type {tname}", "22P02", detail=detail,
                      context=f"JSON data, line 1: {line[:40]}") from None
    v = _jnorm(v)
    return Json(v, binary, None if binary else s)


def _bad_const(name):
    raise ValueError(name)


def _pairs(pairs):
    d = {}
    for k, v in pairs:
        d.pop(k, None)
        d[k] = v
    return d


def _jnorm(v):
    if v is None:
        return JNULL
    if isinstance(v, dict):
        return {k: _jnorm(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jnorm(x) for x in v]
    if isinstance(v, Decimal):
        if v.as_tuple().exponent > 0:
            v = v.quantize(Decimal(1))
        return v
    return v


def jkey_order(k: str):
    b = k.encode("utf-8")
    return (len(b), b)


def json_out(j: Json) -> str:
    if not j.binary and j.text is not None:
        return j.text
    return jsonb_text(j.v)


def jsonb_text(v, indent=None, level=0) -> str:
    if v is JNULL:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, Decimal):
        return num_out(v)
    if isinstance(v, str):
        return json_quote(v)
    if isinstance(v, list):
        if indent is not None:
            if not v:
                return "[\n" + "    " * level + "]" if False else "[]"
            pad = "    " * (level + 1)
            return "[\n" + ",\n".join(pad + jsonb_text(x, indent, level + 1) for x in v) + "\n" + "    " * level + "]"
        return "[" + ", ".join(jsonb_text(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = sorted(v, key=jkey_order)
        if indent is not None:
            if not v:
                return "{}"
            pad = "    " * (level + 1)
            return "{\n" + ",\n".join(f"{pad}{json_quote(k)}: {jsonb_text(v[k], indent, level + 1)}" for k in keys) \
                + "\n" + "    " * level + "}"
        return "{" + ", ".join(f"{json_quote(k)}: {jsonb_text(v[k])}" for k in keys) + "}"
    if isinstance(v, (int, float)):
        return str(v)
    raise TypeError(v)


def json_quote(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def jsonb_type_rank(v) -> int:
    """jsonb ordering: Object > Array > Boolean > Number > String > Null."""
    if v is JNULL:
        return 0
    if isinstance(v, str):
        return 1
    if isinstance(v, Decimal):
        return 2
    if isinstance(v, bool):
        return 3
    if isinstance(v, list):
        return 4
    return 5


def jsonb_cmp_key(v):
    r = jsonb_type_rank(v)
    if r == 0:
        return (0,)
    if r == 1:
        return (1, v.encode("utf-8"))
    if r == 2:
        return (2, _NumKey(v))
    if r == 3:
        return (3, v)
    if r == 4:
        return (4, len(v), tuple(jsonb_cmp_key(x) for x in v))
    keys = sorted(v, key=jkey_order)
    return (5, len(v), tuple((jkey_order(k), jsonb_cmp_key(v[k])) for k in keys))


import functools


@functools.total_ordering
class _NumKey:
    """Numeric sort key: NaN sorts above everything."""
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def _k(self):
        v = self.v
        if isinstance(v, Decimal) and v.is_nan():
            return (2, 0)
        if isinstance(v, float) and math.isnan(v):
            return (2, 0)
        return (1, v)

    def __lt__(self, o):
        return self._k() < o._k()

    def __eq__(self, o):
        return self._k() == o._k()

    def __hash__(self):
        return hash(self._k())


def to_json_value(v, t: Type):
    """A SQL value as a JSON value (to_jsonb / json_build_object / jsonb_agg)."""
    if v is None:
        return JNULL
    n = t.name
    if isinstance(v, Json):
        return v.v
    if n == "bool":
        return v
    if n in ("int2", "int4", "int8"):
        return Decimal(v)
    if n == "numeric":
        if not v.is_finite():
            return num_out(v)
        return v
    if n in ("float4", "float8"):
        if math.isnan(v) or math.isinf(v):
            return float8_out(v)
        return num_in(float8_out(v) if n == "float8" else float4_out(v))
    if t.is_array:
        return [to_json_value(x, t.elem) for x in v]
    if n == "record" and isinstance(v, tuple):
        names = getattr(v, "_names", None) or [f"f{i + 1}" for i in range(len(v))]
        types = getattr(v, "_types", None) or [UNKNOWN] * len(v)
        return {nm: to_json_value(x, tt) for nm, x, tt in zip(names, v, types)}
    if n == "timestamp":
        return ts_out(v).replace(" ", "T")
    if n == "timestamptz":
        s = tstz_out(v)
        s = s.replace(" ", "T", 1)
        if re.search(r"[+-]\d\d$", s):
            s += ":00"
        return s
    if n == "date":
        return date_out(v)
    return output(v, t)


def json_text_sql(v, t: Type) -> str:
    """A SQL value as json text the way to_json / array_to_json write it: compact arrays and records."""
    if v is None:
        return "null"
    if isinstance(v, Json):
        return json_out(v) if not v.binary else jsonb_text(v.v)
    if t.is_array or isinstance(v, list):
        el = t.elem if t.is_array else UNKNOWN
        return "[" + ",".join(json_text_sql(x, el) for x in v) + "]"
    if isinstance(v, tuple):
        names = getattr(v, "_names", None) or [f"f{i + 1}" for i in range(len(v))]
        types = getattr(v, "_types", None) or [guess_type(x) for x in v]
        return "{" + ",".join(f"{json_quote(n)}:{json_text_sql(x, tt)}" for n, x, tt in zip(names, v, types)) + "}"
    return jsonb_text(to_json_value(v, t))


# ---------------------------------------------------------------- arrays

def array_out(v, elem: Type) -> str:
    if isinstance(v, list) and v and isinstance(v[0], list) and not elem.is_array:
        return "{" + ",".join(array_out(x, elem) for x in v) + "}"
    parts = []
    for x in v:
        if isinstance(x, list) and not elem.is_array:
            parts.append(array_out(x, elem))
            continue
        if x is None:
            parts.append("NULL")
            continue
        s = output(x, elem)
        if s == "" or s.upper() == "NULL" or any(c in s for c in '{},"\\ \t\n\r\f\v'):
            s = '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
        parts.append(s)
    return "{" + ",".join(parts) + "}"


def array_in(s: str, elem: Type, now=None):
    raw = s
    t = s.strip()
    if t.startswith("["):     # [1:3]={...} explicit bounds
        m = re.match(r"(\[\d+:\d+\])+=", t)
        if m:
            t = t[m.end():]
    if not t.startswith("{"):
        raise PgError(f'malformed array literal: "{raw}"', "22P02",
                      detail='Array value must start with "{" or dimension information.')
    pos = [0]

    def parse_level():
        assert t[pos[0]] == "{"
        pos[0] += 1
        items = []
        expect_item = True
        while True:
            while pos[0] < len(t) and t[pos[0]].isspace():
                pos[0] += 1
            if pos[0] >= len(t):
                raise PgError(f'malformed array literal: "{raw}"', "22P02", detail="Unexpected end of input.")
            c = t[pos[0]]
            if c == "}":
                pos[0] += 1
                return items
            if c == ",":
                pos[0] += 1
                continue
            if c == "{":
                items.append(parse_level())
                continue
            if c == '"':
                pos[0] += 1
                buf = []
                while pos[0] < len(t) and t[pos[0]] != '"':
                    if t[pos[0]] == "\\":
                        pos[0] += 1
                    buf.append(t[pos[0]])
                    pos[0] += 1
                pos[0] += 1
                items.append(input_value("".join(buf), elem, now=now))
                continue
            buf = []
            while pos[0] < len(t) and t[pos[0]] not in ",}":
                if t[pos[0]] == "\\":
                    pos[0] += 1
                buf.append(t[pos[0]])
                pos[0] += 1
            word = "".join(buf).strip()
            if word.upper() == "NULL":
                items.append(None)
            else:
                items.append(input_value(word, elem, now=now))
    out = parse_level()
    if t[pos[0]:].strip():
        raise PgError(f'malformed array literal: "{raw}"', "22P02",
                      detail="Junk after closing right brace.")
    return out


def array_flat(v):
    for x in v:
        if isinstance(x, list):
            yield from array_flat(x)
        else:
            yield x


def array_dims(v) -> list[int]:
    dims = []
    cur = v
    while isinstance(cur, list):
        dims.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return dims


# ---------------------------------------------------------------- input / output

def output(v, t: Type) -> str:
    """The text form of a value (what psql prints)."""
    if v is None:
        return ""
    n = t.name
    if t.is_array or (isinstance(v, list) and n not in ("json", "jsonb")):
        return array_out(v, t.elem if t.is_array else UNKNOWN)
    if n == "bool":
        return "t" if v else "f"
    if n in ("int2", "int4", "int8", "oid"):
        return str(v)
    if n == "numeric":
        return num_out(v)
    if n == "float8":
        return float8_out(v)
    if n == "float4":
        return float4_out(v)
    if n in ("text", "varchar", "name", "unknown", "uuid", "char", "regclass") or t.cat == "E":
        return str(v)
    if n == "bpchar":
        return v
    if n == "date":
        return date_out(v)
    if n == "timestamp":
        return ts_out(v)
    if n == "timestamptz":
        return tstz_out(v)
    if n == "time":
        return time_out(v)
    if n == "timetz":
        t0, off = v
        return time_out(t0) + tz_suffix(off)
    if n == "interval":
        return interval_out(v)
    if n in ("json", "jsonb"):
        return json_out(v)
    if n == "bytea":
        return "\\x" + v.hex()
    if n == "record" or isinstance(v, tuple):
        return record_out(v)
    if n == "void":
        return ""
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, Decimal):
        return num_out(v)
    if isinstance(v, float):
        return float8_out(v)
    if isinstance(v, dt.datetime):
        return tstz_out(v) if v.tzinfo else ts_out(v)
    if isinstance(v, dt.date):
        return date_out(v)
    if isinstance(v, Interval):
        return interval_out(v)
    if isinstance(v, Json):
        return json_out(v)
    return str(v)


def record_out(v: tuple) -> str:
    types = getattr(v, "_types", None) or [None] * len(v)
    parts = []
    for x, tt in zip(v, types):
        if x is None:
            parts.append("")
            continue
        s = output(x, tt) if tt is not None else output(x, guess_type(x))
        if s == "" or any(c in s for c in '(),"\\ \t\n'):
            s = '"' + s.replace("\\", "\\\\").replace('"', '""') + '"'
        parts.append(s)
    return "(" + ",".join(parts) + ")"


class Row(tuple):
    """A record value that remembers its column names and types (for output and to_jsonb)."""

    def __new__(cls, vals, names=None, types=None):
        r = super().__new__(cls, vals)
        r._names = names
        r._types = types
        return r


def guess_type(v) -> Type:
    if isinstance(v, bool):
        return BOOL
    if isinstance(v, int):
        return INT8
    if isinstance(v, Decimal):
        return NUMERIC
    if isinstance(v, float):
        return FLOAT8
    if isinstance(v, str):
        return TEXT
    if isinstance(v, dt.datetime):
        return TIMESTAMPTZ if v.tzinfo else TIMESTAMP
    if isinstance(v, dt.date):
        return DATE
    if isinstance(v, dt.time):
        return TIME
    if isinstance(v, Interval):
        return INTERVAL
    if isinstance(v, Json):
        return JSONB if v.binary else JSON
    if isinstance(v, list):
        return UNKNOWN.array() if False else TEXT_ARRAY
    if isinstance(v, tuple):
        return RECORD
    if isinstance(v, bytes):
        return BYTEA
    return TEXT


_BOOL_TRUE = ("t", "true", "y", "yes", "on", "1")
_BOOL_FALSE = ("f", "false", "n", "no", "off", "0")


def bool_in(s: str) -> bool:
    t = s.strip().lower()
    if t in ("o",):
        raise invalid_input("bool", s)
    if t and (any(w.startswith(t) for w in ("true", "yes")) or t in ("on", "1", "t", "y")):
        return True
    if t and (any(w.startswith(t) for w in ("false", "no")) or t in ("off", "of", "0", "f", "n")):
        return False
    raise invalid_input("bool", s)


def uuid_in(s: str) -> str:
    t = s.strip()
    h = t.strip("{}").replace("-", "").lower()
    if len(h) != 32 or not re.fullmatch(r"[0-9a-f]{32}", h):
        raise invalid_input("uuid", s)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def bytea_in(s: str) -> bytes:
    if s.startswith("\\x"):
        try:
            return bytes.fromhex(s[2:])
        except ValueError:
            raise invalid_input("bytea", s) from None
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if s[i + 1:i + 2] == "\\":
                out.append(92)
                i += 2
                continue
            if re.fullmatch(r"[0-3][0-7]{2}", s[i + 1:i + 4]):
                out.append(int(s[i + 1:i + 4], 8))
                i += 4
                continue
            raise invalid_input("bytea", s)
        out += c.encode("utf-8")
        i += 1
    return bytes(out)


def input_value(s: str, t: Type, mod=None, now=None, explicit=False):
    """Text -> a value of type t (type_in), with its typmod applied."""
    n = t.name
    if t.is_array:
        v = array_in(s, t.elem, now)
        if mod is not None:
            v = _map_array(v, lambda x: apply_typmod(x, t.elem, mod, explicit))
        return v
    if n == "bool":
        v = bool_in(s)
    elif n in ("int2", "int4", "int8"):
        v = int_in(s, n)
    elif n == "oid":
        v = int_in(s, "int8")
    elif n == "numeric":
        v = num_in(s)
    elif n in ("float4", "float8"):
        v = float_in(s, n)
    elif n in ("text", "varchar", "bpchar", "name", "unknown", "char"):
        v = s
        if n == "name":
            v = s[:63]
        if n == "char":
            v = s[:1]
    elif n == "date":
        v = date_in(s, now)
    elif n == "timestamp":
        v = ts_in(s, now)
    elif n == "timestamptz":
        v = tstz_in(s, now)
    elif n == "time":
        v = time_in(s)
    elif n == "timetz":
        v = (time_in(s), 0)
    elif n == "interval":
        v = interval_in(s, mod if isinstance(mod, str) else None)
    elif n == "json":
        v = json_parse(s, False)
    elif n == "jsonb":
        v = json_parse(s, True)
    elif n == "uuid":
        v = uuid_in(s)
    elif n == "bytea":
        v = bytea_in(s)
    elif t.cat == "E":
        if s not in t.labels:
            raise PgError(f'invalid input value for enum {t.display}: "{s}"', "22P02")
        v = s
    elif n == "record":
        raise PgError('input of anonymous composite types is not implemented', "0A000")
    elif n == "regclass":
        v = s
    else:
        raise PgError(f"type {t.display} is not supported by postgres-lite", "0A000")
    if mod is not None:
        v = apply_typmod(v, t, mod, explicit)
    return v


def _map_array(v, fn):
    return [_map_array(x, fn) if isinstance(x, list) else (None if x is None else fn(x)) for x in v]


def apply_typmod(v, t: Type, mod, explicit=False):
    """numeric(p, s), varchar(n), char(n), timestamp(p): round, check or pad a value for its column."""
    if v is None or mod is None:
        return v
    n = t.name
    if n == "numeric":
        p, s = mod
        if not v.is_finite():
            if v.is_nan():
                return v
            raise PgError("numeric field overflow", "22003",
                          detail=f"A field with precision {p}, scale {s} cannot hold an infinite value.")
        r = num_round(v, s)
        if r != 0 and r.adjusted() >= p - s:
            raise PgError("numeric field overflow", "22003",
                          detail=f"A field with precision {p}, scale {s} must round to an absolute value less "
                                 f"than {'10^' + str(p - s) if p - s != 0 else '1'}.")
        return r
    if n == "varchar":
        if len(v) > mod:
            if explicit or not v[mod:].strip(" "):
                return v[:mod]
            raise PgError(f"value too long for type character varying({mod})", "22001")
        return v
    if n == "bpchar":
        if len(v) > mod:
            if explicit or not v[mod:].strip(" "):
                return v[:mod]
            raise PgError(f"value too long for type character({mod})", "22001")
        return v.ljust(mod)
    if n in ("timestamp", "timestamptz") and isinstance(mod, int) and mod < 6:
        if v in (POS_INF_TS, NEG_INF_TS, POS_INF_TSTZ, NEG_INF_TSTZ):
            return v
        q = 10 ** (6 - mod)
        us = v.microsecond
        r = int((us + q // 2) // q * q)
        return v.replace(microsecond=0) + dt.timedelta(microseconds=r)
    if n == "time" and isinstance(mod, int) and mod < 6:
        q = 10 ** (6 - mod)
        us = v.microsecond
        r = int((us + q // 2) // q * q)
        base = dt.datetime.combine(dt.date(2000, 1, 1), v.replace(microsecond=0)) + dt.timedelta(microseconds=r)
        return base.time()
    if n == "interval":
        if isinstance(mod, str):
            return interval_fields(v, mod)
        if isinstance(mod, int) and mod < 6:
            q = 10 ** (6 - mod)
            return Interval(v.months, v.days, int(round(v.micros / q)) * q)
    return v


# ---------------------------------------------------------------- casts

def find_cast(src: Type, dst: Type):
    """-> (context i/a/e, method) or None. Includes the automatic I/O casts to and from string types."""
    if src == dst:
        return ("i", "b")
    c = CASTS.get((src.name, dst.name))
    if c is not None:
        return c
    if src.is_array and dst.is_array:
        ec = find_cast(src.elem, dst.elem)
        if ec is not None:
            return (ec[0], "f")
        return None
    if src.cat == "E" and dst.cat == "E":
        return None
    if dst.name in ("text", "varchar", "bpchar", "name") and src.name not in POLY:
        return ("a", "i")
    if src.name in ("text", "varchar", "bpchar", "name") and dst.name not in POLY:
        return ("e", "i")
    return None


def can_coerce(src: Type, dst: Type, ctx: str = "i") -> bool:
    """can_coerce_type for one argument: unknown goes anywhere; polymorphic targets accept their family."""
    if src == dst:
        return True
    if src.name == "unknown":
        return True
    if dst.name == "any":
        return True
    if dst.name in ("anyelement", "anycompatible"):
        return True
    if dst.name in ("anyarray", "anycompatiblearray"):
        return src.is_array
    if dst.name in ("anynonarray", "anycompatiblenonarray"):
        return not src.is_array
    if dst.name == "anyenum":
        return src.cat == "E"
    if dst.name == "record" and (src.name == "record" or src.cat == "C"):
        return True
    if dst.name in POLY:
        return False
    if src.name == "record" and dst.name == "record":
        return True
    c = find_cast(src, dst)
    if c is None:
        return False
    order = {"i": 0, "a": 1, "e": 2}
    return order[c[0]] <= order[ctx]


def cast_value(v, src: Type, dst: Type, explicit: bool = True, now=None):
    """Convert a value. Typmods are applied by the caller (apply_typmod)."""
    if v is None:
        return None
    s, d = src.name, dst.name
    if s == d and not src.is_array:
        return v
    if src.is_array and dst.is_array:
        return _map_array(v, lambda x: cast_value(x, src.elem, dst.elem, explicit, now))
    if d in ("text", "varchar", "bpchar", "name") or (d == "char" and s != "int4"):
        if s in ("bool",) and d in ("text", "varchar", "bpchar"):
            return "true" if v else "false"
        if s == "bpchar":
            return v.rstrip(" ")
        out = output(v, src)
        return out[:63] if d == "name" else out
    if s in ("text", "varchar", "bpchar", "name", "unknown", "char"):
        return input_value(v.rstrip(" ") if s == "bpchar" else v, dst, now=now, explicit=explicit)
    if d in ("int2", "int4", "int8"):
        if s in ("int2", "int4", "int8"):
            return int_check(v, d)
        if s == "numeric":
            if v.is_nan():
                raise PgError(f"cannot convert NaN to {TYPES[d].display}", "0A000")
            if v.is_infinite():
                raise PgError(f"cannot convert infinity to {TYPES[d].display}", "0A000")
            return int_check(int(v.to_integral_value(rounding=decimal.ROUND_HALF_UP)), d)
        if s in ("float4", "float8"):
            if math.isnan(v) or math.isinf(v):
                raise PgError(f"{TYPES[d].display} out of range", "22003")
            return int_check(int(round(v)), d)
        if s == "bool":
            return 1 if v else 0
        if s == "jsonb":
            return _jsonb_to(v, dst)
        if s == "oid":
            return v
    if d == "numeric":
        if s in ("int2", "int4", "int8"):
            return Decimal(v)
        if s in ("float4", "float8"):
            return float_to_numeric(v, s)
        if s == "jsonb":
            return _jsonb_to(v, dst)
    if d in ("float4", "float8"):
        if s in ("int2", "int4", "int8"):
            f = float(v)
        elif s == "numeric":
            f = float(v)
            if math.isinf(f) and v.is_finite():
                raise PgError(f'"{num_out(v)}" is out of range for type {dst.display}', "22003")
        elif s in ("float4", "float8"):
            f = v
        elif s == "jsonb":
            return _jsonb_to(v, dst)
        else:
            f = None
        if f is not None:
            return to_float4(f) if d == "float4" else f
    if d == "bool":
        if s in ("int4", "int2", "int8"):
            return v != 0
        if s == "jsonb":
            return _jsonb_to(v, dst)
    if d == "date":
        if s == "timestamp":
            return v.date() if v not in (POS_INF_TS, NEG_INF_TS) else (POS_INF_DATE if v == POS_INF_TS else NEG_INF_DATE)
        if s == "timestamptz":
            if v in (POS_INF_TSTZ, NEG_INF_TSTZ):
                return POS_INF_DATE if v == POS_INF_TSTZ else NEG_INF_DATE
            return to_local(v)[0].date()
    if d == "timestamp":
        if s == "date":
            if v in (POS_INF_DATE, NEG_INF_DATE):
                return POS_INF_TS if v == POS_INF_DATE else NEG_INF_TS
            return dt.datetime(v.year, v.month, v.day)
        if s == "timestamptz":
            return to_local(v)[0]
    if d == "timestamptz":
        if s == "date":
            if v in (POS_INF_DATE, NEG_INF_DATE):
                return POS_INF_TSTZ if v == POS_INF_DATE else NEG_INF_TSTZ
            return from_local(dt.datetime(v.year, v.month, v.day))
        if s == "timestamp":
            return from_local(v)
    if d == "time":
        if s == "timestamp":
            return v.time()
        if s == "timestamptz":
            return to_local(v)[0].time()
        if s == "interval":
            us = v.micros % 86_400_000_000
            return (dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=us)).time()
        if s == "timetz":
            return v[0]
    if d == "interval" and s == "time":
        return Interval(0, 0, (v.hour * 3600 + v.minute * 60 + v.second) * 1_000_000 + v.microsecond)
    if d == "jsonb" and s == "json":
        return Json(v.v, True)
    if d == "json" and s == "jsonb":
        return Json(v.v, False, jsonb_text(v.v))
    if dst.cat == "E" and s in STRINGS:
        return input_value(v, dst)
    if d == "regclass":
        return str(v)
    raise PgError(f"cannot cast type {src.display} to {dst.display}", "42846")


def _jsonb_to(j: Json, dst: Type):
    v = j.v
    want = {"bool": "boolean", "numeric": "numeric", "int2": "smallint", "int4": "integer", "int8": "bigint",
            "float4": "real", "float8": "double precision"}[dst.name]
    if dst.name == "bool":
        if isinstance(v, bool):
            return v
    elif isinstance(v, Decimal) and not isinstance(v, bool):
        if dst.name == "numeric":
            return v
        if dst.name in ("float4", "float8"):
            return float(v)
        return int_check(int(v.to_integral_value(rounding=decimal.ROUND_HALF_UP)), dst.name)
    kind = {0: "null", 1: "string", 2: "numeric", 3: "boolean", 4: "array", 5: "object"}[jsonb_type_rank(v)]
    raise PgError(f"cannot cast jsonb {kind} to type {want}", "22023")


# ---------------------------------------------------------------- comparison

def sort_key(v, t: Type | None = None):
    """A key that orders values of one type the way PostgreSQL does (NULLs handled by the caller)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        if v.is_nan():
            return _NumKey(v)
        return _NumKey(v)
    if isinstance(v, float):
        return _NumKey(v)
    if isinstance(v, str):
        if t is not None and t.cat == "E" and t.labels:
            return t.labels.index(v) if v in t.labels else -1
        if t is not None and t.name == "bpchar":
            return v.rstrip(" ")
        return v
    if isinstance(v, Interval):
        return v.key()
    if isinstance(v, Json):
        return jsonb_cmp_key(v.v)
    if isinstance(v, list):
        el = t.elem if t is not None and t.is_array else None
        return _ArrKey([sort_key(x, el) for x in array_flat(v)])
    if isinstance(v, tuple):
        types = getattr(v, "_types", None) or [None] * len(v)
        return _ArrKey([sort_key(x, tt) for x, tt in zip(v, types)])
    return v


@functools.total_ordering
class _ArrKey:
    """Element-wise comparison; NULL elements sort after non-NULL ones."""
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __lt__(self, o):
        for a, b in zip(self.k, o.k):
            if a is None or b is None:
                if a is None and b is None:
                    continue
                return b is None
            if a < b:
                return True
            if b < a:
                return False
        return len(self.k) < len(o.k)

    def __eq__(self, o):
        return isinstance(o, _ArrKey) and self.k == o.k

    def __hash__(self):
        return hash(tuple(self.k))


def hash_key(v, t: Type | None = None):
    """A key for grouping / DISTINCT / hash joins: equal values (1 = 1.0, '1 mon' = '30 days') share it."""
    if v is None:
        return None
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, Decimal, float)):
        if isinstance(v, float) and math.isnan(v):
            return ("nan",)
        if isinstance(v, Decimal):
            if v.is_nan():
                return ("nan",)
            if v.is_infinite():
                return ("inf", v > 0)
            n = v.normalize()
            if n == n.to_integral_value():
                return int(n)
            return n
        if isinstance(v, float):
            if math.isinf(v):
                return ("inf", v > 0)
            if v == int(v):
                return int(v)
            return Decimal(repr(v))
        return v
    if isinstance(v, str):
        if t is not None and t.name == "bpchar":
            return v.rstrip(" ")
        return v
    if isinstance(v, Interval):
        return ("iv", v.key())
    if isinstance(v, Json):
        return ("j", _freeze(v.v))
    if isinstance(v, list):
        return ("a",) + tuple(hash_key(x) for x in v)
    if isinstance(v, tuple):
        return ("r",) + tuple(hash_key(x) for x in v)
    return v

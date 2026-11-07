"""Operators and scalar functions: implementations chosen at bind time for the resolved argument types.

op_impl(op, param types, result type) -> callable(a, b) (strict: never called with NULL)
make_func(name, param types, result type, nargs) -> (impl(args, ctx), strict, volatile)
"""

from __future__ import annotations

import calendar
import datetime as dt
import decimal
import hashlib
import math
import re
import uuid as _uuid
from decimal import Decimal

from . import types as T
from .errors import PgError
from .types import Interval, Json, JNULL

DAY_US = 86_400_000_000


def div0():
    return PgError("division by zero", "22012")


# ---------------------------------------------------------------- numbers

def _int_res(tname):
    lo, hi, disp = T.INT_LIMITS[tname]

    def chk(v):
        if v < lo or v > hi:
            raise PgError(f"{disp} out of range", "22003")
        return v
    return chk


def _float_chk(v, a=None, b=None):
    if math.isinf(v) and not ((a is not None and isinstance(a, float) and math.isinf(a)) or
                              (b is not None and isinstance(b, float) and math.isinf(b))):
        raise PgError("value out of range: overflow", "22003")
    return v


def _num(v):
    return v if isinstance(v, Decimal) else Decimal(v)


def num_div(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        raise div0()
    if a.is_nan() or b.is_nan():
        return Decimal("NaN")
    if a.is_infinite() or b.is_infinite():
        if a.is_infinite() and b.is_infinite():
            return Decimal("NaN")
        if b.is_infinite():
            return Decimal(0)
        return a if b > 0 else -a
    scale = T.div_scale(a, b)
    q = a / b
    return T.num_round(q, scale)


def num_mod(a: Decimal, b: Decimal) -> Decimal:
    if b == 0:
        raise div0()
    r = a.remainder_near(b) if False else a % b      # Decimal % keeps the dividend's sign, like C
    return r.quantize(Decimal(1).scaleb(-max(T.num_scale(a), T.num_scale(b))))


def int_div(a, b, chk):
    if b == 0:
        raise div0()
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return chk(q)


def int_mod(a, b):
    if b == 0:
        raise div0()
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def num_pow(a: Decimal, b: Decimal) -> Decimal:
    if a == 0 and b < 0:
        raise PgError("zero raised to a negative power is undefined", "2201F")
    if a < 0 and b != b.to_integral_value():
        raise PgError("a negative number raised to a non-integer power yields a complex result", "2201F")
    if b == 0:
        return Decimal("1.0000000000000000")
    if a == 0:
        return Decimal("0.0000000000000000")
    if b == b.to_integral_value() and abs(b) < 1_000_000:
        e = int(b)
        try:
            f = e * math.log10(abs(float(a)))
        except (ValueError, OverflowError):
            f = 0
        rscale = 16 - int(f)
        rscale = max(rscale, T.num_scale(a), 0)
        rscale = min(rscale, 1000)
        with decimal.localcontext(T.D) as ctx:
            ctx.prec = max(50, int(abs(f)) + rscale + 20)
            r = a ** e if e >= 0 else Decimal(1) / (a ** -e)
        return T.num_round(r, rscale)
    with decimal.localcontext(T.D) as ctx:
        ctx.prec = 60
        r = (b * a.ln()).exp()
    dw = int(math.floor(math.log10(abs(float(r))))) if r != 0 else 0
    rscale = max(16 - dw, T.num_scale(a), T.num_scale(b), 0)
    return T.num_round(r, min(rscale, 1000))


def ln_dweight(x: Decimal) -> int:
    """estimate_ln_dweight (numeric.c)."""
    if x <= 0:
        return 0
    if Decimal("0.9") <= x <= Decimal("1.1"):
        d = x - 1
        if d == 0:
            return 0
        w, first = T.num_weight_first(d)
        return w * 4 + int(math.log10(first))
    w, first = T.num_weight_first(x)
    # the first two base-10000 digits
    digits = first
    dweight = w * 4
    rest = abs(x).scaleb(-4 * (w - 1))
    second = int(rest) - first * 10000
    if abs(x) != first * Decimal(10000) ** w:
        digits = first * 10000 + second
        dweight -= 4
    lnv = math.log(digits) + dweight * math.log(10)
    return int(math.floor(math.log10(abs(lnv)))) if lnv != 0 else 0


def num_ln(x: Decimal) -> Decimal:
    if x == 0:
        raise PgError("cannot take logarithm of zero", "2201E")
    if x < 0:
        raise PgError("cannot take logarithm of a negative number", "2201E")
    rscale = max(16 - ln_dweight(x), T.num_scale(x), 0)
    with decimal.localcontext(T.D) as ctx:
        ctx.prec = rscale + 30
        return T.num_round(x.ln(), min(rscale, 1000))


def num_log(base: Decimal, x: Decimal) -> Decimal:
    for v in (base, x):
        if v == 0:
            raise PgError("cannot take logarithm of zero", "2201E")
        if v < 0:
            raise PgError("cannot take logarithm of a negative number", "2201E")
    if base == 1:
        raise div0()
    dw = ln_dweight(x) - ln_dweight(base)
    rscale = max(16 - dw, T.num_scale(base), T.num_scale(x), 0)
    with decimal.localcontext(T.D) as ctx:
        ctx.prec = rscale + 30
        return T.num_round(x.ln() / base.ln(), min(rscale, 1000))


def num_exp(x: Decimal) -> Decimal:
    val = float(x) * 0.434294481903252
    val = max(min(val, 2000), -2000)
    rscale = max(16 - int(val), T.num_scale(x), 0)
    with decimal.localcontext(T.D) as ctx:
        ctx.prec = rscale + int(abs(val)) + 30
        return T.num_round(x.exp(), min(rscale, 1000))


def num_sqrt(x: Decimal) -> Decimal:
    if x < 0:
        raise PgError("cannot take square root of a negative number", "2201F")
    if x == 0:
        w = 0
    else:
        w, _ = T.num_weight_first(x)
    sweight = w * 2 + 1
    rscale = max(16 - sweight, T.num_scale(x), 0)
    with decimal.localcontext(T.D) as ctx:
        ctx.prec = rscale + 40
        return T.num_round(x.sqrt(), min(rscale, 1000))


def _fl(v):
    return float(v)


def fl_pow(a, b):
    if a == 0 and b < 0:
        raise PgError("zero raised to a negative power is undefined", "2201F")
    if a < 0 and b != int(b):
        raise PgError("a negative number raised to a non-integer power yields a complex result", "2201F")
    try:
        r = math.pow(a, b)
    except OverflowError:
        raise PgError("value out of range: overflow", "22003") from None
    return r


# ---------------------------------------------------------------- comparisons

def cmp_key_fn(t: T.Type):
    """v -> key comparable with other values of the same comparison domain."""
    n = t.name
    if n in ("int2", "int4", "int8", "oid"):
        return None
    if n in ("numeric", "float4", "float8"):
        return lambda v: T._NumKey(v)
    if n in ("text", "varchar", "name", "char", "uuid", "unknown", "bool"):
        return None
    if n == "bpchar":
        return lambda v: v.rstrip(" ")
    if n == "date":
        return None
    if n in ("timestamp", "time"):
        return None
    if n == "timestamptz":
        return None
    if n == "interval":
        return lambda v: v.key()
    if t.cat == "E":
        labels = t.labels or []
        return lambda v: labels.index(v) if v in labels else -1
    return lambda v: T.sort_key(v, t)


def _cross(lt: T.Type, rt: T.Type):
    """Comparable forms for cross-type comparisons that exist without casts (date vs timestamp ...)."""
    a, b = lt.name, rt.name
    if a == b:
        k = cmp_key_fn(lt)
        return k, k
    nums = ("int2", "int4", "int8", "numeric", "float4", "float8")
    if a in nums and b in nums:
        if "float4" in (a, b) or "float8" in (a, b) or "numeric" in (a, b):
            k = lambda v: T._NumKey(v)
            return k, k
        return None, None
    dtm = {"date": 0, "timestamp": 1, "timestamptz": 2}
    if a in dtm and b in dtm:
        target = max(dtm[a], dtm[b])
        tt = [T.DATE, T.TIMESTAMP, T.TIMESTAMPTZ][target]

        def conv(src):
            if src.name == tt.name:
                return None
            return lambda v: T.cast_value(v, src, tt)
        return conv(lt), conv(rt)
    if a in T.STRINGS and b in T.STRINGS:
        return (lambda v: v.rstrip(" ")) if a == "bpchar" else None, (lambda v: v.rstrip(" ")) if b == "bpchar" else None
    k1, k2 = cmp_key_fn(lt), cmp_key_fn(rt)
    return k1, k2


_PYOPS = {"=": lambda a, b: a == b, "<>": lambda a, b: a != b, "<": lambda a, b: a < b,
          ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


def compare_impl(op, lt: T.Type, rt: T.Type):
    k1, k2 = _cross(lt, rt)
    if lt.is_array or rt.is_array or lt.name == "record" or rt.name == "record":
        def key(v, t=lt):
            return T.sort_key(v, t)
        k1 = lambda v: T.sort_key(v, lt)
        k2 = lambda v: T.sort_key(v, rt)
        if op in ("=", "<>"):
            k1 = lambda v: T.hash_key(v, lt)
            k2 = lambda v: T.hash_key(v, rt)
    if lt.name == "jsonb" and op in ("=", "<>"):
        k1 = k2 = lambda v: T.hash_key(v, lt)
    py = _PYOPS[op]
    if k1 is None and k2 is None:
        return py
    k1 = k1 or (lambda v: v)
    k2 = k2 or (lambda v: v)
    if op == "<>":
        return lambda a, b: not (k1(a) == k2(b))
    if op in (">", ">="):
        inv = {">": "<", ">=": "<="}[op]
        pyi = _PYOPS[inv]
        return lambda a, b: pyi(k2(b), k1(a))
    return lambda a, b: py(k1(a), k2(b))


def eq_impl(lt: T.Type, rt: T.Type | None = None):
    return compare_impl("=", lt, rt or lt)


def lt_impl(t: T.Type):
    return compare_impl("<", t, t)


# ---------------------------------------------------------------- dates

def days_in_month(y, m):
    return calendar.monthrange(y, m)[1]


def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    m += 1
    if y < 1 or y > 9999:
        raise PgError("timestamp out of range", "22008")
    day = min(d.day, days_in_month(y, m))
    return d.replace(year=y, month=m, day=day)


def ts_add(ts: dt.datetime, iv: Interval, sign=1) -> dt.datetime:
    if ts in (T.POS_INF_TS, T.NEG_INF_TS, T.POS_INF_TSTZ, T.NEG_INF_TSTZ):
        return ts
    try:
        if ts.tzinfo is not None:
            local, _ = T.to_local(ts)
            if iv.months:
                local = add_months(local, sign * iv.months)
            if iv.days:
                local = local + dt.timedelta(days=sign * iv.days)
            out = T.from_local(local) if (iv.months or iv.days) else ts
            return out + dt.timedelta(microseconds=sign * iv.micros)
        if iv.months:
            ts = add_months(ts, sign * iv.months)
        return ts + dt.timedelta(days=sign * iv.days, microseconds=sign * iv.micros)
    except OverflowError:
        raise PgError("timestamp out of range", "22008") from None


def ts_diff(a: dt.datetime, b: dt.datetime) -> Interval:
    if a.tzinfo is not None:
        a = a.astimezone(T.UTC)
        b = b.astimezone(T.UTC)
    delta = a - b
    us = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    return justify_hours(Interval(0, 0, us))


def justify_hours(iv: Interval) -> Interval:
    days = int(iv.micros / DAY_US)
    time = iv.micros - days * DAY_US
    d = iv.days + days
    if d > 0 and time < 0:
        time += DAY_US
        d -= 1
    elif d < 0 and time > 0:
        time -= DAY_US
        d += 1
    return Interval(iv.months, d, time)


def justify_days(iv: Interval) -> Interval:
    months = int(iv.days / 30)
    d = iv.days - months * 30
    m = iv.months + months
    if m > 0 and d < 0:
        d += 30
        m -= 1
    elif m < 0 and d > 0:
        d -= 30
        m += 1
    return Interval(m, d, iv.micros)


def age(a: dt.datetime, b: dt.datetime) -> Interval:
    """timestamp_age: field by field, borrowing from the next field (PostgreSQL's algorithm)."""
    if a.tzinfo is not None:
        a = T.to_local(a)[0]
        b = T.to_local(b)[0]
    neg = a < b
    f = [a.microsecond - b.microsecond, a.second - b.second, a.minute - b.minute, a.hour - b.hour,
         a.day - b.day, a.month - b.month, a.year - b.year]
    if neg:
        f = [-x for x in f]
    us, sec, mi, hr, mday, mon, yr = f
    while us < 0:
        us += 1_000_000
        sec -= 1
    while sec < 0:
        sec += 60
        mi -= 1
    while mi < 0:
        mi += 60
        hr -= 1
    while hr < 0:
        hr += 24
        mday -= 1
    while mday < 0:
        ref = a if neg else b
        mday += days_in_month(ref.year, ref.month)
        mon -= 1
    while mon < 0:
        mon += 12
        yr -= 1
    if neg:
        us, sec, mi, hr, mday, mon, yr = -us, -sec, -mi, -hr, -mday, -mon, -yr
    return Interval(yr * 12 + mon, mday, ((hr * 60 + mi) * 60 + sec) * 1_000_000 + us)


def iv_mul(iv: Interval, f: float) -> Interval:
    """interval_mul: fractional months spill into days, fractional days into time."""
    if isinstance(f, Decimal):
        f = float(f)
    months = iv.months * f
    days = iv.days * f
    whole_m = int(months)
    frac_m = months - whole_m
    days_from_m = frac_m * 30
    total_days = days + days_from_m
    whole_d = int(total_days)
    frac_d = total_days - whole_d
    micros = iv.micros * f + frac_d * DAY_US
    if abs(micros) > 2 ** 62:
        raise PgError("interval out of range", "22008")
    return Interval(whole_m, whole_d, int(round(micros)))


def iv_div(iv: Interval, f: float) -> Interval:
    if f == 0:
        raise div0()
    return iv_mul(iv, 1.0 / float(f))


# ---------------------------------------------------------------- text / patterns

_LIKE_CACHE: dict = {}


def like_regex(pat: str, esc: str | None = "\\", ci=False):
    key = (pat, esc, ci)
    r = _LIKE_CACHE.get(key)
    if r is not None:
        return r
    out = []
    i = 0
    n = len(pat)
    while i < n:
        ch = pat[i]
        if esc and ch == esc:
            if i + 1 >= n:
                raise PgError("LIKE pattern must not end with escape character", "22025")
            out.append(re.escape(pat[i + 1]))
            i += 2
            continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    r = re.compile("".join(out) + r"\Z", re.S | (re.I if ci else 0))
    if len(_LIKE_CACHE) > 2000:
        _LIKE_CACHE.clear()
    _LIKE_CACHE[key] = r
    return r


def like(s, pat, esc="\\", ci=False):
    return like_regex(pat, esc, ci).match(s) is not None


def similar_regex(pat: str, esc: str | None = "\\"):
    out = []
    i = 0
    n = len(pat)
    in_br = False
    while i < n:
        ch = pat[i]
        if esc and ch == esc and i + 1 < n:
            out.append(re.escape(pat[i + 1]))
            i += 2
            continue
        if in_br:
            out.append(ch)
            if ch == "]":
                in_br = False
        elif ch == "[":
            in_br = True
            out.append(ch)
        elif ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        elif ch in "|*+?{}()":
            out.append(ch)
        elif ch == ".":
            out.append(r"\.")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("(?:" + "".join(out) + r")\Z", re.S)


_RX_CACHE: dict = {}
_POSIX = {"alpha": "a-zA-Z", "digit": "0-9", "alnum": "a-zA-Z0-9", "upper": "A-Z", "lower": "a-z",
          "space": r"\s", "punct": r"!-/:-@\[-`{-~", "xdigit": "0-9A-Fa-f", "word": r"\w", "blank": r" \t",
          "cntrl": r"\x00-\x1f\x7f", "print": r" -~", "graph": r"!-~"}


def pg_regex(pat: str, flags: str = ""):
    """POSIX ARE (PostgreSQL's regex flavour) -> Python re."""
    key = (pat, flags)
    r = _RX_CACHE.get(key)
    if r is not None:
        return r
    p = pat
    p = re.sub(r"\[:(\w+):\]", lambda m: _POSIX.get(m.group(1), m.group(0)), p)
    p = p.replace(r"\m", r"\b(?=\w)").replace(r"\M", r"\b(?<=\w)").replace(r"\y", r"\b").replace(r"\Y", r"\B")
    if p.startswith("***:"):
        p = p[4:]
    fl = 0
    if "i" in flags:
        fl |= re.I
    if "n" in flags or "w" in flags:
        fl |= re.M
    else:
        fl |= re.S
    if "x" in flags:
        fl |= re.X
    try:
        r = re.compile(p, fl)
    except re.error as e:
        raise PgError(f"invalid regular expression: {e}", "2201B") from None
    if len(_RX_CACHE) > 2000:
        _RX_CACHE.clear()
    _RX_CACHE[key] = r
    return r


def regex_flags(flags: str | None):
    flags = flags or ""
    for ch in flags:
        if ch not in "bcegimnpqstwx":
            raise PgError(f'invalid regular expression option: "{ch}"', "22023")
    return flags.replace("c", "") if "i" not in flags else flags


def repl_template(rep: str) -> str:
    out = []
    i = 0
    while i < len(rep):
        ch = rep[i]
        if ch == "\\" and i + 1 < len(rep):
            nx = rep[i + 1]
            if nx.isdigit():
                out.append(f"\\g<{nx}>")
            elif nx == "&":
                out.append("\\g<0>")
            elif nx == "\\":
                out.append("\\\\")
            else:
                out.append("\\\\" + nx)
            i += 2
            continue
        if ch == "\\":
            out.append("\\\\")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def regexp_replace(s, pat, rep, flags=None, start=1, n=None):
    fl = regex_flags(flags)
    rx = pg_regex(pat, fl)
    tmpl = repl_template(rep)
    count = 0 if ("g" in fl or n == 0) else 1

    def sub(m):
        try:
            return m.expand(tmpl)
        except (re.error, IndexError):
            return m.group(0)
    head, tail = s[:start - 1], s[start - 1:]
    if n is not None and n > 0:
        k = [0]

        def subn(m):
            k[0] += 1
            return sub(m) if k[0] == n else m.group(0)
        return head + rx.sub(subn, tail)
    return head + rx.sub(sub, tail, count=count)


def substring_regex(s, pat):
    m = pg_regex(pat).search(s)
    if not m:
        return None
    if m.re.groups:
        return m.group(1)
    return m.group(0)


def pg_initcap(s: str) -> str:
    out = []
    prev_alnum = False
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower() if prev_alnum else ch.upper())
            prev_alnum = True
        else:
            out.append(ch)
            prev_alnum = False
    return "".join(out)


def pg_format(fmt: str, args: list, types: list) -> str:
    out = []
    i = 0
    argi = 0
    n = len(fmt)
    while i < n:
        ch = fmt[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            raise PgError("unterminated format() type specifier", "22023",
                          hint='For a single "%" use "%%".')
        if fmt[i] == "%":
            out.append("%")
            i += 1
            continue
        m = re.match(r"(?:(\d+)\$)?(-)?(\d+|\*(?:\d+\$)?)?([sIL])", fmt[i:])
        if not m:
            raise PgError(f'unrecognized format() type specifier "{fmt[i]}"', "22023",
                          hint='For a single "%" use "%%".')
        i += m.end()
        if m.group(1):
            argi = int(m.group(1)) - 1
        width = 0
        if m.group(3):
            if m.group(3).startswith("*"):
                width = int(args[argi] or 0)
                argi += 1
            else:
                width = int(m.group(3))
        if argi >= len(args):
            raise PgError("too few arguments for format()", "22023")
        v, t = args[argi], types[argi]
        argi += 1
        kind = m.group(4)
        if kind == "s":
            s = "" if v is None else T.output(v, t)
        elif kind == "I":
            if v is None:
                raise PgError("null values cannot be formatted as an SQL identifier", "22004")
            from .expr import quote_ident
            s = quote_ident(T.output(v, t))
        else:
            s = "NULL" if v is None else quote_literal(T.output(v, t))
        left = bool(m.group(2)) or width < 0
        width = abs(width)
        s = s.ljust(width) if left else s.rjust(width)
        out.append(s)
    return "".join(out)


def quote_literal(s: str) -> str:
    q = "'" + s.replace("'", "''") + "'"
    if "\\" in s:
        q = "E" + q.replace("\\", "\\\\")
    return q


# ---------------------------------------------------------------- hashtext (PostgreSQL's hash_bytes, ported)

_M = 0xFFFFFFFF


def _rot(x, k):
    return ((x << k) | (x >> (32 - k))) & _M


def _mix(a, b, c):
    a = (a - c) & _M; a ^= _rot(c, 4); c = (c + b) & _M      # noqa: E702
    b = (b - a) & _M; b ^= _rot(a, 6); a = (a + c) & _M      # noqa: E702
    c = (c - b) & _M; c ^= _rot(b, 8); b = (b + a) & _M      # noqa: E702
    a = (a - c) & _M; a ^= _rot(c, 16); c = (c + b) & _M     # noqa: E702
    b = (b - a) & _M; b ^= _rot(a, 19); a = (a + c) & _M     # noqa: E702
    c = (c - b) & _M; c ^= _rot(b, 4); b = (b + a) & _M      # noqa: E702
    return a, b, c


def _final(a, b, c):
    c ^= b; c = (c - _rot(b, 14)) & _M      # noqa: E702
    a ^= c; a = (a - _rot(c, 11)) & _M      # noqa: E702
    b ^= a; b = (b - _rot(a, 25)) & _M      # noqa: E702
    c ^= b; c = (c - _rot(b, 16)) & _M      # noqa: E702
    a ^= c; a = (a - _rot(c, 4)) & _M       # noqa: E702
    b ^= a; b = (b - _rot(a, 14)) & _M      # noqa: E702
    c ^= b; c = (c - _rot(b, 24)) & _M      # noqa: E702
    return a, b, c


def hash_bytes(k: bytes) -> int:
    n = len(k)
    a = b = c = (0x9e3779b9 + n + 3923095) & _M
    i = 0
    le = int.from_bytes
    while n >= 12:
        a = (a + le(k[i:i + 4], "little")) & _M
        b = (b + le(k[i + 4:i + 8], "little")) & _M
        c = (c + le(k[i + 8:i + 12], "little")) & _M
        a, b, c = _mix(a, b, c)
        i += 12
        n -= 12
    r = k[i:]
    if n >= 11:
        c = (c + (r[10] << 24)) & _M
    if n >= 10:
        c = (c + (r[9] << 16)) & _M
    if n >= 9:
        c = (c + (r[8] << 8)) & _M
    if n >= 8:
        b = (b + le(r[4:8], "little")) & _M
        a = (a + le(r[0:4], "little")) & _M
    else:
        if n >= 7:
            b = (b + (r[6] << 16)) & _M
        if n >= 6:
            b = (b + (r[5] << 8)) & _M
        if n >= 5:
            b = (b + r[4]) & _M
        if n >= 4:
            a = (a + le(r[0:4], "little")) & _M
        else:
            if n >= 3:
                a = (a + (r[2] << 16)) & _M
            if n >= 2:
                a = (a + (r[1] << 8)) & _M
            if n >= 1:
                a = (a + r[0]) & _M
    a, b, c = _final(a, b, c)
    return c - (1 << 32) if c & 0x80000000 else c


# ---------------------------------------------------------------- json helpers

def j_get(v, key):
    if isinstance(key, str):
        if isinstance(v, dict):
            return v.get(key)
        return None
    if isinstance(v, list):
        i = key
        if i < 0:
            i += len(v)
        if 0 <= i < len(v):
            return v[i]
    return None


def j_text(v):
    """->> : strings unquoted, null -> SQL NULL, other values as JSON text."""
    if v is None or v is JNULL:
        return None
    if isinstance(v, str):
        return v
    return T.jsonb_text(v)


def j_path(v, path):
    for p in path:
        if p is None:
            return None
        if isinstance(v, list):
            try:
                idx = int(p)
            except ValueError:
                return None
            v = j_get(v, idx)
        elif isinstance(v, dict):
            v = v.get(p)
        else:
            return None
        if v is None:
            return None
    return v


def j_contains(a, b) -> bool:
    """jsonb @>: objects contain sub-objects, arrays contain elements (in any order)."""
    if isinstance(b, dict):
        if not isinstance(a, dict):
            return False
        return all(k in a and j_contains(a[k], bv) for k, bv in b.items())
    if isinstance(b, list):
        if not isinstance(a, list):
            if not b:
                return False
            return False
        return all(any(j_contains(x, y) for x in a) if not isinstance(y, (dict, list)) else
                   any(j_contains(x, y) for x in a if isinstance(x, type(y))) for y in b)
    if isinstance(a, list):
        return any(_jeq(x, b) for x in a) if False else False
    return _jeq(a, b)


def _jeq(a, b):
    return T.jsonb_cmp_key(a) == T.jsonb_cmp_key(b)


def j_wrap(v):
    return None if v is None else Json(v, True)


def j_set(v, path, new, create=True, insert=False, after=False):
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(v, dict):
        out = dict(v)
        if head in out:
            if not rest and insert:
                raise PgError("cannot replace existing key", "22023",
                              hint="Try using the function jsonb_set to replace key value.")
            out[head] = j_set(out[head], rest, new, create, insert, after)
        elif not rest and create:
            out[head] = new
        return out
    if isinstance(v, list):
        try:
            i = int(head)
        except ValueError:
            raise PgError(f'path element at position {1} is not an integer: "{head}"', "22P02") from None
        out = list(v)
        n = len(out)
        if i < 0:
            i += n
        if not rest:
            if insert:
                pos = i + (1 if after else 0)
                pos = max(0, min(pos, n))
                out.insert(pos, new)
            elif 0 <= i < n:
                out[i] = new
            elif create:
                if i >= n:
                    out.append(new)
                else:
                    out.insert(0, new)
            return out
        if 0 <= i < n:
            out[i] = j_set(out[i], rest, new, create, insert, after)
        return out
    return v


def j_delete_path(v, path):
    if not path:
        return v
    head, rest = path[0], path[1:]
    if isinstance(v, dict):
        if head not in v:
            return v
        out = dict(v)
        if rest:
            out[head] = j_delete_path(out[head], rest)
        else:
            del out[head]
        return out
    if isinstance(v, list):
        try:
            i = int(head)
        except ValueError:
            return v
        out = list(v)
        if i < 0:
            i += len(out)
        if 0 <= i < len(out):
            if rest:
                out[i] = j_delete_path(out[i], rest)
            else:
                del out[i]
        return out
    return v


def j_strip_nulls(v):
    if isinstance(v, dict):
        return {k: j_strip_nulls(x) for k, x in v.items() if x is not JNULL}
    if isinstance(v, list):
        return [j_strip_nulls(x) for x in v]
    return v


def j_typeof(v):
    return {0: "null", 1: "string", 2: "number", 3: "boolean", 4: "array", 5: "object"}[T.jsonb_type_rank(v)]


def j_concat(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        out.update(b)
        return out
    la = a if isinstance(a, list) else [a]
    lb = b if isinstance(b, list) else [b]
    return la + lb


# ---------------------------------------------------------------- operators

NUMS = ("int2", "int4", "int8", "numeric", "float4", "float8")


def op_impl(op: str, params: list, ret: T.Type):
    """The implementation of a resolved operator."""
    if len(params) == 1:
        return _prefix_impl(op, params[0], ret)
    lt, rt = params
    a, b, r = lt.name, rt.name, ret.name
    if op in ("=", "<>", "<", ">", "<=", ">="):
        return compare_impl(op, lt, rt)
    if a in NUMS and b in NUMS and r in NUMS:
        return _arith_impl(op, a, b, r)
    if op in ("~~", "!~~", "~~*", "!~~*"):
        ci = "*" in op
        neg = op.startswith("!")
        return lambda s, p: like(_s(s), _s(p), "\\", ci) != neg
    if op in ("~", "~*", "!~", "!~*") and a in T.STRINGS:
        fl = "i" if "*" in op else ""
        neg = op.startswith("!")
        return lambda s, p: (pg_regex(p, fl).search(_s(s)) is not None) != neg
    if op == "^@":
        return lambda s, p: s.startswith(p)
    if op == "||":
        if lt.is_array or rt.is_array:
            if lt.is_array and rt.is_array:
                return lambda x, y: x + y
            if lt.is_array:
                return lambda x, y: x + [y]
            return lambda x, y: [x] + y
        if a == "jsonb":
            return lambda x, y: Json(j_concat(x.v, y.v), True)
        if a == "bytea":
            return lambda x, y: x + y
        return lambda x, y: _s(T.output(x, lt) if a not in T.STRINGS else x) + \
            _s(T.output(y, rt) if b not in T.STRINGS else y)
    if a in ("json", "jsonb") and op in ("->", "->>"):
        txt = op == "->>"
        binary = a == "jsonb"
        if txt:
            return lambda j, k: j_text(j_get(j.v, k))
        return lambda j, k: _jw(j_get(j.v, k), binary)
    if a in ("json", "jsonb") and op in ("#>", "#>>"):
        txt = op == "#>>"
        binary = a == "jsonb"
        if txt:
            return lambda j, p: j_text(j_path(j.v, p))
        return lambda j, p: _jw(j_path(j.v, p), binary)
    if a == "jsonb":
        if op == "@>":
            return lambda x, y: j_contains(x.v, y.v)
        if op == "<@":
            return lambda x, y: j_contains(y.v, x.v)
        if op == "?":
            return lambda x, k: (isinstance(x.v, dict) and k in x.v) or (isinstance(x.v, list) and k in x.v) or \
                (isinstance(x.v, str) and x.v == k)
        if op == "?|":
            return lambda x, ks: any(k is not None and ((isinstance(x.v, dict) and k in x.v) or
                                                        (isinstance(x.v, list) and k in x.v)) for k in ks)
        if op == "?&":
            return lambda x, ks: all(k is not None and ((isinstance(x.v, dict) and k in x.v) or
                                                        (isinstance(x.v, list) and k in x.v)) for k in ks)
        if op == "-":
            if b == "int4":
                def dele(x, i):
                    if not isinstance(x.v, list):
                        raise PgError("cannot delete from scalar" if not isinstance(x.v, dict) else
                                      "cannot delete from object using integer index", "22023")
                    out = list(x.v)
                    j = i + len(out) if i < 0 else i
                    if 0 <= j < len(out):
                        del out[j]
                    return Json(out, True)
                return dele
            if b == "text":
                def delk(x, k):
                    if isinstance(x.v, dict):
                        return Json({kk: vv for kk, vv in x.v.items() if kk != k}, True)
                    if isinstance(x.v, list):
                        return Json([e for e in x.v if e != k], True)
                    raise PgError("cannot delete from scalar", "22023")
                return delk
            return lambda x, ks: Json({kk: vv for kk, vv in x.v.items() if kk not in ks}, True) \
                if isinstance(x.v, dict) else x
        if op == "#-":
            return lambda x, p: Json(j_delete_path(x.v, p), True)
    if lt.is_array and rt.is_array:
        if op == "@>":
            return lambda x, y: all(any(T.hash_key(e) == T.hash_key(f) for f in T.array_flat(x))
                                    for e in T.array_flat(y) if e is not None) and None not in list(T.array_flat(y))
        if op == "<@":
            return lambda x, y: all(any(T.hash_key(e) == T.hash_key(f) for f in T.array_flat(y))
                                    for e in T.array_flat(x) if e is not None) and None not in list(T.array_flat(x))
        if op == "&&":
            return lambda x, y: bool({T.hash_key(e) for e in T.array_flat(x) if e is not None} &
                                     {T.hash_key(e) for e in T.array_flat(y) if e is not None})
    # dates and times
    if op == "+":
        if a == "date" and b == "int4":
            return lambda d, n: _date_add(d, n)
        if a == "int4" and b == "date":
            return lambda n, d: _date_add(d, n)
        if a == "date" and b == "interval":
            return lambda d, iv: ts_add(dt.datetime(d.year, d.month, d.day), iv)
        if a == "interval" and b == "date":
            return lambda iv, d: ts_add(dt.datetime(d.year, d.month, d.day), iv)
        if a in ("timestamp", "timestamptz") and b == "interval":
            return lambda t, iv: ts_add(t, iv)
        if a == "interval" and b in ("timestamp", "timestamptz"):
            return lambda iv, t: ts_add(t, iv)
        if a == "interval" and b == "interval":
            return lambda x, y: x + y
        if a == "date" and b == "time":
            return lambda d, t: dt.datetime.combine(d, t)
        if a == "time" and b == "date":
            return lambda t, d: dt.datetime.combine(d, t)
        if a == "time" and b == "interval":
            return lambda t, iv: _time_add(t, iv.micros)
        if a == "interval" and b == "time":
            return lambda iv, t: _time_add(t, iv.micros)
    if op == "-":
        if a == "date" and b == "date":
            return lambda x, y: (x - y).days
        if a == "date" and b == "int4":
            return lambda d, n: _date_add(d, -n)
        if a == "date" and b == "interval":
            return lambda d, iv: ts_add(dt.datetime(d.year, d.month, d.day), iv, -1)
        if a in ("timestamp", "timestamptz") and b == "interval":
            return lambda t, iv: ts_add(t, iv, -1)
        if a in ("timestamp", "timestamptz") and b == a:
            return ts_diff
        if a == "interval" and b == "interval":
            return lambda x, y: x - y
        if a == "time" and b == "time":
            return lambda x, y: Interval(0, 0, _tus(x) - _tus(y))
        if a == "time" and b == "interval":
            return lambda t, iv: _time_add(t, -iv.micros)
    if a == "interval" and b in ("float8",) and op == "*":
        return iv_mul
    if a == "float8" and b == "interval" and op == "*":
        return lambda f, iv: iv_mul(iv, f)
    if a == "interval" and op == "/":
        return iv_div
    if op in ("&", "|", "#", "<<", ">>") and a in ("int2", "int4", "int8"):
        chk = _int_res(r)
        fn = {"&": lambda x, y: x & y, "|": lambda x, y: x | y, "#": lambda x, y: x ^ y,
              "<<": lambda x, y: x << y, ">>": lambda x, y: x >> y}[op]
        return lambda x, y: chk(_wrap_int(fn(x, y), r) if op == "<<" else fn(x, y))
    raise PgError(f"operator {lt.format()} {op} {rt.format()} is not supported by postgres-lite", "0A000")


def _jw(v, binary):
    if v is None:
        return None
    return Json(v, True) if binary else Json(v, False, T.jsonb_text(v))


def _wrap_int(v, tname):
    bits = {"int2": 16, "int4": 32, "int8": 64}[tname]
    v &= (1 << bits) - 1
    if v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


def _s(v):
    return v.rstrip(" ") if False else v


def _tus(t):
    return ((t.hour * 60 + t.minute) * 60 + t.second) * 1_000_000 + t.microsecond


def _time_add(t, us):
    total = (_tus(t) + us) % DAY_US
    return (dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=total)).time()


def _date_add(d, n):
    try:
        return d + dt.timedelta(days=n)
    except OverflowError:
        raise PgError("date out of range", "22008") from None


def _arith_impl(op, a, b, r):
    if r in ("int2", "int4", "int8"):
        chk = _int_res(r)
        if op == "+":
            return lambda x, y: chk(x + y)
        if op == "-":
            return lambda x, y: chk(x - y)
        if op == "*":
            return lambda x, y: chk(x * y)
        if op == "/":
            return lambda x, y: int_div(x, y, chk)
        if op == "%":
            return int_mod
    if r == "numeric":
        if op == "+":
            return lambda x, y: _num(x) + _num(y)
        if op == "-":
            return lambda x, y: _num(x) - _num(y)
        if op == "*":
            return lambda x, y: _num(x) * _num(y)
        if op == "/":
            return lambda x, y: num_div(_num(x), _num(y))
        if op == "%":
            return lambda x, y: num_mod(_num(x), _num(y))
        if op == "^":
            return lambda x, y: num_pow(_num(x), _num(y))
    if r in ("float4", "float8"):
        f4 = r == "float4"

        def wrap(fn):
            if f4:
                return lambda x, y: T.to_float4(_float_chk(fn(float(x), float(y)), x, y))
            return lambda x, y: _float_chk(fn(float(x), float(y)), x, y)
        if op == "+":
            return wrap(lambda x, y: x + y)
        if op == "-":
            return wrap(lambda x, y: x - y)
        if op == "*":
            return wrap(lambda x, y: x * y)
        if op == "/":
            def fdiv(x, y):
                if y == 0:
                    raise div0()
                return x / y
            return wrap(fdiv)
        if op == "^":
            return wrap(fl_pow)
    raise PgError(f"operator {op} for {a}, {b} is not supported by postgres-lite", "0A000")


def _prefix_impl(op, t: T.Type, ret: T.Type):
    n = t.name
    if op == "-":
        if n in ("int2", "int4", "int8"):
            chk = _int_res(n)
            return lambda x: chk(-x)
        if n == "interval":
            return lambda x: -x
        return lambda x: -x
    if op == "+":
        return lambda x: x
    if op == "@":
        if n in ("int2", "int4", "int8"):
            chk = _int_res(n)
            return lambda x: chk(abs(x))
        return abs
    if op == "|/":
        return lambda x: _fsqrt(x)
    if op == "||/":
        return lambda x: math.copysign(abs(x) ** (1 / 3), x)
    if op == "~" and n in ("int2", "int4", "int8"):
        return lambda x: ~x
    raise PgError(f"operator {op} {t.format()} is not supported by postgres-lite", "0A000")


def _fsqrt(x):
    if x < 0:
        raise PgError("cannot take square root of a negative number", "2201F")
    return math.sqrt(x)


# ---------------------------------------------------------------- scalar functions

FUNCS: dict = {}            # name -> factory(params, ret, nargs) -> impl(args, ctx)
NONSTRICT = {"concat", "concat_ws", "format", "num_nulls", "num_nonnulls", "jsonb_build_object", "jsonb_build_array",
             "json_build_object", "json_build_array", "array_append", "array_prepend", "array_cat", "make_date",
             "to_json", "to_jsonb", "array_fill", "jsonb_set_lax", "pg_typeof", "row_to_json", "array_remove",
             "array_replace", "array_position", "array_positions", "nullif", "random", "setseed",
             "gen_random_uuid", "now", "clock_timestamp", "statement_timestamp", "transaction_timestamp",
             "current_database", "current_user", "version", "txid_current", "pi", "array_to_string",
             "jsonb_strip_nulls", "quote_nullable", "date_bin", "make_interval", "json_object_agg"}
VOLATILE = {"random", "setseed", "gen_random_uuid", "clock_timestamp", "txid_current", "nextval", "setval",
            "currval"}


def fn(*names):
    def reg(f):
        for n in names:
            FUNCS[n] = f
        return f
    return reg


def simple(*names):
    """A factory for functions that ignore the argument types: f(*args)."""
    def reg(f):
        for n in names:
            FUNCS[n] = lambda params, ret, nargs, f=f: (lambda args, c: f(*args))
        return f
    return reg


def make_func(name: str, params: list, ret: T.Type, nargs: int):
    fac = FUNCS.get(name)
    if fac is None:
        raise PgError(f"function {name}() is not supported by postgres-lite", "0A000")
    impl = fac(params, ret, nargs)
    return impl, name not in NONSTRICT, name in VOLATILE


def _numeric_or_float(params):
    return params[0].name if params else "numeric"


@fn("abs")
def _abs(params, ret, nargs):
    if ret.name in ("int2", "int4", "int8"):
        chk = _int_res(ret.name)
        return lambda a, c: chk(abs(a[0]))
    return lambda a, c: abs(a[0])


@fn("round")
def _round(params, ret, nargs):
    if ret.name == "numeric":
        if nargs == 1:
            return lambda a, c: T.num_round(a[0], 0)

        def r2(a, c):
            v, s = a
            if not v.is_finite():
                return v
            if s >= 0:
                return T.num_round(v, s)
            q = Decimal(1).scaleb(-s)
            return (v / q).quantize(Decimal(1), rounding=decimal.ROUND_HALF_UP) * q
        return r2
    return lambda a, c: float(round(a[0])) if not math.isinf(a[0]) else a[0]   # rint: half to even


@fn("trunc")
def _trunc(params, ret, nargs):
    if ret.name == "numeric":
        def t(a, c):
            v = a[0]
            s = a[1] if nargs > 1 else 0
            if not v.is_finite():
                return v
            if s >= 0:
                return v.quantize(Decimal(1).scaleb(-s), rounding=decimal.ROUND_DOWN)
            q = Decimal(1).scaleb(-s)
            return (v / q).quantize(Decimal(1), rounding=decimal.ROUND_DOWN) * q
        return t
    return lambda a, c: float(math.trunc(a[0]))


@fn("ceil", "ceiling")
def _ceil(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: a[0].to_integral_value(rounding=decimal.ROUND_CEILING) if a[0].is_finite() else a[0]
    return lambda a, c: float(math.ceil(a[0])) if math.isfinite(a[0]) else a[0]


@fn("floor")
def _floor(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: a[0].to_integral_value(rounding=decimal.ROUND_FLOOR) if a[0].is_finite() else a[0]
    return lambda a, c: float(math.floor(a[0])) if math.isfinite(a[0]) else a[0]


@fn("sign")
def _sign(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: Decimal(0 if a[0] == 0 else (1 if a[0] > 0 else -1))
    return lambda a, c: 0.0 if a[0] == 0 else (1.0 if a[0] > 0 else -1.0)


@fn("sqrt")
def _sqrt(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: num_sqrt(a[0])
    return lambda a, c: _fsqrt(a[0])


@fn("cbrt")
def _cbrt(params, ret, nargs):
    return lambda a, c: math.copysign(round(abs(a[0]) ** (1.0 / 3), 12), a[0])


@fn("power", "pow")
def _power(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: num_pow(a[0], a[1])
    return lambda a, c: _float_chk(fl_pow(float(a[0]), float(a[1])))


@fn("exp")
def _exp(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: num_exp(a[0])

    def e(a, c):
        try:
            return math.exp(a[0])
        except OverflowError:
            raise PgError("value out of range: overflow", "22003") from None
    return e


def _fln(x):
    if x == 0:
        raise PgError("cannot take logarithm of zero", "2201E")
    if x < 0:
        raise PgError("cannot take logarithm of a negative number", "2201E")
    return math.log(x)


@fn("ln")
def _ln(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: num_ln(a[0])
    return lambda a, c: _fln(a[0])


@fn("log", "log10")
def _log(params, ret, nargs):
    if ret.name == "numeric":
        if nargs == 2:
            return lambda a, c: num_log(a[0], a[1])
        return lambda a, c: num_log(Decimal(10), a[0])
    return lambda a, c: _flog10(a[0])


def _flog10(x):
    if x == 0:
        raise PgError("cannot take logarithm of zero", "2201E")
    if x < 0:
        raise PgError("cannot take logarithm of a negative number", "2201E")
    return math.log10(x)


@fn("mod")
def _mod(params, ret, nargs):
    if ret.name == "numeric":
        return lambda a, c: num_mod(a[0], a[1])
    return lambda a, c: int_mod(a[0], a[1])


@fn("div")
def _div(params, ret, nargs):
    def d(a, c):
        x, y = a
        if y == 0:
            raise div0()
        q = (x / y).to_integral_value(rounding=decimal.ROUND_DOWN)
        return q
    return d


@fn("gcd", "lcm")
def _gcd(params, ret, nargs):
    return lambda a, c: math.gcd(int(a[0]), int(a[1]))


@simple("factorial")
def _factorial(n):
    if n < 0:
        raise PgError("factorial of a negative number is undefined", "22003")
    return Decimal(math.factorial(n))


@simple("pi")
def _pi():
    return math.pi


@simple("degrees")
def _degrees(x):
    return math.degrees(x)


@simple("radians")
def _radians(x):
    return math.radians(x)


for _name, _f in (("sin", math.sin), ("cos", math.cos), ("tan", math.tan), ("asin", math.asin),
                  ("acos", math.acos), ("atan", math.atan), ("sinh", math.sinh), ("cosh", math.cosh),
                  ("tanh", math.tanh)):
    FUNCS[_name] = (lambda f: (lambda params, ret, nargs: (lambda a, c: f(a[0]))))(_f)


@simple("atan2")
def _atan2(y, x):
    return math.atan2(y, x)


@fn("scale")
def _scale(params, ret, nargs):
    return lambda a, c: T.num_scale(a[0]) if a[0].is_finite() else None


@fn("min_scale")
def _min_scale(params, ret, nargs):
    def ms(a, c):
        v = a[0]
        if not v.is_finite():
            return None
        n = v.normalize()
        return max(0, -n.as_tuple().exponent)
    return ms


@fn("trim_scale")
def _trim_scale(params, ret, nargs):
    def ts(a, c):
        v = a[0]
        if not v.is_finite():
            return v
        n = v.normalize()
        if n.as_tuple().exponent > 0:
            n = n.quantize(Decimal(1))
        return n
    return ts


@fn("width_bucket")
def _width_bucket(params, ret, nargs):
    if nargs == 2:
        def wb2(a, c):
            v, bounds = a
            k = 0
            for b in bounds:
                if b is not None and v >= b:
                    k += 1
            return k
        return wb2

    def wb(a, c):
        v, lo, hi, n = a
        if n <= 0:
            raise PgError("count must be greater than zero", "2201G")
        if lo == hi:
            raise PgError("lower bound cannot equal upper bound", "2201G")
        if isinstance(v, float) and (math.isnan(v) or math.isnan(lo) or math.isnan(hi)):
            raise PgError("operand, lower bound, and upper bound cannot be NaN", "2201G")
        if lo < hi:
            if v < lo:
                return 0
            if v >= hi:
                return n + 1
            if isinstance(v, Decimal):
                return int(((v - lo) * n / (hi - lo)).to_integral_value(rounding=decimal.ROUND_FLOOR)) + 1
            return int(math.floor((v - lo) * n / (hi - lo))) + 1
        if v > lo:
            return 0
        if v <= hi:
            return n + 1
        if isinstance(v, Decimal):
            return int(((lo - v) * n / (lo - hi)).to_integral_value(rounding=decimal.ROUND_FLOOR)) + 1
        return int(math.floor((lo - v) * n / (lo - hi))) + 1
    return wb


@fn("random")
def _random(params, ret, nargs):
    if nargs == 2:
        if ret.name in ("int4", "int8"):
            return lambda a, c: c.rng.randint(a[0], a[1])
        return lambda a, c: c.rng.uniform(float(a[0]), float(a[1]))
    return lambda a, c: c.rng.random()


@fn("setseed")
def _setseed(params, ret, nargs):
    def s(a, c):
        x = a[0]
        if x is None:
            return None
        if x < -1 or x > 1:
            raise PgError(f"setseed parameter {T.float8_out(x)} is out of allowed range [-1,1]", "22023")
        c.rng.seed(x)
        return None
    return s


@fn("gen_random_uuid")
def _gen_uuid(params, ret, nargs):
    def g(a, c):
        b = bytearray(c.rng.getrandbits(128).to_bytes(16, "big"))
        b[6] = (b[6] & 0x0F) | 0x40
        b[8] = (b[8] & 0x3F) | 0x80
        return str(_uuid.UUID(bytes=bytes(b)))
    return g


@fn("num_nulls", "num_nonnulls")
def _num_nulls(params, ret, nargs):
    return lambda a, c: None


# ---- strings

@simple("length", "char_length", "character_length")
def _length(s):
    if isinstance(s, (bytes, bytearray)):
        return len(s)
    return len(s.rstrip(" ")) if False else len(s)


@fn("length", "char_length", "character_length")
def _length_f(params, ret, nargs):
    if params and params[0].name == "bpchar":
        return lambda a, c: len(a[0].rstrip(" "))
    if params and params[0].name == "jsonb":
        return lambda a, c: None
    return lambda a, c: len(a[0])


@simple("octet_length")
def _octet_length(s):
    return len(s) if isinstance(s, bytes) else len(s.encode("utf-8"))


@simple("bit_length")
def _bit_length(s):
    return 8 * (len(s) if isinstance(s, bytes) else len(s.encode("utf-8")))


@simple("upper")
def _upper(s):
    return "".join(ch.upper() if ch.isascii() else ch for ch in s)


@simple("lower")
def _lower(s):
    return "".join(ch.lower() if ch.isascii() else ch for ch in s)


@simple("initcap")
def _initcap(s):
    return pg_initcap(s)


def _trim_chars(chars):
    return chars


@fn("btrim", "trim")
def _btrim(params, ret, nargs):
    return lambda a, c: a[0].strip(a[1] if nargs > 1 else " ")


@fn("ltrim")
def _ltrim(params, ret, nargs):
    return lambda a, c: a[0].lstrip(a[1] if nargs > 1 else " ")


@fn("rtrim")
def _rtrim(params, ret, nargs):
    return lambda a, c: a[0].rstrip(a[1] if nargs > 1 else " ")


@fn("substring", "substr")
def _substring(params, ret, nargs):
    if nargs == 2 and params[1].name in ("text",):
        return lambda a, c: substring_regex(a[0], a[1])
    if nargs == 3 and params[1].name == "text":
        def sim(a, c):
            s, pat, esc = a
            # SQL regex: the part between the two escape-double-quote markers is returned
            marker = esc + '"'
            if pat.count(marker) == 2:
                p1, p2, p3 = pat.split(marker)
                full = similar_regex(p1 + "(" + p2 + ")" + p3 if False else p1, esc)
                rx = re.compile("(?:" + similar_regex(p1, esc).pattern[3:-3] + ")(" +
                                similar_regex(p2, esc).pattern[3:-3] + ")(?:" + similar_regex(p3, esc).pattern[3:-3]
                                + r")\Z", re.S)
                m = rx.match(s)
                return m.group(1) if m else None
            m = similar_regex(pat, esc).match(s)
            return s if m else None
        return sim

    def sub(a, c):
        s = a[0]
        start = a[1]
        if nargs == 3:
            ln = a[2]
            if ln < 0:
                raise PgError("negative substring length not allowed", "22011")
            end = start + ln
        else:
            end = None
        b = max(start, 1) - 1
        if end is None:
            return s[b:]
        e = end - 1
        if e <= b:
            return ""
        return s[b:e]
    return sub


@fn("left")
def _left(params, ret, nargs):
    return lambda a, c: a[0][:a[1]] if a[1] >= 0 else a[0][:max(0, len(a[0]) + a[1])]


@fn("right")
def _right(params, ret, nargs):
    return lambda a, c: (a[0][-a[1]:] if a[1] > 0 else "") if a[1] >= 0 else a[0][-a[1]:]


@fn("position", "strpos")
def _position(params, ret, nargs):
    return lambda a, c: a[0].find(a[1]) + 1


@simple("replace")
def _replace(s, a, b):
    if a == "":
        return s
    return s.replace(a, b)


@simple("reverse")
def _reverse(s):
    return s[::-1]


@simple("repeat")
def _repeat(s, n):
    return s * max(n, 0)


def _make_pad(left):
    def factory(params, ret, nargs):
        def p(a, c):
            s, n = a[0], a[1]
            fill = a[2] if nargs > 2 else " "
            if n <= 0:
                return ""
            if len(s) >= n:
                return s[:n]
            if not fill:
                return s
            need = n - len(s)
            pad = (fill * (need // len(fill) + 1))[:need]
            return pad + s if left else s + pad
        return p
    return factory


FUNCS["lpad"] = _make_pad(True)
FUNCS["rpad"] = _make_pad(False)


@fn("split_part")
def _split_part(params, ret, nargs):
    def sp(a, c):
        s, d, n = a
        if n == 0:
            raise PgError("field position must not be zero", "22023")
        parts = s.split(d) if d else [s]
        if s == "":
            return ""
        if n < 0:
            n = len(parts) + n + 1
            if n < 1:
                return ""
        return parts[n - 1] if n <= len(parts) else ""
    return sp


@fn("concat")
def _concat(params, ret, nargs):
    return lambda a, c: "".join(T.output(v, t) for v, t in zip(a, params) if v is not None)


@fn("concat_ws")
def _concat_ws(params, ret, nargs):
    def cw(a, c):
        sep = a[0]
        if sep is None:
            return None
        return sep.join(T.output(v, t) for v, t in zip(a[1:], params[1:]) if v is not None)
    return cw


@fn("format")
def _format(params, ret, nargs):
    def f(a, c):
        if a[0] is None:
            return None
        return pg_format(a[0], a[1:], params[1:])
    return f


@simple("ascii")
def _ascii(s):
    return ord(s[0]) if s else 0


@simple("chr")
def _chr(n):
    if n == 0:
        raise PgError("null character not permitted", "54000")
    return chr(n)


@simple("md5")
def _md5(s):
    return hashlib.md5(s if isinstance(s, bytes) else s.encode("utf-8")).hexdigest()


@simple("sha256")
def _sha256(s):
    return hashlib.sha256(s).digest()


@simple("hashtext")
def _hashtext(s):
    return hash_bytes(s.encode("utf-8"))


@simple("to_hex")
def _to_hex(n):
    return format(n & ((1 << 64) - 1) if n < 0 else n, "x")


@simple("translate")
def _translate(s, frm, to):
    table = {}
    for i, ch in enumerate(frm):
        if ch in table:
            continue
        table[ch] = to[i] if i < len(to) else None
    return "".join(table.get(ch, ch) or "" if ch in table else ch for ch in s)


@simple("starts_with")
def _starts_with(s, p):
    return s.startswith(p)


@simple("quote_ident")
def _quote_ident(s):
    from .expr import quote_ident
    return quote_ident(s)


@simple("quote_literal")
def _quote_literal(s):
    return quote_literal(s)


@fn("quote_nullable")
def _quote_nullable(params, ret, nargs):
    return lambda a, c: "NULL" if a[0] is None else quote_literal(T.output(a[0], params[0]))


@fn("regexp_replace")
def _regexp_replace(params, ret, nargs):
    def rr(a, c):
        if nargs == 4 and params[3].name == "text":
            return regexp_replace(a[0], a[1], a[2], a[3])
        if nargs >= 4:
            start = a[3]
            n = a[4] if nargs > 4 else 1
            flags = a[5] if nargs > 5 else ""
            return regexp_replace(a[0], a[1], a[2], flags, start, n)
        return regexp_replace(a[0], a[1], a[2])
    return rr


@fn("regexp_match")
def _regexp_match(params, ret, nargs):
    def rm(a, c):
        m = pg_regex(a[1], regex_flags(a[2] if nargs > 2 else "")).search(a[0])
        if not m:
            return None
        if m.re.groups:
            return list(m.groups())
        return [m.group(0)]
    return rm


@fn("regexp_like")
def _regexp_like(params, ret, nargs):
    return lambda a, c: pg_regex(a[1], regex_flags(a[2] if nargs > 2 else "")).search(a[0]) is not None


@fn("regexp_count")
def _regexp_count(params, ret, nargs):
    def rc(a, c):
        start = a[2] if nargs > 2 else 1
        flags = a[3] if nargs > 3 else ""
        return len(pg_regex(a[1], regex_flags(flags)).findall(a[0][start - 1:]))
    return rc


@fn("regexp_substr")
def _regexp_substr(params, ret, nargs):
    def rs(a, c):
        start = a[2] if nargs > 2 else 1
        n = a[3] if nargs > 3 else 1
        flags = a[4] if nargs > 4 else ""
        sub = a[5] if nargs > 5 else 0
        ms = list(pg_regex(a[1], regex_flags(flags)).finditer(a[0][start - 1:]))
        if n > len(ms):
            return None
        m = ms[n - 1]
        return m.group(sub) if sub <= (m.re.groups or 0) else None
    return rs


@fn("regexp_split_to_array")
def _regexp_split_to_array(params, ret, nargs):
    return lambda a, c: regex_split(a[0], a[1], a[2] if nargs > 2 else "")


def regex_split(s, pat, flags=""):
    rx = pg_regex(pat, regex_flags(flags))
    out = []
    last = 0
    for m in rx.finditer(s):
        if m.end() == m.start():
            if m.start() == 0 or m.start() >= len(s):
                continue
        if m.start() == m.end() and m.start() == last and last != 0:
            pass
        out.append(s[last:m.start()])
        last = m.end()
    out.append(s[last:])
    if rx.pattern == "" or (out and all(len(x) <= 1 for x in out) and rx.match("") and False):
        return list(s)
    return out


@fn("string_to_array")
def _string_to_array(params, ret, nargs):
    def sta(a, c):
        s, d = a[0], a[1]
        null = a[2] if nargs > 2 else None
        if s is None:
            return None
        if s == "":
            return []
        if d is None:
            parts = list(s)
        elif d == "":
            parts = [s]
        else:
            parts = s.split(d)
        return [None if (null is not None and p == null) else p for p in parts]
    return sta


FUNCS_NONSTRICT_EXTRA = {"string_to_array"}
NONSTRICT |= FUNCS_NONSTRICT_EXTRA


@fn("array_to_string")
def _array_to_string(params, ret, nargs):
    def ats(a, c):
        arr, d = a[0], a[1]
        if arr is None or d is None:
            return None
        null = a[2] if nargs > 2 else None
        el = params[0].elem
        out = []
        for x in T.array_flat(arr):
            if x is None:
                if null is not None:
                    out.append(null)
            else:
                out.append(T.output(x, el))
        return d.join(out)
    return ats


@fn("encode")
def _encode(params, ret, nargs):
    import base64

    def e(a, c):
        data, f = a
        if f == "hex":
            return data.hex()
        if f == "base64":
            return base64.b64encode(data).decode("ascii")
        if f == "escape":
            return data.decode("latin-1")
        raise PgError(f'unrecognized encoding: "{f}"', "22023")
    return e


@fn("decode")
def _decode(params, ret, nargs):
    import base64

    def d(a, c):
        s, f = a
        if f == "hex":
            return bytes.fromhex(s)
        if f == "base64":
            return base64.b64decode(s)
        if f == "escape":
            return s.encode("latin-1")
        raise PgError(f'unrecognized encoding: "{f}"', "22023")
    return d


# ---- dates and times

@fn("now", "transaction_timestamp")
def _now(params, ret, nargs):
    return lambda a, c: c.txn_now


@fn("statement_timestamp")
def _stmt_now(params, ret, nargs):
    return lambda a, c: c.stmt_now


@fn("clock_timestamp")
def _clock(params, ret, nargs):
    return lambda a, c: c.clock_now()


@fn("timeofday")
def _timeofday(params, ret, nargs):
    return lambda a, c: T.tstz_out(c.clock_now())


def ts_parts_input(v, t: T.Type):
    """A date / timestamp / timestamptz as a local naive datetime (+ its offset)."""
    if t.name == "date":
        return dt.datetime(v.year, v.month, v.day), 0
    if t.name == "timestamptz":
        return T.to_local(v)
    if t.name == "time":
        return dt.datetime.combine(dt.date(2000, 1, 1), v), 0
    return v, 0


EXTRACT_UNITS = {"millennium": "millennium", "millenniums": "millennium", "mil": "millennium",
                 "century": "century", "centuries": "century", "c": "century", "cent": "century",
                 "decade": "decade", "decades": "decade", "dec": "decade",
                 "year": "year", "years": "year", "y": "year", "yr": "year", "yrs": "year",
                 "quarter": "quarter", "qtr": "quarter",
                 "month": "month", "months": "month", "mon": "month", "mons": "month",
                 "week": "week", "weeks": "week", "w": "week",
                 "day": "day", "days": "day", "d": "day",
                 "hour": "hour", "hours": "hour", "h": "hour", "hr": "hour", "hrs": "hour",
                 "minute": "minute", "minutes": "minute", "m": "minute", "min": "minute", "mins": "minute",
                 "second": "second", "seconds": "second", "s": "second", "sec": "second", "secs": "second",
                 "millisecond": "milliseconds", "milliseconds": "milliseconds", "ms": "milliseconds",
                 "msec": "milliseconds", "msecs": "milliseconds",
                 "microsecond": "microseconds", "microseconds": "microseconds", "us": "microseconds",
                 "usec": "microseconds", "usecs": "microseconds",
                 "dow": "dow", "isodow": "isodow", "doy": "doy", "epoch": "epoch", "isoyear": "isoyear",
                 "julian": "julian", "timezone": "timezone", "timezone_hour": "timezone_hour",
                 "timezone_minute": "timezone_minute"}


def extract_field(field: str, v, t: T.Type, numeric: bool):
    """EXTRACT (numeric result) / date_part (float8)."""
    unit = EXTRACT_UNITS.get(field.lower())
    disp = {"date": "date", "timestamp": "timestamp without time zone",
            "timestamptz": "timestamp with time zone", "interval": "interval",
            "time": "time without time zone"}.get(t.name, t.display)
    if unit is None:
        raise PgError(f'unit "{field.lower()}" not recognized for type {disp}', "22023")
    if t.name == "interval":
        return _extract_interval(unit, v, numeric, field)
    if t.name == "date" and unit in ("hour", "minute", "second", "milliseconds", "microseconds", "timezone",
                                      "timezone_hour", "timezone_minute"):
        raise PgError(f'unit "{field.lower()}" not supported for type date', "0A000")
    if t.name in ("timestamp", "date") and unit in ("timezone", "timezone_hour", "timezone_minute"):
        raise PgError(f'unit "{field.lower()}" not supported for type {disp}', "0A000")
    local, off = ts_parts_input(v, t)
    if v in (T.POS_INF_TS, T.NEG_INF_TS, T.POS_INF_TSTZ, T.NEG_INF_TSTZ, T.POS_INF_DATE, T.NEG_INF_DATE):
        if unit in ("epoch", "year", "isoyear", "decade", "century", "millennium", "julian"):
            inf = Decimal("Infinity") if v in (T.POS_INF_TS, T.POS_INF_TSTZ, T.POS_INF_DATE) else Decimal("-Infinity")
            return inf if numeric else float(inf)
        return None
    us_frac = local.microsecond
    if unit == "epoch":
        if t.name == "timestamptz":
            delta = v - T.EPOCH_TZ
        elif t.name == "date":
            delta = local - T.EPOCH
            val = Decimal(delta.days * 86400)
            return val if numeric else float(val)
        else:
            delta = local - T.EPOCH
        us = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        val = Decimal(us).scaleb(-6)
        return val if numeric else us / 1e6
    if unit == "second":
        val = Decimal(local.second * 1_000_000 + us_frac).scaleb(-6)
        return val if numeric else float(val)
    if unit == "milliseconds":
        val = Decimal(local.second * 1_000_000 + us_frac).scaleb(-3)
        return val if numeric else float(val)
    if unit == "microseconds":
        val = Decimal(local.second * 1_000_000 + us_frac)
        return val if numeric else float(val)
    if unit == "julian":
        jd = local.date().toordinal() + 1721425
        if t.name == "date":
            val = Decimal(jd)
        else:
            secs = (local.hour * 3600 + local.minute * 60 + local.second) * 1_000_000 + us_frac
            val = Decimal(jd) + Decimal(secs) / Decimal(86_400_000_000)
        return val if numeric else float(val)
    y = local.year
    iv = {
        "millennium": (y - 1) // 1000 + 1 if y > 0 else -((-y) // 1000 + 1),
        "century": (y - 1) // 100 + 1 if y > 0 else -((-y) // 100 + 1),
        "decade": y // 10,
        "year": y,
        "quarter": (local.month - 1) // 3 + 1,
        "month": local.month,
        "week": local.isocalendar()[1],
        "day": local.day,
        "hour": local.hour,
        "minute": local.minute,
        "dow": (local.weekday() + 1) % 7,
        "isodow": local.isoweekday(),
        "doy": local.timetuple().tm_yday,
        "isoyear": local.isocalendar()[0],
        "timezone": off,
        "timezone_hour": int(off / 3600),
        "timezone_minute": int((off % 3600) / 60) if off >= 0 else -int(((-off) % 3600) / 60),
    }[unit]
    return Decimal(iv) if numeric else float(iv)


def _extract_interval(unit, iv: Interval, numeric, field):
    years = int(iv.months / 12)
    mons = iv.months - years * 12
    us = iv.micros
    if unit == "epoch":
        secs = Decimal(years * 365) * 86400 + Decimal(years) * 21600 + Decimal(mons * 30 * 86400) + \
            Decimal(iv.days * 86400) + Decimal(us).scaleb(-6)
        val = secs.quantize(Decimal("0.000001"))
        return val if numeric else float(val)
    tsec = int(us / 1_000_000)
    rem_us = us - tsec * 1_000_000
    hours = int(tsec / 3600)
    mins = int((tsec - hours * 3600) / 60)
    secs = tsec - hours * 3600 - mins * 60
    if unit == "second":
        val = Decimal(secs * 1_000_000 + rem_us).scaleb(-6)
        return val if numeric else float(val)
    if unit == "milliseconds":
        val = Decimal(secs * 1_000_000 + rem_us).scaleb(-3)
        return val if numeric else float(val)
    if unit == "microseconds":
        val = Decimal(secs * 1_000_000 + rem_us)
        return val if numeric else float(val)
    vals = {"millennium": int(years / 1000), "century": int(years / 100), "decade": int(years / 10),
            "year": years, "quarter": mons // 3 + 1, "month": mons, "day": iv.days, "hour": hours,
            "minute": mins}
    if unit not in vals:
        raise PgError(f'unit "{field.lower()}" not supported for type interval', "0A000")
    v = vals[unit]
    return Decimal(v) if numeric else float(v)


@fn("extract")
def _extract(params, ret, nargs):
    t = params[1]
    return lambda a, c: extract_field(a[0], a[1], t, True)


@fn("date_part")
def _date_part(params, ret, nargs):
    t = params[1]
    return lambda a, c: extract_field(a[0], a[1], t, False)


TRUNC_UNITS = ("microseconds", "milliseconds", "second", "minute", "hour", "day", "week", "month", "quarter", "year",
               "decade", "century", "millennium")


def trunc_local(unit, local: dt.datetime) -> dt.datetime:
    if unit == "microseconds":
        return local
    if unit == "milliseconds":
        return local.replace(microsecond=local.microsecond // 1000 * 1000)
    if unit == "second":
        return local.replace(microsecond=0)
    if unit == "minute":
        return local.replace(second=0, microsecond=0)
    if unit == "hour":
        return local.replace(minute=0, second=0, microsecond=0)
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "day":
        return day
    if unit == "week":
        return day - dt.timedelta(days=day.weekday())
    if unit == "month":
        return day.replace(day=1)
    if unit == "quarter":
        return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)
    if unit == "year":
        return day.replace(month=1, day=1)
    if unit == "decade":
        return day.replace(year=day.year // 10 * 10, month=1, day=1)
    if unit == "century":
        return day.replace(year=(day.year - 1) // 100 * 100 + 1, month=1, day=1)
    if unit == "millennium":
        return day.replace(year=(day.year - 1) // 1000 * 1000 + 1, month=1, day=1)
    raise AssertionError(unit)


@fn("date_trunc")
def _date_trunc(params, ret, nargs):
    t = params[1]

    def dtr(a, c):
        field = a[0]
        unit = EXTRACT_UNITS.get(field.lower())
        disp = {"timestamp": "timestamp without time zone", "timestamptz": "timestamp with time zone",
                "interval": "interval"}[t.name]
        if unit not in TRUNC_UNITS:
            if unit is None:
                raise PgError(f'unit "{field.lower()}" not recognized for type {disp}', "22023")
            raise PgError(f'unit "{field.lower()}" not supported for type {disp}', "0A000")
        v = a[1]
        if t.name == "interval":
            return _trunc_interval(unit, v)
        if v in (T.POS_INF_TS, T.NEG_INF_TS, T.POS_INF_TSTZ, T.NEG_INF_TSTZ):
            return v
        if t.name == "timestamptz":
            tz = a[2] if nargs > 2 else None
            if tz:
                T.zone_offset(tz)
            local, _ = T.to_local(v, tz)
            return T.from_local(trunc_local(unit, local), tz)
        return trunc_local(unit, v)
    return dtr


def _trunc_interval(unit, iv):
    years = int(iv.months / 12)
    if unit in ("millennium", "century", "decade", "year"):
        div = {"millennium": 1000, "century": 100, "decade": 10, "year": 1}[unit]
        return Interval(int(years / div) * div * 12, 0, 0)
    if unit == "quarter":
        return Interval(years * 12 + int((iv.months - years * 12) / 3) * 3, 0, 0)
    if unit == "month":
        return Interval(iv.months, 0, 0)
    if unit == "day":
        return Interval(iv.months, iv.days, 0)
    q = {"hour": 3_600_000_000, "minute": 60_000_000, "second": 1_000_000, "milliseconds": 1000,
         "microseconds": 1}.get(unit)
    if q is None:
        raise PgError(f'unit "{unit}" not supported for type interval', "0A000")
    return Interval(iv.months, iv.days, int(iv.micros / q) * q)


@fn("date_bin")
def _date_bin(params, ret, nargs):
    def db(a, c):
        stride, src, origin = a
        if None in (stride, src, origin):
            return None
        if stride.months:
            raise PgError("timestamps cannot be binned into intervals containing months or years", "22008")
        s = stride.days * DAY_US + stride.micros
        if s <= 0:
            raise PgError("stride must be greater than zero", "22008")
        delta = src - origin
        us = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        k = us // s
        return origin + dt.timedelta(microseconds=k * s)
    return db


@fn("age")
def _age(params, ret, nargs):
    if nargs == 1:
        def a1(a, c):
            v = a[0]
            today = T.to_local(c.txn_now)[0].replace(hour=0, minute=0, second=0, microsecond=0)
            if params[0].name == "timestamptz":
                return age(T.from_local(today), v)
            return age(today, v)
        return a1
    return lambda a, c: age(a[0], a[1])


@fn("justify_days")
def _jd(params, ret, nargs):
    return lambda a, c: justify_days(a[0])


@fn("justify_hours")
def _jh(params, ret, nargs):
    return lambda a, c: justify_hours(a[0])


@fn("justify_interval")
def _ji(params, ret, nargs):
    return lambda a, c: justify_days(justify_hours(a[0]))


@fn("isfinite")
def _isfinite(params, ret, nargs):
    return lambda a, c: a[0] not in (T.POS_INF_TS, T.NEG_INF_TS, T.POS_INF_TSTZ, T.NEG_INF_TSTZ, T.POS_INF_DATE,
                                     T.NEG_INF_DATE)


@fn("make_date")
def _make_date(params, ret, nargs):
    def md(a, c):
        if None in a:
            return None
        try:
            return dt.date(a[0], a[1], a[2])
        except ValueError:
            raise PgError(f"date field value out of range: {a[0]}-{a[1]:02d}-{a[2]:02d}", "22008") from None
    return md


@fn("make_timestamp")
def _make_ts(params, ret, nargs):
    def mt(a, c):
        y, mo, d, h, mi, s = a
        whole = int(s)
        us = int(round((s - whole) * 1_000_000))
        try:
            return dt.datetime(y, mo, d, h, mi, whole, us)
        except ValueError:
            raise PgError("date/time field value out of range", "22008") from None
    return mt


@fn("make_interval")
def _make_interval(params, ret, nargs):
    def mi(a, c):
        vals = list(a) + [0] * (7 - len(a))
        if any(v is None for v in vals):
            return None
        years, months, weeks, days, hours, mins, secs = vals
        return Interval(years * 12 + months, weeks * 7 + days,
                        int(round((hours * 3600 + mins * 60) * 1_000_000 + float(secs) * 1_000_000)))
    return mi


@fn("timezone")
def _timezone(params, ret, nargs):
    t = params[1]

    def tz(a, c):
        zone, v = a
        if t.name == "timestamptz":     # -> timestamp: the wall time in that zone
            if v in (T.POS_INF_TSTZ, T.NEG_INF_TSTZ):
                return v.replace(tzinfo=None)
            return T.to_local(v, zone)[0]
        if t.name == "timestamp":       # -> timestamptz: this wall time in that zone
            return T.from_local(v, zone)
        return v
    return tz


@fn("to_timestamp")
def _to_timestamp(params, ret, nargs):
    if nargs == 1:
        def tt(a, c):
            x = a[0]
            if math.isnan(x):
                raise PgError("timestamp cannot be NaN", "22008")
            if math.isinf(x):
                return T.POS_INF_TSTZ if x > 0 else T.NEG_INF_TSTZ
            us = int(round(x * 1_000_000))
            return T.EPOCH_TZ + dt.timedelta(microseconds=us)
        return tt
    from .fmt import to_timestamp
    return lambda a, c: to_timestamp(a[0], a[1], tz=True)


@fn("to_date")
def _to_date(params, ret, nargs):
    from .fmt import to_timestamp
    return lambda a, c: to_timestamp(a[0], a[1], date_only=True)


@fn("to_char")
def _to_char(params, ret, nargs):
    from .fmt import to_char
    t = params[0]
    return lambda a, c: to_char(a[0], t, a[1])


@fn("to_number")
def _to_number(params, ret, nargs):
    from .fmt import to_number
    return lambda a, c: to_number(a[0], a[1])


# ---- conversions by name (int4(x), text(x), date(x) ...)

def _conv(target):
    def factory(params, ret, nargs):
        src = params[0]
        return lambda a, c: T.cast_value(a[0], src, ret, True)
    return factory


for _n in ("int4", "int8", "float8", "numeric", "text", "date", "timestamp", "timestamptz", "interval"):
    FUNCS[_n] = _conv(_n)


# ---- json

def _j(v, t):
    return T.to_json_value(v, t)


@fn("to_jsonb", "to_json")
def _to_jsonb(params, ret, nargs):
    t = params[0]
    if ret.name == "jsonb":
        return lambda a, c: None if a[0] is None else Json(_j(a[0], t), True)
    return lambda a, c: None if a[0] is None else Json(_j(a[0], t), False, T.json_text_sql(a[0], t))


@fn("jsonb_build_object", "json_build_object")
def _jbo(params, ret, nargs):
    binary = ret.name == "jsonb"

    def jbo(a, c):
        if len(a) % 2:
            raise PgError("argument list must have even number of elements", "22023",
                          hint=f"The arguments of {'jsonb' if binary else 'json'}_build_object() must consist of "
                               "alternating keys and values.")
        d = {}
        parts = []
        for i in range(0, len(a), 2):
            k = a[i]
            if k is None:
                raise PgError(f"argument {i + 1}: key must not be null", "22004")
            key = T.output(k, params[i]) if not isinstance(k, str) else k
            d[key] = _j(a[i + 1], params[i + 1])
            if not binary:
                parts.append(f"{T.json_quote(key)} : {T.json_text_sql(a[i + 1], params[i + 1])}")
        if binary:
            return Json(d, True)
        return Json(d, False, "{" + ", ".join(parts) + "}")
    return jbo


@fn("jsonb_build_array", "json_build_array")
def _jba(params, ret, nargs):
    binary = ret.name == "jsonb"

    def jba(a, c):
        vals = [_j(x, t) for x, t in zip(a, params)]
        if binary:
            return Json(vals, True)
        return Json(vals, False, "[" + ", ".join(T.json_text_sql(x, t) for x, t in zip(a, params)) + "]")
    return jba


@fn("jsonb_typeof", "json_typeof")
def _jtypeof(params, ret, nargs):
    return lambda a, c: j_typeof(a[0].v)


@fn("jsonb_array_length", "json_array_length")
def _jlen(params, ret, nargs):
    def jl(a, c):
        v = a[0].v
        if not isinstance(v, list):
            raise PgError("cannot get array length of a scalar" if not isinstance(v, dict) else
                          "cannot get array length of a non-array", "22023")
        return len(v)
    return jl


@fn("jsonb_extract_path", "json_extract_path")
def _jep(params, ret, nargs):
    binary = ret.name == "jsonb"
    return lambda a, c: _jw(j_path(a[0].v, a[1:]), binary)


@fn("jsonb_extract_path_text", "json_extract_path_text")
def _jept(params, ret, nargs):
    return lambda a, c: j_text(j_path(a[0].v, a[1:]))


@fn("jsonb_set")
def _jset(params, ret, nargs):
    def js(a, c):
        create = a[3] if nargs > 3 else True
        if not isinstance(a[0].v, (dict, list)):
            raise PgError("cannot set path in scalar", "22023")
        return Json(j_set(a[0].v, a[1], a[2].v, create), True)
    return js


@fn("jsonb_insert")
def _jins(params, ret, nargs):
    def ji(a, c):
        after = a[3] if nargs > 3 else False
        return Json(j_set(a[0].v, a[1], a[2].v, True, True, after), True)
    return ji


@fn("jsonb_strip_nulls", "json_strip_nulls")
def _jsn(params, ret, nargs):
    return lambda a, c: None if a[0] is None else Json(j_strip_nulls(a[0].v), True)


@fn("jsonb_pretty")
def _jpretty(params, ret, nargs):
    return lambda a, c: T.jsonb_text(a[0].v, indent=4)


@fn("row_to_json")
def _row_to_json(params, ret, nargs):
    def rj(a, c):
        if a[0] is None:
            return None
        return Json(_j(a[0], T.RECORD), False, T.json_text_sql(a[0], T.RECORD))
    return rj


@fn("pg_typeof")
def _pg_typeof(params, ret, nargs):
    t = params[0]
    return lambda a, c: t.format() if t.name != "unknown" else "unknown"


# ---- arrays

@fn("array_length")
def _array_length(params, ret, nargs):
    def al(a, c):
        arr, d = a
        dims = T.array_dims(arr)
        if not arr or d < 1 or d > len(dims):
            return None
        return dims[d - 1]
    return al


@fn("cardinality")
def _cardinality(params, ret, nargs):
    return lambda a, c: len(list(T.array_flat(a[0])))


@fn("array_upper")
def _array_upper(params, ret, nargs):
    def au(a, c):
        dims = T.array_dims(a[0])
        d = a[1]
        if not a[0] or d < 1 or d > len(dims):
            return None
        return dims[d - 1]
    return au


@fn("array_lower")
def _array_lower(params, ret, nargs):
    def al(a, c):
        dims = T.array_dims(a[0])
        if not a[0] or a[1] < 1 or a[1] > len(dims):
            return None
        return 1
    return al


@fn("array_dims")
def _array_dims(params, ret, nargs):
    def ad(a, c):
        if not a[0]:
            return None
        return "".join(f"[1:{d}]" for d in T.array_dims(a[0]))
    return ad


@fn("array_append")
def _array_append(params, ret, nargs):
    return lambda a, c: (a[0] or []) + [a[1]]


@fn("array_prepend")
def _array_prepend(params, ret, nargs):
    return lambda a, c: [a[0]] + (a[1] or [])


@fn("array_cat")
def _array_cat(params, ret, nargs):
    def ac(a, c):
        if a[0] is None:
            return a[1]
        if a[1] is None:
            return a[0]
        return a[0] + a[1]
    return ac


@fn("array_remove")
def _array_remove(params, ret, nargs):
    def ar(a, c):
        if a[0] is None:
            return None
        k = T.hash_key(a[1])
        return [x for x in a[0] if not (x is None and a[1] is None) and (x is None or T.hash_key(x) != k)] \
            if a[1] is not None else [x for x in a[0] if x is not None]
    return ar


@fn("array_replace")
def _array_replace(params, ret, nargs):
    def ar(a, c):
        if a[0] is None:
            return None
        k = T.hash_key(a[1]) if a[1] is not None else None
        out = []
        for x in a[0]:
            hit = (x is None and a[1] is None) or (x is not None and a[1] is not None and T.hash_key(x) == k)
            out.append(a[2] if hit else x)
        return out
    return ar


@fn("array_position")
def _array_position(params, ret, nargs):
    def ap(a, c):
        if a[0] is None:
            return None
        start = a[2] if nargs > 2 else 1
        for i, x in enumerate(a[0], 1):
            if i < start:
                continue
            if (x is None and a[1] is None) or (x is not None and a[1] is not None and T.hash_key(x) == T.hash_key(a[1])):
                return i
        return None
    return ap


@fn("array_positions")
def _array_positions(params, ret, nargs):
    def ap(a, c):
        if a[0] is None:
            return None
        return [i for i, x in enumerate(a[0], 1)
                if (x is None and a[1] is None) or (x is not None and a[1] is not None and T.hash_key(x) == T.hash_key(a[1]))]
    return ap


@fn("array_fill")
def _array_fill(params, ret, nargs):
    def af(a, c):
        v, dims = a[0], a[1]
        if dims is None:
            raise PgError("dimension array or low bound array cannot be null", "22004")

        def build(ds):
            if not ds:
                return v
            return [build(ds[1:]) for _ in range(ds[0])]
        return build(list(dims))
    return af


# ---- system

@fn("current_database")
def _current_database(params, ret, nargs):
    return lambda a, c: c.session.dbname


@fn("current_user")
def _current_user(params, ret, nargs):
    return lambda a, c: c.session.user


@fn("version")
def _version(params, ret, nargs):
    return lambda a, c: ("PostgreSQL 16.4 (Ubuntu 16.4-0ubuntu0.24.04.2) on x86_64-pc-linux-gnu, compiled by gcc "
                         "(Ubuntu 13.2.0-23ubuntu4) 13.2.0, 64-bit")


@fn("txid_current")
def _txid(params, ret, nargs):
    return lambda a, c: c.session.txid()


# function signatures the dump doesn't have
for _sig in (("json_typeof", "json", "text"), ("jsonb_path_exists", "jsonb, jsonpath", "boolean"),
             ("sin", "double precision", "double precision"), ("cos", "double precision", "double precision"),
             ("tan", "double precision", "double precision"), ("asin", "double precision", "double precision"),
             ("acos", "double precision", "double precision"), ("atan", "double precision", "double precision"),
             ("atan2", "double precision, double precision", "double precision"),
             ("tanh", "double precision", "double precision"), ("sinh", "double precision", "double precision"),
             ("cosh", "double precision", "double precision"),
             ("json_array_length", "json", "integer"), ("json_extract_path", "json, VARIADIC text[]", "json"),
             ("json_extract_path_text", "json, VARIADIC text[]", "text"),
             ("json_strip_nulls", "json", "json"), ("timeofday", "", "text"),
             ("string_to_table", "text, text", "text"), ("json_each", "json", "record"),
             ("json_each_text", "json", "record"), ("json_object_keys", "json", "text")):
    try:
        if _sig[0] not in T.PROCS or not any(len(p.args) == len([x for x in _sig[1].split(",") if x.strip()])
                                             for p in T.PROCS[_sig[0]]):
            vt = "text[]" if "VARIADIC text[]" in _sig[1] else "-"
            T.add_proc(_sig[0], _sig[1].replace("VARIADIC text[]", "text[]"), _sig[2],
                       retset=_sig[0] in ("string_to_table", "json_each", "json_each_text", "json_object_keys"),
                       variadic=vt)
    except KeyError:
        pass

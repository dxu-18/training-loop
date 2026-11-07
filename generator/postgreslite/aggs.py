"""Aggregates (plain, DISTINCT, ORDER BY, FILTER, ordered-set) and window functions with frames.

An aggregate is a factory make_agg(name, arg types, result type) -> new_state() -> state with
add(values) and result() (result() leaves the state usable, so running window frames can keep adding).
"""

from __future__ import annotations

import math
from decimal import Decimal

from . import types as T
from .errors import PgError
from .types import Interval, Json, JNULL


class Agg:
    def __init__(self, rtype, argtypes):
        self.rtype = rtype
        self.argtypes = argtypes

    def add(self, vals):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class Count(Agg):
    def __init__(self, rtype, argtypes, star=False):
        super().__init__(rtype, argtypes)
        self.n = 0
        self.star = star

    def add(self, vals):
        if self.star or vals[0] is not None:
            self.n += 1

    def result(self):
        return self.n


class Sum(Agg):
    def __init__(self, rtype, argtypes):
        super().__init__(rtype, argtypes)
        self.s = None
        self.kind = rtype.name

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        if self.s is None:
            self.s = Decimal(v) if self.kind == "numeric" and not isinstance(v, Decimal) else v
        else:
            self.s = self.s + (Decimal(v) if self.kind == "numeric" and not isinstance(v, Decimal) else v)

    def result(self):
        s = self.s
        if s is None:
            return None
        if self.kind == "int8":
            return T.int_check(s, "int8")
        if self.kind == "float4":
            return T.to_float4(s)
        return s


class Avg(Agg):
    def __init__(self, rtype, argtypes):
        super().__init__(rtype, argtypes)
        self.s = None
        self.n = 0
        self.kind = rtype.name

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        self.n += 1
        if self.kind == "numeric":
            v = v if isinstance(v, Decimal) else Decimal(v)
        if self.s is None:
            self.s = v
        else:
            self.s = self.s + v

    def result(self):
        if self.n == 0:
            return None
        if self.kind == "numeric":
            from .funcs import num_div
            return num_div(self.s, Decimal(self.n))
        if self.kind == "interval":
            from .funcs import iv_div
            return iv_div(self.s, float(self.n))
        return self.s / self.n


class MinMax(Agg):
    def __init__(self, rtype, argtypes, is_max):
        super().__init__(rtype, argtypes)
        self.v = None
        self.k = None
        self.is_max = is_max
        self.t = argtypes[0]

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        k = T.sort_key(v, self.t)
        if self.v is None or (k > self.k if self.is_max else k < self.k):
            self.v, self.k = v, k

    def result(self):
        return self.v


class BoolAgg(Agg):
    def __init__(self, rtype, argtypes, is_and):
        super().__init__(rtype, argtypes)
        self.v = None
        self.is_and = is_and

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        if self.v is None:
            self.v = v
        else:
            self.v = (self.v and v) if self.is_and else (self.v or v)

    def result(self):
        return self.v


class BitAgg(Agg):
    def __init__(self, rtype, argtypes, op):
        super().__init__(rtype, argtypes)
        self.v = None
        self.op = op

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        if self.v is None:
            self.v = v
        else:
            self.v = {"and": self.v & v, "or": self.v | v, "xor": self.v ^ v}[self.op]

    def result(self):
        return self.v


class Collect(Agg):
    """Aggregates that need every value: string_agg, array_agg, json(b)_agg, percentiles, mode."""

    def __init__(self, rtype, argtypes, kind, direct=None):
        super().__init__(rtype, argtypes)
        self.vals = []
        self.kind = kind
        self.direct = direct

    def add(self, vals):
        self.vals.append(vals)

    def result(self):
        k = self.kind
        vs = self.vals
        if k == "string_agg":
            items = [(v[0], v[1]) for v in vs if v[0] is not None]
            if not items:
                return None
            out = items[0][0]
            for val, sep in items[1:]:
                out += (sep if sep is not None else (b"" if isinstance(val, bytes) else "")) + val
            return out
        if k == "array_agg":
            if not vs:
                return None
            if self.argtypes[0].is_array:
                arrs = [v[0] for v in vs]
                if any(a is None for a in arrs):
                    raise PgError("cannot accumulate null arrays", "22004")
                dims = {len(a) for a in arrs}
                if len(dims) > 1:
                    raise PgError("cannot accumulate arrays of different dimensionality", "2202E")
                return [list(a) for a in arrs]
            return [v[0] for v in vs]
        if k in ("jsonb_agg", "json_agg"):
            if not vs:
                return None
            t = self.argtypes[0]
            items = [T.to_json_value(v[0], t) for v in vs]
            if k == "jsonb_agg":
                return Json(items, True)
            text = "["
            for i, v in enumerate(vs):
                if i:
                    text += ", "
                    if v[0] is not None and (t.is_array or isinstance(v[0], tuple)):
                        text += "\n "
                text += T.json_text_sql(v[0], t)
            return Json(items, False, text + "]")
        if k in ("jsonb_object_agg", "json_object_agg"):
            if not vs:
                return None
            d = {}
            for key, val in vs:
                if key is None:
                    raise PgError("field name must not be null", "22004")
                d[T.output(key, self.argtypes[0])] = T.to_json_value(val, self.argtypes[1])
            if k == "jsonb_object_agg":
                return Json(d, True)
            return Json(d, False, "{ " + ", ".join(f"{T.json_quote(T.output(key, self.argtypes[0]))} : "
                                                   f"{T.json_text_sql(val, self.argtypes[1])}" for key, val in vs)
                        + " }")
        if k == "percentile_cont":
            frac = self.direct[0]
            xs = [v[0] for v in vs if v[0] is not None]
            if frac is None:
                return None
            if isinstance(frac, list):
                return [_pct_cont(xs, f, self.argtypes[0]) for f in frac]
            return _pct_cont(xs, frac, self.argtypes[0])
        if k == "percentile_disc":
            frac = self.direct[0]
            xs = [v[0] for v in vs if v[0] is not None]
            if frac is None:
                return None
            if isinstance(frac, list):
                return [_pct_disc(xs, f) for f in frac]
            return _pct_disc(xs, frac)
        if k == "mode":
            xs = [v[0] for v in vs if v[0] is not None]
            if not xs:
                return None
            best, bestn, cur, curn = None, 0, None, 0
            for x in xs:
                hk = T.hash_key(x)
                if curn and hk == T.hash_key(cur):
                    curn += 1
                else:
                    cur, curn = x, 1
                if curn > bestn:
                    best, bestn = cur, curn
            return best
        raise AssertionError(k)


def _pct_cont(xs, frac, t):
    if frac < 0 or frac > 1 or (isinstance(frac, float) and math.isnan(frac)):
        raise PgError(f"percentile value {T.float8_out(frac)} is not between 0 and 1", "22003")
    if not xs:
        return None
    n = len(xs)
    pos = frac * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if t.name == "interval":
        a, b = xs[lo], xs[hi]
        if lo == hi:
            return a
        from .funcs import iv_mul
        return a + iv_mul(b - a, pos - lo)
    a, b = float(xs[lo]), float(xs[hi])
    if lo == hi:
        return a
    return a + (pos - lo) * (b - a)


def _pct_disc(xs, frac):
    if frac < 0 or frac > 1:
        raise PgError(f"percentile value {T.float8_out(frac)} is not between 0 and 1", "22003")
    if not xs:
        return None
    n = len(xs)
    k = int(math.ceil(frac * n))
    if k > 0:
        k -= 1
    return xs[min(k, n - 1)]


class Stats(Agg):
    """stddev / variance (numeric with numeric.c's scale rules, or float8)."""

    def __init__(self, rtype, argtypes, kind):
        super().__init__(rtype, argtypes)
        self.kind = kind
        self.n = 0
        self.sx = None
        self.sx2 = None
        self.vals = []
        self.numeric = rtype.name == "numeric"

    def add(self, vals):
        v = vals[0]
        if v is None:
            return
        self.n += 1
        if self.numeric:
            d = v if isinstance(v, Decimal) else Decimal(v)
            self.sx = d if self.sx is None else self.sx + d
            sq = d * d
            self.sx2 = sq if self.sx2 is None else self.sx2 + sq
        else:
            self.vals.append(float(v))

    def result(self):
        n = self.n
        sample = self.kind in ("stddev", "stddev_samp", "variance", "var_samp")
        is_var = self.kind in ("variance", "var_samp", "var_pop")
        if n == 0 or (sample and n == 1):
            return None
        if not self.numeric:
            # float8_accum (Youngs-Cramer)
            N = sx = sxx = 0.0
            for x in self.vals:
                old = N
                N += 1.0
                sx += x
                if old > 0.0:
                    t = x * N - sx
                    sxx += t * t / (N * old)
            var = sxx / (N - 1.0 if sample else N)
            return var if is_var else math.sqrt(var)
        from .funcs import num_div
        N = Decimal(n)
        num = N * self.sx2 - self.sx * self.sx
        if num <= 0:
            return Decimal(0)
        den = N * (N - 1) if sample else N * N
        scale = T.div_scale(num, den)
        var = T.num_round(num / den, scale)
        if is_var:
            return var
        import decimal
        with decimal.localcontext(T.D) as ctx:
            ctx.prec = scale + 40
            return T.num_round(var.sqrt(), scale)


class Corr(Agg):
    """corr, covar_*, regr_* over (y, x) pairs: PostgreSQL's float8_regr_accum (Youngs-Cramer), bit for bit."""

    def __init__(self, rtype, argtypes, kind):
        super().__init__(rtype, argtypes)
        self.kind = kind
        self.n = 0.0
        self.sx = self.sxx = self.sy = self.syy = self.sxy = 0.0

    def add(self, vals):
        y, x = vals
        if y is None or x is None:
            return
        x, y = float(x), float(y)
        old = self.n
        n = old + 1.0
        sx = self.sx + x
        sy = self.sy + y
        if old > 0.0:
            tx = x * n - sx
            ty = y * n - sy
            scale = 1.0 / (n * old)
            self.sxx += tx * tx * scale
            self.syy += ty * ty * scale
            self.sxy += tx * ty * scale
        self.n, self.sx, self.sy = n, sx, sy

    def result(self):
        k = self.kind
        n = self.n
        if k == "regr_count":
            return int(n)
        if n < 1.0:
            return None
        sx, sxx, sy, syy, sxy = self.sx, self.sxx, self.sy, self.syy, self.sxy
        if k == "covar_pop":
            return sxy / n
        if k == "covar_samp":
            return None if n < 2.0 else sxy / (n - 1.0)
        if k == "regr_avgx":
            return sx / n
        if k == "regr_avgy":
            return sy / n
        if k == "regr_sxx":
            return sxx
        if k == "regr_syy":
            return syy
        if k == "regr_sxy":
            return sxy
        if k == "corr":
            if sxx == 0 or syy == 0:
                return None
            return sxy / math.sqrt(sxx * syy)
        if k == "regr_slope":
            return None if sxx == 0 else sxy / sxx
        if k == "regr_intercept":
            return None if sxx == 0 else (sy - sx * sxy / sxx) / n
        if k == "regr_r2":
            if sxx == 0:
                return None
            if syy == 0:
                return 1.0
            return (sxy * sxy) / (sxx * syy)
        raise AssertionError(k)


AGG_NAMES = {"count", "sum", "avg", "min", "max", "bool_and", "bool_or", "every", "string_agg", "array_agg",
             "jsonb_agg", "json_agg", "jsonb_object_agg", "json_object_agg", "stddev", "stddev_samp", "stddev_pop",
             "variance", "var_samp", "var_pop", "corr", "covar_pop", "covar_samp", "regr_slope", "regr_intercept",
             "regr_r2", "regr_count", "regr_avgx", "regr_avgy", "regr_sxx", "regr_syy", "regr_sxy",
             "percentile_cont", "percentile_disc", "mode", "bit_and", "bit_or", "bit_xor"}


def make_agg(name: str, argtypes: list, rtype: T.Type, star=False):
    """-> a function that creates a fresh state."""
    if name == "count":
        return lambda direct=None: Count(rtype, argtypes, star)
    if name == "sum":
        return lambda direct=None: Sum(rtype, argtypes)
    if name == "avg":
        return lambda direct=None: Avg(rtype, argtypes)
    if name in ("min", "max"):
        return lambda direct=None: MinMax(rtype, argtypes, name == "max")
    if name in ("bool_and", "every", "bool_or"):
        return lambda direct=None: BoolAgg(rtype, argtypes, name != "bool_or")
    if name in ("bit_and", "bit_or", "bit_xor"):
        return lambda direct=None: BitAgg(rtype, argtypes, name[4:])
    if name in ("string_agg", "array_agg", "jsonb_agg", "json_agg", "jsonb_object_agg", "json_object_agg", "mode",
                "percentile_cont", "percentile_disc"):
        return lambda direct=None: Collect(rtype, argtypes, name, direct)
    if name in ("stddev", "stddev_samp", "stddev_pop", "variance", "var_samp", "var_pop"):
        return lambda direct=None: Stats(rtype, argtypes, name)
    if name.startswith("regr_") or name in ("corr", "covar_pop", "covar_samp"):
        return lambda direct=None: Corr(rtype, argtypes, name)
    raise PgError(f"aggregate {name} is not supported by postgres-lite", "0A000")


# ---------------------------------------------------------------- window functions

WINDOW_ONLY = {"row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile", "lag", "lead",
               "first_value", "last_value", "nth_value"}


def frame_bounds(n, i, peers_start, peers_end, groups, gidx, mode, start, end, offs, exclude, order_vals=None,
                 range_cmp=None):
    """The rows of row i's frame as a list of indices (into the partition)."""
    s_kind, e_kind = start, end
    so, eo = offs
    if mode == "rows":
        lo = {"unbounded_preceding": 0, "current_row": i, "preceding": i - (so or 0),
              "following": i + (so or 0)}[s_kind]
        hi = {"unbounded_following": n - 1, "current_row": i, "preceding": i - (eo or 0),
              "following": i + (eo or 0)}[e_kind]
    elif mode == "groups":
        g = gidx[i]
        if s_kind == "unbounded_preceding":
            lo = 0
        elif s_kind == "current_row":
            lo = peers_start[i]
        else:
            tg = g - so if s_kind == "preceding" else g + so
            lo = groups[tg][0] if 0 <= tg < len(groups) else (0 if tg < 0 else n)
        if e_kind == "unbounded_following":
            hi = n - 1
        elif e_kind == "current_row":
            hi = peers_end[i]
        else:
            tg = g - eo if e_kind == "preceding" else g + eo
            hi = groups[tg][1] if 0 <= tg < len(groups) else (-1 if tg < 0 else n - 1)
    else:   # range
        if s_kind == "unbounded_preceding":
            lo = 0
        elif s_kind == "current_row":
            lo = peers_start[i]
        else:
            lo = range_cmp(i, "start", s_kind, so)
        if e_kind == "unbounded_following":
            hi = n - 1
        elif e_kind == "current_row":
            hi = peers_end[i]
        else:
            hi = range_cmp(i, "end", e_kind, eo)
    lo = max(lo, 0)
    hi = min(hi, n - 1)
    if lo > hi:
        return []
    idx = range(lo, hi + 1)
    if exclude is None:
        return idx
    if exclude == "current_row":
        return [j for j in idx if j != i]
    if exclude == "group":
        return [j for j in idx if not (peers_start[i] <= j <= peers_end[i])]
    if exclude == "ties":
        return [j for j in idx if j == i or not (peers_start[i] <= j <= peers_end[i])]
    return idx

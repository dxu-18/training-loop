"""to_char / to_date / to_timestamp / to_number templates (after formatting.c)."""

from __future__ import annotations

import datetime as dt
import decimal
import re
from decimal import Decimal

from . import types as T
from .errors import PgError

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

_KEYS = ["A.D.", "A.M.", "AD", "AM", "B.C.", "BC", "CC", "DAY", "DDD", "DD", "DY", "Day", "Dy", "D", "FF1", "FF2",
         "FF3", "FF4", "FF5", "FF6", "FX", "HH24", "HH12", "HH", "IDDD", "ID", "IW", "IYYY", "IYY", "IY", "I", "J",
         "MI", "MM", "MONTH", "MON", "MS", "Month", "Mon", "OF", "P.M.", "PM", "Q", "RM", "SSSSS", "SSSS", "SS",
         "TZH", "TZM", "TZ", "US", "WW", "W", "Y,YYY", "YYYY", "YYY", "YY", "Y",
         "a.d.", "a.m.", "ad", "am", "b.c.", "bc", "cc", "day", "ddd", "dd", "dy", "d", "ff1", "ff2", "ff3", "ff4",
         "ff5", "ff6", "fx", "hh24", "hh12", "hh", "iddd", "id", "iw", "iyyy", "iyy", "iy", "i", "j", "mi", "mm",
         "month", "mon", "ms", "of", "p.m.", "pm", "q", "rm", "sssss", "ssss", "ss", "tz", "us", "ww", "w",
         "y,yyy", "yyyy", "yyy", "yy", "y"]
_KEYS.sort(key=len, reverse=True)
_CASELESS = {"cc", "ddd", "dd", "d", "hh24", "hh12", "hh", "iddd", "id", "iw", "iyyy", "iyy", "iy", "i", "j", "mi",
             "mm", "ms", "q", "sssss", "ssss", "ss", "us", "ww", "w", "y,yyy", "yyyy", "yyy", "yy", "y", "of",
             "tzh", "tzm", "ff1", "ff2", "ff3", "ff4", "ff5", "ff6"}


def _tokens(fmt: str):
    out = []
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] == '"':
            j = i + 1
            buf = []
            while j < n and fmt[j] != '"':
                if fmt[j] == "\\" and j + 1 < n:
                    j += 1
                buf.append(fmt[j])
                j += 1
            out.append(("lit", "".join(buf), False))
            i = j + 1
            continue
        if fmt[i] == "\\" and i + 1 < n:
            out.append(("lit", fmt[i + 1], False))
            i += 2
            continue
        fm = False
        if fmt.startswith(("FM", "fm"), i):
            fm = True
            i += 2
        hit = None
        for k in _KEYS:
            if fmt.startswith(k, i):
                hit = k
                break
            if k.lower() in _CASELESS and fmt[i:i + len(k)].lower() == k.lower():
                hit = fmt[i:i + len(k)]
                break
        if hit is None:
            if fm:
                continue
            out.append(("lit", fmt[i], False))
            i += 1
            continue
        i += len(hit)
        suffix = None
        if fmt.startswith(("TH", "th"), i):
            suffix = fmt[i:i + 2]
            i += 2
        out.append(("key", hit, fm, suffix))
    return out


def _ordinal(n: int, upper: bool) -> str:
    if 11 <= n % 100 <= 13:
        s = "th"
    else:
        s = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return s.upper() if upper else s


def to_char(v, t: T.Type, fmt: str) -> str:
    if v is None or fmt is None:
        return None
    if t.name in ("int2", "int4", "int8", "numeric", "float4", "float8"):
        return num_to_char(v, fmt)
    if t.name == "interval":
        return _interval_to_char(v, fmt)
    tz_abbr, off = "", None
    if t.name == "timestamptz":
        local, off = T.to_local(v)
        tz_abbr = T.zone_abbrev(T.CTX.timezone, v.replace(tzinfo=None)) if T.CTX.timezone else "UTC"
        if tz_abbr in ("UTC", "Etc/UTC"):
            tz_abbr = "UTC"
    elif t.name == "date":
        local = dt.datetime(v.year, v.month, v.day)
    else:
        local = v
    return _dt_to_char(local, fmt, tz_abbr, off)


def _dt_to_char(d: dt.datetime, fmt: str, tz_abbr="", off=None) -> str:
    out = []
    for tok in _tokens(fmt):
        if tok[0] == "lit":
            out.append(tok[1])
            continue
        _, k, fm, suffix = tok
        num = None
        width = 0
        kl = k.lower()
        if kl in ("yyyy",):
            num, width = d.year, 4
        elif kl == "y,yyy":
            s = f"{d.year // 1000},{d.year % 1000:03d}"
            out.append(s)
            continue
        elif kl == "yyy":
            num, width = d.year % 1000, 3
        elif kl == "yy":
            num, width = d.year % 100, 2
        elif kl == "y":
            num, width = d.year % 10, 1
        elif kl == "iyyy":
            num, width = d.isocalendar()[0], 4
        elif kl == "iyy":
            num, width = d.isocalendar()[0] % 1000, 3
        elif kl == "iy":
            num, width = d.isocalendar()[0] % 100, 2
        elif kl == "i":
            num, width = d.isocalendar()[0] % 10, 1
        elif kl == "mm":
            num, width = d.month, 2
        elif kl == "dd":
            num, width = d.day, 2
        elif kl == "ddd":
            num, width = d.timetuple().tm_yday, 3
        elif kl == "iddd":
            iso = d.isocalendar()
            num, width = (iso[1] - 1) * 7 + iso[2], 3
        elif kl == "d":
            num, width = (d.weekday() + 1) % 7 + 1, 1
        elif kl == "id":
            num, width = d.isoweekday(), 1
        elif kl in ("hh", "hh12"):
            h = d.hour % 12
            num, width = (12 if h == 0 else h), 2
        elif kl == "hh24":
            num, width = d.hour, 2
        elif kl == "mi":
            num, width = d.minute, 2
        elif kl == "ss":
            num, width = d.second, 2
        elif kl == "ms":
            num, width = d.microsecond // 1000, 3
        elif kl == "us":
            num, width = d.microsecond, 6
        elif kl.startswith("ff"):
            nd = int(kl[2])
            num, width = d.microsecond // (10 ** (6 - nd)), nd
        elif kl in ("ssss", "sssss"):
            num, width = d.hour * 3600 + d.minute * 60 + d.second, 0
        elif kl == "q":
            num, width = (d.month - 1) // 3 + 1, 1
        elif kl == "ww":
            num, width = (d.timetuple().tm_yday - 1) // 7 + 1, 2
        elif kl == "w":
            num, width = (d.day - 1) // 7 + 1, 1
        elif kl == "iw":
            num, width = d.isocalendar()[1], 2
        elif kl == "j":
            num, width = d.toordinal() + 1721425, 0
        elif kl == "cc":
            num, width = (d.year - 1) // 100 + 1, 2
        if num is not None:
            s = str(num) if fm else str(num).zfill(width)
            if suffix:
                s += _ordinal(num, suffix == "TH")
            out.append(s)
            continue
        if k in ("MONTH", "Month", "month"):
            name = MONTHS[d.month - 1]
            s = name.upper() if k == "MONTH" else (name.lower() if k == "month" else name)
            out.append(s if fm else s.ljust(9))
        elif k in ("MON", "Mon", "mon"):
            name = MONTHS[d.month - 1][:3]
            out.append(name.upper() if k == "MON" else (name.lower() if k == "mon" else name))
        elif k in ("DAY", "Day", "day"):
            name = DAYS[(d.weekday() + 1) % 7]
            s = name.upper() if k == "DAY" else (name.lower() if k == "day" else name)
            out.append(s if fm else s.ljust(9))
        elif k in ("DY", "Dy", "dy"):
            name = DAYS[(d.weekday() + 1) % 7][:3]
            out.append(name.upper() if k == "DY" else (name.lower() if k == "dy" else name))
        elif k in ("AM", "PM"):
            out.append("PM" if d.hour >= 12 else "AM")
        elif k in ("am", "pm"):
            out.append("pm" if d.hour >= 12 else "am")
        elif k in ("A.M.", "P.M."):
            out.append("P.M." if d.hour >= 12 else "A.M.")
        elif k in ("a.m.", "p.m."):
            out.append("p.m." if d.hour >= 12 else "a.m.")
        elif k in ("AD", "BC"):
            out.append("AD")
        elif k in ("ad", "bc"):
            out.append("ad")
        elif k in ("A.D.", "B.C."):
            out.append("A.D.")
        elif k in ("a.d.", "b.c."):
            out.append("a.d.")
        elif k in ("RM", "rm"):
            r = ROMAN[d.month - 1]
            s = r if k == "RM" else r.lower()
            out.append(s if fm else s.ljust(4))
        elif k in ("TZ", "tz"):
            out.append(tz_abbr.upper() if k == "TZ" else tz_abbr.lower())
        elif kl == "of":
            out.append(T.tz_suffix(off) if off is not None else "")
        elif kl == "tzh":
            o = off or 0
            out.append(("-" if o < 0 else "+") + f"{abs(o) // 3600:02d}")
        elif kl == "tzm":
            out.append(f"{abs(off or 0) % 3600 // 60:02d}")
        elif kl == "fx":
            pass
        else:
            out.append(k)
    return "".join(out)


def _interval_to_char(iv: T.Interval, fmt: str) -> str:
    years = int(iv.months / 12)
    mons = iv.months - years * 12
    tsec, us = divmod(iv.micros, 1_000_000)
    h, rem = divmod(tsec, 3600)
    m, s = divmod(rem, 60)
    out = []
    for tok in _tokens(fmt):
        if tok[0] == "lit":
            out.append(tok[1])
            continue
        _, k, fm, suffix = tok
        kl = k.lower()
        vals = {"hh24": (h, 2), "hh": (h % 12 or 12, 2), "hh12": (h % 12 or 12, 2), "mi": (m, 2), "ss": (s, 2),
                "ms": (us // 1000, 3), "us": (us, 6), "dd": (iv.days, 2), "ddd": (iv.days, 3), "mm": (mons, 2),
                "yyyy": (years, 4), "yy": (years % 100, 2), "y": (years % 10, 1), "d": (iv.days, 1),
                "sssss": (tsec, 0), "ssss": (tsec, 0)}
        if kl in vals:
            v, w = vals[kl]
            sgn = "-" if v < 0 else ""
            out.append(sgn + (str(abs(v)) if fm else str(abs(v)).zfill(w)))
        else:
            out.append(k)
    return "".join(out)


# ---------------------------------------------------------------- numbers

def num_to_char(v, fmt: str) -> str:
    f = fmt
    fm = "FM" in f.upper()
    up = f.upper()
    if "EEEE" in up:
        m = re.match(r"(FM)?([90]*)(?:[.D]([90]*))?EEEE", f, re.I)
        nd = len(m.group(3) or "") if m else 0
        x = float(v)
        s = f"{x:.{nd}e}"
        mant, exp = s.split("e")
        e = int(exp)
        s = f"{mant}e{'-' if e < 0 else '+'}{abs(e):02d}"
        return (s if x < 0 or fm else " " + s)
    # tokens
    int_part = ""
    frac_part = ""
    seen_point = False
    toks = []
    i = 0
    sign_tok = None
    currency = False
    while i < len(f):
        ch = f[i]
        u = f[i:i + 2].upper()
        if u == "FM":
            i += 2
            continue
        if u in ("MI", "PL", "SG", "PR", "TH"):
            sign_tok = u if u not in ("TH",) else sign_tok
            toks.append(("sign", u))
            i += 2
            continue
        if ch in "9" or ch == "0":
            toks.append(("d", ch, seen_point))
            if seen_point:
                frac_part += ch
            else:
                int_part += ch
        elif ch in ".Dd":
            if ch == "." or ch in "Dd":
                if seen_point:
                    raise PgError('multiple decimal points', "42601")
                seen_point = True
                toks.append(("point", "."))
        elif ch in ",Gg":
            toks.append(("group", ","))
        elif ch in "Ss":
            sign_tok = "S"
            toks.append(("sign", "S"))
        elif ch in "Ll":
            currency = True
            toks.append(("lit", "$"))
        elif ch in "Vv":
            toks.append(("lit", ""))
        else:
            toks.append(("lit", ch))
        i += 1
    nint = len(int_part)
    nfrac = len(frac_part)
    x = Decimal(v) if not isinstance(v, float) else Decimal(repr(v))
    neg = x < 0
    x = abs(x)
    q = x.quantize(Decimal(1).scaleb(-nfrac), rounding=decimal.ROUND_HALF_UP) if x.is_finite() else x
    digits = format(q, "f")
    ip, _, fp = digits.partition(".")
    if ip == "0":
        ip = ""
    overflow = len(ip) > nint
    fp = fp.ljust(nfrac, "0")[:nfrac]
    # integer digits right-aligned in the 9/0 slots
    ip_padded = ip.rjust(nint) if not overflow else "#" * nint
    first_zero = int_part.find("0")
    chars = []
    k = 0
    started = False
    for tok in toks:
        if tok[0] == "d" and not tok[2]:
            ch = ip_padded[k]
            if overflow:
                chars.append(("#", True))
            elif ch == " " and (first_zero != -1 and k >= first_zero):
                chars.append(("0", True))
                started = True
            elif ch == " ":
                chars.append((" ", False))
            else:
                chars.append((ch, True))
                started = True
            k += 1
        elif tok[0] == "group":
            chars.append(("," if started else " ", started))
        elif tok[0] == "point":
            chars.append((".", True))
        elif tok[0] == "d" and tok[2]:
            chars.append(("#" if overflow else None, True))
        else:
            chars.append((tok, None))
    # fraction digits
    fi = 0
    out = []
    for ch, real in chars:
        if real is None:
            continue
        if ch is None:
            out.append(fp[fi] if fi < len(fp) else "0")
            fi += 1
        else:
            out.append(ch)
    body = "".join(out)
    if fm:
        if "." in body:
            body = body.rstrip("0") if any(t[0] == "d" and t[1] == "9" and t[2] for t in toks) else body
        body = body.lstrip(" ")
    # sign
    if sign_tok in (None,):
        stripped = body.lstrip(" ")
        lead = len(body) - len(stripped)
        sign = "-" if neg else (" " if not fm else "")
        if lead > 0:
            body = body[:lead - 1] + sign + stripped if sign else body
            if not sign and not fm:
                body = " " + body
        else:
            body = sign + body
        if neg and lead == 0 and not fm:
            pass
    elif sign_tok == "S":
        stripped = body.lstrip(" ")
        lead = len(body) - len(stripped)
        sign = "-" if neg else "+"
        idx = [t[0] for t in toks].index("sign")
        if idx == 0:
            body = body[:max(lead - 1, 0)] + sign + stripped if lead > 0 else sign + stripped
            if lead > 0:
                body = " " * (lead - 1) + sign + stripped
            if len(body) < len(stripped) + lead + 0:
                pass
            body = " " * max(0, lead - 0) + sign + stripped if lead == 0 else body
            if fm:
                body = sign + stripped
        else:
            body = body + sign
    elif sign_tok == "MI":
        body = body + ("-" if neg else " ")
        if not fm:
            body = " " + body if not body.startswith(" ") else body
    elif sign_tok == "PL":
        body = ("+" if not neg else "-") + body
    elif sign_tok == "PR":
        body = ("<" + body.lstrip() + ">") if neg else (" " + body + " ")
    return body


def to_number(s: str, fmt: str) -> Decimal:
    if s is None or fmt is None:
        return None
    neg = "-" in s or ("(" in s and ")" in s) or (s.strip().endswith("-"))
    digits = []
    point = False
    nfrac = 0
    fmt_frac = 0
    if "." in fmt or "D" in fmt.upper():
        after = re.split(r"[.D]", fmt, flags=re.I)[1]
        fmt_frac = len([ch for ch in after if ch in "90"])
    for ch in s:
        if ch.isdigit():
            digits.append(ch)
            if point:
                nfrac += 1
        elif ch == "." and not point:
            point = True
            digits.append(".")
    txt = "".join(digits).strip(".") if digits else "0"
    if not txt:
        raise PgError(f'invalid input syntax for type numeric: " "', "22P02")
    d = Decimal(txt)
    if neg:
        d = -d
    if fmt_frac and nfrac > fmt_frac:
        d = d.quantize(Decimal(1).scaleb(-fmt_frac))
    return d


# ---------------------------------------------------------------- parsing dates

def to_timestamp(s: str, fmt: str, tz=False, date_only=False):
    if s is None or fmt is None:
        return None
    y = 1
    mo = d = 1
    h = mi = sec = us = 0
    pm = None
    off = None
    pos = 0
    n = len(s)
    toks = _tokens(fmt)

    def skip_sep():
        nonlocal pos
        while pos < n and not s[pos].isalnum():
            pos += 1

    def read_num(maxw):
        nonlocal pos
        skip_sep_ws()
        start = pos
        if pos < n and s[pos] in "+-":
            pos += 1
        while pos < n and s[pos].isdigit() and (maxw == 0 or pos - start < maxw):
            pos += 1
        txt = s[start:pos]
        if not txt or txt in "+-":
            raise PgError(f'invalid value "{s[start:start + max(maxw, 1)]}" for "{kk}"', "22007",
                          detail="Value must be an integer.")
        return int(txt)

    def skip_sep_ws():
        nonlocal pos
        while pos < n and s[pos] == " ":
            pos += 1
    seen_year = False
    yy_short = None
    for tok in toks:
        if tok[0] == "lit":
            lit = tok[1]
            if lit.strip() and pos < n and s[pos] == lit:
                pos += 1
            elif not lit.strip():
                skip_sep_ws()
            else:
                skip_sep()
            continue
        _, kk, fm, suffix = tok
        kl = kk.lower()
        if kl == "yyyy" or kl == "iyyy":
            y = read_num(0 if fm else 4) if False else read_num(4 if not _more_digits(s, pos) else 0)
            seen_year = True
        elif kl in ("yy", "iy"):
            v = read_num(2)
            y = 2000 + v if v < 70 else 1900 + v
            seen_year = True
        elif kl in ("yyy", "iyy"):
            v = read_num(3)
            y = 2000 + v if v < 100 else 1000 + v
            seen_year = True
        elif kl == "y,yyy":
            skip_sep_ws()
            m = re.match(r"(\d+),(\d{3})", s[pos:])
            if not m:
                raise PgError(f'invalid input string for "Y,YYY"', "22007")
            y = int(m.group(1)) * 1000 + int(m.group(2))
            pos += m.end()
        elif kl == "mm":
            mo = read_num(2)
        elif kl == "dd":
            d = read_num(2)
        elif kl == "ddd":
            doy = read_num(3)
            base = dt.date(y, 1, 1) + dt.timedelta(days=doy - 1)
            mo, d = base.month, base.day
        elif kl in ("hh", "hh12"):
            h = read_num(2)
            if h > 12 or h < 1:
                raise PgError(f'hour "{h}" is invalid for the 12-hour clock', "22008",
                              hint="Use the 24-hour clock, or give an hour between 1 and 12.")
        elif kl == "hh24":
            h = read_num(2)
        elif kl == "mi":
            mi = read_num(2)
        elif kl == "ss":
            sec = read_num(2)
        elif kl == "ms":
            us = read_num(3) * 1000
        elif kl == "us":
            us = read_num(6)
        elif kl in ("month", "mon"):
            skip_sep_ws()
            hit = None
            for i, name in enumerate(MONTHS):
                cand = name if kl == "month" else name[:3]
                if s[pos:pos + len(cand)].lower() == cand.lower():
                    hit = (i + 1, len(cand))
                    break
            if hit is None:
                raise PgError(f'invalid value "{s[pos:pos + 3]}" for "{kk}"', "22007",
                              detail="The given value did not match any of the allowed values for this field.")
            mo = hit[0]
            pos += hit[1]
        elif kl in ("day", "dy"):
            skip_sep_ws()
            for name in DAYS:
                cand = name if kl == "day" else name[:3]
                if s[pos:pos + len(cand)].lower() == cand.lower():
                    pos += len(cand)
                    break
        elif kl in ("am", "pm", "a.m.", "p.m."):
            skip_sep_ws()
            frag = s[pos:pos + len(kk)].lower()
            pm = frag.startswith("p")
            pos += len(kk)
        elif kl == "tzh":
            skip_sep_ws()
            sign = -1 if pos < n and s[pos] == "-" else 1
            if pos < n and s[pos] in "+-":
                pos += 1
            v = read_num(2)
            off = sign * v * 3600 + (off % 3600 if off else 0)
        elif kl == "tzm":
            v = read_num(2)
            off = (off or 0) + (v * 60 if (off or 0) >= 0 else -v * 60)
        elif kl == "of":
            skip_sep_ws()
            m = re.match(r"([+-])(\d{1,2})(?::(\d\d))?", s[pos:])
            if m:
                off = (int(m.group(2)) * 3600 + int(m.group(3) or 0) * 60) * (1 if m.group(1) == "+" else -1)
                pos += m.end()
        elif kl == "tz":
            skip_sep_ws()
            m = re.match(r"[A-Za-z]+", s[pos:])
            if m:
                off = T.zone_offset(m.group(0))
                pos += m.end()
        elif kl == "q":
            read_num(1)
        elif kl in ("j",):
            jd = read_num(0)
            base = dt.date.fromordinal(jd - 1721425)
            y, mo, d = base.year, base.month, base.day
        else:
            skip_sep()
    if pm is not None:
        if pm and h < 12:
            h += 12
        elif not pm and h == 12:
            h = 0
    try:
        date = dt.date(y, mo, d)
    except ValueError:
        raise PgError(f'date/time field value out of range: "{s}"', "22008") from None
    if date_only:
        return date
    try:
        naive = dt.datetime(y, mo, d) + dt.timedelta(hours=h, minutes=mi, seconds=sec, microseconds=us)
    except ValueError:
        raise PgError(f'date/time field value out of range: "{s}"', "22008") from None
    if off is not None:
        return (naive - dt.timedelta(seconds=off)).replace(tzinfo=T.UTC)
    return T.from_local(naive)


def _more_digits(s, pos):
    k = pos
    while k < len(s) and s[k] == " ":
        k += 1
    j = k
    while j < len(s) and s[j].isdigit():
        j += 1
    return j - k > 4

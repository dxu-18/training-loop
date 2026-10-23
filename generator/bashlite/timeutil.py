"""Time zones, GNU strftime and `date -d` parsing for the simulated clock (all times are UTC nanoseconds)."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone

NS = 1_000_000_000
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def _nth_sunday(year, month, n):
    """n-th Sunday (n=-1: last) of a month, as a day number."""
    if n > 0:
        first = calendar.weekday(year, month, 1)
        return 1 + (6 - first) % 7 + 7 * (n - 1)
    last = calendar.monthrange(year, month)[1]
    return last - (calendar.weekday(year, month, last) + 1) % 7


def _us_dst(utc: datetime, std_off_h: int) -> bool:
    y = utc.year
    start = datetime(y, 3, _nth_sunday(y, 3, 2), 2, tzinfo=timezone.utc) - timedelta(hours=std_off_h)
    end = datetime(y, 11, _nth_sunday(y, 11, 1), 2, tzinfo=timezone.utc) - timedelta(hours=std_off_h + 1)
    return start <= utc < end


def _eu_dst(utc: datetime) -> bool:
    y = utc.year
    start = datetime(y, 3, _nth_sunday(y, 3, -1), 1, tzinfo=timezone.utc)
    end = datetime(y, 10, _nth_sunday(y, 10, -1), 1, tzinfo=timezone.utc)
    return start <= utc < end


def _au_dst(utc: datetime) -> bool:
    y = utc.year
    end = datetime(y, 4, _nth_sunday(y, 4, 1), 3, tzinfo=timezone.utc) - timedelta(hours=11)
    start = datetime(y, 10, _nth_sunday(y, 10, 1), 2, tzinfo=timezone.utc) - timedelta(hours=10)
    return utc < end or utc >= start


# name -> (std offset hours, std abbr, dst abbr, dst rule)
ZONES = {
    "UTC": (0, "UTC", None, None), "Etc/UTC": (0, "UTC", None, None), "GMT": (0, "GMT", None, None),
    "UTC0": (0, "UTC", None, None), "Universal": (0, "UTC", None, None), "Zulu": (0, "UTC", None, None),
    "America/New_York": (-5, "EST", "EDT", "us"), "US/Eastern": (-5, "EST", "EDT", "us"),
    "America/Chicago": (-6, "CST", "CDT", "us"), "America/Denver": (-7, "MST", "MDT", "us"),
    "America/Los_Angeles": (-8, "PST", "PDT", "us"), "US/Pacific": (-8, "PST", "PDT", "us"),
    "Europe/London": (0, "GMT", "BST", "eu"), "Europe/Berlin": (1, "CET", "CEST", "eu"),
    "Europe/Paris": (1, "CET", "CEST", "eu"), "Europe/Amsterdam": (1, "CET", "CEST", "eu"),
    "Asia/Tokyo": (9, "JST", None, None), "Asia/Kolkata": (5.5, "IST", None, None),
    "Asia/Shanghai": (8, "CST", None, None), "Asia/Singapore": (8, "+08", None, None),
    "Australia/Sydney": (10, "AEST", "AEDT", "au"),
}


def zone_info(tz: str | None, utc: datetime):
    """-> (offset seconds, abbreviation)."""
    if not tz:
        tz = "UTC"
    tz = tz.lstrip(":")
    z = ZONES.get(tz)
    if z is None:
        m = re.fullmatch(r"([A-Za-z]{3,})([+-]?\d{1,2})(?::(\d\d))?", tz)
        if m:   # POSIX style like EST5 / UTC-3
            sign = -1 if not m.group(2).startswith("-") else 1
            hours = abs(int(m.group(2)))
            mins = int(m.group(3) or 0)
            return sign * (hours * 3600 + mins * 60), m.group(1)
        return 0, "UTC" if "/" not in tz else tz.split("/")[0][:3].upper() if False else "UTC"
    std, sa, da, rule = z
    dst = False
    if rule == "us":
        dst = _us_dst(utc, int(std))
    elif rule == "eu":
        dst = _eu_dst(utc)
    elif rule == "au":
        dst = _au_dst(utc)
    off = int(std * 3600) + (3600 if dst else 0)
    return off, (da if dst else sa)


def local_dt(ns: int, tz: str | None) -> datetime:
    utc = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ns // 1000)
    off, abbr = zone_info(tz, utc)
    d = utc.astimezone(timezone(timedelta(seconds=off), abbr))
    return d


def to_ns(d: datetime) -> int:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    delta = d - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * NS + delta.microseconds * 1000


def localize(naive: datetime, tz: str | None) -> datetime:
    """Interpret a naive local time in `tz`."""
    guess = naive.replace(tzinfo=timezone.utc)
    off, _ = zone_info(tz, guess - timedelta(hours=0))
    utc = guess - timedelta(seconds=off)
    off2, abbr = zone_info(tz, utc)
    if off2 != off:
        utc = guess - timedelta(seconds=off2)
        off2, abbr = zone_info(tz, utc)
    return utc.astimezone(timezone(timedelta(seconds=off2), abbr))


def strftime(fmt: str, d: datetime, ns_frac: int = 0) -> str:
    out = []
    i, n = 0, len(fmt)
    while i < n:
        c = fmt[i]
        if c != "%" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        i += 1
        flag = ""
        while i < n and fmt[i] in "-_0^#":
            flag += fmt[i]
            i += 1
        width = ""
        while i < n and fmt[i].isdigit():
            width += fmt[i]
            i += 1
        if i < n and fmt[i] in "EO":
            i += 1
        if i >= n:
            out.append("%" + flag + width)
            break
        k = fmt[i]
        i += 1
        out.append(_field(k, d, ns_frac, flag, width))
    return "".join(out)


def _pad(v: int, w: int, flag: str, default="0", width=""):
    w = int(width) if width else w
    if "-" in flag:
        return str(v)
    ch = "_" in flag and " " or ("0" in flag and "0") or default
    s = str(abs(v))
    s = s.rjust(w, ch) if ch != " " else s.rjust(w)
    return ("-" + s) if v < 0 else s


def _field(k, d: datetime, ns_frac, flag, width):
    up = "^" in flag
    wd = d.weekday()

    def text(s):
        s = s.upper() if up else s
        return s.rjust(int(width)) if width else s
    off = d.utcoffset() or timedelta(0)
    offs = int(off.total_seconds())
    if k == "Y":
        return _pad(d.year, 4, flag, width=width) if width else str(d.year)
    if k == "C":
        return _pad(d.year // 100, 2, flag, width=width)
    if k == "y":
        return _pad(d.year % 100, 2, flag, width=width)
    if k == "m":
        return _pad(d.month, 2, flag, width=width)
    if k == "d":
        return _pad(d.day, 2, flag, width=width)
    if k == "e":
        return _pad(d.day, 2, flag, " ", width)
    if k == "H":
        return _pad(d.hour, 2, flag, width=width)
    if k == "k":
        return _pad(d.hour, 2, flag, " ", width)
    if k == "I":
        return _pad((d.hour % 12) or 12, 2, flag, width=width)
    if k == "l":
        return _pad((d.hour % 12) or 12, 2, flag, " ", width)
    if k == "M":
        return _pad(d.minute, 2, flag, width=width)
    if k == "S":
        return _pad(d.second, 2, flag, width=width)
    if k == "N":
        s = f"{ns_frac:09d}"
        return s[:int(width)] if width else s
    if k == "j":
        return _pad(d.timetuple().tm_yday, 3, flag, width=width)
    if k == "a":
        return text(DAYS[wd][:3])
    if k == "A":
        return text(DAYS[wd])
    if k in ("b", "h"):
        return text(MONTHS[d.month - 1][:3])
    if k == "B":
        return text(MONTHS[d.month - 1])
    if k == "p":
        s = "AM" if d.hour < 12 else "PM"
        return s.lower() if "#" in flag else s
    if k == "P":
        return "am" if d.hour < 12 else "pm"
    if k == "Z":
        return text(d.tzname() or "UTC")
    if k == "z":
        sign = "+" if offs >= 0 else "-"
        a = abs(offs)
        return f"{sign}{a // 3600:02d}{(a % 3600) // 60:02d}"
    if k == ":":
        return ""
    if k == "s":
        return str(int((d - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()))
    if k == "u":
        return str(wd + 1)
    if k == "w":
        return str((wd + 1) % 7)
    if k == "U":
        return _pad((d.timetuple().tm_yday + 6 - (wd + 1) % 7) // 7, 2, flag, width=width)
    if k == "W":
        return _pad((d.timetuple().tm_yday + 6 - wd) // 7, 2, flag, width=width)
    if k == "V":
        return _pad(d.isocalendar()[1], 2, flag, width=width)
    if k == "G":
        return str(d.isocalendar()[0])
    if k == "F":
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
    if k == "T":
        return f"{d.hour:02d}:{d.minute:02d}:{d.second:02d}"
    if k == "R":
        return f"{d.hour:02d}:{d.minute:02d}"
    if k == "D":
        return f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}"
    if k == "r":
        return f"{(d.hour % 12) or 12:02d}:{d.minute:02d}:{d.second:02d} {'AM' if d.hour < 12 else 'PM'}"
    if k == "c":
        return f"{DAYS[wd][:3]} {MONTHS[d.month - 1][:3]} {d.day:2d} {d.hour:02d}:{d.minute:02d}:{d.second:02d} {d.year}"
    if k == "x":
        return f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}"
    if k == "X":
        return f"{d.hour:02d}:{d.minute:02d}:{d.second:02d}"
    if k == "n":
        return "\n"
    if k == "t":
        return "\t"
    if k == "%":
        return "%"
    return "%" + k


def strftime_epoch(sh, fmt, v):
    """printf %(fmt)T: v = epoch seconds, -1 now, -2 shell start."""
    k = sh.kernel
    if v == -1:
        ns = k.now
    elif v == -2:
        ns = sh.seconds_base
    else:
        ns = v * NS
    return strftime(fmt or "%X", local_dt(ns, sh.get("TZ")))


# ---------------------------------------------------------------------------- date -d

UNITS = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hour": 3600, "day": 86400, "week": 604800,
         "fortnight": 1209600, "month": "m", "year": "y"}
MON = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)}
WD = {d[:3].lower(): i for i, d in enumerate(DAYS)}


class DateError(ValueError):
    pass


def parse_date(text: str, now_ns: int, tz: str | None) -> int:
    """GNU-ish `date -d` / `touch -d` parsing. Returns UTC nanoseconds."""
    s = text.strip()
    if s.startswith("@"):
        try:
            return int(float(s[1:]) * NS)
        except ValueError:
            raise DateError(text) from None
    low = s.lower()
    now = local_dt(now_ns, tz)
    base = now.replace(tzinfo=None)
    date_set = time_set = False
    explicit_tz = None
    rel_sec = 0
    rel_mon = 0
    wday_target = None
    toks = re.findall(r"[a-z]+|[+-]?\d+(?::\d+){0,2}(?:\.\d+)?|[-+/,:.]|\S", low.replace("t", " t ") if re.match(
        r"\d{4}-\d\d-\d\dt", low) else low)
    # ISO forms first
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?\s*(z|utc|[+-]\d{2}:?\d{2})?", low)
    rest = low
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            base = datetime(y, mo, d)
        except ValueError:
            raise DateError(text) from None
        date_set = True
        if m.group(4):
            base = base.replace(hour=int(m.group(4)), minute=int(m.group(5)), second=int(m.group(6) or 0))
            time_set = True
        if m.group(7):
            explicit_tz = m.group(7)
        rest = low[m.end():]
    elif re.match(r"\d{8}\b", low):
        y, mo, d = int(low[:4]), int(low[4:6]), int(low[6:8])
        base = datetime(y, mo, d)
        date_set = True
        rest = low[8:]
    toks = re.findall(r"[a-z]+|\d+:\d+(?::\d+)?|[+-]?\d+|[-+]", rest)
    i = 0
    pending_num = None
    while i < len(toks):
        t = toks[i]
        if t in ("now", "today"):
            i += 1
            continue
        if t == "tomorrow":
            rel_sec += 86400
            i += 1
            continue
        if t == "yesterday":
            rel_sec -= 86400
            i += 1
            continue
        if t in ("noon", "midnight"):
            base = base.replace(hour=12 if t == "noon" else 0, minute=0, second=0)
            time_set = True
            i += 1
            continue
        if t in ("utc", "gmt", "z"):
            explicit_tz = "utc"
            i += 1
            continue
        if re.fullmatch(r"\d+:\d+(?::\d+)?", t):
            parts = [int(x) for x in t.split(":")]
            h, mi = parts[0], parts[1]
            sec = parts[2] if len(parts) > 2 else 0
            if i + 1 < len(toks) and toks[i + 1] in ("am", "pm"):
                h = h % 12 + (12 if toks[i + 1] == "pm" else 0)
                i += 1
            base = base.replace(hour=h, minute=mi, second=sec)
            time_set = True
            i += 1
            continue
        if re.fullmatch(r"\d+", t) and i + 1 < len(toks) and toks[i + 1] in ("am", "pm"):
            h = int(t) % 12 + (12 if toks[i + 1] == "pm" else 0)
            base = base.replace(hour=h, minute=0, second=0)
            time_set = True
            i += 2
            continue
        if t in ("next", "last", "this"):
            n = {"next": 1, "last": -1, "this": 0}[t]
            if i + 1 < len(toks):
                u = toks[i + 1].rstrip("s") if toks[i + 1] not in ("days",) else "day"
                if toks[i + 1][:3] in WD:
                    wday_target = (WD[toks[i + 1][:3]], n)
                    i += 2
                    continue
                if u in UNITS:
                    if UNITS[u] == "m":
                        rel_mon += n
                    elif UNITS[u] == "y":
                        rel_mon += 12 * n
                    else:
                        rel_sec += n * UNITS[u]
                    i += 2
                    continue
            raise DateError(text)
        if re.fullmatch(r"[+-]?\d+", t) or t in ("+", "-"):
            if t in ("+", "-"):
                if i + 1 < len(toks) and re.fullmatch(r"\d+", toks[i + 1]):
                    t = t + toks[i + 1]
                    i += 1
                else:
                    i += 1
                    continue
            num = int(t)
            if i + 1 < len(toks):
                u = toks[i + 1]
                u1 = u[:-1] if u.endswith("s") and u[:-1] in UNITS else u
                if u1 in UNITS:
                    mult = 1
                    j = i + 2
                    if j < len(toks) and toks[j] == "ago":
                        mult = -1
                        j += 1
                    if UNITS[u1] == "m":
                        rel_mon += num * mult
                    elif UNITS[u1] == "y":
                        rel_mon += 12 * num * mult
                    else:
                        rel_sec += num * UNITS[u1] * mult
                    i = j
                    continue
                if u[:3] in MON:   # "14 mar 2026"
                    mo = MON[u[:3]]
                    y = base.year
                    j = i + 2
                    if j < len(toks) and re.fullmatch(r"\d{4}", toks[j]):
                        y = int(toks[j])
                        j += 1
                    base = base.replace(year=y, month=mo, day=num)
                    date_set = True
                    i = j
                    continue
            if re.fullmatch(r"\d{4}", t) and date_set is False and pending_num is None:
                base = base.replace(year=num)
                i += 1
                continue
            raise DateError(text)
        if t[:3] in MON and (len(t) == 3 or t in [m.lower() for m in MONTHS] or t == "sept"):
            mo = MON[t[:3]]
            j = i + 1
            day = 1
            y = base.year
            if j < len(toks) and re.fullmatch(r"\d{1,2}", toks[j]):
                day = int(toks[j])
                j += 1
            if j < len(toks) and re.fullmatch(r"\d{4}", toks[j]):
                y = int(toks[j])
                j += 1
            base = base.replace(year=y, month=mo, day=day)
            date_set = True
            i = j
            continue
        if t[:3] in WD and (len(t) == 3 or t in [d.lower() for d in DAYS]):
            wday_target = (WD[t[:3]], 0)
            i += 1
            continue
        if t == "ago":
            rel_sec = -rel_sec
            rel_mon = -rel_mon
            i += 1
            continue
        if t in ("at", "and", "of"):
            i += 1
            continue
        u1 = t[:-1] if t.endswith("s") and t[:-1] in UNITS else t
        if u1 in UNITS:   # "day" alone = 1 day
            if UNITS[u1] == "m":
                rel_mon += 1
            elif UNITS[u1] == "y":
                rel_mon += 12
            else:
                rel_sec += UNITS[u1]
            i += 1
            continue
        raise DateError(text)
    if (date_set or wday_target is not None) and not time_set:
        base = base.replace(hour=0, minute=0, second=0, microsecond=0)
    if wday_target is not None:
        wd, n = wday_target
        cur = base.weekday()
        delta = (wd - cur) % 7
        if n == 1:
            delta = delta or 7
        elif n == -1:
            delta = -((cur - wd) % 7 or 7)
        base = base + timedelta(days=delta)
    if rel_mon:
        mo = base.month - 1 + rel_mon
        y = base.year + mo // 12
        mo = mo % 12 + 1
        day = min(base.day, calendar.monthrange(y, mo)[1]) if False else base.day
        try:
            base = base.replace(year=y, month=mo, day=day)
        except ValueError:   # GNU date normalizes Jan 31 + 1 month -> Mar 3
            overflow = day - calendar.monthrange(y, mo)[1]
            base = base.replace(year=y, month=mo, day=calendar.monthrange(y, mo)[1]) + timedelta(days=overflow)
    if explicit_tz:
        if explicit_tz in ("utc", "z"):
            aware = base.replace(tzinfo=timezone.utc)
        else:
            sign = 1 if explicit_tz[0] == "+" else -1
            hh, mm = int(explicit_tz[1:3]), int(explicit_tz[-2:])
            aware = base.replace(tzinfo=timezone(sign * timedelta(hours=hh, minutes=mm)))
    else:
        aware = localize(base, tz)
    return to_ns(aware) + rel_sec * NS


def ls_time(ns: int, now_ns: int, tz: str | None, style: str | None = None) -> str:
    d = local_dt(ns, tz)
    if style in ("full-iso",):
        return strftime("%Y-%m-%d %H:%M:%S.%N %z", d, ns % NS)
    if style in ("long-iso",):
        return strftime("%Y-%m-%d %H:%M", d)
    six_months = 15778476 * NS
    recent = now_ns - six_months < ns <= now_ns + 60 * NS
    if style == "iso":
        return strftime("%m-%d %H:%M", d) if recent else strftime("%Y-%m-%d ", d)
    if style and style.startswith("+"):
        return strftime(style[1:].split("\n")[0], d, ns % NS)
    return strftime("%b %e %H:%M", d) if recent else strftime("%b %e  %Y", d)


def stat_time(ns: int, tz: str | None) -> str:
    return strftime("%Y-%m-%d %H:%M:%S.%N %z", local_dt(ns, tz), ns % NS)

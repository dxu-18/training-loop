"""2.11 Dates, times & time series (frequency aliases: D, h, min, s, B, W-MON, ME, MS, QE, YE)."""

from ._common import P, challenge, seeded, template

M = "2.11"


def _stamps(rng, n, with_time=True):
    out = []
    for _ in range(n):
        d = f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        out.append(d + (f" {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}" if with_time else ""))
    return out


def _daily(rng, days=None):
    days = days or rng.choice([28, 35, 42, 60])
    start = f"2024-{rng.randint(1, 6):02d}-01"
    return (f"sales = pd.Series(np.random.default_rng({seeded(rng)}).integers(50, 150, {days}), "
            f"index=pd.date_range({start!r}, periods={days}, freq='D'))"), start, days


# ================================================================= lite

@template(M, "lite")
def parse_datetimes(rng):
    n = rng.randint(3, 5)
    variant = rng.choice(["iso", "format", "coerce", "unix"])
    if variant == "iso":
        return P("`raw` holds ISO timestamps as strings. Convert them to datetimes with `pd.to_datetime`.",
                 f"raw = pd.Series({_stamps(rng, n)!r})", "pd.to_datetime(raw)", require=["to_datetime"])
    if variant == "format":
        vals = [f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2024" for _ in range(n)]
        return P("`raw` holds European dates like `'15/01/2024'` (day first). Parse them with an explicit "
                 "`format='%d/%m/%Y'`.", f"raw = pd.Series({vals!r})", "pd.to_datetime(raw, format='%d/%m/%Y')",
                 require=["to_datetime"])
    if variant == "coerce":
        vals = _stamps(rng, n, False)
        vals[rng.randrange(1, n)] = rng.choice(["oops", "N/A", "2024-13-45"])  # not first: pandas infers the format from it
        return P("`raw` holds dates as strings, one of them garbage. Parse them so the bad one becomes `NaT` "
                 "(`errors='coerce'`).", f"raw = pd.Series({vals!r})", "pd.to_datetime(raw, errors='coerce')",
                 require=["to_datetime"])
    secs = [1_700_000_000 + rng.randint(0, 10**7) for _ in range(n)]
    return P("`raw` holds Unix timestamps in **seconds**. Convert them to datetimes (`unit='s'`).",
             f"raw = pd.Series({secs!r})", "pd.to_datetime(raw, unit='s')", require=["to_datetime"])


@template(M, "lite")
def datetime_parts(rng):
    stamps = _stamps(rng, rng.randint(3, 5))
    desc, sol = rng.choice([
        ("the year", "s.dt.year"), ("the month number", "s.dt.month"), ("the weekday name (`dt.day_name()`)", "s.dt.day_name()"),
        ("the day of the week as a number, Monday = 0 (`dt.dayofweek`)", "s.dt.dayofweek"), ("the hour", "s.dt.hour"),
        ("the quarter", "s.dt.quarter"), ("the month name (`dt.month_name()`)", "s.dt.month_name()"),
    ])
    return P(f"`s` holds event timestamps (datetime64). Using the `.dt` accessor, get {desc} of each.",
             f"s = pd.to_datetime(pd.Series({stamps!r}))", sol, require=["dt"])


# ================================================================= easy

@template(M, "easy")
def date_arithmetic(rng):
    stamps = sorted(_stamps(rng, rng.randint(3, 5)))
    k = rng.randint(1, 30)
    h = rng.randint(1, 12)
    desc, sol = rng.choice([
        (f"each timestamp plus {k} days and {h} hours (`pd.Timedelta`)", f"s + pd.Timedelta(days={k}, hours={h})"),
        ("each timestamp plus one **calendar month** (`pd.DateOffset(months=1)`; Jan 31 → Feb 29 in 2024)",
         "s + pd.DateOffset(months=1)"),
        ("the whole number of days since the earliest timestamp (`(s - s.min()).dt.days`)", "(s - s.min()).dt.days"),
        ("the gap from the previous timestamp in **hours** (`diff()` then `dt.total_seconds() / 3600`)",
         "s.diff().dt.total_seconds() / 3600"),
    ])
    return P(f"`s` holds sorted event timestamps. Compute {desc}.", f"s = pd.to_datetime(pd.Series({stamps!r}))", sol)


@template(M, "easy")
def date_ranges(rng):
    y, m = 2024, rng.randint(1, 6)
    start = f"{y}-{m:02d}-01"
    end = f"{y}-{m + rng.randint(2, 5):02d}-28"
    n = rng.randint(4, 8)
    desc, sol, req = rng.choice([
        (f"every **month end** from {start} to {end} (`freq='ME'`)", f"pd.date_range({start!r}, {end!r}, freq='ME')", "date_range"),
        (f"every **Monday** from {start} to {end} (`freq='W-MON'`)", f"pd.date_range({start!r}, {end!r}, freq='W-MON')", "date_range"),
        (f"the first {n} **business days** starting {start} (`pd.bdate_range` or `freq='B'`)",
         f"pd.bdate_range({start!r}, periods={n})", "bdate_range|date_range"),
        (f"{n} timestamps every **15 minutes** starting {start} 00:00 (`freq='15min'`)",
         f"pd.date_range({start!r}, periods={n}, freq='15min')", "date_range"),
        (f"every **month start** from {start} to {end} (`freq='MS'`)", f"pd.date_range({start!r}, {end!r}, freq='MS')", "date_range"),
    ])
    return P(f"Build a DatetimeIndex of {desc}. (pandas 3 aliases: `ME`, `MS`, `h`, `min`, `s`.)", "", sol, require=[req])


@template(M, "easy")
def round_and_format(rng):
    stamps = _stamps(rng, rng.randint(3, 5))
    desc, sol = rng.choice([
        ("floored to the **hour** (`dt.floor('h')`)", "s.dt.floor('h')"),
        ("with the time dropped (midnight of the same day, `dt.normalize()`)", "s.dt.normalize()"),
        ("formatted as strings like `'2024-03-07 14:05'` (`dt.strftime('%Y-%m-%d %H:%M')`)", "s.dt.strftime('%Y-%m-%d %H:%M')"),
        ("as monthly periods (`dt.to_period('M')`)", "s.dt.to_period('M')"),
        ("rounded to the nearest **15 minutes** (`dt.round('15min')`)", "s.dt.round('15min')"),
    ])
    return P(f"`s` holds log timestamps. Return them {desc}.", f"s = pd.to_datetime(pd.Series({stamps!r}))", sol,
             require=["dt"])


# ================================================================= medium

@template(M, "medium")
def resample_series(rng):
    setup, start, days = _daily(rng)
    desc, sol = rng.choice([
        ("weekly totals (`resample('W').sum()`)", "sales.resample('W').sum()"),
        ("month-end means (`resample('ME').mean()`)", "sales.resample('ME').mean()"),
        ("the first value of each month, labelled by month start (`resample('MS').first()`)", "sales.resample('MS').first()"),
        ("the maximum of each 10-day block (`resample('10D').max()`)", "sales.resample('10D').max()"),
    ])
    return P(f"`sales` holds daily sales for {days} days. Downsample it to {desc}.", setup, sol, require=["resample"])


@template(M, "medium")
def date_slicing(rng):
    setup, start, days = _daily(rng, 60)
    y, m = start[:4], int(start[5:7])
    a = rng.randint(3, 10)
    b = a + rng.randint(3, 6)
    if rng.random() < 0.5:
        return P(f"`sales` holds daily sales over 60 days from {start}. Select **all of month {y}-{m + 1:02d}** with "
                 f"partial-string indexing (`sales.loc['{y}-{m + 1:02d}']`).", setup, f"sales.loc['{y}-{m + 1:02d}']",
                 require=["loc"])
    return P(f"`sales` holds daily sales over 60 days from {start}. Select {y}-{m:02d}-{a:02d} through "
             f"{y}-{m:02d}-{b:02d}, **both included**, with a date-string slice.", setup,
             f"sales.loc['{y}-{m:02d}-{a:02d}':'{y}-{m:02d}-{b:02d}']", require=["loc"])


@template(M, "medium")
def lags_and_changes(rng):
    setup, start, days = _daily(rng)
    desc, sol = rng.choice([
        ("yesterday's value for every day (a **lag** of 1, `shift(1)`)", "sales.shift(1)"),
        ("tomorrow's value for every day (a **lead**, `shift(-1)`)", "sales.shift(-1)"),
        ("the change versus the **same weekday last week** (`diff(7)`)", "sales.diff(7)"),
        ("the percentage change from the previous day (`pct_change()`)", "sales.pct_change()"),
    ])
    return P(f"`sales` holds daily sales. Compute {desc}.", setup, sol)


@template(M, "medium")
def time_grouper(rng):
    n = rng.randint(8, 12)
    days = sorted(_stamps(rng, n, False))
    vals = [rng.randint(1, 50) for _ in range(n)]
    freq, word = rng.choice([("ME", "month"), ("QE", "quarter"), ("W", "week")])
    return P(f"`df` has one row per order with a `date` column and an `amount`. Total the amounts per **{word}** by "
             f"grouping on `pd.Grouper(key='date', freq={freq!r})`.",
             f"df = pd.DataFrame({{'date': pd.to_datetime({days!r}), 'amount': {vals!r}}})",
             f"df.groupby(pd.Grouper(key='date', freq={freq!r}))['amount'].sum()", require=["Grouper|resample"])


# ================================================================= hard

@template(M, "hard")
def date_features(rng):
    stamps = _stamps(rng, rng.randint(4, 6))
    return P("`df` has one row per user event with a `ts` timestamp. Add these model features and return `df`: "
             "`weekday` (the day name), `is_weekend` (Saturday or Sunday), `hour`, and `days_since_first` (whole days "
             "since the earliest event).", f"df = pd.DataFrame({{'ts': pd.to_datetime({stamps!r})}})",
             "df['weekday'] = df['ts'].dt.day_name()\ndf['is_weekend'] = df['ts'].dt.dayofweek >= 5\n"
             "df['hour'] = df['ts'].dt.hour\ndf['days_since_first'] = (df['ts'] - df['ts'].min()).dt.days\nresult = df",
             require=["dt"])


@template(M, "hard")
def weekly_report(rng):
    setup, start, days = _daily(rng, rng.choice([42, 56, 63]))
    return P("`sales` holds daily sales. Build a weekly report DataFrame with columns `total` (weekly sums, weeks "
             "ending on Sunday: `resample('W')`) and `wow` (week-over-week percentage change of `total`).", setup,
             "weekly = sales.resample('W').sum()\nresult = pd.DataFrame({'total': weekly, 'wow': weekly.pct_change()})",
             require=["resample"])


@template(M, "hard")
def upsample(rng):
    n = rng.randint(3, 4)
    start = f"2024-0{rng.randint(1, 9)}-0{rng.randint(1, 5)}"
    vals = [rng.randint(10, 99) for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"`daily` holds one reading per day for {n} days. **Upsample** it to every 6 hours with "
                 f"`resample('6h')`, carrying each day's value forward (`ffill`).",
                 f"daily = pd.Series({vals!r}, index=pd.date_range({start!r}, periods={n}, freq='D'))",
                 "daily.resample('6h').ffill()", require=["resample|asfreq"])
    return P(f"`daily` holds one reading per day for {n} days. Upsample it to every 6 hours and **linearly "
             f"interpolate** between the daily readings (`resample('6h').interpolate()`).",
             f"daily = pd.Series({vals!r}, index=pd.date_range({start!r}, periods={n}, freq='D'), dtype=float)",
             "daily.resample('6h').interpolate()", require=["resample|asfreq"])


# ================================================================= challenge

@template(M, "challenge")
def sessionize(rng):
    n = rng.randint(7, 10)
    users = [rng.choice(["u1", "u2", "u3"]) for _ in range(n)]
    mins = [rng.randint(0, 200) for _ in range(n)]
    sol = ("d = events.sort_values(['user', 'ts'])\nresult = {}\nprev_user, prev_ts = None, None\n"
           "for user, ts in zip(d['user'], d['ts']):\n    if user != prev_user or ts - prev_ts > pd.Timedelta(minutes=30):\n"
           "        result[user] = result.get(user, 0) + 1\n    prev_user, prev_ts = user, ts")

    def big(seed, n_, k, span):
        return (f"g = np.random.default_rng({seed})\nevents = pd.DataFrame({{'user': [f'u{{i}}' for i in g.integers(0, {k}, {n_})], "
                f"'ts': pd.Timestamp('2024-01-01') + pd.to_timedelta(g.integers(0, {span}, {n_}), unit='min')}})")

    tests = [big(seeded(rng), 20000, 300, 10**5), big(seeded(rng), 20000, 2, 10**6),
             "events = pd.DataFrame({'user': ['a', 'a'], 'ts': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30'])})"]
    return challenge("Sessionize the clickstream",
                     "`events` holds `(user, ts)` clicks in **no particular order**. A user's session ends when they "
                     "are idle for **more than 30 minutes**: a click more than 30 minutes after that user's previous "
                     "click starts a new session (exactly 30 minutes still continues it). Return a dict mapping every "
                     "user to their number of sessions.",
                     f"events = pd.DataFrame({{'user': {users!r}, 'ts': pd.Timestamp('2024-01-01') + "
                     f"pd.to_timedelta({mins!r}, unit='min')}})", sol, tests, "≤ 2·10⁴ events",
                     speed="O(n log n): sort by (user, ts), then compare neighbours (vectorized `diff` works too)")


@template(M, "challenge")
def longest_daily_streak(rng):
    n = rng.randint(7, 10)
    users = [rng.choice(["ana", "ben", "cal"]) for _ in range(n)]
    days = [f"2024-03-{rng.randint(1, 9):02d}" for _ in range(n)]
    sol = ("best = {}\nfor user, dates in log.groupby('user')['day']:\n    ds = sorted(set(dates.dt.normalize()))\n"
           "    run = top = 1\n    for a, b in zip(ds, ds[1:]):\n        run = run + 1 if b - a == pd.Timedelta(days=1) else 1\n"
           "        top = max(top, run)\n    best[user] = top\nlength = max(best.values())\n"
           "result = (min(u for u, v in best.items() if v == length), length)")

    def big(seed, n_, k, span):
        return (f"g = np.random.default_rng({seed})\nlog = pd.DataFrame({{'user': [f'u{{i:04d}}' for i in g.integers(0, {k}, {n_})], "
                f"'day': pd.Timestamp('2024-01-01') + pd.to_timedelta(g.integers(0, {span}, {n_}), unit='D')}})")

    tests = [big(seeded(rng), 20000, 100, 365), big(seeded(rng), 20000, 5000, 30),
             "log = pd.DataFrame({'user': ['z'], 'day': pd.to_datetime(['2024-02-28'])})"]
    return challenge("Longest daily streak",
                     "`log` lists the days on which users were active (a user can appear several times on the same "
                     "day; rows are unsorted). A **streak** is a run of consecutive calendar days with activity. Return "
                     "the tuple `(user, length)` for the longest streak of any user; if several users tie, the "
                     "alphabetically smallest user.",
                     f"log = pd.DataFrame({{'user': {users!r}, 'day': pd.to_datetime({days!r})}})", sol, tests,
                     "≤ 2·10⁴ rows", speed="O(n log n): deduplicate and sort each user's days")

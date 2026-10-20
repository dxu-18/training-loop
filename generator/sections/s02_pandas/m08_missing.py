"""2.08 Missing data."""

from ._common import P, challenge, floats, frame, lit, seeded, template

M = "2.08"


def _gappy(rng, n=(7, 10), k=None):
    vals = floats(rng, rng.randint(*n), 0, 50, 1)
    for i in rng.sample(range(1, len(vals) - 1), k or rng.randint(2, 3)):
        vals[i] = float("nan")
    return vals


# ================================================================= lite

@template(M, "lite")
def detect_missing(rng):
    f = frame(rng, (6, 9), nums=2, nan=rng.randint(1, 3))
    a = f.num[0]
    desc, sol = rng.choice([
        (f"how many `{a}` values are missing", f"df[{a!r}].isna().sum()"),
        (f"the rows where `{a}` is missing", f"df[df[{a!r}].isna()]"),
        (f"whether the `{a}` column has any missing value at all (`hasnans`)", f"df[{a!r}].hasnans"),
        ("a same-shape boolean frame marking every missing cell", "df.isna()"),
    ])
    return P(f"{f.describe()}; some `{a}` readings are missing (NaN). Get {desc}.", f.code(), sol,
             require=["isna|isnull|hasnans|notna"])


@template(M, "lite")
def dropna_basic(rng):
    f = frame(rng, (6, 9), nums=2, nan=rng.randint(1, 3))
    a = f.num[0]
    desc, sol = rng.choice([
        ("every row that has a missing value", "df.dropna()"),
        (f"the rows where `{a}` is missing (`subset=`)", f"df.dropna(subset=[{a!r}])"),
        ("every **column** that has a missing value (`axis=1`)", "df.dropna(axis=1)"),
    ])
    return P(f"{f.describe()}; some `{a}` values are missing. Drop {desc}.", f.code(), sol, require=["dropna"])


# ================================================================= easy

@template(M, "easy")
def fill_values(rng):
    f = frame(rng, (6, 9), nums=2, nan=rng.randint(1, 3))
    a = f.num[0]
    v = rng.choice([0, -1])
    desc, sol = rng.choice([
        (f"{v}", f"df[{a!r}].fillna({v})"),
        (f"the column's **mean**", f"df[{a!r}].fillna(df[{a!r}].mean())"),
        (f"the column's **median**", f"df[{a!r}].fillna(df[{a!r}].median())"),
    ])
    if rng.random() < 0.3:
        return P(f"{f.describe()}; some `{a}` values are missing. Fill them with the median of `{a}` using "
                 f"`df.fillna({{column: value}})` and return the whole frame.", f.code(),
                 f"df.fillna({{{a!r}: df[{a!r}].median()}})", require=["fillna"])
    return P(f"{f.describe()}; some `{a}` values are missing. Return the `{a}` column with the gaps filled with {desc}.",
             f.code(), sol, require=["fillna"])


@template(M, "easy")
def carry_values(rng):
    vals = _gappy(rng)
    what = rng.choice(["GPU temperature", "heart rate", "stock of an item", "queue length"])
    desc, sol, req = rng.choice([
        ("carry the **last** known value forward (`ffill`)", "s.ffill()", "ffill"),
        ("fill each gap with the **next** known value (`bfill`)", "s.bfill()", "bfill"),
        ("carry the last value forward, but at most **1** step (`ffill(limit=1)`)", "s.ffill(limit=1)", "ffill"),
        ("fill the gaps by **linear interpolation** between neighbours", "s.interpolate()", "interpolate"),
    ])
    return P(f"`s` is a sensor log of {what} sampled at regular intervals, with dropouts (NaN). {desc[0].upper() + desc[1:]}.",
             f"s = pd.Series({lit(vals)})", sol, require=[req])


@template(M, "easy")
def replace_sentinels(rng):
    n = rng.randint(5, 7)
    vals = [rng.randint(1, 99) for _ in range(n)]
    variant = rng.choice(["number", "string", "blank"])
    if variant == "number":
        for i in rng.sample(range(n), 2):
            vals[i] = -999
        return P("A legacy system writes `-999` for missing readings. Replace every `-999` in `df` with NaN "
                 "(`replace`).", f"df = pd.DataFrame({{'reading': {vals!r}}})", "df.replace(-999, np.nan)",
                 require=["replace|mask|where"])
    words = [rng.choice(["ok", "late", "lost"]) for _ in range(n)]
    if variant == "string":
        for i in rng.sample(range(n), 2):
            words[i] = "?"
        return P("In `df['status']`, `'?'` means unknown. Replace it with NaN **in that column only** using a nested "
                 "dict in `df.replace({'status': {...}})`.", f"df = pd.DataFrame({{'status': {words!r}, 'n': {vals!r}}})",
                 "df.replace({'status': {'?': np.nan}})", require=["replace"])
    for i in rng.sample(range(n), 2):
        words[i] = rng.choice(["", "  ", " "])
    return P("Some `status` cells are empty or only spaces. Replace them with NaN using a **regex** "
             "(`df.replace(r'^\\s*$', np.nan, regex=True)`).", f"df = pd.DataFrame({{'status': {words!r}}})",
             "df.replace(r'^\\s*$', np.nan, regex=True)", require=["replace"])


# ================================================================= medium

@template(M, "medium")
def group_fill(rng):
    f = frame(rng, (8, 11), nums=1, nan=rng.randint(2, 3))
    a = f.num[0]
    stat = rng.choice(["mean", "median"])
    return P(f"{f.describe()}; some `{a}` values are missing. Fill each gap with the **{stat} of its own `{f.group}`** "
             f"(`groupby(...)[...].transform(...)`), and return the filled `{a}` column.", f.code(),
             f"df[{a!r}].fillna(df.groupby({f.group!r})[{a!r}].transform({stat!r}))", require=["transform"])


@template(M, "medium")
def dropna_thresholds(rng):
    f = frame(rng, (7, 10), nums=3, nan=0)
    for col in f.num:
        vals = [float(v) for v in f.cols[col]]
        for i in rng.sample(range(f.n), rng.randint(1, 3)):
            vals[i] = float("nan")
        f.cols[col] = vals
    k = rng.randint(len(f.cols) - 2, len(f.cols) - 1)
    a = f.num[0]
    if rng.random() < 0.5:
        return P(f"{f.describe()}; values are missing here and there. Keep only the rows with **at least {k} non-missing "
                 f"values** (`dropna(thresh={k})`).", f.code(), f"df.dropna(thresh={k})", require=["dropna"])
    return P(f"{f.describe()}; values are missing here and there. Drop the rows where `{a}` is missing and return "
             f"**how many rows were dropped**.", f.code(), f"len(df) - len(df.dropna(subset=[{a!r}]))",
             require=["dropna|isna|notna"])


@template(M, "medium")
def time_interpolation(rng):
    days = sorted(rng.sample(range(1, 28), 6))
    vals = [round(rng.uniform(10, 30), 1) for _ in days]
    for i in rng.sample(range(1, 5), 2):
        vals[i] = float("nan")
    idx = [f"2024-03-{d:02d}" for d in days]
    return P("`temp` is a temperature log at **irregular** dates with gaps. Interpolate the gaps in proportion to the "
             "time between readings (`interpolate(method='time')`; it needs the DatetimeIndex).",
             f"temp = pd.Series({lit(vals)}, index=pd.to_datetime({idx!r}))", "temp.interpolate(method='time')",
             require=["interpolate"])


@template(M, "medium")
def nullable_aggregates(rng):
    n = rng.randint(5, 7)
    vals = [rng.randint(0, 20) for _ in range(n)]
    vals[rng.randrange(n)] = None
    backup = [rng.randint(0, 20) for _ in range(n)]
    desc, setup, sol = rng.choice([
        ("the sum, skipping the missing value (the default)", f"s = pd.Series({vals!r}, dtype='Int64')", "s.sum()"),
        ("the sum **without** skipping missing values (`skipna=False`), which is `<NA>`",
         f"s = pd.Series({vals!r}, dtype='Int64')", "s.sum(skipna=False)"),
        ("`s` with its gaps filled from `backup` at the same positions (`combine_first`)",
         f"s = pd.Series({vals!r}, dtype='Int64')\nbackup = pd.Series({backup!r}, dtype='Int64')", "s.combine_first(backup)"),
    ])
    return P(f"`s` holds nullable integer counts (`Int64`) with one `<NA>`. Compute {desc}.", setup, sol)


# ================================================================= hard

@template(M, "hard")
def impute_pipeline(rng):
    f = frame(rng, (7, 10), nums=2, nan=rng.randint(1, 2))
    a, b = f.num
    g = list(f.cols[f.group])
    for i in rng.sample(range(f.n), 2):
        g[i] = None
    f.cols[f.group] = g
    return P(f"{f.describe()}; `{a}` and `{f.group}` have missing values. Impute and return `df`:\n\n"
             f"1. add a boolean column `{a}_missing` (True where `{a}` was missing), before filling\n"
             f"2. fill `{a}` with its median\n3. fill `{f.group}` with its most common value (`mode()[0]`)", f.code(),
             f"df['{a}_missing'] = df[{a!r}].isna()\ndf[{a!r}] = df[{a!r}].fillna(df[{a!r}].median())\n"
             f"df[{f.group!r}] = df[{f.group!r}].fillna(df[{f.group!r}].mode()[0])\nresult = df")


@template(M, "hard")
def missing_by_group(rng):
    f = frame(rng, (8, 11), nums=1, nan=rng.randint(2, 4))
    a = f.num[0]
    return P(f"{f.describe()}; some `{a}` values are missing. Which `{f.group}` has the worst data? Return the "
             f"**share of missing `{a}` per `{f.group}`** (a Series indexed by `{f.group}`), sorted from highest to "
             f"lowest share.", f.code(),
             f"df[{a!r}].isna().groupby(df[{f.group!r}]).mean().sort_values(ascending=False)", require=["groupby"])


@template(M, "hard")
def combine_sources(rng):
    vals = _gappy(rng, (8, 10), 3)
    backup = [v if rng.random() < 0.5 else float("nan") for v in floats(rng, len(vals), 0, 50, 1)]
    backup[0] = round(rng.uniform(0, 50), 1)
    return P("`primary` and `backup` are two sensors measuring the same thing at the same timestamps, each with gaps. "
             "Build the best series: take `primary`, fill its gaps from `backup` (`combine_first`), then fill whatever "
             "is still missing by carrying the last value forward.",
             f"primary = pd.Series({lit(vals)})\nbackup = pd.Series({lit(backup)})",
             "primary.combine_first(backup).ffill()", require=["combine_first|fillna"])


# ================================================================= challenge

@template(M, "challenge")
def nearest_fill(rng):
    n = rng.randint(7, 10)
    vals = [float(rng.randint(0, 9)) for _ in range(n)]
    for i in rng.sample(range(n), rng.randint(2, 4)):
        vals[i] = float("nan")
    vals[rng.randrange(n)] = float(rng.randint(0, 9))
    sol = ("v = s.tolist()\nn = len(v)\nprev, nxt = [None] * n, [None] * n\nlast = None\nfor i in range(n):\n"
           "    if v[i] == v[i]:\n        last = i\n    prev[i] = last\nlast = None\nfor i in range(n - 1, -1, -1):\n"
           "    if v[i] == v[i]:\n        last = i\n    nxt[i] = last\nout = []\nfor i in range(n):\n"
           "    p, q = prev[i], nxt[i]\n    if q is None or (p is not None and i - p <= q - i):\n        out.append(v[p])\n"
           "    else:\n        out.append(v[q])\nresult = pd.Series(out)")

    def big(seed, n_, p):
        return f"g = np.random.default_rng({seed})\nv = g.integers(0, 1000, {n_}).astype(float)\nv[g.random({n_}) < {p}] = np.nan\nv[0] = 7.0\ns = pd.Series(v)"

    tests = [big(seeded(rng), 20000, 0.5), big(seeded(rng), 20000, 0.99), "s = pd.Series([np.nan, np.nan, 4.0])"]
    return challenge("Fill from the nearest reading",
                     "`s` is a sensor Series with gaps (NaN); at least one value is present. Replace every NaN with the "
                     "value of the **nearest non-missing position** (by index distance); if two are equally near, use "
                     "the **earlier** one. Return the filled Series.", f"s = pd.Series({lit(vals)})", sol, tests,
                     "1 ≤ n ≤ 2·10⁴", answer_desc=" (a Series)", speed="O(n) with two passes")


@template(M, "challenge")
def largest_complete_block(rng):
    r, c = rng.randint(4, 6), rng.randint(4, 6)
    grid = [[round(rng.uniform(0, 9), 1) if rng.random() < 0.75 else float("nan") for _ in range(c)] for _ in range(r)]
    sol = ("ok = df.notna().to_numpy().tolist()\nm = len(ok[0])\nheights = [0] * m\nbest = 0\nfor row in ok:\n"
           "    heights = [h + 1 if v else 0 for h, v in zip(heights, row)]\n    stack = []\n"
           "    for j in range(m + 1):\n        h = heights[j] if j < m else 0\n        start = j\n"
           "        while stack and stack[-1][1] >= h:\n            start, hh = stack.pop()\n            best = max(best, hh * (j - start))\n"
           "        stack.append((start, h))\nresult = best")

    def big(seed, r_, c_, p):
        return (f"g = np.random.default_rng({seed})\nv = g.random(({r_}, {c_}))\nv[g.random(({r_}, {c_})) < {p}] = np.nan\n"
                f"df = pd.DataFrame(v)")

    tests = [big(seeded(rng), 300, 300, 0.02), big(seeded(rng), 300, 300, 0.5), "df = pd.DataFrame(np.ones((200, 300)))",
             "df = pd.DataFrame([[np.nan]])"]
    return challenge("Largest complete block",
                     "`df` is a sensor × time table with missing readings (NaN). A model needs a **complete** block: a "
                     "rectangle of consecutive rows and consecutive columns with **no** missing value. Return the "
                     "**largest possible number of cells** in such a block (0 if every cell is missing).",
                     f"df = pd.DataFrame({lit(grid)})", sol, tests, "≤ 300 × 300 cells",
                     speed="O(rows · columns): a histogram + monotonic stack per row")

"""2.02 Creating DataFrames & dtypes."""

from ... import themes as T
from ._common import P, challenge, frame, lit, seeded, template

M = "2.02"


# ================================================================= lite

@template(M, "lite")
def frame_from_dict(rng):
    f = frame(rng, (4, 6), nums=2, group=False)
    cols = list(f.cols)
    setup = "\n".join(f"{c} = {lit(v)}" for c, v in f.cols.items())
    order = rng.sample(cols, len(cols)) if rng.random() < 0.4 else cols
    return P(f"You have three parallel Python lists about {f.th.plural}. Build a DataFrame with columns "
             f"{', '.join(f'`{c}`' for c in order)} (in that order) from a **dict of columns**.",
             setup, "pd.DataFrame({" + ", ".join(f"{c!r}: {c}" for c in order) + "})")


@template(M, "lite")
def frame_dtypes(rng):
    f = frame(rng, (4, 6), nums=2, date=rng.random() < 0.5)
    col = rng.choice(list(f.cols))
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Get the **dtype of every column** (a Series indexed by column name).",
                 f.code(), "df.dtypes", require=["dtypes"])
    return P(f"{f.describe()}. Get the **dtype** of the `{col}` column.", f.code(), f"df[{col!r}].dtype",
             require=["dtype|dtypes"])


# ================================================================= easy

@template(M, "easy")
def frame_from_rows(rng):
    th = T.theme(rng)
    names = T.items(rng, th, rng.randint(3, 4))
    m1, m2 = rng.sample(th.measures, 2)
    rows = [(n, T.value(rng, m1), T.value(rng, m2)) for n in names]
    variant = rng.choice(["dicts", "tuples", "lists"])
    if variant == "dicts":
        recs = [{"name": n, m1.var: a, m2.var: b} for n, a, b in rows]
        del recs[rng.randrange(len(recs))][m2.var]  # one record lacks a field -> NaN
        return P(f"`records` is a list of row dicts parsed from JSON (one record is missing `{m2.var}`). Build a "
                 f"DataFrame from it; the missing field becomes NaN.", f"records = {recs!r}", "pd.DataFrame(records)")
    if variant == "tuples":
        return P(f"`rows` is a list of `(name, {m1.var}, {m2.var})` tuples from a database cursor. Build a DataFrame "
                 f"with those column names using `pd.DataFrame.from_records` (or `pd.DataFrame(..., columns=...)`).",
                 f"rows = {rows!r}", f"pd.DataFrame.from_records(rows, columns=['name', {m1.var!r}, {m2.var!r}])")
    return P(f"`rows` is a list of `[{m1.var}, {m2.var}]` rows and `names` the matching {th.noun} names. Build a "
             f"DataFrame with columns `{m1.var}`, `{m2.var}` and `names` as the **index**.",
             f"rows = {[[a, b] for _, a, b in rows]!r}\nnames = {names!r}",
             f"pd.DataFrame(rows, columns=[{m1.var!r}, {m2.var!r}], index=names)")


@template(M, "easy")
def astype_columns(rng):
    f = frame(rng, (4, 6), nums=2)
    a, b = f.num
    ints = [c for c in (a, b) if all(float(v).is_integer() for v in f.cols[c])]
    desc, sol = rng.choice([
        (f"`{f.group}` to `category` (saves memory for repeated strings)", f"df.astype({{{f.group!r}: 'category'}})"),
        (f"`{a}` and `{b}` to `float32`", f"df.astype({{{a!r}: 'float32', {b!r}: 'float32'}})"),
    ] + ([(f"`{ints[0]}` to `int32` and `{f.group}` to `category`",
           f"df.astype({{{ints[0]!r}: 'int32', {f.group!r}: 'category'}})")] if ints else []))
    return P(f"{f.describe()}. With one `astype` call (pass a dict), convert {desc}. Return the new frame.",
             f.code(), sol, require=["astype"], check_dtype=True)


@template(M, "easy")
def nullable_dtypes(rng):
    n = rng.randint(4, 6)
    vals = [rng.randint(0, 40) for _ in range(n)]
    vals[rng.randrange(n)] = None
    what = rng.choice(["ages", "token counts", "retries", "GPU counts"])
    variant = rng.choice(["int", "sum", "bool", "string"])
    if variant == "int":
        return P(f"`raw` holds {what} with one missing value (`None`). Store them as a Series **without turning "
                 f"them into floats**: use the nullable integer dtype `'Int64'` (capital I).", f"raw = {vals!r}",
                 "pd.Series(raw, dtype='Int64')", check_dtype=True)
    if variant == "sum":
        return P(f"`raw` holds {what} with one missing value. Put them in an `'Int64'` Series and return its sum "
                 f"(aggregations skip `pd.NA` by default).", f"raw = {vals!r}", "pd.Series(raw, dtype='Int64').sum()")
    if variant == "bool":
        flags = [rng.choice([True, False]) for _ in range(n)]
        flags[rng.randrange(n)] = None
        return P("`raw` holds yes/no flags with one unknown (`None`). Store them as a Series with the nullable "
                 "`'boolean'` dtype.", f"raw = {flags!r}", "pd.Series(raw, dtype='boolean')", check_dtype=True)
    words = rng.sample(["spam", "ham", "eggs", "toast", "jam"], n - 1) + [None]
    return P("`raw` holds labels with one missing value. Store them as a Series with the nullable `'string'` "
             "dtype.", f"raw = {words!r}", "pd.Series(raw, dtype='string')", check_dtype=True)


# ================================================================= medium

@template(M, "medium")
def nested_dict_frame(rng):
    th = T.theme(rng)
    names = T.items(rng, th, rng.randint(3, 4))
    m1, m2 = rng.sample(th.measures, 2)
    if rng.random() < 0.5:
        d = {n: {m1.var: T.value(rng, m1), m2.var: T.value(rng, m2)} for n in names}
        return P(f"`stats` maps each {th.noun} to a dict of its metrics. Build a DataFrame with **one row per "
                 f"{th.noun}** using `pd.DataFrame.from_dict(..., orient='index')`.", f"stats = {d!r}",
                 "pd.DataFrame.from_dict(stats, orient='index')", require=["from_dict"])
    s = seeded(rng)
    return P(f"`X` is a NumPy feature matrix with one row per {th.noun}. Wrap it in a DataFrame with columns "
             f"`{m1.var}` and `{m2.var}` and the {th.noun} names `names` as the index.",
             f"X = np.random.default_rng({s}).random(({len(names)}, 2)).round(3)\nnames = {names!r}",
             f"pd.DataFrame(X, columns=[{m1.var!r}, {m2.var!r}], index=names)")


@template(M, "medium")
def convert_and_infer(rng):
    n = rng.randint(4, 6)
    setup = (f"df = pd.DataFrame({{\n    'id': {list(range(1, n + 1))!r},\n"
             f"    'score': {[None if i == 1 else round(rng.uniform(0, 1), 2) for i in range(n)]!r},\n"
             f"    'tag': {[None if i == 2 else rng.choice(['a', 'b']) for i in range(n)]!r},\n"
             f"    'ok': {[None if i == 0 else rng.choice([True, False]) for i in range(n)]!r},\n}})")
    if rng.random() < 0.5:
        return P("`df` has missing values in several columns. Let pandas pick the best **nullable** dtype for every "
                 "column with `convert_dtypes()`, and return the resulting `dtypes`.", setup,
                 "df.convert_dtypes().dtypes", require=["convert_dtypes"])
    return P("`df` has missing values in several columns. Convert it with `convert_dtypes()` (nullable dtypes) and "
             "return the converted frame.", setup, "df.convert_dtypes()", require=["convert_dtypes"], check_dtype=True)


@template(M, "medium")
def to_numeric_coerce(rng):
    n = rng.randint(5, 7)
    good = [str(rng.randint(1, 500)) for _ in range(n)]
    for i in rng.sample(range(n), 2):
        good[i] = rng.choice(["n/a", "?", "error", "", "twelve"])
    what = rng.choice(["latencies (ms)", "token counts", "prices", "download counts"])
    if rng.random() < 0.5:
        return P(f"`raw` holds {what} scraped as **strings**, some of them junk. Convert to numbers with "
                 f"`pd.to_numeric(..., errors='coerce')`, so junk becomes NaN.", f"raw = pd.Series({good!r})",
                 "pd.to_numeric(raw, errors='coerce')", require=["to_numeric"])
    return P(f"`raw` holds {what} scraped as **strings**, some of them junk. Convert to numbers (junk → NaN with "
             f"`errors='coerce'`), fill the gaps with 0, and cast to `int64`.", f"raw = pd.Series({good!r})",
             "pd.to_numeric(raw, errors='coerce').fillna(0).astype('int64')", require=["to_numeric"], check_dtype=True)


# ================================================================= hard

@template(M, "hard")
def typed_records(rng):
    n = rng.randint(3, 5)
    recs = []
    for i in range(n):
        r = {"id": str(100 + i), "price": f"{rng.uniform(1, 99):.2f}", "count": str(rng.randint(0, 9)),
             "ts": f"2024-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"}
        recs.append(r)
    del recs[rng.randrange(n)]["count"]
    return P("`records` came from a JSON API that sends **every field as a string**, and one record has no `count`. "
             "Build a DataFrame and fix the types: `id` → `int64`, `price` → `float64`, `count` → nullable `'Int64'` "
             "(keep the missing one as `<NA>`; go through `pd.to_numeric` first), `ts` → datetime with "
             "`pd.to_datetime`. Keep the column order `id, price, count, ts`.", f"records = {recs!r}",
             "df = pd.DataFrame(records)[['id', 'price', 'count', 'ts']]\n"
             "df['id'] = df['id'].astype('int64')\ndf['price'] = df['price'].astype('float64')\n"
             "df['count'] = pd.to_numeric(df['count']).astype('Int64')\ndf['ts'] = pd.to_datetime(df['ts'])\nresult = df",
             check_dtype=True)


@template(M, "hard")
def frame_from_arrays(rng):
    n, d = rng.randint(3, 5), rng.randint(2, 3)
    th = T.theme(rng)
    names = T.items(rng, th, n)
    cols = [m.var for m in rng.sample(th.measures, d)]
    s = seeded(rng)
    return P(f"`X` is a `({n}, {d})` float64 NumPy array of features for the {th.plural} in `ids`. Build a "
             f"DataFrame with columns `cols`, `ids` as the index (named `{T.ident(th.noun)}`), every column cast to "
             f"`float32`, and an extra int column `n_missing` that counts the NaNs in each row (there are a few).",
             f"g = np.random.default_rng({s})\nX = g.random(({n}, {d})).round(3)\nX[g.integers(0, {n}, 2), g.integers(0, {d}, 2)] = np.nan\n"
             f"ids = {names!r}\ncols = {cols!r}",
             f"df = pd.DataFrame(X, columns=cols, index=pd.Index(ids, name={T.ident(th.noun)!r})).astype('float32')\n"
             "df['n_missing'] = df.isna().sum(axis=1)\nresult = df",
             post="{'index name': _.index.name, 'frame': _}", check_dtype=True)


# ================================================================= challenge

@template(M, "challenge")
def sparse_to_dense(rng):
    k = rng.randint(6, 9)
    rows = [(rng.randint(1, 3), rng.choice(["f1", "f2", "f3"]), rng.randint(0, 9)) for _ in range(k)]
    sol = ("table = {}\nfor r, c, v in updates.itertuples(index=False):\n    table[(r, c)] = v\n"
           "rows = sorted({r for r, _ in table})\ncols = sorted({c for _, c in table})\n"
           "result = pd.DataFrame([[table.get((r, c), np.nan) for c in cols] for r in rows], index=rows, columns=cols, dtype=float)")

    def big(seed, n_, nrows, ncols):
        return (f"g = np.random.default_rng({seed})\nupdates = pd.DataFrame({{'row': g.integers(0, {nrows}, {n_}), "
                f"'col': [f'f{{i}}' for i in g.integers(0, {ncols}, {n_})], 'value': g.integers(-99, 100, {n_})}})")

    tests = [big(seeded(rng), 20000, 500, 20), big(seeded(rng), 20000, 20000, 3), "updates = pd.DataFrame({'row': [5, 5], 'col': ['b', 'b'], 'value': [1, 2]})"]
    return challenge("Replay the update log",
                     "`updates` is an append-only log of cell writes `(row, col, value)` to a feature store; a later "
                     "write to the same cell **overwrites** earlier ones. Rebuild the current table as a float "
                     "DataFrame: one row per row id (sorted ascending), one column per column name (sorted as "
                     "strings), NaN for cells never written.",
                     f"updates = pd.DataFrame({rows!r}, columns=['row', 'col', 'value'])", sol, tests,
                     "≤ 2·10⁴ updates", answer_desc=" (a DataFrame)",
                     speed="O(n log n), e.g. `drop_duplicates(keep='last')` + `pivot`")


@template(M, "challenge")
def na_gap_budget(rng):
    n = rng.randint(9, 12)
    vals = [float(rng.randint(0, 9)) for _ in range(n)]
    for i in rng.sample(range(n), 2):
        vals[i] = float("nan")
    S_ = rng.randint(8, 25)
    sol = ("best = 0\nleft = 0\ntotal = 0.0\nvals = x.tolist()\nfor right, v in enumerate(vals):\n"
           "    if v != v:  # NaN breaks every window\n        left, total = right + 1, 0.0\n        continue\n"
           "    total += v\n    while total > S:\n        total -= vals[left]\n        left += 1\n"
           "    best = max(best, right - left + 1)\nresult = best")

    def big(seed, n_, p_nan, S_v):
        return (f"g = np.random.default_rng({seed})\nv = g.integers(0, 100, {n_}).astype(float)\n"
                f"v[g.random({n_}) < {p_nan}] = np.nan\nx = pd.Series(v)\nS = {S_v}")

    tests = [big(seeded(rng), 20000, 0.01, 5000), big(seeded(rng), 20000, 0.3, 300), big(seeded(rng), 20000, 0.0, 10**7),
             "x = pd.Series([np.nan, np.nan])\nS = 5"]
    return challenge("Longest clean stretch under budget",
                     "`x` is a Series of non-negative per-second power readings with **missing values (NaN)** where "
                     "the sensor dropped out. Find the length of the longest **contiguous stretch without any NaN** "
                     "whose total is at most `S` (0 if no single reading fits).",
                     f"x = pd.Series({lit(vals)})\nS = {S_}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ x[i] ≤ 100",
                     speed="O(n) with two pointers that restart after every NaN")

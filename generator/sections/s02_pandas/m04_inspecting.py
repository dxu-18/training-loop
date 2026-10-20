"""2.04 Inspecting data."""

from ._common import P, challenge, frame, lit, seeded, template

M = "2.04"


# ================================================================= lite

@template(M, "lite")
def shape_and_size(rng):
    f = frame(rng, (5, 9), nums=rng.randint(1, 3), date=rng.random() < 0.3)
    desc, sol = rng.choice([
        ("its shape `(rows, columns)`", "df.shape"), ("the number of rows", "len(df)"),
        ("the total number of cells (`size`)", "df.size"), ("the column names as a list", "df.columns.tolist()"),
        ("the number of columns", "df.shape[1]"), ("whether it is empty (`empty`)", "df.empty"),
    ])
    return P(f"{f.describe()}. Get {desc}.", f.code(), sol)


@template(M, "lite")
def head_tail_sample(rng):
    f = frame(rng, (7, 10), nums=1)
    k = rng.randint(2, 4)
    s = rng.randint(0, 99)
    desc, sol, req = rng.choice([
        (f"the first {k} rows (`head`)", f"df.head({k})", "head|iloc"),
        (f"the last {k} rows (`tail`)", f"df.tail({k})", "tail|iloc"),
        (f"a reproducible random sample of {k} rows (`sample` with `random_state={s}`)", f"df.sample({k}, random_state={s})", "sample"),
        (f"a reproducible random half of the rows (`sample(frac=0.5, random_state={s})`)", f"df.sample(frac=0.5, random_state={s})", "sample"),
    ])
    return P(f"{f.describe()}. Take a first look: get {desc}.", f.code(), sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def value_counts_basic(rng):
    f = frame(rng, (6, 10), nums=1)
    g = f.group
    desc, sol, req = rng.choice([
        (f"how many rows have each `{g}` value (`value_counts`)", f"df[{g!r}].value_counts()", "value_counts"),
        (f"the **share** of rows for each `{g}` value (`value_counts(normalize=True)`)",
         f"df[{g!r}].value_counts(normalize=True)", "value_counts"),
        (f"the number of distinct `{g}` values (`nunique`)", f"df[{g!r}].nunique()", "nunique|unique"),
        ("the number of distinct values in **every** column (`df.nunique()`)", "df.nunique()", "nunique"),
    ])
    return P(f"{f.describe()}. Get {desc}.", f.code(), sol, require=[req])


@template(M, "easy")
def missing_summary(rng):
    f = frame(rng, (6, 9), nums=2, nan=rng.randint(1, 3))
    desc, sol = rng.choice([
        ("the number of missing values in **each** column", "df.isna().sum()"),
        ("the **share** of missing values in each column, rounded to 2 decimals", "df.isna().mean().round(2)"),
        ("the number of rows that contain **at least one** missing value", "df.isna().any(axis=1).sum()"),
        ("a boolean per column: does it have **no** missing values?", "df.notna().all()"),
    ])
    return P(f"{f.describe()}; some `{f.num[0]}` values are missing. Compute {desc}.", f.code(), sol,
             require=["isna|isnull|notna|notnull|count"])


@template(M, "easy")
def column_stats(rng):
    f = frame(rng, (6, 9), nums=2)
    a = f.num[0]
    desc, sol, req = rng.choice([
        (f"the min, max and mean of `{a}` in one call (`agg([...])`)", f"df[{a!r}].agg(['min', 'max', 'mean'])", "agg"),
        (f"the 25th, 50th and 75th percentiles of `{a}` (`quantile([...])`)", f"df[{a!r}].quantile([0.25, 0.5, 0.75])", "quantile"),
        ("summary statistics of every numeric column (`describe`)", "df.describe()", "describe"),
        (f"the median of `{a}`", f"df[{a!r}].median()", "median|quantile|describe"),
    ])
    return P(f"{f.describe()}. Compute {desc}.", f.code(), sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def extreme_rows(rng):
    f = frame(rng, (6, 9), nums=2, distinct=True)
    a = f.num[0]
    hi = rng.random() < 0.5
    fn = "idxmax" if hi else "idxmin"
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Return the `{f.key}` of the row with the {'highest' if hi else 'lowest'} `{a}` "
                 f"(`{fn}` gives that row's index label; look it up with `.loc`).", f.code(),
                 f"df.loc[df[{a!r}].{fn}(), {f.key!r}]", require=[fn])
    k = rng.randint(2, 3)
    big = "nlargest" if hi else "nsmallest"
    return P(f"{f.describe()}. Return the {k} rows with the {'highest' if hi else 'lowest'} `{a}` (`{big}`).",
             f.code(), f"df.{big}({k}, {a!r})", require=[big])


@template(M, "medium")
def combination_counts(rng):
    f = frame(rng, (8, 12), nums=1)
    a = f.num[0]
    variant = rng.choice(["pairs", "bins", "top"])
    if variant == "pairs":
        f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
        return P(f"{f.describe()} and `flag`. Count the rows for every **(`{f.group}`, `flag`) combination** "
                 f"(`df.value_counts([...])`).", f.code(), f"df.value_counts([{f.group!r}, 'flag'])", require=["value_counts"])
    if variant == "bins":
        k = rng.randint(2, 4)
        return P(f"{f.describe()}. Count how many `{a}` values fall in each of {k} equal-width bins, in bin order "
                 f"(`value_counts(bins={k}, sort=False)`).", f.code(), f"df[{a!r}].value_counts(bins={k}, sort=False)",
                 require=["value_counts|cut"])
    return P(f"{f.describe()}. Return the **most common** `{f.group}` value (`mode()` or the first label of "
             f"`value_counts()`); if several tie, the alphabetically first.", f.code(),
             f"df[{f.group!r}].mode().iloc[0]", require=["mode|value_counts"])


@template(M, "medium")
def duplicates_info(rng):
    f = frame(rng, (6, 8), nums=1)
    a = f.num[0]
    for _ in range(2):  # plant duplicates
        i, j = rng.sample(range(f.n), 2)
        for c in f.cols:
            f.cols[c][j] = f.cols[c][i]
    desc, sol = rng.choice([
        ("how many rows are exact duplicates of an earlier row (`duplicated().sum()`)", "df.duplicated().sum()"),
        (f"how many rows repeat an earlier row's `{a}` (`duplicated(subset=[...])`)", f"df.duplicated(subset=[{a!r}]).sum()"),
        ("**all** rows involved in an exact duplicate, including the first copies (`keep=False` as a mask)",
         "df[df.duplicated(keep=False)]"),
    ])
    return P(f"{f.describe()}. A few rows were logged twice. Find {desc}.", f.code(), sol, require=["duplicated"])


@template(M, "medium")
def correlations(rng):
    f = frame(rng, (6, 9), nums=3, group=rng.random() < 0.5)
    a, b = f.num[:2]
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Compute the Pearson **correlation between `{a}` and `{b}`** (one number).",
                 f.code(), f"df[{a!r}].corr(df[{b!r}])", require=["corr"])
    return P(f"{f.describe()}. Compute the **correlation matrix** of the numeric columns "
             f"(`df.corr(numeric_only=True)`).", f.code(), "df.corr(numeric_only=True)", require=["corr"])


# ================================================================= hard

@template(M, "hard")
def quality_report(rng):
    f = frame(rng, (6, 9), nums=2, nan=rng.randint(1, 2))
    i, j = rng.sample(range(f.n), 2)
    for c in f.cols:
        f.cols[c][j] = f.cols[c][i]
    return P(f"{f.describe()}. Build a **data-quality report** dict with keys:\n\n"
             f"* `rows`: number of rows\n* `duplicates`: rows that exactly repeat an earlier row\n"
             f"* `missing`: dict column → number of missing values (every column)\n"
             f"* `top_{f.group}`: the most common `{f.group}` (first in `value_counts()`)", f.code(),
             "result = {'rows': len(df), 'duplicates': int(df.duplicated().sum()),\n"
             "          'missing': df.isna().sum().to_dict(),\n"
             f"          'top_{f.group}': df[{f.group!r}].value_counts().index[0]}}")


@template(M, "hard")
def custom_describe(rng):
    f = frame(rng, (8, 12), nums=2)
    lo, hi = rng.choice([(0.1, 0.9), (0.05, 0.95), (0.25, 0.75)])
    return P(f"{f.describe()}. Build a compact summary: `describe` the numeric columns with the extra percentiles "
             f"{lo} and {hi} (`percentiles=[...]`), **transpose** it so each column is a row, and keep only the "
             f"columns `mean`, `{int(lo * 100)}%` and `{int(hi * 100)}%`.", f.code(),
             f"df.describe(percentiles=[{lo}, {hi}]).T[['mean', '{int(lo * 100)}%', '{int(hi * 100)}%']]",
             require=["describe"])


@template(M, "hard")
def top_shares(rng):
    f = frame(rng, (8, 12), nums=1)
    k = 2
    return P(f"{f.describe()}. Return a dict mapping the {k} most common `{f.group}` values to their **share of "
             f"rows**, rounded to 3 decimals, most common first (`value_counts(normalize=True)`; ties keep pandas' "
             f"order).", f.code(), f"df[{f.group!r}].value_counts(normalize=True).head({k}).round(3).to_dict()",
             require=["value_counts"])


# ================================================================= challenge

@template(M, "challenge")
def kth_pair_distance(rng):
    n = rng.randint(5, 8)
    ages = [rng.randint(18, 60) for _ in range(n)]
    k = rng.randint(1, n * (n - 1) // 2)
    sol = ("vals = sorted(df['age'].tolist())\nn = len(vals)\n\ndef pairs_within(d):\n    count = left = 0\n"
           "    for right in range(n):\n        while vals[right] - vals[left] > d:\n            left += 1\n"
           "        count += right - left\n    return count\n\nlo, hi = 0, vals[-1] - vals[0]\nwhile lo < hi:\n"
           "    mid = (lo + hi) // 2\n    if pairs_within(mid) >= k:\n        hi = mid\n    else:\n        lo = mid + 1\nresult = lo")

    def big(seed, n_, hi, k_):
        return f"df = pd.DataFrame({{'age': np.random.default_rng({seed}).integers(0, {hi}, {n_})}})\nk = {k_}"

    tests = [big(seeded(rng), 20000, 10**9, 12345), big(seeded(rng), 20000, 10**9, 199990000),
             big(seeded(rng), 20000, 50, 10**7), "df = pd.DataFrame({'age': [3, 7]})\nk = 1"]
    return challenge("k-th closest pair",
                     "`df['age']` holds the ages of n users. Among all `n·(n−1)/2` pairs of users, sort the absolute "
                     "age differences and return the **k-th smallest** (1-based).",
                     f"df = pd.DataFrame({{'age': {lit(ages)}}})\nk = {k}", sol, tests, "2 ≤ n ≤ 2·10⁴, 1 ≤ k ≤ n(n−1)/2",
                     speed="O(n log n · log(range)): binary search on the answer + two pointers")


@template(M, "challenge")
def flip_equal_rows(rng):
    r, c = rng.randint(4, 6), rng.randint(3, 4)
    base = [[rng.randint(0, 1) for _ in range(c)] for _ in range(2)]
    rows = []
    for _ in range(r):
        b = rng.choice(base)
        rows.append([1 - v for v in b] if rng.random() < 0.4 else list(b))
    rows[rng.randrange(r)] = [rng.randint(0, 1) for _ in range(c)]
    sol = ("from collections import Counter\nkeys = Counter()\nfor row in flags.itertuples(index=False):\n"
           "    first = row[0]\n    keys[tuple(v ^ first for v in row)] += 1\nresult = max(keys.values())")

    def big(seed, r_, c_, patterns):
        return (f"g = np.random.default_rng({seed})\nbase = g.integers(0, 2, ({patterns}, {c_}))\n"
                f"pick = base[g.integers(0, {patterns}, {r_})]\nflip = g.integers(0, 2, ({r_}, 1))\n"
                f"flags = pd.DataFrame(pick ^ flip, columns=[f'f{{i}}' for i in range({c_})])")

    tests = [big(seeded(rng), 20000, 12, 40), big(seeded(rng), 20000, 20, 20000), big(seeded(rng), 300, 1, 1)]
    return challenge("Equal rows after flipping columns",
                     "`flags` is a 0/1 feature table (rows = users, columns = binary features). You may choose any "
                     "set of columns and **flip** every value in them (0 ↔ 1). Return the **maximum number of rows** "
                     "that can be made all-equal (every value in the row the same) by one such choice.",
                     f"flags = pd.DataFrame({rows!r}, columns={[f'f{i}' for i in range(c)]!r})", sol, tests,
                     "≤ 2·10⁴ rows, ≤ 20 columns", speed="O(rows · columns) with hashing")

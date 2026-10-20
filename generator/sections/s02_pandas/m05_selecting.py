"""2.05 Selecting & filtering."""

from ._common import P, challenge, frame, lit, seeded, template, threshold

M = "2.05"


def _thr(rng, f, col):
    return threshold(rng, f.cols[col], next(m for m in f.nums if m.var == col))


# ================================================================= lite

@template(M, "lite")
def select_columns(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    i = rng.randint(1, f.n - 3)
    desc, sol = rng.choice([
        (f"the `{a}` column as a Series", f"df[{a!r}]"),
        (f"the columns `{f.key}` and `{b}` as a DataFrame (a list of names)", f"df[[{f.key!r}, {b!r}]]"),
        (f"rows {i} through {i + 1} by **position** with a plain slice `df[start:stop]`", f"df[{i}:{i + 2}]"),
        ("the last row as a Series (`iloc[-1]`)", "df.iloc[-1]"),
    ])
    return P(f"{f.describe()}. Select {desc}.", f.code(), sol)


@template(M, "lite")
def loc_and_iloc(rng):
    f = frame(rng, (6, 9), nums=2)
    a, b = f.num
    i = rng.randint(0, f.n - 4)
    j = i + rng.randint(1, 2)
    desc, sol = rng.choice([
        (f"the `{a}` value of the row labelled {i} (`.loc[row, col]`)", f"df.loc[{i}, {a!r}]"),
        (f"rows labelled {i} **through** {j} (label slices **include** the end) and columns `{f.key}`, `{a}`",
         f"df.loc[{i}:{j}, [{f.key!r}, {a!r}]]"),
        (f"rows at positions {i} up to (not including) {j} and the first 2 columns (`.iloc`)", f"df.iloc[{i}:{j}, :2]"),
        (f"the single cell at row position {i}, column position 0, quickly (`.iat`)", f"df.iat[{i}, 0]"),
        (f"the single cell at row label {i}, column `{b}`, quickly (`.at`)", f"df.at[{i}, {b!r}]"),
    ])
    return P(f"{f.describe()}; the index is the default 0..{f.n - 1}. Select {desc}.", f.code(), sol)


# ================================================================= easy

@template(M, "easy")
def mask_rows(rng):
    f = frame(rng, (7, 10), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    lvl = rng.choice(f.levels[:2])
    lo, hi = sorted([_thr(rng, f, b), _thr(rng, f, b)])
    desc, sol = rng.choice([
        (f"whose `{a}` is greater than {t}", f"df[df[{a!r}] > {t}]"),
        (f"whose `{a}` is greater than {t} **and** whose `{f.group}` is `{lvl!r}` (`&`, with parentheses)",
         f"df[(df[{a!r}] > {t}) & (df[{f.group!r}] == {lvl!r})]"),
        (f"whose `{f.group}` is **not** `{f.levels[0]!r}` or `{f.levels[1]!r}` (`~` and `isin`)",
         f"df[~df[{f.group!r}].isin([{f.levels[0]!r}, {f.levels[1]!r}])]"),
        (f"whose `{b}` is between {lo} and {hi}, inclusive (`between`)", f"df[df[{b!r}].between({lo}, {hi})]"),
    ])
    return P(f"{f.describe()}. Select the rows {desc}.", f.code(), sol, no_loops=True)


@template(M, "easy")
def query_filter(rng):
    f = frame(rng, (7, 10), nums=2)
    a = f.num[0]
    t = _thr(rng, f, a)
    lvl = rng.choice(f.levels[:2])
    variant = rng.choice(["and", "var", "in"])
    if variant == "and":
        return P(f"{f.describe()}. With `df.query`, select the rows where `{a}` is greater than {t} **and** `{f.group}` "
                 f"is `{lvl!r}`.", f.code(), f"df.query(\"{a} > {t} and {f.group} == '{lvl}'\")", require=["query"])
    if variant == "var":
        return P(f"{f.describe()}. The threshold is in the Python variable `cutoff`. With `df.query`, select the rows "
                 f"whose `{a}` is at least `cutoff`, referencing the variable with `@`.",
                 f.code() + f"\ncutoff = {t}", f"df.query('{a} >= @cutoff')", require=["query"])
    return P(f"{f.describe()}. With `df.query`, select the rows whose `{f.group}` is in `['{f.levels[0]}', "
             f"'{f.levels[1]}']`.", f.code(), f"df.query(\"{f.group} in ['{f.levels[0]}', '{f.levels[1]}']\")",
             require=["query"])


@template(M, "easy")
def loc_mask_columns(rng):
    f = frame(rng, (7, 10), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. In **one** `.loc` call, select the `{f.key}` and `{b}` of the rows whose `{a}` is "
                 f"greater than {t} (rows by mask, columns by label).", f.code(),
                 f"df.loc[df[{a!r}] > {t}, [{f.key!r}, {b!r}]]", require=["loc"])
    return P(f"{f.describe()}. Return the `{f.key}` values (a list) of the rows whose `{a}` is less than {t}.",
             f.code(), f"df.loc[df[{a!r}] < {t}, {f.key!r}].tolist()", no_loops=True)


# ================================================================= medium

@template(M, "medium")
def string_and_missing_filters(rng):
    f = frame(rng, (7, 10), nums=2, nan=rng.randint(1, 2))
    a = f.num[0]
    letter = f.cols[f.key][0][0]
    frag = f.cols[f.key][1][1:3].lower()
    desc, sol = rng.choice([
        (f"whose `{f.key}` starts with `{letter!r}` (`.str.startswith`)", f"df[df[{f.key!r}].str.startswith({letter!r})]"),
        (f"whose `{f.key}` contains `{frag!r}`, ignoring case (`.str.contains(..., case=False)`)",
         f"df[df[{f.key!r}].str.contains({frag!r}, case=False, regex=False)]"),
        (f"whose `{a}` is missing (`isna`)", f"df[df[{a!r}].isna()]"),
        (f"whose `{a}` is **not** missing", f"df[df[{a!r}].notna()]"),
    ])
    return P(f"{f.describe()}. Select the rows {desc}.", f.code(), sol, no_loops=True)


@template(M, "medium")
def column_helpers(rng):
    f = frame(rng, (4, 6), nums=3, date=rng.random() < 0.5)
    sub = f.num[0].split("_")[0][:3]
    desc, sol, req = rng.choice([
        ("only the **numeric** columns (`select_dtypes(include='number')`)", "df.select_dtypes(include='number')", "select_dtypes"),
        ("every column **except** the numeric ones (`select_dtypes(exclude='number')`)", "df.select_dtypes(exclude='number')", "select_dtypes"),
        (f"the columns whose name contains `{sub!r}` (`filter(like=...)`)", f"df.filter(like={sub!r})", "filter"),
        (f"the columns `{f.key}` and `{f.num[1]}` in that order with `filter(items=[...])`",
         f"df.filter(items=[{f.key!r}, {f.num[1]!r}])", "filter"),
    ])
    return P(f"{f.describe()}. Select {desc}.", f.code(), sol, require=[req])


@template(M, "medium")
def where_and_mask(rng):
    f = frame(rng, (6, 9), nums=2)
    a = f.num[0]
    t = _thr(rng, f, a)
    v = rng.choice([0, -1, 100])
    desc, sol, req = rng.choice([
        (f"keeps `{a}` where it is greater than {t} and puts {v} elsewhere (`Series.where(cond, other=...)`)",
         f"df[{a!r}].where(df[{a!r}] > {t}, other={v})", "where"),
        (f"replaces `{a}` with {v} where it is greater than {t} and keeps it elsewhere (`Series.mask`)",
         f"df[{a!r}].mask(df[{a!r}] > {t}, other={v})", "mask"),
        (f"keeps the whole frame's shape but turns every row with `{a}` ≤ {t} into NaN (`df.where`)",
         f"df.where(df[{a!r}] > {t})", "where"),
    ])
    return P(f"{f.describe()}. Return a result that {desc}.", f.code(), sol, require=[req])


# ================================================================= hard

@template(M, "hard")
def filter_chain(rng):
    f = frame(rng, (8, 11), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    lvls = list(f.levels[:2])
    return P(f"{f.describe()}. In one method chain, keep the rows whose `{f.group}` is in {lvls} and whose `{a}` is "
             f"greater than {t} (use `.loc[lambda d: ...]` so it works mid-chain), sort them by `{b}` descending, and "
             f"return only the `{f.key}` and `{b}` columns.", f.code(),
             f"(df.loc[lambda d: d[{f.group!r}].isin({lvls}) & (d[{a!r}] > {t})]\n"
             f"   .sort_values({b!r}, ascending=False)[[{f.key!r}, {b!r}]])", require=["loc"])


@template(M, "hard")
def best_under_constraint(rng):
    f = frame(rng, (8, 11), nums=2, distinct=True)
    a, b = f.num
    t = _thr(rng, f, b)
    while sum(v < t for v in f.cols[b]) < 1:
        t = _thr(rng, f, b)
    return P(f"{f.describe()}. Among the rows whose `{b}` is below {t}, return the `{f.key}` with the **highest** "
             f"`{a}` (filter, then `nlargest(1, ...)` or `idxmax`).", f.code(),
             f"df.loc[df[{b!r}] < {t}].nlargest(1, {a!r})[{f.key!r}].iloc[0]")


@template(M, "hard")
def query_with_variables(rng):
    f = frame(rng, (8, 11), nums=2)
    a, b = f.num
    lo = _thr(rng, f, a)
    allowed = list(f.levels[:2])
    return P(f"{f.describe()}. The filter settings live in Python variables `min_{a}` and `allowed`. Using **one** "
             f"`df.query` string that references both with `@`, select the rows with `{a}` ≥ `min_{a}` and `{f.group}` "
             f"in `allowed`, then return their `{f.key}` values sorted alphabetically as a list.",
             f.code() + f"\nmin_{a} = {lo}\nallowed = {allowed!r}",
             f"result = sorted(df.query('{a} >= @min_{a} and {f.group} in @allowed')[{f.key!r}].tolist())",
             require=["query"])


# ================================================================= challenge

@template(M, "challenge")
def pareto_front(rng):
    n = rng.randint(6, 9)
    pts = [(rng.randint(1, 9), rng.randint(1, 9)) for _ in range(n)]
    sol = ("pts = sorted(zip(df['acc'].tolist(), df['speed'].tolist()), key=lambda p: (-p[0], -p[1]))\n"
           "best = float('-inf')  # best speed among strictly higher accuracy\ncount = 0\ni = 0\n"
           "while i < len(pts):\n    j = i\n    while j < len(pts) and pts[j][0] == pts[i][0]:\n        j += 1\n"
           "    top = pts[i][1]  # the highest speed at this accuracy\n"
           "    for a, s in pts[i:j]:\n        if s == top and s > best:\n            count += 1\n"
           "    best = max(best, top)\n    i = j\nresult = count")

    def big(seed, n_, hi):
        return (f"g = np.random.default_rng({seed})\n"
                f"df = pd.DataFrame({{'acc': g.integers(0, {hi}, {n_}), 'speed': g.integers(0, {hi}, {n_})}})")

    tests = [big(seeded(rng), 20000, 10**9), big(seeded(rng), 20000, 30), "df = pd.DataFrame({'acc': [5, 5], 'speed': [3, 3]})"]
    return challenge("Pareto-optimal models",
                     "Each row of `df` is a model with an `acc` (accuracy) and a `speed` score; higher is better for "
                     "both. A model is **dominated** if another model is at least as good on both and strictly better "
                     "on at least one. Count the models that are **not** dominated (the Pareto front; identical models "
                     "don't dominate each other).", f"df = pd.DataFrame({{'acc': {lit([p[0] for p in pts])}, "
                     f"'speed': {lit([p[1] for p in pts])}}})", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     speed="O(n log n): sort by accuracy, sweep the best speed")


@template(M, "challenge")
def range_count_queries(rng):
    n, q = rng.randint(7, 10), rng.randint(3, 4)
    teams = ["red", "blue", "green"]
    data = [(rng.choice(teams), rng.randint(0, 20)) for _ in range(n)]
    qs = []
    for _ in range(q):
        lo = rng.randint(0, 15)
        qs.append((rng.choice(teams), lo, lo + rng.randint(0, 8)))
    sol = ("from bisect import bisect_left, bisect_right\nby_team = {t: sorted(g.tolist()) for t, g in df.groupby('team')['score']}\n"
           "result = []\nfor t, lo, hi in queries.itertuples(index=False):\n    v = by_team.get(t, [])\n"
           "    result.append(bisect_right(v, hi) - bisect_left(v, lo))")

    def big(seed, n_, q_, k):
        return (f"g = np.random.default_rng({seed})\nteams = np.array([f't{{i}}' for i in range({k})])\n"
                f"df = pd.DataFrame({{'team': teams[g.integers(0, {k}, {n_})], 'score': g.integers(0, 10**6, {n_})}})\n"
                f"lo = g.integers(0, 10**6, {q_})\n"
                f"queries = pd.DataFrame({{'team': teams[g.integers(0, {k}, {q_})], 'lo': lo, 'hi': lo + g.integers(0, 10**5, {q_})}})")

    tests = [big(seeded(rng), 20000, 20000, 50), big(seeded(rng), 20000, 20000, 1),
             "df = pd.DataFrame({'team': ['a'], 'score': [5]})\nqueries = pd.DataFrame({'team': ['b', 'a'], 'lo': [0, 5], 'hi': [9, 5]})"]
    return challenge("Filter queries, fast",
                     "`df` holds players with a `team` and a `score`. A dashboard sends `queries`, each a "
                     "`(team, lo, hi)` row asking how many players of that team have `lo ≤ score ≤ hi`. Answer all of "
                     "them in order, as a list of ints. (Filtering `df` once per query is too slow for the hidden tests.)",
                     f"df = pd.DataFrame({data!r}, columns=['team', 'score'])\n"
                     f"queries = pd.DataFrame({qs!r}, columns=['team', 'lo', 'hi'])", sol, tests,
                     "≤ 2·10⁴ players and queries", speed="O((n + q) log n): sort each team once, then binary search")

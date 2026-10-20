"""2.07 Sorting, ranking & duplicates."""

from ._common import P, challenge, frame, lit, seeded, template

M = "2.07"


# ================================================================= lite

@template(M, "lite")
def sort_by_column(rng):
    f = frame(rng, (5, 8), nums=2, distinct=True)
    a = f.num[0]
    desc, sol = rng.choice([
        (f"by `{a}`, smallest first", f"df.sort_values({a!r})"),
        (f"by `{a}`, largest first", f"df.sort_values({a!r}, ascending=False)"),
        (f"by `{f.key}` alphabetically", f"df.sort_values({f.key!r})"),
        ("by its index labels, highest label first (`sort_index`)", "df.sort_index(ascending=False)"),
    ])
    return P(f"{f.describe()}. Sort the rows {desc}.", f.code(), sol, require=["sort_values|sort_index"])


@template(M, "lite")
def rank_column(rng):
    f = frame(rng, (5, 8), nums=1)
    a = f.num[0]
    f.cols[a][1] = f.cols[a][0]  # a tie
    desc, sol = rng.choice([
        ("average ranks, smallest value = 1 (the default `rank()`)", f"df[{a!r}].rank()"),
        ("**dense** ranks with the **largest** value = 1 (`method='dense', ascending=False`)",
         f"df[{a!r}].rank(method='dense', ascending=False)"),
        ("**percentile** ranks between 0 and 1 (`pct=True`)", f"df[{a!r}].rank(pct=True)"),
        ("competition ranks: ties share the **lowest** rank (`method='min'`)", f"df[{a!r}].rank(method='min')"),
    ])
    return P(f"{f.describe()}. Rank the `{a}` column with {desc}.", f.code(), sol, require=["rank"])


# ================================================================= easy

@template(M, "easy")
def sort_two_keys(rng):
    f = frame(rng, (6, 9), nums=1, distinct=True)
    a = f.num[0]
    return P(f"{f.describe()}. Sort by `{f.group}` A→Z, then by `{a}` from high to low within each `{f.group}` "
             f"(`ascending=[...]`).", f.code(), f"df.sort_values([{f.group!r}, {a!r}], ascending=[True, False])",
             require=["sort_values"])


@template(M, "easy")
def drop_duplicate_rows(rng):
    f = frame(rng, (6, 8), nums=1)
    for _ in range(2):
        i, j = rng.sample(range(f.n), 2)
        for c in f.cols:
            f.cols[c][j] = f.cols[c][i]
    desc, sol = rng.choice([
        ("exact duplicate rows (keep the first copy)", "df.drop_duplicates()"),
        (f"rows that repeat an earlier `{f.group}`, keeping the **last** row of each `{f.group}`",
         f"df.drop_duplicates(subset=[{f.group!r}], keep='last')"),
        ("**every** row that has an exact duplicate, including the first copies (`keep=False`)",
         "df.drop_duplicates(keep=False)"),
    ])
    return P(f"{f.describe()}. Some rows were logged twice. Remove {desc}.", f.code(), sol, require=["drop_duplicates"])


@template(M, "easy")
def sort_with_key(rng):
    f = frame(rng, (5, 7), nums=1)
    f.cols[f.key] = [n.lower() if i % 2 else n.upper() for i, n in enumerate(f.cols[f.key])]
    if rng.random() < 0.6:
        return P(f"{f.describe()}; `{f.key}` has inconsistent case. Sort the rows by `{f.key}` **case-insensitively** "
                 f"by passing `key=lambda s: s.str.lower()` to `sort_values`.", f.code(),
                 f"df.sort_values({f.key!r}, key=lambda s: s.str.lower())", require=["sort_values"])
    return P(f"{f.describe()}. Reorder the **columns** alphabetically by name (`sort_index(axis=1)`).", f.code(),
             "df.sort_index(axis=1)", require=["sort_index"])


# ================================================================= medium

@template(M, "medium")
def latest_per_group(rng):
    f = frame(rng, (6, 9), nums=1, date=True)
    f.cols[f.date] = [d[:8] + f"{i + 1:02d}" for i, d in enumerate(f.cols[f.date])]  # unique dates
    which = rng.choice(["latest", "earliest"])
    keep = "last" if which == "latest" else "first"
    return P(f"{f.describe()}. Keep only the **{which}** row of each `{f.group}` by `{f.date}`: sort by `{f.date}`, "
             f"then `drop_duplicates({f.group!r}, keep={keep!r})`.", f.code(),
             f"df.sort_values({f.date!r}).drop_duplicates({f.group!r}, keep={keep!r})",
             require=["drop_duplicates|groupby|idxmax|idxmin"])


@template(M, "medium")
def rank_within_group(rng):
    f = frame(rng, (6, 9), nums=1)
    a = f.num[0]
    method = rng.choice(["first", "dense", "min"])
    return P(f"{f.describe()}. Rank every row's `{a}` **within its `{f.group}`**, highest = 1, with "
             f"`method={method!r}` (`groupby(...)[...].rank(...)`).", f.code(),
             f"df.groupby({f.group!r})[{a!r}].rank(ascending=False, method={method!r})", require=["groupby", "rank"])


@template(M, "medium")
def missing_last_or_first(rng):
    f = frame(rng, (6, 9), nums=1, nan=rng.randint(1, 2))
    a = f.num[0]
    pos = rng.choice(["first", "last"])
    asc = rng.random() < 0.5
    return P(f"{f.describe()}; some `{a}` values are missing. Sort by `{a}` {'ascending' if asc else 'descending'} "
             f"with the missing values placed **{pos}** (`na_position=`).", f.code(),
             f"df.sort_values({a!r}, ascending={asc}, na_position={pos!r})", require=["sort_values"])


# ================================================================= hard

@template(M, "hard")
def top_n_per_group(rng):
    f = frame(rng, (8, 11), nums=1, distinct=True)
    a = f.num[0]
    k = rng.randint(1, 2)
    return P(f"{f.describe()}. Return the top {k} row{'s' if k > 1 else ''} by `{a}` for **each** `{f.group}`, ordered by "
             f"`{f.group}` A→Z and then `{a}` high→low (sort, then `groupby(...).head({k})`).", f.code(),
             f"df.sort_values([{f.group!r}, {a!r}], ascending=[True, False]).groupby({f.group!r}).head({k})",
             require=["sort_values"])


@template(M, "hard")
def leaderboard(rng):
    f = frame(rng, (6, 9), nums=1)
    a = f.num[0]
    f.cols[a][2] = f.cols[a][0]
    return P(f"{f.describe()}. Build a leaderboard: dense-rank `{a}` (highest = 1), sort by that rank and then by "
             f"`{f.key}` A→Z, and return a list of strings like `'1. <{f.key}>'`.", f.code(),
             f"d = df.assign(rank=df[{a!r}].rank(method='dense', ascending=False).astype(int))\n"
             f"d = d.sort_values(['rank', {f.key!r}])\nresult = [f'{{r}}. {{n}}' for r, n in zip(d['rank'], d[{f.key!r}])]",
             require=["rank"])


@template(M, "hard")
def best_row_per_key(rng):
    f = frame(rng, (7, 10), nums=1, date=True)
    a = f.num[0]
    f.cols[f.date] = [d[:8] + f"{i + 1:02d}" for i, d in enumerate(f.cols[f.date])]
    same = [k for k in range(f.n) if f.cols[f.group][k] == f.cols[f.group][0]]
    if len(same) >= 2:  # plant a tie inside one group so the date tie-break matters
        f.cols[a][same[1]] = f.cols[a][same[0]]
    return P(f"{f.describe()}. For **each** `{f.group}`, keep the row with the **highest** `{a}`; if several tie, keep "
             f"the one with the **latest** `{f.date}`. Return the kept rows sorted by `{f.group}`.", f.code(),
             f"(df.sort_values([{a!r}, {f.date!r}])\n   .drop_duplicates({f.group!r}, keep='last')\n   .sort_values({f.group!r}))",
             require=["sort_values"])


# ================================================================= challenge

@template(M, "challenge")
def nesting_chain(rng):
    n = rng.randint(5, 8)
    boxes = [(rng.randint(1, 9), rng.randint(1, 9)) for _ in range(n)]
    sol = ("from bisect import bisect_left\npairs = sorted(zip(df['w'].tolist(), df['h'].tolist()), key=lambda p: (p[0], -p[1]))\n"
           "tails = []\nfor _, h in pairs:\n    k = bisect_left(tails, h)\n    if k == len(tails):\n        tails.append(h)\n"
           "    else:\n        tails[k] = h\nresult = len(tails)")

    def big(seed, n_, hi):
        return f"g = np.random.default_rng({seed})\ndf = pd.DataFrame({{'w': g.integers(1, {hi}, {n_}), 'h': g.integers(1, {hi}, {n_})}})"

    tests = [big(seeded(rng), 20000, 10**9), big(seeded(rng), 20000, 40), "df = pd.DataFrame({'w': [3, 3], 'h': [4, 5]})"]
    return challenge("Nest the thumbnails",
                     "Each row of `df` is an image thumbnail with width `w` and height `h`. One thumbnail fits inside "
                     "another only if it is **strictly smaller in both** width and height (no rotating). Return the "
                     "length of the **longest chain** of thumbnails that nest one inside the next.",
                     f"df = pd.DataFrame({{'w': {lit([b[0] for b in boxes])}, 'h': {lit([b[1] for b in boxes])}}})",
                     sol, tests, "1 ≤ n ≤ 2·10⁴", speed="O(n log n): sort by w (ties by h descending), then LIS on h")


@template(M, "challenge")
def make_unique_cost(rng):
    n = rng.randint(6, 9)
    vals = [rng.randint(1, 8) for _ in range(n)]
    sol = ("total, nxt = 0, None\nfor v in sorted(df['slot'].tolist()):\n    if nxt is not None and v < nxt:\n"
           "        total += nxt - v\n        v = nxt\n    nxt = v + 1\nresult = total")
    tests = [f"df = pd.DataFrame({{'slot': np.random.default_rng({seeded(rng)}).integers(0, 10**9, 20000)}})",
             f"df = pd.DataFrame({{'slot': np.random.default_rng({seeded(rng)}).integers(0, 50, 20000)}})",
             "df = pd.DataFrame({'slot': np.zeros(20000, dtype=int)})", "df = pd.DataFrame({'slot': [5]})"]
    return challenge("Make the ids unique",
                     "`df['slot']` assigns each job a requested integer time slot, but slots must be **unique**. The "
                     "only fix allowed is moving a job to a **later** slot, one step at a time (each step costs 1). "
                     "Return the **minimum total cost** to make all slots distinct.",
                     f"df = pd.DataFrame({{'slot': {lit(vals)}}})", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ slot ≤ 10⁹",
                     speed="O(n log n): sort, then greedily bump each value past the previous one")

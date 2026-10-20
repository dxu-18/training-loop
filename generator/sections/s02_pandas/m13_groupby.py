"""2.13 GroupBy (split-apply-combine)."""

from ._common import P, challenge, frame, seeded, template, threshold

M = "2.13"


# ================================================================= lite

@template(M, "lite")
def group_reduce(rng):
    f = frame(rng, (6, 10), nums=1)
    g, a = f.group, f.num[0]
    word, fn = rng.choice([("mean", "mean"), ("total", "sum"), ("highest", "max"), ("lowest", "min"), ("median", "median")])
    return P(f"{f.describe()}. Compute the {word} `{a}` per `{f.group}` (`groupby`).", f.code(),
             f"df.groupby({g!r})[{a!r}].{fn}()", require=["groupby"])


@template(M, "lite")
def size_vs_count(rng):
    f = frame(rng, (7, 10), nums=1, nan=rng.randint(1, 3))
    g, a = f.group, f.num[0]
    desc, sol = rng.choice([
        ("the number of **rows** per group, NaN or not (`size()`)", f"df.groupby({g!r}).size()"),
        (f"the number of **non-missing** `{a}` values per group (`count()`)", f"df.groupby({g!r})[{a!r}].count()"),
        ("the number of groups (`ngroups`)", f"df.groupby({g!r}).ngroups"),
    ])
    return P(f"{f.describe()}; some `{a}` values are missing. Grouping by `{g}`, get {desc}.", f.code(), sol,
             require=["groupby"])


# ================================================================= easy

@template(M, "easy")
def group_agg_list(rng):
    f = frame(rng, (6, 10), nums=1)
    g, a = f.group, f.num[0]
    funcs = rng.sample(["mean", "max", "min", "count", "std", "sum"], 3)
    return P(f"{f.describe()}. Per `{g}`, compute {', '.join(funcs)} of `{a}` in one `agg` call.", f.code(),
             f"df.groupby({g!r})[{a!r}].agg({funcs!r})", require=["groupby", "agg|aggregate"])


@template(M, "easy")
def group_flat(rng):
    f = frame(rng, (6, 10), nums=1)
    g, a = f.group, f.num[0]
    return P(f"{f.describe()}. Compute the mean `{a}` per `{g}`, keeping `{g}` as a **regular column** instead of the "
             f"index (`as_index=False`).", f.code(), f"df.groupby({g!r}, as_index=False)[{a!r}].mean()",
             require=["groupby"])


@template(M, "easy")
def group_two_keys(rng):
    f = frame(rng, (8, 12), nums=1)
    f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
    g, a = f.group, f.num[0]
    return P(f"{f.describe()} and `flag`. Sum `{a}` per **(`{g}`, `flag`)** pair (group by a list of keys).",
             f.code(), f"df.groupby([{g!r}, 'flag'])[{a!r}].sum()", require=["groupby"])


# ================================================================= medium

@template(M, "medium")
def named_aggregation(rng):
    f = frame(rng, (7, 11), nums=2)
    g, a, b = f.group, *f.num
    return P(f"{f.describe()}. Per `{g}`, build a table with readable column names using **named aggregation**: "
             f"`n` (row count), `avg_{a}` (mean `{a}`) and `max_{b}` (max `{b}`).", f.code(),
             f"df.groupby({g!r}).agg(n=({f.key!r}, 'size'), avg_{a}=({a!r}, 'mean'), max_{b}=({b!r}, 'max'))",
             require=["agg|aggregate"])


@template(M, "medium")
def transform_back(rng):
    f = frame(rng, (7, 10), nums=1)
    g, a = f.group, f.num[0]
    desc, sol = rng.choice([
        (f"each row's `{g}` mean of `{a}` (one value per **original** row)", f"df.groupby({g!r})[{a!r}].transform('mean')"),
        (f"how far each row's `{a}` is above its `{g}`'s mean", f"df[{a!r}] - df.groupby({g!r})[{a!r}].transform('mean')"),
        (f"each row's share of its `{g}`'s total `{a}`", f"df[{a!r}] / df.groupby({g!r})[{a!r}].transform('sum')"),
    ])
    return P(f"{f.describe()}. Compute {desc}, using `transform` so the result lines up with `df`.", f.code(), sol,
             require=["transform"])


@template(M, "medium")
def filter_groups(rng):
    f = frame(rng, (8, 12), nums=1)
    g, a = f.group, f.num[0]
    if rng.random() < 0.5:
        k = rng.randint(2, 4)
        return P(f"{f.describe()}. Keep only the rows of `{g}` groups that have **at least {k} rows** (`groupby(...).filter`).",
                 f.code(), f"df.groupby({g!r}).filter(lambda d: len(d) >= {k})", require=["filter"])
    t = threshold(rng, f.cols[a], f.nums[0])
    return P(f"{f.describe()}. Keep only the rows of `{g}` groups whose **mean `{a}`** is above {t} (`groupby(...).filter`).",
             f.code(), f"df.groupby({g!r}).filter(lambda d: d[{a!r}].mean() > {t})", require=["filter"])


@template(M, "medium")
def best_row_per_group(rng):
    f = frame(rng, (7, 10), nums=1, distinct=True)
    g, a = f.group, f.num[0]
    hi = rng.random() < 0.5
    fn = "idxmax" if hi else "idxmin"
    return P(f"{f.describe()}. Return the full row with the {'highest' if hi else 'lowest'} `{a}` in each `{g}`: "
             f"`df.loc[df.groupby(...)[...].{fn}()]`.", f.code(), f"df.loc[df.groupby({g!r})[{a!r}].{fn}()]",
             require=[fn])


# ================================================================= hard

@template(M, "hard")
def zscore_in_group(rng):
    f = frame(rng, (9, 12), nums=1)
    g, a = f.group, f.num[0]
    for lvl in f.levels:  # every group needs at least two rows for a std
        if f.cols[g].count(lvl) == 1:
            f.cols[g][f.cols[g].index(lvl)] = f.levels[0]
    return P(f"{f.describe()}. Add a column `{a}_z`: each row's z-score **within its `{g}`** (subtract the group mean, "
             f"divide by the group's `std()`). Return `df`.", f.code(),
             f"grp = df.groupby({g!r})[{a!r}]\ndf['{a}_z'] = (df[{a!r}] - grp.transform('mean')) / grp.transform('std')\nresult = df",
             require=["transform"])


@template(M, "hard")
def group_running(rng):
    f = frame(rng, (8, 11), nums=1)
    g, a = f.group, f.num[0]
    return P(f"{f.describe()}; rows are in time order. Add three per-`{g}` features and return `df`: `{a}_cum` "
             f"(running total within the group), `nth` (0-based position within the group, `cumcount`) and `{a}_prev` "
             f"(the group's previous `{a}`, NaN for the first row).", f.code(),
             f"grp = df.groupby({g!r})\ndf['{a}_cum'] = grp[{a!r}].cumsum()\ndf['nth'] = grp.cumcount()\n"
             f"df['{a}_prev'] = grp[{a!r}].shift(1)\nresult = df", require=["groupby"])


@template(M, "hard")
def roster(rng):
    f = frame(rng, (7, 10), nums=1)
    g, a = f.group, f.num[0]
    return P(f"{f.describe()}. Build a roster: one row per `{g}` with `members` (the `{f.key}` values sorted A→Z and "
             f"joined with `', '`) and `total` (sum of `{a}`), sorted by `total` descending.", f.code(),
             f"result = (df.groupby({g!r})\n            .agg(members=({f.key!r}, lambda s: ', '.join(sorted(s))), total=({a!r}, 'sum'))\n"
             "            .sort_values('total', ascending=False))", require=["groupby"])


# ================================================================= challenge

@template(M, "challenge")
def pairs_within_d(rng):
    n = rng.randint(7, 10)
    rows = [(rng.choice(["red", "blue"]), rng.randint(0, 30)) for _ in range(n)]
    d = rng.randint(1, 8)
    sol = ("total = 0\nfor _, s in df.groupby('team')['score']:\n    v = sorted(s.tolist())\n    left = 0\n"
           "    for right in range(len(v)):\n        while v[right] - v[left] > D:\n            left += 1\n"
           "        total += right - left\nresult = total")

    def big(seed, n_, k, hi, d_):
        return (f"g = np.random.default_rng({seed})\ndf = pd.DataFrame({{'team': [f't{{i}}' for i in g.integers(0, {k}, {n_})], "
                f"'score': g.integers(0, {hi}, {n_})}})\nD = {d_}")

    tests = [big(seeded(rng), 20000, 3, 10**6, 500), big(seeded(rng), 20000, 1, 100, 3), big(seeded(rng), 20000, 5000, 10, 0),
             "df = pd.DataFrame({'team': ['a'], 'score': [1]})\nD = 5"]
    return challenge("Close pairs within a team",
                     "`df` lists players with a `team` and a `score`. Count the **pairs of players on the same team** "
                     "whose scores differ by **at most D**.", f"df = pd.DataFrame({rows!r}, columns=['team', 'score'])\nD = {d}",
                     sol, tests, "≤ 2·10⁴ players", speed="O(n log n): sort each team, then two pointers")


@template(M, "challenge")
def entity_resolution(rng):
    n = rng.randint(6, 9)
    emails = [f"u{rng.randint(1, 5)}@x.io" for _ in range(n)]
    phones = [f"555-{rng.randint(10, 15)}" for _ in range(n)]
    sol = ("parent = list(range(len(df)))\n\ndef find(x):\n    while parent[x] != x:\n        parent[x] = parent[parent[x]]\n"
           "        x = parent[x]\n    return x\n\nfor col in ('email', 'phone'):\n    first = {}\n"
           "    for i, key in enumerate(df[col].tolist()):\n        if key in first:\n            a, b = find(i), find(first[key])\n"
           "            if a != b:\n                parent[a] = b\n        else:\n            first[key] = i\n"
           "from collections import Counter\nsizes = Counter(find(i) for i in range(len(df)))\nresult = (len(sizes), max(sizes.values()))")

    def big(seed, n_, ke, kp):
        return (f"g = np.random.default_rng({seed})\ndf = pd.DataFrame({{'email': [f'e{{i}}' for i in g.integers(0, {ke}, {n_})], "
                f"'phone': [f'p{{i}}' for i in g.integers(0, {kp}, {n_})]}})")

    tests = [big(seeded(rng), 20000, 15000, 15000), big(seeded(rng), 20000, 20000, 20000), big(seeded(rng), 20000, 10, 10**6),
             "df = pd.DataFrame({'email': ['a'], 'phone': ['b']})"]
    return challenge("Resolve the customers",
                     "Each row of `df` is a customer sign-up with an `email` and a `phone`. Two rows belong to the "
                     "**same person** if they share an email **or** a phone, directly or through a chain of other rows. "
                     "Return the tuple `(number_of_people, size_of_largest_person)`.",
                     f"df = pd.DataFrame({{'email': {emails!r}, 'phone': {phones!r}}})", sol, tests, "≤ 2·10⁴ rows",
                     speed="O(n α(n)) with union-find")

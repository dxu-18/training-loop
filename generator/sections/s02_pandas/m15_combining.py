"""2.15 Combining frames (merge, join, concat, merge_asof, combine_first)."""

from ._common import P, challenge, frame, lit, seeded, template

M = "2.15"
COACHES = ["Kim", "Lee", "Max", "Ola", "Raj", "Sam"]


def _people_and_teams(rng):
    f = frame(rng, (5, 8), nums=1)
    g = f.group
    have = list(f.levels[:-1]) + ["gold"]  # one level has no coach, one coach has no members
    coaches = rng.sample(COACHES, len(have))
    setup = (f.code("people") + f"\nteams = pd.DataFrame({{{g!r}: {have!r}, 'coach': {coaches!r}}})")
    return f, g, setup


# ================================================================= lite

@template(M, "lite")
def inner_or_left(rng):
    f, g, setup = _people_and_teams(rng)
    if rng.random() < 0.5:
        return P(f"`people` lists {f.th.plural} with a `{g}`; `teams` gives each `{g}`'s coach (not every `{g}` has one). "
                 f"Attach the coach with an **inner** join on `{g}` (only matching keys survive).", setup,
                 f"pd.merge(people, teams, on={g!r})", require=["merge"])
    return P(f"`people` lists {f.th.plural} with a `{g}`; `teams` gives each `{g}`'s coach (not every `{g}` has one). "
             f"Attach the coach with a **left** join on `{g}`, so every row of `people` survives (coach NaN when missing).",
             setup, f"people.merge(teams, on={g!r}, how='left')", require=["merge|join"])


@template(M, "lite")
def concat_frames(rng):
    f = frame(rng, (6, 9), nums=1)
    k = rng.randint(2, 3)
    setup = f.code() + f"\na, b = df.head({k}), df.tail({k})"
    desc, sol = rng.choice([
        ("stack `a` on top of `b` (rows), keeping their original index labels", "pd.concat([a, b])"),
        ("stack `a` on top of `b` with a fresh 0..n−1 index (`ignore_index=True`)", "pd.concat([a, b], ignore_index=True)"),
        ("put `a` and `b.reset_index(drop=True)` side by side (`axis=1`) after resetting `a`'s index too",
         "pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)"),
    ])
    return P(f"{f.describe()}. `a` is its first {k} rows and `b` its last {k}. With `pd.concat`, {desc}.", setup, sol,
             require=["concat"])


# ================================================================= easy

@template(M, "easy")
def merge_how(rng):
    f, g, setup = _people_and_teams(rng)
    if rng.random() < 0.5:
        return P(f"`people` and `teams` share the key `{g}`, but each has keys the other lacks. Do an **outer** join with "
                 f"`indicator=True`, so a `_merge` column says where each row came from.", setup,
                 f"people.merge(teams, on={g!r}, how='outer', indicator=True)", require=["merge"])
    return P(f"`people` and `teams` share the key `{g}`. Do a **right** join: keep every team, even ones with no members.",
             setup, f"people.merge(teams, on={g!r}, how='right')", require=["merge"])


@template(M, "easy")
def merge_other_keys(rng):
    f, g, setup = _people_and_teams(rng)
    if rng.random() < 0.5:
        return P(f"`teams` names its key column `squad` instead of `{g}`. Join it to `people` anyway with "
                 f"`left_on={g!r}, right_on='squad'`.", setup + f"\nteams = teams.rename(columns={{{g!r}: 'squad'}})",
                 f"people.merge(teams, left_on={g!r}, right_on='squad')", require=["merge"])
    return P(f"Pair up every two {f.th.plural} on the **same** `{g}` with a self-join: `people.merge(people, on={g!r}, "
             f"suffixes=('_a', '_b'))`.", setup, f"people.merge(people, on={g!r}, suffixes=('_a', '_b'))", require=["merge"])


@template(M, "easy")
def join_on_index(rng):
    f = frame(rng, (5, 7), nums=2)
    a, b = f.num
    k = f.key
    drop = rng.randint(1, 2)
    setup = (f.code() + f"\nstats = df.set_index({k!r})[[{a!r}]]\ninfo = df.set_index({k!r})[[{b!r}]].iloc[{drop}:]")
    if rng.random() < 0.5:
        return P(f"`stats` and `info` are both indexed by `{k}` (`info` is missing the first {drop}). Combine them side by "
                 f"side with `stats.join(info)` (a left join on the **index**).", setup, "stats.join(info)", require=["join"])
    return P(f"`stats` and `info` are both indexed by `{k}` (`info` is missing the first {drop}). Join them on the index, "
             f"keeping only labels present in **both** (`how='inner'`).", setup, "stats.join(info, how='inner')",
             require=["join|merge"])


# ================================================================= medium

@template(M, "medium")
def anti_join(rng):
    f, g, setup = _people_and_teams(rng)
    if rng.random() < 0.5:
        return P(f"Return the rows of `people` whose `{g}` has **no** coach in `teams` (an anti-join with `~isin`).", setup,
                 f"people[~people[{g!r}].isin(teams[{g!r}])]", require=["isin|merge"])
    return P(f"Return the rows of `people` whose `{g}` has **no** coach in `teams`, using a left merge with "
             f"`indicator=True` and keeping only the `'left_only'` rows (then drop the `_merge` and `coach` columns).",
             setup, f"m = people.merge(teams, on={g!r}, how='left', indicator=True)\n"
                    "result = m[m['_merge'] == 'left_only'].drop(columns=['_merge', 'coach'])", require=["merge"])


@template(M, "medium")
def concat_with_keys(rng):
    f = frame(rng, (6, 9), nums=1)
    k = rng.randint(2, 3)
    setup = f.code() + f"\ntrain, test = df.head({k}), df.tail({k})"
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Stack `train` and `test` with `pd.concat` and a **dict** so the dict keys become an "
                 f"outer index level named `split` (`names=['split']`).", setup,
                 "pd.concat({'train': train, 'test': test}, names=['split'])", require=["concat"])
    return P(f"{f.describe()}. Stack `train` and `test` into one frame with a new `split` column saying `'train'` or "
             f"`'test'` for each row (`assign` before `concat`), with a fresh index.", setup,
             "pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)", require=["concat"])


@template(M, "medium")
def asof_merge(rng):
    n, m = rng.randint(3, 5), rng.randint(3, 4)
    qs = sorted(rng.sample(range(0, 60), m))
    ts = sorted(rng.sample(range(qs[0] + 1, 70), n))
    prices = [round(100 + rng.uniform(-5, 5), 2) for _ in qs]
    qty = [rng.randint(1, 9) for _ in ts]
    setup = (f"base = pd.Timestamp('2024-01-01 10:00:00')\n"
             f"trades = pd.DataFrame({{'t': base + pd.to_timedelta({ts!r}, unit='s'), 'qty': {qty!r}}})\n"
             f"quotes = pd.DataFrame({{'t': base + pd.to_timedelta({qs!r}, unit='s'), 'price': {prices!r}}})")
    direction = rng.choice(["backward", "forward"])
    word = "last quote **at or before**" if direction == "backward" else "first quote **at or after**"
    return P(f"`trades` and `quotes` are both sorted by time `t`. Attach to each trade the {word} it with "
             f"`pd.merge_asof(..., on='t', direction={direction!r})`.", setup,
             f"pd.merge_asof(trades, quotes, on='t', direction={direction!r})", require=["merge_asof"])


@template(M, "medium")
def combine_and_compare(rng):
    n = rng.randint(4, 6)
    a = [round(rng.uniform(0, 9), 1) for _ in range(n)]
    for i in rng.sample(range(n), 2):
        a[i] = float("nan")
    b = [round(rng.uniform(0, 9), 1) for _ in range(n)]
    if rng.random() < 0.5:
        return P("`a` is the primary measurement and `b` a backup. Fill `a`'s gaps from `b` at the same index "
                 "(`a.combine_first(b)`).", f"a = pd.Series({lit(a)})\nb = pd.Series({b!r})", "a.combine_first(b)",
                 require=["combine_first|fillna"])
    old = [round(rng.uniform(0, 9), 1) for _ in range(n)]
    new = list(old)
    for i in rng.sample(range(n), 2):
        new[i] = round(new[i] + 1, 1)
    return P("`old` and `new` are two versions of the same table. Show only the cells that changed with "
             "`old.compare(new)`.", f"old = pd.DataFrame({{'score': {old!r}}})\nnew = pd.DataFrame({{'score': {new!r}}})",
             "old.compare(new)", require=["compare"])


# ================================================================= hard

@template(M, "hard")
def validated_merge(rng):
    f, g, setup = _people_and_teams(rng)
    setup += f"\nteams = pd.concat([teams, teams.head(1)], ignore_index=True)  # a duplicate key slipped in"
    return P(f"`teams` accidentally contains a **duplicate** `{g}`, which would silently duplicate people in a merge. "
             f"First try `people.merge(teams, on={g!r}, how='left', validate='many_to_one')` inside `try` / `except "
             f"pd.errors.MergeError` and record `caught = True` if it raises. Then drop duplicate keys from `teams` "
             f"(keep the first) and do the validated merge for real into `merged`.", setup,
             "caught = False\ntry:\n    people.merge(teams, on=" + repr(g) + ", how='left', validate='many_to_one')\n"
             "except pd.errors.MergeError:\n    caught = True\n"
             f"merged = people.merge(teams.drop_duplicates({g!r}), on={g!r}, how='left', validate='many_to_one')",
             answer=["caught", "merged"], require=["validate"])


@template(M, "hard")
def stack_monthly_files(rng):
    months = rng.sample(["2024-01", "2024-02", "2024-03", "2024-04"], 3)
    files = {m: [rng.randint(10, 99) for _ in range(rng.randint(2, 3))] for m in sorted(months)}
    setup = "frames = {\n" + "".join(f"    {m!r}: pd.DataFrame({{'sales': {v!r}}}),\n" for m, v in files.items()) + "}"
    return P("`frames` maps a month to that month's sales file (a DataFrame). Stack them into one frame with a `month` "
             "column naming the source month of each row (fresh index), then return the total sales per month.", setup,
             "all_rows = pd.concat([d.assign(month=m) for m, d in frames.items()], ignore_index=True)\n"
             "result = all_rows.groupby('month')['sales'].sum()", require=["concat"])


@template(M, "hard")
def teammate_pairs(rng):
    f, g, setup = _people_and_teams(rng)
    k = f.key
    return P(f"List every **unordered pair** of {f.th.plural} on the same `{g}`: self-merge `people` on `{g}` with "
             f"suffixes `_a`/`_b`, keep rows where `{k}_a` < `{k}_b` (alphabetically, so each pair appears once), and "
             f"return a list of `({k}_a, {k}_b)` tuples sorted.", setup,
             f"m = people.merge(people, on={g!r}, suffixes=('_a', '_b'))\nm = m[m['{k}_a'] < m['{k}_b']]\n"
             f"result = sorted(zip(m['{k}_a'], m['{k}_b']))", require=["merge"])


# ================================================================= challenge

@template(M, "challenge")
def interval_lookup(rng):
    k = rng.randint(3, 4)
    edges = sorted(rng.sample(range(0, 50), 2 * k))
    shifts = [(edges[2 * i], edges[2 * i + 1], f"s{i}") for i in range(k)]
    events = [rng.randint(0, 55) for _ in range(rng.randint(5, 7))]
    sol = ("from bisect import bisect_right\nst = shifts['start'].tolist()\nen = shifts['end'].tolist()\n"
           "ids = shifts['shift'].tolist()\nout = []\nfor t in events['t'].tolist():\n    i = bisect_right(st, t) - 1\n"
           "    out.append(ids[i] if i >= 0 and t < en[i] else None)\nresult = out")

    def big(seed, k_, n_):
        return (f"g = np.random.default_rng({seed})\nedges = np.unique(g.integers(0, 10**7, {2 * k_}))\n"
                f"edges = edges[:len(edges) // 2 * 2]\n"
                f"shifts = pd.DataFrame({{'start': edges[0::2], 'end': edges[1::2], 'shift': [f's{{i}}' for i in range(len(edges) // 2)]}})\n"
                f"events = pd.DataFrame({{'t': g.integers(0, 10**7, {n_})}})")

    tests = [big(seeded(rng), 5000, 20000), big(seeded(rng), 1, 20000),
             "shifts = pd.DataFrame({'start': [0], 'end': [5], 'shift': ['a']})\nevents = pd.DataFrame({'t': [0, 4, 5]})"]
    return challenge("Which shift was it?",
                     "`shifts` lists **non-overlapping** work shifts `[start, end)` sorted by `start`, each with a "
                     "`shift` id; there can be gaps between shifts. For each event time in `events['t']`, return the id "
                     "of the shift that contains it, or `None` if it falls in a gap. Return a list in event order.",
                     f"shifts = pd.DataFrame({shifts!r}, columns=['start', 'end', 'shift'])\nevents = pd.DataFrame({{'t': {events!r}}})",
                     sol, tests, "≤ 2·10⁴ shifts and events",
                     speed="O((n + m) log n): binary search (or `merge_asof` plus an end check)")


@template(M, "challenge")
def join_size(rng):
    left = [rng.choice("abcd") for _ in range(rng.randint(4, 7))]
    right = [rng.choice("abce") for _ in range(rng.randint(3, 6))]
    sol = ("lc = L['key'].value_counts()\nrc = R['key'].value_counts()\ninner = int((lc * rc).dropna().sum())\n"
           "left_only = int(lc[~lc.index.isin(rc.index)].sum())\nresult = (inner, inner + left_only)")

    def big(seed, n_, m_, k):
        return (f"g = np.random.default_rng({seed})\nL = pd.DataFrame({{'key': g.integers(0, {k}, {n_})}})\n"
                f"R = pd.DataFrame({{'key': g.integers(0, {k}, {m_})}})")

    tests = [big(seeded(rng), 20000, 20000, 5), big(seeded(rng), 20000, 20000, 100000), big(seeded(rng), 20000, 1, 3),
             "L = pd.DataFrame({'key': ['x']})\nR = pd.DataFrame({'key': ['y']})"]
    return challenge("Predict the join size",
                     "Before running an expensive join, a pipeline checks how big it will be. `L` and `R` each have a "
                     "`key` column with repeated keys. Return the tuple `(inner_rows, left_rows)`: the number of rows "
                     "`L.merge(R, on='key', how='inner')` and `how='left'` would produce. The hidden tests would produce "
                     "~10⁸ rows, so **don't** materialize the join.",
                     f"L = pd.DataFrame({{'key': {left!r}}})\nR = pd.DataFrame({{'key': {right!r}}})", sol, tests,
                     "≤ 2·10⁴ rows per side", speed="O(n + m) with key counts")

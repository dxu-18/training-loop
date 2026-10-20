"""2.06 Adding, modifying & removing columns."""

from ._common import P, challenge, frame, lit, seeded, template, threshold

M = "2.06"


def _thr(rng, f, col):
    return threshold(rng, f.cols[col], next(m for m in f.nums if m.var == col))


# ================================================================= lite

@template(M, "lite")
def new_column(rng):
    f = frame(rng, (4, 7), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    k = rng.choice([2, 10, 100, 0.5])
    desc, sol = rng.choice([
        (f"`{a}_x{k}`, equal to `{a}` times {k}", f"df[{f'{a}_x{k}'!r}] = df[{a!r}] * {k}"),
        (f"`high_{a}`, a boolean that is True where `{a}` is greater than {t}", f"df['high_{a}'] = df[{a!r}] > {t}"),
        (f"`{a}_plus_{b}`, the sum of the two numeric columns", f"df['{a}_plus_{b}'] = df[{a!r}] + df[{b!r}]"),
    ])
    return P(f"{f.describe()}. Add a column {desc} (vectorized, no loops). Return `df`.", f.code(),
             sol + "\nresult = df", no_loops=True)


@template(M, "lite")
def rename_columns(rng):
    f = frame(rng, (3, 5), nums=2)
    a = f.num[0]
    new = rng.choice(["value", "metric", "target", "y"])
    desc, sol = rng.choice([
        (f"`{a}` to `{new}` (`rename(columns={{...}})`)", f"df.rename(columns={{{a!r}: {new!r}}})"),
        ("every column to UPPER CASE in one call (pass a function to `rename`)", "df.rename(columns=str.upper)"),
        ("every column with the prefix `f_` (`add_prefix`)", "df.add_prefix('f_')"),
        ("every column with the suffix `_v2` (`add_suffix`)", "df.add_suffix('_v2')"),
    ])
    return P(f"{f.describe()}. Rename {desc} and return the new frame.", f.code(), sol)


# ================================================================= easy

@template(M, "easy")
def where_column(rng):
    f = frame(rng, (5, 8), nums=2)
    a = f.num[0]
    t = _thr(rng, f, a)
    hi, lo = rng.choice([("A", "B"), ("pass", "fail"), ("keep", "drop"), ("fast", "slow")])
    return P(f"{f.describe()}. Add a column `label` that is `{hi!r}` where `{a}` is at least {t} and `{lo!r}` "
             f"otherwise, using `np.where`. Return `df`.", f.code(),
             f"df['label'] = np.where(df[{a!r}] >= {t}, {hi!r}, {lo!r})\nresult = df", require=["where"])


@template(M, "easy")
def drop_things(rng):
    f = frame(rng, (5, 7), nums=3)
    a, b, c = f.num
    i, j = sorted(rng.sample(range(f.n), 2))
    desc, sol = rng.choice([
        (f"the columns `{a}` and `{c}` (`drop(columns=[...])`)", f"df.drop(columns=[{a!r}, {c!r}])"),
        (f"the rows labelled {i} and {j} (`drop(index=[...])`)", f"df.drop(index=[{i}, {j}])"),
        (f"the columns `{b}` and `notes`, without failing because `notes` doesn't exist (`errors='ignore'`)",
         f"df.drop(columns=[{b!r}, 'notes'], errors='ignore')"),
    ])
    return P(f"{f.describe()}. Return a new frame with {desc} removed.", f.code(), sol, require=["drop"])


@template(M, "easy")
def insert_and_pop(rng):
    f = frame(rng, (4, 6), nums=2)
    a = f.num[0]
    if rng.random() < 0.5:
        pos = rng.randint(0, 2)
        return P(f"{f.describe()}. Insert a column `id` holding `0, 1, 2, …` at **position {pos}** with `df.insert` "
                 f"(it modifies `df` in place), then return `df`.", f.code(),
                 f"df.insert({pos}, 'id', range(len(df)))\nresult = df", require=["insert"])
    return P(f"{f.describe()}. Remove the `{a}` column with `df.pop` (which returns it) and return the tuple "
             f"`(popped_series, df)`.", f.code(), f"popped = df.pop({a!r})\nresult = (popped, df)", require=["pop"])


# ================================================================= medium

@template(M, "medium")
def assign_chain(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. With **one** `df.assign(...)` call (lambdas, so it would work mid-chain), add "
             f"`{a}_z`, the z-score of `{a}` (subtract the mean, divide by `std()`, pandas' sample std), and "
             f"`name_len`, the length of each `{f.key}` string.", f.code(),
             f"df.assign({a}_z=lambda d: (d[{a!r}] - d[{a!r}].mean()) / d[{a!r}].std(),\n"
             f"          name_len=lambda d: d[{f.key!r}].str.len())", require=["assign"])


@template(M, "medium")
def conditional_update(rng):
    f = frame(rng, (6, 9), nums=2)
    a, b = f.num
    lvl = f.levels[0]
    bonus = rng.choice([1, 5, 10])
    t = _thr(rng, f, b)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Add {bonus} to `{a}` for every row whose `{f.group}` is `{lvl!r}`, in a **single** "
                 f"`.loc` statement (mask and column together). Return `df`.", f.code(),
                 f"df.loc[df[{f.group!r}] == {lvl!r}, {a!r}] += {bonus}\nresult = df", require=["loc"])
    return P(f"{f.describe()}. Set `{a}` to 0 for every row whose `{b}` is below {t}, in a single `.loc` statement. "
             f"Return `df`.", f.code(), f"df.loc[df[{b!r}] < {t}, {a!r}] = 0\nresult = df", require=["loc"])


@template(M, "medium")
def index_moves(rng):
    f = frame(rng, (4, 6), nums=2)
    a, b = f.num
    desc, setup_extra, sol = rng.choice([
        (f"move `{f.key}` into the index (`set_index`)", "", f"df.set_index({f.key!r})"),
        (f"move the `{f.key}` index back into a regular column (`reset_index`)", f"\ndf = df.set_index({f.key!r})",
         "df.reset_index()"),
        ("throw away the current index and number the rows 0, 1, 2, … (`reset_index(drop=True)`)",
         "\ndf = df.iloc[::-1]", "df.reset_index(drop=True)"),
        (f"reorder the columns to `{b}`, `{f.key}`, `{a}` with `reindex(columns=[...])`", "",
         f"df.reindex(columns=[{b!r}, {f.key!r}, {a!r}])"),
    ])
    return P(f"{f.describe()}. {desc[0].upper() + desc[1:]}.", f.code() + setup_extra, sol)


@template(M, "medium")
def clean_column_names(rng):
    raw = rng.sample([" User ID", "Signup Date ", "Total Spend", " Is Active", "Plan Type", " Region Name "], 4)
    clean = [c.strip().lower().replace(" ", "_") for c in raw]
    rows = [[rng.randint(1, 9) for _ in raw] for _ in range(2)]
    return P(f"An export gave `df` messy column names like `{raw[0]!r}`. Clean **all** of them at once: strip spaces, "
             f"lower-case, and replace inner spaces with `_` (e.g. `{clean[0]!r}`). Assign the new list to "
             f"`df.columns` and return `df`.", f"df = pd.DataFrame({rows!r}, columns={raw!r})",
             "df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]\nresult = df")


# ================================================================= hard

@template(M, "hard")
def feature_engineering(rng):
    f = frame(rng, (6, 9), nums=2)
    a, b = f.num
    q1, q2 = sorted([_thr(rng, f, a), _thr(rng, f, a)])
    if q1 == q2:
        q2 = q1 + 1
    return P(f"{f.describe()}. Engineer features and return `df` with exactly these new columns appended, in order:\n\n"
             f"* `ratio` = `{a}` / `{b}`\n* `tier` = `'low'` if `{a}` < {q1}, `'mid'` if `{a}` < {q2}, else `'high'` "
             f"(`np.select`)\n* `{f.group}_code`: the integer code of `{f.group}` after `astype('category')` (`.cat.codes`)",
             f.code(), f"df['ratio'] = df[{a!r}] / df[{b!r}]\n"
             f"df['tier'] = np.select([df[{a!r}] < {q1}, df[{a!r}] < {q2}], ['low', 'mid'], default='high')\n"
             f"df['{f.group}_code'] = df[{f.group!r}].astype('category').cat.codes\nresult = df", no_loops=True)


@template(M, "hard")
def reshape_columns(rng):
    f = frame(rng, (4, 6), nums=3)
    a, b, c = f.num
    return P(f"{f.describe()}. In one chain: rename `{a}` to `primary`, drop `{c}`, and move `{f.group}` to be the "
             f"**first** column (keep the others in their current order).", f.code(),
             f"d = df.rename(columns={{{a!r}: 'primary'}}).drop(columns=[{c!r}])\n"
             f"result = d[[{f.group!r}] + [col for col in d.columns if col != {f.group!r}]]")


@template(M, "hard")
def fix_chained_assignment(rng):
    f = frame(rng, (6, 9), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    return P(f"{f.describe()}. A teammate wrote\n\n```\ndf[df[{a!r}] > {t}][{b!r}] = 0\n```\n\nto zero `{b}` for "
             f"the rows with `{a}` > {t}, but under pandas 3's Copy-on-Write this **chained assignment** changes "
             f"nothing (it writes into a temporary copy). Do it correctly, then also add a column `flagged` that is "
             f"True on exactly those rows. Return `df`.", f.code(),
             f"mask = df[{a!r}] > {t}\ndf.loc[mask, {b!r}] = 0\ndf['flagged'] = mask\nresult = df", require=["loc"])


# ================================================================= challenge

@template(M, "challenge")
def range_add_updates(rng):
    n, q = rng.randint(6, 9), rng.randint(3, 4)
    x = [rng.randint(0, 9) for _ in range(n)]
    ups = []
    for _ in range(q):
        l_ = rng.randrange(n)
        ups.append((l_, rng.randrange(l_, n), rng.randint(-5, 5)))
    sol = ("diff = [0] * (len(df) + 1)\nfor l, r, d in updates.itertuples(index=False):\n    diff[l] += d\n"
           "    diff[r + 1] -= d\nrun, out = 0, []\nfor v, d in zip(df['x'].tolist(), diff):\n    run += d\n"
           "    out.append(v + run)\nresult = pd.Series(out, name='x')")

    def big(seed, n_, q_):
        return (f"g = np.random.default_rng({seed})\ndf = pd.DataFrame({{'x': g.integers(-10**6, 10**6, {n_})}})\n"
                f"l = g.integers(0, {n_}, {q_})\nupdates = pd.DataFrame({{'l': l, 'r': np.maximum(l, g.integers(0, {n_}, {q_})), "
                f"'delta': g.integers(-1000, 1001, {q_})}})")

    tests = [big(seeded(rng), 20000, 20000), big(seeded(rng), 20000, 1),
             "df = pd.DataFrame({'x': [1]})\nupdates = pd.DataFrame({'l': [0, 0], 'r': [0, 0], 'delta': [5, -2]})"]
    return challenge("Apply many range updates",
                     "`df['x']` is a column of per-row budgets. `updates` holds `(l, r, delta)` rows: add `delta` to "
                     "every row at positions `l..r` (inclusive). Apply all updates and return the final column as an "
                     "int Series (same index as `df`).",
                     f"df = pd.DataFrame({{'x': {lit(x)}}})\nupdates = pd.DataFrame({ups!r}, columns=['l', 'r', 'delta'])",
                     sol, tests, "≤ 2·10⁴ rows and updates", answer_desc=" (a Series)",
                     speed="O(n + q) with a difference array and a running sum")


@template(M, "challenge")
def min_swaps_sort(rng):
    n = rng.randint(6, 9)
    vals = rng.sample(range(1, 50), n)
    sol = ("v = df['rank'].tolist()\ntarget = {x: i for i, x in enumerate(sorted(v))}\nseen = [False] * len(v)\n"
           "swaps = 0\nfor i in range(len(v)):\n    length, j = 0, i\n    while not seen[j]:\n        seen[j] = True\n"
           "        j = target[v[j]]\n        length += 1\n    if length:\n        swaps += length - 1\nresult = swaps")
    tests = [f"df = pd.DataFrame({{'rank': np.random.default_rng({seeded(rng)}).permutation(20000)}})",
             "df = pd.DataFrame({'rank': np.arange(20000)[::-1]})",
             f"df = pd.DataFrame({{'rank': np.random.default_rng({seeded(rng)}).choice(10**9, 20000, replace=False)}})",
             "df = pd.DataFrame({'rank': [7]})"]
    return challenge("Fewest swaps to sort",
                     "`df['rank']` holds **distinct** priority values in the order a scheduler queued jobs. One "
                     "operation swaps the values of two rows. Return the **minimum number of swaps** that sorts the "
                     "column ascending.", f"df = pd.DataFrame({{'rank': {lit(vals)}}})", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, distinct values", speed="O(n log n): count the cycles of the sorting permutation")

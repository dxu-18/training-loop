"""2.19 Performance & handing data to models."""

from ._common import P, challenge, frame, lit, rand_ints, seeded, template, threshold

M = "2.19"


# ================================================================= lite

@template(M, "lite")
def to_float32_matrix(rng):
    f = frame(rng, (4, 7), nums=2)
    a, b = f.num
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Hand the features to a model: return `df[[{a!r}, {b!r}]]` as a **float32** NumPy matrix "
                 f"(`to_numpy(dtype=np.float32)`).", f.code(), f"df[[{a!r}, {b!r}]].to_numpy(dtype=np.float32)",
                 require=["to_numpy"], check_dtype=True)
    return P(f"{f.describe()}. Return the `{a}` column as a plain Python list (`to_list`).", f.code(),
             f"df[{a!r}].to_list()", require=["to_list|tolist"])


@template(M, "lite")
def label_codes(rng):
    f = frame(rng, (5, 8), nums=1)
    return P(f"{f.describe()}. Turn `{f.group}` into **integer class labels** for a classifier: convert to `category` and "
             f"return the codes as a NumPy array (`.cat.codes.to_numpy()`).", f.code(),
             f"df[{f.group!r}].astype('category').cat.codes.to_numpy()", require=["codes"])


# ================================================================= easy

@template(M, "easy")
def itertuples_loop(rng):
    f = frame(rng, (3, 5), nums=1)
    a = f.num[0]
    return P(f"{f.describe()}. When you truly need a Python loop, `itertuples` is far faster than `iterrows`. Build a list "
             f"of strings like `'<{f.key}>=<{a}>'` by looping over `df.itertuples(index=False)` (attribute access: "
             f"`row.{f.key}`).", f.code(), f"[f'{{row.{f.key}}}={{row.{a}}}' for row in df.itertuples(index=False)]",
             require=["itertuples"])


@template(M, "easy")
def downcast_numbers(rng):
    n = rng.randint(4, 6)
    s = seeded(rng)
    kind = rng.choice(["integer", "float"])
    if kind == "integer":
        return P("`counts` is an int64 column whose values fit in a small type. Shrink it with "
                 "`pd.to_numeric(..., downcast='integer')`.", f"counts = pd.Series(np.random.default_rng({s}).integers(0, 100, {n}))",
                 "pd.to_numeric(counts, downcast='integer')", require=["to_numeric"], check_dtype=True)
    return P("`probs` is a float64 column. Shrink it to float32 with `pd.to_numeric(..., downcast='float')`.",
             f"probs = pd.Series(np.random.default_rng({s}).random({n}))", "pd.to_numeric(probs, downcast='float')",
             require=["to_numeric"], check_dtype=True)


@template(M, "easy")
def dtype_checks(rng):
    f = frame(rng, (3, 5), nums=1, date=True)
    col, fn = rng.choice([(f.num[0], "is_numeric_dtype"), (f.key, "is_string_dtype"), (f.date, "is_datetime64_any_dtype"),
                          (f.key, "is_numeric_dtype")])
    return P(f"{f.describe()}. Check whether `{col}` has a {fn.replace('is_', '').replace('_dtype', '').replace('_any', '').replace('64', '')} "
             f"dtype with `pd.api.types.{fn}`.", f.code(), f"pd.api.types.{fn}(df[{col!r}])", require=[fn])


# ================================================================= medium

@template(M, "medium")
def shrink_frame(rng):
    n = rng.randint(6, 9)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\ndf = pd.DataFrame({{'team': g.choice(['red', 'blue', 'green'], {n}), "
             f"'age': g.integers(18, 70, {n}), 'score': g.random({n})}})")
    return P("Cut `df`'s memory: make `team` a `category`, downcast `age` to the smallest integer type "
             "(`pd.to_numeric(..., downcast='integer')`) and `score` to float32 (`downcast='float'`). Return the frame.",
             setup, "result = df.assign(team=df['team'].astype('category'),\n"
                    "                   age=pd.to_numeric(df['age'], downcast='integer'),\n"
                    "                   score=pd.to_numeric(df['score'], downcast='float'))",
             check_dtype=True)


@template(M, "medium")
def group_split(rng):
    f = frame(rng, (8, 12), nums=1)
    for i in range(0, f.n, 3):  # repeat some entities so a random row split would leak
        f.cols[f.key][i] = f.cols[f.key][0]
    k = rng.randint(1, 2)
    return P(f"{f.describe()}; some `{f.key}` values repeat. Split into train/validation **by entity** so no `{f.key}` "
             f"is in both: the first {k} distinct `{f.key}` values (in order of appearance, `unique()`) go to "
             f"validation. Create `train_df` and `val_df`.", f.code(),
             f"val_names = set(df[{f.key!r}].unique()[:{k}])\nis_val = df[{f.key!r}].isin(val_names)\n"
             "train_df, val_df = df[~is_val], df[is_val]", answer=["train_df", "val_df"], require=["isin|unique"])


@template(M, "medium")
def frame_assertions(rng):
    f = frame(rng, (4, 6), nums=1)
    a = f.num[0]
    same = rng.random() < 0.5
    tweak = "" if same else f"\ngot.loc[0, {a!r}] = got.loc[0, {a!r}] + 1"
    return P(f"{f.describe()}. A pipeline test compares its output `got` with `expected`. Call "
             f"`pd.testing.assert_frame_equal(got, expected)` inside `try` / `except AssertionError` and set `ok` to True "
             f"if it passes, False otherwise.", f.code() + "\nexpected = df.copy()\ngot = df.copy()" + tweak,
             "try:\n    pd.testing.assert_frame_equal(got, expected)\n    ok = True\nexcept AssertionError:\n    ok = False",
             answer=["ok"], require=["assert_frame_equal"])


@template(M, "medium")
def vectorize_rewrite(rng):
    f = frame(rng, (6, 9), nums=2)
    a, b = f.num
    t = threshold(rng, f.cols[a], f.nums[0])
    return P(f"{f.describe()}. This is slow on big data:\n\n```\ndf.apply(lambda r: r[{a!r}] * 2 if r[{a!r}] > {t} else "
             f"r[{b!r}], axis=1)\n```\n\nRewrite it **vectorized** with `np.where` (no `apply`, no loops) and return the "
             f"result as a Series.", f.code(), f"pd.Series(np.where(df[{a!r}] > {t}, df[{a!r}] * 2, df[{b!r}]), index=df.index)",
             require=["where"], no_loops=True)


# ================================================================= hard

@template(M, "hard")
def method_chain_summary(rng):
    f = frame(rng, (9, 12), nums=2, nan=rng.randint(1, 2))
    a, b = f.num
    cut = sorted(v for v in f.cols[b])[f.n // 2]
    return P(f"{f.describe()}. Rewrite a 5-step cleaning script as **one method chain**: drop rows with missing `{a}`; "
             f"add `band` = `pd.cut({b}, [-inf, {cut}, inf], labels=['low', 'high'])`; group by `{f.group}` and `band` "
             f"(`observed=True`); aggregate `n` (row count) and `mean_{a}`; `reset_index()`; sort by `mean_{a}` "
             f"descending.", f.code(),
             f"result = (df.dropna(subset=[{a!r}])\n    .assign(band=lambda d: pd.cut(d[{b!r}], [-np.inf, {cut}, np.inf], labels=['low', 'high']))\n"
             f"    .groupby([{f.group!r}, 'band'], observed=True)\n    .agg(n=({f.key!r}, 'size'), mean_{a}=({a!r}, 'mean'))\n"
             f"    .reset_index()\n    .sort_values('mean_{a}', ascending=False))")


@template(M, "hard")
def model_inputs(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. Prepare model inputs: `X`, a **float32** matrix of `{a}`, `{b}` and the one-hot columns of "
             f"`{f.group}` (`pd.get_dummies(..., dtype=int)`, all levels, in that order), and `y`, the int codes of "
             f"`{f.group}` as a category (`.cat.codes.to_numpy()`).", f.code(),
             f"X = pd.concat([df[[{a!r}, {b!r}]], pd.get_dummies(df[{f.group!r}], dtype=int)], axis=1).to_numpy(dtype=np.float32)\n"
             f"y = df[{f.group!r}].astype('category').cat.codes.to_numpy()", answer=["X", "y"], check_dtype=True)


@template(M, "hard")
def shrink_at_load(rng):
    f = frame(rng, (6, 9), nums=2)
    a = f.num[0]
    return P(f"`data.csv` holds a table of {f.th.plural}. Shrink it **while loading**: read only `{f.group}` and `{a}` "
             f"(`usecols`) with `{f.group}` as a `category` (`dtype=`). Return the frame's `dtypes`.",
             f.code() + "\ndf.to_csv('data.csv', index=False)\ndel df",
             f"pd.read_csv('data.csv', usecols=[{f.group!r}, {a!r}], dtype={{{f.group!r}: 'category'}}).dtypes",
             require=["read_csv"])


# ================================================================= challenge

@template(M, "challenge")
def pairs_sum_divisible(rng):
    n = rng.randint(6, 9)
    x = [rng.randint(0, 30) for _ in range(n)]
    k = rng.randint(2, 6)
    sol = ("from collections import Counter\nres = Counter(v % k for v in s.tolist())\ntotal = res[0] * (res[0] - 1) // 2\n"
           "for r in range(1, (k + 1) // 2):\n    total += res[r] * res[k - r]\nif k % 2 == 0:\n"
           "    total += res[k // 2] * (res[k // 2] - 1) // 2\nresult = total")
    tests = [rand_ints("s", seeded(rng), 0, 10**9, 20000).replace("s = ", "s = pd.Series(") + ")\nk = 97",
             rand_ints("s", seeded(rng), 0, 10**9, 20000).replace("s = ", "s = pd.Series(") + ")\nk = 2",
             "s = pd.Series(np.zeros(20000, dtype=int))\nk = 1000", "s = pd.Series([4])\nk = 2"]
    return challenge("Pairs that fill the batch",
                     "`s` holds the token counts of n sequences. Two sequences pack perfectly into GPU tiles when their "
                     "**total is divisible by `k`**. Count the pairs `(i, j)` with `i < j` whose sum is divisible by `k`.",
                     f"s = pd.Series({lit(x)})\nk = {k}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ k ≤ 10⁴",
                     speed="O(n + k): count the remainders")


@template(M, "challenge")
def max_subarray_mod(rng):
    n = rng.randint(5, 8)
    x = [rng.randint(0, 20) for _ in range(n)]
    m = rng.randint(5, 13)
    sol = ("from bisect import bisect_right, insort\nseen = [0]\nprefix = 0\nbest = 0\nfor v in s.tolist():\n"
           "    prefix = (prefix + v) % m\n    best = max(best, prefix)\n    i = bisect_right(seen, prefix)\n"
           "    if i < len(seen):\n        best = max(best, (prefix - seen[i]) % m)\n    insort(seen, prefix)\nresult = best")
    tests = [rand_ints("s", seeded(rng), 0, 10**9, 20000).replace("s = ", "s = pd.Series(") + ")\nm = 10**9 + 7",
             rand_ints("s", seeded(rng), 0, 100, 20000).replace("s = ", "s = pd.Series(") + ")\nm = 7",
             "s = pd.Series([3, 3, 9, 9, 5])\nm = 7", "s = pd.Series([6])\nm = 6"]
    return challenge("Best window modulo m",
                     "A hashing scheme only cares about sums **modulo m**. `s` holds non-negative integers. Over all "
                     "contiguous windows, return the **maximum value of `sum(window) % m`**.",
                     f"s = pd.Series({lit(x)})\nm = {m}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ m ≤ 10⁹ + 7",
                     speed="O(n log n): for each prefix, the smallest larger earlier prefix gives the best wrap-around")

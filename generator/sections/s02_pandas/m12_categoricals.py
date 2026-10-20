"""2.12 Categoricals & binning."""

from ._common import P, challenge, frame, lit, seeded, template

M = "2.12"

ORDERED = (("size", ["S", "M", "L", "XL"]), ("priority", ["low", "medium", "high", "urgent"]),
           ("rating", ["poor", "fair", "good", "excellent"]), ("level", ["junior", "mid", "senior", "staff"]))


def _ordered(rng, n=(5, 8)):
    name, levels = rng.choice(ORDERED)
    vals = [rng.choice(levels) for _ in range(rng.randint(*n))]
    return name, levels, vals


# ================================================================= lite

@template(M, "lite")
def to_category(rng):
    f = frame(rng, (5, 8), nums=1)
    g = f.group
    desc, sol = rng.choice([
        (f"`{g}` converted to the `category` dtype", f"df[{g!r}].astype('category')"),
        (f"the **categories** of `{g}` after converting it to `category` (`.cat.categories`)",
         f"df[{g!r}].astype('category').cat.categories"),
        (f"the integer **codes** of `{g}` after converting it to `category` (`.cat.codes`), handy as model inputs",
         f"df[{g!r}].astype('category').cat.codes"),
    ])
    return P(f"{f.describe()}. Get {desc}.", f.code(), sol, require=["astype|Categorical|CategoricalDtype"])


@template(M, "lite")
def cut_basic(rng):
    f = frame(rng, (6, 9), nums=1)
    a = f.num[0]
    k = rng.randint(2, 4)
    lo, hi = min(f.cols[a]), max(f.cols[a])
    mid = round((lo + hi) / 2, 1)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Bin `{a}` into {k} **equal-width** bins with `pd.cut(..., bins={k})`.", f.code(),
                 f"pd.cut(df[{a!r}], bins={k})", require=["cut"])
    return P(f"{f.describe()}. Bin `{a}` into two labelled bins with `pd.cut`: edges `[{lo - 1}, {mid}, {hi + 1}]` and "
             f"labels `['low', 'high']`.", f.code(), f"pd.cut(df[{a!r}], bins=[{lo - 1}, {mid}, {hi + 1}], labels=['low', 'high'])",
             require=["cut"])


# ================================================================= easy

@template(M, "easy")
def qcut_basic(rng):
    f = frame(rng, (8, 12), nums=1, distinct=True)
    a = f.num[0]
    q = rng.choice([2, 4])
    desc, sol = rng.choice([
        (f"{q} **equal-count** bins (`pd.qcut(..., q={q})`)", f"pd.qcut(df[{a!r}], q={q})"),
        (f"{q} equal-count bins labelled `Q1`…`Q{q}`", f"pd.qcut(df[{a!r}], q={q}, labels={[f'Q{i}' for i in range(1, q + 1)]!r})"),
        (f"{q} equal-count bins, returning just the integer bin numbers (`labels=False`)", f"pd.qcut(df[{a!r}], q={q}, labels=False)"),
    ])
    return P(f"{f.describe()}. Bin `{a}` into {desc}.", f.code(), sol, require=["qcut"])


@template(M, "easy")
def ordered_categories(rng):
    name, levels, vals = _ordered(rng)
    cut = rng.choice(levels[1:3])
    setup = f"s = pd.Series(pd.Categorical({vals!r}, categories={levels!r}, ordered=True))"
    desc, sol = rng.choice([
        (f"the entries that are **at least** `{cut!r}` (ordered categories support `>=`)", f"s[s >= {cut!r}]"),
        (f"`s` sorted by the category order ({' < '.join(levels)}), not alphabetically", "s.sort_values()"),
        ("the **largest** value in category order (`max()`)", "s.max()"),
    ])
    return P(f"`s` holds `{name}` values as an **ordered** categorical ({' < '.join(levels)}). Get {desc}.", setup, sol)


@template(M, "easy")
def one_hot(rng):
    f = frame(rng, (5, 7), nums=1)
    g = f.group
    desc, sol = rng.choice([
        (f"`{g}` as 0/1 indicator columns with the prefix `{g}` (`pd.get_dummies(..., prefix=..., dtype=int)`)",
         f"pd.get_dummies(df[{g!r}], prefix={g!r}, dtype=int)"),
        (f"the whole frame with `{g}` replaced by 0/1 columns, **dropping the first level** to avoid collinearity "
         f"(`pd.get_dummies(df, columns=[...], drop_first=True, dtype=int)`)",
         f"pd.get_dummies(df, columns=[{g!r}], drop_first=True, dtype=int)"),
    ])
    return P(f"{f.describe()}. One-hot encode: return {desc}.", f.code(), sol, require=["get_dummies"])


# ================================================================= medium

@template(M, "medium")
def category_methods(rng):
    f = frame(rng, (5, 7), nums=1)
    g = f.group
    lv = list(f.levels)
    setup = f.code() + f"\ndf[{g!r}] = df[{g!r}].astype('category')"
    desc, sol = rng.choice([
        (f"rename the category `{lv[0]!r}` to `'X'` (`cat.rename_categories`)", f"df[{g!r}].cat.rename_categories({{{lv[0]!r}: 'X'}})"),
        ("add an unused category `'other'` (`cat.add_categories`)", f"df[{g!r}].cat.add_categories(['other'])"),
        (f"set the categories to exactly `{lv[::-1] + ['other']}` (`cat.set_categories`)",
         f"df[{g!r}].cat.set_categories({lv[::-1] + ['other']!r})"),
        ("drop categories that no row uses (`cat.remove_unused_categories`) after adding `'tmp'`",
         f"df[{g!r}].cat.add_categories(['tmp']).cat.remove_unused_categories()"),
    ])
    return P(f"{f.describe()}; `{g}` is already a `category`. Using the `.cat` accessor, {desc}. Return the Series.",
             setup, sol, require=["cat"], check_dtype=True)


@template(M, "medium")
def factorize_labels(rng):
    f = frame(rng, (6, 9), nums=1)
    g = f.group
    if rng.random() < 0.5:
        return P(f"{f.describe()}. **Label-encode** `{g}` with `pd.factorize`: return the tuple `(codes, uniques)`; codes "
                 f"follow the order of first appearance.", f.code(), f"pd.factorize(df[{g!r}])", require=["factorize"])
    return P(f"{f.describe()}. Label-encode `{g}` with `pd.factorize(..., sort=True)` so codes follow alphabetical order; "
             f"return just the codes.", f.code(), f"pd.factorize(df[{g!r}], sort=True)[0]", require=["factorize"])


@template(M, "medium")
def interval_edges(rng):
    f = frame(rng, (8, 11), nums=1)
    a = f.num[0]
    s = sorted(set(f.cols[a]))
    edges = [s[0], s[len(s) // 3], s[2 * len(s) // 3], s[-1] + 1]
    edges = sorted(set(edges))
    labels = ["low", "mid", "high", "max"][:len(edges) - 1]
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Bin `{a}` with edges `{edges}` and labels `{labels}`, where each bin **includes its "
                 f"left edge** and excludes its right one (`right=False`).", f.code(),
                 f"pd.cut(df[{a!r}], bins={edges!r}, labels={labels!r}, right=False)", require=["cut"])
    return P(f"{f.describe()}. Bin `{a}` with edges `{edges}` and labels `{labels}` (`right=False`), then count the rows "
             f"in each bin **in bin order** (`value_counts(sort=False)`).", f.code(),
             f"pd.cut(df[{a!r}], bins={edges!r}, labels={labels!r}, right=False).value_counts(sort=False)", require=["cut"])


@template(M, "medium")
def observed_groups(rng):
    f = frame(rng, (6, 9), nums=1)
    g, a = f.group, f.num[0]
    setup = f.code() + f"\ndf[{g!r}] = pd.Categorical(df[{g!r}], categories={list(f.levels) + ['unused']!r})"
    obs = rng.random() < 0.5
    return P(f"{f.describe()}; `{g}` is a categorical that also lists an `'unused'` category nobody belongs to. Compute "
             f"the mean `{a}` per `{g}` {'**skipping**' if obs else '**including**'} empty categories "
             f"(`groupby(..., observed={obs})`).", setup, f"df.groupby({g!r}, observed={obs})[{a!r}].mean()",
             require=["observed"])


# ================================================================= hard

@template(M, "hard")
def band_report(rng):
    f = frame(rng, (9, 12), nums=2)
    a, b = f.num
    s = sorted(f.cols[a])
    cuts = [s[0] - 1, s[len(s) // 2], s[-1] + 1]
    return P(f"{f.describe()}. Bin `{a}` into `'lower'` / `'upper'` halves with `pd.cut` (edges `{cuts}`), then report "
             f"per band (with named aggregation, `observed=True`): `n` (row count) and `mean_{b}` (mean `{b}`).",
             f.code(), f"band = pd.cut(df[{a!r}], bins={cuts!r}, labels=['lower', 'upper'])\n"
             f"result = df.groupby(band, observed=True).agg(n=({b!r}, 'size'), mean_{b}=({b!r}, 'mean'))",
             require=["cut"])


@template(M, "hard")
def model_matrix(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. Build a model-ready matrix: drop `{f.key}`, one-hot encode `{f.group}` with "
             f"`drop_first=True` and `dtype=int`, and return the result as a `float32` NumPy array (`to_numpy`).",
             f.code(), f"pd.get_dummies(df.drop(columns=[{f.key!r}]), columns=[{f.group!r}], drop_first=True, dtype=int)"
             ".to_numpy(dtype='float32')", require=["get_dummies"], check_dtype=True)


@template(M, "hard")
def ordered_filter_report(rng):
    name, levels, vals = _ordered(rng, (7, 10))
    th = rng.choice(levels[1:3])
    scores = [rng.randint(1, 99) for _ in vals]
    return P(f"`df` has an ordered `{name}` column ({' < '.join(levels)}) and a `score`. Keep the rows whose `{name}` is "
             f"at least `{th!r}`, and return the mean `score` per `{name}` level in category order (only levels that "
             f"occur, `observed=True`).",
             f"df = pd.DataFrame({{{name!r}: pd.Categorical({vals!r}, categories={levels!r}, ordered=True), 'score': {scores!r}}})",
             f"df[df[{name!r}] >= {th!r}].groupby({name!r}, observed=True)['score'].mean()")


# ================================================================= challenge

@template(M, "challenge")
def optimal_binning(rng):
    n, k = rng.randint(5, 7), rng.randint(2, 3)
    vals = sorted(rng.randint(0, 30) for _ in range(n))
    sol = ("v = sorted(x.tolist())\nn = len(v)\npre, pre2 = [0.0], [0.0]\nfor t in v:\n    pre.append(pre[-1] + t)\n"
           "    pre2.append(pre2[-1] + t * t)\n\ndef cost(i, j):  # squared error of v[i:j] around its mean\n"
           "    s, s2, m = pre[j] - pre[i], pre2[j] - pre2[i], j - i\n    return s2 - s * s / m\n\n"
           "INF = float('inf')\ndp = [0.0] + [INF] * n\nfor _ in range(k):\n    new = [INF] * (n + 1)\n"
           "    for j in range(1, n + 1):\n        new[j] = min(dp[i] + cost(i, j) for i in range(j))\n    dp = new\n"
           "result = round(dp[n], 6)")

    def big(seed, n_, k_):
        return f"x = pd.Series(np.random.default_rng({seed}).integers(0, 1000, {n_}))\nk = {k_}"

    tests = [big(seeded(rng), 200, 8), big(seeded(rng), 200, 1), big(seeded(rng), 120, 30), "x = pd.Series([5, 5])\nk = 2"]
    return challenge("Optimal bins",
                     "`x` holds latency measurements. Sort them and split the sorted values into **exactly k non-empty "
                     "contiguous bins**. A bin's cost is the sum of squared differences from the bin's mean. Return the "
                     "**minimum total cost**, rounded to 6 decimals (this is optimal 1-D k-means).",
                     f"x = pd.Series({lit(vals)})\nk = {k}", sol, tests, "1 ≤ k ≤ n ≤ 200",
                     speed="O(k·n²) dynamic programming with prefix sums")


@template(M, "challenge")
def balanced_window(rng):
    cats = rng.sample(["cat", "dog", "bird"], rng.randint(2, 3))
    vals = [rng.choice(cats) for _ in range(rng.randint(8, 11))]
    sol = ("levels = sorted(set(labels.tolist()))\nidx = {c: i for i, c in enumerate(levels)}\ncounts = [0] * len(levels)\n"
           "first = {tuple([0] * (len(levels) - 1)): -1}\nbest = 0\nfor i, c in enumerate(labels.tolist()):\n"
           "    counts[idx[c]] += 1\n    key = tuple(v - counts[0] for v in counts[1:])\n    if key in first:\n"
           "        best = max(best, i - first[key])\n    else:\n        first[key] = i\n"
           "result = best if len(levels) > 1 else 0")

    def big(seed, n_, k):
        return f"g = np.random.default_rng({seed})\nlabels = pd.Series(np.array({['c' + str(i) for i in range(k)]!r})[g.integers(0, {k}, {n_})])"

    tests = [big(seeded(rng), 20000, 3), big(seeded(rng), 20000, 4), big(seeded(rng), 20000, 2), "labels = pd.Series(['a', 'a'])"]
    return challenge("Longest balanced window",
                     "`labels` is a categorical Series of class labels in arrival order. Find the length of the "
                     "**longest contiguous window** in which **every class that appears anywhere in `labels`** occurs "
                     "the **same number of times** (each at least once). Return 0 if `labels` has fewer than two classes "
                     "or no window qualifies.", f"labels = pd.Series({vals!r}, dtype='category')", sol, tests,
                     "≤ 2·10⁴ labels, ≤ 4 classes", speed="O(n·classes): hash the count differences of prefixes")

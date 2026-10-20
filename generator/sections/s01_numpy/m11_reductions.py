"""1.11 Reductions & statistics."""

from ... import themes as T
from ._common import (P, about, an, arr, challenge, feature_matrix, lit, mat, pick_measure, rand_ints,
                      seeded, series, table, template)

M = "1.11"


# ================================================================= lite

@template(M, "lite")
def reduce_all(rng):
    th, m, _, vals = series(rng)
    v = m.var
    desc, sol = rng.choice([
        (f"the total {m.name}", f"{v}.sum()"), (f"the average {m.name}", f"{v}.mean()"),
        (f"the highest {m.name}", f"{v}.max()"), (f"the lowest {m.name}", f"{v}.min()"),
        (f"the median {m.name}", f"np.median({v})"),
        (f"the standard deviation of the {m.name} (population std, NumPy's default)", f"{v}.std()"),
        (f"the range of the {m.name} (highest minus lowest, `np.ptp`)", f"np.ptp({v})"),
    ])
    return P(f"{about(th, m, len(vals), v)}. Compute {desc} across all of them.", f"{v} = {arr(vals)}", sol)


@template(M, "lite")
def sample_std(rng):
    th, m, _, vals = series(rng)
    v = m.var
    word, fn = rng.choice([("standard deviation", "std"), ("variance", "var")])
    return P(f"{about(th, m, len(vals), v)}. Compute the **sample** {word} (`ddof=1`, the default in pandas "
             f"and in most statistics books), not NumPy's default `ddof=0`.", f"{v} = {arr(vals)}",
             f"{v}.{fn}(ddof=1)", require=["ddof"])


# ================================================================= easy

@template(M, "easy")
def axis_reduce(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    word, fn = rng.choice([("average", "mean"), ("total", "sum"), ("highest", "max"), ("lowest", "min")])
    if rng.random() < 0.5:
        desc, a = f"for each {th.noun} (across all {ax.plural})", 1
    else:
        desc, a = f"for each {ax.noun} (across all {th.plural})", 0
    return P(f"`{v}` is a table of {m.label} with one row per {th.noun} and one column per "
             f"{ax.noun}. Compute the {word} {desc} using the `axis` argument.",
             f"{v} = {mat(data)}", f"{v}.{fn}(axis={a})", require=["axis"])


@template(M, "easy")
def percentile_stat(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    if rng.random() < 0.3:
        return P(f"{about(th, m, len(vals), v)}. Compute the **median** value.",
                 f"{v} = {arr(vals)}", f"np.median({v})", require=["median|percentile|quantile"])
    if rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. Compute the **10th, 50th and 90th percentiles** in one call "
                 f"(pass a list to `np.percentile`).", f"{v} = {arr(vals)}", f"np.percentile({v}, [10, 50, 90])",
                 require=["percentile|quantile"])
    q = rng.choice([10, 25, 75, 90, 95])
    return P(f"{about(th, m, len(vals), v)}. Compute the **{q}th percentile** (NumPy's default "
             f"linear interpolation).", f"{v} = {arr(vals)}", f"np.percentile({v}, {q})",
             require=["percentile|quantile"])


@template(M, "easy")
def running_total(rng):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=5, max_labels=12)
    who = rng.choice(th.items)
    n = len(ax.labels)
    m = pick_measure(rng, th, n, positive=True)
    vals = T.values(rng, m, n)
    v = m.var
    if rng.random() < 0.7:
        return P(f"`{v}` holds the {m.label} of the {who} for each of {n} {ax.plural}, in order. "
                 f"Compute the **running total** with `np.cumsum`.",
                 f"{v} = {arr(vals)}", f"np.cumsum({v})", require=["cumsum|cumulative_sum"])
    g = [round(rng.uniform(0.9, 1.2), 2) for _ in range(n)]
    return P(f"`growth` holds the growth factor of the {who}'s {m.name} for each of {n} {ax.plural} "
             f"(1.05 = +5%). Compute the **cumulative growth** after each {ax.noun} with `np.cumprod`.",
             f"growth = {arr(g)}", "np.cumprod(growth)", require=["cumprod"])


@template(M, "easy")
def keepdims_and_coords(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 5)
    s = seeded(rng)
    setup = f"M = np.random.default_rng({s}).permutation({r * c}).reshape({r}, {c})"
    desc, sol, req = rng.choice([
        (f"the mean of each **row** as a column vector of shape `({r}, 1)` (`keepdims=True`)",
         "M.mean(axis=1, keepdims=True)", "keepdims"),
        (f"the maximum of each **column** as a row vector of shape `(1, {c})` (`keepdims=True`)",
         "M.max(axis=0, keepdims=True)", "keepdims"),
        ("the `(row, col)` position of the **global maximum** as a tuple (`argmax` gives a flat index; convert "
         "it with `np.unravel_index`)", "np.unravel_index(M.argmax(), M.shape)", "unravel_index"),
        ("the `(row, col)` position of the **global minimum** as a tuple (`np.unravel_index` on the flat "
         "argmin)", "np.unravel_index(M.argmin(), M.shape)", "unravel_index"),
    ])
    return P(f"`M` is a `({r}, {c})` score matrix. Get {desc}.", setup, sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def group_totals(rng):
    th = T.theme(rng)
    k = rng.randint(3, 5)
    names = T.items(rng, th, k)
    m = pick_measure(rng, th, 10, positive=True)
    n = rng.randint(8, 14)
    cat = list(range(k)) + [rng.randrange(k) for _ in range(n - k)]
    rng.shuffle(cat)
    vals = T.values(rng, m, n)
    v = m.var
    ctx = (f"Each of {n} log records belongs to one {th.noun}: `cat[i]` is an index into "
           f"`{th.var}` and `{v}[i]` is the record's {m.label}.")
    setup = f"{th.var} = {lit(names)}\ncat = {arr(cat)}\n{v} = {arr(vals)}"
    variant = rng.choice(["sum", "count", "mean"])
    if variant == "sum":
        return P(f"{ctx} Without loops, compute the **sum** of `{v}` per {th.noun} as an array of "
                 f"length {k} in index order (`np.bincount` with `weights=`).",
                 setup, f"np.bincount(cat, weights={v}, minlength={k})", no_loops=True)
    if variant == "count":
        return P(f"{ctx} Without loops, count the **records** per {th.noun} as an array of length "
                 f"{k} in index order (`np.bincount`).", setup, f"np.bincount(cat, minlength={k})", no_loops=True)
    return P(f"{ctx} Without loops, compute the **mean** of `{v}` per {th.noun} as an array of length "
             f"{k} in index order (sum per group / count per group).", setup,
             f"np.bincount(cat, weights={v}, minlength={k}) / np.bincount(cat, minlength={k})", no_loops=True)


@template(M, "medium")
def weighted_average(rng):
    th, m, _, vals = series(rng, 4, 7)
    v = m.var
    w = [rng.randint(1, 9) for _ in vals]
    if rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}; `w` holds how many samples each one was measured on. Compute "
                 f"the **weighted mean** with `np.average(..., weights=w)`.", f"{v} = {arr(vals)}\nw = {arr(w)}",
                 f"np.average({v}, weights=w)", require=["average"])
    r, c = rng.randint(2, 4), rng.randint(3, 4)
    s = seeded(rng)
    bw = [rng.randint(1, 5) for _ in range(c)]
    return P(f"`scores` holds {r} models' scores (rows) on {c} benchmarks (columns); `w` holds the importance "
             f"of each benchmark. Compute each model's **weighted average** score across benchmarks (shape "
             f"`({r},)`) with `np.average(..., axis=1, weights=w)`.",
             f"scores = np.random.default_rng({s}).integers(40, 100, size=({r}, {c}))\nw = {arr(bw)}",
             "np.average(scores, axis=1, weights=w)", require=["average"])


@template(M, "medium")
def histogram_counts(rng):
    th, m, _, vals = series(rng, 10, 14)
    v = m.var
    lo, hi = min(vals), max(vals)
    if rng.random() < 0.5:
        k = rng.randint(3, 5)
        return P(f"{about(th, m, len(vals), v)}. Count how many values fall into each of {k} equal-width bins "
                 f"spanning the data (`np.histogram(..., bins={k})`). Return just the counts.",
                 f"{v} = {arr(vals)}", f"np.histogram({v}, bins={k})[0]", require=["histogram"])
    edges = sorted({lo, hi, *(T.value(rng, m) for _ in range(2))})
    edges[0], edges[-1] = min(edges[0], lo), max(edges[-1], hi)
    return P(f"{about(th, m, len(vals), v)}. Count how many values fall into the bins with edges `edges` "
             f"(`np.histogram(..., bins=edges)`; the last bin includes its right edge). Return just the counts.",
             f"{v} = {arr(vals)}\nedges = {arr(edges)}", f"np.histogram({v}, bins=edges)[0]", require=["histogram"])


@template(M, "medium")
def correlation(rng):
    th, ms, names, X = feature_matrix(rng, 6, 9)
    feats = ", ".join(f"column {i} = {m.label}" for i, m in enumerate(ms))
    i, j = rng.sample(range(len(ms)), 2)
    desc, sol, req = rng.choice([
        (f"the **Pearson correlation** between column {i} and column {j} (one number from `np.corrcoef`)",
         f"np.corrcoef(X[:, {i}], X[:, {j}])[0, 1]", "corrcoef"),
        ("the full `(3, 3)` **correlation matrix** of the columns (`np.corrcoef` with `rowvar=False`)",
         "np.corrcoef(X, rowvar=False)", "corrcoef"),
        ("the `(3, 3)` **covariance matrix** of the columns (`np.cov` with `rowvar=False`)",
         "np.cov(X, rowvar=False)", "cov"),
    ])
    return P(f"`X` has one row per {th.noun} ({feats}). Compute {desc}.", f"X = {mat(X)}", sol, require=[req])


@template(M, "medium")
def masked_reductions(rng):
    r, c = rng.randint(3, 5), rng.randint(4, 6)
    s = seeded(rng)
    setup = f"M = np.random.default_rng({s}).integers(-5, 10, size=({r}, {c}))"
    desc, sol, req = rng.choice([
        ("the mean of each row over its **positive** entries only (`np.mean(..., where=M > 0)`)",
         "np.mean(M, axis=1, where=M > 0)", "where"),
        ("the sum of each column over its **non-negative** entries only (`np.sum(..., where=M >= 0)`)",
         "np.sum(M, axis=0, where=M >= 0)", "where"),
        ("for each row, the **count** of entries greater than 3 (`np.count_nonzero(..., axis=1)`)",
         "np.count_nonzero(M > 3, axis=1)", "count_nonzero|sum"),
        ("whether **every** row has at least one negative entry (`np.any` per row, then `np.all`)",
         "np.all(np.any(M < 0, axis=1))", "any|all"),
    ])
    return P(f"`M` is a `({r}, {c})` table of rewards (rows = episodes). Compute {desc}.", setup, sol,
             require=[req])


# ================================================================= hard

@template(M, "hard")
def per_class_accuracy(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 4))
    n = rng.randint(12, 20)
    yt = list(range(k)) + [rng.randrange(k) for _ in range(n - k)]
    yp = [t if rng.random() < 0.65 else rng.randrange(k) for t in yt]
    return P(f"{an(what, cap=True)} classifier with {k} classes (`classes`) made predictions `y_pred` for {n} samples with "
             f"true labels `y_true`. Compute the **accuracy per class** (correct predictions of class i / true "
             f"samples of class i) as an array of length {k}, with `np.bincount` and no loops.",
             f"classes = {lit(list(classes[:k]))}\ny_true = {arr(yt)}\ny_pred = {arr(yp)}",
             f"correct = np.bincount(y_true, weights=(y_true == y_pred), minlength={k})\n"
             f"result = correct / np.bincount(y_true, minlength={k})", no_loops=True)


@template(M, "hard")
def zscore_outliers(rng):
    r, c = rng.randint(12, 20), rng.randint(2, 4)
    s = seeded(rng)
    z = rng.choice([1.5, 2.0, 2.5])
    return P(f"`X` holds {r} sensor readings (rows) of {c} channels (columns), with some spikes. For **each "
             f"channel**, count the readings whose z-score `|x − mean| / std` (per-column mean and population "
             f"std) is greater than {z}. Return the counts, shape `({c},)`. No loops.",
             f"g = np.random.default_rng({s})\nX = g.normal(size=({r}, {c})).round(2)\nX[g.integers(0, {r}, 3), g.integers(0, {c}, 3)] += 6",
             f"zs = np.abs(X - X.mean(axis=0)) / X.std(axis=0)\nresult = (zs > {z}).sum(axis=0)", no_loops=True)


@template(M, "hard")
def robust_scale(rng):
    r, c = rng.randint(8, 14), rng.randint(2, 4)
    s = seeded(rng)
    return P(f"Scale the `({r}, {c})` feature matrix `X` **robustly** (outliers barely matter): for every "
             f"column subtract its median and divide by its interquartile range (75th − 25th percentile). "
             f"Compute the statistics with `np.median` / `np.percentile(..., axis=0)`. No loops.",
             f"g = np.random.default_rng({s})\nX = g.lognormal(0, 1, size=({r}, {c})).round(3)",
             "q75, q25 = np.percentile(X, [75, 25], axis=0)\nresult = (X - np.median(X, axis=0)) / (q75 - q25)",
             no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def count_subarrays_sum_k(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(-4, 5) for _ in range(n)]
    K = rng.randint(-2, 6)
    var, story = rng.choice([
        ("reward", "`reward[i]` is the reward an RL agent received at step i (can be negative)."),
        ("delta", "`delta[i]` is the day-over-day change in a model's leaderboard rank points (can be negative)."),
        ("balance", "`balance[i]` is the net GPU-hours a team gained (+) or spent (−) on day i."),
    ])
    sol = (f"from collections import defaultdict\nseen = defaultdict(int)\nseen[0] = 1\nprefix = count = 0\n"
           f"for x in {var}.tolist():\n    prefix += x\n    count += seen[prefix - K]\n    seen[prefix] += 1\n"
           "result = count")
    tests = [rand_ints(var, seeded(rng), -3, 4, 20000) + "\nK = 0",
             rand_ints(var, seeded(rng), -50, 51, 20000) + "\nK = 37",
             f"{var} = np.zeros(2000, dtype=int)\nK = 0",
             f"{var} = np.array([5])\nK = 5"]
    return challenge("Windows with an exact total",
                     story + " Count the **contiguous subarrays** (windows of length ≥ 1) whose sum is exactly `K`.",
                     f"{var} = {arr(vals)}\nK = {K}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, |values| ≤ 50, |K| ≤ 10⁶")


@template(M, "challenge")
def count_submatrices_sum_k(rng):
    n, m = rng.randint(3, 4), rng.randint(3, 5)
    G = [[rng.randint(-2, 2) for _ in range(m)] for _ in range(n)]
    K = rng.randint(-1, 3)
    sol = ("g = grid.tolist()\nn, m = len(g), len(g[0])\ncount = 0\nfor top in range(n):\n    col = [0] * m\n"
           "    for bottom in range(top, n):\n        for j in range(m):\n            col[j] += g[bottom][j]\n"
           "        seen = {0: 1}\n        prefix = 0\n        for v in col:\n            prefix += v\n"
           "            count += seen.get(prefix - K, 0)\n            seen[prefix] = seen.get(prefix, 0) + 1\n"
           "result = count")
    big = lambda s, n_, m_, lo, hi, k: (f"grid = np.random.default_rng({s}).integers({lo}, {hi}, size=({n_}, {m_}))\nK = {k}")  # noqa: E731
    tests = [big(seeded(rng), 30, 200, -1, 2, 0), big(seeded(rng), 30, 200, -9, 10, 5),
             "grid = np.zeros((30, 200), dtype=int)\nK = 0", "grid = np.array([[3]])\nK = 3"]
    return challenge("Heat-map blocks with an exact total",
                     "`grid` is a heat map of net changes (rows = days, columns = regions). Count the "
                     "**axis-aligned rectangles** (any contiguous block of rows × contiguous block of "
                     "columns) whose values sum to exactly `K`.", f"grid = {mat(G)}\nK = {K}", sol, tests,
                     "1 ≤ n ≤ 30, 1 ≤ m ≤ 200, |grid| ≤ 10",
                     speed="O(n²·m): fix the top and bottom rows, then count column-prefix matches")

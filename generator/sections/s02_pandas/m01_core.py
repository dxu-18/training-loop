"""2.01 Setup, options & core objects (Series, Index, options)."""

from ... import themes as T
from ._common import STDOUT, P, challenge, floats, lit, seeded, template

M = "2.01"

OPTIONS = (("display.max_rows", "maximum number of rows printed before pandas truncates"),
           ("display.max_columns", "maximum number of columns printed"),
           ("display.precision", "number of decimals shown for floats"),
           ("display.width", "width of the console in characters"))


def _series(rng, n_lo=4, n_hi=7):
    th = T.theme(rng)
    n = rng.randint(n_lo, n_hi)
    m = rng.choice(th.measures)
    return th, m, T.items(rng, th, n), T.values(rng, m, n, distinct=True)


def _series_code(th, m, labels, vals, var="s"):
    return f"{var} = pd.Series({lit(vals)}, index={lit(labels)}, name={m.var!r})"


# ================================================================= lite

@template(M, "lite")
def version_and_options(rng):
    ctx = rng.choice(["At the top of a new notebook", "Before reporting a bug", "While debugging a teammate's script"])
    if rng.random() < 0.25:
        return P(f"{ctx}, check which pandas you are running: get the **version string**.", "",
                 "pd.__version__", require=["__version__"])
    key, what = rng.choice(OPTIONS)
    return P(f"{ctx}, read the current value of the option `{key}` (the {what}) with `pd.get_option`.", "",
             f"pd.get_option({key!r})", require=["get_option"])


@template(M, "lite")
def build_series(rng):
    th, m, labels, vals = _series(rng)
    variant = rng.choice(["labels", "dict", "scalar", "float32"])
    if variant == "labels":
        return P(f"Build a Series of the {m.label} of these {th.plural}: values `vals`, index `labels`, and the "
                 f"name `{m.var!r}` (`pd.Series(..., index=..., name=...)`).",
                 f"labels = {lit(labels)}\nvals = {lit(vals)}", f"pd.Series(vals, index=labels, name={m.var!r})",
                 post="{'name': _.name, 'series': _}")
    if variant == "dict":
        d = dict(zip(labels, vals))
        return P(f"`d` maps each {th.noun} to its {m.label}. Turn it into a Series with `pd.Series` (keys become "
                 f"the index).", f"d = {d!r}", "pd.Series(d)")
    if variant == "scalar":
        v = rng.choice([0, 1, -1, 100])
        return P(f"Build a Series that holds the default value {v} for every {th.noun} in `labels` (pass a scalar "
                 f"and an `index=`).", f"labels = {lit(labels)}", f"pd.Series({v}, index=labels)")
    return P(f"Build a Series from the list `vals` ({m.label}) with dtype `float32` (`dtype=`).",
             f"vals = {lit(vals)}", "pd.Series(vals, dtype='float32')", check_dtype=True)


@template(M, "lite")
def series_access(rng):
    th, m, labels, vals = _series(rng, 5, 7)
    i = rng.randrange(len(labels))
    a, b = sorted(rng.sample(range(len(labels)), 2))
    t = T.threshold(rng, vals, m)
    desc, sol = rng.choice([
        (f"the value for the label `{labels[i]!r}` (label lookup `s[...]` or `s.loc[...]`)", f"s[{labels[i]!r}]"),
        (f"the value at **position** {i} (`s.iloc[...]`)", f"s.iloc[{i}]"),
        (f"the sub-Series for the labels `{labels[a]!r}` and `{labels[b]!r}` (a list of labels)",
         f"s[[{labels[a]!r}, {labels[b]!r}]]"),
        (f"only the entries greater than {t} (a boolean mask)", f"s[s > {t}]"),
        ("the index labels as a Python list", "s.index.tolist()"),
        ("the values as a NumPy array (`.to_numpy()`)", "s.to_numpy()"),
    ])
    return P(f"`s` holds the {m.label} of {len(labels)} {th.plural}, indexed by name. Get {desc}.",
             _series_code(th, m, labels, vals), sol)


# ================================================================= easy

@template(M, "easy")
def option_context_print(rng):
    variant = rng.choice(["rows", "precision", "float_format"])
    if variant == "rows":
        n, k = rng.randint(20, 60), rng.choice([4, 6, 8, 10])
        return P(f"`df` has {n} rows. **Print** it so that at most {k} rows are shown (pandas truncates the "
                 f"middle), using the temporary `with pd.option_context('display.max_rows', {k}):` block.",
                 f"df = pd.DataFrame({{'step': range({n}), 'loss': np.round(np.linspace(3, 0.5, {n}), 3)}})",
                 f"with pd.option_context('display.max_rows', {k}):\n    print(df)",
                 answer=[STDOUT], require=["option_context"])
    s = seeded(rng)
    if variant == "precision":
        p = rng.randint(1, 3)
        return P(f"**Print** `df` with {p} decimal{'s' if p > 1 else ''} for every float, without changing the "
                 f"global options: `with pd.option_context('display.precision', {p}):`.",
                 f"df = pd.DataFrame(np.random.default_rng({s}).normal(size=(4, 3)), columns=['a', 'b', 'c'])",
                 f"with pd.option_context('display.precision', {p}):\n    print(df)",
                 answer=[STDOUT], require=["option_context"])
    return P("**Print** `costs` with thousands separators and 2 decimals (e.g. `1,234.50`): use "
             "`pd.option_context('display.float_format', '{:,.2f}'.format)` in a `with` block.",
             f"costs = pd.DataFrame({{'usd': np.random.default_rng({s}).uniform(100, 99999, size=4).round(3)}})",
             "with pd.option_context('display.float_format', '{:,.2f}'.format):\n    print(costs)",
             answer=[STDOUT], require=["option_context"])


@template(M, "easy")
def index_basics(rng):
    th = T.theme(rng)
    labels = T.items(rng, th, rng.randint(4, 6))
    dup = rng.random() < 0.3
    if dup:
        labels = labels + [labels[0]]
    target = rng.choice(labels[1:-1] if dup else labels)
    desc, sol = rng.choice([
        (f"the integer **position** of the label `{target!r}` (`idx.get_loc`)", f"idx.get_loc({target!r})"),
        ("whether all labels are **unique** (`idx.is_unique`)", "idx.is_unique"),
        ("the labels as a Python **list**", "idx.tolist()"),
        ("the index's **name**", "idx.name"),
        (f"whether `{target!r}` is one of the labels (the `in` operator)", f"{target!r} in idx"),
    ])
    return P(f"`idx` is a `pd.Index` of {th.noun} names. Get {desc}.",
             f"idx = pd.Index({lit(labels)}, name={T.ident(th.noun)!r})", sol)


@template(M, "easy")
def special_indexes(rng):
    y, mo, d = rng.choice([2024, 2025]), rng.randint(1, 12), rng.randint(1, 28)
    start = f"{y}-{mo:02d}-{d:02d}"
    n = rng.randint(3, 7)
    desc, sol, req = rng.choice([
        (f"a `DatetimeIndex` of {n} consecutive **days** starting {start} (`pd.date_range` with `periods=`)",
         f"pd.date_range({start!r}, periods={n})", "date_range"),
        (f"a `DatetimeIndex` of {n} consecutive **hours** starting {start} 00:00 (`freq='h'`)",
         f"pd.date_range({start!r}, periods={n}, freq='h')", "date_range"),
        (f"a `DatetimeIndex` of {n} **month ends** starting from {start} (`freq='ME'`)",
         f"pd.date_range({start!r}, periods={n}, freq='ME')", "date_range"),
        (f"a `RangeIndex` from 0 to {n * 10} (exclusive) in steps of 10", f"pd.RangeIndex(0, {n * 10}, 10)", "RangeIndex"),
        ("a `CategoricalIndex` of the labels `['low', 'high', 'low', 'mid']`",
         "pd.CategoricalIndex(['low', 'high', 'low', 'mid'])", "CategoricalIndex"),
    ])
    return P(f"Build {desc}.", "", sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def series_alignment(rng):
    th = T.theme(rng)
    pool = T.items(rng, th, 5)
    a, b = pool[:4], pool[1:]
    va, vb = [rng.randint(1, 50) for _ in a], [rng.randint(1, 50) for _ in b]
    setup = (f"server_a = pd.Series({lit(va)}, index={lit(a)})\nserver_b = pd.Series({lit(vb)}, index={lit(b)})")
    ctx = f"`server_a` and `server_b` count requests per {th.noun} on two servers; each only saw some {th.plural}."
    variant = rng.choice(["plus", "fill", "dropna"])
    if variant == "plus":
        return P(f"{ctx} Add them with `+`. pandas **aligns on the index**, so labels missing on either side "
                 f"become NaN.", setup, "server_a + server_b")
    if variant == "fill":
        return P(f"{ctx} Compute the **total** per {th.noun}, treating a missing count as 0 "
                 f"(`server_a.add(server_b, fill_value=0)`).", setup, "server_a.add(server_b, fill_value=0)",
                 require=["add"])
    return P(f"{ctx} Add them with `+` and keep only the {th.plural} both servers saw (drop the NaNs).",
             setup, "(server_a + server_b).dropna()")


@template(M, "medium")
def set_options_repr(rng):
    s = seeded(rng)
    if rng.random() < 0.5:
        p = rng.randint(1, 3)
        return P(f"Call `pd.set_option('display.precision', {p})`, then return `repr(df)` (a string). (Options are "
                 f"global; prefer `option_context` in real code.)",
                 f"df = pd.DataFrame(np.random.default_rng({s}).random((3, 2)), columns=['p', 'q'])",
                 f"pd.set_option('display.precision', {p})\nrepr(df)", require=["set_option"])
    k = rng.choice([4, 6])
    return P(f"Call `pd.set_option('display.max_rows', {k})` so long frames are truncated, then return "
             f"`repr(df)` (a string).", f"df = pd.DataFrame({{'id': range(30)}})",
             f"pd.set_option('display.max_rows', {k})\nrepr(df)", require=["set_option"])


@template(M, "medium")
def reindex_series(rng):
    th, m, labels, vals = _series(rng, 4, 5)
    extra = [x for x in th.items if x not in labels][:2]
    order = rng.sample(labels, len(labels) - 1) + extra
    fill = rng.choice([0, -1])
    if rng.random() < 0.5:
        return P(f"`s` holds the {m.label} of some {th.plural}. Conform it to the label order in `wanted` with "
                 f"`reindex`: missing labels get NaN, labels not in `wanted` are dropped.",
                 _series_code(th, m, labels, vals) + f"\nwanted = {lit(order)}", "s.reindex(wanted)", require=["reindex"])
    return P(f"`s` holds the {m.label} of some {th.plural}. Conform it to the labels in `wanted` with `reindex`, "
             f"filling labels it doesn't have with {fill} (`fill_value=`).",
             _series_code(th, m, labels, vals) + f"\nwanted = {lit(order)}", f"s.reindex(wanted, fill_value={fill})",
             require=["reindex"])


# ================================================================= hard

@template(M, "hard")
def combine_daily_counts(rng):
    days = [f"2024-0{rng.randint(1, 9)}-{d:02d}" for d in rng.sample(range(1, 28), 6)]
    a, b = sorted(rng.sample(days, 4)), sorted(rng.sample(days, 4))
    da = {d: rng.randint(0, 30) for d in a}
    db = {d: rng.randint(0, 30) for d in b}
    t = rng.randint(15, 35)
    return P(f"Two log shards counted errors per day: `shard_a` and `shard_b` map date strings to counts. Build a "
             f"Series from each, add them with a missing day counting as 0, convert the index to datetimes "
             f"(`pd.to_datetime`), sort it by date, and return only the days with **more than {t}** errors "
             f"(as an int Series).", f"shard_a = {da!r}\nshard_b = {db!r}",
             "total = pd.Series(shard_a).add(pd.Series(shard_b), fill_value=0).astype(int)\n"
             f"total.index = pd.to_datetime(total.index)\ntotal = total.sort_index()\nresult = total[total > {t}]")


@template(M, "hard")
def above_average_names(rng):
    th, m, labels, vals = _series(rng, 6, 9)
    d = dict(zip(labels, vals))
    return P(f"`scores` maps each {th.noun} to its {m.label}. Build a Series, keep the {th.plural} **above the "
             f"mean**, sort them from highest to lowest, and return their names as a list.",
             f"scores = {d!r}", "s = pd.Series(scores)\nresult = s[s > s.mean()].sort_values(ascending=False).index.tolist()")


# ================================================================= challenge

@template(M, "challenge")
def asof_lookup(rng):
    n, q = rng.randint(5, 8), rng.randint(4, 6)
    times = sorted(rng.sample(range(0, 60), n))
    vals = floats(rng, n, 0, 10, 1)
    qs = [rng.randint(-5, 65) for _ in range(q)]
    sol = ("from bisect import bisect_right\nt = s.index.tolist()\nv = s.tolist()\nout = []\n"
           "for x in queries.tolist():\n    i = bisect_right(t, x) - 1\n    out.append(v[i] if i >= 0 else np.nan)\n"
           "result = np.array(out, dtype=float)")

    def big(seed, n_, q_):
        return (f"g = np.random.default_rng({seed})\nt = np.unique(g.integers(0, 10**9, size={n_}))\n"
                f"s = pd.Series(g.random(len(t)).round(4), index=t)\nqueries = g.integers(-10, 10**9 + 10, size={q_})")

    tests = [big(seeded(rng), 20000, 20000), big(seeded(rng), 5, 20000),
             "s = pd.Series([1.5], index=[10])\nqueries = np.array([9, 10, 11])"]
    return challenge("As-of lookup",
                     "`s` holds a sensor's readings indexed by **sorted, unique** integer timestamps. For every "
                     "query time in `queries`, return the reading at the **latest timestamp ≤ the query** (NaN if "
                     "there is none). Return a float NumPy array aligned with `queries`.",
                     f"s = pd.Series({lit(vals)}, index={lit(times)})\nqueries = np.array({lit(qs)})", sol, tests,
                     "1 ≤ n, q ≤ 2·10⁴, timestamps ≤ 10⁹", answer_desc=" (a NumPy array)", speed="O((n + q) log n)")


@template(M, "challenge")
def k_distinct_windows(rng):
    n = rng.randint(8, 11)
    pool = rng.sample(["cat", "dog", "bird", "fish", "frog", "deer"], rng.randint(3, 4))
    labels = [rng.choice(pool) for _ in range(n)]
    K = rng.randint(1, 3)
    sol = ("from collections import defaultdict\nvals = labels.tolist()\ncount = defaultdict(int)\n"
           "left = distinct = total = 0\nfor right, x in enumerate(vals):\n    count[x] += 1\n"
           "    if count[x] == 1:\n        distinct += 1\n    while distinct > K:\n        y = vals[left]\n"
           "        count[y] -= 1\n        if count[y] == 0:\n            distinct -= 1\n        left += 1\n"
           "    total += right - left + 1\nresult = total")

    def big(seed, n_, pool_, k_):
        return (f"g = np.random.default_rng({seed})\n"
                f"labels = pd.Series(g.choice([f'c{{i}}' for i in range({pool_})], size={n_}))\nK = {k_}")

    tests = [big(seeded(rng), 20000, 40, 5), big(seeded(rng), 20000, 3, 2), big(seeded(rng), 20000, 1000, 1000),
             "labels = pd.Series(['a'])\nK = 1"]
    return challenge("Windows with few labels",
                     "`labels` is a Series of class labels predicted for consecutive video frames. Count the "
                     "**contiguous windows** (any length ≥ 1) that contain **at most K distinct labels**.",
                     f"labels = pd.Series({lit(labels)})\nK = {K}", sol, tests, "1 ≤ K, n ≤ 2·10⁴",
                     speed="O(n) with two pointers")

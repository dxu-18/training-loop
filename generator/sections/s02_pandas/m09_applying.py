"""2.09 Applying functions (vectorized ops, map, apply, agg, transform, pipe, eval)."""

from ._common import P, challenge, frame, lit, seeded, template, threshold

M = "2.09"


def _thr(rng, f, col):
    return threshold(rng, f.cols[col], next(m for m in f.nums if m.var == col))


def _codes(levels):
    return {lvl: lvl[0].upper() + str(i) for i, lvl in enumerate(levels)}


# ================================================================= lite

@template(M, "lite")
def vectorized_ops(rng):
    f = frame(rng, (5, 8), nums=1)
    a = f.num[0]
    lo = _thr(rng, f, a)
    desc, sol = rng.choice([
        (f"`{a}` doubled", f"df[{a!r}] * 2"),
        (f"`log(1 + {a})` with a NumPy ufunc applied to the Series", f"np.log1p(df[{a!r}])"),
        (f"`{a}` with every value below {lo} raised to {lo} (`clip(lower=...)`)", f"df[{a!r}].clip(lower={lo})"),
        (f"`{a}` rounded to the nearest whole number (`round(0)`)", f"df[{a!r}].round(0)"),
    ])
    return P(f"{f.describe()}. Vectorized operations come first: compute {desc}, with no loops and no `apply`.",
             f.code(), sol, no_loops=True)


@template(M, "lite")
def map_values(rng):
    f = frame(rng, (5, 8), nums=1)
    codes = _codes(f.levels)
    desc, setup, sol = rng.choice([
        (f"each `{f.group}` mapped to a short code with the dict `codes` (`map`)", f"\ncodes = {codes!r}",
         f"df[{f.group!r}].map(codes)"),
        (f"each `{f.key}` mapped to its length (`map(len)`)", "", f"df[{f.key!r}].map(len)"),
        (f"each `{f.group}` mapped to upper case with `map(str.upper)`", "", f"df[{f.group!r}].map(str.upper)"),
        (f"`{f.group}` with only `{f.levels[0]!r}` replaced by `'X'`, other values unchanged (`replace`, not `map`)",
         "", f"df[{f.group!r}].replace({{{f.levels[0]!r}: 'X'}})"),
    ])
    return P(f"{f.describe()}. Compute {desc}.", f.code() + setup, sol)


# ================================================================= easy

@template(M, "easy")
def apply_functions(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    t = _thr(rng, f, a)
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Use `apply` with a lambda to label each `{a}` value `'high'` if it is at least {t}, "
                 f"else `'low'`.", f.code(), f"df[{a!r}].apply(lambda v: 'high' if v >= {t} else 'low')",
                 require=["apply|map"])
    return P(f"{f.describe()}. For each of the columns `{a}` and `{b}`, compute its **range** (max − min) with "
             f"`df[[...]].apply(lambda col: ...)`.", f.code(),
             f"df[[{a!r}, {b!r}]].apply(lambda col: col.max() - col.min())", require=["apply"])


@template(M, "easy")
def aggregate_multiple(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Compute the mean and max of both `{a}` and `{b}` in **one** `agg` call (a 2×2 "
                 f"result).", f.code(), f"df[[{a!r}, {b!r}]].agg(['mean', 'max'])", require=["agg|aggregate"])
    return P(f"{f.describe()}. With one `agg` call and a dict, compute the mean of `{a}` and the min and max of `{b}`.",
             f.code(), f"df.agg({{{a!r}: 'mean', {b!r}: ['min', 'max']}})", require=["agg|aggregate"])


@template(M, "easy")
def format_cells(rng):
    f = frame(rng, (3, 5), nums=2)
    a, b = f.num
    p = rng.randint(0, 2)
    return P(f"{f.describe()}. For a report, turn every cell of `df[[{a!r}, {b!r}]]` into a string with {p} decimals "
             f"using the element-wise `DataFrame.map` (formerly `applymap`) and an f-string.", f.code(),
             f"df[[{a!r}, {b!r}]].map(lambda v: f'{{v:.{p}f}}')", require=["map|applymap"])


# ================================================================= medium

@template(M, "medium")
def vectorized_label(rng):
    f = frame(rng, (5, 8), nums=1)
    sep = rng.choice([" (", " / ", ": "])
    end = ")" if sep == " (" else ""
    return P(f"{f.describe()}. Build a string column like `'{f.cols[f.key][0]}{sep}{f.cols[f.group][0]}{end}'` from "
             f"`{f.key}` and `{f.group}` **without** `apply`: string Series support `+`.", f.code(),
             f"df[{f.key!r}] + {sep!r} + df[{f.group!r}]" + (f" + {end!r}" if end else ""))


@template(M, "medium")
def multi_branch(rng):
    f = frame(rng, (6, 9), nums=1)
    a = f.num[0]
    lo, hi = sorted([_thr(rng, f, a), _thr(rng, f, a)])
    if lo == hi:
        hi = lo + 1
    names = rng.choice([("gold", "silver", "bronze"), ("A", "B", "C"), ("hot", "warm", "cold")])
    if rng.random() < 0.5:
        return P(f"{f.describe()}. Label `{a}` as `{names[0]!r}` if ≥ {hi}, `{names[1]!r}` if ≥ {lo}, else "
                 f"`{names[2]!r}` with `np.select`.", f.code(),
                 f"np.select([df[{a!r}] >= {hi}, df[{a!r}] >= {lo}], [{names[0]!r}, {names[1]!r}], default={names[2]!r})",
                 require=["select"])
    return P(f"{f.describe()}. Label `{a}` as `{names[0]!r}` if ≥ {hi}, `{names[1]!r}` if ≥ {lo}, else `{names[2]!r}`, "
             f"using `Series.case_when` on a Series of defaults (pandas 2.2+).", f.code(),
             f"pd.Series({names[2]!r}, index=df.index).case_when([(df[{a!r}] >= {hi}, {names[0]!r}), "
             f"(df[{a!r}] >= {lo}, {names[1]!r})])", require=["case_when"])


@template(M, "medium")
def transform_columns(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    desc, sol = rng.choice([
        ("subtract each column's mean (same shape as the input)", f"df[[{a!r}, {b!r}]].transform(lambda c: c - c.mean())"),
        ("divide each column by its max", f"df[[{a!r}, {b!r}]].transform(lambda c: c / c.max())"),
    ])
    return P(f"{f.describe()}. With `transform` on `df[[{a!r}, {b!r}]]`, {desc}.", f.code(), sol, require=["transform"])


@template(M, "medium")
def pipe_function(rng):
    f = frame(rng, (5, 8), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. `add_ratio(d, num, den)` returns `d` with a new `ratio = d[num] / d[den]` column. Call it "
             f"with `df.pipe(...)`, passing `{a!r}` and `{b!r}`, so it would slot into a method chain.",
             f.code() + "\n\ndef add_ratio(d, num, den):\n    return d.assign(ratio=d[num] / d[den])",
             f"df.pipe(add_ratio, {a!r}, {b!r})", require=["pipe"])


# ================================================================= hard

@template(M, "hard")
def eval_chain(rng):
    f = frame(rng, (7, 10), nums=2)
    a, b = f.num
    k = rng.randint(2, 3)
    return P(f"{f.describe()}. In one chain: add `ratio = {a} / {b}` with `df.eval('ratio = ...')`, keep the rows "
             f"with `ratio` above its median (`query` or a mask), and return the {k} rows with the largest `ratio`, "
             f"largest first.", f.code(),
             f"d = df.eval('ratio = {a} / {b}')\nresult = d[d['ratio'] > d['ratio'].median()].nlargest({k}, 'ratio')",
             require=["eval"])


@template(M, "hard")
def cleaning_function(rng):
    f = frame(rng, (7, 10), nums=2, nan=rng.randint(1, 2))
    a, b = f.num
    return P(f"{f.describe()}. Write `clean(d)` that (1) drops rows with a missing `{a}`, (2) renames `{b}` to "
             f"`target`, and (3) adds `{a}_rank`, the dense rank of `{a}` (highest = 1, as int); then apply it with "
             f"`df.pipe(clean)`.", f.code(),
             "def clean(d):\n    d = d.dropna(subset=[" + repr(a) + "]).rename(columns={" + repr(b) + ": 'target'})\n"
             f"    return d.assign({a}_rank=d[{a!r}].rank(method='dense', ascending=False).astype(int))\n\nresult = df.pipe(clean)",
             require=["pipe"])


@template(M, "hard")
def map_with_fallback(rng):
    f = frame(rng, (6, 9), nums=1)
    codes = {lvl: i for i, lvl in enumerate(f.levels[:-1])}
    return P(f"{f.describe()}. `ids` maps some `{f.group}` values to integer ids, but not all. Map `{f.group}` to its id, "
             f"use **-1** for values missing from `ids`, and return an int Series (`map` gives NaN for unmatched keys).",
             f.code() + f"\nids = {codes!r}", f"df[{f.group!r}].map(ids).fillna(-1).astype(int)", require=["map"])


# ================================================================= challenge

@template(M, "challenge")
def alias_resolution(rng):
    n = rng.randint(6, 8)
    parent = [rng.randrange(n) if rng.random() < 0.7 else i for i in range(n)]
    sol = ("par = parent.to_dict()\nroot = {}\nfor start in par:\n    path, on_path, x = [], set(), start\n"
           "    while x not in root and par[x] != x and x not in on_path:\n        path.append(x)\n        on_path.add(x)\n"
           "        x = par[x]\n"
           "    if x in root:\n        r = root[x]\n    elif par[x] == x:\n        r = x\n    else:  # walked into a cycle\n        r = -1\n"
           "    for y in path:\n        root[y] = r\n    root.setdefault(x, r if par[x] != x else x)\n"
           "result = pd.Series([root[i] for i in parent.index], index=parent.index)")

    def big(seed, n_, p_self):
        return (f"g = np.random.default_rng({seed})\nn = {n_}\npar = g.integers(0, n, n)\n"
                f"keep = g.random(n) < {p_self}\npar[keep] = np.arange(n)[keep]\nparent = pd.Series(par)")

    tests = [big(seeded(rng), 20000, 0.2), big(seeded(rng), 20000, 0.0),
             "parent = pd.Series(list(range(1, 20000)) + [19999])", "parent = pd.Series([1, 0, 2])"]
    return challenge("Resolve the aliases",
                     "`parent` maps every entity id (the index) to the id it was **merged into**; an id that maps to "
                     "itself is a canonical root. Follow the chain from every id to its root. If the chain runs into a "
                     "**cycle** (no root is ever reached), the answer for that id is −1. Return a Series of roots with "
                     "the same index as `parent`.", f"parent = pd.Series({lit(parent)})", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     answer_desc=" (a Series)", speed="O(n): memoize every chain you walk")


@template(M, "challenge")
def tiered_cost(rng):
    k = rng.randint(2, 3)
    caps = sorted(rng.sample(range(10, 100, 10), k))
    rates = [rng.randint(1, 9) for _ in range(k + 1)]  # integer price units keep the bill exact
    usage = [rng.randint(0, 130) for _ in range(rng.randint(4, 6))]
    sol = ("import bisect\ncaps = tiers['up_to'].tolist()\nrates = tiers['rate'].tolist()\n"
           "base = [0]\nprev = 0\nfor cap, rate in zip(caps, rates):\n    base.append(base[-1] + (cap - prev) * rate)\n    prev = cap\n"
           "total = 0\nfor u in usage.tolist():\n    i = bisect.bisect_left(caps, u)\n    lo = caps[i - 1] if i else 0\n"
           "    total += base[i] + (u - lo) * (rates[i] if i < len(caps) else last_rate)\nresult = total")

    def big(seed, n_, k_):
        return (f"g = np.random.default_rng({seed})\ncaps = np.sort(g.choice(np.arange(1, 10**6), {k_}, replace=False))\n"
                f"tiers = pd.DataFrame({{'up_to': caps, 'rate': g.integers(1, 50, {k_})}})\nlast_rate = 3\n"
                f"usage = pd.Series(g.integers(0, 2 * 10**6, {n_}))")

    tests = [big(seeded(rng), 20000, 60), big(seeded(rng), 20000, 1), "tiers = pd.DataFrame({'up_to': [10], 'rate': [1]})\nlast_rate = 2\nusage = pd.Series([10, 11, 0])"]
    return challenge("Tiered pricing",
                     "An API bills tokens in **tiers**: `tiers` has rows `(up_to, rate)` sorted by `up_to`. The first "
                     "`up_to[0]` tokens of a request cost `rate[0]` each, the tokens after that up to `up_to[1]` cost "
                     "`rate[1]` each, and so on; tokens beyond the last `up_to` cost `last_rate` each (rates are integer "
                     "price units). `usage` holds the token count of every request. Return the **total bill** over all "
                     "requests as an int.",
                     f"tiers = pd.DataFrame({{'up_to': {caps!r}, 'rate': {rates[:-1]!r}}})\nlast_rate = {rates[-1]}\n"
                     f"usage = pd.Series({usage!r})", sol, tests, "≤ 2·10⁴ requests, ≤ 60 tiers",
                     speed="O(n log tiers) with precomputed tier totals")

"""2.17 MultiIndex."""

from ._common import P, challenge, frame, seeded, template

M = "2.17"


def _mi(rng, n=(6, 9), nums=2):
    f = frame(rng, n, nums=nums)
    setup = f.code() + f"\nmi = df.set_index([{f.group!r}, {f.key!r}]).sort_index()"
    return f, setup


# ================================================================= lite

@template(M, "lite")
def build_multiindex(rng):
    f = frame(rng, (5, 8), nums=1)
    g, k = f.group, f.key
    desc, sol = rng.choice([
        (f"index `df` by (`{g}`, `{k}`) and sort the index", f"df.set_index([{g!r}, {k!r}]).sort_index()"),
        (f"the **names** of the levels after `df.set_index([{g!r}, {k!r}])`", f"list(df.set_index([{g!r}, {k!r}]).index.names)"),
        (f"the `{g}` value of every row after `df.set_index([{g!r}, {k!r}])` (`index.get_level_values`)",
         f"df.set_index([{g!r}, {k!r}]).index.get_level_values({g!r})"),
    ])
    return P(f"{f.describe()}. Get {desc}.", f.code(), sol, require=["set_index"])


@template(M, "lite")
def select_outer_level(rng):
    f, setup = _mi(rng)
    lvl = f.cols[f.group][0]
    k = next(n for n, gr in zip(f.cols[f.key], f.cols[f.group]) if gr == lvl)
    a = f.num[0]
    desc, sol = rng.choice([
        (f"every row in the `{lvl!r}` group (`mi.loc[{lvl!r}]`)", f"mi.loc[{lvl!r}]"),
        (f"the row for (`{lvl!r}`, `{k!r}`) (a tuple key)", f"mi.loc[({lvl!r}, {k!r})]"),
        (f"the `{a}` value for (`{lvl!r}`, `{k!r}`)", f"mi.loc[({lvl!r}, {k!r}), {a!r}]"),
    ])
    return P(f"`mi` is `df` indexed by (`{f.group}`, `{f.key}`), sorted. Select {desc}.", setup, sol, require=["loc"])


# ================================================================= easy

@template(M, "easy")
def cross_section(rng):
    f, setup = _mi(rng)
    k = f.cols[f.key][0]
    if rng.random() < 0.5:
        return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). Select the row(s) for `{f.key}` `{k!r}` from the **inner** "
                 f"level with a cross-section (`mi.xs(..., level={f.key!r})`).", setup, f"mi.xs({k!r}, level={f.key!r})",
                 require=["xs"])
    return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). Get the distinct `{f.group}` values present in the index "
             f"(`mi.index.levels[0]`, or `get_level_values(0).unique()`).", setup,
             "mi.index.get_level_values(0).unique()", require=["levels|get_level_values"])


@template(M, "easy")
def construct_multiindex(rng):
    a = rng.sample(["train", "val", "test", "dev"], rng.randint(2, 3))
    b = sorted(rng.sample(range(1, 6), rng.randint(2, 3)))
    names = rng.choice([["split", "fold"], ["model", "seed"], ["region", "rack"]])
    desc, sol = rng.choice([
        (f"every combination of `{a}` and `{b}` (`pd.MultiIndex.from_product`, names `{names}`)",
         f"pd.MultiIndex.from_product([{a!r}, {b!r}], names={names!r})"),
        (f"the pairs `{list(zip(a, b))}` (`pd.MultiIndex.from_tuples`, names `{names}`)",
         f"pd.MultiIndex.from_tuples({list(zip(a, b))!r}, names={names!r})"),
        (f"the parallel arrays `{a[:len(b)]}` and `{b[:len(a)]}` (`pd.MultiIndex.from_arrays`, names `{names}`)",
         f"pd.MultiIndex.from_arrays([{a[:len(b)]!r}, {b[:len(a)]!r}], names={names!r})"),
    ])
    return P(f"Build a MultiIndex from {desc}.", "", sol, require=["MultiIndex"], post="(list(_.names), _)")


@template(M, "easy")
def reshape_levels(rng):
    f, setup = _mi(rng)
    desc, sol = rng.choice([
        ("swap the two index levels and re-sort (`swaplevel().sort_index()`)", "mi.swaplevel().sort_index()"),
        (f"drop the `{f.key}` level from the index (`droplevel`)", f"mi.droplevel({f.key!r})"),
        (f"move just the `{f.key}` level back into a column (`reset_index(level=...)`)", f"mi.reset_index(level={f.key!r})"),
    ])
    return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). {desc[0].upper() + desc[1:]}.", setup, sol)


# ================================================================= medium

@template(M, "medium")
def index_slice(rng):
    f, setup = _mi(rng, (7, 10))
    picks = rng.sample(f.cols[f.key], 2)
    return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`), sorted. Select the rows for `{f.key}` in `{picks}` across "
             f"**all** `{f.group}` values with `mi.loc[pd.IndexSlice[:, [...]], :]`.", setup,
             f"mi.loc[pd.IndexSlice[:, {picks!r}], :]", require=["IndexSlice|xs|get_level_values"])


@template(M, "medium")
def level_operations(rng):
    f, setup = _mi(rng, (7, 10))
    a = f.num[0]
    if rng.random() < 0.5:
        return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). Compute the mean `{a}` per `{f.group}` by grouping on the "
                 f"index **level** (`groupby(level=...)`).", setup, f"mi.groupby(level={f.group!r})[{a!r}].mean()",
                 require=["level"])
    return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). Turn the `{f.group}` level into **columns** of a `{a}` table "
             f"(`mi[{a!r}].unstack(level={f.group!r})`).", setup, f"mi[{a!r}].unstack(level={f.group!r})",
             require=["unstack"])


@template(M, "medium")
def flatten_columns(rng):
    f = frame(rng, (7, 10), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. `agg = df.groupby({f.group!r}).agg({{{a!r}: ['mean', 'max'], {b!r}: ['min']}})` has "
             f"**MultiIndex columns** like `({a!r}, 'mean')`. Flatten them to `{a}_mean`, `{a}_max`, `{b}_min` with "
             f"`'_'.join` and return `agg`.", f.code() + f"\nagg = df.groupby({f.group!r}).agg({{{a!r}: ['mean', 'max'], {b!r}: ['min']}})",
             "agg.columns = ['_'.join(c) for c in agg.columns]\nresult = agg", post="(list(_.columns), _)")


@template(M, "medium")
def rename_levels(rng):
    f, setup = _mi(rng)
    new = rng.choice(["squad", "segment", "bucket"])
    return P(f"`mi` is indexed by (`{f.group}`, `{f.key}`). Rename the `{f.group}` level to `{new}` with "
             f"`rename_axis(index={{...}})`.", setup, f"mi.rename_axis(index={{{f.group!r}: {new!r}}})",
             require=["rename_axis"], post="(list(_.index.names), _)")


# ================================================================= hard

@template(M, "hard")
def agg_flat_report(rng):
    f = frame(rng, (8, 11), nums=2)
    a, b = f.num
    return P(f"{f.describe()}. Build a flat report: per `{f.group}`, the mean and max of `{a}` and the sum of `{b}` "
             f"(`agg` with a dict of lists), flatten the column MultiIndex to names like `{a}_mean`, make `{f.group}` a "
             f"column (`reset_index`), and sort by `{a}_mean` descending.", f.code(),
             f"agg = df.groupby({f.group!r}).agg({{{a!r}: ['mean', 'max'], {b!r}: ['sum']}})\n"
             "agg.columns = ['_'.join(c) for c in agg.columns]\n"
             f"result = agg.reset_index().sort_values('{a}_mean', ascending=False)", post="(list(_.columns), _)")


@template(M, "hard")
def two_level_table(rng):
    f = frame(rng, (9, 12), nums=1)
    f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
    a = f.num[0]
    return P(f"{f.describe()} and `flag`. Sum `{a}` per (`{f.group}`, `flag`) with a two-key groupby, then `unstack` "
             f"the `flag` level into columns, filling missing combinations with 0 (`fill_value=0`).", f.code(),
             f"df.groupby([{f.group!r}, 'flag'])[{a!r}].sum().unstack('flag', fill_value=0)", require=["unstack"])


@template(M, "hard")
def complete_grid(rng):
    days = [f"2024-05-0{d}" for d in range(1, rng.randint(3, 5) + 1)]
    stores = rng.sample(["north", "south", "east", "west"], rng.randint(2, 3))
    rows = [(d, s, rng.randint(1, 50)) for d in days for s in stores if rng.random() < 0.6]
    if not rows:
        rows = [(days[0], stores[0], 7)]
    return P("`sales` only has rows for (day, store) pairs that sold something. Build the **complete grid**: index it by "
             "(day, store), `reindex` against `pd.MultiIndex.from_product([days, stores], names=['day', 'store'])`, and "
             "fill missing sales with 0 (as ints).",
             f"sales = pd.DataFrame({rows!r}, columns=['day', 'store', 'amount'])\ndays = {days!r}\nstores = {stores!r}",
             "full = pd.MultiIndex.from_product([days, stores], names=['day', 'store'])\n"
             "result = sales.set_index(['day', 'store'])['amount'].reindex(full, fill_value=0).astype(int)",
             require=["MultiIndex", "reindex"])


# ================================================================= challenge

@template(M, "challenge")
def grid_components(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 5)
    cells = [(i, j) for i in range(r) for j in range(c) if rng.random() < 0.45] or [(0, 0)]
    sol = ("cells = set(active.index.tolist())\nseen = set()\nsizes = []\nfor start in cells:\n    if start in seen:\n"
           "        continue\n    stack, size = [start], 0\n    seen.add(start)\n    while stack:\n        i, j = stack.pop()\n"
           "        size += 1\n        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):\n"
           "            if nb in cells and nb not in seen:\n                seen.add(nb)\n                stack.append(nb)\n"
           "    sizes.append(size)\nresult = (len(sizes), max(sizes))")

    def big(seed, n_, side):
        return (f"g = np.random.default_rng({seed})\nidx = pd.MultiIndex.from_arrays([g.integers(0, {side}, {n_}), "
                f"g.integers(0, {side}, {n_})], names=['row', 'col']).unique()\nactive = pd.Series(1, index=idx)")

    tests = [big(seeded(rng), 20000, 250), big(seeded(rng), 20000, 150), big(seeded(rng), 20000, 1000),
             "active = pd.Series(1, index=pd.MultiIndex.from_tuples([(5, 5)], names=['row', 'col']))"]
    return challenge("Count the regions",
                     "`active` is a Series indexed by a **(row, col) MultiIndex** listing the active pixels of a "
                     "segmentation mask (sparse: only active cells appear). Two active pixels are connected if they share "
                     "an edge (up/down/left/right). Return the tuple `(number_of_regions, size_of_largest_region)`.",
                     f"active = pd.Series(1, index=pd.MultiIndex.from_tuples({cells!r}, names=['row', 'col']))", sol, tests,
                     "≤ 2·10⁴ active pixels", speed="O(n) with BFS/DFS over a set of cells")


@template(M, "challenge")
def subtree_sums(rng):
    n = rng.randint(5, 8)
    parent = [-1] + [rng.randrange(i) for i in range(1, n)]
    cost = [rng.randint(1, 20) for _ in range(n)]
    sol = ("par = tree['parent'].tolist()\ntotal = tree['cost'].tolist()\nchildren = [[] for _ in par]\nroots = []\n"
           "for v, p in enumerate(par):\n    (roots if p == -1 else children[p]).append(v)\norder = []\nstack = list(roots)\n"
           "while stack:\n    v = stack.pop()\n    order.append(v)\n    stack.extend(children[v])\nfor v in reversed(order):\n"
           "    if par[v] != -1:\n        total[par[v]] += total[v]\nresult = pd.Series(total, index=tree.index)")

    def big(seed, n_, depth_bias):
        return (f"g = np.random.default_rng({seed})\nn = {n_}\n"
                f"par = np.r_[-1, np.maximum(0, np.arange(1, n) - 1 - g.integers(0, {depth_bias}, n - 1))]\n"
                f"tree = pd.DataFrame({{'parent': par, 'cost': g.integers(1, 1000, n)}})")

    tests = [big(seeded(rng), 20000, 20000), big(seeded(rng), 20000, 1), "tree = pd.DataFrame({'parent': [-1], 'cost': [9]})"]
    return challenge("Roll up the org chart",
                     "`tree` describes a hierarchy of cost centres: row i is node i, `parent` is its parent's row (−1 for "
                     "the root; parents always come before their children) and `cost` its own cost. Return a Series of "
                     "every node's **subtree total** (its own cost plus all descendants'), indexed like `tree`.",
                     f"tree = pd.DataFrame({{'parent': {parent!r}, 'cost': {cost!r}}})", sol, tests, "1 ≤ n ≤ 2·10⁴",
                     answer_desc=" (a Series)", speed="O(n): process nodes from the leaves up (a deep chain breaks naive recursion)")

"""2.14 Reshaping (pivot, melt, pivot_table, crosstab, stack/unstack, explode, dummies)."""

from ._common import P, challenge, frame, seeded, template

M = "2.14"

SUBJECTS = (("subject", ["math", "art", "music", "cs"]), ("benchmark", ["mmlu", "gsm8k", "humaneval", "arc"]),
            ("metric", ["precision", "recall", "f1", "auc"]), ("quarter", ["q1", "q2", "q3", "q4"]))


def _long(rng, n_ids=(2, 4), n_cols=(2, 4)):
    who = rng.sample(["ana", "ben", "cal", "dee", "eve"], rng.randint(*n_ids))
    col, levels = rng.choice(SUBJECTS)
    levels = levels[:rng.randint(*n_cols)]
    rows = [(w, lv, rng.randint(40, 100)) for w in who for lv in levels]
    rng.shuffle(rows)
    return who, col, levels, rows


def _long_code(col, rows, var="long"):
    return f"{var} = pd.DataFrame({rows!r}, columns=['id', {col!r}, 'value'])"


def _wide_code(who, levels, rng, var="wide"):
    data = {lv: [rng.randint(40, 100) for _ in who] for lv in levels}
    return f"{var} = pd.DataFrame({{'id': {who!r}, " + ", ".join(f"{k!r}: {v!r}" for k, v in data.items()) + "})"


# ================================================================= lite

@template(M, "lite")
def pivot_basic(rng):
    who, col, levels, rows = _long(rng)
    return P(f"`long` has one row per (id, {col}) pair. Turn it **wide**: one row per `id`, one column per `{col}`, "
             f"cells from `value` (`pivot`).", _long_code(col, rows), f"long.pivot(index='id', columns={col!r}, values='value')",
             require=["pivot|pivot_table|unstack"])


@template(M, "lite")
def melt_basic(rng):
    who, col, levels, _ = _long(rng)
    return P(f"`wide` has one row per `id` and one column per {col} ({', '.join(levels)}). Turn it **long** with `melt`: "
             f"columns `id`, `{col}`, `value`.", _wide_code(who, levels, rng),
             f"wide.melt(id_vars='id', var_name={col!r}, value_name='value')", require=["melt|stack"])


# ================================================================= easy

@template(M, "easy")
def pivot_table_mean(rng):
    f = frame(rng, (8, 12), nums=1)
    f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
    g, a = f.group, f.num[0]
    if rng.random() < 0.5:
        return P(f"{f.describe()} and `flag`. Make a `pivot_table` of the mean `{a}` with one row per `{g}`.", f.code(),
                 f"df.pivot_table(index={g!r}, values={a!r}, aggfunc='mean')", require=["pivot_table"])
    return P(f"{f.describe()} and `flag`. Make a `{g}` × `flag` table of the mean `{a}` with `pivot_table`, filling "
             f"empty cells with 0 (`fill_value=0`).", f.code(),
             f"df.pivot_table(index={g!r}, columns='flag', values={a!r}, aggfunc='mean', fill_value=0)",
             require=["pivot_table"])


@template(M, "easy")
def crosstab_counts(rng):
    f = frame(rng, (8, 12), nums=1)
    f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
    g = f.group
    if rng.random() < 0.5:
        return P(f"{f.describe()} and `flag`. Count the rows for every `{g}` × `flag` combination with `pd.crosstab`.",
                 f.code(), f"pd.crosstab(df[{g!r}], df['flag'])", require=["crosstab"])
    return P(f"{f.describe()} and `flag`. With `pd.crosstab`, give the **share** of each `flag` value **within each "
             f"`{g}`** (rows sum to 1, `normalize='index'`).", f.code(), f"pd.crosstab(df[{g!r}], df['flag'], normalize='index')",
             require=["crosstab"])


@template(M, "easy")
def explode_lists(rng):
    tags = ["ml", "nlp", "cv", "speech", "rl"]
    rows = [(i + 1, rng.sample(tags, rng.randint(1, 3))) for i in range(rng.randint(3, 5))]
    setup = f"papers = pd.DataFrame({{'id': {[r[0] for r in rows]!r}, 'tags': {[r[1] for r in rows]!r}}})"
    if rng.random() < 0.5:
        return P("`papers` has a `tags` column holding a **list** per paper. Give each tag its own row (`explode`, "
                 "with `ignore_index=True`).", setup, "papers.explode('tags', ignore_index=True)", require=["explode"])
    return P("`papers` has a `tags` column holding a list per paper. Count how many papers carry each tag "
             "(`explode`, then `value_counts`).", setup, "papers.explode('tags')['tags'].value_counts()",
             require=["explode"])


# ================================================================= medium

@template(M, "medium")
def stack_unstack(rng):
    who, col, levels, rows = _long(rng)
    setup = _long_code(col, rows) + f"\nwide = long.pivot(index='id', columns={col!r}, values='value')"
    desc, sol, req = rng.choice([
        ("`wide` stacked into a Series with an (id, " + col + ") MultiIndex (`stack`)", "wide.stack()", "stack"),
        (f"`wide` stacked and then unstacked again with the **{col}** level as rows (`unstack(level=0)`)",
         "wide.stack().unstack(level=0)", "unstack"),
        ("`wide` transposed so ids become columns (`.T`)", "wide.T", "T|transpose"),
    ])
    return P(f"`wide` (built from `long`) has one row per id and one column per {col}. Get {desc}.", setup, sol,
             require=[req])


@template(M, "medium")
def pivot_margins(rng):
    f = frame(rng, (8, 12), nums=1)
    f.cols["flag"] = [rng.choice(["yes", "no"]) for _ in range(f.n)]
    g, a = f.group, f.num[0]
    return P(f"{f.describe()} and `flag`. Make a `{g}` × `flag` table of the **sum** of `{a}` with `pivot_table`, "
             f"`fill_value=0`, and row/column **totals** (`margins=True`).", f.code(),
             f"df.pivot_table(index={g!r}, columns='flag', values={a!r}, aggfunc='sum', fill_value=0, margins=True)",
             require=["pivot_table"])


@template(M, "medium")
def wide_to_long_years(rng):
    n = rng.randint(2, 4)
    years = sorted(rng.sample(range(2019, 2026), rng.randint(2, 3)))
    data = {f"score{y}": [rng.randint(1, 99) for _ in range(n)] for y in years}
    setup = f"df = pd.DataFrame({{'id': {list(range(1, n + 1))!r}, " + ", ".join(f"{k!r}: {v!r}" for k, v in data.items()) + "})"
    return P(f"`df` has one `score<year>` column per year ({', '.join(data)}). Reshape it to long format with "
             f"`pd.wide_to_long(df, stubnames='score', i='id', j='year')`: a frame indexed by (id, year).", setup,
             "pd.wide_to_long(df, stubnames='score', i='id', j='year')", require=["wide_to_long|melt"])


@template(M, "medium")
def quick_wide(rng):
    f = frame(rng, (5, 8), nums=1)
    g, a = f.group, f.num[0]
    return P(f"{f.describe()}. Make a quick wide table with one row per `{g}` and one column per `{f.key}`: set a "
             f"(`{g}`, `{f.key}`) index, take `{a}`, and `unstack()`.", f.code(),
             f"df.set_index([{g!r}, {f.key!r}])[{a!r}].unstack()", require=["unstack|pivot|pivot_table"])


# ================================================================= hard

@template(M, "hard")
def tidy_pivot(rng):
    who, col, levels, rows = _long(rng)
    return P(f"`long` has one row per (id, {col}) pair. Build a **flat, tidy** wide table: `pivot_table` with the mean "
             f"`value`, remove the columns' axis name (`rename_axis(columns=None)`), and `reset_index()` so `id` is an "
             f"ordinary first column.", _long_code(col, rows),
             f"long.pivot_table(index='id', columns={col!r}, values='value', aggfunc='mean').rename_axis(columns=None).reset_index()",
             post="{'columns': list(_.columns), 'frame': _}", require=["pivot_table|pivot"])


@template(M, "hard")
def quarters_long(rng):
    n = rng.randint(2, 4)
    data = {f"q{i}": [rng.randint(10, 99) for _ in range(n)] for i in range(1, 5)}
    setup = f"sales = pd.DataFrame({{'store': {[f's{i}' for i in range(1, n + 1)]!r}, " + \
        ", ".join(f"{k!r}: {v!r}" for k, v in data.items()) + "})"
    return P("`sales` has one row per store and columns `q1`…`q4`. Reshape it to long format with columns `store`, "
             "`quarter` (the **integer** 1–4) and `amount`, sorted by `store` then `quarter`, with a fresh 0..n−1 index.",
             setup, "d = sales.melt(id_vars='store', var_name='quarter', value_name='amount')\n"
                    "d['quarter'] = d['quarter'].str[1:].astype(int)\nresult = d.sort_values(['store', 'quarter']).reset_index(drop=True)",
             require=["melt|stack|wide_to_long"])


@template(M, "hard")
def dummies_roundtrip(rng):
    f = frame(rng, (5, 8), nums=1)
    g = f.group
    return P(f"{f.describe()}. One-hot encode `{g}` with `pd.get_dummies(df[[{g!r}]])` (columns like `{g}_<value>`), then "
             f"recover the original labels with `pd.from_dummies(..., sep='_')`, and return the recovered frame.",
             f.code(), f"pd.from_dummies(pd.get_dummies(df[[{g!r}]]), sep='_')", require=["from_dummies"])


# ================================================================= challenge

@template(M, "challenge")
def grid_max_path(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 4)
    cells = [(i, j, rng.randint(-5, 9)) for i in range(r) for j in range(c)]
    rng.shuffle(cells)
    sol = ("grid = cells.pivot(index='row', columns='col', values='value').sort_index().sort_index(axis=1).to_numpy().tolist()\n"
           "R, C = len(grid), len(grid[0])\nbest = [[0] * C for _ in range(R)]\nfor i in range(R):\n    for j in range(C):\n"
           "        prev = []\n        if i:\n            prev.append(best[i - 1][j])\n        if j:\n            prev.append(best[i][j - 1])\n"
           "        best[i][j] = grid[i][j] + (max(prev) if prev else 0)\nresult = int(best[-1][-1])")

    def big(seed, r_, c_):
        return (f"g = np.random.default_rng({seed})\nii, jj = np.meshgrid(np.arange({r_}), np.arange({c_}), indexing='ij')\n"
                f"cells = pd.DataFrame({{'row': ii.ravel(), 'col': jj.ravel(), 'value': g.integers(-100, 101, {r_ * c_})}})"
                f".sample(frac=1, random_state={seed})")

    tests = [big(seeded(rng), 300, 300), big(seeded(rng), 1, 20000), "cells = pd.DataFrame({'row': [0], 'col': [0], 'value': [-7]})"]
    return challenge("Best path through the grid",
                     "`cells` is a **long** table `(row, col, value)` listing every cell of a grid exactly once, shuffled. "
                     "Pivot it into the grid, then find the path from the top-left to the bottom-right cell that moves "
                     "only **right or down** and has the **largest total value**. Return that total.",
                     f"cells = pd.DataFrame({cells!r}, columns=['row', 'col', 'value'])", sol, tests, "≤ 9·10⁴ cells",
                     speed="O(rows · columns) dynamic programming")


@template(M, "challenge")
def cooccurring_tags(rng):
    tags = ["ml", "nlp", "cv", "rl", "speech"]
    rows = [rng.sample(tags, rng.randint(1, 3)) for _ in range(rng.randint(4, 6))]
    rows[0] = ["ml", "nlp"]
    sol = ("from collections import Counter\nfrom itertools import combinations\ncount = Counter()\n"
           "for t in docs['tags']:\n    for pair in combinations(sorted(set(t)), 2):\n        count[pair] += 1\n"
           "best = max(count.values())\nresult = (min(p for p, c in count.items() if c == best), best)")

    def big(seed, n_, k):
        return (f"g = np.random.default_rng({seed})\nvocab = [f't{{i:03d}}' for i in range({k})]\n"
                f"docs = pd.DataFrame({{'tags': [list(g.choice(vocab, g.integers(2, 6), replace=False)) for _ in range({n_})]}})")

    tests = [big(seeded(rng), 20000, 40), big(seeded(rng), 20000, 500), "docs = pd.DataFrame({'tags': [['b', 'a'], ['a', 'b', 'a']]})"]
    return challenge("Most common tag pair",
                     "`docs` has a `tags` column with a **list** of tags per document (a tag may repeat inside one list; "
                     "count it once). Two tags **co-occur** when they appear in the same document. Return the tuple "
                     "`((tag_a, tag_b), count)` for the pair that co-occurs in the most documents, with `tag_a < tag_b`; "
                     "break ties by the lexicographically smallest pair. At least one document has two distinct tags.",
                     f"docs = pd.DataFrame({{'tags': {rows!r}}})", sol, tests, "≤ 2·10⁴ documents, ≤ 5 tags each",
                     speed="O(documents · tags²) with a Counter")

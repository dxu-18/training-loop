"""1.13 Sorting, searching, sets & counting."""

from ... import themes as T
from ._common import (P, about, arr, challenge, feature_matrix, lit, mat, rand_ints, seeded, series, table,
                      template)

M = "1.13"


# ================================================================= lite

@template(M, "lite")
def arg_extreme(rng):
    th, m, _, vals = series(rng, distinct=True)
    v = m.var
    hi = rng.random() < 0.5
    fn = "argmax" if hi else "argmin"
    return P(f"{about(th, m, len(vals), v)}. Use `np.{fn}` to find the **index** of the "
             f"{'highest' if hi else 'lowest'} value.", f"{v} = {arr(vals)}", f"np.{fn}({v})", require=[fn])


@template(M, "lite")
def sort_values(rng):
    th, m, _, vals = series(rng)
    v = m.var
    desc = rng.random() < 0.5
    return P(f"{about(th, m, len(vals), v)}. Return the values sorted in "
             f"{'**descending** (largest first)' if desc else '**ascending**'} order using `np.sort`.",
             f"{v} = {arr(vals)}", f"np.sort({v})[::-1]" if desc else f"np.sort({v})", require=["sort"])


@template(M, "lite")
def membership(rng):
    th = T.theme(rng)
    pool = list(th.items)
    a = rng.sample(pool, rng.randint(4, 6))
    b = rng.sample(pool, rng.randint(3, 6))
    if rng.random() < 0.5:
        return P(f"`deployed` lists {th.plural} running in production and `approved` lists {th.plural} that "
                 f"passed review. Return a boolean array saying, for each entry of `deployed`, whether it is "
                 f"in `approved` (`np.isin`).", f"deployed = {arr(a)}\napproved = {arr(b)}",
                 "np.isin(deployed, approved)", require=["isin"])
    return P(f"`deployed` lists {th.plural} running in production and `approved` lists {th.plural} that passed "
             f"review. Return the entries of `deployed` that are **not** approved, in their original order "
             f"(`np.isin` with `invert=True` as a mask).", f"deployed = {arr(a)}\napproved = {arr(b)}",
             "deployed[np.isin(deployed, approved, invert=True)]", require=["isin"])


# ================================================================= easy

@template(M, "easy")
def name_of_extreme(rng):
    th, m, names, vals = series(rng, distinct=True)
    v, nv = m.var, th.var
    hi = rng.random() < 0.5
    fn = "argmax" if hi else "argmin"
    return P(f"`{nv}` and `{v}` are parallel arrays: `{v}[i]` is the {m.label} of `{nv}[i]`. "
             f"Return the **name** of the {th.noun} with the {'highest' if hi else 'lowest'} "
             f"{m.name}, using `{fn}` to index into `{nv}`.",
             f"{nv} = {arr(names)}\n{v} = {arr(vals)}", f"{nv}[np.{fn}({v})]", require=[fn])


@template(M, "easy")
def top_k(rng):
    th, m, names, vals = series(rng, 6, 10, distinct=True)
    v, nv = m.var, th.var
    k = rng.randint(2, 4)
    big = rng.random() < 0.6
    order = f"np.argsort({v})[::-1][:{k}]" if big else f"np.argsort({v})[:{k}]"
    which = f"{'highest' if big else 'lowest'} {m.name}, {'highest' if big else 'lowest'} first"
    if rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. Using `np.argsort`, return the **indices** of the {k} "
                 f"{th.plural} with the {which}.", f"{v} = {arr(vals)}", order, require=["argsort"])
    return P(f"`{nv}` and `{v}` are parallel arrays ({m.label}). Using `np.argsort`, return the "
             f"**names** of the {k} {th.plural} with the {which}.",
             f"{nv} = {arr(names)}\n{v} = {arr(vals)}", f"{nv}[{order}]", require=["argsort"])


@template(M, "easy")
def unique_labels(rng):
    what, classes = T.labels(rng, min_classes=3)
    n = rng.randint(10, 16)
    used = rng.sample(classes, rng.randint(2, min(len(classes), 5)))
    y = [rng.choice(used) for _ in range(n)]
    y[:len(used)] = used
    rng.shuffle(y)
    if rng.random() < 0.5:
        return P(f"`y` holds the {what} label of {n} samples. Use `np.unique` to get the sorted "
                 f"array of **distinct** labels.", f"y = {arr(y)}", "np.unique(y)", require=["unique"])
    return P(f"`y` holds the {what} label of {n} samples. Use `np.unique` to count **how many "
             f"distinct** labels appear.", f"y = {arr(y)}", "len(np.unique(y))", require=["unique"])


@template(M, "easy")
def set_operations(rng):
    th = T.theme(rng)
    pool = list(th.items)
    a = rng.sample(pool, rng.randint(4, 7))
    b = rng.sample(pool, rng.randint(4, 7))
    desc, sol, req = rng.choice([
        ("evaluated on **both** benchmarks (`np.intersect1d`)", "np.intersect1d(bench_a, bench_b)", "intersect1d"),
        ("evaluated on **at least one** benchmark (`np.union1d`)", "np.union1d(bench_a, bench_b)", "union1d"),
        ("evaluated on benchmark A but **not** B (`np.setdiff1d`)", "np.setdiff1d(bench_a, bench_b)", "setdiff1d"),
        ("evaluated on **exactly one** of the two benchmarks (`np.setxor1d`)", "np.setxor1d(bench_a, bench_b)", "setxor1d"),
    ])
    return P(f"`bench_a` and `bench_b` list the {th.plural} evaluated on two benchmarks. Return the sorted array "
             f"of {th.plural} {desc}.", f"bench_a = {arr(a)}\nbench_b = {arr(b)}", sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def argmax_labels(rng):
    th, m, ax, rows, cols, data = table(rng, distinct_rows=True)
    v = m.var
    if rng.random() < 0.6:
        return P(f"`{v}` is a table of {m.label}: row `i` is `{th.var}[i]`, column `j` is "
                 f"`{ax.var}[j]`. For **each {th.noun}**, return the {ax.noun} label where its value "
                 f"was highest (an array of {len(rows)} labels).",
                 f"{th.var} = {arr(rows)}\n{ax.var} = {arr(cols)}\n{v} = {mat(data)}",
                 f"{ax.var}[{v}.argmax(axis=1)]", no_loops=True)
    data = [list(r) for r in zip(*[T.values(rng, m, len(rows), distinct=True) for _ in cols])]
    return P(f"`{v}` is a table of {m.label}: row `i` is `{th.var}[i]`, column `j` is `{ax.var}[j]`. "
             f"For **each {ax.noun}**, return the name of the {th.noun} with the lowest value "
             f"(an array of {len(cols)} names).",
             f"{th.var} = {arr(rows)}\n{ax.var} = {arr(cols)}\n{v} = {mat(data)}",
             f"{th.var}[{v}.argmin(axis=0)]", no_loops=True)


@template(M, "medium")
def label_counts(rng):
    what, classes = T.labels(rng, min_classes=3)
    used = rng.sample(classes, rng.randint(3, min(len(classes), 5)))
    counts = rng.sample(range(1, 8), len(used))  # distinct counts -> unique mode
    y = [c for c, k in zip(used, counts) for _ in range(k)]
    rng.shuffle(y)
    kind = rng.choice(["most", "least", "dict"])
    setup = f"y = {arr(y)}"
    if kind == "dict":
        return P(f"`y` holds the {what} label of {len(y)} samples. Use `np.unique(..., return_counts=True)` "
                 f"to build a dict mapping each label (a `str`) to its count (an `int`).", setup,
                 "labels, counts = np.unique(y, return_counts=True)\n"
                 "result = dict(zip(labels.tolist(), counts.tolist()))", require=["unique"])
    fn = "argmax" if kind == "most" else "argmin"
    return P(f"`y` holds the {what} label of {len(y)} samples. Use `np.unique` with "
             f"`return_counts=True` to find the **{kind} common** label.", setup,
             f"labels, counts = np.unique(y, return_counts=True)\nresult = labels[counts.{fn}()]",
             require=["unique"])


@template(M, "medium")
def bucketize(rng):
    th, m, _, vals = series(rng, 7, 11)
    v = m.var
    s = sorted(set(vals))
    k = rng.randint(2, 3)
    picks = sorted(rng.sample(range(1, len(s)), k))
    edges = [s[i] for i in picks]
    rule = ", ".join([f"0 if x < {edges[0]}"] +
                     [f"{i + 1} if {edges[i]} ≤ x < {edges[i + 1]}" for i in range(k - 1)] +
                     [f"{k} if x ≥ {edges[-1]}"])
    return P(f"{about(th, m, len(vals), v)}. `edges` holds sorted bucket boundaries. Give every value "
             f"its **bucket index**: {rule}. Use `np.searchsorted` (mind the `side=` argument) or "
             f"`np.digitize`.", f"{v} = {arr(vals)}\nedges = {arr(edges)}",
             f"np.searchsorted(edges, {v}, side='right')", require=["searchsorted|digitize"])


@template(M, "medium")
def sort_rows_by_column(rng):
    th, ms, names, X = feature_matrix(rng, 5, 8)
    j = rng.randrange(len(ms))
    for r, v in zip(X, T.values(rng, ms[j], len(X), distinct=True)):  # no ties: the order is unique
        r[j] = v
    feats = ", ".join(f"column {i} = {m.label}" for i, m in enumerate(ms))
    desc = rng.random() < 0.5
    return P(f"`X` has one row per {th.noun} ({feats}). **Sort the rows** of `X` by column {j} "
             f"({ms[j].name}), {'largest first' if desc else 'smallest first'}, keeping each row intact "
             f"(`argsort` on that column, then index the rows).", f"X = {mat(X)}",
             f"X[X[:, {j}].argsort()[::-1]]" if desc else f"X[X[:, {j}].argsort()]", require=["argsort"])


@template(M, "medium")
def multi_key_sort(rng):
    th = T.theme(rng)
    n = rng.randint(5, 7)
    names = T.items(rng, th, n)
    scores = [rng.choice([70, 80, 90]) for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"`names` and `score` describe {n} {th.plural}. Return the **indices** that sort them by "
                 f"`score` ascending, breaking ties by `names` alphabetically, using `np.lexsort` (the **last** "
                 f"key you pass is the primary one).", f"names = {arr(names)}\nscore = {arr(scores)}",
                 "np.lexsort((names, score))", require=["lexsort"])
    k = rng.randint(2, 3)
    v = rng.sample(range(5, 400), n + 2)
    return P(f"`latency` holds {len(v)} distinct request latencies. Find the **{k} largest** with "
             f"`np.argpartition` (O(n), no full sort), then order just those {k} largest first. Return their "
             f"indices.", f"latency = {arr(v)}",
             f"top = np.argpartition(latency, -{k})[-{k}:]\nresult = top[np.argsort(-latency[top])]",
             require=["argpartition"])


# ================================================================= hard

@template(M, "hard")
def knn_vote(rng):
    what, classes = T.labels(rng)
    classes = list(classes[:2])
    n, k = rng.randint(8, 12), rng.choice([3, 5])
    X = [[round(rng.uniform(0, 10), 1) for _ in range(2)] for _ in range(n)]
    y = [rng.randrange(2) for _ in range(n)]
    q = [round(rng.uniform(2, 8), 1) for _ in range(2)]
    return P(f"`X` holds 2-D features of {n} training samples and `y` their {what} labels (indices into "
             f"`classes`). Classify the query point `q` with **k-nearest neighbours** (k = {k}, Euclidean "
             f"distance, majority vote). Return the predicted class **name**.",
             f"classes = {lit(classes)}\nX = {mat(X)}\ny = {arr(y)}\nq = {arr(q)}",
             f"d = np.linalg.norm(X - q, axis=1)\nnearest = y[np.argsort(d)[:{k}]]\n"
             f"result = classes[np.bincount(nearest, minlength=2).argmax()]")


@template(M, "hard")
def recall_at_k(rng):
    nq, nd, k = rng.randint(3, 5), rng.randint(5, 8), rng.randint(1, 3)
    s = seeded(rng)
    rel = [rng.randrange(nd) for _ in range(nq)]
    return P(f"A retrieval model scored {nd} documents for each of {nq} queries: `scores` has shape "
             f"`({nq}, {nd})` and `relevant[i]` is the one correct document for query i. Compute **recall@{k}**: "
             f"the fraction of queries whose correct document is among their top-{k} scores. No loops "
             f"(`argsort` per row, then compare with `relevant[:, None]`).",
             f"scores = np.random.default_rng({s}).random(({nq}, {nd})).round(3)\nrelevant = {arr(rel)}\n"
             f"scores[0, relevant[0]] = 1.5  # query 0 is an easy hit",
             f"topk = np.argsort(-scores, axis=1)[:, :{k}]\nresult = (topk == relevant[:, None]).any(axis=1).mean()",
             no_loops=True)


@template(M, "hard")
def dedupe_rows(rng):
    n, d = rng.randint(6, 9), rng.randint(2, 3)
    base = [[rng.randint(0, 3) for _ in range(d)] for _ in range(n - 2)]
    rows = base + [list(rng.choice(base)) for _ in range(2)]
    rng.shuffle(rows)
    if rng.random() < 0.5:
        return P(f"`X` holds {n} training examples (rows), some exact duplicates. Remove duplicate rows but "
                 f"keep the **first occurrence of each, in the original order**: `np.unique(..., axis=0, "
                 f"return_index=True)` gives first positions; sort those positions and index `X`.",
                 f"X = {mat(rows)}", "_, first = np.unique(X, axis=0, return_index=True)\nresult = X[np.sort(first)]",
                 require=["unique"])
    return P(f"`X` holds {n} training examples (rows), some exact duplicates. Return a tuple `(unique_rows, "
             f"counts)`: the distinct rows in sorted (lexicographic) order and how many times each appears "
             f"(`np.unique` with `axis=0` and `return_counts=True`).", f"X = {mat(rows)}",
             "result = tuple(np.unique(X, axis=0, return_counts=True))", require=["unique"])


# ================================================================= challenge

@template(M, "challenge")
def split_min_max_load(rng):
    n = rng.randint(8, 12)
    vals = [rng.randint(1, 20) for _ in range(n)]
    D = rng.randint(2, 4)
    var, story = rng.choice([
        ("shards", "`shards[i]` is the size (GB) of the i-th training-data shard, in order. You must assign the "
                   "shards to `D` GPUs so that each GPU gets one **contiguous, non-empty** block of shards, in order."),
        ("chapters", "`chapters[i]` is the length (minutes) of the i-th audiobook chapter. You must split the "
                     "chapters into `D` **contiguous, non-empty** parts to record on `D` days, in order."),
    ])
    sol = (f"vals = {var}.tolist()\n\ndef parts_needed(cap):\n    parts, cur = 1, 0\n    for x in vals:\n"
           "        if cur + x > cap:\n            parts += 1\n            cur = 0\n        cur += x\n    return parts\n\n"
           "lo, hi = max(vals), sum(vals)\nwhile lo < hi:\n    mid = (lo + hi) // 2\n"
           "    if parts_needed(mid) <= D:\n        hi = mid\n    else:\n        lo = mid + 1\nresult = lo")
    tests = [rand_ints(var, seeded(rng), 1, 1001, 20000) + "\nD = 37",
             rand_ints(var, seeded(rng), 1, 10, 20000) + "\nD = 20000",
             f"{var} = np.array([7, 2, 5, 10, 8])\nD = 1",
             f"{var} = np.array([4])\nD = 1"]
    return challenge("Balance the load",
                     story + " A part's load is the sum of its values. **Minimize the largest load** and "
                     "return that minimum.", f"{var} = {arr(vals)}\nD = {D}", sol, tests,
                     "1 ≤ D ≤ n ≤ 2·10⁴, 1 ≤ values ≤ 10³", speed="O(n log(sum)) with binary search on the answer")


@template(M, "challenge")
def fewest_drops_sorted(rng):
    n = rng.randint(9, 13)
    base = sorted(rng.randint(0, 50) for _ in range(n))
    vals = [x if rng.random() < 0.65 else rng.randint(0, 50) for x in base]
    var, story = rng.choice([
        ("ts", "`ts[i]` is the timestamp (ms) a video decoder attached to the i-th frame, but some frames "
               "carry garbage timestamps."),
        ("version", "`version[i]` is the model version number logged by the i-th request in a log file, but "
                    "some log lines are corrupted."),
    ])
    sol = (f"from bisect import bisect_right\ntails = []\nfor x in {var}.tolist():\n"
           "    k = bisect_right(tails, x)\n    if k == len(tails):\n        tails.append(x)\n"
           "    else:\n        tails[k] = x\nresult = len(" + var + ") - len(tails)")
    tests = [rand_ints(var, seeded(rng), 0, 10**6, 20000), rand_ints(var, seeded(rng), 0, 5, 20000),
             f"{var} = np.arange(20000, 0, -1)", f"{var} = np.array([3])"]
    return challenge("Fewest drops to restore order",
                     story + " Find the **minimum number of entries to delete** so that the remaining "
                     "sequence is **non-decreasing** (equal neighbours are fine).",
                     f"{var} = {arr(vals)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10⁶",
                     speed="O(n log n): longest non-decreasing subsequence")

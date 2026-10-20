"""1.05 Indexing & slicing."""

from ... import themes as T
from ._common import (P, SAMPLE_NOUNS, about, an, arr, challenge, feature_matrix, lit, mat, rand_ints,
                      seeded, series, table, template)

M = "1.05"


# ================================================================= lite

@template(M, "lite")
def index_2d(rng):
    th, m, ax, rows, cols, data = table(rng)
    i, j = rng.randrange(len(rows)), rng.randrange(len(cols))
    v = m.var
    return P(f"`{v}` is a {len(rows)}×{len(cols)} table of {m.label}. Row `i` is the {th.noun} "
             f"`{th.var}[i]` and column `j` is the {ax.noun} `{ax.var}[j]`. Use 2-D indexing "
             f"(`{v}[row, col]`) to get the value for {rows[i]} on {cols[j]}.",
             f"{th.var} = {lit(rows)}\n{ax.var} = {lit(cols)}\n{v} = {mat(data)}",
             f"{v}[{i}, {j}]")


@template(M, "lite")
def slice_1d(rng):
    th, m, _, vals = series(rng, 8, 12)
    n, v, k = len(vals), m.var, rng.randint(2, 4)
    a = rng.randint(1, n - 4)
    b = rng.randint(a + 1, n - 2)
    desc, sol = rng.choice([
        (f"the first {k} values", f"{v}[:{k}]"),
        (f"the last {k} values", f"{v}[-{k}:]"),
        ("every second value, starting with the first", f"{v}[::2]"),
        ("every second value, starting with the second", f"{v}[1::2]"),
        ("all values in reverse order", f"{v}[::-1]"),
        (f"the values at positions {a} through {b}, inclusive (0-based)", f"{v}[{a}:{b + 1}]"),
        ("everything except the first and last value", f"{v}[1:-1]"),
    ])
    return P(f"{about(th, m, n, v)}. Use **slicing** to get {desc}.", f"{v} = {arr(vals)}", sol)


@template(M, "lite")
def new_axes(rng):
    n = rng.randint(3, 6)
    h, w = rng.randint(2, 4), rng.randint(2, 4)
    s = seeded(rng)
    desc, setup, sol = rng.choice([
        (f"turn `bias` (shape `({n},)`) into a **column** vector of shape `({n}, 1)` with `None` "
         f"(or `np.newaxis`)", f"bias = np.arange({n}) * 0.5", "bias[:, None]"),
        (f"turn `bias` (shape `({n},)`) into a **row** vector of shape `(1, {n})` with `None` "
         f"(or `np.newaxis`)", f"bias = np.arange({n}) * 0.5", "bias[None, :]"),
        (f"take the **red channel** (index 0 of the last axis) of the `({h}, {w}, 3)` image `img` using "
         f"`...` (Ellipsis)", f"img = np.random.default_rng({s}).integers(0, 256, size=({h}, {w}, 3))",
         "img[..., 0]"),
        (f"take the **last channel** of the `(2, {h}, {w}, 4)` RGBA batch `imgs` (the alpha channel) "
         f"using `...`", f"imgs = np.random.default_rng({s}).integers(0, 256, size=(2, {h}, {w}, 4))",
         "imgs[..., -1]"),
    ])
    return P(f"Indexing practice: {desc}.", setup, sol)


# ================================================================= easy

@template(M, "easy")
def mask_filter(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than"), (">=", "at least"), ("<=", "at most")])
    return P(f"{about(th, m, len(vals), v)}. Use a **boolean mask** to select only the values {word} {t}.",
             f"{v} = {arr(vals)}", f"{v}[{v} {op} {t}]", no_loops=True)


@template(M, "easy")
def where_replace(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    hi_word, lo_word = rng.choice([("high", "low"), ("above", "below"), ("flag", "ok"), ("hot", "cold")])
    desc, sol = rng.choice([
        (f"replace every value below {t} with {t} (a floor) and keep the others",
         f"np.where({v} < {t}, {t}, {v})"),
        (f"replace every value above {t} with {t} (a cap) and keep the others",
         f"np.where({v} > {t}, {t}, {v})"),
        (f"set every value below {t} to 0 and keep the others", f"np.where({v} < {t}, 0, {v})"),
        (f"build an array of strings: `'{hi_word}'` where the value is greater than {t}, "
         f"otherwise `'{lo_word}'`", f"np.where({v} > {t}, '{hi_word}', '{lo_word}')"),
    ])
    return P(f"{about(th, m, len(vals), v)}. Use `np.where` to {desc}.",
             f"{v} = {arr(vals)}", sol, require=["where"])


@template(M, "easy")
def slice_2d(rng):
    th, m, ax, rows, cols, data = table(rng, rows=(5, 7), cols=(5, 7))
    v = m.var
    r, c = len(rows), len(cols)
    a = rng.randint(0, r - 3)
    b = rng.randint(a + 1, r - 1)
    j = rng.randrange(c)
    desc, sol = rng.choice([
        (f"rows {a} through {b} (inclusive) and all columns", f"{v}[{a}:{b + 1}]"),
        (f"column {j} for every row (a 1-D array)", f"{v}[:, {j}]"),
        ("the last 2 rows and the first 3 columns", f"{v}[-2:, :3]"),
        ("every other row, starting with row 0", f"{v}[::2]"),
        (f"rows {a} through {b} (inclusive) and columns 1 through 3 (inclusive)", f"{v}[{a}:{b + 1}, 1:4]"),
        ("every other row in reverse order (start from the last row)", f"{v}[::-2]"),
    ])
    return P(f"`{v}` is a {r}×{c} table of {m.label} (rows: {th.plural}, columns: {ax.plural}). "
             f"Using 2-D **slicing**, get {desc}. Indices are 0-based.", f"{v} = {mat(data)}", sol)


@template(M, "easy")
def masked_assignment(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    desc, sol = rng.choice([
        (f"set every value greater than {t} to {t}", f"{v}[{v} > {t}] = {t}"),
        (f"set every value less than {t} to 0", f"{v}[{v} < {t}] = 0"),
        (f"set every value greater than {t} to -1 (a sentinel)", f"{v}[{v} > {t}] = -1"),
    ])
    return P(f"{about(th, m, len(vals), v)}. Using **masked assignment** (`{v}[mask] = value`), {desc}, "
             f"changing `{v}` in place.", f"{v} = {arr(vals)}", sol, answer=[v], no_loops=True)


# ================================================================= medium

@template(M, "medium")
def select_bins(rng):
    th, m, _, vals = series(rng, 8, 12)
    v = m.var
    s = sorted(set(vals))
    a = T.threshold(rng, s[: len(s) // 2 + 1], m)
    b = T.threshold(rng, s[len(s) // 2:], m)
    lo, mid, hi = rng.choice([("low", "medium", "high"), ("small", "medium", "large"),
                              ("bronze", "silver", "gold"), ("cold", "mild", "hot"), ("green", "amber", "red")])
    return P(f"{about(th, m, len(vals), v)}. Use `np.select` to label each value `'{lo}'` if it is "
             f"below {a}, `'{mid}'` if it is below {b} (but not below {a}), and `'{hi}'` otherwise.",
             f"{v} = {arr(vals)}", f"np.select([{v} < {a}, {v} < {b}], ['{lo}', '{mid}'], default='{hi}')",
             require=["select"])


@template(M, "medium")
def fancy_index(rng):
    th, m, ax, rows, cols, data = table(rng, rows=(5, 7), cols=(4, 7))
    v = m.var
    r, c = len(rows), len(cols)
    ri = rng.sample(range(r), rng.randint(2, 3))
    ci = rng.sample(range(c), rng.randint(2, 3))
    pairs = [(a, rng.randrange(c)) for a in ri]
    desc, sol = rng.choice([
        (f"rows {ri} in exactly that order, with all columns", f"{v}[{ri}]"),
        (f"columns {ci} in exactly that order, for every row", f"{v}[:, {ci}]"),
        (f"the sub-table made of rows {ri} and columns {ci} (in those orders)", f"{v}[np.ix_({ri}, {ci})]"),
        ("the 1-D array of values at the (row, column) pairs " + ", ".join(map(str, pairs)),
         f"{v}[{[a for a, _ in pairs]}, {[b for _, b in pairs]}]"),
    ])
    return P(f"`{v}` is a {r}×{c} table of {m.label}. Using **integer (fancy) indexing** with lists, "
             f"get {desc}.", f"{v} = {mat(data)}", sol)


@template(M, "medium")
def filter_rows(rng):
    th, ms, names, X = feature_matrix(rng, 6, 9)
    j, k = rng.sample(range(len(ms)), 2)
    tj = T.threshold(rng, [r[j] for r in X], ms[j])
    tk = T.threshold(rng, [r[k] for r in X], ms[k])
    feats = ", ".join(f"column {i} = {m.label}" for i, m in enumerate(ms))
    setup = f"{th.var} = {arr(names)}\nX = {mat(X)}"
    if rng.random() < 0.5:
        return P(f"`X` has one row per {th.noun} ({feats}). Using a boolean mask on a column, select "
                 f"the **rows** of `X` whose {ms[j].name} is greater than {tj}.", setup,
                 f"X[X[:, {j}] > {tj}]", no_loops=True)
    return P(f"`X` has one row per {th.noun} ({feats}); `{th.var}` holds the row names. Return the "
             f"**names** of the {th.plural} whose {ms[j].name} is greater than {tj} **and** whose "
             f"{ms[k].name} is less than {tk}. Combine masks with `&` (and parentheses).", setup,
             f"{th.var}[(X[:, {j}] > {tj}) & (X[:, {k}] < {tk})]", no_loops=True)


@template(M, "medium")
def gather_true_class(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 5))
    n = rng.randint(4, 7)
    s = seeded(rng)
    y = [rng.randrange(k) for _ in range(n)]
    noun = rng.choice(SAMPLE_NOUNS)
    return P(f"`probs` (shape `({n}, {k})`) holds {an(what)} classifier's predicted probabilities for {n} "
             f"{noun}s, and `labels` holds each one's true class index. Pull out each row's probability of "
             f"its **true** class (shape `({n},)`) by pairing a row index array with `labels`. No loops.",
             f"g = np.random.default_rng({s}).random(({n}, {k}))\nprobs = (g / g.sum(axis=1, keepdims=True)).round(3)\n"
             f"labels = {arr(y)}", f"probs[np.arange({n}), labels]", no_loops=True)


@template(M, "medium")
def find_positions(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 6)
    s = seeded(rng)
    t = rng.choice([0.8, 0.9, 0.7])
    setup = f"conf = np.random.default_rng({s}).random(({r}, {c})).round(2)"
    desc, sol, req = rng.choice([
        (f"the `(row, col)` coordinates of every value greater than {t}, as a `(k, 2)` array "
         f"(`np.argwhere`)", f"np.argwhere(conf > {t})", "argwhere|nonzero|where"),
        (f"the **flat** indices (as if `conf` were 1-D) of every value greater than {t} "
         f"(`np.flatnonzero`)", f"np.flatnonzero(conf > {t})", "flatnonzero|nonzero|where|argwhere"),
        (f"a tuple `(rows, cols)` of index arrays for every value greater than {t} (`np.nonzero` "
         f"or one-argument `np.where`)", f"np.nonzero(conf > {t})", "nonzero|where"),
    ])
    return P(f"`conf` is a `({r}, {c})` grid of detection confidences (rows = frames, columns = anchors). "
             f"Find {desc}.", setup, sol, require=[req])


@template(M, "medium")
def take_along_rows(rng):
    r, c = rng.randint(3, 5), rng.randint(3, 6)
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nscores = g.permutation(100)[:{r * c}].reshape({r}, {c})\n"
             f"ids = np.arange({r * c}).reshape({r}, {c})")
    if rng.random() < 0.5:
        return P(f"Each row of `scores` scores {c} candidates whose ids are in the same row of `ids`. Reorder "
                 f"every row of `ids` by **ascending** score (sort each row of `ids` by the matching row of "
                 f"`scores`). Use `np.argsort(..., axis=1)` and `np.take_along_axis`.", setup,
                 "np.take_along_axis(ids, np.argsort(scores, axis=1), axis=1)",
                 require=["take_along_axis"])
    return P(f"Each row of `scores` scores {c} candidates whose ids are in the same row of `ids`. For every "
             f"row, return the id with the **highest** score (shape `({r},)`): build the per-row argmax with "
             f"`keepdims=True` and gather with `np.take_along_axis`.", setup,
             "np.take_along_axis(ids, scores.argmax(axis=1, keepdims=True), axis=1)[:, 0]",
             require=["take_along_axis"])


# ================================================================= hard

@template(M, "hard")
def top_k_per_row(rng):
    n, V, k = rng.randint(2, 4), rng.randint(5, 8), rng.randint(2, 3)
    s = seeded(rng)
    return P(f"`logits` (shape `({n}, {V})`) holds a language model's next-token scores for {n} positions. "
             f"For every row find the **top-{k}** tokens, best first: `idx` holds their token ids (shape "
             f"`({n}, {k})`) and `vals` their scores (gather with `np.take_along_axis`). No loops.",
             f"logits = np.random.default_rng({s}).normal(size=({n}, {V})).round(3)",
             f"idx = np.argsort(-logits, axis=1)[:, :{k}]\nvals = np.take_along_axis(logits, idx, axis=1)",
             answer=["idx", "vals"], no_loops=True)


@template(M, "hard")
def center_crop_batch(rng):
    N, H, W = rng.randint(2, 3), rng.choice([6, 7, 8]), rng.choice([6, 7, 8])
    h, w = rng.choice([2, 3, 4]), rng.choice([2, 3, 4])
    return P(f"`imgs` is an NHWC batch of shape `({N}, {H}, {W}, 3)`. Take the **center crop** of size "
             f"{h}×{w} from every image in one slice: the top edge is `(H - h) // 2` and the left edge is "
             f"`(W - w) // 2`. Return shape `({N}, {h}, {w}, 3)`.",
             f"imgs = np.arange({N * H * W * 3}).reshape({N}, {H}, {W}, 3)\nh, w = {h}, {w}",
             "N, H, W, C = imgs.shape\ntop, left = (H - h) // 2, (W - w) // 2\n"
             "result = imgs[:, top:top + h, left:left + w]", no_loops=True)


@template(M, "hard")
def hard_predictions(rng):
    n, k = rng.randint(3, 6), rng.randint(3, 5)
    s = seeded(rng)
    what, _ = T.labels(rng)
    if rng.random() < 0.5:
        return P(f"`probs` (shape `({n}, {k})`) holds soft {what} predictions. Turn them into **hard one-hot "
                 f"predictions**: an array of zeros with a 1 at each row's argmax. Use `np.put_along_axis` (or "
                 f"index an identity matrix). No loops.",
                 f"probs = np.random.default_rng({s}).dirichlet(np.ones({k}), size={n}).round(3)",
                 "out = np.zeros_like(probs)\nnp.put_along_axis(out, probs.argmax(axis=1, keepdims=True), 1, axis=1)\n"
                 "result = out", no_loops=True)
    t = rng.choice([0.4, 0.5, 0.6])
    return P(f"`probs` (shape `({n}, {k})`) holds soft {what} predictions. Return each row's predicted class "
             f"index, but use `-1` (\"abstain\") for rows whose top probability is below {t}. No loops.",
             f"probs = np.random.default_rng({s}).dirichlet(np.ones({k}), size={n}).round(3)",
             f"result = np.where(probs.max(axis=1) >= {t}, probs.argmax(axis=1), -1)", no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def count_above_in_ranges(rng):
    n, q = rng.randint(8, 11), rng.randint(3, 5)
    x = [rng.randint(0, 50) for _ in range(n)]
    L, R, Tq = [], [], []
    for _ in range(q):
        a = rng.randrange(n)
        b = rng.randrange(a, n)
        L.append(a), R.append(b), Tq.append(rng.randint(0, 50))
    sol = ("vals = x.tolist()\nn = len(vals)\ntree = [0] * (n + 1)\n\ndef add(i):\n    i += 1\n"
           "    while i <= n:\n        tree[i] += 1\n        i += i & -i\n\ndef prefix(i):  # count in [0, i)\n"
           "    s = 0\n    while i > 0:\n        s += tree[i]\n        i -= i & -i\n    return s\n\n"
           "Lq, Rq, Tq = L.tolist(), R.tolist(), T.tolist()\norder = sorted(range(n), key=lambda i: -vals[i])\n"
           "qs = sorted(range(len(Tq)), key=lambda j: -Tq[j])\nans = [0] * len(Tq)\np = 0\nfor j in qs:\n"
           "    while p < n and vals[order[p]] > Tq[j]:\n        add(order[p])\n        p += 1\n"
           "    ans[j] = prefix(Rq[j] + 1) - prefix(Lq[j])\nresult = np.array(ans)")
    big = lambda s, n_, q_, hi: (f"g = np.random.default_rng({s})\nx = g.integers(0, {hi}, size={n_})\n"  # noqa: E731
                                 f"L = g.integers(0, {n_}, size={q_})\nR = np.maximum(L, g.integers(0, {n_}, size={q_}))\n"
                                 f"T = g.integers(0, {hi}, size={q_})")
    tests = [big(seeded(rng), 20000, 20000, 10**9), big(seeded(rng), 20000, 20000, 5),
             "x = np.array([7])\nL = np.array([0])\nR = np.array([0])\nT = np.array([6])"]
    return challenge("Count above in ranges",
                     "`x[i]` is the latency of request i. A dashboard asks q questions: question j asks how "
                     "many requests in positions `L[j]..R[j]` (inclusive) have latency **strictly greater** "
                     "than `T[j]`. Answer all of them, as an int array of length q.",
                     f"x = {arr(x)}\nL = {arr(L)}\nR = {arr(R)}\nT = {arr(Tq)}", sol, tests,
                     "1 ≤ n, q ≤ 2·10⁴, 0 ≤ x[i], T[j] ≤ 10⁹", answer_desc=" (a NumPy array)",
                     speed="O((n + q) log n), e.g. offline with a Fenwick tree")


@template(M, "challenge")
def min_jumps(rng):
    n = rng.randint(8, 12)
    a = [rng.randint(0, 3) for _ in range(n)]
    a[0] = max(a[0], 1)
    sol = ("a = jump.tolist()\nn = len(a)\nif n == 1:\n    result = 0\nelse:\n    steps, end, far = 0, 0, 0\n"
           "    result = -1\n    for i in range(n - 1):\n        far = max(far, i + a[i])\n        if i == end:\n"
           "            if far <= i:\n                break\n            steps += 1\n            end = far\n"
           "            if end >= n - 1:\n                result = steps\n                break")
    tests = [rand_ints("jump", seeded(rng), 0, 4, 20000), rand_ints("jump", seeded(rng), 1, 3, 20000),
             "jump = np.ones(20000, dtype=int)", "jump = np.array([0])", "jump = np.array([0, 5])"]
    return challenge("Fewest hops",
                     "A frame sampler walks through a video's frame index array. From frame i it may jump "
                     "forward to any frame `i + 1 … i + jump[i]` (a value of 0 means it is stuck there). "
                     "Starting at frame 0, return the **minimum number of jumps** to reach the last frame "
                     "(0 if there is only one frame, −1 if the last frame can't be reached).",
                     f"jump = {arr(a)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ jump[i] ≤ n", speed="O(n)")

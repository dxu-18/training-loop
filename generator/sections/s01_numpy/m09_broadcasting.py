"""1.09 Broadcasting."""

from ... import themes as T
from ._common import P, arr, challenge, feature_matrix, mat, rand_ints, seeded, table, template

M = "1.09"


# ================================================================= lite

@template(M, "lite")
def broadcast_shape_of(rng):
    pairs = [((8, 1, 6, 1), (7, 1, 5)), ((3, 1), (4,)), ((2, 1, 5), (3, 1)), ((5, 4), (1,)),
             ((16, 3, 32, 32), (3, 1, 1)), ((4, 1, 3), (1, 6, 1)), ((10, 1), (1, 20)), ((2, 3, 4), (3, 1))]
    a, b = rng.choice(pairs)
    if rng.random() < 0.5:
        a, b = b, a
    return P(f"Predict the shape of `x + y` for `x.shape == {a}` and `y.shape == {b}`, then check it "
             f"without building any arrays: use `np.broadcast_shapes`.", "", f"np.broadcast_shapes({a}, {b})",
             require=["broadcast_shapes"])


@template(M, "lite")
def add_per_row(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    off = [T.value(rng, m) for _ in rows]
    if rng.random() < 0.5:
        return P(f"`{v}` is a table of {m.label} (one row per {th.noun}). `offset[i]` is a correction for "
                 f"row i. Add `offset[i]` to **every value in row i** by turning `offset` into a column "
                 f"(`offset[:, None]`) and broadcasting.", f"{v} = {mat(data)}\noffset = {arr(off)}",
                 f"{v} + offset[:, None]", no_loops=True)
    col = [T.value(rng, m) for _ in cols]
    return P(f"`{v}` is a table of {m.label} (one row per {th.noun}, one column per {ax.noun}). `base` holds "
             f"one value per column. Subtract `base` from **every row** (shapes `({len(rows)}, {len(cols)})` "
             f"and `({len(cols)},)` broadcast directly).", f"{v} = {mat(data)}\nbase = {arr(col)}",
             f"{v} - base", no_loops=True)


# ================================================================= easy

@template(M, "easy")
def broadcast_row_vector(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    c = len(cols)
    if rng.random() < 0.5:
        base = T.values(rng, m, c)
        return P(f"`{v}` is a table of {m.label} (one row per {th.noun}, one column per {ax.noun}). "
                 f"`baseline` holds one reference value per {ax.noun}. Use **broadcasting** to "
                 f"subtract each column's baseline from every row.",
                 f"{v} = {mat(data)}\nbaseline = {arr(base)}", f"{v} - baseline", no_loops=True)
    scale = [rng.choice([2, 4, 5, 10, 0.5, 1.5]) for _ in range(c)]
    return P(f"`{v}` is a table of {m.label} (one row per {th.noun}, one column per {ax.noun}). "
             f"`scale` holds one divisor per {ax.noun}. Use **broadcasting** to divide every row by "
             f"`scale`, column by column.", f"{v} = {mat(data)}\nscale = {arr(scale)}",
             f"{v} / scale", no_loops=True)


@template(M, "easy")
def outer_by_broadcasting(rng):
    n, m = rng.randint(3, 6), rng.randint(3, 6)
    s = seeded(rng)
    setup = f"g = np.random.default_rng({s})\na = g.integers(1, 9, size={n})\nb = g.integers(1, 9, size={m})"
    desc, sol, st = rng.choice([
        (f"the `({n}, {m})` table of products `a[i] * b[j]`", "a[:, None] * b[None, :]", setup),
        (f"the `({n}, {m})` table of sums `a[i] + b[j]` (all pairwise combined costs)", "a[:, None] + b", setup),
        (f"the `({n}, {m})` table of differences `a[i] - b[j]`", "a[:, None] - b[None, :]", setup),
        (f"the {n}×{n} matrix `M[i, j] = i - j` from `np.arange({n})`",
         f"np.arange({n})[:, None] - np.arange({n})[None, :]", ""),
    ])
    return P(f"Without loops, use `None` (np.newaxis) to broadcast and build {desc}.", st, sol, no_loops=True)


@template(M, "easy")
def broadcast_to_view(rng):
    n, r = rng.randint(3, 5), rng.randint(2, 4)
    x = [rng.randint(0, 9) for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"Repeat the bias vector `b` (length {n}) for a batch of {r} rows **without copying**: use "
                 f"`np.broadcast_to` to get a read-only `({r}, {n})` view.", f"b = {arr(x)}",
                 f"np.broadcast_to(b, ({r}, {n}))", require=["broadcast_to"])
    return P(f"`col` has shape `({n}, 1)` and `row` has shape `(1, {r})`. Use `np.broadcast_arrays` to get "
             f"both expanded to their common shape `({n}, {r})`; return the pair as a tuple.",
             f"col = np.arange({n}).reshape({n}, 1)\nrow = np.arange({r}).reshape(1, {r}) * 10",
             "tuple(np.broadcast_arrays(col, row))", require=["broadcast_arrays|broadcast_to"])


# ================================================================= medium

@template(M, "medium")
def standardize_columns(rng):
    th, ms, names, X = feature_matrix(rng)
    feats = ", ".join(m.label for m in ms)
    return P(f"`X` has one row per {th.noun} and one column per feature: {feats}. **Standardize** "
             f"each column: subtract the column mean and divide by the column standard deviation "
             f"(NumPy's default population std). Use broadcasting, no loops.",
             f"X = {mat(X)}", "(X - X.mean(axis=0)) / X.std(axis=0)", no_loops=True)


@template(M, "medium")
def normalize_rows(rng):
    th, m, ax, rows, cols, data = table(rng, distinct_rows=True)
    if m.lo < 0:
        data = [[abs(x) + 1 for x in r] for r in data]
    v = m.var
    desc, sol = rng.choice([
        ("min-max scale each **row** to [0, 1]: subtract the row minimum, then divide by the row's "
         "(max - min)",
         f"({v} - {v}.min(axis=1, keepdims=True)) / ({v}.max(axis=1, keepdims=True) - {v}.min(axis=1, keepdims=True))"),
        ("center each **row**: subtract that row's mean from every value in it",
         f"{v} - {v}.mean(axis=1, keepdims=True)"),
        ("express every value as a **fraction of its row total** (each row sums to 1)",
         f"{v} / {v}.sum(axis=1, keepdims=True)"),
    ])
    return P(f"`{v}` is a table of {m.label} (one row per {th.noun}, one column per {ax.noun}). "
             f"Without loops, {desc}. (Hint: `keepdims=True` keeps the reduced axis for broadcasting.)",
             f"{v} = {mat(data)}", sol, no_loops=True)


@template(M, "medium")
def pairwise_distances(rng):
    th = T.theme(rng)
    n, k = rng.randint(4, 7), rng.randint(2, 4)
    A = [[round(rng.uniform(-5, 5), 1) for _ in range(2)] for _ in range(n)]
    C = [[round(rng.uniform(-5, 5), 1) for _ in range(2)] for _ in range(k)]
    dist = "np.sqrt(((A[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1))"
    ctx = (f"`A` holds 2-D embedding coordinates of {n} {th.plural} (shape `({n}, 2)`) and `C` holds "
           f"{k} cluster centroids (shape `({k}, 2)`).")
    setup = f"A = {mat(A)}\nC = {mat(C)}"
    variant = rng.choice(["l2", "argmin", "l1"])
    if variant == "l2":
        return P(f"{ctx} Using **broadcasting** (no loops), compute the `({n}, {k})` matrix `D` where "
                 f"`D[i, j]` is the Euclidean distance between `A[i]` and `C[j]`.", setup, dist, no_loops=True)
    if variant == "l1":
        return P(f"{ctx} Using **broadcasting** (no loops), compute the `({n}, {k})` matrix of **L1 (Manhattan)** "
                 f"distances `|A[i] - C[j]|` summed over the 2 coordinates.", setup,
                 "np.abs(A[:, None, :] - C[None, :, :]).sum(axis=-1)", no_loops=True)
    return P(f"{ctx} This is the k-means assignment step: using **broadcasting** (no loops), return "
             f"the index of the nearest centroid for each point (shape `({n},)`).", setup,
             dist + ".argmin(axis=1)", no_loops=True)


@template(M, "medium")
def minmax_columns(rng):
    th, ms, names, X = feature_matrix(rng)
    feats = ", ".join(m.label for m in ms)
    lo, hi = rng.choice([(0, 1), (-1, 1)])
    if lo == 0:
        return P(f"`X` has one row per {th.noun} and columns {feats}. **Min-max scale every column** to "
                 f"[0, 1] in one expression: `(X - min) / (max - min)` with per-column min and max.",
                 f"X = {mat(X)}", "(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))", no_loops=True)
    return P(f"`X` has one row per {th.noun} and columns {feats}. Scale **every column** to [-1, 1]: "
             f"`2 · (X - min) / (max - min) - 1` with per-column min and max.", f"X = {mat(X)}",
             "2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 1", no_loops=True)


# ================================================================= hard

def _box(rng, base=0, spread=40):
    x, y = base + rng.randint(0, spread), base + rng.randint(0, spread)
    return [x, y, x + rng.randint(5, 40), y + rng.randint(5, 40)]


IOU_SOL = ("ix1 = np.maximum(boxes[:, 0], gt[0])\niy1 = np.maximum(boxes[:, 1], gt[1])\n"
           "ix2 = np.minimum(boxes[:, 2], gt[2])\niy2 = np.minimum(boxes[:, 3], gt[3])\n"
           "inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)\n"
           "area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])\n"
           "gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])\nresult = inter / (area + gt_area - inter)")


@template(M, "hard")
def box_iou(rng):
    cls = rng.choice(next(t for t in T.THEMES if t.key == "object_classes").items)
    x1, y1 = rng.randint(0, 50), rng.randint(0, 50)
    gt = [x1, y1, x1 + rng.randint(20, 60), y1 + rng.randint(20, 60)]
    N = rng.randint(4, 6)
    boxes = []
    for _ in range(N):
        bx, by = gt[0] + rng.randint(-25, 25), gt[1] + rng.randint(-25, 25)
        boxes.append([bx, by, bx + rng.randint(10, 60), by + rng.randint(10, 60)])
    return P(f"A detector proposed {N} boxes for a **{cls}**; `gt` is the ground-truth box. Boxes are "
             f"`[x1, y1, x2, y2]` with x2 > x1 and y2 > y1. Compute the **IoU** (intersection over union) "
             f"of every box in `boxes` with `gt`, shape `({N},)`. Vectorize with `np.maximum`/`np.minimum`; "
             f"boxes that don't overlap have IoU 0.", f"gt = {arr(gt)}\nboxes = {mat(boxes)}", IOU_SOL,
             no_loops=True)


@template(M, "hard")
def pairwise_iou(rng):
    n, m = rng.randint(2, 4), rng.randint(2, 4)
    A = [_box(rng) for _ in range(n)]
    B = [_box(rng) for _ in range(m)]
    return P(f"`A` holds {n} predicted boxes and `B` holds {m} ground-truth boxes, all `[x1, y1, x2, y2]`. "
             f"Compute the `({n}, {m})` **IoU matrix** between every pair with broadcasting (`A[:, None, :]` "
             f"against `B[None, :, :]`). Non-overlapping pairs have IoU 0. No loops.",
             f"A = {mat(A)}\nB = {mat(B)}",
             "a, b = A[:, None, :], B[None, :, :]\nw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)\n"
             "h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)\n"
             "inter = w * h\narea_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])\n"
             "area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])\n"
             "result = inter / (area_a[:, None] + area_b[None, :] - inter)", no_loops=True)


@template(M, "hard")
def masked_mean_pool(rng):
    B, Tn, D = rng.randint(2, 3), rng.randint(4, 6), rng.randint(2, 4)
    s = seeded(rng)
    mask = []
    for _ in range(B):
        L = rng.randint(1, Tn)
        mask.append([1] * L + [0] * (Tn - L))
    return P(f"`emb` holds token embeddings of shape `({B}, {Tn}, {D})` for a padded batch of "
             f"{B} sentences, and `mask` (shape `({B}, {Tn})`) is 1 for real tokens and 0 for padding. "
             f"Compute the **masked mean pooling**: for each sentence, average the embeddings of its real "
             f"tokens only. Return shape `({B}, {D})`. No loops (hint: `mask[:, :, None]`).",
             f"emb = np.random.default_rng({s}).integers(-5, 6, size=({B}, {Tn}, {D}))\nmask = {mat(mask)}",
             "m = mask[:, :, None]\nresult = (emb * m).sum(axis=1) / m.sum(axis=1)", no_loops=True)


@template(M, "hard")
def relative_position_bias(rng):
    Tn, K = rng.randint(4, 6), rng.randint(1, 2)
    table_vals = [round(rng.uniform(-1, 1), 2) for _ in range(2 * K + 1)]
    return P(f"T5-style attention adds a learned bias that depends on the **relative position** `j - i` "
             f"between key j and query i, clipped to [−{K}, {K}]. `table` has {2 * K + 1} entries: `table[d + {K}]` "
             f"is the bias for clipped offset d. Build the `({Tn}, {Tn})` bias matrix `bias[i, j]` with "
             f"broadcasting (`np.arange` + `np.clip` + fancy indexing). No loops.",
             f"table = {arr(table_vals)}\nT, K = {Tn}, {K}",
             "rel = np.arange(T)[None, :] - np.arange(T)[:, None]\nresult = table[np.clip(rel, -K, K) + K]",
             no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def count_pairs_under(rng):
    n = rng.randint(8, 12)
    vals = [rng.randint(1, 40) for _ in range(n)]
    T_ = rng.randint(20, 60)
    var, story = rng.choice([
        ("mem_gb", "`mem_gb[i]` is the GPU memory (GB) model i needs. Two models can share one GPU if their "
                   "combined memory is at most `T`."),
        ("lengths", "`lengths[i]` is the token length of sequence i. Two sequences can be packed into one "
                    "training example if their combined length is at most `T`."),
    ])
    sol = (f"vals = sorted({var}.tolist())\ni, j, count = 0, len(vals) - 1, 0\nwhile i < j:\n"
           "    if vals[i] + vals[j] <= T:\n        count += j - i\n        i += 1\n    else:\n        j -= 1\n"
           "result = count")
    tests = [rand_ints(var, seeded(rng), 1, 10**6, 20000) + "\nT = 10**6",
             rand_ints(var, seeded(rng), 1, 100, 20000) + "\nT = 150",
             f"{var} = np.array([5])\nT = 100",
             f"{var} = np.array([9, 9, 9])\nT = 17"]
    return challenge("Pairs that fit",
                     story + " Count the **pairs** `(i, j)` with `i < j` that fit. (Broadcasting all pairs "
                     "would need an n×n array: too big for the hidden tests.)",
                     f"{var} = {arr(vals)}\nT = {T_}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ values, T ≤ 2·10⁶")


@template(M, "challenge")
def count_within_radius(rng):
    n, q = rng.randint(8, 11), rng.randint(3, 5)
    x = [rng.randint(0, 40) for _ in range(n)]
    qs = [rng.randint(0, 40) for _ in range(q)]
    r = rng.randint(1, 6)
    sol = ("from bisect import bisect_left, bisect_right\nxs = sorted(x.tolist())\n"
           "result = np.array([bisect_right(xs, v + r) - bisect_left(xs, v - r) for v in queries.tolist()])")
    big = lambda s, n_, q_, hi, r_: (f"g = np.random.default_rng({s})\nx = g.integers(0, {hi}, size={n_})\n"  # noqa: E731
                                     f"queries = g.integers(0, {hi}, size={q_})\nr = {r_}")
    tests = [big(seeded(rng), 20000, 20000, 10**9, 10**6), big(seeded(rng), 20000, 20000, 100, 3),
             "x = np.array([5])\nqueries = np.array([5, 100])\nr = 0"]
    return challenge("Neighbours within a radius",
                     "`x` holds 1-D embedding coordinates of n training points. For every query value "
                     "`queries[j]`, count the training points with `|x[i] - queries[j]| ≤ r`. Return the counts "
                     "as an int array. (Broadcasting `x[:, None] - queries` would need n·q memory: too big "
                     "for the hidden tests.)", f"x = {arr(x)}\nqueries = {arr(qs)}\nr = {r}", sol, tests,
                     "1 ≤ n, q ≤ 2·10⁴, 0 ≤ x[i], queries[j], r ≤ 10⁹", answer_desc=" (a NumPy array)",
                     speed="O((n + q) log n)")

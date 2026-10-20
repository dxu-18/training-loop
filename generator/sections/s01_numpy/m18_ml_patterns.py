"""1.18 Vectorized ML patterns & testing."""

from ... import themes as T
from ._common import P, SAMPLE_NOUNS, an, arr, challenge, floats, lit, mat, rand_ints, seeded, template

M = "1.18"


def _g(rng) -> str:
    return f"g = np.random.default_rng({seeded(rng)})\n"


# ================================================================= lite

@template(M, "lite")
def activation_basics(rng):
    x = floats(rng, rng.randint(4, 7), -4, 4, 2)
    desc, sol = rng.choice([
        ("the **sigmoid** `1 / (1 + e^(−x))` of every logit", "1 / (1 + np.exp(-x))"),
        ("**ReLU**: `max(x, 0)` element-wise", "np.maximum(x, 0)"),
        ("binary **predictions**: 1 where `sigmoid(x) ≥ 0.5` (equivalently x ≥ 0), else 0, as ints",
         "(x >= 0).astype(int)"),
    ])
    return P(f"`x` holds raw logits from a binary classifier. Compute {desc}.", f"x = {arr(x)}", sol, no_loops=True)


@template(M, "lite")
def accuracy_basics(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 5))
    n = rng.randint(6, 10)
    y = [rng.randrange(k) for _ in range(n)]
    if rng.random() < 0.5:
        pred = [t if rng.random() < 0.7 else rng.randrange(k) for t in y]
        return P(f"`pred` and `y` hold predicted and true {what} class indices for {n} samples. Compute the "
                 f"**accuracy** (fraction correct) in one expression: compare, then take the mean.",
                 f"pred = {arr(pred)}\ny = {arr(y)}", "(pred == y).mean()", no_loops=True)
    logits = [floats(rng, k, -3, 3, 2) for _ in range(n)]
    y[0] = max(range(k), key=lambda j: logits[0][j])  # at least one correct prediction
    return P(f"`logits` (shape `({n}, {k})`) holds {an(what)} classifier's scores and `y` the true class indices. "
             f"Compute the **accuracy**: predicted class = argmax over each row.",
             f"logits = {mat(logits)}\ny = {arr(y)}", "(logits.argmax(axis=1) == y).mean()", no_loops=True)


# ================================================================= easy

@template(M, "easy")
def regression_losses(rng):
    n = rng.randint(5, 8)
    th = T.theme(rng)
    m = rng.choice(th.measures)
    y = [round(rng.uniform(m.lo, m.hi), 2) for _ in range(n)]
    p = [round(v + rng.gauss(0, (m.hi - m.lo) / 10), 2) for v in y]
    word, sol = rng.choice([("mean squared error (MSE)", "np.mean((pred - y) ** 2)"),
                            ("mean absolute error (MAE)", "np.mean(np.abs(pred - y))"),
                            ("root mean squared error (RMSE)", "np.sqrt(np.mean((pred - y) ** 2))")])
    return P(f"A model predicts the {m.label} of {th.plural}: `pred` vs. the truth `y`. Compute the **{word}**.",
             f"pred = {arr(p)}\ny = {arr(y)}", sol, no_loops=True)


@template(M, "easy")
def binary_cross_entropy(rng):
    n = rng.randint(5, 8)
    p = [round(rng.uniform(0.01, 0.99), 3) for _ in range(n)]
    p[0] = rng.choice([0.0, 1.0])  # a saturated prediction: needs the epsilon
    t = [rng.randint(0, 1) for _ in range(n)]
    t[0] = 1 - int(p[0])
    return P(f"`p` holds predicted probabilities and `t` the 0/1 targets for {n} samples. Compute the **binary "
             f"cross-entropy** `-mean(t·log(p + 1e-12) + (1 − t)·log(1 − p + 1e-12))`. The `1e-12` keeps `log(0)` "
             f"finite (one prediction is exactly {p[0]}).", f"p = {arr(p)}\nt = {arr(t)}",
             "-np.mean(t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12))", no_loops=True)


@template(M, "easy")
def numpy_testing(rng):
    n = rng.randint(3, 5)
    a = floats(rng, n, 0, 5, 4)
    eps = rng.choice([1e-9, 1e-7, 1e-3])
    b = [round(v + eps, 12) for v in a]
    rtol = rng.choice([1e-6, 1e-4])
    if rng.random() < 0.5:
        return P(f"Write a test check: call `np.testing.assert_allclose(got, expected, rtol={rtol})` inside "
                 f"`try`/`except AssertionError`, and set `ok = True` if it passes, `ok = False` if it raises.",
                 f"got = {arr(b)}\nexpected = {arr(a)}",
                 f"try:\n    np.testing.assert_allclose(got, expected, rtol={rtol})\n    ok = True\n"
                 "except AssertionError:\n    ok = False", answer=["ok"], require=["assert_allclose"])
    y = [rng.randint(0, 3) for _ in range(n)]
    y2 = list(y)
    if rng.random() < 0.5:
        y2[0] = (y2[0] + 1) % 4
    return P("Write a test check: call `np.testing.assert_array_equal(pred, y)` inside `try`/`except "
             "AssertionError`, and set `ok = True` if the arrays are exactly equal, `ok = False` otherwise.",
             f"pred = {arr(y2)}\ny = {arr(y)}",
             "try:\n    np.testing.assert_array_equal(pred, y)\n    ok = True\nexcept AssertionError:\n    ok = False",
             answer=["ok"], require=["assert_array_equal"])


# ================================================================= medium

@template(M, "medium")
def one_hot(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 6))
    classes = list(classes[:k])
    n = rng.randint(5, 9)
    y = [rng.randrange(k) for _ in range(n)]
    if rng.random() < 0.6:
        return P(f"`y` holds integer {what} labels for {n} samples; class `i` is `classes[i]`. "
                 f"**One-hot encode** `y` into a `({n}, {k})` float array without loops "
                 f"(hint: index rows of `np.eye({k})`).",
                 f"classes = {lit(classes)}\ny = {arr(y)}", f"np.eye({k})[y]", no_loops=True)
    Y = [[1 if j == c else 0 for j in range(k)] for c in y]
    return P(f"`Y` is a one-hot `({n}, {k})` array of {what} labels; class `i` is `classes[i]`. "
             f"Recover the integer label of every row (shape `({n},)`) without loops.",
             f"classes = {lit(classes)}\nY = {mat(Y)}", "Y.argmax(axis=1)", no_loops=True)


@template(M, "medium")
def linear_backward(rng):
    n, d, k = rng.randint(3, 5), rng.randint(2, 4), rng.randint(2, 3)
    setup = _g(rng) + (f"X = g.normal(size=({n}, {d})).round(2)\nW = g.normal(size=({d}, {k})).round(2)\n"
                       f"b = np.zeros({k})\ndout = g.normal(size=({n}, {k})).round(2)")
    return P(f"Backprop through a linear layer `out = X @ W + b` by hand. Given the upstream gradient `dout = dL/dout` "
             f"(shape `({n}, {k})`), compute `dW` (shape `({d}, {k})`), `db` (shape `({k},)`) and `dX` (shape "
             f"`({n}, {d})`). Each is one matrix product or one sum.", setup,
             "dW = X.T @ dout\ndb = dout.sum(axis=0)\ndX = dout @ W.T", answer=["dW", "db", "dX"], no_loops=True)


@template(M, "medium")
def log_softmax(rng):
    n, k = rng.randint(2, 4), rng.randint(3, 5)
    z = [[round(rng.gauss(0, 2), 2) for _ in range(k)] for _ in range(n)]
    z[rng.randrange(n)] = [round(v + 900, 2) for v in z[0]]
    return P(f"Compute the row-wise **log-softmax** of `logits` (shape `({n}, {k})`) stably: subtract each row's "
             f"max `m`, then `log_probs = (x − m) − log(sum(exp(x − m)))` per row. One row has logits near 900, so "
             f"the naive `np.log(softmax)` would fail. No loops.", f"logits = {mat(z)}",
             "z = logits - logits.max(axis=1, keepdims=True)\nresult = z - np.log(np.exp(z).sum(axis=1, keepdims=True))",
             no_loops=True)


@template(M, "medium")
def squared_distance_trick(rng):
    n, m, d = rng.randint(3, 5), rng.randint(2, 4), rng.randint(2, 4)
    setup = _g(rng) + f"A = g.normal(size=({n}, {d})).round(2)\nB = g.normal(size=({m}, {d})).round(2)"
    expand = "d2 = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T"
    if rng.random() < 0.5:
        return P(f"Compute all pairwise **squared** Euclidean distances between the rows of `A` (`({n}, {d})`) and `B` "
                 f"(`({m}, {d})`) **without** building an `(n, m, d)` intermediate: expand "
                 f"`|a − b|² = |a|² + |b|² − 2·a·b` with broadcasting and one matrix product. Shape `({n}, {m})`.",
                 setup, expand + "\nresult = d2", no_loops=True, tol=1e-6)
    return P(f"Build a **1-nearest-neighbour** lookup: for every row of `A` (`({n}, {d})`) return the index of the "
             f"closest row of `B` (`({m}, {d})`). Compute the squared distances with the expansion "
             f"`|a|² + |b|² − 2·A @ B.T` (no `(n, m, d)` intermediate), then `argmin`.", setup,
             expand + "\nresult = d2.argmin(axis=1)", no_loops=True)


@template(M, "medium")
def minibatches(rng):
    n, b = rng.randint(7, 13), rng.randint(2, 4)
    setup = f"rng = np.random.default_rng({seeded(rng)})\nn, batch_size = {n}, {b}"
    if rng.random() < 0.5:
        return P(f"Make one epoch of **shuffled minibatch indices** for {n} samples: `idx = rng.permutation(n)` (one "
                 f"call), then cut it into consecutive chunks of `batch_size` (the last one may be shorter). Return "
                 f"the list of index arrays.", setup,
                 "idx = rng.permutation(n)\nresult = [idx[i:i + batch_size] for i in range(0, n, batch_size)]",
                 require=["permutation"])
    return P(f"Make one epoch of **shuffled minibatch indices** for {n} samples, dropping the incomplete last batch "
             f"(`drop_last=True`): `idx = rng.permutation(n)` (one call), keep the first `(n // batch_size) * "
             f"batch_size` indices, and reshape to `(n_batches, batch_size)`.", setup,
             "idx = rng.permutation(n)\nk = n // batch_size\nresult = idx[:k * batch_size].reshape(k, batch_size)",
             require=["permutation"])


# ================================================================= hard

@template(M, "hard")
def stable_softmax(rng):
    noun = rng.choice(SAMPLE_NOUNS)
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 5))
    n = rng.randint(3, 5)
    z = [[round(rng.gauss(0, 3), 2) for _ in range(k)] for _ in range(n)]
    big = rng.randrange(n)
    off = rng.randint(750, 1000)
    z[big] = [round(x + off, 2) for x in z[big]]
    return P(f"`logits` holds raw model scores with one row per {noun} and one column per {what} class "
             f"in `classes`. Compute the row-wise **softmax** (each row sums to 1). One row has logits "
             f"near {off}, so `np.exp` would overflow: subtract each row's max first. No loops.",
             f"classes = {lit(list(classes[:k]))}\nlogits = {mat(z)}",
             "z = logits - logits.max(axis=1, keepdims=True)\ne = np.exp(z)\nresult = e / e.sum(axis=1, keepdims=True)",
             no_loops=True)


@template(M, "hard")
def confusion_matrix(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 4))
    n = rng.randint(12, 20)
    yt = [rng.randrange(k) for _ in range(n)]
    yp = [t if rng.random() < 0.6 else rng.randrange(k) for t in yt]
    return P(f"{an(what, cap=True)} classifier with {k} classes (`classes`) was evaluated on {n} samples. Build the "
             f"`{k}×{k}` **confusion matrix** `cm` where `cm[i, j]` counts samples whose true class is "
             f"`i` and predicted class is `j`. No loops: use `np.add.at` or `np.bincount`.",
             f"classes = {lit(list(classes[:k]))}\ny_true = {arr(yt)}\ny_pred = {arr(yp)}",
             f"cm = np.zeros(({k}, {k}), dtype=int)\nnp.add.at(cm, (y_true, y_pred), 1)\nresult = cm",
             no_loops=True)


@template(M, "hard")
def train_val_split(rng):
    noun = rng.choice(SAMPLE_NOUNS)
    n, d = rng.randint(8, 14), rng.randint(2, 4)
    n_val = rng.randint(2, n // 3)
    s, s2 = seeded(rng), seeded(rng)
    setup = (f"data = np.random.default_rng({s})\nX = data.normal(size=({n}, {d})).round(2)\n"
             f"y = data.integers(0, 2, size={n})\nrng = np.random.default_rng({s2})")
    split = (f"perm = rng.permutation(len(X))\nval_idx, train_idx = perm[:{n_val}], perm[{n_val}:]\n")
    if rng.random() < 0.5:
        return P(f"`X` ({n} {noun}s × {d} features) and `y` are a dataset. Make a **train/validation "
                 f"split**: call `rng.permutation(len(X))` once; the first {n_val} shuffled indices are "
                 f"the validation set and the rest are training. Create `X_train`, `X_val`, `y_train`, "
                 f"`y_val`.", setup,
                 split + "X_train, X_val = X[train_idx], X[val_idx]\ny_train, y_val = y[train_idx], y[val_idx]",
                 answer=["X_train", "X_val", "y_train", "y_val"], require=["permutation"])
    return P(f"`X` ({n} {noun}s × {d} features) is a dataset. Split it: call `rng.permutation(len(X))` "
             f"once; the first {n_val} shuffled indices are validation, the rest training. Then "
             f"**standardize both splits using the training mean and std only** (per column), to avoid "
             f"leakage. Create `X_train` and `X_val` (both standardized).", setup,
             split + "X_train, X_val = X[train_idx], X[val_idx]\nmu, sd = X_train.mean(axis=0), X_train.std(axis=0)\n"
             "X_train, X_val = (X_train - mu) / sd, (X_val - mu) / sd",
             answer=["X_train", "X_val"], require=["permutation"])


@template(M, "hard")
def least_squares(rng):
    th = T.theme(rng)
    m = rng.choice(th.measures)
    n, d = rng.randint(8, 12), rng.randint(2, 3)
    w = [rng.randint(-4, 4) or 1 for _ in range(d)]
    b = rng.randint(-5, 5)
    X = [[round(rng.uniform(-3, 3), 2) for _ in range(d)] for _ in range(n)]
    y = [round(sum(a * c for a, c in zip(r, w)) + b + rng.gauss(0, 0.3), 3) for r in X]
    return P(f"Predict the {m.label} of {th.plural} from {d} features. `X` is `({n}, {d})` and `y` is "
             f"`({n},)`. Fit linear regression **with an intercept** by least squares: append a column "
             f"of ones to `X` and use `np.linalg.lstsq` (`rcond=None`). Return the coefficient vector "
             f"of length {d + 1} with the bias **last**.", f"X = {mat(X)}\ny = {arr(y)}",
             "A = np.column_stack([X, np.ones(len(X))])\nw, *_ = np.linalg.lstsq(A, y, rcond=None)\nresult = w",
             tol=1e-5)


@template(M, "hard")
def gradient_step(rng):
    th = T.theme(rng)
    m = rng.choice(th.measures)
    n, d = rng.randint(4, 7), rng.randint(2, 3)
    s = seeded(rng)
    lr = rng.choice([0.1, 0.05, 0.01, 0.2])
    return P(f"Linear regression predicts the {m.name} of {th.plural}: `pred = X @ w`, with loss "
             f"`L = mean((X @ w - y) ** 2)`. Compute the gradient of `L` with respect to `w` "
             f"(`2/n · Xᵀ(Xw − y)`) and take **one gradient-descent step** with learning rate `lr`. "
             f"Return the new weights.",
             f"g = np.random.default_rng({s})\nX = g.normal(size=({n}, {d})).round(2)\n"
             f"y = g.normal(size={n}).round(2)\nw = np.zeros({d})\nlr = {lr}",
             "grad = 2 / len(X) * X.T @ (X @ w - y)\nresult = w - lr * grad")


@template(M, "hard")
def image_preprocess(rng):
    N, H, W = rng.randint(1, 3), rng.randint(2, 3), rng.randint(2, 3)
    s = seeded(rng)
    if rng.random() < 0.5:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean = [round(rng.uniform(0.3, 0.6), 2) for _ in range(3)]
        std = [round(rng.uniform(0.15, 0.3), 2) for _ in range(3)]
    return P(f"`imgs` is a uint8 batch of RGB images, shape `({N}, {H}, {W}, 3)` (NHWC). Prepare it for "
             f"a PyTorch model: convert to float32 in [0, 1] (divide by 255), **normalize each channel** "
             f"with `mean` and `std`, and return it in NCHW layout, shape `({N}, 3, {H}, {W})`. No loops.",
             f"imgs = np.random.default_rng({s}).integers(0, 256, size=({N}, {H}, {W}, 3), dtype=np.uint8)\n"
             f"mean = {arr(mean)}\nstd = {arr(std)}",
             "x = imgs.astype(np.float32) / 255.0\nx = (x - mean) / std\nresult = x.transpose(0, 3, 1, 2)",
             tol=1e-5, no_loops=True)


@template(M, "hard")
def cross_entropy(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 5))
    n = rng.randint(3, 6)
    z = [[round(rng.gauss(0, 2), 2) for _ in range(k)] for _ in range(n)]
    big = rng.randrange(n)
    z[big] = [round(x + rng.randint(800, 1000), 2) for x in z[big]]
    y = [rng.randrange(k) for _ in range(n)]
    return P(f"`logits` (shape `({n}, {k})`) are {an(what)} classifier's raw scores and `y` holds the true "
             f"class index of each sample. Compute the **mean cross-entropy loss** "
             f"`mean(logsumexp(logits[i]) - logits[i, y[i]])`. One row has huge logits, so use the "
             f"max trick for log-sum-exp. No loops (pick `logits[i, y[i]]` with `np.arange(n)`).",
             f"logits = {mat(z)}\ny = {arr(y)}",
             "mx = logits.max(axis=1, keepdims=True)\nlse = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))\n"
             "result = np.mean(lse - logits[np.arange(len(y)), y])", no_loops=True)


@template(M, "hard")
def logistic_regression_gd(rng):
    n, d = rng.randint(8, 12), rng.randint(2, 3)
    steps, lr = rng.choice([20, 50, 100]), rng.choice([0.1, 0.5])
    setup = _g(rng) + (f"X = g.normal(size=({n}, {d})).round(2)\ny = (X[:, 0] + 0.5 * g.normal(size={n}) > 0).astype(float)\n"
                       f"steps, lr = {steps}, {lr}")
    return P(f"Train **logistic regression** with plain gradient descent, starting from `w = np.zeros({d})` and "
             f"`b = 0.0`. In each of `steps` iterations: `p = sigmoid(X @ w + b)`, `grad_w = Xᵀ(p − y) / n`, "
             f"`grad_b = mean(p − y)`, then `w -= lr·grad_w` and `b -= lr·grad_b`. Return the tuple `(w, b)`.",
             setup, "w, b = np.zeros(X.shape[1]), 0.0\nfor _ in range(steps):\n    p = 1 / (1 + np.exp(-(X @ w + b)))\n"
             "    w = w - lr * X.T @ (p - y) / len(X)\n    b = b - lr * np.mean(p - y)\nresult = (w, b)")


# ================================================================= challenge

@template(M, "challenge")
def best_threshold(rng):
    n = rng.randint(7, 11)
    scores = [round(rng.random(), 1) for _ in range(n)]
    y = [int(rng.random() < 0.3 + 0.5 * s_) for s_ in scores]
    sol = ("pairs = sorted(zip(scores.tolist(), y.tolist()))\nneg_below = 0\npos_total = sum(t for _, t in pairs)\n"
           "best = pos_total  # threshold below every score: everything predicted 1\npos_below = 0\ni = 0\n"
           "while i < len(pairs):\n    j = i\n    while j < len(pairs) and pairs[j][0] == pairs[i][0]:\n"
           "        if pairs[j][1]:\n            pos_below += 1\n        else:\n            neg_below += 1\n        j += 1\n"
           "    best = max(best, neg_below + pos_total - pos_below)\n    i = j\nresult = best")
    big = lambda s, n_, dec: (f"g = np.random.default_rng({s})\nscores = g.random({n_}).round({dec})\n"  # noqa: E731
                              f"y = (g.random({n_}) < 0.2 + 0.6 * scores).astype(int)")
    tests = [big(seeded(rng), 20000, 6), big(seeded(rng), 20000, 1), "scores = np.zeros(2000)\ny = np.ones(2000, dtype=int)",
             "scores = np.array([0.5])\ny = np.array([0])"]
    return challenge("Best decision threshold",
                     "A detector outputs `scores` (ties possible) and the truth is `y` (1 = positive). It predicts 1 "
                     "when `score ≥ t`. Over **all real thresholds t**, find the **maximum number of correctly "
                     "classified samples**.", f"scores = {arr(scores)}\ny = {arr(y)}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, 0 ≤ scores ≤ 1", speed="O(n log n): sort once, sweep the cut")


@template(M, "challenge")
def dynamic_padding(rng):
    n = rng.randint(5, 8)
    L = [rng.randint(1, 30) for _ in range(n)]
    K = rng.randint(2, 3)
    sol = ("lens = lengths.tolist()\nn = len(lens)\nINF = float('inf')\ndp = [0] + [INF] * n\nfor i in range(1, n + 1):\n"
           "    mx = 0\n    for size in range(1, min(K, i) + 1):\n        mx = max(mx, lens[i - size])\n"
           "        dp[i] = min(dp[i], dp[i - size] + size * mx)\nresult = dp[n]")
    tests = [rand_ints("lengths", seeded(rng), 1, 2048, 2000) + "\nK = 64",
             rand_ints("lengths", seeded(rng), 1, 2048, 2000) + "\nK = 1",
             "lengths = np.full(2000, 7)\nK = 64", "lengths = np.array([5])\nK = 3"]
    return challenge("Least padding",
                     "Sequences arrive in order with token counts `lengths`. You group **consecutive** sequences into "
                     "batches of at most `K` sequences. Each batch is padded to its longest sequence, so it costs "
                     "`(number of sequences) × (max length)` tokens. **Minimize the total tokens** over all batches.",
                     f"lengths = {arr(L)}\nK = {K}", sol, tests, "1 ≤ n ≤ 2000, 1 ≤ K ≤ 64, 1 ≤ lengths ≤ 2048",
                     speed="O(n·K) dynamic programming")

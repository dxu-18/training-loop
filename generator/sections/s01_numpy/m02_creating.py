"""1.02 Creating arrays."""

from ... import themes as T
from ._common import P, an, arr, challenge, mat, seeded, template

M = "1.02"


# ================================================================= lite

@template(M, "lite")
def filled_array(rng):
    th, ax = T.theme(rng), T.axis(rng, max_labels=12)
    r, c = rng.randint(2, 6), rng.randint(2, min(7, len(ax.labels)))
    kind = rng.choice(["zeros", "ones", "full"])
    v = rng.choice([-1, 0.5, 7, 255, 100, 3])
    what = {"zeros": "zeros", "ones": "ones", "full": f"the value {v}"}[kind]
    sol = {"zeros": f"np.zeros(({r}, {c}))", "ones": f"np.ones(({r}, {c}))",
           "full": f"np.full(({r}, {c}), {v})"}[kind]
    return P(f"You need a placeholder table with one row per {th.noun} ({r} {th.plural}) and one "
             f"column per {ax.noun} ({c} {ax.plural}). Use `np.{kind}` to create a 2-D array of "
             f"shape `({r}, {c})` filled with {what}.", "", sol, require=[kind])


@template(M, "lite")
def arange_steps(rng):
    step = rng.choice([50, 100, 250, 500, 1000, 2000])
    start = step * rng.randint(0, 3)
    stop = start + step * rng.randint(4, 12) + rng.choice([0, step // 2])
    ctx = rng.choice([
        f"A training run saves a checkpoint every {step} steps, starting at step {start}.",
        f"An audio pipeline starts a new window every {step} samples, beginning at sample {start}.",
        f"A video sampler grabs a frame every {step} ms, starting at {start} ms.",
        f"A logger records metrics every {step} iterations, starting at iteration {start}.",
    ])
    return P(f"{ctx} Use `np.arange` to create the array of all such positions below {stop} "
             f"(the stop value is excluded).", "", f"np.arange({start}, {stop}, {step})", require=["arange"])


@template(M, "lite")
def linspace_grid(rng):
    desc, a, b = rng.choice([
        ("learning-rate warmup factors", 0, 1),
        ("dropout rates to try", 0.0, 0.5),
        ("sampling temperatures", 0.1, 1.5),
        ("x-coordinates across an image", -1, 1),
        ("timestamps (in seconds) across a clip", 0, rng.choice([2, 5, 10, 30])),
        ("mixing weights for two embeddings", 0, 1),
    ])
    n = rng.randint(5, 21)
    return P(f"Create {n} evenly spaced {desc} from {a} to {b}, with **both** endpoints included, "
             f"using `np.linspace`.", "", f"np.linspace({a}, {b}, {n})", require=["linspace"])


@template(M, "lite")
def identity(rng):
    n = rng.randint(3, 8)
    ctx = rng.choice([
        f"an attention mask where each of {n} tokens may only attend to itself",
        f"the transition matrix of a {n}-state Markov chain in which every state stays put",
        f"the identity transform for {n}-dimensional embeddings",
        f"the starting point of a {n}×{n} confusion matrix for a perfect classifier",
    ])
    return P(f"Create {ctx}: the {n}×{n} **identity matrix** (ones on the diagonal, zeros elsewhere).",
             "", f"np.eye({n})", require=["eye|identity"])


@template(M, "lite")
def like_arrays(rng):
    r, c = rng.randint(2, 5), rng.randint(2, 6)
    dt = rng.choice(["float32", "int64", "float64", "int32"])
    name, what = rng.choice([("grads", "gradient buffer"), ("mask", "attention mask"),
                             ("counts", "count table"), ("acc", "accumulator")])
    kind = rng.choice(["zeros_like", "ones_like", "full_like"])
    v = rng.choice([-1, 7, 255]) if "int" in dt else rng.choice([-1.0, 0.5, 2.5])
    fill = {"zeros_like": "zeros", "ones_like": "ones", "full_like": f"the value {v}"}[kind]
    sol = f"np.{kind}(x, {v})" if kind == "full_like" else f"np.{kind}(x)"
    return P(f"`x` is a `({r}, {c})` array of dtype `{dt}`. Create {an(what)} with the **same shape and "
             f"dtype** as `x`, filled with {fill}, using `np.{kind}`.",
             f"x = np.arange({r * c}, dtype=np.{dt}).reshape({r}, {c})", sol,
             require=[kind], check_dtype=True)


# ================================================================= easy

@template(M, "easy")
def shifted_eye(rng):
    n = rng.randint(3, 6)
    m = n + rng.randint(0, 2)
    k = rng.choice([1, -1, 2, -2])
    where = f"{abs(k)} place{'s' if abs(k) > 1 else ''} {'above' if k > 0 else 'below'} the main diagonal"
    ctx = rng.choice(["a shift matrix for a sequence model", "a band in a toy attention pattern",
                      "the next-token transition matrix of a chain", "an off-diagonal test matrix"])
    return P(f"Build {ctx}: an {n}×{m} array with ones {where} and zeros elsewhere. Use `np.eye` with "
             f"the `k=` argument.", "", f"np.eye({n}, {m}, k={k})", require=["eye"])


@template(M, "easy")
def log_spaced(rng):
    variant = rng.choice(["lr", "reg", "batch", "freq"])
    if variant == "lr":
        a, b = rng.choice([(-5, -1), (-6, -2), (-4, 0)])
        n = b - a + 1 + rng.choice([0, b - a])
        return P(f"Make a learning-rate sweep: {n} values spaced evenly on a log scale from 1e{a} to "
                 f"1e{b} (both included). Use `np.logspace` (its arguments are exponents of 10).",
                 "", f"np.logspace({a}, {b}, {n})", require=["logspace"])
    if variant == "reg":
        a, b, n = rng.choice([(-3, 3, 7), (-2, 2, 5), (-4, 1, 6)])
        return P(f"Make a grid of {n} regularization strengths from 10^{a} to 10^{b}, evenly spaced on a "
                 f"log scale, with `np.logspace`.", "", f"np.logspace({a}, {b}, {n})", require=["logspace"])
    if variant == "batch":
        lo, n = rng.choice([(8, 6), (16, 7), (32, 5), (4, 8)])
        hi = lo * 2 ** (n - 1)
        return P(f"Make {n} batch sizes from {lo} to {hi}, each double the previous one, with "
                 f"`np.geomspace` (it takes the endpoints themselves).", "", f"np.geomspace({lo}, {hi}, {n})",
                 require=["geomspace"])
    lo, hi, n = rng.choice([(20, 20000, 4), (50, 8000, 5), (100, 10000, 3)])
    return P(f"Make {n} audio frequencies from {lo} Hz to {hi} Hz, spaced evenly on a log scale, with "
             f"`np.geomspace`.", "", f"np.geomspace({lo}, {hi}, {n})", require=["geomspace"])


@template(M, "easy")
def triangular_masks(rng):
    n = rng.randint(3, 7)
    desc, sol, req = rng.choice([
        (f"a **causal attention mask**: an {n}×{n} array with ones on and below the diagonal "
         f"(token i may see tokens 0..i) and zeros above. Use `np.tril` on a matrix of ones, or `np.tri`",
         f"np.tril(np.ones(({n}, {n})))", "tril|tri"),
        (f"a **future mask**: an {n}×{n} array with ones strictly above the diagonal (the positions "
         f"token i must not see) and zeros elsewhere. Use `np.triu` with `k=1`",
         f"np.triu(np.ones(({n}, {n})), k=1)", "triu"),
        (f"a {n}×{n} lower-triangular matrix of ones that also excludes the diagonal "
         f"(ones strictly below it). Use `np.tri` or `np.tril` with `k=-1`", f"np.tri({n}, k=-1)", "tri|tril"),
    ])
    return P(f"Create {desc}.", "", sol, require=[req])


@template(M, "easy")
def diagonal_matrices(rng):
    n = rng.randint(3, 5)
    s = seeded(rng)
    if rng.random() < 0.5:
        scales = [rng.choice([0.5, 1, 2, 3, 10]) for _ in range(n)]
        return P(f"`scales` holds one scaling factor per feature. Build the {n}×{n} **diagonal matrix** with "
                 f"`scales` on its diagonal using `np.diag`.", f"scales = {arr(scales)}", "np.diag(scales)",
                 require=["diag|diagflat"])
    return P(f"`C` is a {n}×{n} covariance matrix. **Extract its diagonal** (the variances) with `np.diag`.",
             f"g = np.random.default_rng({s}).normal(size=({n}, {n}))\nC = g @ g.T",
             "np.diag(C)", require=["diag|diagonal"])


# ================================================================= medium

@template(M, "medium")
def border_matrix(rng):
    r, c = rng.randint(4, 7), rng.randint(4, 7)
    variant = rng.choice(["border", "inside", "thick"])
    if variant == "border":
        return P(f"Build an {r}×{c} array with **1s on the border** and 0s inside, e.g. a padding mask "
                 f"for an image. Start from `np.ones` and zero the interior with a slice.", "",
                 f"z = np.ones(({r}, {c}))\nz[1:-1, 1:-1] = 0\nresult = z")
    if variant == "inside":
        return P(f"Build an {r}×{c} array with **0s on the border** and 1s inside: a mask that ignores the "
                 f"outer pixels of a feature map. Start from `np.zeros` and fill the interior with a slice.",
                 "", f"z = np.zeros(({r}, {c}))\nz[1:-1, 1:-1] = 1\nresult = z")
    r, c = r + 2, c + 2
    return P(f"Build an {r}×{c} array with a **border of 1s that is 2 cells thick** and 0s inside.", "",
             f"z = np.ones(({r}, {c}))\nz[2:-2, 2:-2] = 0\nresult = z")


@template(M, "medium")
def fromfunction_table(rng):
    n, m = rng.randint(3, 6), rng.randint(3, 6)
    desc, fn = rng.choice([
        (f"a {n}×{m} multiplication table where entry (i, j) is (i + 1) * (j + 1)", "(i + 1) * (j + 1)"),
        (f"an {n}×{m} distance table where entry (i, j) is |i - j| (how far apart two positions are)",
         "np.abs(i - j)"),
        (f"an {n}×{m} table of flat indices where entry (i, j) is i * {m} + j", f"i * {m} + j"),
        (f"an {n}×{m} relative-position table where entry (i, j) is j - i", "j - i"),
    ])
    return P(f"Use `np.fromfunction` (with `dtype=int`) to build {desc}.", "",
             f"np.fromfunction(lambda i, j: {fn}, ({n}, {m}), dtype=int)", require=["fromfunction"])


@template(M, "medium")
def coordinate_grids(rng):
    h, w = rng.randint(2, 4), rng.randint(2, 5)
    variant = rng.choice(["xy", "ij", "stack"])
    if variant == "xy":
        return P(f"For a {h}×{w} image (height {h}, width {w}), build pixel coordinate grids with "
                 f"`np.meshgrid`: `xs` holds each pixel's column index and `ys` its row index, both of shape "
                 f"`({h}, {w})`.", "", f"xs, ys = np.meshgrid(np.arange({w}), np.arange({h}))",
                 answer=["xs", "ys"], require=["meshgrid|indices|mgrid"])
    if variant == "ij":
        return P(f"Build matrix-style index grids for a {h}×{w} array with `np.meshgrid(..., indexing='ij')`: "
                 f"`rows[i, j] = i` and `cols[i, j] = j`, both of shape `({h}, {w})`.", "",
                 f"rows, cols = np.meshgrid(np.arange({h}), np.arange({w}), indexing='ij')",
                 answer=["rows", "cols"], require=["meshgrid|indices|mgrid"])
    return P(f"Build a `({h}, {w}, 2)` array where entry `[i, j]` is the pair `(i, j)`: the (row, col) "
             f"coordinate of every pixel. `np.indices` or `np.meshgrid(..., indexing='ij')` plus "
             f"`np.stack(..., axis=-1)` both work.", "",
             f"np.stack(np.indices(({h}, {w})), axis=-1)", require=["meshgrid|indices|mgrid"])


@template(M, "medium")
def from_iterables(rng):
    n = rng.randint(4, 9)
    variant = rng.choice(["squares", "decay", "bytes"])
    if variant == "squares":
        return P(f"Use `np.fromiter` with a generator expression to build a float array of the squares "
                 f"of 0..{n - 1}, without making a list first.", "",
                 f"np.fromiter((i * i for i in range({n})), dtype=float)", require=["fromiter"])
    if variant == "decay":
        r = rng.choice([0.5, 0.9, 0.99])
        return P(f"Use `np.fromiter` with a generator expression to build the first {n} discount factors "
                 f"{r}**t for t = 0..{n - 1} (float dtype).", "",
                 f"np.fromiter(({r} ** t for t in range({n})), dtype=float)", require=["fromiter"])
    data = bytes(rng.randrange(256) for _ in range(n))
    return P(f"`raw` holds {n} bytes read from a sensor. Interpret them as unsigned 8-bit integers with "
             f"`np.frombuffer`.", f"raw = {data!r}", "np.frombuffer(raw, dtype=np.uint8)",
             require=["frombuffer"])


# ================================================================= hard

@template(M, "hard")
def positional_encoding(rng):
    Tn, d = rng.randint(3, 7), rng.choice([4, 6, 8])
    return P(f"Build the **sinusoidal positional encoding** of a Transformer for {Tn} positions and model "
             f"dimension {d}: an array `pe` of shape `({Tn}, {d})` with\n\n"
             f"`pe[t, 2k] = sin(t / 10000**(2k/d))` and `pe[t, 2k+1] = cos(t / 10000**(2k/d))`\n\n"
             f"for t = 0..{Tn - 1} and k = 0..{d // 2 - 1}. Build the angles with `np.arange` and "
             f"broadcasting, then fill the even and odd columns with slices.", f"T, d = {Tn}, {d}",
             "pos = np.arange(T)[:, None]\nangles = pos / 10000 ** (np.arange(0, d, 2) / d)\n"
             "pe = np.zeros((T, d))\npe[:, 0::2] = np.sin(angles)\npe[:, 1::2] = np.cos(angles)\nresult = pe",
             no_loops=True)


@template(M, "hard")
def attention_bias(rng):
    n = rng.randint(4, 7)
    if rng.random() < 0.5:
        return P(f"Build the **additive causal mask** used inside attention for {n} tokens: an {n}×{n} float "
                 f"array that is 0 where key j ≤ query i and `-np.inf` where j > i (adding it to the scores "
                 f"before softmax hides the future). Use `np.triu` with `k=1` on an array full of `-np.inf`.",
                 "", f"np.triu(np.full(({n}, {n}), -np.inf), k=1)", no_loops=True)
    w = rng.randint(2, 3)
    return P(f"Build a **sliding-window attention bias** for {n} tokens with window {w}: an {n}×{n} float "
             f"array that is 0 where `0 <= i - j < {w}` (query i sees itself and the {w - 1} previous "
             f"token{'s' if w > 2 else ''}) and `-np.inf` everywhere else. Build `i - j` with `np.arange` and "
             f"broadcasting (or `np.fromfunction`), then use `np.where`.", "",
             f"diff = np.arange({n})[:, None] - np.arange({n})[None, :]\n"
             f"result = np.where((diff >= 0) & (diff < {w}), 0.0, -np.inf)", no_loops=True)


@template(M, "hard")
def polynomial_features(rng):
    n, deg = rng.randint(4, 6), rng.randint(2, 4)
    x = [round(rng.uniform(-2, 2), 1) for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"`x` holds {n} values of one feature. Build the polynomial design matrix with columns "
                 f"`1, x, x², …, x^{deg}` (shape `({n}, {deg + 1})`, lowest power first) using `np.vander` "
                 f"with `N=` and `increasing=True`.", f"x = {arr(x)}",
                 f"np.vander(x, N={deg + 1}, increasing=True)", require=["vander"])
    w = [rng.randint(-3, 3) for _ in range(deg + 1)]
    return P(f"`x` holds {n} inputs and `w` the coefficients of a degree-{deg} polynomial, lowest power first "
             f"(`w[0] + w[1]·x + …`). Build the design matrix with `np.vander(x, N={deg + 1}, increasing=True)` "
             f"and return the predictions as one matrix-vector product.", f"x = {arr(x)}\nw = {arr(w)}",
             f"np.vander(x, N={deg + 1}, increasing=True) @ w", require=["vander"])


# ================================================================= challenge

def _grid(rng, n, m, p):
    return [[1 if rng.random() < p else 0 for _ in range(m)] for _ in range(n)]


def _rand_grid(s, n, m, p):
    return f"grid = (np.random.default_rng({s}).random(({n}, {m})) < {p}).astype(int)"


@template(M, "challenge")
def count_free_squares(rng):
    n, m = rng.randint(4, 6), rng.randint(4, 7)
    g = _grid(rng, n, m, 0.75)
    sol = ("g = grid.tolist()\nn, m = len(g), len(g[0])\ndp = [[0] * (m + 1) for _ in range(n + 1)]\n"
           "total = 0\nfor i in range(n):\n    for j in range(m):\n        if g[i][j]:\n"
           "            dp[i + 1][j + 1] = 1 + min(dp[i][j + 1], dp[i + 1][j], dp[i][j])\n"
           "            total += dp[i + 1][j + 1]\nresult = total")
    tests = [_rand_grid(seeded(rng), 300, 300, 0.9), _rand_grid(seeded(rng), 300, 300, 0.995),
             "grid = np.ones((250, 250), dtype=int)", "grid = np.zeros((1, 1), dtype=int)"]
    return challenge("Count the free squares",
                     "`grid` is a robot's occupancy map built with NumPy: 1 = free cell, 0 = obstacle. "
                     "Count the **square sub-grids of any size** (1×1, 2×2, …) that contain only free cells.",
                     f"grid = {mat(g)}", sol, tests, "1 ≤ n, m ≤ 300", speed="O(n·m)")


@template(M, "challenge")
def largest_plus(rng):
    n, m = rng.randint(5, 7), rng.randint(5, 7)
    g = _grid(rng, n, m, 0.8)
    sol = ("g = grid.tolist()\nn, m = len(g), len(g[0])\nreach = [[n + m] * m for _ in range(n)]\n"
           "for i in range(n):\n    run = 0\n    for j in range(m):\n        run = run + 1 if g[i][j] else 0\n"
           "        reach[i][j] = min(reach[i][j], run)\n    run = 0\n    for j in range(m - 1, -1, -1):\n"
           "        run = run + 1 if g[i][j] else 0\n        reach[i][j] = min(reach[i][j], run)\n"
           "for j in range(m):\n    run = 0\n    for i in range(n):\n        run = run + 1 if g[i][j] else 0\n"
           "        reach[i][j] = min(reach[i][j], run)\n    run = 0\n    for i in range(n - 1, -1, -1):\n"
           "        run = run + 1 if g[i][j] else 0\n        reach[i][j] = min(reach[i][j], run)\n"
           "result = max(max(row) for row in reach)")
    tests = [_rand_grid(seeded(rng), 300, 300, 0.97), _rand_grid(seeded(rng), 300, 300, 0.5),
             "grid = np.ones((299, 300), dtype=int)", "grid = np.zeros((3, 4), dtype=int)"]
    return challenge("Largest plus sign",
                     "`grid` is a binary image (1 = foreground). A **plus sign of order k** is a centre cell "
                     "with k − 1 foreground cells continuing straight up, down, left and right from it (all "
                     "1s, centre included); order 1 is a single foreground cell. Return the **largest order** "
                     "of any plus sign in the grid (0 if there are no foreground cells).",
                     f"grid = {mat(g)}", sol, tests, "1 ≤ n, m ≤ 300", speed="O(n·m)")

"""1.03 Random numbers."""

from ... import themes as T
from ._common import P, arr, challenge, lit, seeded, template

M = "1.03"


def _rng_setup(rng, extra: str = "") -> str:
    return f"rng = np.random.default_rng({seeded(rng)})" + (f"\n{extra}" if extra else "")


# ================================================================= lite

@template(M, "lite")
def rng_draws(rng):
    n = rng.randint(3, 10)
    th = T.theme(rng)
    names = T.items(rng, th, rng.randint(5, 8))
    lo, hi = rng.choice([(0, 10), (1, 7), (0, 256), (100, 1000), (-5, 6)])
    mu, sd = rng.choice([(0, 1), (0, 0.02), (5, 2), (100, 15)])
    k = rng.randint(2, 4)
    desc, sol, req = rng.choice([
        (f"{n} random integers from {lo} (inclusive) to {hi} (exclusive) with `rng.integers`",
         f"rng.integers({lo}, {hi}, size={n})", "integers"),
        (f"{n} random floats in [0, 1) with `rng.random`", f"rng.random({n})", "random"),
        (f"{n} samples from a normal distribution with mean {mu} and standard deviation {sd} "
         f"with `rng.normal`", f"rng.normal({mu}, {sd}, size={n})", "normal"),
        (f"{k} different {th.plural} from `{th.var}` (no repeats) with `rng.choice`",
         f"rng.choice({th.var}, size={k}, replace=False)", "choice"),
        (f"a random permutation of the integers 0..{n - 1} with `rng.permutation`",
         f"rng.permutation({n})", "permutation"),
    ])
    return P(f"`rng` is a seeded NumPy random Generator. Draw {desc}. Make exactly one call on `rng`.",
             _rng_setup(rng, f"{th.var} = {lit(names)}"), sol, require=[req])


@template(M, "lite")
def rng_shapes(rng):
    r, c = rng.randint(2, 4), rng.randint(2, 5)
    lo, hi = rng.choice([(-1, 1), (0, 0.1), (-0.5, 0.5), (10, 20)])
    a, b = rng.choice([(1, 6), (0, 9), (1, 100)])
    what = rng.choice(["weight matrix", "noise tensor", "random feature table", "toy dataset"])
    desc, sol, req = rng.choice([
        (f"a `({r}, {c})` array of uniform floats in [0, 1) (`rng.random` takes a shape tuple)",
         f"rng.random(({r}, {c}))", "random"),
        (f"a `({r}, {c})` array of uniform floats in [{lo}, {hi}) with `rng.uniform`",
         f"rng.uniform({lo}, {hi}, size=({r}, {c}))", "uniform"),
        (f"a `({r}, {c})` array of standard normal samples with `rng.standard_normal`",
         f"rng.standard_normal(({r}, {c}))", "standard_normal"),
        (f"a `({r}, {c})` array of integers from {a} to {b}, **both included**, with `rng.integers` "
         f"and `endpoint=True`", f"rng.integers({a}, {b}, size=({r}, {c}), endpoint=True)", "integers"),
    ])
    return P(f"Initialize a {what}: with the seeded Generator `rng`, draw {desc}.", _rng_setup(rng), sol,
             require=[req])


# ================================================================= easy

@template(M, "easy")
def weighted_choice(rng):
    probs = rng.choice([[0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.4, 0.4, 0.2],
                        [0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    what, classes = T.labels(rng, min_classes=len(probs))
    classes = list(classes[:len(probs)])
    n = rng.randint(5, 12)
    return P(f"Simulate {n} {what} labels drawn with replacement from `classes` with probabilities `p` "
             f"(an imbalanced dataset). Use one `rng.choice` call with `size=` and `p=`.",
             _rng_setup(rng, f"classes = {lit(classes)}\np = {lit(probs)}"),
             f"rng.choice(classes, size={n}, p=p)", require=["choice"])


@template(M, "easy")
def shuffle_arrays(rng):
    r, c = rng.randint(3, 6), rng.randint(2, 4)
    data = f"X = np.arange({r * c}).reshape({r}, {c})"
    variant = rng.choice(["rows", "within", "inplace"])
    if variant == "rows":
        return P(f"`X` has one sample per row. Return a copy with the **rows shuffled** (each row stays "
                 f"intact) using `rng.permutation`.", _rng_setup(rng, data), "rng.permutation(X)",
                 require=["permutation"])
    if variant == "within":
        return P(f"Shuffle the values **within each row** of `X` independently (rows don't mix) using "
                 f"`rng.permuted` with `axis=1`.", _rng_setup(rng, data), "rng.permuted(X, axis=1)",
                 require=["permuted"])
    n = rng.randint(6, 12)
    return P(f"`order` holds the indices 0..{n - 1}. Shuffle it **in place** with `rng.shuffle` "
             f"(it returns None and changes `order` itself).", _rng_setup(rng, f"order = np.arange({n})"),
             "rng.shuffle(order)", answer=["order"], require=["shuffle"])


@template(M, "easy")
def distributions(rng):
    n = rng.randint(3, 8)
    desc, sol, req = rng.choice([
        (f"the number of correct answers out of 20 questions for {n} students who each guess with "
         f"probability 0.25 (binomial)", f"rng.binomial(20, 0.25, size={n})", "binomial"),
        (f"the number of API requests arriving in each of {n} seconds when the average is 3.5 per second "
         f"(Poisson, `lam=`)", f"rng.poisson(lam=3.5, size={n})", "poisson"),
        (f"{n} waiting times between events with mean 2.0 seconds (exponential, `scale=`)",
         f"rng.exponential(scale=2.0, size={n})", "exponential"),
        (f"{n} click-through rates from a Beta(2, 5) distribution", f"rng.beta(2, 5, size={n})", "beta"),
        (f"{n} positive response times from a log-normal with mean 0 and sigma 0.5 (of the underlying "
         f"normal)", f"rng.lognormal(0, 0.5, size={n})", "lognormal"),
        (f"{n} samples from a Gamma distribution with shape 2 and scale 1.5",
         f"rng.gamma(shape=2, scale=1.5, size={n})", "gamma"),
    ])
    return P(f"With the seeded Generator `rng`, simulate {desc}.", _rng_setup(rng), sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def sample_statistics(rng):
    mu, sd = rng.choice([(5, 2), (0, 1), (100, 15), (-3, 0.5), (50, 10)])
    n = rng.choice([500, 1000, 2000])
    variant = rng.choice(["meanstd", "within", "minmax"])
    if variant == "meanstd":
        return P(f"Draw {n} samples from N({mu}, {sd}) with the seeded Generator `rng` (one call), then "
                 f"return the tuple `(sample_mean, sample_std)` (NumPy's default std, ddof=0).",
                 _rng_setup(rng), f"x = rng.normal({mu}, {sd}, size={n})\nresult = (x.mean(), x.std())",
                 require=["normal"])
    if variant == "within":
        return P(f"Draw {n} samples from N({mu}, {sd}) with the seeded Generator `rng` (one call), then "
                 f"return the **fraction** that lie within one standard deviation of the true mean "
                 f"(|x − {mu}| < {sd}). It should be close to 0.68.", _rng_setup(rng),
                 f"x = rng.normal({mu}, {sd}, size={n})\nresult = np.mean(np.abs(x - {mu}) < {sd})",
                 require=["normal"])
    return P(f"Draw {n} samples from a uniform distribution on [{mu - sd}, {mu + sd}) with `rng.uniform` "
             f"(one call), then return `(min, max)` of the samples.", _rng_setup(rng),
             f"x = rng.uniform({mu - sd}, {mu + sd}, size={n})\nresult = (x.min(), x.max())", require=["uniform"])


@template(M, "medium")
def spawn_workers(rng):
    k = rng.randint(2, 6)
    hi = rng.choice([10, 100, 1000])
    if rng.random() < 0.5:
        return P(f"A data loader has {k} workers that each need their own independent random stream. Create "
                 f"{k} child generators with `rng.spawn({k})`, then return a list with **one** "
                 f"`integers(0, {hi})` draw from each child, in order.", _rng_setup(rng),
                 f"children = rng.spawn({k})\nresult = [int(c.integers(0, {hi})) for c in children]",
                 require=["spawn"])
    return P(f"Create {k} independent child generators with `rng.spawn({k})` (one per GPU). Each child draws "
             f"3 standard normal values; return a `({k}, 3)` array of the draws, one row per child.",
             _rng_setup(rng), f"children = rng.spawn({k})\nresult = np.array([c.standard_normal(3) for c in children])",
             require=["spawn"])


@template(M, "medium")
def legacy_global_api(rng):
    s = seeded(rng)
    r, c, n = rng.randint(2, 3), rng.randint(2, 4), rng.randint(3, 6)
    desc, sol, req = rng.choice([
        (f"seed the global generator with {s}, then draw a `({r}, {c})` array of uniform floats with "
         f"`np.random.rand`", f"np.random.seed({s})\nresult = np.random.rand({r}, {c})", "rand"),
        (f"seed the global generator with {s}, then draw {n} standard normal values with `np.random.randn`",
         f"np.random.seed({s})\nresult = np.random.randn({n})", "randn"),
        (f"seed the global generator with {s}, then draw {n} integers in [0, 10) with `np.random.randint`",
         f"np.random.seed({s})\nresult = np.random.randint(0, 10, {n})", "randint"),
    ])
    return P(f"You're reading older code that uses NumPy's **legacy global-state API**. Reproduce it: "
             f"{desc}. (New code should use `np.random.default_rng` instead.)", "", sol,
             require=["seed", req])


@template(M, "medium")
def correlated_samples(rng):
    n = rng.randint(4, 8)
    rho = rng.choice([0.8, -0.5, 0.3, 0.9])
    sx, sy = rng.choice([1, 2]), rng.choice([1, 0.5])
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    return P(f"Simulate {n} (height, weight)-style feature pairs with correlation {rho}: draw from a 2-D "
             f"normal with mean `[0, 0]` and covariance `cov` using `rng.multivariate_normal` (shape "
             f"`({n}, 2)`).", _rng_setup(rng, f"cov = {lit(cov)}"),
             f"rng.multivariate_normal(mean=[0, 0], cov=cov, size={n})", require=["multivariate_normal"])


# ================================================================= hard

@template(M, "hard")
def bootstrap_interval(rng):
    th = T.theme(rng)
    m = rng.choice(th.measures)
    n, B = rng.randint(8, 15), rng.choice([200, 500, 1000])
    vals = [round(rng.uniform(m.lo, m.hi), 2) for _ in range(n)]
    return P(f"`x` holds the {m.label} of {n} {th.plural}. Estimate a 95% **bootstrap confidence interval** "
             f"for the mean: draw all resample indices in one call `idx = rng.integers(0, n, size=({B}, n))`, "
             f"compute the mean of each resample (`x[idx]` is `({B}, n)`), and return the 2.5th and 97.5th "
             f"percentiles of those means as an array. No loops.", _rng_setup(rng, f"x = {arr(vals)}"),
             f"n = len(x)\nidx = rng.integers(0, n, size=({B}, n))\nmeans = x[idx].mean(axis=1)\n"
             "result = np.percentile(means, [2.5, 97.5])", require=["integers"], no_loops=True)


@template(M, "hard")
def inverted_dropout(rng):
    r, c = rng.randint(2, 4), rng.randint(3, 6)
    p = rng.choice([0.1, 0.2, 0.25, 0.5])
    return P(f"`h` is a `({r}, {c})` activation matrix. Apply **inverted dropout** with drop probability "
             f"p = {p}: draw `rng.random(h.shape)` once, keep an element when its draw is ≥ p, zero it "
             f"otherwise, and scale the kept elements by 1 / (1 − p) so the expected value is unchanged.",
             _rng_setup(rng, f"h = np.arange(1, {r * c + 1}, dtype=float).reshape({r}, {c})\np = {p}"),
             "keep = rng.random(h.shape) >= p\nresult = np.where(keep, h / (1 - p), 0.0)",
             require=["random"], no_loops=True)


@template(M, "hard")
def mixup_batch(rng):
    n, d, k = rng.randint(3, 5), rng.randint(2, 4), rng.randint(2, 3)
    a = rng.choice([0.2, 0.4, 1.0])
    s = seeded(rng)
    y = [rng.randrange(k) for _ in range(n)]
    return P(f"Implement **mixup** for a batch: `X` is `({n}, {d})` and `Y` is one-hot `({n}, {k})`. First "
             f"draw `lam = rng.beta({a}, {a})`, then `perm = rng.permutation({n})` (in that order). Return "
             f"`X_mix = lam·X + (1 − lam)·X[perm]` and `Y_mix` the same way for `Y`.",
             _rng_setup(rng, f"X = np.random.default_rng({s}).normal(size=({n}, {d})).round(2)\n"
                             f"Y = np.eye({k})[{lit(y)}]"),
             f"lam = rng.beta({a}, {a})\nperm = rng.permutation({n})\nX_mix = lam * X + (1 - lam) * X[perm]\n"
             "Y_mix = lam * Y + (1 - lam) * Y[perm]", answer=["X_mix", "Y_mix"], require=["beta", "permutation"])


@template(M, "hard")
def random_crop(rng):
    H, W = rng.randint(5, 8), rng.randint(5, 8)
    h, w = rng.randint(2, H - 2), rng.randint(2, W - 2)
    noun = rng.choice(["image", "spectrogram", "depth map", "satellite tile"])
    return P(f"`img` is an {H}×{W} {noun}. Take a **random crop** of size {h}×{w} for data augmentation: draw "
             f"`top = rng.integers(0, H - h + 1)` first, then `left = rng.integers(0, W - w + 1)`, and slice. "
             f"Return the crop.", _rng_setup(rng, f"img = np.arange({H * W}).reshape({H}, {W})\nh, w = {h}, {w}"),
             "H, W = img.shape\ntop = rng.integers(0, H - h + 1)\nleft = rng.integers(0, W - w + 1)\n"
             "result = img[top:top + h, left:left + w]", require=["integers"])


# ================================================================= challenge

@template(M, "challenge")
def inverse_cdf_counts(rng):
    n, q = rng.randint(4, 6), rng.randint(8, 12)
    w = [rng.randint(1, 9) for _ in range(n)]
    u = [round(rng.random() * 0.999, 3) for _ in range(q)]  # stays below 1
    sol = ("from bisect import bisect_right\ncdf = []\ntotal = 0\nfor x in weights.tolist():\n"
           "    total += x\n    cdf.append(total)\ncounts = [0] * len(cdf)\nfor v in u.tolist():\n"
           "    counts[bisect_right(cdf, v * total)] += 1\nresult = np.array(counts)")
    g = lambda s, n_, q_: (f"g = np.random.default_rng({s})\nweights = g.integers(1, 1000, size={n_})\n"  # noqa: E731
                           f"u = g.random({q_})")
    tests = [g(seeded(rng), 20000, 20000), g(seeded(rng), 3, 20000), g(seeded(rng), 20000, 5),
             "weights = np.array([5])\nu = np.array([0.0, 0.999])"]
    return challenge("Inverse-CDF sampling",
                     "A sampler picks item i with probability `weights[i] / sum(weights)`. It turns each "
                     "uniform number `u[j]` in [0, 1) into an index: the **smallest i** whose cumulative "
                     "weight `weights[0] + … + weights[i]` is **greater than** `u[j] · sum(weights)`. Return "
                     "how many times each item was picked, as an int array of length n.",
                     f"weights = {arr(w)}\nu = {arr(u)}", sol, tests,
                     "1 ≤ n, q ≤ 2·10⁴, 1 ≤ weights[i] ≤ 10³", answer_desc=" (a NumPy array)",
                     speed="O((n + q) log n)")


@template(M, "challenge")
def shuffle_period(rng):
    n = rng.randint(6, 10)
    p = list(range(n))
    rng.shuffle(p)
    sol = ("from math import lcm\nperm = p.tolist()\nseen = [False] * len(perm)\nperiod = 1\n"
           "for i in range(len(perm)):\n    if not seen[i]:\n        length, j = 0, i\n"
           "        while not seen[j]:\n            seen[j] = True\n            j = perm[j]\n"
           "            length += 1\n        period = lcm(period, length)\nresult = period")
    tests = [f"p = np.random.default_rng({seeded(rng)}).permutation(20000)",
             f"p = np.random.default_rng({seeded(rng)}).permutation(20000)",
             "p = np.roll(np.arange(20000), 1)", "p = np.array([0])"]
    return challenge("When does the order repeat?",
                     "A data loader reorders the dataset every epoch by applying the same permutation: "
                     "`order = order[p]`, where `p` came from `rng.permutation(n)`. Starting from "
                     "`order = np.arange(n)`, after how many epochs (≥ 1) is `order` sorted again for the "
                     "first time? Return the exact number as a Python int (it can be huge).",
                     f"p = {arr(p)}", sol, tests, "1 ≤ n ≤ 2·10⁴", speed="O(n)")

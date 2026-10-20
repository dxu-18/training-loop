"""1.10 Elementwise math & ufuncs."""

from ... import themes as T
from ._common import P, about, arr, challenge, floats, lit, rand_ints, seeded, series, template

M = "1.10"

CONVERSIONS = {
    "ms": ("seconds", "/ 1000"), "s": ("minutes", "/ 60"), "kg": ("grams", "* 1000"),
    "°C": ("degrees Fahrenheit", "* 9 / 5 + 32"), "W": ("kilowatts", "/ 1000"),
    "%": ("fractions between 0 and 1", "/ 100"), "GB": ("megabytes (1 GB = 1024 MB)", "* 1024"),
    "$": ("cents", "* 100"), "min": ("hours", "/ 60"), "cm": ("meters", "/ 100"),
}


# ================================================================= lite

@template(M, "lite")
def elementwise(rng):
    th, m, _, vals = series(rng)
    v = m.var
    if m.unit in CONVERSIONS and rng.random() < 0.6:
        target, op = CONVERSIONS[m.unit]
        return P(f"{about(th, m, len(vals), v)}. Convert every value to {target} with one "
                 f"**element-wise** arithmetic expression (no loops).",
                 f"{v} = {arr(vals)}", f"{v} {op}", no_loops=True)
    if m.lo >= 0:
        word, fn = rng.choice([("square root", "sqrt"), ("natural log of (1 + x)", "log1p"),
                               ("square", "square"), ("base-10 logarithm of (x + 1)", "log10")])
    else:
        word, fn = rng.choice([("absolute value", "abs"), ("square", "square"), ("sign (-1, 0 or 1)", "sign")])
    sol = f"np.log10({v} + 1)" if fn == "log10" else f"np.{fn}({v})"
    return P(f"{about(th, m, len(vals), v)}. Apply the **ufunc** `np.{fn}` to get the {word} of "
             f"every value.", f"{v} = {arr(vals)}", sol, require=[fn])


@template(M, "lite")
def rounding_functions(rng):
    vals = floats(rng, rng.randint(5, 7), -5, 5, 2)
    vals[0] = rng.choice([2.5, -2.5, 0.5, -1.5])  # a tie, so round-half-to-even shows up
    fn, word = rng.choice([("floor", "round **down** (toward −∞)"), ("ceil", "round **up** (toward +∞)"),
                           ("trunc", "drop the fractional part (round **toward zero**)"),
                           ("rint", "round to the **nearest** integer (ties go to the even one)")])
    return P(f"`x` holds model outputs. Use `np.{fn}` to {word} every value.", f"x = {arr(vals)}",
             f"np.{fn}(x)", require=[fn])


@template(M, "lite")
def elementwise_max_min(rng):
    n = rng.randint(4, 7)
    a, b = floats(rng, n, -3, 3, 1), floats(rng, n, -3, 3, 1)
    desc, sol, req = rng.choice([
        ("the **element-wise maximum** of `a` and `b` (`np.maximum`, not `np.max`)", "np.maximum(a, b)", "maximum"),
        ("the **element-wise minimum** of `a` and `b` (`np.minimum`)", "np.minimum(a, b)", "minimum"),
        ("ReLU of `a`: the element-wise maximum of `a` and 0", "np.maximum(a, 0)", "maximum"),
    ])
    return P(f"`a` and `b` are two models' scores for the same {n} items. Compute {desc}.",
             f"a = {arr(a)}\nb = {arr(b)}", sol, require=[req])


# ================================================================= easy

@template(M, "easy")
def clip_range(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    s = sorted(set(vals))
    lo = T.threshold(rng, s[: len(s) // 2 + 1], m)
    hi = T.threshold(rng, s[len(s) // 2:], m)
    return P(f"{about(th, m, len(vals), v)}. Use `np.clip` so every value lies between {lo} and "
             f"{hi}: values below {lo} become {lo}, values above {hi} become {hi}.",
             f"{v} = {arr(vals)}", f"np.clip({v}, {lo}, {hi})", require=["clip"])


@template(M, "easy")
def safe_divide(rng):
    n = rng.randint(5, 8)
    num = [rng.randint(0, 50) for _ in range(n)]
    den = [rng.choice([0, rng.randint(1, 10)]) for _ in range(n)]
    den[rng.randrange(n)] = 0
    den[rng.randrange(n)] = rng.randint(1, 10)
    num_name, den_name, what = rng.choice([("correct", "total", "per-class accuracy"),
                                           ("clicks", "views", "click-through rate"),
                                           ("tokens", "seconds", "tokens per second")])
    return P(f"Compute the {what} `{num_name} / {den_name}` element-wise, writing **0 wherever `{den_name}` is "
             f"0**, with no warnings: `np.divide(..., out=np.zeros(...), where={den_name} != 0)`.",
             f"{num_name} = np.array({lit([float(x) for x in num])})\n{den_name} = np.array({lit([float(x) for x in den])})",
             f"np.divide({num_name}, {den_name}, out=np.zeros_like({num_name}), where={den_name} != 0)",
             require=["divide"])


@template(M, "easy")
def ufunc_methods(rng):
    n = rng.randint(4, 7)
    x = [rng.randint(-5, 9) for _ in range(n)]
    y = [rng.randint(1, 5) for _ in range(rng.randint(2, 4))]
    desc, sol, req = rng.choice([
        ("the **running maximum** of `x` (best value so far) with `np.maximum.accumulate`",
         "np.maximum.accumulate(x)", "accumulate"),
        ("the **running total** of `x` with the ufunc method `np.add.accumulate`", "np.add.accumulate(x)", "accumulate"),
        ("the sum of `x` with the ufunc method `np.add.reduce`", "np.add.reduce(x)", "reduce"),
        ("the product table `x[i] * y[j]` with `np.multiply.outer`", "np.multiply.outer(x, y)", "outer"),
        ("the running minimum of `x` with `np.minimum.accumulate`", "np.minimum.accumulate(x)", "accumulate"),
    ])
    return P(f"Ufuncs have methods. Compute {desc}.", f"x = {arr(x)}\ny = {arr(y)}", sol, require=[req])


# ================================================================= medium

@template(M, "medium")
def scatter_add(rng):
    V = rng.randint(4, 7)
    n = rng.randint(8, 14)
    ids = [rng.randrange(V) for _ in range(n)]
    if rng.random() < 0.5:
        return P(f"`ids` holds {n} token ids from a vocabulary of size {V} (repeats included). Count how often "
                 f"each id appears with `np.add.at(counts, ids, 1)` on a zeros array of length {V}. (Plain "
                 f"`counts[ids] += 1` would count each repeated id only once.)", f"ids = {arr(ids)}\nV = {V}",
                 "counts = np.zeros(V, dtype=int)\nnp.add.at(counts, ids, 1)\nresult = counts", require=["at"])
    w = [round(rng.uniform(0, 2), 2) for _ in range(n)]
    return P(f"Sparse gradient update: `ids` holds {n} row indices of an embedding table with {V} rows "
             f"(repeats included) and `g` the gradient value for each. Accumulate them into a length-{V} "
             f"array with `np.add.at` so repeated ids add up.", f"ids = {arr(ids)}\ng = {arr(w)}\nV = {V}",
             "acc = np.zeros(V)\nnp.add.at(acc, ids, g)\nresult = acc", require=["at"])


@template(M, "medium")
def segment_sums(rng):
    docs = [rng.randint(1, 4) for _ in range(rng.randint(3, 5))]
    starts = [sum(docs[:i]) for i in range(len(docs))]
    total = sum(docs)
    s = seeded(rng)
    word, fn = rng.choice([("total", "add"), ("maximum", "maximum"), ("minimum", "minimum")])
    return P(f"`losses` holds per-token losses for {len(docs)} documents laid end to end ({total} tokens); "
             f"`starts` holds the index where each document begins. Compute each document's **{word}** loss "
             f"in one call with `np.{fn}.reduceat(losses, starts)`.",
             f"losses = np.random.default_rng({s}).random({total}).round(3)\nstarts = {arr(starts)}",
             f"np.{fn}.reduceat(losses, starts)", require=["reduceat"])


@template(M, "medium")
def activations(rng):
    x = floats(rng, rng.randint(5, 8), -4, 4, 2)
    desc, sol = rng.choice([
        ("**ReLU**: max(x, 0)", "np.maximum(x, 0)"),
        ("**leaky ReLU** with slope 0.1 for negatives: x if x > 0 else 0.1·x", "np.where(x > 0, x, 0.1 * x)"),
        ("**sigmoid**: 1 / (1 + e^(−x))", "1 / (1 + np.exp(-x))"),
        ("**tanh**", "np.tanh(x)"),
        ("**softplus** log(1 + e^x), computed safely with `np.logaddexp(0, x)`", "np.logaddexp(0, x)"),
        ("**SiLU / swish**: x · sigmoid(x)", "x / (1 + np.exp(-x))"),
    ])
    return P(f"`x` holds pre-activations of a layer. Compute {desc} element-wise, without loops.",
             f"x = {arr(x)}", sol, no_loops=True)


# ================================================================= hard

@template(M, "hard")
def gelu_approx(rng):
    x = floats(rng, rng.randint(5, 8), -4, 4, 2)
    if rng.random() < 0.5:
        return P("Implement the **GELU** activation with the tanh approximation used by GPT-2:\n\n"
                 "`0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`\n\nelement-wise, without loops.",
                 f"x = {arr(x)}", "0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))",
                 no_loops=True)
    x = [v * 100 for v in x]
    return P(f"`x` holds logits, some huge in magnitude. Compute **log(sigmoid(x))** in a numerically stable "
             f"way, element-wise: `log(sigmoid(x)) = -log(1 + e^(−x)) = -np.logaddexp(0, -x)`. (The naive "
             f"`np.log(1 / (1 + np.exp(-x)))` gives -inf for very negative x.)", f"x = {arr(x)}",
             "-np.logaddexp(0, -x)", require=["logaddexp"], no_loops=True)


@template(M, "hard")
def lr_schedule(rng):
    total, warm = rng.choice([(20, 4), (30, 5), (16, 3), (40, 8)])
    lr_max = rng.choice([1e-3, 3e-4, 0.1])
    lr_min = rng.choice([0.0, lr_max / 10])
    return P(f"Build a **learning-rate schedule** for {total} steps (t = 0..{total - 1}) without loops:\n\n"
             f"* warmup for t < {warm}: `lr = lr_max · (t + 1) / {warm}`\n"
             f"* afterwards, cosine decay: `lr = lr_min + 0.5 · (lr_max − lr_min) · (1 + cos(π · (t − {warm}) / "
             f"({total} − {warm})))`\n\nUse `np.arange`, `np.cos` and `np.where`.",
             f"lr_max, lr_min = {lr_max}, {lr_min}\nsteps, warmup = {total}, {warm}",
             "t = np.arange(steps)\nwarm = lr_max * (t + 1) / warmup\n"
             "cos = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - warmup) / (steps - warmup)))\n"
             "result = np.where(t < warmup, warm, cos)", no_loops=True)


@template(M, "hard")
def headings(rng):
    n = rng.randint(4, 6)
    vx, vy = floats(rng, n, -3, 3, 1), floats(rng, n, -3, 3, 1)
    for i in range(n):
        if vx[i] == 0 and vy[i] == 0:
            vx[i] = 1.0
    return P(f"A robot logged {n} velocity vectors `(vx[i], vy[i])` in m/s. Return a tuple `(speed, heading)`: "
             f"`speed` with `np.hypot`, and `heading` in **degrees in [0, 360)** measured counter-clockwise from "
             f"the +x axis, using `np.arctan2`, `np.degrees` and `% 360`. No loops.",
             f"vx = {arr(vx)}\nvy = {arr(vy)}",
             "result = (np.hypot(vx, vy), np.degrees(np.arctan2(vy, vx)) % 360)",
             require=["arctan2|atan2"], no_loops=True)


# ================================================================= challenge

@template(M, "challenge")
def gcd_subarrays(rng):
    n = rng.randint(8, 11)
    g = rng.choice([1, 2, 3, 6])
    x = [g * rng.choice([1, 2, 3, 4, 5, 6, 10]) for _ in range(n)]
    sol = ("from math import gcd\ncount = 0\nending = {}  # gcd value -> number of subarrays ending here\n"
           "for v in x.tolist():\n    nxt = {v: 1}\n    for d, c in ending.items():\n        h = gcd(d, v)\n"
           "        nxt[h] = nxt.get(h, 0) + c\n    ending = nxt\n    count += ending.get(G, 0)\nresult = count")

    def big(s, n_, G_):
        return (f"g = np.random.default_rng({s})\n"
                f"x = {G_} * g.choice([1, 2, 3, 4, 5, 6, 7, 10, 12, 15], size={n_})\nG = {G_}")

    tests = [big(seeded(rng), 20000, 1), big(seeded(rng), 20000, 2), big(seeded(rng), 20000, 6),
             "x = np.full(20000, 4)\nG = 4", "x = np.array([7])\nG = 3"]
    return challenge("Windows with a given gcd",
                     "`x` holds the batch sizes a scheduler used over time (`np.gcd` is a ufunc!). Count the "
                     "**contiguous windows** whose greatest common divisor is **exactly** `G`.",
                     f"x = {arr(x)}\nG = {g}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ x[i] ≤ 10⁶",
                     speed="O(n log max(x)): only O(log) distinct gcds end at each position")


@template(M, "challenge")
def halving_ops(rng):
    n = rng.randint(3, 5)
    x = [rng.randint(1, 60) for _ in range(n)]
    S_ = rng.randint(0, sum(x) // 2)
    sol = ("import heapq\nheap = [-v for v in x.tolist()]\nheapq.heapify(heap)\ntotal = -sum(heap)\nops = 0\n"
           "while total > S:\n    v = -heapq.heappop(heap)\n    total -= v - v // 2\n    heapq.heappush(heap, -(v // 2))\n"
           "    ops += 1\nresult = ops")
    tests = [rand_ints("x", seeded(rng), 1, 10**6, 5000) + "\nS = 10**6",
             rand_ints("x", seeded(rng), 1, 10**6, 5000) + "\nS = 0",
             "x = np.full(5000, 1)\nS = 4999", "x = np.array([5])\nS = 5"]
    return challenge("Halve the activations",
                     "To fit a memory budget you may repeatedly pick one activation value and replace it "
                     "with `floor(value / 2)` (one operation, e.g. a right shift). Return the **minimum number "
                     "of operations** so that `sum(x) ≤ S`.", f"x = {arr(x)}\nS = {S_}", sol, tests,
                     "1 ≤ n ≤ 5000, 1 ≤ x[i] ≤ 10⁶, 0 ≤ S", speed="O((n + ops) log n) with a heap")

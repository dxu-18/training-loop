"""Section 1: NumPy (COMP_GUIDE 1.01-1.18).

Every template returns a Problem whose `solution` is run to get the expected
answer, so the prompt only has to describe the task precisely.
"""

from __future__ import annotations

import math
import random

from .. import themes as T
from ..core import Problem, template

S = "numpy"
HEADER = "import numpy as np\n"


# ---------------------------------------------------------------- helpers

def lit(x) -> str:
    """Python literal for setup code (NaN -> np.nan)."""
    if isinstance(x, float) and math.isnan(x):
        return "np.nan"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(lit(v) for v in x) + "]"
    return repr(x)


def arr(x) -> str:
    return f"np.array({lit(x)})"


def mat(rows) -> str:
    return "np.array([\n" + "".join(f"    {lit(r)},\n" for r in rows) + "])"


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    return Problem(prompt=prompt.strip(), setup=(HEADER + setup.strip()).strip(),
                   solution=solution.strip(), **kw)


def pick_measure(rng: random.Random, th: T.Theme, n: int, positive: bool = False) -> T.Measure:
    """A measure with enough distinct values for n draws (and lo >= 0 if positive)."""
    ok = [m for m in th.measures
          if (m.hi - m.lo) * 10 ** m.decimals >= 4 * n and (not positive or m.lo >= 0)]
    return rng.choice(ok or list(th.measures))


def series(rng: random.Random, lo: int = 6, hi: int = 10, distinct: bool = False, positive: bool = False):
    th = T.theme(rng)
    n = rng.randint(lo, min(hi, len(th.items)))
    m = pick_measure(rng, th, n, positive)
    return th, m, T.items(rng, th, n), T.values(rng, m, n, distinct)


def table(rng: random.Random, rows=(3, 5), cols=(3, 6), distinct_rows: bool = False):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=cols[0], max_labels=12)
    r = rng.randint(*rows)
    c = rng.randint(cols[0], min(cols[1], len(ax.labels)))
    m = pick_measure(rng, th, c)
    data = [T.values(rng, m, c, distinct=distinct_rows) for _ in range(r)]
    return th, m, ax, T.items(rng, th, r), list(ax.labels[:c]), data


def about(th: T.Theme, m: T.Measure, n: int, var: str) -> str:
    return f"`{var}` holds the {m.label} of {n} {th.plural}"


def seeded(rng: random.Random) -> int:
    return rng.randrange(1, 10_000)


def solve(setup: str, solution: str):
    """Run a reference solution locally (used to print sample answers in prompts)."""
    ns: dict = {}
    exec(HEADER + setup, ns)
    exec(solution, ns)
    return ns["result"]


SAMPLE_NOUNS = ("audio clip", "image", "utterance", "product review", "email", "video frame", "support ticket")


# ================================================================= LITE

@template(S, "lite", "1.02")
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


@template(S, "lite", "1.02")
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


@template(S, "lite", "1.02")
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


@template(S, "lite", "1.04")
def list_to_dtype(rng):
    th = T.theme(rng)
    n = rng.randint(4, 8)
    m = pick_measure(rng, th, n)
    vals = T.values(rng, m, n)
    dt = rng.choice(["float32", "float64", "float16"] if m.decimals else ["int32", "int64", "float32"])
    return P(f"`{m.var}` is a plain Python list with the {m.label} of {n} {th.plural}. Convert it to a "
             f"NumPy array with dtype `{dt}` (pass `dtype=` or use `.astype`).",
             f"{m.var} = {lit(vals)}", f"np.array({m.var}, dtype=np.{dt})",
             require=["array|asarray"], check_dtype=True)


@template(S, "lite", "1.04")
def array_attribute(rng):
    B = rng.choice([2, 4, 8])
    desc, var, shape, dt = rng.choice([
        ("a batch of RGB images laid out as (batch, height, width, channels)", "images",
         (B, rng.choice([28, 32, 64]), rng.choice([28, 32, 64]), 3), "uint8"),
        ("token embeddings laid out as (batch, sequence length, embedding dim)", "embeddings",
         (B, rng.choice([16, 64, 128]), rng.choice([64, 256, 768])), "float32"),
        ("log-mel spectrograms laid out as (batch, mel bins, frames)", "spectrograms",
         (B, rng.choice([64, 80, 128]), rng.choice([100, 300, 500])), "float32"),
        ("video clips laid out as (batch, frames, channels, height, width)", "clips",
         (B, rng.choice([8, 16]), 3, 64, 64), "uint8"),
        ("attention weights laid out as (batch, heads, queries, keys)", "attn",
         (B, rng.choice([4, 8, 12]), 32, 32), "float16"),
    ])
    attr, what = rng.choice([("shape", "shape (a tuple)"), ("ndim", "number of dimensions"),
                             ("size", "total number of elements"), ("dtype", "data type (dtype)")])
    return P(f"`{var}` holds {desc}. Use an array **attribute** to get its {what}.",
             f"{var} = np.zeros({shape}, dtype=np.{dt})", f"{var}.{attr}", require=[attr])


@template(S, "lite", "1.05")
def index_2d(rng):
    th, m, ax, rows, cols, data = table(rng)
    i, j = rng.randrange(len(rows)), rng.randrange(len(cols))
    v = m.var
    return P(f"`{v}` is a {len(rows)}×{len(cols)} table of {m.label}. Row `i` is the {th.noun} "
             f"`{th.var}[i]` and column `j` is the {ax.noun} `{ax.var}[j]`. Use 2-D indexing "
             f"(`{v}[row, col]`) to get the value for {rows[i]} on {cols[j]}.",
             f"{th.var} = {lit(rows)}\n{ax.var} = {lit(cols)}\n{v} = {mat(data)}",
             f"{v}[{i}, {j}]")


@template(S, "lite", "1.05")
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
    ])
    return P(f"{about(th, m, n, v)}. Use **slicing** to get {desc}.", f"{v} = {arr(vals)}", sol)


@template(S, "lite", "1.11")
def reduce_all(rng):
    th, m, _, vals = series(rng)
    v = m.var
    desc, sol = rng.choice([
        (f"the total {m.name}", f"{v}.sum()"), (f"the average {m.name}", f"{v}.mean()"),
        (f"the highest {m.name}", f"{v}.max()"), (f"the lowest {m.name}", f"{v}.min()"),
        (f"the median {m.name}", f"np.median({v})"),
        (f"the standard deviation of the {m.name} (population std, NumPy's default)", f"{v}.std()"),
        (f"the range of the {m.name} (highest minus lowest)", f"{v}.max() - {v}.min()"),
    ])
    return P(f"{about(th, m, len(vals), v)}. Compute {desc} across all of them.",
             f"{v} = {arr(vals)}", sol)


CONVERSIONS = {
    "ms": ("seconds", "/ 1000"), "s": ("minutes", "/ 60"), "kg": ("grams", "* 1000"),
    "°C": ("degrees Fahrenheit", "* 9 / 5 + 32"), "W": ("kilowatts", "/ 1000"),
    "%": ("fractions between 0 and 1", "/ 100"), "GB": ("megabytes (1 GB = 1024 MB)", "* 1024"),
    "$": ("cents", "* 100"), "min": ("hours", "/ 60"), "cm": ("meters", "/ 100"),
}


@template(S, "lite", "1.10")
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
                               ("square", "square")])
    else:
        word, fn = rng.choice([("absolute value", "abs"), ("square", "square"), ("sign (-1, 0 or 1)", "sign")])
    return P(f"{about(th, m, len(vals), v)}. Apply the **ufunc** `np.{fn}` to get the {word} of "
             f"every value.", f"{v} = {arr(vals)}", f"np.{fn}({v})", require=[fn])


@template(S, "lite", "1.04")
def round_and_cast(rng):
    th = T.theme(rng)
    m = rng.choice([x for x in th.measures if x.decimals] or list(th.measures))
    n = rng.randint(5, 8)
    vals = [round(rng.uniform(m.lo, m.hi), 2) for _ in range(n)]
    v = m.var
    dt = rng.choice(["int64", "int32"])
    return P(f"{about(th, m, n, v)}. Round each value to the nearest whole number with `np.round`, "
             f"then cast the result to dtype `{dt}` with `.astype`.",
             f"{v} = {arr(vals)}", f"np.round({v}).astype(np.{dt})",
             require=["round|rint|around", "astype"], check_dtype=True)


@template(S, "lite", "1.02")
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


@template(S, "lite", "1.03")
def rng_draws(rng):
    seed, n = seeded(rng), rng.randint(3, 10)
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
             f"rng = np.random.default_rng({seed})\n{th.var} = {lit(names)}", sol, require=[req])


@template(S, "lite", "1.07")
def reshape_simple(rng):
    r, c = rng.randint(2, 6), rng.randint(2, 6)
    th = T.theme(rng)
    m = pick_measure(rng, th, r * c)
    vals = T.values(rng, m, r * c)
    v = m.var
    infer = rng.random() < 0.4
    how = f"with `-1` for the number of columns so NumPy infers it" if infer else ""
    sol = f"{v}.reshape({r}, -1)" if infer else f"{v}.reshape({r}, {c})"
    return P(f"`{v}` is a flat array of {r * c} {m.label} readings. **Reshape** it into {r} rows "
             f"and {c} columns {how}".strip() + ".", f"{v} = {arr(vals)}", sol, require=["reshape"])


@template(S, "lite", "1.13")
def arg_extreme(rng):
    th, m, _, vals = series(rng, distinct=True)
    v = m.var
    hi = rng.random() < 0.5
    fn = "argmax" if hi else "argmin"
    return P(f"{about(th, m, len(vals), v)}. Use `np.{fn}` to find the **index** of the "
             f"{'highest' if hi else 'lowest'} value.", f"{v} = {arr(vals)}", f"np.{fn}({v})", require=[fn])


@template(S, "lite", "1.12")
def compare_mask(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than"), (">=", "at least"), ("<=", "at most")])
    return P(f"{about(th, m, len(vals), v)}. Create a **boolean array** that is `True` where the value "
             f"is {word} {t}.", f"{v} = {arr(vals)}", f"{v} {op} {t}")


@template(S, "lite", "1.13")
def sort_values(rng):
    th, m, _, vals = series(rng)
    v = m.var
    desc = rng.random() < 0.5
    return P(f"{about(th, m, len(vals), v)}. Return the values sorted in "
             f"{'**descending** (largest first)' if desc else '**ascending**'} order using `np.sort`.",
             f"{v} = {arr(vals)}", f"np.sort({v})[::-1]" if desc else f"np.sort({v})", require=["sort"])


# ================================================================= EASY

@template(S, "easy", "1.05")
def mask_filter(rng):
    th, m, _, vals = series(rng)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than"), (">=", "at least"), ("<=", "at most")])
    return P(f"{about(th, m, len(vals), v)}. Use a **boolean mask** to select only the values {word} {t}.",
             f"{v} = {arr(vals)}", f"{v}[{v} {op} {t}]", no_loops=True)


@template(S, "easy", "1.11")
def axis_reduce(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    word, fn = rng.choice([("average", "mean"), ("total", "sum"), ("highest", "max"), ("lowest", "min")])
    if rng.random() < 0.5:
        desc, a = f"for each {th.noun} (across all {ax.plural})", 1
    else:
        desc, a = f"for each {ax.noun} (across all {th.plural})", 0
    return P(f"`{v}` is a table of {m.label} with one row per {th.noun} and one column per "
             f"{ax.noun}. Compute the {word} {desc} using the `axis` argument.",
             f"{v} = {mat(data)}", f"{v}.{fn}(axis={a})", require=["axis"])


@template(S, "easy", "1.13")
def name_of_extreme(rng):
    th, m, names, vals = series(rng, distinct=True)
    v, nv = m.var, th.var
    hi = rng.random() < 0.5
    fn = "argmax" if hi else "argmin"
    return P(f"`{nv}` and `{v}` are parallel arrays: `{v}[i]` is the {m.label} of `{nv}[i]`. "
             f"Return the **name** of the {th.noun} with the {'highest' if hi else 'lowest'} "
             f"{m.name}, using `{fn}` to index into `{nv}`.",
             f"{nv} = {arr(names)}\n{v} = {arr(vals)}", f"{nv}[np.{fn}({v})]", require=[fn])


@template(S, "easy", "1.05")
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


@template(S, "easy", "1.10")
def clip_range(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    s = sorted(set(vals))
    lo = T.threshold(rng, s[: len(s) // 2 + 1], m)
    hi = T.threshold(rng, s[len(s) // 2:], m)
    return P(f"{about(th, m, len(vals), v)}. Use `np.clip` so every value lies between {lo} and "
             f"{hi}: values below {lo} become {lo}, values above {hi} become {hi}.",
             f"{v} = {arr(vals)}", f"np.clip({v}, {lo}, {hi})", require=["clip"])


@template(S, "easy", "1.08")
def stack_two(rng):
    th = T.theme(rng)
    ax = rng.choice([a for a in T.AXES if a.labels[0][0].isalpha()])
    n = rng.randint(4, 7)
    m = pick_measure(rng, th, n)
    l1, l2 = ax.labels[0], ax.labels[1]
    a, b = T.ident(l1), T.ident(l2)
    va, vb = T.values(rng, m, n), T.values(rng, m, n)
    desc, sol = rng.choice([
        (f"Stack them as the **rows** of a 2-D array with shape `(2, {n})`", f"np.stack([{a}, {b}])"),
        (f"Stack them as the **columns** of a 2-D array with shape `({n}, 2)`", f"np.stack([{a}, {b}], axis=1)"),
        (f"Join them end to end into one 1-D array of length {2 * n} (`{a}` first)", f"np.concatenate([{a}, {b}])"),
    ])
    return P(f"`{a}` and `{b}` hold the {m.label} of the same {n} {th.plural} on {l1} and {l2}. {desc}.",
             f"{a} = {arr(va)}\n{b} = {arr(vb)}", sol)


@template(S, "easy", "1.07")
def transpose_table(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    flipped = [list(c) for c in zip(*data)]
    return P(f"`{v}` holds {m.label} with one row per {ax.noun} and one column per {th.noun}. "
             f"**Transpose** it so there is one row per {th.noun} instead.",
             f"{v} = {mat(flipped)}", f"{v}.T", require=["T|transpose|swapaxes|permute_dims"])


@template(S, "easy", "1.12")
def count_condition(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    t = T.threshold(rng, vals, m)
    op, word = rng.choice([(">", "greater than"), ("<", "less than")])
    if rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. **How many** of them have a value {word} {t}? Count "
                 f"with a boolean mask (each `True` counts as 1).",
                 f"{v} = {arr(vals)}", f"np.sum({v} {op} {t})", no_loops=True)
    return P(f"{about(th, m, len(vals), v)}. What **fraction** of them have a value {word} {t}? "
             f"(Hint: the mean of a boolean array.)", f"{v} = {arr(vals)}", f"np.mean({v} {op} {t})",
             no_loops=True)


@template(S, "easy", "1.13")
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


@template(S, "easy", "1.11")
def running_total(rng):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=5, max_labels=12)
    who = rng.choice(th.items)
    n = len(ax.labels)
    m = pick_measure(rng, th, n, positive=True)
    vals = T.values(rng, m, n)
    v = m.var
    if rng.random() < 0.6:
        return P(f"`{v}` holds the {m.label} of the {who} for each of {n} {ax.plural}, in order. "
                 f"Compute the **running total** with `np.cumsum`.",
                 f"{v} = {arr(vals)}", f"np.cumsum({v})", require=["cumsum"])
    return P(f"`{v}` holds the {m.label} of the {who} for each of {n} {ax.plural}, in order. Compute "
             f"the **running maximum** (best value so far at each position) with the ufunc method "
             f"`np.maximum.accumulate`.", f"{v} = {arr(vals)}", f"np.maximum.accumulate({v})",
             require=["accumulate"])


@template(S, "easy", "1.09")
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


@template(S, "easy", "1.12")
def missing_values(rng):
    th, m, _, vals = series(rng, 7, 11)
    vals = [float(x) for x in vals]
    for i in rng.sample(range(len(vals)), rng.randint(1, 3)):
        vals[i] = float("nan")
    v = m.var
    desc, sol, req = rng.choice([
        ("the mean, ignoring missing values (use a NaN-aware function)", f"np.nanmean({v})", ["nanmean"]),
        ("the maximum, ignoring missing values (use a NaN-aware function)", f"np.nanmax({v})", ["nanmax"]),
        ("the number of missing values (use `np.isnan`)", f"np.isnan({v}).sum()", ["isnan"]),
        ("the array with missing values removed (use `np.isnan` and `~`)", f"{v}[~np.isnan({v})]", ["isnan"]),
        ("the array with every missing value replaced by 0", f"np.nan_to_num({v}, nan=0.0)", []),
    ])
    return P(f"{about(th, m, len(vals), v)}; some readings are missing (`np.nan`). Compute {desc}. "
             f"Remember that `x == np.nan` is always False.", f"{v} = {arr(vals)}", sol, require=req)


@template(S, "easy", "1.13")
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


@template(S, "easy", "1.07")
def flatten_table(rng):
    th, m, ax, rows, cols, data = table(rng)
    v = m.var
    return P(f"`{v}` is a 2-D table of {m.label} ({len(rows)} {th.plural} × {len(cols)} {ax.plural}). "
             f"**Flatten** it into a 1-D array, row by row (`ravel`, `flatten` or `reshape(-1)`).",
             f"{v} = {mat(data)}", f"{v}.ravel()", require=["ravel|flatten|reshape"])


@template(S, "easy", "1.14")
def matmul_scores(rng):
    th = T.theme(rng)
    n, d = rng.randint(3, 6), rng.randint(2, 4)
    X = [[rng.randint(-5, 9) for _ in range(d)] for _ in range(n)]
    w = [round(rng.uniform(-2, 2), 1) for _ in range(d)]
    if rng.random() < 0.5:
        b = round(rng.uniform(-3, 3), 1)
        return P(f"`X` has one row per {th.noun} ({n}) and one column per feature ({d}). A linear "
                 f"model has weights `w` and bias `b`. Compute every row's score `X·w + b` using the "
                 f"`@` operator.", f"X = {mat(X)}\nw = {arr(w)}\nb = {b}", "X @ w + b", no_loops=True)
    k = rng.randint(2, 4)
    W = [[round(rng.uniform(-1, 1), 1) for _ in range(k)] for _ in range(d)]
    return P(f"`X` has one row per {th.noun} ({n}) and one column per feature ({d}). `W` is a "
             f"({d}, {k}) weight matrix of a linear layer with {k} outputs. Compute the ({n}, {k}) "
             f"output with a matrix product.", f"X = {mat(X)}\nW = {mat(W)}", "X @ W", no_loops=True)


@template(S, "easy", "1.05")
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
    ])
    return P(f"`{v}` is a {r}×{c} table of {m.label} (rows: {th.plural}, columns: {ax.plural}). "
             f"Using 2-D **slicing**, get {desc}. Indices are 0-based.", f"{v} = {mat(data)}", sol)


@template(S, "easy", "1.11")
def percentile_stat(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    if rng.random() < 0.3:
        return P(f"{about(th, m, len(vals), v)}. Compute the **median** value.",
                 f"{v} = {arr(vals)}", f"np.median({v})", require=["median|percentile|quantile"])
    q = rng.choice([10, 25, 75, 90, 95])
    return P(f"{about(th, m, len(vals), v)}. Compute the **{q}th percentile** (NumPy's default "
             f"linear interpolation).", f"{v} = {arr(vals)}", f"np.percentile({v}, {q})",
             require=["percentile|quantile"])


@template(S, "easy", "1.11")
def differences(rng):
    th = T.theme(rng)
    ax = T.axis(rng, min_labels=5, max_labels=12)
    who = rng.choice(th.items)
    m = pick_measure(rng, th, len(ax.labels))
    vals = T.values(rng, m, len(ax.labels))
    v = m.var
    return P(f"`{v}` holds the {m.label} of the {who} for each of {len(vals)} {ax.plural}, in order. "
             f"Compute the change from each {ax.noun} to the next (an array of length {len(vals) - 1}).",
             f"{v} = {arr(vals)}", f"np.diff({v})", no_loops=True)


# ================================================================= MEDIUM

def _feature_matrix(rng, n_lo=5, n_hi=8):
    th = T.theme(rng)
    ms = rng.sample(th.measures, min(3, len(th.measures)))
    n = rng.randint(n_lo, n_hi)
    cols = []
    for m in ms:
        col = T.values(rng, m, n)
        while len(set(col)) < 2:
            col = T.values(rng, m, n)
        cols.append(col)
    X = [list(r) for r in zip(*cols)]
    return th, ms, T.items(rng, th, n), X


@template(S, "medium", "1.09")
def standardize_columns(rng):
    th, ms, names, X = _feature_matrix(rng)
    feats = ", ".join(m.label for m in ms)
    return P(f"`X` has one row per {th.noun} and one column per feature: {feats}. **Standardize** "
             f"each column: subtract the column mean and divide by the column standard deviation "
             f"(NumPy's default population std). Use broadcasting, no loops.",
             f"X = {mat(X)}", "(X - X.mean(axis=0)) / X.std(axis=0)", no_loops=True)


@template(S, "medium", "1.09")
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


@template(S, "medium", "1.13")
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


@template(S, "medium", "1.05")
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


@template(S, "medium", "1.13")
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


@template(S, "medium", "1.05")
def fancy_index(rng):
    th, m, ax, rows, cols, data = table(rng, rows=(5, 7), cols=(4, 7))
    v = m.var
    r, c = len(rows), len(cols)
    ri = rng.sample(range(r), rng.randint(2, 3))
    ci = rng.sample(range(c), rng.randint(2, 3))
    pairs = [(a, rng.randrange(c)) for a in ri]
    desc, sol = rng.choice([
        (f"rows {ri} in exactly that order, with all columns", f"{v}[{ri}]"),
        (f"the sub-table made of rows {ri} and columns {ci} (in those orders)", f"{v}[np.ix_({ri}, {ci})]"),
        ("the 1-D array of values at the (row, column) pairs " + ", ".join(map(str, pairs)),
         f"{v}[{[a for a, _ in pairs]}, {[b for _, b in pairs]}]"),
    ])
    return P(f"`{v}` is a {r}×{c} table of {m.label}. Using **integer (fancy) indexing** with lists, "
             f"get {desc}.", f"{v} = {mat(data)}", sol)


@template(S, "medium", "1.14")
def einsum_ops(rng):
    s = seeded(rng)
    n, d, k, b = rng.randint(3, 6), rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 3)
    variant = rng.choice(["rowdot", "outer", "bmm", "scores"])
    gen = f"g = np.random.default_rng({s})\n"
    if variant == "rowdot":
        return P(f"`Q` and `K` hold query and key vectors for {n} tokens (shape `({n}, {d})`). Use "
                 f"`np.einsum` to compute each token's score: the dot product of row i of `Q` with "
                 f"row i of `K` (shape `({n},)`).",
                 gen + f"Q = g.integers(-3, 4, size=({n}, {d}))\nK = g.integers(-3, 4, size=({n}, {d}))",
                 "np.einsum('ij,ij->i', Q, K)", require=["einsum"])
    if variant == "outer":
        return P(f"`a` (length {n}) and `b` (length {k}) are two vectors. Use `np.einsum` to compute "
                 f"their **outer product** `M[i, j] = a[i] * b[j]`.",
                 gen + f"a = g.integers(-3, 4, size={n})\nb = g.integers(-3, 4, size={k})",
                 "np.einsum('i,j->ij', a, b)", require=["einsum"])
    if variant == "bmm":
        return P(f"`A` has shape `({b}, {n}, {d})` and `B` has shape `({b}, {d}, {k})`: a batch of "
                 f"{b} matrix pairs. Use `np.einsum` to compute the **batched matrix product** "
                 f"(shape `({b}, {n}, {k})`).",
                 gen + f"A = g.integers(-3, 4, size=({b}, {n}, {d}))\nB = g.integers(-3, 4, size=({b}, {d}, {k}))",
                 "np.einsum('bnd,bdk->bnk', A, B)", require=["einsum"])
    return P(f"`Q` and `K` have shape `({b}, {n}, {d})`: (heads, tokens, head_dim). Use `np.einsum` to "
             f"compute attention scores `S[h, i, j] = Q[h, i] · K[h, j]` (shape `({b}, {n}, {n})`).",
             gen + f"Q = g.integers(-3, 4, size=({b}, {n}, {d}))\nK = g.integers(-3, 4, size=({b}, {n}, {d}))",
             "np.einsum('hid,hjd->hij', Q, K)", require=["einsum"])


@template(S, "medium", "1.09")
def pairwise_distances(rng):
    th = T.theme(rng)
    n, k = rng.randint(4, 7), rng.randint(2, 4)
    A = [[round(rng.uniform(-5, 5), 1) for _ in range(2)] for _ in range(n)]
    C = [[round(rng.uniform(-5, 5), 1) for _ in range(2)] for _ in range(k)]
    dist = "np.sqrt(((A[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1))"
    ctx = (f"`A` holds 2-D embedding coordinates of {n} {th.plural} (shape `({n}, 2)`) and `C` holds "
           f"{k} cluster centroids (shape `({k}, 2)`).")
    setup = f"A = {mat(A)}\nC = {mat(C)}"
    if rng.random() < 0.5:
        return P(f"{ctx} Using **broadcasting** (no loops), compute the `({n}, {k})` matrix `D` where "
                 f"`D[i, j]` is the Euclidean distance between `A[i]` and `C[j]`.", setup, dist, no_loops=True)
    return P(f"{ctx} This is the k-means assignment step: using **broadcasting** (no loops), return "
             f"the index of the nearest centroid for each point (shape `({n},)`).", setup,
             dist + ".argmin(axis=1)", no_loops=True)


@template(S, "medium", "1.13")
def group_totals(rng):
    th = T.theme(rng)
    k = rng.randint(3, 5)
    names = T.items(rng, th, k)
    m = pick_measure(rng, th, 10, positive=True)
    n = rng.randint(8, 14)
    cat = list(range(k)) + [rng.randrange(k) for _ in range(n - k)]
    rng.shuffle(cat)
    vals = T.values(rng, m, n)
    v = m.var
    ctx = (f"Each of {n} log records belongs to one {th.noun}: `cat[i]` is an index into "
           f"`{th.var}` and `{v}[i]` is the record's {m.label}.")
    setup = f"{th.var} = {lit(names)}\ncat = {arr(cat)}\n{v} = {arr(vals)}"
    variant = rng.choice(["sum", "count", "mean"])
    if variant == "sum":
        return P(f"{ctx} Without loops, compute the **total** {m.name} per {th.noun} as an array of "
                 f"length {k} in index order (`np.bincount` with `weights=`, or `np.add.at`).",
                 setup, f"np.bincount(cat, weights={v}, minlength={k})", no_loops=True)
    if variant == "count":
        return P(f"{ctx} Without loops, count the **records** per {th.noun} as an array of length "
                 f"{k} in index order (`np.bincount`).", setup, f"np.bincount(cat, minlength={k})", no_loops=True)
    return P(f"{ctx} Without loops, compute the **mean** {m.name} per {th.noun} as an array of length "
             f"{k} in index order (sum per group / count per group).", setup,
             f"np.bincount(cat, weights={v}, minlength={k}) / np.bincount(cat, minlength={k})", no_loops=True)


@template(S, "medium", "1.18")
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


@template(S, "medium", "1.15")
def moving_average(rng):
    n, w = rng.randint(8, 14), rng.randint(3, 5)
    desc, var, vals = rng.choice([
        ("training loss logged at each step", "loss", [round(3 * math.exp(-i / 6) + rng.uniform(0, 0.4), 3) for i in range(n)]),
        ("GPU utilization (%) sampled every minute", "util", [rng.randint(20, 100) for _ in range(n)]),
        ("frame-level speech probability from a voice-activity detector", "vad", [round(rng.random(), 2) for _ in range(n)]),
        ("daily active users (thousands) of a chatbot", "dau_k", [rng.randint(40, 90) for _ in range(n)]),
    ])
    return P(f"`{var}` holds the {desc} ({n} values). Compute its **{w}-point moving average**, keeping "
             f"only full windows (length {n - w + 1}). `np.convolve(..., mode='valid')` with a kernel of "
             f"`{w}` equal weights does it in one call.", f"{var} = {arr(vals)}",
             f"np.convolve({var}, np.ones({w}) / {w}, mode='valid')", no_loops=True)


@template(S, "medium", "1.07")
def reshape_then_reduce(rng):
    s = seeded(rng)
    variant = rng.choice(["daily", "weekly", "frames", "tokens"])
    if variant == "daily":
        D = rng.randint(2, 5)
        return P(f"`temp_c` holds hourly GPU temperatures for {D} full days ({D * 24} values, in order). "
                 f"Use `reshape` and a reduction to get the **mean temperature per day** (shape `({D},)`).",
                 f"temp_c = np.random.default_rng({s}).integers(35, 90, size={D * 24})",
                 f"temp_c.reshape({D}, 24).mean(axis=1)", require=["reshape"], no_loops=True)
    if variant == "weekly":
        W = rng.randint(2, 6)
        return P(f"`signups` holds daily sign-ups for an AI app over {W} full weeks ({W * 7} values). "
                 f"Use `reshape` and a reduction to get the **total per week** (shape `({W},)`).",
                 f"signups = np.random.default_rng({s}).integers(0, 500, size={W * 7})",
                 f"signups.reshape({W}, 7).sum(axis=1)", require=["reshape"], no_loops=True)
    if variant == "frames":
        F, k = rng.choice([4, 5, 8]), rng.randint(3, 6)
        return P(f"`audio` is a mono waveform of {F * k} samples. Split it into non-overlapping frames "
                 f"of {F} samples with `reshape`, then compute each frame's **energy** = mean of the "
                 f"squared samples (shape `({k},)`).",
                 f"audio = np.random.default_rng({s}).normal(0, 0.1, size={F * k}).round(3)",
                 f"(audio.reshape(-1, {F}) ** 2).mean(axis=1)", require=["reshape"], no_loops=True)
    B, Tn, Dm = rng.randint(2, 4), rng.randint(3, 6), rng.randint(2, 4)
    return P(f"`flat` holds token features for a batch flattened to shape `({B * Tn}, {Dm})`: "
             f"{B} sequences of {Tn} tokens each, in order. Reshape to `({B}, {Tn}, {Dm})` and "
             f"**average over the tokens** of each sequence (shape `({B}, {Dm})`).",
             f"flat = np.random.default_rng({s}).integers(0, 10, size=({B * Tn}, {Dm}))",
             f"flat.reshape({B}, {Tn}, {Dm}).mean(axis=1)", require=["reshape"], no_loops=True)


@template(S, "medium", "1.08")
def pad_arrays(rng):
    s = seeded(rng)
    variant = rng.choice(["right", "left", "border", "reflect"])
    if variant in ("right", "left"):
        n = rng.randint(3, 7)
        L = n + rng.randint(2, 5)
        pad = rng.choice([0, 1, 50256, -100])
        ids = [rng.randint(2, 32000) for _ in range(n)]
        side = ("at the **end** (right padding)" if variant == "right"
                else "at the **front** (left padding, used for batched generation with decoder-only LMs)")
        width = f"(0, {L} - len(ids))" if variant == "right" else f"({L} - len(ids), 0)"
        return P(f"`ids` holds {n} token ids. Use `np.pad` to pad it to length {L} with the pad id "
                 f"{pad}, {side}.", f"ids = {arr(ids)}",
                 f"np.pad(ids, {width}, constant_values={pad})", require=["pad"])
    h, w, b = rng.randint(2, 4), rng.randint(2, 4), rng.randint(1, 2)
    if variant == "border":
        return P(f"`img` is a {h}×{w} grayscale image. Use `np.pad` to add a border of zeros "
                 f"{b} pixel{'s' if b > 1 else ''} wide on every side (like `padding={b}` in a conv layer).",
                 f"img = np.random.default_rng({s}).integers(0, 256, size=({h}, {w}))",
                 f"np.pad(img, {b})", require=["pad"])
    return P(f"`img` is a {h + 1}×{w + 1} grayscale image. Use `np.pad` with `mode='reflect'` to add a "
             f"1-pixel reflected border on every side.",
             f"img = np.random.default_rng({s}).integers(0, 256, size=({h + 1}, {w + 1}))",
             "np.pad(img, 1, mode='reflect')", require=["pad"])


@template(S, "medium", "1.05")
def filter_rows(rng):
    th, ms, names, X = _feature_matrix(rng, 6, 9)
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


@template(S, "medium", "1.14")
def linalg_ops(rng):
    s = seeded(rng)
    n = rng.randint(2, 4)
    variant = rng.choice(["solve", "norms", "unit", "inv"])
    g = f"g = np.random.default_rng({s})\n"
    if variant == "solve":
        return P(f"Solve the linear system `A @ x = b` for `x` with `np.linalg.solve` "
                 f"(`A` is {n}×{n}).", g + f"A = g.integers(-3, 4, size=({n}, {n})) + {2 * n} * np.eye({n})\n"
                 f"b = g.integers(-9, 10, size={n})", "np.linalg.solve(A, b)", require=["solve"])
    if variant == "inv":
        return P(f"Compute the inverse of the {n}×{n} matrix `A` with `np.linalg.inv`.",
                 g + f"A = g.integers(-3, 4, size=({n}, {n})) + {2 * n} * np.eye({n})",
                 "np.linalg.inv(A)", require=["inv"])
    th = T.theme(rng)
    k, d = rng.randint(3, 6), rng.randint(3, 5)
    setup = g + f"E = g.normal(size=({k}, {d})).round(2)"
    if variant == "norms":
        return P(f"`E` holds {d}-dimensional embeddings of {k} {th.plural}, one per row. Compute the "
                 f"**L2 norm of each row** (shape `({k},)`) with `np.linalg.norm`.", setup,
                 "np.linalg.norm(E, axis=1)", require=["norm"])
    return P(f"`E` holds {d}-dimensional embeddings of {k} {th.plural}, one per row. **L2-normalize** "
             f"every row so it has length 1 (`np.linalg.norm` with `axis=1, keepdims=True`).", setup,
             "E / np.linalg.norm(E, axis=1, keepdims=True)", require=["norm"], no_loops=True)


@template(S, "medium", "1.13")
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


@template(S, "medium", "1.07")
def move_axes(rng):
    s = seeded(rng)
    B, H, W = rng.randint(2, 3), rng.randint(2, 4), rng.randint(2, 4)
    req = ["transpose|moveaxis|swapaxes|permute_dims"]
    variant = rng.choice(["nhwc", "nchw", "heads", "time"])
    if variant == "nhwc":
        return P(f"`x` is an image batch in NHWC layout, shape `({B}, {H}, {W}, 3)`. PyTorch wants NCHW. "
                 f"Rearrange the axes to get shape `({B}, 3, {H}, {W})`.",
                 f"x = np.arange({B * H * W * 3}).reshape({B}, {H}, {W}, 3)", "x.transpose(0, 3, 1, 2)", require=req)
    if variant == "nchw":
        return P(f"`x` is an image batch in NCHW layout, shape `({B}, 3, {H}, {W})`. Matplotlib wants "
                 f"channels last. Rearrange the axes to get NHWC, shape `({B}, {H}, {W}, 3)`.",
                 f"x = np.arange({B * H * W * 3}).reshape({B}, 3, {H}, {W})", "x.transpose(0, 2, 3, 1)", require=req)
    if variant == "heads":
        Tn, nh, hd = rng.randint(2, 4), rng.choice([2, 3]), rng.choice([2, 4])
        return P(f"`x` has shape `({B}, {Tn}, {nh * hd})` = (batch, tokens, heads·head_dim) with {nh} "
                 f"heads of size {hd}. **Split the heads**: reshape to `({B}, {Tn}, {nh}, {hd})` and "
                 f"move the head axis forward to get shape `({B}, {nh}, {Tn}, {hd})`.",
                 f"x = np.arange({B * Tn * nh * hd}).reshape({B}, {Tn}, {nh * hd})",
                 f"x.reshape({B}, {Tn}, {nh}, {hd}).transpose(0, 2, 1, 3)", require=req)
    Tn, D = rng.randint(3, 5), rng.randint(2, 4)
    return P(f"`x` has shape `({B}, {Tn}, {D})` = (batch, time, features). An old RNN API wants "
             f"time first: swap the first two axes to get shape `({Tn}, {B}, {D})`.",
             f"x = np.arange({B * Tn * D}).reshape({B}, {Tn}, {D})", "x.swapaxes(0, 1)", require=req)


@template(S, "medium", "1.08")
def split_chunks(rng):
    th, m, _, vals = series(rng, 7, 12)
    v = m.var
    k = rng.randint(2, 4)
    if len(vals) % k == 0 or rng.random() < 0.5:
        return P(f"{about(th, m, len(vals), v)}. Use `np.array_split` to split it into {k} nearly equal "
                 f"consecutive chunks; return the **list** of chunks.", f"{v} = {arr(vals)}",
                 f"np.array_split({v}, {k})", require=["array_split|split"])
    return P(f"{about(th, m, len(vals), v)}. Use `np.array_split` to split it into {k} nearly equal "
             f"consecutive chunks, and return a **list of each chunk's mean**.", f"{v} = {arr(vals)}",
             f"[chunk.mean() for chunk in np.array_split({v}, {k})]", require=["array_split|split"])


# ================================================================= HARD

@template(S, "hard", "1.18")
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


@template(S, "hard", "1.18")
def confusion_matrix(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 4))
    n = rng.randint(12, 20)
    yt = [rng.randrange(k) for _ in range(n)]
    yp = [t if rng.random() < 0.6 else rng.randrange(k) for t in yt]
    return P(f"A {what} classifier with {k} classes (`classes`) was evaluated on {n} samples. Build the "
             f"`{k}×{k}` **confusion matrix** `cm` where `cm[i, j]` counts samples whose true class is "
             f"`i` and predicted class is `j`. No loops: use `np.add.at` or `np.bincount`.",
             f"classes = {lit(list(classes[:k]))}\ny_true = {arr(yt)}\ny_pred = {arr(yp)}",
             f"cm = np.zeros(({k}, {k}), dtype=int)\nnp.add.at(cm, (y_true, y_pred), 1)\nresult = cm",
             no_loops=True)


@template(S, "hard", "1.12")
def precision_recall_f1(rng):
    pos = rng.choice(["spam", "toxic", "defect", "fraud", "speech", "cat"])
    n = rng.randint(12, 20)
    while True:
        yt = [int(rng.random() < 0.45) for _ in range(n)]
        yp = [t if rng.random() < 0.7 else 1 - t for t in yt]
        tp = sum(a and b for a, b in zip(yt, yp))
        if tp >= 1 and sum(yt) >= 1 and sum(yp) >= 1:
            break
    return P(f"A binary {pos} detector made predictions `y_pred` for {n} samples with true labels "
             f"`y_true` (1 = {pos}). Compute **precision, recall and F1** for the positive class and "
             f"return them as a tuple `(precision, recall, f1)`. Use boolean masks, no loops.",
             f"y_true = {arr(yt)}\ny_pred = {arr(yp)}",
             "tp = np.sum((y_pred == 1) & (y_true == 1))\nprecision = tp / np.sum(y_pred == 1)\n"
             "recall = tp / np.sum(y_true == 1)\nf1 = 2 * precision * recall / (precision + recall)\n"
             "result = (precision, recall, f1)", no_loops=True)


@template(S, "hard", "1.14")
def cosine_top_k(rng):
    th = T.theme(rng)
    n, d, k = rng.randint(6, 9), rng.randint(4, 6), rng.randint(2, 3)
    names = T.items(rng, th, n)
    s = seeded(rng)
    return P(f"`E` holds a {d}-dimensional embedding for each of {n} {th.plural} (`{th.var}`), and `q` "
             f"is the embedding of a search query. Rank the {th.plural} by **cosine similarity** to `q` "
             f"and return the names of the {k} most similar, most similar first. No loops.",
             f"{th.var} = {arr(names)}\ng = np.random.default_rng({s})\nE = g.normal(size=({n}, {d})).round(2)\n"
             f"q = g.normal(size={d}).round(2)",
             f"sims = E @ q / (np.linalg.norm(E, axis=1) * np.linalg.norm(q))\n"
             f"result = {th.var}[np.argsort(sims)[::-1][:{k}]]", no_loops=True)


@template(S, "hard", "1.18")
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


@template(S, "hard", "1.18")
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


@template(S, "hard", "1.09")
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
             f"boxes that don't overlap have IoU 0.", f"gt = {arr(gt)}\nboxes = {mat(boxes)}",
             "ix1 = np.maximum(boxes[:, 0], gt[0])\niy1 = np.maximum(boxes[:, 1], gt[1])\n"
             "ix2 = np.minimum(boxes[:, 2], gt[2])\niy2 = np.minimum(boxes[:, 3], gt[3])\n"
             "inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)\n"
             "area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])\n"
             "gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])\nresult = inter / (area + gt_area - inter)",
             no_loops=True)


@template(S, "hard", "1.18")
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


@template(S, "hard", "1.09")
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


@template(S, "hard", "1.18")
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


@template(S, "hard", "1.18")
def cross_entropy(rng):
    what, classes = T.labels(rng, min_classes=3)
    k = min(len(classes), rng.randint(3, 5))
    n = rng.randint(3, 6)
    z = [[round(rng.gauss(0, 2), 2) for _ in range(k)] for _ in range(n)]
    big = rng.randrange(n)
    z[big] = [round(x + rng.randint(800, 1000), 2) for x in z[big]]
    y = [rng.randrange(k) for _ in range(n)]
    return P(f"`logits` (shape `({n}, {k})`) are a {what} classifier's raw scores and `y` holds the true "
             f"class index of each sample. Compute the **mean cross-entropy loss** "
             f"`mean(logsumexp(logits[i]) - logits[i, y[i]])`. One row has huge logits, so use the "
             f"max trick for log-sum-exp. No loops (pick `logits[i, y[i]]` with `np.arange(n)`).",
             f"logits = {mat(z)}\ny = {arr(y)}",
             "mx = logits.max(axis=1, keepdims=True)\nlse = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))\n"
             "result = np.mean(lse - logits[np.arange(len(y)), y])", no_loops=True)


@template(S, "hard", "1.13")
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


# ================================================================= CHALLENGE
# Full application problems (~Codeforces 1800). The visible setup is small; hidden
# tests are large, so quadratic solutions time out. Any Python is allowed.

def _rand(var: str, s: int, lo: int, hi: int, n: int) -> str:
    return f"{var} = np.random.default_rng({s}).integers({lo}, {hi}, size={n})"


def challenge(rng, title, story, setup, solution, tests, constraints, answer_desc="") -> Problem:
    sample = solve(setup, solution)
    shown = sample.tolist() if hasattr(sample, "tolist") else sample
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden "
             f"tests up to these limits (time limit a few seconds), so aim for O(n) or O(n log n).\n\n"
             f"For the data shown, the answer is `{shown!r}`. Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], timeout=12.0)


@template(S, "challenge", "1.18")
def longest_budget_window(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(1, 30) for _ in range(n)]
    S_ = rng.randint(25, 70)
    var, story = rng.choice([
        ("durations", "`durations[i]` is the length in seconds of the i-th audio clip, in recording order. "
                      "A transcription job takes one **contiguous** run of clips whose total duration is at most `S`."),
        ("tokens", "`tokens[i]` is the token count of the i-th chat message, in order. A context window can "
                   "hold one **contiguous** run of messages whose total token count is at most `S`."),
        ("frames_mb", "`frames_mb[i]` is the size in MB of the i-th video frame. A GPU batch holds one "
                      "**contiguous** run of frames whose total size is at most `S` MB."),
    ])
    sol = (f"best = left = total = 0\nvals = {var}.tolist()\nfor right, x in enumerate(vals):\n"
           "    total += x\n    while total > S:\n        total -= vals[left]\n        left += 1\n"
           "    best = max(best, right - left + 1)\nresult = best")
    tests = [_rand(var, seeded(rng), 0, 1000, 20000) + "\nS = 250000",
             _rand(var, seeded(rng), 1, 50, 20000) + "\nS = 5000",
             f"{var} = np.array([100, 200, 300])\nS = 50",
             f"{var} = np.array([0, 0, 0, 0])\nS = 0"]
    return challenge(rng, "Longest batch under budget",
                     story + " What is the **maximum number of items** one job can take? (0 if none fit.)",
                     f"{var} = {arr(vals)}\nS = {S_}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10³, 0 ≤ S ≤ 10⁹")


@template(S, "challenge", "1.18")
def count_subarrays_sum_k(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(-4, 5) for _ in range(n)]
    K = rng.randint(-2, 6)
    var, story = rng.choice([
        ("reward", "`reward[i]` is the reward an RL agent received at step i (can be negative)."),
        ("delta", "`delta[i]` is the day-over-day change in a model's leaderboard rank points (can be negative)."),
        ("balance", "`balance[i]` is the net GPU-hours a team gained (+) or spent (−) on day i."),
    ])
    sol = (f"from collections import defaultdict\nseen = defaultdict(int)\nseen[0] = 1\nprefix = count = 0\n"
           f"for x in {var}.tolist():\n    prefix += x\n    count += seen[prefix - K]\n    seen[prefix] += 1\n"
           "result = count")
    tests = [_rand(var, seeded(rng), -3, 4, 20000) + "\nK = 0",
             _rand(var, seeded(rng), -50, 51, 20000) + "\nK = 37",
             f"{var} = np.zeros(2000, dtype=int)\nK = 0",
             f"{var} = np.array([5])\nK = 5"]
    return challenge(rng, "Windows with an exact total",
                     story + " Count the **contiguous subarrays** (windows of length ≥ 1) whose sum is exactly `K`.",
                     f"{var} = {arr(vals)}\nK = {K}", sol, tests,
                     "1 ≤ n ≤ 2·10⁴, |values| ≤ 50, |K| ≤ 10⁶")


@template(S, "challenge", "1.18")
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
    tests = [_rand(var, seeded(rng), 1, 1001, 20000) + "\nD = 37",
             _rand(var, seeded(rng), 1, 10, 20000) + "\nD = 20000",
             f"{var} = np.array([7, 2, 5, 10, 8])\nD = 1",
             f"{var} = np.array([4])\nD = 1"]
    return challenge(rng, "Balance the load",
                     story + " A part's load is the sum of its values. **Minimize the largest load** and "
                     "return that minimum.", f"{var} = {arr(vals)}\nD = {D}", sol, tests,
                     "1 ≤ D ≤ n ≤ 2·10⁴, 1 ≤ values ≤ 10³")


@template(S, "challenge", "1.18")
def wait_for_higher(rng):
    n = rng.randint(8, 12)
    vals = [rng.randint(30, 60) for _ in range(n)]
    var, story = rng.choice([
        ("temp_c", "`temp_c[i]` is a GPU's temperature at minute i."),
        ("acc", "`acc[i]` is a model's validation accuracy (%) after epoch i."),
        ("price", "`price[i]` is the spot price of a cloud GPU on day i."),
    ])
    sol = (f"vals = {var}.tolist()\nwait = [0] * len(vals)\nstack = []\nfor i, x in enumerate(vals):\n"
           "    while stack and vals[stack[-1]] < x:\n        j = stack.pop()\n        wait[j] = i - j\n"
           "    stack.append(i)\nresult = np.array(wait)")
    tests = [_rand(var, seeded(rng), 0, 100, 20000),
             f"{var} = np.arange(20000, 0, -1)",
             f"{var} = np.full(20000, 7)",
             f"{var} = np.array([1, 2, 3])"]
    return challenge(rng, "How long until it's higher?",
                     story + " For every i, find how many steps you must wait after i to see a **strictly "
                     "higher** value (0 if it never happens). Return a NumPy int array of length n.",
                     f"{var} = {arr(vals)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10⁵",
                     " (a NumPy array)")


@template(S, "challenge", "1.18")
def merge_segments(rng):
    n = rng.randint(6, 9)
    segs = []
    for _ in range(n):
        a = rng.randint(0, 60)
        segs.append([a, a + rng.randint(1, 12)])
    story = rng.choice([
        "A voice-activity detector and a diarization model produced speech segments `[start, end)` in "
        "seconds, as rows of `segs`, in no particular order and possibly overlapping.",
        "Several annotators marked the time ranges `[start, end)` (in frames) where an object is visible "
        "in a video; the rows of `segs` are unsorted and may overlap.",
    ])
    sol = ("pairs = sorted(map(tuple, segs.tolist()))\nmerged = []\nfor a, b in pairs:\n"
           "    if merged and a <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], b)\n"
           "    else:\n        merged.append([a, b])\nresult = (len(merged), sum(b - a for a, b in merged))")
    def big(s, n_, span):
        return (f"g = np.random.default_rng({s})\nstarts = g.integers(0, {span}, size={n_})\n"
                f"segs = np.stack([starts, starts + g.integers(1, 50, size={n_})], axis=1)")
    tests = [big(seeded(rng), 20000, 10**6), big(seeded(rng), 20000, 5000),
             "segs = np.array([[0, 5], [5, 9]])", "segs = np.array([[3, 4]])"]
    return challenge(rng, "Merge the segments",
                     story + " Segments that overlap **or touch** (one ends exactly where the next starts) "
                     "merge into one. Return a tuple `(number_of_merged_segments, total_covered_length)`.",
                     f"segs = {mat(segs)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ start < end ≤ 10⁶")


@template(S, "challenge", "1.13")
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
    tests = [_rand(var, seeded(rng), 1, 10**6, 20000) + "\nT = 10**6",
             _rand(var, seeded(rng), 1, 100, 20000) + "\nT = 150",
             f"{var} = np.array([5])\nT = 100",
             f"{var} = np.array([9, 9, 9])\nT = 17"]
    return challenge(rng, "Pairs that fit",
                     story + " Count the **pairs** `(i, j)` with `i < j` that fit.",
                     f"{var} = {arr(vals)}\nT = {T_}", sol, tests, "1 ≤ n ≤ 2·10⁴, 1 ≤ values, T ≤ 2·10⁶")


@template(S, "challenge", "1.13")
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
    tests = [_rand(var, seeded(rng), 0, 10**6, 20000), _rand(var, seeded(rng), 0, 5, 20000),
             f"{var} = np.arange(20000, 0, -1)", f"{var} = np.array([3])"]
    return challenge(rng, "Fewest drops to restore order",
                     story + " Find the **minimum number of entries to delete** so that the remaining "
                     "sequence is **non-decreasing** (equal neighbours are fine).",
                     f"{var} = {arr(vals)}", sol, tests, "1 ≤ n ≤ 2·10⁴, 0 ≤ values ≤ 10⁶")


@template(S, "challenge", "1.11")
def best_window_length_range(rng):
    n = rng.randint(9, 13)
    vals = [rng.randint(-6, 8) for _ in range(n)]
    lo = rng.randint(1, 3)
    hi = rng.randint(lo + 1, min(n, lo + 4))
    var, story = rng.choice([
        ("gain", "`gain[i]` is the accuracy gain (can be negative) from including the i-th data shard."),
        ("reward", "`reward[i]` is the reward of an RL agent at step i (can be negative)."),
    ])
    sol = (f"from collections import deque\nvals = {var}.tolist()\nprefix = [0]\nfor x in vals:\n"
           "    prefix.append(prefix[-1] + x)\nbest = None\ndq = deque()\n"
           "for j in range(L, len(vals) + 1):\n    i = j - L\n"
           "    while dq and prefix[dq[-1]] >= prefix[i]:\n        dq.pop()\n    dq.append(i)\n"
           "    while dq[0] < j - R:\n        dq.popleft()\n    cand = prefix[j] - prefix[dq[0]]\n"
           "    best = cand if best is None else max(best, cand)\nresult = best")
    tests = [_rand(var, seeded(rng), -100, 101, 20000) + "\nL, R = 50, 5000",
             _rand(var, seeded(rng), -5, 3, 20000) + "\nL, R = 1, 20000",
             f"{var} = np.array([-3, -1, -2])\nL, R = 2, 3",
             f"{var} = np.array([4])\nL, R = 1, 1"]
    return challenge(rng, "Best window with a length limit",
                     story + " Choose one **contiguous** window whose length is between `L` and `R` "
                     "(inclusive) and maximize its total. Return that maximum total.",
                     f"{var} = {arr(vals)}\nL, R = {lo}, {hi}", sol, tests,
                     "1 ≤ L ≤ R ≤ n ≤ 2·10⁴, |values| ≤ 10³")

"""Helpers shared by the NumPy module templates."""

from __future__ import annotations

import math
import os
import random
import tempfile

from ... import themes as T
from ...core import Problem, template as _template

S = "numpy"
HEADER = "import numpy as np\n"
STDOUT = "<stdout>"

SAMPLE_NOUNS = ("audio clip", "image", "utterance", "product review", "email", "video frame", "support ticket")


def template(module: str, tier: str):
    return _template(S, module, tier)


# ---------------------------------------------------------------- literals

def lit(x) -> str:
    """Python literal for setup code (NaN/inf -> np.nan/np.inf)."""
    if isinstance(x, float) and math.isnan(x):
        return "np.nan"
    if isinstance(x, float) and math.isinf(x):
        return "np.inf" if x > 0 else "-np.inf"
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


# ---------------------------------------------------------------- themed data

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


def feature_matrix(rng: random.Random, n_lo: int = 5, n_hi: int = 8):
    """(theme, 3 measures, row names, X) with one non-constant column per measure."""
    th = T.theme(rng)
    ms = rng.sample(th.measures, min(3, len(th.measures)))
    n = rng.randint(n_lo, n_hi)
    cols = []
    for m in ms:
        col = T.values(rng, m, n)
        while len(set(col)) < 2:
            col = T.values(rng, m, n)
        cols.append(col)
    return th, ms, T.items(rng, th, n), [list(r) for r in zip(*cols)]


def an(word: str, cap: bool = False) -> str:
    """'a word' / 'an word' (by first letter, good enough for our vocabulary)."""
    art = "an" if word[:1].lower() in "aeiou" else "a"
    return f"{art.capitalize() if cap else art} {word}"


def about(th: T.Theme, m: T.Measure, n: int, var: str) -> str:
    return f"`{var}` holds the {m.label} of {n} {th.plural}"


def seeded(rng: random.Random) -> int:
    return rng.randrange(1, 10_000)


def floats(rng: random.Random, n: int, lo: float = -3, hi: float = 3, nd: int = 2) -> list[float]:
    return [round(rng.uniform(lo, hi), nd) for _ in range(n)]


def solve(setup: str, solution: str):
    """Run a reference solution (to print sample answers in prompts), in a throwaway directory."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="tl-solve-") as tmp:
        os.chdir(tmp)
        try:
            ns: dict = {}
            exec(HEADER + setup, ns)
            exec(solution, ns)
            return ns["result"]
        finally:
            os.chdir(old)


# ---------------------------------------------------------------- challenge problems
# Full application problems (~Codeforces 1800). The visible setup is small; hidden
# tests are large, so quadratic solutions time out. Any Python is allowed.

def rand_ints(var: str, s: int, lo: int, hi: int, n: int) -> str:
    return f"{var} = np.random.default_rng({s}).integers({lo}, {hi}, size={n})"


def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              answer_desc: str = "", speed: str = "O(n) or O(n log n)") -> Problem:
    sample = solve(setup, solution)
    shown = sample.tolist() if hasattr(sample, "tolist") else sample
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden "
             f"tests up to these limits (time limit a few seconds), so aim for {speed}.\n\n"
             f"For the data shown, the answer is `{shown!r}`. Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], timeout=12.0)

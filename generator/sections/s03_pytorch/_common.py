"""Helpers shared by the PyTorch module templates.

In the browser `torch` is torch-lite (generator/shims/torch): NumPy-backed, CPU-only, with the
same semantics for everything these templates use. The grader re-seeds torch (manual_seed(0))
before every run, so a reference solution and the user's code see the same random numbers.
"""

from __future__ import annotations

import os
import random
import tempfile

from ... import themes as T
from ...core import Problem, template as _template
from ..s01_numpy._common import an, floats, lit, seeded  # noqa: F401  (re-exported)

S = "pytorch"
HEADER = "import math\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
STDOUT = "<stdout>"


def template(module: str, tier: str):
    return _template(S, module, tier)


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    return Problem(prompt=prompt.strip(), setup=(HEADER + setup.strip()).strip(),
                   solution=solution.strip(), **kw)


def tt(x, dtype: str = "") -> str:
    """torch.tensor literal."""
    return f"torch.tensor({lit(x)}{', dtype=torch.' + dtype if dtype else ''})"


def ftt(x) -> str:
    """Float tensor literal: ints are written as floats so the dtype is float32."""
    def f(v):
        if isinstance(v, (list, tuple)):
            return [f(u) for u in v]
        return float(v)
    return tt(f(x))


def rows_tt(rows, dtype: str = "") -> str:
    body = "".join(f"    {lit(r)},\n" for r in rows)
    return f"torch.tensor([\n{body}]{', dtype=torch.' + dtype if dtype else ''})"


def frows_tt(rows) -> str:
    return rows_tt([[float(v) for v in r] for r in rows])


def shp(shape) -> str:
    return "(" + ", ".join(map(str, shape)) + ("," if len(shape) == 1 else "") + ")"


def seed_line(rng: random.Random) -> str:
    return f"torch.manual_seed({seeded(rng)})"


def classes(rng: random.Random, lo: int = 3, hi: int = 5) -> tuple[str, list[str]]:
    name, labels = T.labels(rng, min_classes=lo)
    k = rng.randint(lo, min(hi, len(labels)))
    return name, list(labels[:k])


def solve(setup: str, solution: str):
    """Run a reference solution (to show sample answers in prompts), in a throwaway directory."""
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


def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              answer_desc: str = "", speed: str = "O(n log n) or better") -> Problem:
    sample = solve(setup, solution)
    shown = sample.tolist() if hasattr(sample, "tolist") else sample
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden "
             f"tests up to these limits (time limit a few seconds), so aim for {speed}.\n\n"
             f"For the data shown, the answer is `{shown!r}`. Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], timeout=12.0)


def big_ints(var: str, s: int, lo: int, hi: int, n: int) -> str:
    """A large random int64 tensor for hidden tests (generated with NumPy, so it's fast)."""
    return f"{var} = torch.from_numpy(np.random.default_rng({s}).integers({lo}, {hi}, size={n}).astype(np.int64))"


def pick_theme(rng: random.Random, n: int) -> tuple[T.Theme, list[str]]:
    th = T.theme(rng, min_items=n)
    return th, T.items(rng, th, n)


# ---------------------------------------------------------------- catalogs
# A catalog template practises many related APIs: shared random (literal, so diff-testable) data, then ONE
# item chosen with core.pick(). Item: (api, prompt, solution) or (api, prompt, solution, {options}); options
# may add `setup` lines, `require` (default [api]), `tol`, `check_dtype`, `answer`, `post`.

def catalog(rng: random.Random, items: list, setup: str, intro: str, tol: float = 1e-5, **kw) -> Problem:
    from ...core import pick
    item = pick(rng, items)
    api, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", [api] if api else [])
    extra = opts.pop("setup", "")
    opts.setdefault("tol", tol)
    return P(f"{intro} {prompt}".strip(), (setup + ("\n" + extra if extra else "")).strip(), sol, require=req,
             **{**kw, **opts})


def fv(rng: random.Random, n: int, lo: float = -3, hi: float = 3, nd: int = 1, nonzero: bool = False) -> list[float]:
    out = [round(rng.uniform(lo, hi), nd) for _ in range(n)]
    return [v if v or not nonzero else 0.5 for v in out]


def fm(rng: random.Random, r: int, c: int, lo: float = -3, hi: float = 3, nd: int = 1) -> list[list[float]]:
    return [fv(rng, c, lo, hi, nd) for _ in range(r)]

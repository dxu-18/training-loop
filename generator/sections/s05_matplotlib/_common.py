"""Helpers shared by the Matplotlib templates (COMP_GUIDE §5).

Problems run in Pyodide's Matplotlib 3.10 (headless: the grader forces the Agg backend). The setup imports NumPy,
`matplotlib as mpl` and `matplotlib.pyplot as plt` and builds the data; the learner draws. Plots are graded by
**inspecting the Matplotlib objects** after the answer ran (`post` code), never pixels: `post` code gets
`generator.figcheck` as `fc` (lines, bars, scatter offsets, labels, ticks, legend texts, images, patches ...).
Pick only what the prompt asks for, so a figure that is right in every requested respect passes even if the
learner styled something else differently; say in the prompt exactly what must be true.

The grader closes every figure and resets rcParams (`mpl.rcdefaults()`) after each run, so the reference and the
answer both start from a clean slate. Keep problems fast (Pyodide is a few times slower than CPython): small
figures, few points, `savefig` only when the task is about saving, tiny GIFs.
"""

from __future__ import annotations

import os
import random
import tempfile

from ... import themes as T
from ...core import Problem, pick, template as _template
from ..s01_numpy._common import an, arr, floats, lit, mat, seeded  # noqa: F401  (re-exported)

S = "matplotlib"
HEADER = "import numpy as np\nimport matplotlib as mpl\nimport matplotlib.pyplot as plt\n"
FC = "import generator.figcheck as fc\n"
STDOUT = "<stdout>"


def template(module: str, tier: str):
    return _template(S, module, tier)


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    """A problem. `post` (if given) is check code with `fc` (generator.figcheck) imported; its value is graded."""
    kw.setdefault("timeout", 15.0)
    if kw.get("post"):
        kw["post"] = FC + kw["post"]
    if kw.get("require"):   # the grader sees names and attributes: "plt.subplots" is checked as "subplots"
        kw["require"] = ["|".join(a.rsplit(".", 1)[-1] for a in r.split("|")) for r in kw["require"]]
    return Problem(prompt=prompt.strip(), setup=(HEADER + setup.strip()).strip(), solution=solution.strip(), **kw)


def solve(setup: str, solution: str, var: str = "result"):
    """Run a reference solution (to show a sample answer in a prompt), in a throwaway directory."""
    import warnings
    from ...grader import numpy_32bit_compat
    numpy_32bit_compat()
    old = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="tl-solve-") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        os.chdir(tmp)
        try:
            ns: dict = {}
            exec(HEADER + setup, ns)
            exec(solution, ns)
            return ns[var]
        finally:
            os.chdir(old)
            import matplotlib.pyplot as plt
            plt.close("all")


def fmt(v, nd: int = 4) -> str:
    if isinstance(v, float):
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return str(v)


# ---------------------------------------------------------------- catalogs
# One template, many related tasks on shared data: ONE item per problem, chosen with core.pick() (the tests force
# every index). Item: (api, prompt, solution[, {options}]); options may add `setup` lines, `require` (default [api];
# [] for none), `post` (check code; `fc` available), `tol`, `answer`, `timeout`.

def catalog(rng: random.Random, items: list, setup: str, intro: str = "", **kw) -> Problem:
    item = pick(rng, items)
    api, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", kw.pop("require", None) or ([api] if api else []))
    extra = opts.pop("setup", "")
    return P(f"{intro} {prompt}".strip(), (setup + ("\n" + extra if extra else "")).strip(), sol, require=req,
             **{**kw, **opts})


# ---------------------------------------------------------------- challenge problems
# Algorithmic problems (~Codeforces 1800) about what plotting computes: tick placement, histogram binning, box-plot
# statistics, colormap / norm lookups, line decimation, point-in-polygon, box overlap ... The visible data is small;
# hidden tests are large. Brute forces live in tests/mpl_brutes/ (often Matplotlib itself on the small inputs).

def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              answer_desc: str = "", speed: str = "O(n log n) or better", **kw) -> Problem:
    sample = solve(setup, solution)
    shown = sample.tolist() if hasattr(sample, "tolist") else sample
    if isinstance(shown, float):
        shown = round(shown, 6)
    kw.setdefault("timeout", 20.0)
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden tests up to these "
             f"limits (time limit a few seconds), so aim for {speed}.\n\nFor the data shown, the answer is "
             f"`{shown!r}`. Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], **kw)


# ---------------------------------------------------------------- themed data (setup code + a description)

def curves(rng: random.Random, n=(15, 40), acc: bool = False) -> tuple[str, str, int]:
    """Training curves: `epochs` (1..n), `train_loss`, `val_loss` (val starts overfitting late), maybe `val_acc`."""
    n_ = rng.randint(*n)
    s = seeded(rng)
    tau, over = rng.uniform(4, 10), rng.randint(n_ // 2, n_ - 3)
    code = (f"g = np.random.default_rng({s})\nepochs = np.arange(1, {n_ + 1})\n"
            f"train_loss = np.round(2.0 * np.exp(-epochs / {tau:.1f}) + 0.05 * g.random({n_}), 4)\n"
            f"val_loss = np.round(train_loss + 0.1 + 0.004 * np.clip(epochs - {over}, 0, None) ** 1.5, 4)")
    desc = f"`epochs` (1 … {n_}) with `train_loss` and `val_loss` from a training run"
    if acc:
        code += f"\nval_acc = np.round(np.clip(0.98 - 0.4 * val_loss + 0.01 * g.normal(size={n_}), 0, 1), 4)"
        desc += " (and `val_acc`, the validation accuracy)"
    return code, desc, n_


def categories(rng: random.Random, k=(3, 6), positive: bool = True) -> tuple[str, str, list, list, T.Theme, T.Measure]:
    """`names` (a list of k themed item names) and `values` (one measure each)."""
    th = T.theme(rng, min_items=k[1])
    ms = [m for m in th.measures if not positive or m.lo >= 0] or list(th.measures)
    m = rng.choice(ms)
    k_ = rng.randint(*k)
    names = T.items(rng, th, k_)
    vals = T.values(rng, m, k_)
    code = f"names = {lit(list(names))}\nvalues = {lit(list(vals))}"
    return code, f"`names` holds {k_} {th.plural} and `values` their {m.label}", list(names), list(vals), th, m


def classes(rng: random.Random, k=(2, 4), n=(30, 80)) -> tuple[str, str, int, list[str]]:
    """2-D points in k clusters: `pts` (n × 2), `labels` (ints 0..k-1) and `class_names`."""
    name, labs = T.labels(rng, min_classes=k[1])
    k_ = rng.randint(*k)
    labs = list(labs[:k_])
    n_ = rng.randint(*n)
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nlabels = g.integers(0, {k_}, {n_})\n"
            f"centers = g.uniform(-4, 4, ({k_}, 2))\npts = np.round(centers[labels] + g.normal(0, 0.8, ({n_}, 2)), 3)\n"
            f"class_names = {lit(labs)}")
    return code, (f"`pts` holds {n_} 2-D points of {k_} {name} classes, `labels` their class (0 … {k_ - 1}) and "
                  f"`class_names` the class names"), k_, labs


def signal(rng: random.Random, n=(20, 60)) -> tuple[str, str, int]:
    """`x` (n points) and a noisy periodic `y`."""
    n_ = rng.randint(*n)
    s = seeded(rng)
    f = rng.choice([0.5, 1.0, 1.5, 2.0])
    code = (f"g = np.random.default_rng({s})\nx = np.linspace(0, 10, {n_})\n"
            f"y = np.round(np.sin({f} * x) + 0.1 * g.normal(size={n_}), 4)")
    return code, f"`x` ({n_} points from 0 to 10) and a noisy wave `y`", n_


def matrix(rng: random.Random, kind: str = "random", n=(4, 7)) -> tuple[str, str, int]:
    """A small square matrix `M`: 'random' in [0, 1), 'corr' (a correlation matrix), 'confusion' (counts)."""
    n_ = rng.randint(*n)
    s = seeded(rng)
    if kind == "corr":
        return (f"g = np.random.default_rng({s})\nM = np.round(np.corrcoef(g.normal(size=({n_}, 40)) + g.normal(size=40)), 3)",
                f"`M` is a {n_} × {n_} correlation matrix", n_)
    if kind == "confusion":
        return (f"g = np.random.default_rng({s})\nM = g.integers(0, 8, ({n_}, {n_})) + np.diag(g.integers(30, 60, {n_}))",
                f"`M` is a {n_} × {n_} confusion matrix (rows: true class, columns: predicted)", n_)
    return (f"M = np.round(np.random.default_rng({s}).random(({n_}, {n_})), 3)", f"`M` is a {n_} × {n_} matrix of "
            f"values in [0, 1)", n_)

"""Helpers shared by the pandas module templates."""

from __future__ import annotations

import keyword
import os
import random
import tempfile
from dataclasses import dataclass, field

from ... import themes as T
from ...core import Problem, template as _template
from ..s01_numpy._common import an, floats, lit, rand_ints, seeded  # noqa: F401  (re-exported)

S = "pandas"
HEADER = "import numpy as np\nimport pandas as pd\n"
STDOUT = "<stdout>"

GROUPS = (("team", ("red", "blue", "green")), ("region", ("us-east", "eu-west", "ap-south")),
          ("tier", ("free", "pro", "enterprise")), ("split", ("train", "val", "test")),
          ("source", ("web", "mobile", "api")), ("shift", ("day", "night")),
          ("vendor", ("acme", "globex", "initech")))
DATE_COLS = ("joined", "released", "created", "updated", "logged")


def template(module: str, tier: str):
    return _template(S, module, tier)


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    return Problem(prompt=prompt.strip(), setup=(HEADER + setup.strip()).strip(),
                   solution=solution.strip(), **kw)


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


def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              answer_desc: str = "", speed: str = "O(n log n) or better") -> Problem:
    sample = solve(setup, solution)
    if hasattr(sample, "to_string"):  # DataFrame / Series: show it as a table
        shown = f"\n\n```\n{sample.to_string()}\n```\n\n"
    else:
        shown = f"`{(sample.tolist() if hasattr(sample, 'tolist') else sample)!r}`. "
    return P(f"**{title}.** {story}\n\n**Constraints:** {constraints}. Your code also runs on hidden "
             f"tests up to these limits (time limit a few seconds), so aim for {speed}.\n\n"
             f"For the data shown, the answer is {shown}Store your answer in `result`{answer_desc}.",
             setup, solution, tests=[HEADER + t for t in tests], timeout=12.0)


# ---------------------------------------------------------------- themed frames

def key_col(th: T.Theme) -> str:
    """Column name for a theme's entity names: "model", "lab", "pet", ..."""
    words = th.noun.lower().replace("-", " ").split()
    word = words[-1]
    if keyword.iskeyword(word):
        word = words[0]
    return T.ident(word)


@dataclass
class Frame:
    th: T.Theme
    key: str                        # entity-name column
    group: str | None               # categorical column
    levels: tuple                   # its possible values
    nums: list                      # Measures; column name = m.var
    date: str | None
    cols: dict = field(default_factory=dict)   # column -> python values, in order

    @property
    def n(self) -> int:
        return len(self.cols[self.key])

    @property
    def num(self) -> list[str]:
        return [m.var for m in self.nums]

    def code(self, var: str = "df") -> str:
        lines = []
        for c, vals in self.cols.items():
            v = f"pd.to_datetime({lit(vals)})" if c == self.date else lit(vals)
            lines.append(f'    "{c}": {v},')
        return f"{var} = pd.DataFrame({{\n" + "\n".join(lines) + "\n})"

    def describe(self, var: str = "df") -> str:
        parts = [f"`{self.key}`"]
        if self.group:
            parts.append(f"`{self.group}`")
        parts += [f"`{m.var}` ({m.name}, {m.unit})" if m.unit else f"`{m.var}` ({m.name})" for m in self.nums]
        if self.date:
            parts.append(f"`{self.date}` (a date)")
        return f"`{var}` has one row per {self.th.noun} with columns " + ", ".join(parts)

    def label_of(self, col: str) -> str:
        m = next((m for m in self.nums if m.var == col), None)
        return m.name if m else col


def frame(rng: random.Random, n=(6, 9), nums: int = 2, group: bool = True, date: bool = False,
          nan: int = 0, distinct: bool = False) -> Frame:
    th = T.theme(rng)
    n_ = rng.randint(n[0], min(n[1], len(th.items)))
    ms = rng.sample(th.measures, min(nums, len(th.measures)))
    f = Frame(th, key_col(th), None, (), ms, None)
    f.cols[f.key] = T.items(rng, th, n_)
    if group:
        f.group, f.levels = rng.choice([g for g in GROUPS if g[0] != f.key and g[0] not in [m.var for m in ms]])
        vals = [rng.choice(f.levels) for _ in range(n_)]
        vals[0], vals[1] = f.levels[0], f.levels[1]  # at least two groups
        rng.shuffle(vals)
        f.cols[f.group] = vals
    for m in ms:
        f.cols[m.var] = T.values(rng, m, n_, distinct=distinct)
    if nan:
        col = ms[0].var
        vals = [float(v) for v in f.cols[col]]
        for i in rng.sample(range(n_), min(nan, n_ - 2)):
            vals[i] = float("nan")
        f.cols[col] = vals
    if date:
        f.date = rng.choice(DATE_COLS)
        f.cols[f.date] = sorted(f"20{rng.randint(21, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                                for _ in range(n_))
        rng.shuffle(f.cols[f.date])
    return f


def threshold(rng: random.Random, vals: list, m) -> float:
    return T.threshold(rng, [v for v in vals if v == v], m)  # skip NaN

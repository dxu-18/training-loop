"""Problem type, difficulty schedule and template registry.

A *template* is a function ``f(rng) -> Problem`` registered for one section and
one tier. ``generate(section, tier, seed)`` picks a template uniformly at random,
builds the problem, and attaches an example: a second, independent instance of
the same template, shown together with its solution.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable

TIERS = ("lite", "easy", "medium", "hard", "challenge")

# (last question number, tier) for questions 1..250
SCHEDULE = ((50, "lite"), (120, "easy"), (200, "medium"), (245, "hard"), (250, "challenge"))

# after question 250, tiers keep the same share they had in 1..250
WEIGHTS = {"lite": 50, "easy": 70, "medium": 80, "hard": 45, "challenge": 5}

EXAMPLE_UNTIL = 10    # questions 1..10 show the example automatically
MEDAL_AT = 250        # solved count that earns the section medal
REVEAL_AFTER = 250    # answers can be revealed for questions after this number


def tier_for(n: int, seed: int | None = None) -> str:
    """Difficulty tier of question number ``n`` (1-based)."""
    for last, tier in SCHEDULE:
        if n <= last:
            return tier
    rng = random.Random(seed)
    return rng.choices(TIERS, weights=[WEIGHTS[t] for t in TIERS])[0]


@dataclass
class Problem:
    prompt: str                           # English task; `code` and ```blocks``` allowed
    setup: str                            # run before the user's code and shown as "Given"
    solution: str                         # reference solution, shown as the example / revealed answer
    answer: list[str] = field(default_factory=list)   # variables to compare; [] -> `result` or last expression
    tests: list[str] = field(default_factory=list)    # extra hidden setups the code must also pass
    require: list[str] = field(default_factory=list)  # names that must appear in the code; "a|b" = either
    no_loops: bool = False                # forbid for/while/comprehensions
    check_dtype: bool = False             # compare array dtypes too
    tol: float = 1e-6                     # relative/absolute tolerance for floats
    kind: str = "python"                  # python | sql | command | text
    lang: str = "python"                  # editor mode
    timeout: float = 8.0                  # seconds the whole grading run may take
    # filled in by generate()
    section: str = ""
    tier: str = ""
    template: str = ""
    module: str = ""
    seed: int = 0
    example: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Template:
    fn: Callable[[random.Random], Problem]
    section: str
    tier: str
    module: str   # COMP_GUIDE module number, e.g. "1.05"

    @property
    def name(self) -> str:
        return self.fn.__name__


REGISTRY: dict[str, dict[str, list[Template]]] = {}


def template(section: str, tier: str, module: str):
    """Decorator that registers a template function."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    def register(fn):
        REGISTRY.setdefault(section, {t: [] for t in TIERS})[tier].append(
            Template(fn, section, tier, module))
        return fn

    return register


def templates(section: str, tier: str | None = None) -> list[Template]:
    tiers = TIERS if tier is None else (tier,)
    return [t for tr in tiers for t in REGISTRY.get(section, {}).get(tr, [])]


def available_tier(section: str, tier: str) -> str:
    """``tier`` if the section has templates for it, else the nearest tier that does."""
    order = sorted(TIERS, key=lambda t: abs(TIERS.index(t) - TIERS.index(tier)))
    for t in order:
        if templates(section, t):
            return t
    raise LookupError(f"section {section!r} has no templates")


def build(tpl: Template, seed: int) -> Problem:
    p = tpl.fn(random.Random(seed))
    p.section, p.tier, p.template, p.module, p.seed = tpl.section, tpl.tier, tpl.name, tpl.module, seed
    return p


def generate(section: str, tier: str, seed: int) -> dict:
    """A random problem for (section, tier), with an example attached."""
    tier = available_tier(section, tier)
    rng = random.Random(seed)
    tpl = rng.choice(templates(section, tier))
    problem = build(tpl, seed)
    ex = build(tpl, rng.randrange(2**31))
    problem.example = {"prompt": ex.prompt, "setup": ex.setup, "solution": ex.solution}
    return problem.to_dict()


def sections() -> list[str]:
    return list(REGISTRY)

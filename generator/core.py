"""Problem type, difficulty schedule and template registry.

A *template* is a function ``f(rng) -> Problem`` registered for one section
(e.g. "numpy"), one module of that section (a COMP_GUIDE subsection, e.g. "1.05")
and one tier. Every module is its own practice page with its own counter, so the
tier schedule below applies per module. ``generate(section, module, tier, seed)``
picks one of that cell's templates uniformly at random, builds the problem, and
attaches its Instructions: the functions it is meant to practise, with what they
do (generator/instructions.py).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Callable

TIERS = ("lite", "easy", "medium", "hard", "challenge")

# (last question number, tier) for questions 1..250 of a module
SCHEDULE = ((50, "lite"), (120, "easy"), (200, "medium"), (245, "hard"), (250, "challenge"))

# after question 250, tiers keep the same share they had in 1..250
WEIGHTS = {"lite": 50, "easy": 70, "medium": 80, "hard": 45, "challenge": 5}

INSTRUCTIONS_UNTIL = 10    # questions 1..10 show the Instructions open
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
    solution: str                         # reference solution, shown after a wrong answer / revealed answer
    answer: list[str] = field(default_factory=list)   # variables to compare; [] -> `result` or last expression
    tests: list[str] = field(default_factory=list)    # extra hidden setups the code must also pass
    require: list[str] = field(default_factory=list)  # names that must appear in the code; "a|b" = either
    no_loops: bool = False                # forbid for/while/comprehensions
    forbid: list[str] = field(default_factory=list)   # bash features that must not be used (see bashlite.features)
    unordered: bool = False               # bash: compare output lines as a set (order-free)
    check_dtype: bool = False             # compare array dtypes too
    post: str = ""                        # check code run after the user's; its value is graded
    tol: float = 1e-6                     # relative/absolute tolerance for floats
    kind: str = "python"                  # python | bash | sql | text
    lang: str = "python"                  # editor mode
    timeout: float = 8.0                  # seconds the whole grading run may take
    # filled in by generate()
    section: str = ""
    tier: str = ""
    template: str = ""
    module: str = ""
    seed: int = 0
    instructions: list = field(default_factory=list)   # [{name, sig, desc}] (generator/instructions.py)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Template:
    fn: Callable[[random.Random], Problem]
    section: str
    module: str   # COMP_GUIDE module number, e.g. "1.05"
    tier: str

    @property
    def name(self) -> str:
        return self.fn.__name__


# section -> module -> tier -> templates
REGISTRY: dict[str, dict[str, dict[str, list[Template]]]] = {}


def template(section: str, module: str, tier: str):
    """Decorator that registers a template function for one (section, module, tier) cell."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    def register(fn):
        cells = REGISTRY.setdefault(section, {}).setdefault(module, {t: [] for t in TIERS})
        cells[tier].append(Template(fn, section, module, tier))
        return fn

    return register


def sections() -> list[str]:
    return list(REGISTRY)


def modules(section: str) -> list[str]:
    return sorted(REGISTRY.get(section, {}), key=lambda m: tuple(int(x) for x in m.split(".")))


def templates(section: str, module: str | None = None, tier: str | None = None) -> list[Template]:
    mods = modules(section) if module is None else [module]
    tiers = TIERS if tier is None else (tier,)
    cells = REGISTRY.get(section, {})
    return [t for m in mods for tr in tiers for t in cells.get(m, {}).get(tr, [])]


def is_live(section: str, module: str) -> bool:
    """A module is playable once every tier has at least one template."""
    return all(templates(section, module, t) for t in TIERS)


def available_tier(section: str, module: str, tier: str) -> str:
    """``tier`` if the module has templates for it, else the nearest tier that does."""
    order = sorted(TIERS, key=lambda t: abs(TIERS.index(t) - TIERS.index(tier)))
    for t in order:
        if templates(section, module, t):
            return t
    raise LookupError(f"{section} {module} has no templates")


# ---------------------------------------------------------------- catalogs
# A catalog template practises many related APIs: it builds fresh random data, then picks ONE item
# (one API) with pick(). tests/test_catalogs.py forces every index in turn so every item is graded.

_FORCE_ITEM: list = [None]
_PICKED: list = []


def pick(rng: random.Random, items: list):
    """Choose the catalog item for this problem (uniformly; tests can force an index)."""
    _PICKED.append(len(items))
    if _FORCE_ITEM[0] is not None:
        return items[_FORCE_ITEM[0] % len(items)]
    return items[rng.randrange(len(items))]


def build(tpl: Template, seed: int) -> Problem:
    p = tpl.fn(random.Random(seed))
    p.section, p.module, p.tier, p.template, p.seed = tpl.section, tpl.module, tpl.tier, tpl.name, seed
    return p


def generate(section: str, module: str, tier: str, seed: int) -> dict:
    """A random problem for (section, module, tier), with its Instructions attached."""
    from .instructions import instructions
    tier = available_tier(section, module, tier)
    rng = random.Random(seed)
    tpl = rng.choice(templates(section, module, tier))
    problem = build(tpl, seed)
    problem.instructions = instructions(problem)
    return problem.to_dict()

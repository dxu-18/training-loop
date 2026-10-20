"""Random practice-question generator and grader for the COMP_GUIDE topics."""

from . import sections  # noqa: F401  (registers every template)
from .core import EXAMPLE_UNTIL, MEDAL_AT, REVEAL_AFTER, TIERS, generate, modules, templates, tier_for
from .grader import grade, run

__all__ = ["EXAMPLE_UNTIL", "MEDAL_AT", "REVEAL_AFTER", "TIERS", "generate", "grade", "modules", "run",
           "templates", "tier_for"]

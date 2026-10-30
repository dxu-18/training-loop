"""Random practice-question generator and grader for the COMP_GUIDE topics."""

import importlib.util as _ilu
import os as _os
import sys as _sys

if _ilu.find_spec("torch") is None:  # no PyTorch (e.g. in the browser): use the NumPy-backed torch-lite
    _sys.path.append(_os.path.join(_os.path.dirname(__file__), "shims"))

from . import sections  # noqa: E402,F401  (registers every template)
from .core import INSTRUCTIONS_UNTIL, MEDAL_AT, REVEAL_AFTER, TIERS, generate, modules, templates, tier_for  # noqa: E402
from .grader import grade, run  # noqa: E402

__all__ = ["INSTRUCTIONS_UNTIL", "MEDAL_AT", "REVEAL_AFTER", "TIERS", "generate", "grade", "modules", "run",
           "templates", "tier_for"]

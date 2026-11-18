"""COMP_GUIDE §5 Matplotlib: one file per subsection, m01_setup.py … m15_ml_recipes.py (imported in order)."""

import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name.startswith("m")):
    importlib.import_module(f"{__name__}.{_m}")

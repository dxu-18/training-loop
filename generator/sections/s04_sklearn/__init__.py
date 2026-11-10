"""COMP_GUIDE §4 scikit-learn: one file per subsection, m01_estimators.py … m20_custom.py (imported in order)."""

import importlib
import pkgutil

for _m in sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name.startswith("m")):
    importlib.import_module(f"{__name__}.{_m}")

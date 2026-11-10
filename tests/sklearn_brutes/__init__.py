"""Brute forces for the scikit-learn challenges (generator/sections/s04_sklearn), one file per module (m01.py ...).

Each file defines `BRUTE = {}` and registers `name(ns) -> expected result` functions with `@brute` (name = the
challenge template's function name). `ns` is the namespace after the problem's setup ran; return what `result`
should be, computed independently (naively, or with scikit-learn itself on the small inputs)."""

import importlib
import pkgutil

BRUTE: dict = {}


def brute(fn):
    BRUTE[fn.__name__] = fn
    return fn


def load() -> dict:
    for m in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{m.name}")
    return BRUTE

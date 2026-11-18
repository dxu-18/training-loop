"""Brute forces for the Matplotlib challenges (generator/sections/s05_matplotlib), one file per module (m01.py ...).

Each file defines `BRUTE = {}` and registers `name(ns) -> expected result` functions with `@brute` (name = the
challenge template's function name). `ns` is the namespace after the problem's setup ran; return what `result`
should be, computed independently (naively, or with Matplotlib itself on the small inputs)."""

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

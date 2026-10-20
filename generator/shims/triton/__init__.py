"""triton-lite: runs @triton.jit kernels on the CPU, one program instance at a time (like TRITON_INTERPRET=1).

Tensors passed to a kernel become pointers into their memory; tl.load / tl.store gather and scatter
blocks of elements, and an unmasked out-of-bounds access raises an error instead of corrupting memory."""

import functools
import itertools
import time

from . import language, testing  # noqa: F401
from .language import _launch

__version__ = "3.2.0+lite"


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, pre_hook=None):
        self.kwargs, self.num_warps, self.num_stages, self.num_ctas = dict(kwargs), num_warps, num_stages, num_ctas

    def __repr__(self):
        kw = ", ".join(f"{k}: {v}" for k, v in self.kwargs.items())
        return f"{kw}, num_warps: {self.num_warps}, num_stages: {self.num_stages}"


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: _launch(self.fn, grid, args, kwargs)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel; launch it with kernel[grid](...)")

    def run(self, *args, grid, **kwargs):
        return _launch(self.fn, grid, args, kwargs)


def jit(fn=None, **options):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


class Autotuner:
    def __init__(self, kernel, configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, warmup=None,
                 rep=None):
        self.kernel, self.configs, self.key = kernel, list(configs), list(key)
        self.cache = {}
        self.best_config = None

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            config = self.configs[0]  # the interpreter has no timing worth tuning on: use the first config
            self.best_config = config
            for k in config.kwargs:
                if k in kwargs:
                    raise ValueError(f"Conflicting meta-parameters: {k}. Make sure that you don't re-define auto-tuned symbols.")
            return _launch(self.kernel.fn, grid, args, {**kwargs, **config.kwargs})
        return run


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, warmup=None, rep=None, **kw):
    def deco(kernel):
        return Autotuner(kernel, configs, key, prune_configs_by, reset_to_zero, restore_value)
    return deco


def heuristics(values):
    def deco(kernel):
        class _H:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    import inspect
                    names = list(inspect.signature(kernel.fn).parameters)
                    meta = {**dict(zip(names, args)), **kwargs}
                    extra = {k: f(meta) for k, f in values.items()}
                    return _launch(kernel.fn, grid, args, {**kwargs, **extra})
                return run
        return _H()
    return deco


_ = (itertools, time)

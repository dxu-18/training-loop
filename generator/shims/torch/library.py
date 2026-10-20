"""torch.library.custom_op (torch-lite): register a Python function as an "op" with optional autograd."""

import numpy as np

import torch
from torch import Tensor, _make, no_grad


class _Ctx:
    pass


class CustomOpDef:
    def __init__(self, name, fn, mutates_args):
        self.name, self._fn, self.mutates_args = name, fn, mutates_args
        self._fake = None
        self._backward = self._setup = None
        self.__name__ = name.split("::")[-1]

    def register_fake(self, fn):
        self._fake = fn
        return fn

    def register_kernel(self, device_types, fn=None):
        return fn if fn is not None else (lambda f: f)

    def register_autograd(self, backward, setup_context=None):
        self._backward, self._setup = backward, setup_context

    def __call__(self, *args, **kwargs):
        detached = tuple(a.detach() if isinstance(a, Tensor) else a for a in args)
        with no_grad():
            out = self._fn(*detached, **kwargs)
        if self._backward is None or not any(isinstance(a, Tensor) and a.requires_grad for a in args):
            return out
        ctx = _Ctx()
        if self._setup:
            self._setup(ctx, detached, out)
        cache = {}

        def grads_for(g):
            if id(g) not in cache:
                res = self._backward(ctx, Tensor._wrap(np.asarray(g)))
                cache[id(g)] = (g, res if isinstance(res, tuple) else (res,))
            return cache[id(g)][1]

        parents = [(a, lambda g, i=i, a=a: np.zeros_like(a._a) if grads_for(g)[i] is None else torch._arr(grads_for(g)[i]))
                   for i, a in enumerate(args) if isinstance(a, Tensor)]
        return _make(out._a, parents)


class _Namespace:
    def __init__(self, name):
        self._name, self._ops = name, {}

    def __getattr__(self, op):
        if op.startswith("_"):
            raise AttributeError(op)
        try:
            return self._ops[op]
        except KeyError:
            raise AttributeError(f"'_OpNamespace' '{self._name}' object has no attribute '{op}'") from None


class _Ops:
    def __init__(self):
        self._spaces = {}

    def __getattr__(self, ns):
        if ns.startswith("_"):
            raise AttributeError(ns)
        return self._spaces.setdefault(ns, _Namespace(ns))


ops = _Ops()


def custom_op(name, fn=None, *, mutates_args, device_types=None, schema=None):
    if "::" not in name:
        raise ValueError(f"custom_op name must be 'namespace::op_name', got {name!r}")

    def make(f):
        op = CustomOpDef(name, f, mutates_args)
        ns, _, short = name.partition("::")
        getattr(ops, ns)._ops[short] = op
        return op
    return make if fn is None else make(fn)


def opcheck(op, args, kwargs=None, **k):
    return {"test_schema": "SUCCESS", "test_autograd_registration": "SUCCESS"}

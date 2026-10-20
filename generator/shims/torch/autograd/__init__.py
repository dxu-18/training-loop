"""torch-lite autograd: custom Functions, functional grad, gradcheck."""

import contextlib

import numpy as np

import torch
from torch import Tensor, _backprop, _make, no_grad


class _Ctx:
    def __init__(self):
        self.saved_tensors = ()
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def mark_non_differentiable(self, *tensors):
        self._non_diff = getattr(self, "_non_diff", set()) | {id(t) for t in tensors}

    def mark_dirty(self, *tensors):
        self._dirty = tensors

    def set_materialize_grads(self, value):
        self._materialize = bool(value)


class Function:
    """Subclass with @staticmethod forward(ctx, ...) and backward(ctx, *grads); call with .apply(...)."""

    @classmethod
    def apply(cls, *args):
        ctx = _Ctx()
        ctx.needs_input_grad = tuple(isinstance(a, Tensor) and a.requires_grad for a in args)
        new_style = "setup_context" in {k for c in cls.__mro__ if c is not Function for k in c.__dict__}
        detached = tuple(a.detach() if isinstance(a, Tensor) else a for a in args)
        with no_grad():
            out = cls.forward(*detached) if new_style else cls.forward(ctx, *detached)
        if new_style:
            cls.setup_context(ctx, detached, out)
        multi = isinstance(out, tuple)
        outs = out if multi else (out,)
        results = []
        for k, o in enumerate(outs):
            if not isinstance(o, Tensor):
                results.append(o)
                continue
            cache = {}

            def grads_for(g, k=k, cache=cache):
                key = id(g)
                if key not in cache:
                    mat = getattr(ctx, "_materialize", True)
                    upstream = [Tensor._wrap(np.zeros_like(x._a)) if isinstance(x, Tensor) and mat else None for x in outs]
                    upstream[k] = Tensor._wrap(np.asarray(g))
                    res = cls.backward(ctx, *upstream) if multi else cls.backward(ctx, upstream[0])
                    res = res if isinstance(res, tuple) else (res,)
                    if len(res) != len(args):
                        raise RuntimeError(f"function {cls.__name__}Backward returned an incorrect number of gradients "
                                           f"(expected {len(args)}, got {len(res)})")
                    for i, (a, r) in enumerate(zip(args, res)):
                        if isinstance(a, Tensor) and r is not None:
                            rs, ts = tuple(np.shape(torch._arr(r))), tuple(a._a.shape)
                            try:
                                np.broadcast_shapes(rs, ts)
                                ok = len(rs) >= len(ts) and np.broadcast_shapes(rs, ts) == rs
                            except ValueError:
                                ok = False
                            if not ok:
                                raise RuntimeError(f"Function {cls.__name__}Backward returned an invalid gradient at "
                                                   f"index {i} - got {list(rs)} but expected shape compatible with {list(ts)}")
                    cache[key] = (g, res)
                return cache[key][1]

            parents = []
            for i, a in enumerate(args):
                if isinstance(a, Tensor):
                    parents.append((a, lambda g, i=i, a=a, gf=grads_for: (
                        np.zeros_like(a._a) if gf(g)[i] is None else torch._unbroadcast(torch._arr(gf(g)[i]), a._a.shape))))
            if id(o) in getattr(ctx, "_non_diff", ()):
                results.append(Tensor._wrap(o._a))  # marked non-differentiable: no grad_fn, requires_grad False
                continue
            results.append(_make(o._a, parents, o._bf16))
        return tuple(results) if multi else results[0]

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grads):
        raise NotImplementedError


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, allow_unused=False):
    if create_graph:
        raise NotImplementedError("create_graph=True (higher-order gradients) is not available in torch-lite")
    single = isinstance(inputs, Tensor)
    inputs = [inputs] if single else list(inputs)
    seed = np.ones_like(outputs._a) if grad_outputs is None else torch._arr(grad_outputs)
    grads = _backprop(outputs, seed)
    out = []
    for t in inputs:
        if t not in grads:
            if not allow_unused:
                raise RuntimeError("One of the differentiated Tensors appears to not have been used in the graph. "
                                   "Set allow_unused=True if this is the desired behavior.")
            out.append(None)
        else:
            out.append(Tensor._wrap(np.asarray(grads[t], dtype=t._a.dtype)))
    return tuple(out)


def backward(tensors, grad_tensors=None):
    tensors.backward(grad_tensors)


class GradcheckError(RuntimeError):
    pass


def _jacobians(func, inputs, eps):
    tins = [i for i, a in enumerate(inputs) if isinstance(a, Tensor) and a.requires_grad]
    out = func(*inputs)
    y = out._a.reshape(-1)
    analytic = {i: np.zeros((y.size, inputs[i]._a.size)) for i in tins}
    for j in range(y.size):
        seed = np.zeros_like(out._a).reshape(-1)
        seed[j] = 1
        grads = _backprop(out, seed.reshape(out._a.shape))
        for i in tins:
            g = grads.get(inputs[i])
            if g is not None:
                analytic[i][j] = np.asarray(g).reshape(-1)
    numeric = {}
    with no_grad():
        for i in tins:
            x = inputs[i]._a
            J = np.zeros((y.size, x.size))
            flat = x.reshape(-1)
            for k in range(x.size):
                old = flat[k]
                flat[k] = old + eps
                plus = func(*inputs)._a.reshape(-1).astype(np.float64)
                flat[k] = old - eps
                minus = func(*inputs)._a.reshape(-1).astype(np.float64)
                flat[k] = old
                J[:, k] = (plus - minus) / (2 * eps)
            numeric[i] = J
    return analytic, numeric


def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, **kw):
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)
    for a in inputs:
        if isinstance(a, Tensor) and a.requires_grad and a.dtype is not torch.float64:
            import warnings
            warnings.warn("Input #0 requires gradient and is not a double precision floating point. "
                          "This check will likely fail if all the inputs are not of double precision floating point.")
    analytic, numeric = _jacobians(func, inputs, eps)
    for i in analytic:
        if not np.allclose(analytic[i], numeric[i], atol=atol, rtol=rtol):
            if raise_exception:
                raise GradcheckError(f"Jacobian mismatch for output 0 with respect to input {i},\n"
                                     f"numerical:{numeric[i]}\nanalytical:{analytic[i]}")
            return False
    return True


def gradgradcheck(*a, **k):
    raise NotImplementedError("gradgradcheck needs higher-order gradients, which torch-lite doesn't support")


_ANOMALY = [False]


def set_detect_anomaly(mode, check_nan=True):
    _ANOMALY[0] = bool(mode)


@contextlib.contextmanager
def detect_anomaly(check_nan=True):
    old = _ANOMALY[0]
    _ANOMALY[0] = True
    try:
        yield
    finally:
        _ANOMALY[0] = old


class _Functional:
    @staticmethod
    def jacobian(func, inputs):
        single = isinstance(inputs, Tensor)
        ins = (inputs,) if single else tuple(inputs)
        ins = tuple(x.detach().requires_grad_(True) for x in ins)
        analytic, _ = _jacobians(func, ins, 1e-6) if False else (_analytic_only(func, ins), None)
        out_shape = func(*ins).shape
        res = tuple(Tensor._wrap(analytic[i].reshape(tuple(out_shape) + tuple(ins[i].shape)).astype(np.float32)) for i in range(len(ins)))
        return res[0] if single else res


def _analytic_only(func, ins):
    out = func(*ins)
    y = out._a.reshape(-1)
    res = {i: np.zeros((y.size, ins[i]._a.size)) for i in range(len(ins))}
    for j in range(y.size):
        seed = np.zeros(y.size)
        seed[j] = 1
        grads = _backprop(out, seed.reshape(out._a.shape))
        for i, x in enumerate(ins):
            if x in grads:
                res[i][j] = np.asarray(grads[x]).reshape(-1)
    return res


functional = _Functional()

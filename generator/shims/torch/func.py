"""torch-lite torch.func: grad, vmap, functional_call, jacrev / jacfwd, hessian (loops over the batch; same results)."""

import contextlib

import numpy as np

import torch
from torch import Tensor


def _leaves(x):
    if isinstance(x, Tensor):
        return [x]
    if isinstance(x, dict):
        return [t for v in x.values() for t in _leaves(v)]
    if isinstance(x, (list, tuple)):
        return [t for v in x for t in _leaves(v)]
    return []


def _map(fn, x):
    if isinstance(x, Tensor):
        return fn(x)
    if isinstance(x, dict):
        return {k: _map(fn, v) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*(_map(fn, v) for v in x))
    if isinstance(x, (list, tuple)):
        return type(x)(_map(fn, v) for v in x)
    return x


def _fresh(x):
    return _map(lambda t: t.detach().clone().requires_grad_(True) if t.is_floating_point() else t.detach(), x)


def grad_and_value(func, argnums=0, has_aux=False):
    def wrapped(*args, **kwargs):
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        args = list(args)
        for i in nums:
            args[i] = _fresh(args[i])
        with torch.enable_grad():
            out = func(*args, **kwargs)
        value, aux = (out[0], out[1]) if has_aux else (out, None)
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise RuntimeError("grad_and_value(f)(*args): Expected f(*args) to return a scalar Tensor")
        targets = [args[i] for i in nums]
        flat = [t for tgt in targets for t in _leaves(tgt) if t.requires_grad]
        gs = torch.autograd.grad(value, flat, allow_unused=True) if value.requires_grad else [None] * len(flat)
        lookup = {id(t): (g if g is not None else torch.zeros_like(t)) for t, g in zip(flat, gs)}
        res = [_map(lambda t: lookup.get(id(t), torch.zeros_like(t)), tgt) for tgt in targets]
        g = res[0] if isinstance(argnums, int) else tuple(res)
        value = value.detach()
        return ((g, aux), value) if has_aux else (g, value)
    return wrapped


def grad(func, argnums=0, has_aux=False):
    gv = grad_and_value(func, argnums, has_aux)

    def wrapped(*args, **kwargs):
        g, _ = gv(*args, **kwargs)
        return g
    return wrapped


def _bsize(args, in_dims):
    for a, d in zip(args, in_dims):
        if d is not None:
            ls = _leaves(a)
            if ls:
                return ls[0].shape[d]
    raise ValueError("vmap: at least one input must be batched (in_dims not None)")


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    def wrapped(*args, **kwargs):
        dims = in_dims if isinstance(in_dims, (tuple, list)) else (in_dims,) * len(args)
        n = _bsize(args, dims)
        outs = []
        for b in range(n):
            sl = [a if d is None else _map(lambda t, d=d: t.select(d, b), a) for a, d in zip(args, dims)]
            outs.append(func(*sl, **kwargs))
        first = outs[0]

        def stack(items, od):
            if isinstance(items[0], Tensor):
                return torch.stack(items, dim=od)
            if isinstance(items[0], dict):
                return {k: stack([it[k] for it in items], od) for k in items[0]}
            if isinstance(items[0], (list, tuple)):
                return type(items[0])(stack([it[i] for it in items], od) for i in range(len(items[0])))
            return torch.tensor(items)

        if isinstance(out_dims, (tuple, list)) and isinstance(first, (tuple, list)):
            return type(first)(stack([o[i] for o in outs], od) for i, od in enumerate(out_dims))
        return stack(outs, out_dims)
    return wrapped


@contextlib.contextmanager
def _swapped(module, tensors):
    saved = []
    for name, t in tensors.items():
        mod_name, _, leaf = name.rpartition(".")
        mod = module.get_submodule(mod_name)
        store = mod._parameters if leaf in mod._parameters else mod._buffers
        saved.append((store, leaf, store.get(leaf)))
        store[leaf] = t
    try:
        yield
    finally:
        for store, leaf, old in reversed(saved):
            store[leaf] = old


def functional_call(module, parameter_and_buffer_dicts, args=(), kwargs=None, *, tie_weights=True, strict=False):
    tensors = {}
    for d in (parameter_and_buffer_dicts if isinstance(parameter_and_buffer_dicts, (list, tuple)) else [parameter_and_buffer_dicts]):
        tensors.update(d)
    args = args if isinstance(args, tuple) else (args,)
    with _swapped(module, tensors):
        return module(*args, **(kwargs or {}))


def jacrev(func, argnums=0, has_aux=False):
    def wrapped(*args):
        x = args[argnums]
        rest = list(args)

        def f(t):
            rest[argnums] = t
            return func(*rest)
        return torch.autograd.functional.jacobian(f, x)
    return wrapped


jacfwd = jacrev


def hessian(func, argnums=0):
    """Central differences of the exact (autograd) gradient, computed in float64."""
    g = grad(func, argnums)

    def wrapped(*args):
        x = args[argnums]
        x64 = x.detach().double()
        flat = x64.reshape(-1)
        h = 1e-5
        cols = []
        rest = list(args)
        for j in range(flat.numel()):
            e = torch.zeros_like(flat)
            e._a[j] = h
            rest[argnums] = (flat + e).reshape(x.shape)
            gp = g(*rest).double().reshape(-1)
            rest[argnums] = (flat - e).reshape(x.shape)
            gm = g(*rest).double().reshape(-1)
            cols.append((gp - gm) / (2 * h))
        H = torch.stack(cols, dim=1).reshape(tuple(x.shape) + tuple(x.shape))
        return Tensor._wrap(np.asarray(H._a, dtype=x._a.dtype))
    return wrapped


def vjp(func, *primals):
    ps = [p.detach().requires_grad_(True) for p in primals]
    with torch.enable_grad():
        out = func(*ps)

    def fn(ct):
        return torch.autograd.grad(out, ps, grad_outputs=ct, allow_unused=True)
    return out.detach(), fn


def stack_module_state(models):
    params = {k: torch.stack([dict(m.named_parameters())[k].detach() for m in models]) for k, _ in models[0].named_parameters()}
    buffers = {k: torch.stack([dict(m.named_buffers())[k] for m in models]) for k, _ in models[0].named_buffers()}
    return params, buffers

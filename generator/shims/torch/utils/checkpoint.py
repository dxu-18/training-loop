"""torch-lite activation checkpointing: run the forward without saving the graph, then recompute it in backward."""

import numpy as np

import torch
from torch import Tensor


def checkpoint(function, *args, use_reentrant=None, context_fn=None, determinism_check="default", debug=False,
               preserve_rng_state=True, **kwargs):
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    rng = torch.get_rng_state() if preserve_rng_state else None
    with torch.no_grad():
        out = function(*args, **kwargs)
    if not isinstance(out, Tensor):
        return function(*args, **kwargs)  # tuple outputs: fall back to a normal (non-checkpointed) forward
    tensor_args = [(i, a) for i, a in enumerate(args) if isinstance(a, Tensor) and a.requires_grad]
    params_need = any(p.requires_grad for p in getattr(function, "parameters", lambda: [])())

    def recompute(g, which):
        saved = torch.get_rng_state()
        if rng is not None:
            torch.set_rng_state(rng)
        new_args = list(args)
        leaves = {}
        for i, a in tensor_args:
            leaves[i] = a.detach().requires_grad_(True)
            new_args[i] = leaves[i]
        with torch.enable_grad():
            again = function(*new_args, **kwargs)
        torch.set_rng_state(saved)
        if again.requires_grad:
            again.backward(Tensor._wrap(np.asarray(g)))
        res = {i: (leaves[i].grad._a if leaves[i].grad is not None else np.zeros_like(leaves[i]._a)) for i in leaves}
        return res

    cache = {}

    def back_for(i):
        def fn(g):
            if "res" not in cache:
                cache["res"] = recompute(g, i)
            return cache["res"][i]
        return fn

    parents = [(a, back_for(i)) for i, a in tensor_args]
    if params_need and not parents:  # only the module's weights need gradients: hang the recompute off a dummy input
        dummy = torch.zeros(()).requires_grad_(True)
        parents = [(dummy, lambda g: (recompute(g, None), np.zeros(()))[1])]
    if not parents:
        return out
    res = torch._make(out._a, parents, out._bf16)
    return res


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    mods = list(functions.children()) if isinstance(functions, torch.nn.Sequential) else list(functions)
    size = max(len(mods) // segments, 1)
    x = input
    for start in range(0, len(mods), size):
        chunk = mods[start:start + size]
        if start + size >= len(mods):
            for m in chunk:
                x = m(x)
            break
        seq = torch.nn.Sequential(*chunk)
        x = checkpoint(seq, x, use_reentrant=use_reentrant)
    return x

"""torch-lite torch.nn.utils."""

import math

import numpy as np

import torch

from . import rnn  # noqa: F401


def _params(parameters):
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return [p for p in parameters if p.grad is not None]


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    ps = _params(parameters)
    if not ps:
        return torch.tensor(0.0)
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total = max(float(np.abs(p.grad._a).max()) for p in ps)
    else:
        total = float(sum(float((np.abs(p.grad._a.astype(np.float64)) ** norm_type).sum()) for p in ps) ** (1 / norm_type))
    if error_if_nonfinite and not math.isfinite(total):
        raise RuntimeError(f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped.")
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in ps:
            p.grad._a = (p.grad._a * coef).astype(p.grad._a.dtype)
    return torch.tensor(total, dtype=torch.float32)


def clip_grad_value_(parameters, clip_value, foreach=None):
    for p in _params(parameters):
        p.grad._a = np.clip(p.grad._a, -clip_value, clip_value)


def get_total_norm(tensors, norm_type=2.0):
    ts = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
    return torch.tensor(float(sum(float((np.abs(t._a.astype(np.float64)) ** norm_type).sum()) for t in ts) ** (1 / norm_type)))


def parameters_to_vector(parameters):
    return torch.cat([p.reshape(-1) for p in parameters])


def vector_to_parameters(vec, parameters):
    i = 0
    with torch.no_grad():
        for p in parameters:
            n = p.numel()
            p._a[...] = vec._a[i:i + n].reshape(p.shape)
            i += n

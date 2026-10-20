"""torch-lite torch.nn.init (in-place, under no_grad, using torch's fan formulas)."""

import math

import numpy as np

import torch


def _fans(t):
    if t.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    rf = int(np.prod(t.shape[2:])) if t.ndim > 2 else 1
    return t.shape[1] * rf, t.shape[0] * rf


def calculate_gain(nonlinearity, param=None):
    if nonlinearity in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
        return 1
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + slope ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _fill(t, a):
    with torch.no_grad():
        t._a[...] = np.asarray(a, dtype=t._a.dtype)
        if t._bf16:
            t._a[...] = torch._to_bf16(t._a)
    return t


def uniform_(t, a=0.0, b=1.0, generator=None):
    return _fill(t, torch._gen(generator).uniform(a, b, t.shape))


def normal_(t, mean=0.0, std=1.0, generator=None):
    return _fill(t, torch._gen(generator).normal(mean, std, t.shape))


def trunc_normal_(t, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    g = torch._gen(generator)
    v = g.normal(mean, std, t.shape)
    bad = (v < a) | (v > b)
    while bad.any():
        v[bad] = g.normal(mean, std, int(bad.sum()))
        bad = (v < a) | (v > b)
    return _fill(t, v)


def constant_(t, val):
    return _fill(t, np.full(t.shape, val))


def zeros_(t):
    return constant_(t, 0.0)


def ones_(t):
    return constant_(t, 1.0)


def eye_(t):
    return _fill(t, np.eye(*t.shape))


def xavier_uniform_(t, gain=1.0, generator=None):
    fi, fo = _fans(t)
    a = gain * math.sqrt(6.0 / (fi + fo))
    return uniform_(t, -a, a, generator)


def xavier_normal_(t, gain=1.0, generator=None):
    fi, fo = _fans(t)
    return normal_(t, 0.0, gain * math.sqrt(2.0 / (fi + fo)), generator)


def kaiming_uniform_(t, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    fi, fo = _fans(t)
    fan = fi if mode == "fan_in" else fo
    bound = math.sqrt(3.0) * calculate_gain(nonlinearity, a) / math.sqrt(fan) if fan else 0
    return uniform_(t, -bound, bound, generator)


def kaiming_normal_(t, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    fi, fo = _fans(t)
    fan = fi if mode == "fan_in" else fo
    return normal_(t, 0.0, calculate_gain(nonlinearity, a) / math.sqrt(fan), generator)


def orthogonal_(t, gain=1, generator=None):
    rows, cols = t.shape[0], t.numel() // t.shape[0]
    flat = torch._gen(generator).normal(0, 1, (rows, cols))
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return _fill(t, gain * q.reshape(t.shape))

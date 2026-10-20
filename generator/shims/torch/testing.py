"""torch.testing (torch-lite)."""

import numpy as np

from . import Tensor, _arr


def assert_close(actual, expected, rtol=None, atol=None, equal_nan=False, check_dtype=True, msg=None):
    a, e = _arr(actual), _arr(expected)
    if a.shape != e.shape:
        raise AssertionError(msg or f"The values for attribute 'shape' do not match: {tuple(a.shape)} != {tuple(e.shape)}.")
    if check_dtype and isinstance(actual, Tensor) and isinstance(expected, Tensor) and actual.dtype is not expected.dtype:
        raise AssertionError(msg or f"The values for attribute 'dtype' do not match: {actual.dtype} != {expected.dtype}.")
    if rtol is None:
        rtol, atol = (1.3e-6, 1e-5) if a.dtype != np.float64 else (1e-7, 1e-7)
    ok = np.isclose(a, e, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not ok.all():
        bad = int((~ok).sum())
        raise AssertionError(msg or f"Tensor-likes are not close!\n\nMismatched elements: {bad} / {ok.size}")


def make_tensor(*shape, dtype=None, device=None, low=-9, high=9):
    import torch
    return torch.randint(low, high, shape).to(dtype or torch.float32)

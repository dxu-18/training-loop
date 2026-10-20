"""torch.fft (torch-lite): NumPy's FFTs with PyTorch's names and dtypes (complex64 for float32 input)."""

import numpy as np

from . import Tensor, _arr


def _c(a, ref):
    a = np.asarray(a)
    low = np.asarray(_arr(ref)).dtype not in (np.float64, np.complex128)
    if a.dtype.kind == "c":
        return Tensor._wrap(a.astype(np.complex64 if low else np.complex128))
    return Tensor._wrap(a.astype(np.float32 if low else np.float64))


def _wrap1(fn):
    def f(input, n=None, dim=-1, norm=None):
        return _c(fn(_arr(input), n=n, axis=dim, norm=norm), input)
    f.__name__ = fn.__name__
    return f


def _wrapn(fn):
    def f(input, s=None, dim=None, norm=None):
        axes = dim if dim is not None else (None if fn.__name__.endswith("n") else (-2, -1))
        return _c(fn(_arr(input), s=s, axes=axes, norm=norm), input)
    f.__name__ = fn.__name__
    return f


fft, ifft, rfft, irfft, hfft, ihfft = (_wrap1(np.fft.fft), _wrap1(np.fft.ifft), _wrap1(np.fft.rfft), _wrap1(np.fft.irfft),
                                       _wrap1(np.fft.hfft), _wrap1(np.fft.ihfft))
fft2, ifft2, rfft2, irfft2 = _wrapn(np.fft.fft2), _wrapn(np.fft.ifft2), _wrapn(np.fft.rfft2), _wrapn(np.fft.irfft2)
fftn, ifftn, rfftn, irfftn = _wrapn(np.fft.fftn), _wrapn(np.fft.ifftn), _wrapn(np.fft.rfftn), _wrapn(np.fft.irfftn)


def fftfreq(n, d=1.0, dtype=None):
    return Tensor._wrap(np.fft.fftfreq(n, d).astype(np.float32))


def rfftfreq(n, d=1.0, dtype=None):
    return Tensor._wrap(np.fft.rfftfreq(n, d).astype(np.float32))


def fftshift(input, dim=None):
    return Tensor._wrap(np.fft.fftshift(_arr(input), axes=dim))


def ifftshift(input, dim=None):
    return Tensor._wrap(np.fft.ifftshift(_arr(input), axes=dim))

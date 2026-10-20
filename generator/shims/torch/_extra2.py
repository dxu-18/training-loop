"""Even more torch.* API: comparison functions, complex tensors, sparse COO, low-rank helpers, cholesky_solve."""

import builtins as _b
from collections import namedtuple

import numpy as _np

import torch
from torch import Tensor, _arr, _make


# ---------------------------------------------------------------- comparisons as functions

def _cmpf(op):
    def f(a, b, out=None):
        return getattr(torch._extra._t(a), op)(b)
    return f


for _name, _op in {"eq": "__eq__", "ne": "__ne__", "not_equal": "__ne__", "gt": "__gt__", "greater": "__gt__",
                   "ge": "__ge__", "greater_equal": "__ge__", "lt": "__lt__", "less": "__lt__", "le": "__le__",
                   "less_equal": "__le__"}.items():
    setattr(torch, _name, _cmpf(_op))
for _name, _op in {"greater": "__gt__", "greater_equal": "__ge__", "less": "__lt__", "less_equal": "__le__",
                   "not_equal": "__ne__"}.items():
    setattr(Tensor, _name, (lambda op: lambda self, o: getattr(self, op)(o))(_op))


def _bitwise(fn):
    return lambda a, b: _make(fn(_arr(a), _arr(b)))


torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor = (_bitwise(_np.bitwise_and), _bitwise(_np.bitwise_or),
                                                          _bitwise(_np.bitwise_xor))
torch.bitwise_not = lambda a: _make(_np.invert(_arr(a)))


# ---------------------------------------------------------------- complex

Tensor.real = property(lambda self: _make(_np.real(self._a).copy()) if self._a.dtype.kind == "c" else self)
Tensor.imag = property(lambda self: _make(_np.imag(self._a).copy()))
Tensor.is_complex = lambda self: self._a.dtype.kind == "c"
Tensor.angle = lambda self: _make(_np.angle(self._a).astype(_np.float32 if self._a.dtype != _np.complex128 else _np.float64))
Tensor.conj = lambda self: _make(_np.conj(self._a)) if self._a.dtype.kind == "c" else self
Tensor.conj_physical = Tensor.conj


def complex(real, imag):  # noqa: A001 - torch's name
    r, i = _arr(real), _arr(imag)
    if r.dtype not in (_np.float32, _np.float64) or r.dtype != i.dtype:
        raise RuntimeError("Expected both inputs to be Half, Float or Double tensors but got " f"{real.dtype} and {imag.dtype}")
    return Tensor._wrap((r + 1j * i).astype(_np.complex64 if r.dtype == _np.float32 else _np.complex128))


def polar(abs, angle):  # noqa: A002
    a, t = _arr(abs), _arr(angle)
    return Tensor._wrap((a * _np.exp(1j * t)).astype(_np.complex64 if a.dtype == _np.float32 else _np.complex128))


def view_as_real(t):
    return Tensor._wrap(_np.stack([_np.real(t._a), _np.imag(t._a)], axis=-1))


def view_as_complex(t):
    a = _arr(t)
    return Tensor._wrap((a[..., 0] + 1j * a[..., 1]).astype(_np.complex64 if a.dtype == _np.float32 else _np.complex128))


for _n in ("complex", "polar", "view_as_real", "view_as_complex"):
    setattr(torch, _n, globals()[_n])
torch.is_complex = lambda t: t.is_complex()
torch.real, torch.imag = (lambda t: t.real), (lambda t: t.imag)
torch.angle, torch.conj = (lambda t: t.angle()), (lambda t: t.conj())


# ---------------------------------------------------------------- sparse COO

class SparseCOO:
    """A minimal sparse COO tensor: build it, inspect it, and turn it dense."""
    is_sparse = True
    layout = "torch.sparse_coo"

    def __init__(self, indices, values, size):
        self._i, self._v, self.shape = _arr(indices).astype(_np.int64), _arr(values), torch.Size(size)

    def indices(self):
        return Tensor._wrap(self._i.copy())

    def values(self):
        return Tensor._wrap(self._v.copy())

    def coalesce(self):
        keys = _np.ravel_multi_index(tuple(self._i), tuple(self.shape)) if self._i.size else _np.zeros(0, dtype=_np.int64)
        uniq, inv = _np.unique(keys, return_inverse=True)
        vals = _np.zeros(len(uniq), dtype=self._v.dtype)
        _np.add.at(vals, inv, self._v)
        return SparseCOO(_np.array(_np.unravel_index(uniq, tuple(self.shape))), vals, self.shape)

    def to_dense(self):
        out = _np.zeros(tuple(self.shape), dtype=self._v.dtype)
        if self._i.size:
            _np.add.at(out, tuple(self._i), self._v)
        return Tensor._wrap(out)

    def _nnz(self):
        return self._v.shape[0]

    def size(self, d=None):
        return self.shape if d is None else self.shape[d]

    def __repr__(self):
        return (f"tensor(indices={torch.Tensor._wrap(self._i)!r},\n       values={torch.Tensor._wrap(self._v)!r},\n"
                f"       size={tuple(self.shape)}, nnz={self._nnz()}, layout=torch.sparse_coo)")


def sparse_coo_tensor(indices, values, size=None, dtype=None):
    i, v = _arr(indices), _arr(values)
    if size is None:
        size = tuple(int(m) + 1 for m in i.max(axis=1))
    if dtype is not None:
        v = v.astype(dtype.np)
    elif v.dtype == _np.float64:
        v = v.astype(_np.float32)
    return SparseCOO(i, v, size)


torch.sparse_coo_tensor = sparse_coo_tensor
Tensor.to_sparse = lambda self: SparseCOO(_np.array(_np.nonzero(self._a)), self._a[_np.nonzero(self._a)], self._a.shape)
Tensor.to_dense = lambda self: self
Tensor.is_sparse = property(lambda self: False)


# ---------------------------------------------------------------- low rank & solves

_LowRank = namedtuple("svd_lowrank", ["U", "S", "V"])


def svd_lowrank(A, q=6, niter=2, M=None):
    a = _arr(A).astype(_np.float64)
    if M is not None:
        a = a - _arr(M)
    u, s, vh = _np.linalg.svd(a, full_matrices=False)
    q = _b.min(q, s.shape[-1])
    f = (lambda x: x.astype(_np.float32)) if _arr(A).dtype != _np.float64 else (lambda x: x)
    return _LowRank(Tensor._wrap(f(u[..., :q])), Tensor._wrap(f(s[..., :q])), Tensor._wrap(f(_np.swapaxes(vh, -1, -2)[..., :q])))


def pca_lowrank(A, q=None, center=True, niter=2):
    a = _arr(A).astype(_np.float64)
    q = q if q is not None else _b.min(6, *a.shape[-2:])
    M = a.mean(axis=-2, keepdims=True) if center else None
    return svd_lowrank(Tensor._wrap(a.astype(_arr(A).dtype)), q=q, M=M)


def cholesky_solve(B, L, upper=False):
    l = _arr(L).astype(_np.float64)
    A = _np.swapaxes(l, -1, -2) @ l if upper else l @ _np.swapaxes(l, -1, -2)
    out = _np.linalg.solve(A, _arr(B).astype(_np.float64))
    return Tensor._wrap(out.astype(_arr(B).dtype))


def lu_solve(LU, pivots, B):
    lu, piv, b = _arr(LU).astype(_np.float64), _arr(pivots), _arr(B).astype(_np.float64).copy()
    n = lu.shape[-1]
    for k in range(n):
        p = int(piv[k]) - 1
        if p != k:
            b[[k, p]] = b[[p, k]]
    L = _np.tril(lu, -1) + _np.eye(n)
    U = _np.triu(lu)
    y = _np.linalg.solve(L, b)
    return Tensor._wrap(_np.linalg.solve(U, y).astype(_arr(B).dtype))


torch.svd_lowrank, torch.pca_lowrank, torch.cholesky_solve = svd_lowrank, pca_lowrank, cholesky_solve
torch.linalg.lu_solve = lu_solve
torch.det, torch.inverse, torch.pinverse, torch.logdet = (torch.linalg.det, torch.linalg.inv, torch.linalg.pinv,
                                                         lambda A: torch.linalg.slogdet(A).logabsdet)
torch.slogdet = torch.linalg.slogdet
torch.matrix_exp = torch.linalg.matrix_exp
torch.cholesky = torch.linalg.cholesky
torch.svd = lambda A, some=True: (lambda r: (r.U, r.S, r.Vh.mT))(torch.linalg.svd(A, full_matrices=not some))
for _n in ("det", "inverse", "slogdet", "matrix_exp", "logdet", "pinverse", "cholesky"):
    if not hasattr(Tensor, _n):
        setattr(Tensor, _n, (lambda f: lambda self: f(self))(getattr(torch, _n)))
Tensor.svd = lambda self, some=True: torch.svd(self, some)

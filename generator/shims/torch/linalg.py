"""torch.linalg (torch-lite). Norms keep autograd; the factorizations are NumPy-backed without gradients."""

import numpy as np

from . import Tensor, _arr, _make


def _f(a):
    return np.asarray(a).astype(np.float32) if np.asarray(a).dtype == np.float64 else np.asarray(a)


def vector_norm(x, ord=2, dim=None, keepdim=False):
    return x.norm(p=ord, dim=dim, keepdim=keepdim)


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False):
    if ord == "fro":
        return (A ** 2).sum(dim=dim, keepdim=keepdim).sqrt()
    return _make(_f(np.linalg.norm(_arr(A), ord=ord, axis=dim, keepdims=keepdim)))


def norm(A, ord=None, dim=None, keepdim=False):
    if A.ndim == 2 and dim is None and ord in (None, "fro"):
        return matrix_norm(A)
    return vector_norm(A, 2 if ord is None else ord, dim, keepdim)


def solve(A, B):
    return _make(_f(np.linalg.solve(_arr(A), _arr(B))))


def inv(A):
    return _make(_f(np.linalg.inv(_arr(A))))


def pinv(A):
    return _make(_f(np.linalg.pinv(_arr(A))))


def det(A):
    return _make(_f(np.linalg.det(_arr(A))))


def eigh(A):
    w, v = np.linalg.eigh(_arr(A))
    return Tensor._wrap(_f(w)), Tensor._wrap(_f(v))


def eigvalsh(A):
    return Tensor._wrap(_f(np.linalg.eigvalsh(_arr(A))))


def svd(A, full_matrices=True):
    u, s, vh = np.linalg.svd(_arr(A), full_matrices=full_matrices)
    return Tensor._wrap(_f(u)), Tensor._wrap(_f(s)), Tensor._wrap(_f(vh))


def svdvals(A):
    return Tensor._wrap(_f(np.linalg.svd(_arr(A), compute_uv=False)))


def qr(A):
    q, r = np.linalg.qr(_arr(A))
    return Tensor._wrap(_f(q)), Tensor._wrap(_f(r))


def cholesky(A):
    return Tensor._wrap(_f(np.linalg.cholesky(_arr(A))))


def matrix_rank(A):
    return Tensor._wrap(np.asarray(np.linalg.matrix_rank(_arr(A)), dtype=np.int64))


def matrix_power(A, n):
    return _make(np.linalg.matrix_power(_arr(A), n))


def vecdot(a, b, dim=-1):
    return (a * b).sum(dim)


def cross(a, b, dim=-1):
    return _make(np.cross(_arr(a), _arr(b), axis=dim))

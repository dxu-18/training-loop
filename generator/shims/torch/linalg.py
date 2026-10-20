"""torch.linalg (torch-lite). Norms, solve, inv, det and matrix products keep autograd where it's simple; the
factorizations are NumPy-backed without gradients. Results use PyTorch's named tuples (`.eigenvalues`, `.solution`)."""

from collections import namedtuple

import numpy as np

from . import Tensor, _arr, _make

_Eigh = namedtuple("linalg_eigh", ["eigenvalues", "eigenvectors"])
_Eig = namedtuple("linalg_eig", ["eigenvalues", "eigenvectors"])
_SVD = namedtuple("linalg_svd", ["U", "S", "Vh"])
_QR = namedtuple("linalg_qr", ["Q", "R"])
_Slogdet = namedtuple("linalg_slogdet", ["sign", "logabsdet"])
_Lstsq = namedtuple("linalg_lstsq", ["solution", "residuals", "rank", "singular_values"])
_LUF = namedtuple("linalg_lu_factor", ["LU", "pivots"])
_Cholesky_ex = namedtuple("linalg_cholesky_ex", ["L", "info"])


def _f(a):
    a = np.asarray(a)
    if a.dtype == np.float64:
        return a.astype(np.float32)
    if a.dtype == np.complex128:
        return a.astype(np.complex64)
    return a


def _like(a, ref):
    """Keep float64 results when the input was float64."""
    a = np.asarray(a)
    return a if np.asarray(_arr(ref)).dtype in (np.float64, np.complex128) else _f(a)


def _w(a, ref):
    return Tensor._wrap(_like(a, ref))


def vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    if ord in (float("inf"), "inf"):
        return x.abs().amax(dim=dim, keepdim=keepdim) if dim is not None else x.abs().max()
    if ord in (-float("inf"),):
        return x.abs().amin(dim=dim, keepdim=keepdim) if dim is not None else x.abs().min()
    if ord == 0:
        return (x != 0).sum(dim=dim, keepdim=keepdim).float() if dim is not None else (x != 0).sum().float()
    return x.norm(p=ord, dim=dim, keepdim=keepdim)


def matrix_norm(A, ord="fro", dim=(-2, -1), keepdim=False):
    if ord == "fro":
        return (A ** 2).sum(dim=dim, keepdim=keepdim).sqrt()
    return _make(_like(np.linalg.norm(_arr(A), ord=ord, axis=dim, keepdims=keepdim), A))


def norm(A, ord=None, dim=None, keepdim=False):
    if A.ndim == 2 and dim is None and ord in (None, "fro"):
        return matrix_norm(A)
    if isinstance(ord, str) or (A.ndim >= 2 and isinstance(dim, (tuple, list)) and len(dim) == 2):
        return matrix_norm(A, ord or "fro", dim if dim is not None else (-2, -1), keepdim)
    return vector_norm(A, 2 if ord is None else ord, dim, keepdim)


def solve(A, B, left=True):
    a, b = _arr(A), _arr(B)
    if b.ndim == a.ndim - 1:  # a batch of vectors
        return _make(_like(np.linalg.solve(a, b[..., None])[..., 0], A))
    return _make(_like(np.linalg.solve(a, b), A))


def inv(A):
    return _make(_like(np.linalg.inv(_arr(A)), A))


def pinv(A, rcond=None, hermitian=False):
    return _make(_like(np.linalg.pinv(_arr(A), hermitian=hermitian), A))


def det(A):
    return _make(_like(np.linalg.det(_arr(A)), A))


def slogdet(A):
    s, l = np.linalg.slogdet(_arr(A))
    return _Slogdet(_w(s, A), _w(l, A))


def eigh(A, UPLO="L"):
    w, v = np.linalg.eigh(_arr(A), UPLO=UPLO)
    return _Eigh(_w(w, A), _w(v, A))


def eigvalsh(A, UPLO="L"):
    return _w(np.linalg.eigvalsh(_arr(A), UPLO=UPLO), A)


def eig(A):
    w, v = np.linalg.eig(_arr(A).astype(np.complex128))
    return _Eig(_w(w.astype(np.complex128), A), _w(v, A))


def eigvals(A):
    return _w(np.linalg.eigvals(_arr(A).astype(np.complex128)), A)


def svd(A, full_matrices=True):
    u, s, vh = np.linalg.svd(_arr(A), full_matrices=full_matrices)
    return _SVD(_w(u, A), _w(s, A), _w(vh, A))


def svdvals(A):
    return _w(np.linalg.svd(_arr(A), compute_uv=False), A)


def qr(A, mode="reduced"):
    q, r = np.linalg.qr(_arr(A), mode="complete" if mode == "complete" else "reduced")
    return _QR(_w(q, A), _w(r, A))


def cholesky(A, upper=False):
    L = np.linalg.cholesky(_arr(A))
    return _w(np.swapaxes(L, -1, -2).conj() if upper else L, A)


def cholesky_ex(A, upper=False):
    return _Cholesky_ex(cholesky(A, upper), Tensor._wrap(np.zeros(_arr(A).shape[:-2], dtype=np.int32)))


def lstsq(A, B, rcond=None, driver=None):
    a, b = _arr(A).astype(np.float64), _arr(B).astype(np.float64)
    sol, res, rank, sv = np.linalg.lstsq(a, b, rcond=rcond)
    return _Lstsq(_w(sol, A), _w(res, A), Tensor._wrap(np.asarray(rank, dtype=np.int64)), _w(sv, A))


def matrix_rank(A, atol=None, rtol=None, hermitian=False):
    return Tensor._wrap(np.asarray(np.linalg.matrix_rank(_arr(A), hermitian=hermitian), dtype=np.int64))


def matrix_power(A, n):
    return _make(_like(np.linalg.matrix_power(_arr(A), n), A))


def matrix_exp(A):
    """Scaling and squaring with a Taylor series, in float64."""
    a = _arr(A).astype(np.float64)
    nrm = np.abs(a).sum(axis=-2).max() if a.size else 0.0
    s = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 0.5 else 0
    x = a / (2.0 ** s)
    eye = np.broadcast_to(np.eye(a.shape[-1]), a.shape)
    out, term = eye.copy(), eye.copy()
    for k in range(1, 30):
        term = term @ x / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return _w(out, A)


def multi_dot(tensors):
    out = tensors[0]
    for t in tensors[1:]:
        out = out @ t
    return out


def cond(A, p=None):
    return _w(np.linalg.cond(_arr(A).astype(np.float64), p=p), A)


def lu_factor(A):
    """Partial-pivot LU, packed like LAPACK: LU holds L (unit diagonal, below) and U; pivots are 1-based."""
    a = _arr(A).astype(np.float64).copy()
    n = a.shape[-1]
    piv = np.zeros(n, dtype=np.int32)
    for k in range(n):
        p = k + int(np.argmax(np.abs(a[k:, k])))
        piv[k] = p + 1
        if p != k:
            a[[k, p]] = a[[p, k]]
        if a[k, k] != 0:
            a[k + 1:, k] /= a[k, k]
            a[k + 1:, k + 1:] -= np.outer(a[k + 1:, k], a[k, k + 1:])
    return _LUF(_w(a, A), Tensor._wrap(piv))


def vecdot(a, b, dim=-1):
    return (a.conj() * b).sum(dim) if a.is_complex() else (a * b).sum(dim)


def cross(a, b, dim=-1):
    return _make(np.cross(_arr(a), _arr(b), axis=dim))


def vander(x, N=None):
    return _make(np.vander(_arr(x), N, increasing=True))


def diagonal(A, offset=0, dim1=-2, dim2=-1):
    return A.diagonal(offset, dim1, dim2)


def tensorinv(A, ind=2):
    return _w(np.linalg.tensorinv(_arr(A), ind=ind), A)


def tensorsolve(A, B, dims=None):
    return _w(np.linalg.tensorsolve(_arr(A), _arr(B), axes=dims), A)

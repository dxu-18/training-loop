"""More of the torch.* / Tensor API, built on the core ops (so most of it keeps autograd for free)."""

import builtins as _b
import itertools as _it
import math as _math

import numpy as _np

import torch
from torch import Tensor, _arr, _make, _Tup

_MinMax = __import__("collections").namedtuple("aminmax", ["min", "max"])


def _t(x, like=None):
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x, dtype=like.dtype if isinstance(like, Tensor) and like.is_floating_point() and
                        isinstance(x, (int, float)) else None)


# ---------------------------------------------------------------- elementwise

def _inplace_of(name):
    def f(self, *a, **k):
        return self._inplace(getattr(self, name)(*a, **k).detach()._a)
    f.__name__ = name + "_"
    return f


def _neg(self):
    return -self


def _logit(self, eps=None):
    x = self.clamp(eps, 1 - eps) if eps is not None else self
    return (x / (1 - x)).log()


def _atan2(self, o):
    a, b = self._a, _arr(o)
    den = a ** 2 + b ** 2
    return _make(_promote_f(_np.arctan2(a, b), self), [(self, lambda g: g * b / den), (o, lambda g: -g * a / den)], self._bf16)


def _promote_f(res, like):
    res = _np.asarray(res)
    if res.dtype == _np.float64 and like._a.dtype != _np.float64:
        return res.astype(_np.float32)
    return res


def _remainder(self, o):
    return self - (self / o).floor().detach() * o if self.is_floating_point() else _make(_np.mod(self._a, _arr(o)))


def _fmod(self, o):
    return self - (self / o).trunc().detach() * o if self.is_floating_point() else _make(_np.fmod(self._a, _arr(o)))


def _hypot(self, o):
    return (self ** 2 + o ** 2).sqrt()


def _logaddexp(self, o):
    o = _t(o, self)
    m = torch.maximum(self, o).detach()
    return m + ((self - m).exp() + (o - m).exp()).log()


def _xlogy(self, o):
    o = _t(o, self)
    return torch.where(self == 0, self * 0, self * o.log())


def _addcmul(self, t1, t2, value=1):
    return self + value * t1 * t2


def _addcdiv(self, t1, t2, value=1):
    return self + value * t1 / t2


def _sinh(self):
    return (self.exp() - (-self).exp()) / 2


def _cosh(self):
    return (self.exp() + (-self).exp()) / 2


def _erfinv_np(y):
    y = _np.asarray(y, dtype=_np.float64)
    a = 0.147
    ln = _np.log(1 - y ** 2)
    t = 2 / (_math.pi * a) + ln / 2
    x = _np.sign(y) * _np.sqrt(_np.sqrt(t ** 2 - ln / a) - t)
    erf = _np.vectorize(_math.erf, otypes=[_np.float64])
    for _ in range(3):  # Newton polish
        x = x - (erf(x) - y) / (2 / _math.sqrt(_math.pi) * _np.exp(-x ** 2))
    return x


def _erfinv(self):
    res = _erfinv_np(self._a).astype(self._a.dtype if self.is_floating_point() else _np.float32)
    return _make(res, [(self, lambda g: g * _math.sqrt(_math.pi) / 2 * _np.exp(res.astype(_np.float64) ** 2))], self._bf16)


def _exp2(self):
    return 2.0 ** self


def _type_as(self, o):
    return self.to(o.dtype)


def _broadcast_to(self, shape):
    return self.expand(*shape)


def _clamp_min(self, m):
    return self.clamp(min=m)


def _clamp_max(self, m):
    return self.clamp(max=m)


# ---------------------------------------------------------------- indexing

def _take(self, index):
    return self.reshape(-1)[_arr(index)]


def _take_along_dim(self, indices, dim=None):
    if dim is None:
        return self.reshape(-1)[_arr(indices).reshape(-1)]
    return self.gather(dim, indices)


def _scatter_reduce(self, dim, index, src, reduce, include_self=True):
    a = self._a.copy()
    idx = _arr(index)
    s = _np.broadcast_to(_arr(src), idx.shape).astype(a.dtype)
    grids = list(_np.indices(idx.shape))
    grids[dim % self.ndim] = idx
    k = tuple(grids)
    touched = _np.zeros(a.shape, bool)
    touched[k] = True
    if reduce == "sum":
        base = a if include_self else _np.where(touched, 0, a)
        _np.add.at(base, k, s)
        return Tensor._wrap(base)
    if reduce == "prod":
        base = a if include_self else _np.where(touched, 1, a).astype(a.dtype)
        _np.multiply.at(base, k, s)
        return Tensor._wrap(base)
    if reduce in ("amax", "amin"):
        fill = (_np.finfo if a.dtype.kind == "f" else _np.iinfo)(a.dtype)
        base = a if include_self else _np.where(touched, fill.min if reduce == "amax" else fill.max, a).astype(a.dtype)
        (_np.maximum if reduce == "amax" else _np.minimum).at(base, k, s)
        return Tensor._wrap(base)
    if reduce == "mean":
        tot = a.astype(_np.float64) if include_self else _np.where(touched, 0, a).astype(_np.float64)
        cnt = _np.where(touched, 1 if include_self else 0, 1).astype(_np.float64)
        _np.add.at(tot, k, s)
        _np.add.at(cnt, k, 1)
        out = tot / cnt
        return Tensor._wrap(out.astype(a.dtype) if a.dtype.kind == "f" else _np.floor(out).astype(a.dtype))
    raise RuntimeError(f"reduce argument must be either sum, prod, mean, amax or amin, got {reduce}")


def _index_fill_(self, dim, index, value):
    out = self._a.copy()
    sl = [slice(None)] * self.ndim
    sl[dim] = _arr(index)
    out[tuple(sl)] = _arr(value)
    return self._inplace(out)


def _index_copy_(self, dim, index, source):
    out = self._a.copy()
    sl = [slice(None)] * self.ndim
    sl[dim] = _arr(index)
    out[tuple(sl)] = _arr(source)
    return self._inplace(out)


def _index_put_(self, indices, values, accumulate=False):
    out = self._a.copy()
    k = tuple(_arr(i) for i in indices)
    if accumulate:
        _np.add.at(out, k, _arr(values))
    else:
        out[k] = _arr(values)
    return self._inplace(out)


def _put_(self, index, source, accumulate=False):
    flat = self._a.reshape(-1).copy()
    if accumulate:
        _np.add.at(flat, _arr(index), _arr(source))
    else:
        flat[_arr(index)] = _arr(source)
    return self._inplace(flat.reshape(self._a.shape))


# ---------------------------------------------------------------- reductions & scans

def _aminmax(self, dim=None, keepdim=False):
    return _MinMax(self.amin(dim=dim, keepdim=keepdim) if dim is not None else self.min(),
                   self.amax(dim=dim, keepdim=keepdim) if dim is not None else self.max())


def _cummin(self, dim):
    vals = _np.minimum.accumulate(self._a, axis=dim)
    n = self._a.shape[dim]
    ar = _np.arange(n).reshape([-1 if i == dim % self.ndim else 1 for i in range(self.ndim)])
    idx = _np.maximum.accumulate(_np.where(self._a == vals, ar, 0), axis=dim)
    return _Tup(Tensor._wrap(vals), Tensor._wrap(idx.astype(_np.int64)))


def _logcumsumexp(self, dim):
    m = self.amax(dim=dim, keepdim=True).detach()
    return (self - m).exp().cumsum(dim).log() + m


def _mode(self, dim=-1, keepdim=False):
    a = _np.moveaxis(self._a, dim, -1)
    flat = a.reshape(-1, a.shape[-1])
    vals, idxs = [], []
    for row in flat:
        u, c = _np.unique(row, return_counts=True)
        v = u[_np.argmax(c)]  # the smallest of the most frequent values
        vals.append(v)
        idxs.append(int(_np.nonzero(row == v)[0][-1]))
    shape = a.shape[:-1]
    v = _np.array(vals, dtype=self._a.dtype).reshape(shape)
    i = _np.array(idxs, dtype=_np.int64).reshape(shape)
    if keepdim:
        v, i = _np.expand_dims(v, dim), _np.expand_dims(i, dim)
    return _Tup(Tensor._wrap(v), Tensor._wrap(i))


def _nanquantile(self, q, dim=None, keepdim=False):
    return Tensor._wrap(_np.asarray(_np.nanquantile(self._a, _arr(q), axis=dim, keepdims=keepdim), dtype=self._a.dtype))


def _histc(self, bins=100, min=0, max=0):
    a = self._a.astype(_np.float64)
    lo, hi = (a.min(), a.max()) if min == 0 and max == 0 else (min, max)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    keep = a[(a >= lo) & (a <= hi)]
    h, _ = _np.histogram(keep, bins=bins, range=(lo, hi))
    return Tensor._wrap(h.astype(self._a.dtype if self.is_floating_point() else _np.float32))


def _unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
    a = self._a.reshape(-1)
    if a.size == 0:
        out = [Tensor._wrap(a)]
    else:
        starts = _np.concatenate([[True], a[1:] != a[:-1]])
        out = [Tensor._wrap(a[starts])]
        if return_inverse:
            out.append(Tensor._wrap((_np.cumsum(starts) - 1).reshape(self._a.shape).astype(_np.int64)))
        if return_counts:
            pos = _np.nonzero(starts)[0]
            out.append(Tensor._wrap(_np.diff(_np.append(pos, a.size)).astype(_np.int64)))
    return out[0] if len(out) == 1 else tuple(out)


def _trapezoid(self, x=None, dx=1, dim=-1):
    y = self
    n = y.shape[dim]
    a, b = y.narrow(dim, 0, n - 1), y.narrow(dim, 1, n - 1)
    if x is None:
        return ((a + b) * (dx / 2)).sum(dim)
    xs = _t(x)
    d = xs.narrow(-1, 1, n - 1) - xs.narrow(-1, 0, n - 1)
    return ((a + b) / 2 * d).sum(dim)


# ---------------------------------------------------------------- rearranging

def _rot90(self, k=1, dims=(0, 1)):
    k %= 4
    d0, d1 = dims
    out = self
    for _ in range(k):
        out = out.flip([d1]).transpose(d0, d1)
    return out


def _fliplr(self):
    return self.flip([1])


def _flipud(self):
    return self.flip([0])


def _tensor_split(self, indices_or_sections, dim=0):
    n = self.shape[dim]
    if isinstance(indices_or_sections, int):
        k = indices_or_sections
        q, r = divmod(n, k)
        sizes = [q + 1] * r + [q] * (k - r)
        cuts = list(_it.accumulate(sizes))[:-1]
    else:
        cuts = list(_arr(indices_or_sections).tolist()) if isinstance(indices_or_sections, Tensor) else list(indices_or_sections)
    bounds = [0] + [_b.min(_b.max(c, 0), n) for c in cuts] + [n]
    return tuple(self.narrow(dim, s, _b.max(e - s, 0)) for s, e in zip(bounds, bounds[1:]))


def _diag_m(self, diagonal=0):
    return torch.diag(self, diagonal)


for _name, _fn in {
    "neg": _neg, "negative": _neg, "logit": _logit, "atan2": _atan2, "arctan2": _atan2, "remainder": _remainder,
    "fmod": _fmod, "hypot": _hypot, "logaddexp": _logaddexp, "xlogy": _xlogy, "addcmul": _addcmul, "addcdiv": _addcdiv,
    "sinh": _sinh, "cosh": _cosh, "erfinv": _erfinv, "exp2": _exp2, "type_as": _type_as, "broadcast_to": _broadcast_to,
    "clamp_min": _clamp_min, "clamp_max": _clamp_max, "take": _take, "take_along_dim": _take_along_dim,
    "scatter_reduce": _scatter_reduce, "index_fill_": _index_fill_, "index_copy_": _index_copy_,
    "index_put_": _index_put_, "put_": _put_, "aminmax": _aminmax, "cummin": _cummin, "logcumsumexp": _logcumsumexp,
    "mode": _mode, "nanquantile": _nanquantile, "histc": _histc, "unique_consecutive": _unique_consecutive,
    "trapezoid": _trapezoid, "rot90": _rot90, "fliplr": _fliplr, "flipud": _flipud, "tensor_split": _tensor_split,
    "diag": _diag_m,
}.items():
    setattr(Tensor, _name, _fn)

for _name in ("neg", "abs", "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "sin", "cos", "floor", "ceil", "round",
              "trunc", "sign", "reciprocal", "square", "logit", "erf", "rsqrt", "log1p", "expm1", "frac", "remainder",
              "fmod", "atan2", "hypot", "addcmul", "addcdiv", "lerp", "nan_to_num", "triu", "tril", "t"):
    if not hasattr(Tensor, _name + "_") and hasattr(Tensor, _name) or _name in ("triu", "tril"):
        if _name in ("triu", "tril"):
            setattr(Tensor, _name + "_", (lambda nm: lambda self, diagonal=0: self._inplace(
                getattr(torch, nm)(self.detach(), diagonal)._a))(_name))
        else:
            setattr(Tensor, _name + "_", _inplace_of(_name))


def _exponential_(self, lambd=1.0, generator=None):
    return self._inplace(torch._gen(generator).exponential(1 / lambd, self._a.shape))


def _random_(self, from_=0, to=None, generator=None):
    if to is None:
        from_, to = 0, (from_ if from_ else (2 ** 24 if self.is_floating_point() else 2 ** 31))
    return self._inplace(torch._gen(generator).integers(from_, to, self._a.shape))


def _geometric_(self, p, generator=None):
    return self._inplace(torch._gen(generator).geometric(p, self._a.shape))


def _bernoulli_(self, p=0.5, generator=None):
    return self._inplace((torch._gen(generator).random(self._a.shape) < _arr(p)).astype(self._a.dtype))


Tensor.exponential_ = _exponential_
Tensor.random_ = _random_
Tensor.geometric_ = _geometric_
Tensor.bernoulli_ = _bernoulli_


# ---------------------------------------------------------------- torch.* functions

def _fn(name):
    def f(t, *a, **k):
        return getattr(t if isinstance(t, Tensor) else torch.tensor(t), name)(*a, **k)
    f.__name__ = name
    return f


_EXPORT = {}
for _name in ("neg", "negative", "logit", "atan2", "arctan2", "remainder", "fmod", "hypot", "logaddexp", "xlogy",
              "addcmul", "addcdiv", "sinh", "cosh", "erfinv", "exp2", "take", "take_along_dim", "scatter_reduce", "aminmax",
              "cummin", "logcumsumexp", "mode", "nanquantile", "histc", "unique_consecutive", "trapezoid", "rot90",
              "fliplr", "flipud", "tensor_split", "clamp_min", "clamp_max", "scatter", "scatter_add", "index_add",
              "index_fill", "index_copy", "gather", "square", "clone", "detach", "conj", "positive", "frac"):
    _EXPORT[_name] = _fn(_name)


def _scatter(t, dim, index, src):
    return t.clone().detach().scatter_(dim, index, src) if not t.requires_grad else t.scatter(dim, index, src)


def add(input, other, alpha=1, out=None):
    r = _t(input) + (other * alpha if alpha != 1 else other)
    return _out(r, out)


def sub(input, other, alpha=1, out=None):
    return _out(_t(input) - (other * alpha if alpha != 1 else other), out)


subtract = sub


def mul(input, other, out=None):
    return _out(_t(input) * other, out)


multiply = mul


def div(input, other, rounding_mode=None, out=None):
    return _out(_t(input).div(other, rounding_mode=rounding_mode), out)


divide = div


def true_divide(input, other):
    return _t(input) / other


def floor_divide(input, other):
    return _t(input) // other


def pow(input, exponent, out=None):  # noqa: A001
    base = input if isinstance(input, Tensor) else torch.tensor(input, dtype=torch.float32 if isinstance(input, float) else None)
    if not isinstance(input, Tensor) and isinstance(exponent, Tensor):
        return exponent.__rpow__(input)
    return _out(base ** exponent, out)


def _out(r, out):
    if out is not None:
        out._a = _np.asarray(r._a, dtype=out._a.dtype).copy() if out._a.shape != r._a.shape else out._a
        out._a[...] = r._a
        return out
    return r


def std_mean(t, dim=None, unbiased=True, keepdim=False, correction=None):
    return t.std(dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction), t.mean(dim=dim, keepdim=keepdim)


def var_mean(t, dim=None, unbiased=True, keepdim=False, correction=None):
    return t.var(dim=dim, unbiased=unbiased, keepdim=keepdim, correction=correction), t.mean(dim=dim, keepdim=keepdim)


def cartesian_prod(*tensors):
    grids = torch.meshgrid(*tensors, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=1) if len(tensors) > 1 else tensors[0]


def combinations(t, r=2, with_replacement=False):
    vals = t.tolist()
    pick = _it.combinations_with_replacement if with_replacement else _it.combinations
    idx = list(pick(range(len(vals)), r))
    if not idx:
        return torch.empty(0, r, dtype=t.dtype)
    return torch.stack([torch.stack([t[i] for i in row]) for row in idx])


def diag_embed(t, offset=0, dim1=-2, dim2=-1):
    n = t.shape[-1] + _b.abs(offset)
    out = _np.zeros(tuple(t.shape[:-1]) + (n, n), dtype=t._a.dtype)
    i = _np.arange(t.shape[-1])
    out[..., i + _b.max(-offset, 0), i + _b.max(offset, 0)] = t._a
    return Tensor._wrap(out)


def diagflat(t, offset=0):
    return torch.diag(t.reshape(-1), offset)


def block_diag(*tensors):
    mats = [_np.atleast_2d(_arr(t)) for t in tensors]
    r, c = _b.sum(m.shape[0] for m in mats), _b.sum(m.shape[1] for m in mats)
    out = _np.zeros((r, c), dtype=_np.result_type(*mats))
    i = j = 0
    for m in mats:
        out[i:i + m.shape[0], j:j + m.shape[1]] = m
        i, j = i + m.shape[0], j + m.shape[1]
    return Tensor._wrap(out)


def kron(a, b):
    return Tensor._wrap(_np.kron(_arr(a), _arr(b)))


def vander(x, N=None, increasing=False):
    return Tensor._wrap(_np.vander(_arr(x), N, increasing=increasing))


def dstack(tensors):
    return Tensor._wrap(_np.dstack([_arr(t) for t in tensors]))


def histogram(t, bins=10, range=None, weight=None, density=False):  # noqa: A002
    h, edges = _np.histogram(_arr(t), bins=bins if not isinstance(bins, Tensor) else _arr(bins), range=range,
                             weights=None if weight is None else _arr(weight), density=density)
    return _nt_hist(Tensor._wrap(h.astype(_np.float32)), Tensor._wrap(edges.astype(_np.float32)))


_nt_hist = __import__("collections").namedtuple("histogram", ["hist", "bin_edges"])


def result_type(a, b):
    def kind(x):
        if isinstance(x, Tensor):
            return x.dtype
        if isinstance(x, bool):
            return torch.bool
        if isinstance(x, int):
            return torch.int64
        return torch.get_default_dtype()
    da, db = kind(a), kind(b)
    if isinstance(a, Tensor) and not isinstance(b, Tensor):
        if da.is_floating_point or not db.is_floating_point:
            return da
        return torch.get_default_dtype()
    if isinstance(b, Tensor) and not isinstance(a, Tensor):
        return result_type(b, a)
    return torch.promote_types(da, db)


def atleast_1d(*ts):
    out = [t if t.ndim >= 1 else t.reshape(1) for t in ts]
    return out[0] if len(out) == 1 else tuple(out)


def atleast_2d(*ts):
    out = [t if t.ndim >= 2 else t.reshape(1, -1) for t in ts]
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*ts):
    out = []
    for t in ts:
        out.append(t if t.ndim >= 3 else t.reshape(1, -1, 1) if t.ndim == 1 else t.reshape(1, 1, 1) if t.ndim == 0 else t[..., None])
    return out[0] if len(out) == 1 else tuple(out)


def empty_strided(size, stride, dtype=None, device=None):
    base = torch.zeros(_b.max(1, 1 + _b.sum((n - 1) * s for n, s in zip(size, stride))), dtype=dtype)
    return torch.as_strided(base, size, stride)


def addmm(input, mat1, mat2, beta=1, alpha=1):
    return beta * input + alpha * (mat1 @ mat2)


def baddbmm(input, b1, b2, beta=1, alpha=1):
    return beta * input + alpha * (b1 @ b2)


def addmv(input, mat, vec, beta=1, alpha=1):
    return beta * input + alpha * (mat @ vec)


def tensordot(a, b, dims=2):
    if isinstance(dims, int):
        ax_a, ax_b = list(range(a.ndim - dims, a.ndim)), list(range(dims))
    else:
        ax_a, ax_b = [list(d) if isinstance(d, (list, tuple)) else [d] for d in dims]
    la = "abcdefghijklm"[:a.ndim]
    lb = list("nopqrstuvwxyz"[:b.ndim])
    for i, j in zip(ax_a, ax_b):
        lb[j] = la[i]
    lb = "".join(lb)
    out = "".join(c for c in la if c not in lb) + "".join(c for c in lb if c not in la)
    return torch.einsum(f"{la},{lb}->{out}", a, b)


def cross(a, b, dim=None):
    return torch.linalg.cross(a, b, dim=-1 if dim is None else dim)


def dist(a, b, p=2):
    return (a - b).norm(p=p)


def chain_matmul(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = out @ m
    return out


def asarray(obj, dtype=None, device=None, copy=None, requires_grad=False):
    if isinstance(obj, Tensor):
        return obj.to(dtype) if dtype is not None else obj
    t = torch.as_tensor(obj, dtype=dtype)
    return t.requires_grad_(True) if requires_grad else t


def scatter(t, dim, index, src, reduce=None):
    return t.detach().clone().scatter_(dim, index, src)


def scatter_add(t, dim, index, src):
    return t.detach().clone().scatter_add_(dim, index, src)


def index_add(t, dim, index, source, alpha=1):
    return t.detach().clone().index_add_(dim, index, source * alpha if alpha != 1 else source)


def index_fill(t, dim, index, value):
    return t.detach().clone().index_fill_(dim, index, value)


def index_copy(t, dim, index, source):
    return t.detach().clone().index_copy_(dim, index, source)


def index_put(t, indices, values, accumulate=False):
    return t.detach().clone().index_put_(indices, values, accumulate)


for _name in ("scatter", "scatter_add", "index_add", "index_fill", "index_copy", "index_put"):
    if not hasattr(Tensor, _name):
        setattr(Tensor, _name, (lambda f: lambda self, *a, **k: f(self, *a, **k))(globals()[_name]))
    _EXPORT[_name] = globals()[_name]

for _name in ("add", "sub", "subtract", "mul", "multiply", "div", "divide", "true_divide", "floor_divide", "pow", "std_mean",
              "var_mean", "cartesian_prod", "combinations", "diag_embed", "diagflat", "block_diag", "kron", "vander", "dstack",
              "histogram", "result_type", "atleast_1d", "atleast_2d", "atleast_3d", "empty_strided", "addmm", "baddbmm",
              "addmv", "tensordot", "cross", "dist", "chain_matmul", "asarray"):
    _EXPORT[_name] = globals()[_name]

for _name, _f in _EXPORT.items():
    setattr(torch, _name, _f)

for _name in ("positive", "conj", "resolve_conj", "resolve_neg"):
    if not hasattr(Tensor, _name):
        setattr(Tensor, _name, lambda self: self)

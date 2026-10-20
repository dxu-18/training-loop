"""torch-lite: a NumPy-backed stand-in for the parts of PyTorch the practice site uses.

The browser (Pyodide) has no PyTorch build, so `generator/__init__.py` puts this package on
sys.path when real torch isn't installed. It follows PyTorch's semantics for what it covers:
default float32 / int64 dtypes, broadcasting, `dim=` / `keepdim=`, `(values, indices)` from
`max(dim=...)`, unbiased `std`, views that share memory (strides, `view` vs `reshape`,
`contiguous`), and reverse-mode autograd with `.grad` accumulation, hooks and `no_grad()`.
Random numbers are reproducible under `manual_seed` but differ from real PyTorch's.
Anything else raises a clear "not available in the browser" error.
"""

from __future__ import annotations

import builtins as _b
import contextlib as _ctx
import math as _math
import pickle as _pickle
from collections import namedtuple as _nt

import numpy as _np

__version__ = "2.6.0+lite"
LITE = True
_Tup = _nt("return_types", ["values", "indices"])


# ---------------------------------------------------------------- dtypes

class dtype:
    def __init__(self, name, np_dtype, is_floating_point, itemsize=None):
        self.name, self.np, self.is_floating_point = name, _np.dtype(np_dtype), is_floating_point
        self.itemsize = itemsize or self.np.itemsize

    def __repr__(self):
        return f"torch.{self.name}"


float32 = float = dtype("float32", _np.float32, True)
float64 = double = dtype("float64", _np.float64, True)
float16 = half = dtype("float16", _np.float16, True)
bfloat16 = dtype("bfloat16", _np.float32, True, itemsize=2)  # emulated: float32 storage rounded to 8 mantissa bits
int64 = long = dtype("int64", _np.int64, False)
int32 = int = dtype("int32", _np.int32, False)
int16 = short = dtype("int16", _np.int16, False)
int8 = dtype("int8", _np.int8, False)
uint8 = dtype("uint8", _np.uint8, False)
bool = dtype("bool", _np.bool_, False)
_BY_NP = {d.np: d for d in (float32, float64, float16, int64, int32, int16, int8, uint8, bool)}
contiguous_format, channels_last, preserve_format = "contiguous_format", "channels_last", "preserve_format"


def _dt(t):
    return bfloat16 if t._bf16 else _BY_NP.get(t._a.dtype, float32)


def _to_bf16(a):
    """Round float32 values to bfloat16 precision (round-to-nearest-even), keeping float32 storage."""
    u = _np.ascontiguousarray(a, dtype=_np.float32).view(_np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(_np.float32)


def get_default_dtype():
    return float32


def set_default_dtype(d):
    if d is not float32:
        raise RuntimeError("torch-lite only supports float32 as the default dtype")


class finfo:
    def __init__(self, d=float32):
        if d is bfloat16:
            self.bits, self.eps, self.max, self.min, self.tiny = 16, 0.0078125, 3.3895313892515355e38, -3.3895313892515355e38, 1.1754943508222875e-38
        else:
            f = _np.finfo(d.np)
            self.bits, self.eps, self.max, self.min, self.tiny = f.bits, _b.float(f.eps), _b.float(f.max), _b.float(f.min), _b.float(f.tiny)
        self.dtype = d.name
        self.resolution = 10 ** -_b.int(_math.floor(-_math.log10(self.eps)))
        self.smallest_normal = self.tiny


class iinfo:
    def __init__(self, d):
        i = _np.iinfo(d.np)
        self.bits, self.max, self.min, self.dtype = i.bits, _b.int(i.max), _b.int(i.min), d.name


def promote_types(a, b):
    if bfloat16 in (a, b):
        other = b if a is bfloat16 else a
        return other if other in (float32, float64) else bfloat16
    return _BY_NP[_np.promote_types(a.np, b.np)]


def can_cast(src, dst):
    return _b.bool(_np.can_cast(src.np, dst.np, "same_kind")) and not (src.is_floating_point and not dst.is_floating_point)


class Size(tuple):
    def numel(self):
        return _math.prod(self)

    def __repr__(self):
        return f"torch.Size({list(self)})"


class device:
    def __init__(self, type_="cpu", index=None):
        type_ = str(type_)
        self.type, _, idx = type_.partition(":")
        self.index = _b.int(idx) if idx else index

    def __repr__(self):
        return f"device(type='{self.type}'" + (f", index={self.index})" if self.index is not None else ")")

    def __eq__(self, other):
        return str(self) == str(device(other) if isinstance(other, str) else other)

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _check_device(dev):
    if dev is not None and str(dev).split(":")[0] not in ("cpu", "meta"):
        raise RuntimeError("Torch not compiled with CUDA enabled (torch-lite runs on the CPU only)")


# ---------------------------------------------------------------- grad mode & autocast

_GRAD = [True]
_AUTOCAST = [None]  # dtype while inside torch.autocast(...)


@_ctx.contextmanager
def _grad_mode(on):
    old = _GRAD[0]
    _GRAD[0] = on
    try:
        yield
    finally:
        _GRAD[0] = old


class no_grad(_ctx.ContextDecorator):
    _on = False

    def __enter__(self):
        self._cm = _grad_mode(self._on)
        self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


class enable_grad(no_grad):
    _on = True


class inference_mode(no_grad):
    def __init__(self, mode=True):
        self._on = not mode


class set_grad_enabled(no_grad):
    def __init__(self, mode):
        self._on = mode
        self._prev = _GRAD[0]
        _GRAD[0] = mode

    def __enter__(self):
        _GRAD[0] = self._on

    def __exit__(self, *exc):
        _GRAD[0] = self._prev
        return False


def is_grad_enabled():
    return _GRAD[0]


class autocast(_ctx.ContextDecorator):
    def __init__(self, device_type="cpu", dtype=None, enabled=True, cache_enabled=None):
        self.dtype = dtype or (bfloat16 if device_type == "cpu" else float16)
        self.enabled = enabled

    def __enter__(self):
        self._prev = _AUTOCAST[0]
        _AUTOCAST[0] = self.dtype if self.enabled else None
        return self

    def __exit__(self, *exc):
        _AUTOCAST[0] = self._prev
        return False


def is_autocast_enabled(device_type=None):
    return _AUTOCAST[0] is not None


# ---------------------------------------------------------------- profiler hook

_PROF = []  # active profilers (see torch.profiler)


_PROF_MUTE = [0]  # >0 while a composite op (e.g. F.linear) records its own name instead of its parts


def _event(name):
    if _PROF_MUTE[0]:
        return
    for p in _PROF:
        p._record(name)


# ---------------------------------------------------------------- helpers

def _arr(x):
    return x._a if isinstance(x, Tensor) else _np.asarray(x)


def _new(data, dtype_=None):
    """data -> numpy array with torch's default dtypes (float32, int64)."""
    a = _np.asarray(data)
    if dtype_ is not None:
        d = dtype_.np if isinstance(dtype_, dtype) else _np.dtype(dtype_)
        return a.astype(d)
    if a.dtype.kind == "f" and a.dtype == _np.float64:
        return a.astype(_np.float32)
    if a.dtype.kind in "iu" and a.dtype != _np.uint8:
        return a.astype(_np.int64)
    if a.dtype.kind in "OUS":
        raise TypeError(f"can't convert {a.dtype} data to a tensor")
    return a


def _promote(res, ops):
    """numpy promotes int/py-float math to float64; torch keeps the default float32."""
    if res.dtype == _np.float64 and not _b.any(isinstance(o, Tensor) and o._a.dtype == _np.float64 for o in ops):
        return res.astype(_np.float32)
    return res


def _unbroadcast(g, shape):
    g = _np.asarray(g)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _make(a, parents=(), bf16=False):
    """Wrap an op result; parents = [(tensor, fn(upstream_grad) -> grad)] build the graph."""
    out = Tensor.__new__(Tensor)
    a = _np.asarray(a)
    if bf16 and a.dtype == _np.float32:
        a = _to_bf16(a)
    out._init(a)
    out._bf16 = bf16 and a.dtype == _np.float32
    live = [(p, f) for p, f in parents if isinstance(p, Tensor) and p.requires_grad]
    if _GRAD[0] and live:
        out.requires_grad, out.is_leaf, out._parents = True, False, live
    return out


def _bf(*ops):
    return _b.any(isinstance(o, Tensor) and o._bf16 for o in ops)


def _ax(dim, ndim):
    if dim is None:
        return None
    if isinstance(dim, (tuple, list)):
        return tuple(d % _b.max(ndim, 1) for d in dim)
    return dim % _b.max(ndim, 1)


def _expand_back(g, shape, axis, keepdim):
    if axis is not None and not keepdim:
        g = _np.expand_dims(g, axis)
    return _np.broadcast_to(g, shape)


def _unsupported(name):
    raise AttributeError(f"'{name}' is not available in the browser's torch-lite: use the functions shown in "
                         f"the problem and the example.")


def _size_args(args):
    if len(args) == 1 and isinstance(args[0], (tuple, list, Size)):
        return tuple(args[0])
    return tuple(args)


# ---------------------------------------------------------------- Tensor

class Tensor:
    __torch_lite__ = True
    __array_priority__ = 1000

    def _init(self, a):
        self._a = _np.asarray(a)
        self._bf16 = False
        self.requires_grad = False
        self.grad = None
        self.is_leaf = True
        self._parents = ()
        self._retain = False
        self._hooks = []
        self._version = 0

    def __init__(self, *args):
        self._init(_np.zeros(args, dtype=_np.float32) if args and isinstance(args[0], _b.int)
                   else _new(args[0] if args else [], float32))

    def __reduce__(self):
        a = _np.ascontiguousarray(self._a)
        return _rebuild, ((a.tobytes(), a.dtype.str, a.shape), self.requires_grad and self.is_leaf, self._bf16)

    # ---- attributes
    shape = property(lambda self: Size(self._a.shape))
    ndim = property(lambda self: self._a.ndim)
    dtype = property(lambda self: _dt(self))
    device = property(lambda self: device("cpu"))
    is_cuda = property(lambda self: False)
    nbytes = property(lambda self: self._a.size * self.dtype.itemsize)
    itemsize = property(lambda self: self.dtype.itemsize)
    T = property(lambda self: self.permute(*range(self.ndim - 1, -1, -1)))
    mT = property(lambda self: self.transpose(-2, -1))
    layout = property(lambda self: "torch.strided")

    @property
    def data(self):
        t = Tensor._wrap(self._a)
        t._bf16 = self._bf16
        return t

    @data.setter
    def data(self, other):
        self._a = _arr(other)

    def size(self, dim=None):
        return self.shape if dim is None else self._a.shape[dim]

    def dim(self):
        return self._a.ndim

    def numel(self):
        return _b.int(self._a.size)

    nelement = numel

    def element_size(self):
        return self.dtype.itemsize

    def is_floating_point(self):
        return self._a.dtype.kind == "f"

    def stride(self, dim=None):
        s = tuple(_b.int(x // self._a.itemsize) for x in self._a.strides)
        return s if dim is None else s[dim]

    def is_contiguous(self, memory_format=contiguous_format):
        if memory_format == channels_last:
            if self.ndim != 4:
                return False
            n, c, h, w = self._a.shape
            s = self.stride()
            return s[1] == 1 and s[3] == c and s[2] == c * w
        expected, acc = [], 1
        for n in reversed(self._a.shape):
            expected.append(acc)
            acc *= n
        return _b.all(n == 1 or s == e for n, s, e in zip(self._a.shape, self.stride(), reversed(expected)))

    def data_ptr(self):
        return self._a.__array_interface__["data"][0]

    def storage_offset(self):
        base = self._a
        while isinstance(base.base, _np.ndarray):
            base = base.base
        return (self.data_ptr() - base.__array_interface__["data"][0]) // self._a.itemsize

    def contiguous(self, memory_format=contiguous_format):
        if memory_format == channels_last:
            return self.to(memory_format=channels_last)
        if self.is_contiguous():
            return self
        return _make(_np.ascontiguousarray(self._a), [(self, lambda g: g)], self._bf16)

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return len(self._a)

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(len(self)))

    def __repr__(self):
        a = self._a
        if a.dtype.kind == "f" and a.size:
            body = _np.array2string(a, separator=", ", formatter={"float_kind": _float_formatter(a)}, prefix="tensor(",
                                    threshold=1000, edgeitems=3)
        else:
            body = _np.array2string(a, separator=", ", prefix="tensor(", threshold=1000, edgeitems=3)
        extra = "" if self.dtype in (float32, int64, bool) else f", dtype={self.dtype}"
        grad = ", requires_grad=True" if self.requires_grad and self.is_leaf else (", grad_fn=<LiteBackward>" if self.requires_grad else "")
        return f"tensor({body}{extra}{grad})"

    __str__ = __repr__

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        _unsupported(f"Tensor.{name}")

    # ---- conversion
    def item(self):
        if self._a.size != 1:
            raise RuntimeError(f"a Tensor with {self._a.size} elements cannot be converted to Scalar")
        return self._a.reshape(()).item()

    def tolist(self):
        return self._a.tolist()

    def numpy(self, force=False):
        if self.requires_grad and not force:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self._a

    def __array__(self, dtype=None, copy=None):
        return self._a if dtype is None else self._a.astype(dtype)

    def __bool__(self):
        if self._a.size != 1:
            raise RuntimeError(f"Boolean value of Tensor with more than one value is ambiguous")
        return _b.bool(self.item())

    def __float__(self):
        return _b.float(self.item())

    def __int__(self):
        return _b.int(self.item())

    def __index__(self):
        if self._a.dtype.kind not in "iub":
            raise TypeError("only integer tensors of a single element can be converted to an index")
        return _b.int(self.item())

    def __format__(self, spec):
        return format(self.item(), spec) if self._a.size == 1 else repr(self)

    def detach(self):
        t = Tensor._wrap(self._a)
        t._bf16 = self._bf16
        return t

    def detach_(self):
        self.requires_grad, self._parents, self.is_leaf = False, (), True
        return self

    def clone(self, memory_format=None):
        return _make(self._a.copy(), [(self, lambda g: g)], self._bf16)

    def to(self, *args, dtype=None, device=None, non_blocking=False, copy=False, memory_format=None):
        for a in args:
            if isinstance(a, globals()["dtype"]):
                dtype = a
            elif isinstance(a, (str, globals()["device"])):
                device = a
            elif isinstance(a, Tensor):
                dtype = a.dtype
        _check_device(device)
        out = self if dtype is None else self._cast(dtype)
        if memory_format == channels_last and out.ndim == 4:
            nhwc = _np.ascontiguousarray(out._a.transpose(0, 2, 3, 1))
            out = _make(nhwc.transpose(0, 3, 1, 2), [(out, lambda g: g)], out._bf16)
        return out.clone() if copy and out is self else out

    def _cast(self, d):
        if d is bfloat16:
            return _make(self._a.astype(_np.float32), [(self, lambda g: g)], True)
        if d is self.dtype:
            return self
        with _np.errstate(all="ignore"):  # overflow to inf, like torch
            a = self._a.astype(d.np)
        return _make(a, [(self, lambda g: g)] if d.is_floating_point else [])

    def type(self, d=None):
        return f"torch.{self.dtype.name.capitalize()}Tensor" if d is None else self._cast(d)

    float = lambda self: self._cast(float32)  # noqa: E731
    double = lambda self: self._cast(float64)  # noqa: E731
    half = lambda self: self._cast(float16)  # noqa: E731
    bfloat16 = lambda self: self._cast(bfloat16)  # noqa: E731
    long = lambda self: self._cast(int64)  # noqa: E731
    int = lambda self: self._cast(int32)  # noqa: E731
    bool = lambda self: self._cast(bool)  # noqa: E731

    def cpu(self):
        return self

    def cuda(self, *a, **k):
        _check_device("cuda")

    def pin_memory(self):
        return self

    def new_zeros(self, *size, dtype=None):
        return zeros(*size, dtype=dtype or self.dtype)

    def new_ones(self, *size, dtype=None):
        return ones(*size, dtype=dtype or self.dtype)

    def new_full(self, size, fill_value, dtype=None):
        return full(size, fill_value, dtype=dtype or self.dtype)

    def new_empty(self, *size, dtype=None):
        return zeros(*size, dtype=dtype or self.dtype)

    def new_tensor(self, data, dtype=None):
        return tensor(data, dtype=dtype or self.dtype)

    # ---- autograd
    def requires_grad_(self, flag=True):
        if flag and not self.is_floating_point():
            raise RuntimeError("only Tensors of floating point dtype can require gradients")
        self.requires_grad = flag
        return self

    def retain_grad(self):
        self._retain = True

    @property
    def grad_fn(self):
        return None if self.is_leaf else "<LiteBackward>"

    def register_hook(self, fn):
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
        self._hooks.append(fn)
        return _Handle(self._hooks, fn)

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        if gradient is None:
            if self._a.size != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            gradient = _np.ones_like(self._a)
        for t, g in _backprop(self, _arr(gradient)).items():
            if (t.is_leaf and t.requires_grad) or t._retain:
                with _np.errstate(all="ignore"):
                    g = _np.asarray(g, dtype=t._a.dtype if t._a.dtype.kind == "f" else _np.float32)
                if t.grad is None:
                    t.grad = Tensor._wrap(g.copy())
                    t.grad._bf16 = t._bf16
                else:
                    t.grad._a = t.grad._a + g

    @staticmethod
    def _wrap(a):
        t = Tensor.__new__(Tensor)
        t._init(a)
        return t

    # ---- in-place (writes into the same memory, so views see the change)
    def _inplace(self, a):
        if self.requires_grad and self.is_leaf and _GRAD[0]:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        a = _np.asarray(a)
        if self._bf16:
            a = _to_bf16(a)
        if a.dtype.kind == "f" and self._a.dtype.kind in "iub":
            raise RuntimeError(f"result type Float can't be cast to the desired output type {self.dtype.name.capitalize()}")
        self._a[...] = a
        self._version += 1
        return self

    def zero_(self):
        return self._inplace(_np.zeros_like(self._a))

    def fill_(self, v):
        return self._inplace(_np.full_like(self._a, _arr(v)))

    def add_(self, o, alpha=1):
        return self._inplace(self._a + alpha * _arr(o))

    def sub_(self, o, alpha=1):
        return self._inplace(self._a - alpha * _arr(o))

    def mul_(self, o):
        return self._inplace(self._a * _arr(o))

    def div_(self, o):
        return self._inplace(self._a / _arr(o))

    def pow_(self, o):
        return self._inplace(self._a ** _arr(o))

    def clamp_(self, min=None, max=None):
        return self._inplace(_np.clip(self._a, min, max))

    def copy_(self, o):
        return self._inplace(_np.broadcast_to(_arr(o), self._a.shape))

    def uniform_(self, a=0.0, b=1.0):
        return self._inplace(_RNG[0].uniform(a, b, self._a.shape))

    def normal_(self, mean=0.0, std=1.0):
        return self._inplace(_RNG[0].normal(mean, std, self._a.shape))

    def masked_fill_(self, mask, value):
        return self._inplace(_np.where(_arr(mask), value, self._a))

    def scatter_(self, dim, index, src):
        out = self._a.copy()
        idx = _arr(index)
        vals = _np.broadcast_to(_arr(src), idx.shape)
        _np.put_along_axis(out, idx, vals, dim)
        return self._inplace(out)

    def scatter_add_(self, dim, index, src):
        out = self._a.copy()
        idx = _arr(index)
        grids = list(_np.indices(idx.shape))
        grids[dim % self.ndim] = idx
        _np.add.at(out, tuple(grids), _np.broadcast_to(_arr(src), idx.shape))
        return self._inplace(out)

    def index_add_(self, dim, index, source):
        out = self._a.copy()
        sl = [slice(None)] * self.ndim
        sl[dim] = _arr(index)
        _np.add.at(out, tuple(sl), _arr(source))
        return self._inplace(out)

    def __iadd__(self, o):
        return self.add_(o)

    def __isub__(self, o):
        return self.sub_(o)

    def __imul__(self, o):
        return self.mul_(o)

    def __itruediv__(self, o):
        return self.div_(o)

    def __setitem__(self, key, value):
        if self.requires_grad and self.is_leaf and _GRAD[0]:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        self._a[_key(key)] = _arr(value)
        self._version += 1

    # ---- indexing
    def __getitem__(self, key):
        k = _key(key)

        def back(g, shape=self._a.shape, k=k):
            out = _np.zeros(shape, dtype=_np.asarray(g).dtype)
            _np.add.at(out, k, g)
            return out

        return _make(self._a[k], [(self, back)], self._bf16)

    # ---- arithmetic
    def _bin(self, o, fn, ga, gb, name):
        _event(name)
        a, b = self._a, _arr(o)
        with _np.errstate(all="ignore"):  # torch never warns about inf/nan results
            res = _promote(_np.asarray(fn(a, b)), (self, o))
        return _make(res, [(self, lambda g: _unbroadcast(ga(g, a, b), a.shape)),
                           (o, lambda g: _unbroadcast(gb(g, a, b), _np.shape(b)))], _bf(self, o))

    def __add__(self, o, alpha=1):
        if alpha != 1:
            o = o * alpha
        return self._bin(o, _np.add, lambda g, a, b: g, lambda g, a, b: g, "aten::add")

    def __sub__(self, o, alpha=1):
        if alpha != 1:
            o = o * alpha
        return self._bin(o, _np.subtract, lambda g, a, b: g, lambda g, a, b: -g, "aten::sub")

    def __mul__(self, o):
        return self._bin(o, _np.multiply, lambda g, a, b: g * b, lambda g, a, b: g * a, "aten::mul")

    def __truediv__(self, o):
        return self._bin(o, _np.true_divide, lambda g, a, b: g / b, lambda g, a, b: -g * a / b ** 2, "aten::div")

    def __pow__(self, o):
        return self._bin(o, _np.power, lambda g, a, b: g * b * _np.power(a, b - 1),
                         lambda g, a, b: g * _np.log(_np.where(a > 0, a, 1)) * _np.power(a, b), "aten::pow")

    def __floordiv__(self, o):
        return _make(_np.floor_divide(self._a, _arr(o)))

    def __mod__(self, o):
        return _make(_np.mod(self._a, _arr(o)))

    def __radd__(self, o):
        return self + o

    def __rsub__(self, o):
        return -self + o

    def __rmul__(self, o):
        return self * o

    def __rtruediv__(self, o):
        return _const(o, self) / self

    def __rpow__(self, o):
        return _const(o, self) ** self

    def __neg__(self):
        return _make(-self._a, [(self, lambda g: -g)], self._bf16)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __matmul__(self, o):
        _event("aten::matmul")
        a, b = self._a, _arr(o)
        if a.ndim == 0 or _np.ndim(b) == 0:
            raise RuntimeError("both arguments to matmul need to be at least 1D")
        if a.shape[-1] != _np.shape(b)[-2 if _np.ndim(b) > 1 else -1]:
            raise RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({'x'.join(map(str, a.shape))} and "
                               f"{'x'.join(map(str, _np.shape(b)))})")
        low = _AUTOCAST[0]
        res = _np.asarray(a.astype(low.np) @ b.astype(low.np)) if low is not None and a.dtype.kind == "f" else _np.asarray(a @ b)

        def ga(g):
            if b.ndim == 1:
                return _unbroadcast(_np.multiply.outer(g, b) if a.ndim == 1 else g[..., None] * b, a.shape)
            return _unbroadcast(g @ _np.swapaxes(b, -1, -2) if a.ndim > 1 else g @ b.T, a.shape)

        def gb(g):
            if a.ndim == 1:
                return _unbroadcast(_np.multiply.outer(a, g) if b.ndim > 1 else g * a, b.shape)
            if b.ndim == 1:
                return _unbroadcast((_np.swapaxes(a, -1, -2) @ g[..., None])[..., 0], b.shape)
            return _unbroadcast(_np.swapaxes(a, -1, -2) @ g, b.shape)

        return _make(res, [(self, ga), (o, gb)], (low is bfloat16) or _bf(self, o))

    def __rmatmul__(self, o):
        return _const(o, self) @ self

    def _cmp(self, o, fn):
        return _make(fn(self._a, _arr(o)))

    def __eq__(self, o):
        return self._cmp(o, _np.equal)

    def __ne__(self, o):
        return self._cmp(o, _np.not_equal)

    def __lt__(self, o):
        return self._cmp(o, _np.less)

    def __le__(self, o):
        return self._cmp(o, _np.less_equal)

    def __gt__(self, o):
        return self._cmp(o, _np.greater)

    def __ge__(self, o):
        return self._cmp(o, _np.greater_equal)

    def __and__(self, o):
        return self._cmp(o, _np.logical_and if self.dtype is bool else _np.bitwise_and)

    def __or__(self, o):
        return self._cmp(o, _np.logical_or if self.dtype is bool else _np.bitwise_or)

    def __xor__(self, o):
        return self._cmp(o, _np.logical_xor if self.dtype is bool else _np.bitwise_xor)

    def __invert__(self):
        return _make(_np.logical_not(self._a) if self._a.dtype == _np.bool_ else ~self._a)

    __hash__ = object.__hash__

    def eq(self, o):
        return self == o

    def ne(self, o):
        return self != o

    def lt(self, o):
        return self < o

    def le(self, o):
        return self <= o

    def gt(self, o):
        return self > o

    def ge(self, o):
        return self >= o

    def add(self, o, alpha=1):
        return self.__add__(o, alpha)

    def sub(self, o, alpha=1):
        return self.__sub__(o, alpha)

    def mul(self, o):
        return self * o

    def div(self, o, rounding_mode=None):
        if rounding_mode == "floor":
            return self // o
        if rounding_mode == "trunc":
            return _make(_np.trunc(self._a / _arr(o)).astype(self._a.dtype))
        return self / o

    def pow(self, o):
        return self ** o

    def matmul(self, o):
        return self @ o

    def mm(self, o):
        if self.ndim != 2 or _arr(o).ndim != 2:
            raise RuntimeError("self must be a matrix")
        return self @ o

    def bmm(self, o):
        if self.ndim != 3 or _arr(o).ndim != 3:
            raise RuntimeError("batch1 must be a 3D tensor")
        return self @ o

    def dot(self, o):
        if self.ndim != 1 or _arr(o).ndim != 1:
            raise RuntimeError("1D tensors expected")
        return self @ o

    def mv(self, v):
        return self @ v

    def t(self):
        if self.ndim > 2:
            raise RuntimeError("t() expects a tensor with <= 2 dimensions")
        return self.T if self.ndim == 2 else self

    def equal(self, o):
        return equal(self, o)

    def allclose(self, o, rtol=1e-5, atol=1e-8):
        return allclose(self, o, rtol, atol)

    def logical_not(self):
        return _make(_np.logical_not(self._a))

    def isnan(self):
        return _make(_np.isnan(self._a))

    def isinf(self):
        return _make(_np.isinf(self._a))

    def isfinite(self):
        return _make(_np.isfinite(self._a))

    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        return _make(_np.nan_to_num(self._a, nan=nan, posinf=posinf, neginf=neginf))

    # ---- elementwise functions
    def _un(self, fn, grad=None, name=None):
        if name:
            _event(name)
        with _np.errstate(all="ignore"):
            res = _promote(_np.asarray(fn(self._a)), (self,))
        if self._a.dtype.kind in "iub" and res.dtype.kind == "f":
            res = res.astype(_np.float32)
        return _make(res, [(self, lambda g: grad(g, self._a, res))] if grad else [], self._bf16)

    def exp(self):
        return self._un(_np.exp, lambda g, x, y: g * y, "aten::exp")

    def log(self):
        return self._un(_np.log, lambda g, x, y: g / x, "aten::log")

    def log1p(self):
        return self._un(_np.log1p, lambda g, x, y: g / (1 + x))

    def log2(self):
        return self._un(_np.log2, lambda g, x, y: g / (x * _np.log(2)))

    def expm1(self):
        return self._un(_np.expm1, lambda g, x, y: g * (y + 1))

    def sqrt(self):
        return self._un(_np.sqrt, lambda g, x, y: g / (2 * y))

    def rsqrt(self):
        return self._un(lambda x: 1 / _np.sqrt(x), lambda g, x, y: -0.5 * g * y ** 3)

    def abs(self):
        return self._un(_np.abs, lambda g, x, y: g * _np.sign(x))

    def sin(self):
        return self._un(_np.sin, lambda g, x, y: g * _np.cos(x))

    def cos(self):
        return self._un(_np.cos, lambda g, x, y: -g * _np.sin(x))

    def tanh(self):
        return self._un(_np.tanh, lambda g, x, y: g * (1 - y ** 2), "aten::tanh")

    def sigmoid(self):
        return self._un(lambda x: 1 / (1 + _np.exp(-x)), lambda g, x, y: g * y * (1 - y), "aten::sigmoid")

    def relu(self):
        return self._un(lambda x: _np.maximum(x, 0).astype(x.dtype), lambda g, x, y: g * (x > 0), "aten::relu")

    def erf(self):
        return self._un(_erf, lambda g, x, y: g * 2 / _math.sqrt(_math.pi) * _np.exp(-x ** 2))

    def square(self):
        return self ** 2

    def reciprocal(self):
        return 1 / self

    def sign(self):
        return self._un(_np.sign)

    def floor(self):
        return self._un(_np.floor)

    def ceil(self):
        return self._un(_np.ceil)

    def trunc(self):
        return self._un(_np.trunc)

    def frac(self):
        return self - self.trunc().detach()

    def round(self, decimals=0):
        return self._un(lambda x: _np.round(x, decimals))

    def clamp(self, min=None, max=None):
        return self._un(lambda x: _np.clip(x, _arr(min) if min is not None else None, _arr(max) if max is not None else None),
                        lambda g, x, y: g * (y == x))

    clip = clamp

    def masked_fill(self, mask, value):
        m = _arr(mask).astype(_b.bool)
        v = _arr(value).astype(self._a.dtype) if isinstance(value, Tensor) else _np.asarray(value, dtype=self._a.dtype)
        return _make(_np.where(m, v, self._a), [(self, lambda g: _unbroadcast(_np.where(m, 0, g), self._a.shape))], self._bf16)

    def where(self, cond, other):
        return where(cond, self, other)

    def lerp(self, end, weight):
        return self + (end - self) * weight

    # ---- shapes and views
    def view(self, *shape):
        if shape and isinstance(shape[0], globals()["dtype"]):
            return Tensor._wrap(self._a.view(shape[0].np))
        shape = _size_args(shape)
        v = self._a.view()
        try:
            v.shape = shape
        except AttributeError:
            raise RuntimeError("view size is not compatible with input tensor's size and stride (at least one dimension "
                               "spans across two contiguous subspaces). Use .reshape(...) instead.") from None
        except ValueError:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {self._a.size}") from None
        return _make(v, [(self, lambda g: _np.asarray(g).reshape(self._a.shape))], self._bf16)

    def reshape(self, *shape):
        shape = _size_args(shape)
        try:
            r = self._a.reshape(shape)
        except ValueError:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {self._a.size}") from None
        return _make(r, [(self, lambda g: _np.asarray(g).reshape(self._a.shape))], self._bf16)

    def view_as(self, other):
        return self.view(other.shape)

    def reshape_as(self, other):
        return self.reshape(other.shape)

    def flatten(self, start_dim=0, end_dim=-1):
        if self.ndim == 0:
            return self.reshape(1)
        s, e = start_dim % self.ndim, end_dim % self.ndim
        sh = self._a.shape
        return self.reshape(sh[:s] + (_math.prod(sh[s:e + 1]),) + sh[e + 1:])

    ravel = lambda self: self.reshape(-1)  # noqa: E731

    def unflatten(self, dim, sizes):
        dim = dim % self.ndim
        sh = self._a.shape
        return self.reshape(sh[:dim] + tuple(sizes) + sh[dim + 1:])

    def unsqueeze(self, dim):
        return self.reshape(_np.expand_dims(self._a, dim).shape)

    def squeeze(self, dim=None):
        if dim is None:
            return self.reshape(tuple(n for n in self._a.shape if n != 1))
        dims = {d % _b.max(self.ndim, 1) for d in (dim if isinstance(dim, (tuple, list)) else (dim,))}
        return self.reshape(tuple(n for i, n in enumerate(self._a.shape) if not (i in dims and n == 1)))

    def permute(self, *dims):
        dims = _size_args(dims)
        dims = tuple(d % self.ndim for d in dims)
        inv = _np.argsort(dims)
        return _make(self._a.transpose(dims), [(self, lambda g: _np.asarray(g).transpose(inv))], self._bf16)

    def transpose(self, d0, d1):
        order = list(range(self.ndim))
        order[d0], order[d1] = order[d1], order[d0]
        return self.permute(*order)

    swapaxes = swapdims = transpose

    def movedim(self, src, dst):
        order = list(range(self.ndim))
        order.insert(dst % self.ndim, order.pop(src % self.ndim))
        return self.permute(*order)

    moveaxis = movedim

    def expand(self, *sizes):
        sizes = _size_args(sizes)
        lead = (1,) * (len(sizes) - self.ndim) + self._a.shape
        full_ = tuple(o if s == -1 else s for s, o in zip(sizes, lead))
        return _make(_np.broadcast_to(self._a, full_), [(self, lambda g: _unbroadcast(g, self._a.shape))], self._bf16)

    def expand_as(self, other):
        return self.expand(*other.shape)

    def repeat(self, *reps):
        reps = _size_args(reps)
        a = self._a.reshape((1,) * (len(reps) - self.ndim) + self._a.shape)

        def back(g):
            g = _np.asarray(g).reshape(tuple(x for r, s in zip(reps, a.shape) for x in (r, s)))
            return g.sum(axis=tuple(range(0, g.ndim, 2))).reshape(self._a.shape)

        return _make(_np.tile(a, reps), [(self, back)], self._bf16)

    def repeat_interleave(self, repeats, dim=None):
        if dim is None:
            return _make(_np.repeat(self._a.reshape(-1), _arr(repeats)))
        return _make(_np.repeat(self._a, _arr(repeats), axis=dim))

    def flip(self, dims):
        dims = tuple(dims) if isinstance(dims, (list, tuple)) else (dims,)
        return _make(_np.flip(self._a, dims), [(self, lambda g: _np.flip(g, dims))])

    def roll(self, shifts, dims=None):
        return _make(_np.roll(self._a, shifts, dims))

    def split(self, size, dim=0):
        n = self._a.shape[dim]
        cuts = list(range(size, n, size)) if isinstance(size, _b.int) else [_b.int(c) for c in _np.cumsum(size)[:-1]]
        out, start = [], 0
        for end in cuts + [n]:
            sl = [slice(None)] * self.ndim
            sl[dim] = slice(start, end)
            out.append(self[tuple(sl)])
            start = end
        return tuple(out)

    def chunk(self, chunks, dim=0):
        return self.split(-(-self._a.shape[dim] // chunks), dim)

    def unbind(self, dim=0):
        return tuple(self.select(dim, i) for i in range(self._a.shape[dim]))

    def select(self, dim, index):
        sl = [slice(None)] * self.ndim
        sl[dim] = index
        return self[tuple(sl)]

    def narrow(self, dim, start, length):
        sl = [slice(None)] * self.ndim
        sl[dim] = slice(start, start + length)
        return self[tuple(sl)]

    def unfold(self, dimension, size, step):
        win = _np.lib.stride_tricks.sliding_window_view(self._a, size, axis=dimension)
        sl = [slice(None)] * win.ndim
        sl[dimension] = slice(None, None, step)
        return _make(win[tuple(sl)])

    def gather(self, dim, index):
        idx = _arr(index)
        d = dim % self.ndim

        def back(g):
            out = _np.zeros(self._a.shape, dtype=_np.asarray(g).dtype)
            grids = list(_np.indices(idx.shape))
            grids[d] = idx
            _np.add.at(out, tuple(grids), g)
            return out

        return _make(_np.take_along_axis(self._a, idx, d), [(self, back)], self._bf16)

    def index_select(self, dim, index):
        sl = [slice(None)] * self.ndim
        sl[dim] = _arr(index)
        return self[tuple(sl)]

    def masked_select(self, mask):
        return self[_np.broadcast_to(_arr(mask), self._a.shape)]

    def nonzero(self, as_tuple=False):
        nz = _np.nonzero(self._a)
        if as_tuple:
            return tuple(Tensor._wrap(i.astype(_np.int64)) for i in nz)
        return Tensor._wrap(_np.stack(nz, axis=1).astype(_np.int64))

    argwhere = nonzero

    def tril(self, diagonal=0):
        return _make(_np.tril(self._a, diagonal), [(self, lambda g: _np.tril(g, diagonal))])

    def triu(self, diagonal=0):
        return _make(_np.triu(self._a, diagonal), [(self, lambda g: _np.triu(g, diagonal))])

    def diagonal(self, offset=0, dim1=0, dim2=1):
        return _make(_np.diagonal(self._a, offset, dim1, dim2).copy())

    def trace(self):
        return self.diagonal().sum()

    # ---- reductions
    def sum(self, dim=None, keepdim=False, dtype=None):
        _event("aten::sum")
        ax = _ax(dim, self.ndim)
        res = self._a.sum(axis=ax, keepdims=keepdim)
        res = _np.asarray(res).astype(_np.int64) if self._a.dtype.kind in "biu" else _np.asarray(res)
        if dtype is not None:
            res = res.astype(dtype.np)
        return _make(res, [(self, lambda g: _expand_back(g, self._a.shape, ax, keepdim))], self._bf16)

    def nansum(self, dim=None, keepdim=False):
        return _make(_np.asarray(_np.nansum(self._a, axis=_ax(dim, self.ndim), keepdims=keepdim)))

    def mean(self, dim=None, keepdim=False, dtype=None):
        if not self.is_floating_point() and dtype is None:
            raise RuntimeError(f"mean(): could not infer output dtype. Input dtype must be either a floating point or "
                               f"complex dtype. Got: {self.dtype.name.capitalize()}")
        src = self if dtype is None else self.to(dtype)
        ax = _ax(dim, src.ndim)
        n = src._a.size / _b.max(_np.asarray(src._a.sum(axis=ax, keepdims=keepdim)).size, 1)
        return _make(_np.asarray(src._a.mean(axis=ax, keepdims=keepdim)),
                     [(src, lambda g: _expand_back(g, src._a.shape, ax, keepdim) / n)], src._bf16)

    def nanmean(self, dim=None, keepdim=False):
        return _make(_np.asarray(_np.nanmean(self._a, axis=_ax(dim, self.ndim), keepdims=keepdim)))

    def var(self, dim=None, unbiased=True, keepdim=False, correction=None):
        c = (1 if unbiased else 0) if correction is None else correction
        ax = _ax(dim, self.ndim)
        n = self._a.size / _b.max(_np.asarray(self._a.sum(axis=ax, keepdims=keepdim)).size, 1)
        dev = (self - self.mean(dim=dim, keepdim=True)) ** 2
        return dev.sum(dim=dim, keepdim=keepdim) / (n - c)

    def std(self, dim=None, unbiased=True, keepdim=False, correction=None):
        return self.var(dim, unbiased, keepdim, correction).sqrt()

    def _extreme(self, argfn, dim, keepdim):
        if dim is None:
            return self.reshape(-1)[_b.int(argfn(self._a.reshape(-1)))]
        d = dim % self.ndim
        idx = _np.expand_dims(argfn(self._a, axis=d), d)
        vals = self.gather(d, idx)
        if not keepdim:
            vals, idx = vals.squeeze(d), _np.squeeze(idx, d)
        return _Tup(vals, Tensor._wrap(idx.astype(_np.int64)))

    def max(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return maximum(self, dim)
        return self._extreme(_np.argmax, dim, keepdim)

    def min(self, dim=None, keepdim=False):
        if isinstance(dim, Tensor):
            return minimum(self, dim)
        return self._extreme(_np.argmin, dim, keepdim)

    def amax(self, dim=None, keepdim=False):
        ax = _ax(dim, self.ndim)
        res = _np.asarray(self._a.max(axis=ax, keepdims=keepdim))
        mask = self._a == _np.asarray(self._a.max(axis=ax, keepdims=True))

        def back(g):
            g = _expand_back(g, self._a.shape, ax, keepdim)
            return g * mask / mask.sum(axis=ax, keepdims=True)

        return _make(res, [(self, back)], self._bf16)

    def amin(self, dim=None, keepdim=False):
        return -((-self).amax(dim, keepdim))

    def argmax(self, dim=None, keepdim=False):
        r = _np.argmax(self._a, axis=dim, keepdims=keepdim) if dim is not None else _np.argmax(self._a)
        return _make(_np.asarray(r, dtype=_np.int64))

    def argmin(self, dim=None, keepdim=False):
        r = _np.argmin(self._a, axis=dim, keepdims=keepdim) if dim is not None else _np.argmin(self._a)
        return _make(_np.asarray(r, dtype=_np.int64))

    def median(self, dim=None, keepdim=False):
        if dim is None:
            s = _np.sort(self._a.reshape(-1))
            return Tensor._wrap(_np.asarray(s[(len(s) - 1) // 2]))
        idx = _np.argsort(self._a, axis=dim, kind="stable")
        mid = _np.take(idx, [(self._a.shape[dim] - 1) // 2], axis=dim)
        vals = _np.take_along_axis(self._a, mid, dim)
        if not keepdim:
            vals, mid = _np.squeeze(vals, dim), _np.squeeze(mid, dim)
        return _Tup(Tensor._wrap(vals), Tensor._wrap(mid.astype(_np.int64)))

    def quantile(self, q, dim=None, keepdim=False):
        return _make(_np.asarray(_np.quantile(self._a, _arr(q), axis=dim, keepdims=keepdim)).astype(self._a.dtype))

    def prod(self, dim=None, keepdim=False):
        return _make(_np.asarray(self._a.prod(axis=dim, keepdims=keepdim)))

    def cumsum(self, dim):
        return _make(_np.cumsum(self._a, axis=dim), [(self, lambda g: _np.flip(_np.cumsum(_np.flip(g, dim), axis=dim), dim))])

    def cumprod(self, dim):
        return _make(_np.cumprod(self._a, axis=dim))

    def cummax(self, dim):
        vals = _np.maximum.accumulate(self._a, axis=dim)
        n = self._a.shape[dim]
        ar = _np.arange(n).reshape([-1 if i == dim % self.ndim else 1 for i in range(self.ndim)])
        idx = _np.maximum.accumulate(_np.where(self._a == vals, ar, 0), axis=dim)
        return _Tup(Tensor._wrap(vals), Tensor._wrap(idx.astype(_np.int64)))

    def diff(self, n=1, dim=-1):
        return _make(_np.diff(self._a, n=n, axis=dim))

    def all(self, dim=None, keepdim=False):
        return _make(_np.asarray(self._a.all(axis=dim, keepdims=keepdim)))

    def any(self, dim=None, keepdim=False):
        return _make(_np.asarray(self._a.any(axis=dim, keepdims=keepdim)))

    def count_nonzero(self, dim=None):
        return _make(_np.asarray(_np.count_nonzero(self._a, axis=dim), dtype=_np.int64))

    def logsumexp(self, dim, keepdim=False):
        m = self.amax(dim=dim, keepdim=True).detach()
        m = m.masked_fill(~m.isfinite(), 0.0)
        out = (self - m).exp().sum(dim=dim, keepdim=True).log() + m
        return out if keepdim else out.squeeze(dim)

    def softmax(self, dim, dtype=None):
        _event("aten::softmax")
        src = self.float() if _AUTOCAST[0] is not None else self
        e = (src - src.amax(dim=dim, keepdim=True).detach()).exp()
        return e / e.sum(dim=dim, keepdim=True)

    def log_softmax(self, dim, dtype=None):
        src = self.float() if _AUTOCAST[0] is not None else self
        return src - src.logsumexp(dim, keepdim=True)

    def norm(self, p=2, dim=None, keepdim=False):
        if p == "fro":
            p = 2
        if p == _math.inf or p == "inf":
            return self.abs().amax(dim=dim, keepdim=keepdim)
        return (self.abs() ** p).sum(dim=dim, keepdim=keepdim) ** (1 / p)

    def sort(self, dim=-1, descending=False, stable=True):
        idx = _np.argsort(-self._a if descending else self._a, axis=dim, kind="stable")
        return _Tup(self.gather(dim, idx), Tensor._wrap(idx.astype(_np.int64)))

    def argsort(self, dim=-1, descending=False, stable=True):
        return self.sort(dim, descending).indices

    def topk(self, k, dim=-1, largest=True, sorted=True):
        v, i = self.sort(dim, descending=largest)
        sl = [slice(None)] * self.ndim
        sl[dim] = slice(0, k)
        return _Tup(v[tuple(sl)], i[tuple(sl)])

    def kthvalue(self, k, dim=-1):
        v, i = self.sort(dim)
        return _Tup(v.select(dim, k - 1), i.select(dim, k - 1))

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        return unique(self, sorted, return_inverse, return_counts, dim)

    def bincount(self, minlength=0):
        return bincount(self, minlength=minlength)


def _float_formatter(a):
    """PyTorch's float layout: integers as '3.', else 4 decimals, scientific for wide ranges; padded to one width."""
    p = _PRINT["precision"]
    fin = _np.abs(a[_np.isfinite(a)].astype(_np.float64))
    nz = fin[fin != 0]
    integral = fin.size == 0 or _np.all(fin == _np.round(fin))
    sci = False
    if nz.size and _PRINT["sci_mode"]:
        lo, hi = nz.min(), nz.max()
        sci = hi > 1e8 if integral else (hi / lo > 1000 or hi > 1e8 or lo < 1e-4)
    if sci:
        f = lambda v: f"{v:.{p}e}"  # noqa: E731
    elif integral:
        f = lambda v: f"{v:.0f}."  # noqa: E731
    else:
        f = lambda v: f"{v:.{p}f}"  # noqa: E731

    def g(v):
        return "nan" if v != v else ("inf" if v > 0 else "-inf") if _np.isinf(v) else f(v)

    width = _b.max(len(g(v)) for v in a.reshape(-1)[:2000])
    return lambda v: g(v).rjust(width)


def _const(o, like):
    t = Tensor._wrap(_new(_arr(o)))
    if like.is_floating_point() and t._a.dtype.kind != "f":
        t._a = t._a.astype(like._a.dtype)
    return t


def _key(key):
    if isinstance(key, tuple):
        return tuple(_arr(k) if isinstance(k, (Tensor, list)) else k for k in key)
    return _arr(key) if isinstance(key, (Tensor, list)) else key


def _from_bytes(a):
    if isinstance(a, tuple):  # (raw bytes, dtype string, shape): pickles without any NumPy globals
        raw, dt, shape = a
        return _np.frombuffer(raw, dtype=_np.dtype(dt)).reshape(shape).copy()
    return a


def _rebuild(a, requires_grad, bf16=False):
    t = Tensor._wrap(_from_bytes(a))
    t._bf16 = bf16
    t.requires_grad = requires_grad
    return t


def _erf(x):
    return _np.vectorize(_math.erf, otypes=[_np.float64])(x).astype(_np.asarray(x).dtype if _np.asarray(x).dtype.kind == "f" else _np.float32)


class _Handle:
    _ids = [0]

    def __init__(self, store, item):
        self._store, self._item = store, item
        _Handle._ids[0] += 1
        self.id = _Handle._ids[0]

    def remove(self):
        if isinstance(self._store, dict):
            self._store.pop(self._item, None)
        elif self._item in self._store:
            self._store.remove(self._item)


def _backprop(root, seed):
    with _np.errstate(all="ignore"):
        return _backprop_impl(root, seed)


def _backprop_impl(root, seed):
    order, seen = [], set()
    stack = [(root, False)]
    while stack:  # iterative topological sort: deep graphs must not hit the recursion limit
        t, done = stack.pop()
        if done:
            order.append(t)
            continue
        if id(t) in seen:
            continue
        seen.add(id(t))
        stack.append((t, True))
        stack.extend((p, False) for p, _ in t._parents)
    grads = {id(root): _np.asarray(seed)}
    tensors = {id(root): root}
    for t in reversed(order):
        g = grads.get(id(t))
        if g is None:
            continue
        for h in t._hooks:  # tensor hooks may inspect or replace the gradient
            r = h(Tensor._wrap(g))
            if r is not None:
                g = _arr(r)
        grads[id(t)] = g
        for p, fn in t._parents:
            pg = _np.asarray(fn(g))
            grads[id(p)] = pg if id(p) not in grads else grads[id(p)] + pg
            tensors[id(p)] = p
    return {tensors[i]: g for i, g in grads.items()}


# ---------------------------------------------------------------- creation

def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    _check_device(device)
    if isinstance(data, (list, tuple)):
        data = [_arr(x) if isinstance(x, Tensor) else x for x in data]
    t = Tensor._wrap(_new(_np.array(data) if not isinstance(data, Tensor) else _arr(data).copy(), dtype if dtype is not bfloat16 else float32))
    if dtype is bfloat16:
        t._a, t._bf16 = _to_bf16(t._a), True
    if requires_grad:
        t.requires_grad_(True)
    return t


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor):
        return data if dtype is None else data.to(dtype)
    if isinstance(data, _np.ndarray) and (dtype is None or _np.dtype(dtype.np) == data.dtype):
        return Tensor._wrap(data)  # shares memory with the NumPy array
    return tensor(data, dtype=dtype)


asarray = as_tensor


def from_numpy(a):
    return Tensor._wrap(a)  # shares memory and keeps the NumPy dtype (float64 stays float64)


def _filled(fill, args, dtype=None, requires_grad=False, device=None):
    _check_device(device)
    t = Tensor._wrap(_np.full(_size_args(args), fill, dtype=(dtype or float32).np))
    if dtype is bfloat16:
        t._bf16 = True
    if requires_grad:
        t.requires_grad_(True)
    return t


def zeros(*size, dtype=None, device=None, requires_grad=False):
    return _filled(0, size, dtype, requires_grad, device)


def ones(*size, dtype=None, device=None, requires_grad=False):
    return _filled(1, size, dtype, requires_grad, device)


empty = zeros


def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = (bool if isinstance(fill_value, _b.bool) else int64 if isinstance(fill_value, _b.int) else float32)
    return _filled(fill_value, (size,) if isinstance(size, _b.int) else (tuple(size),), dtype, requires_grad, device)


def zeros_like(t, dtype=None, requires_grad=False):
    return zeros(*t.shape, dtype=dtype or t.dtype, requires_grad=requires_grad)


def ones_like(t, dtype=None, requires_grad=False):
    return ones(*t.shape, dtype=dtype or t.dtype, requires_grad=requires_grad)


empty_like = zeros_like


def full_like(t, fill_value, dtype=None):
    return full(tuple(t.shape), fill_value, dtype=dtype or t.dtype)


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    floaty = _b.any(isinstance(v, _b.float) for v in (start, end, step))
    t = Tensor._wrap(_np.arange(start, end, step, dtype=(dtype or (float32 if floaty else int64)).np))
    return t.requires_grad_(True) if requires_grad else t


def linspace(start, end, steps, dtype=None, device=None):
    return Tensor._wrap(_np.linspace(start, end, steps).astype((dtype or float32).np))


def logspace(start, end, steps, base=10.0, dtype=None):
    return Tensor._wrap(_np.logspace(start, end, steps, base=base).astype((dtype or float32).np))


def eye(n, m=None, dtype=None, device=None):
    return Tensor._wrap(_np.eye(n, m, dtype=(dtype or float32).np))


def triu(t, diagonal=0):
    return t.triu(diagonal)


def tril(t, diagonal=0):
    return t.tril(diagonal)


def diag(t, diagonal=0):
    return _make(_np.diag(t._a, diagonal))


def tril_indices(row, col, offset=0):
    return Tensor._wrap(_np.stack(_np.tril_indices(row, offset, col)).astype(_np.int64))


def triu_indices(row, col, offset=0):
    return Tensor._wrap(_np.stack(_np.triu_indices(row, offset, col)).astype(_np.int64))


def meshgrid(*tensors, indexing="ij"):
    tensors = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)) else tensors
    return tuple(Tensor._wrap(g) for g in _np.meshgrid(*[_arr(t) for t in tensors], indexing=indexing))


def as_strided(t, size, stride, storage_offset=0):
    a = t._a
    return _make(_np.lib.stride_tricks.as_strided(a.reshape(-1)[storage_offset:], tuple(size),
                                                  tuple(s * a.itemsize for s in stride), writeable=False))


# ---------------------------------------------------------------- random

class Generator:
    def __init__(self, device="cpu"):
        self._g = _np.random.default_rng(0)
        self._seed = 0

    def manual_seed(self, seed):
        self._g = _np.random.default_rng(seed)
        self._seed = seed
        return self

    def initial_seed(self):
        return self._seed

    def get_state(self):
        return Tensor._wrap(_np.frombuffer(_pickle.dumps(self._g.bit_generator.state), dtype=_np.uint8).copy())

    def set_state(self, state):
        self._g.bit_generator.state = _pickle.loads(_arr(state).tobytes())


_DEFAULT_GEN = Generator()
_RNG = [_DEFAULT_GEN._g]


def _gen(generator):
    return generator._g if generator is not None else _RNG[0]


def manual_seed(seed):
    _DEFAULT_GEN.manual_seed(seed)
    _RNG[0] = _DEFAULT_GEN._g
    return _DEFAULT_GEN


def seed():
    return _DEFAULT_GEN._seed


def initial_seed():
    return _DEFAULT_GEN._seed


def get_rng_state():
    return _DEFAULT_GEN.get_state()


def set_rng_state(state):
    _DEFAULT_GEN.set_state(state)
    _RNG[0] = _DEFAULT_GEN._g


def rand(*size, generator=None, dtype=None, device=None, requires_grad=False):
    t = Tensor._wrap(_gen(generator).random(_size_args(size)).astype((dtype or float32).np))
    return t.requires_grad_(True) if requires_grad else t


def randn(*size, generator=None, dtype=None, device=None, requires_grad=False):
    t = Tensor._wrap(_gen(generator).standard_normal(_size_args(size)).astype((dtype or float32).np))
    return t.requires_grad_(True) if requires_grad else t


def randint(low, high=None, size=None, generator=None, dtype=None, device=None):
    if high is None:
        low, high = 0, low
    return Tensor._wrap(_gen(generator).integers(low, high, size=tuple(size)).astype((dtype or int64).np))


def randperm(n, generator=None, dtype=None, device=None):
    return Tensor._wrap(_gen(generator).permutation(n).astype((dtype or int64).np))


def rand_like(t, dtype=None):
    return rand(*t.shape, dtype=dtype or t.dtype)


def randn_like(t, dtype=None):
    return randn(*t.shape, dtype=dtype or t.dtype)


def randint_like(t, low, high=None):
    return randint(low, high, tuple(t.shape), dtype=t.dtype)


def normal(mean, std=1.0, size=None, generator=None):
    if size is not None:
        return Tensor._wrap(_gen(generator).normal(mean, std, size=tuple(size)).astype(_np.float32))
    m, s = _arr(mean), _arr(std)
    return Tensor._wrap(_gen(generator).normal(m, s).astype(_np.float32))


def bernoulli(p, generator=None):
    return Tensor._wrap((_gen(generator).random(_arr(p).shape) < _arr(p)).astype(_arr(p).dtype))


def multinomial(probs, num_samples, replacement=False, generator=None):
    p = _np.asarray(_arr(probs), dtype=_np.float64)
    g = _gen(generator)

    def draw(row):
        row = row / row.sum()
        return g.choice(len(row), size=num_samples, replace=replacement, p=row)

    out = draw(p) if p.ndim == 1 else _np.stack([draw(r) for r in p])
    return Tensor._wrap(out.astype(_np.int64))


def poisson(rates, generator=None):
    return Tensor._wrap(_gen(generator).poisson(_arr(rates)).astype(_arr(rates).dtype))


# ---------------------------------------------------------------- functions

def _method(name):
    def f(t, *a, **k):
        return getattr(t if isinstance(t, Tensor) else tensor(t), name)(*a, **k)
    f.__name__ = name
    return f


for _name in ("exp", "log", "log1p", "log2", "expm1", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh", "sigmoid", "relu", "erf",
              "square", "sign", "floor", "ceil", "trunc", "frac", "reciprocal", "sum", "nansum", "mean", "nanmean", "std",
              "var", "prod", "cumsum", "cumprod", "cummax", "argmax", "argmin", "amax", "amin", "median", "quantile",
              "logsumexp", "softmax", "log_softmax", "reshape", "flatten", "unsqueeze", "squeeze", "permute", "transpose",
              "swapaxes", "movedim", "masked_fill", "gather", "sort", "argsort", "topk", "kthvalue", "clamp", "clip", "norm",
              "all", "any", "round", "split", "chunk", "unbind", "narrow", "select", "index_select", "masked_select",
              "nonzero", "argwhere", "matmul", "mm", "bmm", "dot", "mv", "max", "min", "numel", "diff", "count_nonzero",
              "isnan", "isinf", "isfinite", "nan_to_num", "flip", "roll", "repeat_interleave", "t", "diagonal", "trace",
              "logical_not", "lerp", "ravel"):
    globals()[_name] = _method(_name)


def is_tensor(x):
    return isinstance(x, Tensor)


def is_floating_point(t):
    return t.is_floating_point()


def where(cond, a=None, b=None):
    c = _arr(cond).astype(_b.bool)
    if a is None:
        return tuple(Tensor._wrap(i.astype(_np.int64)) for i in _np.nonzero(c))
    av, bv = _arr(a), _arr(b)
    res = _promote(_np.where(c, av, bv), (a, b))
    if isinstance(a, Tensor) and not isinstance(b, Tensor):
        res = res.astype(av.dtype) if av.dtype.kind == "f" else res
    elif isinstance(b, Tensor) and not isinstance(a, Tensor):
        res = res.astype(bv.dtype) if bv.dtype.kind == "f" else res
    return _make(res, [(a, lambda g: _unbroadcast(_np.where(c, g, 0), _np.shape(av))),
                       (b, lambda g: _unbroadcast(_np.where(c, 0, g), _np.shape(bv)))], _bf(a, b))


def maximum(a, b):
    av, bv = _arr(a), _arr(b)
    return _make(_np.maximum(av, bv), [(a, lambda g: _unbroadcast(g * (av >= bv), av.shape)),
                                         (b, lambda g: _unbroadcast(g * (bv > av), bv.shape))])


def minimum(a, b):
    av, bv = _arr(a), _arr(b)
    return _make(_np.minimum(av, bv), [(a, lambda g: _unbroadcast(g * (av <= bv), av.shape)),
                                         (b, lambda g: _unbroadcast(g * (bv < av), bv.shape))])


def logical_and(a, b):
    return _make(_np.logical_and(_arr(a), _arr(b)))


def logical_or(a, b):
    return _make(_np.logical_or(_arr(a), _arr(b)))


def logical_xor(a, b):
    return _make(_np.logical_xor(_arr(a), _arr(b)))


def cat(tensors, dim=0):
    tensors = list(tensors)
    arrs = [_arr(t) for t in tensors]
    d = dim % arrs[0].ndim
    cuts = list(_np.cumsum([a.shape[d] for a in arrs])[:-1])
    parts = [(t, (lambda i: lambda g: _np.split(g, cuts, axis=d)[i])(i)) for i, t in enumerate(tensors)]
    return _make(_np.concatenate(arrs, axis=d), parts, _bf(*tensors))


concat = concatenate = cat


def stack(tensors, dim=0):
    tensors = list(tensors)
    if len({tuple(_np.shape(_arr(t))) for t in tensors}) > 1:
        raise RuntimeError("stack expects each tensor to be equal size")
    return cat([t.unsqueeze(dim) if isinstance(t, Tensor) else tensor(t).unsqueeze(dim) for t in tensors], dim=dim)


def hstack(tensors):
    return cat(tensors, dim=1 if _arr(tensors[0]).ndim > 1 else 0)


def vstack(tensors):
    return cat([t if t.ndim > 1 else t.unsqueeze(0) for t in tensors], dim=0)


def column_stack(tensors):
    return cat([t.unsqueeze(1) if t.ndim == 1 else t for t in tensors], dim=1)


def tile(t, reps):
    return t.repeat(*((1,) * (len(reps) - t.ndim) + tuple(reps))) if len(reps) >= t.ndim else t.repeat(*reps)


def einsum(eq, *ops):
    ops = ops[0] if len(ops) == 1 and isinstance(ops[0], (list, tuple)) else ops
    arrs = [_arr(o) for o in ops]
    lhs, rhs = eq.replace(" ", "").split("->") if "->" in eq else (eq.replace(" ", ""), None)
    terms = lhs.split(",")
    if rhs is None:
        letters = "".join(terms)
        rhs = "".join(sorted(c for c in set(letters) if letters.count(c) == 1))
    res = _np.einsum(eq, *arrs)
    parents = []
    for i, (t, term) in enumerate(zip(ops, terms)):
        others = [x for j, x in enumerate(terms) if j != i]
        sub = f"{rhs}," + ",".join(others) + f"->{term}" if others else f"{rhs}->{term}"

        def back(g, i=i, sub=sub, term=term):
            r = _np.einsum(sub, g, *[a for j, a in enumerate(arrs) if j != i])
            return _np.broadcast_to(r, arrs[i].shape) if r.shape != arrs[i].shape else r

        parents.append((t, back))
    return _make(_np.asarray(_promote(_np.asarray(res), ops)), parents)


def outer(a, b):
    return a.unsqueeze(-1) * b


def inner(a, b):
    return (a * b).sum(-1)


def cdist(x1, x2, p=2.0):
    d = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
    return (d ** p).sum(-1) ** (1 / p)


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    return _b.bool(_np.allclose(_arr(a), _arr(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    return _make(_np.isclose(_arr(a), _arr(b), rtol=rtol, atol=atol, equal_nan=equal_nan))


def equal(a, b):
    return _arr(a).shape == _arr(b).shape and _b.bool((_arr(a) == _arr(b)).all())


def broadcast_shapes(*shapes):
    try:
        return Size(_np.broadcast_shapes(*[(s,) if isinstance(s, _b.int) else tuple(s) for s in shapes]))
    except ValueError:
        raise RuntimeError(f"Shape mismatch: objects cannot be broadcast to a single shape {shapes}") from None


def broadcast_to(t, shape):
    return t.expand(*shape)


def broadcast_tensors(*tensors):
    shape = broadcast_shapes(*[t.shape for t in tensors])
    return tuple(t.expand(*shape) for t in tensors)


def bincount(t, weights=None, minlength=0):
    w = None if weights is None else _arr(weights)
    out = _np.bincount(_arr(t), weights=w, minlength=minlength)
    return Tensor._wrap(out.astype(_np.int64 if w is None else _np.float32))


def unique(t, sorted=True, return_inverse=False, return_counts=False, dim=None):
    vals, inv, cnt = _np.unique(_arr(t), return_inverse=True, return_counts=True, axis=dim)
    out = [Tensor._wrap(vals)]
    if return_inverse:
        out.append(Tensor._wrap(inv.reshape(_arr(t).shape if dim is None else -1).astype(_np.int64)))
    if return_counts:
        out.append(Tensor._wrap(cnt.astype(_np.int64)))
    return out[0] if len(out) == 1 else tuple(out)


def searchsorted(sorted_sequence, values, right=False, side=None):
    side = side or ("right" if right else "left")
    return Tensor._wrap(_np.searchsorted(_arr(sorted_sequence), _arr(values), side=side).astype(_np.int64))


def bucketize(values, boundaries, right=False):
    return Tensor._wrap(_np.searchsorted(_arr(boundaries), _arr(values), side="left" if not right else "right").astype(_np.int64))


def isin(elements, test_elements, invert=False):
    return _make(_np.isin(_arr(elements), _arr(test_elements), invert=invert))


def numel(t):
    return t.numel()


_PRINT = {"precision": 4, "sci_mode": True}


def set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None):
    if profile == "default":
        _PRINT.update(precision=4, sci_mode=True)
    if precision is not None:
        _PRINT["precision"] = precision
    if sci_mode is not None:
        _PRINT["sci_mode"] = sci_mode


def use_deterministic_algorithms(mode, warn_only=False):
    pass


def set_float32_matmul_precision(precision):
    if precision not in ("highest", "high", "medium"):
        raise ValueError("precision must be 'highest', 'high' or 'medium'")


def get_float32_matmul_precision():
    return "highest"


def save(obj, f):
    with open(f, "wb") if isinstance(f, str) else _ctx.nullcontext(f) as fh:
        _pickle.dump(obj, fh)


class _SafeUnpickler(_pickle.Unpickler):
    _OK = {("collections", "OrderedDict"), ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
           ("builtins", "set"), ("builtins", "int"), ("builtins", "float"), ("builtins", "str"), ("builtins", "bool"),
           ("torch", "_rebuild"), ("torch.nn", "_rebuild_param"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
           ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy.core.multiarray", "scalar"),
           ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self._OK or module.startswith("numpy") and name in ("dtype", "ndarray", "_reconstruct", "scalar"):
            return super().find_class(module, name)
        raise _pickle.UnpicklingError(f"Weights only load failed: Unsupported global: GLOBAL {module}.{name}. "
                                      f"Re-run with weights_only=False only if you trust the file.")


def load(f, map_location=None, weights_only=True, mmap=None):
    with open(f, "rb") if isinstance(f, str) else _ctx.nullcontext(f) as fh:
        return (_SafeUnpickler(fh) if weights_only else _pickle.Unpickler(fh)).load()


def compile(model=None, *, fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False):
    from ._compile import compile_ as _c
    return _c(model, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable)


def __getattr__(name):
    _unsupported(f"torch.{name}")


from . import (_dynamo, amp, autograd, backends, compiler, cuda, distributed, library, linalg, nn, optim,  # noqa: E402,F401
               profiler, testing, utils)
from .autograd import Function  # noqa: E402,F401
from . import _extra, func  # noqa: E402,F401
from .library import ops  # noqa: E402,F401

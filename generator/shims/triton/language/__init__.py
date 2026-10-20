"""triton-lite triton.language: blocks are NumPy arrays; pointers are (memory, offsets) pairs."""

import builtins
import inspect
import itertools

import numpy as np

_STATE = {"pid": (0, 0, 0), "grid": (1, 1, 1)}


class dtype:
    def __init__(self, name, np_dtype):
        self.name, self.np = name, np.dtype(np_dtype)

    def __repr__(self):
        return self.name

    @property
    def element_ty(self):
        return self

    def is_floating(self):
        return self.np.kind == "f"


float16, bfloat16, float32, float64 = dtype("fp16", np.float16), dtype("bf16", np.float32), dtype("fp32", np.float32), dtype("fp64", np.float64)
int1, int8, int16, int32, int64 = dtype("int1", bool), dtype("int8", np.int8), dtype("int16", np.int16), dtype("int32", np.int32), dtype("int64", np.int64)
uint8, uint32 = dtype("uint8", np.uint8), dtype("uint32", np.uint32)
float8e4nv = dtype("fp8e4nv", np.float16)
_BY_NP = {d.np: d for d in (float16, float32, float64, int1, int8, int16, int32, int64, uint8, uint32)}


class constexpr:
    """Annotation for compile-time constants (`BLOCK_SIZE: tl.constexpr`)."""

    def __init__(self, value=None):
        self.value = value


class tensor(np.ndarray):
    """A block of values. Arithmetic is NumPy's; `.to(tl.float16)` casts."""

    def to(self, dt, bitcast=False):
        return _blk(np.asarray(self).astype(dt.np))


def _blk(a):
    return np.asarray(a).view(tensor)


def _np(x):
    if isinstance(x, tensor):
        return x.view(np.ndarray)
    return np.asarray(x)


class pointer_type:
    def __init__(self, element_ty):
        self.element_ty = element_ty

    def __repr__(self):
        return f"pointer<{self.element_ty}>"


class _Pointer:
    __array_ufunc__ = None  # make `offsets + ptr` call _Pointer.__radd__ instead of broadcasting

    def __init__(self, mem, offsets, name="ptr"):
        self.mem, self.offsets, self.name = mem, offsets, name

    @property
    def dtype(self):
        return pointer_type(_BY_NP.get(self.mem.dtype, float32))

    def __add__(self, o):
        if isinstance(o, _Pointer):
            raise TypeError("cannot add two pointers")
        o = _np(o)
        if o.dtype.kind == "f":
            raise TypeError("pointer offsets must be integers")
        return _Pointer(self.mem, self.offsets + o.astype(np.int64), self.name)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-_np(o))

    def __getitem__(self, key):
        return _Pointer(self.mem, np.asarray(self.offsets)[key], self.name)

    def __repr__(self):
        return f"<pointer {self.name} + offsets of shape {np.shape(self.offsets)}>"


def _flat_memory(t):
    """A 1-D view of the tensor's memory: offset k = element k from the tensor's data pointer."""
    a = t._a if hasattr(t, "_a") else np.asarray(t)
    if a.size == 0:
        return a.reshape(-1)
    if builtins.any(s < 0 for s in a.strides):
        raise ValueError("triton-lite: tensors with negative strides can't be passed to a kernel; call .contiguous()")
    span = 1 + builtins.sum((n - 1) * (s // a.itemsize) for n, s in zip(a.shape, a.strides))
    return np.lib.stride_tricks.as_strided(a, shape=(span,), strides=(a.itemsize,))


def _launch(fn, grid, args, kwargs):
    names = list(inspect.signature(fn).parameters)
    meta = dict(kwargs)
    for k in ("num_warps", "num_stages", "num_ctas"):
        meta.pop(k, None)
    bound = {**dict(zip(names, args)), **meta}
    g = grid(bound) if callable(grid) else grid
    g = tuple(g) if isinstance(g, (tuple, list)) else (g,)
    if not g or builtins.any(int(x) < 0 for x in g):
        raise ValueError(f"invalid grid {g}")
    g = tuple(int(x) for x in g) + (1,) * (3 - len(g))
    call = []
    for i, a in enumerate(args):
        if hasattr(a, "_a") and hasattr(a, "detach"):
            call.append(_Pointer(_flat_memory(a), np.int64(0), names[i] if i < len(names) else "ptr"))
        else:
            call.append(a)
    old = dict(_STATE)
    _STATE["grid"] = g
    try:
        for z, y, x in itertools.product(range(g[2]), range(g[1]), range(g[0])):
            _STATE["pid"] = (x, y, z)
            fn(*call, **meta)
    finally:
        _STATE.update(old)


# ---------------------------------------------------------------- programs

def program_id(axis):
    if axis not in (0, 1, 2):
        raise ValueError(f"program_id axis must be 0, 1, or 2 but got {axis}")
    return _STATE["pid"][axis]


def num_programs(axis):
    return _STATE["grid"][axis]


def arange(start, end):
    n = end - start
    if n <= 0 or n & (n - 1):
        raise ValueError(f"arange's range must be a power of 2, got {n}")
    return _blk(np.arange(start, end, dtype=np.int32))


def full(shape, value, dtype):
    return _blk(np.full(tuple(shape), value, dtype=dtype.np))


def zeros(shape, dtype):
    return full(shape, 0, dtype)


def zeros_like(x):
    return _blk(np.zeros_like(_np(x)))


def full_like(x, v):
    return _blk(np.full_like(_np(x), v))


def cast(x, dt):
    return _blk(_np(x).astype(dt.np))


# ---------------------------------------------------------------- memory

def _check(ptr, mask, op):
    if not isinstance(ptr, _Pointer):
        raise TypeError(f"{op} expects a pointer (a tensor argument plus offsets), got {type(ptr).__name__}")
    offs = np.asarray(ptr.offsets, dtype=np.int64)
    m = np.ones(offs.shape, bool) if mask is None else np.broadcast_to(_np(mask).astype(bool), offs.shape)
    bad = m & ((offs < 0) | (offs >= ptr.mem.shape[0]))
    if bad.any():
        o = int(offs[bad].flat[0])
        raise IndexError(f"{op}: out-of-bounds access at offset {o} of {ptr.name} ({ptr.mem.shape[0]} elements); "
                         f"guard the ragged tail with mask=")
    return offs, m


def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False):
    offs, m = _check(pointer, mask, "tl.load")
    vals = pointer.mem[np.where(m, offs, 0)]
    if mask is not None:
        fill = np.zeros((), vals.dtype) if other is None else np.asarray(_np(other)).astype(vals.dtype)
        vals = np.where(m, vals, fill)
    return _blk(np.asarray(vals))


def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy=""):
    offs, m = _check(pointer, mask, "tl.store")
    v = np.broadcast_to(_np(value), offs.shape).astype(pointer.mem.dtype)
    pointer.mem[offs[m]] = v[m]


def atomic_add(pointer, val, mask=None, sem=None, scope=None):
    offs, m = _check(pointer, mask, "tl.atomic_add")
    old = pointer.mem[np.where(m, offs, 0)].copy()
    v = np.broadcast_to(_np(val), offs.shape).astype(pointer.mem.dtype)
    np.add.at(pointer.mem, offs[m], v[m])
    return _blk(old)


def atomic_max(pointer, val, mask=None, sem=None, scope=None):
    offs, m = _check(pointer, mask, "tl.atomic_max")
    old = pointer.mem[np.where(m, offs, 0)].copy()
    v = np.broadcast_to(_np(val), offs.shape).astype(pointer.mem.dtype)
    np.maximum.at(pointer.mem, offs[m], v[m])
    return _blk(old)


def atomic_min(pointer, val, mask=None, sem=None, scope=None):
    offs, m = _check(pointer, mask, "tl.atomic_min")
    old = pointer.mem[np.where(m, offs, 0)].copy()
    v = np.broadcast_to(_np(val), offs.shape).astype(pointer.mem.dtype)
    np.minimum.at(pointer.mem, offs[m], v[m])
    return _blk(old)


# ---------------------------------------------------------------- math

def _u(f):
    def g(x, *a, **k):
        with np.errstate(all="ignore"):  # masked-out lanes may hold garbage; GPUs don't warn either
            return _blk(f(_np(x), *a))
    return g


exp, log, sqrt, abs, sin, cos = (_u(np.exp), _u(np.log), _u(np.sqrt), _u(np.abs), _u(np.sin), _u(np.cos))
exp2, log2, floor, ceil = _u(np.exp2), _u(np.log2), _u(np.floor), _u(np.ceil)
rsqrt = _u(lambda x: 1 / np.sqrt(x))
sigmoid = _u(lambda x: 1 / (1 + np.exp(-x)))
erf = _u(lambda x: np.vectorize(__import__("math").erf, otypes=[x.dtype if x.dtype.kind == "f" else np.float32])(x))


def fma(a, b, c):
    return _blk(_np(a) * _np(b) + _np(c))


def fdiv(a, b, ieee_rounding=False):
    return _blk(_np(a) / _np(b))


def div_rn(a, b):
    return fdiv(a, b)


def where(cond, a, b):
    return _blk(np.where(_np(cond), _np(a), _np(b)))


def maximum(a, b):
    return _blk(np.maximum(_np(a), _np(b)))


def minimum(a, b):
    return _blk(np.minimum(_np(a), _np(b)))


def clamp(x, lo, hi):
    return _blk(np.clip(_np(x), _np(lo), _np(hi)))


def _red(f):
    def r(x, axis=None, keep_dims=False, **k):
        return _blk(np.asarray(f(_np(x), axis=axis, keepdims=keep_dims)))
    return r


max, min, sum, argmax, argmin = _red(np.max), _red(np.min), _red(np.sum), None, None
xor_sum = None


def argmax(x, axis, tie_break_left=True, keep_dims=False):  # noqa: F811
    return _blk(np.asarray(np.argmax(_np(x), axis=axis)).astype(np.int32))


def argmin(x, axis, tie_break_left=True, keep_dims=False):  # noqa: F811
    return _blk(np.asarray(np.argmin(_np(x), axis=axis)).astype(np.int32))


def cumsum(x, axis=0, reverse=False):
    a = _np(x)
    if reverse:
        return _blk(np.flip(np.cumsum(np.flip(a, axis), axis=axis), axis))
    return _blk(np.cumsum(a, axis=axis))


def cumprod(x, axis=0, reverse=False):
    return _blk(np.cumprod(_np(x), axis=axis))


def softmax(x, ieee_rounding=False):
    a = _np(x)
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return _blk(e / e.sum(axis=-1, keepdims=True))


def dot(a, b, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=float32):
    A, B = _np(a), _np(b)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"tl.dot expects 2-D blocks, got shapes {A.shape} and {B.shape}")
    out = (A.astype(np.float32) @ B.astype(np.float32)) if A.dtype.kind == "f" else A.astype(np.int32) @ B.astype(np.int32)
    if acc is not None:
        out = out + _np(acc)
    return _blk(out.astype(out_dtype.np if A.dtype.kind == "f" else np.int32))


def trans(x, *dims):
    return _blk(np.transpose(_np(x), dims or None))


permute = trans


def reshape(x, *shape, can_reorder=False):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return _blk(_np(x).reshape(shape))


view = reshape


def broadcast_to(x, *shape):
    shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
    return _blk(np.broadcast_to(_np(x), shape))


def expand_dims(x, axis):
    return _blk(np.expand_dims(_np(x), axis))


def join(a, b):
    return _blk(np.stack([_np(a), _np(b)], axis=-1))


def ravel(x):
    return _blk(_np(x).reshape(-1))


def multiple_of(x, values):
    return x


def max_contiguous(x, values):
    return x


def static_assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "static_assert failed")


def static_print(*args):
    print(*args)


def device_print(prefix, *args, hex=False):
    print(prefix, *[_np(a) for a in args])


def device_assert(cond, msg=""):
    if not np.all(_np(cond)):
        raise AssertionError(msg or "device_assert failed")


def range(*args, num_stages=None, loop_unroll_factor=None, **kw):  # noqa: A001 - triton's name
    return builtins.range(*args)


static_range = range


class math:  # noqa: N801 - mirrors triton.language.math
    exp, log, sqrt, rsqrt, erf, abs, sigmoid = exp, log, sqrt, rsqrt, erf, abs, sigmoid
    fma, div_rn = fma, div_rn
    tanh = _u(np.tanh)
    pow = staticmethod(lambda a, b: _blk(np.power(_np(a), _np(b))))


extra = math

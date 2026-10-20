"""torch-lite nn: Module, Parameter, the common layers and losses (built from autograd-aware tensor ops)."""

from __future__ import annotations

import math
from collections import OrderedDict, namedtuple

import numpy as np

import torch
from torch import Tensor, _Handle

from . import functional as F
from . import init

_IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Parameter(Tensor):
    """A Tensor that a Module registers as trainable (requires_grad=True by default)."""

    def __new__(cls, data=None, requires_grad=True):
        p = Tensor.__new__(cls)
        src = data if data is not None else torch.zeros(0)
        p._init(src._a)
        p._bf16 = src._bf16
        p.requires_grad = requires_grad
        return p

    def __init__(self, data=None, requires_grad=True):
        pass

    def __repr__(self):
        return "Parameter containing:\n" + super().__repr__()

    def __reduce__(self):
        a = np.ascontiguousarray(self._a)
        return _rebuild_param, ((a.tobytes(), a.dtype.str, a.shape), self.requires_grad)


def _rebuild_param(a, requires_grad):
    return Parameter(Tensor._wrap(torch._from_bytes(a)), requires_grad)


class Buffer(Tensor):
    def __new__(cls, data, persistent=True):
        b = Tensor.__new__(cls)
        b._init(data._a)
        b.persistent = persistent
        return b

    def __init__(self, data, persistent=True):
        pass


class Module:
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent", set())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_forward_hooks", OrderedDict())
        object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_backward_hooks", OrderedDict())
        object.__setattr__(self, "training", True)

    # ---- registration
    def __setattr__(self, name, value):
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign module attributes before Module.__init__() call (call super().__init__() first)")
        if isinstance(value, Parameter):
            store = self._parameters
        elif isinstance(value, Module):
            store = self._modules
        elif isinstance(value, Buffer) or (name in self._buffers and isinstance(value, Tensor)):
            store = self._buffers
            if isinstance(value, Buffer) and not value.persistent:
                self._non_persistent.add(name)
        elif value is None and name in self._parameters:  # e.g. self.bias = None keeps the slot
            store = self._parameters
        else:
            store = None
        for d in (self._parameters, self._buffers, self._modules):
            if d is not store:
                d.pop(name, None)
        if store is None:
            object.__setattr__(self, name, value)
            return
        store[name] = value  # replacing keeps the original position, like PyTorch
        self.__dict__.pop(name, None)

    def __getattr__(self, name):
        d = self.__dict__
        for store in ("_parameters", "_buffers", "_modules"):
            if store in d and name in d[store]:
                return d[store][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name):
        for d in (self._parameters, self._buffers, self._modules):
            if name in d:
                del d[name]
                return
        object.__delattr__(self, name)

    def register_buffer(self, name, tensor, persistent=True):
        self._buffers[name] = tensor
        if not persistent:
            self._non_persistent.add(name)
        self.__dict__.setdefault("_buffer_names", set()).add(name)

    def register_parameter(self, name, param):
        if param is None:
            self._parameters[name] = None
        else:
            self._parameters[name] = param

    def add_module(self, name, module):
        self._modules[name] = module

    register_module = add_module

    # ---- calling
    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        for hook in list(self._forward_pre_hooks.values()):
            r = hook(self, args)
            if r is not None:
                args = r if isinstance(r, tuple) else (r,)
        if self._backward_hooks:
            args = tuple(a if not isinstance(a, Tensor) or not a.requires_grad else a.view_as(a) for a in args)
        out = self.forward(*args, **kwargs)
        for hook in list(self._forward_hooks.values()):
            r = hook(self, args, out)
            if r is not None:
                out = r
        if self._backward_hooks and isinstance(out, Tensor) and out.requires_grad:
            saved_in = [a for a in args if isinstance(a, Tensor) and a.requires_grad]

            def on_grad(g, mod=self):
                gin = (None,) * max(len(saved_in), 1)
                for hook in list(mod._backward_hooks.values()):
                    r = hook(mod, gin, (g,))
                    if r is not None:
                        pass
                return None

            out = out.view_as(out)
            out.register_hook(on_grad)
        return out

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        key = object()
        self._forward_hooks[key] = hook
        return _Handle(self._forward_hooks, key)

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        key = object()
        self._forward_pre_hooks[key] = hook
        return _Handle(self._forward_pre_hooks, key)

    def register_full_backward_hook(self, hook, prepend=False):
        key = object()
        self._backward_hooks[key] = hook
        return _Handle(self._backward_hooks, key)

    # ---- traversal
    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        memo = set() if memo is None else memo
        if id(self) in memo and remove_duplicate:
            return
        memo.add(id(self))
        yield prefix, self
        for name, m in self._modules.items():
            if m is not None:
                yield from m.named_modules(memo, f"{prefix}.{name}" if prefix else name, remove_duplicate)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def named_children(self):
        seen = set()
        for name, m in self._modules.items():
            if m is not None and id(m) not in seen:
                seen.add(id(m))
                yield name, m

    def children(self):
        for _, m in self.named_children():
            yield m

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        mods = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mp, m in mods:
            for name, p in m._parameters.items():
                if p is None or (remove_duplicate and id(p) in seen):
                    continue
                seen.add(id(p))
                yield (f"{mp}.{name}" if mp else name), p

    def parameters(self, recurse=True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_buffers(self, prefix="", recurse=True):
        mods = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mp, m in mods:
            for name, b in m._buffers.items():
                if b is not None:
                    yield (f"{mp}.{name}" if mp else name), b

    def buffers(self, recurse=True):
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def get_submodule(self, target):
        m = self
        for part in target.split(".") if target else []:
            m = m._modules[part] if part in m._modules else getattr(m, part)
        return m

    def get_parameter(self, target):
        mod, _, name = target.rpartition(".")
        return self.get_submodule(mod)._parameters[name]

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    # ---- state
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        out = OrderedDict() if destination is None else destination
        for name, p in self._parameters.items():
            if p is not None:
                out[prefix + name] = p if keep_vars else p.detach()
        for name, b in self._buffers.items():
            if b is not None and name not in self._non_persistent:
                out[prefix + name] = b if keep_vars else b.detach()
        for name, m in self._modules.items():
            if m is not None:
                m.state_dict(out, prefix + name + ".", keep_vars)
        return out

    def load_state_dict(self, state_dict, strict=True, assign=False):
        own = self.state_dict(keep_vars=True)
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        if strict and (missing or unexpected):
            msg = [f"Error(s) in loading state_dict for {type(self).__name__}:"]
            if missing:
                msg.append("Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing) + ".")
            if unexpected:
                msg.append("Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected) + ".")
            raise RuntimeError("\n\t".join(msg))
        with torch.no_grad():
            for k, t in own.items():
                if k in state_dict:
                    src = torch._arr(state_dict[k])
                    if src.shape != t._a.shape:
                        raise RuntimeError(f"size mismatch for {k}: copying a param with shape {tuple(src.shape)} from "
                                           f"checkpoint, the shape in current model is {tuple(t._a.shape)}.")
                    t._a[...] = src
        return _IncompatibleKeys(missing, unexpected)

    def train(self, mode=True):
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def zero_grad(self, set_to_none=True):
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype") or next((a for a in args if isinstance(a, torch.dtype)), None)
        dev = kwargs.get("device") or next((a for a in args if isinstance(a, (str, torch.device))), None)
        torch._check_device(dev)
        if dtype is not None:
            for p in self.parameters():
                if p.is_floating_point():
                    p._a = p._a.astype(np.float32 if dtype is torch.bfloat16 else dtype.np)
                    p._bf16 = dtype is torch.bfloat16
                    if p._bf16:
                        p._a = torch._to_bf16(p._a)
        return self

    def float(self):
        return self.to(torch.float32)

    def double(self):
        return self.to(torch.float64)

    def half(self):
        return self.to(torch.float16)

    def bfloat16(self):
        return self.to(torch.bfloat16)

    def cpu(self):
        return self

    def cuda(self, device=None):
        torch._check_device("cuda")

    def compile(self, *args, **kwargs):
        from torch._compile import _check
        _check(kwargs.get("mode"), kwargs.get("backend", "inductor"))
        object.__setattr__(self, "_lite_compile_options", dict(kwargs))

    def extra_repr(self):
        return ""

    def __repr__(self):
        lines = []
        for name, m in self._modules.items():
            sub = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {sub}")
        extra = self.extra_repr()
        if not lines:
            return f"{type(self).__name__}({extra})"
        return f"{type(self).__name__}(" + (extra if extra else "") + "\n" + "\n".join(lines) + "\n)"

    def __dir__(self):
        return list(super().__dir__()) + list(self._parameters) + list(self._modules) + list(self._buffers)


# ---------------------------------------------------------------- containers

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for k, m in args[0].items():
                self.add_module(k, m)
        else:
            for i, m in enumerate(args):
                self.add_module(str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx):
        mods = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(*mods[idx])
        return mods[idx]

    def __setitem__(self, idx, module):
        key = list(self._modules)[idx]
        self._modules[key] = module

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        for m in modules or []:
            self.append(m)

    def append(self, module):
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules):
        for m in modules:
            self.append(m)
        return self

    def __getitem__(self, idx):
        mods = list(self._modules.values())
        return ModuleList(mods[idx]) if isinstance(idx, slice) else mods[idx]

    def __setitem__(self, idx, module):
        self._modules[str(idx % len(self))] = module

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        for k, m in (modules or {}).items():
            self.add_module(k, m)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()


class ParameterList(Module):
    def __init__(self, values=None):
        super().__init__()
        for v in values or []:
            self.append(v)

    def append(self, p):
        self.register_parameter(str(len(self._parameters)), p if isinstance(p, Parameter) else Parameter(p))
        return self

    def __getitem__(self, i):
        return list(self._parameters.values())[i]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())


# ---------------------------------------------------------------- layers

class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        if not bias:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Bilinear(Module):
    def __init__(self, in1, in2, out, bias=True):
        super().__init__()
        bound = 1 / math.sqrt(in1)
        self.weight = Parameter(torch.empty(out, in1, in2).uniform_(-bound, bound))
        self.bias = Parameter(torch.empty(out).uniform_(-bound, bound)) if bias else None

    def forward(self, x1, x2):
        y = torch.einsum("bi,oij,bj->bo", x1, self.weight, x2)
        return y + self.bias if self.bias is not None else y


def _act(name, fn, extra=""):
    def forward(self, x):
        return fn(self, x)
    cls = type(name, (Module,), {"forward": forward, "extra_repr": lambda self: extra.format(self=self) if extra else ""})
    return cls


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)

    def extra_repr(self):
        return f"negative_slope={self.negative_slope}"


class GELU(Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(x, approximate=self.approximate)

    def extra_repr(self):
        return f"approximate='{self.approximate}'"


class SiLU(Module):
    def forward(self, x):
        return F.silu(x)


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Softplus(Module):
    def forward(self, x):
        return F.softplus(x)


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, self.alpha)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.softmax(self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class LogSoftmax(Softmax):
    def forward(self, x):
        return x.log_softmax(self.dim)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p}, inplace=False"


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

    def extra_repr(self):
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim, self.unflattened_size = dim, unflattened_size

    def forward(self, x):
        return x.unflatten(self.dim, self.unflattened_size)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = Parameter(torch.ones(*self.normalized_shape))
            self.bias = Parameter(torch.zeros(*self.normalized_shape)) if bias else None
        else:
            self.weight = self.bias = None

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.weight is not None}"


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(*self.normalized_shape)) if elementwise_affine else None

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(torch.ones(num_features)) if affine else None
        self.bias = Parameter(torch.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x):
        dims = (0,) + tuple(range(2, x.ndim))
        shape = (1, -1) + (1,) * (x.ndim - 2)
        if self.training:
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            n = x.numel() / x.shape[1]
            with torch.no_grad():
                self.running_mean._a[...] = (1 - self.momentum) * self.running_mean._a + self.momentum * mean._a
                self.running_var._a[...] = (1 - self.momentum) * self.running_var._a + self.momentum * var._a * n / max(n - 1, 1)
                self.num_batches_tracked._a[...] = self.num_batches_tracked._a + 1
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean.reshape(shape)) / (var.reshape(shape) + self.eps).sqrt()
        if self.weight is not None:
            y = y * self.weight.reshape(shape) + self.bias.reshape(shape)
        return y

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.weight is not None}, track_running_stats=True"


class BatchNorm1d(_BatchNorm):
    pass


class BatchNorm2d(_BatchNorm):
    pass


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, _weight=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.padding_idx = num_embeddings, embedding_dim, padding_idx
        self.weight = Parameter(torch.randn(num_embeddings, embedding_dim) if _weight is None else _weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight._a[padding_idx] = 0

    def forward(self, idx):
        return F.embedding(idx, self.weight, self.padding_idx)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}" + (f", padding_idx={self.padding_idx}" if self.padding_idx is not None else "")


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode="zeros"):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("torch-lite Conv2d supports groups=1 only")
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = padding if isinstance(padding, str) else ((padding, padding) if isinstance(padding, int) else tuple(padding))
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.weight = Parameter(torch.empty(out_channels, in_channels, *k))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_channels * k[0] * k[1])
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)

    def extra_repr(self):
        s = f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}"
        if self.padding not in ((0, 0), 0):
            s += f", padding={self.padding}"
        return s + ("" if self.bias is not None else ", bias=False")


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride or kernel_size, padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(MaxPool2d):
    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        if self.output_size not in (1, (1, 1)):
            raise NotImplementedError("torch-lite AdaptiveAvgPool2d supports output_size=1 only")
        return x.mean(dim=(-2, -1), keepdim=True)


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()
        self.embed_dim, self.num_heads, self.batch_first = embed_dim, num_heads, batch_first
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        init.xavier_uniform_(self.in_proj_weight)
        self.in_proj_bias = Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = Linear(embed_dim, embed_dim)
        init.zeros_(self.out_proj.bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        E, H, D = self.embed_dim, self.num_heads, self.head_dim
        Wq, Wk, Wv = self.in_proj_weight.split(E)
        bq, bk, bv = self.in_proj_bias.split(E)
        B, L, S = query.shape[0], query.shape[1], key.shape[1]
        q = F.linear(query, Wq, bq).view(B, L, H, D).transpose(1, 2)
        k = F.linear(key, Wk, bk).view(B, S, H, D).transpose(1, 2)
        v = F.linear(value, Wv, bv).view(B, S, H, D).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype is torch.bool else scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        w = scores.softmax(-1)
        out = self.out_proj((w @ v).transpose(1, 2).reshape(B, L, E))
        if not self.batch_first:
            out = out.transpose(0, 1)
        weights = (w.mean(1) if average_attn_weights else w) if need_weights else None
        return out, weights


class Transformer:
    @staticmethod
    def generate_square_subsequent_mask(sz, device=None, dtype=None):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


# ---------------------------------------------------------------- losses

class _Loss(Module):
    def __init__(self, reduction="mean", **kw):
        super().__init__()
        self.reduction = reduction
        for k, v in kw.items():
            object.__setattr__(self, k, v)


class MSELoss(_Loss):
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    def forward(self, input, target):
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    def __init__(self, reduction="mean", beta=1.0):
        super().__init__(reduction, beta=beta)

    def forward(self, input, target):
        return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)


class HuberLoss(_Loss):
    def __init__(self, reduction="mean", delta=1.0):
        super().__init__(reduction, delta=delta)

    def forward(self, input, target):
        return F.huber_loss(input, target, reduction=self.reduction, delta=self.delta)


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        super().__init__(reduction, weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction, label_smoothing=self.label_smoothing)


class NLLLoss(_Loss):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean"):
        super().__init__(reduction, weight=weight, ignore_index=ignore_index)

    def forward(self, input, target):
        return F.nll_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight=None, reduction="mean", pos_weight=None):
        super().__init__(reduction, weight=weight, pos_weight=pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, weight=self.weight, reduction=self.reduction,
                                                  pos_weight=self.pos_weight)


class BCELoss(_Loss):
    def __init__(self, weight=None, reduction="mean"):
        super().__init__(reduction, weight=weight)

    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class KLDivLoss(_Loss):
    def __init__(self, reduction="mean", log_target=False):
        super().__init__(reduction, log_target=log_target)

    def forward(self, input, target):
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class SyncBatchNorm:
    @staticmethod
    def convert_sync_batchnorm(module, process_group=None):
        return module


class Mish(Module):
    def forward(self, x):
        return F.mish(x)


class ReLU6(Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return x.clamp(0, 6)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val

    def forward(self, x):
        return x.clamp(self.min_val, self.max_val)


class Hardswish(Module):
    def forward(self, x):
        return x * (x + 3).clamp(0, 6) / 6


class SELU(Module):
    def forward(self, x):
        return 1.0507009873554805 * F.elu(x, 1.6732632423543772)


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        w = self.weight if self.weight.numel() == 1 or x.ndim < 2 else self.weight.reshape((-1,) + (1,) * (x.ndim - 2))
        return torch.where(x > 0, x, x * w)


class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, self.dim)


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        if num_channels % num_groups:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = Parameter(torch.ones(num_channels)) if affine else None
        self.bias = Parameter(torch.zeros(num_channels)) if affine else None

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if groups != 1 or dilation != 1:
            raise NotImplementedError("torch-lite Conv1d supports groups=1, dilation=1 only")
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, (kernel_size,)
        self.stride, self.padding = (stride,), padding
        self.weight = Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_channels * kernel_size)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride[0], self.padding)


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_norm_eps=1e-5,
                 batch_first=False, norm_first=False, bias=True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1, self.linear2 = Linear(d_model, dim_feedforward), Linear(dim_feedforward, d_model)
        self.norm1, self.norm2 = LayerNorm(d_model, eps=layer_norm_eps), LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout, self.dropout1, self.dropout2 = Dropout(dropout), Dropout(dropout), Dropout(dropout)
        object.__setattr__(self, "norm_first", norm_first)
        object.__setattr__(self, "_act", F.gelu if activation == "gelu" else F.relu if activation == "relu" else activation)

    def _sa(self, x, mask, pad, causal):
        return self.dropout1(self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=pad, need_weights=False)[0])

    def _ff(self, x):
        return self.dropout2(self.linear2(self.dropout(self._act(self.linear1(x)))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            return x + self._ff(self.norm2(x))
        x = self.norm1(x + self._sa(x, src_mask, src_key_padding_mask, is_causal))
        return self.norm2(x + self._ff(x))


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super().__init__()
        import copy
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x) if self.norm is not None else x


def __getattr__(name):
    raise AttributeError(f"'torch.nn.{name}' is not available in the browser's torch-lite.")


from . import attention, parallel, utils  # noqa: E402,F401

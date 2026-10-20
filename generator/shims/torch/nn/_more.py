"""More torch.nn for torch-lite: recurrent layers and packed sequences, 3-D / transposed convolutions, upsampling,
pixel shuffle, embedding bags, channel dropout, instance norm, adaptive / 1-D pooling, LazyLinear, and the
remaining losses (CTC, triplet, margin ranking, cosine embedding, Gaussian / Poisson NLL). All built from
autograd-aware tensor ops, so gradients work."""

import math
from collections import namedtuple

import numpy as np

import torch
from torch import Tensor

from . import Linear, Module, Parameter, _Loss, functional as F, init


# ---------------------------------------------------------------- functional additions

def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _triple(v):
    return (v, v, v) if isinstance(v, int) else tuple(v)


def interpolate(x, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None):
    spatial = x.shape[2:]
    nd = len(spatial)
    if size is None:
        sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor,) * nd
        size = tuple(int(math.floor(n * s)) for n, s in zip(spatial, sf))
        scales = tuple(1.0 / s for s in sf)
    else:
        size = (size,) * nd if isinstance(size, int) else tuple(size)
        scales = tuple(n / o for n, o in zip(spatial, size))
    if mode == "nearest":
        out = x
        for d, (n, o, sc) in enumerate(zip(spatial, size, scales)):
            idx = np.minimum(np.floor(np.arange(o) * sc).astype(np.int64), n - 1)
            out = out.index_select(2 + d, torch.tensor(idx))
        return out
    if mode not in ("linear", "bilinear", "trilinear"):
        raise NotImplementedError(f"torch-lite interpolate supports nearest, linear, bilinear (got {mode!r})")
    out = x
    for d, (n, o, sc) in enumerate(zip(spatial, size, scales)):
        if align_corners:
            src = np.arange(o) * ((n - 1) / (o - 1) if o > 1 else 0.0)
        else:
            src = np.maximum((np.arange(o) + 0.5) * sc - 0.5, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), n - 1)
        i1 = np.minimum(i0 + 1, n - 1)
        w = torch.tensor((src - i0).astype(np.float32))
        shape = [1] * out.ndim
        shape[2 + d] = o
        w = w.reshape(shape)
        out = out.index_select(2 + d, torch.tensor(i0)) * (1 - w) + out.index_select(2 + d, torch.tensor(i1)) * w
    return out


def upsample(x, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return interpolate(x, size, scale_factor, mode, align_corners)


def pixel_shuffle(x, r):
    N, C, H, W = x.shape
    c = C // (r * r)
    return x.reshape(N, c, r, r, H, W).permute(0, 1, 4, 2, 5, 3).reshape(N, c, H * r, W * r)


def pixel_unshuffle(x, r):
    N, C, H, W = x.shape
    return x.reshape(N, C, H // r, r, W // r, r).permute(0, 1, 3, 5, 2, 4).reshape(N, C * r * r, H // r, W // r)


def dropout2d(x, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return x
    shape = tuple(x.shape[:2]) + (1,) * (x.ndim - 2)
    keep = Tensor._wrap((torch._RNG[0].random(shape) >= p).astype(x._a.dtype))
    return x * keep / (1 - p)


dropout1d = dropout3d = dropout2d


def instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1,
                  eps=1e-5):
    dims = tuple(range(2, x.ndim))
    y = (x - x.mean(dim=dims, keepdim=True)) / (x.var(dim=dims, unbiased=False, keepdim=True) + eps).sqrt()
    shape = (1, -1) + (1,) * (x.ndim - 2)
    if weight is not None:
        y = y * weight.reshape(shape)
    if bias is not None:
        y = y + bias.reshape(shape)
    return y


def _bins(n, out):
    return [(int(math.floor(i * n / out)), int(math.ceil((i + 1) * n / out))) for i in range(out)]


def _adaptive(x, output_size, nd, reduce):
    out_sz = (output_size,) * nd if isinstance(output_size, int) or output_size is None else tuple(output_size)
    out_sz = tuple(x.shape[-nd + i] if o is None else o for i, o in enumerate(out_sz))
    res = x
    for d in range(nd):
        axis = x.ndim - nd + d
        pieces = [reduce(res.narrow(axis, s, e - s), axis) for s, e in _bins(res.shape[axis], out_sz[d])]
        res = torch.stack(pieces, dim=axis)
    return res


def adaptive_avg_pool1d(x, output_size):
    return _adaptive(x, output_size, 1, lambda t, a: t.mean(dim=a))


def adaptive_avg_pool2d(x, output_size):
    return _adaptive(x, output_size, 2, lambda t, a: t.mean(dim=a))


def adaptive_max_pool1d(x, output_size, return_indices=False):
    return _adaptive(x, output_size, 1, lambda t, a: t.amax(dim=a))


def adaptive_max_pool2d(x, output_size, return_indices=False):
    return _adaptive(x, output_size, 2, lambda t, a: t.amax(dim=a))


def max_pool1d(x, kernel_size, stride=None, padding=0):
    s = stride or kernel_size
    if padding:
        x = F.pad(x, (padding, padding), value=float("-inf"))
    return x.unfold(-1, kernel_size, s).amax(-1)


def avg_pool1d(x, kernel_size, stride=None, padding=0):
    s = stride or kernel_size
    if padding:
        x = F.pad(x, (padding, padding))
    return x.unfold(-1, kernel_size, s).mean(-1)


def conv3d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if groups != 1 or dilation not in (1, (1, 1, 1)):
        raise NotImplementedError("torch-lite conv3d supports groups=1, dilation=1 only")
    sd, sh, sw = _triple(stride)
    pd, ph, pw = _triple(padding)
    if pd:
        x = F.pad(x, (0, 0, 0, 0, pd, pd))
    kd = weight.shape[2]
    D = x.shape[2]
    od = (D - kd) // sd + 1
    outs = []
    for o in range(od):
        acc = None
        for k in range(kd):
            y = F.conv2d(x[:, :, o * sd + k], weight[:, :, k], None, (sh, sw), (ph, pw))
            acc = y if acc is None else acc + y
        outs.append(acc)
    out = torch.stack(outs, dim=2)
    return out + bias.reshape(1, -1, 1, 1, 1) if bias is not None else out


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if groups != 1 or dilation not in (1, (1, 1)):
        raise NotImplementedError("torch-lite conv_transpose2d supports groups=1, dilation=1 only")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    oph, opw = _pair(output_padding)
    N, C, H, W = x.shape
    kh, kw = weight.shape[2], weight.shape[3]
    if sh > 1 or sw > 1:  # insert (stride - 1) zeros between input pixels
        z = F.pad(x[:, :, :, None, :, None], (0, sw - 1, 0, 0, 0, sh - 1))
        x = z.reshape(N, C, H * sh, W * sw)[:, :, :(H - 1) * sh + 1, :(W - 1) * sw + 1]
    x = F.pad(x, (kw - 1 - pw, kw - 1 - pw + opw, kh - 1 - ph, kh - 1 - ph + oph))
    w = weight.flip([2, 3]).transpose(0, 1)  # (in, out, kh, kw) -> (out, in, kh, kw)
    return F.conv2d(x, w, bias)


def embedding_bag(input, weight, offsets=None, mode="mean", per_sample_weights=None, padding_idx=None):
    idx = input._a
    if idx.ndim == 2:
        bags = [idx[i] for i in range(idx.shape[0])]
        starts = None
    else:
        starts = list(offsets.tolist()) + [len(idx)]
        bags = [idx[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    rows, pos = [], 0
    for b_i, bag in enumerate(bags):
        e = weight[torch.tensor(bag.astype(np.int64))]
        if per_sample_weights is not None:
            if starts is None:
                w = per_sample_weights[b_i]
            else:
                w = per_sample_weights[starts[b_i]:starts[b_i + 1]]
            e = e * w[:, None]
        if len(bag) == 0:
            rows.append(torch.zeros(weight.shape[1]))
        elif mode == "sum":
            rows.append(e.sum(0))
        elif mode == "max":
            rows.append(e.amax(0))
        else:
            rows.append(e.mean(0))
        pos += len(bag)
    return torch.stack(rows)


def _reduce(v, reduction):
    return v.mean() if reduction == "mean" else v.sum() if reduction == "sum" else v


def margin_ranking_loss(x1, x2, target, margin=0.0, reduction="mean"):
    return _reduce((-target * (x1 - x2) + margin).clamp(min=0), reduction)


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction="mean"):
    d_ap = F.pairwise_distance(anchor, positive, p, eps)
    d_an = F.pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_an = torch.minimum(d_an, F.pairwise_distance(positive, negative, p, eps))
    return _reduce((d_ap - d_an + margin).clamp(min=0), reduction)


def cosine_embedding_loss(x1, x2, target, margin=0.0, reduction="mean"):
    eps = 1e-12
    cos = (x1 * x2).sum(-1) / ((x1 * x1).sum(-1) + eps).sqrt() / ((x2 * x2).sum(-1) + eps).sqrt()
    loss = torch.where(target == 1, 1 - cos, (cos - margin).clamp(min=0))
    return _reduce(loss, reduction)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction="mean"):
    var = var.clamp(min=eps)
    loss = 0.5 * (var.log() + (input - target) ** 2 / var)
    if full:
        loss = loss + 0.5 * math.log(2 * math.pi)
    return _reduce(loss, reduction)


def poisson_nll_loss(input, target, log_input=True, full=False, eps=1e-8, reduction="mean"):
    loss = input.exp() - target * input if log_input else input - target * (input + eps).log()
    if full:
        t = target
        stirling = torch.where(t > 1, t * t.log() - t + 0.5 * (2 * math.pi * t).log(), torch.zeros_like(t))
        loss = loss + stirling
    return _reduce(loss, reduction)


def _lse(*xs):
    s = torch.stack(xs)
    m = s.amax(0).detach()
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    return (s - m).exp().sum(0).log() + m


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False):
    """Connectionist temporal classification via the forward algorithm in log space (log_probs: (T, N, C))."""
    T, N, C = log_probs.shape
    tl = torch._arr(target_lengths).tolist()
    il = torch._arr(input_lengths).tolist()
    tg = torch._arr(targets)
    flat = tg.ndim == 1
    off = 0
    losses = []
    for n in range(N):
        lab = (tg[off:off + tl[n]] if flat else tg[n, :tl[n]]).tolist()
        off += tl[n] if flat else 0
        ext = [blank]
        for c in lab:
            ext += [c, blank]
        L = len(ext)
        skip = torch.tensor([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(L)])
        ninf = torch.full((L,), float("-inf"))
        lp = log_probs[:, n, :]
        idx = torch.tensor(ext)
        alpha = torch.where(torch.arange(L) < 2, lp[0][idx], ninf)
        for t in range(1, il[n]):
            a1 = torch.cat([ninf[:1], alpha[:-1]])
            a2 = torch.where(skip, torch.cat([ninf[:2], alpha[:-2]]), ninf)
            alpha = _lse(alpha, a1, a2) + lp[t][idx]
        ll = _lse(alpha[L - 1], alpha[L - 2]) if L > 1 else alpha[L - 1]
        loss = -ll
        if zero_infinity and torch.isinf(loss).item():
            loss = loss * 0
        losses.append(loss)
    losses = torch.stack(losses)
    if reduction == "mean":
        return (losses / torch.tensor([max(t, 1) for t in tl], dtype=losses.dtype)).mean()
    return _reduce(losses, reduction)


for _n in ("interpolate", "upsample", "pixel_shuffle", "pixel_unshuffle", "dropout2d", "dropout1d", "dropout3d",
           "instance_norm", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool1d", "adaptive_max_pool2d",
           "max_pool1d", "avg_pool1d", "conv3d", "conv_transpose2d", "embedding_bag", "margin_ranking_loss",
           "triplet_margin_loss", "cosine_embedding_loss", "gaussian_nll_loss", "poisson_nll_loss", "ctc_loss"):
    setattr(F, _n, globals()[_n])


# ---------------------------------------------------------------- modules

class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.size, self.scale_factor, self.mode, self.align_corners = size, scale_factor, mode, align_corners

    def forward(self, x):
        return interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        return f"scale_factor={self.scale_factor}, mode='{self.mode}'"


class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.upscale_factor)


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        return pixel_unshuffle(x, self.downscale_factor)


class Dropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p

    def forward(self, x):
        return dropout2d(x, self.p, self.training)


Dropout1d = Dropout3d = Dropout2d


class _InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features, self.eps = num_features, eps
        self.weight = Parameter(torch.ones(num_features)) if affine else None
        self.bias = Parameter(torch.zeros(num_features)) if affine else None

    def forward(self, x):
        return instance_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class InstanceNorm1d(_InstanceNorm):
    pass


class InstanceNorm2d(_InstanceNorm):
    pass


class _AdaptivePool(Module):
    _fn = None

    def __init__(self, output_size, return_indices=False):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return type(self)._fn(x, self.output_size)

    def extra_repr(self):
        return f"output_size={self.output_size}"


class AdaptiveAvgPool1d(_AdaptivePool):
    _fn = staticmethod(adaptive_avg_pool1d)


class AdaptiveAvgPool2d(_AdaptivePool):
    _fn = staticmethod(adaptive_avg_pool2d)


class AdaptiveMaxPool1d(_AdaptivePool):
    _fn = staticmethod(adaptive_max_pool1d)


class AdaptiveMaxPool2d(_AdaptivePool):
    _fn = staticmethod(adaptive_max_pool2d)


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride or kernel_size, padding

    def forward(self, x):
        return max_pool1d(x, self.kernel_size, self.stride, self.padding)


class AvgPool1d(MaxPool1d):
    def forward(self, x):
        return avg_pool1d(x, self.kernel_size, self.stride, self.padding)


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        k = _triple(kernel_size)
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.stride, self.padding = _triple(stride), _triple(padding)
        self.weight = Parameter(torch.empty(out_channels, in_channels, *k))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_channels * k[0] * k[1] * k[2])
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding)


class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1):
        super().__init__()
        k = _pair(kernel_size)
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, k
        self.stride, self.padding, self.output_padding = _pair(stride), _pair(padding), _pair(output_padding)
        self.weight = Parameter(torch.empty(in_channels, out_channels, *k))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(out_channels * k[0] * k[1])
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)


class EmbeddingBag(Module):
    def __init__(self, num_embeddings, embedding_dim, mode="mean", padding_idx=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.mode = num_embeddings, embedding_dim, mode
        self.weight = Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input, offsets=None, per_sample_weights=None):
        return embedding_bag(input, self.weight, offsets, self.mode, per_sample_weights)


class LazyLinear(Module):
    """in_features is inferred from the first input."""

    def __init__(self, out_features, bias=True):
        super().__init__()
        self.out_features, self._use_bias, self.in_features = out_features, bias, 0
        self.weight = None
        self.bias = None

    def forward(self, x):
        if self.weight is None:
            lin = Linear(x.shape[-1], self.out_features, bias=self._use_bias)
            self.in_features = x.shape[-1]
            self.weight, self.bias = lin.weight, lin.bias
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------- recurrent layers

PackedSequence = namedtuple("PackedSequence", ["data", "batch_sizes", "sorted_indices", "unsorted_indices"])


class _Packed(PackedSequence):
    """A PackedSequence that also remembers the padded batch, which torch-lite's RNNs consume."""
    __slots__ = ()
    _padded = {}

    @property
    def padded(self):
        return _Packed._padded[id(self)]


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lens = [int(v) for v in torch._arr(lengths).tolist()]
    if enforce_sorted and lens != sorted(lens, reverse=True):
        raise RuntimeError("`lengths` array must be sorted in decreasing order when `enforce_sorted` is True. "
                           "You can pass `enforce_sorted=False` to pack_padded_sequence")
    x = input if batch_first else input.transpose(0, 1)  # (B, T, *)
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    T = max(lens)
    rows, sizes = [], []
    for t in range(T):
        keep = [i for i in order if lens[i] > t]
        sizes.append(len(keep))
        rows.append(x[torch.tensor(keep), t])
    data = torch.cat(rows)
    unsorted = sorted(range(len(order)), key=lambda i: order[i])
    p = _Packed(data, torch.tensor(sizes), torch.tensor(order), torch.tensor(unsorted))
    _Packed._padded[id(p)] = (x, lens)
    return p


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    x, lens = sequence.padded
    T = total_length or max(lens)
    mask = (torch.arange(T)[None, :] < torch.tensor(lens)[:, None]).float()
    shape = mask.shape + (1,) * (x.ndim - 2)
    x = x[:, :T] * mask.reshape(shape) + padding_value * (1 - mask.reshape(shape))
    return (x if batch_first else x.transpose(0, 1)), torch.tensor(lens)


def pack_sequence(sequences, enforce_sorted=True):
    from .utils.rnn import pad_sequence
    return pack_padded_sequence(pad_sequence(sequences, batch_first=True), [len(s) for s in sequences], True,
                                enforce_sorted)


class _RNNBase(Module):
    _gates, _mode = 1, "RNN_TANH"

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0,
                 bidirectional=False, nonlinearity="tanh", proj_size=0):
        super().__init__()
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.bias, self.batch_first, self.dropout, self.bidirectional = bias, batch_first, dropout, bidirectional
        self.nonlinearity = nonlinearity
        D = 2 if bidirectional else 1
        G = self._gates * hidden_size
        k = 1 / math.sqrt(hidden_size)
        for layer in range(num_layers):
            for d in range(D):
                sfx = f"_l{layer}" + ("_reverse" if d else "")
                inp = input_size if layer == 0 else hidden_size * D
                setattr(self, "weight_ih" + sfx, Parameter(torch.empty(G, inp).uniform_(-k, k)))
                setattr(self, "weight_hh" + sfx, Parameter(torch.empty(G, hidden_size).uniform_(-k, k)))
                if bias:
                    setattr(self, "bias_ih" + sfx, Parameter(torch.empty(G).uniform_(-k, k)))
                    setattr(self, "bias_hh" + sfx, Parameter(torch.empty(G).uniform_(-k, k)))

    def _cell(self, x, h, c, sfx):
        gi = F.linear(x, getattr(self, "weight_ih" + sfx), getattr(self, "bias_ih" + sfx, None))
        gh = F.linear(h, getattr(self, "weight_hh" + sfx), getattr(self, "bias_hh" + sfx, None))
        return self._step(gi, gh, h, c)

    def _run(self, x, lens, h0, c0, sfx, reverse):
        B, T = x.shape[0], x.shape[1]
        h, c = h0, c0
        outs = [None] * T
        lens_t = torch.tensor(lens)
        for step in range(T):
            t = T - 1 - step if reverse else step
            if reverse:  # each sequence starts from its own last real token
                idx = torch.tensor([max(lens[b] - 1 - step, 0) for b in range(B)])
                xt = x[torch.arange(B), idx]
            else:
                xt = x[:, t]
            nh, nc = self._cell(xt, h, c, sfx)
            valid = (lens_t > step).float()[:, None]
            h = nh * valid + h * (1 - valid)
            if c is not None:
                c = nc * valid + c * (1 - valid)
            outs[step] = nh * valid
        if reverse:  # put outputs back at their time positions
            stacked = torch.stack(outs, dim=1)  # (B, step, H)
            rows = []
            for b in range(B):
                lb = lens[b]
                order = [lb - 1 - t if t < lb else t for t in range(T)]
                rows.append(stacked[b][torch.tensor(order)])
            out = torch.stack(rows)
        else:
            out = torch.stack(outs, dim=1)
        return out, h, c

    def forward(self, input, hx=None):
        packed = isinstance(input, _Packed)
        if packed:
            x, lens = input.padded
        else:
            x = input if self.batch_first else input.transpose(0, 1)
            lens = [x.shape[1]] * x.shape[0]
        B = x.shape[0]
        D = 2 if self.bidirectional else 1
        lstm = self._gates == 4
        if hx is None:
            h0 = torch.zeros(self.num_layers * D, B, self.hidden_size)
            c0 = torch.zeros(self.num_layers * D, B, self.hidden_size) if lstm else None
        else:
            h0, c0 = hx if lstm else (hx, None)
        hs, cs = [], []
        layer_in = x
        for layer in range(self.num_layers):
            outs = []
            for d in range(D):
                sfx = f"_l{layer}" + ("_reverse" if d else "")
                i = layer * D + d
                o, h, c = self._run(layer_in, lens, h0[i], c0[i] if lstm else None, sfx, bool(d))
                outs.append(o)
                hs.append(h)
                cs.append(c)
            layer_in = torch.cat(outs, dim=-1) if D == 2 else outs[0]
            if self.dropout and self.training and layer < self.num_layers - 1:
                layer_in = F.dropout(layer_in, self.dropout, True)
        h_n = torch.stack(hs)
        out = layer_in
        if packed:
            mask = (torch.arange(out.shape[1])[None, :] < torch.tensor(lens)[:, None]).float()[..., None]
            out = pack_padded_sequence(out * mask, lens, batch_first=True, enforce_sorted=False)
        elif not self.batch_first:
            out = out.transpose(0, 1)
        return (out, (h_n, torch.stack(cs))) if lstm else (out, h_n)

    def extra_repr(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers != 1:
            s += f", num_layers={self.num_layers}"
        if self.batch_first:
            s += ", batch_first=True"
        if self.dropout:
            s += f", dropout={self.dropout}"
        if self.bidirectional:
            s += ", bidirectional=True"
        return s


class RNN(_RNNBase):
    _gates = 1

    def _step(self, gi, gh, h, c):
        z = gi + gh
        return (z.tanh() if self.nonlinearity == "tanh" else z.relu()), None


class LSTM(_RNNBase):
    _gates = 4

    def _step(self, gi, gh, h, c):
        i, f, g, o = (gi + gh).chunk(4, dim=-1)
        c2 = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c2.tanh(), c2


class GRU(_RNNBase):
    _gates = 3

    def _step(self, gi, gh, h, c):
        ir, iz, in_ = gi.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
        n = (in_ + r * hn).tanh()
        return (1 - z) * n + z * h, None


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__()
        self._rnn = RNN(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)

    def forward(self, x, h=None):
        h = torch.zeros(x.shape[0], self._rnn.hidden_size) if h is None else h
        return self._rnn._cell(x, h, None, "_l0")[0]


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self._rnn = LSTM(input_size, hidden_size, bias=bias)

    def forward(self, x, hx=None):
        H = self._rnn.hidden_size
        h, c = hx if hx is not None else (torch.zeros(x.shape[0], H), torch.zeros(x.shape[0], H))
        return self._rnn._cell(x, h, c, "_l0")


class GRUCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self._rnn = GRU(input_size, hidden_size, bias=bias)

    def forward(self, x, h=None):
        h = torch.zeros(x.shape[0], self._rnn.hidden_size) if h is None else h
        return self._rnn._cell(x, h, None, "_l0")[0]


# ---------------------------------------------------------------- losses

class MarginRankingLoss(_Loss):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__(reduction, margin=margin)

    def forward(self, x1, x2, target):
        return margin_ranking_loss(x1, x2, target, self.margin, self.reduction)


class TripletMarginLoss(_Loss):
    def __init__(self, margin=1.0, p=2.0, eps=1e-6, swap=False, reduction="mean"):
        super().__init__(reduction, margin=margin, p=p, eps=eps, swap=swap)

    def forward(self, a, p, n):
        return triplet_margin_loss(a, p, n, self.margin, self.p, self.eps, self.swap, self.reduction)


class CosineEmbeddingLoss(_Loss):
    def __init__(self, margin=0.0, reduction="mean"):
        super().__init__(reduction, margin=margin)

    def forward(self, x1, x2, target):
        return cosine_embedding_loss(x1, x2, target, self.margin, self.reduction)


class GaussianNLLLoss(_Loss):
    def __init__(self, full=False, eps=1e-6, reduction="mean"):
        super().__init__(reduction, full=full, eps=eps)

    def forward(self, input, target, var):
        return gaussian_nll_loss(input, target, var, self.full, self.eps, self.reduction)


class PoissonNLLLoss(_Loss):
    def __init__(self, log_input=True, full=False, eps=1e-8, reduction="mean"):
        super().__init__(reduction, log_input=log_input, full=full, eps=eps)

    def forward(self, input, target):
        return poisson_nll_loss(input, target, self.log_input, self.full, self.eps, self.reduction)


class CTCLoss(_Loss):
    def __init__(self, blank=0, reduction="mean", zero_infinity=False):
        super().__init__(reduction, blank=blank, zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction, self.zero_infinity)


EXPORTS = ["Upsample", "PixelShuffle", "PixelUnshuffle", "Dropout1d", "Dropout2d", "Dropout3d", "InstanceNorm1d",
           "InstanceNorm2d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d",
           "MaxPool1d", "AvgPool1d", "Conv3d", "ConvTranspose2d", "EmbeddingBag", "LazyLinear", "RNN", "LSTM", "GRU",
           "RNNCell", "LSTMCell", "GRUCell", "MarginRankingLoss", "TripletMarginLoss", "CosineEmbeddingLoss",
           "GaussianNLLLoss", "PoissonNLLLoss", "CTCLoss"]

"""torch-lite torch.nn.functional (autograd-aware: built from Tensor ops)."""

import math

import numpy as np

import torch
from torch import Tensor, _arr, _make


def _reduce(x, reduction):
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"{reduction} is not a valid value for reduction")


def _fp32_if_autocast(x):
    return x.float() if torch._AUTOCAST[0] is not None and isinstance(x, Tensor) else x


# ---------------------------------------------------------------- activations

def relu(x, inplace=False):
    return x.relu()


def relu6(x):
    return x.clamp(0, 6)


def leaky_relu(x, negative_slope=0.01, inplace=False):
    return torch.where(x > 0, x, x * negative_slope)


def elu(x, alpha=1.0):
    return torch.where(x > 0, x, (x.exp() - 1) * alpha)


def gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)).tanh())
    return 0.5 * x * (1 + (x / math.sqrt(2)).erf())


def silu(x):
    return x * x.sigmoid()


def mish(x):
    return x * softplus(x).tanh()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def softplus(x, beta=1.0, threshold=20.0):
    bx = x * beta
    return torch.where(bx > threshold, x, (bx.exp() + 1).log() / beta)


def logsigmoid(x):
    return -((-x).relu() + (-(x.abs())).exp().log1p())


def softmax(x, dim=None, dtype=None):
    return x.softmax(dim)


def log_softmax(x, dim=None, dtype=None):
    return x.log_softmax(dim)


def glu(x, dim=-1):
    a, b = x.chunk(2, dim)
    return a * b.sigmoid()


# ---------------------------------------------------------------- layers

def linear(x, weight, bias=None):
    torch._event("aten::linear")
    torch._event("aten::addmm" if bias is not None and x.ndim == 2 else "aten::matmul")
    torch._PROF_MUTE[0] += 1
    try:
        out = x @ weight.T
        return out + bias if bias is not None else out
    finally:
        torch._PROF_MUTE[0] -= 1


def bilinear(x1, x2, weight, bias=None):
    y = torch.einsum("bi,oij,bj->bo", x1, weight, x2)
    return y + bias if bias is not None else y


def dropout(x, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return x
    if p == 1:
        return x * 0
    keep = Tensor._wrap((torch._RNG[0].random(x.shape) >= p).astype(x._a.dtype))
    return x * keep / (1 - p)


def embedding(idx, weight, padding_idx=None, max_norm=None):
    return weight[_arr(idx)]


def one_hot(t, num_classes=-1):
    n = int(t._a.max()) + 1 if num_classes == -1 else num_classes
    if t._a.size and (t._a.max() >= n or t._a.min() < 0):
        raise RuntimeError("Class values must be smaller than num_classes.")
    return Tensor._wrap(np.eye(n, dtype=np.int64)[t._a])


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    x = _fp32_if_autocast(x)
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    y = (x - mean) / (var + eps).sqrt()
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def rms_norm(x, normalized_shape, weight=None, eps=None):
    eps = torch.finfo(x.dtype).eps if eps is None else eps
    dims = tuple(range(-len(normalized_shape), 0))
    y = x * ((x ** 2).mean(dim=dims, keepdim=True) + eps).rsqrt()
    return y * weight if weight is not None else y


def normalize(x, p=2.0, dim=1, eps=1e-12):
    return x / x.norm(p=p, dim=dim, keepdim=True).clamp(min=eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    num = (x1 * x2).sum(dim)
    return num / (x1.norm(dim=dim) * x2.norm(dim=dim)).clamp(min=eps)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6):
    return (x1 - x2 + eps).norm(p=p, dim=-1)


def pad(x, pad, mode="constant", value=0.0):
    if mode != "constant":
        width = [(0, 0)] * x.ndim
        for i in range(len(pad) // 2):
            width[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
        np_mode = {"reflect": "reflect", "replicate": "edge", "circular": "wrap"}[mode]
        return _make(np.pad(x._a, width, mode=np_mode))
    width = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):  # pairs start from the LAST dimension
        width[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    out = np.pad(x._a, [(max(a, 0), max(b, 0)) for a, b in width], constant_values=value)
    sl = tuple(slice(-a if a < 0 else 0, out.shape[d] + b if b < 0 else out.shape[d]) for d, (a, b) in enumerate(width))
    inner = tuple(slice(max(a, 0), max(a, 0) + x.shape[d]) for d, (a, b) in enumerate(width))

    def back(g):
        full = np.zeros(out.shape, dtype=np.asarray(g).dtype)
        full[sl] = g
        return full[inner]

    return _make(out[sl], [(x, back)], x._bf16)


def unfold(x, kernel_size, dilation=1, padding=0, stride=1):
    """im2col: (N, C, H, W) -> (N, C*kh*kw, L), with gradients."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    if (dilation if isinstance(dilation, int) else max(dilation)) != 1:
        raise NotImplementedError("torch-lite unfold supports dilation=1 only")
    xp = pad(x, (pw, pw, ph, ph)) if ph or pw else x
    N, C, H, W = xp.shape
    oh, ow = (H - kh) // sh + 1, (W - kw) // sw + 1
    ii = (np.arange(oh) * sh)[:, None, None, None] + np.arange(kh)[None, None, :, None]
    jj = (np.arange(ow) * sw)[None, :, None, None] + np.arange(kw)[None, None, None, :]
    patches = xp[:, :, ii, jj]  # (N, C, oh, ow, kh, kw), gather keeps autograd
    return patches.permute(0, 1, 4, 5, 2, 3).reshape(N, C * kh * kw, oh * ow)


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if groups != 1:
        raise NotImplementedError("torch-lite conv2d supports groups=1 only")
    torch._event("aten::conv2d")
    O, C, kh, kw = weight.shape
    if padding == "same":
        padding = (kh // 2, kw // 2)
    elif padding == "valid":
        padding = 0
    N, _, H, W = x.shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    oh, ow = (H + 2 * ph - kh) // sh + 1, (W + 2 * pw - kw) // sw + 1
    cols = unfold(x, (kh, kw), padding=(ph, pw), stride=(sh, sw))  # (N, C*kh*kw, L)
    out = (weight.reshape(O, -1) @ cols).reshape(N, O, oh, ow)
    return out + bias.reshape(1, O, 1, 1) if bias is not None else out


def max_pool2d(x, kernel_size, stride=None, padding=0):
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    s = stride or k
    N, C, H, W = x.shape
    cols = unfold(x.reshape(N * C, 1, H, W), k, padding=padding, stride=s)
    oh, ow = (H + 2 * padding - k) // s + 1, (W + 2 * padding - k) // s + 1
    return cols.max(dim=1).values.reshape(N, C, oh, ow)


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    s = stride or k
    N, C, H, W = x.shape
    cols = unfold(x.reshape(N * C, 1, H, W), k, padding=padding, stride=s)
    oh, ow = (H + 2 * padding - k) // s + 1, (W + 2 * padding - k) // s + 1
    return cols.mean(dim=1).reshape(N, C, oh, ow)


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    out = conv2d(x.unsqueeze(2), weight.unsqueeze(2), bias, (1, stride), (0, padding), 1, groups)
    return out.squeeze(2)


def group_norm(x, num_groups, weight=None, bias=None, eps=1e-5):
    N, C = x.shape[0], x.shape[1]
    g = x.reshape(N, num_groups, -1)
    y = ((g - g.mean(-1, keepdim=True)) / (g.var(-1, unbiased=False, keepdim=True) + eps).sqrt()).reshape(x.shape)
    shape = (1, C) + (1,) * (x.ndim - 2)
    if weight is not None:
        y = y * weight.reshape(shape)
    if bias is not None:
        y = y + bias.reshape(shape)
    return y


def batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    dims = (0,) + tuple(range(2, x.ndim))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    if training:
        mean, var = x.mean(dim=dims), x.var(dim=dims, unbiased=False)
    else:
        mean, var = running_mean, running_var
    y = (x - mean.reshape(shape)) / (var.reshape(shape) + eps).sqrt()
    if weight is not None:
        y = y * weight.reshape(shape)
    if bias is not None:
        y = y + bias.reshape(shape)
    return y


def interpolate(x, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if mode != "nearest":
        raise NotImplementedError("torch-lite interpolate supports mode='nearest' only")
    H, W = x.shape[-2], x.shape[-1]
    if size is None:
        s = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
        size = (int(H * s[0]), int(W * s[1]))
    oh, ow = (size, size) if isinstance(size, int) else size
    ri = (np.arange(oh) * H // oh)
    ci = (np.arange(ow) * W // ow)
    return x[..., ri[:, None], ci[None, :]]


def adaptive_avg_pool2d(x, output_size):
    if output_size not in (1, (1, 1)):
        raise NotImplementedError("torch-lite adaptive_avg_pool2d supports output_size=1 only")
    return x.mean(dim=(-2, -1), keepdim=True)


# ---------------------------------------------------------------- attention

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    scale = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    if enable_gqa and key.shape[-3] != query.shape[-3]:
        rep = query.shape[-3] // key.shape[-3]
        key, value = key.repeat_interleave(rep, dim=-3), value.repeat_interleave(rep, dim=-3)
    scores = query @ key.transpose(-2, -1) * scale
    L, S = query.shape[-2], key.shape[-2]
    if is_causal:
        scores = scores.masked_fill(Tensor._wrap(np.triu(np.ones((L, S), dtype=bool), 1)), float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype is torch.bool else scores + attn_mask
    w = scores.softmax(-1)
    if dropout_p > 0:
        w = dropout(w, dropout_p, True)
    return w @ value


# ---------------------------------------------------------------- losses

def mse_loss(input, target, reduction="mean"):
    input, target = _fp32_if_autocast(input), _fp32_if_autocast(target)
    return _reduce((input - target) ** 2, reduction)


def l1_loss(input, target, reduction="mean"):
    return _reduce((input - target).abs(), reduction)


def smooth_l1_loss(input, target, reduction="mean", beta=1.0):
    d = (input - target).abs()
    return _reduce(torch.where(d < beta, 0.5 * d ** 2 / beta, d - 0.5 * beta), reduction)


def huber_loss(input, target, reduction="mean", delta=1.0):
    d = (input - target).abs()
    return _reduce(torch.where(d < delta, 0.5 * d ** 2, delta * (d - 0.5 * delta)), reduction)


def nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean"):
    if input.ndim > 2:  # (N, C, d1, ...) -> (N*d1*..., C)
        C = input.shape[1]
        input = input.movedim(1, -1).reshape(-1, C)
        target = target.reshape(-1)
    t = _arr(target)
    if t.dtype.kind not in "iu":
        raise RuntimeError(f"expected scalar type Long but found {target.dtype.name.capitalize()}")
    keep = t != ignore_index
    safe = np.where(keep, t, 0)
    picked = -input[np.arange(input.shape[0]), safe]
    w = np.where(keep, 1.0 if weight is None else _arr(weight)[safe], 0.0).astype(np.float32)
    losses = picked * Tensor._wrap(w)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.sum() / float(w.sum())


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    input = _fp32_if_autocast(input)
    if isinstance(target, Tensor) and target.is_floating_point():  # soft labels (N, C)
        logp = input.log_softmax(1 if input.ndim > 1 else 0)
        return _reduce(-(target * logp).sum(1 if input.ndim > 1 else 0), reduction)
    logp = input.log_softmax(1 if input.ndim > 1 else 0)
    nll = nll_loss(logp, target, weight, ignore_index, reduction)
    if label_smoothing == 0:
        return nll
    C = input.shape[1]
    flat = logp.movedim(1, -1).reshape(-1, C) if logp.ndim > 2 else logp
    keep = Tensor._wrap((_arr(target).reshape(-1) != ignore_index).astype(np.float32))
    smooth = -flat.mean(dim=-1) * keep
    if reduction == "none":
        smooth_r = smooth.reshape(_arr(target).shape)
    elif reduction == "sum":
        smooth_r = smooth.sum()
    else:
        smooth_r = smooth.sum() / keep.sum()
    return (1 - label_smoothing) * nll + label_smoothing * smooth_r


def binary_cross_entropy_with_logits(input, target, weight=None, reduction="mean", pos_weight=None):
    log_sig = logsigmoid(input)
    log_one_minus = logsigmoid(-input)
    pw = 1 if pos_weight is None else pos_weight
    loss = -(pw * target * log_sig + (1 - target) * log_one_minus)
    if weight is not None:
        loss = loss * weight
    return _reduce(loss, reduction)


def binary_cross_entropy(input, target, weight=None, reduction="mean"):
    loss = -(target * input.log().clamp(min=-100) + (1 - target) * (1 - input).log().clamp(min=-100))
    if weight is not None:
        loss = loss * weight
    return _reduce(loss, reduction)


def kl_div(input, target, reduction="mean", log_target=False):
    pointwise = target.exp() * (target - input) if log_target else torch.where(target > 0, target * (target.log() - input), target * 0)
    if reduction == "batchmean":
        return pointwise.sum() / input.shape[0]
    return _reduce(pointwise, reduction)


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    g = -Tensor._wrap(-np.log(torch._RNG[0].random(logits.shape) + 1e-20).astype(np.float32)).log()
    y = ((logits + g) / tau).softmax(dim)
    if hard:
        idx = y.argmax(dim, keepdim=True)
        hard_y = torch.zeros_like(y)
        hard_y.scatter_(dim, idx, 1.0)
        return hard_y - y.detach() + y
    return y

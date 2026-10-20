"""torch-lite torch.nn.attention: backend selection is a no-op (there is only the math backend)."""

import contextlib
import enum


class SDPBackend(enum.Enum):
    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    CUDNN_ATTENTION = 3


@contextlib.contextmanager
def sdpa_kernel(backends, set_priority=False):
    yield


class _FlexModule:
    """torch.nn.attention.flex_attention: score_mod / mask_mod run on broadcast index tensors (the math is exact)."""

    class BlockMask:
        def __init__(self, dense, B, H, Q, KV):
            self.mask, self.shape = dense, (B, H, Q, KV)

        def to_dense(self):
            return self.mask

        def __repr__(self):
            return f"BlockMask(shape={self.shape})"

    @staticmethod
    def _idx(B, H, Q, KV):
        import torch
        return (torch.arange(B).reshape(B, 1, 1, 1), torch.arange(H).reshape(1, H, 1, 1),
                torch.arange(Q).reshape(1, 1, Q, 1), torch.arange(KV).reshape(1, 1, 1, KV))

    def create_block_mask(self, mask_mod, B, H, Q_LEN, KV_LEN, device="cpu", BLOCK_SIZE=128, _compile=False):
        import torch
        b, h, q, kv = self._idx(B or 1, H or 1, Q_LEN, KV_LEN)
        dense = mask_mod(b, h, q, kv)
        dense = torch.broadcast_to(dense, ((B or 1), (H or 1), Q_LEN, KV_LEN))
        return _FlexModule.BlockMask(dense, B, H, Q_LEN, KV_LEN)

    def create_mask(self, mod_fn, B, H, Q_LEN, KV_LEN, device="cpu"):
        return self.create_block_mask(mod_fn, B, H, Q_LEN, KV_LEN).mask

    def flex_attention(self, query, key, value, score_mod=None, block_mask=None, scale=None, enable_gqa=False,
                       return_lse=False, kernel_options=None):
        import math
        import torch
        Bq, Hq, Q, D = query.shape
        KV = key.shape[-2]
        if enable_gqa and key.shape[1] != Hq:
            rep = Hq // key.shape[1]
            key, value = key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)
        scores = query @ key.transpose(-2, -1) * (1 / math.sqrt(D) if scale is None else scale)
        if score_mod is not None:
            b, h, q, kv = self._idx(Bq, Hq, Q, KV)
            scores = score_mod(scores, b, h, q, kv)
        if block_mask is not None:
            scores = scores.masked_fill(~block_mask.mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        out = scores.softmax(-1) @ value
        return (out, lse) if return_lse else out


import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_flex = _FlexModule()
flex_attention = _types.ModuleType("torch.nn.attention.flex_attention")
flex_attention.flex_attention = _flex.flex_attention
flex_attention.create_block_mask = _flex.create_block_mask
flex_attention.create_mask = _flex.create_mask
flex_attention.BlockMask = _FlexModule.BlockMask
_sys.modules["torch.nn.attention.flex_attention"] = flex_attention

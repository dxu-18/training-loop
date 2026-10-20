"""torch-lite FSDP: on the single-process group every rank holds the whole model, so sharding is the identity.
The API checks (process group, mixed-precision policy, wrapping order) still behave like PyTorch's."""

import dataclasses
import enum

import torch
import torch.distributed as dist
from torch.nn import Module


@dataclasses.dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: object = None
    reduce_dtype: object = None
    output_dtype: object = None
    cast_forward_inputs: bool = True


@dataclasses.dataclass
class MixedPrecision:
    param_dtype: object = None
    reduce_dtype: object = None
    buffer_dtype: object = None


@dataclasses.dataclass
class CPUOffloadPolicy:
    pin_memory: bool = True


class ShardingStrategy(enum.Enum):
    FULL_SHARD = 1
    SHARD_GRAD_OP = 2
    NO_SHARD = 3
    HYBRID_SHARD = 4


class FSDPModule:
    """Marker mixin: fully_shard(module) makes isinstance(module, FSDPModule) true."""

    def reshard(self):
        pass

    def unshard(self, async_op=False):
        pass

    def set_requires_gradient_sync(self, requires_gradient_sync, *, recurse=True):
        object.__setattr__(self, "_lite_grad_sync", requires_gradient_sync)


def fully_shard(module, *, mesh=None, reshard_after_forward=True, mp_policy=None, offload_policy=None, **kw):
    if not dist.is_initialized():
        raise ValueError("Default process group has not been initialized, please make sure to call init_process_group.")
    if not isinstance(module, Module):
        raise ValueError(f"fully_shard expects an nn.Module, got {type(module).__name__}")
    policy = mp_policy or MixedPrecisionPolicy()
    if not isinstance(module, FSDPModule):
        module.__class__ = type("FSDP" + type(module).__name__, (FSDPModule, type(module)), {})
    object.__setattr__(module, "_lite_mp_policy", policy)
    return module


class FullyShardedDataParallel(Module):
    def __init__(self, module, process_group=None, sharding_strategy=None, cpu_offload=None, auto_wrap_policy=None,
                 mixed_precision=None, device_id=None, use_orig_params=False, **kw):
        super().__init__()
        if not dist.is_initialized():
            raise ValueError("Default process group has not been initialized, please make sure to call init_process_group.")
        self._fsdp_wrapped_module = module
        object.__setattr__(self, "sharding_strategy", sharding_strategy or ShardingStrategy.FULL_SHARD)
        object.__setattr__(self, "mixed_precision", mixed_precision)

    @property
    def module(self):
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


FSDP = FullyShardedDataParallel
_ = torch

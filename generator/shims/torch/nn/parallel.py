"""torch-lite torch.nn.parallel: DistributedDataParallel over the single-process group."""

import contextlib

import torch
import torch.distributed as dist

from . import Module


class DistributedDataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, broadcast_buffers=True, find_unused_parameters=False,
                 bucket_cap_mb=None, gradient_as_bucket_view=False, static_graph=False):
        super().__init__()
        if not dist.is_initialized():
            raise ValueError("Default process group has not been initialized, please make sure to call init_process_group.")
        if device_ids is not None and any(str(d).startswith("cuda") or isinstance(d, int) and d > 0 for d in device_ids):
            torch._check_device("cuda")
        self.module = module
        object.__setattr__(self, "require_backward_grad_sync", True)
        object.__setattr__(self, "find_unused_parameters", find_unused_parameters)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out

    @contextlib.contextmanager
    def no_sync(self):
        old = self.require_backward_grad_sync
        object.__setattr__(self, "require_backward_grad_sync", False)
        try:
            yield
        finally:
            object.__setattr__(self, "require_backward_grad_sync", old)


DataParallel = DistributedDataParallel

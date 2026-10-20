"""torch-lite torch.distributed: a real single-process group (world_size 1, like `torchrun --nproc_per_node=1`).
Multi-process collectives need real processes, which the browser can't start."""

import os

import numpy as np


class ReduceOp:
    SUM, AVG, PRODUCT, MIN, MAX = "sum", "avg", "product", "min", "max"


_STATE = {"init": False, "rank": 0, "world": 1, "backend": None}


def is_available():
    return True


def is_initialized():
    return _STATE["init"]


def init_process_group(backend=None, init_method=None, world_size=-1, rank=-1, **kwargs):
    world = world_size if world_size > 0 else int(os.environ.get("WORLD_SIZE", 1))
    r = rank if rank >= 0 else int(os.environ.get("RANK", 0))
    if world != 1:
        raise RuntimeError("torch-lite simulates a single process: world_size must be 1 (launch real multi-process "
                           "jobs with torchrun on your machine)")
    if backend not in (None, "gloo", "nccl", "cpu:gloo"):
        raise RuntimeError(f"Invalid backend: '{backend}'")
    _STATE.update(init=True, rank=r, world=world, backend=backend or "gloo")


def _need():
    if not _STATE["init"]:
        raise ValueError("Default process group has not been initialized, please make sure to call init_process_group.")


def get_rank(group=None):
    _need()
    return _STATE["rank"]


def get_world_size(group=None):
    _need()
    return _STATE["world"]


def get_backend(group=None):
    _need()
    return _STATE["backend"]


def barrier(group=None, async_op=False, device_ids=None):
    _need()


def destroy_process_group(group=None):
    _STATE.update(init=False, rank=0, world=1, backend=None)


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    _need()  # with one process the reduction of one tensor is the tensor itself


def reduce(tensor, dst=0, op=ReduceOp.SUM, group=None, async_op=False):
    _need()


def broadcast(tensor, src=0, group=None, async_op=False):
    _need()


def all_gather(tensor_list, tensor, group=None, async_op=False):
    _need()
    tensor_list[0].copy_(tensor)


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    _need()
    output_tensor.copy_(input_tensor)


def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    _need()
    output.copy_(input)


def all_to_all_single(output, input, *a, **k):
    _need()
    output.copy_(input)


def broadcast_object_list(object_list, src=0, group=None):
    _need()


def get_world_size_or_1():
    return _STATE["world"] if _STATE["init"] else 1


_ = np  # numpy kept for future collectives

"""torch-lite torch.distributed.checkpoint (DCP): save / load a (nested) state dict to a checkpoint directory.
`load` fills the tensors of the given state dict IN PLACE, like the real DCP."""

import os
import pickle

import numpy as np

import torch

from . import state_dict  # noqa: F401


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = obj
    return out


def save(state_dict, *, checkpoint_id=None, storage_writer=None, planner=None, process_group=None):
    if checkpoint_id is None:
        raise ValueError("checkpoint_id (a directory) is required")
    os.makedirs(checkpoint_id, exist_ok=True)
    flat = {k: (np.array(v._a) if isinstance(v, torch.Tensor) else v) for k, v in _flatten(state_dict).items()}
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    with open(os.path.join(checkpoint_id, f"__{rank}_0.distcp"), "wb") as f:
        pickle.dump(flat, f)
    with open(os.path.join(checkpoint_id, ".metadata"), "wb") as f:
        pickle.dump(sorted(flat), f)


def load(state_dict, *, checkpoint_id=None, storage_reader=None, planner=None, process_group=None):
    path = os.path.join(checkpoint_id, "__0_0.distcp")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no DCP checkpoint at {checkpoint_id!r}")
    with open(path, "rb") as f:
        saved = pickle.load(f)

    def fill(obj, prefix=""):
        for k, v in list(obj.items()):
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                fill(v, key + ".")
            elif key not in saved:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {key}.")
            elif isinstance(v, torch.Tensor):
                v._a[...] = saved[key]
            else:
                obj[k] = saved[key]

    fill(state_dict)


def async_save(state_dict, *, checkpoint_id=None, **kw):
    save(state_dict, checkpoint_id=checkpoint_id)

    class _Done:
        def result(self):
            return None

    return _Done()

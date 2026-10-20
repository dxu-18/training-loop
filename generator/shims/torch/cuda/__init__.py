"""torch-lite has no GPU: these report "no CUDA" the way PyTorch does on a CPU-only machine."""

from .. import _check_device


def is_available():
    return False


def device_count():
    return 0


def is_bf16_supported(including_emulation=True):
    return False


def synchronize(device=None):
    pass


def empty_cache():
    pass


def memory_allocated(device=None):
    return 0


def max_memory_allocated(device=None):
    return 0


def memory_reserved(device=None):
    return 0


def reset_peak_memory_stats(device=None):
    pass


def manual_seed(seed):
    pass


manual_seed_all = manual_seed


def current_device():
    _check_device("cuda")


def get_device_name(device=None):
    _check_device("cuda")


def set_device(device):
    _check_device("cuda")


class Event:
    def __init__(self, enable_timing=False):
        _check_device("cuda")


class _Amp:
    from ..amp import GradScaler  # deprecated spelling torch.cuda.amp.GradScaler


amp = _Amp()

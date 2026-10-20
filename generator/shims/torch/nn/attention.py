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

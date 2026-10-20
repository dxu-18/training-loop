"""torch.backends stubs (CPU only)."""


class _Flag:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def is_available(self):
        return False

    def is_built(self):
        return False


cudnn = _Flag(benchmark=False, deterministic=False, enabled=True, allow_tf32=True)
mps = _Flag()
cuda = _Flag(matmul=_Flag(allow_tf32=False), is_built=lambda: False)

"""torch-lite torch.utils.cpp_extension: there is no C++ compiler in the browser."""

_MSG = ("C++/CUDA extensions need a compiler (and usually a GPU), which the browser doesn't have. "
        "Run load_inline / load / setup.py builds on your own machine.")


def load_inline(name, cpp_sources, cuda_sources=None, functions=None, **kwargs):
    raise RuntimeError(_MSG)


def load(name, sources, **kwargs):
    raise RuntimeError(_MSG)


class CppExtension:
    def __init__(self, *a, **k):
        raise RuntimeError(_MSG)


CUDAExtension = BuildExtension = CppExtension

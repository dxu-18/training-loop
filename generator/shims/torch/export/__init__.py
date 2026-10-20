"""torch-lite torch.export: capture a module and its example inputs; .module() runs it.
(No graph capture in the browser: the exported program re-runs the original module.)"""

import copy

import torch


class ExportedProgram:
    def __init__(self, mod, args, kwargs, dynamic_shapes):
        self._mod = copy.deepcopy(mod)
        self.example_inputs = (tuple(args), dict(kwargs or {}))
        self.dynamic_shapes = dynamic_shapes
        out = self._mod(*args, **(kwargs or {}))
        self._out_shapes = [tuple(o.shape) for o in (out if isinstance(out, (tuple, list)) else [out]) if hasattr(o, "shape")]

    def module(self):
        prog = self

        class _Exported(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = prog._mod

            def forward(self, *args, **kwargs):
                if prog.dynamic_shapes is None:
                    for a, e in zip(args, prog.example_inputs[0]):
                        if hasattr(a, "shape") and tuple(a.shape) != tuple(e.shape):
                            raise RuntimeError(f"Expected input shape {tuple(e.shape)}, got {tuple(a.shape)} "
                                               f"(export specializes on shapes unless you pass dynamic_shapes)")
                return self.inner(*args, **kwargs)
        return _Exported()

    def state_dict(self):
        return self._mod.state_dict()

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Calling an ExportedProgram directly isn't supported; use ep.module()(...)")

    def __repr__(self):
        return f"ExportedProgram: {type(self._mod).__name__} with output shapes {self._out_shapes}"


def export(mod, args, kwargs=None, *, dynamic_shapes=None, strict=False):
    if not isinstance(args, tuple):
        raise ValueError("Expecting `args` to be a tuple of example positional inputs")
    return ExportedProgram(mod, args, kwargs, dynamic_shapes)


class Dim:
    def __init__(self, name, *, min=None, max=None):
        self.name, self.min, self.max = name, min, max

    AUTO = "AUTO"


def save(ep, f):
    torch.save({"state": ep.state_dict()}, f)

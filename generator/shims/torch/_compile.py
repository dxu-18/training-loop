"""torch-lite torch.compile: same results as eager (there is no compiler in the browser), with option checks
and a simple graph-break detector so `fullgraph=True` and `torch._dynamo.explain` behave sensibly."""

import ast
import functools
import inspect
import textwrap

MODES = (None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
BACKENDS = ("inductor", "eager", "aot_eager", "cudagraphs")
_BREAK_METHODS = {"item": "Tensor.item() returns a Python number", "tolist": "Tensor.tolist()",
                  "numpy": "Tensor.numpy() leaves PyTorch"}


class Unsupported(RuntimeError):
    pass


def graph_breaks(fn):
    """Reasons for graph breaks found in fn's source: print(), .item(), .tolist(), .numpy()."""
    fn = inspect.unwrap(fn)
    if getattr(fn, "_lite_disabled", False):
        return []
    try:
        src = textwrap.dedent(inspect.getsource(fn.forward if hasattr(fn, "forward") and not inspect.isfunction(fn) else fn))
    except (OSError, TypeError):
        return []
    reasons = []
    scope = dict(getattr(fn, "__globals__", {}))
    if getattr(fn, "__closure__", None):
        scope.update(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ if c.cell_contents is not None)))
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "print":
                reasons.append("builtin print() is called")
            elif isinstance(f, ast.Attribute) and f.attr in _BREAK_METHODS:
                reasons.append(_BREAK_METHODS[f.attr])
            elif isinstance(f, ast.Name) and getattr(scope.get(f.id), "_lite_disabled", False):
                reasons.append(f"call to {f.id}, which is marked torch.compiler.disable")
    return reasons


def _check(mode, backend):
    if mode not in MODES:
        raise RuntimeError(f"Unrecognized mode={mode}, should be one of: {', '.join(m for m in MODES if m)}")
    if backend not in BACKENDS and not callable(backend):
        raise RuntimeError(f"invalid backend {backend!r}")


def compile_(model=None, **kw):
    if model is None:
        return lambda m: compile_(m, **kw)
    _check(kw.get("mode"), kw.get("backend", "inductor"))
    from .nn import Module
    if isinstance(model, Module):
        return _optimized_module()(model, kw)

    @functools.wraps(model)
    def compiled(*args, **kwargs):
        if kw.get("fullgraph") and not kw.get("disable"):
            reasons = graph_breaks(model)
            if reasons:
                raise Unsupported(f"torch._dynamo hit a graph break with fullgraph=True: {reasons[0]}")
        return model(*args, **kwargs)

    compiled._lite_compile_options = dict(kw)
    return compiled


def _make_optimized():
    from .nn import Module

    class OptimizedModule(Module):
        def __init__(self, mod, opts):
            super().__init__()
            self._orig_mod = mod
            self._lite_compile_options = dict(opts)

        def forward(self, *args, **kwargs):
            if self._lite_compile_options.get("fullgraph"):
                reasons = graph_breaks(self._orig_mod)
                if reasons:
                    raise Unsupported(f"torch._dynamo hit a graph break with fullgraph=True: {reasons[0]}")
            return self._orig_mod(*args, **kwargs)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._modules["_orig_mod"], name)

    return OptimizedModule


def _optimized_module():
    if "OptimizedModule" not in globals():
        globals()["OptimizedModule"] = _make_optimized()
    return globals()["OptimizedModule"]


def __getattr__(name):
    if name == "OptimizedModule":
        return _optimized_module()
    raise AttributeError(name)


class ExplainOutput:
    def __init__(self, reasons):
        self.graph_break_count = len(reasons)
        self.graph_count = len(reasons) + 1
        self.break_reasons = reasons
        self.op_count = None

    def __repr__(self):
        return f"Graph Count: {self.graph_count}\nGraph Break Count: {self.graph_break_count}\nBreak Reasons: {self.break_reasons}"

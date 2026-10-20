"""torch._dynamo pieces used in practice: explain, reset, mark_dynamic, exc.Unsupported."""

from . import _compile


class exc:
    Unsupported = _compile.Unsupported


class _Config:
    cache_size_limit = 8
    suppress_errors = False


config = _Config()


def explain(fn):
    def run(*args, **kwargs):
        fn(*args, **kwargs)
        target = getattr(fn, "_orig_mod", fn)
        return _compile.ExplainOutput(_compile.graph_breaks(target))
    return run


def reset():
    pass


def mark_dynamic(t, dim):
    pass

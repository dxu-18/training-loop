"""torch.compiler helpers (torch-lite)."""


def disable(fn=None, recursive=True):
    if fn is None:
        return lambda f: disable(f, recursive)
    fn._lite_disabled = True
    return fn


def reset():
    pass


def is_compiling():
    return False


def allow_in_graph(fn):
    return fn

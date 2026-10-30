"""Helper for make_instructions.py, run by an interpreter that has the real numpy, pandas and torch:
reads {section: [key, ...]} on stdin, writes {section: {key: {"n": shown name, "s": signature, "d": description}}}.

A key is a dotted name as a solution writes it ('np.cumsum', 'F.softmax', 'torch.linalg.solve') or a method or
attribute ('.reshape', '.str', '.shape'), looked up on the section's usual classes.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import torch  # noqa: E402
    import torch.nn.functional as F  # noqa: E402
except ImportError:  # descriptions for torch then come from the existing data file
    torch = None

import collections  # noqa: E402
import heapq  # noqa: E402
import math  # noqa: E402

try:
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.axes import Axes  # noqa: E402
    from matplotlib.figure import Figure  # noqa: E402
except ImportError:
    plt = Axes = Figure = None

import time  # noqa: E402

ROOTS = {"np": np, "pd": pd, "math": math, "plt": plt, "time": time}
if torch is not None:
    ROOTS.update({"torch": torch, "F": F, "nn": torch.nn, "S": torch.optim.lr_scheduler, "dist": torch.distributed})
    try:
        import torch.distributed.checkpoint as dcp  # noqa: E402
        import torch.utils.benchmark as benchmark  # noqa: E402
        ROOTS.update({"dcp": dcp, "benchmark": benchmark})
    except ImportError:
        pass


def _resolve(path):
    root, *rest = path.split(".")
    obj = ROOTS.get(root)
    for part in rest:
        if obj is None:
            return None
        try:
            obj = getattr(obj, part)
        except Exception:  # noqa: BLE001
            return None
    return obj


def _method_owners(section):
    """(class, how the name is shown) in lookup order for '.name' keys."""
    common = [(np.random.Generator, "rng."), (collections.deque, "dq."), (heapq, "heapq."), (int, "n.")]
    if section == "numpy":
        return [(np.ndarray, "a."), (np, "np."), (np.ma.MaskedArray, "m.")] + common
    if section == "pandas":
        g = pd.core.groupby
        owners = [(pd.DataFrame, "df."), (pd.Series, "s."), (pd.Series.str, "s.str."), (pd.Series.dt, "s.dt."),
                  (g.DataFrameGroupBy, "df.groupby(...)."), (pd.core.window.rolling.Rolling, "s.rolling(...)."),
                  (pd.Index, "idx."), (pd.Timestamp, "ts."), (pd.Timedelta, "td."), (pd.Series.cat, "s.cat."),
                  (pd.core.resample.Resampler, "s.resample(...)."), (pd, "pd."), (np.ndarray, "a."), (np, "np."),
                  (pd.DataFrame.plot, "df.plot.")] + common
        try:
            from pandas.io.formats.style import Styler
            owners.append((Styler, "df.style."))
        except ImportError:
            pass
        if Axes is not None:
            owners += [(Axes, "ax."), (Figure, "fig.")]
        try:
            owners.insert(5, (pd.core.window.expanding.Expanding, "s.expanding(...)."))
            owners.insert(5, (pd.core.window.ewm.ExponentialMovingWindow, "s.ewm(...)."))
        except AttributeError:
            pass
        return owners
    if section == "pytorch" and torch is not None:
        extra = [(torch.amp.GradScaler, "scaler."), (torch.nn.parallel.DistributedDataParallel, "ddp."),
                 (torch.profiler.profile, "prof."), (time, "time.")]
        try:
            extra.append((torch.utils.benchmark.Timer, "timer."))
        except AttributeError:
            pass
        return [(torch.Tensor, "x."), (torch.nn.Module, "model."), (torch.optim.Optimizer, "opt.")] + extra + [
                (torch.optim.lr_scheduler.LRScheduler, "sched."), (torch, "torch."), (torch.nn, "nn."),
                (torch.utils.data.DataLoader, "loader."), (torch.utils.data.Dataset, "ds."),
                (torch.autograd.Function, "Function."), (torch.nn.functional, "F.")] + common
    return []


def _clean(text):
    text = re.sub(r":(?:attr|func|meth|class|mod|obj|data|ref|math):`~?([^`<]*?)(?:\s*<[^>]*>)?`", r"`\1`", text)
    text = re.sub(r"``([^`]*)``", r"`\1`", text)
    text = re.sub(r"`([^`]+)`_", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SIG_LINE = re.compile(r"^[\w.]+\(.*\)(\s*->\s*\S.*)?$")


def _first_sentence(text):
    m = re.match(r"(.+?[.!?])(\s|$)", text + " ")
    s = m.group(1) if m else text
    while s.endswith(("e.g.", "i.e.", "etc.", "vs.")) and len(s) < len(text):
        m = re.match(re.escape(s) + r"(.+?[.!?])(\s|$)", text + " ")
        if not m:
            break
        s += m.group(1)
    if len(s) > 170:
        s = s[:167].rsplit(" ", 1)[0] + "…"
    return s


def _sig(obj, shown):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    parts = []
    for p in params[:3]:
        if p.kind is p.VAR_POSITIONAL:
            parts.append("*" + p.name)
        elif p.kind is p.VAR_KEYWORD:
            parts.append("**" + p.name)
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            d = repr(p.default)
            parts.append(f"{p.name}={d if len(d) <= 12 and not d.startswith('<') else '…'}")
    if len(params) > 3:
        parts.append("…")
    return f"{shown}({', '.join(parts)})"


def describe(obj, shown, _depth=0):
    """(signature, description) from a docstring."""
    if obj is None or isinstance(obj, (int, float, str, bool, complex)) or type(obj).__name__ in ("dtype", "finfo"):
        return None
    if torch is not None and isinstance(obj, torch.dtype):
        return None
    doc = inspect.getdoc(obj) or ""
    if not doc.strip():
        return None
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
    sig = ""
    while paras and (SIG_LINE.match(paras[0].replace("\n", " ")) or paras[0].startswith((".. ", "Warning", "Note"))):
        first = paras.pop(0).replace("\n", " ")
        if SIG_LINE.match(first) and not sig:
            sig = shown.rsplit(".", 1)[0] + "." + first.split(" ->")[0] if "." in shown else first.split(" ->")[0]
    if not paras:
        return None
    desc = _first_sentence(_clean(paras[0]))
    m = re.match(r"alias (?:for|of) `?([\w.]+?)`?(\(\))?\.?$", desc, re.I)
    if m and _depth < 2:                  # torch.relu: "Alias for torch.nn.functional.relu()" -> describe that one
        target = _resolve(m.group(1).replace("torch.nn.functional", "F")) if m.group(1).startswith("torch") else None
        if target is None and m.group(1).startswith("torch."):
            target = _resolve(m.group(1))
        r = describe(target, shown, _depth + 1) if target is not None else None
        if r:
            return (sig or r[0]), r[1], r[2]
        return None
    if len(desc) < 8 or desc.lower().startswith(("see ", "alias", "deprecated", "this function is deprecated")):
        return None
    callable_ = callable(obj) and not isinstance(obj, type) or isinstance(obj, type) and shown[:1].isupper()
    if not sig and callable(obj):
        sig = _sig(obj, shown)
    return sig, desc, callable_


def lookup(section, key):
    if not key.startswith("."):
        obj = _resolve(key)
        r = describe(obj, key)
        if r:
            return {"n": key, "s": r[0], "d": r[1]}
        return None
    name = key[1:]
    for owner, prefix in _method_owners(section):
        try:
            obj = inspect.getattr_static(owner, name)
            live = getattr(owner, name)
        except Exception:  # noqa: BLE001
            continue
        shown = prefix + name
        prop = isinstance(obj, property) or type(obj).__name__ in ("getset_descriptor", "member_descriptor",
                                                                  "CachedAccessor", "Accessor")
        r = describe(live if not isinstance(obj, property) else obj, shown)
        if r:
            sig, desc, _ = r
            return {"n": shown if prop else shown + "()", "s": "" if prop else sig, "d": desc}
    return None


def main():
    req = json.load(sys.stdin)
    out = {}
    for section, keys in req.items():
        if section == "pytorch" and torch is None:
            continue
        out[section] = {}
        for key in keys:
            try:
                r = lookup(section, key)
            except Exception:  # noqa: BLE001
                r = None
            if r:
                out[section][key] = r
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()

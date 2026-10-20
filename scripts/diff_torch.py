"""Differential test: does torch-lite give the same answers as real PyTorch?

Runs every PyTorch template's reference solution (and every catalog item) under torch-lite and under a real
PyTorch install, and compares the graded values. Problems whose setup draws random numbers can't be compared
(the two random streams differ), so a problem is compared only when all tensors the setup creates are equal
in both runs.

    python scripts/diff_torch.py --torch-python /path/to/venv-with-torch/bin/python [--seeds 3] [--module 3.01]

The real-torch half runs in that interpreter (`--worker`); it imports this repo's grader but not torch-lite.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def to_plain(v, depth=0):
    """Graded value -> picklable plain data (tensors become ('T', dtype, ndarray))."""
    import numpy as np
    if depth > 6:
        return ("?", type(v).__name__)
    if type(v).__module__.split(".")[0] == "torch" and hasattr(v, "detach"):
        a = v.detach()
        dt = str(a.dtype).replace("torch.", "")
        if dt == "bfloat16":
            a = a.float()
        return ("T", dt, np.array(a.cpu().numpy() if hasattr(a, "cpu") else a.numpy()))
    if type(v).__name__ == "dtype" and type(v).__module__.split(".")[0] == "torch":
        return ("D", str(v))
    if type(v).__name__ == "Size":
        return ("L", [int(x) for x in v])
    if isinstance(v, np.ndarray):
        return ("A", v.dtype.str, v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return ("M", {str(k): to_plain(x, depth + 1) for k, x in v.items()})
    if isinstance(v, (list, tuple)):
        return ("L", [to_plain(x, depth + 1) for x in v])
    if isinstance(v, (int, float, str, bool, type(None), complex)):
        return v
    return ("?", type(v).__name__)


def setup_tensors(problem: dict) -> dict:
    """The tensors the setup creates (to tell deterministic setups from random ones)."""
    from generator import grader
    ns: dict = {"__name__": "__main__"}
    import contextlib
    import io
    with grader._sandbox():
        with contextlib.redirect_stdout(io.StringIO()):
            grader._exec(problem.get("setup", ""), ns, "<setup>")
        torch = sys.modules.get("torch")
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    out = {}

    def is_t(v):
        return type(v).__module__.split(".")[0] == "torch" and hasattr(v, "detach")

    for k, v in ns.items():
        if is_t(v):
            out[k] = to_plain(v)
        elif hasattr(v, "state_dict") and hasattr(v, "parameters") and callable(v.state_dict):
            try:
                for n, t in v.state_dict().items():
                    out[f"{k}.{n}"] = to_plain(t)
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(v, dict):
            out.update({f"{k}[{n}]": to_plain(t) for n, t in v.items() if is_t(t)})
        elif isinstance(v, (list, tuple)):
            out.update({f"{k}[{i}]": to_plain(t) for i, t in enumerate(v) if is_t(t)})
    return out


def run_all(problems: list[dict]) -> list:
    """For each problem: (setup tensors, graded value of the reference solution) or an error string."""
    from generator import grader
    out = []
    for p in problems:
        try:
            st = setup_tensors(p)
            val, _ = grader._run_once(p, p["solution"], p.get("setup", ""), "<solution>")
            out.append((st, to_plain(val)))
        except Exception as e:  # noqa: BLE001
            out.append(f"{type(e).__name__}: {e}")
        torch = sys.modules.get("torch")
        if torch is not None and not getattr(torch, "LITE", False):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            torch.manual_seed(0)
    return out


def same(a, b, tol=1e-4) -> str | None:
    import numpy as np
    if isinstance(a, tuple) and a and a[0] == "T":
        if not (isinstance(b, tuple) and b and b[0] == "T"):
            return f"tensor vs {b!r:.60}"
        if a[1] != b[1]:
            return f"dtype {a[1]} vs {b[1]}"
        x, y = a[2], b[2]
        if x.shape != y.shape:
            return f"shape {x.shape} vs {y.shape}"
        if x.dtype.kind in "fc":
            ok = np.allclose(x, y, rtol=tol, atol=tol, equal_nan=True)
        else:
            ok = np.array_equal(x, y)
        return None if ok else f"values differ (max diff {np.max(np.abs(x.astype(complex) - y.astype(complex))) if x.size else 0:.3g})"
    if isinstance(a, tuple) and a and a[0] in ("L", "M"):
        if not (isinstance(b, tuple) and b and b[0] == a[0]) or len(a[1]) != len(b[1]):
            return f"structure {str(a)[:60]} vs {str(b)[:60]}"
        items = zip(a[1], b[1]) if a[0] == "L" else ((a[1][k], b[1].get(k)) for k in a[1])
        for x, y in items:
            msg = same(x, y, tol)
            if msg:
                return msg
        return None
    if isinstance(a, tuple) and a and a[0] == "A":
        return same(("T", "x", a[2]), ("T", "x", b[2]) if isinstance(b, tuple) and b[0] == "A" else b, tol)
    if isinstance(a, float) and isinstance(b, (int, float)):
        return None if abs(a - b) <= tol * max(1, abs(a)) else f"{a} vs {b}"
    return None if a == b else f"{a!r:.60} vs {b!r:.60}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-python")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--module", default="")
    ap.add_argument("--worker", nargs=2, metavar=("IN", "OUT"))
    args = ap.parse_args()
    if args.worker:
        import generator  # noqa: F401  (real torch is importable here, so torch-lite isn't used)
        import torch
        assert not getattr(torch, "LITE", False), "the worker must run with real PyTorch"
        probs = json.loads(Path(args.worker[0]).read_text())
        Path(args.worker[1]).write_bytes(pickle.dumps(run_all(probs)))
        return
    import generator  # noqa: F401
    from generator import core
    from generator.core import build, templates
    probs, labels = [], []
    for t in templates("pytorch"):
        if args.module and t.module != args.module or t.tier == "challenge":
            continue
        core._PICKED.clear()
        build(t, 0)
        n_items = core._PICKED[0] if core._PICKED else 1
        for i in range(n_items):
            core._FORCE_ITEM[0] = i if core._PICKED else None
            for s in range(args.seeds):
                p = build(t, s).to_dict()
                probs.append(p)
                labels.append(f"{t.module} {t.name}" + (f"[{i}]" if core._PICKED else "") + f" seed {s}")
        core._FORCE_ITEM[0] = None
    lite = run_all(probs)
    with tempfile.TemporaryDirectory() as d:
        inp, outp = Path(d, "in.json"), Path(d, "out.pkl")
        inp.write_text(json.dumps(probs))
        log = Path(d, "worker.log")
        with open(log, "w") as fh:
            r = subprocess.run([args.torch_python, __file__, "--worker", str(inp), str(outp)], cwd=ROOT, stdout=fh, stderr=fh)
        if r.returncode:
            print(log.read_text()[-3000:])
            sys.exit(1)
        real = pickle.loads(outp.read_bytes())
    import re
    randomish = re.compile(r"\b(rand|randn|randint|randperm|rand_like|randn_like|randint_like|normal|bernoulli|multinomial|"
                           r"poisson|uniform_|normal_|exponential_|random_|random_split|Generator|svd_lowrank|pca_lowrank|"
                           r"RandomSampler|SubsetRandomSampler|WeightedRandomSampler|gumbel_softmax|kaiming_\w+|xavier_\w+|"
                           r"trunc_normal_|orthogonal_|nn\.Linear|nn\.Conv\w*|nn\.Embedding\w*|nn\.LSTM|nn\.GRU|nn\.RNN|"
                           r"LazyLinear|nn\.Bilinear|nn\.MultiheadAttention|nn\.Transformer\w*|dropout|Dropout)\b|shuffle=True")
    compared = skipped = 0
    bad = []
    for lab, p, a, b in zip(labels, probs, lite, real):
        if randomish.search(p["solution"] + "\n" + (p.get("post") or "")):
            skipped += 1  # the solution draws random numbers (or random init): streams differ
            continue
        if isinstance(b, str) and "No module named 'triton'" in b:
            skipped += 1  # real Triton needs a GPU
            continue
        if isinstance(b, str):
            bad.append(f"{lab}: real torch raised {b[:150]}")
            continue
        if isinstance(a, str):
            bad.append(f"{lab}: torch-lite raised {a[:150]}")
            continue
        if any(same(a[0].get(k), v) for k, v in b[0].items()) or set(a[0]) != set(b[0]):
            skipped += 1  # random setup: inputs differ between the two
            continue
        compared += 1
        msg = same(b[1], a[1])
        if msg:
            bad.append(f"{lab}: {msg}")
    print(f"compared {compared} problems with real PyTorch ({skipped} skipped: random numbers involved), {len(bad)} mismatches")
    for b_ in bad[:200]:
        print("  ", b_)


if __name__ == "__main__":
    main()

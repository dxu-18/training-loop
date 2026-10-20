"""How much of COMP_GUIDE's syntax do the generated problems actually practise?

For one section, collect the API names used in the guide's code blocks, then generate many problems per
template and collect the names used in their code (setup, solution, check code, and `code` spans in the
prompt). A guide name is covered if a problem uses it. `torch.cumsum` and `x.cumsum(...)` count as the same
API; keyword arguments are reported separately. Names the guide defines itself (`self.net`, `TinyGPT(vocab=)`)
are ignored, and EXCLUDED lists what can't run in the browser, with the reason.

    python scripts/guide_coverage.py pytorch            # per-module report
    python scripts/guide_coverage.py numpy --seeds 300 --module 1.10
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SECTIONS = {"numpy": 1, "pandas": 2, "pytorch": 3}

# alias -> prefix kept in the key ("" = the top-level namespace, so torch.cumsum and x.cumsum() match)
ALIASES = {"np": "", "pd": "", "torch": "", "F": "F.", "nn": "nn.", "tl": "tl.", "triton": "triton.", "dist": "dist.",
           "dcp": "dcp.", "benchmark": "benchmark.", "S": "lr_scheduler.", "plt": "plt.", "math": "math."}
PY_METHODS = {"append", "extend", "update", "items", "keys", "values", "get", "format", "pop", "add", "copy",
              "setdefault", "remove", "insert", "count", "index", "sort", "split", "join", "strip", "lower", "upper",
              "startswith", "endswith", "replace", "read", "write", "close", "seek", "flush"}
PY_METHODS_KEEP = {"pandas": {"count", "index", "sort", "split", "join", "strip", "lower", "upper", "startswith",
                              "endswith", "replace", "get", "pop", "update", "copy", "add", "values", "keys", "items"},
                   "numpy": {"sort", "copy", "split", "flush", "index", "count"},
                   "pytorch": {"split", "add", "copy", "index", "sort", "count"}}

# Guide syntax that cannot be practised in the browser (Pyodide), with the reason.
EXCLUDED = {
    "pytorch": {
        "cuda.is_available": "", "cuda.device_count": "no GPU", "cuda.get_device_name": "no GPU", "cuda.set_device": "no GPU",
        "cuda.empty_cache": "no GPU", "cuda.mem_get_info": "no GPU", "cuda.memory_allocated": "no GPU",
        "cuda.memory_summary": "no GPU", "cuda.memory._dump_snapshot": "no GPU", "cuda.memory._record_memory_history": "no GPU",
        "cuda.Event": "no GPU", ".elapsed_time": "no GPU", ".cuda": "no GPU", ".pin_memory": "no GPU",
        "cuda.is_bf16_supported": "no GPU", "cuda.amp": "deprecated spelling", ".get_device_name": "no GPU",
        ".is_bf16_supported": "no GPU", ".empty_cache": "no GPU", ".mem_get_info": "no GPU", ".memory_allocated": "no GPU",
        ".memory_summary": "no GPU", "._dump_snapshot": "no GPU", "._record_memory_history": "no GPU", ".Event": "no GPU",
        ".device_count": "no GPU", ".set_device": "no GPU",
        "onnx.export": "needs the onnx package", "utils.cpp_extension": "needs a C++ compiler",
        ".MNIST": "torchvision isn't in the browser", ".ToTensor": "torchvision isn't in the browser",
        ".DPOTrainer": "trl isn't in the browser", ".gradient_checkpointing_enable": "Hugging Face API",
        ".past_key_values": "Hugging Face API", ".summary": "torchinfo isn't in the browser", ".record": "CUDA events: no GPU",
    },
    "pandas": {
        "read_excel": "openpyxl isn't in Pyodide", "ExcelWriter": "openpyxl isn't in Pyodide", ".to_excel": "openpyxl isn't in Pyodide",
        "read_parquet": "loading pyarrow switches pandas' string storage mid-session", ".to_parquet": "see read_parquet",
        ".to_markdown": "tabulate isn't in Pyodide", "kde": "needs SciPy (not loaded on the plotting page)",
    },
    "numpy": {"trapz": "removed in NumPy 2.4 (use np.trapezoid, which is practised)",
              "memmap": "writable memory maps don't work in Pyodide", "flush": "writable memory maps don't work in Pyodide"},
}


@lru_cache(maxsize=None)
def guide_blocks(section: str) -> dict[str, str]:
    """module id -> the guide's Python code for that module."""
    num = SECTIONS[section]
    text = (ROOT / "COMP_GUIDE.md").read_text()
    out: dict[str, list[str]] = {}
    cur, in_code = None, False
    for line in text.splitlines():
        m = re.match(r"### (\d+)\.(\d+) ", line)
        if m:
            cur = f"{m[1]}.{m[2]}" if int(m[1]) == num else None
            continue
        if line.startswith("## "):
            cur = None
        if line.startswith("```"):
            in_code = not in_code
            continue
        if cur and in_code:
            out.setdefault(cur, []).append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def user_names(code: str) -> set[str]:
    """Names the code defines itself: functions, classes, parameters, assigned variables and attributes."""
    names = set(re.findall(r"\bdef\s+(\w+)", code)) | set(re.findall(r"\bclass\s+(\w+)", code))
    for params in re.findall(r"\bdef\s+\w+\(([^)]*)\)", code):
        names |= {p.split("=")[0].split(":")[0].strip().lstrip("*") for p in params.split(",")}
    names |= set(re.findall(r"self\.(\w+)\s*(?:,\s*self\.\w+\s*)*=", code))
    names |= set(re.findall(r"self\.\w+\s*,\s*self\.(\w+)\s*=", code))
    names |= set(re.findall(r"^\s*(\w+)\s*(?:,\s*\w+\s*)*=(?!=)", code, re.M))
    return names - {""}


def keys(code: str, section: str, ignore: set[str] = frozenset()) -> tuple[set[str], set[str]]:
    """(API keys, keyword-argument keys) used in code."""
    code = re.sub(r"(?m)#.*$", lambda m: m[0] if "·" in m[0] or ":" in m[0] else "", code)  # keep API-list comments
    code = re.sub(r"(?s)(\"\"\"|\'\'\').*?\1", '""', code)  # string contents aren't syntax ('model.pt')
    code = re.sub(r"(?<![\w])[rbfu]?(\"[^\"\n]*\"|'[^'\n]*')", '""', code)
    api, kws = set(), set()
    keep = PY_METHODS_KEEP.get(section, set())
    for m in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)", code):
        root, rest = m[1], m[2][1:].split(".")
        if root in ALIASES:
            api.add(ALIASES[root] + ".".join(rest))
            continue
        if root in ("self", "ctx"):
            rest = rest[1:]
        for attr in rest:  # x.mean().round() -> .mean, .round
            if attr in ignore or (attr in PY_METHODS and attr not in keep):
                continue
            api.add("." + attr)
    for m in re.finditer(r"[)\]]((?:\s*\.\s*[A-Za-z_]\w*)+)", code):  # (...).a.b() chains after a closing paren
        for attr in re.findall(r"[A-Za-z_]\w*", m[1]):
            if attr not in ignore and (attr not in PY_METHODS or attr in keep):
                api.add("." + attr)
    for m in re.finditer(r"[(,]\s*([a-z_]\w*)=(?!=)", code):
        if m[1] not in ignore:
            kws.add(m[1] + "=")
    return api, kws


NOISE = {"py", "cpp", "cu", "org", "json", "pickle", "perfetto", "dev", "pt", "csv", "npy", "txt", "torch", "nn",
         "functional", "np", "pd", "rnn", "linalg", "fft", "optim", "lr_scheduler", "utils", "data", "random", "def"}


def canon(key: str) -> str:
    """One name per API: torch.linalg.solve, np.cumsum / x.cumsum() and S.StepLR map to 'solve', 'cumsum', 'StepLR'."""
    return key.rsplit(".", 1)[-1]


def covered(key: str, corpus: set[str]) -> bool:
    return key in corpus


def problem_code(p) -> str:
    spans = "\n".join(re.findall(r"`([^`\n]+)`", p.prompt)) + "\n" + "\n".join(re.findall(r"```\n(.*?)```", p.prompt, re.S))
    return "\n".join([p.setup, p.solution, p.post or "", spans])


CRASHES: list[str] = []


def report(section: str, seeds: int, only: str = "") -> dict:
    import generator  # noqa: F401  (registers templates; puts torch-lite on the path)
    from generator import core
    from generator.core import build, templates

    def builds(t):
        """Every catalog item (forced) plus `seeds` random builds of the template."""
        core._PICKED.clear()
        core._FORCE_ITEM[0] = None
        yield build(t, 0)
        n_items = core._PICKED[0] if core._PICKED else 0
        for i in range(n_items):
            core._FORCE_ITEM[0] = i
            try:
                for s in range(2):
                    yield build(t, s)
            finally:
                core._FORCE_ITEM[0] = None
        for s in range(1, seeds if t.tier != "challenge" else 2):
            yield build(t, s)
    excluded = EXCLUDED.get(section, {})
    result = {}
    for mid, code in guide_blocks(section).items():
        if only and mid != only:
            continue
        ignore = user_names(code)
        g_api, g_kw = keys(code, section, ignore)
        g_api = {canon(k) for k in g_api if k not in excluded and "." + canon(k) not in excluded and canon(k) not in excluded
                 and canon(k) not in ignore and canon(k) not in NOISE and (len(canon(k)) > 1 or canon(k) == "T")}
        corpus = []
        for t in templates(section, mid):
            try:
                corpus.extend(problem_code(p) for p in builds(t))
            except Exception as e:  # noqa: BLE001 - a template bug: report it, keep measuring
                CRASHES.append(f"{mid} {t.name}: {type(e).__name__}: {e}")
        c_api, c_kw = keys("\n".join(corpus), section)
        c_api = {canon(k) for k in c_api}
        miss_api = sorted(k for k in g_api if not covered(k, c_api))
        miss_kw = sorted(g_kw - c_kw)
        result[mid] = {"api": len(g_api), "api_missing": miss_api, "kw": len(g_kw), "kw_missing": miss_kw}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", choices=sorted(SECTIONS))
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--module", default="")
    ap.add_argument("--kw", action="store_true", help="also list missing keyword arguments")
    args = ap.parse_args()
    rep = report(args.section, args.seeds, args.module)
    tot = sum(r["api"] for r in rep.values())
    miss = sum(len(r["api_missing"]) for r in rep.values())
    ktot = sum(r["kw"] for r in rep.values())
    kmiss = sum(len(r["kw_missing"]) for r in rep.values())
    print(f"{args.section}: {tot - miss}/{tot} guide APIs practised ({100 * (tot - miss) / max(tot, 1):.0f}%), "
          f"keyword arguments {ktot - kmiss}/{ktot}")
    for mid, r in rep.items():
        print(f"  {mid}: {r['api'] - len(r['api_missing'])}/{r['api']}  missing: {', '.join(r['api_missing'])}")
        if args.kw and r["kw_missing"]:
            print(f"        kwargs missing: {', '.join(r['kw_missing'])}")
    for c in CRASHES[:20]:
        print("  CRASH", c)
    excl = EXCLUDED.get(args.section, {})
    if excl:
        print(f"  excluded (can't run in the browser): {len(excl)} names, e.g. " +
              ", ".join(f"{k} ({v})" for k, v in list(excl.items())[:3] if v))


if __name__ == "__main__":
    main()

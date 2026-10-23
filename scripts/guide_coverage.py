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


# ---------------------------------------------------------------- bash (section 7)
# The guide's bash is commands, options and syntax rather than Python APIs, so each module lists what it uses
# (bashlite.features names: "grep -c", "tar -x", "<(", "param:%", "[[ -f", ...); 7.17 also lists keys (C-r, M-.).
# A feature is practised when some reference solution of that module uses it.
GUIDE_BASH = {
    "7.01": "echo, $name, bash --version, man, ls --help, help, type, which, command -v, whatis, apropos, tldr, clear, "
            "history, tail, exit, |, sed -i, date -v, date -d",
    "7.02": "pwd, ls, ls -l, ls -a, ls -h, ls -t, ls -S, ls -1, ls -d, tree -L, cd, cd -, pushd, popd, mkdir, mkdir -p, "
            "{,}, touch, rmdir, cp, cp -r, cp -i, cp -p, mv, mv -n, rm, rm -i, rm -r, rm -f, ln -s, readlink, realpath, "
            "basename, dirname, file, stat, wc -l, glob, ~",
    "7.03": "cat, cat -n, tac, nl, less, head, head -n, head -c, tail -n, tail -f, tail -F, diff, diff -u, diff -r, cmp, "
            "comm, <(, sort, md5sum, sha256sum, xxd, strings, column -s, column -t, less -S, |",
    "7.04": "ls -l, chmod, chmod -R, chown, sudo, $name, tar -c, tar -z, tar -v, tar -f, tar -x, tar -C, tar -t, tar -J, "
            "zip -r, unzip -d, unzip -l, gzip, gunzip, zcat, zgrep, du -s, du -h, du -d, sort -h, df -h, head",
    "7.05": "find -name, find -iname, find -type, find -size, find -mtime, find -maxdepth, find -not, find -path, "
            "find -delete, find -exec, find -print0, xargs -0, xargs -n, cp -t, grep, grep -i, grep -n, grep -v, grep -c, "
            "grep -l, grep -w, grep -o, grep -r, grep --include, grep --exclude-dir, grep -E, grep -A, grep -B, grep -F, "
            "grep -P, rg, rg -t, rg -l, wc -l",
    "7.06": "cut -d, cut -f, cut -c, sort, sort -n, sort -r, sort -u, sort -h, sort -t, sort -k, uniq -c, uniq -d, tr, "
            "tr -d, tr -s, sed, sed -i, sed -n, sed -E, awk, awk -F, awk -v, paste -d, paste -s, bc, join -t, split -l, "
            "jq, jq -r, jq -s, jq -c, seq, xargs -I{}, xargs -n, xargs -P, wget -q, <, |",
    "7.07": ">, >>, 2>, 2>&1, &>, <, |, tee, tee -a, <<, param::-, <<<, <(, set -o, diff, grep, python",
    "7.08": "$name, ${, echo, readonly, unset, export, $(, date, wc -l, ls, :, param::-, param::=, param::?, param:len, "
            "param:sub, param:#, param:##, param:%, param:/, param://, param:^^, param:,,, {..}, {,}, mkdir -p, glob, "
            "shopt -s, **, printf",
    "7.09": "arr=(, +=, array, array[@], param:len, param:!, param:sub, for, declare -A, mapfile -t, $((, ((, bc, awk, "
            "python -c",
    "7.10": "if, [[ -f, [[ -d, ((, [[ !, &&, ||, case, for, for((, while, read -r, <(, until, ping -c, sleep, {..}, "
            "[[ -eq, continue, break, exit, tail -n, mkdir -p, cd",
    "7.11": "set -e, set -u, set -o, function, >&2, exit, printf, $(, date, $*, getopts, case, shift, $((, [[ -eq, $#, "
            "||, mktemp -d, trap, local, python, chmod, bash -x, shellcheck, source, $1, $0, $?",
    "7.12": "ps, grep, pgrep -a, pgrep -f, top, htop, kill, kill -9, pkill -f, &, jobs -l, fg, bg, disown -h, wait, "
            "nohup, 2>&1, $!, time, timeout, sleep, watch -n, nvidia-smi, lsof -i, lsof -t, nice -n, uptime, free -h, "
            "nproc, lscpu, uname -a, whoami, hostname, python",
    "7.13": "env, printenv, tr, export, $name, alias, unalias, set -a, set +a, source, crontab, crontab -e, date, python",
    "7.14": "ssh, ssh -p, ssh -i, ssh-keygen -t, ssh-keygen -C, ssh-copy-id, eval, ssh-agent, ssh-agent -s, ssh-add, "
            "ssh -L, ssh -N, ssh -f, ssh -J, scp, scp -r, rsync -a, rsync -v, rsync -z, rsync -P, rsync --exclude, "
            "rsync -n, rsync --delete, curl -O, curl -L, curl -o, curl -s, jq, curl -X, curl -H, curl -d, wget -c, "
            "tmux new, tmux ls, tmux attach, tmux kill-session, tmux -s, tmux -t",
    "7.15": "python3 -m venv, source, deactivate, which, python -m pip, pip install, pip -U, pip install -r, "
            "pip install -e, pip freeze, pip list, pip --outdated, pip show, pip uninstall, pip -y, uv venv, "
            "uv pip install, uv init, uv add, uv sync, uv run, uv python, uv lock, conda create, conda activate, "
            "conda install, conda -c, conda env, conda env create, conda remove, conda --all, python -c, jupyter, "
            "python -m ipykernel",
    "7.16": "nvidia-smi, nvidia-smi -l, nvtop, nvidia-smi --query-gpu, nvidia-smi --format, "
            "nvidia-smi --query-compute-apps, python, torchrun, nohup, date, $(, for, xargs -P, xargs -I{}, seq, sh -c, "
            "grep -h, sort -t, sort -k, tail, grep -o, cut -d, tail -f, grep --line-buffered, du -sh, ls -t, rm -f, "
            "tensorboard, python -m http.server",
    "7.17": "Tab, C-a, C-e, M-b, M-f, C-u, C-k, C-w, C-y, C-r, Enter, Esc, C-l, C-c, C-d, C-z, M-., !!, sudo, !$, !str, "
            "^^, set -o",
}
# guide features no problem practises, with the reason
EXCLUDED_BASH = {"7.13": {"crontab -e": "needs an interactive editor (crontab - is practised instead)"}}


def bash_features(p) -> set[str]:
    from generator.bashlite.features import features
    from generator.bashlite.keys import key_features, parse_keys
    if "#: keys" in p.setup:
        out = key_features(p.solution)
        typed = []
        for kind, v in parse_keys(p.solution):
            if kind == "text":
                typed.append(v)
            elif v == "Enter":
                typed.append("\n")
        return out | features("".join(typed))
    out = features(p.solution)
    return out | {f.replace("python3", "python", 1) for f in out if f.startswith("python3")}


def report_bash(seeds: int, only: str = "") -> dict:
    import generator  # noqa: F401
    from generator import core
    from generator.core import build, templates
    result = {}
    for mid, spec in GUIDE_BASH.items():
        if only and mid != only:
            continue
        want = {x.strip() for x in spec.split(", ")} - set(EXCLUDED_BASH.get(mid, {}))
        have: set[str] = set()
        for t in templates("bash", mid):
            try:
                core._PICKED.clear()
                core._FORCE_ITEM[0] = None
                have |= bash_features(build(t, 0))
                n_items = core._PICKED[0] if core._PICKED else 0
                for i in range(n_items):
                    core._FORCE_ITEM[0] = i
                    try:
                        for s in range(2):
                            have |= bash_features(build(t, s))
                    finally:
                        core._FORCE_ITEM[0] = None
                for s in range(1, seeds if t.tier != "challenge" else 3):
                    have |= bash_features(build(t, s))
            except Exception as e:  # noqa: BLE001
                CRASHES.append(f"{mid} {t.name}: {type(e).__name__}: {e}")
        result[mid] = {"api": len(want), "api_missing": sorted(want - have), "kw": 0, "kw_missing": []}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", choices=sorted(SECTIONS) + ["bash"])
    ap.add_argument("--seeds", type=int, default=150)
    ap.add_argument("--module", default="")
    ap.add_argument("--kw", action="store_true", help="also list missing keyword arguments")
    args = ap.parse_args()
    rep = report_bash(args.seeds, args.module) if args.section == "bash" else report(args.section, args.seeds, args.module)
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

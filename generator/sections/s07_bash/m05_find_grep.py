"""7.05 Finding files & searching text."""

from ._common import MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template, train_log

M = "7.05"

PY = ["import torch\nimport torch.nn as nn\n", "import numpy as np\nimport pandas as pd\n", "import os\nimport sys\n",
      "from pathlib import Path\nimport json\n", "import pandas as pd\nimport matplotlib.pyplot as plt\n",
      "import wandb\nimport torch\n"]


def _repo(rng, n=6, todo=True):
    """A small Python repo with a .venv, __pycache__ dirs, docs and data."""
    spec = {}
    mods = rng.sample(["train", "model", "data", "utils", "eval", "plot", "config", "losses", "metrics"], n)
    for m in mods:
        d = rng.choice(["src", "src/core", "scripts", "tests"])
        body = rng.choice(PY) + "".join(
            f"def {rng.choice(WORDS)}_{i}(x):\n    return x * {rng.randint(1, 9)}"
            + (f"  # TODO: {rng.choice(WORDS)}" if todo and rng.random() < 0.35 else "") + "\n"
            for i in range(rng.randint(1, 4)))
        spec[f"{d}/{m}.py"] = body
        if rng.random() < 0.5:
            spec[f"{d}/__pycache__/{m}.cpython-312.pyc"] = "\x00bytecode\n"
    spec[".venv/lib/python3.12/site-packages/pkg/core.py"] = "import torch  # TODO vendored\nx = 1\n"
    spec["README.md"] = f"# {rng.choice(WORDS)}\nTODO: write docs\n"
    spec["docs/Readme_old.txt"] = "old readme\n"
    return spec


# ================================================================= lite

@template(M, "lite")
def find_basics(rng):
    spec = _repo(rng)
    times = [f"touch -d '2026-10-{rng.randint(1, 15):02d} 12:00' {q(p)}" for p in spec if rng.random() < 0.6]
    big = rng.choice(["data/blob.bin", "data/cache.npy", "models/weights.bin"])
    spec[big] = "x" * rng.randint(30, 60) * 1000 + "\n"
    setup = "#: cwd ~/repo\n#: date 2026-10-20 09:41\n" + files(spec) + "\n" + "\n".join(times)
    kb = rng.choice([10, 20, 25])
    items = [
        ("find -name", "Find every Python file (`*.py`) under the current directory.", 'find . -name "*.py"'),
        ("find -iname", "Find every file whose name contains `readme`, ignoring case.", 'find . -iname "*readme*"'),
        ("find -type", "Find every directory named `__pycache__`.", 'find . -type d -name "__pycache__"'),
        ("find -size", f"Find regular files bigger than {kb} KiB (`-size +{kb}k`).", f"find . -type f -size +{kb}k"),
        ("find -mtime", "Find files modified in the last 24 hours.", "find . -type f -mtime -1"),
        ("find -maxdepth", "Find `.py` files at most two levels deep (`./x.py` and `./dir/x.py`, not deeper).",
         'find . -maxdepth 2 -name "*.py"'),
        ("find -not", "Find every `.py` file except those inside `./.venv`.", 'find . -name "*.py" -not -path "./.venv/*"',
         {"require": ["find -not|find -prune|find !"]}),
        ("find -delete", "Delete every compiled `*.pyc` file.", 'find . -name "*.pyc" -delete',
         {"post": "find . -name '*.pyc' | wc -l; find . -type f | sort"}),
        ("find -exec", "Delete every `__pycache__` directory and its contents (`-exec rm -rf {} +`).",
         'find . -name "__pycache__" -type d -exec rm -rf {} +', {"post": "find . | sort"}),
        ("find -exec", "Count the lines of every `.py` file outside `.venv`, with one `wc -l` run (`-exec ... +`).",
         'find . -name "*.py" -not -path "./.venv/*" -exec wc -l {} +', {"unordered": False}),
    ]
    return catalog(rng, items, setup, unordered=True)


@template(M, "lite")
def grep_basics(rng):
    log = train_log(rng, rng.randint(3, 6))
    log = log.replace("\n", "\n", 1)
    extra = [f"2026-10-20 10:0{i} ERROR {rng.choice(['CUDA OOM', 'NaN loss', 'disk full'])} in step {rng.randint(1, 900)}"
             for i in range(rng.randint(1, 3))]
    log += "\n".join(extra) + "\n"
    spec = {"train.log": log, "config.py": f"lr = {rng.choice(['1e-3', '3e-4', '0.01'])}\nlr_decay = 0.9\nwarmup_lr = 1e-5\n"
                                           f"batch = {rng.choice([16, 32, 64])}\n",
            "a.py": "import torch\n# TODO: speed up\n", "b.py": "x = 1\n", "c.py": "# TODO: tests\n",
            "notes.txt": "a.b[0] is weird\nab0 is not\n"}
    setup = files(spec)
    items = [
        ("grep", "Print the lines of `train.log` that contain `ERROR`.", 'grep "ERROR" train.log'),
        ("grep -i", "Print the lines of `train.log` that contain `error` in any case.", 'grep -i "error" train.log'),
        ("grep -n", "Print the lines of `train.log` containing `val_acc`, with line numbers.", 'grep -n "val_acc" train.log'),
        ("grep -v", "Print `train.log` without its DEBUG lines.", 'grep -v "DEBUG" train.log'),
        ("grep -c", "How many lines of `train.log` mention `epoch`? Print the count.", 'grep -c "epoch" train.log'),
        ("grep -l", "Which `.py` files contain `TODO`? Print their names.", 'grep -l "TODO" *.py'),
        ("grep -w", "Print the lines of `config.py` with the whole word `lr` (not `lr_decay` or `warmup_lr`).",
         'grep -w "lr" config.py'),
        ("grep -o", "Print every `loss=...` value in `train.log` (only the matching part, like `loss=0.4123`).",
         'grep -o "loss=[0-9.]*" train.log'),
        ("grep -E", "Print the lines of `train.log` that contain `ERROR` or `WARNING` (one extended regex).",
         'grep -E "ERROR|WARNING" train.log'),
        ("grep -F", "Print the lines of `notes.txt` that contain the literal text `a.b[0]` (no regex).",
         'grep -F "a.b[0]" notes.txt'),
        ("grep -P", "Print just the numbers of every `loss=` value in `train.log`, using a Perl regex with `\\K`.",
         'grep -oP "loss=\\K[0-9.]+" train.log', {"require": ["grep -P", "grep -o"]}),
        ("grep -A", "Print each `ERROR` line of `train.log` with the line after it.", 'grep -A 1 "ERROR" train.log'),
        ("grep -B", "Print each `val_acc` line of `train.log` with the 2 lines before it.", 'grep -B 2 "val_acc" train.log'),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def recursive_search(rng):
    spec = _repo(rng)
    setup = "#: cwd ~/repo\n" + files(spec)
    lib = rng.choice(["torch", "pandas", "numpy", "wandb", "json"])
    items = [
        ("grep -r", f"Search all `.py` files under the current directory for `import {lib}` (`--include`).",
         f'grep -r "import {lib}" --include="*.py" .', {"require": ["grep -r", "grep --include"], "unordered": True}),
        ("grep --exclude-dir", "Find every `TODO` under the current directory with line numbers, skipping `.venv`.",
         'grep -rn "TODO" . --exclude-dir=.venv', {"require": ["grep -r", "grep -n", "grep --exclude-dir"], "unordered": True}),
        ("grep -l", f"Which Python files (outside `.venv`) import `{lib}`? Print just their paths.",
         f'grep -rl "import {lib}" --include="*.py" --exclude-dir=.venv .', {"require": ["grep -l", "grep -r"],
                                                                            "unordered": True}),
        ("rg", "Search for `TODO` with ripgrep (it skips hidden directories like `.venv` by itself).", 'rg "TODO"',
         {"unordered": True}),
        ("rg -t", f"Search only Python files for `{lib}` with ripgrep.", f'rg -t py "{lib}"', {"unordered": True}),
        ("rg -l", "List the files that contain `return` with ripgrep.", 'rg -l "return"', {"unordered": True}),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def counting(rng):
    spec = _repo(rng, rng.randint(5, 8))
    setup = "#: cwd ~/repo\n" + files(spec)
    kind = rng.choice(["py-count", "todo-files", "dirs", "todo-total"])
    if kind == "py-count":
        return P("How many `.py` files are there outside `.venv`? Print just the number.", setup,
                 'find . -name "*.py" -not -path "./.venv/*" | wc -l', require=["find", "wc"])
    if kind == "todo-files":
        return P("How many files (outside `.venv`) contain `TODO`? Print just the number.", setup,
                 'grep -rl "TODO" --exclude-dir=.venv . | wc -l', require=["grep", "wc"])
    if kind == "dirs":
        return P("How many directories are there under the current one (not counting `.` itself)?", setup,
                 "find . -mindepth 1 -type d | wc -l", require=["find -type", "wc"])
    return P("How many `TODO` lines are there in total in the `.py` files outside `.venv`?", setup,
             'grep -r "TODO" --include="*.py" --exclude-dir=.venv . | wc -l', require=["grep", "wc"])


# ================================================================= medium

@template(M, "medium")
def find_xargs(rng):
    n = rng.randint(3, 6)
    spec = {}
    for i in range(n):
        d = rng.choice(["camera", "phone/DCIM", "downloads", "shared photos"])
        nm = rng.choice(["IMG", "photo", "my pic", "holiday"]) + f"_{i}.jpg"
        spec[f"{d}/{nm}"] = "jpg\n"
    spec["downloads/doc.pdf"] = "pdf\n"
    setup = "#: cwd ~/sync\n" + files(spec)
    kind = rng.choice(["copy", "count-lines", "remove"])
    if kind == "copy":
        return P("Copy every `.jpg` file under the current directory into `images/` (create it). Some names "
                 "contain spaces, so use `-print0` with `xargs -0`, and `cp -t`.", setup,
                 'mkdir -p images\nfind . -name "*.jpg" -not -path "./images/*" -print0 | xargs -0 -n 100 cp -t images/',
                 post="ls -1 images", require=["find -print0", "xargs -0", "cp -t"])
    if kind == "count-lines":
        return P("Print how many `.jpg` files there are in each top-level directory as `COUNT DIR`, sorted by "
                 "directory name (careful: some names have spaces).", setup,
                 "find . -name '*.jpg' -printf '%P\\n' | cut -d/ -f1 | sort | uniq -c | awk '{c=$1; $1=\"\"; print c $0}'",
                 require=["find"])
    return P("Delete every `.jpg` under `downloads/` and `camera/` (names may contain spaces), safely, with "
             "`find ... -print0 | xargs -0 rm`.", setup,
             'find downloads camera -name "*.jpg" -print0 | xargs -0 rm -f', post="find . -type f | sort",
             require=["find -print0", "xargs -0"])


@template(M, "medium")
def mine_log(rng):
    log = train_log(rng, rng.randint(6, 10))
    setup = files({"train.log": log})
    k = rng.randint(3, 5)
    kind = rng.choice(["last-acc", "best-acc", "loss-count", "epochs"])
    if kind == "last-acc":
        return P(f"Print the last {k} `val_acc` values from `train.log`, just the numbers.", setup,
                 f'grep -oP "val_acc=\\K[0-9.]+" train.log | tail -{k}', require=["grep", "tail"])
    if kind == "best-acc":
        return P("Print the best (largest) `val_acc` value in `train.log`, just the number.", setup,
                 'grep -oP "val_acc=\\K[0-9.]+" train.log | sort -n | tail -1', require=["grep", "sort"])
    if kind == "loss-count":
        return P("How many `loss=` readings are below `0.5`? Print just the count.", setup,
                 "grep -oP 'loss=\\K[0-9.]+' train.log | awk '$1 < 0.5' | wc -l", require=["grep"])
    return P("Which epochs logged a `val_acc`? Print the epoch numbers, one per line.", setup,
             "grep val_acc train.log | grep -oP 'epoch=\\K[0-9]+'", require=["grep"])


# ================================================================= hard

@template(M, "hard")
def tracebacks(rng):
    lines = []
    n_tb = rng.randint(1, 3)
    for i in range(rng.randint(8, 14)):
        lines.append(f"INFO step {i * 10} ok")
        if n_tb and rng.random() < 0.3:
            n_tb -= 1
            lines += ["Traceback (most recent call last):", f'  File "train.py", line {rng.randint(10, 99)}, in step',
                      f"    loss = model({rng.choice(WORDS)})", f"RuntimeError: {rng.choice(['CUDA out of memory', 'shape mismatch', 'NaN detected'])}"]
    if n_tb:
        lines += ["Traceback (most recent call last):", '  File "train.py", line 7, in <module>', "    main()",
                  "KeyError: 'lr'"]
    setup = files({"train.log": "\n".join(lines) + "\n"})
    kind = rng.choice(["context", "errors"])
    if kind == "context":
        return P("Print every `Traceback` in `train.log` with 2 lines of context before and 3 after (`grep -A/-B`).",
                 setup, 'grep -A 3 -B 2 "Traceback" train.log', require=["grep -A", "grep -B"])
    return P("Print just the final error line of each traceback (the line 3 lines after `Traceback`), e.g. "
             "`KeyError: 'lr'`.", setup, "grep -A 3 Traceback train.log | grep -E '^[A-Za-z]+Error'",
             require=["grep"])


@template(M, "hard")
def recent_big(rng):
    spec, times = {}, []
    for i in range(rng.randint(6, 10)):
        p = f"{rng.choice(['ckpt', 'logs', 'data', 'ckpt/old'])}/{rng.choice(WORDS)}_{i}.{rng.choice(['pt', 'log', 'bin'])}"
        spec[p] = "x" * rng.choice([2, 8, 15, 40, 90]) * 1000 + "\n"
        times.append(f"touch -d '2026-10-{rng.choice([12, 17, 19, 20]):02d} 08:00' {p}")
    setup = "#: date 2026-10-20 09:41\n#: cwd ~/lab\n" + files(spec) + "\n" + "\n".join(times)
    kb = rng.choice([10, 20, 30])
    days = rng.choice([2, 4])
    return P(f"Print the paths of regular files bigger than {kb} KiB modified in the last {days} days, sorted.",
             setup, f"find . -type f -size +{kb}k -mtime -{days} | sort", require=["find -size", "find -mtime"])


# ================================================================= challenge

def _todos(rng, n):
    spec = _repo(rng, min(n, 9), todo=True)
    for i in range(n - 9 if n > 9 else 0):
        spec[f"pkg/mod_{i}.py"] = "".join(
            ("x = 1  # TODO: " + rng.choice(WORDS) if rng.random() < 0.3 else f"y{j} = {j}") + "\n" for j in range(rng.randint(2, 8)))
    spec["build/gen.py"] = "# TODO generated\n"
    return "#: cwd ~/repo\n" + files(spec)


@template(M, "challenge")
def todo_report(rng):
    setup = _todos(rng, 6)
    hidden = [_todos(rng, rng.randint(25, 35)) for _ in range(2)]
    sol = ("grep -rn 'TODO' --include='*.py' --exclude-dir=.venv --exclude-dir=build . | sed 's|^\\./||' | "
           "sort -t: -k1,1 -k2,2n | awk -F: '{print $1 \":\" $2}' | awk -F: '{c[$1]++; l[$1] = l[$1] (l[$1] ? \",\" : \"\") $2} "
           "END {for (f in c) print f, c[f], l[f]}' | sort")
    return challenge("TODO inventory", "For every `.py` file (outside `.venv/` and `build/`) that contains `TODO`, "
                     "print `PATH COUNT LINES`: the path without a leading `./`, how many lines mention TODO, and their "
                     "line numbers comma-separated in increasing order. Sort the output by path.", setup, sol, hidden,
                     "up to 35 files")


def _name_clash(rng, n):
    spec = {}
    pool = [f"{rng.choice(WORDS)}.{rng.choice(['txt', 'csv', 'py'])}" for _ in range(max(3, n // 3))]
    for i in range(n):
        spec[f"{rng.choice(['a', 'b', 'c', 'a/x', 'b/y'])}/{rng.choice(pool)}"] = f"{i}\n"
    return "#: cwd ~/merge\n" + files(spec)


@template(M, "challenge")
def name_clashes(rng):
    setup = _name_clash(rng, 7)
    hidden = [_name_clash(rng, rng.randint(30, 45)) for _ in range(2)]
    sol = ("find . -type f -printf '%f %P\\n' | sort | awk '{n[$1]++; p[$1] = p[$1] \" \" $2} "
           "END {for (k in n) if (n[k] > 1) print k \":\" p[k]}' | sort")
    return challenge("Before flattening", "You're about to copy every file under the current directory into one flat "
                     "folder. Find the file names that occur more than once and print `NAME: PATH PATH ...` (paths "
                     "relative to here, without `./`, sorted), one line per clashing name, sorted by name.",
                     setup, sol, hidden, "up to 45 files", require=["find"])

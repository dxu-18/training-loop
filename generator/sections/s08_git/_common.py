"""Helpers shared by the Git & GitHub templates (section 8). Problems run in bash-lite's live terminal with its git
and GitHub-lite (generator/bashlite/git): kind "bash", graded by bashlite/check.py.

Most problems are graded on the state the session leaves: `check=(sections, dir)` makes the post-check run
/usr/lib/bashlite/gitstate (git/gitstate.py), which prints branches, history, the index, the working tree, the
stash, remotes and the GitHub-lite side with date-independent commit ids, so a person's commits match the
reference solution's even though they were made at other times. Problems about reading history compare what the
answer prints instead (only for commits made by the setup, whose ids are fixed).
"""

from __future__ import annotations

import random
import shlex

from ...core import Problem, pick, template as _template

S = "git"
GIT = "#: git"
GH = "#: github key login"
STATE = "/usr/lib/bashlite/gitstate"

PEOPLE = [("Ada Lovelace", "ada@example.com"), ("Grace Hopper", "grace@example.com"),
          ("Alan Turing", "alan@example.com"), ("Katherine Johnson", "kj@example.com"),
          ("Linus Park", "linus@example.com"), ("Barbara Liskov", "barbara@example.com")]
WORDS = ("attention", "tokenizer", "dataloader", "scheduler", "optimizer", "checkpoint", "metrics", "embedding",
         "augment", "sampler", "trainer", "logger", "config", "encoder", "decoder", "pipeline", "cache", "loss")
FILES = ("train.py", "model.py", "data.py", "utils.py", "eval.py", "config.yaml", "README.md", "notes.md",
         "requirements.txt", "metrics.py")
BRANCHES = ("feature/kv-cache", "feature/dropout", "fix/nan-loss", "feature/mixed-precision", "docs/readme",
            "feature/early-stopping", "fix/off-by-one", "feature/wandb", "refactor/data", "feature/lr-warmup")
SUBJECTS = ("Add {w} module", "Fix {w} edge case", "Refactor {w}", "Speed up {w}", "Document {w}",
            "Add tests for {w}", "Clean up {w}", "Tune {w} defaults", "Log {w} stats", "Rename {w} helpers")


def template(module: str, tier: str):
    return _template(S, module, tier)


def q(s) -> str:
    return shlex.quote(str(s))


def P(prompt: str, setup: str, solution: str, check=None, **kw) -> Problem:
    """A git problem. check=(sections, dir): grade the state gitstate prints for `dir` (default: the start
    directory) after the session; otherwise what the answer prints is compared (kw answer=...)."""
    kw.setdefault("timeout", 20.0)
    if check is not None:
        sections, where = check if isinstance(check, tuple) else (check, ".")
        kw["post"] = f"{STATE} -C {where} {' '.join(sections)}"
        kw.setdefault("answer", ["<post>"])
    return Problem(prompt=prompt.strip(), setup=setup.strip("\n"), solution=solution.strip("\n"), kind="bash",
                   lang="bash", **kw)


def catalog(rng: random.Random, items: list, setup: str = "", intro: str = "", **kw) -> Problem:
    """One task from a list. Item: (feature, prompt, solution[, {options}]); options may add `setup` lines, a
    `check`, and any Problem field (`require` defaults to [feature] when feature is given)."""
    item = pick(rng, items)
    feat, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", [feat] if feat else [])
    extra = opts.pop("setup", "")
    full_setup = "\n".join(x for x in (setup, extra) if x)
    merged = {**kw, **opts}
    return P(f"{intro} {prompt}".strip(), full_setup, sol, require=req, **merged)


def heredoc(path: str, text: str) -> str:
    if text and not text.endswith("\n"):
        text += "\n"
    delim = "EOF"
    while delim in text.split("\n"):
        delim += "_"
    if "/" in path:
        return f"mkdir -p {q(path.rsplit('/', 1)[0])} && cat > {q(path)} <<'{delim}'\n{text}{delim}"
    return f"cat > {q(path)} <<'{delim}'\n{text}{delim}"


class Repo:
    """Bash that builds a repository in the current directory, commit by commit (the setup runs it quietly)."""

    def __init__(self, rng: random.Random | None = None, branch: str = "main", init: bool = True):
        self.rng = rng or random.Random(0)
        self.lines = [f"git init -q -b {branch}"] if init else []
        self.files: dict[str, str] = {}
        self.cur = branch
        self.saved: dict[str, dict] = {branch: {}}

    def write(self, path, text):
        self.files[path] = text if text.endswith("\n") else text + "\n"
        self.lines.append(heredoc(path, self.files[path]))
        return self

    def append(self, path, text):
        return self.write(path, self.files.get(path, "") + (text if text.endswith("\n") else text + "\n"))

    def rm(self, path):
        self.files.pop(path, None)
        self.lines.append(f"git rm -q {q(path)}")
        return self

    def commit(self, msg, files=None, author=None):
        for p, t in (files or {}).items():
            self.write(p, t)
        env = ""
        if author:
            env = f"GIT_AUTHOR_NAME={q(author[0])} GIT_AUTHOR_EMAIL={q(author[1])} "
        self.lines.append(f"git add -A && {env}git commit -qm {q(msg)}")
        self.saved[self.cur] = dict(self.files)
        return self

    def branch(self, name):
        self.lines.append(f"git switch -qc {q(name)}")
        self.saved[name] = dict(self.files)
        self.cur = name
        return self

    def switch(self, name):
        self.lines.append(f"git switch -q {q(name)}")
        self.files = dict(self.saved.get(name, self.files))
        self.cur = name
        return self

    def merge(self, name, msg=None):
        self.lines.append(f"git merge -q --no-ff {q(name)} -m {q(msg or f'Merge branch {name}')}")
        return self

    def tag(self, name, msg=None):
        self.lines.append(f"git tag -a {q(name)} -m {q(msg)}" if msg else f"git tag {q(name)}")
        return self

    def run(self, line):
        self.lines.append(line)
        return self

    def code(self) -> str:
        return "\n".join(self.lines)


def ml_files(rng: random.Random, n: int = 3) -> dict:
    """A few small ML-project files with varied content."""
    w = rng.sample(WORDS, 4)
    base = {
        "train.py": f"import torch\n\nfrom model import Net\n\nEPOCHS = {rng.choice([5, 10, 20])}\nLR = "
                    f"{rng.choice(['1e-3', '3e-4', '5e-4'])}\n\n\ndef train(loader):\n    model = Net()\n"
                    f"    for epoch in range(EPOCHS):\n        for batch in loader:\n            pass\n",
        "model.py": f"import torch.nn as nn\n\n\nclass Net(nn.Module):\n    def __init__(self, hidden={rng.choice([64, 128, 256])}):\n"
                    f"        super().__init__()\n        self.fc = nn.Linear(hidden, 10)\n\n    def forward(self, x):\n"
                    f"        return self.fc(x)\n",
        "data.py": f"def load_{w[0]}(path):\n    with open(path) as f:\n        return [line.split(',') for line in f]\n",
        "README.md": f"# {w[1]}-{w[2]}\n\nTraining code for the {w[1]} experiments.\n",
        "config.yaml": f"lr: {rng.choice(['0.001', '0.0003'])}\nbatch_size: {rng.choice([16, 32, 64])}\n"
                       f"epochs: {rng.choice([5, 10, 20])}\n",
        "utils.py": f"def {w[3]}_stats(xs):\n    return sum(xs) / len(xs)\n",
    }
    keys = rng.sample(sorted(base), n)
    return {k: base[k] for k in keys}


def subject(rng: random.Random) -> str:
    return rng.choice(SUBJECTS).format(w=rng.choice(WORDS))


def history(rng: random.Random, n: int, files=None, authors=None) -> Repo:
    """A repository with n commits, each changing one file (a line appended)."""
    r = Repo(rng)
    fs = files or ml_files(rng, 3)
    r.commit("Initial commit", fs)
    names = sorted(fs)
    for i in range(n - 1):
        p = rng.choice(names)
        r.append(p, f"# {rng.choice(WORDS)} {i}")
        r.commit(subject(rng), author=rng.choice(authors) if authors else None)
    return r


def setup(*parts: str) -> str:
    return "\n".join(p for p in parts if p)

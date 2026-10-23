"""Helpers shared by the Bash templates (problem kind "bash", run on bash-lite: generator/bashlite).

A problem's setup is bash run quietly before the answer, in the same interactive session, plus `#:` directive lines
that describe the machine (see bashlite/directives.py). The answer is typed at the prompt line by line; by default its
stdout is compared with the reference solution's (see bashlite/check.py).
"""

from __future__ import annotations

import random
import shlex

from ...core import Problem, pick, template as _template

S = "bash"

WORDS = ("apple", "river", "stone", "cloud", "tiger", "maple", "ocean", "piano", "lemon", "orbit", "pixel", "quartz",
         "raven", "solar", "tulip", "violet", "willow", "zebra", "amber", "basil", "cedar", "delta", "ember", "falcon",
         "garnet", "harbor", "indigo", "jasper", "kernel", "lunar", "meadow", "nectar", "onyx", "pepper", "quill",
         "rocket", "saffron", "thistle", "umber", "vector")
NAMES = ("ada", "alan", "grace", "linus", "barbara", "dennis", "ken", "margaret", "edsger", "donald", "frances",
         "john", "katherine", "tim", "radia", "guido", "bjarne", "yann", "geoffrey", "fei", "andrej", "ilya")
MODELS = ("resnet18", "resnet50", "vit-b16", "bert-base", "gpt2", "t5-small", "llama-7b", "mistral-7b", "clip-vit",
          "whisper-tiny", "unet", "yolov8n", "efficientnet-b0", "distilbert", "mobilenet-v3")
DATASETS = ("cifar10", "imagenet", "mnist", "squad", "glue", "coco", "wikitext", "librispeech", "celeba", "voc")
TEAMS = ("vision", "nlp", "speech", "infra", "robotics", "rl")
EXTS = ("py", "txt", "csv", "md", "json", "log", "yaml", "sh")


def template(module: str, tier: str):
    return _template(S, module, tier)


def P(prompt: str, setup: str, solution: str, **kw) -> Problem:
    kw.setdefault("timeout", 10.0)
    return Problem(prompt=prompt.strip(), setup=setup.strip("\n"), solution=solution.strip("\n"), kind="bash",
                   lang="bash", **kw)


def K(prompt: str, setup: str, solution: str, **kw) -> Problem:
    """A keystrokes problem (7.17): the answer is keys like `C-a "sudo " Enter`."""
    kw.setdefault("timeout", 10.0)
    head = "#: keys" + (" tmux" if kw.pop("tmux", False) else "")
    return Problem(prompt=prompt.strip(), setup=(head + "\n" + setup.strip("\n")).strip("\n"),
                   solution=solution.strip(), kind="bash", lang="text", **kw)


def q(s) -> str:
    return shlex.quote(str(s))


def heredoc(path: str, text: str) -> str:
    """`cat > path <<'EOF'` writing `text` exactly (a final newline is added if missing)."""
    if text and not text.endswith("\n"):
        text += "\n"
    delim = "EOF"
    while delim in text.split("\n"):
        delim += "_"
    if not text:
        return f": > {q(path)}"
    return f"cat > {q(path)} <<'{delim}'\n{text}{delim}"


def files(spec: dict, mkdirs=()) -> str:
    """Setup lines creating directories and files (content str; None = empty file)."""
    dirs = sorted({p.rsplit("/", 1)[0] for p in spec if "/" in p} | set(mkdirs))
    lines = []
    if dirs:
        lines.append("mkdir -p " + " ".join(q(d) for d in dirs))
    for path, text in spec.items():
        lines.append(f"touch {q(path)}" if text is None else heredoc(path, text))
    return "\n".join(lines)


def run(setup: str, solution: str, seed: int = 0) -> str:
    """The reference solution's stdout (to quote sample answers in prompts)."""
    from ...bashlite.check import run_case
    return run_case({"seed": seed}, solution, setup, echo=False)["stdout"]


def sample(rng: random.Random, pool, k: int) -> list:
    return rng.sample(list(pool), k)


def catalog(rng: random.Random, items: list, setup: str = "", intro: str = "", **kw) -> Problem:
    """One task from a list. Item: (feature, prompt, solution[, {options}]); options may add `setup` lines and any
    Problem field (`require` defaults to [feature] when feature is given)."""
    item = pick(rng, items)
    feat, prompt, sol = item[:3]
    opts = dict(item[3]) if len(item) > 3 else {}
    req = opts.pop("require", [feat] if feat else [])
    extra = opts.pop("setup", "")
    full_setup = "\n".join(x for x in (setup, extra) if x)
    return P(f"{intro} {prompt}".strip(), full_setup, sol, require=req, **{**kw, **opts})


def challenge(title: str, story: str, setup: str, solution: str, tests: list[str], constraints: str,
              seed: int = 0, **kw) -> Problem:
    sample_out = run(setup, solution, seed).rstrip("\n")
    shown = "\n".join(sample_out.split("\n")[:12])
    more = " (first lines)" if sample_out.count("\n") >= 12 else ""
    return P(f"**{title}{'' if title[-1] in '.?!' else '.'}** {story}\n\n**Constraints:** {constraints}. Your commands also run on hidden tests with "
             f"different data.\n\nFor the files shown, the expected output{more} is:\n\n```\n{shown}\n```",
             setup, solution, tests=tests, timeout=20.0, **kw)


def csv_text(header, rows, sep=",") -> str:
    return sep.join(header) + "\n" + "".join(sep.join(str(c) for c in r) + "\n" for r in rows)


def train_log(rng: random.Random, epochs: int, steps: int = 100, start_loss: float | None = None,
              noise: bool = True, val: bool = True) -> str:
    """A training log: INFO lines with epoch/step/loss, val_acc per epoch, a few WARNING/DEBUG lines."""
    loss = start_loss or round(rng.uniform(1.8, 2.6), 4)
    acc = round(rng.uniform(0.35, 0.55), 4)
    lines = []
    t = rng.randint(0, 50)
    for e in range(1, epochs + 1):
        for s in range(steps, steps * 3 + 1, steps):
            loss = round(max(0.01, loss * rng.uniform(0.82, 0.97)), 4)
            t += rng.randint(20, 90)
            lines.append(f"2026-10-20 09:{(t // 60) % 60:02d}:{t % 60:02d} INFO epoch={e} step={(e - 1) * steps * 3 + s} "
                         f"loss={loss:.4f}")
            if noise and rng.random() < 0.25:
                lines.append(f"2026-10-20 09:{(t // 60) % 60:02d}:{t % 60:02d} DEBUG grad_norm={rng.uniform(0.1, 3):.3f}")
        if val:
            acc = round(min(0.99, acc + rng.uniform(0.01, 0.09)), 4)
            lines.append(f"2026-10-20 09:{(t // 60) % 60:02d}:{t % 60:02d} INFO epoch={e} val_acc={acc:.4f}")
        if noise and rng.random() < 0.2:
            lines.append(f"2026-10-20 09:{(t // 60) % 60:02d}:{t % 60:02d} WARNING lr scheduler stepped late")
    return "\n".join(lines) + "\n"

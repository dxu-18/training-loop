"""Helpers shared by the Vim templates (COMP_GUIDE §9).

A Vim problem runs on bash-lite's machine (generator/bashlite) with the Vim engine (generator/bashlite/vim): the
setup writes the files, then `#: vim file` opens Vim on them before the session starts (quitting is turned off,
except in the problems that practise it). The user's keys are graded by bashlite/vimcheck.py: Vim's state at the end
is compared with the reference keys' (the text by default), `require` / `forbid` name the commands to use or avoid
(as Vim.used records them), and `par` caps the keys pressed, VimGolf style.

`V()` builds a problem: it types the reference keys into a fresh terminal once, to check they work, to show the
target text in the prompt, to highlight a cursor target, and to list the commands for the Instructions.
"""

from __future__ import annotations

import difflib
import random
import re

from ...core import Problem, pick, template as _template

S = "vim"

MODELS = ("resnet18", "resnet50", "vit_b16", "bert_base", "gpt2", "t5_small", "llama_7b", "mistral_7b", "clip_vit",
          "whisper_tiny", "unet", "yolov8n", "mobilenet_v3", "distilbert")
DATASETS = ("cifar10", "imagenet", "mnist", "squad", "glue", "coco", "wikitext", "librispeech", "celeba", "voc")
OPTIMS = ("SGD", "Adam", "AdamW", "RMSprop", "Adagrad")
ACTS = ("relu", "gelu", "silu", "tanh", "sigmoid")
WORDS = ("apple", "river", "stone", "cloud", "tiger", "maple", "ocean", "piano", "lemon", "orbit", "pixel", "quartz",
         "raven", "solar", "tulip", "violet", "willow", "zebra", "amber", "basil", "cedar", "delta", "ember", "falcon")
NAMES = ("ada", "alan", "grace", "linus", "barbara", "dennis", "ken", "margaret", "edsger", "donald", "frances",
         "john", "katherine", "tim", "radia", "guido", "bjarne", "yann", "geoffrey", "fei", "andrej", "ilya")
VARS = ("loss", "logits", "preds", "batch", "inputs", "targets", "hidden", "scores", "weights", "grads", "labels",
        "outputs", "features", "tokens", "mask", "probs")
LRS = ("0.1", "0.01", "0.001", "3e-4", "1e-4", "5e-5")


def template(module: str, tier: str):
    return _template(S, module, tier)


class VimSetupError(AssertionError):
    pass


def heredoc(path: str, text: str) -> str:
    if text and not text.endswith("\n"):
        text += "\n"
    delim = "EOF"
    while delim in text.split("\n"):
        delim += "_"
    if not text:
        return f": > '{path}'"
    return f"cat > '{path}' <<'{delim}'\n{text}{delim}"


def lines_of(text: str) -> list[str]:
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


def _q(s: str) -> str:
    return "`` " + s + " ``" if "`" in s else "`" + s + "`"


def show_lines(lines: list[str], first: int = 1, numbers: bool = True) -> str:
    """A code block of lines (numbered like :set number)."""
    w = len(str(first + len(lines) - 1))
    body = "\n".join((f"{first + i:>{w}}  " if numbers else "") + ln.expandtabs(8) for i, ln in enumerate(lines))
    return "```\n" + body + "\n```"


def diff_view(before: list[str], after: list[str], context: int = 1) -> str:
    """The changed part of the text, as it should look afterwards (numbered), with a note for removed lines."""
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return "(no change)"
    lo = max(0, min(op[3] for op in ops) - context)
    hi = min(len(after), max(op[4] for op in ops) + context)
    if hi <= lo:   # everything was deleted
        return "_(the file ends up empty)_"
    block = show_lines(after[lo:hi], lo + 1)
    removed = sum(op[2] - op[1] for op in ops if op[0] in ("delete", "replace")) - \
        sum(op[4] - op[3] for op in ops if op[0] == "replace")
    note = ""
    if removed > 0 and all(op[0] == "delete" for op in ops):
        note = f" ({removed} line{'s' if removed != 1 else ''} removed)"
    where = f"line {lo + 1}" if hi - lo == 1 else f"lines {lo + 1}–{hi}"
    return f"Afterwards, {where}{note}:\n\n{block}"


def V(prompt: str, files: dict, keys: str, *, start: str | None = None, shell: bool = False, quit_ok: bool = False,
      check=("text",), require=(), forbid=(), par: int | None = None, cursor: tuple | None = None,
      prep=(), show: str | None = "diff", target: bool | None = None, regs: str = "", marks: str = "",
      options=(), check_files=None, extra_setup: str = "", hint: str | None = None, timeout: float = 20.0,
      expect: list | None = None, errors_ok: bool = False) -> Problem:
    """A Vim problem.

    files: {path (relative to ~): text}; the first is the one Vim opens unless `start` gives Vim's arguments.
    keys: the reference solution in Vim notation (`ciwfit<Esc>`); with shell=True it starts at the bash prompt
      (`vim notes.md<CR>...`), else Vim is already open (and can't be quit unless quit_ok).
    check: what is compared at the end: text files quit cursor line topline layout sizes buffers regs marks folds
      options maps mode (see bashlite/vimcheck.py); regs / marks / options name which ones.
    cursor: where the cursor starts (line, column), 1-based. prep: [(seconds before the start, keys)] typed earlier.
    show: "diff" appends the changed lines as they should end up, "target" the whole file, None nothing.
    target: highlight where the cursor should end (default: when the cursor is checked).
    expect: the text the reference must produce (a guard for the template's own logic).
    errors_ok: the reference runs into a Vim error on purpose (else an error means the template is broken).
    """
    names = list(files)
    dirs = sorted({n.rsplit("/", 1)[0] for n in names if "/" in n})
    setup = ([f"mkdir -p {' '.join(dirs)}"] if dirs else []) + [heredoc(p, t) for p, t in files.items()]
    if extra_setup:
        setup.append(extra_setup)
    first = start if start is not None else (names[0] if names else "")
    if not shell:
        setup.append(f"#: vim {first}".rstrip())
        if quit_ok:
            setup.append("#: vimquit")
        pre = list(prep)
        if cursor is not None:
            pre.append((0, f"{cursor[0]}G{cursor[1]}|"))
        for sec, k in pre:
            setup.append(f"#: vimprep {sec} {k}")
    checks = list(check)
    spec = {"check": checks}
    if par:
        spec["max_keys"] = par
    if regs:
        spec["regs"] = regs
    if marks:
        spec["marks"] = marks
    if options:
        spec["options"] = list(options)
    if check_files:
        spec["files"] = list(check_files)
    p = Problem(prompt="", setup="\n".join(setup), solution=keys, kind="bash", lang="vim", require=list(require),
                forbid=list(forbid), timeout=timeout, vim=spec)
    ref = _reference(p, errors_ok)
    main = names[0] if names else None
    if expect is not None:
        got = next((b["lines"] for b in ref["buffers"] if b["name"] == main), None)
        if got != expect:
            raise VimSetupError(f"the reference keys {keys!r} give {got!r}, not {expect!r}")
    if target is None:
        target = "cursor" in checks and not shell
    if target:
        l, c = ref["cursor"]
        p.setup += f"\n#: vimtarget {l + 1} {c + 1}"
    text = prompt.strip()
    if show and main is not None and main in files:
        after = next((b["lines"] for b in ref["buffers"] if b["name"] == main), None)
        if after is None:
            after = lines_of(ref["files"].get(main, ""))
        if show == "diff" and after != lines_of(files[main]):
            text += "\n\n" + diff_view(lines_of(files[main]), after)
        elif show == "target":
            text += "\n\nAfterwards the file reads:\n\n" + show_lines(after)
    text += "\n\n" + (hint if hint is not None else footer(p, shell, quit_ok, first))
    p.prompt = text
    p.vim["uses"] = [k for k in ref["used"]]
    return p


def footer(p: Problem, shell: bool, quit_ok: bool, first: str) -> str:
    bits = []
    if shell:
        bits.append("Start at the bash prompt.")
    else:
        what = [x for x in first.split() if not x.startswith(("-", "+"))]
        shown = " and ".join(f"`{x}`" for x in what)
        bits.append(f"Vim is open on {shown} in the terminal." if what else "Vim is open in the terminal.")
    chk = p.vim["check"]
    if "cursor" in chk:
        bits.append("The highlighted cell is where the cursor should end.")
    if p.vim.get("max_keys"):
        par = p.vim["max_keys"]
        bits.append(f"Par: {par} key{'s' if par != 1 else ''} (the bar under the terminal counts them).")
    bits.append("Submit checks the result" + (" and the commands you used" if p.require or p.forbid else "") +
                "; Reset starts over.")
    return " ".join(bits)


def _reference(p: Problem, errors_ok: bool = False) -> dict:
    from ...bashlite.vimcheck import reference_state
    st = reference_state(p.to_dict())
    if st["errors"] and not errors_ok:
        raise VimSetupError(f"the reference keys {p.solution!r} hit a Vim error")
    return st


def drill(rng: random.Random, items: list, files: dict, **common) -> Problem:
    """One task from a list. Item: (prompt, keys, {V options}); the options override `common`."""
    prompt, keys, *rest = pick(rng, items)
    opts = {**common, **(rest[0] if rest else {})}
    fl = opts.pop("files", files)
    return V(prompt, fl, keys, **opts)


def count_keys(keys: str) -> int:
    from ...bashlite.vim import keys as VK
    return len(VK.parse(keys))


# ---------------------------------------------------------------- texts

def ident(rng: random.Random, pool=VARS, k: int = 1):
    return rng.sample(list(pool), k) if k > 1 else rng.choice(pool)


def train_py(rng: random.Random, epochs: int | None = None, lr: str | None = None, bs: int | None = None,
             model: str | None = None) -> tuple[str, dict]:
    """A small training script. -> (text, facts)."""
    epochs = epochs or rng.choice((5, 10, 20, 30))
    lr = lr or rng.choice(LRS)
    bs = bs or rng.choice((16, 32, 64, 128))
    model = model or rng.choice(MODELS)
    data = rng.choice(DATASETS)
    opt = rng.choice(OPTIMS)
    text = f"""import torch
from torch import nn
from data import load_{data}

EPOCHS = {epochs}
LR = {lr}
BATCH_SIZE = {bs}


def train(model, loader, epochs=EPOCHS):
    optimizer = torch.optim.{opt}(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {{epoch}}: loss {{total / len(loader):.4f}}")
    return model


if __name__ == "__main__":
    model = torch.hub.load("pytorch/vision", "{model}")
    train(model, load_{data}(batch_size=BATCH_SIZE))
"""
    return text, {"epochs": epochs, "lr": lr, "bs": bs, "model": model, "data": data, "opt": opt}


def model_py(rng: random.Random) -> tuple[str, dict]:
    name = rng.choice(("Classifier", "Encoder", "MLP", "TinyNet", "Probe", "Head"))
    dim = rng.choice((64, 128, 256, 512))
    hidden = rng.choice((128, 256, 512, 1024))
    n = rng.choice((2, 10, 100, 1000))
    act = rng.choice(ACTS)
    text = f"""import torch
from torch import nn


class {name}(nn.Module):
    def __init__(self, dim={dim}, hidden={hidden}, n_classes={n}):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.{act}(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
"""
    return text, {"name": name, "dim": dim, "hidden": hidden, "n": n, "act": act}


def config_yaml(rng: random.Random) -> tuple[str, dict]:
    f = {"model": rng.choice(MODELS), "data": rng.choice(DATASETS), "epochs": rng.choice((10, 20, 50)),
         "lr": rng.choice(LRS), "bs": rng.choice((32, 64, 128)), "seed": rng.randint(0, 99),
         "wd": rng.choice(("0.0", "0.01", "0.05"))}
    text = f"""# experiment config
model: {f['model']}
dataset: {f['data']}
training:
  epochs: {f['epochs']}
  batch_size: {f['bs']}
  lr: {f['lr']}
  weight_decay: {f['wd']}
seed: {f['seed']}
output_dir: runs/{f['model']}
"""
    return text, f


SENTENCES = [
    "The validation loss stopped improving after epoch {e}.",
    "We halved the learning rate and the curve smoothed out.",
    "Mixed precision cut the step time by a third.",
    "Gradient clipping at 1.0 fixed the loss spikes.",
    "The {m} baseline reached {a}% accuracy on {d}.",
    "Data augmentation helped more than a bigger model.",
    "Warmup for the first 500 steps kept training stable.",
    "Label smoothing gave a small but consistent gain.",
    "The tokenizer splits rare words into several pieces.",
    "Checkpoints are saved every {e} epochs to runs/.",
    "Early stopping triggered at epoch {e}.",
    "Batch norm statistics drifted when the batch was small.",
]


def notes_md(rng: random.Random, paras: int = 3, per: int = 3) -> tuple[str, list[list[str]]]:
    """Markdown notes: a title, then paragraphs of sentences (two per line). -> (text, paragraphs as sentences)."""
    pool = SENTENCES[:]
    rng.shuffle(pool)
    out, ps = [f"# Notes on {rng.choice(MODELS)}", ""], []
    k = 0
    for _ in range(paras):
        sents = []
        for _ in range(per):
            s = pool[k % len(pool)].format(e=rng.randint(3, 40), m=rng.choice(MODELS), a=rng.randint(60, 95),
                                           d=rng.choice(DATASETS))
            k += 1
            sents.append(s)
        ps.append(sents)
        out.append(" ".join(sents))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n", ps


def log_lines(rng: random.Random, n: int) -> list[str]:
    out = []
    loss = rng.uniform(1.8, 2.5)
    for i in range(n):
        loss *= rng.uniform(0.85, 0.98)
        lvl = rng.choice(("INFO", "INFO", "INFO", "WARN", "DEBUG"))
        out.append(f"{lvl} step={100 * (i + 1)} loss={loss:.4f}")
    return out


def long_py(rng: random.Random, n_funcs: int = 12) -> tuple[str, list[int]]:
    """A long module of small functions (for scrolling and jumping). -> (text, 1-based line of each def)."""
    out, defs = ['"""Metrics and helpers."""', "import math", ""], []
    names = rng.sample(["accuracy", "precision", "recall", "f1", "rmse", "mae", "perplexity", "top_k", "auc",
                        "log_loss", "brier", "entropy", "kl_div", "cosine", "jaccard", "dice", "iou", "bleu"],
                       n_funcs)
    for nm in names:
        out.append("")
        defs.append(len(out) + 1)
        a, b = rng.sample(VARS, 2)
        out += [f"def {nm}({a}, {b}):", f'    """{nm.replace("_", " ")} of {a} against {b}."""',
                f"    total = 0", f"    for x, y in zip({a}, {b}):", f"        total += abs(x - y)",
                f"    return total / max(1, len({a}))"]
    return "\n".join(out) + "\n", defs

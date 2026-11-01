"""Helpers shared by the LaTeX templates (COMP_GUIDE §11), graded by latex-lite (generator/latexlite).

`L()` builds a problem: the answer is the body (the preamble is given), a whole document, or the preamble (the body
is given). It compiles the reference once: it must compile without errors or warnings; the compiled page becomes
the target shown with the prompt (unless show_target=False), and the commands it uses feed the Instructions.
"""

from __future__ import annotations

import random

from ...core import Problem, pick, template as _template

S = "latex"
MODELS = ("ResNet-18", "ResNet-50", "ViT-B/16", "BERT-base", "GPT-2", "T5-small", "Llama-7B", "Mistral-7B", "CLIP",
          "Whisper-tiny", "U-Net", "YOLOv8n", "MobileNetV3", "DistilBERT")
DATASETS = ("CIFAR-10", "ImageNet", "MNIST", "SQuAD", "GLUE", "COCO", "WikiText", "LibriSpeech", "CelebA", "VOC")
AUTHORS = ("Ada Lovelace", "Alan Turing", "Grace Hopper", "Katherine Johnson", "Edsger Dijkstra", "Barbara Liskov",
           "Donald Knuth", "Frances Allen", "John McCarthy", "Radia Perlman", "Geoffrey Hinton", "Fei-Fei Li",
           "Yann LeCun", "Yoshua Bengio")
TITLES = ("Attention Is All I Practiced", "Gradient Descent by Hand", "Scaling Laws for Small Models",
          "A Tale of Two Optimizers", "Notes on Batch Normalization", "Why Warmup Works", "Learning Rate Folklore",
          "Dropout, Revisited", "Tokenizers Explained", "On the Width of Networks", "Mixed Precision in Practice")
TOPICS = ("optimization", "regularization", "attention", "tokenization", "evaluation", "data augmentation",
          "distillation", "pretraining", "fine-tuning", "calibration")

BASE = "\\documentclass{article}\n"
AMS = BASE + "\\usepackage{amsmath, amssymb}\n"


def template(module: str, tier: str):
    return _template(S, module, tier)


def fixed(fn):
    """Mark a template whose problem is one fixed text on purpose: writing a block of the guide from memory (the
    guide's SGD algorithm, its talk, the equation gallery). tests/test_templates.py's variety check skips these."""
    fn.fixed = True
    return fn


class LatexSetupError(AssertionError):
    pass


def L(prompt: str, solution: str, *, mode: str = "body", preamble: str = BASE, body: str = "", files=None,
      check=(), require=(), forbid=(), starter: str = "", show_target: bool = True, warnings_ok: bool = False,
      timeout: float = 15.0) -> Problem:
    """A LaTeX problem. mode: "body" (answer = the body; `preamble` given), "document" (a whole file),
    "preamble" (answer = the preamble; `body` given) or "bib" (answer = references.bib; `preamble` and `body` given).
    files: other files (references.bib, data.csv ...)."""
    spec = {"mode": mode, "files": dict(files or {}), "check": list(check)}
    if mode in ("body", "bib"):
        spec["preamble"] = preamble
    if mode in ("preamble", "bib"):
        spec["body"] = body
    if starter:
        spec["starter"] = starter
    p = Problem(prompt=prompt.strip(), setup="", solution=solution.strip("\n") + "\n", kind="latex", lang="latex",
                require=list(require), forbid=list(forbid), timeout=timeout, latex=spec)
    from ...latexlite.check import build
    from ...latexlite.render import render
    r = build(p.to_dict(), p.solution)
    if r["errors"]:
        e = r["errors"][0]
        raise LatexSetupError(f"the reference doesn't compile: l.{e['line']} {e['msg']} ({e['before'][-40:]})")
    if r["warnings"] and not warnings_ok:
        raise LatexSetupError(f"the reference compiles with a warning: {r['warnings'][0]['msg']}")
    used = {k for k, v in r["used"].items() if v}
    for req in p.require:
        if not any(a in used for a in req.split("|")):
            raise LatexSetupError(f"the reference doesn't use {req}")
    if show_target:
        spec["target"] = render(r)
        if r["errors"]:     # headers and footers run when the page renders
            raise LatexSetupError(f"the reference's page doesn't render: {r['errors'][0]['msg']}")
    spec["uses"] = [k for k, v in r["used"].items() if v]
    return p


def drill(rng: random.Random, items: list, **common) -> Problem:
    """One task from a list. Item: (prompt, solution, {L options})."""
    prompt, sol, *rest = pick(rng, items)
    opts = {**common, **(rest[0] if rest else {})}
    return L(prompt, sol, **opts)

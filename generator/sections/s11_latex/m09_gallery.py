"""11.09 ML equations gallery: softmax, cross-entropy, attention, SGD, Adam, KL, LayerNorm, LoRA, DPO, backprop."""

from ...core import pick
from ._common import AMS, L, fixed, template

M = "11.09"
PRE = AMS + "\\DeclareMathOperator{\\softmax}{softmax}\n\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\E}{\\mathbb{E}}\n"

EQS = {
    "softmax": ("the softmax", "\\softmax(z)_i &= \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}"),
    "ce": ("the cross-entropy loss", "\\mathcal{L}_{\\text{CE}} &= -\\sum_{k=1}^{K} y_k \\log \\hat{y}_k"),
    "attention": ("attention",
                  "\\operatorname{Attention}(Q, K, V) &= \\softmax\\!\\left( \\frac{Q K^\\top}{\\sqrt{d_k}} \\right) V"),
    "sgd": ("the SGD update", "\\theta_{t+1} &= \\theta_t - \\eta \\, \\nabla_\\theta \\mathcal{L}(\\theta_t)"),
    "adam_m": ("Adam's moments",
               "m_t &= \\beta_1 m_{t-1} + (1 - \\beta_1) g_t, \\qquad v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2"),
    "adam": ("Adam's update",
             "\\theta_{t+1} &= \\theta_t - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}, \\qquad "
             "\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\ \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}"),
    "kl": ("the KL divergence", "D_{\\mathrm{KL}}(P \\,\\|\\, Q) &= \\sum_x P(x) \\log \\frac{P(x)}{Q(x)}"),
    "layernorm": ("LayerNorm",
                  "\\operatorname{LayerNorm}(x) &= \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta"),
    "lora": ("LoRA", "h &= W_0 x + \\frac{\\alpha}{r} B A x, \\qquad B \\in \\R^{d \\times r},\\ A \\in \\R^{r \\times k}"),
    "dpo": ("the DPO loss",
            "\\mathcal{L}_{\\text{DPO}} &= -\\E_{(x, y_w, y_l)} \\left[ \\log \\sigma\\!\\left( \\beta \\log "
            "\\frac{\\pi_\\theta(y_w \\mid x)}{\\pi_{\\text{ref}}(y_w \\mid x)} - \\beta \\log \\frac{\\pi_\\theta(y_l "
            "\\mid x)}{\\pi_{\\text{ref}}(y_l \\mid x)} \\right) \\right]"),
    "backprop": ("backpropagation",
                 "\\delta^{(l)} &= \\bigl( W^{(l+1)\\top} \\delta^{(l+1)} \\bigr) \\odot f'\\bigl(z^{(l)}\\bigr), \\qquad "
                 "\\frac{\\partial \\mathcal{L}}{\\partial W^{(l)}} = \\delta^{(l)} \\bigl(a^{(l-1)}\\bigr)^\\top"),
}
EASY = ["softmax", "ce", "sgd", "kl"]
MEDIUM = ["attention", "layernorm", "lora", "adam_m"]
HARD = ["adam", "backprop", "dpo", "attention"]


def _one(key):
    name, line = EQS[key]
    return name, "\\begin{equation}\n  " + line.replace(" &=", " =").replace("&=", "=") + "\n\\end{equation}"


def _single(rng, keys, tier_note=""):
    name, sol = _one(pick(rng, keys))
    return L(f"Typeset {name} from memory as a numbered equation (the target shows it){tier_note}. The preamble "
             f"declares \\softmax, \\R and \\E.", sol, preamble=PRE)


@template(M, "lite")
def lite_a(rng):
    return _single(rng, ["softmax", "sgd", "ce", "kl"])


@template(M, "lite")
def lite_b(rng):
    return _single(rng, ["ce", "kl", "sgd", "layernorm"])


@template(M, "easy")
def easy_a(rng):
    return _single(rng, EASY)


@template(M, "easy")
def easy_b(rng):
    keys = pick(rng, [("softmax", "ce"), ("sgd", "kl"), ("softmax", "sgd"), ("ce", "kl")])
    lines = " \\\\\n  ".join(EQS[k][1] for k in keys)
    return L(f"Typeset {EQS[keys[0]][0]} and {EQS[keys[1]][0]} as two aligned, numbered lines (align).",
             f"\\begin{{align}}\n  {lines}\n\\end{{align}}", preamble=PRE, require=["env:align"])


@template(M, "medium")
def medium_a(rng):
    return _single(rng, MEDIUM)


@template(M, "medium")
def medium_b(rng):
    keys = pick(rng, [("attention", "layernorm"), ("lora", "attention"), ("adam_m", "sgd"), ("layernorm", "kl")])
    lines = " \\\\\n  ".join(EQS[k][1] for k in keys)
    return L(f"Typeset {EQS[keys[0]][0]} and {EQS[keys[1]][0]} as two aligned, numbered lines.",
             f"\\begin{{align}}\n  {lines}\n\\end{{align}}", preamble=PRE, require=["env:align"])


@template(M, "hard")
def hard_a(rng):
    return _single(rng, HARD, " (watch the \\! negative spaces and the \\bigl brackets)")


@template(M, "hard")
def hard_b(rng):
    keys = pick(rng, [("adam_m", "adam"), ("dpo", "kl"), ("backprop", "sgd"), ("lora", "dpo")])
    lines = " \\\\\n  ".join(EQS[k][1] for k in keys)
    return L(f"Typeset {EQS[keys[0]][0]} and {EQS[keys[1]][0]} as two aligned, numbered lines.",
             f"\\begin{{align}}\n  {lines}\n\\end{{align}}", preamble=PRE, require=["env:align"])


@template(M, "challenge")
@fixed
def gallery_half(rng):
    keys = ["softmax", "ce", "attention", "sgd", "adam_m"]
    lines = " \\\\\n  ".join(EQS[k][1] for k in keys)
    return L("**The gallery, first half.** Typeset the five equations shown (softmax, cross-entropy, attention, SGD, "
             "Adam's moments) in one align, numbered (1)\u2013(5).", f"\\begin{{align}}\n  {lines}\n\\end{{align}}",
             preamble=PRE, require=["env:align"])


@template(M, "challenge")
@fixed
def gallery_rest(rng):
    keys = ["adam", "kl", "layernorm", "lora", "dpo", "backprop"]
    lines = " \\\\\n  ".join(EQS[k][1] for k in keys)
    return L("**The gallery, second half.** Typeset the six equations shown (Adam's update, KL, LayerNorm, LoRA, DPO, "
             "backprop) in one align, numbered (1)\u2013(6).", f"\\begin{{align}}\n  {lines}\n\\end{{align}}",
             preamble=PRE, require=["env:align"])

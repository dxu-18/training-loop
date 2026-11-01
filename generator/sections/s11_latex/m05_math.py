"""11.05 Math mode basics: inline and display math, numbered equations and \\eqref, scripts, fractions, roots,
sums and integrals, \\text and \\operatorname, fonts (\\mathbb \\mathcal \\mathbf \\boldsymbol), accents, norms."""

from ._common import AMS, BASE, L, drill, fixed, template

M = "11.05"


# ================================================================= lite

@template(M, "lite")
def inline_display(rng):
    a = rng.choice(["x", "z", "t"])
    k = rng.randint(2, 4)
    items = [
        (f"Write the sentence \u201cThe loss is $f({a}) = {a}^{k}$.\u201d with the formula as inline math.",
         f"The loss is $f({a}) = {a}^{k}$.", {"require": ["$"]}),
        (f"Write \u201cWe minimize\u201d, then the formula $f({a}) = {a}^{k} + 1$ as unnumbered display math "
         f"(\\[ ... \\]).", f"We minimize\n\\[ f({a}) = {a}^{k} + 1 \\]", {"require": ["\\["]}),
        (f"Write the formula $\\hat{{y}} = \\sigma(Wx + b)$ as a numbered equation (equation environment).",
         "\\begin{equation}\n  \\hat{y} = \\sigma(Wx + b)\n\\end{equation}", {"require": ["env:equation"]}),
        (f"Write $E = mc^2$ as an UNnumbered equation with equation* (amsmath).",
         "\\begin{equation*}\n  E = mc^2\n\\end{equation*}", {"require": ["env:equation*"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "lite")
def scripts(rng):
    items = [
        ("Write inline: $x^2$, $x_i$, $x_i^2$, $x^{10}$, $x_{i,j}$ (comma-separated). Note the braces for "
         "multi-character scripts.", "$x^2$, $x_i$, $x_i^2$, $x^{10}$, $x_{i,j}$", {}),
        ("Write inline: $e^{-x^2/2}$ (the whole exponent in braces).", "$e^{-x^2/2}$", {}),
        ("Write inline: $W_{ij}^{(l)}$ (subscript ij, superscript (l)).", "$W_{ij}^{(l)}$", {}),
        ("Write inline: $\\theta_{t+1}$ and $\\beta_1^t$.", "$\\theta_{t+1}$ and $\\beta_1^t$", {}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= easy

@template(M, "easy")
def fractions_roots(rng):
    items = [
        ("Write inline: $\\frac{a}{b}$, $\\dfrac{a}{b}$ (display size) and $\\tfrac{1}{2}$ (text size).",
         "$\\frac{a}{b}$, $\\dfrac{a}{b}$ and $\\tfrac{1}{2}$", {"require": ["\\dfrac", "\\tfrac"]}),
        ("Write inline: $\\sqrt{x}$ and the cube root $\\sqrt[3]{x}$.", "$\\sqrt{x}$ and $\\sqrt[3]{x}$",
         {"require": ["\\sqrt"]}),
        ("Display (\\[ \\]): the standard deviation $\\sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(x_i - \\mu)^2}$.",
         "\\[ \\sigma = \\sqrt{\\frac{1}{N} \\sum_{i=1}^{N} (x_i - \\mu)^2} \\]", {"require": ["\\frac", "\\sum"]}),
        ("Display (\\[ \\]): the scaled dot product $\\frac{q^\\top k}{\\sqrt{d_k}}$.",
         "\\[ \\frac{q^\\top k}{\\sqrt{d_k}} \\]", {"require": ["\\frac", "\\sqrt"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "easy")
def big_operators(rng):
    n = rng.choice(["n", "N", "K"])
    items = [
        (f"Write inline: $\\sum_{{i=1}}^{{{n}}} x_i$ and $\\prod_{{k}} p_k$.",
         f"$\\sum_{{i=1}}^{{{n}}} x_i$ and $\\prod_{{k}} p_k$", {"require": ["\\sum", "\\prod"]}),
        ("Write inline the integral $\\int_0^\\infty f(x)\\,dx$ (thin space \\, before dx).",
         "$\\int_0^\\infty f(x)\\,dx$", {"require": ["\\int"]}),
        ("Write inline: $\\lim_{x \\to 0} f(x)$ and $\\max_{\\theta} J(\\theta)$.",
         "$\\lim_{x \\to 0} f(x)$ and $\\max_{\\theta} J(\\theta)$", {"require": ["\\lim", "\\max"]}),
        ("Write inline: $\\frac{\\partial L}{\\partial w}$ and $\\nabla_\\theta L$.",
         "$\\frac{\\partial L}{\\partial w}$ and $\\nabla_\\theta L$", {"require": ["\\partial", "\\nabla"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= medium

@template(M, "medium")
def text_in_math(rng):
    items = [
        ("Write inline: $\\text{if } x > 0$ (words in math with \\text, amsmath).", "$\\text{if } x > 0$",
         {"require": ["\\text"]}),
        ("Write inline: the differential $\\mathrm{d}x$ upright, and $\\operatorname{softmax}(z)$.",
         "$\\mathrm{d}x$ and $\\operatorname{softmax}(z)$", {"require": ["\\mathrm", "\\operatorname"]}),
        ("Write inline: the expectation $\\mathbb{E}_{x \\sim p}[f(x)]$ (blackboard-bold E: amssymb).",
         "$\\mathbb{E}_{x \\sim p}[f(x)]$", {"require": ["\\mathbb"]}),
        ("Write inline: the norm $\\|x\\|_2$, the absolute value $|x|$ and $\\langle u, v \\rangle$.",
         "$\\|x\\|_2$, $|x|$ and $\\langle u, v \\rangle$", {"require": ["\\langle"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "medium")
def math_fonts(rng):
    items = [
        ("Write inline: $\\hat{y}$ $\\bar{x}$ $\\tilde{x}$ $\\vec{v}$ $\\dot{x}$ (accents).",
         "$\\hat{y}$ $\\bar{x}$ $\\tilde{x}$ $\\vec{v}$ $\\dot{x}$", {"require": ["\\hat", "\\bar"]}),
        ("Write inline: $\\mathbf{W}$, $\\boldsymbol{\\theta}$, $\\mathbb{R}^{d}$ and $\\mathcal{D}$.",
         "$\\mathbf{W}$, $\\boldsymbol{\\theta}$, $\\mathbb{R}^{d}$ and $\\mathcal{D}$",
         {"require": ["\\mathbf", "\\boldsymbol", "\\mathcal"]}),
        ("Write inline the transpose $x^\\top$ and $W^\\top x$.", "$x^\\top$ and $W^\\top x$", {"require": ["\\top"]}),
        ("Write inline: $\\mathrm{d}x$, $\\mathsf{T}$, $\\mathtt{x}$ and $\\mathit{diff}$ (upright, sans, typewriter, "
         "italic word).", "$\\mathrm{d}x$, $\\mathsf{T}$, $\\mathtt{x}$ and $\\mathit{diff}$",
         {"require": ["\\mathrm", "\\mathsf"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= hard

@template(M, "hard")
def labelled(rng):
    name = rng.choice(["mse", "nll", "loss", "obj"])
    items = [
        (f"Write the negative log-likelihood as a numbered equation labelled eq:{name}, then the sentence "
         f"\u201cEquation (1) is the negative log-likelihood.\u201d with the number from \\eqref.",
         f"\\begin{{equation}}\\label{{eq:{name}}}\n  \\mathcal{{L}}(\\theta) = -\\frac{{1}}{{N}} \\sum_{{i=1}}^{{N}} "
         f"\\log p_\\theta(y_i \\mid x_i)\n\\end{{equation}}\nEquation~\\eqref{{eq:{name}}} is the negative "
         f"log-likelihood.", {"require": ["\\eqref", "env:equation"]}),
        (f"Write the MSE loss inline ($\\frac{{1}}{{n}}\\sum_i (y_i - \\hat{{y}}_i)^2$), then as a numbered equation "
         f"labelled eq:{name}, then \u201cSee (1).\u201d with \\eqref.",
         f"The loss is $\\frac{{1}}{{n}}\\sum_i (y_i - \\hat{{y}}_i)^2$.\n\\begin{{equation}}\\label{{eq:{name}}}\n"
         f"  \\text{{MSE}} = \\frac{{1}}{{n}} \\sum_{{i=1}}^{{n}} (y_i - \\hat{{y}}_i)^2\n\\end{{equation}}\n"
         f"See~\\eqref{{eq:{name}}}.", {"require": ["\\eqref"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "hard")
def fix_math(rng):
    items = [
        ("This doesn't compile (Missing $ inserted): the subscript is outside math. Fix it so x_i is math.",
         "The value $x_i$ is positive.", {"starter": "The value x_i is positive."}),
        ("\\mathbb needs a package that isn't loaded. Without changing the preamble, write the same with "
         "\\mathbf instead: $x \\in \\mathbf{R}^d$.", "$x \\in \\mathbf{R}^d$",
         {"starter": "$x \\in \\mathbb{R}^d$", "preamble": BASE}),
        ("Double subscript: fix $x_i_j$ so it reads x with subscript ij.", "$x_{ij}$", {"starter": "$x_i_j$"}),
        ("This doesn't compile (Missing $ inserted): \\alpha is outside math. Fix it so the learning rate "
         "$\\alpha$ is math.", "The learning rate $\\alpha$ decays.", {"starter": "The learning rate \\alpha decays."}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def expectation(rng):
    sol = ("For a model $p_\\theta$ and data $\\mathcal{D}$,\n\\begin{equation}\\label{eq:risk}\n"
           "  R(\\theta) = \\mathbb{E}_{(x, y) \\sim \\mathcal{D}} \\left[ \\ell\\bigl(f_\\theta(x), y\\bigr) \\right]"
           "\n\\end{equation}\nand Equation~\\eqref{eq:risk} is the expected risk.")
    return L("**Expected risk.** Recreate the target: the sentence, the numbered equation (with \\left[ ... \\right] "
             "and \\bigl( ... \\bigr)), and the reference to it.", sol, preamble=AMS,
             require=["\\eqref", "\\mathbb", "\\mathcal"])


@template(M, "challenge")
@fixed
def gradient(rng):
    sol = ("\\[ \\nabla_\\theta \\mathcal{L} = \\frac{1}{|B|} \\sum_{(x, y) \\in B} \\frac{\\partial "
           "\\ell(f_\\theta(x), y)}{\\partial \\theta} \\]\nwhere $\\|\\nabla_\\theta \\mathcal{L}\\|_2 \\le G$ "
           "and $\\theta \\in \\mathbb{R}^{d}$.")
    return L("**The mini-batch gradient.** Recreate the display and the sentence under it.", sol, preamble=AMS,
             require=["\\nabla", "\\partial"])

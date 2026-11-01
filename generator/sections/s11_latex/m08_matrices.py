"""11.08 Matrices, cases, brackets & custom operators: bmatrix / pmatrix / vmatrix with dots, cases, \\left \\right
and \\bigl, \\underbrace / \\overbrace / \\overset / \\xrightarrow, \\DeclareMathOperator and \\newcommand."""

from ._common import AMS, L, drill, fixed, template

M = "11.08"
MACROS = AMS + ("\\DeclareMathOperator*{\\argmax}{arg\\,max}\n\\DeclareMathOperator*{\\argmin}{arg\\,min}\n"
                "\\DeclareMathOperator{\\softmax}{softmax}\n\\newcommand{\\R}{\\mathbb{R}}\n"
                "\\newcommand{\\E}{\\mathbb{E}}\n\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}\n")


# ================================================================= lite

@template(M, "lite")
def matrices(rng):
    a, b, c, d = rng.sample(range(-5, 10), 4)
    items = [
        ("Write the 2\u00d72 weight matrix in square brackets (bmatrix): W = [w11 w12; w21 w22].",
         "\\[ W = \\begin{bmatrix} w_{11} & w_{12} \\\\ w_{21} & w_{22} \\end{bmatrix} \\]", {"require": ["env:bmatrix"]}),
        ("Write the 2\u00d72 identity matrix in parentheses (pmatrix).",
         "\\[ \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix} \\]", {"require": ["env:pmatrix"]}),
        (f"Write the determinant |{a} {b}; {c} {d}| with vmatrix.",
         f"\\[ \\begin{{vmatrix}} {a} & {b} \\\\ {c} & {d} \\end{{vmatrix}} \\]", {"require": ["env:vmatrix"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "lite")
def cases_env(rng):
    items = [
        ("Define ReLU piecewise with cases: x if x > 0, 0 otherwise.",
         "\\[ \\operatorname{ReLU}(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ 0 & \\text{otherwise} \\end{cases} "
         "\\]", {"require": ["env:cases"]}),
        ("Define the sign function with cases: 1 if x > 0, 0 if x = 0, \u22121 if x < 0.",
         "\\[ \\operatorname{sign}(x) = \\begin{cases} 1 & \\text{if } x > 0 \\\\ 0 & \\text{if } x = 0 \\\\ -1 & "
         "\\text{if } x < 0 \\end{cases} \\]", {"require": ["env:cases"]}),
        ("Define the absolute value with cases: x if x \u2265 0, \u2212x otherwise.",
         "\\[ |x| = \\begin{cases} x & \\text{if } x \\ge 0 \\\\ -x & \\text{otherwise} \\end{cases} \\]",
         {"require": ["env:cases"]}),
        ("Define the hinge loss with cases: 1 \u2212 yz if yz < 1, 0 otherwise.",
         "\\[ \\ell(y, z) = \\begin{cases} 1 - yz & \\text{if } yz < 1 \\\\ 0 & \\text{otherwise} \\end{cases} \\]",
         {"require": ["env:cases"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= easy

@template(M, "easy")
def general_matrix(rng):
    items = [
        ("Write a general m\u00d7n matrix X with dots (\\cdots, \\vdots, \\ddots) in bmatrix: x11 \u2026 x1n; "
         "\u22ee \u22f1 \u22ee; xm1 \u2026 xmn.",
         "\\[ X = \\begin{bmatrix} x_{11} & \\cdots & x_{1n} \\\\ \\vdots & \\ddots & \\vdots \\\\ x_{m1} & \\cdots "
         "& x_{mn} \\end{bmatrix} \\]", {"require": ["\\ddots"]}),
        ("Write a 3\u00d73 diagonal matrix diag(\u03bb1, \u03bb2, \u03bb3) in pmatrix (zeros off the diagonal).",
         "\\[ \\begin{pmatrix} \\lambda_1 & 0 & 0 \\\\ 0 & \\lambda_2 & 0 \\\\ 0 & 0 & \\lambda_3 \\end{pmatrix} \\]",
         {"require": ["env:pmatrix"]}),
        ("Write the column vector x = (x1, \u2026, xn) as a bmatrix with \\vdots: x1; \u22ee; xn.",
         "\\[ x = \\begin{bmatrix} x_1 \\\\ \\vdots \\\\ x_n \\end{bmatrix} \\]", {"require": ["\\vdots"]}),
        ("Write the n\u00d7n identity matrix I with dots in bmatrix: 1 0 \u2026 0; 0 1 \u2026 0; \u22ee \u22ee \u22f1 \u22ee; "
         "0 0 \u2026 1.",
         "\\[ I = \\begin{bmatrix} 1 & 0 & \\cdots & 0 \\\\ 0 & 1 & \\cdots & 0 \\\\ \\vdots & \\vdots & \\ddots & "
         "\\vdots \\\\ 0 & 0 & \\cdots & 1 \\end{bmatrix} \\]", {"require": ["\\ddots"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "easy")
def brackets(rng):
    items = [
        ("Write \\left( \\frac{a}{b} \\right) so the parentheses grow with the fraction.",
         "\\[ \\left( \\frac{a}{b} \\right) \\]", {"require": ["\\left"]}),
        ("Write the set \\left\\{ x \\mid x > 0 \\right\\} with growing braces.",
         "\\[ \\left\\{ x \\mid x > 0 \\right\\} \\]", {"require": ["\\left"]}),
        ("Write the evaluation bar: \\left. \\frac{df}{dx} \\right|_{x=0} (an invisible left delimiter).",
         "\\[ \\left. \\frac{df}{dx} \\right|_{x=0} \\]", {"require": ["\\left"]}),
        ("Write fixed bigger brackets: \\bigl( x \\bigr) and \\Bigl[ x \\Bigr].", "\\[ \\bigl( x \\bigr), \\Bigl[ x \\Bigr] \\]",
         {"require": ["\\bigl"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= medium

@template(M, "medium")
def braces_arrows(rng):
    n = rng.choice(["n", "k", "m"])
    items = [
        (f"Write a + \u22ef + a with a brace under it labelled \u201c{n} times\u201d (\\underbrace).",
         f"\\[ \\underbrace{{a + \\cdots + a}}_{{{n} \\text{{ times}}}} \\]", {"require": ["\\underbrace"]}),
        ("Write x1, \u2026, xk with a brace over it labelled k (\\overbrace).",
         "\\[ \\overbrace{x_1, \\dots, x_k}^{k} \\]", {"require": ["\\overbrace"]}),
        ("Write \u201cdef\u201d set over an equals sign (\\overset{\\text{def}}{=}).", "\\[ f \\overset{\\text{def}}{=} g \\]",
         {"require": ["\\overset"]}),
        ("Write an arrow labelled \u201ctrain\u201d: data \u2192 model (\\xrightarrow{\\text{train}}).",
         "\\[ \\text{data} \\xrightarrow{\\text{train}} \\text{model} \\]", {"require": ["\\xrightarrow"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "medium")
def declare_operator(rng):
    body = "\\[ \\hat{c} = \\argmax_{c} p(c \\mid x), \\quad \\theta^* = \\argmin_\\theta L(\\theta) \\]"
    items = [
        ("Write the preamble so this body compiles: it uses \\argmax and \\argmin with the subscript UNDER them in "
         "displays (\\DeclareMathOperator* in the preamble; amsmath and amssymb too).",
         AMS + "\\DeclareMathOperator*{\\argmax}{arg\\,max}\n\\DeclareMathOperator*{\\argmin}{arg\\,min}",
         {"mode": "preamble", "body": body, "require": ["\\DeclareMathOperator"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble so this body compiles: \\softmax is an operator (upright, no limits).",
         AMS + "\\DeclareMathOperator{\\softmax}{softmax}",
         {"mode": "preamble", "body": "\\[ y = \\softmax(Wx) \\]", "require": ["\\DeclareMathOperator"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble so this body compiles: \\Tr is the trace operator (upright Tr, no limits).",
         AMS + "\\DeclareMathOperator{\\Tr}{Tr}",
         {"mode": "preamble", "body": "\\[ \\Tr(A B) = \\Tr(B A) \\]", "require": ["\\DeclareMathOperator"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble so this body compiles: \\diag (upright diag) and \\rank (upright rank) are operators.",
         AMS + "\\DeclareMathOperator{\\diag}{diag}\n\\DeclareMathOperator{\\rank}{rank}",
         {"mode": "preamble", "body": "\\[ \\rank(\\diag(\\lambda_1, \\dots, \\lambda_n)) \\le n \\]",
          "require": ["\\DeclareMathOperator"], "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def macros(rng):
    body = "Let $x \\in \\R^d$ and $\\norm{x}_2 \\le 1$, so $\\E[X] = 0$."
    items = [
        ("Write the preamble (amsmath, amssymb) so this body compiles: \\R is \\mathbb{R}, \\E is \\mathbb{E}, and "
         "\\norm{x} is \\left\\lVert x \\right\\rVert (a macro with one argument).",
         AMS + "\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\E}{\\mathbb{E}}\n"
               "\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}",
         {"mode": "preamble", "body": body, "require": ["\\newcommand"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble (amsmath, amssymb) so this body compiles: \\N is \\mathbb{N}, \\Z is \\mathbb{Z}, and "
         "\\abs{x} is \\left| x \\right| (one argument).",
         AMS + "\\newcommand{\\N}{\\mathbb{N}}\n\\newcommand{\\Z}{\\mathbb{Z}}\n"
               "\\newcommand{\\abs}[1]{\\left| #1 \\right|}",
         {"mode": "preamble", "body": "For $n \\in \\N$ and $k \\in \\Z$, $\\abs{k} \\le n$.", "require": ["\\newcommand"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble (amsmath, amssymb) so this body compiles: \\Var is \\operatorname{Var}, \\E is "
         "\\mathbb{E}, and \\given is \\mid (no arguments).",
         AMS + "\\newcommand{\\Var}{\\operatorname{Var}}\n\\newcommand{\\E}{\\mathbb{E}}\n"
               "\\newcommand{\\given}{\\mid}",
         {"mode": "preamble", "body": "$\\Var(X) = \\E[X^2] - \\E[X]^2$ and $\\E[Y \\given X]$.",
          "require": ["\\newcommand"], "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
        ("Write the preamble (amsmath, amssymb) so this body compiles: \\vect{x} is \\mathbf{x} and \\T is "
         "^\\top (a transpose you write after the symbol).",
         AMS + "\\newcommand{\\vect}[1]{\\mathbf{#1}}\n\\newcommand{\\T}{^\\top}",
         {"mode": "preamble", "body": "$y = \\vect{w}\\T \\vect{x}$", "require": ["\\newcommand"],
          "starter": "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"}),
    ]
    return drill(rng, items)


@template(M, "hard")
def use_macros(rng):
    items = [
        ("The preamble defines \\argmax, \\softmax, \\R, \\E and \\norm. Use them to write: "
         "\\hat{c} = \\argmax_c p(c | x), \u2016x\u2016\u2082 for x in R^d, and E[X], in one display separated by "
         "\\quad.", "\\[ \\hat{c} = \\argmax_{c} p(c \\mid x), \\quad \\norm{x}_2, \\quad x \\in \\R^d, \\quad \\E[X] \\]",
         {"require": ["\\argmax", "\\norm", "\\R", "\\E"]}),
        ("The preamble defines \\argmin, \\R and \\norm. Use them to write ridge regression: "
         "\\hat{w} = \\argmin_{w \\in R^d} \u2016Xw \u2212 y\u2016\u2082\u00b2 + \u03bb\u2016w\u2016\u2082\u00b2.",
         "\\[ \\hat{w} = \\argmin_{w \\in \\R^d} \\norm{Xw - y}_2^2 + \\lambda \\norm{w}_2^2 \\]",
         {"require": ["\\argmin", "\\norm", "\\R"]}),
        ("The preamble defines \\softmax, \\E and \\R. Use them to write: p = \\softmax(z) with z in R^K, and the "
         "expected loss E[\u2113(p, y)], in one display separated by \\quad.",
         "\\[ p = \\softmax(z), \\quad z \\in \\R^K, \\quad \\E[\\ell(p, y)] \\]",
         {"require": ["\\softmax", "\\E", "\\R"]}),
        ("The preamble defines \\argmax, \\E and \\norm. Use them to write the policy \u03c0(s) = \\argmax_a "
         "E[R | s, a] and the bound \u2016\u2207J\u2016 \u2264 G, separated by \\quad.",
         "\\[ \\pi(s) = \\argmax_{a} \\E[R \\mid s, a], \\quad \\norm{\\nabla J} \\le G \\]",
         {"require": ["\\argmax", "\\E", "\\norm"]}),
    ]
    return drill(rng, items, preamble=MACROS)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def attention_block(rng):
    sol = ("\\[ \\operatorname{Attn}(Q, K, V) = \\softmax\\!\\left( \\frac{Q K^\\top}{\\sqrt{d_k}} \\right) V, "
           "\\quad Q = \\begin{bmatrix} q_1^\\top \\\\ \\vdots \\\\ q_n^\\top \\end{bmatrix} \\in \\R^{n \\times d_k} "
           "\\]")
    return L("**Attention, in matrices.** Recreate the target with the preamble's macros (\\softmax, \\R).", sol,
             preamble=MACROS, require=["\\softmax", "env:bmatrix"])


@template(M, "challenge")
@fixed
def piecewise_loss(rng):
    sol = ("\\[ \\ell_\\delta(r) = \\begin{cases} \\frac{1}{2} r^2 & \\text{if } |r| \\le \\delta \\\\ "
           "\\delta \\left( |r| - \\frac{1}{2} \\delta \\right) & \\text{otherwise} \\end{cases} \\]")
    return L("**The Huber loss.** Recreate the target.", sol, preamble=AMS, require=["env:cases", "\\left"])

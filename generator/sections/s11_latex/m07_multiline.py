"""11.07 Multi-line equations & numbering: align / align*, \\nonumber, && \\text{...}, gather, multline, split."""

from ._common import AMS, L, drill, fixed, template

M = "11.07"
# (left side, right side) of short equations the drills combine
PAIRS = [("y", "Wx + b"), ("h", "\\sigma(y)"), ("p", "\\operatorname{softmax}(h)"), ("L", "-\\log p_y"),
         ("m", "\\beta m + g"), ("z", "x / \\lVert x \\rVert")]


def _lines(rng, n):
    return rng.sample(PAIRS, n)


# ================================================================= lite

@template(M, "lite")
def align_basic(rng):
    v = rng.choice(["W", "\\theta", "w"])
    items = [
        (f"Write the gradient step as two aligned, numbered lines (align, aligned at =):\n\n"
         f"\u2207L = X\u1d40(\u0176 \u2212 Y)\n\n{v} \u2190 {v} \u2212 \u03b7\u2207L",
         f"\\begin{{align}}\n  \\nabla L &= X^\\top (\\hat{{Y}} - Y) \\\\\n  {v} &\\gets {v} - \\eta \\nabla L\n"
         f"\\end{{align}}", {"require": ["env:align"]}),
        ("Write the same two lines, unnumbered (align*): $\\nabla L = X^\\top(\\hat{Y} - Y)$ and "
         "$W \\gets W - \\eta \\nabla L$, aligned at the relation.",
         "\\begin{align*}\n  \\nabla L &= X^\\top (\\hat{Y} - Y) \\\\\n  W &\\gets W - \\eta \\nabla L\n\\end{align*}",
         {"require": ["env:align*"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "lite")
def nonumber(rng):
    (a, ra), (b, rb), (c, rc) = _lines(rng, 3)
    items = [
        (f"Write two aligned lines where only the FIRST is numbered (\\nonumber on the second): ${a} = {ra}$ and "
         f"${b} = {rb}$.", f"\\begin{{align}}\n  {a} &= {ra} \\\\\n  {b} &= {rb} \\nonumber\n\\end{{align}}",
         {"require": ["\\nonumber"]}),
        (f"Write three aligned lines numbered 1 and 2 only on the first and last (${a} = {ra}$, ${b} = {rb}$, "
         f"${c} = {rc}$: the middle one unnumbered).",
         f"\\begin{{align}}\n  {a} &= {ra} \\\\\n  {b} &= {rb} \\nonumber \\\\\n  {c} &= {rc}\n\\end{{align}}",
         {"require": ["\\nonumber"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= easy

@template(M, "easy")
def derivation(rng):
    items = [
        ("Expand $(x+1)^2$ in two aligned unnumbered lines, with the note \u201cexpand\u201d at the right of the second "
         "line (&& \\text{expand}).",
         "\\begin{align*}\n  f(x) &= (x + 1)^2 \\\\\n       &= x^2 + 2x + 1 && \\text{expand}\n\\end{align*}",
         {"require": ["env:align*", "\\text"]}),
        ("Write $(a+b)^2 = a^2 + 2ab + b^2$ and then $\\ge 4ab$ on the next line, aligned at the relations, "
         "unnumbered, with the note \u201cAM\u2013GM\u201d on the second line.",
         "\\begin{align*}\n  (a + b)^2 &= a^2 + 2ab + b^2 \\\\\n            &\\ge 4ab && \\text{AM--GM}\n\\end{align*}",
         {"require": ["env:align*"]}),
        ("Write $\\log(xy) = \\log x + \\log y$ and then $= \\log x - \\log(1/y)$ on the next line, aligned at the "
         "equals signs, unnumbered, with the note \u201cproduct rule\u201d on the first line.",
         "\\begin{align*}\n  \\log(xy) &= \\log x + \\log y && \\text{product rule} \\\\\n"
         "           &= \\log x - \\log(1/y)\n\\end{align*}", {"require": ["env:align*", "\\text"]}),
        ("Write $\\sigma'(z) = \\frac{e^{-z}}{(1 + e^{-z})^2}$ and then $= \\sigma(z)(1 - \\sigma(z))$ on the next line, "
         "aligned, unnumbered, with the note \u201csimplify\u201d on the second line.",
         "\\begin{align*}\n  \\sigma'(z) &= \\frac{e^{-z}}{(1 + e^{-z})^2} \\\\\n"
         "             &= \\sigma(z)(1 - \\sigma(z)) && \\text{simplify}\n\\end{align*}",
         {"require": ["env:align*", "\\text"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "easy")
def gather_env(rng):
    (a, ra), (b, rb) = _lines(rng, 2)
    items = [
        (f"Write two centered, unaligned, unnumbered equations: ${a} = {ra}$ and ${b} = {rb}$ (gather*).",
         f"\\begin{{gather*}}\n  {a} = {ra} \\\\\n  {b} = {rb}\n\\end{{gather*}}", {"require": ["env:gather*"]}),
        (f"Write two centered equations, each numbered (gather): ${a} = {ra}$ and ${b} = {rb}$.",
         f"\\begin{{gather}}\n  {a} = {ra} \\\\\n  {b} = {rb}\n\\end{{gather}}", {"require": ["env:gather"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= medium

@template(M, "medium")
def multline_env(rng):
    k = rng.randint(3, 7)
    first = " + ".join(f"a_{i} x^{i}" if i > 1 else ("a_0" if i == 0 else "a_1 x") for i in range(k))
    rest = " + ".join(f"a_{i} x^{i}" for i in range(k, k + 2))
    items = [
        (f"Break this long polynomial over two lines with multline* (first line flush left, second flush right): "
         f"p(x) = {first} + {rest}.",
         f"\\begin{{multline*}}\n  p(x) = {first} \\\\\n  + {rest}\n\\end{{multline*}}",
         {"require": ["env:multline*"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "medium")
def split_env(rng):
    items = [
        ("Write one numbered equation whose two lines are aligned (split inside equation): $(a + b)^2 = a^2 + 2ab + "
         "b^2$ and then $\\ge 4ab$.",
         "\\begin{equation}\n  \\begin{split}\n    (a + b)^2 &= a^2 + 2ab + b^2 \\\\\n              &\\ge 4ab\n"
         "  \\end{split}\n\\end{equation}", {"require": ["env:split"]}),
        ("Write one numbered equation whose two lines are aligned (split inside equation): $(a - b)^2 = a^2 - 2ab + "
         "b^2$ and then $\\ge 0$.",
         "\\begin{equation}\n  \\begin{split}\n    (a - b)^2 &= a^2 - 2ab + b^2 \\\\\n              &\\ge 0\n"
         "  \\end{split}\n\\end{equation}", {"require": ["env:split"]}),
        ("Write one numbered equation whose two lines are aligned (split inside equation): $\\mathcal{L} = "
         "\\frac{1}{n} \\sum_i \\ell_i$ and then $= \\bar{\\ell}$.",
         "\\begin{equation}\n  \\begin{split}\n    \\mathcal{L} &= \\frac{1}{n} \\sum_i \\ell_i \\\\\n"
         "                &= \\bar{\\ell}\n  \\end{split}\n\\end{equation}", {"require": ["env:split"]}),
        ("Write one numbered equation whose two lines are aligned (split inside equation): $\\nabla_w \\frac{1}{2} "
         "\\lVert Xw - y \\rVert^2 = X^\\top (Xw - y)$ and then $= X^\\top X w - X^\\top y$.",
         "\\begin{equation}\n  \\begin{split}\n    \\nabla_w \\frac{1}{2} \\lVert Xw - y \\rVert^2 &= X^\\top (Xw - y) "
         "\\\\\n      &= X^\\top X w - X^\\top y\n  \\end{split}\n\\end{equation}", {"require": ["env:split"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= hard

@template(M, "hard")
@fixed
def mse_gradient(rng):
    sol = ("\\begin{align}\n  \\frac{\\partial}{\\partial w} \\frac{1}{n} \\sum_{i=1}^{n} (w x_i - y_i)^2\n"
           "    &= \\frac{1}{n} \\sum_{i=1}^{n} 2 (w x_i - y_i) x_i \\nonumber \\\\\n"
           "    &= \\frac{2}{n} \\sum_{i=1}^{n} (w x_i - y_i) x_i \\label{eq:grad}\n\\end{align}\n"
           "Setting \\eqref{eq:grad} to zero gives $w$.")
    return L("**The MSE gradient.** Derive it in aligned steps with a single equation number on the last line "
             "(labelled eq:grad), then refer to it.", sol, preamble=AMS, require=["env:align", "\\nonumber",
                                                                                 "\\eqref"])


@template(M, "hard")
def fix_alignment(rng):
    (a, ra), (b, rb) = _lines(rng, 2)
    good = f"\\begin{{align}}\n  {a} &= {ra} \\\\\n  {b} &= {rb}\n\\end{{align}}"
    items = [
        ("The second line doesn't line up: it's missing its &. Fix it.", good,
         {"starter": f"\\begin{{align}}\n  {a} &= {ra} \\\\\n  {b} = {rb}\n\\end{{align}}"}),
        ("This doesn't compile: & has no meaning in equation (Misplaced alignment tab). Use align so the two "
         "lines align.", good, {"starter": f"\\begin{{equation}}\n  {a} &= {ra} \\\\\n  {b} &= {rb}\n\\end{{equation}}"}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def em_step(rng):
    sol = ("\\begin{align}\n  \\log p(x) &= \\log \\sum_z p(x, z) \\nonumber \\\\\n"
           "             &= \\log \\sum_z q(z) \\frac{p(x, z)}{q(z)} \\nonumber \\\\\n"
           "             &\\ge \\sum_z q(z) \\log \\frac{p(x, z)}{q(z)} && \\text{Jensen} \\label{eq:elbo}\n"
           "\\end{align}")
    return L("**The ELBO.** Recreate the derivation: three aligned lines, only the last numbered, with the note "
             "\u201cJensen\u201d on the right.", sol, preamble=AMS, require=["env:align", "\\nonumber"])


@template(M, "challenge")
@fixed
def mixed_envs(rng):
    sol = ("\\begin{gather*}\n  y = Wx + b\n\\end{gather*}\n\\begin{align}\n  h &= \\sigma(y) \\\\\n"
           "  \\hat{y} &= \\operatorname{softmax}(V h)\n\\end{align}\n\\begin{multline*}\n"
           "  L = \\ell_1 + \\ell_2 + \\ell_3 \\\\\n  + \\ell_4 + \\ell_5\n\\end{multline*}")
    return L("**Three environments.** Recreate the target: a centered unnumbered line, two aligned numbered lines, "
             "and a long sum broken over two lines.", sol, preamble=AMS,
             require=["env:gather*", "env:align", "env:multline*"])

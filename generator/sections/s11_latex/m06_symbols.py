"""11.06 Symbols reference: Greek letters, relations, operators, set notation and logic, arrows, dots, functions."""

from ._common import AMS, L, drill, template

M = "11.06"
GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "kappa", "lambda", "mu",
         "nu", "xi", "pi", "rho", "sigma", "tau", "phi", "varphi", "chi", "psi", "omega"]
CAPS = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega"]
RELS = ["le", "ge", "ne", "approx", "equiv", "sim", "simeq", "propto", "ll", "gg"]
OPS = ["pm", "mp", "times", "cdot", "div", "circ", "odot", "otimes", "oplus"]
SETS = ["in", "notin", "subset", "subseteq", "cup", "cap", "emptyset", "forall", "exists", "neg", "land", "lor"]
ARROWS = ["to", "leftarrow", "gets", "Rightarrow", "Leftrightarrow", "mapsto", "uparrow", "downarrow"]
FUNCS = ["sin", "cos", "log", "ln", "exp", "det", "min", "max", "sup", "inf", "arg"]


def _seq(names):
    return " ".join(f"\\{n}" for n in names)


# ================================================================= lite

@template(M, "lite")
def greek(rng):
    k = rng.randint(4, 6)
    xs = rng.sample(GREEK, k)
    caps = rng.sample(CAPS, 3)
    items = [
        (f"Write these Greek letters in one inline formula, separated by spaces: {', '.join(xs)}.",
         f"${_seq(xs)}$", {"require": ["\\" + xs[0]]}),
        (f"Write these capital Greek letters inline: {', '.join(caps)}.", f"${_seq(caps)}$",
         {"require": ["\\" + caps[0]]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "lite")
def relations(rng):
    xs = rng.sample(RELS, 4)
    sol = "$" + " ".join(f"a \\{r} b" + ("," if i < 3 else "") for i, r in enumerate(xs)) + "$"
    return L("Write inline, comma-separated: " + ", ".join(f"a {r} b" for r in xs) + " (each relation as its "
             "command: \\le, \\approx ...).", sol, preamble=AMS, require=["\\" + xs[0]])


# ================================================================= easy

@template(M, "easy")
def operators(rng):
    xs = rng.sample(OPS, 4)
    items = [
        ("Write inline, comma-separated: " + ", ".join(f"a {o} b" for o in xs) + ".",
         "$" + ", ".join(f"a \\{o} b" for o in xs) + "$", {"require": ["\\" + xs[0]]}),
        ("Write the Hadamard (elementwise) product of two matrices inline: $A \\odot B$, and their Kronecker product "
         "$A \\otimes B$.", "$A \\odot B$ and $A \\otimes B$", {"require": ["\\odot", "\\otimes"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "easy")
def sets_logic(rng):
    items = [
        ("Write inline: for all x in R, there exists y such that y > x (\\forall, \\in, \\mathbb{R}, \\exists).",
         "$\\forall x \\in \\mathbb{R}, \\exists y : y > x$", {"require": ["\\forall", "\\exists"]}),
        ("Write inline: $A \\cup B$, $A \\cap B$, $A \\subseteq B$, $x \\notin \\emptyset$.",
         "$A \\cup B$, $A \\cap B$, $A \\subseteq B$, $x \\notin \\emptyset$",
         {"require": ["\\cup", "\\notin", "\\emptyset"]}),
        ("Write inline: $p \\land q \\implies p$ and $\\neg (p \\lor q) \\iff \\neg p \\land \\neg q$ "
         "(\\implies needs amsmath).", "$p \\land q \\implies p$ and $\\neg (p \\lor q) \\iff \\neg p \\land \\neg q$",
         {"require": ["\\implies", "\\iff"]}),
        ("Write inline: $A \\setminus B$, $A \\subset B \\subseteq C$, and $x \\in A \\cap B$.",
         "$A \\setminus B$, $A \\subset B \\subseteq C$, and $x \\in A \\cap B$", {"require": ["\\setminus"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= medium

@template(M, "medium")
def arrows(rng):
    xs = rng.sample(ARROWS, 4)
    items = [
        ("Write inline, comma-separated: " + ", ".join(f"a {x} b" for x in xs) + ".",
         "$" + ", ".join(f"a \\{x} b" for x in xs) + "$", {"require": ["\\" + xs[0]]}),
        ("Write inline: the function $f : \\mathbb{R}^n \\to \\mathbb{R}$, $x \\mapsto \\|x\\|^2$.",
         "$f : \\mathbb{R}^n \\to \\mathbb{R}$, $x \\mapsto \\|x\\|^2$", {"require": ["\\to", "\\mapsto"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "medium")
def misc_symbols(rng):
    items = [
        ("Write inline: $\\infty$, $\\partial$, $\\nabla$, $\\ell$, $\\top$, $\\perp$, $x'$, $A^\\dagger$.",
         "$\\infty$, $\\partial$, $\\nabla$, $\\ell$, $\\top$, $\\perp$, $x'$, $A^\\dagger$",
         {"require": ["\\infty", "\\ell"]}),
        ("Write inline: $x_1, \\ldots, x_n$ and $x_1 + \\cdots + x_n$ (\\ldots on the baseline, \\cdots centered).",
         "$x_1, \\ldots, x_n$ and $x_1 + \\cdots + x_n$", {"require": ["\\cdots"]}),
        ("Write inline: $a \\approx b$, $p \\propto q$, $x \\sim \\mathcal{N}(0, 1)$, $f \\equiv g$.",
         "$a \\approx b$, $p \\propto q$, $x \\sim \\mathcal{N}(0, 1)$, $f \\equiv g$",
         {"require": ["\\approx", "\\propto"]}),
        ("Write inline: $\\pm 1$, $a \\times b$, $a \\div b$, $f \\circ g$, $a \\cdot b$.",
         "$\\pm 1$, $a \\times b$, $a \\div b$, $f \\circ g$, $a \\cdot b$", {"require": ["\\times", "\\circ"]}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= hard

@template(M, "hard")
def functions(rng):
    xs = rng.sample(FUNCS, 5)
    items = [
        ("Write inline with the upright function commands: " + ", ".join(f"{f} x" for f in xs) + ".",
         "$" + ", ".join(f"\\{f} x" for f in xs) + "$", {"require": ["\\" + xs[0]]}),
        ("Write inline: $\\log(1 + e^x)$ and $\\operatorname{tr}(A)$ and $\\det(A)$ (\\operatorname for trace).",
         "$\\log(1 + e^x)$ and $\\operatorname{tr}(A)$ and $\\det(A)$", {"require": ["\\log", "\\operatorname"]}),
        ("Write inline: $\\sin^2 x + \\cos^2 x = 1$ and $x \\ne 0$.", "$\\sin^2 x + \\cos^2 x = 1$ and $x \\ne 0$",
         {"require": ["\\sin", "\\ne"]}),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "hard")
def italic_vs_upright(rng):
    f, g = rng.choice([("log", "sin"), ("ln", "cos"), ("exp", "tanh"), ("log", "det")])
    m, e = rng.choice([("max", "exp"), ("min", "log"), ("sup", "exp"), ("max", "tanh")])
    items = [
        (f"This prints \u201c{f}\u201d as italic variables. Fix it with the upright commands (\\{f}, \\{g}): "
         f"${f}(x) + {g}(x)$.", f"$\\{f}(x) + \\{g}(x)$", {"starter": f"${f}(x) + {g}(x)$"}),
        (f"Fix the spacing of \u201c{m}\u201d and \u201c{e}\u201d: ${m}_i {e}(z_i)$.", f"$\\{m}_i \\{e}(z_i)$",
         {"starter": f"${m}_i {e}(z_i)$"}),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= challenge

@template(M, "challenge")
def from_memory(rng):
    xs = rng.sample(GREEK, 12)
    sol = f"${_seq(xs)}$"
    return L("**From memory.** Write the twelve Greek letters shown in the target, in that order, in one inline "
             "formula.", sol, preamble=AMS)


@template(M, "challenge")
def sentence(rng):
    name, sol = rng.choice([
        ("Continuity", "$\\forall \\varepsilon > 0, \\exists \\delta > 0 : |x - y| < \\delta \\implies |f(x) - f(y)| < "
                       "\\varepsilon$"),
        ("A limit", "$\\forall \\varepsilon > 0, \\exists N \\in \\mathbb{N} : n \\ge N \\implies |a_n - a| < "
                    "\\varepsilon$"),
        ("Convexity", "$\\forall x, y \\in \\mathbb{R}^n, \\forall \\lambda \\in [0, 1] : f(\\lambda x + (1 - \\lambda) y) "
                      "\\le \\lambda f(x) + (1 - \\lambda) f(y)$"),
        ("Lipschitz", "$\\exists L \\ge 0, \\forall x, y : \\|f(x) - f(y)\\| \\le L \\|x - y\\|$"),
    ])
    return L(f"**{name}.** Write the target's sentence in one inline formula.", sol, preamble=AMS,
             require=["\\forall"])

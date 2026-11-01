"""11.14 Theorems & custom macros: \\newtheorem (numbered within sections, shared counters), \\theoremstyle, proof,
\\newenvironment, \\newcommand with arguments and defaults."""

from ._common import AMS, L, drill, fixed, template

M = "11.14"
THM = AMS + "\\usepackage{amsthm}\n"
DEFS = THM + ("\\newtheorem{theorem}{Theorem}[section]\n\\newtheorem{lemma}[theorem]{Lemma}\n\\theoremstyle{definition}\n"
              "\\newtheorem{definition}{Definition}\n")
MACROS = DEFS + ("\\newenvironment{note}{\\par\\noindent\\textbf{Note.}\\ }{\\par}\n\\newcommand{\\vect}[1]{\\mathbf{#1}}\n"
                 "\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\E}{\\mathbb{E}}\n"
                 "\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}\n")
CONVEX = ("A function $f$ is convex if $f(\\lambda x + (1 - \\lambda) y) \\le \\lambda f(x) + (1 - \\lambda) f(y)$ for "
          "all $\\lambda \\in [0, 1]$.")
GD = "For convex, $L$-smooth $f$ and step size $\\eta \\le 1/L$, gradient descent converges at rate $O(1/t)$."
DEFINITIONS = {
    "Convexity": CONVEX,
    "Smoothness": "A function $f$ is $L$-smooth if $\\lVert \\nabla f(x) - \\nabla f(y) \\rVert \\le L \\lVert x - y \\rVert$.",
    "Calibration": "A classifier is calibrated if $P(Y = \\hat{y} \\mid \\hat{p} = p) = p$ for all $p$.",
    "Overfitting": "A model overfits when its training loss falls while its validation loss rises.",
}
STATEMENTS = [GD, "Adam's bias-corrected moments are unbiased estimates of the gradient's moments.",
              "Temperature scaling leaves the accuracy unchanged.",
              "With dropout rate $p$, the expected activation is unchanged after rescaling by $1/(1 - p)$."]
PROOFS = ["Combine smoothness with convexity and telescope the per-step decrease.",
          "Take expectations on both sides and divide by $1 - \\beta^t$.",
          "The argmax of $z / T$ equals the argmax of $z$ for every $T > 0$.",
          "It follows from Jensen's inequality."]
SECTIONS = ["Analysis", "Theory", "Convergence", "Results"]
PRE_STARTER = "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"
THM_STARTER = PRE_STARTER + "\\usepackage{amsthm}\n"


# ================================================================= lite

@template(M, "lite")
def use_theorems(rng):
    name = rng.choice(list(DEFINITIONS))
    sec = rng.choice(SECTIONS)
    st = rng.choice(STATEMENTS)
    items = [
        (f"The preamble defines theorem, lemma and definition. Write the definition titled \u201c{name}\u201d (the "
         f"optional argument) with the text shown.",
         f"\\begin{{definition}}[{name}]\n{DEFINITIONS[name]}\n\\end{{definition}}", {"require": ["env:definition"]}),
        (f"The preamble defines theorem, lemma and definition. Write the theorem shown, in a section titled {sec}.",
         f"\\section{{{sec}}}\n\\begin{{theorem}}\n{st}\n\\end{{theorem}}", {"require": ["env:theorem"]}),
    ]
    return drill(rng, items, preamble=DEFS)


@template(M, "lite")
def proof(rng):
    i = rng.randrange(len(STATEMENTS))
    st, pf = STATEMENTS[i], PROOFS[i]
    title = rng.choice(["Proof sketch", "Sketch", "Proof of the bound"])
    items = [
        (f"Write the theorem shown, then its proof (the proof environment ends it with a box): \u201c{pf}\u201d",
         f"\\begin{{theorem}}\n{st}\n\\end{{theorem}}\n\\begin{{proof}}\n{pf}\n\\end{{proof}}", {"require": ["env:proof"]}),
        (f"Write a proof titled \u201c{title}\u201d (the optional argument): \u201c{pf}\u201d",
         f"\\begin{{proof}}[{title}]\n{pf}\n\\end{{proof}}", {"require": ["env:proof"]}),
    ]
    return drill(rng, items, preamble=DEFS)


# ================================================================= easy

@template(M, "easy")
def define_theorem(rng):
    sec = rng.choice(SECTIONS)
    st = rng.choice(STATEMENTS)
    body = (f"\\section{{{sec}}}\n\\begin{{theorem}}\n{st}\n\\end{{theorem}}\n\\begin{{lemma}}\nThe loss decreases "
            f"every step.\n\\end{{lemma}}")
    items = [
        ("Write the preamble (amsmath, amssymb, amsthm) so this body compiles as shown: a theorem numbered within "
         "sections (Theorem 1.1) and a lemma that SHARES its counter (Lemma 1.2).",
         THM + "\\newtheorem{theorem}{Theorem}[section]\n\\newtheorem{lemma}[theorem]{Lemma}",
         {"mode": "preamble", "body": body, "require": ["\\newtheorem"], "starter": THM_STARTER}),
        ("Write the preamble (amsmath, amssymb, amsthm) so this body compiles as shown: theorem and lemma with "
         "their OWN counters (Theorem 1, Lemma 1), not numbered within sections.",
         THM + "\\newtheorem{theorem}{Theorem}\n\\newtheorem{lemma}{Lemma}",
         {"mode": "preamble", "body": body, "require": ["\\newtheorem"], "starter": THM_STARTER}),
    ]
    return drill(rng, items)


@template(M, "easy")
def simple_macros(rng):
    items = [
        ("Write the preamble so the body compiles: \\loss prints $\\mathcal{L}$ (the guide's drill); amsmath and "
         "amssymb too.", AMS + "\\newcommand{\\loss}{\\mathcal{L}}",
         {"mode": "preamble", "body": "Minimize $\\loss(\\theta)$ over $\\theta$.", "require": ["\\newcommand"],
          "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\vect{x} prints a bold vector $\\mathbf{x}$ (one argument); "
         "amsmath and amssymb too.", AMS + "\\newcommand{\\vect}[1]{\\mathbf{#1}}",
         {"mode": "preamble", "body": "Let $\\vect{x}$ and $\\vect{w}$ be vectors, $y = \\vect{w}^\\top \\vect{x}$.",
          "require": ["\\newcommand"], "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\R prints $\\mathbb{R}$ and \\E prints $\\mathbb{E}$; amsmath "
         "and amssymb too.", AMS + "\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\E}{\\mathbb{E}}",
         {"mode": "preamble", "body": "For $x \\in \\R^d$, $\\E[x] = 0$.", "require": ["\\newcommand"],
          "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\model prints the model's name, \\textsc{TinyNet}, in text; "
         "amsmath and amssymb too.", AMS + "\\newcommand{\\model}{\\textsc{TinyNet}}",
         {"mode": "preamble", "body": "We train \\model{} for ten epochs; \\model{} converges.",
          "require": ["\\newcommand"], "starter": PRE_STARTER}),
    ]
    return drill(rng, items)


# ================================================================= medium

@template(M, "medium")
def theorem_styles(rng):
    d_name, d_text = rng.choice(list(DEFINITIONS.items()))
    st = rng.choice(STATEMENTS)
    rem = rng.choice(["It only rescales the logits.", "The constant is not tight.", "Convexity can be relaxed.",
                      "The rate holds in expectation."])
    body = (f"\\begin{{definition}}[{d_name}]\n{d_text}\n\\end{{definition}}\n\\begin{{theorem}}\n{st}\n\\end{{theorem}}\n"
            f"\\begin{{remark}}\n{rem}\n\\end{{remark}}")
    return L("Write the preamble so this body compiles as shown: theorem in the plain style (italic body), "
             "definition in the definition style (upright), remark in the remark style (italic label), each with its "
             "own counter (amsmath, amssymb, amsthm).",
             THM + "\\newtheorem{theorem}{Theorem}\n\\theoremstyle{definition}\n\\newtheorem{definition}{Definition}\n"
                   "\\theoremstyle{remark}\n\\newtheorem{remark}{Remark}",
             mode="preamble", body=body, require=["\\theoremstyle"], starter=THM_STARTER)


@template(M, "medium")
def macro_args(rng):
    items = [
        ("Write the preamble so the body compiles: \\norm{x} prints $\\left\\lVert x \\right\\rVert$ and \\R prints "
         "$\\mathbb{R}$ (amsmath, amssymb).",
         AMS + "\\newcommand{\\norm}[1]{\\left\\lVert #1 \\right\\rVert}\n\\newcommand{\\R}{\\mathbb{R}}",
         {"mode": "preamble", "body": "For $x \\in \\R^d$, $\\norm{x}_2 \\le \\norm{x}_1$.",
          "require": ["\\newcommand"], "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\grad{f} prints $\\nabla f$, and \\grad[\\theta]{L} prints "
         "$\\nabla_{\\theta} L$ (an optional first argument whose default is empty): "
         "\\newcommand{\\grad}[2][]{\\nabla_{#1} #2}.",
         AMS + "\\newcommand{\\grad}[2][]{\\nabla_{#1} #2}",
         {"mode": "preamble", "body": "$\\grad{f}$ and $\\grad[\\theta]{L}$",
          "require": ["\\newcommand"], "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\abs{x} prints $\\left| x \\right|$ (one argument).",
         AMS + "\\newcommand{\\abs}[1]{\\left| #1 \\right|}",
         {"mode": "preamble", "body": "The error is $\\abs{y - \\hat{y}}$.", "require": ["\\newcommand"],
          "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles: \\inner{x}{y} prints $\\langle x, y \\rangle$ (two arguments).",
         AMS + "\\newcommand{\\inner}[2]{\\langle #1, #2 \\rangle}",
         {"mode": "preamble", "body": "Attention scores are $\\inner{q}{k}$ scaled by $1/\\sqrt{d}$.",
          "require": ["\\newcommand"], "starter": PRE_STARTER}),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def new_environment(rng):
    items = [
        ("Write the preamble so the body compiles as shown: a note environment that starts a paragraph with a bold "
         "\u201cNote.\u201d (\\newenvironment{note}{\\par\\noindent\\textbf{Note.}\\ }{\\par}); amsmath and amssymb.",
         AMS + "\\newenvironment{note}{\\par\\noindent\\textbf{Note.}\\ }{\\par}",
         {"mode": "preamble", "body": "Some text.\n\\begin{note}\nCall zero\\_grad() every step.\n\\end{note}\nMore text.",
          "require": ["\\newenvironment"], "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles as shown: a warning environment that prints a bold red "
         "\u201cWarning:\u201d before its text, in its own paragraph "
         "(\\newenvironment{warning}{\\par\\noindent\\textbf{\\textcolor{red}{Warning:}}\\ }{\\par}); xcolor too.",
         "\\documentclass{article}\n\\usepackage{xcolor}\n"
         "\\newenvironment{warning}{\\par\\noindent\\textbf{\\textcolor{red}{Warning:}}\\ }{\\par}",
         {"mode": "preamble", "body": "\\begin{warning}\nThe learning rate is too high.\n\\end{warning}",
          "require": ["\\newenvironment"], "starter": "\\documentclass{article}\n"}),
        ("Write the preamble so the body compiles as shown: a tip environment whose whole text is italic, after an "
         "upright bold \u201cTip.\u201d (\\newenvironment{tip}{\\par\\noindent\\textbf{Tip.}\\ \\itshape}{\\par}).",
         AMS + "\\newenvironment{tip}{\\par\\noindent\\textbf{Tip.}\\ \\itshape}{\\par}",
         {"mode": "preamble", "body": "\\begin{tip}\nPlot the learning rate before training.\n\\end{tip}\nBack to normal.",
          "require": ["\\newenvironment"], "starter": PRE_STARTER}),
        ("Write the preamble so the body compiles as shown: an example environment with a title argument, printing "
         "a bold \u201cExample (title).\u201d first: \\newenvironment{example}[1]{\\par\\noindent\\textbf{Example (#1).}"
         "\\ }{\\par}.",
         AMS + "\\newenvironment{example}[1]{\\par\\noindent\\textbf{Example (#1).}\\ }{\\par}",
         {"mode": "preamble", "body": "\\begin{example}{XOR}\nNo linear model separates it.\n\\end{example}",
          "require": ["\\newenvironment"], "starter": PRE_STARTER}),
    ]
    return drill(rng, items)


@template(M, "hard")
def fix_macros(rng):
    items = [
        ("This doesn't compile: \\R is defined twice (Command \\R already defined). Remove the second definition.",
         AMS + "\\newcommand{\\R}{\\mathbb{R}}",
         {"mode": "preamble", "body": "$x \\in \\R^n$",
          "starter": AMS + "\\newcommand{\\R}{\\mathbb{R}}\n\\newcommand{\\R}{\\mathbb{R}}"}),
        ("This doesn't compile: amsthm already defines \\proof (for the proof environment), so \\newcommand{\\proof} "
         "fails. Rename the macro to \\pf, which the body uses.",
         AMS + "\\usepackage{amsthm}\n\\newcommand{\\pf}{\\textit{Proof.}}",
         {"mode": "preamble", "body": "\\pf{} Trivial.",
          "starter": AMS + "\\usepackage{amsthm}\n\\newcommand{\\proof}{\\textit{Proof.}}"}),
        ("This doesn't compile: the theorem environment is used but never defined (Environment theorem undefined). "
         "Define it in the preamble (\\newtheorem{theorem}{Theorem}).",
         THM + "\\newtheorem{theorem}{Theorem}",
         {"mode": "preamble", "body": f"\\begin{{theorem}}\n{GD}\n\\end{{theorem}}", "starter": THM.rstrip("\n")}),
        ("This doesn't compile: \\mathbb in the \\R macro needs amssymb, which isn't loaded. Load it.",
         AMS + "\\newcommand{\\R}{\\mathbb{R}}",
         {"mode": "preamble", "body": "$x \\in \\R^n$",
          "starter": "\\documentclass{article}\n\\usepackage{amsmath}\n\\newcommand{\\R}{\\mathbb{R}}"}),
    ]
    return drill(rng, items)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def guide_example(rng):
    sol = (f"\\section{{Optimization}}\n\\begin{{definition}}[Convexity]\n{CONVEX}\n\\end{{definition}}\n"
           f"\\begin{{theorem}}\\label{{thm:gd}}\n{GD}\n\\end{{theorem}}\n\\begin{{proof}}\nCombine smoothness with "
           f"convexity and telescope the per-step decrease.\n\\end{{proof}}\n\\begin{{note}} Macros keep notation "
           f"consistent: $\\vect{{x}} \\in \\R^d$, $\\norm{{\\vect{{x}}}}_2$, $\\E[X]$. \\end{{note}}")
    return L("**The guide's page.** Recreate the target with the preamble's environments and macros: a section, a "
             "titled definition, a labelled theorem, its proof, and a note.", sol, preamble=MACROS,
             require=["env:definition", "env:theorem", "env:proof", "\\vect"])


@template(M, "challenge")
def lemma_chain(rng):
    sec = rng.choice(SECTIONS)
    i = rng.randrange(len(STATEMENTS))
    sol = (f"\\section{{{sec}}}\n\\begin{{lemma}}\\label{{lem:descent}}\nEach step decreases $f$ by at least "
           "$\\frac{\\eta}{2} \\norm{\\nabla f(x_t)}^2$.\n\\end{lemma}\n\\begin{proof}\nApply $L$-smoothness with "
           "$\\eta \\le 1/L$.\n\\end{proof}\n\\begin{theorem}\\label{thm:rate}\n" + STATEMENTS[i] + "\n\\end{theorem}\n"
           "\\begin{proof}\nSum Lemma~\\ref{lem:descent} over $t$ steps.\n\\end{proof}")
    return L("**A lemma, then the theorem.** Recreate the target: Lemma 1.1 and Theorem 1.2 share a counter numbered "
             "within the section; the theorem's proof refers to the lemma with \\ref.", sol, preamble=MACROS,
             require=["env:lemma", "env:theorem", "\\ref"])

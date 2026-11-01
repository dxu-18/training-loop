"""11.18 Troubleshooting: every problem starts from code that doesn't compile (or compiles wrong) and names the
symptom the way pdflatex reports it. Undefined control sequence, Missing $ inserted, Missing \\begin{document},
File not found, Option clash, ?? and undefined citations, environment mismatches, unbalanced braces, stray & and %."""

from ._common import AMS, BASE, DATASETS, L, MODELS, drill, fixed, template

M = "11.18"
BIB = """@article{kingma2014adam,
  title   = {Adam: A Method for Stochastic Optimization},
  author  = {Kingma, Diederik P. and Ba, Jimmy},
  journal = {arXiv preprint arXiv:1412.6980},
  year    = {2014}
}
"""
WORDS = ["important", "essential", "the default", "deprecated", "required"]


def _doc(pre, body):
    return f"{pre.rstrip()}\n\\begin{{document}}\n{body.strip()}\n\\end{{document}}"


def fix(prompt, good, bad, **kw):
    """A fix-it problem: the starter is the broken code; the answer compiles to what the fixed code does."""
    return (prompt, good, {"starter": bad, **kw})


# ================================================================= lite

@template(M, "lite")
def undefined_cs(rng):
    m = rng.choice(MODELS)
    items = [
        fix("! Undefined control sequence. Fix the typo in the command.",
            f"The \\textbf{{loss}} of {m} went down.", f"The \\textbff{{loss}} of {m} went down."),
        fix("! Undefined control sequence. Fix the typo in the command.",
            "Set $\\alpha = 0.1$ and $\\beta = 0.9$.", "Set $\\aplha = 0.1$ and $\\beta = 0.9$."),
        fix("! Undefined control sequence. Fix the typo in the command.",
            f"\\section{{Results}}\n{m} \\emph{{works}}.", f"\\section{{Results}}\n{m} \\enph{{works}}."),
        fix("! Undefined control sequence. Fix the typo in the command.",
            f"\\section{{{m}}}\nIt converged.", f"\\sectoin{{{m}}}\nIt converged."),
        fix("! Undefined control sequence. Fix the typo in the command.",
            "Half the data: $\\frac{1}{2}$.", "Half the data: $\\farc{1}{2}$."),
        fix("! Undefined control sequence. Fix the typo in the command.",
            f"We call \\texttt{{model.fit()}} on {m}.", f"We call \\textt{{model.fit()}} on {m}."),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "lite")
def missing_dollar(rng):
    v = rng.choice(["x_i", "w_j", "h_t", "z_k"])
    items = [
        fix(f"! Missing $ inserted. The variable {v} is math: put it in $...$.",
            f"Each input ${v}$ is normalized.", f"Each input {v} is normalized."),
        fix("! Missing $ inserted. The underscore in learning_rate is text here: escape it (\\_).",
            "The learning\\_rate is 0.001.", "The learning_rate is 0.001."),
        fix("! Missing $ inserted. \\alpha only works in math: put it in $...$.",
            "The step size $\\alpha$ decays.", "The step size \\alpha decays."),
        fix("! Missing $ inserted. The file name train_model.py is text: escape the underscore (\\_).",
            "Run train\\_model.py first.", "Run train_model.py first."),
        fix("! Missing $ inserted. x^2 is math: put it in $...$.",
            "The cost grows like $x^2$.", "The cost grows like x^2."),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= easy

@template(M, "easy")
def specials(rng):
    n = rng.randint(60, 99)
    d = rng.choice(DATASETS)
    items = [
        fix("The rest of the line vanished from the PDF: % starts a comment. Escape it (\\%).",
            f"Accuracy rose to {n}\\% on {d}.", f"Accuracy rose to {n}% on {d}."),
        fix("! Misplaced alignment tab character &. Escape the ampersand (\\&).",
            f"Research \\& development on {d}.", f"Research & development on {d}."),
        fix("! You can't use `macro parameter character #' in horizontal mode. Escape it (\\#).",
            f"We trained run \\#{n % 10 + 1} for longer.", f"We trained run #{n % 10 + 1} for longer."),
    ]
    return drill(rng, items, preamble=BASE)


@template(M, "easy")
def braces(rng):
    w = rng.choice(WORDS)
    items = [
        fix(f"! File ended while scanning use of \\textbf. A brace is never closed: close it after \u201c{w}\u201d.",
            f"This is \\textbf{{{w}}}. The rest is not.", f"This is \\textbf{{{w}. The rest is not."),
        fix("! Too many }'s. Remove the extra brace.", f"This is \\emph{{{w}}}.", f"This is \\emph{{{w}}}}}."),
    ]
    return drill(rng, items, preamble=BASE)


# ================================================================= medium

@template(M, "medium")
def environments(rng):
    a, b = rng.sample(["Data", "Model", "Loss", "Optimizer", "Metrics"], 2)
    items = [
        fix("! LaTeX Error: \\begin{itemize} on input line 1 ended by \\end{enumerate}. Make the ends match the "
            "itemize.", f"\\begin{{itemize}}\n  \\item {a}\n  \\item {b}\n\\end{{itemize}}",
            f"\\begin{{itemize}}\n  \\item {a}\n  \\item {b}\n\\end{{enumerate}}"),
        fix("! LaTeX Error: \\begin{enumerate} on input line 1 ended by \\end{itemize}. Make the ends match the "
            "enumerate.", f"\\begin{{enumerate}}\n  \\item {a}\n  \\item {b}\n\\end{{enumerate}}",
            f"\\begin{{enumerate}}\n  \\item {a}\n  \\item {b}\n\\end{{itemize}}"),
        fix("! LaTeX Error: Environment equaton undefined. Fix the environment's name (both ends).",
            "\\begin{equation}\n  y = Wx + b\n\\end{equation}", "\\begin{equaton}\n  y = Wx + b\n\\end{equaton}"),
        fix("! LaTeX Error: Environment align undefined. You can't change the preamble here: write the two lines "
            "as two separate equation environments instead.",
            "\\begin{equation}\n  a = b + c\n\\end{equation}\n\\begin{equation}\n  d = e + f\n\\end{equation}",
            "\\begin{align}\n  a &= b + c \\\\\n  d &= e + f\n\\end{align}", preamble=BASE),
    ]
    return drill(rng, items, preamble=AMS)


@template(M, "medium")
def preamble_errors(rng):
    body = "$x \\in \\mathbb{R}$"
    items = [
        fix("! LaTeX Error: File `amsmaths.sty' not found. Fix the package name.",
            _doc("\\documentclass{article}\n\\usepackage{amsmath, amssymb}", body),
            _doc("\\documentclass{article}\n\\usepackage{amsmaths, amssymb}", body), mode="document"),
        fix("! LaTeX Error: Missing \\begin{document}. Stray text sits in the preamble: delete it.",
            _doc("\\documentclass{article}\n\\usepackage{amsmath, amssymb}", body),
            _doc("\\documentclass{article}\n\\usepackage{amsmath, amssymb}\nx", body), mode="document"),
        fix("! Undefined control sequence: \\mathbb comes from amssymb, which isn't loaded. Load it.",
            _doc("\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}", body),
            _doc("\\documentclass{article}\n\\usepackage{amsmath}", body), mode="document"),
        fix("! LaTeX Error: Option clash for package geometry. It's loaded twice with different options: keep only "
            "the margin=1in one.",
            _doc("\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{amsmath, amssymb}", body),
            _doc("\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{amsmath, amssymb}\n"
                 "\\usepackage[margin=2cm]{geometry}", body), mode="document", check=["page"]),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def references(rng):
    name = rng.choice(["loss", "acc", "arch", "attn", "lr"])
    bad = rng.choice([name[:-1], name + "s", name.upper()])
    fig = ("\\begin{figure}[htbp]\n  \\centering\n  \\includegraphics[width=0.5\\linewidth]{example-image}\n"
           f"  \\caption{{Loss.}}\\label{{fig:{name}}}\n\\end{{figure}}\n")
    items = [
        fix(f"The PDF shows ?? (LaTeX Warning: Reference `fig:{bad}' undefined). Fix the \\ref.",
            fig + f"See Figure~\\ref{{fig:{name}}}.", fig + f"See Figure~\\ref{{fig:{bad}}}.",
            preamble=BASE + "\\usepackage{graphicx}\n"),
        fix("The PDF shows [?] (Citation `kingma2014adam' undefined): BibTeX never ran because there's no "
            "\\bibliographystyle. Add one (plain).",
            "We use Adam~\\cite{kingma2014adam}.\n\\bibliographystyle{plain}\n\\bibliography{references}",
            "We use Adam~\\cite{kingma2014adam}.\n\\bibliography{references}", files={"references.bib": BIB}),
        fix("The PDF shows [?] (Citation `kingma2015adam' undefined): the key doesn't match references.bib "
            "(kingma2014adam). Fix the \\cite.",
            "We use Adam~\\cite{kingma2014adam}.\n\\bibliographystyle{plain}\n\\bibliography{references}",
            "We use Adam~\\cite{kingma2015adam}.\n\\bibliographystyle{plain}\n\\bibliography{references}",
            files={"references.bib": BIB}),
    ]
    return drill(rng, items, preamble=BASE)


@template(M, "hard")
def math_errors(rng):
    i, j = rng.choice([("i", "j"), ("t", "k"), ("m", "n")])
    items = [
        fix(f"! Double subscript. x_{i}_{j} is ambiguous: write x_{{{i}{j}}}.", f"$x_{{{i}{j}}} = 0$", f"$x_{i}_{j} = 0$"),
        fix("! Double superscript. x^2^3 is ambiguous: write the power of a power as x^{2^3}.",
            "$x^{2^3}$", "$x^2^3$"),
        fix("! Missing \\right. inserted. Every \\left needs a \\right: close the parenthesis with \\right).",
            "\\[ \\left( \\frac{a}{b} \\right) \\]", "\\[ \\left( \\frac{a}{b} ) \\]"),
        fix("! Display math should end with $$. The display opens with $$ but closes with a single $. Write it the "
            "LaTeX way, with \\[ ... \\].", "\\[ E = mc^2 \\]", "$$ E = mc^2 $"),
    ]
    return drill(rng, items, preamble=AMS)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def broken_paper(rng):
    good_pre = "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n\\usepackage{graphicx}"
    good = _doc(good_pre, "\\section{Method}\\label{sec:method}\nWe minimize the loss $L(\\theta)$ with a "
                          "learning\\_rate of $10^{-3}$, which gave 91\\% accuracy.\n\\begin{equation}\\label{eq:sgd}\n"
                          "  \\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)\n\\end{equation}\n"
                          "Equation~\\eqref{eq:sgd} is applied in Section~\\ref{sec:method}.")
    bad = _doc(good_pre.replace("graphicx", "graphix"),
               "\\section{Method}\\label{sec:method}\nWe minimize the loss L(\\theta) with a learning_rate of "
               "$10^{-3}$, which gave 91% accuracy.\n\\begin{equation}\\label{eq:sgd}\n"
               "  \\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)\n\\end{equaton}\n"
               "Equation~\\eqref{eq:sdg} is applied in Section~\\ref{sec:method}.")
    return L("**Fix the paper.** This document has six mistakes: a misspelled package, math outside $...$, an "
             "unescaped underscore, a % that eats the end of a line, a misspelled \\end, and a reference to a label "
             "that doesn't exist. Fix them all (Compile shows the log).", good, mode="document", starter=bad)


@template(M, "challenge")
@fixed
def broken_slides(rng):
    good = ("\\documentclass{beamer}\n\\usetheme{Madrid}\n\\title{Debugging \\LaTeX}\n\\author{Ada Lovelace}\n"
            "\\begin{document}\n\\frame{\\titlepage}\n\\begin{frame}[fragile]{Code}\n\\begin{verbatim}\n"
            "loss.backward()\n\\end{verbatim}\n\\end{frame}\n\\begin{frame}{List}\n  \\begin{itemize}\n"
            "    \\item<1-> One\n    \\item<2-> Two\n  \\end{itemize}\n\\end{frame}\n\\end{document}")
    bad = good.replace("[fragile]", "").replace("\\end{itemize}", "\\end{enumerate}").replace(
        "\\usetheme{Madrid}", "\\usetheme{Madrid}\nNotes for me")
    return L("**Fix the slides.** Three mistakes: stray text in the preamble, verbatim in a frame without [fragile], "
             "and a list closed by the wrong \\end. Fix them all.", good, mode="document", starter=bad)

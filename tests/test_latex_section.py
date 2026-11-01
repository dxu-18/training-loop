"""COMP_GUIDE §11 LaTeX (generator/sections/s11_latex) on latex-lite (generator/latexlite): every catalog item of
every template passes with its reference and fails empty, every command the guide teaches is practised, errors read
the way pdflatex prints them, the page renders like LaTeX's, and wrong answers get specific messages."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generator import core, grader  # noqa: E402
from generator.instructions import instructions  # noqa: E402
from generator.latexlite import compile_doc  # noqa: E402
from generator.latexlite.catalog import CATALOG, MODULE_OF  # noqa: E402
from generator.latexlite.render import render  # noqa: E402
from latex_templates import problems  # noqa: E402

PROBLEMS = [(f"{tpl.module}-{tpl.name}-{item}-{seed}", p, err) for tpl, item, seed, p, err in problems(None, 1)]
AMS = "\\documentclass{article}\n\\usepackage{amsmath, amssymb}\n"


def body(src, preamble=AMS):
    return compile_doc({"main.tex": src}, preamble=preamble)


def doc(src):
    return compile_doc({"main.tex": src})


def first_error(r):
    return r["errors"][0]["msg"] if r["errors"] else None


def problem(module, name, seed=1):
    tpl = next(t for t in core.templates("latex", module) if t.name == name)
    return core.build(tpl, random.Random(seed).randrange(2**31)).to_dict()


# ---------------------------------------------------------------- the templates

@pytest.mark.parametrize("pid,p,err", PROBLEMS, ids=[x[0] for x in PROBLEMS])
def test_reference_passes_and_empty_fails(pid, p, err):
    assert err is None, err
    res = grader.grade(p, p["solution"])
    assert res["correct"], res["message"] or res["error"]
    assert not grader.grade(p, p["latex"].get("starter", ""))["correct"]


def test_every_catalog_entry_is_practised():
    used = set()
    for _, p, err in PROBLEMS:
        used |= set(p["latex"]["uses"]) | {a for r in p["require"] for a in r.split("|")}
    assert not [k for k in CATALOG if k not in used]


def test_all_eighteen_modules_are_live():
    assert sorted({t.module for t in core.templates("latex")}) == [f"11.{i:02d}" for i in range(1, 19)]
    assert set(MODULE_OF.values()) <= {f"11.{i:02d}" for i in range(1, 19)}


def test_instructions_come_from_the_catalog():
    """Each problem lists what it requires, then what its module introduces; every name has a description."""
    for _, p, _ in PROBLEMS[::7]:
        items = instructions(p, with_keys=True)
        assert all(x["key"] in CATALOG and x["desc"] for x in items), items
        need = [a for r in p["require"] for a in r.split("|") if a in CATALOG]
        assert all(any(x["key"] == k for x in items) for k in need[:4]), (p["prompt"][:60], need, items)


def test_targets_are_compiled_pages():
    for _, p, _ in PROBLEMS[::11]:
        if p["latex"].get("target"):
            assert 'class="tex-doc' in p["latex"]["target"]


# ---------------------------------------------------------------- errors, as pdflatex prints them

@pytest.mark.parametrize("src,msg", [
    ("The \\textbff{loss}.", "Undefined control sequence."),
    ("Each x_i is positive.", "Missing $ inserted."),
    ("The step size \\alpha decays.", "Missing $ inserted."),
    ("This is \\textbf{important. The rest.", "File ended while scanning use of \\textbf."),
    ("The \\emph{model}} converged.", "Too many }'s."),
    ("Research & development.", "Misplaced alignment tab character &."),
    ("Run #3.", "You can't use `macro parameter character #' in horizontal mode."),
    ("\\begin{itemize}\n\\item a\n\\end{enumerate}", "LaTeX Error: \\begin{itemize} on input line 1 ended by "
                                                    "\\end{enumerate}."),
    ("\\begin{equaton}\nx\n\\end{equaton}", "LaTeX Error: Environment equaton undefined."),
    ("$x_i_j$", "Double subscript."),
    ("$x^2^3$", "Double superscript."),
    ("\\[ \\left( \\frac{a}{b} ) \\]", "Missing \\right. inserted."),
    ("$$ E = mc^2 $", "Display math should end with $$."),
])
def test_body_errors(src, msg):
    assert first_error(body(src)) == msg


@pytest.mark.parametrize("src,msg", [
    ("\\documentclass{article}\n\\usepackage{amsmaths}\n\\begin{document}\nx\n\\end{document}",
     "LaTeX Error: File `amsmaths.sty' not found."),
    ("\\documentclass{article}\nstray\n\\begin{document}\nx\n\\end{document}", "LaTeX Error: Missing \\begin{document}."),
    ("\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage[margin=2cm]{geometry}\n"
     "\\begin{document}\nx\n\\end{document}", "LaTeX Error: Option clash for package geometry."),
    ("\\documentclass{article}\n\\begin{document}\n$\\mathbb{R}$\n\\end{document}", "Undefined control sequence."),
    ("\\documentclass{beamer}\n\\begin{document}\n\\begin{frame}{C}\n\\begin{verbatim}\nx\n\\end{verbatim}\n"
     "\\end{frame}\n\\end{document}", None),
])
def test_document_errors(src, msg):
    r = doc(src)
    if msg is None:
        assert r["errors"], "verbatim in a frame without [fragile] must fail"
    else:
        assert first_error(r) == msg


def test_reference_warnings():
    r = body("\\section{A}\\label{sec:a}\nSee~\\ref{sec:b}.")
    assert not r["errors"] and "Reference `sec:b' on page 1 undefined" in r["warnings"][0]["msg"]


# ---------------------------------------------------------------- the page

def test_header_and_footer_print_in_full():
    r = doc("\\documentclass{article}\n\\usepackage{fancyhdr}\n\\pagestyle{fancy}\n\\fancyhf{}\n\\lhead{COMP Guide}\n"
            "\\rhead{\\thepage}\n\\cfoot{Page \\thepage}\n\\begin{document}\nHi.\n\\end{document}")
    html = render(r)
    assert not r["errors"] and "COMP Guide" in html and "Page 1" in html


def test_multirow_cells_are_left_out_below():
    r = doc("\\documentclass{article}\n\\usepackage{multirow}\n\\begin{document}\n\\begin{tabular}{|l|l|}\n"
            "\\multirow{2}{*}{M} & a \\\\\n & b \\\\\n\\end{tabular}\n\\end{document}")
    html = render(r)
    rows = html.split("<tr>")[1:]
    assert 'rowspan="2"' in rows[0] and rows[1].count("<td") == 1


def test_twocolumn_and_include():
    r = compile_doc({"main.tex": "\\documentclass[twocolumn]{article}\n\\begin{document}\n\\include{a}\n"
                                 "\\include{b}\nAfter.\n\\end{document}", "a.tex": "A.", "b.tex": "B."})
    html = render(r)
    assert "tex-twocolumn" in html and html.count('class="tex-page"') == 3      # a, b, then After.


def test_beamer_overlays_make_slides():
    r = doc("\\documentclass{beamer}\n\\usetheme{Madrid}\n\\title{T}\n\\author{A}\n\\begin{document}\n"
            "\\frame{\\titlepage}\n\\begin{frame}{L}\n\\begin{itemize}\n\\item<1-> a\n\\item<2-> b\n\\end{itemize}\n"
            "\\end{frame}\n\\end{document}")
    html = render(r)
    assert html.count('class="tex-slide') == 3 and "1 / 2" in html and "2 / 2" in html


def test_algorithm_keywords_keep_their_spaces():
    r = body("\\begin{algorithmic}[1]\n\\For{$i = 1$ \\textbf{to} $n$}\n\\State \\Return $x$ \\Comment{done}\n"
             "\\EndFor\n\\end{algorithmic}", AMS + "\\usepackage{algpseudocode}\n")
    html = render(r)
    assert not r["errors"] and "tex-alcode" in html and "tex-algcomment" in html


# ---------------------------------------------------------------- grading messages

def test_near_misses_get_specific_messages():
    p = problem("11.16", "mlp")
    msg = grader.grade(p, p["solution"].replace("fill=red!10", "fill=green!10"))["message"]
    assert "should be filled" in msg
    p = problem("11.13", "guide_sgd")
    msg = grader.grade(p, p["solution"].replace("\\Comment{gradient step}", "\\Comment{step}"))["message"]
    assert msg.startswith("Line 5 of the algorithm") and "gradient step" in msg
    p = problem("11.17", "guide_deck")
    msg = grader.grade(p, p["solution"].replace("\\item<2-> Broadcasting", "\\item Broadcasting"))["message"]
    assert "overlay <2->" in msg and "every slide" in msg
    p = problem("11.15", "handout")
    msg = grader.grade(p, p["solution"].replace("\\lhead{COMP Guide}", "\\chead{COMP Guide}"))["message"]
    assert "header" in msg


@pytest.mark.parametrize("a,b", [
    ("Some \\textbf{bold} text.", "Some {\\bfseries bold} text."),
    ("An \\emph{idea}.", "An \\textit{idea}."),
    ("$\\frac{a}{b}$", "$\\frac a b$"),
    ("$x_{i}^{2}$", "$x^2_i$"),
])
def test_equivalent_markup_passes(a, b):
    from generator.latexlite.check import compare, norm_doc
    assert compare(norm_doc(body(a), set()), norm_doc(body(b), set())) is None


def test_compile_errors_keep_the_problem_open():
    p = problem("11.18", "undefined_cs")
    res = grader.grade(p, p["latex"]["starter"])
    assert not res["correct"]

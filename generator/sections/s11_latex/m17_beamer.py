"""11.17 Beamer slides: frames and titles, \\titlepage, \\tableofcontents, themes, overlays (\\pause, \\item<2->,
\\only, \\alert), columns, blocks, fragile frames for code."""

from ._common import AUTHORS, DATASETS, L, TITLES, drill, fixed, template

M = "11.17"
PRE = "\\documentclass{beamer}\n\\usetheme{Madrid}\n"
TOPICS = {"Tensors": ["Creation", "Broadcasting", "Autograd", "Indexing"],
          "Training": ["Forward", "Loss", "Backward", "Step"],
          "Data": ["Download", "Split", "Augment", "Batch"],
          "Evaluation": ["Accuracy", "Calibration", "Robustness", "Latency"]}


def _deck(body, title="Muscle-Memory ML", author="Ada Lovelace", theme="Madrid", extra=""):
    return (f"\\documentclass{{beamer}}\n\\usetheme{{{theme}}}\n{extra}\\title{{{title}}}\n\\author{{{author}}}\n"
            f"\\begin{{document}}\n{body.strip()}\n\\end{{document}}")


# ================================================================= lite

@template(M, "lite")
def first_frame(rng):
    t = rng.choice(["Tensors", "Autograd", "Optimizers", "Data loaders"])
    items = [
        (f"Write the body: one frame titled {t} with a bulleted list of two items, Creation and Indexing.",
         f"\\begin{{frame}}{{{t}}}\n  \\begin{{itemize}}\n    \\item Creation\n    \\item Indexing\n  \\end{{itemize}}\n"
         f"\\end{{frame}}", {"require": ["env:frame"]}),
        (f"Write the body: one frame whose title is set with \\frametitle{{{t}}}, saying \u201cEverything is a "
         f"tensor.\u201d", f"\\begin{{frame}}\n  \\frametitle{{{t}}}\n  Everything is a tensor.\n\\end{{frame}}",
         {"require": ["\\frametitle"]}),
    ]
    return drill(rng, items, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n")


@template(M, "lite")
def title_slide(rng):
    title, author = rng.choice(TITLES), rng.choice(AUTHORS)
    items = [
        (f"Write the whole deck: a beamer document with the Madrid theme, the title \u201c{title}\u201d by {author}, "
         f"and one slide: the title page (\\frame{{\\titlepage}}).",
         _deck("\\frame{\\titlepage}", title, author), {"require": ["\\titlepage", "\\usetheme"]}),
        (f"Write the whole deck: beamer with the default theme (no \\usetheme), title \u201c{title}\u201d by {author}, "
         f"and a title-page frame (\\begin{{frame}} \\titlepage \\end{{frame}}).",
         f"\\documentclass{{beamer}}\n\\title{{{title}}}\n\\author{{{author}}}\n\\begin{{document}}\n"
         f"\\begin{{frame}}\n  \\titlepage\n\\end{{frame}}\n\\end{{document}}", {"require": ["\\titlepage"]}),
    ]
    return drill(rng, items, mode="document")


# ================================================================= easy

@template(M, "easy")
def reveal(rng):
    title = rng.choice(list(TOPICS))
    a, b, c = TOPICS[title][:3]
    items = [
        (f"Write the body: a frame titled {title} whose three bullets appear one at a time (\\item<1->, \\item<2->, "
         f"\\item<3->): {a}, {b}, {c}. (It's three slides in the PDF.)",
         f"\\begin{{frame}}{{{title}}}\n  \\begin{{itemize}}\n    \\item<1-> {a}\n    \\item<2-> {b}\n"
         f"    \\item<3-> {c}\n  \\end{{itemize}}\n\\end{{frame}}", {"require": ["env:frame"]}),
        (f"Write the body: a frame titled {title} saying \u201cFirst, {a.lower()}.\u201d then \\pause, then \u201cThen, "
         f"{b.lower()}.\u201d (two slides).", f"\\begin{{frame}}{{{title}}}\n  First, {a.lower()}.\n  \\pause\n"
                                          f"  Then, {b.lower()}.\n\\end{{frame}}", {"require": ["\\pause"]}),
    ]
    return drill(rng, items, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n")


@template(M, "easy")
def outline_slide(rng):
    secs = rng.sample(["Tensors", "Autograd", "Training", "Evaluation"], 2)
    body = ("\\frame{\\titlepage}\n\\begin{frame}{Outline}\n  \\tableofcontents\n\\end{frame}\n" +
            "".join(f"\\section{{{s}}}\n\\begin{{frame}}{{{s}}}\n  About {s.lower()}.\n\\end{{frame}}\n" for s in secs))
    return L(f"Write the body: a title slide, an Outline slide with \\tableofcontents, then sections {secs[0]} and "
             f"{secs[1]}, each with one frame of that title saying \u201cAbout {secs[0].lower()}.\u201d / \u201cAbout "
             f"{secs[1].lower()}.\u201d", body, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n",
             require=["\\tableofcontents", "\\titlepage"])


# ================================================================= medium

@template(M, "medium")
def columns_frame(rng):
    claim = rng.choice(["Warmup halves the loss.", "Dropout fixes the overfitting.", "Mixed precision is 2x faster.",
                        "Label smoothing helps calibration."])
    tip = rng.choice(["Tip", "Hint", "Remember"])
    sol = ("\\begin{frame}{Two columns}\n  \\begin{columns}\n    \\column{0.5\\textwidth} Text on the left.\n"
           f"    \\column{{0.5\\textwidth}} \\begin{{block}}{{{tip}}} Reveal step by step with "
           "\\texttt{\\textbackslash pause}. "
           "\\end{block}\n  \\end{columns}\n\\end{frame}")
    items = [
        (f"Write the body: the guide's Two columns frame: text on the left, a {tip} block on the right (columns, "
         "\\column{0.5\\textwidth}, block).", sol, {"require": ["env:columns", "\\column", "env:block"]}),
        (f"Write the body: a frame titled Takeaway with a plot and its takeaway side by side (the guide's drill): "
         "example-image at \\linewidth in a 0.6\\textwidth column, and in a 0.4\\textwidth column an alertblock "
         f"titled Result saying \u201c{claim}\u201d",
         "\\begin{frame}{Takeaway}\n  \\begin{columns}\n    \\column{0.6\\textwidth}\n"
         "      \\includegraphics[width=\\linewidth]{example-image}\n    \\column{0.4\\textwidth}\n"
         f"      \\begin{{alertblock}}{{Result}}\n        {claim}\n      \\end{{alertblock}}\n"
         "  \\end{columns}\n\\end{frame}", {"require": ["env:columns", "env:alertblock"]}),
    ]
    return drill(rng, items, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n")


@template(M, "medium")
def code_frame(rng):
    code = rng.choice(["x = torch.randn(3, requires_grad=True)", "loss.backward()\noptimizer.step()"])
    sol = f"\\begin{{frame}}[fragile]{{Code needs the fragile option}}\n\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}\n" \
          f"\\end{{frame}}"
    items = [
        ("Write the body: a frame titled \u201cCode needs the fragile option\u201d showing the code in verbatim. "
         "(Verbatim inside a frame needs [fragile].)", sol, {"require": ["env:verbatim"]}),
        ("This doesn't compile: verbatim inside a frame needs the fragile option. Fix it.", sol,
         {"starter": sol.replace("[fragile]", "")}),
    ]
    return drill(rng, items, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n")


# ================================================================= hard

@template(M, "hard")
def overlays(rng):
    n = rng.randint(70, 97)
    d = rng.choice(DATASETS)
    title = rng.choice(list(TOPICS))
    steps = TOPICS[title]
    items = [
        (f"Write the body: a frame titled Results with \u201cAccuracy rose to {n}%.\u201d (escape the %) where the "
         f"number is highlighted with \\alert, then on slide 2 only (\\only<2>) the line \u201cBut only on {d}.\u201d",
         f"\\begin{{frame}}{{Results}}\n  Accuracy rose to \\alert{{{n}\\%}}.\n\n  \\only<2>{{But only on {d}.}}\n"
         f"\\end{{frame}}", {"require": ["\\alert", "\\only"]}),
        (f"Write the body: a frame titled {title} with an enumerate list whose items each appear on the next slide "
         f"([<+->] on the list): {', '.join(steps)}.",
         f"\\begin{{frame}}{{{title}}}\n  \\begin{{enumerate}}[<+->]\n" + "".join(f"    \\item {x}\n" for x in steps) +
         "  \\end{enumerate}\n\\end{frame}", {"require": ["env:enumerate"]}),
    ]
    return drill(rng, items, preamble=PRE + "\\title{Muscle-Memory ML}\n\\author{Ada Lovelace}\n")


@template(M, "hard")
def theme_choice(rng):
    theme = rng.choice(["Madrid", "Berlin", "metropolis", "CambridgeUS", "Warsaw"])
    title, author = rng.choice(TITLES), rng.choice(AUTHORS)
    body = "\\frame{\\titlepage}\n\\begin{frame}{Why}\n  Because it works.\n\\end{frame}"
    sol = _deck(body, title, author, theme)
    return L(f"Write the whole deck: the {theme} theme, title \u201c{title}\u201d by {author}, a title slide, and a "
             f"frame titled Why saying \u201cBecause it works.\u201d", sol, mode="document", require=["\\usetheme"])


# ================================================================= challenge

@template(M, "challenge")
@fixed
def guide_deck(rng):
    body = """\\frame{\\titlepage}
\\begin{frame}{Outline}
  \\tableofcontents
\\end{frame}
\\section{Tensors}
\\begin{frame}{Tensors}
  \\begin{itemize}
    \\item<1-> Creation
    \\item<2-> Broadcasting
    \\item<3-> Autograd
  \\end{itemize}
\\end{frame}
\\begin{frame}[fragile]{Code needs the fragile option}
\\begin{verbatim}
x = torch.randn(3, requires_grad=True)
\\end{verbatim}
\\end{frame}
\\begin{frame}{Two columns}
  \\begin{columns}
    \\column{0.5\\textwidth} Text on the left. \\pause
    \\column{0.5\\textwidth} \\begin{block}{Tip} Reveal step by step with \\texttt{\\textbackslash pause}. \\end{block}
  \\end{columns}
\\end{frame}"""
    return L("**The guide's talk** (a 5-frame talk: the drill). Write the whole deck from memory: the Madrid theme, "
             "\u201cMuscle-Memory ML\u201d by Ada Lovelace, title and outline slides, a Tensors section with "
             "revealed bullets, a fragile code frame, and the two-column frame with a \\pause.", _deck(body),
             mode="document", require=["\\titlepage", "\\tableofcontents", "env:columns", "\\pause"])


@template(M, "challenge")
@fixed
def results_talk(rng):
    body = """\\frame{\\titlepage}
\\section{Results}
\\begin{frame}{Main result}
  \\begin{columns}
    \\column{0.55\\textwidth}
      \\includegraphics[width=\\linewidth]{example-image}
    \\column{0.45\\textwidth}
      \\begin{itemize}
        \\item<1-> Loss falls \\alert{40\\%} faster
        \\item<2-> Same final accuracy
      \\end{itemize}
  \\end{columns}
\\end{frame}
\\begin{frame}{The update}
  \\begin{block}{Adam}
    $\\theta_{t+1} = \\theta_t - \\eta \\, \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$
  \\end{block}
\\end{frame}"""
    return L("**A results talk.** Recreate the target deck: the Madrid theme, the title slide, a two-column frame "
             "with a plot and revealed bullets (one alerted), and a block with Adam's update.",
             _deck(body, "Warmup Considered Helpful", "Grace Hopper"), mode="document",
             require=["env:columns", "\\alert", "env:block"])

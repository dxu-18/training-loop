"""11.02 Structure: sections, subsections, \\paragraph, starred sections, \\appendix, chapters (report), \\input and
\\include, \\newpage / \\clearpage."""

from ._common import BASE, TOPICS, L, drill, template

M = "11.02"
SECS = ["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion", "Conclusion"]


# ================================================================= lite

@template(M, "lite")
def sections(rng):
    a, b, c = rng.sample(SECS, 3)
    items = [
        (f"Write a section {a} containing a subsection Data, which contains a subsubsection Cleaning. Put the word "
         f"Text. under each heading.",
         f"\\section{{{a}}}\nText.\n\\subsection{{Data}}\nText.\n\\subsubsection{{Cleaning}}\nText.",
         {"require": ["\\subsubsection"]}),
        (f"Write three sections, {a}, {b} and {c}, each with the text To do.",
         f"\\section{{{a}}}\nTo do.\n\\section{{{b}}}\nTo do.\n\\section{{{c}}}\nTo do.", {"require": ["\\section"]}),
        (f"Write a section {a}, then two subsections Setup and Results inside it, each with the text To do.",
         f"\\section{{{a}}}\n\\subsection{{Setup}}\nTo do.\n\\subsection{{Results}}\nTo do.",
         {"require": ["\\subsection"]}),
    ]
    return drill(rng, items)


@template(M, "lite")
def starred(rng):
    a = rng.choice(SECS)
    who = rng.choice(["the reviewers", "our GPU cluster", "coffee", "the open-source community"])
    word, note = rng.choice([("Note.", "Results are preliminary."), ("Caveat.", "We tuned on the test set once."),
                             ("Setup.", "All runs use three seeds."), ("Remark.", "The baseline is untuned.")])
    items = [
        (f"Write a numbered section {a} (text: To do.), then an UNNUMBERED section Acknowledgments thanking {who}: "
         f"\u201cWe thank {who}.\u201d (a starred section).",
         f"\\section{{{a}}}\nTo do.\n\\section*{{Acknowledgments}}\nWe thank {who}.", {}),
        (f"Write a run-in heading with \\paragraph: the bold word {word} followed on the same line by \u201c{note}\u201d",
         f"\\paragraph{{{word}}} {note}", {"require": ["\\paragraph"]}),
    ]
    return drill(rng, items)


# ================================================================= easy

@template(M, "easy")
def appendix(rng):
    a = rng.choice(SECS[:4])
    x, y = rng.sample(["Extra results", "Proofs", "Hyperparameters", "Dataset details", "More plots"], 2)
    items = [
        (f"Write a section {a} (text: Main text.), then an appendix with two sections, {x} and {y} (each with To do.): "
         f"after \\appendix they're numbered A and B.",
         f"\\section{{{a}}}\nMain text.\n\\appendix\n\\section{{{x}}}\nTo do.\n\\section{{{y}}}\nTo do.",
         {"require": ["\\appendix"]}),
        (f"Write two numbered sections, {a} and Conclusion (text: Done.), then an appendix section {x} (To do.).",
         f"\\section{{{a}}}\nDone.\n\\section{{Conclusion}}\nDone.\n\\appendix\n\\section{{{x}}}\nTo do.",
         {"require": ["\\appendix"]}),
    ]
    return drill(rng, items)


@template(M, "easy")
def chapters(rng):
    a, b = rng.sample(["Background", "Methods", "Experiments", "Results", "Conclusions"], 2)
    pre = "\\documentclass{report}\n"
    items = [
        (f"This is a report (thesis-style). Write a chapter {a} with a section Overview inside it (text: To do.).",
         f"\\chapter{{{a}}}\n\\section{{Overview}}\nTo do.", {"require": ["\\chapter"]}),
        (f"Write two chapters, {a} and {b}, each with the text To do. (In a report, sections are numbered within "
         f"chapters: 1.1, 2.1 ...)", f"\\chapter{{{a}}}\nTo do.\n\\chapter{{{b}}}\nTo do.", {"require": ["\\chapter"]}),
    ]
    return drill(rng, items, preamble=pre)


# ================================================================= medium

@template(M, "medium")
def input_files(rng):
    a, b = rng.sample(TOPICS, 2)
    files = {"sections/intro.tex": f"\\section{{Introduction}}\nThis paper is about {a}.\n",
             "sections/method.tex": f"\\section{{Method}}\nWe use {b}.\n"}
    items = [
        ("The sections live in their own files: sections/intro.tex and sections/method.tex (shown). Pull both in, in "
         "that order, with \\input.", "\\input{sections/intro}\n\\input{sections/method}", {"require": ["\\input"]}),
        ("Pull in sections/method.tex FIRST, then sections/intro.tex, with \\input (the numbers follow the order).",
         "\\input{sections/method}\n\\input{sections/intro}", {"require": ["\\input"]}),
        ("Pull in sections/intro.tex and sections/method.tex with \\include instead: each included file starts on a "
         "new page (\\include does a \\clearpage before and after it).",
         "\\include{sections/intro}\n\\include{sections/method}", {"require": ["\\include"]}),
    ]
    return drill(rng, items, files=files)


@template(M, "medium")
def pages(rng):
    a, b = rng.sample(SECS, 2)
    items = [
        (f"Write a section {a} (text: Page one.), then start a new page (\\newpage) with a section {b} (text: Page "
         f"two.).", f"\\section{{{a}}}\nPage one.\n\\newpage\n\\section{{{b}}}\nPage two.", {"require": ["\\newpage"]}),
        (f"Write a section {a} (text: First.), then \\clearpage (a new page, after any pending figures), then a "
         f"section {b} (text: Second.).", f"\\section{{{a}}}\nFirst.\n\\clearpage\n\\section{{{b}}}\nSecond.",
         {"require": ["\\clearpage"]}),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def outline(rng):
    a, b = rng.sample(SECS[1:6], 2)
    sol = (f"\\section{{Introduction}}\nWhy.\n\\section{{{a}}}\n\\subsection{{Setup}}\nHow.\n\\paragraph{{Data.}} "
           f"Small.\n\\section{{{b}}}\nWhat.\n\\section*{{Acknowledgments}}\nThanks.\n\\appendix\n"
           f"\\section{{Extra results}}\nMore.")
    return L("Recreate the outline shown: numbered sections and a subsection, a run-in \\paragraph heading, an "
             "unnumbered acknowledgments section and an appendix section.", sol,
             require=["\\paragraph", "\\appendix"])


@template(M, "hard")
def short_titles(rng):
    a = rng.choice(TOPICS)
    long_t = f"A Very Long Section Title About {a.title()} That Overflows"
    sol = f"\\tableofcontents\n\\section[{a.title()}]{{{long_t}}}\nText."
    return L(f"Write a table of contents, then a section whose heading is \u201c{long_t}\u201d but whose entry in the "
             f"contents is just \u201c{a.title()}\u201d (\\section[short]{{long}}). Text: Text.", sol,
             require=["\\tableofcontents"])


# ================================================================= challenge

@template(M, "challenge")
def thesis(rng):
    a, b = rng.sample(["Background", "Methods", "Experiments", "Results"], 2)
    files = {"chapters/one.tex": f"\\chapter{{{a}}}\n\\section{{Overview}}\nOne.\n",
             "chapters/two.tex": f"\\chapter{{{b}}}\n\\section{{Overview}}\nTwo.\n"}
    sol = "\\tableofcontents\n\\input{chapters/one}\n\\input{chapters/two}\n\\appendix\n\\chapter{Data}\nTables."
    return L("**Thesis skeleton.** This is a report. Write the body: the table of contents, the chapters in "
             "chapters/one.tex and chapters/two.tex (in that order, with \\input), then an appendix chapter Data "
             "(text: Tables.).", sol, preamble="\\documentclass{report}\n", files=files,
             require=["\\input", "\\appendix", "\\tableofcontents"])


@template(M, "challenge")
def numbering_puzzle(rng):
    k = rng.randint(2, 5)
    sub = rng.choice(["Detail", "Proof", "Example", "Notes"])
    sol = "".join(f"\\section{{Part {i}}}\n" for i in range(1, k + 1)) + "\\section*{Interlude}\nRest.\n" + \
        f"\\section{{Part {k + 1}}}\n\\subsection{{{sub}}}\nEnd."
    return L(f"Recreate the target exactly: its numbers come from LaTeX (look at which heading has no number).", sol,
             require=["\\section"])

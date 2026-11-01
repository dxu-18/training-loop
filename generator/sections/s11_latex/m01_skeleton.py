"""11.01 Document skeleton & compiling: \\documentclass and its options, \\usepackage, the title block, abstract,
table of contents, and the errors a broken skeleton gives."""

from ...core import pick
from ._common import AUTHORS, TITLES, TOPICS, L, drill, template

M = "11.01"


def _doc(pre: str, body: str) -> str:
    return pre.rstrip("\n") + "\n\\begin{document}\n" + body.strip("\n") + "\n\\end{document}\n"


# ================================================================= lite

@template(M, "lite")
def skeleton(rng):
    word = rng.choice(["world", "LaTeX", "paper", "notes"])
    text = rng.choice([f"Hello, {word}!", f"My first {word}.", "Training loss goes down."])
    items = [
        (f"Write the smallest complete LaTeX document (article class) whose page says: {text}",
         _doc("\\documentclass{article}", text), {"require": ["env:document"]}),
        ("Write a complete document whose page says: Hello, \\LaTeX! (with the LaTeX logo, `\\LaTeX`).",
         _doc("\\documentclass{article}", "Hello, \\LaTeX!"), {"require": ["\\LaTeX"]}),
        (f"Write a complete article in 11pt type (a class option) whose page says: {text}",
         _doc("\\documentclass[11pt]{article}", text), {}),
        (f"Write a complete article on A4 paper in 12pt type whose page says: {text}",
         _doc("\\documentclass[12pt, a4paper]{article}", text), {}),
    ]
    return drill(rng, items, mode="document")


@template(M, "lite")
def title_block(rng):
    title = rng.choice(TITLES)
    a, b = rng.sample(AUTHORS, 2)
    date = rng.choice(["October 2026", "Spring 2026", "Draft of 1 May 2026"])
    items = [
        (f"Write a complete article with the title \u201c{title}\u201d by {a}, dated {date}, and print the title "
         f"block with \\maketitle (nothing else on the page).",
         _doc(f"\\documentclass{{article}}\n\\title{{{title}}}\n\\author{{{a}}}\n\\date{{{date}}}", "\\maketitle"),
         {"require": ["\\maketitle"]}),
        (f"Same, with two authors, {a} and {b} (separated with \\and), and no date at all (\\date{{}}).",
         _doc(f"\\documentclass{{article}}\n\\title{{{title}}}\n\\author{{{a} \\and {b}}}\n\\date{{}}", "\\maketitle"),
         {"require": ["\\and"]}),
        (f"A complete article titled \u201c{title}\u201d by {a}, dated today (\\today), with its title block.",
         _doc(f"\\documentclass{{article}}\n\\title{{{title}}}\n\\author{{{a}}}\n\\date{{\\today}}", "\\maketitle"),
         {"require": ["\\today"]}),
    ]
    return drill(rng, items, mode="document")


# ================================================================= easy

@template(M, "easy")
def abstract_and_section(rng):
    title = rng.choice(TITLES)
    a = rng.choice(AUTHORS)
    topic = rng.choice(TOPICS)
    pre = f"\\documentclass{{article}}\n\\title{{{title}}}\n\\author{{{a}}}\n\\date{{}}\n"
    abstract = f"We study {topic} in small models and report what helps."
    body = (f"\\maketitle\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\\section{{Introduction}}\n"
            f"This note is about {topic}.")
    items = [
        (f"The preamble is given. Write the body: the title block, an abstract saying \u201c{abstract}\u201d, then a "
         f"section Introduction with the text \u201cThis note is about {topic}.\u201d", body,
         {"require": ["env:abstract", "\\section"]}),
        (f"Write the body: the title block, then two sections, Introduction and Method, each with one sentence: "
         f"\u201cWe study {topic}.\u201d and \u201cWe train small models.\u201d",
         f"\\maketitle\n\\section{{Introduction}}\nWe study {topic}.\n\\section{{Method}}\nWe train small models.",
         {"require": ["\\maketitle", "\\section"]}),
    ]
    return drill(rng, items, preamble=pre)


@template(M, "easy")
def fix_skeleton(rng):
    text = rng.choice(["Results are in Section 2.", "Hello again.", "The model converges."])
    good = _doc("\\documentclass{article}", text)
    items = [
        ("This document doesn't compile (Missing \\begin{document}). Fix it.", good,
         {"starter": f"\\documentclass{{article}}\n{text}\n\\end{{document}}\n"}),
        ("This document doesn't compile: the class name is misspelled. Fix it.", good,
         {"starter": good.replace("{article}", "{artcle}")}),
        ("This document ends with an emergency stop (*** job aborted, no legal \\end found). Fix it.", good,
         {"starter": good.replace("\\end{document}\n", "")}),
        ("Compile this: a stray word in the preamble breaks it (Missing \\begin{document}). Fix it.", good,
         {"starter": good.replace("\\begin{document}", "Draft\n\\begin{document}")}),
    ]
    return drill(rng, items, mode="document")


# ================================================================= medium

@template(M, "medium")
def packages(rng):
    n = rng.choice(["n", "d", "k"])
    body = (f"Let $x \\in \\mathbb{{R}}^{n}$.\n\\begin{{align}}\n  y &= Wx + b \\\\\n  z &= \\operatorname{{relu}}(y)\n"
            f"\\end{{align}}")
    items = [
        ("This document uses \\mathbb and the align environment, so it doesn't compile as it is. Add the packages it "
         "needs (in the preamble).", _doc("\\documentclass{article}\n\\usepackage{amsmath, amssymb}", body),
         {"starter": _doc("\\documentclass{article}", body), "require": ["pkg:amsmath", "pkg:amssymb"]}),
        ("Load geometry with 1-inch margins (`\\usepackage[margin=1in]{geometry}`), and keep the text.",
         _doc("\\documentclass{article}\n\\usepackage[margin=1in]{geometry}", "Wide margins waste paper."),
         {"starter": _doc("\\documentclass{article}", "Wide margins waste paper."), "check": ["page"],
          "require": ["pkg:geometry"]}),
    ]
    return drill(rng, items, mode="document")


@template(M, "medium")
def contents(rng):
    secs = rng.sample(["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion",
                       "Conclusion"], 3)
    body = "\\tableofcontents\n" + "".join(f"\\section{{{s}}}\nText.\n" for s in secs)
    items = [
        (f"Write the body: a table of contents, then three sections ({', '.join(secs)}), each containing the word "
         f"Text.", body, {"require": ["\\tableofcontents"]}),
        (f"Write the body: a table of contents, then the section {secs[0]} with a subsection Setup inside it (each "
         f"containing Text.).", f"\\tableofcontents\n\\section{{{secs[0]}}}\nText.\n\\subsection{{Setup}}\nText.",
         {"require": ["\\tableofcontents", "\\subsection"]}),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def class_options(rng):
    text = rng.choice(["Two columns save space in conference papers.", "Landscape pages fit wide tables."])
    items = [
        ("Make this a two-column article in 10pt type on letter paper, as many conference templates are "
         "(class options).", _doc("\\documentclass[twocolumn]{article}", text),
         {"starter": _doc("\\documentclass{article}", text)}),
        ("Turn this into a landscape A4 article (class options).", _doc("\\documentclass[a4paper, landscape]{article}",
                                                                         text),
         {"starter": _doc("\\documentclass{article}", text)}),
        ("This should be a report (it has a chapter). Change the class so it compiles.",
         _doc("\\documentclass{report}", "\\chapter{Background}\n" + text),
         {"starter": _doc("\\documentclass{article}", "\\chapter{Background}\n" + text)}),
    ]
    return drill(rng, items, mode="document")


@template(M, "hard")
def title_page_details(rng):
    title = rng.choice(TITLES)
    a, b, c = rng.sample(AUTHORS, 3)
    pre = (f"\\documentclass[11pt]{{article}}\n\\title{{{title}}}\n\\author{{{a} \\and {b} \\and {c}}}\n"
           f"\\date{{}}\n")
    items = [
        (f"Write the complete document: 11pt article, \u201c{title}\u201d by {a}, {b} and {c} (three authors), no "
         f"date, the title block, and an abstract \u201cShort and sweet.\u201d",
         _doc(pre, "\\maketitle\n\\begin{abstract}\nShort and sweet.\n\\end{abstract}"),
         {"require": ["\\and", "env:abstract"]}),
        (f"Write the complete document: \u201c{title}\u201d by {a} (with a footnote after the name: "
         f"\\thanks{{Work done while procrastinating.}}), no date, and the title block.",
         _doc(f"\\documentclass{{article}}\n\\title{{{title}}}\n\\author{{{a}\\thanks{{Work done while "
              f"procrastinating.}}}}\n\\date{{}}", "\\maketitle"), {}),
    ]
    return drill(rng, items, mode="document")


# ================================================================= challenge

@template(M, "challenge")
def paper_skeleton(rng):
    title = rng.choice(TITLES)
    a, b = rng.sample(AUTHORS, 2)
    topic = rng.choice(TOPICS)
    sol = _doc(f"\\documentclass[11pt, a4paper]{{article}}\n\\usepackage[margin=1in]{{geometry}}\n"
               f"\\usepackage{{amsmath, amssymb}}\n\\usepackage{{graphicx}}\n\\usepackage{{booktabs}}\n"
               f"\\usepackage{{hyperref}}\n\\title{{{title}}}\n\\author{{{a} \\and {b}}}\n\\date{{\\today}}",
               f"\\maketitle\n\\begin{{abstract}}\nWe revisit {topic}.\n\\end{{abstract}}\n\\tableofcontents\n"
               f"\\section{{Introduction}}\\label{{sec:intro}}\nHello, \\LaTeX!\n\\section{{Method}}\nSee "
               f"Section~\\ref{{sec:intro}}.")
    return L(f"**The paper skeleton.** Write the guide's skeleton from memory: an 11pt A4 article with 1in margins "
             f"(geometry), loading amsmath, amssymb, graphicx, booktabs and hyperref; title \u201c{title}\u201d by {a} "
             f"and {b}, dated today. The body: the title block, an abstract \u201cWe revisit {topic}.\u201d, the table "
             f"of contents, a section Introduction (labelled sec:intro) saying Hello, \\LaTeX!, and a section Method "
             f"saying \u201cSee Section 1.\u201d with the number from \\ref.",
             sol, mode="document", check=["page"],
             require=["pkg:amsmath", "pkg:amssymb", "pkg:graphicx", "pkg:booktabs", "pkg:hyperref", "\\ref",
                      "\\tableofcontents"], show_target=True)


@template(M, "challenge")
def recreate(rng):
    title = rng.choice(TITLES)
    a = rng.choice(AUTHORS)
    secs = rng.sample(["Motivation", "Setup", "Findings", "Limitations", "Outlook"], 3)
    sol = _doc(f"\\documentclass[12pt]{{article}}\n\\title{{{title}}}\n\\author{{{a}}}\n\\date{{}}",
               "\\maketitle\n\\tableofcontents\n" + "".join(f"\\section{{{s}}}\nTo do.\n" for s in secs))
    return L("**Recreate the target.** Write the complete document that compiles to the page shown (look closely at "
             "the type size: it's 12pt).", sol, mode="document", require=["\\tableofcontents"])

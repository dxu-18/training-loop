"""11.15 Page layout, colors & boxes: fancyhdr headers and footers, geometry margins, setspace, \\parskip, xcolor
(\\textcolor, \\colorbox, \\color, \\definecolor, mixes), tcolorbox, minipages side by side, multicols, page styles."""

from ._common import AUTHORS, BASE, L, TITLES, drill, template

M = "11.15"
COLOR = BASE + "\\usepackage{xcolor}\n"
BOX = COLOR + "\\definecolor{brand}{HTML}{1F77B4}\n\\usepackage{tcolorbox}\n"
GFX = BASE + "\\usepackage{graphicx}\n"
TEXT = ("Gradients accumulate across backward passes, so clear them before every step. Forgetting to do so is one "
        "of the most common bugs in a hand-written training loop.")


def _doc(pre, body):
    return f"{pre.rstrip()}\n\\begin{{document}}\n{body.strip()}\n\\end{{document}}"


# ================================================================= lite

@template(M, "lite")
def colors(rng):
    c = rng.choice(["red", "blue", "teal", "orange"])
    items = [
        (f"Write \u201cThe loss is NaN.\u201d with NaN in {c} (\\textcolor).",
         f"The loss is \\textcolor{{{c}}}{{NaN}}.", {"require": ["\\textcolor"]}),
        ("Write \u201cCheck the highlighted line.\u201d with \u201chighlighted\u201d on a yellow background (\\colorbox).",
         "Check the \\colorbox{yellow}{highlighted} line.", {"require": ["\\colorbox"]}),
        (f"Write \u201cOne span is {c}, the rest isn't.\u201d where \u201cOne span is {c},\u201d is {c} using "
         f"{{\\color{{{c}}} ...}} (a switch inside a group).",
         f"{{\\color{{{c}}} One span is {c},}} the rest isn't.", {"require": ["\\color"]}),
        ("Write \u201cA lighter shade.\u201d in 50% blue mixed with white (blue!50).",
         "\\textcolor{blue!50}{A lighter shade.}", {"require": ["\\textcolor"]}),
    ]
    return drill(rng, items, preamble=COLOR)


@template(M, "lite")
def define_color(rng):
    hexes = rng.choice([("brand", "1F77B4"), ("accent", "D62728"), ("good", "2CA02C"), ("warn", "FF7F0E"),
                        ("muted", "7F7F7F"), ("deep", "9467BD")])
    name, hx = hexes
    items = [
        (f"Write the preamble so the body compiles: xcolor, and a color named {name} with the HTML code {hx} "
         f"(\\definecolor).", f"\\documentclass{{article}}\n\\usepackage{{xcolor}}\n\\definecolor{{{name}}}{{HTML}}{{{hx}}}",
         {"mode": "preamble", "body": f"\\textcolor{{{name}}}{{Colored text}} in the {name} color.",
          "require": ["\\definecolor"], "starter": "\\documentclass{article}\n\\usepackage{xcolor}\n"}),
    ]
    return drill(rng, items)


# ================================================================= easy

@template(M, "easy")
def key_idea_box(rng):
    t = rng.choice(["Key idea", "Remember", "Tip"])
    items = [
        (f"Make the guide's box: a tcolorbox with colback=blue!5, colframe=brand and title={t}, containing "
         f"\u201cGradients accumulate: call \\texttt{{zero\\_grad()}} every step.\u201d",
         f"\\begin{{tcolorbox}}[colback=blue!5, colframe=brand, title={t}]\nGradients accumulate: call "
         f"\\texttt{{zero\\_grad()}} every step.\n\\end{{tcolorbox}}", {"require": ["env:tcolorbox"]}),
        ("Make a plain tcolorbox (no options) containing \u201cAlways set the random seed.\u201d",
         "\\begin{tcolorbox}\nAlways set the random seed.\n\\end{tcolorbox}", {"require": ["env:tcolorbox"]}),
    ]
    return drill(rng, items, preamble=BOX)


@template(M, "easy")
def minipages(rng):
    w = rng.choice(["0.45", "0.48", "0.4"])
    left, right = rng.choice([("Left column text.", "Right column text."), ("Before training.", "After training."),
                              ("The baseline.", "Our method."), ("Inputs.", "Outputs.")])
    items = [
        (f"Put two minipages side by side, each {w}\\linewidth, separated by \\hfill: \u201c{left}\u201d and "
         f"\u201c{right}\u201d",
         f"\\begin{{minipage}}{{{w}\\linewidth}} {left} \\end{{minipage}}\\hfill\n"
         f"\\begin{{minipage}}{{{w}\\linewidth}} {right} \\end{{minipage}}", {"require": ["env:minipage"]}),
        ("Lay out text and a figure side by side (the guide's drill): a 0.55\\linewidth minipage with the text shown, "
         "\\hfill, then a 0.4\\linewidth minipage with example-image at \\linewidth.",
         f"\\begin{{minipage}}{{0.55\\linewidth}}\n{TEXT}\n\\end{{minipage}}\\hfill\n\\begin{{minipage}}{{0.4\\linewidth}}\n"
         f"  \\includegraphics[width=\\linewidth]{{example-image}}\n\\end{{minipage}}",
         {"require": ["env:minipage", "\\includegraphics"], "preamble": GFX}),
    ]
    return drill(rng, items, preamble=BASE)


# ================================================================= medium

@template(M, "medium")
def header_footer(rng):
    name = rng.choice(AUTHORS)
    title = rng.choice(TITLES)
    body = "\\section{Introduction}\nHello.\n\\newpage\n\\section{Method}\nMore."
    items = [
        (f"Write the preamble for a header with your name ({name}) at the left and the page number at the right, and "
         f"an empty footer (the guide's drill: fancyhdr, \\pagestyle{{fancy}}, \\fancyhf{{}}, \\lhead, \\rhead).",
         f"\\documentclass{{article}}\n\\usepackage{{fancyhdr}}\n\\pagestyle{{fancy}}\n\\fancyhf{{}}\n\\lhead{{{name}}}\n"
         f"\\rhead{{\\thepage}}", {"require": ["\\lhead", "\\rhead", "\\fancyhf"]}),
        (f"Write the preamble for a header with the title \u201c{title}\u201d centered, and the page number centered "
         f"in the footer (fancyhdr: \\fancyhf{{}}, \\chead, \\cfoot).",
         f"\\documentclass{{article}}\n\\usepackage{{fancyhdr}}\n\\pagestyle{{fancy}}\n\\fancyhf{{}}\n\\chead{{{title}}}\n"
         f"\\cfoot{{\\thepage}}", {"require": ["\\cfoot", "\\fancyhf"]}),
    ]
    return drill(rng, items, mode="preamble", body=body, check=["page"], starter="\\documentclass{article}\n")


@template(M, "medium")
def line_spacing(rng):
    body = f"{TEXT}\n\n{TEXT}"
    m = rng.choice(["2.5cm", "2cm", "1in", "3cm"])
    items = [
        (f"Write the preamble for 1.5 line spacing (setspace's \\onehalfspacing) and {m} margins (geometry).",
         f"\\documentclass{{article}}\n\\usepackage[margin={m}]{{geometry}}\n\\usepackage{{setspace}}\n\\onehalfspacing",
         {"require": ["\\onehalfspacing", "pkg:geometry"]}),
        ("Write the preamble for unindented paragraphs separated by 4pt of space (\\setlength for \\parindent and "
         "\\parskip), with 1in margins (geometry).",
         "\\documentclass{article}\n\\usepackage[margin=1in]{geometry}\n\\setlength{\\parindent}{0pt}\n"
         "\\setlength{\\parskip}{4pt}", {"require": ["\\setlength"]}),
        ("Write the preamble for double spacing (setspace's \\doublespacing) on A4 paper (the a4paper class option).",
         "\\documentclass[a4paper]{article}\n\\usepackage{setspace}\n\\doublespacing",
         {"require": ["\\doublespacing"]}),
    ]
    return drill(rng, items, mode="preamble", body=body, check=["page"], starter="\\documentclass{article}\n")


# ================================================================= hard

@template(M, "hard")
def columns(rng):
    items = [
        ("Write the whole document: a two-column article (the twocolumn class option) with a section Introduction "
         "and the text shown.", _doc("\\documentclass[twocolumn]{article}",
                                     f"\\section{{Introduction}}\n{TEXT}"), {"mode": "document"}),
        ("Write the body: the text shown in two columns with multicol (\\begin{multicols}{2}), under a section "
         "Notes.", f"\\section{{Notes}}\n\\begin{{multicols}}{{2}}\n{TEXT}\n\n{TEXT}\n\\end{{multicols}}",
         {"preamble": BASE + "\\usepackage{multicol}\n", "require": ["env:multicols"]}),
        ("Write the body: the text shown in THREE columns with multicol (\\begin{multicols}{3}), under a section "
         "Notes.", f"\\section{{Notes}}\n\\begin{{multicols}}{{3}}\n{TEXT}\n\n{TEXT}\n\\end{{multicols}}",
         {"preamble": BASE + "\\usepackage{multicol}\n", "require": ["env:multicols"]}),
        ("Write the whole document: a landscape article (the landscape class option) with the text shown.",
         _doc("\\documentclass[landscape]{article}", TEXT), {"mode": "document"}),
    ]
    return drill(rng, items)


@template(M, "hard")
def page_styles(rng):
    a, b = rng.choice([("Cover", "Body"), ("Title", "Introduction"), ("Abstract", "Method"), ("Summary", "Details")])
    body = f"\\thispagestyle{{empty}}\n\\section{{{a}}}\nNo page number here.\n\\newpage\n\\section{{{b}}}\nNumbered."
    items = [
        (f"Write the body: the first page has no page number (\\thispagestyle{{empty}}), then a section {a} saying "
         f"\u201cNo page number here.\u201d, a new page, and a section {b} saying \u201cNumbered.\u201d",
         body, {"require": ["\\thispagestyle"]}),
        ("Write the body: no page numbers anywhere (\\pagestyle{empty}), and the text shown.",
         f"\\pagestyle{{empty}}\n{TEXT}", {"require": ["\\pagestyle"]}),
    ]
    return drill(rng, items, preamble=BASE, check=["page"])


# ================================================================= challenge

@template(M, "challenge")
def handout(rng):
    name = rng.choice(AUTHORS)
    sol = _doc(f"\\documentclass{{article}}\n\\usepackage[margin=2cm]{{geometry}}\n\\usepackage{{fancyhdr}}\n"
               f"\\pagestyle{{fancy}}\n\\fancyhf{{}}\n\\lhead{{COMP Guide}}\n\\rhead{{{name}}}\n\\cfoot{{\\thepage}}\n"
               f"\\usepackage{{xcolor}}\n\\definecolor{{brand}}{{HTML}}{{1F77B4}}\n\\usepackage{{tcolorbox}}",
               f"\\section{{Training loops}}\n{TEXT}\n\\begin{{tcolorbox}}[colback=blue!5, colframe=brand, title=Key idea]\n"
               f"Call \\texttt{{zero\\_grad()}} every step.\n\\end{{tcolorbox}}\n\\textcolor{{brand}}{{Next:}} evaluation.")
    return L(f"**A handout.** Write the whole document for the target: 2cm margins, a header (COMP Guide left, "
             f"{name} right), the page number in the footer's center, a brand color #1F77B4, a Key idea box, and the "
             f"colored word at the end.", sol, mode="document", check=["page"],
             require=["env:tcolorbox", "\\definecolor", "\\fancyhf"])


@template(M, "challenge")
def side_by_side(rng):
    before, after, num, end = rng.choice([
        ("The loss plateaus at 2.3: the learning rate is too low.", "With warmup the loss falls to", "0.4",
         "Warmup fixed it."),
        ("Validation loss climbs after epoch 5.", "With dropout it keeps falling, to", "0.31", "Dropout fixed it."),
        ("Training takes 9 hours per run.", "Mixed precision cuts it to", "4 hours", "Use AMP."),
        ("Accuracy is stuck at 10%: the labels are shuffled.", "After the fix it reaches", "92%", "Check the data.")])
    color = rng.choice(["teal", "blue", "red"])
    sol = ("\\noindent\n\\begin{minipage}[t]{0.48\\linewidth}\n  \\textbf{Before.} " + before.replace("%", "\\%") +
           "\n\\end{minipage}\\hfill\n\\begin{minipage}[t]{0.48\\linewidth}\n  \\textbf{After.} " + after +
           f" \\textcolor{{{color}}}{{{num.replace('%', chr(92) + '%')}}}.\n\\end{{minipage}}\n"
           f"\\begin{{center}}\n  \\colorbox{{yellow}}{{{end}}}\n\\end{{center}}")
    return L("**Before and after.** Recreate the target: two top-aligned minipages ([t], 0.48\\linewidth each) with "
             "bold run-in labels, then a centered highlighted conclusion.", sol, preamble=GFX + "\\usepackage{xcolor}\n",
             require=["env:minipage", "\\colorbox"])

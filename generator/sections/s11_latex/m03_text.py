"""11.03 Text formatting, special characters & spacing: font commands and sizes, reserved characters, quotes and
dashes, ties and spaces, line and paragraph breaks, alignment, footnotes, links and verbatim."""

from ._common import BASE, DATASETS, MODELS, TOPICS, L, drill, template

M = "11.03"
HYPER = BASE + "\\usepackage{hyperref}\n"


# ================================================================= lite

@template(M, "lite")
def fonts(rng):
    m = rng.choice(MODELS)
    w = rng.choice(["fast", "robust", "simple", "stable"])
    items = [
        (f"Write: {m} is **{w}**. (bold with \\textbf)", f"{m} is \\textbf{{{w}}}.", {"require": ["\\textbf"]}),
        (f"Write: {m} is *{w}*. (italic with \\textit)", f"{m} is \\textit{{{w}}}.", {"require": ["\\textit"]}),
        (f"Write: The model is \\emph-asised: {m} is *really* {w}. (\\emph)", f"{m} is \\emph{{really}} {w}.",
         {"require": ["\\emph"]}),
        (f"Write the call `model.fit()` in monospace with \\texttt, inside: Call model.fit() to train.",
         "Call \\texttt{model.fit()} to train.", {"require": ["\\texttt"]}),
        (f"Write the name {m} in small caps (\\textsc) inside: We compare against {m}.",
         f"We compare against \\textsc{{{m}}}.", {"require": ["\\textsc"]}),
        (f"Write: The word {w} is underlined (\\underline).", f"The word \\underline{{{w}}} is underlined.",
         {"require": ["\\underline"]}),
        (f"Write the word {w} in sans serif (\\textsf): A {w} model.", f"A \\textsf{{{w}}} model.",
         {"require": ["\\textsf"]}),
        (f"Write: {m} is ***{w}*** (bold AND italic: nest \\textbf and \\textit).",
         f"{m} is \\textbf{{\\textit{{{w}}}}}.", {}),
    ]
    return drill(rng, items)


@template(M, "lite")
def reserved(rng):
    pct = rng.randint(5, 95)
    cost = rng.randint(2, 99)
    items = [
        (f"Write this sentence so it compiles and prints exactly: Accuracy rose by {pct}% for ${cost} of compute.",
         f"Accuracy rose by {pct}\\% for \\${cost} of compute.", {"require": ["\\%", "\\$"]}),
        ("Write: R&D on snake_case names costs #3 in {braces}. (each special character escaped)",
         "R\\&D on snake\\_case names costs \\#3 in \\{braces\\}.", {"require": ["\\&", "\\_", "\\#", "\\{"]}),
        ("Write: Use a backslash \\ and a tilde ~ and a caret ^ literally. (\\textbackslash{}, \\textasciitilde{}, "
         "\\textasciicircum{})", "Use a backslash \\textbackslash{} and a tilde \\textasciitilde{} and a caret "
                                  "\\textasciicircum{} literally.", {"require": ["\\textbackslash"]}),
    ]
    return drill(rng, items)


# ================================================================= easy

@template(M, "easy")
def quotes_dashes(rng):
    a, b = sorted(rng.sample(range(1, 40), 2))
    word = rng.choice(["overfitting", "grokking", "double descent", "mode collapse"])
    items = [
        (f"Write with proper typographic quotes (`` and ''): The so-called \u201c{word}\u201d is real.",
         f"The so-called ``{word}'' is real.", {}),
        (f"Write the page range with an en dash (--): See pages {a}\u2013{b}.", f"See pages {a}--{b}.", {}),
        (f"Write with an em dash (---) and an ellipsis (\\ldots): Training is slow\u2014very slow\u2026",
         "Training is slow---very slow\\ldots", {"require": ["\\ldots"]}),
        (f"Write with single quotes (` and '): the \u2018{word}\u2019 regime.", f"the `{word}' regime.", {}),
    ]
    return drill(rng, items)


@template(M, "easy")
def spacing(rng):
    n = rng.randint(1, 9)
    items = [
        (f"Write with a non-breaking space (a tie, ~) so the number never starts a line: see Figure {n}.",
         f"see Figure~{n}.", {}),
        ("Write \u201ca b\u201d with a thin space (\\,), then \u201cc d\u201d with a \\quad, then \u201ce f\u201d with a "
         "\\qquad between the letters (in one line, separated by commas).", "a\\,b, c\\quad d, e\\qquad f",
         {"require": ["\\,", "\\quad", "\\qquad"]}),
        ("Write \u201cFirst line\u201d and \u201cSecond line\u201d in one paragraph, with a line break (\\\\) between "
         "them.", "First line\\\\\nSecond line", {"require": ["\\\\"]}),
        ("Write two paragraphs: \u201cOne.\u201d and \u201cTwo.\u201d (a blank line between), the second one not "
         "indented (\\noindent).", "One.\n\n\\noindent Two.", {"require": ["\\noindent"]}),
    ]
    return drill(rng, items)


# ================================================================= medium

@template(M, "medium")
def alignment(rng):
    m = rng.choice(MODELS)
    items = [
        (f"Center the sentence \u201c{m} wins.\u201d with the center environment.",
         f"\\begin{{center}}\n{m} wins.\n\\end{{center}}", {"require": ["env:center"]}),
        (f"Right-align \u201cDraft, do not cite.\u201d with flushright.",
         "\\begin{flushright}\nDraft, do not cite.\n\\end{flushright}", {"require": ["env:flushright"]}),
        ("Write one line with \u201cLeft\u201d at the left margin and \u201cRight\u201d pushed to the right margin "
         "(\\hfill).", "\\noindent Left \\hfill Right", {"require": ["\\hfill"]}),
        ("Write \u201cAbove.\u201d, then 1em of vertical space (\\vspace{1em}), then a new paragraph \u201cBelow.\u201d",
         "Above.\n\n\\vspace{1em}\nBelow.", {"require": ["\\vspace"]}),
    ]
    return drill(rng, items)


@template(M, "medium")
def sizes(rng):
    m = rng.choice(MODELS)
    items = [
        (f"Write: {m} is {{small}} and {{LARGE}} at once \u2014 that is, the word small in \\small and the word "
         f"large in \\Large, each in a group: {m} is small and large.",
         f"{m} is {{\\small small}} and {{\\Large large}}.", {"require": ["\\small", "\\Large"]}),
        ("Write a tiny \u201cfine print\u201d and a huge \u201cHEADLINE\u201d (\\tiny, \\huge, in groups), separated by "
         "a space.", "{\\tiny fine print} {\\huge HEADLINE}", {"require": ["\\tiny", "\\huge"]}),
        ("Write the whole sentence \u201cAll of this is footnote-sized.\u201d in \\footnotesize.",
         "{\\footnotesize All of this is footnote-sized.}", {"require": ["\\footnotesize"]}),
        (f"Write \u201c{m} is large.\u201d with the word large in \\large (one step up), in a group.",
         f"{m} is {{\\large large}}.", {"require": ["\\large"]}),
    ]
    return drill(rng, items)


# ================================================================= hard

@template(M, "hard")
def footnotes_links(rng):
    lib = rng.choice([("PyTorch", "https://pytorch.org"), ("NumPy", "https://numpy.org"),
                      ("JAX", "https://jax.readthedocs.io")])
    d = rng.choice(DATASETS)
    items = [
        (f"Write: We use {d}.\u00b9 with a footnote saying \u201cDownloaded in 2026.\u201d",
         f"We use {d}.\\footnote{{Downloaded in 2026.}}", {"require": ["\\footnote"], "preamble": BASE}),
        (f"Write: The code uses {lib[0]}, with a footnote containing the URL {lib[1]} (\\url).",
         f"The code uses {lib[0]}.\\footnote{{\\url{{{lib[1]}}}}}", {"require": ["\\footnote", "\\url"]}),
        (f"Write: Read the {lib[0]} docs. \u2014 where \u201c{lib[0]} docs\u201d is a link to {lib[1]} (\\href).",
         f"Read the \\href{{{lib[1]}}}{{{lib[0]} docs}}.", {"require": ["\\href"]}),
    ]
    return drill(rng, items, preamble=HYPER)


@template(M, "hard")
def verbatim_code(rng):
    v = rng.choice(["x", "w", "grad"])
    items = [
        (f"Write inline code with \\verb: Call \\verb|{v} = np.zeros(3)| to start. (\\verb keeps every character, "
         f"underscores included.)", f"Call \\verb|{v} = np.zeros(3)| to start.", {"require": ["\\verb"]}),
        ("Write this code block with the verbatim environment:\n\n```\nfor i in range(3):\n    print(i)\n```",
         "\\begin{verbatim}\nfor i in range(3):\n    print(i)\n\\end{verbatim}", {"require": ["env:verbatim"]}),
        ("Write: The flag \\verb|--lr_decay| controls it. (an inline \\verb with dashes and an underscore)",
         "The flag \\verb|--lr_decay| controls it.", {"require": ["\\verb"]}),
    ]
    return drill(rng, items)


# ================================================================= challenge

@template(M, "challenge")
def typography(rng):
    pct = rng.randint(10, 60)
    m = rng.choice(MODELS)
    sol = (f"\\noindent\\textbf{{Summary.}} ``{m}'' cut the error by {pct}\\%---on pages 3--7\\ldots\\footnote{{See "
           f"\\url{{https://example.com/{m.lower().replace('/', '-')}}}.}}\n\n\\begin{{center}}\n\\textsc{{Draft}} "
           f"\\quad \\texttt{{v0.1}}\n\\end{{center}}")
    return L("**Typography.** Recreate the target exactly: quotes, dashes, the ellipsis, the escaped percent sign, a "
             "footnote with a URL, the bold run-in label, and the centered line with small caps and monospace.", sol,
             preamble=HYPER, require=["\\footnote", "\\url", "env:center"])


@template(M, "challenge")
def escape_everything(rng):
    pct = rng.randint(1, 99)
    sol = (f"Cost: \\${pct}, gain: {pct}\\%, \\texttt{{snake\\_case}}, R\\&D \\#{pct}, "
           f"\\{{set\\}}, \\textbackslash{{}}n.")
    return L(f"**Escape everything.** Write this line so it compiles and prints exactly (monospace where shown):\n\n"
             f"Cost: ${pct}, gain: {pct}%, `snake_case`, R&D #{pct}, {{set}}, \\n.", sol,
             require=["\\%", "\\$", "\\_", "\\&", "\\#", "\\textbackslash"])

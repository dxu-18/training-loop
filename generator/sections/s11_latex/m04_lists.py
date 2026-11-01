"""11.04 Lists: itemize, enumerate, description, nesting, and enumitem's labels and spacing."""

from ._common import BASE, MODELS, TOPICS, L, drill, template

M = "11.04"
ENUMITEM = BASE + "\\usepackage{enumitem}\n"
STEPS = ["Load data", "Build the model", "Train", "Evaluate", "Save the checkpoint", "Plot the curves",
         "Tune the learning rate"]


def _items(xs, indent="  "):
    return "".join(f"{indent}\\item {x}\n" for x in xs)


# ================================================================= lite

@template(M, "lite")
def bullets(rng):
    xs = rng.sample(STEPS, 3)
    items = [
        (f"Make a bullet list (itemize) of: {', '.join(xs)}.",
         "\\begin{itemize}\n" + _items(xs) + "\\end{itemize}", {"require": ["env:itemize"]}),
        (f"Make a numbered list (enumerate) of: {', '.join(xs)}.",
         "\\begin{enumerate}\n" + _items(xs) + "\\end{enumerate}", {"require": ["env:enumerate"]}),
    ]
    return drill(rng, items)


@template(M, "lite")
def glossary(rng):
    terms = rng.sample([("Epoch", "One pass over the training data."),
                        ("Batch", "The samples processed together in one step."),
                        ("Learning rate", "The step size of gradient descent."),
                        ("Overfitting", "Fitting noise in the training set."),
                        ("Dropout", "Randomly zeroing activations while training.")], 2)
    sol = "\\begin{description}\n" + "".join(f"  \\item[{t}] {d}\n" for t, d in terms) + "\\end{description}"
    return L("Write a glossary with the description environment: " +
             "; ".join(f"{t}: {d}" for t, d in terms), sol, require=["env:description"])


# ================================================================= easy

@template(M, "easy")
def nested(rng):
    a, b = rng.sample(STEPS, 2)
    subs = rng.sample(["shuffle", "normalize", "augment", "cache", "split"], 2)
    items = [
        (f"Make a numbered list: {a}, then {b}; under {a}, nest a bullet list of: {', '.join(subs)}.",
         f"\\begin{{enumerate}}\n  \\item {a}\n  \\begin{{itemize}}\n{_items(subs, '    ')}  \\end{{itemize}}\n"
         f"  \\item {b}\n\\end{{enumerate}}", {"require": ["env:enumerate", "env:itemize"]}),
        (f"Make a bullet list: {a}, then {b}; under {b}, nest a numbered list of: {', '.join(subs)} (it's numbered "
         f"(a), (b) at the second level).",
         f"\\begin{{itemize}}\n  \\item {a}\n  \\item {b}\n  \\begin{{enumerate}}\n{_items(subs, '    ')}"
         f"  \\end{{enumerate}}\n\\end{{itemize}}", {"require": ["env:enumerate", "env:itemize"]}),
        (f"Make a bullet list with {a} and {b}, and under {a} a second-level bullet list with {subs[0]} (its bullet "
         f"is a dash).", f"\\begin{{itemize}}\n  \\item {a}\n  \\begin{{itemize}}\n    \\item {subs[0]}\n"
                         f"  \\end{{itemize}}\n  \\item {b}\n\\end{{itemize}}", {"require": ["env:itemize"]}),
    ]
    return drill(rng, items)


@template(M, "easy")
def custom_labels(rng):
    xs = rng.sample(STEPS, 3)
    items = [
        (f"Make a bullet list of {xs[0]} and {xs[1]} where the second item's bullet is replaced by \u201c--\u201d "
         f"(\\item[--]).", f"\\begin{{itemize}}\n  \\item {xs[0]}\n  \\item[--] {xs[1]}\n\\end{{itemize}}", {}),
        (f"Make an enumerate list of {', '.join(xs)} where the middle item is labelled \u201c2b.\u201d instead of a "
         f"number (\\item[2b.]): the last item is then 2.", f"\\begin{{enumerate}}\n  \\item {xs[0]}\n"
                                                             f"  \\item[2b.] {xs[1]}\n  \\item {xs[2]}\n\\end{{enumerate}}",
         {}),
    ]
    return drill(rng, items)


# ================================================================= medium

@template(M, "medium")
def enumitem_labels(rng):
    xs = rng.sample(STEPS, 3)
    fmt = rng.choice([("(\\alph*)", "(a), (b), (c)"), ("(\\roman*)", "(i), (ii), (iii)"),
                      ("\\Alph*.", "A., B., C."), ("Step \\arabic*:", "Step 1:, Step 2:, Step 3:")])
    items = [
        (f"Make a numbered list of {', '.join(xs)} labelled {fmt[1]} (enumitem: [label={fmt[0]}]).",
         f"\\begin{{enumerate}}[label={fmt[0]}]\n" + _items(xs) + "\\end{enumerate}", {"require": ["pkg:enumitem"]}),
        (f"Make a numbered list of {', '.join(xs)} starting at 4 (enumitem's start=4).",
         "\\begin{enumerate}[start=4]\n" + _items(xs) + "\\end{enumerate}", {"require": ["pkg:enumitem"]}),
    ]
    return drill(rng, items, preamble=ENUMITEM)


@template(M, "medium")
def compact(rng):
    xs = rng.sample(STEPS, 4)
    items = [
        (f"Make a compact bullet list (no space between the items: enumitem's noitemsep) of: {', '.join(xs)}.",
         "\\begin{itemize}[noitemsep]\n" + _items(xs) + "\\end{itemize}", {"require": ["pkg:enumitem"]}),
        (f"Make a bullet list flush with the left margin (leftmargin=*) of: {', '.join(xs)}.",
         "\\begin{itemize}[leftmargin=*]\n" + _items(xs) + "\\end{itemize}", {"require": ["pkg:enumitem"]}),
    ]
    return drill(rng, items, preamble=ENUMITEM)


# ================================================================= hard

@template(M, "hard")
def fix_lists(rng):
    xs = rng.sample(STEPS, 3)
    good = "\\begin{itemize}\n" + _items(xs) + "\\end{itemize}"
    items = [
        ("This list doesn't compile (\\begin{itemize} ended by \\end{enumerate}). Fix it.", good,
         {"starter": good.replace("\\end{itemize}", "\\end{enumerate}")}),
        ("This list doesn't compile (Something's wrong--perhaps a missing \\item). Fix it.", good,
         {"starter": good.replace(f"\\item {xs[0]}", xs[0])}),
        ("The labels here need a package: [label=(\\alph*)] gives an error without it. Keep the labels and make it "
         "compile (in the body: there's no preamble here, so use \\item[(a)] ... by hand).",
         "\\begin{enumerate}\n" + "".join(f"  \\item[({c})] {x}\n" for c, x in zip("abc", xs)) + "\\end{enumerate}",
         {"starter": "\\begin{enumerate}[label=(\\alph*)]\n" + _items(xs) + "\\end{enumerate}"}),
    ]
    return drill(rng, items)


@template(M, "hard")
def deep_nesting(rng):
    a, b, c = rng.sample(STEPS, 3)
    sol = (f"\\begin{{enumerate}}\n  \\item {a}\n  \\begin{{enumerate}}\n    \\item {b}\n    \\begin{{enumerate}}\n"
           f"      \\item {c}\n    \\end{{enumerate}}\n  \\end{{enumerate}}\n\\end{{enumerate}}")
    return L(f"Three levels of enumerate: {a} contains {b}, which contains {c}. (The labels go 1., (a), i.)", sol,
             require=["env:enumerate"])


# ================================================================= challenge

@template(M, "challenge")
def checklist(rng):
    m = rng.choice(MODELS)
    t = rng.choice(TOPICS)
    sol = (f"\\begin{{enumerate}}[label=\\textbf{{Step \\arabic*.}}, leftmargin=*]\n  \\item Reproduce {m}.\n"
           f"  \\begin{{itemize}}[noitemsep]\n    \\item fix the seed\n    \\item log everything\n  \\end{{itemize}}\n"
           f"  \\item Study {t}.\n  \\begin{{description}}\n    \\item[Goal] Beat the baseline.\n"
           f"    \\item[Budget] One GPU.\n  \\end{{description}}\n\\end{{enumerate}}")
    return L("**A research checklist.** Recreate the target: an enumerate labelled Step 1., Step 2. (bold, flush left), "
             "with a compact bullet list under the first step and a description list under the second.", sol,
             preamble=ENUMITEM, require=["pkg:enumitem", "env:description"])


@template(M, "challenge")
def mixed(rng):
    xs = rng.sample(STEPS, 4)
    sol = ("\\begin{itemize}\n" + f"  \\item {xs[0]}\n  \\item[$\\star$] {xs[1]}\n" +
           "  \\begin{enumerate}\n" + f"    \\item {xs[2]}\n    \\item {xs[3]}\n" + "  \\end{enumerate}\n" +
           "\\end{itemize}")
    return L("**Recreate the target**: a bullet list whose second bullet is a star ($\\star$), with a numbered list "
             "nested under it.", sol, require=["env:itemize", "env:enumerate"])

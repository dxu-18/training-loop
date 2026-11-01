"""11.11 Tables: tabular column specs, booktabs rules, table floats with captions above, \\multicolumn, \\multirow,
\\cmidrule, p{width} columns and siunitx's S columns."""

from ._common import BASE, DATASETS, MODELS, L, drill, template

M = "11.11"
BOOK = BASE + "\\usepackage{booktabs}\n"
FULL = BASE + "\\usepackage{booktabs}\n\\usepackage{multirow}\n\\usepackage{siunitx}\n"


def _results(rng, n=3):
    ms = rng.sample(MODELS, n)
    d1, d2 = rng.sample(DATASETS, 2)
    rows = [(m, round(rng.uniform(60, 98), 1), round(rng.uniform(40, 90), 1)) for m in ms]
    return d1, d2, rows


def _rows(rows, bold_best=False):
    best1 = max(r[1] for r in rows)
    best2 = max(r[2] for r in rows)
    out = []
    for m, a, b in rows:
        sa = f"\\textbf{{{a}}}" if bold_best and a == best1 else str(a)
        sb = f"\\textbf{{{b}}}" if bold_best and b == best2 else str(b)
        out.append(f"    {m} & {sa} & {sb} \\\\\n")
    return "".join(out)


# ================================================================= lite

@template(M, "lite")
def basic_tabular(rng):
    d1, d2, rows = _results(rng, 2)
    items = [
        (f"Make a tabular with columns left, center, center (l c c): a header row Model & {d1} & {d2}, then one row "
         f"per model as shown.", f"\\begin{{tabular}}{{l c c}}\n    Model & {d1} & {d2} \\\\\n{_rows(rows)}"
                                 f"\\end{{tabular}}", {"require": ["env:tabular"]}),
        (f"Same table with vertical lines and \\hline rules everywhere ({{|l|c|c|}}, \\hline above, below the header "
         f"and at the end).", f"\\begin{{tabular}}{{|l|c|c|}}\n    \\hline\n    Model & {d1} & {d2} \\\\\n    \\hline\n"
                              f"{_rows(rows)}    \\hline\n\\end{{tabular}}", {"require": ["\\hline"]}),
    ]
    return drill(rng, items, preamble=BOOK)


@template(M, "lite")
def booktabs_rules(rng):
    d1, d2, rows = _results(rng, 3)
    sol = (f"\\begin{{tabular}}{{l c c}}\n    \\toprule\n    Model & {d1} & {d2} \\\\\n    \\midrule\n{_rows(rows)}"
           f"    \\bottomrule\n\\end{{tabular}}")
    return L("Make a booktabs table (\\toprule, \\midrule under the header, \\bottomrule; no vertical lines) with "
             f"the header Model & {d1} & {d2} and the rows shown.", sol, preamble=BOOK,
             require=["\\toprule", "\\midrule", "\\bottomrule"])


# ================================================================= easy

@template(M, "easy")
def table_float(rng):
    d1, d2, rows = _results(rng, 3)
    body = (f"  \\begin{{tabular}}{{l c c}}\n    \\toprule\n    Model & {d1} & {d2} \\\\\n    \\midrule\n"
            f"{_rows(rows, True)}    \\bottomrule\n  \\end{{tabular}}")
    sol = (f"\\begin{{table}}[t]\n  \\centering\n  \\caption{{Test accuracy (\\%) on two benchmarks.}}\n"
           f"  \\label{{tab:results}}\n{body}\n\\end{{table}}")
    items = [
        ("Put the results in a table float: centered, the caption \u201cTest accuracy (%) on two benchmarks.\u201d "
         "ABOVE the table (labelled tab:results), booktabs rules, and the best number in each column in bold.", sol,
         {"require": ["env:table", "\\caption", "\\toprule"]}),
        ("Same table float, then the sentence \u201cTable 1 lists the results.\u201d with \\ref.",
         sol + "\nTable~\\ref{tab:results} lists the results.", {"require": ["\\ref"]}),
    ]
    return drill(rng, items, preamble=BOOK)


@template(M, "easy")
def column_alignment(rng):
    rows = [(m, rng.randint(1, 900), f"{rng.uniform(0.1, 99):.2f}") for m in rng.sample(MODELS, 3)]
    sol = "\\begin{tabular}{l r r}\n    Model & Params (M) & Score \\\\\n" + \
        "".join(f"    {m} & {p} & {s} \\\\\n" for m, p, s in rows) + "\\end{tabular}"
    return L("Make a tabular with the model name left-aligned and the two number columns RIGHT-aligned (l r r): "
             "header Model & Params (M) & Score, then the rows shown.", sol, preamble=BOOK)


# ================================================================= medium

@template(M, "medium")
def multicolumn(rng):
    d1, d2, rows = _results(rng, 2)
    sol = (f"\\begin{{tabular}}{{l c c}}\n    \\toprule\n     & \\multicolumn{{2}}{{c}}{{Accuracy}} \\\\\n"
           f"    \\cmidrule(lr){{2-3}}\n    Model & {d1} & {d2} \\\\\n    \\midrule\n{_rows(rows)}    \\bottomrule\n"
           f"\\end{{tabular}}")
    items = [
        ("Add a group header: an Accuracy heading spanning the two number columns (\\multicolumn{2}{c}{Accuracy}), "
         "with a partial rule under it (\\cmidrule(lr){2-3}), above the usual header row.", sol,
         {"require": ["\\multicolumn", "\\cmidrule"]}),
    ]
    return drill(rng, items, preamble=BOOK)


@template(M, "medium")
def multirow_pcol(rng):
    term, desc = rng.choice([
        ("Dropout", "Randomly zeroes activations during training so the network can't rely on any single unit."),
        ("Weight decay", "Adds a penalty on the size of the weights, pulling them toward zero at every step."),
        ("Early stopping", "Stops training when the validation loss hasn't improved for a few epochs."),
        ("Batch norm", "Normalizes each feature over the mini-batch, then rescales it with learned parameters.")])
    items = [
        ("Make the guide's table: {|l|p{4cm}|}, a header spanning both columns (\\multicolumn{2}{|c|}{Header "
         "spanning two columns}), then a cell Merged spanning two rows (\\multirow{2}{*}{Merged}) next to row 1 and "
         "row 2, with \\hline rules.",
         "\\begin{tabular}{|l|p{4cm}|}\n  \\hline\n  \\multicolumn{2}{|c|}{Header spanning two columns} \\\\ \\hline\n"
         "  \\multirow{2}{*}{Merged} & row 1 \\\\\n                          & row 2 \\\\ \\hline\n\\end{tabular}",
         {"require": ["\\multirow", "\\multicolumn"]}),
        (f"Make a two-column table {{l p{{5cm}}}} where the second column wraps: {term} & {desc}",
         f"\\begin{{tabular}}{{l p{{5cm}}}}\n  {term} & {desc} \\\\\n\\end{{tabular}}", {}),
    ]
    return drill(rng, items, preamble=FULL)


# ================================================================= hard

@template(M, "hard")
def siunitx_cols(rng):
    runs = [(c, f"{rng.uniform(1, 40):.2f}", f"{rng.uniform(0.5, 0.99):.3f}") for c in "abc"]
    sol = ("\\begin{tabular}{l S[table-format=2.2] S[table-format=1.3]}\n  \\toprule\n  Run & {Loss} & {Acc} \\\\\n"
           "  \\midrule\n" + "".join(f"  {c} & {l} & {a} \\\\\n" for c, l, a in runs) + "  \\bottomrule\n\\end{tabular}")
    return L("Align the numbers on their decimal points with siunitx S columns (S[table-format=2.2] and "
             "S[table-format=1.3]); the header cells are text, so they go in braces: {Loss}, {Acc}.", sol,
             preamble=FULL, require=["pkg:siunitx"])


@template(M, "hard")
def fix_table(rng):
    d1, d2, rows = _results(rng, 2)
    good = (f"\\begin{{tabular}}{{l c c}}\n    \\toprule\n    Model & {d1} & {d2} \\\\\n    \\midrule\n{_rows(rows)}"
            f"    \\bottomrule\n\\end{{tabular}}")
    items = [
        ("A row here has one cell too many (Extra alignment tab has been changed to \\cr). Fix it: the extra & "
         "is in the header.", good, {"starter": good.replace(f"Model & {d1}", f"Model & & {d1}")}),
        ("This doesn't compile: \\toprule needs a package that isn't loaded. Without a preamble change, use "
         "\\hline instead of the three booktabs rules.", good.replace("\\toprule", "\\hline").replace(
             "\\midrule", "\\hline").replace("\\bottomrule", "\\hline"), {"preamble": BASE, "starter": good}),
        ("A cell holds R&D and breaks the table. Escape it (\\&).", good.replace("Model &", "R\\&D model &"),
         {"starter": good.replace("Model &", "R&D model &")}),
    ]
    return drill(rng, items, preamble=BOOK)


# ================================================================= challenge

@template(M, "challenge")
def paper_table(rng):
    ms = rng.sample(MODELS, 4)
    d1, d2 = rng.sample(DATASETS, 2)
    nums = [(m, round(rng.uniform(60, 95), 1), round(rng.uniform(60, 95), 1), rng.randint(5, 300)) for m in ms]
    b1, b2 = max(n[1] for n in nums), max(n[2] for n in nums)
    rows = "".join(f"    {m} & {f'\\textbf{{{a}}}' if a == b1 else a} & {f'\\textbf{{{b}}}' if b == b2 else b} & {p}M \\\\\n"
                   for m, a, b, p in nums)
    sol = (f"\\begin{{table}}[t]\n  \\centering\n  \\caption{{Results on {d1} and {d2}.}}\\label{{tab:main}}\n"
           f"  \\begin{{tabular}}{{l c c r}}\n    \\toprule\n     & \\multicolumn{{2}}{{c}}{{Accuracy (\\%)}} & \\\\\n"
           f"    \\cmidrule(lr){{2-3}}\n    Model & {d1} & {d2} & Params \\\\\n    \\midrule\n{rows}"
           f"    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}")
    return L("**The results table.** Recreate the target: a table float with the caption above, booktabs rules, a "
             "group header with a partial rule, best numbers in bold.", sol, preamble=FULL,
             require=["\\multicolumn", "\\cmidrule", "env:table"])


@template(M, "challenge")
def from_pandas(rng):
    rows = [(rng.choice(["a", "b", "c", "d"]) + str(i), f"{rng.uniform(0.1, 3):.3f}", f"{rng.uniform(0.5, 1):.3f}")
            for i in range(3)]
    sol = ("\\begin{tabular}{lrr}\n\\toprule\nrun & loss & acc \\\\\n\\midrule\n" +
           "".join(f"{r} & {l} & {a} \\\\\n" for r, l, a in rows) + "\\bottomrule\n\\end{tabular}")
    return L("**From pandas.** df.to_latex(index=False, float_format=\"%.3f\") printed the target table. Write it: "
             "columns {lrr}, booktabs rules, the header run & loss & acc.", sol, preamble=BOOK, require=["\\toprule"])

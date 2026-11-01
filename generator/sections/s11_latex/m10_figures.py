"""11.10 Figures: figure floats, \\includegraphics options, captions and labels (\\label after \\caption),
subfigures, references."""

from ._common import BASE, L, drill, template

M = "11.10"
GFX = BASE + "\\usepackage{graphicx}\n"
SUB = GFX + "\\usepackage{subcaption}\n"
CAPS = ["Training and validation loss.", "Accuracy over epochs.", "The learning-rate schedule.",
        "Attention weights of layer 4.", "Gradient norms per step."]


def _plot_svg(kind):
    """A small plot image for problems that include the user's own figure."""
    pts = {"loss": "10,20 60,80 110,110 160,125 210,130", "acc": "10,130 60,70 110,45 160,35 210,30"}[kind]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 150"><rect x="0.5" y="0.5" width="219" '
            f'height="149" fill="#fff" stroke="#000"/><polyline points="{pts}" fill="none" stroke="#1f77b4" '
            f'stroke-width="2"/><text x="110" y="145" font-size="9" text-anchor="middle">epoch</text></svg>')


# ================================================================= lite

@template(M, "lite")
def basic_figure(rng):
    cap = rng.choice(CAPS)
    w = rng.choice(["0.8", "0.6", "0.5"])
    items = [
        (f"Make a figure with example-image at {w}\\linewidth, centered, captioned \u201c{cap}\u201d.",
         f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width={w}\\linewidth]{{example-image}}\n"
         f"  \\caption{{{cap}}}\n\\end{{figure}}", {"require": ["env:figure", "\\includegraphics", "\\caption"]}),
        (f"Include example-image-a, 5cm wide, centered, in a figure captioned \u201c{cap}\u201d.",
         f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=5cm]{{example-image-a}}\n"
         f"  \\caption{{{cap}}}\n\\end{{figure}}", {"require": ["\\includegraphics"]}),
    ]
    return drill(rng, items, preamble=GFX)


@template(M, "lite")
def own_plot(rng):
    cap = rng.choice(CAPS)
    files = {"loss.pdf": _plot_svg("loss")}
    sol = (f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.8\\linewidth]{{loss.pdf}}\n"
           f"  \\caption{{{cap}}}\n  \\label{{fig:loss}}\n\\end{{figure}}")
    return L(f"Your Matplotlib plot was saved as loss.pdf (fig.savefig(\"loss.pdf\")). Insert it at 80% of the text "
             f"width, centered, with the caption \u201c{cap}\u201d and the label fig:loss.", sol, preamble=GFX,
             files=files, require=["\\includegraphics", "\\label"])


# ================================================================= easy

@template(M, "easy")
def reference_figure(rng):
    cap = rng.choice(CAPS)
    sol = (f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.6\\linewidth]{{example-image}}\n"
           f"  \\caption{{{cap}}}\n  \\label{{fig:main}}\n\\end{{figure}}\nAs Figure~\\ref{{fig:main}} shows, it works.")
    items = [
        (f"Make a figure (example-image at 0.6\\linewidth, caption \u201c{cap}\u201d, label fig:main), then the "
         f"sentence \u201cAs Figure 1 shows, it works.\u201d with the number from \\ref.", sol,
         {"require": ["\\ref", "\\label"]}),
        ("The label here is BEFORE the caption, so \\ref prints the section number instead of the figure's. Fix "
         "it (move \\label after \\caption).",
         "\\section{Setup}\nData.\n\\section{Results}\n" + sol,
         {"starter": "\\section{Setup}\nData.\n\\section{Results}\n" + sol.replace(f"  \\caption{{{cap}}}\n  \\label{{fig:main}}",
                                                          f"  \\label{{fig:main}}\n  \\caption{{{cap}}}"),
          "require": ["\\ref"]}),
    ]
    return drill(rng, items, preamble=GFX)


@template(M, "easy")
def graphics_options(rng):
    sc = rng.choice(["0.5", "0.3", "0.25", "0.4"])
    h = rng.choice(["3cm", "2cm", "4cm", "2.5cm"])
    ang = rng.choice(["90", "45", "180"])
    items = [
        (f"Include example-image at {sc} of its natural size (scale={sc}), centered, in a figure without a caption.",
         f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[scale={sc}]{{example-image}}\n\\end{{figure}}", {}),
        (f"Include example-image {h} tall (height={h}), centered, in a figure captioned \u201cTall.\u201d",
         f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[height={h}]{{example-image}}\n"
         f"  \\caption{{Tall.}}\n\\end{{figure}}", {}),
        (f"Include example-image at 0.5\\linewidth rotated by {ang} degrees (angle={ang}), centered, captioned "
         f"\u201cRotated.\u201d", f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.5\\linewidth, "
                                  f"angle={ang}]{{example-image}}\n  \\caption{{Rotated.}}\n\\end{{figure}}", {}),
    ]
    return drill(rng, items, preamble=GFX, require=["\\includegraphics"])


# ================================================================= medium

@template(M, "medium")
def subfigures(rng):
    a, b = rng.sample(["Loss", "Accuracy", "Gradient norm", "Learning rate"], 2)
    sol = (f"\\begin{{figure}}[t]\n  \\centering\n  \\begin{{subfigure}}{{0.48\\linewidth}}\n"
           f"    \\includegraphics[width=\\linewidth]{{example-image-a}}\n    \\caption{{{a}}}\\label{{fig:a}}\n"
           f"  \\end{{subfigure}}\\hfill\n  \\begin{{subfigure}}{{0.48\\linewidth}}\n"
           f"    \\includegraphics[width=\\linewidth]{{example-image-b}}\n    \\caption{{{b}}}\\label{{fig:b}}\n"
           f"  \\end{{subfigure}}\n  \\caption{{Two panels side by side.}}\\label{{fig:pair}}\n\\end{{figure}}")
    items = [
        (f"Put two panels side by side in one figure: example-image-a captioned \u201c{a}\u201d and example-image-b "
         f"captioned \u201c{b}\u201d, each 0.48\\linewidth wide with \\hfill between, and the figure caption "
         f"\u201cTwo panels side by side.\u201d", sol, {"require": ["env:subfigure"]}),
        (f"Same two panels, then the sentence \u201cPanel 1b shows {b.lower()}.\u201d using \\ref to fig:b.",
         sol + f"\nPanel~\\ref{{fig:b}} shows {b.lower()}.", {"require": ["env:subfigure", "\\ref"]}),
    ]
    return drill(rng, items, preamble=SUB)


@template(M, "medium")
def placement(rng):
    cap = rng.choice(CAPS)
    items = [
        (f"Put a figure exactly here with [H] (needs the float package; load it in the preamble), with "
         f"example-image at 0.5\\linewidth and the caption \u201c{cap}\u201d. (You write the preamble too: the whole "
         f"document.)",
         f"\\documentclass{{article}}\n\\usepackage{{graphicx}}\n\\usepackage{{float}}\n\\begin{{document}}\n"
         f"\\begin{{figure}}[H]\n  \\centering\n  \\includegraphics[width=0.5\\linewidth]{{example-image}}\n"
         f"  \\caption{{{cap}}}\n\\end{{figure}}\n\\end{{document}}", {"mode": "document", "require": ["pkg:float"]}),
    ]
    return drill(rng, items, preamble=GFX)


# ================================================================= hard

@template(M, "hard")
def two_figures(rng):
    c1, c2 = rng.sample(CAPS, 2)
    files = {"loss.pdf": _plot_svg("loss"), "acc.pdf": _plot_svg("acc")}
    sol = (f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.7\\linewidth]{{loss.pdf}}\n"
           f"  \\caption{{{c1}}}\\label{{fig:loss}}\n\\end{{figure}}\n\\begin{{figure}}[htbp]\n  \\centering\n"
           f"  \\includegraphics[width=0.7\\linewidth]{{acc.pdf}}\n  \\caption{{{c2}}}\\label{{fig:acc}}\n"
           f"\\end{{figure}}\nFigures~\\ref{{fig:loss}} and~\\ref{{fig:acc}} tell the same story.")
    return L(f"Two figures, loss.pdf then acc.pdf (each 0.7\\linewidth, centered), captioned \u201c{c1}\u201d and "
             f"\u201c{c2}\u201d, labelled fig:loss and fig:acc, then the sentence \u201cFigures 1 and 2 tell the same "
             f"story.\u201d with \\ref.", sol, preamble=GFX, files=files, require=["\\ref"])


@template(M, "hard")
def fix_figure(rng):
    cap = rng.choice(CAPS)
    good = (f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.6\\linewidth]{{example-image}}\n"
            f"  \\caption{{{cap}}}\n\\end{{figure}}")
    items = [
        ("This doesn't compile: the image name is wrong (File not found). The example image is example-image. Fix it.",
         good, {"starter": good.replace("{example-image}", "{example_image}")}),
        ("This doesn't compile: `size` isn't an \\includegraphics option (keyval Error). Use width.", good,
         {"starter": good.replace("width=0.6", "size=0.6")}),
        ("\\caption here is outside any float. Put the image and its caption in a figure environment.", good,
         {"starter": f"\\centering\n\\includegraphics[width=0.6\\linewidth]{{example-image}}\n\\caption{{{cap}}}"}),
    ]
    return drill(rng, items, preamble=GFX)


# ================================================================= challenge

@template(M, "challenge")
def three_panels(rng):
    names = rng.sample(["Loss", "Accuracy", "Gradient norm", "Learning rate", "Throughput"], 3)
    imgs = ["example-image-a", "example-image-b", "example-image-c"]
    panels = "\\hfill\n".join(
        f"  \\begin{{subfigure}}{{0.32\\linewidth}}\n    \\includegraphics[width=\\linewidth]{{{im}}}\n"
        f"    \\caption{{{n}}}\n  \\end{{subfigure}}" for n, im in zip(names, imgs))
    sol = f"\\begin{{figure}}[t]\n  \\centering\n{panels}\n  \\caption{{Three runs.}}\\label{{fig:runs}}\n\\end{{figure}}"
    return L("**Three panels.** Recreate the target figure: three subfigures in a row (each a third of the line, "
             "\\hfill between them), captioned as shown, and the figure caption.", sol, preamble=SUB,
             require=["env:subfigure"])


@template(M, "challenge")
def figure_report(rng):
    cap = rng.choice(CAPS)
    files = {"loss.pdf": _plot_svg("loss")}
    sol = (f"\\section{{Results}}\\label{{sec:res}}\n\\begin{{figure}}[htbp]\n  \\centering\n"
           f"  \\includegraphics[width=0.8\\linewidth]{{loss.pdf}}\n  \\caption{{{cap}}}\\label{{fig:loss}}\n"
           f"\\end{{figure}}\nSection~\\ref{{sec:res}} discusses Figure~\\ref{{fig:loss}}.")
    return L("**Results section.** Recreate the target: the section (labelled sec:res), the figure with loss.pdf, "
             "and the sentence whose two numbers come from \\ref.", sol, preamble=GFX, files=files, require=["\\ref"])

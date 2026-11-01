"""11.16 TikZ & pgfplots: nodes and styles, \\draw / \\fill paths (arrows, rectangles, circles), \\foreach, the
positioning library, neural-net and block diagrams; pgfplots axes from coordinates, expressions and CSV files."""

from ._common import AMS, L, drill, fixed, template

M = "11.16"
TIKZ = AMS + "\\usepackage{tikz}\n\\usetikzlibrary{arrows.meta, positioning}\n"
PLOTS = TIKZ + "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"


def _csv(rng, n=6):
    loss, val = rng.uniform(2.0, 2.6), rng.uniform(2.1, 2.7)
    rows = ["epoch,loss,val_loss"]
    for ep in range(1, n + 1):
        rows.append(f"{ep},{loss:.3f},{val:.3f}")
        loss *= rng.uniform(0.55, 0.75)
        val *= rng.uniform(0.65, 0.85)
    return "\n".join(rows) + "\n"


# ================================================================= lite

@template(M, "lite")
def shapes(rng):
    c = rng.choice(["blue", "red", "teal"])
    items = [
        ("Draw a line from (0,0) to (2,1) and an arrow from (0,0) to (2,0) (\\draw, \\draw[->]).",
         "\\begin{tikzpicture}\n  \\draw (0,0) -- (2,1);\n  \\draw[->] (0,0) -- (2,0);\n\\end{tikzpicture}",
         {"require": ["env:tikzpicture", "\\draw"]}),
        (f"Draw a thick {c} rectangle from (0,0) to (2,1), and fill a red circle of radius 2pt at (3,0.5) (the "
         f"guide's \\draw[thick, {c}] ... rectangle and \\fill[red] ... circle).",
         f"\\begin{{tikzpicture}}\n  \\draw[thick, {c}] (0,0) rectangle (2,1);\n  \\fill[red] (3,0.5) circle (2pt);\n"
         f"\\end{{tikzpicture}}", {"require": ["\\fill"]}),
        ("Draw x and y axes as arrows from (0,0) to (3,0) and (0,0) to (0,2), and a dashed line from (0,0) to (2.5,1.5).",
         "\\begin{tikzpicture}\n  \\draw[->] (0,0) -- (3,0);\n  \\draw[->] (0,0) -- (0,2);\n"
         "  \\draw[dashed] (0,0) -- (2.5,1.5);\n\\end{tikzpicture}", {"require": ["\\draw"]}),
    ]
    return drill(rng, items, preamble=TIKZ)


@template(M, "lite")
def nodes(rng):
    a, b = rng.choice([("x", "y"), ("u", "v"), ("z", "\\hat{y}"), ("h", "o")])
    d = rng.choice([2, 3, 4])
    box, out = rng.choice([("Model", "loss"), ("Encoder", "code"), ("Tokenizer", "ids"), ("Network", "logits")])
    items = [
        (f"Place two circled nodes, ${a}$ at (0,0) named a and ${b}$ at ({d},0) named b, and draw an arrow from a to b.",
         f"\\begin{{tikzpicture}}\n  \\node[circle, draw] (a) at (0,0) {{${a}$}};\n"
         f"  \\node[circle, draw] (b) at ({d},0) {{${b}$}};\n  \\draw[->] (a) -- (b);\n\\end{{tikzpicture}}",
         {"require": ["\\node"]}),
        (f"Place a rectangle node \u201c{box}\u201d (draw) at (0,0) named m, and a plain text node \u201c{out}\u201d at "
         f"({d},0) named l, with an arrow from m to l.",
         f"\\begin{{tikzpicture}}\n  \\node[draw] (m) at (0,0) {{{box}}};\n  \\node (l) at ({d},0) {{{out}}};\n"
         f"  \\draw[->] (m) -- (l);\n\\end{{tikzpicture}}", {"require": ["\\node"]}),
    ]
    return drill(rng, items, preamble=TIKZ)


# ================================================================= easy

@template(M, "easy")
def foreach(rng):
    n = rng.choice([3, 4, 5])
    items = [
        (f"Draw {n} circled nodes $x_1$ \u2026 $x_{n}$ in a column at (0,-1) \u2026 (0,-{n}) with one \\foreach, named "
         f"x1 \u2026 x{n}.",
         f"\\begin{{tikzpicture}}\n  \\foreach \\i in {{1,...,{n}}} \\node[circle, draw] (x\\i) at (0, -\\i) {{$x_\\i$}};\n"
         f"\\end{{tikzpicture}}", {"require": ["\\foreach"]}),
        (f"Draw {n} filled dots (\\fill circle (2pt)) at (1,0), (2,0) \u2026 ({n},0) with \\foreach.",
         f"\\begin{{tikzpicture}}\n  \\foreach \\x in {{1,...,{n}}} \\fill (\\x, 0) circle (2pt);\n\\end{{tikzpicture}}",
         {"require": ["\\foreach", "\\fill"]}),
    ]
    return drill(rng, items, preamble=TIKZ)


@template(M, "easy")
def first_plot(rng):
    rate = rng.choice([0.5, 0.6, 0.7])
    start = rng.choice([2.3, 1.8, 3.1])
    pts = [(i, round(start * rate ** (i - 1) + 0.3, 2)) for i in range(1, 6)]
    coords = " ".join(f"({x}, {y})" for x, y in pts)
    fn, tex, dom = rng.choice([("exp(-x)", "e^{-x}", "0:4"), ("x^2", "x^2", "-2:2"), ("1/(1 + exp(-x))",
                                                                                       "\\sigma(x)", "-6:6"),
                               ("max(x, 0)", "\\max(x, 0)", "-2:2"), ("ln(x)", "\\ln x", "0.1:5")])
    items = [
        ("Plot the loss values shown with pgfplots: an axis with xlabel=epoch and ylabel=loss, and \\addplot "
         "coordinates in blue with dot marks (mark=*).",
         f"\\begin{{tikzpicture}}\n  \\begin{{axis}}[xlabel=epoch, ylabel=loss]\n"
         f"    \\addplot[blue, mark=*] coordinates {{{coords}}};\n  \\end{{axis}}\n\\end{{tikzpicture}}",
         {"require": ["env:axis", "\\addplot"]}),
        (f"Plot ${tex}$ for x in {dom} (domain={dom}, samples=50) as a red curve, on an axis with xlabel=$x$ and "
         f"ylabel=$y$. (pgfplots writes it {fn}.)",
         f"\\begin{{tikzpicture}}\n  \\begin{{axis}}[xlabel=$x$, ylabel=$y$]\n"
         f"    \\addplot[red, domain={dom}, samples=50] {{{fn}}};\n  \\end{{axis}}\n\\end{{tikzpicture}}",
         {"require": ["\\addplot"]}),
    ]
    return drill(rng, items, preamble=PLOTS)


# ================================================================= medium

@template(M, "medium")
def mlp(rng):
    a, b, c = rng.choice([(3, 4, 2), (2, 3, 1), (3, 2, 1), (4, 3, 2)])
    off_h = (a - b) / 2
    off_y = (a - c) / 2
    hy = f"-\\j-{off_h:g}" if off_h > 0 else (f"-\\j+{-off_h:g}" if off_h < 0 else "-\\j")
    yy = f"-\\k-{off_y:g}" if off_y > 0 else (f"-\\k+{-off_y:g}" if off_y < 0 else "-\\k")
    sol = (f"\\begin{{tikzpicture}}[neuron/.style={{circle, draw, minimum size=7mm}}, >={{Stealth}}]\n"
           f"  \\foreach \\i in {{1,...,{a}}} \\node[neuron] (x\\i) at (0, -\\i) {{$x_\\i$}};\n"
           f"  \\foreach \\j in {{1,...,{b}}} \\node[neuron, fill=blue!10] (h\\j) at (2, {hy}) {{$h_\\j$}};\n"
           f"  \\foreach \\k in {{1,...,{c}}} \\node[neuron, fill=red!10] (y\\k) at (4, {yy}) {{$y_\\k$}};\n"
           f"  \\foreach \\i in {{1,...,{a}}} \\foreach \\j in {{1,...,{b}}} \\draw[->] (x\\i) -- (h\\j);\n"
           f"  \\foreach \\j in {{1,...,{b}}} \\foreach \\k in {{1,...,{c}}} \\draw[->] (h\\j) -- (y\\k);\n"
           f"\\end{{tikzpicture}}")
    return L(f"**A {a}-{b}-{c} MLP** (the guide's drill). Draw the target: a neuron style (circle, draw, minimum "
             f"size=7mm), the layers at x = 0, 2, 4 centered on each other (each layer's nodes 1 apart), every node "
             f"connected to the next layer with Stealth arrows; nested \\foreach loops do the edges.", sol,
             preamble=TIKZ, require=["\\foreach", "\\node", "\\draw"])


@template(M, "medium")
def block_diagram(rng):
    a, b = rng.choice([("Encoder", "Decoder"), ("Tokenizer", "Model"), ("Generator", "Discriminator")])
    lab = rng.choice(["$z$", "$h$", "logits"])
    sol = (f"\\begin{{tikzpicture}}[box/.style={{draw, rounded corners, minimum width=2cm, minimum height=8mm}}, "
           f">={{Stealth}}]\n  \\node[box] (a) {{{a}}};\n  \\node[box, right=1.5cm of a] (b) {{{b}}};\n"
           f"  \\draw[->, thick] (a) -- node[above] {{{lab}}} (b);\n\\end{{tikzpicture}}")
    return L(f"**{a} \u2192 {b}** (the guide's drill). Draw the target: a box style (draw, rounded corners, minimum "
             f"width=2cm, minimum height=8mm), the second box 1.5cm to the right of the first (positioning: "
             f"right=1.5cm of a), and a thick arrow labelled {lab} above it.", sol, preamble=TIKZ,
             require=["\\node", "\\usetikzlibrary"])


# ================================================================= hard

@template(M, "hard")
def csv_plot(rng):
    data = _csv(rng)
    sol = ("\\begin{tikzpicture}\n  \\begin{axis}[xlabel=epoch, ylabel=loss, legend pos=north east, grid=major]\n"
           "    \\addplot[blue, mark=*] table[x=epoch, y=loss, col sep=comma] {log.csv};\n"
           "    \\addlegendentry{train}\n"
           "    \\addplot[red, dashed, mark=square*] table[x=epoch, y=val_loss, col sep=comma] {log.csv};\n"
           "    \\addlegendentry{val}\n  \\end{axis}\n\\end{tikzpicture}")
    return L("**Plot from a CSV** (the guide's drill). pandas wrote log.csv (df.to_csv(\"log.csv\", index=False)). "
             "Plot train (loss: blue, mark=*) and val (val_loss: red, dashed, mark=square*) with \\addplot table "
             "(col sep=comma), a legend at the north east, and a major grid.", sol, preamble=PLOTS,
             files={"log.csv": data}, require=["\\addplot", "\\addlegendentry"])


@template(M, "hard")
@fixed
def log_plot(rng):
    sol = ("\\begin{tikzpicture}\n  \\begin{axis}[xlabel=epoch, ylabel=loss, ymode=log, legend pos=north east, "
           "width=8cm, height=5cm, grid=major]\n    \\addplot[blue, mark=*] coordinates {(1, 2.3) (2, 1.1) (3, 0.6) "
           "(4, 0.4) (5, 0.35)};\n    \\addlegendentry{train}\n    \\addplot[red, dashed, domain=1:5, samples=50] "
           "{2.5 * exp(-0.5 * x) + 0.3};\n    \\addlegendentry{val}\n  \\end{axis}\n\\end{tikzpicture}")
    return L("**The guide's loss plot.** Recreate the target: a log-scale y axis (ymode=log), 8cm \u00d7 5cm, a major "
             "grid, train as coordinates and val as the expression $2.5 e^{-0.5x} + 0.3$, with a legend.", sol,
             preamble=PLOTS, require=["\\addplot", "\\addlegendentry"])


# ================================================================= challenge

@template(M, "challenge")
def transformer_block(rng):
    names = rng.choice([("Embeddings", "Self-attention", "Feed-forward", "Softmax"),
                        ("Image", "Convolution", "Pooling", "Classifier"),
                        ("Tokens", "Encoder", "Decoder", "Output"),
                        ("Audio", "Spectrogram", "Transformer", "Text")])
    gap = rng.choice(["6mm", "5mm", "8mm"])
    keys = ["in", "b", "c", "out"]
    lines = [f"  \\node[box] (in) {{{names[0]}}};",
             f"  \\node[box, fill=blue!10, above={gap} of in] (b) {{{names[1]}}};",
             f"  \\node[box, fill=green!10, above={gap} of b] (c) {{{names[2]}}};",
             f"  \\node[box, above={gap} of c] (out) {{{names[3]}}};"]
    lines += [f"  \\draw[->] ({x}) -- ({y});" for x, y in zip(keys, keys[1:])]
    sol = ("\\begin{tikzpicture}[box/.style={draw, rounded corners, minimum width=3cm, minimum height=7mm}, "
           ">={Stealth}]\n" + "\n".join(lines) + "\n\\end{tikzpicture}")
    return L(f"**A pipeline, stacked.** Recreate the target: four boxes stacked with above={gap} of ..., the middle "
             f"two filled, joined by arrows.", sol, preamble=TIKZ, require=["\\node", "\\draw"])


@template(M, "challenge")
@fixed
def figure_with_plot(rng):
    sol = ("\\begin{figure}[htbp]\n  \\centering\n  \\begin{tikzpicture}\n    \\begin{axis}[xlabel=step, "
           "ylabel=learning rate, width=8cm, height=5cm]\n      \\addplot[blue, domain=0:100, samples=101] "
           "{min(x / 10, 1) * 0.5 * (1 + cos(180 * x / 100))};\n    \\end{axis}\n  \\end{tikzpicture}\n"
           "  \\caption{Linear warmup, then cosine decay.}\\label{fig:lr}\n\\end{figure}")
    return L("**A plot as a figure.** Recreate the target: a figure whose body is a pgfplots axis (8cm \u00d7 5cm) "
             "plotting warmup-then-cosine, $\\min(x/10, 1) \\cdot \\frac{1}{2}(1 + \\cos(180 x / 100))$ (pgfplots' "
             "cos takes degrees) for x from 0 to 100 with 101 samples, captioned.", sol, preamble=PLOTS,
             require=["env:axis", "env:figure"])

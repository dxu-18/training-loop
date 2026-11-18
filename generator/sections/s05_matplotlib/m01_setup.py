"""5.01 Setup, backends & the two interfaces (pyplot's implicit state vs explicit Figure / Axes objects)."""

from ._common import P, catalog, categories, challenge, curves, lit, seeded, signal, template

M = "5.01"
LINES = "fc.lines(fc.ax())"
LAB = "fc.labels(fc.ax())"


def _wave(rng):
    n = rng.choice([50, 80, 100])
    f = rng.choice([1, 2, 3])
    return f"x = np.linspace(0, 2 * np.pi, {n})", n, f


# ================================================================= lite

@template(M, "lite")
def pyplot_basics(rng):
    code, n, f = _wave(rng)
    title = rng.choice(["Waves", "Signals", "Sine and cosine", "Two curves"])
    xl, yl = rng.choice([("x", "y"), ("time (s)", "amplitude"), ("angle", "value")])
    items = [
        ("plt.plot", f"With the pyplot interface, plot `np.sin({f} * x)` against `x`.", f"plt.plot(x, np.sin({f} * x))",
         {"post": LINES}),
        ("plt.plot", f"With pyplot, plot `np.sin(x)` and then `np.cos(x)` against `x` (two lines on the same plot).",
         "plt.plot(x, np.sin(x))\nplt.plot(x, np.cos(x))", {"post": LINES}),
        ("plt.title", f"Plot `np.sin(x)` against `x` with pyplot and give it the title {title!r}.",
         f"plt.plot(x, np.sin(x))\nplt.title({title!r})", {"post": f"{LINES}, {LAB}"}),
        ("plt.xlabel", f"Plot `np.cos(x)` against `x` with pyplot and label the axes: x is {xl!r}, y is {yl!r}.",
         f"plt.plot(x, np.cos(x))\nplt.xlabel({xl!r})\nplt.ylabel({yl!r})",
         {"post": f"{LINES}, {LAB}", "require": ["xlabel", "ylabel"]}),
        ("plt.legend", "With pyplot, plot `np.sin(x)` labelled `'sin'` and `np.cos(x)` labelled `'cos'`, then show a "
                       "legend.", "plt.plot(x, np.sin(x), label='sin')\nplt.plot(x, np.cos(x), label='cos')\nplt.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()"}),
        ("plt.grid", "Plot `np.sin(x)` against `x` with pyplot and turn the grid on.",
         "plt.plot(x, np.sin(x))\nplt.grid(True)", {"post": f"{LINES}, fc.grid_on()"}),
        ("plt.figure", f"Start a new figure of size 6 × 3 inches with `plt.figure(figsize=(6, 3))`, then plot "
                       f"`np.sin(x)` against `x`.", "plt.figure(figsize=(6, 3))\nplt.plot(x, np.sin(x))",
         {"post": f"{LINES}, fc.size()"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π. Don't call `plt.show()`.")


@template(M, "lite")
def explicit_basics(rng):
    code, n, f = _wave(rng)
    w, h = rng.choice([(6, 3), (8, 4), (5, 5), (7, 2.5)])
    title = rng.choice(["Explicit API", "Model output", "Activations"])
    items = [
        ("plt.subplots", f"Create a figure and one Axes with `fig, ax = plt.subplots()` and plot `np.sin({f} * x)` "
                         f"on `ax`.", f"fig, ax = plt.subplots()\nax.plot(x, np.sin({f} * x))", {"post": LINES}),
        ("plt.subplots", f"Create `fig, ax` with a figure size of {w} × {h} inches and plot `np.cos(x)` on `ax`.",
         f"fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(x, np.cos(x))", {"post": f"{LINES}, fc.size()"}),
        (".set", f"Create `fig, ax`, plot `np.sin(x)`, and set the title {title!r}, x label 'x' and y label 'y' in "
                 f"**one** call to `ax.set(...)`.",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nax.set(title={title!r}, xlabel='x', ylabel='y')",
         {"post": f"{LINES}, {LAB}"}),
        (".legend", "Create `fig, ax`, plot `np.sin(x)` labelled 'sin' and `np.cos(x)` labelled 'cos' on it, and add "
                    "a legend with `ax.legend()`.",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x), label='sin')\nax.plot(x, np.cos(x), label='cos')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()"}),
        (".grid", "Create `fig, ax`, plot `np.cos(x)`, and turn on the grid with `ax.grid(True)`.",
         "fig, ax = plt.subplots()\nax.plot(x, np.cos(x))\nax.grid(True)", {"post": f"{LINES}, fc.grid_on()"}),
        (".set_title", f"Create `fig, ax`, plot `np.sin(x)`, and set the title {title!r} with `ax.set_title`.",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nax.set_title({title!r})", {"post": f"{LINES}, {LAB}"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "lite")
def environment_facts(rng):
    k = rng.randint(2, 6)
    items = [
        ("__version__", "Return the installed Matplotlib version string (`mpl` is imported).", "result = mpl.__version__",
         {"require": ["__version__"]}),
        ("plt.close", f"Create {k} figures with `plt.figure()` in a loop, then close them all with `plt.close('all')` "
                      f"and return how many are still open (`len(plt.get_fignums())`).",
         f"for _ in range({k}):\n    plt.figure()\nplt.close('all')\nresult = len(plt.get_fignums())"),
        ("plt.get_fignums", f"Create {k} figures with `plt.figure()` and return the list of open figure numbers "
                            f"(`plt.get_fignums()`).", f"for _ in range({k}):\n    plt.figure()\nresult = plt.get_fignums()"),
        ("plt.gcf", "Create `fig, ax = plt.subplots()` and return whether pyplot's current figure (`plt.gcf()`) is "
                    "`fig`.", "fig, ax = plt.subplots()\nresult = plt.gcf() is fig"),
        ("plt.gca", "Create `fig, ax = plt.subplots()` and return whether pyplot's current Axes (`plt.gca()`) is `ax`.",
         "fig, ax = plt.subplots()\nresult = plt.gca() is ax"),
        ("get_backend", "Return the name of the backend Matplotlib uses here (`mpl.get_backend()`; this page renders "
                        "off-screen, without windows).", "result = mpl.get_backend()", {"require": ["get_backend"]}),
        ("plt.close", "Create `fig, ax = plt.subplots()`, close **that figure** with `plt.close(fig)`, and return the "
                      "number of open figures.", "fig, ax = plt.subplots()\nplt.close(fig)\nresult = len(plt.get_fignums())"),
    ]
    return catalog(rng, items, "", "")


@template(M, "lite")
def save_files(rng):
    code, n, f = _wave(rng)
    name = rng.choice(["plot", "figure", "curves", "wave"])
    dpi = rng.choice([50, 72, 100])
    items = [
        ("plt.savefig", f"Plot `np.sin(x)` with pyplot and save it as `{name}.png` with `plt.savefig`.",
         f"plt.plot(x, np.sin(x))\nplt.savefig({name + '.png'!r})",
         {"post": f"open({name + '.png'!r}, 'rb').read(8)"}),
        (".savefig", f"Create `fig, ax`, plot `np.cos(x)`, and save the figure as `{name}.pdf` with `fig.savefig`.",
         f"fig, ax = plt.subplots()\nax.plot(x, np.cos(x))\nfig.savefig({name + '.pdf'!r})",
         {"post": f"open({name + '.pdf'!r}, 'rb').read(5)"}),
        (".savefig", f"Create a 4 × 3 inch figure (`fig, ax`), plot `np.sin(x)`, and save it as `{name}.png` at "
                     f"{dpi} dpi. (It is then {4 * dpi} × {3 * dpi} pixels.)",
         f"fig, ax = plt.subplots(figsize=(4, 3))\nax.plot(x, np.sin(x))\nfig.savefig({name + '.png'!r}, dpi={dpi})",
         {"post": f"plt.imread({name + '.png'!r}).shape"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


# ================================================================= easy

@template(M, "easy")
def both_styles(rng):
    code, desc, n = curves(rng)
    title = rng.choice(["Training curves", "Loss", "Run 7"])
    pyplot = (f"plt.figure(figsize=(6, 3))\nplt.plot(epochs, train_loss, label='train')\nplt.plot(epochs, val_loss, "
              f"label='val')\nplt.title({title!r})\nplt.xlabel('epoch')\nplt.ylabel('loss')\nplt.legend()")
    explicit = (f"fig, ax = plt.subplots(figsize=(6, 3))\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, "
                f"val_loss, label='val')\nax.set(title={title!r}, xlabel='epoch', ylabel='loss')\nax.legend()")
    post = "fc.lines(fc.ax(), 'label'), fc.labels(fc.ax()), fc.legend(), fc.size()"
    items = [
        ("plt.subplots", f"This pyplot (implicit) code draws a plot:\n```python\n{pyplot}\n```\nDraw the same figure in "
                         f"the **explicit** style: `fig, ax = plt.subplots(...)` and only `ax.` / `fig.` methods (no "
                         f"`plt.plot` / `plt.title` ...).", explicit, {"post": post}),
        ("plt.plot", f"This explicit (object-oriented) code draws a plot:\n```python\n{explicit}\n```\nDraw the same "
                     f"figure with the **pyplot** functions only (`plt.figure`, `plt.plot`, `plt.title`, ...).", pyplot,
         {"post": post, "require": ["plt.plot|plot", "title"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def translate_calls(rng):
    code, desc, n = signal(rng)
    lo, hi = rng.choice([(0, 5), (2, 8), (1, 9)])
    items = [
        ("set_xlim", f"Pyplot's `plt.xlim({lo}, {hi})` is `ax.set_xlim` on an Axes. Create `fig, ax`, plot `y` against "
                     f"`x`, and limit the x-axis to {lo} … {hi}.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nax.set_xlim({lo}, {hi})", {"post": f"{LINES}, fc.limits()['xlim']"}),
        ("set_ylim", "Create `fig, ax`, plot `y` against `x`, and set the y-axis limits to -2 … 2 (the explicit "
                     "version of `plt.ylim(-2, 2)`).", "fig, ax = plt.subplots()\nax.plot(x, y)\nax.set_ylim(-2, 2)",
         {"post": f"{LINES}, fc.limits()['ylim']"}),
        ("set_xscale", "Create `fig, ax`, plot `x + 1` against `x`, and make the **y**-axis logarithmic (explicit "
                       "version of `plt.yscale('log')`).", "fig, ax = plt.subplots()\nax.plot(x, x + 1)\nax.set_yscale('log')",
         {"post": f"{LINES}, fc.scales()", "require": ["set_yscale"]}),
        ("set_xticks", "Create `fig, ax`, plot `y` against `x`, and put x ticks at 0, 5 and 10 labelled 'start', 'mid', "
                       "'end' (explicit version of `plt.xticks(ticks, labels)`).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.set_xticks([0, 5, 10], labels=['start', 'mid', 'end'])",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x')"}),
        ("plt.xticks", "With pyplot only, plot `y` against `x` and put x ticks at 0, 5 and 10 labelled 'start', 'mid', "
                       "'end'.", "plt.plot(x, y)\nplt.xticks([0, 5, 10], ['start', 'mid', 'end'])",
         {"post": f"{LINES}, fc.ticks(fc.ax(), 'x')"}),
        (".text", "Create `fig, ax`, plot `y` against `x`, and write the text 'peak' at data position (5, 1) with "
                  "`ax.text` (the explicit `plt.text`).", "fig, ax = plt.subplots()\nax.plot(x, y)\nax.text(5, 1, 'peak')",
         {"post": f"{LINES}, fc.text_items()"}),
        ("plt.xlim", f"With pyplot only, plot `y` against `x` and limit the x-axis to {lo} … {hi}.",
         f"plt.plot(x, y)\nplt.xlim({lo}, {hi})", {"post": f"{LINES}, fc.limits()['xlim']"}),
        ("plt.yscale", "With pyplot only, plot `x + 1` against `x` on a logarithmic y-axis.",
         "plt.plot(x, x + 1)\nplt.yscale('log')", {"post": f"{LINES}, fc.scales()"}),
        (".colorbar", "Create `fig, ax`, show `np.outer(y, y)` with `im = ax.imshow(...)`, and add its colorbar with "
                      "`fig.colorbar(im, ax=ax)` (the explicit `plt.colorbar()`).",
         "fig, ax = plt.subplots()\nim = ax.imshow(np.outer(y, y))\nfig.colorbar(im, ax=ax)",
         {"post": "fc.images(fc.ax(0)), len(fc.axes(colorbars=True))"}),
        ("plt.colorbar", "With pyplot only, show `np.outer(y, y)` with `plt.imshow` and add a colorbar.",
         "plt.imshow(np.outer(y, y))\nplt.colorbar()", {"post": "fc.images(fc.ax(0)), len(fc.axes(colorbars=True))"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def current_state(rng):
    code, n, f = _wave(rng)
    k = rng.randint(2, 4)
    items = [
        ("plt.sca", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`. Make `ax1` the current Axes with `plt.sca` and then "
                    "plot `np.sin(x)` with **pyplot's** `plt.plot`, so the line lands on the left plot.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nplt.sca(ax1)\nplt.plot(x, np.sin(x))",
         {"post": "[len(a.get_lines()) for a in fc.axes()], fc.lines(fc.ax(0))"}),
        ("plt.gca", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`. Pyplot's current Axes is the **last** one created. "
                    "Return `[plt.gca() is ax1, plt.gca() is ax2]`.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nresult = [plt.gca() is ax1, plt.gca() is ax2]"),
        ("plt.figure", f"Pyplot acts on the **current** figure. Create figures 1 and 2 with `plt.figure(1)` and "
                       f"`plt.figure(2)`, then switch back to figure 1 with `plt.figure(1)` and plot `np.sin(x)` with "
                       f"`plt.plot`. Return how many lines each figure has, as `[lines in fig 1, lines in fig 2]`.",
         "plt.figure(1)\nplt.figure(2)\nplt.figure(1)\nplt.plot(x, np.sin(x))\n"
         "result = [len(plt.figure(n).gca().get_lines()) for n in (1, 2)]"),
        ("plt.gcf", f"Create {k} figures with `plt.figure()`; return the **number** of the current figure "
                    f"(`plt.gcf().number`).", f"for _ in range({k}):\n    plt.figure()\nresult = plt.gcf().number"),
        ("plt.close", f"Create figures 1 … {k} with `plt.figure()`, close the current one with `plt.close()`, and "
                      f"return the number of the figure that is current now.",
         f"for _ in range({k}):\n    plt.figure()\nplt.close()\nresult = plt.gcf().number"),
        ("plt.gca", "Create `fig, ax = plt.subplots()` and plot `np.sin(x)` with **pyplot**, then return how many lines "
                    "`ax` has (pyplot drew on the current Axes, which is `ax`).",
         "fig, ax = plt.subplots()\nplt.plot(x, np.sin(x))\nresult = len(ax.get_lines())", {"require": ["plt.plot"]}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "easy")
def file_formats(rng):
    code, n, f = _wave(rng)
    name = rng.choice(["report", "result", "fig1", "curves"])
    w, h, dpi = rng.choice([(4, 3, 50), (6, 4, 50), (5, 2, 80), (3, 3, 60)])
    items = [
        (".savefig", f"Create `fig, ax`, plot `np.sin(x)`, and save the **same** figure as both `{name}.png` and "
                     f"`{name}.pdf`. Return the sorted list of files in the current folder (`sorted(os.listdir())`).",
         f"import os\nfig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nfig.savefig({name + '.png'!r})\n"
         f"fig.savefig({name + '.pdf'!r})\nresult = sorted(os.listdir())", {"require": ["savefig"]}),
        ("dpi", f"Inches × dpi = pixels. Create a {w} × {h} inch figure, plot `np.cos(x)`, save it as `{name}.png` at "
                f"{dpi} dpi, and return its shape as read back with `plt.imread` (rows, columns, channels).",
         f"fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(x, np.cos(x))\nfig.savefig({name + '.png'!r}, dpi={dpi})\n"
         f"result = plt.imread({name + '.png'!r}).shape", {"require": ["dpi", "imread"]}),
        (".savefig", f"Save one figure (`np.sin(x)` on `fig, ax`) as `{name}.svg` and return its first 5 characters "
                     f"(SVG is XML text: it starts with `<?xml`).",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nfig.savefig({name + '.svg'!r})\n"
         f"result = open({name + '.svg'!r}).read(5)"),
        ("plt.savefig", f"With pyplot, plot `np.sin(x)`, save it as `{name}.png` at {dpi} dpi with `plt.savefig`, and "
                        f"close the figure. Return `(os.path.exists({name + '.png'!r}), len(plt.get_fignums()))`.",
         f"import os\nplt.plot(x, np.sin(x))\nplt.savefig({name + '.png'!r}, dpi={dpi})\nplt.close()\n"
         f"result = (os.path.exists({name + '.png'!r}), len(plt.get_fignums()))", {"require": ["savefig", "close"]}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "easy")
def first_plots(rng):
    code, desc, names, vals, th, m = categories(rng)
    items = [
        ("plt.plot", f"Plot `values` against their position (`plt.plot(values)`: x = 0, 1, 2 …) with pyplot and title "
                     f"it {m.name.capitalize()!r}.", f"plt.plot(values)\nplt.title({m.name.capitalize()!r})",
         {"post": f"{LINES}, {LAB}"}),
        (".plot", f"Create `fig, ax` and plot `values` against `names` (categorical x positions) as a line with "
                  f"circle markers (`'o-'`).", "fig, ax = plt.subplots()\nax.plot(names, values, 'o-')",
         {"post": "fc.lines(fc.ax(), 'marker', 'ls', data=False), fc.tick_labels(fc.ax(), 'x')"}),
        (".set_ylabel", f"Create `fig, ax`, plot `values` with markers only (`'o'`), and label the y-axis "
                        f"{m.label!r}.", f"fig, ax = plt.subplots()\nax.plot(values, 'o')\nax.set_ylabel({m.label!r})",
         {"post": f"fc.lines(fc.ax(), 'marker', 'ls'), {LAB}"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def figure_objects(rng):
    w, h = rng.choice([(6, 3), (4, 4), (8, 2), (5, 3.5)])
    items = [
        ("plt.subplots", f"Create `fig, ax = plt.subplots(figsize=({w}, {h}))` and return `fig.get_size_inches()` as a "
                         f"list.", f"fig, ax = plt.subplots(figsize=({w}, {h}))\nresult = fig.get_size_inches().tolist()"),
        ("plt.subplots", "Return the type names of what `plt.subplots()` returns, as `[type(fig).__name__, "
                         "type(ax).__name__]`.", "fig, ax = plt.subplots()\nresult = [type(fig).__name__, type(ax).__name__]"),
        (".axes", "Create `fig, ax = plt.subplots()` and return whether `ax` is the one entry of `fig.axes` "
                  "(`fig.axes == [ax]`).", "fig, ax = plt.subplots()\nresult = fig.axes == [ax]"),
        ("get_figure", "Every Axes knows its figure. Create `fig, ax` and return `ax.get_figure() is fig`.",
         "fig, ax = plt.subplots()\nresult = ax.get_figure() is fig"),
        ("plt.figure", f"Create a figure with `plt.figure(figsize=({w}, {h}))` and return how many Axes it has "
                       f"(`len(fig.axes)`: none until you plot or add one).",
         f"fig = plt.figure(figsize=({w}, {h}))\nresult = len(fig.axes)"),
        (".plot", "`ax.plot` returns a **list** of the lines it drew. Create `fig, ax`, call `lines = ax.plot([1, 2, 3], "
                  "[3, 1, 2])` and return `[len(lines), type(lines[0]).__name__]`.",
         "fig, ax = plt.subplots()\nlines = ax.plot([1, 2, 3], [3, 1, 2])\nresult = [len(lines), type(lines[0]).__name__]"),
    ]
    return catalog(rng, items, "", "")


# ================================================================= medium

@template(M, "medium")
def report_figure(rng):
    code, desc, n = curves(rng, acc=True)
    name = rng.choice(["training", "report", "run"])
    items = [
        ("plt.subplots", "Create `fig, ax`, plot `train_loss` and `val_loss` against `epochs` labelled 'train' and 'val', "
                         "set the x label 'epoch', the y label 'loss', the title 'Loss', add a legend and a grid.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
         "ax.set(xlabel='epoch', ylabel='loss', title='Loss')\nax.legend()\nax.grid(True)",
         {"post": "fc.lines(fc.ax(), 'label'), fc.labels(fc.ax()), fc.legend(), fc.grid_on()"}),
        ("plt.plot", "With **pyplot only**, plot `val_acc` against `epochs` labelled 'val acc', add a horizontal "
                     "reference line at 0.9 with `plt.axhline(0.9, linestyle='--')`, a legend and the title 'Accuracy'.",
         "plt.plot(epochs, val_acc, label='val acc')\nplt.axhline(0.9, linestyle='--')\nplt.legend()\nplt.title('Accuracy')",
         {"post": "fc.lines(fc.ax(), 'label', 'ls'), fc.legend(), fc.labels(fc.ax())", "require": ["axhline"]}),
        (".savefig", f"Create `fig, ax` of size 6 × 3, plot `train_loss` against `epochs`, save it as `{name}.png` at 50 "
                     f"dpi, close the figure, and return `(plt.imread({name + '.png'!r}).shape, len(plt.get_fignums()))`.",
         f"fig, ax = plt.subplots(figsize=(6, 3))\nax.plot(epochs, train_loss)\nfig.savefig({name + '.png'!r}, dpi=50)\n"
         f"plt.close(fig)\nresult = (plt.imread({name + '.png'!r}).shape, len(plt.get_fignums()))",
         {"require": ["savefig", "close"]}),
        ("plt.subplots", "Plot the loss and the accuracy in **two separate figures**: `fig1, ax1` with `train_loss` and "
                         "`fig2, ax2` with `val_acc` (both against `epochs`). Return `[len(ax1.get_lines()), "
                         "len(ax2.get_lines()), len(plt.get_fignums())]`.",
         "fig1, ax1 = plt.subplots()\nax1.plot(epochs, train_loss)\nfig2, ax2 = plt.subplots()\nax2.plot(epochs, val_acc)\n"
         "result = [len(ax1.get_lines()), len(ax2.get_lines()), len(plt.get_fignums())]"),
        ("argmin", "Create `fig, ax`, plot `val_loss` against `epochs`, and mark the best epoch (lowest `val_loss`) with "
                   "a red dot: `ax.plot(best_epoch, best_loss, 'ro')`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nb = val_loss.argmin()\nax.plot(epochs[b], val_loss[b], 'ro')",
         {"post": "fc.lines(fc.ax(), 'color', 'marker')"}),
        (".set", "Create `fig, ax`, plot `train_loss` against `epochs`, and in **one** `ax.set` call set the x label "
                 "'epoch', the y label 'loss', the y limits 0 … 2.5 and the title 'Training'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set(xlabel='epoch', ylabel='loss', ylim=(0, 2.5), title='Training')",
         {"post": "fc.lines(fc.ax()), fc.labels(fc.ax()), fc.limits()['ylim']"}),
        ("plt.figure", "With pyplot only: in figure 1 plot `train_loss`, in figure 2 plot `val_loss` (both against "
                       "`epochs`), then go back to figure 1 and give it the title 'train'. Return "
                       "`[plt.figure(1).axes[0].get_title(), plt.figure(2).axes[0].get_title()]`.",
         "plt.figure(1)\nplt.plot(epochs, train_loss)\nplt.figure(2)\nplt.plot(epochs, val_loss)\nplt.figure(1)\n"
         "plt.title('train')\nresult = [plt.figure(1).axes[0].get_title(), plt.figure(2).axes[0].get_title()]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def many_figures(rng):
    k = rng.randint(8, 30)
    s = seeded(rng)
    items = [
        ("plt.close", f"Make {k} figures in a loop, one per run: plot `runs[i]` on a new `fig, ax`, save it as "
                      f"`run_0.png`, `run_1.png`, … (dpi 30), and **close** it right away. Return `(number of .png files, number of "
                      f"open figures)`.",
         f"import os\nfor i, r in enumerate(runs):\n    fig, ax = plt.subplots(figsize=(2, 1.5))\n    ax.plot(r)\n"
         f"    fig.savefig(f'run_{{i}}.png', dpi=30)\n    plt.close(fig)\n"
         f"result = (sum(f.endswith('.png') for f in os.listdir()), len(plt.get_fignums()))",
         {"require": ["close", "savefig"], "timeout": 25.0}),
        ("plt.close", f"Pyplot keeps every figure until closed, and warns after 20. Create {k} figures in a loop, "
                      f"closing each before making the next, and return the largest number of figures that were open at "
                      f"the same time (measure `len(plt.get_fignums())` inside the loop).",
         f"most = 0\nfor r in runs:\n    fig, ax = plt.subplots()\n    ax.plot(r)\n    most = max(most, len(plt.get_fignums()))\n"
         f"    plt.close(fig)\nresult = most"),
        ("plt.subplots", f"Plot all {k} runs on **one** Axes (one line each) and return the number of lines on it.",
         "fig, ax = plt.subplots()\nfor r in runs:\n    ax.plot(r)\nresult = len(ax.get_lines())"),
        (".plot", f"Plot all {k} runs on one Axes in a **single** `ax.plot` call (a 2-D array plots one line per "
                  f"column: pass `runs.T`) and return the number of lines.",
         "fig, ax = plt.subplots()\nax.plot(runs.T)\nresult = len(ax.get_lines())"),
        ("plt.close", f"Count leaked figures: create {k} figures in a loop but close only every other one (when `i % 2 "
                      f"== 0`), then return how many are still open.",
         f"for i in range({k}):\n    fig, ax = plt.subplots()\n    if i % 2 == 0:\n        plt.close(fig)\n"
         f"result = len(plt.get_fignums())"),
    ]
    return catalog(rng, items, f"runs = np.round(np.random.default_rng({s}).normal(size=({k}, 20)).cumsum(axis=1), 3)",
                   f"`runs` holds {k} random-walk curves (one per row, 20 steps each).")


@template(M, "medium")
def axes_objects(rng):
    code, desc, n = signal(rng)
    items = [
        (".get_lines", "Create `fig, ax`, plot `y` against `x` twice (once as is, once as `y * 2`) and return the y data "
                       "of each line read back from `ax.get_lines()` (a list of 2 lists).",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.plot(x, y * 2)\nresult = [l.get_ydata().tolist() for l in ax.get_lines()]"),
        (".plot", "Keep the line handle: `line, = ax.plot(x, y)`. Change its color to 'red' and its label to 'signal' "
                  "through the handle (`line.set_color`, `line.set_label`) and add a legend.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_color('red')\nline.set_label('signal')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'color', 'label'), fc.legend()", "require": ["set_color", "set_label"]}),
        ("get_title", "Create `fig, ax`, plot `y`, set the title 'raw' and the y label 'amplitude', and return `[ax."
                      "get_title(), ax.get_ylabel()]`.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.set(title='raw', ylabel='amplitude')\nresult = [ax.get_title(), ax.get_ylabel()]"),
        ("get_xlim", "Create `fig, ax`, plot `y` against `x`, and return the x limits Matplotlib picked automatically "
                     "(`ax.get_xlim()` as a list): the data range plus a 5% margin each side.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nresult = list(ax.get_xlim())", {"tol": 1e-9}),
        (".set_ydata", "Create `fig, ax`, draw `line, = ax.plot(x, y)`, then **update** the existing line to show "
                       "`np.abs(y)` (`line.set_ydata`) instead of plotting a second one.",
         "fig, ax = plt.subplots()\nline, = ax.plot(x, y)\nline.set_ydata(np.abs(y))", {"post": LINES}),
        ("remove", "Create `fig, ax`, plot `y` and `y + 1`, then remove the first line (`line.remove()`), leaving one.",
         "fig, ax = plt.subplots()\nfirst, = ax.plot(x, y)\nax.plot(x, y + 1)\nfirst.remove()", {"post": LINES}),
        ("cla", "Create `fig, ax`, plot `y`, then clear the Axes with `ax.cla()` and plot `-y` instead.",
         "fig, ax = plt.subplots()\nax.plot(x, y)\nax.cla()\nax.plot(x, -y)", {"post": LINES, "require": ["cla|clear"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def implicit_pitfalls(rng):
    code, desc, n = curves(rng)
    title = rng.choice(["train", "Loss curve", "Experiment A"])
    items = [
        ("plt.figure", f"This code means to title the **first** figure, but `plt.title` acts on the current one, which "
                       f"is the second:\n```python\nplt.figure(1); plt.plot(epochs, train_loss)\nplt.figure(2); "
                       f"plt.plot(epochs, val_loss)\nplt.title({title!r})\n```\nFix it (keep both figures, title only "
                       f"figure 1). Return `[plt.figure(1).axes[0].get_title(), plt.figure(2).axes[0].get_title()]`.",
         f"plt.figure(1)\nplt.plot(epochs, train_loss)\nplt.title({title!r})\nplt.figure(2)\nplt.plot(epochs, val_loss)\n"
         "result = [plt.figure(1).axes[0].get_title(), plt.figure(2).axes[0].get_title()]", {"require": []}),
        ("set_title", f"Same bug, explicit fix: draw `train_loss` on `fig1, ax1` and `val_loss` on `fig2, ax2` (both "
                      f"`plt.subplots()`), then title the first with `ax1.set_title({title!r})` — the order no longer "
                      f"matters. Return `[ax1.get_title(), ax2.get_title()]`.",
         f"fig1, ax1 = plt.subplots()\nax1.plot(epochs, train_loss)\nfig2, ax2 = plt.subplots()\nax2.plot(epochs, val_loss)\n"
         f"ax1.set_title({title!r})\nresult = [ax1.get_title(), ax2.get_title()]"),
        ("plt.close", "This loop leaks a figure per iteration:\n```python\nfor e in range(25):\n    plt.figure()\n    "
                      "plt.plot(epochs[:e + 2], train_loss[:e + 2])\n```\nFix it so no figure is left open afterwards, "
                      "and return the number of open figures at the end.",
         "for e in range(25):\n    plt.figure()\n    plt.plot(epochs[:e + 2], train_loss[:e + 2])\n    plt.close()\n"
         "result = len(plt.get_fignums())"),
        ("plt.sca", "`fig, (a1, a2) = plt.subplots(1, 2)` then `plt.plot(epochs, train_loss)` puts the line on the "
                    "**right** plot (the current Axes). Put `train_loss` on the left one and `val_loss` on the right one "
                    "using pyplot calls only for plotting (`plt.sca` to switch). Return the number of lines on each.",
         "fig, (a1, a2) = plt.subplots(1, 2)\nplt.sca(a1)\nplt.plot(epochs, train_loss)\nplt.sca(a2)\nplt.plot(epochs, val_loss)\n"
         "result = [len(a1.get_lines()), len(a2.get_lines())]"),
        ("plt.subplot", "With pyplot's `plt.subplot(1, 2, k)` (which creates and selects the k-th cell), plot "
                        "`train_loss` in the left cell and `val_loss` in the right one, with titles 'train' and 'val'.",
         "plt.subplot(1, 2, 1)\nplt.plot(epochs, train_loss)\nplt.title('train')\nplt.subplot(1, 2, 2)\n"
         "plt.plot(epochs, val_loss)\nplt.title('val')",
         {"post": "[(fc.labels(a)['title'], fc.lines(a)) for a in fc.axes()]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def pixel_math(rng):
    w, h = rng.choice([(4, 3), (6, 2), (5, 5), (8, 4.5)])
    dpi = rng.choice([40, 50, 60, 80])
    pw, ph = rng.choice([(320, 240), (400, 200), (300, 300), (480, 270)])
    items = [
        ("get_size_inches", f"A figure of {w} × {h} inches at {dpi} dpi is ({w}·{dpi}) × ({h}·{dpi}) pixels. Create "
                            f"`fig = plt.figure(figsize=({w}, {h}), dpi={dpi})` and return its size in pixels as a list "
                            f"`[width, height]` computed from `fig.get_size_inches()` and `fig.dpi`.",
         f"fig = plt.figure(figsize=({w}, {h}), dpi={dpi})\nresult = (fig.get_size_inches() * fig.dpi).tolist()"),
        ("set_size_inches", f"Make a figure that is exactly {pw} × {ph} pixels when saved at {dpi} dpi: create `fig, ax`, "
                            f"plot `[0, 1]`, set its size with `fig.set_size_inches(...)`, save it as `out.png` at {dpi} "
                            f"dpi, and return the image shape from `plt.imread('out.png').shape`.",
         f"fig, ax = plt.subplots()\nax.plot([0, 1])\nfig.set_size_inches({pw} / {dpi}, {ph} / {dpi})\n"
         f"fig.savefig('out.png', dpi={dpi})\nresult = plt.imread('out.png').shape", {"require": ["set_size_inches"]}),
        ("savefig", f"Save the same {w} × {h} inch figure (a line `[0, 1]`) at {dpi} and at {2 * dpi} dpi as `lo.png` "
                    f"and `hi.png`, and return the two image shapes as a list.",
         f"fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot([0, 1])\nfig.savefig('lo.png', dpi={dpi})\n"
         f"fig.savefig('hi.png', dpi={2 * dpi})\nresult = [plt.imread('lo.png').shape, plt.imread('hi.png').shape]"),
    ]
    return catalog(rng, items, "", "")


@template(M, "medium")
def styles_side_by_side(rng):
    code, n, f = _wave(rng)
    items = [
        ("plt.subplots", "Draw the same two curves (`np.sin(x)` labelled 'sin', `np.cos(x)` labelled 'cos') twice, side "
                         "by side: the left Axes with pyplot (`plt.subplot(1, 2, 1)` then `plt.plot`), the right one "
                         "with the explicit API (`fig.add_subplot(1, 2, 2)` then `ax.plot`), each with a legend.",
         "fig = plt.figure()\nplt.subplot(1, 2, 1)\nplt.plot(x, np.sin(x), label='sin')\nplt.plot(x, np.cos(x), label='cos')\n"
         "plt.legend()\nax = fig.add_subplot(1, 2, 2)\nax.plot(x, np.sin(x), label='sin')\nax.plot(x, np.cos(x), label='cos')\n"
         "ax.legend()", {"post": "[(fc.lines(a, 'label'), fc.legend(a)) for a in fc.axes()]", "require": ["add_subplot"]}),
        ("plt.gca", "Write the plot with pyplot (`plt.plot(x, np.sin(x))`), then grab the Axes pyplot used with "
                    "`ax = plt.gca()` and finish with Axes methods: title 'sin', x label 'radians'.",
         "plt.plot(x, np.sin(x))\nax = plt.gca()\nax.set_title('sin')\nax.set_xlabel('radians')",
         {"post": f"{LINES}, {LAB}"}),
        ("plt.gcf", "Draw `np.sin(x)` with pyplot, then get the figure with `fig = plt.gcf()` and give it a figure-level "
                    "title 'Waves' with `fig.suptitle`.",
         "plt.plot(x, np.sin(x))\nfig = plt.gcf()\nfig.suptitle('Waves')", {"post": f"{LINES}, fc.suptitles()['suptitle']"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


# ================================================================= hard

@template(M, "hard")
def reusable_plot_function(rng):
    code, desc, n = curves(rng)
    color = rng.choice(["C0", "C1", "C2", "tab:red", "black"])
    items = [
        ("plt.gca", "Write `plot_curve(x, y, ax=None, **kwargs)` — the standard pattern for reusable plotting code: if "
                    "`ax` is None use the current Axes (`plt.gca()`), draw `ax.plot(x, y, **kwargs)`, and return the "
                    "Line2D it made. (It's checked on fresh Axes with your function.)",
         "def plot_curve(x, y, ax=None, **kwargs):\n    if ax is None:\n        ax = plt.gca()\n    line, = ax.plot(x, y, **kwargs)\n"
         "    return line",
         {"post": f"_f, (_a, _b) = plt.subplots(1, 2)\n_l = plot_curve(epochs, val_loss, ax=_a, color={color!r}, label='v')\n"
                  "plt.sca(_b)\n_m = plot_curve(epochs, train_loss)\n"
                  "(fc.lines(_a, 'color', 'label'), fc.lines(_b), type(_l).__name__, _m in _b.get_lines())",
          "require": ["gca"]}),
        ("plt.subplots", "Write `loss_figure(train, val)` that creates its **own** figure (`fig, ax = plt.subplots()`), "
                         "plots `train` and `val` labelled 'train' and 'val', adds a legend, and returns `fig`. "
                         "(Checked by calling it.)",
         "def loss_figure(train, val):\n    fig, ax = plt.subplots()\n    ax.plot(train, label='train')\n"
         "    ax.plot(val, label='val')\n    ax.legend()\n    return fig",
         {"post": "_fig = loss_figure(train_loss, val_loss)\n(fc.lines(_fig.axes[0], 'label'), fc.legend(_fig.axes[0]), "
                  "len(_fig.axes))"}),
        ("plt.close", "Write `save_plot(y, path, dpi=50)` that draws `y` on a new 3 × 2 inch figure, saves it to "
                      "`path` at `dpi`, **closes** the figure, and returns the path. Calling it many times must not "
                      "leave figures open.",
         "def save_plot(y, path, dpi=50):\n    fig, ax = plt.subplots(figsize=(3, 2))\n    ax.plot(y)\n"
         "    fig.savefig(path, dpi=dpi)\n    plt.close(fig)\n    return path",
         {"post": "_p = [save_plot(train_loss[:k + 2], f'c{k}.png') for k in range(5)]\n"
                  "(_p, len(plt.get_fignums()), plt.imread('c4.png').shape)"}),
        ("plt.subplots", "Write `compare(*curves, labels=None)` that plots any number of 1-D arrays on one new Axes "
                         "(`fig, ax = plt.subplots()`), labels them with `labels` when given (and adds a legend only "
                         "then), and returns `ax`.",
         "def compare(*curves, labels=None):\n    fig, ax = plt.subplots()\n    for i, c in enumerate(curves):\n"
         "        ax.plot(c, label=labels[i] if labels else None)\n    if labels:\n        ax.legend()\n    return ax",
         {"post": "_a = compare(train_loss, val_loss, labels=['t', 'v'])\n_b = compare(val_loss)\n"
                  "(fc.lines(_a), fc.legend(_a), fc.lines(_b), fc.legend(_b))"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def state_puzzles(rng):
    k = rng.randint(3, 6)
    j = rng.randint(1, k - 1)
    items = [
        ("plt.figure", f"Predict pyplot's state, then check it in code: create figures 1 … {k} with `plt.figure(n)`, "
                       f"activate figure {j} again with `plt.figure({j})`, close it with `plt.close()`, then call "
                       f"`plt.plot([0, 1])`. Return `[plt.gcf().number, plt.get_fignums()]` (closing the current figure "
                       f"makes the most recently active remaining one current).",
         f"for n in range(1, {k + 1}):\n    plt.figure(n)\nplt.figure({j})\nplt.close()\nplt.plot([0, 1])\n"
         f"result = [plt.gcf().number, plt.get_fignums()]"),
        ("plt.figure", f"Create figures 1 … {k} with `plt.figure()`, close figure {k} (`plt.close({k})`), then create one "
                       f"more with `plt.figure()`. A new figure gets the largest open number plus one: return its number.",
         f"for _ in range({k}):\n    plt.figure()\nplt.close({k})\nresult = plt.figure().number"),
        ("plt.close", f"Close everything with `plt.close('all')` first. Then call `plt.plot([1, 2, 3])` without "
                      f"creating a figure: pyplot makes one. Return `[len(plt.get_fignums()), plt.gcf().number, "
                      f"len(plt.gca().get_lines())]`.",
         "plt.close('all')\nplt.plot([1, 2, 3])\nresult = [len(plt.get_fignums()), plt.gcf().number, len(plt.gca().get_lines())]"),
        ("plt.subplots", f"Create `fig, axes = plt.subplots(1, {k})`, then call `plt.plot([0, 1])`. Which Axes got the "
                         f"line? Return the list of line counts per Axes.",
         f"fig, axes = plt.subplots(1, {k})\nplt.plot([0, 1])\nresult = [len(a.get_lines()) for a in axes]"),
        ("plt.sca", f"Create `fig, axes = plt.subplots(1, {k})`. Using pyplot only for drawing, plot `[0, i]` on the "
                    f"i-th Axes (i = 0 … {k - 1}) — switch with `plt.sca`. Return each Axes' line y data.",
         f"fig, axes = plt.subplots(1, {k})\nfor i, a in enumerate(axes):\n    plt.sca(a)\n    plt.plot([0, i])\n"
         "result = [[l.get_ydata().tolist() for l in a.get_lines()] for a in axes]"),
    ]
    return catalog(rng, items, "", "")


@template(M, "hard")
def export_pipeline(rng):
    code, desc, n = curves(rng)
    stem = rng.choice(["loss", "report", "curves"])
    dpis = sorted(rng.sample([30, 50, 72, 100], 2))
    items = [
        ("savefig", f"Export one figure (train and val loss against `epochs`, 4 × 3 inches) for two audiences: "
                    f"`{stem}.pdf` for the paper and `{stem}_web.png` at {dpis[0]} dpi. Return `(the PDF's first 4 bytes, "
                    f"the PNG's shape)`.",
         f"fig, ax = plt.subplots(figsize=(4, 3))\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
         f"fig.savefig({stem + '.pdf'!r})\nfig.savefig({stem + '_web.png'!r}, dpi={dpis[0]})\n"
         f"result = (open({stem + '.pdf'!r}, 'rb').read(4), plt.imread({stem + '_web.png'!r}).shape)"),
        ("savefig", f"Save the same 4 × 3 inch figure (train loss against `epochs`) at {dpis[0]} and {dpis[1]} dpi and "
                    f"return the ratio of the PNG pixel counts (high / low). (It's about ({dpis[1]}/{dpis[0]})².)",
         f"fig, ax = plt.subplots(figsize=(4, 3))\nax.plot(epochs, train_loss)\nfig.savefig('a.png', dpi={dpis[0]})\n"
         f"fig.savefig('b.png', dpi={dpis[1]})\na, b = plt.imread('a.png'), plt.imread('b.png')\n"
         f"result = (b.shape[0] * b.shape[1]) / (a.shape[0] * a.shape[1])", {"tol": 1e-9}),
        ("savefig", "Save three figures in a loop — the first 10, 20 and 30 epochs of `train_loss` — as `e10.png`, "
                    "`e20.png`, `e30.png` (dpi 30), closing each. Return the sorted file list and the open-figure count.",
         "import os\nfor k in (10, 20, 30):\n    fig, ax = plt.subplots(figsize=(3, 2))\n    ax.plot(epochs[:k], train_loss[:k])\n"
         "    fig.savefig(f'e{k}.png', dpi=30)\n    plt.close(fig)\nresult = (sorted(os.listdir()), len(plt.get_fignums()))"),
        ("savefig", "Write the figure to memory instead of a file: save `fig` (train loss) into an `io.BytesIO()` "
                    "buffer as PNG and return the first 8 bytes of the buffer (the PNG signature).",
         "import io\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nbuf = io.BytesIO()\nfig.savefig(buf, format='png')\n"
         "result = buf.getvalue()[:8]", {"require": ["BytesIO", "savefig"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def oo_refactor(rng):
    code, desc, n = curves(rng, acc=True)
    items = [
        ("plt.subplots", "Refactor this pyplot script into the explicit style, drawing the same two panels on `fig, (ax1, "
                         "ax2) = plt.subplots(1, 2, figsize=(8, 3))`:\n```python\nplt.figure(figsize=(8, 3))\n"
                         "plt.subplot(1, 2, 1); plt.plot(epochs, train_loss, label='train'); plt.plot(epochs, val_loss, "
                         "label='val')\nplt.title('loss'); plt.legend()\nplt.subplot(1, 2, 2); plt.plot(epochs, val_acc, "
                         "color='C2'); plt.title('accuracy'); plt.ylim(0, 1)\n```",
         "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))\nax1.plot(epochs, train_loss, label='train')\n"
         "ax1.plot(epochs, val_loss, label='val')\nax1.set_title('loss')\nax1.legend()\nax2.plot(epochs, val_acc, color='C2')\n"
         "ax2.set_title('accuracy')\nax2.set_ylim(0, 1)",
         {"post": "[(fc.lines(a, 'label', 'color'), fc.labels(a)['title'], fc.legend(a), fc.limits(a)['ylim']) for a in fc.axes()], fc.size()"}),
        ("twinx", "Accuracy and loss have different scales. On `fig, ax`, plot `val_loss` (color 'C0'), then make a "
                  "second y-axis sharing x with `ax2 = ax.twinx()` and plot `val_acc` on it (color 'C1'); label the y "
                  "axes 'loss' and 'accuracy'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss, color='C0')\nax.set_ylabel('loss')\nax2 = ax.twinx()\n"
         "ax2.plot(epochs, val_acc, color='C1')\nax2.set_ylabel('accuracy')",
         {"post": "[(fc.lines(a, 'color'), fc.labels(a)['ylabel']) for a in fc.axes()]"}),
        ("plt.subplots", "Draw the three curves on three stacked panels sharing the x-axis: `fig, axes = plt.subplots(3, "
                         "1, sharex=True)`, with `train_loss`, `val_loss`, `val_acc` and y labels 'train', 'val', 'acc'; "
                         "the bottom panel gets the x label 'epoch'.",
         "fig, axes = plt.subplots(3, 1, sharex=True)\nfor a, y, name in zip(axes, (train_loss, val_loss, val_acc), "
         "('train', 'val', 'acc')):\n    a.plot(epochs, y)\n    a.set_ylabel(name)\naxes[-1].set_xlabel('epoch')",
         {"post": "[(fc.lines(a), fc.labels(a)) for a in fc.axes()], fc.ax(0).get_shared_x_axes().joined(fc.ax(0), fc.ax(2))"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

@template(M, "challenge")
def pyplot_state_machine(rng):
    n, K, s = rng.randint(14, 22), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["current", "warnings", "lines"])
    limit = rng.randint(2, 3)
    setup = (f"g = np.random.default_rng({s})\nop = g.choice(4, {n}, p=[0.3, 0.2, 0.2, 0.3])\n"
             f"arg = g.integers(1, {K + 1}, {n})\nlimit = {limit}\nreport = {report!r}")
    sol = ("import heapq\nfrom collections import OrderedDict\nopen_ = OrderedDict()   # number -> lines (last = current)\n"
           "heap, warnings_ = [], 0\n\ndef create(k):\n    global warnings_\n    if len(open_) == limit:\n"
           "        warnings_ += 1\n    open_[k] = 0\n    heapq.heappush(heap, -k)\n\n"
           "def biggest():\n    while heap and -heap[0] not in open_:\n        heapq.heappop(heap)\n"
           "    return -heap[0] if heap else 0\n\nfor o, k in zip(op.tolist(), arg.tolist()):\n    if o == 0:\n"
           "        if k in open_:\n            open_.move_to_end(k)\n        else:\n            create(k)\n"
           "    elif o == 1:\n        create(biggest() + 1)\n    elif o == 2:\n        open_.pop(k, None)\n    else:\n"
           "        if not open_:\n            create(biggest() + 1)\n        cur = next(reversed(open_))\n"
           "        open_[cur] += 1\ncur = next(reversed(open_)) if open_ else 0\n"
           + {"current": "result = cur", "warnings": "result = warnings_", "lines": "result = open_.get(cur, 0)"}[report])
    tests = [f"g = np.random.default_rng({s + i})\nop = g.choice(4, 200000, p=[0.3, 0.2, 0.2, 0.3])\n"
             f"arg = g.integers(1, {kk + 1}, 200000)\nlimit = {lim}\nreport = {report!r}"
             for i, (kk, lim) in ((1, (50, 20)), (2, (2000, 300)))]
    want = {"current": "the number of the **current** figure at the end (0 if none is open)",
            "warnings": "how many times the too-many-figures warning fires",
            "lines": "how many lines the figure that is current at the end holds (0 if none is open)"}[report]
    return challenge(
        "Pyplot's bookkeeping",
        "Pyplot keeps a list of open figures ordered by when they were last activated; the last one is *current*. "
        "Replay a script of operations `op[i]` with argument `arg[i]`: **0** = `plt.figure(arg)` (activate figure "
        "`arg`, creating it if it isn't open), **1** = `plt.figure()` (create a figure numbered one more than the "
        "largest open number, or 1 if none is open), **2** = `plt.close(arg)` (close it if open; if it was current, "
        "the most recently activated remaining figure becomes current), **3** = `plt.plot(...)` (add one line to the "
        "current figure, creating a figure the way `plt.figure()` does if none is open). A figure that is closed and "
        "created again starts empty. Creating a figure while exactly `limit` figures are open fires the "
        f"too-many-figures warning (`figure.max_open_warning = limit`). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "len(op) ≤ 2·10^5, figure numbers ≤ 2000",
        speed="O(n log n): an ordered dict for activation order and a heap with lazy deletion for the largest number",
        answer_desc=" (an int)")


@template(M, "challenge")
def autoscale_ranges(rng):
    n, q, s = rng.randint(6, 10), rng.randint(4, 6), seeded(rng)
    report = rng.choice(["ylim", "height", "log"])
    setup = (f"g = np.random.default_rng({s})\nlo = np.round(g.uniform(1, 50, {n}), 2)\n"
             f"hi = np.round(lo + g.uniform(0.5, 40, {n}), 2)\na = g.integers(0, {n}, {q})\n"
             f"b = a + g.integers(0, {n}, {q}) % ({n} - a)\nreport = {report!r}")
    sol = ("def table(v, f):\n    t = [v]\n    j = 1\n    while 2 * j <= len(v):\n        p = t[-1]\n"
           "        t.append(f(p[:-j], p[j:]))\n        j *= 2\n    return t\n\n"
           "def query(t, f, a, b):\n    k = np.floor(np.log2(b - a + 1)).astype(int)\n"
           "    out = np.empty(len(a))\n    for lvl in np.unique(k):\n        m = k == lvl\n"
           "        out[m] = f(t[lvl][a[m]], t[lvl][b[m] - (1 << lvl) + 1])\n    return out\n\n"
           "L = query(table(lo, np.minimum), np.minimum, a, b)\nH = query(table(hi, np.maximum), np.maximum, a, b)\n"
           + {"log": "l, h = np.log10(L), np.log10(H)\nd = h - l\n"
                     "result = np.column_stack([10 ** (l - 0.05 * d), 10 ** (h + 0.05 * d)])",
              "ylim": "d = H - L\nresult = np.column_stack([L - 0.05 * d, H + 0.05 * d])",
              "height": "result = 1.1 * (H - L)"}[report])
    tests = [f"g = np.random.default_rng({s + i})\nlo = np.round(g.uniform(1, 50, 200000), 2)\n"
             f"hi = np.round(lo + g.uniform(0.5, 40, 200000), 2)\na = g.integers(0, 200000, 200000)\n"
             f"b = a + g.integers(0, 200000, 200000) % (200000 - a)\nreport = {report!r}" for i in (1, 2)]
    want = {"ylim": "the y limits `(bottom, top)` Matplotlib picks, as a q × 2 array",
            "height": "the height `top - bottom` of the y-axis Matplotlib picks, one per query",
            "log": "the y limits `(bottom, top)` when the Axes uses a **log** y-scale (set before plotting: margins "
                   "are then applied in log10 space), as a q × 2 array"}[report]
    return challenge(
        "Autoscale over ranges of lines",
        "Line `i` spans the y values `lo[i] … hi[i]` (`lo[i] < hi[i]`). For query `j`, someone plots lines `a[j] … "
        "b[j]` (inclusive) on a fresh Axes and lets Matplotlib autoscale: the y-axis covers the smallest `lo` to the "
        "largest `hi`, plus a margin of 5% of that span on each side (`axes.ymargin = 0.05`). Return "
        f"{want}. (`report` = {report!r}.)",
        setup, sol, tests, "number of lines, number of queries ≤ 2·10^5",
        speed="O((n + q) log n): a sparse table for range min / max", tol=1e-6)


# ================================================================= more catalogs

@template(M, "medium")
def artist_tree(rng):
    code, desc, names, vals, th, m = categories(rng)
    k = len(names)
    items = [
        (".axes", "A Figure holds Axes; an Axes holds artists. Create `fig, axes = plt.subplots(2, 2)` and return "
                  "`len(fig.axes)`.", "fig, axes = plt.subplots(2, 2)\nresult = len(fig.axes)"),
        (".bar", f"`ax.bar` returns a container of Rectangle patches. Draw `ax.bar(names, values)` and return "
                 f"`[len(bars), len(ax.patches)]`.",
         "fig, ax = plt.subplots()\nbars = ax.bar(names, values)\nresult = [len(bars), len(ax.patches)]"),
        (".texts", "Write each value above its bar with `ax.text(i, v, str(v), ha='center')` after `ax.bar(names, "
                   "values)`, and return `len(ax.texts)`.",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nfor i, v in enumerate(values):\n    ax.text(i, v, str(v), ha='center')\n"
         "result = len(ax.texts)"),
        (".collections", "A scatter plot is one PathCollection. Draw `ax.scatter(range(len(values)), values)` and return "
                         "`[len(ax.collections), len(ax.get_lines())]`.",
         "fig, ax = plt.subplots()\nax.scatter(range(len(values)), values)\nresult = [len(ax.collections), len(ax.get_lines())]"),
        ("get_children", "Draw `values` as a line on `fig, ax` and return whether that line is among `ax.get_children()`.",
         "fig, ax = plt.subplots()\nline, = ax.plot(values)\nresult = line in ax.get_children()"),
        (".bar", f"Draw `ax.bar(names, values)` and return the bar heights read back from the patches "
                 f"(`[p.get_height() for p in ax.patches]`).",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nresult = [p.get_height() for p in ax.patches]"),
        ("get_xticklabels", f"Draw `ax.bar(names, values)` and return the x tick label texts (the {k} names).",
         "fig, ax = plt.subplots()\nax.bar(names, values)\nresult = [t.get_text() for t in ax.get_xticklabels()]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def defaults_and_config(rng):
    key = rng.choice(["figure.figsize", "figure.dpi", "lines.linewidth", "axes.titlesize", "font.size"])
    items = [
        ("rcParams", f"Return the default value of the rcParam `{key!r}` (`plt.rcParams[...]`; a list for sizes).",
         f"v = plt.rcParams[{key!r}]\nresult = list(v) if isinstance(v, (list, tuple)) else v"),
        ("plt.figure", "Create `fig = plt.figure()` without arguments and return its size in inches as a list: it "
                       "comes from `rcParams['figure.figsize']`.", "fig = plt.figure()\nresult = fig.get_size_inches().tolist()"),
        ("rcParams", "Set `plt.rcParams['lines.linewidth'] = 3`, then draw `fig, ax` with `ax.plot([0, 1])` and return "
                     "the line's width. (The grader resets rcParams after each run.)",
         "plt.rcParams['lines.linewidth'] = 3\nfig, ax = plt.subplots()\nline, = ax.plot([0, 1])\nresult = line.get_linewidth()"),
        ("rcParams", "Return the default color cycle's first three colors (`plt.rcParams['axes.prop_cycle']."
                     "by_key()['color'][:3]`).", "result = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]"),
        ("get_backend", "Return `[mpl.get_backend().lower(), mpl.is_interactive()]`: this page draws off-screen, so "
                        "pyplot is not in interactive mode.", "result = [mpl.get_backend().lower(), mpl.is_interactive()]"),
        ("rcParams", "Plot two lines with `ax.plot([0, 1])` and `ax.plot([1, 0])` and return their colors: the first two "
                     "of the default cycle, 'C0' and 'C1' as hex strings.",
         "fig, ax = plt.subplots()\na, = ax.plot([0, 1])\nb, = ax.plot([1, 0])\nresult = [a.get_color(), b.get_color()]",
         {"require": ["plot"]}),
    ]
    return catalog(rng, items, "", "")


@template(M, "hard")
def debug_broken_plots(rng):
    code, desc, n = curves(rng)
    title = rng.choice(["Loss", "Training run", "Validation"])
    items = [
        ("set_title", f"This crashes with *'Text' object is not callable*:\n```python\nfig, ax = plt.subplots()\n"
                      f"ax.plot(epochs, train_loss)\nax.title({title!r})\n```\n`ax.title` is the title's Text "
                      f"object, not a function. Fix it.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_title({title!r})", {"post": f"{LINES}, {LAB}"}),
        (".plot", "This crashes with *'Figure' object has no attribute 'plot'*:\n```python\nfig, ax = plt.subplots()\n"
                  "fig.plot(epochs, train_loss)\n```\nFigures hold Axes; Axes draw. Fix it.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)", {"post": LINES}),
        ("plt.subplots", "This crashes with *'numpy.ndarray' object has no attribute 'plot'*:\n```python\nfig, ax = "
                         "plt.subplots(2)\nax.plot(epochs, train_loss)\n```\nWith several Axes you get an array. Plot "
                         "`train_loss` on the first Axes and `val_loss` on the second.",
         "fig, axes = plt.subplots(2)\naxes[0].plot(epochs, train_loss)\naxes[1].plot(epochs, val_loss)",
         {"post": "[fc.lines(a) for a in fc.axes()]"}),
        (".legend", "This draws an **empty** legend and warns *No artists with labels found*:\n```python\nfig, ax = "
                    "plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nax.legend()\n```\nFix it "
                    "so the legend says 'train' and 'val'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\nax.legend()",
         {"post": "fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["label"]}),
        ("plt.savefig", "This saves a **blank** image: the figure is closed before saving, so `savefig` makes a new "
                        "empty one.\n```python\nplt.plot(epochs, train_loss)\nplt.close()\nplt.savefig('loss.png')\n```\n"
                        "Fix the order. (Checked: `loss.png` has some non-white pixels.)",
         "plt.plot(epochs, train_loss)\nplt.savefig('loss.png')\nplt.close()",
         {"post": "bool((plt.imread('loss.png')[..., :3] < 0.9).any())", "require": ["savefig"]}),
        ("plt.subplots", "This crashes with *cannot unpack non-iterable Axes object*:\n```python\nfig, ax = plt.subplot()\n"
                         "ax.plot(epochs, val_loss)\n```\n`plt.subplot` (singular) returns one Axes. Fix it with the "
                         "function that returns `(fig, ax)`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)", {"post": LINES}),
        (".set_xlabel", "This crashes with *'Axes' object has no attribute 'xlabel'*:\n```python\nfig, ax = "
                        "plt.subplots()\nax.plot(epochs, train_loss)\nax.xlabel('epoch'); ax.ylabel('loss')\n```\n"
                        "Fix it.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.set_xlabel('epoch')\nax.set_ylabel('loss')",
         {"post": f"{LINES}, {LAB}", "require": ["set_xlabel", "set_ylabel"]}),
        ("plt.subplots", f"The loop below draws every curve on the **same** figure, because it never makes a new one:\n"
                         f"```python\nfor name, curve in [('train', train_loss), ('val', val_loss)]:\n    plt.plot(epochs, "
                         f"curve)\n    plt.savefig(f'{{name}}.png')\n```\nMake each saved file contain only its own "
                         f"curve: give each its own figure and close it. Return the number of lines drawn per figure as "
                         f"a list (count them before closing).",
         "counts = []\nfor name, curve in [('train', train_loss), ('val', val_loss)]:\n    fig, ax = plt.subplots()\n"
         "    ax.plot(epochs, curve)\n    fig.savefig(f'{name}.png')\n    counts.append(len(ax.get_lines()))\n"
         "    plt.close(fig)\nresult = counts", {"require": ["close"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def mixing_interfaces(rng):
    code, desc, n = curves(rng)
    label = rng.choice(["loss", "metrics", "lr"])
    items = [
        ("plt.figure", f"Figures can have a name instead of a number. Create `plt.figure({label!r})`, plot `train_loss` "
                       f"with pyplot, and return `[plt.get_figlabels(), plt.fignum_exists({label!r})]`.",
         f"plt.figure({label!r})\nplt.plot(epochs, train_loss)\nresult = [plt.get_figlabels(), plt.fignum_exists({label!r})]",
         {"require": ["get_figlabels"]}),
        ("add_axes", "Place an Axes by hand: `fig = plt.figure()`, then `ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))` "
                     "(left, bottom, width, height in figure fractions), and plot `val_loss` on it. Return "
                     "`[round(v, 3) for v in ax.get_position().bounds]`.",
         "fig = plt.figure()\nax = fig.add_axes((0.1, 0.1, 0.8, 0.8))\nax.plot(epochs, val_loss)\n"
         "result = [round(v, 3) for v in ax.get_position().bounds]"),
        ("plt.setp", "`plt.setp` sets a property on several artists at once. Draw `train_loss` and `val_loss` on `fig, "
                     "ax` and set **both** lines' linewidth to 3 with one `plt.setp(ax.get_lines(), linewidth=3)`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nplt.setp(ax.get_lines(), linewidth=3)",
         {"post": "fc.lines(fc.ax(), 'lw')"}),
        ("fignum_exists", "Create `fig, ax = plt.subplots()`, remember `num = fig.number`, close it, and return "
                          "`[num, plt.fignum_exists(num)]`.",
         "fig, ax = plt.subplots()\nnum = fig.number\nplt.close(fig)\nresult = [num, plt.fignum_exists(num)]"),
        ("add_subplot", "Build a 1 × 2 figure piece by piece: `fig = plt.figure()`, then `fig.add_subplot(1, 2, 1)` for "
                        "`train_loss` and `fig.add_subplot(1, 2, 2)` for `val_loss`. Return `fc`-style: the number of "
                        "Axes and each one's line count.",
         "fig = plt.figure()\na = fig.add_subplot(1, 2, 1)\na.plot(epochs, train_loss)\nb = fig.add_subplot(1, 2, 2)\n"
         "b.plot(epochs, val_loss)\nresult = [len(fig.axes), [len(x.get_lines()) for x in fig.axes]]"),
        ("plt.gcf", "Mix both styles on purpose: draw with pyplot (`plt.plot(epochs, train_loss)`), then fetch the "
                    "objects (`fig, ax = plt.gcf(), plt.gca()`) and save through the figure (`fig.savefig('mixed.png', "
                    "dpi=40)`). Return `[len(ax.get_lines()), plt.imread('mixed.png').shape[2]]`.",
         "plt.plot(epochs, train_loss)\nfig, ax = plt.gcf(), plt.gca()\nfig.savefig('mixed.png', dpi=40)\n"
         "result = [len(ax.get_lines()), plt.imread('mixed.png').shape[2]]"),
    ]
    return catalog(rng, items, code, f"{desc}.")

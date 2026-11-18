"""5.11 Styles & rcParams (the grader resets rcParams after every run, so problems may change styles freely)."""

from ._common import P, catalog, challenge, curves, lit, seeded, signal, template

M = "5.11"

# ---------------------------------------------------------------- check code (post): rcParams and styled artists
_PRE = '''def _v(v):
    if hasattr(v, 'by_key'):
        return {k: [_v(x) for x in xs] for k, xs in v.by_key().items()}
    if isinstance(v, (tuple, list)):
        return [_v(x) for x in v]
    if hasattr(v, 'tolist'):
        return v.tolist()
    return v

def _rc(*keys):
    return {k: _v(plt.rcParams[k]) for k in keys}

def _ax(a, *keys):
    out = {}
    for k in keys:
        if k == 'face':
            out[k] = fc.hexcolor(a.get_facecolor())
        elif k == 'grid':
            out[k] = fc.grid_on(a)
        elif k == 'spines':
            out[k] = fc.spines(a)
        elif k == 'titlesize':
            out[k] = round(a.title.get_fontsize(), 2)
        elif k == 'labelsize':
            out[k] = round(a.xaxis.label.get_fontsize(), 2)
        elif k == 'ticksize':
            out[k] = round(a.xaxis.get_major_ticks()[0].label1.get_fontsize(), 2)
        elif k == 'family':
            out[k] = a.title.get_fontfamily()
        elif k == 'colors':
            out[k] = [fc.hexcolor(l.get_color()) for l in a.get_lines()]
        elif k == 'ls':
            out[k] = [l.get_linestyle() for l in a.get_lines()]
        elif k == 'lw':
            out[k] = [round(l.get_linewidth(), 3) for l in a.get_lines()]
        elif k == 'markers':
            out[k] = [l.get_marker() for l in a.get_lines()]
        elif k == 'ms':
            out[k] = [round(l.get_markersize(), 3) for l in a.get_lines()]
        elif k == 'sketch':
            out[k] = [l.get_sketch_params() for l in a.get_lines()]
        elif k == 'edge':
            out[k] = fc.hexcolor(a.spines['left'].get_edgecolor())
        elif k == 'gridalpha':
            out[k] = a.xaxis.get_major_ticks()[0].gridline.get_alpha()
        elif k == 'title':
            out[k] = a.get_title()
        elif k == 'legendframe':
            out[k] = None if a.get_legend() is None else a.get_legend().get_frame_on()
    return out
'''


def post(expr: str) -> str:
    return _PRE + expr


PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


# ================================================================= lite

@template(M, "lite")
def rc_basics(rng):
    code, desc, n = signal(rng)
    key = rng.choice(["font.size", "lines.linewidth", "figure.dpi", "axes.titlesize", "legend.fontsize"])
    fs = rng.choice([9, 11, 12, 14])
    dpi = rng.choice([80, 110, 120])
    lw = rng.choice([1.5, 2, 2.5, 3])
    ls = rng.choice(["--", ":", "-."])
    w, h = rng.choice([(6, 4), (5, 3), (8, 3), (4, 4)])
    pal = PAL[:rng.randint(3, 5)]
    items = [
        ("rcParams", f"`plt.rcParams` is a dict of every default. Return the current value of {key!r}.",
         f"result = plt.rcParams[{key!r}]"),
        ("rcParams", f"Set the default font size to {fs} pt through `plt.rcParams`. (The grader resets the defaults "
                     f"after each run.)", f"plt.rcParams['font.size'] = {fs}", {"post": post("_rc('font.size')")}),
        ("rcParams", "Remove the top and right spines from all future plots: set `'axes.spines.top'` and "
                     "`'axes.spines.right'` to False in `plt.rcParams`.",
         "plt.rcParams['axes.spines.top'] = False\nplt.rcParams['axes.spines.right'] = False",
         {"post": post("_rc('axes.spines.top', 'axes.spines.right', 'axes.spines.left')")}),
        ("update", f"Set several defaults in **one** call with `mpl.rcParams.update({{...}})`: 'figure.dpi' {dpi}, "
                   f"'lines.linewidth' {lw} and 'axes.grid' True.",
         f"mpl.rcParams.update({{'figure.dpi': {dpi}, 'lines.linewidth': {lw}, 'axes.grid': True}})",
         {"post": post("_rc('figure.dpi', 'lines.linewidth', 'axes.grid')")}),
        ("plt.rc", f"`plt.rc(group, **settings)` sets a whole group at once: make lines {lw} wide and styled {ls!r} by "
                   f"default with `plt.rc('lines', ...)`.", f"plt.rc('lines', linewidth={lw}, linestyle={ls!r})",
         {"post": post("_rc('lines.linewidth', 'lines.linestyle')")}),
        ("cycler", f"Make {pal} the default color cycle: `plt.rc('axes', prop_cycle=plt.cycler(color=[...]))`.",
         f"plt.rc('axes', prop_cycle=plt.cycler(color={pal}))", {"post": post("_rc('axes.prop_cycle')")}),
        ("rcdefaults", f"Set `plt.rcParams['font.size'] = {fs + 10}`, then undo every change with `mpl.rcdefaults()` "
                       f"and return the font size you get back.",
         f"plt.rcParams['font.size'] = {fs + 10}\nmpl.rcdefaults()\nresult = plt.rcParams['font.size']"),
        ("rcParams", f"Defaults apply to artists made **after** the change. Set `plt.rcParams['lines.linewidth'] = {lw}`, "
                     f"then create `fig, ax` and plot `y` against `x`.",
         f"plt.rcParams['lines.linewidth'] = {lw}\nfig, ax = plt.subplots()\nax.plot(x, y)", {"post": post("_ax(fc.ax(), 'lw')")}),
        ("rcParams", f"Set the default figure size to {w} × {h} inches (`'figure.figsize'`) and create a figure with "
                     f"`plt.figure()` (no `figsize=`).", f"plt.rcParams['figure.figsize'] = ({w}, {h})\nfig = plt.figure()",
         {"post": "fc.size()"}),
        ("rcParamsDefault", f"`mpl.rcParamsDefault` keeps Matplotlib's built-in defaults even after you change "
                            f"`rcParams`. Set `plt.rcParams[{key!r}]` to 20 and return `[plt.rcParams[{key!r}], "
                            f"mpl.rcParamsDefault[{key!r}]]`.",
         f"plt.rcParams[{key!r}] = 20\nresult = [plt.rcParams[{key!r}], mpl.rcParamsDefault[{key!r}]]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def style_basics(rng):
    code, desc, n = signal(rng)
    st = rng.choice(["ggplot", "bmh", "fivethirtyeight", "Solarize_Light2", "grayscale"])
    lw = rng.choice([3, 4, 5])
    sc = rng.choice([1.5, 2, 3])
    items = [
        ("available", f"`plt.style.available` lists the built-in style sheets. Return whether {st!r} is one of them.",
         f"result = {st!r} in plt.style.available"),
        ("available", "Return the names in `plt.style.available` that start with 'seaborn', sorted, as a list.",
         "result = sorted(s for s in plt.style.available if s.startswith('seaborn'))"),
        ("use", f"Switch every following plot to the {st!r} style with `plt.style.use`.", f"plt.style.use({st!r})",
         {"post": post("_rc('axes.facecolor', 'axes.grid', 'axes.prop_cycle')")}),
        ("use", "Apply the 'dark_background' style, then create `fig, ax` and plot `y` against `x`.",
         "plt.style.use('dark_background')\nfig, ax = plt.subplots()\nax.plot(x, y)",
         {"post": post("_ax(fc.ax(), 'face', 'colors'), fc.hexcolor(fc.fig().get_facecolor())")}),
        ("context", f"Use {st!r} for **one** figure only: inside `with plt.style.context({st!r}):` create `fig, ax` and "
                    f"plot `y` against `x`. (Checked: the Axes has the style; afterwards the defaults are back.)",
         f"with plt.style.context({st!r}):\n    fig, ax = plt.subplots()\n    ax.plot(x, y)",
         {"post": post("_ax(fc.ax(), 'face', 'colors'), _rc('axes.facecolor')")}),
        ("use", "Styles can be combined; later ones win where they overlap. Apply `['ggplot', 'tableau-colorblind10']` "
                "in one `plt.style.use` call (ggplot's gray background, the colorblind-safe colors).",
         "plt.style.use(['ggplot', 'tableau-colorblind10'])", {"post": post("_rc('axes.facecolor', 'axes.prop_cycle')")}),
        ("use", f"Apply {st!r}, then go back to Matplotlib's defaults with `plt.style.use('default')`.",
         f"plt.style.use({st!r})\nplt.style.use('default')", {"post": post("_rc('axes.facecolor', 'axes.grid', 'axes.prop_cycle')")}),
        ("xkcd", "Draw in the hand-drawn xkcd look: inside `with plt.xkcd():` create `fig, ax` and plot `y` against `x`. "
                 "(Checked: the line's sketch parameters, and that the look is gone afterwards.)",
         "with plt.xkcd():\n    fig, ax = plt.subplots()\n    ax.plot(x, y)",
         {"post": post("_ax(fc.ax(), 'sketch'), _rc('path.sketch')")}),
        ("rc_context", f"`plt.rc_context({{...}})` changes rcParams temporarily. Inside `with plt.rc_context("
                       f"{{'lines.linewidth': {lw}}}):` create `fig, ax` and plot `y` against `x`.",
         f"with plt.rc_context({{'lines.linewidth': {lw}}}):\n    fig, ax = plt.subplots()\n    ax.plot(x, y)",
         {"post": post("_ax(fc.ax(), 'lw'), _rc('lines.linewidth')")}),
        ("matplotlib_fname", "`mpl.matplotlib_fname()` is the path of the matplotlibrc file in use. Return its file "
                             "name (`os.path.basename`).",
         "import os\nresult = os.path.basename(mpl.matplotlib_fname())"),
        ("get_configdir", "`mpl.get_configdir()` is the folder for your own matplotlibrc and style files. Return "
                          "whether it is an absolute path (`os.path.isabs`).",
         "import os\nresult = os.path.isabs(mpl.get_configdir())"),
        ("xkcd", f"`plt.xkcd(scale=...)` sets how wobbly the lines are. Inside `with plt.xkcd(scale={sc}):` create "
                 f"`fig, ax` and plot `y` against `x`.",
         f"with plt.xkcd(scale={sc}):\n    fig, ax = plt.subplots()\n    ax.plot(x, y)", {"post": post("_ax(fc.ax(), 'sketch')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def rc_effects(rng):
    code, desc, n = curves(rng)
    lw = rng.choice([1.5, 2.5, 3])
    ts, ls_ = rng.choice([(13, 11), (16, 12), (11, 9)])
    ga = rng.choice([0.2, 0.3, 0.5])
    w, h = rng.choice([(6, 4), (5, 3), (7, 2.5)])
    fs = rng.choice([12, 14, 8])
    mk, ms = rng.choice([("o", 4), ("s", 3), (".", 6)])
    face = rng.choice(["#f0f0f0", "whitesmoke", "lavender"])
    items = [
        ("rcParams", f"Set the default line width to {lw}, then create `fig, ax` and plot `train_loss` and `val_loss` "
                     f"against `epochs`.",
         f"plt.rcParams['lines.linewidth'] = {lw}\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'lw')")}),
        ("update", f"With one `mpl.rcParams.update`, make titles {ts} pt ('axes.titlesize') and axis labels {ls_} pt "
                   f"('axes.labelsize'); then create `fig, ax`, plot `train_loss`, and set the title 'loss' and the x "
                   f"label 'epoch'.",
         f"mpl.rcParams.update({{'axes.titlesize': {ts}, 'axes.labelsize': {ls_}}})\nfig, ax = plt.subplots()\n"
         f"ax.plot(epochs, train_loss)\nax.set_title('loss')\nax.set_xlabel('epoch')",
         {"post": post("_ax(fc.ax(), 'titlesize', 'labelsize', 'title')")}),
        ("plt.rc", f"Turn the grid on for all Axes (`plt.rc('axes', grid=True)`) and make it faint (`plt.rc('grid', "
                   f"alpha={ga})`); then create `fig, ax` and plot `val_loss`.",
         f"plt.rc('axes', grid=True)\nplt.rc('grid', alpha={ga})\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'grid', 'gridalpha')")}),
        ("rcParams", "Make legends frameless by default (`'legend.frameon'`), then create `fig, ax`, plot both losses "
                     "labelled 'train' and 'val', and add a legend.",
         "plt.rcParams['legend.frameon'] = False\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss, label='train')\n"
         "ax.plot(epochs, val_loss, label='val')\nax.legend()", {"post": post("_ax(fc.ax(), 'legendframe')")}),
        ("rcParams", f"Make {w} × {h} inches the default figure size and create `fig, ax = plt.subplots()` (no "
                     f"`figsize=`). Return `fig.get_size_inches().tolist()`.",
         f"plt.rcParams['figure.figsize'] = ({w}, {h})\nfig, ax = plt.subplots()\nresult = fig.get_size_inches().tolist()"),
        ("rcParams", f"Most text sizes are relative to `'font.size'`. Set it to {fs}, create `fig, ax`, plot `val_loss`, "
                     f"set a title, and return `[tick label size, title size]` (`ax.xaxis.get_major_ticks()[0]."
                     f"label1.get_fontsize()`, `ax.title.get_fontsize()`): 'medium' and 'large' of {fs} pt.",
         f"plt.rcParams['font.size'] = {fs}\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.set_title('val')\n"
         f"result = [ax.xaxis.get_major_ticks()[0].label1.get_fontsize(), ax.title.get_fontsize()]"),
        ("plt.rc", f"Give every line markers by default: `plt.rc('lines', marker={mk!r}, markersize={ms})`; then create "
                   f"`fig, ax` and plot `train_loss`.",
         f"plt.rc('lines', marker={mk!r}, markersize={ms})\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_ax(fc.ax(), 'markers', 'ms')")}),
        ("plt.rc", f"Style the Axes box: `plt.rc('axes', facecolor={face!r}, edgecolor='gray')`, then create `fig, ax`.",
         f"plt.rc('axes', facecolor={face!r}, edgecolor='gray')\nfig, ax = plt.subplots()",
         {"post": post("_ax(fc.ax(), 'face', 'edge')")}),
        ("update", "Drop the top and right spines for every plot with one `mpl.rcParams.update`, then create a 1 × 2 "
                   "figure (`fig, axs = plt.subplots(1, 2)`).",
         "mpl.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})\nfig, axs = plt.subplots(1, 2)",
         {"post": post("[_ax(a, 'spines') for a in fc.axes()]")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def scoped_styles(rng):
    code, desc, n = curves(rng)
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2", "seaborn-v0_8-darkgrid"])
    lw = rng.choice([2, 3, 4])
    sc = rng.choice([1.5, 2, 3])
    items = [
        ("context", f"Draw two figures: the first (`fig1, ax1`, plotting `train_loss`) inside `with "
                    f"plt.style.context({st!r}):`, the second (`fig2, ax2`, plotting `val_loss`) after the block. "
                    f"Return the two Axes background colors as hex (`mpl.colors.to_hex(ax.get_facecolor())`).",
         f"with plt.style.context({st!r}):\n    fig1, ax1 = plt.subplots()\n    ax1.plot(epochs, train_loss)\n"
         f"fig2, ax2 = plt.subplots()\nax2.plot(epochs, val_loss)\nresult = [mpl.colors.to_hex(a.get_facecolor()) for a in (ax1, ax2)]"),
        ("rc_context", "Paper look for one figure: inside `with plt.rc_context({'font.family': 'serif', "
                       "'mathtext.fontset': 'cm'}):` create `fig, ax`, plot `train_loss`, and set the title "
                       "`r'$\\alpha$ paper style'`.",
         "with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm'}):\n    fig, ax = plt.subplots()\n"
         "    ax.plot(epochs, train_loss)\n    ax.set_title(r'$\\alpha$ paper style')",
         {"post": post("_ax(fc.ax(), 'family', 'title'), _rc('font.family', 'mathtext.fontset')")}),
        ("rc_context", f"`plt.rc_context` also works as a **decorator**. Write `plot_thick(ax, y)` decorated with "
                       f"`@plt.rc_context({{'lines.linewidth': {lw}}})` that plots `y` on `ax` and returns the line. "
                       f"(Checked by calling it; outside the call the default width is back.)",
         f"@plt.rc_context({{'lines.linewidth': {lw}}})\ndef plot_thick(ax, y):\n    line, = ax.plot(y)\n    return line",
         {"post": post("_f, _a = plt.subplots()\n_l = plot_thick(_a, val_loss)\n_a.plot(train_loss)\n"
                       "_ax(_a, 'lw'), _l is _a.get_lines()[0]")}),
        ("context", f"Nest the two: inside `with plt.style.context({st!r}):` open `with plt.rc_context({{'axes.facecolor': "
                    f"'white'}}):` and create `fig, ax` there, plotting `val_loss` — the style's other settings with a "
                    f"white background.",
         f"with plt.style.context({st!r}):\n    with plt.rc_context({{'axes.facecolor': 'white'}}):\n"
         f"        fig, ax = plt.subplots()\n        ax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'grid', 'colors')")}),
        ("context", f"`plt.style.context` takes a list mixing style names and dicts. Inside `with "
                    f"plt.style.context(['dark_background', {{'lines.linewidth': {lw}}}]):` create `fig, ax` and plot "
                    f"both losses.",
         f"with plt.style.context(['dark_background', {{'lines.linewidth': {lw}}}]):\n    fig, ax = plt.subplots()\n"
         f"    ax.plot(epochs, train_loss)\n    ax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'lw', 'colors')")}),
        ("copy", f"Save and restore by hand: keep `old = plt.rcParams.copy()`, set 'lines.linewidth' to {lw}, plot "
                 f"`train_loss` on a new `fig, ax`, then restore with `plt.rcParams.update(old)` and plot `val_loss` "
                 f"on the same Axes.",
         f"old = plt.rcParams.copy()\nplt.rcParams['lines.linewidth'] = {lw}\nfig, ax = plt.subplots()\n"
         f"ax.plot(epochs, train_loss)\nplt.rcParams.update(old)\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'lw'), _rc('lines.linewidth')")}),
        ("xkcd", f"Inside `with plt.xkcd(scale={sc}, length=50):` create `fig, ax` and plot `train_loss`; then, after "
                 f"the block, plot `val_loss` on the same Axes (no wobble).",
         f"with plt.xkcd(scale={sc}, length=50):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, train_loss)\n"
         f"ax.plot(epochs, val_loss)", {"post": post("_ax(fc.ax(), 'sketch')")}),
        ("rcdefaults", f"Two ways back to the defaults. Apply {st!r} and record `plt.rcParams['axes.facecolor']`; reset "
                       f"with `mpl.rcdefaults()` and record it again; apply {st!r} again, reset with "
                       f"`plt.style.use('default')` and record it a third time. Return the three values.",
         f"out = []\nplt.style.use({st!r})\nout.append(plt.rcParams['axes.facecolor'])\nmpl.rcdefaults()\n"
         f"out.append(plt.rcParams['axes.facecolor'])\nplt.style.use({st!r})\nplt.style.use('default')\n"
         f"out.append(plt.rcParams['axes.facecolor'])\nresult = out"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def cycler_tasks(rng):
    code, desc, n = signal(rng)
    k = rng.randint(3, 4)
    pal = PAL[:k]
    lss = ["-", "--", ":", "-."][:k]
    m = rng.choice(["viridis", "plasma", "cividis"])
    items = [
        ("cycler", f"Set the global color cycle to {pal} with `plt.rc('axes', prop_cycle=plt.cycler(color=...))`, "
                   f"then plot `y + i` for i = 0 … {k} ({k + 1} lines) on a new `fig, ax`: the cycle wraps around.",
         f"plt.rc('axes', prop_cycle=plt.cycler(color={pal}))\nfig, ax = plt.subplots()\nfor i in range({k + 1}):\n"
         f"    ax.plot(x, y + i)", {"post": post("_ax(fc.ax(), 'colors')")}),
        ("cycler", f"Adding cyclers zips them (equal lengths): set `plt.rcParams['axes.prop_cycle']` to "
                   f"`plt.cycler(color={pal}) + plt.cycler(linestyle={lss})`, then plot `y + i` for i = 0 … {k - 1}.",
         f"plt.rcParams['axes.prop_cycle'] = plt.cycler(color={pal}) + plt.cycler(linestyle={lss})\n"
         f"fig, ax = plt.subplots()\nfor i in range({k}):\n    ax.plot(x, y + i)",
         {"post": post("_ax(fc.ax(), 'colors', 'ls')")}),
        ("cycler", "Multiplying cyclers gives every combination: on `fig, ax` set `ax.set_prop_cycle(plt.cycler("
                   "linestyle=['-', '--']) * plt.cycler(color=['k', 'r']))` and plot `y + i` for i = 0 … 3.",
         "fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(linestyle=['-', '--']) * plt.cycler(color=['k', 'r']))\n"
         "for i in range(4):\n    ax.plot(x, y + i)", {"post": post("_ax(fc.ax(), 'colors', 'ls')")}),
        ("set_prop_cycle", f"A cycle can belong to one Axes only. Create `fig, (ax1, ax2) = plt.subplots(1, 2)`, give "
                           f"`ax1` the cycle `plt.cycler(color={pal})`, and plot `y` and `-y` on **both** Axes.",
         f"fig, (ax1, ax2) = plt.subplots(1, 2)\nax1.set_prop_cycle(plt.cycler(color={pal}))\nfor a in (ax1, ax2):\n"
         f"    a.plot(x, y)\n    a.plot(x, -y)", {"post": post("[_ax(a, 'colors') for a in fc.axes()]")}),
        ("colormaps", f"Build the default cycle from a colormap: `plt.rc('axes', prop_cycle=plt.cycler(color="
                      f"mpl.colormaps[{m!r}](np.linspace(0, 1, {k}))))`, then plot `y * i` for i = 1 … {k}.",
         f"plt.rc('axes', prop_cycle=plt.cycler(color=mpl.colormaps[{m!r}](np.linspace(0, 1, {k}))))\n"
         f"fig, ax = plt.subplots()\nfor i in range(1, {k + 1}):\n    ax.plot(x, y * i)",
         {"post": post("_ax(fc.ax(), 'colors')")}),
        ("library", "`plt.style.library` maps style names to their settings. Return the color list of the "
                    "'tableau-colorblind10' style's prop cycle (`['axes.prop_cycle'].by_key()['color']`).",
         "result = plt.style.library['tableau-colorblind10']['axes.prop_cycle'].by_key()['color']"),
        ("set_prop_cycle", f"Cycle markers along with colors: on `fig, ax` set `ax.set_prop_cycle(color={pal[:3]}, "
                   f"marker=['o', 's', '^'])` (keywords zip like `+`), then plot `y + i` for i = 0 … 2.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color={pal[:3]}, marker=['o', 's', '^'])\nfor i in range(3):\n"
         f"    ax.plot(x, y + i)", {"post": post("_ax(fc.ax(), 'colors', 'markers')")}),
        ("cycler", f"Return `[len(c), c.keys, list(c)[{k}]]` for `c = plt.cycler(color={pal}) * plt.cycler(lw=[1, 2])`, "
                   f"with `c.keys` as a sorted list.",
         f"c = plt.cycler(color={pal}) * plt.cycler(lw=[1, 2])\nresult = [len(c), sorted(c.keys), list(c)[{k}]]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def mplstyle_files(rng):
    code, desc, n = curves(rng)
    w, h = rng.choice([(5, 3), (4, 3), (6, 2.5)])
    lw = rng.choice([1.5, 2, 2.5])
    fs = rng.choice([8, 9, 11])
    st = rng.choice(["ggplot", "bmh", "fivethirtyeight", "Solarize_Light2"])
    paper = "font.family: serif\naxes.spines.top: False\naxes.spines.right: False\n"
    write = f"open('paper.mplstyle', 'w').write({paper!r})\n"
    items = [
        ("use", "A style sheet is a text file of `key: value` lines. Write `paper.mplstyle` with serif fonts and no top "
                "or right spines (`font.family: serif`, `axes.spines.top: False`, `axes.spines.right: False`), apply "
                "it with `plt.style.use('./paper.mplstyle')`, and create `fig, ax`.",
         write + "plt.style.use('./paper.mplstyle')\nfig, ax = plt.subplots()",
         {"post": post("_rc('font.family', 'axes.spines.top', 'axes.spines.right'), _ax(fc.ax(), 'spines')")}),
        ("context", f"Write `slide.mplstyle` with `figure.figsize: {w}, {h}` and `lines.linewidth: {lw}`, and use it for "
                    f"one figure only (`with plt.style.context('./slide.mplstyle'):`), plotting `train_loss` on "
                    f"`fig, ax`.",
         f"open('slide.mplstyle', 'w').write('figure.figsize: {w}, {h}\\nlines.linewidth: {lw}\\n')\n"
         f"with plt.style.context('./slide.mplstyle'):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, train_loss)",
         {"post": post("fc.size(), _ax(fc.ax(), 'lw'), _rc('lines.linewidth')")}),
        ("rc_params_from_file", "Write the paper style above to `paper.mplstyle` and read it back as a dict with "
                                "`mpl.rc_params_from_file('paper.mplstyle', use_default_template=False)` (only the "
                                "keys in the file). Return `dict(...)`.",
         write + "result = dict(mpl.rc_params_from_file('paper.mplstyle', use_default_template=False))"),
        ("use", f"Compose a file and a dict: write the paper style to `paper.mplstyle`, then `plt.style.use(["
                f"'./paper.mplstyle', {{'font.size': {fs}}}])`.",
         write + f"plt.style.use(['./paper.mplstyle', {{'font.size': {fs}}}])",
         {"post": post("_rc('font.family', 'axes.spines.top', 'font.size')")}),
        ("library", f"Look inside a built-in style without applying it: return "
                    f"`plt.style.library[{st!r}]['axes.facecolor']`.", f"result = plt.style.library[{st!r}]['axes.facecolor']"),
        ("library", f"Return the sorted list of rcParams keys the {st!r} style sets (`plt.style.library[{st!r}]`).",
         f"result = sorted(plt.style.library[{st!r}])"),
        ("write", f"Generate a style file from a dict: for `settings = {{'axes.grid': True, 'grid.alpha': 0.3, "
                  f"'lines.linewidth': {lw}}}` write one `key: value` line per item to `grid.mplstyle`, apply it, and "
                  f"create `fig, ax` with `train_loss`.",
         f"settings = {{'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': {lw}}}\n"
         f"with open('grid.mplstyle', 'w') as f:\n    for k, v in settings.items():\n        f.write(f'{{k}}: {{v}}\\n')\n"
         f"plt.style.use('./grid.mplstyle')\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_rc('axes.grid', 'grid.alpha', 'lines.linewidth'), _ax(fc.ax(), 'grid', 'lw')")}),
        ("get_configdir", "Your own style sheets go in a `stylelib` folder under `mpl.get_configdir()`; a file "
                          "`stylelib/paper.mplstyle` there is then found by name (`plt.style.use('paper')`). Build that "
                          "path with `os.path.join` and return it relative to the config folder (`os.path.relpath`).",
         "import os\npath = os.path.join(mpl.get_configdir(), 'stylelib', 'paper.mplstyle')\n"
         "result = os.path.relpath(path, mpl.get_configdir())"),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= medium

GUIDE_RC = ("{'axes.titlesize': 13, 'axes.labelsize': 11, 'legend.frameon': False, 'lines.linewidth': 2, "
            "'axes.grid': True, 'grid.alpha': 0.3, 'figure.figsize': (6, 4), 'savefig.bbox': 'tight', 'savefig.dpi': 200}")


@template(M, "medium")
def paper_figures(rng):
    code, desc, n = curves(rng, acc=True)
    a, b = rng.choice([("train", "val"), ("training", "validation")])
    fs = rng.choice([14, 16, 18])
    dpi = rng.choice([30, 40])
    lw, ms = rng.choice([(0.8, 3), (1, 2), (0.6, 2.5)])
    two = (f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, label={a!r})\nax.plot(epochs, val_loss, label={b!r})\n")
    items = [
        ("update", f"Set the guide's paper defaults with one `mpl.rcParams.update({GUIDE_RC})`, then create `fig, ax`, "
                   f"plot the two losses labelled {a!r} and {b!r}, title 'Loss', x label 'epoch', and a legend.",
         f"mpl.rcParams.update({GUIDE_RC})\n" + two + "ax.set_title('Loss')\nax.set_xlabel('epoch')\nax.legend()",
         {"post": post("_ax(fc.ax(), 'titlesize', 'labelsize', 'legendframe', 'lw', 'grid', 'gridalpha'), fc.size(), "
                       "_rc('savefig.bbox', 'savefig.dpi')")}),
        ("rcParams", f"Paper look with a colorblind-safe cycle: set `'font.family'` to 'serif', drop the top and right "
                     f"spines, and set the cycle to {PAL[:4]} (all through `plt.rcParams`); then plot the two losses on "
                     f"`fig, ax`.",
         f"plt.rcParams['font.family'] = 'serif'\nplt.rcParams['axes.spines.top'] = False\nplt.rcParams['axes.spines.right'] = False\n"
         f"plt.rcParams['axes.prop_cycle'] = plt.cycler(color={PAL[:4]})\n" + two,
         {"post": post("_ax(fc.ax(), 'spines', 'colors', 'family')")}),
        ("rc_context", f"Use rc defaults for saving: inside `with plt.rc_context({{'savefig.dpi': {dpi}, 'savefig.bbox': "
                       f"'tight'}}):` draw the two losses on a 4 × 3 inch `fig, ax` and `fig.savefig('loss.png')` "
                       f"without passing dpi or bbox. Return `plt.imread('loss.png').shape`.",
         f"with plt.rc_context({{'savefig.dpi': {dpi}, 'savefig.bbox': 'tight'}}):\n    fig, ax = plt.subplots(figsize=(4, 3))\n"
         f"    ax.plot(epochs, train_loss)\n    ax.plot(epochs, val_loss)\n    fig.savefig('loss.png')\n"
         f"result = plt.imread('loss.png').shape", {"post": post("_, _rc('savefig.dpi')")}),
        ("rcParams", f"Slides need big text: set `'font.size'` to {fs}, then create `fig, ax`, plot `val_acc` and give it "
                     f"the title 'accuracy' and the x label 'epoch'. (Checked: tick, label and title sizes follow.)",
         f"plt.rcParams['font.size'] = {fs}\nfig, ax = plt.subplots()\nax.plot(epochs, val_acc)\nax.set_title('accuracy')\n"
         f"ax.set_xlabel('epoch')", {"post": post("_ax(fc.ax(), 'ticksize', 'labelsize', 'titlesize')")}),
        ("plt.rc", f"Dense plots want thin lines and small markers: `plt.rc('lines', linewidth={lw}, marker='o', "
                   f"markersize={ms})`; then plot both losses on `fig, ax`.",
         f"plt.rc('lines', linewidth={lw}, marker='o', markersize={ms})\n" + two,
         {"post": post("_ax(fc.ax(), 'lw', 'markers', 'ms')")}),
        ("update", "Left-aligned bold titles everywhere: update `'axes.titlelocation'` to 'left' and `'axes.titleweight'` "
                   "to 'bold', then create `fig, ax`, plot `val_loss` and call `ax.set_title('validation')` (no `loc=`).",
         "mpl.rcParams.update({'axes.titlelocation': 'left', 'axes.titleweight': 'bold'})\nfig, ax = plt.subplots()\n"
         "ax.plot(epochs, val_loss)\nax.set_title('validation')",
         {"post": "fc.titles(), fc.ax()._left_title.get_fontweight()"}),
        ("rc_context", f"Transparent exports by default: inside `with plt.rc_context({{'savefig.transparent': True}}):` "
                       f"draw `train_loss` on a 3 × 2 inch `fig, ax` and save `t.png` at {dpi} dpi. Return the alpha of "
                       f"the image's top-left pixel (`plt.imread('t.png')[0, 0, 3]`) as a float.",
         f"with plt.rc_context({{'savefig.transparent': True}}):\n    fig, ax = plt.subplots(figsize=(3, 2))\n"
         f"    ax.plot(epochs, train_loss)\n    fig.savefig('t.png', dpi={dpi})\nresult = float(plt.imread('t.png')[0, 0, 3])",
         {"post": post("_, _rc('savefig.transparent')")}),
        ("update", "Inward ticks on all four sides (a common journal look): update `'xtick.direction'` and "
                   "`'ytick.direction'` to 'in' and `'xtick.top'`, `'ytick.right'` to True; then plot `val_loss` on "
                   "`fig, ax`.",
         "mpl.rcParams.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})\n"
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)",
         {"post": "_t = [fc.ax().xaxis.get_major_ticks()[0], fc.ax().yaxis.get_major_ticks()[0]]\n"
                  "[(t._tickdir, t.tick2line.get_visible()) for t in _t]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def style_gallery(rng):
    code, desc, n = curves(rng)
    styles = rng.sample(["ggplot", "bmh", "fivethirtyeight", "Solarize_Light2", "seaborn-v0_8-whitegrid", "grayscale"], 3)
    st = styles[0]
    items = [
        ("context", f"Compare styles side by side in **one** figure: `fig = plt.figure(figsize=(9, 3))`, then for each "
                    f"style i in {styles} create subplot i + 1 of a 1 × 3 grid **inside** `with "
                    f"plt.style.context(style):` (`fig.add_subplot(1, 3, i + 1)`), plot both losses and title it with "
                    f"the style name.",
         f"fig = plt.figure(figsize=(9, 3))\nfor i, style in enumerate({styles}):\n    with plt.style.context(style):\n"
         f"        ax = fig.add_subplot(1, 3, i + 1)\n        ax.plot(epochs, train_loss)\n        ax.plot(epochs, val_loss)\n"
         f"        ax.set_title(style)", {"post": post("[_ax(a, 'face', 'colors', 'title') for a in fc.axes()]")}),
        ("context", f"Make one **figure** per style in {styles} (inside `with plt.style.context(style):` call "
                    f"`fig, ax = plt.subplots()` and plot `val_loss`), keeping all three figures open.",
         f"for style in {styles}:\n    with plt.style.context(style):\n        fig, ax = plt.subplots()\n"
         f"        ax.plot(epochs, val_loss)", {"post": post("[_ax(f.axes[0], 'face', 'grid', 'colors') for f in fc.figs()]")}),
        ("use", "Slides on a dark projector: apply 'dark_background', then create `fig, ax`, plot both losses and set "
                "the title 'loss'. (Checked: figure and Axes background, title color, line colors.)",
         "plt.style.use('dark_background')\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
         "ax.set_title('loss')",
         {"post": post("_ax(fc.ax(), 'face', 'colors'), fc.hexcolor(fc.fig().get_facecolor()), fc.hexcolor(fc.ax().title.get_color())")}),
        ("context", "Inside `with plt.style.context('fivethirtyeight'):` create `fig, ax` and plot both losses. "
                    "(Checked: line widths, grid and background the style sets.)",
         "with plt.style.context('fivethirtyeight'):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, train_loss)\n"
         "    ax.plot(epochs, val_loss)", {"post": post("_ax(fc.ax(), 'lw', 'grid', 'face')")}),
        ("library", "Return the first three colors of the default cycle (`mpl.rcParamsDefault['axes.prop_cycle']`) and "
                    "of the 'tableau-colorblind10' style (`plt.style.library`), as `[default, colorblind]`.",
         "d = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color'][:3]\n"
         "c = plt.style.library['tableau-colorblind10']['axes.prop_cycle'].by_key()['color'][:3]\nresult = [d, c]"),
        ("context", f"The style is captured when the Axes is **created**: create `fig = plt.figure()` and its left "
                    f"subplot (`fig.add_subplot(1, 2, 1)`) inside `with plt.style.context({st!r}):`, and the right one "
                    f"after the block. Plot `train_loss` on both.",
         f"with plt.style.context({st!r}):\n    fig = plt.figure()\n    left = fig.add_subplot(1, 2, 1)\n"
         f"right = fig.add_subplot(1, 2, 2)\nfor a in (left, right):\n    a.plot(epochs, train_loss)",
         {"post": post("[_ax(a, 'face', 'grid') for a in fc.axes()]")}),
        ("use", "Clean white grid: apply 'seaborn-v0_8-whitegrid', create `fig, ax` and plot `val_loss`.",
         "plt.style.use('seaborn-v0_8-whitegrid')\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'grid', 'colors')")}),
        ("xkcd", "A fun slide: inside `with plt.xkcd():` create `fig, ax`, plot `train_loss` and title it 'it learns!'. "
                 "(Checked: sketchy line, title, and that the xkcd look ends with the block.)",
         "with plt.xkcd():\n    fig, ax = plt.subplots()\n    ax.plot(epochs, train_loss)\n    ax.set_title('it learns!')",
         {"post": post("_ax(fc.ax(), 'sketch', 'title'), _rc('path.sketch')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def rc_state_puzzles(rng):
    fs = rng.choice([14, 16, 20])
    lw = rng.choice([3, 4, 5])
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2"])
    items = [
        ("rc_context", f"Predict, then check: inside `with plt.rc_context({{'font.size': {fs}}}):` someone sets "
                       f"`plt.rcParams['font.size'] = {fs - 3}`. Return the font size **after** the block (the "
                       f"context restores what was there when it was entered).",
         f"with plt.rc_context({{'font.size': {fs}}}):\n    plt.rcParams['font.size'] = {fs - 3}\nresult = plt.rcParams['font.size']"),
        ("context", f"Inside `with plt.style.context({st!r}):` call `mpl.rcdefaults()` and record "
                    f"`plt.rcParams['axes.facecolor']`; record it again after the block. Return both.",
         f"with plt.style.context({st!r}):\n    mpl.rcdefaults()\n    inside = plt.rcParams['axes.facecolor']\n"
         f"result = [inside, plt.rcParams['axes.facecolor']]"),
        ("rc_context", f"Lines read rcParams when **they** are created, not the Axes. Create `fig, ax` inside `with "
                       f"plt.rc_context({{'lines.linewidth': {lw}}}):`, plot `[0, 1]` after the block, and return the "
                       f"line's width.",
         f"with plt.rc_context({{'lines.linewidth': {lw}}}):\n    fig, ax = plt.subplots()\nline, = ax.plot([0, 1])\n"
         f"result = line.get_linewidth()"),
        ("plt.rc", f"`plt.rc` accepts short aliases: run `plt.rc('lines', lw={lw}, ls='--')` and return "
                   f"`[plt.rcParams['lines.linewidth'], plt.rcParams['lines.linestyle']]`.",
         f"plt.rc('lines', lw={lw}, ls='--')\nresult = [plt.rcParams['lines.linewidth'], plt.rcParams['lines.linestyle']]"),
        ("cycler", "An Axes copies the color cycle when it is created. Create `fig, ax`, **then** set the global cycle "
                   "to `plt.cycler(color=['k', 'r'])`, plot `[0, 1]` on `ax` and on a new `fig2, ax2`. Return both "
                   "line colors as hex.",
         "fig, ax = plt.subplots()\nplt.rc('axes', prop_cycle=plt.cycler(color=['k', 'r']))\nfig2, ax2 = plt.subplots()\n"
         "a, = ax.plot([0, 1])\nb, = ax2.plot([0, 1])\nresult = [mpl.colors.to_hex(l.get_color()) for l in (a, b)]"),
        ("color", "An explicit `color=` doesn't consume the color cycle (when the cycle only has colors). On `fig, ax` "
                  "plot three lines `[0, 1]`: the first and third with no color, the middle one with `color='k'`. "
                  "Return the three colors as hex.",
         "fig, ax = plt.subplots()\nlines = [ax.plot([0, 1])[0], ax.plot([0, 1], color='k')[0], ax.plot([0, 1])[0]]\n"
         "result = [mpl.colors.to_hex(l.get_color()) for l in lines]"),
        ("set_prop_cycle", "On `fig, ax` set the cycle to `color=['k', 'r']`, plot `[0, 1]`, reset to the default cycle "
                           "with `ax.set_prop_cycle(None)`, and plot `[1, 0]`. Return both colors as hex.",
         "fig, ax = plt.subplots()\nax.set_prop_cycle(color=['k', 'r'])\na, = ax.plot([0, 1])\nax.set_prop_cycle(None)\n"
         "b, = ax.plot([1, 0])\nresult = [mpl.colors.to_hex(l.get_color()) for l in (a, b)]"),
        ("rcParamsDefault", f"Apply {st!r} and return the sorted list of rcParams keys whose value now differs from "
                            f"`mpl.rcParamsDefault`.",
         f"plt.style.use({st!r})\nresult = sorted(k for k in plt.style.library[{st!r}] if plt.rcParams[k] != mpl.rcParamsDefault[k])"),
    ]
    return catalog(rng, items, "", "")


@template(M, "medium")
def cycle_designs(rng):
    k = rng.randint(4, 5)
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nsteps = np.arange(30)\n"
            f"runs = np.round(np.exp(-steps / g.uniform(5, 15, ({k}, 1))) + 0.02 * g.random(({k}, 30)), 4)")
    desc = f"`runs` holds {k} loss curves (rows, 30 `steps` each)"
    pal = PAL[:3]
    COL = "_ax(fc.ax(), 'colors', 'ls')"
    loop = "for r in runs:\n    ax.plot(steps, r)"
    items = [
        ("cycler", "Grayscale-printable: give `fig, ax` a cycle of black lines with different dash styles, "
                   "`plt.cycler(color=['k'] * 4) + plt.cycler(linestyle=['-', '--', ':', '-.'])`, and plot every run.",
         "fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(color=['k'] * 4) + plt.cycler(linestyle=['-', '--', ':', '-.']))\n" + loop,
         {"post": post(COL)}),
        ("cycler", f"Color *and* marker for colorblind readers: `ax.set_prop_cycle(plt.cycler(marker=['o', 's']) * "
                   f"plt.cycler(color={pal}))` (6 combinations), then plot every run with `markevery=5`.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(marker=['o', 's']) * plt.cycler(color={pal}))\n"
         f"for r in runs:\n    ax.plot(steps, r, markevery=5)", {"post": post("_ax(fc.ax(), 'colors', 'markers')")}),
        ("cycler", f"Pair train and val curves by color: a cycle `plt.cycler(color=[c for c in {pal[:2]} for _ in "
                   f"range(2)]) + plt.cycler(linestyle=['-', '--'] * 2)`, then plot `runs[0]`, `runs[0] + 0.1`, "
                   f"`runs[1]`, `runs[1] + 0.1`.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(color=[c for c in {pal[:2]} for _ in range(2)]) + "
         f"plt.cycler(linestyle=['-', '--'] * 2))\nfor r in runs[:2]:\n    ax.plot(steps, r)\n    ax.plot(steps, r + 0.1)",
         {"post": post(COL)}),
        ("set_prop_cycle", f"Create `fig, (ax1, ax2) = plt.subplots(1, 2)`; give **both** the cycle `color={pal}` with "
                           f"one loop over the Axes, then plot every run on both.",
         f"fig, (ax1, ax2) = plt.subplots(1, 2)\nfor a in (ax1, ax2):\n    a.set_prop_cycle(color={pal})\n"
         f"    for r in runs:\n        a.plot(steps, r)", {"post": post("[_ax(a, 'colors') for a in fc.axes()]")}),
        ("rcParams", f"Set the colorblind cycle {PAL[:4]} **globally** (`plt.rcParams['axes.prop_cycle']`), then make a "
                     f"1 × 2 figure and plot the first two runs on the left and the rest on the right.",
         f"plt.rcParams['axes.prop_cycle'] = plt.cycler(color={PAL[:4]})\nfig, (ax1, ax2) = plt.subplots(1, 2)\n"
         f"for r in runs[:2]:\n    ax1.plot(steps, r)\nfor r in runs[2:]:\n    ax2.plot(steps, r)",
         {"post": post("[_ax(a, 'colors') for a in fc.axes()]")}),
        ("cycler", "Vary line width instead of color: `ax.set_prop_cycle(plt.cycler(color=['tab:blue'] * 3) + "
                   "plt.cycler(lw=[1, 2, 3]))`, then plot the first three runs.",
         "fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(color=['tab:blue'] * 3) + plt.cycler(lw=[1, 2, 3]))\n"
         "for r in runs[:3]:\n    ax.plot(steps, r)", {"post": post("_ax(fc.ax(), 'colors', 'lw')")}),
        ("set_prop_cycle", f"Highlight the best run without breaking the cycle: set `ax.set_prop_cycle(color={pal})`, plot every "
                   f"run, then plot the run with the lowest final loss **again** in black ('k', lw 2).",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color={pal})\n" + loop +
         "\nax.plot(steps, runs[runs[:, -1].argmin()], color='k', lw=2)", {"post": post("_ax(fc.ax(), 'colors', 'lw')")}),
        ("by_key", f"Read a cycle back: build `cyc = plt.cycler(color={pal}) + plt.cycler(linestyle=['-', '--', "
                   f"':'])`, set it on `fig, ax` with `ax.set_prop_cycle(cyc)`, and return `cyc.by_key()` as a dict of "
                   f"lists.",
         f"fig, ax = plt.subplots()\ncyc = plt.cycler(color={pal}) + plt.cycler(linestyle=['-', '--', ':'])\n"
         f"ax.set_prop_cycle(cyc)\nresult = {{k: list(v) for k, v in cyc.by_key().items()}}"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def rc_inspection(rng):
    pat = rng.choice(["^legend\\.f", "^grid\\.", "^lines\\.m", "^axes\\.title", "^xtick\\.major"])
    st = rng.choice(["ggplot", "bmh", "fivethirtyeight", "dark_background"])
    key = rng.choice(["lines.linewidth", "font.size", "axes.titlesize"])
    items = [
        ("find_all", f"`plt.rcParams.find_all(pattern)` returns the rcParams whose key matches a regular expression. "
                     f"Return the sorted keys matching {pat!r}.", f"result = sorted(plt.rcParams.find_all({pat!r}))"),
        ("library", f"Return how many of the {st!r} style's settings differ from Matplotlib's defaults "
                    f"(`plt.style.library[{st!r}]` vs `mpl.rcParamsDefault`).",
         f"result = sum(v != mpl.rcParamsDefault[k] for k, v in plt.style.library[{st!r}].items())"),
        ("rcParamsDefault", "Change 'font.size' to 12 and 'lines.linewidth' to 3, apply nothing else, and return the "
                            "sorted list of keys where `plt.rcParams` differs from `mpl.rcParamsDefault`.",
         "plt.rcParams['font.size'] = 12\nplt.rcParams['lines.linewidth'] = 3\n"
         "result = sorted(k for k in plt.rcParams if plt.rcParams[k] != mpl.rcParamsDefault[k])"),
        ("rcParams", f"rcParams validates what you assign. Try `plt.rcParams[{key!r}] = 'thick'` inside `try` / "
                     f"`except ValueError`, and return `[the exception class name, plt.rcParams[{key!r}]]` (the value "
                     f"is unchanged).",
         f"try:\n    plt.rcParams[{key!r}] = 'thick'\n    name = None\nexcept ValueError as e:\n    name = type(e).__name__\n"
         f"result = [name, plt.rcParams[{key!r}]]"),
        ("rcParams", "A misspelled key raises: try `plt.rcParams['line.linewidth'] = 2` and return the exception "
                     "class name (catch `Exception`).",
         "try:\n    plt.rcParams['line.linewidth'] = 2\n    result = None\nexcept Exception as e:\n    result = type(e).__name__"),
        ("rcParams", "Return the Python type names of `plt.rcParams['figure.figsize']`, `plt.rcParams['font.family']`, "
                     "`plt.rcParams['axes.grid']` and `plt.rcParams['lines.linewidth']`.",
         "result = [type(plt.rcParams[k]).__name__ for k in ('figure.figsize', 'font.family', 'axes.grid', 'lines.linewidth')]"),
        ("library", "Return the sorted names of the built-in styles that turn the grid on (`plt.style.library[s]."
                    "get('axes.grid') is True`) among `plt.style.available`.",
         "result = sorted(s for s in plt.style.available if plt.style.library[s].get('axes.grid') is True)"),
        ("context", f"Return `plt.rcParams[{key!r}]` read inside `with plt.style.context({st!r}):` and again after "
                    f"the block, as `[inside, outside]`.",
         f"with plt.style.context({st!r}):\n    inside = plt.rcParams[{key!r}]\nresult = [inside, plt.rcParams[{key!r}]]",
         {"require": ["context"]}),
    ]
    return catalog(rng, items, "", "")


# ================================================================= hard

@template(M, "hard")
def style_helpers(rng):
    code, desc, n = curves(rng)
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2"])
    base = rng.choice([9, 10, 11, 12])
    k = rng.choice([1.5, 2])
    items = [
        ("rc_context", f"Write `paper_rc(base={base})` returning the dict `{{'font.size': base, 'axes.titlesize': base + "
                       f"2, 'axes.labelsize': base, 'legend.fontsize': base - 2, 'font.family': 'serif', "
                       f"'axes.spines.top': False, 'axes.spines.right': False}}` — meant for `with "
                       f"plt.rc_context(paper_rc()):`. (Checked: the dicts, and a figure made inside such a block.)",
         f"def paper_rc(base={base}):\n    return {{'font.size': base, 'axes.titlesize': base + 2, 'axes.labelsize': base,\n"
         f"            'legend.fontsize': base - 2, 'font.family': 'serif', 'axes.spines.top': False,\n"
         f"            'axes.spines.right': False}}",
         {"post": post("with plt.rc_context(paper_rc(8)):\n    _f, _a = plt.subplots()\n    _a.set_title('t')\n"
                       "paper_rc(), _ax(_a, 'titlesize', 'spines', 'family'), _rc('font.size')"), "require": []}),
        ("context", "Write a decorator factory `with_style(style)`: `@with_style('ggplot')` makes the decorated function "
                    "run inside `plt.style.context(style)` (keep its return value and arguments; `functools.wraps` is "
                    "nice).",
         "import functools\n\ndef with_style(style):\n    def deco(fn):\n        @functools.wraps(fn)\n"
         "        def wrapper(*args, **kwargs):\n            with plt.style.context(style):\n"
         "                return fn(*args, **kwargs)\n        return wrapper\n    return deco",
         {"post": post(f"@with_style({st!r})\ndef _mk(y, title='x'):\n    _f, _a = plt.subplots()\n    _a.plot(y)\n"
                       f"    _a.set_title(title)\n    return _a\n_a = _mk(train_loss, title='loss')\n"
                       f"_ax(_a, 'face', 'grid', 'colors', 'title'), _rc('axes.facecolor'), _mk.__name__")}),
        ("context", "Write `styled_subplots(style, *args, **kwargs)`: create and return `plt.subplots(*args, **kwargs)` "
                    "inside `plt.style.context(style)`, so the figure gets the style but the global defaults don't.",
         "def styled_subplots(style, *args, **kwargs):\n    with plt.style.context(style):\n"
         "        return plt.subplots(*args, **kwargs)",
         {"post": post(f"_f, _axs = styled_subplots({st!r}, 1, 2, figsize=(6, 2))\n_g, _b = plt.subplots()\n"
                       f"[_ax(a, 'face') for a in _axs], fc.size(_f), _ax(_b, 'face')")}),
        ("set_prop_cycle", "Write `use_colorblind_cycle(ax=None)`: with an Axes, give it the color cycle `okabe` "
                           "(given); with `ax=None`, make `okabe` the **global** default cycle instead.",
         "def use_colorblind_cycle(ax=None):\n    if ax is None:\n        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=okabe)\n"
         "    else:\n        ax.set_prop_cycle(color=okabe)",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\nuse_colorblind_cycle(_a)\nfor _x in (_a, _b):\n"
                       "    _x.plot(train_loss)\n    _x.plot(val_loss)\nuse_colorblind_cycle()\n_g, _c = plt.subplots()\n"
                       "_c.plot(train_loss)\n[_ax(x, 'colors') for x in (_a, _b, _c)]"),
          "setup": f"okabe = {PAL[:4]}"}),
        ("library", "Write `rc_diff(style)` returning a dict of the settings of the built-in `style` "
                    "(`plt.style.library[style]`) whose value differs from `mpl.rcParamsDefault`.",
         "def rc_diff(style):\n    return {k: v for k, v in plt.style.library[style].items() if v != mpl.rcParamsDefault[k]}",
         {"post": f"_d = rc_diff({st!r})\nsorted(_d), _d.get('axes.facecolor'), sorted(rc_diff('grayscale'))"}),
        ("rc_context", "Write `save_for_paper(fig, path, dpi=300)` that saves `fig` with a tight bounding box, the given "
                       "dpi and a transparent background — by setting `'savefig.bbox'`, `'savefig.dpi'` and "
                       "`'savefig.transparent'` in a `plt.rc_context` around a plain `fig.savefig(path)` — and returns "
                       "`path`.",
         "def save_for_paper(fig, path, dpi=300):\n    with plt.rc_context({'savefig.bbox': 'tight', 'savefig.dpi': dpi,\n"
         "                         'savefig.transparent': True}):\n        fig.savefig(path)\n    return path",
         {"post": post("_f, _a = plt.subplots(figsize=(3, 2))\n_a.plot(train_loss)\n_p = save_for_paper(_f, 'p.png', dpi=30)\n"
                       "_im = plt.imread(_p)\n_p, _im.shape, float(_im[0, 0, 3]), _rc('savefig.bbox', 'savefig.dpi')")}),
        ("rcParams", f"Write `font_scale(k)`: multiply every rcParams entry whose key ends in 'size' **and** whose value "
                     f"is a number (int or float, not bool) by `k`, in place. (Sizes like 'large' are relative and stay.)",
         "def font_scale(k):\n    for key, v in plt.rcParams.items():\n"
         "        if key.endswith('size') and isinstance(v, (int, float)) and not isinstance(v, bool):\n"
         "            plt.rcParams[key] = v * k",
         {"post": post(f"font_scale({k})\n_rc('font.size', 'axes.titlesize', 'legend.fontsize', 'lines.markersize', "
                       f"'xtick.major.size')")}),
        ("contextmanager", "Write a context manager `temporary_rc(**settings)` where a double underscore stands for a "
                           "dot (`temporary_rc(lines__linewidth=3, axes__grid=True)`): inside, those rcParams apply; "
                           "afterwards everything is restored (`contextlib.contextmanager` + `plt.rc_context`).",
         "import contextlib\n\n@contextlib.contextmanager\ndef temporary_rc(**settings):\n"
         "    with plt.rc_context({k.replace('__', '.'): v for k, v in settings.items()}):\n        yield",
         {"post": post("with temporary_rc(lines__linewidth=4, axes__grid=True):\n    _f, _a = plt.subplots()\n"
                       "    _a.plot(train_loss)\n_ax(_a, 'lw', 'grid'), _rc('lines.linewidth', 'axes.grid')")}),
    ]
    return catalog(rng, items, code, f"{desc}. Write the function; it is checked by calling it.")


@template(M, "hard")
def fix_style_bugs(rng):
    code, desc, n = curves(rng)
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2"])
    lw = rng.choice([2, 2.5, 3])
    w, h = rng.choice([(6, 4), (5, 3), (8, 4)])
    items = [
        ("use", f"The figure stays plain: a style only affects Axes created **after** it is applied.\n```python\n"
                f"fig, ax = plt.subplots()\nplt.style.use({st!r})\nax.plot(epochs, train_loss)\n```\nFix the order so "
                f"the plot gets the {st!r} look.",
         f"plt.style.use({st!r})\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'grid', 'colors')")}),
        ("rcParams", f"This crashes with *KeyError: 'line.linewidth is not a valid rc parameter'*:\n```python\n"
                     f"plt.rcParams['line.linewidth'] = {lw}\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n```\n"
                     f"Fix it.",
         f"plt.rcParams['lines.linewidth'] = {lw}\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_ax(fc.ax(), 'lw'), _rc('lines.linewidth')")}),
        ("use", "This crashes with *OSError: 'seaborn' is not a valid package style...*: since Matplotlib 3.6 the "
                "seaborn styles are named 'seaborn-v0_8…'.\n```python\nplt.style.use('seaborn')\n```\nFix it (the "
                "plain seaborn look).", "plt.style.use('seaborn-v0_8')",
         {"post": post("_rc('axes.facecolor', 'axes.grid', 'axes.prop_cycle')")}),
        ("context", "The dark slide style leaks into the report figure drawn afterwards:\n```python\n"
                    "plt.style.use('dark_background')\nslide, ax1 = plt.subplots()\nax1.plot(epochs, train_loss)\n"
                    "report, ax2 = plt.subplots()\nax2.plot(epochs, val_loss)\n```\nKeep the slide dark but give the "
                    "report the default look (and leave the defaults untouched afterwards).",
         "with plt.style.context('dark_background'):\n    slide, ax1 = plt.subplots()\n    ax1.plot(epochs, train_loss)\n"
         "report, ax2 = plt.subplots()\nax2.plot(epochs, val_loss)",
         {"post": post("[_ax(f.axes[0], 'face', 'colors') for f in fc.figs()], _rc('axes.facecolor')")}),
        ("cycler", "This crashes with *Key axes.prop_cycle: Object is not a string or Cycler instance*:\n```python\n"
                   f"plt.rc('axes', prop_cycle={PAL[:3]})\n```\nFix it (those colors as the cycle).",
         f"plt.rc('axes', prop_cycle=plt.cycler(color={PAL[:3]}))", {"post": post("_rc('axes.prop_cycle')")}),
        ("cycler", "This crashes with *ValueError: Can only add equal length cycles, not 3 and 2*. The goal was every "
                   "color with every line style (6 combinations).\n```python\nfig, ax = plt.subplots()\n"
                   "ax.set_prop_cycle(plt.cycler(color=['k', 'r', 'b']) + plt.cycler(linestyle=['-', '--']))\n"
                   "for i in range(6):\n    ax.plot(epochs, train_loss + i)\n```\nFix it (colors change fastest).",
         "fig, ax = plt.subplots()\nax.set_prop_cycle(plt.cycler(linestyle=['-', '--']) * plt.cycler(color=['k', 'r', 'b']))\n"
         "for i in range(6):\n    ax.plot(epochs, train_loss + i)", {"post": post("_ax(fc.ax(), 'colors', 'ls')")}),
        ("xkcd", "After this, **every** later figure is sketchy too, because `plt.xkcd()` changed the global "
                 "defaults:\n```python\nplt.xkcd()\nfun, ax1 = plt.subplots()\nax1.plot(epochs, train_loss)\n"
                 "serious, ax2 = plt.subplots()\nax2.plot(epochs, val_loss)\n```\nLimit the xkcd look to the first "
                 "figure.",
         "with plt.xkcd():\n    fun, ax1 = plt.subplots()\n    ax1.plot(epochs, train_loss)\nserious, ax2 = plt.subplots()\n"
         "ax2.plot(epochs, val_loss)", {"post": post("[_ax(f.axes[0], 'sketch') for f in fc.figs()], _rc('path.sketch')")}),
        ("rcParams", f"This crashes with *Key figure.figsize: Could not convert '{w}x{h}' to float*:\n```python\n"
                     f"plt.rcParams['figure.figsize'] = '{w}x{h}'\nfig = plt.figure()\n```\nFix it.",
         f"plt.rcParams['figure.figsize'] = ({w}, {h})\nfig = plt.figure()", {"post": "fc.size()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def stylesheet_drills(rng):
    code, desc, n = curves(rng)
    st = rng.choice(["ggplot", "bmh", "seaborn-v0_8-darkgrid", "Solarize_Light2"])
    fs = rng.choice([9, 10, 11])
    w, h = rng.choice([(3.5, 2.5), (4, 3), (3, 2)])
    pal = PAL[:4]
    items = [
        ("use", f"Drill: write `paper.mplstyle` (serif fonts, no top/right spines, `font.size: {fs}`, "
                f"`figure.figsize: {w}, {h}`), apply it with `plt.style.use`, plot both losses on `fig, ax`, save "
                f"`paper.png` at 30 dpi and close the figure. (Checked: the settings, the PNG's shape, no open figure.)",
         f"open('paper.mplstyle', 'w').write('font.family: serif\\naxes.spines.top: False\\naxes.spines.right: False\\n'\n"
         f"                                  'font.size: {fs}\\nfigure.figsize: {w}, {h}\\n')\nplt.style.use('./paper.mplstyle')\n"
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nfig.savefig('paper.png', dpi=30)\n"
         f"plt.close(fig)",
         {"post": post("_rc('font.family', 'font.size', 'axes.spines.top', 'axes.spines.right'), "
                       "plt.imread('paper.png').shape, len(plt.get_fignums())")}),
        ("context", f"Drill: switch styles for **one** figure only. Draw `train_loss` on `fig1, ax1` with the defaults, "
                    f"then `val_loss` on `fig2, ax2` inside `with plt.style.context({st!r}):`, then `train_loss` on "
                    f"`fig3, ax3` after the block.",
         f"fig1, ax1 = plt.subplots()\nax1.plot(epochs, train_loss)\nwith plt.style.context({st!r}):\n"
         f"    fig2, ax2 = plt.subplots()\n    ax2.plot(epochs, val_loss)\nfig3, ax3 = plt.subplots()\nax3.plot(epochs, train_loss)",
         {"post": post("[_ax(f.axes[0], 'face', 'grid', 'colors') for f in fc.figs()]")}),
        ("plt.rc", f"Drill: set the colorblind-safe cycle {pal} **globally** with `plt.rc('axes', prop_cycle=...)`, then "
                   f"make two figures, each plotting both losses.",
         f"plt.rc('axes', prop_cycle=plt.cycler(color={pal}))\nfor _ in range(2):\n    fig, ax = plt.subplots()\n"
         f"    ax.plot(epochs, train_loss)\n    ax.plot(epochs, val_loss)",
         {"post": post("[_ax(f.axes[0], 'colors') for f in fc.figs()], _rc('axes.prop_cycle')")}),
        ("rc_context", "Paper mode for one figure: inside `with plt.rc_context({'font.family': 'serif', "
                       "'mathtext.fontset': 'cm', 'axes.spines.top': False, 'axes.spines.right': False}):` plot "
                       "`val_loss` on `fig, ax` with the y label `r'$\\mathcal{L}_{val}$'`.",
         "with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'cm', 'axes.spines.top': False,\n"
         "                     'axes.spines.right': False}):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, val_loss)\n"
         "    ax.set_ylabel(r'$\\mathcal{L}_{val}$')",
         {"post": post("_ax(fc.ax(), 'spines'), fc.ax().yaxis.label.get_fontfamily(), fc.labels()['ylabel'], "
                       "_rc('font.family', 'mathtext.fontset')")}),
        ("library", "Take **only** the colors of the 'tableau-colorblind10' style (not the rest of it): copy its "
                    "`'axes.prop_cycle'` from `plt.style.library` into `plt.rcParams`, then plot both losses on "
                    "`fig, ax`.",
         "plt.rcParams['axes.prop_cycle'] = plt.style.library['tableau-colorblind10']['axes.prop_cycle']\n"
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'colors', 'face')")}),
        ("available", "As in the guide: return `plt.style.available[:5]` — and before that, inside `with "
                      "plt.style.context('seaborn-v0_8-darkgrid'):`, create `fig, ax` and plot `[1, 3, 2]`.",
         "with plt.style.context('seaborn-v0_8-darkgrid'):\n    fig, ax = plt.subplots()\n    ax.plot([1, 3, 2])\n"
         "result = plt.style.available[:5]", {"post": post("_, _ax(fc.ax(), 'face', 'grid')")}),
        ("rcdefaults", f"Customize heavily — `mpl.rcParams.update({{'font.size': {fs + 4}, 'axes.grid': True, "
                       f"'lines.linewidth': 3}})` and `plt.style.use({st!r})` — then undo it all with "
                       f"`mpl.rcdefaults()` and draw `train_loss` on a new `fig, ax`: it must look like the defaults.",
         f"mpl.rcParams.update({{'font.size': {fs + 4}, 'axes.grid': True, 'lines.linewidth': 3}})\nplt.style.use({st!r})\n"
         f"mpl.rcdefaults()\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'grid', 'lw', 'colors', 'ticksize')")}),
        ("rc", "`mpl.rc` is the same as `plt.rc`. Use it to make the grid dashed and gray for all plots "
               "(`mpl.rc('grid', linestyle='--', color='gray')`) and turn the grid on (`mpl.rc('axes', grid=True)`); "
               "then plot `val_loss` on `fig, ax`.",
         "mpl.rc('grid', linestyle='--', color='gray')\nmpl.rc('axes', grid=True)\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)",
         {"post": post("_ax(fc.ax(), 'grid'), _rc('grid.linestyle', 'grid.color'), "
                       "fc.ax().xaxis.get_major_ticks()[0].gridline.get_linestyle()")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def themed_dashboards(rng):
    code, desc, n = curves(rng, acc=True)
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2"])
    ts = rng.choice([14, 16, 18])
    pal = PAL[:3]
    panels = "for a, (y, t) in zip(axs.flat, [(train_loss, 'train'), (val_loss, 'val'), (val_acc, 'acc'), (val_loss - train_loss, 'gap')]):\n"
    items = [
        ("rc_context", "A 2 × 2 dashboard in one theme: inside `with plt.rc_context({'axes.titlesize': 12, "
                       "'axes.labelsize': 10, 'axes.grid': True, 'axes.spines.top': False, 'axes.spines.right': "
                       "False}):` create `fig, axs = plt.subplots(2, 2)` and plot `train_loss`, `val_loss`, `val_acc` "
                       "and `val_loss - train_loss` (row-major) titled 'train', 'val', 'acc', 'gap'.",
         "with plt.rc_context({'axes.titlesize': 12, 'axes.labelsize': 10, 'axes.grid': True, 'axes.spines.top': False,\n"
         "                     'axes.spines.right': False}):\n    fig, axs = plt.subplots(2, 2)\n"
         "    " + panels.replace("\n", "\n    ") + "    a.plot(epochs, y)\n        a.set_title(t)",
         {"post": post("[_ax(a, 'title', 'titlesize', 'grid', 'spines') for a in fc.axes()]")}),
        ("context", f"One figure, two looks: `fig = plt.figure(figsize=(8, 3))`; the left subplot (1 × 2 grid) created "
                    f"inside `with plt.style.context({st!r}):`, the right one with the defaults; `train_loss` on the "
                    f"left, `val_acc` on the right.",
         f"fig = plt.figure(figsize=(8, 3))\nwith plt.style.context({st!r}):\n    left = fig.add_subplot(1, 2, 1)\n"
         f"    left.plot(epochs, train_loss)\nright = fig.add_subplot(1, 2, 2)\nright.plot(epochs, val_acc)",
         {"post": post("[_ax(a, 'face', 'colors') for a in fc.axes()]")}),
        ("use", f"A style can also be a dict: `plt.style.use({{'figure.titlesize': {ts}, 'axes.prop_cycle': "
                f"plt.cycler(color={pal})}})`. Then make a 1 × 3 figure with `train_loss`, `val_loss`, `val_acc` and "
                f"the figure title 'run summary' (`fig.suptitle`).",
         f"plt.style.use({{'figure.titlesize': {ts}, 'axes.prop_cycle': plt.cycler(color={pal})}})\n"
         f"fig, axs = plt.subplots(1, 3)\nfor a, y in zip(axs, (train_loss, val_loss, val_acc)):\n    a.plot(epochs, y)\n"
         f"fig.suptitle('run summary')",
         {"post": post("[_ax(a, 'colors') for a in fc.axes()], fc.suptitles()['suptitle'], fc.fig()._suptitle.get_fontsize()")}),
        ("context", f"Dark dashboard: inside `with plt.style.context(['dark_background', {{'axes.prop_cycle': "
                    f"plt.cycler(color={pal})}}]):` create `fig, (ax1, ax2) = plt.subplots(1, 2)`; both losses on "
                    f"`ax1`, `val_acc` on `ax2`.",
         f"with plt.style.context(['dark_background', {{'axes.prop_cycle': plt.cycler(color={pal})}}]):\n"
         f"    fig, (ax1, ax2) = plt.subplots(1, 2)\n    ax1.plot(epochs, train_loss)\n    ax1.plot(epochs, val_loss)\n"
         f"    ax2.plot(epochs, val_acc)", {"post": post("[_ax(a, 'face', 'colors') for a in fc.axes()]")}),
        ("rcParams", "Make constrained layout the default for new figures (`'figure.constrained_layout.use'`), then "
                     "create `fig, axs = plt.subplots(1, 2)` with `train_loss` and `val_acc`, titled 'loss' and 'acc'. "
                     "Return the layout engine's class name.",
         "plt.rcParams['figure.constrained_layout.use'] = True\nfig, axs = plt.subplots(1, 2)\n"
         "for a, y, t in zip(axs, (train_loss, val_acc), ('loss', 'acc')):\n    a.plot(epochs, y)\n    a.set_title(t)\n"
         "result = type(fig.get_layout_engine()).__name__", {"post": post("_, [_ax(a, 'title') for a in fc.axes()]")}),
        ("update", "Report-style titles: `mpl.rcParams.update({'axes.titlelocation': 'left', 'axes.titleweight': "
                   "'bold', 'axes.titlesize': 'medium'})`, then a 1 × 2 figure with `train_loss` titled 'Loss' and "
                   "`val_acc` titled 'Accuracy'.",
         "mpl.rcParams.update({'axes.titlelocation': 'left', 'axes.titleweight': 'bold', 'axes.titlesize': 'medium'})\n"
         "fig, axs = plt.subplots(1, 2)\nfor a, y, t in zip(axs, (train_loss, val_acc), ('Loss', 'Accuracy')):\n"
         "    a.plot(epochs, y)\n    a.set_title(t)",
         {"post": "[(fc.titles(a), a._left_title.get_fontweight(), a._left_title.get_fontsize()) for a in fc.axes()]"}),
        ("rcParams", "Code-like labels: set `'font.family'` to 'monospace', then plot `val_loss` on `fig, ax` with the "
                     "title 'val_loss' and x label 'epoch'.",
         "plt.rcParams['font.family'] = 'monospace'\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)\n"
         "ax.set_title('val_loss')\nax.set_xlabel('epoch')",
         {"post": post("_ax(fc.ax(), 'family', 'title'), fc.ax().xaxis.get_major_ticks()[0].label1.get_fontfamily()")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= more medium

@template(M, "medium")
def rc_scoping(rng):
    code, desc, n = curves(rng)
    st = rng.choice(["ggplot", "bmh", "Solarize_Light2"])
    lw = rng.choice([3, 4])
    items = [
        ("rc_context", f"`plt.style.use` inside an `rc_context` is undone at its end too. Inside `with plt.rc_context():` "
                       f"apply {st!r}; after the block create `fig, ax` and plot `train_loss`.",
         f"with plt.rc_context():\n    plt.style.use({st!r})\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)",
         {"post": post("_ax(fc.ax(), 'face', 'colors'), _rc('axes.facecolor')")}),
        ("rc_context", f"Lines read the defaults when they are made. On one `fig, ax` plot `train_loss` inside `with "
                       f"plt.rc_context({{'lines.linewidth': {lw}, 'lines.linestyle': '--'}}):` and `val_loss` after it.",
         f"fig, ax = plt.subplots()\nwith plt.rc_context({{'lines.linewidth': {lw}, 'lines.linestyle': '--'}}):\n"
         f"    ax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)", {"post": post("_ax(fc.ax(), 'lw', 'ls')")}),
        ("context", f"Text reads the defaults when it is made, too: create `fig, ax` and plot `val_loss` with the "
                    f"defaults, then set the title 'val' inside `with plt.style.context({{'axes.titlesize': 20, "
                    f"'axes.titlecolor': 'red'}}):`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nwith plt.style.context({'axes.titlesize': 20, 'axes.titlecolor': 'red'}):\n"
         "    ax.set_title('val')", {"post": "fc.ax().title.get_fontsize(), fc.hexcolor(fc.ax().title.get_color())"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

@template(M, "challenge")
def rc_context_stack(rng):
    n, K, s = rng.randint(16, 24), rng.randint(3, 5), seeded(rng)
    report = rng.choice(["queries", "sum", "changed"])

    def mk(seed, nn, kk):
        return (f"g = np.random.default_rng({seed})\nop = g.choice(5, {nn}, p=[0.35, 0.15, 0.15, 0.05, 0.3])\n"
                f"key = g.integers(0, {kk}, {nn})\nval = g.integers(1, 100, {nn})\nK = {kk}\nreport = {report!r}")
    setup = mk(s, n, K)
    sol = ("hist = []                         # the ops in effect, oldest first: a key, or -1 for a reset\n"
           "sets = [[] for _ in range(K)]     # per key: (position in hist, value) of its sets in effect\n"
           "resets, marks = [], []            # positions of resets in effect; hist length at each open context\n\n"
           "def value(k):\n    last = resets[-1] if resets else -1\n"
           "    return sets[k][-1][1] if sets[k] and sets[k][-1][0] > last else 0\n\nanswers = []\n"
           "for o, k, v in zip(op.tolist(), key.tolist(), val.tolist()):\n    if o == 0:\n"
           "        sets[k].append((len(hist), v))\n        hist.append(k)\n    elif o == 1:\n        marks.append(len(hist))\n"
           "    elif o == 2 and marks:\n        m = marks.pop()\n        while len(hist) > m:   # every op is undone at most once\n"
           "            j = hist.pop()\n            if j < 0:\n                resets.pop()\n            else:\n"
           "                sets[j].pop()\n    elif o == 3:\n        resets.append(len(hist))\n        hist.append(-1)\n"
           "    elif o == 4:\n        answers.append(value(k))\n"
           + {"queries": "result = np.array(answers, dtype=int)", "sum": "result = sum(value(k) for k in range(K))",
              "changed": "result = sum(value(k) != 0 for k in range(K))"}[report])
    tests = [mk(s + 1, 200000, 1000), mk(s + 2, 200000, 20)]
    want = {"queries": "the answers of the queries, in order, as an int array",
            "sum": "the sum of all `K` values at the end", "changed": "how many keys differ from their default at the end"}[report]
    return challenge(
        "Nested `rc_context` bookkeeping",
        "Model `K` rcParams keys (all 0 by default) and a script of operations `op[i]` with arguments `key[i]`, "
        "`val[i]`: **0** = `rcParams[key] = val`; **1** = enter a `plt.rc_context()` (snapshot everything); **2** = "
        "leave the innermost open context, restoring every key to its value when that context was entered (does "
        "nothing if no context is open); **3** = `mpl.rcdefaults()` (every key back to 0); **4** = read "
        f"`rcParams[key]` (a query). Contexts still open at the end stay open. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "≤ 2·10^5 operations, K ≤ 1000",
        speed="O(n + K): undo the ops since the matching enter, each at most once (copying all K keys per context is too slow)",
        answer_desc=" (an int array)" if report == "queries" else " (an int)")


@template(M, "challenge")
def style_cascade(rng):
    S, K, q, s = rng.randint(5, 7), rng.randint(3, 5), rng.randint(6, 9), seeded(rng)
    report = rng.choice(["value", "source", "defaults"])

    def mk(seed, SS, KK, P, Q):
        return (f"g = np.random.default_rng({seed})\nS, K = {SS}, {KK}\n"
                f"code = np.unique(g.integers(0, S * K, {P}))     # style * K + key, each pair once\n"
                f"style, key = code // K, code % K\nvalue = g.integers(1, 1000, len(code))\n"
                f"qa = g.integers(0, S, {Q})\nqb = np.minimum(qa + g.integers(0, S, {Q}), S - 1)\n"
                f"qk = g.integers(0, K, {Q})\nreport = {report!r}")
    setup = mk(s, S, K, S * K // 2, q)
    sol = ("order = np.lexsort((style, key))            # by key, then by style\nks, ss, vs = key[order], style[order], value[order]\n"
           "pos = np.searchsorted(ks * S + ss, qk * S + qb, side='right') - 1   # last setter of qk with style <= qb\n"
           "hit = (pos >= 0) & (ks[np.maximum(pos, 0)] == qk) & (ss[np.maximum(pos, 0)] >= qa)\n"
           + {"value": "result = np.where(hit, vs[np.maximum(pos, 0)], 0)",
              "source": "result = np.where(hit, ss[np.maximum(pos, 0)], -1)",
              "defaults": "result = int((~hit).sum())"}[report])
    tests = [mk(s + 1, 200000, 50, 200000, 200000), mk(s + 2, 1000, 1000, 150000, 200000)]
    want = {"value": "the value of `qk[j]` for every query (0 = the default), as an int array",
            "source": "for every query the index of the style whose setting wins (-1 if none sets the key), as an "
                      "int array", "defaults": "how many queries end up with the default"}[report]
    return challenge(
        "Stacking style sheets",
        "`plt.style.use([s1, s2, ...])` applies the sheets left to right, so for each key the **last** sheet that "
        "sets it wins. There are `S` style sheets; sheet `style[p]` sets key `key[p]` to `value[p]` (every (sheet, "
        "key) pair at most once; keys are 0 … K - 1 and default to 0). Query `j` applies sheets `qa[j]` … `qb[j]` "
        f"(inclusive, in order) to fresh defaults and reads key `qk[j]`. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "S·K ≤ 10^7, ≤ 2·10^5 settings and queries",
        speed="O((P + Q) log P): sort the settings by (key, sheet) and binary-search each query",
        answer_desc=" (an int)" if report == "defaults" else " (an int array)")


@template(M, "challenge")
def cycle_advance_rule(rng):
    n, A, c, s = rng.randint(12, 18), rng.randint(2, 3), rng.randint(3, 4), seeded(rng)
    report = rng.choice(["used", "next", "counts"])

    def mk(seed, nn, AA, cc):
        return (f"g = np.random.default_rng({seed})\nA, c = {AA}, {cc}\nax_i = g.integers(0, A, {nn})\n"
                f"kind = g.choice(4, {nn}, p=[0.5, 0.2, 0.2, 0.1])\nreport = {report!r}")
    setup = mk(s, n, A, c)
    sol = ("pos = [0] * A                 # next cycle entry of each Axes\nused = []\n"
           "for a, k in zip(ax_i.tolist(), kind.tolist()):\n    if k == 3:                    # set_prop_cycle: start over\n"
           "        pos[a] = 0\n    elif k == 2:                  # every cycled property given: the cycle doesn't move\n"
           "        used.append(-1)\n    else:                         # something taken from the cycle: it advances\n"
           "        used.append(pos[a])\n        pos[a] = (pos[a] + 1) % c\n"
           + {"used": "result = np.array(used, dtype=int)", "next": "result = np.array(pos)",
              "counts": "u = np.array(used, dtype=int)\nresult = np.bincount(u[u >= 0], minlength=c)"}[report])
    tests = [mk(s + 1, 200000, 1000, 7), mk(s + 2, 200000, 3, 10)]
    want = {"used": "for every plot call (kind 0–2), the cycle entry it took, or -1 if it took none, as an int array",
            "next": "for every Axes the cycle entry its **next** line would take, as an int array",
            "counts": "how many lines took each cycle entry, as an int array of length `c`"}[report]
    return challenge(
        "Who advances the property cycle?",
        "Each Axes has its own position in its property cycle, here `ax.set_prop_cycle(color=[...], linewidth=[...])` "
        "with `c` entries. Call `i` acts on Axes `ax_i[i]`: kind **0** = `ax.plot(y)`, **1** = `ax.plot(y, "
        "color='k')`, **2** = `ax.plot(y, color='k', linewidth=0.5)`, **3** = `ax.set_prop_cycle(...)` again (back "
        "to entry 0). Matplotlib's rule: a plot call takes the next entry (and the cycle advances) unless it gives "
        "**every** property the cycle sets; the cycle wraps around after `c` entries. Return "
        f"{want}. (`report` = {report!r}.)",
        setup, sol, tests, "≤ 2·10^5 calls, A ≤ 1000 Axes, c ≤ 10", speed="O(n)",
        answer_desc=" (an int array)")

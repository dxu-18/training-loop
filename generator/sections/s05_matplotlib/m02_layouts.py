"""5.02 Figures, subplots & layouts: grids, mosaics, gridspecs, hand-placed Axes, subfigures, insets, twin and
secondary axes, and the layout engines that keep panels apart."""

from ._common import P, catalog, challenge, classes, curves, lit, seeded, signal, template

M = "5.02"
NLINES = "[len(a.get_lines()) for a in fc.axes()]"
YDATA = "[fc.ydata(a) for a in fc.axes()]"
TITLES = "[a.get_title() for a in fc.axes()]"
SPANS = "fc.grid_shape(), fc.spans()"
TICKLABELS = ("[(a.xaxis._major_tick_kw.get('label1On', True), a.yaxis._major_tick_kw.get('label1On', True)) "
              "for a in fc.axes()]")
ENGINE = "type(fc.fig().get_layout_engine()).__name__"
SUBPARS = "[fc.r(getattr(fc.fig().subplotpars, k), 3) for k in ('left', 'right', 'bottom', 'top', 'wspace', 'hspace')]"
CHILDREN = "[(type(c).__name__, fc.r(c.get_axes_locator()._bounds, 3)) for c in fc.ax(0).child_axes]"


def _share(axis: str) -> str:
    """Post snippet: which pairs of the figure's Axes share the x (or y) axis."""
    return f"[[a.get_shared_{axis}_axes().joined(a, b) for b in fc.axes()] for a in fc.axes()]"


def _x02(rng):
    n = rng.choice([40, 60, 80, 100])
    return f"x = np.linspace(0, 2 * np.pi, {n})", n


def _lr02(rng) -> tuple[str, float]:
    """A step-decay learning-rate schedule `lr` over `epochs` (the setup must define epochs)."""
    base, step = rng.choice([0.1, 0.05, 0.01]), rng.choice([5, 8, 10])
    return f"lr = np.round({base} * 0.5 ** ((epochs - 1) // {step}), 6)", base


MOSAICS = ["AAB;CCB", "AB;CC", "ABC;DDD", "AA;BC", "A.;BC", "AAA;BCD", "AB;AC;AD", "ABB;ACC", "A.B;CCC", "AB;CB"]


def _mosaic_spans(s: str) -> dict:
    rows = s.split(";")
    out = {}
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            if ch != ".":
                r0, r1, c0, c1 = out.get(ch, (i, i + 1, j, j + 1))
                out[ch] = (min(r0, i), max(r1, i + 1), min(c0, j), max(c1, j + 1))
    return out


# ================================================================= lite

@template(M, "lite")
def subplot_grids(rng):
    code, n = _x02(rng)
    r, c = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4), (3, 3)])
    i, j = rng.randrange(r), rng.randrange(c)
    w, h = rng.choice([(8, 3), (10, 4), (9, 2.5), (6, 6)])
    a = rng.choice([2, 3, 4])
    items = [
        ("plt.subplots", f"Create a {r} × {c} grid with `fig, axes = plt.subplots({r}, {c})` and plot `np.sin(x)` on the "
                         f"Axes in row {i}, column {j} (counting from 0).",
         f"fig, axes = plt.subplots({r}, {c})\naxes[{i}, {j}].plot(x, np.sin(x))", {"post": f"fc.grid_shape(), {NLINES}"}),
        (".shape", f"`plt.subplots({r}, {c})` hands back the Axes as a 2-D NumPy array. Create it and return `axes.shape`.",
         f"fig, axes = plt.subplots({r}, {c})\nresult = axes.shape"),
        (".flat", f"Create a {r} × {c} grid and loop over `axes.flat` (row by row): on the k-th Axes plot `np.sin(x + k)`.",
         f"fig, axes = plt.subplots({r}, {c})\nfor k, ax in enumerate(axes.flat):\n    ax.plot(x, np.sin(x + k))",
         {"post": f"fc.grid_shape(), {YDATA}"}),
        ("figsize", f"Create a row of {c} Axes in a {w} × {h} inch figure (`figsize`) and plot `np.cos(x)` on the last one.",
         f"fig, axes = plt.subplots(1, {c}, figsize=({w}, {h}))\naxes[-1].plot(x, np.cos(x))",
         {"post": f"fc.size(), fc.grid_shape(), {NLINES}"}),
        ("sharex", f"Create {r} Axes stacked vertically that **share the x-axis** (`sharex=True`), and plot "
                   f"`np.sin((k + 1) * x)` on the k-th one (k = 0 at the top).",
         f"fig, axes = plt.subplots({r}, 1, sharex=True)\nfor k, ax in enumerate(axes):\n    ax.plot(x, np.sin((k + 1) * x))",
         {"post": f"fc.grid_shape(), {_share('x')}, {YDATA}"}),
        ("sharey", f"Create a 2 × {c} grid in which the Axes of each **row** share their y-axis (`sharey='row'`) and "
                   f"nothing else is shared.",
         f"fig, axes = plt.subplots(2, {c}, sharey='row')", {"post": f"fc.grid_shape(), {_share('y')}, {_share('x')}"}),
        ("layout", f"Create a {r} × {c} grid that uses the **constrained** layout engine (`layout='constrained'`), which "
                   f"keeps titles and tick labels from overlapping.",
         f"fig, axes = plt.subplots({r}, {c}, layout='constrained')", {"post": f"fc.grid_shape(), {ENGINE}"}),
        ("width_ratios", f"Create two side-by-side Axes, unpacked directly as `fig, (ax1, ax2) = plt.subplots(1, 2, ...)`, "
                         f"with the left one {a}× as wide as the right one (`width_ratios`). Plot `np.sin(x)` on `ax1`.",
         f"fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[{a}, 1])\nax1.plot(x, np.sin(x))",
         {"post": f"list(fc.ax(0).get_subplotspec().get_gridspec().get_width_ratios()), {NLINES}"}),
        ("height_ratios", f"Create two stacked Axes, the top one {a}× as tall as the bottom one (`height_ratios`), and "
                          f"plot `np.cos(x)` on the bottom one.",
         f"fig, (top, bottom) = plt.subplots(2, 1, height_ratios=[{a}, 1])\nbottom.plot(x, np.cos(x))",
         {"post": f"list(fc.ax(0).get_subplotspec().get_gridspec().get_height_ratios()), {NLINES}"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "lite")
def figure_labels(rng):
    code, n = _x02(rng)
    r, c = rng.choice([(2, 2), (2, 3), (3, 2), (2, 4)])
    i, j = rng.randrange(r), rng.randrange(c)
    title = rng.choice(["Grid of subplots", "Experiment overview", "Signals", "Ablations"])
    xl, yl = rng.choice([("step", "value"), ("epoch", "loss"), ("time (s)", "amplitude")])
    mk = f"fig, axes = plt.subplots({r}, {c})\n"
    items = [
        ("suptitle", f"Create a {r} × {c} grid and give the **whole figure** the title {title!r} (`fig.suptitle`).",
         mk + f"fig.suptitle({title!r})", {"post": f"fc.suptitles()['suptitle'], fc.grid_shape()"}),
        ("supxlabel", f"Create a {r} × {c} grid and give it **one** x label for the whole figure, {xl!r}, centred under "
                      f"all the panels (`fig.supxlabel`).", mk + f"fig.supxlabel({xl!r})",
         {"post": "fc.suptitles(), fc.grid_shape()"}),
        ("supylabel", f"Create a {r} × {c} grid and give it one figure-level y label, {yl!r} (`fig.supylabel`).",
         mk + f"fig.supylabel({yl!r})", {"post": "fc.suptitles(), fc.grid_shape()"}),
        ("set_title", f"Create a {r} × {c} grid and title **only** the Axes in row {i}, column {j}: "
                      f"{f'row {i}, col {j}'!r}.", mk + f"axes[{i}, {j}].set_title('row {i}, col {j}')",
         {"post": TITLES}),
        ("label_outer", f"Create a {r} × {c} grid (nothing shared), plot `np.sin(x)` on every Axes, then call "
                        f"`ax.label_outer()` on each one, so only the bottom row keeps its x tick labels and only the "
                        f"left column its y tick labels.",
         mk + "for ax in axes.flat:\n    ax.plot(x, np.sin(x))\n    ax.label_outer()", {"post": f"{TICKLABELS}, {NLINES}"}),
        ("set_xlabel", f"Create a {r} × {c} grid. Label the x-axis {xl!r} on the **bottom row** only and the y-axis "
                       f"{yl!r} on the **left column** only.",
         mk + f"for ax in axes[-1, :]:\n    ax.set_xlabel({xl!r})\nfor ax in axes[:, 0]:\n    ax.set_ylabel({yl!r})",
         {"post": "[fc.labels(a) for a in fc.axes()]", "require": ["set_xlabel", "set_ylabel"]}),
        ("set_title", f"Create a {r} × {c} grid and title every Axes `'panel k'`, where k counts the panels row by row "
                      f"from 1 (`enumerate(axes.flat, 1)`).",
         mk + "for k, ax in enumerate(axes.flat, 1):\n    ax.set_title(f'panel {k}')", {"post": TITLES}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "lite")
def placing_axes(rng):
    code, n = _x02(rng)
    r, c = rng.choice([(2, 2), (2, 3), (3, 2), (1, 3)])
    k = rng.randint(1, r * c)
    l, b, w, h = rng.choice([(0.1, 0.1, 0.8, 0.8), (0.55, 0.1, 0.4, 0.3), (0.2, 0.5, 0.6, 0.4), (0.05, 0.6, 0.3, 0.35)])
    mos = rng.choice(MOSAICS)
    ib = rng.choice([[0.6, 0.6, 0.35, 0.35], [0.05, 0.6, 0.3, 0.3], [0.55, 0.08, 0.4, 0.4]])
    gr, gc = rng.choice([(2, 2), (3, 3), (2, 3), (3, 2)])
    items = [
        ("add_subplot", f"Start from an empty `fig = plt.figure()` and add **only** the Axes at position {k} of a {r} × {c} "
                        f"grid (`fig.add_subplot({r}, {c}, {k})`); plot `np.sin(x)` on it.",
         f"fig = plt.figure()\nax = fig.add_subplot({r}, {c}, {k})\nax.plot(x, np.sin(x))", {"post": f"{SPANS}, {NLINES}"}),
        ("projection", f"Start from `fig = plt.figure()` and add a **polar** Axes at position {k} of a {r} × {c} grid "
                       f"(`projection='polar'`); plot `np.sin(x)` on it (θ = `x`, r = `np.sin(x)`).",
         f"fig = plt.figure()\nax = fig.add_subplot({r}, {c}, {k}, projection='polar')\nax.plot(x, np.sin(x))",
         {"post": f"[a.name for a in fc.axes()], {SPANS}, {NLINES}"}),
        ("add_axes", f"Place an Axes by hand with `fig.add_axes`: left {l}, bottom {b}, width {w}, height {h} (fractions "
                     f"of the figure). Plot `np.cos(x)` on it.",
         f"fig = plt.figure()\nax = fig.add_axes(({l}, {b}, {w}, {h}))\nax.plot(x, np.cos(x))",
         {"post": f"[fc.r(a.get_position().bounds, 4) for a in fc.axes()], {NLINES}"}),
        ("add_gridspec", f"Make `gs = fig.add_gridspec({gr}, {gc})` on a new figure and add two Axes: one spanning the "
                         f"whole **top row** (`gs[0, :]`) and one in the **bottom-left** cell (`gs[-1, 0]`).",
         f"fig = plt.figure()\ngs = fig.add_gridspec({gr}, {gc})\ntop = fig.add_subplot(gs[0, :])\n"
         f"corner = fig.add_subplot(gs[-1, 0])", {"post": SPANS}),
        ("subplot_mosaic", f"Build the layout `{mos!r}` with `fig, axd = plt.subplot_mosaic({mos!r})` (one letter per "
                           f"panel, `;` separates rows, `.` leaves a cell empty) and plot `np.sin(x)` on `axd['A']`.",
         f"fig, axd = plt.subplot_mosaic({mos!r})\naxd['A'].plot(x, np.sin(x))",
         {"post": f"sorted(zip(fc.spans(), {NLINES})), fc.grid_shape()"}),
        ("inset_axes", f"Create `fig, ax`, plot `np.sin(x)`, and add a small Axes **inside** it with "
                       f"`ax.inset_axes({ib})` ([left, bottom, width, height] as fractions of `ax`).",
         f"fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nins = ax.inset_axes({ib})", {"post": f"{CHILDREN}, {NLINES}"}),
        ("twinx", "Create `fig, ax` and plot `np.sin(x)` on it; then make a second y-axis on the right that shares x "
                  "(`ax2 = ax.twinx()`) and plot `100 * np.cos(x)` on it.",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nax2 = ax.twinx()\nax2.plot(x, 100 * np.cos(x))",
         {"post": f"{YDATA}, {_share('x')}, fc.ax(1).yaxis.get_label_position()"}),
        ("secondary_xaxis", "Create `fig, ax`, plot `np.sin(x)` (x in radians) and add a second x-axis on **top** that "
                            "shows the same positions in degrees: `ax.secondary_xaxis('top', functions=(np.rad2deg, "
                            "np.deg2rad))`.",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nsec = ax.secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))",
         {"post": "_c = fc.ax(0).child_axes\n[type(c).__name__ for c in _c], fc.r(_c[0]._functions[0](np.array([np.pi, 1.0])), 4)",
          "require": ["secondary_xaxis", "rad2deg", "deg2rad"]}),
        ("subfigures", "Split a new figure into two side-by-side **subfigures** (`left, right = fig.subfigures(1, 2)`), "
                       "give the left one 2 stacked Axes (`left.subplots(2, 1)`) and the right one a single Axes.",
         "fig = plt.figure()\nleft, right = fig.subfigures(1, 2)\nleft.subplots(2, 1)\nright.subplots()",
         {"post": "[len(s.axes) for s in fc.fig().subfigs], len(fc.axes())"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "lite")
def layout_tuning(rng):
    code, n = _x02(rng)
    r, c = rng.choice([(2, 2), (2, 3), (3, 2)])
    ws, hs = rng.choice([(0.4, 0.5), (0.05, 0.1), (0.6, 0.3), (0.0, 0.0)])
    lft, rgt = rng.choice([(0.15, 0.8), (0.1, 0.85), (0.2, 0.95)])
    w, h = rng.choice([(4, 3), (6, 2), (5, 4)])
    k = rng.randint(3, 8)
    mk = f"fig, axes = plt.subplots({r}, {c})\n"
    items = [
        ("subplots_adjust", f"Create a {r} × {c} grid and set the horizontal gap between panels to {ws} and the "
                            f"vertical gap to {hs} (fractions of the average Axes size) with `fig.subplots_adjust`.",
         mk + f"fig.subplots_adjust(wspace={ws}, hspace={hs})", {"post": f"{SUBPARS}, fc.grid_shape()"}),
        ("subplots_adjust", f"Create a {r} × {c} grid and move the left edge of the grid to {lft} and the right edge to "
                            f"{rgt} of the figure width (`fig.subplots_adjust(left=..., right=...)`), e.g. to leave room "
                            f"for a legend.", mk + f"fig.subplots_adjust(left={lft}, right={rgt})",
         {"post": f"{SUBPARS}, fc.grid_shape()"}),
        ("tight_layout", f"Create a {r} × {c} grid, give every Axes the title 'a fairly long title', plot `np.sin(x)` on "
                         f"each, and call `fig.tight_layout()` so titles and tick labels no longer overlap.",
         mk + "for ax in axes.flat:\n    ax.plot(x, np.sin(x))\n    ax.set_title('a fairly long title')\nfig.tight_layout()",
         {"post": f"{SUBPARS}, {TITLES}"}),
        ("layout", f"Create a {w} × {h} inch figure with `plt.figure(figsize=..., layout='constrained')` and then add a "
                   f"{r} × {c} grid to it with `fig.subplots({r}, {c})`.",
         f"fig = plt.figure(figsize=({w}, {h}), layout='constrained')\naxes = fig.subplots({r}, {c})",
         {"post": f"{ENGINE}, fc.size(), fc.grid_shape()"}),
        ("align_labels", "Create 2 stacked Axes; plot `np.sin(x)` on the top one and `1000 * np.sin(x)` on the bottom "
                         "one (its tick labels are wider), label the y-axes 'small' and 'large', and line the two y "
                         "labels up with `fig.align_labels()`.",
         "fig, (top, bottom) = plt.subplots(2, 1)\ntop.plot(x, np.sin(x))\nbottom.plot(x, 1000 * np.sin(x))\n"
         "top.set_ylabel('small')\nbottom.set_ylabel('large')\nfig.align_labels()",
         {"post": "fc.fig()._align_label_groups['y'].joined(fc.ax(0), fc.ax(1)), [a.get_ylabel() for a in fc.axes()]"}),
        ("savefig", f"Create a {r} × {c} grid in a {w} × {h} inch figure, save it as `grid.png` at 20 dpi and return the "
                    f"shape of the saved image (`plt.imread('grid.png').shape`).",
         f"fig, axes = plt.subplots({r}, {c}, figsize=({w}, {h}))\nfig.savefig('grid.png', dpi=20)\n"
         f"result = plt.imread('grid.png').shape"),
        ("hspace", f"Build a {r} × {c} grid by hand: `gs = fig.add_gridspec({r}, {c}, hspace={hs}, wspace={ws})` on a new "
                   f"figure, then one `fig.add_subplot(gs[i, j])` per cell.",
         f"fig = plt.figure()\ngs = fig.add_gridspec({r}, {c}, hspace={hs}, wspace={ws})\nfor i in range({r}):\n"
         f"    for j in range({c}):\n        fig.add_subplot(gs[i, j])",
         {"post": "_g = fc.ax(0).get_subplotspec().get_gridspec()\n" + f"{SPANS}, fc.r([_g.hspace, _g.wspace], 4)",
          "require": ["hspace", "wspace"]}),
        ("plt.close", f"Create {k} figures in a loop, each a 2 × 2 grid from `plt.subplots(2, 2)`; add up how many Axes "
                      f"they hold (`len(fig.axes)`) and close each one right away. Return `(total Axes, figures still "
                      f"open)`.",
         f"total = 0\nfor _ in range({k}):\n    fig, axes = plt.subplots(2, 2)\n    total += len(fig.axes)\n"
         f"    plt.close(fig)\nresult = (total, len(plt.get_fignums()))"),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


# ================================================================= easy

@template(M, "easy")
def grid_readback(rng):
    code, n = _x02(rng)
    r, c = rng.choice([(2, 3), (3, 2), (2, 4), (3, 3), (4, 2)])
    i, j = rng.randrange(r), rng.randrange(c)
    k = rng.randrange(r * c)
    lo, hi = rng.choice([(0, 3), (1, 5), (-2, 2)])
    items = [
        (".axes", f"`fig.axes` lists a figure's Axes in the order they were made: row by row for `plt.subplots`. Create "
                  f"a {r} × {c} grid and return the position of `axes[{i}, {j}]` in `fig.axes`.",
         f"fig, axes = plt.subplots({r}, {c})\nresult = fig.axes.index(axes[{i}, {j}])"),
        ("plt.subplots", f"`plt.subplots` squeezes away length-1 dimensions. Return `[a.ndim, b.ndim, type(c).__name__]` "
                         f"for `a` from `plt.subplots({r}, {c})`, `b` from `plt.subplots(1, {c})` and `c` from "
                         f"`plt.subplots()` (the second value each call returns).",
         f"_, a = plt.subplots({r}, {c})\n_, b = plt.subplots(1, {c})\n_, c = plt.subplots()\n"
         f"result = [a.ndim, b.ndim, type(c).__name__]"),
        ("squeeze", f"With `squeeze=False` you always get a 2-D array, which keeps code uniform. Create "
                    f"`plt.subplots(1, {c}, squeeze=False)`, plot `np.sin(x)` on `axes[0, -1]`, and return `axes.shape`.",
         f"fig, axes = plt.subplots(1, {c}, squeeze=False)\naxes[0, -1].plot(x, np.sin(x))\nresult = axes.shape"),
        ("get_subplotspec", f"Create a {r} × {c} grid and return the (row, column) of `fig.axes[{k}]`, read from "
                            f"`ax.get_subplotspec().rowspan.start` and `.colspan.start`.",
         f"fig, axes = plt.subplots({r}, {c})\nss = fig.axes[{k}].get_subplotspec()\n"
         f"result = (ss.rowspan.start, ss.colspan.start)"),
        ("get_position", f"Create a 1 × {c} grid, set `fig.subplots_adjust(left=0.1, right=0.9, wspace=0)`, and return "
                         f"each Axes' width as a fraction of the figure (`ax.get_position().width`, rounded to 4 "
                         f"decimals).",
         f"fig, axes = plt.subplots(1, {c})\nfig.subplots_adjust(left=0.1, right=0.9, wspace=0)\n"
         f"result = [round(ax.get_position().width, 4) for ax in axes]", {"require": ["subplots_adjust", "get_position"]}),
        ("sharex", "Create a 2 × 2 grid with `sharex='col'` and return `[top-left shares x with bottom-left, top-left "
                   "shares x with top-right]` (use `axes[0, 0].get_shared_x_axes().joined(a, b)`).",
         "fig, axes = plt.subplots(2, 2, sharex='col')\ng = axes[0, 0].get_shared_x_axes()\n"
         "result = [g.joined(axes[0, 0], axes[1, 0]), g.joined(axes[0, 0], axes[0, 1])]"),
        ("set_xlim", f"Create {r} stacked Axes with `sharex=True`, set the x limits {lo} … {hi} on the **top** one only, "
                     f"and return every Axes' x limits (`ax.get_xlim()`): sharing propagates them.",
         f"fig, axes = plt.subplots({r}, 1, sharex=True)\naxes[0].set_xlim({lo}, {hi})\n"
         f"result = [list(ax.get_xlim()) for ax in axes]"),
        (".flat", f"Plot `np.sin(x)` on every **other** Axes of a {r} × {c} grid (positions 0, 2, 4, … of `axes.flat`) "
                  f"and return the number of lines per row (a list of {r} counts).",
         f"fig, axes = plt.subplots({r}, {c})\nfor ax in axes.flat[::2]:\n    ax.plot(x, np.sin(x))\n"
         f"result = [sum(len(ax.get_lines()) for ax in row) for row in axes]"),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "easy")
def mosaic_layouts(rng):
    code, n = _x02(rng)
    mos = rng.choice(MOSAICS)
    keys = sorted(_mosaic_spans(mos))
    lab = rng.choice(keys)
    a = rng.choice([3, 4, 5])
    names = rng.choice([("main", "side"), ("curve", "hist"), ("plot", "legend")])
    s = seeded(rng)
    items = [
        ("subplot_mosaic", f"Build the mosaic `{mos!r}` and plot `np.sin(x)` on **every** panel (loop over "
                           f"`axd.values()`). Return the panel names, sorted.",
         f"fig, axd = plt.subplot_mosaic({mos!r})\nfor ax in axd.values():\n    ax.plot(x, np.sin(x))\nresult = sorted(axd)",
         {"post": f"_, sorted(zip(fc.spans(), {NLINES}))"}),
        ("get_subplotspec", f"Build the mosaic `{mos!r}` and return the rows and columns panel {lab!r} covers, as "
                            f"`(rows.start, rows.stop, cols.start, cols.stop)` from its "
                            f"`get_subplotspec().rowspan` / `.colspan`.",
         f"fig, axd = plt.subplot_mosaic({mos!r})\nss = axd[{lab!r}].get_subplotspec()\n"
         f"result = (ss.rowspan.start, ss.rowspan.stop, ss.colspan.start, ss.colspan.stop)"),
        ("subplot_mosaic", f"A mosaic can also be a list of rows. Build `[[{names[0]!r}, {names[1]!r}]]` with the left "
                           f"panel {a}× as wide (`width_ratios=[{a}, 1]`); plot `np.sin(x)` on {names[0]!r} and a "
                           f"histogram of `np.sin(x)` on {names[1]!r}.",
         f"fig, axd = plt.subplot_mosaic([[{names[0]!r}, {names[1]!r}]], width_ratios=[{a}, 1])\n"
         f"axd[{names[0]!r}].plot(x, np.sin(x))\naxd[{names[1]!r}].hist(np.sin(x))",
         {"post": f"[(s, len(a.get_lines()), len(a.patches)) for s, a in zip(fc.spans(), fc.axes())], "
                  f"list(fc.ax(0).get_subplotspec().get_gridspec().get_width_ratios())", "require": ["subplot_mosaic", "hist"]}),
        ("subplot_mosaic", "Reproduce the guide's layout: a tall 'main' panel on the left of two stacked panels, 'side' "
                           "(top right) and 'bottom' (bottom right): `[['main', 'side'], ['main', 'bottom']]`. Plot "
                           "`np.sin(x)` on 'main', a histogram of `noise` on 'side' and bars `[3, 1, 2]` at x = 1, 2, 3 "
                           "on 'bottom'.",
         "fig, axd = plt.subplot_mosaic([['main', 'side'], ['main', 'bottom']])\naxd['main'].plot(x, np.sin(x))\n"
         "axd['side'].hist(noise)\naxd['bottom'].bar([1, 2, 3], [3, 1, 2])",
         {"post": "sorted((s, len(a.get_lines()), len(a.patches)) for s, a in zip(fc.spans(), fc.axes()))",
          "setup": f"noise = np.random.default_rng({s}).normal(size=200)", "require": ["subplot_mosaic", "hist", "bar"]}),
        ("subplot_mosaic", f"Build the mosaic `{mos!r}` with **shared x-axes** across all panels (`sharex=True`) and "
                           f"return the number of Axes the figure holds.",
         f"fig, axd = plt.subplot_mosaic({mos!r}, sharex=True)\nresult = len(fig.axes)", {"post": f"_, {_share('x')}"}),
        ("empty_sentinel", "In a mosaic, `'.'` marks an empty cell. Build `'A.;BC'` and return `[len(fig.axes), "
                           "sorted(axd)]`.", "fig, axd = plt.subplot_mosaic('A.;BC')\nresult = [len(fig.axes), sorted(axd)]",
         {"require": ["subplot_mosaic"]}),
        ("per_subplot_kw", f"Build the mosaic `'AB'` where panel 'B' is **polar**: pass "
                           f"`per_subplot_kw={{'B': {{'projection': 'polar'}}}}`. Plot `np.sin(x)` on both panels.",
         "fig, axd = plt.subplot_mosaic('AB', per_subplot_kw={'B': {'projection': 'polar'}})\n"
         "for ax in axd.values():\n    ax.plot(x, np.sin(x))", {"post": f"[a.name for a in fc.axes()], {NLINES}"}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "easy")
def twin_and_secondary(rng):
    code, desc, n = curves(rng, acc=True)
    lr_code, _ = _lr02(rng)
    spe = rng.choice([50, 100, 250, 390])
    c1, c2 = rng.choice([("C0", "C1"), ("tab:blue", "tab:red"), ("black", "tab:orange")])
    items = [
        ("twinx", f"Plot `train_loss` against `epochs` on `ax` in {c1!r}; then add a twin y-axis (`ax.twinx()`) and plot "
                  f"`lr` on it in {c2!r}. Label the left y-axis 'loss' and the right one 'learning rate', each in its "
                  f"line's color (`set_ylabel(..., color=...)`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss, color={c1!r})\nax.set_ylabel('loss', color={c1!r})\n"
         f"ax2 = ax.twinx()\nax2.plot(epochs, lr, color={c2!r})\nax2.set_ylabel('learning rate', color={c2!r})",
         {"post": "[(fc.lines(a, 'color'), a.get_ylabel(), fc.hexcolor(a.yaxis.label.get_color())) for a in fc.axes()]"}),
        ("twiny", "`ax.twiny()` is the mirror image of `twinx`: a second **x**-axis (on top) sharing y. Plot `val_loss` "
                  "against `epochs` on `ax`, then plot `val_loss` against `epochs / epochs[-1]` (fraction of training) "
                  "on `ax.twiny()`, with that top axis labelled 'fraction of training'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\ntop = ax.twiny()\ntop.plot(epochs / epochs[-1], val_loss)\n"
         "top.set_xlabel('fraction of training')",
         {"post": f"{YDATA}, {_share('y')}, fc.ax(1).get_xlabel()"}),
        ("secondary_xaxis", f"Each epoch is {spe} optimizer steps. Plot `train_loss` against `epochs` and add a top "
                            f"x-axis counting **steps**: `ax.secondary_xaxis('top', functions=(to_steps, to_epochs))` "
                            f"with two lambdas that convert back and forth; label it 'step'.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\n"
         f"sec = ax.secondary_xaxis('top', functions=(lambda e: e * {spe}, lambda s: s / {spe}))\nsec.set_xlabel('step')",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([1.0, 3.0])), 4), "
                  "fc.r(_c._functions[1](np.array([100.0])), 4), _c.get_xlabel()"}),
        ("secondary_yaxis", "Plot `val_acc` (a fraction) against `epochs` and add a right-hand y-axis that reads the same "
                            "values in **percent**: `ax.secondary_yaxis('right', functions=(...))`, labelled '%'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_acc)\n"
         "sec = ax.secondary_yaxis('right', functions=(lambda a: 100 * a, lambda p: p / 100))\nsec.set_ylabel('%')",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([0.5, 0.9])), 4), "
                  "fc.r(_c._functions[1](np.array([50.0])), 4), _c.get_ylabel()"}),
        ("twinx", "Twin Axes share x limits. Plot `train_loss` against `epochs` on `ax`, create `ax2 = ax.twinx()`, then "
                  "set `ax.set_xlim(1, 10)` and return `list(ax2.get_xlim())`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax2 = ax.twinx()\nax.set_xlim(1, 10)\n"
         "result = list(ax2.get_xlim())"),
        ("secondary_yaxis", "Cross-entropy loss and perplexity are linked by `perplexity = exp(loss)`. Plot `val_loss` "
                            "against `epochs` and add a right-hand axis showing perplexity with "
                            "`functions=(np.exp, np.log)`, labelled 'perplexity'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nsec = ax.secondary_yaxis('right', functions=(np.exp, np.log))\n"
         "sec.set_ylabel('perplexity')",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([0.0, 1.0])), 4), _c.get_ylabel()"}),
    ]
    return catalog(rng, items, f"{code}\n{lr_code}", f"{desc}, and `lr`, the learning rate of each epoch.")


@template(M, "easy")
def gridspec_spans(rng):
    code, n = _x02(rng)
    a, b = rng.choice([(3, 1), (4, 1), (2, 1), (5, 2)])
    r, c = rng.choice([(2, 3), (3, 3), (3, 4), (2, 4)])
    k1, k2 = sorted(rng.sample(range(1, c + 1), 2))
    items = [
        ("add_gridspec", "Reproduce the guide's layout on a 3 × 3 gridspec (`gs = fig.add_gridspec(3, 3)`): `top` spans "
                         "the first row (`gs[0, :]`), `left` the 2 × 2 block below it (`gs[1:, :2]`) and `right` the "
                         "rest of the last column (`gs[1:, 2]`). Plot `np.sin(x)` on `left`.",
         "fig = plt.figure()\ngs = fig.add_gridspec(3, 3)\ntop = fig.add_subplot(gs[0, :])\nleft = fig.add_subplot(gs[1:, :2])\n"
         "right = fig.add_subplot(gs[1:, 2])\nleft.plot(x, np.sin(x))", {"post": f"{SPANS}, {NLINES}"}),
        ("width_ratios", f"Make a 2 × 2 gridspec with `width_ratios=[{a}, {b}]` and `height_ratios=[{b}, {a}]`, the classic "
                         f"scatter-with-marginals layout, and add three Axes: `main` (bottom-left), `top` (top-left) and "
                         f"`side` (bottom-right).",
         f"fig = plt.figure()\ngs = fig.add_gridspec(2, 2, width_ratios=[{a}, {b}], height_ratios=[{b}, {a}])\n"
         f"main = fig.add_subplot(gs[1, 0])\ntop = fig.add_subplot(gs[0, 0])\nside = fig.add_subplot(gs[1, 1])",
         {"post": "_g = fc.ax(0).get_subplotspec().get_gridspec()\n"
                  "sorted(fc.spans()), list(_g.get_width_ratios()), list(_g.get_height_ratios())",
          "require": ["width_ratios", "height_ratios"]}),
        ("add_subplot", f"`add_subplot` accepts a **range** of cells: `fig.add_subplot({r}, {c}, ({k1}, {k2}))` spans "
                        f"positions {k1} … {k2} of a {r} × {c} grid. Add that Axes to a new figure and return its "
                        f"columns as `(colspan.start, colspan.stop)`.",
         f"fig = plt.figure()\nax = fig.add_subplot({r}, {c}, ({k1}, {k2}))\nss = ax.get_subplotspec()\n"
         f"result = (ss.colspan.start, ss.colspan.stop)"),
        ("add_gridspec", f"Fill a {r} × {c} gridspec cell by cell with two nested loops (`fig.add_subplot(gs[i, j])`) and "
                         f"plot `np.sin(x + i + j)` in each. Return `len(fig.axes)`.",
         f"fig = plt.figure()\ngs = fig.add_gridspec({r}, {c})\nfor i in range({r}):\n    for j in range({c}):\n"
         f"        fig.add_subplot(gs[i, j]).plot(x, np.sin(x + i + j))\nresult = len(fig.axes)",
         {"post": f"_, {SPANS}, {YDATA}"}),
        ("subgridspec", "Nest grids: on a 1 × 2 gridspec, put one Axes in the left cell and split the right cell into a "
                        "2 × 2 grid with `gs[1].subgridspec(2, 2)`, adding all four small Axes. Return `len(fig.axes)`.",
         "fig = plt.figure()\ngs = fig.add_gridspec(1, 2)\nfig.add_subplot(gs[0])\ninner = gs[1].subgridspec(2, 2)\n"
         "for i in range(2):\n    for j in range(2):\n        fig.add_subplot(inner[i, j])\nresult = len(fig.axes)",
         {"post": "_, fc.spans()"}),
        ("add_gridspec", f"Make a {r} × {c} gridspec and add one Axes covering everything **except** the last row and "
                         f"last column (`gs[:-1, :-1]`). Return its `get_subplotspec().rowspan` and `.colspan` as "
                         f"`(rows.stop, cols.stop)`.",
         f"fig = plt.figure()\ngs = fig.add_gridspec({r}, {c})\nax = fig.add_subplot(gs[:-1, :-1])\nss = ax.get_subplotspec()\n"
         f"result = (ss.rowspan.stop, ss.colspan.stop)"),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


@template(M, "easy")
def pyplot_to_explicit_layouts(rng):
    code, n = _x02(rng)
    title = rng.choice(["Overview", "Four signals", "Phase shifts"])
    ws = rng.choice([0.3, 0.4, 0.5])
    items = [
        ("plt.subplots", "Rewrite this pyplot code in the explicit style with `fig, axes = plt.subplots(2, 2)` (only "
                         "`axes[...]` and `fig.` methods):\n```python\nfor k in range(4):\n    plt.subplot(2, 2, k + 1)\n"
                         "    plt.plot(x, np.sin(x + k))\n    plt.title(f'shift {k}')\n```",
         "fig, axes = plt.subplots(2, 2)\nfor k, ax in enumerate(axes.flat):\n    ax.plot(x, np.sin(x + k))\n"
         "    ax.set_title(f'shift {k}')", {"post": f"{SPANS}, {YDATA}, {TITLES}"}),
        ("plt.subplot", "Rewrite this explicit code with **pyplot only** (`plt.subplot(1, 2, k)` selects a panel, then "
                        "`plt.plot` / `plt.title` act on it):\n```python\nfig, (a, b) = plt.subplots(1, 2)\n"
                        "a.plot(x, np.sin(x)); a.set_title('sin')\nb.plot(x, np.cos(x)); b.set_title('cos')\n```",
         "plt.subplot(1, 2, 1)\nplt.plot(x, np.sin(x))\nplt.title('sin')\nplt.subplot(1, 2, 2)\nplt.plot(x, np.cos(x))\n"
         "plt.title('cos')", {"post": f"{SPANS}, {YDATA}, {TITLES}", "require": ["subplot", "title"]}),
        ("suptitle", f"Translate to the explicit style (`fig.` methods):\n```python\nplt.subplots(1, 2)\n"
                     f"plt.suptitle({title!r})\nplt.subplots_adjust(wspace={ws})\n```",
         f"fig, axes = plt.subplots(1, 2)\nfig.suptitle({title!r})\nfig.subplots_adjust(wspace={ws})",
         {"post": f"fc.suptitles()['suptitle'], {SUBPARS}", "require": ["suptitle", "subplots_adjust"]}),
        ("plt.twinx", "Rewrite with Axes methods (`fig, ax = plt.subplots()` and `ax.twinx()`):\n```python\n"
                      "plt.plot(x, np.sin(x), color='C0')\nplt.ylabel('sin')\nplt.twinx()\n"
                      "plt.plot(x, 10 * np.cos(x), color='C1')\nplt.ylabel('10 cos')\n```",
         "fig, ax = plt.subplots()\nax.plot(x, np.sin(x), color='C0')\nax.set_ylabel('sin')\nax2 = ax.twinx()\n"
         "ax2.plot(x, 10 * np.cos(x), color='C1')\nax2.set_ylabel('10 cos')",
         {"post": "[(fc.lines(a, 'color'), a.get_ylabel()) for a in fc.axes()], " + _share('x'),
          "require": ["twinx", "set_ylabel"]}),
        ("tight_layout", "Translate to the explicit style:\n```python\nplt.subplot(2, 1, 1); plt.plot(x, np.sin(x)); "
                         "plt.xlabel('x')\nplt.subplot(2, 1, 2); plt.plot(x, np.cos(x)); plt.title('cos')\n"
                         "plt.tight_layout()\n```",
         "fig, (a, b) = plt.subplots(2, 1)\na.plot(x, np.sin(x))\na.set_xlabel('x')\nb.plot(x, np.cos(x))\nb.set_title('cos')\n"
         "fig.tight_layout()", {"post": f"{SPANS}, [fc.labels(a) for a in fc.axes()], {SUBPARS}",
                                "require": ["tight_layout", "set_xlabel"]}),
        ("projection", "Rewrite with `fig = plt.figure()` and `fig.add_subplot(...)`:\n```python\nplt.subplot(1, 2, 1)\n"
                       "plt.plot(x, np.sin(x))\nplt.subplot(1, 2, 2, projection='polar')\nplt.plot(x, 1 + np.sin(x))\n```",
         "fig = plt.figure()\na = fig.add_subplot(1, 2, 1)\na.plot(x, np.sin(x))\nb = fig.add_subplot(1, 2, 2, projection='polar')\n"
         "b.plot(x, 1 + np.sin(x))", {"post": f"[a.name for a in fc.axes()], {SPANS}, {YDATA}", "require": ["add_subplot"]}),
    ]
    return catalog(rng, items, code, f"`x` holds {n} angles from 0 to 2π.")


# ================================================================= medium

@template(M, "medium")
def training_dashboard(rng):
    code, desc, n = curves(rng, acc=True)
    lr_code, _ = _lr02(rng)
    title = rng.choice(["Run 12", "Baseline", "ResNet-18 on CIFAR-10", "Ablation: no dropout"])
    w = rng.choice([2, 3])
    items = [
        ("plt.subplots", f"Build a 2 × 2 dashboard sharing the x-axis (`sharex=True`): top-left `train_loss`, top-right "
                         f"`val_loss`, bottom-left `val_acc`, bottom-right `lr`, all against `epochs`, titled 'train "
                         f"loss', 'val loss', 'val acc', 'lr'. Add the figure title {title!r} and one figure-level x "
                         f"label 'epoch'.",
         f"fig, axes = plt.subplots(2, 2, sharex=True)\nfor ax, y, t in zip(axes.flat, (train_loss, val_loss, val_acc, lr), "
         f"('train loss', 'val loss', 'val acc', 'lr')):\n    ax.plot(epochs, y)\n    ax.set_title(t)\n"
         f"fig.suptitle({title!r})\nfig.supxlabel('epoch')",
         {"post": f"{SPANS}, {YDATA}, {TITLES}, {_share('x')}, fc.suptitles()",
          "require": ["subplots", "sharex", "suptitle", "supxlabel"]}),
        ("layout", "Build a 1 × 3 row, 12 × 3 inches, with the constrained layout: the losses ('train' and 'val' "
                   "lines, with a legend) on the left, `val_acc` in the middle and `lr` on the right; titles 'loss', "
                   "'accuracy', 'learning rate'.",
         "fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3), layout='constrained')\n"
         "a.plot(epochs, train_loss, label='train')\na.plot(epochs, val_loss, label='val')\na.legend()\na.set_title('loss')\n"
         "b.plot(epochs, val_acc)\nb.set_title('accuracy')\nc.plot(epochs, lr)\nc.set_title('learning rate')",
         {"post": f"fc.size(), {ENGINE}, {SPANS}, {YDATA}, {TITLES}, [fc.legend(a) for a in fc.axes()]"}),
        ("align_labels", "Stack three panels sharing x (`plt.subplots(3, 1, sharex=True)`): `train_loss`, `val_acc` and "
                         "`lr` (×1000, so its tick labels are wider) against `epochs`, with y labels 'loss', 'acc', "
                         "'lr × 1000'. Only the bottom panel gets the x label 'epoch'. Line the y labels up with "
                         "`fig.align_labels()`.",
         "fig, axes = plt.subplots(3, 1, sharex=True)\nfor ax, y, name in zip(axes, (train_loss, val_acc, lr * 1000), "
         "('loss', 'acc', 'lr × 1000')):\n    ax.plot(epochs, y)\n    ax.set_ylabel(name)\naxes[-1].set_xlabel('epoch')\n"
         "fig.align_labels()",
         {"post": "[fc.labels(a) for a in fc.axes()], " + YDATA + ", fc.fig()._align_label_groups['y'].joined(fc.ax(0), fc.ax(2))"}),
        ("width_ratios", f"Give the loss curves the room: a 1 × 2 figure whose left panel is {w}× wider. Left: "
                         f"`train_loss` and `val_loss` against `epochs`; right: `val_acc`. Title them 'loss' and 'acc'.",
         f"fig, (a, b) = plt.subplots(1, 2, width_ratios=[{w}, 1])\na.plot(epochs, train_loss)\na.plot(epochs, val_loss)\n"
         f"a.set_title('loss')\nb.plot(epochs, val_acc)\nb.set_title('acc')",
         {"post": f"list(fc.ax(0).get_subplotspec().get_gridspec().get_width_ratios()), {YDATA}, {TITLES}"}),
        ("label_outer", "Draw a 2 × 2 grid sharing **both** axes (`sharex=True, sharey=True`): `train_loss`, `val_loss`, "
                        "`train_loss - val_loss` and `np.abs(train_loss - val_loss)` against `epochs`. Give every panel "
                        "the y label 'loss' and the x label 'epoch', then call `ax.label_outer()` on each so the inner "
                        "axis labels disappear. Return the y labels that are still visible "
                        "(`[ax.get_ylabel() for ax in axes.flat]`).",
         "fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)\nfor ax, y in zip(axes.flat, (train_loss, val_loss, "
         "train_loss - val_loss, np.abs(train_loss - val_loss))):\n    ax.plot(epochs, y)\n    ax.set(xlabel='epoch', ylabel='loss')\n"
         "    ax.label_outer()\nresult = [ax.get_ylabel() for ax in axes.flat]",
         {"post": f"_, {YDATA}, {TICKLABELS}, [a.xaxis.label.get_visible() for a in fc.axes()]"}),
        ("sharey", "Two runs, side by side, on the **same** y scale so they can be compared: plot `train_loss` on the left "
                   "and `val_loss` on the right of `plt.subplots(1, 2, sharey=True)`, titled 'train' and 'val'. Return "
                   "both Axes' y limits.",
         "fig, (a, b) = plt.subplots(1, 2, sharey=True)\na.plot(epochs, train_loss)\nb.plot(epochs, val_loss)\n"
         "a.set_title('train')\nb.set_title('val')\nresult = [list(a.get_ylim()), list(b.get_ylim())]",
         {"post": f"_, {YDATA}, {TITLES}"}),
        ("set_axis_off", "Turn one panel of a 2 × 2 dashboard into a text box: plot `train_loss`, `val_loss` and "
                         "`val_acc` against `epochs` on the first three panels (row by row), then switch the axis of "
                         "the bottom-right panel off (`ax.set_axis_off()`) and write `f'best epoch: {best}'` at its "
                         "centre with `ax.text(0.5, 0.5, ..., ha='center')`, where best is the epoch with the lowest "
                         "`val_loss`.",
         "fig, axes = plt.subplots(2, 2)\nfor ax, y in zip(axes.flat, (train_loss, val_loss, val_acc)):\n    ax.plot(epochs, y)\n"
         "best = epochs[val_loss.argmin()]\nnote = axes[1, 1]\nnote.set_axis_off()\nnote.text(0.5, 0.5, f'best epoch: {best}', ha='center')",
         {"post": f"{YDATA}, [a.axison for a in fc.axes()], fc.text_items(fc.ax(3))", "require": ["set_axis_off", "text"]}),
    ]
    return catalog(rng, items, f"{code}\n{lr_code}", f"{desc}, and `lr`, each epoch's learning rate.")


@template(M, "medium")
def class_panels(rng):
    code, desc, k, names = classes(rng, k=(2, 4))
    cols = rng.choice([2, 3])
    items = [
        ("sharey", f"Small multiples: make one panel per class in a single row (`plt.subplots(1, {k}, sharex=True, "
                   f"sharey=True)`), scatter that class' points in it and title it with the class name.",
         f"fig, axes = plt.subplots(1, {k}, sharex=True, sharey=True)\nfor c, ax in enumerate(axes):\n"
         f"    ax.scatter(pts[labels == c, 0], pts[labels == c, 1])\n    ax.set_title(class_names[c])",
         {"post": f"{SPANS}, [fc.r(np.asarray(a.collections[0].get_offsets()), 4) for a in fc.axes()], {TITLES}, {_share('x')}, {_share('y')}",
          "require": ["sharex", "sharey", "scatter"]}),
        ("plt.subplots", f"Put {k} class panels on a grid with **{cols} columns** (as many rows as needed: "
                         f"`-(-{k} // {cols})`), scatter class c's points on `axes.flat[c]`, and **remove** the panels "
                         f"left empty (`ax.remove()`). Return the number of Axes left.",
         f"rows = -(-{k} // {cols})\nfig, axes = plt.subplots(rows, {cols}, squeeze=False)\nfor c, ax in enumerate(axes.flat):\n"
         f"    if c < {k}:\n        ax.scatter(pts[labels == c, 0], pts[labels == c, 1])\n    else:\n        ax.remove()\n"
         f"result = len(fig.axes)", {"post": "_, fc.spans(), [fc.r(np.asarray(a.collections[0].get_offsets()), 4) for a in fc.axes()]",
                                     "require": ["remove"]}),
        ("hist", f"Stack {k} panels vertically sharing x (`plt.subplots({k}, 1, sharex=True)`): in panel c draw a "
                 f"histogram of the **x coordinates** of class c with `bins=np.linspace(-6, 6, 25)`, and give it the y "
                 f"label of the class name.",
         f"fig, axes = plt.subplots({k}, 1, sharex=True)\nfor c, ax in enumerate(axes):\n"
         f"    ax.hist(pts[labels == c, 0], bins=np.linspace(-6, 6, 25))\n    ax.set_ylabel(class_names[c])",
         {"post": "[fc.heights(a) for a in fc.axes()], [a.get_ylabel() for a in fc.axes()], " + _share('x')}),
        ("suptitle", f"Make a 1 × {k} row of class panels (scatter each class), and give the figure the title "
                     f"`'{{n}} points, {k} classes'` with n the total number of points (an f-string), and one "
                     f"figure-level x label 'feature 0' and y label 'feature 1'.",
         f"fig, axes = plt.subplots(1, {k})\nfor c, ax in enumerate(axes):\n    ax.scatter(*pts[labels == c].T)\n"
         f"fig.suptitle(f'{{len(pts)}} points, {k} classes')\nfig.supxlabel('feature 0')\nfig.supylabel('feature 1')",
         {"post": "fc.suptitles(), [fc.r(np.asarray(a.collections[0].get_offsets()), 4) for a in fc.axes()]",
          "require": ["suptitle", "supxlabel", "supylabel"]}),
        ("indicate_inset_zoom", "Scatter all points on `fig, ax`, then zoom on class 0: add an inset in the upper-right "
                                "corner (`ax.inset_axes([0.6, 0.6, 0.38, 0.38])`), scatter class 0 in it, set its limits "
                                "to class 0's bounding box (min/max of its coordinates) and connect it to the main plot "
                                "with `ax.indicate_inset_zoom(ins)`.",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1])\nins = ax.inset_axes([0.6, 0.6, 0.38, 0.38])\n"
         "p = pts[labels == 0]\nins.scatter(p[:, 0], p[:, 1])\nins.set_xlim(p[:, 0].min(), p[:, 0].max())\n"
         "ins.set_ylim(p[:, 1].min(), p[:, 1].max())\nax.indicate_inset_zoom(ins)",
         {"post": f"{CHILDREN}, fc.limits(fc.ax(0).child_axes[0]), fc.r(np.asarray(fc.ax(0).child_axes[0].collections[0].get_offsets()), 4), "
                  f"fc.r(fc.ax(0).artists[0].rectangle.get_bbox().bounds, 4)",
          "require": ["inset_axes", "indicate_inset_zoom", "set_xlim", "set_ylim"]}),
        ("subplot_mosaic", f"Overview plus details: a mosaic whose top row is one wide panel 'all' and whose bottom row "
                           f"has one panel per class (named by the class names). Scatter every point on 'all' and each "
                           f"class on its own panel. Build the mosaic as a list of rows: "
                           f"`[['all'] * {k}, class_names]`.",
         f"fig, axd = plt.subplot_mosaic([['all'] * {k}, class_names])\naxd['all'].scatter(pts[:, 0], pts[:, 1])\n"
         f"for c, name in enumerate(class_names):\n    axd[name].scatter(pts[labels == c, 0], pts[labels == c, 1])",
         {"post": "[(s, fc.r(np.asarray(a.collections[0].get_offsets()), 4)) for s, a in sorted(zip(fc.spans(), fc.axes()), key=lambda t: t[0])]"}),
        ("sharex", f"A 2 × {k} grid with **one column per class** and `sharex='col'`: the top row scatters the class' "
                   f"points, the bottom row shows a histogram (10 bins) of its first coordinate, so each histogram lines "
                   f"up with the scatter above it. Title the top panels with the class names.",
         f"fig, axes = plt.subplots(2, {k}, sharex='col')\nfor c in range({k}):\n    p = pts[labels == c]\n"
         f"    axes[0, c].scatter(p[:, 0], p[:, 1])\n    axes[1, c].hist(p[:, 0], bins=10)\n    axes[0, c].set_title(class_names[c])",
         {"post": f"{SPANS}, {TITLES}, {_share('x')}, [fc.heights(a) for a in fc.axes()]", "require": ["sharex", "hist"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def marginal_layouts(rng):
    s = seeded(rng)
    n = rng.randint(80, 200)
    rho = rng.choice([0.3, 0.6, 0.8, -0.5])
    a = rng.choice([3, 4, 5])
    bins = rng.choice([10, 15, 20])
    setup = (f"g = np.random.default_rng({s})\nxy = np.round(g.multivariate_normal([0, 0], [[1, {rho}], [{rho}, 1]], {n}), 3)\n"
             f"x, y = xy[:, 0], xy[:, 1]")
    hist_post = "[(len(a.collections), len(a.patches)) for a in fc.axes()]"
    items = [
        ("subplot_mosaic", f"Drill: a big main plot with a narrow histogram on its right. Build "
                           f"`plt.subplot_mosaic([['main', 'side']], width_ratios=[{a}, 1], sharey=True)`, scatter "
                           f"`y` against `x` on 'main' and draw a **horizontal** histogram of `y` with {bins} bins on "
                           f"'side' (`orientation='horizontal'`).",
         f"fig, axd = plt.subplot_mosaic([['main', 'side']], width_ratios=[{a}, 1], sharey=True)\n"
         f"axd['main'].scatter(x, y)\naxd['side'].hist(y, bins={bins}, orientation='horizontal')",
         {"post": f"fc.spans(), {hist_post}, fc.widths(fc.ax(1)), {_share('y')}, "
                  "list(fc.ax(0).get_subplotspec().get_gridspec().get_width_ratios())",
          "require": ["subplot_mosaic", "orientation"]}),
        ("add_gridspec", f"The classic scatter-with-marginals figure, on a gridspec: `gs = fig.add_gridspec(2, 2, "
                         f"width_ratios=({a}, 1), height_ratios=(1, {a}))`; `ax = fig.add_subplot(gs[1, 0])` gets the "
                         f"scatter, `top = fig.add_subplot(gs[0, 0], sharex=ax)` a histogram of `x` and "
                         f"`right = fig.add_subplot(gs[1, 1], sharey=ax)` a horizontal histogram of `y` ({bins} bins "
                         f"each).",
         f"fig = plt.figure()\ngs = fig.add_gridspec(2, 2, width_ratios=({a}, 1), height_ratios=(1, {a}))\n"
         f"ax = fig.add_subplot(gs[1, 0])\ntop = fig.add_subplot(gs[0, 0], sharex=ax)\n"
         f"right = fig.add_subplot(gs[1, 1], sharey=ax)\nax.scatter(x, y)\ntop.hist(x, bins={bins})\n"
         f"right.hist(y, bins={bins}, orientation='horizontal')",
         {"post": f"fc.spans(), {hist_post}, fc.heights(fc.ax(1)), fc.widths(fc.ax(2)), {_share('x')}, {_share('y')}"}),
        ("inset_axes", f"Marginals without a grid: scatter `y` against `x` on `fig, ax`, then attach a histogram of `x` "
                       f"**above** the plot with `top = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)` and one of `y` "
                       f"to its **right** with `right = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)` (horizontal, "
                       f"{bins} bins each).",
         f"fig, ax = plt.subplots()\nax.scatter(x, y)\ntop = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)\n"
         f"right = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)\ntop.hist(x, bins={bins})\n"
         f"right.hist(y, bins={bins}, orientation='horizontal')",
         {"post": f"{CHILDREN}, [fc.heights(c) for c in fc.ax(0).child_axes], "
                  "[fc.ax(0).get_shared_x_axes().joined(fc.ax(0), c) for c in fc.ax(0).child_axes], "
                  "[fc.ax(0).get_shared_y_axes().joined(fc.ax(0), c) for c in fc.ax(0).child_axes]"}),
        ("subplot_mosaic", f"A mosaic with an empty corner: `[['top', '.'], ['main', 'right']]` with "
                           f"`width_ratios=[{a}, 1]` and `height_ratios=[1, {a}]`. Scatter on 'main', histogram of `x` "
                           f"on 'top', horizontal histogram of `y` on 'right' ({bins} bins), then make 'top' share x "
                           f"with 'main' (`axd['top'].sharex(axd['main'])`) and 'right' share y with it.",
         f"fig, axd = plt.subplot_mosaic([['top', '.'], ['main', 'right']], width_ratios=[{a}, 1], height_ratios=[1, {a}])\n"
         f"axd['main'].scatter(x, y)\naxd['top'].hist(x, bins={bins})\naxd['right'].hist(y, bins={bins}, orientation='horizontal')\n"
         f"axd['top'].sharex(axd['main'])\naxd['right'].sharey(axd['main'])",
         {"post": "_d = {a.get_label(): a for a in fc.axes()}\n"
                  "(sorted((a.get_label(), s) for a, s in zip(fc.axes(), fc.spans())), fc.heights(_d['top']), "
                  "fc.widths(_d['right']), _d['main'].get_shared_x_axes().joined(_d['main'], _d['top']), "
                  "_d['main'].get_shared_y_axes().joined(_d['main'], _d['right']))", "require": ["subplot_mosaic", "sharex", "sharey"]}),
        ("hist2d", f"Side by side, same axes ranges: scatter `y` against `x` on the left and a 2-D histogram "
                   f"(`hist2d(x, y, bins={bins})`) on the right of `plt.subplots(1, 2, sharex=True, sharey=True, "
                   f"figsize=(8, 4))`; titles 'points' and 'density'.",
         f"fig, (a, b) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))\na.scatter(x, y)\nb.hist2d(x, y, bins={bins})\n"
         f"a.set_title('points')\nb.set_title('density')",
         {"post": f"fc.size(), {TITLES}, fc.r(np.asarray(fc.ax(0).collections[0].get_offsets()), 4), [type(c).__name__ for c in fc.ax(1).collections], "
                  f"{_share('x')}, {_share('y')}", "require": ["sharex", "sharey", "hist2d"]}),
        ("colorbar", f"Give a density plot its own colorbar panel: `plt.subplot_mosaic([['main', 'cbar']], "
                     f"width_ratios=[20, 1])`, draw `hb = axd['main'].hexbin(x, y, gridsize={bins})` and put its colorbar "
                     f"**into** the 'cbar' panel with `fig.colorbar(hb, cax=axd['cbar'], label='count')`.",
         f"fig, axd = plt.subplot_mosaic([['main', 'cbar']], width_ratios=[20, 1])\nhb = axd['main'].hexbin(x, y, gridsize={bins})\n"
         f"fig.colorbar(hb, cax=axd['cbar'], label='count')",
         {"post": "fc.spans(), fc.colorbars(), [type(c).__name__ for c in fc.ax(0).collections], "
                  "fc.r(fc.ax(0).collections[0].get_array().sum())", "require": ["subplot_mosaic", "hexbin", "colorbar", "cax"]}),
    ]
    return catalog(rng, items, setup, f"`x` and `y` hold {n} correlated measurements (correlation ≈ {rho}).")


@template(M, "medium")
def zoom_insets(rng):
    code, desc, n = curves(rng)
    e0 = n - rng.randint(5, 10)
    loc = rng.choice([("upper right", [0.55, 0.55, 0.4, 0.4]), ("centre right", [0.55, 0.3, 0.4, 0.4]),
                      ("upper middle", [0.3, 0.55, 0.4, 0.4])])
    items = [
        ("indicate_inset_zoom", f"The interesting part of the loss curve is the end. Plot `val_loss` against `epochs` on "
                                f"`fig, ax`; add an inset in the {loc[0]} (`ax.inset_axes({loc[1]})`) showing the same "
                                f"curve limited to epochs {e0} … {n} (x) and to the min … max of `val_loss` over those "
                                f"epochs (y); then draw the zoom box and connectors with `ax.indicate_inset_zoom(ins)`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nins = ax.inset_axes({loc[1]})\nins.plot(epochs, val_loss)\n"
         f"tail = val_loss[epochs >= {e0}]\nins.set_xlim({e0}, {n})\nins.set_ylim(tail.min(), tail.max())\n"
         f"ax.indicate_inset_zoom(ins)",
         {"post": f"{CHILDREN}, fc.limits(fc.ax(0).child_axes[0]), fc.lines(fc.ax(0).child_axes[0]), "
                  f"fc.r(fc.ax(0).artists[0].rectangle.get_bbox().bounds, 4)",
          "require": ["inset_axes", "indicate_inset_zoom", "set_xlim", "set_ylim"]}),
        ("inset_axes", f"Plot both `train_loss` and `val_loss` on `fig, ax`, and show their **gap** "
                       f"(`val_loss - train_loss`) in a small inset in the {loc[0]} (`ax.inset_axes({loc[1]})`) titled "
                       f"'gap'.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nins = ax.inset_axes({loc[1]})\n"
         f"ins.plot(epochs, val_loss - train_loss)\nins.set_title('gap')",
         {"post": f"{CHILDREN}, fc.lines(fc.ax(0)), fc.lines(fc.ax(0).child_axes[0]), fc.ax(0).child_axes[0].get_title()"}),
        ("indicate_inset", f"Mark a region without an inset: plot `train_loss` against `epochs` and outline the box "
                           f"from epoch {e0} to {n} and loss 0 to 0.5 with `ax.indicate_inset([x0, y0, width, height])`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.indicate_inset([{e0}, 0, {n - e0}, 0.5])",
         {"post": "fc.lines(fc.ax(0)), [fc.r(a.rectangle.get_bbox().bounds, 4) for a in fc.ax(0).artists]"}),
        ("inset_axes", "Two insets side by side **below** the curves' plateau: plot `val_loss` on `fig, ax`; add "
                       "`left = ax.inset_axes([0.1, 0.5, 0.35, 0.35])` showing the first 5 epochs of `val_loss` and "
                       "`right = ax.inset_axes([0.55, 0.5, 0.35, 0.35])` showing the last 5.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nleft = ax.inset_axes([0.1, 0.5, 0.35, 0.35])\n"
         "left.plot(epochs[:5], val_loss[:5])\nright = ax.inset_axes([0.55, 0.5, 0.35, 0.35])\n"
         "right.plot(epochs[-5:], val_loss[-5:])",
         {"post": f"{CHILDREN}, [fc.lines(c) for c in fc.ax(0).child_axes]"}),
        ("indicate_inset_zoom", f"Zoom on the best epoch. Plot `val_loss` against `epochs`; add an inset "
                                f"(`ax.inset_axes({loc[1]})`) with the same curve, limited to the best epoch ± 2 in x "
                                f"(best = lowest `val_loss`) and to `best_loss - 0.05 … best_loss + 0.05` in y; mark the "
                                f"zoom with `ax.indicate_inset_zoom(ins, edgecolor='red')`.",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nb = val_loss.argmin()\nins = ax.inset_axes({loc[1]})\n"
         f"ins.plot(epochs, val_loss)\nins.set_xlim(epochs[b] - 2, epochs[b] + 2)\n"
         f"ins.set_ylim(val_loss[b] - 0.05, val_loss[b] + 0.05)\nax.indicate_inset_zoom(ins, edgecolor='red')",
         {"post": "fc.limits(fc.ax(0).child_axes[0]), fc.r(fc.ax(0).artists[0].rectangle.get_bbox().bounds, 4), "
                  "fc.hexcolor(fc.ax(0).artists[0].rectangle.get_edgecolor())",
          "require": ["inset_axes", "indicate_inset_zoom", "edgecolor"]}),
        ("inset_axes", f"Place an inset in **data** coordinates instead of Axes fractions: plot `val_loss` against "
                       f"`epochs` and add `ins = ax.inset_axes([{n // 2}, 1.2, {n // 2 - 1}, 0.6], transform=ax.transData)` "
                       f"(its corner at epoch {n // 2}, loss 1.2), showing `val_loss - train_loss`, with its ticks "
                       f"removed (`ins.set_xticks([])`, `ins.set_yticks([])`).",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nins = ax.inset_axes([{n // 2}, 1.2, {n // 2 - 1}, 0.6], transform=ax.transData)\n"
         f"ins.plot(epochs, val_loss - train_loss)\nins.set_xticks([])\nins.set_yticks([])",
         {"post": f"{CHILDREN}, fc.ax(0).child_axes[0].get_axes_locator()._transform is fc.ax(0).transData, "
                  f"fc.lines(fc.ax(0).child_axes[0]), [list(fc.ax(0).child_axes[0].get_xticks()), list(fc.ax(0).child_axes[0].get_yticks())]",
          "require": ["inset_axes", "transData", "set_xticks", "set_yticks"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def twin_scales(rng):
    code, desc, n = curves(rng, acc=True)
    lr_code, _ = _lr02(rng)
    spe = rng.choice([100, 200, 500])
    items = [
        ("twinx", "Drill: loss and learning rate on twin y-axes. Plot `train_loss` against `epochs` in 'C0' on `ax` "
                  "(y label 'loss'), then `lr` in 'C1' on `ax.twinx()` with a **log** y-scale (y label 'lr'). Return "
                  "the two y scales, left then right.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, color='C0')\nax.set_ylabel('loss')\nax2 = ax.twinx()\n"
         "ax2.plot(epochs, lr, color='C1')\nax2.set_yscale('log')\nax2.set_ylabel('lr')\n"
         "result = [ax.get_yscale(), ax2.get_yscale()]",
         {"post": f"_, [(fc.lines(a, 'color'), a.get_ylabel()) for a in fc.axes()], {_share('x')}",
          "require": ["twinx", "set_yscale"]}),
        ("legend", "A twin plot needs **one** legend for lines on two Axes. Plot `train_loss` (label 'loss') on `ax` and "
                   "`val_acc` (label 'acc', color 'C1') on `ax2 = ax.twinx()`, then build a single legend on `ax` from "
                   "both handles: `ax.legend(handles=[l1, l2])`.",
         "fig, ax = plt.subplots()\nl1, = ax.plot(epochs, train_loss, label='loss')\nax2 = ax.twinx()\n"
         "l2, = ax2.plot(epochs, val_acc, color='C1', label='acc')\nax.legend(handles=[l1, l2])",
         {"post": "fc.legend(fc.ax(0)), fc.legend(fc.ax(1)), " + YDATA, "require": ["twinx", "handles"]}),
        ("secondary_xaxis", f"One epoch is {spe} steps. Plot `val_loss` against `epochs` (x label 'epoch') and add a top "
                            f"axis in **thousands of steps** (`secondary_xaxis('top', functions=...)`, label 'k steps').",
         f"fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.set_xlabel('epoch')\n"
         f"sec = ax.secondary_xaxis('top', functions=(lambda e: e * {spe} / 1000, lambda k: k * 1000 / {spe}))\n"
         f"sec.set_xlabel('k steps')",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([1.0, 10.0])), 4), "
                  "fc.r(_c._functions[1](np.array([1.0])), 4), _c.get_xlabel(), fc.labels(fc.ax(0))['xlabel']"}),
        ("twinx", "Three quantities, two y-axes: plot `train_loss` and `val_loss` on `ax` (y label 'loss') and `val_acc` "
                  "on `ax.twinx()` with y limits 0 … 1 (y label 'accuracy'). Return how many lines each Axes holds.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nax.set_ylabel('loss')\n"
         "ax2 = ax.twinx()\nax2.plot(epochs, val_acc)\nax2.set_ylim(0, 1)\nax2.set_ylabel('accuracy')\n"
         "result = [len(ax.get_lines()), len(ax2.get_lines())]",
         {"post": "_, [(a.get_ylabel(), fc.limits(a)['ylim']) for a in fc.axes()][1:], " + YDATA}),
        ("tick_params", "Color-code a twin axis: plot `val_loss` in 'C0' on `ax` and `lr` in 'C3' on `ax2 = "
                        "ax.twinx()`; color the right axis' tick labels 'C3' with `ax2.tick_params(axis='y', "
                        "labelcolor='C3')`.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss, color='C0')\nax2 = ax.twinx()\nax2.plot(epochs, lr, color='C3')\n"
         "ax2.tick_params(axis='y', labelcolor='C3')",
         {"post": "[fc.lines(a, 'color') for a in fc.axes()], fc.hexcolor(fc.ax(1).yaxis._major_tick_kw.get('label2.color', "
                  "fc.ax(1).yaxis._major_tick_kw.get('labelcolor')))"}),
        ("secondary_yaxis", "Read the loss as **improvement over epoch 1**: plot `val_loss` against `epochs` and add a "
                            "right-hand axis with `functions=(lambda l: 100 * (1 - l / l0), lambda p: l0 * (1 - p / 100))` "
                            "where `l0 = val_loss[0]`, labelled 'improvement (%)'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nl0 = val_loss[0]\n"
         "sec = ax.secondary_yaxis('right', functions=(lambda l: 100 * (1 - l / l0), lambda p: l0 * (1 - p / 100)))\n"
         "sec.set_ylabel('improvement (%)')",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([val_loss[0], val_loss[-1]])), 4), "
                  "fc.r(_c._functions[1](np.array([50.0])), 4), _c.get_ylabel()"}),
        ("spines", "Make the twin axis obviously belong to its line: plot `train_loss` in 'C0' on `ax`, `val_acc` in 'C1' "
                   "on `ax2 = ax.twinx()`, then color the right spine of `ax2` 'C1' (`ax2.spines['right'].set_color('C1')`) "
                   "and the left spine of `ax2` 'C0'.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss, color='C0')\nax2 = ax.twinx()\nax2.plot(epochs, val_acc, color='C1')\n"
         "ax2.spines['right'].set_color('C1')\nax2.spines['left'].set_color('C0')",
         {"post": "[fc.lines(a, 'color') for a in fc.axes()], [fc.hexcolor(fc.ax(1).spines[k].get_edgecolor()) for k in ('left', 'right')]",
          "require": ["twinx", "spines", "set_color"]}),
    ]
    return catalog(rng, items, f"{code}\n{lr_code}", f"{desc}, and `lr`, each epoch's learning rate.")


@template(M, "medium")
def subfigure_layouts(rng):
    code, desc, n = curves(rng, acc=True)
    a = rng.choice([2, 3])
    t1, t2 = rng.choice([("losses", "accuracy"), ("optimisation", "evaluation"), ("train", "eval")])
    items = [
        ("subfigures", f"Two subfigures side by side with `width_ratios=[{a}, 1]`: the left one ({t1!r}, as its own "
                       f"`suptitle`) holds two stacked panels with `train_loss` and `val_loss`; the right one ({t2!r}) "
                       f"one panel with `val_acc` (all against `epochs`). Use `layout='constrained'` on the figure.",
         f"fig = plt.figure(layout='constrained')\nleft, right = fig.subfigures(1, 2, width_ratios=[{a}, 1])\n"
         f"top, bottom = left.subplots(2, 1)\ntop.plot(epochs, train_loss)\nbottom.plot(epochs, val_loss)\n"
         f"left.suptitle({t1!r})\nax = right.subplots()\nax.plot(epochs, val_acc)\nright.suptitle({t2!r})",
         {"post": "_s = fc.fig().subfigs\n([len(s.axes) for s in _s], [s._suptitle.get_text() for s in _s], "
                  "[[fc.ydata(a) for a in s.axes] for s in _s], list(_s[0]._subplotspec.get_gridspec().get_width_ratios()))",
          "require": ["subfigures", "suptitle"]}),
        ("subfigures", f"Two subfigures stacked vertically (`fig.subfigures(2, 1)`): the top one is a row of 3 panels "
                       f"(`train_loss`, `val_loss`, `val_acc`), the bottom one a single wide panel with all three "
                       f"curves. Give the bottom subfigure the x label 'epoch' with `supxlabel`.",
         "fig = plt.figure()\ntop, bottom = fig.subfigures(2, 1)\nfor ax, y in zip(top.subplots(1, 3), (train_loss, val_loss, val_acc)):\n"
         "    ax.plot(epochs, y)\nax = bottom.subplots()\nfor y in (train_loss, val_loss, val_acc):\n    ax.plot(epochs, y)\n"
         "bottom.supxlabel('epoch')",
         {"post": "_s = fc.fig().subfigs\n([len(s.axes) for s in _s], [[fc.ydata(a) for a in s.axes] for s in _s], "
                  "_s[1]._supxlabel.get_text())", "require": ["subfigures", "supxlabel"]}),
        ("subfigures", f"Subfigures can share axes inside themselves: split the figure into two columns; in the left "
                       f"one make 2 stacked panels that share x (`left.subplots(2, 1, sharex=True)`) with `train_loss` "
                       f"and `val_loss`; in the right one 2 panels that do **not** share, with `val_acc` and "
                       f"`val_loss - train_loss`.",
         "fig = plt.figure()\nleft, right = fig.subfigures(1, 2)\na, b = left.subplots(2, 1, sharex=True)\n"
         "a.plot(epochs, train_loss)\nb.plot(epochs, val_loss)\nc, d = right.subplots(2, 1)\nc.plot(epochs, val_acc)\n"
         "d.plot(epochs, val_loss - train_loss)",
         {"post": f"[len(s.axes) for s in fc.fig().subfigs], {YDATA}, {_share('x')}"}),
        ("suptitle", f"A figure title **and** subfigure titles: split into two side-by-side subfigures titled "
                     f"{t1!r} and {t2!r}, each with one panel (`train_loss` / `val_acc`), and give the whole figure "
                     f"the title 'Run summary'.",
         f"fig = plt.figure()\nleft, right = fig.subfigures(1, 2)\nleft.subplots().plot(epochs, train_loss)\n"
         f"right.subplots().plot(epochs, val_acc)\nleft.suptitle({t1!r})\nright.suptitle({t2!r})\nfig.suptitle('Run summary')",
         {"post": "fc.suptitles()['suptitle'], [s._suptitle.get_text() for s in fc.fig().subfigs], " + YDATA}),
        ("subfigures", "A 2 × 2 grid of **subfigures** (`fig.subfigures(2, 2)`), one per curve — `train_loss`, "
                       "`val_loss`, `val_acc` and `val_loss - train_loss`, row by row — each subfigure with one Axes, a "
                       "light-grey background (`sf.set_facecolor('0.9')`) and its curve's name as suptitle ('train', "
                       "'val', 'acc', 'gap').",
         "fig = plt.figure()\nsfs = fig.subfigures(2, 2)\nfor sf, y, name in zip(sfs.flat, (train_loss, val_loss, val_acc, "
         "val_loss - train_loss), ('train', 'val', 'acc', 'gap')):\n    sf.subplots().plot(epochs, y)\n    sf.set_facecolor('0.9')\n"
         "    sf.suptitle(name)",
         {"post": "_s = fc.fig().subfigs\n([s._suptitle.get_text() for s in _s], [fc.hexcolor(s.get_facecolor()) for s in _s], "
                  "[[fc.ydata(a) for a in s.axes] for s in _s])", "require": ["subfigures", "set_facecolor", "suptitle"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def small_multiples(rng):
    k = rng.randint(4, 7)
    s = seeded(rng)
    lrs = sorted(rng.sample([0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003], k))[::-1]
    cols = rng.choice([2, 3, 4])
    setup = (f"g = np.random.default_rng({s})\nlrs = {lit(lrs)}\nepochs = np.arange(1, 21)\n"
             f"runs = np.round(np.exp(-np.outer(np.log10(lrs) + 4, epochs) / 10) + 0.05 * g.random(({k}, 20)), 4)")
    rows = -(-k // cols)
    items = [
        ("plt.subplots", f"One panel per learning rate: a grid with {cols} columns and as many rows as needed "
                         f"({rows}), sharing both axes. Plot run i against `epochs` on `axes.flat[i]`, title it "
                         f"`f'lr={{lrs[i]}}'`, and remove the unused panels.",
         f"fig, axes = plt.subplots({rows}, {cols}, sharex=True, sharey=True, squeeze=False)\n"
         f"for i, ax in enumerate(axes.flat):\n    if i < len(lrs):\n        ax.plot(epochs, runs[i])\n"
         f"        ax.set_title(f'lr={{lrs[i]}}')\n    else:\n        ax.remove()",
         {"post": f"fc.spans(), {YDATA}, {TITLES}", "require": ["sharex", "sharey", "remove"]}),
        ("set_axis_off", f"Same grid ({rows} × {cols}, one run per panel, in order), but **hide** the unused panels "
                         f"with `ax.set_axis_off()` instead of removing them. Return the number of Axes that are "
                         f"still visible (`ax.axison`).",
         f"fig, axes = plt.subplots({rows}, {cols})\nfor i, ax in enumerate(axes.flat):\n    if i < len(lrs):\n"
         f"        ax.plot(epochs, runs[i])\n    else:\n        ax.set_axis_off()\nresult = sum(ax.axison for ax in axes.flat)",
         {"post": "_, [a.axison for a in fc.axes()], " + YDATA}),
        ("suptitle", f"All runs in a single row (`plt.subplots(1, {k}, sharey=True, figsize=({2 * k}, 2.5))`), panel i "
                     f"titled with its learning rate as `f'{{lrs[i]:g}}'`, plus the figure title 'learning-rate sweep' "
                     f"and the figure-level labels 'epoch' (x) and 'loss' (y).",
         f"fig, axes = plt.subplots(1, {k}, sharey=True, figsize=({2 * k}, 2.5))\nfor ax, lr, run in zip(axes, lrs, runs):\n"
         f"    ax.plot(epochs, run)\n    ax.set_title(f'{{lr:g}}')\nfig.suptitle('learning-rate sweep')\n"
         f"fig.supxlabel('epoch')\nfig.supylabel('loss')",
         {"post": f"fc.size(), {YDATA}, {TITLES}, fc.suptitles(), {_share('y')}"}),
        ("label_outer", f"A {rows} × {cols} grid (sharing nothing) with run i on panel i, x label 'epoch' and y label "
                        f"'loss' on every used panel, then `ax.label_outer()` on all of them. Remove unused panels "
                        f"**first**.",
         f"fig, axes = plt.subplots({rows}, {cols})\nfor ax in axes.flat[len(lrs):]:\n    ax.remove()\n"
         f"for ax, run in zip(axes.flat, runs):\n    ax.plot(epochs, run)\n    ax.set(xlabel='epoch', ylabel='loss')\n"
         f"    ax.label_outer()", {"post": f"{YDATA}, {TICKLABELS}"}),
        ("figsize", f"Size the figure to the grid: {rows} × {cols} panels of 3 × 2 inches each "
                    f"(`figsize=(3 * cols, 2 * rows)`), constrained layout, runs in order, unused panels removed. Return "
                    f"`fig.get_size_inches().tolist()`.",
         f"fig, axes = plt.subplots({rows}, {cols}, figsize=(3 * {cols}, 2 * {rows}), layout='constrained', squeeze=False)\n"
         f"for i, ax in enumerate(axes.flat):\n    if i < len(runs):\n        ax.plot(epochs, runs[i])\n    else:\n"
         f"        ax.remove()\nresult = fig.get_size_inches().tolist()", {"post": f"_, {ENGINE}, {YDATA}"}),
    ]
    return catalog(rng, items, setup, f"`runs` holds {k} loss curves (one row per learning rate in `lrs`, 20 epochs "
                                      f"each, `epochs` = 1 … 20).")


# ================================================================= hard

@template(M, "hard")
def layout_functions(rng):
    code, desc, n = curves(rng, acc=True)
    cols = rng.choice([2, 3])
    color = rng.choice(["C1", "tab:red", "black"])
    items = [
        ("remove", "Write `plot_grid(curves, ncols=3, titles=None)`: one panel per curve on a grid with `ncols` columns "
                   "and just enough rows, each curve plotted against its index, titled from `titles` when given; the "
                   "unused panels are **removed**; return `(fig, used_axes)` (a list). (It's called with 5 and with 2 "
                   "curves.)",
         "def plot_grid(curves, ncols=3, titles=None):\n    nrows = -(-len(curves) // ncols)\n"
         "    fig, axes = plt.subplots(nrows, ncols, squeeze=False)\n    used = []\n"
         "    for i, ax in enumerate(axes.flat):\n        if i >= len(curves):\n            ax.remove()\n            continue\n"
         "        ax.plot(curves[i])\n        if titles:\n            ax.set_title(titles[i])\n        used.append(ax)\n"
         "    return fig, used",
         {"post": f"_f, _u = plot_grid([train_loss, val_loss, val_acc, train_loss * 2, val_loss * 2], ncols={cols}, "
                  f"titles=list('abcde'))\n_g, _v = plot_grid([train_loss, val_acc])\n"
                  "(fc.spans(_f), [fc.ydata(a) for a in _u], [a.get_title() for a in _u], fc.spans(_g), len(_v))"}),
        ("twinx", f"Write `twin_plot(x, y1, y2, ax=None, colors=('C0', {color!r}))` that draws `y1` on `ax` (the current "
                  f"Axes when None, via `plt.gca()`) and `y2` on a new twin y-axis, each in its color, colors each y "
                  f"axis' tick labels to match (`tick_params(axis='y', labelcolor=...)`) and returns `(ax, ax2)`.",
         "def twin_plot(x, y1, y2, ax=None, colors=('C0', " + repr(color) + ")):\n    if ax is None:\n        ax = plt.gca()\n"
         "    ax.plot(x, y1, color=colors[0])\n    ax2 = ax.twinx()\n    ax2.plot(x, y2, color=colors[1])\n"
         "    ax.tick_params(axis='y', labelcolor=colors[0])\n    ax2.tick_params(axis='y', labelcolor=colors[1])\n"
         "    return ax, ax2",
         {"post": "_f, (_a, _b) = plt.subplots(1, 2)\n_r = twin_plot(epochs, train_loss, val_acc, ax=_b)\nplt.sca(_a)\n"
                  "_s = twin_plot(epochs, val_loss, train_loss, colors=('k', 'C2'))\n"
                  "([fc.lines(a, 'color') for a in (_r + _s)], _r[0] is _b, _s[0] is _a, len(_f.axes), "
                  "[fc.hexcolor(a.yaxis._major_tick_kw.get('label1.color', a.yaxis._major_tick_kw.get('labelcolor', 'w'))) "
                  "for a in (_r[0], _s[0])])", "require": ["twinx", "gca"]}),
        ("indicate_inset_zoom", "Write `zoom(ax, xlim, ylim, bounds=(0.55, 0.55, 0.4, 0.4))`: add an inset at `bounds`, "
                                "copy every line of `ax` into it (`ins.plot(*line.get_data())`), set the inset's limits, "
                                "mark the zoom with `ax.indicate_inset_zoom(ins)` and return the inset.",
         "def zoom(ax, xlim, ylim, bounds=(0.55, 0.55, 0.4, 0.4)):\n    ins = ax.inset_axes(bounds)\n"
         "    for line in ax.get_lines():\n        ins.plot(*line.get_data())\n    ins.set_xlim(*xlim)\n    ins.set_ylim(*ylim)\n"
         "    ax.indicate_inset_zoom(ins)\n    return ins",
         {"post": "_f, _a = plt.subplots()\n_a.plot(epochs, train_loss)\n_a.plot(epochs, val_loss)\n"
                  "_i = zoom(_a, (epochs[-6], epochs[-1]), (0, 0.6), bounds=(0.5, 0.5, 0.45, 0.45))\n"
                  "(fc.lines(_i), fc.limits(_i), fc.r(_i.get_axes_locator()._bounds, 3), len(_a.artists))",
          "require": ["inset_axes", "indicate_inset_zoom"]}),
        ("sharex", "Write `history_figure(history)`: `history` maps metric names to lists of per-epoch values. Make one "
                   "panel per metric, stacked vertically and sharing x (in the dict's order), plot each metric against "
                   "epochs 1, 2, … with the metric name as the panel's y label, put 'epoch' as x label on the bottom "
                   "panel only, and return the figure. With one metric there's a single Axes (`squeeze=False` helps).",
         "def history_figure(history):\n    fig, axes = plt.subplots(len(history), 1, sharex=True, squeeze=False)\n"
         "    for ax, (name, vals) in zip(axes[:, 0], history.items()):\n        ax.plot(range(1, len(vals) + 1), vals)\n"
         "        ax.set_ylabel(name)\n    axes[-1, 0].set_xlabel('epoch')\n    return fig",
         {"post": "_f = history_figure({'loss': list(train_loss), 'val': list(val_loss), 'acc': list(val_acc)})\n"
                  "_g = history_figure({'acc': [0.5, 0.7, 0.8]})\n"
                  "([fc.lines(a) for a in _f.axes], [fc.labels(a) for a in _f.axes], "
                  "_f.axes[0].get_shared_x_axes().joined(_f.axes[0], _f.axes[2]), [fc.labels(a) for a in _g.axes])"}),
        ("set_xlabel", "Write `label_grid(axes, xlabel, ylabel)` for any 2-D array of Axes: x label on the bottom row "
                       "only, y label on the left column only, and every other Axes left without labels. It returns "
                       "nothing. (Checked on a 2 × 3 and a 3 × 1 grid.)",
         "def label_grid(axes, xlabel, ylabel):\n    for ax in axes[-1, :]:\n        ax.set_xlabel(xlabel)\n"
         "    for ax in axes[:, 0]:\n        ax.set_ylabel(ylabel)",
         {"post": "_f, _a = plt.subplots(2, 3)\nlabel_grid(_a, 'x', 'y')\n_g, _b = plt.subplots(3, 1, squeeze=False)\n"
                  "label_grid(_b, 'epoch', 'loss')\n([fc.labels(a) for a in _f.axes], [fc.labels(a) for a in _g.axes])",
          "require": ["set_xlabel", "set_ylabel"]}),
        ("subplot_mosaic", "Write `panel_figure(spec)`: `spec` maps panel names to 1-D arrays; build a mosaic with "
                           "**one row per panel**, all panels in a single column (`[[name] for name in spec]`), sharing "
                           "x; plot each array on its panel, title it with its name, and return the `axd` dict.",
         "def panel_figure(spec):\n    fig, axd = plt.subplot_mosaic([[name] for name in spec], sharex=True)\n"
         "    for name, ys in spec.items():\n        axd[name].plot(ys)\n        axd[name].set_title(name)\n    return axd",
         {"post": "_d = panel_figure({'train': train_loss, 'val': val_loss})\n"
                  "(sorted(_d), [fc.ydata(_d[k]) for k in ('train', 'val')], [_d[k].get_title() for k in ('train', 'val')], "
                  "_d['train'].get_shared_x_axes().joined(_d['train'], _d['val']), fc.grid_shape(_d['train'].figure))"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def debug_layouts(rng):
    code, desc, n = curves(rng, acc=True)
    lr_code, _ = _lr02(rng)
    title = rng.choice(["Run 3", "Training", "Summary"])
    items = [
        (".plot", "This crashes with *'numpy.ndarray' object has no attribute 'plot'*:\n```python\n"
                  "fig, axes = plt.subplots(2, 2)\naxes[0].plot(epochs, train_loss)\naxes[1].plot(epochs, val_loss)\n```\n"
                  "On a 2 × 2 grid, `axes[0]` is a whole **row**. Fix it so `train_loss` lands top-left and `val_loss` "
                  "top-right.",
         "fig, axes = plt.subplots(2, 2)\naxes[0, 0].plot(epochs, train_loss)\naxes[0, 1].plot(epochs, val_loss)",
         {"post": f"{SPANS}, {YDATA}"}),
        ("plt.subplots", "Unpacking a 2 × 2 grid into two names gives the two **rows**, so `.plot` then fails with "
                         "*'numpy.ndarray' object has no attribute 'plot'*:\n```python\nfig, (ax1, ax2) = plt.subplots(2, 2)\n"
                         "ax1.plot(epochs, train_loss)\n```\nUnpack all four Axes "
                         "(`fig, ((ax1, ax2), (ax3, ax4)) = ...`) and plot `train_loss`, `val_loss`, `val_acc`, `lr` on "
                         "them in that order.",
         "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)\nax1.plot(epochs, train_loss)\nax2.plot(epochs, val_loss)\n"
         "ax3.plot(epochs, val_acc)\nax4.plot(epochs, lr)", {"post": f"{SPANS}, {YDATA}"}),
        ("subplot_mosaic", "This raises *ValueError: ... must form a rectangle*:\n```python\n"
                           "fig, axd = plt.subplot_mosaic('AB;BA')\n```\nThe intended layout was a wide panel 'A' on top "
                           "and two panels 'B' (left) and 'C' (right) below it. Fix the mosaic and plot `train_loss` on "
                           "'A'.",
         "fig, axd = plt.subplot_mosaic('AA;BC')\naxd['A'].plot(epochs, train_loss)",
         {"post": "sorted((s, len(a.get_lines())) for s, a in zip(fc.spans(), fc.axes())), fc.grid_shape()"}),
        ("twinx", "The learning rate ends up squashed on the loss axis: the second line goes to the wrong Axes.\n"
                  "```python\nfig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax2 = ax.twinx()\nax.plot(epochs, lr)\n"
                  "ax2.set_ylabel('lr')\n```\nFix it.",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax2 = ax.twinx()\nax2.plot(epochs, lr)\nax2.set_ylabel('lr')",
         {"post": f"{YDATA}, [a.get_ylabel() for a in fc.axes()]"}),
        ("suptitle", f"This crashes with *'Axes' object has no attribute 'suptitle'*:\n```python\n"
                     f"fig, axes = plt.subplots(1, 2)\naxes[0].plot(epochs, train_loss)\naxes[1].plot(epochs, val_acc)\n"
                     f"axes[0].suptitle({title!r})\n```\nA figure-level title belongs to the figure. Fix it.",
         f"fig, axes = plt.subplots(1, 2)\naxes[0].plot(epochs, train_loss)\naxes[1].plot(epochs, val_acc)\nfig.suptitle({title!r})",
         {"post": f"fc.suptitles()['suptitle'], {YDATA}, {TITLES}"}),
        ("add_subplot", "This raises *num must be an integer with 1 <= num <= 4*:\n```python\nfig = plt.figure()\n"
                        "for k in range(4):\n    ax = fig.add_subplot(2, 2, k)\n    ax.plot(epochs, train_loss * (k + 1))\n"
                        "```\nSubplot positions count from 1. Fix it.",
         "fig = plt.figure()\nfor k in range(4):\n    ax = fig.add_subplot(2, 2, k + 1)\n    ax.plot(epochs, train_loss * (k + 1))",
         {"post": f"{SPANS}, {YDATA}"}),
        ("sharey", "The accuracy panel is flattened into a line at the bottom because all three panels share y:\n"
                   "```python\nfig, (a, b, c) = plt.subplots(1, 3, sharey=True)\na.plot(epochs, train_loss)\n"
                   "b.plot(epochs, val_loss)\nc.plot(epochs, val_acc * 100)\n```\nKeep y shared between the two "
                   "**loss** panels only (`plt.subplots(1, 3)` then `b.sharey(a)`).",
         "fig, (a, b, c) = plt.subplots(1, 3)\nb.sharey(a)\na.plot(epochs, train_loss)\nb.plot(epochs, val_loss)\n"
         "c.plot(epochs, val_acc * 100)", {"post": f"{YDATA}, {_share('y')}"}),
        ("set_xlim", "The inset is supposed to zoom on the last 5 epochs, but it shows the whole curve:\n```python\n"
                     "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])\n"
                     "ins.plot(epochs, val_loss)\nax.indicate_inset_zoom(ins)\n```\nLimit the inset to x = `epochs[-5]` … "
                     "`epochs[-1]` and y = min … max of the last 5 losses (before indicating the zoom).",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nins = ax.inset_axes([0.5, 0.5, 0.45, 0.45])\n"
         "ins.plot(epochs, val_loss)\nins.set_xlim(epochs[-5], epochs[-1])\nins.set_ylim(val_loss[-5:].min(), val_loss[-5:].max())\n"
         "ax.indicate_inset_zoom(ins)",
         {"post": "fc.limits(fc.ax(0).child_axes[0]), fc.r(fc.ax(0).artists[0].rectangle.get_bbox().bounds, 4)",
          "require": ["set_xlim", "set_ylim"]}),
        ("subplots_adjust", "`subplots_adjust` is ignored (with a warning) when a layout engine is active:\n```python\n"
                            "fig, axes = plt.subplots(1, 2, layout='constrained')\nfig.subplots_adjust(wspace=0.5)\n"
                            "axes[0].plot(epochs, train_loss)\n```\nDrop the layout engine so the manual spacing "
                            "applies.",
         "fig, axes = plt.subplots(1, 2)\nfig.subplots_adjust(wspace=0.5)\naxes[0].plot(epochs, train_loss)",
         {"post": f"{SUBPARS}, {ENGINE}, {YDATA}"}),
        ("secondary_xaxis", "The top axis should show steps (100 per epoch), but the code passes only one function and "
                            "crashes:\n```python\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)\n"
                            "ax.secondary_xaxis('top', functions=lambda e: e * 100)\n```\nGive it the forward **and** "
                            "inverse functions.",
         "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.secondary_xaxis('top', functions=(lambda e: e * 100, lambda s: s / 100))",
         {"post": "_c = fc.ax(0).child_axes[0]\nfc.r(_c._functions[0](np.array([2.0])), 4), fc.r(_c._functions[1](np.array([300.0])), 4)"}),
    ]
    return catalog(rng, items, f"{code}\n{lr_code}", f"{desc}, and `lr`, each epoch's learning rate.")


@template(M, "hard")
def layout_drills(rng):
    code, desc, n = curves(rng, acc=True)
    lr_code, _ = _lr02(rng)
    title = rng.choice(["Training overview", "Run 42", "Baseline vs. tuned"])
    s = seeded(rng)
    a = rng.choice([3, 4])
    mos = rng.choice(["AAB;CCB", "ABB;ACC", "AB;CC"])
    sp = _mosaic_spans(mos)
    content = {"A": "train_loss", "B": "val_acc", "C": "lr"}
    items = [
        ("sharex", f"Drill 1: a 2 × 2 grid with **shared x-axes** and a figure-level title {title!r}. Plot "
                   f"`train_loss`, `val_loss`, `val_acc` and `lr` (in that order, row by row) against `epochs`, "
                   f"each panel titled with the variable's name.",
         f"fig, axes = plt.subplots(2, 2, sharex=True)\nfor ax, y, name in zip(axes.flat, (train_loss, val_loss, val_acc, lr), "
         f"('train_loss', 'val_loss', 'val_acc', 'lr')):\n    ax.plot(epochs, y)\n    ax.set_title(name)\nfig.suptitle({title!r})",
         {"post": f"{SPANS}, {YDATA}, {TITLES}, {_share('x')}, fc.suptitles()['suptitle']",
          "require": ["sharex", "suptitle"]}),
        ("subplot_mosaic", f"Drill 2: a big main plot with a narrow histogram on the right, via `subplot_mosaic`: "
                           f"`[['main', 'hist']]` with `width_ratios=[{a}, 1]` and a shared y-axis. 'main' plots "
                           f"`errors` against its index; 'hist' shows a horizontal histogram of `errors` (20 bins).",
         f"fig, axd = plt.subplot_mosaic([['main', 'hist']], width_ratios=[{a}, 1], sharey=True)\naxd['main'].plot(errors)\n"
         f"axd['hist'].hist(errors, bins=20, orientation='horizontal')",
         {"post": f"fc.spans(), fc.lines(fc.ax(0)), fc.widths(fc.ax(1)), {_share('y')}, "
                  f"list(fc.ax(0).get_subplotspec().get_gridspec().get_width_ratios())",
          "setup": f"errors = np.round(np.random.default_rng({s}).normal(0, 1, 150).cumsum() / 5, 4)",
          "require": ["subplot_mosaic", "hist"]}),
        ("twinx", "Drill 3: loss and learning rate on twin y-axes. `train_loss` and `val_loss` (labels 'train', 'val') "
                  "on the left axis with y label 'loss'; `lr` dashed (`'--'`) in 'C2' on the right axis (y label "
                  "'lr'); x label 'epoch'; one legend on the left Axes listing all three lines (label the lr line "
                  "'lr').",
         "fig, ax = plt.subplots()\nl1, = ax.plot(epochs, train_loss, label='train')\nl2, = ax.plot(epochs, val_loss, label='val')\n"
         "ax.set(xlabel='epoch', ylabel='loss')\nax2 = ax.twinx()\nl3, = ax2.plot(epochs, lr, '--', color='C2', label='lr')\n"
         "ax2.set_ylabel('lr')\nax.legend(handles=[l1, l2, l3])",
         {"post": "[(fc.lines(a, 'color', 'ls', 'label'), fc.labels(a)) for a in fc.axes()], fc.legend(fc.ax(0))"}),
        ("subplot_mosaic", f"Build the mosaic `{mos!r}` and fill it: 'A' plots `{content['A']}`, 'B' `{content['B']}`, "
                           f"'C' `{content['C']}` (against `epochs`), each titled with its letter; then give the figure "
                           f"a figure-level x label 'epoch'.",
         f"fig, axd = plt.subplot_mosaic({mos!r})\nfor k, y in (('A', train_loss), ('B', val_acc), ('C', lr)):\n"
         f"    axd[k].plot(epochs, y)\n    axd[k].set_title(k)\nfig.supxlabel('epoch')",
         {"post": "sorted((s, a.get_title(), fc.ydata(a)) for s, a in zip(fc.spans(), fc.axes())), fc.suptitles()['supxlabel']",
          "require": ["subplot_mosaic", "supxlabel"]}),
        ("label_outer", "A 2 × 3 grid of `epochs` against `val_loss * (k + 1)` (k = panel index row by row), sharing "
                        "both axes, every panel with x label 'epoch' and y label 'loss', then `ax.label_outer()` "
                        "everywhere, plus the figure-level title 'scaled losses' and a constrained layout.",
         "fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, layout='constrained')\nfor k, ax in enumerate(axes.flat):\n"
         "    ax.plot(epochs, val_loss * (k + 1))\n    ax.set(xlabel='epoch', ylabel='loss')\n    ax.label_outer()\n"
         "fig.suptitle('scaled losses')",
         {"post": f"{YDATA}, {TICKLABELS}, [(a.xaxis.label.get_visible(), a.yaxis.label.get_visible()) for a in fc.axes()], "
                  f"{ENGINE}, fc.suptitles()['suptitle']"}),
    ]
    return catalog(rng, items, f"{code}\n{lr_code}", f"{desc}, and `lr`, each epoch's learning rate.")


@template(M, "hard")
def layout_specs(rng):
    code, desc, k, names = classes(rng, k=(3, 4))
    s = seeded(rng)
    l, b, w, h = rng.choice([(0.1, 0.1, 0.6, 0.8), (0.12, 0.15, 0.55, 0.7), (0.08, 0.2, 0.65, 0.7)])
    gr = rng.choice([3, 4])
    items = [
        ("add_gridspec", f"Build this figure on a {gr} × {gr} gridspec (`fig.add_gridspec({gr}, {gr}, hspace=0.4, "
                         f"wspace=0.3)`): a scatter of all points spanning rows 1 … {gr - 1} and columns 0 … {gr - 2} "
                         f"(`gs[1:, :-1]`); above it (row 0, same columns) a histogram of `pts[:, 0]` that shares x with "
                         f"the scatter; to its right (rows 1 …, last column) a horizontal histogram of `pts[:, 1]` "
                         f"sharing y. Bins: 15. The top-right cell stays empty.",
         f"fig = plt.figure()\ngs = fig.add_gridspec({gr}, {gr}, hspace=0.4, wspace=0.3)\nax = fig.add_subplot(gs[1:, :-1])\n"
         f"top = fig.add_subplot(gs[0, :-1], sharex=ax)\nright = fig.add_subplot(gs[1:, -1], sharey=ax)\n"
         f"ax.scatter(pts[:, 0], pts[:, 1])\ntop.hist(pts[:, 0], bins=15)\nright.hist(pts[:, 1], bins=15, orientation='horizontal')",
         {"post": "_d = {s: a for s, a in zip(fc.spans(), fc.axes())}\n_g = fc.ax(0).get_subplotspec().get_gridspec()\n"
                  f"(sorted(_d), fc.r([_g.hspace, _g.wspace], 3), {_share('x')}, {_share('y')}, "
                  "[(len(a.collections), fc.r(sorted(fc.heights(a) + fc.widths(a)), 4)) for s, a in sorted(_d.items())])",
          "require": ["add_gridspec", "hspace", "wspace", "sharex", "sharey"]}),
        ("projection", "A cartesian and a polar view of the same class centres side by side: `fig = plt.figure(figsize="
                       "(8, 4))`; on `fig.add_subplot(1, 2, 1)` scatter the class means (one point per class, in class "
                       "order); on `fig.add_subplot(1, 2, 2, projection='polar')` plot their angles "
                       "`np.arctan2(my, mx)` against radii `np.hypot(mx, my)` as dots (`'o'`).",
         f"fig = plt.figure(figsize=(8, 4))\nm = np.array([pts[labels == c].mean(axis=0) for c in range({k})])\n"
         f"mx, my = m[:, 0], m[:, 1]\nax = fig.add_subplot(1, 2, 1)\nax.scatter(mx, my)\n"
         f"pol = fig.add_subplot(1, 2, 2, projection='polar')\npol.plot(np.arctan2(my, mx), np.hypot(mx, my), 'o')",
         {"post": "[a.name for a in fc.axes()], fc.r(np.asarray(fc.ax(0).collections[0].get_offsets()), 4), fc.lines(fc.ax(1), 'marker'), fc.size()",
          "require": ["projection", "add_subplot"], "tol": 1e-5}),
        ("add_axes", f"Lay out a figure by hand with `fig.add_axes`: the main Axes at [{l}, {b}, {w}, {h}] with a "
                     f"scatter of the points colored by class (`c=labels`), and a thin strip right of it for a colorbar "
                     f"at [{round(l + w + 0.03, 2)}, {b}, 0.03, {h}] (`fig.colorbar(sc, cax=cax)`).",
         f"fig = plt.figure()\nax = fig.add_axes(({l}, {b}, {w}, {h}))\ncax = fig.add_axes(({round(l + w + 0.03, 2)}, {b}, 0.03, {h}))\n"
         f"sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels)\nfig.colorbar(sc, cax=cax)",
         {"post": "[fc.r(a.get_position().bounds, 4) for a in fc.axes(colorbars=True)], fc.r(np.asarray(fc.ax(0).collections[0].get_offsets()), 4), fc.r(fc.ax(0).collections[0].get_array(), 4)",
          "require": ["add_axes", "colorbar", "cax"]}),
        ("subfigures", f"A two-part figure: the left subfigure (`width_ratios=[2, 1]`) holds a scatter of every point; the "
                       f"right subfigure holds {k} small stacked panels sharing x, the c-th showing a histogram of class "
                       f"c's first coordinate (`bins=np.linspace(-6, 6, 13)`), each with the class name as y label. "
                       f"Subfigure titles: 'all points' and 'per class'.",
         f"fig = plt.figure(layout='constrained')\nleft, right = fig.subfigures(1, 2, width_ratios=[2, 1])\n"
         f"left.subplots().scatter(pts[:, 0], pts[:, 1])\nleft.suptitle('all points')\naxes = right.subplots({k}, 1, sharex=True)\n"
         f"for c, ax in enumerate(axes):\n    ax.hist(pts[labels == c, 0], bins=np.linspace(-6, 6, 13))\n"
         f"    ax.set_ylabel(class_names[c])\nright.suptitle('per class')",
         {"post": "_s = fc.fig().subfigs\n([len(s.axes) for s in _s], [t._suptitle.get_text() for t in _s], "
                  "fc.r(np.asarray(_s[0].axes[0].collections[0].get_offsets()), 4), [fc.heights(a) for a in _s[1].axes], [a.get_ylabel() for a in _s[1].axes], "
                  "_s[1].axes[0].get_shared_x_axes().joined(_s[1].axes[0], _s[1].axes[-1]))", "require": ["subfigures", "sharex"]}),
        ("inset_axes", "Three insets along the top of one scatter of all points: one per class (use only the first 3 "
                       "classes), inset c at `[0.02 + 0.33 * c, 0.7, 0.3, 0.28]`, each scattering only that class' "
                       "points with the class name as inset title. Return the inset titles.",
         "fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1])\ntitles = []\nfor c in range(3):\n"
         "    ins = ax.inset_axes([0.02 + 0.33 * c, 0.7, 0.3, 0.28])\n    ins.scatter(pts[labels == c, 0], pts[labels == c, 1])\n"
         "    ins.set_title(class_names[c])\n    titles.append(ins.get_title())\nresult = titles",
         {"post": f"_, {CHILDREN}, [fc.r(np.asarray(c.collections[0].get_offsets()), 4) for c in fc.ax(0).child_axes]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def position_math(rng):
    c = rng.choice([2, 3, 4, 5])
    lft, rgt = rng.choice([(0.1, 0.9), (0.125, 0.9), (0.05, 0.95), (0.15, 0.85)])
    ws = rng.choice([0.0, 0.2, 0.25, 0.5])
    wr = [rng.randint(1, 4) for _ in range(c)]
    n = rng.randint(5, 13)
    mc = rng.choice([3, 4, 5])
    items = [
        ("get_position", f"For a 1 × {c} grid with `left={lft}`, `right={rgt}` and `wspace={ws}`, Matplotlib divides the "
                         f"width `right - left` into {c} equal panels separated by gaps of `wspace` × panel width. "
                         f"Compute each panel's `[x0, x1]` (figure fractions) **yourself** and check against "
                         f"`ax.get_position()`; return your list rounded to 4 decimals.",
         f"W = {rgt} - {lft}\npanel = W / ({c} + {ws} * ({c} - 1))\ngap = {ws} * panel\n"
         f"result = [[round({lft} + i * (panel + gap), 4), round({lft} + i * (panel + gap) + panel, 4)] for i in range({c})]",
         {"post": f"_f, _a = plt.subplots(1, {c})\n_f.subplots_adjust(left={lft}, right={rgt}, wspace={ws})\n"
                  "(_, [[round(a.get_position().x0, 4), round(a.get_position().x1, 4)] for a in _a])", "require": []}),
        ("width_ratios", f"With `width_ratios={wr}` on a 1 × {c} grid (`left={lft}`, `right={rgt}`, `wspace=0`), each "
                         f"panel gets a share of the width proportional to its ratio. Build the grid and return the "
                         f"panel widths (`ax.get_position().width`, rounded to 4 decimals).",
         f"fig, axes = plt.subplots(1, {c}, width_ratios={wr})\nfig.subplots_adjust(left={lft}, right={rgt}, wspace=0)\n"
         f"result = [round(ax.get_position().width, 4) for ax in axes]",
         {"require": ["width_ratios", "subplots_adjust"]}),
        ("plt.subplots", f"Write `grid_for(n, max_cols={mc})` returning `(rows, cols)` for n panels: one row of n panels "
                         f"when n ≤ max_cols, else max_cols columns and as many rows as needed. Then build "
                         f"`fig, axes = plt.subplots(*grid_for({n}), squeeze=False)`, remove the surplus panels, and "
                         f"return `(grid_for({n}), len(fig.axes))`.",
         f"def grid_for(n, max_cols={mc}):\n    cols = min(n, max_cols)\n    return -(-n // cols), cols\n"
         f"fig, axes = plt.subplots(*grid_for({n}), squeeze=False)\nfor ax in axes.flat[{n}:]:\n    ax.remove()\n"
         f"result = (grid_for({n}), len(fig.axes))",
         {"post": "_, [grid_for(k) for k in (1, 3, 7, 12)], fc.spans()"}),
        ("add_axes", f"Tile {c} hand-placed Axes across the figure with `fig.add_axes`: equal widths, a gap of 0.02 "
                     f"between neighbours, starting at x = {lft} and ending at x = {rgt}, all at bottom 0.15 with "
                     f"height 0.7. Return their bounds rounded to 4 decimals.",
         f"fig = plt.figure()\nw = ({rgt} - {lft} - 0.02 * ({c} - 1)) / {c}\n"
         f"axes = [fig.add_axes(({lft} + i * (w + 0.02), 0.15, w, 0.7)) for i in range({c})]\n"
         f"result = [[round(v, 4) for v in ax.get_position().bounds] for ax in axes]"),
        ("get_position", f"A plain figure places Axes from the fixed `subplotpars`; `layout='constrained'` recomputes "
                         f"the positions **at draw time** to fit titles and tick labels. Create two 1 × {c} grids, one "
                         f"plain and one constrained, give every Axes the title 'a title', and return the first Axes' "
                         f"height in each (plain, then constrained; rounded to 3 decimals) **before** and **after** "
                         f"`fig.canvas.draw()`: `[[plain, constrained] before, [plain, constrained] after]`.",
         f"f1, a1 = plt.subplots(1, {c})\nf2, a2 = plt.subplots(1, {c}, layout='constrained')\nfor ax in list(a1) + list(a2):\n"
         f"    ax.set_title('a title')\nh = lambda ax: round(ax.get_position().height, 3)\nbefore = [h(a1[0]), h(a2[0])]\n"
         f"f1.canvas.draw()\nf2.canvas.draw()\nresult = [before, [h(a1[0]), h(a2[0])]]",
         {"post": "_, " + "[type(f.get_layout_engine()).__name__ for f in fc.figs()]", "require": ["layout", "draw"]}),
    ]
    return catalog(rng, items, "", "")


# ================================================================= challenge

def _mosaic_setup(seed: int, R: int, C: int, p: float, corrupt: float, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nR, C = {R}, {C}\n"
            f"ri = np.cumsum(g.random(R) < {p}) - 1\nri[ri < 0] = 0\nci = np.cumsum(g.random(C) < {p}) - 1\nci[ci < 0] = 0\n"
            f"nc = ci.max() + 1\nbid = ri[:, None] * nc + ci[None, :]\nbanner = g.random(ri.max() + 1) < 0.3\n"
            f"bid = np.where(banner[ri][:, None], ri[:, None] * nc, bid)\nu, inv = np.unique(bid, return_inverse=True)\n"
            f"keep = g.random(len(u)) >= 0.15\nkeep[0] = True\nids = np.full(len(u), -1)\nids[keep] = g.permutation(keep.sum())\n"
            f"grid = ids[inv.reshape(R, C)]\nbad = g.random((R, C)) < {corrupt}\n"
            f"grid[bad] = g.integers(-1, keep.sum(), bad.sum())\nreport = {report!r}")


@template(M, "challenge")
def mosaic_spans(rng):
    s = seeded(rng)
    report = rng.choice(["spans", "invalid", "touching"])
    R, C = rng.randint(4, 6), rng.randint(4, 7)
    corrupt = 0.12 if report == "invalid" else 0.0
    setup = _mosaic_setup(s, R, C, 0.45, corrupt, report)
    head = ("lab = grid.ravel()\nrows = np.repeat(np.arange(grid.shape[0]), grid.shape[1])\n"
            "cols = np.tile(np.arange(grid.shape[1]), grid.shape[0])\nm = lab >= 0\n"
            "lab, rows, cols = lab[m], rows[m], cols[m]\norder = np.argsort(lab, kind='stable')\n"
            "lab, rows, cols = lab[order], rows[order], cols[order]\n"
            "starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])\nnames = lab[starts]\n"
            "r0 = np.minimum.reduceat(rows, starts)\nr1 = np.maximum.reduceat(rows, starts) + 1\n"
            "c0 = np.minimum.reduceat(cols, starts)\nc1 = np.maximum.reduceat(cols, starts) + 1\n")
    tail = {
        "spans": "result = np.column_stack([r0, r1, c0, c1])",
        "invalid": "count = np.diff(np.r_[starts, len(lab)])\nresult = names[count != (r1 - r0) * (c1 - c0)].tolist()",
        "touching": ("a = np.concatenate([grid[:, :-1].ravel(), grid[:-1, :].ravel()])\n"
                     "b = np.concatenate([grid[:, 1:].ravel(), grid[1:, :].ravel()])\nm = (a >= 0) & (b >= 0) & (a != b)\n"
                     "lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])\n"
                     "result = len(np.unique(lo.astype(np.int64) * (grid.max() + 1) + hi))"),
    }[report]
    sol = (head + tail) if report != "touching" else tail
    tests = [_mosaic_setup(s + i, RR, CC, p, corrupt / 20 if corrupt else 0.0, report)
             for i, (RR, CC, p) in ((1, (600, 700, 0.1)), (2, (900, 500, 0.02)))]
    want = {"spans": "a k × 4 integer array: row i is `(row0, row1, col0, col1)` (end-exclusive) of the panel "
                     "labelled i — the cells `subplot_mosaic` gives it. Every label 0 … k-1 occurs and forms a rectangle",
            "invalid": "the sorted list of labels whose cells do **not** form one filled rectangle (the panels that "
                       "make `subplot_mosaic` raise *must form a rectangle*)",
            "touching": "how many unordered pairs of different panels touch along an edge (some cell of one is directly "
                        "left of, right of, above or below some cell of the other)"}[report]
    return challenge(
        "Mosaic bookkeeping",
        "`plt.subplot_mosaic` takes a grid of panel labels and gives each label the smallest block of rows and "
        "columns that contains all its cells; the layout is only valid if those cells fill the block exactly. Here "
        "the mosaic is an integer array `grid` (R × C): `grid[r, c]` is the label of the panel covering that cell, "
        f"or -1 for an empty cell (`.`). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "R, C ≤ 1000", speed="O(R·C log(R·C)) with vectorized NumPy (no Python loop over cells)")


def _gs_setup(seed: int, nr: int, nc: int, q: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nnrows, ncols = {nr}, {nc}\n"
            f"left, right = np.round(g.uniform(0.05, 0.2), 3), np.round(g.uniform(0.8, 0.95), 3)\n"
            f"bottom, top = np.round(g.uniform(0.05, 0.2), 3), np.round(g.uniform(0.8, 0.95), 3)\n"
            f"wspace, hspace = np.round(g.uniform(0, 0.5, 2), 2)\nwidth_ratios = g.integers(1, 5, ncols)\n"
            f"height_ratios = g.integers(1, 5, nrows)\nr0 = g.integers(0, nrows, {q})\nr1 = r0 + 1 + g.integers(0, nrows, {q}) % (nrows - r0)\n"
            f"c0 = g.integers(0, ncols, {q})\nc1 = c0 + 1 + g.integers(0, ncols, {q}) % (ncols - c0)\n"
            f"figsize = (8.0, 6.0)\nreport = {report!r}")


@template(M, "challenge")
def gridspec_positions(rng):
    s = seeded(rng)
    report = rng.choice(["bounds", "aspect", "center"])
    nr, nc, q = rng.randint(2, 4), rng.randint(2, 5), rng.randint(4, 6)
    setup = _gs_setup(s, nr, nc, q, report)
    sol = ("def edges(lo, hi, n, space, ratios):\n    cell = (hi - lo) / (n + space * (n - 1))\n"
           "    sizes = ratios * (cell * n / ratios.sum())\n    gaps = np.r_[0.0, np.full(n - 1, space * cell)]\n"
           "    ends = np.cumsum(np.column_stack([gaps, sizes]).ravel()).reshape(n, 2)\n"
           "    return ends[:, 0], ends[:, 1]      # start / end offsets of each cell\n\n"
           "x_start, x_end = edges(left, right, ncols, wspace, width_ratios)\n"
           "y_start, y_end = edges(bottom, top, nrows, hspace, height_ratios)   # measured down from the top\n"
           "x0, x1 = left + x_start[c0], left + x_end[c1 - 1]\ny1, y0 = top - y_start[r0], top - y_end[r1 - 1]\n"
           + {"bounds": "result = np.column_stack([x0, y0, x1 - x0, y1 - y0])",
              "aspect": "result = (x1 - x0) * figsize[0] / ((y1 - y0) * figsize[1])",
              "center": "result = np.column_stack([(x0 + x1) / 2, (y0 + y1) / 2])"}[report])
    tests = [_gs_setup(s + i, a, b, qq, report) for i, (a, b, qq) in ((1, (100000, 100000, 100000)),
                                                                      (2, (3, 200000, 150000)))]
    want = {"bounds": "a q × 4 array of `[left, bottom, width, height]` per query (figure fractions), what "
                      "`gs[r0:r1, c0:c1].get_position(fig).bounds` gives",
            "aspect": "a length-q array with each Axes' aspect ratio width / height **in inches** (the figure is "
                      "`figsize` inches)",
            "center": "a q × 2 array with the centre `(x, y)` of each Axes in figure fractions"}[report]
    return challenge(
        "Where does the panel go?",
        "A `GridSpec(nrows, ncols, left, right, bottom, top, wspace, hspace, width_ratios, height_ratios)` splits the "
        "box `left … right` × `bottom … top` of the figure into cells. Along x: a cell unit is `cell = (right - left) "
        "/ (ncols + wspace·(ncols - 1))`, every gap between neighbouring columns is `wspace·cell`, and the columns "
        "share `ncols·cell` in proportion to `width_ratios`, from left to right. Rows work the same way with "
        "`hspace`, `height_ratios` and `top - bottom`, row 0 at the **top**. An Axes on `gs[r0:r1, c0:c1]` (end-exclusive) "
        f"covers its cells and the gaps between them. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "nrows, ncols, q ≤ 2·10^5", speed="O(nrows + ncols + q): prefix sums", tol=1e-6)


def _share_setup(seed: int, n: int, m: int, report: str) -> str:
    return (f"g = np.random.default_rng({seed})\nn = {n}\nlo = np.round(g.uniform(-50, 50, n), 2)\n"
            f"hi = np.round(lo + g.uniform(0.5, 30, n), 2)\na = g.integers(0, n, {m})\nb = g.integers(0, n, {m})\n"
            f"report = {report!r}")


@template(M, "challenge")
def shared_axes_groups(rng):
    s = seeded(rng)
    report = rng.choice(["xlim", "groups", "largest"])
    n, m = rng.randint(5, 8), rng.randint(2, 5)
    setup = _share_setup(s, n, m, report)
    sol = ("parent = list(range(n))\n\ndef find(i):\n    root = i\n    while parent[root] != root:\n        root = parent[root]\n"
           "    while parent[i] != root:\n        parent[i], i = root, parent[i]\n    return root\n\n"
           "for i, j in zip(a.tolist(), b.tolist()):\n    ri, rj = find(i), find(j)\n    if ri != rj:\n        parent[ri] = rj\n"
           "roots = np.array([find(i) for i in range(n)])\n"
           + {"xlim": "glo = np.full(n, np.inf)\nghi = np.full(n, -np.inf)\nnp.minimum.at(glo, roots, lo)\n"
                      "np.maximum.at(ghi, roots, hi)\nL, H = glo[roots], ghi[roots]\nd = 0.05 * (H - L)\n"
                      "result = np.column_stack([L - d, H + d])",
              "groups": "result = len(np.unique(roots))",
              "largest": "result = int(np.bincount(roots).max())"}[report])
    tests = [_share_setup(s + i, nn, mm, report) for i, (nn, mm) in ((1, (100000, 80000)), (2, (30000, 120000)))]
    want = {"xlim": "an n × 2 array with every Axes' final x limits: a shared group autoscales to the smallest "
                    "`lo` and the largest `hi` of **all** its members, plus a 5% margin of that span on each side",
            "groups": "the number of groups (an Axes that shares with nobody is a group of its own)",
            "largest": "the size of the largest group"}[report]
    return challenge(
        "Shared axes",
        "`sharex` joins Axes into groups that move together: sharing is transitive, so if A shares with B and B "
        "with C, all three show the same x-range. There are `n` Axes; Axes `i` holds one line spanning x = `lo[i]` … "
        "`hi[i]`. Then `ax[a[j]]` and `ax[b[j]]` are joined for every j (in Matplotlib: "
        f"`ax._shared_axes['x'].join(...)`). Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 10^5 Axes, ≤ 1.2·10^5 joins",
        speed="about O((n + m) α(n)): union-find with path compression", tol=1e-6)

"""5.10 Colors, colormaps, norms & colorbars."""

from ._common import P, catalog, categories, challenge, classes, lit, matrix, seeded, signal, template
from ... import themes as T

M = "5.10"

# ---------------------------------------------------------------- check code (post): images, norms, colorbars as data
_PRE = '''from matplotlib.colors import to_hex as _hex

def _img(im, *keys):
    out = {}
    cm, nm = im.get_cmap(), im.norm
    for k in keys:
        if k == 'cmap':
            out[k] = cm.name
        elif k == 'N':
            out[k] = cm.N
        elif k == 'norm':
            out[k] = type(nm).__name__
        elif k == 'clim':
            out[k] = [float(f'{float(v):.6g}') for v in im.get_clim()]
        elif k == 'vcenter':
            out[k] = None if getattr(nm, 'vcenter', None) is None else round(float(nm.vcenter), 4)
        elif k == 'bounds':
            out[k] = fc.r(nm.boundaries, 4) if hasattr(nm, 'boundaries') else None
        elif k == 'lut':
            out[k] = [_hex(c) for c in cm(np.arange(cm.N))] if cm.N <= 16 else None
        elif k == 'sample':
            out[k] = [_hex(c) for c in cm(np.linspace(0, 1, 5))]
        elif k == 'extremes':
            out[k] = [_hex(cm.get_under()), _hex(cm.get_over()), _hex(cm.get_bad(), keep_alpha=True)]
        elif k == 'data':
            out[k] = fc.r(im.get_array(), 4)
        elif k == 'masked':
            out[k] = int(np.ma.getmaskarray(im.get_array()).sum())
    return out

def _imgs(a, *keys):
    return [_img(im, *keys) for im in a.get_images()]

def _sc(a, *keys, i=0):
    from matplotlib.collections import PathCollection
    return _img([c for c in a.collections if isinstance(c, PathCollection)][i], *keys)

def _cbs(*keys, f=None):
    f = f or fc.fig()
    axes = fc.axes(f)
    out = []
    for a in list(f.axes) + [c for b in f.axes for c in b.child_axes]:
        cb = getattr(a, '_colorbar', None)
        if cb is None:
            continue
        vert = cb.orientation == 'vertical'
        axis = cb.ax.yaxis if vert else cb.ax.xaxis
        d = {}
        for k in keys:
            if k == 'label':
                d[k] = axis.get_label().get_text()
            elif k == 'orientation':
                d[k] = cb.orientation
            elif k == 'extend':
                d[k] = cb.extend
            elif k == 'ticks':
                d[k] = [float(f'{float(v):.6g}') for v in cb.get_ticks()]
            elif k == 'ticklabels':
                d[k] = [t.get_text() for t in axis.get_ticklabels()]
            elif k == 'labelsize':
                d[k] = round(axis.get_major_ticks()[0].label1.get_fontsize(), 2)
            elif k == 'of':
                ma = getattr(cb.mappable, 'axes', None)
                d[k] = axes.index(ma) if ma in axes else None
            elif k == 'shrink':
                d[k] = round(cb.ax.get_position().height / cb.mappable.axes.get_position().height, 3)
            elif k == 'cmap':
                d[k] = cb.cmap.name
            elif k == 'norm':
                d[k] = type(cb.norm).__name__
            elif k == 'clim':
                d[k] = [float(f'{float(v):.6g}') for v in (cb.norm.vmin, cb.norm.vmax)]
            elif k == 'box':
                d[k] = [round(float(v), 3) for v in cb.ax.get_position().bounds]
            elif k == 'left':
                d[k] = bool(cb.ax.get_position().x0 < cb.mappable.axes.get_position().x0)
            elif k == 'fmt':
                d[k] = [cb.formatter(v) for v in (0.25, 1.5, 12.0)]
        out.append(d)
    return out

def _lines(a, *keys):
    out = []
    for l in a.get_lines():
        d = {}
        for k in keys:
            if k == 'color':
                d[k] = fc.hexcolor(l.get_color())
            elif k == 'alpha':
                a_ = l.get_alpha()
                d[k] = round(a_ if a_ is not None else fc.alpha(l.get_color()), 3)
            elif k == 'lw':
                d[k] = round(l.get_linewidth(), 3)
            elif k == 'y':
                d[k] = fc.r(l.get_ydata(), 4)
        out.append(d)
    return out
'''


def post(expr: str) -> str:
    return _PRE + expr


def _mats(rng, n=(6, 10)):
    """`data` (positive, log-normal) and `signed` (centred at 0) n × n matrices."""
    n_ = rng.randint(*n)
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\ndata = np.round(g.lognormal(size=({n_}, {n_})), 4)\n"
            f"signed = np.round(g.normal(size=({n_}, {n_})), 4)")
    return code, f"`data` is a {n_} × {n_} matrix of positive (log-normal) values, `signed` one centred at 0", n_


def _sweep(rng, k=None):
    """`lrs` (log-spaced learning rates) and `runs` (one loss curve per lr)."""
    k = k or rng.randint(5, 8)
    hi = rng.choice([-1, -2])
    s = seeded(rng)
    code = (f"lrs = np.logspace({hi}, {hi - 3}, {k})\ng = np.random.default_rng({s})\nsteps = np.arange(40)\n"
            f"tau = 4 + 30 * np.abs(np.log10(lrs) + 2.5)\n"
            f"runs = np.round(np.exp(-steps / tau[:, None]) + 0.03 * g.random(({k}, 40)), 4)")
    return code, f"`lrs` holds {k} learning rates (log-spaced, largest first) and `runs[i]` the loss curve (40 `steps`) of lr `lrs[i]`", k


# ================================================================= lite

@template(M, "lite")
def color_specs(rng):
    code, desc, n = signal(rng)
    spec = rng.choice(["C0", "C3", "tab:green", "navy", "#ff7f0e", "0.4"])
    rgb = rng.choice([(1, 0, 0), (0, 0.5, 1), (0.2, 0.4, 0.6), (1, 1, 0)])
    cm = rng.choice(["viridis", "plasma", "magma", "cividis", "inferno"])
    v = rng.choice([0, 0.25, 0.5, 0.75, 1])
    k = rng.choice([4, 5, 6, 8])
    named = rng.choice(["tab:orange", "tab:purple", "darkgreen", "crimson"])
    gray = rng.choice(["0.6", "0.3", "0.85"])
    hx = rng.choice(["#2b83ba", "#d7191c", "#1a9850", "#7570b3"])
    LINE = "fc.lines(fc.ax(), 'color')"
    items = [
        ("to_rgba", f"Return `to_rgba({spec!r})` (import it from `matplotlib.colors`): the color as an (r, g, b, a) "
                    f"tuple of floats in 0 … 1.", f"from matplotlib.colors import to_rgba\nresult = to_rgba({spec!r})"),
        ("to_hex", f"Return the hex string of the RGB color {rgb} (floats in 0 … 1) with `to_hex`.",
         f"from matplotlib.colors import to_hex\nresult = to_hex({rgb})"),
        ("colormaps", f"A colormap maps a number in 0 … 1 to a color. Return the RGBA color `mpl.colormaps[{cm!r}]` "
                      f"gives for {v}, as a tuple.", f"result = tuple(mpl.colormaps[{cm!r}]({v}))"),
        ("colormaps", "`mpl.colormaps` is a registry you can test names against. Return `[name in mpl.colormaps for "
                      "name in ('viridis_r', 'RdBu_r', 'virdis', 'tab10')]`.",
         "result = [name in mpl.colormaps for name in ('viridis_r', 'RdBu_r', 'virdis', 'tab10')]"),
        ("color", f"Create `fig, ax` and plot `y` against `x` in the named color {named!r}.",
         f"fig, ax = plt.subplots()\nax.plot(x, y, color={named!r})", {"post": LINE}),
        ("color", f"Create `fig, ax` and plot `y` against `x` in a gray level: the **string** {gray!r} (0 = black, "
                  f"1 = white).", f"fig, ax = plt.subplots()\nax.plot(x, y, color={gray!r})", {"post": LINE}),
        ("color", "Create `fig, ax` and plot `y` against `x` with the RGBA tuple `(0.1, 0.2, 0.5, 0.3)` as its color "
                  "(the 4th number is the opacity).", "fig, ax = plt.subplots()\nax.plot(x, y, color=(0.1, 0.2, 0.5, 0.3))",
         {"post": post(f"{LINE}, _lines(fc.ax(), 'alpha')")}),
        ("reversed", f"Every colormap has a reversed twin. Return the name of `mpl.colormaps[{cm!r}].reversed()`.",
         f"result = mpl.colormaps[{cm!r}].reversed().name"),
        ("resampled", f"`cmap.resampled({k})` keeps only {k} colors. Return them as hex strings: call the resampled "
                      f"colormap on `np.arange({k})` (integers index the color table directly) and convert each row "
                      f"with `mpl.colors.to_hex`.",
         f"cmap = mpl.colormaps[{cm!r}].resampled({k})\nresult = [mpl.colors.to_hex(c) for c in cmap(np.arange({k}))]"),
        ("linspace", f"Sample {k} evenly spaced colors from {cm!r}: `mpl.colormaps[{cm!r}](np.linspace(0, 1, {k}))`. "
                     f"Return the shape of the array you get.",
         f"colors = mpl.colormaps[{cm!r}](np.linspace(0, 1, {k}))\nresult = colors.shape"),
        ("color", f"Create `fig, ax` and plot `y` against `x` in the hex color {hx!r}.",
         f"fig, ax = plt.subplots()\nax.plot(x, y, color={hx!r})", {"post": LINE}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def colormap_images(rng):
    code, desc, n = matrix(rng)
    cm = rng.choice(["magma", "cividis", "Greys", "Blues", "YlOrRd", "plasma"])
    k = rng.choice([3, 4, 5, 6])
    lab = rng.choice(["value", "score", "probability", "weight"])
    base = "fig, ax = plt.subplots()\n"
    IM = "_imgs(fc.ax(0), {})"
    items = [
        ("imshow", f"Create `fig, ax` and show `M` with `ax.imshow`, using the colormap {cm!r}.",
         base + f"ax.imshow(M, cmap={cm!r})", {"post": post(IM.format("'cmap', 'data'"))}),
        ("vmin", "Create `fig, ax` and show `M` with fixed color limits 0 … 1 (`vmin=`, `vmax=`) instead of the data's "
                 "own min and max.", base + "ax.imshow(M, vmin=0, vmax=1)",
         {"post": post(IM.format("'clim'")), "require": ["vmin", "vmax"]}),
        ("colorbar", "Create `fig, ax`, show `M` with `im = ax.imshow(M)`, and add its colorbar with "
                     "`fig.colorbar(im, ax=ax)`.", base + "im = ax.imshow(M)\nfig.colorbar(im, ax=ax)",
         {"post": post("_cbs('of', 'orientation')")}),
        ("label", f"Create `fig, ax`, show `M` with `ax.imshow`, and add a colorbar labelled {lab!r} (`label=`).",
         base + f"im = ax.imshow(M)\nfig.colorbar(im, ax=ax, label={lab!r})", {"post": post("_cbs('label')")}),
        ("orientation", "Create `fig, ax`, show `M` with `ax.imshow`, and add a **horizontal** colorbar.",
         base + "im = ax.imshow(M)\nfig.colorbar(im, ax=ax, orientation='horizontal')",
         {"post": post("_cbs('orientation')")}),
        ("reversed", f"Create `fig, ax` and show `M` with the reversed {cm!r} colormap, made with "
                     f"`mpl.colormaps[{cm!r}].reversed()`.", base + f"ax.imshow(M, cmap=mpl.colormaps[{cm!r}].reversed())",
         {"post": post(IM.format("'cmap'"))}),
        ("resampled", f"Create `fig, ax` and show `M` with only {k} colors: `mpl.colormaps['viridis'].resampled({k})`.",
         base + f"ax.imshow(M, cmap=mpl.colormaps['viridis'].resampled({k}))",
         {"post": post(IM.format("'cmap', 'N', 'lut'"))}),
        ("LogNorm", "Create `fig, ax` and show `M + 0.01` on a **logarithmic** color scale: `norm=LogNorm()` (from "
                    "`matplotlib.colors`).", "from matplotlib.colors import LogNorm\n" + base +
         "ax.imshow(M + 0.01, norm=LogNorm())", {"post": post(IM.format("'norm', 'clim'"))}),
        ("plt.colorbar", f"With **pyplot only**, show `M` with `plt.imshow` in the colormap {cm!r} and add a colorbar "
                         f"(`plt.colorbar()`).", f"plt.imshow(M, cmap={cm!r})\nplt.colorbar()",
         {"post": post(f"{IM.format(chr(39) + 'cmap' + chr(39))}, _cbs('of')")}),
        ("scatter", f"Create `fig, ax` and scatter the first row of `M` against its column index, colored by its own "
                    f"values with {cm!r}: `ax.scatter(np.arange(len(M)), M[0], c=M[0], cmap={cm!r})`.",
         base + f"ax.scatter(np.arange(len(M)), M[0], c=M[0], cmap={cm!r})",
         {"post": post("_sc(fc.ax(), 'cmap', 'data')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def norm_mapping(rng):
    code, desc, n = _mats(rng)
    s = seeded(rng)
    lo, hi = rng.choice([(0, 10), (-5, 5), (2, 4), (-1, 3)])
    c = rng.choice([0, 0.5, 1])
    h = rng.choice([1, 2, 4])
    bounds = rng.choice([[-3, -1, 0, 1, 3], [-2, -0.5, 0.5, 2], [0, 1, 2, 4, 8]])
    extra = f"vals = np.round(np.random.default_rng({s}).uniform(-4, 9, 6), 2)"
    items = [
        ("Normalize", f"A norm maps data to 0 … 1 before the colormap looks it up. Return `Normalize(vmin={lo}, "
                      f"vmax={hi})(vals)` as a list (values outside the range land below 0 or above 1).",
         f"from matplotlib.colors import Normalize\nresult = Normalize(vmin={lo}, vmax={hi})(vals).tolist()",
         {"setup": extra}),
        ("LogNorm", "Return `LogNorm(vmin=1, vmax=1000)` applied to `[1, 10, 100, 1000, 31.6]` as a list: each decade "
                    "is a third of the range.",
         "from matplotlib.colors import LogNorm\nresult = LogNorm(vmin=1, vmax=1000)([1, 10, 100, 1000, 31.6]).tolist()",
         {"tol": 1e-4}),
        ("TwoSlopeNorm", f"`TwoSlopeNorm` uses two linear pieces so that `vcenter` lands exactly on 0.5. Return "
                         f"`TwoSlopeNorm(vcenter={c}, vmin=-2, vmax=8)` applied to `[-2, -1, {c}, 3, 8]` as a list.",
         f"from matplotlib.colors import TwoSlopeNorm\nresult = TwoSlopeNorm(vcenter={c}, vmin=-2, vmax=8)([-2, -1, {c}, 3, 8]).tolist()"),
        ("CenteredNorm", f"`CenteredNorm(vcenter=0, halfrange={h})` maps -{h} … {h} to 0 … 1. Apply it to `vals` and "
                         f"return the result as a list.",
         f"from matplotlib.colors import CenteredNorm\nresult = CenteredNorm(vcenter=0, halfrange={h})(vals).tolist()",
         {"setup": extra}),
        ("BoundaryNorm", f"`BoundaryNorm(boundaries, ncolors)` maps each value to the **index** of its bin. Return "
                         f"`BoundaryNorm({bounds}, ncolors={len(bounds) - 1})(vals)` as a list (-1 = below the first "
                         f"boundary, {len(bounds) - 1} = at or above the last).",
         f"from matplotlib.colors import BoundaryNorm\nresult = BoundaryNorm({bounds}, ncolors={len(bounds) - 1})(vals).tolist()",
         {"setup": extra}),
        ("inverse", "Norms can be inverted. Return the data value that `LogNorm(vmin=1, vmax=1000)` maps to 0.5 "
                    "(`.inverse(0.5)`), as a float.",
         "from matplotlib.colors import LogNorm\nresult = float(LogNorm(vmin=1, vmax=1000).inverse(0.5))", {"tol": 1e-6}),
        ("LogNorm", "Create `fig, ax` and show `data` with a log color scale spanning its own range: "
                    "`norm=LogNorm(vmin=data.min(), vmax=data.max())`, colormap 'viridis'.",
         "from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\n"
         "ax.imshow(data, cmap='viridis', norm=LogNorm(vmin=data.min(), vmax=data.max()))",
         {"post": post("_imgs(fc.ax(0), 'norm', 'clim', 'cmap')")}),
        ("TwoSlopeNorm", "Create `fig, ax` and show `signed` with the diverging colormap 'RdBu_r' so that white sits "
                         "exactly at 0: `norm=TwoSlopeNorm(vcenter=0)`.",
         "from matplotlib.colors import TwoSlopeNorm\nfig, ax = plt.subplots()\nax.imshow(signed, cmap='RdBu_r', "
         "norm=TwoSlopeNorm(vcenter=0))", {"post": post("_imgs(fc.ax(0), 'norm', 'vcenter', 'cmap')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def colorbar_options(rng):
    code, desc, n = _mats(rng)
    sh = rng.choice([0.5, 0.6, 0.8, 0.9])
    ls = rng.choice([6, 7, 8, 9])
    lab = rng.choice(["z", "weight", "activation", "update"])
    base = "fig, ax = plt.subplots()\nim = ax.imshow(signed, cmap='RdBu_r')\n"
    start = "Create `fig, ax` and show `signed` with `im = ax.imshow(signed, cmap='RdBu_r')`."
    items = [
        ("shrink", f"{start} Add a colorbar {int(sh * 100)}% as tall as the Axes (`shrink=`).",
         base + f"fig.colorbar(im, ax=ax, shrink={sh})", {"post": post("_cbs('shrink')")}),
        ("set_ticks", f"{start} Add a colorbar and put its ticks at -2, 0 and 2 (`cb.set_ticks`).",
         base + "cb = fig.colorbar(im, ax=ax)\ncb.set_ticks([-2, 0, 2])", {"post": post("_cbs('ticks')")}),
        ("tick_params", f"{start} Add a colorbar whose tick labels are {ls} pt: the colorbar's own Axes is `cb.ax`, so "
                        f"use `cb.ax.tick_params(labelsize={ls})`.",
         base + f"cb = fig.colorbar(im, ax=ax)\ncb.ax.tick_params(labelsize={ls})",
         {"post": post("_cbs('labelsize')"), "require": ["tick_params", "labelsize"]}),
        ("extend", "Create `fig, ax`, show `signed` with `cmap='RdBu_r'` clipped to -1 … 1 (`vmin`, `vmax`), and add a "
                   "colorbar with pointed ends showing that values go beyond both limits (`extend='both'`).",
         "fig, ax = plt.subplots()\nim = ax.imshow(signed, cmap='RdBu_r', vmin=-1, vmax=1)\n"
         "fig.colorbar(im, ax=ax, extend='both')", {"post": post("_cbs('extend', 'clim')")}),
        ("orientation", f"{start} Add a horizontal colorbar labelled {lab!r}.",
         base + f"fig.colorbar(im, ax=ax, orientation='horizontal', label={lab!r})",
         {"post": post("_cbs('orientation', 'label')")}),
        ("set_label", f"{start} Add a colorbar, keep it (`cb = fig.colorbar(...)`), and label it {lab!r} afterwards "
                      f"with `cb.set_label`.", base + f"cb = fig.colorbar(im, ax=ax)\ncb.set_label({lab!r})",
         {"post": post("_cbs('label')")}),
        ("set_ticks", f"{start} Add a colorbar with ticks at -1, 0, 1 labelled 'neg', 'zero', 'pos' "
                      f"(`cb.set_ticks(ticks, labels=...)`).",
         base + "cb = fig.colorbar(im, ax=ax)\ncb.set_ticks([-1, 0, 1], labels=['neg', 'zero', 'pos'])",
         {"post": post("_cbs('ticks', 'ticklabels')")}),
        ("colorbar", "Create `fig, (ax1, ax2) = plt.subplots(1, 2)`: show `data` on `ax1` (colormap 'viridis') and "
                     "`signed` on `ax2` (colormap 'RdBu_r'), each with its **own** colorbar next to it.",
         "fig, (ax1, ax2) = plt.subplots(1, 2)\nim1 = ax1.imshow(data, cmap='viridis')\nfig.colorbar(im1, ax=ax1)\n"
         "im2 = ax2.imshow(signed, cmap='RdBu_r')\nfig.colorbar(im2, ax=ax2)",
         {"post": post("[_imgs(a, 'cmap') for a in fc.axes()], _cbs('of', 'cmap')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def line_colors(rng):
    code, desc, k = _sweep(rng)
    cm = rng.choice(["plasma", "viridis", "magma", "cividis"])
    cyc = rng.choice([["k", "r", "b"], ["tab:blue", "0.5"], ["#0072B2", "#E69F00", "#009E73"]])
    loop = "fig, ax = plt.subplots()\nfor r in runs:\n    ax.plot(steps, r)\n"
    COL = "_lines(fc.ax(), 'color')"
    items = [
        ("colormaps", f"Plot every run on `fig, ax`, colored by sampling {cm!r} evenly: `colors = "
                      f"mpl.colormaps[{cm!r}](np.linspace(0, 1, len(runs)))`, run i in `colors[i]`.",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}](np.linspace(0, 1, len(runs)))\n"
         f"for r, c in zip(runs, colors):\n    ax.plot(steps, r, color=c)", {"post": post(COL)}),
        ("set_prop_cycle", f"Create `fig, ax`, set its color cycle to {cyc} with `ax.set_prop_cycle(color=...)`, then "
                           f"plot every run **without** `color=` (the cycle repeats when it runs out).",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color={cyc})\nfor r in runs:\n    ax.plot(steps, r)",
         {"post": post(COL)}),
        ("color", "Plot every run on `fig, ax` in the cycle colors given explicitly: run i in `f'C{i}'`, but run 0 in "
                  "black ('k') to stand out.",
         "fig, ax = plt.subplots()\nfor i, r in enumerate(runs):\n    ax.plot(steps, r, color='k' if i == 0 else f'C{i}')",
         {"post": post(COL)}),
        ("reversed", f"Plot every run on `fig, ax` with colors sampled evenly from the **reversed** {cm!r} "
                     f"(`mpl.colormaps[{cm!r}].reversed()`), so the first run gets the other end of the map.",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}].reversed()(np.linspace(0, 1, len(runs)))\n"
         f"for r, c in zip(runs, colors):\n    ax.plot(steps, r, color=c)", {"post": post(COL)}),
        ("alpha", "Plot every run on `fig, ax` in 'C0' with `alpha=0.3`, then their mean (`runs.mean(axis=0)`) in "
                  "black with line width 2.",
         "fig, ax = plt.subplots()\nfor r in runs:\n    ax.plot(steps, r, color='C0', alpha=0.3)\n"
         "ax.plot(steps, runs.mean(axis=0), color='k', lw=2)", {"post": post("_lines(fc.ax(), 'color', 'alpha', 'lw', 'y')")}),
        ("to_hex", "Plot every run on `fig, ax` with the default colors and return the line colors as hex strings "
                   "(`mpl.colors.to_hex(line.get_color())`): the default cycle has 10 colors.",
         loop + "result = [mpl.colors.to_hex(l.get_color()) for l in ax.get_lines()]"),
        ("LogNorm", f"Color each run by its learning rate on a log scale: `norm = LogNorm(lrs.min(), lrs.max())`, "
                    f"`cmap = mpl.colormaps[{cm!r}]`, and run i gets `cmap(norm(lrs[i]))`.",
         f"from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\nnorm = LogNorm(lrs.min(), lrs.max())\n"
         f"cmap = mpl.colormaps[{cm!r}]\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, color=cmap(norm(lr)))",
         {"post": post(COL)}),
        ("set_prop_cycle", "The guide's colorblind-safe palette: set the Axes' cycle to `['#0072B2', '#E69F00', "
                           "'#009E73', '#D55E00']` and plot the first two runs, then **reset** the cycle with the same "
                           "list and plot the last run (it starts again at '#0072B2'). Return the three line colors as "
                           "hex.",
         "fig, ax = plt.subplots()\npal = ['#0072B2', '#E69F00', '#009E73', '#D55E00']\nax.set_prop_cycle(color=pal)\n"
         "ax.plot(steps, runs[0])\nax.plot(steps, runs[1])\nax.set_prop_cycle(color=pal)\nax.plot(steps, runs[-1])\n"
         "result = [mpl.colors.to_hex(l.get_color()) for l in ax.get_lines()]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def custom_colormaps(rng):
    code, desc, n = matrix(rng)
    pal = rng.choice([["#2b83ba", "#fdae61", "#d7191c"], ["#1b9e77", "#d95f02", "#7570b3"], ["white", "gray", "black"]])
    ends = rng.choice([["white", "navy"], ["white", "darkred"], ["lightyellow", "darkgreen"]])
    k = rng.choice([3, 4, 5, 6])
    name = "_".join(e.replace("light", "l").replace("dark", "d") for e in ends)
    items = [
        ("ListedColormap", f"A `ListedColormap` is a plain list of colors. Show `classes` (integers 0 … 2) on `fig, ax` "
                           f"with `ListedColormap({pal})`.",
         f"from matplotlib.colors import ListedColormap\nfig, ax = plt.subplots()\nax.imshow(classes, cmap=ListedColormap({pal}))",
         {"post": post("_imgs(fc.ax(0), 'lut', 'data')"),
          "setup": f"classes = (M * 3).astype(int)"}),
        ("LinearSegmentedColormap", f"Build a smooth colormap between two colors: "
                                    f"`LinearSegmentedColormap.from_list({name!r}, {ends})`, and show `M` with it.",
         f"from matplotlib.colors import LinearSegmentedColormap\ncmap = LinearSegmentedColormap.from_list({name!r}, {ends})\n"
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap=cmap)", {"post": post("_imgs(fc.ax(0), 'cmap', 'N', 'sample')")}),
        ("from_list", f"Build `LinearSegmentedColormap.from_list('bwr3', ['blue', 'white', 'red'], N={k})` and return "
                      f"its {k} colors as hex strings (call it on `np.arange({k})`).",
         f"from matplotlib.colors import LinearSegmentedColormap\ncmap = LinearSegmentedColormap.from_list('bwr3', "
         f"['blue', 'white', 'red'], N={k})\nresult = [mpl.colors.to_hex(c) for c in cmap(np.arange({k}))]"),
        ("resampled", f"Show `M` on `fig, ax` with 'RdBu_r' cut down to {k} discrete colors (`.resampled({k})`) and "
                      f"add a colorbar: it shows {k} bands.",
         f"fig, ax = plt.subplots()\nim = ax.imshow(M, cmap=mpl.colormaps['RdBu_r'].resampled({k}))\nfig.colorbar(im, ax=ax)",
         {"post": post("_imgs(fc.ax(0), 'N', 'lut'), _cbs('cmap')")}),
        ("with_extremes", "Show `M` on `fig, ax` with `vmin=0.2, vmax=0.8` and a 'viridis' copy whose out-of-range "
                          "colors are black below and white above (`.with_extremes(under='k', over='w')`); add a "
                          "colorbar with `extend='both'` to show them.",
         "fig, ax = plt.subplots()\ncmap = mpl.colormaps['viridis'].with_extremes(under='k', over='w')\n"
         "im = ax.imshow(M, cmap=cmap, vmin=0.2, vmax=0.8)\nfig.colorbar(im, ax=ax, extend='both')",
         {"post": post("_imgs(fc.ax(0), 'extremes', 'clim'), _cbs('extend')")}),
        ("with_extremes", "Hide the small entries: show `np.ma.masked_less(M, 0.2)` on `fig, ax` with a 'magma' copy "
                          "that paints masked cells light gray (`.with_extremes(bad='lightgray')`).",
         "fig, ax = plt.subplots()\nax.imshow(np.ma.masked_less(M, 0.2), cmap=mpl.colormaps['magma'].with_extremes(bad='lightgray'))",
         {"post": post("_imgs(fc.ax(0), 'masked', 'extremes', 'cmap')")}),
        ("ListedColormap", f"Make a {k}-color `ListedColormap` from the first {k} colors of 'tab10' "
                           f"(`mpl.colormaps['tab10'].colors[:{k}]`) and return `[cmap.N, the hex of its last color]`.",
         f"from matplotlib.colors import ListedColormap\ncmap = ListedColormap(mpl.colormaps['tab10'].colors[:{k}])\n"
         f"result = [cmap.N, mpl.colors.to_hex(cmap({k - 1}))]"),
        ("reversed", f"Build `ListedColormap({pal})`, reverse it, and return the reversed colormap's colors as hex "
                     f"strings (call it on `np.arange(3)`).",
         f"from matplotlib.colors import ListedColormap\ncmap = ListedColormap({pal}).reversed()\n"
         f"result = [mpl.colors.to_hex(c) for c in cmap(np.arange(3))]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def color_conversions(rng):
    spec = rng.choice(["C2", "tab:red", "navy", "#33aa77", "0.5"])
    a = rng.choice([0.25, 0.5, 0.8])
    specs = rng.sample(["red", "C1", "#00ff00", "0.7", "tab:blue", "k", "w", "skyblue"], 4)
    bad = rng.choice(["redd", "C", "#12345", "tab:bluee"])
    hx = rng.choice(["#1f77b4", "#d62728", "#8c564b", "#17becf"])
    cm = rng.choice(["viridis", "plasma", "cividis"])
    v = rng.choice([0.1, 0.3, 0.6, 0.9])
    items = [
        ("to_rgba", f"Return `to_rgba({spec!r}, alpha={a})`: the color with its opacity replaced.",
         f"from matplotlib.colors import to_rgba\nresult = to_rgba({spec!r}, alpha={a})"),
        ("to_hex", f"Return the hex string **with** alpha (`#rrggbbaa`) of `to_rgba({spec!r}, alpha={a})` "
                   f"(`to_hex(..., keep_alpha=True)`).",
         f"from matplotlib.colors import to_hex, to_rgba\nresult = to_hex(to_rgba({spec!r}, alpha={a}), keep_alpha=True)"),
        ("to_rgba_array", f"Convert the list {specs} into one N × 4 RGBA array with `mpl.colors.to_rgba_array` and "
                          f"return it.", f"result = mpl.colors.to_rgba_array({specs})"),
        ("same_color", f"Different specs can name the same color. Return `[same_color('red', '#ff0000'), "
                       f"same_color('C0', {hx!r}), same_color('0.5', 'gray')]` (`mpl.colors.same_color`).",
         f"result = [mpl.colors.same_color('red', '#ff0000'), mpl.colors.same_color('C0', {hx!r}), "
         f"mpl.colors.same_color('0.5', 'gray')]"),
        ("is_color_like", f"Return which of {specs + [bad]} Matplotlib accepts as colors (`mpl.colors.is_color_like`), "
                          f"as a list of bools.", f"result = [mpl.colors.is_color_like(c) for c in {specs + [bad]}]"),
        ("to_rgba", f"Web colors are often written as 0–255 integers. Return the RGB of {hx!r} as three ints in "
                    f"0 … 255 (`round(c * 255)` of each channel of `to_rgba`).",
         f"from matplotlib.colors import to_rgba\nresult = [round(c * 255) for c in to_rgba({hx!r})[:3]]"),
        ("colormaps", f"Ask the colormap for bytes instead of floats: return `mpl.colormaps[{cm!r}]({v}, bytes=True)` "
                      f"as a list of ints.", f"result = [int(c) for c in mpl.colormaps[{cm!r}]({v}, bytes=True)]"),
        ("to_rgba", f"Return the relative luminance `0.2126 r + 0.7152 g + 0.0722 b` of each color in {specs} "
                    f"(from `to_rgba`), as a list of floats.",
         f"from matplotlib.colors import to_rgba\nresult = []\nfor c in {specs}:\n    r, g, b, _ = to_rgba(c)\n"
         f"    result.append(0.2126 * r + 0.7152 * g + 0.0722 * b)"),
    ]
    return catalog(rng, items, "", "")


# ================================================================= medium

def _confusion(rng):
    name, labs = T.labels(rng, min_classes=3)
    k = min(len(labs), rng.randint(3, 5))
    labs = list(labs[:k])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nM = g.integers(0, 9, ({k}, {k})) + np.diag(g.integers(20, 60, {k}))\n"
            f"names = {lit(labs)}")
    return code, f"`M` is a {k} × {k} confusion matrix of {name} classes (rows: true, columns: predicted) and `names` the class names", k


@template(M, "medium")
def heatmap_figures(rng):
    code, desc, k = _confusion(rng)
    sh = rng.choice([0.6, 0.7, 0.8])
    ls = rng.choice([7, 8, 9])
    base = "fig, ax = plt.subplots()\n"
    items = [
        ("imshow", "Show `M` on `fig, ax` with the colormap 'Blues', add a colorbar labelled 'count', put the class names "
                   "on both axes (`ax.set_xticks(range(len(names)), labels=names)` and the same for y), and label the "
                   "axes 'predicted' (x) and 'true' (y).",
         base + "im = ax.imshow(M, cmap='Blues')\nfig.colorbar(im, ax=ax, label='count')\n"
                "ax.set_xticks(range(len(names)), labels=names)\nax.set_yticks(range(len(names)), labels=names)\n"
                "ax.set(xlabel='predicted', ylabel='true')",
         {"post": post("_imgs(fc.ax(0), 'cmap', 'data'), _cbs('label'), fc.ticks(fc.ax(0), 'x'), fc.ticks(fc.ax(0), 'y'), "
                       "fc.labels(fc.ax(0))"), "require": ["set_xticks", "set_yticks"]}),
        (".text", "Show `M` on `fig, ax` with 'Blues' and write each count in its cell (`str(M[i, j])` at x = j, "
                  "y = i), centred both ways, **white** where the count is above `M.max() / 2` and black elsewhere, "
                  "so it stays readable on dark cells (rows first, then columns).",
         base + "ax.imshow(M, cmap='Blues')\nfor i in range(M.shape[0]):\n    for j in range(M.shape[1]):\n"
                "        ax.text(j, i, str(M[i, j]), ha='center', va='center', color='white' if M[i, j] > M.max() / 2 else 'black')",
         {"post": post("[(t.get_text(), fc.r(t.get_position(), 3), fc.hexcolor(t.get_color()), t.get_ha(), t.get_va()) "
                       "for t in fc.ax(0).texts]")}),
        ("vmin", "Normalize each row to recall (`R = M / M.sum(axis=1, keepdims=True)`) and show `R` on `fig, ax` with "
                 "'viridis' on a fixed 0 … 1 scale, with a colorbar whose ticks are 0, 0.5 and 1.",
         base + "R = M / M.sum(axis=1, keepdims=True)\nim = ax.imshow(R, cmap='viridis', vmin=0, vmax=1)\n"
                "cb = fig.colorbar(im, ax=ax)\ncb.set_ticks([0, 0.5, 1])",
         {"post": post("_imgs(fc.ax(0), 'data', 'clim', 'cmap'), _cbs('ticks')"), "require": ["vmin", "set_ticks"]}),
        ("LogNorm", "Counts span orders of magnitude: show `M + 1` on `fig, ax` with 'magma' and "
                    "`norm=LogNorm(vmin=1, vmax=(M + 1).max())`, plus a colorbar labelled 'count + 1'.",
         "from matplotlib.colors import LogNorm\n" + base + "im = ax.imshow(M + 1, cmap='magma', norm=LogNorm(vmin=1, "
         "vmax=(M + 1).max()))\nfig.colorbar(im, ax=ax, label='count + 1')",
         {"post": post("_imgs(fc.ax(0), 'norm', 'clim', 'cmap'), _cbs('label', 'norm')")}),
        ("BoundaryNorm", "Show what each cell is: `cat = np.where(M == 0, 0, np.where(np.eye(len(M), dtype=bool), 2, 1))` "
                         "(0 = empty, 1 = error, 2 = correct). Draw it with `ListedColormap(['white', 'tab:red', "
                         "'tab:green'])` and `BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)`, and a colorbar with ticks 0, 1, "
                         "2 labelled 'empty', 'error', 'correct'.",
         "from matplotlib.colors import BoundaryNorm, ListedColormap\n" + base +
         "cat = np.where(M == 0, 0, np.where(np.eye(len(M), dtype=bool), 2, 1))\n"
         "im = ax.imshow(cat, cmap=ListedColormap(['white', 'tab:red', 'tab:green']), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))\n"
         "cb = fig.colorbar(im, ax=ax)\ncb.set_ticks([0, 1, 2], labels=['empty', 'error', 'correct'])",
         {"post": post("_imgs(fc.ax(0), 'data', 'lut', 'bounds'), _cbs('ticks', 'ticklabels')")}),
        ("TwoSlopeNorm", "Which confusions are one-sided? Show `D = M - M.T` on `fig, ax` with 'RdBu_r' and "
                         "`TwoSlopeNorm(vcenter=0)` so equal swaps stay white, with a colorbar.",
         "from matplotlib.colors import TwoSlopeNorm\n" + base + "D = M - M.T\n"
         "im = ax.imshow(D, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))\nfig.colorbar(im, ax=ax)",
         {"post": post("_imgs(fc.ax(0), 'data', 'norm', 'vcenter', 'cmap'), len(_cbs())")}),
        ("with_extremes", "Hide the (large) diagonal so the errors get the whole color range: show "
                          "`np.ma.masked_where(np.eye(len(M), dtype=bool), M)` on `fig, ax` with a 'Reds' copy that "
                          "paints masked cells light gray (`.with_extremes(bad='lightgray')`), plus a colorbar.",
         base + "off = np.ma.masked_where(np.eye(len(M), dtype=bool), M)\n"
                "im = ax.imshow(off, cmap=mpl.colormaps['Reds'].with_extremes(bad='lightgray'))\nfig.colorbar(im, ax=ax)",
         {"post": post("_imgs(fc.ax(0), 'masked', 'extremes', 'clim'), len(_cbs())")}),
        ("colorbar", "Create `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))`: the counts `M` on `ax1` with 'Blues' "
                     "and a colorbar labelled 'count'; the recall matrix (`M / M.sum(axis=1, keepdims=True)`) on `ax2` "
                     "with 'Greens' and a colorbar labelled 'rate'.",
         "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))\nim1 = ax1.imshow(M, cmap='Blues')\n"
         "fig.colorbar(im1, ax=ax1, label='count')\nim2 = ax2.imshow(M / M.sum(axis=1, keepdims=True), cmap='Greens')\n"
         "fig.colorbar(im2, ax=ax2, label='rate')",
         {"post": post("[_imgs(a, 'cmap', 'data') for a in fc.axes()], _cbs('of', 'label')")}),
        ("shrink", f"Show `M` on `fig, ax` with 'Purples' and a colorbar {int(sh * 100)}% as tall as the Axes, labelled "
                   f"'count', with {ls} pt tick labels.",
         base + f"im = ax.imshow(M, cmap='Purples')\ncb = fig.colorbar(im, ax=ax, shrink={sh}, label='count')\n"
                f"cb.ax.tick_params(labelsize={ls})", {"post": post("_cbs('shrink', 'label', 'labelsize')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def lr_sweep_colors(rng):
    code, desc, k = _sweep(rng, k=rng.randint(6, 8))
    cm = rng.choice(["viridis", "plasma", "cividis", "magma"])
    COL = "_lines(fc.ax(0), 'color')"
    items = [
        ("colormaps", f"Plot every run on `fig, ax`, colored along the sequential colormap {cm!r} in the order of "
                      f"`lrs`: sample it evenly from 0 to 1. Label the axes 'step' and 'loss'.",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}](np.linspace(0, 1, len(lrs)))\n"
         f"for r, c in zip(runs, colors):\n    ax.plot(steps, r, color=c)\nax.set(xlabel='step', ylabel='loss')",
         {"post": post(f"{COL}, fc.labels(fc.ax(0))")}),
        ("ScalarMappable", f"Color each run by its learning rate on a **log** scale (`norm = LogNorm(lrs.min(), "
                           f"lrs.max())`, `cmap = mpl.colormaps[{cm!r}]`, color `cmap(norm(lr))`) and explain the "
                           f"colors with a colorbar: there is no image, so build one from "
                           f"`mpl.cm.ScalarMappable(norm=norm, cmap=cmap)`, labelled 'learning rate'.",
         f"from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\nnorm = LogNorm(lrs.min(), lrs.max())\n"
         f"cmap = mpl.colormaps[{cm!r}]\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, color=cmap(norm(lr)))\n"
         f"fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='learning rate')",
         {"post": post(f"{COL}, _cbs('label', 'norm', 'cmap', 'clim')")}),
        ("resampled", f"Give each run its own discrete color: `cmap = mpl.colormaps[{cm!r}].resampled(len(lrs))` and "
                      f"run i gets `cmap(i)` (an integer picks entry i).",
         f"fig, ax = plt.subplots()\ncmap = mpl.colormaps[{cm!r}].resampled(len(lrs))\n"
         f"for i, r in enumerate(runs):\n    ax.plot(steps, r, color=cmap(i))", {"post": post(COL)}),
        ("set_prop_cycle", f"Make the colormap the Axes' color cycle: `ax.set_prop_cycle(color=...)` with "
                           f"`len(lrs)` colors sampled evenly from {cm!r}; then plot every run **without** `color=`.",
         f"fig, ax = plt.subplots()\nax.set_prop_cycle(color=mpl.colormaps[{cm!r}](np.linspace(0, 1, len(lrs))))\n"
         f"for r in runs:\n    ax.plot(steps, r)", {"post": post(COL)}),
        ("argsort", "Shade by quality: rank the runs by final loss (rank 0 = lowest) and color run i with "
                    "`mpl.colormaps['Blues'](1 - 0.7 * rank[i] / (len(runs) - 1))`, so the best run is the darkest.",
         "fig, ax = plt.subplots()\nrank = np.empty(len(runs), dtype=int)\nrank[np.argsort(runs[:, -1])] = np.arange(len(runs))\n"
         "for r, q in zip(runs, rank):\n    ax.plot(steps, r, color=mpl.colormaps['Blues'](1 - 0.7 * q / (len(runs) - 1)))",
         {"post": post(COL)}),
        ("legend", f"Color the runs along {cm!r} (evenly sampled, in `lrs` order), label them `f'{{lr:.0e}}'` and add a "
                   f"legend titled 'lr' in 2 columns.",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}](np.linspace(0, 1, len(lrs)))\n"
         f"for lr, r, c in zip(lrs, runs, colors):\n    ax.plot(steps, r, color=c, label=f'{{lr:.0e}}')\n"
         f"ax.legend(title='lr', ncols=2)",
         {"post": post(f"{COL}, [t.get_text() for t in fc.ax(0).get_legend().get_texts()], "
                       f"fc.ax(0).get_legend().get_title().get_text()")}),
        ("alpha", "Highlight the winner: plot every run in 'gray' with `alpha=0.4`, then the run with the lowest final "
                  "loss again on top in 'C3' with line width 2.",
         "fig, ax = plt.subplots()\nfor r in runs:\n    ax.plot(steps, r, color='gray', alpha=0.4)\n"
         "ax.plot(steps, runs[runs[:, -1].argmin()], color='C3', lw=2)",
         {"post": post("_lines(fc.ax(0), 'color', 'alpha', 'lw', 'y')")}),
        ("LogNorm", f"Scatter the final loss of each run against its lr on a log x-axis, colored by lr: "
                    f"`ax.scatter(lrs, runs[:, -1], c=lrs, cmap={cm!r}, norm=LogNorm())`, with a colorbar labelled "
                    f"'lr'.",
         f"from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\nsc = ax.scatter(lrs, runs[:, -1], c=lrs, "
         f"cmap={cm!r}, norm=LogNorm())\nax.set_xscale('log')\nfig.colorbar(sc, ax=ax, label='lr')",
         {"post": post("_sc(fc.ax(0), 'norm', 'cmap', 'data', 'clim'), fc.scales(fc.ax(0)), _cbs('label')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def shared_colorbars(rng):
    s = seeded(rng)
    n = rng.randint(5, 8)
    lab = rng.choice(["activation", "attention", "score", "value"])
    lo, hi = rng.choice([(5, 95), (2, 98), (10, 90)])
    code = (f"g = np.random.default_rng({s})\nmats = np.round(g.random((4, {n}, {n})) * np.array([1, 2, 3, 4])[:, None, None], 3)")
    CL = "[_imgs(a, 'clim', 'data') for a in fc.axes()]"
    items = [
        ("colorbar", "Show the four matrices on a 2 × 2 grid (`fig, axs = plt.subplots(2, 2)`, `mats[i]` on "
                     "`axs.flat[i]`), all on the **same** color scale (`vmin=mats.min()`, `vmax=mats.max()`), with "
                     "**one** colorbar for the whole grid: `fig.colorbar(im, ax=axs)`.",
         "fig, axs = plt.subplots(2, 2)\nfor a, m in zip(axs.flat, mats):\n    im = a.imshow(m, vmin=mats.min(), vmax=mats.max())\n"
         "fig.colorbar(im, ax=axs)", {"post": post(f"{CL}, len(_cbs())")}),
        ("Normalize", "Share one norm object instead of repeating vmin/vmax: `norm = Normalize(mats.min(), mats.max())`, "
                      "pass `norm=norm` to every `imshow` on a 1 × 4 row, and add one colorbar for all panels.",
         "from matplotlib.colors import Normalize\nnorm = Normalize(mats.min(), mats.max())\n"
         "fig, axs = plt.subplots(1, 4)\nfor a, m in zip(axs, mats):\n    im = a.imshow(m, norm=norm)\nfig.colorbar(im, ax=axs)",
         {"post": post(f"{CL}, _cbs('clim')")}),
        ("orientation", "Show `mats[0]`, `mats[1]`, `mats[2]` on a 1 × 3 row with a common scale (the min and max of "
                        "those three), and one **horizontal** colorbar under the row, 60% as wide (`shrink=0.6`).",
         "fig, axs = plt.subplots(1, 3)\nlo, hi = mats[:3].min(), mats[:3].max()\nfor a, m in zip(axs, mats[:3]):\n"
         "    im = a.imshow(m, vmin=lo, vmax=hi)\nfig.colorbar(im, ax=axs, orientation='horizontal', shrink=0.6)",
         {"post": post(f"{CL}, _cbs('orientation', 'clim')")}),
        ("colorbar", "The opposite case: show `mats[1]`, `mats[2]`, `mats[3]` on a 1 × 3 row, each with its **own** "
                     "scale and its own colorbar.",
         "fig, axs = plt.subplots(1, 3)\nfor a, m in zip(axs, mats[1:]):\n    fig.colorbar(a.imshow(m), ax=a)",
         {"post": post(f"{CL}, _cbs('of', 'clim')")}),
        ("extend", f"Outliers can wash out a shared scale. Show the four matrices on a 2 × 2 grid sharing the range "
                   f"from the {lo}th to the {hi}th percentile of all values (`np.percentile(mats, [{lo}, {hi}])`), with "
                   f"one colorbar whose ends show the clipping (`extend='both'`).",
         f"lo, hi = np.percentile(mats, [{lo}, {hi}])\nfig, axs = plt.subplots(2, 2)\nfor a, m in zip(axs.flat, mats):\n"
         f"    im = a.imshow(m, vmin=lo, vmax=hi)\nfig.colorbar(im, ax=axs, extend='both')",
         {"post": post(f"{CL}, _cbs('extend', 'clim')")}),
        ("layout", f"With `fig, axs = plt.subplots(2, 2, layout='constrained')`, show the four matrices on one common "
                   f"scale (overall min / max) with a single colorbar for the grid labelled {lab!r}.",
         f"fig, axs = plt.subplots(2, 2, layout='constrained')\nfor a, m in zip(axs.flat, mats):\n"
         f"    im = a.imshow(m, vmin=mats.min(), vmax=mats.max())\nfig.colorbar(im, ax=axs, label={lab!r})",
         {"post": post(f"{CL}, _cbs('label'), type(fc.fig().get_layout_engine()).__name__")}),
        ("CenteredNorm", "Turn the matrices into updates centred at 0: `upd = mats - mats.mean(axis=(1, 2), "
                         "keepdims=True)`. Show `upd[0]` and `upd[1]` side by side with 'RdBu_r' and **one** shared "
                         "`CenteredNorm(halfrange=np.abs(upd[:2]).max())`, plus one colorbar for both.",
         "from matplotlib.colors import CenteredNorm\nupd = mats - mats.mean(axis=(1, 2), keepdims=True)\n"
         "norm = CenteredNorm(halfrange=np.abs(upd[:2]).max())\nfig, axs = plt.subplots(1, 2)\n"
         "for a, u in zip(axs, upd[:2]):\n    im = a.imshow(u, cmap='RdBu_r', norm=norm)\nfig.colorbar(im, ax=axs)",
         {"post": post(f"[_imgs(a, 'clim', 'norm', 'cmap') for a in fc.axes()], len(_cbs())")}),
        ("set_clim", "Someone drew `ims = [a.imshow(m) for a, m in zip(axs.flat, mats)]` on a 2 × 2 grid, each panel "
                     "with its own scale. Without redrawing, make them comparable: call `im.set_clim(mats.min(), "
                     "mats.max())` on each, then add one colorbar for the grid.",
         "fig, axs = plt.subplots(2, 2)\nims = [a.imshow(m) for a, m in zip(axs.flat, mats)]\nfor im in ims:\n"
         "    im.set_clim(mats.min(), mats.max())\nfig.colorbar(ims[0], ax=axs)", {"post": post(f"{CL}, _cbs('clim')")}),
    ]
    return catalog(rng, items, code, f"`mats` holds four {n} × {n} matrices; the i-th has values in 0 … i + 1.")


@template(M, "medium")
def scatter_color_maps(rng):
    code, desc, k, names = classes(rng)
    cm = rng.choice(["viridis", "plasma", "cividis"])
    a_ = rng.choice([0.5, 0.6, 0.7])
    base = "fig, ax = plt.subplots()\n"
    items = [
        ("scatter", f"Scatter `pts` on `fig, ax` colored by each point's distance from the origin "
                    f"(`np.linalg.norm(pts, axis=1)`) with {cm!r}, and add a colorbar labelled 'distance'.",
         base + f"d = np.linalg.norm(pts, axis=1)\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=d, cmap={cm!r})\n"
                f"fig.colorbar(sc, ax=ax, label='distance')", {"post": post("_sc(fc.ax(0), 'cmap', 'data'), _cbs('label')")}),
        ("BoundaryNorm", "Color by class with one flat color each: `cmap = ListedColormap(mpl.colormaps['tab10'].colors"
                         "[:len(class_names)])` and `norm = BoundaryNorm(np.arange(len(class_names) + 1) - 0.5, "
                         "len(class_names))`; scatter with `c=labels` and add a colorbar with a tick per class "
                         "(`range(len(class_names))`) labelled by `class_names`.",
         "from matplotlib.colors import BoundaryNorm, ListedColormap\n" + base +
         "k = len(class_names)\ncmap = ListedColormap(mpl.colormaps['tab10'].colors[:k])\n"
         "norm = BoundaryNorm(np.arange(k + 1) - 0.5, k)\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap=cmap, norm=norm)\n"
         "cb = fig.colorbar(sc, ax=ax)\ncb.set_ticks(range(k), labels=class_names)",
         {"post": post("_sc(fc.ax(0), 'lut', 'bounds', 'data'), _cbs('ticks', 'ticklabels')")}),
        ("CenteredNorm", "Fit a line `a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)` and scatter the points colored by "
                         "their residual `pts[:, 1] - (a * pts[:, 0] + b)` with 'RdBu_r' and `CenteredNorm()` (0 = white), "
                         "plus a colorbar labelled 'residual'.",
         "from matplotlib.colors import CenteredNorm\n" + base + "a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)\n"
         "res = pts[:, 1] - (a * pts[:, 0] + b)\nsc = ax.scatter(pts[:, 0], pts[:, 1], c=res, cmap='RdBu_r', norm=CenteredNorm())\n"
         "fig.colorbar(sc, ax=ax, label='residual')",
         {"post": post("_sc(fc.ax(0), 'norm', 'vcenter', 'cmap', 'data'), _cbs('label')")}),
        ("LogNorm", f"Scatter `pts` colored by `np.exp(pts[:, 0])` (spanning orders of magnitude) with {cm!r} on a log "
                    f"color scale (`norm=LogNorm()`), plus a colorbar.",
         "from matplotlib.colors import LogNorm\n" + base +
         f"sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.exp(pts[:, 0]), cmap={cm!r}, norm=LogNorm())\nfig.colorbar(sc, ax=ax)",
         {"post": post("_sc(fc.ax(0), 'norm', 'clim', 'cmap'), len(_cbs())")}),
        ("vmin", "With the qualitative 'tab10' (10 colors), `c=labels` alone stretches the classes over the whole map. "
                 "Scatter with `c=labels, cmap='tab10', vmin=0, vmax=9` so class i gets the i-th tab10 color.",
         base + "ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10', vmin=0, vmax=9)",
         {"post": post("_sc(fc.ax(0), 'cmap', 'clim', 'data')"), "require": ["vmin", "vmax"]}),
        ("edgecolors", f"Scatter `pts` in one call with a color per point `c=[f'C{{l}}' for l in labels]`, black edges "
                       f"(`edgecolors='k'`) and `alpha={a_}`.",
         base + f"sc = ax.scatter(pts[:, 0], pts[:, 1], c=[f'C{{l}}' for l in labels], edgecolors='k', alpha={a_})",
         {"post": "_s = fc.ax().collections[0]\n([fc.hexcolor(c) for c in _s.get_facecolors()], fc.hexcolor(_s.get_edgecolors()), _s.get_alpha())"}),
        ("to_rgba_array", "Build the colors yourself: start from the tab10 color of each point's class "
                          "(`mpl.colors.to_rgba_array([f'C{l}' for l in labels])`), set the alpha column to the "
                          "point's confidence `conf`, and scatter with `c=rgba`.",
         base + "rgba = mpl.colors.to_rgba_array([f'C{l}' for l in labels])\nrgba[:, 3] = conf\n"
                "ax.scatter(pts[:, 0], pts[:, 1], c=rgba)",
         {"post": "fc.r(fc.ax().collections[0].get_facecolors(), 3)",
          "setup": "conf = np.round(np.random.default_rng(len(pts)).uniform(0.2, 1, len(pts)), 2)"}),
        ("Normalize", f"Create `fig, (ax1, ax2) = plt.subplots(1, 2)`: class-0 points on `ax1`, all other points on "
                      f"`ax2`, both colored by their x coordinate with {cm!r} and **one** shared `Normalize(pts[:, 0].min(), "
                      f"pts[:, 0].max())`, plus one colorbar for both Axes.",
         f"from matplotlib.colors import Normalize\nfig, (ax1, ax2) = plt.subplots(1, 2)\n"
         f"norm = Normalize(pts[:, 0].min(), pts[:, 0].max())\nfor a, m in ((ax1, labels == 0), (ax2, labels != 0)):\n"
         f"    sc = a.scatter(pts[m, 0], pts[m, 1], c=pts[m, 0], cmap={cm!r}, norm=norm)\nfig.colorbar(sc, ax=[ax1, ax2])",
         {"post": post("[_sc(a, 'clim', 'cmap', 'data') for a in fc.axes()], len(_cbs())")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def palette_checks(rng):
    cms = rng.sample(["viridis", "jet", "gray", "hsv", "plasma", "rainbow", "cividis", "coolwarm"], 4)
    bgs = rng.sample(["white", "black", "navy", "yellow", "C0", "C1", "lightgray", "darkred", "#ffcc00"], 5)
    target = rng.choice([(0.9, 0.2, 0.1), (0.2, 0.6, 0.3), (0.1, 0.1, 0.8), (0.95, 0.9, 0.2)])
    cm = rng.choice(["viridis", "plasma", "RdBu_r", "coolwarm", "magma"])
    k = rng.choice([3, 5, 7])
    rgba = rng.choice([(0.2, 0.4, 0.8, 0.5), (1, 0, 0, 0.25), (0, 0.5, 0, 0.8)])
    thr = rng.choice([0.5, 0.6, 0.7])
    LUM = "def lum(c):\n    r, g, b, _ = mpl.colors.to_rgba(c)\n    return 0.2126 * r + 0.7152 * g + 0.0722 * b\n"
    items = [
        ("colormaps", f"A colormap reads well in grayscale if its luminance (`0.2126 r + 0.7152 g + 0.0722 b`) rises "
                      f"steadily. For each of {cms}, sample 32 colors (`np.linspace(0, 1, 32)`) and return whether the "
                      f"luminance is strictly increasing, as a list of bools.",
         f"result = []\nfor name in {cms}:\n    c = mpl.colormaps[name](np.linspace(0, 1, 32))\n"
         f"    lum = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]\n    result.append(bool(np.all(np.diff(lum) > 0)))"),
        ("to_rgba", f"Pick a readable text color for each background in {bgs}: 'black' if its luminance (`0.2126 r + "
                    f"0.7152 g + 0.0722 b` from `to_rgba`) is above 0.5, else 'white'. Return the list.",
         "from matplotlib.colors import to_rgba\n" + LUM.replace("mpl.colors.", "") +
         f"result = ['black' if lum(c) > 0.5 else 'white' for c in {bgs}]"),
        ("resampled", f"Two ways to get {k} colors from {cm!r}: `.resampled({k})` called on `np.arange({k})`, and "
                      f"sampling the full map at `np.linspace(0, 1, {k})`. Return both as lists of hex strings "
                      f"(`[resampled, sampled]`).",
         f"cmap = mpl.colormaps[{cm!r}]\na = [mpl.colors.to_hex(c) for c in cmap.resampled({k})(np.arange({k}))]\n"
         f"b = [mpl.colors.to_hex(c) for c in cmap(np.linspace(0, 1, {k}))]\nresult = [a, b]"),
        ("get_under", f"Values below / above the norm's range get the colormap's *under* / *over* colors (by default "
                      f"its end colors). Return the hex of `get_under()` and `get_over()` for {cm!r}, then for its copy "
                      f"`.with_extremes(under='k', over='w')`: a list of 4 strings.",
         f"c1 = mpl.colormaps[{cm!r}]\nc2 = c1.with_extremes(under='k', over='w')\n"
         f"result = [mpl.colors.to_hex(c) for c in (c1.get_under(), c1.get_over(), c2.get_under(), c2.get_over())]"),
        ("argmin", f"Find where a color sits on a colormap: sample {cm!r} at 256 points (`np.linspace(0, 1, 256)`) and "
                   f"return the index of the sample closest (Euclidean distance in RGB) to {target}.",
         f"c = mpl.colormaps[{cm!r}](np.linspace(0, 1, 256))[:, :3]\nresult = int(np.argmin(np.linalg.norm(c - "
         f"np.array({target}), axis=1)))"),
        ("to_hex", f"A semi-transparent color drawn on white shows as `alpha * c + (1 - alpha) * 1` per channel. Return "
                   f"the hex of what {rgba} looks like on a white background.",
         f"r, g, b, a = {rgba}\nresult = mpl.colors.to_hex([a * c + (1 - a) for c in (r, g, b)])"),
        ("reversed", f"Check that the reversed map is the map read backwards: with `x = (np.arange(256) + 0.5) / 256`, "
                     f"return whether `mpl.colormaps[{cm!r}].reversed()(x)` equals `mpl.colormaps[{cm!r}](1 - x)` "
                     f"(`np.allclose`).",
         f"x = (np.arange(256) + 0.5) / 256\ncmap = mpl.colormaps[{cm!r}]\nresult = bool(np.allclose(cmap.reversed()(x), cmap(1 - x)))"),
        (".colors", f"'tab20' is a `ListedColormap` with 20 colors (`.colors`). Return how many of them have a "
                           f"luminance (`0.2126 r + 0.7152 g + 0.0722 b`) above {thr}.",
         LUM + f"result = sum(lum(c) > {thr} for c in mpl.colormaps['tab20'].colors)"),
    ]
    return catalog(rng, items, "", "")


# ================================================================= hard

@template(M, "hard")
def color_helpers(rng):
    code, desc, n = _mats(rng)
    thr = rng.choice([0.4, 0.5, 0.6])
    items = [
        ("colormaps", "Write `colors_for(n, cmap='viridis')` returning n colors as hex strings, sampled evenly from the "
                      "colormap from 0 to 1 — but a single color (n == 1) is the middle one (0.5).",
         "def colors_for(n, cmap='viridis'):\n    xs = [0.5] if n == 1 else np.linspace(0, 1, n)\n"
         "    return [mpl.colors.to_hex(c) for c in mpl.colormaps[cmap](xs)]",
         {"post": "colors_for(4), colors_for(1, 'plasma'), colors_for(3, 'RdBu_r')"}),
        ("to_rgba", f"Write `text_color_for(color)` returning 'black' when the color's luminance (`0.2126 r + 0.7152 g "
                    f"+ 0.0722 b` of `to_rgba(color)`) is above {thr}, else 'white' — for labels drawn on colored cells.",
         f"from matplotlib.colors import to_rgba\n\ndef text_color_for(color):\n    r, g, b, _ = to_rgba(color)\n"
         f"    return 'black' if 0.2126 * r + 0.7152 * g + 0.0722 * b > {thr} else 'white'",
         {"post": "[text_color_for(c) for c in ('white', 'black', 'navy', 'yellow', 'C1', (0.5, 0.5, 0.5), '#20a0ff', '0.45')]"}),
        ("colorbar", "Write `heatmap(ax, M, cmap='viridis', label='')`: show `M` on `ax` with `cmap`, add a colorbar for "
                     "that Axes labelled `label`, and return `(im, cb)`.",
         "def heatmap(ax, M, cmap='viridis', label=''):\n    im = ax.imshow(M, cmap=cmap)\n"
         "    cb = ax.figure.colorbar(im, ax=ax, label=label)\n    return im, cb",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_r = heatmap(_a, data)\nheatmap(_b, signed, 'RdBu_r', label='z')\n"
                       "([_imgs(a, 'cmap', 'data') for a in (_a, _b)], _cbs('of', 'label', f=_f), "
                       "[type(v).__name__ for v in _r])")}),
        ("TwoSlopeNorm", "Write `diverging_norm(data, center=0.0)` returning a `TwoSlopeNorm` centred at `center` with "
                         "**symmetric** limits: `center ± r` where r is the largest distance of any value from "
                         "`center`, so equal distances get equally strong colors.",
         "from matplotlib.colors import TwoSlopeNorm\n\ndef diverging_norm(data, center=0.0):\n"
         "    r = np.abs(np.asarray(data) - center).max()\n    return TwoSlopeNorm(vcenter=center, vmin=center - r, vmax=center + r)",
         {"post": "_a = diverging_norm(signed)\n_b = diverging_norm(data, center=1.0)\n"
                  "[(type(x).__name__, fc.r([x.vmin, x.vcenter, x.vmax], 6)) for x in (_a, _b)], fc.r(_a([0, signed.max()]), 6)"}),
        ("resampled", "Write `discrete_cmap(name, n)` returning the colormap `name` cut down to `n` colors, and "
                      "check it: the post calls it and reads `.N` and the colors.",
         "def discrete_cmap(name, n):\n    return mpl.colormaps[name].resampled(n)",
         {"post": "[(c.N, [mpl.colors.to_hex(x) for x in c(np.arange(c.N))]) for c in (discrete_cmap('viridis', 4), "
                  "discrete_cmap('RdBu_r', 3))]"}),
        ("ScalarMappable", "Write `color_by_value(ax, x, ys, values, cmap='plasma', log=False)`: draw each row of `ys` "
                           "against `x` in the color of its value (norm: `LogNorm` of the values' min / max when `log`, "
                           "else `Normalize`), add a colorbar for that norm and colormap (via `mpl.cm.ScalarMappable`), "
                           "and return the colorbar.",
         "from matplotlib.colors import LogNorm, Normalize\n\ndef color_by_value(ax, x, ys, values, cmap='plasma', log=False):\n"
         "    norm = (LogNorm if log else Normalize)(min(values), max(values))\n    cm = mpl.colormaps[cmap]\n"
         "    for y, v in zip(ys, values):\n        ax.plot(x, y, color=cm(norm(v)))\n"
         "    return ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax)",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_r = color_by_value(_a, np.arange(len(data)), data, data[:, 0])\n"
                       "color_by_value(_b, np.arange(len(data)), data[:3], [0.001, 0.01, 0.1], cmap='viridis', log=True)\n"
                       "[_lines(a, 'color') for a in (_a, _b)], _cbs('norm', 'cmap', 'clim', f=_f), type(_r).__name__")}),
        (".text", "Write `annotate_heatmap(ax, im, fmt='{:.2f}')`: write every cell value of the image `im` (its "
                  "array, `im.get_array()`) at its cell centre (x = column, y = row), centred, in black when the "
                  "cell's own color (`im.cmap(im.norm(v))`) has luminance `0.2126 r + 0.7152 g + 0.0722 b` above "
                  "0.5, else white. Return how many texts you wrote.",
         "def annotate_heatmap(ax, im, fmt='{:.2f}'):\n    A = im.get_array()\n    for i in range(A.shape[0]):\n"
         "        for j in range(A.shape[1]):\n            r, g, b, _ = im.cmap(im.norm(A[i, j]))\n"
         "            color = 'black' if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else 'white'\n"
         "            ax.text(j, i, fmt.format(A[i, j]), ha='center', va='center', color=color)\n    return A.size",
         {"post": "_f, _a = plt.subplots()\n_im = _a.imshow(data[:3, :4], cmap='viridis')\n_k = annotate_heatmap(_a, _im, fmt='{:.1f}')\n"
                  "_k, [(t.get_text(), fc.r(t.get_position(), 3), fc.hexcolor(t.get_color()), t.get_ha()) for t in _a.texts]"}),
        ("set_clim", "Write `shared_clim(images)`: put every image of the list on one common color scale (the min and "
                     "max over all their arrays) with `set_clim`, and return `(lo, hi)`.",
         "def shared_clim(images):\n    lo = min(im.get_array().min() for im in images)\n"
         "    hi = max(im.get_array().max() for im in images)\n    for im in images:\n        im.set_clim(lo, hi)\n    return lo, hi",
         {"post": post("_f, (_a, _b) = plt.subplots(1, 2)\n_ims = [_a.imshow(data), _b.imshow(signed)]\n"
                       "_r = shared_clim(_ims)\nfc.r(_r, 6), [_img(i, 'clim') for i in _ims]")}),
    ]
    return catalog(rng, items, code, f"{desc}. Write the function; it is checked by calling it.")


@template(M, "hard")
def fix_color_bugs(rng):
    code, desc, k = _sweep(rng)
    s = seeded(rng)
    cm = rng.choice(["viridis", "plasma", "magma"])
    extra = (f"g2 = np.random.default_rng({s})\ndata = np.round(g2.lognormal(size=(6, 6)), 3)\n"
             f"counts = g2.poisson(3, (6, 6)) * g2.integers(0, 2, (6, 6))")
    items = [
        ("colorbar", "This crashes with *RuntimeError: No mappable was found to use for colorbar creation*: "
                     "`plt.colorbar()` only knows images made by **pyplot**, and this one came from `ax.imshow`.\n"
                     "```python\nfig, ax = plt.subplots()\nax.imshow(data)\nplt.colorbar()\n```\nFix it the explicit "
                     "way: keep the image and pass it to `fig.colorbar`.",
         "fig, ax = plt.subplots()\nim = ax.imshow(data)\nfig.colorbar(im, ax=ax)",
         {"post": post("_imgs(fc.ax(0), 'data'), _cbs('of')"), "setup": extra}),
        ("vmin", "The single colorbar lies: each panel picks its own color scale, so it only matches the last one.\n"
                 "```python\nfig, axs = plt.subplots(1, 2)\nfor a, m in zip(axs, (data, 2 * data)):\n"
                 "    im = a.imshow(m)\nfig.colorbar(im, ax=axs)\n```\nPut both panels on the common scale 0 … "
                 "`(2 * data).max()` (keep one colorbar).",
         "fig, axs = plt.subplots(1, 2)\nfor a, m in zip(axs, (data, 2 * data)):\n    im = a.imshow(m, vmin=0, vmax=(2 * data).max())\n"
         "fig.colorbar(im, ax=axs)", {"post": post("[_imgs(a, 'clim', 'data') for a in fc.axes()], len(_cbs())"), "setup": extra}),
        ("TwoSlopeNorm", "`data` are ratios (1 = no change), but every cell comes out red: all values are above the "
                         "center 0, so the blue half of the colormap is never used.\n```python\n"
                         "from matplotlib.colors import TwoSlopeNorm\nfig, ax = plt.subplots()\n"
                         "ax.imshow(data, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))\n```\nCenter the colors on 1.",
         "from matplotlib.colors import TwoSlopeNorm\nfig, ax = plt.subplots()\nax.imshow(data, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=1))",
         {"post": post("_imgs(fc.ax(0), 'norm', 'vcenter', 'cmap', 'data')"), "setup": extra}),
        ("cmap", f"This crashes with *ValueError: '{cm}_reversed' is not a valid value for cmap*:\n```python\n"
                 f"fig, ax = plt.subplots()\nax.imshow(data, cmap='{cm}_reversed')\n```\nUse the real name of the "
                 f"reversed colormap.",
         f"fig, ax = plt.subplots()\nax.imshow(data, cmap='{cm}_r')", {"post": post("_imgs(fc.ax(0), 'cmap')"), "setup": extra}),
        ("color", "This crashes with *ValueError: (230, 159, 0) is not a valid value for color*: Matplotlib wants RGB "
                  "floats in 0 … 1, not 0 … 255.\n```python\nfig, ax = plt.subplots()\n"
                  "ax.plot(steps, runs[0], color=(230, 159, 0))\n```\nFix the color (same orange).",
         "fig, ax = plt.subplots()\nax.plot(steps, runs[0], color=(230 / 255, 159 / 255, 0))",
         {"post": post("_lines(fc.ax(0), 'color', 'y')")}),
        ("colormaps", f"This crashes with *Line2D.set() got an unexpected keyword argument 'cmap'*: lines don't take a "
                      f"colormap.\n```python\nfig, ax = plt.subplots()\nax.plot(steps, runs.T, cmap='{cm}')\n```\nDraw "
                      f"each run with its own color sampled evenly from {cm!r} (run i gets the i-th of `len(runs)` "
                      f"samples from 0 to 1).",
         f"fig, ax = plt.subplots()\ncolors = mpl.colormaps[{cm!r}](np.linspace(0, 1, len(runs)))\n"
         f"for r, c in zip(runs, colors):\n    ax.plot(steps, r, color=c)", {"post": post("_lines(fc.ax(0), 'color', 'y')")}),
        ("BoundaryNorm", "This crashes with *There are 4 color bins including extensions, but ncolors = 2*:\n"
                         "```python\nfrom matplotlib.colors import BoundaryNorm\nfig, ax = plt.subplots()\n"
                         "ax.imshow(data, cmap=mpl.colormaps['viridis'].resampled(2), norm=BoundaryNorm([0, 1, 2, 3, 4], 2))\n```\n"
                         "Five boundaries make four bins: use four colors (in the colormap and in the norm).",
         "from matplotlib.colors import BoundaryNorm\nfig, ax = plt.subplots()\n"
         "ax.imshow(data, cmap=mpl.colormaps['viridis'].resampled(4), norm=BoundaryNorm([0, 1, 2, 3, 4], 4))",
         {"post": post("_imgs(fc.ax(0), 'N', 'bounds', 'lut')"), "setup": extra}),
        ("LogNorm", "On a log color scale the empty cells (count 0) vanish: log(0) is undefined, so they're drawn as "
                    "'bad' (blank).\n```python\nfrom matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\n"
                    "ax.imshow(counts, norm=LogNorm())\n```\nShow `counts + 1` instead (with `LogNorm()`) so every "
                    "cell gets a color.",
         "from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\nax.imshow(counts + 1, norm=LogNorm())",
         {"post": post("_imgs(fc.ax(0), 'norm', 'clim', 'data'), int(np.ma.getmaskarray(fc.ax(0).get_images()[0].norm("
                       "fc.ax(0).get_images()[0].get_array())).sum())"), "setup": extra}),
        ("colorbar", "This crashes with *'Axes' object has no attribute 'colorbar'*:\n```python\nfig, ax = plt.subplots()\n"
                     "im = ax.imshow(counts)\nax.colorbar(im)\n```\nColorbars belong to the figure. Fix it.",
         "fig, ax = plt.subplots()\nim = ax.imshow(counts)\nfig.colorbar(im, ax=ax)",
         {"post": post("_cbs('of')"), "setup": extra}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def color_drills(rng):
    code, desc, k = _sweep(rng, k=8)
    cm = rng.choice(["viridis", "plasma", "cividis"])
    s = seeded(rng)
    n = rng.randint(6, 9)
    items = [
        ("ScalarMappable", f"Drill: plot the 8 learning-rate runs colored by the sequential colormap {cm!r} on a log "
                           f"scale (`norm = LogNorm(lrs.min(), lrs.max())`, run color `cmap(norm(lr))`), and add a "
                           f"colorbar for it (`mpl.cm.ScalarMappable`) with one tick per learning rate, labelled "
                           f"`f'{{lr:.0e}}'`.",
         f"from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\nnorm = LogNorm(lrs.min(), lrs.max())\n"
         f"cmap = mpl.colormaps[{cm!r}]\nfor lr, r in zip(lrs, runs):\n    ax.plot(steps, r, color=cmap(norm(lr)))\n"
         f"cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)\n"
         f"cb.set_ticks(lrs, labels=[f'{{lr:.0e}}' for lr in lrs])",
         {"post": post("_lines(fc.ax(0), 'color'), _cbs('norm', 'clim', 'ticks', 'ticklabels')"), "require": ["set_ticks"]}),
        ("imshow", "Drill: show the weight-update matrix `dW` with the diverging colormap 'RdBu_r' so that **0 is exactly "
                   "white** (the middle of the map) whatever the data's range — e.g. `TwoSlopeNorm(vcenter=0)`, "
                   "`CenteredNorm()` or symmetric `vmin`/`vmax` — and add a colorbar labelled 'Δw'. (Checked: the "
                   "colormap, that the norm maps 0 to 0.5, and the colorbar.)",
         "from matplotlib.colors import TwoSlopeNorm\nfig, ax = plt.subplots()\n"
         "im = ax.imshow(dW, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))\nfig.colorbar(im, ax=ax, label='Δw')",
         {"post": post("_im = fc.ax(0).get_images()[0]\n_imgs(fc.ax(0), 'cmap', 'data'), round(float(_im.norm(0.0)), 6), "
                       "_cbs('label')"),
          "setup": f"dW = np.round(np.random.default_rng({s}).normal(0.3, 1, ({n}, {n})), 3)"}),
        ("colorbar", "Drill: four attention heads `att[h]` (rows sum to 1). Show them on `fig, axs = plt.subplots(2, 2, "
                     "layout='constrained')` with titles 'head 0' … 'head 3', all on the fixed scale 0 … 1, and share "
                     "**one** colorbar labelled 'attention'.",
         "fig, axs = plt.subplots(2, 2, layout='constrained')\nfor h, a in enumerate(axs.flat):\n"
         "    im = a.imshow(att[h], vmin=0, vmax=1)\n    a.set_title(f'head {h}')\nfig.colorbar(im, ax=axs, label='attention')",
         {"post": post("[(fc.labels(a)['title'], _imgs(a, 'clim')) for a in fc.axes()], _cbs('label')"),
          "setup": f"z = np.random.default_rng({s}).normal(0, 2, (4, {n}, {n}))\natt = np.round(np.exp(z) / np.exp(z).sum(-1, keepdims=True), 4)"}),
        ("BoundaryNorm", "The guide's four-panel figure: `fig, axs = plt.subplots(1, 4, figsize=(14, 3), "
                         "layout='constrained')`. (0) `data` with 'viridis' and `LogNorm(vmin=data.min(), "
                         "vmax=data.max())`; (1) `signed` with 'RdBu_r' and `TwoSlopeNorm(vcenter=0)`; (2) `signed` with "
                         "'RdBu_r' and `CenteredNorm()`, its colorbar `shrink=0.8, label='z'`, ticks -2, 0, 2 and 7 pt "
                         "tick labels; (3) `signed` with 'RdBu_r' resampled to 4 colors and `BoundaryNorm([-3, -1, 0, 1, "
                         "3], 4)`, its colorbar horizontal with `extend='both'`. Every panel gets its own colorbar.",
         "from matplotlib.colors import BoundaryNorm, CenteredNorm, LogNorm, TwoSlopeNorm\n"
         "fig, axs = plt.subplots(1, 4, figsize=(14, 3), layout='constrained')\n"
         "im0 = axs[0].imshow(data, cmap='viridis', norm=LogNorm(vmin=data.min(), vmax=data.max()))\nfig.colorbar(im0, ax=axs[0])\n"
         "im1 = axs[1].imshow(signed, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0))\nfig.colorbar(im1, ax=axs[1])\n"
         "im2 = axs[2].imshow(signed, cmap='RdBu_r', norm=CenteredNorm())\ncb = fig.colorbar(im2, ax=axs[2], shrink=0.8, label='z')\n"
         "cb.set_ticks([-2, 0, 2])\ncb.ax.tick_params(labelsize=7)\n"
         "im3 = axs[3].imshow(signed, cmap=mpl.colormaps['RdBu_r'].resampled(4), norm=BoundaryNorm([-3, -1, 0, 1, 3], 4))\n"
         "fig.colorbar(im3, ax=axs[3], orientation='horizontal', extend='both')",
         {"post": post("[_imgs(a, 'norm', 'cmap', 'N', 'clim') for a in fc.axes()], _cbs('of', 'label', 'orientation', 'extend'), "
                       "[d for d in _cbs('of', 'ticks', 'labelsize') if d['of'] == 2]"),
          "setup": f"g = np.random.default_rng({s})\ndata = np.round(g.lognormal(size=(12, 12)), 4)\nsigned = np.round(g.normal(size=(12, 12)), 4)"}),
        ("ListedColormap", "Grade bands for accuracies `acc` (a matrix): color them with "
                           "`ListedColormap(['tab:red', 'tab:orange', 'gold', 'tab:green'])` and `BoundaryNorm([0, 0.5, "
                           "0.7, 0.9, 1.0], 4)`, and add a colorbar with ticks at those five boundaries.",
         "from matplotlib.colors import BoundaryNorm, ListedColormap\nfig, ax = plt.subplots()\n"
         "bounds = [0, 0.5, 0.7, 0.9, 1.0]\nim = ax.imshow(acc, cmap=ListedColormap(['tab:red', 'tab:orange', 'gold', 'tab:green']), "
         "norm=BoundaryNorm(bounds, 4))\ncb = fig.colorbar(im, ax=ax)\ncb.set_ticks(bounds)",
         {"post": post("_imgs(fc.ax(0), 'lut', 'bounds', 'data'), _cbs('ticks')"),
          "setup": f"acc = np.round(np.random.default_rng({s}).beta(5, 2, ({n}, {n})), 3)"}),
        ("savefig", f"Plot the runs colored along {cm!r} (evenly sampled, in `lrs` order) on a 4 × 3 inch figure, save it "
                    f"as `colors.png` at 30 dpi, and close **all** figures (`plt.close('all')`). Return "
                    f"`plt.imread('colors.png').shape`.",
         f"fig, ax = plt.subplots(figsize=(4, 3))\nfor r, c in zip(runs, mpl.colormaps[{cm!r}](np.linspace(0, 1, len(runs)))):\n"
         f"    ax.plot(steps, r, color=c)\nfig.savefig('colors.png', dpi=30)\nplt.close('all')\nresult = plt.imread('colors.png').shape",
         {"post": "_, len(plt.get_fignums())", "require": ["savefig", "close"]}),
        ("Normalize", f"Color bars by their height: draw `ax.bar(range(len(final)), final, color=...)` where `final = "
                      f"runs[:, -1]` and bar i is `cmap(norm(final[i]))` with `norm = Normalize(final.min(), "
                      f"final.max())` and `cmap = mpl.colormaps[{cm!r}]`; add a colorbar for it (ScalarMappable) "
                      f"labelled 'final loss'.",
         f"from matplotlib.colors import Normalize\nfig, ax = plt.subplots()\nfinal = runs[:, -1]\n"
         f"norm = Normalize(final.min(), final.max())\ncmap = mpl.colormaps[{cm!r}]\n"
         f"ax.bar(range(len(final)), final, color=cmap(norm(final)))\n"
         f"fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='final loss')",
         {"post": post("[b['color'] for b in fc.bars(fc.ax(0), 6, 'color')], fc.heights(fc.ax(0), nd=4), _cbs('label', 'clim', 'cmap')")}),
        ("set_prop_cycle", "As in the guide: plot the first 5 runs on `fig, ax` with colors sampled evenly from 'plasma' "
                           "(`np.linspace(0, 1, 5)`), then switch the Axes' cycle with `ax.set_prop_cycle(color=['k', "
                           "'r', 'b'])` and plot the remaining runs without `color=`.",
         "fig, ax = plt.subplots()\nfor r, c in zip(runs[:5], mpl.colormaps['plasma'](np.linspace(0, 1, 5))):\n"
         "    ax.plot(steps, r, color=c)\nax.set_prop_cycle(color=['k', 'r', 'b'])\nfor r in runs[5:]:\n    ax.plot(steps, r)",
         {"post": post("_lines(fc.ax(0), 'color')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def colorbar_layouts(rng):
    code, desc, n = _mats(rng)
    x0 = rng.choice([0.85, 0.88, 0.9])
    fmt = rng.choice(["%.1f", "%.2f", "%.0f"])
    k = rng.choice([4, 5, 6])
    items = [
        ("colorbar", "Create `fig, axs = plt.subplots(2, 2)`. Top row: `data` and `2 * data` on one shared scale "
                     "(0 … `(2 * data).max()`) with **one** colorbar for both (`ax=axs[0, :]`). Bottom row: `signed` "
                     "with 'RdBu_r' and `np.abs(signed)` with 'Greys', each with its own colorbar.",
         "fig, axs = plt.subplots(2, 2)\nhi = (2 * data).max()\nfor a, m in zip(axs[0], (data, 2 * data)):\n"
         "    im = a.imshow(m, vmin=0, vmax=hi)\nfig.colorbar(im, ax=axs[0, :])\n"
         "fig.colorbar(axs[1, 0].imshow(signed, cmap='RdBu_r'), ax=axs[1, 0])\n"
         "fig.colorbar(axs[1, 1].imshow(np.abs(signed), cmap='Greys'), ax=axs[1, 1])",
         {"post": post("[_imgs(a, 'clim', 'cmap') for a in fc.axes()], sorted(d['of'] for d in _cbs('of'))")}),
        ("cax", f"Place the colorbar by hand: create `fig, ax = plt.subplots()`, `fig.subplots_adjust(right=0.8)`, show "
                f"`data`, add an Axes for the colorbar with `cax = fig.add_axes(({x0}, 0.15, 0.03, 0.7))` and draw the "
                f"colorbar into it (`fig.colorbar(im, cax=cax)`).",
         f"fig, ax = plt.subplots()\nfig.subplots_adjust(right=0.8)\nim = ax.imshow(data)\n"
         f"cax = fig.add_axes(({x0}, 0.15, 0.03, 0.7))\nfig.colorbar(im, cax=cax)",
         {"post": post("_cbs('box', 'orientation')")}),
        ("inset_axes", "Put a small horizontal colorbar **inside** the plot: show `signed` with 'RdBu_r' on `fig, ax`, "
                       "make `cax = ax.inset_axes((0.55, 0.9, 0.4, 0.05))` (axes fractions) and draw a horizontal "
                       "colorbar into it.",
         "fig, ax = plt.subplots()\nim = ax.imshow(signed, cmap='RdBu_r')\ncax = ax.inset_axes((0.55, 0.9, 0.4, 0.05))\n"
         "fig.colorbar(im, cax=cax, orientation='horizontal')",
         {"post": post("_cbs('orientation', 'box'), len(fc.ax(0).child_axes)")}),
        ("set_ticks", "Show `data` on `fig, ax` with a colorbar whose ticks sit at the 5th, 50th and 95th percentiles "
                      "of `data` (`np.percentile`), labelled 'p5', 'p50', 'p95'.",
         "fig, ax = plt.subplots()\nim = ax.imshow(data)\ncb = fig.colorbar(im, ax=ax)\n"
         "cb.set_ticks(np.percentile(data, [5, 50, 95]), labels=['p5', 'p50', 'p95'])",
         {"post": post("_cbs('ticks', 'ticklabels')")}),
        ("LogNorm", "Show `100 * data` on `fig, ax` with 'magma' and `LogNorm(vmin=1, vmax=1000)`, and a colorbar "
                    "labelled 'count (log)' with ticks at 1, 10, 100, 1000 written '1', '10', '100', '1k'.",
         "from matplotlib.colors import LogNorm\nfig, ax = plt.subplots()\n"
         "im = ax.imshow(100 * data, cmap='magma', norm=LogNorm(vmin=1, vmax=1000))\n"
         "cb = fig.colorbar(im, ax=ax, label='count (log)')\ncb.set_ticks([1, 10, 100, 1000], labels=['1', '10', '100', '1k'])",
         {"post": post("_imgs(fc.ax(0), 'norm', 'clim'), _cbs('label', 'ticks', 'ticklabels')")}),
        ("resampled", f"Compare continuous and quantized colors: `fig, (ax1, ax2) = plt.subplots(1, 2)`, `data` on both "
                      f"— 'viridis' on the left, 'viridis' resampled to {k} colors on the right — each with its own "
                      f"colorbar.",
         f"fig, (ax1, ax2) = plt.subplots(1, 2)\nfig.colorbar(ax1.imshow(data, cmap='viridis'), ax=ax1)\n"
         f"fig.colorbar(ax2.imshow(data, cmap=mpl.colormaps['viridis'].resampled({k})), ax=ax2)",
         {"post": post("[_imgs(a, 'cmap', 'N') for a in fc.axes()], _cbs('of')")}),
        ("location", "Show `signed` with 'RdBu_r' on `fig, ax` and put the colorbar on the **left** of the Axes "
                     "(`location='left'`), labelled 'z'.",
         "fig, ax = plt.subplots()\nim = ax.imshow(signed, cmap='RdBu_r')\nfig.colorbar(im, ax=ax, location='left', label='z')",
         {"post": post("_cbs('left', 'label', 'orientation')")}),
        ("format", f"Show `data` on `fig, ax` with a colorbar whose tick labels use the printf-style format {fmt!r} "
                   f"(`format=` of `fig.colorbar`).",
         f"fig, ax = plt.subplots()\nim = ax.imshow(data)\nfig.colorbar(im, ax=ax, format={fmt!r})",
         {"post": post("_cbs('fmt')")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

@template(M, "challenge")
def boundary_bins(rng):
    n, s = rng.randint(12, 18), seeded(rng)
    nb = rng.randint(3, 6)
    extend = rng.choice(["neither", "min", "max", "both"])
    ncol = rng.choice([nb + 2, 10, 16, 256])
    report = rng.choice(["index", "counts", "used"])
    setup = (f"g = np.random.default_rng({s})\nb = np.sort(g.choice(np.arange(-20, 21), {nb}, replace=False)) / 4\n"
             f"x = np.concatenate([g.uniform(b[0] - 2, b[-1] + 2, {n - 2}), g.choice(b, 2)])\n"
             f"ncolors, extend = {ncol}, {extend!r}\nreport = {report!r}")
    sol = ("offset = 1 if extend in ('min', 'both') else 0\nregions = len(b) - 1 + offset + (extend in ('max', 'both'))\n"
           "r = np.searchsorted(b, x, side='right') - 1 + offset     # boundaries <= x, minus 1, plus the offset\n"
           "if ncolors > regions:\n    r = ((ncolors - 1) / (regions - 1) * r).astype(int)   # stretch over the colormap\n"
           "idx = np.where(x < b[0], -1, np.where(x >= b[-1], ncolors, r))\n"
           + {"index": "result = idx",
              "counts": "result = np.bincount(idx + 1, minlength=ncolors + 2)",
              "used": "result = len(np.unique(idx[(idx >= 0) & (idx < ncolors)]))"}[report])
    tests = [f"g = np.random.default_rng({s + i})\nb = np.sort(g.choice(np.arange(-5000, 5001), {nbb}, replace=False)) / 4\n"
             f"x = np.concatenate([g.uniform(b[0] - 50, b[-1] + 50, 199000), g.choice(b, 1000)])\n"
             f"ncolors, extend = {nc}, {extend!r}\nreport = {report!r}"
             for i, (nbb, nc) in ((1, (200, 256)), (2, (60, 1000)))]
    want = {"index": "the color index of every value, as an int array",
            "counts": "how many values get each index, as an array of length `ncolors + 2` (entry 0 counts index -1, "
                      "the last entry counts index `ncolors`)",
            "used": "how many different in-range color indices (0 … ncolors - 1) the values use"}[report]
    return challenge(
        "Inside `BoundaryNorm`",
        "`BoundaryNorm(b, ncolors, extend=extend)` turns values into colormap indices. A value below `b[0]` gets -1 "
        "(the 'under' color) and one at or above `b[-1]` gets `ncolors` ('over'). Otherwise its region is "
        "`r = (number of boundaries ≤ x) - 1 + offset`, where `offset = 1` if `extend` is 'min' or 'both' (else 0). "
        "There are `regions = len(b) - 1` regions, plus one for an extended lower end ('min' / 'both') and one for an "
        "extended upper end ('max' / 'both'). If `ncolors > regions` the region is stretched over the colormap: the "
        "index is `int((ncolors - 1) / (regions - 1) * r)` (computed in this order, then truncated); otherwise it is "
        f"`r`. Return {want}. (`report` = {report!r}; `b` has at least 3 boundaries.)",
        setup, sol, tests, "≤ 2·10^5 values, ≤ 2·10^4 boundaries, regions ≤ ncolors ≤ 2·10^4",
        speed="O((n + len(b)) log len(b)) with a binary search (`np.searchsorted`)",
        answer_desc=" (an int)" if report == "used" else " (an int array)")


@template(M, "challenge")
def lut_lookup(rng):
    n, s = rng.randint(25, 35), seeded(rng)
    N = rng.choice([8, 10, 16])
    norm = rng.choice(["linear", "log", "twoslope"])
    gen = {"linear": "g.normal(1, 2, {n})", "log": "g.lognormal(0, 1.5, {n}) * g.choice([1, 1, 1, -1], {n})",
           "twoslope": "g.normal(0.5, 2, {n})"}[norm]
    params = {"linear": "vmin, vmax = -1.0, 3.5", "log": "vmin, vmax = 0.1, 20.0",
              "twoslope": "vmin, vcenter, vmax = -2.0, 0.5, 5.0"}[norm]

    def mk(seed, nn):
        return (f"g = np.random.default_rng({seed})\nx = {gen.format(n=nn)}\nx[g.integers(0, {nn}, {max(2, nn // 20)})] = np.nan\n"
                f"N = {N}\n{params}\nnorm = {norm!r}")
    setup = mk(s, n)
    step = {"linear": "t = (x - vmin) / (vmax - vmin)",
            "log": "with np.errstate(divide='ignore', invalid='ignore'):\n"
                   "    t = (np.log10(x) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))\n"
                   "t[~np.isfinite(t)] = np.nan          # log of x <= 0: 'bad'",
            "twoslope": "t = np.interp(x, [vmin, vcenter, vmax], [0, 0.5, 1], left=-np.inf, right=np.inf)"}[norm]
    sol = (step + "\nxa = t * N\nxa[xa == N] = N - 1                   # exactly 1.0 is the top color, not 'over'\n"
           "bad, under, over = np.isnan(xa), xa < 0, xa >= N\nidx = np.where(bad | under | over, 0, xa).astype(int)\n"
           "idx[under], idx[over], idx[bad] = N, N + 1, N + 2\nresult = np.bincount(idx, minlength=N + 3)")
    tests = [mk(s + 1, 200000), mk(s + 2, 150000)]
    rule = {"linear": "`Normalize(vmin, vmax)`: `t = (x - vmin) / (vmax - vmin)`",
            "log": "`LogNorm(vmin, vmax)`: `t = (log10(x) - log10(vmin)) / (log10(vmax) - log10(vmin))`, and values "
                   "≤ 0 are 'bad'",
            "twoslope": "`TwoSlopeNorm(vcenter, vmin, vmax)`: `t = np.interp(x, [vmin, vcenter, vmax], [0, 0.5, 1])` "
                        "inside the range, -∞ below `vmin` and +∞ above `vmax`"}[norm]
    return challenge(
        "Which color does each value get?",
        "To color a value Matplotlib first normalizes it, then looks the result up in the colormap's table of `N` "
        f"colors. Here the norm is {rule} (`norm` = {norm!r}). The lookup multiplies by `N`: `t * N == N` (t = 1) "
        "is the top color `N - 1`; otherwise a result below 0 is 'under', at least `N` is 'over', NaN is 'bad', and "
        "anything else uses entry `floor(t * N)`. Return how many values use each entry, as an int array of length "
        "`N + 3`: entries 0 … N - 1, then 'under', 'over', 'bad'.",
        setup, sol, tests, "≤ 2·10^5 values", speed="O(n), vectorized", answer_desc=" (an int array)")


@template(M, "challenge")
def gradient_lut(rng):
    m, q, s = rng.randint(3, 5), rng.randint(8, 12), seeded(rng)
    N = rng.choice([5, 8, 12, 20])
    report = rng.choice(["rgb", "gray", "distinct"])

    def mk(seed, mm, NN, qq):
        return (f"g = np.random.default_rng({seed})\n"
                f"pos = np.concatenate([[0], np.unique(g.integers(1, 1000, {mm - 2})) / 1000, [1]])\n"
                f"rgb = np.round(g.random((len(pos), 3)), 3)\nN = {NN}\n"
                f"v = np.concatenate([[0.0, 1.0], g.random({qq - 2})])\nreport = {report!r}")
    setup = mk(s, m, N, q)
    sol = ("x = pos * (N - 1)                              # anchors on the 0 … N-1 grid of table entries\n"
           "grid = np.arange(N, dtype=float)\nj = np.searchsorted(x, grid[1:-1])           # anchor right of each entry\n"
           "w = (grid[1:-1] - x[j - 1]) / (x[j] - x[j - 1])\n"
           "lut = np.vstack([rgb[:1], w[:, None] * (rgb[j] - rgb[j - 1]) + rgb[j - 1], rgb[-1:]])\n"
           "i = np.minimum((v * N).astype(int), N - 1)  # which table entry each value uses\n"
           + {"rgb": "result = lut[i]", "gray": "c = lut[i]\nresult = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]",
              "distinct": "result = len(np.unique(i))"}[report])
    tests = [mk(s + 1, 2000, 50000, 200000), mk(s + 2, 50, 256, 200000)]
    want = {"rgb": "the RGB color of every value of `v`, as a q × 3 float array",
            "gray": "the luminance `0.2126 r + 0.7152 g + 0.0722 b` of the color of every value of `v`",
            "distinct": "how many different table entries the values of `v` use"}[report]
    return challenge(
        "A gradient colormap, by hand",
        "`LinearSegmentedColormap.from_list('c', list(zip(pos, rgb)), N=N)` builds a table of `N` colors: entry `i` "
        "is the color at position `i / (N - 1)`, linearly interpolated (per channel) between the two anchors around "
        "it (`pos` is increasing, from 0 to 1; `rgb[k]` is the color at `pos[k]`). A value `v` in 0 … 1 then uses "
        "entry `floor(v * N)` — except `v = 1`, which uses the last entry — so colors are *quantized*, not "
        f"interpolated at `v` itself. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "≤ 2·10^5 anchors, N ≤ 10^5, ≤ 2·10^5 values",
        speed="O((m + N) log m + q): a binary search per table entry",
        answer_desc={"rgb": " (a float array)", "gray": " (a float array)", "distinct": " (an int)"}[report], tol=1e-6)

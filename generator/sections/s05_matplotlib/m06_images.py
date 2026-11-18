"""5.06 Images, heatmaps, contours & vector fields (imshow, matshow, pcolormesh, contour / contourf / clabel, quiver,
streamplot, spy)."""

from ... import themes as T
from ._common import catalog, challenge, lit, matrix, seeded, template

M = "5.06"
NCB = "len(fc.colorbars())"
CS = "[(c.filled, fc.r(c.levels, 4)) for c in fc.ax().collections if hasattr(c, 'allsegs')]"
QS = "[q for q in fc.ax().collections if type(q).__name__ == 'Quiver']"
QV = f"[(fc.r(q.X, 4), fc.r(q.Y, 4), fc.r(q.U, 4), fc.r(q.V, 4)) for q in {QS}]"


def _pix(expr: str, nd: int = 4) -> str:
    """Post code: an array that may be masked (images, meshes) as rounded lists, masked / NaN -> None.
    (`fc.r` can't take 2-D masked arrays: it turns their rows into None.)"""
    return f"fc.r(np.ma.filled(np.ma.asarray({expr}, float), np.nan), {nd})"


QM = f"[{_pix('c.get_array()')} for c in fc.ax().collections if type(c).__name__ == 'QuadMesh']"
LC = "[c for c in fc.ax().collections if type(c).__name__ == 'LineCollection']"
SP = f"[len(c.get_segments()) for c in {LC}]"
LAB = "fc.labels(fc.ax())"
OFF = f"[{_pix('c.get_offsets()')} for c in fc.ax().collections if type(c).__name__ == 'PathCollection']"


def _im(*keys, a="fc.ax()"):
    """Post code: every image's pixels plus the listed fc.images fields (cmap, clim, origin, extent, shape)."""
    return (f"[dict({{k: d[k] for k in {keys!r}}}, data={_pix('i.get_array()')}) "
            f"for d, i in zip(fc.images({a}), {a}.get_images())]")


def _cs(a="fc.ax()"):
    """Post code: (filled?, levels) of every contour set on an Axes."""
    return f"[(c.filled, fc.r(c.levels, 4)) for c in {a}.collections if hasattr(c, 'allsegs')]"


def _per_axes(expr: str) -> str:
    """Post code: `expr` (using `a`) for every Axes of the figure."""
    return f"[{expr} for a in fc.axes()]"


WHAT = ["a grayscale digit", "an activation map", "a saliency map", "a feature map", "a depth map", "a weight matrix"]


def _img(rng, sizes=((28, 28), (16, 16), (12, 20), (20, 12), (10, 14), (24, 18))):
    h, w = rng.choice(sizes)
    s = seeded(rng)
    return (f"img = np.round(np.random.default_rng({s}).random(({h}, {w})), 3)",
            f"`img` is {rng.choice(WHAT)} of {h} × {w} pixels (values in [0, 1))", h, w)


SURFACES = [
    ("np.exp(-(X ** 2 + Y ** 2)) - 0.5 * np.exp(-((X - 1) ** 2 + (Y - 1) ** 2))", "two Gaussian bumps"),
    ("X ** 2 + 3 * Y ** 2", "a stretched bowl (a quadratic loss)"),
    ("X ** 2 - Y ** 2", "a saddle"),
    ("np.sin(X) * np.cos(Y)", "an egg-crate surface"),
    ("(1 - X) ** 2 + 2 * (Y - X ** 2) ** 2", "a Rosenbrock-like valley"),
    ("np.log1p((X - 0.5) ** 2 + 2 * (Y + 0.5) ** 2)", "a log-loss bowl"),
]


def _surface(rng, n=(18, 36)):
    nx, ny = rng.randint(*n), rng.randint(*n)
    a, b = rng.choice([2, 3]), rng.choice([2, 3])
    f, name = rng.choice(SURFACES)
    return (f"X, Y = np.meshgrid(np.linspace(-{a}, {a}, {nx}), np.linspace(-{b}, {b}, {ny}))\nZ = np.round({f}, 4)",
            f"`X`, `Y` are {ny} × {nx} grids from `np.meshgrid` (x from -{a} to {a}, y from -{b} to {b}) and `Z` is "
            f"{name} on them", nx, ny)


FIELDS = [("-Y, X", "a rotation (a vortex)"), ("-X, -Y", "a sink (gradient descent on a round bowl)"),
          ("-2 * X, -6 * Y", "gradient descent on the bowl x² + 3y²"), ("X, -Y", "a saddle flow"),
          ("np.sin(Y), np.cos(X)", "a wavy flow"), ("Y, -np.sin(X)", "a pendulum's phase flow")]


def _field(rng, sizes=(7, 9, 11, 13, 15)):
    n = rng.choice(sizes)
    a = rng.choice([1, 2, 3])
    f, name = rng.choice(FIELDS)
    return (f"X, Y = np.meshgrid(np.linspace(-{a}, {a}, {n}), np.linspace(-{a}, {a}, {n}))\nU, V = {f}",
            f"`X`, `Y` hold a {n} × {n} grid of points (-{a} … {a}) and `U`, `V` the components of {name} there", n)


# ================================================================= lite

@template(M, "lite")
def imshow_options(rng):
    code, desc, h, w = _img(rng)
    cmap = rng.choice(["gray", "viridis", "magma", "Greys", "inferno"])
    interp = rng.choice(["nearest", "bilinear", "bicubic"])
    al = rng.choice([0.3, 0.5, 0.7])
    ex = rng.choice([[0, 10, 0, 5], [0, 1, 0, 1], [-1, 1, -1, 1], [0, 28, 0, 14]])
    lo, hi = rng.choice([(0, 1), (0.2, 0.8), (0.25, 0.75), (0, 0.5)])
    items = [
        (".imshow", f"Create `fig, ax` and show `img` with `ax.imshow` in the colormap {cmap!r}.",
         f"fig, ax = plt.subplots()\nax.imshow(img, cmap={cmap!r})", {"post": _im("cmap")}),
        ("vmin", f"Show `img` with `ax.imshow` on a fixed color scale: {lo} gets the lowest color and {hi} the highest "
                 f"(`vmin`, `vmax`; values outside are clipped).",
         f"fig, ax = plt.subplots()\nax.imshow(img, vmin={lo}, vmax={hi})", {"post": _im("clim"), "require": ["vmin", "vmax"]}),
        ("interpolation", f"Show `img` with `ax.imshow` using `interpolation={interp!r}` (how pixels are resampled "
                          f"when the image is drawn bigger or smaller than its size).",
         f"fig, ax = plt.subplots()\nax.imshow(img, interpolation={interp!r})",
         {"post": f"{_im()}, [i.get_interpolation() for i in fc.ax().images]"}),
        ("origin", "Show `img` with `ax.imshow` so that row 0 is at the **bottom** (the default puts it at the top).",
         "fig, ax = plt.subplots()\nax.imshow(img, origin='lower')", {"post": _im("origin")}),
        ("extent", f"Show `img` with `ax.imshow` stretched over the data rectangle x from {ex[0]} to {ex[1]} and y from "
                   f"{ex[2]} to {ex[3]} (`extent=[left, right, bottom, top]`).",
         f"fig, ax = plt.subplots()\nax.imshow(img, extent={ex})", {"post": _im("extent")}),
        ("aspect", "Show `img` with `ax.imshow` filling the whole Axes box: pixels need not be square (`aspect='auto'`).",
         "fig, ax = plt.subplots()\nax.imshow(img, aspect='auto')", {"post": f"{_im()}, fc.ax().get_aspect()"}),
        ("alpha", f"Show `img` with `ax.imshow` at opacity {al} (`alpha`).",
         f"fig, ax = plt.subplots()\nax.imshow(img, alpha={al})", {"post": f"{_im()}, [i.get_alpha() for i in fc.ax().images]"}),
        (".colorbar", "Show `img` with `im = ax.imshow(img)` and add a colorbar for it with `fig.colorbar(im, ax=ax)`.",
         "fig, ax = plt.subplots()\nim = ax.imshow(img)\nfig.colorbar(im, ax=ax)", {"post": f"{_im()}, {NCB}"}),
        ("plt.imshow", f"With pyplot only, show `img` in the colormap {cmap!r} and add a colorbar.",
         f"plt.imshow(img, cmap={cmap!r})\nplt.colorbar()", {"post": f"{_im('cmap')}, {NCB}", "require": ["imshow", "colorbar"]}),
        ("axis", "Show `img` with `ax.imshow` and hide the axes entirely (no ticks, no frame) with `ax.axis('off')`.",
         "fig, ax = plt.subplots()\nax.imshow(img)\nax.axis('off')", {"post": f"{_im()}, fc.ax().axison"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def matrix_views(rng):
    kind = rng.choice(["random", "corr", "confusion"])
    code, desc, n = matrix(rng, kind)
    cond = {"random": f"M > {rng.choice([0.6, 0.7, 0.8])}", "corr": f"np.abs(M) > {rng.choice([0.2, 0.3])}",
            "confusion": f"M > {rng.choice([3, 5])}"}[kind]
    lo, hi, cm = {"random": (0, 1, "viridis"), "corr": (-1, 1, "RdBu_r"), "confusion": (0, 60, "Blues")}[kind]
    p = {"random": 0.5, "corr": 0.25, "confusion": 4}[kind]
    items = [
        ("matshow", "Create `fig, ax` and show `M` with `ax.matshow` (an image with matrix conventions: row 0 at the "
                    "top, column ticks along the top).", "fig, ax = plt.subplots()\nax.matshow(M)", {"post": _im("origin")}),
        ("matshow", f"Show `M` with `ax.matshow` in the colormap {cm!r} with the color limits {lo} … {hi}.",
         f"fig, ax = plt.subplots()\nax.matshow(M, cmap={cm!r}, vmin={lo}, vmax={hi})", {"post": _im("cmap", "clim")}),
        ("pcolormesh", "Draw `M` as a grid of colored cells with `ax.pcolormesh(M)` (cell (i, j) spans x = j … j+1, "
                       "y = i … i+1, so row 0 is at the bottom).", "fig, ax = plt.subplots()\nax.pcolormesh(M)",
         {"post": QM}),
        ("edgecolors", "Draw `M` with `ax.pcolormesh` with white borders between the cells (`edgecolors='w'`).",
         "fig, ax = plt.subplots()\nax.pcolormesh(M, edgecolors='w')",
         {"post": f"{QM}, [fc.hexcolor(c.get_edgecolor()) for c in fc.ax().collections]",
          "require": ["pcolormesh", "edgecolors|edgecolor|ec"]}),
        ("spy", f"Show which entries satisfy `{cond}` with `ax.spy(...)` (a sparsity-pattern plot).",
         f"fig, ax = plt.subplots()\nax.spy({cond})", {"post": _im()}),
        ("markersize", f"Show the pattern of `{cond}` with `ax.spy` drawing one square marker per True entry "
                       f"(`markersize=4`) instead of an image.", f"fig, ax = plt.subplots()\nax.spy({cond}, markersize=4)",
         {"post": "fc.lines(fc.ax(), 'ms')", "require": ["spy", "markersize|ms"]}),
        ("precision", f"`ax.spy(M, precision=p)` marks the entries whose absolute value is above `p`. Show the entries "
                      f"of `M` above {p} that way.", f"fig, ax = plt.subplots()\nax.spy(M, precision={p})",
         {"post": _im(), "require": ["spy", "precision"]}),
        ("plt.matshow", "With pyplot only, show `M` with `plt.matshow` (it opens its own figure) and add a colorbar.",
         "plt.matshow(M)\nplt.colorbar()", {"post": f"{_im()}, {NCB}", "require": ["matshow", "colorbar"]}),
        (".imshow", f"A heatmap is just an image: show `M` with `ax.imshow` in the colormap {cm!r}.",
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap={cm!r})", {"post": _im("cmap")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def contour_calls(rng):
    code, desc, nx, ny = _surface(rng)
    k = rng.choice([4, 5, 6, 8, 10])
    cm = rng.choice(["coolwarm", "viridis", "plasma", "RdYlBu"])
    qs = rng.choice([[0.25, 0.5, 0.75], [0.1, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]])
    lw = rng.choice([0.5, 1, 2])
    items = [
        ("contour", "Create `fig, ax` and draw the contour lines of `Z` over `X`, `Y` with `ax.contour(X, Y, Z)`.",
         "fig, ax = plt.subplots()\nax.contour(X, Y, Z)", {"post": CS}),
        ("levels", f"Draw contour lines of `Z` (over `X`, `Y`) with `levels={k}`: Matplotlib then picks about {k} "
                   f"round values.", f"fig, ax = plt.subplots()\nax.contour(X, Y, Z, levels={k})",
         {"post": CS, "require": ["contour", "levels"]}),
        ("contourf", f"Draw **filled** contours of `Z` over `X`, `Y` with `levels={k}` in the colormap {cm!r}.",
         f"fig, ax = plt.subplots()\nax.contourf(X, Y, Z, levels={k}, cmap={cm!r})",
         {"post": f"{CS}, [c.get_cmap().name for c in fc.ax().collections]"}),
        ("quantile", f"Draw contour lines of `Z` only at its quantiles {qs} (`levels=np.quantile(Z, [...])`).",
         f"fig, ax = plt.subplots()\nax.contour(X, Y, Z, levels=np.quantile(Z, {qs}))",
         {"post": CS, "require": ["contour", "levels"]}),
        ("colors", f"Draw black contour lines of `Z` (`colors='k'`) of width {lw} (`linewidths`).",
         f"fig, ax = plt.subplots()\nax.contour(X, Y, Z, colors='k', linewidths={lw})",
         {"post": f"{CS}, [(fc.hexcolor(c.get_edgecolor()), fc.r(c.get_linewidths()[0], 3)) for c in fc.ax().collections]",
          "require": ["colors", "linewidths"]}),
        ("clabel", "Draw contour lines with `cs = ax.contour(X, Y, Z, levels=5)` and write each level's value on its "
                   "line with `ax.clabel(cs)`.", "fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels=5)\nax.clabel(cs)",
         {"post": f"{CS}, fc.texts()"}),
        (".colorbar", "Draw filled contours `cf = ax.contourf(X, Y, Z)` and add a colorbar for them with "
                      "`fig.colorbar(cf, ax=ax)`.", "fig, ax = plt.subplots()\ncf = ax.contourf(X, Y, Z)\nfig.colorbar(cf, ax=ax)",
         {"post": f"{CS}, {NCB}"}),
        ("plt.contourf", f"With pyplot only, draw filled contours of `Z` over `X`, `Y` with `levels={k}` and add a "
                         f"colorbar.", f"plt.contourf(X, Y, Z, levels={k})\nplt.colorbar()",
         {"post": f"{CS}, {NCB}", "require": ["contourf", "colorbar"]}),
        ("contour", "Contour `Z` **without** `X` and `Y` (`ax.contour(Z)`): the positions are then the column and row "
                    "indices.", "fig, ax = plt.subplots()\nax.contour(Z)", {"post": f"{CS}, fc.limits()"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def vector_calls(rng):
    code, desc, n = _field(rng)
    d = rng.choice([0.5, 0.8, 1.2])
    cm = rng.choice(["plasma", "viridis", "magma"])
    title = rng.choice(["Field", "Gradient flow", "Velocity"])
    items = [
        ("quiver", "Create `fig, ax` and draw an arrow `(U, V)` at every grid point `(X, Y)` with `ax.quiver`.",
         "fig, ax = plt.subplots()\nax.quiver(X, Y, U, V)", {"post": QV}),
        ("hypot", f"Draw the arrows with `ax.quiver` colored by their length: pass `np.hypot(U, V)` as the fifth "
                  f"argument and use the colormap {cm!r}.",
         f"fig, ax = plt.subplots()\nax.quiver(X, Y, U, V, np.hypot(U, V), cmap={cm!r})",
         {"post": f"{QV}, [({_pix('q.get_array()')}, q.get_cmap().name) for q in {QS}]", "require": ["quiver", "hypot"]}),
        ("quiver", "Draw the arrows with `ax.quiver(U, V)` alone: without positions, arrow (i, j) sits at x = j, y = i.",
         "fig, ax = plt.subplots()\nax.quiver(U, V)", {"post": QV}),
        ("streamplot", "Draw the field's streamlines with `ax.streamplot(X, Y, U, V)`.",
         "fig, ax = plt.subplots()\nax.streamplot(X, Y, U, V)", {"post": SP}),
        ("density", f"Draw streamlines with `ax.streamplot` with `density={d}`, in 'gray', with line width 0.6.",
         f"fig, ax = plt.subplots()\nax.streamplot(X, Y, U, V, density={d}, color='gray', linewidth=0.6)",
         {"post": f"{SP}, [(fc.hexcolor(c.get_color()), fc.r(c.get_linewidth()[0], 3)) for c in {LC}]",
          "require": ["streamplot", "density"]}),
        ("color", "Draw the field with `ax.quiver` in black arrows (`color='k'`).",
         "fig, ax = plt.subplots()\nax.quiver(X, Y, U, V, color='k')",
         {"post": f"{QV}, [fc.hexcolor(q.get_facecolor()) for q in {QS}]", "require": ["quiver", "color"]}),
        ("scale_units", "Draw the arrows with `ax.quiver` at true data scale, so arrow `(U, V)` spans exactly U by V "
                        "data units: `angles='xy', scale_units='xy', scale=1`.",
         "fig, ax = plt.subplots()\nax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)",
         {"post": f"{QV}, [(q.angles, q.scale_units, q.scale) for q in {QS}]",
          "require": ["angles", "scale_units", "scale"]}),
        ("plt.quiver", f"With pyplot only, draw the arrows with `plt.quiver` and title the plot {title!r}.",
         f"plt.quiver(X, Y, U, V)\nplt.title({title!r})", {"post": f"{QV}, {LAB}", "require": ["quiver", "title"]}),
        ("streamplot", f"Draw streamlines with `ax.streamplot`, colored by the speed `np.hypot(U, V)` (pass it as "
                       f"`color=`) in the colormap {cm!r}.",
         f"fig, ax = plt.subplots()\nax.streamplot(X, Y, U, V, color=np.hypot(U, V), cmap={cm!r})",
         {"post": f"{SP}, [c.get_cmap().name for c in {LC}]"}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= easy

@template(M, "easy")
def image_readback(rng):
    code, desc, h, w = _img(rng)
    lo, hi = rng.choice([(0.2, 0.8), (0.1, 0.6), (0.3, 0.9), (0.25, 0.75)])
    r, c = rng.randrange(h), rng.randrange(w)
    cm = rng.choice(["viridis", "magma", "gray", "cividis"])
    items = [
        ("get_clim", "Show `img` with `im = ax.imshow(img)` and return the color limits Matplotlib chose "
                     "(`im.get_clim()` as a list): without `vmin` / `vmax` they are the data's minimum and maximum.",
         "fig, ax = plt.subplots()\nim = ax.imshow(img)\nresult = list(im.get_clim())"),
        ("get_extent", f"Show `img` with `im = ax.imshow(img)` and return `im.get_extent()` as a list. (Pixel centers sit "
                       f"on integer coordinates, so the image spans -0.5 … {w} - 0.5 across; the y order is "
                       f"[bottom, top].)", "fig, ax = plt.subplots()\nim = ax.imshow(img)\nresult = list(im.get_extent())"),
        ("get_ylim", "Show `img` with `ax.imshow` and return the y limits (`ax.get_ylim()` as a list): with the "
                     "default `origin='upper'` row 0 is drawn at the top, so the y-axis runs downwards.",
         "fig, ax = plt.subplots()\nax.imshow(img)\nresult = list(ax.get_ylim())"),
        ("norm", f"Show `img` with `im = ax.imshow(img, cmap={cm!r}, vmin={lo}, vmax={hi})` and return the RGBA color "
                 f"of pixel ({r}, {c}), i.e. `im.cmap(im.norm(img[{r}, {c}]))`, as a list.",
         f"fig, ax = plt.subplots()\nim = ax.imshow(img, cmap={cm!r}, vmin={lo}, vmax={hi})\n"
         f"result = list(im.cmap(im.norm(img[{r}, {c}])))", {"require": ["norm"]}),
        ("norm", f"Show `img` with `im = ax.imshow(img, vmin={lo}, vmax={hi})`. Its norm maps `vmin` → 0 and `vmax` → "
                 f"1. Return how many pixels end up in the **top** color (`im.norm(img) >= 1`) and how many in the "
                 f"bottom one (`<= 0`), as `[top, bottom]`.",
         f"fig, ax = plt.subplots()\nim = ax.imshow(img, vmin={lo}, vmax={hi})\nv = im.norm(img)\n"
         f"result = [int((v >= 1).sum()), int((v <= 0).sum())]", {"require": ["norm"]}),
        ("get_array", "Show the three-channel version `np.dstack([img, img, img])` with `im = ax.imshow(...)` and "
                      "return the shape of the array the image holds (`im.get_array().shape`).",
         "fig, ax = plt.subplots()\nim = ax.imshow(np.dstack([img, img, img]))\nresult = im.get_array().shape",
         {"require": ["get_array"]}),
        ("cmap", "Show `img` with `im = ax.imshow(img, cmap='gray')` and return the colors at both ends of its colormap, "
                 "`[im.cmap(0.0), im.cmap(1.0)]`.",
         "fig, ax = plt.subplots()\nim = ax.imshow(img, cmap='gray')\nresult = [im.cmap(0.0), im.cmap(1.0)]"),
        ("yaxis_inverted", "Show `img` on two Axes side by side (`fig, (a1, a2) = plt.subplots(1, 2)`): on `a1` with the "
                           "default origin, on `a2` with `origin='lower'`. Return `[a1.yaxis_inverted(), "
                           "a2.yaxis_inverted()]`.",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.imshow(img)\na2.imshow(img, origin='lower')\n"
         "result = [a1.yaxis_inverted(), a2.yaxis_inverted()]", {"require": ["origin"]}),
        ("matshow", "Show `img` with `im = ax.matshow(img)` and return `[type(im).__name__, im.origin]`: matshow makes "
                    "an ordinary image.", "fig, ax = plt.subplots()\nim = ax.matshow(img)\nresult = [type(im).__name__, im.origin]"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def contour_readback(rng):
    code, desc, nx, ny = _surface(rng)
    k = rng.choice([3, 4, 5, 6, 8])
    qs = rng.choice([[0.1, 0.5, 0.9], [0.25, 0.75], [0.05, 0.5, 0.95]])
    f = rng.choice(["%.1f", "%.2f", "%1.3f"])
    items = [
        ("levels", f"Draw `cs = ax.contour(X, Y, Z, levels={k})` and return the levels Matplotlib picked "
                   f"(`cs.levels` as a list): round numbers spanning the data, about {k} of them.",
         f"fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels={k})\nresult = cs.levels.tolist()", {"tol": 1e-9}),
        ("contourf", f"Draw `cf = ax.contourf(X, Y, Z, levels={k})` and return the number of filled color bands (one "
                     f"between each pair of consecutive levels).",
         f"fig, ax = plt.subplots()\ncf = ax.contourf(X, Y, Z, levels={k})\nresult = len(cf.levels) - 1"),
        ("clabel", f"Draw `cs = ax.contour(X, Y, Z, levels={k})`, label the lines with `labels = ax.clabel(cs)` and "
                   f"return the label texts (lines too short for a label are skipped).",
         f"fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels={k})\nlabels = ax.clabel(cs)\n"
         f"result = [t.get_text() for t in labels]"),
        ("fmt", f"Draw `cs = ax.contour(X, Y, Z, levels={k})` and label it with `ax.clabel(cs, fmt={f!r})`; return "
                f"the label texts.",
         f"fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels={k})\nresult = [t.get_text() for t in ax.clabel(cs, fmt={f!r})]",
         {"require": ["clabel", "fmt"]}),
        ("allsegs", f"Draw `cs = ax.contour(X, Y, Z, levels=np.quantile(Z, {qs}))`. `cs.allsegs[i]` lists the separate "
                    f"curves (arrays of points) at level i: return how many curves each level has.",
         f"fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels=np.quantile(Z, {qs}))\n"
         f"result = [len(segs) for segs in cs.allsegs]"),
        ("contourf", f"Automatic levels bracket the data. Draw `cf = ax.contourf(X, Y, Z, levels={k})` and return "
                     f"`[cf.levels[0], Z.min(), Z.max(), cf.levels[-1]]` (a non-decreasing list).",
         f"fig, ax = plt.subplots()\ncf = ax.contourf(X, Y, Z, levels={k})\n"
         f"result = [float(cf.levels[0]), float(Z.min()), float(Z.max()), float(cf.levels[-1])]", {"tol": 1e-9}),
        ("clabel", "Draw two contour lines at the 30% and 70% quantiles of `Z` (`levels=np.quantile(Z, [0.3, 0.7])`) "
                   "and label them with words instead of numbers: `ax.clabel(cs, fmt={cs.levels[0]: 'low', "
                   "cs.levels[1]: 'high'})`. Return the label texts.",
         "fig, ax = plt.subplots()\ncs = ax.contour(X, Y, Z, levels=np.quantile(Z, [0.3, 0.7]))\n"
         "result = [t.get_text() for t in ax.clabel(cs, fmt={cs.levels[0]: 'low', cs.levels[1]: 'high'})]"),
        ("get_cmap", f"Draw `cf = ax.contourf(X, Y, Z, levels={k}, cmap='coolwarm')` and return the colormap's name and "
                     f"the number of levels as `[cf.get_cmap().name, len(cf.levels)]`.",
         f"fig, ax = plt.subplots()\ncf = ax.contourf(X, Y, Z, levels={k}, cmap='coolwarm')\n"
         f"result = [cf.get_cmap().name, len(cf.levels)]", {"require": ["contourf"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def rgb_images(rng):
    h, w = rng.choice([(8, 8), (12, 16), (16, 12), (10, 10)])
    s = seeded(rng)
    code = f"r, g, b = np.round(np.random.default_rng({s}).random((3, {h}, {w})), 3)"
    desc = f"`r`, `g`, `b` are the red, green and blue channels of a {h} × {w} image (floats in [0, 1))"
    items = [
        ("dstack", "Stack the channels into one H × W × 3 image (`np.dstack` or `np.stack(..., axis=-1)`) and show it "
                   "with `ax.imshow`.", "fig, ax = plt.subplots()\nax.imshow(np.dstack([r, g, b]))",
         {"post": _im("shape"), "require": ["dstack|stack"]}),
        ("transpose", "`chw = np.stack([r, g, b])` is channel-first, (3, H, W), like a PyTorch image tensor, but "
                      "`imshow` wants (H, W, 3). Reorder the axes and show it.",
         "fig, ax = plt.subplots()\nax.imshow(chw.transpose(1, 2, 0))",
         {"post": _im("shape"), "setup": "chw = np.stack([r, g, b])", "require": ["transpose|moveaxis"]}),
        ("astype", "Convert the RGB image `rgb = np.dstack([r, g, b])` to 8-bit integers in 0 … 255 "
                   "(`(rgb * 255).astype(np.uint8)`) and show that with `ax.imshow` (uint8 images use the 0 … 255 "
                   "range).", "fig, ax = plt.subplots()\nax.imshow((rgb * 255).astype(np.uint8))",
         {"post": _im("shape"), "setup": "rgb = np.dstack([r, g, b])", "require": ["astype", "uint8"]}),
        ("repeat", "Show the red channel as a gray RGB image: repeat `r` into 3 identical channels (shape (H, W, 3), "
                   "e.g. `np.repeat(r[..., None], 3, axis=2)`) and show that with `ax.imshow`.",
         "fig, ax = plt.subplots()\nax.imshow(np.repeat(r[..., None], 3, axis=2))",
         {"post": _im("shape"), "require": ["repeat|stack|dstack|tile"]}),
        ("dstack", "Make an RGBA image: the three channels plus a fourth, alpha channel equal to `g` (shape (H, W, 4)), "
                   "and show it with `ax.imshow`.", "fig, ax = plt.subplots()\nax.imshow(np.dstack([r, g, b, g]))",
         {"post": _im("shape"), "require": ["dstack|stack"]}),
        ("imshow", "Show only the red channel `r` with `ax.imshow`, in the colormap 'Reds' with the limits 0 … 1.",
         "fig, ax = plt.subplots()\nax.imshow(r, cmap='Reds', vmin=0, vmax=1)", {"post": _im("cmap", "clim")}),
        ("min", "`x = np.dstack([r, g, b]) * 4 - 2` has values outside [0, 1], which `imshow` would clip for an RGB "
                "image. Rescale it to [0, 1] with `(x - x.min()) / (x.max() - x.min())` and show the result.",
         "fig, ax = plt.subplots()\nax.imshow((x - x.min()) / (x.max() - x.min()))",
         {"post": _im("shape"), "setup": "x = np.dstack([r, g, b]) * 4 - 2", "require": ["min", "max"]}),
        ("mean", "Convert to grayscale as the mean of the three channels and show it with `ax.imshow` in 'gray' "
                 "(`vmin=0, vmax=1`).", "fig, ax = plt.subplots()\nax.imshow((r + g + b) / 3, cmap='gray', vmin=0, vmax=1)",
         {"post": _im("cmap", "clim"), "require": []}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def translate_image_calls(rng):
    code, desc, h, w = _img(rng)
    gcode, gdesc, nx, ny = _surface(rng, n=(15, 25))
    cm = rng.choice(["gray", "magma", "viridis"])
    title = rng.choice(["Input", "Activation", "Sample 3"])
    k = rng.choice([5, 8, 10])
    t = rng.choice([0.7, 0.8, 0.9])
    items = [
        ("plt.subplots", f"Rewrite this pyplot code in the explicit style (`fig, ax = plt.subplots()`, then `ax.` / "
                         f"`fig.` methods only):\n```python\nplt.imshow(img, cmap={cm!r})\nplt.colorbar()\n"
                         f"plt.title({title!r})\n```",
         f"fig, ax = plt.subplots()\nim = ax.imshow(img, cmap={cm!r})\nfig.colorbar(im, ax=ax)\nax.set_title({title!r})",
         {"post": f"{_im('cmap')}, {NCB}, {LAB}", "require": ["subplots", "colorbar"]}),
        ("plt.contourf", f"{gdesc}. Rewrite this explicit code with pyplot functions only:\n```python\nfig, ax = plt.subplots()\n"
                         f"cf = ax.contourf(X, Y, Z, levels={k})\nfig.colorbar(cf, ax=ax)\nax.set_xlabel('w1')\n```",
         f"plt.contourf(X, Y, Z, levels={k})\nplt.colorbar()\nplt.xlabel('w1')",
         {"post": f"{CS}, {NCB}, {LAB}", "setup": gcode, "require": ["contourf", "xlabel"]}),
        ("quiver", "Rewrite in the explicit style:\n```python\nplt.quiver(X, Y, -Y, X)\nplt.title('vortex')\n```",
         "fig, ax = plt.subplots()\nax.quiver(X, Y, -Y, X)\nax.set_title('vortex')",
         {"post": f"{QV}, {LAB}", "setup": "X, Y = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))",
          "require": ["subplots", "quiver"]}),
        ("plt.matshow", f"Rewrite with pyplot only (`plt.matshow` opens its own figure, so you need no "
                        f"`plt.figure`):\n```python\nfig, ax = plt.subplots()\nax.matshow(img, cmap={cm!r})\n```",
         f"plt.matshow(img, cmap={cm!r})", {"post": _im("cmap")}),
        ("pcolormesh", f"{gdesc}. Rewrite in the explicit style:\n```python\nplt.pcolormesh(X, Y, Z, shading='auto')\n"
                       "plt.colorbar()\n```",
         "fig, ax = plt.subplots()\npc = ax.pcolormesh(X, Y, Z, shading='auto')\nfig.colorbar(pc, ax=ax)",
         {"post": f"{QM}, {NCB}", "setup": gcode, "require": ["subplots", "pcolormesh", "colorbar"]}),
        ("plt.spy", f"Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.spy(img > {t})\n"
                    f"ax.set_title('mask')\n```", f"plt.spy(img > {t})\nplt.title('mask')",
         {"post": f"{_im()}, {LAB}", "require": ["spy", "title"]}),
        ("plt.imshow", f"Rewrite with pyplot only:\n```python\nfig, ax = plt.subplots()\nax.imshow(img, origin='lower', "
                       f"extent=[0, 1, 0, 1])\nax.axis('off')\n```",
         "plt.imshow(img, origin='lower', extent=[0, 1, 0, 1])\nplt.axis('off')",
         {"post": f"{_im('origin', 'extent')}, fc.ax().axison", "require": ["imshow", "axis"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def image_row(rng):
    k, (rows, cols) = rng.choice([(4, (2, 2)), (6, (2, 3)), (8, (2, 4)), (3, (1, 3)), (6, (3, 2))])
    h, w = rng.choice([(8, 8), (12, 12), (10, 14)])
    name, labs = T.labels(rng, 2)
    names = [labs[i % len(labs)] for i in range(k)]
    s = seeded(rng)
    code = f"imgs = np.round(np.random.default_rng({s}).random(({k}, {h}, {w})) * {rng.choice([1, 2, 5])}, 3)\nnames = {lit(names)}"
    desc = f"`imgs` holds {k} images of {h} × {w} pixels and `names` their {name} labels"
    data = _pix("a.images[0].get_array()")
    items = [
        ("subplots", f"Show the {k} images side by side in one row (`plt.subplots(1, {k})`), each titled with its "
                     f"name and with its axes hidden (`axis('off')`).",
         f"fig, axs = plt.subplots(1, {k})\nfor a, im, nm in zip(axs, imgs, names):\n    a.imshow(im)\n    a.set_title(nm)\n"
         f"    a.axis('off')", {"post": _per_axes(f"({data}, a.get_title(), a.axison)") + ", fc.grid_shape()"}),
        ("vmin", f"Show the {k} images in one row on **one** color scale (`vmin=imgs.min(), vmax=imgs.max()` for "
                 f"every image), so equal values get equal colors.",
         f"fig, axs = plt.subplots(1, {k})\nfor a, im in zip(axs, imgs):\n    a.imshow(im, vmin=imgs.min(), vmax=imgs.max())",
         {"post": _per_axes("fc.images(a)[0]['clim']") + ", " + _per_axes(data), "require": ["vmin", "vmax"]}),
        ("flat", f"Show the {k} images on a {rows} × {cols} grid in the colormap 'gray', looping over `axs.flat`.",
         f"fig, axs = plt.subplots({rows}, {cols})\nfor a, im in zip(axs.flat, imgs):\n    a.imshow(im, cmap='gray')",
         {"post": _per_axes(f"({data}, fc.images(a)[0]['cmap'])") + ", fc.grid_shape()"}),
        ("set_xticks", f"Show the {k} images in one row, keeping each frame but removing all ticks (`set_xticks([])` "
                       f"and `set_yticks([])`).",
         f"fig, axs = plt.subplots(1, {k})\nfor a, im in zip(axs, imgs):\n    a.imshow(im)\n    a.set_xticks([])\n"
         f"    a.set_yticks([])", {"post": _per_axes(f"({data}, a.get_xticks().tolist(), a.get_yticks().tolist(), a.axison)"),
                                  "require": ["set_xticks", "set_yticks"]}),
        ("colorbar", "Show `imgs[0]` and `imgs[1]` side by side (`fig, axs = plt.subplots(1, 2)`) on the same color "
                     "scale (`vmin=imgs.min(), vmax=imgs.max()`) and add **one** colorbar shared by both Axes: "
                     "`fig.colorbar(im, ax=axs)`.",
         "fig, axs = plt.subplots(1, 2)\nfor a, x in zip(axs, imgs[:2]):\n    im = a.imshow(x, vmin=imgs.min(), vmax=imgs.max())\n"
         "fig.colorbar(im, ax=axs)", {"post": _per_axes(f"({data}, fc.images(a)[0]['clim'])") + f", {NCB}"}),
        ("abs", "Show `imgs[0]`, `imgs[1]` and their difference `imgs[1] - imgs[0]` in a row of 3. Draw the "
                "difference in 'RdBu_r' with limits symmetric around 0: ±(largest absolute difference).",
         "d = imgs[1] - imgs[0]\nm = np.abs(d).max()\nfig, axs = plt.subplots(1, 3)\naxs[0].imshow(imgs[0])\n"
         "axs[1].imshow(imgs[1])\naxs[2].imshow(d, cmap='RdBu_r', vmin=-m, vmax=m)",
         {"post": _per_axes(data) + ", " + _im("cmap", "clim", a="fc.axes()[2]")}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def grid_data_prep(rng):
    n = rng.choice([20, 25, 30])
    a = rng.choice([1, 2, 3])
    f, name = rng.choice(SURFACES[:4])
    fcode, fdesc, fn = _field(rng)
    k = rng.randint(4, 6)
    s = seeded(rng)
    grid = f"x = np.linspace(-{a}, {a}, {n})"
    feats = f"F = np.round(np.random.default_rng({s}).normal(size=({k}, 40)).cumsum(axis=0), 3)"
    sol_grid = f"X, Y = np.meshgrid(x, x)\nZ = {f}"
    items = [
        ("meshgrid", f"`x = np.linspace(-{a}, {a}, {n})`. Build the 2-D grid `X, Y = np.meshgrid(x, x)`, compute "
                     f"`Z = {f}` ({name}) and show it with `ax.imshow` so that its pixels line up with the data: "
                     f"`extent=[-{a}, {a}, -{a}, {a}]`, `origin='lower'`.",
         f"{sol_grid}\nfig, ax = plt.subplots()\nax.imshow(Z, extent=[-{a}, {a}, -{a}, {a}], origin='lower')",
         {"setup": grid, "post": _im("extent", "origin"), "require": ["meshgrid", "extent"]}),
        ("meshgrid", f"`x = np.linspace(-{a}, {a}, {n})`. Build `X, Y = np.meshgrid(x, x)`, compute `Z = {f}` "
                     f"({name}) and draw it with `ax.pcolormesh(X, Y, Z, shading='auto')`.",
         f"{sol_grid}\nfig, ax = plt.subplots()\nax.pcolormesh(X, Y, Z, shading='auto')",
         {"setup": grid, "post": f"{QM}, fc.limits()", "require": ["meshgrid", "pcolormesh"]}),
        ("hypot", f"{fdesc}. Compute the speed `np.hypot(U, V)` and draw the arrows with `ax.quiver`, colored by speed "
                  f"(fifth argument) in the colormap 'viridis'.",
         "speed = np.hypot(U, V)\nfig, ax = plt.subplots()\nax.quiver(X, Y, U, V, speed, cmap='viridis')",
         {"setup": fcode, "post": f"{QV}, [{_pix('q.get_array()')} for q in {QS}]", "require": ["hypot", "quiver"]}),
        ("hypot", f"{fdesc}. Draw every arrow with the same length 1 (only the direction shows): divide `U` and `V` by "
                  f"the speed `np.hypot(U, V)`, leaving zero vectors at 0 (divide those by 1 instead: "
                  f"`np.where(speed > 0, speed, 1)`). Draw with `ax.quiver`.",
         "speed = np.hypot(U, V)\nsafe = np.where(speed > 0, speed, 1)\nfig, ax = plt.subplots()\n"
         "ax.quiver(X, Y, U / safe, V / safe)", {"setup": fcode, "post": QV, "require": ["hypot", "quiver"]}),
        ("corrcoef", f"`F` holds {k} features (rows) × 40 samples. Compute their {k} × {k} correlation matrix with "
                     f"`np.corrcoef(F)` and show it with `ax.matshow` in 'RdBu_r' with the limits -1 … 1.",
         "C = np.corrcoef(F)\nfig, ax = plt.subplots()\nax.matshow(C, cmap='RdBu_r', vmin=-1, vmax=1)",
         {"setup": feats, "post": _im("cmap", "clim")}),
        ("exp", f"`x = np.linspace(-{a}, {a}, {n})`. Build `X, Y = np.meshgrid(x, x)`, compute the Gaussian bump "
                f"`Z = np.exp(-(X ** 2 + Y ** 2))` and draw its filled contours with `levels=6`.",
         "X, Y = np.meshgrid(x, x)\nZ = np.exp(-(X ** 2 + Y ** 2))\nfig, ax = plt.subplots()\nax.contourf(X, Y, Z, levels=6)",
         {"setup": grid, "post": f"{CS}, fc.limits()", "require": ["meshgrid", "exp", "contourf"]}),
        ("meshgrid", f"`x = np.linspace(-{a}, {a}, {n})`. Draw the vector field (U, V) = (-y, x) (a rotation) as "
                     f"streamlines over the grid `np.meshgrid(x, x)` with `ax.streamplot`.",
         "X, Y = np.meshgrid(x, x)\nfig, ax = plt.subplots()\nax.streamplot(X, Y, -Y, X)",
         {"setup": grid, "post": f"{SP}, fc.limits()", "require": ["meshgrid", "streamplot"]}),
    ]
    return catalog(rng, items, "", "")


# ================================================================= medium

TXT = "sorted((t['xy'], t['text'], t['ha'], t['va']) for t in fc.text_items())"
FEATURES = [["age", "income", "tenure", "visits", "spend", "clicks", "score"],
            ["loss", "grad_norm", "lr", "acc", "tokens", "throughput", "mem"],
            ["pitch", "energy", "tempo", "zcr", "rolloff", "mfcc1", "mfcc2"],
            ["sepal_l", "sepal_w", "petal_l", "petal_w", "stem", "leaf", "root"]]


def _confusion(rng, k=(3, 6)):
    k_ = rng.randint(*k)
    name, labs = T.labels(rng, k_)
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nM = g.integers(0, 9, ({k_}, {k_})) + np.diag(g.integers(25, 60, {k_}))\n"
            f"class_names = {lit(list(labs[:k_]))}")
    return code, (f"`M` is the {k_} × {k_} confusion matrix of a {name} classifier (rows: true class, columns: "
                  f"predicted) and `class_names` the class names"), k_


@template(M, "medium")
def confusion_heatmap(rng):
    code, desc, k = _confusion(rng)
    xl, yl = rng.choice([("predicted", "true"), ("predicted label", "true label"), ("prediction", "actual")])
    rot = rng.choice([30, 45, 90])
    ticks = "fc.ticks(fc.ax(), 'x')['labels'], fc.ticks(fc.ax(), 'y')['labels']"
    items = [
        ("set_xticks", f"Show `M` with `ax.imshow(M, cmap='Blues')`, put a tick per class on both axes labelled with "
                       f"the class names (`ax.set_xticks(range({k}), labels=class_names)`, same for y), and label the "
                       f"axes {xl!r} (x) and {yl!r} (y).",
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap='Blues')\nax.set_xticks(range({k}), labels=class_names)\n"
         f"ax.set_yticks(range({k}), labels=class_names)\nax.set_xlabel({xl!r})\nax.set_ylabel({yl!r})",
         {"post": f"{_im('cmap')}, {ticks}, {LAB}", "require": ["set_xticks", "set_yticks"]}),
        ("text", "Show `M` with `ax.imshow(M, cmap='Blues')` and write each count in the middle of its cell: "
                 "`ax.text(j, i, M[i, j], ha='center', va='center')` (cell (i, j) is centered at x = j, y = i).",
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap='Blues')\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        ax.text(j, i, M[i, j], ha='center', va='center')", {"post": f"{_im()}, {TXT}"}),
        ("sum", "Normalize each row to sum to 1 (the per-class recall: `M / M.sum(axis=1, keepdims=True)`) and show "
                "the result with `ax.imshow` in 'Blues' with the limits 0 … 1, plus a colorbar.",
         "R = M / M.sum(axis=1, keepdims=True)\nfig, ax = plt.subplots()\nim = ax.imshow(R, cmap='Blues', vmin=0, vmax=1)\n"
         "fig.colorbar(im, ax=ax)", {"post": f"{_im('cmap', 'clim')}, {NCB}"}),
        ("text", "Show `M` with `ax.imshow(M, cmap='Blues')` and write each count centered in its cell, in **white** "
                 "where the count is above `M.max() / 2` (dark cells) and **black** elsewhere.",
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap='Blues')\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        c = 'white' if M[i, j] > M.max() / 2 else 'black'\n"
         f"        ax.text(j, i, M[i, j], ha='center', va='center', color=c)",
         {"post": "sorted((t['xy'], t['text'], t['color']) for t in fc.text_items())", "require": ["text", "color"]}),
        ("matshow", f"Show `M` with `ax.matshow` and label its ticks with the class names on both axes (column ticks "
                    f"are along the top).", f"fig, ax = plt.subplots()\nax.matshow(M)\n"
                                           f"ax.set_xticks(range({k}), labels=class_names)\nax.set_yticks(range({k}), labels=class_names)",
         {"post": f"{_im('origin')}, {ticks}", "require": ["matshow", "set_xticks", "set_yticks"]}),
        ("fill_diagonal", "Show only the mistakes: copy `M`, set the copy's diagonal to 0 (`np.fill_diagonal`) and show "
                          "it with `ax.imshow` in 'Reds'.",
         "E = M.copy()\nnp.fill_diagonal(E, 0)\nfig, ax = plt.subplots()\nax.imshow(E, cmap='Reds')",
         {"post": _im("cmap"), "require": ["fill_diagonal|diag|eye"]}),
        ("text", "Show `M` with `ax.imshow(M, cmap='Blues')` and write in each cell, centered, its share of the row as "
                 "a whole percentage, like '87%' (`f'{p:.0%}'` of `M[i, j] / M[i].sum()`).",
         f"fig, ax = plt.subplots()\nax.imshow(M, cmap='Blues')\nfor i in range({k}):\n    for j in range({k}):\n"
         f"        ax.text(j, i, f'{{M[i, j] / M[i].sum():.0%}}', ha='center', va='center')", {"post": f"{_im()}, {TXT}"}),
        ("rotation", f"Show `M` with `ax.imshow`, label the x ticks with the class names rotated by {rot}° and the y "
                     f"ticks with the class names (horizontal).",
         f"fig, ax = plt.subplots()\nax.imshow(M)\nax.set_xticks(range({k}), labels=class_names, rotation={rot})\n"
         f"ax.set_yticks(range({k}), labels=class_names)",
         {"post": f"{_im()}, {ticks}, sorted({{t.get_rotation() for t in fc.ax().get_xticklabels()}})",
          "require": ["set_xticks"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def correlation_heatmap(rng):
    k = rng.randint(4, 6)
    names = rng.choice(FEATURES)[:k]
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nF = np.round(g.normal(size=(60, {k})) @ g.normal(size=({k}, {k})), 3)\n"
            f"names = {lit(names)}")
    desc = f"`F` holds 60 samples (rows) of {k} features (columns) named in `names`"
    corr = "C = np.corrcoef(F, rowvar=False)"
    ticks = "fc.ticks(fc.ax(), 'x')['labels'], fc.ticks(fc.ax(), 'y')['labels']"
    cm = rng.choice(["RdBu_r", "coolwarm", "bwr", "PuOr"])
    items = [
        ("corrcoef", f"Compute the {k} × {k} correlation matrix (`np.corrcoef(F, rowvar=False)`: the features are "
                     f"columns) and show it with `ax.imshow` in {cm!r} with the limits -1 … 1 and a colorbar.",
         f"{corr}\nfig, ax = plt.subplots()\nim = ax.imshow(C, cmap={cm!r}, vmin=-1, vmax=1)\nfig.colorbar(im, ax=ax)",
         {"post": f"{_im('cmap', 'clim')}, {NCB}", "require": ["corrcoef", "colorbar"]}),
        ("matshow", f"Compute the correlation matrix of the features and show it with `ax.matshow` in {cm!r} (limits "
                    f"-1 … 1), with the feature names as tick labels on both axes.",
         f"{corr}\nfig, ax = plt.subplots()\nax.matshow(C, cmap={cm!r}, vmin=-1, vmax=1)\n"
         f"ax.set_xticks(range({k}), labels=names)\nax.set_yticks(range({k}), labels=names)",
         {"post": f"{_im('cmap', 'clim')}, {ticks}", "require": ["corrcoef", "matshow"]}),
        ("triu", f"Compute the correlation matrix and hide its redundant upper triangle (the cells strictly above the "
                 f"diagonal) by setting them to NaN (`np.triu(np.ones(({k}, {k}), bool), k=1)` marks them); NaN cells "
                 f"are left blank. Show it with `ax.imshow` in {cm!r}, limits -1 … 1.",
         f"{corr}\nC[np.triu(np.ones(({k}, {k}), bool), k=1)] = np.nan\nfig, ax = plt.subplots()\n"
         f"ax.imshow(C, cmap={cm!r}, vmin=-1, vmax=1)", {"post": _im("cmap", "clim"), "require": ["corrcoef", "triu"]}),
        ("CenteredNorm", f"Compute the correlation matrix of the features **without the diagonal** (set it to 0 with "
                         f"`np.fill_diagonal(C, 0)`) and show it with `ax.imshow` in {cm!r} with a norm centered on 0 "
                         f"that still fits the data: `norm=mpl.colors.CenteredNorm()`.",
         f"{corr}\nnp.fill_diagonal(C, 0)\nfig, ax = plt.subplots()\nim = ax.imshow(C, cmap={cm!r}, norm=mpl.colors.CenteredNorm())",
         {"post": f"{_im('cmap')}, fc.norm(fc.ax().images[0])", "require": ["CenteredNorm"]}),
        ("text", f"Compute the correlation matrix, show it with `ax.imshow` in {cm!r} (limits -1 … 1) and write each "
                 f"value with 2 decimals centered in its cell (`f'{{C[i, j]:.2f}}'`, ha and va 'center').",
         f"{corr}\nfig, ax = plt.subplots()\nax.imshow(C, cmap={cm!r}, vmin=-1, vmax=1)\nfor i in range({k}):\n"
         f"    for j in range({k}):\n        ax.text(j, i, f'{{C[i, j]:.2f}}', ha='center', va='center')",
         {"post": f"{_im('cmap')}, {TXT}", "require": ["corrcoef", "text"]}),
        ("argsort", f"Order the features by their correlation with the first one, strongest first (`idx = "
                    f"np.argsort(-C[0])`), and show the reordered matrix `C[np.ix_(idx, idx)]` with `ax.imshow` "
                    f"(limits -1 … 1), with the reordered names as tick labels on both axes.",
         f"{corr}\nidx = np.argsort(-C[0])\nfig, ax = plt.subplots()\nax.imshow(C[np.ix_(idx, idx)], vmin=-1, vmax=1)\n"
         f"labels = [names[i] for i in idx]\nax.set_xticks(range({k}), labels=labels)\nax.set_yticks(range({k}), labels=labels)",
         {"post": f"{_im('clim')}, {ticks}", "require": ["argsort"]}),
        ("abs", "Show the **strength** of each correlation regardless of sign: `np.abs` of the correlation matrix, "
                "with `ax.imshow` in 'viridis' (limits 0 … 1) and a colorbar labelled '|r|'.",
         f"{corr}\nfig, ax = plt.subplots()\nim = ax.imshow(np.abs(C), cmap='viridis', vmin=0, vmax=1)\n"
         f"fig.colorbar(im, ax=ax, label='|r|')", {"post": f"{_im('cmap', 'clim')}, fc.colorbars()", "require": ["abs"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


LOSSES = [("w1 ** 2 + 3 * w2 ** 2", "a stretched bowl", (2, 6)), ("(1 - w1) ** 2 + 5 * (w2 - w1 ** 2) ** 2",
          "a Rosenbrock-like valley", None), ("np.log1p((w1 - 0.5) ** 2 + 4 * (w2 + 0.3) ** 2)", "a log-loss bowl", None),
          ("0.5 * w1 ** 2 + w1 * w2 + 2 * w2 ** 2", "a tilted quadratic", None),
          ("(w1 ** 2 + w2 - 1) ** 2 + (w1 + w2 ** 2 - 1) ** 2", "a two-valley loss", None)]


@template(M, "medium")
def loss_landscape(rng):
    n = rng.choice([30, 35, 40])
    a = rng.choice([1.5, 2, 2.5])
    f, name, _ = rng.choice(LOSSES)
    k = rng.choice([8, 10, 12, 15])
    cm = rng.choice(["viridis", "magma", "cividis", "YlGnBu"])
    code = (f"w1, w2 = np.meshgrid(np.linspace(-{a}, {a}, {n}), np.linspace(-{a}, {a}, {n}))\nL = np.round({f}, 5)\n"
            f"path = np.round(np.array([[-{a} * 0.8, {a} * 0.7]]) * 0.7 ** np.arange(6)[:, None], 4)")
    desc = (f"`w1`, `w2` are {n} × {n} grids of two weights (-{a} … {a}), `L` is {name} on them, and `path` (6 × 2) "
            f"holds the (w1, w2) points visited by an optimizer")
    cf = f"fig, ax = plt.subplots()\ncf = ax.contourf(w1, w2, L, levels={k}, cmap={cm!r})"
    items = [
        ("clabel", f"Drill: draw `L` as filled contours with `levels={k}` in {cm!r}, overlay black contour lines at "
                   f"the same levels (`levels=cf.levels`, linewidth 0.5), and label those lines with `ax.clabel` "
                   f"(`inline=True`, `fontsize=7`).",
         f"{cf}\ncs = ax.contour(w1, w2, L, levels=cf.levels, colors='k', linewidths=0.5)\n"
         f"ax.clabel(cs, inline=True, fontsize=7)",
         {"post": f"{CS}, fc.texts()", "require": ["contourf", "contour", "clabel"]}),
        ("unravel_index", f"Draw `L` as filled contours (`levels={k}`, {cm!r}) and mark its lowest grid point with a "
                          f"red star (`'r*'`): find it with `np.unravel_index(L.argmin(), L.shape)` and plot "
                          f"`w1[i, j], w2[i, j]`.",
         f"{cf}\ni, j = np.unravel_index(L.argmin(), L.shape)\nax.plot(w1[i, j], w2[i, j], 'r*')",
         {"post": f"{CS}, fc.lines(fc.ax(), 'color', 'marker')", "require": ["argmin"]}),
        ("geomspace", "Losses span orders of magnitude: draw contour lines of `L` at 8 geometrically spaced levels "
                      "from `L.min() + 0.01` to `L.max()` (`np.geomspace`), and label them with `ax.clabel(cs, "
                      "fmt='%.2g')`.",
         "fig, ax = plt.subplots()\ncs = ax.contour(w1, w2, L, levels=np.geomspace(L.min() + 0.01, L.max(), 8))\n"
         "ax.clabel(cs, fmt='%.2g')", {"post": f"{CS}, fc.texts()", "require": ["geomspace|logspace", "clabel"]}),
        ("plot", f"Draw `L` as filled contours (`levels={k}`, {cm!r}) and overlay the optimizer's `path` as a white "
                 f"line with circle markers (`'wo-'`).", f"{cf}\nax.plot(path[:, 0], path[:, 1], 'wo-')",
         {"post": f"{CS}, fc.lines(fc.ax(), 'color', 'marker')", "require": ["contourf", "plot"]}),
        ("colorbar", f"Draw `L` as filled contours (`levels={k}`, {cm!r}) with a colorbar labelled 'loss', and label "
                     f"the axes 'w1' and 'w2'.",
         f"{cf}\nfig.colorbar(cf, ax=ax, label='loss')\nax.set(xlabel='w1', ylabel='w2')",
         {"post": f"{CS}, fc.colorbars(), {LAB}", "require": ["contourf", "colorbar"]}),
        ("extent", f"Draw `L` as an image whose pixels line up with the weights (`extent=[-{a}, {a}, -{a}, {a}]`, "
                   f"`origin='lower'`, colormap {cm!r}) and overlay white contour lines with `levels={k}` on the same "
                   f"Axes.", f"fig, ax = plt.subplots()\nax.imshow(L, extent=[-{a}, {a}, -{a}, {a}], origin='lower', "
                             f"cmap={cm!r})\nax.contour(w1, w2, L, levels={k}, colors='w')",
         {"post": f"{_im('extent', 'origin', 'cmap')}, {CS}", "require": ["imshow", "extent", "contour"]}),
        ("log1p", f"Compare scales side by side (`fig, (a1, a2) = plt.subplots(1, 2)`): filled contours of `L` on the "
                  f"left and of `np.log1p(L)` on the right, both with `levels={k}`, titled 'loss' and 'log(1 + loss)'.",
         f"fig, (a1, a2) = plt.subplots(1, 2)\na1.contourf(w1, w2, L, levels={k})\na1.set_title('loss')\n"
         f"a2.contourf(w1, w2, np.log1p(L), levels={k})\na2.set_title('log(1 + loss)')",
         {"post": _per_axes(f"({_cs('a')}, a.get_title())"), "require": ["log1p", "contourf"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def attention_maps(rng):
    T_, H = rng.randint(5, 8), rng.choice([2, 3, 4])
    sent = rng.choice(["the cat sat on the warm mat today", "we train a small model on toy data",
                       "attention is all you need for this demo", "a quick brown fox jumps over dogs"])
    tokens = sent.split()[:T_]
    s = seeded(rng)
    t = rng.choice([0.2, 0.25, 0.3])
    cm = rng.choice(["viridis", "magma", "Blues", "inferno"])
    code = (f"g = np.random.default_rng({s})\nlogits = g.normal(0, 1.5, ({H}, {T_}, {T_}))\n"
            f"A = np.round(np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True), 4)\ntokens = {lit(tokens)}")
    desc = (f"`A` holds the attention weights of {H} heads over the {T_} `tokens`: `A[h, i, j]` is how much query token "
            f"i attends to key token j in head h (each row sums to 1)")
    ticks = "fc.ticks(fc.ax(), 'x')['labels'], fc.ticks(fc.ax(), 'y')['labels']"
    items = [
        ("imshow", f"Show head 0's map `A[0]` with `ax.imshow` in {cm!r} with the limits 0 … 1, with the tokens as tick "
                   f"labels on both axes (rotate the x labels by 90°).",
         f"fig, ax = plt.subplots()\nax.imshow(A[0], cmap={cm!r}, vmin=0, vmax=1)\n"
         f"ax.set_xticks(range({T_}), labels=tokens, rotation=90)\nax.set_yticks(range({T_}), labels=tokens)",
         {"post": f"{_im('cmap', 'clim')}, {ticks}, sorted({{t.get_rotation() for t in fc.ax().get_xticklabels()}})",
          "require": ["set_xticks", "set_yticks"]}),
        ("tril", "Apply a causal mask to `A[0]`: keep only the weights on and below the diagonal (`np.tril`), "
                 "renormalize each row to sum to 1, and show the result with `ax.imshow` (limits 0 … 1).",
         "C = np.tril(A[0])\nC = C / C.sum(axis=1, keepdims=True)\nfig, ax = plt.subplots()\nax.imshow(C, vmin=0, vmax=1)",
         {"post": _im("clim"), "require": ["tril"]}),
        ("subplots", f"Show all {H} heads in one row (`plt.subplots(1, {H})`), titled 'head 0', 'head 1', …, each with "
                     f"`ax.imshow` in {cm!r} with the limits 0 … 1.",
         f"fig, axs = plt.subplots(1, {H})\nfor h, a in enumerate(axs):\n    a.imshow(A[h], cmap={cm!r}, vmin=0, vmax=1)\n"
         f"    a.set_title(f'head {{h}}')",
         {"post": _per_axes("(" + _pix("a.images[0].get_array()") + ", fc.images(a)[0]['cmap'], "
                            "fc.images(a)[0]['clim'], a.get_title())")}),
        ("argmax", "Show `A[0]` with `ax.imshow`, then mark in every row the key with the largest weight: plot x = "
                   "`A[0].argmax(axis=1)` against y = the row index as white crosses (`'wx'`, no line).",
         f"fig, ax = plt.subplots()\nax.imshow(A[0])\nax.plot(A[0].argmax(axis=1), np.arange({T_}), 'wx')",
         {"post": f"{_im()}, fc.lines(fc.ax(), 'color', 'marker', 'ls')", "require": ["argmax"]}),
        ("spy", f"Show which weights of head 1 (`A[1]`) are above {t} with `ax.spy`.",
         f"fig, ax = plt.subplots()\nax.spy(A[1] > {t})", {"post": _im()}),
        ("mean", "Average the heads (`A.mean(axis=0)`) and show the result with `ax.matshow` in 'Blues', with a "
                 "colorbar.", "fig, ax = plt.subplots()\nim = ax.matshow(A.mean(axis=0), cmap='Blues')\nfig.colorbar(im, ax=ax)",
         {"post": f"{_im('cmap')}, {NCB}", "require": ["mean", "matshow"]}),
        ("invert_yaxis", "Draw `A[0]` as cells with `ax.pcolormesh`, then flip the y-axis (`ax.invert_yaxis()`) so "
                         "that query 0 is the top row, as in `imshow`.",
         "fig, ax = plt.subplots()\nax.pcolormesh(A[0])\nax.invert_yaxis()",
         {"post": f"{QM}, fc.ax().yaxis_inverted()", "require": ["pcolormesh", "invert_yaxis"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def gradient_fields(rng):
    n = rng.choice([13, 15, 17])
    a, b, c = rng.choice([(1, 3, 0), (1, 2, 1), (2, 1, -1), (1, 1, 0.5), (0.5, 2, 0)])
    k = rng.choice([6, 8, 10])
    cm = rng.choice(["autumn", "plasma", "cool"])
    code = (f"x = np.linspace(-2, 2, {n})\nX, Y = np.meshgrid(x, x)\nL = {a} * X ** 2 + {b} * Y ** 2 + {c} * X * Y\n"
            f"starts = np.array([[-1.8, 1.5], [1.6, -1.2], [1.0, 1.8]])")
    desc = (f"`L` = {a}·x² + {b}·y² + {c}·xy is a quadratic loss on the {n} × {n} grid `X, Y` (built from `x`); its "
            f"gradient is (dL/dx, dL/dy) = ({2 * a}·x + {c}·y, {2 * b}·y + {c}·x). `starts` holds 3 starting points")
    grad = f"U = -({2 * a} * X + {c} * Y)\nV = -({2 * b} * Y + {c} * X)"
    items = [
        ("quiver", f"Draw contour lines of `L` (`levels={k}`) and, on the same Axes, the gradient-descent direction "
                   f"(the **negative** gradient) as arrows with `ax.quiver`.",
         f"{grad}\nfig, ax = plt.subplots()\nax.contour(X, Y, L, levels={k})\nax.quiver(X, Y, U, V)",
         {"post": f"{CS}, {QV}", "require": ["contour", "quiver"]}),
        ("gradient", "Estimate the gradient numerically instead: `dLdy, dLdx = np.gradient(L, x, x)` (derivative along "
                     "rows, i.e. y, comes first) and draw the negative gradient with `ax.quiver`.",
         "dLdy, dLdx = np.gradient(L, x, x)\nfig, ax = plt.subplots()\nax.quiver(X, Y, -dLdx, -dLdy)",
         {"post": QV, "require": ["gradient", "quiver"]}),
        ("streamplot", f"Draw the streamlines of the descent field (the negative gradient) with `ax.streamplot`, "
                       f"colored by speed (`color=np.hypot(U, V)`) in {cm!r}, and add a colorbar for them "
                       f"(`fig.colorbar(sp.lines, ax=ax)`).",
         f"{grad}\nfig, ax = plt.subplots()\nsp = ax.streamplot(X, Y, U, V, color=np.hypot(U, V), cmap={cm!r})\n"
         f"fig.colorbar(sp.lines, ax=ax)", {"post": f"{SP}, {NCB}", "require": ["streamplot", "colorbar"]}),
        ("quiver", f"Draw filled contours of `L` (`levels={k}`) and over them only every 3rd descent arrow in both "
                   f"directions (`X[::3, ::3]`, …) with `ax.quiver`, in white.",
         f"{grad}\nfig, ax = plt.subplots()\nax.contourf(X, Y, L, levels={k})\n"
         f"ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], color='w')",
         {"post": f"{CS}, {QV}, [fc.hexcolor(q.get_facecolor()) for q in {QS}]", "require": ["contourf", "quiver"]}),
        ("colorbar", "Draw the descent arrows with `ax.quiver`, colored by the gradient's magnitude (fifth argument), "
                     "with a colorbar labelled '|grad|'.",
         f"{grad}\nfig, ax = plt.subplots()\nq = ax.quiver(X, Y, U, V, np.hypot(U, V))\nfig.colorbar(q, ax=ax, label='|grad|')",
         {"post": f"{QV}, [{_pix('q.get_array()')} for q in {QS}], fc.colorbars()", "require": ["quiver", "colorbar"]}),
        ("start_points", "Draw only the streamlines of the descent field that start at the 3 points in `starts` "
                         "(`start_points=starts`), over contour lines of `L` (`levels=8`).",
         f"{grad}\nfig, ax = plt.subplots()\nax.contour(X, Y, L, levels=8)\nax.streamplot(X, Y, U, V, start_points=starts)",
         {"post": f"{CS}, {SP}", "require": ["streamplot", "start_points"]}),
        ("pivot", "Show only directions: draw unit-length descent arrows (divide by the speed; the zero vector at the "
                  "minimum stays 0) with `ax.quiver`, centered on their grid points (`pivot='mid'`).",
         f"{grad}\ns = np.hypot(U, V)\ns[s == 0] = 1\nfig, ax = plt.subplots()\nax.quiver(X, Y, U / s, V / s, pivot='mid')",
         {"post": f"{QV}, [q.pivot for q in {QS}]", "require": ["quiver", "pivot"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def image_gallery(rng):
    n, (rows, cols) = rng.choice([(16, (4, 4)), (12, (3, 4)), (9, (3, 3)), (8, (2, 4))])
    h = rng.choice([8, 10, 12])
    name, labs = T.labels(rng, 3)
    k = rng.randint(3, min(4, len(labs)))
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nlabels = g.integers(0, {k}, {n})\n"
            f"imgs = np.round(g.random(({n}, {h}, {h})) * 0.5 + 0.1 * labels[:, None, None], 3)\n"
            f"preds = np.where(g.random({n}) < 0.7, labels, (labels + 1) % {k})\nclass_names = {lit(list(labs[:k]))}")
    desc = (f"`imgs` holds {n} images of {h} × {h} pixels, `labels` / `preds` their true / predicted classes (0 … "
            f"{k - 1}) and `class_names` the {name} names")
    data = _pix("a.images[0].get_array()") + " if a.images else None"
    items = [
        ("axis", f"Drill: show the {n} images on a {rows} × {cols} grid with no axes (`axis('off')` on every Axes).",
         f"fig, axs = plt.subplots({rows}, {cols})\nfor a, im in zip(axs.flat, imgs):\n    a.imshow(im)\n    a.axis('off')",
         {"post": _per_axes(f"({data}, a.axison)") + ", fc.grid_shape()", "require": ["axis"]}),
        ("set_title", f"Show the {n} images on a {rows} × {cols} grid in 'gray', each titled with its class name "
                      f"(`class_names[labels[i]]`), with no axes.",
         f"fig, axs = plt.subplots({rows}, {cols})\nfor i, a in enumerate(axs.flat):\n    a.imshow(imgs[i], cmap='gray')\n"
         f"    a.set_title(class_names[labels[i]])\n    a.axis('off')",
         {"post": _per_axes(f"({data}, a.get_title(), a.axison)"), "require": ["set_title"]}),
        ("colorbar", f"Show the {n} images on a {rows} × {cols} grid on one color scale (`vmin=0, vmax=1` for all) and "
                     f"add **one** colorbar for the whole grid (`fig.colorbar(im, ax=axs)`).",
         f"fig, axs = plt.subplots({rows}, {cols})\nfor a, x in zip(axs.flat, imgs):\n    im = a.imshow(x, vmin=0, vmax=1)\n"
         f"fig.colorbar(im, ax=axs)", {"post": _per_axes(f"({data}, fc.images(a)[0]['clim'])") + f", {NCB}"}),
        ("flatnonzero", "Show the misclassified images (where `preds != labels`, at most the first 4, found with "
                        "`np.flatnonzero`) in one row (`plt.subplots(1, m, squeeze=False)` or similar), each titled "
                        "'true→pred' with class names, e.g. 'cat→dog'. (There is at least one.)",
         "bad = np.flatnonzero(preds != labels)[:4]\nfig, axs = plt.subplots(1, len(bad), squeeze=False)\n"
         "for a, i in zip(axs[0], bad):\n    a.imshow(imgs[i])\n    a.set_title(f'{class_names[labels[i]]}→{class_names[preds[i]]}')",
         {"post": _per_axes(f"({data}, a.get_title())"), "require": ["flatnonzero|nonzero|where"],
          "setup": "preds[0] = (labels[0] + 1) % len(class_names)"}),
        ("mean", f"Show the **mean image** of each class in one row (`plt.subplots(1, {k})`), titled with the class "
                 f"name.", f"fig, axs = plt.subplots(1, {k})\nfor c, a in enumerate(axs):\n"
                           f"    a.imshow(imgs[labels == c].mean(axis=0))\n    a.set_title(class_names[c])",
         {"post": _per_axes(f"({data}, a.get_title())"), "require": ["mean"],
          "setup": f"labels[:{k}] = np.arange({k})"}),
        ("reshape", f"Tile the {n} images into **one** {rows * h} × {cols * h} mosaic array (`imgs.reshape({rows}, "
                    f"{cols}, {h}, {h}).transpose(0, 2, 1, 3).reshape({rows * h}, {cols * h})`: row-major, like the "
                    f"grid) and show it with a single `ax.imshow`, axes off.",
         f"mosaic = imgs.reshape({rows}, {cols}, {h}, {h}).transpose(0, 2, 1, 3).reshape({rows * h}, {cols * h})\n"
         f"fig, ax = plt.subplots()\nax.imshow(mosaic)\nax.axis('off')",
         {"post": f"{_im()}, fc.ax().axison", "require": ["reshape"]}),
        ("savefig", f"Show the {n} images on a {rows} × {cols} grid (`figsize=({cols}, {rows})`, axes off), save "
                    f"the figure as 'gallery.png' at 30 dpi and close it. Return `(plt.imread('gallery.png').shape, "
                    f"len(plt.get_fignums()))`.",
         f"fig, axs = plt.subplots({rows}, {cols}, figsize=({cols}, {rows}))\nfor a, im in zip(axs.flat, imgs):\n"
         f"    a.imshow(im)\n    a.axis('off')\nfig.savefig('gallery.png', dpi=30)\nplt.close(fig)\n"
         f"result = (plt.imread('gallery.png').shape, len(plt.get_fignums()))", {"require": ["savefig", "close"]}),
        ("std", "Show two summary images side by side: the per-pixel mean over all images (title 'mean') and the "
                "per-pixel standard deviation (title 'std').",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.imshow(imgs.mean(axis=0))\na1.set_title('mean')\n"
         "a2.imshow(imgs.std(axis=0))\na2.set_title('std')", {"post": _per_axes(f"({data}, a.get_title())"),
                                                             "require": ["mean", "std"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def hyperparam_grid(rng):
    lrs = rng.choice([[1e-4, 1e-3, 1e-2, 1e-1], [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], [1e-5, 1e-4, 1e-3]])
    bss = rng.choice([[16, 32, 64, 128], [8, 32, 128], [32, 64, 128, 256, 512]])
    s = seeded(rng)
    code = (f"lrs = np.array({lit(lrs)})\nbss = np.array({lit(bss)})\ng = np.random.default_rng({s})\n"
            f"scores = np.round(0.9 - 0.04 * (np.log10(lrs)[:, None] + 2.5) ** 2 - 0.02 * np.abs(np.log2(bss / 64)) "
            f"+ 0.01 * g.random(({len(lrs)}, {len(bss)})), 3)")
    desc = (f"`scores[i, j]` is the validation accuracy for learning rate `lrs[i]` and batch size `bss[j]` "
            f"({len(lrs)} × {len(bss)})")
    nl, nb = len(lrs), len(bss)
    ticks = "fc.ticks(fc.ax(), 'x')['labels'], fc.ticks(fc.ax(), 'y')['labels']"
    items = [
        ("pcolormesh", "Draw `scores` with `ax.pcolormesh(bss, lrs, scores, shading='nearest')` so each cell is "
                       "centered on its (batch size, learning rate) pair, and make both axes logarithmic.",
         "fig, ax = plt.subplots()\nax.pcolormesh(bss, lrs, scores, shading='nearest')\nax.set_xscale('log')\nax.set_yscale('log')",
         {"post": f"{QM}, fc.scales(), [fc.r(c.get_coordinates(), 6) for c in fc.ax().collections]",
          "require": ["pcolormesh", "shading"]}),
        ("imshow", f"Show `scores` with `ax.imshow`, with x ticks at the {nb} columns labelled by the batch sizes and "
                   f"y ticks at the {nl} rows labelled by the learning rates (`str(v)` of each).",
         f"fig, ax = plt.subplots()\nax.imshow(scores)\nax.set_xticks(range({nb}), labels=[str(b) for b in bss])\n"
         f"ax.set_yticks(range({nl}), labels=[str(l) for l in lrs])",
         {"post": f"{_im()}, {ticks}", "require": ["set_xticks", "set_yticks"]}),
        ("argmax", "Show `scores` with `ax.imshow` and mark the best cell with a red circle (`'ro'`) at its column "
                   "(x) and row (y): `np.unravel_index(scores.argmax(), scores.shape)`.",
         "fig, ax = plt.subplots()\nax.imshow(scores)\ni, j = np.unravel_index(scores.argmax(), scores.shape)\nax.plot(j, i, 'ro')",
         {"post": f"{_im()}, fc.lines(fc.ax(), 'color', 'marker')", "require": ["argmax"]}),
        ("text", f"Show `scores` with `ax.imshow` and write each score with 2 decimals centered in its cell "
                 f"(`f'{{v:.2f}}'`).",
         f"fig, ax = plt.subplots()\nax.imshow(scores)\nfor i in range({nl}):\n    for j in range({nb}):\n"
         f"        ax.text(j, i, f'{{scores[i, j]:.2f}}', ha='center', va='center')", {"post": f"{_im()}, {TXT}"}),
        ("contourf", "Interpolate between the runs: draw filled contours of `scores` over x = `np.log2(bss)` and "
                     "y = `np.log10(lrs)` with `levels=8`, and label the axes 'log2(batch size)' and 'log10(lr)'.",
         "fig, ax = plt.subplots()\nax.contourf(np.log2(bss), np.log10(lrs), scores, levels=8)\n"
         "ax.set(xlabel='log2(batch size)', ylabel='log10(lr)')",
         {"post": f"{CS}, fc.limits(), {LAB}", "require": ["contourf", "log2", "log10"]}),
        ("pcolormesh", f"Draw `scores` with `ax.pcolormesh` using explicit cell **edges** (`shading='flat'`): x edges "
                       f"`np.arange({nb + 1})`, y edges `np.arange({nl + 1})`; then put x ticks at the cell centers "
                       f"(`np.arange({nb}) + 0.5`) labelled by the batch sizes.",
         f"fig, ax = plt.subplots()\nax.pcolormesh(np.arange({nb + 1}), np.arange({nl + 1}), scores, shading='flat')\n"
         f"ax.set_xticks(np.arange({nb}) + 0.5, labels=[str(b) for b in bss])",
         {"post": f"{QM}, fc.ticks(fc.ax(), 'x'), fc.limits()", "require": ["pcolormesh"]}),
        ("orientation", "Show `scores` with `ax.imshow(scores, origin='lower')` and a **horizontal** colorbar labelled "
                        "'val acc' (`orientation='horizontal'`).",
         "fig, ax = plt.subplots()\nim = ax.imshow(scores, origin='lower')\n"
         "fig.colorbar(im, ax=ax, orientation='horizontal', label='val acc')",
         {"post": f"{_im('origin')}, fc.colorbars()", "require": ["colorbar", "orientation"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "medium")
def image_coordinates(rng):
    code, desc, h, w = _img(rng, sizes=((12, 16), (16, 12), (10, 10), (20, 15)))
    px = rng.choice([0.5, 2, 0.1])
    s = seeded(rng)
    r0, c0 = rng.randrange(h // 2), rng.randrange(w // 2)
    t = rng.choice([0.5, 0.6, 0.7])
    x0, x1, y0, y1 = rng.choice([(0, 4, 0, 3), (-1, 1, -1, 1), (10, 20, 0, 5)])
    k = rng.choice([3, 4, 5])
    code += (f"\ng = np.random.default_rng({s})\npts = np.round(g.uniform(0, 1, (6, 2)) * [{w * px}, {h * px}], 3)\n"
             f"heat = np.round(np.exp(-((np.arange({h})[:, None] - {h / 2}) ** 2 + (np.arange({w}) - {w / 3:.1f}) ** 2) "
             f"/ 20), 3)")
    desc += (f"; each pixel is {px} mm wide. `pts` holds 6 (x, y) positions in mm and `heat` a saliency map of the "
             f"same size as `img`")
    items = [
        ("extent", f"Show `img` in millimetres: `extent=[0, {w} * {px}, 0, {h} * {px}]` with `origin='lower'` (row 0 "
                   f"at y = 0), then draw the `pts` on top in red with `ax.scatter`.",
         f"fig, ax = plt.subplots()\nax.imshow(img, extent=[0, {w} * {px}, 0, {h} * {px}], origin='lower')\n"
         f"ax.scatter(pts[:, 0], pts[:, 1], color='red')",
         {"post": f"{_im('extent', 'origin')}, {OFF}, [fc.hexcolor(c.get_facecolor()) for c in fc.ax().collections]", "require": ["extent", "scatter"]}),
        ("alpha", "Overlay a saliency map: show `img` in 'gray', then `heat` on the same Axes in 'jet' at opacity "
                  "0.4.", "fig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nax.imshow(heat, cmap='jet', alpha=0.4)",
         {"post": f"{_im('cmap')}, [i.get_alpha() for i in fc.ax().images]", "require": ["alpha"]}),
        ("extent", "Place two images next to each other on **one** Axes: `img` over x 0 … 1 and `heat` over x 1 … 2 "
                   "(both y 0 … 1, via `extent`), then set the x limits to 0 … 2.",
         "fig, ax = plt.subplots()\nax.imshow(img, extent=[0, 1, 0, 1])\nax.imshow(heat, extent=[1, 2, 0, 1])\nax.set_xlim(0, 2)",
         {"post": f"{_im('extent')}, fc.limits()['xlim']", "require": ["extent"]}),
        ("argsort", f"Show `img` with `ax.imshow` and mark the centers of its {k} brightest pixels with red plus signs "
                    f"(`'r+'`): pixel (row r, col c) is centered at x = c, y = r. (Any order.)",
         f"fig, ax = plt.subplots()\nax.imshow(img)\nr, c = np.unravel_index(np.argsort(img, axis=None)[-{k}:], img.shape)\n"
         f"ax.plot(c, r, 'r+')",
         {"post": f"{_im()}, [sorted(zip(l['x'], l['y'])) + [l['color'], l['marker']] for l in fc.lines(fc.ax(), 'color', 'marker')]",
          "require": ["plot"]}),
        ("contour", f"Show `img` in 'gray' and outline where it is above {t}: `ax.contour(img, levels=[{t}], "
                    f"colors='r')` (contour without X, Y uses the same pixel coordinates as the image).",
         f"fig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nax.contour(img, levels=[{t}], colors='r')",
         {"post": f"{_im('cmap')}, {CS}", "require": ["imshow", "contour"]}),
        ("origin", f"`img` is shown with `extent=[{x0}, {x1}, {y0}, {y1}]` and `origin='lower'`. Return the data "
                   f"coordinates `[x, y]` of the center of pixel (row {r0}, column {c0}): pixels are "
                   f"({x1} - {x0}) / {w} wide and ({y1} - {y0}) / {h} tall, row 0 at the bottom.",
         f"result = [{x0} + ({c0} + 0.5) * ({x1} - {x0}) / {w}, {y0} + ({r0} + 0.5) * ({y1} - {y0}) / {h}]",
         {"require": [], "tol": 1e-9}),
        ("extent", f"Show `img` with `origin='upper'` (the default) and `extent=[{x0}, {x1}, {y0}, {y1}]` and return "
                   f"the data coordinates `[x, y]` of the center of pixel (row {r0}, column {c0}). (Read them off "
                   f"the formula, or check with `im.get_extent()`; row 0 is now at the **top**, y = {y1}.)",
         f"result = [{x0} + ({c0} + 0.5) * ({x1} - {x0}) / {w}, {y1} - ({r0} + 0.5) * ({y1} - {y0}) / {h}]",
         {"require": [], "tol": 1e-9}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= hard

@template(M, "hard")
def heatmap_helpers(rng):
    kind = rng.choice(["random", "corr", "confusion"])
    mcode, mdesc, n = matrix(rng, kind, n=(3, 5))
    rows = [f"r{i}" for i in range(n)]
    cols = [rng.choice("abcxyz") + str(i) for i in range(n)]
    icode, idesc, h, w = _img(rng, sizes=((8, 8), (6, 10), (10, 6)))
    fcode, fdesc, fn = _field(rng, sizes=(5, 7, 9))
    cm = rng.choice(["magma", "Blues", "viridis"])
    fmt = rng.choice(["{:.1f}", "{:.2f}", "{:.0f}"])
    imgs = f"imgs = np.round(np.random.default_rng({seeded(rng)}).random((5, {h}, {w})), 3)"
    items = [
        ("plt.gca", f"{mdesc}. Write `heatmap(data, row_labels, col_labels, ax=None, **kwargs)`: draw `data` with "
                    f"`ax.imshow(data, **kwargs)` on `ax` (the current Axes, `plt.gca()`, when None), put one tick per "
                    f"column / row labelled with `col_labels` / `row_labels`, and return the image. (Checked by calling "
                    f"it on fresh Axes, with and without `ax`.)",
         "def heatmap(data, row_labels, col_labels, ax=None, **kwargs):\n    if ax is None:\n        ax = plt.gca()\n"
         "    im = ax.imshow(data, **kwargs)\n    ax.set_xticks(range(data.shape[1]), labels=col_labels)\n"
         "    ax.set_yticks(range(data.shape[0]), labels=row_labels)\n    return im",
         {"setup": mcode, "require": ["gca", "imshow"],
          "post": f"_f, (_a, _b) = plt.subplots(1, 2)\n_i = heatmap(M, {rows!r}, {cols!r}, ax=_a, cmap={cm!r})\n"
                  f"plt.sca(_b)\n_j = heatmap(M[:, :2], {rows!r}, {cols[:2]!r})\n"
                  f"({_im('cmap', a='_a')}, fc.ticks(_a, 'x')['labels'], fc.ticks(_a, 'y')['labels'], "
                  f"{_im('cmap', a='_b')}, fc.ticks(_b, 'x')['labels'], _i in _a.images, _j in _b.images)"}),
        ("text", f"{mdesc}. Write `annotate_heatmap(im, fmt='{{:.2f}}', threshold=None)` that writes every value of an "
                 f"image `im` (from `im.get_array()`) centered in its cell on `im.axes`, formatted with `fmt.format(v)`, "
                 f"in 'white' where the value is above `threshold` and 'black' elsewhere (`threshold` defaults to the "
                 f"midpoint of the data's min and max). Return the list of Text objects.",
         "def annotate_heatmap(im, fmt='{:.2f}', threshold=None):\n    data = im.get_array()\n    if threshold is None:\n"
         "        threshold = (data.min() + data.max()) / 2\n    texts = []\n    for i in range(data.shape[0]):\n"
         "        for j in range(data.shape[1]):\n            color = 'white' if data[i, j] > threshold else 'black'\n"
         "            texts.append(im.axes.text(j, i, fmt.format(data[i, j]), ha='center', va='center', color=color))\n"
         "    return texts",
         {"setup": mcode, "require": ["text"],
          "post": f"_f, (_a, _b) = plt.subplots(1, 2)\n_t = annotate_heatmap(_a.imshow(M))\n"
                  f"_u = annotate_heatmap(_b.imshow(M), fmt={fmt!r}, threshold=float(np.median(M)))\n"
                  "(len(_t), len(_u), [sorted((t['xy'], t['text'], t['color']) for t in fc.text_items(a)) for a in (_a, _b)])"}),
        ("subplots", f"{idesc.replace('`img` is', 'Each image is')}. Write `show_images(imgs, titles=None, ncols=4)` "
                     f"that shows the images on a grid with `ncols` columns and as many rows as needed "
                     f"(`plt.subplots(nrows, ncols, squeeze=False)`), titles them when `titles` is given, turns the "
                     f"axes off on **every** Axes (also the unused ones at the end), and returns the figure.",
         "def show_images(imgs, titles=None, ncols=4):\n    nrows = -(-len(imgs) // ncols)\n"
         "    fig, axs = plt.subplots(nrows, ncols, squeeze=False)\n    for i, a in enumerate(axs.flat):\n"
         "        a.axis('off')\n        if i < len(imgs):\n            a.imshow(imgs[i])\n            if titles:\n"
         "                a.set_title(titles[i])\n    return fig",
         {"setup": imgs, "require": ["subplots"],
          "post": "_f = show_images(imgs, titles=list('abcde'), ncols=3)\n_g = show_images(imgs[:2])\n"
                  "[[(len(a.images), a.get_title(), a.axison) for a in f.axes] for f in (_f, _g)], fc.grid_shape(_f), "
                  "fc.grid_shape(_g), [" + _pix("a.images[0].get_array()") + " for a in _f.axes[:5]]"}),
        ("quiver", f"{fdesc}. Write `plot_field(X, Y, U, V, ax=None, normalize=False, **kwargs)` that draws the arrows "
                   f"with `ax.quiver(X, Y, U, V, **kwargs)` (current Axes when `ax` is None) and returns the Quiver; with "
                   f"`normalize=True` it first scales every nonzero arrow to length 1 (zero arrows stay 0).",
         "def plot_field(X, Y, U, V, ax=None, normalize=False, **kwargs):\n    if ax is None:\n        ax = plt.gca()\n"
         "    if normalize:\n        s = np.hypot(U, V)\n        s = np.where(s > 0, s, 1)\n        U, V = U / s, V / s\n"
         "    return ax.quiver(X, Y, U, V, **kwargs)",
         {"setup": fcode, "require": ["quiver"],
          "post": "_f, (_a, _b) = plt.subplots(1, 2)\n_q = plot_field(X, Y, U, V, ax=_a, color='r')\nplt.sca(_b)\n"
                  "_r = plot_field(X, Y, U, V, normalize=True)\n"
                  "[(fc.r(q.U, 4), fc.r(q.V, 4), fc.r(q.X, 4)) for q in (_q, _r)], fc.hexcolor(_q.get_facecolor()), "
                  "_q in _a.collections, _r in _b.collections"}),
        ("clabel", "`X`, `Y`, `Z` are a grid and a surface. Write `contour_figure(X, Y, Z, levels=10, ax=None)` that "
                   "draws filled contours (`levels`), black contour lines at the same levels (`levels=cf.levels`) "
                   "labelled with `clabel`, on `ax` (current Axes when None), and returns `(cf, cs)`.",
         "def contour_figure(X, Y, Z, levels=10, ax=None):\n    if ax is None:\n        ax = plt.gca()\n"
         "    cf = ax.contourf(X, Y, Z, levels=levels)\n    cs = ax.contour(X, Y, Z, levels=cf.levels, colors='k')\n"
         "    ax.clabel(cs)\n    return cf, cs",
         {"setup": _surface(rng, n=(15, 25))[0], "require": ["contourf", "clabel"],
          "post": "_f, (_a, _b) = plt.subplots(1, 2)\n_r = contour_figure(X, Y, Z, levels=5, ax=_a)\nplt.sca(_b)\n"
                  "_s = contour_figure(X, Y, Z)\n"
                  f"[({_cs('a')}, a.texts and fc.texts(a)) for a in (_a, _b)], [type(x).__name__ for x in _r], "
                  "fc.hexcolor(_s[1].get_edgecolor())"}),
        ("RdBu_r", "Write `diff_image(a, b, ax=None)` that shows the difference `b - a` of two images with "
                   "`imshow` in 'RdBu_r' with color limits symmetric around 0 (±the largest absolute difference, so "
                   "0 is white), adds a colorbar for it on the same figure, and returns the image. Use the current "
                   "Axes when `ax` is None.",
         "def diff_image(a, b, ax=None):\n    if ax is None:\n        ax = plt.gca()\n    d = b - a\n    m = np.abs(d).max()\n"
         "    im = ax.imshow(d, cmap='RdBu_r', vmin=-m, vmax=m)\n    ax.figure.colorbar(im, ax=ax)\n    return im",
         {"setup": imgs, "require": ["imshow", "colorbar"],
          "post": "_f, (_a, _b) = plt.subplots(1, 2)\n_i = diff_image(imgs[0], imgs[1], ax=_a)\nplt.sca(_b)\n"
                  "_j = diff_image(imgs[2], imgs[3] * 2)\n"
                  f"[{_im('cmap', 'clim', a='a')} for a in (_a, _b)], len(fc.colorbars(_f)), _i in _a.images"}),
    ]
    return catalog(rng, items, icode, "")


@template(M, "hard")
def debug_image_code(rng):
    h, w = rng.choice([(8, 10), (10, 8), (12, 12), (6, 9)])
    s = seeded(rng)
    nx, ny = rng.choice([(30, 20), (25, 15), (40, 30)])
    cm = rng.choice(["gray", "viridis", "magma"])
    base = f"g = np.random.default_rng({s})\nimg = np.round(g.random(({h}, {w})), 3)"
    items = [
        ("transpose", f"This crashes with *TypeError: Invalid shape (3, {h}, {w}) for image data*:\n```python\n"
                      f"fig, ax = plt.subplots()\nax.imshow(x)\n```\n`x` is a channel-first RGB tensor. Fix it.",
         "fig, ax = plt.subplots()\nax.imshow(x.transpose(1, 2, 0))",
         {"setup": f"x = np.round(g.random((3, {h}, {w})), 3)", "post": _im("shape"),
          "require": ["transpose|moveaxis"]}),
        (".T", f"This crashes with *TypeError: Length of x ({nx}) must match number of columns in z ({ny})*:\n"
               f"```python\nx = np.linspace(-2, 2, {nx})\ny = np.linspace(-1, 1, {ny})\nZ = x[:, None] ** 2 + 3 * y[None, :] ** 2\n"
               f"fig, ax = plt.subplots()\nax.contour(x, y, Z)\n```\n`contour` wants rows to follow y and columns to "
               f"follow x. Fix it (keep `x`, `y` and the function).",
         f"x = np.linspace(-2, 2, {nx})\ny = np.linspace(-1, 1, {ny})\nZ = x[:, None] ** 2 + 3 * y[None, :] ** 2\n"
         f"fig, ax = plt.subplots()\nax.contour(x, y, Z.T)", {"post": f"{CS}, fc.limits()", "require": ["contour"]}),
        (".colorbar", "This crashes with *AttributeError: 'Axes' object has no attribute 'cmap'*:\n```python\n"
                      f"fig, ax = plt.subplots()\nax.imshow(img, cmap={cm!r})\nfig.colorbar(ax)\n```\nA colorbar needs "
                      "the image (the *mappable*), not the Axes. Fix it.",
         f"fig, ax = plt.subplots()\nim = ax.imshow(img, cmap={cm!r})\nfig.colorbar(im, ax=ax)",
         {"post": f"{_im('cmap')}, {NCB}"}),
        ("shading", f"This crashes with *TypeError: Dimensions of C ({h}, {w}) should be one smaller than X({w}) and "
                    f"Y({h}) while using shading='flat'*:\n```python\nX, Y = np.meshgrid(np.arange({w}), np.arange({h}))\n"
                    f"fig, ax = plt.subplots()\nax.pcolormesh(X, Y, img, shading='flat')\n```\nKeep each cell "
                    f"**centered** on its grid point. Fix it.",
         f"X, Y = np.meshgrid(np.arange({w}), np.arange({h}))\nfig, ax = plt.subplots()\nax.pcolormesh(X, Y, img, shading='nearest')",
         {"post": f"{QM}, [fc.r(c.get_coordinates(), 4) for c in fc.ax().collections]",
          "require": ["pcolormesh", "shading"]}),
        ("imshow", "This shows an almost white image and logs *Clipping input data to the valid range for imshow with "
                   "RGB data ([0..1] for floats or [0..255] for integers)*:\n```python\nrgb = np.dstack([img, img ** 2, "
                   "1 - img]) * 255\nfig, ax = plt.subplots()\nax.imshow(rgb)\n```\nFix it by scaling the floats back "
                   "to [0, 1] (divide by 255).",
         "rgb = np.dstack([img, img ** 2, 1 - img]) * 255\nfig, ax = plt.subplots()\nax.imshow(rgb / 255)",
         {"post": _im("shape"), "tol": 1e-6}),
        ("origin", f"`Z[i, j]` is a function sampled at x = `x[j]`, y = `y[i]` (row 0 is y = 0):\n```python\nx = "
                   f"np.linspace(0, 4, {w})\ny = np.linspace(0, 2, {h})\nX, Y = np.meshgrid(x, y)\nZ = np.sin(X) * Y\n"
                   f"fig, ax = plt.subplots()\nax.imshow(Z)\n```\nThe picture is upside down and the axes show pixel "
                   f"indices. Fix it: row 0 at the bottom and the axes spanning x 0 … 4, y 0 … 2.",
         f"x = np.linspace(0, 4, {w})\ny = np.linspace(0, 2, {h})\nX, Y = np.meshgrid(x, y)\nZ = np.sin(X) * Y\n"
         f"fig, ax = plt.subplots()\nax.imshow(Z, origin='lower', extent=[0, 4, 0, 2])",
         {"post": _im("origin", "extent"), "require": ["origin", "extent"]}),
        ("extent", "This should place the image over x 0 … 4, y 0 … 2, but it comes out squashed to a line (with a "
                   "warning about identical limits):\n```python\nfig, ax = plt.subplots()\nax.imshow(img, extent=[0, 0, "
                   "4, 2])\n```\n`extent` is not (xmin, ymin, xmax, ymax). Fix it.",
         "fig, ax = plt.subplots()\nax.imshow(img, extent=[0, 4, 0, 2])", {"post": _im("extent")}),
        ("vmin", "These two panels look **identical** although the right image is half as bright, because each image "
                 "gets its own color scale:\n```python\nfig, (a1, a2) = plt.subplots(1, 2)\na1.imshow(img)\n"
                 "a2.imshow(img * 0.5)\n```\nFix it: put both on the scale 0 … 1.",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.imshow(img, vmin=0, vmax=1)\na2.imshow(img * 0.5, vmin=0, vmax=1)",
         {"post": _per_axes("fc.images(a)[0]['clim']") + ", " + _per_axes(_pix("a.images[0].get_array()")),
          "require": ["vmin", "vmax"]}),
        ("quiver", "This draws the wrong arrows, silently: with three arrays `quiver` reads them as U, V and the color "
                   "array, so the positions become the arrows.\n```python\nX, Y = np.meshgrid(np.linspace(-1, 1, 5), "
                   "np.linspace(-1, 1, 5))\nU, V = -Y, X\nfig, ax = plt.subplots()\nax.quiver(X, Y, U)\n```\nFix it so "
                   "the arrow (U, V) is drawn at each (X, Y).",
         "X, Y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))\nU, V = -Y, X\nfig, ax = plt.subplots()\n"
         "ax.quiver(X, Y, U, V)", {"post": f"{QV}, [q.get_array() is None for q in {QS}]"}),
        ("matshow", "This leaves the two subplots **empty** and opens two more figures, because `plt.matshow` always "
                    "makes its own figure:\n```python\nfig, (a1, a2) = plt.subplots(1, 2)\nplt.matshow(img)\n"
                    "plt.matshow(img.T)\n```\nFix it: one figure, `img` on the left Axes, `img.T` on the right.",
         "fig, (a1, a2) = plt.subplots(1, 2)\na1.matshow(img)\na2.matshow(img.T)",
         {"post": _per_axes(_pix("a.images[0].get_array()")) + ", len(plt.get_fignums())"}),
    ]
    return catalog(rng, items, base, "")


@template(M, "hard")
def panel_specs(rng):
    s = seeded(rng)
    k = rng.choice([3, 4, 5])
    h = rng.choice([8, 10, 12])
    kinds = "[(len(a.images), sorted(type(c).__name__ for c in a.collections)) for a in fc.axes()]"
    data = "[" + _pix("a.images[0].get_array()") + " if a.images else None for a in fc.axes()]"
    items = [
        ("subplots", "Build the guide's 2 × 3 figure (`plt.subplots(2, 3, figsize=(9, 6), layout='constrained')`): [0, 0] `img` in "
                     "'gray' (limits 0 … 1, `interpolation='nearest'`) with a colorbar; [0, 1] the RGB image `rgb`; "
                     "[0, 2] `C` with `matshow` in 'RdBu_r' (limits -1 … 1); [1, 0] `Z` over `X`, `Y` with "
                     "`pcolormesh` (`shading='auto'`) and a colorbar; [1, 1] filled contours of `Z` (`levels=20`, "
                     "'coolwarm') with black contour lines (`levels=8`) labelled by `clabel`, and a colorbar for the "
                     "filled ones; [1, 2] arrows `quiver(P, Q, -Q, P)` colored by their length and gray streamlines "
                     "`streamplot(P, Q, -Q, P, density=0.8)`.",
         "fig, axs = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')\n"
         "im = axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')\nfig.colorbar(im, ax=axs[0, 0])\n"
         "axs[0, 1].imshow(rgb)\naxs[0, 2].matshow(C, cmap='RdBu_r', vmin=-1, vmax=1)\n"
         "pc = axs[1, 0].pcolormesh(X, Y, Z, shading='auto')\nfig.colorbar(pc, ax=axs[1, 0])\n"
         "cf = axs[1, 1].contourf(X, Y, Z, levels=20, cmap='coolwarm')\n"
         "cs = axs[1, 1].contour(X, Y, Z, levels=8, colors='k', linewidths=0.5)\naxs[1, 1].clabel(cs, inline=True, fontsize=7)\n"
         "fig.colorbar(cf, ax=axs[1, 1])\naxs[1, 2].quiver(P, Q, -Q, P, np.hypot(P, Q))\n"
         "axs[1, 2].streamplot(P, Q, -Q, P, density=0.8, color='gray', linewidth=0.6)",
         {"setup": f"g = np.random.default_rng({s})\nimg = np.round(g.random(({h}, {h})), 3)\n"
                   f"rgb = np.round(g.random(({h}, {h}, 3)), 3)\nC = np.round(np.corrcoef(g.normal(size=(5, 50))), 3)\n"
                   "X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))\n"
                   "Z = np.exp(-(X ** 2 + Y ** 2)) - 0.5 * np.exp(-((X - 1) ** 2 + (Y - 1) ** 2))\n"
                   "P, Q = np.meshgrid(np.linspace(-2, 2, 11), np.linspace(-2, 2, 11))",
          "post": f"{kinds}, {data}, fc.grid_shape(), fc.size(), {NCB}, " + _per_axes("[c.get_cmap().name for c in a.collections "
                                                                        "if hasattr(c, 'allsegs')]"),
          "require": ["subplots", "pcolormesh", "clabel", "quiver", "streamplot"]}),
        ("set_ylabel", f"An autoencoder's results: `x` holds {k} inputs and `xr` their reconstructions ({h} × {h}). "
                       f"Draw a 3 × {k} grid: inputs on row 0, reconstructions on row 1, and the absolute error "
                       f"`np.abs(x - xr)` on row 2 in 'hot'. Remove all ticks (`set_xticks([])`, `set_yticks([])`) and "
                       f"label the rows with `set_ylabel` on the first column: 'input', 'recon', 'error'.",
         f"fig, axs = plt.subplots(3, {k})\nfor j in range({k}):\n    axs[0, j].imshow(x[j])\n    axs[1, j].imshow(xr[j])\n"
         f"    axs[2, j].imshow(np.abs(x[j] - xr[j]), cmap='hot')\nfor a in axs.flat:\n    a.set_xticks([])\n"
         f"    a.set_yticks([])\nfor a, name in zip(axs[:, 0], ['input', 'recon', 'error']):\n    a.set_ylabel(name)",
         {"setup": f"g = np.random.default_rng({s})\nx = np.round(g.random(({k}, {h}, {h})), 3)\n"
                   f"xr = np.round(np.clip(x + g.normal(0, 0.1, x.shape), 0, 1), 3)",
          "post": f"{data}, " + _per_axes("(a.get_ylabel(), len(a.get_xticks()) + len(a.get_yticks()), "
                                         "fc.images(a)[0]['cmap'])") + ", fc.grid_shape()",
          "require": ["set_ylabel", "set_xticks"]}),
        ("sharex", "A potential `Pot = X ** 2 + 2 * Y ** 2` on the grid `X, Y`: draw a row of 3 Axes sharing both "
                   "axes (`plt.subplots(1, 3, sharex=True, sharey=True)`): filled contours of `Pot` (`levels=10`) "
                   "titled 'potential'; arrows of its negative gradient (-2X, -4Y) with `quiver`, titled 'gradient'; "
                   "streamlines of the same field, titled 'flow'.",
         "Pot = X ** 2 + 2 * Y ** 2\nfig, axs = plt.subplots(1, 3, sharex=True, sharey=True)\n"
         "axs[0].contourf(X, Y, Pot, levels=10)\naxs[0].set_title('potential')\naxs[1].quiver(X, Y, -2 * X, -4 * Y)\n"
         "axs[1].set_title('gradient')\naxs[2].streamplot(X, Y, -2 * X, -4 * Y)\naxs[2].set_title('flow')",
         {"setup": "X, Y = np.meshgrid(np.linspace(-2, 2, 13), np.linspace(-2, 2, 13))",
          "post": f"{kinds}, " + _per_axes("a.get_title()") + ", fc.ax(0).get_shared_x_axes().joined(fc.ax(0), fc.ax(2)), "
                  "fc.ax(0).get_shared_y_axes().joined(fc.ax(0), fc.ax(2)), "
                  "[fc.r(q.U, 4) for q in fc.ax(1).collections]",
          "require": ["contourf", "quiver", "streamplot", "sharex|sharey"]}),
        ("spy", f"A pruned weight matrix `W` ({h} × {2 * h}, mostly zeros): draw two Axes side by side: the sparsity "
                f"pattern (`spy(W)`, title 'nonzeros') and the values (`imshow` in 'RdBu_r' with limits ±`np.abs(W).max()`, "
                f"title 'weights') with a colorbar on that one.",
         "m = np.abs(W).max()\nfig, (a1, a2) = plt.subplots(1, 2)\na1.spy(W)\na1.set_title('nonzeros')\n"
         "im = a2.imshow(W, cmap='RdBu_r', vmin=-m, vmax=m)\na2.set_title('weights')\nfig.colorbar(im, ax=a2)",
         {"setup": f"g = np.random.default_rng({s})\nW = np.round(g.normal(size=({h}, {2 * h})) * (g.random(({h}, {2 * h})) < 0.2), 3)",
          "post": f"{data}, " + _per_axes("(a.get_title(), fc.images(a)[0]['clim'])") + f", {NCB}",
          "require": ["spy", "imshow", "colorbar"]}),
        ("suptitle", f"A conv layer's {2 * k} feature maps `fmap` ({2 * k} × {h} × {h}): show them on a 2 × {k} grid "
                     f"on one color scale (limits `fmap.min()` … `fmap.max()`), axes off, with one colorbar for the "
                     f"whole grid and the figure title 'layer 1'.",
         f"fig, axs = plt.subplots(2, {k})\nfor a, f in zip(axs.flat, fmap):\n    im = a.imshow(f, vmin=fmap.min(), vmax=fmap.max())\n"
         f"    a.axis('off')\nfig.colorbar(im, ax=axs)\nfig.suptitle('layer 1')",
         {"setup": f"fmap = np.round(np.random.default_rng({s}).normal(size=({2 * k}, {h}, {h})), 3)",
          "post": f"{data}, " + _per_axes("(fc.images(a)[0]['clim'], a.axison)") + f", {NCB}, fc.suptitles()['suptitle']",
          "require": ["suptitle", "colorbar"]}),
    ]
    return catalog(rng, items, "", "")


@template(M, "hard")
def landscape_drills(rng):
    a, b = rng.choice([(1, 4), (1, 3), (0.5, 2), (2, 5)])
    lr = rng.choice([0.05, 0.08, 0.1])
    lrs = rng.choice([(0.05, 0.2), (0.02, 0.15), (0.1, 0.22)])
    steps = rng.randint(8, 15)
    w0 = rng.choice([(-1.8, 1.5), (1.5, 1.8), (-1.5, -1.6), (1.9, -1.2)])
    t = rng.choice([0.5, 1.0, 2.0])
    beta = rng.choice([0.5, 0.8, 0.9])
    k = rng.choice([8, 10, 12])
    code = (f"w1, w2 = np.meshgrid(np.linspace(-2, 2, 35), np.linspace(-2, 2, 35))\nL = {a} * w1 ** 2 + {b} * w2 ** 2\n"
            f"w0 = np.array({list(w0)})")
    desc = (f"`L` = {a}·w1² + {b}·w2² is a quadratic loss on the 35 × 35 grids `w1`, `w2` (gradient: "
            f"({2 * a}·w1, {2 * b}·w2)); `w0` is a starting point")
    gd = (f"def descend(lr):\n    w = w0.astype(float)\n    path = [w.copy()]\n    for _ in range({steps}):\n"
          f"        w = w - lr * np.array([{2 * a} * w[0], {2 * b} * w[1]])\n        path.append(w.copy())\n"
          f"    return np.array(path)\n")
    items = [
        ("plot", f"Run {steps} steps of gradient descent from `w0` with learning rate {lr} (w ← w − lr·∇L), keeping "
                 f"every point (so {steps + 1} with the start). Draw contour lines of `L` (`levels={k}`) and the path "
                 f"on top as a red line with circle markers (`'ro-'`).",
         f"{gd}path = descend({lr})\nfig, ax = plt.subplots()\nax.contour(w1, w2, L, levels={k})\n"
         f"ax.plot(path[:, 0], path[:, 1], 'ro-')",
         {"post": f"{CS}, fc.lines(fc.ax(), 'color', 'marker')", "require": ["contour", "plot"]}),
        ("subplots", f"Compare learning rates {lrs[0]} and {lrs[1]}: {steps} gradient-descent steps from `w0` each "
                     f"({steps + 1} points with the start), drawn on two Axes side by side (`plt.subplots(1, 2)`), "
                     f"each with contour lines of `L` (`levels={k}`), the path as `'o-'`, and the title "
                     f"`f'lr={{lr}}'`.",
         f"{gd}fig, axs = plt.subplots(1, 2)\nfor a, lr in zip(axs, {list(lrs)}):\n    path = descend(lr)\n"
         f"    a.contour(w1, w2, L, levels={k})\n    a.plot(path[:, 0], path[:, 1], 'o-')\n    a.set_title(f'lr={{lr}}')",
         {"post": _per_axes(f"({_cs('a')}, fc.lines(a), a.get_title())")}),
        ("legend", f"Compare plain gradient descent with momentum (β = {beta}: v ← β·v + ∇L, w ← w − lr·v, v starts "
                   f"at 0), both {steps} steps from `w0` with lr {lr} ({steps + 1} points each). Draw contour lines of "
                   f"`L` (`levels={k}`) and both paths, labelled 'gd' and 'momentum', with a legend.",
         f"{gd}def momentum(lr, beta):\n    w = w0.astype(float)\n    v = np.zeros(2)\n    path = [w.copy()]\n"
         f"    for _ in range({steps}):\n        v = beta * v + np.array([{2 * a} * w[0], {2 * b} * w[1]])\n"
         f"        w = w - lr * v\n        path.append(w.copy())\n    return np.array(path)\n\n"
         f"fig, ax = plt.subplots()\nax.contour(w1, w2, L, levels={k})\np, m = descend({lr}), momentum({lr}, {beta})\n"
         f"ax.plot(p[:, 0], p[:, 1], label='gd')\nax.plot(m[:, 0], m[:, 1], label='momentum')\nax.legend()",
         {"post": f"{CS}, fc.lines(fc.ax(), 'label'), fc.legend()", "require": ["legend"]}),
        ("allsegs", f"Draw the single contour `cs = ax.contour(w1, w2, L, levels=[{t}])` and return the **area** "
                    f"enclosed by its curve, from the curve's vertices `v = cs.allsegs[0][0]` with the shoelace "
                    f"formula: ½·|Σ (x_i·y_(i+1) − x_(i+1)·y_i)| (wrapping around). (The exact ellipse has area "
                    f"π·{t}/√({a}·{b}); the polygon is a little smaller.)",
         f"fig, ax = plt.subplots()\ncs = ax.contour(w1, w2, L, levels=[{t}])\nv = cs.allsegs[0][0]\nx, y = v[:, 0], v[:, 1]\n"
         f"result = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))",
         {"require": ["allsegs"], "tol": 1e-6}),
        ("clabel", f"Draw contour lines `cs` of `L` with `levels={k}` and label only every other level "
                   f"(`levels=cs.levels[::2]` in `clabel`), with labels like 'L=1.5' "
                   f"(`fmt=lambda v: f'L={{v:.1f}}'`).",
         f"fig, ax = plt.subplots()\ncs = ax.contour(w1, w2, L, levels={k})\n"
         f"ax.clabel(cs, levels=cs.levels[::2], fmt=lambda v: f'L={{v:.1f}}')",
         {"post": f"{CS}, sorted(fc.texts())", "require": ["clabel"]}),
        ("gradient", f"Plot how steep the loss is: estimate the gradient with `np.gradient(L, x, x)` (with `x = "
                     f"np.linspace(-2, 2, 35)`; it returns d/dw2 first, then d/dw1), draw filled contours of its norm "
                     f"`np.hypot(...)` with `levels={k}` in 'magma', and overlay the gradient-descent path ({steps} "
                     f"steps from `w0`, lr {lr}) as a white line with markers (`'wo-'`).",
         f"{gd}x = np.linspace(-2, 2, 35)\ngy, gx = np.gradient(L, x, x)\nfig, ax = plt.subplots()\n"
         f"ax.contourf(w1, w2, np.hypot(gx, gy), levels={k}, cmap='magma')\npath = descend({lr})\n"
         f"ax.plot(path[:, 0], path[:, 1], 'wo-')",
         {"post": f"{CS}, fc.lines(fc.ax(), 'color', 'marker')", "require": ["gradient", "hypot"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "hard")
def image_processing_views(rng):
    h, w = rng.choice([(8, 12), (12, 12), (12, 8), (16, 16)])
    p = 4
    f = rng.choice([2, 3, 4])
    t = rng.choice([0.2, 0.25, 0.3])
    ts = rng.choice([[0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.7], [0.25, 0.5, 0.75]])
    s = seeded(rng)
    code = (f"g = np.random.default_rng({s})\nimg = np.round(g.random(({h}, {w})), 3)\n"
            f"heat = np.round(np.exp(-((np.arange({h})[:, None] - {h // 2}) ** 2 + (np.arange({w}) - {w // 3}) ** 2) / 10), 3)")
    desc = f"`img` is a {h} × {w} grayscale image (values in [0, 1)) and `heat` a saliency map of the same size"
    data = "[" + _pix("a.images[0].get_array()") + " if a.images else None for a in fc.axes()]"
    items = [
        ("diff", f"Edge detection in a row of 3 Axes: `img` in 'gray' (title 'image'); its horizontal gradient "
                 f"`np.diff(img, axis=1)` in 'RdBu_r' with limits ± its largest absolute value (title 'd/dx'); and "
                 f"`spy` of where that gradient's absolute value is above {t} (title 'edges').",
         f"d = np.diff(img, axis=1)\nm = np.abs(d).max()\nfig, axs = plt.subplots(1, 3)\naxs[0].imshow(img, cmap='gray')\n"
         f"axs[0].set_title('image')\naxs[1].imshow(d, cmap='RdBu_r', vmin=-m, vmax=m)\naxs[1].set_title('d/dx')\n"
         f"axs[2].spy(np.abs(d) > {t})\naxs[2].set_title('edges')",
         {"post": f"{data}, " + _per_axes("(a.get_title(), fc.images(a)[0]['cmap'], fc.images(a)[0]['clim'])"),
          "require": ["diff", "spy"]}),
        ("kron", f"Upsample `img` {f}× by repeating every pixel into a {f} × {f} block (`np.kron(img, np.ones(({f}, "
                 f"{f})))`) and show the original and the upsampled image side by side, both with "
                 f"`interpolation='nearest'`.",
         f"big = np.kron(img, np.ones(({f}, {f})))\nfig, (a1, a2) = plt.subplots(1, 2)\na1.imshow(img, interpolation='nearest')\n"
         f"a2.imshow(big, interpolation='nearest')",
         {"post": f"{data}, " + _per_axes("[i.get_interpolation() for i in a.images]"), "require": ["kron|repeat"]}),
        ("reshape", f"ViT-style patches: cut `img` into {p} × {p} patches and show them on a {h // p} × {w // p} grid "
                    f"in reading order (patch (r, c) = `img[r*{p}:(r+1)*{p}, c*{p}:(c+1)*{p}]` on Axes (r, c)), all on "
                    f"the scale 0 … 1, axes off.",
         f"fig, axs = plt.subplots({h // p}, {w // p})\nfor r in range({h // p}):\n    for c in range({w // p}):\n"
         f"        axs[r, c].imshow(img[r * {p}:(r + 1) * {p}, c * {p}:(c + 1) * {p}], vmin=0, vmax=1)\n"
         f"        axs[r, c].axis('off')",
         {"post": f"{data}, " + _per_axes("(fc.images(a)[0]['clim'], a.axison)") + ", fc.grid_shape()",
          "require": ["imshow", "axis"]}),
        ("alpha", "Saliency overlay: show `img` in 'gray', then `heat` on top in 'jet' at opacity 0.5, and add a "
                  "colorbar for the **heat** map only.",
         "fig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nim = ax.imshow(heat, cmap='jet', alpha=0.5)\nfig.colorbar(im, ax=ax)",
         {"post": f"{_im('cmap')}, [i.get_alpha() for i in fc.ax().images], {NCB}, "
                  "[" + _pix("a._colorbar.mappable.get_array()") + " for a in fc.fig().axes if hasattr(a, '_colorbar')]",
          "require": ["alpha", "colorbar"]}),
        ("max", f"2 × 2 max-pooling: reduce `img` to {h // 2} × {w // 2} by taking the max of every 2 × 2 block "
                f"(`img.reshape({h // 2}, 2, {w // 2}, 2).max(axis=(1, 3))`) and show the original and the pooled "
                f"image side by side, **both** with `extent=[0, {w}, {h}, 0]` so they cover the same coordinates.",
         f"pooled = img.reshape({h // 2}, 2, {w // 2}, 2).max(axis=(1, 3))\nfig, (a1, a2) = plt.subplots(1, 2)\n"
         f"a1.imshow(img, extent=[0, {w}, {h}, 0])\na2.imshow(pooled, extent=[0, {w}, {h}, 0])",
         {"post": f"{data}, " + _per_axes("fc.images(a)[0]['extent']"), "require": ["reshape", "extent"]}),
        ("spy", f"Threshold sweep: one Axes per threshold in {ts} (a single row), each showing `spy(img > t)` titled "
                f"`f't={{t}}'`.",
         f"ts = {ts}\nfig, axs = plt.subplots(1, {len(ts)})\nfor a, t in zip(axs, ts):\n    a.spy(img > t)\n    a.set_title(f't={{t}}')",
         {"post": f"{data}, " + _per_axes("a.get_title()"), "require": ["spy"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


# ================================================================= challenge

@template(M, "challenge")
def contour_crossings(rng):
    R, C, q, H, s = rng.randint(4, 6), rng.randint(4, 7), rng.randint(3, 5), rng.choice([6, 9, 12]), seeded(rng)
    report = rng.choice(["edges", "cells", "segments"])
    setup = (f"g = np.random.default_rng({s})\nZ = g.integers(0, {H}, ({R}, {C}))\nlevels = g.integers(0, {H + 1}, {q})\n"
             f"report = {report!r}")
    sol = ("def crossed(lo, hi, t):\n    \"\"\"For every t: how many items have lo < t <= hi.\"\"\"\n"
           "    return np.searchsorted(np.sort(lo.ravel()), t) - np.searchsorted(np.sort(hi.ravel()), t)\n\n"
           "def edges(a, b):\n    return crossed(np.minimum(a, b), np.maximum(a, b), levels)\n\n")
    sol += {"edges": "result = edges(Z[:, :-1], Z[:, 1:]) + edges(Z[:-1, :], Z[1:, :])",
            "cells": "corners = [Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]]\n"
                     "result = crossed(np.minimum.reduce(corners), np.maximum.reduce(corners), levels)",
            "segments": "inner = edges(Z[1:-1, :-1], Z[1:-1, 1:]) + edges(Z[:-1, 1:-1], Z[1:, 1:-1])\n"
                        "outer = edges(Z[[0, -1], :-1], Z[[0, -1], 1:]) + edges(Z[:-1, [0, -1]], Z[1:, [0, -1]])\n"
                        "result = inner + outer // 2   # every inner edge borders 2 cells, a border edge 1"}[report]
    tests = [f"g = np.random.default_rng({s + i})\nZ = g.integers(0, {hh}, ({rr}, {cc}))\n"
             f"levels = g.integers(0, {hh + 1}, 200000)\nreport = {report!r}"
             for i, (rr, cc, hh) in ((1, (700, 650, 1000)), (2, (300, 900, 50)))]
    want = {"edges": "how many grid edges (between horizontally or vertically adjacent nodes) the level crosses",
            "cells": "how many grid cells (squares of 4 neighboring nodes) the contour passes through, i.e. cells whose "
                     "corners are not all on the same side",
            "segments": "how many line segments marching squares draws: in each cell, one segment per pair of crossed "
                        "cell edges (a cell has 0, 2 or 4 crossed edges; 4 is a saddle, drawn as 2 segments)"}[report]
    return challenge(
        "Where contour lines cross the grid",
        "`contour` runs *marching squares* on a grid of heights `Z` (R rows × C columns of nodes). For a level `t`, "
        "a node is *above* when `Z >= t` and *below* when `Z < t`; an edge between two adjacent nodes is crossed "
        "when one end is above and the other below, and the contour line passes through every crossed edge. For "
        f"every level in `levels`, count {want}. Return the counts as an integer array, one per level. "
        f"(`report` = {report!r}.)",
        setup, sol, tests, "2 ≤ R, C, R·C ≤ 10^6, len(levels) ≤ 2·10^5, heights and levels are integers",
        speed="O((R·C + q) log(R·C)): an edge is crossed exactly when min < t ≤ max, so sort the minima and maxima "
              "once and answer each level with two binary searches", answer_desc=" (an int array)")


@template(M, "challenge")
def colormap_lookup(rng):
    n, N, s = rng.randint(8, 12), rng.choice([4, 5, 8, 10]), seeded(rng)
    report = rng.choice(["auto", "fixed", "log"])
    if report == "log":
        data = (f"v = np.round(g.lognormal(0, 1.5, {n}) - 0.3, 3)\nv[g.random({n}) < 0.15] = np.nan\n"
                f"vmin, vmax = {rng.choice([0.1, 0.2, 0.5])}, {rng.choice([5.0, 8.0, 10.0])}")
    elif report == "fixed":
        data = (f"v = np.round(g.normal(0, 2, {n}), 3)\nv[g.random({n}) < 0.15] = np.nan\n"
                f"vmin, vmax = {rng.choice([-2.5, -2.0, -1.5])}, {rng.choice([1.5, 2.0, 3.0])}")
    else:
        data = f"v = np.round(g.normal(0, 2, {n}), 3)\nv[g.random({n}) < 0.15] = np.nan\nvmin, vmax = None, None"
    setup = f"g = np.random.default_rng({s})\n{data}\nN = {N}\nreport = {report!r}"
    sol = ("def lut_index(x):\n    xa = x * N\n    xa[xa == N] = N - 1          # x == 1 is the top color, not 'over'\n"
           "    idx = np.full(len(xa), N + 2)  # 'bad'\n    ok = ~np.isnan(xa)\n    idx[ok] = xa[ok].astype(int)\n"
           "    idx[ok & (xa < 0)] = N         # 'under'\n    idx[ok & (xa >= N)] = N + 1    # 'over'\n    return idx\n\n")
    sol += {"auto": "lo, hi = np.nanmin(v), np.nanmax(v)\nresult = lut_index((v - lo) / (hi - lo))",
            "fixed": "idx = lut_index((v - vmin) / (vmax - vmin))\nresult = np.bincount(idx, minlength=N + 3)",
            "log": "with np.errstate(divide='ignore', invalid='ignore'):\n"
                   "    x = (np.log10(v) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))\n"
                   "x[~(v > 0)] = np.nan               # no logarithm: drawn as 'bad'\nresult = lut_index(x)"}[report]
    big = {"log": "v = np.round(g.lognormal(0, 1.5, 200000) - 0.3, 3)\nv[g.random(200000) < 0.1] = np.nan\n"
                  "vmin, vmax = 0.05, 20.0",
           "fixed": "v = np.round(g.normal(0, 2, 200000), 3)\nv[g.random(200000) < 0.1] = np.nan\nvmin, vmax = -3.0, 2.5",
           "auto": "v = np.round(g.normal(0, 2, 200000), 3)\nv[g.random(200000) < 0.1] = np.nan\nvmin, vmax = None, None"}
    tests = [f"g = np.random.default_rng({s + i})\n{big[report]}\nN = {NN}\nreport = {report!r}"
             for i, NN in ((1, 256), (2, 7))]
    norm = {"auto": "`vmin` and `vmax` are None: like `imshow`, use the smallest and largest non-NaN value of `v`; "
                    "x = (v − vmin) / (vmax − vmin)",
            "fixed": "x = (v − vmin) / (vmax − vmin) with the given limits",
            "log": "this is a `LogNorm`: x = (log10(v) − log10(vmin)) / (log10(vmax) − log10(vmin)); values ≤ 0 "
                   "have no logarithm and are drawn in the bad color like NaN"}[report]
    want = {"auto": "the table index of every value (an int array like `v`)",
            "fixed": "how many values land on each of the N + 3 table entries (an int array of length N + 3)",
            "log": "the table index of every value (an int array like `v`)"}[report]
    return challenge(
        "Colormap lookup",
        "A colormap with `N` colors is a lookup table with N + 3 rows: colors 0 … N − 1, then *under* (N), *over* "
        "(N + 1) and *bad* (N + 2). To color a value, a norm maps it to x (" + norm + "), then Matplotlib computes, "
        "in float64 and in this order, `xa = x * N`; sets `xa = N - 1` where `xa == N` (so x = 1 is the top color); "
        "and picks index `int(xa)` (truncation) if 0 ≤ xa < N, *under* if xa < 0, *over* if xa ≥ N and *bad* "
        f"if the value is NaN. Return {want}. (`report` = {report!r}.)",
        setup, sol, tests, "len(v) ≤ 2·10^5, N ≤ 1024",
        speed="O(n) with vectorized NumPy (compute x exactly as written, so values on a color boundary agree)",
        answer_desc=" (an int array)")


@template(M, "challenge")
def spy_blocks(rng):
    n, B, m, s = rng.randint(12, 24), rng.randint(3, 5), rng.randint(8, 16), seeded(rng)
    report = rng.choice(["dark", "busiest", "rows"])
    setup = (f"g = np.random.default_rng({s})\nn, B = {n}, {B}\nrows = g.integers(0, n, {m})\ncols = g.integers(0, n, {m})\n"
             f"report = {report!r}")
    sol = "s = -(-n // B)                       # block side: ceil(n / B)\n"
    sol += {"dark": "blocks = (rows // s) * B + cols // s\nresult = len(np.unique(blocks))",
            "busiest": "cells = np.unique(rows.astype(np.int64) * n + cols)   # distinct nonzeros\n"
                       "r, c = cells // n, cells % n\nresult = int(np.unique((r // s) * B + c // s, return_counts=True)[1].max())",
            "rows": "blocks = np.unique((rows // s) * B + cols // s)\nresult = np.bincount(blocks // B, minlength=B)"}[report]
    tests = [f"g = np.random.default_rng({s + i})\nn, B = {nn}, {bb}\nrows = g.integers(0, n, 200000)\n"
             f"cols = np.clip(rows + g.integers(-n // 50, n // 50 + 1, 200000), 0, n - 1)\nreport = {report!r}"
             for i, (nn, bb) in ((1, (10 ** 9, 1000)), (2, (50000, 300)))]
    want = {"dark": "the number of dark blocks",
            "busiest": "the largest number of **distinct** nonzero positions inside one block",
            "rows": "the number of dark blocks in each block-row (an int array of length B, top to bottom)"}[report]
    return challenge(
        "A thumbnail of a huge sparse matrix",
        "`spy` shows where a matrix is nonzero, but an n × n matrix with n up to 10^9 can't be drawn pixel by pixel. "
        "Instead you draw a B × B thumbnail: with block side s = ⌈n / B⌉, block (I, J) covers rows I·s … I·s + s − 1 "
        "and columns J·s … J·s + s − 1, and is *dark* when it contains at least one nonzero. The nonzeros are given "
        "as positions `(rows[k], cols[k])`; a position may be listed more than once. Return " + want +
        f". (`report` = {report!r}.)",
        setup, sol, tests, "n ≤ 10^9, B ≤ 1000, up to 2·10^5 listed positions",
        speed="O(m log m): turn every position into a block id and count distinct ids (no dense matrix!)",
        answer_desc=" (an int)" if report != "rows" else " (an int array)")

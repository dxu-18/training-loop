"""Matplotlib figures as plain data, for grading plots (`post` code) without looking at pixels.

    import generator.figcheck as fc
    fc.lines(fc.ax())                    # [{'x': [...], 'y': [...]}, ...] of the current Axes
    fc.summary(fc.ax(), "labels", "legend", lines=("color", "ls"))

Everything returns lists / dicts / strings / rounded floats, so the grader compares a reference figure with the
learner's one value by value. Pick only what the prompt asks for: a figure that is right in every requested respect
must pass even if the learner styled something else differently. Colors come back as '#rrggbb' (alpha separately),
numbers rounded to `nd` decimals. Matplotlib is imported lazily (the module is imported by the grader on every page).
"""

from __future__ import annotations


def _plt():
    import matplotlib.pyplot as plt
    return plt


def r(v, nd: int = 6):
    """Round numbers inside lists / arrays / tuples (masked values -> None)."""
    import numpy as np
    if isinstance(v, np.ma.MaskedArray):
        return r(v.tolist(None), nd)          # masked cells -> None, at any number of dimensions
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [r(x, nd) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f != f:
        return None
    return round(f, nd) + 0.0


def hexcolor(c) -> str | None:
    """'#rrggbb' for any color spec or RGBA array (None for 'none' / fully transparent)."""
    from matplotlib.colors import to_hex, to_rgba
    import numpy as np
    if c is None:
        return None
    a = np.asarray(c)
    if a.ndim == 2:
        if len(a) == 0:
            return None
        c = a[0]
    if isinstance(c, str) and c.lower() == "none":
        return None
    rgba = to_rgba(c)
    return None if rgba[3] == 0 else to_hex(rgba, keep_alpha=False)


def alpha(c):
    from matplotlib.colors import to_rgba
    import numpy as np
    a = np.asarray(c)
    if a.ndim == 2:
        c = a[0] if len(a) else (0, 0, 0, 0)
    return r(to_rgba(c)[3], 3)


# ---------------------------------------------------------------- figure / axes

def fig():
    return _plt().gcf()


def figs() -> list:
    plt = _plt()
    return [plt.figure(n) for n in plt.get_fignums()]


def ax(i: int | None = None, f=None):
    """The current Axes, or the i-th Axes of the (current) figure."""
    f = f or fig()
    return f.gca() if i is None else f.axes[i]


def axes(f=None, colorbars: bool = False) -> list:
    """The figure's Axes in creation order (without colorbar Axes unless asked)."""
    f = f or fig()
    return [a for a in f.axes if colorbars or not _is_colorbar(a)]


def _is_colorbar(a) -> bool:
    return a.get_label() == "<colorbar>" or hasattr(a, "_colorbar")


def grid_shape(f=None):
    """(rows, cols) of the subplot grid the first Axes belongs to (None if not on a grid)."""
    for a in axes(f):
        ss = a.get_subplotspec()
        if ss is not None:
            g = ss.get_gridspec()
            return (g.nrows, g.ncols)
    return None


def spans(f=None) -> list:
    """Each Axes' cells on its grid: [(row0, row1, col0, col1)] (end-exclusive), None for free-floating Axes."""
    out = []
    for a in axes(f):
        ss = a.get_subplotspec()
        if ss is None:
            out.append(None)
            continue
        rs, cs = ss.rowspan, ss.colspan
        out.append((rs.start, rs.stop, cs.start, cs.stop))
    return out


def size(f=None, nd: int = 3):
    f = f or fig()
    return r(f.get_size_inches(), nd)


def labels(a=None) -> dict:
    a = a or ax()
    return {"title": a.get_title(), "xlabel": a.get_xlabel(), "ylabel": a.get_ylabel()}


def titles(a=None) -> dict:
    """Titles at all three positions (set_title(loc=...))."""
    a = a or ax()
    return {loc: a.get_title(loc) for loc in ("left", "center", "right")}


def suptitles(f=None) -> dict:
    f = f or fig()
    get = lambda t: t.get_text() if t is not None else ""  # noqa: E731
    return {"suptitle": get(f._suptitle), "supxlabel": get(f._supxlabel), "supylabel": get(f._supylabel)}


def limits(a=None, nd: int = 4) -> dict:
    a = a or ax()
    return {"xlim": r(a.get_xlim(), nd), "ylim": r(a.get_ylim(), nd)}


def scales(a=None) -> dict:
    a = a or ax()
    return {"x": a.get_xscale(), "y": a.get_yscale()}


def ticks(a=None, axis: str = "x", minor: bool = False, nd: int = 6) -> dict:
    """Tick positions and their label texts (labels as the formatter writes them)."""
    a = a or ax()
    obj = a.xaxis if axis == "x" else a.yaxis
    locs = obj.get_minorticklocs() if minor else obj.get_majorticklocs()
    texts = [t.get_text() for t in (obj.get_minorticklabels() if minor else obj.get_majorticklabels())]
    return {"locs": r(locs, nd), "labels": texts}


def tick_labels(a=None, axis: str = "x") -> list:
    return ticks(a, axis)["labels"]


def formatted(a=None, axis: str = "y", values=(0, 0.5, 1)) -> list:
    """What the axis' major formatter writes for some values (no draw needed)."""
    a = a or ax()
    fmt = (a.xaxis if axis == "x" else a.yaxis).get_major_formatter()
    fmt.set_locs(list(values))
    return [fmt(v, i) for i, v in enumerate(values)]


def spines(a=None) -> dict:
    a = a or ax()
    return {k: s.get_visible() for k, s in a.spines.items()}


def grid_on(a=None) -> dict:
    a = a or ax()
    on = lambda axis: any(t.gridline.get_visible() for t in axis.get_major_ticks())  # noqa: E731
    return {"x": on(a.xaxis), "y": on(a.yaxis)}


# ---------------------------------------------------------------- artists

LINE_PROPS = {"color": lambda l: hexcolor(l.get_color()), "ls": lambda l: l.get_linestyle(),
              "lw": lambda l: r(l.get_linewidth(), 3), "marker": lambda l: l.get_marker(),
              "ms": lambda l: r(l.get_markersize(), 3), "alpha": lambda l: l.get_alpha(),
              "label": lambda l: l.get_label(), "drawstyle": lambda l: l.get_drawstyle(),
              "zorder": lambda l: l.get_zorder(), "mfc": lambda l: hexcolor(l.get_markerfacecolor()),
              "markevery": lambda l: l.get_markevery()}


def lines(a=None, *props, nd: int = 6, data: bool = True, labelled: bool = False) -> list:
    """The Axes' Line2D artists (in drawing order): {'x', 'y', **props}. labelled: skip '_child...' lines."""
    a = a or ax()
    out = []
    for l in a.get_lines():
        if labelled and l.get_label().startswith("_"):
            continue
        d = {"x": r(l.get_xdata(), nd), "y": r(l.get_ydata(), nd)} if data else {}
        for p in props:
            d[p] = LINE_PROPS[p](l)
        out.append(d)
    return out


def ydata(a=None, nd: int = 6) -> list:
    return [d["y"] for d in lines(a, nd=nd)]


def collections(a=None) -> list:
    """Type names of the Axes' collections (PathCollection = scatter, PolyCollection = fill_between ...)."""
    a = a or ax()
    return [type(c).__name__ for c in a.collections]


def scatter(a=None, i: int = 0, nd: int = 6, *props) -> dict:
    """The i-th PathCollection (scatter): offsets, plus 'sizes', 'array' (c values), 'color', 'cmap', 'alpha',
    'marker' (via its paths' identity is not kept: ask for 'n' instead), 'edgecolor', 'label', 'clim'."""
    from matplotlib.collections import PathCollection
    a = a or ax()
    pcs = [c for c in a.collections if isinstance(c, PathCollection)]
    c = pcs[i]
    d = {"offsets": r(c.get_offsets(), nd)}
    for p in props:
        if p == "sizes":
            d[p] = r(c.get_sizes(), 4)
        elif p == "array":
            arr = c.get_array()
            d[p] = None if arr is None else r(arr, nd)
        elif p == "color":
            d[p] = hexcolor(c.get_facecolor())
        elif p == "edgecolor":
            d[p] = hexcolor(c.get_edgecolor())
        elif p == "cmap":
            d[p] = c.get_cmap().name if c.get_array() is not None else None
        elif p == "alpha":
            d[p] = c.get_alpha()
        elif p == "label":
            d[p] = c.get_label()
        elif p == "clim":
            d[p] = r(c.get_clim(), nd)
        elif p == "n":
            d[p] = len(c.get_offsets())
    return d


def n_scatters(a=None) -> int:
    from matplotlib.collections import PathCollection
    a = a or ax()
    return sum(isinstance(c, PathCollection) for c in a.collections)


def bars(a=None, nd: int = 6, *props) -> list:
    """Rectangle patches (bars, histogram bins): {'x', 'y', 'w', 'h'} (+ 'color', 'edgecolor', 'label', 'alpha')."""
    from matplotlib.patches import Rectangle
    a = a or ax()
    out = []
    for p in a.patches:
        if not isinstance(p, Rectangle):
            continue
        d = {"x": r(p.get_x(), nd), "y": r(p.get_y(), nd), "w": r(p.get_width(), nd), "h": r(p.get_height(), nd)}
        for q in props:
            d[q] = {"color": lambda: hexcolor(p.get_facecolor()), "edgecolor": lambda: hexcolor(p.get_edgecolor()),
                    "label": lambda: p.get_label(), "alpha": lambda: p.get_alpha(),
                    "fill": lambda: p.get_fill()}[q]()
        out.append(d)
    return out


def heights(a=None, nd: int = 6) -> list:
    return [b["h"] for b in bars(a, nd)]


def widths(a=None, nd: int = 6) -> list:
    return [b["w"] for b in bars(a, nd)]


def patches(a=None, nd: int = 4) -> list:
    """Every patch: its class name and the defining numbers of common shapes."""
    from matplotlib import patches as mp
    a = a or ax()
    out = []
    for p in a.patches:
        d = {"type": type(p).__name__}
        if isinstance(p, mp.Rectangle):
            d.update(xy=r(p.get_xy(), nd), w=r(p.get_width(), nd), h=r(p.get_height(), nd))
        elif isinstance(p, mp.Circle):
            d.update(center=r(p.get_center(), nd), radius=r(p.get_radius(), nd))
        elif isinstance(p, mp.Ellipse):
            d.update(center=r(p.get_center(), nd), w=r(p.width, nd), h=r(p.height, nd), angle=r(p.angle, nd))
        elif isinstance(p, mp.Wedge):
            d.update(center=r(p.center, nd), r=r(p.r, nd), theta=r((p.theta1, p.theta2), nd))
        elif isinstance(p, mp.Polygon):
            d.update(xy=r(p.get_xy(), nd))
        out.append(d)
    return out


def texts(a=None) -> list:
    """Texts added with ax.text / ax.annotate, in order."""
    a = a or ax()
    return [t.get_text() for t in a.texts]


def text_items(a=None, nd: int = 4) -> list:
    """[{'text', 'xy' (position), 'ha', 'va', 'color'}] of ax.text / ax.annotate texts."""
    a = a or ax()
    out = []
    for t in a.texts:
        d = {"text": t.get_text(), "xy": r(t.get_position(), nd), "ha": t.get_horizontalalignment(),
             "va": t.get_verticalalignment(), "color": hexcolor(t.get_color())}
        if hasattr(t, "xy"):          # an Annotation: the point it annotates
            d["target"] = r(t.xy, nd)
        out.append(d)
    return out


def legend(a=None) -> list | None:
    """The legend's entry texts (None if the Axes has no legend)."""
    a = a or ax()
    lg = a.get_legend()
    return None if lg is None else [t.get_text() for t in lg.get_texts()]


def legend_title(a=None) -> str | None:
    a = a or ax()
    lg = a.get_legend()
    return None if lg is None else lg.get_title().get_text()


def fig_legends(f=None) -> list:
    f = f or fig()
    return [[t.get_text() for t in lg.get_texts()] for lg in f.legends]


def images(a=None, nd: int = 6, data: bool = False) -> list:
    """AxesImages (imshow / matshow): shape, cmap, clim, origin, extent (+ the array if data)."""
    a = a or ax()
    out = []
    for im in a.get_images():
        arr = im.get_array()
        d = {"shape": list(arr.shape), "cmap": im.get_cmap().name, "clim": r(im.get_clim(), nd),
             "origin": im.origin, "extent": r(im.get_extent(), nd)}
        if data:
            d["data"] = r(arr, nd)
        out.append(d)
    return out


def norm(mappable) -> dict:
    """A mappable's norm: its class and vmin / vmax (and vcenter / boundaries when it has them)."""
    n = mappable.norm
    d = {"type": type(n).__name__, "vmin": r(n.vmin), "vmax": r(n.vmax)}
    if hasattr(n, "vcenter"):
        d["vcenter"] = r(n.vcenter)
    if hasattr(n, "boundaries"):
        d["boundaries"] = r(n.boundaries)
    return d


def colorbars(f=None) -> list:
    """Each colorbar on the figure: its label and orientation."""
    f = f or fig()
    out = []
    for a in f.axes:
        cb = getattr(a, "_colorbar", None)
        if cb is not None:
            out.append({"label": cb.ax.get_ylabel() if cb.orientation == "vertical" else cb.ax.get_xlabel(),
                        "orientation": cb.orientation})
    return out


def summary(a=None, *what, lines=None, nd: int = 6) -> dict:
    """Several of the above at once: what in {'labels', 'limits', 'scales', 'legend', 'texts', 'bars', 'images',
    'collections', 'xticks', 'yticks', 'spines'}; lines=(props...) adds the line data with those properties."""
    a = a or ax()
    fns = {"labels": labels, "limits": limits, "scales": scales, "legend": legend, "texts": texts, "bars": bars,
           "images": images, "collections": collections, "spines": spines,
           "xticks": lambda a: ticks(a, "x"), "yticks": lambda a: ticks(a, "y")}
    d = {w: fns[w](a) for w in what}
    if lines is not None:
        d["lines"] = globals()["lines"](a, *lines, nd=nd)
    return d

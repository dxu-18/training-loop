"""5.13 Patches, collections & coordinate systems (shapes, LineCollection / PatchCollection, transData / transAxes /
transFigure, blended transforms, boxes over images)."""

from ._common import P, catalog, challenge, curves, lit, seeded, template
from ... import themes as T

M = "5.13"

# Check helpers, defined at the top of `post` code: _pp(p, *props) reads a patch's style, _ext(p, how) its extent
# (in data / axes / blended coordinates, from where it really lands on the canvas, so any transform that puts it in
# the right place passes), _at(t) a text's anchor in axes (or figure) fractions.
PRE = '''def _pp(p, *props):
    out = {}
    for k in props:
        if k == 'fc':
            out[k] = fc.hexcolor(p.get_facecolor())
        elif k == 'ec':
            out[k] = fc.hexcolor(p.get_edgecolor())
        elif k == 'fill':
            out[k] = bool(p.get_fill())
        elif k == 'lw':
            out[k] = fc.r(p.get_linewidth(), 3)
        elif k == 'alpha':
            out[k] = fc.alpha(p.get_facecolor())
        elif k == 'theta':
            out[k] = fc.r([p.theta1, p.theta2], 3)
        elif k == 'width':
            out[k] = fc.r(p.width, 4)
        elif k == 'ls':
            out[k] = p.get_linestyle()
        elif k == 'closed':
            out[k] = p.get_closed()
        elif k == 'box':
            s = p.get_boxstyle()
            out[k] = [fc.r([p.get_x(), p.get_y(), p.get_width(), p.get_height()], 4), type(s).__name__.lower(),
                      fc.r(getattr(s, 'pad', None), 4)]
        elif k == 'arrow':
            out[k] = [fc.r(p._posA_posB, 4), type(p.get_arrowstyle()).__name__, fc.r(p.get_mutation_scale(), 3)]
    return out

def _ext(p, how='data', a=None):
    a = a or fc.ax()
    a.get_xlim(); a.get_ylim()
    b = p.get_window_extent()
    pts = [[b.x0, b.y0], [b.x1, b.y1]]
    d = a.transData.inverted().transform(pts)
    x = a.transAxes.inverted().transform(pts)
    return fc.r({'data': d, 'axes': x, 'blend': [[d[0][0], x[0][1]], [d[1][0], x[1][1]]],
                 'yblend': [[x[0][0], d[0][1]], [x[1][0], d[1][1]]]}[how], 3)

def _at(t, a=None, fig=False):
    a = a or fc.ax()
    a.get_xlim(); a.get_ylim()
    tr = (a.figure.transFigure if fig else a.transAxes).inverted()
    return fc.r(tr.transform(t.get_transform().transform(t.get_position())), 3)

def _segs(a=None):
    from matplotlib.collections import LineCollection
    a = a or fc.ax()
    out = []
    for c in a.collections:
        if isinstance(c, LineCollection):
            out += [sorted(fc.r(s, 4)) for s in c.get_segments()]
    return sorted(out)

def _boxes(a=None, *props):
    from matplotlib.patches import Rectangle
    a = a or fc.ax()
    return sorted([fc.r(list(p.get_xy()) + [p.get_width(), p.get_height()], 3)] + [str(v) for v in _pp(p, *props).values()]
                  for p in a.patches if isinstance(p, Rectangle))

def _circles(a=None, *props):
    from matplotlib.patches import Circle
    a = a or fc.ax()
    return sorted([fc.r(list(p.get_center()) + [p.get_radius()], 3)] + [str(v) for v in _pp(p, *props).values()]
                  for p in a.patches if isinstance(p, Circle))
'''
SHAPES = PRE + "fc.patches(fc.ax())"
IMPORTS = "Import what you need from `matplotlib.patches` / `matplotlib.collections`."


def _post(*exprs):
    return PRE + ", ".join(exprs)


def _num(rng, lo, hi, nd=2):
    return round(rng.uniform(lo, hi), nd)


def _color(rng):
    return rng.choice(["red", "C0", "C1", "C2", "tab:purple", "navy", "darkorange", "teal"])


def _detection(rng, n=(2, 4)):
    """An image `img` and detection boxes (x, y, w, h, label, score) inside it."""
    H, W = rng.choice([(40, 60), (48, 64), (50, 80), (60, 90)])
    _, labs = T.labels(rng, min_classes=3)
    k = rng.randint(*n)
    boxes = []
    for _ in range(k):
        w, h = rng.randint(8, W // 3), rng.randint(8, H // 3)
        x, y = rng.randint(2, W - w - 2), rng.randint(4, H - h - 2)
        boxes.append((x, y, w, h, rng.choice(labs), round(rng.uniform(0.3, 0.99), 2)))
    s = seeded(rng)
    code = f"img = np.random.default_rng({s}).random(({H}, {W}, 3))\nboxes = {lit(boxes)}   # x, y, w, h, label, score"
    return code, boxes, (H, W)


def _gt_pred(rng):
    """Ground-truth boxes and noisy predictions of them (x, y, w, h [, score])."""
    H, W = rng.choice([(40, 60), (50, 80), (60, 90)])
    k = rng.randint(2, 3)
    gt, pred = [], []
    for _ in range(k):
        w, h = rng.randint(10, W // 3), rng.randint(10, H // 3)
        x, y = rng.randint(3, W - w - 3), rng.randint(3, H - h - 3)
        gt.append((x, y, w, h))
        pred.append((x + rng.randint(-3, 3), y + rng.randint(-3, 3), w + rng.randint(-3, 3), h + rng.randint(-3, 3),
                     round(rng.uniform(0.4, 0.99), 2)))
    s = seeded(rng)
    code = (f"img = np.random.default_rng({s}).random(({H}, {W}))\ngt = {lit(gt)}   # x, y, w, h\n"
            f"pred = {lit(pred)}   # x, y, w, h, score")
    return code, gt, pred


# ================================================================= lite

@template(M, "lite")
def patch_primitives(rng):
    x, y = _num(rng, 0.1, 0.3), _num(rng, 0.1, 0.3)
    w, h = _num(rng, 0.2, 0.5), _num(rng, 0.1, 0.4)
    cx, cy, r = _num(rng, 0.4, 0.6), _num(rng, 0.4, 0.6), _num(rng, 0.1, 0.3)
    c = _color(rng)
    lw = rng.choice([1.5, 2, 3])
    a = rng.choice([0.3, 0.5, 0.7])
    ang = rng.choice([15, 30, 45, 60, 120])
    t1, t2 = rng.choice([(0, 90), (30, 150), (0, 120), (45, 300), (90, 270)])
    tri = [[x, y], [round(x + w, 2), y], [round(x + w / 2, 2), round(y + h, 2)]]
    pad = rng.choice([0.01, 0.02, 0.05])
    ms = rng.choice([10, 15, 20])
    items = [
        ("Rectangle", f"Create `fig, ax`, then add a `Rectangle` whose lower-left corner is ({x}, {y}), {w} wide and {h} "
                      f"tall.", f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\n"
                                f"ax.add_patch(Rectangle(({x}, {y}), {w}, {h}))", {"post": SHAPES, "require": ["Rectangle", "add_patch"]}),
        ("Rectangle", f"Add an **outline** rectangle (no fill) at ({x}, {y}), size {w} × {h}, with edge color {c!r} and "
                      f"line width {lw}.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\n"
         f"ax.add_patch(Rectangle(({x}, {y}), {w}, {h}, fill=False, edgecolor={c!r}, linewidth={lw}))",
         {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'fill', 'ec', 'lw') for p in fc.ax().patches]")}),
        ("Circle", f"Add a `Circle` centered at ({cx}, {cy}) with radius {r}, color {c!r} and alpha {a}.",
         f"from matplotlib.patches import Circle\nfig, ax = plt.subplots()\nax.add_patch(Circle(({cx}, {cy}), {r}, "
         f"color={c!r}, alpha={a}))", {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]")}),
        ("Circle", f"Add a hollow circle (`fill=False`) at ({cx}, {cy}), radius {r}, with edge color {c!r}.",
         f"from matplotlib.patches import Circle\nfig, ax = plt.subplots()\n"
         f"ax.add_patch(Circle(({cx}, {cy}), {r}, fill=False, edgecolor={c!r}))",
         {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'fill', 'ec') for p in fc.ax().patches]")}),
        ("Ellipse", f"Add an `Ellipse` centered at ({cx}, {cy}), {w} wide and {h} tall (full axes, not radii), rotated "
                    f"{ang}° counter-clockwise.",
         f"from matplotlib.patches import Ellipse\nfig, ax = plt.subplots()\nax.add_patch(Ellipse(({cx}, {cy}), {w}, {h}, "
         f"angle={ang}))", {"post": SHAPES}),
        ("Polygon", f"Add a filled triangle with the corners {tri} as a `Polygon` (closed) in color {c!r}.",
         f"from matplotlib.patches import Polygon\nfig, ax = plt.subplots()\nax.add_patch(Polygon({tri}, closed=True, "
         f"color={c!r}))", {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'fc') for p in fc.ax().patches]")}),
        ("Polygon", f"Draw the open polyline through {tri} as a `Polygon` with `closed=False` and `fill=False` (the "
                    f"last corner is not joined back to the first).",
         f"from matplotlib.patches import Polygon\nfig, ax = plt.subplots()\nax.add_patch(Polygon({tri}, closed=False, "
         f"fill=False))", {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'closed', 'fill') for p in fc.ax().patches]")}),
        ("Wedge", f"Add a `Wedge` (a pie slice) centered at ({cx}, {cy}) with radius {r}, from {t1}° to {t2}° "
                  f"(counter-clockwise from the +x direction).",
         f"from matplotlib.patches import Wedge\nfig, ax = plt.subplots()\nax.add_patch(Wedge(({cx}, {cy}), {r}, {t1}, {t2}))",
         {"post": SHAPES}),
        ("Wedge", f"Add a ring segment: a `Wedge` at ({cx}, {cy}), outer radius {r}, from {t1}° to {t2}°, with "
                  f"`width={round(r / 3, 3)}` (the ring's thickness).",
         f"from matplotlib.patches import Wedge\nfig, ax = plt.subplots()\nax.add_patch(Wedge(({cx}, {cy}), {r}, {t1}, "
         f"{t2}, width={round(r / 3, 3)}))", {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'width') for p in fc.ax().patches]")}),
        ("Arc", f"Add an `Arc` (an unfilled piece of an ellipse) centered at ({cx}, {cy}), {w} wide and {w} tall, from "
                f"`theta1={t1}` to `theta2={t2}` degrees.",
         f"from matplotlib.patches import Arc\nfig, ax = plt.subplots()\nax.add_patch(Arc(({cx}, {cy}), {w}, {w}, "
         f"theta1={t1}, theta2={t2}))", {"post": _post("fc.patches(fc.ax())", "[_pp(p, 'theta') for p in fc.ax().patches]")}),
        ("FancyBboxPatch", f"Add a rounded box: a `FancyBboxPatch` at ({x}, {y}), {w} × {h}, with `boxstyle=\"round,"
                           f"pad={pad}\"`, face color 'lightyellow' and edge color 'k'.",
         f"from matplotlib.patches import FancyBboxPatch\nfig, ax = plt.subplots()\nax.add_patch(FancyBboxPatch(({x}, {y}), "
         f"{w}, {h}, boxstyle='round,pad={pad}', fc='lightyellow', ec='k'))",
         {"post": _post("[_pp(p, 'box', 'fc', 'ec') for p in fc.ax().patches]")}),
        ("FancyArrowPatch", f"Add an arrow from ({x}, {y}) to ({cx}, {cy}) as a `FancyArrowPatch` with "
                            f"`arrowstyle='-|>'` and `mutation_scale={ms}` (the arrow-head size).",
         f"from matplotlib.patches import FancyArrowPatch\nfig, ax = plt.subplots()\nax.add_patch(FancyArrowPatch(({x}, {y}), "
         f"({cx}, {cy}), arrowstyle='-|>', mutation_scale={ms}))",
         {"post": _post("[_pp(p, 'arrow') for p in fc.ax().patches]")}),
    ]
    return catalog(rng, items, "", f"Create `fig, ax = plt.subplots()` and use `ax.add_patch`. {IMPORTS}")


@template(M, "lite")
def coordinate_systems(rng):
    W, H = rng.choice([(4, 3), (5, 4), (6, 3), (3, 3)])
    dpi = rng.choice([50, 80, 100])
    x1, y1 = rng.choice([(10, 5), (100, 1), (4, 8), (20, 20)])
    px, py = round(rng.uniform(0.2, 0.8) * x1, 2), round(rng.uniform(0.2, 0.8) * y1, 2)
    label = rng.choice(["axes coords", "top", "note", "subtitle"])
    yy = rng.choice([0.25, 0.5, 0.75])
    a, b = sorted(round(rng.uniform(0.1, 0.9) * x1, 1) for _ in range(2))
    setup = f"fig, ax = plt.subplots(figsize=({W}, {H}), dpi={dpi})\nax.set_xlim(0, {x1})\nax.set_ylim(0, {y1})"
    items = [
        (".text", f"Write {label!r} centered just above the plot, at (0.5, 1.02) in **axes** coordinates (`transform="
                  f"ax.transAxes`, `ha='center'`), whatever the data limits are.",
         f"ax.text(0.5, 1.02, {label!r}, transform=ax.transAxes, ha='center')",
         {"post": _post("fc.texts()", "[_at(t) for t in fc.ax().texts]", "[t.get_ha() for t in fc.ax().texts]"),
          "require": ["transAxes"]}),
        (".text", f"Write 'figure footnote' at (0.01, 0.01) in **figure** coordinates with `fig.text` (font size 8).",
         "fig.text(0.01, 0.01, 'figure footnote', fontsize=8)",
         {"post": "[(t.get_text(), fc.r(t.get_position(), 3), t.get_fontsize()) for t in fc.fig().texts]"}),
        (".plot", f"Draw a dotted gray horizontal line across the whole Axes at {yy} of its height, in axes coordinates: "
                  f"`ax.plot([0, 1], [{yy}, {yy}], transform=ax.transAxes, ls=':', color='gray')`.",
         f"ax.plot([0, 1], [{yy}, {yy}], transform=ax.transAxes, ls=':', color='gray')",
         {"post": "[(fc.r(fc.ax().transAxes.inverted().transform(l.get_transform().transform(l.get_xydata())), 3), "
                  "l.get_linestyle(), fc.hexcolor(l.get_color())) for l in fc.ax().get_lines()]",
          "require": ["transAxes"]}),
        (".set_aspect", "Make one data unit equally long on both axes (`ax.set_aspect('equal')`) so circles look round.",
         "ax.set_aspect('equal')", {"post": "fc.ax().get_aspect()"}),
        (".axis", "Hide the axis lines, ticks and labels with `ax.axis('off')`.", "ax.axis('off')",
         {"post": "fc.ax().axison"}),
        ("transData", f"Return where the data point ({px}, {py}) lands on the canvas, in display pixels: "
                      f"`ax.transData.transform(...)`.", f"result = ax.transData.transform(({px}, {py}))"),
        ("transAxes", "Return the display (pixel) position of the Axes' top-right corner: axes coordinates (1, 1) "
                      "through `ax.transAxes`.", "result = ax.transAxes.transform((1, 1))"),
        (".inverted", f"Go backwards: return the data coordinates of the pixel (100, 80), using the inverse of "
                      f"`ax.transData` (`.inverted()`).", "result = ax.transData.inverted().transform((100, 80))",
         {"require": ["inverted", "transData"]}),
        ("transFigure", f"Return the pixel position of the figure's center, (0.5, 0.5) in figure coordinates, through "
                        f"`fig.transFigure`. (The figure is {W} × {H} inches at {dpi} dpi.)",
         "result = fig.transFigure.transform((0.5, 0.5))"),
        ("blended_transform_factory", f"Shade a vertical band from x = {a} to x = {b} (data units) over the **full "
                                      f"height** of the Axes: a `Rectangle(({a}, 0), {round(b - a, 1)}, 1)` with a "
                                      f"blended transform (x in data, y in axes coordinates), color 'gray', alpha 0.2.",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\n"
         f"trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         f"ax.add_patch(Rectangle(({a}, 0), {round(b - a, 1)}, 1, transform=trans, color='gray', alpha=0.2))",
         {"post": _post("[_ext(p, 'blend') for p in fc.ax().patches]", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]")}),
        ("LineCollection", f"Add these two segments as **one** `LineCollection`: from (0, 0) to ({x1}, {y1}) and from "
                           f"(0, {y1}) to ({x1}, 0).",
         f"from matplotlib.collections import LineCollection\nax.add_collection(LineCollection([[(0, 0), ({x1}, {y1})], "
         f"[(0, {y1}), ({x1}, 0)]]))", {"post": _post("_segs()", "fc.collections()"), "require": ["LineCollection", "add_collection"]}),
        ("PatchCollection", f"Add 5 small circles (radius {round(min(x1, y1) / 20, 2)}) at x = {round(x1 / 2, 2)}, y = "
                            f"`np.linspace(0, {y1}, 5)` as **one** `PatchCollection` with face color 'k'.",
         f"from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\n"
         f"ax.add_collection(PatchCollection([Circle(({round(x1 / 2, 2)}, yy), {round(min(x1, y1) / 20, 2)}) for yy in "
         f"np.linspace(0, {y1}, 5)], facecolor='k'))",
         {"post": "c = fc.ax().collections\n[type(x).__name__ for x in c], [len(x.get_paths()) for x in c], "
                  "[fc.hexcolor(x.get_facecolor()) for x in c], [fc.r(p.get_extents(x.get_transform() - fc.ax().transData).bounds, 3) "
                  "for x in c for p in x.get_paths()]", "require": ["PatchCollection", "Circle"]}),
    ]
    return catalog(rng, items, setup, f"`fig, ax` exist: a {W} × {H} inch figure at {dpi} dpi with x limits 0 … {x1} and "
                                      f"y limits 0 … {y1}.")


# ================================================================= easy

@template(M, "easy")
def patch_readback(rng):
    x, y = rng.randint(0, 5), rng.randint(0, 5)
    w, h = rng.randint(1, 6), rng.randint(1, 6)
    cx, cy, r = rng.randint(2, 6), rng.randint(2, 6), rng.randint(1, 3)
    ang = rng.choice([0, 30, 45, 90])
    k = rng.randint(3, 7)
    s = seeded(rng)
    pts = f"pts = np.round(np.random.default_rng({s}).uniform(0, 10, (12, 2)), 2)"
    tri = [[x, y], [x + w, y], [x, y + h]]
    items = [
        (".get_corners", f"Create `rect = Rectangle(({x}, {y}), {w}, {h})` and return its four corners "
                         f"(`rect.get_corners()`, counter-clockwise from the anchor).",
         f"from matplotlib.patches import Rectangle\nrect = Rectangle(({x}, {y}), {w}, {h})\nresult = rect.get_corners()"),
        (".get_center", f"Create `Rectangle(({x}, {y}), {w}, {h}, angle={ang})` (rotated about its anchor corner) and "
                        f"return its center (`get_center()`).",
         f"from matplotlib.patches import Rectangle\nrect = Rectangle(({x}, {y}), {w}, {h}, angle={ang})\n"
         f"result = rect.get_center()", {"tol": 1e-9}),
        (".get_xy", f"A closed `Polygon` stores its first vertex again at the end. Create `Polygon({tri})` and return "
                    f"`poly.get_xy()`.", f"from matplotlib.patches import Polygon\npoly = Polygon({tri})\nresult = poly.get_xy()"),
        (".get_vertices", f"Create `Ellipse(({cx}, {cy}), {2 * w}, {h}, angle={ang})` and return the two ends of its "
                          f"major axis (`get_vertices()`) as an array.",
         f"from matplotlib.patches import Ellipse\ne = Ellipse(({cx}, {cy}), {2 * w}, {h}, angle={ang})\n"
         f"result = np.array(e.get_vertices())", {"tol": 1e-9}),
        (".contains_points", f"Which points lie inside the triangle `Polygon({tri})`? A patch's path can test many "
                             f"points at once: return `poly.get_path().contains_points(pts)` (a boolean array).",
         f"from matplotlib.patches import Polygon\npoly = Polygon({tri})\nresult = poly.get_path().contains_points(pts)",
         {"setup": pts}),
        (".contains_points", f"Count the points of `pts` inside `Circle(({cx}, {cy}), {r})`. A circle's path is the "
                             f"**unit** circle, placed by its patch transform: pass `transform=c.get_patch_transform()` "
                             f"to `c.get_path().contains_points(...)`.",
         f"from matplotlib.patches import Circle\nc = Circle(({cx}, {cy}), {r})\n"
         f"result = int(c.get_path().contains_points(pts, transform=c.get_patch_transform()).sum())",
         {"setup": pts, "require": ["get_patch_transform", "contains_points"]}),
        (".set_xy", f"Create `fig, ax`, add `rect = Rectangle((0, 0), 1, 1)`, then **move and resize the same patch**: "
                    f"anchor ({x}, {y}) with `set_xy`, width {w} with `set_width`, height {h} with `set_height`.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nrect = Rectangle((0, 0), 1, 1)\n"
         f"ax.add_patch(rect)\nrect.set_xy(({x}, {y}))\nrect.set_width({w})\nrect.set_height({h})",
         {"post": SHAPES, "require": ["set_xy", "set_width", "set_height"]}),
        (".patches", f"Create `fig, ax` and add {k} circles in a loop (centers `(i, i)`, radius 0.4). Return "
                     f"`len(ax.patches)`.",
         f"from matplotlib.patches import Circle\nfig, ax = plt.subplots()\nfor i in range({k}):\n"
         f"    ax.add_patch(Circle((i, i), 0.4))\nresult = len(ax.patches)"),
        (".get_bbox", f"Return the bounds `(x0, y0, width, height)` of `Rectangle(({x}, {y}), {w}, {h})` via "
                      f"`get_bbox().bounds`.",
         f"from matplotlib.patches import Rectangle\nresult = Rectangle(({x}, {y}), {w}, {h}).get_bbox().bounds"),
        (".set_radius", f"Create `c = Circle(({cx}, {cy}), 1)`, grow it with `c.set_radius({r + 1})`, then return "
                        f"`[c.get_radius(), c.get_center()]` with the center as a list.",
         f"from matplotlib.patches import Circle\nc = Circle(({cx}, {cy}), 1)\nc.set_radius({r + 1})\n"
         f"result = [c.get_radius(), list(c.get_center())]"),
    ]
    return catalog(rng, items, "", IMPORTS)


@template(M, "easy")
def transform_readback(rng):
    W, H = rng.choice([(4, 3), (5, 4), (6, 3)])
    dpi = rng.choice([50, 80, 100])
    x0, x1 = rng.choice([(0, 10), (-5, 5), (0, 100), (2, 4)])
    y0, y1 = rng.choice([(0, 1), (-1, 1), (0, 50), (10, 20)])
    l, b, w, h = rng.choice([(0.1, 0.1, 0.8, 0.8), (0.125, 0.11, 0.775, 0.77), (0.2, 0.15, 0.7, 0.75)])
    px = round(x0 + rng.uniform(0.2, 0.8) * (x1 - x0), 2)
    py = round(y0 + rng.uniform(0.2, 0.8) * (y1 - y0), 2)
    setup = (f"fig = plt.figure(figsize=({W}, {H}), dpi={dpi})\nax = fig.add_axes(({l}, {b}, {w}, {h}))\n"
             f"ax.set_xlim({x0}, {x1})\nax.set_ylim({y0}, {y1})\npts = np.array([[{px}, {py}], [{x0}, {y0}], [{x1}, {y1}]])")
    items = [
        ("transData", "Return the display (pixel) coordinates of the three data points `pts` (`ax.transData.transform`).",
         "result = ax.transData.transform(pts)"),
        (".inverted", "Return the data coordinates of the pixels `[[100, 50], [150, 120]]` (invert `ax.transData`).",
         "result = ax.transData.inverted().transform([[100, 50], [150, 120]])", {"require": ["inverted"]}),
        ("transAxes", "Convert the axes-fraction points `[[0, 0], [0.5, 0.5], [1, 1]]` to **data** coordinates: first "
                      "to pixels with `ax.transAxes`, then back with the inverse of `ax.transData` (or compose the two "
                      "with `+`).", "result = (ax.transAxes + ax.transData.inverted()).transform([[0, 0], [0.5, 0.5], [1, 1]])",
         {"require": ["transAxes", "inverted"]}),
        ("transAxes", "Convert the data points `pts` to **axes fractions** (0 … 1 across the Axes): data → pixels with "
                      "`ax.transData`, pixels → axes with the inverse of `ax.transAxes`.",
         "result = (ax.transData + ax.transAxes.inverted()).transform(pts)", {"require": ["transAxes", "inverted"]}),
        ("transFigure", "Return the pixel coordinates of the figure corners (0, 0) and (1, 1) through `fig.transFigure` "
                        "(the second one is the canvas size: inches × dpi).",
         "result = fig.transFigure.transform([[0, 0], [1, 1]])"),
        ("dpi_scale_trans", "`fig.dpi_scale_trans` maps **inches** to pixels. Return where the point 1 inch right and "
                            "0.5 inch up from the bottom-left corner lands.", "result = fig.dpi_scale_trans.transform((1, 0.5))"),
        ("blended_transform_factory", f"Build `trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)` "
                                      f"(x in data units, y in axes fractions) and return the pixels of `[[{px}, 0], [{px}, 1]]` "
                                      f"(the bottom and top of a vertical line at x = {px}).",
         f"import matplotlib.transforms as mtransforms\ntrans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         f"result = trans.transform([[{px}, 0], [{px}, 1]])"),
        (".bbox", "Return the Axes' rectangle in pixels as `ax.bbox.bounds` (x0, y0, width, height).",
         "result = ax.bbox.bounds"),
        ("transData", "How many pixels long is one data unit along x, and along y? Transform (0, 0) and (1, 1) with "
                      "`ax.transData` and return the difference as `[px_per_x_unit, px_per_y_unit]`.",
         "a, b = ax.transData.transform([[0, 0], [1, 1]])\nresult = (b - a).tolist()", {"tol": 1e-9}),
        ("set_yscale", f"Transforms follow the scales: set `ax.set_yscale('log')` and `ax.set_ylim(1, 1000)`, then return "
                       f"the pixel **y** of the data points y = 1, 10, 100, 1000 at x = {px} (equally spaced!).",
         f"ax.set_yscale('log')\nax.set_ylim(1, 1000)\nresult = ax.transData.transform([[{px}, 1], [{px}, 10], [{px}, 100], "
         f"[{px}, 1000]])[:, 1]", {"tol": 1e-9}),
    ]
    return catalog(rng, items, setup, f"`fig` is {W} × {H} inches at {dpi} dpi; `ax` was placed by hand at ({l}, {b}, "
                                      f"{w}, {h}) (figure fractions) with x limits {x0} … {x1} and y limits {y0} … {y1}; "
                                      f"`pts` holds three data points.")


@template(M, "easy")
def boxes_on_images(rng):
    code, boxes, (H, W) = _detection(rng)
    c = rng.choice(["lime", "red", "yellow", "cyan"])
    thr = rng.choice([0.5, 0.6, 0.7])
    keep = [b for b in boxes if b[5] >= thr] or boxes[:1]
    thr = min(thr, min(b[5] for b in keep))
    corners = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boxes]
    centers = [(b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]) for b in boxes]
    norm = [(round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)) for b in boxes]
    box = PRE + "fc.images(), _boxes(None, 'fill', 'ec', 'lw')"
    items = [
        ("Rectangle", f"Show `img` with `ax.imshow` and outline every box with a `Rectangle` (no fill, edge color {c!r}, "
                      f"line width 2). Rectangles take the lower-left corner in data coordinates — on an image that's "
                      f"(column x, row y) of the top-left pixel.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         f"for bx, by, bw, bh, lbl, score in boxes:\n    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, "
         f"edgecolor={c!r}, linewidth=2))", {"post": box}),
        ("Rectangle", f"These boxes come in **corner** format (x1, y1, x2, y2): `corners = {lit(corners)}`. Show `img` "
                      f"and draw each as an unfilled `Rectangle` with edge color {c!r} and line width 2.",
         f"from matplotlib.patches import Rectangle\ncorners = {lit(corners)}\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         f"for x1, y1, x2, y2 in corners:\n    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, "
         f"edgecolor={c!r}, linewidth=2))", {"post": box}),
        ("Rectangle", f"These boxes come in **center** format (cx, cy, w, h): `centers = {lit(centers)}`. Show `img` and "
                      f"draw each as an unfilled `Rectangle` (edge color {c!r}, line width 2).",
         f"from matplotlib.patches import Rectangle\ncenters = {lit(centers)}\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         f"for cx, cy, w, h in centers:\n    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, "
         f"edgecolor={c!r}, linewidth=2))", {"post": box}),
        ("Rectangle", f"These boxes are **normalized** (fractions of the image width and height): `norm = {lit(norm)}`. "
                      f"Show `img` and draw them in pixel units as unfilled rectangles (edge color {c!r}, line width "
                      f"2). The image is `img.shape[1]` wide and `img.shape[0]` tall.",
         f"from matplotlib.patches import Rectangle\nnorm = {lit(norm)}\nH, W = img.shape[:2]\nfig, ax = plt.subplots()\n"
         f"ax.imshow(img)\nfor x, y, w, h in norm:\n    ax.add_patch(Rectangle((x * W, y * H), w * W, h * H, fill=False, "
         f"edgecolor={c!r}, linewidth=2))", {"post": box, "tol": 1e-3}),
        (".text", "Show `img` and, for every box, write `f\"{label} {score:.2f}\"` at (x, y - 2) — just above its top-left "
                  "corner — in black, font size 8. (No rectangles needed.)",
         "fig, ax = plt.subplots()\nax.imshow(img)\nfor bx, by, bw, bh, lbl, score in boxes:\n"
         "    ax.text(bx, by - 2, f'{lbl} {score:.2f}', color='black', fontsize=8)",
         {"post": "sorted((t['text'], t['xy'], t['color']) for t in fc.text_items()), [t.get_fontsize() for t in fc.ax().texts]"}),
        ("bbox", f"Show `img` and label each box with `f\"{{label}} {{score:.2f}}\"` at (x, y - 2) inside a colored tag: "
                 f"`bbox=dict(facecolor={c!r}, pad=1, edgecolor='none')`.",
         f"fig, ax = plt.subplots()\nax.imshow(img)\nfor bx, by, bw, bh, lbl, score in boxes:\n"
         f"    ax.text(bx, by - 2, f'{{lbl}} {{score:.2f}}', bbox=dict(facecolor={c!r}, pad=1, edgecolor='none'))",
         {"post": "sorted((t.get_text(), fc.r(t.get_position(), 3), fc.hexcolor(t.get_bbox_patch().get_facecolor()), "
                  "fc.hexcolor(t.get_bbox_patch().get_edgecolor())) for t in fc.ax().texts)", "require": ["bbox"]}),
        ("Rectangle", f"Show `img` and outline each box (no fill, line width 2) in 'lime' if its score is at least {thr}, "
                      f"else in 'red'.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         f"for bx, by, bw, bh, lbl, score in boxes:\n    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, linewidth=2, "
         f"edgecolor='lime' if score >= {thr} else 'red'))", {"post": PRE + "_boxes(None, 'ec')"}),
        ("Rectangle", f"Show `img` and outline (no fill) only the boxes whose score is at least {thr}. Then hide the axes "
                      f"with `ax.axis('off')`.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         f"for bx, by, bw, bh, lbl, score in boxes:\n    if score >= {thr}:\n"
         f"        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False))\nax.axis('off')",
         {"post": PRE + "_boxes(None, 'fill'), fc.ax().axison", "require": ["Rectangle", "axis"]}),
        (".savefig", f"Show `img` with every box outlined (unfilled `Rectangle`, edge color {c!r}), hide the axes, save "
                     f"the figure as `boxes.png` at 30 dpi and close everything with `plt.close('all')`. Return "
                     f"`(plt.imread('boxes.png').shape, len(plt.get_fignums()))`.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots(figsize=(4, 3))\nax.imshow(img)\n"
         f"for bx, by, bw, bh, lbl, score in boxes:\n    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor={c!r}))\n"
         f"ax.axis('off')\nfig.savefig('boxes.png', dpi=30)\nplt.close('all')\n"
         f"result = (plt.imread('boxes.png').shape, len(plt.get_fignums()))",
         {"require": ["savefig", "close"]}),
        (".patches", "Show `img`, add one unfilled `Rectangle` per box, then read the boxes **back** from the Axes and "
                     "return their areas (`p.get_width() * p.get_height()` for `p` in `ax.patches`).",
         "from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img)\n"
         "for bx, by, bw, bh, lbl, score in boxes:\n    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False))\n"
         "result = [p.get_width() * p.get_height() for p in ax.patches]"),
    ]
    return catalog(rng, items, code, f"`img` is an {H} × {W} RGB image and `boxes` holds detections (x, y, w, h, label, "
                                     f"score), with (x, y) the top-left pixel. Create `fig, ax`. Import `Rectangle` from "
                                     f"`matplotlib.patches` when you need it.")


@template(M, "easy")
def collections_intro(rng):
    n = rng.randint(6, 12)
    cmap = rng.choice(["viridis", "plasma", "magma", "cividis"])
    lw = rng.choice([0.8, 1.5, 2])
    a = rng.choice([0.4, 0.6, 0.8])
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nx = np.linspace(0, 1, {n})\ny = np.round(g.random({n}), 3)\n"
             f"segs = [[(0, i / {n}), (1, i / {n} + 0.05)] for i in range({n})]")
    lc_post = "[fc.r(s, 4) for c in fc.ax().collections for s in c.get_segments()]"
    items = [
        ("LineCollection", f"Add `segs` as one `LineCollection` colored along the {cmap!r} colormap (one color per "
                           f"segment: `mpl.colormaps[{cmap!r}](np.linspace(0, 1, {n}))`).",
         f"from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\n"
         f"ax.add_collection(LineCollection(segs, colors=mpl.colormaps[{cmap!r}](np.linspace(0, 1, {n}))))",
         {"post": f"{lc_post}, [fc.hexcolor(c) for c in fc.ax().collections[0].get_colors()]",
          "require": ["LineCollection", "colormaps"]}),
        ("LineCollection", f"Add `segs` as one `LineCollection` with line width {lw} and alpha {a}, all in 'k'.",
         f"from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\n"
         f"ax.add_collection(LineCollection(segs, colors='k', linewidths={lw}, alpha={a}))",
         {"post": f"{lc_post}, [fc.r(c.get_linewidths(), 3) for c in fc.ax().collections], "
                  "[(fc.hexcolor(c.get_colors()), fc.alpha(c.get_colors())) for c in fc.ax().collections]"}),
        ("LineCollection", "Turn the curve (`x`, `y`) into segments — segment i joins point i to point i + 1, an array of "
                           "shape (n - 1, 2, 2): `pts = np.column_stack([x, y])`, `np.stack([pts[:-1], pts[1:]], axis=1)` "
                           "— and add them as a `LineCollection` colored by `y[:-1]` (`array=y[:-1]`, "
                           f"`cmap={cmap!r}`).",
         f"from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\npts = np.column_stack([x, y])\n"
         f"seg = np.stack([pts[:-1], pts[1:]], axis=1)\nax.add_collection(LineCollection(seg, array=y[:-1], cmap={cmap!r}))",
         {"post": f"{lc_post}, [fc.r(c.get_array(), 4) for c in fc.ax().collections], [c.get_cmap().name for c in fc.ax().collections]"}),
        (".autoscale_view", "Collections don't rescale the view on their own: add `segs` as a `LineCollection` with x "
                            "stretched to 0 … 10 (`[[(10 * x0, y0), (10 * x1, y1)] for (x0, y0), (x1, y1) in segs]`), then "
                            "call `ax.autoscale_view()` so the limits fit them.",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\n"
         "ax.add_collection(LineCollection([[(10 * x0, y0), (10 * x1, y1)] for (x0, y0), (x1, y1) in segs]))\nax.autoscale_view()",
         {"post": f"{lc_post}, fc.limits()", "require": ["autoscale_view|autoscale"]}),
        ("PatchCollection", f"Add a circle of radius 0.03 at every point (`x[i]`, `y[i]`) as **one** `PatchCollection` "
                            f"with face color 'C1' and edge color 'k'.",
         "from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "ax.add_collection(PatchCollection([Circle((a, b), 0.03) for a, b in zip(x, y)], facecolor='C1', edgecolor='k'))",
         {"post": "c = fc.ax().collections\n[len(k.get_paths()) for k in c], [(fc.hexcolor(k.get_facecolor()), "
                  "fc.hexcolor(k.get_edgecolor())) for k in c], sorted(fc.r(p.get_extents(k.get_transform() - fc.ax().transData)"
                  ".bounds, 3) for k in c for p in k.get_paths())"}),
        ("PatchCollection", f"Add a circle of radius 0.03 at every point (`x[i]`, `y[i]`) as one `PatchCollection` colored "
                            f"by `y` through the {cmap!r} colormap: `PatchCollection(circles, cmap={cmap!r})`, then "
                            f"`pc.set_array(y)`.",
         f"from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         f"pc = PatchCollection([Circle((a, b), 0.03) for a, b in zip(x, y)], cmap={cmap!r})\npc.set_array(y)\n"
         f"ax.add_collection(pc)",
         {"post": "c = fc.ax().collections\n[len(k.get_paths()) for k in c], [fc.r(k.get_array(), 4) for k in c], "
                  "[k.get_cmap().name for k in c]", "require": ["PatchCollection", "set_array"]}),
        ("match_original", "A `PatchCollection` normally restyles all its patches alike. Make circles of radius 0.03 at the "
                           "first three points with face colors 'C0', 'C1', 'C2' and keep those colors with "
                           "`match_original=True`.",
         "from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "circles = [Circle((x[i], y[i]), 0.03, facecolor=f'C{i}') for i in range(3)]\n"
         "ax.add_collection(PatchCollection(circles, match_original=True))",
         {"post": "[[fc.hexcolor(c) for c in k.get_facecolor()] for k in fc.ax().collections]", "require": ["match_original"]}),
        (".collections", f"One collection or many patches? Add {n} circles (radius 0.03, at the points) as one "
                         f"`PatchCollection`, and return `[len(ax.collections), len(ax.patches)]`.",
         "from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "ax.add_collection(PatchCollection([Circle((a, b), 0.03) for a, b in zip(x, y)]))\n"
         "result = [len(ax.collections), len(ax.patches)]", {"require": ["PatchCollection"]}),
        (".get_segments", "Add `segs` as a `LineCollection` and read it back: return the number of segments and the first "
                          "segment's points (`lc.get_segments()`), as `[count, first.tolist()]`.",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\nlc = LineCollection(segs)\n"
         "ax.add_collection(lc)\nfirst = lc.get_segments()[0]\nresult = [len(lc.get_segments()), first.tolist()]"),
    ]
    return catalog(rng, items, setup, f"`segs` holds {n} line segments (each a pair of points), and `x`, `y` a curve of "
                                      f"{n} points. Create `fig, ax`.")


# ================================================================= medium

@template(M, "medium")
def detection_overlays(rng):
    code, gt, pred = _gt_pred(rng)
    cg, cp = rng.choice([("lime", "red"), ("green", "magenta"), ("cyan", "orange")])
    thr = rng.choice([0.5, 0.6, 0.7])
    if all(p[4] < thr for p in pred):
        thr = 0.4
    both = PRE + "fc.images(), _boxes(None, 'ec')"
    items = [
        ("Rectangle", f"Drill: draw the ground-truth boxes `gt` in {cg!r} and the predicted boxes `pred` in {cp!r} on the "
                      f"image (`ax.imshow(img, cmap='gray')`), all unfilled.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for x, y, w, h in gt:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"for x, y, w, h, s in pred:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cp!r}))",
         {"post": both}),
        ("Rectangle", f"Draw the ground truth dashed (`ls='--'`) in {cg!r} and the predictions solid in {cp!r} (all "
                      f"unfilled, line width 2) over `ax.imshow(img, cmap='gray')`.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for x, y, w, h in gt:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, ls='--', lw=2, edgecolor={cg!r}))\n"
         f"for x, y, w, h, s in pred:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=2, edgecolor={cp!r}))",
         {"post": PRE + "_boxes(None, 'ec', 'ls', 'lw')"}),
        (".legend", f"Draw `gt` in {cg!r} and `pred` in {cp!r} (unfilled) over the image and add a legend with the "
                    f"entries 'ground truth' and 'prediction'. Patches without labels don't show up, so pass proxy "
                    f"handles: `Patch(facecolor='none', edgecolor=..., label=...)`.",
         f"from matplotlib.patches import Rectangle, Patch\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for x, y, w, h in gt:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"for x, y, w, h, s in pred:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cp!r}))\n"
         f"ax.legend(handles=[Patch(facecolor='none', edgecolor={cg!r}, label='ground truth'), "
         f"Patch(facecolor='none', edgecolor={cp!r}, label='prediction')])",
         {"post": PRE + "_boxes(None, 'ec'), fc.legend()", "require": ["Rectangle", "legend"]}),
        (".text", f"Draw each prediction (unfilled, {cp!r}) with its score written at its top-left corner as "
                  f"`f'{{s:.2f}}'` (at (x, y - 1), white text, font size 7).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for x, y, w, h, s in pred:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cp!r}))\n"
         f"    ax.text(x, y - 1, f'{{s:.2f}}', color='white', fontsize=7)",
         {"post": PRE + "_boxes(None, 'ec'), sorted((t['text'], t['xy'], t['color']) for t in fc.text_items())"}),
        ("Rectangle", f"Only show confident predictions: draw the predictions with score ≥ {thr} (unfilled, {cp!r}) and "
                      f"all ground-truth boxes (unfilled, {cg!r}).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for x, y, w, h in gt:\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"for x, y, w, h, s in pred:\n    if s >= {thr}:\n"
         f"        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cp!r}))", {"post": both}),
        ("transAxes", f"`gt[i]` and `pred[i]` belong together. Compute each pair's IoU (intersection area / union area) "
                      f"and write `'IoU ' + ', '.join(f'{{v:.2f}}' for v in ious)` in the top-left corner of the Axes: "
                      f"(0.02, 0.98) in axes coordinates, `va='top'`. Draw both box sets too (unfilled, {cg!r} / {cp!r}).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nious = []\n"
         f"for (x, y, w, h), (px, py, pw, ph, s) in zip(gt, pred):\n"
         f"    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"    ax.add_patch(Rectangle((px, py), pw, ph, fill=False, edgecolor={cp!r}))\n"
         f"    iw = max(0, min(x + w, px + pw) - max(x, px))\n    ih = max(0, min(y + h, py + ph) - max(y, py))\n"
         f"    inter = iw * ih\n    ious.append(inter / (w * h + pw * ph - inter))\n"
         f"ax.text(0.02, 0.98, 'IoU ' + ', '.join(f'{{v:.2f}}' for v in ious), transform=ax.transAxes, va='top')",
         {"post": PRE + "_boxes(None, 'ec'), fc.texts(), [_at(t) for t in fc.ax().texts], [t.get_va() for t in fc.ax().texts]",
          "require": ["Rectangle", "transAxes"]}),
        ("plt.subplots", f"Two panels side by side (`plt.subplots(1, 2)`): ground truth on the left (titled 'ground "
                         f"truth', {cg!r} boxes), predictions on the right (titled 'prediction', {cp!r} boxes), the same "
                         f"image in both (`cmap='gray'`), boxes unfilled, axes hidden.",
         f"from matplotlib.patches import Rectangle\nfig, (a1, a2) = plt.subplots(1, 2)\n"
         f"for ax, bxs, c, t in ((a1, gt, {cg!r}, 'ground truth'), (a2, [p[:4] for p in pred], {cp!r}, 'prediction')):\n"
         f"    ax.imshow(img, cmap='gray')\n    for x, y, w, h in bxs:\n"
         f"        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c))\n    ax.set_title(t)\n    ax.axis('off')",
         {"post": PRE + "[(fc.images(a), _boxes(a, 'ec'), a.get_title(), a.axison) for a in fc.axes()]"}),
    ]
    return catalog(rng, items, code, f"`img` is a grayscale image, `gt` the ground-truth boxes (x, y, w, h) and `pred` "
                                     f"the model's boxes (x, y, w, h, score). Rectangles take (x, y) = the top-left "
                                     f"pixel on an image. Create `fig, ax`.")


@template(M, "medium")
def network_diagrams(rng):
    layers = rng.choice([[2, 3, 1], [3, 4, 2], [2, 4, 3], [3, 3, 2], [4, 3, 2], [2, 3, 3, 1]])
    r = rng.choice([0.15, 0.2])
    s = seeded(rng)
    n_edges = sum(a * b for a, b in zip(layers, layers[1:]))
    setup = (f"layers = {layers}   # nodes per layer, input first\n"
             f"pos = [[(i, j - (n - 1) / 2) for j in range(n)] for i, n in enumerate(layers)]   # node centers per layer\n"
             f"weights = [np.round(np.random.default_rng({s} + i).normal(size=(a, b)), 2) "
             f"for i, (a, b) in enumerate(zip(layers, layers[1:]))]")
    nodes = (f"from matplotlib.patches import Circle\nfig, ax = plt.subplots()\nfor layer in pos:\n    for x, y in layer:\n"
             f"        ax.add_patch(Circle((x, y), {r}, facecolor='white', edgecolor='k', zorder=3))\n")
    edges = ("segs = [[p, q] for a, b in zip(pos, pos[1:]) for p in a for q in b]\n")
    circ = PRE + "_circles()"
    items = [
        ("Circle", f"Drill: draw the network's nodes as circles of radius {r} at `pos` (white face, black edge), make the "
                   f"aspect equal and hide the axes.",
         f"{nodes}ax.set_aspect('equal')\nax.axis('off')\nax.autoscale_view()",
         {"post": PRE + "_circles(None, 'fc', 'ec'), fc.ax().get_aspect(), fc.ax().axison", "require": ["Circle", "set_aspect"]}),
        ("LineCollection", f"Draw the nodes (circles of radius {r} at `pos`) and **all** edges between consecutive layers "
                           f"as one `LineCollection` (gray, from center to center: {n_edges} segments). Set the aspect "
                           f"equal.",
         f"from matplotlib.collections import LineCollection\n{nodes}{edges}"
         f"ax.add_collection(LineCollection(segs, colors='gray', zorder=1))\nax.set_aspect('equal')\nax.autoscale_view()",
         {"post": circ + ", _segs(), fc.ax().get_aspect()", "require": ["Circle", "LineCollection"]}),
        ("FancyArrowPatch", f"Draw the nodes (radius {r}) and connect every node to every node of the next layer with a "
                            f"`FancyArrowPatch` (`arrowstyle='->'`, `mutation_scale=10`), from center to center.",
         f"from matplotlib.patches import FancyArrowPatch\n{nodes}for a, b in zip(pos, pos[1:]):\n    for p in a:\n"
         f"        for q in b:\n            ax.add_patch(FancyArrowPatch(p, q, arrowstyle='->', mutation_scale=10))\n"
         f"ax.set_aspect('equal')\nax.autoscale_view()",
         {"post": PRE + "_circles(), sorted(_pp(p, 'arrow')['arrow'][0] for p in fc.ax().patches if type(p).__name__ == "
                  "'FancyArrowPatch'), sorted({_pp(p, 'arrow')['arrow'][1] for p in fc.ax().patches if type(p).__name__ == "
                  "'FancyArrowPatch'})", "require": ["FancyArrowPatch", "Circle"]}),
        ("Circle", f"Color the nodes by role: input layer 'C0', hidden layers 'C1', output layer 'C2' (circles of radius "
                   f"{r} at `pos`, aspect equal).",
         f"from matplotlib.patches import Circle\nfig, ax = plt.subplots()\nfor i, layer in enumerate(pos):\n"
         f"    c = 'C0' if i == 0 else 'C2' if i == len(pos) - 1 else 'C1'\n    for x, y in layer:\n"
         f"        ax.add_patch(Circle((x, y), {r}, facecolor=c))\nax.set_aspect('equal')\nax.autoscale_view()",
         {"post": PRE + "_circles(None, 'fc'), fc.ax().get_aspect()"}),
        (".text", f"Draw the nodes (circles of radius {r}, white face, black edge) and write a name in each, centered "
                  f"(`ha='center', va='center'`): 'x1', 'x2', … in the input layer, 'y1', … in the output layer and "
                  f"'h1', … in hidden layers (numbering restarts in each layer).",
         f"{nodes}for i, layer in enumerate(pos):\n    p = 'x' if i == 0 else 'y' if i == len(pos) - 1 else 'h'\n"
         f"    for j, (x, y) in enumerate(layer):\n        ax.text(x, y, f'{{p}}{{j + 1}}', ha='center', va='center')\n"
         f"ax.set_aspect('equal')\nax.autoscale_view()",
         {"post": circ + ", sorted((t['text'], t['xy'], t['ha'], t['va']) for t in fc.text_items())"}),
        ("LineCollection", "Show the weights on the edges: one `LineCollection` with an edge from `pos[i][a]` to "
                           "`pos[i + 1][b]` for every weight `weights[i][a, b]`, colored 'C0' when the weight is positive "
                           "and 'C3' otherwise, with line width `2 * abs(w)`. (No nodes needed; call `ax.autoscale_view()`.)",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\nsegs, colors, widths = [], [], []\n"
         "for i, W in enumerate(weights):\n    for a in range(W.shape[0]):\n        for b in range(W.shape[1]):\n"
         "            segs.append([pos[i][a], pos[i + 1][b]])\n            colors.append('C0' if W[a, b] > 0 else 'C3')\n"
         "            widths.append(2 * abs(W[a, b]))\nax.add_collection(LineCollection(segs, colors=colors, linewidths=widths))\n"
         "ax.autoscale_view()",
         {"post": "c = fc.ax().collections[0]\nsorted(zip([fc.r(s, 4) for s in c.get_segments()], "
                  "[fc.hexcolor(k) for k in c.get_colors()], fc.r(c.get_linewidths(), 3)))", "require": ["LineCollection"]}),
        ("PatchCollection", f"Draw all nodes as **one** `PatchCollection` of circles (radius {r}) colored by layer index "
                            f"through the 'viridis' colormap (`pc.set_array(...)` with the layer number of each node), and "
                            f"set the aspect equal.",
         f"from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         f"circles = [Circle(p, {r}) for layer in pos for p in layer]\n"
         f"pc = PatchCollection(circles, cmap='viridis')\npc.set_array(np.array([i for i, layer in enumerate(pos) for p in layer]))\n"
         f"ax.add_collection(pc)\nax.set_aspect('equal')\nax.autoscale_view()",
         {"post": "c = fc.ax().collections[0]\nsorted(zip([fc.r(p.get_extents(c.get_transform() - fc.ax().transData).bounds, 3) "
                  "for p in c.get_paths()], fc.r(c.get_array(), 3))), c.get_cmap().name, fc.ax().get_aspect()",
          "require": ["PatchCollection", "set_array"]}),
    ]
    return catalog(rng, items, setup, f"An MLP with layer sizes `layers` = {layers}: `pos[i]` lists the (x, y) centers of "
                                      f"layer i's nodes and `weights[i]` its weight matrix to layer i + 1. Create `fig, ax`; "
                                      f"patches/collections don't autoscale the view, so finish with `ax.autoscale_view()` "
                                      f"(or set limits) — only what's asked is checked.")


@template(M, "medium")
def shaded_bands(rng):
    code, desc, n = curves(rng)
    a = rng.randint(2, n // 3)
    b = rng.randint(n // 2, n - 2)
    lo, hi = sorted(round(rng.uniform(0.2, 1.5), 2) for _ in range(2))
    phases = [(1, a, "warmup"), (a, b, "main"), (b, n, "decay")]
    c = rng.choice(["gray", "C1", "C2", "gold"])
    al = rng.choice([0.1, 0.2, 0.3])
    plot = "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
    items = [
        ("blended_transform_factory", f"Drill: plot both curves and shade epochs {a} … {b} over the **full height** of the "
                                      f"Axes with a `Rectangle` in a blended transform (x in data, y in axes coordinates; "
                                      f"color {c!r}, alpha {al}).",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\n{plot}"
         f"trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         f"ax.add_patch(Rectangle(({a}, 0), {b - a}, 1, transform=trans, color={c!r}, alpha={al}))",
         {"post": _post("fc.lines()", "[_ext(p, 'blend') for p in fc.ax().patches]", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]"),
          "require": ["blended_transform_factory", "Rectangle"]}),
        (".axvspan", f"Same band, the shortcut way: plot both curves and shade epochs {a} … {b} with `ax.axvspan` (color "
                     f"{c!r}, alpha {al}). It builds the same blended rectangle for you.",
         f"{plot}ax.axvspan({a}, {b}, color={c!r}, alpha={al})",
         {"post": _post("fc.lines()", "[_ext(p, 'blend') for p in fc.ax().patches]", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]")}),
        ("blended_transform_factory", f"Plot both curves and mark the target loss region {lo} … {hi} as a **horizontal** "
                                      f"band across the full width: a `Rectangle` in a blended transform with x in axes "
                                      f"and y in data coordinates (`blended_transform_factory(ax.transAxes, ax.transData)`), "
                                      f"color {c!r}, alpha {al}.",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\n{plot}"
         f"trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)\n"
         f"ax.add_patch(Rectangle((0, {lo}), 1, {round(hi - lo, 2)}, transform=trans, color={c!r}, alpha={al}))",
         {"post": _post("fc.lines()", "[_ext(p, 'yblend') for p in fc.ax().patches]", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]")}),
        ("blended_transform_factory", f"Plot both curves and shade the three phases `phases = {phases}` as full-height bands "
                                      f"(colors 'C0', 'C1', 'C2', alpha 0.15), and label each just above the plot: its name "
                                      f"at x = the middle of the phase (data) and y = 1.01 (axes), `ha='center'`. One "
                                      f"blended transform does both.",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\nphases = {phases}\n{plot}"
         f"trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         f"for i, (s, e, name) in enumerate(phases):\n"
         f"    ax.add_patch(Rectangle((s, 0), e - s, 1, transform=trans, color=f'C{{i}}', alpha=0.15))\n"
         f"    ax.text((s + e) / 2, 1.01, name, transform=trans, ha='center')",
         {"post": _post("sorted([_ext(p, 'blend'), _pp(p, 'fc')['fc']] for p in fc.ax().patches)",
                        "sorted((t.get_text(), [fc.r(fc.ax().transData.inverted().transform(t.get_transform().transform("
                        "t.get_position()))[0], 3), _at(t)[1]], t.get_ha()) for t in fc.ax().texts)")}),
        ("transAxes", f"Plot both curves and shade the **right quarter** of the plot, whatever the data: a `Rectangle((0.75, "
                      f"0), 0.25, 1)` in axes coordinates (`transform=ax.transAxes`), color {c!r}, alpha {al}.",
         f"from matplotlib.patches import Rectangle\n{plot}"
         f"ax.add_patch(Rectangle((0.75, 0), 0.25, 1, transform=ax.transAxes, color={c!r}, alpha={al}))",
         {"post": _post("fc.lines()", "[_ext(p, 'axes') for p in fc.ax().patches]", "[_pp(p, 'fc', 'alpha') for p in fc.ax().patches]"),
          "require": ["transAxes", "Rectangle"]}),
        ("blended_transform_factory", "Plot both curves and mark the epoch with the lowest `val_loss` by a dashed black "
                                      "line spanning the full height: `ax.plot([e, e], [0, 1], transform=trans, ls='--', "
                                      "color='k')` with a data/axes blended transform.",
         f"import matplotlib.transforms as mtransforms\n{plot}e = epochs[val_loss.argmin()]\n"
         f"trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         f"ax.plot([e, e], [0, 1], transform=trans, ls='--', color='k')",
         {"post": "a = fc.ax()\nl = a.get_lines()[-1]\nd = l.get_transform().transform(l.get_xydata())\n"
                  "fc.lines(a)[:2], fc.r(a.transData.inverted().transform(d)[:, 0], 3), fc.r(a.transAxes.inverted().transform(d)[:, 1], 3), "
                  "l.get_linestyle(), fc.hexcolor(l.get_color())", "require": ["blended_transform_factory", "argmin"]}),
        (".text", f"Plot both curves and put a caption under the figure with `fig.text(0.5, 0.01, 'shaded: epochs {a}-{b}', "
                  f"ha='center', fontsize=8)` (figure coordinates), and shade epochs {a} … {b} with `ax.axvspan` "
                  f"(color {c!r}, alpha {al}).",
         f"{plot}ax.axvspan({a}, {b}, color={c!r}, alpha={al})\nfig.text(0.5, 0.01, 'shaded: epochs {a}-{b}', ha='center', fontsize=8)",
         {"post": _post("[_ext(p, 'blend') for p in fc.ax().patches]",
                        "[(t.get_text(), fc.r(t.get_position(), 3), t.get_ha(), t.get_fontsize()) for t in fc.fig().texts]"),
          "require": ["text", "axvspan"]}),
    ]
    return catalog(rng, items, code, f"{desc}. Import `Rectangle` / `matplotlib.transforms as mtransforms` as needed.")


@template(M, "medium")
def confidence_ellipses(rng):
    k = rng.randint(2, 3)
    nstd = rng.choice([1, 2, 3])
    s = seeded(rng)
    setup = (f"g = np.random.default_rng({s})\nclusters = []\nfor i in range({k}):\n    A = g.normal(size=(2, 2))\n"
             f"    clusters.append(np.round(g.normal(size=(60, 2)) @ A + g.uniform(-6, 6, 2), 3))")
    body = (f"from matplotlib.patches import Ellipse\nfig, ax = plt.subplots()\nfor i, P in enumerate(clusters):\n"
            f"    mu = P.mean(axis=0)\n    vals, vecs = np.linalg.eigh(np.cov(P.T))\n"
            f"    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))\n")
    ell = (f"    ax.add_patch(Ellipse(mu, 2 * {nstd} * np.sqrt(vals[1]), 2 * {nstd} * np.sqrt(vals[0]), angle=angle, "
           f"fill=False, edgecolor=f'C{{i}}'))\n")
    # an ellipse is pinned down by its center and A·Aᵀ of its patch transform (any width/height/angle convention)
    post = ("from matplotlib.patches import Ellipse\n"
            "def _cov(p):\n    A = p.get_patch_transform().get_matrix()[:2, :2]\n    return fc.r(A @ A.T, 3)\n"
            "sorted([fc.r(p.get_center(), 3), _cov(p), fc.hexcolor(p.get_edgecolor())] for p in fc.ax().patches "
            "if isinstance(p, Ellipse))")
    items = [
        ("Ellipse", f"For each cluster draw its {nstd}-σ confidence ellipse: centered at the mean, axes along the "
                    f"eigenvectors of `np.cov(P.T)`, full width `2·{nstd}·√λ` along each eigenvector, rotated to the "
                    f"major eigenvector's angle (degrees). Unfilled, edge color `f'C{{i}}'` for cluster i.",
         body + ell + "ax.autoscale_view()", {"post": post, "require": ["Ellipse", "eigh|eig"]}),
        ("Ellipse", f"Scatter each cluster in `f'C{{i}}'` (`s=8`) and draw its {nstd}-σ confidence ellipse in the same "
                    f"color (unfilled): mean center, eigen-decomposition of the covariance for the axes and angle.",
         body + "    ax.scatter(P[:, 0], P[:, 1], s=8, color=f'C{i}')\n" + ell,
         {"post": post + ", [fc.scatter(fc.ax(), i, 6, 'color') for i in range(fc.n_scatters())]", "require": ["Ellipse", "scatter"]}),
        ("Ellipse", f"Draw the 1-, 2- and 3-σ ellipses of the **first** cluster only (unfilled, edge colors 'C0', 'C1', "
                    f"'C2' for 1, 2, 3 σ), same construction (mean, eigenvectors of the covariance).",
         "from matplotlib.patches import Ellipse\nfig, ax = plt.subplots()\nP = clusters[0]\nmu = P.mean(axis=0)\n"
         "vals, vecs = np.linalg.eigh(np.cov(P.T))\nangle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))\n"
         "for k in (1, 2, 3):\n    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=angle, "
         "fill=False, edgecolor=f'C{k - 1}'))\nax.autoscale_view()", {"post": post}),
        ("Ellipse", f"Shade each cluster's {nstd}-σ ellipse instead of outlining it: face color `f'C{{i}}'`, alpha 0.3 (no "
                    f"edge needed). Return nothing; the patches are checked (center, shape, face color).",
         body + f"    ax.add_patch(Ellipse(mu, 2 * {nstd} * np.sqrt(vals[1]), 2 * {nstd} * np.sqrt(vals[0]), angle=angle, "
                f"facecolor=f'C{{i}}', alpha=0.3))\nax.autoscale_view()",
         {"post": post.replace("p.get_edgecolor()", "p.get_facecolor()") + ", [fc.alpha(p.get_facecolor()) for p in fc.ax().patches]"}),
    ]
    return catalog(rng, items, setup, f"`clusters` holds {k} point clouds (60 × 2 each), e.g. classes in a 2-D "
                                      f"embedding. Create `fig, ax`.")


@template(M, "medium")
def vit_patch_grid(rng):
    H, W = rng.choice([(32, 48), (32, 32), (48, 64), (24, 40)])
    p = rng.choice([8, 8, 16]) if H % 16 == 0 and W % 16 == 0 else 8
    nr, nc = H // p, W // p
    i, j = rng.randrange(nr), rng.randrange(nc)
    s = seeded(rng)
    c = rng.choice(["red", "yellow", "cyan"])
    setup = (f"g = np.random.default_rng({s})\nimg = g.random(({H}, {W}))\np = {p}   # patch size in pixels\n"
             f"att = np.round(g.random(({nr}, {nc})), 3)   # attention of the [CLS] token to each patch")
    grid = (f"segs = [[(x, -0.5), (x, {H} - 0.5)] for x in np.arange(-0.5, {W}, p)] + "
            f"[[(-0.5, y), ({W} - 0.5, y)] for y in np.arange(-0.5, {H}, p)]\n")
    items = [
        ("LineCollection", f"A ViT cuts the image into {p} × {p} patches. Show `img` (`cmap='gray'`) and draw the patch "
                           f"grid as one `LineCollection` (color {c!r}): vertical lines at x = -0.5, {p} - 0.5, … up to "
                           f"{W} - 0.5 from top to bottom, horizontal lines at y = -0.5, … up to {H} - 0.5 (pixel centers "
                           f"are at integers, so edges sit at half-integers).",
         f"from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n{grid}"
         f"ax.add_collection(LineCollection(segs, colors={c!r}))",
         {"post": PRE + "fc.images(), _segs(), [fc.hexcolor(c.get_colors()) for c in fc.ax().collections]"}),
        ("Rectangle", f"Show `img` (`cmap='gray'`) and outline patch (row {i}, column {j}) with an unfilled {c!r} "
                      f"`Rectangle` (line width 2). Patch (r, c) covers pixels r·p … r·p + p - 1, so its corner is at "
                      f"(c·p - 0.5, r·p - 0.5).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"ax.add_patch(Rectangle(({j} * p - 0.5, {i} * p - 0.5), p, p, fill=False, edgecolor={c!r}, linewidth=2))",
         {"post": PRE + "fc.images(), _boxes(None, 'ec', 'lw')"}),
        ("PatchCollection", "Overlay the attention map: show `img` (`cmap='gray'`) and add one `Rectangle` per patch "
                            "(corner (c·p - 0.5, r·p - 0.5), size p × p) in a single `PatchCollection` colored by "
                            "`att.ravel()` (row by row) with `cmap='inferno'` and `alpha=0.5`.",
         "from matplotlib.patches import Rectangle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "ax.imshow(img, cmap='gray')\nrects = [Rectangle((c * p - 0.5, r * p - 0.5), p, p) for r in range(att.shape[0]) "
         "for c in range(att.shape[1])]\npc = PatchCollection(rects, cmap='inferno', alpha=0.5)\npc.set_array(att.ravel())\n"
         "ax.add_collection(pc)",
         {"post": "c = fc.ax().collections[0]\nfc.images(), sorted(zip([fc.r(q.get_extents(c.get_transform() - fc.ax().transData)"
                  ".bounds, 3) for q in c.get_paths()], fc.r(c.get_array(), 3))), c.get_cmap().name, c.get_alpha()",
          "require": ["PatchCollection", "set_array"]}),
        (".text", f"Show `img` (`cmap='gray'`) and number the patches row by row (0, 1, 2, …): write each number at its "
                  f"patch center (c·p + p/2 - 0.5, r·p + p/2 - 0.5), centered, color {c!r}, font size 6.",
         f"fig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\nk = 0\nfor r in range(att.shape[0]):\n"
         f"    for c in range(att.shape[1]):\n        ax.text(c * p + p / 2 - 0.5, r * p + p / 2 - 0.5, str(k), ha='center', "
         f"va='center', color={c!r}, fontsize=6)\n        k += 1",
         {"post": "fc.images(), sorted((t['text'], t['xy'], t['ha'], t['va'], t['color']) for t in fc.text_items())"}),
        ("Rectangle", f"Highlight the top-3 attended patches: show `img` (`cmap='gray'`) and outline the 3 patches with "
                      f"the largest `att` (unfilled {c!r} rectangles, line width 2).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for k in np.argsort(att.ravel())[-3:]:\n    r, c = divmod(k, att.shape[1])\n"
         f"    ax.add_patch(Rectangle((c * p - 0.5, r * p - 0.5), p, p, fill=False, edgecolor={c!r}, linewidth=2))",
         {"post": PRE + "fc.images(), _boxes(None, 'ec', 'lw')", "require": ["Rectangle", "argsort"]}),
        ("imshow", f"Show the attention as a heat overlay that lines up with the image: `ax.imshow(img, cmap='gray')`, then "
                   f"`ax.imshow(np.kron(att, np.ones((p, p))), cmap='jet', alpha=0.4)` (each patch value blown up to {p} × "
                   f"{p} pixels), and outline the most attended patch in {c!r} (unfilled).",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"ax.imshow(np.kron(att, np.ones((p, p))), cmap='jet', alpha=0.4)\nr, c = divmod(att.argmax(), att.shape[1])\n"
         f"ax.add_patch(Rectangle((c * p - 0.5, r * p - 0.5), p, p, fill=False, edgecolor={c!r}))",
         {"post": PRE + "fc.images(), [i.get_alpha() for i in fc.ax().get_images()], _boxes(None, 'ec')",
          "require": ["kron", "Rectangle"]}),
    ]
    return catalog(rng, items, setup, f"`img` is a {H} × {W} image, `p` = {p} the patch size and `att` ({nr} × {nc}) the "
                                      f"attention per patch. Create `fig, ax`.")


@template(M, "medium")
def gpu_timeline(rng):
    G = rng.randint(2, 4)
    jobs = []
    for gpu in range(G):
        t = round(rng.uniform(0, 2), 1)
        for _ in range(rng.randint(1, 3)):
            d = round(rng.uniform(1, 5), 1)
            jobs.append((gpu, t, d, f"job{len(jobs)}", rng.choice(["done", "done", "failed"])))
            t = round(t + d + rng.uniform(0.2, 1.5), 1)
    names = [f"gpu{i}" for i in range(G)]
    now = round(max(j[1] + j[2] for j in jobs) * rng.uniform(0.4, 0.8), 1)
    setup = f"jobs = {lit(jobs)}   # (gpu, start, duration, name, status)\ngpus = {lit(names)}"
    rect = "ax.add_patch(Rectangle((start, gpu - 0.4), dur, 0.8"
    head = "from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nfor gpu, start, dur, name, status in jobs:\n"
    items = [
        ("Rectangle", "Draw a job timeline: one `Rectangle((start, gpu - 0.4), dur, 0.8)` per job, GPU rows on the y-axis "
                      "labelled with `gpus` (ticks at 0, 1, …), and x limits from 0 to the end of the last job.",
         f"{head}    {rect}))\nax.set_yticks(range(len(gpus)), gpus)\nax.set_xlim(0, max(s + d for g, s, d, n, st in jobs))",
         {"post": PRE + "_boxes(), fc.ticks(fc.ax(), 'y'), fc.limits()['xlim']"}),
        ("Rectangle", "Draw one `Rectangle((start, gpu - 0.4), dur, 0.8)` per job, colored 'C2' when `status == 'done'` "
                      "and 'C3' when it failed, and set the y limits to -0.5 … len(gpus) - 0.5.",
         f"{head}    {rect}, color='C2' if status == 'done' else 'C3'))\nax.set_ylim(-0.5, len(gpus) - 0.5)",
         {"post": PRE + "_boxes(None, 'fc'), fc.limits()['ylim']"}),
        (".text", "Draw the job rectangles (`Rectangle((start, gpu - 0.4), dur, 0.8)`, face color 'lightgray') and write "
                  "each job's name at its center (`ha='center', va='center'`).",
         f"{head}    {rect}, facecolor='lightgray'))\n    ax.text(start + dur / 2, gpu, name, ha='center', va='center')\n"
         f"ax.autoscale_view()", {"post": PRE + "_boxes(None, 'fc'), sorted((t['text'], t['xy'], t['ha'], t['va']) for t in fc.text_items())"}),
        (".axvline", f"Draw the job rectangles (`Rectangle((start, gpu - 0.4), dur, 0.8)`) and mark the current time "
                     f"t = {now} with a red vertical line over the full height (`ax.axvline`), then shade everything "
                     f"after it with `ax.axvspan({now}, end, color='gray', alpha=0.2)` where `end` is the end of the "
                     f"last job.",
         f"{head}    {rect}))\nend = max(s + d for g, s, d, n, st in jobs)\nax.axvline({now}, color='red')\n"
         f"ax.axvspan({now}, end, color='gray', alpha=0.2)",
         {"post": PRE + "_boxes(), [fc.r(l.get_xdata(), 3) for l in fc.ax().get_lines()], "
                  "[_ext(p, 'blend') for p in fc.ax().patches if p.get_alpha() == 0.2]", "require": ["axvline", "axvspan"]}),
        ("Rectangle", "Return each GPU's busy time (sum of its jobs' durations, rounded to 1 decimal) as a list in GPU "
                      "order, after drawing the job rectangles (`Rectangle((start, gpu - 0.4), dur, 0.8)`) and reading "
                      "the durations **back from the patches** (`p.get_width()`, row from `p.get_y()`).",
         f"{head}    {rect}))\nbusy = [0.0] * len(gpus)\nfor p in ax.patches:\n"
         f"    busy[round(p.get_y() + 0.4)] += p.get_width()\nresult = [round(b, 1) for b in busy]",
         {"require": ["Rectangle", "get_width"]}),
        ("PatchCollection", "Draw all jobs as **one** `PatchCollection` of `Rectangle((start, gpu - 0.4), dur, 0.8)` with "
                            "face color 'C0' and edge color 'k', then `ax.autoscale_view()`.",
         "from matplotlib.patches import Rectangle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "rects = [Rectangle((s, g - 0.4), d, 0.8) for g, s, d, n, st in jobs]\n"
         "ax.add_collection(PatchCollection(rects, facecolor='C0', edgecolor='k'))\nax.autoscale_view()",
         {"post": "c = fc.ax().collections[0]\nsorted(fc.r(q.get_extents(c.get_transform() - fc.ax().transData).bounds, 3) "
                  "for q in c.get_paths()), fc.hexcolor(c.get_facecolor()), fc.hexcolor(c.get_edgecolor()), fc.limits()",
          "require": ["PatchCollection"]}),
    ]
    return catalog(rng, items, setup, f"`jobs` lists training jobs on {G} GPUs as (gpu, start, duration, name, status) "
                                      f"(hours) and `gpus` the row names. Create `fig, ax`.")


@template(M, "medium")
def wedge_charts(rng):
    _, labs = T.labels(rng, min_classes=3)
    k = rng.randint(3, min(5, len(labs)))
    counts = [rng.randint(5, 60) for _ in range(k)]
    names = list(labs[:k])
    acc = round(rng.uniform(0.55, 0.97), 3)
    width = rng.choice([0.3, 0.4])
    setup = f"names = {lit(names)}\ncounts = np.array({counts})   # samples per class\nacc = {acc}"
    pie = ("from matplotlib.patches import Wedge\nfig, ax = plt.subplots()\nfrac = counts / counts.sum()\n"
           "starts = 90 + 360 * np.concatenate([[0], np.cumsum(frac)[:-1]])\n")
    wedges = PRE + "sorted([fc.r(list(p.center) + [p.r, p.theta1, p.theta2], 3), _pp(p, 'fc', 'width')] for p in fc.ax().patches)"
    items = [
        ("Wedge", "Build a pie chart of the class balance by hand: one `Wedge((0, 0), 1, start, start + 360 * frac)` per "
                  "class, starting at 90° and going counter-clockwise in class order, colored `f'C{i}'`. Make the aspect "
                  "equal and set both limits to -1.2 … 1.2.",
         f"{pie}for i, (s, f) in enumerate(zip(starts, frac)):\n    ax.add_patch(Wedge((0, 0), 1, s, s + 360 * f, color=f'C{{i}}'))\n"
         f"ax.set_aspect('equal')\nax.set_xlim(-1.2, 1.2)\nax.set_ylim(-1.2, 1.2)",
         {"post": wedges + ", fc.ax().get_aspect(), fc.limits()"}),
        ("Wedge", f"Same pie as a **donut**: wedges of radius 1 with `width={width}`, from 90° counter-clockwise in class "
                  f"order, colored `f'C{{i}}'`, aspect equal, limits -1.2 … 1.2.",
         f"{pie}for i, (s, f) in enumerate(zip(starts, frac)):\n"
         f"    ax.add_patch(Wedge((0, 0), 1, s, s + 360 * f, width={width}, color=f'C{{i}}'))\n"
         f"ax.set_aspect('equal')\nax.set_xlim(-1.2, 1.2)\nax.set_ylim(-1.2, 1.2)", {"post": wedges}),
        (".text", "Draw the pie wedges (radius 1, from 90° counter-clockwise, colors `f'C{i}'`) and write each class name "
                  "outside its slice: at radius 1.15 along the slice's middle angle, `ha='center', va='center'`.",
         f"{pie}for i, (s, f) in enumerate(zip(starts, frac)):\n    ax.add_patch(Wedge((0, 0), 1, s, s + 360 * f, color=f'C{{i}}'))\n"
         f"    mid = np.radians(s + 180 * f)\n    ax.text(1.15 * np.cos(mid), 1.15 * np.sin(mid), names[i], ha='center', va='center')\n"
         f"ax.set_aspect('equal')\nax.set_xlim(-1.5, 1.5)\nax.set_ylim(-1.5, 1.5)",
         {"post": wedges + ", sorted((t['text'], t['xy']) for t in fc.text_items())"}),
        ("Wedge", "Explode the largest class: same pie (radius 1, from 90° counter-clockwise, colors `f'C{i}'`), but "
                  "the wedge of the class with the most samples is centered 0.1 away from the origin along its middle "
                  "angle.",
         f"{pie}big = counts.argmax()\nfor i, (s, f) in enumerate(zip(starts, frac)):\n    c = (0, 0)\n    if i == big:\n"
         f"        mid = np.radians(s + 180 * f)\n        c = (0.1 * np.cos(mid), 0.1 * np.sin(mid))\n"
         f"    ax.add_patch(Wedge(c, 1, s, s + 360 * f, color=f'C{{i}}'))\nax.set_aspect('equal')\nax.set_xlim(-1.3, 1.3)\n"
         f"ax.set_ylim(-1.3, 1.3)", {"post": wedges}),
        ("Wedge", f"An accuracy gauge: a gray half-ring `Wedge((0, 0), 1, 0, 180, width=0.25, color='lightgray')`, then on "
                  f"top a 'C2' half-ring segment of the same radius/width that covers the fraction `acc` of it **from "
                  f"the left** (from 180 - 180·acc to 180 degrees), and the text `f'{{acc:.0%}}'` centered at (0, 0.2). "
                  f"Aspect equal, x limits -1.1 … 1.1, y limits -0.1 … 1.1.",
         "from matplotlib.patches import Wedge\nfig, ax = plt.subplots()\n"
         "ax.add_patch(Wedge((0, 0), 1, 0, 180, width=0.25, color='lightgray'))\n"
         "ax.add_patch(Wedge((0, 0), 1, 180 - 180 * acc, 180, width=0.25, color='C2'))\n"
         "ax.text(0, 0.2, f'{acc:.0%}', ha='center')\nax.set_aspect('equal')\nax.set_xlim(-1.1, 1.1)\nax.set_ylim(-0.1, 1.1)",
         {"post": wedges + ", [(t['text'], t['xy'], t['ha']) for t in fc.text_items()], fc.limits()"}),
        ("Arc", f"Draw a protractor-style angle marker: two unit rays from the origin, at 0° and at `360 * acc` degrees "
                f"(lines with `ax.plot`), and an `Arc((0, 0), 0.6, 0.6, theta1=0, theta2=360 * acc)` between them. Aspect "
                f"equal, limits -1.1 … 1.1.",
         "from matplotlib.patches import Arc\nfig, ax = plt.subplots()\nt = np.radians(360 * acc)\nax.plot([0, 1], [0, 0])\n"
         "ax.plot([0, np.cos(t)], [0, np.sin(t)])\nax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=360 * acc))\n"
         "ax.set_aspect('equal')\nax.set_xlim(-1.1, 1.1)\nax.set_ylim(-1.1, 1.1)",
         {"post": PRE + "sorted(fc.r(l.get_xydata(), 3) for l in fc.ax().get_lines()), fc.patches(fc.ax()), "
                  "[_pp(p, 'theta') for p in fc.ax().patches]"}),
    ]
    return catalog(rng, items, setup, f"`names` and `counts` give the class balance of a dataset ({k} classes) and `acc` "
                                      f"a model's accuracy. Create `fig, ax`; import `Wedge` / `Arc` from "
                                      f"`matplotlib.patches`.")


@template(M, "medium")
def colored_lines(rng):
    code, desc, n = curves(rng)
    cmap = rng.choice(["viridis", "plasma", "coolwarm", "magma"])
    s = seeded(rng)
    k = rng.randint(4, 8)
    setup = (code + f"\nlr = np.round(np.geomspace(1e-2, 1e-4, {n}), 6)   # learning rate per epoch\n"
             f"runs = np.round(np.random.default_rng({s}).normal(0, 0.05, ({k}, {n})).cumsum(1) + train_loss, 4)")
    seg = "pts = np.column_stack([epochs, train_loss])\nsegs = np.stack([pts[:-1], pts[1:]], axis=1)\n"
    head = "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\n"
    lcp = "c = [k for k in fc.ax().collections if type(k).__name__ == 'LineCollection'][0]\n"
    items = [
        ("LineCollection", f"Plot `train_loss` against `epochs` as a line **colored by the learning rate**: segments "
                           f"between consecutive points, a `LineCollection` with `array=lr[:-1]` (one value per "
                           f"segment) and `cmap={cmap!r}`; then `ax.autoscale_view()`.",
         f"{head}{seg}lc = LineCollection(segs, array=lr[:-1], cmap={cmap!r})\nax.add_collection(lc)\nax.autoscale_view()",
         {"post": lcp + "[fc.r(x, 4) for x in c.get_segments()], fc.r(c.get_array(), 6), c.get_cmap().name, fc.limits()"}),
        (".colorbar", f"Same colored line (`array=lr[:-1]`, `cmap={cmap!r}`), plus a colorbar for it labelled 'learning "
                      f"rate' (`fig.colorbar(lc, ax=ax, label=...)`) and `ax.autoscale_view()`.",
         f"{head}{seg}lc = LineCollection(segs, array=lr[:-1], cmap={cmap!r})\nax.add_collection(lc)\nax.autoscale_view()\n"
         f"fig.colorbar(lc, ax=ax, label='learning rate')",
         {"post": lcp + "fc.r(c.get_array(), 6), c.get_cmap().name, fc.colorbars()"}),
        ("LogNorm", f"The learning rate spans two decades, so color the line on a **log** scale: same `LineCollection` "
                    f"(`array=lr[:-1]`, `cmap={cmap!r}`) with `norm=LogNorm(1e-4, 1e-2)` from `matplotlib.colors`, and "
                    f"`ax.autoscale_view()`.",
         f"from matplotlib.colors import LogNorm\n{head}{seg}lc = LineCollection(segs, array=lr[:-1], cmap={cmap!r}, "
         f"norm=LogNorm(1e-4, 1e-2))\nax.add_collection(lc)\nax.autoscale_view()",
         {"post": lcp + "fc.r(c.get_array(), 6), fc.norm(c)", "require": ["LogNorm", "LineCollection"]}),
        ("LineCollection", "Make the line thicker where the loss drops fastest: segments of the (`epochs`, `train_loss`) "
                           "curve as a `LineCollection` with `linewidths = 1 + 20 * np.abs(np.diff(train_loss))`, color "
                           "'k', then `ax.autoscale_view()`.",
         f"{head}{seg}ax.add_collection(LineCollection(segs, linewidths=1 + 20 * np.abs(np.diff(train_loss)), colors='k'))\n"
         f"ax.autoscale_view()",
         {"post": lcp + "[fc.r(x, 4) for x in c.get_segments()], fc.r(c.get_linewidths(), 3), fc.hexcolor(c.get_colors())"}),
        ("LineCollection", f"Draw all {k} runs (rows of `runs`, against `epochs`) as **one** `LineCollection` of "
                           f"polylines (each element a whole (n, 2) curve: `np.column_stack([epochs, r])`), colored by "
                           f"each run's final loss (`array=runs[:, -1]`, `cmap={cmap!r}`), then `ax.autoscale_view()`.",
         f"{head}lc = LineCollection([np.column_stack([epochs, r]) for r in runs], array=runs[:, -1], cmap={cmap!r})\n"
         f"ax.add_collection(lc)\nax.autoscale_view()",
         {"post": lcp + "[fc.r(x, 4) for x in c.get_segments()], fc.r(c.get_array(), 4), c.get_cmap().name"}),
        ("LineCollection", f"Draw all {k} runs as one `LineCollection` of polylines in 'gray' with alpha 0.3 and line "
                           f"width 0.8 (fast even for thousands of runs), plus their **mean** curve on top with `ax.plot` "
                           f"in 'C0' (line width 2), then `ax.autoscale_view()`.",
         f"{head}ax.add_collection(LineCollection([np.column_stack([epochs, r]) for r in runs], colors='gray', alpha=0.3, "
         f"linewidths=0.8))\nax.plot(epochs, runs.mean(axis=0), color='C0', linewidth=2)\nax.autoscale_view()",
         {"post": lcp + "[fc.r(x, 4) for x in c.get_segments()], fc.hexcolor(c.get_colors()), fc.alpha(c.get_colors()), "
                  "fc.r(c.get_linewidths(), 3), fc.lines(fc.ax(), 'color', 'lw', nd=4)"}),
    ]
    return catalog(rng, items, setup, f"{desc}; `lr` is the learning rate per epoch and `runs` holds {k} noisy re-runs "
                                      f"of the training curve. Create `fig, ax`.")


# ================================================================= hard

@template(M, "hard")
def patch_helper_functions(rng):
    code, boxes, (H, W) = _detection(rng)
    c = rng.choice(["lime", "red", "cyan"])
    a, b = sorted(rng.sample(range(2, 30), 2))
    s = seeded(rng)
    layers = rng.choice([[2, 3, 1], [3, 2], [2, 4, 2]])
    setup = code + f"\nP = np.round(np.random.default_rng({s}).multivariate_normal([1, 2], [[2, 1.2], [1.2, 1]], 80), 3)"
    items = [
        ("Rectangle", f"Write `draw_boxes(ax, boxes, color='lime')` that outlines each (x, y, w, h, label, score) box with "
                      f"an unfilled `Rectangle` (line width 2) in `color`, writes `f\"{{label}} {{score:.2f}}\"` at (x, "
                      f"y - 2), and returns the list of rectangles. (Checked by calling it on fresh Axes with "
                      f"color {c!r}.)",
         "from matplotlib.patches import Rectangle\ndef draw_boxes(ax, boxes, color='lime'):\n    out = []\n"
         "    for x, y, w, h, lbl, score in boxes:\n        r = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)\n"
         "        ax.add_patch(r)\n        ax.text(x, y - 2, f'{lbl} {score:.2f}')\n        out.append(r)\n    return out",
         {"post": PRE + f"_f, _a = plt.subplots()\n_a.imshow(img)\n_r = draw_boxes(_a, boxes, color={c!r})\n"
                        f"_boxes(_a, 'ec', 'lw', 'fill'), sorted((t['text'], t['xy']) for t in fc.text_items(_a)), "
                        f"len(_r), all(p in _a.patches for p in _r)"}),
        ("blended_transform_factory", f"Write `shade(start, end, ax=None, **kw)` that shades x = start … end over the full "
                                      f"height of `ax` (default: `plt.gca()`) with a blended-transform `Rectangle`, passes "
                                      f"`**kw` on (color, alpha …), and returns the patch.",
         "import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\n"
         "def shade(start, end, ax=None, **kw):\n    ax = ax or plt.gca()\n"
         "    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         "    return ax.add_patch(Rectangle((start, 0), end - start, 1, transform=trans, **kw))",
         {"post": PRE + f"_f, (_a, _b) = plt.subplots(1, 2)\n_a.plot([0, 40], [0, 5])\n_p = shade({a}, {b}, ax=_a, color='red', "
                        f"alpha=0.3)\nplt.sca(_b)\n_b.plot([0, 40], [-1, 1])\nshade({b}, {b + 5})\n"
                        f"_ext(_p, 'blend', _a), _pp(_p, 'fc', 'alpha'), [_ext(q, 'blend', _b) for q in _b.patches]",
          "require": ["blended_transform_factory", "gca"]}),
        ("Ellipse", "Write `confidence_ellipse(P, ax, n_std=2.0, **kw)` that adds and returns the n_std-σ ellipse of the "
                    "point cloud P (center = mean, axes along the covariance eigenvectors, full widths 2·n_std·√λ), "
                    "passing `**kw` to `Ellipse`.",
         "from matplotlib.patches import Ellipse\ndef confidence_ellipse(P, ax, n_std=2.0, **kw):\n    mu = P.mean(axis=0)\n"
         "    vals, vecs = np.linalg.eigh(np.cov(P.T))\n    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))\n"
         "    e = Ellipse(mu, 2 * n_std * np.sqrt(vals[1]), 2 * n_std * np.sqrt(vals[0]), angle=angle, **kw)\n"
         "    ax.add_patch(e)\n    return e",
         {"post": "_f, _a = plt.subplots()\n_e1 = confidence_ellipse(P, _a, fill=False, edgecolor='red')\n"
                  "_e2 = confidence_ellipse(P[:40], _a, n_std=1)\n"
                  "def _cov(p):\n    A = p.get_patch_transform().get_matrix()[:2, :2]\n    return fc.r(A @ A.T, 3)\n"
                  "[(fc.r(e.get_center(), 3), _cov(e)) for e in (_e1, _e2)], fc.hexcolor(_e1.get_edgecolor()), len(_a.patches)"}),
        ("LineCollection", "Write `colored_line(x, y, c, ax=None, cmap='viridis')` that draws the curve (x, y) as a "
                           "`LineCollection` of segments colored by `c[:-1]`, adds it to `ax` (default `plt.gca()`), "
                           "rescales the view to it (`ax.autoscale_view()`), and returns the collection.",
         "from matplotlib.collections import LineCollection\ndef colored_line(x, y, c, ax=None, cmap='viridis'):\n"
         "    ax = ax or plt.gca()\n    pts = np.column_stack([x, y])\n"
         "    lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), array=np.asarray(c)[:-1], cmap=cmap)\n"
         "    ax.add_collection(lc)\n    ax.autoscale_view()\n    return lc",
         {"post": "_f, _a = plt.subplots()\n_x = np.linspace(0, 3, 7)\n_l = colored_line(_x, _x ** 2, _x, ax=_a, cmap='plasma')\n"
                  "[fc.r(s, 4) for s in _l.get_segments()], fc.r(_l.get_array(), 4), _l.get_cmap().name, fc.limits(_a), "
                  "_l in _a.collections"}),
        ("Circle", "Write `draw_network(layers, ax=None, r=0.15)` that draws an MLP diagram: layer i's n nodes are "
                   "circles at (i, j - (n - 1) / 2), j = 0 … n - 1, connected to all nodes of the next layer by one "
                   "`LineCollection`; set the aspect equal, hide the axes, `autoscale_view()`, and return `ax`.",
         "from matplotlib.patches import Circle\nfrom matplotlib.collections import LineCollection\n"
         "def draw_network(layers, ax=None, r=0.15):\n    ax = ax or plt.gca()\n"
         "    pos = [[(i, j - (n - 1) / 2) for j in range(n)] for i, n in enumerate(layers)]\n"
         "    for layer in pos:\n        for p in layer:\n            ax.add_patch(Circle(p, r, zorder=3))\n"
         "    ax.add_collection(LineCollection([[p, q] for a, b in zip(pos, pos[1:]) for p in a for q in b], colors='gray'))\n"
         "    ax.set_aspect('equal')\n    ax.axis('off')\n    ax.autoscale_view()\n    return ax",
         {"post": PRE + f"_f, _a = plt.subplots()\n_r = draw_network({layers}, ax=_a, r=0.2)\n"
                        f"_circles(_a), _segs(_a), _a.get_aspect(), _a.axison, _r is _a",
          "require": ["Circle", "LineCollection"]}),
        ("Bbox", "Write `box_iou(a, b)` for two (x, y, w, h) boxes using Matplotlib's bounding boxes: "
                 "`Bbox.from_bounds(*a)` and `Bbox.intersection(A, B)` (from `matplotlib.transforms`; it returns None "
                 "when they don't overlap). Return intersection area / union area.",
         "from matplotlib.transforms import Bbox\ndef box_iou(a, b):\n    A, B = Bbox.from_bounds(*a), Bbox.from_bounds(*b)\n"
         "    I = Bbox.intersection(A, B)\n    inter = 0 if I is None else I.width * I.height\n"
         "    return inter / (A.width * A.height + B.width * B.height - inter)",
         {"post": "[round(box_iou(p[:4], q[:4]), 6) for p in boxes for q in boxes] + [box_iou((0, 0, 2, 2), (1, 1, 2, 2)), "
                  "box_iou((0, 0, 1, 1), (5, 5, 1, 1))]", "require": ["Bbox", "intersection"]}),
    ]
    return catalog(rng, items, setup, f"`img` is an {H} × {W} image with detections `boxes` (x, y, w, h, label, score) "
                                      f"and `P` a 2-D point cloud. Reusable plotting helpers take the Axes to draw on; "
                                      f"they're checked by calling them.")


@template(M, "hard")
def debug_patches(rng):
    x, y, w, h = rng.randint(1, 4), rng.randint(1, 4), rng.randint(2, 5), rng.randint(2, 5)
    r = rng.choice([1, 2, 3])
    t1, t2 = rng.choice([(0, 90), (30, 120), (45, 180)])
    c = _color(rng)
    n = rng.randint(4, 8)
    s = seeded(rng)
    setup = f"g = np.random.default_rng({s})\nxs = np.linspace(0, 10, {n})\nys = np.round(g.random({n}), 3)"
    items = [
        ("add_patch", f"Nothing shows up:\n```python\nfrom matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\n"
                      f"rect = Rectangle(({x}, {y}), {w}, {h}, color={c!r})\nax.set_xlim(0, 10); ax.set_ylim(0, 10)\n```\n"
                      f"A patch must be added to an Axes. Fix it.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nrect = Rectangle(({x}, {y}), {w}, {h}, color={c!r})\n"
         f"ax.add_patch(rect)\nax.set_xlim(0, 10)\nax.set_ylim(0, 10)", {"post": _post("fc.patches(fc.ax())", "fc.limits()")}),
        ("Rectangle", f"This crashes with *Can not put single artist in more than one figure*:\n```python\n"
                      f"fig, (a1, a2) = plt.subplots(1, 2)\nmark = Circle(({x}, {y}), {r})\na1.add_patch(mark)\n"
                      f"a2.add_patch(mark)\n```\nAn artist lives in one Axes. Give each Axes its own circle.",
         f"from matplotlib.patches import Circle\nfig, (a1, a2) = plt.subplots(1, 2)\na1.add_patch(Circle(({x}, {y}), {r}))\n"
         f"a2.add_patch(Circle(({x}, {y}), {r}))", {"post": "[fc.patches(a) for a in fc.axes()]", "require": ["Circle"]}),
        ("Rectangle", f"The box should span from corner ({x}, {y}) to corner ({x + w}, {y + h}), but it comes out far too "
                      f"big:\n```python\nfig, ax = plt.subplots()\nax.add_patch(Rectangle(({x}, {y}), {x + w}, {y + h}, "
                      f"fill=False))\nax.set_xlim(0, 12); ax.set_ylim(0, 12)\n```\n`Rectangle` wants width and height, "
                      f"not the far corner. Fix it.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.add_patch(Rectangle(({x}, {y}), {w}, {h}, "
         f"fill=False))\nax.set_xlim(0, 12)\nax.set_ylim(0, 12)", {"post": SHAPES}),
        (".set_aspect", f"This circle looks like an ellipse, because the Axes stretches x and y differently:\n```python\n"
                        f"fig, ax = plt.subplots(figsize=(6, 3))\nax.add_patch(Circle((0, 0), {r}))\nax.set_xlim(-4, 4); "
                        f"ax.set_ylim(-4, 4)\n```\nKeep the figure size and limits but make it round.",
         f"from matplotlib.patches import Circle\nfig, ax = plt.subplots(figsize=(6, 3))\nax.add_patch(Circle((0, 0), {r}))\n"
         f"ax.set_xlim(-4, 4)\nax.set_ylim(-4, 4)\nax.set_aspect('equal')",
         {"post": "fc.patches(fc.ax()), fc.ax().get_aspect(), fc.size()", "require": ["set_aspect"]}),
        ("transAxes", "The title-like note should sit just above the plot, centered, but it lands somewhere near the "
                      "origin because (0.5, 1.02) is read as **data** coordinates:\n```python\nfig, ax = plt.subplots()\n"
                      "ax.plot(xs * 100, ys * 50)\nax.text(0.5, 1.02, 'note', ha='center')\n```\nFix it.",
         "fig, ax = plt.subplots()\nax.plot(xs * 100, ys * 50)\nax.text(0.5, 1.02, 'note', ha='center', transform=ax.transAxes)",
         {"post": _post("fc.lines()", "fc.texts()", "[_at(t) for t in fc.ax().texts]"), "require": ["transAxes"]}),
        (".autoscale_view", "The segments are drawn outside the view: the limits stay at 0 … 1.\n```python\n"
                            "fig, ax = plt.subplots()\nsegs = [[(x, 0), (x, y)] for x, y in zip(xs, ys * 10)]\n"
                            "ax.add_collection(LineCollection(segs))\n```\nCollections don't trigger autoscaling: fix it "
                            "so the view fits the data.",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\n"
         "segs = [[(x, 0), (x, y)] for x, y in zip(xs, ys * 10)]\nax.add_collection(LineCollection(segs))\nax.autoscale_view()",
         {"post": PRE + "_segs(), fc.limits()", "require": ["LineCollection"]}),
        ("LineCollection", "This crashes: a `LineCollection` wants a list of **segments** (each a list of points), but it "
                           "got one array of points:\n```python\nfig, ax = plt.subplots()\n"
                           "ax.add_collection(LineCollection(np.column_stack([xs, ys])))\n```\nPass the curve as segments "
                           "between consecutive points (n - 1 segments), then `ax.autoscale_view()`.",
         "from matplotlib.collections import LineCollection\nfig, ax = plt.subplots()\npts = np.column_stack([xs, ys])\n"
         "ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1)))\nax.autoscale_view()",
         {"post": PRE + "_segs()"}),
        ("Wedge", f"The slice should cover {t1}° … {t2}° but is a sliver: `Wedge` takes **degrees**.\n```python\n"
                  f"fig, ax = plt.subplots()\nax.add_patch(Wedge((0, 0), 1, np.radians({t1}), np.radians({t2})))\n"
                  f"ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)\n```\nFix it.",
         f"from matplotlib.patches import Wedge\nfig, ax = plt.subplots()\nax.add_patch(Wedge((0, 0), 1, {t1}, {t2}))\n"
         f"ax.set_xlim(-1, 1)\nax.set_ylim(-1, 1)", {"post": SHAPES}),
        ("Polygon", "This crashes: `Polygon` wants an (N, 2) array of vertices, but got a flat list.\n```python\n"
                    "flat = [0, 0, 4, 0, 2, 3]\nfig, ax = plt.subplots()\nax.add_patch(Polygon(flat))\n```\nKeep `flat` "
                    "and fix the call (`np.reshape(flat, (-1, 2))`).",
         "from matplotlib.patches import Polygon\nflat = [0, 0, 4, 0, 2, 3]\nfig, ax = plt.subplots()\n"
         "ax.add_patch(Polygon(np.reshape(flat, (-1, 2))))", {"post": SHAPES, "require": ["Polygon", "reshape"]}),
        ("match_original", "All circles come out the same default blue although each was given a color:\n```python\n"
                           "fig, ax = plt.subplots()\ncircles = [Circle((x, y), 0.2, facecolor=f'C{i}') for i, (x, y) in "
                           "enumerate(zip(xs, ys))]\nax.add_collection(PatchCollection(circles))\n```\nKeep each patch's own "
                           "color.",
         "from matplotlib.patches import Circle\nfrom matplotlib.collections import PatchCollection\nfig, ax = plt.subplots()\n"
         "circles = [Circle((x, y), 0.2, facecolor=f'C{i}') for i, (x, y) in enumerate(zip(xs, ys))]\n"
         "ax.add_collection(PatchCollection(circles, match_original=True))",
         {"post": "[[fc.hexcolor(c) for c in k.get_facecolor()] for k in fc.ax().collections]", "require": ["match_original"]}),
    ]
    return catalog(rng, items, setup, f"Debug the plotting code (imports from `matplotlib.patches` / "
                                      f"`matplotlib.collections` are yours to add). `xs`, `ys` hold {n} points.")


@template(M, "hard")
def patch_figure_specs(rng):
    code, gt, pred = _gt_pred(rng)
    cg, cp = rng.choice([("lime", "red"), ("green", "magenta"), ("cyan", "orange")])
    cut = rng.randrange(len(gt))
    x, y, w, h = gt[cut]
    items = [
        ("plt.subplots", f"A 2 × 2 gallery (`plt.subplots(2, 2, figsize=(6, 6))`), each panel with limits 0 … 1, equal "
                         f"aspect and one shape: top-left a `Rectangle((0.2, 0.2), 0.6, 0.4)` titled 'Rectangle', "
                         f"top-right a `Circle((0.5, 0.5), 0.3)` titled 'Circle', bottom-left an `Ellipse((0.5, 0.5), 0.8, "
                         f"0.4, angle=30)` titled 'Ellipse', bottom-right a `Wedge((0.5, 0.5), 0.4, 30, 300)` titled "
                         f"'Wedge'.",
         "from matplotlib.patches import Rectangle, Circle, Ellipse, Wedge\nfig, axes = plt.subplots(2, 2, figsize=(6, 6))\n"
         "shapes = [Rectangle((0.2, 0.2), 0.6, 0.4), Circle((0.5, 0.5), 0.3), Ellipse((0.5, 0.5), 0.8, 0.4, angle=30), "
         "Wedge((0.5, 0.5), 0.4, 30, 300)]\nfor ax, shp in zip(axes.flat, shapes):\n    ax.add_patch(shp)\n"
         "    ax.set_title(type(shp).__name__)\n    ax.set_xlim(0, 1)\n    ax.set_ylim(0, 1)\n    ax.set_aspect('equal')",
         {"post": "[(fc.patches(a), a.get_title(), fc.limits(a), a.get_aspect()) for a in fc.axes()], fc.grid_shape()"}),
        ("imshow", f"Zoom panel: left, the image (`cmap='gray'`) with ground-truth box {cut} outlined in {cg!r} (unfilled); "
                   f"right, the crop `img[y:y + h, x:x + w]` of that box (`cmap='gray'`) titled 'crop'. Both axes hidden.",
         f"from matplotlib.patches import Rectangle\nfig, (a1, a2) = plt.subplots(1, 2)\nx, y, w, h = gt[{cut}]\n"
         f"a1.imshow(img, cmap='gray')\na1.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"a2.imshow(img[y:y + h, x:x + w], cmap='gray')\na2.set_title('crop')\na1.axis('off')\na2.axis('off')",
         {"post": PRE + "[(fc.images(a), _boxes(a, 'ec'), a.get_title(), a.axison) for a in fc.axes()]"}),
        ("ConnectionPatch", f"Same zoom panel (left: image with box {cut} outlined in {cg!r}; right: the crop), and connect "
                            f"the box's top-right corner (x + w, y) on the left to the crop's top-left pixel corner (-0.5, "
                            f"-0.5) on the right with a `ConnectionPatch(xyA, xyB, coordsA=a1.transData, "
                            f"coordsB=a2.transData)` added to the **figure** (`fig.add_artist`).",
         f"from matplotlib.patches import Rectangle, ConnectionPatch\nfig, (a1, a2) = plt.subplots(1, 2)\nx, y, w, h = gt[{cut}]\n"
         f"a1.imshow(img, cmap='gray')\na1.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}))\n"
         f"a2.imshow(img[y:y + h, x:x + w], cmap='gray')\n"
         f"fig.add_artist(ConnectionPatch((x + w, y), (-0.5, -0.5), coordsA=a1.transData, coordsB=a2.transData))",
         {"post": PRE + "f = fc.fig()\na1, a2 = fc.axes()\n_boxes(a1, 'ec'), fc.images(a2), "
                  "[(fc.r(c.xy1, 3), fc.r(c.xy2, 3), c.coords1 == a1.transData, c.coords2 == a2.transData) for c in f.artists]",
          "require": ["ConnectionPatch", "add_artist"]}),
        ("transFigure", "Frame the whole figure: make `fig, (a1, a2) = plt.subplots(1, 2)` with `a1.plot([0, 1])` and "
                        "`a2.plot([1, 0])`, then add one unfilled `Rectangle((0.02, 0.02), 0.96, 0.96)` in **figure** "
                        "coordinates (`transform=fig.transFigure`) to the figure with `fig.add_artist`, edge color 'gray', "
                        "and a footnote `fig.text(0.02, 0.0, 'two models')`.",
         "from matplotlib.patches import Rectangle\nfig, (a1, a2) = plt.subplots(1, 2)\na1.plot([0, 1])\na2.plot([1, 0])\n"
         "fig.add_artist(Rectangle((0.02, 0.02), 0.96, 0.96, transform=fig.transFigure, fill=False, edgecolor='gray'))\n"
         "fig.text(0.02, 0.0, 'two models')",
         {"post": "f = fc.fig()\n[(type(a).__name__, fc.r(f.transFigure.inverted().transform(a.get_window_extent()), 3), "
                  "fc.hexcolor(a.get_edgecolor())) for a in f.artists], [t.get_text() for t in f.texts], "
                  "[fc.lines(a) for a in fc.axes()]", "require": ["transFigure", "add_artist"]}),
        ("Rectangle", f"Ground truth vs prediction on **one** image with a legend outside: draw `gt` in {cg!r} and `pred` "
                      f"in {cp!r} (unfilled), give the first rectangle of each kind a label ('ground truth' / "
                      f"'prediction') so `ax.legend(loc='upper left', bbox_to_anchor=(1, 1))` lists exactly those two, and "
                      f"hide the axes.",
         f"from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.imshow(img, cmap='gray')\n"
         f"for i, (x, y, w, h) in enumerate(gt):\n    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cg!r}, "
         f"label='ground truth' if i == 0 else None))\nfor i, (x, y, w, h, s) in enumerate(pred):\n"
         f"    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor={cp!r}, label='prediction' if i == 0 else None))\n"
         f"ax.legend(loc='upper left', bbox_to_anchor=(1, 1))\nax.axis('off')",
         {"post": PRE + "_boxes(None, 'ec'), fc.legend(), fc.ax().axison"}),
        ("FancyBboxPatch", "A model card in axes coordinates: on a hidden Axes (`ax.axis('off')`), draw a rounded box "
                           "`FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='round,pad=0.02', transform=ax.transAxes, "
                           "fc='lightyellow', ec='k')` and write one line per prediction inside it: `f'box {i}: {s:.2f}'` "
                           "at (0.15, 0.8 - 0.1·i) in axes coordinates.",
         "from matplotlib.patches import FancyBboxPatch\nfig, ax = plt.subplots()\nax.axis('off')\n"
         "ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='round,pad=0.02', transform=ax.transAxes, "
         "fc='lightyellow', ec='k'))\nfor i, (x, y, w, h, s) in enumerate(pred):\n"
         "    ax.text(0.15, 0.8 - 0.1 * i, f'box {i}: {s:.2f}', transform=ax.transAxes)",
         {"post": _post("[_ext(p, 'axes') for p in fc.ax().patches]", "[_pp(p, 'box', 'fc') for p in fc.ax().patches]",
                        "sorted((t.get_text(), _at(t)) for t in fc.ax().texts)", "fc.ax().axison")}),
    ]
    return catalog(rng, items, code, f"`img` is a grayscale image, `gt` ground-truth boxes (x, y, w, h) and `pred` "
                                     f"predictions (x, y, w, h, score). Build each figure to the spec; import what you "
                                     f"need from `matplotlib.patches`.")


@template(M, "hard")
def transform_tricks(rng):
    code, desc, n = curves(rng)
    dx, dy = rng.choice([(5, 5), (10, 0), (0, 8), (6, -6)])
    ang = rng.choice([20, 30, 45, 60])
    cx, cy = rng.randint(2, 5), rng.randint(2, 5)
    rin = rng.choice([0.1, 0.15, 0.2])
    click = (rng.randint(150, 450), rng.randint(100, 350))
    items = [
        ("ScaledTranslation", f"Label the best epoch (lowest `val_loss`) with the text 'best' **offset by a fixed {dx} "
                              f"points right and {dy} points up** from the data point, whatever the zoom: "
                              f"`trans = ax.transData + mtransforms.ScaledTranslation({dx} / 72, {dy} / 72, "
                              f"fig.dpi_scale_trans)`, then `ax.text(x, y, 'best', transform=trans)`. Plot `val_loss` too.",
         f"import matplotlib.transforms as mtransforms\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)\n"
         f"b = val_loss.argmin()\ntrans = ax.transData + mtransforms.ScaledTranslation({dx} / 72, {dy} / 72, fig.dpi_scale_trans)\n"
         f"ax.text(epochs[b], val_loss[b], 'best', transform=trans)",
         {"post": "a = fc.ax()\nt = a.texts[0]\nb = val_loss.argmin()\n"
                  "fc.r(t.get_transform().transform(t.get_position()) - a.transData.transform((epochs[b], val_loss[b])), 3), "
                  "t.get_text(), fc.lines(a)", "require": ["ScaledTranslation", "dpi_scale_trans"]}),
        ("Affine2D", f"Draw a `Rectangle(({cx}, {cy}), 3, 1)` rotated {ang}° about its **center** ({cx + 1.5}, {cy + 0.5}). "
                     f"Either compose `mtransforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData` as its "
                     f"transform, or use `angle=` with `rotation_point='center'`. Limits 0 … 8 on both axes, aspect equal.",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\n"
         f"rot = mtransforms.Affine2D().rotate_deg_around({cx + 1.5}, {cy + 0.5}, {ang}) + ax.transData\n"
         f"ax.add_patch(Rectangle(({cx}, {cy}), 3, 1, transform=rot))\nax.set_xlim(0, 8)\nax.set_ylim(0, 8)\nax.set_aspect('equal')",
         {"post": "a = fc.ax()\np = a.patches[0]\nv = a.transData.inverted().transform(p.get_transform().transform(p.get_path().vertices[:4]))\n"
                  "sorted(fc.r(v, 3)), a.get_aspect()", "require": ["Rectangle"]}),
        ("dpi_scale_trans", f"Mark the last epoch's `val_loss` with a circle that is **{rin} inch** in radius on screen no "
                            f"matter the data scale: `Circle((0, 0), {rin}, transform=fig.dpi_scale_trans + "
                            f"mtransforms.ScaledTranslation(x, y, ax.transData), fill=False)` (the circle is drawn in "
                            f"inches, then moved to the data point). Plot `val_loss` against `epochs` first.",
         f"import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Circle\nfig, ax = plt.subplots()\n"
         f"ax.plot(epochs, val_loss)\nax.add_patch(Circle((0, 0), {rin}, transform=fig.dpi_scale_trans + "
         f"mtransforms.ScaledTranslation(epochs[-1], val_loss[-1], ax.transData), fill=False))",
         {"post": "a = fc.ax()\na.get_xlim()\np = a.patches[0]\nb = p.get_window_extent()\n"
                  "fc.r([b.width / a.figure.dpi, b.height / a.figure.dpi], 3), "
                  "fc.r(a.transData.inverted().transform(((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)), 3), p.get_fill()",
          "require": ["dpi_scale_trans", "ScaledTranslation"]}),
        ("transData", f"Hit-testing a click: plot `val_loss` against `epochs` as dots (`'o'`) on `fig, ax = "
                      f"plt.subplots(figsize=(6, 4), dpi=100)` and return the index of the point **nearest on screen** "
                      f"to the pixel {click} — compare distances in display pixels (`ax.transData.transform`), not in "
                      f"data units.",
         f"fig, ax = plt.subplots(figsize=(6, 4), dpi=100)\nax.plot(epochs, val_loss, 'o')\n"
         f"d = ax.transData.transform(np.column_stack([epochs, val_loss])) - np.array({click})\n"
         f"result = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))"),
        ("transLimits", "For linear scales `ax.transLimits` maps data to axes fractions. Plot `train_loss` against "
                        "`epochs` and return the axes fractions of the first and last points, both via `ax.transLimits` "
                        "and via `ax.transData + ax.transAxes.inverted()`, as `[via_limits, via_display]` (2 × 2 lists).",
         "fig, ax = plt.subplots()\nax.plot(epochs, train_loss)\npts = [[epochs[0], train_loss[0]], [epochs[-1], train_loss[-1]]]\n"
         "result = [ax.transLimits.transform(pts).tolist(), (ax.transData + ax.transAxes.inverted()).transform(pts).tolist()]",
         {"tol": 1e-6}),
        ("blended_transform_factory", "Put a small marker on the x-axis at every epoch where `val_loss` went **up**: a "
                                      "`Rectangle((e - 0.3, 0), 0.6, 0.03)` per such epoch e in a data/axes blended "
                                      "transform (so it hugs the bottom whatever the y range), color 'C3'. Plot "
                                      "`val_loss` too.",
         "import matplotlib.transforms as mtransforms\nfrom matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\n"
         "ax.plot(epochs, val_loss)\ntrans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)\n"
         "for e in epochs[1:][np.diff(val_loss) > 0]:\n    ax.add_patch(Rectangle((e - 0.3, 0), 0.6, 0.03, transform=trans, color='C3'))",
         {"post": _post("sorted(_ext(p, 'blend') for p in fc.ax().patches)", "sorted({_pp(p, 'fc')['fc'] for p in fc.ax().patches})",
                        "fc.lines()")}),
    ]
    return catalog(rng, items, code, f"{desc}. Import `matplotlib.transforms as mtransforms` and patches as needed.")


@template(M, "hard")
def geometry_by_hand(rng):
    s = seeded(rng)
    k = rng.randint(12, 25)
    setup = (f"g = np.random.default_rng({s})\npts = np.round(g.normal(size=({k}, 2)) * [2, 1] + [3, 2], 3)\n"
             f"boxes = np.array([[1, 1, 4, 3], [3, 2, 4, 4], [6, 0, 2, 2]])   # x, y, w, h")
    hull = ("def cross(o, a, b):\n    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])\n"
            "P = sorted(map(tuple, pts))\nlower, upper = [], []\nfor p in P:\n"
            "    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:\n        lower.pop()\n    lower.append(p)\n"
            "for p in reversed(P):\n    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:\n        upper.pop()\n"
            "    upper.append(p)\nhull = lower[:-1] + upper[:-1]\n")
    items = [
        ("Polygon", "Draw the **convex hull** of `pts` as an unfilled `Polygon` (compute it yourself, e.g. with Andrew's "
                    "monotone chain) and scatter the points. The hull's vertex set is checked, in any order and starting "
                    "point.",
         f"from matplotlib.patches import Polygon\n{hull}fig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1])\n"
         f"ax.add_patch(Polygon(hull, fill=False))",
         {"post": "p = fc.ax().patches[0]\nsorted(map(tuple, np.unique(np.round(p.get_xy(), 3), axis=0).tolist())), p.get_fill(), "
                  "fc.scatter()"}),
        ("Polygon", "Return the area of the convex hull of `pts` (shoelace formula over the hull's vertices, in order). "
                    "Draw the hull as a `Polygon` too.",
         f"from matplotlib.patches import Polygon\n{hull}fig, ax = plt.subplots()\nax.add_patch(Polygon(hull))\n"
         f"h = np.array(hull)\nresult = 0.5 * abs(np.dot(h[:, 0], np.roll(h[:, 1], -1)) - np.dot(h[:, 1], np.roll(h[:, 0], -1)))",
         {"tol": 1e-6}),
        ("Rectangle", "Draw the tight axis-aligned bounding box of `pts` (min to max on each axis) as an unfilled "
                      "`Rectangle`, and scatter the points.",
         "from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nax.scatter(pts[:, 0], pts[:, 1])\n"
         "lo, hi = pts.min(axis=0), pts.max(axis=0)\nax.add_patch(Rectangle(lo, *(hi - lo), fill=False))",
         {"post": "fc.patches(fc.ax(), 3), [p.get_fill() for p in fc.ax().patches], fc.scatter()"}),
        ("Bbox", "Return the bounds `(x0, y0, width, height)` of the smallest box containing all three `boxes`, using "
                 "`Bbox.union([Bbox.from_bounds(*b) for b in boxes])` from `matplotlib.transforms`.",
         "from matplotlib.transforms import Bbox\nresult = Bbox.union([Bbox.from_bounds(*b) for b in boxes]).bounds",
         {"require": ["Bbox", "union"]}),
        ("Bbox", "For every pair (i, j), i < j, of `boxes`, return whether they overlap (`Bbox.overlaps`), as a list of "
                 "bools in the order (0, 1), (0, 2), (1, 2).",
         "from matplotlib.transforms import Bbox\nB = [Bbox.from_bounds(*b) for b in boxes]\n"
         "result = [bool(B[i].overlaps(B[j])) for i in range(3) for j in range(i + 1, 3)]", {"require": ["Bbox", "overlaps"]}),
        ("Wedge", "Count the points of `pts` inside `Wedge((3, 2), 2, 0, 90)` (a quarter disc). A wedge's path is already "
                  "in data coordinates: `w.get_path().contains_points(pts)`. Check it by hand too: inside means radius "
                  "< 2 and angle in 0 … 90°. Return `[count_with_path, count_by_hand]`.",
         "from matplotlib.patches import Wedge\nw = Wedge((3, 2), 2, 0, 90)\nd = pts - [3, 2]\n"
         "ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))\nhand = (np.hypot(d[:, 0], d[:, 1]) < 2) & (ang >= 0) & (ang <= 90)\n"
         "result = [int(w.get_path().contains_points(pts).sum()), int(hand.sum())]", {"require": ["Wedge", "contains_points"]}),
        ("Rectangle", "Draw the three `boxes` (unfilled, 'C0') and fill their pairwise **intersections** in 'C3' (alpha "
                      "0.5) as extra rectangles, for every overlapping pair (in the order (0, 1), (0, 2), (1, 2)).",
         "from matplotlib.patches import Rectangle\nfig, ax = plt.subplots()\nfor x, y, w, h in boxes:\n"
         "    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='C0'))\nfor i in range(3):\n    for j in range(i + 1, 3):\n"
         "        a, b = boxes[i], boxes[j]\n        x0, y0 = max(a[0], b[0]), max(a[1], b[1])\n"
         "        x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])\n        if x1 > x0 and y1 > y0:\n"
         "            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color='C3', alpha=0.5))\nax.autoscale_view()",
         {"post": PRE + "_boxes(None, 'fill', 'fc')"}),
    ]
    return catalog(rng, items, setup, f"`pts` holds {k} 2-D points and `boxes` three (x, y, w, h) boxes.")


# ================================================================= challenge

@template(M, "challenge")
def lasso_selection(rng):
    m, n, k, s = rng.randint(6, 9), rng.randint(10, 16), rng.randint(2, 3), seeded(rng)
    report = rng.choice(["mask", "per_class", "count"])

    def gen(seed, m_, n_):
        return (f"g = np.random.default_rng({seed})\nang = np.sort(g.uniform(0, 2 * np.pi, {m_}))\n"
                f"k = {k}\nrad = g.uniform(3, 9, {m_})\npoly = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])\n"
                f"pts = g.uniform(-10, 10, ({n_}, 2))\nlabels = g.integers(0, {k}, {n_})\nreport = {report!r}")
    sol = ("x, y = pts[:, 0], pts[:, 1]\ninside = np.zeros(len(pts), dtype=bool)\nfor i in range(len(poly)):\n"
           "    (x1, y1), (x2, y2) = poly[i - 1], poly[i]\n    crosses = (y1 > y) != (y2 > y)\n    if not crosses.any():\n"
           "        continue\n    t = (y[crosses] - y1) / (y2 - y1)\n"
           "    inside[crosses] ^= x[crosses] < x1 + t * (x2 - x1)   # the ray to the right crosses this edge\n"
           + {"mask": "result = inside",
              "per_class": "result = np.bincount(labels[inside], minlength=k)",
              "count": "result = int(inside.sum())"}[report])
    want = {"mask": "a boolean array, True for the selected points",
            "per_class": f"how many selected points each class has, as an array of length {k} (class 0 first)",
            "count": "the number of selected points"}[report]
    return challenge(
        "Lasso selection",
        "A user drew a lasso around part of a scatter plot of an embedding; Matplotlib's `LassoSelector` hands you its "
        "vertices `poly` (m × 2, in order; the last vertex connects back to the first). A point is selected when it "
        "lies **inside** the polygon: count how many polygon edges a ray from the point to the right crosses (an edge "
        "counts when one end is above the point and the other is not) — odd means inside. No point lies on an edge. "
        f"`pts` (n × 2) are the scatter points and `labels` their classes 0 … k - 1. Return {want}. "
        f"(`report` = {report!r}.)",
        gen(s, m, n), sol, [gen(s + 1, 300, 30000), gen(s + 2, 40, 200000)],
        "m ≤ 300 vertices, n ≤ 2·10^5 points (m·n ≤ 10^7)",
        speed="O(m·n) with NumPy: loop over the edges, vectorize over the points")


@template(M, "challenge")
def detection_nms(rng):
    n, s = rng.randint(8, 12), seeded(rng)
    report = rng.choice(["nms", "max_iou", "tp"])
    thr = rng.choice([0.3, 0.5]) if report != "max_iou" else 0.5

    def gen(seed, n_, m_, side):
        return (f"g = np.random.default_rng({seed})\nW = {side}\n"
                f"gt = np.column_stack([g.integers(0, W, ({m_}, 2)), g.integers(4, 30, ({m_}, 2))])   # x, y, w, h\n"
                f"src = gt[g.integers(0, {m_}, {n_})]\n"
                f"boxes = np.column_stack([src[:, :2] + g.integers(-6, 7, ({n_}, 2)), np.maximum(1, src[:, 2:] + "
                f"g.integers(-6, 7, ({n_}, 2)))])\nscores = np.round(g.random({n_}), 3)\nthr = {thr}\nreport = {report!r}")
    iou = ("def iou(b, B):\n    w = np.maximum(0, np.minimum(b[0] + b[2], B[:, 0] + B[:, 2]) - np.maximum(b[0], B[:, 0]))\n"
           "    h = np.maximum(0, np.minimum(b[1] + b[3], B[:, 1] + B[:, 3]) - np.maximum(b[1], B[:, 1]))\n"
           "    inter = w * h\n    return inter / (b[2] * b[3] + B[:, 2] * B[:, 3] - inter)\n\n"
           "order = np.lexsort((np.arange(len(scores)), -scores))   # by score, high first; ties: lower index first\n")
    tail = {"nms": ("alive = np.ones(len(boxes), dtype=bool)\nkeep = []\nfor i in order:\n    if not alive[i]:\n        continue\n"
                    "    keep.append(int(i))\n    alive &= ~(iou(boxes[i], boxes) > thr)\nresult = keep"),
            "max_iou": ("result = np.array([iou(b, gt).max() for b in boxes])"),
            "tp": ("free = np.ones(len(gt), dtype=bool)\ntp = 0\nfor i in order:\n    v = np.where(free, iou(boxes[i], gt), -1.0)\n"
                   "    j = int(v.argmax())\n    if v[j] >= thr:\n        free[j] = False\n        tp += 1\nresult = tp")}[report]
    want = {"nms": "the indices of the boxes greedy **non-maximum suppression** keeps, in the order it keeps them: take "
                   "the best-scoring remaining box, keep it, and discard every remaining box whose IoU with it is "
                   "**greater** than `thr`; repeat",
            "max_iou": "for every predicted box, its best IoU with any ground-truth box (a float array)",
            "tp": "the number of **true positives**: go through the predictions from the highest score down; each one "
                  "takes the still-unmatched ground-truth box it overlaps most (ties: lowest index) if that IoU is at "
                  "least `thr`, else it is a false positive"}[report]
    return challenge(
        "Cleaning up detections",
        "A detector drew boxes over an image (green for ground truth, red for predictions, like the boxes you drew with "
        "`Rectangle`). Boxes are (x, y, w, h) with integer coordinates; IoU = intersection area / union area. "
        "`boxes` (n × 4) are the predictions with `scores`, `gt` (m × 4) the ground truth. Process predictions by "
        f"score, highest first, breaking ties by lower index. Return {want}. (`report` = {report!r}.)",
        gen(s, n, 4, 60), iou + tail, [gen(s + 1, 3000, 400, 2000), gen(s + 2, 2500, 50, 300)],
        "n ≤ 3000 predictions, m ≤ 400 ground-truth boxes",
        speed="O(n·m) or O(n·kept) with NumPy: vectorize each box against all others",
        answer_desc=" (a list of ints)" if report == "nms" else "")


@template(M, "challenge")
def box_union_area(rng):
    n, s = rng.randint(5, 9), seeded(rng)
    report = rng.choice(["union", "overlap", "exactly_one"])

    def gen(seed, n_, side, big):
        return (f"g = np.random.default_rng({seed})\nxy = g.integers(0, {side}, ({n_}, 2))\n"
                f"rects = np.column_stack([xy, g.integers(1, {big}, ({n_}, 2))])   # x, y, w, h\nreport = {report!r}")
    sol = ("x0, y0 = rects[:, 0], rects[:, 1]\nx1, y1 = x0 + rects[:, 2], y0 + rects[:, 3]\n"
           "xs, ys = np.unique(np.concatenate([x0, x1])), np.unique(np.concatenate([y0, y1]))\n"
           "i0, i1 = np.searchsorted(xs, x0), np.searchsorted(xs, x1)\nj0, j1 = np.searchsorted(ys, y0), np.searchsorted(ys, y1)\n"
           "D = np.zeros((len(xs), len(ys)), dtype=np.int64)   # 2-D difference array over the compressed grid\n"
           "np.add.at(D, (i0, j0), 1)\nnp.add.at(D, (i1, j0), -1)\nnp.add.at(D, (i0, j1), -1)\nnp.add.at(D, (i1, j1), 1)\n"
           "cover = D.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]   # how many rectangles cover each cell\n"
           "cell = np.outer(np.diff(xs), np.diff(ys))\n"
           + {"union": "result = int(cell[cover >= 1].sum())",
              "overlap": "result = int(cell[cover >= 2].sum())",
              "exactly_one": "result = int(cell[cover == 1].sum())"}[report])
    want = {"union": "the total area covered by **at least one** rectangle",
            "overlap": "the total area covered by **two or more** rectangles (where shaded regions stack up)",
            "exactly_one": "the total area covered by **exactly one** rectangle"}[report]
    return challenge(
        "Area under the boxes",
        "You shaded `n` axis-aligned rectangles `rects` (x, y, w, h, integers) on an image with `ax.add_patch`. "
        f"Overlapping shading stacks up. Return {want}. (`report` = {report!r}.)",
        gen(s, n, 30, 15), sol, [gen(s + 1, 600, 10 ** 6, 2 * 10 ** 5), gen(s + 2, 500, 1000, 300)],
        "n ≤ 600 rectangles, coordinates ≤ 10^6",
        speed="O(n²) with coordinate compression and a 2-D prefix sum (NumPy)", answer_desc=" (an int)")


@template(M, "challenge")
def label_declutter(rng):
    n, s = rng.randint(10, 14), seeded(rng)
    report = rng.choice(["pixels", "placed", "hidden"])
    W, H, dpi = rng.choice([(6, 4, 100), (5, 5, 80), (8, 3, 50)])
    l, b, w, h = rng.choice([(0.125, 0.11, 0.775, 0.77), (0.1, 0.1, 0.8, 0.8), (0.15, 0.2, 0.7, 0.7)])
    xscale = rng.choice(["linear", "log"])
    flip = rng.random() < 0.5
    lw, lh = rng.choice([(60, 14), (80, 16), (50, 12)])

    def gen(seed, n_):
        ylim = "(10, -10)" if flip else "(-10, 10)"
        xlim = "(1, 1000)" if xscale == "log" else "(0, 100)"
        xs = "10 ** g.uniform(0, 3, n)" if xscale == "log" else "g.uniform(0, 100, n)"
        return (f"g = np.random.default_rng({seed})\nn = {n_}\nfigsize, dpi = ({W}, {H}), {dpi}\n"
                f"position = ({l}, {b}, {w}, {h})   # the Axes in figure fractions: left, bottom, width, height\n"
                f"xscale, xlim, ylim = {xscale!r}, {xlim}, {ylim}\npts = np.column_stack([{xs}, g.uniform(-10, 10, n)])\n"
                f"score = np.round(g.random(n), 4)   # label priority\nbox = ({lw}, {lh})   # label width, height in pixels\n"
                f"report = {report!r}")
    sol = ("f = np.log10 if xscale == 'log' else (lambda v: v)\nfw, fh = figsize[0] * dpi, figsize[1] * dpi\n"
           "l, b, w, h = position\nfx = (f(pts[:, 0]) - f(xlim[0])) / (f(xlim[1]) - f(xlim[0]))\n"
           "fy = (pts[:, 1] - ylim[0]) / (ylim[1] - ylim[0])\npix = np.column_stack([(l + w * fx) * fw, (b + h * fy) * fh])\n"
           + ("result = pix" if report == "pixels" else
              "bw, bh = box\ncells = {}   # grid of label-sized cells -> placed labels in it\nplaced = []\n"
              "for i in np.lexsort((np.arange(n), -score)).tolist():\n    x, y = pix[i]\n"
              "    cx, cy = int(x // bw), int(y // bh)\n"
              "    clash = any(abs(pix[j, 0] - x) < bw and abs(pix[j, 1] - y) < bh\n"
              "                for a in (cx - 1, cx, cx + 1) for c in (cy - 1, cy, cy + 1) for j in cells.get((a, c), ()))\n"
              "    if not clash:\n        placed.append(i)\n        cells.setdefault((cx, cy), []).append(i)\n"
              + ("result = placed" if report == "placed" else "result = n - len(placed)")))
    want = {"pixels": "the display coordinates of every point (an n × 2 float array: what `ax.transData.transform(pts)` "
                      "gives)",
            "placed": "the indices of the labels that get drawn, in the order they are placed",
            "hidden": "how many labels are hidden"}[report]
    return challenge(
        "Decluttering point labels",
        "A figure of `figsize` inches at `dpi` holds one Axes placed at `position` (figure fractions), with x scale "
        "`xscale` ('linear', or 'log': x is mapped through log10) and limits `xlim`, `ylim` (a pair may be reversed: an "
        "inverted axis). Display coordinates are pixels from the figure's bottom-left corner. Each scatter point "
        "`pts[i]` has a text label: a `box` = (width, height) pixel rectangle centered on the point's display "
        "position. Labels are placed greedily by `score` (highest first, ties: lower index); a label is hidden if its "
        "rectangle overlaps (with positive area) one already placed. Work out the data → display transform yourself "
        f"and return {want}. (`report` = {report!r}.)",
        gen(s, n), sol, [gen(s + 1, 100000), gen(s + 2, 60000)], "n ≤ 10^5 points",
        speed="O(n) expected: a grid of label-sized cells, so each label only checks its 3 × 3 neighborhood",
        tol=1e-6, answer_desc=" (a list of ints)" if report == "placed" else "")

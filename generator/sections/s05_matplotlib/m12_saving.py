"""5.12 Saving & exporting: savefig's formats (PNG / PDF / SVG / JPEG), dpi and pixel sizes, tight bounding boxes,
transparent and colored backgrounds, in-memory exports (BytesIO, base64, the canvas' RGBA buffer), multi-page PDF
reports (PdfPages), imsave / imread, metadata and reproducible files, and the create -> save -> close loop.

Every grading run starts in a fresh temporary folder, so problems check the files the code wrote: signatures (first
bytes), `plt.imread` shapes, page counts and page sizes read from a PDF's bytes, SVG sizes, the alpha / color of a
corner pixel. Figures stay tiny and dpi low (≤ 50) so exports are fast in the browser."""

from ._common import P, catalog, challenge, curves, lit, seeded, signal, template

M = "5.12"

# post helpers: _bytes / _head (file contents), _shape (imread shape), _pages (PDF page objects), _boxes (PDF page
# sizes in points), _svgsize (the <svg> width / height), _px (one pixel, RGBA floats), _files (sorted listing).
PRE = r'''import os as _os, re as _re

def _bytes(p):
    with open(p, 'rb') as f:
        return f.read()

def _head(p, n=8):
    return _bytes(p)[:n]

def _shape(p):
    return list(plt.imread(p).shape)

def _pages(p):
    return len(_re.findall(rb'/Type\s*/Page\b', _bytes(p)))

def _boxes(p):
    return [[round(float(v), 2) for v in m.split()] for m in _re.findall(rb'/MediaBox\s*\[([^\]]*)\]', _bytes(p))]

def _svgsize(p):
    return [g.decode() for g in _re.search(rb'<svg[^>]*?width="([^"]+)"[^>]*?height="([^"]+)"', _bytes(p)).groups()]

def _px(p, r=0, c=0, nd=3):
    return [round(float(v), nd) for v in plt.imread(p)[r, c]]

def _files(d='.'):
    return sorted(_os.listdir(d))
'''
PAGES_RE = r"rb'/Type\s*/Page\b'"          # how solutions count a PDF's pages (/Pages is the page tree, not a page)
IMP_PDF = "from matplotlib.backends.backend_pdf import PdfPages\n"


def _post(*exprs):
    return PRE + ", ".join(exprs)


def _size(rng):
    return rng.choice([(4, 3), (3, 2), (5, 2.5), (2, 2), (6, 3), (3, 3), (4, 2)])


def _dpi(rng):
    return rng.choice([20, 25, 30, 40, 50])


# ================================================================= lite

@template(M, "lite")
def savefig_options(rng):
    code, desc, n = signal(rng)
    w, h = _size(rng)
    dpi = _dpi(rng)
    name = rng.choice(["plot", "wave", "signal", "fig1", "chart"])
    q = rng.choice([50, 75, 90, 95])
    color = rng.choice(["black", "lightgray", "navy", "#ffeecc", "tab:blue"])
    pad = rng.choice([0, 0.2, 0.5])
    png, pdf, svg, jpg = (f"{name}.{e}" for e in ("png", "pdf", "svg", "jpg"))
    base = f"fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(x, y)\n"
    mk = f"Create `fig, ax = plt.subplots(figsize=({w}, {h}))`, plot `y` against `x` (nothing else), and "
    items = [
        (".savefig", mk + f"save the figure as `{png}` at {dpi} dpi.", base + f"fig.savefig({png!r}, dpi={dpi})",
         {"post": _post(f"_shape({png!r})")}),
        (".savefig", mk + f"save the figure as `{pdf}` (a vector format: no dpi needed).", base + f"fig.savefig({pdf!r})",
         {"post": _post(f"_head({pdf!r}, 5)", f"_pages({pdf!r})")}),
        (".savefig", mk + f"save the figure as `{svg}` (SVG: vector graphics written as XML text, for websites).",
         base + f"fig.savefig({svg!r})", {"post": _post(f"_head({svg!r}, 5)", f"_svgsize({svg!r})")}),
        ("pil_kwargs", mk + f"save the figure as `{jpg}` at {dpi} dpi with JPEG quality {q}. Pillow writes the JPEG, so "
                            f"pass its options as `pil_kwargs={{'quality': {q}}}`.",
         base + f"fig.savefig({jpg!r}, dpi={dpi}, pil_kwargs={{'quality': {q}}})",
         {"post": _post(f"_head({jpg!r}, 3)", f"_shape({jpg!r})")}),
        ("transparent", mk + f"save the figure as `{png}` at {dpi} dpi with a transparent background "
                             f"(`transparent=True`: the background pixels get alpha 0).",
         base + f"fig.savefig({png!r}, dpi={dpi}, transparent=True)",
         {"post": _post(f"_px({png!r})[3]", f"_shape({png!r})")}),
        ("facecolor", mk + f"save the figure as `{png}` at {dpi} dpi on a {color!r} background (`facecolor=`).",
         base + f"fig.savefig({png!r}, dpi={dpi}, facecolor={color!r})", {"post": _post(f"_px({png!r})", f"_shape({png!r})")}),
        ("bbox_inches", mk + f"save the figure as `{png}` at {dpi} dpi cropped to what is drawn plus a small pad "
                             f"(`bbox_inches='tight'`).",
         base + f"fig.savefig({png!r}, dpi={dpi}, bbox_inches='tight')", {"post": _post(f"_shape({png!r})")}),
        ("pad_inches", mk + f"save the figure as `{png}` at {dpi} dpi with `bbox_inches='tight'` and {pad} inches of "
                            f"padding around the drawing (`pad_inches`).",
         base + f"fig.savefig({png!r}, dpi={dpi}, bbox_inches='tight', pad_inches={pad})",
         {"post": _post(f"_shape({png!r})"), "require": ["bbox_inches", "pad_inches"]}),
        ("plt.savefig", f"With pyplot only: `plt.figure(figsize=({w}, {h}))`, plot `y` against `x` with `plt.plot`, and "
                        f"save the current figure as `{png}` at {dpi} dpi with `plt.savefig`.",
         f"plt.figure(figsize=({w}, {h}))\nplt.plot(x, y)\nplt.savefig({png!r}, dpi={dpi})",
         {"post": _post(f"_shape({png!r})")}),
        ("dpi", mk + f"save it twice: `{name}_lo.png` at {dpi // 2} dpi and `{name}_hi.png` at {dpi} dpi (inches × "
                     f"dpi = pixels).",
         base + f"fig.savefig('{name}_lo.png', dpi={dpi // 2})\nfig.savefig('{name}_hi.png', dpi={dpi})",
         {"post": _post(f"_shape('{name}_lo.png')", f"_shape('{name}_hi.png')"), "require": ["savefig", "dpi"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def export_calls(rng):
    code, desc, n = signal(rng)
    w, h = _size(rng)
    dpi = _dpi(rng)
    w2, h2 = rng.choice([(8, 3), (6, 4), (5, 5), (7, 2)])
    k = rng.randint(2, 5)
    ext = rng.choice(["pdf", "svg", "png", "jpg", "eps"])
    base = f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(x, y)\n"
    mk = f"Create `fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})` and plot `y` against `x`. "
    items = [
        ("set_size_inches", mk + f"Resize the figure to {w2} × {h2} inches with `fig.set_size_inches` and return "
                                 f"`fig.get_size_inches()` as a list.",
         base + f"fig.set_size_inches({w2}, {h2})\nresult = fig.get_size_inches().tolist()",
         {"require": ["set_size_inches", "get_size_inches"]}),
        (".dpi", mk + "Return `[fig.dpi, fig.canvas.get_width_height()]`: the canvas measures inches × dpi pixels.",
         base + "result = [fig.dpi, fig.canvas.get_width_height()]", {"require": ["dpi", "get_width_height"]}),
        ("BytesIO", mk + "Save it into memory instead of a file — `buf = io.BytesIO()`, then `fig.savefig(buf, "
                         "format='png')` — and return the first 8 bytes of `buf.getvalue()` (the PNG signature).",
         "import io\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='png')\nresult = buf.getvalue()[:8]",
         {"require": ["BytesIO", "getvalue"]}),
        ("buffer_rgba", mk + "Render it with `fig.canvas.draw()` and return the shape of "
                             "`np.asarray(fig.canvas.buffer_rgba())`.",
         base + "fig.canvas.draw()\nresult = np.asarray(fig.canvas.buffer_rgba()).shape", {"require": ["draw", "buffer_rgba"]}),
        ("imread", mk + "Save it as `out.png` and return the dtype name of `plt.imread('out.png')` (a PNG comes back as "
                        "floats in 0 … 1).",
         base + "fig.savefig('out.png')\nresult = plt.imread('out.png').dtype.name", {"require": ["savefig", "imread"]}),
        ("plt.close", mk + "Save it as `out.png`, close it with `plt.close(fig)`, and return "
                           "`[os.path.exists('out.png'), len(plt.get_fignums())]`.",
         "import os\n" + base + "fig.savefig('out.png')\nplt.close(fig)\nresult = [os.path.exists('out.png'), len(plt.get_fignums())]",
         {"require": ["savefig", "close"]}),
        ("get_supported_filetypes", mk + f"Return how Matplotlib describes the {ext!r} format: "
                                         f"`fig.canvas.get_supported_filetypes()[{ext!r}]`.",
         base + f"result = fig.canvas.get_supported_filetypes()[{ext!r}]"),
        ("PdfPages", f"Write a {k}-page PDF with `PdfPages`: inside `with PdfPages('report.pdf') as pdf:`, for k = 1 … {k} "
                     f"create `f, a = plt.subplots()`, plot `y ** k` against `x`, add the page with `pdf.savefig(f)` and "
                     f"close `f`. Return `pdf.get_pagecount()` (inside the `with`).",
         IMP_PDF + f"with PdfPages('report.pdf') as pdf:\n    for k in range(1, {k + 1}):\n        f, a = plt.subplots()\n"
                   f"        a.plot(x, y ** k)\n        pdf.savefig(f)\n        plt.close(f)\n    result = pdf.get_pagecount()",
         {"require": ["PdfPages", "get_pagecount"]}),
        ("canvas", mk + "Return the width and height in pixels as `fig.canvas.get_width_height()` gives them, then the "
                        "same after `fig.set_size_inches(2, 1)`: `[before, after]`.",
         base + "before = fig.canvas.get_width_height()\nfig.set_size_inches(2, 1)\nresult = [before, fig.canvas.get_width_height()]",
         {"require": ["get_width_height", "set_size_inches"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "lite")
def image_files(rng):
    s = seeded(rng)
    hh, ww = rng.randint(6, 16), rng.randint(6, 16)
    lo, hi = rng.choice([(0, 1), (0, 255), (-1, 1), (10, 20)])
    name = rng.choice(["array", "mask", "weights", "heat"])
    cmap = rng.choice(["viridis", "magma", "Blues", "gray_r"])
    setup = (f"img = np.round(np.random.default_rng({s}).uniform({lo}, {hi}, ({hh}, {ww})), 3)\n"
             f"rgb = np.round(np.random.default_rng({s + 1}).random(({hh}, {ww}, 3)), 3)")
    png = f"{name}.png"
    items = [
        ("imsave", f"Write `img` straight to `{png}`, one pixel per value, with `plt.imsave(..., cmap='gray')`, and return "
                   f"`plt.imread({png!r}).shape`.",
         f"plt.imsave({png!r}, img, cmap='gray')\nresult = plt.imread({png!r}).shape"),
        ("imsave", f"`plt.imsave` stretches the values over the colormap's full range by default. Save `img` as `{png}` "
                   f"with `cmap='gray'` and return `[min, max]` of channel 0 of the image read back (as floats).",
         f"plt.imsave({png!r}, img, cmap='gray')\nr = plt.imread({png!r})[..., 0]\nresult = [float(r.min()), float(r.max())]"),
        ("vmin", f"Save `img` as `{png}` with `cmap='gray', vmin={lo}, vmax={lo + 2 * (hi - lo)}` (fixed limits instead of "
                 f"the data's) and return channel 0 of the image read back, rounded to 2 decimals.",
         f"plt.imsave({png!r}, img, cmap='gray', vmin={lo}, vmax={lo + 2 * (hi - lo)})\n"
         f"result = np.round(plt.imread({png!r})[..., 0], 2)", {"require": ["imsave", "vmin"]}),
        ("origin", "Save `img` twice with `cmap='gray'`: `up.png` normally and `down.png` with `origin='lower'`. Return "
                   "whether the second file read back equals the first one flipped upside down (`np.array_equal(b, a[::-1])`).",
         "plt.imsave('up.png', img, cmap='gray')\nplt.imsave('down.png', img, cmap='gray', origin='lower')\n"
         "a, b = plt.imread('up.png'), plt.imread('down.png')\nresult = np.array_equal(b, a[::-1])",
         {"require": ["imsave", "origin"]}),
        ("imsave", f"Save the RGB array `rgb` (values in 0 … 1: no colormap needed) as `{png}` and return the shape read "
                   f"back: PNGs come back with 4 channels (RGBA).",
         f"plt.imsave({png!r}, rgb)\nresult = plt.imread({png!r}).shape"),
        ("imread", f"Save `img` as the JPEG `{name}.jpg` with `plt.imsave(..., cmap='gray')` and return `[shape, dtype "
                   f"name]` of `plt.imread`: JPEGs have no alpha and come back as 8-bit integers.",
         f"plt.imsave('{name}.jpg', img, cmap='gray')\nr = plt.imread('{name}.jpg')\nresult = [r.shape, r.dtype.name]",
         {"require": ["imsave", "imread"]}),
        ("cmap", f"Save `img` as `{png}` with `cmap={cmap!r}` and return the pixel read back where `img` is largest "
                 f"(`np.unravel_index(img.argmax(), img.shape)`) as a list rounded to 2 decimals: the colormap's top color.",
         f"plt.imsave({png!r}, img, cmap={cmap!r})\ni, j = np.unravel_index(img.argmax(), img.shape)\n"
         f"result = np.round(plt.imread({png!r})[i, j], 2).tolist()", {"require": ["imsave", "cmap"]}),
    ]
    return catalog(rng, items, setup, f"`img` is a {hh} × {ww} array of values in {lo} … {hi} and `rgb` a {hh} × {ww} "
                                      f"× 3 RGB image.")


# ================================================================= easy

@template(M, "easy")
def export_sizes(rng):
    code, desc, n = signal(rng)
    w, h = _size(rng)
    dpi = _dpi(rng)
    d2 = rng.choice([v for v in (20, 25, 30, 40, 50) if v != dpi])
    W, H, D = rng.choice([(200, 100, 50), (150, 150, 50), (240, 120, 40), (300, 200, 50), (160, 90, 20), (120, 80, 40)])
    w2, h2 = rng.choice([(6, 2), (2, 4), (5, 5), (3, 1)])
    base = f"fig, ax = plt.subplots(figsize=({w}, {h}))\nax.plot(x, y)\n"
    mk = f"Create `fig, ax = plt.subplots(figsize=({w}, {h}))` and plot `y` against `x`. "
    items = [
        ("set_size_inches", f"A web page wants a {W} × {H} pixel image. Create `fig, ax`, plot `y` against `x`, give the "
                            f"figure the size in inches (`fig.set_size_inches`) that makes {W} × {H} pixels at {D} dpi, "
                            f"save `web.png` at {D} dpi, and return `plt.imread('web.png').shape`.",
         f"fig, ax = plt.subplots()\nax.plot(x, y)\nfig.set_size_inches({W} / {D}, {H} / {D})\nfig.savefig('web.png', dpi={D})\n"
         f"result = plt.imread('web.png').shape"),
        ("dpi", f"The canvas and the file can have different resolutions. Create `fig, ax = plt.subplots(figsize=({w}, "
                f"{h}), dpi={dpi})`, plot `y`, render it and take `np.asarray(fig.canvas.buffer_rgba()).shape`; then save "
                f"`out.png` at {d2} dpi and take its `plt.imread` shape. Return both shapes.",
         f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(x, y)\nfig.canvas.draw()\n"
         f"a = np.asarray(fig.canvas.buffer_rgba()).shape\nfig.savefig('out.png', dpi={d2})\nresult = [a, plt.imread('out.png').shape]",
         {"require": ["buffer_rgba", "savefig"]}),
        ("savefig", mk + f"SVG sizes are in points (72 per inch), whatever the dpi: save the figure as `v.svg` with "
                         f"`dpi={dpi}` and return the `width` and `height` attributes of its `<svg>` tag as a list of "
                         f"strings (`re.search(r'width=\"([^\"]+)\" height=\"([^\"]+)\"', text).groups()`).",
         "import re\n" + base + f"fig.savefig('v.svg', dpi={dpi})\ntext = open('v.svg').read()\n"
         "result = list(re.search(r'width=\"([^\"]+)\" height=\"([^\"]+)\"', text).groups())"),
        ("savefig", mk + "A PDF page is sized in points too: save the figure as `v.pdf` and return the numbers of its "
                         "`/MediaBox [ x0 y0 x1 y1 ]` entry as floats, read from the file's bytes with "
                         "`re.search(rb'/MediaBox\\s*\\[([^\\]]*)\\]', data)`.",
         "import re\n" + base + "fig.savefig('v.pdf')\ndata = open('v.pdf', 'rb').read()\n"
         "result = [float(v) for v in re.search(rb'/MediaBox\\s*\\[([^\\]]*)\\]', data).group(1).split()]"),
        ("savefig", f"Without `dpi=`, `savefig` uses `rcParams['savefig.dpi']`, which is `'figure'`: the figure's own "
                    f"dpi. Create `fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})`, plot `y`, save `out.png` "
                    f"without a dpi, and return its `plt.imread` shape.",
         f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(x, y)\nfig.savefig('out.png')\n"
         f"result = plt.imread('out.png').shape"),
        ("rcParams", mk + f"Set the default export resolution with `plt.rcParams['savefig.dpi'] = {dpi}`, save `out.png` "
                          f"without passing a dpi, and return its `plt.imread` shape.",
         base + f"plt.rcParams['savefig.dpi'] = {dpi}\nfig.savefig('out.png')\nresult = plt.imread('out.png').shape"),
        ("set_dpi", f"Create `fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})` and plot `y`. Double the resolution "
                    f"without changing the size in inches (`fig.set_dpi(2 * fig.dpi)`) and return "
                    f"`[fig.get_size_inches().tolist(), fig.canvas.get_width_height()]`.",
         f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(x, y)\nfig.set_dpi(2 * fig.dpi)\n"
         f"result = [fig.get_size_inches().tolist(), fig.canvas.get_width_height()]", {"require": ["set_dpi", "get_size_inches"]}),
        ("get_size_inches", mk + f"After plotting, resize the figure to {w2} × {h2} inches, save `out.png` at {dpi} dpi, "
                                 f"and return `[fig.get_size_inches().tolist(), plt.imread('out.png').shape]`.",
         base + f"fig.set_size_inches({w2}, {h2})\nfig.savefig('out.png', dpi={dpi})\n"
         f"result = [fig.get_size_inches().tolist(), plt.imread('out.png').shape]",
         {"require": ["set_size_inches", "get_size_inches"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def memory_exports(rng):
    code, desc, n = curves(rng)
    w, h = rng.choice([(3, 2), (2, 2), (4, 2), (2.5, 1.5)])
    dpi = rng.choice([20, 25, 30, 40])
    base = (f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(epochs, train_loss)\n"
            f"ax.plot(epochs, val_loss)\n")
    mk = (f"Create `fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})` and plot `train_loss` and `val_loss` "
          f"against `epochs`. ")
    items = [
        ("BytesIO", mk + "Save it as PNG into `buf = io.BytesIO()`, rewind the buffer (`buf.seek(0)`), read it back "
                         "with `plt.imread(buf)` and return the shape.",
         "import io\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='png')\nbuf.seek(0)\nresult = plt.imread(buf).shape",
         {"require": ["BytesIO", "seek", "imread"]}),
        ("b64encode", mk + "Embed it in a web page: save it as PNG into a `BytesIO`, build the data URI `uri = "
                           "'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()`, then decode it back "
                           "(`base64.b64decode(uri[22:])`) and return `[uri[:22], the decoded image's plt.imread shape]`.",
         "import io, base64\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='png')\n"
         "uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()\n"
         "back = io.BytesIO(base64.b64decode(uri[22:]))\nresult = [uri[:22], plt.imread(back).shape]",
         {"require": ["b64encode", "getvalue"]}),
        ("getvalue", mk + "Save it as SVG into a `BytesIO` (`format='svg'`) and return `[the first 5 bytes, b'<svg' in "
                          "buf.getvalue()]`.",
         "import io\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='svg')\ndata = buf.getvalue()\n"
         "result = [data[:5], b'<svg' in data]", {"require": ["BytesIO", "getvalue"]}),
        ("frombuffer", mk + "Get raw pixels without any file format: `fig.savefig(buf, format='rgba')` writes the RGBA "
                            "bytes row by row. Rebuild the image with `np.frombuffer(buf.getvalue(), np.uint8).reshape(h, "
                            "w, 4)` where `w, h = fig.canvas.get_width_height()`, and return `[its shape, the pixel at "
                            "[0, 0] as a list]`.",
         "import io\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='rgba')\nw, h = fig.canvas.get_width_height()\n"
         "img = np.frombuffer(buf.getvalue(), np.uint8).reshape(h, w, 4)\nresult = [img.shape, img[0, 0].tolist()]",
         {"require": ["frombuffer", "get_width_height"]}),
        ("buffer_rgba", mk + "Render the canvas (`fig.canvas.draw()`) and keep the RGB channels as a uint8 array: "
                             "`np.asarray(fig.canvas.buffer_rgba())[..., :3]`. Return `[its shape, its dtype name]`.",
         base + "fig.canvas.draw()\nrgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]\nresult = [rgb.shape, rgb.dtype.name]",
         {"require": ["draw", "buffer_rgba"]}),
        ("Image", mk + "Save it as PNG into a `BytesIO` and open that with Pillow (`from PIL import Image`; "
                       "`im = Image.open(buf)`). Return `[im.size, im.mode]` (Pillow's size is (width, height)).",
         "import io\nfrom PIL import Image\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='png')\n"
         "im = Image.open(buf)\nresult = [im.size, im.mode]", {"require": ["BytesIO", "Image"]}),
        ("BytesIO", mk + "Save it as PDF into a `BytesIO` (`format='pdf'`) and return `[the first 5 bytes, the number of "
                         f"pages]`, counting page objects with `len(re.findall({PAGES_RE}, data))`.",
         "import io, re\n" + base + "buf = io.BytesIO()\nfig.savefig(buf, format='pdf')\ndata = buf.getvalue()\n"
         f"result = [data[:5], len(re.findall({PAGES_RE}, data))]", {"require": ["BytesIO", "getvalue"]}),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def export_backgrounds(rng):
    code, desc, n = signal(rng)
    dpi = rng.choice([20, 25, 30])
    color = rng.choice(["black", "lightgray", "navy", "#ffeecc", "tab:orange", "#202830"])
    a = rng.choice([0.25, 0.5, 0.75])
    base = "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(x, y)\n"
    mk = "Create `fig, ax = plt.subplots(figsize=(3, 2))` and plot `y` against `x`. "
    corner = "result = plt.imread('out.png')[0, 0].round(3).tolist()"
    items = [
        ("transparent", mk + f"Save it twice at {dpi} dpi: `clear.png` with `transparent=True` and `solid.png` without. "
                             f"Return the alpha of the top-left pixel of each, `[clear, solid]` (as floats).",
         base + f"fig.savefig('clear.png', dpi={dpi}, transparent=True)\nfig.savefig('solid.png', dpi={dpi})\n"
         f"result = [float(plt.imread(f)[0, 0, 3]) for f in ('clear.png', 'solid.png')]"),
        ("set_facecolor", mk + f"Color the figure itself with `fig.set_facecolor({color!r})` and save `out.png` at {dpi} "
                               f"dpi with savefig's defaults (its `facecolor='auto'` uses the figure's color). Return the "
                               f"top-left pixel (RGBA) as a list rounded to 3 decimals.",
         base + f"fig.set_facecolor({color!r})\nfig.savefig('out.png', dpi={dpi})\n" + corner),
        ("transparent", mk + f"`transparent=True` also clears the Axes' white background. Save `out.png` at {dpi} dpi "
                             f"that way and return `[the smallest alpha, the median alpha]` over the whole image (as "
                             f"floats).",
         base + f"fig.savefig('out.png', dpi={dpi}, transparent=True)\nalpha = plt.imread('out.png')[..., 3]\n"
         f"result = [float(alpha.min()), float(np.median(alpha))]"),
        ("rcParams", mk + f"Make transparency the default with `plt.rcParams['savefig.transparent'] = True`, save "
                          f"`out.png` at {dpi} dpi without passing `transparent`, and return the alpha of the top-left "
                          f"pixel.",
         base + f"plt.rcParams['savefig.transparent'] = True\nfig.savefig('out.png', dpi={dpi})\n"
         f"result = float(plt.imread('out.png')[0, 0, 3])"),
        ("facecolor", mk + f"Give the figure the color {color!r} (`fig.set_facecolor`), but save `out.png` at {dpi} dpi "
                           f"with `facecolor='white'`: the export setting wins and the figure keeps its color. Return "
                           f"`[the saved top-left pixel rounded to 3 decimals, mpl.colors.to_hex(fig.get_facecolor())]`.",
         base + f"fig.set_facecolor({color!r})\nfig.savefig('out.png', dpi={dpi}, facecolor='white')\n"
         f"result = [plt.imread('out.png')[0, 0].round(3).tolist(), mpl.colors.to_hex(fig.get_facecolor())]",
         {"require": ["set_facecolor", "facecolor"]}),
        ("set_alpha", mk + f"Make the figure's background {round(a * 100)}% opaque with `fig.patch.set_alpha({a})`, save "
                           f"`out.png` at {dpi} dpi, and return the top-left pixel's alpha rounded to 2 decimals.",
         base + f"fig.patch.set_alpha({a})\nfig.savefig('out.png', dpi={dpi})\n"
         f"result = round(float(plt.imread('out.png')[0, 0, 3]), 2)", {"require": ["patch", "set_alpha"]}),
        ("rcParams", mk + f"Set the default export background with `plt.rcParams['savefig.facecolor'] = {color!r}`, save "
                          f"`out.png` at {dpi} dpi, and return the top-left pixel (RGBA) rounded to 3 decimals.",
         base + f"plt.rcParams['savefig.facecolor'] = {color!r}\nfig.savefig('out.png', dpi={dpi})\n" + corner),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def export_file_names(rng):
    code, desc, n = curves(rng)
    stem = rng.choice(["loss", "curves", "training", "run1"])
    fmt = rng.choice(["svg", "pdf", "jpg"])
    sub = rng.choice(["figures", "plots", "out"])
    eps = sorted(rng.sample([e for e in (1, 2, 5, 8, 10, 12, 15) if e <= n], 3))
    base = "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\n"
    mk = "Create `fig, ax = plt.subplots(figsize=(3, 2))` and plot `train_loss` against `epochs`. "
    items = [
        ("savefig", mk + f"Save it as just `{stem!r}` (no extension) at 20 dpi: Matplotlib then picks the format from "
                         f"`rcParams['savefig.format']` and adds the extension. Return `sorted(os.listdir())`.",
         "import os\n" + base + f"fig.savefig({stem!r}, dpi=20)\nresult = sorted(os.listdir())"),
        ("rcParams", mk + f"Set `plt.rcParams['savefig.format'] = {fmt!r}`, save the figure as `{stem!r}` (no extension, "
                          f"dpi 20), and return `sorted(os.listdir())`.",
         "import os\n" + base + f"plt.rcParams['savefig.format'] = {fmt!r}\nfig.savefig({stem!r}, dpi=20)\n"
         f"result = sorted(os.listdir())"),
        ("format", mk + f"An explicit `format=` beats the file name: save the figure as `{stem}.png` with `format='pdf'` "
                        f"and return the file's first 4 bytes.",
         base + f"fig.savefig('{stem}.png', format='pdf')\nresult = open('{stem}.png', 'rb').read(4)", {"require": ["format"]}),
        ("makedirs", mk + f"`savefig` doesn't create folders: make `{sub}` with `os.makedirs(..., exist_ok=True)`, save the "
                          f"figure as `{sub}/{stem}.png` at 20 dpi, and return `os.listdir({sub!r})`.",
         "import os\n" + base + f"os.makedirs({sub!r}, exist_ok=True)\nfig.savefig('{sub}/{stem}.png', dpi=20)\n"
         f"result = os.listdir({sub!r})"),
        ("Path", mk + f"`savefig` also takes `pathlib.Path` objects: make `out = Path({sub!r})`, create it with "
                      f"`out.mkdir()`, save the figure to `out / '{stem}.svg'`, and return `[p.name for p in out.iterdir()]`.",
         "from pathlib import Path\n" + base + f"out = Path({sub!r})\nout.mkdir()\nfig.savefig(out / '{stem}.svg')\n"
         f"result = [p.name for p in out.iterdir()]", {"require": ["Path", "mkdir"]}),
        ("savefig", f"Zero-padded numbers keep files in order: for each epoch e in {eps}, plot `train_loss[:e]` on a new "
                    f"`plt.subplots(figsize=(2, 1.5))` figure, save it as `f'epoch_{{e:03d}}.png'` at 20 dpi and close it. "
                    f"Return `sorted(os.listdir())`.",
         f"import os\nfor e in {eps}:\n    fig, ax = plt.subplots(figsize=(2, 1.5))\n    ax.plot(train_loss[:e])\n"
         f"    fig.savefig(f'epoch_{{e:03d}}.png', dpi=20)\n    plt.close(fig)\nresult = sorted(os.listdir())",
         {"require": ["savefig", "close"]}),
        ("savefig", mk + f"Save it in three formats with one loop over `['png', 'pdf', 'svg']` (`f'{stem}.{{ext}}'`, dpi "
                         f"20) and return `sorted(os.listdir())`.",
         "import os\n" + base + f"for ext in ['png', 'pdf', 'svg']:\n    fig.savefig(f'{stem}.{{ext}}', dpi=20)\n"
         f"result = sorted(os.listdir())"),
    ]
    return catalog(rng, items, code, f"{desc}.")


@template(M, "easy")
def pdf_pages_basics(rng):
    code, desc, n = curves(rng)
    k = rng.randint(2, 5)
    title = rng.choice(["Training report", "Run 7", "Experiment A", "Weekly metrics"])
    author = rng.choice(["Ada", "Alan", "Grace", "Team ML"])
    note = rng.choice(["best epoch marked", "lr = 3e-4", "seed 0"])
    (w1, h1), (w2, h2) = rng.sample([(4, 3), (6, 2), (3, 3), (5, 4), (8, 4)], 2)
    loop = (f"    for k in range(1, {k + 1}):\n        f, a = plt.subplots()\n        a.plot(epochs, train_loss ** k)\n"
            f"        pdf.savefig(f)\n        plt.close(f)\n")
    pages = f"len(re.findall({PAGES_RE}, data))"
    task = (f"one page per k = 1 … {k} (a new `f, a = plt.subplots()` plotting `train_loss ** k` against `epochs`, "
            f"closed after saving)")
    items = [
        ("PdfPages", f"Write `report.pdf` with `with PdfPages('report.pdf') as pdf:` and `pdf.savefig(f)`: {task}. Then "
                     f"count the pages in the finished file: read its bytes and return `{pages}`.",
         IMP_PDF + "import re\nwith PdfPages('report.pdf') as pdf:\n" + loop + "data = open('report.pdf', 'rb').read()\n"
         f"result = {pages}", {"require": ["PdfPages", "savefig"]}),
        ("plt.figure", f"Without an argument, `pdf.savefig()` saves pyplot's **current** figure. With pyplot only, write "
                       f"`report.pdf` with {k} pages: page k (1 … {k}) is a `plt.figure()` with `plt.plot(epochs, "
                       f"train_loss ** k)` and `plt.title(f'page {{k}}')`, added with `pdf.savefig()` and closed with "
                       f"`plt.close()`. Return `pdf.get_pagecount()` (inside the `with`).",
         IMP_PDF + f"with PdfPages('report.pdf') as pdf:\n    for k in range(1, {k + 1}):\n        plt.figure()\n"
                   f"        plt.plot(epochs, train_loss ** k)\n        plt.title(f'page {{k}}')\n        pdf.savefig()\n"
                   f"        plt.close()\n    result = pdf.get_pagecount()", {"require": ["PdfPages", "get_pagecount"]}),
        ("infodict", f"Fill in the document info: in `with PdfPages('report.pdf') as pdf:`, set `pdf.infodict()['Title'] = "
                     f"{title!r}` and `['Author'] = {author!r}`, and save one page (`train_loss` against `epochs`). Return "
                     f"whether the file's bytes contain `b'/Title ({title})'` and `b'/Author ({author})'` (a list of 2 "
                     f"bools).",
         IMP_PDF + f"with PdfPages('report.pdf') as pdf:\n    info = pdf.infodict()\n    info['Title'] = {title!r}\n"
                   f"    info['Author'] = {author!r}\n    f, a = plt.subplots()\n    a.plot(epochs, train_loss)\n"
                   f"    pdf.savefig(f)\n    plt.close(f)\ndata = open('report.pdf', 'rb').read()\n"
                   f"result = [b'/Title ({title})' in data, b'/Author ({author})' in data]", {"require": ["infodict"]}),
        ("metadata", f"Pass the document info when opening the file: `PdfPages('report.pdf', metadata={{'Title': "
                     f"{title!r}}})`, then write {task}. Return `[b'/Title ({title})' in data, {pages}]` for the "
                     f"finished file's bytes `data`.",
         IMP_PDF + f"import re\nwith PdfPages('report.pdf', metadata={{'Title': {title!r}}}) as pdf:\n" + loop
         + f"data = open('report.pdf', 'rb').read()\nresult = [b'/Title ({title})' in data, {pages}]",
         {"require": ["PdfPages", "metadata"]}),
        ("PdfPages", f"Pages can differ in size: write `mixed.pdf` whose page 1 is a {w1} × {h1} inch figure of "
                     f"`train_loss` and page 2 a {w2} × {h2} inch figure of `val_loss`. Return the page sizes in points "
                     f"(1 inch = 72 points) read from the bytes: every `/MediaBox [ 0 0 w h ]` as `[w, h]` floats, via "
                     f"`re.findall(rb'/MediaBox\\s*\\[([^\\]]*)\\]', data)`.",
         IMP_PDF + f"import re\nwith PdfPages('mixed.pdf') as pdf:\n    for size, c in ((({w1}, {h1}), train_loss), "
                   f"(({w2}, {h2}), val_loss)):\n        f, a = plt.subplots(figsize=size)\n        a.plot(epochs, c)\n"
                   f"        pdf.savefig(f)\n        plt.close(f)\ndata = open('mixed.pdf', 'rb').read()\n"
                   f"result = [[float(v) for v in m.split()[2:]] for m in re.findall(rb'/MediaBox\\s*\\[([^\\]]*)\\]', data)]"),
        (".savefig", f"A `PdfPages` object can also be the target of `fig.savefig(pdf, format='pdf')`. Write `report.pdf` "
                     f"that way with {task}, and return `pdf.get_pagecount()` (inside the `with`).",
         IMP_PDF + f"with PdfPages('report.pdf') as pdf:\n    for k in range(1, {k + 1}):\n        f, a = plt.subplots()\n"
                   f"        a.plot(epochs, train_loss ** k)\n        f.savefig(pdf, format='pdf')\n        plt.close(f)\n"
                   f"    result = pdf.get_pagecount()", {"require": ["PdfPages", "format"]}),
        ("attach_note", f"Write `report.pdf` with {task}, and attach the note {note!r} to the first page "
                        f"(`pdf.attach_note(...)` before that page's `pdf.savefig`). Return `[{pages}, b{note!r} in data]` "
                        f"for the finished file's bytes `data` (the note is stored as text).",
         IMP_PDF + f"import re\nwith PdfPages('report.pdf') as pdf:\n    for k in range(1, {k + 1}):\n"
                   f"        f, a = plt.subplots()\n        a.plot(epochs, train_loss ** k)\n        if k == 1:\n"
                   f"            pdf.attach_note({note!r})\n        pdf.savefig(f)\n        plt.close(f)\n"
                   f"data = open('report.pdf', 'rb').read()\nresult = [{pages}, b{note!r} in data]"),
    ]
    return catalog(rng, items, code, f"{desc}. Import `PdfPages` from `matplotlib.backends.backend_pdf`.")


# ================================================================= medium

@template(M, "medium")
def publication_exports(rng):
    code, desc, n = curves(rng, acc=True)
    dpi = rng.choice([20, 25, 30, 40])
    W = rng.choice([200, 300, 400])
    title = rng.choice(["Loss", "Validation", "Run 3", "Training"])
    cw, ch = rng.choice([(2, 1), (1.5, 1.5), (3, 1), (1, 2)])
    losses = "ax.plot(epochs, train_loss, label='train')\nax.plot(epochs, val_loss, label='val')\n"
    items = [
        ("bbox_inches", "A figure for one journal column: `fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="
                        "'constrained')`, `train_loss` and `val_loss` against `epochs` labelled 'train' / 'val', x label "
                        "'epoch', y label 'loss', a legend; save it as `fig1.pdf` with `bbox_inches='tight', "
                        "pad_inches=0.02`. Keep the figure open. (Checked: lines, labels, legend and the PDF's page size.)",
         "fig, ax = plt.subplots(figsize=(3.5, 2.5), layout='constrained')\n" + losses +
         "ax.set(xlabel='epoch', ylabel='loss')\nax.legend()\nfig.savefig('fig1.pdf', bbox_inches='tight', pad_inches=0.02)",
         {"post": _post("fc.lines(fc.ax(), 'label')", "fc.labels()", "fc.legend()", "_boxes('fig1.pdf')"),
          "require": ["bbox_inches", "pad_inches"]}),
        ("transparent", f"A slide figure: 16:9 (`figsize=(8, 4.5)`), `val_acc` against `epochs`, title 'Accuracy', "
                        f"saved as `slide.png` at {dpi} dpi with a transparent background. Keep the figure open. "
                        f"(Checked: the line, the title, the image's shape and its corner alpha.)",
         f"fig, ax = plt.subplots(figsize=(8, 4.5))\nax.plot(epochs, val_acc)\nax.set_title('Accuracy')\n"
         f"fig.savefig('slide.png', dpi={dpi}, transparent=True)",
         {"post": _post("fc.lines(fc.ax())", "fc.labels()['title']", "_shape('slide.png')", "_px('slide.png')[3]")}),
        ("set_size_inches", f"A web figure exactly {W} pixels wide at 50 dpi with a 2:1 aspect ratio: plot both losses "
                            f"(labels 'train', 'val', with a legend), set the size with `fig.set_size_inches`, and save "
                            f"`web.png` at 50 dpi. (Checked: the legend and the PNG's shape.)",
         f"fig, ax = plt.subplots()\n{losses}ax.legend()\nfig.set_size_inches({W} / 50, {W} / 100)\nfig.savefig('web.png', dpi=50)",
         {"post": _post("fc.legend()", "_shape('web.png')")}),
        ("facecolor", f"For light and dark slides: plot `val_loss` against `epochs` on `plt.subplots(figsize=(4, 3))` and "
                      f"save it twice at {dpi} dpi — `light.png` with `facecolor='white'`, `dark.png` with "
                      f"`facecolor='black'`. (Checked: both corner pixels and shapes.)",
         f"fig, ax = plt.subplots(figsize=(4, 3))\nax.plot(epochs, val_loss)\nfig.savefig('light.png', dpi={dpi}, facecolor='white')\n"
         f"fig.savefig('dark.png', dpi={dpi}, facecolor='black')",
         {"post": _post("_px('light.png')", "_px('dark.png')", "_shape('dark.png')")}),
        ("rc_context", f"Many journals reject Type 3 fonts. Inside `with plt.rc_context({{'pdf.fonttype': 42}}):` make the "
                       f"loss figure (`train_loss` against `epochs`, title {title!r}) and save it as `paper.pdf`, so the "
                       f"fonts are embedded as TrueType. (Checked: the title and `b'/FontFile2'` in the PDF's bytes.)",
         f"with plt.rc_context({{'pdf.fonttype': 42}}):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, train_loss)\n"
         f"    ax.set_title({title!r})\n    fig.savefig('paper.pdf')",
         {"post": _post("fc.labels()['title']", "b'/FontFile2' in _bytes('paper.pdf')", "_pages('paper.pdf')")}),
        ("rcParams", f"Editable text for Inkscape / Illustrator: set `plt.rcParams['svg.fonttype'] = 'none'` (text stays "
                     f"`<text>` instead of outlines), plot `val_loss` against `epochs` titled {title!r} with no ticks "
                     f"(`ax.set_xticks([])`, `ax.set_yticks([])`), and save `fig.svg`. (Checked: the texts in the SVG.)",
         f"plt.rcParams['svg.fonttype'] = 'none'\nfig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nax.set_title({title!r})\n"
         f"ax.set_xticks([])\nax.set_yticks([])\nfig.savefig('fig.svg')",
         {"post": _post("_re.findall(rb'<text[^>]*>([^<]*)</text>', _bytes('fig.svg'))"), "require": ["rcParams", "set_title"]}),
        ("get_window_extent", f"Export one panel of a dashboard: `fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 2.5))` "
                              f"with both losses on `a1` and `val_acc` on `a2`; save **only** the right Axes' area as "
                              f"`right.png` at {dpi} dpi with `bbox_inches=a2.get_window_extent().transformed("
                              f"fig.dpi_scale_trans.inverted())` (the Axes' box in inches).",
         f"fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 2.5))\na1.plot(epochs, train_loss)\na1.plot(epochs, val_loss)\n"
         f"a2.plot(epochs, val_acc)\nbox = a2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())\n"
         f"fig.savefig('right.png', dpi={dpi}, bbox_inches=box)",
         {"post": _post("[len(a.get_lines()) for a in fc.axes()]", "_shape('right.png')"),
          "require": ["get_window_extent", "dpi_scale_trans", "bbox_inches"]}),
        ("Bbox", f"Crop by hand: plot both losses on `plt.subplots(figsize=(4, 3))` and save only the bottom-left {cw} × "
                 f"{ch} inches of the figure as `crop.png` at {dpi} dpi, with `bbox_inches=Bbox([[0, 0], [{cw}, {ch}]])` "
                 f"(`from matplotlib.transforms import Bbox`).",
         f"from matplotlib.transforms import Bbox\nfig, ax = plt.subplots(figsize=(4, 3))\nax.plot(epochs, train_loss)\n"
         f"ax.plot(epochs, val_loss)\nfig.savefig('crop.png', dpi={dpi}, bbox_inches=Bbox([[0, 0], [{cw}, {ch}]]))",
         {"post": _post("_shape('crop.png')"), "require": ["Bbox", "bbox_inches"]}),
    ]
    return catalog(rng, items, code, f"{desc}. Export each figure to the spec.")


@template(M, "medium")
def figure_to_arrays(rng):
    code, desc, n = curves(rng)
    w, h = rng.choice([(2, 1.5), (3, 2), (2.5, 2.5), (4, 2)])
    dpi = rng.choice([20, 25, 30, 40])
    snaps = sorted(rng.sample(range(3, n + 1), rng.randint(3, 5)))
    base = (f"fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
            f"fig.canvas.draw()\nrgba = np.asarray(fig.canvas.buffer_rgba())\n")
    mk = (f"Plot `train_loss` and `val_loss` against `epochs` on `fig, ax = plt.subplots(figsize=({w}, {h}), "
          f"dpi={dpi})`, render it with `fig.canvas.draw()` and take `rgba = np.asarray(fig.canvas.buffer_rgba())`. ")
    items = [
        ("buffer_rgba", mk + "Image loggers want RGB uint8 of shape (height, width, 3): drop the alpha channel and return "
                             "`[shape, dtype name, the top-left pixel as a list]`.",
         base + "rgb = rgba[..., :3]\nresult = [rgb.shape, rgb.dtype.name, rgb[0, 0].tolist()]", {"require": ["buffer_rgba", "draw"]}),
        ("transpose", mk + "TensorBoard's `add_image` takes channels first: return the shape of the RGB part transposed "
                           "to (3, height, width).",
         base + "result = np.transpose(rgba[..., :3], (2, 0, 1)).shape", {"require": ["buffer_rgba", "transpose"]}),
        ("astype", mk + "Return the RGB part as float32 in 0 … 1 (divide by 255): `[shape, dtype name, max value]`.",
         base + "img = rgba[..., :3].astype(np.float32) / 255\nresult = [img.shape, img.dtype.name, float(img.max())]",
         {"require": ["buffer_rgba", "astype"]}),
        ("stack", f"Log a snapshot of the run at epochs {snaps}: for each e, plot `train_loss[:e]` against `epochs[:e]` on "
                  f"a new `plt.subplots(figsize=({w}, {h}), dpi={dpi})`, render it, keep a **copy** of the RGB part of "
                  f"the buffer and close the figure. Return `[np.stack(frames).shape, len(plt.get_fignums())]`.",
         f"frames = []\nfor e in {snaps}:\n    fig, ax = plt.subplots(figsize=({w}, {h}), dpi={dpi})\n"
         f"    ax.plot(epochs[:e], train_loss[:e])\n    fig.canvas.draw()\n    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())\n"
         f"    plt.close(fig)\nresult = [np.stack(frames).shape, len(plt.get_fignums())]", {"require": ["buffer_rgba", "close"]}),
        ("array_equal", mk + f"Also save the figure as `out.png` at {dpi} dpi. Return whether the PNG read back and scaled "
                             f"to 0 … 255 (`np.round(plt.imread('out.png') * 255).astype(np.uint8)`) equals `rgba`: the "
                             f"canvas and the file hold the same pixels.",
         base + f"fig.savefig('out.png', dpi={dpi})\npng = np.round(plt.imread('out.png') * 255).astype(np.uint8)\n"
         f"result = np.array_equal(png, rgba)", {"require": ["array_equal", "imread"]}),
        ("frombuffer", mk + "The buffer is also a flat run of bytes: return the shape of `np.frombuffer(fig.canvas."
                            "buffer_rgba(), np.uint8).reshape(h, w, 4)` with `w, h = fig.canvas.get_width_height()`.",
         base + "w, h = fig.canvas.get_width_height()\nresult = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4).shape",
         {"require": ["frombuffer", "get_width_height"]}),
        ("fromarray", mk + "Hand the RGB part to Pillow (`from PIL import Image`; `im = Image.fromarray(rgba[..., :3])`) "
                           "and return `[im.size, im.mode]`.",
         "from PIL import Image\n" + base + "im = Image.fromarray(rgba[..., :3])\nresult = [im.size, im.mode]",
         {"require": ["fromarray"]}),
        ("get_width_height", mk + "Return `[fig.canvas.get_width_height(), rgba.shape[:2]]`: (width, height) for the "
                                  "canvas, (rows, columns) for the array.",
         base + "result = [fig.canvas.get_width_height(), rgba.shape[:2]]", {"require": ["get_width_height", "buffer_rgba"]}),
    ]
    return catalog(rng, items, code, f"{desc}. Figures become NumPy images (for TensorBoard, W&B, or a video).")


@template(M, "medium")
def pdf_reports(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    title = rng.choice(["Training report", "Run summary", "Experiment 12"])
    author = rng.choice(["Ada", "Grace", "Team ML"])
    names = rng.sample(["adam", "sgd", "rmsprop", "lion", "adamw", "adagrad"], rng.randint(4, 5))
    bad = sorted(rng.sample(range(len(names)), 2))
    setup = (code + "\nmetrics = {'train loss': train_loss, 'val loss': val_loss, 'val acc': val_acc}\n"
             f"g = np.random.default_rng({s})\nruns = {{name: np.round(np.exp(-epochs / 8) + 0.05 * g.random(len(epochs)), 3) "
             f"for name in {lit(names)}}}\n" + "".join(f"runs[{names[b]!r}][-3:] = np.nan   # diverged\n" for b in bad))
    loop = ("    for name, values in metrics.items():\n        fig, ax = plt.subplots()\n        ax.plot(epochs, values)\n"
            "        ax.set_title(name)\n        pdf.savefig(fig)\n        plt.close(fig)\n")
    items = [
        ("PdfPages", "One page per metric: write `metrics.pdf` with a page for each entry of `metrics` (the values "
                     "against `epochs`, titled with the name), closing every figure. (Checked: pages and open figures.)",
         IMP_PDF + "with PdfPages('metrics.pdf') as pdf:\n" + loop,
         {"post": _post("_pages('metrics.pdf')", "len(plt.get_fignums())")}),
        ("text", f"A report with a cover: page 1 is a figure with no Axes, just `fig.text(0.5, 0.5, {title!r}, ha='center', "
                 f"va='center', fontsize=20)`; then one page per metric as above. Write `report.pdf`. (Checked: the "
                 f"page count and page sizes.)",
         IMP_PDF + f"with PdfPages('report.pdf') as pdf:\n    fig = plt.figure()\n    fig.text(0.5, 0.5, {title!r}, ha='center', "
                   f"va='center', fontsize=20)\n    pdf.savefig(fig)\n    plt.close(fig)\n" + loop,
         {"post": _post("_pages('report.pdf')", "_boxes('report.pdf')"), "require": ["PdfPages", "text"]}),
        ("PdfPages", "An overview page plus detail pages of another shape: page 1 (8 × 4 inches) plots all three metrics "
                     "on one Axes, then one 4 × 6 inch page per metric. Write `report.pdf`. (Checked: the page sizes.)",
         IMP_PDF + "with PdfPages('report.pdf') as pdf:\n    fig, ax = plt.subplots(figsize=(8, 4))\n"
         "    for values in metrics.values():\n        ax.plot(epochs, values)\n    pdf.savefig(fig)\n    plt.close(fig)\n"
         "    for name, values in metrics.items():\n        fig, ax = plt.subplots(figsize=(4, 6))\n        ax.plot(epochs, values)\n"
         "        pdf.savefig(fig)\n        plt.close(fig)", {"post": _post("_boxes('report.pdf')")}),
        ("subplots", "A4 portrait pages (8.27 × 11.69 inches) with two panels each (`plt.subplots(2, 1, figsize=(8.27, "
                     "11.69))`): the three metrics in order, two per page, and on the last page remove the unused panel "
                     "(`ax.remove()`). Write `a4.pdf`. (Checked: page count and sizes.)",
         IMP_PDF + "items = list(metrics.items())\nwith PdfPages('a4.pdf') as pdf:\n    for i in range(0, len(items), 2):\n"
         "        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))\n        for ax, (name, values) in zip(axes, items[i:i + 2]):\n"
         "            ax.plot(epochs, values)\n            ax.set_title(name)\n        if len(items[i:i + 2]) < 2:\n"
         "            axes[1].remove()\n        pdf.savefig(fig)\n        plt.close(fig)",
         {"post": _post("_pages('a4.pdf')", "_boxes('a4.pdf')"), "require": ["PdfPages", "subplots", "remove|delaxes"]}),
        ("metadata", f"Open the report with document info — `PdfPages('report.pdf', metadata={{'Title': {title!r}, "
                     f"'Author': {author!r}, 'Subject': 'training'}})` — and write one page per metric. (Checked: pages "
                     f"and the info in the file's bytes.)",
         IMP_PDF + f"info = {{'Title': {title!r}, 'Author': {author!r}, 'Subject': 'training'}}\n"
                   f"with PdfPages('report.pdf', metadata=info) as pdf:\n" + loop,
         {"post": _post("_pages('report.pdf')", f"[b'/Title ({title})' in _bytes('report.pdf'), b'/Author ({author})' in "
                                                 f"_bytes('report.pdf'), b'/Subject (training)' in _bytes('report.pdf')]")}),
        ("isfinite", "Skip diverged runs: `runs` maps optimizer names to loss curves, and some end in `nan`. Write "
                     "`runs.pdf` with one page per run whose curve is all finite (`np.isfinite(c).all()`), titled with its "
                     "name, and return the list of names that got a page (in `runs` order).",
         IMP_PDF + "kept = []\nwith PdfPages('runs.pdf') as pdf:\n    for name, c in runs.items():\n"
         "        if not np.isfinite(c).all():\n            continue\n        fig, ax = plt.subplots()\n        ax.plot(epochs, c)\n"
         "        ax.set_title(name)\n        pdf.savefig(fig)\n        plt.close(fig)\n        kept.append(name)\nresult = kept",
         {"post": _post("_", "_pages('runs.pdf')")}),
        ("savefig", "Each page twice: write `report.pdf` (one page per metric, in order) and also save each page's figure "
                    "as a thumbnail `page1.png`, `page2.png`, … at 10 dpi before closing it. (Checked: the files, the page "
                    "count and a thumbnail's shape.)",
         IMP_PDF + "with PdfPages('report.pdf') as pdf:\n    for i, (name, values) in enumerate(metrics.items(), start=1):\n"
         "        fig, ax = plt.subplots()\n        ax.plot(epochs, values)\n        pdf.savefig(fig)\n"
         "        fig.savefig(f'page{i}.png', dpi=10)\n        plt.close(fig)",
         {"post": _post("_files()", "_pages('report.pdf')", "_shape('page1.png')")}),
    ]
    return catalog(rng, setup=setup, items=items, intro=f"{desc}; `metrics` maps 'train loss', 'val loss' and 'val acc' "
                                                         f"to their curves. Import `PdfPages` from "
                                                         f"`matplotlib.backends.backend_pdf`.")


@template(M, "medium")
def batch_exports(rng):
    s = seeded(rng)
    names = rng.sample(["adam", "sgd", "rmsprop", "lion", "adamw", "adagrad", "nadam"], rng.randint(3, 5))
    k = len(names)
    setup = (f"names = {lit(names)}\nruns = np.round(np.random.default_rng({s}).normal(size=({k}, 20)).cumsum(axis=1), 3)"
             f"   # one curve per optimizer")
    small = "fig, ax = plt.subplots(figsize=(2, 1.5))"
    items = [
        ("plt.close", f"Save one PNG per run, `f'{{name}}.png'` (`plt.subplots(figsize=(2, 1.5))`, dpi 20), closing each figure: create → save "
                      f"→ close. (Checked: the files and the open-figure count.)",
         f"for name, r in zip(names, runs):\n    {small}\n    ax.plot(r)\n    fig.savefig(f'{{name}}.png', dpi=20)\n    plt.close(fig)",
         {"post": _post("_files()", "len(plt.get_fignums())", "_shape(names[0] + '.png')"), "require": ["savefig", "close"]}),
        ("Figure", "Skip pyplot entirely: a `matplotlib.figure.Figure` (`fig = Figure(figsize=(2, 1.5))`, `ax = "
                   "fig.subplots()`) isn't tracked by pyplot, so nothing needs closing — the pattern for web servers. "
                   "Save `f'{name}.png'` per run at dpi 20. (Checked: the files and that pyplot has no open figures.)",
         "from matplotlib.figure import Figure\nfor name, r in zip(names, runs):\n    fig = Figure(figsize=(2, 1.5))\n"
         "    ax = fig.subplots()\n    ax.plot(r)\n    fig.savefig(f'{name}.png', dpi=20)",
         {"post": _post("_files()", "plt.get_fignums()", "_shape(names[0] + '.png')"), "require": ["Figure", "savefig"]}),
        ("rc_context", "Put the export settings in one place: within `with plt.rc_context({'savefig.dpi': 20, "
                       "'savefig.transparent': True}):` save every run as `f'{name}.png'` (2 × 1.5 inches) **without** "
                       "savefig arguments, closing each figure. (Checked: files, a shape, a corner's alpha, and that "
                       "the settings are back to normal afterwards.)",
         f"with plt.rc_context({{'savefig.dpi': 20, 'savefig.transparent': True}}):\n    for name, r in zip(names, runs):\n"
         f"        {small}\n        ax.plot(r)\n        fig.savefig(f'{{name}}.png')\n        plt.close(fig)",
         {"post": _post("_files()", "_shape(names[0] + '.png')", "_px(names[0] + '.png')[3]",
                        "plt.rcParams['savefig.transparent']"), "require": ["rc_context"]}),
        ("makedirs", "One folder per format: save every run as `png/{name}.png` (dpi 20) and `svg/{name}.svg`, creating "
                     "the folders first (`os.makedirs(..., exist_ok=True)`), closing each figure. (Checked: both "
                     "folders' listings.)",
         f"import os\nfor ext in ('png', 'svg'):\n    os.makedirs(ext, exist_ok=True)\nfor name, r in zip(names, runs):\n"
         f"    {small}\n    ax.plot(r)\n    fig.savefig(f'png/{{name}}.png', dpi=20)\n    fig.savefig(f'svg/{{name}}.svg')\n"
         f"    plt.close(fig)", {"post": _post("_files('png')", "_files('svg')"), "require": ["makedirs"]}),
        ("clear", f"Reuse one figure instead of making {k}: create `{small}` once, and for each run clear the Axes "
                  f"(`ax.clear()`), plot the run, title it with the name, and save `f'{{name}}.png'` at dpi 20. "
                  f"(Checked: the files, the open-figure count and the last title.)",
         f"{small}\nfor name, r in zip(names, runs):\n    ax.clear()\n    ax.plot(r)\n    ax.set_title(name)\n"
         f"    fig.savefig(f'{{name}}.png', dpi=20)",
         {"post": _post("_files()", "len(plt.get_fignums())", "fc.labels()['title']", "len(fc.ax().get_lines())"),
          "require": ["clear|cla"]}),
        ("BytesIO", "Keep the PNGs in memory: build `pngs = {name: PNG bytes}` (one 2 × 1.5 inch figure per run, dpi 20, "
                    "saved into an `io.BytesIO()`, closed after). Return `[sorted(pngs), the plt.imread shape of the "
                    "first run's bytes]`.",
         f"import io\npngs = {{}}\nfor name, r in zip(names, runs):\n    {small}\n    ax.plot(r)\n    buf = io.BytesIO()\n"
         f"    fig.savefig(buf, format='png', dpi=20)\n    plt.close(fig)\n    pngs[name] = buf.getvalue()\n"
         f"result = [sorted(pngs), plt.imread(io.BytesIO(pngs[names[0]])).shape]", {"require": ["BytesIO"]}),
        ("argmin", "Export only the best run (lowest final value): save it as `f'best_{name}.png'` at dpi 20 and return "
                   "its file name.",
         f"i = int(runs[:, -1].argmin())\n{small}\nax.plot(runs[i])\nfig.savefig(f'best_{{names[i]}}.png', dpi=20)\n"
         f"result = f'best_{{names[i]}}.png'", {"post": _post("_", "_files()"), "require": ["argmin"]}),
        ("plt.subplots", f"All runs in one file: `fig, axes = plt.subplots(1, {k}, figsize=({2 * k}, 1.5), sharey=True)`, "
                         f"run i on `axes[i]` titled with its name, saved as `all_runs.png` at dpi 20. (Checked: titles, "
                         f"line counts and the PNG's shape.)",
         f"fig, axes = plt.subplots(1, {k}, figsize=({2 * k}, 1.5), sharey=True)\nfor ax, name, r in zip(axes, names, runs):\n"
         f"    ax.plot(r)\n    ax.set_title(name)\nfig.savefig('all_runs.png', dpi=20)",
         {"post": _post("[(fc.labels(a)['title'], len(a.get_lines())) for a in fc.axes()]", "_shape('all_runs.png')")}),
    ]
    return catalog(rng, items, setup, f"`runs` holds {k} training curves (one row each, 20 steps) for the optimizers in "
                                      f"`names`.")


@template(M, "medium")
def export_metadata(rng):
    code, desc, n = curves(rng)
    title = rng.choice(["Loss curve", "Run 7", "Baseline"])
    author = rng.choice(["Ada", "Grace", "Alan"])
    lr = rng.choice(["1e-3", "3e-4", "0.01"])
    base = "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\n"
    mk = "Plot `train_loss` against `epochs` on `fig, ax = plt.subplots(figsize=(3, 2))`. "
    items = [
        ("metadata", mk + f"Save `loss.png` at 20 dpi with `metadata={{'Title': {title!r}, 'Author': {author!r}}}`, read "
                          f"the tags back with Pillow (`Image.open('loss.png').text`) and return `[text['Title'], "
                          f"text['Author']]`.",
         "from PIL import Image\n" + base + f"fig.savefig('loss.png', dpi=20, metadata={{'Title': {title!r}, 'Author': {author!r}}})\n"
         "text = Image.open('loss.png').text\nresult = [text['Title'], text['Author']]", {"require": ["metadata", "Image"]}),
        ("metadata", mk + f"Record how the figure was made: save `loss.png` at 20 dpi with the PNG tag 'Description' set "
                          f"to `f'lr={lr}, epochs={{len(epochs)}}'`. (Checked: the tag, read back with Pillow.)",
         base + f"fig.savefig('loss.png', dpi=20, metadata={{'Description': f'lr={lr}, epochs={{len(epochs)}}'}})",
         {"post": _post("__import__('PIL.Image').Image.open('loss.png').text['Description']"), "require": ["metadata"]}),
        ("metadata", mk + f"Save `loss.pdf` with `metadata={{'Title': {title!r}, 'Author': {author!r}}}` and return "
                          f"whether its bytes contain `b'/Title ({title})'` and `b'/Author ({author})'`.",
         base + f"fig.savefig('loss.pdf', metadata={{'Title': {title!r}, 'Author': {author!r}}})\n"
         f"data = open('loss.pdf', 'rb').read()\nresult = [b'/Title ({title})' in data, b'/Author ({author})' in data]",
         {"require": ["metadata"]}),
        ("metadata", mk + "Reproducible PDFs: a PDF records when it was made (`/CreationDate`), so two exports usually "
                          "differ. Save it as `a.pdf` and `b.pdf`, both with `metadata={'CreationDate': None}`, and return "
                          "whether the two files' bytes are identical.",
         base + "fig.savefig('a.pdf', metadata={'CreationDate': None})\nfig.savefig('b.pdf', metadata={'CreationDate': None})\n"
         "result = open('a.pdf', 'rb').read() == open('b.pdf', 'rb').read()", {"require": ["metadata"]}),
        ("rcParams", mk + "Reproducible SVGs need two things: no date (`metadata={'Date': None}`) and fixed element ids "
                          "(`plt.rcParams['svg.hashsalt'] = 'fixed'`, set before saving). Save `a.svg` and `b.svg` that "
                          "way and return whether their contents are identical.",
         "plt.rcParams['svg.hashsalt'] = 'fixed'\n" + base + "fig.savefig('a.svg', metadata={'Date': None})\n"
         "fig.savefig('b.svg', metadata={'Date': None})\nresult = open('a.svg').read() == open('b.svg').read()",
         {"require": ["rcParams", "metadata"]}),
        ("rcParams", f"Editable SVG text: with `plt.rcParams['svg.fonttype'] = 'none'` text is written as `<text>` "
                     f"elements instead of outlines. Plot `train_loss`, title it {title!r}, hide the ticks "
                     f"(`ax.set_xticks([])`, `ax.set_yticks([])`), save `fig.svg`, and return the texts of its `<text>` "
                     f"elements: `re.findall(r'<text[^>]*>([^<]*)</text>', svg)`.",
         f"import re\nplt.rcParams['svg.fonttype'] = 'none'\n" + base + f"ax.set_title({title!r})\nax.set_xticks([])\n"
         f"ax.set_yticks([])\nfig.savefig('fig.svg')\nsvg = open('fig.svg').read()\n"
         f"result = re.findall(r'<text[^>]*>([^<]*)</text>', svg)", {"require": ["rcParams"]}),
        ("rcParams", mk + f"With `plt.rcParams['pdf.fonttype'] = 42` (set before saving) Matplotlib embeds TrueType fonts "
                          f"instead of Type 3. Title the plot {title!r}, save `paper.pdf`, and return `[b'/FontFile2' in "
                          f"data, b'/Type3' in data]` for its bytes.",
         "plt.rcParams['pdf.fonttype'] = 42\n" + base + f"ax.set_title({title!r})\nfig.savefig('paper.pdf')\n"
         "data = open('paper.pdf', 'rb').read()\nresult = [b'/FontFile2' in data, b'/Type3' in data]", {"require": ["rcParams"]}),
        ("savefig", mk + "PNGs carry no date, so the same figure gives the same bytes: save it as `a.png` and `b.png` at "
                         "20 dpi and return `[the two files are identical, the first 8 bytes]`.",
         base + "fig.savefig('a.png', dpi=20)\nfig.savefig('b.png', dpi=20)\nda, db = open('a.png', 'rb').read(), open('b.png', 'rb').read()\n"
         "result = [da == db, da[:8]]"),
    ]
    return catalog(rng, items, code, f"{desc}. Put information in the files, and make exports reproducible.")


@template(M, "medium")
def raster_vs_vector(rng):
    s = seeded(rng)
    n1 = rng.choice([30, 40, 50])
    n2 = 4 * n1
    setup = f"g = np.random.default_rng({s})\npts = np.round(g.normal(size=({n2}, 2)), 3)   # embedding points"
    sc = "fig, ax = plt.subplots(figsize=(2, 2))\nax.scatter(pts[:, 0], pts[:, 1], rasterized=True)\n"
    items = [
        ("getsize", f"Vector files grow with the data, PNGs don't: on two 2 × 2 inch figures scatter the first {n1} "
                    f"points and all {n2}, and save each as SVG (`small.svg`, `big.svg`) and PNG at 20 dpi (`small.png`, "
                    f"`big.png`). Return `[big.svg is larger than small.svg, the two PNG shapes are equal]` "
                    f"(`os.path.getsize`).",
         f"import os\nfor stem, k in (('small', {n1}), ('big', {n2})):\n    fig, ax = plt.subplots(figsize=(2, 2))\n"
         f"    ax.scatter(pts[:k, 0], pts[:k, 1])\n    fig.savefig(f'{{stem}}.svg')\n    fig.savefig(f'{{stem}}.png', dpi=20)\n"
         f"result = [os.path.getsize('big.svg') > os.path.getsize('small.svg'), "
         f"plt.imread('big.png').shape == plt.imread('small.png').shape]", {"require": ["getsize", "savefig"]}),
        ("rasterized", "Keep a vector file light by rasterizing just the point cloud: scatter all points with "
                       "`rasterized=True` on `plt.subplots(figsize=(2, 2))` and save `fig.svg`. Return how many `<image` "
                       "elements the SVG contains (axes, ticks and labels stay vector).",
         sc + "fig.savefig('fig.svg')\nresult = open('fig.svg').read().count('<image')", {"require": ["rasterized"]}),
        ("rasterized", "Same in PDF: save the scatter of all points twice, `r.pdf` with `rasterized=True` and `v.pdf` "
                       "without (2 × 2 inch figures). Return whether each file embeds a bitmap, `[b'/Subtype /Image' in "
                       "r_bytes, b'/Subtype /Image' in v_bytes]`.",
         "for name, flag in (('r', True), ('v', False)):\n    fig, ax = plt.subplots(figsize=(2, 2))\n"
         "    ax.scatter(pts[:, 0], pts[:, 1], rasterized=flag)\n    fig.savefig(f'{name}.pdf')\n"
         "result = [b'/Subtype /Image' in open(f, 'rb').read() for f in ('r.pdf', 'v.pdf')]", {"require": ["rasterized"]}),
        ("set_rasterization_zorder", "Rasterize everything below a z-order: on `plt.subplots(figsize=(2, 2))` call "
                                     "`ax.set_rasterization_zorder(0)`, scatter all points with `zorder=-1`, add a vector "
                                     "line `ax.plot([-3, 3], [-3, 3])`, save `fig.svg`, and return the number of `<image` "
                                     "elements.",
         "fig, ax = plt.subplots(figsize=(2, 2))\nax.set_rasterization_zorder(0)\nax.scatter(pts[:, 0], pts[:, 1], zorder=-1)\n"
         "ax.plot([-3, 3], [-3, 3])\nfig.savefig('fig.svg')\nresult = open('fig.svg').read().count('<image')",
         {"require": ["set_rasterization_zorder", "zorder"]}),
        ("rasterized", "In a vector file the dpi only affects rasterized parts: save the rasterized scatter (2 × 2 inches) "
                       "as `lo.pdf` at 20 dpi and `hi.pdf` at 50 dpi, and return whether `hi.pdf` is the larger file.",
         "import os\n" + sc + "fig.savefig('lo.pdf', dpi=20)\nfig.savefig('hi.pdf', dpi=50)\n"
         "result = os.path.getsize('hi.pdf') > os.path.getsize('lo.pdf')", {"require": ["rasterized", "dpi"]}),
    ]
    return catalog(rng, items, setup, f"`pts` holds {n2} 2-D points.")


# ================================================================= hard

@template(M, "hard")
def export_helpers(rng):
    code, desc, n = curves(rng)
    widths = sorted(rng.sample([60, 80, 100, 120, 150], 2))
    nimg, ncols = rng.choice([(5, 3), (7, 4), (4, 2), (6, 4)])
    mkfig = "_f, _a = plt.subplots(figsize=(2, 1)); _a.plot(epochs, train_loss)\n"
    items = [
        ("savefig", "Write `save_all(fig, stem, formats=('png', 'pdf', 'svg'), dpi=50)` that saves `fig` as "
                    "`f'{stem}.{fmt}'` for every format (at `dpi`) and returns the list of paths. (Checked by calling it "
                    "on a small figure with `formats=('png', 'svg')` and `dpi=20`, and with the defaults.)",
         "def save_all(fig, stem, formats=('png', 'pdf', 'svg'), dpi=50):\n    paths = []\n    for fmt in formats:\n"
         "        path = f'{stem}.{fmt}'\n        fig.savefig(path, dpi=dpi)\n        paths.append(path)\n    return paths",
         {"post": _post(mkfig + "_r = save_all(_f, 'a', formats=('png', 'svg'), dpi=20), save_all(_f, 'b')\n_r",
                        "_files()", "_shape('a.png')", "_shape('b.png')", "_head('b.pdf', 5)")}),
        ("buffer_rgba", "Write `fig_to_array(fig)` returning the figure's pixels as a uint8 RGB array of shape (height, "
                        "width, 3) at the figure's dpi, **copied** so it survives `plt.close(fig)`. (Checked on a 2 × 1 "
                        "inch figure at 30 dpi, after closing it.)",
         "def fig_to_array(fig):\n    fig.canvas.draw()\n    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()",
         {"post": _post("_f, _a = plt.subplots(figsize=(2, 1), dpi=30); _a.plot(epochs, train_loss)\n_x = fig_to_array(_f)\n"
                        "plt.close(_f)\n_x.shape", "_x.dtype.name", "_x[0, 0].tolist()", "_x.flags['OWNDATA'] or _x.base is "
                        "not None and _x.base.flags['OWNDATA']"), "require": ["buffer_rgba", "draw"]}),
        ("BytesIO", "Write `fig_to_png_bytes(fig, dpi=50)` returning the PNG file's bytes, made in memory with "
                    "`io.BytesIO` (no file on disk). (Checked: the signature, no files written, the image shape at dpi "
                    "20.)",
         "import io\n\ndef fig_to_png_bytes(fig, dpi=50):\n    buf = io.BytesIO()\n    fig.savefig(buf, format='png', dpi=dpi)\n"
         "    return buf.getvalue()",
         {"post": _post("import io as _io\n" + mkfig + "_b = fig_to_png_bytes(_f, dpi=20)\n_b[:8]", "_files()",
                        "list(plt.imread(_io.BytesIO(_b)).shape)", "list(plt.imread(_io.BytesIO(fig_to_png_bytes(_f))).shape)"),
          "require": ["BytesIO"]}),
        ("b64encode", "Write `fig_to_data_uri(fig, dpi=50)` returning `'data:image/png;base64,...'` (the PNG bytes "
                      "base64-encoded) for an `<img src=...>` in HTML. (Checked by decoding the result and reading the "
                      "image back.)",
         "import io, base64\n\ndef fig_to_data_uri(fig, dpi=50):\n    buf = io.BytesIO()\n    fig.savefig(buf, format='png', dpi=dpi)\n"
         "    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()",
         {"post": _post("import io as _io, base64 as _b64\n" + mkfig + "_u = fig_to_data_uri(_f, dpi=20)\n_u[:22]",
                        "list(plt.imread(_io.BytesIO(_b64.b64decode(_u.split(',', 1)[1]))).shape)"),
          "require": ["b64encode"]}),
        ("PdfPages", "Write `write_report(path, curves)` where `curves` maps titles to 1-D arrays: a PDF with one page per "
                     "entry (the curve, titled with its key), returning the number of pages written and leaving no "
                     "figure open. (Checked by calling it twice.)",
         IMP_PDF + "\ndef write_report(path, curves):\n    with PdfPages(path) as pdf:\n        for title, c in curves.items():\n"
         "            fig, ax = plt.subplots()\n            ax.plot(c)\n            ax.set_title(title)\n            pdf.savefig(fig)\n"
         "            plt.close(fig)\n        return pdf.get_pagecount()",
         {"post": _post("_n1 = write_report('a.pdf', {'train': train_loss, 'val': val_loss, 'gap': val_loss - train_loss})\n"
                        "_n2 = write_report('b.pdf', {'train': train_loss})\n[_n1, _n2]", "[_pages('a.pdf'), _pages('b.pdf')]",
                        "plt.get_fignums()"), "require": ["PdfPages"]}),
        ("set_size_inches", "Write `save_for_width(fig, path, width_px, dpi=50)`: resize `fig` so that it is exactly "
                            "`width_px` pixels wide at `dpi`, keeping its aspect ratio, and save it there at that dpi. "
                            f"(Checked: the PNG shapes for widths {widths} at dpi 20 on a 4 × 2 inch figure.)",
         "def save_for_width(fig, path, width_px, dpi=50):\n    w, h = fig.get_size_inches()\n"
         "    fig.set_size_inches(width_px / dpi, width_px / dpi * h / w)\n    fig.savefig(path, dpi=dpi)",
         {"post": _post("_f, _a = plt.subplots(figsize=(4, 2)); _a.plot(epochs, train_loss)\n"
                        f"for _w in {widths}:\n    save_for_width(_f, f'w{{_w}}.png', _w, dpi=20)\n"
                        f"[_shape(f'w{{_w}}.png') for _w in {widths}]"), "require": ["set_size_inches", "get_size_inches"]}),
        ("get_window_extent", "Write `save_axes(ax, path, dpi=50, pad=0.0)` that saves only the area of `ax` — its box "
                              "in inches, `ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())`, grown by "
                              "`pad` inches on every side (`.padded(pad)`) — using `bbox_inches`. (Checked on the right "
                              "Axes of a 1 × 2 figure, with and without padding.)",
         "def save_axes(ax, path, dpi=50, pad=0.0):\n    fig = ax.get_figure()\n"
         "    box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).padded(pad)\n"
         "    fig.savefig(path, dpi=dpi, bbox_inches=box)",
         {"post": _post("_f, (_a, _b) = plt.subplots(1, 2, figsize=(5, 2)); _b.plot(epochs, val_loss)\n"
                        "save_axes(_b, 'b0.png', dpi=20)\nsave_axes(_b, 'b1.png', dpi=20, pad=0.25)\n_shape('b0.png')",
                        "_shape('b1.png')"), "require": ["get_window_extent", "bbox_inches"]}),
        ("metadata", "Write `save_reproducible(fig, path)` that exports `fig` so that saving the same figure twice "
                     "gives byte-identical files: for a `.pdf` path pass `metadata={'CreationDate': None}`, for `.svg` "
                     "`metadata={'Date': None}` with `svg.hashsalt` fixed (use `plt.rc_context({'svg.hashsalt': 'tl'})` "
                     "so the global setting is untouched), anything else as is. (Checked by saving one figure twice in "
                     "each of PDF, SVG and PNG and comparing the bytes.)",
         "def save_reproducible(fig, path):\n    if path.endswith('.pdf'):\n        fig.savefig(path, metadata={'CreationDate': None})\n"
         "    elif path.endswith('.svg'):\n        with plt.rc_context({'svg.hashsalt': 'tl'}):\n"
         "            fig.savefig(path, metadata={'Date': None})\n    else:\n        fig.savefig(path)",
         {"post": _post("_f, _a = plt.subplots(figsize=(2, 1.5)); _a.plot(epochs, val_loss); _a.set_title('val')\n"
                        "_same = []\nfor _e in ('pdf', 'svg', 'png'):\n    save_reproducible(_f, f'a.{_e}')\n"
                        "    save_reproducible(_f, f'b.{_e}')\n    _same.append(_bytes(f'a.{_e}') == _bytes(f'b.{_e}'))\n_same",
                        "_head('a.pdf', 5)", "_head('a.svg', 5)", "plt.rcParams['svg.hashsalt']"),
          "require": ["metadata", "rc_context"]}),
        ("imsave", "Write `save_grid(images, path, ncols)`: tile equally sized 2-D arrays into one array (row-major, "
                   "`ncols` per row, unused tiles filled with 0), write it with `plt.imsave(path, grid, cmap='gray')`, "
                   f"and return the grid. (Checked with {nimg} random 4 × 6 images and ncols = {ncols}.)",
         "def save_grid(images, path, ncols):\n    h, w = images[0].shape\n    nrows = -(-len(images) // ncols)\n"
         "    grid = np.zeros((nrows * h, ncols * w))\n    for i, im in enumerate(images):\n        r, c = divmod(i, ncols)\n"
         "        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im\n    plt.imsave(path, grid, cmap='gray')\n    return grid",
         {"post": _post(f"_g = save_grid(list(np.random.default_rng(0).random(({nimg}, 4, 6))), 'grid.png', {ncols})\n"
                        f"fc.r(_g, 6)", "_shape('grid.png')"), "require": ["imsave"]}),
    ]
    return catalog(rng, items, code, f"{desc}. Write each export helper as a reusable function; it is checked by "
                                     f"calling it (the data is only for your own tests).")


@template(M, "hard")
def export_bugs(rng):
    code, desc, n = curves(rng)
    s = seeded(rng)
    setup = code + f"\nheat = np.round(np.random.default_rng({s}).random((8, 8)), 3)"
    long = rng.choice(["training loss (cross-entropy)", "validation loss per token", "mean squared error (val)"])
    k = rng.choice([21, 22, 24])
    items = [
        ("savefig", "This saves a **blank** image:\n```python\nplt.plot(epochs, train_loss)\nplt.figure(figsize=(3, 2))\n"
                    "plt.savefig('loss.png', dpi=20)\n```\n`plt.figure` made a new, empty current figure. Fix it so "
                    "`loss.png` (3 × 2 inches, 20 dpi) shows the curve. (Checked: the shape and that some pixels are dark.)",
         "plt.figure(figsize=(3, 2))\nplt.plot(epochs, train_loss)\nplt.savefig('loss.png', dpi=20)",
         {"post": _post("_shape('loss.png')", "bool((plt.imread('loss.png')[..., :3] < 0.5).any())")}),
        ("bbox_inches", f"In the saved file the long y label is cut off at the left edge:\n```python\nfig, ax = "
                        f"plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\nax.set_ylabel({long!r}, fontsize=14)\n"
                        f"fig.savefig('loss.png', dpi=30)\n```\nKeep the figure as it is and fix the export so nothing is "
                        f"clipped.",
         f"fig, ax = plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\nax.set_ylabel({long!r}, fontsize=14)\n"
         f"fig.savefig('loss.png', dpi=30, bbox_inches='tight')",
         {"post": _post("fc.labels()['ylabel']", "_shape('loss.png')"), "require": ["bbox_inches"]}),
        ("draw", "This crashes with *'FigureCanvasAgg' object has no attribute 'renderer'*:\n```python\nfig, ax = "
                 "plt.subplots(figsize=(2, 1.5), dpi=30)\nax.plot(epochs, val_loss)\nrgba = np.asarray(fig.canvas."
                 "buffer_rgba())\n```\nThe canvas holds no pixels until it is drawn. Fix it and return `rgba.shape`.",
         "fig, ax = plt.subplots(figsize=(2, 1.5), dpi=30)\nax.plot(epochs, val_loss)\nfig.canvas.draw()\n"
         "rgba = np.asarray(fig.canvas.buffer_rgba())\nresult = rgba.shape", {"require": ["draw", "buffer_rgba"]}),
        ("plt.close", f"Matplotlib warns *More than 20 figures have been opened*:\n```python\nfor i in range({k}):\n"
                      f"    fig, ax = plt.subplots(figsize=(1, 1))\n    ax.plot(epochs[:i + 2], train_loss[:i + 2])\n"
                      f"    fig.savefig(f'step{{i:02d}}.png', dpi=10)\n```\nFix the leak. Return `[number of PNG files, "
                      f"number of open figures]`.",
         f"import os\nfor i in range({k}):\n    fig, ax = plt.subplots(figsize=(1, 1))\n    ax.plot(epochs[:i + 2], train_loss[:i + 2])\n"
         f"    fig.savefig(f'step{{i:02d}}.png', dpi=10)\n    plt.close(fig)\n"
         f"result = [sum(f.endswith('.png') for f in os.listdir()), len(plt.get_fignums())]", {"require": ["close"]}),
        ("format", "The paper's build fails because `report.pdf` isn't a PDF:\n```python\nfig, ax = plt.subplots()\n"
                   "ax.plot(epochs, val_loss)\nfig.savefig('report.pdf', format='png')\n```\nFix it. (Checked: the "
                   "file's first 5 bytes.)", "fig, ax = plt.subplots()\nax.plot(epochs, val_loss)\nfig.savefig('report.pdf')",
         {"post": _post("_head('report.pdf', 5)", "_pages('report.pdf')"), "require": ["savefig"]}),
        ("pil_kwargs", "This crashes with *print_jpg() got an unexpected keyword argument 'quality'*:\n```python\n"
                       "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\nfig.savefig('loss.jpg', "
                       "dpi=20, quality=85)\n```\nJPEG options are Pillow's. Fix it (quality 85).",
         "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\nfig.savefig('loss.jpg', dpi=20, pil_kwargs={'quality': 85})",
         {"post": _post("_head('loss.jpg', 3)", "_shape('loss.jpg')")}),
        ("PdfPages", "`report.pdf` comes out **empty** (0 bytes):\n```python\npdf = PdfPages('report.pdf')\nfor c in "
                     "(train_loss, val_loss):\n    fig, ax = plt.subplots()\n    ax.plot(epochs, c)\n    pdf.savefig(fig)\n"
                     "    plt.close(fig)\n```\nThe file is only completed when the `PdfPages` is closed. Fix it (a `with` "
                     "block is safest). `PdfPages` comes from `matplotlib.backends.backend_pdf`.",
         IMP_PDF + "with PdfPages('report.pdf') as pdf:\n    for c in (train_loss, val_loss):\n        fig, ax = plt.subplots()\n"
         "        ax.plot(epochs, c)\n        pdf.savefig(fig)\n        plt.close(fig)",
         {"post": _post("_pages('report.pdf')", "len(plt.get_fignums())"), "require": ["PdfPages"]}),
        ("seek", "This crashes with *not a PNG file*:\n```python\nimport io\nbuf = io.BytesIO()\nfig, ax = "
                 "plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\nfig.savefig(buf, format='png', dpi=20)\n"
                 "img = plt.imread(buf)\n```\nAfter writing, the buffer's position is at its end. Fix it and return "
                 "`img.shape`.",
         "import io\nbuf = io.BytesIO()\nfig, ax = plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\n"
         "fig.savefig(buf, format='png', dpi=20)\nbuf.seek(0)\nimg = plt.imread(buf)\nresult = img.shape", {"require": ["seek"]}),
        ("imsave", "This crashes with *Floating point image RGB values must be in the 0..1 range*:\n```python\n"
                   "rgb = np.stack([heat, heat, heat], axis=-1) * 255\nplt.imsave('heat.png', rgb)\n```\nSave the same "
                   "gray picture with floats scaled to 0 … 1. (Checked: the shape and two pixels of `heat.png`.)",
         "rgb = np.stack([heat, heat, heat], axis=-1)\nplt.imsave('heat.png', rgb)",
         {"post": _post("_shape('heat.png')", "_px('heat.png')", "_px('heat.png', 7, 7)"), "require": ["imsave"]}),
        ("plt.subplots", "The PNG should be 300 × 200 pixels, but `figsize` is in **inches**, so this makes a 300 × 200 "
                         "inch figure:\n```python\nfig, ax = plt.subplots(figsize=(300, 200))\nax.plot(epochs, train_loss)\n"
                         "fig.savefig('small.png', dpi=1)\n```\nUse a 6 × 4 inch figure saved at 50 dpi.",
         "fig, ax = plt.subplots(figsize=(6, 4))\nax.plot(epochs, train_loss)\nfig.savefig('small.png', dpi=50)",
         {"post": _post("fc.size()", "_shape('small.png')")}),
        ("makedirs", "This crashes with *FileNotFoundError: ... 'plots/loss.png'*:\n```python\nfig, ax = "
                     "plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, val_loss)\nfig.savefig('plots/loss.png', dpi=20)\n```\n"
                     "`savefig` doesn't create folders. Fix it (`os.makedirs(..., exist_ok=True)`), so it also works when "
                     "`plots` already exists.",
         "import os\nfig, ax = plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, val_loss)\nos.makedirs('plots', exist_ok=True)\n"
         "fig.savefig('plots/loss.png', dpi=20)", {"post": _post("_files('plots')", "_shape('plots/loss.png')")}),
        ("imread", "This sanity check claims the JPEG and the PNG export of the same figure are completely different:\n"
                   "```python\nfig, ax = plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\n"
                   "fig.savefig('a.png', dpi=20)\nfig.savefig('a.jpg', dpi=20)\n"
                   "diff = abs(plt.imread('a.jpg').mean() - plt.imread('a.png')[..., :3].mean())\nresult = diff < 0.05\n```\n"
                   "`imread` returns a PNG as floats in 0 … 1 but a JPEG as integers in 0 … 255. Scale the JPEG to 0 … 1 "
                   "so the comparison means something, and return `result`.",
         "fig, ax = plt.subplots(figsize=(2, 1.5))\nax.plot(epochs, train_loss)\nfig.savefig('a.png', dpi=20)\n"
         "fig.savefig('a.jpg', dpi=20)\ndiff = abs(plt.imread('a.jpg').mean() / 255 - plt.imread('a.png')[..., :3].mean())\n"
         "result = bool(diff < 0.05)", {"require": ["imread"]}),
        ("transparent", "This 'transparent' PNG still has a white box in the middle:\n```python\nfig, ax = "
                        "plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\nfig.patch.set_alpha(0)\n"
                        "fig.savefig('t.png', dpi=20)\n```\nOnly the figure's background was cleared, not the Axes'. Use "
                        "the savefig option that clears both. (Checked: the median alpha of `t.png`.)",
         "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\nfig.savefig('t.png', dpi=20, transparent=True)",
         {"post": _post("float(np.median(plt.imread('t.png')[..., 3]))", "_shape('t.png')"), "require": ["transparent"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `heat` is an 8 × 8 array in 0 … 1. Fix each export bug.")


@template(M, "hard")
def export_drills(rng):
    code, desc, n = curves(rng, acc=True)
    s = seeded(rng)
    lo, hi = rng.choice([(20, 50), (10, 40), (25, 50), (15, 45)])
    k = rng.randint(3, 5)
    setup = code + f"\nA = np.round(np.random.default_rng({s}).normal(size=(6, 9)), 3)"
    two = "fig, ax = plt.subplots(figsize=(4, 3))\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
    items = [
        ("savefig", f"Drill 1: export the same figure (both losses on `plt.subplots(figsize=(4, 3))`) at {lo} and {hi} dpi "
                    f"as `lo.png` and `hi.png`. Return `[lo shape, hi shape, size of hi.png / size of lo.png > 1]` "
                    f"(`os.path.getsize`).",
         "import os\n" + two + f"fig.savefig('lo.png', dpi={lo})\nfig.savefig('hi.png', dpi={hi})\n"
         "result = [plt.imread('lo.png').shape, plt.imread('hi.png').shape, os.path.getsize('hi.png') / os.path.getsize('lo.png') > 1]",
         {"require": ["savefig", "getsize"]}),
        ("PdfPages", f"Drill 2: a {k}-page PDF report `report.pdf`, one plot per page: page k (k = 0 … {k - 1}) plots "
                     f"`np.arange(10) ** k` titled `f'page {{k}}'`, each figure closed after `pdf.savefig`. Return `[the "
                     f"page count from the file's bytes, len(plt.get_fignums())]` (count `{PAGES_RE}` matches).",
         IMP_PDF + f"import re\nwith PdfPages('report.pdf') as pdf:\n    for k in range({k}):\n        f, a = plt.subplots()\n"
                   f"        a.plot(np.arange(10) ** k)\n        a.set_title(f'page {{k}}')\n        pdf.savefig(f)\n        plt.close(f)\n"
                   f"data = open('report.pdf', 'rb').read()\nresult = [len(re.findall({PAGES_RE}, data)), len(plt.get_fignums())]",
         {"require": ["PdfPages", "arange"]}),
        ("buffer_rgba", "Drill 3: convert a figure (both losses on `plt.subplots(figsize=(2, 1.5), dpi=40)`) to a NumPy "
                        "RGBA array. Return `[shape, dtype name, the top-left pixel as a list]`.",
         "fig, ax = plt.subplots(figsize=(2, 1.5), dpi=40)\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\n"
         "fig.canvas.draw()\nrgba = np.asarray(fig.canvas.buffer_rgba())\nresult = [rgba.shape, rgba.dtype.name, rgba[0, 0].tolist()]",
         {"require": ["buffer_rgba", "draw", "asarray"]}),
        ("imsave", "Round trip through a file: save `A` with `plt.imsave('a.png', A, cmap='gray')`, read it back, and "
                   "return whether channel 0 equals the min-max scaled `A`, `(A - A.min()) / (A.max() - A.min())`, to "
                   "within 8-bit precision (largest difference ≤ 1/255), together with the read-back shape.",
         "plt.imsave('a.png', A, cmap='gray')\nback = plt.imread('a.png')\nscaled = (A - A.min()) / (A.max() - A.min())\n"
         "result = [bool(np.abs(back[..., 0] - scaled).max() <= 1 / 255), back.shape]", {"require": ["imsave", "imread"]}),
        ("imshow", "A figure of a figure: save the loss plot (`plt.subplots(figsize=(3, 2))`, both losses) as `plot.png` "
                   "at 20 dpi, read it back with `plt.imread`, and show it on a new `fig2, ax2 = plt.subplots()` with "
                   "`ax2.imshow(img)` and `ax2.axis('off')`. (Checked: the image on `ax2` and that its axis is off.)",
         "fig, ax = plt.subplots(figsize=(3, 2))\nax.plot(epochs, train_loss)\nax.plot(epochs, val_loss)\nfig.savefig('plot.png', dpi=20)\n"
         "img = plt.imread('plot.png')\nfig2, ax2 = plt.subplots()\nax2.imshow(img)\nax2.axis('off')",
         {"post": _post("[i['shape'] for i in fc.images(fc.ax())]", "fc.ax().axison"),
          "require": ["imread", "imshow"]}),
        ("thumbnail", "`matplotlib.image.thumbnail(infile, thumbfile, scale)` writes a small preview of an image file. "
                      "Save the accuracy plot (`val_acc` on `plt.subplots(figsize=(4, 2))`) as `big.png` at 40 dpi, then "
                      "make `small.png` at `scale=0.25`. Return both `plt.imread` shapes.",
         "import matplotlib.image as mpimg\nfig, ax = plt.subplots(figsize=(4, 2))\nax.plot(epochs, val_acc)\n"
         "fig.savefig('big.png', dpi=40)\nmpimg.thumbnail('big.png', 'small.png', scale=0.25)\n"
         "result = [plt.imread('big.png').shape, plt.imread('small.png').shape]", {"require": ["thumbnail"]}),
        ("savefig", "Drill 1, vector side: save the 4 × 3 inch loss figure as `a.svg` and `a.pdf` (the dpi doesn't "
                    "matter for their size). Return `[the <svg> width and height attributes, the PDF's MediaBox numbers "
                    "as floats]` — both in points, inches × 72.",
         "import re\n" + two + "fig.savefig('a.svg')\nfig.savefig('a.pdf')\nsvg = open('a.svg').read()\npdf = open('a.pdf', 'rb').read()\n"
         "wh = list(re.search(r'width=\"([^\"]+)\" height=\"([^\"]+)\"', svg).groups())\n"
         "box = [float(v) for v in re.search(rb'/MediaBox\\s*\\[([^\\]]*)\\]', pdf).group(1).split()]\nresult = [wh, box]"),
    ]
    return catalog(rng, items, setup, f"{desc}; `A` is a 6 × 9 array. The guide's drills, done in code.")


@template(M, "hard")
def logging_pipelines(rng):
    code, desc, n = curves(rng, acc=True)
    every = rng.choice([3, 4, 5])
    setup = code + "\nmetrics = {'train loss': train_loss, 'val loss': val_loss, 'val acc': val_acc}"
    items = [
        ("b64encode", "An HTML report: for each entry of `metrics` render a 2 × 1.5 inch figure at 20 dpi to PNG bytes "
                      "(`io.BytesIO`), base64-encode them, and build `html` from `f'<h2>{name}</h2><img "
                      "src=\"data:image/png;base64,{b64}\">'` per metric (in order). Write it to `report.html`, close the "
                      "figures, and return `html.count('<img')`. (Checked also: the file and one decoded image.)",
         "import io, base64\nparts = []\nfor name, values in metrics.items():\n    fig, ax = plt.subplots(figsize=(2, 1.5))\n"
         "    ax.plot(epochs, values)\n    buf = io.BytesIO()\n    fig.savefig(buf, format='png', dpi=20)\n    plt.close(fig)\n"
         "    b64 = base64.b64encode(buf.getvalue()).decode()\n    parts.append(f'<h2>{name}</h2><img src=\"data:image/png;base64,{b64}\">')\n"
         "html = ''.join(parts)\nwith open('report.html', 'w') as f:\n    f.write(html)\nresult = html.count('<img')",
         {"post": _post("import io as _io, base64 as _b64\n_h = open('report.html').read()\n_",
                        "_re.findall(r'<h2>([^<]*)</h2>', _h)",
                        "list(plt.imread(_io.BytesIO(_b64.b64decode(_re.search(r'base64,([^\"]+)', _h).group(1)))).shape)",
                        "plt.get_fignums()"), "require": ["b64encode", "BytesIO"]}),
        ("buffer_rgba", f"A TensorBoard-style image log: `log = {{}}`; at every epoch e divisible by {every}, render "
                        f"`val_loss[:e]` against `epochs[:e]` on a `plt.subplots(figsize=(2, 1.5), dpi=20)` figure, store a "
                        f"copy of its RGB pixels under the key `f'val/{{e}}'`, and close the figure. Return `[sorted(log), "
                        f"sorted(set(a.shape for a in log.values()))]`.",
         f"log = {{}}\nfor e in epochs:\n    if e % {every}:\n        continue\n    fig, ax = plt.subplots(figsize=(2, 1.5), dpi=20)\n"
         f"    ax.plot(epochs[:e], val_loss[:e])\n    fig.canvas.draw()\n    log[f'val/{{e}}'] = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()\n"
         f"    plt.close(fig)\nresult = [sorted(log), sorted(set(a.shape for a in log.values()))]",
         {"require": ["buffer_rgba", "close"]}),
        ("savefig", f"Frames for a GIF later: in a folder `frames` (create it), save frame i for every epoch e divisible "
                    f"by {every} — `val_loss[:e]` against `epochs[:e]` on a `plt.subplots(figsize=(1.5, 1))` figure, as "
                    f"`frames/f000.png`, `frames/f001.png`, … at 15 dpi — closing each. (Checked: the folder's listing, "
                    f"open figures and a frame's shape.)",
         f"import os\nos.makedirs('frames', exist_ok=True)\nfor i, e in enumerate(e for e in epochs if e % {every} == 0):\n"
         f"    fig, ax = plt.subplots(figsize=(1.5, 1))\n    ax.plot(epochs[:e], val_loss[:e])\n"
         f"    fig.savefig(f'frames/f{{i:03d}}.png', dpi=15)\n    plt.close(fig)",
         {"post": _post("_files('frames')", "len(plt.get_fignums())", "_shape('frames/f000.png')"), "require": ["savefig", "close"]}),
        ("Figure", "A plotting endpoint for a web app: write `render(values, fmt='png')` that draws `values` with "
                   "`matplotlib.figure.Figure(figsize=(3, 2))` (never pyplot, so a long-running server can't leak "
                   "figures), saves it into an `io.BytesIO` in format `fmt` at 20 dpi, and returns the bytes. (Checked: "
                   "PNG / SVG / PDF signatures and that pyplot holds no figures.)",
         "import io\nfrom matplotlib.figure import Figure\n\ndef render(values, fmt='png'):\n    fig = Figure(figsize=(3, 2))\n"
         "    ax = fig.subplots()\n    ax.plot(values)\n    buf = io.BytesIO()\n    fig.savefig(buf, format=fmt, dpi=20)\n"
         "    return buf.getvalue()",
         {"post": "[render(val_loss)[:8], render(val_acc, 'svg')[:5], render(train_loss, fmt='pdf')[:5], plt.get_fignums()]",
          "require": ["Figure", "BytesIO"]}),
        ("rc_context", "One call for print and screen: write `export(fig, stem)` that, inside `plt.rc_context({'pdf.fonttype': "
                       "42, 'savefig.transparent': True})`, saves `f'{stem}.pdf'` and `f'{stem}.png'` (at 20 dpi), so the "
                       "global settings stay untouched. (Checked: the files, the PDF's TrueType fonts, the PNG's corner "
                       "alpha, and `rcParams['pdf.fonttype']` afterwards.)",
         "def export(fig, stem):\n    with plt.rc_context({'pdf.fonttype': 42, 'savefig.transparent': True}):\n"
         "        fig.savefig(f'{stem}.pdf')\n        fig.savefig(f'{stem}.png', dpi=20)",
         {"post": _post("_f, _a = plt.subplots(figsize=(3, 2)); _a.plot(epochs, val_acc); _a.set_title('acc')\nexport(_f, 'acc')\n_files()",
                        "b'/FontFile2' in _bytes('acc.pdf')", "_px('acc.png')[3]", "plt.rcParams['pdf.fonttype']",
                        "plt.rcParams['savefig.transparent']"), "require": ["rc_context"]}),
    ]
    return catalog(rng, items, setup, f"{desc}; `metrics` maps 'train loss', 'val loss' and 'val acc' to their curves. "
                                      f"Export pipelines for experiment logging.")


# ================================================================= challenge

@template(M, "challenge")
def png_row_filters(rng):
    s = seeded(rng)
    report = rng.choice(["filters", "cost", "adler"])
    H, W = rng.randint(4, 6), rng.randint(5, 8)

    def gen(seed, h, w):
        return (f"g = np.random.default_rng({seed})\nkind = g.integers(0, 3, ({h}, 1))   # rows: 0 a ramp, 1 a shared "
                f"texture, 2 dark noise\nramp = 100 + g.integers(-40, 41, ({h}, 1)) + g.integers(-6, 7, ({h}, 1)) * "
                f"np.arange({w})\ntexture = g.integers(0, 256, {w}) + g.integers(-2, 3, ({h}, 1))\n"
                f"img = (np.where(kind == 0, ramp, np.where(kind == 1, texture, g.integers(0, 6, ({h}, {w})))) % 256)"
                f".astype(np.uint8)\nreport = {report!r}")
    sol = ("x = img.astype(np.int64)\na = np.zeros_like(x)\na[:, 1:] = x[:, :-1]   # left\nb = np.zeros_like(x)\n"
           "b[1:] = x[:-1]   # up\nc = np.zeros_like(x)\nc[1:, 1:] = x[:-1, :-1]   # upper left\np = a + b - c\n"
           "pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)\n"
           "paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))\n"
           "F = np.stack([(x - q) % 256 for q in (0, a, b, (a + b) // 2, paeth)])   # 5 x H x W filtered bytes\n"
           "cost = np.minimum(F, 256 - F).sum(axis=2)   # 5 x H: sum of |byte as signed|\n"
           "best = cost.argmin(axis=0)   # ties -> the lowest filter type\nrows = np.arange(len(best))\n"
           + {"filters": "result = best",
              "cost": "result = int(cost[best, rows].sum())",
              "adler": ("stream = np.column_stack([best, F[best, rows]]).ravel()   # type byte, then the row\n"
                        "n = len(stream)\nA = (1 + int(stream.sum())) % 65521\n"
                        "B = (n + int(np.arange(n, 0, -1, dtype=np.int64) @ stream)) % 65521\nresult = B * 65536 + A")}[report])
    want = {"filters": "the chosen filter type of every row, as an array of length H",
            "cost": "the total heuristic cost of the chosen filters over all rows (an int)",
            "adler": "the Adler-32 checksum zlib stores for the filtered stream — every row's filter-type byte followed "
                     "by its filtered bytes, rows in order. Adler-32 over bytes d₁ … dₙ: A = 1 + Σ dᵢ and B = Σ (1 + d₁ + … "
                     "+ dᵢ), both mod 65521, checksum = B · 65536 + A (an int)"}[report]
    return challenge(
        "How PNG picks its row filters",
        "Before compressing, a PNG encoder (libpng, and Pillow when Matplotlib saves a PNG) filters every row of bytes "
        "so that it compresses better. For the gray image `img` (H × W bytes), let x be a byte, a the byte to its left, "
        "b the byte above and c the byte above-left (0 outside the image). The five filters store (x − pred) mod 256 "
        "with pred = **0** None: 0, **1** Sub: a, **2** Up: b, **3** Average: ⌊(a + b) / 2⌋, **4** Paeth: of a, b, c the "
        "one closest to p = a + b − c (ties: a, then b, then c). Encoders choose per row the filter whose filtered bytes "
        "have the smallest sum of |v| with v read as a signed byte, i.e. min(v, 256 − v) (ties: the lowest filter "
        f"type). Return {want}. (`report` = {report!r}.)",
        gen(s, H, W), sol, [gen(s + 1, 1000, 1000), gen(s + 2, 250, 4000)], "H · W ≤ 10^6",
        speed="O(H·W) with whole-array NumPy operations")


@template(M, "challenge")
def pdf_pagination(rng):
    s = seeded(rng)
    report = rng.choice(["pages", "fill", "suffix"])
    n, q = rng.randint(8, 14), rng.randint(4, 6)
    page, gap = rng.choice([(600, 12), (700, 20), (500, 10)])

    def gen(seed, n_, q_, pg, gp):
        return (f"g = np.random.default_rng({seed})\nh = g.integers(40, {pg // 2}, {n_})   # figure heights (points)\n"
                f"page, gap = {pg}, {gp}\nqs = g.integers(0, {n_}, {q_})\nqt = qs + g.integers(0, {n_}, {q_}) % ({n_} - qs)\n"
                f"report = {report!r}")
    head = ("S = np.concatenate([[0], np.cumsum(h + gap)])   # S[k] = sum of (h + gap) over the first k figures\n"
            "nxt = np.searchsorted(S, S[:-1] + page + gap, side='right') - 1   # first figure of the next page\n")
    lift = ("n = len(h)\nup = [np.append(nxt, n)]   # up[j][i]: start of the page 2^j pages after the one starting at i\n"
            "while (1 << len(up)) <= n:\n    up.append(up[-1][up[-1]])\ncur, cnt = qs.copy(), np.zeros(len(qs), dtype=np.int64)\n"
            "for j in reversed(range(len(up))):\n    nx = up[j][cur]\n    ok = nx <= qt\n    cur = np.where(ok, nx, cur)\n"
            "    cnt += ok.astype(np.int64) << j\n")
    sol = head + {"pages": lift + "result = cnt + 1",
                  "fill": lift + "result = S[qt + 1] - S[cur] - gap   # cur = first figure on the last page",
                  "suffix": ("n = len(h)\npages = np.zeros(n + 1, dtype=np.int64)\nfor i in range(n - 1, -1, -1):\n"
                             "    pages[i] = 1 + pages[nxt[i]]\nresult = pages[:n]")}[report]
    want = {"pages": "for every query j, the number of pages needed to print figures `qs[j] … qt[j]` (inclusive), "
                     "starting on a fresh page",
            "fill": "for every query j, the height used on the **last** page when figures `qs[j] … qt[j]` are printed "
                    "starting on a fresh page (its figures plus the gaps between them)",
            "suffix": "for every start i, the number of pages needed to print figures i … n − 1 (an array of length n; "
                      "the queries are not used)"}[report]
    return challenge(
        "Paginating a PDF report",
        "A long report stacks its figures top to bottom onto PDF pages (written with `PdfPages`). Figure i is `h[i]` "
        "points tall (`h[i] ≤ page`); a page holds a run of consecutive figures whose heights plus a `gap` between "
        "neighbours fit in `page` points, and figures are placed greedily: a new page starts when the next figure "
        f"doesn't fit. Return {want}. (`report` = {report!r}.)",
        gen(s, n, q, page, gap), sol, [gen(s + 1, 200000, 200000, page, gap), gen(s + 2, 200000, 1000, 2000, 5)],
        "n, number of queries ≤ 2·10^5",
        speed="O((n + q) log n): prefix sums + binary search for each page break, binary lifting over the breaks")


@template(M, "challenge")
def palette_quantize(rng):
    s = seeded(rng)
    report = rng.choice(["bits", "palette", "top"])
    H, W = rng.randint(4, 6), rng.randint(4, 7)

    def gen(seed, h, w, k, nb, m):
        return (f"g = np.random.default_rng({seed})\npal = g.integers(0, 256, ({k}, 3))\n"
                f"img = (pal[g.integers(0, {k}, ({h}, {w}))] ^ g.integers(0, {1 << nb}, ({h}, {w}, 3))).astype(np.uint8)\n"
                f"M = {m}   # the palette may hold at most M colors\nreport = {report!r}")
    k, nb, m = rng.randint(3, 5), rng.randint(2, 4), rng.randint(3, 8)
    sol = ("v = img.astype(np.int64)\n\ndef colors(k):\n    q = (v >> k) << k   # drop the k lowest bits of every channel\n"
           "    return ((q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]).ravel()\n\n"
           "lo, hi = 0, 8   # fewer bits never means more colors: binary search the smallest k\nwhile lo < hi:\n"
           "    mid = (lo + hi) // 2\n    if len(np.unique(colors(mid))) <= M:\n        hi = mid\n    else:\n"
           "        lo = mid + 1\n"
           + {"bits": "result = lo",
              "palette": "result = len(np.unique(colors(lo)))",
              "top": "u, cnt = np.unique(colors(lo), return_counts=True)\nresult = int(u[cnt.argmax()])   # ties: the smallest"}[report])
    want = {"bits": "that smallest k",
            "palette": "the number of distinct colors left after dropping k bits",
            "top": "the most frequent color after dropping k bits, packed as R·65536 + G·256 + B (ties: the smallest "
                   "packed value)"}[report]
    return challenge(
        "Shrinking an image to a palette",
        "An indexed PNG or a GIF frame stores a palette of at most `M` colors. A crude way to get there is to drop the "
        "k lowest bits of each channel of the RGB image `img` (H × W × 3, uint8: every value v becomes v with its k low "
        "bits cleared) for the smallest k (0 … 8) that leaves at most `M` distinct colors. Return "
        f"{want}. (`report` = {report!r}.)",
        gen(s, H, W, k, nb, m), sol, [gen(s + 1, 1000, 1000, 300, 4, 256), gen(s + 2, 500, 400, 40, 6, 64)],
        "H · W ≤ 10^6, M ≥ 1", speed="O(H·W·log H·W) for a few np.unique calls (binary search over k)",
        answer_desc=" (an int)")

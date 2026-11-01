"""Floats (figure, table, algorithm, subfigure), captions and \\includegraphics."""

from __future__ import annotations

from .commands import EXAMPLE_IMAGES, dim_pt
from .engine import cmd, env
from .tokens import detokenize

FLOAT_NAMES = {"figure": "Figure", "table": "Table", "algorithm": "Algorithm", "lstlisting": "Listing"}
GRAPHIC_KEYS = {"width", "height", "scale", "angle", "keepaspectratio", "trim", "clip", "page", "origin",
                "totalheight", "viewport", "draft", "interpolate"}


@env("figure", "figure*", "table", "table*", "algorithm", "wrapfigure", "wraptable")
def float_env(e, name, t):
    kind = name.rstrip("*").replace("wrap", "")
    if name.startswith("wrap"):
        e.read_opt()
        e.arg_text(name)
        e.read_opt()
        e.arg_text(name)
        placement = "wrap"
    else:
        p = e.read_opt()
        placement = detokenize(p).strip() if p is not None else ""
    bad = [c for c in placement if c not in "htbpH!"] if placement != "wrap" else []
    if bad:
        e.error(f"LaTeX Error: Unknown float option `{bad[0]}'.", t, "The placements are h, t, b, p, ! (and H with "
                                                                       "the float package).")
    if "H" in placement and "float" not in e.pkg_closure:
        e.error("LaTeX Error: Unknown float option `H'.", t, "[H] needs \\usepackage{float}.")
    if e.float_stack or any(f["kind"] in ("minipage",) for f in e.frames):
        e.error("LaTeX Error: Not in outer par mode.", t, "A figure or table can't go inside another float or a "
                                                          "minipage.")
    body = e.env_body(name, t)
    saved_label = e.cur_label
    e.float_stack.append({"kind": kind, "captions": 0})
    blocks = e.capture_blocks(body, kind="float", align="j")
    fl = e.float_stack.pop()
    e.cur_label = saved_label
    e.add_block({"t": "float", "kind": kind, "placement": placement, "c": blocks, "star": name.endswith("*"),
                 "captions": fl["captions"]})


@env("subfigure", "subtable")
def subfloat_env(e, name, t):
    e.read_opt()
    e.read_opt()
    e.read_opt()
    w = e.arg_text(name)
    if dim_pt(w) is None:
        e.error("Missing number, treated as zero.", t, "\\begin{subfigure}{width}: e.g. 0.48\\linewidth.")
    if not e.float_stack:
        e.error("LaTeX Error: \\caption outside float.", t, f"{name} goes inside a figure (or table).")
    body = e.env_body(name, t)
    e.float_stack.append({"kind": "sub" + (e.float_stack[-1]["kind"] if e.float_stack else "figure"), "captions": 0,
                          "sub": True})
    blocks = e.capture_blocks(body, kind="subfloat", align="j")
    e.float_stack.pop()
    e.add_inline({"t": "box", "b": {"t": "subfloat", "w": w, "wpt": dim_pt(w), "c": blocks}})


def do_caption(e, t, sub=False):
    is_of = t.v == "captionof"
    kind_of = e.arg_text("captionof") if is_of else None
    short = e.read_opt()
    arg = e.read_arg("caption", allow_par=False)
    if not e.float_stack and not is_of:
        e.error("LaTeX Error: \\caption outside float.", t,
                "\\caption only works inside figure, table or algorithm (or use \\captionof from the caption "
                "package).")
        return
    fl = e.float_stack[-1] if e.float_stack else {"kind": kind_of, "captions": 0}
    kind = kind_of or fl["kind"]
    fl["captions"] = fl.get("captions", 0) + 1
    if kind.startswith("sub"):
        parent = kind[3:]
        e.counters.step("sub" + parent)
        num = e.counters.number("sub" + parent)
        # \ref to a subfigure prints the figure number and the letter: 1a
        outer = e.float_stack[-2] if len(e.float_stack) > 1 else None
        fig = e.counters.the(parent) if outer is not None and outer.get("captions") else _next_number(e, parent)
        e.set_label_target("sub" + parent, f"{fig}{num}")
        lab = f"({num})"
    else:
        counter = kind if kind in e.counters.v else "figure"
        e.counters.step(counter)
        num = e.counters.the(counter)
        e.set_label_target(kind, num)
        lab = f"{FLOAT_NAMES.get(kind, kind.title())} {num}" + ("" if kind == "algorithm" else ":")
    inl = e.capture(arg)
    e.add_block({"t": "caption", "kind": kind, "label": lab, "num": num, "c": inl})


def _next_number(e, parent):
    """Subfigure captions come before the figure's own \\caption steps its counter: the number it will get."""
    v = e.counters.v.get(parent, 0)
    e.counters.v[parent] = v + 1
    s = e.counters.the(parent)
    e.counters.v[parent] = v
    return s


def do_includegraphics(e, t):
    star = e.read_star()
    opt = e.read_opt()
    opts = e.kv(opt)
    name = e.arg_text("includegraphics")
    for k in opts:
        if k not in GRAPHIC_KEYS:
            e.error(f"Package keyval Error: {k} undefined.", t,
                    "\\includegraphics options: width, height, scale, angle, keepaspectratio, trim, clip.")
    found = _find_image(e, name)
    if found is None:
        e.error(f"LaTeX Error: File `{name}' not found.", t,
                "Check the file name (it's case-sensitive); the example images are example-image, example-image-a "
                "and example-image-b.")
    for k in ("width", "height"):
        if isinstance(opts.get(k), str) and dim_pt(opts[k]) is None:
            e.error("Illegal unit of measure (pt inserted).", t, f"{k}= needs a length: 0.8\\linewidth, 5cm ...")
    e.add_inline({"t": "box", "b": {"t": "graphic", "file": name, "found": found, "w": opts.get("width"),
                                    "h": opts.get("height"), "scale": opts.get("scale"), "angle": opts.get("angle"),
                                    "keep": "keepaspectratio" in opts}})


def _find_image(e, name):
    if name in EXAMPLE_IMAGES:
        return {"example": name}
    cands = [name] + [name + ext for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg")]
    for p in [""] + list(e.graphics_paths):
        for c in cands:
            if p + c in e.files:
                return {"file": p + c, "svg": e.files[p + c] if e.files[p + c].lstrip().startswith("<svg") else None}
    return None

"""Command handlers (registered with @cmd): the preamble, sectioning, references, citations, fonts, footnotes,
links, graphics, spacing and boxes."""

from __future__ import annotations

import datetime
import re

from . import packages as P
from .engine import SECTION_CMDS, Stop, cmd, PREAMBLE_OK
from .model import Counters, SIZE_ORDER, text
from .tokens import Tok, detokenize, tokenize

COLOR_NAMES = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00ff00", "blue": "#0000ff",
               "cyan": "#00ffff", "magenta": "#ff00ff", "yellow": "#ffff00", "gray": "#808080",
               "darkgray": "#404040", "lightgray": "#bfbfbf", "brown": "#bf8040", "lime": "#bfff00",
               "olive": "#808000", "orange": "#ff8000", "pink": "#ffbfbf", "purple": "#bf0040", "teal": "#008080",
               "violet": "#800080"}
EXAMPLE_IMAGES = {"example-image", "example-image-a", "example-image-b", "example-image-c", "example-image-golden",
                  "example-image-1x1", "example-image-16x9", "example-image-10x16", "example-image-plain"}
DIM_RE = re.compile(r"^\s*(-?[\d.]+)?\s*\\?(pt|mm|cm|in|em|ex|bp|pc|sp|linewidth|textwidth|columnwidth|paperwidth|"
                    r"textheight|baselineskip|fill)?\s*$")
UNIT_PT = {"pt": 1, "mm": 2.845276, "cm": 28.45276, "in": 72.27, "bp": 1.00375, "pc": 12, "sp": 1 / 65536}


def dim_pt(s: str, e=None, ref_width=345.0, em=10.0) -> float | None:
    """A TeX dimension in points (\\linewidth fractions relative to `ref_width`)."""
    s = (s or "").strip().replace(" ", "")
    m = re.match(r"^(-?[\d.]*)\\?(pt|mm|cm|in|em|ex|bp|pc|sp|linewidth|textwidth|columnwidth|hsize|paperwidth|"
                 r"textheight|baselineskip)$", s)
    if not m:
        return None
    num = float(m.group(1)) if m.group(1) not in ("", "-", ".") else (1.0 if m.group(1) != "-" else -1.0)
    u = m.group(2)
    if u in UNIT_PT:
        return num * UNIT_PT[u]
    if u in ("em",):
        return num * em
    if u == "ex":
        return num * em * 0.43
    if u in ("linewidth", "textwidth", "columnwidth", "hsize"):
        return num * ref_width
    if u == "paperwidth":
        return num * 597.5
    if u == "textheight":
        return num * 550
    if u == "baselineskip":
        return num * em * 1.2
    return None


def color_of(e, spec: str) -> str | None:
    """xcolor: a name, a defined color, or a mix like blue!10 / red!50!black."""
    spec = spec.strip()
    if not spec:
        return None
    parts = spec.split("!")
    base = _named(e, parts[0])
    if base is None:
        return None
    if len(parts) == 1:
        return base
    try:
        pct = float(parts[1]) / 100
    except ValueError:
        return None
    other = _named(e, parts[2]) if len(parts) > 2 else "#ffffff"
    if other is None:
        return None
    return _mix(base, other, pct)


def _named(e, name):
    name = name.strip()
    if name in e.colors:
        return e.colors[name]
    return COLOR_NAMES.get(name)


def _mix(a, b, w):
    ra, ga, ba = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    rb, gb, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return "#%02x%02x%02x" % (round(ra * w + rb * (1 - w)), round(ga * w + gb * (1 - w)), round(ba * w + bb * (1 - w)))


def need_color(e, spec, t):
    c = color_of(e, spec)
    if c is None:
        e.error(f"Package xcolor Error: Undefined color `{spec}'.", t,
                f"Define it first: \\definecolor{{{spec}}}{{HTML}}{{1F77B4}}, or use a named color (red, blue, ...).")
    return c


# ---------------------------------------------------------------- the preamble

@cmd("documentclass")
def documentclass(e, t):
    opts = e.kv(e.read_opt())
    name = e.arg_text("documentclass")
    if e.in_document:
        e.error("LaTeX Error: Can be used only in preamble.", t, "\\documentclass is the first line of the file.")
        return
    if e.cls is not None:
        e.error("LaTeX Error: Two \\documentclass or \\documentstyle commands.", t, "")
        return
    if name not in P.CLASSES:
        e.error(f"LaTeX Error: File `{name}.cls' not found.", t,
                "Check the spelling: article, report, book or beamer.")
        name = "article"
    e.cls = name
    e.cls_opts = [k for k, v in opts.items() if v is True]
    e.counters = Counters(name)
    if name in ("report", "book", "memoir"):
        e.secnumdepth = 2
    e.refresh_available()


@cmd("usepackage", "RequirePackage")
def usepackage(e, t):
    opt_toks = e.read_opt()
    opts = [k if v is True else f"{k}={v}" for k, v in e.kv(opt_toks).items()]
    names = [n.strip() for n in e.arg_text("usepackage").split(",") if n.strip()]
    if e.in_document:
        e.error("LaTeX Error: Can be used only in preamble.", t, "\\usepackage lines go before \\begin{document}.")
        return
    for name in names:
        e.used["pkg:" + name] += 1
        if not P.known_package(name):
            e.error(f"LaTeX Error: File `{name}.sty' not found.", t,
                    f"There's no package called {name}: check the spelling (or install it: tlmgr install {name}).")
            continue
        if name in e.pkgs:
            if opts and sorted(opts) != sorted(e.pkgs[name]):
                e.error(f"LaTeX Error: Option clash for package {name}.", t,
                        f"{name} is loaded twice with different options: load it once, with all of them.")
            continue
        e.pkgs[name] = opts
        if name == "geometry":
            e.page["geometry"].update(e.kv(opt_toks))
    e.refresh_available()


@cmd("title", "author", "date", "subtitle", "institute")
def title_parts(e, t):
    opt = e.read_opt()
    e.title[t.v] = e.read_arg(t.v)
    if opt is not None:
        e.title[t.v + "_short"] = opt


@cmd("and")
def and_(e, t):
    e.add_inline({"t": "and"})


@cmd("thanks")
def thanks(e, t):
    arg = e.read_arg("thanks")
    e.title["thanks"].append(arg)
    e.add_inline({"t": "fn", "n": "*" * len(e.title["thanks"]), "thanks": True})


def _cs_name(e, t, what):
    """\\newcommand{\\name} or \\newcommand\\name -> name."""
    e.skip_spaces()
    nt = e.peek()
    if nt is not None and nt.k == "bg":
        arg = e.read_arg(what)
        arg = [x for x in arg if x.k != "sp"]
        if len(arg) == 1 and arg[0].k == "cs":
            return arg[0].v
        e.error("LaTeX Error: Missing control sequence inserted.", t,
                f"\\{what} needs a command name starting with a backslash: \\{what}{{\\name}}{{...}}.")
        return None
    nt = e.next()
    if nt is not None and nt.k == "cs":
        return nt.v
    e.error("LaTeX Error: Missing control sequence inserted.", t, "")
    return None


@cmd("newcommand", "renewcommand", "providecommand")
def newcommand(e, t):
    e.read_star()
    name = _cs_name(e, t, t.v)
    n_toks = e.read_opt()
    n = 0
    if n_toks is not None:
        try:
            n = int(detokenize(n_toks).strip())
        except ValueError:
            e.error("Missing number, treated as zero.", t,
                    "The number of arguments goes in [ ]: \\newcommand{\\name}[1]{...#1...}.")
    default = e.read_opt() if n else None
    body = e.read_arg(t.v)
    if name is None:
        return
    if name in ("headrulewidth", "footrulewidth", "baselinestretch", "arraystretch", "thesection",
                "theequation", "thefigure", "thetable", "labelenumi", "labelitemi", "qedsymbol", "proofname",
                "abstractname", "refname", "bibname", "contentsname", "figurename", "tablename"):
        e.macros[name] = {"n": 0, "default": None, "body": body, "setting": True}
        return
    how = {"newcommand": "new", "renewcommand": "renew", "providecommand": "provide"}[t.v]
    e.define(name, n, default, body, t, how)


@cmd("def")
def def_(e, t):
    nt = e.next()
    if nt is None or nt.k != "cs":
        e.error("Missing control sequence inserted.", t, "")
        return
    params = 0
    while True:
        u = e.peek()
        if u is None or u.k == "bg":
            break
        e.next()
        if u.k == "param":
            params += 1
    body = e.read_arg("def")
    e.macros[nt.v] = {"n": params, "default": None, "body": body}


@cmd("let")
def let_(e, t):
    a = e.next()
    b = e.next()
    if b is not None and b.k == "ch" and b.v == "=":
        b = e.next()
    if a is None or a.k != "cs" or b is None:
        return
    if b.k == "cs" and b.v in e.macros:
        e.macros[a.v] = dict(e.macros[b.v])
    else:
        e.macros[a.v] = {"n": 0, "default": None, "body": [b]}


@cmd("DeclareMathOperator")
def declare_math_operator(e, t):
    star = e.read_star()
    name = _cs_name(e, t, "DeclareMathOperator")
    body = e.read_arg("DeclareMathOperator")
    if name is None:
        return
    op = Tok("cs", "operatorname", t.file, t.line, t.col)
    toks = [op] + ([Tok("ch", "*", t.file, t.line, t.col)] if star else []) + \
        [Tok("bg", "{", t.file, t.line, t.col)] + body + [Tok("eg", "}", t.file, t.line, t.col)]
    if name in e.macros:
        e.error(f"LaTeX Error: Command \\{name} already defined.", t, "")
        return
    e.macros[name] = {"n": 0, "default": None, "body": toks, "operator": True}


@cmd("newenvironment", "renewenvironment")
def newenvironment(e, t):
    e.read_star()
    name = e.arg_text(t.v)
    n_toks = e.read_opt()
    n = int(detokenize(n_toks).strip() or 0) if n_toks is not None and detokenize(n_toks).strip().isdigit() else 0
    default = e.read_opt() if n else None
    begin = e.read_arg(t.v)
    end = e.read_arg(t.v)
    exists = e.env_available(name)
    if t.v == "newenvironment" and exists:
        e.error(f"LaTeX Error: Command \\{name} already defined.", t,
                f"An environment {name} exists already: use \\renewenvironment, or another name.")
        return
    if t.v == "renewenvironment" and not exists:
        e.error(f"LaTeX Error: Environment {name} undefined.", t, "\\renewenvironment only redefines existing ones.")
        return
    e.envs[name] = {"n": n, "default": default, "begin": begin, "end": end}


@cmd("newtheorem")
def newtheorem(e, t):
    star = e.read_star()
    name = e.arg_text("newtheorem")
    shared = e.read_opt()
    title = e.read_arg("newtheorem")
    within = e.read_opt()
    if star and "amsthm" not in e.pkg_closure:
        e.error("LaTeX Error: Missing \\begin{document}.", t, "\\newtheorem* needs the amsthm package.")
    if name in e.theorems or e.env_available(name):
        e.error(f"LaTeX Error: Command \\{name} already defined.", t, "")
        return
    if shared is not None:
        sh = detokenize(shared).strip()
        if sh not in e.theorems:
            e.error(f"LaTeX Error: No counter '{sh}' defined.", t,
                    f"[{sh}] must name a theorem defined before this one.")
            counter = name
            e.counters.new(counter)
        else:
            counter = e.theorems[sh]["counter"]
    else:
        counter = name
        parent = detokenize(within).strip() if within is not None else None
        e.counters.new(counter, parent)
    e.theorems[name] = {"counter": counter, "title": title, "style": e.thm_style, "numbered": not star}
    e.refresh_available()


@cmd("theoremstyle")
def theoremstyle(e, t):
    s = e.arg_text("theoremstyle")
    if s not in ("plain", "definition", "remark"):
        e.error("Undefined control sequence.", t,
                "The styles are plain, definition and remark.")
        return
    e.thm_style = s


@cmd("setlength", "addtolength")
def setlength(e, t):
    target = [x for x in e.read_arg(t.v) if x.k != "sp"]
    val = e.arg_text(t.v)
    if len(target) == 1 and target[0].k == "cs":
        e.lengths[target[0].v] = val
        if target[0].v == "parskip":
            e.page["parskip"] = val
        if target[0].v == "parindent":
            e.page["parindent"] = val


@cmd("definecolor")
def definecolor(e, t):
    name = e.arg_text("definecolor")
    model = e.arg_text("definecolor")
    spec = e.arg_text("definecolor")
    try:
        if model == "HTML":
            if not re.fullmatch(r"[0-9A-Fa-f]{6}", spec):
                raise ValueError
            e.colors[name] = "#" + spec.lower()
        elif model == "RGB":
            r, g, b = (int(x) for x in spec.split(","))
            e.colors[name] = "#%02x%02x%02x" % (r, g, b)
        elif model == "rgb":
            r, g, b = (float(x) for x in spec.split(","))
            e.colors[name] = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
        elif model == "gray":
            g = round(float(spec) * 255)
            e.colors[name] = "#%02x%02x%02x" % (g, g, g)
        elif model == "cmyk":
            c, m, y, k = (float(x) for x in spec.split(","))
            e.colors[name] = "#%02x%02x%02x" % (round(255 * (1 - c) * (1 - k)), round(255 * (1 - m) * (1 - k)),
                                                round(255 * (1 - y) * (1 - k)))
        else:
            e.error(f"Package xcolor Error: Undefined color model `{model}'.", t,
                    "The models are HTML, RGB, rgb, gray and cmyk.")
    except ValueError:
        e.error(f"Package xcolor Error: Bad color specification `{spec}'.", t,
                "HTML colors are six hex digits (1F77B4); rgb is three numbers 0-1; RGB three numbers 0-255.")


@cmd("colorlet")
def colorlet(e, t):
    name = e.arg_text("colorlet")
    spec = e.arg_text("colorlet")
    c = need_color(e, spec, t)
    if c:
        e.colors[name] = c


@cmd("pagestyle", "thispagestyle")
def pagestyle(e, t):
    s = e.arg_text(t.v)
    if s not in ("plain", "empty", "headings", "myheadings", "fancy") and s not in e.page.get("custom", {}):
        e.error("Undefined control sequence.", t,
                "The page styles are plain, empty, headings and (with fancyhdr) fancy.")
        return
    if s == "fancy" and "fancyhdr" not in e.pkg_closure:
        e.error("Undefined control sequence.", t, "The fancy page style comes from \\usepackage{fancyhdr}.")
        return
    if t.v == "pagestyle":
        e.page["style"] = s
    else:
        e.page["this"] = s


@cmd("fancyhf")
def fancyhf(e, t):
    e.read_opt()
    arg = e.read_arg("fancyhf")
    e.page["head"] = {}
    e.page["foot"] = {}
    if arg:
        for pos in "lcr":
            e.page["head"][pos] = arg
            e.page["foot"][pos] = arg


@cmd("lhead", "chead", "rhead", "lfoot", "cfoot", "rfoot")
def lhead(e, t):
    e.read_opt()
    arg = e.read_arg(t.v)
    where = "head" if "head" in t.v else "foot"
    e.page[where][t.v[0]] = arg


@cmd("fancyhead", "fancyfoot")
def fancyhead(e, t):
    pos = e.read_opt()
    arg = e.read_arg(t.v)
    where = "head" if t.v == "fancyhead" else "foot"
    spots = detokenize(pos).upper() if pos is not None else "LCR"
    for c in "LCR":
        if c in spots:
            e.page[where][c.lower()] = arg


@cmd("geometry", "newgeometry")
def geometry(e, t):
    e.page["geometry"].update(e.kv(e.read_arg(t.v)))


@cmd("onehalfspacing", "doublespacing", "singlespacing")
def spacing(e, t):
    e.page["spacing"] = {"onehalfspacing": 1.25, "doublespacing": 1.67, "singlespacing": 1.0}[t.v]


@cmd("setstretch", "linespread")
def setstretch(e, t):
    try:
        e.page["spacing"] = float(e.arg_text(t.v))
    except ValueError:
        pass


@cmd("numberwithin", "counterwithin")
def numberwithin(e, t):
    a = e.arg_text(t.v)
    b = e.arg_text(t.v)
    e.counters.parent[a] = b


@cmd("newcounter")
def newcounter(e, t):
    name = e.arg_text("newcounter")
    within = e.read_opt()
    e.counters.new(name, detokenize(within).strip() if within else None)


@cmd("setcounter", "addtocounter")
def setcounter(e, t):
    name = e.arg_text(t.v)
    val = e.arg_text(t.v)
    try:
        v = int(val)
    except ValueError:
        e.error("Missing number, treated as zero.", t, "")
        v = 0
    if name not in e.counters.v:
        e.error(f"LaTeX Error: No counter '{name}' defined.", t, "")
        return
    e.counters.v[name] = v if t.v == "setcounter" else e.counters.v[name] + v


@cmd("stepcounter", "refstepcounter")
def stepcounter(e, t):
    name = e.arg_text(t.v)
    if name not in e.counters.v:
        e.error(f"LaTeX Error: No counter '{name}' defined.", t, "")
        return
    e.counters.step(name)
    if t.v == "refstepcounter":
        e.set_label_target(name, e.counters.the(name))


@cmd("arabic", "alph", "Alph", "roman", "Roman", "fnsymbol")
def number_fmt(e, t):
    name = e.arg_text(t.v)
    from .model import FORMATS
    e.emit_text(FORMATS[t.v](e.counters.value(name)))


@cmd("the")
def the_(e, t):
    nt = e.next()
    if nt is not None and nt.k == "cs":
        name = nt.v[3:] if nt.v.startswith("the") else nt.v
        e.emit_text(e.counters.the(name))


@cmd("makeatletter", "makeatother", "relax", "protect", "ignorespaces", "nobreak", "allowbreak", "sloppy",
     "fussy", "frenchspacing", "nonfrenchspacing", "FloatBarrier", "phantomsection", "leavevmode", "strut",
     "null", "noindent_", "hyphenation", "urlstyle", "hypersetup", "graphicspath", "usetikzlibrary",
     "pgfplotsset", "tikzset", "lstset", "lstdefinestyle", "captionsetup", "sisetup", "crefname", "Crefname",
     "setlist", "tcbuselibrary", "tcbset", "newcolumntype", "setbeamertemplate", "setbeamercolor",
     "setbeamerfont", "usetheme", "usecolortheme", "usefonttheme", "useinnertheme", "useoutertheme",
     "addbibresource", "titleformat", "titlespacing", "floatstyle", "restylefloat", "pagenumbering",
     "includeonly", "logo", "titlegraphic", "algrenewcommand", "mathtoolsset", "DeclarePairedDelimiter",
     "AtBeginSection", "newtcolorbox", "beamertemplatenavigationsymbolsempty")
def settings(e, t):
    """Settings latex-lite records (or accepts and ignores): read their arguments."""
    name = t.v
    if name in ("makeatletter", "makeatother", "relax", "protect", "ignorespaces", "nobreak", "allowbreak",
                "sloppy", "fussy", "frenchspacing", "nonfrenchspacing", "FloatBarrier", "phantomsection",
                "leavevmode", "strut", "null", "beamertemplatenavigationsymbolsempty"):
        return
    if name == "setbeamertemplate":
        a = e.arg_text(name)
        e.read_opt()
        b = e.read_arg(name)
        e.page.setdefault("beamer_templates", {})[a] = detokenize(b)
        return
    if name in ("usetheme", "usecolortheme", "usefonttheme", "useinnertheme", "useoutertheme"):
        e.read_opt()
        e.page["beamer_" + name] = e.arg_text(name)
        return
    if name in ("setbeamercolor", "setbeamerfont"):
        e.arg_text(name)
        e.arg_text(name)
        return
    if name in ("crefname", "Crefname"):
        k = e.arg_text(name)
        s1 = e.arg_text(name)
        s2 = e.arg_text(name)
        e.page.setdefault("crefnames", {})[(name, k)] = (s1, s2)
        return
    if name in ("titleformat",):
        e.read_star()
        e.read_arg(name)
        e.read_opt()
        for _ in range(4):
            e.read_arg(name)
        e.read_opt()
        return
    if name == "titlespacing":
        e.read_star()
        for _ in range(4):
            e.read_arg(name)
        return
    if name in ("newcolumntype", "newtcolorbox", "DeclarePairedDelimiter"):
        e.read_arg(name)
        e.read_opt()
        e.read_arg(name)
        if name in ("DeclarePairedDelimiter",):
            e.read_arg(name)
        return
    if name == "AtBeginSection":
        e.read_opt()
        e.read_arg(name)
        return
    if name == "algrenewcommand":
        e.read_arg(name)
        e.read_opt()
        e.read_arg(name)
        return
    if name == "lstset" or name == "lstdefinestyle":
        if name == "lstdefinestyle":
            e.arg_text(name)
        e.lst_settings.update(e.kv(e.read_arg(name)))
        return
    if name == "pgfplotsset":
        e.pgfplots_settings.update(e.kv(e.read_arg(name)))
        return
    if name == "tikzset":
        from .tikz import parse_styles
        parse_styles(e, e.read_arg(name))
        return
    if name == "graphicspath":
        arg = detokenize(e.read_arg(name))
        e.graphics_paths = re.findall(r"\{([^{}]*)\}", arg) or [arg]
        return
    if name == "addbibresource":
        e.read_opt()
        e.bib["files"].append(e.arg_text(name))
        e.bib["style"] = e.bib["style"] or "biblatex"
        return
    if name == "hypersetup":
        e.page["hyperref"] = e.kv(e.read_arg(name))
        return
    if name in ("logo", "titlegraphic"):
        e.read_arg(name)
        return
    # one argument, maybe an optional one first
    e.read_opt()
    if name in ("captionsetup",):
        e.read_opt()
    e.read_arg(name)


# ---------------------------------------------------------------- paragraphs, lines, spacing

@cmd("par")
def par(e, t):
    e.end_par()


@cmd("noindent")
def noindent(e, t):
    f = e.frame()
    if f["par"] is None:
        f["noindent"] = True
    elif not f["par"]["c"]:
        f["par"]["indent"] = False


@cmd("indent")
def indent(e, t):
    e.start_par()


@cmd("\\", "newline", "linebreak")
def linebreak(e, t):
    e.read_star()
    e.read_opt()
    f = e.frame()
    if f["par"] is None or not f["par"]["c"]:
        if t.v == "\\" or t.v == "newline":
            e.error("LaTeX Error: There's no line here to end.", t,
                    "\\\\ ends a line of text; there's no text before it (use \\vspace for vertical space).")
        return
    e.add_inline({"t": "br"})


@cmd("newpage", "clearpage", "pagebreak", "cleardoublepage")
def newpage(e, t):
    e.read_opt()
    e.add_block({"t": "newpage"})


@cmd("centering", "raggedright", "raggedleft")
def centering(e, t):
    a = {"centering": "c", "raggedright": "l", "raggedleft": "r"}[t.v]
    f = e.frame()
    f["align"] = a
    if f["par"] is not None:
        f["par"]["align"] = a


@cmd("hspace")
def hspace(e, t):
    e.read_star()
    w = e.arg_text("hspace")
    pt = dim_pt(w)
    if pt is None and w.strip() not in ("\\fill", "\\stretch{1}"):
        e.error("Illegal unit of measure (pt inserted).", t,
                "A length needs a unit: 1em, 2pt, 0.5cm, 1in ...")
        pt = 0
    e.add_inline({"t": "space", "w": "fill" if pt is None else pt, "st": e.style})


@cmd("vspace")
def vspace(e, t):
    e.read_star()
    w = e.arg_text("vspace")
    pt = dim_pt(w)
    if pt is None and w.strip() not in ("\\fill", "\\stretch{1}"):
        e.error("Illegal unit of measure (pt inserted).", t, "A length needs a unit: 1em, 2pt, 0.5cm ...")
        pt = 0
    f = e.frame()
    if f["par"] is not None and f["par"]["c"]:
        # in a paragraph, \vspace adds the space after the current line
        e.add_inline({"t": "vspace_after", "h": pt or 0})
    else:
        e.add_block({"t": "vspace", "h": pt if pt is not None else "fill"})


@cmd("vfill")
def vfill(e, t):
    e.add_block({"t": "vspace", "h": "fill"})


@cmd("smallskip", "medskip", "bigskip")
def skips(e, t):
    e.add_block({"t": "vspace", "h": {"smallskip": 3, "medskip": 6, "bigskip": 12}[t.v]})


@cmd("-")
def soft_hyphen(e, t):
    e.emit_text("­")


@cmd("em")
def em_(e, t):
    e.style = e.style.with_(shape="n" if e.style.shape in ("it", "sl") else "it")


@cmd("emph")
def emph(e, t):
    arg = e.read_arg("emph")
    saved = e.style
    e.style = e.style.with_(shape="n" if e.style.shape in ("it", "sl") else "it")
    e.run_tokens(arg)
    e.style = saved


@cmd("underline")
def underline(e, t):
    arg = e.read_arg("underline")
    saved = e.style
    e.style = e.style.with_(under=True)
    e.run_tokens(arg)
    e.style = saved


@cmd("tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge")
def size(e, t):
    e.style = e.style.with_(size=t.v)


@cmd("textcolor")
def textcolor(e, t):
    e.read_opt()
    spec = e.arg_text("textcolor")
    arg = e.read_arg("textcolor")
    c = need_color(e, spec, t)
    saved = e.style
    if c:
        e.style = e.style.with_(color=c)
    e.run_tokens(arg)
    e.style = saved


@cmd("color")
def color(e, t):
    e.read_opt()
    spec = e.arg_text("color")
    c = need_color(e, spec, t)
    if c:
        e.style = e.style.with_(color=c)


@cmd("colorbox", "fcolorbox")
def colorbox(e, t):
    e.read_opt()
    frame_c = need_color(e, e.arg_text(t.v), t) if t.v == "fcolorbox" else None
    if t.v == "fcolorbox":
        e.read_opt()
    bg = need_color(e, e.arg_text(t.v), t)
    arg = e.read_arg(t.v)
    inl = e.capture(arg, e.style)
    e.add_inline({"t": "box", "b": {"t": "colorbox", "bg": bg, "frame": frame_c, "c": inl}})


@cmd("mbox", "fbox", "makebox", "framebox", "hbox")
def mbox(e, t):
    w = e.read_opt() if t.v in ("makebox", "framebox") else None
    if w is not None:
        e.read_opt()
    arg = e.read_arg(t.v)
    inl = e.capture(arg, e.style)
    e.add_inline({"t": "box", "b": {"t": "mbox", "frame": t.v in ("fbox", "framebox"), "c": inl,
                                    "w": dim_pt(detokenize(w)) if w else None}})


@cmd("parbox")
def parbox(e, t):
    e.read_opt()
    w = e.arg_text("parbox")
    arg = e.read_arg("parbox")
    blocks = e.capture_blocks(arg, align="l")
    e.add_inline({"t": "box", "b": {"t": "minipage", "w": w, "wpt": dim_pt(w), "c": blocks}})


@cmd("raisebox")
def raisebox(e, t):
    e.arg_text("raisebox")
    e.read_opt()
    e.read_opt()
    e.run_tokens(e.read_arg("raisebox"))


@cmd("rule")
def rule(e, t):
    e.read_opt()
    w = dim_pt(e.arg_text("rule")) or 0
    h = dim_pt(e.arg_text("rule")) or 0
    e.add_inline({"t": "rule", "w": w, "h": h})


@cmd("LaTeX", "TeX", "LaTeXe")
def logo(e, t):
    e.add_inline({"t": "logo", "s": t.v, "st": e.style})


@cmd("today")
def today(e, t):
    d = datetime.date.today()
    e.emit_text(f"{d.strftime('%B')} {d.day}, {d.year}")


@cmd("footnote")
def footnote(e, t):
    num = e.read_opt()
    arg = e.read_arg("footnote", allow_par=True)
    if num is not None:
        n = detokenize(num).strip()
    else:
        e.counters.step("footnote")
        n = e.counters.number("footnote")
    saved = e.cur_label
    e.set_label_target("footnote", n)
    blocks = e.capture_blocks(arg, kind="footnote")
    e.cur_label = saved
    e.footnotes.append({"n": n, "c": blocks})
    e.add_inline({"t": "fn", "n": n})


@cmd("footnotemark")
def footnotemark(e, t):
    num = e.read_opt()
    if num is None:
        e.counters.step("footnote")
        n = e.counters.number("footnote")
    else:
        n = detokenize(num).strip()
    e.add_inline({"t": "fn", "n": n})


@cmd("footnotetext")
def footnotetext(e, t):
    num = e.read_opt()
    arg = e.read_arg("footnotetext", allow_par=True)
    n = detokenize(num).strip() if num is not None else e.counters.number("footnote")
    e.footnotes.append({"n": n, "c": e.capture_blocks(arg, kind="footnote")})


@cmd("url", "path")
def url(e, t):
    nt = e.peek(skip_space=True)
    if nt is not None and nt.k == "verb":
        e.skip_spaces()
        e.next()
        u = nt.v
    else:
        u = e.arg_text("url")
    e.add_inline({"t": "link", "url": u, "c": [text(u, e.style.with_(family="tt"))], "kind": "url"})


@cmd("href")
def href(e, t):
    nt = e.peek(skip_space=True)
    if nt is not None and nt.k == "verb":
        e.skip_spaces()
        e.next()
        u = nt.v
    else:
        u = e.arg_text("href")
    inl = e.capture(e.read_arg("href"), e.style)
    e.add_inline({"t": "link", "url": u, "c": inl, "kind": "href"})


@cmd("lipsum", "blindtext", "Blindtext")
def lipsum(e, t):
    rng = e.read_opt()
    from .features import LOREM
    n1, n2 = 1, 1
    if rng is not None:
        s = detokenize(rng).strip()
        m = re.match(r"(\d+)(?:-(\d+))?", s)
        if m:
            n1 = int(m.group(1))
            n2 = int(m.group(2) or n1)
    else:
        n1, n2 = 1, 7 if t.v == "lipsum" else 1
    for k in range(n1, n2 + 1):
        e.emit_text(LOREM[(k - 1) % len(LOREM)])
        e.end_par()


@cmd("item")
def item_outside(e, t):
    e.error("LaTeX Error: Lonely \\item--perhaps a missing list environment.", t,
            "\\item belongs inside itemize, enumerate or description.")


@cmd("hline", "toprule", "midrule", "bottomrule", "cline", "cmidrule", "addlinespace", "specialrule")
def misplaced_noalign(e, t):
    e.error("Misplaced \\noalign.", t, f"\\{t.v} only works at the start of a row of a tabular.")


@cmd("multicolumn", "multirow")
def misplaced_omit(e, t):
    e.error("Misplaced \\omit." if t.v == "multicolumn" else "Undefined control sequence.", t,
            f"\\{t.v} only works inside a tabular.")


@cmd("caption", "captionof")
def caption(e, t):
    from .floats import do_caption
    do_caption(e, t)


@cmd("subcaption")
def subcaption(e, t):
    from .floats import do_caption
    do_caption(e, t, sub=True)


@cmd("includegraphics")
def includegraphics(e, t):
    from .floats import do_includegraphics
    do_includegraphics(e, t)


@cmd("rotatebox", "scalebox", "resizebox", "reflectbox")
def transform_box(e, t):
    a = e.read_opt()
    if t.v == "resizebox":
        w = e.arg_text(t.v)
        e.read_star()
        h = e.arg_text(t.v)
        arg = e.read_arg(t.v)
        inl = e.capture(arg)
        e.add_inline({"t": "box", "b": {"t": "resized", "w": w, "h": h, "c": inl}})
        return
    if t.v == "reflectbox":
        e.run_tokens(e.read_arg(t.v))
        return
    v = e.arg_text(t.v)
    arg = e.read_arg(t.v)
    inl = e.capture(arg)
    e.add_inline({"t": "box", "b": {"t": "transform", "op": t.v, "v": v, "c": inl}})


@cmd("qed")
def qed(e, t):
    e.add_inline({"t": "qed"})


@cmd("qedhere")
def qedhere(e, t):
    e.add_inline({"t": "qed"})
    e.qed_done = True


@cmd("input", "include")
def input_(e, t):
    name = e.arg_text(t.v)
    fname = name if name.endswith(".tex") else name + ".tex"
    src = e.files.get(fname)
    if src is None:
        e.error(f"LaTeX Error: File `{fname}' not found.", t,
                f"There's no file {fname} next to main.tex (paths are relative to it).")
        return
    if t.v == "include":       # \include = \clearpage, the file, \clearpage
        e.end_par()
        e.add_block({"t": "newpage"})
        e.run_tokens(tokenize(src, fname))
        e.end_par()
        e.add_block({"t": "newpage"})
        return
    e.push(tokenize(src, fname))


@cmd("enquote")
def enquote(e, t):
    arg = e.read_arg("enquote")
    e.emit_text("“")
    e.run_tokens(arg)
    e.emit_text("”")


@cmd("xspace")
def xspace(e, t):
    nt = e.peek()
    if nt is not None and nt.k == "ch" and nt.v in ".,;:!?)'":
        return
    e.emit_text(" ")


@cmd("nicefrac")
def nicefrac(e, t):
    a = e.arg_text("nicefrac")
    b = e.arg_text("nicefrac")
    e.add_inline({"t": "math", "tex": f"{{}}^{{{a}}}\\!/\\!{{}}_{{{b}}}", "canon": f"nicefrac{a}/{b}", "st": e.style})


@cmd("SI", "qty")
def si(e, t):
    e.read_opt()
    num = e.arg_text(t.v)
    unit = e.arg_text(t.v)
    e.add_inline({"t": "math", "tex": num + "\\," + _si_unit(unit), "canon": f"SI{num}{unit}", "st": e.style})


@cmd("num")
def num_(e, t):
    e.read_opt()
    n = e.arg_text("num")
    e.add_inline({"t": "math", "tex": n, "canon": n, "st": e.style})


@cmd("si", "unit")
def si_unit(e, t):
    e.read_opt()
    u = e.arg_text(t.v)
    e.add_inline({"t": "math", "tex": _si_unit(u), "canon": u, "st": e.style})


def _si_unit(u):
    m = {"\\percent": "\\%", "\\second": "\\mathrm{s}", "\\meter": "\\mathrm{m}", "\\kilo": "\\mathrm{k}",
         "\\gram": "\\mathrm{g}", "\\milli": "\\mathrm{m}", "\\hour": "\\mathrm{h}", "\\byte": "\\mathrm{B}",
         "\\giga": "\\mathrm{G}", "\\mega": "\\mathrm{M}", "\\watt": "\\mathrm{W}", "\\per": "/"}
    for k, v in m.items():
        u = u.replace(k, v)
    return u.strip() or "\\,"


# ---------------------------------------------------------------- title, abstract, contents

@cmd("maketitle")
def maketitle(e, t):
    if e.title.get("title") is None:
        e.error("LaTeX Error: No \\title given.", t, "Add \\title{...} to the preamble.")
        return
    from .features import title_block
    if e.beamer is not None:
        e.beamer.titlepage(t)
        return
    e.add_block(title_block(e))
    e.frame()["noindent"] = False


@cmd("titlepage")
def titlepage(e, t):
    if e.beamer is not None:
        e.beamer.titlepage(t)


@cmd("tableofcontents", "listoffigures", "listoftables")
def tableofcontents(e, t):
    if e.beamer is not None:
        e.read_opt()
        e.beamer.toc(t)
        return
    e.add_block({"t": {"tableofcontents": "toc", "listoffigures": "lof", "listoftables": "lot"}[t.v]})


@cmd("appendix")
def appendix(e, t):
    e.appendix = True
    top = "chapter" if e.counters.report else "section"
    e.counters.v[top] = 0
    e.counters.fmt[top] = "Alph"


@cmd(*SECTION_CMDS)
def section(e, t):
    from .features import do_section
    do_section(e, t)


# ---------------------------------------------------------------- references and citations

@cmd("label")
def label(e, t):
    key = e.arg_text("label")
    e.add_label(key, t)
    if e.float_stack:
        e.float_stack[-1].setdefault("labels", []).append(key)


@cmd("ref", "pageref", "eqref", "autoref", "Autoref", "nameref", "cref", "Cref", "labelcref", "crefrange",
     "Crefrange")
def ref(e, t):
    star = e.read_star()
    keys = [k.strip() for k in e.arg_text(t.v).split(",") if k.strip()]
    if t.v in ("crefrange", "Crefrange"):
        keys += [e.arg_text(t.v)]
    e.add_inline({"t": "ref", "keys": keys, "how": t.v, "st": e.style, "line": t.line, "file": t.file})


@cmd("cite", "citet", "citep", "citealt", "citealp", "citeauthor", "citeyear", "citenum", "Citet", "Citep",
     "textcite", "parencite", "autocite", "Textcite", "Parencite", "footcite")
def cite(e, t):
    pre = e.read_opt()
    post = e.read_opt()
    if post is None and pre is not None:
        pre, post = None, pre
    keys = [k.strip() for k in e.arg_text(t.v).split(",") if k.strip()]
    for k in keys:
        if k not in e.bib["cited"]:
            e.bib["cited"].append(k)
    e.add_inline({"t": "cite", "keys": keys, "how": t.v, "st": e.style,
                  "pre": e.capture(pre) if pre else None, "post": e.capture(post) if post else None,
                  "line": t.line, "file": t.file})


@cmd("nocite")
def nocite(e, t):
    for k in e.arg_text("nocite").split(","):
        k = k.strip()
        if k and k not in e.bib["cited"]:
            e.bib["cited"].append(k)


@cmd("bibliographystyle")
def bibliographystyle(e, t):
    e.bib["style"] = e.arg_text("bibliographystyle")


@cmd("bibliography", "printbibliography")
def bibliography(e, t):
    if t.v == "printbibliography":
        e.read_opt()
        files = e.bib["files"]
    else:
        files = [f.strip() for f in e.arg_text("bibliography").split(",") if f.strip()]
    from .bib import load
    load(e, files, t)
    e.add_block({"t": "bibliography", "line": t.line})


@cmd("bibitem")
def bibitem(e, t):
    lab = e.read_opt()
    key = e.arg_text("bibitem")
    e.end_par()
    e.manual_bib = getattr(e, "manual_bib", [])
    e.manual_bib.append(key)
    if key not in e.bib["entries"]:
        e.bib["entries"][key] = {"key": key, "manual": True, "label": detokenize(lab) if lab else None}
    e.add_inline({"t": "bibitem", "key": key})


# the theorem-related \qed, \proofname etc. are read by features.py
PREAMBLE_OK.update({"lstset", "pgfplotsset", "tikzset"})

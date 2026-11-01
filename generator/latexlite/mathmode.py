"""Math mode: checking (as LaTeX would: commands need their packages, & only in alignments, no double scripts,
balanced braces and \\left...\\right), a canonical form for grading (equivalent markup compares equal) and the
source KaTeX typesets on the page."""

from __future__ import annotations

from .tokens import Tok

# ---------------------------------------------------------------- what exists, and where it comes from

_W = str.split
GREEK = set(_W("""alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi
rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi
Omega"""))
CORE_SYMBOLS = set(_W("""le leq ge geq ne neq equiv sim simeq approx cong propto ll gg subset subseteq supset supseteq
in ni notin mid parallel perp models prec succ preceq succeq doteq bowtie vdash dashv asymp smile frown
pm mp times div cdot ast star circ bullet oplus ominus otimes oslash odot cap cup wedge vee land lor setminus wr
diamond bigtriangleup bigtriangledown triangleleft triangleright amalg dagger ddagger sqcap sqcup uplus
to rightarrow leftarrow gets Rightarrow Leftarrow leftrightarrow Leftrightarrow mapsto longrightarrow longleftarrow
Longrightarrow Longleftarrow longleftrightarrow Longleftrightarrow longmapsto uparrow downarrow Uparrow Downarrow
updownarrow Updownarrow nearrow searrow swarrow nwarrow hookrightarrow hookleftarrow rightharpoonup rightharpoondown
leftharpoonup leftharpoondown rightleftharpoons iff
infty partial nabla forall exists neg lnot emptyset top bot ell prime hbar imath jmath wp Re Im aleph angle triangle
backslash surd flat natural sharp clubsuit diamondsuit heartsuit spadesuit
ldots cdots vdots ddots dots dotsc dotsb dotsm dotsi dotso
sum prod coprod int oint bigcup bigcap bigvee bigwedge bigoplus bigotimes bigodot biguplus bigsqcup
arccos arcsin arctan arg cos cosh cot coth csc deg det dim exp gcd hom inf ker lg lim liminf limsup ln log max min Pr
sec sin sinh sup tan tanh
langle rangle lceil rceil lfloor rfloor vert Vert lbrace rbrace lbrack rbrack
quad qquad displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits not bmod pmod
colon cdotp ldotp vdotp
nonumber label"""))
CORE_ONE_ARG = set(_W("""hat bar tilde vec dot ddot check breve acute grave mathring widehat widetilde overline underline
overbrace underbrace overrightarrow overleftarrow mathrm mathbf mathit mathsf mathtt mathcal mathnormal sqrt mbox
textbf textit textrm texttt textsf textup textmd textnormal emph hbox phantom hphantom vphantom ensuremath
underline fbox boldmath operatornamewithlimits"""))
CORE_TWO_ARG = {"frac", "stackrel", "genfrac"}
AMSMATH = set(_W("""implies impliedby dfrac tfrac binom dbinom tbinom overset underset xrightarrow xleftarrow text
operatorname boxed substack lVert rVert lvert rvert iint iiint iiiint idotsint mod tag notag intertext
shortintertext dotsc boldsymbol pmb sideset leftroot uproot varGamma varDelta varTheta varLambda varXi varPi
varSigma varUpsilon varPhi varPsi varOmega eqref cfrac smash overleftrightarrow underleftarrow underrightarrow
underleftrightarrow"""))
AMSSYMB = set(_W("""mathbb mathfrak varnothing leqslant geqslant lesssim gtrsim nleq ngeq nless ngtr subsetneq supsetneq
nsubseteq nsupseteq triangleq therefore because square blacksquare checkmark circledast boxplus boxtimes boxminus
boxdot ltimes rtimes complement nexists varkappa digamma lll ggg Box Diamond lozenge blacklozenge
measuredangle sphericalangle smallsetminus restriction upharpoonright leftleftarrows rightrightarrows
twoheadrightarrow twoheadleftarrow rightsquigarrow leadsto ulcorner urcorner llcorner lrcorner dashrightarrow
dashleftarrow eqcirc circeq thicksim thickapprox approxeq backsim backsimeq vDash Vdash Vvdash nmid nparallel
ncong nsim npreceq nsucceq ntriangleleft ntriangleright varpropto beth gimel daleth hslash mho Bbbk eth
curlyvee curlywedge barwedge veebar doublebarwedge intercal divideontimes dotplus centerdot lessdot gtrdot
leqq geqq lneq gneq lneqq gneqq lnsim gnsim lessgtr gtrless trianglelefteq trianglerighteq"""))
MATHTOOLS = set(_W("""coloneqq eqqcolon Coloneqq coloneq eqcolon xmapsto xhookrightarrow xhookleftarrow prescript
mathclap mathllap mathrlap shortvdotswithin MoveEqLeft adjustlimits"""))
BM = {"bm"}
MATH_SPACING = {",", ":", ";", "!", " ", "quad", "qquad", "enspace", "thinspace", "medspace", "thickspace",
                "negthinspace", "hspace", "hfill", "\\"}
DELIMS = {"left", "right", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr", "big", "Big", "bigg",
          "Bigg", "middle", "bigm", "Bigm"}
SIZE_CMDS = {"tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge"}

# arguments: name -> (mandatory, optional first)
ARITY = {n: (1, False) for n in CORE_ONE_ARG}
ARITY.update({n: (2, False) for n in CORE_TWO_ARG})
ARITY.update({"sqrt": (1, True), "frac": (2, False), "dfrac": (2, False), "tfrac": (2, False), "binom": (2, False),
              "dbinom": (2, False), "tbinom": (2, False), "overset": (2, False), "underset": (2, False),
              "xrightarrow": (1, True), "xleftarrow": (1, True), "text": (1, False), "operatorname": (1, False),
              "boxed": (1, False), "substack": (1, False), "boldsymbol": (1, False), "bm": (1, False),
              "mathbb": (1, False), "mathfrak": (1, False), "tag": (1, False), "label": (1, False),
              "pmod": (1, False), "mod": (1, False), "cfrac": (2, False), "pmb": (1, False), "hspace": (1, False),
              "textcolor": (2, False), "color": (1, False), "colorbox": (2, False), "mathclap": (1, False),
              "mathllap": (1, False), "mathrlap": (1, False), "xmapsto": (1, True), "smash": (1, True),
              "genfrac": (6, False), "intertext": (1, False), "shortintertext": (1, False), "eqref": (1, False),
              "overleftrightarrow": (1, False), "underleftarrow": (1, False), "underrightarrow": (1, False),
              "operatornamewithlimits": (1, False), "mathscr": (1, False), "mathds": (1, False),
              "cancel": (1, False)})
TEXT_ARGS = {"text", "textbf", "textit", "textrm", "texttt", "textsf", "textup", "textmd", "textnormal", "emph",
             "mbox", "hbox", "intertext", "shortintertext", "fbox", "operatorname", "tag"}

SYNONYMS = {"to": "rightarrow", "gets": "leftarrow", "le": "leq", "ge": "geq", "ne": "neq", "lnot": "neg",
            "land": "wedge", "lor": "vee", "|": "Vert", "lbrace": "{", "rbrace": "}", "lbrack": "[",
            "rbrack": "]", "vert": "|", "dotsc": "ldots", "dotso": "ldots", "dotsb": "cdots", "dotsm": "cdots",
            "dotsi": "cdots", "colon": ":", "bm": "boldsymbol", "iff": "Longleftrightarrow~",
            "notag": "nonumber", "mathnormal": "mathit"}
ALIGN_ENVS = {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "cases", "dcases",
              "rcases", "array", "aligned", "alignedat", "gathered", "split", "subarray", "pmatrix*", "bmatrix*",
              "align", "align*", "alignat", "alignat*", "flalign", "flalign*", "eqnarray", "eqnarray*",
              "gather", "gather*", "multline", "multline*", "CD"}
INNER_ENV_PKG = {"matrix": "amsmath", "pmatrix": "amsmath", "bmatrix": "amsmath", "Bmatrix": "amsmath",
                 "vmatrix": "amsmath", "Vmatrix": "amsmath", "smallmatrix": "amsmath", "cases": "amsmath",
                 "aligned": "amsmath", "alignedat": "amsmath", "gathered": "amsmath", "split": "amsmath",
                 "subarray": "amsmath", "dcases": "mathtools", "rcases": "mathtools", "pmatrix*": "mathtools",
                 "bmatrix*": "mathtools", "array": None}


def math_available(name: str, packages: set, user: set) -> str | None:
    """None if `\\name` can be used in math with these packages (and user macros); else the package it needs (or
    "" when nothing provides it)."""
    if name in user:
        return None
    if len(name) == 1 and not name.isalpha():
        return None
    if name in GREEK or name in CORE_SYMBOLS or name in CORE_ONE_ARG or name in CORE_TWO_ARG or name in DELIMS \
            or name in MATH_SPACING or name in SIZE_CMDS or name in ("begin", "end", "left", "right", "sqrt",
                                                                    "textcolor", "color", "colorbox"):
        if name in ("textcolor", "color", "colorbox") and not ({"xcolor", "color"} & packages):
            return "xcolor"
        return None
    if name in AMSMATH:
        return None if "amsmath" in packages else "amsmath"
    if name in AMSSYMB:
        return None if ({"amssymb", "amsfonts"} & packages and name in ("mathbb", "mathfrak")) or \
            "amssymb" in packages else "amssymb"
    if name in MATHTOOLS:
        return None if "mathtools" in packages else "mathtools"
    if name in BM:
        return None if "bm" in packages else "bm"
    if name == "mathscr":
        return None if "mathrsfs" in packages else "mathrsfs"
    if name == "cancel":
        return None if "cancel" in packages else "cancel"
    if name in ("text", "tag"):
        return "amsmath"
    return ""


# ---------------------------------------------------------------- the math tree

class MathError(Exception):
    def __init__(self, msg, tok=None, help_=""):
        super().__init__(msg)
        self.msg, self.tok, self.help = msg, tok, help_


def check(toks: list[Tok], packages: set, user: set, env: str | None = None, display: bool = False):
    """Problems in math tokens, as (message, token, help) in the order LaTeX would find them."""
    errs = []
    stack = []          # open groups / environments / \left
    align_depth = 1 if env in ALIGN_ENVS else 0
    last_script = {}    # depth -> (atom index, '^'/'_' seen)
    i = 0
    n = len(toks)
    atoms: dict = {}           # atoms seen at each brace depth
    scripts = {}
    script_next = False
    while i < n:
        t = toks[i]
        depth = len(stack)
        if script_next and t.k != "sp":
            script_next = False
            if t.k == "bg":
                stack.append(("{", t, "script"))
                i += 1
                continue
            if t.k == "cs":
                need = math_available(t.v, packages, user)
                if need is not None and t.v not in ("begin", "end", "left", "right"):
                    errs.append(("Undefined control sequence.", t,
                                 f"\\{t.v} comes from the {need} package: \\usepackage{{{need}}}." if need else
                                 f"There's no command \\{t.v}."))
            i += 1
            continue
        if t.k == "bg":
            stack.append(("{", t))
        elif t.k == "eg":
            if stack and stack[-1][0] == "{" and len(stack[-1]) > 2:
                stack.pop()          # the end of a script's argument: not a new atom
                i += 1
                continue
            if not stack or stack[-1][0] != "{":
                if stack and stack[-1][0] == "left":
                    errs.append(("Missing \\right. inserted.", t, "A \\left needs a \\right before the group "
                                                                "closes."))
                    stack.pop()
                    continue
                errs.append(("Extra }, or forgotten $.", t, "There's a `}` without a matching `{`."))
            else:
                stack.pop()
                atoms[len(stack)] = atoms.get(len(stack), 0) + 1     # a {group} is one atom
        elif t.k == "tab":
            if not any(s[0] == "env" and s[2] for s in stack) and not align_depth:
                errs.append(("Misplaced alignment tab character &.", t,
                             "`&` only works in alignments (align, matrices, cases, tabular). Write \\& for an "
                             "ampersand."))
        elif t.k in ("sup", "sub"):
            key = (depth, t.k)
            prev = scripts.get(key)
            if prev is not None and prev == atoms.get(depth, 0):
                errs.append(("Double superscript." if t.k == "sup" else "Double subscript.", t,
                             "Use braces: x^{a^b} or {x^a}^b, x_{ij}."))
            scripts[key] = atoms.get(depth, 0)
            j = i + 1
            while j < n and toks[j].k == "sp":
                j += 1
            if j >= n or toks[j].k in ("eg",):
                errs.append(("Missing { inserted." if j < n else "Missing $ inserted.", t, "A ^ or _ needs "
                                                                                          "something after it."))
            script_next = True
            i += 1
            continue
        elif t.k == "cs":
            name = t.v
            if name == "begin":
                envname, j = _read_name(toks, i + 1)
                if envname is None:
                    errs.append(("Missing \\begin argument.", t, ""))
                else:
                    need = INNER_ENV_PKG.get(envname, "?")
                    if need == "?" and envname not in ALIGN_ENVS:
                        errs.append((f"LaTeX Error: Environment {envname} undefined.", t,
                                     f"No environment called {envname} (check the spelling, or load its "
                                     f"package)."))
                    elif need not in (None, "?") and need not in packages and not (need == "amsmath" and
                                                                                   "mathtools" in packages):
                        errs.append((f"LaTeX Error: Environment {envname} undefined.", t,
                                     f"{envname} comes from the {need} package: \\usepackage{{{need}}}."))
                    stack.append(("env", envname, envname in ALIGN_ENVS))
                    i = j
                    if envname == "array" or envname == "alignedat" or envname == "subarray":
                        # the column spec / count argument
                        _, i = _read_group(toks, i)
                    continue
            elif name == "end":
                envname, j = _read_name(toks, i + 1)
                if stack and stack[-1][0] == "env" and stack[-1][1] == envname:
                    stack.pop()
                elif stack and stack[-1][0] == "env":
                    errs.append((f"LaTeX Error: \\begin{{{stack[-1][1]}}} ended by \\end{{{envname}}}.", t,
                                 "The environments are closed in the wrong order."))
                    stack.pop()
                else:
                    errs.append(("LaTeX Error: Bad math environment delimiter.", t,
                                 f"\\end{{{envname}}} without a matching \\begin."))
                i = j
                continue
            elif name == "left":
                stack.append(("left", t))
            elif name == "right":
                if stack and stack[-1][0] == "left":
                    stack.pop()
                else:
                    errs.append(("Extra \\right.", t, "A \\right needs a \\left before it."))
            elif name == "\\":
                pass
            else:
                need = math_available(name, packages, user)
                if need is not None:
                    hint = f"\\{name} comes from the {need} package: \\usepackage{{{need}}}." if need else \
                        f"There's no command \\{name} (check the spelling, or define it with \\newcommand)."
                    errs.append(("Undefined control sequence.", t, hint))
            atoms[depth] = atoms.get(depth, 0) + 1
        elif t.k not in ("sp",):
            atoms[depth] = atoms.get(depth, 0) + 1
        i += 1
    for s in stack:
        if s[0] == "{":
            errs.append(("Missing } inserted.", s[1], "A `{` was never closed."))
        elif s[0] == "left":
            errs.append(("Missing \\right. inserted.", s[1], "A \\left needs a \\right."))
        elif s[0] == "env":
            errs.append((f"LaTeX Error: \\begin{{{s[1]}}} ended by \\end{{...}}.", None, f"\\begin{{{s[1]}}} is "
                                                                                       f"never closed."))
    return errs


def _read_name(toks, i):
    """{name} -> (name, index after)."""
    while i < len(toks) and toks[i].k == "sp":
        i += 1
    if i < len(toks) and toks[i].k == "bg":
        j = i + 1
        name = []
        while j < len(toks) and toks[j].k != "eg":
            name.append(toks[j].text())
            j += 1
        return "".join(name).strip(), j + 1
    return None, i


def _read_group(toks, i):
    while i < len(toks) and toks[i].k == "sp":
        i += 1
    if i < len(toks) and toks[i].k == "bg":
        depth = 0
        j = i
        while j < len(toks):
            if toks[j].k == "bg":
                depth += 1
            elif toks[j].k == "eg":
                depth -= 1
                if depth == 0:
                    return toks[i + 1:j], j + 1
            j += 1
    return [], i


# ---------------------------------------------------------------- canonical form (grading)

def canonical(toks: list[Tok]) -> str:
    """A string that equivalent math markup shares: spaces dropped, braces around a single atom dropped (x^{2} =
    x^2, \\frac{1}{2} = \\frac12), _ before ^, synonyms (\\le = \\leq, \\to = \\rightarrow)."""
    tree, _ = _parse(toks, 0, None)
    return _canon_seq(tree)


def _parse(toks, i, stop):
    """-> (list of nodes, index). Node: ('a', text) atom | ('g', [nodes]) group | ('c', name, [args], opt)
    | ('s', base, sub, sup) | ('env', name, [nodes]) | ('lr', l, [nodes], r)."""
    out = []
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.k == "sp":
            i += 1
            continue
        if t.k == "eg":
            if stop == "}":
                return out, i + 1
            i += 1
            continue
        if t.k == "bg":
            g, i = _parse(toks, i + 1, "}")
            out.append(("g", g))
            continue
        if t.k in ("sup", "sub"):
            arg, i = _one(toks, i + 1)
            base = out.pop() if out else ("a", "")
            if base[0] == "s":
                _, b, sb, sp = base
            else:
                b, sb, sp = base, None, None
            if t.k == "sub":
                sb = arg
            else:
                sp = arg
            out.append(("s", b, sb, sp))
            continue
        if t.k == "cs":
            name = SYNONYMS.get(t.v, t.v)
            if name == "end" and stop and stop.startswith("env:"):
                _, i = _read_name(toks, i + 1)
                return out, i
            if name == "right" and stop == "right":
                d, i = _delim(toks, i + 1)
                return out + [("rd", d)], i
            if name == "begin":
                envname, i = _read_name(toks, i + 1)
                pre = []
                if envname in ("array", "alignedat", "subarray"):
                    g, i = _read_group(toks, i)
                    pre = [("a", "".join(x.text() for x in g).replace(" ", ""))]
                body, i = _parse(toks, i, "env:" + str(envname))
                out.append(("env", envname, pre + body))
                continue
            if name == "left":
                d, i = _delim(toks, i + 1)
                body, i = _parse(toks, i, "right")
                rd = body.pop()[1] if body and body[-1][0] == "rd" else "."
                out.append(("lr", d, body, rd))
                continue
            if name in DELIMS:
                d, i = _delim(toks, i + 1)
                out.append(("c", name, [("a", d)], None))
                continue
            i += 1
            m, opt = ARITY.get(name, (0, False))
            o = None
            if opt:
                j = i
                while j < n and toks[j].k == "sp":
                    j += 1
                if j < n and toks[j].k == "ch" and toks[j].v == "[":
                    k = j + 1
                    depth = 0
                    while k < n and not (toks[k].k == "ch" and toks[k].v == "]" and depth == 0):
                        depth += toks[k].k == "bg"
                        depth -= toks[k].k == "eg"
                        k += 1
                    o, _ = _parse(toks[j + 1:k], 0, None)
                    i = k + 1
            args = []
            for _ in range(m):
                if name in TEXT_ARGS:
                    g, i = _read_group(toks, i) if _peek(toks, i) == "bg" else ([toks[i]] if i < n else [], i + 1)
                    args.append(("t", " ".join("".join(x.text() for x in g).split())))
                else:
                    a, i = _one(toks, i)
                    args.append(a)
            out.append(("c", name, args, o))
            continue
        if t.k == "ch" and t.v == "'":
            # x' = x^\prime
            j = i
            primes = 0
            while j < n and toks[j].k == "ch" and toks[j].v == "'":
                primes += 1
                j += 1
            base = out.pop() if out else ("a", "")
            b, sb, sp = (base[1], base[2], base[3]) if base[0] == "s" else (base, None, None)
            sp = ("g", [("c", "prime", [], None)] * primes)
            out.append(("s", b, sb, sp))
            i = j
            continue
        if t.k == "tab":
            out.append(("a", "&"))
            i += 1
            continue
        out.append(("a", t.text()))
        i += 1
    return out, i


def _peek(toks, i):
    while i < len(toks) and toks[i].k == "sp":
        i += 1
    return toks[i].k if i < len(toks) else None


def _one(toks, i):
    """One argument: a group or a single token (atom / command without args)."""
    n = len(toks)
    while i < n and toks[i].k == "sp":
        i += 1
    if i >= n:
        return ("g", []), i
    t = toks[i]
    if t.k == "bg":
        g, i = _parse(toks, i + 1, "}")
        return ("g", g), i
    if t.k == "cs":
        name = SYNONYMS.get(t.v, t.v)
        m, _ = ARITY.get(name, (0, False))
        if m == 0:
            return ("c", name, [], None), i + 1
        return _parse_one_cmd(toks, i)       # a command with arguments as a script (x^\frac12)
    return ("a", t.text()), i + 1


def _parse_one_cmd(toks, i):
    t = toks[i]
    name = SYNONYMS.get(t.v, t.v)
    m, _ = ARITY.get(name, (0, False))
    i += 1
    args = []
    for _ in range(m):
        a, i = _one(toks, i)
        args.append(a)
    return ("c", name, args, None), i


def _delim(toks, i):
    n = len(toks)
    while i < n and toks[i].k == "sp":
        i += 1
    if i >= n:
        return ".", i
    t = toks[i]
    if t.k == "cs":
        return "\\" + SYNONYMS.get(t.v, t.v), i + 1
    return t.text(), i + 1


def _canon(node) -> str:
    k = node[0]
    if k == "a":
        return node[1]
    if k == "g":
        inner = node[1]
        if len(inner) == 1:
            return _canon(inner[0])
        return "{" + _canon_seq(inner) + "}"
    if k == "c":
        _, name, args, opt = node
        s = "\\" + name
        if opt is not None:
            s += "[" + _canon_seq(opt) + "]"
        for a in args:
            if a[0] == "t":
                s += "{" + a[1] + "}"
            else:
                s += "{" + _canon_arg(a) + "}"
        return s
    if k == "s":
        _, b, sb, sp = node
        s = _canon(b)
        if sb is not None:
            s += "_{" + _canon_arg(sb) + "}"
        if sp is not None:
            s += "^{" + _canon_arg(sp) + "}"
        return s
    if k == "env":
        return "\\begin{" + str(node[1]) + "}" + _canon_seq(node[2]) + "\\end{" + str(node[1]) + "}"
    if k == "lr":
        return "\\left" + node[1] + _canon_seq(node[2]) + "\\right" + node[3]
    if k == "rd":
        return ""
    return ""


def _canon_arg(node) -> str:
    """An argument's contents: {x} and x are the same argument."""
    if node[0] == "g":
        return _canon_seq(node[1])
    return _canon(node)


def _canon_seq(nodes) -> str:
    out = []
    for nd in nodes:
        c = _canon(nd)
        if out and out[-1][-1:].isalpha() and out[-1].startswith("\\") and c[:1].isalpha():
            out.append(" ")
        out.append(c)
    return "".join(out)


# ---------------------------------------------------------------- KaTeX source

def to_katex(toks: list[Tok]) -> str:
    """The math as KaTeX reads it: labels dropped (latex-lite resolves them), the rest as written."""
    out = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.k == "cs" and t.v in ("label", "nonumber", "notag"):
            if t.v == "label":
                _, i = _read_group(toks, i + 1)
            else:
                i += 1
            continue
        if t.k == "cs" and t.v == "mathds":
            out.append("\\mathbb")
            i += 1
            continue
        out.append(t.text())
        if t.k == "cs" and t.v[:1].isalpha() and i + 1 < n and toks[i + 1].k == "ch" and toks[i + 1].v[:1].isalpha():
            out.append(" ")
        i += 1
    return "".join(out).strip()

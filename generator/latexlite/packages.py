"""What each document class and package provides (so a command works only when its package is loaded, as in real
LaTeX), and which packages exist at all (an unknown one is "File `x.sty' not found")."""

from __future__ import annotations

# ---------------------------------------------------------------- text-mode commands and environments

CORE_TEXT = set("""
documentclass usepackage RequirePackage begin end par item newline linebreak nolinebreak newpage clearpage
cleardoublepage pagebreak nopagebreak noindent indent centering raggedright raggedleft
textbf textit emph underline texttt textsc textsf textrm textup textmd textsl textnormal
bfseries itshape ttfamily sffamily scshape rmfamily mdseries upshape slshape normalfont em bf it tt sf rm sc sl
tiny scriptsize footnotesize small normalsize large Large LARGE huge Huge
textbackslash textasciitilde textasciicircum textbar textless textgreater textbullet textendash textemdash
ldots dots textellipsis LaTeX TeX LaTeXe today
quad qquad hspace vspace hfill vfill smallskip medskip bigskip enspace thinspace negthinspace hskip vskip
footnote footnotemark footnotetext thanks
title author date and maketitle tableofcontents listoffigures listoftables
part section subsection subsubsection paragraph subparagraph appendix
label ref pageref cite nocite bibliography bibliographystyle
caption
newcommand renewcommand providecommand newenvironment renewenvironment def let newtheorem
newcounter setcounter addtocounter stepcounter refstepcounter value arabic alph Alph roman Roman fnsymbol the
setlength addtolength parskip parindent baselineskip textwidth linewidth columnwidth textheight paperwidth
baselinestretch renewcommand
input include includeonly
mbox fbox makebox framebox parbox raisebox rule
verb
pagestyle thispagestyle pagenumbering
markboth markright
protect relax ignorespaces
string expandafter csname endcsname
P S copyright dag ddag pounds slash
i j l L o O oe OE ae AE aa AA ss
textquoteleft textquoteright textquotedblleft textquotedblright guillemotleft guillemotright
hrule hrulefill dotfill
twocolumn onecolumn
addcontentsline addtocontents
frac sqrt
ensuremath
bibitem hline cline multicolumn
""".split()) | set("% $ & # _ { } , ; : ! \\ ` ' ^ ~ \" = . - / @ ( ) [ ] |".split()) | {" ", "\n"}

CORE_ENVS = set("""
document center flushleft flushright quote quotation verse itemize enumerate description
figure figure* table table* tabular tabular* array minipage verbatim verbatim* abstract
equation displaymath math eqnarray eqnarray* thebibliography titlepage list trivlist comment
""".split())

CLASS_TEXT = {
    "article": set(),
    "report": {"chapter"},
    "book": {"chapter", "frontmatter", "mainmatter", "backmatter"},
    "memoir": {"chapter", "frontmatter", "mainmatter", "backmatter"},
    "beamer": set("""frame frametitle framesubtitle titlepage pause onslide only uncover visible invisible alert
                     structure usetheme usecolortheme usefonttheme useinnertheme useoutertheme setbeamertemplate
                     setbeamercolor setbeamerfont institute subtitle logo titlegraphic column againframe
                     AtBeginSection AtBeginSubsection insertsectionhead inserttitle insertframenumber
                     insertauthor beamertemplatenavigationsymbolsempty note""".split()),
}
CLASS_ENVS = {
    "beamer": {"frame", "block", "alertblock", "exampleblock", "columns", "column", "onlyenv", "overlayarea",
               "itemize", "enumerate", "theorem", "definition", "lemma", "proof", "example", "corollary"},
}
CLASSES = {"article", "report", "book", "beamer", "memoir", "letter", "scrartcl", "scrreprt", "scrbook", "amsart",
           "standalone", "minimal"}

# ---------------------------------------------------------------- packages: text commands, environments, math

PKG = {
    "amsmath": {"text": {"eqref", "numberwithin", "DeclareMathOperator", "intertext", "shortintertext", "tag",
                         "notag", "nonumber", "allowdisplaybreaks", "text", "boxed", "substack"},
                "envs": {"align", "align*", "gather", "gather*", "multline", "multline*", "flalign", "flalign*",
                         "alignat", "alignat*", "equation*", "split", "aligned", "gathered", "alignedat", "cases",
                         "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
                         "subequations", "subarray"}},
    "amssymb": {"loads": ["amsfonts"]},
    "amsfonts": {},
    "amsthm": {"text": {"theoremstyle", "qedhere", "qed", "qedsymbol", "proofname", "newtheorem*"},
               "envs": {"proof"}},
    "mathtools": {"loads": ["amsmath"], "text": {"DeclarePairedDelimiter", "mathtoolsset"},
                  "envs": {"dcases", "dcases*", "rcases", "pmatrix*", "bmatrix*"}},
    "bm": {},
    "geometry": {"text": {"geometry", "newgeometry", "restoregeometry"}},
    "graphicx": {"loads": ["graphics"], "text": {"includegraphics", "graphicspath", "rotatebox", "scalebox",
                                                 "resizebox", "reflectbox"}},
    "graphics": {"text": {"includegraphics"}},
    "booktabs": {"text": {"toprule", "midrule", "bottomrule", "cmidrule", "addlinespace", "specialrule"}},
    "multirow": {"text": {"multirow"}},
    "siunitx": {"text": {"SI", "si", "num", "qty", "unit", "sisetup", "ang", "tablenum"}, "colspec": {"S"}},
    "tabularx": {"envs": {"tabularx"}, "loads": ["array"], "colspec": {"X"}},
    "array": {"text": {"newcolumntype"}},
    "longtable": {"envs": {"longtable"}},
    "hyperref": {"text": {"href", "url", "autoref", "hypersetup", "nameref", "hyperlink", "hypertarget",
                          "texorpdfstring", "phantomsection", "Autoref"}, "loads": ["url"]},
    "url": {"text": {"url", "path", "urlstyle"}},
    "cleveref": {"text": {"cref", "Cref", "crefname", "Crefname", "crefrange", "Crefrange", "labelcref"}},
    "natbib": {"text": {"citet", "citep", "citealt", "citealp", "citeauthor", "citeyear", "citenum", "Citet",
                        "Citep", "setcitestyle", "bibpunct"}},
    "biblatex": {"text": {"addbibresource", "printbibliography", "textcite", "parencite", "autocite",
                          "footcite", "citeauthor", "citeyear", "Textcite", "Parencite"}},
    "subcaption": {"loads": ["caption"], "envs": {"subfigure", "subtable"}, "text": {"subcaption", "subcaptionbox"}},
    "caption": {"text": {"captionsetup", "captionof"}},
    "float": {"text": {"floatstyle", "newfloat", "restylefloat", "floatname"}},
    "placeins": {"text": {"FloatBarrier"}},
    "wrapfig": {"envs": {"wrapfigure", "wraptable"}},
    "algorithm": {"envs": {"algorithm"}, "loads": ["float"], "text": {"listofalgorithms"}},
    "algpseudocode": {"loads": ["algorithmicx"], "envs": {"algorithmic"},
                      "text": set("""State Statex Require Ensure Input Output For ForAll EndFor While EndWhile If
                                     ElsIf Else EndIf Repeat Until Loop EndLoop Function EndFunction Procedure
                                     EndProcedure Call Return Comment algrenewcommand algnewcommand""".split())},
    "algorithmicx": {},
    "algorithm2e": {"envs": {"algorithm"}},
    "listings": {"envs": {"lstlisting"}, "text": {"lstinline", "lstset", "lstinputlisting", "lstdefinestyle",
                                                    "lstlistoflistings"}},
    "minted": {"envs": {"minted"}, "text": {"mintinline", "inputminted", "setminted"}},
    "xcolor": {"loads": ["color"], "text": {"textcolor", "color", "colorbox", "fcolorbox", "definecolor",
                                             "pagecolor", "colorlet"}},
    "color": {"text": {"textcolor", "color", "colorbox", "fcolorbox", "definecolor", "pagecolor"}},
    "tcolorbox": {"envs": {"tcolorbox"}, "text": {"tcbuselibrary", "newtcolorbox", "tcbset", "tcbox"},
                  "loads": ["xcolor"]},
    "fancyhdr": {"text": {"fancyhf", "lhead", "chead", "rhead", "lfoot", "cfoot", "rfoot", "fancyhead",
                          "fancyfoot", "headrulewidth", "footrulewidth", "fancypagestyle"}},
    "setspace": {"text": {"onehalfspacing", "doublespacing", "singlespacing", "setstretch"},
                 "envs": {"spacing", "singlespace", "onehalfspace", "doublespace"}},
    "multicol": {"envs": {"multicols", "multicols*"}, "text": {"columnbreak"}},
    "enumitem": {"text": {"setlist", "newlist", "setlistdepth"}},
    "tikz": {"loads": ["pgf", "xcolor"], "envs": {"tikzpicture", "scope"},
             "text": {"tikz", "usetikzlibrary", "tikzset", "draw", "node", "fill", "filldraw", "path", "foreach",
                      "coordinate", "shade", "clip", "pgfmathsetmacro", "tikzstyle"}},
    "pgf": {},
    "pgfplots": {"loads": ["tikz"], "envs": {"axis", "semilogyaxis", "semilogxaxis", "loglogaxis"},
                 "text": {"pgfplotsset", "addplot", "addlegendentry", "legend"}},
    "lipsum": {"text": {"lipsum"}},
    "blindtext": {"text": {"blindtext", "Blindtext"}},
    "mwe": {"loads": ["graphicx", "lipsum", "blindtext"]},
    "inputenc": {}, "fontenc": {}, "lmodern": {}, "babel": {}, "microtype": {}, "csquotes": {"text": {"enquote"}},
    "textcomp": {}, "upgreek": {}, "xspace": {"text": {"xspace"}}, "parskip": {}, "titlesec": {"text": {"titleformat", "titlespacing"}},
    "fontspec": {"text": {"setmainfont", "setsansfont", "setmonofont"}}, "ragged2e": {"text": {"justifying", "RaggedRight"}},
    "verbatim": {"envs": {"comment"}}, "fancyvrb": {"envs": {"Verbatim"}}, "makeidx": {}, "nicefrac": {"text": {"nicefrac"}},
    "xparse": {"text": {"NewDocumentCommand"}}, "etoolbox": {}, "calc": {}, "ifthen": {}, "soul": {"text": {"hl", "st", "ul"}},
    "ulem": {"text": {"uline", "sout", "uwave"}}, "changepage": {}, "afterpage": {}, "pdfpages": {}, "standalone": {},
    "cite": {}, "abstract": {}, "appendix": {}, "titling": {}, "authblk": {"text": {"affil"}},
    "hyperxmp": {}, "bookmark": {}, "footmisc": {}, "threeparttable": {"envs": {"threeparttable", "tablenotes"}},
    "dcolumn": {}, "colortbl": {"text": {"rowcolor", "cellcolor", "columncolor"}}, "pdflscape": {"envs": {"landscape"}},
    "lscape": {"envs": {"landscape"}}, "rotating": {"envs": {"sidewaystable", "sidewaysfigure"}},
    "physics": {}, "esint": {}, "mathrsfs": {}, "dsfont": {}, "stmaryrd": {}, "cancel": {}, "tensor": {},
    "thmtools": {"text": {"declaretheorem"}}, "chngcntr": {"text": {"counterwithin", "counterwithout"}},
}

SECTION_LEVELS = {"part": -1, "chapter": 0, "section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4,
                  "subparagraph": 5}


def closure(packages) -> set:
    """The packages loaded, with the ones they load themselves."""
    out, todo = set(), list(packages)
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.add(p)
        todo += PKG.get(p, {}).get("loads", [])
    return out


def text_commands(cls: str, packages) -> set:
    s = set(CORE_TEXT) | CLASS_TEXT.get(cls, set())
    for p in closure(packages):
        s |= PKG.get(p, {}).get("text", set())
    if cls == "beamer":
        s |= {"url", "href", "textcolor", "color", "includegraphics", "hypersetup"} | PKG["xcolor"]["text"] | \
            PKG["graphicx"]["text"] | PKG["amsthm"]["text"]
    return s


def environments(cls: str, packages) -> set:
    s = set(CORE_ENVS) | CLASS_ENVS.get(cls, set())
    for p in closure(packages):
        s |= PKG.get(p, {}).get("envs", set())
    if cls == "beamer":
        s |= {"proof"} | PKG["amsthm"].get("envs", set())
    return s


def known_package(name: str) -> bool:
    return name in PKG


# where a command that isn't available comes from (for the hint under the error)
def provider(name: str, kind: str = "text") -> str | None:
    for p, d in PKG.items():
        if name in d.get("text" if kind == "text" else "envs", set()):
            return p
    return None

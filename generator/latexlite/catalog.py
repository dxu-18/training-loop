"""Every LaTeX command, environment and package COMP_GUIDE section 11 teaches, with a one-line description and the
module that introduces it. Keys are what latex-lite records as used: `\\textbf`, `env:itemize`, `pkg:booktabs`."""

_MODULES = {
    "11.01": {
        "\\documentclass": "the class: \\documentclass[11pt, a4paper]{article} (also report, book, beamer)",
        "\\usepackage": "load packages in the preamble: \\usepackage{amsmath, amssymb}",
        "\\title": "the title (preamble); printed by \\maketitle",
        "\\author": "the authors, separated by \\and",
        "\\date": "the date (\\today, or \\date{} for none)",
        "env:document": "the body: everything printed goes between \\begin{document} and \\end{document}",
        "\\maketitle": "print the title block",
        "env:abstract": "the abstract (small, indented, under the title)",
        "\\tableofcontents": "the table of contents (from the sections)",
        "\\and": "separates authors in \\author",
        "\\today": "today's date",
        "\\LaTeX": "the LaTeX logo",
        "pkg:geometry": "page size and margins: \\usepackage[margin=1in]{geometry}",
    },
    "11.02": {
        "\\section": "a numbered section (\\section*{...}: unnumbered, not in the contents)",
        "\\subsection": "a subsection (1.1)", "\\subsubsection": "a subsubsection (1.1.1)",
        "\\paragraph": "a run-in heading in bold: \\paragraph{Note.} text",
        "\\appendix": "after it, sections are numbered A, B, ...",
        "\\chapter": "a chapter (report and book classes only)",
        "\\label": "name a number: \\label{sec:intro} (after \\caption in floats)",
        "\\input": "paste a file here: \\input{sections/method}",
        "\\include": "a file on a new page", "\\newpage": "start a new page",
        "\\clearpage": "a new page, after placing pending figures and tables",
    },
    "11.03": {
        "\\textbf": "bold", "\\textit": "italic", "\\emph": "emphasis (italic, or upright inside italic)",
        "\\underline": "underline", "\\texttt": "monospace", "\\textsc": "small caps", "\\textsf": "sans serif",
        "\\small": "smaller text (in a group: {\\small ...})", "\\large": "larger text", "\\Large": "larger still",
        "\\huge": "huge text", "\\tiny": "tiny text", "\\footnotesize": "footnote-size text",
        "\\%": "a percent sign (% alone starts a comment)", "\\$": "a dollar sign", "\\&": "an ampersand",
        "\\#": "a hash sign", "\\_": "an underscore", "\\{": "a left brace", "\\}": "a right brace",
        "\\textbackslash": "a backslash", "\\textasciitilde": "a tilde ~",
        "\\textasciicircum": "a caret ^",
        "\\ldots": "an ellipsis …", "\\,": "a thin space", "\\quad": "a 1em space", "\\qquad": "a 2em space",
        "\\\\": "end the line (the paragraph continues)", "\\noindent": "this paragraph isn't indented",
        "\\hfill": "push what follows to the right", "\\vspace": "vertical space: \\vspace{1em}",
        "env:center": "center a block", "env:flushright": "right-align a block",
        "\\footnote": "a footnote: text.\\footnote{The note.}", "\\url": "a URL in monospace (url / hyperref)",
        "\\href": "a link with text: \\href{https://numpy.org}{NumPy} (hyperref)",
        "\\verb": "inline code, verbatim: \\verb|x = np.zeros(3)|", "env:verbatim": "a block of code, verbatim",
    },
    "11.04": {
        "env:itemize": "a bullet list: \\begin{itemize} \\item ... \\end{itemize}",
        "env:enumerate": "a numbered list", "env:description": "a list of terms: \\item[Term] definition",
        "\\item": "an entry of a list (\\item[label] for a custom label)",
        "pkg:enumitem": "list options: [label=(\\alph*), noitemsep, leftmargin=*]",
    },
    "11.05": {
        "$": "inline math: $f(x) = x^2$", "\\[": "display math: \\[ ... \\] (unnumbered)",
        "env:equation": "a numbered equation", "env:equation*": "an unnumbered equation (amsmath)",
        "\\eqref": "refer to an equation with its parentheses: (1) (amsmath)",
        "^": "superscript: x^2, x^{10}", "_": "subscript: x_i, x_{i,j}",
        "\\frac": "a fraction: \\frac{a}{b}", "\\dfrac": "a display-size fraction (amsmath)",
        "\\tfrac": "a text-size fraction (amsmath)", "\\sqrt": "a root: \\sqrt{x}, \\sqrt[3]{x}",
        "\\sum": "a sum: \\sum_{i=1}^{n}", "\\prod": "a product", "\\int": "an integral: \\int_0^\\infty f(x)\\,dx",
        "\\lim": "a limit: \\lim_{x \\to 0}", "\\max": "max (subscripts go under it in displays)",
        "\\partial": "∂ for partial derivatives", "\\nabla": "∇ for gradients",
        "\\text": "text inside math: $\\text{if } x > 0$ (amsmath)", "\\mathrm": "upright letters: \\mathrm{d}x",
        "\\operatorname": "an upright operator name: \\operatorname{softmax}(z) (amsmath)",
        "\\mathbb": "blackboard bold: \\mathbb{R}, \\mathbb{E} (amssymb)",
        "\\mathcal": "calligraphic: \\mathcal{L}, \\mathcal{D}", "\\mathbf": "bold letters: \\mathbf{W}",
        "\\boldsymbol": "bold symbols: \\boldsymbol{\\theta} (amsmath)", "\\hat": "a hat accent: \\hat{y}",
        "\\bar": "a bar accent: \\bar{x}", "\\tilde": "a tilde accent", "\\vec": "a vector arrow",
        "\\langle": "⟨ (with \\rangle ⟩)", "\\|": "‖ for norms: \\|x\\|_2", "\\top": "⊤ for transposes: x^\\top",
        "\\mid": "| as a relation: p(y \\mid x)", "pkg:amsmath": "align, equation*, \\text, \\eqref, matrices ...",
        "pkg:amssymb": "\\mathbb and many symbols",
    },
    "11.06": {
        "\\alpha": "α (and β γ δ ε ...)", "\\varepsilon": "ε (the curly epsilon)", "\\theta": "θ",
        "\\Delta": "Δ (capital Greek: \\Gamma \\Delta \\Theta ...)", "\\le": "≤", "\\ge": "≥", "\\ne": "≠",
        "\\approx": "≈", "\\equiv": "≡", "\\sim": "∼ (x \\sim p)", "\\propto": "∝", "\\pm": "±",
        "\\times": "×", "\\cdot": "·", "\\odot": "⊙ (Hadamard product)", "\\otimes": "⊗", "\\in": "∈",
        "\\notin": "∉", "\\subseteq": "⊆", "\\cup": "∪", "\\cap": "∩", "\\emptyset": "∅", "\\forall": "∀",
        "\\exists": "∃", "\\implies": "⟹ (amsmath)", "\\iff": "⟺", "\\to": "→", "\\gets": "←",
        "\\Rightarrow": "⇒", "\\mapsto": "↦", "\\infty": "∞", "\\ell": "ℓ", "\\cdots": "⋯", "\\vdots": "⋮",
        "\\ddots": "⋱", "\\log": "log (upright, like \\sin \\exp \\det ...)", "\\exp": "exp", "\\sin": "sin",
    },
    "11.07": {
        "env:align": "several numbered lines aligned at & (amsmath)", "env:align*": "aligned, unnumbered",
        "\\nonumber": "no number on this line of an align", "env:gather*": "centered lines, unnumbered",
        "env:gather": "centered lines, numbered", "env:multline*": "one long equation broken over lines",
        "env:split": "lines inside one equation (one number)", "&&": "&& \\text{...}: a note on the right",
    },
    "11.08": {
        "env:bmatrix": "a matrix in [ ] (amsmath)", "env:pmatrix": "a matrix in ( )",
        "env:vmatrix": "a determinant | |", "env:cases": "a piecewise definition (amsmath)",
        "\\left": "brackets that grow: \\left( \\frac{a}{b} \\right)", "\\right": "closes a \\left",
        "\\bigl": "a fixed bigger bracket: \\bigl( x \\bigr)", "\\underbrace": "a brace under: \\underbrace{...}_{n}",
        "\\overbrace": "a brace over", "\\overset": "a symbol over another: \\overset{\\text{def}}{=}",
        "\\xrightarrow": "an arrow with a label: \\xrightarrow{\\text{train}}",
        "\\DeclareMathOperator": "a new operator: \\DeclareMathOperator*{\\argmax}{arg\\,max} (preamble)",
        "\\newcommand": "a macro: \\newcommand{\\R}{\\mathbb{R}}, with arguments [1] ... #1",
        "\\lVert": "‖ opening a norm (\\rVert closes it)",
    },
    "11.09": {
        "\\softmax": "a declared operator (\\DeclareMathOperator{\\softmax}{softmax})",
        "\\sigma": "σ", "\\eta": "η (learning rate)", "\\beta": "β", "\\pi": "π",
        "\\!": "a negative thin space: \\softmax\\!\\left( ... \\right)",
    },
    "11.10": {
        "env:figure": "a floating figure: [htbp] placement", "\\includegraphics": "an image: "
        "\\includegraphics[width=0.8\\linewidth]{loss.pdf} (graphicx)", "\\caption": "the caption (and the number)",
        "\\centering": "center the float's contents", "env:subfigure": "a panel inside a figure (subcaption)",
        "pkg:graphicx": "\\includegraphics", "pkg:subcaption": "subfigure panels", "pkg:float": "[H]: exactly here",
        "\\ref": "the number of a label: Figure~\\ref{fig:loss}",
    },
    "11.11": {
        "env:table": "a floating table (caption ABOVE)", "env:tabular": "the grid: \\begin{tabular}{l c r}",
        "\\toprule": "the top rule (booktabs)", "\\midrule": "the rule under the header",
        "\\bottomrule": "the bottom rule", "\\hline": "a full rule (with | columns)",
        "\\multicolumn": "a cell across columns: \\multicolumn{2}{c}{Header}",
        "\\multirow": "a cell across rows (multirow)", "\\cmidrule": "a rule under some columns: \\cmidrule(lr){2-3}",
        "pkg:booktabs": "\\toprule \\midrule \\bottomrule", "pkg:siunitx": "S columns align numbers",
        "pkg:multirow": "\\multirow",
    },
    "11.12": {
        "\\pageref": "the page of a label", "\\cref": "the number with its name: fig. 1 (cleveref)",
        "\\Cref": "the same, capitalized: Figure 1", "\\cite": "a citation: \\cite{vaswani2017attention}",
        "\\citet": "a textual citation: Vaswani et al. (2017) (natbib)",
        "\\citep": "a parenthetical citation: (Vaswani et al., 2017) (natbib)",
        "\\bibliographystyle": "the reference style: plainnat, plain, unsrt ...",
        "\\bibliography": "print the references from references.bib",
        "pkg:hyperref": "clickable references and links", "pkg:cleveref": "\\cref and \\Cref (load it last)",
        "pkg:natbib": "\\citet and \\citep",
    },
    "11.13": {
        "env:algorithm": "a floating algorithm with its caption (algorithm)",
        "env:algorithmic": "the pseudocode: \\begin{algorithmic}[1] numbers every line (algpseudocode)",
        "\\Require": "the inputs", "\\Ensure": "the outputs", "\\State": "a statement",
        "\\For": "a loop: \\For{...} ... \\EndFor", "\\EndFor": "closes \\For",
        "\\If": "\\If{...} ... \\Else ... \\EndIf", "\\EndIf": "closes \\If", "\\Return": "return",
        "\\Comment": "a comment at the right: \\Comment{gradient step}", "\\While": "\\While{...} ... \\EndWhile",
        "env:lstlisting": "a code listing: [language=Python, numbers=left, frame=single] (listings)",
        "pkg:algpseudocode": "the algorithmic commands", "pkg:listings": "lstlisting",
    },
    "11.14": {
        "\\newtheorem": "a theorem environment: \\newtheorem{theorem}{Theorem}[section]",
        "\\theoremstyle": "plain (italic), definition (upright) or remark (amsthm)",
        "env:proof": "a proof ending with □ (amsthm)", "\\newenvironment": "your own environment",
        "pkg:amsthm": "\\theoremstyle and proof",
    },
    "11.15": {
        "\\pagestyle": "plain, empty or fancy (fancyhdr)", "\\fancyhf": "clear the header and footer",
        "\\lhead": "the header's left (\\rhead right, \\chead center)", "\\rhead": "the header's right",
        "\\cfoot": "the footer's center", "\\thepage": "the page number",
        "\\textcolor": "colored text: \\textcolor{red}{...} (xcolor)", "\\colorbox": "a colored background",
        "\\color": "switch color in a group: {\\color{red} ...}", "\\definecolor": "\\definecolor{brand}{HTML}{1F77B4}",
        "env:tcolorbox": "a framed colored box: [colback=blue!5, colframe=brand, title=Key idea]",
        "env:minipage": "a box of a given width: side-by-side columns",
        "\\setlength": "\\setlength{\\parskip}{4pt}", "\\onehalfspacing": "1.5 line spacing (setspace)",
        "\\thispagestyle": "this page's style only", "env:multicols": "columns of text (multicol)",
        "pkg:xcolor": "colors", "pkg:fancyhdr": "headers and footers", "pkg:tcolorbox": "boxes",
        "pkg:setspace": "line spacing",
    },
    "11.16": {
        "env:tikzpicture": "a TikZ drawing", "\\node": "\\node[circle, draw] (a) at (0,0) {$x$};",
        "\\draw": "lines and shapes: \\draw[->] (a) -- (b);", "\\fill": "a filled shape: \\fill[red] (0,0) circle (2pt);",
        "\\foreach": "repeat: \\foreach \\i in {1,2,3} ...", "\\usetikzlibrary": "TikZ libraries: positioning, arrows.meta",
        "env:axis": "a pgfplots plot", "\\addplot": "a curve: coordinates {...}, {expression} or table {file}",
        "\\addlegendentry": "the curve's legend entry", "pkg:tikz": "TikZ", "pkg:pgfplots": "plots",
        "\\pgfplotsset": "\\pgfplotsset{compat=1.18}",
    },
    "11.17": {
        "env:frame": "a slide: \\begin{frame}{Title} ... \\end{frame}", "\\titlepage": "the title slide",
        "\\usetheme": "the look: \\usetheme{Madrid}", "\\pause": "reveal what follows on the next click",
        "env:columns": "side-by-side columns (\\column{0.5\\textwidth})", "\\column": "starts a column",
        "env:block": "a titled box", "\\frametitle": "the frame's title",
        "\\only": "content shown on some overlays only", "\\alert": "highlight in red",
    },
    "11.18": {},
}

CATALOG: dict[str, str] = {}
MODULE_OF: dict[str, str] = {}
for _m, _d in _MODULES.items():
    for _k, _v in _d.items():
        if _k not in CATALOG:
            CATALOG[_k] = _v
            MODULE_OF[_k] = _m

TRIVIAL = {"env:document", "\\documentclass", "\\usepackage", "$", "^", "_", "\\item", "\\label", "\\\\",
           "\\begin", "\\end"}


def shown(key: str) -> str:
    if key.startswith("env:"):
        return f"\\begin{{{key[4:]}}}"
    if key.startswith("pkg:"):
        return f"\\usepackage{{{key[4:]}}}"
    return key

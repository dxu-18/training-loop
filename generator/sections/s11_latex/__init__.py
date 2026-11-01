"""COMP_GUIDE §11 LaTeX: one module per subsection (11.01-11.18), answered in an editor and compiled by latex-lite
(generator/latexlite): Compile shows the PDF it makes, Submit compares it with the reference's."""

from ...grader import _CHECKERS
from ...latexlite.check import checker as _checker

_CHECKERS["latex"] = _checker

from . import m01_skeleton  # noqa: E402,F401
from . import m02_structure, m03_text  # noqa: E402,F401
from . import m04_lists, m05_math, m06_symbols  # noqa: E402,F401
from . import m07_multiline, m08_matrices, m09_gallery  # noqa: E402,F401
from . import m10_figures, m11_tables, m12_refs  # noqa: E402,F401
from . import m13_algorithms, m14_theorems  # noqa: E402,F401
from . import m15_layout  # noqa: E402,F401
from . import m16_tikz  # noqa: E402,F401
from . import m17_beamer  # noqa: E402,F401
from . import m18_troubleshooting  # noqa: E402,F401

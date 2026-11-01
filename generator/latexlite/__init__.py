"""latex-lite: LaTeX interpreted in Python (it runs in the browser with Pyodide).

compile_doc() turns source into a document model (model.py) with LaTeX's errors and warnings; render.py turns the
model into the page HTML (math typeset by KaTeX in the browser); the grader (generator/latexcheck.py) compares
models.
"""

from .compile import compile_doc  # noqa: F401

"""COMP_GUIDE §9 Vim: one module per subsection (9.01-9.12).

Problems open Vim on bash-lite's machine (the engine is generator/bashlite/vim): the user edits with Vim's keys in
the terminal, and bashlite/vimcheck.py grades Vim's final state and the commands used. Problems have kind "bash" (the
live terminal and its checker) with a `vim` spec.
"""

from . import (m01_modes, m02_motions, m03_editing, m04_grammar, m05_visual, m06_search,  # noqa: F401
               m07_registers, m08_macros, m09_windows, m10_marks_folds_undo, m11_cmdline, m12_vimrc)

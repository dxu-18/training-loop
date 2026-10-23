"""Section 7: Bash & the command line. One module file per COMP_GUIDE subsection (7.01-7.17).

Answers run on bash-lite (generator/bashlite), a simulated Ubuntu machine: problems have kind "bash", and the grader
compares the answer's output (or a check run afterwards) with the reference solution's (bashlite/check.py).
"""

from ...bashlite.check import checker as _checker
from ...grader import _CHECKERS

_CHECKERS["bash"] = _checker

from . import m01_basics, m02_files, m03_viewing, m04_perms_archives, m05_find_grep, m06_text, m07_redirection, m08_expansion, m09_arrays, m10_control, m11_scripts, m12_processes, m13_environment, m14_remote, m15_pyenv, m16_ml_workflow, m17_keys_history  # noqa: E402,F401

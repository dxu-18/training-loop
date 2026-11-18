"""Browser check of the scikit-learn pages: `scripts/py_e2e.py --section sklearn`.

    python scripts/build.py && python scripts/sklearn_e2e.py <screenshot-dir> [--seeds N] [--only 4.03]
"""

import runpy
import sys
from pathlib import Path

if "--section" not in sys.argv:
    sys.argv += ["--section", "sklearn"]
runpy.run_path(str(Path(__file__).resolve().parent / "py_e2e.py"), run_name="__main__")

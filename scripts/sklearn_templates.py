"""Check the scikit-learn templates (COMP_GUIDE §4): `scripts/py_templates.py --section sklearn`.

    python scripts/sklearn_templates.py 4.03 4.04        # or no argument: every module
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py_templates import main  # noqa: E402

if __name__ == "__main__":
    main("sklearn")

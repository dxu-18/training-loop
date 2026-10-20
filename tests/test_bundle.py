"""docs/generator.zip is committed and served as-is, so it must match the sources."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build import source_hash  # noqa: E402


def test_bundle_is_fresh():
    built = json.loads((ROOT / "docs" / "build.json").read_text())["hash"]
    assert built == source_hash(), "docs/ bundle is stale: run `python scripts/build.py`"

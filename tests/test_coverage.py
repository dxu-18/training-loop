"""Every API the guide's code uses is practised by some problem of that module (scripts/guide_coverage.py).

Names that can't run in the browser are listed, with the reason, in coverage.EXCLUDED."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import guide_coverage as coverage  # noqa: E402


@pytest.mark.parametrize("section", ["numpy", "pandas", "pytorch"])
def test_every_guide_api_is_practised(section):
    rep = coverage.report(section, seeds=40)
    missing = {m: r["api_missing"] for m, r in rep.items() if r["api_missing"]}
    assert not missing, missing
    assert not coverage.CRASHES, coverage.CRASHES[:5]


def test_every_guide_bash_feature_is_practised():
    rep = coverage.report_bash(seeds=20)
    missing = {m: r["api_missing"] for m, r in rep.items() if r["api_missing"]}
    assert not missing, missing
    assert not coverage.CRASHES, coverage.CRASHES[:5]

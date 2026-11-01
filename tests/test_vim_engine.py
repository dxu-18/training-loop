"""bash-lite's Vim against real Vim 9.1: every case of tests/vim_corpus/cases.py must end in the state real Vim
reached (tests/vim_corpus/expected.json, recorded by scripts/diff_vim.py --record)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "vim_corpus"))

import diff_vim  # noqa: E402
from cases import CASES  # noqa: E402

EXPECTED = json.loads((ROOT / "tests" / "vim_corpus" / "expected.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_same_as_real_vim(case):
    exp = EXPECTED.get(case[0])
    if exp is None:
        pytest.skip("not recorded yet (scripts/diff_vim.py --record)")
    got = diff_vim.ours(case)
    assert diff_vim.compare(exp, got, case) == []

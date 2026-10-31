"""Part III (Data Structures & Algorithms tutorials): the Python runner the page's worker calls, and the tutorials'
reference solutions in Python (scripts/dsa_selftest.mjs checks all four languages and the mutants)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from generator.api import dsa_run

ROOT = Path(__file__).resolve().parents[1]


def run(user, tests=""):
    return json.loads(dsa_run(user, tests))


def test_tests_see_the_users_definitions_and_only_their_output_counts():
    r = run("def double(x):\n    return 2 * x\nprint('ignored')\n", "print(double(21))\n")
    assert r == {"compiled": True, "stdout": "42\n", "stderr": "", "exit": 0}


def test_without_tests_the_files_own_output_is_shown():
    assert run("print('hi')\n")["stdout"] == "hi\n"


def test_syntax_error_does_not_compile():
    r = run("def f(:\n    pass\n", "print(1)\n")
    assert not r["compiled"] and "SyntaxError" in r["compileOutput"]


def test_a_failed_contract_points_at_the_users_line():
    user = "def insert(L, x):\n    assert len(L) < 1\n    L.append(x)\n"
    r = run(user, "L = []\ninsert(L, 1)\nprint('one')\ninsert(L, 2)\nprint('two')\n")
    assert r["stdout"] == "one\n" and r["exit"] == 1
    assert "line 2, in insert" in r["stderr"] and "assert len(L) < 1" in r["stderr"] and "AssertionError" in r["stderr"]
    assert "tests.py" not in r["stderr"]


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_python_references_pass_every_step():
    r = subprocess.run(["node", str(ROOT / "scripts" / "dsa_selftest.mjs"), "--lang", "py"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert " 0 failures" in r.stdout

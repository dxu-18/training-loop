"""COMP_GUIDE §8 Git & GitHub (generator/sections/s08_git) on bash-lite's git and GitHub-lite: every template's
reference session passes in the live terminal and an idle one doesn't, the practice repositories build
deterministically, and bash-lite's git matches real git on the recorded corpus (when git is installed)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from git_templates import check, problems  # noqa: E402

PROBLEMS = [(f"{tpl.module}-{tpl.name}-{item}-{seed}", p) for tpl, item, seed, p in problems(None, 1)]


@pytest.mark.parametrize("pid,p", PROBLEMS, ids=[x[0] for x in PROBLEMS])
def test_template(pid, p):
    assert check(p) == []


def test_modules_complete():
    from generator.core import is_live, modules, templates, TIERS
    assert modules("git") == [f"8.{i:02d}" for i in range(1, 15)]
    for m in modules("git"):
        assert is_live("git", m), m
        for tier in TIERS:
            assert len(templates("git", m, tier)) >= 2, (m, tier)


def test_catalog_is_deterministic():
    from generator.bashlite.world import World
    from generator.bashlite.git import catalog
    from generator.bashlite.git.github import github_of
    catalog._load()
    assert len([r for r in catalog.RECIPES.values() if r.kind == "light"]) == 30
    assert len([r for r in catalog.RECIPES.values() if r.kind == "medium"]) == 5
    for full in ("minilibs/requests-mini", "kernel-lab/tiny-kernel"):
        refs = []
        for _ in range(2):
            gh = github_of(World(seed=1))
            refs.append(dict(gh.get(*full.split("/")).refs))
        assert refs[0] == refs[1]
    sizes = {}
    for key, rec in catalog.RECIPES.items():
        r = github_of(World(seed=1)).get(rec.owner, rec.name)
        sizes[key] = r.size_kb / 1024
        assert (1 <= sizes[key] <= 5.5) if rec.kind == "light" else (75 <= sizes[key] <= 110), (key, sizes[key])


@pytest.mark.skipif(shutil.which("git") is None, reason="needs real git")
def test_matches_real_git():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "diff_git.py"),
                        *map(str, sorted((ROOT / "tests" / "git_corpus").glob("*.sh")))],
                       capture_output=True, text=True, timeout=600)
    assert "0 differing command(s)" in r.stdout, r.stdout[-3000:]

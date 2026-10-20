"""Every template, over many seeds: the reference solution must pass its own grader,
obviously wrong answers must fail, and different seeds must give different problems."""

import pytest

from generator import TIERS
from generator.core import REGISTRY, build, generate, is_live, modules, templates, tier_for
from generator.grader import grade

SEEDS = range(25)
ALL = [t for s in REGISTRY for t in templates(s)]
IDS = lambda t: f"{t.section}-{t.module}-{t.tier}-{t.name}"  # noqa: E731

NUMPY_MODULES = [f"1.{i:02d}" for i in range(1, 19)]


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_reference_passes(tpl):
    for seed in SEEDS:
        p = build(tpl, seed).to_dict()
        res = grade(p, p["solution"])
        assert res["correct"], f"seed {seed}: {res}\n{p['prompt']}\n{p['setup']}\n{p['solution']}"


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_wrong_answers_fail(tpl):
    for seed in range(5):
        p = build(tpl, seed).to_dict()
        for wrong in ("result = None", "result = 0", "1 +"):
            assert not grade(p, wrong)["correct"], (seed, wrong)


@pytest.mark.parametrize("tpl", ALL, ids=IDS)
def test_variety(tpl):
    """Different seeds should give different problems."""
    seen = {build(tpl, s).prompt + build(tpl, s).setup for s in range(12)}
    assert len(seen) >= 4


def test_numpy_modules_complete():
    assert modules("numpy") == NUMPY_MODULES
    for m in NUMPY_MODULES:
        assert is_live("numpy", m), m
        for tier in TIERS:
            assert len(templates("numpy", m, tier)) >= 2, (m, tier)


def test_template_names_unique():
    for s in REGISTRY:
        names = [t.name for t in templates(s)]
        assert len(names) == len(set(names))


def test_schedule():
    assert [tier_for(n) for n in (1, 50, 51, 120, 121, 200, 201, 245, 246, 250)] == [
        "lite", "lite", "easy", "easy", "medium", "medium", "hard", "hard", "challenge", "challenge"]
    later = [tier_for(250 + i, seed=i) for i in range(5000)]
    assert 0 < later.count("challenge") < 300  # ~2% of draws


def test_generate_attaches_example():
    p = generate("numpy", "1.05", "easy", 123)
    assert p["example"]["solution"] and p["tier"] == "easy" and p["module"] == "1.05"

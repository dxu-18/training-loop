"""Every template, over many seeds: the reference solution must pass its own grader,
and obviously wrong answers must fail."""

import random

import pytest

from generator import TIERS, grade, templates
from generator.core import REGISTRY, build, generate, tier_for

SEEDS = range(25)
ALL = [t for s in REGISTRY for t in templates(s)]


@pytest.mark.parametrize("tpl", ALL, ids=lambda t: f"{t.section}-{t.tier}-{t.name}")
def test_reference_passes(tpl):
    for seed in SEEDS:
        p = build(tpl, seed).to_dict()
        res = grade(p, p["solution"])
        assert res["correct"], f"seed {seed}: {res}\n{p['prompt']}\n{p['setup']}\n{p['solution']}"


@pytest.mark.parametrize("tpl", ALL, ids=lambda t: f"{t.section}-{t.tier}-{t.name}")
def test_wrong_answers_fail(tpl):
    for seed in range(5):
        p = build(tpl, seed).to_dict()
        for wrong in ("result = None", "result = 0", "1 +"):
            assert not grade(p, wrong)["correct"], (seed, wrong)


@pytest.mark.parametrize("tpl", ALL, ids=lambda t: f"{t.section}-{t.tier}-{t.name}")
def test_variety(tpl):
    """Different seeds should give different problems."""
    prompts = {build(tpl, s).prompt + build(tpl, s).setup for s in range(10)}
    assert len(prompts) >= 5


def test_every_tier_has_templates():
    for section in REGISTRY:
        for tier in TIERS:
            assert templates(section, tier), (section, tier)


def test_schedule():
    assert [tier_for(n) for n in (1, 50, 51, 120, 121, 200, 201, 245, 246, 250)] == [
        "lite", "lite", "easy", "easy", "medium", "medium", "hard", "hard", "challenge", "challenge"]
    later = [tier_for(250 + i, seed=i) for i in range(5000)]
    assert 0 < later.count("challenge") < 300  # ~2% of draws


def test_generate_attaches_example():
    p = generate("numpy", "easy", 123)
    assert p["example"]["solution"] and p["tier"] == "easy"

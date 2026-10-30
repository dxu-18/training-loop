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
    # challenges also run large hidden tests, and test_challenges.py checks them against brute force
    for seed in (range(6) if tpl.tier == "challenge" else SEEDS):
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


def test_pandas_modules_complete():
    assert modules("pandas") == [f"2.{i:02d}" for i in range(1, 20)]
    for m in modules("pandas"):
        for tier in TIERS:
            assert len(templates("pandas", m, tier)) >= 2, (m, tier)


def test_pytorch_modules_complete():
    assert modules("pytorch") == [f"3.{i:02d}" for i in range(1, 20)]
    for m in modules("pytorch"):
        for tier in TIERS:
            assert len(templates("pytorch", m, tier)) >= 2, (m, tier)


def test_numpy_modules_complete():
    assert modules("numpy") == NUMPY_MODULES
    for m in NUMPY_MODULES:
        assert is_live("numpy", m), m
        for tier in TIERS:
            assert len(templates("numpy", m, tier)) >= 2, (m, tier)


def test_bash_modules_complete():
    assert modules("bash") == [f"7.{i:02d}" for i in range(1, 18)]
    for m in modules("bash"):
        assert is_live("bash", m), m
        for tier in TIERS:
            assert len(templates("bash", m, tier)) >= 2, (m, tier)


def test_template_names_unique():
    for s in REGISTRY:
        names = [t.name for t in templates(s)]
        assert len(names) == len(set(names))


def test_schedule():
    assert [tier_for(n) for n in (1, 10, 11, 35, 36, 70, 71, 95, 96, 100)] == [
        "lite", "lite", "easy", "easy", "medium", "medium", "hard", "hard", "challenge", "challenge"]
    first = [tier_for(n) for n in range(1, 101)]
    assert [first.count(t) for t in ("lite", "easy", "medium", "hard", "challenge")] == [10, 25, 35, 25, 5]
    later = [tier_for(100 + i, seed=i) for i in range(5000)]
    assert 100 < later.count("challenge") < 400  # ~5% of draws


def test_generate_attaches_instructions():
    p = generate("numpy", "1.05", "easy", 123)
    assert p["tier"] == "easy" and p["module"] == "1.05" and "example" not in p
    assert 1 <= len(p["instructions"]) <= 4
    assert all(it["name"] and it["desc"] for it in p["instructions"])


@pytest.mark.parametrize("section", sorted(REGISTRY))
def test_instructions_are_short_and_described(section):
    """Every problem lists at most a few functions, each with a description; almost none list nothing."""
    from generator.instructions import MAX_ITEMS, instructions
    empty = total = 0
    for tpl in templates(section):
        for seed in range(2):
            items = instructions(build(tpl, seed))
            total += 1
            empty += not items
            assert len(items) <= MAX_ITEMS, (tpl.name, items)
            assert all(it["name"] and it["desc"] and len(it["desc"]) < 260 for it in items), (tpl.name, items)
    assert empty <= 0.03 * total, f"{empty}/{total} {section} problems have no Instructions"


@pytest.mark.parametrize("section", sorted(REGISTRY))
def test_instructions_list_what_is_new(section):
    """When a problem uses names that are new at its module (introduced there or later), only those are listed;
    names from earlier modules appear only when nothing new is used."""
    from generator.instructions import _required, canon, data, instructions
    intro = data()[section]["intro"]
    rank = {m: i for i, m in enumerate(data()[section]["modules"])}
    for tpl in templates(section):
        p = build(tpl, 0)
        here = rank[tpl.module]
        need = _required(p.to_dict(), section)
        new = [rank[intro.get(canon(x["key"], section), [tpl.module])[0]] >= here
               for x in instructions(p, with_keys=True) if canon(x["key"], section) not in need]
        assert all(new) or not any(new), (tpl.name, new)
    tree = [it["name"] for it in generate("bash", "7.02", "lite", 10)["instructions"]]
    assert tree[0] == "mkdir" and any(n.startswith("touch") for n in tree), tree


@pytest.mark.parametrize("section", sorted(REGISTRY))
def test_instructions_list_required_names(section):
    """What a problem requires (the grader insists on it) is always in its Instructions, if it can be described."""
    from generator.instructions import _required, canon, describe, instructions, names_of
    for tpl in templates(section):
        p = build(tpl, 1)
        need = _required(p.to_dict(), section)
        used = [k for k in names_of(p) if canon(k, section) in need and describe(section, k)]
        listed = {canon(x["key"], section) for x in instructions(p, with_keys=True)}

        def shown(k):   # `wc` is shown as `wc -l`; an accessor (.str) inside its method's name (s.str.title())
            c = canon(k, section)
            return c in listed or any(x.startswith(c + " ") for x in listed) or c in ("str", "dt", "cat")
        assert all(shown(k) for k in used[:4]), (tpl.name, used, listed)
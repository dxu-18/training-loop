"""Catalog templates hold many items (one API each); grade every item, not just the ones random seeds hit."""

import pytest

from generator import core
from generator.core import REGISTRY, build, templates
from generator.grader import grade


def _items():
    out = []
    for s in REGISTRY:
        for t in templates(s):
            core._PICKED.clear()
            core._FORCE_ITEM[0] = None
            build(t, 0)
            if core._PICKED:
                out += [(t, i) for i in range(core._PICKED[0])]
    core._PICKED.clear()
    return out


ITEMS = _items()


@pytest.mark.parametrize("tpl,i", ITEMS, ids=lambda v: v.name if hasattr(v, "name") else str(v))
def test_catalog_item(tpl, i):
    core._FORCE_ITEM[0] = i
    try:
        for seed in range(4):
            p = build(tpl, seed).to_dict()
            res = grade(p, p["solution"])
            assert res["correct"], f"item {i} seed {seed}: {res}\n{p['prompt']}\n{p['setup']}\n{p['solution']}"
            for wrong in ("result = None", "result = 0"):
                assert not grade(p, wrong)["correct"], (i, seed, wrong)
    finally:
        core._FORCE_ITEM[0] = None


def test_catalog_items_differ():
    """Each item of a catalog asks something different (no copy-paste duplicates)."""
    by_tpl = {}
    for tpl, i in ITEMS:
        core._FORCE_ITEM[0] = i
        try:
            by_tpl.setdefault(tpl.name, []).append(build(tpl, 1).solution)
        finally:
            core._FORCE_ITEM[0] = None
    for name, sols in by_tpl.items():
        dup = {s for s in sols if sols.count(s) > 1}
        assert not dup, (name, sorted(dup)[:3])

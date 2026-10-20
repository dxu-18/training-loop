"""JSON-in / JSON-out entry points called from the browser (docs/worker.js)."""

from __future__ import annotations

import json
import random

from . import core, grader


def generate(section: str, module: str, n: int, seed: int, tier: str | None = None) -> str:
    """Problem for question number n of a module. Pass `tier` to reroll at a fixed difficulty."""
    tier = tier or core.tier_for(n, seed)
    return json.dumps(core.generate(section, module, tier, seed))


def run(problem_json: str, code: str) -> str:
    return json.dumps(grader.run(json.loads(problem_json), code))


def grade(problem_json: str, code: str) -> str:
    return json.dumps(grader.grade(json.loads(problem_json), code))


def info() -> str:
    return json.dumps({
        s: {m: {t: len(core.templates(s, m, t)) for t in core.TIERS} for m in core.modules(s)}
        for s in core.sections()
    })


def self_test(section: str, seeds: int = 2) -> str:
    """Grade every template's own solution in this runtime (used to check Pyodide)."""
    failures, count = [], 0
    for tpl in core.templates(section):
        for seed in range(seeds):
            count += 1
            try:
                p = core.build(tpl, random.Random(seed).randrange(2**31)).to_dict()
                res = grader.grade(p, p["solution"])
                if not res["correct"]:
                    failures.append({"template": tpl.name, "module": tpl.module, "seed": seed,
                                     "message": res["message"], "error": res["error"]})
            except Exception as e:  # noqa: BLE001
                failures.append({"template": tpl.name, "module": tpl.module, "seed": seed,
                                 "message": "generation crashed", "error": repr(e)})
    return json.dumps({"checked": count, "failures": failures})

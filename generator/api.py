"""JSON-in / JSON-out entry points called from the browser (docs/worker.js)."""

from __future__ import annotations

import json
import random
import time

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


def self_test(section: str, seeds: int = 2, only: str = "", skip: str = "") -> str:
    """Grade every template's own solution in this runtime (used to check Pyodide).

    `only` / `skip` are comma-separated module ids, so each module can be checked with
    exactly the packages its page loads."""
    want = set(filter(None, only.split(",")))
    drop = set(filter(None, skip.split(",")))
    failures, count, timings = [], 0, []
    for tpl in core.templates(section):
        if (want and tpl.module not in want) or tpl.module in drop:
            continue
        for seed in range(seeds):
            count += 1
            try:
                p = core.build(tpl, random.Random(seed).randrange(2**31)).to_dict()
                t0 = time.perf_counter()
                res = grader.grade(p, p["solution"])
                timings.append((time.perf_counter() - t0, tpl.module, tpl.name, p["timeout"]))
                if not res["correct"]:
                    failures.append({"template": tpl.name, "module": tpl.module, "seed": seed,
                                     "message": res["message"], "error": res["error"]})
            except Exception as e:  # noqa: BLE001
                failures.append({"template": tpl.name, "module": tpl.module, "seed": seed,
                                 "message": "generation crashed", "error": repr(e)})
    slow = sorted(timings, reverse=True)[:5]
    return json.dumps({"checked": count, "failures": failures,
                       "slowest": [{"seconds": round(t, 2), "module": m, "template": n, "limit": lim} for t, m, n, lim in slow]})

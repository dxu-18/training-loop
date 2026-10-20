"""JSON-in / JSON-out entry points called from the browser (docs/worker.js)."""

from __future__ import annotations

import json

from . import core, grader


def generate(section: str, n: int, seed: int, tier: str | None = None) -> str:
    """Problem for question number n. Pass `tier` to reroll at a fixed difficulty."""
    tier = tier or core.tier_for(n, seed)
    return json.dumps(core.generate(section, tier, seed))


def run(problem_json: str, code: str) -> str:
    return json.dumps(grader.run(json.loads(problem_json), code))


def grade(problem_json: str, code: str) -> str:
    return json.dumps(grader.grade(json.loads(problem_json), code))


def info() -> str:
    return json.dumps({
        "sections": {s: {t: len(core.templates(s, t)) for t in core.TIERS} for s in core.sections()},
        "example_until": core.EXAMPLE_UNTIL, "medal_at": core.MEDAL_AT, "reveal_after": core.REVEAL_AFTER,
    })

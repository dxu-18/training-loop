"""Check the scikit-learn templates (COMP_GUIDE §4, generator/sections/s04_sklearn) quickly, module by module.

    python scripts/sklearn_templates.py 4.03 4.04        # or no argument: every module
    python scripts/sklearn_templates.py 4.03 --seeds 8

For every template (every item of a catalog): the reference solution grades correct on several seeds (it runs twice,
so unseeded randomness shows up as a failure), obviously wrong answers fail, seeds give different problems, the
Instructions list is non-empty and short, and the reference is fast enough for Pyodide (a few times slower than
here). Prints how many distinct problems each tier has next to how many questions the schedule asks of it.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from generator import core  # noqa: E402
from generator.core import TIERS, build, modules, templates  # noqa: E402
from generator.grader import grade  # noqa: E402
from generator.instructions import MAX_ITEMS, instructions  # noqa: E402

QUESTIONS = {"lite": 10, "easy": 25, "medium": 35, "hard": 25, "challenge": 5}
SLOW = 0.5          # seconds for one reference run here (Pyodide: ~3x, and grading runs it at least twice)
HAZARDS = re.compile(r"\bn_jobs\b|\bfetch_\w+\(|\bthreading\b|\bmultiprocessing\b|\bplt\.show\(")


def builds(t, seeds):
    """(label, problem) for every catalog item (2 seeds each) or `seeds` seeds; -> also the item count."""
    core._PICKED.clear()
    core._FORCE_ITEM[0] = None
    first = build(t, 0)
    n_items = core._PICKED[0] if core._PICKED else 0
    out = []
    if n_items:
        for i in range(n_items):
            core._FORCE_ITEM[0] = i
            try:
                out += [(f"[{i}] s{s}", build(t, s)) for s in range(2)]
            finally:
                core._FORCE_ITEM[0] = None
        out += [(f"s{s}", build(t, s)) for s in range(2, seeds)]
    else:
        out = [("s0", first)] + [(f"s{s}", build(t, s)) for s in range(1, seeds)]
    return out, n_items


def skeleton(p) -> str:
    """A problem's solution with numbers and strings blanked: different skeletons = different kinds of problem."""
    s = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"", "S", p.solution)
    return re.sub(r"\b\d+(\.\d+)?(e-?\d+)?\b", "#", s)


def kinds(t, n_items) -> int:
    seen = set()
    core._PICKED.clear()
    for i in range(n_items or 1):
        core._FORCE_ITEM[0] = i if n_items else None
        try:
            seen |= {skeleton(build(t, s)) for s in range(30 if not n_items else 6)}
        finally:
            core._FORCE_ITEM[0] = None
    return len(seen)


def check_template(t, seeds) -> tuple[list[str], list[str], int, float]:
    fails, warns = [], []
    try:
        probs, n_items = builds(t, seeds)
    except Exception as e:  # noqa: BLE001
        return [f"build crashed: {type(e).__name__}: {e}"], [], 0, 0.0
    worst = 0.0
    for label, p in probs:
        d = p.to_dict()
        t0 = time.perf_counter()
        try:
            res = grade(d, d["solution"])
        except Exception as e:  # noqa: BLE001  (the reference itself raised)
            fails.append(f"{label}: reference raises {type(e).__name__}: {str(e)[:300]}")
            continue
        dt = (time.perf_counter() - t0) / (2 * (1 + len(d.get("tests") or [])))
        worst = max(worst, dt)
        if not res["correct"]:
            fails.append(f"{label}: reference fails: {res['message'] or ''} {(res['error'] or '')[-400:]}")
            continue
        for wrong in ("result = None", "result = 0", "1 +"):
            if grade(d, wrong)["correct"]:
                fails.append(f"{label}: wrong answer {wrong!r} passes")
        if HAZARDS.search(d["solution"] + "\n" + d["setup"]):
            fails.append(f"{label}: uses something Pyodide can't do (n_jobs, fetch_*, threads, plt.show)")
        items = instructions(d)
        if not items:
            warns.append(f"{label}: no Instructions")
        if len(items) > MAX_ITEMS:
            fails.append(f"{label}: {len(items)} Instructions")
        if re.search(r"\{[a-z_]+\}", d["prompt"]):
            warns.append(f"{label}: prompt has an unfilled {{...}}: {d['prompt'][:80]}")
    if not getattr(t.fn, "fixed", False):
        seen = {build(t, s).prompt + build(t, s).setup for s in range(12)}
        if len(seen) < 4:
            fails.append(f"only {len(seen)} different problems over 12 seeds")
    if worst > SLOW:
        warns.append(f"slow: {worst:.2f}s per reference run")
    return fails, warns, kinds(t, n_items), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    mods = args.modules or modules("sklearn")
    import sklearn.linear_model, sklearn.preprocessing, sklearn.tree, sklearn.model_selection  # noqa: E401,F401 (warm-up)
    total_fail = 0
    for m in mods:
        counts = {tr: [0, 0] for tr in TIERS}      # templates, distinct problem kinds (catalog items)
        for t in templates("sklearn", m):
            fails, warns, kinds, worst = check_template(t, args.seeds)
            counts[t.tier][0] += 1
            counts[t.tier][1] += kinds
            for f in fails:
                print(f"FAIL {m} {t.tier:9} {t.name}: {f}")
            for w in warns:
                print(f"warn {m} {t.tier:9} {t.name}: {w}")
            total_fail += len(fails)
        print(f"{m}: " + "  ".join(f"{tr} {c[0]} tpl/{c[1]} kinds (of {QUESTIONS[tr]})" for tr, c in counts.items()))
    print(f"{total_fail} failure(s)")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()

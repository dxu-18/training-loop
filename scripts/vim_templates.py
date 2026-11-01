"""Check the Vim templates (generator/sections/s09_vim): every catalog item of every template, over a few seeds.

For each problem: the reference keys, sent as the page would, must pass; a session that only presses Esc must not;
the prompt is printed with -v.

    python scripts/vim_templates.py                 # all modules
    python scripts/vim_templates.py 9.03 -v         # one module, with prompts
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator import core, grader  # noqa: E402
from generator.bashlite.live import reference_session, vim_session  # noqa: E402


def problems(module=None, seeds=2):
    for tpl in core.templates("vim", module):
        core._PICKED.clear()
        try:
            core.build(tpl, 0)
        except Exception:  # noqa: BLE001  (reported below, item by item)
            pass
        n = core._PICKED[0] if core._PICKED else 1
        for item in range(n):
            for seed in range(seeds):
                core._FORCE_ITEM[0] = item if core._PICKED else None
                try:
                    yield tpl, item, seed, core.build(tpl, random.Random(seed * 7919 + item).randrange(2**31)).to_dict(), None
                except Exception as e:  # noqa: BLE001
                    yield tpl, item, seed, None, e
                finally:
                    core._FORCE_ITEM[0] = None


def idle_answer(p):
    """A session that opens Vim (on no file, if the problem starts at the prompt) and presses Esc."""
    import json
    start = "vim<CR>" if not any(l.startswith("#: vim ") for l in p["setup"].split("\n")) else ""
    t = vim_session(p, start + "<Esc>")
    end = t.stamp("end")
    return json.dumps({"log": t.log, "end": {"m": end["m"], "d": end.get("d", 0)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    bad = 0
    n = 0
    t0 = time.time()
    for tpl, item, seed, p, err in problems(a.module, a.seeds):
        n += 1
        tag = f"{tpl.module} {tpl.tier:9} {tpl.name}[{item}] seed {seed}"
        if err is not None:
            bad += 1
            print(f"FAIL {tag}: generation: {err!r}")
            continue
        if a.v and seed == 0:
            from generator.instructions import instructions
            print(f"\n=== {tag}\n{p['prompt']}\n--- solution: {p['solution']}\n--- require {p['require']} forbid "
                  f"{p['forbid']} vim {({k: v for k, v in p['vim'].items() if k != 'uses'})}")
            print("--- instructions:", [i["name"] for i in instructions(p)])
        res = grader.grade(p, reference_session(p))
        if not res["correct"]:
            bad += 1
            print(f"FAIL {tag}: reference: {res['message'] or res['error']}")
            continue
        res = grader.grade(p, idle_answer(p))
        if res["correct"]:
            bad += 1
            print(f"FAIL {tag}: an idle session passes")
    print(f"{n} problems, {bad} failures, {time.time() - t0:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

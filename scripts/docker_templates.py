"""Check every Docker template (COMP_GUIDE §12) in the live terminal: for each catalog item and two seeds, the reference
session passes, its commands print no fatal:/error:, the setup runs cleanly, and a session that only runs
`docker ps` fails.

    python scripts/docker_templates.py [12.03 12.07 ...] [-v]
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generator  # noqa: E402,F401
from generator import core, grader  # noqa: E402
from generator.bashlite.directives import build, parse, prepare, split  # noqa: E402
from generator.bashlite.live import LiveTerm, reference_session, text_log  # noqa: E402

SECTION = "docker"
BAD = re.compile(r"^(fatal:|error:|ERROR|docker: |Error response|Error:|bash: .*command not found|bash-lite: internal|Traceback|.*Permission denied|.*No such (file|container|image|object))", re.M)


def problems(mods=None, seeds=2):
    """(template, item, seed, problem dict) for every catalog item and `seeds` seeds of every git template."""
    for tpl in core.templates(SECTION):
        if mods and tpl.module not in mods:
            continue
        core._PICKED.clear()
        core.build(tpl, 0)
        n = core._PICKED[0] if core._PICKED else 0
        for seed, item in [(s, None) for s in range(seeds)] + [(i, i) for i in range(n)]:
            core._FORCE_ITEM[0] = item
            try:
                yield tpl, item, seed, core.build(tpl, random.Random(seed).randrange(2 ** 31)).to_dict()
            finally:
                core._FORCE_ITEM[0] = None


def idle_answer(p):
    return reference_session({**p, "solution": "true" if p["solution"].strip() == "docker ps" else "docker ps"})


def check(p) -> list[str]:
    """What's wrong with one problem (empty when it's fine)."""
    out = []
    res = grader.grade(p, reference_session(p))
    if not res["correct"]:
        out.append(f"reference fails: {res['message']} {res['error']}")
    dirs, code = split(p["setup"])
    spec = parse(dirs)
    _, sess = build(spec, seed=7)
    prepare(sess, spec)
    text = sess.run(code)["text"] if code.strip() else ""
    if BAD.search(text):
        out.append("setup prints: " + text[BAD.search(text).start():][:200].replace("\n", " | "))
    t = LiveTerm(p, p["setup"], typed=True)
    t.replay(text_log(p["solution"]))
    text = "".join(c["stdout"] + c["stderr"] for c in t.commands())
    if BAD.search(text):
        out.append("reference prints: " + text[BAD.search(text).start():][:200].replace("\n", " | "))
    if grader.grade(p, idle_answer(p))["correct"]:
        out.append("an idle session passes")
    return out


def main():
    mods = [a for a in sys.argv[1:] if not a.startswith("-")]
    n = bad = 0
    for tpl, item, seed, p in problems(mods):
        n += 1
        errs = check(p)
        if errs or "-v" in sys.argv:
            print(("FAIL " if errs else "ok   ") + f"{tpl.module} {tpl.tier:9} {tpl.name}" +
                  (f"[{item}]" if item is not None else f"(s{seed})"))
            for e in errs:
                print("      " + e)
        bad += bool(errs)
    print(f"{n} checked, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

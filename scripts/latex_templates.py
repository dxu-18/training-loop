"""Check the LaTeX templates (generator/sections/s11_latex): every catalog item of every template, over a few seeds.

For each problem: the reference answer passes; an empty answer (or the starter code, unchanged) doesn't; -v prints
the prompts. --katex also typesets every formula of the compiled targets with KaTeX (needs node and the katex npm
package: --katex <path to katex.js>).

    python scripts/latex_templates.py                 # all modules
    python scripts/latex_templates.py 11.03 -v
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator import core, grader  # noqa: E402


def problems(module=None, seeds=2):
    for tpl in core.templates("latex", module):
        core._PICKED.clear()
        try:
            core.build(tpl, 0)
        except Exception:  # noqa: BLE001
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--katex")
    a = ap.parse_args()
    bad = n = 0
    t0 = time.time()
    formulas = []
    for tpl, item, seed, p, err in problems(a.module, a.seeds):
        n += 1
        tag = f"{tpl.module} {tpl.tier:9} {tpl.name}[{item}] seed {seed}"
        if err is not None:
            bad += 1
            print(f"FAIL {tag}: generation: {err!r}")
            continue
        if a.v and seed == 0:
            from generator.instructions import instructions
            print(f"\n=== {tag}\n{p['prompt']}\n--- mode {p['latex']['mode']} require {p['require']} forbid {p['forbid']}"
                  f"\n--- solution:\n{p['solution']}--- instructions: {[i['name'] for i in instructions(p)]}")
        res = grader.grade(p, p["solution"])
        if not res["correct"]:
            bad += 1
            print(f"FAIL {tag}: reference: {res['message'] or res['error']}")
            continue
        empty = p["latex"].get("starter", "")
        res = grader.grade(p, empty)
        if res["correct"]:
            bad += 1
            print(f"FAIL {tag}: an empty answer (or the unchanged starter) passes")
        if a.katex and p["latex"].get("target"):
            formulas += [x for x in re.findall(r'data-tex="([^"]*)"', p["latex"]["target"])]
    if a.katex and formulas:
        import html
        uniq = sorted(set(html.unescape(f) for f in formulas))
        js = ("const k=require(process.argv[1]);const fs=require('fs');const xs=JSON.parse(fs.readFileSync(0,'utf8'));"
              "for(const x of xs){try{k.renderToString(x,{displayMode:true,throwOnError:true,strict:'ignore'})}"
              "catch(e){console.log('KATEX '+JSON.stringify(x)+' '+e.message.slice(0,100))}}")
        out = subprocess.run(["node", "-e", js, a.katex], input=json.dumps(uniq), capture_output=True, text=True)
        for line in out.stdout.splitlines():
            bad += 1
            print(line)
        print(f"{len(uniq)} formulas typeset with KaTeX")
    print(f"{n} problems, {bad} failures, {time.time() - t0:.1f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

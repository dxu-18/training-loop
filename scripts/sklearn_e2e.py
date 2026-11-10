"""Browser check of the scikit-learn pages (COMP_GUIDE §4) in Pyodide's scikit-learn.

    python scripts/build.py && python scripts/sklearn_e2e.py <screenshot-dir> [--seeds N] [--only 4.03,4.10]

The topic page lists the 20 modules; on a few modules the reference solution submitted through the page is graded
correct; then every template's reference is graded inside Pyodide (the worker's self-test), on a page that loaded the
module's packages (4.17 also loads matplotlib), and the slowest grade must fit its time limit.
"""

import argparse
import json
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--only", default="")
args = ap.parse_args()

BASE = "http://127.0.0.1:8773"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8773", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
SET_CODE = "s => document.querySelector('.CodeMirror').CodeMirror.setValue(s)"
VERDICT = "document.querySelector('.verdict').textContent"
problems = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        problems.append(what)


def selftest(pg, only="", skip=""):
    t0 = time.time()
    rep = pg.evaluate(f"py.call('selftest', {{section: 'sklearn', seeds: {args.seeds}, only: '{only}', skip: '{skip}'}})")
    fails = rep["failures"]
    what = only or f"all but {skip}" if skip else only or "all"
    check(not fails, f"Pyodide self-test sklearn {what}: {rep['checked']} runs, {len(fails)} failures "
                     f"({time.time() - t0:.0f}s)")
    for f in fails[:30]:
        print("     ", json.dumps(f)[:500])
    for slow in rep["slowest"][:5]:
        check(slow["seconds"] < slow["limit"], f"grade {slow['seconds']}s ({slow['module']} {slow['template']}) within "
                                               f"its {slow['limit']}s limit")


with sync_playwright() as p:
    browser = p.chromium.launch()
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.set_default_timeout(600000)
    try:
        pg.goto(BASE + "/#/sklearn")
        pg.wait_for_selector(".modules")
        n_mod = pg.locator(".modules a.topic").count()
        check(n_mod == 20, f"sklearn page lists 20 live modules (got {n_mod})")
        for mod in ("4.01", "4.06", "4.13"):
            pg.goto(BASE + f"/#/sklearn/{mod}")
            pg.wait_for_selector(".prompt", timeout=300000)
            pg.evaluate(SET_CODE, pg.evaluate("view.problem.solution"))
            pg.click(".submit")
            pg.wait_for_function(f"{VERDICT}.length > 0", timeout=120000)
            check("Correct" in pg.inner_text(".verdict"), f"sklearn {mod} reference solution is graded correct")
            pg.screenshot(path=f"{args.out}/sklearn_{mod}.png", full_page=True)
        if args.only:
            selftest(pg, only=args.only)
        else:
            selftest(pg, skip="4.17")
            pg.goto(BASE + "/#/sklearn/4.17")
            pg.reload()
            pg.wait_for_selector(".prompt", timeout=300000)
            selftest(pg, only="4.17")
    except Exception as e:  # noqa: BLE001
        check(False, f"{type(e).__name__}: {str(e)[:300]}")
    check(not errors, f"no page errors ({errors[:3]})")
    browser.close()
srv.terminate()
print(f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)

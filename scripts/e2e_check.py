"""Browser click-through of the site, plus a self-test of every template inside Pyodide.

Needs: pip install playwright && playwright install chromium

    python scripts/build.py && python scripts/e2e_check.py <screenshot-dir> [base-url] [--selftest-seeds N]

Without base-url it serves docs/ on 127.0.0.1:8765. Covers: topic list -> topic page -> module
practice, problem render, syntax error, Run, infinite-loop time limit, correct -> next, wrong ->
reroll, a challenge with hidden tests, the medal at 250, answer reveal at 251, counters on the
topic and home pages, the phone layout, and (unless --selftest-seeds 0) grading every NumPy
template's own solution inside Pyodide.
"""

import argparse
import json
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("base", nargs="?")
ap.add_argument("--selftest-seeds", type=int, default=1)
args = ap.parse_args()

BASE = (args.base or "http://127.0.0.1:8765").rstrip("/")
srv = None if args.base else subprocess.Popen(
    [sys.executable, "-m", "http.server", "8765", "-b", "127.0.0.1", "-d", "docs"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

SET_CODE = "s => document.querySelector('.CodeMirror').CodeMirror.setValue(s)"
VERDICT = "document.querySelector('.verdict').textContent"
problems = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))

        # home -> topic -> module
        pg.goto(BASE + "/")
        pg.wait_for_selector(".topic")
        check(pg.locator("a.topic").count() >= 1, "home lists at least one live topic")
        pg.screenshot(path=f"{args.out}/home.png", full_page=True)
        pg.click("a[href='#/numpy']")
        pg.wait_for_selector(".modules")
        n_mod = pg.locator(".modules a.topic").count()
        check(n_mod == 18, f"NumPy page lists 18 live modules (got {n_mod})")
        pg.screenshot(path=f"{args.out}/topic.png", full_page=True)
        pg.click("a[href='#/numpy/1.05']")
        pg.wait_for_selector(".prompt", timeout=120000)
        check(pg.inner_text(".qnum") == "1", "practice starts at question 1")
        check("1.05" in pg.inner_text(".bar"), "practice bar shows the module")
        check(pg.evaluate("document.querySelector('details.example').open"), "example is open for question 1")
        pg.screenshot(path=f"{args.out}/practice.png", full_page=True)

        # syntax error keeps the problem
        pg.evaluate(SET_CODE, "1 +")
        pg.click(".submit")
        pg.wait_for_function(f"{VERDICT}.length > 0")
        check("compile" in pg.inner_text(".verdict"), "syntax error -> Doesn't compile")
        # Run shows stdout + traceback
        pg.evaluate(SET_CODE, "print('hello')\nundefined_name")
        pg.click(".run")
        pg.wait_for_function("!document.querySelector('.out').textContent.includes('Running')")
        out = pg.inner_text(".out")
        check("hello" in out and "NameError" in out, "Run shows stdout and the error")
        # infinite loop -> time limit, then Python restarts
        pg.evaluate(SET_CODE, "while True: pass")
        pg.click(".run")
        pg.wait_for_function("!document.querySelector('.out').textContent.includes('Running')", timeout=60000)
        check("Time limit" in pg.inner_text(".out"), "infinite loop hits the time limit")
        pg.wait_for_function("!document.querySelector('.run').disabled", timeout=60000)
        # correct -> next number
        pg.evaluate(SET_CODE, pg.evaluate("view.problem.solution"))
        pg.click(".submit")
        pg.wait_for_function(f"{VERDICT}.includes('Correct')", timeout=60000)
        pg.click(".next")
        pg.wait_for_function("document.querySelector('.qnum').textContent === '2'", timeout=30000)
        check(True, "correct answer -> question 2")
        # wrong -> same number, new problem, same tier
        before = pg.evaluate("[view.problem.seed, view.problem.tier]")
        pg.evaluate(SET_CODE, "result = None")
        pg.click(".submit")
        pg.wait_for_function(f"{VERDICT}.includes('Wrong')")
        pg.click(".next")
        pg.wait_for_function(f"view.problem && view.problem.seed !== {before[0]}")
        check(pg.inner_text(".qnum") == "2" and pg.evaluate("view.problem.tier") == before[1],
              "wrong answer -> same number and tier, new problem")

        # jump to question 250 of this module: a challenge, then the medal
        pg.evaluate("() => { state.modules['numpy/1.05'] = {solved: 249, medal: false, current: {n: 250, seed: 7, tier: null}}; save(); }")
        pg.reload()
        pg.wait_for_selector(".prompt", timeout=120000)
        check(pg.evaluate("view.problem.tier") == "challenge", "question 250 is a challenge")
        pg.evaluate(SET_CODE, pg.evaluate("view.problem.solution"))
        t0 = time.time()
        pg.click(".submit")
        pg.wait_for_function(f"{VERDICT}.length > 0", timeout=60000)
        check("Correct" in pg.inner_text(".verdict"), f"challenge reference passes hidden tests ({time.time() - t0:.1f}s)")
        pg.wait_for_selector(".modal", timeout=5000)
        pg.screenshot(path=f"{args.out}/medal.png")
        pg.click(".modal button")
        pg.click(".next")
        pg.wait_for_function("document.querySelector('.qnum').textContent === '251'", timeout=30000)
        check(pg.locator("details.answer").count() == 1, "question 251 offers Show answer")
        check(not pg.evaluate("document.querySelector('details.example').open"), "example is collapsed after question 10")

        # counters on the topic page and the home page
        pg.goto(BASE + "/#/numpy")
        pg.wait_for_selector(".modules")
        row = pg.inner_text("a[href='#/numpy/1.05']")
        check("250" in row and pg.locator("a[href='#/numpy/1.05'] svg").count() == 1, "topic page: 250 + medal on 1.05")
        pg.screenshot(path=f"{args.out}/topic_after.png", full_page=True)
        pg.goto(BASE + "/#/")
        pg.wait_for_selector(".topic")
        check("×1" in pg.inner_text("a[href='#/numpy']"), "home shows the medal count")
        pg.screenshot(path=f"{args.out}/home_after.png", full_page=True)

        # phone layout
        m = b.new_page(viewport={"width": 390, "height": 844})
        m.goto(BASE + "/#/numpy")
        m.wait_for_selector(".modules")
        m.screenshot(path=f"{args.out}/mobile_topic.png", full_page=True)
        m.goto(BASE + "/#/numpy/1.18")
        m.wait_for_selector(".prompt", timeout=120000)
        m.screenshot(path=f"{args.out}/mobile_practice.png", full_page=True)

        # every template's own solution, graded inside Pyodide
        if args.selftest_seeds:
            pg.goto(BASE + "/#/numpy/1.01")
            pg.wait_for_selector(".prompt", timeout=120000)
            t0 = time.time()
            rep = pg.evaluate(f"py.call('selftest', {{section: 'numpy', seeds: {args.selftest_seeds}}})")
            fails = rep["failures"]
            check(not fails, f"Pyodide self-test: {rep['checked']} runs, {len(fails)} failures ({time.time() - t0:.0f}s)")
            for f in fails[:20]:
                print("     ", json.dumps(f)[:400])

        errors = [l for l in logs if "error" in l.lower()]
        check(not errors, f"no console errors {errors[:5]}")
        b.close()
finally:
    if srv:
        srv.terminate()

print(f"\n{'ALL CHECKS PASSED' if not problems else f'{len(problems)} FAILED'}")
sys.exit(1 if problems else 0)

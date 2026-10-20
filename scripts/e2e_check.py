"""Browser click-through of the site (needs: pip install playwright && playwright install chromium).

    python scripts/build.py && python scripts/e2e_check.py <screenshot-dir> [base-url]

Without base-url it serves docs/ on 127.0.0.1:8765. Covers: topic list, problem render, syntax
error, Run, infinite-loop time limit, correct -> next, wrong -> reroll, a challenge with hidden
tests, the medal at 250, and answer reveal at 251.
"""
import sys, json, subprocess, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
BASE = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://127.0.0.1:8765"
srv = None if len(sys.argv) > 2 else subprocess.Popen([sys.executable, "-m", "http.server", "8765", "-b", "127.0.0.1", "-d", "docs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        pg.goto(BASE + "/")
        pg.wait_for_selector(".topic")
        pg.screenshot(path=f"{out}/home.png", full_page=True)
        pg.click("a[href='#/numpy']")
        pg.wait_for_selector(".prompt", timeout=120000)
        pg.screenshot(path=f"{out}/practice.png", full_page=True)
        # wrong answer path
        prompt = pg.inner_text(".prompt")
        print("PROMPT:", prompt[:200])
        print("QNUM:", pg.inner_text(".qnum"))
        pg.click(".CodeMirror")
        pg.keyboard.type("1 +")
        pg.click(".submit"); pg.wait_for_function("document.querySelector('.verdict').textContent.length > 0")
        print("SYNTAX:", pg.inner_text(".verdict"), "|", pg.inner_text(".out")[:200])
        # run with print + error
        pg.evaluate("document.querySelector('.CodeMirror').CodeMirror.setValue(\"print('hello')\\nundefined_name\")")
        pg.click(".run"); pg.wait_for_function("!document.querySelector('.out').textContent.includes('Running')")
        print("RUN:", pg.inner_text(".out"))
        # infinite loop -> time limit
        pg.evaluate("document.querySelector('.CodeMirror').CodeMirror.setValue('while True: pass')")
        t0 = time.time(); pg.click(".run"); pg.wait_for_function("!document.querySelector('.out').textContent.includes('Running')", timeout=60000)
        print("TLE:", pg.inner_text(".out"), f"({time.time()-t0:.1f}s)")
        # submit the reference solution (pull from the problem object)
        pg.wait_for_function("!document.querySelector('.run').disabled", timeout=60000)
        sol = pg.evaluate("view.problem.solution")
        pg.evaluate("s => document.querySelector('.CodeMirror').CodeMirror.setValue(s)", sol)
        pg.click(".submit"); pg.wait_for_function("document.querySelector('.verdict').textContent.includes('Correct')", timeout=60000)
        print("CORRECT:", pg.inner_text(".verdict"), "|", pg.inner_text(".verdict-msg"))
        pg.screenshot(path=f"{out}/correct.png", full_page=True)
        pg.click(".next"); pg.wait_for_function("document.querySelector('.qnum').textContent === '2'", timeout=30000)
        print("NEXT QNUM:", pg.inner_text(".qnum"))
        # wrong answer -> same number, reroll
        before = pg.evaluate("view.problem.seed")
        pg.evaluate("document.querySelector('.CodeMirror').CodeMirror.setValue('result = None')")
        pg.click(".submit"); pg.wait_for_function("document.querySelector('.verdict').textContent.includes('Wrong')")
        print("WRONG:", pg.inner_text(".verdict"), "|", pg.inner_text(".verdict-msg"))
        pg.click(".next"); pg.wait_for_function(f"view.problem && view.problem.seed !== {before}")
        print("REROLL QNUM:", pg.inner_text(".qnum"), "tier:", pg.evaluate("view.problem.tier"))
        # jump to a challenge: question 248 with a hacked state, submit reference
        pg.evaluate("""() => { state.topics.numpy = {solved: 249, medal: false, current: {n: 250, seed: 7, tier: null}}; save(); }""")
        pg.goto(BASE + "/#/numpy"); pg.reload()
        pg.wait_for_selector(".prompt", timeout=120000)
        print("Q250 tier:", pg.evaluate("view.problem.tier"), pg.evaluate("view.problem.template"))
        sol = pg.evaluate("view.problem.solution")
        pg.evaluate("s => document.querySelector('.CodeMirror').CodeMirror.setValue(s)", sol)
        t0 = time.time(); pg.click(".submit"); pg.wait_for_function("document.querySelector('.verdict').textContent.length > 0", timeout=60000)
        print("CHALLENGE:", pg.inner_text(".verdict"), "|", pg.inner_text(".verdict-msg"), f"({time.time()-t0:.1f}s)")
        pg.wait_for_selector(".modal", timeout=5000)
        pg.screenshot(path=f"{out}/medal.png")
        pg.click(".modal button")
        pg.click(".next"); pg.wait_for_function("document.querySelector('.qnum').textContent === '251'", timeout=30000)
        print("251 has answer reveal:", pg.locator("details.answer").count(), "example open:", pg.evaluate("document.querySelector('details.example').open"))
        pg.goto(BASE + "/#/"); pg.wait_for_selector(".topic")
        pg.screenshot(path=f"{out}/home_after.png", full_page=True)
        m = b.new_page(viewport={"width": 390, "height": 844}); m.goto(BASE + "/#/numpy"); m.wait_for_selector(".prompt", timeout=120000)
        m.screenshot(path=f"{out}/mobile.png", full_page=True)
        print("LOGS:", [l for l in logs if "error" in l.lower()][:10])
        b.close()
finally:
    if srv:
        srv.terminate()

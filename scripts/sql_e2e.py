"""Browser check of the SQL pages (COMP_GUIDE §10): psql in the live terminal, on postgres-lite in Pyodide.

    python scripts/build.py && python scripts/sql_e2e.py <screenshot-dir> [first module number]

For every module 10.01-10.16: the page starts in psql (or at the bash prompt for 10.15), the Database panel lists the
tables, the reference solution typed on the keyboard is graded correct, and the bar under the terminal says when the
last result matches the target. Then, on one module: a reload restores the session, \\q is refused, a wrong answer is
graded wrong, Reset starts over, \\e opens nano, and the phone layout. Last, every template's reference session is
graded inside Pyodide (skipped when a first module is given).
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sql-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8771"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8771", "-b", "127.0.0.1", "-d", str(ROOT / "docs")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
problems = []

SCREEN = """() => { const b = view.term.mt.term.buffer.active; const out = [];
  for (let i = 0; i < b.length; i++) out.push(b.getLine(i).translateToString(true)); return out.join('\\n'); }"""
IDLE = "view.term && !view.term.inflight && !view.term.queue.length && !view.term.pumpDue"


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        problems.append(what)


def idle(pg, extra=0.3):
    pg.wait_for_function(IDLE, timeout=120000)
    time.sleep(extra)
    pg.wait_for_function(IDLE, timeout=120000)


def type_lines(pg, text):
    """Type each line and press Enter, as a person would (heredocs and multi-line statements included)."""
    for line in text.rstrip("\n").split("\n"):
        if line:
            pg.keyboard.type(line, delay=2)
        pg.keyboard.press("Enter")
        idle(pg, 0.05)
    idle(pg)


def open_module(pg, mod):
    pg.goto(f"{BASE}/#/sql/{mod}")
    try:
        pg.wait_for_function("view && view.term && view.term.alive", timeout=240000)
    except Exception:
        info = pg.evaluate("JSON.stringify({t: view.problem && view.problem.template, s: view.problem && "
                           "view.problem.seed, phase: view.phase})")
        screen = pg.evaluate(SCREEN) if pg.evaluate("!!(view && view.term)") else pg.inner_text(".problem")[-800:]
        raise SystemExit(f"{mod}: the terminal didn't start {info}\n{screen[-1500:]}")
    idle(pg)
    pg.click(".term-host")


def submit(pg):
    pg.click(".submit")
    pg.wait_for_function("(document.querySelector('.verdict').textContent.length > 0 || "
                         "document.querySelector('.verdict-msg').textContent.length > 0) && "
                         "!document.querySelector('.verdict-msg').textContent.startsWith('Checking')", timeout=240000)
    return pg.inner_text(".verdict"), pg.inner_text(".verdict-msg")


try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        logs = []
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type == "error" else None)

        pg.goto(BASE + "/#/sql")
        pg.wait_for_selector(".modules")
        n = pg.locator(".modules a.topic").count()
        check(n == 16, f"the SQL page lists 16 live modules (got {n})")
        pg.screenshot(path=str(OUT / "sql_topic.png"), full_page=True)

        first = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        for i in range(first, 17):
            mod = f"10.{i:02d}"
            t0 = time.time()
            open_module(pg, mod)
            prob = pg.evaluate("view.problem")
            in_psql = any(ln.startswith("#: psql ") for ln in prob["setup"].split("\n"))
            text = pg.inner_text(".problem")
            check("#:" not in text, f"{mod}: the directives are hidden")
            screen = pg.evaluate(SCREEN)
            if in_psql:
                check(f"{prob['sql']['db']}=>" in screen, f"{mod}: the page starts in psql ({screen.strip()[-30:]!r})")
            else:
                check("$" in screen, f"{mod}: the page starts at the bash prompt")
            panel = (pg.inner_text(".dbp") if pg.locator(".dbp").count() else "").lower()
            check("database" in panel and "row" in panel, f"{mod}: the Database panel lists tables ({panel[:60]!r})")
            type_lines(pg, prob["solution"])
            if prob["sql"].get("check") == "query" and in_psql:
                bar = pg.inner_text(".mt-vim") if pg.is_visible(".mt-vim") else ""
                check("matches the target" in bar, f"{mod}: the bar says the result matches ({bar.strip()[:60]!r})")
            pg.screenshot(path=str(OUT / f"sql_{mod}.png"), full_page=True)
            verdict, msg = submit(pg)
            check("Correct" in verdict, f"{mod} {prob['template']}: the reference typed in the page is graded correct "
                                        f"({verdict} {msg[:120]}) [{time.time() - t0:.0f}s]")

        # one module in depth
        mod = "10.03"
        open_module(pg, mod)
        type_lines(pg, "SELECT 'not an answer' AS probe;")     # no problem asks for this
        check("(1 row)" in pg.evaluate(SCREEN), "a query prints its result")
        pg.reload()
        pg.wait_for_function("view && view.term && view.term.alive", timeout=240000)
        idle(pg, 0.5)
        check("(1 row)" in pg.evaluate(SCREEN), "a reload restores the session")
        pg.click(".term-host")
        type_lines(pg, "\\q")
        check("this problem keeps psql open" in pg.evaluate(SCREEN) and
              pg.evaluate("view.term.lastView.prompt").endswith("=> "), "\\q is refused and psql stays open")
        verdict, msg = submit(pg)
        check("Correct" not in verdict, f"a wrong answer isn't accepted ({verdict}: {msg[:80]})")
        if pg.is_visible(".next"):       # a Wrong verdict ends the question: on to the next one
            pg.click(".next")
            pg.wait_for_function("view && view.term && view.term.alive && view.phase === 'answering'",
                                 timeout=240000)
            idle(pg)
            pg.click(".term-host")
            type_lines(pg, "SELECT 'not an answer' AS probe;")
        pg.click(".reset")
        idle(pg, 0.5)
        check("not an answer" not in pg.evaluate(SCREEN), "Reset starts over")
        pg.click(".term-host")
        type_lines(pg, "SELECT 1 AS one")
        check(pg.evaluate("view.term.lastView.prompt").endswith("-> "), "an unfinished statement continues (-> prompt)")
        type_lines(pg, ";")
        type_lines(pg, "\\e")
        check(pg.locator(".nano").count() > 0 or "GNU nano" in pg.evaluate(SCREEN), "\\e opens nano")
        pg.screenshot(path=str(OUT / "sql_editor.png"), full_page=True)

        ph = b.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        ph.goto(f"{BASE}/#/sql/10.01")
        ph.wait_for_function("view && view.term && view.term.alive", timeout=240000)
        time.sleep(1)
        ph.screenshot(path=str(OUT / "sql_phone.png"), full_page=True)
        w = ph.evaluate("document.documentElement.scrollWidth")
        check(w <= 400, f"the phone layout fits (scrollWidth {w})")

        # every template's reference session graded inside Pyodide (10.15 on its own page: numpy + pandas)
        for only, skip, page in (("", "10.15", "10.01"), ("10.15", "", "10.15")):
            if len(sys.argv) > 2:
                break
            pg.goto(f"{BASE}/#/sql/{page}")
            pg.reload()
            pg.wait_for_function("view && view.term && view.term.alive", timeout=240000)
            t0 = time.time()
            rep = pg.evaluate(f"py.call('selftest', {{section: 'sql', seeds: 1, only: '{only}', skip: '{skip}'}}, "
                              "3600000)")
            fails = rep["failures"]
            check(not fails, f"Pyodide self-test sql {only or 'all but ' + skip}: {rep['checked']} runs, {len(fails)} "
                             f"failures ({time.time() - t0:.0f}s)")
            for f in fails[:10]:
                print("     ", str(f)[:300])
        check(not logs, f"no page errors ({logs[:3]})")
        b.close()
finally:
    srv.terminate()

print(f"\n{len(problems)} problem(s)")
sys.exit(1 if problems else 0)

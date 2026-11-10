"""Browser check of the Docker pages (COMP_GUIDE §12): docker-lite in the live terminal, with the Docker panel.

    python scripts/build.py && python scripts/docker_e2e.py <screenshot-dir> [modules...]

On a few modules: the page starts, the reference solution typed in the terminal is graded correct, and the Docker
tab appears with the containers / images it made. Screenshots of each page are saved for a look.
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/docker-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8772"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8772", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
IDLE = "typeof view !== 'undefined' && view && view.term && view.term.alive && !view.term.inflight && !view.term.queue.length"
problems = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


def run(page, module):
    page.goto(f"{BASE}/#/docker/{module}")
    page.wait_for_function(IDLE, timeout=180000)
    page.wait_for_timeout(500)
    sol = page.evaluate("view.problem.solution")
    page.click(".term-host")
    for line in sol.split("\n"):
        page.keyboard.type(line, delay=2)
        page.keyboard.press("Enter")
        page.wait_for_timeout(250)
        page.wait_for_function(IDLE, timeout=60000)
    page.wait_for_timeout(1200)
    tab = page.evaluate("!document.querySelector('.sp-tab[data-tab=\"docker\"]').hidden")
    check(tab, f"{module}: the Docker tab is shown")
    if tab:
        page.click('.sp-tab[data-tab="docker"]')
        page.wait_for_timeout(200)
    rows = page.evaluate("document.querySelectorAll('[data-pane=\"docker\"] .dk-c').length")
    page.screenshot(path=str(OUT / f"{module}-docker.png"), full_page=True)
    page.click(".submit")
    page.wait_for_function("document.querySelector('.verdict').textContent.trim().length > 0", timeout=120000)
    page.wait_for_timeout(500)
    verdict = page.inner_text(".verdict")
    check("correct" in verdict.lower() and "incorrect" not in verdict.lower() and "not" not in verdict.lower(),
          f"{module}: the reference is graded correct ({verdict.strip()[:60]!r}); container rows {rows}")


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    for m in sys.argv[2:] or ["12.02", "12.05", "12.10"]:
        try:
            run(page, m)
        except Exception as e:  # noqa: BLE001
            check(False, f"{m}: {type(e).__name__}: {str(e)[:200]}")
            page.screenshot(path=str(OUT / f"{m}-error.png"), full_page=True)
    check(not errors, f"no page errors ({errors[:3]})")
    browser.close()
srv.terminate()
print(f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)

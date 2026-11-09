"""Browser check of the Git pages (COMP_GUIDE §8): git and GitHub-lite in the live terminal, with the Git and GitHub
panels.

    python scripts/build.py && python scripts/git_e2e.py <screenshot-dir>

On a few modules: the page starts, the Git / GitHub tabs appear, the reference solution typed in the terminal is
graded correct, and the panels show the repository (commit graph, branches) and the GitHub side. Screenshots of
each page are saved for a look.
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/git-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8771"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8771", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
IDLE = "view.term && view.term.alive && !view.term.inflight && !view.term.queue.length"
problems = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


def run(page, module):
    page.goto(f"{BASE}/#/git/{module}")
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
    tabs = page.evaluate("!document.querySelector('.sp-tabs').hidden")
    check(tabs, f"{module}: the Files / Git / GitHub tabs are shown")
    page.click('.sp-tab[data-tab="git"]')
    page.wait_for_timeout(200)
    graph = page.evaluate("document.querySelectorAll('.gp-row').length")
    page.screenshot(path=str(OUT / f"{module}-git.png"), full_page=True)
    gh_tab = page.evaluate("!document.querySelector('.sp-tab[data-tab=\"github\"]').hidden")
    if gh_tab:
        page.click('.sp-tab[data-tab="github"]')
        page.wait_for_timeout(200)
        page.screenshot(path=str(OUT / f"{module}-github.png"), full_page=True)
    page.click(".submit")
    page.wait_for_function("document.querySelector('.verdict').textContent.trim().length > 0", timeout=120000)
    page.wait_for_timeout(500)
    verdict = page.inner_text(".verdict")
    check("correct" in verdict.lower() and "incorrect" not in verdict.lower() and "not" not in verdict.lower(),
          f"{module}: the reference is graded correct ({verdict.strip()[:60]!r}); graph rows {graph}")


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    for m in sys.argv[2:] or ["8.05", "8.08", "8.13"]:
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

"""Browser check of the LaTeX pages (COMP_GUIDE §11): the editor, Compile, the embedded PDF and Submit.

    python scripts/build.py && python scripts/latex_e2e.py <screenshot-dir>

For every module 11.01-11.18: the page loads a problem with Compile and Submit (no Run), the target preview is
typeset, the reference answer compiles to a page (or slides) without errors and Submit grades it correct. Then, on
one module: a broken answer shows the error list and the pdfTeX log and is graded "doesn't compile", the draft
survives a reload, Next resets the PDF panel, and the phone layout has no sideways scroll.
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/latex-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8771"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8771", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
problems = []
SET_CODE = "s => view.editor.setValue(s)"
READY = "view && view.problem && view.phase === 'answering'"
DONE = "!['', 'compiling\u2026'].includes(document.querySelector('.pdf-status').textContent)"
PDF = ".latex-out .pdf-view"


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


def open_module(pg, mod):
    pg.goto(f"{BASE}/#/latex/{mod}")
    pg.wait_for_function(READY, timeout=180000)


def compile_(pg, code):
    pg.evaluate(SET_CODE, code)
    pg.evaluate("document.querySelector('.pdf-status').textContent = ''")
    pg.click(".compile")
    pg.wait_for_function(DONE, timeout=60000)
    time.sleep(0.3)   # KaTeX typesets after the HTML lands


def submit(pg):
    pg.click(".submit")
    pg.wait_for_function("document.querySelector('.verdict').textContent !== ''", timeout=60000)
    return pg.evaluate("document.querySelector('.verdict').textContent")


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: logs.append(str(e)))

        pg.goto(BASE + "/#/latex")
        pg.wait_for_selector(".modules")
        n = pg.evaluate("document.querySelectorAll('.modules a, .modules .module').length")
        check(n >= 18, f"topic page lists the 18 modules ({n})")
        pg.screenshot(path=str(OUT / "topic.png"), full_page=True)

        for i in range(1, 19):
            mod = f"11.{i:02d}"
            open_module(pg, mod)
            labels = pg.evaluate("[...document.querySelectorAll('.actions button:not([hidden])')].map(b => b.textContent)")
            check("Compile" in labels and "Submit" in labels and "Run" not in labels, f"{mod}: Compile and Submit {labels}")
            tgt = pg.evaluate("(() => { const t = document.querySelector('.latex-target'); if (!t) return null; "
                              "return {math: t.querySelectorAll('.tex-m, .tex-dm').length, "
                              "typeset: t.querySelectorAll('.katex').length, raw: [...t.querySelectorAll('.tex-m, .tex-dm')]"
                              ".filter(e => !e.querySelector('.katex')).length}; })()")
            check(tgt is not None, f"{mod}: the prompt shows the compiled target")
            if tgt is not None:
                check(tgt["raw"] == 0 and (tgt["math"] > 0 or not "11.05" <= mod <= "11.09"),
                      f"{mod}: the target's math is typeset ({tgt['math']} formulas)")
            sol = pg.evaluate("view.problem.solution")
            compile_(pg, sol)
            st = pg.evaluate("document.querySelector('.pdf-status').textContent")
            pages = pg.evaluate(f"document.querySelectorAll('{PDF} .tex-page, {PDF} .tex-slide').length")
            check(st == "compiled" and pages >= 1, f"{mod}: the reference compiles ({st}, {pages} page(s))")
            pg.screenshot(path=str(OUT / f"{mod}.png"))
            v = submit(pg)
            check("Correct" in v, f"{mod}: Submit grades the reference correct ({v})")

        # one module in depth: errors, the log, the verdict, reload, Next, phone
        open_module(pg, "11.03")
        compile_(pg, "This is \\textbff{wrong}.\n")
        st = pg.evaluate("document.querySelector('.pdf-status').textContent")
        prob = pg.evaluate("document.querySelector('.pdf-problems').textContent")
        log = pg.evaluate("document.querySelector('.pdf-log pre').textContent")
        check(st.startswith("1 error"), f"a typo shows one error ({st})")
        check("Undefined control sequence" in prob, "the problem list names the error")
        check("! Undefined control sequence." in log and "l.1" in log, "the log reads like pdfTeX's")
        pg.screenshot(path=str(OUT / "error.png"))
        v = submit(pg)
        msg = pg.evaluate("document.querySelector('.verdict-msg').textContent")
        check("Correct" not in v, f"a broken answer isn't correct ({v}: {msg[:80]})")
        pg.evaluate(SET_CODE, "Draft text.")
        time.sleep(0.5)
        pg.reload()
        pg.wait_for_function(READY, timeout=180000)
        check(pg.evaluate("view.editor.getValue()") == "Draft text.", "the draft survives a reload")
        compile_(pg, pg.evaluate("view.problem.solution"))
        submit(pg)
        pg.click(".next")
        pg.wait_for_function(READY, timeout=60000)
        check(pg.evaluate(f"!!document.querySelector('{PDF} .pdf-empty')"), "Next resets the PDF panel")
        check(pg.evaluate("!!document.querySelector('.latex-target .tex-page')"), "... and the target preview stays")

        m = b.new_page(viewport={"width": 390, "height": 844})
        m.goto(f"{BASE}/#/latex/11.16")
        m.wait_for_function(READY, timeout=180000)
        m.evaluate(SET_CODE, m.evaluate("view.problem.solution"))
        m.click(".compile")
        m.wait_for_function(DONE, timeout=60000)
        time.sleep(0.5)
        wide = m.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
        check(wide <= 1, f"phone: no sideways scroll ({wide}px)")
        check(m.evaluate(f"document.querySelectorAll('{PDF} svg').length") >= 1, "phone: the TikZ picture renders")
        m.screenshot(path=str(OUT / "mobile.png"), full_page=True)

        errors = [x for x in logs if "favicon" not in x]
        check(not errors, f"no console errors {errors[:5]}")
        b.close()
finally:
    srv.terminate()

print(f"\n{'ALL CHECKS PASSED' if not problems else f'{len(problems)} FAILED'}")
sys.exit(1 if problems else 0)

"""Browser check of the Vim pages (COMP_GUIDE §9): Vim in the live terminal.

    python scripts/build.py && python scripts/vim_e2e.py <screenshot-dir>

For every module 9.01-9.12: the page starts (in Vim, or at the prompt for problems that start there), the setup is
hidden, the reference keys typed on the keyboard are graded correct, and the bar under the terminal counts the keys.
Then, on one module: a reload restores Vim and its text, quitting is refused, Reset starts over, a wrong answer is
graded wrong, and the phone layout.
"""

import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.bashlite.vim import keys as VK  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vim-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8769"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8769", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
problems = []

SCREEN = """() => { const b = view.term.mt.term.buffer.active; const out = [];
  for (let i = 0; i < b.length; i++) out.push(b.getLine(i).translateToString(true)); return out.join('\\n'); }"""
IDLE = "view.term && !view.term.inflight && !view.term.queue.length"
KEYS = {"<Esc>": "Escape", "<CR>": "Enter", "<BS>": "Backspace", "<Tab>": "Tab", "<Up>": "ArrowUp",
        "<Down>": "ArrowDown", "<Left>": "ArrowLeft", "<Right>": "ArrowRight", "<Del>": "Delete",
        "<C-^>": "Control+6"}


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


def press(pg, keys):
    """Type Vim-notation keys on the keyboard, as a person would."""
    run = []

    def flush():
        if run:
            pg.keyboard.type("".join(run), delay=5)
            run.clear()
    for tok in VK.parse(keys):
        if len(tok) == 1:
            run.append(tok)
            continue
        flush()
        if tok in KEYS:
            pg.keyboard.press(KEYS[tok])
        elif tok.startswith("<C-") and len(tok) == 5:
            pg.keyboard.press("Control+" + tok[3])
        else:
            raise ValueError(tok)
        time.sleep(0.02)
    flush()
    pg.wait_for_function(IDLE, timeout=30000)
    time.sleep(0.3)
    pg.wait_for_function(IDLE, timeout=30000)


def open_module(pg, mod):
    pg.goto(f"{BASE}/#/vim/{mod}")
    pg.wait_for_function("view && view.term && view.term.alive", timeout=180000)
    pg.wait_for_function(IDLE, timeout=30000)
    time.sleep(0.3)
    pg.click(".term-host")


def submit(pg):
    pg.click(".submit")
    pg.wait_for_function("document.querySelector('.verdict').textContent.length > 0 || "
                         "document.querySelector('.verdict-msg').textContent.length > 0 && "
                         "!document.querySelector('.verdict-msg').textContent.startsWith('Checking')", timeout=180000)
    return pg.inner_text(".verdict"), pg.inner_text(".verdict-msg")


try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        logs = []
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type == "error" else None)

        pg.goto(BASE + "/#/vim")
        pg.wait_for_selector(".modules")
        n = pg.locator(".modules a.topic").count()
        check(n == 12, f"the Vim page lists 12 live modules (got {n})")
        pg.screenshot(path=str(OUT / "vim_topic.png"), full_page=True)

        for i in range(1, 13):
            mod = f"9.{i:02d}"
            open_module(pg, mod)
            prob = pg.evaluate("view.problem")
            starts_in_vim = any(ln.startswith("#: vim ") for ln in prob["setup"].split("\n"))
            text = pg.inner_text(".problem")
            check("#:" not in text and "Setup" not in text, f"{mod}: the setup is hidden")
            screen = pg.evaluate(SCREEN)
            if starts_in_vim:
                check(pg.evaluate("view.term.lastView.mode") == "vim", f"{mod}: the page starts in Vim")
                check(pg.is_visible(".mt-vim"), f"{mod}: the keys bar is shown")
            else:
                check("$" in screen, f"{mod}: the page starts at the prompt")
            press(pg, prob["solution"])
            if pg.evaluate("view.term.lastView.mode") == "vim":
                bar = pg.inner_text(".mt-vim")
                check("key" in bar, f"{mod}: the bar counts the keys ({bar.strip()[:40]})")
            pg.screenshot(path=str(OUT / f"vim_{mod}.png"), full_page=True)
            verdict, msg = submit(pg)
            check("Correct" in verdict, f"{mod} {prob['template']}: the reference keys typed in the page are graded "
                                        f"correct ({verdict} {msg[:100]})")

        # one module in depth: reload, quitting refused, wrong answer, Reset
        mod = "9.03"
        open_module(pg, mod)
        prob = pg.evaluate("view.problem")
        press(pg, "ggOimport os<Esc>")
        before = pg.evaluate(SCREEN)
        check("import os" in before, "typing edits the buffer on the screen")
        pg.reload()
        pg.wait_for_function("view && view.term && view.term.alive", timeout=180000)
        pg.wait_for_function(IDLE, timeout=30000)
        time.sleep(0.5)
        check("import os" in pg.evaluate(SCREEN) and pg.evaluate("view.term.lastView.mode") == "vim",
              "a reload restores Vim and the edit")
        pg.click(".term-host")
        press(pg, ":q!<CR>")
        check("Quitting is turned off" in pg.evaluate(SCREEN) and pg.evaluate("view.term.lastView.mode") == "vim",
              ":q! is refused and Vim stays open")
        verdict, msg = submit(pg)
        check("Wrong" in verdict or "Not yet" in verdict, f"a wrong answer isn't accepted ({verdict}: {msg[:80]})")
        pg.click(".reset")
        pg.wait_for_function("view && view.term && view.term.alive", timeout=180000)
        pg.wait_for_function(IDLE, timeout=30000)
        time.sleep(0.5)
        check(pg.evaluate("view.term.lastView.vim.keys") == 0, "Reset starts over (no keys counted)")

        ph = b.new_page(viewport={"width": 390, "height": 844})
        ph.goto(f"{BASE}/#/vim/9.02")
        ph.wait_for_function("view && view.term && view.term.alive", timeout=180000)
        time.sleep(0.5)
        wide = ph.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        check(wide, "phone: no horizontal page scroll")
        ph.screenshot(path=str(OUT / "vim_phone.png"), full_page=True)
        check(not logs, "no page errors" + (f": {logs[:3]}" if logs else ""))
        b.close()
finally:
    srv.terminate()

print(f"\n{len(problems)} problem(s)")
sys.exit(1 if problems else 0)

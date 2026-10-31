"""Browser click-through of Part III (the Data Structures & Algorithms tutorials, docs/dsa/).

Compiler Explorer is not called: requests to godbolt.org are answered by compiling locally (clang++, javac/java,
rustc), in the same JSON shape, so the page's request/response handling is exercised without sending code anywhere.
Python runs in the page's real Pyodide worker.

    python scripts/build.py && python scripts/dsa_e2e.py <screenshot-dir>

Covers: the home page's Part III rows and status labels, options updating the complexity boxes and the diagram,
every step of the linked-list tutorial in Python, C++, Java and Rust (reference code passes, wrong code fails with the
first differing line, empty and non-compiling code), Run, the final challenge, Completed, Reset (cancel and confirm),
reload keeping progress, and the phone layout.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dsa-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8766"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8766", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
problems = []
ce_calls = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


def sh(argv, cwd):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60)


def fake_compiler_explorer(route):
    """Answer a Compiler Explorer execute request by compiling and running locally."""
    req = route.request
    cid = req.url.rstrip("/").split("/")[-2]
    body = json.loads(req.post_data)
    ce_calls.append(cid)
    src = body["source"]
    lines = lambda s: [{"text": t} for t in s.splitlines()]
    with tempfile.TemporaryDirectory() as d:
        if cid.startswith("g"):
            Path(d, "example.cpp").write_text(src)
            b = sh(["clang++", "-std=c++20", "-O1", "-w", "-o", "prog", "example.cpp"], d)
            run = ["./prog"]
        elif cid.startswith("java"):
            Path(d, "example.java").write_text(src)
            b = sh(["javac", "-nowarn", "-d", ".", "example.java"], d)
            main = "TLTests" if "class TLTests" in src else next(
                (ln.split("class ")[1].split()[0].split("{")[0] for ln in src.splitlines() if "class " in ln and "static void main" in src), "Main")
            run = ["java", "-cp", ".", main]
        else:
            Path(d, "example.rs").write_text(src)
            b = sh(["rustc", "--edition", "2021", "-A", "warnings", "-o", "prog", "example.rs"], d)
            run = ["./prog"]
        if b.returncode:
            res = {"code": -1, "didExecute": False, "stdout": [], "stderr": [{"text": "Build failed"}],
                   "buildResult": {"code": b.returncode, "stdout": lines(b.stdout), "stderr": lines(b.stderr)}}
        else:
            r = subprocess.run(run, cwd=d, capture_output=True, text=True, timeout=20)
            res = {"code": r.returncode, "didExecute": True, "stdout": lines(r.stdout), "stderr": lines(r.stderr),
                   "timedOut": False, "buildResult": {"code": 0, "stdout": [], "stderr": []}}
    route.fulfill(status=200, content_type="application/json", body=json.dumps(res),
                  headers={"Access-Control-Allow-Origin": "*"})


SET = "s => document.querySelector('.CodeMirror').CodeMirror.setValue(s)"
VERDICT = "document.querySelector('.verdict').textContent"
STATUS = "document.querySelector('.bar .dsa-status').textContent"  # (textContent: not uppercased by CSS)


def wait_verdict(pg, timeout=120000):
    pg.wait_for_function(f"{VERDICT}.trim().length > 0", timeout=timeout)
    return pg.evaluate(VERDICT)


def program(pg, upto):
    return pg.evaluate(f"""() => {{ const s = JSON.parse(localStorage.getItem('training-loop:dsa:v1')).tutorials['linked-lists'];
        return DSATutorials['linked-lists'].program(s.opts, {upto}); }}""")


def choose(pg, opts):
    for k, v in opts.items():
        pg.click(f".opt[data-opt='{k}'] button[data-v='{v}']")


def run_all_steps(pg, label, wrong_at=None):
    for k in range(1, 9):
        pg.wait_for_selector(".check")
        if wrong_at == k:
            # drop one line of the reference: the step must fail and say where
            code = program(pg, k).replace("L.size += 1", "L.size += 2").replace("L->size += 1", "L->size += 2")
            pg.evaluate(SET, code)
            pg.click(".check")
            v = wait_verdict(pg)
            out = pg.inner_text(".out")
            check("Not yet" in v and "expected" in out and "got" in out, f"{label}: a wrong step {k} fails and shows the first differing line")
            pg.screenshot(path=str(OUT / f"{label}-wrong.png"), full_page=True)
        pg.evaluate(SET, program(pg, k))
        pg.click(".check")
        v = wait_verdict(pg)
        ok = f"Step {k} passed" in v
        check(ok, f"{label}: step {k} passes with the reference code")
        if not ok:
            print(pg.inner_text(".out")[:3000])
            return False
        if k == 1:
            pg.screenshot(path=str(OUT / f"{label}-step1.png"), full_page=True)
        pg.click(".next")
    return True


try:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        pg.route("https://godbolt.org/api/**", fake_compiler_explorer)

        # ---- home: Part III
        pg.goto(BASE + "/")
        pg.evaluate("localStorage.removeItem('training-loop:dsa:v1')")
        pg.reload()
        pg.wait_for_selector(".part h2")
        heads = pg.locator(".part h2").all_inner_texts()
        check(any("part iii" in h.lower() and "data structures" in h.lower() for h in heads), "home has Part III · Data Structures & Algorithms")
        row = pg.locator("a.topic[href='#/dsa/linked-lists']")
        check(row.count() == 1 and "not started" in row.inner_text().lower(), "Linked lists row says Not Started")
        check(pg.locator(".part:last-child .topic.soon").count() == 11, "the other 11 tutorials say soon")
        pg.screenshot(path=str(OUT / "home.png"), full_page=True)

        # ---- setup: options and live complexity
        row.click()
        pg.wait_for_selector(".cx-grid .cx")
        n_cx = pg.locator(".cx-grid .cx").count()
        check(n_cx >= 10, f"setup shows the complexity boxes ({n_cx})")
        last = lambda: pg.inner_text(".cx[data-id='delete_last'] .cx-big")
        check(last() == "O(n)", "singly: delete last is O(n)")
        choose(pg, {"links": "doubly"})
        check(last() == "O(1)" and "changed" in pg.get_attribute(".cx[data-id='delete_last']", "class"),
              "doubly: delete last becomes O(1) and the box flashes")
        check("⇄" in pg.inner_text(".diagram"), "the diagram shows double links")
        choose(pg, {"end": "dummy", "shape": "circular"})
        check("dummy" in pg.inner_text(".diagram") and "back to the dummy" in pg.inner_text(".diagram"), "the diagram shows a circular list with a dummy")
        pg.screenshot(path=str(OUT / "setup.png"), full_page=True)
        choose(pg, {"lang": "py", "type": "int", "end": "null", "links": "singly", "shape": "linear"})
        pg.click(".begin")
        pg.wait_for_selector(".micro")
        check(pg.evaluate(STATUS) == "In Progress", "Begin -> In Progress")
        check(pg.locator(".outline li").count() == 8, "the outline lists 8 steps")
        check("class Node" in pg.inner_text(".problem"), "step 1 shows the Python node code")

        # empty and broken code
        pg.click(".check")
        check("Not yet" in wait_verdict(pg), "Check with an empty editor: Not yet")
        pg.evaluate(SET, "class Node:\n    def __init__(self, data)\n")
        pg.click(".check")
        check("Doesn't compile" in wait_verdict(pg, 180000), "a syntax error: Doesn't compile")
        pg.evaluate(SET, "print('hello')\n")
        pg.click(".run")
        pg.wait_for_function("document.querySelector('.out').textContent.includes('hello')", timeout=60000)
        check(True, "Run prints the file's own output")

        # ---- Python: every step, one wrong attempt, then the final
        if run_all_steps(pg, "py", wrong_at=3):
            check(pg.evaluate(STATUS) == "Almost!", "all steps passed -> Almost!")
            pg.reload()
            pg.wait_for_selector(".problem")
            check(pg.evaluate(STATUS) == "Almost!", "reload keeps the progress")
            pg.wait_for_selector(".submit")
            check(pg.evaluate("document.querySelector('.CodeMirror').CodeMirror.getValue()") == "", "the final editor starts blank")
            check("from a blank file" in pg.inner_text(".problem") and "insert_front" in pg.inner_text(".problem"), "the final shows only the problem")
            pg.screenshot(path=str(OUT / "py-final.png"), full_page=True)
            pg.evaluate(SET, program(pg, 8))
            pg.click(".submit")
            v = wait_verdict(pg)
            check("Completed" in v, "the final passes with the reference")
            pg.wait_for_selector(".modal")
            check("Completed" in pg.inner_text(".modal"), "the Completed modal")
            pg.click(".modal button")
            check(pg.evaluate(STATUS) == "Completed", "status Completed")
            pg.goto(BASE + "/#/")
            pg.wait_for_selector("a.topic[href='#/dsa/linked-lists']")
            check("completed" in pg.inner_text("a.topic[href='#/dsa/linked-lists']").lower(), "home row says Completed")

        # ---- Reset: cancel, then confirm
        pg.goto(BASE + "/#/dsa/linked-lists")
        pg.wait_for_selector(".reset")
        pg.click(".reset")
        pg.click(".modal .cancel")
        check(pg.evaluate(STATUS) == "Completed", "Reset → Cancel keeps everything")
        pg.click(".reset")
        pg.click(".modal .confirm")
        pg.wait_for_selector(".begin")
        check(pg.evaluate(STATUS) == "Not Started", "Reset → confirm: Not Started, back to the options")

        # ---- C++, Java, Rust (Compiler Explorer answered locally)
        for lang, opts in [("cpp", {"type": "generic", "end": "dummy", "links": "doubly", "shape": "circular"}),
                           ("java", {"type": "generic", "end": "null", "links": "doubly", "shape": "linear"}),
                           ("rs", {"type": "int", "end": "dummy", "links": "singly", "shape": "linear"})]:
            pg.goto(BASE + "/#/dsa/linked-lists")
            pg.wait_for_selector(".begin, .reset")
            if pg.locator(".reset").count():
                pg.click(".reset"); pg.click(".modal .confirm"); pg.wait_for_selector(".begin")
            choose(pg, {"lang": lang, **opts})
            pg.click(".begin")
            pg.wait_for_selector(".micro")
            if lang == "cpp":
                pg.evaluate(SET, "int f() { return 1 }\n")
                pg.click(".check")
                check("Doesn't compile" in wait_verdict(pg), "C++: a compile error is shown")
                pg.evaluate(SET, "#include <cstdio>\nint x;\n")
                pg.click(".run")
                pg.wait_for_function("document.querySelector('.out').textContent.includes('no ')", timeout=10000)
                check("main" in pg.inner_text(".out"), "C++: Run without a main explains why nothing ran")
            calls = len(ce_calls)
            if run_all_steps(pg, lang, wrong_at=4 if lang == "rs" else None):
                check(len(ce_calls) > calls, f"{lang}: the page called Compiler Explorer")
                pg.wait_for_selector(".submit")
                pg.evaluate(SET, program(pg, 8))
                pg.click(".submit")
                check("Completed" in wait_verdict(pg), f"{lang}: the final passes")
                pg.screenshot(path=str(OUT / f"{lang}-final-done.png"), full_page=True)
                if pg.locator(".modal button").count():
                    pg.click(".modal button")

        # ---- phone layout
        ph = b.new_page(viewport={"width": 390, "height": 844})
        ph.goto(BASE + "/#/dsa/linked-lists")
        ph.wait_for_selector(".problem")
        wide = ph.evaluate("document.documentElement.scrollWidth")
        check(wide <= 392, f"phone: no horizontal scroll ({wide}px)")
        ph.screenshot(path=str(OUT / "phone.png"), full_page=True)

        errs = [l for l in logs if "error" in l.lower()]
        check(not errs, "no console errors" + (": " + "; ".join(errs[:5]) if errs else ""))
        b.close()
finally:
    srv.terminate()

print(f"\n{'ALL CHECKS PASSED' if not problems else str(len(problems)) + ' FAILED'}; Compiler Explorer calls answered locally: {len(ce_calls)}")
sys.exit(1 if problems else 0)

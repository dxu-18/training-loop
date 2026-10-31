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


TID = ["linked-lists"]   # the tutorial being clicked through


def program(pg, upto):
    return pg.evaluate(f"""() => {{ const s = JSON.parse(localStorage.getItem('training-loop:dsa:v1')).tutorials['{TID[0]}'];
        return DSATutorials['{TID[0]}'].program(s.opts, {upto}); }}""")


def step_count(pg):
    return pg.evaluate(f"""() => {{ const s = JSON.parse(localStorage.getItem('training-loop:dsa:v1')).tutorials['{TID[0]}'];
        return DSATutorials['{TID[0]}'].stepCount(s.opts); }}""")


def start(pg, tid, opts):
    """Open a tutorial from its options page (resetting it first) and begin with these options."""
    TID[0] = tid
    pg.goto(BASE + f"/#/dsa/{tid}")
    pg.wait_for_selector(".begin, .reset")
    if pg.locator(".reset").count():
        pg.click(".reset"); pg.click(".modal .confirm"); pg.wait_for_selector(".begin")
    choose(pg, opts)
    pg.click(".begin")
    pg.wait_for_selector(".check")


def finish(pg, label):
    pg.wait_for_selector(".submit")
    pg.evaluate(SET, program(pg, step_count(pg)))
    pg.click(".submit")
    check("Completed" in wait_verdict(pg), f"{label}: the final passes")
    if pg.locator(".modal button").count():
        pg.click(".modal button")


def choose(pg, opts):
    for k, v in opts.items():
        pg.click(f".opt[data-opt='{k}'] button[data-v='{v}']")


def run_all_steps(pg, label, wrong_at=None):
    for k in range(1, step_count(pg) + 1):
        pg.wait_for_selector(".check")
        if wrong_at == k:
            # break one line of the reference (the newest "size += 1"): the step must fail and say where
            code = program(pg, k)
            bad, good = ("size += 1", "size += 2") if "size += 1" in code else ("mid + 1", "mid + 2")
            i = code.rfind(bad)
            code = code[:i] + good + code[i + len(bad):]
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
        check(pg.locator(".part:last-child .topic.soon").count() == 8, "the other 8 tutorials say soon")
        first = pg.locator(".part:last-child li").first.inner_text()
        check("13.01" in first and "Binary search" in first, "Part III starts with 13.01 Binary search")
        check("Mergesort" in pg.inner_text(".part:last-child") and "Merge sort" not in pg.inner_text(".part:last-child"), "Mergesort is one word")
        for tid in ("binary-search", "stacks", "queues"):
            r = pg.locator(f"a.topic[href='#/dsa/{tid}']")
            check(r.count() == 1 and "not started" in r.inner_text().lower(), f"{tid}: live and Not Started")
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
        pg.wait_for_selector(".check")
        check(pg.evaluate(STATUS) == "In Progress", "Begin -> In Progress")
        check(pg.locator(".outline li").count() == 8, "the outline lists 8 steps")
        check(pg.locator(".all-micro").count() == 1 and not pg.get_attribute(".all-micro", "open") is not None,
              "step 1: the micro-steps are folded away under the overview")
        pg.wait_for_selector(".coach[data-k='0']")
        check("class Node" in pg.inner_text(".coach") and "1 /" in pg.inner_text(".coach"), "the first hint bubble asks for class Node")
        pg.evaluate(SET, "class Node:\n    ")
        pg.wait_for_selector(".coach[data-k='1']:not(.leaving)")
        check("__init__" in pg.inner_text(".coach:not(.leaving)"), "typing it moves the bubble to the next micro-step")
        pg.screenshot(path=str(OUT / "coach.png"), full_page=False)
        pg.evaluate(SET, "")
        pg.wait_for_selector(".coach[data-k='0']:not(.leaving)")
        check(True, "deleting the code brings the earlier bubble back")
        pg.evaluate(SET, program(pg, 1))
        pg.wait_for_selector(".coach.done:not(.leaving)")
        check("Check" in pg.inner_text(".coach.done"), "the whole step typed: the bubble says to press Check")
        # every step of every layout and language: its reference satisfies all its bubbles, in a monotonic order
        bad = pg.evaluate(r"""() => {
          const C = DSAPage._coach, bad = [];
          for (const T of Object.values(DSATutorials)) {
          let combos = [{}];
          for (const o of T.OPTIONS) combos = combos.flatMap((c) => o.choices.map(([v]) => ({ ...c, [o.id]: v })));
          for (const o of combos) {
            const steps = T.render(o);
            for (let k = 0; k < steps.length; k++) {
              const ts = C.coachTargets(steps[k].items);
              ts.forEach((t) => { t.want = C.coachWant(t, o.lang); });
              const before = k ? T.program(o, k) : "", after = T.program(o, k + 1);
              const p0 = C.coachProgress(ts, before, o.lang, new Set()).k, p1 = C.coachProgress(ts, after, o.lang, new Set()).k;
              if (p1 !== ts.length) bad.push(`${T.id} ${JSON.stringify(o)} step ${k + 1}: reference reaches ${p1}/${ts.length}`);
              if (p0 !== 0) bad.push(`${T.id} ${JSON.stringify(o)} step ${k + 1}: ${p0} hints done before typing`);
              const lines = after.split("\n");
              let prev = 0;
              for (let i = 1; i <= lines.length; i++) {
                const q = C.coachProgress(ts, lines.slice(0, i).join("\n"), o.lang, new Set()).k;
                if (q < prev) { bad.push(`${T.id} ${JSON.stringify(o)} step ${k + 1}: progress went back at line ${i}`); break; }
                prev = q;
              }
            }
          }
          }
          return bad;
        }""")
        check(not bad, "every tutorial x option x language x step: the reference satisfies all its hint bubbles, in order" + (": " + "; ".join(bad[:4]) if bad else ""))

        # empty and broken code
        pg.evaluate(SET, "")
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
            pg.wait_for_selector(".check")
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

        # ---- 13.01 Binary search: the style option, then four languages
        pg.goto(BASE + "/#/dsa/binary-search")
        pg.wait_for_selector(".begin")
        TID[0] = "binary-search"
        space = lambda: pg.inner_text(".cx[data-id='space'] .cx-big")
        choose(pg, {"style": "iterative"})
        check(space() == "O(1)", "binary search, iterative: O(1) extra space")
        choose(pg, {"style": "recursive"})
        check(space() == "O(log n)", "recursive: O(log n) stack space")
        check("hi = n" in pg.inner_text(".diagram"), "the diagram shows the segment [lo, hi)")
        pg.screenshot(path=str(OUT / "bs-setup.png"), full_page=True)
        start(pg, "binary-search", {"lang": "py", "type": "int", "style": "iterative"})
        check(pg.locator(".outline li").count() == 4, "binary search: 4 steps")
        if run_all_steps(pg, "bs-py", wrong_at=3):
            finish(pg, "bs-py")
        for lang, typ, style in (("cpp", "generic", "recursive"), ("rs", "generic", "iterative"), ("java", "generic", "recursive")):
            start(pg, "binary-search", {"lang": lang, "type": typ, "style": style})
            if run_all_steps(pg, f"bs-{lang}"):
                finish(pg, f"bs-{lang}")

        # ---- 13.05 Stacks: Python (with a wrong step) and C++ generic
        start(pg, "stacks", {"lang": "py", "type": "int"})
        check(pg.evaluate(STATUS) == "In Progress" and pg.locator(".outline li").count() == 5, "stacks: 5 steps, In Progress")
        if run_all_steps(pg, "stack-py", wrong_at=3):
            check(pg.evaluate(STATUS) == "Almost!", "stacks: all steps -> Almost!")
            finish(pg, "stack-py")
            check(pg.evaluate(STATUS) == "Completed", "stacks: Completed")
        start(pg, "stacks", {"lang": "cpp", "type": "generic"})
        if run_all_steps(pg, "stack-cpp"):
            finish(pg, "stack-cpp")

        # ---- 13.06 Queues: the options change the steps and the complexity
        pg.goto(BASE + "/#/dsa/queues")
        pg.wait_for_selector(".begin")
        TID[0] = "queues"
        choose(pg, {"impl": "list"})
        n_list = pg.locator(".dsa-preview li").count()
        worst = lambda: pg.inner_text(".cx[data-id='deq_worst'] .cx-big")
        check(n_list == 5 and worst() == "O(1)", "queues on a list: 5 steps, every deq O(1)")
        choose(pg, {"impl": "stacks"})
        check(pg.locator(".dsa-preview li").count() == 7 and worst() == "O(n)" and "amortized" in pg.inner_text(".cx[data-id='deq']"),
              "queues on two stacks: 7 steps, deq amortized O(1), one deq O(n)")
        check("outbox" in pg.inner_text(".diagram") and "inbox" in pg.inner_text(".diagram"), "the two-stack diagram shows both stacks")
        pg.screenshot(path=str(OUT / "queue-setup.png"), full_page=True)
        start(pg, "queues", {"lang": "py", "type": "generic", "impl": "stacks"})
        if run_all_steps(pg, "queue2-py", wrong_at=3):
            finish(pg, "queue2-py")
        start(pg, "queues", {"lang": "rs", "type": "generic", "impl": "list"})
        if run_all_steps(pg, "queue-rs", wrong_at=3):
            finish(pg, "queue-rs")
        start(pg, "queues", {"lang": "java", "type": "int", "impl": "stacks"})
        if run_all_steps(pg, "queue2-java"):
            finish(pg, "queue2-java")
        pg.goto(BASE + "/#/")
        pg.wait_for_selector("a.topic[href='#/dsa/queues']")
        check("completed" in pg.inner_text("a.topic[href='#/dsa/queues']").lower() and "completed" in pg.inner_text("a.topic[href='#/dsa/stacks']").lower(),
              "home: Stacks and Queues say Completed")

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

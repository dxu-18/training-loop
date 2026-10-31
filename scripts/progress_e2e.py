"""Browser check of progress files (docs/progress.js): the home page's Download and Upload.

    python scripts/progress_e2e.py <screenshot-dir>

Covers: the buttons in the page's top right corner (desktop and phone); Download through the browser's Save dialog
(stubbed: a real one can't be clicked here) and through the name box browsers without one get, with the downloaded
file checked against the browser's storage; the upload picker accepting only .tlsave; restoring into an empty browser,
then Merge, Replace and Cancel into one that has progress; the pages reading restored progress; and files that are
rejected (not JSON, not a progress file, a newer version, damaged).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/progress-shots")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8768"
srv = subprocess.Popen([sys.executable, "-m", "http.server", "8768", "-b", "127.0.0.1", "-d", "docs"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
problems = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        problems.append(what)


STORAGE = """() => Object.fromEntries(Object.keys(localStorage).filter(k => k.startsWith('training-loop:') && k !== 'training-loop:v1')
                                     .map(k => [k, JSON.parse(localStorage.getItem(k))]))"""
HEADS = "[...document.querySelectorAll('.save-cmp th')].slice(1).map(t => t.textContent)"   # (not uppercased by CSS)
NO_PICKER = "delete window.showSaveFilePicker; delete Window.prototype.showSaveFilePicker;"
FAKE_PICKER = """window.showSaveFilePicker = async (opts) => {
  window.__pickerOpts = opts;
  if (window.__abort) throw new DOMException('closed', 'AbortError');
  return { name: 'picked.tlsave', createWritable: async () => ({ write: async (t) => { window.__written = t; }, close: async () => {} }) };
};"""


def module(solved, seed, medal=False):
    return {"solved": solved, "medal": medal, "current": {"n": solved + 1, "seed": seed, "tier": None}}


def home(pg):
    pg.goto(BASE + "/#/")
    pg.wait_for_selector(".save-bar")


def count_of(pg, topic):
    return pg.locator(f'a.topic[href="#/{topic}"] .t-count').inner_text()


def upload(pg, path):
    pg.set_input_files(".save-file", str(path))
    pg.wait_for_selector(".modal")


def restore(pg, button):
    """Click Restore / Merge / Replace: the page reloads, then says what it did."""
    with pg.expect_navigation():
        pg.click(button)
    return wait_reload_toast(pg)


def wait_reload_toast(pg):
    pg.wait_for_selector(".save-toast")
    return pg.inner_text(".save-toast")


try:
    with sync_playwright() as p:
        b = p.chromium.launch()

        # ---------------------------------------------------------------- a browser with progress
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        ctx.add_init_script(NO_PICKER)
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        home(pg)
        # Part III: two tutorials begun through the page, one of them then marked finished
        for tid in ("stacks", "queues"):
            pg.goto(BASE + f"/#/dsa/{tid}")
            pg.wait_for_selector(".begin")
            pg.click(".begin")
            pg.wait_for_selector(".check")
        pg.evaluate("""() => {
          const s = JSON.parse(localStorage.getItem('training-loop:dsa:v1'));
          Object.assign(s.tutorials.queues, { phase: 'final', passed: 99, final: { code: 'done', passed: true } });
          localStorage.setItem('training-loop:dsa:v1', JSON.stringify(s));
          localStorage.setItem('training-loop:v2', JSON.stringify({ modules: {
            'numpy/1.01': %s, 'numpy/1.02': %s, 'bash/7.01': %s } }));
          localStorage.setItem('training-loop:term:bash/7.01', JSON.stringify({ seed: 701, log: ['ls\\r'], at: 1 }));
          localStorage.setItem('training-loop:term-font', '15');
          localStorage.setItem('training-loop:v1', '{"old": true}');
        }""" % (json.dumps(module(100, 101, True)), json.dumps(module(7, 102)), json.dumps(module(3, 701))))
        original = pg.evaluate(STORAGE)
        pg.reload()   # (the pages read storage when they load)
        home(pg)
        check(count_of(pg, "numpy") == "107" and count_of(pg, "bash") == "3", "the seeded progress shows on the home page")

        # the buttons: top right corner, Download then Upload
        bar = pg.locator(".save-bar").bounding_box()
        check(abs(bar["x"] + bar["width"] - (1280 - 16)) < 2 and abs(bar["y"] - 16) < 2, "Download/Upload sit in the page's top right corner")
        check(pg.locator(".save-bar .btn").all_inner_texts() == ["Download", "Upload"], "the buttons read Download, Upload")
        check(pg.get_attribute(".save-file", "accept") == ".tlsave", "the upload picker accepts only .tlsave files")
        pg.screenshot(path=str(OUT / "home.png"))

        # ---------------------------------------------------------------- Download without a Save dialog: the name box
        pg.click(".save-dl")
        pg.wait_for_selector(".modal .save-name input")
        today = pg.evaluate("() => { const d = new Date(); return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0'); }")
        check(pg.input_value(".save-name input") == f"training-loop-progress-{today}", "the name box suggests training-loop-progress-<date>")
        check(pg.evaluate("document.activeElement === document.querySelector('.save-name input')"), "the name box has the focus")
        pg.screenshot(path=str(OUT / "name-box.png"))
        pg.fill(".save-name input", "my backup")
        with pg.expect_download() as dl:
            pg.click(".modal .confirm")
        d = dl.value
        check(d.suggested_filename == "my backup.tlsave", f"the file gets the typed name plus .tlsave ({d.suggested_filename})")
        saved = Path(d.path()).read_text()
        doc = json.loads(saved)
        check(doc["format"] == "training-loop-progress" and doc["version"] == 1 and doc["site"].startswith(BASE), "the file is a version 1 progress file")
        check(doc["stores"] == original, "the file holds exactly this browser's stores")
        check("training-loop:v1" not in doc["stores"], "the retired v1 counter isn't saved")
        check(all(line.startswith('    "training-loop:') for line in saved.splitlines()[6:-2]), "one store per line")
        check(wait_reload_toast(pg).startswith("Downloaded my backup.tlsave"), "a toast confirms the download")
        backup = OUT / "my backup.tlsave"
        backup.write_text(saved)

        # names: .tlsave typed again isn't doubled, path characters are replaced, empty falls back to the suggestion
        for typed, want in (("week1.tlsave", "week1.tlsave"), ("a/b:c", "a-b-c.tlsave"), ("   ", f"training-loop-progress-{today}.tlsave")):
            pg.click(".save-dl")
            pg.fill(".save-name input", typed)
            with pg.expect_download() as dl:
                pg.press(".save-name input", "Enter")
            check(dl.value.suggested_filename == want, f"typed {typed!r} -> {want}")
        pg.click(".save-dl")
        pg.keyboard.press("Escape")
        check(pg.locator(".modal").count() == 0, "Escape closes the name box")

        # ---------------------------------------------------------------- Download with a Save dialog (stubbed)
        ctx2 = b.new_context(viewport={"width": 1280, "height": 900})
        ctx2.add_init_script(FAKE_PICKER)
        pg2 = ctx2.new_page()
        home(pg2)
        pg2.evaluate("s => { for (const [k, v] of Object.entries(s)) localStorage.setItem(k, JSON.stringify(v)); }", original)
        pg2.click(".save-dl")
        pg2.wait_for_function("window.__written")
        opts = pg2.evaluate("window.__pickerOpts")
        check(opts["suggestedName"] == f"training-loop-progress-{today}.tlsave", "the Save dialog suggests training-loop-progress-<date>.tlsave")
        check(opts["types"][0]["accept"] == {"application/x-training-loop-progress": [".tlsave"]}, "the Save dialog offers the .tlsave type")
        check(json.loads(pg2.evaluate("window.__written"))["stores"] == original, "the file written through the dialog holds this browser's stores")
        check(pg2.locator(".modal").count() == 0 and wait_reload_toast(pg2) == "Saved picked.tlsave", "no name box; a toast names the saved file")
        pg2.evaluate("window.__abort = true; window.__written = null")
        pg2.click(".save-dl")
        pg2.wait_for_timeout(300)
        check(pg2.locator(".modal").count() == 0 and not pg2.evaluate("window.__written"), "closing the Save dialog does nothing")
        # the real dialog accepts these options: without a click it refuses for that reason, not for the options
        pg3 = b.new_page()
        pg3.goto(BASE + "/#/")
        err = pg3.evaluate("""async () => {
          const t = async (o) => { try { await window.showSaveFilePicker(o); return 'opened'; } catch (e) { return e.name; } };
          return [await t({ suggestedName: 'x.tlsave', types: [{ description: 'training-loop progress', accept: { 'application/x-training-loop-progress': ['.tlsave'] } }] }),
                  await t({ types: [{ accept: { 'not a type': ['.tlsave'] } }] })];
        }""")
        check(err[0] != "TypeError" and err[1] == "TypeError", f"Chromium's own Save dialog takes the .tlsave type ({err})")
        pg3.close()
        ctx2.close()

        # ---------------------------------------------------------------- Upload into an empty browser: Restore
        fresh = b.new_context(viewport={"width": 1280, "height": 900})
        pf = fresh.new_page()
        home(pf)
        upload(pf, backup)
        check("Restore progress?" in pf.inner_text(".modal") and "my backup.tlsave" in pf.inner_text(".modal"), "the restore box names the file")
        check(pf.evaluate(HEADS) == ["The file"], "an empty browser: only the file's column")
        cells = pf.locator(".save-cmp td").all_inner_texts()
        check(cells == ["Problems solved", "110", "Medals", "1", "Tutorials completed", "1", "Tutorials started", "2"], f"the file's progress is summarized ({cells})")
        check(pf.locator(".modal .btn").all_inner_texts() == ["Cancel", "Restore"], "an empty browser: Cancel or Restore")
        pf.screenshot(path=str(OUT / "restore-empty.png"))
        check(restore(pf, ".modal .merge") == "Restored progress from my backup.tlsave", "after the reload, a toast says it was restored")
        check(pf.evaluate(STORAGE) == original, "the restored storage equals the original")
        check(count_of(pf, "numpy") == "107" and count_of(pf, "bash") == "3", "the home page counts the restored problems")
        chips = pf.locator(".dsa-status").all_inner_texts()
        check([c.lower() for c in chips[3:6]] == ["not started", "in progress", "completed"], f"the tutorials' statuses are restored ({chips[3:6]})")
        pf.goto(BASE + "/#/dsa/stacks")
        pf.wait_for_selector(".check")
        check(pf.evaluate("document.querySelector('.step-kicker').textContent").startswith("Step 1 of"), "a restored tutorial opens at its step")
        pf.goto(BASE + "/#/numpy")
        pf.wait_for_selector('a.topic[href="#/numpy/1.01"]')
        check(pf.locator('a.topic[href="#/numpy/1.01"] .badge svg').count() == 1, "a restored medal shows on its module")

        # ---------------------------------------------------------------- Upload into a browser with progress: Merge
        other = {
            "training-loop:v2": {"modules": {"numpy/1.02": module(20, 202), "bash/7.01": module(1, 799), "pandas/2.01": module(4, 204)}},
            "training-loop:term:bash/7.01": {"seed": 799, "log": ["pwd\r"], "at": 2},
            "training-loop:term-font": 18,
        }

        def seeded():
            c = b.new_context(viewport={"width": 1280, "height": 900})
            q = c.new_page()
            home(q)
            q.evaluate("s => { for (const [k, v] of Object.entries(s)) localStorage.setItem(k, JSON.stringify(v)); localStorage.setItem('training-loop:v1', '1'); }", other)
            home(q)
            return c, q

        c, q = seeded()
        upload(q, backup)
        check(q.evaluate(HEADS) == ["This browser", "The file"], "a browser with progress: both columns")
        check(q.locator(".save-cmp td").all_inner_texts()[:3] == ["Problems solved", "25", "110"], "this browser's progress next to the file's")
        check(q.locator(".modal .btn").all_inner_texts() == ["Cancel", "Replace", "Merge"], "Cancel, Replace or Merge")
        q.screenshot(path=str(OUT / "restore-merge.png"))
        check(restore(q, ".modal .merge") == "Merged progress from my backup.tlsave", "the toast says it merged")
        got = q.evaluate(STORAGE)
        mods = got["training-loop:v2"]["modules"]
        check(mods["numpy/1.01"]["solved"] == 100 and mods["numpy/1.02"]["solved"] == 20 and mods["bash/7.01"]["solved"] == 3
              and mods["pandas/2.01"]["solved"] == 4, "Merge: each module keeps the further-along record")
        check(got["training-loop:term:bash/7.01"]["seed"] == 701, "Merge: the terminal session follows its module's winning record")
        check(got["training-loop:term-font"] == 18, "Merge: this browser's preferences stay")
        check(got["training-loop:dsa:v1"] == original["training-loop:dsa:v1"], "Merge: tutorials only the file had are added")
        check(q.evaluate("localStorage.getItem('training-loop:v1')") == "1", "the retired key isn't touched")
        c.close()

        # Replace
        c, q = seeded()
        upload(q, backup)
        restore(q, ".modal .replace")
        check(q.evaluate(STORAGE) == original, "Replace: this browser now matches the file exactly")
        c.close()

        # Cancel, and a click outside the box
        c, q = seeded()
        for how in ("cancel", "outside"):
            upload(q, backup)
            if how == "cancel":
                q.click(".modal .cancel")
            else:
                q.mouse.click(5, 5)
            check(q.locator(".modal").count() == 0 and q.evaluate(STORAGE) == other, f"{how}: nothing changes")
        c.close()

        # ---------------------------------------------------------------- files that are refused
        bad = {
            "not-json.tlsave": "hello",
            "other.tlsave": json.dumps({"some": "json"}),
            "newer.tlsave": json.dumps({"format": "training-loop-progress", "version": 2, "stores": {}}),
            "damaged.tlsave": json.dumps({"format": "training-loop-progress", "version": 1,
                                          "stores": {"training-loop:v2": {"modules": {"numpy/1.01": {"solved": "lots"}}}}}),
        }
        want = {"not-json.tlsave": "can't be read", "other.tlsave": "isn't a training-loop progress file",
                "newer.tlsave": "newer version", "damaged.tlsave": "damaged (module numpy/1.01)"}
        before = pg.evaluate(STORAGE)
        for name, text in bad.items():
            (OUT / name).write_text(text)
            upload(pg, OUT / name)
            msg = pg.inner_text(".modal")
            check("Can't use this file" in msg and want[name] in msg, f"{name} is refused: {pg.inner_text('.modal p')}")
            if name == "newer.tlsave":
                pg.screenshot(path=str(OUT / "refused.png"))
            pg.click(".modal .ok")
        check(pg.evaluate(STORAGE) == before, "a refused file changes nothing")
        # a store this version doesn't know is kept as it is; keys that aren't the site's are ignored
        (OUT / "extra.tlsave").write_text(json.dumps({"format": "training-loop-progress", "version": 1, "savedAt": "2026-10-31T00:00:00Z",
                                                      "stores": {"training-loop:future:v1": {"x": 1}, "someone-else": 1}}))
        upload(pg, OUT / "extra.tlsave")
        restore(pg, ".modal .merge")
        check(pg.evaluate("[localStorage.getItem('training-loop:future:v1'), localStorage.getItem('someone-else')]") == ['{"x":1}', None],
              "an unknown training-loop store is kept; other keys are ignored")

        # ---------------------------------------------------------------- phone, dark
        ph = b.new_context(viewport={"width": 390, "height": 844}, color_scheme="dark")
        pp = ph.new_page()
        home(pp)
        bar = pp.locator(".save-bar").bounding_box()
        h1 = pp.locator(".home h1").bounding_box()
        check(bar["y"] + bar["height"] <= h1["y"] and bar["x"] + bar["width"] <= 390 - 15, "phone: the buttons fit above the title")
        check(pp.evaluate("document.documentElement.scrollWidth <= 390"), "phone: no sideways scroll")
        pp.screenshot(path=str(OUT / "phone-dark.png"))
        upload(pp, backup)
        pp.screenshot(path=str(OUT / "phone-dark-restore.png"))
        ph.close()

        check(not errors, f"no page errors ({errors[:2]})")
        b.close()
finally:
    srv.terminate()

print("\nALL CHECKS PASSED" if not problems else f"\n{len(problems)} FAILED:\n  " + "\n  ".join(problems))
sys.exit(1 if problems else 0)

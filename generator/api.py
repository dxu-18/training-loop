"""JSON-in / JSON-out entry points called from the browser (docs/worker.js)."""

from __future__ import annotations

import json
import random
import time

from . import core, grader


def generate(section: str, module: str, n: int, seed: int, tier: str | None = None) -> str:
    """Problem for question number n of a module. Pass `tier` to reroll at a fixed difficulty."""
    tier = tier or core.tier_for(n, seed)
    return json.dumps(core.generate(section, module, tier, seed))


def run(problem_json: str, code: str) -> str:
    return json.dumps(grader.run(json.loads(problem_json), code))


def grade(problem_json: str, code: str) -> str:
    return json.dumps(grader.grade(json.loads(problem_json), code))


def term_start(problem_json: str, log_json: str, real_ms: float) -> str:
    """Open the live terminal for a bash problem (log: an earlier session from the browser's storage)."""
    from .bashlite import live
    from .bashlite.check import SetupError
    from .bashlite.directives import DirectiveError
    try:
        return json.dumps(live.start(json.loads(problem_json), json.loads(log_json or "[]"), real_ms))
    except (SetupError, DirectiveError) as e:
        return json.dumps({"error": f"The problem's setup failed: {e}"})


def term_keys(events_json: str, real_ms: float) -> str:
    from .bashlite import live
    return json.dumps(live.keys(json.loads(events_json), real_ms))


def term_pump(real_ms: float) -> str:
    from .bashlite import live
    return json.dumps(live.pump(real_ms))


def term_edit(action: str, content, name, real_ms: float) -> str:
    from .bashlite import live
    return json.dumps(live.edit(action, content, real_ms, name or None))


def term_resize(cols: int, real_ms: float) -> str:
    from .bashlite import live
    return json.dumps(live.resize(cols, real_ms))


def term_answer(real_ms: float) -> str:
    """The session as the answer to grade (JSON with the key log)."""
    from .bashlite import live
    return live.answer(real_ms)


def dsa_run(user: str, tests: str = "") -> str:
    """Data Structures tutorials (docs/dsa): run the user's Python file, then the checker's tests in the same namespace.

    stdout is what the tests printed (what the user's file printed itself is dropped), or the file's own output when
    there are no tests. A traceback lists only the user's lines (solution.py)."""
    import contextlib
    import io
    import linecache
    import traceback

    res = {"compiled": True, "stdout": "", "stderr": "", "exit": 0}
    try:
        code = compile(user, "solution.py", "exec")
    except SyntaxError as e:
        res.update(compiled=False, compileOutput="".join(traceback.format_exception_only(e)).strip())
        return json.dumps(res)
    linecache.cache["solution.py"] = (len(user), None, user.splitlines(True), "solution.py")
    out = io.StringIO()
    ns = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO() if tests else out):
            exec(code, ns)
        if tests:
            with contextlib.redirect_stdout(out):
                exec(compile(tests, "tests.py", "exec"), ns)
    except BaseException as e:  # noqa: BLE001  (contract failures are AssertionErrors; SystemExit too)
        tb = traceback.TracebackException.from_exception(e)
        lines = [f"  line {f.lineno}, in {f.name}\n    {(f.line or '').strip()}" for f in tb.stack if f.filename == "solution.py"]
        res["stderr"] = ("Traceback (your code):\n" + "\n".join(lines) + "\n" if lines else "") + "".join(tb.format_exception_only()).strip()
        res["exit"] = 1
    finally:
        linecache.cache.pop("solution.py", None)
    res["stdout"] = out.getvalue()[:200_000]
    return json.dumps(res)


def info() -> str:
    return json.dumps({
        s: {m: {t: len(core.templates(s, m, t)) for t in core.TIERS} for m in core.modules(s)}
        for s in core.sections()
    })


def self_test(section: str, seeds: int = 2, only: str = "", skip: str = "") -> str:
    """Grade every template's own solution in this runtime (used to check Pyodide).

    `only` / `skip` are comma-separated module ids, so each module can be checked with
    exactly the packages its page loads. Bash solutions are sent the way the page sends them: as a session
    typed into the live terminal."""
    want = set(filter(None, only.split(",")))
    drop = set(filter(None, skip.split(",")))
    failures, count, timings = [], 0, []
    runs = []
    for tpl in core.templates(section):
        if (want and tpl.module not in want) or tpl.module in drop:
            continue
        core._PICKED.clear()
        core.build(tpl, 0)
        n_items = core._PICKED[0] if core._PICKED else 0
        runs += [(tpl, seed, None) for seed in range(seeds)]
        runs += [(tpl, i, i) for i in range(n_items)]  # catalogs: every item once, too
    for tpl, seed, item in runs:
        count += 1
        core._FORCE_ITEM[0] = item
        try:
            p = core.build(tpl, random.Random(seed).randrange(2**31)).to_dict()
            t0 = time.perf_counter()
            if p.get("kind") == "bash":
                from .bashlite.live import reference_session
                res = grader.grade(p, reference_session(p))
            else:
                res = grader.grade(p, p["solution"])
            limit = 60.0 if p.get("kind") == "bash" else p["timeout"]   # the page allows a terminal session more
            timings.append((time.perf_counter() - t0, tpl.module, tpl.name, limit))
            if not res["correct"]:
                failures.append({"template": tpl.name + (f"[{item}]" if item is not None else ""), "module": tpl.module,
                                 "seed": seed, "message": res["message"], "error": res["error"]})
        except Exception as e:  # noqa: BLE001
            failures.append({"template": tpl.name, "module": tpl.module, "seed": seed,
                             "message": "generation crashed", "error": repr(e)})
        finally:
            core._FORCE_ITEM[0] = None
    slow = sorted(timings, reverse=True)[:5]
    return json.dumps({"checked": count, "failures": failures,
                       "slowest": [{"seconds": round(t, 2), "module": m, "template": n, "limit": lim} for t, m, n, lim in slow]})

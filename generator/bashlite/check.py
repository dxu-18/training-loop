"""Run and grade bash answers (problem kind "bash").

A run builds the machine from the setup's `#:` directives, runs the setup's bash code quietly in the user's
interactive session (so its variables, functions, jobs and files are there), then types the answer line by line.
Graded, per problem["answer"] (default: "<stdout>", or "<post>" when the problem has a `post` check):
  "<stdout>"  what the answer printed          "<stderr>"  its error output
  "<status>"  the exit status at the end ($?)  "<post>"    what problem["post"] (bash) prints afterwards
The reference solution runs the same way on a fresh, identical machine; outputs are compared after removing
trailing spaces on each line and trailing blank lines (`unordered`: as sorted lines).
"""

from __future__ import annotations

import re

from .directives import DirectiveError, build, parse, prepare, split, start_session
from .features import forbidden, missing

SEED = 7


class SetupError(Exception):
    pass


def prepare_case(problem: dict, setup: str):
    """The machine a setup describes, ready for the answer: (world, session, spec)."""
    dirs, setup_code = split(setup)
    spec = parse(dirs)
    world, sess = build(spec, seed=SEED + int(problem.get("seed", 0)) % 1000)
    prepare(sess, spec)
    if setup_code.strip():
        r = sess.run(setup_code, quiet=True, history=False)
        if r["error"]:
            raise SetupError(r["error"])
    start_session(sess, spec)
    return world, sess, spec


def run_case(problem: dict, code: str, setup: str, echo: bool = True) -> dict:
    world, sess, spec = prepare_case(problem, setup)
    fs = world.local.fs
    if spec.script:
        cwd = sess.proc.cwd
        path = spec.script if spec.script.startswith("/") else cwd.rstrip("/") + "/" + spec.script
        node = fs.write_file(path, code if code.endswith("\n") else code + "\n", "/", 0o755, sess.proc.uid, sess.proc.gid)
        node.mode = 0o755
        typed = list(spec.run) or [f"./{spec.script}"]
    else:
        typed = code.split("\n")
    typed += spec.typed
    world.tty.echo = echo
    res = sess.run("\n".join(typed))
    world.tty.echo = False
    out = {"stdout": res["stdout"], "stderr": res["stderr"], "status": res["status"], "error": res["error"],
           "segments": res["segments"], "post": ""}
    if problem.get("post") and res["error"] is None:
        from .world import Session
        s2 = sess if sess.alive else Session(world, cwd=sess.proc.cwd)
        r2 = s2.run(problem["post"], history=False)
        out["post"] = r2["stdout"] + ("" if not r2["error"] else "")
        out["post_error"] = r2["error"]
    return out


ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")


def normalize(text: str, unordered: bool = False) -> list[str]:
    """Compare text as shown: colours don't matter (jq, grep and ls colour only on a terminal)."""
    lines = [ln.rstrip() for ln in ANSI.sub("", text).replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return sorted(lines) if unordered else lines


def _diff(exp: list[str], got: list[str], what: str) -> str | None:
    if exp == got:
        return None
    for i, (a, b) in enumerate(zip(exp, got)):
        if a != b:
            shown = b if len(b) <= 120 else b[:117] + "..."
            return f"{what} differs at line {i + 1}: you printed `{shown}`." if b else \
                f"{what} differs at line {i + 1}: you printed an empty line."
    if len(got) < len(exp):
        return (f"{what} is missing lines: you printed {len(got)} line{'s' if len(got) != 1 else ''}, "
                f"the expected output has {len(exp)}.") if got else f"{what} is empty, but something should be printed."
    return f"{what} has extra lines: you printed {len(got)} lines, the expected output has {len(exp)}."


def answers(problem: dict) -> list[str]:
    ans = problem.get("answer") or []
    return ans or (["<post>"] if problem.get("post") else ["<stdout>"])


def compare(problem: dict, exp: dict, got: dict) -> str | None:
    uo = bool(problem.get("unordered"))
    for a in answers(problem):
        if a == "<stdout>":
            msg = _diff(normalize(exp["stdout"], uo), normalize(got["stdout"], uo), "Your output")
        elif a == "<stderr>":
            msg = _diff(normalize(exp["stderr"], uo), normalize(got["stderr"], uo), "Your error output")
        elif a == "<status>":
            msg = None if exp["status"] == got["status"] else \
                f"The exit status is {got['status']}; it should be different."
        elif a == "<post>":
            msg = _diff(normalize(exp["post"], uo), normalize(got["post"], uo), "The check after your commands")
        else:
            raise ValueError(f"unknown answer {a!r}")
        if msg:
            return msg
    return None


def _console(res: dict) -> list:
    return [[s, t] for s, t in res["segments"]]


def checker(problem: dict, code: str, grade: bool) -> dict:
    from .live import parse_answer
    session = parse_answer(code)
    if session is not None:
        return term_checker(problem, session)
    if "#: keys" in problem.get("setup", ""):
        from .keys import checker as keys_checker
        return keys_checker(problem, code, grade)
    try:
        if not grade:
            res = run_case(problem, code, problem.get("setup", ""))
            return {"ok": res["error"] is None, "stdout": res["stdout"], "value": None, "error": res["error"],
                    "console": _console(res)}
        out = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": "",
               "console": []}
        msg = missing(code, problem.get("require") or []) or forbidden(code, problem.get("forbid") or [])
        if msg:
            res = run_case(problem, code, problem.get("setup", ""))
            out.update(stdout=res["stdout"], console=_console(res), message=msg)
            return out
        setups = [problem.get("setup", "")] + list(problem.get("tests") or [])
        for i, setup in enumerate(setups):
            exp = run_case(problem, problem["solution"], setup, echo=False)
            got = run_case(problem, code, setup, echo=i == 0)
            if i == 0:
                out["stdout"] = got["stdout"]
                out["console"] = _console(got)
            if got["error"]:
                out["error"] = got["error"]
                if i:
                    out["message"] = f"Hidden test {i}: {got['error']}"
                return out
            msg = compare(problem, exp, got)
            if msg:
                out["message"] = msg if i == 0 else f"Wrong answer on hidden test {i}."
                return out
        out["correct"] = True
        out["message"] = "" if len(setups) == 1 else f"Passed all {len(setups)} tests, including the hidden ones."
        return out
    except (SetupError, DirectiveError) as e:
        return {"ok": False, "correct": False, "compiled": True, "stdout": "", "value": None,
                "error": f"The problem's setup failed: {e}", "message": "", "console": []}


# ---------------------------------------------------------------------------- answers from the live terminal

def _result(**kw):
    out = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": "", "console": []}
    out.update(kw)
    return out


def _session_features(t, extra=""):
    """Features of everything the session ran (each command line as typed, and after history expansion)."""
    from .features import features
    have: set[str] = set()
    for m in t.marks:
        for src in {m["raw"], "\n".join(m["lines"])}:
            if src.strip():
                try:
                    have |= features(src)
                except Exception:  # noqa: BLE001
                    pass
    if extra.strip():
        have |= features(extra)
    return have


def _feature_message(problem, have):
    from .features import _shown, alternatives
    for req in problem.get("require") or []:
        alts = alternatives(req)
        if not any(a in have for a in alts):
            return "This problem asks you to use " + " or ".join(f"`{_shown(a)}`" for a in alts) + "."
    for f in problem.get("forbid") or []:
        if f in have:
            return f"Solve this one without `{_shown(f)}` (your session used it; press Reset to start over)."
    return None


def _same(exp: str, got: str, uo: bool, loose: bool) -> bool:
    e, g = normalize(exp, uo), normalize(got, uo)
    if e == g:
        return True
    if loose and e and g and not uo:     # a command stopped with Ctrl-C: it printed the start of the output
        n = min(len(e), len(g))
        return e[:n] == g[:n]
    return False


def _find_run(problem, exp, cmds, start=None):
    """The commands i..j (in a row) whose output is the expected one (i = start if given). -> (i, j) or None."""
    uo = bool(problem.get("unordered"))
    want = [a for a in answers(problem) if a in ("<stdout>", "<stderr>")]
    if not want:
        return (0, len(cmds) - 1) if cmds else (0, -1)
    limit = 2 * max(len(exp["stdout"]), len(exp["stderr"])) + 2000
    for i in (range(len(cmds)) if start is None else [start]):
        acc = {"<stdout>": "", "<stderr>": ""}
        for j in range(i, len(cmds)):
            acc["<stdout>"] += cmds[j]["stdout"]
            acc["<stderr>"] += cmds[j]["stderr"]
            loose = cmds[j]["interrupted"] or exp.get("interrupted", False)
            if all(_same(exp[a[1:-1]], acc[a], uo, loose) for a in want):
                return i, j
            if len(acc["<stdout>"]) + len(acc["<stderr>"]) > limit and not loose:
                break
    return None


def _run_message(problem, exp, cmds):
    """Why no command printed the expected output: the last one that printed something, compared."""
    uo = bool(problem.get("unordered"))
    stream = "stdout" if "<stdout>" in answers(problem) else "stderr"
    printed = [c for c in cmds if c[stream].strip()]
    if not cmds:
        return "Your session didn't run any commands: type them in the terminal, then Submit."
    if not printed:
        return ("None of your commands printed anything" + (" on stderr" if stream == "stderr" else "") +
                ", but the task expects output.")
    c = printed[-1]
    shown = c["raw"] if len(c["raw"]) <= 60 else c["raw"][:57] + "..."
    what = f"The output of `{shown}`" if stream == "stdout" else f"The error output of `{shown}`"
    return _diff(normalize(exp[stream], uo), normalize(c[stream], uo), what) or \
        "Your commands print the right lines, but not together in one go."


CONTEXT_TRIES = 12     # recent commands tried as the start of the answer, with the reference typed there


def _output_found(problem, setup, exp, t, cmds):
    """Some commands in a row printed the expected output: as on a fresh machine, or as the reference prints when
    typed at that point of the session (after your exploring changed process ids, history or the directory)."""
    if _find_run(problem, exp, cmds) is not None:
        return True
    for i in range(len(cmds) - 1, max(0, len(cmds) - CONTEXT_TRIES - 1), -1):
        if i == 0 or not (cmds[i]["stdout"] or cmds[i]["stderr"]):
            continue
        exp_i = _expected(problem, setup, t.prefix_before(i))
        if _find_run(problem, exp_i, cmds, start=i) is not None:
            return True
    return False


def _script_path(t):
    s = t.spec.script
    return s if s.startswith("/") else t.start_cwd.rstrip("/") + "/" + s


def _script_run(t, code_for_file=None):
    """Run a script problem's run lines after the session. code_for_file: the reference's script text."""
    spec = t.spec
    path = _script_path(t)
    if code_for_file is not None:
        node = t.fs.write_file(path, code_for_file if code_for_file.endswith("\n") else code_for_file + "\n", "/",
                               0o755, t.sess.proc.uid, t.sess.proc.gid)
        node.mode = 0o755
    else:
        node = t.fs.lookup(path)
        node.mode |= 0o755
    typed = list(spec.run) or [f"./{spec.script}"]
    return t.run_after("\n".join(typed + list(spec.typed)))


def _expected(problem, setup, prefix=None):
    """The reference solution, typed into a fresh terminal (or after `prefix`, the start of the user's session)."""
    from .live import LiveTerm, text_log
    t = LiveTerm(problem, setup, typed=not prefix)
    if prefix:
        t.replay(prefix, final=False)
        t.world.ctrl_c_after = t.ctrl_c_typed
        first = len(t.marks)
        t.replay(text_log(problem["solution"], t.spec.typed))
        cmds = t.commands()[first:]
        return {"stdout": "".join(c["stdout"] for c in cmds), "stderr": "".join(c["stderr"] for c in cmds),
                "status": t.status(), "interrupted": any(c["interrupted"] for c in cmds), "post": ""}
    if t.spec.script:
        r = _script_run(t, problem["solution"])
        exp = {"stdout": r["stdout"], "stderr": r["stderr"], "status": r["status"], "interrupted": False}
    else:
        t.replay(text_log(problem["solution"], t.spec.typed))
        cmds = t.commands()
        exp = {"stdout": "".join(c["stdout"] for c in cmds), "stderr": "".join(c["stderr"] for c in cmds),
               "status": t.status(), "interrupted": any(c["interrupted"] for c in cmds)}
    exp["post"] = t.run_after(problem["post"])["stdout"] if problem.get("post") else ""
    return exp


def _session(problem, setup, log, end):
    from .live import LiveTerm
    t = LiveTerm(problem, setup)
    t.replay(log, end)
    return t


def term_checker(problem: dict, session) -> dict:
    """Grade a live-terminal session: replay it on the problem's machine and on the hidden ones.
    Every command counts for require/forbid, in any order; the output asked for must come from some commands in a
    row (exploring with ls or cat first is fine); a `post` check sees the files (and shell) the session left."""
    log, end = session
    if "#: keys" in problem.get("setup", ""):
        return _keys_checker(problem, log, end)
    try:
        setups = [problem.get("setup", "")] + list(problem.get("tests") or [])
        for i, setup in enumerate(setups):
            t = _session(problem, setup, log, end)
            if i == 0 and not t.marks:
                return _result(compiled=False, message="Your session didn't run any commands yet: type them in the "
                                                       "terminal, then Submit.")
            got = {"stdout": "", "stderr": "", "status": t.status(), "post": ""}
            extra = ""
            if t.spec.script:
                path = _script_path(t)
                if not t.fs.exists(path):
                    return _result(compiled=False, message=f"There's no `{t.short(path)}` yet: create the script (for "
                                                           f"example with `nano {t.spec.script}`), save it, then Submit.")
                node = t.fs.lookup(path)
                if node.type != "f":
                    return _result(message=f"`{t.short(path)}` should be a file.")
                extra = bytes(node.data).decode("utf-8", "replace")
            if i == 0:
                msg = _feature_message(problem, _session_features(t, extra))
                if msg:
                    return _result(message=msg)
            exp = _expected(problem, setup)
            if t.spec.script:
                r = _script_run(t)
                if r["error"]:
                    return _result(error=r["error"], message=f"Hidden test {i}: {r['error']}" if i else "")
                got.update(stdout=r["stdout"], stderr=r["stderr"], status=r["status"])
                cmds = [{"raw": f"./{t.spec.script}", "stdout": r["stdout"], "stderr": r["stderr"],
                         "interrupted": False}]
            else:
                cmds = t.commands()
            if problem.get("post"):
                pr = t.run_after(problem["post"])
                got["post"] = pr["stdout"]
            msg = None
            if any(a in ("<stdout>", "<stderr>") for a in answers(problem)):
                if t.spec.script:
                    msg = compare(problem, exp, got)
                elif not _output_found(problem, setup, exp, t, cmds):
                    msg = _run_message(problem, exp, cmds)
            if msg is None and "<status>" in answers(problem) and exp["status"] != got["status"]:
                msg = (f"Your session ends with exit status {got['status']}; it should be different." if t.ended else
                       f"Your last command's exit status ($?) is {got['status']}; it should be different.")
            if msg is None and "<post>" in answers(problem):
                uo = bool(problem.get("unordered"))
                msg = _diff(normalize(exp["post"], uo), normalize(got["post"], uo), "The check after your session")
            if msg:
                return _result(message=msg if i == 0 else f"Wrong answer on hidden test {i}: your commands work on the "
                                                          f"machine shown, but not on one with different files.")
        return _result(correct=True, message="" if len(setups) == 1 else
                       f"Passed all {len(setups)} tests, including the hidden ones.")
    except (SetupError, DirectiveError) as e:
        return _result(error=f"The problem's setup failed: {e}")


def _keys_checker(problem, log, end):
    """7.17 in the live terminal: the keys pressed are replayed; graded like keys.checker (commands run, the line
    left on the prompt, tmux's state)."""
    from .keys import KeySim, compare_state, run_keys
    from .live import LiveTerm, key_features
    have = key_features(log)
    for req in problem.get("require") or []:
        if not any(a in have for a in req.split("|")):
            return _result(message="This problem asks you to use " + " or ".join(f"`{a}`" for a in req.split("|")) + ".")
    for f in problem.get("forbid") or []:
        if f in have:
            return _result(message=f"Solve this one without `{f}`." if f != "text" else
                           "Solve this one without typing text (use the keys).")
    try:
        setups = [problem.get("setup", "")] + list(problem.get("tests") or [])
        for i, setup in enumerate(setups):
            exp = run_keys(problem, problem["solution"], setup)
            t = LiveTerm(problem, setup)
            t.replay(log, end)
            got = KeySim.state(t)
            msg = compare_state(exp["state"], got)
            if msg:
                return _result(message=msg if i == 0 else f"Wrong answer on hidden test {i}.")
        return _result(correct=True)
    except (SetupError, DirectiveError) as e:
        return _result(error=f"The problem's setup failed: {e}")

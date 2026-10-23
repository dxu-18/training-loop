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

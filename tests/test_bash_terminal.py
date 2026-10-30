"""The Bash topic's live terminal (generator/bashlite/live.py) and the grading of terminal sessions.

Every template's reference solution, sent the way the page sends it (a recorded session), must pass; so must a
session that explores first; a wrong or empty session must not. The terminal itself: real-time waits, Ctrl-C,
continuation prompts, nano, quotas, and rebuilding a session from its log after a reload.
"""

import json

import pytest

import generator.sections.s07_bash  # noqa: F401
from generator.bashlite import live
from generator.bashlite.check import checker
from generator.bashlite.live import LiveTerm, reference_session, text_log
from generator.core import build, templates

BASH = templates("bash")
IDS = lambda t: f"{t.module}-{t.tier}-{t.name}"  # noqa: E731


def session(log, end=None):
    return json.dumps({"log": log, "end": end})


@pytest.mark.parametrize("tpl", BASH, ids=IDS)
def test_reference_session_passes(tpl):
    for seed in range(3):
        p = build(tpl, seed).to_dict()
        res = checker(p, reference_session(p), True)
        assert res["correct"], f"seed {seed}: {res}\n{p['prompt']}\n{p['solution']}"


@pytest.mark.parametrize("tpl", [t for t in BASH if t.module != "7.17"], ids=IDS)
def test_exploring_first_still_passes(tpl):
    """Looking around before answering (ls, pwd, echo) must not cost the answer."""
    for seed in range(2):
        p = build(tpl, seed).to_dict()
        if "history" in p["solution"] or "!" in p["solution"]:
            continue      # the history itself is the answer
        ref = json.loads(reference_session(p))
        log = text_log("ls\npwd\necho looking") + ref["log"]
        res = checker(p, session(log), True)
        assert res["correct"], f"seed {seed}: {res}\n{p['prompt']}\n{p['solution']}"


@pytest.mark.parametrize("tpl", BASH, ids=IDS)
def test_wrong_sessions_fail(tpl):
    for seed in range(2):
        p = build(tpl, seed).to_dict()
        for log in ([], text_log("echo nope")):
            assert not checker(p, session(log), True)["correct"], (seed, log)


def _term(setup="", **kw):
    return LiveTerm({"seed": 0}, setup, **kw)


def _type(t, line):
    t.feed("text", line)
    t.feed("key", "Enter")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() != "r")


def test_ctrl_c_stops_a_loop_and_the_rest_of_the_line():
    t = _term()
    t.feed("text", "while true; do sleep 1; done; echo after")
    t.feed("key", "Enter")
    t.advance(t.k.now + 3 * 10**9)
    assert t.where() == "r"
    t.feed("key", "C-c")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() != "r")
    assert t.where() == "p"
    assert "^C" in "".join(t.screen)
    assert t.commands()[-1]["stdout"] == "" and t.commands()[-1]["interrupted"]
    _type(t, "echo $?")
    assert t.commands()[-1]["stdout"] == "130\n"


def test_ctrl_c_at_the_continuation_prompt_and_comments():
    t = _term()
    _type(t, "for i in 1 2")
    assert t.sh.awaiting == "PS2"
    t.feed("key", "C-c")
    assert t.sh.awaiting == "PS1"
    _type(t, "# just a comment")
    assert t.sh.awaiting == "PS1"
    _type(t, "echo ok")
    assert t.commands()[-1]["stdout"] == "ok\n"


def test_programs_read_typed_lines_and_eof():
    t = _term()
    _type(t, "cat > notes.txt")
    t.feed("text", "first line")
    t.feed("key", "Enter")
    t.feed("key", "C-d")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() != "r")
    _type(t, "cat notes.txt")
    assert t.commands()[-1]["stdout"] == "first line\n"


def test_nano_saves_and_renames():
    t = _term()
    _type(t, "nano todo.txt")
    assert t.where() == "e" and t.editor()["shown"] == "todo.txt"
    assert t.edit("save", "a\nb\n") is None
    assert t.edit("save", "c\n", name="other.txt") is None
    t.edit("close")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() != "r")
    _type(t, "cat todo.txt other.txt")
    assert t.commands()[-1]["stdout"] == "a\nb\nc\n"


def test_quotas_end_in_an_error_not_a_crash():
    t = _term()
    _type(t, "touch f{1..4000}")
    assert "No space left on device" in t.commands()[-1]["stderr"]
    _type(t, "head -c 20000000 /dev/zero > big")
    assert "No space left on device" in t.commands()[-1]["stderr"]
    _type(t, "rm -f f* big; echo fine")
    assert t.commands()[-1]["stdout"] == "fine\n"


def test_reload_rebuilds_the_session():
    p = {"seed": 0, "setup": "#: cwd ~/work"}
    r = live.start(p, [], 0)
    assert r["view"]["mode"] == "prompt"
    live.keys([["text", "mkdir -p a/b && cd a"], ["key", "Enter"]], 100)
    live.keys([["text", "sleep 2 &"], ["key", "Enter"]], 200)
    r = live.pump(3000)
    log = json.loads(live.answer(3100))["log"]
    r2 = live.start(p, log, 50000)
    assert "a/b" in r2["out"] and r2["tree"]["cwd"].endswith("/work/a")
    names = [c["n"] for c in r2["tree"]["root"]["c"]]
    assert names == ["a"]


def test_ctrl_c_ends_a_follow_with_what_it_printed():
    """`tail -f` runs until Ctrl-C and has printed the lines written until then (the grader accepts that prefix)."""
    t = _term("#: cwd ~/w")
    _type(t, ": > log; (for i in 1 2 3 4 5 6 7 8 9; do echo line $i >> log; sleep 1; done) &")
    t.feed("text", "tail -f log")
    t.feed("key", "Enter")
    t.advance(t.k.now + int(3.5 * 10**9))
    t.feed("key", "C-c")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() != "r")
    c = t.commands()[-1]
    assert c["interrupted"] and c["stdout"].startswith("line 1\n") and "line 9" not in c["stdout"]

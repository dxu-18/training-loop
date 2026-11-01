"""COMP_GUIDE §9 Vim (generator/sections/s09_vim) in the live terminal (bashlite/live.py) and its grader
(bashlite/vimcheck.py): every template's reference keys pass and doing nothing doesn't, every command the guide
teaches is practised, quitting is blocked except where it's the point, and a stored session replays."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generator import grader  # noqa: E402
from generator.bashlite import live  # noqa: E402
from generator.bashlite.live import LiveTerm, reference_session, vim_session  # noqa: E402
from generator.bashlite.vim.catalog import CATALOG  # noqa: E402
from vim_templates import idle_answer, problems  # noqa: E402

PROBLEMS = [(f"{tpl.module}-{tpl.name}-{item}-{seed}", p, err) for tpl, item, seed, p, err in problems(None, 1)]


@pytest.mark.parametrize("pid,p,err", PROBLEMS, ids=[x[0] for x in PROBLEMS])
def test_reference_passes_and_idle_fails(pid, p, err):
    assert err is None, err
    res = grader.grade(p, reference_session(p))
    assert res["correct"], res["message"] or res["error"]
    assert not grader.grade(p, idle_answer(p))["correct"]


def test_every_guide_command_is_practised():
    used, solutions = set(), ""
    for _, p, err in PROBLEMS:
        used |= set(p["vim"]["uses"]) | {a for r in p["require"] for a in r.split("|")}
        solutions += p["solution"]
    text_only = {"\\<", "\\c", "\\v", "\\1", "&", "\\r", "set -o vi"}
    missing = []
    for k in CATALOG:
        if k in used or "key:" + k in used or (k in text_only and k in solutions):
            continue
        if k in ('"a', '"A') and any(len(u) == 2 and u[0] == '"' and (u[1].islower() if k == '"a' else u[1].isupper())
                                     for u in used):
            continue
        missing.append(k)
    assert not missing, missing


def _vim_problem(extra="", keys="x"):
    setup = "cat > notes.py <<'EOF'\ndef train(model):\n    return model\nEOF\n#: vim notes.py" + extra
    return {"seed": 1, "kind": "bash", "setup": setup, "solution": keys, "vim": {"check": ["text"]}}


@pytest.mark.parametrize("keys", [":q<CR>", ":q!<CR>", ":wq<CR>", ":x<CR>", "ZZ", "ZQ", ":qa!<CR>", "<C-z>",
                                  ":shell<CR>", ":stop<CR>"])
def test_quitting_is_blocked(keys):
    t = vim_session(_vim_problem(), keys)
    assert t.vim() is not None and t.where() == "v"


def test_quitting_allowed_when_practised():
    t = vim_session(_vim_problem("\n#: vimquit"), ":q<CR>")
    assert t.vim() is None and t.where() == "p"


def test_live_session_replays():
    p = _vim_problem()
    r = live.start(p, [], 0)
    assert r["view"]["mode"] == "vim" and "\x1b[?1049h" in r["out"]
    t = live._term()
    for kind, val in [("text", "c"), ("text", "i"), ("text", "w"), ("text", "fit"), ("key", "Esc"), ("text", "j"),
                      ("text", "dd")]:
        t.feed(kind, val)
    lines = list(t.vim().buf.lines)
    assert lines == ["fit train(model):"]
    log = json.loads(live.answer(10))["log"]
    r2 = live.start(p, log, 20)
    assert list(live._term().vim().buf.lines) == lines
    assert r2["view"]["vim"]["keys"] == 10
    out = live.resize(60, 30)["out"]          # every row is drawn again at the new width
    assert "\x1b[1;1H" in out and "\x1b[24;1H" in out


def test_quitting_from_the_shell_returns_to_the_prompt():
    setup = "cat > a.md <<'EOF'\nhi\nEOF"
    t = LiveTerm({"seed": 1, "setup": setup}, setup)
    t.feed("text", "vim a.md")
    t.feed("key", "Enter")
    t.advance(t.k.now + 10**9, cond=lambda: t.where() == "v")
    assert t.where() == "v"
    assert "\x1b[?1049h" in t.state()["out"]
    for kind, val in [("text", "A!"), ("key", "Esc"), ("text", ":wq"), ("key", "Enter")]:
        t.feed(kind, val)
    assert t.where() == "p"
    assert bytes(t.fs.lookup("/home/ada/a.md").data) == b"hi!\n"
    assert "\x1b[?1049l" in t.state()["out"]


def test_messages():
    p = dict(_vim_problem(keys="ciwfit<Esc>"), require=["ciw"], vim={"check": ["text"], "max_keys": 7})
    wrong = vim_session(p, "cwfit<Esc>")
    res = grader.grade(p, json.dumps({"log": wrong.log, "end": {"m": "v", "d": 0}}))
    assert not res["correct"] and not res["compiled"] and "ciw" in res["message"]
    slow = vim_session(p, "ciwfiq<BS>t<Esc>")
    res = grader.grade(p, json.dumps({"log": slow.log, "end": {"m": "v", "d": 0}}))
    assert not res["compiled"] and "par" in res["message"]
    off = vim_session(p, "ciwfat<Esc>")
    res = grader.grade(p, json.dumps({"log": off.log, "end": {"m": "v", "d": 0}}))
    assert res["compiled"] and not res["correct"] and "Line 1" in res["message"]

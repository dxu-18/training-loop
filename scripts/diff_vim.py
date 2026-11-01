"""Compare bash-lite's Vim with real Vim 9.1, case by case (tests/vim_corpus/cases.py).

Real Vim runs headless with the keys as a script: vim -u DEFAULTS -i NONE -n -s keys file (defaults.vim and the
filetype plugins: what Ubuntu's vim does without a ~/.vimrc). At the end it writes its state (lines, cursor,
curswant, the top line, registers) as JSON. The same keys run through generator/bashlite/vim and must give the same
state.

    python scripts/diff_vim.py                  compare every case, print the differences
    python scripts/diff_vim.py --record         also save real Vim's states to tests/vim_corpus/expected.json
    python scripts/diff_vim.py --only NAME,...  some cases
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "vim_corpus"))

from generator.bashlite.vim import Vim  # noqa: E402
from generator.bashlite.vim import keys as K  # noqa: E402
from generator.bashlite.vim.memio import MemIO  # noqa: E402

REGS = ['"', "0", "1", "2", "3", "-", "a", "b", "c", "."]
DUMP = ("\x1b:call writefile([json_encode({'lines': getline(1, '$'), 'cursor': [line('.'), col('.')], "
        "'want': getcurpos()[4], 'top': line('w0'), 'regs': {" +
        ", ".join(f"'{r}': [getreg('{r}'), getregtype('{r}')]" for r in REGS) +
        "}, 'modified': &modified})], 'STATE')\r:qa!\r")


def to_bytes(tokens):
    out = []
    for t in tokens:
        if len(t) == 1:
            out.append(t)
        elif t in K.CTRL_CHAR:
            out.append(K.CTRL_CHAR[t])
        elif t == "<Up>":
            out.append("\x1b[A")
        elif t == "<Down>":
            out.append("\x1b[B")
        elif t == "<Right>":
            out.append("\x1b[C")
        elif t == "<Left>":
            out.append("\x1b[D")
        elif t == "<Del>":
            out.append("\x1b[3~")
        else:
            raise ValueError(t)
    return "".join(out)


def real_vim(case):
    name, fname, text, keys = case[:4]
    with tempfile.TemporaryDirectory() as d:
        Path(d, fname).write_text(text)
        # <Sync> closes an undo block, as a pause between typed keys does (keys from a script never do)
        toks = K.parse(keys.replace("<Sync>", ":let &ul=&ul<CR>"))
        # the dump starts with <Esc>: a case must end in Normal mode for the cursor to compare
        Path(d, "keys").write_text(to_bytes(toks) + DUMP.replace("STATE", d + "/state.json"))
        env = dict(os.environ, HOME=d, TERM="xterm")
        try:
            subprocess.run(["vim", "-u", "DEFAULTS", "-i", "NONE", "-n", "-N", "-s", d + "/keys", fname],
                           cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        p = Path(d, "state.json")
        if not p.exists():
            return {"error": "no state"}
        return json.loads(p.read_text())


def ours(case):
    name, fname, text, keys = case[:4]
    io = MemIO({"/home/ada/" + fname: text}, cwd="/home/ada")
    v = Vim(io, [fname], cwd="/home/ada", home="/nonexistent")
    for k in K.parse(keys.replace("<Sync>", "")):
        v.feed(k)
    # the dump's <Esc>
    v.feed("<Esc>")
    b = v.buf
    regs = {}
    for r in REGS:
        if r == '"':
            src = v.regs.get("__unnamed")
            t = v.regs.get(src) if src else None
        else:
            t = v.regs.get(r)
        if t is None:
            regs[r] = ["", ""]
        else:
            text_, kind = t
            if kind == "b":
                width = max((len(x) for x in text_.split("\n")), default=0)
                regs[r] = [text_, f"\x16{width}"]
            else:
                regs[r] = [text_, kind]
    from generator.bashlite.vim.text import vcol
    w = v.win
    want = w.curswant + 1 if w.curswant < (1 << 30) else 2147483647
    return {"lines": list(b.lines), "cursor": [w.cursor[0] + 1, w.cursor[1] + 1], "want": want,
            "top": w.topline + 1, "regs": regs, "modified": int(b.modified)}


def compare(exp, got, case):
    diffs = []
    if exp.get("error"):
        return [f"real vim: {exp['error']}"]
    for k in ("lines", "cursor"):
        if exp[k] != got[k]:
            diffs.append(f"{k}: vim {exp[k]!r} / ours {got[k]!r}")
    opts = case[4] if len(case) > 4 else {}
    if opts.get("want", True) and exp["want"] != got["want"] and exp["lines"] == got["lines"]:
        diffs.append(f"curswant: vim {exp['want']} / ours {got['want']}")
    if opts.get("top") and exp["top"] != got["top"]:
        diffs.append(f"top: vim {exp['top']} / ours {got['top']}")
    for r in opts.get("regs", ['"', "0", "1", "-", "a"]):
        e, g = exp["regs"].get(r), got["regs"].get(r)
        if e != g:
            diffs.append(f'reg "{r}: vim {e!r} / ours {g!r}')
    return diffs


def main():
    from cases import CASES
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    cases = [c for c in CASES if only is None or c[0] in only]
    with ThreadPoolExecutor(max_workers=8) as ex:
        reals = list(ex.map(real_vim, cases))
    bad = 0
    record = {}
    for case, exp in zip(cases, reals):
        record[case[0]] = exp
        try:
            got = ours(case)
        except Exception as e:  # noqa: BLE001
            import traceback
            bad += 1
            print(f"CRASH {case[0]} {case[3]!r}: {e!r}")
            if "-v" in sys.argv:
                traceback.print_exc()
            continue
        d = compare(exp, got, case)
        if d:
            bad += 1
            print(f"DIFF {case[0]}  keys {case[3]!r}")
            for x in d:
                print("   ", x)
    if "--record" in sys.argv:
        out = ROOT / "tests" / "vim_corpus" / "expected.json"
        old = json.loads(out.read_text()) if out.exists() and only else {}
        old.update(record)
        out.write_text(json.dumps(old, indent=0, sort_keys=True) + "\n")
    print(f"{len(cases)} cases, {bad} differ")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

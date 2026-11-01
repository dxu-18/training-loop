"""Run the Vim templates' reference solutions in real Vim 9.1 too, and compare with bash-lite's Vim.

For each problem that starts in Vim (generator/sections/s09_vim), the files are written to a temporary directory,
real Vim opens them with the same arguments (vim -u DEFAULTS, as scripts/diff_vim.py), the `#: vimprep` keys and the
reference keys are typed, and the buffers, the files on disk, the cursor, the window layout and the named registers
are compared with what bash-lite's Vim ends up with (as far as the problem checks them).

Skipped: problems that start at the shell or quit Vim, use shell commands (:! :r ! :w ! filters :terminal), travel
through undo history by time, depend on a register a scripted :let would change, use netrw (not loaded with
-u DEFAULTS), or pick a spelling suggestion by its number in our list.

    python scripts/diff_vim_templates.py            # all modules
    python scripts/diff_vim_templates.py 9.04 -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diff_vim import to_bytes  # noqa: E402
from generator.bashlite.vim import keys as K  # noqa: E402
from generator.bashlite.vimcheck import reference_state  # noqa: E402
from vim_templates import problems  # noqa: E402

SYNC = ":let &ul=&ul<CR>"
DUMP = ("\x1b:call writefile([json_encode({'bufs': map(getbufinfo({'buflisted': 1}), "
        "{_, b -> [b.name, b.loaded ? getbufline(b.bufnr, 1, '$') : v:null]}), 'cur': expand('%:p'), "
        "'cursor': [line('.') - 1, col('.') - 1], 'layout': map(range(1, tabpagenr('$')), "
        "{_, t -> winlayout(t)}), 'wins': map(getwininfo(), {_, w -> [w.winid, bufname(w.bufnr)]}), "
        "'regs': {" + ", ".join(f"'{r}': getreg('{r}')" for r in "abcdefgt") + "}})], 'STATE')\r:qa!\r")


def skip_reason(p):
    sol = p["solution"]
    setup = p["setup"]
    if not any(ln.startswith("#: vim ") for ln in setup.split("\n")):
        return "starts at the shell"
    if "quit" in p["vim"]["check"] or "#: vimquit" in setup:
        return "quits"
    if re.search(r":(r |w |\S*)!|:terminal|:!|!sort|!tr|!nl", sol) or "sudo" in setup:
        return "shell command"
    if re.search(r":earlier \d+[smhd]|:later \d+[smhd]", sol):
        return "undo by time"
    if '":' in sol:
        return "the : register"
    if re.search(r":(Explore|Vexplore|Ex)\b", sol):
        return "netrw (vim -u DEFAULTS loads no plugins)"
    if '"+' in sol:
        return "the system clipboard (real Vim's is this computer's)"
    if "z=" in sol:
        return "spelling suggestions (picked by number from our own list)"
    return None


def setup_files(setup):
    """The files a V() setup writes (heredocs), its Vim arguments and prep keys."""
    files, args, prep = {}, "", []
    lines = setup.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"cat > '([^']+)' <<'(\w+)'$", ln)
        if m:
            body = []
            i += 1
            while lines[i] != m.group(2):
                body.append(lines[i])
                i += 1
            files[m.group(1)] = "\n".join(body) + "\n"
        elif ln.startswith("#: vim "):
            args = ln[7:]
        elif ln.startswith("#: vimprep "):
            prep.append(ln.split(" ", 3)[3])
        i += 1
    return files, args, prep


def real(p):
    files, args, prep = setup_files(p["setup"])
    keys = "".join(k + SYNC for k in prep) + p["solution"]
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            Path(d, name).parent.mkdir(parents=True, exist_ok=True)
            Path(d, name).write_text(text)
        Path(d, "keys").write_text(to_bytes(K.parse(keys)) + DUMP.replace("STATE", d + "/state.json"))
        env = dict(os.environ, HOME=d + "/home", TERM="xterm")
        os.makedirs(d + "/home", exist_ok=True)
        try:
            subprocess.run(["vim", "-u", "DEFAULTS", "-i", "NONE", "-n", "-N", "-s", d + "/keys"] + shlex.split(args),
                           cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        sp = Path(d, "state.json")
        if not sp.exists():
            return {"error": "no state"}
        st = json.loads(sp.read_text())
        root = os.path.realpath(d) + "/"
        st["bufs"] = {os.path.realpath(n).replace(root, "").replace("home/", "", 1) if n else "": ls
                      for n, ls in st["bufs"] if ls is not None}
        st["cur"] = os.path.realpath(st["cur"]).replace(root, "") if st["cur"] else ""
        st["files"] = {str(f.relative_to(d)): f.read_text() for f in Path(d).rglob("*")
                       if f.is_file() and f.name not in ("keys", "state.json") and not f.name.startswith(".")
                       and "home" not in f.relative_to(d).parts}
        if Path(d, "home", ".vimrc").exists():
            st["files"][".vimrc"] = Path(d, "home", ".vimrc").read_text()
        names = {w: n for w, n in st["wins"]}

        def shape(node):
            if node[0] == "leaf":
                name = names.get(node[1], "?") or "[Quickfix List]"
                return name.rsplit("/", 1)[-1] if name.startswith("/usr/share/vim") else name
            return [node[0], [shape(c) for c in node[1]]]
        st["layout"] = [shape(t) for t in st["layout"]]
        return st


def compare(p, exp, got):
    checks = p["vim"]["check"]
    out = []
    bufs = {b["name"]: b["lines"] for b in got["buffers"] if b["kind"] == "file" and b["name"]}
    if "text" in checks or "files" not in checks:
        for name, lines in bufs.items():
            if name in exp["bufs"] and exp["bufs"][name] != lines:
                out.append(f"buffer {name}: vim {exp['bufs'][name]!r}\n      ours {lines!r}")
    if "files" in checks:
        for name in sorted(set(exp["files"]) | set(got["files"])):
            if exp["files"].get(name) != got["files"].get(name):
                out.append(f"file {name}: vim {exp['files'].get(name)!r}\n      ours {got['files'].get(name)!r}")
    if "cursor" in checks and (exp["cur"], exp["cursor"]) != (got["buf"], got["cursor"]):
        out.append(f"cursor: vim {exp['cur']} {exp['cursor']} / ours {got['buf']} {got['cursor']}")
    if "layout" in checks and exp["layout"] != got["layout"]:
        out.append(f"layout: vim {exp['layout']} / ours {got['layout']}")
    if "regs" in checks:
        for r in p["vim"].get("regs", ""):
            if r in exp["regs"]:
                mine = (got["regs"].get(r) or [""])[0]
                if exp["regs"][r] != mine:
                    out.append(f"register {r}: vim {exp['regs'][r]!r} / ours {mine!r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", nargs="?")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    todo, skipped = [], {}
    for tpl, item, seed, p, err in problems(a.module, a.seeds):
        if err is not None:
            print(f"generation failed: {tpl.name}[{item}]: {err!r}")
            continue
        why = skip_reason(p)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        todo.append((f"{tpl.module} {tpl.name}[{item}] seed {seed}", p))

    def one(job):
        tag, p = job
        exp = real(p)
        if "error" in exp:
            return tag, [f"real vim: {exp['error']}"], p
        return tag, compare(p, exp, reference_state(p)), p
    bad = 0
    with ThreadPoolExecutor(8) as ex:
        for tag, diffs, p in ex.map(one, todo):
            if diffs:
                bad += 1
                print(f"DIFF {tag}: {p['solution']}")
                for d in diffs:
                    print("    " + d)
            elif a.v:
                print(f"ok   {tag}")
    print(f"{len(todo)} compared, {bad} differ; skipped {sum(skipped.values())}: {skipped}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

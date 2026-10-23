"""Compare bash-lite with real bash 5.2 + GNU tools on a corpus of scripts.

Each case runs as `bash case.sh` in an empty working directory (bash-lite: /home/ada/work on the simulated
machine; real: a fresh temp dir with HOME pointing at its parent), and stdout, stderr and the exit status are
compared after replacing the real temp path with /home/ada.

    python scripts/diff_bash.py --gnu <dir with GNU bash, coreutils, sed, gawk, grep, findutils, jq>  [corpus.sh ...]

A corpus file holds cases separated by lines `### name`. Default corpus: tests/bash_corpus/*.sh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cases_from(path: Path):
    name, buf = None, []
    for line in path.read_text().splitlines(keepends=True):
        m = re.match(r"### (.+)", line)
        if m:
            if name is not None:
                yield name, "".join(buf)
            name, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def run_sim(script: str, fixed_time: bool = True):
    from generator.bashlite.world import Session, World
    w = World(seed=7)
    w.kernel.strict = True
    s = Session(w, cwd="/home/ada")
    fs = w.local.fs
    fs.mkdir("/home/ada/work", "/", 0o755, 1000, 1000)
    fs.write_file("/home/ada/work/case.sh", script, "/", 0o644, 1000, 1000)
    try:
        r = s.run("cd work && LC_ALL=C bash case.sh > ../.out 2> ../.err < /dev/null; echo \"$?\" > ../.status")
    except Exception as e:  # noqa: BLE001
        import traceback
        return "", f"bash-lite crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}", -1

    def read(p):
        try:
            return bytes(fs.lookup(p).data).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""
    out, err = read("/home/ada/.out"), read("/home/ada/.err")
    st = read("/home/ada/.status").strip()
    if r["error"]:
        err += f"\n[{r['error']}]"
    if r["text"]:
        err += r["text"]
    return out, err, int(st) if st.isdigit() else -1


def run_real(script: str, gnu: str):
    with tempfile.TemporaryDirectory(prefix="dbash-") as tmp:
        tmp = os.path.realpath(tmp)
        home = os.path.join(tmp, "home", "ada")
        work = os.path.join(home, "work")
        os.makedirs(work)
        Path(work, "case.sh").write_text(script)
        env = {"PATH": f"{gnu}:/usr/bin:/bin", "HOME": home, "LC_ALL": "C", "LANG": "C", "TZ": "UTC",
               "USER": "ada", "LOGNAME": "ada", "TERM": "dumb", "SHELL": "/bin/bash"}
        try:
            p = subprocess.run([os.path.join(gnu, "bash"), "case.sh"], cwd=work, env=env, capture_output=True,
                               timeout=20, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "", "[timeout]", -1
        out = p.stdout.decode("utf-8", "replace").replace(tmp + "/home/ada", "/home/ada").replace(tmp, "")
        err = p.stderr.decode("utf-8", "replace").replace(tmp + "/home/ada", "/home/ada").replace(tmp, "")
        err = err.replace(os.path.join(gnu, "bash"), "bash")
        out, err = out.replace(gnu, "/usr/bin"), err.replace(gnu, "/usr/bin")
        return out, err, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="*")
    ap.add_argument("--gnu", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-stderr", action="store_true")
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    files = [Path(c) for c in args.corpus] or sorted((ROOT / "tests" / "bash_corpus").glob("*.sh"))
    total = bad = 0
    for f in files:
        for name, script in cases_from(f):
            if args.only and args.only not in name:
                continue
            total += 1
            so, se, ss = run_sim(script)
            ro, re_, rs = run_real(script, args.gnu)
            diffs = []
            if so != ro:
                diffs.append(("stdout", ro, so))
            if not args.no_stderr and se != re_:
                diffs.append(("stderr", re_, se))
            if ss != rs:
                diffs.append(("status", str(rs), str(ss)))
            if diffs:
                bad += 1
                print(f"FAIL {f.name}:{name}")
                for what, real, sim in diffs:
                    print(f"  --- {what} (real) ---\n{_indent(real)}\n  --- {what} (bash-lite) ---\n{_indent(sim)}")
            elif args.v:
                print(f"ok   {f.name}:{name}")
    print(f"\n{total - bad}/{total} cases agree")
    sys.exit(1 if bad else 0)


def _indent(s):
    return "\n".join("    " + line for line in s.splitlines()) if s else "    (empty)"


if __name__ == "__main__":
    main()

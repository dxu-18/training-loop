"""Compare bash-lite's git with real git: run the same shell script in both and diff what they print.

    python scripts/diff_git.py tests/git_corpus/*.sh        # every case
    python scripts/diff_git.py case.sh -v                    # show both outputs

Both sides get the same identity, a fixed clock (GIT_AUTHOR_DATE / GIT_COMMITTER_DATE) and no colors (output is
piped), so commit ids must match exactly when objects are encoded the way git encodes them. Paths of the real
temporary directory are rewritten to bash-lite's /home/ada. A case may mark lines real git and bash-lite are allowed
to differ on with a trailing `#~` (version wording); those commands' output is not compared.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRELUDE = """exec 2>&1
export TZ=UTC GIT_AUTHOR_DATE='1792661167 +0000' GIT_COMMITTER_DATE='1792661167 +0000'
export GIT_AUTHOR_NAME='Ada Lovelace' GIT_AUTHOR_EMAIL=ada@example.com
export GIT_COMMITTER_NAME='Ada Lovelace' GIT_COMMITTER_EMAIL=ada@example.com
git config --global init.defaultBranch main
git config --global user.name 'Ada Lovelace'
git config --global user.email ada@example.com
git config --global advice.detachedHead false
"""


def marked(src: str) -> str:
    """Put a marker before every top-level command line so the outputs can be split per command."""
    out = []
    for i, ln in enumerate(src.split("\n")):
        s = ln.strip()
        if not s or s.startswith("#") or ln.startswith((" ", "\t")):
            out.append(ln)
            continue
        out.append(f"echo '@@@ {i}'")
        out.append(ln)
    return "\n".join(out)


def run_real(script: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home" / "ada"
        home.mkdir(parents=True)
        env = {"HOME": str(home), "PATH": os.environ["PATH"], "LANG": "C", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
               "TERM": "dumb", "EDITOR": "true", "GIT_PAGER": "cat", "PAGER": "cat"}
        r = subprocess.run(["bash", "-c", PRELUDE + script], cwd=home, env=env, capture_output=True, text=True,
                           timeout=60, stdin=subprocess.DEVNULL)
        out = r.stdout + r.stderr
        return out.replace(os.path.realpath(str(home)), "/home/ada").replace(str(home), "/home/ada")


def run_lite(script: str) -> str:
    from generator.bashlite.world import Session, World
    import generator.bashlite.utils  # noqa: F401
    w = World(seed=1)
    s = Session(w)
    # piped like the real run (no terminal: no colors, no log decorations)
    body = PRELUDE.replace("exec 2>&1\n", "") + script
    r = s.run("export EDITOR=true GIT_PAGER=cat TERM=dumb\n( " + body + "\n) 2>&1 | cat")
    text = r["text"].replace("\r\n", "\n").replace("\r", "\n")     # as Python's text mode reads the real run
    if r["error"]:
        text += f"\nERROR: {r['error']}\n"
    return text


def by_command(text: str) -> dict:
    out = {}
    cur = None
    for ln in text.split("\n"):
        if ln.startswith("@@@ "):
            cur = ln
            out[cur] = []
            continue
        if cur is not None:
            out[cur].append(ln)
    return out


def compare(path: Path, verbose=False) -> int:
    src = path.read_text()
    script = marked(src)
    real = by_command(run_real(script))
    lite = by_command(run_lite(script))
    skip = {i for i, ln in enumerate(src.split("\n")) if " #~" in ln}
    bad = 0
    for key, exp in real.items():
        idx = int(key.split()[1])
        got = lite.get(key)
        if idx in skip:
            continue
        exp_t = "\n".join(exp).rstrip()
        got_t = "\n".join(got if got is not None else ["<missing>"]).rstrip()
        if exp_t != got_t:
            bad += 1
            print(f"--- {path.name}:{idx + 1}: {src.split(chr(10))[idx].strip()[:70]}")
            import difflib
            for d in difflib.unified_diff(exp_t.split("\n"), got_t.split("\n"), "real", "lite", lineterm="", n=1):
                print("    " + d)
    if verbose:
        print("=== real\n" + "\n".join(sum(real.values(), [])))
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    total = 0
    for a in args:
        total += compare(Path(a), verbose)
    print(f"{len(args)} case(s), {total} differing command(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

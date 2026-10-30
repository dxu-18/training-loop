"""Build generator/data/instructions.json: what each problem's Instructions list can show.

For every section: generate problems from every template (every catalog item, several seeds), collect the names the
reference solutions use (generator/instructions.py extracts them: APIs and syntax, or Bash commands, options and
keys), and record where each name is introduced (its first module, and the first tier there). Then describe every
name: the real library's docstring (numpy, pandas, torch: run in an interpreter that has them, --python), bash-lite's
manual pages for commands and options, and the hand-written notes in instruction_notes.py (which win).

    python scripts/make_instructions.py --python /path/to/venv-with-numpy-pandas-torch/bin/python

Without torch the PyTorch descriptions already in the data file are kept. Names nobody describes are listed; they
simply don't appear in Instructions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generator  # noqa: E402,F401  (registers the templates)
from generator import core  # noqa: E402
from generator.core import TIERS, build, modules, templates  # noqa: E402
from generator.instructions import DATA_FILE, canon, names_of  # noqa: E402
import instruction_notes as notes  # noqa: E402

SECTIONS = ("numpy", "pandas", "pytorch", "bash")


def problems(section, seeds):
    for t in templates(section):
        core._PICKED.clear()
        core._FORCE_ITEM[0] = None
        yield t, build(t, 0)
        n_items = core._PICKED[0] if core._PICKED else 0
        for i in range(n_items):
            core._FORCE_ITEM[0] = i
            try:
                for s in range(2):
                    yield t, build(t, s)
            finally:
                core._FORCE_ITEM[0] = None
        for s in range(1, seeds if t.tier != "challenge" else 3):
            yield t, build(t, s)


def guide_modules(section):
    """canonical name -> the first module whose COMP_GUIDE code lists it (where the guide introduces it)."""
    import guide_coverage as gc
    first = {}
    if section == "bash":
        for mid, spec in gc.GUIDE_BASH.items():
            for f in spec.split(", "):
                first.setdefault(f.strip(), mid)
        return first
    for mid, code in sorted(gc.guide_blocks(section).items(), key=lambda kv: tuple(map(int, kv[0].split(".")))):
        api, _ = gc.keys(code, section, gc.user_names(code))
        for k in api:
            first.setdefault(gc.canon(k), mid)
    return first


def collect(section, seeds):
    """-> module order, {name: [module, tier] where it is introduced}, Counter of keys.
    The module is the guide's (first module listing the name), else the first module whose solutions use it; the tier
    is the first tier of that module whose solutions use it (lite if none does)."""
    order = modules(section)
    rank = {m: i for i, m in enumerate(order)}
    guide = guide_modules(section)
    first_use: dict[str, tuple] = {}
    tiers: dict[tuple, int] = {}
    keys: Counter = Counter()
    for t, p in problems(section, seeds):
        for key in names_of(p):
            keys[key] += 1
            c = canon(key, section)
            where = (rank[t.module], TIERS.index(t.tier))
            if c not in first_use or where < first_use[c]:
                first_use[c] = where
            tk = (c, t.module)
            tiers[tk] = min(tiers.get(tk, 9), TIERS.index(t.tier))
    intro = {}
    for c, (m, tr) in first_use.items():
        mod = guide.get(c) if guide.get(c) in rank else order[m]
        intro[c] = [mod, tiers.get((c, mod), 0)]
    return order, intro, keys


def bash_doc(key):
    """Commands and their options from bash-lite's manual pages, builtins and theirs from `help`."""
    import re
    from generator.bashlite.helptext import BUILTIN_HELP
    from generator.bashlite.utils import manpages
    words = key.split(" ")
    cmd = manpages.ALIASES.get(words[0], words[0])
    page = manpages.PAGES.get(cmd)
    builtin = BUILTIN_HELP.get(words[0])
    if len(words) == 1:
        if page:
            return {"n": key, "s": "", "d": page[1][0].upper() + page[1][1:] + "."}
        if builtin:
            return {"n": key, "s": "", "d": builtin[1]}
        return None
    if not words[1].startswith("-"):
        return None
    flag = words[1]
    if cmd in ("head", "tail") and re.match(r"^-\d+$", flag):
        n = flag[1:]
        return {"n": key, "s": "", "d": f"{'First' if cmd == 'head' else 'Last'} {n} line{'s' if n != '1' else ''} "
                                          f"(short for {cmd} -n {n})."}
    if page:
        for flags, text in page[4]:
            names = [f.strip().split("=")[0].split("[")[0].split(" ")[0] for f in flags.split(",")]
            if flag in names:
                return {"n": key, "s": "", "d": text[0].upper() + text[1:].rstrip(".") + "."}
    if builtin:
        for m in re.finditer(r"^\s+(-\w)\s+(?:\w+\s+)?(.+)$", builtin[2], re.M):
            if m.group(1) == flag:
                text = m.group(2).strip()
                return {"n": key, "s": "", "d": text[0].upper() + text[1:].rstrip(".") + "."}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="interpreter with numpy, pandas and torch")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    old = json.loads(DATA_FILE.read_text()) if DATA_FILE.exists() else {}
    out = dict(old)
    found = {}
    for section in SECTIONS:
        if args.only and section != args.only:
            continue
        order, intro, keys = collect(section, args.seeds)
        found[section] = (order, intro, keys)
    want = {s: sorted(k for k in keys if not k.startswith("syntax:")) for s, (_, _, keys) in found.items() if s != "bash"}
    auto = {}
    if want:
        r = subprocess.run([args.python, str(ROOT / "scripts" / "_docstrings.py")], input=json.dumps(want),
                           capture_output=True, text=True, check=True)
        auto = json.loads(r.stdout)
    for section, (order, intro, keys) in found.items():
        docs = {}
        prev = old.get(section, {}).get("docs", {})
        manual = notes.NOTES.get(section, {})
        for key in keys:
            if key in manual:
                n, s, d = manual[key]
                if d:
                    docs[key] = {"n": n, "s": s, "d": d}
                continue
            if section == "bash":
                e = bash_doc(key)
                words = key.split(" ")
                if e is None and len(words) == 3 and words[2].startswith("-"):   # pip install -q -> pip -q
                    short = f"{words[0]} {words[2]}"
                    e = ({"n": key, "s": "", "d": manual[short][2]} if short in manual and manual[short][2]
                         else bash_doc(short))
                    if e:
                        e = dict(e, n=key)
            else:
                e = auto.get(section, {}).get(key) or (prev.get(key) if section not in auto else None)
            if e:
                docs[key] = e
        alt = {}
        for key in sorted(docs, key=lambda k: -keys[k]):
            alt.setdefault(canon(key, section), key)
        out[section] = {"modules": order, "intro": intro, "docs": docs, "canon": alt}
        missing = sorted((k for k in keys if k not in docs), key=lambda k: -keys[k])
        print(f"{section}: {len(keys)} names, {len(docs)} described; undescribed (most used first): "
              f"{', '.join(f'{k} ({keys[k]})' for k in missing[:60])}")
    DATA_FILE.parent.mkdir(exist_ok=True)
    DATA_FILE.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    print(f"wrote {DATA_FILE.relative_to(ROOT)} ({DATA_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

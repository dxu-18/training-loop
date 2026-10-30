"""Package the generator for the website.

Writes docs/generator.zip (the `generator` package, unpacked by Pyodide in the
browser) and docs/build.json: a content hash for cache-busting plus the catalog of
topics and their modules (parsed from COMP_GUIDE.md), each marked live once every
difficulty tier has templates.

    python scripts/build.py
"""

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator.core import TIERS, is_live, templates  # noqa: E402

# COMP_GUIDE section number -> topic id used by the generator and the site
TOPIC_IDS = {1: "numpy", 2: "pandas", 3: "pytorch", 4: "sklearn", 5: "matplotlib", 6: "huggingface",
             7: "bash", 8: "git", 9: "vim", 10: "sql", 11: "latex", 12: "docker"}


def sources() -> list[Path]:
    """The package's code and its data files (generator/data/*.json: Instructions)."""
    gen = ROOT / "generator"
    return sorted([p for p in gen.rglob("*.py")] + [p for p in (gen / "data").glob("*.json")])


def source_hash() -> str:
    digest = hashlib.sha256()
    for p in sources():
        digest.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()[:12]


def catalog() -> dict:
    """{topic_id: {"title", "modules": [{"id", "title", "live", "templates"}]}} from COMP_GUIDE headings."""
    topics, current = {}, None
    for line in (ROOT / "COMP_GUIDE.md").read_text().splitlines():
        if m := re.match(r"^## (\d+)\. (.+)$", line):
            tid = TOPIC_IDS[int(m[1])]
            current = topics[tid] = {"title": m[2].strip(), "modules": []}
        elif (m := re.match(r"^### (\d+\.\d+) (.+)$", line)) and current is not None:
            mid, title = m[1], m[2].strip()
            if mid.endswith(".00"):  # e.g. "10.00 Practice schema": reference material, not a module
                continue
            tid = next(k for k, v in topics.items() if v is current)
            current["modules"].append({"id": mid, "title": title, "live": is_live(tid, mid),
                                       "templates": {t: len(templates(tid, mid, t)) for t in TIERS}})
    return topics


def main():
    files = sources()
    zpath = ROOT / "docs" / "generator.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))  # reproducible
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    topics = catalog()
    build = {"hash": source_hash(), "topics": topics}
    (ROOT / "docs" / "build.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
    live = [f"{t}:{sum(m['live'] for m in v['modules'])}/{len(v['modules'])}" for t, v in topics.items()]
    print(f"wrote {zpath.relative_to(ROOT)} ({len(files)} files, {zpath.stat().st_size // 1024} KB), "
          f"build {build['hash']}, live modules {' '.join(live)}")


if __name__ == "__main__":
    main()

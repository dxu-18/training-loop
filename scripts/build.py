"""Package the generator for the website.

Writes docs/generator.zip (the `generator` package, unpacked by Pyodide in the
browser) and docs/build.json (a content hash for cache-busting plus the
template counts per section, which the topic list uses to know what's live).

    python scripts/build.py
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generator.core import REGISTRY, TIERS  # noqa: E402


def sources() -> list[Path]:
    return sorted(p for p in (ROOT / "generator").rglob("*.py"))


def source_hash() -> str:
    digest = hashlib.sha256()
    for p in sources():
        digest.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()[:12]


def main():
    files = sources()
    zpath = ROOT / "docs" / "generator.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            info = zipfile.ZipInfo(rel, date_time=(2020, 1, 1, 0, 0, 0))  # reproducible zip
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    sections = {s: {t: len(REGISTRY[s][t]) for t in TIERS} for s in REGISTRY}
    build = {"hash": source_hash(), "sections": sections}
    (ROOT / "docs" / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    total = sum(sum(v.values()) for v in sections.values())
    print(f"wrote {zpath.relative_to(ROOT)} ({len(files)} files, {zpath.stat().st_size // 1024} KB), "
          f"build {build['hash']}, {total} templates: {sections}")


if __name__ == "__main__":
    main()

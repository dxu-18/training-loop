"""postgres-lite against recorded PostgreSQL 16 output: tests/sql_corpus/NAME.sql holds blocks of psql input
(separated by `-- ===` lines) run on the shop database (shop.sql); NAME.json holds what real psql 16 / PostgreSQL 16.2
printed for each block. Unordered results may come in another order; float results may differ in the 15th+ digit
(the recording machine's FMA); EXPLAIN widths (width=) are estimates of their own and are ignored.

  python3 scripts/sql_corpus.py                       compare every corpus file, print the differences
  python3 scripts/sql_corpus.py --record [--psql P --host H --port N --user U]   re-record from a real server
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CORPUS = ROOT / "tests" / "sql_corpus"
NOW_US = 1717243200 * 10 ** 6


def blocks(path):
    src = Path(path).read_text()
    return [b.strip("\n") + "\n" for b in re.split(r"^-- ===.*$", src, flags=re.M) if b.strip()]


def lite(sql, shop=None):
    from generator.postgreslite.server import Cluster
    from generator.postgreslite.psql import Psql, Env

    class E(Env):
        def now_us(self):
            return NOW_US
    cl = Cluster()
    cl.create_database("dtest", "postgres")
    p = Psql(cl, "dtest", "postgres", E(), interactive=False, quiet=True)
    p.run_script(shop if shop is not None else (CORPUS / "shop.sql").read_text(), "shop")
    p.take_output()
    p.quiet = False
    p.run_script(sql, None)
    return "".join(t for _, t in p.take_output())


def oracle(args, sql, db="dtest"):
    env = dict(os.environ, PGTZ="Etc/UTC", PGDATESTYLE="ISO, MDY", PAGER="", PSQL_PAGER="", LC_ALL="C.UTF-8",
               COLUMNS="80")
    base = [args.psql, "-h", args.host, "-p", str(args.port), "-U", args.user, "-X", "-P", "pager=off"]
    subprocess.run(base + ["-d", "postgres", "-q", "-c", f"DROP DATABASE IF EXISTS {db}", "-c",
                           f"CREATE DATABASE {db}"], env=env, capture_output=True)
    subprocess.run(base + ["-d", db, "-q"], input=(CORPUS / "shop.sql").read_text(), text=True, env=env,
                   capture_output=True)
    import tempfile
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run(base + ["-d", db], input=sql, text=True, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, cwd=cwd)
    return r.stdout


COST = re.compile(r"\(cost=([0-9.]+)\.\.([0-9.]+) rows=(\d+) width=\d+\)")


def same(expected, got, tolerance=0.02):
    """Equal output; for plans, costs and row estimates may differ by `tolerance` (relative) and widths at all."""
    from sql_templates import loose, norm
    if expected == got:
        return True
    if "QUERY PLAN" in expected:
        ne, ng = [COST.findall(x) for x in (expected, got)]
        if len(ne) != len(ng):
            return False
        for a, b in zip(ne, ng):
            for x, y in zip(map(float, a), map(float, b)):
                if abs(x - y) > max(0.015, tolerance * abs(x)):
                    return False
        e, g = COST.sub("(cost)", expected), COST.sub("(cost)", got)
        # the plan header's width follows the widest line
        e, g = re.sub(r"-{10,}", "-", e), re.sub(r"-{10,}", "-", g)
        e, g = re.sub(r" +QUERY PLAN +", "QUERY PLAN", e), re.sub(r" +QUERY PLAN +", "QUERY PLAN", g)
        e, g = re.sub(r" +\n", "\n", e), re.sub(r" +\n", "\n", g)
        return e == g
    return norm(expected) == norm(got) or norm(loose(expected)) == norm(loose(got))


def compare(name):
    """-> [(block index, sql, expected, got)] for the blocks that differ."""
    exp = json.loads((CORPUS / f"{name}.json").read_text())
    out = []
    for i, (sql, want) in enumerate(zip(blocks(CORPUS / f"{name}.sql"), exp)):
        try:
            got = lite(sql)
        except Exception as e:  # noqa: BLE001
            got = f"CRASH {e!r}"
        if not same(want, got):
            out.append((i, sql, want, got))
    return out


def names():
    return sorted(p.stem for p in CORPUS.glob("*.sql") if p.stem != "shop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--psql", default=os.environ.get("PSQL", "psql"))
    ap.add_argument("--host", default=os.environ.get("PGHOST", "/var/run/postgresql"))
    ap.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    args = ap.parse_args()
    bad = 0
    for name in names():
        if args.record:
            outs = [oracle(args, b) for b in blocks(CORPUS / f"{name}.sql")]
            (CORPUS / f"{name}.json").write_text(json.dumps(outs, indent=0, ensure_ascii=False) + "\n")
            print(f"{name}: recorded {len(outs)} blocks")
            continue
        diffs = compare(name)
        bad += len(diffs)
        for i, sql, want, got in diffs:
            print(f"=== {name} block {i} differs")
            for ln in difflib.unified_diff(want.splitlines(), got.splitlines(), "postgres", "lite", lineterm="", n=1):
                print(ln)
        print(f"{name}: {len(blocks(CORPUS / f'{name}.sql')) - len(diffs)}/{len(blocks(CORPUS / f'{name}.sql'))} "
              f"blocks match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

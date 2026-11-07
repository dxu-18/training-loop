"""Check every SQL template (COMP_GUIDE §10): each catalog item x seeds builds, its reference solution passes its own
grader, and an idle answer fails. With --oracle, each problem's database and reference solution also run on a real
PostgreSQL 16 server and the psql output must match postgres-lite's (row order is ignored for unordered results;
EXPLAIN plans are compared without costs; last-digit float differences and \\timing values are ignored; problems
that start at the bash prompt or list the server's databases, roles or owners are skipped).

  python3 scripts/sql_templates.py [--modules 10.01,10.02] [--seeds 2] [--oracle] [--psql PATH] [--host /tmp] [--port 5432]
"""

from __future__ import annotations

import argparse
import difflib
import os
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generator import core  # noqa: E402
import generator.sections  # noqa: E402,F401
from generator.grader import grade  # noqa: E402
from generator.bashlite.live import reference_session  # noqa: E402


def problems(module=None, seeds=2):
    """(template, catalog item, seed, problem dict or None, build error or None) for every item x seed."""
    for tpl in core.templates("sql", module):
        core._PICKED.clear()
        try:
            core.build(tpl, 0)
        except Exception:  # noqa: BLE001
            pass
        items = core._PICKED[0] if core._PICKED else 1
        for item in range(items):
            for seed in range(seeds):
                core._FORCE_ITEM[0] = item if core._PICKED else None
                try:
                    yield tpl, item, seed, core.build(tpl, random.Random(seed * 7919 + item).randrange(2**31)) \
                        .to_dict(), None
                except Exception as e:  # noqa: BLE001
                    yield tpl, item, seed, None, e
                finally:
                    core._FORCE_ITEM[0] = None


def norm(text):
    """Sort the data rows of every table (unordered results may come in another order)."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        if re.match(r"^-+(\+-+)*$", lines[i]):
            j = i + 1
            while j < len(lines) and not re.match(r"^\(\d+ rows?\)$", lines[j]):
                j += 1
            out += sorted(lines[i + 1:j])
            i = j
            continue
        i += 1
    return "\n".join(out)


def loose(text):
    """Float results with 15+ significant digits to 12 (the oracle may be an ARM build whose FMA contraction changes
    the last digit), with table widths and rules collapsed."""
    def short(m):
        digits = re.sub(r"[^0-9]", "", m.group(0).split("e")[0]).lstrip("0")
        return format(float(m.group(0)), ".12g") if len(digits) >= 15 else m.group(0)
    text = re.sub(r"-?\d+\.\d+(?:e[+-]\d+)?", short, text)
    text = re.sub(r"Time: [0-9.]+ ms", "Time: ms", text)
    text = re.sub(r"-{2,}", "-", text)
    return "\n".join(re.sub(r" +", " ", ln).strip() for ln in text.split("\n"))


def oracle_run(args, setup, solution, db="tplcheck"):
    env = dict(os.environ, PGTZ="Etc/UTC", PGDATESTYLE="ISO, MDY", PAGER="", PSQL_PAGER="", LC_ALL="C.UTF-8",
               COLUMNS="80")
    base = [args.psql, "-h", args.host, "-p", str(args.port), "-U", args.user, "-X", "-P", "pager=off"]
    subprocess.run(base + ["-d", "postgres", "-q", "-c", f"DROP DATABASE IF EXISTS {db}", "-c",
                           f"CREATE DATABASE {db}"], env=env, capture_output=True)
    subprocess.run(base + ["-d", db, "-q"], input=setup + "\nANALYZE;\n", text=True, env=env, capture_output=True)
    import tempfile
    with tempfile.TemporaryDirectory() as cwd:    # \copy and \o write files: not into the repository
        r = subprocess.run(base + ["-d", db], input=solution, text=True, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, cwd=cwd)
    return r.stdout


def lite_run(setup, solution):
    from generator.sections.s10_sql._common import run_reference
    p, _ = run_reference(setup, "", "tplcheck")
    p.take_output()
    p.run_script(solution, None)
    return "".join(t for _, t in p.take_output())


def _drop_costs(text):
    text = re.sub(r"  \(cost=[^)]*\)", "", text)
    text = re.sub(r" \(actual [^)]*\)", "", text)
    text = re.sub(r"(Planning|Execution) Time: [0-9.]+ ms", r"\1 Time", text)
    text = re.sub(r"\n(Time: .*)", "", text)
    lines = text.split("\n")
    return "\n".join(ln for ln in lines if not re.match(r"^-+$", ln.strip()) and "QUERY PLAN" not in ln and
                     not ln.strip().startswith(("Heap Blocks", "Buckets", "Batches", "Sort Method", "Memory",
                                                "Rows Removed", "Heap Fetches")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--psql", default=os.environ.get("PSQL", "psql"))
    ap.add_argument("--host", default=os.environ.get("PGHOST", "/var/run/postgresql"))
    ap.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    args = ap.parse_args()
    mods = [m for m in args.modules.split(",") if m] or core.modules("sql")
    bad, n, odiff, ochecked = 0, 0, 0, 0
    t0 = time.time()
    for m in mods:
        for tpl, item, seed, p, err in problems(m, args.seeds):
            n += 1
            where = f"{m} {tpl.name}[{item}] seed {seed}"
            if err is not None:
                bad += 1
                print(f"CRASH {where}: {err!r}")
                continue
            try:
                r = grade(p, reference_session(p))
                if not r["correct"]:
                    bad += 1
                    print(f"REF  {where}: {r['message']}")
                if grade(p, "SELECT 42 AS x;")["correct"]:
                    bad += 1
                    print(f"IDLE {where} passes")
                spec = p["sql"]
                env_dependent = re.search(r"\\(l|conninfo|dt|dn|du)\b", p["solution"])  # owner, sockets, dbs
                if args.oracle and "#: psql" in p["setup"] and "#: psqlquit" not in p["setup"] and not env_dependent:
                    ochecked += 1
                    setup = spec["dbs"][spec["db"]]
                    o = oracle_run(args, setup, p["solution"])
                    lt = lite_run(setup, p["solution"])
                    if "EXPLAIN" in p["solution"].upper():
                        o, lt = _drop_costs(o), _drop_costs(lt)
                    if o != lt and norm(o) != norm(lt) and norm(loose(o)) != norm(loose(lt)):
                        odiff += 1
                        print(f"PG   {where} differs from PostgreSQL:")
                        for ln in list(difflib.unified_diff(o.splitlines(), lt.splitlines(), "pg", "lite",
                                                            lineterm="", n=1))[:30]:
                            print("     " + ln)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"CRASH {where}: {e!r}")
            sys.stdout.flush()
    print(f"{n} problems checked in {time.time() - t0:.0f}s, {bad} failing" +
          (f"; {ochecked} compared with PostgreSQL, {odiff} differ" if args.oracle else ""))
    sys.exit(1 if bad or odiff else 0)


if __name__ == "__main__":
    main()

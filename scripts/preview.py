"""Print sample questions from the command line.

    python scripts/preview.py numpy --tier hard -n 3
    python scripts/preview.py numpy --template stable_softmax --solution
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator import TIERS, templates  # noqa: E402
from generator.core import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--template", help="only this template (function name)")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--solution", action="store_true", help="also print the reference solution")
    args = ap.parse_args()

    pool = [t for t in templates(args.section, args.tier) if not args.template or t.name == args.template]
    if not pool:
        sys.exit("no matching templates")
    rng = random.Random(args.seed)
    for _ in range(args.n):
        tpl = rng.choice(pool)
        p = build(tpl, rng.randrange(2**31))
        print(f"── {p.section} · {p.tier} · {p.template} (module {p.module}, seed {p.seed})")
        print(p.prompt, "\n")
        print("Given:\n" + "\n".join("    " + l for l in p.setup.splitlines()))
        if args.solution:
            print("Solution:\n" + "\n".join("    " + l for l in p.solution.splitlines()))
        print()


if __name__ == "__main__":
    main()

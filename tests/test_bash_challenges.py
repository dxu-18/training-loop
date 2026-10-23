"""Cross-check every Bash challenge's reference solution against an independent Python computation.

Each brute force looks at the simulated machine the setup builds (files, mtimes, modes, processes, PATH, history)
and computes the expected output; the reference solution (bash, run on bash-lite) must print exactly that, on the
visible setup and on the hidden tests.
"""

import hashlib
import json
import re

import pytest

import generator.sections.s07_bash  # noqa: F401
from generator.bashlite.check import answers, normalize, prepare_case, run_case
from generator.core import build, templates

CHALLENGES = [t for t in templates("bash") if t.tier == "challenge"]
BRUTE = {}


def brute(fn):
    BRUTE[fn.__name__] = fn
    return fn


class Box:
    """What a brute force can look at: the machine after the setup ran."""

    def __init__(self, world, sess, spec, problem, setup):
        self.world, self.sess, self.spec, self.problem, self.setup = world, sess, spec, problem, setup
        self.fs = world.local.fs
        self.cwd = sess.proc.cwd

    def abs(self, path, host=None):
        if path.startswith("/"):
            return path
        return self.cwd.rstrip("/") + "/" + path

    def node(self, path, fs=None):
        return (fs or self.fs).lookup(self.abs(path))

    def files(self, root=".", fs=None):
        """{relative path: text} of the regular files under `root` (recursively)."""
        fs = fs or self.fs
        base = self.abs(root).rstrip("/") if root != "." else self.cwd.rstrip("/")
        out = {}

        def walk(d, rel):
            node = fs.lookup(d)
            for name in fs.listdir(node):
                p = f"{d}/{name}"
                ch = fs.lookup(p, follow=False)
                r = f"{rel}{name}"
                if ch.type == "d":
                    walk(p, r + "/")
                elif ch.type == "f":
                    out[r] = bytes(ch.data).decode("utf-8", "replace")
        walk(base, "")
        return out

    def text(self, path):
        return bytes(self.node(path).data).decode()

    def lines(self, path):
        return self.text(path).splitlines()

    def var(self, name):
        return self.sess.sh.get(name)

    def num(self, pattern, src=None):
        return int(re.search(pattern, src if src is not None else self.problem["solution"]).group(1))


def _uniq_c(pairs):
    return "".join(f"{c:7d} {w}\n" for c, w in pairs)


# ---------------------------------------------------------------- 7.01

@brute
def kind_histogram(b):
    from generator.sections.s07_bash.m01_basics import KINDS
    counts = {}
    for n in b.lines("names.txt"):
        k = KINDS.get(n) or "missing"
        counts[k] = counts.get(k, 0) + 1
    return _uniq_c(sorted(((c, k) for k, c in counts.items()), key=lambda t: (-t[0], t[1])))


@brute
def path_shadowing(b):
    path = b.var("PATH").split(":")
    out = []
    for t in b.lines("tools.txt"):
        hits = []
        for d in path:
            try:
                node = b.fs.lookup(f"{d}/{t}")
            except Exception:  # noqa: BLE001
                continue
            if node.type == "f" and b.fs.can(node, 1, 1000, {1000}) and node.mode & 0o111:
                hits.append(f"{d}/{t}")
        out.append(f"{t} {hits[0] if hits else 'none'} {len(hits)}\n")
    return "".join(out)


# ---------------------------------------------------------------- 7.02

@brute
def sort_by_extension(b):
    paths = {"sorted"}
    for name in b.files("inbox"):
        ext = name.rsplit(".", 1)[1] if "." in name else "other"
        paths |= {f"sorted/{ext}", f"sorted/{ext}/{name}"}
    return {"post": "".join(p + "\n" for p in sorted(paths, key=lambda s: s.encode()))}


@brute
def latest_links(b):
    out = []
    runs = sorted(n for n in b.fs.listdir(b.node("runs")))
    for r in runs:
        ck = [n for n in b.fs.listdir(b.node(f"runs/{r}")) if n.startswith("ckpt_")]
        newest = sorted(ck, key=lambda n: (-b.node(f"runs/{r}/{n}").mtime, n))[0]
        out.append(f"runs/{r}/latest -> {newest}\n")
    return "".join(out)


# ---------------------------------------------------------------- 7.03

@brute
def snapshot_report(b):
    old, new = b.files("old"), b.files("new")
    out = [f"removed {p}" for p in sorted(set(old) - set(new))]
    out += [f"added {p}" for p in sorted(set(new) - set(old))]
    out += [f"changed {p}" for p in sorted(set(old) & set(new)) if old[p] != new[p]]
    return "".join(x + "\n" for x in out)


@brute
def duplicate_groups(b):
    groups = {}
    for p, t in b.files("photos").items():
        groups.setdefault(t, []).append(f"photos/{p}")
    lines = sorted(" ".join(sorted(g)) for g in groups.values() if len(g) > 1)
    return "".join(x + "\n" for x in lines)


# ---------------------------------------------------------------- 7.04

@brute
def space_hogs(b):
    sums = {}
    for p, t in b.files("store").items():
        d = "store/" + p.rsplit("/", 1)[0] if "/" in p else "store"
        sums[d] = sums.get(d, 0) + len(t.encode())
    top = sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return "".join(f"{v} {k}\n" for k, v in top)


@brute
def world_writable(b):
    files = sorted(b.files("shared"))
    bad = [f"shared/{p}" for p in files if b.node(f"shared/{p}").mode & 0o002]
    out = "".join(f"fixed {p}\n" for p in bad) + "0 world-writable files left\n"
    post = "".join(f"{oct(b.node('shared/' + p).mode & 0o775)[2:]} shared/{p}\n" for p in sorted(files))
    return {"stdout": out, "post": post}


# ---------------------------------------------------------------- 7.05

@brute
def todo_report(b):
    out = []
    for p, t in sorted(b.files().items()):
        parts = p.split("/")
        if not p.endswith(".py") or ".venv" in parts[:-1] or "build" in parts[:-1]:
            continue
        nums = [str(i) for i, ln in enumerate(t.split("\n"), 1) if "TODO" in ln]
        if nums:
            out.append(f"{p} {len(nums)} {','.join(nums)}")
    return "".join(x + "\n" for x in sorted(out))


@brute
def name_clashes(b):
    by = {}
    for p in b.files():
        by.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    lines = sorted(f"{n}: {' '.join(sorted(ps))}" for n, ps in by.items() if len(ps) > 1)
    return "".join(x + "\n" for x in lines)


# ---------------------------------------------------------------- 7.06

@brute
def word_frequencies(b):
    k = b.num(r"head -(\d+)")
    counts = {}
    for w in re.findall(r"[A-Za-z]+", b.text("book.txt")):
        w = w.lower()
        counts[w] = counts.get(w, 0) + 1
    return _uniq_c(sorted(((c, w) for w, c in counts.items()), key=lambda t: (-t[0], t[1]))[:k])


@brute
def leaderboard(b):
    best = {}
    for r in json.loads(b.text("scores.json")):
        key = (r["user"], r["task"])
        best[key] = max(best.get(key, -1), r["score"])
    tot = {}
    for (u, _), s in best.items():
        tot[u] = tot.get(u, 0) + s
    return "".join(f"{u} {t}\n" for u, t in sorted(tot.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------- 7.07

@brute
def stream_split(b):
    out, err = [], []
    for ln in b.lines("service.sh")[1:]:
        m = re.match(r'echo "(.*)"( >&2)?$', ln)
        (err if m.group(2) else out).append(m.group(1))
    errors = [x for x in err if x.startswith("ERROR")]
    text = f"stdout {len(out)}\nstderr {len(err)}\n{len(errors)}\n" + (errors[-1] + "\n" if errors else "")
    body = "".join(x + "\n" for x in sorted(out + err, key=lambda s: s.encode()))
    return {"stdout": text, "post": hashlib.md5(body.encode()).hexdigest() + "  -\n"}


@brute
def requirement_drift(b):
    def names(f):
        return {re.sub(r"[<>=].*", "", ln).lower() for ln in b.lines(f)}
    prod, dev = names("prod.txt"), names("dev.txt")
    out = [f"only in prod: {n}" for n in sorted(prod - dev)] + [f"only in dev: {n}" for n in sorted(dev - prod)]
    return "".join(x + "\n" for x in out)


# ---------------------------------------------------------------- 7.08

@brute
def zero_pad(b):
    names = []
    for n in b.files("shots"):
        m = re.fullmatch(r"(.+)_(\d+)\.(\w+)", n)
        names.append(f"{m.group(1)}_{int(m.group(2)):05d}.{m.group(3)}" if m else n)
    return {"post": "".join(x + "\n" for x in sorted(names, key=lambda s: s.encode()))}


@brute
def version_bumps(b):
    ma, mi, pa = map(int, b.text("VERSION").strip().split("."))
    out = []
    for bump in b.lines("bumps.txt"):
        if bump == "major":
            ma, mi, pa = ma + 1, 0, 0
        elif bump == "minor":
            mi, pa = mi + 1, 0
        else:
            pa += 1
        out.append(f"{bump} -> {ma}.{mi}.{pa}")
    return "".join(x + "\n" for x in out)


# ---------------------------------------------------------------- 7.09

@brute
def column_sums(b):
    rows = [list(map(int, ln.split())) for ln in b.lines("grid.txt")]
    sums = [sum(c) for c in zip(*rows)]
    best = max(range(len(sums)), key=lambda i: (sums[i], -i))
    return " ".join(map(str, sums)) + f"\ncolumn {best}\n"


@brute
def port_allocator(b):
    nxt = b.num(r"next=(\d+)")
    got, out = {}, []
    for ln in b.lines("requests.txt"):
        user, port = ln.split()
        if (user, port) not in got:
            got[(user, port)] = nxt
            nxt += 1
        out.append(f"{user} {port} -> {got[(user, port)]}")
    return "".join(x + "\n" for x in out)


# ---------------------------------------------------------------- 7.10

@brute
def retry_backoff(b):
    tries = b.num(r"i <= (\d+)")
    need = b.num(r"n \+ 1 < (\d+)", b.text("flaky.sh"))
    out, delay = [], 1
    for i in range(1, tries + 1):
        if i >= need:
            out += [f"attempt {i} ok", f"succeeded on try {i}"]
            break
        if i == tries:
            out.append(f"giving up after {tries} tries")
            break
        out.append(f"waiting {delay}s")
        delay *= 2
    return "".join(x + "\n" for x in out)


@brute
def validate_params(b):
    out = []
    for n, ln in enumerate(b.lines("params.env"), 1):
        key, _, val = ln.partition("=")
        if key.startswith("#"):
            continue
        if key in ("epochs", "batch_size", "seed") and not re.fullmatch(r"[0-9]+", val):
            out.append(f"line {n}: {key} must be a non-negative integer")
        elif key in ("lr", "dropout") and not re.fullmatch(r"[0-9]*\.?[0-9]+", val):
            out.append(f"line {n}: {key} must be a number")
        elif key == "name" and not val:
            out.append(f"line {n}: name must not be empty")
    return "".join(x + "\n" for x in out)


# ---------------------------------------------------------------- 7.11

@brute
def launcher(b):
    out, err = [], []
    usage = "usage: ./launch.sh [-e epochs] [-g gpu] config.yaml"
    for run in b.spec.run:
        args = run.split(";")[0].split()[1:]
        epochs, gpu, i, status, ops = "10", "0", 0, 0, None
        while i < len(args) and args[i].startswith("-"):
            if args[i] in ("-e", "-g"):
                if args[i] == "-e":
                    epochs = args[i + 1]
                else:
                    gpu = args[i + 1]
                i += 2
                continue
            err += [f"./launch.sh: illegal option -- {args[i][1:]}", usage]
            status = 1
            break
        if not status:
            ops = args[i:]
            if len(ops) != 1:
                err.append(usage)
                status = 1
            else:
                out.append(f"CUDA_VISIBLE_DEVICES={gpu} python train.py --config {ops[0]} --epochs {epochs}")
        out.append(f"exit {status}")
    return {"stdout": "".join(x + "\n" for x in out), "stderr": "".join(x + "\n" for x in err)}


@brute
def slug_library(b):
    import shlex
    out = []
    for run in b.spec.run:
        title = shlex.split(run.split("slugify ", 1)[1])[0]
        out.append(re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"))
    return "".join(x + "\n" for x in out)


# ---------------------------------------------------------------- 7.12

@brute
def cpu_by_user(b):
    tot, cnt = {}, {}
    for opts, argv in b.spec.procs:
        if argv[0] != "python3":
            continue
        u = opts.get("user", "ada")
        tot[u] = tot.get(u, 0.0) + float(opts.get("cpu", 0))
        cnt[u] = cnt.get(u, 0) + 1
    rows = sorted(((u, cnt[u], f"{tot[u]:.1f}") for u in tot), key=lambda r: (-float(r[2]), r[0]))
    return "".join(f"{u} {n} {t}\n" for u, n, t in rows)


@brute
def bounded_parallelism(b):
    k = b.num(r"running == (\d+)")
    tasks = [(ln.split()[0], int(ln.split()[1])) for ln in b.lines("tasks.txt")]
    t, running, finished = 0, [], []      # running: [end, order, name] not yet reaped by wait -n
    for order, (name, d) in enumerate(tasks):
        if len(running) == k:
            done = min(running, key=lambda r: (r[0], r[1]))
            t = max(t, done[0])
            running.remove(done)
        running.append([t + d, order, name])
        finished.append((t + d, order, name))
    lines = [f"{n} finished at {e}s" for e, _, n in sorted(finished)]
    return "".join(x + "\n" for x in lines) + f"total {max(e for e, _, _ in finished)}s\n"


# ---------------------------------------------------------------- 7.13

@brute
def path_audit(b):
    seen, out = set(), []
    for d in b.var("PATH").split(":"):
        if d in seen:
            out.append(f"{d} duplicate")
            continue
        seen.add(d)
        try:
            node = b.fs.lookup(d)
        except Exception:  # noqa: BLE001
            node = None
        if node is None or node.type != "d":
            out.append(f"{d} missing")
            continue
        n = 0
        for name in b.fs.listdir(node):
            try:
                ch = b.fs.lookup(f"{d}/{name}")
            except Exception:  # noqa: BLE001
                continue
            if ch.type == "f" and b.fs.can(ch, 1, 1000, {1000}):
                n += 1
        out.append(f"{d} {n} executables")
    return "".join(x + "\n" for x in out)


@brute
def resolve_dotenv(b):
    vals = {}
    for ln in b.lines(".env"):
        if ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        vals[k] = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), ""), v)
    return "".join(f"{k}={vals[k]}\n" for k in sorted(vals))


# ---------------------------------------------------------------- 7.14

@brute
def mirror_check(b):
    local = b.files()
    rfs = b.world.hosts["gpu-box"].fs
    try:
        remote = b.files("/home/ada/project", fs=rfs)
    except Exception:  # noqa: BLE001
        remote = {}
    out = sorted([f"only local: {p}" for p in set(local) - set(remote)] +
                 [f"only remote: {p}" for p in set(remote) - set(local)])
    out += [f"differs: {p}" for p in sorted(set(local) & set(remote)) if local[p] != remote[p]]
    return "".join(x + "\n" for x in out)


@brute
def paginate(b):
    pages = {}
    for line in b.setup.split("\n"):
        m = re.match(r"#: url (\S+) (.*)$", line)
        if m:
            pages[m.group(1)] = json.loads(m.group(2))
    url, items = "https://api.example.com/runs?page=1", []
    while url:
        items += pages[url]["items"]
        url = pages[url]["next"]
    best = max(items, key=lambda r: r["acc"])        # max() keeps the first of equal values
    return f"{len(items)}\n{best['run']} {best['acc']}\n"


# ---------------------------------------------------------------- 7.15

def _pick(index, name, spec, have):
    info = index.get(name)
    if info is None:
        return None
    versions = sorted([info[0]] + info[1], key=lambda v: [int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v)])

    def key(v):
        return [int(x) if x.isdigit() else x for x in re.split(r"[.+-]", v)]

    def ok(v):
        if not spec:
            return True
        op, target = re.match(r"(==|>=|<)(.+)", spec).groups()
        return v == target if op == "==" else key(v) >= key(target) if op == ">=" else key(v) < key(target)
    if name in have and ok(have[name]):
        return have[name]
    good = [v for v in versions if ok(v)]
    return good[-1] if good else None


@brute
def resolve_report(b):
    from generator.bashlite.utils.pyenv import INDEX
    have, out = {"pip": "24.0"}, []
    for req in b.lines("requirements.txt"):
        m = re.match(r"([A-Za-z0-9_-]+)(.*)$", req)
        name, spec = m.group(1), m.group(2)
        v = _pick(INDEX, name, spec, have)
        if v is None:
            out.append(f"FAILED {req}")
            continue

        def install(n, ver):
            have[n] = ver
            for dep in INDEX[n][2]:
                if dep not in have:
                    install(dep, INDEX[dep][0])
        install(name, v)
        out.append(f"ok {req} -> {v}")
    return "".join(x + "\n" for x in out)


@brute
def package_matrix(b):
    from generator.bashlite.utils.pyenv import INDEX, INDEX_NAME
    by = {}
    for path, pkgs in b.spec.venvs:
        env = path.rsplit("/", 1)[-1]
        have = set()

        def add(n):
            if n in have:
                return
            have.add(n)
            for d in INDEX[n][2]:
                add(d)
        for p in pkgs:
            add(p)
        for n in have:
            by.setdefault(INDEX_NAME.get(n, n), []).append(env)
    lines = sorted((f"{p} {len(es)}: {' '.join(sorted(es))}" for p, es in by.items()), key=lambda s: s.encode())
    return "".join(x + "\n" for x in lines)


# ---------------------------------------------------------------- 7.16

@brute
def sweep_table(b):
    per = {}
    for name, text in b.files("logs").items():
        lr = name[2:].split("_s")[0]
        best = max(float(x) for x in re.findall(r"val_acc=([0-9.]+)", text))
        per.setdefault(lr, []).append(best)
    rows = [(lr, f"{sum(v) / len(v):.4f}", len(v)) for lr, v in per.items()]
    rows.sort(key=lambda r: (-float(r[1]), r[0]))
    return "".join(f"{lr} {m} {n}\n" for lr, m, n in rows)


@brute
def retention(b):
    k = b.num(r"tail -n \+(\d+)") - 1
    keep = []
    for run in sorted(b.fs.listdir(b.node("runs"))):
        names = b.fs.listdir(b.node(f"runs/{run}"))
        newest = sorted(names, key=lambda n: -b.node(f"runs/{run}/{n}").mtime)[:k]
        best = max(names, key=lambda n: (float(re.search(r"_val([0-9.]+)\.pt$", n).group(1)), n))
        keep += [f"runs/{run}/{n}" for n in set(newest) | {best}]
    return "".join(x + "\n" for x in sorted(keep))


# ---------------------------------------------------------------- 7.17

def _last_echo(b):
    return next(h for h in reversed(b.spec.history) if h.startswith("echo"))


@brute
def history_surgery(b):
    return _last_echo(b).replace("train.py", "eval.py", 1)[5:] + "\n"


@brute
def word_designators(b):
    w = _last_echo(b).split()
    return " ".join(w[3:5] + [w[-1]]) + "\n"


# ---------------------------------------------------------------- the checks

def test_every_bash_challenge_has_a_brute_force():
    missing = sorted({t.name for t in CHALLENGES} - set(BRUTE))
    assert not missing, missing


@pytest.mark.parametrize("tpl", CHALLENGES, ids=lambda t: f"{t.module}-{t.name}")
def test_matches_brute_force(tpl):
    for seed in range(6):
        p = build(tpl, seed).to_dict()
        for setup in [p["setup"]] + list(p["tests"]):
            world, sess, spec = prepare_case(p, setup)
            exp = BRUTE[tpl.name](Box(world, sess, spec, p, setup))
            exp = exp if isinstance(exp, dict) else {"stdout": exp}
            got = run_case(p, p["solution"], setup, echo=False)
            for a in answers(p):
                part = a.strip("<>")
                if part in exp:
                    assert normalize(got[part]) == normalize(exp[part]), (seed, part, setup[:400], got[part], exp[part])

"""7.04 Permissions, archives & disk usage."""

from ._common import DATASETS, MODELS, WORDS, P, catalog, challenge, files, q, template

M = "7.04"

MODES = [("rwxr-xr-x", "755"), ("rw-r--r--", "644"), ("rw-------", "600"), ("rwx------", "700"), ("rwxr-x---", "750"),
         ("rw-rw-r--", "664"), ("r--r--r--", "444"), ("rwxrwxr-x", "775"), ("r-x------", "500"), ("rw-r-----", "640")]


def _results(rng, n=4):
    run = rng.choice(["results", "outputs", "runs", "exp"])
    spec = {}
    for i in range(n):
        sub = rng.choice(["", "plots/", "metrics/", "logs/"])
        spec[f"{run}/{sub}{rng.choice(WORDS)}_{i}.{rng.choice(['txt', 'csv', 'json', 'log'])}"] = \
            "".join(f"{rng.choice(WORDS)} {rng.randint(1, 999)}\n" for _ in range(rng.randint(1, 6)))
    return run, spec


# ================================================================= lite

@template(M, "lite")
def permissions(rng):
    script = rng.choice(["run.sh", "train.sh", "setup.sh", "deploy.sh", "eval.sh"])
    f = rng.choice(["notes.md", "config.yaml", "params.json", "data.csv"])
    key = "~/.ssh/id_ed25519"
    sym, octal = rng.choice(MODES)
    d = rng.choice(["shared", "project", "team_data"])
    setup = (files({script: f"#!/bin/bash\necho \"{rng.choice(WORDS)} done\"\n", f: "x\n", f"{d}/a.txt": "a\n",
                    f"{d}/sub/b.txt": "b\n"}) + "\nmkdir -p ~/.ssh && echo 'PRIVATE' > ~/.ssh/id_ed25519 && "
             "chmod 644 ~/.ssh/id_ed25519" + f"\nchmod 664 {f}\nchmod -R go-w {d} && chmod -R u-w {d}")
    items = [
        ("ls -l", f"Show the permissions of `{script}` (long listing).", f"ls -l {script}"),
        ("chmod", f"Make `{script}` executable, then run it.", f"chmod +x {script}\n./{script}",
         {"require": ["chmod", f"./{script}"]}),
        ("chmod", f"Give `{f}` the permissions `{sym}` using octal notation.", f"chmod {octal} {f}",
         {"post": f"stat -c '%A %a' {f}"}),
        ("chmod", f"Without changing anything else, add execute permission for the owner and remove write "
                  f"permission for the group on `{f}` (one symbolic `chmod`).", f"chmod u+x,g-w {f}",
         {"post": f"stat -c '%A' {f}"}),
        ("chmod", "ssh refuses private keys others can read. Make `~/.ssh/id_ed25519` readable and writable by you "
                  "only.", f"chmod 600 {key}", {"post": f"stat -c '%A' {key}"}),
        ("chmod -R", f"Give yourself read and write permission on everything inside `{d}/`, recursively.",
         f"chmod -R u+rw {d}", {"post": f"find {d} -exec stat -c '%A %n' {{}} + | sort -k2"}),
        ("chown", f"Set the owner and group of `{f}` to `ada` and `ada` explicitly (`user:group` syntax).",
         f"chown ada:ada {f}", {"post": f"stat -c '%U:%G %n' {f}"}),
        ("sudo", f"`/srv/{d}` belongs to root. Take ownership of it and everything inside it with `sudo chown` "
                 f"(use `$USER`).", f'sudo chown -R "$USER" /srv/{d}',
         {"setup": f"sudo mkdir -p /srv/{d}/inner && sudo touch /srv/{d}/inner/x", "require": ["sudo", "chown -R"],
          "post": f"find /srv/{d} -exec stat -c '%U %n' {{}} + | sort -k2"}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def archives(rng):
    run, spec = _results(rng)
    arc = rng.choice(["archive", "results", "backup", "snapshot"])
    big = rng.choice(["big.csv", "events.log", "dump.json"])
    spec[big] = "".join(f"{rng.choice(WORDS)},{rng.randint(1, 99)}\n" for _ in range(rng.randint(10, 30)))
    word = spec[big].split(",")[0]
    setup = "#: cwd ~/work\n" + files(spec)
    pre = f"tar -czf {arc}.tar.gz {run}/"
    items = [
        ("tar -c", f"Archive `{run}/` into `{arc}.tar.gz` (gzip), listing files as they're added.",
         f"tar -czvf {arc}.tar.gz {run}/", {"require": ["tar -c", "tar -z", "tar -v", "tar -f"]}),
        ("tar -t", f"List what's inside `{arc}.tar.gz` without extracting it.", f"tar -tzf {arc}.tar.gz",
         {"setup": pre}),
        ("tar -x", f"Extract `{arc}.tar.gz` into the directory `restored/` (create it first).",
         f"mkdir restored && tar -xzf {arc}.tar.gz -C restored", {"setup": pre + f" && rm -r {run}",
                                                                  "post": "find restored | sort", "require": ["tar -x", "tar -C"]}),
        ("tar -x", f"Extract `{arc}.tar.gz` here, verbosely.", f"tar -xzvf {arc}.tar.gz", {"setup": pre + f" && rm -r {run}"}),
        ("tar -J", f"Archive `{run}/` into `{arc}.tar.xz` with xz compression, then list the archive.",
         f"tar -cJf {arc}.tar.xz {run}/ && tar -tJf {arc}.tar.xz", {"require": ["tar -J"]}),
        ("zip -r", f"Zip `{run}/` (recursively) into `{arc}.zip`.", f"zip -r {arc}.zip {run}/"),
        ("unzip -l", f"List the contents of `{arc}.zip`.", f"unzip -l {arc}.zip", {"setup": f"zip -qr {arc}.zip {run}/"}),
        ("unzip -d", f"Unzip `{arc}.zip` into the directory `dest/`.", f"unzip {arc}.zip -d dest",
         {"setup": f"zip -qr {arc}.zip {run}/", "post": "find dest | sort"}),
        ("gzip", f"Compress `{big}` with gzip (it gets replaced by `{big}.gz`).", f"gzip {big}", {"post": "ls"}),
        ("gunzip", f"Decompress `{big}.gz`.", f"gunzip {big}.gz", {"setup": f"gzip {big}", "post": f"ls; head -3 {big}"}),
        ("zcat", f"Print the first 3 lines of `{big}.gz` without decompressing it to disk.", f"zcat {big}.gz | head -3",
         {"setup": f"gzip {big}"}),
        ("zgrep", f"Search the compressed `{big}.gz` for lines containing `{word}`.", f"zgrep {word} {big}.gz",
         {"setup": f"gzip {big}"}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def disk_usage(rng):
    spec = {}
    dirs = rng.sample(["data", "models", "logs", "notebooks", "src", "cache", "results"], 4)
    for d in dirs:
        for i in range(rng.randint(1, 4)):
            spec[f"{d}/f{i}.bin"] = "x" * rng.choice([100, 3000, 9000, 20000, 50000]) + "\n"
    setup = "#: cwd ~/project\n" + files(spec)
    d = rng.choice(dirs)
    items = [
        ("du -s", f"How much space does `{d}/` use? Print its total in human-readable form.", f"du -sh {d}/",
         {"require": ["du -s", "du -h"]}),
        ("du", "Print the human-readable size of every item here, smallest first.", "du -sh * | sort -h",
         {"require": ["du", "sort -h"]}),
        ("du -d", "Print the human-readable size of each directory one level below the current one (and the total).",
         "du -h -d 1 .", {"require": ["du -d", "du -h"]}),
        ("df -h", "Show the free space on every mounted filesystem, human-readable.", "df -h"),
        ("df -h", "Show the free space of the filesystem that holds the current directory.", "df -h ."),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def read_modes(rng):
    picks = rng.sample(MODES, 3)
    names = rng.sample(["a.sh", "b.txt", "c.key", "d.csv", "e.py", "f.md"], 3)
    setup = files({n: "x\n" for n in names}) + "\n" + "\n".join(f"chmod {o} {n}" for (_, o), n in zip(picks, names))
    kind = rng.choice(["octal", "who-exec", "copy-mode"])
    if kind == "octal":
        return P("Print each file's octal permissions and name, like `644 notes.md`, for every file here.", setup,
                 "stat -c '%a %n' *", require=["stat"])
    if kind == "who-exec":
        return P("Which files here can their owner execute? Print their names, one per line.", setup,
                 "for f in *; do [[ -x $f ]] && echo \"$f\"; done", require=["[[ -x|test|find"])
    a, b = names[0], names[1]
    return P(f"Give `{b}` exactly the same permissions as `{a}` (look them up, don't hard-code).", setup,
             f"chmod \"$(stat -c '%a' {a})\" {b}", post=f"stat -c '%a %n' {a} {b}", require=["chmod"])


# ================================================================= medium

@template(M, "medium")
def largest_dirs(rng):
    spec = {}
    dirs = rng.sample(["datasets", "models", "logs", "notebooks", "code", "cache", "results", "venvs", "papers"], 6)
    for d in dirs:
        for i in range(rng.randint(1, 5)):
            spec[f"{d}/part{i}.bin"] = "x" * rng.choice([500, 4000, 12000, 30000, 70000, 150000]) + "\n"
    setup = "#: cwd ~\n" + files(spec)
    k = rng.randint(2, 4)
    kind = rng.choice(["largest", "smallest"])
    if kind == "largest":
        return P(f"Print the {k} largest directories in your home folder (human-readable sizes, largest last).",
                 setup, f"du -sh ~/*/ | sort -h | tail -{k}", require=["du", "sort -h"])
    return P(f"Print the {k} smallest directories in your home folder with their sizes in KiB (`du -sk`), "
             f"smallest first.", setup, f"du -sk ~/*/ | sort -n | head -{k}", require=["du", "sort"])


@template(M, "medium")
def backup_round_trip(rng):
    run, spec = _results(rng, rng.randint(3, 5))
    date = f"2026-0{rng.randint(1, 9)}-{rng.randint(10, 28)}"
    setup = f"#: cwd ~/work\n#: date {date}\n" + files(spec)
    kind = rng.choice(["dated", "count", "exclude"])
    if kind == "dated":
        return P(f"Archive `{run}/` as a gzipped tarball named after today's date, like `{run}_YYYY-MM-DD.tar.gz` "
                 f"(use `date +%F`), then list the archive.", setup,
                 f"tar -czf {run}_$(date +%F).tar.gz {run}/ && tar -tzf {run}_$(date +%F).tar.gz",
                 require=["tar", "date", "$("])
    if kind == "count":
        return P(f"Archive `{run}/` into `{run}.tar.gz` and print how many regular files the archive contains.",
                 setup, f"tar -czf {run}.tar.gz {run}/\ntar -tzf {run}.tar.gz | grep -v '/$' | wc -l",
                 require=["tar", "wc"])
    return P(f"Archive `{run}/` into `{run}.tar.gz` but leave out every `.log` file, then list the archive.", setup,
             f"tar -czf {run}.tar.gz --exclude='*.log' {run}/ && tar -tzf {run}.tar.gz", require=["tar"])


# ================================================================= hard

@template(M, "hard")
def fix_permissions(rng):
    spec = {}
    d = rng.choice(["site", "public", "webroot", "share"])
    for i in range(rng.randint(3, 6)):
        sub = rng.choice(["", "css/", "img/", "js/"])
        spec[f"{d}/{sub}{rng.choice(WORDS)}_{i}.{rng.choice(['html', 'css', 'png', 'js'])}"] = "x\n"
    spec[f"{d}/deploy.sh"] = "#!/bin/sh\necho ok\n"
    setup = files(spec) + f"\nchmod -R 600 {d}\nchmod 700 {d}"
    return P(f"Everything under `{d}/` got permissions 600 by mistake. Fix it: directories `755`, regular files "
             f"`644`, except `deploy.sh`, which must be `755`. (A directory without `x` can't be entered, so fix each "
             f"directory as `find` reaches it: `-exec ... \\;` rather than the batched `+`.)", setup,
             f"find {d} -type d -exec chmod 755 {{}} \\;\nfind {d} -type f -exec chmod 644 {{}} +\nchmod 755 {d}/deploy.sh",
             post=f"find {d} -exec stat -c '%a %n' {{}} + | sort -k2", require=["chmod"])


@template(M, "hard")
def compressed_logs(rng):
    days = rng.sample(range(1, 28), rng.randint(3, 5))
    spec = {}
    word = rng.choice(["ERROR", "WARNING", "Traceback", "CUDA out of memory"])
    for d in sorted(days):
        lines = [f"2026-10-{d:02d} {rng.choice(['INFO', 'DEBUG', 'ERROR', 'WARNING'])} {rng.choice(WORDS)}"
                 for _ in range(rng.randint(5, 12))]
        if rng.random() < 0.6:
            lines.append(f"2026-10-{d:02d} {word} in {rng.choice(WORDS)}")
        spec[f"logs/app-2026-10-{d:02d}.log"] = "\n".join(lines) + "\n"
    setup = files(spec) + "\ngzip logs/*.log"
    kind = rng.choice(["count", "files"])
    if kind == "count":
        return P(f"The logs in `logs/` are gzipped. Print how many lines across all of them contain `{word}`.",
                 setup, f"zcat logs/*.gz | grep -c {q(word)}", require=["zcat|zgrep"])
    return P(f"Which compressed logs in `logs/` mention `{word}`? Print their names, one per line.", setup,
             f"zgrep -l {q(word)} logs/*.gz", require=["zgrep|zcat"])


# ================================================================= challenge

def _tree(rng, n):
    spec = {}
    for i in range(n):
        depth = rng.randint(1, 3)
        path = "/".join(rng.choice(["data", "raw", "img", "cache", "tmp", "v1", "v2", "train", "val"]) for _ in range(depth))
        spec[f"store/{path}/f{i}.bin"] = "x" * rng.randint(1, 60) * 1000 + "\n"
    return "#: cwd ~\n" + files(spec)


@template(M, "challenge")
def space_hogs(rng):
    setup = _tree(rng, 8)
    hidden = [_tree(rng, rng.randint(30, 50)) for _ in range(2)]
    sol = ("find store -type f -printf '%s %h\\n' | awk '{s[$2] += $1} END {for (d in s) print s[d], d}' | "
           "sort -k1,1nr -k2,2 | head -3")
    return challenge("Where did the space go?", "For every directory under `store/`, add up the sizes (in bytes) of "
                     "the regular files directly inside it (not in subdirectories). Print the three directories with "
                     "the largest totals as `BYTES PATH`, largest first (ties: by path).", setup, sol, hidden,
                     "up to 50 files")


def _perm_audit(rng, n):
    spec, chm = {}, []
    for i in range(n):
        p = f"shared/{rng.choice(['', 'a/', 'b/', 'a/c/'])}{rng.choice(WORDS)}_{i}.txt"
        spec[p] = "x\n"
        chm.append(f"chmod {rng.choice(['644', '664', '666', '600', '777', '755', '640', '646'])} {p}")
    return files(spec) + "\n" + "\n".join(chm)


@template(M, "challenge")
def world_writable(rng):
    setup = _perm_audit(rng, 6)
    hidden = [_perm_audit(rng, rng.randint(25, 40)) for _ in range(2)]
    sol = ("find shared -type f -perm -o+w | sort | while read -r f; do chmod o-w \"$f\"; echo \"fixed $f\"; done\n"
           "echo \"$(find shared -type f -perm -o+w | wc -l) world-writable files left\"")
    return challenge("World-writable audit", "Some files under `shared/` are writable by everyone. For each such "
                     "regular file (in sorted path order) remove the write permission for others only and print "
                     "`fixed PATH`; finally print `N world-writable files left` (it should be 0).", setup, sol,
                     hidden, "up to 40 files", post="find shared -type f -exec stat -c '%a %n' {} + | sort -k2",
                     answer=["<stdout>", "<post>"])

"""7.02 Navigating & managing files."""

from ._common import DATASETS, EXTS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.02"
TREE = "find . -mindepth 1 | sort"


def _project(rng, n_files=5):
    """A small project directory: a few files with random sizes and modification times."""
    spec = {}
    names = rng.sample(["train.py", "model.py", "utils.py", "README.md", "notes.md", "config.yaml", "data.csv",
                        "results.json", "run.sh", "eval.py", "requirements.txt", "LICENSE"], n_files)
    for nm in names:
        spec[nm] = "".join(f"{rng.choice(WORDS)} {rng.choice(WORDS)}\n" for _ in range(rng.randint(1, 30)))
    times = rng.sample(range(1, 28), n_files)
    touch = "\n".join(f"touch -d '2026-09-{d:02d} 1{d % 10}:00' {q(nm)}" for nm, d in zip(names, times))
    return names, spec, touch


# ================================================================= lite

@template(M, "lite")
def look_around(rng):
    names, spec, touch = _project(rng, rng.randint(4, 6))
    sub = rng.choice(["data", "src", "logs", "notebooks"])
    deep = f"{sub}/{rng.choice(['raw', 'v1', 'old'])}/{rng.choice(['a', 'b', 'tmp'])}"
    hidden = rng.choice([".env", ".gitignore", ".cache"])
    spec[hidden] = "x\n"
    spec[f"{deep}/keep.txt"] = "keep\n"
    setup = "#: cwd ~/project\n" + files(spec) + "\n" + touch
    other = rng.choice(["/tmp", "/etc", "/var/log", "/usr/share"])
    items = [
        ("pwd", f"Go into `{deep}` and print the current directory.", f"cd {deep} && pwd", {"require": ["cd", "pwd"]}),
        ("ls", "List the current directory (plain `ls`).", "ls"),
        ("ls -1", "List the current directory, one name per line.", "ls -1"),
        ("ls -a", f"List everything in the current directory, including hidden files like `{hidden}`.", "ls -a"),
        ("ls -l", "Show a long listing (permissions, owner, size, date) of the current directory.", "ls -l"),
        ("ls -t", "List the current directory newest first (by modification time), one per line.", "ls -1t",
         {"require": ["ls -t"]}),
        ("ls -S", "List the current directory largest file first, one per line.", "ls -1S", {"require": ["ls -S"]}),
        ("ls -d", "List only the directories here (names ending in `/`), using a glob and `ls -d`.", "ls -d */"),
        ("tree -L", f"Show the directory tree two levels deep.", "tree -L 2"),
        ("cd -", f"Go to `{other}`, then jump back to the previous directory with `cd -`.", f"cd {other}\ncd -"),
        ("cd ..", f"Go into `{deep}`, then up two levels with a single `cd`, and print where you are.",
         f"cd {deep}\ncd ../..\npwd", {"require": ["cd", "pwd"]}),
        ("pushd", f"Remember where you are and jump to `{other}` with `pushd`, then come back with `popd`.",
         f"pushd {other}\npopd", {"require": ["pushd", "popd"]}),
        ("ls -h", f"Long-list `{names[0]}` with a human-readable size.", f"ls -lh {names[0]}", {"require": ["ls -l", "ls -h"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def make_and_remove(rng):
    names, spec, touch = _project(rng, 3)
    d = rng.choice(["data", "results", "runs", "checkpoints", "logs"])
    splits = rng.sample(["train", "val", "test", "dev", "holdout"], 3)
    exp = rng.choice(["exp", "experiments", "runs"])
    new = rng.choice(["notes.md", "TODO.md", "todo.txt", "ideas.txt"])
    empty = rng.choice(["empty_dir", "old", "scratch"])
    build = rng.choice(["build", "dist", "__pycache__", "tmp"])
    spec[f"{build}/a.o"] = "x\n"
    spec[f"{build}/sub/b.o"] = "y\n"
    src, dst = names[0], rng.choice(["copy_" + names[0], names[0] + ".orig", "backup_" + names[0]])
    setup = "#: cwd ~/work\n" + files(spec, mkdirs=[empty]) + "\n" + touch
    items = [
        ("mkdir", f"Create a directory called `{d}`.", f"mkdir {d}"),
        ("mkdir -p", f"Create `{exp}/{splits[0]}`, `{exp}/{splits[1]}` and `{exp}/{splits[2]}` in one command "
                     f"(brace expansion).", f"mkdir -p {exp}/{{{','.join(splits)}}}", {"require": ["mkdir -p", "{,}"]}),
        ("touch", f"Create an empty file `{new}`.", f"touch {new}"),
        ("rmdir", f"Remove the empty directory `{empty}` (with the command that only removes empty ones).", f"rmdir {empty}"),
        ("cp", f"Copy `{src}` to `{dst}`.", f"cp {src} {dst}"),
        ("cp -r", f"Copy the whole `{build}` directory to `{build}_backup`.", f"cp -r {build} {build}_backup"),
        ("mv", f"Rename `{names[1]}` to `old_{names[1]}`.", f"mv {names[1]} old_{names[1]}"),
        ("rm", f"Delete `{names[2]}`.", f"rm {names[2]}"),
        ("rm -r", f"Delete the `{build}` directory and everything in it.", f"rm -r {build}", {"require": ["rm -r|rm -rf"]}),
        ("rm -rf", f"Force-delete `{build}` recursively, without any questions.", f"rm -rf {build}"),
        ("ln -s", f"Make a symbolic link called `latest` that points to `{names[0]}`.", f"ln -s {names[0]} latest"),
    ]
    return catalog(rng, items, setup, post=TREE)


# ================================================================= easy

@template(M, "easy")
def path_tools(rng):
    run = rng.randint(1, 99)
    model = rng.choice(MODELS)
    ext = rng.choice(["pt", "ckpt", "safetensors", "onnx"])
    path = f"/data/runs/exp{run}/{model}_best.{ext}"
    names, spec, touch = _project(rng, 3)
    target = names[0]
    spec["links/.keep"] = None
    setup = "#: cwd ~/work\n" + files(spec) + "\n" + touch + f"\nln -s ../{target} links/current"
    items = [
        ("basename", f"Print just the file name of `{path}`.", f"basename {path}"),
        ("basename", f"Print the file name of `{path}` without the `.{ext}` extension.", f"basename {path} .{ext}"),
        ("dirname", f"Print the directory part of `{path}`.", f"dirname {path}"),
        ("readlink", "Print where the symbolic link `links/current` points (the link's target as stored).",
         "readlink links/current"),
        ("realpath", "Print the absolute, symlink-free path of `links/current`.", "realpath links/current"),
        ("realpath", f"Print the absolute path of `{names[1]}`.", f"realpath {names[1]}"),
        ("file", f"Ask `file` what kind of file `{names[1]}` is.", f"file {names[1]}"),
        ("file", "Ask `file` what `links/current` is (without following it).", "file links/current"),
        ("stat", f"Print the full `stat` information of `{names[2]}`.", f"stat {names[2]}"),
        ("wc -l", f"Print the number of lines in `{names[0]}`, followed by its name (as `wc` prints it).",
         f"wc -l {names[0]}"),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def careful_copies(rng):
    a, b = rng.sample(["a.txt", "config.yaml", "params.json", "notes.md", "train.csv", "model.cfg"], 2)
    old = "".join(f"{w}\n" for w in rng.sample(WORDS, 3))
    new = "".join(f"{w}\n" for w in rng.sample(WORDS, 3))
    setup = "#: cwd ~/work\n" + files({a: new, b: old}) + f"\ntouch -d '2026-0{rng.randint(1, 9)}-1{rng.randint(0, 9)} 08:30' {a}"
    kind = rng.choice(["i-no", "i-yes", "n", "p", "bak", "rm-i"])
    if kind == "i-no":
        return P(f"Copy `{a}` over `{b}`, but make `cp` ask first; then answer `n` when it asks.", setup + "\n#: type n",
                 f"cp -i {a} {b}", post=f"cat {b}", require=["cp -i"])
    if kind == "i-yes":
        return P(f"Copy `{a}` over `{b}`, but make `cp` ask first; then answer `y` when it asks.", setup + "\n#: type y",
                 f"cp -i {a} {b}", post=f"cat {b}", require=["cp -i"])
    if kind == "n":
        return P(f"Move `{a}` to `{b}`, but never overwrite an existing file.", setup, f"mv -n {a} {b}",
                 post=f"ls; cat {b}", require=["mv -n"])
    if kind == "p":
        return P(f"Copy `{a}` to `{a}.copy` keeping its modification time (and mode).", setup, f"cp -p {a} {a}.copy",
                 post=f"ls -l --time-style=+%F {a}.copy", require=["cp -p"])
    if kind == "bak":
        return P(f"Make a backup copy `{b}.bak` of `{b}` using brace expansion (don't type the name twice).", setup,
                 f"cp {b}{{,.bak}}", post="ls", require=["cp", "{,}"])
    return P(f"Delete `{a}` with `rm`, asking for confirmation first; answer `y`.", setup + "\n#: type y",
             f"rm -i {a}", post="ls", require=["rm -i"])


# ================================================================= medium

@template(M, "medium")
def organize(rng):
    n = rng.randint(3, 6)
    csvs = [f"{rng.choice(DATASETS)}_{i}.csv" for i in range(n)]
    others = rng.sample(["train.py", "README.md", "plot.png", "notes.txt", "run.sh"], 3)
    spec = {c: "a,b\n1,2\n" for c in csvs} | {o: "x\n" for o in others}
    setup = "#: cwd ~/inbox\n" + files(spec)
    dest = rng.choice(["data", "csv", "tables", "raw"])
    kind = rng.choice(["move", "copy", "count"])
    if kind == "move":
        return P(f"Create a `{dest}` directory and move every `.csv` file into it, in one line.", setup,
                 f"mkdir -p {dest} && mv *.csv {dest}/", post=TREE, require=["mkdir", "mv", "glob"])
    if kind == "copy":
        return P(f"Copy every `.csv` file into a new directory `{dest}` (keep the originals), then list `{dest}` "
                 f"one name per line.", setup,
                 f"mkdir {dest}\ncp *.csv {dest}/\nls -1 {dest}", require=["cp", "glob"])
    return P("How many `.csv` files are there here? Print just the number.", setup, "ls *.csv | wc -l",
             require=["ls", "wc"])


@template(M, "medium")
def biggest_newest(rng):
    names, spec, touch = _project(rng, rng.randint(5, 7))
    setup = "#: cwd ~/project\n" + files(spec) + "\n" + touch
    k = rng.randint(2, 3)
    kind = rng.choice(["largest", "newest", "oldest", "smallest"])
    if kind == "largest":
        return P(f"Print the names of the {k} largest files here, largest first.", setup, f"ls -S | head -{k}",
                 require=["ls -S", "head"])
    if kind == "newest":
        return P("Print the name of the most recently modified file here.", setup, "ls -t | head -1",
                 require=["ls -t", "head"])
    if kind == "oldest":
        return P("Print the name of the least recently modified file here.", setup, "ls -t | tail -1",
                 require=["ls -t", "tail"])
    return P(f"Print the names of the {k} smallest files, smallest first.", setup, f"ls -Sr | head -{k}",
             require=["ls -S", "head"])


# ================================================================= hard

@template(M, "hard")
def skeleton(rng):
    proj = rng.choice(["vision-lab", "nlp-demo", "speech-asr", "rl-agent", "tabular-ml"])
    data = rng.sample(["raw", "interim", "processed", "external"], 2)
    code = rng.sample(["src", "lib", "app"], 1)[0]
    extra = rng.sample(["notebooks", "tests", "configs", "scripts", "docs"], 2)
    pkg = proj.split("-")[0]
    spec = f"`{proj}/` containing `data/{data[0]}`, `data/{data[1]}`, `{code}/{pkg}/`, `{extra[0]}/` and `{extra[1]}/`, " \
           f"plus empty files `{code}/{pkg}/__init__.py` and `README.md`"
    sol = (f"mkdir -p {proj}/{{data/{{{data[0]},{data[1]}}},{code}/{pkg},{extra[0]},{extra[1]}}}\n"
           f"touch {proj}/{code}/{pkg}/__init__.py {proj}/README.md")
    return P(f"Create this project skeleton under the current directory: {spec}. Use `mkdir -p` with brace "
             f"expansion (one `mkdir`).", "#: cwd ~/code", sol, post=f"tree {proj}", require=["mkdir -p", "{,}", "touch"])


@template(M, "hard")
def link_dataset(rng):
    ds = rng.choice(DATASETS)
    v = rng.randint(1, 4)
    shards = rng.randint(2, 4)
    spec = {f"/home/ada/datasets/{ds}/v{v}/shard_{i}.bin": "x" * rng.randint(5, 40) + "\n" for i in range(shards)}
    setup = files(spec) + "\n#: cwd ~/project"
    kind = rng.choice(["link", "resolve", "relink"])
    if kind == "link":
        return P(f"The dataset lives in `~/datasets/{ds}/v{v}`. Link it into the current project as `data` "
                 f"(a symbolic link), then list the shards through the link.", setup,
                 f"ln -s ~/datasets/{ds}/v{v} data\nls data", require=["ln -s"])
    if kind == "resolve":
        return P(f"Link `~/datasets/{ds}/v{v}` into the project as `data`, then print the real (absolute, symlink-free) "
                 f"path of `data/shard_0.bin`.", setup, f"ln -s ~/datasets/{ds}/v{v} data\nrealpath data/shard_0.bin",
                 require=["ln -s", "realpath|readlink"])
    other = v + 1
    spec2 = {f"/home/ada/datasets/{ds}/v{other}/shard_0.bin": "new\n"}
    return P(f"`data` points at `~/datasets/{ds}/v{v}`. A new version appeared: make `data` point at "
             f"`~/datasets/{ds}/v{other}` instead (replace the link in one command), then print the link's target.",
             setup + "\n" + files(spec2) + f"\nln -s ~/datasets/{ds}/v{v} data",
             f"ln -sfn ~/datasets/{ds}/v{other} data\nreadlink data", require=["ln -s", "readlink"])


# ================================================================= trees (built in the terminal, watched in the Files panel)

TREE_ROOTS = ("ml-project", "vision-lab", "nlp-demo", "speech-asr", "rl-agent", "kaggle", "thesis", "capstone")
SUBDIRS = {"data": ["raw", "processed", "interim", "external"], "src": ["utils", "models", "datasets"],
           "models": ["checkpoints", "exported"], "notebooks": ["exploration", "reports"], "configs": ["experiments"],
           "results": ["figures", "tables"], "logs": ["train", "eval"], "tests": ["unit", "integration"],
           "docs": ["images"], "scripts": ["slurm"]}
DIR_FILES = {"src": ["train.py", "model.py", "dataset.py", "__init__.py"], "configs": ["base.yaml", "small.yaml"],
             "notebooks": ["eda.ipynb", "baseline.ipynb"], "docs": ["index.md", "setup.md"],
             "tests": ["test_model.py", "test_data.py"], "scripts": ["run.sh", "sweep.sh"], "results": ["summary.csv"],
             "logs": ["train.log"], "models": ["best.pt"], "data": ["README.md"]}
TOP_FILES = ("README.md", "requirements.txt", "Makefile", "setup.py", "LICENSE", "pyproject.toml")


def draw_tree(root: str, dirs, files_) -> str:
    """A `tree -F` style drawing of `root`: dirs and files are paths relative to it."""
    kids: dict = {}
    for d in dirs:
        parts = d.split("/")
        for k in range(len(parts)):
            kids.setdefault("/".join(parts[:k]), set()).add(parts[k] + "/")
    for f in files_:
        parts = f.split("/")
        for k in range(len(parts) - 1):
            kids.setdefault("/".join(parts[:k]), set()).add(parts[k] + "/")
        kids.setdefault("/".join(parts[:-1]), set()).add(parts[-1])
    lines = [root.rstrip("/") + "/"]

    def walk(parent, prefix):
        items = sorted(kids.get(parent, ()), key=lambda x: x.rstrip("/").lower())
        for i, name in enumerate(items):
            last = i == len(items) - 1
            lines.append(prefix + ("└── " if last else "├── ") + name)
            if name.endswith("/"):
                walk(f"{parent}/{name[:-1]}" if parent else name[:-1], prefix + ("    " if last else "│   "))
    walk("", "")
    return "\n".join(lines)


def tree_check(root: str) -> str:
    """The post check of a tree task: every directory, then every file with its checksum (so contents count)."""
    return (f"cd {root} && find . -mindepth 1 -type d | sort; echo ---; "
            "find . -type f -exec md5sum {} + | sort -k2")


def _leaf_dirs(dirs):
    return sorted(d for d in dirs if not any(o.startswith(d + "/") for o in dirs))


def _random_tree(rng, n_top, deep, n_files):
    tops = rng.sample(sorted(SUBDIRS), n_top)
    dirs = set(tops)
    for t in tops:
        if deep and rng.random() < 0.7:
            for sub in rng.sample(SUBDIRS[t], rng.randint(1, min(2, len(SUBDIRS[t])))):
                dirs.add(f"{t}/{sub}")
    pool = [f"{t}/{f}" for t in tops for f in DIR_FILES.get(t, [])] + list(TOP_FILES)
    files_ = sorted(rng.sample(pool, min(n_files, len(pool))))
    return sorted(dirs), files_


@template(M, "lite")
def build_tree(rng):
    root = rng.choice(TREE_ROOTS)
    dirs, files_ = _random_tree(rng, rng.randint(2, 3), False, rng.randint(1, 3))
    sol = f"mkdir {root}\ncd {root}\nmkdir {' '.join(_leaf_dirs(dirs))}\ntouch {' '.join(files_)}"
    return P(f"Starting in your home directory (`~`), create this tree. The files are empty.\n\n"
             f"```\n{draw_tree(root, dirs, files_)}\n```\n\nThe **Files** panel shows what you have made so far.",
             "", sol, post=tree_check(f"~/{root}"), require=["mkdir", "touch"])


@template(M, "easy")
def build_deep_tree(rng):
    root = rng.choice(TREE_ROOTS)
    dirs, files_ = _random_tree(rng, rng.randint(3, 4), True, rng.randint(3, 5))
    paths = " ".join(f"{root}/{d}" for d in _leaf_dirs(dirs))
    sol = f"mkdir -p {paths}\ntouch {' '.join(f'{root}/{f}' for f in files_)}"
    return P(f"Starting in `~`, create this tree (the files are empty). `mkdir -p` makes a directory and its "
             f"parents at once.\n\n```\n{draw_tree(root, dirs, files_)}\n```",
             "", sol, post=tree_check(f"~/{root}"), require=["mkdir -p", "touch"])


MAZE_NAMES = ("red", "blue", "green", "amber", "north", "south", "east", "west", "alpha", "beta", "gamma", "delta",
              "cedar", "maple", "oak", "pine")


@template(M, "easy")
def find_your_way(rng):
    names = rng.sample(MAZE_NAMES, 9)
    level1, rest = names[:3], names[3:]
    spec, leaves = {}, []
    for a in level1:
        for b in rng.sample(rest, 2):
            leaves.append(f"maze/{a}/{b}")
            spec[f"/home/ada/maze/{a}/{b}/{rng.choice(['map.txt', 'todo.txt', 'empty.txt'])}"] = f"not here ({b})\n"
    goal = rng.choice(leaves)
    secret = f"{rng.choice(WORDS)}-{rng.randint(100, 999)}"
    spec[f"/home/ada/{goal}/note.txt"] = f"the password is {secret}\n"
    start = rng.choice(["maze", f"maze/{rng.choice(level1)}"])
    setup = files(spec) + f"\n#: cwd ~/{start}"
    rel = goal[len(start) + 1:] if goal.startswith(start + "/") else "../" + goal[len("maze/"):]
    return P(f"You're in `~/{start}`. Somewhere below `~/maze` there is a `note.txt` (look for it with `ls`, or in "
             f"the Files panel). Go to its directory with `cd`, then print it with `cat note.txt`.",
             setup, f"cd {rel}\ncat note.txt", post="pwd", answer=["<stdout>", "<post>"], require=["cd", "cat"])


def _transform(rng, kinds):
    """A start tree, operations of the given kinds, and the tree they lead to. -> (dirs0, files0, cmds, dirs, files)"""
    dirs, files_ = _random_tree(rng, 3, True, 5)
    dirs, files_ = set(dirs), {f: "".join(f"{rng.choice(WORDS)}\n" for _ in range(rng.randint(1, 4))) for f in files_}
    if "rmdir" in kinds:
        dirs.add(rng.choice(["old", "tmp", "scratch", "unused"]))
    if "rm -r" in kinds:
        b = rng.choice(["build", "dist", "cache", "wandb"])
        dirs.add(b)
        for k in range(rng.randint(2, 3)):
            files_[f"{b}/part{k}.bin"] = f"{b} {k}\n"
    dirs0, files0 = sorted(dirs), dict(files_)
    cmds = []
    protected: set[str] = set()      # paths the next operations must not touch again
    doomed = ("build", "dist", "cache", "wandb")   # the directory `rm -r` deletes
    for kind in kinds:
        movable = [f for f in sorted(files_) if f not in protected and f.split("/")[0] not in doomed]
        places = [d for d in sorted(dirs) if d.split("/")[0] not in doomed]
        if kind in ("rm", "mv", "cp") and not movable:
            continue
        if kind == "mkdir":
            parent = rng.choice(["", rng.choice(places)])
            name = rng.choice(["archive", "backup", "figures", "exports", "drafts", "v2"])
            new = f"{parent}/{name}" if parent else name
            if new in dirs:
                continue
            dirs.add(new)
            protected.add(new)
            cmds.append(f"mkdir {new}")
        elif kind == "touch":
            d = rng.choice(places)
            new = f"{d}/{rng.choice(['notes.md', 'TODO.txt', 'ideas.md', 'draft.txt'])}"
            if new in files_:
                continue
            files_[new] = ""
            protected.add(new)
            cmds.append(f"touch {new}")
        elif kind == "rm":
            f = rng.choice(movable)
            del files_[f]
            cmds.append(f"rm {f}")
        elif kind == "rmdir":
            empty = [d for d in sorted(dirs) if not any(x.startswith(d + "/") for x in list(dirs) + list(files_))
                     and d not in protected]
            if not empty:
                continue
            d = rng.choice(empty)
            dirs.discard(d)
            cmds.append(f"rmdir {d}")
        elif kind == "mv":
            f = rng.choice(movable)
            base = f.rsplit("/", 1)[-1]
            if rng.random() < 0.5:
                new = (f.rsplit("/", 1)[0] + "/" if "/" in f else "") + "old_" + base
                cmds.append(f"mv {f} {new}")
            else:
                d = rng.choice([x for x in places if not f.startswith(x + "/")] or [""])
                new = f"{d}/{base}" if d else base
                if new in files_:
                    continue
                cmds.append(f"mv {f} {d}/" if d else f"mv {f} .")
            files_[new] = files_.pop(f)
            protected.add(new)
        elif kind == "cp":
            f = rng.choice(movable)
            new = f + ".bak"
            files_[new] = files_[f]
            protected |= {f, new}
            cmds.append(f"cp {f} {new}")
        elif kind == "cp -r":
            tops = [d for d in places if "/" not in d and d not in protected]
            if not tops:
                continue
            src = rng.choice(tops)
            dst = src + "_backup"
            for d in sorted(dirs):
                if d == src or d.startswith(src + "/"):
                    dirs.add(dst + d[len(src):])
                    protected.add(dst + d[len(src):])
            for f in sorted(files_):
                if f.startswith(src + "/"):
                    files_[dst + f[len(src):]] = files_[f]
                    protected |= {f, dst + f[len(src):]}
            cmds.append(f"cp -r {src} {dst}")
        elif kind == "rm -r":
            b = next(d for d in dirs0 if d in ("build", "dist", "cache", "wandb"))
            dirs = {d for d in dirs if d != b and not d.startswith(b + "/")}
            files_ = {f: v for f, v in files_.items() if not f.startswith(b + "/")}
            cmds.append(f"rm -r {b}")
    return dirs0, files0, cmds, sorted(dirs), files_


def _transform_problem(rng, kinds, root):
    dirs0, files0, cmds, dirs, files_ = _transform(rng, kinds)
    setup = f"#: cwd ~/{root}\n" + files(files0, mkdirs=dirs0)
    used = sorted({c.split()[0] + (" -r" if " -r " in c else "") for c in cmds})
    need = ", ".join(f"`{u}`" for u in used)
    return P(f"Your project `~/{root}` looks like this (it's also in the Files panel):\n\n"
             f"```\n{draw_tree(root, dirs0, files0)}\n```\n\nTurn it into this:\n\n"
             f"```\n{draw_tree(root, dirs, files_)}\n```\n\nUse {need} (each at least once, in any order). Moved "
             f"and copied files keep their contents.", setup, "\n".join(cmds), post=tree_check(f"~/{root}"),
             require=[u if u != "rm -r" else "rm -r|rm -rf" for u in used])


@template(M, "medium")
def reshape_tree(rng):
    kinds = rng.sample(["mkdir", "touch", "rm", "rmdir", "mv"], rng.randint(3, 4))
    rng.shuffle(kinds)
    return _transform_problem(rng, kinds, rng.choice(TREE_ROOTS))


@template(M, "hard")
def reorganize_tree(rng):
    kinds = rng.sample(["mkdir", "touch", "rm", "mv"], 2) + rng.sample(["cp", "cp -r", "rm -r", "rmdir"], 3)
    rng.shuffle(kinds)
    return _transform_problem(rng, kinds, rng.choice(TREE_ROOTS))


def _messy(rng, n):
    """A repo with junk: *.tmp files, __pycache__ directories (with .pyc files) and empty directories."""
    dirs, files_ = _random_tree(rng, 4, True, n)
    spec = {f: f"{rng.choice(WORDS)}\n" for f in files_}
    empties = []
    for d in dirs:
        r = rng.random()
        if r < 0.35:
            spec[f"{d}/{rng.choice(WORDS)}.tmp"] = "junk\n"
        elif r < 0.6:
            spec[f"{d}/__pycache__/{rng.choice(WORDS)}.cpython-312.pyc"] = "bytecode\n"
        elif r < 0.75:
            empties.append(f"{d}/{rng.choice(['old', 'empty', 'unused'])}")
    spec[f"{rng.choice(TOP_FILES)}.tmp"] = "junk\n"
    return files(spec, mkdirs=sorted(dirs) + empties)


@template(M, "hard")
def clean_up(rng):
    root = rng.choice(TREE_ROOTS)
    head = f"#: cwd ~/{root}\n"
    setup = head + _messy(rng, 5)
    hidden = [head + _messy(rng, rng.randint(6, 9)) for _ in range(2)]
    kind = rng.choice(["junk", "empty"])
    if kind == "junk":
        return P(f"Clean up `~/{root}`: delete every `*.tmp` file and every `__pycache__` directory, at any depth, "
                 f"and keep everything else. Your commands also run on hidden tests with different files, so don't "
                 f"just list today's paths.", setup,
                 "find . -type f -name '*.tmp' -delete\nfind . -type d -name __pycache__ -prune -exec rm -rf {} +",
                 post=tree_check(f"~/{root}"), tests=hidden, require=["find|**"])
    return P(f"Delete every empty directory under `~/{root}` at any depth, including directories that are left "
             f"empty once their empty subdirectories are gone. Your commands also run on hidden tests with different files.", setup,
             "find . -mindepth 1 -type d -empty -delete", post=tree_check(f"~/{root}"), tests=hidden,
             require=["find"])


# ================================================================= challenge

def _inbox(rng, n):
    exts = list(EXTS) + ["png", "jpg", ""]
    names = set()
    while len(names) < n:
        e = rng.choice(exts)
        base = f"{rng.choice(WORDS)}_{rng.randint(1, 99)}"
        names.add(base + (f".{e}" if e else ""))
    return files({f"inbox/{x}": f"{x}\n" for x in sorted(names)})


@template(M, "challenge")
def sort_by_extension(rng):
    setup = "#: cwd ~/desk\n" + _inbox(rng, 6)
    hidden = ["#: cwd ~/desk\n" + _inbox(rng, rng.randint(25, 40)) for _ in range(2)]
    sol = ('for f in inbox/*; do\n  name=${f##*/}\n  if [[ $name == *.* ]]; then ext=${name##*.}; else ext=other; fi\n'
           '  mkdir -p "sorted/$ext"\n  mv "$f" "sorted/$ext/"\ndone')
    return challenge("Tidy the inbox", "Move every file from `inbox/` into `sorted/EXT/`, where EXT is the file's "
                     "extension (`sorted/other/` for files without one), creating folders as needed.", setup, sol,
                     hidden, "up to 40 files", post="find sorted | sort")


def _runs(rng, n_runs):
    spec, times = {}, []
    for r in range(n_runs):
        run = f"runs/{rng.choice(MODELS)}-{r}"
        for s in rng.sample(range(100, 5000, 100), rng.randint(2, 5)):
            p = f"{run}/ckpt_{s:05d}.pt"
            spec[p] = "w\n"
            times.append(f"touch -d '2026-10-{rng.randint(1, 19):02d} {rng.randint(10, 23)}:{rng.randint(10, 59)}' {p}")
    return "#: cwd ~/lab\n" + files(spec) + "\n" + "\n".join(times)


@template(M, "challenge")
def latest_links(rng):
    setup = _runs(rng, 2)
    hidden = [_runs(rng, rng.randint(5, 8)) for _ in range(2)]
    sol = ('for d in runs/*/; do\n  newest=$(ls -t "$d" | grep "^ckpt_" | head -1)\n  ln -sfn "$newest" "${d}latest"\n'
           'done\nfor l in runs/*/latest; do echo "$l -> $(readlink "$l")"; done')
    return challenge("Point at the newest checkpoint", "Each run directory under `runs/` holds checkpoints "
                     "`ckpt_*.pt`. In every run directory create a symbolic link `latest` pointing at its most recently "
                     "modified checkpoint (link target: just the file name), then print `runs/RUN/latest -> TARGET` "
                     "for each run, in alphabetical order.", setup, sol, hidden, "up to 8 runs, 5 checkpoints each",
                     require=["ln -s"])

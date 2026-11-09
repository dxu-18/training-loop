"""8.02 Creating repositories & .gitignore."""

from ._common import GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template

M = "8.02"
LIGHT = ["minilibs/requests-mini", "minilibs/flask-mini", "mini-ml/nanogpt-mini", "toolsmiths/hugo-mini",
         "toolsmiths/ripgrep-mini", "mira-chen/dotfiles", "git-gym/calculator", "open-learning/cs-notes",
         "minilibs/leftpad", "tomas-ruiz/leetcode-solutions"]


def _junk(rng):
    """Project files plus the things that never belong in git."""
    fs = ml_files(rng, 3)
    fs.update({"venv/bin/python": "#!/bin/sh\n", "venv/pyvenv.cfg": "home = /usr/bin\n",
               "__pycache__/model.cpython-312.pyc": "\x00compiled\n", ".env": "WANDB_API_KEY=fake-0000\n",
               "checkpoints/epoch_3.pt": "weights\n", "notes.md": "ideas\n"})
    return fs


def _files(fs):
    return "\n".join(heredoc(p, t) for p, t in fs.items())


# ================================================================= lite

@template(M, "lite")
def init_repo(rng):
    d = rng.choice(["my-project", "curriculum", "experiments", "scratch", "ml-notes"])
    b = rng.choice(["trunk", "develop", "dev"])
    items = [
        ("git init", "Turn the current directory into a git repository.", "git init", {"check": (["head"], ".")}),
        ("git init", f"Create a new repository in a new directory `{d}` with one command.", f"git init {d}",
         {"check": (["head"], d)}),
        ("git init", f"Create a repository here whose first branch is called `{b}` (not the default).",
         f"git init -b {b}", {"check": (["head"], ".")}),
        ("git init --bare", f"Create a *bare* repository `{d}.git` (no working tree, as on a server).",
         f"git init --bare {d}.git", {"post": f"git -C {d}.git rev-parse --is-bare-repository"}),
    ]
    s = setup(GIT, "#: cwd ~/code", _files(ml_files(rng, 2)))
    return catalog(rng, items, s, answer=["<post>"])


@template(M, "lite")
def clone_repo(rng):
    full = rng.choice(LIGHT)
    owner, name = full.split("/")
    d = rng.choice(["work", "src", "copy", "mine"])
    items = [
        ("git clone", f"Clone `{full}` from GitHub over SSH into the current directory.",
         f"git clone git@github.com:{full}.git", {"check": (["head", "branches", "remotes"], name)}),
        ("git clone", f"Clone `{full}` over HTTPS (`https://github.com/{full}.git`).",
         f"git clone https://github.com/{full}.git", {"check": (["head", "branches", "remotes"], name)}),
        ("git clone", f"Clone `{full}` over SSH into a directory called `{d}` (not `{name}`).",
         f"git clone git@github.com:{full}.git {d}", {"check": (["head", "remotes"], d)}),
        ("git clone --depth", f"Make a shallow clone of `{full}` over SSH: only the latest commit.",
         f"git clone --depth 1 git@github.com:{full}.git",
         {"post": f"cd {name} && git rev-list --count HEAD && git rev-parse --is-shallow-repository"}),
    ]
    return catalog(rng, items, setup(GIT, GH), answer=["<post>"])


# ================================================================= easy

@template(M, "easy")
def write_gitignore(rng):
    fs = _junk(rng)
    s = setup(GIT, "#: cwd ~/project", "git init -q", _files(fs))
    keep = sorted(p for p in fs if not p.startswith(("venv/", "__pycache__/", "checkpoints/", ".env")))
    return P("This new repository has a virtual environment, Python's bytecode cache, a `.env` with secrets and "
             "model checkpoints next to the code. Write a `.gitignore` so that `git status` only offers the source "
             f"files ({', '.join(f'`{k}`' for k in keep)}) and `.gitignore` itself.",
             s, "printf 'venv/\\n__pycache__/\\n.env\\ncheckpoints/\\n' > .gitignore\ngit status --short",
             check=(["untracked"], "."))


@template(M, "easy")
def check_ignore(rng):
    rules = ["venv/", "__pycache__/", "*.pyc", ".env", "data/", "checkpoints/", "*.pt", "wandb/", "outputs/"]
    rng.shuffle(rules)
    target = rng.choice([("checkpoints/best.pt", "*.pt"), ("data/train.csv", "data/"), ("wandb/run-1/log", "wandb/"),
                         ("outputs/fig.png", "outputs/"), ("src/__pycache__/a.pyc", "*.pyc")])
    path = target[0]
    s = setup(GIT, "#: cwd ~/project", Repo(rng).commit("Initial commit", {".gitignore": "\n".join(rules) + "\n",
                                                                        "train.py": "print('hi')\n"}).code(),
              f"mkdir -p {q(path.rsplit('/', 1)[0])} && echo x > {q(path)}")
    items = [
        ("git check-ignore -v", f"`git status` doesn't show `{path}`. Show which `.gitignore` rule (file, line, "
                                f"pattern) ignores it.", f"git check-ignore -v {path}"),
        ("git status --ignored", "List the files git is ignoring here (in `git status`).",
         "git status --ignored --short", {"require": ["git status --ignored"]}),
    ]
    return catalog(rng, items, s)


# ================================================================= medium

@template(M, "medium")
def untrack(rng):
    what = rng.choice([("venv", "venv/"), (".env", ".env"), ("checkpoints", "checkpoints/"),
                       ("__pycache__", "__pycache__/")])
    fs = _junk(rng)
    r = Repo(rng)
    for p, t in fs.items():
        r.write(p, t)
    r.run("git add -A && git commit -qm 'Initial commit'")
    s = setup(GIT, "#: cwd ~/project", r.code())
    flag = "-r " if what[0] != ".env" else ""
    return P(f"`{what[0]}` was committed by mistake. Stop tracking it **without deleting it from disk**, ignore it "
             f"from now on, and commit both changes with the message `chore: stop tracking {what[0]}`.",
             s, f"git rm {flag}--cached {what[0]}\necho {q(what[1])} >> .gitignore\ngit add .gitignore\n"
                f"git commit -m 'chore: stop tracking {what[0]}'",
             post=f"/usr/lib/bashlite/gitstate files log index untracked; ls -A {what[0]} | head -3",
             answer=["<post>"], require=["git rm --cached"])


@template(M, "medium")
def negation(rng):
    keep = rng.choice(["README.md", "SOURCES.md", "download.sh"])
    fs = {"train.py": "print('train')\n", f"data/{keep}": "Where the data comes from.\n",
          "data/train.csv": "a,b\n1,2\n", "data/val.csv": "a,b\n3,4\n", "data/raw/dump.json": "{}\n"}
    s = setup(GIT, "#: cwd ~/project", "git init -q", _files(fs))
    return P(f"Ignore everything inside `data/` **except** `data/{keep}`, then stage everything that isn't ignored "
             f"(`git add .`).", s, f"printf 'data/*\\n!data/{keep}\\n' > .gitignore\ngit add .",
             check=(["index", "untracked"], "."))


# ================================================================= hard

@template(M, "hard")
def clone_options(rng):
    items = [
        ("git clone --recurse-submodules", "Clone `git-gym/app-with-submodule` over SSH **with** its submodule "
                                           "checked out, in one command.",
         "git clone --recurse-submodules git@github.com:git-gym/app-with-submodule.git",
         {"check": (["submodules"], "app-with-submodule")}),
        ("git clone -b", "Clone `minilibs/flask-mini` over SSH so that the maintenance branch `2.x` is checked out "
                         "(not main).", "git clone -b 2.x git@github.com:minilibs/flask-mini.git",
         {"check": (["head", "branches", "upstream"], "flask-mini")}),
        ("git clone --depth", "`openmaps/city-transit-data` is about 100 MB because of its history. Clone only the "
                              "latest commit of it over SSH.",
         "git clone --depth 1 git@github.com:openmaps/city-transit-data.git",
         {"post": "cd city-transit-data && git rev-list --count HEAD && git rev-parse --is-shallow-repository && "
                  "ls"}),
        ("git clone --bare", "Make a bare clone of `git-gym/recipe-book` (a server-style copy, `recipe-book.git`).",
         "git clone --bare git@github.com:git-gym/recipe-book.git",
         {"post": "git -C recipe-book.git rev-parse --is-bare-repository && git -C recipe-book.git branch"}),
    ]
    return catalog(rng, items, setup(GIT, GH), answer=["<post>"])


@template(M, "hard")
def global_excludes(rng):
    junk = rng.sample([".DS_Store", "*.swp", ".idea/", ".vscode/", "Thumbs.db"], 2)
    fs = ml_files(rng, 2)
    for j in junk:
        name = j.replace("*", "notes").rstrip("/")
        fs[name + ("/workspace.xml" if j.endswith("/") else "")] = "x\n"
    s = setup(GIT, "#: cwd ~/project", "git init -q", _files(fs))
    lines = "\\n".join(junk)
    return P(f"Editor and OS junk ({', '.join(f'`{j}`' for j in junk)}) shows up in every repository you make. "
             f"Ignore those patterns for **all** your repositories with a global excludes file `~/.gitignore_global`"
             f" (don't add a `.gitignore` here).",
             s, f"printf '{lines}\\n' > ~/.gitignore_global\ngit config --global core.excludesFile ~/.gitignore_global",
             check=(["untracked"], "."), require=["git config"])


# ================================================================= challenge

@template(M, "challenge")
def ml_project(rng):
    fs = _junk(rng)
    fs.update({"data/README.md": "How to download the data.\n", "data/train.csv": "x\n", "wandb/run-1/files": "x\n"})
    s = setup(GIT, "#: cwd ~/project", _files(fs))
    return P("**Start an ML repository.** Make `~/project` a git repository. Its `.gitignore` must keep out the "
             "virtual environment, bytecode caches, `.env`, `checkpoints/`, every `*.pt` file anywhere, `wandb/`, "
             "and everything in `data/` except `data/README.md`. Commit what's left with the message `Initial "
             "commit`.",
             s, "git init\nprintf 'venv/\\n__pycache__/\\n.env\\ncheckpoints/\\n*.pt\\nwandb/\\ndata/*\\n"
                "!data/README.md\\n' > .gitignore\ngit add .\ngit commit -m 'Initial commit'",
             post="/usr/lib/bashlite/gitstate files untracked; for p in x.pt sub/y.pt data/z.csv; do git check-ignore "
                  "-q $p && echo \"$p ignored\"; done", answer=["<post>"], require=["git init", "git commit"])


@template(M, "challenge")
def mirror(rng):
    return P("**A backup remote.** Make a bare clone of `minilibs/requests-mini` called `~/backup/requests-mini.git`"
             ", then clone that backup into `~/restored` and check out its tag `v1.0.0` as a detached HEAD.",
             setup(GIT, GH), "git clone --bare git@github.com:minilibs/requests-mini.git ~/backup/requests-mini.git\n"
                             "git clone ~/backup/requests-mini.git ~/restored\ncd ~/restored\n"
                             "git switch --detach v1.0.0",
             check=(["head", "tags", "remotes"], "~/restored"), require=["git clone"])

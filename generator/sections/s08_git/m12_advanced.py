"""8.12 Advanced: worktrees, submodules, LFS, hooks & history surgery."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.12"
LFS = "sudo apt install -y git-lfs"


def _clone(full, d=None):
    name = d or full.split("/")[1]
    return f"#: cwd ~/{name}\ngit clone -q git@github.com:{full}.git ."


def _venv(*pkgs):
    return f"#: venv /home/ada/.venv {' '.join(pkgs)}\n#: activate /home/ada/.venv"


# ================================================================= lite

@template(M, "lite")
def worktrees(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    exp = rng.choice(["experiment", "feature/kv-cache", "hotfix"])
    r.branch(exp).write("exp.py", "x = 1").commit("Experiment").switch("main")
    new = rng.choice(["feature/dropout", "fix/nan-loss", "spike"])
    items = [
        ("git worktree add", f"Check out the branch `{exp}` in a second working tree at `../proj-exp`, so main "
                             f"stays checked out here.", f"git worktree add ../proj-exp {exp}"),
        ("git worktree add", f"Create a new branch `{new}` from main in a new working tree `../proj-new`.",
         f"git worktree add -b {new} ../proj-new"),
        ("git worktree remove", f"Add a working tree `../proj-exp` for `{exp}`, then remove it again.",
         f"git worktree add ../proj-exp {exp}\ngit worktree remove ../proj-exp"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(["head", "branches", "worktrees"], "."))


@template(M, "lite")
def submodule_init(rng):
    items = [
        ("git submodule update", "`lib/shared` is an empty submodule directory. Fetch and check out the submodule.",
         "git submodule update --init", {"require": ["git submodule update"]}),
        ("git submodule status", "Show the submodule's recorded commit and whether it's checked out.",
         "git submodule status", {"answer": ["<stdout>"], "check": None}),
        ("git submodule update", "Check out the submodule, recursively, in one command.",
         "git submodule update --init --recursive", {"require": ["git submodule update"]}),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("git-gym/app-with-submodule")), check=(["submodules"], "."))


# ================================================================= easy

@template(M, "easy")
def lfs_track(rng):
    pat = rng.choice(["*.safetensors", "*.pt", "*.ckpt", "*.psd", "*.mp4"])
    ext = pat[1:]
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"head -c 200000 /dev/urandom > weights{ext}")
    return P(f"Large `{pat}` files should be stored with Git LFS. Install git-lfs (`{LFS}`), set it up, track "
             f"`{pat}`, and commit `.gitattributes` together with `weights{ext}` as `Track {pat} with LFS`.",
             s, f"{LFS}\ngit lfs install\ngit lfs track '{pat}'\ngit add .gitattributes weights{ext}\n"
                f"git commit -m 'Track {pat} with LFS'",
             check=(["log", "lfs", "index", "untracked"], "."), require=["git lfs track"])


@template(M, "easy")
def sparse(rng):
    a, b = rng.sample(["algorithms", "systems", "ml", "math"], 2)
    items = [
        ("git sparse-checkout set", f"This repository is big. Check out only the `{a}` and `{b}` folders (plus the "
                                    f"top-level files).", f"git sparse-checkout set {a} {b}"),
        ("git sparse-checkout disable", f"Check out only `{a}`, then go back to a full checkout.",
         f"git sparse-checkout set {a}\ngit sparse-checkout disable"),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("open-learning/cs-notes")),
                   post="/usr/lib/bashlite/gitstate sparse; ls", answer=["<post>"])


# ================================================================= medium

@template(M, "medium")
def lfs_pull(rng):
    items = [
        ("git lfs pull", "The images in this clone are tiny pointer files. Install git-lfs, set it up and download "
                         "the real contents.", f"{LFS}\ngit lfs install\ngit lfs pull"),
        ("git lfs ls-files", "Install git-lfs, set it up, download the real files, and list the files LFS manages.",
         f"{LFS}\ngit lfs install\ngit lfs pull\ngit lfs ls-files", {"require": ["git lfs ls-files"]}),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("git-gym/lfs-assets")), check=(["lfs", "worktree"], "."))


@template(M, "medium")
def submodule_add(rng):
    path = rng.choice(["external/shared", "vendor/shared-lib", "lib/shared"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    return P(f"Add `git-gym/shared-lib` (SSH URL) to this project as a submodule at `{path}`, and commit it as "
             f"`Add shared-lib submodule`.", setup(GIT, GH, "#: cwd ~/proj", r.code()),
             f"git submodule add git@github.com:git-gym/shared-lib.git {path}\n"
             f"git commit -m 'Add shared-lib submodule'",
             check=(["log", "files", "submodules", "index"], "."), require=["git submodule add"])


# ================================================================= hard

@template(M, "hard")
def purge_history(rng):
    what = rng.choice([[".env"], ["data/dump.sql"], [".env", "data/dump.sql"]])
    paths = " ".join(f"--path {p}" for p in what)
    shown = " and ".join(f"`{p}`" for p in what)
    return P(f"`git-gym/leaky-app` once committed {shown}; later commits deleted {'them' if len(what) > 1 else 'it'}, "
             f"but {'they are' if len(what) > 1 else 'it is'} still in history. In this fresh clone, remove "
             f"{'them' if len(what) > 1 else 'it'} from **every** commit with `git filter-repo` (installed in the "
             f"active virtual environment).",
             setup(GIT, GH, _venv("git-filter-repo"), _clone("git-gym/leaky-app")),
             f"git filter-repo {paths} --invert-paths",
             post="/usr/lib/bashlite/gitstate head log remotes; git log --all --format= --name-only | sort -u",
             answer=["<post>"], require=["git filter-repo"])


@template(M, "hard")
def hooks(rng):
    items = [
        ("pre-commit install", "This project uses pre-commit (installed in the active virtual environment). "
                               "Install its git hook, run every hook on all files, then commit what the hooks fixed "
                               "as `style: apply pre-commit fixes` (fix the remaining lint error first: drop the "
                               "unused `import os` and `import sys` in `src/report.py`).",
         "pre-commit install\npre-commit run --all-files\nsed -i '/^import os$/d; /^import sys$/d' src/report.py\n"
         "git add -A\ngit commit -m 'style: apply pre-commit fixes'",
         {"check": (["hooks", "log", "files", "worktree"], ".")}),
        ("pre-commit run", "Install pre-commit's git hook and run all hooks on every file once.",
         "pre-commit install\npre-commit run --all-files", {"check": (["hooks", "worktree"], ".")}),
        ("git archive", "Save a zip of the current commit as `~/snapshot.zip` with `git archive`.",
         "git archive --format=zip HEAD > ~/snapshot.zip", {"post": "unzip -l ~/snapshot.zip | awk '{print $4}' | "
                                                                    "grep . | sort"}),
    ]
    return catalog(rng, items, setup(GIT, GH, _venv("pre-commit"), _clone("git-gym/hooks-demo")), answer=["<post>"])


# ================================================================= challenge

@template(M, "challenge")
def bump_submodule(rng):
    return P("**Update a dependency.** `lib/shared` is pinned to shared-lib `v1.0.0`. Initialise it, move it to the "
             "tag `v1.1.0`, and commit the new pin in the app as `Bump shared-lib to v1.1.0`.",
             setup(GIT, GH, _clone("git-gym/app-with-submodule")),
             "git submodule update --init\ngit -C lib/shared switch --detach v1.1.0\ngit add lib/shared\n"
             "git commit -m 'Bump shared-lib to v1.1.0'",
             check=(["log", "files", "submodules"], "."), require=["git submodule update"])


@template(M, "challenge")
def parallel_fix(rng):
    s = setup(GIT, GH, _clone("minilibs/flask-mini"))
    return P("**A hotfix without stashing.** You're in the middle of something on main (leave the change to "
             "`README.md` alone). In a second working tree `../flask-hotfix` on the maintenance branch `2.x`, "
             "cherry-pick main's commit `Send X-Content-Type-Options: nosniff on 404 pages`, then remove the extra "
             "working tree.", s + "\necho 'half-done' >> README.md",
             "git worktree add ../flask-hotfix 2.x\ncd ../flask-hotfix\ngit cherry-pick ':/Send X-Content-Type'\n"
             "cd ../flask-mini\ngit worktree remove ../flask-hotfix",
             check=(["head", "branches", "worktree", "worktrees"], "."), require=["git worktree add"])

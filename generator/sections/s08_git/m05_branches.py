"""8.05 Branches."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, ml_files, q, setup, template, WORDS

M = "8.05"
ST = ["head", "branches", "log"]


def _repo(rng, branches=2, merged=1):
    """main plus feature branches: the first `merged` are merged into main, the rest not."""
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    r.append("README.md", "more").commit("Write README")
    names = rng.sample(BRANCHES, branches)
    for i, b in enumerate(names):
        r.switch("main").branch(b)
        r.write(f"{b.split('/')[1]}.py", f"# {b}\n").commit(f"Start {b}")
        if i < merged:
            r.switch("main").merge(b, f"Merge {b}")
    r.switch("main")
    return r, names


# ================================================================= lite

@template(M, "lite")
def create_switch(rng):
    r, names = _repo(rng, 1, 0)
    b = rng.choice([x for x in BRANCHES if x not in names])
    other = names[0]
    items = [
        ("git switch -c", f"Create a branch `{b}` and switch to it.", f"git switch -c {b}"),
        ("git branch", f"Create a branch `{b}` **without** switching to it.", f"git branch {b}"),
        ("git switch", f"Switch to the existing branch `{other}`.", f"git switch {other}",
         {"require": ["git switch|git checkout"]}),
        ("git switch -", f"Switch to `{other}`, then jump back to the branch you were on with one short command.",
         f"git switch {other}\ngit switch -", {"require": ["git switch -|git checkout -"]}),
        ("git checkout -b", f"Create and switch to `{b}` the older way, with `checkout`.", f"git checkout -b {b}"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "lite")
def list_branches(rng):
    r, names = _repo(rng, 3, 1)
    s = setup(GIT, GH, "#: cwd ~/requests-mini", "git clone -q git@github.com:minilibs/requests-mini.git .",
              "git switch -q -c local-work")
    items = [
        ("git branch", "List your local branches.", "git branch"),
        ("git branch -a", "List all branches, including the remote-tracking ones.", "git branch -a"),
        ("git branch -r", "List only the remote-tracking branches.", "git branch -r"),
        ("git branch -vv", "List local branches with their last commit and upstream.", "git branch -vv"),
    ]
    return catalog(rng, items, s)


# ================================================================= easy

@template(M, "easy")
def rename_delete(rng):
    r, names = _repo(rng, 3, 2)
    merged, unmerged = names[0], names[2]
    new = rng.choice(["feature/attention-v2", "feature/new-name", "wip/renamed"])
    items = [
        ("git branch -m", f"Rename the branch `{merged}` to `{new}`.", f"git branch -m {merged} {new}"),
        ("git branch -d", f"Delete the branch `{merged}` (it's already merged).", f"git branch -d {merged}"),
        ("git branch -D", f"`{unmerged}` was an experiment that failed. Delete it even though it isn't merged.",
         f"git branch -D {unmerged}"),
        ("git branch -m", "Rename the current branch to `trunk`.", "git branch -m trunk"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "easy")
def merged_lists(rng):
    r, names = _repo(rng, 4, 2)
    items = [
        ("git branch --merged", "List the branches already merged into the current branch.", "git branch --merged"),
        ("git branch --no-merged", "List the branches that still have commits main doesn't.",
         "git branch --no-merged"),
        ("git branch -v", "List branches with the hash and subject of their last commit.", "git branch -v"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


# ================================================================= medium

@template(M, "medium")
def feature_cycle(rng):
    r, names = _repo(rng, 1, 1)
    b = rng.choice([x for x in BRANCHES if x not in names])
    w = rng.choice(WORDS)
    return P(f"Start `{b}` from main, commit a new file `{w}.py` containing `{w} = True` with the message `Add "
             f"{w}`, then switch back to main.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git switch -c {b}\necho '{w} = True' > {w}.py\ngit add {w}.py\ngit commit -m 'Add {w}'\n"
             f"git switch main", check=(ST + ["untracked"], "."), require=["git switch|git checkout"])


@template(M, "medium")
def upstream(rng):
    s = setup(GIT, GH, "#: cwd ~/flask-mini", "git clone -q git@github.com:minilibs/flask-mini.git .")
    items = [
        ("git switch", "Create a local branch for the remote's `2.x` branch that tracks `origin/2.x` (one "
                       "command).", "git switch 2.x", {"check": (["head", "branches", "upstream"], ".")}),
        ("git branch -u", "Create a local branch `dev` at `origin/dev`, then set its upstream to `origin/dev` "
                          "with `-u`.", "git branch dev origin/dev\ngit branch -u origin/dev dev",
         {"check": (["branches", "upstream"], ".")}),
        ("git switch --track", "Create a local branch `maint` that starts at and tracks `origin/2.x`.",
         "git switch --track -c maint origin/2.x", {"check": (["head", "branches", "upstream"], "."),
                                                    "require": ["git switch|git checkout"]}),
    ]
    return catalog(rng, items, s)


# ================================================================= hard

@template(M, "hard")
def detached(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    for v in ("v1.0", "v1.1", "v2.0"):
        r.append("README.md", f"release {v}").commit(f"Release {v}").tag(v, f"Version {v}")
    r.append("README.md", "after").commit("Work after v2.0")
    v = rng.choice(["v1.0", "v1.1"])
    fix = rng.choice(["hotfix", "fix/old-release", "maint"])
    items = [
        ("git switch --detach", f"Look at the code as it was at `{v}`: check it out as a detached HEAD.",
         f"git switch --detach {v}", {"check": (["head"], ".")}),
        ("git switch -c", f"Check out `{v}` detached, then turn that spot into a new branch `{fix}` (switching to "
                          f"it).", f"git switch --detach {v}\ngit switch -c {fix}", {"check": (ST, ".")}),
        ("git branch", f"Without leaving main, create a branch `{fix}` that starts at `{v}`.",
         f"git branch {fix} {v}", {"check": (ST, ".")}),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "hard")
def orphan(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    b = rng.choice(["gh-pages", "site", "docs-site"])
    return P(f"Make a branch `{b}` with **no history** (an orphan branch) for a website: it should start with no "
             f"files, then contain only `index.html` with `<h1>Hello</h1>`, committed as `Site`.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git switch --orphan {b}\necho '<h1>Hello</h1>' > index.html\ngit add index.html\ngit commit -m Site",
             check=(["head", "branches", "log", "files"], "."), require=["git switch --orphan|git checkout --orphan"])


# ================================================================= challenge

@template(M, "challenge")
def cleanup(rng):
    r, names = _repo(rng, 5, 3)
    return P("**Branch spring cleaning.** Delete every branch that is already merged into main (keep main and the "
             "unmerged ones), then list the branches that remain.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             "git branch --merged main | grep -v main | xargs git branch -d\ngit branch",
             check=(["branches"], "."), require=["git branch -d|git branch -D"])


@template(M, "challenge")
def wrong_branch(rng):
    r, names = _repo(rng, 1, 1)
    b = rng.choice([x for x in BRANCHES if x not in names])
    w = rng.choice(WORDS)
    r.write(f"{w}.py", f"{w} = 1").commit(f"Add {w}")
    r.append(f"{w}.py", f"{w} += 1").commit(f"Improve {w}")
    return P(f"**Committed on the wrong branch.** The last two commits on main (`Add {w}`, `Improve {w}`) belong on "
             f"a new branch `{b}`. Put them there and move main back to where it was before them; end on main.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git branch {b}\ngit reset --hard HEAD~2", check=(ST + ["worktree", "untracked"], "."))

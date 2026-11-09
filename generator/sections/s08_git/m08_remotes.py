"""8.08 Remotes: fetch, pull, push & forks."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, ml_files, q, setup, template, WORDS

M = "8.08"
MATE = "GIT_AUTHOR_NAME='Grace Hopper' GIT_AUTHOR_EMAIL=grace@example.com"


def _own(name, src=None):
    """Setup: your GitHub repository ada/<name> (a copy of a catalog one, or empty) and a clone of it in ~/<name>."""
    lines = [GIT, GH, f"#: ghrepo ada/{name}" + (f" from={src}" if src else ""), f"#: cwd ~/{name}"]
    if src:
        lines.append(f"git clone -q git@github.com:ada/{name}.git .")
    return "\n".join(lines)


def _mate_push(name, n=1, file="NOTES.md", branch="main"):
    """Setup lines: a teammate clones ada/<name> elsewhere and pushes n commits to `branch`."""
    out = [f"git clone -q git@github.com:ada/{name}.git /tmp/mate", "cd /tmp/mate",
           f"git switch -q {branch}" if branch != "main" else ""]
    for i in range(n):
        out.append(f"echo 'teammate note {i}' >> {file} && git add {file} && {MATE} git commit -qm 'Teammate note {i + 1}'")
    out += [f"git push -q origin {branch}", f"cd ~/{name}", "rm -rf /tmp/mate"]
    return "\n".join(x for x in out if x)


# ================================================================= lite

@template(M, "lite")
def manage_remotes(rng):
    s = _own("cs-notes", "open-learning/cs-notes")
    new = rng.choice(["notes", "cs-notes-2026", "lecture-notes"])
    items = [
        ("git remote -v", "Show the remotes with their URLs.", "git remote -v", {"answer": ["<stdout>"]}),
        ("git remote add", "Add a second remote `upstream` for the original `open-learning/cs-notes` (SSH URL).",
         "git remote add upstream git@github.com:open-learning/cs-notes.git"),
        ("git remote set-url", f"The GitHub repository was renamed to `ada/{new}`. Point `origin` at "
                               f"`git@github.com:ada/{new}.git`.",
         f"git remote set-url origin git@github.com:ada/{new}.git"),
        ("git remote rename", "Rename the remote `origin` to `github`.", "git remote rename origin github"),
        ("git remote remove", "Remove the remote `origin` from this repository.", "git remote remove origin",
         {"require": ["git remote remove|git remote rm"]}),
        ("git remote show", "Show everything git knows about `origin` (`remote show`).", "git remote show origin",
         {"answer": ["<stdout>"]}),
    ]
    return catalog(rng, items, s, check=(["remotes", "upstream"], "."))


@template(M, "lite")
def fetch_pull(rng):
    s = setup(_own("leftpad", "minilibs/leftpad"), _mate_push("leftpad", rng.randint(1, 2)))
    items = [
        ("git fetch", "A teammate pushed to GitHub. Download their commits **without** changing your branch or "
                      "files.", "git fetch", {"check": (["head", "branches", "remotes", "files"], ".")}),
        ("git pull", "A teammate pushed to GitHub. Bring their commits into your `main`.", "git pull",
         {"check": (["head", "branches", "remotes", "files"], ".")}),
        ("git log", "A teammate pushed to GitHub. Fetch, then list (one line each) the commits on `origin/main` "
                    "that your `main` doesn't have.", "git fetch\ngit log --oneline main..origin/main",
         {"answer": ["<stdout>"]}),
    ]
    return catalog(rng, items, s)


# ================================================================= easy

@template(M, "easy")
def first_push(rng):
    name = rng.choice(["ml-notes", "dotfiles", "playground", "thesis-code"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    r.append("README.md", "more").commit("Write README")
    s = setup(GIT, GH, f"#: ghrepo ada/{name}", f"#: cwd ~/{name}", r.code())
    return P(f"Your empty GitHub repository `ada/{name}` is waiting. Connect this local repository to it as "
             f"`origin` (SSH URL) and push `main`, making `origin/main` its upstream.",
             s, f"git remote add origin git@github.com:ada/{name}.git\ngit push -u origin main",
             check=(["remotes", "upstream", f"gh=ada/{name}"], "."), require=["git push -u|git push --set-upstream"])


@template(M, "easy")
def push_things(rng):
    b = rng.choice(BRANCHES)
    s = setup(_own("requests-mini", "minilibs/requests-mini"), f"git switch -qc {b}",
              f"echo '# {b}' >> README.md && git commit -qam 'Work on {b}'", "git switch -q main",
              "git tag -a v1.2.0 -m 'requests-mini 1.2.0'", "git tag nightly")
    items = [
        ("git push", f"Publish your local branch `{b}` to GitHub (and track it).", f"git push -u origin {b}"),
        ("git push --tags", "Push all your local tags to GitHub.", "git push --tags"),
        ("git push", "Push only the tag `v1.2.0` to GitHub.", "git push origin v1.2.0"),
        ("git push --delete", "Delete the branch `feature/retries` on GitHub (not locally).",
         "git push origin --delete feature/retries", {"require": ["git push --delete|git push"]}),
    ]
    return catalog(rng, items, s, check=(["branches", "upstream", "remotes", "gh=ada/requests-mini"], "."))


# ================================================================= medium

@template(M, "medium")
def rejected_push(rng):
    w = rng.choice(WORDS)
    s = setup(_own("hugo-mini", "toolsmiths/hugo-mini"), _mate_push("hugo-mini", 1),
              f"echo '# {w}' >> README.md && git commit -qam 'Document {w}'")
    items = [
        ("git pull --rebase", "Your push would be rejected: a teammate pushed first. Put your commit on top of "
                              "theirs (rebase, no merge commit) and push.", "git pull --rebase\ngit push"),
        ("git pull", "Your push would be rejected: a teammate pushed first. Integrate their work with a merge "
                     "(git's default merge message) and push.", "git pull --no-rebase --no-edit\ngit push",
         {"require": ["git pull"]}),
    ]
    return catalog(rng, items, s, check=(["head", "log", "files", "gh=ada/hugo-mini"], "."))


@template(M, "medium")
def prune(rng):
    s = setup(_own("nanogpt-mini", "mini-ml/nanogpt-mini"),
              "git clone -q git@github.com:ada/nanogpt-mini.git /tmp/m && git -C /tmp/m push -q origin --delete "
              "experiment/rope && rm -rf /tmp/m")
    items = [
        ("git fetch --prune", "Someone deleted `experiment/rope` on GitHub, but your `origin/experiment/rope` is "
                              "still there. Fetch and drop remote-tracking branches that no longer exist.",
         "git fetch --prune", {"require": ["git fetch --prune|git fetch -p|git remote prune"]}),
        ("git remote prune", "Remove stale remote-tracking branches for `origin` without fetching anything new.",
         "git remote prune origin"),
    ]
    return catalog(rng, items, s, check=(["remotes", "branches"], "."))


# ================================================================= hard

@template(M, "hard")
def sync_fork(rng):
    s = setup(_own("awesome-ml", "open-learning/awesome-ml"), "git push -q -f origin HEAD~2:main",
              "git fetch -q && git reset -q --hard origin/main")
    how = rng.choice(["rebase", "merge"])
    sol = ("git remote add upstream git@github.com:open-learning/awesome-ml.git\ngit fetch upstream\n" +
           ("git rebase upstream/main" if how == "rebase" else "git merge --ff-only upstream/main") +
           "\ngit push origin main")
    return P("`ada/awesome-ml` is your fork of `open-learning/awesome-ml`, and it has fallen behind. Add the "
             "original as a remote `upstream` (SSH URL), bring your `main` up to date with `upstream/main`, and push "
             "it to your fork.", s, sol, check=(["head", "log", "remotes", "gh=ada/awesome-ml"], "."),
             require=["git remote add", "git fetch"])


@template(M, "hard")
def tracking_info(rng):
    s = setup(_own("flask-mini", "minilibs/flask-mini"), "git switch -q 2.x", "git switch -q main",
              "echo x >> README.md && git commit -qam 'Local change'", "git reset -q --hard HEAD")
    items = [
        ("git branch -vv", "Show each local branch with its upstream and how far ahead/behind it is.",
         "git branch -vv"),
        ("git status -sb", "Show the short status with the branch line (ahead/behind).", "git status -sb",
         {"require": ["git status"]}),
        ("git rev-list --count", "Print how many commits `main` is ahead of its upstream (just the number).",
         "git rev-list --count @{u}..HEAD", {"require": ["git rev-list"]}),
    ]
    return catalog(rng, items, s)


# ================================================================= challenge

@template(M, "challenge")
def collaboration(rng):
    s = setup(_own("datetime-utils", "minilibs/datetime-utils"), _mate_push("datetime-utils", 1, "CHANGELOG.md"),
              "echo '- local note' >> CHANGELOG.md && git commit -qam 'docs: local note'")
    return P("**Team sync.** A teammate pushed a change to `CHANGELOG.md`; your own unpushed commit edits the end of "
             "the same file. Rebase your commit onto theirs; in the conflict keep **both** additions (the "
             "teammate's line first, then yours), then push.",
             s, "git pull --rebase\ngit show origin/main:CHANGELOG.md > CHANGELOG.md\n"
                "echo '- local note' >> CHANGELOG.md\ngit add CHANGELOG.md\nGIT_EDITOR=true git rebase --continue\n"
                "git push",
             check=(["head", "log", "files", "state", "gh=ada/datetime-utils"], "."), require=["git push"])


@template(M, "challenge")
def second_remote(rng):
    s = setup(_own("dotfiles", "mira-chen/dotfiles"), "git switch -q macos", "git switch -q main",
              "git tag -a v1 -m 'my setup v1'")
    return P("**A backup copy.** Create a bare repository `~/backup/dotfiles.git`, add it as a remote called "
             "`backup`, and push **all** local branches and tags to it.",
             s, "git init --bare ~/backup/dotfiles.git\ngit remote add backup ~/backup/dotfiles.git\n"
                "git push backup --all\ngit push backup --tags",
             post="/usr/lib/bashlite/gitstate remotes; cd ~/backup/dotfiles.git && git branch && git tag",
             answer=["<post>"], require=["git remote add", "git push"])

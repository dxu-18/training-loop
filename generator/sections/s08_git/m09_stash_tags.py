"""8.09 Stash, tags & cherry-pick."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, ml_files, q, setup, template, WORDS

M = "8.09"
ST = ["head", "branches", "log", "index", "worktree", "untracked", "stash"]


def _dirty(rng):
    fs = ml_files(rng, 3)
    r = Repo(rng).commit("Initial commit", fs)
    a, b = sorted(fs)[:2]
    lines = [f"echo '# wip {rng.choice(WORDS)}' >> {a}", f"echo '# wip' >> {b}", f"git add {b}",
             "echo 'scratch' > scratch.txt"]
    return r, a, b, lines


# ================================================================= lite

@template(M, "lite")
def stash_basics(rng):
    r, a, b, dirty = _dirty(rng)
    msg = rng.choice(["wip attention", "half-done refactor", "try new lr"])
    s = setup(GIT, "#: cwd ~/proj", r.code(), *dirty)
    items = [
        ("git stash", "Put your uncommitted changes to tracked files aside (stash them).", "git stash"),
        ("git stash -u", "Stash everything, including the untracked `scratch.txt`.", "git stash -u",
         {"require": ["git stash -u|git stash --include-untracked"]}),
        ("git stash push", f"Stash only the changes to `{a}`, with the message `{msg}`.",
         f"git stash push -m {q(msg)} -- {a}", {"require": ["git stash push"]}),
        ("git stash pop", "Stash your changes, then bring them back with `pop`.", "git stash\ngit stash pop"),
    ]
    return catalog(rng, items, s, check=(ST, "."))


@template(M, "lite")
def tags(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.append("README.md", "x").commit("Prepare release")
    v = rng.choice(["v1.0", "v0.3.0", "v2.1.0"])
    items = [
        ("git tag", f"Tag the current commit `{v}` (a lightweight tag).", f"git tag {v}"),
        ("git tag -a", f"Make an annotated tag `{v}` with the message `first release`.",
         f"git tag -a {v} -m 'first release'"),
        ("git tag", f"Tag the commit **before** the current one as `{v}-rc1` (lightweight).", f"git tag {v}-rc1 HEAD~1"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(["tags"], "."))


# ================================================================= easy

@template(M, "easy")
def stash_list(rng):
    r, a, b, dirty = _dirty(rng)
    s = setup(GIT, "#: cwd ~/proj", r.code(), dirty[0], "git stash -q -m 'first idea'", dirty[1],
              "git stash -q -m 'second idea'", "echo 'third idea' > third.txt")
    items = [
        ("git stash list", "List your stashes.", "git stash list", {"answer": ["<stdout>"]}),
        ("git stash show -p", "Show the full diff of the **older** stash (`first idea`).",
         "git stash show -p stash@{1}", {"answer": ["<stdout>"], "require": ["git stash show"]}),
        ("git stash apply", "Apply the `first idea` stash on top of your current changes but keep it in the "
                            "stash list.", "git stash apply stash@{1}"),
        ("git stash drop", "Throw away the `second idea` stash (the newest one).", "git stash drop stash@{0}",
         {"require": ["git stash drop"]}),
        ("git stash clear", "Delete all your stashes at once.", "git stash clear"),
    ]
    return catalog(rng, items, s, check=(ST, "."))


@template(M, "easy")
def push_tags(rng):
    s = setup(GIT, GH, "#: ghrepo ada/leftpad from=minilibs/leftpad", "#: cwd ~/leftpad",
              "git clone -q git@github.com:ada/leftpad.git .", "echo 'v' >> README.md && git commit -qam 'Release'",
              "git push -q")
    v = rng.choice(["v1.4.0", "v2.0.0"])
    items = [
        ("git push", f"Tag the latest commit as an annotated `{v}` (message `leftpad {v}`) and push that tag to "
                     f"GitHub.", f"git tag -a {v} -m 'leftpad {v}'\ngit push origin {v}"),
        ("git push --delete", "The tag `v1.0.1` on GitHub was a mistake. Delete it both locally and on GitHub.",
         "git tag -d v1.0.1\ngit push origin --delete v1.0.1", {"require": ["git tag -d"]}),
        ("git describe", "Print how the current commit relates to the newest tag (`git describe --tags`).",
         "git describe --tags", {"answer": ["<stdout>"], "check": None}),
    ]
    return catalog(rng, items, s, check=(["tags", "gh=ada/leftpad"], "."))


# ================================================================= medium

@template(M, "medium")
def interrupt(rng):
    r, a, b, dirty = _dirty(rng)
    w = rng.choice(WORDS)
    r.branch("feature/work").switch("main")
    s = setup(GIT, "#: cwd ~/proj", r.code(), "git switch -q feature/work", *dirty[:3])
    return P(f"You're mid-change on `feature/work` when a bug report arrives. Stash your work, switch to main, "
             f"commit a fix that appends `# fix {w}` to `README.md` as `Fix {w}`, go back to `feature/work` and "
             f"restore your stashed changes (staged ones staged again).",
             s, f"git stash\ngit switch main\necho '# fix {w}' >> README.md\ngit commit -am 'Fix {w}'\n"
                f"git switch feature/work\ngit stash pop --index",
             check=(ST, "."), require=["git stash"])


@template(M, "medium")
def cherry_pick(rng):
    s = setup(GIT, GH, "#: cwd ~/payments-service", "git clone -q git@github.com:acme-corp/payments-service.git .",
              "git switch -q release/1.5")
    items = [
        ("git cherry-pick", "On `release/1.5`, copy main's commit `Validate webhook signatures with constant-time "
                            "comparison` (find it with `git log main`).",
         "git cherry-pick ':/Validate webhook signatures'"),
        ("git cherry-pick -x", "On `release/1.5`, copy main's webhook-signature fix, recording where it came from "
                               "in the message (`-x`).", "git cherry-pick -x ':/Validate webhook signatures'"),
        ("git cherry-pick --no-commit", "On `release/1.5`, apply main's webhook-signature fix to the working tree "
                                        "and index **without** committing it.",
         "git cherry-pick --no-commit ':/Validate webhook signatures'", {"require": ["git cherry-pick --no-commit|"
                                                                                    "git cherry-pick -n"]}),
    ]
    return catalog(rng, items, s, check=(ST + ["files"], "."))


# ================================================================= hard

@template(M, "hard")
def stash_branch(rng):
    r, a, b, dirty = _dirty(rng)
    r.append("README.md", "moved on").commit("Main moved on")
    br = rng.choice(["try-fix", "experiment/stash", "wip/idea"])
    s = setup(GIT, "#: cwd ~/proj", r.code(), "git switch -q --detach HEAD~1", *dirty[:2], "git stash -q",
              "git switch -q main")
    return P(f"An old stash was made on an earlier commit and doesn't apply cleanly to main any more. Turn it into a "
             f"new branch `{br}` (starting where the stash was made, with the changes applied) using `git stash "
             f"branch`.", s, f"git stash branch {br}", check=(ST, "."), require=["git stash branch"])


@template(M, "hard")
def cherry_range(rng):
    b = rng.choice(BRANCHES)
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.branch(b)
    ws = rng.sample(WORDS, 4)
    for w in ws:
        r.write(f"{w}.py", f"{w} = 1").commit(f"Add {w}")
    r.switch("main").append("README.md", "main").commit("Main work")
    items = [
        ("git cherry-pick", f"Copy the **last two** commits of `{b}` (`Add {ws[2]}`, `Add {ws[3]}`) onto main, in "
                            f"order, with one command.", f"git cherry-pick {b}~2..{b}"),
        ("git cherry-pick", f"Copy only `Add {ws[1]}` and `Add {ws[3]}` from `{b}` onto main (in that order).",
         f"git cherry-pick {b}~2 {b}"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST + ["files"], "."))


# ================================================================= challenge

@template(M, "challenge")
def backport(rng):
    s = setup(GIT, GH, "#: ghrepo ada/payments-service from=acme-corp/payments-service", "#: cwd ~/payments-service",
              "git clone -q git@github.com:ada/payments-service.git .")
    return P("**Backport a security fix.** Issue #9: main has `Validate webhook signatures with constant-time "
             "comparison`. Cherry-pick it (with `-x`) onto **both** `release/1.4` and `release/1.5`, tag the results "
             "`v1.4.2` and `v1.5.1` (annotated, messages `v1.4.2` / `v1.5.1`), and push both branches and both tags "
             "to your repository.",
             s, "git switch release/1.4\ngit cherry-pick -x ':/Validate webhook'\ngit tag -a v1.4.2 -m v1.4.2\n"
                "git switch release/1.5\ngit cherry-pick -x ':/Validate webhook'\ngit tag -a v1.5.1 -m v1.5.1\n"
                "git push origin release/1.4 release/1.5 v1.4.2 v1.5.1",
             check=(["tags", "gh=ada/payments-service"], "."), require=["git cherry-pick"])


@template(M, "challenge")
def release_notes(rng):
    s = setup(GIT, GH, "#: cwd ~/requests-mini", "git clone -q git@github.com:minilibs/requests-mini.git .")
    a, b = rng.choice([("v0.2.0", "v1.0.0"), ("v1.0.0", "v1.1.0"), ("v0.1.0", "v0.2.0")])
    return P(f"**Release notes.** Print the subjects of the commits that went into `{b}` since `{a}` (not merges), "
             f"oldest first, each prefixed by `- `.",
             s, f"git log --no-merges --reverse --format='- %s' {a}..{b}", require=["git log"])

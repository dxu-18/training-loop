"""8.10 Undoing things."""

from ._common import GH, GIT, P, Repo, catalog, history, ml_files, q, setup, template, WORDS

M = "8.10"
ST = ["head", "branches", "log", "index", "worktree", "untracked", "files"]


def _repo(rng, n=5):
    r = history(rng, n)
    return r


# ================================================================= lite

@template(M, "lite")
def restore(rng):
    r = _repo(rng, 3)
    fs = sorted(r.files)
    a, b = fs[:2]
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo 'broken' >> {a}", f"echo 'staged' >> {b}", f"git add {b}")
    items = [
        ("git restore", f"Throw away your unstaged edits to `{a}`.", f"git restore {a}"),
        ("git restore --staged", f"Unstage `{b}` but keep the edit in the file.", f"git restore --staged {b}"),
        ("git restore --source", f"Get `{a}` back exactly as it was two commits ago (in the working tree).",
         f"git restore --source=HEAD~2 {a}"),
        ("git restore", f"Throw away **every** change to `{b}`, staged and unstaged.",
         f"git restore --staged --worktree {b}", {"require": ["git restore"]}),
    ]
    return catalog(rng, items, s, check=(ST, "."))


@template(M, "lite")
def reset_modes(rng):
    r = _repo(rng, 5)
    items = [
        ("git reset --soft", "Undo the last commit but keep its changes **staged**.", "git reset --soft HEAD~1"),
        ("git reset", "Undo the last commit and keep its changes in the files, **unstaged**.", "git reset HEAD~1",
         {"require": ["git reset"]}),
        ("git reset --hard", "Undo the last commit and throw its changes away.", "git reset --hard HEAD~1"),
        ("git reset --hard", "Throw away the last **two** commits entirely.", "git reset --hard HEAD~2"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


# ================================================================= easy

@template(M, "easy")
def revert(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    for w in rng.sample(WORDS, 4):                    # each commit adds its own file: any of them reverts cleanly
        r.write(f"{w}.py", f"{w} = True").commit(f"Add {w}")
    k = rng.randint(1, 2)
    items = [
        ("git revert", "The last commit was already pushed. Undo it safely with a new commit (keep git's "
                       "message).", "git revert --no-edit HEAD", {"require": ["git revert"]}),
        ("git revert", f"Undo the commit `HEAD~{k}` (already pushed) with a new commit, keeping git's message.",
         f"git revert --no-edit HEAD~{k}", {"require": ["git revert"]}),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "easy")
def clean(rng):
    r = _repo(rng, 2)
    s = setup(GIT, "#: cwd ~/proj", r.code(), "echo 'tmp' > scratch.txt", "mkdir -p out && echo 1 > out/run.log",
              "echo '*.log' > .gitignore && git add .gitignore && git commit -qm 'Ignore logs'",
              "echo 'debug' > debug.log")
    items = [
        ("git clean -n", "Show which untracked files `git clean` **would** delete, without deleting anything.",
         "git clean -n", {"answer": ["<stdout>"], "check": None}),
        ("git clean -f", "Delete the untracked files and directories (not the ignored ones).", "git clean -fd",
         {"require": ["git clean -f|git clean -fd"]}),
        ("git clean -x", "Delete everything untracked, **ignored files included**, directories too.",
         "git clean -fdx", {"require": ["git clean -x|git clean -fdx|git clean -xfd"]}),
    ]
    return catalog(rng, items, s, check=(ST, "."))


# ================================================================= medium

@template(M, "medium")
def reflog_rescue(rng):
    r = _repo(rng, 6)
    n = rng.randint(2, 3)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git reset -q --hard HEAD~{n}")
    items = [
        ("git reflog", f"You ran `git reset --hard HEAD~{n}` by mistake. Use the reflog to put `main` back where "
                       f"it was before.", "git reset --hard HEAD@{1}", {"require": ["git reset"]}),
        ("git branch", f"You ran `git reset --hard HEAD~{n}` by mistake. Keep main as it is now, but save the lost "
                       f"commits on a new branch `rescue` (find them in the reflog).", "git branch rescue HEAD@{1}"),
    ]
    return catalog(rng, items, s, check=(ST, "."))


@template(M, "medium")
def match_remote(rng):
    s = setup(GIT, GH, "#: cwd ~/hugo-mini", "git clone -q git@github.com:toolsmiths/hugo-mini.git .",
              "echo junk >> README.md && git commit -qam 'Local experiment 1'",
              "echo more >> main.go && git commit -qam 'Local experiment 2'", "echo dirty >> go.mod")
    return P("Your local `main` is a mess of experiments. Make it identical to GitHub's `main` again (commits and "
             "files), throwing the local work away.", s, "git fetch\ngit reset --hard origin/main",
             check=(ST, "."), require=["git reset --hard"])


# ================================================================= hard

@template(M, "hard")
def recommit(rng):
    r = _repo(rng, 3)
    w = rng.sample(WORDS, 3)
    for x in w:
        r.write(f"{x}.py", f"{x} = 1").commit(f"wip {x}")
    msg = f"feat: add {w[0]}, {w[1]} and {w[2]}"
    return P(f"The last three commits are `wip` commits. Undo them softly and recommit their combined changes as "
             f"one commit `{msg}`.", setup(GIT, "#: cwd ~/proj", r.code()),
             f"git reset --soft HEAD~3\ngit commit -m {q(msg)}", check=(ST, "."), require=["git reset"])


@template(M, "hard")
def revert_merge(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.branch("feature/bad")
    r.write("bad.py", "boom = True").commit("Add bad feature")
    r.switch("main").append("README.md", "x").commit("Main work")
    r.merge("feature/bad", "Merge feature/bad")
    r.append("README.md", "after").commit("Later work")
    return P("`Merge feature/bad` (one commit before HEAD) shipped a broken feature and is already pushed. Undo the "
             "whole merge with a revert, keeping main's side (mainline 1), with git's message.",
             setup(GIT, "#: cwd ~/proj", r.code()), "git revert --no-edit -m 1 HEAD~1",
             check=(ST, "."), require=["git revert -m|git revert"])


# ================================================================= challenge

@template(M, "challenge")
def lost_branch(rng):
    r = _repo(rng, 3)
    b = rng.choice(["feature/kv-cache", "experiment/lr", "feature/wandb"])
    r.branch(b)
    for i in range(2):
        r.write(f"exp_{i}.py", f"x = {i}").commit(f"Experiment {i + 1}")
    r.switch("main")
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git branch -q -D {b}")
    return P(f"**Deleted the wrong branch.** You just force-deleted `{b}`, which had two unmerged commits. Bring the "
             f"branch back exactly as it was (stay on main).", s,
             f"git branch {b} HEAD@{{1}}",
             check=(ST, "."), require=["git branch"])


@template(M, "challenge")
def undo_table(rng):
    r = _repo(rng, 4)
    fs = sorted(r.files)
    a, b = fs[:2]
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo 'oops' >> {a}", f"echo 'keep' >> {b}", f"git add {b}",
              "echo junk > junk.tmp")
    return P(f"**A careful clean-up.** Discard the unstaged edit to `{a}`, keep the staged change to `{b}` but turn "
             f"it into its own commit `Update {b}`, delete the untracked `junk.tmp`, and then undo that new commit "
             f"again with a revert (git's message).",
             s, f"git restore {a}\ngit commit -m 'Update {b}'\ngit clean -f\ngit revert --no-edit HEAD",
             check=(ST, "."), require=["git revert", "git restore"])

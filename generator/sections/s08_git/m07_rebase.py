"""8.07 Rebasing & rewriting history.

Interactive rebases open the todo list in the editor (vim here, like EDITOR on a fresh Ubuntu); the reference
solutions use GIT_SEQUENCE_EDITOR with sed instead, which is equally valid."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.07"
ST = ["head", "branches", "log", "files", "state", "index", "worktree"]
EDIT = "The todo list opens in vim (`:wq` saves), or set `GIT_SEQUENCE_EDITOR` to edit it with a command."


def _behind(rng, conflict=False):
    b = rng.choice(BRANCHES)
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2) | {"settings.txt": "mode = base\n"})
    r.branch(b)
    for i in range(2):
        r.write(f"{b.split('/')[1]}_{i}.py", f"# part {i}\n").commit(f"{b}: part {i + 1}")
    if conflict:
        r.write("settings.txt", "mode = branch\n").commit(f"{b}: settings")
    r.switch("main")
    for i in range(2):
        r.append("README.md", f"main {i}").commit(f"Main work {i + 1}")
    if conflict:
        r.write("settings.txt", "mode = main\n").commit("Main settings")
    return r, b


def _wip(rng, n=4):
    w = rng.choice(WORDS)
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    msgs = ["wip", "more wip", "fix typo", "tests", "oops"][:n]
    for i, m in enumerate(msgs):
        r.append(f"{w}.py", f"step_{i} = {i}").commit(m)
    return r, w


# ================================================================= lite

@template(M, "lite")
def rebase_onto_main(rng):
    r, b = _behind(rng)
    items = [
        ("git rebase", f"`{b}` is behind main. Replay its commits on top of the latest main (stay on `{b}`).",
         f"git switch {b}\ngit rebase main"),
        ("git rebase", f"Rebase `{b}` onto main without switching to it first (one command), then go back to main.",
         f"git rebase main {b}\ngit switch main"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "lite")
def rebase_controls(rng):
    r, b = _behind(rng, conflict=True)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git switch -q {b}", "git rebase main >/dev/null 2>&1 || true")
    items = [
        ("git rebase --abort", f"Rebasing `{b}` onto main stopped on a conflict in `settings.txt`. Cancel the "
                               f"whole rebase, back to how the branch was.", "git rebase --abort"),
        ("git rebase --skip", f"Rebasing `{b}` stopped on its `settings` commit, which you don't need any more. "
                              f"Skip that commit and let the rebase finish.", "git rebase --skip"),
        ("git rebase --continue", "A rebase stopped on a conflict in `settings.txt`. Keep main's line "
                                  "(`mode = main`), mark it resolved and continue (the commit becomes empty: keep "
                                  "going until the rebase is done).",
         "git checkout --ours settings.txt\ngit add settings.txt\nGIT_EDITOR=true git rebase --continue",
         {"require": ["git rebase --continue|git rebase --skip"]}),
    ]
    return catalog(rng, items, s, check=(ST, "."))


# ================================================================= easy

@template(M, "easy")
def squash(rng):
    n = rng.randint(3, 4)
    r, w = _wip(rng, n)
    msg = f"feat: add {w}"
    items = [
        ("git rebase -i", f"Squash the last {n} commits into one commit with the message `{msg}`. {EDIT}",
         f"GIT_SEQUENCE_EDITOR=\"sed -i '2,$ s/^pick/fixup/'\" git rebase -i HEAD~{n}\ngit commit --amend -m {q(msg)}"),
        ("git reset --soft", f"Squash the last {n} commits into one commit `{msg}` using a soft reset.",
         f"git reset --soft HEAD~{n}\ngit commit -m {q(msg)}"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "easy")
def drop_commit(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    subjects = [f"Add {w}" for w in rng.sample(WORDS, 4)]
    for s_ in subjects:
        r.write(s_.split()[1] + ".py", "x = 1").commit(s_)
    k = rng.randint(1, 2)
    victim = subjects[k]
    return P(f"Remove the commit `{victim}` from history (its file goes with it); keep the commits around it. "
             f"{EDIT}", setup(GIT, "#: cwd ~/proj", r.code()),
             f"GIT_SEQUENCE_EDITOR=\"sed -i '/{victim}/d'\" git rebase -i HEAD~4",
             check=(ST, "."), require=["git rebase -i|git rebase"])


# ================================================================= medium

@template(M, "medium")
def reword_old(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    w = rng.sample(WORDS, 3)
    r.write(f"{w[0]}.py", "a = 1").commit(f"Add {w[0]} modlue")
    r.write(f"{w[1]}.py", "b = 2").commit(f"Add {w[1]}")
    r.write(f"{w[2]}.py", "c = 3").commit(f"Add {w[2]}")
    return P(f"Two commits back there's a typo: `Add {w[0]} modlue`. Change that message to `Add {w[0]} module` "
             f"without touching the later commits' content. {EDIT}",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"GIT_SEQUENCE_EDITOR=\"sed -i '1s/^pick/edit/'\" git rebase -i HEAD~3\n"
             f"git commit --amend -m 'Add {w[0]} module'\ngit rebase --continue",
             check=(ST, "."), require=["git rebase -i"])


@template(M, "medium")
def fixup_autosquash(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    w = rng.sample(WORDS, 3)
    r.write(f"{w[0]}.py", f"def {w[0]}():\n    return 1\n").commit(f"Add {w[0]}")
    r.write(f"{w[1]}.py", "b = 2").commit(f"Add {w[1]}")
    r.write(f"{w[2]}.py", "c = 3").commit(f"Add {w[2]}")
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# docstring for {w[0]}' >> {w[0]}.py")
    return P(f"The change in `{w[0]}.py` belongs in the older commit `Add {w[0]}`. Commit it as a fixup of that "
             f"commit, then fold it in automatically with an autosquash rebase. {EDIT}",
             s, f"git commit -a --fixup ':/Add {w[0]}'\nGIT_SEQUENCE_EDITOR=true git rebase -i --autosquash HEAD~4",
             check=(ST, "."), require=["git commit --fixup", "git rebase --autosquash"])


# ================================================================= hard

@template(M, "hard")
def onto(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.branch("feature/base")
    r.write("base.py", "x = 1").commit("Base work (abandoned)")
    r.branch("feature/top")
    r.write("top.py", "y = 2").commit("Top: part 1")
    r.append("top.py", "y += 1").commit("Top: part 2")
    r.switch("main")
    r.append("README.md", "main moved").commit("Main moves on")
    return P("`feature/top` was started on top of `feature/base`, which has been abandoned. Move only "
             "`feature/top`'s own two commits onto main (`git rebase --onto`), leaving the base work behind.",
             setup(GIT, "#: cwd ~/proj", r.code()), "git rebase --onto main feature/base feature/top",
             check=(ST, "."), require=["git rebase --onto"])


@template(M, "hard")
def rebase_conflict(rng):
    r, b = _behind(rng, conflict=True)
    return P(f"Rebase `{b}` onto main. When `settings.txt` conflicts, make it read `mode = both`, and finish the "
             f"rebase keeping the commit messages.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git switch {b}\ngit rebase main\necho 'mode = both' > settings.txt\ngit add settings.txt\n"
             f"GIT_EDITOR=true git rebase --continue",
             check=(ST, "."), require=["git rebase --continue"])


# ================================================================= challenge

@template(M, "challenge")
def messy_history(rng):
    s = setup(GIT, GH, "#: ghrepo ada/messy-history from=git-gym/messy-history", "#: cwd ~/messy-history",
              "git clone -q git@github.com:ada/messy-history.git .", "git switch -q feature/search")
    return P("**Clean up before review.** `feature/search` (in your repository `ada/messy-history`) is eight messy "
             "commits. Turn them into **one** commit `feat: case-insensitive search` on top of the latest "
             "`origin/main`, and update the branch on GitHub safely (it was pushed before). " + EDIT,
             s, "git fetch\ngit reset --soft $(git merge-base HEAD origin/main)\n"
                "git commit -m 'feat: case-insensitive search'\ngit rebase origin/main\n"
                "git push --force-with-lease",
             check=(["head", "log", "files", "gh=ada/messy-history"], "."),
             require=["git push --force-with-lease"])


@template(M, "challenge")
def reorder(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    w = rng.sample(WORDS, 3)
    for x in w:
        r.write(f"{x}.py", f"{x} = 1").commit(f"Add {x}")
    return P(f"**Reorder.** The last three commits are `Add {w[0]}`, `Add {w[1]}`, `Add {w[2]}` (oldest first). "
             f"Reorder them so `Add {w[2]}` comes first, then `Add {w[0]}`, then `Add {w[1]}`. {EDIT}",
             setup(GIT, "#: cwd ~/proj", r.code()),
             "GIT_SEQUENCE_EDITOR=\"sed -i '1{h;d};2{H;d};3{p;x}'\" git rebase -i HEAD~3",
             check=(ST, "."), require=["git rebase -i"])

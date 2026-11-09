"""8.06 Merging & resolving conflicts."""

from ._common import BRANCHES, GH, GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.06"
ST = ["head", "branches", "log", "index", "worktree", "state"]


def _diverged(rng, conflict=False):
    """main and a feature branch that both moved on; with conflict=True both change config.py's lr line."""
    b = rng.choice(BRANCHES)
    lr0, lr1, lr2 = rng.sample(["1e-2", "1e-3", "3e-4", "5e-4", "2e-4"], 3)
    base = f"model = 'resnet18'\nlr = {lr0}\nbatch_size = 32\nepochs = 10\n"
    r = Repo(rng).commit("Initial commit", {**ml_files(rng, 2), "config.py": base})
    r.branch(b)
    r.write(f"{b.split('/')[1]}.py", f"# {b}\n").commit(f"Add {b.split('/')[1]}")
    if conflict:
        r.write("config.py", base.replace(f"lr = {lr0}", f"lr = {lr2}")).commit(f"Use lr {lr2}")
    r.switch("main")
    r.append("README.md", "Usage notes.").commit("Document usage")
    if conflict:
        r.write("config.py", base.replace(f"lr = {lr0}", f"lr = {lr1}")).commit(f"Use lr {lr1}")
    return r, b, (lr0, lr1, lr2), base


def _ff(rng):
    b = rng.choice(BRANCHES)
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.branch(b)
    for i in range(2):
        r.write(f"{b.split('/')[1]}_{i}.py", f"# {i}\n").commit(f"{b}: part {i + 1}")
    r.switch("main")
    return r, b


# ================================================================= lite

@template(M, "lite")
def merge_basics(rng):
    r, b = _ff(rng)
    items = [
        ("git merge", f"Merge `{b}` into main (it can fast-forward).", f"git merge {b}"),
        ("git merge --no-ff", f"Merge `{b}` into main with a merge commit, even though it could fast-forward.",
         f"git merge --no-ff {b} -m 'Merge {b}'", {"require": ["git merge --no-ff"]}),
        ("git merge --squash", f"Bring all of `{b}`'s changes into main as **one** new commit `feat: {b}` (squash "
                               f"merge).", f"git merge --squash {b}\ngit commit -m 'feat: {b}'"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()), check=(ST, "."))


@template(M, "lite")
def abort(rng):
    r, b, lrs, base = _diverged(rng, conflict=True)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git merge {b} >/dev/null 2>&1 || true")
    return P(f"A merge of `{b}` stopped with a conflict in `config.py` and you'd rather not deal with it now. Back "
             f"out of the merge completely.", s, "git merge --abort", check=(ST, "."),
             require=["git merge --abort|git reset"])


# ================================================================= easy

@template(M, "easy")
def which_conflicts(rng):
    r, b, lrs, base = _diverged(rng, conflict=True)
    r.switch(b).write("README.md", "# conflicting readme\n").commit("Rewrite README").switch("main")
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git merge {b} >/dev/null 2>&1 || true")
    items = [
        ("git diff --diff-filter", "A merge stopped with conflicts. List just the names of the conflicted files.",
         "git diff --name-only --diff-filter=U", {"require": ["git diff --diff-filter|git diff --name-only"]}),
        ("git status", "A merge stopped with conflicts. Show the status, to see which files are unmerged.",
         "git status"),
        ("git diff", "A merge stopped with conflicts. Show the conflicts (the combined diff of what's unresolved).",
         "git diff"),
    ]
    return catalog(rng, items, s)


@template(M, "easy")
def take_side(rng):
    r, b, (lr0, lr1, lr2), base = _diverged(rng, conflict=True)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git merge {b} >/dev/null 2>&1 || true")
    items = [
        ("git checkout --ours", f"The merge of `{b}` conflicts in `config.py`. Keep **main's** version of the file "
                                f"(lr {lr1}) and finish the merge with git's default message.",
         "git checkout --ours config.py\ngit add config.py\ngit commit --no-edit"),
        ("git checkout --theirs", f"The merge of `{b}` conflicts in `config.py`. Take **{b}'s** version (lr {lr2}) "
                                  f"and finish the merge with git's default message.",
         "git checkout --theirs config.py\ngit add config.py\ngit commit --no-edit"),
    ]
    return catalog(rng, items, s, check=(ST + ["files"], "."))


# ================================================================= medium

@template(M, "medium")
def resolve_by_hand(rng):
    r, b, (lr0, lr1, lr2), base = _diverged(rng, conflict=True)
    final = base.replace(f"lr = {lr0}", f"lr = {lr1}  # warmup to {lr2}")
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git merge {b} >/dev/null 2>&1 || true")
    return P(f"Merging `{b}` into main conflicted on the `lr` line of `config.py`. Resolve it so the file reads "
             f"exactly:\n\n```\n{final}```\n\nthen finish the merge (keep git's merge message).",
             s, heredoc("config.py", final) + "\ngit add config.py\ngit commit --no-edit",
             check=(ST + ["files"], "."), require=["git add"])


@template(M, "medium")
def strategy_option(rng):
    r, b, (lr0, lr1, lr2), base = _diverged(rng, conflict=True)
    side, lr = rng.choice([("theirs", lr2), ("ours", lr1)])
    who = b if side == "theirs" else "main"
    return P(f"Merge `{b}` into main, letting every conflict be decided in favour of **{who}** automatically "
             f"(non-conflicting changes from both sides still merge). The result should use lr {lr}.",
             setup(GIT, "#: cwd ~/proj", r.code()), f"git merge -X {side} {b} -m 'Merge {b}'",
             check=(ST + ["files"], "."), require=["git merge -X"])


# ================================================================= hard

@template(M, "hard")
def multi_file(rng):
    b = rng.choice(BRANCHES)
    r = Repo(rng).commit("Initial commit", {"a.txt": "alpha\n", "b.txt": "beta\n", "c.txt": "gamma\n"})
    r.branch(b).commit("Branch edits", {"a.txt": "alpha (branch)\n", "b.txt": "beta (branch)\n",
                                        "c.txt": "gamma (branch)\n"})
    r.switch("main").commit("Main edits", {"a.txt": "alpha (main)\n", "b.txt": "beta (main)\n",
                                           "c.txt": "gamma (main)\n"})
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"git merge {b} >/dev/null 2>&1 || true")
    return P(f"Merging `{b}` conflicted in three files. Keep main's `a.txt`, the branch's `b.txt`, and make "
             f"`c.txt` contain the single line `gamma (both)`. Then conclude the merge with the message `Merge "
             f"{b}`.", s, f"git checkout --ours a.txt\ngit checkout --theirs b.txt\necho 'gamma (both)' > c.txt\n"
                          f"git add a.txt b.txt c.txt\ngit commit -m 'Merge {b}'",
             check=(ST + ["files"], "."))


@template(M, "hard")
def squash_and_delete(rng):
    r, b = _ff(rng)
    r.append("README.md", "main moved").commit("Main moves on")
    return P(f"Land `{b}` on main as a single commit `feat: {b.split('/')[1]}` (squash merge), then delete the "
             f"branch, which git now considers unmerged.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git merge --squash {b}\ngit commit -m 'feat: {b.split('/')[1]}'\ngit branch -D {b}",
             check=(ST + ["files"], "."), require=["git merge --squash"])


# ================================================================= challenge

@template(M, "challenge")
def recipe_book(rng):
    final = ("# Chili\n\nServes 6.\n\n## Ingredients\n\n- 400 g lentils\n- 1 onion\n- 2 cloves garlic\n"
             "- 1 tin kidney beans\n- 1 tin tomatoes\n- 1 tbsp chili powder\n- 2 fresh jalapenos\n- 1 tsp cumin\n\n"
             "## Method\n\n1. Rinse the lentils.\n2. Add onion and garlic, cook 5 minutes.\n"
             "3. Add everything else and simmer for 45 minutes.\n")
    first = rng.choice(["spicy", "vegan"])
    second = "vegan" if first == "spicy" else "spicy"
    fix = heredoc("recipes/chili.md", final) + "\ngit add recipes/chili.md\ngit commit --no-edit"
    # spicy conflicts with main; vegan doesn't (it changes other lines), so exactly one of the merges stops
    sol = (f"git merge origin/spicy\n{fix}\ngit merge --no-edit origin/vegan" if first == "spicy" else
           f"git merge --no-edit origin/vegan\ngit merge origin/spicy\n{fix}")
    return P(f"**Three cooks, one chili.** In `git-gym/recipe-book`, merge `origin/{first}` and then "
             f"`origin/{second}` into main. Wherever they conflict, `recipes/chili.md` must end up exactly as:\n\n"
             f"```\n{final}```\n\nKeep git's default merge messages (`--no-edit` skips the editor).",
             setup(GIT, GH, "#: cwd ~/recipe-book", "git clone -q git@github.com:git-gym/recipe-book.git ."),
             sol, check=(["head", "log", "files", "state", "index", "worktree"], "."))


@template(M, "challenge")
def sync_feature(rng):
    r, b, (lr0, lr1, lr2), base = _diverged(rng, conflict=True)
    final = base.replace(f"lr = {lr0}", f"lr = {lr2}")
    return P(f"**Keep the branch fresh.** Merge main into `{b}` (resolving `config.py` in favour of the branch: lr "
             f"{lr2}, finishing with git's default message), then fast-forward main to `{b}`. End on main.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git switch {b}\ngit merge -X ours main --no-edit\ngit switch main\ngit merge {b}",
             check=(["head", "branches", "files", "state"], "."), require=["git merge"])

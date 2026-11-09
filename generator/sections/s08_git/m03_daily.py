"""8.03 The daily loop: status, add, commit."""

from ._common import GIT, P, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.03"
STATE = ["index", "worktree", "untracked"]


def _base(rng, n=3):
    fs = ml_files(rng, n)
    return Repo(rng).commit("Initial commit", fs), sorted(fs)


# ================================================================= lite

@template(M, "lite")
def stage(rng):
    r, names = _base(rng)
    a, b, c = names
    new = rng.choice(["notes.md", "TODO.md", "plot.py", "LICENSE"])
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# changed' >> {a}", f"echo '# changed' >> {b}",
              f"echo new > {new}", "mkdir -p src && echo 'x = 1' > src/extra.py")
    items = [
        ("git add", f"Stage only `{a}`.", f"git add {a}"),
        ("git add", "Stage the `src/` directory (and nothing else).", "git add src/"),
        ("git add", "Stage every change, including new files.", "git add ."),
        ("git add -u", "Stage the changes to files git already tracks, but none of the new files.", "git add -u"),
        ("git add -A", "Stage everything, new files included, from the whole repository.", "git add -A"),
        ("git restore --staged", f"Stage everything, then take `{new}` back out of the staging area (keep the "
                                 f"file).", f"git add .\ngit restore --staged {new}"),
        ("git rm", f"Delete `{c}` and stage the deletion with one command.", f"git rm {c}"),
        ("git mv", f"Rename `{c}` to `old_{c}` the way git sees as a rename (one command).", f"git mv {c} old_{c}"),
    ]
    return catalog(rng, items, s, check=(STATE, "."))


@template(M, "lite")
def commit(rng):
    r, names = _base(rng)
    a, b = names[:2]
    w = rng.choice(WORDS)
    body = rng.choice(["Padding tokens were attended to when a batch mixed lengths.",
                       "The old default diverged after 3k steps.", "Keeps memory flat for long runs."])
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# {w}' >> {a}", f"git add {a}", f"echo '# more' >> {b}")
    items = [
        ("git commit -m", f"Commit what's staged with the message `feat: add {w}`.", f"git commit -m 'feat: add {w}'"),
        ("git commit -m", f"Commit what's staged with the subject `fix: {w}` and a body paragraph `{body}` (two "
                          f"`-m`).", f"git commit -m 'fix: {w}' -m {q(body)}"),
        ("git commit -a", f"Commit **all** changes to tracked files in one step with the message `chore: {w}`.",
         f"git commit -am 'chore: {w}'", {"require": ["git commit -a|git commit -am"]}),
        ("git commit --amend", f"You forgot `{b}`: stage it and add it to the **last** commit without changing "
                               f"its message. (First commit what's staged as `wip {w}`.)",
         f"git commit -m 'wip {w}'\ngit add {b}\ngit commit --amend --no-edit"),
        ("git commit --amend", f"Commit what's staged as `Add {w} modle`, then fix the typo in that message "
                               f"(`modle` -> `module`).",
         f"git commit -m 'Add {w} modle'\ngit commit --amend -m 'Add {w} module'"),
    ]
    return catalog(rng, items, s, check=(["log"] + STATE, "."))


# ================================================================= easy

@template(M, "easy")
def read_status(rng):
    r, names = _base(rng)
    a, b, c = names
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo x >> {a}", f"git add {a}", f"echo y >> {b}", f"rm {c}",
              "echo scratch > scratch.txt")
    items = [
        ("git status", "Show the repository's status.", "git status"),
        ("git status -s", "Show the status in the short format (two letters per file).", "git status -s",
         {"require": ["git status -s|git status --short"]}),
        ("git diff", "Show the changes you haven't staged yet.", "git diff"),
        ("git diff --staged", "Show what would go into the next commit.", "git diff --staged",
         {"require": ["git diff --staged|git diff --cached"]}),
        ("git diff --name-only", "Print only the names of the files with unstaged changes.",
         "git diff --name-only"),
    ]
    return catalog(rng, items, s)


@template(M, "easy")
def two_commits(rng):
    r, names = _base(rng)
    a, b = names[:2]
    wa, wb = rng.sample(WORDS, 2)
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# {wa}' >> {a}", f"echo '# {wb}' >> {b}")
    return P(f"Two unrelated edits are waiting. Commit `{a}` as `Update {wa}` and then `{b}` as `Update {wb}`: two "
             f"commits, in that order.", s,
             f"git add {a}\ngit commit -m 'Update {wa}'\ngit add {b}\ngit commit -m 'Update {wb}'",
             check=(["log"] + STATE, "."), require=["git add", "git commit"])


# ================================================================= medium

@template(M, "medium")
def add_patch(rng):
    top = "".join(f"line {i}\n" for i in range(1, 21))
    w1, w2 = rng.sample(WORDS, 2)
    first = rng.choice([True, False])
    r = Repo(rng).commit("Initial commit", {"train.py": top})
    edited = top.replace("line 2\n", f"line 2 {w1}\n").replace("line 18\n", f"line 18 {w2}\n")
    s = setup(GIT, "#: cwd ~/proj", r.code(), heredoc("train.py", edited))
    keep, msg = (w1, f"Tweak {w1}") if first else (w2, f"Tweak {w2}")
    answers = "y\nn" if first else "n\ny"
    return P(f"`train.py` has two separate changes (`{w1}` near the top, `{w2}` near the bottom). Commit **only** "
             f"the `{keep}` change, as `{msg}`, using `git add -p` to stage hunk by hunk.",
             s, f"git add -p train.py\n{answers}\ngit commit -m {q(msg)}",
             check=(["log", "index", "worktree"], "."), require=["git add -p"])


@template(M, "medium")
def fix_last_commit(rng):
    r, names = _base(rng)
    a, b = names[:2]
    w = rng.choice(WORDS)
    r.append(a, f"# {w}").commit(f"Add {w} suport")
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# {w} too' >> {b}")
    return P(f"The last commit, `Add {w} suport`, has a typo and forgot the change in `{b}`. Fix both in that same "
             f"commit (message `Add {w} support`), without making a new commit.",
             s, f"git add {b}\ngit commit --amend -m 'Add {w} support'",
             check=(["log"] + STATE, "."), require=["git commit --amend"])


# ================================================================= hard

@template(M, "hard")
def rm_mv(rng):
    r, names = _base(rng)
    a, b, c = names
    new = rng.choice(["helpers", "common", "tools", "lib"]) + "." + a.rsplit(".", 1)[1]
    s = setup(GIT, "#: cwd ~/proj", r.code())
    return P(f"Rename `{a}` to `{new}`, delete `{c}`, and commit both as `refactor: reorganise files`. Use git's "
             f"own commands, so the rename and the deletion are staged as you make them.",
             s, f"git mv {a} {new}\ngit rm {c}\ngit commit -m 'refactor: reorganise files'",
             check=(["files", "log"] + STATE, "."), require=["git mv", "git rm"])


@template(M, "hard")
def unstage_secrets(rng):
    r, names = _base(rng)
    a = names[0]
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# fix' >> {a}", "echo 'API_KEY=fake-123' > .env",
              "head -c 3000 /dev/zero > dump.bin", "git add -A")
    return P(f"`git add -A` staged a `.env` file with a key and a big `dump.bin`. Take both back out of the staging "
             f"area (keep the files on disk), then commit just the change to `{a}` as `Fix {a}`.",
             s, f"git restore --staged .env dump.bin\ngit commit -m 'Fix {a}'",
             check=(["log"] + STATE, "."), require=["git restore --staged|git reset|git rm --cached"])


# ================================================================= challenge

@template(M, "challenge")
def clean_commits(rng):
    fs = {"README.md": "# Proejct\n\nTrains the model.\n", "model.py": "class Net:\n    pass\n",
          "train.py": "def main():\n    return 0\n"}
    r = Repo(rng).commit("Initial commit", fs)
    w = rng.choice(WORDS)
    s = setup(GIT, "#: cwd ~/proj", r.code(), heredoc("README.md", "# Project\n\nTrains the model.\n"),
              heredoc("model.py", f"class Net:\n    pass\n\n\ndef {w}():\n    return 1\n"),
              heredoc("train.py", "def main():\n    print('DEBUG here')\n    return 0\n"))
    return P(f"**Tidy up before lunch.** Three files changed: `README.md` fixes a typo, `model.py` adds a `{w}` "
             f"function, and `train.py` only has a leftover debug print. Make two commits, `docs: fix typo` "
             f"(README) and then `feat: add {w}` (model.py), and throw the debug print away.",
             s, f"git add README.md\ngit commit -m 'docs: fix typo'\ngit add model.py\ngit commit -m 'feat: add {w}'\n"
                "git restore train.py",
             check=(["log", "files"] + STATE, "."))


@template(M, "challenge")
def split_hunks(rng):
    top = "".join(f"step {i}\n" for i in range(1, 25))
    w1, w2, w3 = rng.sample(WORDS, 3)
    r = Repo(rng).commit("Initial commit", {"pipeline.py": top})
    edited = top.replace("step 1\n", f"step 1 {w1}\n").replace("step 12\n", f"step 12 {w2}\n") \
        .replace("step 23\n", f"step 23 {w3}\n")
    s = setup(GIT, "#: cwd ~/proj", r.code(), heredoc("pipeline.py", edited))
    return P(f"**One file, three commits.** `pipeline.py` has three separate changes (`{w1}`, `{w2}`, `{w3}`, top to "
             f"bottom). Commit them as three commits, `Add {w1}`, `Add {w2}`, `Add {w3}`, in that order.",
             s, f"git add -p\ny\nn\nn\ngit commit -m 'Add {w1}'\ngit add -p\ny\nn\ngit commit -m 'Add {w2}'\n"
                f"git commit -am 'Add {w3}'",
             check=(["log"] + STATE, "."))

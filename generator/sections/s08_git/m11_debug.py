"""8.11 Debugging & search."""

from ._common import GH, GIT, P, PEOPLE, Repo, catalog, heredoc, ml_files, q, setup, template, WORDS

M = "8.11"
CALC = "#: cwd ~/calculator\ngit clone -q git@github.com:git-gym/calculator.git ."


def _regression(rng, n=24):
    """n commits; commit k sets retries = 0 (the bug). check.sh exits 1 on a bad commit."""
    k = rng.randint(n // 3, 2 * n // 3)
    r = Repo(rng).commit("Initial commit", {**ml_files(rng, 2), "settings.py": "retries = 3\ntimeout = 30\n",
                                            "check.sh": "#!/bin/sh\n! grep -q 'retries = 0' settings.py\n"})
    r.run("chmod +x check.sh && git commit -qam 'Make check.sh executable'")
    r.tag("v1.0", "Known good")
    for i in range(1, n):
        if i == k:
            r.write("settings.py", "retries = 0\ntimeout = 30\n").commit("Simplify settings")
        else:
            p = rng.choice(["README.md", "notes.md"])
            r.append(p, f"{rng.choice(WORDS)} {i}").commit(rng.choice(["Docs", "Update notes", "Tidy", "Typo"]))
    return r


# ================================================================= lite

@template(M, "lite")
def grep(rng):
    r = Repo(rng).commit("Initial commit", {
        "train.py": "learning_rate = 1e-3  # TODO tune\nbatch_size = 32\n",
        "model.py": "# TODO: add dropout\nclass Net:\n    learning_rate = None\n",
        "README.md": "Set learning_rate in train.py. TODO: docs\n", "data.py": "def load():\n    pass\n"})
    items = [
        ("git grep", "Find every line in the tracked files that contains `TODO`.", "git grep TODO"),
        ("git grep -n", "Find `learning_rate` in the Python files only, with line numbers.",
         "git grep -n learning_rate -- '*.py'"),
        ("git grep -l", "List just the names of the files that mention `TODO`.", "git grep -l TODO"),
        ("git grep -c", "Count the `learning_rate` lines in each file.", "git grep -c learning_rate"),
        ("git grep -i", "Find `todo` in any capitalisation.", "git grep -i todo"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "lite")
def rev_parse(rng):
    r = Repo(rng).commit("Initial commit", {**ml_files(rng, 2), "src/models/net.py": "x = 1\n"})
    r.append("README.md", "x").commit("Second")
    s = setup(GIT, "#: cwd ~/proj", r.code(), "cd src/models")
    items = [
        ("git rev-parse", "Print the full hash of HEAD.", "git rev-parse HEAD"),
        ("git rev-parse --short", "Print the short hash of HEAD.", "git rev-parse --short HEAD"),
        ("git rev-parse --show-toplevel", "You're deep inside the repository. Print the path of its top "
                                          "directory.", "git rev-parse --show-toplevel"),
        ("git rev-parse --abbrev-ref", "Print the name of the current branch with `rev-parse`.",
         "git rev-parse --abbrev-ref HEAD"),
        ("git rev-parse --show-prefix", "Print where you are relative to the repository's top.",
         "git rev-parse --show-prefix"),
    ]
    return catalog(rng, items, s)


# ================================================================= easy

@template(M, "easy")
def bisect_start(rng):
    items = [
        ("git bisect start", "`div(7, 2)` is broken now but worked at tag `v1.0`. Start a bisect, marking HEAD "
                             "bad and `v1.0` good.", "git bisect start\ngit bisect bad\ngit bisect good v1.0",
         {"post": "git bisect log | grep -v '^#'; /usr/lib/bashlite/gitstate state"}),
        ("git bisect start", "Start a bisect between bad HEAD and good `v1.0` with a single command.",
         "git bisect start HEAD v1.0", {"post": "git bisect log | grep -v '^#'"}),
    ]
    return catalog(rng, items, setup(GIT, GH, CALC), answer=["<post>"])


@template(M, "easy")
def bisect_run(rng):
    return P("`div(7, 2)` returns 3 now but 3.5 at tag `v1.0`; `python3 test_calc.py` exits non-zero when it's "
             "wrong. Let `git bisect run` find the first bad commit, save its full hash in `~/first-bad.txt`, then "
             "end the bisect.", setup(GIT, GH, CALC),
             "git bisect start HEAD v1.0\ngit bisect run python3 test_calc.py\n"
             "git rev-parse refs/bisect/bad > ~/first-bad.txt\ngit bisect reset",
             post="git log -1 --format=%s $(cat ~/first-bad.txt); /usr/lib/bashlite/gitstate head state",
             answer=["<post>"], require=["git bisect run"])


# ================================================================= medium

@template(M, "medium")
def function_history(rng):
    fn = rng.choice(["forward", "normalize", "collate", "warmup"])
    r = Repo(rng).commit("Initial commit", {"model.py": f"import math\n\n\ndef helper():\n    return 0\n\n\ndef {fn}(x):\n"
                                                        f"    return x\n\n\ndef other():\n    return 1\n"})
    body = "    return x"
    for i, change in enumerate(["    return x * 2", "    y = x * 2\n    return y + 1", "    y = x * 2\n    return y"]):
        r.write("model.py", r.files["model.py"].replace(body, change)).commit(f"Change {fn} ({i + 1})")
        body = change
    r.write("model.py", r.files["model.py"].replace("return 1", "return 2")).commit("Change other")
    items = [
        ("git log -L", f"Show the history of just the `{fn}` function in `model.py`.", f"git log -L :{fn}:model.py"),
        ("git log -L", f"Show the history of lines 8 to 10 of `model.py`.", "git log -L 8,10:model.py"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "medium")
def contains(rng):
    items = [
        ("git tag --contains", "Which tags contain the commit `Add put()`? (`--contains`; find the commit with "
                               "`:/`.)", "git tag --contains ':/Add put()'"),
        ("git branch --contains", "Which remote-tracking branches contain the commit `Add a Retry adapter`?",
         "git branch -r --contains ':/Add a Retry adapter'", {"require": ["git branch --contains"]}),
        ("git describe", "Describe the commit `Add type hints to get()` relative to the nearest earlier tag.",
         "git describe ':/Add type hints to get'"),
        ("git grep", "Search `requests_mini/api.py` **as it was at tag v0.2.0** for `def `, with line numbers.",
         "git grep -n 'def ' v0.2.0 -- requests_mini/api.py"),
    ]
    return catalog(rng, items, setup(GIT, GH, "#: cwd ~/requests-mini",
                                     "git clone -q git@github.com:minilibs/requests-mini.git ."))


# ================================================================= hard

@template(M, "hard")
def bisect_script(rng):
    r = _regression(rng)
    return P("Requests started failing because a commit set `retries = 0` in `settings.py`; `./check.sh` exits 1 "
             "on such a commit and tag `v1.0` is good. Find the first bad commit with `git bisect run`, put a "
             "branch `first-bad` on it, and end the bisect.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             "git bisect start HEAD v1.0\ngit bisect run ./check.sh\ngit branch first-bad refs/bisect/bad\n"
             "git bisect reset", check=(["head", "branches", "state"], "."), require=["git bisect"])


@template(M, "hard")
def objects(rng):
    items = [
        ("git count-objects", "Show how many objects this clone has and how much space they take (verbose, "
                              "human-readable sizes).", "git count-objects -vH"),
        ("git fsck", "Check the repository's objects for corruption.", "git fsck"),
        ("git cat-file", "Print the type of the object `HEAD^{tree}` (with `cat-file`).", "git cat-file -t "
                                                                                             "'HEAD^{tree}'"),
        ("git ls-tree", "List the files in HEAD's top directory (`ls-tree`).", "git ls-tree HEAD"),
    ]
    return catalog(rng, items, setup(GIT, GH, "#: cwd ~/leftpad", "git clone -q git@github.com:minilibs/leftpad.git ."))


# ================================================================= challenge

@template(M, "challenge")
def find_and_fix(rng):
    return P("**Find it, fix it.** In `git-gym/calculator`, find the commit that broke `div` with `git bisect run "
             "python3 test_calc.py` (v1.0 is good), end the bisect, then undo that commit with `git revert` (git's "
             "message) so the tests pass again.", setup(GIT, GH, CALC),
             "git bisect start HEAD v1.0\ngit bisect run python3 test_calc.py\nbad=$(git rev-parse refs/bisect/bad)\n"
             "git bisect reset\ngit revert --no-edit $bad",
             post="python3 test_calc.py; /usr/lib/bashlite/gitstate head log state files", answer=["<post>"],
             require=["git bisect run", "git revert"])


@template(M, "challenge")
def who_and_when(rng):
    authors = rng.sample(PEOPLE, 3)
    lines = [f"{w} = {i}" for i, w in enumerate(rng.sample(WORDS, 6))]
    r = Repo(rng).commit("Initial commit", {"params.py": "\n".join(lines) + "\n"}, author=authors[0])
    k = rng.randint(1, 4)
    name = lines[k].split(" = ")[0]
    lines[k] = f"{name} = -1"
    r.commit("Small cleanup", {"params.py": "\n".join(lines) + "\n"}, author=authors[1])
    lines.append("seed = 0")
    r.commit("Fix seed", {"params.py": "\n".join(lines) + "\n"}, author=authors[2])
    return P(f"**Who did it?** `{name}` became `-1`. Print the short hash, author and subject (`%h %an %s`) of "
             f"the commit that introduced that line.",
             setup(GIT, "#: cwd ~/proj", r.code()),
             f"git log -S '{name} = -1' --format='%h %an %s' -- params.py",
             require=["git log -S|git blame|git log -L|git log -G"])

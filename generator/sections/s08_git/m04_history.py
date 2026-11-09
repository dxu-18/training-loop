"""8.04 History & diffs."""

from ._common import GH, GIT, PEOPLE, P, Repo, catalog, history, ml_files, q, setup, template, WORDS

M = "8.04"


def _hist(rng, n=8):
    authors = rng.sample(PEOPLE, 3)
    r = history(rng, n, authors=authors)
    return r, authors


def _clone(full, d=None):
    """Setup lines: start in ~/<name>, a clone of a catalog repository."""
    name = d or full.split("/")[1]
    return f"#: cwd ~/{name}\ngit clone -q git@github.com:{full}.git ."


# ================================================================= lite

@template(M, "lite")
def log_views(rng):
    r, authors = _hist(rng, rng.randint(6, 10))
    who = authors[0][0].split()[0]
    w = rng.choice(["Fix", "Add", "Refactor", "Document"])
    n = rng.randint(2, 4)
    items = [
        ("git log --oneline", "Show the history, one line per commit.", "git log --oneline"),
        ("git log -n", f"Show only the last {n} commits, one line each.", f"git log --oneline -n {n}",
         {"require": ["git log -n|git log -" + str(n)]}),
        ("git log --stat", "Show the last commit with the files it changed and how much (`--stat`).",
         "git log --stat -1"),
        ("git log --author", f"Show the commits by {who}, one line each.", f"git log --oneline --author={who}"),
        ("git log --grep", f"Show the commits whose message mentions `{w}`, one line each.",
         f"git log --oneline --grep={w}"),
        ("git log --pretty", "Print every commit as its short hash, author name and subject (`%h %an %s`).",
         "git log --pretty=format:'%h %an %s'", {"require": ["git log --pretty|git log --format"]}),
        ("git show", "Show the most recent commit and its diff.", "git show"),
        ("git show --stat", "Show which files the commit before the last one changed (`--stat`, nothing else).",
         "git show --stat HEAD~1"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "lite")
def diffs(rng):
    fs = ml_files(rng, 3)
    r = Repo(rng).commit("Initial commit", fs)
    names = sorted(fs)
    for i in range(3):
        r.append(names[i % 3], f"# {rng.choice(WORDS)}").commit(f"Update {names[i % 3]}")
    a, b = names[:2]
    s = setup(GIT, "#: cwd ~/proj", r.code(), f"echo '# staged' >> {a}", f"git add {a}", f"echo '# wip' >> {b}")
    items = [
        ("git diff", "Show the changes in the working tree that are not staged.", "git diff"),
        ("git diff --staged", "Show the staged changes.", "git diff --staged",
         {"require": ["git diff --staged|git diff --cached"]}),
        ("git diff", "Show every change since the last commit, staged or not, in one diff.", "git diff HEAD"),
        ("git diff --stat", "Summarise the unstaged changes per file (`--stat`).", "git diff --stat"),
        ("git diff --name-only", "List the files that changed in the last 2 commits (names only).",
         "git diff --name-only HEAD~2", {"require": ["git diff --name-only"]}),
        ("git diff --word-diff", "Show the unstaged changes word by word.", "git diff --word-diff"),
    ]
    return catalog(rng, items, s)


# ================================================================= easy

@template(M, "easy")
def file_history(rng):
    items = [
        ("git log --follow", "In `requests-mini`, `requests_mini/compat.py` was renamed to "
                             "`requests_mini/_compat.py`. Show the one-line history of `_compat.py` **across** the "
                             "rename.", "git log --oneline --follow -- requests_mini/_compat.py"),
        ("git log -p", "In `requests-mini`, show every commit that touched `requests_mini/models.py`, with its "
                       "diff.", "git log -p -- requests_mini/models.py", {"require": ["git log -p"]}),
        ("git log", "In `requests-mini`, list (one line each) the commits that changed `README.md`.",
         "git log --oneline -- README.md"),
        ("git log --stat", "In `requests-mini`, show the history of `requests_mini/api.py` with `--stat`.",
         "git log --stat -- requests_mini/api.py"),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("minilibs/requests-mini")))


@template(M, "easy")
def show_things(rng):
    tag = rng.choice(["v0.1.0", "v0.2.0", "v1.0.0"])
    items = [
        ("git show", f"Show the tag `{tag}` (its message and the commit it points to).", f"git show {tag}"),
        ("git show", "Print the contents of `requests_mini/api.py` as it was at tag `v0.1.0` (without checking it "
                     "out).", "git show v0.1.0:requests_mini/api.py"),
        ("git show", "Show the commit two before HEAD, with its diff.", "git show HEAD~2"),
        ("git show --stat", f"Show which files the tagged commit `{tag}` changed (`--stat`).",
         f"git show --stat {tag}"),
        ("git show", "Show the newest commit whose message mentions `timeout`, using the `:/` syntax.",
         "git show ':/timeout'"),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("minilibs/requests-mini")))


# ================================================================= medium

@template(M, "medium")
def pickaxe(rng):
    w = rng.choice(["lr_scheduler", "grad_clip", "warmup_steps", "label_smoothing", "ema_decay"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    r.append("train.py", "# setup").commit("Set up training")
    r.append("train.py", f"{w} = None").commit(rng.choice(["Tune things", "Small cleanup", "Misc"]))
    r.append("model.py", "# layers").commit("Add layers")
    r.append("README.md", f"Mentions {w} in the docs.").commit("Docs")
    items = [
        ("git log -S", f"Find the commits that added or removed the string `{w}` (one line each).",
         f"git log --oneline -S {w}"),
        ("git log -S", f"Show the commit that **introduced** `{w}` into `train.py`, with its diff.",
         f"git log -p -S {w} -- train.py", {"require": ["git log -S"]}),
        ("git grep", f"List every line in the current files that mentions `{w}`, with line numbers.",
         f"git grep -n {w}"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "medium")
def ranges(rng):
    br = rng.choice(["feature/kv-cache", "feature/dropout", "fix/nan-loss"])
    r = Repo(rng).commit("Initial commit", ml_files(rng, 3))
    r.append("README.md", "base").commit("Base work")
    r.branch(br)
    for i in range(rng.randint(2, 3)):
        r.append("model.py", f"# {br} {i}").commit(f"{br.split('/')[1]}: step {i + 1}")
    r.switch("main")
    for i in range(2):
        r.append("README.md", f"main {i}").commit(f"main: docs {i + 1}")
    items = [
        ("git log main..", f"List the commits on `{br}` that are not on `main` (one line each).",
         f"git log --oneline main..{br}", {"require": ["git log"]}),
        ("git log", f"List the commits on `main` that `{br}` doesn't have yet.", f"git log --oneline {br}..main"),
        ("git diff ...", f"Show what `{br}` changed since it branched off `main` (three dots).",
         f"git diff main...{br}", {"require": ["git diff"]}),
        ("git log --graph", "Show every branch's history as a graph, one line per commit with branch names.",
         "git log --oneline --graph --decorate --all", {"require": ["git log --graph"]}),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


# ================================================================= hard

@template(M, "hard")
def blame(rng):
    authors = rng.sample(PEOPLE, 3)
    lines = [f"{w} = {i}" for i, w in enumerate(rng.sample(WORDS, 8))]
    r = Repo(rng).commit("Initial commit", {"config.py": "\n".join(lines) + "\n"}, author=authors[0])
    k = rng.randint(2, 6)
    lines[k] = lines[k].split(" = ")[0] + " = 99"
    r.commit("Raise a limit", {"config.py": "\n".join(lines) + "\n"}, author=authors[1])
    lines.append("debug = True")
    r.commit("Turn on debugging", {"config.py": "\n".join(lines) + "\n"}, author=authors[2])
    a, b = sorted(rng.sample(range(1, 10), 2))
    items = [
        ("git blame", "Show who last changed each line of `config.py`.", "git blame config.py"),
        ("git blame -L", f"Show who last changed lines {a} to {b} of `config.py`.", f"git blame -L {a},{b} config.py"),
        ("git blame -L", f"Find who set `{lines[k].split(' = ')[0]}` to 99: blame just that line ({k + 1}).",
         f"git blame -L {k + 1},{k + 1} config.py"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))


@template(M, "hard")
def shortlog(rng):
    items = [
        ("git shortlog -sn", "In `tiny-kernel`, rank the authors by number of commits (`-sn`).",
         "git shortlog -sn", {"require": ["git shortlog"]}),
        ("git shortlog -sn", "In `tiny-kernel`, rank the authors by commits, leaving merge commits out.",
         "git shortlog -sn --no-merges", {"require": ["git shortlog", "git shortlog --no-merges"]}),
        ("git log --author", "In `tiny-kernel`, count the commits Aiko Tanaka authored (print just the number).",
         "git log --oneline --author='Aiko Tanaka' | wc -l"),
        ("git log --grep", "In `tiny-kernel`, count the commits whose message mentions `NULL` (just the number).",
         "git log --oneline --grep=NULL | wc -l"),
        ("git log --merges", "In `tiny-kernel`, list the merge commits' subjects (one per line, `%s`).",
         "git log --merges --format=%s"),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("kernel-lab/tiny-kernel")))


# ================================================================= challenge

@template(M, "challenge")
def investigate(rng):
    items = [
        ("git log", "**Detective work.** In `requests-mini`, print the subject of the commit that renamed "
                    "`compat.py` (hint: `--diff-filter=R`).",
         "git log --diff-filter=R --format=%s -- requests_mini/_compat.py"),
        ("git log", "**Detective work.** In `requests-mini`, print the short hash and author of the commit that "
                    "first added `def put` to `requests_mini/api.py` (`%h %an`).",
         "git log -S 'def put' --format='%h %an' -- requests_mini/api.py"),
        ("git diff", "**Detective work.** In `requests-mini`, list the files that changed between tags `v1.0.0` and "
                     "`v1.1.0` (names only).", "git diff --name-only v1.0.0 v1.1.0"),
        ("git log", "**Detective work.** In `requests-mini`, print the subjects of the commits between `v0.2.0` and "
                    "`v1.0.0` (not including v0.2.0), oldest first.",
         "git log --reverse --format=%s v0.2.0..v1.0.0"),
    ]
    return catalog(rng, items, setup(GIT, GH, _clone("minilibs/requests-mini")))


@template(M, "challenge")
def revisions(rng):
    r = Repo(rng).commit("Initial commit", ml_files(rng, 2))
    r.append("README.md", "x").commit("Docs")
    r.branch("feature")
    r.write("feature.py", "x = 1").commit("Add feature")
    r.switch("main")
    r.write("main.py", "y = 2").commit("Work on main")
    r.merge("feature", "Merge feature")
    r.append("README.md", "after").commit("After the merge")
    items = [
        ("git rev-parse", "Print the full hash of the commit two before HEAD.", "git rev-parse HEAD~2"),
        ("git show", "The commit before HEAD is a merge. Show the subject of its **second** parent (`%s`).",
         "git show -s --format=%s HEAD~1^2"),
        ("git log", "Show the subjects of both parents of the merge commit, one per line (`%s`).",
         "git log --no-walk --format=%s HEAD~1^1 HEAD~1^2"),
        ("git rev-parse", "Print the short hash of the newest commit whose message contains `feature`.",
         "git rev-parse --short ':/feature'"),
    ]
    return catalog(rng, items, setup(GIT, "#: cwd ~/proj", r.code()))

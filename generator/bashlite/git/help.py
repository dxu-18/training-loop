"""Help for the simulated git: a short command overview (bash-lite's own wording), suggestions for mistyped
commands, and one-line usage hints."""

from __future__ import annotations

import difflib

# command -> (usage line, what it does): short descriptions written for this simulator
SUMMARY = {
    "init": ("git init [<dir>]", "make a new, empty repository"),
    "clone": ("git clone <url> [<dir>]", "copy a repository (and its history) from a remote"),
    "config": ("git config [--global] <key> [<value>]", "read or set configuration"),
    "status": ("git status [-s] [-b]", "show what changed: staged, not staged, untracked"),
    "add": ("git add [-p] [-A | -u] <pathspec>...", "stage changes for the next commit"),
    "rm": ("git rm [--cached] [-r] <pathspec>...", "remove files from the index (and the working tree)"),
    "mv": ("git mv <source> <destination>", "move or rename a tracked file"),
    "restore": ("git restore [--staged] [--source=<tree>] <pathspec>...", "undo changes in the working tree or index"),
    "commit": ("git commit [-a] [-m <msg>] [--amend]", "record the staged changes as a new commit"),
    "log": ("git log [--oneline] [--graph] [--all] [<revision range>]", "show the commit history"),
    "show": ("git show [<object>]", "show a commit (with its diff), a tag, a tree or a file at a revision"),
    "diff": ("git diff [--staged] [<commit> [<commit>]] [-- <path>]", "show changes between versions"),
    "blame": ("git blame [-L <range>] <file>", "show which commit last changed each line"),
    "branch": ("git branch [-a] [-vv] [-d] [<name>]", "list, create, rename or delete branches"),
    "switch": ("git switch [-c <new>] <branch>", "change to another branch"),
    "checkout": ("git checkout [-b <new>] <branch> | -- <path>", "switch branches or restore files"),
    "merge": ("git merge [--no-ff] [--squash] <branch>", "join another line of history into this one"),
    "rebase": ("git rebase [-i] [--onto <base>] <upstream>", "replay commits on top of another base"),
    "cherry-pick": ("git cherry-pick <commit>...", "apply the change from existing commits"),
    "revert": ("git revert <commit>", "make a new commit that undoes an earlier one"),
    "reset": ("git reset [--soft | --mixed | --hard] [<commit>]", "move the branch and reset the index (and files)"),
    "stash": ("git stash [push | pop | list | apply | drop]", "set changes aside and bring them back later"),
    "tag": ("git tag [-a] [-m <msg>] <name> [<commit>]", "name a commit (a release)"),
    "remote": ("git remote [-v] [add | remove | rename | set-url]", "manage the repositories you track"),
    "fetch": ("git fetch [--all] [--prune] [<remote>]", "download commits from a remote without merging"),
    "pull": ("git pull [--rebase] [<remote> [<branch>]]", "fetch and integrate remote changes"),
    "push": ("git push [-u] [<remote> [<branch>]]", "upload your commits to a remote"),
    "reflog": ("git reflog [<ref>]", "show where HEAD (or a branch) has been"),
    "bisect": ("git bisect start | good | bad | run <cmd> | reset", "binary-search for the commit that broke something"),
    "grep": ("git grep [-n] <pattern> [-- <path>]", "search tracked files"),
    "clean": ("git clean -n | -f [-d] [-x]", "delete untracked files"),
    "describe": ("git describe [--tags]", "name a commit relative to the nearest tag"),
    "shortlog": ("git shortlog [-sn]", "summarize history by author"),
    "worktree": ("git worktree add <path> <branch> | list | remove", "check out several branches at once"),
    "submodule": ("git submodule add <url> <path> | update --init", "embed another repository"),
    "rev-parse": ("git rev-parse [--short] <rev> | --show-toplevel", "turn names into ids, find the repository"),
    "archive": ("git archive --format=zip <tree-ish>", "export a snapshot of a tree"),
    "check-ignore": ("git check-ignore [-v] <path>", "show which ignore rule matches a path"),
    "sparse-checkout": ("git sparse-checkout set <dir>... | disable", "check out only some directories"),
    "gc": ("git gc", "pack objects and tidy the repository"),
    "fsck": ("git fsck", "check the object database"),
    "count-objects": ("git count-objects [-vH]", "show how much space objects use"),
}

GROUPS = [
    ("start", ["init", "clone", "config"]),
    ("everyday work", ["status", "add", "restore", "rm", "mv", "commit"]),
    ("history", ["log", "show", "diff", "blame", "reflog", "bisect", "grep"]),
    ("branches", ["branch", "switch", "merge", "rebase", "cherry-pick", "revert", "reset", "stash", "tag"]),
    ("collaborate", ["remote", "fetch", "pull", "push"]),
]


def usage_text() -> str:
    lines = ["usage: git [-v | --version] [-C <path>] [-c <name>=<value>] [--no-pager] <command> [<args>]", "",
             "Common commands (bash-lite's git):"]
    for title, cmds in GROUPS:
        lines.append("")
        lines.append(f"{title}:")
        for c in cmds:
            lines.append(f"   {c:<12} {SUMMARY[c][1]}")
    lines += ["", "'git <command> -h' shows a command's usage."]
    return "\n".join(lines) + "\n"


def _distance(a: str, b: str, w=0, s=2, add=1, d=3) -> int:
    """git's weighted Damerau-Levenshtein (help.c: swap 0, substitution 2, insertion 1, deletion 3)."""
    n, m = len(a), len(b)
    row2 = [0] * (m + 1)
    row1 = [j * add for j in range(m + 1)]
    row0 = [0] * (m + 1)
    for i in range(n):
        row2[0] = (i + 1) * d
        for j in range(m):
            row2[j + 1] = row1[j] + s * (a[i] != b[j])
            if i > 0 and j > 0 and a[i - 1] == b[j] and a[i] == b[j - 1] and row2[j + 1] > row0[j - 1] + w:
                row2[j + 1] = row0[j - 1] + w
            if row2[j + 1] > row1[j + 1] + d:
                row2[j + 1] = row1[j + 1] + d
            if row2[j + 1] > row2[j] + add:
                row2[j + 1] = row2[j] + add
        row0, row1, row2 = row1, row2, row0
    return row1[m]


def suggest(name: str) -> str:
    """git's "The most similar command is": prefix matches first, else the best weighted distance (< 7)."""
    from .cli import COMMANDS
    cands = sorted(set(COMMANDS) - {"version"})
    scored = sorted((0 if c.startswith(name) else _distance(name, c) + 1, c) for c in cands)
    if not scored:
        return ""
    best = scored[0][0]
    if best >= 7 or (best == 0 and all(x[0] == 0 for x in scored)):
        return ""
    close = [c for sc, c in scored if sc == best]
    head = "\nThe most similar command is\n" if len(close) == 1 else "\nThe most similar commands are\n"
    return head + "".join(f"\t{c}\n" for c in close)


def short_usage(name: str) -> str:
    u = SUMMARY.get(name)
    return f"usage: {u[0]}\n" if u else f"usage: git {name} [<options>]\n"


def man(name: str) -> str:
    u = SUMMARY.get(name)
    if not u:
        return f"No manual entry for git-{name} (bash-lite has short help only: try 'git {name} -h').\n"
    return f"git-{name} - {u[1]}\n\nUsage:\n    {u[0]}\n"


def help_cmd(g, args):
    if not args or args[0] in ("-a", "--all", "-g", "--guides"):
        g.out(usage_text())
        return 0
    g.out(man(args[0]))
    return 0


def _register():
    from .cli import COMMANDS
    COMMANDS.setdefault("help", help_cmd)
    COMMANDS.setdefault("version", lambda g, args: (g.out("git version 2.43.0\n"), 0)[1])


_register()

"""init, config, status, add, rm, mv, commit, restore, reset, clean: the daily loop, with git 2.43's output."""

from __future__ import annotations

import posixpath
import re

from ..vfs import FSError
from .cli import Usage, command, launch_editor, parse, pathspec_matcher, run_hook
from .diff import Colors, compare, numstat
from .discover import find_repo
from .objects import EMPTY_TREE, GitError, encode_commit
from .repo import SAMPLE_HOOKS, Config, ConfigFile, Entry, canon_key
from .revparse import dwim_ref, resolve, try_resolve, unknown_revision
from .status import collect, worktree_flat

INIT_HINT = (
    "hint: Using 'master' as the name for the initial branch. This default branch name\n"
    "hint: is subject to change. To configure the initial branch name to use in all\n"
    "hint: of your new repositories, which will suppress this warning, call:\n"
    "hint: \n"
    "hint: \tgit config --global init.defaultBranch <name>\n"
    "hint: \n"
    "hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and\n"
    "hint: 'development'. The just-created branch can be renamed via this command:\n"
    "hint: \n"
    "hint: \tgit branch -m <name>\n")
EXCLUDE = ("# git ls-files --others --exclude-from=.git/info/exclude\n# Lines that start with '#' are comments.\n"
           "# For a project mostly in C, the following would be a good set of\n"
           "# exclude patterns (uncomment them if you want to use them):\n# *.[oa]\n# *~\n")
DESCRIPTION = "Unnamed repository; edit this file 'description' to name the repository.\n"


def make_gitdir(ctx, gitdir: str, branch: str, bare: bool = False):
    fs = ctx.fs
    uid, gid = ctx.uid, ctx.proc.gid
    for d in ("", "hooks", "info", "objects/info", "objects/pack", "refs/heads", "refs/tags", "branches"):
        fs.mkdir(f"{gitdir}/{d}".rstrip("/"), "/", 0o755, uid, gid, parents=True)
    fs.write_file(f"{gitdir}/HEAD", f"ref: refs/heads/{branch}\n", "/", 0o644, uid, gid)
    fs.write_file(f"{gitdir}/description", DESCRIPTION, "/", 0o644, uid, gid)
    fs.write_file(f"{gitdir}/info/exclude", EXCLUDE, "/", 0o644, uid, gid)
    cfg = ("[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = " + ("true" if bare else "false") +
           "\n" + ("" if bare else "\tlogallrefupdates = true\n"))
    fs.write_file(f"{gitdir}/config", cfg, "/", 0o644, uid, gid)
    for h in SAMPLE_HOOKS:
        node = fs.write_file(f"{gitdir}/hooks/{h}.sample", f"#!/bin/sh\n#\n# An example hook script ({h}).\n"
                                                               f"# To enable this hook, rename this file to \"{h}\".\n",
                             "/", 0o755, uid, gid)
        node.mode = 0o755


@command("init")
def git_init(g, args):
    o = parse(args, "bare quiet|q b:|initial-branch= template= separate-git-dir= shared=?")
    target = o.args[0] if o.args else "."
    top = target if target.startswith("/") else posixpath.normpath(posixpath.join(g.cwd, target))
    fs = g.fs
    try:
        fs.mkdir(top, "/", 0o755, g.ctx.uid, g.ctx.proc.gid, parents=True)
    except FSError as e:
        raise GitError(f"fatal: cannot mkdir {target}: {e.msg}")
    bare = bool(o.get("bare"))
    gitdir = top if bare else f"{top}/.git"
    cfg = Config(fs, g.env.get("HOME", "/"), None, g.ctx.git_overrides, g.env)
    branch = o.get("b") or cfg.get("init.defaultBranch")
    re_init = fs.exists(f"{gitdir}/HEAD")
    if re_init:
        if not o.get("quiet"):
            g.out(f"Reinitialized existing Git repository in {gitdir}/\n")
        return 0
    if not branch and not o.get("quiet"):
        g.err(INIT_HINT)
    make_gitdir(g.ctx, gitdir, branch or "master", bare)
    if not o.get("quiet"):
        g.out(f"Initialized empty Git repository in {gitdir}/\n")
    return 0


# ---------------------------------------------------------------------------- config

@command("config")
def git_config(g, args):
    o = parse(args, "global system local file|f= blob= list|l get get-all get-regexp add unset unset-all "
                    "replace-all rename-section remove-section edit|e show-origin show-scope name-only null|z "
                    "bool int bool-or-int path type= default=")
    from .repo import Config
    repo = g.maybe_repo()
    home = g.env.get("HOME", "/")
    cfg = repo.config if repo else Config(g.fs, home, None, g.ctx.git_overrides, g.env)
    if o.get("global"):
        target, scope = cfg.global_file(), "global"
    elif o.get("system"):
        target, scope = ConfigFile(g.fs, "/etc/gitconfig"), "system"
    elif o.get("file"):
        f = o.get("file")
        target, scope = ConfigFile(g.fs, f if f.startswith("/") else posixpath.join(g.cwd, f)), "file"
    else:
        target, scope = (cfg.local_file() if repo else None), "local"
    uid, gid = g.ctx.uid, g.ctx.proc.gid
    a = o.args
    typ = o.get("type") or ("bool" if o.get("bool") else "int" if o.get("int") else None)

    def shown(v):
        if v is None:
            return "true"
        if typ == "bool":
            return "true" if v.lower() in ("true", "yes", "on", "1", "") else "false"
        return v

    def need_repo():
        if target is None:
            raise GitError("fatal: not in a git directory", 128)
    if o.get("edit"):
        if target is None:
            need_repo()
        if not target.exists():
            target.write("", uid, gid)
        st = yield from launch_editor(g, target.path)
        return st
    if o.get("list"):
        scoped = scope if (o.get("global") or o.get("system") or o.get("file") or o.get("local")) else None
        for sc, path, k, v in cfg.all() if not (o.get("file") or o.get("local")) else \
                [(scope, target.path if target else "", k, v) for k, v, _ in (target.entries() if target else [])]:
            if scoped and sc != scoped and not o.get("file"):
                continue
            pre = ""
            if o.get("show-scope"):
                pre += f"{sc}\t"
            if o.get("show-origin"):
                pre += f"file:{display_path(g, path)}\t" if path else "command line:\t"
            g.out(f"{pre}{k}" + ("" if o.get("name-only") else f"={'' if v is None else v}") + "\n")
        return 0
    if o.get("rename-section"):
        need_repo() if scope == "local" else None
        if len(a) != 2 or not target.rename_section(a[0].lower(), a[1], uid, gid):
            raise GitError("fatal: no such section: " + (a[0] if a else ""), 128)
        return 0
    if o.get("remove-section"):
        if not a or target is None or not target.remove_section(a[0], uid, gid):
            raise GitError("fatal: no such section: " + (a[0] if a else ""), 128)
        return 0
    if o.get("unset") or o.get("unset-all"):
        if not a:
            raise Usage("error: wrong number of arguments, should be 1")
        need_repo()
        n = target.unset(a[0], uid, gid, all_=bool(o.get("unset-all")), value_re=a[1] if len(a) > 1 else None)
        if n == 0:
            return 5
        if n < 0:
            g.err(f"warning: {a[0]} has multiple values\n")
            return 5
        return 0
    if o.get("get-regexp"):
        rx = re.compile(a[0] if a else ".")
        hit = False
        for sc, path, k, v in cfg.all():
            if rx.search(k):
                hit = True
                g.out(f"{k}" + ("" if o.get("name-only") else f" {shown(v)}") + "\n")
        return 0 if hit else 1
    if o.get("get") or o.get("get-all") or (len(a) == 1 and not o.get("add")):
        if not a:
            raise Usage("error: wrong number of arguments, should be 1")
        key = a[0]
        if "." not in key:
            g.err(f"error: key does not contain a section: {key}\n")
            return 2
        if o.get("global") or o.get("system") or o.get("file") or o.get("local"):
            vals = [v for k, v, _ in (target.entries() if target else []) if canon_key(k) == canon_key(key)]
            origins = [target.path] * len(vals)
        else:
            hits = [(path, v) for sc, path, k, v in cfg.all() if canon_key(k) == canon_key(key)]
            vals = [v for _, v in hits]
            origins = [p for p, _ in hits]
        if not vals:
            if o.get("default") is not None:
                g.out(o.get("default") + "\n")
                return 0
            return 1
        show = vals if o.get("get-all") else vals[-1:]
        orig = origins if o.get("get-all") else origins[-1:]
        for v, p in zip(show, orig):
            pre = f"file:{display_path(g, p)}\t" if o.get("show-origin") else ""
            g.out(pre + shown(v) + "\n")
        return 0
    if len(a) >= 2:
        key, value = a[0], a[1]
        if "." not in key or key.startswith(".") or key.endswith("."):
            g.err(f"error: key does not contain a section: {key}\n" if "." not in key else
                  f"error: invalid key: {key}\n")
            return 2
        need_repo()
        if typ == "bool":
            value = "true" if value.lower() in ("true", "yes", "on", "1") else "false"
        target.set(key, value, uid, gid, add=bool(o.get("add")), replace_all=bool(o.get("replace-all")),
                   value_re=a[2] if len(a) > 2 else None)
        return 0
    raise Usage("usage: git config [<options>]")


def display_path(g, path):
    home = g.env.get("HOME", "")
    if path.startswith("/") and g.cwd and path.startswith(g.cwd + "/"):
        return path[len(g.cwd) + 1:]
    return path


# ---------------------------------------------------------------------------- status

LABELS = {"A": "new file:", "M": "modified:", "D": "deleted:", "R": "renamed:", "T": "typechange:", "C": "copied:"}
UNMERGED = {"UU": "both modified:", "AA": "both added:", "UD": "deleted by them:", "DU": "deleted by us:",
            "AU": "added by us:", "UA": "added by them:", "DD": "both deleted:"}


def tracking_lines(repo, branch: str, long: bool = True) -> list[str]:
    up = repo.upstream(branch)
    if not up:
        return []
    ref = repo.upstream_ref(branch)
    short = ref[len("refs/remotes/"):] if ref.startswith("refs/remotes/") else ref[len("refs/heads/"):]
    them = repo.resolve_ref(ref)
    if them is None:
        return [f"Your branch is based on '{short}', but the upstream is gone.",
                '  (use "git branch --unset-upstream" to fixup)']
    me = repo.resolve_ref(f"refs/heads/{branch}")
    ahead, behind = ahead_behind(repo, me, them)
    if not ahead and not behind:
        return [f"Your branch is up to date with '{short}'."]
    if ahead and not behind:
        return [f"Your branch is ahead of '{short}' by {ahead} commit{'s' if ahead != 1 else ''}.",
                '  (use "git push" to publish your local commits)']
    if behind and not ahead:
        return [f"Your branch is behind '{short}' by {behind} commit{'s' if behind != 1 else ''}, and can be "
                f"fast-forwarded.", '  (use "git pull" to update your local branch)']
    return [f"Your branch and '{short}' have diverged,",
            f"and have {ahead} and {behind} different commits each, respectively.",
            '  (use "git pull" if you want to integrate the remote branch with yours)']


def ahead_behind(repo, a, b):
    if not a or not b:
        return 0, 0
    ra, rb = repo.odb.reachable([a]), repo.odb.reachable([b])
    return len(ra - rb), len(rb - ra)


def in_progress(repo):
    """('merge' | 'rebase' | 'rebase-i' | 'cherry-pick' | 'revert' | 'bisect' | 'am', detail) or (None, None)."""
    if repo.exists("rebase-merge"):
        return ("rebase-i" if repo.exists("rebase-merge/interactive") else "rebase"), None
    if repo.exists("rebase-apply"):
        return "rebase", None
    if repo.exists("MERGE_HEAD"):
        return "merge", None
    if repo.exists("CHERRY_PICK_HEAD"):
        return "cherry-pick", (repo.read("CHERRY_PICK_HEAD") or "").strip()
    if repo.exists("REVERT_HEAD"):
        return "revert", (repo.read("REVERT_HEAD") or "").strip()
    if repo.exists("BISECT_LOG"):
        return "bisect", None
    return None, None


def head_line(repo):
    b = repo.head_branch()
    if b:
        return f"On branch {b}"
    h = repo.head()
    detached_from = None
    for old, new, who, msg in repo.reflog("HEAD"):
        m = re.match(r"checkout: moving from \S+ to (\S+)", msg)
        if m:
            detached_from = m.group(1)
            target_sha = new
            break
    db = repo.odb
    if detached_from:
        # a tag or remote branch is named; anything else (HEAD~1, an id) as the commit it was then
        full, sha_ref = dwim_ref(repo, detached_from)
        if full and full.startswith(("refs/tags/", "refs/remotes/")):
            shown = full.split("/", 2)[-1]
            tsha = repo.peel(sha_ref)
        else:
            tsha = target_sha
            shown = db.abbrev(tsha)
        word = "at" if tsha == h else "from"
        return f"HEAD detached {word} {shown}"
    if repo.exists("rebase-merge") or repo.exists("rebase-apply"):
        onto = (repo.read("rebase-merge/onto") or repo.read("rebase-apply/onto") or "").strip()
        return f"interactive rebase in progress; onto {db.abbrev(onto)}" if onto else "rebase in progress"
    return f"HEAD detached at {db.abbrev(h)}" if h else "Not currently on any branch."


@command("status", wt=True)
def git_status(g, args):
    o = parse(args, "short|s branch|b porcelain=? long untracked-files|u=? ignored=? verbose|v show-stash ahead-behind "
                    "null|z column renames find-renames=?")
    repo = g.repo(wt=True)
    if repo.bare:
        raise GitError("fatal: this operation must be run in a work tree")
    unt = o.get("untracked-files")
    unt = "all" if unt is True else (unt or repo.config.get("status.showUntrackedFiles") or "normal")
    if unt in ("no", "false"):
        unt = "no"
    st = collect(repo, untracked=unt, ignored=bool(o.get("ignored")))
    match = pathspec_matcher(g, repo, o.args + (o.paths or []))
    if match:
        st.staged = {k: v for k, v in st.staged.items() if match(k)}
        st.unstaged = {k: v for k, v in st.unstaged.items() if match(k)}
        st.untracked = [p for p in st.untracked if match(p.rstrip("/"))]
        st.unmerged = {k: v for k, v in st.unmerged.items() if match(k)}
    porcelain = o.get("porcelain")
    if o.get("short") or porcelain:
        return _status_short(g, repo, st, bool(o.get("branch")), porcelain=bool(porcelain),
                             color=Colors(g.color("status") and not porcelain))
    c = Colors(g.color("status"))
    lines = [head_line(repo)]
    b = repo.head_branch()
    if b:
        tl = tracking_lines(repo, b)
        lines += tl
    state, detail = in_progress(repo)
    if not repo.head():
        lines += ["", "No commits yet"]
    elif state == "merge":
        if st.unmerged:
            lines += ["You have unmerged paths.", '  (fix conflicts and run "git commit")',
                      '  (use "git merge --abort" to abort the merge)']
        else:
            lines += ["All conflicts fixed but you are still merging.", '  (use "git commit" to conclude merge)']
    elif state in ("rebase", "rebase-i"):
        onto = (repo.read("rebase-merge/onto") or "").strip()
        head_name = (repo.read("rebase-merge/head-name") or "").strip().replace("refs/heads/", "")
        if state == "rebase-i":
            done = [x for x in (repo.read("rebase-merge/done") or "").split("\n") if x.strip()]
            todo = [x for x in (repo.read("rebase-merge/git-rebase-todo") or "").split("\n")
                    if x.strip() and not x.startswith("#")]
            lines += [f"Last command{'s' if len(done) != 1 else ''} done ({len(done)} command{'s' if len(done) != 1 else ''} done):"]
            lines += [f"   {x}" for x in done[-2:]]
            if todo:
                lines += [f"Next command{'s' if len(todo) != 1 else ''} to do ({len(todo)} remaining command{'s' if len(todo) != 1 else ''}):"]
                lines += [f"   {x}" for x in todo[:2]]
            else:
                lines += ["No commands remaining."]
        if st.unmerged:
            lines += [f"You are currently rebasing branch '{head_name}' on '{repo.odb.abbrev(onto)}'.",
                      '  (fix conflicts and then run "git rebase --continue")',
                      '  (use "git rebase --skip" to skip this patch)',
                      '  (use "git rebase --abort" to check out the original branch)']
        elif repo.exists("rebase-merge/amend") or repo.exists("rebase-merge/stopped-sha"):
            lines += [f"You are currently editing a commit while rebasing branch '{head_name}' on "
                      f"'{repo.odb.abbrev(onto)}'.", '  (use "git commit --amend" to amend the current commit)',
                      '  (use "git rebase --continue" once you are satisfied with your changes)']
        else:
            lines += [f"You are currently rebasing branch '{head_name}' on '{repo.odb.abbrev(onto)}'.",
                      '  (all conflicts fixed: run "git rebase --continue")']
    elif state == "cherry-pick":
        lines += [f"You are currently cherry-picking commit {repo.odb.abbrev(detail)}."]
        if st.unmerged:
            lines += ['  (fix conflicts and run "git cherry-pick --continue")',
                      '  (use "git cherry-pick --skip" to skip this patch)',
                      '  (use "git cherry-pick --abort" to cancel the cherry-pick operation)']
        else:
            lines += ['  (all conflicts fixed: run "git cherry-pick --continue")',
                      '  (use "git cherry-pick --skip" to skip this patch)',
                      '  (use "git cherry-pick --abort" to cancel the cherry-pick operation)']
    elif state == "revert":
        lines += [f"You are currently reverting commit {repo.odb.abbrev(detail)}."]
        lines += ['  (fix conflicts and run "git revert --continue")' if st.unmerged else
                  '  (all conflicts fixed: run "git revert --continue")',
                  '  (use "git revert --skip" to skip this patch)',
                  '  (use "git revert --abort" to cancel the revert operation)']
    elif state == "bisect":
        start = (repo.read("BISECT_START") or "").strip()
        lines += [f"You are currently bisecting, started from branch '{start}'.",
                  '  (use "git bisect reset" to get back to the original branch)']
    if repo.config.get_bool("core.sparseCheckout", False):
        ents = repo.index.stage0().values()
        total = len(ents)
        present = sum(1 for e in ents if not e.flags2 & 0x4000)
        if total:
            lines.append(f"You are in a sparse checkout with {present * 100 // total}% of tracked files present.")
    if len(lines) > 1 and lines[-1] != "":      # a blank line only after tracking / state information
        lines.append("")
    staged_sorted = sorted(st.staged)
    if st.staged:
        lines.append("Changes to be committed:")
        lines.append('  (use "git restore --staged <file>..." to unstage)' if repo.head() else
                     '  (use "git rm --cached <file>..." to unstage)')
        for p in staged_sorted:
            v = st.staged[p]
            if isinstance(v, tuple):
                label = LABELS["R"]
                name = f"{g.shown(repo, v[1])} -> {g.shown(repo, p)}"
            else:
                label = LABELS[v]
                name = g.shown(repo, p)
            lines.append("\t" + c("32", f"{label:<12}{name}"))
        lines.append("")
    if st.unmerged:
        lines.append("Unmerged paths:")
        if state in ("rebase", "rebase-i", "cherry-pick", "revert"):
            lines.append('  (use "git restore --staged <file>..." to unstage)')
        lines.append('  (use "git add <file>..." to mark resolution)' if not any(
            v in ("UD", "DU", "AU", "UA", "DD") for v in st.unmerged.values()) else
            '  (use "git add/rm <file>..." as appropriate to mark resolution)')
        for p in sorted(st.unmerged):
            label = UNMERGED[st.unmerged[p]]
            lines.append("\t" + c("31", f"{label:<17}{g.shown(repo, p)}"))
        lines.append("")
    if st.unstaged:
        lines.append("Changes not staged for commit:")
        has_del = any(v == "D" for v in st.unstaged.values())
        lines.append('  (use "git add/rm <file>..." to update what will be committed)' if has_del else
                     '  (use "git add <file>..." to update what will be committed)')
        lines.append('  (use "git restore <file>..." to discard changes in working directory)')
        for p in sorted(st.unstaged):
            v = st.unstaged[p]
            sub = ""
            node = repo.wt_node(p)
            if node is not None and node.type == "d":
                sub = " (new commits)"
            lines.append("\t" + c("31", f"{LABELS[v]:<12}{g.shown(repo, p)}{sub}"))
        lines.append("")
    if st.untracked:
        lines.append("Untracked files:")
        lines.append('  (use "git add <file>..." to include in what will be committed)')
        for p in st.untracked:
            lines.append("\t" + c("31", g.shown(repo, p)))
        lines.append("")
    if o.get("ignored") and st.ignored:
        lines.append("Ignored files:")
        lines.append('  (use "git add -f <file>..." to include in what will be committed)')
        for p in st.ignored:
            lines.append("\t" + c("31", g.shown(repo, p)))
        lines.append("")
    if st.staged:
        pass                        # git prints no closing line when something is staged
    elif st.unstaged or st.unmerged:
        lines.append('no changes added to commit (use "git add" and/or "git commit -a")')
    elif st.untracked:
        lines.append('nothing added to commit but untracked files present (use "git add" to track)')
    else:
        if not repo.head():
            lines.append('nothing to commit (create/copy files and use "git add" to track)')
        elif unt == "no":
            lines.append("nothing to commit (use -u to show untracked files)")
        else:
            lines.append("nothing to commit, working tree clean")
    if lines and lines[-1] == "":
        lines.pop()
    if o.get("show-stash"):
        n = len(repo.reflog("refs/stash"))
        if n:
            lines.append(f"Your stash currently has {n} entr{'ies' if n != 1 else 'y'}")
    g.out("\n".join(lines) + "\n")
    return 0


def _status_short(g, repo, st, branch, porcelain, color):
    rows = []
    if branch:
        b = repo.head_branch()
        if not repo.head():
            head = f"## No commits yet on {b}"
        elif b:
            head = f"## {color('32', b)}"
            up = repo.upstream(b)
            if up:
                ref = repo.upstream_ref(b)
                short = ref[len('refs/remotes/'):] if ref.startswith('refs/remotes/') else ref[len('refs/heads/'):]
                head += f"...{color('31', short)}"
                them = repo.resolve_ref(ref)
                if them is None:
                    head += " [gone]"
                else:
                    a, bh = ahead_behind(repo, repo.head(), them)
                    parts = []
                    if a:
                        parts.append(f"ahead {color('32', str(a))}")
                    if bh:
                        parts.append(f"behind {color('31', str(bh))}")
                    if parts:
                        head += f" [{', '.join(parts)}]"
        else:
            head = f"## {color('31', 'HEAD (no branch)')}"
        rows.append(head)
    paths = sorted(set(st.staged) | set(st.unstaged) | set(st.unmerged))
    for p in paths:
        shown = p if porcelain else g.shown(repo, p)
        if p in st.unmerged:
            code = st.unmerged[p]
            rows.append(f"{color('31', code)} {shown}")
            continue
        x = st.staged.get(p, " ")
        y = st.unstaged.get(p, " ")
        if isinstance(x, tuple):
            old = x[1] if porcelain else g.shown(repo, x[1])
            rows.append(f"{color('32', 'R')}{color('31', y) if y != ' ' else ' '} {old} -> {shown}")
            continue
        rows.append(f"{color('32', x) if x != ' ' else ' '}{color('31', y) if y != ' ' else ' '} {shown}")
    for p in st.untracked:
        rows.append(f"{color('31', '??')} {p if porcelain else g.shown(repo, p)}")
    for p in st.ignored:
        rows.append(f"{color('31', '!!')} {p if porcelain else g.shown(repo, p)}")
    if rows:
        g.out("\n".join(rows) + "\n")
    return 0


# ---------------------------------------------------------------------------- add / rm / mv

@command("add", wt=True)
def git_add(g, args):
    o = parse(args, "all|A update|u patch|p force|f dry-run|n verbose|v intent-to-add|N ignore-removal "
                    "no-all interactive|i edit|e renormalize chmod=")
    repo = g.repo(wt=True)
    specs = o.args + (o.paths or [])
    if o.get("patch") or o.get("interactive"):
        from .interactive import add_patch
        return (yield from add_patch(g, repo, specs))
    if not specs and not o.get("all") and not o.get("update"):
        g.err("Nothing specified, nothing added.\nhint: Maybe you wanted to say 'git add .'?\n"
              "hint: Turn this message off by running\nhint: \"git config advice.addEmptyPathspec false\"\n")
        return 0
    if not specs:
        specs = [":/"]
    match = pathspec_matcher(g, repo, specs)
    idx = repo.index
    files = repo.worktree_files(include_ignored=True)
    tracked = idx.stage0()
    conflicted = idx.conflicts()
    to_add, to_remove, ignored_hits = [], [], []
    # pathspecs that name nothing at all are errors
    for spec in specs:
        if spec.startswith(":"):
            continue
        m1 = pathspec_matcher(g, repo, [spec])
        if not any(m1(p) for p in files) and not any(m1(p) for p in tracked) and not any(m1(p) for p in conflicted):
            raise GitError(f"fatal: pathspec '{spec}' did not match any files")
    for p, node in sorted(files.items()):
        if not match(p):
            continue
        is_tracked = p in tracked or p in conflicted
        if o.get("update") and not is_tracked:
            continue
        if not is_tracked and repo.is_ignored(p, node.type == "d") and not o.get("force"):
            # explicitly named ignored files are refused; ones a directory spec covers are skipped quietly
            if any(_names_exactly(g, repo, s, p) for s in specs):
                ignored_hits.append(p)
            continue
        if node.type == "d" and ".git" not in (node.entries or {}):
            continue
        to_add.append((p, node))
    if not o.get("ignore-removal"):
        for p in list(tracked) + list(conflicted):
            if match(p) and p not in files and p not in [x for x, _ in to_add]:
                to_remove.append(p)
    if ignored_hits:
        shown = sorted({_ignored_spec(g, repo, s, ignored_hits) for s in specs} - {None})
        g.err("The following paths are ignored by one of your .gitignore files:\n" +
              "".join(f"{x}\n" for x in shown) +
              "hint: Use -f if you really want to add them.\nhint: Turn this message off by running\n"
              "hint: \"git config advice.addIgnoredFile false\"\n")
        if not to_add and not to_remove:
            return 1
    for p, node in to_add:
        if o.get("intent-to-add"):
            if idx.get(p) is None:
                e = Entry(p, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0o100644)
                e.flags2 = 0x2000
                if not o.get("dry-run"):
                    idx.add(e)
            continue
        old = idx.get(p)
        if old is not None and repo.stat_matches(old, node) and old.mode == repo.file_mode(node) and p not in conflicted:
            continue
        if node.type == "d":
            got = repo.wt_sha(p)
            if got is None:
                continue
            sha, mode = got
        else:
            sha, mode = repo.wt_sha(p, write=not o.get("dry-run"))
        chmod = o.get("chmod")
        if chmod == "+x" and mode == 0o100644:
            mode = 0o100755
        elif chmod == "-x" and mode == 0o100755:
            mode = 0o100644
        if old is not None and old.sha == sha and old.mode == mode and p not in conflicted:
            if not o.get("dry-run"):
                idx.add(repo.index_entry_for(p, sha, mode, node))
            continue
        if o.get("verbose") or o.get("dry-run"):
            g.out(f"add '{g.shown(repo, p)}'\n")
        if not o.get("dry-run"):
            idx.add(repo.index_entry_for(p, sha, mode, node))
    for p in to_remove:
        if o.get("verbose") or o.get("dry-run"):
            g.out(f"remove '{g.shown(repo, p)}'\n")
        if not o.get("dry-run"):
            idx.remove(p)
    if not o.get("dry-run"):
        repo.write_index()
    return 1 if ignored_hits else 0


def _names_exactly(g, repo, spec, p):
    try:
        rel = g.to_rel(repo, spec)
    except GitError:
        return False
    return rel == p or (rel and p.startswith(rel + "/") and repo.is_ignored(rel, True))


def _ignored_spec(g, repo, spec, hits):
    try:
        rel = g.to_rel(repo, spec)
    except GitError:
        return None
    for p in hits:
        if rel == p:
            return spec
        if rel and p.startswith(rel + "/"):
            return spec
    return None


@command("rm", wt=True)
def git_rm(g, args):
    o = parse(args, "cached force|f recursive|r dry-run|n quiet|q ignore-unmatch")
    repo = g.repo(wt=True)
    specs = o.args + (o.paths or [])
    if not specs:
        raise Usage("usage: git rm [-f | --force] [-n] [-r] [--cached] [--ignore-unmatch]\n"
                    "              [--quiet] [--pathspec-from-file=<file> [--pathspec-file-nul]]\n"
                    "              [--] [<pathspec>...]")
    idx = repo.index
    tracked = sorted({p for p, _ in idx.entries})
    victims = []
    for spec in specs:
        rel = g.to_rel(repo, spec) if not any(ch in spec for ch in "*?[") else None
        m = pathspec_matcher(g, repo, [spec])
        hits = [p for p in tracked if m(p)]
        if not hits:
            if o.get("ignore-unmatch"):
                continue
            raise GitError(f"fatal: pathspec '{spec}' did not match any files")
        if rel is not None and rel not in tracked and not o.get("recursive"):
            raise GitError(f"fatal: not removing '{spec}' recursively without -r")
        victims += [h for h in hits if h not in victims]
    if not o.get("force") and not o.get("cached"):
        head = repo.odb.flat(repo.head_tree())
        staged, local = [], []
        wt = worktree_flat(repo, paths=victims)
        for p in victims:
            e = idx.get(p)
            if e is None:
                continue
            h = head.get(p)
            w = wt.get(p)
            in_index_differs = h is None or h != (e.mode_str, e.sha)
            wt_differs = w is not None and w != (e.mode_str, e.sha)
            if in_index_differs and wt_differs:
                staged.append(p)
            elif in_index_differs:
                staged.append(p)
            elif wt_differs:
                local.append(p)
        if staged or local:
            msg = []
            both = [p for p in staged if wt.get(p) is not None and wt.get(p) != (idx.get(p).mode_str, idx.get(p).sha)]
            only_staged = [p for p in staged if p not in both]
            if both:
                msg.append("error: the following file has staged content different from both the\nfile and the HEAD:"
                           if len(both) == 1 else "error: the following files have staged content different from "
                                                  "both the\nfile and the HEAD:")
                msg += [f"    {g.shown(repo, p)}" for p in both]
                msg.append("(use -f to force removal)")
            if only_staged:
                msg.append("error: the following file has changes staged in the index:" if len(only_staged) == 1
                           else "error: the following files have changes staged in the index:")
                msg += [f"    {g.shown(repo, p)}" for p in only_staged]
                msg.append("(use --cached to keep the file, or -f to force removal)")
            if local:
                msg.append("error: the following file has local modifications:" if len(local) == 1 else
                           "error: the following files have local modifications:")
                msg += [f"    {g.shown(repo, p)}" for p in local]
                msg.append("(use --cached to keep the file, or -f to force removal)")
            raise GitError("\n".join(msg), 1)
    for p in victims:
        if not o.get("quiet"):
            g.out(f"rm '{p}'\n")
        if o.get("dry-run"):
            continue
        idx.remove(p)
        if not o.get("cached"):
            repo.remove_file(p)
    if not o.get("dry-run"):
        repo.write_index()
    return 0


@command("mv", wt=True)
def git_mv(g, args):
    o = parse(args, "force|f dry-run|n verbose|v k")
    repo = g.repo(wt=True)
    a = o.args + (o.paths or [])
    if len(a) < 2:
        raise Usage("usage: git mv [<options>] <source>... <destination>")
    *srcs, dst = a
    idx = repo.index
    dst_rel = g.to_rel(repo, dst)
    dst_node = repo.wt_node(dst_rel)
    into_dir = dst_node is not None and dst_node.type == "d"
    if len(srcs) > 1 and not into_dir:
        raise GitError(f"fatal: destination '{dst}' is not a directory")
    moves = []
    for s in srcs:
        rel = g.to_rel(repo, s)
        node = repo.wt_node(rel)
        if node is None:
            raise GitError(f"fatal: bad source, source={s}, destination={dst}")
        target = f"{dst_rel}/{posixpath.basename(rel)}" if into_dir else dst_rel
        target = target.lstrip("/")
        if node.type == "d":
            inside = [p for p in idx.paths() if p.startswith(rel + "/")]
            if not inside:
                raise GitError(f"fatal: source directory is empty, source={s}, destination={dst}")
            for p in inside:
                moves.append((p, target + p[len(rel):]))
            moves.append((rel, target, True))
            continue
        if idx.get(rel) is None:
            raise GitError(f"fatal: not under version control, source={s}, destination={dst}")
        if repo.wt_node(target) is not None and not o.get("force"):
            raise GitError(f"fatal: destination exists, source={s}, destination={dst}")
        moves.append((rel, target))
    for mv in moves:
        if len(mv) == 3:
            src, tgt, _ = mv
            if o.get("verbose") or o.get("dry-run"):
                g.out(f"Renaming {src} to {tgt}\n")
            if not o.get("dry-run"):
                repo._make_dirs(posixpath.dirname(repo.wt_path(tgt)))
                g.fs.rename(repo.wt_path(src), repo.wt_path(tgt))
            continue
        src, tgt = mv
        if o.get("dry-run"):
            g.out(f"Checking rename of '{src}' to '{tgt}'\nRenaming {src} to {tgt}\n")
            continue
        if not any(len(m) == 3 and src.startswith(m[0] + "/") for m in moves):
            if o.get("verbose"):
                g.out(f"Renaming {src} to {tgt}\n")
            repo._make_dirs(posixpath.dirname(repo.wt_path(tgt)))
            if repo.wt_node(tgt) is not None:
                repo.remove_file(tgt)
            g.fs.rename(repo.wt_path(src), repo.wt_path(tgt))
        e = idx.get(src)
        idx.remove(src)
        node = repo.wt_node(tgt)
        ne = Entry(tgt, e.sha, e.mode, 0, e.ctime, node.mtime if node else e.mtime, e.dev, node.ino if node else e.ino,
                   e.uid, e.gid, e.size)
        idx.add(ne)
    if not o.get("dry-run"):
        repo.write_index()
    return 0


# ---------------------------------------------------------------------------- commit

COMMENT_HELP = ("# Please enter the commit message for your changes. Lines starting\n"
                "# with '#' will be ignored, and an empty message aborts the commit.\n")


def cleanup_message(text: str, strip_comments: bool = True) -> str:
    lines = [ln.rstrip() for ln in text.split("\n")]
    if strip_comments:
        cut = None
        for i, ln in enumerate(lines):
            if ln.startswith("# ------------------------ >8 ------------------------"):
                cut = i
                break
        if cut is not None:
            lines = lines[:cut]
        lines = [ln for ln in lines if not ln.startswith("#")]
    out = []
    for ln in lines:
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n" if out else ""


def commit_summary(g, repo, sha, parent_tree, branch_label, show_date=False, initial=None):
    """[main (root-commit) abc1234] subject, Author/Date lines when they're interesting, the diffstat summary and
    create/delete/rename/mode lines."""
    db = repo.odb
    c = db.commit(sha)
    root = " (root-commit)" if (not c.parents if initial is None else initial) else ""
    lines = [f"[{branch_label}{root} {db.abbrev(sha)}] {c.subject}"]
    if c.author.email != c.committer.email or c.author.name != c.committer.name:
        lines.append(f" Author: {c.author.name} <{c.author.email}>")
    if show_date:
        lines.append(f" Date: {_ctime(c.author)}")
    changes = compare(db.flat(parent_tree) if parent_tree else {}, db.flat(c.tree), renames=True, db=db)
    ins = dels = 0
    for ch in changes:
        a, d = numstat(ch, db)
        ins += a or 0
        dels += d or 0
    files = len(changes)
    s = f" {files} file{'s' if files != 1 else ''} changed"
    if ins or not dels:
        s += f", {ins} insertion{'s' if ins != 1 else ''}(+)"
    if dels or not ins:
        s += f", {dels} deletion{'s' if dels != 1 else ''}(-)"
    lines.append(s)
    for ch in changes:
        if ch.status == "A":
            lines.append(f" create mode {ch.new_mode} {ch.new_path}")
        elif ch.status == "D":
            lines.append(f" delete mode {ch.old_mode} {ch.old_path}")
        elif ch.status == "R":
            from .diff import stat_name
            lines.append(f" rename {stat_name(ch)} ({ch.score}%)")
            if ch.old_mode != ch.new_mode:
                lines.append(f" mode change {ch.old_mode} => {ch.new_mode} {ch.new_path}")
        elif ch.old_mode != ch.new_mode:
            lines.append(f" mode change {ch.old_mode} => {ch.new_mode} {ch.new_path}")
    return "\n".join(lines) + "\n"


def status_comment(g, repo, st, branch):
    lines = ["#"]
    if branch:
        lines.append(f"# On branch {branch}")
    else:
        lines.append(f"# {head_line(repo)}")
    if not repo.head():
        lines += ["#", "# Initial commit"]
    if st.staged:
        lines += ["#", "# Changes to be committed:"]
        for p in sorted(st.staged):
            v = st.staged[p]
            if isinstance(v, tuple):
                lines.append(f"#\t{LABELS['R']:<12}{v[1]} -> {p}")
            else:
                lines.append(f"#\t{LABELS[v]:<12}{p}")
    if st.unstaged:
        lines += ["#", "# Changes not staged for commit:"]
        for p in sorted(st.unstaged):
            lines.append(f"#\t{LABELS[st.unstaged[p]]:<12}{p}")
    if st.untracked:
        lines += ["#", "# Untracked files:"]
        for p in st.untracked:
            lines.append(f"#\t{p}")
    lines.append("#")
    return "\n".join(lines) + "\n"


@command("commit", wt=True)
def git_commit(g, args):
    o = parse(args, "message|m= all|a amend no-edit edit allow-empty allow-empty-message fixup= squash= "
                    "file|F= reuse-message|C= reedit-message|c= author= date= signoff|s verbose|v quiet|q "
                    "dry-run no-verify|n include|i only|o reset-author cleanup= status no-status "
                    "untracked-files|u=? short porcelain patch|p trailer= gpg-sign|S=? no-gpg-sign")
    repo = g.repo(wt=True)
    db = repo.odb
    idx = repo.index
    state, _ = in_progress(repo)
    if o.get("patch"):
        from .interactive import add_patch
        st = yield from add_patch(g, repo, o.args + (o.paths or []))
        if st:
            return st
        idx = repo.index
    if idx.conflicts():
        raise GitError("error: Committing is not possible because you have unmerged files.\nhint: Fix them up in the "
                       "work tree, and then use 'git add/rm <file>'\nhint: as appropriate to mark resolution and make "
                       "a commit.\nfatal: Exiting because of an unresolved conflict.", 128)
    specs = o.args + (o.paths or [])
    if o.get("all"):
        if specs:
            raise GitError("fatal: paths '" + " ".join(specs) + " ...' with -a does not make sense")
        st0 = collect(repo, untracked="no")
        for p, v in st0.unstaged.items():
            if v == "D":
                idx.remove(p)
            else:
                node = repo.wt_node(p)
                sha, mode = repo.wt_sha(p, write=True)
                idx.add(repo.index_entry_for(p, sha, mode, node))
    elif specs:
        # git commit <paths>: commit just those paths as they are in the working tree
        match = pathspec_matcher(g, repo, specs)
        hit = False
        for p in idx.paths():
            if match(p):
                hit = True
                node = repo.wt_node(p)
                if node is None:
                    idx.remove(p)
                else:
                    sha, mode = repo.wt_sha(p, write=True)
                    idx.add(repo.index_entry_for(p, sha, mode, node))
        if not hit:
            raise GitError(f"error: pathspec '{specs[0]}' did not match any file(s) known to git", 1)
    head = repo.head()
    parents = []
    if o.get("amend"):
        if not head:
            raise GitError("fatal: You have nothing to amend.")
        hc = db.commit(head)
        parents = list(hc.parents)
    elif head:
        parents = [head]
    merge_head = (repo.read("MERGE_HEAD") or "").split()
    if state == "merge" and not o.get("amend"):
        parents += [m for m in merge_head if m not in parents]
    tree = db.write_tree(idx.flat())
    parent_tree = db.commit(parents[0]).tree if parents else None
    if not o.get("amend") and not o.get("allow-empty") and state != "merge" and len(parents) < 2:
        base_tree = db.commit(head).tree if head else EMPTY_TREE
        if tree == base_tree or (not head and not idx.flat()):
            st = collect(repo)
            b = repo.head_branch()
            if state == "cherry-pick":
                g.out("On branch " + (b or "") + "\nYou are currently cherry-picking commit " +
                      db.abbrev((repo.read("CHERRY_PICK_HEAD") or "").strip()) + ".\n\nnothing to commit, working tree "
                      "clean\nThe previous cherry-pick is now empty, possibly due to conflict resolution.\nIf you wish "
                      "to commit it anyway, use:\n\n    git commit --allow-empty\n\nOtherwise, please use 'git "
                      "cherry-pick --skip'\n")
                return 1
            # git commit with nothing staged prints the status and fails
            out = []
            out.append(head_line(repo))
            if b:
                out += tracking_lines(repo, b)
            if not head:
                out += ["", "Initial commit"]            # (commit says this where status says "No commits yet")
            if len(out) > 1:                             # a blank line only after tracking information
                out.append("")
            if st.unstaged:
                out.append("Changes not staged for commit:")
                has_del = any(v == "D" for v in st.unstaged.values())
                out.append('  (use "git add/rm <file>..." to update what will be committed)' if has_del else
                           '  (use "git add <file>..." to update what will be committed)')
                out.append('  (use "git restore <file>..." to discard changes in working directory)')
                for p in sorted(st.unstaged):
                    out.append(f"\t{LABELS[st.unstaged[p]]:<12}{g.shown(repo, p)}")
                out.append("")
            if st.untracked:
                out.append("Untracked files:")
                out.append('  (use "git add <file>..." to include in what will be committed)')
                for p in st.untracked:
                    out.append(f"\t{g.shown(repo, p)}")
                out.append("")
            if st.unstaged:
                out.append('no changes added to commit (use "git add" and/or "git commit -a")')
            elif st.untracked:
                out.append('nothing added to commit but untracked files present (use "git add" to track)')
            elif not head:
                out.append('nothing to commit (create/copy files and use "git add" to track)')
            else:
                out.append("nothing to commit, working tree clean")
            g.out("\n".join(out) + "\n")
            return 1
    # the message
    msg = None
    if o.get("message") is not None:
        msgs = o.all("message")
        msg = "\n\n".join(m.strip("\n") for m in msgs) + "\n"
    elif o.get("file"):
        f = o.get("file")
        if f == "-":
            data = yield from g.ctx.read_fd(0)
        else:
            try:
                data = bytes(g.fs.lookup(f if f.startswith("/") else posixpath.join(g.cwd, f)).data)
            except FSError:
                raise GitError(f"fatal: could not read log file '{f}': No such file or directory")
        msg = data.decode("utf-8", "replace")
    elif o.get("reuse-message") or o.get("reedit-message"):
        src = resolve(repo, o.get("reuse-message") or o.get("reedit-message"), "commit")
        msg = db.commit(src).message
    elif o.get("fixup"):
        target = o.get("fixup")
        kind = ""
        if target.startswith(("amend:", "reword:")):
            kind, _, target = target.partition(":")
        t = resolve(repo, target, "commit")
        msg = f"{'amend' if kind else 'fixup'}! {db.commit(t).subject}\n"
        if kind:
            msg += "\n" + db.commit(t).body
    elif o.get("squash"):
        t = resolve(repo, o.get("squash"), "commit")
        msg = f"squash! {db.commit(t).subject}\n"
    elif o.get("amend") and o.get("no-edit"):
        msg = db.commit(head).message
    elif state == "merge" and o.get("no-edit"):
        msg = repo.read("MERGE_MSG") or "Merge\n"
    edit = msg is None or bool(o.get("edit")) or (o.get("reedit-message") and not o.get("no-edit"))
    if edit:
        if msg is None:
            if o.get("amend"):
                msg = db.commit(head).message
            elif state in ("merge", "cherry-pick", "revert"):
                msg = repo.read("MERGE_MSG") or ""
            elif repo.config.get("commit.template"):
                t = repo.config.get("commit.template").replace("~", g.env.get("HOME", ""), 1)
                try:
                    msg = bytes(g.fs.lookup(t if t.startswith("/") else posixpath.join(g.cwd, t)).data).decode()
                except FSError:
                    msg = ""
            else:
                msg = ""
        st = collect(repo, untracked="normal")
        text = (msg if msg.endswith("\n") or not msg else msg + "\n") + "\n" + COMMENT_HELP
        if o.get("amend"):
            text += "#\n# Date:      " + _ctime(db.commit(head).author) + "\n"
        text += status_comment(g, repo, st, repo.head_branch())
        repo.write("COMMIT_EDITMSG", text)
        st_ed = yield from launch_editor(g, repo.path("COMMIT_EDITMSG"))
        if st_ed:
            raise GitError("Please supply the message using either -m or -F option.", 1)
        msg = repo.read("COMMIT_EDITMSG") or ""
    cleanup = o.get("cleanup") or ("strip" if edit else "whitespace")
    msg = cleanup_message(msg, strip_comments=cleanup in ("strip", "default", "scissors"))
    if o.get("signoff"):
        who = repo.ident("committer")
        line = f"Signed-off-by: {who.name} <{who.email}>"
        if line not in msg:
            msg = msg.rstrip("\n") + ("\n" if re.search(r"^[A-Za-z-]+: ", msg.strip().split("\n")[-1]) else "\n\n") + line + "\n"
    for tr in o.all("trailer"):
        msg = msg.rstrip("\n") + "\n\n" + tr.replace("=", ": ", 1) + "\n" if ":" not in tr else msg.rstrip("\n") + "\n\n" + tr + "\n"
    if not msg.strip() and not o.get("allow-empty-message"):
        g.err("Aborting commit due to empty commit message.\n")
        return 1
    repo.write_index()          # hooks (and their git commands) see what is being committed
    if not o.get("no-verify"):
        st = yield from run_hook(g, repo, "pre-commit")
        if st:
            return 1
        repo.write("COMMIT_EDITMSG", msg)
        st = yield from run_hook(g, repo, "commit-msg", [repo.path("COMMIT_EDITMSG")])
        if st:
            return 1
        msg = repo.read("COMMIT_EDITMSG") or msg
        repo._index = None
        idx = repo.index
        tree = db.write_tree(idx.flat())
    author = repo.ident("author")
    if o.get("amend") and not o.get("reset-author"):
        author = db.commit(head).author
    if o.get("author"):
        m = re.match(r"^\s*(.*?)\s*<(.*)>\s*$", o.get("author"))
        if not m:
            raise GitError(f"fatal: --author '{o.get('author')}' is not 'Name <email>' and matches no existing author")
        from .objects import Person
        author = Person(m.group(1), m.group(2), author.when, author.tz)
    if o.get("date"):
        from .objects import Person
        from .repo import parse_git_date
        w, tz = parse_git_date(o.get("date"), repo)
        author = Person(author.name, author.email, w, tz or repo.tz(w))
    if (o.get("reuse-message") or o.get("reedit-message")) and not o.get("reset-author"):
        src = resolve(repo, o.get("reuse-message") or o.get("reedit-message"), "commit")
        author = db.commit(src).author
    committer = repo.ident("committer")
    if o.get("dry-run"):
        return 0
    sha = repo.store("commit", encode_commit(tree, parents, author, committer, msg))
    branch = repo.head_branch()
    subject = db.commit(sha).subject
    if o.get("amend"):
        reflog = f"commit (amend): {subject}"
    elif not parents:
        reflog = f"commit (initial): {subject}"
    elif state == "merge":
        reflog = f"commit (merge): {subject}"
    elif state == "cherry-pick":
        reflog = f"commit (cherry-pick): {subject}"
    else:
        reflog = f"commit: {subject}"
    if repo.head_ref():
        repo.update_ref(repo.head_ref(), sha, reflog)
    else:
        repo.set_head_detached(sha, reflog)
    for f in ("MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "CHERRY_PICK_HEAD", "REVERT_HEAD", "AUTO_MERGE"):
        repo.remove(f)
    if state in ("cherry-pick", "revert"):
        repo.rmtree("sequencer")
    repo.write_index()
    if not o.get("quiet"):
        label = branch or "detached HEAD"
        reused_author = bool(o.get("amend") or o.get("reuse-message") or o.get("reedit-message") or o.get("date"))
        g.out(commit_summary(g, repo, sha, parent_tree, label, show_date=reused_author, initial=not head))
    yield from run_hook(g, repo, "post-commit")
    return 0


def _ctime(p):
    from datetime import datetime, timedelta, timezone
    sign = 1 if p.tz[0] == "+" else -1
    off = sign * (int(p.tz[1:3]) * 60 + int(p.tz[3:5]))
    d = datetime.fromtimestamp(p.when, timezone(timedelta(minutes=off)))
    return d.strftime("%a %b ") + f"{d.day} " + d.strftime("%H:%M:%S %Y ") + p.tz


# ---------------------------------------------------------------------------- restore / reset

@command("restore", wt=True)
def git_restore(g, args):
    o = parse(args, "source|s= staged|S worktree|W patch|p ours theirs merge|m quiet|q overlay ignore-unmerged "
                    "progress conflict=")
    repo = g.repo(wt=True)
    specs = o.args + (o.paths or [])
    if not specs:
        raise GitError("fatal: you must specify path(s) to restore")
    staged = bool(o.get("staged"))
    worktree = bool(o.get("worktree")) or not staged
    db = repo.odb
    idx = repo.index
    if o.get("source"):
        try:
            src_tree = resolve(repo, o.get("source"), "tree")
        except GitError:
            raise GitError(f"fatal: could not resolve {o.get('source')}")
        src = db.flat(src_tree)
    elif staged:
        src = db.flat(repo.head_tree())
    else:
        src = None   # the index
    match = pathspec_matcher(g, repo, specs)
    if o.get("patch"):
        from .interactive import restore_patch
        return (yield from restore_patch(g, repo, specs, staged))
    for spec in specs:
        m1 = pathspec_matcher(g, repo, [spec])
        pool = set(src or {}) | set(idx.stage0()) | {p for p, _ in idx.entries}
        if staged and src is not None:
            pool |= set(db.flat(repo.head_tree()))
        if not any(m1(p) for p in pool):
            raise GitError(f"error: pathspec '{spec}' did not match any file(s) known to git", 1)
    conflicts = idx.conflicts()
    if staged:
        for p in sorted(set(idx.stage0()) | set(src or {}) | set(conflicts)):
            if not match(p):
                continue
            s = (src or {}).get(p)
            if s is None:
                idx.remove(p)
            else:
                old = idx.get(p)
                e = Entry(p, s[1], int(s[0], 8))
                if old is not None and old.sha == s[1] and old.mode == int(s[0], 8):
                    e = old
                idx.add(e)
    if worktree:
        table = src if src is not None else {p: (e.mode_str, e.sha) for p, e in idx.stage0().items()}
        for p in sorted(set(table) | (set(idx.stage0()) if src is not None and not staged else set()) |
                        set(conflicts)):
            if not match(p):
                continue
            if p in conflicts and src is None:
                stg = conflicts[p]
                if o.get("ours") or o.get("theirs"):
                    want = 2 if o.get("ours") else 3
                    if want in stg:
                        repo.checkout_file(p, stg[want].sha, stg[want].mode)
                    continue
                if o.get("merge") or o.get("conflict"):
                    from .merge import merge_text
                    b = db.raw(stg[1].sha) if 1 in stg else b""
                    merged, _ = merge_text(b, db.raw(stg[2].sha), db.raw(stg[3].sha), "ours", "theirs",
                                           style=o.get("conflict") or repo.config.get("merge.conflictStyle", "merge"))
                    node = g.fs.write_file(repo.wt_path(p), merged, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
                    continue
                raise GitError(f"error: path '{p}' is unmerged", 1)
            ent = table.get(p)
            if ent is None:
                if src is not None and not o.get("overlay"):
                    repo.remove_file(p)
                continue
            repo.checkout_file(p, ent[1], ent[0])
            node = repo.wt_node(p)
            if src is None or staged:
                ie = idx.get(p)
                if ie is not None and ie.sha == ent[1] and node is not None:
                    idx.add(repo.index_entry_for(p, ent[1], int(ent[0], 8), node))
    repo.write_index()
    return 0


@command("reset")
def git_reset(g, args):
    o = parse(args, "soft mixed hard merge keep quiet|q patch|p intent-to-add|N recurse-submodules")
    repo = g.repo()
    db = repo.odb
    rest = o.args
    paths = o.paths
    target_spec = "HEAD"
    if rest and paths is None:
        if try_resolve(repo, rest[0], "commit") is not None and not (repo.worktree and repo.wt_node(g.to_rel(repo, rest[0])) is not None and try_resolve(repo, rest[0], "commit") is None):
            target_spec, rest = rest[0], rest[1:]
        if rest:
            paths = rest
    elif rest:
        target_spec = rest[0]
    if paths:
        if o.get("soft") or o.get("hard"):
            mode = "soft" if o.get("soft") else "hard"
            raise GitError(f"fatal: Cannot do {mode} reset with paths.")
        try:
            tree = resolve(repo, target_spec, "tree")
        except GitError:
            raise unknown_revision(target_spec)
        if o.get("patch"):
            from .interactive import restore_patch
            return (yield from restore_patch(g, repo, paths, True))
        flat = db.flat(tree)
        match = pathspec_matcher(g, repo, paths)
        idx = repo.index
        for p in sorted(set(flat) | set(idx.paths())):
            if not match(p):
                continue
            s = flat.get(p)
            if s is None:
                idx.remove(p)
            else:
                old = idx.get(p)
                idx.add(old if old is not None and old.sha == s[1] and old.mode == int(s[0], 8) else Entry(p, s[1], int(s[0], 8)))
        repo.write_index()
        if not o.get("quiet"):
            _unstaged_after_reset(g, repo)
        return 0
    try:
        target = resolve(repo, target_spec, "commit")
    except GitError:
        raise unknown_revision(target_spec)
    mode = "hard" if o.get("hard") else "soft" if o.get("soft") else "keep" if o.get("keep") else \
        "merge" if o.get("merge") else "mixed"
    if mode != "soft" and repo.bare:
        raise GitError(f"fatal: {mode} reset is not allowed in a bare repository")
    old = repo.head()
    if old:
        repo.write("ORIG_HEAD", old + "\n")
    state, _ = in_progress(repo)
    if mode == "soft" and state == "merge":
        raise GitError("fatal: Cannot do a soft reset in the middle of a merge.")
    if mode == "hard":
        from .switching import checkout_tree
        checkout_tree(g, repo, db.commit(target).tree, force=True)
    elif mode in ("keep", "merge"):
        from .switching import checkout_tree
        checkout_tree(g, repo, db.commit(target).tree, force=False)
    elif mode == "mixed":
        repo.reset_index_to(db.commit(target).tree)
        repo.write_index()
    repo.update_ref("HEAD", target, f"reset: moving to {target_spec}")
    for f in ("MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "CHERRY_PICK_HEAD", "REVERT_HEAD", "AUTO_MERGE"):
        repo.remove(f)
    if mode == "hard":
        if not o.get("quiet"):
            c = db.commit(target)
            g.out(f"HEAD is now at {db.abbrev(target)} {c.subject}\n")
    elif mode == "mixed" and not o.get("quiet"):
        _unstaged_after_reset(g, repo)
    return 0


def _unstaged_after_reset(g, repo):
    st = collect(repo, untracked="no")
    if st.unstaged:
        g.out("Unstaged changes after reset:\n" + "".join(f"{st.unstaged[p]}\t{p}\n" for p in sorted(st.unstaged)))


# ---------------------------------------------------------------------------- clean

@command("clean", wt=True)
def git_clean(g, args):
    o = parse(args, "dry-run|n force|f d x X quiet|q interactive|i exclude|e=")
    repo = g.repo(wt=True)
    if not o.get("dry-run") and not o.get("force") and not o.get("interactive") and \
            repo.config.get_bool("clean.requireForce", True):
        raise GitError("fatal: clean.requireForce defaults to true and neither -i, -n, nor -f given; refusing to clean")
    st = collect(repo, untracked="normal", ignored=bool(o.get("x") or o.get("X")))
    victims = []
    if not o.get("X"):
        victims += st.untracked
    if o.get("x") or o.get("X"):
        victims += st.ignored
    match = pathspec_matcher(g, repo, o.args + (o.paths or []))
    out = []
    for p in sorted(victims):
        if match and not match(p.rstrip("/")):
            continue
        if p.endswith("/") and not o.get("d"):
            continue
        for e in o.all("exclude"):
            import fnmatch
            if fnmatch.fnmatch(p.rstrip("/").rsplit("/", 1)[-1], e):
                break
        else:
            out.append(p)
    for p in out:
        shown = g.shown(repo, p)
        if o.get("dry-run"):
            g.out(f"Would remove {shown}\n")
        else:
            if not o.get("quiet"):
                g.out(f"Removing {shown}\n")
            repo.remove_file(p.rstrip("/"))
    return 0

"""branch, merge, cherry-pick, revert, rebase (with -i), stash, tag."""

from __future__ import annotations

import re
import shlex

from .cli import Usage, command, launch_editor, parse, pathspec_matcher, run_maybe
from .cmd_basic import ahead_behind, cleanup_message, commit_summary, in_progress, tracking_lines
from .diff import Colors, compare, numstat, stat_block, stat_name
from .merge import merge_trees
from .objects import EMPTY_TREE, GitError, Person, encode_commit, encode_tag
from .repo import Entry
from .revparse import dwim_ref, resolve, try_resolve, unknown_revision
from .status import collect, worktree_flat
from .switching import check_branch_name, checkout_tree, create_branch, set_upstream, switch_to

DEFAULT_BRANCHES = ("main", "master")


# ---------------------------------------------------------------------------- branch

@command("branch")
def git_branch(g, args):
    o = parse(args, "delete|d D force|f move|m M copy|c C list|l all|a remotes|r verbose|v=? vv merged=? "
                    "no-merged=? contains=? no-contains=? show-current set-upstream-to|u= unset-upstream track|t=? "
                    "no-track sort= format= column no-column ignore-case|i points-at= quiet|q color=? no-color "
                    "abbrev= edit-description")
    repo = g.repo()
    db = repo.odb
    a = o.args
    color = Colors(g.color("branch") and not o.get("no-color"))
    uid, gid = g.ctx.uid, g.ctx.proc.gid
    if o.get("show-current"):
        b = repo.head_branch()
        if b:
            g.out(b + "\n")
        return 0
    if o.get("delete") or o.get("D"):
        if not a:
            raise GitError("fatal: branch name required")
        force = bool(o.get("D") or o.get("force"))
        st = 0
        for name in a:
            if o.get("remotes"):
                ref = f"refs/remotes/{name}"
                sha = repo.resolve_ref(ref)
                if not sha:
                    g.err(f"error: remote-tracking branch '{name}' not found\n")
                    st = 1
                    continue
                repo.delete_ref(ref)
                g.out(f"Deleted remote-tracking branch {name} (was {db.abbrev(sha)}).\n")
                continue
            ref = f"refs/heads/{name}"
            sha = repo.resolve_ref(ref)
            if not sha:
                g.err(f"error: branch '{name}' not found\n")
                st = 1
                continue
            if repo.head_branch() == name:
                g.err(f"error: cannot delete branch '{name}' used by worktree at '{repo.worktree or repo.gitdir}'\n")
                st = 1
                continue
            if not force:
                up = repo.upstream_ref(name)
                base = repo.resolve_ref(up) if up else None
                base = base or repo.head()
                if base and not db.is_ancestor(sha, base):
                    g.err(f"error: the branch '{name}' is not fully merged.\n"
                          f"hint: If you are sure you want to delete it, run 'git branch -D {name}'\n"
                          "hint: Disable this message with \"git config advice.forceDeleteBranch false\"\n")
                    st = 1
                    continue
            repo.delete_ref(ref)
            cfg = repo.config.local_file()
            if cfg:
                cfg.remove_section(f"branch.{name}", uid, gid)
                repo.reload_config()
            if not o.get("quiet"):
                g.out(f"Deleted branch {name} (was {db.abbrev(sha)}).\n")
        return st
    if o.get("move") or o.get("M") or o.get("copy") or o.get("C"):
        copy = bool(o.get("copy") or o.get("C"))
        force = bool(o.get("M") or o.get("C") or o.get("force"))
        if len(a) == 1:
            old, new = repo.head_branch(), a[0]
        elif len(a) == 2:
            old, new = a
        else:
            raise GitError("fatal: branch name required")
        if not old or not repo.resolve_ref(f"refs/heads/{old}") and not (old == repo.head_branch() and not repo.head()):
            raise GitError(f"error: refname refs/heads/{old} not found\nfatal: Branch rename failed" if not copy else
                           f"fatal: no branch named '{old}'")
        check_branch_name(new)
        if repo.resolve_ref(f"refs/heads/{new}") and not force and new != old:
            raise GitError(f"fatal: a branch named '{new}' already exists")
        sha = repo.resolve_ref(f"refs/heads/{old}")
        if sha:
            log = repo.read(f"logs/refs/heads/{old}", True)
            repo.update_ref(f"refs/heads/{new}", sha, None)
            if log is not None:
                repo.write(f"logs/refs/heads/{new}", log, True)
            repo.reflog_append(f"refs/heads/{new}", sha, sha,
                               f"Branch: {'copied' if copy else 'renamed'} refs/heads/{old} to refs/heads/{new}")
            if not copy:
                repo.delete_ref(f"refs/heads/{old}")
        cfg = repo.config.local_file()
        if cfg and not copy:
            cfg.rename_section(f"branch.{old}", f"branch.{new}", uid, gid)
            repo.reload_config()
        if not copy and repo.head_branch() == old:
            repo.write("HEAD", f"ref: refs/heads/{new}\n")
            if sha:
                repo.reflog_append("HEAD", sha, sha, f"Branch: renamed refs/heads/{old} to refs/heads/{new}")
        return 0
    if o.get("set-upstream-to") is not None:
        target = o.get("set-upstream-to")
        branch = a[0] if a else repo.head_branch()
        if not branch:
            raise GitError("fatal: could not set upstream of HEAD to " + target + " when it does not point to any branch.")
        full, sha = dwim_ref(repo, target)
        if not sha:
            raise GitError(f"fatal: the requested upstream branch '{target}' does not exist\n"
                           "hint: \nhint: If you are planning on basing your work on an upstream\n"
                           "hint: branch that already exists at the remote, you may need to\n"
                           "hint: run \"git fetch\" to retrieve it.\nhint: \n"
                           "hint: If you are planning to push out a new local branch that\n"
                           "hint: will track its remote counterpart, you may want to use\n"
                           "hint: \"git push -u\" to set the upstream config as you push.\n"
                           "hint: Disable this message with \"git config advice.setUpstreamFailure false\"")
        if full.startswith("refs/remotes/"):
            rest = full[len("refs/remotes/"):]
            remote, _, rb = rest.partition("/")
            set_upstream(repo, branch, remote, f"refs/heads/{rb}")
            g.out(f"branch '{branch}' set up to track '{remote}/{rb}'.\n")
        else:
            set_upstream(repo, branch, ".", full)
            g.out(f"branch '{branch}' set up to track '{full[len('refs/heads/'):]}'.\n")
        return 0
    if o.get("unset-upstream"):
        branch = a[0] if a else repo.head_branch()
        if not repo.upstream(branch):
            raise GitError(f"fatal: branch '{branch}' has no upstream information")
        cfg = repo.config.local_file()
        cfg.unset(f"branch.{branch}.remote", uid, gid)
        cfg.unset(f"branch.{branch}.merge", uid, gid)
        return 0
    if a and not o.get("list") and o.get("merged") is None and o.get("no-merged") is None and \
            o.get("contains") is None:
        name = a[0]
        start = a[1] if len(a) > 1 else None
        track = None if o.get("track") is None else True
        if o.get("no-track"):
            track = False
        sha, up = create_branch(g, repo, name, start, force=bool(o.get("force")), track=track)
        if up and not o.get("quiet"):
            g.out(up)
        return 0
    # listing
    verbose = 2 if o.get("vv") or o.get("verbose") == "v" or (o.lists.get("verbose") and len(o.lists["verbose"]) > 1) \
        else 1 if o.get("verbose") else 0
    rows = []
    cur = repo.head_branch()
    if not o.get("remotes"):
        if not cur and repo.head():
            label = _detached_label(repo)
            rows.append(("*", label, repo.head(), "detached"))
        from .cmd_extra import other_worktree_branches
        elsewhere = other_worktree_branches(repo)
        for name, sha in repo.branches().items():
            rows.append(("*" if name == cur else "+" if name in elsewhere else " ", name, sha, "local"))
    if o.get("remotes") or o.get("all"):
        for name, sha in repo.remote_branches().items():
            if name.endswith("/HEAD"):
                r = repo.read_ref_raw(f"refs/remotes/{name}")
                if r and r[0] == "sym":
                    tgt = r[1][len("refs/remotes/"):]
                    rows.append((" ", name, sha, "symref:" + tgt))
                    continue
            rows.append((" ", name, sha, "remote"))
    pats = a if (o.get("list") and a) else []
    import fnmatch
    filt = []
    for mark, name, sha, kind in rows:
        if pats and not any(fnmatch.fnmatch(name, p) for p in pats):
            continue
        if o.get("merged") is not None:
            base = resolve(repo, o.get("merged") if o.get("merged") is not True else "HEAD", "commit")
            if not db.is_ancestor(sha, base):
                continue
        if o.get("no-merged") is not None:
            base = resolve(repo, o.get("no-merged") if o.get("no-merged") is not True else "HEAD", "commit")
            if db.is_ancestor(sha, base):
                continue
        if o.get("contains") is not None:
            c = resolve(repo, o.get("contains") if o.get("contains") is not True else "HEAD", "commit")
            if not db.is_ancestor(c, sha):
                continue
        filt.append((mark, name, sha, kind))
    if o.get("sort"):
        key = o.get("sort")
        rev = key.startswith("-")
        key = key.lstrip("-")
        if key in ("committerdate", "creatordate"):
            filt.sort(key=lambda r: db.commit(r[2]).committer.when, reverse=rev)
        elif key == "refname":
            filt.sort(key=lambda r: r[1], reverse=rev)
    out = []
    width = max((len(_shown_name(r, o)) for r in filt), default=0)
    for mark, name, sha, kind in filt:
        shown = _shown_name((mark, name, sha, kind), o)
        if kind.startswith("symref:"):
            tgt = kind.split(":", 1)[1]
            out.append(f"  {color('31', shown)} -> {tgt}")
            continue
        col = "32" if mark == "*" else "36" if mark == "+" else ("31" if kind == "remote" else "")
        nm = color(col, shown) if col else shown
        if not verbose:
            out.append(f"{mark} {nm}")
            continue
        pad = " " * (width - len(shown))
        info = ""
        if kind == "local":
            up = repo.upstream(name)
            if up:
                ref = repo.upstream_ref(name)
                short = ref[len("refs/remotes/"):] if ref.startswith("refs/remotes/") else ref[len("refs/heads/"):]
                them = repo.resolve_ref(ref)
                if them is None:
                    st = "gone"
                else:
                    ah, be = ahead_behind(repo, sha, them)
                    st = ", ".join(x for x in ([f"ahead {ah}"] if ah else []) + ([f"behind {be}"] if be else []))
                if verbose == 2:
                    info = f"[{color('34', short)}{': ' + st if st else ''}] "
                elif st:
                    info = f"[{st}] "
        out.append(f"{mark} {nm}{pad} {db.abbrev(sha)} {info}{db.commit(sha).subject}")
    if out:
        g.out("\n".join(out) + "\n")
    return 0


def _shown_name(row, o):
    mark, name, sha, kind = row
    if kind in ("remote",) or kind.startswith("symref:"):
        return f"remotes/{name}" if o.get("all") and not o.get("remotes") else name
    return name


def _detached_label(repo):
    from .cmd_basic import head_line
    line = head_line(repo)
    if line.startswith("HEAD detached"):
        return f"({line})"
    if "rebase" in line:
        b = (repo.read("rebase-merge/head-name") or "").strip().replace("refs/heads/", "")
        return f"(no branch, rebasing {b})"
    if repo.exists("BISECT_START"):
        start = (repo.read("BISECT_START") or "").strip()
        return f"(no branch, bisect started on {start})"
    return "(no branch)"


# ---------------------------------------------------------------------------- applying merge results

def apply_result(g, repo, res, base_flat_for_check, op="merge"):
    """Write a TreeMerge into the index and working tree. Refuses when local changes would be overwritten."""
    db = repo.odb
    head = db.flat(repo.head_tree())
    idx = repo.index
    changing = set()
    for p in set(head) | set(res.flat) | set(res.stages):
        if p in res.stages or head.get(p) != res.flat.get(p):
            changing.add(p)
    st = collect(repo, untracked="all")
    dirty = sorted(p for p in changing if p in st.staged or p in st.unstaged)
    if dirty:
        verb = {"merge": "merge", "cherry-pick": "merge", "revert": "merge", "rebase": "merge"}.get(op, "merge")
        raise GitError(f"error: Your local changes to the following files would be overwritten by {verb}:\n" +
                       "".join(f"\t{g.shown(repo, p)}\n" for p in dirty) +
                       "Please commit your changes or stash them before you merge.\nAborting", 2)
    untracked = set(p.rstrip("/") for p in st.untracked)
    clash = sorted(p for p in changing if p in untracked and p not in head and (p in res.flat or p in res.stages))
    if clash:
        raise GitError("error: The following untracked working tree files would be overwritten by merge:\n" +
                       "".join(f"\t{g.shown(repo, p)}\n" for p in clash) +
                       "Please move or remove them before you merge.\nAborting", 2)
    for p in sorted(changing):
        if p in res.stages:
            idx.remove(p)
            for k, v in res.stages[p].items():
                idx.add(Entry(p, v[1], int(v[0], 8), k))
            if p in res.worktree:
                mode, data = res.worktree[p]
                node = g.fs.write_file(repo.wt_path(p), bytes(data), "/", 0o755 if mode == "100755" else 0o644,
                                       g.ctx.uid, g.ctx.proc.gid)
            elif 2 not in res.stages[p] and repo.wt_node(p) is not None:
                repo.remove_file(p)
            continue
        new = res.flat.get(p)
        if new is None:
            idx.remove(p)
            repo.remove_file(p)
        else:
            repo.checkout_file(p, new[1], new[0])
            idx.add(repo.index_entry_for(p, new[1], int(new[0], 8), repo.wt_node(p)))
    repo.write_index()


def merge_label(repo, spec):
    full, _ = dwim_ref(repo, spec)
    if full and full.startswith("refs/heads/"):
        return f"branch '{full[len('refs/heads/'):]}'"
    if full and full.startswith("refs/remotes/"):
        return f"remote-tracking branch '{full[len('refs/remotes/'):]}'"
    if full and full.startswith("refs/tags/"):
        return f"tag '{full[len('refs/tags/'):]}'"
    return f"commit '{spec}'"


def merge_message(repo, specs):
    labels = [merge_label(repo, s) for s in specs]
    if len(labels) == 1:
        msg = f"Merge {labels[0]}"
    else:
        kinds = {lb.split(" ", 1)[0] for lb in labels}
        if len(kinds) == 1:
            k = labels[0].split(" ", 1)[0]
            names = [lb.split(" ", 1)[1] for lb in labels]
            plural = {"branch": "branches", "tag": "tags", "commit": "commits",
                      "remote-tracking": "remote-tracking branches"}.get(k, k + "s")
            msg = f"Merge {plural} " + ", ".join(names[:-1]) + " and " + names[-1]
        else:
            msg = "Merge " + ", ".join(labels[:-1]) + " and " + labels[-1]
    b = repo.head_branch()
    if b and b not in DEFAULT_BRANCHES:
        msg += f" into {b}"
    return msg


# ---------------------------------------------------------------------------- merge

@command("merge", wt=True)
def git_merge(g, args):
    o = parse(args, "ff no-ff ff-only squash commit no-commit edit|e no-edit message|m= strategy|s= strategy-option|X= "
                    "abort continue quit stat no-stat|n log=? verbose|v quiet|q allow-unrelated-histories "
                    "verify-signatures signoff autostash no-verify file|F= into-name= progress")
    repo = g.repo(wt=True)
    db = repo.odb
    state, _ = in_progress(repo)
    if o.get("abort"):
        if not repo.exists("MERGE_HEAD"):
            raise GitError("fatal: There is no merge to abort (MERGE_HEAD missing).")
        orig = (repo.read("ORIG_HEAD") or "").strip() or repo.head()
        checkout_tree(g, repo, db.commit(orig).tree, force=True)
        for f in ("MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "AUTO_MERGE", "SQUASH_MSG"):
            repo.remove(f)
        return 0
    if o.get("quit"):
        for f in ("MERGE_HEAD", "MERGE_MSG", "MERGE_MODE", "AUTO_MERGE"):
            repo.remove(f)
        return 0
    if o.get("continue"):
        if not repo.exists("MERGE_HEAD"):
            raise GitError("fatal: There is no merge in progress (MERGE_HEAD missing).")
        from .cmd_basic import git_commit
        st = git_commit(g, [] if not o.get("no-edit") else ["--no-edit"])
        return (yield from st)
    if state == "merge":
        raise GitError("error: Merging is not possible because you have unmerged files.\nhint: Fix them up in the work "
                       "tree, and then use 'git add/rm <file>'\nhint: as appropriate to mark resolution and make a "
                       "commit.\nfatal: Exiting because of an unresolved conflict." if repo.index.conflicts() else
                       "fatal: You have not concluded your merge (MERGE_HEAD exists).\nPlease, commit your changes "
                       "before you merge.")
    specs = o.args
    if not specs:
        b = repo.head_branch()
        up = repo.upstream_ref(b) if b else None
        if not up:
            raise GitError("fatal: No remote for the current branch.")
        specs = [up[len("refs/remotes/"):] if up.startswith("refs/remotes/") else up]
    theirs = []
    for s in specs:
        sha = try_resolve(repo, s, "commit")
        if sha is None:
            raise GitError(f"merge: {s} - not something we can merge")
        theirs.append(sha)
    head = repo.head()
    favor = None
    for x in o.all("strategy-option"):
        if x in ("ours", "theirs"):
            favor = x
    style = repo.config.get("merge.conflictStyle", "merge")
    ff_cfg = repo.config.get("merge.ff")
    ff = "only" if o.get("ff-only") or ff_cfg == "only" else ("no" if o.get("no-ff") or ff_cfg == "false" else "yes")
    if o.get("ff"):
        ff = "yes"
    them = theirs[0]
    if not head:
        # merging into an unborn branch: just take theirs
        checkout_tree(g, repo, db.commit(them).tree)
        repo.update_ref(repo.head_ref(), them, f"merge {specs[0]}: Fast-forward")
        return 0
    if db.is_ancestor(them, head):
        g.out("Already up to date.\n")
        return 0
    bases = db.merge_bases(head, them)
    if not bases and not o.get("allow-unrelated-histories"):
        raise GitError("fatal: refusing to merge unrelated histories")
    stat = not o.get("no-stat") and repo.config.get_bool("merge.stat", True) and not o.get("quiet")
    if o.get("quiet"):
        g = _Quiet(g)
    if bases and bases[0] == head and ff != "no" and not o.get("squash"):
        checkout_tree(g, repo, db.commit(them).tree, op="merge")
        repo.write("ORIG_HEAD", head + "\n")
        g.out(f"Updating {db.abbrev(head)}..{db.abbrev(them)}\nFast-forward\n")
        if stat:
            g.out(_diffstat(repo, db.commit(head).tree, db.commit(them).tree))
        repo.update_ref("HEAD", them, f"merge {specs[0]}: Fast-forward")
        return 0
    if ff == "only":
        raise GitError("fatal: Not possible to fast-forward, aborting.")
    base_tree = db.commit(bases[0]).tree if bases else EMPTY_TREE
    label_t = specs[0]
    res = merge_trees(db, base_tree, db.commit(head).tree, db.commit(them).tree, "HEAD", label_t,
                      db.abbrev(bases[0]) if bases else "empty tree", style, favor)
    repo.write("ORIG_HEAD", head + "\n")
    apply_result(g, repo, res, None)
    for m in res.messages:
        g.out(m + "\n")
    msg = o.get("message") or merge_message(repo, specs)
    if o.get("squash"):
        repo.write("SQUASH_MSG", "Squashed commit of the following:\n\n" + "".join(
            f"commit {c}\nAuthor: {db.commit(c).author.name} <{db.commit(c).author.email}>\n\n    "
            f"{db.commit(c).subject}\n\n" for c in db.ancestors([them], [head])))
        if res.conflicts:
            g.out("Squash commit -- not updating HEAD\nAutomatic merge failed; fix conflicts and then commit the result.\n")
            return 1
        g.out("Squash commit -- not updating HEAD\nAutomatic merge went well; stopped before committing as requested\n")
        return 0
    repo.write("MERGE_HEAD", "".join(t + "\n" for t in theirs))
    repo.write("MERGE_MODE", "no-ff\n" if ff == "no" else "")
    if res.conflicts:
        repo.write("MERGE_MSG", msg + "\n\n# Conflicts:\n" + "".join(f"#\t{p}\n" for p in res.conflicts))
        g.out("Automatic merge failed; fix conflicts and then commit the result.\n")
        return 1
    repo.write("MERGE_MSG", msg + "\n")
    if o.get("no-commit"):
        g.out("Automatic merge went well; stopped before committing as requested\n")
        return 0
    want_edit = o.get("edit") or (not o.get("no-edit") and not o.get("message") and g.ctx.stdin_is_tty()
                                  and g.ctx.stdout_is_tty() and g.env.get("GIT_MERGE_AUTOEDIT", "yes") != "no")
    final = msg + "\n"
    if want_edit:
        text = final + ("\n# Please enter a commit message to explain why this merge is necessary,\n"
                        "# especially if it merges an updated upstream into a topic branch.\n#\n"
                        "# Lines starting with '#' will be ignored, and an empty message aborts\n# the commit.\n")
        repo.write("MERGE_MSG", text)
        st = yield from launch_editor(g, repo.path("MERGE_MSG"))
        if st:
            raise GitError("Not committing merge; use 'git commit' to complete the merge.", 1)
        final = cleanup_message(repo.read("MERGE_MSG") or "")
        if not final.strip():
            g.err("Not committing merge; use 'git commit' to complete the merge.\n")
            return 1
    tree = db.write_tree(repo.index.flat())
    sha = repo.store("commit", encode_commit(tree, [head] + theirs, repo.ident("author"), repo.ident("committer"), final))
    repo.update_ref("HEAD", sha, f"merge {specs[0]}: Merge made by the 'ort' strategy.")
    for f in ("MERGE_HEAD", "MERGE_MSG", "MERGE_MODE"):
        repo.remove(f)
    g.out("Merge made by the 'ort' strategy.\n")
    if stat:
        g.out(_diffstat(repo, db.commit(head).tree, tree))
    return 0


class _Quiet:
    """A G that drops stdout (merge -q)."""

    def __init__(self, g):
        self._g = g

    def __getattr__(self, k):
        return getattr(self._g, k)

    def out(self, text):
        pass


def _diffstat(repo, a_tree, b_tree):
    db = repo.odb
    changes = compare(db.flat(a_tree), db.flat(b_tree), renames=True, db=db)
    rows = []
    for c in changes:
        x, y = numstat(c, db)
        rows.append((stat_name(c), x or 0, y or 0, None if x is not None else (0, 0)))
    lines = stat_block(rows, 80)
    for c in changes:
        if c.status == "A":
            lines.append(f" create mode {c.new_mode} {c.new_path}")
        elif c.status == "D":
            lines.append(f" delete mode {c.old_mode} {c.old_path}")
        elif c.status == "R":
            lines.append(f" rename {stat_name(c)} ({c.score}%)")
    return "\n".join(lines) + "\n" if lines else ""


# ---------------------------------------------------------------------------- the sequencer: pick / revert

def pick(g, repo, sha, mode="pick", mainline=None, record_origin=False, no_commit=False, edit=False,
         label=None, squash_msg=None, keep_author=True, quiet=False, reflog="cherry-pick"):
    """Apply one commit (or its inverse) to HEAD. -> ("ok", new sha) | ("conflict", None) | ("empty", None)."""
    db = repo.odb
    c = db.commit(sha)
    head = repo.head()
    if len(c.parents) > 1 and not mainline:
        raise GitError(f"error: commit {sha} is a merge but no -m option was given.\nfatal: {mode} failed")
    parent = c.parents[(mainline or 1) - 1] if c.parents else None
    ptree = db.commit(parent).tree if parent else EMPTY_TREE
    short = f"{db.abbrev(sha)}... {c.subject}"
    if mode == "revert":
        base, theirs_tree = c.tree, ptree
        lt = f"parent of {db.abbrev(sha)} ({c.subject})"
        lb = f"{db.abbrev(sha)} ({c.subject})"
    else:
        base, theirs_tree = ptree, c.tree
        lt = f"{db.abbrev(sha)} ({c.subject})"
        lb = f"parent of {db.abbrev(sha)} ({c.subject})"
    style = repo.config.get("merge.conflictStyle", "merge")
    res = merge_trees(db, base, repo.head_tree(), theirs_tree, "HEAD", label or lt, lb, style)
    apply_result(g, repo, res, None, op=mode)
    if mode == "revert":
        msg = f'Revert "{c.subject}"\n\nThis reverts commit {sha}'
        msg += (f", reversing\nchanges made to {parent}.\n" if len(c.parents) > 1 else ".\n")
    else:
        msg = squash_msg if squash_msg is not None else c.message
        if record_origin:
            msg = msg.rstrip("\n") + f"\n\n(cherry picked from commit {sha})\n"
    for m in res.messages:
        if not quiet or m.startswith("CONFLICT"):
            g.out(m + "\n")
    if res.conflicts:
        repo.write("MERGE_MSG", msg + "\n# Conflicts:\n" + "".join(f"#\t{p}\n" for p in res.conflicts))
        if mode == "revert":
            repo.write("REVERT_HEAD", sha + "\n")
        elif mode == "pick":
            repo.write("CHERRY_PICK_HEAD", sha + "\n")
        return "conflict", short
    tree = db.write_tree(repo.index.flat())
    if tree == repo.head_tree() and mode != "revert":
        repo.write("MERGE_MSG", msg)
        return "empty", short
    if no_commit:
        repo.write("MERGE_MSG", msg)
        return "nocommit", short
    if edit:
        repo.write("COMMIT_EDITMSG", msg + "\n# Please enter the commit message for your changes. Lines starting\n"
                   "# with '#' will be ignored, and an empty message aborts the commit.\n")
        st = yield from launch_editor(g, repo.path("COMMIT_EDITMSG"))
        if st:
            raise GitError("error: there was a problem with the editor", 1)
        msg = cleanup_message(repo.read("COMMIT_EDITMSG") or "")
    author = c.author if (keep_author and mode != "revert") else repo.ident("author")
    new = repo.store("commit", encode_commit(tree, [head] if head else [], author, repo.ident("committer"), msg))
    subject = db.commit(new).subject
    repo.update_ref("HEAD", new, f"{reflog}: {subject}")
    return "ok", new


def _pick_summary(g, repo, new, parent_tree):
    db = repo.odb
    c = db.commit(new)
    # the sequencer's summaries always show the author date
    return commit_summary(g, repo, new, parent_tree, repo.head_branch() or "detached HEAD", show_date=True)


def _expand(repo, specs):
    db = repo.odb
    out = []
    for s in specs:
        if ".." in s:
            inc, exc = [], []
            a, _, b = s.partition("..")
            inc.append(resolve(repo, b or "HEAD", "commit"))
            exc.append(resolve(repo, a or "HEAD", "commit"))
            out += list(reversed(db.ancestors(inc, exc)))
        else:
            sha = try_resolve(repo, s, "commit")
            if sha is None:
                raise GitError(f"fatal: bad revision '{s}'")
            out.append(sha)
    return out


@command("cherry-pick", wt=True)
def git_cherry_pick(g, args):
    o = parse(args, "x edit|e no-commit|n mainline|m= continue abort skip quit signoff|s ff allow-empty "
                    "keep-redundant-commits strategy= strategy-option|X= no-edit")
    repo = g.repo(wt=True)
    return (yield from _sequence(g, repo, o, "pick"))


@command("revert", wt=True)
def git_revert(g, args):
    o = parse(args, "edit|e no-edit no-commit|n mainline|m= continue abort skip quit signoff|s strategy= "
                    "strategy-option|X= reference")
    repo = g.repo(wt=True)
    return (yield from _sequence(g, repo, o, "revert"))


def _sequence(g, repo, o, mode):
    db = repo.odb
    name = "cherry-pick" if mode == "pick" else "revert"
    head_file = "CHERRY_PICK_HEAD" if mode == "pick" else "REVERT_HEAD"
    if o.get("abort"):
        if not repo.exists(head_file) and not repo.exists("sequencer"):
            raise GitError(f"error: no {name} or revert in progress\nfatal: {name} failed")
        orig = (repo.read("sequencer/head") or repo.read("ORIG_HEAD") or "").strip()
        if orig:
            checkout_tree(g, repo, db.commit(orig).tree, force=True)
            repo.update_ref("HEAD", orig, f"{name}: abort")
        for f in (head_file, "MERGE_MSG", "AUTO_MERGE"):
            repo.remove(f)
        repo.rmtree("sequencer")
        return 0
    if o.get("quit"):
        for f in (head_file, "MERGE_MSG"):
            repo.remove(f)
        repo.rmtree("sequencer")
        return 0
    if o.get("skip"):
        if not repo.exists(head_file):
            raise GitError(f"error: no {name} in progress\nfatal: {name} failed")
        checkout_tree(g, repo, repo.head_tree(), force=True)
        repo.remove(head_file)
        repo.remove("MERGE_MSG")
        todo = [x for x in (repo.read("sequencer/todo") or "").split("\n") if x.strip()][1:]
        if not todo:
            repo.rmtree("sequencer")
            return 0
        repo.write("sequencer/todo", "\n".join(todo) + "\n")
        return (yield from _run_todo(g, repo, mode, o))
    if o.get("continue"):
        if repo.index.conflicts():
            raise GitError("error: Committing is not possible because you have unmerged files.\nhint: Fix them up in the "
                           "work tree, and then use 'git add/rm <file>'\nhint: as appropriate to mark resolution and "
                           f"make a commit.\nfatal: {name} failed", 128)
        if repo.exists(head_file):
            src = (repo.read(head_file) or "").strip()
            c = db.commit(src)
            msg = repo.read("MERGE_MSG") or c.message
            msg = cleanup_message(msg)
            tree = db.write_tree(repo.index.flat())
            head = repo.head()
            ptree = repo.head_tree()
            author = c.author if mode == "pick" else repo.ident("author")
            new = repo.store("commit", encode_commit(tree, [head], author, repo.ident("committer"), msg))
            repo.update_ref("HEAD", new, f"commit ({name}): {db.commit(new).subject}" if mode == "pick" else
                            f"commit: {db.commit(new).subject}")
            repo.remove(head_file)
            repo.remove("MERGE_MSG")
            g.out(_pick_summary(g, repo, new, ptree))
        todo = [x for x in (repo.read("sequencer/todo") or "").split("\n") if x.strip()][1:]
        if not todo:
            repo.rmtree("sequencer")
            return 0
        repo.write("sequencer/todo", "\n".join(todo) + "\n")
        return (yield from _run_todo(g, repo, mode, o))
    if repo.exists(head_file):
        raise GitError(f"error: {name} is already in progress\nhint: try \"git {name} (--continue | --abort | --quit)\"\n"
                       f"fatal: {name} failed")
    if not o.args:
        raise Usage(f"usage: git {name} [--edit] [-n] [-m <parent-number>] [-s] [-x] [--ff]\n"
                    f"                   [-S[<keyid>]] <commit>...")
    commits = _expand(repo, o.args)
    repo.write("ORIG_HEAD", (repo.head() or "") + "\n")
    repo.write("sequencer/head", (repo.head() or "") + "\n")
    repo.write("sequencer/todo", "".join(f"{'pick' if mode == 'pick' else 'revert'} {c} {db.commit(c).subject}\n"
                                         for c in commits))
    return (yield from _run_todo(g, repo, mode, o))


def _run_todo(g, repo, mode, o):
    db = repo.odb
    name = "cherry-pick" if mode == "pick" else "revert"
    while True:
        todo = [x for x in (repo.read("sequencer/todo") or "").split("\n") if x.strip()]
        if not todo:
            repo.rmtree("sequencer")
            return 0
        sha = todo[0].split()[1]
        ptree = repo.head_tree()
        edit = bool(o.get("edit")) or (mode == "revert" and not o.get("no-edit") and g.ctx.stdin_is_tty()
                                       and g.ctx.stdout_is_tty())
        kind, new = yield from pick(g, repo, sha, mode=mode, mainline=int(o.get("mainline")) if o.get("mainline") else None,
                                    record_origin=bool(o.get("x")), no_commit=bool(o.get("no-commit")), edit=edit,
                                    reflog=name)
        if kind == "conflict":
            c = db.commit(sha)
            g.err(f"error: could not {'apply' if mode == 'pick' else 'revert'} {db.abbrev(sha)}... {c.subject}\n"
                  "hint: After resolving the conflicts, mark them with\nhint: \"git add/rm <pathspec>\", then run\n"
                  f"hint: \"git {name} --continue\".\nhint: You can instead skip this commit with \"git {name} --skip\".\n"
                  f"hint: To abort and get back to the state before \"git {name}\",\nhint: run \"git {name} --abort\".\n"
                  "hint: Disable this message with \"git config advice.mergeConflict false\"\n")
            return 1
        if kind == "empty":
            c = db.commit(sha)
            g.out(f"On branch {repo.head_branch()}\nYou are currently cherry-picking commit {db.abbrev(sha)}.\n"
                  f"  (all conflicts fixed: run \"git cherry-pick --continue\")\n"
                  f"  (use \"git cherry-pick --skip\" to skip this patch)\n"
                  f"  (use \"git cherry-pick --abort\" to cancel the cherry-pick operation)\n\nnothing to commit, "
                  "working tree clean\nThe previous cherry-pick is now empty, possibly due to conflict resolution.\n"
                  "If you wish to commit it anyway, use:\n\n    git commit --allow-empty\n\nOtherwise, please use "
                  "'git cherry-pick --skip'\n")
            repo.write("CHERRY_PICK_HEAD", sha + "\n")
            return 1
        if kind == "nocommit":
            pass
        elif kind == "ok":
            g.out(_pick_summary(g, repo, new, ptree))
        repo.write("sequencer/todo", "".join(x + "\n" for x in todo[1:]))


# ---------------------------------------------------------------------------- rebase

TODO_HELP = """
# Rebase {onto_short}..{orig_short} onto {onto_short} ({n} command{s})
#
# Commands:
# p, pick <commit> = use commit
# r, reword <commit> = use commit, but edit the commit message
# e, edit <commit> = use commit, but stop for amending
# s, squash <commit> = use commit, but meld into previous commit
# f, fixup [-C | -c] <commit> = like "squash" but keep only the previous
#                    commit's log message, unless -C is used, in which case
#                    keep only this commit's message; -c is same as -C but
#                    opens the editor
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with 'git rebase --continue')
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
#         create a merge commit using the original merge commit's
#         message (or the oneline, if no original merge commit was
#         specified); use -c <commit> to reword the commit message
# u, update-ref <ref> = track a placeholder for the <ref> to be updated
#                       to this position in the new commits. The <ref> is
#                       updated at the end of the rebase
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
"""
ABBREV = {"p": "pick", "r": "reword", "e": "edit", "s": "squash", "f": "fixup", "x": "exec", "b": "break",
          "d": "drop", "l": "label", "t": "reset", "m": "merge"}


@command("rebase", wt=True)
def git_rebase(g, args):
    o = parse(args, "interactive|i onto= continue skip abort quit edit-todo autosquash no-autosquash exec|x= root "
                    "keep-base strategy|s= strategy-option|X= verbose|v quiet|q stat no-stat|n force-rebase|f "
                    "autostash no-autostash update-refs rebase-merges|r=? apply merge|m signoff committer-date-is-author-date "
                    "ignore-date reset-author-date empty= keep-empty no-keep-empty")
    repo = g.repo(wt=True)
    db = repo.odb
    if o.get("abort"):
        if not repo.exists("rebase-merge"):
            raise GitError("fatal: No rebase in progress?")
        orig = (repo.read("rebase-merge/orig-head") or "").strip()
        head_name = (repo.read("rebase-merge/head-name") or "").strip()
        checkout_tree(g, repo, db.commit(orig).tree, force=True)
        if head_name.startswith("refs/heads/"):
            repo.update_ref(head_name, orig, "rebase (abort): updating " + head_name)
            repo.set_head_symbolic(head_name, f"rebase (abort): returning to {head_name}")
        else:
            repo.set_head_detached(orig, "rebase (abort): returning to " + orig)
        for f in ("CHERRY_PICK_HEAD", "MERGE_MSG", "REBASE_HEAD", "AUTO_MERGE"):
            repo.remove(f)
        repo.rmtree("rebase-merge")
        return 0
    if o.get("quit"):
        repo.rmtree("rebase-merge")
        return 0
    if o.get("edit-todo"):
        if not repo.exists("rebase-merge"):
            raise GitError("fatal: No rebase in progress?")
        st = yield from launch_editor(g, repo.path("rebase-merge/git-rebase-todo"))
        return st
    if o.get("continue") or o.get("skip"):
        if not repo.exists("rebase-merge"):
            raise GitError("fatal: No rebase in progress?")
        if o.get("skip"):
            checkout_tree(g, repo, repo.head_tree(), force=True)
            for f in ("CHERRY_PICK_HEAD", "MERGE_MSG", "REBASE_HEAD", "AUTO_MERGE"):
                repo.remove(f)
            repo.remove("rebase-merge/stopped-sha")
            repo.remove("rebase-merge/amend")
        else:
            if repo.index.conflicts():
                raise GitError("error: you must edit all merge conflicts and then\nmark them as resolved using git add", 1)
            yield from _rebase_commit_pending(g, repo)
        return (yield from _rebase_run(g, repo))
    if repo.exists("rebase-merge"):
        raise GitError("fatal: It seems that there is already a rebase-merge directory, and\nI wonder if you are in the "
                       "middle of another rebase.  If that is the\ncase, please try\n\tgit rebase (--continue | --abort | "
                       "--skip)\nIf that is not the case, please\n\trm -fr \".git/rebase-merge\"\nand run me again.  I "
                       "am stopping in case you still have something\nvaluable there.")
    st = collect(repo, untracked="no")
    if not st.clean and not o.get("autostash"):
        g.err("error: cannot rebase: You have unstaged changes.\nerror: Please commit or stash them.\n" if st.unstaged
              else "error: cannot rebase: Your index contains uncommitted changes.\nerror: Please commit or stash them.\n")
        return 1
    a = o.args
    branch_arg = a[1] if len(a) > 1 else None
    if branch_arg:
        sha = repo.resolve_ref(f"refs/heads/{branch_arg}")
        if not sha:
            raise GitError(f"fatal: invalid upstream '{branch_arg}'")
        switch_to(g, repo, sha, branch=branch_arg, quiet=True)
    head = repo.head()
    head_name = repo.head_ref() or "detached HEAD"
    if o.get("root"):
        upstream = None
    elif a:
        upstream = try_resolve(repo, a[0], "commit")
        if upstream is None:
            raise GitError(f"fatal: invalid upstream '{a[0]}'")
    else:
        b = repo.head_branch()
        up = repo.upstream_ref(b) if b else None
        if not up:
            raise GitError("There is no tracking information for the current branch.\nPlease specify which branch you "
                           "want to rebase against.\nSee git-rebase(1) for details.\n\n    git rebase '<branch>'\n\n"
                           "If you wish to set tracking information for this branch you can do so with:\n\n"
                           f"    git branch --set-upstream-to=<remote>/<branch> {b}\n", 1)
        upstream = repo.resolve_ref(up)
    onto = try_resolve(repo, o.get("onto"), "commit") if o.get("onto") else upstream
    if o.get("onto") and onto is None:
        raise GitError(f"fatal: Does not point to a valid commit '{o.get('onto')}'")
    if o.get("keep-base") and upstream:
        bases = db.merge_bases(upstream, head)
        onto = bases[0] if bases else onto
    if upstream:
        excl = [upstream]
        commits = list(reversed(db.ancestors([head], excl)))
    else:
        commits = list(reversed(db.ancestors([head])))
        onto = None
    commits = [c for c in commits if len(db.commit(c).parents) <= 1]
    # drop commits whose change is already upstream (same patch)
    if upstream:
        up_patches = {_patch_id(db, c) for c in db.ancestors([upstream], [head])}
        kept = []
        for c in commits:
            if _patch_id(db, c) in up_patches:
                g.err(f"warning: skipped previously applied commit {db.abbrev(c)}\n")
                continue
            kept.append(c)
        commits = kept
    interactive = bool(o.get("interactive"))
    if not interactive and not o.get("force-rebase") and onto and db.is_ancestor(onto, head) and \
            (not commits or list(db.commit(commits[0]).parents[:1]) == [onto]):
        g.out(f"Current branch {repo.head_branch() or 'HEAD'} is up to date.\n")
        return 0
    lines = [f"pick {db.abbrev(c)} {db.commit(c).subject}" for c in commits]
    autosquash = (o.get("autosquash") or repo.config.get_bool("rebase.autoSquash", False)) and not o.get("no-autosquash")
    if autosquash:
        lines = _autosquash(db, commits)
    for x in o.all("exec"):
        new = []
        for ln in lines:
            new.append(ln)
            if not ln.split()[0] in ("fixup", "squash"):
                new.append(f"exec {x}")
        lines = new
    repo.write("rebase-merge/head-name", head_name + "\n")
    repo.write("rebase-merge/onto", (onto or "") + "\n")
    repo.write("rebase-merge/orig-head", head + "\n")
    repo.write("ORIG_HEAD", head + "\n")
    if interactive:
        repo.write("rebase-merge/interactive", "")
        help_text = TODO_HELP.format(onto_short=db.abbrev(onto) if onto else "root", orig_short=db.abbrev(head),
                                     n=len(lines), s="s" if len(lines) != 1 else "")
        repo.write("rebase-merge/git-rebase-todo", "\n".join(lines) + "\n" + help_text)
        seq_editor = g.env.get("GIT_SEQUENCE_EDITOR") or repo.config.get("sequence.editor")
        g.ctx._flush_nb()
        if seq_editor:
            st = yield from g.ctx.run_shell(f"{seq_editor} {shlex.quote(repo.path('rebase-merge/git-rebase-todo'))}")
        else:
            st = yield from launch_editor(g, repo.path("rebase-merge/git-rebase-todo"))
        if st:
            repo.rmtree("rebase-merge")
            raise GitError("error: there was a problem with the editor", 1)
        todo = [x for x in (repo.read("rebase-merge/git-rebase-todo") or "").split("\n")
                if x.strip() and not x.strip().startswith("#")]
        if not todo:
            repo.rmtree("rebase-merge")
            g.out("Nothing to do\n")
            return 1
        repo.write("rebase-merge/git-rebase-todo", "\n".join(todo) + "\n")
    else:
        repo.write("rebase-merge/git-rebase-todo", "\n".join(lines) + ("\n" if lines else ""))
    # start: detach at onto
    target = onto or None
    if target:
        checkout_tree(g, repo, db.commit(target).tree, force=False)
        repo.set_head_detached(target, f"rebase (start): checkout {a[0] if a else (o.get('onto') or target)}")
    else:
        checkout_tree(g, repo, EMPTY_TREE, force=True)
        repo.write("HEAD", f"ref: refs/heads/rebase-root-placeholder\n")
        repo.write("rebase-merge/root", "")
    return (yield from _rebase_run(g, repo))


def _patch_id(db, c):
    commit = db.commit(c)
    ptree = db.commit(commit.parents[0]).tree if commit.parents else EMPTY_TREE
    a, b = db.flat(ptree), db.flat(commit.tree)
    return tuple(sorted((p, a.get(p), b.get(p)) for p in set(a) | set(b) if a.get(p) != b.get(p)))


def _autosquash(db, commits):
    lines = []
    pending = list(commits)
    out = []
    by_subject = {}
    for c in commits:
        by_subject.setdefault(db.commit(c).subject, c)
    fixes = {}
    rest = []
    for c in commits:
        s = db.commit(c).subject
        m = re.match(r"^(fixup|squash|amend)! (.*)$", s)
        if m:
            target_subj = m.group(2)
            while re.match(r"^(fixup|squash|amend)! ", target_subj):
                target_subj = target_subj.split("! ", 1)[1]
            t = next((x for x in commits if db.commit(x).subject == target_subj or x.startswith(target_subj)), None)
            if t is not None and t != c:
                kind = "fixup" if m.group(1) in ("fixup",) else ("fixup -C" if m.group(1) == "amend" else "squash")
                fixes.setdefault(t, []).append((kind, c))
                continue
        rest.append(c)
    for c in rest:
        out.append(f"pick {db.abbrev(c)} {db.commit(c).subject}")
        for kind, f in fixes.get(c, []):
            out.append(f"{kind} {db.abbrev(f)} {db.commit(f).subject}")
    return out


def _rebase_commit_pending(g, repo):
    """--continue after a conflict or an edit stop: commit what's staged (keeping the message)."""
    db = repo.odb
    stopped = (repo.read("rebase-merge/stopped-sha") or "").strip()
    pending_msg = repo.read("rebase-merge/message")
    if repo.exists("rebase-merge/amend"):
        # edit stop: the user amends (or not); just continue
        repo.remove("rebase-merge/amend")
        repo.remove("rebase-merge/stopped-sha")
        st = collect(repo, untracked="no")
        if st.staged:
            # staged changes after an edit stop are folded into that commit
            h = repo.head_commit()
            tree = db.write_tree(repo.index.flat())
            new = repo.store("commit", encode_commit(tree, list(h.parents), h.author, repo.ident("committer"), h.message))
            repo.set_head_detached(new, f"rebase (continue): {h.subject}")
        return
    if stopped and repo.exists("CHERRY_PICK_HEAD"):
        c = db.commit(stopped)
        tree = db.write_tree(repo.index.flat())
        msg = pending_msg if pending_msg is not None else c.message
        if tree != repo.head_tree():
            reword = repo.exists("rebase-merge/reword")
            if reword:
                repo.write("COMMIT_EDITMSG", msg)
                yield from launch_editor(g, repo.path("COMMIT_EDITMSG"))
                msg = cleanup_message(repo.read("COMMIT_EDITMSG") or msg)
            new = repo.store("commit", encode_commit(tree, [repo.head()], c.author, repo.ident("committer"), msg))
            repo.set_head_detached(new, f"rebase (continue): {db.commit(new).subject}")
        repo.remove("CHERRY_PICK_HEAD")
        repo.remove("MERGE_MSG")
        repo.remove("REBASE_HEAD")
        repo.remove("rebase-merge/stopped-sha")
        repo.remove("rebase-merge/message")
        repo.remove("rebase-merge/reword")


def _skip_in_place(g, repo):
    """Leading `pick`s whose commit already sits on HEAD are done without replaying (and without progress)."""
    db = repo.odb
    while True:
        todo = [x for x in (repo.read("rebase-merge/git-rebase-todo") or "").split("\n") if x.strip()]
        if not todo:
            return
        parts = todo[0].split(None, 2)
        if ABBREV.get(parts[0], parts[0]) != "pick" or len(parts) < 2:
            return
        sha = try_resolve(repo, parts[1], "commit")
        if sha is None or list(db.commit(sha).parents[:1]) != [repo.head()]:
            return
        repo.write("rebase-merge/done", (repo.read("rebase-merge/done") or "") + todo[0] + "\n")
        repo.write("rebase-merge/git-rebase-todo", "".join(x + "\n" for x in todo[1:]))
        checkout_tree(g, repo, db.commit(sha).tree)
        repo.set_head_detached(sha)


def _rebase_run(g, repo):
    db = repo.odb
    _skip_in_place(g, repo)
    while True:
        todo = [x for x in (repo.read("rebase-merge/git-rebase-todo") or "").split("\n") if x.strip()]
        if not todo:
            return (yield from run_maybe(_rebase_finish(g, repo)))
        line = todo[0]
        n_done = len([x for x in (repo.read("rebase-merge/done") or "").split("\n") if x.strip()]) + 1
        g.err(f"Rebasing ({n_done}/{n_done + len(todo) - 1})\r")
        rest_todo = todo[1:]
        parts = line.split(None, 2)
        cmd = ABBREV.get(parts[0], parts[0])
        done = (repo.read("rebase-merge/done") or "") + line + "\n"
        repo.write("rebase-merge/done", done)
        repo.write("rebase-merge/git-rebase-todo", "".join(x + "\n" for x in rest_todo))
        if cmd == "exec":
            command_line = line.split(None, 1)[1]
            g.out(f"Executing: {command_line}\n")
            g.ctx._flush_nb()
            st = yield from g.ctx.run_shell(command_line)
            if st:
                g.err(f"warning: execution failed: {command_line}\nYou can fix the problem, and then run\n\n  git rebase "
                      "--continue\n\n")
                return 1
            continue
        if cmd == "break":
            g.err("Stopped at " + db.abbrev(repo.head()) + "...  " + repo.head_commit().subject + "\n")
            return 0
        if cmd == "drop":
            continue
        if cmd in ("label", "reset", "merge", "update-ref"):
            continue
        fix_c = None
        if cmd == "fixup" and len(parts) > 1 and parts[1] in ("-C", "-c"):
            fix_c = parts[1]
            parts = [parts[0]] + parts[2].split(None, 1)
        sha_s = parts[1] if len(parts) > 1 else ""
        sha = try_resolve(repo, sha_s, "commit")
        if sha is None:
            raise GitError(f"error: invalid line 1: {line}\nYou can fix this with 'git rebase --edit-todo' and then run "
                           "'git rebase --continue'.\nOr you can abort the rebase with 'git rebase --abort'.")
        c = db.commit(sha)
        if cmd in ("squash", "fixup"):
            head = repo.head()
            hc = db.commit(head)
            res = merge_trees(db, db.commit(c.parents[0]).tree if c.parents else EMPTY_TREE, hc.tree, c.tree, "HEAD",
                              f"{db.abbrev(sha)} ({c.subject})")
            apply_result(g, repo, res, None, op="rebase")
            if res.conflicts:
                for m in res.messages:
                    g.out(m + "\n")
                repo.write("CHERRY_PICK_HEAD", sha + "\n")
                repo.write("rebase-merge/stopped-sha", sha + "\n")
                _conflict_hint(g, db, sha)
                return 1
            import json
            chain = json.loads(repo.read("rebase-merge/squash-msgs") or "null") or [["keep", hc.message]]
            kind = "squash" if cmd == "squash" else ("fixup-C" if fix_c else "fixup")
            chain.append([kind, c.message])
            nxt = rest_todo[0].split()[0] if rest_todo else ""
            last = ABBREV.get(nxt, nxt) not in ("squash", "fixup")
            tree = db.write_tree(repo.index.flat())
            edit = any(k == "squash" for k, _ in chain) or fix_c == "-c"
            text = _squash_text(chain)
            if last and edit:
                repo.write("COMMIT_EDITMSG", text + "\n# Please enter the commit message for your changes. Lines "
                           "starting\n# with '#' will be ignored, and an empty message aborts the commit.\n")
                yield from launch_editor(g, repo.path("COMMIT_EDITMSG"))
                msg = cleanup_message(repo.read("COMMIT_EDITMSG") or text)
            else:
                msg = cleanup_message(text)
            new = repo.store("commit", encode_commit(tree, list(hc.parents), hc.author, repo.ident("committer"), msg))
            repo.set_head_detached(new, f"rebase ({cmd}): {db.commit(new).subject}")
            if last:
                repo.remove("rebase-merge/squash-msgs")
                if edit:      # committed through the editor: git prints the summary
                    ptree = db.commit(hc.parents[0]).tree if hc.parents else None
                    g.out(commit_summary(g, repo, new, ptree, "detached HEAD", show_date=True))
            else:
                repo.write("rebase-merge/squash-msgs", json.dumps(chain))
            continue
        # pick / reword / edit
        parent_head = repo.head()
        if c.parents and c.parents[0] == parent_head and cmd == "pick" and not repo.exists("rebase-merge/force"):
            # fast-forward over an unchanged commit
            checkout_tree(g, repo, c.tree)
            repo.set_head_detached(sha, f"rebase (pick): {c.subject}")
            continue
        kind, new = yield from pick(g, repo, sha, mode="pick", reflog="rebase (pick)", quiet=True)
        if kind == "conflict":
            repo.write("rebase-merge/stopped-sha", sha + "\n")
            repo.write("REBASE_HEAD", sha + "\n")
            if cmd == "reword":
                repo.write("rebase-merge/reword", "")
            _conflict_hint(g, db, sha)
            return 1
        if kind == "empty":
            repo.remove("MERGE_MSG")
            continue
        if cmd == "reword":
            hc = repo.head_commit()
            repo.write("COMMIT_EDITMSG", hc.message + "\n# Please enter the commit message for your changes. Lines "
                       "starting\n# with '#' will be ignored, and an empty message aborts the commit.\n")
            yield from launch_editor(g, repo.path("COMMIT_EDITMSG"))
            msg = cleanup_message(repo.read("COMMIT_EDITMSG") or hc.message)
            new2 = repo.store("commit", encode_commit(hc.tree, list(hc.parents), hc.author, repo.ident("committer"), msg))
            repo.set_head_detached(new2, f"rebase (reword): {db.commit(new2).subject}")
        if cmd == "edit":
            repo.write("rebase-merge/stopped-sha", sha + "\n")
            repo.write("rebase-merge/amend", repo.head() + "\n")
            g.err(f"Stopped at {db.abbrev(sha)}...  {c.subject}\nYou can amend the commit now, with\n\n  git commit "
                  "--amend \n\nOnce you are satisfied with your changes, run\n\n  git rebase --continue\n")
            return 0


def _ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _squash_text(chain):
    """The message a squash chain proposes (git's layout): kept messages, skipped (fixup) ones commented out."""
    if any(k == "fixup-C" for k, _ in chain):
        return [m for k, m in chain if k == "fixup-C"][-1]
    parts = [f"# This is a combination of {len(chain)} commits."]
    for i, (k, m) in enumerate(chain, 1):
        head = f"# This is the {_ordinal(i)} commit message:" if i == 1 else f"# This is the commit message #{i}:"
        if k == "fixup":
            head = f"# The commit message #{i} will be skipped:"
            body = "\n".join("# " + ln if ln else "#" for ln in m.rstrip("\n").split("\n"))
        else:
            body = m.rstrip("\n")
        parts.append(f"{head}\n\n{body}")
    return "\n\n".join(parts) + "\n"


def _conflict_hint(g, db, sha):
    c = db.commit(sha)
    g.err(f"error: could not apply {db.abbrev(sha)}... {c.subject}\n"
          "hint: Resolve all conflicts manually, mark them as resolved with\n"
          "hint: \"git add/rm <conflicted_files>\", then run \"git rebase --continue\".\n"
          "hint: You can instead skip this commit: run \"git rebase --skip\".\n"
          "hint: To abort and get back to the state before \"git rebase\", run \"git rebase --abort\".\n"
          f"Could not apply {db.abbrev(sha)}... {c.subject}\n")


def _rebase_finish(g, repo):
    db = repo.odb
    head_name = (repo.read("rebase-merge/head-name") or "").strip()
    onto = (repo.read("rebase-merge/onto") or "").strip()
    new = repo.head()
    if head_name.startswith("refs/heads/"):
        repo.update_ref(head_name, new, f"rebase (finish): {head_name} onto {onto}")
        repo.write("HEAD", f"ref: {head_name}\n")
        repo.reflog_append("HEAD", new, new, f"rebase (finish): returning to {head_name}")
    repo.rmtree("rebase-merge")
    for f in ("REBASE_HEAD", "AUTO_MERGE"):
        repo.remove(f)
    clear = "\r" + " " * 80 + "\r" if g.env.get("TERM", "dumb") == "dumb" else "\r\x1b[K"
    g.err(f"{clear}Successfully rebased and updated {head_name}.\n")
    return 0


# ---------------------------------------------------------------------------- stash

def _stash_list(repo):
    return repo.reflog("refs/stash")


def _stash_ref(repo, spec):
    """'stash@{1}' / '1' / None -> (index, sha)."""
    entries = _stash_list(repo)
    if not entries:
        raise GitError("No stash entries found.", 1)
    n = 0
    if spec:
        m = re.match(r"^(?:stash)?@?\{?(\d+)\}?$", spec.replace("stash@", "@")) or re.match(r"^(\d+)$", spec)
        if not m:
            raise GitError(f"error: {spec} is not a valid reference", 1)
        n = int(m.group(1))
    if n >= len(entries):
        raise GitError(f"error: stash@{{{n}}} is not a valid reference" if n else "No stash entries found.", 1)
    return n, entries[n][1]


def _stash_drop(repo, n):
    newest_first = _stash_list(repo)
    remaining = [e for i, e in enumerate(newest_first) if i != n]
    if not remaining:
        repo.delete_ref("refs/stash")
        return
    # rewrite the stash reflog (oldest first) and the ref
    lines = []
    prev = "0" * 40
    for old, new, who, msg in reversed(remaining):
        lines.append(f"{prev} {new} {who.name} <{who.email}> {who.when} {who.tz}\t{msg}\n")
        prev = new
    repo.write("logs/refs/stash", "".join(lines), True)
    repo.write("refs/stash", remaining[0][1] + "\n", True)


@command("stash", wt=True)
def git_stash(g, args):
    repo = g.repo(wt=True)
    db = repo.odb
    sub = args[0] if args and not args[0].startswith("-") else "push"
    rest = args[1:] if args and not args[0].startswith("-") else args
    if sub == "save":
        o = parse(rest, "keep-index|k include-untracked|u all|a quiet|q patch|p")
        return (yield from run_maybe(_stash_push(g, repo, o, " ".join(o.args) or None, [])))
    if sub == "push":
        o = parse(rest, "message|m= keep-index|k include-untracked|u all|a quiet|q patch|p staged|S")
        paths = (o.paths or []) + o.args
        return (yield from run_maybe(_stash_push(g, repo, o, o.get("message"), paths)))
    if sub == "list":
        out = []
        for i, (old, new, who, msg) in enumerate(_stash_list(repo)):
            out.append(f"stash@{{{i}}}: {msg}\n")
        g.out("".join(out))
        return 0
    if sub == "show":
        o = parse(rest, "patch|p stat include-untracked|u name-only")
        n, sha = _stash_ref(repo, o.args[0] if o.args else None)
        c = db.commit(sha)
        base = db.commit(c.parents[0]).tree
        changes = compare(db.flat(base), db.flat(c.tree), renames=True, db=db)
        color = Colors(g.color("diff"))
        if o.get("patch"):
            from .diff import patch
            g.out("".join(patch(ch, db, color=color) for ch in changes))
        elif o.get("name-only"):
            g.out("".join(ch.path + "\n" for ch in changes))
        else:
            rows = []
            for ch in changes:
                a, d = numstat(ch, db)
                rows.append((stat_name(ch), a or 0, d or 0, None))
            g.out("\n".join(stat_block(rows, g.columns(), color)) + "\n" if rows else "")
        return 0
    if sub in ("pop", "apply"):
        o = parse(rest, "index quiet|q")
        n, sha = _stash_ref(repo, o.args[0] if o.args else None)
        ok = yield from run_maybe(_stash_apply(g, repo, sha, bool(o.get("index")), bool(o.get("quiet"))))
        if not ok:
            if sub == "pop":
                g.out("The stash entry is kept in case you need it again.\n")
            return 1
        if sub == "pop":
            _stash_drop(repo, n)
            g.out(f"Dropped refs/stash@{{{n}}} ({sha})\n")
        return 0
    if sub == "drop":
        o = parse(rest, "quiet|q")
        n, sha = _stash_ref(repo, o.args[0] if o.args else None)
        _stash_drop(repo, n)
        g.out(f"Dropped refs/stash@{{{n}}} ({sha})\n")
        return 0
    if sub == "clear":
        repo.delete_ref("refs/stash")
        return 0
    if sub == "branch":
        if not rest:
            raise GitError("fatal: No branch name specified")
        name = rest[0]
        n, sha = _stash_ref(repo, rest[1] if len(rest) > 1 else None)
        c = db.commit(sha)
        start = c.parents[0]
        sha_b, _ = create_branch(g, repo, name, start)
        switch_to(g, repo, start, branch=name, quiet=True)
        g.err(f"Switched to a new branch '{name}'\n")
        ok = yield from run_maybe(_stash_apply(g, repo, sha, True, False))
        if ok:
            _stash_drop(repo, n)
            g.out(f"Dropped refs/stash@{{{n}}} ({sha})\n")
        return 0 if ok else 1
    if sub == "create":
        return 0
    raise Usage("usage: git stash list [<log-options>]\n   or: git stash show [-u | --include-untracked | "
                "--only-untracked] [<diff-options>] [<stash>]\n   or: git stash drop [-q | --quiet] [<stash>]\n"
                "   or: git stash pop [--index] [-q | --quiet] [<stash>]\n   or: git stash apply [--index] [-q | "
                "--quiet] [<stash>]\n   or: git stash branch <branchname> [<stash>]\n   or: git stash [push "
                "[-p | --patch] [-S | --staged] [-k | --[no-]keep-index] [-q | --quiet]\n          [-u | "
                "--include-untracked] [-a | --all] [(-m | --message) <message>]\n          [--pathspec-from-file=<file> "
                "[--pathspec-file-nul]]\n          [--] [<pathspec>...]]\n   or: git stash save [-p | --patch] [-S | "
                "--staged] [-k | --[no-]keep-index] [-q | --quiet]\n          [-u | --include-untracked] [-a | --all] "
                "[<message>]\n   or: git stash clear\n   or: git stash create [<message>]\n   or: git stash store "
                "[(-m | --message) <message>] [-q | --quiet] <commit>")


def _stash_push(g, repo, o, message, paths):
    db = repo.odb
    head = repo.head()
    if not head:
        raise GitError("You do not have the initial commit yet", 1)
    st = collect(repo, untracked="all" if (o.get("include-untracked") or o.get("all")) else "no",
                 ignored=bool(o.get("all")))
    match = pathspec_matcher(g, repo, paths) if paths else None
    if paths:
        for spec in paths:
            m1 = pathspec_matcher(g, repo, [spec])
            if not any(m1(p) for p in set(repo.index.paths()) | set(st.untracked)):
                raise GitError(f"error: pathspec '{spec}' did not match any file(s) known to git\n"
                               "Did you forget to 'git add'?", 1)
    staged = {p: v for p, v in st.staged.items() if not match or match(p)}
    unstaged = {p: v for p, v in st.unstaged.items() if not match or match(p)}
    untracked = [p for p in st.untracked + (st.ignored if o.get("all") else []) if not match or match(p.rstrip("/"))]
    if not staged and not unstaged and not untracked:
        g.out("No local changes to save\n")
        return 0
    hc = db.commit(head)
    branch = repo.head_branch() or "(no branch)"
    desc = f"{db.abbrev(head)} {hc.subject}"
    idx_flat = repo.index.flat()
    who = repo.ident("committer")
    i_tree = db.write_tree(idx_flat if not o.get("staged") else idx_flat)
    i_commit = repo.store("commit", encode_commit(i_tree, [head], who, who, f"index on {branch}: {desc}\n"))     # (the WIP message itself has no final newline)
    wt = worktree_flat(repo)
    w_flat = dict(idx_flat)
    if not o.get("staged"):
        for p in set(idx_flat) | set(wt):
            if match and not match(p):
                continue
            if p in wt:
                if wt[p] != idx_flat.get(p):
                    sha, mode = repo.wt_sha(p, write=True)
                    w_flat[p] = (f"{mode:o}", sha)
            else:
                w_flat.pop(p, None)
    parents = [head, i_commit]
    u_files = []
    if untracked:
        u_flat = {}
        for p in untracked:
            p = p.rstrip("/")
            for f, node in repo.worktree_files(include_ignored=bool(o.get("all"))).items():
                if f == p or f.startswith(p + "/"):
                    if node.type == "f" or node.type == "l":
                        sha, mode = repo.wt_sha(f, write=True)
                        u_flat[f] = (f"{mode:o}", sha)
                        u_files.append(f)
        u_tree = db.write_tree(u_flat)
        u_commit = repo.store("commit", encode_commit(u_tree, [], who, who, f"untracked files on {branch}: {desc}\n"))
        parents.append(u_commit)
    w_tree = db.write_tree(w_flat)
    msg = f"On {branch}: {message}" if message else f"WIP on {branch}: {desc}"
    w_commit = repo.store("commit", encode_commit(w_tree, parents, who, who, msg))
    repo.update_ref("refs/stash", w_commit, msg)
    # reset what was stashed
    if paths:
        for p in list(staged) + list(unstaged):
            h = db.flat(repo.head_tree()).get(p)
            if h is None:
                repo.index.remove(p)
                repo.remove_file(p)
            else:
                repo.checkout_file(p, h[1], h[0])
                repo.index.add(repo.index_entry_for(p, h[1], int(h[0], 8), repo.wt_node(p)))
        repo.write_index()
    elif o.get("staged"):
        for p in staged:
            h = db.flat(repo.head_tree()).get(p)
            if h is None:
                repo.index.remove(p)
                repo.remove_file(p)
            else:
                repo.checkout_file(p, h[1], h[0])
                repo.index.add(repo.index_entry_for(p, h[1], int(h[0], 8), repo.wt_node(p)))
        repo.write_index()
    else:
        keep = o.get("keep-index")
        checkout_tree(g, repo, db.commit(head).tree, force=True)
        if keep:
            for p, (mode, sha) in idx_flat.items():
                repo.checkout_file(p, sha, mode)
                repo.index.add(repo.index_entry_for(p, sha, int(mode, 8), repo.wt_node(p)))
            repo.write_index()
    for f in u_files:
        repo.remove_file(f)
    if not o.get("quiet"):
        g.out(f"Saved working directory and index state {msg}\n")
    return 0


def _stash_apply(g, repo, sha, restore_index, quiet):
    db = repo.odb
    c = db.commit(sha)
    base = db.commit(c.parents[0]).tree
    head_tree = repo.head_tree()
    # untracked files come back first (and refuse to overwrite)
    if len(c.parents) > 2:
        u = db.flat(db.commit(c.parents[2]).tree)
        clash = [p for p in u if repo.wt_node(p) is not None]
        if clash:
            g.err("".join(f"{p} already exists, no checkout\n" for p in clash) + "error: could not restore untracked "
                  "files from stash\n")
            return False
    res = merge_trees(db, base, head_tree, c.tree, "Updated upstream", "Stashed changes", "Stash base",
                      repo.config.get("merge.conflictStyle", "merge"))
    st0 = collect(repo, untracked="no")
    changing = {p for p in set(db.flat(head_tree)) | set(res.flat) | set(res.stages)
                if p in res.stages or db.flat(head_tree).get(p) != res.flat.get(p)}
    dirty = sorted(p for p in changing if p in st0.unstaged or p in st0.staged)
    if dirty:
        g.err("error: Your local changes to the following files would be overwritten by merge:\n" +
              "".join(f"\t{p}\n" for p in dirty) + "Please commit your changes or stash them before you merge.\nAborting\n")
        return False
    staged_before = {p: e for p, e in repo.index.stage0().items()}
    apply_result(g, repo, res, None)
    for m in res.messages:
        g.out(m + "\n")
    # without --index, the stash's changes come back unstaged (new files stay added)
    idx = repo.index
    i_flat = db.flat(db.commit(c.parents[1]).tree)
    for p in sorted(changing):
        if p in res.stages:
            continue
        h = db.flat(head_tree).get(p)
        new = res.flat.get(p)
        if restore_index:
            want = i_flat.get(p)
            if want is not None:
                idx.add(Entry(p, want[1], int(want[0], 8)))
            elif p in idx.stage0() and h is None:
                idx.remove(p)
            continue
        if h is not None:
            idx.add(Entry(p, h[1], int(h[0], 8)) if p not in staged_before else staged_before[p])
        elif new is not None and p in db.flat(db.commit(c.parents[1]).tree) and p not in db.flat(base):
            pass        # a file added to the index before stashing stays added
        else:
            idx.remove(p)
    repo.write_index()
    if len(c.parents) > 2:
        u = db.flat(db.commit(c.parents[2]).tree)
        for p, (mode, s) in u.items():
            repo.checkout_file(p, s, mode)
    if res.conflicts:
        return False
    if not quiet:
        from .cmd_basic import git_status
        git_status(g, [])
    return True


# ---------------------------------------------------------------------------- tag

@command("tag")
def git_tag(g, args):
    o = parse(args, "annotate|a message|m= file|F= delete|d list|l n=? force|f contains= no-contains= points-at= "
                    "merged= no-merged= sort= sign|s local-user|u= verify|v format= ignore-case|i column no-column "
                    "cleanup= edit|e create-reflog", numbers=False)
    repo = g.repo()
    db = repo.odb
    a = o.args
    if o.get("delete"):
        st = 0
        for name in a:
            sha = repo.resolve_ref(f"refs/tags/{name}")
            if not sha:
                g.err(f"error: tag '{name}' not found.\n")
                st = 1
                continue
            repo.delete_ref(f"refs/tags/{name}")
            g.out(f"Deleted tag '{name}' (was {db.abbrev(sha)})\n")
        return st
    if o.get("sign"):
        raise GitError("error: gpg failed to sign the data\nerror: unable to sign the tag")
    listing = o.get("list") or not a or o.get("n") is not None or o.get("contains") or o.get("points-at") or \
        o.get("merged") or o.get("no-merged")
    if listing and not (a and not o.get("list") and (o.get("annotate") or o.get("message"))):
        import fnmatch
        names = sorted(repo.tags())
        sort = o.get("sort") or repo.config.get("tag.sort")
        if sort:
            rev = sort.startswith("-")
            key = sort.lstrip("-")
            if key in ("v:refname", "version:refname"):
                names.sort(key=_version_key, reverse=rev)
            elif key in ("creatordate", "taggerdate", "committerdate"):
                def when(n):
                    s = repo.tags()[n]
                    if db.kind(s) == "tag":
                        t = db.tag(s)
                        return t.tagger.when if t.tagger else 0
                    return db.commit(repo.peel(s)).committer.when
                names.sort(key=when, reverse=rev)
            else:
                names.sort(reverse=rev)
        out = []
        for n in names:
            if a and not any(fnmatch.fnmatch(n, p) for p in a):
                continue
            sha = repo.tags()[n]
            target = repo.peel(sha)
            if o.get("contains") and not db.is_ancestor(resolve(repo, o.get("contains"), "commit"), target):
                continue
            if o.get("points-at") and target != resolve(repo, o.get("points-at"), "commit"):
                continue
            if o.get("merged") and not db.is_ancestor(target, resolve(repo, o.get("merged"), "commit")):
                continue
            if o.get("no-merged") and db.is_ancestor(target, resolve(repo, o.get("no-merged"), "commit")):
                continue
            if o.get("n") is not None:
                k = int(o.get("n")) if isinstance(o.get("n"), str) else 1
                if db.kind(sha) == "tag":
                    lines = db.tag(sha).message.strip().split("\n")[:k]
                else:
                    lines = [db.commit(target).subject] if db.kind(target) == "commit" else [""]
                out.append(f"{n:<15} " + ("\n" + " " * 4).join(lines))
            else:
                out.append(n)
        if out:
            g.out("\n".join(out) + "\n")
        return 0
    name = a[0]
    if not name or name.startswith("-") or ".." in name or re.search(r"[\s~^:?*\[\\]", name):
        raise GitError(f"fatal: '{name}' is not a valid tag name.")
    target_spec = a[1] if len(a) > 1 else "HEAD"
    target = try_resolve(repo, target_spec)
    if target is None:
        raise GitError(f"fatal: Failed to resolve '{target_spec}' as a valid ref.")
    exists = repo.resolve_ref(f"refs/tags/{name}")
    if exists and not o.get("force"):
        raise GitError(f"fatal: tag '{name}' already exists")
    msg = None
    if o.get("message") is not None:
        msg = "\n\n".join(o.all("message")) + "\n"
    elif o.get("file"):
        f = o.get("file")
        import posixpath
        msg = bytes(g.fs.lookup(f if f.startswith("/") else posixpath.join(g.cwd, f)).data).decode()
    elif o.get("annotate"):
        repo.write("TAG_EDITMSG", "\n#\n# Write a message for tag:\n#   " + name + "\n# Lines starting with '#' will "
                   "be ignored.\n")
        st = yield from launch_editor(g, repo.path("TAG_EDITMSG"))
        msg = cleanup_message(repo.read("TAG_EDITMSG") or "")
        if not msg.strip():
            raise GitError("fatal: no tag message?")
    if msg is not None:
        sha = repo.store("tag", encode_tag(target, db.kind(target), name, repo.ident("committer"), msg))
    else:
        sha = target
    repo.update_ref(f"refs/tags/{name}", sha, None)
    if exists and o.get("force"):
        g.out(f"Updated tag '{name}' (was {db.abbrev(exists)})\n")
    return 0


def _version_key(name):
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]

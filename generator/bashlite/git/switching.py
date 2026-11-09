"""Moving the working tree and index between commits (git's two-way merge, read-tree -m -u): local changes to
files the switch doesn't touch are carried along, changes it would overwrite stop it. checkout and switch."""

from __future__ import annotations

import re

from .cli import Usage, command, parse, pathspec_matcher
from .objects import GitError
from .repo import Entry
from .revparse import dwim_ref, previous_branch, resolve, try_resolve
from .status import worktree_flat

DETACHED_ADVICE = (
    "You are in 'detached HEAD' state. You can look around, make experimental\n"
    "changes and commit them, and you can discard any commits you make in this\n"
    "state without impacting any branches by switching back to a branch.\n\n"
    "If you want to create a new branch to retain commits you create, you may\n"
    "do so (now or later) by using -c with the switch command. Example:\n\n"
    "  git switch -c <new-branch-name>\n\n"
    "Or undo this operation with:\n\n"
    "  git switch -\n\n"
    "Turn off this advice by setting config variable advice.detachedHead to false\n\n")


def checkout_tree(g, repo, target_tree, force=False, op="checkout", dry=False):
    """Make the index and working tree match target_tree, keeping unrelated local changes unless force.
    -> paths with local changes that were carried (for the `M\\tfile` lines). Raises GitError when unsafe."""
    db = repo.odb
    head = db.flat(repo.head_tree())
    tgt = db.flat(target_tree)
    idx = repo.index
    index = {p: (e.mode_str, e.sha) for p, e in idx.stage0().items()}
    conflicts = idx.conflicts()
    wt = worktree_flat(repo) if not force else {}
    files = repo.worktree_files(include_ignored=False) if not force else {}
    blocked, untracked_blocked, carried = [], [], []
    plan = []
    for p in sorted(set(head) | set(tgt) | set(index) | set(conflicts)):
        h, t, i = head.get(p), tgt.get(p), index.get(p)
        if force:
            plan.append((p, t))
            continue
        w = wt.get(p)
        local = (i != h) or (p in index and w != i) or p in conflicts
        if h == t:
            if local and (i is not None or w is not None):
                carried.append(p)
            continue
        if i == t and (w == t or (t is None and w is None)):
            continue
        if local:
            if i == t:
                continue
            blocked.append(p)
            continue
        if i is None and p in files and t is not None:
            node = files[p]
            got = repo.wt_sha(p)
            if got is None or (f"{got[1]:o}", got[0]) != t:
                untracked_blocked.append(p)
                continue
        plan.append((p, t))
    # an untracked file where a tracked directory's file will go (and the reverse) is left alone
    if blocked:
        verb = {"checkout": "checkout", "merge": "merge", "reset": "reset"}.get(op, op)
        tail = {"checkout": "Please commit your changes or stash them before you switch branches.",
                "merge": "Please commit your changes or stash them before you merge.",
                "reset": "Please commit your changes or stash them before you reset."}.get(op, "")
        raise GitError(f"error: Your local changes to the following files would be overwritten by {verb}:\n" +
                       "".join(f"\t{g.shown(repo, p)}\n" for p in blocked) + tail + "\nAborting", 1)
    if untracked_blocked:
        tail = {"checkout": "Please move or remove them before you switch branches.",
                "merge": "Please move or remove them before you merge."}.get(op, "Please move or remove them.")
        raise GitError(f"error: The following untracked working tree files would be overwritten by {op}:\n" +
                       "".join(f"\t{g.shown(repo, p)}\n" for p in untracked_blocked) + tail + "\nAborting", 1)
    if dry:
        return carried
    for p, t in plan:
        if t is None:
            idx.remove(p)
            if repo.wt_node(p) is not None:
                repo.remove_file(p)
        else:
            repo.checkout_file(p, t[1], t[0])
            node = repo.wt_node(p)
            idx.add(repo.index_entry_for(p, t[1], int(t[0], 8), node))
    if force:
        for p in list(conflicts):
            if p not in tgt:
                idx.remove(p)
    repo.write_index()
    return carried


def show_local_changes(g, repo, carried):
    from .status import collect
    st = collect(repo, untracked="no")
    for p in sorted(set(carried) | set(st.staged) | set(st.unstaged)):
        v = st.unstaged.get(p) or st.staged.get(p)
        if isinstance(v, tuple):
            v = "R"
        if v:
            g.out(f"{v}\t{g.shown(repo, p)}\n")


def lost_commits(repo, old_head, new_target):
    """Commits only reachable from a detached HEAD we are leaving."""
    if repo.head_branch() or not old_head:
        return []
    db = repo.odb
    refs = list(repo.ref_names("refs/").values()) + [new_target]
    keep = db.reachable([repo.peel(x) for x in refs if x and db.has(x) and db.kind(repo.peel(x)) == "commit"])
    return [c for c in db.ancestors([old_head]) if c not in keep]


def switch_to(g, repo, target_sha, branch=None, detach_name=None, force=False, quiet=False, create_msg=None,
              reflog_from=None, reflog_to=None, verb="switch"):
    """Move HEAD (to a branch, or detached at target_sha), updating the index and working tree."""
    db = repo.odb
    old_head = repo.head()
    old_branch = repo.head_branch()
    if branch and branch != old_branch and not force:
        from .cmd_extra import worktree_holding
        at = worktree_holding(repo, branch)
        if at:
            raise GitError(f"fatal: '{branch}' is already checked out at '{at}'")
    target_tree = db.commit(target_sha).tree if target_sha else None
    carried = []
    if target_tree is not None:
        carried = checkout_tree(g, repo, target_tree, force=force)
    lost = lost_commits(repo, old_head, target_sha)
    from_name = reflog_from or old_branch or old_head
    to_name = reflog_to or branch or detach_name or target_sha
    msg = f"checkout: moving from {from_name} to {to_name}"
    if branch:
        repo.set_head_symbolic(f"refs/heads/{branch}", msg)
    else:
        repo.set_head_detached(target_sha, msg)
    if quiet:
        return
    if lost:
        n = len(lost)
        g.err(f"Warning: you are leaving {n} commit{'s' if n != 1 else ''} behind, not connected to\n"
              f"any of your branches:\n\n" +
              "".join(f"  {db.abbrev(c)} {db.commit(c).subject}\n" for c in lost[:4]) +
              (f" ... and {n - 4} more.\n" if n > 4 else "") +
              "\nIf you want to keep it by creating a new branch, this may be a good time\nto do so with:\n\n"
              f" git branch <new-branch-name> {db.abbrev(lost[0])}\n\n")
    elif old_head and not old_branch and old_head != target_sha:
        g.err(f"Previous HEAD position was {db.abbrev(old_head)} {db.commit(old_head).subject}\n")
    if not branch:
        if not old_branch and old_head == target_sha:
            g.err(f"HEAD is now at {db.abbrev(target_sha)} {db.commit(target_sha).subject}\n")
            return
        if repo.config.get_bool("advice.detachedHead", True):
            g.err(f"Note: switching to '{detach_name}'.\n\n" + DETACHED_ADVICE)
        g.err(f"HEAD is now at {db.abbrev(target_sha)} {db.commit(target_sha).subject}\n")
        return
    show_local_changes(g, repo, carried)
    if create_msg:
        g.err(create_msg)
    elif old_branch == branch:
        g.err(f"Already on '{branch}'\n")
    else:
        g.err(f"Switched to branch '{branch}'\n")
    from .cmd_basic import tracking_lines
    tl = tracking_lines(repo, branch)
    if tl:
        g.out("\n".join(tl) + "\n")


def set_upstream(repo, branch, remote, merge_ref):
    cfg = repo.config.local_file()
    uid, gid = repo.ctx.uid, repo.ctx.proc.gid
    cfg.set(f"branch.{branch}.remote", remote, uid, gid)
    cfg.set(f"branch.{branch}.merge", merge_ref, uid, gid)
    repo.reload_config()


def create_branch(g, repo, name, start_spec, force=False, track=None, quiet=False):
    """git branch NAME START (and switch -c): returns (sha, upstream message or None)."""
    check_branch_name(name)
    exists = repo.resolve_ref(f"refs/heads/{name}")
    if exists and not force:
        raise GitError(f"fatal: a branch named '{name}' already exists")
    if exists and force and repo.head_branch() == name:
        raise GitError(f"fatal: cannot force update the branch '{name}' used by worktree at '{repo.worktree}'")
    sha = try_resolve(repo, start_spec, "commit") if start_spec else repo.head()
    if not sha:
        if start_spec:
            raise GitError(f"fatal: not a valid object name: '{start_spec}'")
        raise GitError(f"fatal: not a valid object name: '{repo.head_branch() or 'HEAD'}'")
    repo.update_ref(f"refs/heads/{name}", sha, f"branch: Created from {start_spec or 'HEAD'}")
    up_msg = None
    full, _ = dwim_ref(repo, start_spec) if start_spec else (None, None)
    auto = repo.config.get("branch.autoSetupMerge", "true")
    if track is not False and full and full.startswith("refs/remotes/") and (track or auto in ("true", "always")):
        rest = full[len("refs/remotes/"):]
        remote, _, rb = rest.partition("/")
        set_upstream(repo, name, remote, f"refs/heads/{rb}")
        up_msg = f"branch '{name}' set up to track '{remote}/{rb}'.\n"
    elif track and full and full.startswith("refs/heads/"):
        set_upstream(repo, name, ".", full)
        up_msg = f"branch '{name}' set up to track '{full[len('refs/heads/'):]}'.\n"
    return sha, up_msg


def check_branch_name(name):
    bad = (not name or name.startswith("-") or name.endswith("/") or name.endswith(".lock") or ".." in name or
           re.search(r"[\s~^:?*\[\\]", name) or name.endswith(".") or "@{" in name or name == "@" or "//" in name)
    if bad:
        raise GitError(f"fatal: '{name}' is not a valid branch name")
    if name == "HEAD":
        raise GitError("fatal: 'HEAD' is not a valid branch name")


def guess_remote_branch(repo, name):
    """`git switch foo` with no local foo but exactly one remote-tracking foo: create it from there."""
    refs = repo.ref_names("refs/remotes/")
    hits = [f"refs/remotes/{rm}/{name}" for rm in repo.remotes() if f"refs/remotes/{rm}/{name}" in refs]
    return hits[0] if len(hits) == 1 else None


@command("switch", wt=True)
def git_switch(g, args):
    o = parse(args, "create|c= force-create|C= detach|d guess no-guess force|f discard-changes merge|m "
                    "conflict= quiet|q orphan= track|t=? no-track ignore-other-worktrees recurse-submodules progress")
    repo = g.repo(wt=True)
    a = o.args
    if o.get("orphan"):
        name = o.get("orphan")
        check_branch_name(name)
        if repo.resolve_ref(f"refs/heads/{name}"):
            raise GitError(f"fatal: a branch named '{name}' already exists")
        checkout_tree(g, repo, "4b825dc642cb6eb9a060e54bf8d69288fbee4904", force=True)
        old = repo.head_branch() or repo.head()
        repo.write("HEAD", f"ref: refs/heads/{name}\n")
        repo.reflog_append("HEAD", repo.resolve_ref("HEAD") or "0" * 40, "0" * 40,
                           f"checkout: moving from {old} to {name}")
        g.err(f"Switched to a new branch '{name}'\n")
        return 0
    new = o.get("create") or o.get("force-create")
    if new:
        start = a[0] if a else None
        track = None if o.get("track") is None else (o.get("track") is not False)
        if o.get("no-track"):
            track = False
        sha, up = create_branch(g, repo, new, start, force=bool(o.get("force-create")), track=track)
        msg = (up or "") + f"Switched to a new branch '{new}'\n" if not o.get("force-create") or \
            not repo.head_branch() == new else f"Reset branch '{new}'\n"
        switch_to(g, repo, sha, branch=new, create_msg=msg, quiet=bool(o.get("quiet")),
                  reflog_to=new, force=bool(o.get("discard-changes") or o.get("force")))
        return 0
    if not a and not o.get("detach"):
        raise GitError("fatal: missing branch or commit argument", 128)
    spec = a[0] if a else "HEAD"
    if spec == "-":
        prev = previous_branch(repo, 1)
        if not prev:
            raise GitError("fatal: invalid reference: @{-1}")
        spec = prev
    force = bool(o.get("discard-changes") or o.get("force"))
    if o.get("detach"):
        sha = try_resolve(repo, spec, "commit")
        if sha is None:
            raise GitError(f"fatal: invalid reference: {spec}")
        switch_to(g, repo, sha, detach_name=spec, force=force, quiet=bool(o.get("quiet")))
        return 0
    if repo.resolve_ref(f"refs/heads/{spec}") is not None or (spec == repo.head_branch()):
        sha = repo.resolve_ref(f"refs/heads/{spec}")
        switch_to(g, repo, sha, branch=spec, force=force, quiet=bool(o.get("quiet")))
        return 0
    remote_ref = guess_remote_branch(repo, spec) if not o.get("no-guess") else None
    if remote_ref:
        rest = remote_ref[len("refs/remotes/"):]
        sha, up = create_branch(g, repo, spec, rest, track=True)
        switch_to(g, repo, sha, branch=spec, create_msg=(up or "") + f"Switched to a new branch '{spec}'\n",
                  force=force)
        return 0
    if try_resolve(repo, spec, "commit"):
        raise GitError(f"fatal: a branch is expected, got {'tag' if repo.resolve_ref('refs/tags/' + spec) else 'commit'} '{spec}'\n"
                       "hint: If you want to detach HEAD at the commit, try again with the --detach option.")
    raise GitError(f"fatal: invalid reference: {spec}")


@command("checkout")
def git_checkout(g, args):
    o = parse(args, "b= B= orphan= detach force|f ours theirs merge|m quiet|q patch|p track|t=? no-track "
                    "conflict= ignore-other-worktrees progress recurse-submodules guess no-guess")
    repo = g.repo(wt=True)
    a = list(o.args)
    paths = o.paths
    db = repo.odb
    if o.get("b") or o.get("B"):
        new = o.get("b") or o.get("B")
        start = a[0] if a else None
        track = None if o.get("track") is None else True
        if o.get("no-track"):
            track = False
        existed = repo.resolve_ref(f"refs/heads/{new}") is not None
        sha, up = create_branch(g, repo, new, start, force=bool(o.get("B")), track=track)
        msg = (up or "") + (f"Reset branch '{new}'\n" if existed and o.get("B") else f"Switched to a new branch '{new}'\n")
        switch_to(g, repo, sha, branch=new, create_msg=msg, quiet=bool(o.get("quiet")), force=bool(o.get("force")))
        return 0
    if o.get("orphan"):
        return git_switch(g, ["--orphan", o.get("orphan")])
    # git checkout [<tree-ish>] -- <paths>  /  git checkout <paths>
    if paths is None and a:
        first = a[0]
        is_rev = first == "-" or try_resolve(repo, first, "commit") is not None or \
            repo.resolve_ref(f"refs/heads/{first}") is not None or guess_remote_branch(repo, first)
        if is_rev and len(a) == 1:
            pass
        elif is_rev:
            paths = a[1:]
            a = a[:1]
        else:
            paths = a
            a = []
    if paths is not None:
        source = a[0] if a else None
        if o.paths is not None:
            o.flags["_dashdash"] = True
        return _checkout_paths(g, repo, source, paths, o)
    if not a:
        if o.get("detach"):
            return git_switch(g, ["--detach"])
        # `git checkout` alone: show the tracking state
        b = repo.head_branch()
        if b:
            from .cmd_basic import tracking_lines
            tl = tracking_lines(repo, b)
            if tl:
                g.out("\n".join(tl) + "\n")
        return 0
    spec = a[0]
    if spec == "-":
        prev = previous_branch(repo, 1)
        if not prev:
            raise GitError("error: pathspec '-' did not match any file(s) known to git", 1)
        spec = prev
    force = bool(o.get("force"))
    if repo.resolve_ref(f"refs/heads/{spec}") is not None and not o.get("detach"):
        switch_to(g, repo, repo.resolve_ref(f"refs/heads/{spec}"), branch=spec, force=force, quiet=bool(o.get("quiet")))
        return 0
    if not o.get("detach"):
        remote_ref = guess_remote_branch(repo, spec) if not o.get("no-guess") else None
        if remote_ref and try_resolve(repo, spec, "commit") is None:
            sha, up = create_branch(g, repo, spec, remote_ref[len("refs/remotes/"):], track=True)
            switch_to(g, repo, sha, branch=spec, create_msg=(up or "") + f"Switched to a new branch '{spec}'\n",
                      force=force)
            return 0
    sha = try_resolve(repo, spec, "commit")
    if sha is None:
        raise GitError(f"error: pathspec '{spec}' did not match any file(s) known to git", 1)
    switch_to(g, repo, sha, detach_name=spec, force=force, quiet=bool(o.get("quiet")))
    return 0


def _checkout_paths(g, repo, source, paths, o):
    db = repo.odb
    idx = repo.index
    conflicts = idx.conflicts()
    if o.get("patch"):
        from .interactive import restore_patch
        return (yield from restore_patch(g, repo, paths, False, source=source))
    match = pathspec_matcher(g, repo, paths)
    if source:
        tree = try_resolve(repo, source, "tree")
        if tree is None:
            raise GitError(f"fatal: invalid reference: {source}")
        flat = db.flat(tree)
        hits = [p for p in flat if match(p)]
        for spec in paths:
            m1 = pathspec_matcher(g, repo, [spec])
            if not any(m1(p) for p in flat):
                raise GitError(f"error: pathspec '{spec}' did not match any file(s) known to git", 1)
        for p in hits:
            mode, sha = flat[p]
            repo.checkout_file(p, sha, mode)
            idx.add(repo.index_entry_for(p, sha, int(mode, 8), repo.wt_node(p)))
        repo.write_index()
        if not o.get("quiet"):    # git names the tree it read the paths from
            g.err(f"Updated {len(hits)} path{'s' if len(hits) != 1 else ''} from {db.abbrev(tree)}\n")
        return 0
    known = set(idx.paths())
    for spec in paths:
        m1 = pathspec_matcher(g, repo, [spec])
        if not any(m1(p) for p in known):
            raise GitError(f"error: pathspec '{spec}' did not match any file(s) known to git", 1)
    n = 0
    from .status import worktree_flat
    wt = worktree_flat(repo, refresh=False)
    for p in sorted(known):
        if not match(p):
            continue
        if p in conflicts:
            stg = conflicts[p]
            if o.get("ours") or o.get("theirs"):
                want = 2 if o.get("ours") else 3
                if want not in stg:
                    raise GitError(f"error: path '{p}' does not have {'our' if want == 2 else 'their'} version", 1)
                repo.checkout_file(p, stg[want].sha, stg[want].mode)
                n += 1
                continue
            if o.get("merge") or o.get("conflict"):
                from .merge import merge_text
                b = db.raw(stg[1].sha) if 1 in stg else b""
                merged, _ = merge_text(b, db.raw(stg[2].sha), db.raw(stg[3].sha), "ours", "theirs",
                                       style=o.get("conflict") or repo.config.get("merge.conflictStyle", "merge"))
                g.fs.write_file(repo.wt_path(p), merged, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
                n += 1
                continue
            raise GitError(f"error: path '{p}' is unmerged", 1)
        e = idx.get(p)
        if e is None:
            continue
        if wt.get(p) == (e.mode_str, e.sha):
            continue                     # already up to date: not counted
        repo.checkout_file(p, e.sha, e.mode)
        node = repo.wt_node(p)
        idx.add(repo.index_entry_for(p, e.sha, e.mode, node))
        n += 1
    repo.write_index()
    if not o.get("quiet") and not o.get("_dashdash"):    # git counts paths only without `--`
        g.err(f"Updated {n} path{'s' if n != 1 else ''} from the index\n")
    return 0

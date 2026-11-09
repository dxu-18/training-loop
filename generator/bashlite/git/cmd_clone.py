"""git clone (and the shared "set up a new repository from a far side" steps)."""

from __future__ import annotations

import posixpath
import re

from ..vfs import FSError
from .cli import Usage, command, parse
from .cmd_basic import make_gitdir
from .cmd_plumbing import write_pack
from .github import github_of
from .objects import GitError
from .remote import missing_objects, object_stats, parse_url, remote_progress
from .repo import Repo, rmtree


def default_dir(url: str) -> str:
    u = url.rstrip("/")
    if u.endswith("/.git"):
        u = u[:-5]
    name = re.split(r"[/:]", u)[-1]
    return name[:-4] if name.endswith(".git") else name


@command("clone")
def git_clone(g, args):
    o = parse(args, "branch|b= depth= bare mirror origin|o= no-checkout|n single-branch no-single-branch "
                    "recurse-submodules=? quiet|q verbose|v progress no-tags shallow-submodules filter= "
                    "sparse local|l shared|s jobs|j= template= config|c=")
    if not o.args:
        raise Usage("usage: git clone [<options>] [--] <repo> [<dir>]")
    url = o.args[0]
    target = o.args[1] if len(o.args) > 1 else default_dir(url) + (".git" if o.get("bare") else "")
    top = target if target.startswith("/") else posixpath.normpath(posixpath.join(g.cwd, target))
    fs = g.fs
    try:
        node = fs.lookup(top)
        if node.type != "d" or node.entries:
            raise GitError(f"fatal: destination path '{target}' already exists and is not an empty directory.")
    except FSError:
        pass
    ep = parse_url(g.ctx, url)
    quiet = bool(o.get("quiet"))
    if not quiet:
        g.err(f"Cloning into {'bare repository ' if o.get('bare') else ''}'{target}'...\n")
    # the far side
    from .cmd_remote import connect
    far = None
    try:
        far = yield from connect(g, url)
    except GitError as e:
        if ep.kind != "github" or "not found" not in e.msg.lower():
            raise
        # not one of GitHub-lite's: maybe a real public repository on github.com (fetched read-only, size-capped)
        from .realclone import fetch_real
        got = yield from fetch_real(g, ep, o.get("branch"), quiet)
        if got is None:
            raise
        if got is False:
            return 128
        far = yield from connect(g, url)
    refs = far.refs()
    heads = {r: s for r, s in refs.items() if r.startswith("refs/heads/")}
    tags = {r: s for r, s in refs.items() if r.startswith("refs/tags/")} if not o.get("no-tags") else {}
    head_ref = far.head_symref()
    want_branch = o.get("branch")
    if want_branch:
        if f"refs/heads/{want_branch}" in heads:
            head_ref = f"refs/heads/{want_branch}"
        elif f"refs/tags/{want_branch}" in tags:
            head_ref = f"refs/tags/{want_branch}"
        else:
            raise GitError(f"fatal: Remote branch {want_branch} not found in upstream origin")
    remote = o.get("origin") or "origin"
    bare = bool(o.get("bare") or o.get("mirror"))
    gitdir = top if bare else f"{top}/.git"
    db_branch = head_ref[len("refs/heads/"):] if head_ref and head_ref.startswith("refs/heads/") else "main"
    fs.mkdir(top, "/", 0o755, g.ctx.uid, g.ctx.proc.gid, parents=True)
    make_gitdir(g.ctx, gitdir, db_branch, bare)
    repo = Repo(g.ctx, gitdir, None if bare else top)
    uid, gid = g.ctx.uid, g.ctx.proc.gid
    cfg = repo.config.local_file()
    single = bool(o.get("single-branch") or o.get("depth")) and not o.get("no-single-branch")
    stored_url = far.repo.gitdir if (ep.kind == "local" and far.repo is not None) else url   # local: absolute
    if far.repo is not None and ep.kind == "local" and far.repo.worktree:
        stored_url = far.repo.worktree
    if not bare or o.get("mirror"):
        cfg.set(f"remote.{remote}.url", stored_url, uid, gid)
        spec = f"+refs/heads/{db_branch}:refs/remotes/{remote}/{db_branch}" if single else \
            f"+refs/heads/*:refs/remotes/{remote}/*"
        cfg.set(f"remote.{remote}.fetch", spec, uid, gid)
    for k in o.all("config"):
        key, _, val = k.partition("=")
        cfg.set(key, val, uid, gid)
    repo.reload_config()
    db = repo.odb
    # history (shallow with --depth)
    branch_heads = {r: s for r, s in heads.items() if not single or r == head_ref}
    tips = list(branch_heads.values()) + [repo.peel(s) for s in tags.values() if db.has(repo.peel(s))]
    if o.get("depth"):
        depth = int(o.get("depth"))
        boundary = set()
        keep = set()
        for tip in branch_heads.values():
            frontier = [tip]
            for _ in range(depth):
                nxt = []
                for c in frontier:
                    if c in keep:
                        continue
                    keep.add(c)
                    nxt += list(db.commit(c).parents)
                frontier = nxt
            for c in keep:
                if any(p not in keep for p in db.commit(c).parents):
                    boundary.add(c)
        repo.write("shallow", "".join(c + "\n" for c in sorted(boundary)), True)
        db.cut = boundary
        tags = {r: s for r, s in tags.items() if repo.peel(s) in keep}
        tips = list(branch_heads.values())
    objs = missing_objects(db, tips, [])
    n, size = object_stats(db, objs)
    if far.ghrepo is not None and far.ghrepo.size_kb:
        size = max(size, far.ghrepo.size_kb * 1024 - _lfs_bytes(far.ghrepo))
    deltas = n // 4
    from .remote import stderr_is_tty
    if not quiet and n and ep.kind != "local" and stderr_is_tty(g.ctx):
        g.err(remote_progress(n, size, deltas))
    if n:
        write_pack(repo, sorted(objs), url)
    # refs
    packed = {}
    if bare:
        for r, s in branch_heads.items():
            packed[r] = s
    else:
        for r, s in branch_heads.items():
            packed[f"refs/remotes/{remote}/{r[len('refs/heads/'):]}"] = s
    packed.update(tags)
    repo.write_packed_refs(packed)
    if not bare and heads and head_ref and head_ref.startswith("refs/heads/"):
        repo.write(f"refs/remotes/{remote}/HEAD", f"ref: refs/remotes/{remote}/{db_branch}\n", True)
    if not heads:
        if not bare:
            cfg.set(f"branch.{db_branch}.remote", remote, uid, gid)
            cfg.set(f"branch.{db_branch}.merge", f"refs/heads/{db_branch}", uid, gid)
            repo.reload_config()
        if not quiet:
            g.err("warning: You appear to have cloned an empty repository.\n")
            if ep.kind == "local":
                g.err("done.\n")
        return 0
    if not quiet and ep.kind == "local":
        g.err("done.\n")
    sha = refs.get(head_ref) if head_ref else None
    if head_ref and head_ref.startswith("refs/tags/"):
        sha = repo.peel(sha)
        repo.write("HEAD", sha + "\n")
    elif sha:
        repo.update_ref(head_ref, sha, f"clone: from {url}")
        if not bare:
            cfg.set(f"branch.{db_branch}.remote", remote, uid, gid)
            cfg.set(f"branch.{db_branch}.merge", head_ref, uid, gid)
            repo.reload_config()
    if bare or o.get("no-checkout"):
        return 0
    # the working tree (LFS files come down when git-lfs is installed)
    from .switching import checkout_tree
    from .lfs import active
    tree = db.commit(sha).tree
    checkout_tree(g, repo, tree, force=True)
    if far.ghrepo is not None and far.ghrepo.lfs and active(repo):
        n_lfs = sum(1 for p in db.flat(tree) if repo.lfs_tracked(p))
        if n_lfs and not quiet:
            total = _lfs_bytes(far.ghrepo)
            from .remote import human_bytes
            g.err(f"Filtering content: 100% ({n_lfs}/{n_lfs}), {human_bytes(total)} | {human_bytes(total / 3)}/s, done.\n")
    if o.get("recurse-submodules") is not None:
        from .cmd_extra import update_submodules
        yield from update_submodules(g, repo, init=True, recursive=True)
    return 0


def _lfs_bytes(ghrepo):
    return sum(len(v) for v in ghrepo.lfs.values())


__all__ = ["git_clone", "default_dir", "rmtree", "github_of"]

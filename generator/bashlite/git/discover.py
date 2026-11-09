"""Finding the repository a command runs in: $GIT_DIR / $GIT_WORK_TREE, or the nearest .git walking up from the
current directory (a directory, or a `gitdir: ...` file in a linked worktree or submodule), or a bare repository."""

from __future__ import annotations

import posixpath

from ..vfs import FSError
from .objects import GitError
from .repo import Repo

NOT_A_REPO = "fatal: not a git repository (or any of the parent directories): .git"


def _read(fs, path):
    try:
        node = fs.lookup(path)
    except FSError:
        return None
    if node.type != "f":
        return None
    return bytes(node.data).decode("utf-8", "replace")


def _is_gitdir(fs, d: str) -> bool:
    try:
        node = fs.lookup(d)
    except FSError:
        return False
    if node.type != "d":
        return False
    e = node.entries
    return "HEAD" in e and ("objects" in e or "commondir" in e) and ("refs" in e or "commondir" in e)


def _gitdir_of(fs, dotgit: str):
    """.git as a directory, or a `gitdir: <path>` file -> (gitdir, commondir)."""
    try:
        node = fs.lookup(dotgit)
    except FSError:
        return None
    if node.type == "d":
        return (dotgit, None) if _is_gitdir(fs, dotgit) else None
    text = (_read(fs, dotgit) or "").strip()
    if not text.startswith("gitdir:"):
        return None
    gd = text[7:].strip()
    if not gd.startswith("/"):
        gd = posixpath.normpath(posixpath.join(posixpath.dirname(dotgit), gd))
    common = None
    c = _read(fs, gd + "/commondir")
    if c:
        c = c.strip()
        common = c if c.startswith("/") else posixpath.normpath(posixpath.join(gd, c))
    return gd, common


def open_repo_at(ctx, top: str):
    """The repository whose working tree top is exactly `top` (or None)."""
    found = _gitdir_of(ctx.fs, top.rstrip("/") + "/.git")
    if found is None:
        return None
    gd, common = found
    return Repo(ctx, gd, top, common)


def find_repo(ctx, start: str | None = None, required: bool = True, work_tree_needed: bool = False):
    fs = ctx.fs
    env = ctx.env
    gd = env.get("GIT_DIR")
    if gd:
        gd = gd if gd.startswith("/") else posixpath.normpath(posixpath.join(ctx.cwd, gd))
        wt = env.get("GIT_WORK_TREE")
        if wt:
            wt = wt if wt.startswith("/") else posixpath.normpath(posixpath.join(ctx.cwd, wt))
        elif not _bare_config(fs, gd):
            wt = ctx.cwd
        if not _is_gitdir(fs, gd):
            raise GitError(f"fatal: not a git repository: '{env.get('GIT_DIR')}'")
        return Repo(ctx, gd, wt)
    d = posixpath.normpath(start or ctx.cwd)
    while True:
        found = _gitdir_of(fs, (d.rstrip("/") or "") + "/.git")
        if found is not None:
            gd, common = found
            wt = d
            # a linked worktree's top is the directory holding its .git file
            return Repo(ctx, gd, wt, common)
        if _is_gitdir(fs, d) and _bare_config(fs, d):
            if work_tree_needed:
                raise GitError("fatal: this operation must be run in a work tree")
            return Repo(ctx, d, None)
        if d == "/":
            break
        d = posixpath.dirname(d)
    if required:
        raise GitError(NOT_A_REPO)
    return None


def _bare_config(fs, gd: str) -> bool:
    text = _read(fs, gd + "/config") or ""
    from .repo import parse_config
    for k, v, _ in parse_config(text):
        if k == "core.bare":
            return (v or "true").lower() == "true"
    return False

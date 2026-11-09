"""worktree, submodule, sparse-checkout."""

from __future__ import annotations

import posixpath
import re

from ..vfs import FSError
from .cli import Usage, command, parse
from .objects import GitError
from .repo import Repo, rmtree
from .revparse import try_resolve
from .switching import guess_remote_branch


# ---------------------------------------------------------------------------- worktree

def _worktrees(repo):
    """[(path, Repo)] for the main working tree and every linked one."""
    main_top = posixpath.dirname(repo.common) if repo.common.endswith("/.git") else None
    out = []
    if main_top:
        out.append((main_top, Repo(repo.ctx, repo.common, main_top)))
    base = f"{repo.common}/worktrees"
    try:
        node = repo.fs.lookup(base)
    except FSError:
        return out
    for name in sorted(node.entries):
        gd = f"{base}/{name}"
        try:
            path = bytes(repo.fs.lookup(f"{gd}/gitdir").data).decode().strip()
        except FSError:
            continue
        top = posixpath.dirname(path)
        out.append((top, Repo(repo.ctx, gd, top, repo.common)))
    return out


def other_worktree_branches(repo) -> set:
    """Branches checked out in some other working tree of this repository (`git branch` marks them with +)."""
    if not repo.fs.exists(f"{repo.common}/worktrees"):
        return set()
    return {r.head_branch() for t, r in _worktrees(repo) if r.gitdir != repo.gitdir} - {None}


def worktree_holding(repo, branch: str):
    """The path of another working tree that has `branch` checked out (or None)."""
    if not repo.fs.exists(f"{repo.common}/worktrees"):
        return None
    for t, r in _worktrees(repo):
        if r.gitdir != repo.gitdir and r.head_branch() == branch:
            return t
    return None


@command("worktree")
def git_worktree(g, args):
    repo = g.repo()
    db = repo.odb
    sub = args[0] if args else ""
    rest = args[1:]
    if sub == "add":
        o = parse(rest, "b= B= detach|d force|f quiet|q no-checkout track no-track lock")
        if not o.args:
            raise Usage("usage: git worktree add [-f] [--detach] [--checkout] [--lock [--reason <string>]]\n"
                        "                        [--orphan] [(-b | -B) <new-branch>] <path> [<commit-ish>]")
        path = o.args[0]
        top = path if path.startswith("/") else posixpath.normpath(posixpath.join(g.cwd, path))
        if g.fs.exists(top):
            node = g.fs.lookup(top)
            if node.type != "d" or node.entries:
                raise GitError(f"fatal: '{path}' already exists")
        commitish = o.args[1] if len(o.args) > 1 else None
        new_branch = o.get("b") or o.get("B")
        name = posixpath.basename(top)
        used = {r.head_branch(): t for t, r in _worktrees(repo)}
        if new_branch:
            from .switching import create_branch
            sha, _ = create_branch(g, repo, new_branch, commitish, force=bool(o.get("B")))
            branch = new_branch
            msg = f"Preparing worktree (new branch '{branch}')"
        elif commitish and not o.get("detach") and not repo.resolve_ref(f"refs/heads/{commitish}") and \
                guess_remote_branch(repo, commitish):
            # like `git switch`: a name only a remote has becomes a local branch tracking it
            from .switching import create_branch, set_upstream
            remote_ref = guess_remote_branch(repo, commitish)
            sha, _ = create_branch(g, repo, commitish, remote_ref)
            rname = remote_ref.split("/")[2]
            set_upstream(repo, commitish, rname, "refs/heads/" + commitish)
            g.out(f"branch '{commitish}' set up to track '{rname}/{commitish}'.\n")
            branch = commitish
            msg = f"Preparing worktree (new branch '{branch}')"
        elif commitish and repo.resolve_ref(f"refs/heads/{commitish}") and not o.get("detach"):
            branch = commitish
            if branch in used and not o.get("force"):
                raise GitError(f"fatal: '{branch}' is already used by worktree at '{used[branch]}'")
            sha = repo.resolve_ref(f"refs/heads/{branch}")
            msg = f"Preparing worktree (checking out '{branch}')"
        elif commitish is None and not o.get("detach"):
            branch = name
            from .switching import create_branch
            if repo.resolve_ref(f"refs/heads/{branch}"):
                if branch in used:
                    raise GitError(f"fatal: '{branch}' is already used by worktree at '{used[branch]}'")
                sha = repo.resolve_ref(f"refs/heads/{branch}")
                msg = f"Preparing worktree (checking out '{branch}')"
            else:
                sha, _ = create_branch(g, repo, branch, None)
                msg = f"Preparing worktree (new branch '{branch}')"
        else:
            sha = try_resolve(repo, commitish or "HEAD", "commit")
            if sha is None:
                raise GitError(f"fatal: invalid reference: {commitish}")
            branch = None
            msg = f"Preparing worktree (detached HEAD {db.abbrev(sha)})"
        uid, gid = g.ctx.uid, g.ctx.proc.gid
        wname = name
        n = 1
        while g.fs.exists(f"{repo.common}/worktrees/{wname}"):
            n += 1
            wname = f"{name}{n}"
        gd = f"{repo.common}/worktrees/{wname}"
        g.fs.mkdir(gd, "/", 0o755, uid, gid, parents=True)
        g.fs.mkdir(top, "/", 0o755, uid, gid, parents=True)
        g.fs.write_file(f"{gd}/gitdir", f"{top}/.git\n", "/", 0o644, uid, gid)
        g.fs.write_file(f"{gd}/commondir", "../..\n", "/", 0o644, uid, gid)
        g.fs.write_file(f"{gd}/HEAD", (f"ref: refs/heads/{branch}\n" if branch else sha + "\n"), "/", 0o644, uid, gid)
        g.fs.write_file(f"{top}/.git", f"gitdir: {gd}\n", "/", 0o644, uid, gid)
        wt = Repo(g.ctx, gd, top, repo.common)
        if not o.get("quiet"):
            g.err(msg + "\n")
        if not o.get("no-checkout"):
            from .switching import checkout_tree
            checkout_tree(g, wt, db.commit(sha).tree, force=True)
            if not o.get("quiet"):
                g.out(f"HEAD is now at {db.abbrev(sha)} {db.commit(sha).subject}\n")
        return 0
    if sub == "list":
        o = parse(rest, "porcelain verbose|v expire= z")
        rows = []
        for top, r in _worktrees(repo):
            h = r.head()
            b = r.head_branch()
            rows.append((top, db.abbrev(h) if h else "0000000", f"[{b}]" if b else "(detached HEAD)", r, h, b))
        if o.get("porcelain"):
            for top, _, _, r, h, b in rows:
                g.out(f"worktree {top}\nHEAD {h}\n" + (f"branch refs/heads/{b}\n" if b else "detached\n") + "\n")
            return 0
        w = max(len(x[0]) for x in rows)
        for top, h, b, *_ in rows:
            g.out(f"{top:<{w}}  {h} {b}\n")
        return 0
    if sub in ("remove", "rm"):
        o = parse(rest, "force|f")
        if not o.args:
            raise Usage("usage: git worktree remove [<options>] <worktree>")
        top = o.args[0] if o.args[0].startswith("/") else posixpath.normpath(posixpath.join(g.cwd, o.args[0]))
        for t, r in _worktrees(repo):
            if t == top and r.gitdir != r.common:
                if not o.get("force"):
                    from .status import collect
                    st = collect(r, untracked="normal")
                    if not st.clean or st.untracked:
                        raise GitError(f"fatal: '{o.args[0]}' contains modified or untracked files, use --force to "
                                       "delete it")
                rmtree(g.fs, top)
                rmtree(g.fs, r.gitdir)
                return 0
        raise GitError(f"fatal: '{o.args[0]}' is not a working tree")
    if sub == "prune":
        for t, r in _worktrees(repo):
            if r.gitdir != r.common and not g.fs.exists(t):
                rmtree(g.fs, r.gitdir)
        return 0
    raise Usage("usage: git worktree add [<options>] <path> [<commit-ish>]\n   or: git worktree list [<options>]\n"
                "   or: git worktree lock [<options>] <path>\n   or: git worktree move <worktree> <new-path>\n"
                "   or: git worktree prune [<options>]\n   or: git worktree remove [<options>] <worktree>\n"
                "   or: git worktree repair [<path>...]\n   or: git worktree unlock <worktree>")


# ---------------------------------------------------------------------------- submodule

def _gitmodules(repo):
    """[(name, path, url)] from .gitmodules in the working tree."""
    from .repo import parse_config
    node = repo.wt_node(".gitmodules")
    if node is None:
        return []
    text = bytes(node.data).decode("utf-8", "replace")
    subs: dict = {}
    for k, v, _ in parse_config(text):
        m = re.match(r"^submodule\.(.+)\.(path|url|branch)$", k)
        if m:
            subs.setdefault(m.group(1), {})[m.group(2)] = v
    return [(n, d.get("path", n), d.get("url", "")) for n, d in subs.items()]


def resolve_sub_url(repo, url: str) -> str:
    """./x and ../x are relative to the superproject's origin URL (or its own directory without one)."""
    if not url.startswith(("./", "../")):
        return url
    base = repo.config.get("remote.origin.url") or repo.worktree
    base = base.rstrip("/")
    for part in url.split("/"):
        if part == "..":
            base = re.sub(r"[/:][^/:]*$", "", base) if re.search(r"[/:]", base) else "."
        elif part not in (".", ""):
            sep = "/" if not base.endswith(":") else ""
            base = base + sep + part
    return base


def _file_transport_refused(g, repo, url: str, top: str) -> bool:
    """Since git 2.38.1, submodules may not be cloned from a local path unless protocol.file.allow=always."""
    from .remote import parse_url
    if parse_url(g.ctx, url).kind != "local":
        return False
    allow = None
    for k, v in getattr(g.ctx, "git_overrides", ()) or ():
        if k.lower() == "protocol.file.allow":
            allow = v
    allow = allow or repo.config.get("protocol.file.allow")
    if allow == "always":
        return False
    g.err(f"Cloning into '{top}'...\nfatal: transport 'file' not allowed\n"
          f"fatal: clone of '{url}' into submodule path '{top}' failed\n")
    return True


def update_submodules(g, repo, init=False, recursive=False, quiet=False):
    """git submodule update [--init] [--recursive] (generator)."""
    db = repo.odb
    head = db.flat(repo.head_tree())
    st = 0
    for name, path, url in _gitmodules(repo):
        rec = head.get(path)
        if rec is None or rec[0] != "160000":
            continue
        registered = repo.config.get(f"submodule.{name}.url")
        if not registered:
            if not init:
                continue
            url = resolve_sub_url(repo, url)
            cfg = repo.config.local_file()
            cfg.set(f"submodule.{name}.active", "true", g.ctx.uid, g.ctx.proc.gid)
            cfg.set(f"submodule.{name}.url", url, g.ctx.uid, g.ctx.proc.gid)
            repo.reload_config()
            if not quiet:
                g.err(f"Submodule '{name}' ({url}) registered for path '{path}'\n")
        url = repo.config.get(f"submodule.{name}.url") or url
        top = repo.wt_path(path)
        node = repo.wt_node(path)
        if node is None or ".git" not in (node.entries or {}):
            if _file_transport_refused(g, repo, url, top):
                g.err(f"Failed to clone '{path}'. Retry scheduled\n")
                _file_transport_refused(g, repo, url, top)
                g.err(f"Failed to clone '{path}' a second time, aborting\n")
                st = 1
                continue
            if node is not None:
                rmtree(g.fs, top)
            yield from g.ctx.flush()
            r = yield from g.ctx.run(["git", "clone"] + (["-q"] if quiet else []) + [url, top])
            if r:
                st = r
                continue
        r = yield from g.ctx.run(["git", "-C", top, "checkout", "-q", "--detach", rec[1]])
        if r == 0 and not quiet:
            g.out(f"Submodule path '{path}': checked out '{rec[1]}'\n")
        if recursive:
            from .discover import open_repo_at
            sub = open_repo_at(g.ctx, top)
            if sub is not None:
                yield from update_submodules(g, sub, init, recursive, quiet)
    return st


def _rev_name(repo, sha):
    """What `submodule status` shows after the id: git tries describe, describe --tags, then --all --always."""
    db = repo.odb
    for annotated_only in (True, False):
        tags = {}
        for name, t in repo.tags().items():
            if annotated_only and db.kind(t) != "tag":
                continue
            tags.setdefault(repo.peel(t), []).append(name)
        best = None
        for c in db.ancestors([sha]):
            if c in tags:
                depth = len(db.reachable([sha]) - db.reachable([c]))
                if best is None or depth < best[1]:
                    best = (c, depth)
        if best:
            name = sorted(tags[best[0]])[0]
            return name if not best[1] else f"{name}-{best[1]}-g{db.abbrev(sha)}"
    for nm, s in sorted(repo.ref_names("refs/").items(), key=lambda kv: (not kv[0].startswith("refs/heads/"), kv[0])):
        if s == sha:
            return nm.split("/", 1)[1]
    return db.abbrev(sha)


@command("submodule")
def git_submodule(g, args):
    repo = g.repo(wt=True)
    db = repo.odb
    sub = args[0] if args else "status"
    rest = args[1:]
    if sub == "add":
        o = parse(rest, "branch|b= force|f name= reference= depth= quiet|q")
        if not o.args:
            raise Usage("usage: git submodule add [<options>] [--] <repository> [<path>]")
        url = o.args[0]
        from .cmd_clone import default_dir
        path = o.args[1] if len(o.args) > 1 else default_dir(url)
        rel = g.to_rel(repo, path)
        if repo.index.get(rel) is not None:
            raise GitError(f"fatal: '{rel}' already exists in the index")
        top = repo.wt_path(rel)
        if _file_transport_refused(g, repo, resolve_sub_url(repo, url), top):
            return 128
        yield from g.ctx.flush()
        st = yield from g.ctx.run(["git", "clone"] + (["-b", o.get("branch")] if o.get("branch") else []) +
                                  [resolve_sub_url(repo, url), top])
        if st:
            return st
        name = o.get("name") or rel
        node = repo.wt_node(".gitmodules")
        text = bytes(node.data).decode() if node is not None else ""
        text += f'[submodule "{name}"]\n\tpath = {rel}\n\turl = {url}\n'
        g.fs.write_file(repo.wt_path(".gitmodules"), text, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
        cfg = repo.config.local_file()
        cfg.set(f"submodule.{name}.url", resolve_sub_url(repo, url), g.ctx.uid, g.ctx.proc.gid)
        cfg.set(f"submodule.{name}.active", "true", g.ctx.uid, g.ctx.proc.gid)
        repo.reload_config()
        yield from g.ctx.run(["git", "add", ".gitmodules", rel])
        return 0
    if sub == "update":
        o = parse(rest, "init recursive remote quiet|q force|f checkout merge rebase depth= jobs|j=")
        if o.get("remote"):
            for name, path, url in _gitmodules(repo):
                top = repo.wt_path(path)
                yield from g.ctx.run(["git", "-C", top, "pull", "-q", "--ff-only"])
            return 0
        return (yield from update_submodules(g, repo, bool(o.get("init")), bool(o.get("recursive")),
                                             bool(o.get("quiet"))))
    if sub == "init":
        for name, path, url in _gitmodules(repo):
            if not repo.config.get(f"submodule.{name}.url"):
                url = resolve_sub_url(repo, url)
                cfg = repo.config.local_file()
                cfg.set(f"submodule.{name}.active", "true", g.ctx.uid, g.ctx.proc.gid)
                cfg.set(f"submodule.{name}.url", url, g.ctx.uid, g.ctx.proc.gid)
                g.err(f"Submodule '{name}' ({url}) registered for path '{path}'\n")
        repo.reload_config()
        return 0
    if sub == "status":
        head = db.flat(repo.head_tree())
        from .discover import open_repo_at
        for name, path, url in _gitmodules(repo):
            rec = head.get(path, ("", ""))[1] or (repo.index.get(path).sha if repo.index.get(path) else "")
            sr = open_repo_at(g.ctx, repo.wt_path(path))
            if sr is None or not sr.head():
                g.out(f"-{rec} {path}\n")
                continue
            h = sr.head()
            mark = " " if h == rec else "+"
            g.out(f"{mark}{h} {path} ({_rev_name(sr, h)})\n")
        return 0
    if sub == "foreach":
        o = parse(rest, "recursive quiet|q", stop_at_nonopt=True)
        cmd = " ".join(o.args)
        st = 0
        for name, path, url in _gitmodules(repo):
            top = repo.wt_path(path)
            if not g.fs.exists(top + "/.git"):
                continue
            if not o.get("quiet"):
                g.out(f"Entering '{path}'\n")
            yield from g.ctx.flush()
            env = dict(g.env, name=name, sm_path=path, displaypath=path, toplevel=repo.worktree)
            r = yield from g.ctx.run(["sh", "-c", cmd], cwd=top, env=env)
            if r:
                g.err(f"fatal: run_command returned non-zero status for {path}\n.\n")
                return r
        return st
    if sub == "deinit":
        o = parse(rest, "force|f all")
        for name, path, url in _gitmodules(repo):
            if o.get("all") or path in o.args:
                top = repo.wt_path(path)
                rmtree(g.fs, top)
                g.fs.mkdir(top, "/", 0o755, g.ctx.uid, g.ctx.proc.gid)
                repo.config.local_file().remove_section(f"submodule.{name}", g.ctx.uid, g.ctx.proc.gid)
                g.out(f"Cleared directory '{path}'\nSubmodule '{name}' ({url}) unregistered for path '{path}'\n")
        return 0
    raise Usage("usage: git submodule [--quiet] [--cached]\n   or: git submodule [--quiet] add [-b <branch>] [-f|--force] "
                "[--name <name>] [--reference <repository>] [--] <repository> [<path>]\n   or: git submodule [--quiet] "
                "status [--cached] [--recursive] [--] [<path>...]\n   or: git submodule [--quiet] init [--] [<path>...]\n"
                "   or: git submodule [--quiet] update [--init] [--remote] [-N|--no-fetch] [-f|--force] [--checkout|"
                "--merge|--rebase] [--recursive] [--] [<path>...]\n   or: git submodule [--quiet] foreach "
                "[--recursive] <command>")


# ---------------------------------------------------------------------------- sparse-checkout

@command("sparse-checkout")
def git_sparse_checkout(g, args):
    repo = g.repo(wt=True)
    db = repo.odb
    sub = args[0] if args else ""
    rest = args[1:]
    uid, gid = g.ctx.uid, g.ctx.proc.gid
    cfg = repo.config.local_file()

    def apply(dirs):
        flat = db.flat(repo.head_tree())
        idx = repo.index
        for p, (mode, sha) in flat.items():
            keep = dirs is None or "/" not in p or any(p.startswith(d.rstrip("/") + "/") for d in dirs)
            e = idx.get(p)
            if e is None:
                continue
            if keep:
                e.flags2 &= ~0x4000
                if repo.wt_node(p) is None:
                    repo.checkout_file(p, sha, mode)
                    ne = repo.index_entry_for(p, sha, int(mode, 8), repo.wt_node(p))
                    idx.add(ne)
            else:
                node = repo.wt_node(p)
                if mode == "160000" and node is not None and node.type == "d" and node.entries:
                    g.err(f"warning: unable to rmdir '{p}': Directory not empty\n")
                    continue
                e.flags2 |= 0x4000
                if node is not None:
                    repo.remove_file(p)
        repo.write_index()
    if sub in ("set", "add", "init"):
        o = parse(rest, "cone no-cone sparse-index no-sparse-index stdin skip-checks")
        dirs = [d.strip("/") for d in o.args]
        if sub == "add":
            dirs = sorted(set(_sparse_dirs(repo)) | set(dirs))
        cfg.set("core.sparseCheckout", "true", uid, gid)
        cfg.set("core.sparseCheckoutCone", "true", uid, gid)
        repo.reload_config()
        pats = "/*\n!/*/\n" + "".join(f"/{d}/\n" for d in dirs)
        repo.write("info/sparse-checkout", pats, True)
        if sub == "init" and not dirs:
            apply([])
            return 0
        apply(dirs)
        return 0
    if sub == "list":
        if not repo.config.get_bool("core.sparseCheckout", False):
            raise GitError("fatal: this worktree is not sparse")
        for d in _sparse_dirs(repo):
            g.out(d + "\n")
        return 0
    if sub == "disable":
        cfg.set("core.sparseCheckout", "false", uid, gid)
        repo.reload_config()
        apply(None)
        return 0
    if sub == "reapply":
        apply(_sparse_dirs(repo))
        return 0
    raise Usage("usage: git sparse-checkout (init | list | set | add | reapply | disable | check-rules) [<options>]")


def _sparse_dirs(repo):
    text = repo.read("info/sparse-checkout", True) or ""
    return [ln.strip("/") for ln in text.split("\n") if ln.startswith("/") and ln.endswith("/") and ln != "/*/"
            and not ln.startswith("!")]

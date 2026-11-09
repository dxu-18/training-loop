"""Programs that work alongside git: git-lfs (from apt), and the pip-installed git-filter-repo, pre-commit,
nbstripout, dvc and ruff. Each is a simulated program registered like any other utility; pip writes a console
script whose entry module maps to it (utils/pyenv.py CONSOLE / ENTRY)."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re

from ..exec import util
from ..vfs import FSError, LazyData
from .cli import G, run_maybe
from .objects import GitError, encode_commit
from .remote import human_bytes

LFS_VERSION = "git-lfs/3.4.1 (GitHub; linux amd64; go 1.22.2)"


def _report(ctx, e: GitError) -> int:
    ctx.e(e.msg.rstrip("\n") + "\n")
    return e.status


def _repo(g, wt=True):
    return g.repo(wt=wt)


def _rel_paths(g, repo, args):
    return [g.to_rel(repo, a) for a in args]


# ============================================================================ history rewriting (filter-repo, lfs migrate)

def rewrite_history(repo, refs, tree_fn, msg_fn=None, prune_empty=True):
    """Rewrite every commit reachable from `refs` (ref name -> sha). tree_fn(flat) -> new flat dict (or None to keep).
    Returns (old -> new commit map, number of commits parsed)."""
    from .cmd_history import topo_sort
    db = repo.odb
    tips = [s for s in refs.values() if s and db.kind(repo.peel(s)) == "commit"]
    commits = topo_sort(db, list(db.ancestors([repo.peel(s) for s in tips])))
    commits.reverse()                    # parents before children
    mapping: dict[str, str | None] = {}
    tree_of: dict[str, str] = {}
    for c in commits:
        obj = db.commit(c)
        flat = dict(db.flat(obj.tree))
        new_flat = tree_fn(flat, obj)
        tree = db.write_tree(new_flat) if new_flat is not None else obj.tree
        parents = []
        for p in obj.parents:
            np_ = mapping.get(p, p)
            if np_ is not None and np_ not in parents:
                parents.append(np_)
        msg = msg_fn(obj.message) if msg_fn else obj.message
        if prune_empty and len(obj.parents) <= 1 and parents and tree_of.get(parents[0]) == tree and \
                db.commit(c).tree != (db.commit(obj.parents[0]).tree if obj.parents else None):
            mapping[c] = parents[0]          # became empty: dropped
            continue
        if prune_empty and not parents and new_flat is not None and not new_flat and obj.tree != tree:
            mapping[c] = None
            continue
        if tree == obj.tree and tuple(parents) == tuple(obj.parents) and msg == obj.message:
            mapping[c] = c
        else:
            data = encode_commit(tree, parents, obj.author, obj.committer, msg, obj.extra)
            mapping[c] = repo.store("commit", data)
        tree_of[mapping[c]] = tree
    return mapping, len(commits)


def _apply_ref_rewrites(repo, refs, mapping):
    db = repo.odb
    for name, sha in refs.items():
        if not sha:
            continue
        if db.kind(sha) == "tag":
            t = db.tag(sha)
            new = mapping.get(t.object, t.object)
            if new and new != t.object:
                from .objects import encode_tag
                data = encode_tag(new, "commit", t.name, t.tagger, t.message)
                repo.update_ref(name, repo.store("tag", data))
            continue
        new = mapping.get(sha, sha)
        if new is None:
            repo.delete_ref(name)
        elif new != sha:
            repo.update_ref(name, new, "filter-repo: rewrite" if name.startswith("refs/heads/") else None)


# ============================================================================ git-lfs

def _lfs_patterns(repo):
    """[(pattern, file)] of filter=lfs lines in .gitattributes files."""
    out = []
    for pat, base, kv in repo.attributes():
        if "filter=lfs" in kv:
            out.append((pat, (base + "/" if base else "") + ".gitattributes"))
    return out


def _lfs_pointer_files(repo, tree):
    from .lfs import parse_pointer
    db = repo.odb
    out = []
    for p, (mode, sha) in sorted(db.flat(tree).items()):
        if mode == "160000" or db.size(sha) > 400:
            continue
        ptr = parse_pointer(db.raw(sha))
        if ptr:
            out.append((p, ptr[0], ptr[1]))
    return out


def lfs_upload(g, repo, ghrepo, new_shas, old_shas) -> str:
    """What a push sends to the LFS store of a GitHub-lite repository (the pre-push hook's job). -> progress line"""
    from .lfs import active, store
    if not active(repo):
        return ""
    db = repo.odb
    objs = []
    seen = set()
    for c in db.ancestors([s for s in new_shas if s and db.kind(s) == "commit"]):
        if any(db.is_ancestor(c, o) for o in old_shas if o and db.has(o)):
            continue
        for p, oid, size in _lfs_pointer_files(repo, db.commit(c).tree):
            if oid not in seen and oid not in ghrepo.lfs:
                seen.add(oid)
                objs.append((oid, size))
    if not objs:
        return ""
    st = store(g.world)
    total = 0
    for oid, size in objs:
        if oid in st:
            ghrepo.lfs[oid] = st[oid]
            total += size
    n = len(objs)
    return f"Uploading LFS objects: 100% ({n}/{n}), {_mb(total)} | {_mb(total * 2.3)}/s, done.\n"


def _mb(n):
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} MB"
    if n >= 1000:
        return f"{n / 1e3:.0f} KB"
    return f"{int(n)} B"


@util("git-lfs")
def u_git_lfs(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    try:
        return (yield from run_maybe(_lfs(g, a)))
    except GitError as e:
        return _report(ctx, e)


def _lfs(g, a):
    from .lfs import active, have_locally, parse_pointer, pointer_for, store
    sub = a[0] if a else ""
    rest = a[1:]
    ctx = g.ctx
    uid, gid = ctx.uid, ctx.proc.gid
    if sub in ("", "help", "--help", "-h"):
        g.out(f"{LFS_VERSION}\nGit LFS is a system for managing and versioning large files in\nassociation with a Git "
              "repository.\n\nCommon commands: install, track, untrack, ls-files, status, pull, push, fetch, migrate,\n"
              "env, version.\n")
        return 0 if sub else 1
    if sub in ("version", "--version"):
        g.out(LFS_VERSION + "\n")
        return 0
    if sub == "install":
        from .repo import ConfigFile
        home = ctx.env.get("HOME") or "/"
        cfg = ConfigFile(g.fs, f"{home}/.gitconfig")
        for k, v in (("filter.lfs.clean", "git-lfs clean -- %f"), ("filter.lfs.smudge", "git-lfs smudge -- %f"),
                     ("filter.lfs.process", "git-lfs filter-process"), ("filter.lfs.required", "true")):
            cfg.set(k, v, uid, gid)
        repo = g.maybe_repo()
        if repo is not None and "--skip-repo" not in rest:
            for hook in ("pre-push", "post-checkout", "post-commit", "post-merge"):
                text = (f"#!/bin/sh\ncommand -v git-lfs >/dev/null 2>&1 || {{ printf >&2 \"\\n%s\\n\\n\" \"This "
                        f"repository needs git-lfs on your path: install it, or remove .git/hooks/{hook}.\"; exit 2; }}\n"
                        f"git lfs {hook} \"$@\"\n")
                node = g.fs.write_file(repo.path(f"hooks/{hook}", True), text, "/", 0o755, uid, gid)
                node.mode = (node.mode & ~0o777) | 0o755
            g.out("Updated Git hooks.\n")
        g.out("Git LFS initialized.\n")
        return 0
    if sub == "uninstall":
        from .repo import ConfigFile
        home = ctx.env.get("HOME") or "/"
        ConfigFile(g.fs, f"{home}/.gitconfig").remove_section("filter.lfs", uid, gid)
        repo = g.maybe_repo()
        if repo is not None:
            for hook in ("pre-push", "post-checkout", "post-commit", "post-merge"):
                repo.remove(f"hooks/{hook}", True)
            g.out("Hooks for this repository have been removed.\n")
        g.out("Global Git LFS configuration has been removed.\n")
        return 0
    if sub in ("pre-push", "post-checkout", "post-commit", "post-merge"):
        return 0           # the push itself uploads (see lfs_upload); checkout smudges through the filter
    if sub in ("clean", "smudge", "filter-process"):
        return 0
    repo = _repo(g)
    if sub == "track":
        pats = [x for x in rest if not x.startswith("-")]
        if not pats:
            g.out("Listing tracked patterns\n")
            for pat, f in _lfs_patterns(repo):
                g.out(f"    {pat} ({f})\n")
            g.out("Listing excluded patterns\n")
            return 0
        node = repo.wt_node(".gitattributes")
        text = bytes(node.data).decode() if node is not None else ""
        have = {p for p, _ in _lfs_patterns(repo)}
        for pat in pats:
            if pat in have:
                g.out(f'"{pat}" already supported\n')
                continue
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"{pat} filter=lfs diff=lfs merge=lfs -text\n"
            g.out(f'Tracking "{pat}"\n')
        g.fs.write_file(repo.wt_path(".gitattributes"), text, "/", 0o644, uid, gid)
        return 0
    if sub == "untrack":
        node = repo.wt_node(".gitattributes")
        text = bytes(node.data).decode() if node is not None else ""
        lines = text.split("\n")
        for pat in rest:
            keep = [ln for ln in lines if not (ln.split()[:1] == [pat] and "filter=lfs" in ln)]
            if len(keep) != len(lines):
                g.out(f'Untracking "{pat}"\n')
            lines = keep
        g.fs.write_file(repo.wt_path(".gitattributes"), "\n".join(lines), "/", 0o644, uid, gid)
        return 0
    if sub == "ls-files":
        show_size = "-s" in rest or "--size" in rest
        long = "-l" in rest or "--long" in rest
        db = repo.odb
        head = repo.head()
        if not head:
            return 0
        for p, oid, size in _lfs_pointer_files(repo, db.commit(head).tree):
            node = repo.wt_node(p)
            here = node is not None and parse_pointer(node.data) is None
            mark = "*" if here else "-"
            g.out(f"{oid if long else oid[:10]} {mark} {p}" + (f" ({_mb(size)})" if show_size else "") + "\n")
        return 0
    if sub == "status":
        from .status import collect
        st = collect(repo)
        g.out("On branch " + (repo.head_branch() or "HEAD") + "\n")
        staged = [p for p in sorted(st.staged) if repo.lfs_tracked(p)]
        unstaged = [p for p in sorted(st.unstaged) if repo.lfs_tracked(p)]
        g.out("Objects to be pushed to origin/" + (repo.head_branch() or "main") + ":\n\n")
        g.out("\nObjects to be committed:\n\n")
        for p in staged:
            g.out(f"\t{p} (LFS: {_lfs_short(repo, p)})\n")
        g.out("\nObjects not staged for commit:\n\n")
        for p in unstaged:
            g.out(f"\t{p} (File: {_lfs_short(repo, p)})\n")
        g.out("\n")
        return 0
    if sub in ("pull", "fetch", "checkout"):
        if not active(repo):
            raise GitError("Git LFS is not installed in this repository: run git lfs install first.", 2)
        db = repo.odb
        head = repo.head()
        if not head:
            return 0
        files = _lfs_pointer_files(repo, db.commit(head).tree)
        st = store(g.world)
        got = 0
        n = 0
        for p, oid, size in files:
            if oid not in st:
                continue
            if sub != "checkout" and not have_locally(repo, oid):
                n += 1
                got += size
            if sub == "fetch":
                node = g.fs.write_file(repo.path(f"lfs/objects/{oid[:2]}/{oid[2:4]}/{oid}", True), b"", "/", 0o644,
                                       uid, gid)
                node.data = st[oid].clone() if isinstance(st[oid], LazyData) else LazyData(src=bytes(st[oid]))
                continue
            node = repo.wt_node(p)
            if node is not None and parse_pointer(node.data) is None:
                continue
            mode = db.flat(db.commit(head).tree)[p][0]
            sha = db.flat(db.commit(head).tree)[p][1]
            repo.checkout_file(p, sha, mode)
            e = repo.index.get(p)
            if e is not None:
                fresh = repo.wt_node(p)
                e.mtime, e.ctime, e.ino, e.size = fresh.mtime, fresh.ctime, fresh.ino, len(fresh.data)
        repo.write_index()
        if n and sub != "checkout":
            g.err(f"Downloading LFS objects: 100% ({n}/{n}), {_mb(got)} | {_mb(got * 1.8)}/s, done.\n")
        return 0
    if sub == "push":
        from .cmd_remote import connect
        remote = rest[0] if rest else "origin"
        url = repo.config.get(f"remote.{remote}.url") or remote
        far = yield from connect(g, url)
        if far.ghrepo is None:
            return 0
        heads = [repo.resolve_ref(f"refs/heads/{b}") for b in rest[1:]] or [repo.head()]
        line = lfs_upload(g, repo, far.ghrepo, heads, [])
        g.err(line or "")
        return 0
    if sub == "migrate":
        return (yield from _lfs_migrate(g, repo, rest))
    if sub == "env":
        g.out(f"{LFS_VERSION}\ngit version 2.43.0\n\nEndpoint={repo.config.get('remote.origin.url') or ''}/info/lfs "
              f"(auth=none)\nLocalWorkingDir={repo.worktree}\nLocalGitDir={repo.gitdir}\n"
              f"LocalMediaDir={repo.common}/lfs/objects\n")
        return 0
    g.err(f"Error: unknown command \"{sub}\" for \"git-lfs\"\nRun 'git-lfs --help' for usage.\n")
    return 1


def _lfs_short(repo, p):
    node = repo.wt_node(p)
    if node is None:
        return "deleted"
    from .lfs import oid_of
    return oid_of(node.data)[:10]


def _lfs_migrate(g, repo, rest):
    from .lfs import parse_pointer, pointer_for
    mode = rest[0] if rest else ""
    includes = []
    everything = "--everything" in rest
    for x in rest[1:]:
        m = re.match(r"^--include=(.+)$", x)
        if m:
            includes += [p.strip().strip("'\"") for p in m.group(1).split(",")]
    if mode == "info":
        db = repo.odb
        sizes: dict[str, list] = {}
        for p, (md, sha) in db.flat(repo.head_tree()).items():
            ext = "*." + p.rsplit(".", 1)[-1] if "." in p.rsplit("/", 1)[-1] else p
            sizes.setdefault(ext, [0, 0])
            sizes[ext][0] += db.size(sha)
            sizes[ext][1] += 1
        for ext, (n, c) in sorted(sizes.items(), key=lambda kv: -kv[1][0])[:5]:
            g.out(f"{ext:<12}\t{_mb(n):>8}\t{c}/{c} files\t100%\n")
        return 0
    if mode != "import" or not includes:
        raise GitError("Usage: git lfs migrate import --include=<patterns> [--everything]", 2)
    db = repo.odb
    refs = repo.ref_names("refs/heads/") if everything else {f"refs/heads/{repo.head_branch()}": repo.head()}
    if everything:
        refs.update({k: v for k, v in repo.ref_names("refs/tags/").items()})

    def matches(p):
        name = p.rsplit("/", 1)[-1]
        return any(fnmatch.fnmatchcase(p if "/" in pat else name, pat) for pat in includes)

    attrs_line = "".join(f"{pat} filter=lfs diff=lfs merge=lfs -text\n" for pat in includes)

    def tree_fn(flat, commit):
        changed = False
        for p, (md, sha) in list(flat.items()):
            if md == "160000" or not matches(p):
                continue
            data = db.raw(sha)
            if parse_pointer(data) is not None:
                continue
            ptr = pointer_for(repo, data) if _lfs_on(repo) else _pointer_raw(repo, data)
            flat[p] = (md, repo.store("blob", ptr))
            changed = True
        old = flat.get(".gitattributes")
        text = bytes(db.raw(old[1])).decode() if old else ""
        if attrs_line not in text:
            flat[".gitattributes"] = ("100644", repo.store("blob", (text + attrs_line).encode()))
            changed = True
        return flat if changed else None

    mapping, n = rewrite_history(repo, refs, tree_fn, prune_empty=False)
    _apply_ref_rewrites(repo, refs, mapping)
    g.err(f"migrate: Sorting commits: ..., done.\nmigrate: Rewriting commits: 100% ({n}/{n}), done.\n")
    for name, sha in refs.items():
        if name.startswith("refs/heads/"):
            new = mapping.get(sha, sha)
            g.err(f"  {name[len('refs/heads/'):]}\t{sha} -> {new}\n")
    g.err("migrate: Updating refs: ..., done.\nmigrate: checkout: ..., done.\n")
    _reset_hard(g, repo, repo.head())
    return 0


def _reset_hard(g, repo, sha):
    from .switching import checkout_tree
    if sha and repo.worktree:
        checkout_tree(g, repo, repo.odb.commit(sha).tree, force=True)


def _lfs_on(repo):
    from .lfs import active
    return active(repo)


def _pointer_raw(repo, data):
    from .lfs import oid_of, pointer_text, store
    oid = oid_of(data)
    store(repo.ctx.world)[oid] = data if isinstance(data, LazyData) else LazyData(src=bytes(data))
    return pointer_text(oid, len(data))


# ============================================================================ git-filter-repo

def _parse_size(s: str) -> int:
    m = re.match(r"^(\d+)([KMG]?)$", s.upper())
    if not m:
        raise GitError(f"Error: could not parse size '{s}'", 2)
    return int(m.group(1)) * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]


@util("git-filter-repo")
def u_git_filter_repo(ctx):
    g = G(ctx)
    try:
        return (yield from run_maybe(_filter_repo(g, list(ctx.argv[1:]))))
    except GitError as e:
        return _report(ctx, e)


def _filter_repo(g, a):
    paths, globs, renames, replace_file, strip_big, msg_file = [], [], [], None, None, None
    invert = force = dry = False
    subdir = to_subdir = None
    i = 0
    while i < len(a):
        x = a[i]
        val = a[i + 1] if i + 1 < len(a) else None
        if "=" in x and x.startswith("--"):
            x, val = x.split("=", 1)
            i -= 1
        if x == "--path":
            paths.append(val.rstrip("/")); i += 2; continue
        if x == "--path-glob":
            globs.append(val); i += 2; continue
        if x == "--path-rename":
            renames.append(tuple(val.split(":", 1))); i += 2; continue
        if x == "--subdirectory-filter":
            subdir = val.strip("/"); i += 2; continue
        if x == "--to-subdirectory-filter":
            to_subdir = val.strip("/"); i += 2; continue
        if x == "--replace-text":
            replace_file = val; i += 2; continue
        if x == "--replace-message":
            msg_file = val; i += 2; continue
        if x == "--strip-blobs-bigger-than":
            strip_big = _parse_size(val); i += 2; continue
        if x == "--invert-paths":
            invert = True
        elif x in ("--force", "-f"):
            force = True
        elif x == "--dry-run":
            dry = True
        elif x in ("-h", "--help"):
            g.out("usage: git-filter-repo [--path PATH] [--invert-paths] [--path-glob GLOB] [--path-rename OLD:NEW]\n"
                  "                       [--subdirectory-filter DIR] [--to-subdirectory-filter DIR]\n"
                  "                       [--replace-text FILE] [--replace-message FILE]\n"
                  "                       [--strip-blobs-bigger-than SIZE] [--force] [--dry-run]\n")
            return 0
        elif x == "--analyze":
            return _filter_analyze(g)
        else:
            raise GitError(f"git-filter-repo: error: unrecognized arguments: {x}", 2)
        i += 1
    repo = _repo(g)
    db = repo.odb
    if not force and not _fresh_clone(repo):
        raise GitError("Aborting: Refusing to destructively overwrite repo history since\nthis does not look like a "
                       "fresh clone.\n  (expected freshly packed repo)\nPlease operate on a fresh clone instead.  If "
                       "you want to proceed\nanyway, use --force.", 1)
    replacements = []
    if replace_file:
        text = _read_text(g, replace_file)
        for ln in text.split("\n"):
            if not ln.strip():
                continue
            src, sep, dst = ln.partition("==>")
            dst = dst if sep else "***REMOVED***"
            if src.startswith("regex:"):
                replacements.append((re.compile(src[6:].encode()), dst.encode()))
            else:
                src = src[8:] if src.startswith("literal:") else src
                replacements.append((src.encode(), dst.encode()))
    msg_repl = []
    if msg_file:
        for ln in _read_text(g, msg_file).split("\n"):
            if ln.strip():
                src, sep, dst = ln.partition("==>")
                msg_repl.append((src, dst if sep else "***REMOVED***"))

    def selected(p):
        if not paths and not globs:
            return True
        return any(p == q or p.startswith(q + "/") for q in paths) or any(fnmatch.fnmatchcase(p, gl) for gl in globs)

    blob_cache: dict[str, str] = {}

    def tree_fn(flat, commit):
        out = {}
        for p, (md, sha) in flat.items():
            if paths or globs:
                if selected(p) == invert:
                    continue
            if subdir:
                if not p.startswith(subdir + "/"):
                    continue
                p = p[len(subdir) + 1:]
            for old, new in renames:
                if p == old or p.startswith(old.rstrip("/") + "/"):
                    p = new + p[len(old):]
                    break
            if to_subdir:
                p = f"{to_subdir}/{p}"
            if md != "160000":
                if strip_big is not None and db.size(sha) > strip_big:
                    continue
                if replacements:
                    if sha not in blob_cache:
                        data = bytes(db.raw(sha))
                        new = data
                        for src, dst in replacements:
                            new = src.sub(dst, new) if hasattr(src, "sub") else new.replace(src, dst)
                        blob_cache[sha] = repo.store("blob", new) if new != data else sha
                    sha = blob_cache[sha]
            out[p] = (md, sha)
        return out

    def msg_fn(m):
        for src, dst in msg_repl:
            m = m.replace(src, dst)
        return m

    refs = {k: v for k, v in repo.ref_names("refs/").items() if not k.startswith("refs/remotes/")
            and k != "refs/stash"}
    if dry:
        g.out("NOTE: Not running fast-export or fast-import; --dry-run passed.\n")
        return 0
    mapping, n = rewrite_history(repo, refs, tree_fn, msg_fn if msg_repl else None)
    _apply_ref_rewrites(repo, refs, mapping)
    # the old history goes away: remote-tracking refs, reflogs, the stash, origin itself
    for name in list(repo.ref_names("refs/remotes/")):
        repo.delete_ref(name)
    repo.delete_ref("refs/stash")
    repo.rmtree("logs", True)
    url = repo.config.get("remote.origin.url")
    if url:
        cfg = repo.config.local_file()
        cfg.remove_section("remote.origin", g.ctx.uid, g.ctx.proc.gid)
        repo.reload_config()
    db.cut = set()
    g.out(f"Parsed {n} commits\nNew history written in 0.04 seconds; now repacking/cleaning...\n"
          f"Repacking your repo and cleaning out old unneeded objects\n")
    _reset_hard(g, repo, repo.head())
    h = repo.head()
    if h:
        g.out(f"HEAD is now at {db.abbrev(h)} {db.commit(h).subject}\n")
    g.out("Enumerating objects: done.\nCompletely finished after 0.12 seconds.\n")
    if url:
        g.err(f"NOTICE: Removing 'origin' remote; see 'Why is my origin removed?'\n        in the manual if you want "
              f"to push back there.\n        (was {url})\n")
    return 0


def _fresh_clone(repo) -> bool:
    """filter-repo only rewrites a freshly cloned repository (no local work that could be lost)."""
    if repo.resolve_ref("refs/stash"):
        return False
    log = repo.reflog("HEAD")
    return len(log) <= 1 and len(repo.branches()) <= 1


def _filter_analyze(g):
    repo = _repo(g)
    db = repo.odb
    out = repo.path("filter-repo/analysis", True)
    rows = {}
    for c in db.ancestors([s for s in repo.ref_names("refs/heads/").values()]):
        for p, (md, sha) in db.flat(db.commit(c).tree).items():
            if md != "160000":
                rows[p] = max(rows.get(p, 0), db.size(sha))
    text = "=== Files by sizes ===\n" + "".join(f"{v:>12}  {p}\n" for p, v in sorted(rows.items(), key=lambda kv: -kv[1]))
    g.fs.mkdir(out, "/", 0o755, g.ctx.uid, g.ctx.proc.gid, parents=True)
    g.fs.write_file(f"{out}/path-all-sizes.txt", text, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
    g.out(f"Processed {len(rows)} blob sizes\nWriting reports to {out}...done.\n")
    return 0


def _read_text(g, path):
    p = path if path.startswith("/") else posixpath.join(g.cwd, path)
    try:
        return bytes(g.fs.lookup(p).data).decode("utf-8", "replace")
    except FSError:
        raise GitError(f"Error: {path}: No such file or directory", 2)


# ============================================================================ nbstripout

def strip_notebook(data: bytes, keep_count=False):
    """A notebook without outputs and execution counts (None if it isn't one)."""
    try:
        nb = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(nb, dict) or "cells" not in nb:
        return None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            if not keep_count:
                cell["execution_count"] = None
        md = cell.get("metadata")
        if isinstance(md, dict):
            for k in ("collapsed", "scrolled", "ExecuteTime", "execution"):
                md.pop(k, None)
    meta = nb.get("metadata", {})
    if isinstance(meta, dict):
        meta.pop("widgets", None)
    return (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode()


@util("nbstripout")
def u_nbstripout(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    try:
        if "--install" in a or "--uninstall" in a or "--status" in a or "--is-installed" in a:
            repo = _repo(g)
            uid, gid = ctx.uid, ctx.proc.gid
            cfg = repo.config.local_file()
            attrs_file = "info/attributes"
            if "--install" in a:
                py = "/usr/bin/python3"
                cfg.set("filter.nbstripout.clean", f"'{py}' -m nbstripout", uid, gid)
                cfg.set("filter.nbstripout.smudge", "cat", uid, gid)
                cfg.set("diff.ipynb.textconv", f"'{py}' -m nbstripout -t", uid, gid)
                if "--attributes" in a:
                    pass
                text = repo.read(attrs_file, True) or ""
                for ln in ("*.ipynb filter=nbstripout", "*.zpln filter=nbstripout", "*.ipynb diff=ipynb"):
                    if ln not in text:
                        text += ln + "\n"
                g.fs.mkdir(repo.path("info", True), "/", 0o755, uid, gid, parents=True)
                repo.write(attrs_file, text, True)
                return 0
            if "--uninstall" in a:
                cfg.remove_section("filter.nbstripout", uid, gid)
                cfg.remove_section("diff.ipynb", uid, gid)
                text = repo.read(attrs_file, True) or ""
                repo.write(attrs_file, "".join(ln + "\n" for ln in text.split("\n") if ln and "nbstripout" not in ln
                                               and "diff=ipynb" not in ln), True)
                return 0
            ok = bool(repo.config.get("filter.nbstripout.clean"))
            if "--is-installed" in a:
                return 0 if ok else 1
            if ok:
                g.out(f"nbstripout is installed in repository '{repo.worktree}'\n\nFilter:\n  clean = "
                      f"{repo.config.get('filter.nbstripout.clean')}\n  smudge = cat\n  diff= "
                      f"{repo.config.get('diff.ipynb.textconv')}\n\nAttributes:\n  *.ipynb: filter: nbstripout\n")
                return 0
            g.out(f"nbstripout is not installed in repository '{repo.worktree}'\n")
            return 1
        files = [x for x in a if not x.startswith("-")]
        keep_count = "--keep-count" in a
        dry = "--dry-run" in a
        to_stdout = "-t" in a or "--textconv" in a
        if not files:
            data = yield from _read_stdin(ctx)
            out = strip_notebook(data, keep_count)
            ctx.w((out if out is not None else data).decode("utf-8", "replace"))
            return 0
        st = 0
        for f in files:
            p = ctx.abspath(f)
            try:
                node = g.fs.lookup(p)
            except FSError:
                ctx.e(f"Could not strip '{f}': file not found\n")
                st = 1
                continue
            out = strip_notebook(bytes(node.data), keep_count)
            if out is None:
                ctx.e(f"Could not strip '{f}': invalid notebook\n")
                st = 1
                continue
            if to_stdout:
                ctx.w(out.decode("utf-8", "replace"))
            elif dry:
                if out != bytes(node.data):
                    ctx.w(f"Dry run: would have stripped {f}\n")
            elif out != bytes(node.data):
                g.fs.write_file(p, out, "/", node.mode & 0o777, node.uid, node.gid)
        return st
    except GitError as e:
        return _report(ctx, e)


def _read_stdin(ctx):
    try:
        r = ctx.reader("-")
    except FSError:
        return b""
    chunks = []
    while True:
        ln = yield from r.readline()
        if not ln:
            break
        chunks.append(ln)
    return b"".join(chunks)


# ============================================================================ ruff (a small subset)

def _ruff_check(text: str):
    """[(line, col, code, message)]: unused imports (F401), `== None` (E711), bare except (E722), unused f-strings."""
    probs = []
    lines = text.split("\n")
    body = "\n".join(ln for ln in lines if not re.match(r"^\s*(import|from)\s", ln))
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^(\s*)import\s+([\w.]+)(?:\s+as\s+(\w+))?\s*$", ln)
        if m:
            name = m.group(3) or m.group(2).split(".")[0]
            if not re.search(rf"\b{re.escape(name)}\b", body):
                probs.append((i, len(m.group(1)) + 8, "F401", f"`{m.group(2)}` imported but unused"))
            continue
        m = re.match(r"^(\s*)from\s+([\w.]+)\s+import\s+(\w+)(?:\s+as\s+(\w+))?\s*$", ln)
        if m:
            name = m.group(4) or m.group(3)
            if not re.search(rf"\b{re.escape(name)}\b", body):
                probs.append((i, ln.index(m.group(3)) + 1, "F401", f"`{m.group(2)}.{m.group(3)}` imported but unused"))
            continue
        k = ln.find("== None")
        if k >= 0:
            probs.append((i, k + 1, "E711", "Comparison to `None` should be `cond is None`"))
        if re.match(r"^\s*except\s*:", ln):
            probs.append((i, ln.index("except") + 1, "E722", "Do not use bare `except`"))
    return probs


def _ruff_fix(text: str) -> str:
    drop = {ln for ln, _, code, _ in _ruff_check(text) if code == "F401"}
    return "\n".join(x for i, x in enumerate(text.split("\n"), 1) if i not in drop)


def _ruff_format(text: str) -> str:
    out = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        # simple single-quoted strings -> double quotes (when that needs no escaping)
        ln = re.sub(r"'([^'\"\\\n]*)'", r'"\1"', ln) if "#" not in ln else ln
        ln = re.sub(r"\s*=\s*(?![=])", " = ", ln) if re.match(r"^\s*\w+\s*=[^=]", ln) else ln
        ln = re.sub(r",(?=\S)", ", ", ln) if not re.search(r"[\"']", ln) else ln
        out.append(ln)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def _py_files(g, args):
    out = []
    for a in args or ["."]:
        p = g.ctx.abspath(a)
        try:
            node = g.fs.lookup(p)
        except FSError:
            continue
        if node.type == "f":
            out.append((a, p))
            continue
        stack = [(a.rstrip("/"), p)]
        while stack:
            rel, ab = stack.pop()
            n = g.fs.lookup(ab)
            for name in sorted(n.entries, reverse=True):
                if name.startswith(".") or name in ("venv", "__pycache__", "node_modules"):
                    continue
                child = n.entries[name]
                r = name if rel == "." else f"{rel}/{name}"
                if child.type == "d":
                    stack.append((r, f"{ab}/{name}"))
                elif name.endswith((".py", ".pyi")):
                    out.append((r, f"{ab}/{name}"))
    return sorted(out)


@util("ruff")
def u_ruff(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    sub = a[0] if a else ""
    rest = [x for x in a[1:] if not x.startswith("-")]
    if sub in ("--version", "version"):
        ctx.w("ruff 0.14.1\n")
        return 0
    if sub == "check":
        fix = "--fix" in a
        total = fixed = 0
        lines = []
        for rel, p in _py_files(g, rest):
            text = bytes(g.fs.lookup(p).data).decode("utf-8", "replace")
            probs = _ruff_check(text)
            if fix and any(c == "F401" for _, _, c, _ in probs):
                new = _ruff_fix(text)
                fixed += sum(1 for _, _, c, _ in probs if c == "F401")
                node = g.fs.lookup(p)
                g.fs.write_file(p, new, "/", node.mode & 0o777, node.uid, node.gid)
                probs = _ruff_check(new)
            for ln, col, code, msg in probs:
                lines.append(f"{rel}:{ln}:{col}: {code} {'[*] ' if code == 'F401' else ''}{msg}")
            total += len(probs)
        if lines:
            ctx.w("\n".join(lines) + "\n")
        if total:
            n_fix = sum(1 for x in lines if "[*]" in x)
            ctx.w(f"Found {total + fixed} error{'s' if total + fixed != 1 else ''}" +
                  (f" ({fixed} fixed, {total} remaining)" if fixed else "") + ".\n" +
                  (f"[*] {n_fix} fixable with the `--fix` option.\n" if n_fix and not fix else ""))
            return 1
        ctx.w(f"Found {fixed} error{'s' if fixed != 1 else ''} ({fixed} fixed, 0 remaining).\n" if fixed else
              "All checks passed!\n")
        return 0
    if sub == "format":
        check = "--check" in a
        changed = same = 0
        for rel, p in _py_files(g, rest):
            node = g.fs.lookup(p)
            text = bytes(node.data).decode("utf-8", "replace")
            new = _ruff_format(text)
            if new != text:
                changed += 1
                if check:
                    ctx.w(f"Would reformat: {rel}\n")
                else:
                    g.fs.write_file(p, new, "/", node.mode & 0o777, node.uid, node.gid)
            else:
                same += 1
        parts = []
        if changed:
            parts.append(f"{changed} file{'s' if changed != 1 else ''} {'would be ' if check else ''}reformatted")
        if same:
            parts.append(f"{same} file{'s' if same != 1 else ''} {'already formatted' if check else 'left unchanged'}")
        ctx.w(", ".join(parts) + "\n")
        return 1 if check and changed else 0
    ctx.e("error: unrecognized subcommand; try 'ruff check' or 'ruff format'\n")
    return 2
    yield


# ============================================================================ pre-commit

HOOKS = {
    # id: (shown name, kind)
    "trailing-whitespace": "Trim Trailing Whitespace",
    "end-of-file-fixer": "Fix End of Files",
    "check-yaml": "Check Yaml",
    "check-json": "Check JSON",
    "check-added-large-files": "Check for added large files",
    "check-merge-conflict": "Check for merge conflicts",
    "detect-private-key": "Detect Private Key",
    "no-commit-to-branch": "Don't commit to branch",
    "ruff": "ruff",
    "ruff-check": "ruff check",
    "ruff-format": "ruff format",
    "black": "black",
    "nbstripout": "nbstripout",
}
FILES = {"check-yaml": (".yaml", ".yml"), "check-json": (".json",), "ruff": (".py", ".pyi"), "ruff-check": (".py", ".pyi"),
         "ruff-format": (".py", ".pyi"), "black": (".py", ".pyi"), "nbstripout": (".ipynb",)}


def _precommit_config(g, repo):
    node = repo.wt_node(".pre-commit-config.yaml")
    if node is None:
        return None
    text = bytes(node.data).decode("utf-8", "replace")
    repos = []
    cur = None
    for ln in text.split("\n"):
        m = re.match(r"^\s*-\s*repo:\s*(\S+)", ln)
        if m:
            cur = {"repo": m.group(1), "hooks": []}
            repos.append(cur)
            continue
        m = re.match(r"^\s*-\s*id:\s*([\w.-]+)", ln)
        if m and cur is not None:
            cur["hooks"].append({"id": m.group(1), "args": []})
            continue
        m = re.match(r"^\s*args:\s*\[(.*)\]", ln)
        if m and cur is not None and cur["hooks"]:
            cur["hooks"][-1]["args"] = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    return repos


def _dots(name, status, extra=""):
    width = 79 - len(status) - len(extra)
    return f"{name}{'.' * max(3, width - len(name))}{extra}{status}"


def _run_hook(g, repo, hook, files):
    """-> (status 'Passed'|'Failed'|'Skipped', detail lines, modified paths)"""
    hid = hook["id"]
    exts = FILES.get(hid)
    if exts is not None:
        files = [f for f in files if f.endswith(exts)]
    if not files and hid != "no-commit-to-branch":
        return "Skipped", [], []
    fs = g.fs
    modified, msgs = [], []

    def text_of(f):
        node = repo.wt_node(f)
        return None if node is None or node.type != "f" else bytes(node.data)

    def put(f, data):
        node = repo.wt_node(f)
        fs.write_file(repo.wt_path(f), data, "/", node.mode & 0o777, node.uid, node.gid)
        modified.append(f)

    if hid == "trailing-whitespace":
        for f in files:
            d = text_of(f)
            if d is None or b"\0" in d[:8000]:
                continue
            new = b"\n".join(ln.rstrip(b" \t") for ln in d.split(b"\n"))
            if new != d:
                put(f, new)
                msgs.append(f"Fixing {f}")
    elif hid == "end-of-file-fixer":
        for f in files:
            d = text_of(f)
            if d is None or not d or b"\0" in d[:8000]:
                continue
            new = d.rstrip(b"\n") + b"\n"
            if new != d:
                put(f, new)
                msgs.append(f"Fixing {f}")
    elif hid == "check-yaml":
        for f in files:
            d = (text_of(f) or b"").decode("utf-8", "replace")
            for i, ln in enumerate(d.split("\n"), 1):
                if ln.startswith("\t") or re.match(r"^\s*[\w-]+:\s*[\w\"']+:\s*\S", ln):
                    msgs.append(f"while scanning {f}, line {i}: found character that cannot start any token")
                    break
    elif hid == "check-json":
        for f in files:
            try:
                json.loads((text_of(f) or b"").decode("utf-8"))
            except ValueError as e:
                msgs.append(f"{f}: Failed to json decode ({e})")
    elif hid == "check-added-large-files":
        maxkb = 500
        for x in hook.get("args", []):
            m = re.match(r"--maxkb=(\d+)", x)
            if m:
                maxkb = int(m.group(1))
        for f in files:
            node = repo.wt_node(f)
            if node is not None and node.type == "f" and not repo.lfs_tracked(f):
                kb = (len(node.data) + 1023) // 1024
                if kb > maxkb:
                    msgs.append(f"{f} ({kb} KB) exceeds {maxkb} KB.")
    elif hid == "check-merge-conflict":
        for f in files:
            d = text_of(f) or b""
            for i, ln in enumerate(d.split(b"\n"), 1):
                if ln.startswith((b"<<<<<<< ", b">>>>>>> ")) or ln == b"=======":
                    msgs.append(f"{f}:{i}: Merge conflict string {ln[:7].decode()!r} found")
                    break
    elif hid == "detect-private-key":
        for f in files:
            d = text_of(f) or b""
            if re.search(rb"BEGIN (RSA |DSA |EC |OPENSSH |)PRIVATE KEY", d):
                msgs.append(f"Private key found: {f}")
    elif hid == "no-commit-to-branch":
        branch = repo.head_branch()
        protected = [x.split("=", 1)[1] for x in hook.get("args", []) if x.startswith("--branch")] or ["main", "master"]
        if branch in protected:
            msgs.append("")
            return "Failed", [], []
        return "Passed", [], []
    elif hid in ("ruff", "ruff-check"):
        fix = "--fix" in hook.get("args", [])
        for f in files:
            text = (text_of(f) or b"").decode("utf-8", "replace")
            probs = _ruff_check(text)
            if fix and any(c == "F401" for _, _, c, _ in probs):
                put(f, _ruff_fix(text).encode())
                probs = _ruff_check(_ruff_fix(text))
            for ln, col, code, msg in probs:
                msgs.append(f"{f}:{ln}:{col}: {code} {msg}")
        if msgs:
            msgs.append(f"Found {len(msgs)} error{'s' if len(msgs) != 1 else ''}.")
    elif hid in ("ruff-format", "black"):
        n = 0
        for f in files:
            text = (text_of(f) or b"").decode("utf-8", "replace")
            new = _ruff_format(text)
            if new != text:
                put(f, new.encode())
                n += 1
        if n:
            msgs.append(f"{n} file{'s' if n != 1 else ''} reformatted, {len(files) - n} file"
                        f"{'s' if len(files) - n != 1 else ''} left unchanged")
    elif hid == "nbstripout":
        for f in files:
            d = text_of(f)
            out = strip_notebook(d or b"")
            if out is not None and out != d:
                put(f, out)
    else:
        return "Passed", [], []
    if modified or msgs:
        return "Failed", msgs, modified
    return "Passed", [], []


@util("pre-commit")
def u_pre_commit(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    try:
        return (yield from run_maybe(_pre_commit(g, a)))
    except GitError as e:
        return _report(ctx, e)


def _pre_commit(g, a):
    ctx = g.ctx
    sub = a[0] if a else "run"
    if sub in ("--version", "-V"):
        g.out("pre-commit 4.3.0\n")
        return 0
    if sub in ("-h", "--help", "help"):
        g.out("usage: pre-commit [-h] [-V]\n                  {autoupdate,clean,gc,init-templatedir,install,"
              "install-hooks,migrate-config,run,sample-config,uninstall,validate-config,validate-manifest,help}\n")
        return 0
    if sub == "sample-config":
        g.out("# See https://pre-commit.com for more information\n# See https://pre-commit.com/hooks.html for more "
              "hooks\nrepos:\n-   repo: https://github.com/pre-commit/pre-commit-hooks\n    rev: v3.2.0\n    hooks:\n"
              "    -   id: trailing-whitespace\n    -   id: end-of-file-fixer\n    -   id: check-yaml\n"
              "    -   id: check-added-large-files\n")
        return 0
    repo = _repo(g)
    if sub == "install":
        if repo.config.get("core.hooksPath"):
            raise GitError("[ERROR] Cowardly refusing to install hooks with `core.hooksPath` set.\nhint: `git config "
                           "--unset-all core.hooksPath`", 1)
        exe = getattr(ctx, "path", None) or "pre-commit"
        text = (f"#!/usr/bin/env bash\n# File generated by pre-commit: https://pre-commit.com\nINSTALL_PYTHON={exe}\n"
                f"exec \"$INSTALL_PYTHON\" hook-impl --config=.pre-commit-config.yaml --hook-type=pre-commit "
                f"--hook-dir \"$(cd \"$(dirname \"$0\")\" && pwd)\" -- \"$@\"\n")
        g.fs.mkdir(repo.path("hooks", True), "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
        node = g.fs.write_file(repo.path("hooks/pre-commit", True), text, "/", 0o755, ctx.uid, ctx.proc.gid)
        node.mode = (node.mode & ~0o777) | 0o755
        g.out("pre-commit installed at .git/hooks/pre-commit\n")
        return 0
    if sub == "uninstall":
        repo.remove("hooks/pre-commit", True)
        g.out(".git/hooks/pre-commit uninstalled\n")
        return 0
    if sub == "autoupdate":
        cfg = _precommit_config(g, repo) or []
        for r in cfg:
            g.out(f"[{r['repo']}] already up to date!\n")
        return 0
    if sub in ("clean", "gc", "install-hooks", "validate-config"):
        return 0
    if sub not in ("run", "hook-impl"):
        g.err(f"usage: pre-commit [-h] [-V] ...\npre-commit: error: argument command: invalid choice: '{sub}'\n")
        return 2
    all_files = "--all-files" in a or "-a" in a
    only = [x for x in a[1:] if not x.startswith("-")]
    files_opt = []
    if "--files" in a:
        files_opt = a[a.index("--files") + 1:]
        only = []
    cfg = _precommit_config(g, repo)
    if cfg is None:
        g.out("No .pre-commit-config.yaml file was found\n- To temporarily silence this, run `PRE_COMMIT_ALLOW_NO_CONFIG"
              "=1 git ...`\n- To permanently silence this, install pre-commit with the --allow-missing-config option\n"
              "- To uninstall pre-commit run `pre-commit uninstall`\n")
        return 1
    from .status import collect
    if sub == "hook-impl":
        only = []
    if all_files:
        files = sorted(repo.index.stage0())
    elif files_opt:
        files = [g.to_rel(repo, f) for f in files_opt]
    else:
        st = collect(repo)
        files = sorted(p for p, v in st.staged.items() if v != "D")
        if sub == "run" and st.unstaged and not all_files:
            pass
    # the first run sets up each hook repository's environment
    cache = f"{ctx.env.get('HOME', '/root')}/.cache/pre-commit"
    for r in cfg:
        slug = re.sub(r"\W+", "_", r["repo"])
        if r["repo"] in ("local", "meta"):
            continue
        if not g.fs.exists(f"{cache}/{slug}"):
            g.fs.mkdir(f"{cache}/{slug}", "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
            g.out(f"[INFO] Initializing environment for {r['repo']}.\n")
            g.out(f"[INFO] Installing environment for {r['repo']}.\n[INFO] Once installed this environment will be "
                  "reused.\n[INFO] This may take a few minutes...\n")
    failed = False
    for r in cfg:
        for hook in r["hooks"]:
            if only and hook["id"] not in only:
                continue
            name = HOOKS.get(hook["id"], hook["id"])
            status, msgs, modified = _run_hook(g, repo, hook, files)
            if status == "Skipped":
                g.out(_dots(name, "Skipped", "(no files to check)") + "\n")
                continue
            g.out(_dots(name, status) + "\n")
            if status == "Failed":
                failed = True
                g.out(f"- hook id: {hook['id']}\n")
                if hook["id"] not in ("nbstripout",) or not modified:
                    g.out("- exit code: 1\n")
                if modified:
                    g.out("- files were modified by this hook\n")
                body = [m for m in msgs if m]
                if body:
                    g.out("\n" + "\n".join(body) + "\n\n")
    return 1 if failed else 0


# ============================================================================ dvc

def _dvc_root(g):
    d = g.cwd
    while True:
        if g.fs.exists(f"{d}/.dvc"):
            return d
        if d == "/":
            return None
        d = posixpath.dirname(d)


def _md5(data) -> str:
    if isinstance(data, LazyData) and not data.materialized and data._src is None:
        return hashlib.md5(f"lfs:{data._seed}:{len(data)}".encode()).hexdigest()
    return hashlib.md5(bytes(data)).hexdigest()


def _dvc_store(world) -> dict:
    s = world.__dict__.get("dvc_remotes")
    if s is None:
        s = {}
        world.dvc_remotes = s
    return s


@util("dvc")
def u_dvc(ctx):
    a = list(ctx.argv[1:])
    g = G(ctx)
    fs = g.fs
    uid, gid = ctx.uid, ctx.proc.gid
    sub = a[0] if a else ""
    rest = a[1:]
    if sub in ("--version", "version"):
        g.out("3.63.0\n")
        return 0
    if sub == "init":
        repo = g.maybe_repo()
        if repo is None and "--no-scm" not in rest:
            g.err("ERROR: failed to initiate DVC - /home is not tracked by any supported SCM tool (e.g. Git). Use "
                  "`--no-scm` if you don't want to use any SCM or `--subdir` if initializing inside a subdirectory of a "
                  "parent SCM repository.\n")
            return 1
        root = repo.worktree if repo else g.cwd
        if fs.exists(f"{root}/.dvc") and "-f" not in rest and "--force" not in rest:
            g.err(f"ERROR: failed to initiate DVC - '.dvc' exists. Use `-f` to force.\n")
            return 1
        fs.mkdir(f"{root}/.dvc/tmp", "/", 0o755, uid, gid, parents=True)
        fs.write_file(f"{root}/.dvc/config", "", "/", 0o644, uid, gid)
        fs.write_file(f"{root}/.dvc/.gitignore", "/config.local\n/tmp\n/cache\n", "/", 0o644, uid, gid)
        fs.write_file(f"{root}/.dvcignore", "# Add patterns of files dvc should ignore, which could improve\n# the "
                      "performance. Learn more at\n# https://dvc.org/doc/user-guide/dvcignore\n", "/", 0o644, uid, gid)
        if repo is not None:
            yield from ctx.flush()
            yield from ctx.run(["git", "add", ".dvc/config", ".dvc/.gitignore", ".dvcignore"], cwd=root)
        g.out("Initialized DVC repository.\n\nYou can now commit the changes to git.\n\n"
              "+---------------------------------------------------------------------+\n"
              "|                                                                     |\n"
              "|        DVC has enabled anonymous aggregate usage analytics.         |\n"
              "|     Read the analytics documentation (and how to opt-out) here:     |\n"
              "|             <https://dvc.org/doc/user-guide/analytics>              |\n"
              "|                                                                     |\n"
              "+---------------------------------------------------------------------+\n\n"
              "What's next?\n------------\n- Check out the documentation: <https://dvc.org/doc>\n"
              "- Get help and share ideas: <https://dvc.org/chat>\n- Star us on GitHub: <https://github.com/iterative/dvc>\n")
        return 0
    root = _dvc_root(g)
    if root is None:
        g.err("ERROR: you are not inside of a DVC repository (checked up to mount point '/')\n")
        return 253
    cache = f"{root}/.dvc/cache/files/md5"

    def cache_put(md5, data):
        node = fs.write_file(f"{cache}/{md5[:2]}/{md5[2:]}", b"", "/", 0o444, uid, gid)
        node.data = data.clone() if isinstance(data, LazyData) else LazyData(src=bytes(data))

    def cache_get(md5):
        try:
            return fs.lookup(f"{cache}/{md5[:2]}/{md5[2:]}").data
        except FSError:
            return None

    def dvc_files():
        out = []
        stack = [root]
        while stack:
            d = stack.pop()
            node = fs.lookup(d)
            for name, ch in node.entries.items():
                if name in (".git", ".dvc"):
                    continue
                if ch.type == "d":
                    stack.append(f"{d}/{name}")
                elif name.endswith(".dvc"):
                    out.append(f"{d}/{name}")
        return sorted(out)

    def read_dvc(path):
        text = bytes(fs.lookup(path).data).decode()
        md5 = re.search(r"md5:\s*([0-9a-f]{32})", text)
        out = re.search(r"path:\s*(\S+)", text)
        return (md5.group(1) if md5 else None), posixpath.join(posixpath.dirname(path), out.group(1) if out else "")

    if sub == "add":
        targets = [x for x in rest if not x.startswith("-")]
        if not targets:
            g.err("ERROR: the following arguments are required: targets\n")
            return 254
        repo = g.maybe_repo()
        adds = []
        for t in targets:
            p = ctx.abspath(t)
            try:
                node = fs.lookup(p)
            except FSError:
                g.err(f"ERROR: output '{t}' does not exist\n")
                return 1
            if repo is not None:
                rel = repo.rel(p)
                if repo.index.get(rel) is not None:
                    g.err(f"ERROR: output '{t}' is already tracked by SCM (e.g. Git).\n    You can remove it from Git, "
                          f"then add to DVC.\n        To stop tracking from Git:\n            git rm -r --cached "
                          f"'{t}'\n            git commit -m \"stop tracking {t}\"\n")
                    return 1
            md5 = _md5(node.data)
            cache_put(md5, node.data)
            d, name = posixpath.split(p)
            fs.write_file(f"{p}.dvc", f"outs:\n- md5: {md5}\n  size: {len(node.data)}\n  hash: md5\n  path: {name}\n",
                          "/", 0o644, uid, gid)
            gi = f"{d}/.gitignore"
            text = bytes(fs.lookup(gi).data).decode() if fs.exists(gi) else ""
            if f"/{name}\n" not in text:
                if text and not text.endswith("\n"):
                    text += "\n"
                fs.write_file(gi, text + f"/{name}\n", "/", 0o644, uid, gid)
            rel_dir = posixpath.relpath(d, g.cwd)
            adds += [posixpath.normpath(posixpath.join(rel_dir, name + ".dvc")),
                     posixpath.normpath(posixpath.join(rel_dir, ".gitignore"))]
        n = len(targets)
        g.out(f"100% Adding...|{'█' * 40}|{n}/{n} [00:00, 18.2file/s]\n\nTo track the changes with git, run:\n\n"
              f"\tgit add {' '.join(adds)}\n\nTo enable auto staging, run:\n\n\tdvc config core.autostage true\n")
        return 0
    if sub == "remote":
        cfgp = f"{root}/.dvc/config"
        text = bytes(fs.lookup(cfgp).data).decode() if fs.exists(cfgp) else ""
        op = rest[0] if rest else "list"
        args = [x for x in rest[1:] if not x.startswith("-")]
        if op == "add":
            if len(args) < 2:
                g.err("ERROR: the following arguments are required: name, url\n")
                return 254
            name, url = args[0], args[1]
            if f"['remote \"{name}\"']" in text and "-f" not in rest:
                g.err(f"ERROR: configuration error - config file error: remote '{name}' already exists. Use `-f|--force`"
                      " to overwrite it.\n")
                return 251
            if "-d" in rest or "--default" in rest:
                text = re.sub(r"\[core\]\n(\s+remote = \S+\n)?", "", text)
                text = f"[core]\n    remote = {name}\n" + text
            text += f"['remote \"{name}\"']\n    url = {url}\n"
            fs.write_file(cfgp, text, "/", 0o644, uid, gid)
            return 0
        if op == "list":
            for m in re.finditer(r"\['remote \"([^\"]+)\"'\]\n\s+url = (\S+)", text):
                g.out(f"{m.group(1)}\t{m.group(2)}\n")
            return 0
        if op == "default":
            if args:
                text = re.sub(r"\[core\]\n(\s+remote = \S+\n)?", "", text)
                fs.write_file(cfgp, f"[core]\n    remote = {args[0]}\n" + text, "/", 0o644, uid, gid)
            else:
                m = re.search(r"remote = (\S+)", text)
                g.out((m.group(1) if m else "") + "\n")
            return 0
        if op == "remove" and args:
            text = re.sub(rf"\['remote \"{re.escape(args[0])}\"'\]\n(\s+\w+ = \S+\n)*", "", text)
            fs.write_file(cfgp, text, "/", 0o644, uid, gid)
            return 0
        return 0
    if sub in ("push", "pull", "fetch", "status", "checkout"):
        cfgp = f"{root}/.dvc/config"
        text = bytes(fs.lookup(cfgp).data).decode() if fs.exists(cfgp) else ""
        rname = rest[rest.index("-r") + 1] if "-r" in rest else (re.search(r"remote = (\S+)", text) or [None, None])[1]
        url = None
        if rname:
            m = re.search(rf"\['remote \"{re.escape(rname)}\"'\]\n\s+url = (\S+)", text)
            url = m.group(1) if m else None
        outs = [read_dvc(f) for f in dvc_files()]
        if sub == "status":
            changed = []
            for md5, path in outs:
                try:
                    node = fs.lookup(path)
                except FSError:
                    changed.append((path, "deleted"))
                    continue
                if _md5(node.data) != md5:
                    changed.append((path, "modified"))
            if not changed:
                g.out("Data and pipelines are up to date.\n")
                return 0
            for path, why in changed:
                rel = posixpath.relpath(path, g.cwd)
                g.out(f"{rel}.dvc:\n\tchanged outs:\n\t\t{why}:            {rel}\n")
            return 0
        if sub in ("push", "pull", "fetch") and url is None:
            g.err("ERROR: failed to " + sub + " data " + ("to" if sub == "push" else "from") + " the cloud - config "
                  "file error: no remote specified in " + root + "/.dvc/config. Setup default remote with\n    dvc "
                  "remote default <remote name>\nor use:\n    dvc " + sub + " -r <remote name>\n")
            return 1
        remote = _dvc_store(g.world).setdefault(url, {}) if url else {}
        if sub == "push":
            n = 0
            for md5, path in outs:
                data = cache_get(md5) if md5 else None
                if data is not None and md5 not in remote:
                    remote[md5] = data
                    n += 1
            g.out(f"{n} file{'s' if n != 1 else ''} pushed\n" if n else "Everything is up to date.\n")
            return 0
        got = added = 0
        for md5, path in outs:
            if md5 is None:
                continue
            if cache_get(md5) is None and sub in ("pull", "fetch"):
                if md5 in remote:
                    cache_put(md5, remote[md5])
                    got += 1
                else:
                    g.err(f"WARNING: Some of the cache files do not exist neither locally nor on remote. Missing cache "
                          f"files:\nmd5: {md5}\n")
                    continue
            if sub == "fetch":
                continue
            data = cache_get(md5)
            if data is None:
                continue
            try:
                cur = fs.lookup(path)
                if _md5(cur.data) == md5:
                    continue
            except FSError:
                pass
            node = fs.write_file(path, b"", "/", 0o644, uid, gid)
            node.data = data.clone() if isinstance(data, LazyData) else LazyData(src=bytes(data))
            added += 1
        if sub == "fetch":
            g.out(f"{got} file{'s' if got != 1 else ''} fetched\n" if got else "Everything is up to date.\n")
        elif added:
            g.out(f"A       {posixpath.relpath(outs[0][1], g.cwd)}\n" if added == 1 else "")
            g.out(f"{added} file{'s' if added != 1 else ''} added" + (" and fetched" if got else "") + "\n")
        else:
            g.out("Everything is up to date.\n")
        return 0
    if sub == "config":
        if len(rest) >= 2 and rest[0] == "core.autostage":
            return 0
        return 0
    g.err(f"ERROR: unrecognized arguments: {sub}\n" if sub else "usage: dvc [-q | -v] [-h] [-V] [--cd <path>] "
          "COMMAND ...\n")
    return 254

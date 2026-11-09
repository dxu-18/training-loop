"""rev-parse, cat-file, ls-files, ls-tree, hash-object, check-ignore, show-ref, symbolic-ref, update-ref,
write-tree, commit-tree, merge-base, for-each-ref, count-objects, fsck, gc, archive, apply, var."""

from __future__ import annotations

import io
import posixpath
import re

from ..vfs import FSError
from .cli import Usage, command, parse, pathspec_matcher
from .diff import split_lines
from .objects import GitError, encode_commit, hash_object
from .repo import Entry
from .revparse import dwim_ref, resolve, try_resolve, unknown_revision
from .status import collect


@command("rev-parse")
def git_rev_parse(g, args):
    repo = g.maybe_repo()
    out = []
    short = None
    abbrev_ref = False
    verify = quiet = False
    sym_full = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--show-toplevel":
            if repo is None:
                raise GitError("fatal: not a git repository (or any of the parent directories): .git")
            if repo.bare:
                raise GitError("fatal: this operation must be run in a work tree")
            out.append(repo.worktree)
        elif a in ("--git-dir", "--absolute-git-dir"):
            if repo is None:
                raise GitError("fatal: not a git repository (or any of the parent directories): .git")
            gd = repo.gitdir
            if a == "--git-dir" and repo.worktree and gd == repo.worktree + "/.git" and g.cwd == repo.worktree:
                gd = ".git"
            out.append(gd)
        elif a == "--git-common-dir":
            out.append(repo.common if repo.common != repo.gitdir else (".git" if repo.worktree == g.cwd else repo.common))
        elif a == "--is-inside-work-tree":
            if repo is None:
                raise GitError("fatal: not a git repository (or any of the parent directories): .git")
            out.append("true" if repo.worktree and not g.cwd.startswith(repo.gitdir) else "false")
        elif a == "--is-inside-git-dir":
            out.append("true" if repo and g.cwd.startswith(repo.gitdir) else "false")
        elif a == "--is-bare-repository":
            out.append("true" if repo and repo.bare else "false")
        elif a == "--is-shallow-repository":
            out.append("true" if repo and repo.shallow() else "false")
        elif a == "--show-prefix":
            p = g.prefix(repo)
            out.append(p + "/" if p else "")
        elif a == "--show-cdup":
            p = g.prefix(repo)
            out.append("../" * (p.count("/") + 1) if p else "")
        elif a.startswith("--short"):
            short = int(a.split("=", 1)[1]) if "=" in a else 7
        elif a == "--abbrev-ref" or a.startswith("--abbrev-ref="):
            abbrev_ref = True
        elif a == "--verify":
            verify = True
        elif a in ("-q", "--quiet"):
            quiet = True
        elif a == "--symbolic-full-name":
            sym_full = True
        elif a == "--":
            pass
        elif a.startswith("--"):
            pass
        else:
            if repo is None:
                raise GitError("fatal: not a git repository (or any of the parent directories): .git")
            spec = a
            if abbrev_ref or sym_full:
                if spec in ("HEAD", "@"):
                    hr = repo.head_ref()
                    out.append(("HEAD" if not hr else hr[len("refs/heads/"):] if abbrev_ref else hr))
                elif "@{" in spec and spec.endswith(("@{u}", "@{upstream}", "@{push}")):
                    b = spec.split("@{")[0] or repo.head_branch()
                    if not b:
                        raise GitError("fatal: HEAD does not point to a branch")
                    up = repo.upstream_ref(b)
                    if not up:
                        raise GitError(f"fatal: no upstream configured for branch '{b}'")
                    out.append(up[len("refs/remotes/"):] if abbrev_ref and up.startswith("refs/remotes/") else up)
                else:
                    full, sha = dwim_ref(repo, spec)
                    if not full:
                        raise unknown_revision(spec)
                    out.append(full.split("/", 2)[-1] if abbrev_ref else full)
            else:
                sha = try_resolve(repo, spec)
                if sha is None:
                    if verify:
                        if quiet:
                            return 1
                        raise GitError("fatal: Needed a single revision")
                    raise unknown_revision(spec)
                out.append(repo.odb.abbrev(sha, short) if short else sha)
        i += 1
    if out:
        g.out("\n".join(out) + "\n")
    return 0


@command("cat-file")
def git_cat_file(g, args):
    o = parse(args, "t s p e batch=? batch-check=? textconv filters")
    repo = g.repo()
    db = repo.odb
    a = o.args
    if not a:
        raise Usage("usage: git cat-file <type> <object>\n   or: git cat-file (-e | -p) <object>\n   or: git cat-file "
                    "(-t | -s) [--allow-unknown-type] <object>")
    if len(a) == 2 and not (o.get("t") or o.get("s") or o.get("p") or o.get("e")):
        kind, spec = a
        sha = try_resolve(repo, spec)
        if sha is None:
            raise GitError(f"fatal: Not a valid object name {spec}")
        g.out(bytes(db.raw(sha)).decode("utf-8", "surrogateescape"))
        return 0
    spec = a[0]
    sha = try_resolve(repo, spec)
    if sha is None:
        if o.get("e"):
            return 1
        raise GitError(f"fatal: Not a valid object name {spec}")
    if o.get("e"):
        return 0
    if o.get("t"):
        g.out(db.kind(sha) + "\n")
        return 0
    if o.get("s"):
        g.out(f"{db.size(sha)}\n")
        return 0
    kind = db.kind(sha)
    if kind == "tree":
        rows = []
        for mode, name, s in db.tree(sha):
            k = "tree" if mode == "40000" else "commit" if mode == "160000" else "blob"
            rows.append(f"{mode.rjust(6, '0')} {k} {s}\t{name}\n")
        g.out("".join(rows))
        return 0
    g.out(bytes(db.raw(sha)).decode("utf-8", "surrogateescape"))
    return 0


@command("ls-files")
def git_ls_files(g, args):
    o = parse(args, "cached|c deleted|d modified|m others|o ignored|i stage|s unmerged|u killed|k exclude-standard "
                    "directory error-unmatch z full-name abbrev=? debug eol exclude|x= exclude-from|X= "
                    "recurse-submodules format=")
    repo = g.repo()
    idx = repo.index
    specs = o.args + (o.paths or [])
    match = pathspec_matcher(g, repo, specs)
    out = []
    shown = (lambda p: p) if o.get("full-name") else (lambda p: g.shown(repo, p))
    any_mode = o.get("deleted") or o.get("modified") or o.get("others") or o.get("ignored") or o.get("unmerged")
    if o.get("stage") or o.get("unmerged"):
        for (p, s), e in sorted(idx.entries.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if match and not match(p):
                continue
            if o.get("unmerged") and s == 0:
                continue
            out.append(f"{e.mode:06o} {e.sha} {s}\t{shown(p)}")
    elif o.get("cached") or not any_mode:
        for p in idx.paths():
            if match and not match(p):
                continue
            out.append(shown(p))
    if o.get("deleted") or o.get("modified"):
        st = collect(repo, untracked="no")
        for p in sorted(st.unstaged):
            if match and not match(p):
                continue
            if o.get("deleted") and st.unstaged[p] == "D":
                out.append(shown(p))
            elif o.get("modified"):
                out.append(shown(p))
    if o.get("others"):
        st = collect(repo, untracked="all" if not o.get("directory") else "normal",
                     ignored=bool(o.get("ignored")))
        files = st.ignored if o.get("ignored") else st.untracked
        if not o.get("exclude-standard") and not o.get("ignored"):
            files = sorted(set(st.untracked) | set(collect(repo, untracked="all", ignored=True).ignored))
        for p in files:
            if match and not match(p.rstrip("/")):
                continue
            out.append(shown(p))
    if o.get("error-unmatch"):
        for s in specs:
            m1 = pathspec_matcher(g, repo, [s])
            if not any(m1(p) for p in idx.paths()):
                g.err(f"error: pathspec '{s}' did not match any file(s) known to git\nDid you forget to 'git add'?\n")
                return 1
    sep = "\0" if o.get("z") else "\n"
    if out:
        g.out(sep.join(out) + sep)
    return 0


@command("ls-tree")
def git_ls_tree(g, args):
    o = parse(args, "r t d name-only|name-status long|l full-name full-tree abbrev=? z object-only format=")
    repo = g.repo()
    db = repo.odb
    if not o.args:
        raise Usage("usage: git ls-tree [<options>] <tree-ish> [<path>...]")
    tree = try_resolve(repo, o.args[0], "tree")
    if tree is None:
        raise GitError("fatal: Not a valid object name " + o.args[0])
    paths = o.args[1:] + (o.paths or [])
    out = []

    def emit(mode, name, sha, path):
        k = "tree" if mode == "40000" else "commit" if mode == "160000" else "blob"
        if o.get("d") and k != "tree":
            return
        if o.get("name-only"):
            out.append(path)
        elif o.get("object-only"):
            out.append(sha)
        elif o.get("long"):
            size = "-" if k != "blob" else str(db.size(sha))
            out.append(f"{mode.rjust(6, '0')} {k} {sha} {size:>7}\t{path}")
        else:
            out.append(f"{mode.rjust(6, '0')} {k} {sha}\t{path}")

    def walk(sha, prefix):
        for mode, name, s in db.tree(sha):
            p = prefix + name
            if paths and not any(p == x.rstrip("/") or p.startswith(x.rstrip("/") + "/") or
                                 x.rstrip("/").startswith(p + "/") for x in paths):
                continue
            if mode == "40000" and (o.get("r") or any(x.rstrip("/").startswith(p + "/") for x in paths)):
                if o.get("t"):
                    emit(mode, name, s, p)
                walk(s, p + "/")
            else:
                emit(mode, name, s, p)
    walk(tree, "")
    if out:
        g.out("\n".join(out) + "\n")
    return 0


@command("hash-object")
def git_hash_object(g, args):
    o = parse(args, "w t= stdin stdin-paths literally no-filters path=")
    repo = g.maybe_repo()
    kind = o.get("t") or "blob"
    datas = []
    if o.get("stdin"):
        data = yield from g.ctx.read_fd(0)
        datas.append(data)
    for f in o.args:
        try:
            node = g.ctx.lookup(f)
        except FSError as e:
            raise GitError(f"fatal: could not open '{f}' for reading: {e.msg}")
        datas.append(node.data)
    for d in datas:
        if o.get("w"):
            if repo is None:
                raise GitError("fatal: not a git repository (or any of the parent directories): .git")
            sha = repo.store(kind, d)
        else:
            sha = hash_object(kind, d)
        g.out(sha + "\n")
    return 0


@command("check-ignore")
def git_check_ignore(g, args):
    o = parse(args, "verbose|v non-matching|n no-index quiet|q stdin z")
    repo = g.repo(wt=True)
    paths = o.args + (o.paths or [])
    if o.get("stdin"):
        data = yield from g.ctx.read_fd(0)
        paths += [x for x in data.decode().split("\n") if x]
    if not paths:
        raise GitError("fatal: no path specified")
    hit = False
    out = []
    for p in paths:
        rel = g.to_rel(repo, p)
        node = repo.wt_node(rel)
        is_dir = node is not None and node.type == "d" or p.endswith("/")
        if not o.get("no-index") and repo.index.get(rel) is not None:
            if o.get("non-matching") and o.get("verbose"):
                out.append(f"::\t{p}")
            continue
        r = repo.ignored_by(rel, is_dir)
        if r is not None and not r.negate:
            hit = True
            if o.get("verbose"):
                out.append(f"{r.source}:{r.line}:{r.pattern}\t{p}")
            else:
                out.append(p)
        elif r is not None and o.get("verbose"):
            # a negated pattern decided it: shown with -v (but it isn't ignored)
            out.append(f"{r.source}:{r.line}:{r.pattern}\t{p}")
        elif o.get("non-matching") and o.get("verbose"):
            out.append(f"::\t{p}")
    if out and not o.get("quiet"):
        g.out("\n".join(out) + "\n")
    return 0 if hit else 1


@command("show-ref")
def git_show_ref(g, args):
    o = parse(args, "heads tags branches head verify quiet|q hash|s=? abbrev=? dereference|d exists")
    repo = g.repo()
    refs = repo.ref_names("refs/")
    out = []
    if o.get("verify") or o.get("exists"):
        for a in o.args:
            sha = repo.resolve_ref(a)
            if not sha:
                if not o.get("quiet"):
                    g.err(f"fatal: '{a}' - not a valid ref\n")
                return 1 if o.get("verify") else 2
            if not o.get("quiet") and not o.get("exists"):
                out.append(f"{sha} {a}")
    else:
        if o.get("head") and repo.head():
            out.append(f"{repo.head()} HEAD")
        for name, sha in refs.items():
            if o.get("heads") or o.get("branches") or o.get("tags"):
                if not ((o.get("heads") or o.get("branches")) and name.startswith("refs/heads/") or
                        o.get("tags") and name.startswith("refs/tags/")):
                    continue
            if o.args and not any(name == a or name.endswith("/" + a) for a in o.args):
                continue
            out.append(f"{sha} {name}" if o.get("hash") is None else sha)
            if o.get("dereference") and name.startswith("refs/tags/") and repo.odb.kind(sha) == "tag":
                out.append(f"{repo.peel(sha)} {name}^{{}}")
    if out:
        g.out("\n".join(out) + "\n")
        return 0
    return 1


@command("symbolic-ref")
def git_symbolic_ref(g, args):
    o = parse(args, "short delete|d quiet|q m= no-recurse")
    repo = g.repo()
    a = o.args
    if not a:
        raise Usage("usage: git symbolic-ref [-m <reason>] <name> <ref>\n   or: git symbolic-ref [-q] [--short] "
                    "[--no-recurse] <name>\n   or: git symbolic-ref --delete [-q] <name>")
    if o.get("delete"):
        repo.remove(a[0])
        return 0
    if len(a) == 2:
        if not a[1].startswith("refs/"):
            raise GitError(f"fatal: Refusing to point {a[0]} outside of refs/")
        repo.write(a[0], f"ref: {a[1]}\n", a[0] != "HEAD")
        return 0
    r = repo.read_ref_raw(a[0])
    if not r or r[0] != "sym":
        if o.get("quiet"):
            return 1
        raise GitError(f"fatal: ref {a[0]} is not a symbolic ref")
    t = r[1]
    if o.get("short"):
        t = t.split("/", 2)[-1] if t.startswith(("refs/heads/", "refs/tags/", "refs/remotes/")) else t
    g.out(t + "\n")
    return 0


@command("update-ref")
def git_update_ref(g, args):
    o = parse(args, "d m= no-deref stdin create-reflog")
    repo = g.repo()
    a = o.args
    if o.get("d"):
        repo.delete_ref(a[0])
        return 0
    if len(a) < 2:
        raise Usage("usage: git update-ref [<options>] -d <refname> [<old-val>]\n   or: git update-ref [<options>]    "
                    "<refname> <new-val> [<old-val>]")
    sha = resolve(repo, a[1])
    if len(a) > 2:
        cur = repo.resolve_ref(a[0])
        if cur != resolve(repo, a[2]):
            raise GitError(f"fatal: cannot lock ref '{a[0]}': is at {cur} but expected {a[2]}")
    repo.update_ref(a[0], sha, o.get("m"))
    return 0


@command("write-tree")
def git_write_tree(g, args):
    repo = g.repo()
    if repo.index.conflicts():
        raise GitError("error: " + list(repo.index.conflicts())[0] + ": unmerged (" + "...)\nfatal: git-write-tree: "
                       "error building trees")
    g.out(repo.odb.write_tree(repo.index.flat()) + "\n")
    return 0


@command("commit-tree")
def git_commit_tree(g, args):
    o = parse(args, "p= m= F=")
    repo = g.repo()
    if not o.args:
        raise Usage("usage: git commit-tree [(-p <parent>)...] [-S[<keyid>]] [(-m <message>)...] [(-F <file>)...] <tree>")
    tree = resolve(repo, o.args[0], "tree")
    parents = [resolve(repo, p, "commit") for p in o.all("p")]
    msg = "\n\n".join(o.all("m")) + "\n" if o.all("m") else ""
    sha = repo.store("commit", encode_commit(tree, parents, repo.ident("author"), repo.ident("committer"), msg))
    g.out(sha + "\n")
    return 0


@command("merge-base")
def git_merge_base(g, args):
    o = parse(args, "all|a is-ancestor octopus independent fork-point")
    repo = g.repo()
    db = repo.odb
    shas = [resolve(repo, x, "commit") for x in o.args]
    if o.get("is-ancestor"):
        return 0 if db.is_ancestor(shas[0], shas[1]) else 1
    if len(shas) < 2:
        raise Usage("usage: git merge-base [-a | --all] <commit> <commit>...")
    bases = db.merge_bases(shas[0], shas[1])
    if not bases:
        return 1
    g.out("".join(b + "\n" for b in (bases if o.get("all") else bases[:1])))
    return 0


@command("for-each-ref")
def git_for_each_ref(g, args):
    o = parse(args, "format= sort= count= points-at= merged=? no-merged=? contains= shell python perl tcl")
    repo = g.repo()
    db = repo.odb
    refs = repo.ref_names("refs/")
    names = [n for n in refs if not o.args or any(n.startswith(p.rstrip("*")) for p in o.args)]
    sort = o.get("sort") or "refname"
    rev = sort.startswith("-")
    key = sort.lstrip("-")

    def kv(n):
        sha = repo.peel(refs[n])
        if key in ("committerdate", "creatordate"):
            return db.commit(sha).committer.when if db.kind(sha) == "commit" else 0
        return n
    names.sort(key=kv, reverse=rev)
    if o.get("count"):
        names = names[:int(o.get("count"))]
    fmt = o.get("format") or "%(objectname) %(objecttype)\t%(refname)"
    out = []
    from .cmd_history import fmt_date, relative
    now = repo.now()
    for n in names:
        sha = refs[n]
        target = repo.peel(sha)
        c = db.commit(target) if db.kind(target) == "commit" else None

        def field(m):
            f = m.group(1)
            if f == "refname":
                return n
            if f in ("refname:short", "refname:lstrip=2"):
                return n.split("/", 2)[-1]
            if f == "objectname":
                return sha
            if f == "objectname:short":
                return db.abbrev(sha)
            if f == "objecttype":
                return db.kind(sha)
            if f == "subject" or f == "contents:subject":
                return c.subject if c else ""
            if f == "authorname":
                return c.author.name if c else ""
            if f == "committerdate:relative" or f == "creatordate:relative":
                return relative(c.committer.when, now) if c else ""
            if f.startswith("committerdate") or f.startswith("creatordate"):
                mode = f.split(":", 1)[1] if ":" in f else None
                return fmt_date(c.committer, mode, now) if c else ""
            if f == "upstream:short":
                b = n[len("refs/heads/"):] if n.startswith("refs/heads/") else None
                up = repo.upstream_ref(b) if b else None
                return up.split("/", 2)[-1] if up else ""
            if f == "HEAD":
                return "*" if repo.head_ref() == n else " "
            return ""
        out.append(re.sub(r"%\(([^)]+)\)", field, fmt))
    if out:
        g.out("\n".join(out) + "\n")
    return 0


def _human(n):
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "bytes" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


@command("count-objects")
def git_count_objects(g, args):
    o = parse(args, "verbose|v human-readable|H")
    repo = g.repo()
    loose, loose_size = _loose_objects(repo)
    packs = _pack_files(repo)
    in_pack = sum(n for _, n, _ in packs)
    size_pack = sum(s for _, _, s in packs)
    h = _human if o.get("human-readable") else (lambda n: str((n + 1023) // 1024))
    if o.get("verbose"):
        g.out(f"count: {len(loose)}\nsize: {h(loose_size)}\nin-pack: {in_pack}\npacks: {len(packs)}\n"
              f"size-pack: {h(size_pack)}\nprune-packable: 0\ngarbage: 0\nsize-garbage: {h(0)}\n")
    else:
        g.out(f"{len(loose)} objects, {h(loose_size) if o.get('human-readable') else (loose_size + 1023) // 1024} "
              f"{'' if o.get('human-readable') else 'kilobytes'}".rstrip() + "\n")
    return 0


def _loose_objects(repo):
    out = []
    size = 0
    base = repo.path("objects", True)
    try:
        root = repo.fs.lookup(base)
    except FSError:
        return out, 0
    for d in root.entries:
        if len(d) == 2 and re.fullmatch(r"[0-9a-f]{2}", d):
            node = root.entries[d]
            for f, fn in node.entries.items():
                out.append(d + f)
                size += -(-len(fn.data) // 4096) * 4096
    return out, size


def _pack_files(repo):
    """[(name, objects, bytes)] for .git/objects/pack/*.pack."""
    out = []
    base = repo.path("objects/pack", True)
    try:
        root = repo.fs.lookup(base)
    except FSError:
        return out
    for f, node in root.entries.items():
        if f.endswith(".pack"):
            meta = (node.xattr or {}).get("objects", 0) if node.xattr else 0
            out.append((f, meta, len(node.data)))
    return out


def write_pack(repo, shas, name_seed=""):
    """Pack these objects: one .pack (its size estimated, content generated) and .idx in objects/pack."""
    import hashlib
    from ..vfs import LazyData
    db = repo.odb
    total = 0
    for s in shas:
        try:
            n = db.size(s)
        except GitError:
            continue
        total += max(20, int(n * 0.45)) if db.kind(s) == "blob" else max(30, int(n * 0.6))
    if not shas:
        return None
    pack_id = hashlib.sha1(("".join(sorted(shas)) + name_seed).encode()).hexdigest()
    base = f"objects/pack/pack-{pack_id}"
    p = repo.path(base + ".pack", True)
    node = repo.fs.write_file(p, b"", "/", 0o444, repo.ctx.uid, repo.ctx.proc.gid)
    node.data = LazyData(size=total + 32, seed=f"pack:{pack_id}", head=b"PACK\x00\x00\x00\x02" +
                         len(shas).to_bytes(4, "big"))
    node.mode = 0o100444
    node.xattr = {"objects": len(shas)}
    idx = repo.fs.write_file(repo.path(base + ".idx", True), b"", "/", 0o444, repo.ctx.uid, repo.ctx.proc.gid)
    idx.data = LazyData(size=1072 + 28 * len(shas), seed=f"idx:{pack_id}", head=b"\xfftOc\x00\x00\x00\x02")
    idx.mode = 0o100444
    return pack_id


@command("fsck")
def git_fsck(g, args):
    o = parse(args, "full unreachable dangling no-dangling lost-found root tags cache no-reflogs connectivity-only "
                    "progress no-progress verbose|v name-objects")
    repo = g.repo()
    db = repo.odb
    heads = [repo.peel(v) for v in repo.ref_names("refs/").values()]
    if repo.head():
        heads.append(repo.head())
    if not o.get("no-reflogs"):
        for name in ["HEAD"] + list(repo.ref_names("refs/")):
            for old, new, _, _ in repo.reflog(name):
                heads += [x for x in (old, new) if x != "0" * 40 and db.has(x)]
    reach = db.objects_of([h for h in heads if db.has(h) and db.kind(h) == "commit"])
    reach |= set(heads)
    loose, _ = _loose_objects(repo)
    out = []
    unreach = [s for s in loose if s not in reach]
    # only the tips: unreachable objects that no other unreachable object points to
    pointed = set()
    for s in unreach:
        if db.has(s) and db.kind(s) == "commit":
            c = db.commit(s)
            pointed.update(c.parents)
            pointed.add(c.tree)
        elif db.has(s) and db.kind(s) == "tree":
            pointed.update(x for _, _, x in db.tree(s))
    for s in sorted(unreach):
        if not db.has(s):
            continue
        k = db.kind(s)
        if o.get("unreachable"):
            out.append(f"unreachable {k} {s}")
        elif s not in pointed and not o.get("no-dangling"):
            out.append(f"dangling {k} {s}")
    if out:
        g.out("\n".join(out) + "\n")
    return 0


@command("gc")
def git_gc(g, args):
    o = parse(args, "aggressive auto prune=? no-prune quiet|q force keep-largest-pack")
    repo = g.repo()
    db = repo.odb
    if o.get("auto"):
        return 0
    heads = [repo.peel(v) for v in repo.ref_names("refs/").values()]
    if repo.head():
        heads.append(repo.head())
    for name in ["HEAD"] + list(repo.ref_names("refs/")):
        for old, new, _, _ in repo.reflog(name):
            heads += [x for x in (old, new) if x != "0" * 40 and db.has(x)]
    objs = db.objects_of([h for h in heads if db.has(h) and db.kind(h) == "commit"])
    objs |= {h for h in heads if db.has(h)}
    n = len(objs)
    # repack: one pack with everything, loose objects removed
    for f, _, _ in _pack_files(repo):
        repo.remove(f"objects/pack/{f}", True)
        repo.remove(f"objects/pack/{f[:-5]}.idx", True)
    loose, _ = _loose_objects(repo)
    for s in loose:
        repo.remove(f"objects/{s[:2]}/{s[2:]}", True)
    from .repo import prune_empty_dirs
    base = repo.path("objects", True)
    try:
        root = repo.fs.lookup(base)
        for d in list(root.entries):
            if len(d) == 2:
                prune_empty_dirs(repo.fs, f"{base}/{d}", base)
    except FSError:
        pass
    write_pack(repo, sorted(objs))
    refs = repo.ref_names("refs/")
    repo.write_packed_refs(refs)
    for name in refs:
        repo.remove(name, True)
    if not o.get("quiet"):
        g.err(f"Enumerating objects: {n}, done.\nCounting objects: 100% ({n}/{n}), done.\n"
              f"Delta compression using up to 8 threads\nCompressing objects: 100% ({n // 2}/{n // 2}), done.\n"
              f"Writing objects: 100% ({n}/{n}), done.\nTotal {n} (delta 0), reused 0 (delta 0), pack-reused 0\n")
    return 0


@command("prune")
def git_prune(g, args):
    return 0


@command("archive")
def git_archive(g, args):
    o = parse(args, "format= output|o= prefix= list|l remote= verbose|v worktree-attributes add-file=")
    repo = g.repo()
    db = repo.odb
    if o.get("list"):
        g.out("tar\ntgz\ntar.gz\nzip\n")
        return 0
    if not o.args:
        raise Usage("usage: git archive [<options>] <tree-ish> [<path>...]")
    tree = resolve(repo, o.args[0], "tree")
    commit = try_resolve(repo, o.args[0], "commit")
    flat = db.flat(tree)
    paths = o.args[1:]
    if paths:
        flat = {p: v for p, v in flat.items() if any(p == x or p.startswith(x.rstrip("/") + "/") for x in paths)}
    out_name = o.get("output")
    fmt = o.get("format") or (("zip" if out_name.endswith(".zip") else "tar.gz" if out_name.endswith((".tar.gz",
                              ".tgz")) else "tar") if out_name else "tar")
    prefix = o.get("prefix") or ""
    when = db.commit(commit).committer.when if commit else repo.now()
    buf = io.BytesIO()
    if fmt == "zip":
        import zipfile
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            import time as _t
            dt = _t.gmtime(when)[:6]
            for p in sorted(flat):
                mode, sha = flat[p]
                zi = zipfile.ZipInfo(prefix + p, date_time=dt)
                zi.external_attr = (0o100755 if mode == "100755" else 0o100644) << 16
                z.writestr(zi, bytes(db.raw(sha)))
    else:
        import tarfile
        tb = io.BytesIO()
        with tarfile.open(fileobj=tb, mode="w", format=tarfile.PAX_FORMAT) as t:
            for p in sorted(flat):
                mode, sha = flat[p]
                data = bytes(db.raw(sha))
                ti = tarfile.TarInfo(prefix + p)
                ti.size = len(data)
                ti.mtime = when
                ti.mode = 0o775 if mode == "100755" else 0o664
                ti.uname = ti.gname = "root"
                t.addfile(ti, io.BytesIO(data))
        raw = tb.getvalue()
        if fmt in ("tgz", "tar.gz"):
            import gzip
            raw = gzip.compress(raw, mtime=0)
        buf.write(raw)
    data = buf.getvalue()
    if out_name:
        g.fs.write_file(out_name if out_name.startswith("/") else posixpath.join(g.cwd, out_name), data, "/", 0o644,
                        g.ctx.uid, g.ctx.proc.gid)
    else:
        g.ctx.w(data)
    return 0


@command("apply")
def git_apply(g, args):
    o = parse(args, "check stat numstat summary index cached reverse|R 3way|3 verbose|v p= include= exclude= "
                    "allow-empty reject")
    repo = g.maybe_repo()
    patches = []
    for f in o.args or ["-"]:
        if f == "-":
            data = yield from g.ctx.read_fd(0)
        else:
            try:
                data = bytes(g.ctx.lookup(f).data)
            except FSError as e:
                raise GitError(f"error: can't open patch '{f}': {e.msg}")
        patches.append(data.decode("utf-8", "surrogateescape"))
    files = []
    for text in patches:
        files += _parse_patch(text)
    if not files and not o.get("allow-empty"):
        raise GitError("error: No valid patches in input (allow with \"--allow-empty\")")
    if o.get("stat") or o.get("numstat"):
        from .diff import stat_block
        rows = []
        for fp in files:
            add = sum(1 for h in fp["hunks"] for ln in h["lines"] if ln.startswith("+"))
            rem = sum(1 for h in fp["hunks"] for ln in h["lines"] if ln.startswith("-"))
            rows.append((fp["new"] or fp["old"], add, rem, None))
        if o.get("numstat"):
            g.out("".join(f"{a}\t{r}\t{n}\n" for n, a, r, _ in rows))
        else:
            g.out("\n".join(stat_block(rows, 80)) + "\n")
        return 0
    base = repo.worktree if repo and repo.worktree else g.cwd
    results = []
    for fp in files:
        old, new = fp["old"], fp["new"]
        if o.get("reverse"):
            old, new = new, old
        path = new or old
        target = posixpath.join(base, path) if repo else posixpath.join(g.cwd, path)
        if fp["kind"] == "new" and not o.get("reverse"):
            current = []
        else:
            src = posixpath.join(base, old or new)
            if o.get("cached") and repo:
                e = repo.index.get(old or new)
                if e is None:
                    raise GitError(f"error: {old or new}: does not exist in index")
                current = split_lines(repo.odb.raw(e.sha))
            else:
                try:
                    current = split_lines(bytes(g.fs.lookup(src).data))
                except FSError:
                    raise GitError(f"error: {old or new}: No such file or directory")
        try:
            result = _apply_hunks(current, fp["hunks"], bool(o.get("reverse")))
        except ValueError:
            raise GitError(f"error: patch failed: {path}:{fp['hunks'][0]['old_start'] if fp['hunks'] else 0}\n"
                           f"error: {path}: patch does not apply")
        results.append((fp, target, path, result))
    if o.get("check"):
        return 0
    for fp, target, path, result in results:
        deleting = (fp["kind"] == "delete") != bool(o.get("reverse")) if fp["kind"] in ("delete", "new") else False
        if fp["kind"] == "new" and o.get("reverse"):
            deleting = True
        if not o.get("cached"):
            if deleting:
                try:
                    g.fs.unlink(target)
                except FSError:
                    pass
            else:
                g.fs.write_file(target, "".join(result), "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
        if repo and (o.get("index") or o.get("cached")):
            if deleting:
                repo.index.remove(path)
            else:
                sha = repo.store("blob", "".join(result).encode("utf-8", "surrogateescape"))
                node = repo.wt_node(path)
                repo.index.add(repo.index_entry_for(path, sha, 0o100644, node) if node and not o.get("cached")
                               else Entry(path, sha, 0o100644))
    if repo and (o.get("index") or o.get("cached")):
        repo.write_index()
    return 0


def _parse_patch(text):
    files = []
    cur = None
    hunk = None
    for line in text.split("\n"):
        if line.startswith("diff --git "):
            cur = {"old": None, "new": None, "hunks": [], "kind": "modify"}
            files.append(cur)
            hunk = None
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if m:
                cur["old"], cur["new"] = m.group(1), m.group(2)
            continue
        if line.startswith("--- "):
            if cur is None:
                cur = {"old": None, "new": None, "hunks": [], "kind": "modify"}
                files.append(cur)
            p = line[4:].split("\t")[0]
            if p == "/dev/null":
                cur["kind"] = "new"
                cur["old"] = None
            else:
                cur["old"] = p[2:] if p.startswith("a/") else p
            continue
        if line.startswith("+++ ") and cur is not None:
            p = line[4:].split("\t")[0]
            if p == "/dev/null":
                cur["kind"] = "delete"
                cur["new"] = None
            else:
                cur["new"] = p[2:] if p.startswith("b/") else p
            continue
        if line.startswith("new file mode") and cur is not None:
            cur["kind"] = "new"
            continue
        if line.startswith("deleted file mode") and cur is not None:
            cur["kind"] = "delete"
            continue
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m and cur is not None:
            hunk = {"old_start": int(m.group(1)), "old_len": int(m.group(2) or 1), "new_start": int(m.group(3)),
                    "new_len": int(m.group(4) or 1), "lines": []}
            cur["hunks"].append(hunk)
            continue
        if hunk is not None and line[:1] in (" ", "+", "-"):
            hunk["lines"].append(line)
        elif hunk is not None and line.startswith("\\"):
            if hunk["lines"]:
                hunk["lines"][-1] = hunk["lines"][-1] + "\\NONL"
    return files


def _apply_hunks(lines, hunks, reverse):
    out = list(lines)
    offset = 0
    for h in hunks:
        old, new = [], []
        for ln in h["lines"]:
            nonl = ln.endswith("\\NONL")
            body = ln[1:-5] if nonl else ln[1:]
            text = body if nonl else body + "\n"
            tag = ln[0]
            if reverse:
                tag = {"+": "-", "-": "+"}.get(tag, tag)
            if tag in (" ", "-"):
                old.append(text)
            if tag in (" ", "+"):
                new.append(text)
        start = (h["new_start"] if reverse else h["old_start"]) - 1 + offset
        if not old:
            start = max(0, start + 1 if h["old_len"] == 0 and not reverse else start)
        pos = None
        for delta in range(0, len(out) + 1):
            for cand in (start + delta, start - delta):
                if 0 <= cand <= len(out) and out[cand:cand + len(old)] == old:
                    pos = cand
                    break
            if pos is not None:
                break
        if pos is None:
            raise ValueError("hunk")
        out[pos:pos + len(old)] = new
        offset += len(new) - len(old)
    return out


@command("var")
def git_var(g, args):
    repo = g.maybe_repo()
    from .repo import Config
    cfg = repo.config if repo else Config(g.fs, g.env.get("HOME", "/"), None, (), g.env)
    if not args:
        raise Usage("usage: git var (-l | <variable>)")
    v = args[0]
    if v == "GIT_EDITOR":
        g.out((g.env.get("GIT_EDITOR") or cfg.get("core.editor") or g.env.get("VISUAL") or g.env.get("EDITOR") or "vi")
              + "\n")
    elif v == "GIT_AUTHOR_IDENT" and repo:
        g.out(repo.ident("author").line() + "\n")
    elif v == "GIT_COMMITTER_IDENT" and repo:
        g.out(repo.ident("committer").line() + "\n")
    elif v == "GIT_PAGER":
        g.out((g.env.get("GIT_PAGER") or cfg.get("core.pager") or g.env.get("PAGER") or "less") + "\n")
    elif v == "GIT_DEFAULT_BRANCH":
        g.out((cfg.get("init.defaultBranch") or "master") + "\n")
    else:
        raise Usage("usage: git var (-l | <variable>)")
    return 0

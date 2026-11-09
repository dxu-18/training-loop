"""Revisions as git spells them: names (refs/heads/x, tags, remote branches), full or abbreviated ids, HEAD, @,
x~n, x^n, x^{tree}, x@{n} (reflog), @{-n} (the n-th previous branch), x@{u} (upstream), :/text (the newest commit
whose message matches), rev:path, :path (the index), and ranges a..b, a...b, ^a for log."""

from __future__ import annotations

import re

from .objects import GitError

SPECIAL = ("HEAD", "ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "REBASE_HEAD",
           "BISECT_HEAD", "AUTO_MERGE")


def dwim_ref(repo, name: str):
    """A short name -> (full ref name, sha), in git's order: refs/x, refs/tags/x, refs/heads/x, refs/remotes/x,
    refs/remotes/x/HEAD."""
    if name in SPECIAL or name.startswith("refs/"):
        sha = repo.resolve_ref(name)
        if sha and name == "FETCH_HEAD":
            sha = sha
        return (name, sha) if sha else (None, None)
    for cand in (f"refs/{name}", f"refs/tags/{name}", f"refs/heads/{name}", f"refs/remotes/{name}",
                 f"refs/remotes/{name}/HEAD"):
        sha = repo.resolve_ref(cand)
        if sha:
            return cand, sha
    return None, None


def _fetch_head(repo):
    text = repo.read("FETCH_HEAD") or ""
    for line in text.split("\n"):
        if line and "not-for-merge" not in line:
            return line.split()[0]
    return None


def previous_branch(repo, n: int):
    """@{-n}: the branch checked out n switches ago (from the HEAD reflog)."""
    count = 0
    for old, new, who, msg in repo.reflog("HEAD"):
        m = re.match(r"checkout: moving from (\S+) to (\S+)", msg)
        if m:
            count += 1
            if count == n:
                return m.group(1)
    return None


def resolve(repo, spec: str, want: str | None = None, quiet: bool = False):
    """A revision -> object id (peeled to `want`: commit/tree), or raises GitError("unknown")."""
    sha = _resolve(repo, spec)
    if sha is None:
        raise GitError("unknown revision", 128)
    if want == "commit":
        sha = peel_to(repo, sha, "commit", spec)
    elif want == "tree":
        sha = peel_to(repo, sha, "tree", spec)
    return sha


def try_resolve(repo, spec: str, want: str | None = None):
    try:
        return resolve(repo, spec, want)
    except GitError:
        return None


def peel_to(repo, sha: str, kind: str, spec: str = ""):
    db = repo.odb
    seen = 0
    while seen < 10:
        k = db.kind(sha)
        if k == kind:
            return sha
        if k == "tag":
            sha = db.tag(sha).object
        elif k == "commit" and kind == "tree":
            return db.commit(sha).tree
        else:
            raise GitError(f"fatal: {spec or sha}: expected {kind} type, but the object dereferences to {k} type")
        seen += 1
    raise GitError(f"fatal: {spec}: too many tags")


def _resolve(repo, spec: str):
    db = repo.odb
    if not spec:
        return None
    # rev:path and :path
    if spec.startswith(":/"):
        return _search_message(repo, spec[2:])
    if spec.startswith(":"):
        m = re.match(r"^:([0-3]):(.*)$", spec)
        stage, path = (int(m.group(1)), m.group(2)) if m else (0, spec[1:])
        e = repo.index.get(path, stage)
        return e.sha if e else None
    m = re.match(r"^(.*?[^:]):(.*)$", spec) if ":" in spec and not spec.startswith("refs/") else None
    if m and not re.match(r"^[0-9a-f]{4,40}$", spec):
        base = _resolve(repo, m.group(1))
        if base is None:
            return None
        tree = peel_to(repo, base, "tree", m.group(1))
        path = m.group(2).strip("/")
        if not path:
            return tree
        flat = db.flat(tree)
        if path in flat:
            return flat[path][1]
        # a directory
        return _subtree(repo, tree, path)
    # suffixes: ^{...}, ~n, ^n (right to left)
    m = re.match(r"^(.*)\^\{(\w*)\}$", spec)
    if m:
        base = _resolve(repo, m.group(1))
        if base is None:
            return None
        kind = m.group(2)
        if kind == "":
            return repo.peel(base)
        return peel_to(repo, base, kind, spec)
    m = re.match(r"^(.*?)((?:[~^]\d*)+)$", spec)
    if m and m.group(1) and not spec.startswith("@{"):
        base = _resolve(repo, m.group(1))
        if base is None:
            return None
        sha = peel_to(repo, base, "commit", m.group(1))
        for op, num in re.findall(r"([~^])(\d*)", m.group(2)):
            n = int(num) if num else 1
            if op == "~":
                for _ in range(n):
                    ps = db.commit(sha).parents
                    if not ps:
                        return None
                    sha = ps[0]
            else:
                if n == 0:
                    continue
                ps = db.commit(sha).parents
                if len(ps) < n:
                    return None
                sha = ps[n - 1]
        return sha
    # @{...}
    m = re.match(r"^(.*?)@\{(.+)\}$", spec)
    if m:
        name, what = m.group(1), m.group(2)
        if what.startswith("-") and not name:
            try:
                b = previous_branch(repo, int(what[1:]))
            except ValueError:
                return None
            return _resolve(repo, b) if b else None
        if what.lower() in ("u", "upstream", "push"):
            branch = name or repo.head_branch()
            if name and name.startswith("refs/heads/"):
                branch = name[len("refs/heads/"):]
            if not branch:
                raise GitError("fatal: HEAD does not point to a branch")
            up = repo.upstream_ref(branch)
            if not up:
                raise GitError(f"fatal: no upstream configured for branch '{branch}'")
            sha = repo.resolve_ref(up)
            return sha
        if what.isdigit():
            ref = "HEAD" if name in ("", "HEAD", "@") else (dwim_ref(repo, name)[0] or name)
            log = repo.reflog(ref)
            n = int(what)
            if n < len(log):
                return log[n][1]
            if n == len(log) and log:
                return log[-1][0] if log[-1][0] != "0" * 40 else None
            return None
        return None
    if spec == "@":
        spec = "HEAD"
    if spec == "FETCH_HEAD":
        return _fetch_head(repo)
    full, sha = dwim_ref(repo, spec)
    if sha:
        return sha
    if re.fullmatch(r"[0-9a-f]{4,40}", spec):
        hits = db.resolve_prefix(spec)
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # prefer a commit
            commits = [h for h in hits if db.kind(h) == "commit"]
            if len(commits) == 1:
                return commits[0]
            raise GitError(f"error: short object ID {spec} is ambiguous")
    return None


def _subtree(repo, tree, path):
    db = repo.odb
    sha = tree
    for part in path.split("/"):
        for mode, name, s in db.tree(sha):
            if name == part:
                sha = s
                break
        else:
            return None
    return sha


def _search_message(repo, text: str):
    rx = re.compile(text)
    heads = [v for v in repo.ref_names("refs/").values()]
    h = repo.head()
    if h:
        heads.append(h)
    heads = [repo.peel(x) for x in heads if repo.odb.has(x)]
    heads = [x for x in heads if repo.odb.kind(x) == "commit"]
    for sha in repo.odb.ancestors(heads):
        if rx.search(repo.odb.commit(sha).message):
            return sha
    return None


def is_revision(repo, spec: str) -> bool:
    if spec.startswith("^"):
        spec = spec[1:]
    if "..." in spec:
        a, b = spec.split("...", 1)
        return (not a or try_resolve(repo, a) is not None) and (not b or try_resolve(repo, b) is not None)
    if ".." in spec:
        a, b = spec.split("..", 1)
        return (not a or try_resolve(repo, a) is not None) and (not b or try_resolve(repo, b) is not None)
    return try_resolve(repo, spec) is not None


def rev_list_args(repo, specs):
    """Revision arguments of log/rev-list -> (include ids, exclude ids, symmetric pair or None)."""
    inc, exc = [], []
    sym = None
    for s in specs:
        if s.startswith("^"):
            exc.append(resolve(repo, s[1:], "commit"))
        elif "..." in s:
            a, b = s.split("...", 1)
            a = resolve(repo, a or "HEAD", "commit")
            b = resolve(repo, b or "HEAD", "commit")
            inc += [a, b]
            exc += repo.odb.merge_bases(a, b)
            sym = (a, b)
        elif ".." in s:
            a, b = s.split("..", 1)
            exc.append(resolve(repo, a or "HEAD", "commit"))
            inc.append(resolve(repo, b or "HEAD", "commit"))
        else:
            inc.append(resolve(repo, s, "commit"))
    return inc, exc, sym


def unknown_revision(spec: str) -> GitError:
    return GitError(f"fatal: ambiguous argument '{spec}': unknown revision or path not in the working tree.\n"
                    "Use '--' to separate paths from revisions, like this:\n"
                    "'git <command> [<revision>...] -- [<file>...]'")

"""The git problems' checker: print a repository's state (and its GitHub-lite side) so that two sessions that
reach the same state print the same text, whatever the timing.

Commit ids depend on dates, and a person types slower than the reference solution runs, so commits are named by a
content id instead: a hash of the message, the author, the tree and the parents' content ids (committer and dates
left out). Commit lists are ordered by generation (distance from the root) and content id, never by date.

    /usr/lib/bashlite/gitstate [-C DIR] SECTION...

Sections: head branches log log-all tags index worktree untracked stash remotes upstream state files config=KEY
          gh=OWNER/NAME ghme hooks lfs submodules worktrees sparse ignored=PATH content=PATH
"""

from __future__ import annotations

import hashlib

from ..exec import util
from ..vfs import FSError
from .objects import GitError


class _Ids:
    def __init__(self, db):
        self.db, self.memo = db, {}

    def cid(self, sha):
        """The content id of a commit (7 hex digits)."""
        if sha in self.memo:
            return self.memo[sha]
        stack = [sha]
        while stack:
            s = stack[-1]
            if s in self.memo:
                stack.pop()
                continue
            try:
                c = self.db.commit(s)
            except GitError:
                self.memo[s] = "missing"
                stack.pop()
                continue
            pending = [p for p in c.parents if p not in self.memo and self.db.has(p)]
            if pending:
                stack.extend(pending)
                continue
            parts = [c.message, c.author.name, c.author.email, c.tree] + [self.memo.get(p, "cut") for p in c.parents]
            self.memo[s] = hashlib.sha1("\0".join(parts).encode("utf-8", "surrogateescape")).hexdigest()[:7]
            stack.pop()
        return self.memo[sha]

    def desc(self, sha):
        try:
            kind = self.db.kind(sha)
        except GitError:
            return "missing"
        if kind == "tag":
            t = self.db.tag(sha)
            return f"tag({t.message.strip()[:40]!r})->{self.desc(t.object)}"
        if kind != "commit":
            return f"{kind}:{sha[:7]}"
        return self.cid(sha)

    def generations(self, tips):
        gen = {}
        order = self.db.ancestors(list(tips))
        for s in reversed(order):            # oldest first is enough for a DAG walked newest-first
            ps = [p for p in self.db.parents(s) if p in gen]
            gen[s] = 1 + max((gen[p] for p in ps), default=0)
        # fix-up passes for any parent seen after its child
        changed = True
        while changed:
            changed = False
            for s in order:
                g = 1 + max((gen.get(p, 0) for p in self.db.parents(s)), default=0)
                if g != gen[s]:
                    gen[s] = g
                    changed = True
        return gen


def _short(ref):
    for p in ("refs/heads/", "refs/remotes/", "refs/tags/"):
        if ref.startswith(p):
            return ref[len(p):]
    return ref


def repo_lines(repo, sections, ctx) -> list[str]:
    db = repo.odb
    ids = _Ids(db)
    out = []
    refs = repo.ref_names("refs/")
    for sec in sections:
        key, _, val = sec.partition("=")
        if key == "head":
            b = repo.head_branch()
            h = repo.head()
            out.append(f"HEAD -> {b}" if b else f"HEAD detached at {ids.desc(h) if h else 'nothing'}")
            if b and not h:
                out.append("(no commits yet)")
        elif key == "branches":
            for name, sha in sorted(repo.branches().items()):
                out.append(f"branch {name} {ids.desc(sha)}")
        elif key == "upstream":
            for name in sorted(repo.branches()):
                up = repo.upstream_ref(name)
                out.append(f"upstream {name} -> {_short(up) if up else '(none)'}")
        elif key in ("log", "log-all"):
            tips = [repo.peel(s) for n, s in refs.items() if n != "refs/stash" and not n.endswith("/HEAD") and
                    (key == "log-all" or not n.startswith("refs/remotes/"))]
            if repo.head():
                tips.append(repo.head())
            tips = [t for t in tips if db.has(t) and db.kind(t) == "commit"]
            gen = ids.generations(tips) if tips else {}
            rows = sorted(gen, key=lambda s: (gen[s], ids.cid(s)))
            for s in rows[-80:]:
                c = db.commit(s)
                ps = " ".join(ids.cid(p) for p in c.parents if db.has(p))
                out.append(f"commit {ids.cid(s)} [{c.author.name}] {c.subject!r}" + (f" <- {ps}" if ps else ""))
        elif key == "tags":
            for name, sha in sorted(repo.tags().items()):
                out.append(f"tag {name} {ids.desc(sha)}")
        elif key == "remotes":
            for r in repo.remotes():
                out.append(f"remote {r} {repo.config.get(f'remote.{r}.url')}")
            for name, sha in sorted(repo.remote_branches().items()):
                if not name.endswith("/HEAD"):
                    out.append(f"remote-branch {name} {ids.desc(sha)}")
        elif key in ("index", "worktree", "untracked"):
            from .status import collect
            try:
                st = collect(repo, untracked="all", refresh=False)
            except (GitError, FSError) as e:
                out.append(f"{key}: error {e}")
                continue
            if key == "index":
                for p, v in sorted(st.staged.items()):
                    e = repo.index.get(p)
                    out.append(f"staged {v if isinstance(v, str) else v[0]} {p}" + (f" {e.sha[:7]}" if e else ""))
                for p in sorted(st.unmerged):
                    out.append(f"unmerged {p}")
            elif key == "worktree":
                for p, v in sorted(st.unstaged.items()):
                    got = repo.wt_sha(p) if v != "D" else None
                    out.append(f"unstaged {v} {p}" + (f" {got[0][:7]}" if got else ""))
            else:
                for p in st.untracked:
                    got = repo.wt_sha(p.rstrip("/")) if not p.endswith("/") else None
                    out.append(f"untracked {p}" + (f" {got[0][:7]}" if got else ""))
        elif key == "stash":
            for i, (old, new, who, msg) in enumerate(repo.reflog("refs/stash")):
                c = db.commit(new) if db.has(new) else None
                out.append(f"stash@{{{i}}} {msg.split(': ', 1)[-1]!r} tree {c.tree[:7] if c else '?'}")
        elif key == "state":
            from .cmd_basic import in_progress
            st, _ = in_progress(repo)
            out.append(f"in progress: {st or 'nothing'}")
        elif key == "files":
            for p, (mode, sha) in sorted(db.flat(repo.head_tree()).items()):
                out.append(f"file {mode} {p} {sha[:7]}")
        elif key == "config":
            v = repo.config.get_all(val)
            out.append(f"config {val} = {v if v else '(unset)'}")
        elif key == "hooks":
            try:
                node = ctx.fs.lookup(repo.path("hooks", True))
                for name in sorted(node.entries):
                    n = node.entries[name]
                    if not name.endswith(".sample") and n.type == "f" and n.mode & 0o111:
                        out.append(f"hook {name}")
            except FSError:
                pass
        elif key == "lfs":
            pats = [p for p, base, kv in repo.attributes() if "filter=lfs" in kv]
            out.append(f"lfs patterns {sorted(pats)}")
            from .lfs import parse_pointer
            for p, (mode, sha) in sorted(db.flat(repo.head_tree()).items()):
                if mode != "160000" and db.size(sha) < 400 and parse_pointer(db.raw(sha)):
                    node = repo.wt_node(p)
                    real = node is not None and parse_pointer(node.data) is None
                    out.append(f"lfs {p} {'content' if real else 'pointer'}")
        elif key == "submodules":
            for p, (mode, sha) in sorted(db.flat(repo.head_tree()).items()):
                if mode == "160000":
                    from .discover import open_repo_at
                    sub = open_repo_at(ctx, repo.wt_path(p)) if repo.worktree else None
                    out.append(f"submodule {p} {sha[:7]} checked-out={sub.head()[:7] if sub and sub.head() else '-'}")
        elif key == "worktrees":
            from .cmd_extra import _worktrees
            for t, r in _worktrees(repo):
                b = r.head_branch()
                out.append(f"worktree {t} {b or 'detached'} {ids.desc(r.head()) if r.head() else '-'}")
        elif key == "sparse":
            from .cmd_extra import _sparse_dirs
            on = repo.config.get_bool("core.sparseCheckout", False)
            out.append(f"sparse {on} {_sparse_dirs(repo) if on else []}")
        elif key == "ignored":
            out.append(f"ignored {val} {repo.is_ignored(val, val.endswith('/'))}")
        elif key == "content":
            node = repo.wt_node(val)
            if node is None:
                out.append(f"content {val} (missing)")
            else:
                data = bytes(node.data)
                out.append(f"content {val} {hashlib.sha1(data).hexdigest()[:10]}")
        elif not key.startswith(("gh", "ghme")):
            out.append(f"unknown section {key}")
    return out


def github_lines(world, sections) -> list[str]:
    gh = world.__dict__.get("github")
    out = []
    if gh is None:
        return ["github: none"]
    from .objects import odb
    ids = _Ids(odb(world))
    for sec in sections:
        key, _, val = sec.partition("=")
        if key == "ghme":
            me = gh.me
            out.append(f"github user {me}")
            if me:
                for full in sorted(r.full for r in gh.repos.values() if r.owner == me):
                    r = gh.repos[full]
                    out.append(f"  repo {full} private={r.private} fork_of={r.fork_of}")
                acc = gh.accounts.get(me)
                out.append(f"  ssh keys {len(acc.keys) if acc else 0}")
        elif key == "gh":
            owner, _, name = val.partition("/")
            r = gh.get(owner, name, materialize=False)
            if r is None:
                out.append(f"github {val}: no such repository")
                continue
            out.append(f"github {r.full} private={r.private} default={r.default_branch} fork_of={r.fork_of}")
            for ref, sha in sorted(r.refs.items()):
                if ref.startswith(("refs/heads/", "refs/tags/")):
                    out.append(f"  {'branch' if ref.startswith('refs/heads/') else 'tag'} {_short(ref)} "
                               f"{ids.desc(sha)}")
            for b, rule in sorted(r.protected.items()):
                out.append(f"  protected {b} {sorted(rule.items())}")
            for pr in sorted(gh.pulls_of(r), key=lambda p: p.number):
                out.append(f"  pr #{pr.number} {pr.title!r} {pr.state} {pr.head_repo.split('/')[0]}:{pr.head_branch}"
                           f" -> {pr.base} draft={pr.draft} reviews={sorted((a, s) for a, s, _ in pr.reviews)}"
                           f" comments={len(pr.comments)}")
            for it in sorted(gh.issues_of(r), key=lambda i: i.number):
                out.append(f"  issue #{it.number} {it.title!r} {it.state} labels={sorted(it.labels)} "
                           f"comments={len(it.comments)}")
            for rel in r.releases:
                out.append(f"  release {rel.tag} {rel.name!r} prerelease={rel.prerelease}")
            if r.pages:
                out.append(f"  pages {r.pages.get('branch')} {r.pages.get('path')}")
    return out


@util("bashlite-gitstate")
def u_gitstate(ctx):
    from .cli import G
    a = list(ctx.argv[1:])
    target = None
    if len(a) >= 2 and a[0] == "-C":
        target = ctx.abspath(a[1])
        a = a[2:]
    secs = a or ["head", "branches", "log", "index", "worktree"]
    out = []
    repo_secs = [s for s in secs if not s.startswith(("gh=", "ghme"))]
    if repo_secs:
        from .discover import find_repo
        g = G(ctx)
        try:
            repo = find_repo(ctx, start=target or ctx.cwd, required=False)
        except GitError:
            repo = None
        if repo is None:
            out.append("no repository")
        else:
            out.append(f"repository {ctx.world and (repo.worktree or repo.gitdir)}")
            out += repo_lines(repo, repo_secs, ctx)
        del g
    gh_secs = [s for s in secs if s.startswith(("gh=", "ghme"))]
    if gh_secs:
        out += github_lines(ctx.world, gh_secs)
    ctx.w("\n".join(out) + "\n")
    return 0
    yield

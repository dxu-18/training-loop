"""What the terminal's Git and GitHub panels show: a compact, JSON-ready picture of the repository the user is in
(or the one they most recently used under the start directory) and of its GitHub-lite remote."""

from __future__ import annotations

import posixpath

from ..vfs import FSError
from .objects import GitError

MAX_COMMITS = 60
MAX_FILES = 40


def _find_repo(ctx, cwd: str, start: str):
    """The repository at or above cwd; otherwise the first one found under the start directory."""
    from .discover import find_repo, open_repo_at
    try:
        return find_repo(ctx, start=cwd, required=False)
    except (GitError, FSError, TypeError):
        pass
    fs = ctx.fs
    queue = [start]
    seen = 0
    while queue and seen < 400:
        d = queue.pop(0)
        seen += 1
        try:
            node = fs.lookup(d)
        except FSError:
            continue
        if node.type != "d":
            continue
        if ".git" in node.entries:
            r = open_repo_at(ctx, d)
            if r is not None:
                return r
        for name in sorted(node.entries):
            if not name.startswith(".") and node.entries[name].type == "d":
                queue.append(f"{d.rstrip('/')}/{name}")
    return None


def _short(ref: str) -> str:
    for p in ("refs/heads/", "refs/remotes/", "refs/tags/"):
        if ref.startswith(p):
            return ref[len(p):]
    return ref


def repo_snapshot(repo) -> dict:
    db = repo.odb
    head = repo.head()
    branch = repo.head_branch()
    refs = repo.ref_names("refs/")
    labels: dict[str, list] = {}
    tips = []
    for name, sha in sorted(refs.items()):
        if name == "refs/stash" or name.endswith("/HEAD"):
            continue
        try:
            c = repo.peel(sha)
            if db.kind(c) != "commit":
                continue
        except GitError:
            continue
        kind = "tag" if name.startswith("refs/tags/") else "remote" if name.startswith("refs/remotes/") else "branch"
        labels.setdefault(c, []).append({"n": _short(name), "k": kind, "head": kind == "branch" and
                                         _short(name) == branch})
        tips.append(c)
    if head:
        tips.append(head)
        if not branch:
            labels.setdefault(head, []).insert(0, {"n": "HEAD", "k": "head", "head": True})
    from .cmd_history import topo_sort
    commits = []
    if tips:
        try:
            anc = db.ancestors(list(dict.fromkeys(tips)))
            order = topo_sort(db, anc[:MAX_COMMITS * 3])[:MAX_COMMITS]
        except GitError:
            order = []
        keep = set(order)
        for c in order:
            obj = db.commit(c)
            commits.append({"id": db.abbrev(c), "p": [db.abbrev(p) for p in db.parents(c) if p in keep],
                            "pm": len(obj.parents) > 1, "s": obj.subject[:72], "a": obj.author.name,
                            "t": obj.author.when, "r": labels.get(c, [])})
        truncated = len(anc) > len(order)
    else:
        truncated = False
    # branches with their upstreams
    branches = []
    from .cmd_basic import ahead_behind
    for name, sha in sorted(repo.branches().items()):
        up = repo.upstream_ref(name)
        them = repo.resolve_ref(up) if up else None
        ab = ahead_behind(repo, sha, them) if them else (0, 0)
        branches.append({"n": name, "id": db.abbrev(sha), "head": name == branch,
                         "up": _short(up) if up else None, "ahead": ab[0], "behind": ab[1],
                         "gone": bool(up) and them is None})
    # the three areas
    areas = {"staged": [], "unstaged": [], "untracked": [], "conflicts": []}
    if not repo.bare:
        from .status import collect
        try:
            st = collect(repo, untracked="normal", refresh=False)
            areas["staged"] = [[v if isinstance(v, str) else v[0], p] for p, v in sorted(st.staged.items())][:MAX_FILES]
            areas["unstaged"] = [[v, p] for p, v in sorted(st.unstaged.items())][:MAX_FILES]
            areas["untracked"] = list(st.untracked)[:MAX_FILES]
            areas["conflicts"] = sorted(st.unmerged)[:MAX_FILES]
        except (GitError, FSError):
            pass
    from .cmd_basic import in_progress
    state, _ = in_progress(repo)
    stash = []
    for i, (old, new, who, msg) in enumerate(repo.reflog("refs/stash")):
        stash.append({"n": f"stash@{{{i}}}", "s": msg[:72]})
    remotes = []
    for r in repo.remotes():
        remotes.append({"n": r, "url": repo.config.get(f"remote.{r}.url") or ""})
    return {"path": repo.worktree or repo.gitdir, "bare": repo.bare, "head": {"branch": branch,
            "id": db.abbrev(head) if head else None}, "commits": commits, "more": truncated, "branches": branches,
            "tags": sorted(_short(n) for n in refs if n.startswith("refs/tags/")), "areas": areas, "state": state,
            "stash": stash, "remotes": remotes, "shallow": bool(repo.shallow())}


def github_snapshot(world, repo) -> dict | None:
    gh = world.__dict__.get("github")
    if gh is None:
        return None
    from .remote import parse_url
    target = None
    if repo is not None:
        for r in repo.remotes():
            url = repo.config.get(f"remote.{r}.url") or ""
            ep = parse_url(repo.ctx, url) if url else None
            if ep is not None and ep.kind == "github":
                target = gh.get(ep.owner, ep.repo, materialize=False)
                if target is not None and r == "origin":
                    break
    me = gh.me
    if target is None:
        mine = [r for r in gh.repos.values() if me and r.owner == me]
        target = mine[-1] if mine else None
    out = {"me": me, "keys": len(gh.accounts[me].keys) if me in gh.accounts else 0}
    if me:
        out["repos"] = sorted(r.full for r in gh.repos.values() if r.owner == me)[:12]
    if target is None:
        return out
    db = target.db
    now = gh.now()
    prs = []
    for pr in sorted(gh.pulls_of(target), key=lambda p: -p.number)[:12]:
        runs = [r for r in target.runs if r.sha == pr.head_sha]
        checks = None
        if runs:
            st, concl = runs[-1].status(now)
            checks = concl or st
        prs.append({"n": pr.number, "title": pr.title[:70], "state": "DRAFT" if pr.draft and pr.state == "OPEN" else
                    pr.state, "head": pr.head_branch, "base": pr.base, "by": pr.author, "checks": checks,
                    "approved": sum(1 for _, s, _ in pr.reviews if s == "APPROVED"),
                    "changes": sum(1 for _, s, _ in pr.reviews if s == "CHANGES_REQUESTED")})
    issues = [{"n": i.number, "title": i.title[:70], "state": i.state, "labels": i.labels[:3]}
              for i in sorted(gh.issues_of(target), key=lambda i: -i.number)[:8]]
    branches = []
    for ref, sha in sorted(target.refs.items()):
        if ref.startswith("refs/heads/"):
            name = ref[len("refs/heads/"):]
            branches.append({"n": name, "id": db.abbrev(sha), "default": name == target.default_branch,
                             "protected": name in target.protected})
    runs = []
    for r in target.runs[-5:][::-1]:
        st, concl = r.status(now)
        runs.append({"wf": r.workflow, "title": r.title[:60], "branch": r.branch, "status": concl or st})
    out["repo"] = {"full": target.full, "desc": target.description, "private": target.private,
                   "fork_of": target.fork_of, "default": target.default_branch, "stars": target.stars,
                   "real": target.real, "branches": branches[:20], "tags": sorted(_short(r) for r in target.refs
                                                                                  if r.startswith("refs/tags/"))[-8:],
                   "prs": prs, "issues": issues, "runs": runs, "releases": [x.tag for x in target.releases][-5:],
                   "pages": (f"https://{target.owner.lower()}.github.io/" +
                             ("" if target.name.lower() == f"{target.owner.lower()}.github.io" else target.name + "/"))
                   if target.pages else None}
    return out


def snapshot(ctx, cwd: str, start: str) -> dict | None:
    repo = _find_repo(ctx, cwd, start)
    out = {}
    if repo is not None:
        try:
            out["repo"] = repo_snapshot(repo)
        except (GitError, FSError) as e:
            out["repo"] = {"error": str(e)}
    gh = github_snapshot(ctx.world, repo)
    if gh is not None:
        out["github"] = gh
    if not out:
        return None
    out["cwd"] = cwd
    return out


__all__ = ["snapshot", "posixpath"]

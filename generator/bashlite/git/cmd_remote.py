"""remote, ls-remote, fetch, pull, push."""

from __future__ import annotations

import fnmatch
import hashlib

from .cli import Usage, command, parse, run_maybe
from .github import confirm_host, github_of, https_account, ssh_account
from .objects import GitError
from .remote import Far, missing_objects, object_stats, open_far_repo, parse_url, push_progress, ref_line, \
    remote_progress, stderr_is_tty
from .revparse import dwim_ref, try_resolve

NOT_FOUND_SSH = ("ERROR: Repository not found.\nfatal: Could not read from remote repository.\n\n"
                 "Please make sure you have the correct access rights\nand the repository exists.")
DENIED_SSH = ("git@github.com: Permission denied (publickey).\nfatal: Could not read from remote repository.\n\n"
              "Please make sure you have the correct access rights\nand the repository exists.")


def connect(g, url: str, push: bool = False):
    """Open the far side of a URL (generator) -> Far. Raises GitError the way git reports it."""
    ep = parse_url(g.ctx, url)
    ep.shown_url = url
    if ep.kind != "github":
        return Far(ep, repo=open_far_repo(g.ctx, ep))
    gh = github_of(g.world)
    acc = None
    if ep.protocol == "ssh":
        ok = yield from confirm_host(g.ctx, gh)
        if not ok:
            raise GitError("fatal: Could not read from remote repository.\n\nPlease make sure you have the correct "
                           "access rights\nand the repository exists.")
        acc = ssh_account(g.ctx, gh)
        if acc is None:
            raise GitError(DENIED_SSH)
    r = gh.get(ep.owner, ep.repo)
    if ep.protocol == "https":
        need = push or (r is not None and r.private)
        acc, err = yield from https_account(g.ctx, gh, url, need)
        if err:
            raise GitError(err)
    if r is None or (r.private and not r.can_push(acc.login if acc else None)):
        if ep.protocol == "https":
            raise GitError(f"remote: Repository not found.\nfatal: repository '{url.rstrip('/')}/' not found")
        raise GitError(NOT_FOUND_SSH)
    return Far(ep, gh=gh, ghrepo=r, account=acc)


def remote_url(repo, name: str):
    url = repo.config.get(f"remote.{name}.url")
    return url


def remote_and_url(repo, name):
    """A remote name or a URL -> (remote name or None, url)."""
    url = remote_url(repo, name)
    if url:
        return name, url
    if "/" in name or ":" in name or name.startswith("."):
        return None, name
    raise GitError(f"fatal: '{name}' does not appear to be a git repository\nfatal: Could not read from remote "
                   "repository.\n\nPlease make sure you have the correct access rights\nand the repository exists.")


# ---------------------------------------------------------------------------- remote

@command("remote")
def git_remote(g, args):
    repo = g.repo()
    cfg = repo.config.local_file()
    uid, gid = g.ctx.uid, g.ctx.proc.gid
    sub = args[0] if args and not args[0].startswith("-") else None
    rest = args[1:] if sub else args
    if sub is None:
        o = parse(rest, "verbose|v")
        out = []
        for name in repo.remotes():
            if o.get("verbose"):
                url = repo.config.get(f"remote.{name}.url") or ""
                push = repo.config.get(f"remote.{name}.pushurl") or url
                out.append(f"{name}\t{url} (fetch)\n{name}\t{push} (push)")
            else:
                out.append(name)
        if out:
            g.out("\n".join(out) + "\n")
        return 0
    if sub == "add":
        o = parse(rest, "fetch|f track|t= master|m= tags no-tags mirror=?")
        if len(o.args) != 2:
            raise Usage("usage: git remote add [<options>] <name> <url>")
        name, url = o.args
        if name in repo.remotes():
            raise GitError(f"error: remote {name} already exists.", 3)
        cfg.set(f"remote.{name}.url", url, uid, gid)
        cfg.set(f"remote.{name}.fetch", f"+refs/heads/*:refs/remotes/{name}/*", uid, gid)
        repo.reload_config()
        if o.get("fetch"):
            return (yield from git_fetch(g, [name]))
        return 0
    if sub in ("remove", "rm"):
        if not rest:
            raise Usage("usage: git remote remove <name>")
        name = rest[0]
        if name not in repo.remotes():
            raise GitError(f"error: No such remote: '{name}'", 2)
        cfg.remove_section(f"remote.{name}", uid, gid)
        for b in repo.branches():
            if repo.config.get(f"branch.{b}.remote") == name:
                cfg.unset(f"branch.{b}.remote", uid, gid)
                cfg.unset(f"branch.{b}.merge", uid, gid)
        for ref in list(repo.ref_names(f"refs/remotes/{name}/")):
            repo.delete_ref(ref)
        repo.remove(f"refs/remotes/{name}/HEAD", True)
        repo.reload_config()
        return 0
    if sub == "rename":
        if len(rest) != 2:
            raise Usage("usage: git remote rename [--[no-]progress] <old> <new>")
        old, new = rest
        if old not in repo.remotes():
            raise GitError(f"error: No such remote: '{old}'", 2)
        if new in repo.remotes():
            raise GitError(f"error: remote {new} already exists.", 3)
        cfg.rename_section(f"remote.{old}", f"remote.{new}", uid, gid)
        repo.reload_config()
        fetch = repo.config.get(f"remote.{new}.fetch")
        if fetch:
            cfg.set(f"remote.{new}.fetch", fetch.replace(f"refs/remotes/{old}/", f"refs/remotes/{new}/"), uid, gid)
        for b in repo.branches():
            if repo.config.get(f"branch.{b}.remote") == old:
                cfg.set(f"branch.{b}.remote", new, uid, gid)
        for ref, sha in list(repo.ref_names(f"refs/remotes/{old}/").items()):
            repo.update_ref(ref.replace(f"refs/remotes/{old}/", f"refs/remotes/{new}/"), sha, None)
            repo.delete_ref(ref)
        r = repo.read_ref_raw(f"refs/remotes/{old}/HEAD")
        if r and r[0] == "sym":
            repo.remove(f"refs/remotes/{old}/HEAD", True)
            repo.write(f"refs/remotes/{new}/HEAD", f"ref: {r[1].replace(f'refs/remotes/{old}/', f'refs/remotes/{new}/')}\n",
                       True)
        repo.reload_config()
        return 0
    if sub == "set-url":
        o = parse(rest, "push add delete")
        if len(o.args) < 2:
            raise Usage("usage: git remote set-url [--push] <name> <newurl> [<oldurl>]")
        name, url = o.args[0], o.args[1]
        if name not in repo.remotes():
            raise GitError(f"error: No such remote '{name}'", 2)
        key = f"remote.{name}.pushurl" if o.get("push") else f"remote.{name}.url"
        cfg.set(key, url, uid, gid, add=bool(o.get("add")))
        repo.reload_config()
        return 0
    if sub == "get-url":
        o = parse(rest, "push all")
        name = o.args[0] if o.args else ""
        if name not in repo.remotes():
            raise GitError(f"error: No such remote '{name}'", 2)
        url = repo.config.get(f"remote.{name}.pushurl" if o.get("push") else f"remote.{name}.url") or \
            repo.config.get(f"remote.{name}.url")
        g.out(url + "\n")
        return 0
    if sub == "show":
        o = parse(rest, "n")
        names = o.args or repo.remotes()
        for name in names:
            if name not in repo.remotes():
                raise GitError(f"fatal: '{name}' does not appear to be a git repository")
            url = repo.config.get(f"remote.{name}.url")
            far = None
            try:
                far = yield from connect(g, url)
            except GitError:
                pass
            lines = [f"* remote {name}", f"  Fetch URL: {url}", f"  Push  URL: {url}"]
            fr = {}
            if far is not None:
                head = far.head_symref()
                lines.append(f"  HEAD branch: {head[len('refs/heads/'):] if head else '(unknown)'}")
                fr = far.refs()
                heads = sorted(r[len("refs/heads/"):] for r in fr if r.startswith("refs/heads/"))
                stale = [r[len(f"refs/remotes/{name}/"):] for r in sorted(repo.ref_names(f"refs/remotes/{name}/"))
                         if not r.endswith("/HEAD") and r[len(f"refs/remotes/{name}/"):] not in heads]
                rows = [(b, "tracked" if repo.resolve_ref(f"refs/remotes/{name}/{b}") is not None else
                         f"new (next fetch will store in remotes/{name})") for b in heads]
                rows += [(f"refs/remotes/{name}/{b}", "stale (use 'git remote prune' to remove)") for b in stale]
                w = max((len(b) for b, _ in rows), default=0)
                lines.append(f"  Remote branch{'es' if len(rows) != 1 else ''}:")
                lines += [f"    {b:<{w}} {st}" for b, st in rows]
            pulls = [b for b in repo.branches() if repo.config.get(f"branch.{b}.remote") == name]
            if pulls:
                w = max(len(b) for b in pulls)
                lines.append(f"  Local branch{'es' if len(pulls) != 1 else ''} configured for 'git pull':")
                for b in pulls:
                    m = (repo.config.get(f"branch.{b}.merge") or "").replace("refs/heads/", "")
                    lines.append(f"    {b:<{w}} merges with remote {m}")
            pushes = [b for b in pulls if f"refs/heads/{b}" in fr]
            if pushes:
                db = repo.odb
                w = max(len(b) for b in pushes)
                w2 = max(len(b) for b in pushes)
                lines.append(f"  Local ref{'s' if len(pushes) != 1 else ''} configured for 'git push':")
                for b in pushes:
                    mine, theirs = repo.resolve_ref(f"refs/heads/{b}"), fr[f"refs/heads/{b}"]
                    st = "up to date" if mine == theirs else ("fast-forwardable" if db.is_ancestor(theirs, mine) else
                                                              "local out of date")
                    lines.append(f"    {b:<{w}} pushes to {b:<{w2}} ({st})")
            g.out("\n".join(lines) + "\n")
        return 0
    if sub == "prune":
        for name in rest or repo.remotes():
            url = repo.config.get(f"remote.{name}.url")
            far = yield from connect(g, url)
            fr = far.refs()
            gone = [r for r in repo.ref_names(f"refs/remotes/{name}/")
                    if not r.endswith("/HEAD") and f"refs/heads/{r[len(f'refs/remotes/{name}/'):]}" not in fr]
            if gone:
                g.out(f"Pruning {name}\nURL: {url}\n")
            for r in gone:
                repo.delete_ref(r)
                g.out(f" * [pruned] {r[len('refs/remotes/'):]}\n")
        return 0
    if sub == "set-head":
        o = parse(rest, "auto|a delete|d")
        name = o.args[0]
        if o.get("delete"):
            repo.remove(f"refs/remotes/{name}/HEAD", True)
            return 0
        if o.get("auto"):
            far = yield from connect(g, repo.config.get(f"remote.{name}.url"))
            head = far.head_symref()
            b = head[len("refs/heads/"):]
        else:
            b = o.args[1]
        repo.write(f"refs/remotes/{name}/HEAD", f"ref: refs/remotes/{name}/{b}\n", True)
        g.out(f"{name}/HEAD set to {b}\n")
        return 0
    raise Usage("usage: git remote [-v | --verbose]\n   or: git remote add [-t <branch>] [-m <master>] [-f] [--tags | "
                "--no-tags] [--mirror=<fetch|push>] <name> <url>\n   or: git remote rename [--[no-]progress] <old> "
                "<new>\n   or: git remote remove <name>\n   or: git remote set-head <name> (-a | --auto | -d | "
                "--delete | <branch>)\n   or: git remote [-v | --verbose] show [-n] <name>\n   or: git remote prune "
                "[-n | --dry-run] <name>\n   or: git remote set-url [--push] <name> <newurl> [<oldurl>]")


@command("ls-remote")
def git_ls_remote(g, args):
    o = parse(args, "heads|h tags|t refs quiet|q get-url symref exit-code sort=")
    repo = g.maybe_repo()
    target = o.args[0] if o.args else "origin"
    url = remote_url(repo, target) if repo else None
    url = url or target
    if o.get("get-url"):
        g.out(url + "\n")
        return 0
    far = yield from connect(g, url)
    refs = far.refs()
    out = []
    if not (o.get("heads") or o.get("tags")):
        head = far.head_symref()
        if head and head in refs:
            if o.get("symref"):
                out.append(f"ref: {head}\tHEAD")
            out.append(f"{refs[head]}\tHEAD")
    pats = o.args[1:]
    for name, sha in refs.items():
        if o.get("heads") and not name.startswith("refs/heads/"):
            continue
        if o.get("tags") and not name.startswith("refs/tags/"):
            continue
        if pats and not any(fnmatch.fnmatch(name, "*/" + p) or name.endswith("/" + p) or name == p for p in pats):
            continue
        out.append(f"{sha}\t{name}")
    if out:
        g.out("\n".join(out) + "\n")
    return 0


# ---------------------------------------------------------------------------- fetch

def _refspecs(repo, remote, given):
    """-> [(force, src pattern, dst pattern)]."""
    specs = given or repo.config.get_all(f"remote.{remote}.fetch") or ([f"+refs/heads/*:refs/remotes/{remote}/*"]
                                                                          if remote else [])
    out = []
    for s in specs:
        force = s.startswith("+")
        s = s.lstrip("+")
        src, _, dst = s.partition(":")
        if given and not src.startswith("refs/"):
            if src.startswith("tags/") or src.startswith("heads/"):
                src = "refs/" + src
            # a short name is looked up on the far side (a branch, else a tag): see _match
        if given and dst and not dst.startswith("refs/"):
            dst = f"refs/heads/{dst}"
        if given and not dst and remote and "*" not in src and not src.startswith("refs/tags/"):
            short = src[len("refs/heads/"):] if src.startswith("refs/heads/") else src
            dst = f"refs/remotes/{remote}/{short}"
        out.append((force, src, dst))
    return out


def _match(pattern, ref):
    if not pattern.startswith("refs/") and "*" not in pattern:
        return "" if ref in (f"refs/heads/{pattern}", f"refs/tags/{pattern}") else None
    if "*" in pattern:
        pre, _, post = pattern.partition("*")
        if ref.startswith(pre) and ref.endswith(post) and len(ref) >= len(pre) + len(post):
            return ref[len(pre):len(ref) - len(post) if post else None]
        return None
    return "" if ref == pattern else None


def do_fetch(g, repo, remote, url, given_specs=(), prune=None, tags=None, quiet=False, depth=None, show_from=True):
    """Fetch from one remote: update remote-tracking refs, FETCH_HEAD; print git's lines. Generator."""
    db = repo.odb
    far = yield from connect(g, url)
    frefs = far.refs()
    specs = _refspecs(repo, remote, list(given_specs))
    updates = []                  # (dst ref, new sha, src ref, force)
    fetch_head = []
    for force, src, dst in specs:
        for ref, sha in frefs.items():
            star = _match(src, ref)
            if star is None:
                continue
            d = dst.replace("*", star) if dst and "*" in dst else dst
            updates.append((d, sha, ref, force))
            fetch_head.append((sha, ref))
    have = [repo.peel(v) for v in repo.ref_names("refs/").values()] + ([repo.head()] if repo.head() else [])
    want = [sha for _, sha, _, _ in updates]
    # tags pointing into what we fetch (auto-follow), or all with --tags
    tag_updates = []
    if tags is not False:
        local_tags = repo.tags()
        reach = db.reachable([w for w in want if db.has(w)] + [h for h in have if db.has(h) and db.kind(h) == "commit"])
        for ref, sha in frefs.items():
            if not ref.startswith("refs/tags/") or ref.endswith("^{}"):
                continue
            name = ref[len("refs/tags/"):]
            if name in local_tags:
                continue
            if tags is True or repo.peel(sha) in reach:
                tag_updates.append((ref, sha, ref, False))
    new_objs = missing_objects(db, want + [repo.peel(s) for _, s, _, _ in tag_updates], have)
    n, size = object_stats(db, new_objs)
    lines = []
    width = max([10] + [len(_short_src(s)) for _, _, s, _ in updates + tag_updates])
    errors = []
    for d, sha, src, force in updates:
        if not d:
            continue
        old = repo.resolve_ref(d)
        short_dst = d[len("refs/remotes/"):] if d.startswith("refs/remotes/") else d.split("/", 2)[-1]
        if old == sha:
            continue
        if old is None:
            kind = "[new tag]" if d.startswith("refs/tags/") else "[new branch]" if src.startswith("refs/heads/") \
                else "[new ref]"
            lines.append(ref_line("*", kind, _short_src(src), short_dst, width))
            repo.update_ref(d, sha, f"fetch: storing head" if d.startswith("refs/remotes/") else None)
        elif db.is_ancestor(old, sha):
            lines.append(ref_line(" ", f"{db.abbrev(old)}..{db.abbrev(sha)}", _short_src(src), short_dst, width))
            repo.update_ref(d, sha, "fetch: fast-forward")
        elif force:
            lines.append(ref_line("+", f"{db.abbrev(old)}...{db.abbrev(sha)}", _short_src(src), short_dst, width,
                                  "forced update"))
            repo.update_ref(d, sha, "fetch: forced-update")
        else:
            lines.append(ref_line("!", "[rejected]", _short_src(src), short_dst, width, "non-fast-forward"))
            errors.append(d)
    for d, sha, src, _ in tag_updates:
        lines.append(ref_line("*", "[new tag]", _short_src(src), d[len("refs/tags/"):], width))
        repo.update_ref(d, sha, None)
    do_prune = prune if prune is not None else repo.config.get_bool(f"remote.{remote}.prune",
                                                                     repo.config.get_bool("fetch.prune", False))
    if do_prune and remote:
        for ref in list(repo.ref_names(f"refs/remotes/{remote}/")):
            b = ref[len(f"refs/remotes/{remote}/"):]
            if b == "HEAD":
                continue
            if f"refs/heads/{b}" not in frefs:
                lines.append(ref_line("-", "[deleted]", "(none)", f"{remote}/{b}", width))
                repo.delete_ref(ref)
    # FETCH_HEAD: the current branch's upstream first (for merge), the rest not-for-merge
    b = repo.head_branch()
    merge_ref = repo.config.get(f"branch.{b}.merge") if b and repo.config.get(f"branch.{b}.remote") == remote else None
    fh = []
    for sha, ref in fetch_head:
        kind = "branch" if ref.startswith("refs/heads/") else "tag" if ref.startswith("refs/tags/") else "ref"
        nm = ref.split("/", 2)[-1] if kind != "ref" else ref
        tag = "" if (merge_ref and ref == merge_ref) or (given_specs and len(fetch_head) == 1) else "not-for-merge"
        fh.append((tag != "", f"{sha}\t{tag}\t{kind} '{nm}' of {far.ep.display}"))
    fh.sort(key=lambda x: x[0])
    repo.write("FETCH_HEAD", "".join(x[1] + "\n" for x in fh))
    if not quiet:
        out = remote_progress(n, size) if n and stderr_is_tty(g.ctx) else ""
        if lines:
            out += (f"From {far.ep.display}\n" if show_from else "") + "\n".join(lines) + "\n"
        if out:
            g.err(out)
    return 1 if errors else 0


def _short_src(ref):
    for pre in ("refs/heads/", "refs/tags/"):
        if ref.startswith(pre):
            return ref[len(pre):]
    return ref


@command("fetch")
def git_fetch(g, args):
    o = parse(args, "all prune|p no-prune tags|t no-tags quiet|q verbose|v depth= unshallow dry-run|n force|f "
                    "multiple jobs|j= set-upstream recurse-submodules=? progress update-head-ok append|a")
    repo = g.repo()
    prune = True if o.get("prune") else (False if o.get("no-prune") else None)
    tags = True if o.get("tags") else (False if o.get("no-tags") else None)
    if o.get("all") or o.get("multiple"):
        names = repo.remotes() if o.get("all") else o.args
        st = 0
        for name in names:
            if not o.get("quiet") and len(names) > 1:
                g.out(f"Fetching {name}\n")
            st |= yield from do_fetch(g, repo, name, repo.config.get(f"remote.{name}.url"), prune=prune, tags=tags,
                                      quiet=bool(o.get("quiet")))
        return st
    if o.args:
        remote, url = remote_and_url(repo, o.args[0])
        specs = o.args[1:]
    else:
        b = repo.head_branch()
        remote = (repo.config.get(f"branch.{b}.remote") if b else None) or ("origin" if "origin" in repo.remotes()
                                                                            else None)
        if not remote:
            if not repo.remotes():
                raise GitError("fatal: No remote repository specified.  Please, specify either a URL or a\nremote name "
                               "from which new revisions should be fetched.")
            remote = repo.remotes()[0]
        url = repo.config.get(f"remote.{remote}.url")
        specs = []
    return (yield from do_fetch(g, repo, remote, url, specs, prune=prune, tags=tags, quiet=bool(o.get("quiet"))))


# ---------------------------------------------------------------------------- pull

@command("pull", wt=True)
def git_pull(g, args):
    o = parse(args, "rebase|r=? no-rebase ff no-ff ff-only quiet|q verbose|v all autostash no-autostash "
                    "squash no-commit commit edit|e no-edit allow-unrelated-histories strategy|s= "
                    "strategy-option|X= tags prune|p depth=")
    repo = g.repo(wt=True)
    db = repo.odb
    b = repo.head_branch()
    if o.args:
        remote, url = remote_and_url(repo, o.args[0])
        branches = o.args[1:]
    else:
        if not b:
            raise GitError("You are not currently on a branch.\nPlease specify which branch you want to merge with.\n"
                           "See git-pull(1) for details.\n\n    git pull <remote> <branch>\n", 1)
        up = repo.upstream(b)
        if not up:
            raise GitError("There is no tracking information for the current branch.\nPlease specify which branch you "
                           "want to merge with.\nSee git-pull(1) for details.\n\n    git pull <remote> <branch>\n\n"
                           "If you wish to set tracking information for this branch you can do so with:\n\n"
                           f"    git branch --set-upstream-to=origin/<branch> {b}\n", 1)
        remote = up[0]
        url = repo.config.get(f"remote.{remote}.url")
        branches = []
    st = yield from do_fetch(g, repo, remote, url, branches, prune=True if o.get("prune") else None,
                             quiet=bool(o.get("quiet")))
    if st:
        return st
    # what to integrate: FETCH_HEAD's merge candidate
    text = repo.read("FETCH_HEAD") or ""
    target = None
    for line in text.split("\n"):
        if line and "\tnot-for-merge\t" not in line:
            target = line.split("\t")[0]
            break
    if target is None:
        if branches:
            raise GitError(f"fatal: couldn't find remote ref {branches[0]}")
        raise GitError(f"Your configuration specifies to merge with the ref '{(repo.config.get(f'branch.{b}.merge') or '').replace('refs/heads/', '')}'\n"
                       "from the remote, but no such ref was fetched.", 1)
    head = repo.head()
    label = f"{remote}/{branches[0]}" if branches and remote else (
        (repo.upstream_ref(b) or "")[len("refs/remotes/"):] if b else "FETCH_HEAD")
    rebase = o.get("rebase")
    if rebase is None:
        cfg = repo.config.get(f"branch.{b}.rebase") or repo.config.get("pull.rebase")
        rebase = cfg not in (None, "false") if cfg else None
    elif rebase is not False and rebase not in ("false",):
        rebase = True
    if o.get("no-rebase"):
        rebase = False
    if head is None:
        from .switching import checkout_tree
        checkout_tree(g, repo, db.commit(target).tree)
        repo.update_ref("HEAD", target, "pull: initial commit")
        return 0
    if db.is_ancestor(target, head):
        g.out("Already up to date.\n")
        return 0
    ff_only = o.get("ff-only") or repo.config.get("pull.ff") == "only"
    can_ff = db.is_ancestor(head, target)
    if rebase:
        from .cmd_branch import git_rebase
        return (yield from run_maybe(git_rebase(g, [target] if not can_ff else [target])))
    if not can_ff and ff_only:
        raise GitError("fatal: Not possible to fast-forward, aborting.")
    if not can_ff and rebase is None and not repo.config.get("pull.ff") and not o.get("no-ff") and not o.get("ff"):
        raise GitError("hint: You have divergent branches and need to specify how to reconcile them.\n"
                       "hint: You can do so by running one of the following commands sometime before\n"
                       "hint: your next pull:\nhint: \nhint:   git config pull.rebase false  # merge\n"
                       "hint:   git config pull.rebase true   # rebase\nhint:   git config pull.ff only       # "
                       "fast-forward only\nhint: \nhint: You can replace \"git config\" with \"git config --global\" to "
                       "set a default\nhint: preference for all repositories. You can also pass --rebase, --no-rebase,\n"
                       "hint: or --ff-only on the command line to override the configured default per\nhint: "
                       "invocation.\nfatal: Need to specify how to reconcile divergent branches.")
    from .cmd_branch import git_merge
    margs = [target]
    if o.get("no-ff"):
        margs.insert(0, "--no-ff")
    if o.get("no-edit") or not g.ctx.stdin_is_tty():
        margs.insert(0, "--no-edit")
    if o.get("squash"):
        margs.insert(0, "--squash")
    branch_name = branches[0] if branches else (repo.config.get(f"branch.{b}.merge") or "").replace("refs/heads/", "")
    margs = ["-m", f"Merge branch '{branch_name}' of {_display(url, g)}" + (f" into {b}" if b not in ("main", "master")
                                                                          else "")] + margs if not can_ff else margs
    return (yield from run_maybe(git_merge(g, margs)))


def _display(url, g):
    return parse_url(g.ctx, url).display


# ---------------------------------------------------------------------------- push

@command("push")
def git_push(g, args):
    o = parse(args, "set-upstream|u force|f force-with-lease=? force-if-includes delete|d tags all|branches mirror "
                    "dry-run|n quiet|q verbose|v porcelain follow-tags no-verify atomic prune "
                    "push-option|o= recurse-submodules= no-force-with-lease")
    repo = g.repo()
    db = repo.odb
    b = repo.head_branch()
    if o.args:
        remote, url = remote_and_url(repo, o.args[0])
        specs = o.args[1:]
    else:
        up = repo.upstream(b) if b else None
        remote = (up[0] if up else None) or repo.config.get(f"branch.{b}.pushRemote") or \
            repo.config.get("remote.pushDefault") or ("origin" if "origin" in repo.remotes() else None)
        if not remote:
            raise GitError("fatal: No configured push destination.\nEither specify the URL from the command-line or "
                           "configure a remote repository using\n\n    git remote add <name> <url>\n\nand then push "
                           "using the remote name\n\n    git push <name>\n")
        url = repo.config.get(f"remote.{remote}.pushurl") or repo.config.get(f"remote.{remote}.url")
        specs = []
    # what to push: (local ref or None for delete, remote ref, force)
    pairs = []
    if o.get("tags"):
        for name, sha in repo.tags().items():
            pairs.append((f"refs/tags/{name}", f"refs/tags/{name}", False))
    if o.get("all"):
        for name in repo.branches():
            pairs.append((f"refs/heads/{name}", f"refs/heads/{name}", False))
    for s in specs:
        force = s.startswith("+")
        s = s.lstrip("+")
        if o.get("delete"):
            pairs.append((None, _remote_ref(repo, s), False))
            continue
        src, colon, dst = s.partition(":")
        if colon and not src:
            pairs.append((None, _remote_ref(repo, dst), False))
            continue
        full, sha = dwim_ref(repo, src) if src != "HEAD" else (repo.head_ref() or "HEAD", repo.head())
        if sha is None:
            if try_resolve(repo, src) is not None and colon:
                full = src
            else:
                raise GitError(f"error: src refspec {src} does not match any\nerror: failed to push some refs to "
                               f"'{url}'", 1)
        if not colon:
            dst = full if full.startswith("refs/") and full != "HEAD" else f"refs/heads/{b}"
        elif not dst.startswith("refs/"):
            dst = f"refs/tags/{dst}" if full.startswith("refs/tags/") else f"refs/heads/{dst}"
        pairs.append((full if full != "HEAD" else repo.head_ref(), dst, force))
    if not specs and not o.get("tags") and not o.get("all"):
        if not b:
            raise GitError("fatal: You are not currently on a branch.\nTo push the history leading to the current "
                           "(detached HEAD)\nstate now, use\n\n    git push origin HEAD:<name-of-remote-branch>\n")
        up = repo.upstream(b)
        mode = repo.config.get("push.default", "simple")
        if up and up[0] == remote:
            dst = up[1]
            if mode == "simple" and dst != f"refs/heads/{b}":
                raise GitError("fatal: The upstream branch of your current branch does not match\nthe name of your "
                               "current branch.  To push to the upstream branch\non the remote, use\n\n    git push "
                               f"{remote} HEAD:{dst.replace('refs/heads/', '')}\n\nTo push to the branch of the same name on "
                               f"the remote, use\n\n    git push {remote} HEAD\n")
            pairs.append((f"refs/heads/{b}", dst, False))
        elif o.get("set-upstream") or repo.config.get_bool("push.autoSetupRemote", False):
            pairs.append((f"refs/heads/{b}", f"refs/heads/{b}", False))
            o.flags["set-upstream"] = True
        else:
            raise GitError(f"fatal: The current branch {b} has no upstream branch.\nTo push the current branch and set "
                           f"the remote as upstream, use\n\n    git push --set-upstream {remote} {b}\n\nTo have this "
                           "happen automatically for branches without a tracking\nupstream, see 'push.autoSetupRemote' "
                           "in 'git help config'.\n")
    if o.get("follow-tags"):
        for name, sha in repo.tags().items():
            if db.kind(sha) == "tag":
                pairs.append((f"refs/tags/{name}", f"refs/tags/{name}", False))
    far = yield from connect(g, url, push=True)
    frefs = far.refs()
    lease = o.get("force-with-lease")
    lines, rejected, hints = [], [], set()
    pushed_objs = set()
    pushed_tips = []
    large_files = set()
    new_branch_pr = None
    width = 0                          # push doesn't pad the source column
    for src, dst, force in pairs:
        force = force or bool(o.get("force"))
        if src is None and dst not in frefs and dst.startswith("refs/heads/"):
            alt = "refs/tags/" + dst[len("refs/heads/"):]       # `push --delete NAME`: whatever NAME is over there
            if alt in frefs:
                dst = alt
        old = frefs.get(dst)
        short_dst = _short_src(dst)
        if src is None:
            if old is None:
                lines.append(ref_line("!", "[remote rejected]", "(delete)", short_dst, width, "remote ref does not exist"))
                rejected.append(dst)
                continue
            why = None if o.get("dry-run") else far.update(dst, None, old, True)
            if why:
                lines.append(ref_line("!", "[remote rejected]", "(delete)", short_dst, width, _why(why)))
                rejected.append(dst)
                continue
            lines.append(f" - {'[deleted]':<17} {short_dst}")
            if remote:
                repo.delete_ref(f"refs/remotes/{remote}/{short_dst}")
            continue
        new = repo.resolve_ref(src) if src.startswith("refs/") else try_resolve(repo, src)
        if old == new:
            continue
        if lease is not None and old is not None:
            expect = repo.resolve_ref(f"refs/remotes/{remote}/{short_dst}") if lease is True else \
                try_resolve(repo, lease.split(":", 1)[-1])
            if expect != old:
                lines.append(ref_line("!", "[rejected]", _short_src(src), short_dst, width, "stale info"))
                rejected.append(dst)
                continue
            force = True
        if old is not None and not force and db.kind(new) == "commit" and not db.is_ancestor(old, new):
            known = old in db.reachable([v for v in repo.ref_names("refs/").values() if db.kind(repo.peel(v)) == "commit"])
            note = "non-fast-forward" if known else "fetch first"
            if dst.startswith("refs/tags/"):
                note = "already exists"
            lines.append(ref_line("!", "[rejected]", _short_src(src), short_dst, width, note))
            rejected.append(dst)
            hints.add(note)
            continue
        objs = missing_objects(db, [repo.peel(new)], [repo.peel(v) for v in frefs.values() if db.has(v)])
        why = None
        if far.ghrepo is not None:
            big = _github_big_files(db, objs, new)
            if big:
                why = "large"
                large_files |= big
        if why is None:
            why = None if o.get("dry-run") else far.update(dst, new, old, force)
        if why:
            lines.append(ref_line("!", "[remote rejected]", _short_src(src), short_dst, width, _why(why)))
            rejected.append(dst)
            if why.startswith("protected"):
                hints.add(why)
            if why == "permission":
                raise GitError(_permission_error(far, url))
            continue
        pushed_objs |= objs
        pushed_tips.append(new)
        if old is None:
            kind = "[new tag]" if dst.startswith("refs/tags/") else "[new branch]"
            lines.append(ref_line("*", kind, _short_src(src), short_dst, width))
            if far.ghrepo is not None and dst.startswith("refs/heads/") and dst != f"refs/heads/{far.ghrepo.default_branch}":
                new_branch_pr = short_dst
        elif force and not db.is_ancestor(old, new):
            lines.append(ref_line("+", f"{db.abbrev(old)}...{db.abbrev(new)}", _short_src(src), short_dst, width,
                                  "forced update"))
        else:
            lines.append(ref_line(" ", f"{db.abbrev(old)}..{db.abbrev(new)}", _short_src(src), short_dst, width))
        if remote and dst.startswith("refs/heads/") and not o.get("dry-run"):
            repo.update_ref(f"refs/remotes/{remote}/{short_dst}", new, "update by push")
        if o.get("set-upstream") and src.startswith("refs/heads/") and dst.startswith("refs/heads/") and remote:
            from .switching import set_upstream
            set_upstream(repo, src[len("refs/heads/"):], remote, dst)
            lines.append(f"__UP__branch '{src[len('refs/heads/'):]}' set up to track '{remote}/{short_dst}'.")
    if not lines:
        g.err("Everything up-to-date\n")
        return 0
    out = ""
    if far.ghrepo is not None and pushed_objs and not o.get("dry-run"):
        from .tools import lfs_upload
        out += lfs_upload(g, repo, far.ghrepo, [repo.peel(n_) for n_ in pushed_tips], list(frefs.values()))
    if not o.get("quiet"):
        n, size = object_stats(db, pushed_objs)
        if stderr_is_tty(g.ctx):
            out += push_progress(n, size)
        if new_branch_pr and far.ghrepo is not None:
            out += (f"remote: \nremote: Create a pull request for '{new_branch_pr}' on GitHub by visiting:\n"
                    f"remote:      https://github.com/{far.ghrepo.full}/pull/new/{new_branch_pr}\nremote: \n")
    if large_files:
        n, size = object_stats(db, pushed_objs)
        if stderr_is_tty(g.ctx) and not o.get("quiet"):
            out = out if "Enumerating" in out else out + push_progress(n, size)
        out += "remote: error: Trace: " + hashlib.sha1(repr(sorted(large_files)).encode()).hexdigest()[:32] + "\n"
        out += "remote: error: See https://gh.io/lfs for more information.\n"
        for path, nbytes in sorted(large_files):
            out += (f"remote: error: File {path} is {nbytes / 1024 / 1024:.2f} MB; this exceeds GitHub's file size "
                    f"limit of 100.00 MB\n")
        out += ("remote: error: GH001: Large files detected. You may want to try Git Large File Storage - "
                "https://git-lfs.github.com.\n")
    for why in sorted(hints):
        if why.startswith("protected"):
            out += (f"remote: error: GH006: Protected branch update failed for refs/heads/"
                    f"{far.ghrepo.default_branch if far.ghrepo else ''}.\nremote: error: Changes must be made through a "
                    "pull request.\n") if why == "protected:pr" else "remote: error: GH006: Protected branch update failed.\n"
    shown = [x for x in lines if not x.startswith("__UP__")]
    if not o.get("quiet") or rejected:
        out += f"To {far.ep.display_to}\n" + "\n".join(shown) + "\n"
    g.err(out)
    for x in lines:
        if x.startswith("__UP__"):
            g.out(x[6:] + "\n")
    if rejected:
        msg = f"error: failed to push some refs to '{url}'\n"
        if "fetch first" in hints:
            msg += ("hint: Updates were rejected because the remote contains work that you do\n"
                    "hint: not have locally. This is usually caused by another repository pushing\n"
                    "hint: to the same ref. You may want to first integrate the remote changes\n"
                    "hint: (e.g., 'git pull ...') before pushing again.\n"
                    "hint: See the 'Note about fast-forwards' in 'git push --help' for details.\n")
        elif "non-fast-forward" in hints:
            msg += ("hint: Updates were rejected because the tip of your current branch is behind\n"
                    "hint: its remote counterpart. Integrate the remote changes (e.g.\n"
                    "hint: 'git pull ...') before pushing again.\n"
                    "hint: See the 'Note about fast-forwards' in 'git push --help' for details.\n")
        g.err(msg)
        return 1
    return 0


def _remote_ref(repo, name):
    if name.startswith("refs/"):
        return name
    if repo.resolve_ref(f"refs/tags/{name}") and not repo.resolve_ref(f"refs/heads/{name}"):
        return f"refs/tags/{name}"
    return f"refs/heads/{name}"


def _why(why):
    return {"protected:pr": "protected branch hook declined", "protected:force": "protected branch hook declined",
            "protected:delete": "protected branch hook declined",
            "default:delete": "refusing to delete the current branch: refs/heads/main",
            "branch is currently checked out": "branch is currently checked out",
            "tag:exists": "already exists", "large": "pre-receive hook declined"}.get(why, why)


def _github_big_files(db, objs, tip) -> set:
    """{(path, bytes)} of files over GitHub's 100 MB limit among the blobs a push sends."""
    out = set()
    big = {s for s in objs if db.kind(s) == "blob" and db.size(s) > 100 * 1024 * 1024}
    if not big:
        return out
    for c in db.ancestors([tip]):
        for p, (mode, sha) in db.flat(db.commit(c).tree).items():
            if sha in big:
                out.add((p, db.size(sha)))
    return out


def _permission_error(far, url):
    who = far.account.login if far.account else "anonymous"
    if far.ep.protocol == "https":
        return (f"remote: Permission to {far.ghrepo.full}.git denied to {who}.\n"
                f"fatal: unable to access '{url.rstrip('/')}/': The requested URL returned error: 403")
    return (f"ERROR: Permission to {far.ghrepo.full}.git denied to {who}.\nfatal: Could not read from remote "
            "repository.\n\nPlease make sure you have the correct access rights\nand the repository exists.")

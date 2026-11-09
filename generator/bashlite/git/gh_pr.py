"""gh pr, gh issue, gh run, gh workflow, gh release."""

from __future__ import annotations

from .cli import parse
from .gh_core import GhError, ask, clone_url, current_repo, gh_command, need_login
from .objects import odb


def _ago(now, t):
    d = max(0, now - t)
    if d < 60:
        return "less than a minute ago"
    if d < 3600:
        m = d // 60
        return f"about {m} minute{'s' if m != 1 else ''} ago"
    if d < 86400:
        h = d // 3600
        return f"about {h} hour{'s' if h != 1 else ''} ago"
    days = d // 86400
    return f"about {days} day{'s' if days != 1 else ''} ago"


def _pr_arg(g, gh, r, o):
    """The PR named by a number/URL/branch argument, or the current branch's PR."""
    if o.args:
        a = o.args[0].rstrip("/").split("/")[-1].lstrip("#")
        if a.isdigit():
            pr = gh.find_pr(r, int(a))
            if pr is None:
                raise GhError(f"GraphQL: Could not resolve to a PullRequest with the number of {a}. (repository."
                              "pullRequest)")
            return pr
        for pr in gh.pulls_of(r):
            if pr.head_branch == a:
                return pr
        raise GhError(f"no pull requests found for branch \"{a}\"")
    repo = g.maybe_repo()
    b = repo.head_branch() if repo else None
    for pr in reversed(gh.pulls_of(r)):
        if pr.head_branch == b and pr.state == "OPEN":
            return pr
    raise GhError(f"no pull requests found for branch \"{b}\"")


# ---------------------------------------------------------------------------- pr

@gh_command("pr create")
def gh_pr_create(g, gh, args):
    o = parse(args, "title|t= body|b= body-file|F= base|B= head|H= fill|f fill-first draft|d web|w reviewer|r= "
                    "assignee|a= label|l= repo|R= no-maintainer-edit")
    acc = need_login(gh)
    base_repo = current_repo(g, gh, o.get("repo"))
    db = odb(g.world)
    repo = g.maybe_repo()
    head = o.get("head")
    head_repo = base_repo
    if head and ":" in head:
        owner, head = head.split(":", 1)
        head_repo = gh.get(owner, base_repo.name) or base_repo
    if not head:
        b = repo.head_branch() if repo else None
        if not b:
            raise GhError("could not determine the current branch: not on any branch")
        head = b
        # the branch must be on GitHub (pushed); find which remote repo has it
        cands = []
        for n in repo.remotes():
            from .remote import parse_url
            ep = parse_url(g.ctx, repo.config.get(f"remote.{n}.url") or "")
            if ep.kind == "github":
                rr = gh.get(ep.owner, ep.repo)
                if rr is not None and f"refs/heads/{b}" in rr.refs:
                    cands.append(rr)
        if not cands:
            raise GhError(f"aborted: you must first push the current branch to a remote, or use the --head flag")
        head_repo = next((c for c in cands if c.owner == acc.login), cands[0])
    if head_repo.fork_of and o.get("repo") is None:
        parent = gh.get(*head_repo.fork_of.split("/"))
        if parent is not None:
            base_repo = parent
    base = o.get("base") or base_repo.default_branch
    title, body = o.get("title"), o.get("body")
    if o.get("body-file"):
        import posixpath
        f = o.get("body-file")
        body = bytes(g.fs.lookup(f if f.startswith("/") else posixpath.join(g.cwd, f)).data).decode()
    sha = head_repo.refs.get(f"refs/heads/{head}")
    if sha is None:
        raise GhError(f"aborted: you must first push the current branch to a remote, or use the --head flag")
    base_sha = base_repo.refs.get(f"refs/heads/{base}")
    commits = list(reversed(db.ancestors([sha], [base_sha] if base_sha else [])))
    if o.get("fill") or o.get("fill-first"):
        if len(commits) == 1 or o.get("fill-first"):
            c = db.commit(commits[0]) if commits else None
            title = title or (c.subject if c else head)
            body = body if body is not None else (c.body.strip() if c else "")
        else:
            title = title or head.replace("-", " ").replace("/", " ").capitalize()
            body = body if body is not None else "".join(f"- {db.commit(c).subject}\n" for c in commits).rstrip()
    if title is None:
        title = yield from ask(g, "Title", db.commit(commits[0]).subject if commits else head)
    if body is None:
        body = yield from ask(g, "Body", "")
    head_label = head if head_repo is base_repo else f"{head_repo.owner}:{head}"
    if o.get("web"):
        g.out(f"Opening github.com/{base_repo.full}/compare/{base}...{head_label} in your browser.\n")
        return 0
    g.err(f"\nCreating {'draft ' if o.get('draft') else ''}pull request for {head_label} into {base} in "
          f"{base_repo.full}\n\n")
    pr = gh.open_pr(base_repo, head_repo, head, base, title, body or "", acc.login, bool(o.get("draft")))
    pr.labels = list(o.all("label"))
    g.out(f"https://github.com/{base_repo.full}/pull/{pr.number}\n")
    return 0


@gh_command("pr list")
def gh_pr_list(g, gh, args):
    o = parse(args, "state|s= limit|L= author|A= base|B= head|H= label|l= draft repo|R= web|w json= search|S=")
    r = current_repo(g, gh, o.get("repo"))
    state = (o.get("state") or "open").upper()
    prs = [p for p in gh.pulls_of(r) if state == "ALL" or p.state == state or (state == "CLOSED" and p.state == "MERGED")]
    if o.get("author"):
        prs = [p for p in prs if p.author == o.get("author")]
    if o.get("base"):
        prs = [p for p in prs if p.base == o.get("base")]
    prs = sorted(prs, key=lambda p: -p.number)
    if not prs:
        g.err(f"no {state.lower() if state != 'ALL' else ''} pull requests match your search in {r.full}\n".replace("  ", " "))
        return 0
    now = gh.now()
    kind = {"OPEN": "open", "CLOSED": "closed", "MERGED": "merged", "ALL": ""}[state]
    g.out(f"\nShowing {len(prs)} of {len(prs)} {kind + ' ' if kind else ''}pull request{'s' if len(prs) != 1 else ''} "
          f"in {r.full}\n\n")
    w1 = max(len(f"#{p.number}") for p in prs)
    w2 = max(len(p.title) for p in prs)
    w3 = max(len(p.head_branch) for p in prs)
    g.out(f"{'ID':<{w1}}  {'TITLE':<{w2}}  {'BRANCH':<{w3}}  CREATED AT\n")
    for p in prs:
        g.out(f"{'#' + str(p.number):<{w1}}  {p.title:<{w2}}  {p.head_branch:<{w3}}  {_ago(now, p.created)}\n")
    return 0


@gh_command("pr view")
def gh_pr_view(g, gh, args):
    o = parse(args, "web|w comments|c repo|R= json=")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    if o.get("web"):
        g.out(f"Opening github.com/{r.full}/pull/{pr.number} in your browser.\n")
        return 0
    db = odb(g.world)
    head_repo = gh.get(*pr.head_repo.split("/"))
    base_sha = r.refs.get(f"refs/heads/{pr.base}")
    commits = db.ancestors([pr.head_sha], [base_sha] if base_sha and pr.state == "OPEN" else
                           ([db.commit(pr.merge_commit).parents[0]] if pr.merge_commit else []))
    add = rem = 0
    if base_sha and pr.head_sha:
        from .diff import compare, numstat
        bases = db.merge_bases(base_sha, pr.head_sha)
        a = db.flat(db.commit(bases[0]).tree) if bases else {}
        for ch in compare(a, db.flat(db.commit(pr.head_sha).tree)):
            x, y = numstat(ch, db)
            add += x or 0
            rem += y or 0
    state = {"OPEN": "Draft" if pr.draft else "Open", "CLOSED": "Closed", "MERGED": "Merged"}[pr.state]
    verb = "wants to merge" if pr.state == "OPEN" else ("merged" if pr.state == "MERGED" else "wanted to merge")
    n = len(commits)
    reviews = ""
    if pr.reviews:
        reviews = "Reviewers: " + ", ".join(f"{u} ({st.replace('_', ' ').title()})" for u, st, _ in pr.reviews) + "\n"
    checks = "No checks"
    runs = [x for x in r.runs if x.sha == pr.head_sha]
    if runs:
        st = [x.status(gh.now()) for x in runs]
        if all(s == "completed" and c == "success" for s, c in st):
            checks = "Checks passing"
        elif any(c == "failure" for _, c in st):
            checks = "Some checks were not successful"
        else:
            checks = "Checks pending"
    body = pr.body.strip() or "No description provided"
    g.out(f"{pr.title} {r.full}#{pr.number}\n{state} • {pr.author} {verb} {n} commit{'s' if n != 1 else ''} into "
          f"{pr.base} from {pr.head_branch} • {_ago(gh.now(), pr.created)}\n+{add} -{rem} • {checks}\n{reviews}\n"
          + "\n".join("  " + ln for ln in body.split("\n")) + "\n\n"
          + (f"\n{len(pr.comments)} comment{'s' if len(pr.comments) != 1 else ''}\n" if pr.comments else "")
          + f"View this pull request on GitHub: https://github.com/{r.full}/pull/{pr.number}\n")
    return 0


@gh_command("pr checkout")
def gh_pr_checkout(g, gh, args):
    o = parse(args, "branch|b= force|f repo|R= detach")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    repo = g.repo()
    name = o.get("branch") or pr.head_branch
    remote = None
    for n in repo.remotes():
        from .remote import parse_url
        ep = parse_url(g.ctx, repo.config.get(f"remote.{n}.url") or "")
        if ep.kind == "github" and f"{ep.owner}/{ep.repo}".lower() == r.full.lower():
            remote = n
    remote = remote or "origin"
    if repo.resolve_ref(f"refs/heads/{name}") is None:
        st = yield from g.ctx.run(["git", "fetch", remote, f"refs/pull/{pr.number}/head:{name}"])
        if st:
            return st
        st = yield from g.ctx.run(["git", "switch", name])
        if st == 0 and pr.head_repo == r.full:
            yield from g.ctx.run(["git", "branch", "-q", f"--set-upstream-to={remote}/{pr.head_branch}", name]) \
                if repo.resolve_ref(f"refs/remotes/{remote}/{pr.head_branch}") else _noop()
        return st
    st = yield from g.ctx.run(["git", "switch", name])
    if st:
        return st
    yield from g.ctx.run(["git", "fetch", "-q", remote, f"refs/pull/{pr.number}/head"])
    return (yield from g.ctx.run(["git", "merge", "--ff-only", "FETCH_HEAD"]))


def _noop():
    return 0
    yield  # pragma: no cover


@gh_command("pr diff")
def gh_pr_diff(g, gh, args):
    o = parse(args, "name-only patch color= repo|R= web")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    db = odb(g.world)
    from .diff import Colors, compare, patch
    base_sha = r.refs.get(f"refs/heads/{pr.base}")
    bases = db.merge_bases(base_sha, pr.head_sha) if base_sha else []
    a = db.flat(db.commit(bases[0]).tree) if bases else {}
    changes = compare(a, db.flat(db.commit(pr.head_sha).tree), renames=True, db=db)
    if o.get("name-only"):
        g.out("".join(c.path + "\n" for c in changes))
        return 0
    color = Colors(g.ctx.stdout_is_tty() and o.get("color") != "never")
    g.out("".join(patch(c, db, color=color) for c in changes))
    return 0


@gh_command("pr review")
def gh_pr_review(g, gh, args):
    o = parse(args, "approve|a comment|c request-changes|r body|b= body-file|F= repo|R=")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    if o.get("approve"):
        if pr.author == acc.login:
            raise GhError("failed to create review: GraphQL: Can not approve your own pull request (addPullRequestReview)")
        pr.reviews.append((acc.login, "APPROVED", o.get("body") or ""))
        g.err(f"✓ Approved pull request {r.full}#{pr.number}\n")
    elif o.get("request-changes"):
        if pr.author == acc.login:
            raise GhError("failed to create review: GraphQL: Can not request changes on your own pull request "
                          "(addPullRequestReview)")
        if not o.get("body"):
            raise GhError("body cannot be blank for request-changes review")
        pr.reviews.append((acc.login, "CHANGES_REQUESTED", o.get("body")))
        g.err(f"+ Requested changes to pull request {r.full}#{pr.number}\n")
    else:
        if not o.get("body"):
            raise GhError("body cannot be blank for comment review")
        pr.reviews.append((acc.login, "COMMENTED", o.get("body")))
        g.err(f"- Reviewed pull request {r.full}#{pr.number}\n")
    return 0


@gh_command("pr comment")
def gh_pr_comment(g, gh, args):
    o = parse(args, "body|b= body-file|F= repo|R= web")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    pr.comments.append((acc.login, o.get("body") or ""))
    g.out(f"https://github.com/{r.full}/pull/{pr.number}#issuecomment-{1000 + len(pr.comments)}\n")
    return 0


@gh_command("pr merge")
def gh_pr_merge(g, gh, args):
    o = parse(args, "merge|m squash|s rebase|r delete-branch|d auto admin subject|t= body|b= repo|R= "
                    "disable-auto match-head-commit=")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    if not r.can_push(acc.login):
        raise GhError(f"GraphQL: {acc.login} does not have the correct permissions to execute `MergePullRequest` "
                      "(mergePullRequest)")
    method = "squash" if o.get("squash") else "rebase" if o.get("rebase") else "merge" if o.get("merge") else None
    if method is None:
        ans = yield from ask(g, "What merge method would you like to use? [merge/squash/rebase]", "merge")
        method = {"c": "merge", "m": "merge", "s": "squash", "r": "rebase"}.get(ans[:1].lower(), "merge")
    if o.get("admin"):
        saved = r.protected.get(pr.base)
        r.protected.pop(pr.base, None)
    try:
        gh.merge_pr(r, pr, method, acc.login, o.get("subject"), o.get("body"))
    finally:
        if o.get("admin") and saved is not None:
            r.protected[pr.base] = saved
    verb = {"merge": "Merged", "squash": "Squashed and merged", "rebase": "Rebased and merged"}[method]
    g.err(f"✓ {verb} pull request {r.full}#{pr.number} ({pr.title})\n")
    if o.get("delete-branch"):
        repo = g.maybe_repo()
        if repo is not None and repo.worktree:
            if repo.head_branch() == pr.head_branch:
                st = yield from g.ctx.run(["git", "switch", "-q", pr.base])
                if st == 0:
                    yield from g.ctx.run(["git", "pull", "-q", "--ff-only"])
                    yield from g.ctx.run(["git", "branch", "-q", "-D", pr.head_branch])
                    g.err(f"✓ Deleted local branch {pr.head_branch} and switched to branch {pr.base}\n")
            elif repo.resolve_ref(f"refs/heads/{pr.head_branch}"):
                yield from g.ctx.run(["git", "branch", "-q", "-D", pr.head_branch])
                g.err(f"✓ Deleted local branch {pr.head_branch}\n")
        hr = gh.get(*pr.head_repo.split("/"))
        if hr is not None and f"refs/heads/{pr.head_branch}" in hr.refs and pr.head_branch != hr.default_branch:
            del hr.refs[f"refs/heads/{pr.head_branch}"]
            g.err(f"✓ Deleted remote branch {pr.head_branch}\n")
    return 0


@gh_command("pr close")
def gh_pr_close(g, gh, args):
    o = parse(args, "comment|c= delete-branch|d repo|R=")
    need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    if pr.state != "OPEN":
        g.err(f"! Pull request {r.full}#{pr.number} ({pr.title}) is already {pr.state.lower()}\n")
        return 0
    pr.state = "CLOSED"
    g.err(f"✓ Closed pull request {r.full}#{pr.number} ({pr.title})\n")
    return 0


@gh_command("pr reopen")
def gh_pr_reopen(g, gh, args):
    o = parse(args, "comment|c= repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    pr.state = "OPEN"
    g.err(f"✓ Reopened pull request {r.full}#{pr.number} ({pr.title})\n")
    return 0


@gh_command("pr ready")
def gh_pr_ready(g, gh, args):
    o = parse(args, "undo repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    pr.draft = bool(o.get("undo"))
    g.err(f"✓ Pull request {r.full}#{pr.number} is marked as {'draft' if pr.draft else 'ready for review'}\n")
    return 0


@gh_command("pr status")
def gh_pr_status(g, gh, args):
    acc = need_login(gh)
    r = current_repo(g, gh, None)
    repo = g.maybe_repo()
    b = repo.head_branch() if repo else None
    mine = [p for p in gh.pulls_of(r) if p.author == acc.login and p.state == "OPEN"]
    cur = [p for p in gh.pulls_of(r) if p.head_branch == b]
    out = [f"\nRelevant pull requests in {r.full}\n", "Current branch"]
    out += [f"  #{p.number}  {p.title} [{p.head_branch}]" for p in cur] or [f"  There is no pull request associated "
                                                                          f"with [{b}]"]
    out += ["", "Created by you"]
    out += [f"  #{p.number}  {p.title} [{p.head_branch}]" for p in mine] or ["  You have no open pull requests"]
    g.out("\n".join(out) + "\n\n")
    return 0


@gh_command("pr checks")
def gh_pr_checks(g, gh, args):
    o = parse(args, "watch repo|R= required")
    r = current_repo(g, gh, o.get("repo"))
    pr = _pr_arg(g, gh, r, o)
    runs = [x for x in r.runs if x.sha == pr.head_sha]
    if not runs:
        g.err(f"no checks reported on the '{pr.head_branch}' branch\n")
        return 1
    st = 0
    for x in runs:
        s, c = x.status(gh.now())
        word = c if s == "completed" else "pending"
        st = st or (1 if c == "failure" else 8 if s != "completed" else 0)
        g.out(f"{x.workflow}\t{word}\t{x.duration}s\thttps://github.com/{r.full}/actions/runs/{x.id}\n")
    return st


# ---------------------------------------------------------------------------- issue

@gh_command("issue create")
def gh_issue_create(g, gh, args):
    o = parse(args, "title|t= body|b= label|l= assignee|a= repo|R= web|w body-file|F=")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    title = o.get("title")
    if title is None:
        title = yield from ask(g, "Title")
    body = o.get("body")
    if body is None:
        body = yield from ask(g, "Body", "")
    g.err(f"\nCreating issue in {r.full}\n\n")
    from .github import Issue
    iss = Issue(gh.next_number(r), title, body, acc.login, gh.now(), o.all("label"))
    iss.assignees = [x if x != "@me" else acc.login for x in o.all("assignee")]
    gh.issues_of(r).append(iss)
    g.out(f"https://github.com/{r.full}/issues/{iss.number}\n")
    return 0


def _issue_arg(gh, r, o):
    if not o.args:
        raise GhError("issue number or URL required")
    a = o.args[0].rstrip("/").split("/")[-1].lstrip("#")
    for iss in gh.issues_of(r):
        if str(iss.number) == a:
            return iss
    raise GhError(f"GraphQL: Could not resolve to an issue or pull request with the number of {a}. (repository.issue)")


@gh_command("issue list")
def gh_issue_list(g, gh, args):
    o = parse(args, "state|s= label|l= assignee|a= author|A= limit|L= repo|R= search|S= json=")
    r = current_repo(g, gh, o.get("repo"))
    state = (o.get("state") or "open").upper()
    items = [i for i in gh.issues_of(r) if state == "ALL" or i.state == state]
    if o.get("label"):
        items = [i for i in items if o.get("label") in i.labels]
    items.sort(key=lambda i: -i.number)
    if not items:
        g.err(f"no {'open ' if state == 'OPEN' else ''}issues match your search in {r.full}\n")
        return 0
    now = gh.now()
    g.out(f"\nShowing {len(items)} of {len(items)} {'open ' if state == 'OPEN' else ''}issue{'s' if len(items) != 1 else ''}"
          f" in {r.full}\n\n")
    w1 = max(len(f"#{i.number}") for i in items)
    w2 = max(len(i.title) for i in items)
    w3 = max([len("LABELS")] + [len(", ".join(i.labels)) for i in items])
    g.out(f"{'ID':<{w1}}  {'TITLE':<{w2}}  {'LABELS':<{w3}}  UPDATED\n")
    for i in items:
        g.out(f"{'#' + str(i.number):<{w1}}  {i.title:<{w2}}  {', '.join(i.labels):<{w3}}  {_ago(now, i.created)}\n")
    return 0


@gh_command("issue view")
def gh_issue_view(g, gh, args):
    o = parse(args, "web|w comments|c repo|R= json=")
    r = current_repo(g, gh, o.get("repo"))
    iss = _issue_arg(gh, r, o)
    if o.get("web"):
        g.out(f"Opening github.com/{r.full}/issues/{iss.number} in your browser.\n")
        return 0
    g.out(f"{iss.title} {r.full}#{iss.number}\n{iss.state.title()} • {iss.author} opened {_ago(gh.now(), iss.created)} • "
          f"{len(iss.comments)} comment{'s' if len(iss.comments) != 1 else ''}\n" +
          (f"Labels: {', '.join(iss.labels)}\n" if iss.labels else "") + "\n" +
          "\n".join("  " + ln for ln in (iss.body.strip() or "No description provided").split("\n")) +
          f"\n\n\nView this issue on GitHub: https://github.com/{r.full}/issues/{iss.number}\n")
    return 0


@gh_command("issue close")
def gh_issue_close(g, gh, args):
    o = parse(args, "comment|c= reason|r= repo|R=")
    need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    iss = _issue_arg(gh, r, o)
    iss.state = "CLOSED"
    g.err(f"✓ Closed issue {r.full}#{iss.number} ({iss.title})\n")
    return 0


@gh_command("issue reopen")
def gh_issue_reopen(g, gh, args):
    o = parse(args, "comment|c= repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    iss = _issue_arg(gh, r, o)
    iss.state = "OPEN"
    g.err(f"✓ Reopened issue {r.full}#{iss.number} ({iss.title})\n")
    return 0


@gh_command("issue comment")
def gh_issue_comment(g, gh, args):
    o = parse(args, "body|b= body-file|F= repo|R= web")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    iss = _issue_arg(gh, r, o)
    iss.comments.append((acc.login, o.get("body") or ""))
    g.out(f"https://github.com/{r.full}/issues/{iss.number}#issuecomment-{2000 + len(iss.comments)}\n")
    return 0


# ---------------------------------------------------------------------------- runs and workflows

SYM = {"success": "✓", "failure": "X", "": "*"}


@gh_command("run list")
def gh_run_list(g, gh, args):
    o = parse(args, "limit|L= workflow|w= branch|b= repo|R= status|s= event|e= json=")
    r = current_repo(g, gh, o.get("repo"))
    runs = sorted(r.runs, key=lambda x: -x.id)
    if o.get("branch"):
        runs = [x for x in runs if x.branch == o.get("branch")]
    if o.get("workflow"):
        runs = [x for x in runs if x.workflow == o.get("workflow")]
    if not runs:
        g.err("no runs found\n")
        return 0
    now = gh.now()
    rows = []
    for x in runs[:int(o.get("limit") or 20)]:
        s, c = x.status(now)
        rows.append((SYM.get(c, "*"), x.title, x.workflow, x.branch, x.event, str(x.id),
                     f"{min(x.duration, now - x.created)}s", _ago(now, x.created)))
    heads = ("STATUS", "TITLE", "WORKFLOW", "BRANCH", "EVENT", "ID", "ELAPSED", "AGE")
    ws = [max(len(h), *(len(r_[i]) for r_ in rows)) for i, h in enumerate(heads)]
    g.out("  ".join(h.ljust(w) for h, w in zip(heads, ws)).rstrip() + "\n")
    for r_ in rows:
        g.out("  ".join(c.ljust(w) for c, w in zip(r_, ws)).rstrip() + "\n")
    return 0


def _run_arg(r, o):
    if o.args:
        for x in r.runs:
            if str(x.id) == o.args[0]:
                return x
        raise GhError(f"could not find any workflow run with ID {o.args[0]}")
    if not r.runs:
        raise GhError("no runs found")
    return max(r.runs, key=lambda x: x.id)


@gh_command("run view")
def gh_run_view(g, gh, args):
    o = parse(args, "log log-failed web|w verbose|v job|j= repo|R= exit-status")
    r = current_repo(g, gh, o.get("repo"))
    x = _run_arg(r, o)
    s, c = x.status(gh.now())
    sym = SYM.get(c, "*")
    g.out(f"\n{sym} {x.branch} {x.workflow} · {x.id}\nTriggered via {x.event} {_ago(gh.now(), x.created)}\n\nJOBS\n"
          f"{sym} build in {x.duration}s (ID {x.id + 11})\n\nFor more information about the job, try: gh run view "
          f"--job={x.id + 11}\nView this run on GitHub: https://github.com/{r.full}/actions/runs/{x.id}\n")
    return 1 if o.get("exit-status") and c == "failure" else 0


@gh_command("run watch")
def gh_run_watch(g, gh, args):
    o = parse(args, "interval|i= exit-status repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    x = _run_arg(r, o)
    while x.status(gh.now())[0] != "completed":
        g.ctx._flush_nb()
        yield ("sleep", g.ctx.now + 3_000_000_000)
    s, c = x.status(gh.now())
    g.out(f"Refreshing run status every 3 seconds. Press Ctrl+C to quit.\n\n{SYM.get(c)} {x.branch} {x.workflow} · "
          f"{x.id}\nTriggered via {x.event} {_ago(gh.now(), x.created)}\n\nJOBS\n{SYM.get(c)} build in {x.duration}s "
          f"(ID {x.id + 11})\n\n{SYM.get(c)} Run {x.workflow} ({x.id}) completed with '{c}'\n")
    return 1 if o.get("exit-status") and c == "failure" else 0


@gh_command("workflow list")
def gh_workflow_list(g, gh, args):
    o = parse(args, "all|a repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    head = r.refs.get(f"refs/heads/{r.default_branch}")
    for i, (path, name, ev) in enumerate(gh.workflows(r, head)):
        g.out(f"{name}\tactive\t{8800 + i}\n")
    return 0


@gh_command("workflow run")
def gh_workflow_run(g, gh, args):
    o = parse(args, "ref|r= field|f= raw-field|F= repo|R=")
    r = current_repo(g, gh, o.get("repo"))
    if not o.args:
        raise GhError("workflow name or ID required")
    b = o.get("ref") or r.default_branch
    sha = r.refs.get(f"refs/heads/{b}")
    found = [w for w in gh.workflows(r, sha) if o.args[0] in (w[1], w[0].rsplit("/", 1)[-1])]
    if not found:
        raise GhError(f"could not find any workflows named {o.args[0]}")
    if "workflow_dispatch" not in found[0][2]:
        raise GhError(f"could not create workflow dispatch event: HTTP 422: Workflow does not have 'workflow_dispatch' "
                      "trigger")
    from .github import Run, _tests_fail
    gh.run_ids += 1
    r.runs.append(Run(gh.run_ids, found[0][1], "workflow_dispatch", b, sha, gh.now(), _tests_fail(odb(g.world), sha),
                      found[0][1]))
    g.out(f"✓ Created workflow_dispatch event for {found[0][0].rsplit('/', 1)[-1]} at {b}\n\nTo see runs for this "
          f"workflow, try: gh run list --workflow={found[0][0].rsplit('/', 1)[-1]}\n")
    return 0


# ---------------------------------------------------------------------------- release

@gh_command("release create")
def gh_release_create(g, gh, args):
    o = parse(args, "notes|n= notes-file|F= title|t= target= draft|d prerelease|p generate-notes latest repo|R=")
    acc = need_login(gh)
    r = current_repo(g, gh, o.get("repo"))
    if not o.args:
        raise GhError("could not create: no tag name provided")
    tag = o.args[0]
    if any(rel.tag == tag for rel in r.releases):
        raise GhError("HTTP 422: Validation Failed (https://api.github.com/repos/%s/releases)\nRelease.tag_name already "
                      "exists" % r.full)
    if f"refs/tags/{tag}" not in r.refs:
        target = o.get("target") or r.default_branch
        sha = r.refs.get(f"refs/heads/{target}")
        if sha is None:
            raise GhError(f"HTTP 422: Validation Failed: target_commitish is invalid")
        r.refs[f"refs/tags/{tag}"] = sha
    notes = o.get("notes") or ""
    if o.get("generate-notes") and not notes:
        notes = "## What's Changed\n" + "".join(f"* {p.title} by @{p.author} in #{p.number}\n" for p in gh.pulls_of(r)
                                                 if p.state == "MERGED")
    from .github import Release
    r.releases.append(Release(tag, o.get("title") or tag, notes, acc.login, gh.now(), bool(o.get("draft")),
                              bool(o.get("prerelease"))))
    gh.trigger_runs(r, "release", r.default_branch, r.refs[f"refs/tags/{tag}"])
    g.out(f"https://github.com/{r.full}/releases/tag/{tag}\n")
    return 0


@gh_command("release list")
def gh_release_list(g, gh, args):
    o = parse(args, "limit|L= repo|R= exclude-drafts")
    r = current_repo(g, gh, o.get("repo"))
    rels = sorted(r.releases, key=lambda x: -x.created)
    if not rels:
        g.err("no releases found\n")
        return 0
    now = gh.now()
    for i, rel in enumerate(rels):
        kind = "Draft" if rel.draft else "Pre-release" if rel.prerelease else ("Latest" if i == 0 else "")
        g.out(f"{rel.name}\t{kind}\t{rel.tag}\t{_ago(now, rel.created)}\n")
    return 0


@gh_command("release view")
def gh_release_view(g, gh, args):
    o = parse(args, "web|w repo|R= json=")
    r = current_repo(g, gh, o.get("repo"))
    rels = sorted(r.releases, key=lambda x: -x.created)
    rel = next((x for x in rels if not o.args or x.tag == o.args[0]), None)
    if rel is None:
        raise GhError("release not found")
    g.out(f"{rel.name}\n{rel.author} released this {_ago(gh.now(), rel.created)}\n\n  {rel.notes}\n\n"
          f"View on GitHub: https://github.com/{r.full}/releases/tag/{rel.tag}\n")
    return 0


__all__ = ["clone_url"]

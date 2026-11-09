"""log, show, diff, blame, shortlog, reflog, describe, rev-list, grep, bisect."""

from __future__ import annotations

import posixpath
import re
from datetime import datetime, timedelta, timezone

from ..vfs import FSError
from .cli import Usage, command, parse, pathspec_matcher, run_maybe
from .diff import ZERO, Change, Colors, compare, funcrx_for, numstat, patch, split_lines, stat_block, stat_name
from .objects import EMPTY_TREE, GitError
from .revparse import dwim_ref, resolve, rev_list_args, try_resolve, unknown_revision
from .status import worktree_flat

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------- dates

def _dt(p):
    sign = 1 if p.tz[0] == "+" else -1
    off = sign * (int(p.tz[1:3]) * 60 + int(p.tz[3:5]))
    return datetime.fromtimestamp(p.when, timezone(timedelta(minutes=off)))


def relative(when: int, now: int) -> str:
    d = now - when
    if d < 0:
        return "in the future"
    if d < 90:
        return f"{d} second{'s' if d != 1 else ''} ago"
    m = (d + 30) // 60
    if m < 90:
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = (m + 30) // 60
    if h < 36:
        return f"{h} hour{'s' if h != 1 else ''} ago"
    days = (h + 12) // 24
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 70:
        w = (days + 3) // 7
        return f"{w} week{'s' if w != 1 else ''} ago"
    if days < 365:
        mo = (days + 15) // 30
        return f"{mo} month{'s' if mo != 1 else ''} ago"
    years = days // 365
    months = (days - years * 365 + 15) // 30
    if years < 5 and months:
        return f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''} ago"
    return f"{years} year{'s' if years != 1 else ''} ago"


def fmt_date(p, mode: str, now: int, tz=None) -> str:
    d = _dt(p)
    mode = mode or "default"
    if mode == "relative":
        return relative(p.when, now)
    if mode in ("short",):
        return d.strftime("%Y-%m-%d")
    if mode in ("iso", "iso8601"):
        return d.strftime("%Y-%m-%d %H:%M:%S ") + p.tz
    if mode in ("iso-strict", "iso8601-strict"):
        z = p.tz[:3] + ":" + p.tz[3:]
        return d.strftime("%Y-%m-%dT%H:%M:%S") + z
    if mode == "rfc" or mode == "rfc2822":
        return f"{DAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year} {d.strftime('%H:%M:%S')} {p.tz}"
    if mode == "unix":
        return str(p.when)
    if mode == "raw":
        return f"{p.when} {p.tz}"
    if mode.startswith("format:"):
        return d.strftime(mode[7:])
    if mode == "local":
        from ..timeutil import local_dt
        ld = local_dt(p.when * 1_000_000_000, tz)
        return f"{DAYS[ld.weekday()]} {MONTHS[ld.month - 1]} {ld.day} {ld.strftime('%H:%M:%S')} {ld.year}"
    if mode == "human":
        return relative(p.when, now) if now - p.when < 86400 else f"{DAYS[d.weekday()]} {MONTHS[d.month - 1]} {d.day}"
    return f"{DAYS[d.weekday()]} {MONTHS[d.month - 1]} {d.day} {d.strftime('%H:%M:%S')} {d.year} {p.tz}"


# ---------------------------------------------------------------------------- decorations

def decorations(repo, full=False):
    """sha -> [labels] in git's order (HEAD first, then tags, remote branches, branches: reverse ref order)."""
    db = repo.odb
    out: dict = {}
    refs = repo.ref_names("refs/")
    for name in sorted(refs):
        sha = refs[name]
        if name.startswith("refs/heads/"):
            label = name if full else name[len("refs/heads/"):]
            kind = "branch"
        elif name.startswith("refs/remotes/"):
            label = name if full else name[len("refs/remotes/"):]
            kind = "remote"
        elif name.startswith("refs/tags/"):
            label = ("tag: " + name) if full else "tag: " + name[len("refs/tags/"):]
            kind = "tag"
            sha = repo.peel(sha)
        elif name == "refs/stash":
            label, kind = ("refs/stash", "stash")
        else:
            continue
        out.setdefault(sha, []).insert(0, (label, kind, name))
    # symbolic remote HEADs (origin/HEAD) come right after what they point to
    for rem in repo.remotes():
        r = repo.read_ref_raw(f"refs/remotes/{rem}/HEAD")
        if r and r[0] == "sym":
            sha = repo.resolve_ref(r[1])
            if sha:
                lst = out.setdefault(sha, [])
                pos = next((i for i, x in enumerate(lst) if x[2] == r[1]), len(lst))
                lst.insert(pos + 1, (f"{rem}/HEAD" if not full else f"refs/remotes/{rem}/HEAD", "remote", "x"))
    h = repo.head()
    if h:
        lst = out.setdefault(h, [])
        hb = repo.head_ref()
        if hb:
            label = hb if full else hb[len("refs/heads/"):] if hb.startswith("refs/heads/") else hb
            lst[:] = [x for x in lst if x[2] != hb]
            lst.insert(0, (f"HEAD -> {label}", "head", hb))
        else:
            lst.insert(0, ("HEAD", "head", "HEAD"))
    return out


def deco_text(labels, color: Colors, paren=True) -> str:
    if not labels:
        return ""
    parts = []
    for label, kind, _ in labels:
        if kind == "head":
            if " -> " in label:
                a, b = label.split(" -> ", 1)
                parts.append(color("1;36", a + " ->") + " " + color("1;32", b))
            else:
                parts.append(color("1;36", label))
        elif kind == "branch":
            parts.append(color("1;32", label))
        elif kind == "remote":
            parts.append(color("1;31", label))
        elif kind == "tag":
            parts.append(color("1;33", label))
        else:
            parts.append(color("1;35", label))
    sep = color("33", ", ")
    inner = sep.join(parts)
    return (color("33", " (") + inner + color("33", ")")) if paren else inner


# ---------------------------------------------------------------------------- formats

def format_placeholders(repo, sha, fmt, now, deco, color_on, date_mode=None, reflog=None, abbrev=7):
    db = repo.odb
    c = db.commit(sha)
    color = Colors(color_on)
    out = []
    i = 0
    pending_color = False
    while i < len(fmt):
        ch = fmt[i]
        if ch != "%" or i + 1 >= len(fmt):
            out.append(ch)
            i += 1
            continue
        nxt = fmt[i + 1]
        two = fmt[i + 1:i + 3]
        if nxt == "%":
            out.append("%")
            i += 2
            continue
        if nxt == "n":
            out.append("\n")
            i += 2
            continue
        if nxt == "C":
            m = re.match(r"%C(\((?:auto,)?([a-z ]+)\)|red|green|blue|reset|yellow|cyan|magenta|bold)", fmt[i:])
            if m:
                name = m.group(2) or m.group(1)
                codes = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
                         "white": "37", "bold": "1", "dim": "2", "ul": "4", "reset": "0", "auto": None}
                if color_on and name:
                    seq = []
                    for w in name.split():
                        if w in codes and codes[w]:
                            seq.append(codes[w])
                    if seq:
                        out.append("\x1b[" + ";".join(seq) + "m" if seq != ["0"] else "\x1b[m")
                i += len(m.group(0))
                continue
        two_map = {
            "an": lambda: c.author.name, "ae": lambda: c.author.email, "al": lambda: c.author.email.split("@")[0],
            "ad": lambda: fmt_date(c.author, date_mode, now), "ar": lambda: relative(c.author.when, now),
            "as": lambda: fmt_date(c.author, "short", now), "at": lambda: str(c.author.when),
            "ai": lambda: fmt_date(c.author, "iso", now), "aI": lambda: fmt_date(c.author, "iso-strict", now),
            "aD": lambda: fmt_date(c.author, "rfc", now),
            "cn": lambda: c.committer.name, "ce": lambda: c.committer.email,
            "cd": lambda: fmt_date(c.committer, date_mode, now), "cr": lambda: relative(c.committer.when, now),
            "cs": lambda: fmt_date(c.committer, "short", now), "ct": lambda: str(c.committer.when),
            "ci": lambda: fmt_date(c.committer, "iso", now), "cI": lambda: fmt_date(c.committer, "iso-strict", now),
            "gD": lambda: reflog[0] if reflog else "", "gd": lambda: reflog[1] if reflog else "",
            "gs": lambda: reflog[2] if reflog else "", "gn": lambda: reflog[3] if reflog else "",
        }
        if two in two_map:
            out.append(two_map[two]())
            i += 3
            continue
        one_map = {
            "H": lambda: sha, "h": lambda: db.abbrev(sha, abbrev),
            "T": lambda: c.tree, "t": lambda: db.abbrev(c.tree, abbrev),
            "P": lambda: " ".join(c.parents), "p": lambda: " ".join(db.abbrev(x, abbrev) for x in c.parents),
            "s": lambda: c.subject, "b": lambda: c.body, "B": lambda: c.message,
            "d": lambda: deco_text(deco.get(sha, []), Colors(color_on)) if deco is not None else "",
            "D": lambda: deco_text(deco.get(sha, []), Colors(color_on), paren=False) if deco is not None else "",
            "e": lambda: "", "N": lambda: "", "f": lambda: re.sub(r"[^A-Za-z0-9.]+", "-", c.subject).strip("-"),
        }
        if nxt in one_map:
            out.append(one_map[nxt]())
            i += 2
            continue
        out.append(fmt[i:i + 2])
        i += 2
    return "".join(out)


def commit_header(repo, sha, style, now, deco, color_on, date_mode=None, abbrev_commit=False, reflog=None):
    db = repo.odb
    c = db.commit(sha)
    color = Colors(color_on)
    shown = db.abbrev(sha) if abbrev_commit else sha
    d = deco_text(deco.get(sha, []), color) if deco is not None else ""
    if style == "oneline":
        return color("33", db.abbrev(sha)) + d + " " + c.subject + "\n"
    lines = [color("33", f"commit {shown}") + d]
    if reflog:
        lines.append(f"Reflog: {reflog[0]} ({reflog[3]})")
        lines.append(f"Reflog message: {reflog[2]}")
    if len(c.parents) > 1:
        lines.append("Merge: " + " ".join(db.abbrev(x) for x in c.parents))
    if style in ("short",):
        lines.append(f"Author: {c.author.name} <{c.author.email}>")
    elif style == "full":
        lines.append(f"Author: {c.author.name} <{c.author.email}>")
        lines.append(f"Commit: {c.committer.name} <{c.committer.email}>")
    elif style == "fuller":
        lines.append(f"Author:     {c.author.name} <{c.author.email}>")
        lines.append(f"AuthorDate: {fmt_date(c.author, date_mode, now)}")
        lines.append(f"Commit:     {c.committer.name} <{c.committer.email}>")
        lines.append(f"CommitDate: {fmt_date(c.committer, date_mode, now)}")
    elif style == "raw":
        lines = [color("33", f"commit {sha}") + d, f"tree {c.tree}"] + [f"parent {p}" for p in c.parents] + \
            [f"author {c.author.line()}", f"committer {c.committer.line()}"]
    else:
        lines.append(f"Author: {c.author.name} <{c.author.email}>")
        lines.append(f"Date:   {fmt_date(c.author, date_mode, now)}")
    msg = c.message.rstrip("\n")
    body = "\n".join(("    " + ln) if ln else "    " for ln in msg.split("\n")) if style != "short" else \
        "    " + c.subject
    return "\n".join(lines) + "\n\n" + body + "\n"


# ---------------------------------------------------------------------------- history walking

def topo_sort(db, commits):
    """git's --topo-order (sort_in_topological_order, graph order): a commit appears only after all its children;
    ready commits are taken from a stack, so after a merge its last parent's line comes first."""
    cset = set(commits)
    indeg = {c: 0 for c in commits}
    for c in commits:
        for p in db.parents(c):
            if p in cset:
                indeg[p] += 1
    stack = [c for c in commits if indeg[c] == 0][::-1]     # tips, first one on top
    out = []
    while stack:
        c = stack.pop()
        out.append(c)
        for p in db.parents(c):
            if p in cset:
                indeg[p] -= 1
                if indeg[p] == 0:
                    stack.append(p)
    return out


def walk(repo, includes, excludes, first_parent=False):
    db = repo.odb
    if first_parent:
        out = []
        ex = db.reachable(excludes) if excludes else set()
        seen = set()
        for h in includes:
            c = h
            while c and c not in seen and c not in ex:
                seen.add(c)
                out.append(c)
                ps = db.commit(c).parents
                c = ps[0] if ps else None
        return sorted(out, key=lambda s: -db.commit(s).committer.when) if len(includes) > 1 else out
    return db.ancestors(includes, excludes)


def touches(repo, sha, match, simplify=True):
    """Does commit `sha` change a path the pathspec matches (TREESAME simplification for merges)?"""
    db = repo.odb
    c = db.commit(sha)
    mine = {p: v for p, v in db.flat(c.tree).items() if match(p)}
    if not c.parents:
        return bool(mine)
    diffs = []
    for p in c.parents:
        theirs = {k: v for k, v in db.flat(db.commit(p).tree).items() if match(k)}
        diffs.append(theirs != mine)
    return all(diffs) if simplify else diffs[0]


# ---------------------------------------------------------------------------- graph

class Graph:
    """A simplified port of git's --graph: commit rows (*), merge rows (|\\), and rows where lines join (|/)."""

    def __init__(self, db, color: Colors):
        self.db = db
        self.cols: list[str] = []
        self.color = color

    def rows(self, sha, shown: set):
        """-> (the commit row's graph, rows after it (a merge fanning out, lines joining), padding width)."""
        db = self.db
        cols = self.cols
        if sha in cols:
            idx = cols.index(sha)
        else:
            cols.append(sha)
            idx = len(cols) - 1
        parents = [p for p in db.commit(sha).parents if p in shown]
        n_before = len(cols)
        new = list(cols)
        repl = []
        for p in parents:
            if p not in repl:
                repl.append(p)
        new[idx:idx + 1] = repl
        width = max(n_before, len(new))
        commit_row = " ".join("*" if i == idx else "|" for i in range(n_before))
        after = []
        if len(repl) > 1:
            # the merge fans out to the right
            row = []
            for i in range(n_before):
                if i < idx:
                    row.append("|")
                elif i == idx:
                    row.append("|\\")
                else:
                    row.append("\\")
            after.append(" ".join(row))
        elif len(repl) == 0:
            # a root commit: its column ends
            if idx < n_before - 1:
                row = []
                for i in range(n_before):
                    if i < idx:
                        row.append("|")
                    elif i > idx:
                        row.append("/")
                after.append(" ".join(row[:idx]) + (" " if idx else "") + " ".join(row[idx:]))
        # columns holding the same commit join: the right one slides left into the gap (|/)
        while True:
            dup = None
            for i in range(len(new)):
                for j in range(i + 1, len(new)):
                    if new[i] == new[j]:
                        dup = (i, j)
                        break
                if dup:
                    break
            if not dup:
                break
            i, j = dup
            cells = [" "] * (2 * len(new))
            for k in range(len(new)):
                if k < j:
                    cells[2 * k] = "|"
                else:
                    cells[2 * k - 1] = "/"
            after.append("".join(cells).rstrip())
            del new[j]
        self.cols = new
        pad = 2 * width
        return commit_row, after, pad

    def continuation(self):
        return " ".join("|" for _ in self.cols)


def _paint_graph(text: str, color: Colors) -> str:
    if not color.on:
        return text
    palette = ["31", "32", "33", "34", "35", "36"]
    out = []
    col = 0
    for ch in text:
        if ch in "|/\\":
            out.append(color(palette[(col // 2) % len(palette)], ch))
        else:
            out.append(ch)
        col += 1
    return "".join(out)


# ---------------------------------------------------------------------------- log

LOG_SPEC = ("oneline graph decorate=? no-decorate all branches=? tags=? remotes=? n|max-count= skip= since|after= "
            "until|before= author= committer= grep= all-match invert-grep regexp-ignore-case|i merges no-merges "
            "first-parent reverse patch|p|u stat=? shortstat name-only name-status numstat follow S= G= L= "
            "pretty|format= abbrev-commit no-abbrev-commit date= topo-order date-order left-right ancestry-path "
            "diff-filter= full-diff walk-reflogs|g relative-date color=? no-color word-diff=? unified|U= "
            "show-signature simplify-by-decoration source no-renames find-renames|M=? pickaxe-regex boundary "
            "cherry-pick cherry-mark right-only left-only full-history dense sparse exclude= no-patch|s "
            "parents children compact-summary summary mailmap no-merges- no-walk=?")


@command("log")
def git_log(g, args, default_n=None, reflog_mode=False):
    o = parse(args, LOG_SPEC, numbers=True)
    repo = g.repo()
    db = repo.odb
    return _log(g, repo, o, reflog_mode=reflog_mode)


def _revs_and_paths(g, repo, o):
    revs, paths = [], list(o.paths or [])
    for a in o.args:
        if paths:
            paths.append(a)
            continue
        from .revparse import is_revision
        if is_revision(repo, a):
            revs.append(a)
        elif repo.worktree and repo.wt_node(g.to_rel(repo, a)) is not None:
            paths.append(a)
        else:
            raise unknown_revision(a)
    return revs, paths


def _log(g, repo, o, reflog_mode=False):
    db = repo.odb
    now = repo.now()
    revs, paths = _revs_and_paths(g, repo, o)
    color_on = g.color("diff") if o.get("color") is None else o.get("color") not in (False, "never")
    if o.get("no-color"):
        color_on = False
    color = Colors(color_on)
    includes, excludes, sym = [], [], None
    if o.get("all"):
        includes += [repo.peel(v) for v in repo.ref_names("refs/").values()]
        if repo.head():
            includes.append(repo.head())
    if o.get("branches"):
        includes += list(repo.branches().values())
    if o.get("tags"):
        includes += [repo.peel(v) for v in repo.tags().values()]
    if o.get("remotes"):
        includes += list(repo.remote_branches().values())
    if revs:
        inc, exc, sym = rev_list_args(repo, revs)
        includes += inc
        excludes += exc
    elif not includes:
        h = repo.head()
        if not h:
            b = repo.head_branch() or "master"
            raise GitError(f"fatal: your current branch '{b}' does not have any commits yet")
        includes.append(h)
    includes = [x for x in includes if x and db.has(x) and db.kind(x) == "commit"]
    if reflog_mode:
        return _reflog_log(g, repo, o, revs)
    if o.get("no-walk") is not None:          # just the commits named, newest first (--no-walk=unsorted: as given)
        commits = list(dict.fromkeys(includes))
        if o.get("no-walk") != "unsorted":
            commits.sort(key=lambda s: -db.commit(s).committer.when)
    else:
        commits = walk(repo, includes, excludes, first_parent=bool(o.get("first-parent")))
    if o.get("graph") or o.get("topo-order"):
        commits = topo_sort(db, commits)
    match = pathspec_matcher(g, repo, paths)
    follow_path = None
    if o.get("follow") and paths:
        follow_path = g.to_rel(repo, paths[0])
    sel = []
    rx_grep = [re.compile(x, re.I if o.get("regexp-ignore-case") else 0) for x in o.all("grep")]
    since = until = None
    if o.get("since"):
        from .repo import parse_git_date
        since = parse_git_date(o.get("since"), repo)[0]
    if o.get("until"):
        from .repo import parse_git_date
        until = parse_git_date(o.get("until"), repo)[0]
    pick_s = o.get("S")
    pick_g = re.compile(o.get("G")) if o.get("G") else None
    for sha in commits:
        c = db.commit(sha)
        if o.get("merges") and len(c.parents) < 2:
            continue
        if o.get("no-merges") and len(c.parents) > 1:
            continue
        if o.get("author") and not re.search(o.get("author"), f"{c.author.name} <{c.author.email}>",
                                             re.I if o.get("regexp-ignore-case") else 0):
            continue
        if o.get("committer") and not re.search(o.get("committer"), f"{c.committer.name} <{c.committer.email}>"):
            continue
        if rx_grep:
            hits = [bool(r.search(c.message)) for r in rx_grep]
            ok = all(hits) if o.get("all-match") else any(hits)
            if o.get("invert-grep"):
                ok = not ok
            if not ok:
                continue
        if since is not None and c.committer.when < since:
            continue
        if until is not None and c.committer.when > until:
            continue
        if follow_path:
            continue_ = _follow_hit(repo, sha, follow_path)
            if continue_ is None:
                continue
            follow_path = continue_
        elif match and not touches(repo, sha, match, simplify=not o.get("full-history")):
            continue
        if pick_s is not None or pick_g is not None:
            if not _pickaxe(repo, sha, pick_s, pick_g, match):
                continue
        sel.append(sha)
    if o.get("L"):
        return _log_L(g, repo, o, sel, now, color)
    skip = int(o.get("skip") or 0)
    sel = sel[skip:]
    if o.get("n") is not None:
        sel = sel[:int(o.get("n"))]
    if o.get("reverse"):
        sel = sel[::-1]
    deco_opt = o.get("decorate")
    if o.get("no-decorate"):
        deco = None
    elif deco_opt is not None and deco_opt is not False:
        deco = decorations(repo, full=deco_opt == "full")
    else:
        cfg = repo.config.get("log.decorate", "auto")
        on = cfg in ("true", "short", "full", "yes") or (cfg == "auto" and g.ctx.stdout_is_tty())
        deco = decorations(repo, full=cfg == "full") if (on or o.get("oneline") and g.ctx.stdout_is_tty()) else None
    fmt = o.get("pretty")
    style = "oneline" if o.get("oneline") else "medium"
    if fmt is not None and fmt is not True:
        if fmt in ("oneline", "short", "medium", "full", "fuller", "raw", "reference", "email"):
            style = fmt
            fmt = None
        else:
            if fmt.startswith(("format:", "tformat:")):
                fmt = fmt.split(":", 1)[1]
            style = "format"
    date_mode = o.get("date")
    if o.get("relative-date"):
        date_mode = "relative"
    want_patch = bool(o.get("patch")) or o.get("unified") is not None
    want_stat = o.get("stat") is not None and o.get("stat") is not False
    shown = set(sel) if o.get("graph") else set()
    if o.get("graph"):
        # the graph also passes through commits we filtered out by count, so use everything walked
        shown = set(commits)
    graph = Graph(db, color) if o.get("graph") else None
    out = []
    abbrev_commit = bool(o.get("abbrev-commit")) or style == "oneline"
    funcrx = funcrx_for(repo)
    for k, sha in enumerate(sel):
        c = db.commit(sha)
        if style == "format":
            text = format_placeholders(repo, sha, fmt, now, deco or decorations(repo), color_on, date_mode)
            text = text + "\n"
        elif style == "reference":
            text = f"{db.abbrev(sha)} ({c.subject}, {fmt_date(c.author, 'short', now)})\n"
        else:
            text = commit_header(repo, sha, style, now, deco, color_on, date_mode, abbrev_commit)
        extra = ""
        if want_patch or want_stat or o.get("name-only") or o.get("name-status") or o.get("numstat") or \
                o.get("shortstat"):
            parent_tree = db.commit(c.parents[0]).tree if c.parents else EMPTY_TREE
            if len(c.parents) <= 1 or o.get("first-parent"):
                changes = compare(db.flat(parent_tree), db.flat(c.tree), renames=not o.get("no-renames"), db=db)
                if match and not o.get("full-diff"):
                    changes = [ch for ch in changes if match(ch.path) or match(ch.old_path)]
                extra = render_changes(g, repo, changes, o, color, funcrx, want_patch, want_stat)
        if graph is not None:
            row, after, pad = graph.rows(sha, shown)
            lines = text.rstrip("\n").split("\n")
            if extra:
                lines += [""] + extra.rstrip("\n").split("\n") if style != "oneline" else extra.rstrip("\n").split("\n")
            res = [_paint_graph(row.ljust(pad), color) + lines[0]]
            cont = graph.continuation()
            for ln in after:
                res.append(_paint_graph(ln.ljust(pad), color))
            body_prefix = _paint_graph(cont.ljust(pad) if cont else "", color)
            for ln in lines[1:]:
                res.append((body_prefix + ln).rstrip() if not ln else body_prefix + ln)
            if style not in ("oneline", "format") and k < len(sel) - 1:
                res.append(body_prefix.rstrip())
            out.append("\n".join(r.rstrip() if not r.strip() else r for r in res) + "\n")
            continue
        sep = "" if style in ("oneline", "format", "reference") else ("\n" if k < len(sel) - 1 else "")
        if extra:
            if style in ("oneline", "format", "reference"):
                out.append(text + extra + ("\n" if want_patch and k < len(sel) - 1 and style != "format" else ""))
            else:
                out.append(text + "\n" + extra + sep)
        else:
            out.append(text + sep)
    g.out("".join(out))
    return 0


def render_changes(g, repo, changes, o, color, funcrx, want_patch, want_stat):
    db = repo.odb
    ds = o.get("diff-filter")
    if ds:
        keep = set(ds.replace(" ", "").upper())
        changes = [c for c in changes if c.status in keep]
    parts = []
    if o.get("name-only"):
        parts.append("".join(f"{c.path}\n" for c in changes))
    elif o.get("name-status"):
        parts.append("".join((f"R{c.score:03d}\t{c.old_path}\t{c.new_path}\n" if c.status == "R" else
                              f"{c.status}\t{c.path}\n") for c in changes))
    if o.get("numstat"):
        rows = []
        for c in changes:
            a, d = numstat(c, db)
            rows.append(f"{'-' if a is None else a}\t{'-' if d is None else d}\t{stat_name(c)}\n")
        parts.append("".join(rows))
    if want_stat or o.get("shortstat"):
        rows = []
        for c in changes:
            a, d = numstat(c, db)
            if a is None:
                rows.append((stat_name(c), 0, 0, (db.size(c.old_sha) if c.status != "A" else 0,
                                                  db.size(c.new_sha) if c.status != "D" else 0)))
            else:
                rows.append((stat_name(c), a, d, None))
        sw = o.get("stat")
        width = int(sw) if isinstance(sw, str) and sw.split(",")[0].isdigit() else g.columns()
        lines = stat_block(rows, width, color, summary_only=bool(o.get("shortstat")) and not want_stat)
        if lines:
            parts.append("\n".join(lines) + "\n")
    if want_patch:
        ctx_n = int(o.get("unified")) if o.get("unified") not in (None, True) else 3
        wd = o.get("word-diff")
        if wd is True:
            wd = "plain"
        pt = "".join(patch(c, db, ctx=ctx_n, color=color, funcrx_of=funcrx, word_diff=wd) for c in changes)
        if parts and pt:
            parts.append("\n")
        parts.append(pt)
    return "".join(parts)


def _follow_hit(repo, sha, path):
    """--follow: did this commit change `path` (or create it by renaming)? -> the path to follow next, or None."""
    db = repo.odb
    c = db.commit(sha)
    mine = db.flat(c.tree)
    if path not in mine:
        return None
    if not c.parents:
        return path
    parent = db.flat(db.commit(c.parents[0]).tree)
    if parent.get(path) == mine.get(path):
        return None
    if path not in parent:
        changes = compare(parent, mine, renames=True, db=db)
        for ch in changes:
            if ch.status == "R" and ch.new_path == path:
                return ch.old_path
    return path


def _pickaxe(repo, sha, s, grx, match):
    db = repo.odb
    c = db.commit(sha)
    new = db.flat(c.tree)
    old = db.flat(db.commit(c.parents[0]).tree) if c.parents else {}
    for p in set(new) | set(old):
        if match and not match(p):
            continue
        if new.get(p) == old.get(p):
            continue
        a = bytes(db.raw(old[p][1])).decode("utf-8", "replace") if p in old else ""
        b = bytes(db.raw(new[p][1])).decode("utf-8", "replace") if p in new else ""
        if s is not None and a.count(s) != b.count(s):
            return True
        if grx is not None:
            la, lb = split_lines(a.encode()), split_lines(b.encode())
            from .xdiff import diff_lines
            for c1, n1, c2, n2 in diff_lines(la, lb):
                if any(grx.search(x) for x in la[c1:c1 + n1] + lb[c2:c2 + n2]):
                    return True
    return False


def _function_range(lines, name):
    """Lines of the Python function/class `name`: (start, end) 0-based, end exclusive."""
    rx = re.compile(rf"^(\s*)(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
    for i, ln in enumerate(lines):
        m = rx.match(ln)
        if m:
            ind = len(m.group(1))
            j = i + 1
            while j < len(lines):
                t = lines[j]
                if t.strip() and (len(t) - len(t.lstrip())) <= ind:
                    break
                j += 1
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            return i, j
    return None


def _log_L(g, repo, o, sel, now, color):
    """git log -L :func:file (and -L a,b:file): commits that changed those lines, with the changes."""
    db = repo.odb
    spec = o.get("L")
    m = re.match(r"^:([^:]+):(.+)$", spec) or re.match(r"^(\d+),(\+?\d+):(.+)$", spec)
    if not m:
        raise GitError(f"fatal: invalid -L argument '{spec}'")
    func = None
    if spec.startswith(":"):
        func, path = m.group(1), g.to_rel(repo, m.group(2))
    else:
        a, b, path = m.group(1), m.group(2), g.to_rel(repo, m.group(3))
    out = []
    for sha in sel:
        c = db.commit(sha)
        cur = db.flat(c.tree).get(path)
        if cur is None:
            continue
        new_lines = split_lines(db.raw(cur[1]))
        prev = db.flat(db.commit(c.parents[0]).tree).get(path) if c.parents else None
        old_lines = split_lines(db.raw(prev[1])) if prev else []
        if func:
            rn = _function_range(new_lines, func)
            ro = _function_range(old_lines, func) if old_lines else None
        else:
            s0 = int(a) - 1
            e0 = s0 + int(b[1:]) if b.startswith("+") else int(b)
            rn, ro = (s0, min(e0, len(new_lines))), (s0, min(e0, len(old_lines))) if old_lines else None
        if rn is None:
            continue
        nb = new_lines[rn[0]:rn[1]]
        ob = old_lines[ro[0]:ro[1]] if ro else []
        if nb == ob:
            continue
        head = commit_header(repo, sha, "medium", now, None, color.on)
        body = [color("1", f"diff --git a/{path} b/{path}"), color("1", f"--- a/{path}" if ob else "--- /dev/null"),
                color("1", f"+++ b/{path}"),
                color("36", f"@@ -{(ro[0] + 1) if ro else 0},{len(ob)} +{rn[0] + 1},{len(nb)} @@")]
        from .xdiff import diff_lines
        i1 = 0
        for c1, n1, c2, n2 in diff_lines(ob, nb):
            while i1 < c1:
                body.append(" " + ob[i1].rstrip("\n"))
                i1 += 1
            body += [color("31", "-" + x.rstrip("\n")) for x in ob[c1:c1 + n1]]
            body += [color("32", "+" + x.rstrip("\n")) for x in nb[c2:c2 + n2]]
            i1 = c1 + n1
        while i1 < len(ob):
            body.append(" " + ob[i1].rstrip("\n"))
            i1 += 1
        out.append(head + "\n" + "\n".join(body) + "\n")
    g.out("\n".join(out))
    return 0


def _reflog_log(g, repo, o, revs):
    db = repo.odb
    ref = revs[0] if revs else "HEAD"
    full, _ = dwim_ref(repo, ref) if ref != "HEAD" else ("HEAD", None)
    full = full or ref
    entries = repo.reflog(full)
    now = repo.now()
    out = []
    name = ref if ref == "HEAD" else full.split("/")[-1] if full.startswith("refs/heads/") else ref
    for i, (old, new, who, msg) in enumerate(entries):
        if o.get("n") is not None and i >= int(o.get("n")):
            break
        if not db.has(new):
            continue
        sel = f"{name}@{{{i}}}"
        out.append(commit_header(repo, new, "medium", now, None, g.color(), reflog=(sel, sel, msg, who.name)))
    g.out("\n".join(out))
    return 0


# ---------------------------------------------------------------------------- show

@command("show")
def git_show(g, args):
    o = parse(args, LOG_SPEC + " quiet|q expand-tabs", numbers=True)
    repo = g.repo()
    db = repo.odb
    now = repo.now()
    color_on = g.color("diff") and not o.get("no-color")
    color = Colors(color_on)
    objs = o.args or ["HEAD"]
    out = []
    funcrx = funcrx_for(repo)
    for spec in objs:
        sha = try_resolve(repo, spec)
        if sha is None:
            if ":" in spec:
                rev, _, path = spec.partition(":")
                if try_resolve(repo, rev or "HEAD") is not None:
                    raise GitError(f"fatal: path '{path}' does not exist in '{rev or 'HEAD'}'")
            raise unknown_revision(spec)
        kind = db.kind(sha)
        if kind == "blob":
            out.append(bytes(db.raw(sha)).decode("utf-8", "surrogateescape"))
            continue
        if kind == "tree":
            ents = db.tree(sha)
            out.append(f"tree {spec}\n\n" + "".join(f"{n}{'/' if m == '40000' else ''}\n" for m, n, _ in ents))
            continue
        if kind == "tag":
            t = db.tag(sha)
            text = color("33", f"tag {t.tag}") + "\n"
            if t.tagger:
                text += f"Tagger: {t.tagger.name} <{t.tagger.email}>\nDate:   {fmt_date(t.tagger, o.get('date'), now)}\n"
            text += "\n" + t.message.rstrip("\n") + "\n\n"
            out.append(text)
            sha = repo.peel(sha)
            if db.kind(sha) != "commit":
                continue
        c = db.commit(sha)
        fmt = o.get("pretty")
        cfg = repo.config.get("log.decorate", "auto")
        deco = decorations(repo) if (o.get("decorate") or (cfg == "auto" and g.ctx.stdout_is_tty())) and \
            not o.get("no-decorate") else None
        if o.get("oneline"):
            head = commit_header(repo, sha, "oneline", now, deco, color_on)
        elif fmt and fmt not in ("oneline", "short", "medium", "full", "fuller", "raw"):
            f = fmt.split(":", 1)[1] if fmt.startswith(("format:", "tformat:")) else fmt
            head = format_placeholders(repo, sha, f, now, deco or decorations(repo), color_on, o.get("date")) + "\n"
        else:
            head = commit_header(repo, sha, fmt or "medium", now, deco, color_on, o.get("date"),
                                 bool(o.get("abbrev-commit")))
        if o.get("no-patch") or o.get("quiet"):
            out.append(head)
            continue
        if len(c.parents) > 1 and not o.get("first-parent"):
            out.append(head)
            continue
        parent_tree = db.commit(c.parents[0]).tree if c.parents else EMPTY_TREE
        changes = compare(db.flat(parent_tree), db.flat(c.tree), renames=True, db=db)
        match = pathspec_matcher(g, repo, o.paths or [])
        if match:
            changes = [ch for ch in changes if match(ch.path)]
        want_stat = o.get("stat") is not None and o.get("stat") is not False
        want_patch = not (want_stat or o.get("name-only") or o.get("name-status") or o.get("numstat") or
                          o.get("shortstat")) or bool(o.get("patch"))
        body = render_changes(g, repo, changes, o, color, funcrx, want_patch, want_stat)
        if fmt and fmt.startswith("format:") and body:
            out.append(head + body)
        elif o.get("oneline"):
            out.append(head + body)
        else:
            out.append(head + ("\n" + body if body else ""))
    g.out("\n".join(x.rstrip("\n") + "\n" if i < len(out) - 1 else x for i, x in enumerate(out)) if len(out) > 1
          else "".join(out))
    return 0


# ---------------------------------------------------------------------------- diff

@command("diff")
def git_diff(g, args):
    o = parse(args, "cached staged stat=? numstat shortstat name-only name-status word-diff=? color-words=? "
                    "unified|U= patch|p|u no-patch|s exit-code quiet diff-filter= find-renames|M=? no-renames "
                    "R ignore-all-space|w ignore-space-change|b color=? no-color no-index ours|2 theirs|3 base|1 "
                    "merge-base check minimal histogram patience relative=? text|a binary full-index src-prefix= "
                    "dst-prefix= no-prefix compact-summary summary dirstat=? output=")
    db_repo = g.maybe_repo()
    if o.get("no-index") or db_repo is None:
        return _diff_no_index(g, o)
    repo = g.repo()
    db = repo.odb
    color_on = g.color("diff") if o.get("color") in (None, True, "auto") else o.get("color") == "always"
    if o.get("no-color") or o.get("color") is False:
        color_on = False
    color = Colors(color_on)
    revs, paths = [], list(o.paths or [])
    for a in o.args:
        if paths and o.paths is None:
            paths.append(a)
            continue
        from .revparse import is_revision
        if is_revision(repo, a):
            revs.append(a)
        elif repo.worktree and repo.wt_node(g.to_rel(repo, a)) is not None or (repo.worktree and
                                                                              g.to_rel(repo, a) in repo.index.paths()):
            paths.append(a)
        else:
            raise unknown_revision(a)
    match = pathspec_matcher(g, repo, paths)
    read_new = None
    renames = not o.get("no-renames")
    conflicts = repo.index.conflicts() if repo.worktree else {}
    staged = o.get("cached") or o.get("staged")
    if len(revs) == 1 and ("..." in revs[0] or ".." in revs[0]):
        a, sep, b = revs[0].partition("..." if "..." in revs[0] else "..")
        a_sha = resolve(repo, a or "HEAD", "commit")
        b_sha = resolve(repo, b or "HEAD", "commit")
        if sep == "...":
            bases = db.merge_bases(a_sha, b_sha)
            if not bases:
                raise GitError(f"fatal: {revs[0]}: no merge base")
            a_sha = bases[0]
        old, new = db.flat(db.commit(a_sha).tree), db.flat(db.commit(b_sha).tree)
    elif len(revs) == 2:
        old = db.flat(resolve(repo, revs[0], "tree"))
        new = db.flat(resolve(repo, revs[1], "tree"))
    elif staged:
        base = resolve(repo, revs[0], "tree") if revs else repo.head_tree()
        old = db.flat(base)
        new = {p: (e.mode_str, e.sha) for p, e in repo.index.stage0().items()}
    elif len(revs) == 1:
        if repo.bare:
            raise GitError("fatal: this operation must be run in a work tree")
        old = db.flat(resolve(repo, revs[0], "tree"))
        new = _wt_view(repo, include_new=False, base=old)
        read_new = lambda ch: _wt_data(repo, ch.new_path)  # noqa: E731
    else:
        if repo.bare:
            raise GitError("fatal: this operation must be run in a work tree")
        old = {p: (e.mode_str, e.sha) for p, e in repo.index.stage0().items()}
        new = _wt_view(repo, include_new=False, base=old)
        read_new = lambda ch: _wt_data(repo, ch.new_path)  # noqa: E731
        renames = False
    changes = compare(old, new, renames=renames and read_new is None, db=db)
    if match:
        changes = [c for c in changes if match(c.path) or match(c.old_path)]
    if o.get("R"):
        for c in changes:
            c.old_path, c.new_path = c.new_path, c.old_path
            c.old_mode, c.new_mode = c.new_mode, c.old_mode
            c.old_sha, c.new_sha = c.new_sha, c.old_sha
            c.status = {"A": "D", "D": "A"}.get(c.status, c.status)
    unmerged_text = ""
    if not revs and not staged and conflicts:
        listed = o.get("name-only") or o.get("name-status") or o.get("numstat") or o.get("stat") is not None
        filt = o.get("diff-filter")
        if not filt or "U" in filt.upper():
            for p in sorted(conflicts):
                if match and not match(p):
                    continue
                if o.get("name-only"):
                    unmerged_text += p + "\n"
                elif o.get("name-status"):
                    unmerged_text += f"U\t{p}\n"
                elif not listed:
                    unmerged_text += _combined_diff(repo, p, conflicts[p], color)
        changes = [c for c in changes if c.path not in conflicts]
    if o.get("quiet"):
        return 1 if changes else 0
    want_stat = o.get("stat") is not None and o.get("stat") is not False
    want_patch = not (want_stat or o.get("name-only") or o.get("name-status") or o.get("numstat") or
                      o.get("shortstat") or o.get("no-patch")) or bool(o.get("patch"))
    funcrx = funcrx_for(repo) if repo.worktree else None
    text = _render_diff(g, repo, changes, o, color, funcrx, want_patch, want_stat, read_new)
    if o.get("output"):
        g.fs.write_file(posixpath.join(g.cwd, o.get("output")), text, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
    else:
        g.out(unmerged_text + text)
    if o.get("exit-code"):
        return 1 if changes or unmerged_text else 0
    return 0


def _render_diff(g, repo, changes, o, color, funcrx, want_patch, want_stat, read_new):
    db = repo.odb
    ds = o.get("diff-filter")
    if ds:
        keep = set(ds.replace(" ", "").upper())
        changes = [c for c in changes if c.status in keep]
    parts = []
    if o.get("name-only"):
        parts.append("".join(f"{g.shown(repo, c.path) if o.get('relative') else c.path}\n" for c in changes))
    elif o.get("name-status"):
        parts.append("".join((f"R{c.score:03d}\t{c.old_path}\t{c.new_path}\n" if c.status == "R" else
                              f"{c.status}\t{c.path}\n") for c in changes))
    rows = []
    if o.get("numstat") or want_stat or o.get("shortstat"):
        for c in changes:
            a, d = numstat(c, db, read_new=read_new)
            rows.append((c, a, d))
    if o.get("numstat"):
        parts.append("".join(f"{'-' if a is None else a}\t{'-' if d is None else d}\t{stat_name(c)}\n"
                             for c, a, d in rows))
    if want_stat or o.get("shortstat"):
        srows = [(stat_name(c), a or 0, d or 0, None if a is not None else (0, 0)) for c, a, d in rows]
        sw = o.get("stat")
        width = int(sw) if isinstance(sw, str) and sw.split(",")[0].isdigit() else g.columns()
        lines = stat_block(srows, width, color, summary_only=bool(o.get("shortstat")) and not want_stat)
        if lines:
            parts.append("\n".join(lines) + "\n")
    if want_patch:
        ctx_n = int(o.get("unified")) if o.get("unified") not in (None, True) else 3
        wd = o.get("word-diff")
        if o.get("color-words") is not None:
            wd = "color"
        if wd is True:
            wd = "plain"
        sp = o.get("src-prefix") or ("" if o.get("no-prefix") else "a/")
        dp = o.get("dst-prefix") or ("" if o.get("no-prefix") else "b/")
        pt = "".join(patch(c, db, read_new=read_new, ctx=ctx_n, color=color, funcrx_of=funcrx, word_diff=wd,
                           src_prefix=sp, dst_prefix=dp, binary_ok=bool(o.get("text")),
                           full_index=bool(o.get("full-index"))) for c in changes)
        if parts and pt:
            parts.append("\n")
        parts.append(pt)
    return "".join(parts)


def _wt_view(repo, include_new, base):
    """The working tree as a flat tree for tracked paths: a changed file gets id ZERO (read from disk)."""
    idx = repo.index
    flat = worktree_flat(repo)
    out = {}
    for p, v in flat.items():
        b = base.get(p)
        out[p] = v if (b is not None and b == v) else (v[0], v[1])
    return out


def _wt_data(repo, path):
    node = repo.wt_node(path)
    if node is None:
        return b""
    return repo.file_bytes(node)


def _combined_diff(repo, p, stages, color):
    db = repo.odb
    node = repo.wt_node(p)
    cur = bytes(repo.file_bytes(node)) if node is not None else b""
    ours = bytes(db.raw(stages[2].sha)) if 2 in stages else b""
    theirs = bytes(db.raw(stages[3].sha)) if 3 in stages else b""
    lw = split_lines(cur)
    lo = split_lines(ours)
    lt = split_lines(theirs)
    from .xdiff import diff_lines
    mark_o = ["  "] * len(lw)
    in_o = [False] * len(lw)
    in_t = [False] * len(lw)
    for c1, n1, c2, n2 in diff_lines(lo, lw):
        for k in range(c2, c2 + n2):
            in_o[k] = True
    for c1, n1, c2, n2 in diff_lines(lt, lw):
        for k in range(c2, c2 + n2):
            in_t[k] = True
    body = []
    for k, ln in enumerate(lw):
        pre = ("+" if in_o[k] else " ") + ("+" if in_t[k] else " ")
        t = ln.rstrip("\n")
        body.append(color("32", pre + t) if pre.strip() else pre + t)
    s1 = stages.get(2)
    s2 = stages.get(3)
    head = [color("1", f"diff --cc {p}"),
            color("1", f"index {db.abbrev(s1.sha) if s1 else '0000000'},{db.abbrev(s2.sha) if s2 else '0000000'}..0000000"),
            color("1", f"--- a/{p}"), color("1", f"+++ b/{p}"),
            color("36", f"@@@ -1,{len(lo)} -1,{len(lt)} +1,{len(lw)} @@@")]
    return "\n".join(head + body) + "\n"


def _diff_no_index(g, o):
    a = o.args + (o.paths or [])
    if len(a) != 2:
        raise Usage("usage: git diff --no-index [<options>] <path> <path>")
    try:
        x = bytes(g.ctx.lookup(a[0]).data)
        y = bytes(g.ctx.lookup(a[1]).data)
    except FSError as e:
        raise GitError(f"error: Could not access '{a[0]}'")
    from .objects import ODB
    db = ODB()
    sa, sb = db.put("blob", x), db.put("blob", y)
    if sa == sb:
        return 0
    ch = Change("M", a[0], a[1], "100644", "100644", sa, sb)
    color = Colors(g.color("diff"))
    g.out(patch(ch, db, color=color))
    return 1


# ---------------------------------------------------------------------------- blame

@command("blame")
def git_blame(g, args):
    o = parse(args, "L= s e show-email porcelain|p line-porcelain show-name|f show-number|n w M C c root "
                    "date= incremental abbrev=")
    repo = g.repo()
    db = repo.odb
    a = o.args + (o.paths or [])
    rev = "HEAD"
    if len(a) == 2:
        rev, path = a
    elif len(a) == 1:
        path = a[0]
    else:
        raise Usage("usage: git blame [<options>] [<rev-opts>] [<rev>] [--] <file>")
    rel = g.to_rel(repo, path)
    head = resolve(repo, rev, "commit")
    flat = db.flat(db.commit(head).tree)
    if rel not in flat:
        raise GitError(f"fatal: no such path '{rel}' in {rev}")
    lines = split_lines(db.raw(flat[rel][1]))
    owners = [None] * len(lines)
    pending = {head: (rel, list(range(len(lines))), list(range(len(lines))))}
    from .xdiff import diff_lines
    order = db.ancestors([head])
    for sha in order:
        if sha not in pending:
            continue
        path_here, idxs, positions = pending.pop(sha)
        c = db.commit(sha)
        cur_flat = db.flat(c.tree)
        cur_lines = split_lines(db.raw(cur_flat[path_here][1]))
        remaining = list(zip(idxs, positions))
        for p in c.parents:
            if not remaining:
                break
            pflat = db.flat(db.commit(p).tree)
            ppath = path_here
            if ppath not in pflat:
                ren = [ch for ch in compare(pflat, cur_flat, renames=True, db=db) if ch.status == "R" and
                       ch.new_path == path_here]
                if ren:
                    ppath = ren[0].old_path
                else:
                    continue
            plines = split_lines(db.raw(pflat[ppath][1]))
            # map current positions to parent positions through unchanged lines
            mapping = {}
            i_old = i_new = 0
            for c1, n1, c2, n2 in diff_lines(plines, cur_lines):
                while i_new < c2:
                    mapping[i_new] = i_old
                    i_new += 1
                    i_old += 1
                i_new = c2 + n2
                i_old = c1 + n1
            while i_new < len(cur_lines):
                mapping[i_new] = i_old
                i_new += 1
                i_old += 1
            passed = [(i, mapping[pos]) for i, pos in remaining if pos in mapping]
            if passed:
                prev = pending.get(p)
                if prev:
                    prev[1].extend(i for i, _ in passed)
                    prev[2].extend(q for _, q in passed)
                else:
                    pending[p] = (ppath, [i for i, _ in passed], [q for _, q in passed])
            passed_set = {i for i, _ in passed}
            remaining = [(i, pos) for i, pos in remaining if i not in passed_set]
        for i, pos in remaining:
            owners[i] = (sha, path_here, pos)
    lo, hi = 0, len(lines)
    if o.get("L"):
        spec = o.get("L")
        m = re.match(r"^(\d+),(\+?\d+)$", spec)
        if m:
            lo = int(m.group(1)) - 1
            hi = lo + int(m.group(2)[1:]) if m.group(2).startswith("+") else int(m.group(2))
        else:
            m = re.match(r"^:(.+)$", spec)
            if m:
                r = _function_range(lines, m.group(1))
                if r:
                    lo, hi = r
        hi = min(hi, len(lines))
    rows = []
    roots = {s for s in order if not db.commit(s).parents}
    aw = max((len(db.commit(owners[i][0]).author.name) for i in range(lo, hi) if owners[i]), default=0)
    nw = len(str(hi))
    show_name = o.get("show-name") or any(owners[i] and owners[i][1] != rel for i in range(lo, hi))
    for i in range(lo, hi):
        sha, pth, pos = owners[i] or (head, rel, i)
        c = db.commit(sha)
        boundary = sha in roots and not o.get("root")
        hid = ("^" + sha[:7]) if boundary else sha[:8]
        who = c.author.email if o.get("show-email") else c.author.name
        when = fmt_date(c.author, "iso", repo.now())
        name = f" {pth:<{max(len(x[1]) for x in owners if x) if owners else 0}}" if show_name else ""
        if o.get("s"):
            rows.append(f"{hid}{name} {i + 1:>{nw}}) {lines[i].rstrip(chr(10))}")
        else:
            who_s = f"<{who}>" if o.get("show-email") else who
            rows.append(f"{hid}{name} ({who_s:<{aw + (2 if o.get('show-email') else 0)}} {when} {i + 1:>{nw}}) "
                        f"{lines[i].rstrip(chr(10))}")
    g.out("\n".join(rows) + ("\n" if rows else ""))
    return 0


# ---------------------------------------------------------------------------- shortlog, reflog, describe

@command("shortlog")
def git_shortlog(g, args):
    o = parse(args, "summary|s numbered|n email|e all no-merges group= committer|c")
    repo = g.repo()
    db = repo.odb
    revs = o.args or (["--all"] if o.get("all") else ["HEAD"])
    if o.get("all"):
        inc = [repo.peel(v) for v in repo.ref_names("refs/").values()] + [repo.head()]
        exc = []
    else:
        inc, exc, _ = rev_list_args(repo, revs)
    commits = db.ancestors([x for x in inc if x and db.kind(x) == "commit"], exc)
    groups: dict = {}
    for sha in commits:
        c = db.commit(sha)
        if o.get("no-merges") and len(c.parents) > 1:
            continue
        who = c.committer if o.get("committer") else c.author
        key = f"{who.name} <{who.email}>" if o.get("email") else who.name
        groups.setdefault(key, []).append(c.subject)
    items = list(groups.items())
    if o.get("numbered"):
        items.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    else:
        items.sort(key=lambda kv: kv[0])
    out = []
    for name, subs in items:
        if o.get("summary"):
            out.append(f"{len(subs):>6}\t{name}\n")
        else:
            out.append(f"{name} ({len(subs)}):\n" + "".join(f"      {s}\n" for s in reversed(subs)) + "\n")
    g.out("".join(out))
    return 0


@command("reflog")
def git_reflog(g, args):
    sub = args[0] if args and args[0] in ("show", "expire", "delete", "exists", "list") else "show"
    rest = args[1:] if args and args[0] in ("show", "expire", "delete", "exists", "list") else args
    repo = g.repo()
    db = repo.odb
    if sub == "exists":
        return 0 if repo.exists(f"logs/{rest[0]}", True) else 1
    if sub in ("expire", "delete"):
        return 0
    o = parse(rest, "n|max-count= all date= oneline pretty|format=", numbers=True)
    ref = o.args[0] if o.args else "HEAD"
    full = "HEAD" if ref == "HEAD" else (dwim_ref(repo, ref)[0] or ref)
    entries = repo.reflog(full)
    color = Colors(g.color())
    deco = decorations(repo) if g.ctx.stdout_is_tty() else None
    shown_name = ref if ref == "HEAD" else full[len("refs/heads/"):] if full.startswith("refs/heads/") else ref
    out = []
    n = int(o.get("n")) if o.get("n") else None
    for i, (old, new, who, msg) in enumerate(entries):
        if n is not None and i >= n:
            break
        d = deco_text(deco.get(new, []), color) if deco else ""
        out.append(color("33", db.abbrev(new) if db.has(new) else new[:7]) + d + f" {shown_name}@{{{i}}}: {msg}\n")
    g.out("".join(out))
    return 0


@command("describe")
def git_describe(g, args):
    o = parse(args, "tags all always long abbrev= dirty=? exact-match match= exclude= contains first-parent")
    repo = g.repo()
    db = repo.odb
    target_spec = o.args[0] if o.args else "HEAD"
    target = resolve(repo, target_spec, "commit")
    tags = {}
    for name, sha in repo.tags().items():
        annotated = db.kind(sha) == "tag"
        if not annotated and not o.get("tags") and not o.get("all"):
            continue
        if o.get("match"):
            import fnmatch
            if not fnmatch.fnmatch(name, o.get("match")):
                continue
        tags.setdefault(repo.peel(sha), []).append((name, annotated))
    if not tags:
        if o.get("always"):
            g.out(db.abbrev(target) + "\n")
            return 0
        raise GitError("fatal: No names found, cannot describe anything." if not repo.tags() else
                       "fatal: No annotated tags can describe '" + target + "'.\nHowever, there were unannotated "
                       "tags: try --tags.")
    # nearest tagged ancestor: fewest commits not reachable from the tag
    best = None
    anc = db.ancestors([target])
    for sha in anc:
        if sha in tags:
            depth = len(db.reachable([target]) - db.reachable([sha]))
            if best is None or depth < best[1]:
                best = (sha, depth)
    dirty = ""
    if o.get("dirty") is not None and repo.worktree:
        from .status import collect
        if not collect(repo, untracked="no").clean:
            dirty = o.get("dirty") if isinstance(o.get("dirty"), str) else "-dirty"
    if best is None:
        if o.get("always"):
            g.out(db.abbrev(target) + dirty + "\n")
            return 0
        raise GitError(f"fatal: No tags can describe '{target}'.\nTry --always, or create some tags.")
    sha, depth = best
    name = sorted(tags[sha], key=lambda x: (not x[1], x[0]))[0][0]
    if o.get("exact-match") and depth:
        raise GitError(f"fatal: no tag exactly matches '{target}'")
    if depth == 0 and not o.get("long"):
        g.out(name + dirty + "\n")
    else:
        g.out(f"{name}-{depth}-g{db.abbrev(target)}{dirty}\n")
    return 0


@command("rev-list")
def git_rev_list(g, args):
    o = parse(args, "count all max-count|n= reverse oneline merges no-merges first-parent topo-order "
                    "date-order skip= since= until= left-right objects parents abbrev-commit pretty|format=",
              numbers=True)
    repo = g.repo()
    db = repo.odb
    if o.get("all"):
        inc, exc = [repo.peel(v) for v in repo.ref_names("refs/").values()] + [repo.head()], []
    else:
        if not o.args:
            raise Usage("usage: git rev-list [<options>] <commit>... [--] [<path>...]")
        inc, exc, _ = rev_list_args(repo, o.args)
    commits = walk(repo, [x for x in inc if x and db.kind(x) == "commit"], exc, bool(o.get("first-parent")))
    if o.get("topo-order"):
        commits = topo_sort(db, commits)
    if o.get("merges"):
        commits = [c for c in commits if len(db.commit(c).parents) > 1]
    if o.get("no-merges"):
        commits = [c for c in commits if len(db.commit(c).parents) <= 1]
    if o.get("skip"):
        commits = commits[int(o.get("skip")):]
    if o.get("max-count"):
        commits = commits[:int(o.get("max-count"))]
    if o.get("reverse"):
        commits = commits[::-1]
    if o.get("count"):
        g.out(f"{len(commits)}\n")
        return 0
    out = []
    for c in commits:
        if o.get("oneline"):
            out.append(f"{db.abbrev(c)} {db.commit(c).subject}\n")
        elif o.get("parents"):
            out.append(" ".join([c] + list(db.commit(c).parents)) + "\n")
        else:
            out.append((db.abbrev(c) if o.get("abbrev-commit") else c) + "\n")
    g.out("".join(out))
    return 0


# ---------------------------------------------------------------------------- grep

@command("grep")
def git_grep(g, args):
    o = parse(args, "line-number|n ignore-case|i files-with-matches|l|name-only count|c word-regexp|w e= "
                    "extended-regexp|E fixed-strings|F basic-regexp|G invert-match|v untracked cached "
                    "no-index heading break show-function|p context|C= after-context|A= before-context|B= "
                    "quiet|q all-match max-depth= column only-matching|o and or not perl-regexp|P h H "
                    "full-name null|z files-without-match|L recurse-submodules color=? no-color threads=")
    repo = g.repo()
    db = repo.odb
    pats = o.all("e")
    rest = list(o.args)
    if not pats:
        if not rest:
            raise Usage("usage: git grep [<options>] [-e] <pattern> [<rev>...] [[--] <path>...]")
        pats = [rest.pop(0)]
    trees = []
    paths = list(o.paths or [])
    for r in rest:
        if not paths and try_resolve(repo, r, "tree") is not None:
            trees.append(r)
        else:
            paths.append(r)
    flags = re.I if o.get("ignore-case") else 0
    rxs = []
    for p in pats:
        if o.get("fixed-strings"):
            p = re.escape(p)
        elif not o.get("extended-regexp") and not o.get("perl-regexp"):
            p = p.replace(r"\|", "|").replace(r"\(", "(").replace(r"\)", ")").replace(r"\+", "+").replace(r"\?", "?")
        if o.get("word-regexp"):
            p = rf"\b(?:{p})\b"
        rxs.append(re.compile(p, flags))
    match = pathspec_matcher(g, repo, paths)
    color = Colors(g.color("grep") and not o.get("no-color"))
    sources = []
    if trees:
        for t in trees:
            flat = db.flat(resolve(repo, t, "tree"))
            for p in sorted(flat):
                if flat[p][0] in ("160000",):
                    continue
                sources.append((f"{t}:{p}", p, db.raw(flat[p][1])))
    else:
        if repo.bare:
            raise GitError("fatal: this operation must be run in a work tree")
        files = repo.index.stage0()
        if o.get("untracked"):
            from .status import collect
            st = collect(repo, untracked="all")
            files = dict(files, **{p: None for p in st.untracked})
        for p in sorted(files):
            if o.get("cached") and files[p] is not None:
                data = db.raw(files[p].sha)
            else:
                node = repo.wt_node(p)
                if node is None or node.type != "f":
                    continue
                data = node.data
            sources.append((g.shown(repo, p), p, data))
    hit_any = False
    out = []
    for shown, rel, data in sources:
        if match and not match(rel):
            continue
        raw = bytes(data)
        if b"\0" in raw[:8000]:
            if any(r.search(raw.decode("latin-1")) for r in rxs):
                out.append(f"Binary file {shown} matches\n")
                hit_any = True
            continue
        lines = raw.decode("utf-8", "surrogateescape").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        hits = []
        for n, ln in enumerate(lines, 1):
            ok = any(r.search(ln) for r in rxs)
            if o.get("invert-match"):
                ok = not ok
            if ok:
                hits.append((n, ln))
        if not hits:
            if o.get("files-without-match"):
                out.append(color("35", shown) + "\n")
            continue
        hit_any = True
        if o.get("quiet"):
            return 0
        if o.get("files-with-matches"):
            out.append(color("35", shown) + "\n")
            continue
        if o.get("files-without-match"):
            continue
        if o.get("count"):
            out.append(color("35", shown) + color("36", ":") + f"{len(hits)}\n")
            continue
        if o.get("heading"):
            out.append(color("35", shown) + "\n")
        for n, ln in hits:
            body = ln
            if color.on and not o.get("invert-match"):
                for r in rxs:
                    body = r.sub(lambda m: color("1;31", m.group(0)), body)
            if o.get("only-matching"):
                for r in rxs:
                    for m in r.finditer(ln):
                        pre = "" if o.get("heading") else color("35", shown) + color("36", ":")
                        num = (color("32", str(n)) + color("36", ":")) if o.get("line-number") else ""
                        out.append(pre + num + color("1;31", m.group(0)) + "\n")
                continue
            pre = "" if o.get("heading") else color("35", shown) + color("36", ":")
            num = (color("32", str(n)) + color("36", ":")) if o.get("line-number") else ""
            out.append(pre + num + body + "\n")
        if o.get("break"):
            out.append("\n")
    g.out("".join(out))
    return 0 if hit_any else 1


# ---------------------------------------------------------------------------- bisect

@command("bisect", wt=True)
def git_bisect(g, args):
    repo = g.repo(wt=True)
    db = repo.odb
    sub = args[0] if args else "help"
    rest = args[1:]
    terms_bad = (repo.read("BISECT_TERMS") or "bad\ngood\n").split("\n")
    bad_t, good_t = terms_bad[0] or "bad", terms_bad[1] if len(terms_bad) > 1 and terms_bad[1] else "good"
    if sub in ("new",):
        sub = bad_t
    if sub in ("old",):
        sub = good_t

    def log(line):
        cur = repo.read("BISECT_LOG") or ""
        repo.write("BISECT_LOG", cur + line + "\n")
    if sub == "start":
        o = parse(rest, "term-new|term-bad= term-old|term-good= no-checkout first-parent")
        if repo.exists("BISECT_START"):
            _bisect_reset(g, repo, quiet=True)
        start = repo.head_branch() or repo.head()
        repo.write("BISECT_START", (start or "") + "\n")
        if o.get("term-new") or o.get("term-old"):
            repo.write("BISECT_TERMS", f"{o.get('term-new') or 'bad'}\n{o.get('term-old') or 'good'}\n")
            bad_t, good_t = o.get("term-new") or "bad", o.get("term-old") or "good"
        repo.write("BISECT_LOG", "git bisect start" + (" " + " ".join(rest) if rest else "") + "\n")
        revs = o.args
        if revs:
            b = resolve(repo, revs[0], "commit")
            repo.update_ref(f"refs/bisect/{bad_t}", b)
            for x in revs[1:]:
                gg = resolve(repo, x, "commit")
                repo.update_ref(f"refs/bisect/{good_t}-{gg}", gg)
        return (yield from run_maybe(_bisect_next(g, repo, bad_t, good_t)))
    if not repo.exists("BISECT_START") and sub not in ("help", "log", "reset"):
        raise GitError("You need to start by \"git bisect start\"\n\nDo you want me to do it for you [Y/n]? ", 1)
    if sub == bad_t or sub == good_t or sub == "skip":
        revs = rest or ["HEAD"]
        for r in revs:
            sha = resolve(repo, r, "commit")
            if sub == bad_t:
                repo.update_ref(f"refs/bisect/{bad_t}", sha)
            elif sub == good_t:
                repo.update_ref(f"refs/bisect/{good_t}-{sha}", sha)
            else:
                repo.update_ref(f"refs/bisect/skip-{sha}", sha)
            log(f"# {sub}: [{sha}] {db.commit(sha).subject}")
            log(f"git bisect {sub} {sha}")
        return (yield from run_maybe(_bisect_next(g, repo, bad_t, good_t)))
    if sub == "reset":
        _bisect_reset(g, repo, rest[0] if rest else None)
        return 0
    if sub == "log":
        g.out(repo.read("BISECT_LOG") or "")
        return 0
    if sub in ("visualize", "view"):
        from .cmd_history import git_log
        return git_log(g, ["--oneline", "--decorate"])
    if sub == "terms":
        g.out(f"Your current terms are {good_t} for the old state\nand {bad_t} for the new state.\n")
        return 0
    if sub == "run":
        if not rest:
            raise GitError("error: bisect run failed: no command provided.", 1)
        cmd = " ".join(__import__("shlex").quote(x) for x in rest)
        while True:
            g.out(f"running  {cmd}\n")
            g.ctx._flush_nb()
            st = yield from g.ctx.run_shell(cmd)
            if st == 125:
                sub2 = "skip"
            elif st == 0:
                sub2 = good_t
            elif st < 128:
                sub2 = bad_t
            else:
                raise GitError(f"bisect run failed: exit code {st} from '{cmd}' is < 0 or >= 128", 1)
            sha = repo.head()
            if sub2 == bad_t:
                repo.update_ref(f"refs/bisect/{bad_t}", sha)
            elif sub2 == good_t:
                repo.update_ref(f"refs/bisect/{good_t}-{sha}", sha)
            else:
                repo.update_ref(f"refs/bisect/skip-{sha}", sha)
            log(f"# {sub2}: [{sha}] {db.commit(sha).subject}")
            log(f"git bisect {sub2} {sha}")
            done = yield from run_maybe(_bisect_next(g, repo, bad_t, good_t))
            if done == "found":
                g.out("bisect found first bad commit\n")
                return 0
            if done != 0:
                return done
    raise Usage("usage: git bisect [help|start|bad|good|new|old|terms|skip|next|reset|visualize|view|replay|log|run]")


def _bisect_steps(n: int) -> int:
    """git's estimate_bisect_steps."""
    if n < 3:
        return 0
    e = 1
    k = 0
    while e * 2 <= n:
        e *= 2
        k += 1
    x = n - e
    return k if e < 3 * x else k - 1


def _bisect_reset(g, repo, target=None, quiet=False):
    start = (repo.read("BISECT_START") or "").strip()
    for name in list(repo.ref_names("refs/bisect/")):
        repo.delete_ref(name)
    for f in ("BISECT_START", "BISECT_LOG", "BISECT_TERMS", "BISECT_EXPECTED_REV", "BISECT_ANCESTORS_OK",
              "BISECT_NAMES", "BISECT_RUN"):
        repo.remove(f)
    dest = target or start
    if not dest:
        return
    from .switching import switch_to
    before = repo.head()
    if repo.resolve_ref(f"refs/heads/{dest}"):
        switch_to(g, repo, repo.resolve_ref(f"refs/heads/{dest}"), branch=dest, quiet=True, force=True)
        if not quiet:
            if before and before != repo.head():
                c = repo.odb.commit(before)
                g.err(f"Previous HEAD position was {repo.odb.abbrev(before)} {c.subject}\n")
            g.err(f"Switched to branch '{dest}'\n")
    else:
        sha = try_resolve(repo, dest, "commit")
        if sha:
            switch_to(g, repo, sha, detach_name=dest, quiet=True, force=True)


def _bisect_next(g, repo, bad_t, good_t):
    db = repo.odb
    bad = repo.resolve_ref(f"refs/bisect/{bad_t}")
    goods = [v for k, v in repo.ref_names("refs/bisect/").items() if k.startswith(f"refs/bisect/{good_t}-")]
    skips = {v for k, v in repo.ref_names("refs/bisect/").items() if k.startswith("refs/bisect/skip-")}
    if not bad and not goods:
        g.out("status: waiting for both good and bad commits\n")
        return 0
    if not bad:
        g.out(f"status: waiting for {bad_t} commit, {len(goods)} {good_t} commit known\n")
        return 0
    if not goods:
        g.out(f"status: waiting for {good_t} commit(s), {bad_t} commit known\n")
        return 0
    cands = [c for c in db.ancestors([bad], goods)]
    if len(cands) == 1 or (cands and all(c in skips for c in cands[1:]) and cands[0] == bad):
        c = db.commit(bad)
        g.out(f"{bad} is the first {bad_t} commit\n")
        from .cmd_history import commit_header
        parent_tree = db.commit(c.parents[0]).tree if c.parents else EMPTY_TREE
        changes = compare(db.flat(parent_tree), db.flat(c.tree), renames=True, db=db)
        rows = []
        for ch in changes:
            a, d = numstat(ch, db)
            rows.append((stat_name(ch), a or 0, d or 0, None))
        g.out(commit_header(repo, bad, "medium", repo.now(), None, False) + "\n" +
              "\n".join(stat_block(rows, 80)) + "\n")
        return "found"
    # pick the commit splitting the candidates in half
    pool = [c for c in cands if c not in skips and c != bad]
    if not pool:
        g.out("There are only 'skip'ped commits left to test.\nThe first bad commit could be any of:\n" +
              "".join(f"{c}\n" for c in cands) + "We cannot bisect more!\n")
        return 2
    n = len(cands)
    cset = set(cands)
    best, best_score, best_reach = None, -1, 0
    for c in pool:
        reaches = len(db.reachable([c]) & cset)
        score = min(reaches, n - reaches)
        if score > best_score:
            best, best_score, best_reach = c, score, reaches
    after = n - best_reach - 1
    steps = _bisect_steps(n)
    from .switching import switch_to
    switch_to(g, repo, best, detach_name=best, quiet=True, force=False)
    repo.write("BISECT_EXPECTED_REV", best + "\n")
    g.out(f"Bisecting: {after} revision{'s' if after != 1 else ''} left to test after this (roughly {steps} "
          f"step{'s' if steps != 1 else ''})\n[{best}] {db.commit(best).subject}\n")
    return 0

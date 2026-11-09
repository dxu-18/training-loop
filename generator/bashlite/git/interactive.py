"""git add -p (and restore / reset / checkout -p): hunk by hunk, with git's prompts. Answers are read from the
terminal a line at a time (interactive.singleKey is off by default), so they're typed and followed by Enter."""

from __future__ import annotations

from .diff import Colors, hunks, split_lines

HELP = """y - {v} this hunk
n - do not {v} this hunk
q - quit; do not {v} this hunk or any of the remaining ones
a - {v} this hunk and all later hunks in the file
d - do not {v} this hunk or any of the later hunks in the file
j - leave this hunk undecided, see next undecided hunk
J - leave this hunk undecided, see next hunk
s - split the current hunk into smaller hunks
e - manually edit the current hunk
? - print help
"""
VERBS = {"stage": "Stage", "unstage": "Unstage", "discard": "Discard"}


def _split(h):
    """Split a hunk where its changes are separated by context lines (git's 's')."""
    s1, n1, s2, n2, body = h
    i = 0
    lead = []
    while i < len(body) and body[i][0] == " ":
        lead.append(body[i])
        i += 1
    out = []
    start_old = s1 + len(lead)
    start_new = s2 + len(lead)
    while i < len(body):
        ch = []
        while i < len(body) and body[i][0] != " ":
            ch.append(body[i])
            i += 1
        trail = []
        while i < len(body) and body[i][0] == " ":
            trail.append(body[i])
            i += 1
        if not ch:
            break
        sub = lead + ch + trail
        o = sum(1 for t, _ in sub if t != "+")
        n = sum(1 for t, _ in sub if t != "-")
        out.append((start_old - len(lead), o, start_new - len(lead), n, sub))
        start_old += sum(1 for t, _ in ch if t == "-") + len(trail)
        start_new += sum(1 for t, _ in ch if t == "+") + len(trail)
        lead = trail
    return out if len(out) > 1 else [h]


def _apply(base, chosen):
    """Apply the chosen hunks (old-file coordinates) to `base` lines."""
    out = list(base)
    offset = 0
    for s1, n1, s2, n2, body in sorted(chosen, key=lambda h: h[0]):
        old = [t for tag, t in body if tag != "+"]
        new = [t for tag, t in body if tag != "-"]
        pos = s1 + offset
        out[pos:pos + len(old)] = new
        offset += len(new) - len(old)
    return out


def _read_answer(g):
    rd = g.ctx.reader("-")
    line = yield from rd.readline()
    if not line:
        return None
    return line.decode("utf-8", "replace").strip()


def _show_hunk(g, h, color):
    s1, n1, s2, n2, body = h
    r1 = f"{s1 + 1 if n1 else s1}" + (f",{n1}" if n1 != 1 else "")
    r2 = f"{s2 + 1 if n2 else s2}" + (f",{n2}" if n2 != 1 else "")
    lines = [color("36", f"@@ -{r1} +{r2} @@")]
    for tag, text in body:
        t = text.rstrip("\n")
        lines.append(color("31", "-" + t) if tag == "-" else color("32", "+" + t) if tag == "+" else " " + t)
    g.out("\n".join(lines) + "\n")


def _session(g, files, mode):
    """files: [(path, old bytes, new bytes, header)] -> {path: bytes with the chosen hunks applied to old}."""
    color = Colors(g.color("diff"))
    verb = VERBS[mode]
    where = " from worktree" if mode == "discard" else ""
    results = {}
    quit_all = False
    for path, old, new, header in files:
        if quit_all:
            break
        a, b = split_lines(old), split_lines(new)
        hs = hunks(a, b, 3)
        if not hs:
            continue
        g.out(color("1", header) + "\n")
        decision = [None] * len(hs)
        i = 0
        while i < len(hs):
            if decision[i] is not None:
                i += 1
                continue
            _show_hunk(g, hs[i], color)
            opts = "y,n,q,a,d" + (",K" if i else "") + (",j,J" if i < len(hs) - 1 else "") + \
                (",s" if len(_split(hs[i])) > 1 else "") + ",e,?"
            g.out(color("1;34", f"({i + 1}/{len(hs)}) {verb} this hunk{where} [{opts}]? "))
            g.ctx._flush_nb()
            ans = yield from _read_answer(g)
            if ans is None:
                quit_all = True
                break
            ans = ans[:1]
            if ans == "y":
                decision[i] = True
                i += 1
            elif ans == "n":
                decision[i] = False
                i += 1
            elif ans in ("a", "d"):
                for k in range(i, len(hs)):
                    if decision[k] is None:
                        decision[k] = ans == "a"
                i = len(hs)
            elif ans == "q":
                for k in range(i, len(hs)):
                    if decision[k] is None:
                        decision[k] = False
                quit_all = True
                break
            elif ans == "s":
                parts = _split(hs[i])
                if len(parts) > 1:
                    g.out(color("1;34", f"Split into {len(parts)} hunks.") + "\n")
                    hs[i:i + 1] = parts
                    decision[i:i + 1] = [None] * len(parts)
                else:
                    g.out(color("1;31", "Sorry, cannot split this hunk") + "\n")
            elif ans in ("j", "J"):
                i += 1
            elif ans == "K" and i:
                i -= 1
            elif ans == "e":
                g.out(color("1;31", "Sorry, editing a hunk by hand isn't available here: use y, n or s.") + "\n")
            else:
                g.out(color("1;31", HELP.format(v=verb.lower())))
        chosen = [h for h, d in zip(hs, decision) if d]
        if chosen:
            results[path] = "".join(_apply(a, chosen)).encode("utf-8", "surrogateescape")
    return results


def _header(repo, path, old_sha, new_sha, mode):
    db = repo.odb
    a = db.abbrev(old_sha) if db.has(old_sha) else old_sha[:7]
    return f"diff --git a/{path} b/{path}\nindex {a}..{new_sha[:7]} {mode}\n--- a/{path}\n+++ b/{path}"


def add_patch(g, repo, specs):
    """git add -p: choose hunks of (index -> working tree) to stage."""
    from .cli import pathspec_matcher
    from .objects import hash_object
    from .repo import Entry
    from .status import collect
    st = collect(repo, untracked="no")
    match = pathspec_matcher(g, repo, specs) if specs else None
    idx = repo.index
    files = []
    for p in sorted(st.unstaged):
        if (match and not match(p)) or st.unstaged[p] != "M":
            continue
        e = idx.get(p)
        new = repo.file_bytes(repo.wt_node(p))
        files.append((p, repo.odb.raw(e.sha), new, _header(repo, p, e.sha, hash_object("blob", new), e.mode_str)))
    if not files:
        g.out("No changes.\n")
        return 0
    res = yield from _session(g, files, "stage")
    for p, data in res.items():
        idx.add(Entry(p, repo.store("blob", data), idx.get(p).mode))
    repo.write_index()
    return 0


def restore_patch(g, repo, specs, staged, source=None):
    """restore -p / checkout -p (discard hunks from the working tree), reset -p (unstage hunks)."""
    from .cli import pathspec_matcher
    from .repo import Entry
    from .revparse import resolve
    from .status import collect
    db = repo.odb
    idx = repo.index
    match = pathspec_matcher(g, repo, specs) if specs else None
    files = []
    if staged:
        head = db.flat(resolve(repo, source, "tree") if source else repo.head_tree())
        for p, e in sorted(idx.stage0().items()):
            h = head.get(p)
            if (match and not match(p)) or h is None or h[1] == e.sha:
                continue
            files.append((p, db.raw(e.sha), db.raw(h[1]), _header(repo, p, h[1], e.sha, e.mode_str)))
        if not files:
            g.out("No changes.\n")
            return 0
        res = yield from _session(g, files, "unstage")
        for p, data in res.items():
            idx.add(Entry(p, repo.store("blob", data), idx.get(p).mode))
        repo.write_index()
        return 0
    st = collect(repo, untracked="no")
    for p in sorted(st.unstaged):
        if (match and not match(p)) or st.unstaged[p] != "M":
            continue
        e = idx.get(p)
        cur = repo.file_bytes(repo.wt_node(p))
        files.append((p, cur, db.raw(e.sha), _header(repo, p, e.sha, "0" * 40, e.mode_str)))
    if not files:
        g.out("No changes.\n")
        return 0
    res = yield from _session(g, files, "discard")
    for p, data in res.items():
        g.fs.write_file(repo.wt_path(p), data, "/", 0o644, g.ctx.uid, g.ctx.proc.gid)
    return 0

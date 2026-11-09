"""Three-way merges as git does them: a file merge (xdiff/xmerge.c: changes on both sides of the base that overlap
or touch conflict, the same change on both sides merges, conflicts are trimmed of lines both sides agree on), with
markers in merge.conflictStyle merge | diff3 | zdiff3, and a tree merge (merge-ort: modify/delete, add/add, binary
files, a rename on one side with edits on the other) that fills the index's conflict stages."""

from __future__ import annotations

from .diff import is_binary, split_lines, similarity
from .objects import GitError
from .xdiff import diff_lines


def merge_text(base, ours, theirs, label_ours="HEAD", label_theirs="theirs", label_base="base",
               style="merge", favor=None, marker=7):
    """-> (merged bytes, number of conflicts)."""
    b, o, t = split_lines(base), split_lines(ours), split_lines(theirs)
    c1 = diff_lines(b, o)
    c2 = diff_lines(b, t)
    # walk both change lists in base order
    out: list[str] = []
    conflicts = 0
    i = j = 0
    pos = 0     # next base line not yet emitted
    o_off = 0   # ours index = base index + o_off (outside changes)
    t_off = 0

    def emit_base(upto):
        nonlocal pos
        while pos < upto:
            out.append(b[pos])
            pos += 1
    while i < len(c1) or j < len(c2):
        if j >= len(c2) or (i < len(c1) and c1[i][0] + c1[i][1] < c2[j][0]):
            s, n, s2, n2 = c1[i]
            emit_base(s)
            out.extend(o[s2:s2 + n2])
            pos = s + n
            i += 1
            continue
        if i >= len(c1) or c2[j][0] + c2[j][1] < c1[i][0]:
            s, n, s2, n2 = c2[j]
            emit_base(s)
            out.extend(t[s2:s2 + n2])
            pos = s + n
            j += 1
            continue
        # overlapping (or touching) changes: grow the region until it stands alone
        start = min(c1[i][0], c2[j][0])
        end = max(c1[i][0] + c1[i][1], c2[j][0] + c2[j][1])
        gi, gj = [c1[i]], [c2[j]]
        i += 1
        j += 1
        grew = True
        while grew:
            grew = False
            if i < len(c1) and c1[i][0] <= end:
                end = max(end, c1[i][0] + c1[i][1])
                gi.append(c1[i])
                i += 1
                grew = True
            if j < len(c2) and c2[j][0] <= end:
                end = max(end, c2[j][0] + c2[j][1])
                gj.append(c2[j])
                j += 1
                grew = True
        emit_base(start)
        so = _side(b, o, gi, start, end)
        st = _side(b, t, gj, start, end)
        pos = end
        if so == st:
            out.extend(so)
            continue
        if favor == "ours":
            out.extend(so)
            continue
        if favor == "theirs":
            out.extend(st)
            continue
        if favor == "union":
            out.extend(so + st)
            continue
        base_part = b[start:end]
        pre = post = 0
        if style != "diff3":
            while pre < min(len(so), len(st)) and so[pre] == st[pre]:
                pre += 1
            while post < min(len(so), len(st)) - pre and so[len(so) - 1 - post] == st[len(st) - 1 - post]:
                post += 1
        out.extend(so[:pre])
        conflicts += 1
        mo = so[pre:len(so) - post]
        mt = st[pre:len(st) - post]
        _ensure_nl(out)
        out.append("<" * marker + (f" {label_ours}" if label_ours else "") + "\n")
        out.extend(mo)
        _ensure_nl(out)
        if style in ("diff3", "zdiff3"):
            out.append("|" * marker + (f" {label_base}" if label_base else "") + "\n")
            out.extend(base_part)
            _ensure_nl(out)
        out.append("=" * marker + "\n")
        out.extend(mt)
        _ensure_nl(out)
        out.append(">" * marker + (f" {label_theirs}" if label_theirs else "") + "\n")
        out.extend(so[len(so) - post:] if post else [])
    emit_base(len(b))
    return "".join(out).encode("utf-8", "surrogateescape"), conflicts


def _ensure_nl(out):
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"


def _side(base, side, changes, start, end):
    """The side's lines for base[start:end] given its changes inside that region."""
    res = []
    pos = start
    for s, n, s2, n2 in changes:
        res.extend(base[pos:s])
        res.extend(side[s2:s2 + n2])
        pos = s + n
    res.extend(base[pos:end])
    return res


class TreeMerge:
    """The result of merging three trees: the merged flat tree (conflicted paths hold the working-tree content),
    conflict stages per path, and git's messages (Auto-merging, CONFLICT ...)."""

    def __init__(self):
        self.flat: dict = {}             # path -> (mode, sha) for the index's stage 0 (clean paths)
        self.worktree: dict = {}         # path -> (mode, data) to write for conflicted paths
        self.stages: dict = {}           # path -> {1: (mode, sha), 2: ..., 3: ...}
        self.messages: list[str] = []
        self.conflicts: list[str] = []


def merge_trees(db, base_tree, ours_tree, theirs_tree, label_ours="HEAD", label_theirs="theirs",
                label_base="merged common ancestors", style="merge", favor=None, renames=True):
    base = db.flat(base_tree) if base_tree else {}
    ours = db.flat(ours_tree)
    theirs = db.flat(theirs_tree)
    res = TreeMerge()
    # renames on one side, with edits on the other: follow the rename
    ren_o = _renames(db, base, ours) if renames else {}
    ren_t = _renames(db, base, theirs) if renames else {}
    paths = sorted(set(base) | set(ours) | set(theirs))
    handled = set()
    for old, new in ren_o.items():          # renamed in ours
        if old in theirs and old not in ren_t:
            handled |= {old, new}
            _merge_path(db, res, new, base[old], ours[new], theirs[old], label_ours, label_theirs, label_base, style,
                        favor, rename_note=(old, new, "ours"))
    for old, new in ren_t.items():          # renamed in theirs
        if old in ours and old not in ren_o:
            handled |= {old, new}
            _merge_path(db, res, new, base[old], ours[old], theirs[new], label_ours, label_theirs, label_base, style,
                        favor, rename_note=(old, new, "theirs"))
    for p in paths:
        if p in handled:
            continue
        _merge_path(db, res, p, base.get(p), ours.get(p), theirs.get(p), label_ours, label_theirs, label_base,
                    style, favor)
    return res


def _renames(db, base, side):
    gone = [p for p in base if p not in side]
    new = [p for p in side if p not in base]
    if not gone or not new or len(gone) * len(new) > 200:
        return {}
    out = {}
    used = set()
    for g in gone:
        best, best_s = None, 0
        for n in new:
            if n in used:
                continue
            if side[n][1] == base[g][1]:
                best, best_s = n, 100
                break
            a, b = db.raw(base[g][1]), db.raw(side[n][1])
            if is_binary(a) or is_binary(b):
                continue
            s = similarity(a, b)
            if s > best_s:
                best, best_s = n, s
        if best is not None and best_s >= 50:
            out[g] = best
            used.add(best)
    return out


def _merge_path(db, res, p, b, o, t, lo, lt, lb, style, favor, rename_note=None):
    if o == t:
        if o is not None:
            res.flat[p] = o
        return
    if b == o:
        if t is not None:
            res.flat[p] = t
        return
    if b == t:
        if o is not None:
            res.flat[p] = o
        return
    # both changed
    if o is None or t is None:
        kept = o if o is not None else t
        who_deleted = lt if o is not None else lo
        who_kept = lo if o is not None else lt
        res.messages.append(f"CONFLICT (modify/delete): {p} deleted in {who_deleted} and modified in {who_kept}.  "
                            f"Version {who_kept} of {p} left in tree.")
        res.conflicts.append(p)
        st = {}
        if b is not None:
            st[1] = b
        if o is not None:
            st[2] = o
        if t is not None:
            st[3] = t
        res.stages[p] = st
        res.worktree[p] = (kept[0], db.raw(kept[1]))
        return
    if o[0] == "160000" or t[0] == "160000":
        res.messages.append(f"CONFLICT (submodule): Merge conflict in {p}")
        res.conflicts.append(p)
        res.stages[p] = {k: v for k, v in ((1, b), (2, o), (3, t)) if v is not None}
        return
    bd = db.raw(b[1]) if b else b""
    od, td = db.raw(o[1]), db.raw(t[1])
    mode = o[0] if (b and o[0] != b[0]) else t[0]
    if is_binary(bd) or is_binary(od) or is_binary(td):
        if favor == "ours":
            res.flat[p] = o
            return
        if favor == "theirs":
            res.flat[p] = t
            return
        res.messages.append(f"warning: Cannot merge binary files: {p} ({lo} vs. {lt})")
        res.messages.append(f"Auto-merging {p}")
        res.messages.append(f"CONFLICT ({'add/add' if b is None else 'content'}): Merge conflict in {p}")
        res.conflicts.append(p)
        res.stages[p] = {k: v for k, v in ((1, b), (2, o), (3, t)) if v is not None}
        res.worktree[p] = (o[0], od)
        return
    merged, n = merge_text(bd, od, td, lo, lt, lb, style, favor)
    res.messages.append(f"Auto-merging {p}")
    if n == 0:
        res.flat[p] = (mode, db.put("blob", merged))
        return
    res.messages.append(f"CONFLICT ({'add/add' if b is None else 'content'}): Merge conflict in {p}")
    res.conflicts.append(p)
    res.stages[p] = {k: v for k, v in ((1, b), (2, o), (3, t)) if v is not None}
    res.worktree[p] = (mode, merged)


def check_clean_for_merge(repo, paths_changing) -> None:
    """git refuses to merge over uncommitted changes to the files the merge touches."""
    from .status import collect
    st = collect(repo)
    dirty = sorted(p for p in paths_changing if p in st.unstaged or p in st.staged)
    if dirty:
        raise GitError("error: Your local changes to the following files would be overwritten by merge:\n" +
                       "".join(f"\t{p}\n" for p in dirty) +
                       "Please commit your changes or stash them before you merge.\nAborting", 1)

"""What changed where: HEAD vs the index (staged), the index vs the working tree (not staged), untracked and
ignored files, unmerged paths. Index stat data that still matches a file means the file is unchanged (no hashing),
as in git; a mismatch re-hashes the file and refreshes the stat data when the content turns out to be the same."""

from __future__ import annotations

from .diff import compare


class Status:
    def __init__(self):
        self.staged: dict = {}         # path -> "A" | "M" | "D" | "T" | ("R", old)
        self.unstaged: dict = {}       # path -> "M" | "D" | "T"
        self.untracked: list = []
        self.ignored: list = []
        self.unmerged: dict = {}       # path -> two-letter code (UU, AA, DU, UD, ...)
        self.renames: dict = {}        # new path -> old path (staged renames)
        self.changes_staged: list = []

    @property
    def clean(self) -> bool:
        return not (self.staged or self.unstaged or self.unmerged)


def worktree_flat(repo, refresh: bool = True, paths=None):
    """path -> (mode, sha) for tracked files as they are in the working tree (missing files left out)."""
    idx = repo.index
    out = {}
    changed_stat = False
    for path, e in idx.stage0().items():
        if paths is not None and not any(path == q or path.startswith(q.rstrip("/") + "/") or q in ("", ".")
                                         for q in paths):
            out[path] = (e.mode_str, e.sha)
            continue
        if e.flags2 & 0x4000:          # skip-worktree (sparse checkout): as in the index
            out[path] = (e.mode_str, e.sha)
            continue
        node = repo.wt_node(path)
        if node is None:
            continue
        if node.type == "d" and e.mode != 0o160000:
            continue
        if e.mode == 0o160000:          # a submodule: its checked-out commit (an empty, unpopulated one is unchanged)
            if node.type == "d":
                out[path] = (e.mode_str, repo._submodule_head(path) or e.sha)
            continue
        if repo.stat_matches(e, node) and e.mode == repo.file_mode(node):
            out[path] = (e.mode_str, e.sha)
            continue
        got = repo.wt_sha(path)
        if got is None:
            continue
        sha, mode = got
        out[path] = (f"{mode:o}", sha)
        if refresh and sha == e.sha and mode == e.mode and node.type != "d":
            ne = repo.index_entry_for(path, sha, mode, node)
            ne.flags2 = e.flags2
            idx.add(ne)
            changed_stat = True
    if changed_stat and refresh:
        try:
            repo.write_index()
        except Exception:  # noqa: BLE001
            pass
    return out


def collect(repo, untracked: str = "normal", ignored: bool = False, renames: bool = True,
            refresh: bool = True) -> Status:
    st = Status()
    db = repo.odb
    head = db.flat(repo.head_tree())
    idx = repo.index
    conflicts = idx.conflicts()
    index_flat = idx.flat()
    head_cmp = {p: v for p, v in head.items() if p not in conflicts}     # unmerged paths aren't "staged"
    for ch in compare(head_cmp, {p: v for p, v in index_flat.items() if p not in conflicts}, renames=renames, db=db):
        if ch.status == "R":
            st.staged[ch.new_path] = ("R", ch.old_path)
            st.renames[ch.new_path] = ch.old_path
        else:
            st.staged[ch.path] = ch.status
        st.changes_staged.append(ch)
    for p, stages in conflicts.items():
        code = {frozenset({1, 2, 3}): "UU", frozenset({2, 3}): "AA", frozenset({1, 2}): "UD",
                frozenset({1, 3}): "DU", frozenset({2}): "AU", frozenset({3}): "UA",
                frozenset({1}): "DD"}.get(frozenset(stages), "UU")
        st.unmerged[p] = code
    if not repo.bare:
        wt = worktree_flat(repo, refresh=refresh)
        for p, e in index_flat.items():
            if p in conflicts:
                continue
            w = wt.get(p)
            if w is None:
                st.unstaged[p] = "D"
            elif w != e:
                st.unstaged[p] = "T" if (w[0] == "120000") != (e[0] == "120000") else "M"
        if untracked != "no" or ignored:
            files = repo.worktree_files(include_ignored=True)
            tracked = {p for p, _ in idx.entries}
            unt, ign = [], []
            for p in sorted(files):
                if p in tracked:
                    continue
                node = files[p]
                is_dir = node.type == "d"
                if repo.is_ignored(p, is_dir):
                    ign.append(p)
                else:
                    unt.append(p + ("/" if is_dir else ""))
            if untracked == "all":
                st.untracked = unt
            elif untracked != "no":
                st.untracked = collapse_dirs(unt, tracked)
            if ignored:
                st.ignored = collapse_dirs(ign, tracked)
    return st


def collapse_dirs(paths, tracked):
    """git status shows an untracked directory as dir/ when nothing inside it is tracked."""
    out = []
    tracked_dirs = set()
    for t in tracked:
        parts = t.split("/")
        for i in range(1, len(parts)):
            tracked_dirs.add("/".join(parts[:i]))
    seen = set()
    for p in paths:
        parts = p.rstrip("/").split("/")
        shown = p
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            if d not in tracked_dirs:
                shown = d + "/"
                break
        if shown not in seen:
            seen.add(shown)
            out.append(shown)
    return out

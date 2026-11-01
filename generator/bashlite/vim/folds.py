"""Folds: manual (zf, :fold) and foldmethod=indent. Closed folds show as one line and move as one line."""

from __future__ import annotations

from .buffer import Fold
from .engine import VimError
from .motions import Fail
from .text import indent_of


def _method(v, w):
    return v.opt("foldmethod", win=w)


def indent_folds(v, w):
    """Folds from the indentation (each level of shiftwidth one level deeper). -> [(start, end, level)]."""
    b = w.buf
    lines = b.lines
    sw = v.sw(b)
    levels = []
    for i, line in enumerate(lines):
        if line.strip():
            levels.append(indent_of(line, v.ts(b)) // max(1, sw))
        else:
            levels.append(None)
    # blank lines take the level of the surrounding lines (the smaller of the two neighbours)
    n = len(lines)
    for i in range(n):
        if levels[i] is None:
            prev = next((levels[j] for j in range(i - 1, -1, -1) if levels[j] is not None), 0)
            nxt = next((levels[j] for j in range(i + 1, n) if levels[j] is not None), 0)
            levels[i] = min(prev, nxt) if False else (prev if nxt is None else min(prev, nxt))
    out = []
    maxl = max(levels) if levels else 0
    for lev in range(1, maxl + 1):
        i = 0
        while i < n:
            if levels[i] >= lev:
                j = i
                while j + 1 < n and levels[j + 1] >= lev:
                    j += 1
                out.append((i, j, lev))
                i = j + 1
            else:
                i += 1
    return out


def all_folds(v, w):
    """[(start, end, level, key)] for the window, outermost first."""
    b = w.buf
    if not v.opt("foldenable", win=w):
        return []
    if _method(v, w) == "indent":
        return [(s, e, lev, ("i", s, e)) for s, e, lev in indent_folds(v, w)]
    out = []

    def walk(folds, lev):
        for f in sorted(folds, key=lambda f: f.start):
            out.append((f.start, f.end, lev, f))
            walk(f.nested, lev + 1)
    walk(b.folds, 1)
    return out


def is_closed(v, w, fold):
    s, e, lev, key = fold
    st = getattr(w, "fold_state", None)
    if st is None:
        w.fold_state = st = {}
    if isinstance(key, Fold):
        if id(key) in st:
            return st[id(key)]
        return key.closed if v.opt("foldlevel", win=w) < lev else False
    k = (s, e)
    if k in st:
        return st[k]
    return lev > v.opt("foldlevel", win=w)


def closed_fold_at(v, l, w):
    """The outermost closed fold containing line l: (start, end), or None."""
    for f in all_folds(v, w):
        s, e, lev, key = f
        if s <= l <= e and is_closed(v, w, f):
            return (s, e)
    return None


def fold_level_at(v, w, l):
    return max((f[2] for f in all_folds(v, w) if f[0] <= l <= f[1]), default=0)


def create_fold(v, l1, l2):
    w = v.win
    if _method(v, w) not in ("manual", "marker"):
        raise VimError("E350: Cannot create fold with current 'foldmethod'")
    b = w.buf
    f = Fold(l1, l2, True)
    # nest it: inside an existing fold, or around existing ones
    def place(folds):
        for g in folds:
            if g.start <= l1 and l2 <= g.end and (g.start, g.end) != (l1, l2):
                return place(g.nested)
        inner = [g for g in folds if l1 <= g.start and g.end <= l2]
        for g in inner:
            folds.remove(g)
        f.nested = inner
        folds.append(f)
    place(b.folds)
    st = getattr(w, "fold_state", {})
    w.fold_state = st
    st[id(f)] = True
    v.used["zf"] += 1


def reset_folds(v):
    w = v.win
    w.fold_state = {}


def _folds_at(v, w, l):
    return [f for f in all_folds(v, w) if f[0] <= l <= f[1]]


def fold_command(v, key, n, rng=None):
    w = v.win
    b = w.buf
    st = getattr(w, "fold_state", None)
    if st is None:
        w.fold_state = st = {}
    l = w.cursor[0]
    v.used[key] += 1

    def k(f):
        return id(f[3]) if isinstance(f[3], Fold) else (f[0], f[1])

    lines = range(rng[0], rng[1] + 1) if rng else [l]
    here = []
    for ln in lines:
        for f in _folds_at(v, w, ln):
            if f not in here:
                here.append(f)
    if key in ("zR", "zM", "zr", "zm", "zE", "zn", "zN", "zi", "zx", "zX", "zj", "zk", "zv"):
        allf = all_folds(v, w) if key not in ("zn", "zN", "zi") else []
        maxl = max((f[2] for f in all_folds(v, w)), default=0)
        if key == "zR":
            w.opts["foldlevel"] = maxl
            st.clear()
        elif key == "zM":
            w.opts["foldlevel"] = 0
            st.clear()
        elif key == "zr":
            w.opts["foldlevel"] = min(maxl, v.opt("foldlevel", win=w) + (n or 1))
            st.clear()
        elif key == "zm":
            w.opts["foldlevel"] = max(0, v.opt("foldlevel", win=w) - (n or 1))
            st.clear()
        elif key == "zE":
            if _method(v, w) not in ("manual", "marker"):
                raise VimError("E352: Cannot erase folds with current 'foldmethod'")
            b.folds = []
            st.clear()
        elif key == "zn":
            w.opts["foldenable"] = False
        elif key == "zN":
            w.opts["foldenable"] = True
        elif key == "zi":
            w.opts["foldenable"] = not v.opt("foldenable", win=w)
        elif key in ("zx", "zX"):
            st.clear()
        elif key == "zv":
            for f in _folds_at(v, w, l):
                st[k(f)] = False
        elif key in ("zj", "zk"):
            if key == "zj":
                starts = sorted(f[0] for f in allf if f[0] > l)
                if not starts:
                    raise Fail()
                v.set_cursor(starts[0])
            else:
                ends = sorted(f[1] for f in allf if f[1] < l)
                if not ends:
                    raise Fail()
                v.set_cursor(ends[-1])
        return
    if not here:
        if key in ("zd", "zD", "zo", "zc", "za", "zO", "zC", "zA"):
            raise VimError("E490: No fold found")
        return
    if key in ("zo", "zO"):
        if key == "zO":
            for f in here:
                st[k(f)] = False
        else:
            # open the outermost closed fold (one level)
            closed = [f for f in here if is_closed(v, w, f)]
            for f in closed[:max(1, n or 1)]:
                st[k(f)] = False
            if not closed:
                pass
        return
    if key in ("zc", "zC"):
        if key == "zC":
            for f in here:
                st[k(f)] = True
        else:
            # close the innermost open fold
            opened = [f for f in here if not is_closed(v, w, f)]
            if opened:
                st[k(opened[-1])] = True
        f = closed_fold_at(v, l, w)
        if f:
            v.win.cursor = [f[0], v.win.cursor[1]]
        return
    if key in ("za", "zA"):
        cl = closed_fold_at(v, l, w)
        if cl:
            for f in here:
                if is_closed(v, w, f) and (key == "zA" or (f[0], f[1]) == cl or True):
                    st[k(f)] = False
                    if key == "za":
                        break
        else:
            target = here if key == "zA" else here[-1:]
            for f in target:
                st[k(f)] = True
            f = closed_fold_at(v, l, w)
            if f:
                v.win.cursor = [f[0], v.win.cursor[1]]
        return
    if key in ("zd", "zD"):
        if _method(v, w) not in ("manual", "marker"):
            raise VimError("E351: Cannot delete fold with current 'foldmethod'")
        target = here[-1][3] if key == "zd" else here[0][3]

        def remove(folds):
            for g in folds:
                if g is target:
                    folds.remove(g)
                    if key == "zd":
                        folds.extend(g.nested)
                    return True
                if remove(g.nested):
                    return True
            return False
        remove(b.folds)
        return
    if key == "zF":
        cnt = n or 1
        create_fold(v, l, min(len(b.lines) - 1, l + cnt - 1))
        return


def fold_text(v, w, s, e):
    """+--  5 lines: def train(model):------ (Vim's foldtext())."""
    b = w.buf
    lev = max((f[2] for f in all_folds(v, w) if f[0] == s and f[1] == e), default=1)
    first = b.lines[s].strip()
    n = e - s + 1
    return f"+-{'-' * lev}{n:3d} lines: {first}"

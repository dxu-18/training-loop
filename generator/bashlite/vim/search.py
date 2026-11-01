"""/ ? n N * #: searching the buffer with Vim regular expressions."""

from __future__ import annotations

from .engine import VimError
from .regex import RegexError, compile_


def compiled(v, pat, smartcase=True):
    try:
        return compile_(pat, v.opt("ignorecase"), v.opt("smartcase") and smartcase, v.last_sub or "", v.opt("magic"))
    except RegexError as e:
        raise VimError(str(e) if str(e).startswith("E") else f"E486: Pattern not found: {pat}") from e


def offsets(lines):
    offs = [0]
    for line in lines:
        offs.append(offs[-1] + len(line) + 1)
    return offs


def pos_of(offs, off):
    lo, hi = 0, len(offs) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offs[mid] <= off:
            lo = mid
        else:
            hi = mid - 1
    return lo, off - offs[lo]


def all_matches(v, cre, lines):
    text = "\n".join(lines)
    out = []
    pos = 0
    while pos <= len(text):
        m = cre.rx.search(text, pos)
        if m is None:
            break
        s, e = cre.span(m)
        out.append((s, e))
        pos = max(m.end(), m.start() + 1) if m.end() == m.start() or cre.has_zs else m.end()
        if cre.has_zs:
            pos = max(m.start() + 1, s + 1)
    return out, text


def search(v, pat, d, count, start, off="", wrap=None, smartcase=True, show=True, echo=None):
    """From `start`, the count-th match in direction d. -> (line, col). Sets messages like Vim."""
    b = v.buf
    lines = b.lines
    cre = compiled(v, pat, smartcase)
    matches, text = all_matches(v, cre, lines)
    offs = offsets(lines)
    if wrap is None:
        wrap = v.opt("wrapscan")
    cur = offs[start[0]] + start[1]
    if off and off[:1] == "e":
        # with an "e" offset, the next match is searched from the end of the current one
        pass
    starts = [s for s, e in matches]
    pos = cur
    wrapped = False
    found = None
    for _ in range(count):
        if d > 0:
            nxt = [s for s in starts if s > pos]
            if not nxt:
                if not wrap:
                    raise VimError(f"E385: Search hit BOTTOM without match for: {pat}")
                if not starts:
                    raise VimError(f"E486: Pattern not found: {pat}")
                nxt = starts
                wrapped = True
            found = nxt[0]
        else:
            prv = [s for s in starts if s < pos]
            if not prv:
                if not wrap:
                    raise VimError(f"E384: Search hit TOP without match for: {pat}")
                if not starts:
                    raise VimError(f"E486: Pattern not found: {pat}")
                prv = starts
                wrapped = True
            found = prv[-1]
        pos = found
    if found is None:
        raise VimError(f"E486: Pattern not found: {pat}")
    l, c = pos_of(offs, found)
    # the match may start on a line break (a pattern starting with \n): Vim puts the cursor on that line's end
    if off:
        end = [e for s, e in matches if s == found][0]
        l, c = apply_offset(lines, offs, found, end, off)
    v.no_hl = False
    if show:
        if wrapped:
            v.message("search hit BOTTOM, continuing at TOP" if d > 0 else "search hit TOP, continuing at BOTTOM", "warn")
        else:
            v.message(echo or ("/" if d > 0 else "?") + pat)
    if c >= len(lines[l]) and c > 0:
        c = len(lines[l]) - 1 if lines[l] else 0
    return l, c


def apply_offset(lines, offs, s, e, off):
    import re
    m = re.fullmatch(r"([esb]?)([+-]?\d*)", off)
    if not m:
        return pos_of(offs, s)
    kind, num = m.group(1), m.group(2)
    n = int(num) if num not in ("", "+", "-") else (1 if num == "+" else -1 if num == "-" else 0)
    if kind == "":
        l, _ = pos_of(offs, s)
        l = max(0, min(len(lines) - 1, l + n))
        from .text import first_nonblank
        return l, first_nonblank(lines[l])
    if kind == "e":
        p = max(s, e - 1) + n
    else:
        p = s + n
    p = max(0, min(offs[-1] - 1, p))
    return pos_of(offs, p)


def match_at_or_after(v, pat, pos, d=1):
    """gn: the match under the cursor, else the next one. -> ((l, c), (l, c) inclusive)."""
    lines = v.buf.lines
    cre = compiled(v, pat)
    matches, text = all_matches(v, cre, lines)
    if not matches:
        return None
    offs = offsets(lines)
    cur = offs[pos[0]] + pos[1]
    pick = None
    if d > 0:
        for s, e in matches:
            if s <= cur < max(e, s + 1) or s >= cur:
                pick = (s, e)
                break
        if pick is None and v.opt("wrapscan"):
            pick = matches[0]
    else:
        for s, e in reversed(matches):
            if s <= cur:
                pick = (s, e)
                break
        if pick is None and v.opt("wrapscan"):
            pick = matches[-1]
    if pick is None:
        return None
    s, e = pick
    return pos_of(offs, s), pos_of(offs, max(s, e - 1))


def count_matches(v, pat, lines=None):
    cre = compiled(v, pat)
    m, _ = all_matches(v, cre, lines if lines is not None else v.buf.lines)
    return len(m)

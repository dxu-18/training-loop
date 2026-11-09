"""Line diffs the way git computes them (a port of libxdiff as git uses it, xdiff/xprepare.c and xdiffi.c): trim
common ends, set aside lines without a match on the other side (and runs of multi-match lines among them), Myers'
middle-snake split, then slide change groups with the indent heuristic (diff.indentHeuristic, on by default) so
hunks land where git puts them. -> changes [(old start, old count, new start, new count)], 0-based."""

from __future__ import annotations

MAX_EQLIMIT = 1024
SIMSCAN_WINDOW = 100
KPDIS_RUN = 4
LINE_MAX = 1 << 62

START_OF_FILE_PENALTY = 1
END_OF_FILE_PENALTY = 21
TOTAL_BLANK_WEIGHT = -30
POST_BLANK_WEIGHT = 6
RELATIVE_INDENT_PENALTY = -4
RELATIVE_INDENT_WITH_BLANK_PENALTY = 10
RELATIVE_OUTDENT_PENALTY = 24
RELATIVE_OUTDENT_WITH_BLANK_PENALTY = 17
RELATIVE_DEDENT_PENALTY = 23
RELATIVE_DEDENT_WITH_BLANK_PENALTY = 17
INDENT_WEIGHT = 60
INDENT_HEURISTIC_MAX_SLIDING = 100
MAX_INDENT = 200
MAX_BLANKS = 20


def bogosqrt(n: int) -> int:
    i = 1
    while n > 0:
        i <<= 1
        n >>= 2
    return i


def _clean_mmatch(dis, i, s, e):
    if i - s > SIMSCAN_WINDOW:
        s = i - SIMSCAN_WINDOW
    if e - i > SIMSCAN_WINDOW:
        e = i + SIMSCAN_WINDOW
    r, rdis0, rpdis0 = 1, 0, 1
    while i - r >= s:
        if not dis[i - r]:
            rdis0 += 1
        elif dis[i - r] == 2:
            rpdis0 += 1
        else:
            break
        r += 1
    if rdis0 == 0:
        return False
    r, rdis1, rpdis1 = 1, 0, 1
    while i + r <= e:
        if not dis[i + r]:
            rdis1 += 1
        elif dis[i + r] == 2:
            rpdis1 += 1
        else:
            break
        r += 1
    if rdis1 == 0:
        return False
    rdis1 += rdis0
    rpdis1 += rpdis0
    return rpdis1 * KPDIS_RUN < (rpdis1 + rdis1)


def _split(ha1, off1, lim1, ha2, off2, lim2, kvdf, kvdb):
    dmin, dmax = off1 - lim2, lim1 - off2
    fmid, bmid = off1 - off2, lim1 - lim2
    odd = (fmid - bmid) & 1
    fmin = fmax = fmid
    bmin = bmax = bmid
    kvdf[fmid] = off1
    kvdb[bmid] = lim1
    ec = 0
    while True:
        ec += 1
        if fmin > dmin:
            fmin -= 1
            kvdf[fmin - 1] = -1
        else:
            fmin += 1
        if fmax < dmax:
            fmax += 1
            kvdf[fmax + 1] = -1
        else:
            fmax -= 1
        d = fmax
        while d >= fmin:
            if kvdf.get(d - 1, -1) >= kvdf.get(d + 1, -1):
                i1 = kvdf.get(d - 1, -1) + 1
            else:
                i1 = kvdf.get(d + 1, -1)
            i2 = i1 - d
            while i1 < lim1 and i2 < lim2 and ha1[i1] == ha2[i2]:
                i1 += 1
                i2 += 1
            kvdf[d] = i1
            if odd and bmin <= d <= bmax and kvdb.get(d, LINE_MAX) <= i1:
                return i1, i2
            d -= 2
        if bmin > dmin:
            bmin -= 1
            kvdb[bmin - 1] = LINE_MAX
        else:
            bmin += 1
        if bmax < dmax:
            bmax += 1
            kvdb[bmax + 1] = LINE_MAX
        else:
            bmax -= 1
        d = bmax
        while d >= bmin:
            if kvdb.get(d - 1, LINE_MAX) < kvdb.get(d + 1, LINE_MAX):
                i1 = kvdb.get(d - 1, LINE_MAX)
            else:
                i1 = kvdb.get(d + 1, LINE_MAX) - 1
            i2 = i1 - d
            while i1 > off1 and i2 > off2 and ha1[i1 - 1] == ha2[i2 - 1]:
                i1 -= 1
                i2 -= 1
            kvdb[d] = i1
            if not odd and fmin <= d <= fmax and i1 <= kvdf.get(d, -1):
                return i1, i2
            d -= 2


def _recs_cmp(ha1, rindex1, rchg1, off1, lim1, ha2, rindex2, rchg2, off2, lim2):
    stack = [(off1, lim1, off2, lim2)]
    while stack:
        off1, lim1, off2, lim2 = stack.pop()
        while off1 < lim1 and off2 < lim2 and ha1[off1] == ha2[off2]:
            off1 += 1
            off2 += 1
        while off1 < lim1 and off2 < lim2 and ha1[lim1 - 1] == ha2[lim2 - 1]:
            lim1 -= 1
            lim2 -= 1
        if off1 == lim1:
            for k in range(off2, lim2):
                rchg2[rindex2[k]] = 1
        elif off2 == lim2:
            for k in range(off1, lim1):
                rchg1[rindex1[k]] = 1
        else:
            i1, i2 = _split(ha1, off1, lim1, ha2, off2, lim2, {}, {})
            # the second half is processed after the first (push it first)
            stack.append((i1, lim1, i2, lim2))
            stack.append((off1, i1, off2, i2))


def _get_indent(line: str) -> int:
    ret = 0
    for c in line:
        if c == " ":
            ret += 1
        elif c == "\t":
            ret += 8 - ret % 8
        elif c in "\n\r\f\v":
            continue
        else:
            return ret
        if ret >= MAX_INDENT:
            return MAX_INDENT
    return -1        # all whitespace


def _measure(recs, split):
    n = len(recs)
    if split >= n:
        eof, indent = True, -1
    else:
        eof, indent = False, _get_indent(recs[split])
    pre_blank, pre_indent = 0, -1
    i = split - 1
    while i >= 0:
        pre_indent = _get_indent(recs[i])
        if pre_indent != -1:
            break
        pre_blank += 1
        if pre_blank == MAX_BLANKS:
            pre_indent = 0
            break
        i -= 1
    post_blank, post_indent = 0, -1
    i = split + 1
    while i < n:
        post_indent = _get_indent(recs[i])
        if post_indent != -1:
            break
        post_blank += 1
        if post_blank == MAX_BLANKS:
            post_indent = 0
            break
        i += 1
    return eof, indent, pre_blank, pre_indent, post_blank, post_indent


def _score(m, s):
    eof, indent_, pre_blank, pre_indent, post_blank_, post_indent = m
    eff, pen = s
    if pre_indent == -1 and pre_blank == 0:
        pen += START_OF_FILE_PENALTY
    if eof:
        pen += END_OF_FILE_PENALTY
    post_blank = 1 + post_blank_ if indent_ == -1 else 0
    total_blank = pre_blank + post_blank
    pen += TOTAL_BLANK_WEIGHT * total_blank
    pen += POST_BLANK_WEIGHT * post_blank
    indent = indent_ if indent_ != -1 else post_indent
    any_blanks = total_blank != 0
    eff += indent
    if indent == -1 or pre_indent == -1:
        pass
    elif indent > pre_indent:
        pen += RELATIVE_INDENT_WITH_BLANK_PENALTY if any_blanks else RELATIVE_INDENT_PENALTY
    elif indent == pre_indent:
        pass
    else:
        if post_indent != -1 and post_indent > indent:
            pen += RELATIVE_OUTDENT_WITH_BLANK_PENALTY if any_blanks else RELATIVE_OUTDENT_PENALTY
        else:
            pen += RELATIVE_DEDENT_WITH_BLANK_PENALTY if any_blanks else RELATIVE_DEDENT_PENALTY
    return eff, pen


def _score_cmp(a, b):
    ci = (a[0] > b[0]) - (a[0] < b[0])
    return INDENT_WEIGHT * ci + (a[1] - b[1])


class _Group:
    __slots__ = ("start", "end")


def _compact(recs, rchg, rchgo, heuristic=True):
    """xdl_change_compact: slide each group of changed lines in `recs` (rchg) as git does; rchgo is the other side."""
    n = len(recs)
    no = len(rchgo) - 2   # rchg arrays are padded: index -1 and n are sentinels (stored at [n] and [n+1])

    def ch(arr, i):
        return arr[i] if 0 <= i < len(arr) - 2 else 0

    # groups: g on this file, go on the other, walked in step
    def g_init(arr, size):
        start = end = 0
        while ch(arr, end):
            end += 1
        return [start, end]

    def g_next(arr, size, g):
        if g[1] == size:
            return False
        g[0] = g[1] + 1
        g[1] = g[0]
        while ch(arr, g[1]):
            g[1] += 1
        return True

    def g_prev(arr, size, g):
        if g[0] == 0:
            return False
        g[1] = g[0] - 1
        g[0] = g[1]
        while ch(arr, g[0] - 1):
            g[0] -= 1
        return True

    def slide_down(g):
        if g[1] < n and recs[g[0]] == recs[g[1]]:
            rchg[g[0]] = 0
            rchg[g[1]] = 1
            g[0] += 1
            g[1] += 1
            while ch(rchg, g[1]):
                g[1] += 1
            return True
        return False

    def slide_up(g):
        if g[0] > 0 and recs[g[0] - 1] == recs[g[1] - 1]:
            g[0] -= 1
            g[1] -= 1
            rchg[g[0]] = 1
            rchg[g[1]] = 0
            while ch(rchg, g[0] - 1):
                g[0] -= 1
            return True
        return False

    g = g_init(rchg, n)
    go = g_init(rchgo, no)
    while True:
        if g[1] != g[0]:
            while True:
                groupsize = g[1] - g[0]
                end_matching_other = -1
                while slide_up(g):
                    g_prev(rchgo, no, go)
                earliest_end = g[1]
                if go[1] > go[0]:
                    end_matching_other = g[1]
                while True:
                    if not slide_down(g):
                        break
                    g_next(rchgo, no, go)
                    if go[1] > go[0]:
                        end_matching_other = g[1]
                if groupsize == g[1] - g[0]:
                    break
            if g[1] == earliest_end:
                pass
            elif end_matching_other != -1:
                while go[1] == go[0]:
                    slide_up(g)
                    g_prev(rchgo, no, go)
            elif heuristic:
                shift = earliest_end
                if g[1] - groupsize - 1 > shift:
                    shift = g[1] - groupsize - 1
                if g[1] - INDENT_HEURISTIC_MAX_SLIDING > shift:
                    shift = g[1] - INDENT_HEURISTIC_MAX_SLIDING
                best_shift, best = -1, None
                while shift <= g[1]:
                    sc = (0, 0)
                    sc = _score(_measure(recs, shift), sc)
                    sc = _score(_measure(recs, shift - groupsize), sc)
                    if best_shift == -1 or _score_cmp(sc, best) <= 0:
                        best, best_shift = sc, shift
                    shift += 1
                while g[1] > best_shift:
                    slide_up(g)
                    g_prev(rchgo, no, go)
        if not g_next(rchg, n, g):
            break
        g_next(rchgo, no, go)


def diff_lines(a: list[str], b: list[str], heuristic: bool = True) -> list[tuple[int, int, int, int]]:
    """Changes turning a into b: [(i1, n1, i2, n2)] (0-based starts), in order."""
    n1, n2 = len(a), len(b)
    rchg1 = [0] * (n1 + 2)
    rchg2 = [0] * (n2 + 2)
    # trim common prefix and suffix
    pre = 0
    while pre < n1 and pre < n2 and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < n1 - pre and suf < n2 - pre and a[n1 - 1 - suf] == b[n2 - 1 - suf]:
        suf += 1
    dstart1, dend1 = pre, n1 - suf - 1
    dstart2, dend2 = pre, n2 - suf - 1
    # classify lines: how often each appears on the other side
    from collections import Counter
    c1, c2 = Counter(a), Counter(b)
    ids: dict[str, int] = {}

    def hid(line):
        h = ids.get(line)
        if h is None:
            h = len(ids)
            ids[line] = h
        return h

    def keep_side(lines, dstart, dend, other_counts, rchg, size):
        mlim = min(bogosqrt(size), MAX_EQLIMIT)
        dis = [0] * (size + 2)
        for i in range(dstart, dend + 1):
            nm = other_counts.get(lines[i], 0)
            dis[i] = 0 if nm == 0 else (2 if nm >= mlim else 1)
        rindex, ha = [], []
        for i in range(dstart, dend + 1):
            if dis[i] == 1 or (dis[i] == 2 and not _clean_mmatch(dis, i, dstart, dend)):
                rindex.append(i)
                ha.append(hid(lines[i]))
            else:
                rchg[i] = 1
        return rindex, ha

    rindex1, ha1 = keep_side(a, dstart1, dend1, c2, rchg1, n1)
    rindex2, ha2 = keep_side(b, dstart2, dend2, c1, rchg2, n2)
    _recs_cmp(ha1, rindex1, rchg1, 0, len(ha1), ha2, rindex2, rchg2, 0, len(ha2))
    _compact(a, rchg1, rchg2, heuristic)
    _compact(b, rchg2, rchg1, heuristic)
    # the change script
    out = []
    i1 = i2 = 0
    while i1 < n1 or i2 < n2:
        if (i1 < n1 and rchg1[i1]) or (i2 < n2 and rchg2[i2]):
            s1, s2 = i1, i2
            while i1 < n1 and rchg1[i1]:
                i1 += 1
            while i2 < n2 and rchg2[i2]:
                i2 += 1
            out.append((s1, i1 - s1, s2, i2 - s2))
        else:
            i1 += 1
            i2 += 1
    return out

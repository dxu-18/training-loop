"""Motions and text objects over a list of lines.

A motion returns a target (line, col) and its kind: "excl", "incl" or "line" (:help exclusive, :help linewise).
A text object returns (start, end, kind). Failures raise Fail (Vim beeps, and a macro stops).
"""

from __future__ import annotations

from .text import cls, first_nonblank, is_kw


class Fail(Exception):
    """A motion or command that can't be done: beep, abort the macro / mapping."""


def last_col(line: str) -> int:
    return max(0, len(line) - 1)


# ---------------------------------------------------------------- words
# Positions may sit on the end of a line (col == len(line), Vim's NUL), which counts as white space, exactly as in
# Vim's search.c (fwd_word, bck_word, end_word, bckend_word), which these follow line by line.

def gchar(lines, pos):
    l, c = pos
    return lines[l][c] if c < len(lines[l]) else ""


def ccls(lines, pos, big):
    ch = gchar(lines, pos)
    if ch in ("", " ", "\t"):
        return 0
    if big:
        return 1
    return 2 if is_kw(ch) else 1


def inc(lines, pos):
    """-> (pos, 0 moved | 2 moved onto the end of the line | 1 crossed into the next line | -1 at the end)."""
    l, c = pos
    if c < len(lines[l]):
        c += 1
        return (l, c), (0 if c < len(lines[l]) else 2)
    if l < len(lines) - 1:
        return (l + 1, 0), 1
    return pos, -1


def dec(lines, pos):
    """-> (pos, 0 moved | 1 crossed to the previous line's end | -1 at the start)."""
    l, c = pos
    if c > 0:
        return (l, c - 1), 0
    if l > 0:
        return (l - 1, len(lines[l - 1])), 1
    return pos, -1


def fwd_word(lines, pos, count, big=False, eol=False):
    """w / W. eol: an operator's last word ends at the end of its line. -> (pos, ok)."""
    while count > 0:
        count -= 1
        sclass = ccls(lines, pos, big)
        last_line = pos[0] == len(lines) - 1
        pos, i = inc(lines, pos)
        if i == -1 or (i >= 1 and last_line):
            return pos, False
        if i >= 1 and eol and count == 0:
            return pos, True
        if sclass != 0:
            while ccls(lines, pos, big) == sclass:
                pos, i = inc(lines, pos)
                if i == -1 or (i >= 1 and eol and count == 0):
                    return pos, True
        while ccls(lines, pos, big) == 0:
            if pos[1] == 0 and lines[pos[0]] == "":
                break
            pos, i = inc(lines, pos)
            if i == -1 or (i >= 1 and eol and count == 0):
                return pos, True
    return pos, True


def bck_word(lines, pos, count, big=False, stop=False):
    """b / B. -> (pos, ok)."""
    while count > 0:
        count -= 1
        sclass = ccls(lines, pos, big)
        pos, i = dec(lines, pos)
        if i == -1:
            return pos, False
        finished = False
        if not stop or sclass == ccls(lines, pos, big) or sclass == 0:
            while ccls(lines, pos, big) == 0:
                if pos[1] == 0 and lines[pos[0]] == "":
                    finished = True
                    break
                pos, i = dec(lines, pos)
                if i == -1:
                    return pos, True
            if not finished:
                k = ccls(lines, pos, big)
                while ccls(lines, pos, big) == k:
                    pos, i = dec(lines, pos)
                    if i == -1:
                        return pos, True
        if not finished:
            pos, _ = inc(lines, pos)
        stop = False
    return pos, True


def end_word(lines, pos, count, big=False, stop=False, empty=False):
    """e / E (stop: "cw" on a word's last character stays there). -> (pos, ok)."""
    while count > 0:
        count -= 1
        sclass = ccls(lines, pos, big)
        pos, i = inc(lines, pos)
        if i == -1:
            return pos, False
        finished = False
        if ccls(lines, pos, big) == sclass and sclass != 0:
            while ccls(lines, pos, big) == sclass:
                pos, i = inc(lines, pos)
                if i == -1:
                    return pos, False
        elif not stop or sclass == 0:
            while ccls(lines, pos, big) == 0:
                if pos[1] == 0 and lines[pos[0]] == "" and empty:
                    finished = True
                    break
                pos, i = inc(lines, pos)
                if i == -1:
                    return pos, False
            if not finished:
                k = ccls(lines, pos, big)
                while ccls(lines, pos, big) == k:
                    pos, i = inc(lines, pos)
                    if i == -1:
                        return pos, False
        if not finished:
            pos, _ = dec(lines, pos)
        stop = False
    return pos, True


def bckend_word(lines, pos, count, big=False, eol=False):
    """ge / gE. -> (pos, ok)."""
    while count > 0:
        count -= 1
        sclass = ccls(lines, pos, big)
        pos, i = dec(lines, pos)
        if i == -1:
            return pos, False
        if eol and i == 1:
            return pos, True
        if sclass != 0:
            while ccls(lines, pos, big) == sclass:
                pos, i = dec(lines, pos)
                if i == -1 or (eol and i == 1):
                    return pos, True
        while ccls(lines, pos, big) == 0:
            if pos[1] == 0 and lines[pos[0]] == "":
                break
            pos, i = dec(lines, pos)
            if i == -1 or (eol and i == 1):
                return pos, True
    return pos, True


def _next_pos(lines, l, c):
    """One character forward over the whole buffer, skipping line ends. None at the end."""
    if c + 1 < len(lines[l]):
        return l, c + 1
    if l + 1 < len(lines):
        return l + 1, 0
    return None


def _prev_pos(lines, l, c):
    if c > 0:
        return l, c - 1
    if l > 0:
        return l - 1, max(0, len(lines[l - 1]) - 1)
    return None


# ---------------------------------------------------------------- lines and characters

def find_char(line, c, ch, count, kind):
    """f F t T within the line. -> the new column."""
    pos = c
    for i in range(count):
        if kind in "ft":
            j = line.find(ch, pos + 1 + (1 if kind == "t" and i == 0 and False else 0))
            if kind == "t" and j == pos + 1 and i == 0 and False:
                pass
            if j < 0:
                raise Fail()
            pos = j
        else:
            j = line.rfind(ch, 0, pos)
            if j < 0:
                raise Fail()
            pos = j
    if kind == "t":
        return pos - 1
    if kind == "T":
        return pos + 1
    return pos


def find_char_repeat(line, c, ch, count, kind):
    """; and , after t/T don't get stuck on the adjacent character (cpoptions without ;)."""
    if kind == "t":
        start = c + 1
        pos = start
        for _ in range(count):
            j = line.find(ch, pos + 1)
            if j < 0:
                raise Fail()
            pos = j
        return pos - 1
    if kind == "T":
        pos = c - 1
        for _ in range(count):
            j = line.rfind(ch, 0, pos)
            if j < 0:
                raise Fail()
            pos = j
        return pos + 1
    return find_char(line, c, ch, count, kind)


PAIRS = {"(": ")", "[": "]", "{": "}", ")": "(", "]": "[", "}": "{"}


def match_pair(lines, l, c):
    """%: from the first bracket at or after the cursor on this line to its match."""
    line = lines[l]
    j = c
    while j < len(line) and line[j] not in "()[]{}":
        j += 1
    if j >= len(line):
        # also C comments /* */: not needed here
        raise Fail()
    ch = line[j]
    other = PAIRS[ch]
    fwd = ch in "([{"
    depth = 0
    pl, pc = l, j
    while True:
        cur = lines[pl][pc] if pc < len(lines[pl]) else ""
        if cur == ch:
            depth += 1
        elif cur == other:
            depth -= 1
            if depth == 0:
                return pl, pc
        p = _next_pos(lines, pl, pc) if fwd else _prev_pos(lines, pl, pc)
        if p is None:
            raise Fail()
        pl, pc = p
        if not lines[pl]:
            # an empty line: keep going
            pass


def paragraph(lines, l, count, forward):
    """} and { (Vim's findpar): to the next / previous empty line. -> (line, col, inclusive). Fail past the ends."""
    n = len(lines)
    d = 1 if forward else -1
    curr = l
    while count > 0:
        count -= 1
        did_skip = False
        first = True
        while True:
            if lines[curr] != "":
                did_skip = True
            if not first and did_skip and lines[curr] == "":
                break
            curr += d
            if curr < 0 or curr >= n:
                if count:
                    raise Fail()
                curr -= d
                break
            first = False
    if curr == n - 1 and forward and lines[curr] != "":
        return curr, len(lines[curr]) - 1, True
    return curr, 0, False


# ---------------------------------------------------------------- text objects

def tobj_word(lines, l, c, count, around, big=False):
    """iw aw iW aW -> (start, end inclusive). Each count adds one more word, or one more run of white space
    (aw: a word together with the white space after it)."""
    line = lines[l]
    if not line:
        raise Fail()
    c = min(c, len(line) - 1)

    def run_end(i):
        k = cls(line[i], big)
        while i + 1 < len(line) and cls(line[i + 1], big) == k:
            i += 1
        return i

    k = cls(line[c], big)
    s = c
    while s > 0 and cls(line[s - 1], big) == k:
        s -= 1
    e = run_end(c)
    if not around:
        for _ in range(count - 1):
            if e + 1 >= len(line):
                raise Fail()
            e = run_end(e + 1)
        return (l, s), (l, e)
    if k == 0:
        # white space, then the word after it
        for i in range(count):
            if i > 0:
                if e + 1 >= len(line):
                    raise Fail()
                e = run_end(e + 1)
            if e + 1 >= len(line):
                raise Fail() if i == 0 and False else Fail()
            e = run_end(e + 1)
        return (l, s), (l, e)
    for i in range(count):
        if i > 0:
            if e + 1 >= len(line):
                raise Fail()
            e = run_end(e + 1)
        if e + 1 < len(line) and cls(line[e + 1], big) == 0:
            e = run_end(e + 1)
        elif i == count - 1:
            # no white space after the last word: take the white space before the first one
            if e + 1 >= len(line) or cls(line[e + 1], big) != 0:
                while s > 0 and cls(line[s - 1], big) == 0:
                    s -= 1
    return (l, s), (l, e)


def tobj_quote(lines, l, c, q, around):
    """i" a" i' a' i` a` on the cursor's line."""
    line = lines[l]
    pos = [i for i, ch in enumerate(line) if ch == q and (i == 0 or line[i - 1] != "\\")]
    if len(pos) < 2:
        raise Fail()
    start = end = None
    if line[c:c + 1] == q and c in pos:
        k = pos.index(c)
        if k % 2 == 0 and k + 1 < len(pos):
            start, end = pos[k], pos[k + 1]
        elif k % 2 == 1:
            start, end = pos[k - 1], pos[k]
    if start is None:
        for a, b in zip(pos[::2], pos[1::2]):
            if a <= c <= b:
                start, end = a, b
                break
    if start is None:
        # the first quoted string after the cursor
        for a, b in zip(pos[::2], pos[1::2]):
            if a > c:
                start, end = a, b
                break
    if start is None:
        raise Fail()
    if around:
        s, e = start, end
        if e + 1 < len(line) and line[e + 1] in " \t":
            while e + 1 < len(line) and line[e + 1] in " \t":
                e += 1
        else:
            while s > 0 and line[s - 1] in " \t":
                s -= 1
        return (l, s), (l, e), "incl"
    if end == start + 1:
        return (l, start + 1), (l, start), "empty"
    return (l, start + 1), (l, end - 1), "incl"


def _block_around(lines, pos, open_, close, count):
    """The count-th unmatched open_ at or before pos and its match. -> (open pos, close pos) or None."""
    l, c = pos
    cur = lines[l][c] if c < len(lines[l]) else ""
    depth = 0
    p = pos
    if cur == close:
        p = _prev_pos(lines, l, c)
        depth = 0
        start_inside = True
    found = None
    if cur == open_:
        found = pos
        count -= 1
        p = _prev_pos(lines, l, c)
    elif cur == close:
        pass
    while count > 0:
        while p is not None:
            pl, pc = p
            ch = lines[pl][pc] if pc < len(lines[pl]) else ""
            if ch == close:
                depth += 1
            elif ch == open_:
                if depth == 0:
                    found = (pl, pc)
                    break
                depth -= 1
            p = _prev_pos(lines, pl, pc)
        if p is None:
            return None
        count -= 1
        if count > 0:
            p = _prev_pos(lines, *found)
            depth = 0
    if found is None:
        return None
    depth = 0
    p = found
    while p is not None:
        pl, pc = p
        ch = lines[pl][pc] if pc < len(lines[pl]) else ""
        if ch == open_:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                return found, (pl, pc)
        p = _next_pos(lines, pl, pc)
    return None


def tobj_bracket(lines, l, c, open_, close, around, count=1):
    """i( a( i[ i{ i< ... : the block around the cursor, or (Vim 9) the next one on the line."""
    r = _block_around(lines, (l, c), open_, close, count)
    if r is None:
        j = lines[l].find(open_, c)
        if j < 0:
            raise Fail()
        r = _block_around(lines, (l, j), open_, close, 1)
        if r is None:
            raise Fail()
    found, end = r
    if around:
        return found, end, "incl"
    s = _next_pos(lines, *found)
    if s == end or s is None or (found[0] == end[0] and end[1] == found[1] + 1):
        return (found[0], found[1] + 1), found, "empty"
    if found[0] != end[0]:
        after_open = lines[found[0]][found[1] + 1:]
        before_close = lines[end[0]][:end[1]]
        # ( ending its line and ) starting its line: the lines in between, linewise
        if after_open.strip() == "" and before_close.strip() == "":
            if end[0] - found[0] >= 2:
                return (found[0] + 1, 0), (end[0] - 1, max(0, len(lines[end[0] - 1]) - 1)), "line"
            return (found[0], found[1] + 1), found, "empty"
        if after_open == "":
            s = (found[0] + 1, 0)
        if before_close.strip() == "":
            e = (end[0] - 1, len(lines[end[0] - 1]))
            return s, e if lines[e[0]] else (e[0], 0), "incl"
        e = (end[0], end[1] - 1)
        return s, e, "incl"
    return s, (end[0], end[1] - 1), "incl"


def tobj_tag(lines, l, c, around, count=1):
    """it / at: the XML/HTML element around the cursor."""
    import re
    text = "\n".join(lines)
    offs = [0]
    for line in lines:
        offs.append(offs[-1] + len(line) + 1)
    pos = offs[l] + c
    tags = []
    for m in re.finditer(r"<(/?)([A-Za-z][\w:.-]*)[^<>]*?(/?)>", text):
        if m.group(3):
            continue
        tags.append((m.start(), m.end(), m.group(1) == "/", m.group(2)))
    # pair them up
    stack, pairs = [], []
    for t in tags:
        if not t[2]:
            stack.append(t)
        else:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][3] == t[3]:
                    o = stack[k]
                    del stack[k:]
                    pairs.append((o, t))
                    break
    best = [(o, cl) for o, cl in pairs if o[0] <= pos < cl[1]]
    if not best:
        raise Fail()
    best.sort(key=lambda p: p[1][1] - p[0][0])
    o, cl = best[min(count - 1, len(best) - 1)]

    def lc(off):
        for i in range(len(lines)):
            if offs[i + 1] > off:
                return i, off - offs[i]
        return len(lines) - 1, 0
    if around:
        return lc(o[0]), lc(cl[1] - 1), "incl"
    if o[1] == cl[0]:
        s = lc(o[1])
        return s, s, "empty"
    return lc(o[1]), lc(cl[0] - 1), "incl"


def _blank(line):
    return line.strip() == ""


def tobj_paragraph(lines, l, count, around):
    """ip / ap: linewise. A paragraph is a run of non-blank lines, or a run of blank lines. ip: count runs;
    ap: count paragraphs, each with the blank lines after it (or before it, at the end of the file)."""
    n = len(lines)
    blank = _blank(lines[l])
    s = l
    while s > 0 and _blank(lines[s - 1]) == blank:
        s -= 1

    def run_end(i):
        b = _blank(lines[i])
        while i + 1 < n and _blank(lines[i + 1]) == b:
            i += 1
        return i

    e = run_end(l)
    if not around:
        for _ in range(count - 1):
            if e + 1 >= n:
                raise Fail()
            e = run_end(e + 1)
        return (s, 0), (e, 0), "line"
    for i in range(count):
        if i > 0:
            if e + 1 >= n:
                raise Fail()
            e = run_end(e + 1)
        if e + 1 < n:
            e = run_end(e + 1)
        elif i == count - 1 and not blank:
            # no blank lines after the last paragraph: take the ones before it
            while s > 0 and _blank(lines[s - 1]):
                s -= 1
    return (s, 0), (e, 0), "line"


def tobj_sentence(lines, l, c, around):
    """is / as on the cursor's line (sentences end with . ! or ? and a space or the end of the line)."""
    import re
    line = lines[l]
    if not line.strip():
        raise Fail()
    bounds = [0]
    for m in re.finditer(r"[.!?][)\]\"']*(?=[ \t]|$)", line):
        e = m.end()
        while e < len(line) and line[e] in " \t":
            e += 1
        if e < len(line):
            bounds.append(e)
    bounds.append(len(line))
    for a, b in zip(bounds, bounds[1:]):
        if a <= c < b or b == len(line):
            seg_end = b - 1
            text_end = seg_end
            while text_end > a and line[text_end] in " \t":
                text_end -= 1
            if around:
                if text_end < seg_end:
                    return (l, a), (l, seg_end), "incl"
                s = a
                while s > 0 and line[s - 1] in " \t":
                    s -= 1
                return (l, s), (l, seg_end), "incl"
            return (l, a), (l, text_end), "incl"
    raise Fail()


def keyword_at(line, c):
    """The keyword under or after the cursor (for * # K and <C-r><C-w>). -> (start, end exclusive) or None."""
    j = c
    while j < len(line) and not is_kw(line[j]):
        j += 1
    if j >= len(line):
        # no keyword: a non-blank word
        j = c
        while j < len(line) and line[j] in " \t":
            j += 1
        if j >= len(line):
            return None
        e = j
        while e < len(line) and line[e] not in " \t":
            e += 1
        return j, e
    if j == c:
        while j > 0 and is_kw(line[j - 1]):
            j -= 1
    e = j
    while e < len(line) and is_kw(line[e]):
        e += 1
    return j, e

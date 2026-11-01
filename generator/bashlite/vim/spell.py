"""Spell checking: :set spell, ]s [s (next / previous misspelling), z= (suggestions), zg zw (good / wrong word)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .engine import VimError
from .motions import Fail

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@lru_cache(maxsize=1)
def dictionary() -> frozenset:
    p = Path(__file__).with_name("words.txt")
    try:
        return frozenset(w.strip() for w in p.read_text().split() if w.strip())
    except OSError:
        return frozenset()


def _bases(w):
    """Possible dictionary forms of an inflected word (models -> model, stopped -> stop, trying -> try)."""
    out = {w}
    for suf, repl in (("ies", "y"), ("ied", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""),
                      ("ing", "e"), ("ly", ""), ("er", ""), ("er", "e"), ("est", ""), ("ers", ""), ("'s", ""),
                      ("ment", ""), ("ments", ""), ("ness", ""), ("able", ""), ("ation", "e"), ("ations", "e"),
                      ("ation", ""), ("ers", "e")):
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            b = w[:-len(suf)] + repl
            out.add(b)
            if len(b) > 2 and b[-1] == b[-2] and suf in ("ed", "ing", "er", "est"):
                out.add(b[:-1])
    return out


def good(v, word: str) -> bool:
    if len(word) < 2 or word.isupper() and len(word) > 1:
        return True
    low = word.lower()
    if low in v.spell_good or word in v.spell_good:
        return word not in getattr(v, "spell_bad", set())
    if word in getattr(v, "spell_bad", set()) or low in getattr(v, "spell_bad", set()):
        return False
    d = dictionary()
    return any(b in d for b in _bases(low))


def bad_words(v, line):
    """(start, end) of the misspelled words in a line."""
    out = []
    for m in WORD.finditer(line):
        s, e = m.span()
        # part of an identifier (snake_case, digits) isn't checked
        if (s > 0 and (line[s - 1].isdigit() or line[s - 1] == "_")) or (e < len(line) and (line[e].isdigit() or line[e] == "_")):
            continue
        if not good(v, m.group(0)):
            out.append((s, e))
    return out


def next_bad(v, pos, d, count):
    lines = v.buf.lines
    if not v.opt("spell"):
        raise Fail()
    l, c = pos
    hits = []
    for li, line in enumerate(lines):
        for s, e in bad_words(v, line):
            hits.append((li, s))
    if not hits:
        raise Fail()
    for _ in range(count):
        if d > 0:
            nxt = [h for h in hits if h > (l, c)]
            if not nxt:
                v.message("search hit BOTTOM, continuing at TOP", "warn")
                nxt = hits
            l, c = nxt[0]
        else:
            prv = [h for h in hits if h < (l, c)]
            if not prv:
                v.message("search hit TOP, continuing at BOTTOM", "warn")
                prv = hits
            l, c = prv[-1]
    v.used["]s" if d > 0 else "[s"] += 1
    return l, c


def _edits1(w):
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
    inserts = [a + c + b for a, b in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def suggestions(v, word, limit=9):
    d = dictionary() | {w.lower() for w in v.spell_good}
    low = word.lower()
    e1 = _edits1(low)
    first = sorted(w for w in e1 if w in d)
    second = set()
    if len(first) < limit:
        for x in e1:
            for y in _edits1(x):
                if y in d and y not in first:
                    second.add(y)
    # as Vim's scores: swapping two letters is the cheapest edit; then words sharing the first letter, closer in length
    def swapped(w):
        return any(w == low[:i] + low[i + 1] + low[i] + low[i + 2:] for i in range(len(low) - 1))

    def rank(w):
        return (not swapped(w), w[:1] != low[:1], abs(len(w) - len(low)), w)
    out = sorted(first, key=rank) + sorted(second, key=rank)
    if word[:1].isupper():
        out = [w.capitalize() for w in out]
    return out[:limit]


def spell_command(v, key, n):
    w, b = v.win, v.buf
    l, c = w.cursor
    line = b.lines[l]
    m = None
    for mm in WORD.finditer(line):
        if mm.start() <= c < mm.end() or mm.start() >= c:
            m = mm
            break
    if m is None:
        raise VimError("E348: No string under cursor")
    word = m.group(0)
    v.used[key] += 1
    if key in ("zg", "zG"):
        add_word(v, word, True)
        return
    if key in ("zw", "zW"):
        add_word(v, word, False)
        return
    if key.startswith("zu"):
        v.spell_good.discard(word)
        return
    if key == "z=":
        sugg = suggestions(v, word)
        if not sugg:
            v.message(f'Sorry, no suggestions')
            return
        if n:
            if n > len(sugg):
                raise Fail()
            return replace_word(v, l, m.start(), m.end(), sugg[n - 1])
        lines = [f'Change "{word}" to:']
        for i, s in enumerate(sugg):
            lines.append(f'{i + 1:2d} "{s}"')
        lines.append("Type number and <Enter> or click with the mouse (q or empty cancels): ")
        v.pending_prompt = {"kind": "spell", "lines": lines, "sugg": sugg, "span": (l, m.start(), m.end()),
                            "typed": ""}


def replace_word(v, l, s, e, new):
    b = v.buf
    v.mark_change_start()
    line = b.lines[l]
    v.set_line(b, l, line[:s] + new + line[e:])
    v.set_cursor(l, s)
    v.commit_undo()


def add_word(v, word, good_=True):
    if good_:
        v.spell_good.add(word)
        bad = getattr(v, "spell_bad", set())
        bad.discard(word)
        v.message(f'Word \'{word}\' added to ~/.vim/spell/en.utf-8.add')
    else:
        v.spell_bad = getattr(v, "spell_bad", set())
        v.spell_bad.add(word)
        v.spell_good.discard(word)
        v.message(f'Word \'{word}\' added to ~/.vim/spell/en.utf-8.add')

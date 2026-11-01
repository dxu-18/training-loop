"""A buffer: the lines of one file, its undo tree, marks and folds."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UndoState:
    seq: int                      # 0 = the original text
    parent: "UndoState | None"
    lines: tuple
    cursor: tuple                 # where the cursor goes when this state is restored by undo/redo
    time: int                     # ms (the session's clock) when the change was made
    children: list = field(default_factory=list)
    cur_child: int = -1           # the child <C-r> goes to (the most recent one)


class UndoTree:
    def __init__(self, lines, now=0):
        self.root = UndoState(0, None, tuple(lines), (0, 0), now)
        self.cur = self.root
        self.states = [self.root]   # by seq (chronological)
        self.seq_last = 0

    def commit(self, lines, cursor, now):
        """A change happened: a new state after the current one."""
        st = UndoState(self.seq_last + 1, self.cur, tuple(lines), cursor, now)
        self.seq_last += 1
        self.cur.children.append(st)
        self.cur.cur_child = len(self.cur.children) - 1
        self.cur = st
        self.states.append(st)
        return st


@dataclass
class Fold:
    start: int
    end: int
    closed: bool = True
    nested: list = field(default_factory=list)


class Buffer:
    def __init__(self, num: int, name: str, path: str | None, lines: list[str], now: int = 0):
        self.num = num
        self.name = name                 # as shown (relative to the cwd when opened)
        self.path = path                 # absolute; None for scratch buffers
        self.lines = lines or [""]
        self.modified = False
        self.ff = "unix"
        self.eol = True
        self.readonly = False
        self.kind = "file"               # file, help, netrw, quickfix, terminal, scratch
        self.listed = True
        self.loaded = True
        self.marks: dict[str, tuple[int, int]] = {}
        self.undo = UndoTree(self.lines, now)
        self.saved_seq = 0               # undo state that matches the file
        self.last_cursor = (0, 0)        # the '" mark
        self.folds: list[Fold] = []      # manual folds (zf)
        self.opts: dict = {}             # buffer-local options
        self.new_file = False
        self.uline = None                # U: (lnum, text before the latest run of changes on that line)
        self.changed_lines = set()
        self.extra: dict = {}            # netrw / quickfix / terminal data

    # ------------------------------------------------------------ marks follow the text
    def adjust(self, line1: int, line2: int, amount: int, amount_after: int):
        """Like Vim's mark_adjust: lines line1..line2 move by `amount` (None = deleted), later ones by amount_after."""
        def fix(pos):
            l, c = pos
            if line1 <= l <= line2:
                if amount is None:
                    return None
                return (l + amount, c)
            if l > line2:
                return (l + amount_after, c)
            return pos
        for k in list(self.marks):
            p = fix(self.marks[k])
            if p is None:
                if k in "abcdefghijklmnopqrstuvwxyz":
                    del self.marks[k]
                else:
                    self.marks[k] = (max(0, line1 - 1 if line1 > 0 else 0), 0)
            else:
                self.marks[k] = p
        for f in list(self.folds):
            if f.start > line2:
                f.start += amount_after
                f.end += amount_after
            elif f.end >= line1:
                if amount is None and line1 <= f.start and f.end <= line2:
                    self.folds.remove(f)
                    continue
                if amount is None:
                    # deleted lines inside the fold shrink it
                    gone = min(f.end, line2) - max(f.start, line1) + 1
                    if f.start > line1:
                        f.start = line1
                    f.end -= gone
                    if f.end < f.start:
                        self.folds.remove(f)
                else:
                    if f.end > line2:
                        f.end += amount_after
                    elif f.start >= line1:
                        f.start += amount
                        f.end += amount

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.eol else "")

    def set_lines(self, lines):
        self.lines[:] = list(lines) or [""]

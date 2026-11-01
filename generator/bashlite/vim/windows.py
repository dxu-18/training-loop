"""Windows (splits) and tab pages. A layout is a tree: ["leaf", win] | ["row", [..]] (side by side) |
["col", [..]] (stacked)."""

from __future__ import annotations

from .engine import VimError, Window
from .motions import Fail


class WindowMixin:

    # ------------------------------------------------------------ queries

    def windows_of(self, tab):
        out = []

        def walk(node):
            if node[0] == "leaf":
                out.append(node[1])
            else:
                for c in node[1]:
                    walk(c)
        walk(tab["layout"])
        return out

    def all_windows(self, all_tabs=False):
        if all_tabs:
            return [w for t in self.tabs for w in self.windows_of(t)]
        return self.windows_of(self.tab)

    def _find(self, node, win, parent=None):
        if node[0] == "leaf":
            return (node, parent) if node[1] is win else None
        for c in node[1]:
            r = self._find(c, win, node)
            if r:
                return r
        return None

    # ------------------------------------------------------------ geometry

    def layout(self):
        """Give every window of the current tab its place on the screen."""
        top = 1 if len(self.tabs) > 1 else 0
        rows = self.height - 1 - top
        for t in self.tabs:
            wins = self.windows_of(t)
            status = self.opt("laststatus") == 2 or (self.opt("laststatus") == 1 and len(wins) > 1)
            self._place(t["layout"], top, 0, rows, self.width, status)
        for w in self.all_windows(all_tabs=True):
            self.scroll_to_cursor(w) if w.buf.lines else None

    def _place(self, node, row, col, height, width, status):
        if node[0] == "leaf":
            w = node[1]
            w.row, w.col = row, col
            w.width = width
            w.height = max(1, height - (1 if status else 0))
            w.has_status = status
            return
        kids = node[1]
        horiz = node[0] == "row"
        total = width if horiz else height
        seps = len(kids) - 1 if horiz else 0
        avail = total - seps
        req = []
        for k in kids:
            r = None
            if k[0] == "leaf":
                r = k[1].req_width if horiz else k[1].req_height
                if r is not None:
                    r = r + (0 if horiz else (1 if status else 0))
            req.append(r)
        fixed = sum(r for r in req if r is not None)
        free = [i for i, r in enumerate(req) if r is None]
        sizes = list(req)
        rest = max(len(free), avail - fixed)
        if not free:
            # all were sized: the last one takes the remainder
            sizes[-1] = max(1, avail - sum(sizes[:-1]))
        else:
            base, extra = divmod(rest, len(free))
            for j, i in enumerate(free):
                sizes[i] = base + (1 if j < extra else 0)
        pos = col if horiz else row
        for k, sz in zip(kids, sizes):
            sz = max(1, sz)
            if horiz:
                self._place(k, row, pos, height, sz, status)
                pos += sz + 1
            else:
                self._place(k, pos, col, sz, width, status)
                pos += sz

    # ------------------------------------------------------------ splitting and closing

    def split(self, vertical=False, buf=None, size=None, at_end=False):
        cur = self.win
        new = Window(buf or cur.buf)
        new.cursor = list(cur.cursor) if new.buf is cur.buf else [0, 0]
        new.curswant = cur.curswant
        new.topline = cur.topline if new.buf is cur.buf else 0
        new.opts = dict(cur.opts)
        new.jumps = list(cur.jumps)
        new.jumpidx = cur.jumpidx
        new.alt = cur.alt
        if buf is not None and buf is not cur.buf:
            new.alt = cur.buf.num
        kind = "row" if vertical else "col"
        found = self._find(self.tab["layout"], cur)
        node, parent = found
        after = (self.opt("splitright") if vertical else self.opt("splitbelow")) or at_end
        leaf_new = ["leaf", new]
        if parent is not None and parent[0] == kind:
            i = next(j for j, c in enumerate(parent[1]) if c is node)
            parent[1].insert(i + 1 if after else i, leaf_new)
        else:
            old = ["leaf", cur]
            node[0], node[1] = kind, [old, leaf_new] if after else [leaf_new, old]
        for w in self.all_windows():
            w.req_height = None if not hasattr(w, "req_height") or self.opt("equalalways") else w.req_height
            w.req_width = None if not hasattr(w, "req_width") or self.opt("equalalways") else w.req_width
        if size:
            if vertical:
                new.req_width = size
            else:
                new.req_height = size
        self.tab["prev"] = cur
        self.tab["win"] = new
        self.layout()
        self.used["split" if not vertical else "vsplit"] += 1
        return new

    def close_window(self, win, force=False):
        t = self.tab
        wins = self.windows_of(t)
        if len(wins) == 1:
            if len(self.tabs) > 1:
                return self.close_tab(self.tabidx, force)
            raise VimError("E444: Cannot close last window")
        b = win.buf
        found = self._find(t["layout"], win)
        node, parent = found
        i = next(j for j, c in enumerate(parent[1]) if c is node)
        del parent[1][i]
        if len(parent[1]) == 1:
            only = parent[1][0]
            parent[0], parent[1] = only[0], only[1]
        b.last_cursor = tuple(win.cursor)
        if win is t["win"]:
            prev = t.get("prev")
            remaining = self.windows_of(t)
            if prev in remaining:
                nxt = prev
            else:
                nxt = remaining[min(i, len(remaining) - 1)] if remaining else None
            t["win"] = nxt
        for w in self.all_windows():
            if self.opt("equalalways"):
                w.req_height = None
                w.req_width = None
        if b.kind in ("help", "quickfix", "netrw") and not any(w.buf is b for w in self.all_windows(all_tabs=True)):
            b.listed = False if b.kind != "netrw" else b.listed
        if b.kind == "terminal" and not any(w.buf is b for w in self.all_windows(all_tabs=True)):
            self.delete_buffer(b, wipe=True, closing=True)
        self.layout()

    def only_window(self, force=False):
        cur = self.win
        for w in list(self.all_windows()):
            if w is cur:
                continue
            if w.buf.modified and not force and not self.opt("hidden") and \
                    sum(1 for x in self.all_windows() if x.buf is w.buf) == 1:
                continue
            self.close_window(w, force=True)
        self.layout()

    def goto_win(self, w):
        if w is self.win:
            return
        self.win.buf.last_cursor = tuple(self.win.cursor)
        self.tab["prev"] = self.win
        self.tab["win"] = w
        self.term_mode = w.buf.kind == "terminal" and w.buf.extra.get("running", False)

    def resize_win(self, w, n, vertical=False):
        if vertical:
            w.req_width = max(1, min(self.width - 2, n))
        else:
            w.req_height = max(1, min(self.height - 3, n))
        self.layout()

    # ------------------------------------------------------------ CTRL-W

    def window_cmd(self, k, count=None):
        self.used["<C-w>" + k] += 1
        self.used["<C-w>"] += 1
        wins = self.all_windows()
        cur = self.win
        if k in ("s", "S", "<C-s>"):
            self.split(False, size=count)
            return
        if k in ("v", "<C-v>"):
            self.split(True, size=count)
            return
        if k in ("n", "<C-n>"):
            self.split(False, buf=self.new_buffer(), size=count)
            return
        if k in ("q", "<C-q>"):
            from .ex import c_quit
            return c_quit(self, {"bang": False, "arg": "", "naddr": 0, "l1": None, "l2": None, "raw": "q",
                                 "name": "quit", "alias": None})
        if k in ("c", "<C-c>"):
            if len(wins) == 1 and len(self.tabs) == 1:
                raise VimError("E444: Cannot close last window")
            return self.close_window(cur)
        if k in ("o", "<C-o>"):
            return self.only_window()
        if k in ("w", "<C-w>", "W"):
            if len(wins) == 1:
                return
            i = wins.index(cur)
            if count:
                target = wins[min(count, len(wins)) - 1]
            else:
                target = wins[(i + (1 if k != "W" else -1)) % len(wins)]
            return self.goto_win(target)
        if k in ("p", "<C-p>"):
            prev = self.tab.get("prev")
            if prev is None or prev not in wins:
                raise Fail()
            return self.goto_win(prev)
        if k in ("t", "<C-t>"):
            return self.goto_win(wins[0])
        if k in ("b", "<C-b>"):
            return self.goto_win(wins[-1])
        if k in ("h", "j", "k", "l", "<C-h>", "<C-j>", "<C-k>", "<C-l>", "<Left>", "<Down>", "<Up>", "<Right>",
                 "<BS>"):
            d = {"<C-h>": "h", "<C-j>": "j", "<C-k>": "k", "<C-l>": "l", "<Left>": "h", "<Down>": "j", "<Up>": "k",
                 "<Right>": "l", "<BS>": "h"}.get(k, k)
            target = cur
            for _ in range(count or 1):
                nxt = self.neighbor(target, d)
                if nxt is None:
                    break
                target = nxt
            if target is not cur:
                self.goto_win(target)
            return
        if k == "=":
            for w in wins:
                w.req_height = None
                w.req_width = None
            self.layout()
            return
        if k in ("+", "-"):
            n = count or 1
            return self.resize_win(cur, cur.height + (n if k == "+" else -n))
        if k == "_":
            return self.resize_win(cur, count or 1000)
        if k == "|":
            return self.resize_win(cur, count or 1000, vertical=True)
        if k in ("<", ">"):
            n = count or 1
            return self.resize_win(cur, cur.width + (n if k == ">" else -n), vertical=True)
        if k in ("x", "<C-x>", "r", "R"):
            found = self._find(self.tab["layout"], cur)
            node, parent = found
            if parent is None:
                return
            kids = parent[1]
            i = kids.index(node)
            j = i + 1 if i + 1 < len(kids) else i - 1
            if k in ("r", "R"):
                if k == "r":
                    kids.insert(0, kids.pop())
                else:
                    kids.append(kids.pop(0))
            else:
                kids[i], kids[j] = kids[j], kids[i]
            self.layout()
            return
        if k in ("H", "J", "K", "L"):
            if len(wins) == 1:
                return
            self.close_window(cur, force=True)
            t = self.tab
            kind = "row" if k in ("H", "L") else "col"
            first = k in ("H", "K")
            leaf = ["leaf", cur]
            old = t["layout"]
            t["layout"] = [kind, [leaf, old] if first else [old, leaf]]
            t["win"] = cur
            self.layout()
            return
        if k == "T":
            if len(wins) == 1:
                return
            self.close_window(cur, force=True)
            self.tabs.insert(self.tabidx + 1, {"layout": ["leaf", cur], "win": cur, "prev": None})
            self.tabidx += 1
            self.layout()
            return
        if k in ("^", "<C-^>"):
            alt = self.buffers.get(count) if count else self.buffers.get(cur.alt)
            if alt is None:
                raise VimError("E23: No alternate file")
            self.split(False, buf=alt)
            return
        if k in ("f", "F"):
            from .commands import normal_command
            self.split(False)
            return normal_command(self, {"key": "gf", "arg": None, "n": None, "reg": None, "keys": []})
        if k == "N" and cur.buf.kind == "terminal":
            self.term_mode = False
            return
        if k == ":":
            return self.start_cmdline(":")
        if k in ("g",):
            return
        raise Fail()

    def neighbor(self, w, d):
        """The window next to w in direction d (h j k l), by screen position."""
        best = None
        r = w.row + min(w.height - 1, max(0, w.cursor[0] - w.topline))
        c = w.col
        for o in self.all_windows():
            if o is w:
                continue
            if d == "l" and o.col == w.col + w.width + 1 and o.row <= r <= o.row + o.height:
                best = o if best is None or o.row > best.row and o.row <= r else best
            elif d == "h" and o.col + o.width + 1 == w.col and o.row <= r <= o.row + o.height:
                best = o
            elif d == "j" and o.row == w.row + w.height + (1 if w.has_status else 0) and \
                    o.col <= c <= o.col + o.width:
                best = o
            elif d == "k" and o.row + o.height + (1 if o.has_status else 0) == w.row and \
                    o.col <= c <= o.col + o.width:
                best = o
        if best is None:
            # loosest match: any window on that side overlapping
            cands = []
            for o in self.all_windows():
                if o is w:
                    continue
                if d == "l" and o.col > w.col and not (o.row + o.height < w.row or o.row > w.row + w.height):
                    cands.append((o.col, o))
                elif d == "h" and o.col < w.col and not (o.row + o.height < w.row or o.row > w.row + w.height):
                    cands.append((-o.col, o))
                elif d == "j" and o.row > w.row and not (o.col + o.width < w.col or o.col > w.col + w.width):
                    cands.append((o.row, o))
                elif d == "k" and o.row < w.row and not (o.col + o.width < w.col or o.col > w.col + w.width):
                    cands.append((-o.row, o))
            if cands:
                cands.sort(key=lambda x: x[0])
                best = cands[0][1]
        return best

    # ------------------------------------------------------------ buffers in windows

    def edit_buffer(self, b, keep_jumps=False):
        w = self.win
        old = w.buf
        if old is b:
            return
        if not keep_jumps:
            self.setpcmark()
        old.last_cursor = tuple(w.cursor)
        w.last_cursor[old.num] = tuple(w.cursor)
        if old.kind in ("file",) or old.listed:
            w.alt = old.num
        w.buf = b
        b.listed = True if b.kind == "file" else b.listed
        if not b.loaded:
            self.read_into(b)
        pos = w.last_cursor.get(b.num, b.last_cursor)
        l = min(pos[0], len(b.lines) - 1)
        w.cursor = [l, pos[1]]
        w.topline = 0
        self.clamp(w)
        self.scroll_to_cursor(w)
        self.term_mode = b.kind == "terminal" and b.extra.get("running", False)
        self.fire("BufEnter", b)

    def delete_buffer(self, b, wipe=False, closing=False):
        b.listed = False
        b.loaded = False if b.kind == "file" else b.loaded
        others = [x for x in self.buffers.values() if x.listed and x is not b]
        for t in self.tabs:
            for w in self.windows_of(t):
                if w.buf is b:
                    alt = self.buffers.get(w.alt)
                    nxt = alt if alt is not None and alt.listed and alt is not b else (others[0] if others else None)
                    if nxt is None:
                        nxt = self.new_buffer()
                        others = [nxt]
                    if closing:
                        continue
                    keep = self.tab["win"]
                    w.buf = nxt
                    pos = w.last_cursor.get(nxt.num, nxt.last_cursor)
                    w.cursor = [min(pos[0], len(nxt.lines) - 1), pos[1]]
                    self.clamp(w)
        if wipe:
            self.buffers.pop(b.num, None)
        self.term_mode = self.buf.kind == "terminal" and self.buf.extra.get("running", False)

    # ------------------------------------------------------------ tabs

    def new_tab(self, b):
        w = Window(b)
        w.alt = self.win.alt if b is self.buf else self.buf.num
        w.opts = dict(self.win.opts)
        self.tabs.insert(self.tabidx + 1, {"layout": ["leaf", w], "win": w, "prev": None})
        self.tabidx += 1
        self.layout()
        self.used[":tabnew"] += 1

    def goto_tab(self, i):
        if not self.tabs:
            return
        i = max(0, min(len(self.tabs) - 1, i))
        self.win.buf.last_cursor = tuple(self.win.cursor)
        self.tabidx = i
        self.layout()

    def close_tab(self, idx, force=False):
        t = self.tabs[idx]
        if not force:
            for w in self.windows_of(t):
                if w.buf.modified and not self.opt("hidden") and \
                        not any(x.buf is w.buf for tt in self.tabs if tt is not t for x in self.windows_of(tt)):
                    raise VimError("E445: Other window contains changes")
        del self.tabs[idx]
        self.tabidx = min(self.tabidx if idx > self.tabidx else max(0, idx - 1 if idx == len(self.tabs) else idx),
                          len(self.tabs) - 1)
        self.layout()

    def refresh_qf_window(self):
        from .quickfix import refresh
        refresh(self)

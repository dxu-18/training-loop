"""Visual mode: v (characters), V (lines), <C-v> (a block)."""

from __future__ import annotations

from . import keys as K
from .engine import MAXCOL, VimError
from .motions import Fail
from .text import col_at_vcol, first_nonblank, vcol

VNAMES = {"v": "VISUAL", "V": "VISUAL LINE", "\x16": "VISUAL BLOCK"}


class VisualMixin:

    def start_visual(self, mode, count=None):
        w = self.win
        self.vis = {"anchor": (w.cursor[0], w.cursor[1]), "mode": mode, "keys": []}
        self.used[{"v": "v", "V": "V", "\x16": "<C-v>"}[mode]] += 1
        if count and count > 1 and mode == "v":
            w.cursor = [w.cursor[0], min(len(self.lines[w.cursor[0]]) - 1, w.cursor[1] + count - 1)]
        elif count and count > 1 and mode == "V":
            w.cursor = [min(len(self.lines) - 1, w.cursor[0] + count - 1), w.cursor[1]]
        self.message(f"-- {VNAMES[mode]} --", "mode")

    def reselect(self):
        lv = self.last_vis
        if lv is None:
            raise Fail()
        b = self.buf
        s, e = lv["start"], lv["end"]
        s = (min(s[0], len(b.lines) - 1), s[1])
        e = (min(e[0], len(b.lines) - 1), e[1])
        self.vis = {"anchor": s, "mode": lv["mode"], "keys": []}
        self.win.cursor = [e[0], e[1]]
        if lv["mode"] == "\x16" and lv.get("want") is not None:
            self.win.curswant = lv["want"]
        self.clamp()
        self.used["gv"] += 1
        self.message(f"-- {VNAMES[lv['mode']]} --", "mode")

    def vis_area(self):
        """-> (start, end, kind, block): kind line / incl; block = (vcol1, vcol2) for <C-v>."""
        vis = self.vis
        a = vis["anchor"]
        c = tuple(self.win.cursor)
        lines = self.lines
        mode = vis["mode"]
        if mode == "V":
            s, e = (min(a[0], c[0]), 0), (max(a[0], c[0]), 0)
            return s, e, "line", None
        if mode == "v":
            s, e = (a, c) if a <= c else (c, a)
            # after $ the selection takes the line break too (also once `o` has made that end the anchor)
            if lines[e[0]] and (e[1] >= len(lines[e[0]]) or (self.win.curswant >= MAXCOL and e == c)):
                e = (e[0], len(lines[e[0]]))
            return s, e, "incl", None
        ts = self.ts()
        l1, l2 = min(a[0], c[0]), max(a[0], c[0])
        va = vcol(lines[a[0]], a[1], ts) if lines[a[0]] else 0
        vc = vcol(lines[c[0]], c[1], ts) if lines[c[0]] else 0
        va2 = va + (self._cw(lines[a[0]], a[1]) - 1)
        vc2 = vc + (self._cw(lines[c[0]], c[1]) - 1)
        left = min(va, vc)
        right = max(va2, vc2)
        if self.win.curswant >= MAXCOL:
            right = MAXCOL
        return (l1, 0), (l2, 0), "block", (left, right)

    def _cw(self, line, c):
        if c >= len(line):
            return 1
        if line[c] == "\t":
            v = vcol(line, c, self.ts())
            return self.ts() - v % self.ts()
        return 1

    def leave_visual(self):
        vis = self.vis
        if vis is None:
            return
        b = self.buf
        a = vis["anchor"]
        c = tuple(self.win.cursor)
        s, e = (a, c) if a <= c else (c, a)
        if vis["mode"] == "V":
            s = (s[0], 0)
            e = (e[0], max(0, len(b.lines[e[0]]) - 1) if False else MAXCOL - 1)
            e = (e[0], max(0, len(b.lines[e[0]])))
        b.marks["<"] = s
        b.marks[">"] = e
        self.last_vis = {"start": a, "end": c, "mode": vis["mode"], "want": self.win.curswant}
        self.vis = None
        self.message("")

    def visual_key(self, key):
        vis = self.vis
        w = self.win
        lines = self.lines
        if not vis.get("keys"):
            vis["keys"] = []
            vis["count_keys"] = []
        vis["keys"].append(key)
        ks = vis["keys"]
        # text objects and the keys that take an argument are parsed like normal-mode commands
        st = self.parse_visual(ks)
        if st[0] == "wait":
            self.showcmd = "".join(K.shown(K.to_text([k])) for k in ks)[-10:]
            return
        vis["keys"] = []
        self.showcmd = ""
        if st[0] == "bad":
            raise Fail()
        cmd = st[1]
        cmd["keys"] = ks
        return self.exec_visual(cmd)

    def parse_visual(self, ks):
        i, n = 0, len(ks)
        cmd = {"reg": None, "count": None, "op": None, "key": None, "arg": None, "count2": None}
        if ks[0] == '"':
            if n < 2:
                return ("wait",)
            cmd["reg"] = ks[1]
            i = 2
        d = ""
        while i < n and ks[i].isdigit() and not (ks[i] == "0" and not d):
            d += ks[i]
            i += 1
        if d:
            cmd["count"] = int(d)
        if i >= n:
            return ("wait",)
        k = ks[i]
        if k in ("i", "a"):
            if i + 1 >= n:
                return ("wait",)
            if ks[i + 1] not in "wWsp()b[]{}B<>\"'`t":
                return ("bad",)
            cmd["key"] = k + ks[i + 1]
            cmd["tobj"] = True
            return ("ok", cmd) if i + 2 == n else ("bad",)
        m = self._motion_at(ks, i)
        if m[0] == "ok" and m[3] == n:
            cmd["key"], cmd["arg"] = m[1], m[2]
            cmd["motion"] = True
            return ("ok", cmd)
        if m[0] == "wait":
            return ("wait",)
        if k in ("r",):
            if i + 1 >= n:
                return ("wait",)
            cmd["key"], cmd["arg"] = "r", ks[i + 1]
            return ("ok", cmd)
        if k in ("g", "z", "Z", "<C-w>"):
            if i + 1 >= n:
                return ("wait",)
            two = k + ks[i + 1] if len(ks[i + 1]) == 1 else k + ks[i + 1]
            if k == "<C-w>":
                cmd["key"], cmd["arg"] = "<C-w>", (ks[i + 1], None)
                return ("ok", cmd)
            cmd["key"] = two
            return ("ok", cmd)
        cmd["key"] = k
        return ("ok", cmd) if i + 1 == n else ("bad",)

    def exec_visual(self, cmd):
        w = self.win
        vis = self.vis
        key = cmd["key"]
        n = cmd["count"]
        n1 = n or 1
        lines = self.lines
        if cmd.get("motion"):
            self.used[key] += 1
            if key in ("/", "?"):
                self.start_cmdline(key, count=n)
                return
            cmd["n"] = n
            r = self.motion(cmd, None)
            pos = r[0]
            if vis["mode"] == "\x16" and key in ("$", "<End>"):
                w.cursor = [pos[0], pos[1]]
                self.clamp()
                w.curswant = MAXCOL
                return
            self.move_to(pos, cmd, r[1])
            return
        if cmd.get("tobj"):
            return self.visual_tobj(cmd)
        if key in ("<Esc>", "<C-c>"):
            self.leave_visual()
            return
        if key in ("v", "V", "<C-v>"):
            m = {"v": "v", "V": "V", "<C-v>": "\x16"}[key]
            if vis["mode"] == m:
                self.leave_visual()
            else:
                vis["mode"] = m
                self.message(f"-- {VNAMES[m]} --", "mode")
            return
        if key == "o":
            self.used["v_o"] += 1
            a = vis["anchor"]
            cur = tuple(w.cursor)
            if vis["mode"] == "v" and w.curswant >= MAXCOL and lines[cur[0]]:
                cur = (cur[0], len(lines[cur[0]]))   # after $: the end stays past the last character
            vis["anchor"] = cur
            if vis["mode"] == "v" and lines[a[0]] and a[1] >= len(lines[a[0]]):
                w.cursor = [a[0], len(lines[a[0]]) - 1]   # back on the end that was past the last character
                w.curswant = MAXCOL
                return
            w.cursor = [a[0], a[1]]
            w.curswant = vcol(lines[a[0]], a[1], self.ts())
            return
        if key == "O":
            self.used["v_O"] += 1
            if vis["mode"] != "\x16":
                return self.exec_visual({**cmd, "key": "o"})
            a = vis["anchor"]
            c = w.cursor
            ts = self.ts()
            va, vc = vcol(lines[a[0]], a[1], ts), vcol(lines[c[0]], c[1], ts)
            vis["anchor"] = (a[0], col_at_vcol(lines[a[0]], vc, ts))
            w.cursor = [c[0], col_at_vcol(lines[c[0]], va, ts)]
            w.curswant = va
            return
        if key == "gv":
            cur = {"start": vis["anchor"], "end": tuple(w.cursor), "mode": vis["mode"]}
            if self.last_vis:
                lv = self.last_vis
                self.vis = {"anchor": lv["start"], "mode": lv["mode"], "keys": []}
                w.cursor = list(lv["end"])
            self.last_vis = cur
            return
        if key == ":":
            self.used["v_:"] += 1
            self.leave_visual()
            return self.start_cmdline(":", text="'<,'>")
        if key == "<C-w>":
            self.leave_visual()
            k2, c2 = cmd["arg"]
            return self.window_cmd(k2, c2)
        s, e, kind, block = self.vis_area()
        size = self.vis_size(s, e, kind, block)
        mode = vis["mode"]
        self.used["v_" + key] += 1
        reg = cmd["reg"]

        def do(op, kind_=None, lines_=False):
            if op != "y":
                self.check_modifiable()
                self.mark_change_start()
            k2 = "line" if lines_ or kind == "line" else kind
            s2, e2 = s, e
            if k2 == "line":
                s2, e2 = (s[0], 0), (e[0], 0)
            blk = block if kind == "block" and not lines_ else None
            if blk and op in ("d", "c", "y", "g~", "gu", "gU", "g?"):
                k2 = "block"
            elif kind == "incl" and not lines_:
                # an end on an empty line (or past the end) takes the line break
                if e[1] >= len(lines[e[0]]) and e[0] + 1 < len(lines) and lines[e[0]] == "":
                    e2 = (e[0], 0)
            self.leave_visual()
            c = {**cmd, "op": op, "key": "v", "n": n, "vis": True}
            self.apply_op(op, s2, e2, k2, c, blk, tuple(self.win.cursor))
            if op != "y":
                if self.mode in ("i", "R") and self.ins is not None:
                    self.ins["vis_dot"] = ((mode, *size), [key], self.ins["keys"][:0])
                else:
                    self.dot = {"keys": [key] if not reg else ['"', reg, key], "count": n, "reg": None,
                                "visual": (mode, *size)}

        if key in ("d", "x", "<Del>"):
            return do("d")
        if key in ("X", "D"):
            if mode == "\x16" and key == "D":
                self.win.curswant = MAXCOL
                s, e, kind, block = self.vis_area()
                return do("d")
            return do("d", lines_=True)
        if key == "y":
            return do("y")
        if key == "Y":
            return do("y", lines_=mode != "\x16")
        if key in ("c", "s"):
            if mode == "\x16":
                return do("c")
            if mode == "V":
                return do("c", lines_=True)
            return do("c")
        if key in ("C", "S", "R"):
            if mode == "\x16" and key == "C":
                self.win.curswant = MAXCOL
                s, e, kind, block = self.vis_area()
                return do("c")
            return do("c", lines_=True)
        if key in (">", "<"):
            return do(key, lines_=True)
        if key == "=":
            return do("=", lines_=True)
        if key in ("~", "u", "U", "g~", "gu", "gU", "g?"):
            op = {"~": "g~", "u": "gu", "U": "gU"}.get(key, key)
            return do(op)
        if key in ("gq", "gw"):
            return do(key, lines_=True)
        if key == "!":
            self.leave_visual()
            return self.start_cmdline(":", text="'<,'>!")
        if key in ("J", "gJ"):
            from .commands import join_lines
            self.mark_change_start()
            l1, l2 = s[0], max(e[0], s[0] + 1)
            if l2 >= len(lines):
                self.leave_visual()
                raise Fail()
            self.leave_visual()
            join_lines(self, l1, l2, key == "J")
            self.dot = {"keys": [key], "count": None, "reg": None, "visual": (mode, *size)}
            return
        if key == "r":
            ch = cmd["arg"]
            if len(ch) != 1:
                ch = "\n" if ch in ("<CR>",) else K.CTRL_CHAR.get(ch, "")
            self.mark_change_start()
            self.leave_visual()
            self.map_region(s, e, kind, block, lambda t: "".join(ch if x != "\n" else x for x in t))
            self.set_cursor(s[0], s[1] if kind != "block" else col_at_vcol(lines[s[0]], block[0], self.ts()))
            self.dot = {"keys": ["r", cmd["arg"]], "count": None, "reg": None, "visual": (mode, *size)}
            return
        if key in ("p", "P"):
            return self.visual_put(s, e, kind, block, reg, n1, key)
        if key in ("I", "A"):
            return self.visual_insert(key, s, e, kind, block, size)
        if key in ("<C-a>", "<C-x>", "g<C-a>", "g<C-x>"):
            from .commands import increment
            self.mark_change_start()
            self.leave_visual()
            step = n1 if key.endswith("a>") else -n1
            for i, l in enumerate(range(s[0], e[0] + 1)):
                d = step * (i + 1) if key.startswith("g") else step
                line = self.lines[l]
                c0 = 0 if kind == "line" else (s[1] if l == s[0] else 0)
                if kind == "block":
                    c0 = col_at_vcol(line, block[0], self.ts(), past_end=True)
                new = increment(self, l, c0, d, line_text=line)
                if new != line:
                    self.set_line(self.buf, l, new)
            self.set_cursor(s[0], s[1] if kind != "line" else 0)
            return
        if key in ("zf",):
            self.leave_visual()
            self.create_fold(s[0], e[0])
            return
        if key in ("zo", "zc", "za", "zO", "zC", "zA", "zd", "zD"):
            self.leave_visual()
            from .folds import fold_command
            return fold_command(self, key, None, rng=(s[0], e[0]))
        if key == "zg" or key == "zw":
            self.leave_visual()
            return
        if key == "gJ":
            return
        if key in ("ZZ", "ZQ"):
            self.leave_visual()
            from .commands import normal_command
            return normal_command(self, {**cmd, "key": key})
        raise Fail()

    def vis_size(self, s, e, kind, block):
        if kind == "line":
            return (e[0] - s[0], 0)
        if kind == "block":
            return (e[0] - s[0], (block[1] - block[0]) if block[1] < MAXCOL else MAXCOL)
        return (e[0] - s[0], e[1] - s[1] if e[0] == s[0] else e[1])

    def visual_tobj(self, cmd):
        """A text object in visual mode: select it (extending a selection that's already bigger)."""
        w = self.win
        vis = self.vis
        cmd["n"] = cmd["count"]
        self.used[cmd["key"]] += 1
        s, e, kind = self.text_object(cmd)
        if kind == "empty":
            return
        if kind == "line":
            if vis["mode"] == "v" and True:
                vis["mode"] = "V"
                self.message("-- VISUAL LINE --", "mode")
            a = vis["anchor"]
            if vis["anchor"] != tuple(w.cursor) and a[0] < s[0]:
                s = (a[0], 0)
            vis["anchor"] = (s[0], 0)
            w.cursor = [e[0], 0]
            return
        if vis["mode"] == "V":
            vis["mode"] = "v"
            self.message("-- VISUAL --", "mode")
        a = vis["anchor"]
        if a != tuple(w.cursor):
            # extend the existing selection
            lo = min(a, tuple(w.cursor))
            s = min(s, lo)
        vis["anchor"] = s
        w.cursor = [e[0], e[1]]
        w.curswant = vcol(self.lines[e[0]], e[1], self.ts())

    def visual_put(self, s, e, kind, block, reg, count, key):
        """p in visual mode: the selection is replaced by the register (and goes into it, unless P)."""
        text, rkind = self.reg_get(reg)
        self.mark_change_start()
        self.leave_visual()
        old_text, old_kind = self.region_text(s, e, kind, block)
        cur_reg = self.regs.get("__unnamed")
        b = self.buf
        if kind == "line":
            self.replace_lines(b, s[0], e[0] + 1, [])
            if rkind == "V":
                body = text[:-1].split("\n") * count
                at = s[0]
                self.replace_lines(b, at, at, body)
                self.set_cursor(at)
            else:
                at = s[0]
                self.replace_lines(b, at, at, [""])
                end = self.insert_text(b, (at, 0), text * count)
                self.set_cursor(at, 0)
        else:
            if block:
                self.delete_block(s[0], e[0], block)
                self.set_cursor(s[0], col_at_vcol(self.lines[s[0]], block[0], self.ts()))
            else:
                self.delete_range(b, s, e)
                self.win.cursor = [s[0], s[1]]
            if rkind == "V":
                l = self.win.cursor[0]
                line = b.lines[l]
                c = self.win.cursor[1]
                body = text[:-1].split("\n") * count
                self.replace_lines(b, l, l + 1, [line[:c]] + body + [line[c:]])
                self.set_cursor(l + 1)
            else:
                end = self.insert_text(b, tuple(self.win.cursor), text * count)
                self.set_cursor(end[0], max(0, end[1] - 1))
        if key == "p":
            self.reg_set(None, old_text, old_kind, "d")
        self.dot = None

    def visual_insert(self, key, s, e, kind, block, size):
        """I / A in visual mode: on a block, the same text goes on every line."""
        w = self.win
        lines = self.lines
        mode = self.vis["mode"]
        self.mark_change_start()
        self.check_modifiable()
        if mode != "\x16":
            self.leave_visual()
            if key == "I":
                w.cursor = [s[0], s[1] if kind != "line" else first_nonblank(lines[s[0]])]
                if kind == "line":
                    w.cursor = [s[0], len(lines[s[0]]) - len(lines[s[0]].lstrip(" \t"))]
                return self.start_insert("i_raw", {"n": None, "keys": []})
            if kind == "line":
                w.cursor = [e[0], len(lines[e[0]])]
            else:
                w.cursor = [e[0], min(len(lines[e[0]]), e[1] + 1)]
            return self.start_insert("i_raw", {"n": None, "keys": []})
        ts = self.ts()
        dollar = block[1] >= MAXCOL
        self.leave_visual()
        if key == "I":
            l = s[0]
            c = col_at_vcol(lines[l], block[0], ts, past_end=True)
            w.cursor = [l, c]
            self.start_insert("block_I", {"n": None, "keys": []}, block=(s[0], e[0], block[0]))
        else:
            l = s[0]
            if dollar:
                c = len(lines[l])
                vc = None
            else:
                vc = block[1] + 1
                line = lines[l]
                if vcol(line, len(line), ts) < vc:
                    self.set_line(self.buf, l, line + " " * (vc - vcol(line, len(line), ts)))
                c = col_at_vcol(lines[l], vc, ts, past_end=True)
            w.cursor = [l, c]
            self.start_insert("block_A", {"n": None, "keys": []}, block=(s[0], e[0], vc if vc is not None else 0, dollar))
            self.ins["block_left"] = col_at_vcol(lines[l], block[0], ts)
        self.ins["vis_dot"] = ((mode, *size), [key], [])

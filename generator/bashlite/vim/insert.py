"""Insert and replace mode."""

from __future__ import annotations

import re

from . import keys as K
from .engine import MAXCOL, VimError
from .motions import Fail
from .text import col_at_vcol, indent_of, is_kw, make_indent, set_indent, vcol


class InsertMixin:

    def start_insert(self, how, cmd, block=None):
        """Enter insert (or replace) mode. how: i a I A o O C c s gi gI R c_empty block_c block_I block_A ..."""
        w, b = self.win, self.buf
        if how not in ("R",):
            self.check_modifiable()
        lines = b.lines
        l, c = w.cursor
        n1 = (cmd or {}).get("n") or 1
        if how == "a":
            if lines[l]:
                c += 1
        elif how == "A":
            c = len(lines[l])
        elif how == "I":
            m = re.match(r"[ \t]*", lines[l])
            c = len(m.group(0))
        elif how == "gI":
            c = 0
        elif how == "gi":
            if "^" in b.marks:
                l, c = b.marks["^"]
                l = min(l, len(lines) - 1)
                c = min(c, len(lines[l]))
        elif how in ("o", "O"):
            at = l + 1 if how == "o" else l
            f = self.closed_fold_at(l)
            if f:
                at = f[1] + 1 if how == "o" else f[0]
            ind = self.new_line_indent(at, how)
            self.replace_lines(b, at, at, [ind])
            l, c = at, len(ind)
        w.cursor = [l, c]
        if how in ("i", "a", "A", "I", "gI", "gi", "R", "i_raw"):
            # undo comes back to where the typing started (the first change of a macro wins)
            self.mark_change_start()
        self.mode = "R" if how == "R" else "i"
        self.ins = {"how": how, "start": (l, c), "count": n1, "typed": [], "cmd": cmd, "replaced": [],
                    "block": block, "first_line": l, "ai": None, "orig_line": lines[l],
                    "keys": list(self.cur_keys) if cmd else [], "compl": None}
        if how in ("o", "O") and self.ins_auto_indented(l):
            self.ins["ai"] = l
        if self.mode == "R":
            self.message("-- REPLACE --", "mode")
        else:
            self.message("-- INSERT --", "mode")
        self.used["insert"] += 1

    def ins_auto_indented(self, l):
        return bool(re.match(r"[ \t]+$", self.lines[l]))

    def new_line_indent(self, at, how, lines=None, ai_from=None):
        """Indent for a new line at index `at` (o, O, <CR>): indentexpr / cindent, or 'autoindent' (copied from
        the line above / below, or from `ai_from`: the auto-indent a <CR> left behind)."""
        b = self.buf
        lines = lines if lines is not None else b.lines
        kind = self.indent_kind()
        ref = at - 1 if how in ("o", "cr") else at
        ind = ""
        if self.opt("autoindent") or self.opt("smartindent") or self.opt("cindent"):
            if 0 <= ref < len(lines):
                ind = re.match(r"[ \t]*", lines[ref] if ai_from is None else ai_from).group(0)
                if self.opt("smartindent") and how in ("o", "cr") and lines[ref].rstrip().endswith(("{", ":")):
                    ind = make_indent(indent_of(ind, self.ts()) + self.sw(), self.opt("expandtab"), self.ts())
        if kind:
            # as in Vim: the new line gets the autoindent first, then indentexpr may change it (-1: keep it)
            tmp = lines[:at] + [ind] + lines[at:]
            width = self.indent_for(at, tmp)
            if width is not None:
                return make_indent(width, self.opt("expandtab"), self.ts())
        return ind

    # ------------------------------------------------------------ keys in insert mode

    def insert_key(self, key):
        ins = self.ins
        w, b = self.win, self.buf
        lines = b.lines
        if ins.get("ctrl_r"):
            return self.ins_ctrl_r(key)
        if ins.get("ctrl_v"):
            ins["ctrl_v"] = False
            ch = key if len(key) == 1 else K.to_text([key])
            return self.ins_text(ch, record=key)
        if ins.get("ctrl_o"):
            pass
        if ins.get("compl") is not None and key not in ("<C-n>", "<C-p>"):
            ins["compl"] = None
            if key in ("<C-e>",):
                return
            if key in ("<C-y>",):
                return
        ins["keys"].append(key)
        if key in ("<Esc>", "<C-[>", "<C-c>"):
            return self.leave_insert(abandon=key == "<C-c>")
        if key in ("<CR>", "<NL>", "<C-m>", "<C-j>"):
            return self.ins_newline()
        if key in ("<BS>", "<C-h>"):
            return self.ins_backspace()
        if key == "<Del>":
            l, c = w.cursor
            if c < len(lines[l]):
                self.set_line(b, l, lines[l][:c] + lines[l][c + 1:])
            elif l + 1 < len(lines):
                self.replace_lines(b, l, l + 2, [lines[l] + lines[l + 1]])
            return
        if key == "<C-w>":
            self.used["i_<C-w>"] += 1
            return self.ins_delete_word()
        if key == "<C-u>":
            self.used["i_<C-u>"] += 1
            return self.ins_delete_line()
        if key == "<Tab>":
            return self.ins_tab()
        if key in ("<C-t>", "<C-d>"):
            self.used["i_" + key] += 1
            l, c = w.cursor
            before = len(lines[l])
            self.shift_lines(l, l, 1 if key == "<C-t>" else -1)
            if not lines[l].strip() and key == "<C-t>":
                ind = make_indent(indent_of(lines[l], self.ts()) + self.sw(), self.opt("expandtab"), self.ts())
                self.set_line(b, l, ind)
            w.cursor = [l, max(0, c + len(lines[l]) - before)]
            return
        if key == "<C-r>":
            ins["ctrl_r"] = True
            return
        if key == "<C-v>" or key == "<C-q>":
            ins["ctrl_v"] = True
            return
        if key in ("<C-n>", "<C-p>"):
            self.used["i_" + key] += 1
            return self.complete(1 if key == "<C-n>" else -1)
        if key == "<C-o>":
            self.used["i_<C-o>"] += 1
            # one normal-mode command, then back to insert
            ins["ctrl_o"] = True
            saved = self.ins
            self.mode = "n"
            self.ins_paused = saved
            if w.cursor[1] >= len(lines[w.cursor[0]]) and w.cursor[1] > 0:
                saved["eol_o"] = True
            return
        if key in ("<Left>", "<Right>", "<Up>", "<Down>", "<Home>", "<End>", "<S-Left>", "<S-Right>"):
            return self.ins_move(key)
        if key == "<C-a>":
            last = self.regs.get(".", ("", "v"))[0]
            return self.ins_text(last)
        if key == "<C-e>" or key == "<C-y>":
            l, c = w.cursor
            src = l + (1 if key == "<C-e>" else -1)
            if 0 <= src < len(lines):
                vc = vcol(lines[l], c, self.ts())
                sc = col_at_vcol(lines[src], vc, self.ts(), past_end=True)
                if sc < len(lines[src]):
                    return self.ins_text(lines[src][sc])
            raise Fail()
        if key == "<C-k>":
            return
        if key == "<Insert>":
            self.mode = "R" if self.mode == "i" else "i"
            return
        if len(key) == 1:
            self.ins_text(key)
            if self.mode == "i":
                self.ins_indentkeys(key)
            return
        if key.startswith("<C-"):
            return
        return

    def ins_text(self, text, record=None):
        """Type characters at the cursor (overtyping in replace mode)."""
        w, b = self.win, self.buf
        ins = self.ins
        for ch in text:
            l, c = w.cursor
            line = b.lines[l]
            if ch == "\n" or ch == "\r":
                self.ins_newline()
                continue
            if self.mode == "R" and c < len(line):
                ins["replaced"].append((l, c, line[c]))
                self.set_line(b, l, line[:c] + ch + line[c + 1:])
            else:
                if self.mode == "R":
                    ins["replaced"].append((l, c, None))
                self.set_line(b, l, line[:c] + ch + line[c:])
            w.cursor = [l, c + 1]
            ins["typed"].append(ch)
            if ins.get("ai") == l and ch not in " \t":
                ins["ai"] = None
            self.auto_wrap()

    def ins_indentkeys(self, ch):
        """'indentkeys' / 'cinkeys': typing these re-indents the line (python: <:>,=elif,=except; yaml: <:>,0-,0},0];
        brace languages: 0},0),0])."""
        kind = self.indent_kind()
        if kind not in ("python", "yaml", "brace"):
            return
        w, b = self.win, self.buf
        l, c = w.cursor
        line = b.lines[l]
        before = line[:c]
        if kind == "brace":
            hit = ch in "}])" and before.strip() == ch
        elif ch == ":":
            hit = True
        elif kind == "python":
            hit = bool(re.fullmatch(r"\s*(elif|except)", before))
        else:
            hit = ch in "-}]" and before.strip() == ch
        if not hit:
            return
        width = self.indent_for(l)
        if width is None or width == indent_of(line, self.ts()):
            return
        new = set_indent(line, width, self.opt("expandtab"), self.ts())
        self.set_line(b, l, new)
        w.cursor = [l, max(0, c + len(new) - len(line))]

    def auto_wrap(self):
        """'textwidth' with 'formatoptions' t: a line that grows past it breaks at the last blank."""
        tw = self.opt("textwidth")
        if not tw or "t" not in str(self.opt("formatoptions")):
            return
        w, b = self.win, self.buf
        l, c = w.cursor
        line = b.lines[l]
        if vcol(line, c, self.ts()) <= tw or not line[c - 1:c].strip():
            return
        # break at the last blank before the limit
        cut = None
        for i in range(min(len(line), c) - 1, 0, -1):
            if line[i] in " \t" and vcol(line, i, self.ts()) <= tw:
                cut = i
                break
        if cut is None:
            return
        left = line[:cut].rstrip(" \t")
        rest = line[cut:].lstrip(" \t")
        ind = re.match(r"[ \t]*", line).group(0) if self.opt("autoindent") else ""
        self.replace_lines(b, l, l + 1, [left, ind + rest])
        w.cursor = [l + 1, len(ind) + (c - (len(line) - len(rest)))]

    def ins_newline(self):
        w, b = self.win, self.buf
        ins = self.ins
        l, c = w.cursor
        line = b.lines[l]
        if self.mode == "R":
            left, right = line[:c], line[c:]
        else:
            left, right = line[:c], line[c:]
        # an untouched auto-indent goes away when you leave its line (the new line still gets it)
        ai_from = None
        if ins.get("ai") == l and not left.strip():
            ai_from, left = left, ""
        lines2 = b.lines[:l] + [left] + b.lines[l + 1:]
        ind = self.new_line_indent(l + 1, "cr", lines2, ai_from)
        right_s = right.lstrip(" \t") if ind or self.opt("autoindent") else right
        if self.indent_kind() == "python" and right_s[:1] and False:
            pass
        self.replace_lines(b, l, l + 1, [left, ind + right_s])
        w.cursor = [l + 1, len(ind)]
        ins["typed"].append("\n")
        ins["ai"] = l + 1 if ind else None

    def ins_backspace(self):
        w, b = self.win, self.buf
        ins = self.ins
        l, c = w.cursor
        bs = str(self.opt("backspace"))
        start = ins["start"]
        can_start = "start" in bs or bs in ("2", "3")
        can_eol = "eol" in bs or bs in ("2", "3")
        if (l, c) <= start and not can_start and (l, c) != (0, 0) and not (c == 0 and l > start[0]):
            raise Fail()
        if c == 0:
            if l == 0 or not can_eol and (l, c) <= start:
                raise Fail()
            if not can_eol:
                raise Fail()
            prev = b.lines[l - 1]
            if self.mode == "R":
                w.cursor = [l - 1, len(prev)]
                return
            self.replace_lines(b, l - 1, l + 1, [prev + b.lines[l]])
            w.cursor = [l - 1, len(prev)]
            if ins["typed"]:
                ins["typed"].pop()
            if (l, c) <= start:
                ins["start"] = (l - 1, len(prev))
            return
        line = b.lines[l]
        if self.mode == "R":
            # restore what was typed over
            if ins["replaced"] and ins["replaced"][-1][:2] == (l, c - 1):
                _, _, orig = ins["replaced"].pop()
                if orig is None:
                    self.set_line(b, l, line[:c - 1] + line[c:])
                else:
                    self.set_line(b, l, line[:c - 1] + orig + line[c:])
            w.cursor = [l, c - 1]
            return
        # softtabstop: delete back to the previous tab stop when it's white space
        sts = self.opt("softtabstop")
        if sts and sts < 0:
            sts = self.sw()
        n = 1
        if sts and line[:c].endswith(" "):
            vc = vcol(line, c, self.ts())
            want = ((vc - 1) // sts) * sts
            k = c
            while k > 0 and line[k - 1] == " " and vcol(line, k - 1, self.ts()) >= want:
                k -= 1
            n = max(1, c - k)
        self.set_line(b, l, line[:c - n] + line[c:])
        w.cursor = [l, c - n]
        for _ in range(n):
            if ins["typed"]:
                ins["typed"].pop()
        if (l, c - n) < start:
            ins["start"] = (l, c - n)

    def ins_delete_word(self):
        w, b = self.win, self.buf
        ins = self.ins
        l, c = w.cursor
        line = b.lines[l]
        start = ins["start"]
        if c == 0:
            return self.ins_backspace()
        k = c
        # stop at the start of the insert first (backspace=start lets the next one go further)
        floor = start[1] if start[0] == l and c > start[1] else 0
        while k > floor and line[k - 1] in " \t":
            k -= 1
        if k > floor:
            kind = is_kw(line[k - 1])
            while k > floor and line[k - 1] not in " \t" and is_kw(line[k - 1]) == kind:
                k -= 1
        self.set_line(b, l, line[:k] + line[c:])
        w.cursor = [l, k]
        if (l, k) < start:
            ins["start"] = (l, k)
        del ins["typed"][max(0, len(ins["typed"]) - (c - k)):]

    def ins_delete_line(self):
        w, b = self.win, self.buf
        ins = self.ins
        l, c = w.cursor
        line = b.lines[l]
        start = ins["start"]
        if start[0] == l and c > start[1]:
            k = start[1]
        else:
            k = len(re.match(r"[ \t]*", line).group(0))
            if c <= k:
                k = 0
        self.set_line(b, l, line[:k] + line[c:])
        w.cursor = [l, k]
        if (l, k) < start:
            ins["start"] = (l, k)

    def ins_tab(self):
        w = self.win
        l, c = w.cursor
        line = self.lines[l]
        et = self.opt("expandtab")
        sts = self.opt("softtabstop")
        if sts and sts < 0:
            sts = self.sw()
        vc = vcol(line, c, self.ts())
        if et:
            step = sts or self.ts()
            n = step - vc % step
            return self.ins_text(" " * n)
        if sts:
            n = sts - vc % sts
            return self.ins_text(" " * n)
        return self.ins_text("\t")

    def ins_move(self, key):
        w = self.win
        l, c = w.cursor
        lines = self.lines
        if key == "<Left>":
            if c > 0:
                w.cursor = [l, c - 1]
        elif key == "<Right>":
            if c < len(lines[l]):
                w.cursor = [l, c + 1]
        elif key == "<Home>":
            w.cursor = [l, 0]
        elif key == "<End>":
            w.cursor = [l, len(lines[l])]
        elif key in ("<Up>", "<Down>"):
            nl = l + (1 if key == "<Down>" else -1)
            if 0 <= nl < len(lines):
                vc = vcol(lines[l], c, self.ts())
                w.cursor = [nl, col_at_vcol(lines[nl], vc, self.ts(), past_end=True)]
        # moving starts a new insert (undo and the dot command only repeat what's typed after it)
        self.commit_undo()
        self.mark_change_start()
        self.ins["start"] = tuple(w.cursor)
        self.ins["typed"] = []
        self.ins["keys"] = ["i"]
        self.ins["count"] = 1
        self.ins["moved"] = True

    def ins_ctrl_r(self, key):
        ins = self.ins
        ins["ctrl_r"] = False
        self.used["i_<C-r>"] += 1
        if key in ("<Esc>", "<C-c>"):
            return
        if key == "=":
            self.used["i_<C-r>="] += 1
            self.start_cmdline("=", expr_for="insert")
            return
        name = key if len(key) == 1 else None
        if name is None:
            return
        text, kind = self.reg_get(name)
        if kind == "V" and text.endswith("\n"):
            pass
        ins["keys"].extend(["<C-r>", key])
        self.ins_literal(text)

    def ins_literal(self, text):
        """Insert register text as if typed (line breaks start new lines, without auto-indent)."""
        w, b = self.win, self.buf
        parts = text.split("\n")
        l, c = w.cursor
        end = self.insert_text(b, (l, c), text)
        w.cursor = [end[0], end[1]]
        self.ins["typed"].extend(list(text))

    def complete(self, d):
        """<C-n> / <C-p>: complete the word before the cursor from words in the open buffers."""
        w, b = self.win, self.buf
        ins = self.ins
        l, c = w.cursor
        line = b.lines[l]
        if ins.get("compl") is None:
            k = c
            while k > 0 and is_kw(line[k - 1]):
                k -= 1
            prefix = line[k:c]
            words = []
            seen = {prefix}
            # this buffer: forward from the cursor (wrapping) for <C-n>, backward for <C-p>
            order = []
            n = len(b.lines)
            for i in range(n):
                li = (l + i) % n if d > 0 else (l - i) % n
                order.append(li)
            found = []
            for li in order:
                text = b.lines[li]
                ws = [(m.start(), m.group(0)) for m in re.finditer(r"[0-9A-Za-z_À-ÿ]+", text)]
                if li == l:
                    if d > 0:
                        ws = [x for x in ws if x[0] >= c] + ([] if True else [])
                    else:
                        ws = [x for x in ws if x[0] < k][::-1]
                elif d < 0:
                    ws = ws[::-1]
                for _, wd in ws:
                    if wd.startswith(prefix) and wd not in seen and len(wd) > len(prefix):
                        seen.add(wd)
                        found.append(wd)
            # the current line before the cursor, last for <C-n>
            if d > 0:
                for m in re.finditer(r"[0-9A-Za-z_À-ÿ]+", line[:k]):
                    wd = m.group(0)
                    if wd.startswith(prefix) and wd not in seen and len(wd) > len(prefix):
                        seen.add(wd)
                        found.append(wd)
            for ob in self.buffers.values():
                if ob is b or not ob.loaded or ob.kind != "file":
                    continue
                for text in ob.lines:
                    for m in re.finditer(r"[0-9A-Za-z_À-ÿ]+", text):
                        wd = m.group(0)
                        if wd.startswith(prefix) and wd not in seen and len(wd) > len(prefix):
                            seen.add(wd)
                            found.append(wd)
            if not found:
                self.message("-- Keyword completion (^N^P) Pattern not found", "err")
                return
            ins["compl"] = {"start": k, "prefix": prefix, "words": found, "idx": -1, "end": c}
        cp = ins["compl"]
        if cp["idx"] == -1:
            cp["idx"] = 0
        else:
            cp["idx"] += 1 if d > 0 else -1
            if cp["idx"] >= len(cp["words"]) or cp["idx"] < -1:
                cp["idx"] = -1 if cp["idx"] >= len(cp["words"]) else len(cp["words"]) - 1
        word = cp["prefix"] if cp["idx"] == -1 else cp["words"][cp["idx"]]
        line = b.lines[l]
        new = line[:cp["start"]] + word + line[cp["end"]:]
        self.set_line(b, l, new)
        cp["end"] = cp["start"] + len(word)
        w.cursor = [l, cp["end"]]
        total = len(cp["words"])
        if cp["idx"] >= 0:
            self.message(f"-- Keyword completion (^N^P) match {cp['idx'] + 1} of {total}", "mode")
        else:
            self.message("-- Keyword completion (^N^P) Back at original", "mode")

    # ------------------------------------------------------------ leaving insert mode

    def leave_insert(self, abandon=False):
        w, b = self.win, self.buf
        ins = self.ins
        typed = "".join(ins["typed"])
        how = ins["how"]
        count = ins["count"]
        # repeat the insert for a count (3ix<Esc> -> xxx)
        if count > 1 and not abandon and not ins.get("moved") and how in ("i", "a", "I", "A", "o", "O", "gI", "gi", "R",
                                                                              "i_raw"):
            l, c = w.cursor
            if how in ("o", "O"):
                for i in range(count - 1):
                    at = w.cursor[0] + 1 if how == "o" else w.cursor[0] + 1
                    ind = self.new_line_indent(at, "o")
                    self.replace_lines(b, at, at, [""])
                    w.cursor = [at, 0]
                    self.ins_text(ind + typed.lstrip(" \t") if ind else typed)
            elif how not in ("block_I", "block_A", "block_c"):
                self.ins_text(typed * (count - 1))
        # a block insert / append / change: the same text on the other lines
        if ins.get("block") and not abandon and "\n" not in typed:
            self.finish_block_insert(typed)
        # an auto-indent that got no text is removed (the cursor keeps wanting that column)
        l, c = w.cursor
        keep_want = None
        if ins.get("ai") == l and b.lines[l].strip() == "" and b.lines[l]:
            keep_want = vcol(b.lines[l], c, self.ts())
            self.set_line(b, l, "")
            c = 0
            w.cursor = [l, 0]
        self.regs["."] = (typed, "v")
        b.marks["^"] = (w.cursor[0], w.cursor[1])
        b.marks["."] = (w.cursor[0], max(0, w.cursor[1] - 1))
        self.mode = "n"
        self.ins = None
        self.message("")
        l, c = w.cursor
        w.cursor = [l, max(0, c - 1)] if c > 0 else [l, 0]
        self.clamp()
        w.curswant = vcol(b.lines[w.cursor[0]], w.cursor[1], self.ts()) if keep_want is None else keep_want
        if ins.get("block") and not abandon and how != "block_c":
            # after a block insert the cursor is at the block's top left corner
            bl = ins["block"]
            w.cursor = [bl[0], ins["start"][1] if how != "block_A" else ins.get("block_left", ins["start"][1])]
            self.clamp()
            w.curswant = vcol(b.lines[w.cursor[0]], w.cursor[1], self.ts())
        # the dot command repeats the whole thing: the command, what was typed, <Esc>
        keys = ins["keys"]
        if not getattr(self, "redoing", False) or True:
            cmd = ins.get("cmd") or {}
            if not ins.get("no_dot"):
                if ins.get("vis_dot"):
                    self.dot = {"keys": ins["vis_dot"][1] + ins["keys"][len(ins["vis_dot"][2]):],
                                "count": None, "reg": None, "visual": ins["vis_dot"][0]}
                else:
                    self.set_dot(keys, cmd.get("n") if not ins.get("moved") else None, cmd.get("reg"))
        self.redoing = False

    def finish_block_insert(self, typed):
        l1, l2, vc = self.ins["block"][:3]
        how = self.ins["how"]
        b = self.buf
        ts = self.ts()
        dollar = len(self.ins["block"]) > 3 and self.ins["block"][3]
        for l in range(l1 + 1, l2 + 1):
            line = b.lines[l]
            width = vcol(line, len(line), ts)
            if how == "block_A" and dollar:
                self.set_line(b, l, line + typed)
                continue
            if width < vc:
                if how == "block_A":
                    line = line + " " * (vc - width)
                    self.set_line(b, l, line + typed)
                continue
            a = col_at_vcol(line, vc, ts, past_end=True)
            self.set_line(b, l, line[:a] + typed + line[a:])

    # ------------------------------------------------------------ after <C-o>'s command

    def resume_insert(self):
        saved = getattr(self, "ins_paused", None)
        if saved is None:
            return
        self.ins_paused = None
        self.ins = saved
        saved["ctrl_o"] = False
        self.mode = "i" if saved["how"] != "R" else "R"
        w = self.win
        if saved.get("eol_o"):
            w.cursor = [w.cursor[0], len(self.lines[w.cursor[0]])]
        saved["start"] = tuple(w.cursor)
        saved["typed"] = []

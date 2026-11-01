"""Drawing Vim's screen: a grid of (character, attributes) cells, sent to the terminal as escape sequences.

Colours follow Vim's defaults for a light or dark background (the page tells which), in the terminal's palette.
"""

from __future__ import annotations

import re

from .engine import MAXCOL
from .text import char_width, vcol

LIGHT = {"Comment": "34", "Constant": "31", "Special": "35", "Identifier": "36", "Statement": "33", "PreProc": "35",
         "Type": "32", "LineNr": "38;5;130", "CursorLineNr": "1;38;5;130", "NonText": "1;94", "StatusLine": "1;7",
         "StatusLineNC": "7", "VertSplit": "7", "Visual": "48;5;252", "Search": "30;103", "IncSearch": "7",
         "ErrorMsg": "97;41", "WarningMsg": "31", "ModeMsg": "1", "MoreMsg": "1;32", "Question": "1;32",
         "Folded": "34;48;5;248", "TabLine": "4;30;47", "TabLineSel": "1", "TabLineFill": "7", "MatchParen": "106",
         "SpellBad": "4;31", "Pmenu": "30;48;5;225", "PmenuSel": "30;48;5;250", "ColorColumn": "48;5;224",
         "CursorLine": "4", "Directory": "34", "Target": "30;102", "Title": "1;35", "SpecialKey": "94",
         "Todo": "30;103"}
DARK = {**LIGHT, "Comment": "96", "Constant": "95", "Special": "38;5;224", "Identifier": "96", "Statement": "93",
        "PreProc": "38;5;81", "Type": "38;5;121", "LineNr": "93", "CursorLineNr": "1;93", "Visual": "48;5;242",
        "Folded": "96;48;5;242", "MatchParen": "46", "SpellBad": "4;91", "Pmenu": "30;48;5;13",
        "ColorColumn": "48;5;52", "Directory": "96", "Target": "30;102", "TabLine": "4;97;48;5;242"}

PY_KW = {"def", "class", "return", "if", "elif", "else", "for", "while", "in", "try", "except", "finally", "raise",
         "pass", "break", "continue", "lambda", "yield", "with", "as", "and", "or", "not", "is", "global",
         "nonlocal", "assert", "del", "async", "await"}
PY_INCLUDE = {"import", "from"}
PY_BUILTIN = {"None", "True", "False", "print", "len", "range", "int", "str", "float", "list", "dict", "set",
              "tuple", "open", "enumerate", "zip", "sum", "min", "max", "sorted", "isinstance", "super", "self"}
SH_KW = {"if", "then", "else", "elif", "fi", "for", "do", "done", "while", "case", "esac", "function", "in",
         "export", "local", "return", "echo", "cd"}
VIM_KW = {"set", "let", "syntax", "filetype", "nnoremap", "inoremap", "vnoremap", "noremap", "map", "nmap", "imap",
          "autocmd", "augroup", "if", "endif", "source", "colorscheme"}


def syntax_spans(ft, line):
    """(start, end, group) for a line (a sketch of Vim's syntax files: comments, strings, keywords, numbers)."""
    out = []
    if ft in ("python", "sh", "yaml", "toml", "dosini", "make", "dockerfile", "requirements"):
        cm = "#"
    elif ft in ("vim",):
        cm = '"'
    elif ft in ("c", "cpp", "javascript", "typescript", "java", "rust", "go", "css"):
        cm = "//"
    else:
        cm = None
    if ft == "markdown":
        if re.match(r"\s*#", line):
            return [(0, len(line), "Title")]
        for m in re.finditer(r"`[^`]*`", line):
            out.append((m.start(), m.end(), "Constant"))
        return out
    if ft in ("json",):
        for m in re.finditer(r'"(?:\\.|[^"\\])*"(\s*:)?|-?\b\d+(?:\.\d+)?\b|\b(?:true|false|null)\b', line):
            out.append((m.start(), m.end() - (len(m.group(1)) if m.group(1) else 0),
                        "Label" if m.group(1) else "Constant"))
        return [(s, e, "Statement" if g == "Label" else g) for s, e, g in out]
    if cm is None:
        return out
    i = 0
    n = len(line)
    kw = PY_KW if ft == "python" else SH_KW if ft in ("sh",) else VIM_KW if ft == "vim" else set()
    while i < n:
        ch = line[i]
        if line.startswith(cm, i) and not (ft == "vim" and line[:i].strip()):
            out.append((i, n, "Comment"))
            break
        if ch in "\"'" and not (ft == "vim" and ch == '"'):
            j = i + 1
            while j < n and line[j] != ch:
                if line[j] == "\\":
                    j += 1
                j += 1
            out.append((i, min(n, j + 1), "Constant"))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            if word in kw:
                out.append((i, j, "Statement"))
                if ft == "python" and word in ("def", "class"):
                    m = re.match(r"\s+(\w+)", line[j:])
                    if m:
                        out.append((j + m.start(1), j + m.end(1), "Identifier"))
            elif ft == "python" and word in PY_INCLUDE:
                out.append((i, j, "PreProc"))
            elif ft == "python" and word in PY_BUILTIN:
                out.append((i, j, "Identifier"))
            i = j
            continue
        if ch.isdigit() and (i == 0 or not (line[i - 1].isalnum() or line[i - 1] == "_")):
            j = i
            while j < n and (line[j].isalnum() or line[j] in "._"):
                j += 1
            out.append((i, j, "Constant"))
            i = j
            continue
        if ft == "python" and ch == "@" and not line[:i].strip():
            out.append((i, n, "PreProc"))
            break
        i += 1
    return out


class ScreenMixin:

    def colors(self):
        return DARK if getattr(self, "dark", False) else LIGHT

    def render(self):
        """-> (rows of (char, attr) cells, (cursor row, cursor col))."""
        W, H = self.width, self.height
        C = self.colors()
        grid = [[(" ", "")] * W for _ in range(H)]
        cursor = (0, 0)
        self.layout()
        if len(self.tabs) > 1:
            self.draw_tabline(grid, C)
        hl = self.hl_matches()
        for w in self.all_windows():
            pos = self.draw_window(grid, w, C, hl)
            if w is self.win and pos is not None:
                cursor = pos
            if getattr(w, "has_status", False):
                self.draw_status(grid, w, C)
            # the separator to the right of a window (unless it reaches the edge)
            if w.col + w.width < W:
                for r in range(w.row, w.row + w.height + (1 if getattr(w, "has_status", False) else 0)):
                    if 0 <= r < H - 1:
                        grid[r][w.col + w.width] = ("|", C["VertSplit"])
        cpos = self.draw_cmdline(grid, C)
        if cpos is not None:
            cursor = cpos
        pop = self.draw_popup(grid, C, cursor)
        mp = self.draw_more(grid, C)
        if mp is not None:
            cursor = mp
        return grid, cursor

    # ------------------------------------------------------------ pieces

    def draw_tabline(self, grid, C):
        W = self.width
        row = []
        for i, t in enumerate(self.tabs):
            w = t["win"]
            name = (w.buf.name or "[No Name]").rsplit("/", 1)[-1]
            n = len(self.windows_of(t))
            label = f" {str(n) if n > 1 else ''}{'+' if any(x.buf.modified for x in self.windows_of(t)) else ''}" \
                    f"{' ' if (n > 1 or any(x.buf.modified for x in self.windows_of(t))) else ''}{name} "
            attr = C["TabLineSel"] if i == self.tabidx else C["TabLine"]
            row += [(ch, attr) for ch in label]
        row = row[:W - 1]
        row += [(" ", C["TabLineFill"])] * (W - 1 - len(row)) + [("X", C["TabLine"])]
        grid[0] = row[:W]

    def hl_matches(self):
        """Search highlighting for every window: {bufnum: [(line, start, end)]}."""
        out = {}
        pat = None
        cur = None
        if self.mode == "c" and self.cmdl and self.cmdl["kind"] in "/?" and self.opt("incsearch") and self.cmdl["text"]:
            from .cmdline import split_search
            pat = split_search(self.cmdl["text"], self.cmdl["kind"])[0]
            cur = self.cmdl.get("inc")
        elif self.opt("hlsearch") and not self.no_hl and self.last_pat:
            pat = self.last_pat
        if self.confirm is not None:
            c = self.confirm
            if "cur" in c:
                l, s, e = c["cur"]
                out[self.buf.num] = [(l, s, max(e, s + 1), "IncSearch")]
            return out
        if not pat:
            return out
        from .search import all_matches, compiled, offsets, pos_of
        try:
            cre = compiled(self, pat)
        except Exception:  # noqa: BLE001
            return out
        for w in self.all_windows():
            b = w.buf
            if b.num in out:
                continue
            spans = []
            matches, text = all_matches(self, cre, b.lines)
            offs = offsets(b.lines)
            for s, e in matches[:2000]:
                l1, c1 = pos_of(offs, s)
                l2, c2 = pos_of(offs, max(s, e - 1) if e > s else s)
                group = "Search" if self.opt("hlsearch") and not self.no_hl else None
                if cur is not None and (l1, c1) == tuple(cur) and b is self.buf:
                    group = "IncSearch"
                if group is None:
                    continue
                for l in range(l1, l2 + 1):
                    a = c1 if l == l1 else 0
                    z = (c2 + 1) if l == l2 else len(b.lines[l]) + 1
                    spans.append((l, a, max(z, a + 1), group))
            out[b.num] = spans
        return out

    def draw_window(self, grid, w, C, hl):
        b = w.buf
        lines = b.lines
        ts = self.ts(b)
        nw = self.numberwidth(w)
        text_w = max(1, w.width - nw)
        ft = self.opt("filetype", b) if self.o.get("syntax") == "on" else ""
        rel = self.opt("relativenumber", win=w)
        num = self.opt("number", win=w)
        lst = self.opt("list", win=w)
        wrap = self.opt("wrap", win=w)
        spell = self.opt("spell", win=w)
        cc = str(self.opt("colorcolumn", win=w) or "")
        cul = self.opt("cursorline", win=w)
        vis = self.vis_spans(w) if (w is self.win and self.vis is not None) else None
        mp = self.match_paren(w) if (w is self.win and self.mode == "n" and self.vis is None) else None
        tgt = self.target if (w is self.win and self.target is not None and b.kind == "file") else None
        searches = {}
        for item in hl.get(b.num, []):
            searches.setdefault(item[0], []).append(item[1:])
        row = w.row
        end_row = w.row + w.height
        l = w.topline
        cursor = None
        cl = w.cursor[0]
        from .folds import closed_fold_at, fold_text
        while row < end_row:
            if l >= len(lines):
                self.put(grid, row, w.col, "~", C["NonText"], w.width)
                row += 1
                continue
            fold = closed_fold_at(self, l, w)
            if fold:
                text = fold_text(self, w, fold[0], fold[1])
                prefix = self.number_text(w, l, nw, num, rel, cl)
                self.put(grid, row, w.col, prefix, C["LineNr"], nw)
                body = (text + "-" * text_w)[:text_w]
                self.put(grid, row, w.col + nw, body, C["Folded"], text_w)
                if fold[0] <= cl <= fold[1] and w is self.win:
                    cursor = (row, w.col + nw)
                l = fold[1] + 1
                row += 1
                continue
            line = lines[l]
            cells = self.line_cells(line, ts, lst)
            attrs = [""] * len(cells)
            owner = self.cell_owner(line, ts, lst)
            # syntax
            if ft and b.kind == "file":
                for s, e, g in syntax_spans(ft, line):
                    self._paint(attrs, owner, s, e, C.get(g, ""))
            if b.kind == "netrw":
                if line.endswith("/"):
                    self._paint(attrs, owner, 0, len(line), C["Directory"])
                elif line.startswith('"'):
                    self._paint(attrs, owner, 0, len(line), C["Comment"])
            if b.kind == "help" and l == 0:
                self._paint(attrs, owner, 0, len(line), C["Title"])
            if spell and b.kind == "file":
                from .spell import bad_words
                for s, e in bad_words(self, line):
                    self._paint(attrs, owner, s, e, C["SpellBad"])
            for s, e, g in searches.get(l, []):
                self._paint(attrs, owner, s, min(e, len(line)) if e <= len(line) else len(line), C[g])
                if e > len(line):
                    self._extra_eol = (l, C[g])
            if vis is not None:
                for s, e in vis.get(l, []):
                    self._paint(attrs, owner, s, e, C["Visual"], vis_mode=True)
            if mp is not None and mp[0] == l:
                self._paint(attrs, owner, mp[1], mp[1] + 1, C["MatchParen"])
            if tgt is not None and tgt[0] == l:
                if line:
                    self._paint(attrs, owner, tgt[1], tgt[1] + 1, C["Target"])
            # chunks of the screen width
            if wrap:
                chunks = [(i, min(len(cells), i + text_w)) for i in range(0, max(1, len(cells)), text_w)] or [(0, 0)]
                if len(cells) and len(cells) % text_w == 0 and False:
                    pass
            else:
                left = w.leftcol
                chunks = [(left, min(len(cells), left + text_w))]
            # does the whole line fit? (else "@" lines, or "@@@" with display=truncate)
            if row + len(chunks) > end_row and l != w.topline:
                for r in range(row, end_row):
                    if "truncate" in str(self.opt("display")):
                        if r == end_row - 1:
                            self.put(grid, r, w.col, "@@@", C["NonText"], w.width)
                        else:
                            self.draw_chunk(grid, r, w, cells, attrs, chunks[r - row] if r - row < len(chunks) else (0, 0),
                                            nw, self.number_text(w, l, nw, num, rel, cl) if r == row else " " * nw,
                                            C, l == cl and cul, cc)
                    else:
                        self.put(grid, r, w.col, "@", C["NonText"], w.width)
                break
            for k, (a, z) in enumerate(chunks):
                if row >= end_row:
                    break
                prefix = self.number_text(w, l, nw, num, rel, cl) if k == 0 else " " * nw
                self.draw_chunk(grid, row, w, cells, attrs, (a, z), nw, prefix, C, l == cl and cul, cc,
                                cur_line=(l == cl))
                # the cursor
                if l == cl and w is self.win and cursor is None:
                    ccol = w.cursor[1]
                    if ccol <= len(line):
                        vc = self.cursor_vcol(line, ccol, ts, lst)
                        if a <= vc < z or (k == len(chunks) - 1 and vc >= a):
                            cursor = (row, w.col + nw + min(vc - a, text_w - 1))
                row += 1
            l += 1
        if cursor is None and w is self.win:
            cursor = (w.row, w.col + nw)
        return cursor

    def cursor_vcol(self, line, col, ts, lst):
        """Screen column of the cursor on a character: the end of a tab in normal mode (as Vim shows it)."""
        if col < len(line) and line[col] == "\t" and not lst and self.mode not in ("i", "R"):
            v = vcol(line, col, ts)
            return v + (ts - v % ts) - 1
        cells = self.line_cells(line[:col], ts, lst)
        return len(cells)

    def draw_chunk(self, grid, row, w, cells, attrs, span, nw, prefix, C, cul, cc, cur_line=False):
        a, z = span
        num_attr = C["CursorLineNr"] if cur_line and (self.opt("relativenumber", win=w) or self.opt("cursorline", win=w)) and (self.opt("number", win=w) or self.opt("relativenumber", win=w)) else C["LineNr"]
        self.put(grid, row, w.col, prefix, num_attr, nw)
        text_w = w.width - nw
        for i in range(text_w):
            j = a + i
            ch, at = (cells[j], attrs[j]) if j < z else (" ", "")
            if cul:
                at = (at + ";" if at else "") + C["CursorLine"]
            if cc:
                for col in cc.split(","):
                    if col.strip().isdigit() and j == int(col) - 1:
                        at = (at + ";" if at else "") + C["ColorColumn"]
            if w.col + nw + i < self.width:
                grid[row][w.col + nw + i] = (ch, at)

    def line_cells(self, line, ts, lst):
        out = []
        v = 0
        for ch in line:
            if ch == "\t":
                if lst:
                    out += ["^", "I"]
                    v += 2
                else:
                    n = ts - v % ts
                    out += [" "] * n
                    v += n
            elif ord(ch) < 32 or ch == "\x7f":
                out += ["^", "?" if ch == "\x7f" else chr(ord(ch) + 64)]
                v += 2
            else:
                out.append(ch)
                v += 1
                if char_width(ch) == 2:
                    out.append("")
                    v += 1
        if lst:
            out.append("$")
        return out

    def cell_owner(self, line, ts, lst):
        """For each character index, the range of screen cells it covers."""
        owner = []
        v = 0
        for ch in line:
            if ch == "\t":
                n = 2 if lst else ts - v % ts
            elif ord(ch) < 32 or ch == "\x7f":
                n = 2
            else:
                n = char_width(ch)
            owner.append((v, v + n))
            v += n
        owner.append((v, v + 1))
        return owner

    def _paint(self, attrs, owner, s, e, attr, vis_mode=False):
        if not attr:
            return
        for i in range(max(0, s), min(e, len(owner))):
            a, z = owner[i]
            for k in range(a, min(z, len(attrs))):
                attrs[k] = attr if vis_mode or not attrs[k] or attr in ("7",) else attr if attr != "" else attrs[k]

    def number_text(self, w, l, nw, num, rel, cl):
        if not nw:
            return ""
        if rel:
            if l == cl:
                return (f"{l + 1:<{nw - 1}} " if num else f"{0:>{nw - 1}} ")
            return f"{abs(l - cl):>{nw - 1}} "
        return f"{l + 1:>{nw - 1}} "

    def put(self, grid, row, col, text, attr, width):
        for i, ch in enumerate(text[:width]):
            if col + i < self.width and 0 <= row < self.height:
                grid[row][col + i] = (ch, attr)

    def vis_spans(self, w):
        """{line: [(start, end)]} of the visual selection (end exclusive, may be len+1 for the line break)."""
        s, e, kind, block = self.vis_area()
        out = {}
        lines = w.buf.lines
        if kind == "line":
            for l in range(s[0], e[0] + 1):
                out[l] = [(0, len(lines[l]) + 1)]
        elif kind == "block":
            ts = self.ts()
            for l in range(s[0], e[0] + 1):
                line = lines[l]
                from .text import col_at_vcol
                a = col_at_vcol(line, block[0], ts, past_end=True)
                z = len(line) if block[1] >= MAXCOL else col_at_vcol(line, block[1], ts, past_end=True) + 1
                if a < len(line) or block[1] >= MAXCOL:
                    out[l] = [(a, max(a, min(z, len(line))) if block[1] < MAXCOL else len(line))]
        else:
            for l in range(s[0], e[0] + 1):
                a = s[1] if l == s[0] else 0
                z = e[1] + 1 if l == e[0] else len(lines[l]) + 1
                out[l] = [(a, max(a + 1, z))]
        return out

    def match_paren(self, w):
        l, c = w.cursor
        line = w.buf.lines[l]
        if c >= len(line) or line[c] not in "()[]{}":
            return None
        from .motions import match_pair
        try:
            return match_pair(w.buf.lines, l, c)
        except Exception:  # noqa: BLE001
            return None

    def draw_status(self, grid, w, C):
        row = w.row + w.height
        if row >= self.height - 1:
            return
        b = w.buf
        attr = C["StatusLine"] if w is self.win else C["StatusLineNC"]
        name = b.name or "[No Name]"
        if b.kind == "help":
            name += " [Help][RO]"
        elif b.kind == "terminal":
            name = name + (" [running]" if b.extra.get("running") else "")
        elif b.kind == "quickfix":
            name = "[Quickfix List]"
        else:
            if b.modified:
                name += " [+]"
            if b.readonly:
                name += " [RO]"
        left = " " + name
        ruler = self.ruler_text(w) if self.opt("ruler") else ""
        width = w.width
        text = left.ljust(width)
        if ruler:
            pos = max(len(left) + 1, width - 18)
            text = (left.ljust(pos) + ruler)[:width].ljust(width)
        self.put(grid, row, w.col, text, attr, width)

    def ruler_text(self, w):
        l, c = w.cursor
        line = w.buf.lines[l]
        if not line:
            colpart = "0-1"
        else:
            vc = vcol(line, c, self.ts(w.buf)) + 1
            if line[c:c + 1] == "\t" and self.mode not in ("i", "R"):
                vc = vcol(line, c, self.ts(w.buf)) + (self.ts(w.buf) - vcol(line, c, self.ts(w.buf)) % self.ts(w.buf))
            colpart = str(c + 1) if vc == c + 1 else f"{c + 1}-{vc}"
        top, last = self.visible_range(w)
        n = len(w.buf.lines)
        if top == 0 and last >= n - 1:
            pct = "All"
        elif top == 0:
            pct = "Top"
        elif last >= n - 1:
            pct = "Bot"
        else:
            pct = f"{top * 100 // max(1, n - w.height + 1) if n > w.height else 0}%"
            pct = f"{(top * 100) // max(1, (n - w.height))}%" if n > w.height else "All"
            pct = pct if len(pct) <= 3 else "99%"
        left = f"{l + 1},{colpart}"
        return left + " " * max(1, 14 - len(left)) + pct

    def draw_cmdline(self, grid, C):
        H, W = self.height, self.width
        row = H - 1
        self.put(grid, row, 0, " " * W, "", W)
        if self.mode == "c" and self.cmdl is not None:
            c = self.cmdl
            prefix = {":": ":", "/": "/", "?": "?", "=": "="}[c["kind"]]
            text = prefix + c["text"]
            pos = len(prefix) + c["pos"]
            start = max(0, pos - (W - 1))
            self.put(grid, row, 0, text[start:start + W], "", W)
            return (row, pos - start)
        if self.pending_prompt is not None or self.more is not None:
            return None
        msg = self.msg
        kind = self.msg_kind
        attr = {"err": C["ErrorMsg"], "warn": C["WarningMsg"], "mode": C["ModeMsg"]}.get(kind, "")
        text = msg
        if self.recording is not None:
            rec = f"recording @{self.recording}"
            if kind == "mode":
                text = msg + rec
            elif not msg:
                text, attr = rec, C["ModeMsg"]
        if "\n" in text:
            text = text.split("\n")[-1]
        self.put(grid, row, 0, text[:W], attr, W)
        # showcmd and the ruler
        if self.opt("showcmd"):
            sc = self.showcmd
            if self.vis is not None and not sc:
                s, e, kind_, block = self.vis_area()
                if kind_ == "line" or (kind_ == "incl" and s[0] != e[0]):
                    sc = str(e[0] - s[0] + 1)
                elif kind_ == "block":
                    sc = f"{e[0] - s[0] + 1}x{(block[1] - block[0] + 1) if block[1] < MAXCOL else len(self.lines[self.win.cursor[0]])}"
                else:
                    sc = str(e[1] - s[1] + 1)
            col = W - 10 - (18 if self.opt("ruler") and not getattr(self.win, "has_status", False) else 0) - 1
            if sc and col > len(text):
                self.put(grid, row, col, sc[-10:], "", 11)
        if self.opt("ruler") and not getattr(self.win, "has_status", False):
            r = self.ruler_text(self.win)
            col = W - 18
            if col > len(text):
                self.put(grid, row, col, r[:18], "", 18)
        return None

    def draw_popup(self, grid, C, cursor):
        ins = self.ins if self.mode in ("i", "R") else None
        if not ins or not ins.get("compl"):
            return None
        cp = ins["compl"]
        words = cp["words"][:10]
        width = max(len(x) for x in words) + 2
        r0 = cursor[0] + 1
        if r0 + len(words) >= self.height - 1:
            r0 = max(0, cursor[0] - len(words))
        c0 = max(0, min(self.width - width, cursor[1] - len(cp["prefix"]) - 1))
        for i, wd in enumerate(words):
            attr = C["PmenuSel"] if i == cp["idx"] else C["Pmenu"]
            self.put(grid, r0 + i, c0, (" " + wd).ljust(width), attr, width)
        return True

    def draw_more(self, grid, C):
        """Output waiting for Enter (:ls :reg :!cmd ...), and questions (W12, z=)."""
        H, W = self.height, self.width
        if self.pending_prompt is not None:
            lines = list(self.pending_prompt["lines"])
            if self.pending_prompt.get("kind") == "spell":
                lines[-1] = lines[-1] + self.pending_prompt.get("typed", "")
            attrs = [C["WarningMsg"]] * (len(lines) - 1) + [C["Question"]]
        elif self.more is not None:
            lines = list(self.more) + ["Press ENTER or type command to continue"]
            attrs = [""] * (len(lines) - 1) + [C["MoreMsg"]]
            if self.more_kind == "shell" and lines and lines[0].startswith(":"):
                attrs[0] = ""
        else:
            return None
        flat, fattrs = [], []
        for text, at in zip(lines, attrs):
            parts = [text[i:i + W] for i in range(0, max(1, len(text)), W)] or [""]
            for p in parts:
                flat.append(p)
                fattrs.append(at)
        flat, fattrs = flat[-H:], fattrs[-H:]
        # the screen scrolls up to make room, as in Vim
        start = H - len(flat)
        if start > 0:
            shift = len(flat) - 1
            for r in range(0, H - 1 - shift):
                grid[r] = list(grid[r + shift]) if r + shift < H - 1 else [(" ", "")] * W
        for i, (text, at) in enumerate(zip(flat, fattrs)):
            r = start + i
            grid[r] = [(" ", "")] * W
            self.put(grid, r, 0, text, at, W)
        last = flat[-1]
        return (H - 1, min(W - 1, len(last)))

    # ------------------------------------------------------------ escape sequences

    def frame(self, full=False):
        """The screen as escape sequences: only the rows that changed since the last frame (all, if full)."""
        grid, (cr, cc) = self.render()
        prev = None if full else self.screen_cache
        out = []
        for r, row in enumerate(grid):
            if prev is not None and r < len(prev) and prev[r] == row:
                continue
            out.append(f"\x1b[{r + 1};1H")
            cur = None
            for ch, at in row:
                if ch == "":
                    continue
                if at != cur:
                    out.append("\x1b[0m" + (f"\x1b[{at}m" if at else ""))
                    cur = at
                out.append(ch)
            out.append("\x1b[0m")
        self.screen_cache = grid
        out.append(f"\x1b[{cr + 1};{cc + 1}H")
        return "".join(out)

    def screen_text(self):
        """The screen as plain text rows (tests)."""
        grid, cur = self.render()
        return ["".join(ch for ch, _ in row).rstrip() for row in grid], cur

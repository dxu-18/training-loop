"""Normal-mode commands that aren't motions or operators: x s r p J ~ u . i a o R v q @ m <C-a> z g Z <C-w> ..."""

from __future__ import annotations

import re

from . import keys as K
from .engine import MAXCOL, VimError
from .motions import Fail, keyword_at
from .text import col_at_vcol, first_nonblank, vcol


def normal_command(v, cmd):
    key, arg, n = cmd["key"], cmd["arg"], cmd["n"]
    n1 = n or 1
    w = v.win
    b = v.buf
    lines = b.lines
    l, c = w.cursor
    reg = cmd["reg"]
    keys = cmd["keys"]

    def change_cmd(op_cmd):
        v.mark_change_start()
        return op_cmd

    if key in ("<Esc>", "<C-c>", "nop", "<C-l>", "<Insert>"):
        if key in ("<Esc>",) and not keys[:-1]:
            v.bell = True
        return
    if key == "x" or key == "<Del>":
        if not lines[l]:
            raise Fail()
        v.mark_change_start()
        e = min(len(lines[l]) - 1, c + n1 - 1)
        v.apply_op("d", (l, c), (l, e), "incl", {**cmd, "op": "d", "key": "l"})
        v.set_dot(keys, n, reg)
        return
    if key == "X":
        if c == 0:
            raise Fail()
        v.mark_change_start()
        s = max(0, c - n1)
        v.apply_op("d", (l, s), (l, c - 1), "incl", {**cmd, "op": "d", "key": "h"})
        v.set_dot(keys, n, reg)
        return
    if key == "D" or key == "C":
        v.mark_change_start()
        el = min(len(lines) - 1, l + n1 - 1)
        if not lines[l] and el == l:
            if key == "C":
                return v.start_insert("C", cmd)
            return
        e = (el, max(0, len(lines[el]) - 1))
        if not lines[el] and el > l:
            e = (el - 1, len(lines[el - 1]))
            e = (el, 0) if True else e
        sub = {**cmd, "op": "d" if key == "D" else "c", "key": "$"}
        if lines[el]:
            v.apply_op(sub["op"], (l, c), e, "incl", sub)
        else:
            # the last line is empty: the deletion ends with the line break before it
            v.apply_op(sub["op"], (l, c), (el - 1, len(lines[el - 1])), "incl", sub)
        if key == "D":
            v.clamp()
            v.set_dot(keys, n, reg)
            w.curswant = MAXCOL if False else vcol(lines[w.cursor[0]], w.cursor[1], v.ts())
        return
    if key == "s":
        v.mark_change_start()
        if not lines[l]:
            return v.start_insert("s", cmd)
        e = min(len(lines[l]) - 1, c + n1 - 1)
        return v.apply_op("c", (l, c), (l, e), "incl", {**cmd, "op": "c", "key": "l"})
    if key == "S":
        v.mark_change_start()
        return v.do_operator({**cmd, "op": "c", "key": "line"})
    if key == "Y":
        return v.do_operator({**cmd, "op": "y", "key": "line"})
    if key in ("p", "P", "gp", "gP"):
        v.mark_change_start()
        return put(v, reg, n1, after=key in ("p", "gp"), gp=key.startswith("g"), keys=keys, given=n)
    if key in ("J", "gJ"):
        v.mark_change_start()
        cnt = max(2, n1)
        if l + 1 >= len(lines):
            raise Fail()
        join_lines(v, l, min(len(lines) - 1, l + cnt - 1), key == "J")
        v.set_dot(keys, n, reg)
        return
    if key == "~":
        if v.opt("tildeop"):
            return v.do_operator({**cmd, "op": "g~", "key": "l"})
        line = lines[l]
        if not line:
            raise Fail()
        v.mark_change_start()
        e = min(len(line), c + n1)
        v.set_line(b, l, line[:c] + line[c:e].swapcase() + line[e:])
        v.set_cursor(l, min(e, len(line) - 1))
        v.set_dot(keys, n, reg)
        return
    if key == "r":
        line = lines[l]
        if not line or c + n1 > len(line):
            raise Fail()
        ch = arg if len(arg) == 1 else ("\n" if arg in ("<CR>", "<NL>") else K.CTRL_CHAR.get(arg, ""))
        if arg in ("<C-v>",):
            raise Fail()
        v.mark_change_start()
        if ch == "\n":
            # r<CR>: replace the characters with one line break
            v.set_line(b, l, line[:c].rstrip(" \t") if False else line[:c])
            rest = line[c + n1:]
            ind = ""
            if v.opt("autoindent"):
                ind = re.match(r"[ \t]*", line).group(0)
            v.replace_lines(b, l, l + 1, [line[:c], ind + rest.lstrip(" \t") if v.opt("autoindent") else rest])
            v.set_cursor(l + 1, len(ind))
        else:
            v.set_line(b, l, line[:c] + ch * n1 + line[c + n1:])
            v.set_cursor(l, c + n1 - 1)
        v.set_dot(keys, n, reg)
        return
    if key == "gr":
        return normal_command(v, {**cmd, "key": "r"})
    if key in ("u",):
        v.commit_undo()
        return v.do_undo(n1)
    if key == "<C-r>":
        v.commit_undo()
        return v.do_redo(n1)
    if key == "U":
        v.commit_undo()
        v.line_undo()
        return
    if key in ("g-", "g+"):
        v.commit_undo()
        return v.undo_time(steps=-n1 if key == "g-" else n1)
    if key == ".":
        return v.repeat_dot(n)
    if key in ("i", "a", "I", "A", "o", "O", "gi", "gI", "R"):
        if key in ("o", "O"):
            v.mark_change_start()
        return v.start_insert(key, cmd)
    if key in ("v", "V", "<C-v>"):
        return v.start_visual({"v": "v", "V": "V", "<C-v>": "\x16"}[key], count=n)
    if key == "gv":
        return v.reselect()
    if key in ("gn", "gN"):
        s, e = v.find_gn(key == "gN")
        v.vis = {"anchor": s, "mode": "v"}
        w.cursor = list(e)
        return
    if key == ":":
        if n:
            return v.start_cmdline(":", text="." if n == 1 else f".,.+{n - 1}")
        return v.start_cmdline(":")
    if key in ("<C-a>", "<C-x>"):
        v.mark_change_start()
        increment(v, l, c, n1 if key == "<C-a>" else -n1)
        v.set_dot(keys, n, reg)
        return
    if key in ("<C-o>", "<Tab>"):
        return jump(v, -n1 if key == "<C-o>" else n1)
    if key == "<C-^>":
        target = v.buffers.get(n) if n else v.buffers.get(w.alt)
        if target is None:
            raise VimError("E23: No alternate file")
        return v.edit_buffer(target)
    if key in ("<C-d>", "<C-u>"):
        return scroll_half(v, n, key == "<C-d>")
    if key in ("<C-f>", "<C-b>", "<PageDown>", "<PageUp>"):
        return scroll_page(v, n1, key in ("<C-f>", "<PageDown>"))
    if key in ("<C-e>", "<C-y>"):
        return scroll_lines(v, n1 if key == "<C-e>" else -n1)
    if key in ("<C-g>", "g<C-g>"):
        pct = (l + 1) * 100 // len(lines)
        v.message(f'"{b.name or "[No Name]"}"{" [Modified]" if b.modified else ""} {len(lines)} line'
                  f'{"s" if len(lines) != 1 else ""} --{pct}%--')
        return
    if key == "ga":
        ch = lines[l][c:c + 1]
        if not ch:
            v.message("NUL")
            return
        o = ord(ch)
        v.message(f"<{ch}> {o}, Hex {o:02x}, Oct {o:03o}")
        return
    if key == "&":
        from .ex import repeat_sub
        v.mark_change_start()
        repeat_sub(v, l, l, "")
        v.set_dot(keys, n, reg)
        return
    if key == "g&":
        from .ex import repeat_sub
        v.mark_change_start()
        repeat_sub(v, 0, len(lines) - 1, "&", use_last_search=True)
        return
    if key == "m":
        set_mark(v, arg)
        return
    if key == "q":
        if v.recording is not None:
            stop_recording(v)
            return
        if arg in ("<Esc>", "<C-c>"):
            return
        if arg == ":":
            return v.start_cmdline(":", cmdwin=True)
        if not (arg.isalnum() or arg == '"'):
            raise Fail()
        v.recording = arg
        v.rec_keys = []
        v.used["q"] += 1
        return
    if key == "@":
        return play_macro(v, arg, n1)
    if key in ("ZZ", "ZQ"):
        from .ex import run_ex
        return run_ex(v, "x" if key == "ZZ" else "q!", typed=False, alias=key)
    if key == "<C-w>":
        k2, cnt2 = arg
        return v.window_cmd(k2, cnt2 or n)
    if key in ("gt", "gT"):
        if key == "gt":
            if n:
                v.goto_tab(n - 1)
            else:
                v.goto_tab((v.tabidx + 1) % len(v.tabs))
        else:
            v.goto_tab((v.tabidx - n1) % len(v.tabs))
        return
    if key in ("gf", "gF"):
        kw = re.search(r"[\w./~-]+", lines[l][c:]) if lines[l] else None
        word = None
        if lines[l]:
            s = c
            while s > 0 and re.match(r"[\w./~-]", lines[l][s - 1]):
                s -= 1
            m = re.match(r"[\w./~-]+", lines[l][s:])
            word = m.group(0) if m else None
        if not word:
            raise VimError("E446: No file name under cursor")
        path = v.abspath(word)
        if not v.io.exists(path):
            raise VimError(f'E447: Can\'t find file "{word}" in path')
        from .ex import edit_file
        return edit_file(v, word, force=False, cmd="gf")
    if key in ("z.", "z<CR>", "z-", "zt", "zz", "zb"):
        if n:
            v.set_cursor(n - 1)
        if key in ("z.", "z<CR>", "z-"):
            v.set_cursor(w.cursor[0])
        scroll_line_to(v, {"zt": "top", "z<CR>": "top", "zz": "mid", "z.": "mid", "zb": "bot", "z-": "bot"}[key])
        return
    if key in ("zh", "zl", "zs", "ze", "zH", "zL"):
        return
    if key.startswith("z") and key in ("zo", "zO", "zc", "zC", "za", "zA", "zv", "zx", "zX", "zm", "zM", "zr",
                                       "zR", "zn", "zN", "zi", "zE", "zd", "zD", "zF", "zj", "zk"):
        from .folds import fold_command
        return fold_command(v, key, n)
    if key in ("z=", "zg", "zw", "zG", "zW") or key.startswith("zu"):
        from .spell import spell_command
        return spell_command(v, key, n)
    if key == "g;" or key == "g,":
        return
    if key == "<C-z>":
        if v.noquit:
            v.message("Suspending isn't allowed in this exercise: stay in Vim.", "err")
            return
        raise VimError("E1236: Can't suspend here")
    if key in ("K",):
        kw = keyword_at(lines[l], c)
        if not kw:
            raise VimError("E349: No identifier under cursor")
        from .ex import run_ex
        return run_ex(v, "help " + lines[l][kw[0]:kw[1]], typed=False)
    if key in ("<C-]>",):
        raise VimError("E433: No tags file")
    if key in ("gJ",):
        return
    if key == "Q" or key == "gQ":
        raise VimError("Ex mode isn't available in this terminal")
    if key in ("gh", "gH"):
        raise Fail()
    if key in ("g8", "g<"):
        return
    if key == "gx":
        return
    if key == "gs":
        return
    raise Fail()


# ---------------------------------------------------------------- put

def put(v, reg, count, after=True, gp=False, keys=(), given=None):
    text, kind = v.reg_get(reg)
    if reg in ("_",):
        return
    if text == "" and kind != "V":
        raise VimError(f"E353: Nothing in register {reg or '\"'}")
    b, w = v.buf, v.win
    lines = b.lines
    l, c = w.cursor
    if kind == "V":
        body = text[:-1] if text.endswith("\n") else text
        new = body.split("\n") * count
        at = l + 1 if after else l
        f = v.closed_fold_at(l)
        if f and after:
            at = f[1] + 1
        elif f:
            at = f[0]
        v.replace_lines(b, at, at, new)
        b.marks["["] = (at, 0)
        b.marks["]"] = (at + len(new) - 1, 0)
        if gp:
            v.set_cursor(min(len(lines) - 1, at + len(new)), 0)
        else:
            v.set_cursor(at)
        if len(new) > v.opt("report"):
            v.message(f"{len(new)} more lines")
    elif kind == "b":
        rows = text.split("\n")
        width = max((len(r) for r in rows), default=0)
        col = c + (1 if after and lines[l] else 0)
        vc = vcol(lines[l], col, v.ts())
        for i, r in enumerate(rows):
            li = l + i
            if li >= len(lines):
                v.replace_lines(b, len(lines), len(lines), [""])
            line = b.lines[li]
            a = col_at_vcol(line, vc, v.ts(), past_end=True)
            if vcol(line, len(line), v.ts()) < vc:
                line = line + " " * (vc - vcol(line, len(line), v.ts()))
                a = len(line)
            piece = (r + " " * (width - len(r))) * count if i < len(rows) else r
            if a >= len(line):
                piece = piece.rstrip(" ") if True else piece
            v.set_line(b, li, line[:a] + piece + line[a:])
        v.set_cursor(l, col if not gp else col + width * count)
    else:
        body = text * count
        col = c + 1 if after and lines[l] else c
        end = v.insert_text(b, (l, col), body)
        b.marks["["] = (l, col)
        if "\n" in body:
            if gp:
                v.set_cursor(end[0], end[1])
            else:
                v.set_cursor(l, col)
        else:
            if gp:
                w.cursor = [end[0], end[1]]
                v.clamp(insert=True) if False else None
                if end[1] >= len(b.lines[end[0]]):
                    w.cursor = [end[0], len(b.lines[end[0]])]
            else:
                v.set_cursor(end[0], max(0, end[1] - 1))
        b.marks["]"] = (end[0], max(0, end[1] - 1))
    w.curswant = vcol(b.lines[w.cursor[0]], w.cursor[1], v.ts())
    v.set_dot(list(keys), given, reg)
    if reg and reg.isdigit():
        v.dot["reg"] = reg


def join_lines(v, l1, l2, spaces=True, count_msg=True):
    """J / gJ: the cursor ends where the last two lines were joined (on the first space added)."""
    b = v.buf
    lines = b.lines
    cur = lines[l1]
    col = 0
    for l in range(l1 + 1, l2 + 1):
        nxt = lines[l]
        if spaces:
            stripped = nxt.lstrip(" \t")
            if stripped == "" or stripped.startswith(")") or cur == "" or cur.endswith((" ", "\t")):
                sep = ""
            elif v.opt("joinspaces") and cur.endswith((".", "!", "?")):
                sep = "  "
            else:
                sep = " "
            col = len(cur) if sep or not cur.endswith((" ", "\t")) else len(cur) - 1
            if not sep and cur.endswith((" ", "\t")):
                col = len(cur) - 1
            cur = cur + sep + stripped
        else:
            col = len(cur)
            cur = cur + nxt
    v.replace_lines(b, l1, l2 + 1, [cur])
    v.set_cursor(l1, max(0, min(col, len(cur) - 1)))


# ---------------------------------------------------------------- <C-a> <C-x>

def increment(v, l, c, delta, line_text=None):
    """Add delta to the number at or after the cursor (decimal, or hex 0x.., binary 0b..)."""
    b = v.buf
    line = b.lines[l] if line_text is None else line_text
    nf = str(v.opt("nrformats"))
    for m in re.finditer(r"0[xX][0-9a-fA-F]+|0[bB][01]+|-?\d+", line):
        s, e = m.span()
        if e <= c:
            continue
        txt = m.group(0)
        if txt.lower().startswith("0x") and "hex" in nf:
            val = int(txt, 16) + delta
            digits = txt[2:]
            new = txt[:2] + (format(val % (1 << 64), "x" if digits == digits.lower() else "X")).rjust(len(digits), "0")
        elif txt.lower().startswith("0b") and "bin" in nf:
            val = int(txt, 2) + delta
            new = txt[:2] + format(max(0, val), "b").rjust(len(txt) - 2, "0")
        else:
            if txt.startswith("-") and s > 0 and (line[s - 1].isalnum() or line[s - 1] == "_"):
                # "x-1": the minus isn't a sign
                txt = txt[1:]
                s += 1
            val = int(txt) + delta
            neg = txt.startswith("-")
            body = txt.lstrip("-")
            if len(body) > 1 and body.startswith("0") and "octal" in nf:
                val = int(body, 8) * (-1 if neg else 1) + delta
                new = ("-" if val < 0 else "") + format(abs(val), "o").rjust(len(body), "0")
            else:
                new = str(val)
                if body.startswith("0") and len(body) > 1:
                    new = ("-" if val < 0 else "") + str(abs(val)).rjust(len(body), "0")
        newline = line[:s] + new + line[s + len(m.group(0)) - (len(m.group(0)) - len(txt)):]
        newline = line[:s] + new + line[s + len(txt):]
        if line_text is not None:
            return newline
        v.set_line(b, l, newline)
        v.set_cursor(l, s + len(new) - 1)
        return newline
    if line_text is not None:
        return line_text
    raise Fail()


# ---------------------------------------------------------------- marks and macros

def set_mark(v, m):
    w, b = v.win, v.buf
    pos = (w.cursor[0], w.cursor[1])
    if m in ("'", "`"):
        v.setpcmark()
        return
    if m.islower():
        b.marks[m] = pos
    elif m.isupper():
        v.regs["__mark_" + m] = (b.num, pos)
    elif m in "<>[]":
        b.marks[m] = pos
    else:
        raise Fail()
    v.used["m"] += 1


def stop_recording(v):
    keys = v.rec_keys[:-1]       # without the q that stopped it
    reg = v.recording
    text = K.to_text(keys)
    if reg.isupper():
        old, _ = v.regs.get(reg.lower(), ("", "v"))
        v.regs[reg.lower()] = (old + text, "v")
    elif reg == '"':
        v.regs["0"] = (text, "v")
        v.regs["__unnamed"] = "0"
    else:
        v.regs[reg] = (text, "v")
    v.recording = None
    v.rec_keys = []


def play_macro(v, reg, count):
    if reg == "@":
        reg = v.last_macro
        if reg is None:
            raise VimError("E748: No previously used register")
        v.used["@@"] += 1
    if reg == ":":
        from .ex import run_ex
        last = v.regs.get(":", ("", "v"))[0]
        if not last:
            raise VimError("E30: No previous command line")
        v.last_macro = ":"
        for _ in range(count):
            run_ex(v, last, typed=False)
        return
    v.last_macro = reg
    text, kind = v.reg_get(reg)
    toks = K.from_text(text)
    if kind == "V":
        toks = K.from_text(text)
    v.used["@"] += 1
    for k in reversed(toks * count):
        v.typeahead.appendleft((k, True, None))


# ---------------------------------------------------------------- jumps and scrolling

def jump(v, d):
    w = v.win
    if not w.jumps:
        raise Fail()
    if d < 0 and w.jumpidx == len(w.jumps):
        # remember where we are, so <Tab> can come back
        pos = (v.buf.num, w.cursor[0], w.cursor[1])
        w.jumps = [j for j in w.jumps if not (j[0] == pos[0] and j[1] == pos[1])]
        w.jumps.append(pos)
        w.jumpidx = len(w.jumps) - 1
    idx = w.jumpidx + d
    if idx < 0 or idx >= len(w.jumps):
        raise Fail()
    w.jumpidx = idx
    bufnum, l, c = w.jumps[idx]
    if bufnum != v.buf.num:
        tb = v.buffers.get(bufnum)
        if tb is None:
            raise Fail()
        v.edit_buffer(tb, keep_jumps=True)
    v.set_cursor(l, c)


def scroll_half(v, n, down):
    """<C-d> / <C-u>: the view and the cursor move by 'scroll' lines (half the window)."""
    w = v.win
    lines = w.buf.lines
    if n:
        w.opts["scroll"] = n
    amount = w.opts.get("scroll") or max(1, w.height // 2)
    l = w.cursor[0]
    so = min(v.opt("scrolloff"), (w.height - 1) // 2)
    if down:
        if l >= len(lines) - 1:
            raise Fail()
        top, last = v.visible_range(w)
        if last < len(lines) - 1:
            w.topline = min(len(lines) - 1, w.topline + amount)
        nl = min(len(lines) - 1, l + amount)
        top, last = v.visible_range(w)
        if w.topline > 0:
            nl = max(nl, min(len(lines) - 1, w.topline + so))
    else:
        if l == 0:
            raise Fail()
        if w.topline > 0:
            w.topline = max(0, w.topline - amount)
        nl = max(0, l - amount)
        top, last = v.visible_range(w)
        if last < len(lines) - 1:
            nl = min(nl, last - so)
    v.go_line(nl)
    w.keep_top = True


def scroll_page(v, n, down):
    """<C-f> / <C-b>: a page, keeping two lines of the old one."""
    w = v.win
    lines = w.buf.lines
    so = min(v.opt("scrolloff"), (w.height - 1) // 2)
    for _ in range(n):
        if down:
            top, last = v.visible_range(w)
            if top >= len(lines) - 1:
                raise Fail()
            w.topline = max(top + 1, last - 1)
        else:
            if w.topline == 0:
                raise Fail()
            w.topline = max(0, w.topline - (w.height - 2))
    top, last = v.visible_range(w)
    l = w.cursor[0]
    if down:
        nl = max(l, min(len(lines) - 1, w.topline + (so if w.topline > 0 else 0)))
    else:
        nl = min(l, max(0, last - (so if last < len(lines) - 1 else 0)))
    v.go_line(nl)
    w.keep_top = True


def scroll_lines(v, d):
    w = v.win
    lines = w.buf.lines
    w.topline = max(0, min(len(lines) - 1, w.topline + d))
    so = min(v.opt("scrolloff"), (w.height - 1) // 2)
    top, last = v.visible_range(w)
    l = w.cursor[0]
    if l < top + (so if top > 0 else 0):
        v.go_line(min(len(lines) - 1, top + (so if top > 0 else 0)), sol=False)
    elif l > last - (so if last < len(lines) - 1 else 0):
        v.go_line(max(0, last - (so if last < len(lines) - 1 else 0)), sol=False)
    w.keep_top = True


def scroll_line_to(v, where):
    w = v.win
    l = w.cursor[0]
    if where == "top":
        so = min(v.opt("scrolloff"), (w.height - 1) // 2)
        w.topline = max(0, l - so)
    elif where == "mid":
        w.topline = max(0, l - (w.height - 1) // 2)
    else:
        so = min(v.opt("scrolloff"), (w.height - 1) // 2)
        w.topline = max(0, l - (w.height - 1) + so)
    w.keep_top = True

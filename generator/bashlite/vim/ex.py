"""Ex commands (what you type after ":")."""

from __future__ import annotations

import fnmatch
import re

from . import keys as K
from .engine import ABBREV, MAXCOL, OPTIONS, Fail, VimError
from .regex import RegexError, compile_, expand_replacement, literal_tilde
from .text import first_nonblank, indent_of, set_indent

# command name -> (shortest abbreviation, full name)
_CMDS = [
    ("a", "append"), ("ab", "abbreviate"), ("au", "autocmd"), ("aug", "augroup"), ("b", "buffer"), ("bN", "bNext"),
    ("bd", "bdelete"), ("bf", "bfirst"), ("bl", "blast"), ("bm", "bmodified"), ("bn", "bnext"), ("bp", "bprevious"),
    ("br", "brewind"), ("buffers", "buffers"), ("bufdo", "bufdo"), ("bw", "bwipeout"), ("c", "change"),
    ("cN", "cNext"), ("cal", "call"), ("cc", "cc"), ("ccl", "cclose"), ("cd", "cd"), ("cdo", "cdo"),
    ("cfdo", "cfdo"), ("cfir", "cfirst"), ("cla", "clast"), ("clo", "close"), ("cn", "cnext"), ("cnf", "cnfile"),
    ("cno", "cnoremap"), ("co", "copy"), ("colo", "colorscheme"), ("com", "command"), ("cope", "copen"),
    ("cp", "cprevious"), ("cq", "cquit"), ("cr", "crewind"), ("cw", "cwindow"), ("d", "delete"), ("delm", "delmarks"),
    ("di", "display"), ("dig", "digraphs"), ("e", "edit"), ("ea", "earlier"), ("ec", "echo"), ("echom", "echomsg"),
    ("ene", "enew"), ("ex", "ex"), ("exe", "execute"), ("exi", "exit"), ("Ex", "Explore"), ("f", "file"),
    ("files", "files"), ("filet", "filetype"), ("fo", "fold"), ("foldc", "foldclose"), ("foldo", "foldopen"),
    ("g", "global"), ("h", "help"), ("his", "history"), ("i", "insert"), ("im", "imap"), ("ino", "inoremap"),
    ("iu", "iunmap"), ("j", "join"), ("ju", "jumps"), ("k", "k"), ("kee", "keepmarks"), ("keepj", "keepjumps"),
    ("lat", "later"), ("lcd", "lcd"), ("le", "left"), ("Lex", "Lexplore"), ("let", "let"), ("ls", "ls"),
    ("m", "move"), ("ma", "mark"), ("map", "map"), ("marks", "marks"), ("mes", "messages"), ("n", "next"),
    ("new", "new"), ("nm", "nmap"), ("nn", "nnoremap"), ("no", "noremap"), ("noh", "nohlsearch"),
    ("norm", "normal"), ("nun", "nunmap"), ("o", "open"), ("om", "omap"), ("ono", "onoremap"), ("on", "only"),
    ("p", "print"), ("pu", "put"), ("pw", "pwd"), ("q", "quit"), ("qa", "qall"), ("quita", "quitall"),
    ("r", "read"), ("red", "redo"), ("redr", "redraw"), ("reg", "registers"), ("res", "resize"), ("ret", "retab"),
    ("ri", "right"), ("s", "substitute"), ("sN", "sNext"), ("sav", "saveas"), ("se", "set"), ("setg", "setglobal"),
    ("setl", "setlocal"), ("sh", "shell"), ("sil", "silent"), ("sl", "sleep"), ("so", "source"), ("sor", "sort"),
    ("sp", "split"), ("spellg", "spellgood"), ("spellw", "spellwrong"), ("star", "startinsert"), ("sus", "suspend"),
    ("sv", "sview"), ("sy", "syntax"), ("Sex", "Sexplore"), ("t", "t"), ("tabc", "tabclose"), ("tabe", "tabedit"),
    ("tabf", "tabfirst"), ("tabl", "tablast"), ("tabm", "tabmove"), ("tabn", "tabnext"), ("tabnew", "tabnew"),
    ("tabo", "tabonly"), ("tabp", "tabprevious"), ("tabN", "tabNext"), ("tabr", "tabrewind"), ("tabs", "tabs"),
    ("te", "tearoff"), ("ter", "terminal"), ("u", "undo"), ("una", "unabbreviate"), ("unl", "unlet"),
    ("unm", "unmap"), ("up", "update"), ("v", "vglobal"), ("ve", "version"), ("vert", "vertical"),
    ("Vex", "Vexplore"), ("vim", "vimgrep"), ("vm", "vmap"), ("vn", "vnoremap"), ("vne", "vnew"),
    ("vs", "vsplit"), ("vu", "vunmap"), ("w", "write"), ("wN", "wNext"), ("wa", "wall"), ("windo", "windo"),
    ("wn", "wnext"), ("wq", "wq"), ("wqa", "wqall"), ("x", "xit"), ("xa", "xall"), ("xm", "xmap"),
    ("xn", "xnoremap"), ("xu", "xunmap"), ("y", "yank"), ("z", "z"), ("sm", "smap"), ("smapc", "smapclear"),
    ("mapc", "mapclear"), ("nmapc", "nmapclear"), ("cm", "cmap"), ("cu", "cunmap"), ("bufdo", "bufdo"),
    ("tabnew", "tabnew"), ("vsp", "vsplit"), ("bnext", "bnext"), ("Explore", "Explore"), ("Texplore", "Texplore"),
    ("Hexplore", "Hexplore"), ("cex", "cexpr"), ("ca", "cabbrev"), ("spellr", "spellrepall"), ("wv", "wviminfo"),
    ("bun", "bunload"), ("lv", "lvimgrep"), ("gr", "grep"), ("noa", "noautocmd"), ("lefta", "leftabove"),
    ("abo", "aboveleft"), ("bel", "belowright"), ("bo", "botright"), ("to", "topleft"), ("tab", "tab"),
    ("sf", "sfind"), ("fin", "find"), ("ar", "args"), ("argd", "argdelete"), ("arga", "argadd"),
    ("undol", "undolist"), ("changes", "changes"), ("winc", "wincmd"),
]
COMMAND_NAMES = sorted({full for _, full in _CMDS})
_SHORT = {}
for _abbr, _full in _CMDS:
    for _i in range(len(_abbr), len(_full) + 1):
        _SHORT.setdefault(_full[:_i], _full)
    _SHORT[_abbr] = _full
_SHORT.update({"s": "substitute", "g": "global", "v": "vglobal", "d": "delete", "y": "yank", "p": "print",
               "m": "move", "j": "join", "t": "t", "w": "write", "q": "quit", "e": "edit", "r": "read",
               "b": "buffer", "u": "undo", "x": "xit", "k": "k", "h": "help", "f": "file", "c": "change",
               "n": "next", "a": "append", "i": "insert", "o": "open", "z": "z", "ls": "ls"})

# commands that take | as part of their argument
NO_BAR = {"global", "vglobal", "normal", "map", "nmap", "vmap", "xmap", "omap", "imap", "cmap", "noremap",
          "nnoremap", "vnoremap", "xnoremap", "onoremap", "inoremap", "cnoremap", "autocmd", "!", "read!", "write!",
          "terminal", "help", "command", "execute", "windo", "bufdo", "cdo", "cfdo"}


def full_name(name: str) -> str | None:
    if name in _SHORT:
        return _SHORT[name]
    # user-defined capitalized commands, or unknown
    return None


# ---------------------------------------------------------------- running a command line

def run_ex(v, text: str, typed=True, alias=None):
    """Run one command line (it may hold several commands separated by |)."""
    if typed:
        v.regs[":"] = (text, "v")
    v.in_ex = getattr(v, "in_ex", 0) + 1
    try:
        for part in split_bar(v, text):
            run_one(v, part, alias=alias)
            if v.quit is not None:
                break
    finally:
        v.in_ex -= 1


def split_bar(v, text):
    # :s/pat/rep/flags: a | inside the pattern or the replacement belongs to them
    m = re.match(r"(\s*:*[\d.,$%'<>/?\-+ ;]*\s*(?:s|substitute|sno|smagic|snomagic)\s*)([^A-Za-z0-9\s\"|\\])", text)
    if m:
        d = m.group(2)
        i = m.end()
        seen = 0
        while i < len(text) and seen < 2:
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == d:
                seen += 1
            i += 1
        j = text.find("|", i)
        if j < 0:
            return [text]
        return [text[:j]] + split_bar(v, text[j + 1:])
    parts, cur, i = [], [], 0
    stripped = text.lstrip(" :\t")
    name = re.match(r"[\d.,$%'<>/?\-+ ;]*\s*([A-Za-z!&<>~]+|!)", stripped)
    first = name.group(1) if name else ""
    fn = full_name(re.sub(r"!$", "", first)) or first
    if fn in NO_BAR or first.startswith("!") or re.match(r"(r|re|rea|read|w|wr|wri|writ|write)\s*!", stripped) or \
            re.match(r"[\d.,$%'<>/?\-+ ]*!", stripped):
        return [text]
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == '"' and not "".join(cur).strip().startswith(("s", "%s", "'<,'>s")) and False:
            break
        if ch == "|":
            parts.append("".join(cur))
            cur = []
            i += 1
            # what follows may itself be a no-bar command
            rest = text[i:]
            nm = re.match(r"\s*:?\s*[\d.,$%'<>/?\-+ ;]*([A-Za-z]+)", rest)
            if nm and (full_name(nm.group(1)) in NO_BAR):
                parts.append(rest)
                return parts
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts


ADDR = re.compile(r"\s*(?:(\d+)|(\.)|(\$)|'([a-zA-Z<>\[\]'`.^\"])|/((?:\\.|[^/])*)/?|\?((?:\\.|[^?])*)\??|\\([/?&]))?((?:\s*[+-]\s*\d*)*)")


def parse_range(v, text):
    """-> (line1, line2, count of addresses, rest). Lines are 0-based; None when no range was given."""
    i = 0
    s = text.lstrip(" :\t")
    if s.startswith("%"):
        rest = s[1:]
        # "%" may also be followed by more (e.g. %s)
        return 0, len(v.lines) - 1, 2, rest
    addrs = []
    cur = v.win.cursor[0]
    while True:
        m = ADDR.match(s, i)
        if not m or m.end() == i:
            # nothing more
            if s[i:i + 1] in (",", ";"):
                addrs.append(cur)
                if s[i] == ";":
                    cur = addrs[-1]
                i += 1
                continue
            break
        num, dot, dollar, mark, fpat, bpat, back, offs = m.groups()
        if num is None and dot is None and dollar is None and mark is None and fpat is None and bpat is None and \
                back is None and not offs.strip():
            break
        base = cur
        if num is not None:
            base = int(num) - 1 if int(num) > 0 else -1
        elif dot:
            base = cur
        elif dollar:
            base = len(v.lines) - 1
        elif mark:
            if mark in ("<", ">"):
                pos = v.get_mark(mark)
                base = pos[0]
            else:
                base = v.get_mark(mark)[0]
        elif fpat is not None:
            pat = fpat or v.last_pat
            if pat is None:
                raise VimError("E35: No previous regular expression")
            v.last_pat = pat
            base = find_line(v, pat, cur, 1)
        elif bpat is not None:
            pat = bpat or v.last_pat
            v.last_pat = pat
            base = find_line(v, pat, cur, -1)
        elif back:
            pat = v.last_pat if back in "/?" else (v.last_s[0] if v.last_s else None)
            if pat is None:
                raise VimError("E35: No previous regular expression")
            base = find_line(v, pat, cur, 1 if back in "/&" else -1)
        for om in re.finditer(r"([+-])\s*(\d*)", offs or ""):
            n = int(om.group(2)) if om.group(2) else 1
            base += n if om.group(1) == "+" else -n
        addrs.append(base)
        i = m.end()
        if s[i:i + 1] == ",":
            i += 1
            continue
        if s[i:i + 1] == ";":
            cur = base
            v.win.cursor = [max(0, min(len(v.lines) - 1, base)), 0]
            i += 1
            continue
        break
    rest = s[i:]
    if not addrs:
        return None, None, 0, rest
    if len(addrs) == 1:
        return addrs[0], addrs[0], 1, rest
    return addrs[-2], addrs[-1], 2, rest


def find_line(v, pat, cur, d):
    from .search import compiled
    cre = compiled(v, pat)
    n = len(v.lines)
    for k in range(1, n + 1):
        l = (cur + k * d) % n
        if cre.rx.search(v.lines[l]):
            return l
    raise VimError(f"E486: Pattern not found: {pat}")


def run_one(v, text, alias=None):
    raw = text
    text = text.lstrip(" \t:")
    if not text or text.startswith('"'):
        return
    l1, l2, naddr, rest = parse_range(v, text)
    rest = rest.lstrip(" ")
    if not rest or rest.startswith('"'):
        if naddr:
            # a line number on its own: go there
            ln = l2
            if ln < -1 or ln >= len(v.lines) + 1:
                ln = max(0, min(len(v.lines) - 1, ln))
            ln = max(0, min(len(v.lines) - 1, ln))
            v.setpcmark()
            v.go_line(ln, sol=True)
            v.used[":N"] += 1
            return
        return
    m = re.match(r"([A-Za-z]+|[!&<>~=@#*]|s(?=[^A-Za-z0-9\s\"|]))", rest)
    if not m:
        raise VimError(f"E492: Not an editor command: {raw.strip()}")
    word = m.group(1)
    after = rest[m.end():]
    # s/// with a non-letter delimiter right after s, g, v
    mm = re.match(r"(s|substitute|g|global|v|vglobal|sno|snomagic|sm|smagic)([^A-Za-z0-9\s\"|])", rest)
    if mm:
        word = mm.group(1)
        after = rest[len(word):]
    # :k{mark} without a space
    mk = re.match(r"k([a-zA-Z])$", rest)
    if mk:
        word, after = "k", " " + mk.group(1)
    bang = after.startswith("!") and word not in ("!",)
    if bang:
        after = after[1:]
    name = full_name(word) if word not in ("!", "&", "<", ">", "~", "=", "@", "*", "#") else word
    if name is None:
        # :s with flags (:s/x/y/), or :sno etc.
        if re.fullmatch(r"s[cegiInp#lr&]*", word):
            name = "substitute"
            after = word[1:] + after
        else:
            raise VimError(f"E492: Not an editor command: {raw.strip()}")
    arg = after.strip() if name not in ("substitute", "global", "vglobal", "normal", "!", "&", "~") else after
    if name not in ("substitute", "global", "vglobal", "normal"):
        arg = after.strip()
    ctx = {"l1": l1, "l2": l2, "naddr": naddr, "bang": bang, "arg": arg, "raw": raw, "alias": alias, "name": name}
    canon = ":" + (word if word in ("!", "&", "<", ">", "~", "=") else name)
    v.used[canon] += 1
    short = ":" + {"substitute": "s", "global": "g", "vglobal": "v", "normal": "normal", "write": "w", "quit": "q",
                   "wq": "wq", "xit": "x", "edit": "e", "split": "sp", "vsplit": "vsp", "buffer": "b",
                   "bnext": "bn", "bprevious": "bp", "bdelete": "bd", "tabnew": "tabnew", "nohlsearch": "noh",
                   "registers": "reg", "earlier": "earlier", "later": "later", "read": "r", "sort": "sort",
                   "move": "m", "t": "t", "copy": "t", "delete": "d", "yank": "y", "join": "j", "qall": "qa",
                   "wall": "wa", "set": "set", "put": "pu", "cnext": "cn", "cprevious": "cp", "wqall": "wqa",
                   "xall": "xa", "tabclose": "tabclose", "Explore": "Explore", "Vexplore": "Vexplore"}.get(name, name)
    v.used[short] += 1
    if bang:
        v.used[short + "!"] += 1
    if naddr:
        if l1 == 0 and l2 == len(v.lines) - 1 and text.lstrip().startswith("%"):
            v.used[":%" + short[1:]] += 1
        elif text.lstrip().startswith("'<,'>"):
            v.used[":'<,'>" + short[1:]] += 1
        else:
            v.used[":range" + short[1:]] += 1
    fn = COMMANDS.get(name)
    if fn is None:
        raise VimError(f"E492: Not an editor command: {raw.strip()}")
    if l1 is not None and l2 is not None and l1 > l2 and name not in ("vertical",):
        l1, l2 = l2, l1
        ctx["l1"], ctx["l2"] = l1, l2
    if naddr and name not in ("help",):
        n = len(v.lines)
        allow0 = name in ("read", "put", "move", "t", "copy", "append", "insert", "global", "vglobal", "earlier",
                          "later", "resize", "tabnew", "tabnext", "tabclose", "tabmove", "buffer", "bdelete", "cc",
                          "vertical", "undo", "z", "normal")
        lo = -1 if allow0 else 0
        if l1 < lo or l2 >= n and name not in ("resize", "tabnew", "tabnext", "tabclose", "buffer", "bdelete", "cc",
                                                 "vertical", "undo", "earlier", "later", "z", "tabmove"):
            raise VimError("E16: Invalid range")
    return fn(v, ctx)


# ---------------------------------------------------------------- files

def expand_fname(v, s, for_shell=False):
    """% and # (with :p :h :t :r :e) in file names and shell commands."""
    def repl(m):
        if m.group(1):
            return m.group(1)[1:]
        which = m.group(2)
        mods = m.group(3) or ""
        if which == "%":
            b = v.buf
        else:
            num = m.group(4)
            b = v.buffers.get(int(num)) if num else v.buffers.get(v.win.alt)
            if b is None:
                raise VimError("E194: No alternate file name to substitute for '#'")
        name = b.name
        if not name:
            raise VimError("E499: Empty file name for '%' or '#', only works with \":p:h\"")
        for mod in re.findall(r":([phtre])", mods):
            if mod == "p":
                name = b.path or v.abspath(name)
            elif mod == "h":
                name = name.rsplit("/", 1)[0] if "/" in name else "."
            elif mod == "t":
                name = name.rsplit("/", 1)[-1]
            elif mod == "r":
                name = re.sub(r"\.[^./]*$", "", name)
            elif mod == "e":
                mm = re.search(r"\.([^./]*)$", name)
                name = mm.group(1) if mm else ""
        return name
    return re.sub(r"(\\[%#])|([%#])((?::[phtre])*)(\d*)", lambda m: repl(m) if m.group(1) is None else m.group(1)[1:], s) \
        if True else s


def write_buffer(v, b, path=None, force=False, l1=None, l2=None, append=False, quiet=False):
    """Write lines l1..l2 (default: all) to path (default: the buffer's file). -> True if written."""
    lines = b.lines
    if l1 is None:
        l1, l2 = 0, len(lines) - 1
    whole = l1 == 0 and l2 == len(lines) - 1
    target = v.abspath(path) if path else b.path
    if target is None:
        raise VimError("E32: No file name")
    own = target == b.path
    if own and b.readonly and not force:
        raise VimError("E45: 'readonly' option is set (add ! to override)")
    if not own and not force and not append and v.io.exists(target) and whole and path is not None:
        raise VimError("E13: File exists (add ! to override)")
    if not own and not whole and not force and v.io.exists(target) and not append:
        raise VimError("E140: Use ! to write partial buffer")
    if own:
        v.fire("BufWritePre", b)
        lines = b.lines
        l2 = len(lines) - 1 if whole else l2
    nl = "\r\n" if b.ff == "dos" else "\n"
    body = nl.join(lines[l1:l2 + 1])
    text = body + (nl if (b.eol or True) else "")
    if append:
        old = v.io.read(target) or ""
        text = old + text
    new = not v.io.exists(target)
    err = v.io.write(target, text, force=force)
    if err:
        raise VimError(f"E212: Can't open file for writing" if "ermission" in err or "denied" in err else f"E212: Can't open file for writing: {err}")
    nlines = l2 - l1 + 1
    size = len(text.encode())
    if own and whole:
        b.modified = False
        b.saved_seq = b.undo.cur.seq
        b.new_file = False
        b.disk = text
        if b.opts.get("undofile") or v.o.get("undofile"):
            v.undo_store[b.path] = b.undo
    if not quiet:
        shown = v.relname(target) if not own else b.name
        v.message(f'"{shown}" {"[New] " if new else ""}{"[dos] " if b.ff == "dos" else ""}{nlines}L, {size}B '
                  f'{"appended" if append else "written"}')
    if own and whole:
        v.fire("BufWritePost", b)
    return True


def edit_file(v, name, force=False, cmd="e"):
    """:e file (and gf, netrw, :b): show `name` in the current window."""
    b = v.buf
    if not name:
        # :e: reload the current file
        if b.modified and not force:
            raise VimError("E37: No write since last change (add ! to override)")
        if b.path is None:
            raise VimError("E32: No file name")
        keep = tuple(v.win.cursor)
        v.read_into(b, reload=True)
        v.commit_undo()
        v.set_cursor(min(keep[0], len(b.lines) - 1), keep[1])
        v.file_msg(b)
        return
    path = v.abspath(expand_fname(v, name))
    target = v.find_buffer(path)
    if target is b:
        if force:
            return edit_file(v, "", force=True)
        v.file_msg(b)
        return
    if b.modified and not force and not v.opt("hidden") and not buf_shown_elsewhere(v, b):
        raise VimError("E37: No write since last change (add ! to override)")
    if force and b.modified and not v.opt("hidden"):
        # the changes are thrown away
        v.read_into(b, reload=True)
    if target is None:
        target = v.load_buffer(name if not name.startswith("~") else path)
    v.edit_buffer(target)
    v.file_msg(target)


def buf_shown_elsewhere(v, b):
    return sum(1 for w in v.all_windows(all_tabs=True) if w.buf is b) > 1


def request_quit(v, status=0, force=False, all_=False, write=False):
    if v.noquit:
        v.message("Quitting is turned off in this exercise (you'll practise it in 9.01): your edits are checked "
                  "right here, so press Submit.", "err")
        v.errors += 1
        return
    v.quit = status


# ---------------------------------------------------------------- the commands

def c_write(v, x):
    arg = x["arg"]
    if arg.startswith("!"):
        # :w !cmd: the lines go to the command's standard input
        cmd = expand_fname(v, arg[1:].strip())
        l1, l2 = (x["l1"], x["l2"]) if x["naddr"] else (0, len(v.lines) - 1)
        out = v.io.run(cmd, stdin="\n".join(v.lines[l1:l2 + 1]) + "\n", cwd=v.cwd)
        v.used[":w !"] += 1
        if re.match(r"sudo\s+tee\b", arg[1:].strip()):
            v.used[":w !sudo tee %"] += 1
        show_output(v, f":{x['raw'].strip()}", out)
        check_changed_on_disk(v)
        return
    append = False
    if arg.startswith(">>"):
        append = True
        arg = arg[2:].strip()
    l1, l2 = (x["l1"], x["l2"]) if x["naddr"] else (None, None)
    write_buffer(v, v.buf, expand_fname(v, arg) if arg else None, force=x["bang"], l1=l1, l2=l2, append=append)
    if x["name"] in ("wq",):
        pass


def c_update(v, x):
    if v.buf.modified:
        write_buffer(v, v.buf, force=x["bang"])


def c_wq(v, x):
    if x["name"] == "xit":
        if v.buf.modified or (x["arg"] and x["arg"] != ""):
            write_buffer(v, v.buf, expand_fname(v, x["arg"]) if x["arg"] else None, force=x["bang"])
    else:
        write_buffer(v, v.buf, expand_fname(v, x["arg"]) if x["arg"] else None, force=x["bang"])
    return close_or_quit(v, force=x["bang"])


def close_or_quit(v, force=False):
    wins = v.all_windows()
    if len(wins) > 1 or len(v.tabs) > 1:
        return v.close_window(v.win, force=True)
    others = [b for b in v.buffers.values() if b is not v.buf and b.modified and b.listed and b.kind == "file"]
    if others and not force:
        b = others[0]
        raise VimError(f'E162: No write since last change for buffer "{b.name}"')
    request_quit(v, 0)


def c_quit(v, x):
    wins = v.all_windows()
    b = v.buf
    if len(wins) > 1 or len(v.tabs) > 1:
        if b.kind == "terminal" and b.extra.get("running") and not x["bang"]:
            raise VimError(f'E947: Job still running in buffer "{b.name}"')
        if b.modified and b.kind == "file" and not x["bang"] and not v.opt("hidden") and \
                not buf_shown_elsewhere(v, b):
            raise VimError("E37: No write since last change (add ! to override)")
        return v.close_window(v.win, force=x["bang"])
    if b.modified and not x["bang"] and b.kind == "file":
        raise VimError("E37: No write since last change (add ! to override)")
    if not x["bang"]:
        others = [o for o in v.buffers.values() if o is not b and o.modified and o.listed and o.kind == "file"]
        if others:
            raise VimError(f'E162: No write since last change for buffer "{others[0].name}"')
    request_quit(v, 0)


def c_qall(v, x):
    if not x["bang"]:
        mod = [b for b in v.buffers.values() if b.modified and b.kind == "file"]
        if mod:
            raise VimError(f'E37: No write since last change' if mod[0] is v.buf else
                           f'E162: No write since last change for buffer "{mod[0].name}"')
    request_quit(v, 0)


def c_wall(v, x):
    n = 0
    for b in list(v.buffers.values()):
        if b.modified and b.kind == "file" and b.path:
            write_buffer(v, b, force=x["bang"], quiet=True)
            n += 1
    v.message("")
    if x["name"] in ("wqall", "xall"):
        request_quit(v, 0)


def c_cquit(v, x):
    request_quit(v, 1)


def c_edit(v, x):
    arg = x["arg"]
    if arg.startswith("+"):
        m = re.match(r"\+(\d+|\$)?\s*(.*)", arg)
        line, arg = m.group(1), m.group(2)
        edit_file(v, arg, force=x["bang"])
        if line:
            v.go_line(len(v.lines) - 1 if line == "$" else int(line) - 1)
        return
    if arg == "#" or re.fullmatch(r"#\d+", arg):
        tb = v.buffers.get(int(arg[1:])) if len(arg) > 1 else v.buffers.get(v.win.alt)
        if tb is None:
            raise VimError("E194: No alternate file name to substitute for '#'")
        if v.buf.modified and not x["bang"] and not v.opt("hidden"):
            raise VimError("E37: No write since last change (add ! to override)")
        v.edit_buffer(tb)
        v.file_msg(tb)
        return
    return edit_file(v, arg, force=x["bang"])


def c_enew(v, x):
    if v.buf.modified and not x["bang"] and not v.opt("hidden"):
        raise VimError("E37: No write since last change (add ! to override)")
    b = v.new_buffer()
    v.edit_buffer(b)


def c_saveas(v, x):
    if not x["arg"]:
        raise VimError("E471: Argument required")
    b = v.buf
    path = v.abspath(expand_fname(v, x["arg"]))
    if v.io.exists(path) and not x["bang"]:
        raise VimError("E13: File exists (add ! to override)")
    old = b.path
    b.path = path
    b.name = v.relname(path)
    write_buffer(v, b, force=True)
    if old:
        alt = v.load_buffer(v.relname(old))
        v.win.alt = alt.num
    v.apply_filetype(b)


def c_file(v, x):
    if x["arg"]:
        b = v.buf
        b.path = v.abspath(x["arg"])
        b.name = v.relname(b.path)
        b.new_file = not v.io.exists(b.path)
        b.modified = True
    v.file_msg(v.buf)


def c_read(v, x):
    arg = x["arg"]
    l = x["l2"] if x["naddr"] else v.win.cursor[0]
    if arg.startswith("!"):
        cmd = expand_fname(v, arg[1:].strip())
        out = v.io.run(cmd, cwd=v.cwd)
        text = out["stdout"]
        v.used[":r !"] += 1
    else:
        name = expand_fname(v, arg) if arg else v.buf.name
        text = v.io.read(v.abspath(name))
        if text is None:
            raise VimError(f'E484: Can\'t open file {name}')
    if text.endswith("\n"):
        text = text[:-1]
    new = text.split("\n") if text != "" else []
    if not new:
        return
    v.check_modifiable()
    at = l + 1
    v.replace_lines(v.buf, at, at, new)
    v.set_cursor(at)
    if len(new) > v.opt("report") and not arg.startswith("!"):
        v.message(f'"{arg}" {len(new)}L')


def c_bang(v, x):
    """:!cmd, and :{range}!filter."""
    arg = x["arg"] if not x["arg"].startswith("!") else x["arg"]
    cmd = x["raw"].lstrip(" :")
    cmd = cmd[cmd.index("!") + 1:]
    if cmd.startswith("!"):
        last = getattr(v, "last_bang", None)
        if last is None:
            raise VimError("E34: No previous command")
        cmd = last + cmd[1:]
    v.last_bang = cmd
    cmd = expand_fname(v, cmd.strip())
    if x["naddr"]:
        l1, l2 = x["l1"], x["l2"]
        v.check_modifiable()
        out = v.io.run(cmd, stdin="\n".join(v.lines[l1:l2 + 1]) + "\n", cwd=v.cwd)
        text = out["stdout"] + out["stderr"]
        if text.endswith("\n"):
            text = text[:-1]
        new = text.split("\n") if text else []
        v.mark_change_start()
        v.replace_lines(v.buf, l1, l2 + 1, new if new else [])
        v.go_line(l1, sol=True)
        n = l2 - l1 + 1
        if n > v.opt("report") or True:
            v.message(f"{n} lines filtered" if n > 1 else "")
        v.used[":range!"] += 1
        return
    out = v.io.run(cmd, cwd=v.cwd)
    show_output(v, f":!{cmd}", out, shell=True)
    check_changed_on_disk(v)


def show_output(v, header, out, shell=False):
    text = (out.get("stdout") or "") + (out.get("stderr") or "")
    lines = [header, ""] if shell else [header]
    if text:
        lines += text.rstrip("\n").split("\n")
    if shell and out.get("status"):
        lines += ["", f"shell returned {out['status']}"]
    v.more = lines
    v.more_kind = "shell"


def check_changed_on_disk(v):
    """After a shell command: a file that changed under Vim asks what to do (W11 / W12)."""
    b = v.buf
    if b.kind != "file" or not b.path:
        return
    disk = v.io.read(b.path)
    known = getattr(b, "disk", None)
    if known is None:
        known = b.text() if not b.modified else None
    if disk is None or known is None or disk == known:
        return
    b.disk = disk
    w = "W12" if b.modified else "W11"
    msg = (f'{w}: Warning: File "{b.name}" has changed ' +
           ("and the buffer was changed in Vim as well" if b.modified else "since editing started"))
    v.pending_prompt = {"kind": "changed", "lines": [msg, f'See ":help {w}" for more info.', "[O]K, (L)oad File: "]}


def c_substitute(v, x):
    return substitute(v, x)


def parse_sub(v, s):
    """'/pat/rep/flags count' -> (pat, rep, flags, count)."""
    if not s or s[0].isalnum() or s[0] in "\\\"| ":
        return None
    d = s[0]
    parts = []
    cur = []
    i = 1
    while i < len(s) and len(parts) < 2:
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            if s[i + 1] == d and d not in "\\":
                cur.append(d if len(parts) == 1 or d != "/" else "\\" + d if False else d)
                if len(parts) == 0 and d in "./*[~^$":
                    cur[-1] = "\\" + d
                i += 2
                continue
            cur.append(c + s[i + 1])
            i += 2
            continue
        if c == d:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if len(parts) < 2:
        parts.append("".join(cur))
        rest = ""
    else:
        rest = s[i:]
    pat = parts[0]
    rep = parts[1] if len(parts) > 1 else ""
    if len(parts) == 1:
        rep = ""
    m = re.match(r"([&cegiInp#lr]*)\s*(\d*)\s*(.*)$", rest)
    flags, count, tail = (m.group(1), m.group(2), m.group(3)) if m else ("", "", rest)
    return pat, rep, flags, int(count) if count else None


def substitute(v, x, repeat=None):
    arg = x["arg"]
    b = v.buf
    if repeat is not None:
        pat, rep, flags, count = repeat
    else:
        parsed = parse_sub(v, arg)
        if parsed is None:
            # :s (repeat the last one, with new flags)
            if v.last_s is None:
                raise VimError("E35: No previous regular expression")
            m = re.match(r"\s*([&cegiInp#lr]*)\s*(\d*)", arg)
            fl = m.group(1)
            pat, rep, oldflags = v.last_s
            flags = (oldflags if fl.startswith("&") else "") + fl.lstrip("&")
            count = int(m.group(2)) if m.group(2) else None
        else:
            pat, rep, flags, count = parsed
            if flags.startswith("&") and v.last_s:
                flags = v.last_s[2] + flags[1:]
        if pat == "":
            pat = v.last_pat if "r" in flags or True else (v.last_s[0] if v.last_s else None)
            if pat is None:
                raise VimError("E35: No previous regular expression")
        if rep == "~":
            rep = v.last_sub or ""
        rep = literal_tilde(rep, v.last_sub or "")
        v.last_sub = rep
        v.last_s = (pat, rep, flags.replace("&", ""))
        v.last_pat = pat
        v.regs["/"] = (pat, "v")
        v.add_hist("/", pat)
    v.no_hl = False
    for f in flags:
        if f in "gcne":
            v.used[":s_" + f] += 1
    for tok in ("\\v", "\\<", "\\(", "\\s", "\\zs", "\\r", "&", "\\1", "\\u", "\\U"):
        if tok in pat or tok in rep:
            v.used[tok] += 1
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    if count:
        l1 = l2
        l2 = min(len(b.lines) - 1, l1 + count - 1)
    try:
        cre = compile_(pat, v.opt("ignorecase") if "I" not in flags else False,
                       v.opt("smartcase") and "i" not in flags and "I" not in flags, v.last_sub or "", v.opt("magic"))
        if "i" in flags:
            cre = compile_("\\c" + pat, False, False, v.last_sub or "", v.opt("magic"))
    except RegexError as e:
        raise VimError(str(e))
    glob = ("g" in flags) != bool(v.opt("gdefault"))
    if "n" in flags:
        total, nlines = 0, 0
        for l in range(l1, l2 + 1):
            k = sum(1 for _ in _line_matches(cre, b.lines, l, glob))
            if k:
                total += k
                nlines += 1
        if total == 0:
            if "e" in flags:
                return
            raise VimError(f"E486: Pattern not found: {pat}")
        v.message(f"{total} match{'es' if total != 1 else ''} on {nlines} line{'s' if nlines != 1 else ''}")
        return
    if "c" in flags:
        v.confirm = {"cre": cre, "rep": rep, "glob": glob, "l": l1, "l2": l2, "col": 0, "all": False,
                     "count": 0, "lines": set(), "pat": pat, "flags": flags, "start": tuple(v.win.cursor)}
        v.mark_change_start()
        return next_confirm(v)
    v.mark_change_start()
    total, changed = 0, 0
    last_line = None
    l = l1
    end = l2
    while l <= end and l < len(b.lines):
        res = sub_line(v, cre, rep, l, glob)
        if res is None:
            l += 1
            continue
        n, added = res
        total += n
        changed += 1
        last_line = l + added
        end += added
        l += added + 1
    if total == 0:
        if "e" in flags:
            return
        raise VimError(f"E486: Pattern not found: {pat}")
    if last_line is not None:
        v.set_cursor(min(last_line, len(b.lines) - 1))
    if total > v.opt("report"):
        v.message(f"{total} substitution{'s' if total != 1 else ''} on {changed} line{'s' if changed != 1 else ''}")


def _line_matches(cre, lines, l, glob):
    """Matches that start on line l (searching the text from that line on)."""
    text = "\n".join(lines[l:])
    first = len(lines[l])
    pos = 0
    while pos <= first:
        m = cre.rx.search(text, pos)
        if m is None:
            return
        s, e = cre.span(m)
        if s > first:
            return
        yield m
        if not glob:
            return
        pos = e if e > s else s + 1
        if e == s and s >= first:
            return


def sub_line(v, cre, rep, l, glob, only=None):
    """Substitute on line l. -> (number of substitutions, lines added) or None. only: the column of the one match."""
    b = v.buf
    lines = b.lines
    text_from = "\n".join(lines[l:])
    first = len(lines[l])
    out = []
    pos = 0
    count = 0
    last_end = 0
    consumed_to = first
    while pos <= first:
        m = cre.rx.search(text_from, pos)
        if m is None:
            break
        s, e = cre.span(m)
        if s > first:
            break
        if only is not None and s != only:
            pos = e if e > s else s + 1
            if not glob and only is None:
                break
            continue
        out.append(text_from[last_end:s])
        out.append(expand_replacement(rep, m, cre, v.last_sub or "", v.opt("magic")))
        last_end = e
        count += 1
        consumed_to = max(consumed_to, e)
        if not glob or only is not None:
            break
        pos = e if e > s else s + 1
    if count == 0:
        return None
    # the rest of the text affected (up to the end of the last line touched)
    end_line_off = text_from.find("\n", consumed_to) if consumed_to >= first else first
    if consumed_to <= first:
        end_line_off = first
    if end_line_off < 0:
        end_line_off = len(text_from)
    out.append(text_from[last_end:end_line_off])
    new_text = "".join(out)
    nlines_old = text_from[:end_line_off].count("\n") + 1
    new_lines = new_text.split("\n")
    new_lines = [x.replace("\x00", "\n") if False else x for x in new_lines]
    v.replace_lines(b, l, l + nlines_old, new_lines)
    return count, len(new_lines) - 1 - (nlines_old - 1) + (nlines_old - 1) - (nlines_old - 1)


def next_confirm(v):
    c = v.confirm
    nxt = _next_match(c, v.buf)
    if nxt is None:
        return finish_confirm(v)
    l, s0, e0 = nxt
    c["cur"] = (l, s0, e0)
    v.win.cursor = [l, min(s0, max(0, len(v.buf.lines[l]) - 1))]
    v.scroll_to_cursor()
    v.message(f"replace with {c['rep']} (y/n/a/q/l/^E/^Y)?", "mode")


def _replace_at(v, c, l, s):
    """Replace the match starting at (l, s). -> (column just after the replacement, line it ends on)."""
    b = v.buf
    text = "\n".join(b.lines[l:])
    m = c["cre"].rx.search(text, s)
    while m is not None and c["cre"].span(m)[0] != s:
        m = c["cre"].rx.search(text, m.start() + 1)
    if m is None:
        return s + 1, l
    rep = expand_replacement(c["rep"], m, c["cre"], v.last_sub or "", v.opt("magic"))
    res = sub_line(v, c["cre"], c["rep"], l, True, only=s)
    c["count"] += 1
    c["lines"].add(l)
    if "\n" in rep:
        k = rep.count("\n")
        c["l2"] += k
        return len(rep.rsplit("\n", 1)[1]), l + k
    return s + len(rep), l


def _next_match(c, b):
    """The next match at or after (c["l"], c["col"]) within the range. -> (l, s, e) or None."""
    while c["l"] <= c["l2"] and c["l"] < len(b.lines):
        for m in _line_matches(c["cre"], b.lines, c["l"], True):
            s0, e0 = c["cre"].span(m)
            if s0 >= c["col"]:
                return c["l"], s0, e0
        c["l"] += 1
        c["col"] = 0
    return None


def confirm_key(v, key):
    """y n a q l (and <Esc>) at :s///c's question."""
    c = v.confirm
    b = v.buf
    l, s, e = c["cur"]
    c["last"] = key if key in ("y", "n", "a", "l") else c.get("last")
    if key in ("y", "l"):
        c["last_line"] = l
    if key in ("q", "<Esc>", "<C-c>"):
        return finish_confirm(v, quit_=True)
    if key not in ("y", "n", "a", "l"):
        return
    if key == "n":
        col, line = (e if e > s else s + 1), l
    else:
        col, line = _replace_at(v, c, l, s)
    if key == "l":
        return finish_confirm(v)
    c["l"], c["col"] = line, col
    if not c["glob"]:
        c["l"], c["col"] = line + 1, 0
    if key == "a":
        while True:
            nxt = _next_match(c, b)
            if nxt is None:
                break
            l2, s2, _ = nxt
            col, line = _replace_at(v, c, l2, s2)
            c["l"], c["col"] = (line, col) if c["glob"] else (line + 1, 0)
        return finish_confirm(v)
    return next_confirm(v)


def finish_confirm(v, quit_=False):
    c = v.confirm
    v.confirm = None
    v.message("")
    if (quit_ or c.get("last") == "n") and "cur" in c:
        # stopped (or said no to the last match): the cursor stays on the match
        l, s0, _ = c["cur"]
        v.win.cursor = [l, s0]
        v.clamp()
    elif c["lines"]:
        v.set_cursor(max(c["lines"]) if c.get("last") == "a" else c.get("last_line", max(c["lines"])), 0)
    else:
        v.set_cursor(*c["start"])
    if c["count"] > v.opt("report"):
        n, nl = c["count"], len(c["lines"])
        v.message(f"{n} substitution{'s' if n != 1 else ''} on {nl} line{'s' if nl != 1 else ''}")
    v.commit_undo()


def repeat_sub(v, l1, l2, flags, use_last_search=False):
    if v.last_s is None:
        raise VimError("E35: No previous regular expression")
    pat, rep, oldflags = v.last_s
    if use_last_search and v.last_pat:
        pat = v.last_pat
    fl = oldflags if "&" in flags else ""
    return substitute(v, {"l1": l1, "l2": l2, "naddr": 2, "arg": "", "bang": False}, repeat=(pat, rep, fl, None))


def c_amp(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    flags = x["arg"].strip()
    if x["raw"].lstrip(" :").startswith("&&") or flags.startswith("&"):
        flags = "&" + flags.lstrip("&")
    return repeat_sub(v, l1, l2, flags)


def c_tilde(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    return repeat_sub(v, l1, l2, x["arg"], use_last_search=True)


def c_global(v, x, invert=False):
    arg = x["arg"]
    if x["name"] == "vglobal" or x["bang"]:
        invert = True
    if not arg:
        raise VimError("E148: Regular expression missing from :global")
    d = arg[0]
    i = 1
    pat = []
    while i < len(arg):
        if arg[i] == "\\" and i + 1 < len(arg):
            if arg[i + 1] == d:
                pat.append(d)
            else:
                pat.append(arg[i:i + 2])
            i += 2
            continue
        if arg[i] == d:
            i += 1
            break
        pat.append(arg[i])
        i += 1
    pat = "".join(pat)
    cmd = arg[i:] or "p"
    if re.match(r"\s*norm", cmd):
        v.used[":g_normal"] += 1
    if pat == "":
        pat = v.last_pat
        if pat is None:
            raise VimError("E35: No previous regular expression")
    v.last_pat = pat
    v.add_hist("/", pat)
    from .search import compiled
    cre = compiled(v, pat)
    b = v.buf
    l1 = x["l1"] if x["naddr"] else 0
    l2 = x["l2"] if x["naddr"] else len(b.lines) - 1
    marked = [l for l in range(l1, l2 + 1) if bool(cre.rx.search(b.lines[l])) != invert]
    if not marked:
        if invert:
            v.message(f"Pattern found in every line: {pat}")
        else:
            v.message(f"Pattern not found: {pat}", "err")
        return
    v.g_lines = marked
    v.mark_change_start()
    printed = []
    before = len(b.lines)
    try:
        idx = 0
        while idx < len(v.g_lines):
            l = v.g_lines[idx]
            idx += 1
            if l is None or l >= len(b.lines):
                continue
            v.win.cursor = [l, 0]
            if cmd.strip() in ("p", "print", "#", "nu", "number"):
                printed.append(b.lines[l])
                continue
            try:
                run_ex(v, cmd, typed=False)
            except VimError as e:
                if e.msg:
                    v.message(e.msg, "err")
                break
    finally:
        v.g_lines = None
    if printed:
        v.more = printed
        v.more_kind = "list"
    diff = len(b.lines) - before
    if diff < 0 and -diff > v.opt("report"):
        v.message(f"{-diff} fewer lines")
    elif diff > 0 and diff > v.opt("report"):
        v.message(f"{diff} more lines")


def c_normal(v, x):
    keys_text = x["arg"]
    if keys_text.startswith(" "):
        keys_text = keys_text[1:]
    toks = K.parse(keys_text, v.leader) if False else list(keys_text)
    # :normal takes the keys literally (special keys only via :execute "normal \<Esc>")
    toks = [K.CHAR_KEY.get(ch, ch) for ch in keys_text]
    if x["naddr"]:
        lines = list(range(x["l1"], x["l2"] + 1))
        v.g_lines = lines if getattr(v, "g_lines", None) is None else v.g_lines
        own = v.g_lines is lines
        try:
            idx = 0
            while idx < len(lines):
                l = lines[idx]
                idx += 1
                if l is None or l >= len(v.lines):
                    continue
                v.win.cursor = [l, 0]
                v.run_keys(toks, remap=not x["bang"])
        finally:
            if own:
                v.g_lines = None
    else:
        v.run_keys(toks, remap=not x["bang"])


def c_delete(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    reg, count = _reg_count(x["arg"])
    if count:
        l1 = l2
        l2 = min(len(v.lines) - 1, l1 + count - 1)
    v.check_modifiable()
    v.mark_change_start()
    text = "\n".join(v.lines[l1:l2 + 1]) + "\n"
    v.reg_set(reg, text, "V", "d")
    v.replace_lines(v.buf, l1, l2 + 1, [])
    v.go_line(min(l1, len(v.lines) - 1), sol=True)
    n = l2 - l1 + 1
    if n > v.opt("report"):
        v.message(f"{n} fewer lines")


def _reg_count(arg):
    m = re.match(r"\s*([a-zA-Z0-9\"*+_-])?\s*(\d*)\s*$", arg or "")
    if not m:
        return None, None
    reg = m.group(1)
    if reg and reg.isdigit() and not m.group(2):
        return None, int(reg)
    return reg, int(m.group(2)) if m.group(2) else None


def c_yank(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    reg, count = _reg_count(x["arg"])
    if count:
        l1 = l2
        l2 = min(len(v.lines) - 1, l1 + count - 1)
    v.reg_set(reg, "\n".join(v.lines[l1:l2 + 1]) + "\n", "V", "y")
    n = l2 - l1 + 1
    if n > v.opt("report"):
        v.message(f"{n} lines yanked")


def c_put(v, x):
    l = x["l2"] if x["naddr"] else v.win.cursor[0]
    reg = (x["arg"].strip() or None)
    if reg == "=":
        raise VimError("E15: Invalid expression")
    text, kind = v.reg_get(reg)
    if text == "":
        v.set_cursor(max(0, l))
        raise VimError(f"E353: Nothing in register {reg or '\"'}")
    body = text[:-1] if text.endswith("\n") else text
    new = body.split("\n")
    v.check_modifiable()
    v.mark_change_start()
    at = l if x["bang"] else l + 1
    v.replace_lines(v.buf, at, at, new)
    v.set_cursor(at + len(new) - 1)


def c_move(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    dest = _address(v, x["arg"])
    if l1 <= dest < l2:
        raise VimError("E134: Cannot move a range of lines into itself")
    v.check_modifiable()
    v.mark_change_start()
    b = v.buf
    chunk = b.lines[l1:l2 + 1]
    if dest >= l2:
        v.replace_lines(b, dest + 1, dest + 1, chunk)
        v.replace_lines(b, l1, l2 + 1, [])
        last = dest
    else:
        v.replace_lines(b, l1, l2 + 1, [])
        v.replace_lines(b, dest + 1, dest + 1, chunk)
        last = dest + len(chunk)
    # moving lines that a :g marked: they keep their marks
    if getattr(v, "g_lines", None) is not None:
        pass
    v.set_cursor(last)
    n = l2 - l1 + 1
    if n > v.opt("report"):
        v.message(f"{n} lines moved")


def c_copy(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    dest = _address(v, x["arg"])
    v.check_modifiable()
    v.mark_change_start()
    chunk = v.lines[l1:l2 + 1]
    v.replace_lines(v.buf, dest + 1, dest + 1, chunk)
    v.set_cursor(dest + len(chunk))


def _address(v, s):
    s = s.strip()
    if not s:
        raise VimError("E14: Invalid address")
    l1, l2, n, rest = parse_range(v, s)
    if n == 0:
        if s == "0":
            return -1
        raise VimError("E14: Invalid address")
    return l2


def c_join(v, x):
    from .commands import join_lines
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] > 1 else l1 + 1
    m = re.match(r"\s*(\d+)", x["arg"] or "")
    if m:
        l1 = l2 if x["naddr"] else l1
        l2 = l1 + int(m.group(1)) - 1
    l2 = min(l2, len(v.lines) - 1)
    if l2 <= l1:
        return
    v.mark_change_start()
    join_lines(v, l1, l2, not x["bang"])
    v.set_cursor(l1)


def c_shift(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    name = x["name"]
    amount = 1 + len(re.match(r"[<>]*", x["arg"]).group(0))
    v.mark_change_start()
    v.shift_lines(l1, l2, 1 if name == ">" else -1, amount)
    v.set_cursor(l2)


def c_sort(v, x):
    arg = x["arg"]
    l1 = x["l1"] if x["naddr"] else 0
    l2 = x["l2"] if x["naddr"] else len(v.lines) - 1
    pat = None
    m = re.search(r"/((?:\\.|[^/])*)/", arg)
    if m:
        pat = m.group(1) or v.last_pat
        arg = arg[:m.start()] + arg[m.end():]
    flags = arg.replace(" ", "")
    lines = v.lines[l1:l2 + 1]
    from .search import compiled
    cre = compiled(v, pat) if pat else None

    def key_text(line):
        if cre:
            mm = cre.rx.search(line)
            if mm:
                return mm.group(0) if "r" in flags else line[mm.end():]
            return "" if "r" in flags else line
        return line

    def num(line):
        t = key_text(line)
        if "x" in flags:
            mm = re.search(r"(0[xX])?[0-9a-fA-F]+", t)
            return (1, int(mm.group(0), 16)) if mm else (0, 0)
        if "f" in flags:
            mm = re.search(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", t)
            return (1, float(mm.group(0))) if mm else (0, 0)
        mm = re.search(r"-?\d+", t)
        return (1, int(mm.group(0))) if mm else (0, 0)
    if "n" in flags or "x" in flags or "f" in flags:
        keyf = num
    elif "i" in flags:
        def keyf(line):
            return key_text(line).lower()
    else:
        keyf = key_text
    out = sorted(lines, key=keyf, reverse=x["bang"])
    if x["bang"]:
        # a reverse sort is still stable: equal keys keep their order
        out = _stable_reverse(lines, keyf)
    if "u" in flags:
        uniq = []
        for line in out:
            if uniq and keyf(uniq[-1]) == keyf(line):
                continue
            uniq.append(line)
        out = uniq
    v.mark_change_start()
    v.replace_lines(v.buf, l1, l2 + 1, out)
    v.set_cursor(l1, 0)
    diff = len(lines) - len(out)
    if diff > v.opt("report"):
        v.message(f"{diff} fewer lines")


def _stable_reverse(lines, keyf):
    idx = sorted(range(len(lines)), key=lambda i: (keyf(lines[i]), -i), reverse=True)
    return [lines[i] for i in idx]


def c_noh(v, x):
    v.no_hl = True


def c_set(v, x, scope=None):
    arg = x["arg"].strip()
    if not arg:
        return
    if x["name"] == "setlocal":
        scope = "local"
    elif x["name"] == "setglobal":
        scope = "global"
    msgs = []
    for item in re.findall(r"(?:\\.|[^\s])+", arg):
        item = item.replace("\\ ", " ")
        m = re.match(r"(no|inv)?([a-z]+)(\?|!|&|<)?(?:([+\-^]?=|:)(.*))?$", item)
        if not m:
            raise VimError(f"E518: Unknown option: {item}")
        prefix, name, suffix, opr, val = m.groups()
        full = ABBREV.get(name, name)
        if full == "all":
            continue
        if full not in OPTIONS:
            if prefix and ABBREV.get(prefix + name, prefix + name) in OPTIONS:
                full = ABBREV.get(prefix + name, prefix + name)
                prefix = None
            else:
                raise VimError(f"E518: Unknown option: {item}")
        abbr, sc, default = OPTIONS[full]
        v.used[":set " + full] += 1
        cur = v.opt(full)
        isbool = isinstance(default, bool)
        if suffix == "?" or (not prefix and not opr and not suffix and not isbool):
            msgs.append(_show_opt(full, cur, isbool))
            continue
        if suffix == "&":
            new = default
        elif isbool:
            if opr:
                raise VimError(f"E474: Invalid argument: {item}")
            if prefix == "no":
                new = False
            elif prefix == "inv" or suffix == "!":
                new = not cur
            else:
                new = True
        else:
            if prefix:
                raise VimError(f"E474: Invalid argument: {item}")
            if isinstance(default, int):
                try:
                    n = int(val)
                except (TypeError, ValueError):
                    raise VimError(f"E521: Number required after =: {item}")
                new = cur + n if opr == "+=" else cur - n if opr == "-=" else cur * n if opr == "^=" else n
            else:
                val = val or ""
                if opr == "+=":
                    new = (cur + "," + val) if cur and val else (cur or val)
                elif opr == "-=":
                    parts = [p for p in str(cur).split(",") if p and p != val]
                    new = ",".join(parts) if "," in str(cur) else str(cur).replace(val, "")
                elif opr == "^=":
                    new = (val + "," + cur) if cur else val
                else:
                    new = val
        set_option(v, full, new, scope)
    if msgs:
        v.message("  ".join(msgs))


def _show_opt(name, val, isbool):
    if isbool:
        return ("  " if val else "no") + name
    return f"  {name}={val}"


def set_option(v, name, val, scope=None):
    sc = OPTIONS[name][1]
    if name == "fileformat":
        v.buf.ff = val
        v.buf.modified = True
    if name == "readonly":
        v.buf.readonly = bool(val)
    if name == "modified":
        v.buf.modified = bool(val)
        return
    if name == "filetype":
        v.buf.opts["filetype"] = val
        if v.filetype_on:
            from .engine import FT_OPTS
            for k, x in FT_OPTS.get(val, {}).items():
                v.buf.opts[k] = x
        return
    if name in ("foldmethod", "foldlevel"):
        from .folds import reset_folds
        if sc == "w":
            v.win.opts[name] = val
        reset_folds(v)
        return
    if name == "undofile" and val and v.buf.path:
        v.undo_store[v.buf.path] = v.buf.undo
    if sc == "b":
        v.buf.opts[name] = val
        if scope != "local":
            v.o[name] = val
    elif sc == "w":
        v.win.opts[name] = val
        if scope != "local":
            v.o[name] = val
            for w in v.all_windows(all_tabs=True):
                w.opts[name] = val if w is v.win else w.opts.get(name, val)
    else:
        v.o[name] = val
    if name == "hlsearch" and val:
        v.no_hl = False
    if name in ("number", "relativenumber", "wrap", "laststatus"):
        v.layout()


def c_let(v, x):
    m = re.match(r"\s*([gbwvs]:)?(&[a-z]+|@\S|\w+)\s*([.+\-]?=)\s*(.*)$", x["arg"])
    if not m:
        raise VimError(f"E15: Invalid expression: \"{x['arg']}\"")
    scope, name, opr, expr = m.groups()
    val = eval_expr(v, expr)
    if name.startswith("&"):
        opt = ABBREV.get(name[1:], name[1:])
        if opt not in OPTIONS:
            raise VimError(f"E355: Unknown option: {name[1:]}")
        if isinstance(OPTIONS[opt][2], bool):
            val = bool(val)
        set_option(v, opt, val)
        return
    if name.startswith("@"):
        v.regs[name[1]] = (str(val), "v")
        return
    if name == "mapleader":
        v.leader = str(val)
    v.vars = getattr(v, "vars", {})
    if opr == ".=":
        val = str(v.vars.get(name, "")) + str(val)
    elif opr == "+=":
        val = v.vars.get(name, 0) + val
    v.vars[name] = val
    v.used[":let " + name] += 1


def eval_expr(v, s):
    """A small Vim script expression evaluator: numbers, 'strings', "strings", + - * / % . (concat), ( ),
    comparisons, &options, @registers, g:vars and a few functions."""
    toks = re.findall(r"\s*(\d+\.\d+|\d+|'(?:[^']|'')*'|\"(?:\\.|[^\"\\])*\"|&\w+|@.|[gbwvs]:\w+|v:\w+|\w+\(|\w+|"
                      r"==\??|!=|>=|<=|\.\.|[-+*/%().,<>!?:])", s)
    pos = [0]

    def peek():
        return toks[pos[0]].strip() if pos[0] < len(toks) else None

    def nxt():
        t = peek()
        pos[0] += 1
        return t

    def primary():
        t = nxt()
        if t is None:
            raise VimError("E15: Invalid expression")
        if t == "(":
            val = ternary()
            nxt()
            return val
        if t == "-":
            return -primary()
        if t == "!":
            return int(not primary())
        if re.fullmatch(r"\d+", t):
            return int(t)
        if re.fullmatch(r"\d+\.\d+", t):
            return float(t)
        if t.startswith("'"):
            return t[1:-1].replace("''", "'")
        if t.startswith('"'):
            body = t[1:-1]
            return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "e": "\x1b", "r": "\r"}.get(m.group(1), m.group(1)), body)
        if t.startswith("&"):
            return v.opt(ABBREV.get(t[1:], t[1:]))
        if t.startswith("@"):
            return v.reg_get(t[1])[0]
        if t.endswith("("):
            name = t[:-1]
            args = []
            if peek() != ")":
                args.append(ternary())
                while peek() == ",":
                    nxt()
                    args.append(ternary())
            nxt()
            return call_fn(v, name, args)
        if t.startswith("v:"):
            return {"v:version": 901, "v:true": 1, "v:false": 0, "v:count": 0}.get(t, 0)
        name = t.split(":", 1)[-1]
        vars_ = getattr(v, "vars", {})
        if name in vars_:
            return vars_[name]
        if name == "mapleader":
            return v.leader
        raise VimError(f"E121: Undefined variable: {t}")

    def term():
        val = primary()
        while peek() in ("*", "/", "%"):
            op = nxt()
            r = primary()
            if op == "*":
                val = val * r
            elif op == "/":
                val = int(val / r) if isinstance(val, int) and isinstance(r, int) else val / r
            else:
                val = val % r
        return val

    def arith():
        val = term()
        while peek() in ("+", "-", ".", ".."):
            op = nxt()
            r = term()
            if op in (".", ".."):
                val = f"{val}{r}"
            elif op == "+":
                val = _num(val) + _num(r)
            else:
                val = _num(val) - _num(r)
        return val

    def comparison():
        val = arith()
        while peek() in ("==", "!=", ">", "<", ">=", "<=", "==?"):
            op = nxt()
            r = arith()
            val = int({"==": val == r, "==?": str(val).lower() == str(r).lower(), "!=": val != r, ">": val > r,
                       "<": val < r, ">=": val >= r, "<=": val <= r}[op])
        return val

    def ternary():
        val = comparison()
        if peek() == "?":
            nxt()
            a = ternary()
            nxt()
            b = ternary()
            return a if val else b
        return val

    if not toks:
        raise VimError("E15: Invalid expression")
    return ternary()


def _num(x):
    if isinstance(x, (int, float)):
        return x
    m = re.match(r"\s*-?\d+", str(x))
    return int(m.group(0)) if m else 0


def call_fn(v, name, args):
    w, b = v.win, v.buf
    if name == "has":
        feat = str(args[0])
        return int(feat in ("clipboard", "unnamedplus", "syntax", "folding", "terminal", "python3", "linux", "unix",
                            "autocmd", "eval", "persistent_undo", "spell", "mouse", "vim9script", "patch-9.0.0"))
    if name == "line":
        a = args[0]
        if a == ".":
            return w.cursor[0] + 1
        if a == "$":
            return len(b.lines)
        if a.startswith("'"):
            return v.get_mark(a[1])[0] + 1
        return 0
    if name == "col":
        a = args[0]
        if a == ".":
            return w.cursor[1] + 1
        if a == "$":
            return len(b.lines[w.cursor[0]]) + 1
        return 0
    if name == "expand":
        return expand_fname(v, str(args[0]))
    if name in ("toupper", "tolower"):
        return str(args[0]).upper() if name == "toupper" else str(args[0]).lower()
    if name == "len" or name == "strlen":
        return len(str(args[0]))
    if name == "repeat":
        return str(args[0]) * int(args[1])
    if name == "submatch":
        m = getattr(v, "cur_match", None)
        if m is None:
            return ""
        try:
            return m.group(int(args[0])) or ""
        except IndexError:
            return ""
    if name == "getline":
        a = args[0]
        ln = w.cursor[0] if a == "." else (len(b.lines) - 1 if a == "$" else int(a) - 1)
        return b.lines[ln] if 0 <= ln < len(b.lines) else ""
    if name == "strftime":
        import time
        return time.strftime(str(args[0]), time.gmtime(1792483200 + v.now // 1000))
    if name == "abs":
        return abs(args[0])
    if name == "str2nr":
        return _num(args[0])
    if name == "string":
        return str(args[0])
    if name == "trim":
        return str(args[0]).strip()
    if name == "printf":
        fmt = str(args[0])
        try:
            return fmt % tuple(args[1:])
        except (TypeError, ValueError):
            return fmt
    if name == "exists":
        return int(str(args[0]).split(":", 1)[-1] in getattr(v, "vars", {}))
    if name == "getreg":
        return v.reg_get(str(args[0]) if args else None)[0]
    if name == "setreg":
        v.regs[str(args[0])] = (str(args[1]), "v")
        return 0
    if name == "cursor":
        v.set_cursor(int(args[0]) - 1, int(args[1]) - 1 if len(args) > 1 else 0)
        return 0
    if name == "nr2char":
        return chr(int(args[0]))
    if name == "char2nr":
        return ord(str(args[0])[:1] or "\0")
    raise VimError(f"E117: Unknown function: {name}")


def c_echo(v, x):
    if not x["arg"].strip():
        v.message("")
        return
    parts = []
    for piece in _split_args(x["arg"]):
        parts.append(str(eval_expr(v, piece)))
    v.message(" ".join(parts))


def _split_args(s):
    out, depth, cur, q = [], 0, [], None
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        if ch == "(":
            depth += 1
        if ch == ")":
            depth -= 1
        if ch == " " and depth == 0 and "".join(cur).strip():
            nxt_is_op = False
            if not nxt_is_op:
                out.append("".join(cur))
                cur = []
                continue
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    # join pieces split around operators
    merged = []
    for p in out:
        if merged and (p.strip()[:1] in "+-*/." or merged[-1].strip()[-1:] in "+-*/.("):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def c_call(v, x):
    m = re.match(r"\s*(\w+)\((.*)\)\s*$", x["arg"])
    if not m:
        raise VimError("E129: Function name required")
    eval_expr(v, x["arg"])


def c_execute(v, x):
    val = eval_expr(v, x["arg"]) if x["arg"].strip() else ""
    text = str(val)
    if text.startswith("normal") or text.startswith("norm"):
        mm = re.match(r"norm(?:al)?(!?)\s(.*)", text, re.S)
        if mm:
            toks = [K.CHAR_KEY.get(ch, ch) for ch in mm.group(2)]
            v.run_keys(toks, remap=not mm.group(1))
            return
    run_ex(v, text, typed=False)


# ---------------------------------------------------------------- maps, autocommands, scripts

MAP_CMDS = {"map": ("nvo", False), "nmap": ("n", False), "vmap": ("v", False), "xmap": ("v", False),
            "omap": ("o", False), "imap": ("i", False), "cmap": ("c", False), "noremap": ("nvo", True),
            "nnoremap": ("n", True), "vnoremap": ("v", True), "xnoremap": ("v", True), "onoremap": ("o", True),
            "inoremap": ("i", True), "cnoremap": ("c", True), "smap": ("", False)}


def c_map(v, x):
    modes, noremap = MAP_CMDS[x["name"]]
    if x["bang"] and x["name"] in ("map", "noremap"):
        modes = "ic"
    arg = x["arg"]
    while True:
        m = re.match(r"\s*(<(?:silent|buffer|nowait|expr|unique|special|script)>)\s*", arg, re.I)
        if not m:
            break
        arg = arg[m.end():]
    parts = arg.split(None, 1)
    if not parts:
        lines = []
        for mode in modes:
            for lhs, mp in v.maps[mode].items():
                lines.append(f"{mode}  {K.to_notation(lhs):<12}{'*' if mp['noremap'] else ' '} {mp['rhs']}")
        v.more = lines or ["No mapping found"]
        return
    lhs_text = parts[0]
    lhs = tuple(K.parse(lhs_text, v.leader))
    if len(parts) == 1:
        lines = [f"{mode}  {lhs_text}  {v.maps[mode][lhs]['rhs']}" for mode in modes if lhs in v.maps[mode]]
        v.message(lines[0] if lines else "No mapping found")
        return
    rhs = parts[1]
    for mode in modes:
        v.maps[mode][lhs] = {"rhs": rhs, "noremap": noremap}
    v.used[":" + x["name"]] += 1
    v.used[":map"] += 1


def c_unmap(v, x):
    modes = {"unmap": "nvo", "nunmap": "n", "vunmap": "v", "xunmap": "v", "ounmap": "o", "iunmap": "i",
             "cunmap": "c"}.get(x["name"], "nvo")
    lhs = tuple(K.parse(x["arg"].strip(), v.leader))
    found = False
    for mode in modes:
        if lhs in v.maps[mode]:
            del v.maps[mode][lhs]
            found = True
    if not found:
        raise VimError("E31: No such mapping")


def c_mapclear(v, x):
    for mode in "nvo":
        v.maps[mode].clear()


def c_autocmd(v, x):
    arg = x["arg"]
    if x["bang"]:
        grp = getattr(v, "cur_group", None)
        v.autocmds = [a for a in v.autocmds if a[3] != grp] if grp else []
        if not arg.strip():
            return
    m = re.match(r"\s*(?:(\w+)\s+)?([A-Za-z,]+)\s+(\S+)\s+(.*)$", arg)
    if not m:
        return
    group, events, pat, cmd = m.groups()
    if group and not re.fullmatch(r"[A-Za-z,]+", events or ""):
        pass
    known = {"BufWritePre", "BufWritePost", "BufRead", "BufReadPost", "BufNewFile", "BufEnter", "BufLeave",
             "FileType", "InsertLeave", "InsertEnter", "VimEnter", "BufWrite", "TextChanged", "BufWinEnter"}
    evs = events.split(",")
    if group and group not in known and all(e in known for e in evs):
        pass
    elif group and group in known:
        evs = [group] + evs if group not in evs else evs
    for ev in evs:
        if ev == "BufWrite":
            ev = "BufWritePre"
        if ev == "BufRead":
            ev = "BufReadPost"
        v.autocmds.append((ev, pat, cmd.strip(), getattr(v, "cur_group", None)))
    v.used[":autocmd"] += 1


def c_augroup(v, x):
    name = x["arg"].strip()
    v.cur_group = None if name.upper() == "END" else name


def fire_autocmds(v, event, buf, arg=None):
    for ev, pat, cmd, _ in list(v.autocmds):
        if ev != event:
            continue
        if event == "FileType":
            if not fnmatch.fnmatch(arg or "", pat) and pat != "*":
                continue
        else:
            name = (buf.name or "").rsplit("/", 1)[-1]
            if not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(buf.path or "", p) for p in pat.split(",")):
                continue
        keep = v.buf
        try:
            if keep is buf:
                run_ex(v, cmd, typed=False)
        except VimError as e:
            if e.msg:
                v.message(e.msg, "err")


def source_lines(v, lines, path=""):
    buf = []
    for raw in lines:
        s = raw.strip()
        if s.startswith("\\") and buf:
            buf[-1] += s[1:]
            continue
        buf.append(raw)
    skip = 0
    for line in buf:
        s = line.strip()
        if not s or s.startswith('"'):
            continue
        # if / endif: assume true (has(...) checks), but honour "if 0"
        if re.match(r"if\b", s):
            cond = s[2:].strip()
            try:
                ok = bool(eval_expr(v, cond))
            except VimError:
                ok = True
            if skip or not ok:
                skip += 1
            continue
        if re.match(r"endif\b|en\b", s):
            if skip:
                skip -= 1
            continue
        if re.match(r"else\b", s):
            if skip == 1:
                skip = 0
            elif skip == 0:
                skip = 1
            continue
        if skip:
            continue
        # trailing " comments after set/map lines
        if re.match(r"(se|set|setl|setlocal)\b", s):
            s = re.sub(r'\s+".*$', "", s)
        try:
            run_ex(v, s, typed=False)
        except VimError as e:
            if e.msg:
                v.message(f"Error detected while processing {path}:\n{e.msg}", "err")


def c_source(v, x):
    name = expand_fname(v, x["arg"]) if x["arg"] else v.buf.name
    path = v.abspath(name)
    if path == v.buf.path:
        lines = list(v.buf.lines)
    else:
        text = v.io.read(path)
        if text is None:
            raise VimError(f"E484: Can't open file {name}")
        lines = text.split("\n")
    source_lines(v, lines, name)
    v.used[":source"] += 1


def c_syntax(v, x):
    v.o["syntax"] = "off" if x["arg"].strip() in ("off", "clear") else "on"


def c_filetype(v, x):
    if "on" in x["arg"]:
        v.filetype_on = True
        for b in v.buffers.values():
            b.extra["ft_done"] = False
            v.apply_filetype(b)


def c_colorscheme(v, x):
    if not x["arg"]:
        v.message("default")


# ---------------------------------------------------------------- registers, marks, lists

def c_registers(v, x):
    want = x["arg"].replace(" ", "")
    lines = ["Type Name Content"]
    order = ['"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + list("abcdefghijklmnopqrstuvwxyz") + \
        ["-", ".", ":", "%", "#", "/", "+"]
    for name in order:
        if want and name not in want:
            continue
        if name == '"':
            src = v.regs.get("__unnamed")
            if src is None or src not in v.regs:
                continue
            text, kind = v.regs[src]
        elif name == "%":
            text, kind = v.buf.name, "v"
            if not text:
                continue
        elif name == "#":
            alt = v.buffers.get(v.win.alt)
            if not alt:
                continue
            text, kind = alt.name, "v"
        else:
            if name not in v.regs:
                continue
            text, kind = v.regs[name]
        if not text and name not in ("+",):
            continue
        t = {"v": "c", "V": "l", "b": "b"}.get(kind, "c")
        shown = K.shown(text)
        lines.append(f"  {t}  \"{name}   {shown}"[:v.width - 1])
    v.more = lines
    v.used[":reg"] += 1


def c_marks(v, x):
    b = v.buf
    lines = ["mark line  col file/text"]
    items = []
    if v.win.pcmark is not None:
        items.append(("'", v.win.pcmark))
    for m in sorted(b.marks):
        if m in "abcdefghijklmnopqrstuvwxyz\"[]^.<>":
            items.append((m, b.marks[m]))
    for k, val in sorted(v.regs.items()):
        if k.startswith("__mark_"):
            bufnum, pos = val
            items.append((k[-1], pos, bufnum))
    for item in items:
        m, pos = item[0], item[1]
        if x["arg"] and m not in x["arg"]:
            continue
        if len(item) > 2 and item[2] != b.num:
            tb = v.buffers.get(item[2])
            text = tb.name if tb else ""
        else:
            text = b.lines[pos[0]].strip() if pos[0] < len(b.lines) else ""
        lines.append(f" {m} {pos[0] + 1:6} {pos[1]:4} {text}")
    v.more = lines
    v.used[":marks"] += 1


def c_delmarks(v, x):
    b = v.buf
    if x["bang"]:
        for m in list(b.marks):
            if m.islower():
                del b.marks[m]
        return
    arg = x["arg"].replace(" ", "")
    names = set()
    for a, z in re.findall(r"(\w)-(\w)", arg):
        names |= {chr(c) for c in range(ord(a), ord(z) + 1)}
    names |= set(re.sub(r"\w-\w", "", arg))
    for m in names:
        b.marks.pop(m, None)
        v.regs.pop("__mark_" + m, None)


def c_k(v, x):
    from .commands import set_mark
    l = x["l2"] if x["naddr"] else v.win.cursor[0]
    keep = v.win.cursor
    v.win.cursor = [l, 0]
    set_mark(v, x["arg"].strip()[:1])
    v.win.cursor = keep


def c_ls(v, x):
    lines = []
    cur = v.buf
    alt = v.win.alt
    for b in v.buffers.values():
        if not b.listed and not x["bang"]:
            continue
        shown = any(w.buf is b for w in v.all_windows())
        flags = ("%" if b is cur else "#" if b.num == alt else " ") + ("a" if shown else "h" if b.loaded else " ")
        flags += ("-" if b.opts.get("modifiable", True) is False else "=" if b.readonly else " ")
        flags += "+" if b.modified else " "
        name = f'"{b.name or "[No Name]"}"'
        line = b.last_cursor[0] + 1 if b is not cur else v.win.cursor[0] + 1
        if not b.loaded:
            line = 0
        lines.append(f"{b.num:3d} {flags} {name:<30} line {line}")
    v.more = lines
    v.used[":ls"] += 1


def c_buffer(v, x):
    arg = x["arg"].strip()
    target = None
    if x["naddr"] and not arg:
        target = v.buffers.get(x["l2"] + 1)
        if target is None:
            raise VimError(f"E86: Buffer {x['l2'] + 1} does not exist")
    elif arg.isdigit():
        target = v.buffers.get(int(arg))
        if target is None:
            raise VimError(f"E86: Buffer {arg} does not exist")
    elif arg in ("#",):
        target = v.buffers.get(v.win.alt)
    elif arg:
        exact = [b for b in v.buffers.values() if b.name == arg or b.name.rsplit("/", 1)[-1] == arg]
        cands = exact or [b for b in v.buffers.values() if arg in b.name and b.listed]
        if not cands:
            raise VimError(f"E94: No matching buffer for {arg}")
        if len(cands) > 1:
            raise VimError(f"E93: More than one match for {arg}")
        target = cands[0]
    else:
        return
    if target is v.buf:
        return
    if v.buf.modified and not x["bang"] and not v.opt("hidden") and not buf_shown_elsewhere(v, v.buf):
        raise VimError("E37: No write since last change (add ! to override)")
    target.listed = True
    v.edit_buffer(target)
    v.file_msg(target)


def c_bnext(v, x, d=1):
    listed = [b for b in v.buffers.values() if b.listed]
    if not listed:
        return
    n = int(x["arg"]) if x["arg"].strip().isdigit() else 1
    cur = v.buf
    nums = [b.num for b in listed]
    if cur.num in nums:
        i = nums.index(cur.num)
    else:
        i = 0
    i = (i + d * n) % len(listed)
    target = listed[i]
    if target is cur:
        return
    if cur.modified and not x["bang"] and not v.opt("hidden") and not buf_shown_elsewhere(v, cur):
        raise VimError("E37: No write since last change (add ! to override)")
    v.edit_buffer(target)
    v.file_msg(target)


def c_bfirst(v, x, last=False):
    listed = [b for b in v.buffers.values() if b.listed]
    target = listed[-1] if last else listed[0]
    if target is not v.buf:
        if v.buf.modified and not x["bang"] and not v.opt("hidden"):
            raise VimError("E37: No write since last change (add ! to override)")
        v.edit_buffer(target)


def c_bdelete(v, x):
    arg = x["arg"].strip()
    if arg.isdigit():
        targets = [v.buffers.get(int(arg))]
    elif x["naddr"]:
        targets = [v.buffers.get(n + 1) for n in range(x["l1"], x["l2"] + 1)]
    elif arg:
        targets = [b for b in v.buffers.values() if b.name == arg or arg in b.name][:1]
    else:
        targets = [v.buf]
    for b in targets:
        if b is None:
            raise VimError(f"E516: No buffers were deleted: bd {arg}")
        if b.modified and not x["bang"]:
            raise VimError(f"E89: No write since last change for buffer {b.num} (add ! to override)")
        v.delete_buffer(b, wipe=x["name"] == "bwipeout")


def c_split(v, x, vertical=False, new=False):
    arg = x["arg"].strip()
    size = x["l2"] + 1 if x["naddr"] else None
    if x["name"] in ("vsplit", "vnew"):
        vertical = True
    if x["name"] in ("new", "vnew"):
        new = True
    if new:
        b = v.new_buffer()
        v.split(vertical=vertical, buf=b, size=size)
        if arg:
            edit_file(v, arg)
        return
    v.split(vertical=vertical, size=size)
    if arg:
        edit_file(v, arg)


def c_vertical(v, x):
    rest = x["arg"]
    m = re.match(r"(\w+)!?\s*(.*)", rest)
    if not m:
        raise VimError("E471: Argument required")
    name = full_name(m.group(1))
    v.used[f":vertical {name}"] += 1
    if name == "resize":
        return c_resize(v, {**x, "arg": m.group(2), "name": "resize"}, vertical=True)
    if name in ("split", "new", "sview"):
        return c_split(v, {**x, "arg": m.group(2), "name": "vsplit" if name != "new" else "vnew"}, vertical=True)
    if name in ("help",):
        return c_help(v, {**x, "arg": m.group(2)}, vertical=True)
    if name in ("terminal",):
        from .term import open_terminal
        return open_terminal(v, m.group(2), vertical=True)
    return run_one(v, rest)


def c_resize(v, x, vertical=False):
    arg = x["arg"].strip()
    w = v.win
    cur = w.width if vertical else w.height
    if not arg and x["naddr"]:
        arg = str(x["l2"] + 1)
    if not arg:
        n = 1000
    elif arg[0] in "+-":
        n = cur + int(arg)
    else:
        n = int(arg)
    v.resize_win(w, n, vertical)


def c_close(v, x):
    if len(v.all_windows()) == 1 and len(v.tabs) == 1:
        raise VimError("E444: Cannot close last window")
    v.close_window(v.win, force=x["bang"])


def c_only(v, x):
    v.only_window(force=x["bang"])


def c_tabnew(v, x):
    arg = x["arg"].strip()
    b = v.load_buffer(expand_fname(v, arg)) if arg else v.new_buffer()
    v.new_tab(b)
    if arg:
        v.file_msg(b)


def c_tabclose(v, x):
    if len(v.tabs) == 1:
        raise VimError("E784: Cannot close last tab page")
    idx = v.tabidx
    if x["arg"].strip().isdigit():
        idx = int(x["arg"]) - 1
    v.close_tab(idx, force=x["bang"])


def c_tabonly(v, x):
    keep = v.tabs[v.tabidx]
    v.tabs = [keep]
    v.tabidx = 0
    v.layout()


def c_tabnext(v, x, d=1):
    arg = x["arg"].strip()
    if x["name"] == "tabnext" and arg.isdigit():
        return v.goto_tab(int(arg) - 1)
    n = int(arg) if arg.isdigit() else 1
    v.goto_tab((v.tabidx + d * n) % len(v.tabs))


def c_tabs(v, x):
    lines = []
    for i, t in enumerate(v.tabs):
        lines.append(f"Tab page {i + 1}")
        for w in v.windows_of(t):
            mark = ">" if (t is v.tab and w is v.win) else " "
            lines.append(f"{mark}   {w.buf.name or '[No Name]'}")
    v.more = lines


def c_explore(v, x, how=None):
    from .netrw import explore
    explore(v, x["arg"].strip(), {"Explore": None, "Sexplore": "sp", "Vexplore": "vsp", "Lexplore": "vsp",
                                  "Texplore": "tab", "Hexplore": "sp"}.get(x["name"]))


def c_terminal(v, x):
    from .term import open_terminal
    open_terminal(v, x["arg"].strip())


def c_shell(v, x):
    if v.noquit:
        v.message("There's no shell to go back to in this exercise: use :terminal (or :!cmd) instead.", "err")
        return
    raise VimError("E1236: :shell isn't available in this terminal; use :terminal")


# ---------------------------------------------------------------- quickfix

def c_vimgrep(v, x):
    arg = x["arg"].strip()
    if not arg:
        raise VimError("E683: File name missing or invalid pattern")
    d = arg[0]
    if d.isalnum() or d in " \t":
        parts = arg.split(None, 1)
        pat, rest, flags = parts[0], parts[1] if len(parts) > 1 else "", ""
    else:
        j = 1
        while j < len(arg):
            if arg[j] == "\\":
                j += 2
                continue
            if arg[j] == d:
                break
            j += 1
        pat = arg[1:j]
        m = re.match(r"([gj]*)\s*(.*)", arg[j + 1:])
        flags, rest = m.group(1), m.group(2)
    if not pat:
        pat = v.last_pat or ""
    v.last_pat = pat
    from .search import compiled
    cre = compiled(v, pat)
    files = []
    for g in rest.split():
        g = expand_fname(v, g)
        for f in v.io.glob(v.cwd, g):
            if f not in files:
                files.append(f)
    if not files:
        raise VimError(f"E480: No match: {rest}")
    qf = []
    for f in files:
        path = v.abspath(f)
        b = v.find_buffer(path)
        lines = b.lines if b is not None else (v.io.read(path) or "").split("\n")
        if b is None and lines and lines[-1] == "":
            lines = lines[:-1]
        for i, line in enumerate(lines):
            for m in cre.rx.finditer(line):
                qf.append({"file": v.relname(path), "path": path, "line": i, "col": cre.span(m)[0], "text": line})
                if "g" not in flags:
                    break
    if not qf:
        raise VimError(f"E480: No match: {pat}")
    v.qf = qf
    v.qf_idx = 0
    v.used[":vimgrep"] += 1
    if "j" not in flags:
        qf_jump(v, 0)
    v.refresh_qf_window()


def qf_jump(v, idx):
    if not v.qf:
        raise VimError("E42: No Errors")
    if idx < 0 or idx >= len(v.qf):
        raise VimError("E553: No more items")
    v.qf_idx = idx
    e = v.qf[idx]
    # leave the quickfix window for a normal one
    if v.buf.kind == "quickfix":
        others = [w for w in v.all_windows() if w.buf.kind != "quickfix"]
        if others:
            v.goto_win(others[0])
    if v.buf.path != e["path"]:
        tb = v.find_buffer(e["path"])
        if v.buf.modified and not v.opt("hidden") and not buf_shown_elsewhere(v, v.buf):
            raise VimError("E37: No write since last change (add ! to override)")
        if tb is None:
            tb = v.load_buffer(e["path"])
        v.edit_buffer(tb)
    v.setpcmark()
    v.set_cursor(e["line"], e["col"])
    v.message(f"({idx + 1} of {len(v.qf)}): {e['text'].strip()}")
    v.refresh_qf_window()


def c_cnext(v, x, d=1):
    n = int(x["arg"]) if x["arg"].strip().isdigit() else 1
    qf_jump(v, v.qf_idx + d * n)


def c_cc(v, x):
    n = int(x["arg"]) if x["arg"].strip().isdigit() else (x["l2"] + 1 if x["naddr"] else v.qf_idx + 1)
    qf_jump(v, n - 1)


def c_cfirst(v, x, last=False):
    qf_jump(v, len(v.qf) - 1 if last else 0)


def c_copen(v, x):
    from .quickfix import open_qf
    open_qf(v, int(x["arg"]) if x["arg"].strip().isdigit() else 10)


def c_cclose(v, x):
    for w in v.all_windows():
        if w.buf.kind == "quickfix":
            v.close_window(w, force=True)


def c_cdo(v, x):
    cmd = x["arg"]
    if not v.qf:
        raise VimError("E42: No Errors")
    seen_files = set()
    entries = list(range(len(v.qf)))
    if x["name"] == "cfdo":
        first = {}
        for i, e in enumerate(v.qf):
            first.setdefault(e["path"], i)
        entries = sorted(first.values())
    for i in entries:
        qf_jump(v, i)
        run_ex(v, cmd, typed=False)
    v.used[":" + x["name"]] += 1


def c_windo(v, x):
    for w in list(v.all_windows()):
        v.goto_win(w)
        run_ex(v, x["arg"], typed=False)


def c_bufdo(v, x):
    for b in [b for b in v.buffers.values() if b.listed]:
        if b is not v.buf:
            if v.buf.modified and not v.opt("hidden"):
                raise VimError("E37: No write since last change (add ! to override)")
            v.edit_buffer(b)
        run_ex(v, x["arg"], typed=False)


# ---------------------------------------------------------------- undo in time

def c_earlier(v, x, d=-1):
    arg = x["arg"].strip() or "1"
    v.commit_undo()
    m = re.fullmatch(r"(\d+)([smhdf]?)", arg)
    if not m:
        raise VimError(f"E475: Invalid argument: {arg}")
    n, unit = int(m.group(1)), m.group(2)
    v.used[":" + x["name"]] += 1
    if unit == "":
        return v.undo_time(steps=d * n)
    if unit == "f":
        b = v.buf
        cur = b.undo.cur.seq
        target = b.saved_seq if d < 0 else cur
        return v.undo_time(steps=target - cur if d < 0 else 0)
    ms = n * {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return v.undo_time(ms=d * ms)


def c_undo(v, x):
    v.commit_undo()
    arg = x["arg"].strip()
    if arg.isdigit():
        b = v.buf
        st = b.undo.states[min(int(arg), len(b.undo.states) - 1)]
        v.undo_to(st)
        return
    v.do_undo(1)


def c_redo(v, x):
    v.commit_undo()
    v.do_redo(1)


# ---------------------------------------------------------------- folds, misc

def c_fold(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    v.create_fold(l1, l2)


def c_foldopen(v, x, close=False):
    from .folds import fold_command
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    fold_command(v, ("zC" if x["bang"] else "zc") if close else ("zO" if x["bang"] else "zo"), None, rng=(l1, l2))


def c_retab(v, x):
    b = v.buf
    m = re.match(r"\s*(\d*)", x["arg"])
    if m.group(1):
        b.opts["tabstop"] = int(m.group(1))
    ts = v.ts()
    et = v.opt("expandtab")
    for l, line in enumerate(b.lines):
        lead = re.match(r"[ \t]*", line).group(0)
        if "\t" in line and et:
            from .text import vcol
            new = []
            col = 0
            for ch in line:
                if ch == "\t":
                    n = ts - col % ts
                    new.append(" " * n)
                    col += n
                else:
                    new.append(ch)
                    col += 1
            v.set_line(b, l, "".join(new))
        elif x["bang"] and not et and lead:
            from .text import make_indent
            v.set_line(b, l, make_indent(indent_of(line, ts), False, ts) + line[len(lead):])


def c_cd(v, x):
    arg = x["arg"].strip() or v.home
    path = v.abspath(arg)
    if not v.io.isdir(path):
        raise VimError(f"E344: Can't find directory \"{arg}\" in cdpath")
    v.cwd = path
    for b in v.buffers.values():
        if b.path:
            b.name = v.relname(b.path)
    v.message(path)


def c_pwd(v, x):
    v.message(v.cwd)


def c_print(v, x):
    l1 = x["l1"] if x["naddr"] else v.win.cursor[0]
    l2 = x["l2"] if x["naddr"] else v.win.cursor[0]
    lines = v.lines[l1:l2 + 1]
    if len(lines) == 1:
        v.message(lines[0])
    else:
        v.more = lines
    v.set_cursor(l2)


def c_help(v, x, vertical=False):
    from .help import open_help
    open_help(v, x["arg"].strip(), vertical=vertical)


def c_version(v, x):
    v.more = ["VIM - Vi IMproved 9.1 (2024 Jan 02, compiled in bash-lite)",
              "Included patches: 1-16, 1752", "Huge version without GUI.  Features included (+) or not (-):",
              "+clipboard +folding +persistent_undo +spell +terminal +syntax +autocmd +eval"]


def c_jumps(v, x):
    w = v.win
    lines = [" jump line  col file/text"]
    for i, (bn, l, c) in enumerate(w.jumps):
        b = v.buffers.get(bn)
        text = (b.lines[l].strip() if b is v.buf and l < len(b.lines) else (b.name if b else "")) if b else ""
        lines.append(f"{abs(w.jumpidx - i):4} {l + 1:5} {c:4} {text}")
    lines.append(">")
    v.more = lines


def c_changes(v, x):
    v.more = ["change line  col text"]


def c_messages(v, x):
    v.more = [v.msg] if v.msg else []


def c_history(v, x):
    lines = ["      #  cmd history"]
    for i, h in enumerate(v.hist[":"]):
        lines.append(f"{i + 1:7}  {h}")
    v.more = lines


def c_silent(v, x):
    run_one(v, x["arg"])


def c_wincmd(v, x):
    k = x["arg"].strip()[:1]
    v.window_cmd(k, None)


def c_startinsert(v, x):
    v.start_insert("A" if x["bang"] else "i", {"n": None, "keys": []})


def c_spell(v, x, good=True):
    from .spell import add_word
    word = x["arg"].strip()
    add_word(v, word, good)


def c_nop(v, x):
    return


def c_z(v, x):
    return


COMMANDS = {
    "write": c_write, "update": c_update, "wq": c_wq, "xit": c_wq, "exit": c_wq, "quit": c_quit, "qall": c_qall,
    "quitall": c_qall, "wall": c_wall, "wqall": c_wall, "xall": c_wall, "cquit": c_cquit, "edit": c_edit,
    "enew": c_enew, "saveas": c_saveas, "file": c_file, "read": c_read, "!": c_bang, "substitute": c_substitute,
    "&": c_amp, "~": c_tilde, "global": c_global, "vglobal": c_global, "normal": c_normal, "delete": c_delete,
    "yank": c_yank, "put": c_put, "move": c_move, "t": c_copy, "copy": c_copy, "join": c_join, ">": c_shift,
    "<": c_shift, "sort": c_sort, "nohlsearch": c_noh, "set": c_set, "setlocal": c_set, "setglobal": c_set,
    "let": c_let, "echo": c_echo, "echomsg": c_echo, "call": c_call, "execute": c_execute,
    **{k: c_map for k in MAP_CMDS}, "unmap": c_unmap, "nunmap": c_unmap, "vunmap": c_unmap, "xunmap": c_unmap,
    "iunmap": c_unmap, "cunmap": c_unmap, "mapclear": c_mapclear, "nmapclear": c_mapclear,
    "autocmd": c_autocmd, "augroup": c_augroup, "source": c_source, "syntax": c_syntax, "filetype": c_filetype,
    "colorscheme": c_colorscheme, "registers": c_registers, "display": c_registers, "marks": c_marks,
    "delmarks": c_delmarks, "k": c_k, "mark": c_k, "ls": c_ls, "buffers": c_ls, "files": c_ls, "buffer": c_buffer,
    "bnext": c_bnext, "bprevious": lambda v, x: c_bnext(v, x, -1), "bNext": lambda v, x: c_bnext(v, x, -1),
    "bfirst": c_bfirst, "brewind": c_bfirst, "blast": lambda v, x: c_bfirst(v, x, True), "bdelete": c_bdelete,
    "bwipeout": c_bdelete, "bunload": c_bdelete, "split": c_split, "vsplit": c_split, "new": c_split,
    "vnew": c_split, "sview": c_split, "vertical": c_vertical, "resize": c_resize, "close": c_close,
    "only": c_only, "tabnew": c_tabnew, "tabedit": c_tabnew, "tabclose": c_tabclose, "tabonly": c_tabonly,
    "tabnext": c_tabnext, "tabprevious": lambda v, x: c_tabnext(v, x, -1),
    "tabNext": lambda v, x: c_tabnext(v, x, -1), "tabfirst": lambda v, x: v.goto_tab(0),
    "tabrewind": lambda v, x: v.goto_tab(0), "tablast": lambda v, x: v.goto_tab(len(v.tabs) - 1), "tabs": c_tabs,
    "Explore": c_explore, "Sexplore": c_explore, "Vexplore": c_explore, "Lexplore": c_explore,
    "Texplore": c_explore, "Hexplore": c_explore, "terminal": c_terminal, "shell": c_shell,
    "suspend": c_shell, "vimgrep": c_vimgrep, "lvimgrep": c_vimgrep, "grep": c_vimgrep, "cnext": c_cnext,
    "cprevious": lambda v, x: c_cnext(v, x, -1), "cNext": lambda v, x: c_cnext(v, x, -1), "cc": c_cc,
    "cfirst": c_cfirst, "crewind": c_cfirst, "clast": lambda v, x: c_cfirst(v, x, True), "copen": c_copen,
    "cwindow": c_copen, "cclose": c_cclose, "cdo": c_cdo, "cfdo": c_cdo, "windo": c_windo, "bufdo": c_bufdo,
    "earlier": c_earlier, "later": lambda v, x: c_earlier(v, x, 1), "undo": c_undo, "redo": c_redo,
    "fold": c_fold, "foldopen": c_foldopen, "foldclose": lambda v, x: c_foldopen(v, x, True), "retab": c_retab,
    "cd": c_cd, "lcd": c_cd, "pwd": c_pwd, "print": c_print, "help": c_help, "version": c_version, "jumps": c_jumps,
    "changes": c_changes, "messages": c_messages, "history": c_history, "silent": c_silent, "wincmd": c_wincmd,
    "startinsert": c_startinsert, "spellgood": c_spell, "spellwrong": lambda v, x: c_spell(v, x, False),
    "redraw": c_nop, "sleep": c_nop, "digraphs": c_nop, "keepjumps": c_silent, "keepmarks": c_silent,
    "noautocmd": c_silent, "leftabove": c_silent, "aboveleft": c_silent, "belowright": c_silent,
    "botright": c_silent, "topleft": c_silent, "z": c_z, "abbreviate": c_nop, "unabbreviate": c_nop,
    "command": c_nop, "unlet": c_nop, "undolist": c_changes, "tabmove": c_nop, "args": c_nop, "next": c_nop,
    "wnext": c_nop, "cabbrev": c_nop, "spellrepall": c_nop, "wviminfo": c_nop, "left": c_nop, "right": c_nop,
    "open": c_nop, "append": c_nop, "insert": c_nop, "change": c_nop, "tearoff": c_nop, "tab": c_silent,
    "find": c_edit, "sfind": c_split, "smap": c_nop, "smapclear": c_nop, "cmap": c_map, "cnoremap": c_map,
    "cexpr": c_nop, "argadd": c_nop, "argdelete": c_nop, "ex": c_nop, "bmodified": c_nop,
}

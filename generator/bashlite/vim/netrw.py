""":Explore (netrw): a directory listing; <CR> opens the entry under the cursor, - goes up."""

from __future__ import annotations

from .engine import VimError

HEADER = 8


def fill(v, b, path):
    entries = v.io.listdir(path)
    lines = ['" ============================================================================',
             '" Netrw Directory Listing                                        (netrw v173)',
             f'"   {path}',
             '"   Sorted by      name',
             '"   Sort sequence: [\\/]$,\\<core\\%(\\.\\d\\+\\)\\=\\>,\\.h$,\\.c$,\\.cpp$,\\~\\=\\*$,*,\\.o$,\\.obj$,\\.info$,\\.swp$,\\.bak$,\\~$',
             '"   Quick Help: <F1>:help  -:go up dir  D:delete  R:rename  s:sort-by  x:special',
             '" ==============================================================================',
             "../", "./"]
    dirs = sorted(n + "/" for n, isdir, ex in entries if isdir and not n.startswith("."))
    hid_dirs = sorted(n + "/" for n, isdir, ex in entries if isdir and n.startswith("."))
    files = sorted((n + ("*" if ex else "")) for n, isdir, ex in entries if not isdir)
    lines += hid_dirs + dirs + files
    b.lines[:] = lines
    b.kind = "netrw"
    b.loaded = True
    b.modified = False
    b.extra["dir"] = path
    b.opts["modifiable"] = False
    b.readonly = True
    b.undo.__init__(b.lines, v.now)


def explore(v, arg, how=None):
    base = v.buf.path.rsplit("/", 1)[0] if v.buf.path else v.cwd
    path = v.abspath(arg) if arg else (base or v.cwd)
    if not v.io.isdir(path):
        raise VimError(f"E344: Can't find directory \"{arg}\"")
    b = v.find_buffer(path)
    if b is None:
        b = v.new_buffer(name=v.relname(path) if path != v.cwd else path, kind="netrw")
        b.path = path
    fill(v, b, path)
    if how == "sp":
        v.split(False, buf=b)
    elif how == "vsp":
        v.split(True, buf=b)
    elif how == "tab":
        v.new_tab(b)
    else:
        if v.buf.modified and not v.opt("hidden"):
            raise VimError("E37: No write since last change (add ! to override)")
        v.edit_buffer(b)
    v.set_cursor(min(len(b.lines) - 1, HEADER + 1 if len(b.lines) > HEADER + 1 else HEADER), 0)
    v.used[":Explore"] += 1


def netrw_key(v, key):
    """Keys with a meaning of their own in a netrw listing. -> True if handled."""
    b = v.buf
    if key not in ("<CR>", "-", "o", "v", "t"):
        return False
    path = b.extra["dir"]
    if key == "-":
        up = path.rstrip("/").rsplit("/", 1)[0] or "/"
        fill(v, b, up)
        b.path = up
        b.name = v.relname(up)
        v.set_cursor(HEADER, 0)
        return True
    line = b.lines[v.win.cursor[0]]
    if line.startswith('"') or not line:
        return True
    name = line.rstrip("*@=|")
    if name == "./":
        return True
    if name == "../":
        return netrw_key(v, "-")
    target = path.rstrip("/") + "/" + name.rstrip("/")
    if name.endswith("/"):
        fill(v, b, target)
        b.path = target
        b.name = v.relname(target)
        v.set_cursor(HEADER + 1 if len(b.lines) > HEADER + 1 else HEADER, 0)
        return True
    from .ex import edit_file
    rel = v.relname(target)
    if key == "o":
        v.split(False, buf=v.load_buffer(rel))
    elif key == "v":
        v.split(True, buf=v.load_buffer(rel))
    elif key == "t":
        v.new_tab(v.load_buffer(rel))
    else:
        edit_file(v, rel)
    v.used["netrw_open"] += 1
    return True

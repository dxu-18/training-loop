"""nano: a text editor. The live terminal shows it as an editor over the terminal; replays apply the saved text.

The process asks the world for an edit (world.editor = {...}) and sleeps until the terminal closes the editor;
each save writes the file. In a replay the recorded saves arrive as events at their times, exactly like in the
live session, so the same commands see the same files.
"""

from __future__ import annotations

from ..exec import util
from ..vfs import FSError


@util("nano", "pico")
def u_nano(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-") or x == "-"]
    world = ctx.world
    if not getattr(world, "editor_ok", False):
        ctx.e("Too many errors from stdin\n" if not ctx.stdin_is_tty() else
              "nano: this terminal can't open the editor here\n")
        return 1
    if not a:
        ctx.e("nano: give a file name to edit, e.g. nano notes.txt\n")
        return 1
    path = ctx.abspath(a[0])
    try:
        node = ctx.fs.lookup(path)
        if node.type == "d":
            ctx.e(f"\"{a[0]}\" is a directory\n")
            return 1
        text = bytes(node.data).decode("utf-8", "replace")
        exists = True
    except FSError:
        text, exists = "", False
    req = {"path": path, "shown": a[0], "content": text, "exists": exists, "done": False, "pid": ctx.proc.pid}
    world.editor = req
    while not req["done"]:
        yield ("forever",)
    if world.editor is req:
        world.editor = None
    return 0


def save(world, req, content, name=None):
    """Write the editor's text into the file (as the user who ran nano). `name`: the file name typed at nano's
    "File Name to Write" prompt, when it was changed (nano then edits that file)."""
    host = world.local
    proc = host.procs.get(req["pid"])
    uid = proc.uid if proc else 1000
    gid = proc.gid if proc else 1000
    if name and name != req["shown"]:
        cwd = proc.cwd if proc else "/"
        req["path"] = name if name.startswith("/") else cwd.rstrip("/") + "/" + name
        req["shown"] = name
    try:
        node = host.fs.lookup(req["path"])
        if not host.fs.can(node, 2, uid, proc.gids if proc else {gid}):
            return "Permission denied"
        data = content.encode("utf-8", "surrogateescape")
        host.fs.charge(len(data) - len(node.data))
        node.data[:] = data
        node.mtime = node.ctime = world.kernel.now
    except FSError as e:
        if e.code != "ENOENT":
            return e.msg
        try:
            node = host.fs.write_file(req["path"], content, "/", 0o666 & ~(proc.umask if proc else 0o022), uid, gid)
        except FSError as e2:
            return e2.msg
    req["content"] = content
    req["exists"] = True
    return None


@util("vim", "vi", "emacs")
def u_no_editor(ctx):
    ctx.e(f"{ctx.name}: this terminal doesn't have {ctx.name} yet; edit files with nano instead, e.g. nano notes.txt\n")
    return 127
    yield

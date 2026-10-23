"""tmux: sessions, windows and panes; each pane is a shell on its own terminal that you attach to."""

from __future__ import annotations

from ..exec import util
from ..kernel import NS, SIGHUP, SIGINT, TTY, OpenFile
from ..timeutil import local_dt, strftime


class Pane:
    def __init__(self, pid, tty, proc):
        self.id, self.tty, self.proc = pid, tty, proc


class Window:
    def __init__(self, index, name):
        self.index, self.name = index, name
        self.panes: list[Pane] = []
        self.active = 0
        self.zoomed = False


class TSession:
    def __init__(self, sid, name, created):
        self.id, self.name, self.created = sid, name, created
        self.windows: list[Window] = []
        self.active = 0
        self.attached = 0
        self.detach_requested = False
        self.client = None

    def pane(self):
        w = self.windows[self.active]
        return w.panes[w.active]


class Server:
    def __init__(self, host):
        self.host = host
        self.sessions: dict[str, TSession] = {}
        self.next_id = 0
        self.next_pane = 0
        self.proc = None


def server_of(ctx, create=False):
    srv = ctx.host.extra.get("tmux")
    if srv is not None and not srv.sessions:
        srv = None
        ctx.host.extra.pop("tmux", None)
    if srv is None and create:
        srv = Server(ctx.host)
        ctx.host.extra["tmux"] = srv
        from ..world import forever
        p = ctx.k.spawn(ctx.host, list(ctx.argv), None, None, fds={}, cwd=ctx.cwd, uid=ctx.uid, gid=ctx.proc.gid)
        p.ppid = 1
        p.daemon = True
        p.argv = ["tmux"] + list(ctx.argv[1:])
        ctx.k.start(p, forever())
        srv.proc = p
    return srv


def socket(ctx):
    return f"/tmp/tmux-{ctx.uid}/default"


def new_pane(ctx, srv, sess, command=None, cwd=None):
    """Start a shell (or `command`) on a fresh pane terminal."""
    from ..shell import Shell
    from ..world import init_shell, login_env
    k = ctx.k
    tty = TTY(name=f"pts/{2 + srv.next_pane}")
    tty.input_eof = False
    tty.limit = 400_000
    fds = {0: OpenFile("tty", rd=True, tty=tty, stream="in"), 1: OpenFile("tty", wr=True, tty=tty, stream="out"),
           2: OpenFile("tty", wr=True, tty=tty, stream="err")}
    parent = srv.proc
    p = k.spawn(ctx.host, ["-bash"] if command is None else ["bash", "-c", command], None, None, fds=fds,
                cwd=cwd or ctx.cwd, uid=ctx.uid, gid=ctx.proc.gid)
    p.ppid = parent.pid if parent else 1
    p.gids = set(ctx.proc.gids)
    p.ctty = tty
    p.tty = tty.name
    p.daemon = True
    env = dict(ctx.env)
    env["TMUX"] = f"{socket(ctx)},{parent.pid if parent else 0},{sess.id}"
    env["TMUX_PANE"] = f"%{srv.next_pane}"
    env["TERM"] = "tmux-256color"
    pane = Pane(srv.next_pane, tty, p)
    srv.next_pane += 1
    if command is None:
        sh = Shell(ctx.world, p, interactive=True)
        sh.arg0 = "bash"
        init_shell(sh, env, interactive=True)

        def run(shell):
            rc = shell.get("HOME", "") + "/.bashrc"
            if shell.fs.exists(rc):
                yield from shell.eval_string(f". {rc}")
            return (yield from shell.repl(idle=False, persistent=True))
        k.start(p, run(sh))
    else:
        from ..exec import exec_program
        k.start(p, exec_program(p, "/bin/bash", ["bash", "-c", command], env, None))
    p.pgid = p.pid
    tty.fg_pgid = p.pid

    def on_exit(proc, sess=sess, pane=pane):
        for w in list(sess.windows):
            if pane in w.panes:
                w.panes.remove(pane)
                if not w.panes:
                    sess.windows.remove(w)
                    sess.active = max(0, min(sess.active, len(sess.windows) - 1))
                else:
                    w.active = max(0, min(w.active, len(w.panes) - 1))
        if not sess.windows:
            srv.sessions.pop(sess.name, None)
            if sess.client is not None:
                ctx.k.poke(sess.client)
            if not srv.sessions and srv.proc is not None:
                ctx.k.kill_now(srv.proc, 15)
                ctx.host.extra.pop("tmux", None)
    p.on_exit.append(on_exit)
    return pane


def find_session(srv, target):
    if srv is None:
        return None
    if target is None:
        return list(srv.sessions.values())[-1] if srv.sessions else None
    name = target.split(":")[0].split(".")[0]
    if name.startswith("$"):
        return next((s for s in srv.sessions.values() if f"${s.id}" == name), None)
    if name in srv.sessions:
        return srv.sessions[name]
    hits = [s for s in srv.sessions.values() if s.name.startswith(name)]
    return hits[0] if len(hits) == 1 else None


def attach(ctx, srv, sess):
    """Bridge the session's active pane to our terminal until detach / exit / end of input."""
    main = ctx.world.tty_of(ctx.proc)
    pane = sess.pane()
    tty = pane.tty
    sess.attached += 1
    sess.detach_requested = False
    sess.client = ctx.proc
    # show what is already on the pane's screen
    screen = tty.text()
    if screen:
        lines = screen.split("\n")
        tail = "\n".join(lines[-24:])
        main.write("out", tail.encode())
    tty.source, tty.mirror = main, main
    tty.on_starve = lambda: ctx.k.poke(ctx.proc)
    ctx.k.wake(tty)
    status = 0
    try:
        while True:
            if sess.detach_requested:
                main.write("err", f"[detached (from session {sess.name})]\n".encode())
                break
            if sess.name not in srv.sessions or not sess.windows:
                main.write("err", b"[exited]\n")
                break
            cur = sess.pane()
            if cur is not pane:      # switched window/pane: move the bridge
                tty.source = tty.mirror = None
                tty.on_starve = None
                pane, tty = cur, cur.tty
                tty.source, tty.mirror = main, main
                tty.on_starve = lambda: ctx.k.poke(ctx.proc)
                ctx.k.wake(tty)
            p = pane.proc
            waiting = p.state == "S" and p.waiting and p.waiting[0] == "tty"
            if waiting and not main.lines and not tty.lines and not tty.input:
                break                # nothing more typed: leave the session running
            yield ("wait", [p])
    finally:
        tty.source = tty.mirror = None
        tty.on_starve = None
        sess.attached -= 1
        sess.client = None
    return status


def session_line(s, ctx):
    created = strftime("%a %b %e %H:%M:%S %Y", local_dt(s.created, ctx.tz))
    return f"{s.name}: {len(s.windows)} windows (created {created})" + (" (attached)" if s.attached else "")


CMDS = {"new": "new-session", "new-session": "new-session", "ls": "list-sessions", "list-sessions": "list-sessions",
        "a": "attach-session", "at": "attach-session", "attach": "attach-session", "attach-session": "attach-session",
        "kill-session": "kill-session", "kill-server": "kill-server", "detach": "detach-client", "detach-client": "detach-client",
        "new-window": "new-window", "neww": "new-window", "split-window": "split-window", "splitw": "split-window",
        "send-keys": "send-keys", "send": "send-keys", "capture-pane": "capture-pane", "capturep": "capture-pane",
        "list-windows": "list-windows", "lsw": "list-windows", "select-window": "select-window", "selectw": "select-window",
        "rename-window": "rename-window", "renamew": "rename-window", "rename-session": "rename-session",
        "list-panes": "list-panes", "lsp": "list-panes", "kill-pane": "kill-pane", "kill-window": "kill-window",
        "next-window": "next-window", "previous-window": "previous-window", "has-session": "has-session",
        "has": "has-session", "select-pane": "select-pane", "resize-pane": "resize-pane", "-V": "-V"}


@util("tmux")
def u_tmux(ctx):
    a = ctx.argv[1:]
    if not a:
        a = ["new-session"]
    cmd = CMDS.get(a[0])
    if cmd is None:
        ctx.e(f"unknown command: {a[0]}\n")
        return 1
    if cmd == "-V":
        ctx.w("tmux 3.4\n")
        return 0
    args = a[1:]
    opts = {}
    rest = []
    i = 0
    takes = set("stncFxyFePp")
    while i < len(args):
        x = args[i]
        if x.startswith("-") and len(x) > 1:
            for k2, c in enumerate(x[1:]):
                if c in "stnc" and cmd != "send-keys" or (cmd == "send-keys" and c == "t"):
                    val = x[k2 + 2:] or (args[i + 1] if i + 1 < len(args) else "")
                    if not x[k2 + 2:]:
                        i += 1
                    opts[c] = val
                    break
                opts[c] = True
            i += 1
            continue
        rest.append(x)
        i += 1
    _ = takes
    inside = "TMUX" in ctx.env
    srv = server_of(ctx)
    if cmd == "new-session":
        if inside and not opts.get("d"):
            ctx.e("sessions should be nested with care, unset $TMUX to force\n")
            return 1
        srv = server_of(ctx, create=True)
        name = opts.get("s")
        if name is None:
            n = 0
            while str(n) in srv.sessions:
                n += 1
            name = str(n)
        if name in srv.sessions:
            ctx.e(f"duplicate session: {name}\n")
            return 1
        sess = TSession(srv.next_id, name, ctx.now)
        srv.next_id += 1
        srv.sessions[name] = sess
        command = " ".join(rest) if rest else None
        win = Window(0, opts.get("n") or (rest[0].split()[0].rsplit("/", 1)[-1] if rest else "bash"))
        sess.windows.append(win)
        win.panes.append(new_pane(ctx, srv, sess, command, opts.get("c")))
        if opts.get("d"):
            return 0
        yield from ctx.flush()
        return (yield from attach(ctx, srv, sess))
    if srv is None:
        if cmd in ("list-sessions", "attach-session", "kill-session", "has-session", "send-keys", "capture-pane",
                   "list-windows", "kill-server", "new-window", "split-window"):
            if cmd == "attach-session":
                ctx.e("no sessions\n")
            else:
                ctx.e(f"no server running on {socket(ctx)}\n")
            return 1
    target = opts.get("t")
    if cmd == "list-sessions":
        for s in srv.sessions.values():
            ctx.w(session_line(s, ctx) + "\n")
        return 0
    if cmd == "has-session":
        return 0 if find_session(srv, target) else 1
    if cmd == "kill-server":
        for s in list(srv.sessions.values()):
            for w in list(s.windows):
                for p in list(w.panes):
                    ctx.k.signal(p.proc, SIGHUP)
        yield ("yield",)
        return 0
    sess = find_session(srv, target) if cmd not in ("detach-client",) or target else None
    if cmd == "detach-client":
        if not inside:
            ctx.e("no current client\n")
            return 1
        sid = ctx.env["TMUX"].split(",")[-1]
        for s in srv.sessions.values():
            if str(s.id) == sid:
                s.detach_requested = True
                pane = s.pane()
                pane.tty.source = pane.tty.mirror = None
                if s.client is not None:
                    ctx.k.poke(s.client)
        return 0
    if sess is None:
        ctx.e(f"can't find session: {target}\n" if target else "no current session\n")
        return 1
    if cmd == "attach-session":
        if inside:
            ctx.e("sessions should be nested with care, unset $TMUX to force\n")
            return 1
        yield from ctx.flush()
        return (yield from attach(ctx, srv, sess))
    if cmd == "kill-session":
        for w in list(sess.windows):
            for p in list(w.panes):
                ctx.k.signal(p.proc, SIGHUP)
        srv.sessions.pop(sess.name, None)
        yield ("yield",)
        return 0
    if cmd == "rename-session":
        if rest:
            srv.sessions.pop(sess.name, None)
            sess.name = rest[0]
            srv.sessions[sess.name] = sess
        return 0
    if cmd == "new-window":
        win = Window(max((w.index for w in sess.windows), default=-1) + 1,
                     opts.get("n") or (rest[0].split()[0] if rest else "bash"))
        sess.windows.append(win)
        win.panes.append(new_pane(ctx, srv, sess, " ".join(rest) if rest else None, opts.get("c")))
        if not opts.get("d"):
            sess.active = len(sess.windows) - 1
        return 0
    if cmd == "split-window":
        w = sess.windows[sess.active]
        w.panes.append(new_pane(ctx, srv, sess, " ".join(rest) if rest else None, opts.get("c")))
        w.active = len(w.panes) - 1
        return 0
    if cmd in ("select-window", "next-window", "previous-window"):
        if cmd == "next-window":
            sess.active = (sess.active + 1) % len(sess.windows)
        elif cmd == "previous-window":
            sess.active = (sess.active - 1) % len(sess.windows)
        else:
            idx = target.split(":")[-1] if target and ":" in target else (rest[0] if rest else "0")
            for k2, w in enumerate(sess.windows):
                if str(w.index) == idx or w.name == idx:
                    sess.active = k2
        return 0
    if cmd == "rename-window":
        if rest:
            sess.windows[sess.active].name = rest[0]
        return 0
    if cmd == "list-windows":
        for k2, w in enumerate(sess.windows):
            flag = "*" if k2 == sess.active else ("-" if k2 == sess.active - 1 else "")
            ctx.w(f"{w.index}: {w.name}{flag} ({len(w.panes)} panes) [80x23]" + (" (active)" if k2 == sess.active else "") + "\n")
        return 0
    if cmd == "list-panes":
        w = sess.windows[sess.active]
        for k2, p in enumerate(w.panes):
            ctx.w(f"{k2}: [80x23] [history 0/2000, 0 bytes] %{p.id}" + (" (active)" if k2 == w.active else "") + "\n")
        return 0
    if cmd in ("kill-pane", "kill-window"):
        w = sess.windows[sess.active]
        victims = w.panes if cmd == "kill-window" else [w.panes[w.active]]
        for p in list(victims):
            ctx.k.signal(p.proc, SIGHUP)
        yield ("yield",)
        return 0
    if cmd == "send-keys":
        pane = sess.pane()
        buf = []
        for key in rest:
            if key in ("Enter", "C-m", "KPEnter"):
                pane.tty.lines.append("".join(buf))
                buf = []
            elif key == "C-c":
                fg = pane.tty.fg_pgid
                for p in list(ctx.host.procs.values()):
                    if p.pgid == fg and p.pid != pane.proc.pid:
                        ctx.k.signal(p, SIGINT)
                pane.tty.write("echo", b"^C\n")
            elif key == "Space":
                buf.append(" ")
            else:
                buf.append(key)
        if buf:
            pane.tty.input += "".join(buf).encode()
        ctx.k.wake(pane.tty)
        yield ("yield",)
        return 0
    if cmd == "capture-pane":
        pane = sess.pane()
        text = pane.tty.text()
        import re
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if opts.get("p"):
            ctx.w("".join(x + "\n" for x in lines[-23:]))
        return 0
    if cmd in ("select-pane", "resize-pane"):
        w = sess.windows[sess.active]
        if opts.get("Z") or cmd == "resize-pane" and opts.get("Z"):
            w.zoomed = not w.zoomed
        elif w.panes:
            w.active = (w.active + 1) % len(w.panes)
        return 0
    return 0

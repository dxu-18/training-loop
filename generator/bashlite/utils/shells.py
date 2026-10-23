"""bash / sh: a new (non-interactive unless asked) shell in the same process."""

from __future__ import annotations

from ..exec import util
from ..vfs import FSError

VERSION = """GNU bash, version 5.2.21(1)-release (x86_64-pc-linux-gnu)
Copyright (C) 2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>

This is free software; you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
"""

USAGE = """GNU bash, version 5.2.21(1)-release-(x86_64-pc-linux-gnu)
Usage:\tbash [GNU long option] [option] ...
\tbash [GNU long option] [option] script-file ...
GNU long options:
\t--debug
\t--help
\t--init-file
\t--login
\t--noediting
\t--noprofile
\t--norc
\t--posix
\t--rcfile
\t--restricted
\t--verbose
\t--version
Shell options:
\t-ilrsD or -c command or -O shopt_option\t\t(invocation only)
\t-abefhkmnptuvxBCEHPT or -o option
Type `bash -c "help set"' for more information about shell options.
Type `bash -c help' for more information about shell builtin commands.
Use the `bashbug' command to report bugs.
"""


@util("bash", "sh", "dash", "rbash")
def bash(ctx):
    from ..builtins import SET_LETTERS
    from ..shell import ExitShell, Shell, ShellAbort
    from ..world import init_shell
    args = ctx.argv[1:]
    cmd_mode = interactive = read_stdin = noexec = False
    on, off, shopts = set(), set(), []
    rc = True
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if a == "--version":
            ctx.w(VERSION)
            return 0
        if a == "--help":
            ctx.w(USAGE)
            return 0
        if a in ("--norc", "--noprofile"):
            rc = False
            i += 1
            continue
        if a in ("--posix", "--noediting", "--login", "--restricted", "--verbose", "--debug"):
            if a == "--verbose":
                on.add("verbose")
            i += 1
            continue
        if a in ("-o", "+o") and i + 1 < len(args):
            (on if a[0] == "-" else off).add(args[i + 1])
            i += 2
            continue
        if a in ("-O", "+O") and i + 1 < len(args):
            shopts.append((a[0] == "-", args[i + 1]))
            i += 2
            continue
        if len(a) > 1 and a[0] in "-+" and not a.startswith("--"):
            for c in a[1:]:
                if c == "c":
                    cmd_mode = True
                elif c == "i":
                    interactive = True
                elif c == "s":
                    read_stdin = True
                elif c in "lrD":
                    pass
                elif c == "n":
                    noexec = True
                elif c in SET_LETTERS:
                    (on if a[0] == "-" else off).add(SET_LETTERS[c])
                else:
                    ctx.e(f"{ctx.argv[0]}: {a[0]}{c}: invalid option\n" + USAGE)
                    return 2
            i += 1
            continue
        if a.startswith("--"):
            ctx.e(f"{ctx.argv[0]}: {a}: invalid option\n" + USAGE)
            return 2
        break
    rest = args[i:]
    proc = ctx.proc
    if not cmd_mode and not rest and not read_stdin and ctx.stdin_is_tty():
        interactive = True
    sh = Shell(ctx.world, proc, interactive=interactive, name=ctx.argv[0])
    sh.arg0 = ctx.argv[0]
    init_shell(sh, ctx.env, interactive)
    sh.opts |= on
    sh.opts -= off
    for enable, name in shopts:
        (sh.shopts.add if enable else sh.shopts.discard)(name)
    status = 0
    try:
        if cmd_mode:
            if not rest:
                ctx.e(f"{ctx.argv[0]}: -c: option requires an argument\n")
                return 2
            src = rest[0]
            sh.c_mode = True
            if len(rest) > 1:
                sh.arg0 = rest[1]
            sh.positional = list(rest[2:])
            status = yield from _run(sh, src, noexec)
        elif rest and not read_stdin:
            name = rest[0]
            try:
                data = yield from ctx.read_path(name)
            except FSError as e:
                ctx.e(f"{ctx.argv[0]}: {name}: {e.msg}\n")
                return 127 if e.code == "ENOENT" else 126
            sh.script = name
            sh.arg0 = name
            sh.positional = list(rest[1:])
            text = data.decode("utf-8", "surrogateescape")
            if "\0" in text[:4096]:
                ctx.e(f"{ctx.argv[0]}: {name}: cannot execute binary file\n")
                return 126
            status = yield from _run(sh, text, noexec)
        elif interactive:
            sh.positional = list(rest)
            if rc:
                rcfile = sh.get("HOME", "") + "/.bashrc"
                if ctx.fs.exists(rcfile):
                    yield from sh.eval_string(f". {rcfile}")
            status = yield from sh.repl(idle=False)
            if not getattr(sh, "_exited", False):
                pass
        else:
            sh.positional = list(rest)
            data = yield from ctx.read_fd(0)
            status = yield from _run(sh, data.decode("utf-8", "surrogateescape"), noexec)
    except ExitShell as e:
        status = e.status
    except ShellAbort as e:
        status = e.status
    status = yield from sh.run_exit_trap(status)
    return status


def _run(sh, src, noexec=False):
    from ..parser import ParseError, Parser
    from ..shell import ExitShell
    if noexec or "noexec" in sh.opts:
        try:
            Parser(src).parse_all()
        except ParseError as e:
            sh.lineno = e.line
            yield from sh.error(e.msg)
            return 2
        return 0
    try:
        return (yield from sh.run_script_text(src))
    except ExitShell as e:
        return e.status

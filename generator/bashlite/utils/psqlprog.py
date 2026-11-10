"""psql, createdb, dropdb: the PostgreSQL 16 client on the machine (the server is generator/postgreslite).

The machine has one PostgreSQL cluster (world.local.extra["pg"]), reached through the Unix socket
/var/run/postgresql/.s.PGSQL.5432 as the role named like the Unix user (peer authentication). Problems build it
before the session (their `sql` spec); `createdb` / `CREATE DATABASE` add databases.

In the live terminal an interactive psql registers itself (world.psql) and sleeps; the terminal sends it every
line typed at its prompt (with readline: history, Ctrl-R, arrows) and shows what it prints. Run with -c, -f, a
heredoc or a pipe, it reads its input and writes to stdout / stderr like any program.
"""

from __future__ import annotations

import posixpath

from ..exec import util
from ..vfs import FSError

SOCKET = "/var/run/postgresql/.s.PGSQL.5432"


def cluster_of(world, create=True):
    from ...postgreslite.server import Cluster
    c = world.local.extra.get("pg")
    if c is None and create:
        c = Cluster()
        world.local.extra["pg"] = c
    return c


class BashEnv:
    """psql's view of the machine: files as the user who ran it, a shell for \\!, the machine's clock."""

    def __init__(self, world, proc, env):
        from .vim import VimIO
        self.world = world
        self.io = VimIO(world, proc, env)
        self.proc = proc
        self.home = env.get("HOME", "/home/ada")
        self.cwd = proc.cwd

    @property
    def columns(self):
        return getattr(self.world.tty, "width", 80) or 80

    def _abs(self, path):
        if path.startswith("~"):
            path = self.home + path[1:]
        return posixpath.normpath(posixpath.join(self.proc.cwd, path))

    def read_file(self, path):
        p = self._abs(path)
        if not self.io.exists(p):
            raise OSError("No such file or directory")
        if self.io.isdir(p):
            raise OSError("Is a directory")
        text = self.io.read(p)
        if text is None:
            raise OSError("Permission denied")
        return text

    def write_file(self, path, text, append=False):
        p = self._abs(path)
        if append:
            old = self.io.read(p) if self.io.exists(p) else ""
            text = (old or "") + text
        err = self.io.write(p, text)
        if err:
            raise OSError(err)

    def shell(self, cmd):
        r = self.io.run(cmd, cwd=self.proc.cwd)
        return r["stdout"] + r["stderr"], r["status"]

    def chdir(self, path):
        p = self._abs(path)
        if not self.io.isdir(p):
            raise OSError("No such file or directory")
        self.proc.cwd = p

    def now_us(self):
        return self.world.kernel.now // 1000

    def spend_us(self, us):
        if us > 0:
            self.world.kernel.now += int(us) * 1000


def _usage():
    return ("psql is the PostgreSQL interactive terminal.\n\nUsage:\n  psql [OPTION]... [DBNAME [USERNAME]]\n\n"
            "General options:\n  -c, --command=COMMAND    run only single command (SQL or internal) and exit\n"
            "  -d, --dbname=DBNAME      database name to connect to (default: \"ada\")\n"
            "  -f, --file=FILENAME      execute commands from file, then exit\n"
            "  -l, --list               list available databases, then exit\n"
            "  -v, --set=, --variable=NAME=VALUE\n                           set psql variable NAME to VALUE\n"
            "                           (e.g., -v ON_ERROR_STOP=1)\n"
            "  -V, --version            output version information, then exit\n"
            "  -X, --no-psqlrc          do not read startup file (~/.psqlrc)\n"
            "  -1 (\"one\"), --single-transaction\n                           execute as a single transaction (if non-interactive)\n"
            "  -?, --help[=options]     show this help, then exit\n\nInput and output options:\n"
            "  -a, --echo-all           echo all input from script\n  -e, --echo-queries       echo commands sent to server\n"
            "  -q, --quiet              run quietly (no messages, only query output)\n"
            "  -o, --output=FILENAME    send query results to file (or |pipe)\n\nOutput format options:\n"
            "  -A, --no-align           unaligned table output mode\n      --csv                CSV (Comma-Separated Values) table output mode\n"
            "  -F, --field-separator=STRING\n                           field separator for unaligned output (default: \"|\")\n"
            "  -P, --pset=VAR[=ARG]     set printing option VAR to ARG (see \\pset command)\n"
            "  -t, --tuples-only        print rows only\n  -x, --expanded           turn on expanded table output\n\n"
            "Connection options:\n  -h, --host=HOSTNAME      database server host or socket directory (default: \"local socket\")\n"
            "  -p, --port=PORT          database server port (default: \"5432\")\n"
            "  -U, --username=USERNAME  database user name (default: \"ada\")\n\n"
            "For more information, type \"\\?\" (for internal commands) or \"\\help\" (for SQL\n"
            "commands) from within psql, or consult the psql section in the PostgreSQL\ndocumentation.\n\n"
            "Report bugs to <pgsql-bugs@lists.postgresql.org>.\nPostgreSQL home page: <https://www.postgresql.org/>\n")


def parse_args(argv, user):
    o = {"db": None, "user": None, "cmds": [], "file": None, "quiet": False, "list": False, "vars": {}, "pset": [],
         "align": True, "tuples": False, "expanded": False, "csv": False, "fieldsep": None, "output": None,
         "single": False, "echo": False, "version": False, "help": False, "host": None, "noprc": False,
         "error": None}
    a = argv[1:]
    pos = []
    i = 0

    def val(opt):
        nonlocal i
        if i + 1 >= len(a):
            o["error"] = f"psql: error: option requires an argument -- '{opt}'"
            return None
        i += 1
        return a[i]
    while i < len(a):
        x = a[i]
        if x.startswith("--"):
            name, _, v = x[2:].partition("=")
            longmap = {"dbname": "d", "username": "U", "command": "c", "file": "f", "host": "h", "port": "p",
                       "set": "v", "variable": "v", "pset": "P", "output": "o", "field-separator": "F"}
            if name in longmap:
                if not v:
                    v = val(name)
                x = "-" + longmap[name]
                a_v = v
                _apply(o, x[1], a_v, pos)
            elif name in ("quiet",):
                o["quiet"] = True
            elif name == "list":
                o["list"] = True
            elif name == "csv":
                o["csv"] = True
            elif name in ("no-align",):
                o["align"] = False
            elif name in ("tuples-only",):
                o["tuples"] = True
            elif name == "expanded":
                o["expanded"] = True
            elif name == "single-transaction":
                o["single"] = True
            elif name in ("version",):
                o["version"] = True
            elif name in ("help",):
                o["help"] = True
            elif name in ("no-psqlrc",):
                o["noprc"] = True
            elif name in ("echo-queries", "echo-all"):
                o["echo"] = True
            elif name in ("no-password", "password"):
                pass
            else:
                o["error"] = f"psql: error: unrecognized option '--{name}'"
            i += 1
            continue
        if x.startswith("-") and len(x) > 1:
            k = 1
            while k < len(x):
                f = x[k]
                if f in "dUcfhpvPoF":
                    rest = x[k + 1:]
                    v = rest if rest else val(f)
                    if v is None:
                        return o
                    _apply(o, f, v, pos)
                    break
                if f == "q":
                    o["quiet"] = True
                elif f == "l":
                    o["list"] = True
                elif f == "A":
                    o["align"] = False
                elif f == "t":
                    o["tuples"] = True
                elif f == "x":
                    o["expanded"] = True
                elif f == "X":
                    o["noprc"] = True
                elif f == "1":
                    o["single"] = True
                elif f in ("e", "a", "E"):
                    o["echo"] = True
                elif f == "V":
                    o["version"] = True
                elif f == "?":
                    o["help"] = True
                elif f in ("w", "W", "n", "s", "b", "L", "H", "S", "z", "0"):
                    pass
                else:
                    o["error"] = f"psql: error: invalid option -- '{f}'"
                    return o
                k += 1
            i += 1
            continue
        pos.append(x)
        i += 1
    if pos:
        if o["db"] is None:
            o["db"] = pos[0]
            pos = pos[1:]
        if pos and o["user"] is None:
            o["user"] = pos[0]
            pos = pos[1:]
    if pos:
        o["warn_extra"] = pos
    if o["user"] is None:
        o["user"] = user
    if o["db"] is None:
        o["db"] = o["user"]
    if o["db"].startswith("postgresql://") or o["db"].startswith("postgres://"):
        uri = o["db"].split("://", 1)[1]
        cred, _, hostpart = uri.rpartition("@")
        if cred:
            o["user"] = cred.split(":")[0]
        dbn = hostpart.split("/", 1)[1] if "/" in hostpart else o["user"]
        o["db"] = dbn.split("?")[0]
    return o


def _apply(o, f, v, pos):
    if f == "d":
        if "=" in v and not v.startswith("postgres"):
            for part in v.split():
                k, _, val = part.partition("=")
                if k == "dbname":
                    o["db"] = val
                elif k == "user":
                    o["user"] = val
        else:
            o["db"] = v
    elif f == "U":
        o["user"] = v
    elif f == "c":
        o["cmds"].append(("c", v))
    elif f == "f":
        o["cmds"].append(("f", v))
    elif f == "h":
        o["host"] = v
    elif f == "p":
        o["port"] = v
    elif f == "v":
        name, _, val = v.partition("=")
        o["vars"][name] = val
    elif f == "P":
        o["pset"].append(v)
    elif f == "o":
        o["output"] = v
    elif f == "F":
        o["fieldsep"] = v


def _conn_error(msg):
    return f'psql: error: connection to server on socket "{SOCKET}" failed: FATAL:  {msg}\n'


@util("psql")
def u_psql(ctx):
    from ...postgreslite.psql import Psql, Quit, _StopScript
    from ...postgreslite.errors import PgError
    world = ctx.world
    user = ctx.env.get("USER") or "ada"
    o = parse_args(ctx.argv, user)
    if o["error"]:
        ctx.e(o["error"] + "\npsql: hint: Try \"psql --help\" for more information.\n")
        return 1
    if o["help"]:
        ctx.w(_usage())
        return 0
    if o["version"]:
        ctx.w("psql (PostgreSQL) 16.4 (Ubuntu 16.4-0ubuntu0.24.04.2)\n")
        return 0
    remote = None
    if o.get("host") not in (None, "localhost", "127.0.0.1", "/var/run/postgresql", "::1") or \
            ctx.host.extra.get("container") is not None:
        # a server in a container (docker/programs.py): reached by name on a docker network, or its own socket
        name = o.get("host") or "localhost"
        resolver = ctx.host.extra.get("resolver")
        target = resolver(name) if resolver is not None else world.resolve(name)
        if target is None:
            ctx.e(f'psql: error: could not translate host name "{name}" to address: Name or service not known\n'
                  if o.get("host") else f'psql: error: connection to server on socket "{SOCKET}" failed: No such file '
                                        f'or directory\n\tIs the server running locally and accepting connections on '
                                        f'that socket?\n')
            return 2
        if 5432 not in target.services or target.extra.get("pg") is None:
            ctx.e(f'psql: error: connection to server at "{name}" ({target.ip}), port 5432 failed: Connection refused\n'
                  f'\tIs the server running on that host and accepting TCP/IP connections?\n' if o.get("host") else
                  f'psql: error: connection to server on socket "{SOCKET}" failed: No such file or directory\n\tIs '
                  f'the server running locally and accepting connections on that socket?\n')
            return 2
        pw = target.extra.get("pg_password")
        if o.get("host") and pw is not None and target is not ctx.host:
            given = ctx.env.get("PGPASSWORD")
            if given is None:
                ctx.e(f"Password for user {o['user']}: \n")
                given = ""
            if given != pw:
                ctx.e(f'psql: error: connection to server at "{name}" ({target.ip}), port 5432 failed: FATAL:  '
                      f'password authentication failed for user "{o["user"]}"\n')
                return 2
        remote = target.extra["pg"]
    if o.get("port") not in (None, "5432"):
        ctx.e(f'psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.{o["port"]}" failed: No '
              'such file or directory\n\tIs the server running locally and accepting connections on that socket?\n')
        return 2
    cluster = remote if remote is not None else cluster_of(world)
    if o["user"] not in cluster.roles:
        ctx.e(_conn_error(f'role "{o["user"]}" does not exist'))
        return 2
    if o["list"]:
        o["db"] = "postgres"
    if o["db"] not in cluster.databases:
        ctx.e(_conn_error(f'database "{o["db"]}" does not exist'))
        return 2
    interactive = not o["cmds"] and ctx.stdin_is_tty()
    env = BashEnv(world, ctx.proc, ctx.env)
    p = Psql(cluster, o["db"], o["user"], env, interactive=interactive, quiet=o["quiet"],
             seed=world.kernel.now % 1000003)
    for k, v in o["vars"].items():
        p.vars[k] = v
        p._var_hook(k)
    if not o["align"]:
        p.opts.format = "unaligned"
    if o["csv"]:
        p.opts.format = "csv"
    if o["tuples"]:
        p.opts.tuples_only = True
    if o["expanded"]:
        p.opts.expanded = "on"
    if o["fieldsep"] is not None:
        p.opts.fieldsep = o["fieldsep"]
    for ps in o["pset"]:
        name, _, val = ps.partition("=")
        p.pset([name] + ([val] if val else []))
        p.take_output()
    if o["output"]:
        p.ofile = o["output"]
        try:
            env.write_file(o["output"], "")
        except OSError as err:
            ctx.e(f"{o['output']}: {err}\n")
            return 1
    log = world_log(world)
    p.log = log
    if not o["noprc"]:
        rc = None
        try:
            rc = env.read_file("~/.psqlrc")
        except OSError:
            pass
        if rc:
            saved_q = p.quiet
            p.run_script(rc, None)
            p.quiet = saved_q
    status = 0
    if o["list"]:
        p.run_meta("l", "")
        yield from _flush(ctx, p)
        return 0
    if o["cmds"]:
        try:
            if o["single"]:
                p.session.execute("BEGIN")
            for kind, val in o["cmds"]:
                if kind == "c":
                    p.file_ctx = None
                    run_command(p, val)
                else:
                    try:
                        text = env.read_file(val) if val != "-" else (yield from ctx.text('-'))
                    except OSError as err:
                        p.e(f"psql: error: {val}: {err}\n")
                        status = 1
                        break
                    p.run_script(text, val)
                yield from _flush(ctx, p)
                if p.vars.get("ERROR") == "true" and p.vars.get("ON_ERROR_STOP") in ("on", "1", "true"):
                    status = 1 if kind == "c" else 3
                    break
            if o["single"] and p.session.txn is not None:
                if p.session.txn.failed:
                    p.session.execute("ROLLBACK")
                else:
                    p.session.execute("COMMIT")
        except Quit:
            pass
        except _StopScript:
            status = 3
        yield from _flush(ctx, p)
        if p.vars.get("ERROR") == "true" and status == 0 and o["cmds"][-1][0] == "c":
            status = 1
        p.session.close()
        return status
    if not interactive:
        text = yield from ctx.text('-')
        try:
            p.run_script(text, None)
        except (Quit, _StopScript):
            pass
        yield from _flush(ctx, p)
        p.session.close()
        return 0
    # interactive
    p.w(p.banner())
    if getattr(world, "editor_ok", False):
        term = PsqlTerm(p, ctx.proc.pid)
        world.psql = term
        world.psql_last = term
        yield from _flush(ctx, p)
        while term.quit is None:
            yield ("forever",)
        if world.psql is term:
            world.psql = None
        p.session.close()
        return term.quit
    # no live terminal: a plain loop over the terminal's lines
    from ..fileio import readline
    while True:
        p.w(p.prompt())
        yield from _flush(ctx, p)
        line = yield from readline(ctx.k, ctx.proc.fds.get(0))
        if not line:
            p.w("\\q\n")
            break
        try:
            p.feed(line.decode("utf-8", "replace").rstrip("\n"))
        except Quit:
            break
        yield from _flush(ctx, p)
    yield from _flush(ctx, p)
    p.session.close()
    return 0


def run_command(p, text):
    """psql -c: one string sent as one query (several statements share an implicit transaction), unless it is a
    backslash command."""
    from ...postgreslite.errors import PgError
    s = text.strip()
    if s.startswith("\\"):
        for part in s.split("\\\\"):
            p.scan_line(part)
        if p.scan.text.strip():
            p.send(p.scan.text)
            p.scan.reset()
        return
    p.send(text)


def _flush(ctx, p):
    for stream, text in p.take_output():
        if stream == "out":
            ctx.w(text)
        else:
            ctx.e(text)
    yield from ctx.flush()


def world_log(world):
    """Every statement psql sent on this machine (for grading), across psql runs."""
    log = getattr(world, "sql_log", None)
    if log is None:
        log = []
        world.sql_log = log
    return log


class PsqlTerm:
    """An interactive psql attached to the live terminal."""

    def __init__(self, psql, pid):
        self.psql = psql
        self.pid = pid
        self.quit = None


@util("createdb")
def u_createdb(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-")]
    if not a:
        ctx.e('createdb: error: database creation failed: ERROR:  database "ada" already exists\n'
              if False else "createdb: error: no database name given\n")
        return 1
    from ...postgreslite.errors import PgError
    cluster = cluster_of(ctx.world)
    try:
        cluster.create_database(a[0], ctx.env.get("USER") or "ada")
    except PgError as e:
        ctx.e(f"createdb: error: database creation failed: ERROR:  {e.message}\n")
        return 1
    return 0


@util("dropdb")
def u_dropdb(ctx):
    a = [x for x in ctx.argv[1:] if not x.startswith("-")]
    if not a:
        ctx.e("dropdb: error: missing required argument database name\n")
        return 1
    cluster = cluster_of(ctx.world)
    if a[0] not in cluster.databases:
        if "--if-exists" in ctx.argv:
            ctx.e(f'NOTICE:  database "{a[0]}" does not exist, skipping\n')
            return 0
        ctx.e(f'dropdb: error: database removal failed: ERROR:  database "{a[0]}" does not exist\n')
        return 1
    del cluster.databases[a[0]]
    return 0


@util("pg_isready")
def u_pg_isready(ctx):
    a = ctx.argv[1:]
    host = a[a.index("-h") + 1] if "-h" in a and a.index("-h") + 1 < len(a) else None
    where = f"{host}:5432" if host else "/var/run/postgresql:5432"
    if ctx.host.extra.get("container") is not None or host not in (None, "localhost", "127.0.0.1"):
        resolver = ctx.host.extra.get("resolver")
        target = (resolver(host or "localhost") if resolver else ctx.world.resolve(host)) if (host or resolver) \
            else ctx.host
        if target is None or 5432 not in target.services:
            if "-q" not in a:
                ctx.w(f"{where} - no response\n")
            return 2
    if "-q" not in a:
        ctx.w(f"{where} - accepting connections\n")
    return 0

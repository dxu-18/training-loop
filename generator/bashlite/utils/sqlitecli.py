"""sqlite3: the SQLite 3.45 command-line shell, on database files of the simulated filesystem.

The engine is Python's sqlite3 module: the file's bytes are loaded into an in-memory database (deserialize) and
written back after every statement that changes it (serialize), so the file on the machine always holds the data.
Dot commands: .tables .schema .headers .mode (list csv column table line json) .import .read .output .once
.separator .nullvalue .databases .indexes .print .quit .exit .help.
"""

from __future__ import annotations

import csv
import io
import posixpath

from ..exec import util
from ..vfs import FSError

BANNER = 'SQLite version 3.45.1 2024-01-30 16:01:20\nEnter ".help" for usage hints.\n'
HELP = """.databases               List names and files of attached databases
.exit ?CODE?             Exit this program with return-code CODE
.headers on|off          Turn display of headers on or off
.help ?-all? ?PATTERN?   Show help text for PATTERN
.import FILE TABLE       Import data from FILE into TABLE
.indexes ?TABLE?         Show names of indexes
.mode MODE ?OPTIONS?     Set output mode
.nullvalue STRING        Use STRING in place of NULL values
.once ?OPTIONS? ?FILE?   Output for the next SQL command only to FILE
.output ?FILE?           Send output to FILE or stdout if FILE is omitted
.print STRING...         Print literal STRING
.quit                    Stop interpreting input stream, exit if primary.
.read FILE               Read input from FILE or command output
.schema ?PATTERN?        Show the CREATE statements matching PATTERN
.separator COL ?ROW?     Change the column and row separators
.tables ?TABLE?          List names of tables matching LIKE pattern TABLE
"""


class Shell:
    def __init__(self, ctx, path):
        import sqlite3
        self.ctx = ctx
        self.path = path
        self.con = sqlite3.connect(":memory:", isolation_level=None)
        self.mode = "list"
        self.headers = False
        self.sep = "|"
        self.null = ""
        self.out_file = None
        self.once = False
        self.out = []
        self.err = []
        self.failed = False
        if path:
            data = self._read(path)
            if data:
                try:
                    self.con.deserialize(data)
                except Exception:   # noqa: BLE001
                    self.err.append("Error: file is not a database\n")
                    self.failed = True

    def _read(self, path):
        try:
            node = self.ctx.fs.lookup(path)
        except FSError:
            return None
        return bytes(node.data)

    def save(self):
        if not self.path or self.failed:
            return
        data = serialize(self.con)
        try:
            node = self.ctx.fs.lookup(self.path)
            node.data[:] = data
            node.mtime = node.ctime = self.ctx.k.now
        except FSError:
            self.ctx.fs.write_file(self.path, data, "/", 0o644, self.ctx.proc.uid, self.ctx.proc.gid)

    def emit(self, text):
        if self.out_file:
            try:
                node = self.ctx.fs.lookup(self.out_file)
                node.data += text.encode()
            except FSError:
                self.ctx.fs.write_file(self.out_file, text, "/", 0o644, self.ctx.proc.uid, self.ctx.proc.gid)
            return
        self.out.append(text)

    # ---- SQL
    def run_sql(self, sql):
        import sqlite3
        if not sql.strip():
            return
        try:
            cur = self.con.cursor()
            stmts = _split(sql)
            for st in stmts:
                cur.execute(st)
                if cur.description:
                    names = [d[0] for d in cur.description]
                    self.show(names, cur.fetchall())
            if not self.con.in_transaction:
                self.save()
        except sqlite3.Error as e:
            msg = str(e)
            self.err.append(f"Parse error: {msg}\n" if "syntax" in msg else f"Runtime error: {msg}\n")
            self.failed_stmt = True
        if self.once:
            self.out_file = None
            self.once = False

    def save_if_autocommit(self):
        pass

    def fmt(self, v):
        if v is None:
            return self.null
        if isinstance(v, float):
            r = repr(v)
            return r[:-2] if r.endswith(".0") else r
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        return str(v)

    def show(self, names, rows):
        m = self.mode
        cells = [[self.fmt(v) for v in r] for r in rows]
        if m == "csv":
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            if self.headers:
                w.writerow(names)
            for r in cells:
                w.writerow(r)
            self.emit(buf.getvalue())
        elif m in ("column", "table", "box", "markdown"):
            widths = [len(n) for n in names]
            for r in cells:
                for j, c in enumerate(r):
                    widths[j] = max(widths[j], len(c))
            if m == "column":
                lines = ["  ".join(n.ljust(w) for n, w in zip(names, widths)).rstrip(),
                         "  ".join("-" * w for w in widths)]
                lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells]
            else:
                bar = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
                lines = [bar, "| " + " | ".join(n.ljust(w) for n, w in zip(names, widths)) + " |", bar]
                lines += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in cells]
                lines.append(bar)
            self.emit("\n".join(lines) + "\n")
        elif m == "line":
            w = max((len(n) for n in names), default=0)
            blocks = []
            for r in cells:
                blocks.append("\n".join(f"{n.rjust(w)} = {c}" for n, c in zip(names, r)))
            self.emit("\n\n".join(blocks) + ("\n" if blocks else ""))
        elif m == "json":
            import json
            if rows:
                self.emit("[" + ",\n".join(json.dumps(dict(zip(names, r)), separators=(",", ":")) for r in rows)
                          + "]\n")
        else:
            lines = []
            if self.headers:
                lines.append(self.sep.join(names))
            lines += [self.sep.join(r) for r in cells]
            self.emit("".join(ln + "\n" for ln in lines))

    # ---- dot commands
    def dot(self, line):
        parts = _dot_args(line[1:])
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd == "help":
            self.emit(HELP)
        elif cmd == "tables":
            pat = args[0] if args else "%"
            rows = self.con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE "
                                    "'sqlite_%' AND name LIKE ? ORDER BY 1", (pat,)).fetchall()
            names = [r[0] for r in rows]
            if names:
                w = max(len(n) for n in names) + 2
                per = max(1, 80 // w)
                lines = ["".join(n.ljust(w) for n in names[i:i + per]).rstrip() for i in range(0, len(names), per)]
                self.emit("\n".join(lines) + "\n")
        elif cmd == "schema":
            q = "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            params = ()
            if args:
                q += " AND tbl_name LIKE ?"
                params = (args[0],)
            for (sql,) in self.con.execute(q + " ORDER BY rowid", params).fetchall():
                self.emit(sql + ";\n")
        elif cmd == "indexes" or cmd == "indices":
            q = "SELECT name FROM sqlite_master WHERE type = 'index'"
            params = ()
            if args:
                q += " AND tbl_name LIKE ?"
                params = (args[0],)
            for (n,) in self.con.execute(q + " ORDER BY 1", params).fetchall():
                self.emit(n + "\n")
        elif cmd in ("headers", "header"):
            self.headers = bool(args) and args[0].lower() in ("on", "1", "yes", "true")
        elif cmd == "mode":
            if not args:
                self.emit(f"current output mode: {self.mode}\n")
            else:
                m = args[0].lower()
                if m in ("list", "csv", "column", "table", "box", "line", "json", "markdown", "tabs"):
                    self.mode = "list" if m == "tabs" else m
                    if m == "csv":
                        self.sep = ","
                    elif m == "tabs":
                        self.sep = "\t"
                    elif m == "list":
                        self.sep = "|"
                    if m in ("column", "table", "box", "markdown"):
                        self.headers = True
                else:
                    self.err.append(f'Error: mode should be one of: ascii box column csv html insert json line list '
                                    f'markdown qbox quote table tabs tcl\n')
        elif cmd == "separator":
            if args:
                self.sep = args[0]
        elif cmd == "nullvalue":
            self.null = args[0] if args else ""
        elif cmd == "databases":
            self.emit(f"main: {self.path or ''} r/w\n")
        elif cmd == "print":
            self.emit(" ".join(args) + "\n")
        elif cmd == "import":
            self.do_import(args)
        elif cmd == "read":
            if not args:
                self.err.append("Usage: .read FILE\n")
            else:
                data = self._read(self.abs(args[0]))
                if data is None:
                    self.err.append(f'Error: cannot open "{args[0]}"\n')
                else:
                    return self.script(data.decode("utf-8", "replace"))
        elif cmd in ("output", "once"):
            if args and args[0] not in ("stdout",):
                self.out_file = self.abs(args[-1])
                self.ctx.fs.write_file(self.out_file, b"", "/", 0o644, self.ctx.proc.uid, self.ctx.proc.gid)
                self.once = cmd == "once"
            else:
                self.out_file = None
        else:
            self.err.append(f'Error: unknown command or invalid arguments:  "{cmd}". Enter ".help" for help\n')
        return True

    def abs(self, p):
        return posixpath.normpath(posixpath.join(self.ctx.proc.cwd, p))

    def do_import(self, args):
        opts = [a for a in args if a.startswith("--")]
        args = [a for a in args if not a.startswith("--")]
        skip = 0
        if "--skip" in opts:
            pass
        if len(args) < 2:
            self.err.append("Usage: .import FILE TABLE\n")
            return
        data = self._read(self.abs(args[0]))
        if data is None:
            self.err.append(f"Error: cannot open \"{args[0]}\"\n")
            return
        table = args[1]
        text = data.decode("utf-8", "replace")
        if self.mode == "csv" or "--csv" in opts:
            rows = list(csv.reader(io.StringIO(text)))
        else:
            rows = [ln.split(self.sep) for ln in text.splitlines()]
        if not rows:
            return
        exists = self.con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not exists:
            header, rows = rows[0], rows[1:]
            self.con.execute(f'CREATE TABLE "{table}"(' + ",".join(f'"{h}" TEXT' for h in header) + ")")
        n = len(self.con.execute(f'SELECT * FROM "{table}" LIMIT 0').description)
        for k, r in enumerate(rows, 2):
            if len(r) != n:
                self.err.append(f"{args[0]}:{k}: expected {n} columns but found {len(r)} - "
                                f"{'filling the rest with NULL' if len(r) < n else 'extras ignored'}\n")
                r = (r + [None] * n)[:n]
            self.con.execute(f'INSERT INTO "{table}" VALUES (' + ",".join("?" * n) + ")", r)
        if not self.con.in_transaction:
            self.save()

    def script(self, text):
        buf = []
        for line in text.split("\n"):
            if not buf and line.strip().startswith("."):
                if not self.dot(line.strip()):
                    return False
                continue
            buf.append(line)
            joined = "\n".join(buf)
            import sqlite3
            if sqlite3.complete_statement(joined):
                self.run_sql(joined)
                buf = []
        if "\n".join(buf).strip():
            self.run_sql("\n".join(buf))
        return True


def serialize(con):
    """The database's bytes; an empty database is an empty file (as sqlite3 leaves a new one), and SQLite before
    3.42 refuses to serialize one."""
    import sqlite3
    try:
        return con.serialize()
    except sqlite3.OperationalError:
        return b""


def _split(sql):
    import sqlite3
    out, cur = [], ""
    for ch in sql:
        cur += ch
        if ch == ";" and sqlite3.complete_statement(cur):
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur)
    return out


def _dot_args(s):
    import shlex
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


@util("sqlite3")
def u_sqlite3(ctx):
    a = ctx.argv[1:]
    opts = {"header": None, "mode": None, "sep": None}
    rest = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-version", "--version"):
            ctx.w("3.45.1 2024-01-30 16:01:20 e876e51a0ed5c5b3126f52e532044363a014bc594cfefa87ffb5b82257cc467a "
                  "(64-bit)\n")
            return 0
        if x in ("-help", "--help"):
            ctx.w("Usage: sqlite3 [OPTIONS] [FILENAME [SQL...]]\nFILENAME is the name of an SQLite database. A new "
                  "database is created\nif the file does not previously exist. Defaults to :memory:.\n")
            return 0
        if x in ("-header", "-headers"):
            opts["header"] = True
        elif x == "-noheader":
            opts["header"] = False
        elif x in ("-csv", "-column", "-list", "-line", "-json", "-table", "-box", "-markdown"):
            opts["mode"] = x[1:]
        elif x == "-separator" and i + 1 < len(a):
            i += 1
            opts["sep"] = a[i]
        elif x in ("-batch", "-bail", "-interactive", "-echo"):
            pass
        elif x.startswith("-"):
            ctx.e(f"sqlite3: Error: unknown option: {x}\nUse -help for a list of options.\n")
            return 1
        else:
            rest.append(x)
        i += 1
    path = None
    if rest and rest[0] != ":memory:":
        path = posixpath.normpath(posixpath.join(ctx.proc.cwd, rest[0]))
    sh = Shell(ctx, path)
    if opts["mode"]:
        sh.dot(".mode " + opts["mode"])
    if opts["header"] is not None:
        sh.headers = opts["header"]
    if opts["sep"] is not None:
        sh.sep = opts["sep"]

    def flush():
        for t in sh.out:
            ctx.w(t)
        for t in sh.err:
            ctx.e(t)
        sh.out, sh.err = [], []
    if sh.failed:
        flush()
        return 1
    if path and sh._read(path) is None:
        sh.save()
    if len(rest) > 1:
        for cmd in rest[1:]:
            if cmd.startswith("."):
                sh.dot(cmd)
            else:
                sh.run_sql(cmd)
        flush()
        yield from ctx.flush()
        return 1 if getattr(sh, "failed_stmt", False) else 0
    if not ctx.stdin_is_tty():
        text = yield from ctx.text("-")
        sh.script(text)
        flush()
        yield from ctx.flush()
        return 0
    from ..fileio import readline
    ctx.w(BANNER)
    if not path:
        ctx.w("Connected to a transient in-memory database.\nUse \".open FILENAME\" to reopen on a persistent "
              "database.\n")
    buf = []
    while True:
        ctx.w("sqlite> " if not buf else "   ...> ")
        yield from ctx.flush()
        line = yield from readline(ctx.k, ctx.proc.fds.get(0))
        if not line:
            break
        text = line.decode("utf-8", "replace").rstrip("\n")
        if not buf and text.strip().startswith("."):
            go = sh.dot(text.strip())
            flush()
            if not go:
                break
            continue
        buf.append(text)
        import sqlite3
        joined = "\n".join(buf)
        if sqlite3.complete_statement(joined):
            sh.run_sql(joined)
            buf = []
            flush()
    flush()
    yield from ctx.flush()
    return 0

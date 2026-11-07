"""psql 16, the client: the statement scanner (psqlscan.l: quotes, $$, comments, parentheses, backslash commands,
:variables), prompts (%/%R%x%#), meta-commands, PostgreSQL's table printer (print.c: aligned with borders 0-2,
expanded, unaligned, csv, wrapped multi-line cells), error reports with LINE and a caret (fe-protocol3.c), \\timing,
\\copy, \\i, \\o, \\e, \\g, \\gx, \\set and variables.

The client is independent of the terminal: an Env object gives it files, a shell, an editor and a clock. Output goes
to self.out as (stream, text) pairs that the host writes.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

from . import types as T
from .errors import PgError
from .server import Cluster, Session, SERVER_VERSION

PSQL_VERSION = "16.4 (Ubuntu 16.4-0ubuntu0.24.04.2)"
_RIGHT = {"int2", "int4", "int8", "numeric", "float4", "float8", "oid", "money", "xid", "cid"}


class Env:
    """What psql needs from its machine (bash-lite supplies one; tests use a dict-backed one)."""
    cwd = "/home/ada"
    home = "/home/ada"
    columns = 80

    def read_file(self, path):         # -> text, or raises OSError(msg)
        raise OSError("No such file or directory")

    def write_file(self, path, text, append=False):
        raise OSError("Permission denied")

    def shell(self, cmd):              # -> (stdout text, status)
        return "", 127

    def now_us(self):
        return 0

    def spend_us(self, us):           # time a query takes (the machine's clock advances)
        pass

    def chdir(self, path):
        raise OSError("No such file or directory")


def disp_width(s: str) -> int:
    w = 0
    for ch in s:
        if unicodedata.combining(ch):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


# ---------------------------------------------------------------- the scanner

class Scanner:
    """Accumulates input lines into statements the way psqlscan.l does."""

    def __init__(self):
        self.buf = []          # characters of the current query buffer
        self.state = None      # None, "'", "e", '"', "$", "*"
        self.dollar = ""
        self.depth = 0         # parentheses
        self.cdepth = 0        # nested /* */

    @property
    def text(self):
        return "".join(self.buf)

    def reset(self):
        self.buf = []
        self.state = None
        self.depth = 0
        self.cdepth = 0

    def prompt_char(self):
        if self.state == "'" or self.state == "e":
            return "'"
        if self.state == '"':
            return '"'
        if self.state == "$":
            return "$"
        if self.state == "*":
            return "*"
        if self.depth > 0:
            return "("
        if self.buf:
            return "-"
        return "="


# ---------------------------------------------------------------- printing

class PrintOpts:
    def __init__(self):
        self.format = "aligned"
        self.border = 1
        self.expanded = "off"
        self.null = ""
        self.fieldsep = "|"
        self.recordsep = "\n"
        self.tuples_only = False
        self.footer = True
        self.title = None
        self.pager = "on"
        self.csv_fieldsep = ","
        self.linestyle = "ascii"
        self.columns = 0

    def copy(self):
        o = PrintOpts()
        o.__dict__.update(self.__dict__)
        return o


def _cell_lines(s):
    return s.split("\n")


def print_table(names, types, rows, opts: PrintOpts, title=None, footers=None, columns=80, count_footer=True):
    """psql's print.c for one result set. rows hold Python values; types give the text form and alignment."""
    cells = [[_fmt(v, t, opts) for v, t in zip(r, types)] for r in rows]
    fmt = opts.format
    expanded = opts.expanded == "on"
    if opts.expanded == "auto" and fmt == "aligned" and names:
        width = _aligned_width(names, cells, opts.border)
        cols = opts.columns or columns
        if width > cols and len(names) > 1 or width > cols:
            expanded = True
    if fmt == "csv":
        return _print_csv(names, cells, opts, expanded)
    if fmt == "unaligned":
        return _print_unaligned(names, cells, opts, rows, title, footers, expanded, count_footer)
    if expanded:
        return _print_expanded(names, types, cells, opts, title, footers, count_footer)
    return _print_aligned(names, types, cells, opts, title, footers, count_footer)


def _fmt(v, t, opts):
    if v is None:
        return opts.null
    return T.output(v, t)


def _aligned_width(names, cells, border):
    widths = _widths(names, cells)
    n = len(widths)
    if border == 0:
        return sum(widths) + (n - 1)
    if border == 1:
        return sum(widths) + 3 * (n - 1) + 2
    return sum(widths) + 3 * (n - 1) + 4


def _widths(names, cells):
    widths = [disp_width(n) for n in names] if names else []
    for row in cells:
        for j, s in enumerate(row):
            for ln in _cell_lines(s):
                w = disp_width(ln)
                if w > widths[j]:
                    widths[j] = w
    return widths


def _footer_text(n):
    return f"({n} row{'s' if n != 1 else ''})"


def _print_aligned(names, types, cells, opts, title, footers, count_footer):
    out = []
    border = opts.border
    widths = _widths(names, cells)
    ncol = len(names)
    right = [t.name in _RIGHT for t in types]
    total = _aligned_width(names, cells, border) if ncol else 0
    if title is None:
        title = opts.title
    if title and not opts.tuples_only:
        tw = disp_width(title)
        if total >= tw:
            out.append(" " * ((total - tw) // 2) + title)
        else:
            out.append(title)
    if not opts.tuples_only and ncol:
        hlines = [_cell_lines(n) for n in names]
        nl = max(len(h) for h in hlines)
        if border == 2:
            out.append("+" + "+".join("-" * (w + 2) for w in widths) + "+")
        for k in range(nl):
            parts = []
            line = "| " if border == 2 else (" " if border == 1 else "")
            for j in range(ncol):
                txt = hlines[j][k] if k < len(hlines[j]) else ""
                w = widths[j]
                tw = disp_width(txt)
                lpad = (w - tw) // 2
                cont = k + 1 < len(hlines[j])
                cell = " " * lpad + txt + " " * (w - tw - lpad)
                line += cell
                if j < ncol - 1:
                    if border == 0:
                        line += "+" if cont else " "
                    else:
                        line += ("+" if cont else " ") + "| "
                else:
                    if border == 2:
                        line += ("+" if cont else " ") + "|"
                    elif border == 1:
                        line += "+" if cont else " "
                    elif cont:
                        line += "+"
            out.append(line)
        if border == 0:
            out.append(" ".join("-" * w for w in widths))
        elif border == 1:
            out.append("+".join("-" * (w + (1 if j in (0, ncol - 1) else 2) + (1 if j == 0 else 0) +
                                      (1 if j == ncol - 1 else 0) - (1 if ncol == 1 else 0) * 0)
                                for j, w in enumerate(widths)) if False else
                       "+".join("-" * (w + 2) for w in widths))
        else:
            out.append("+" + "+".join("-" * (w + 2) for w in widths) + "+")
    for row in cells:
        lines = [_cell_lines(s) for s in row]
        nl = max((len(x) for x in lines), default=1)
        for k in range(nl):
            line = "| " if border == 2 else (" " if border == 1 else "")
            for j in range(ncol):
                txt = lines[j][k] if k < len(lines[j]) else ""
                w = widths[j]
                tw = disp_width(txt)
                cont = k + 1 < len(lines[j])
                last = j == ncol - 1
                if right[j] and not ("\n" in row[j]):
                    cell = " " * (w - tw) + txt
                elif last and not cont and border != 2:
                    cell = txt
                else:
                    cell = txt + " " * (w - tw)
                line += cell
                if not last:
                    if border == 0:
                        line += "+" if cont else " "
                    else:
                        line += ("+" if cont else " ") + "| "
                else:
                    if border == 2:
                        line += ("+" if cont else " ") + "|"
                    elif cont:
                        line += "+"
            out.append(line)
    if border == 2 and ncol:
        out.append("+" + "+".join("-" * (w + 2) for w in widths) + "+")
    if not opts.tuples_only:
        if footers:
            out += footers
        if count_footer and opts.footer:
            out.append(_footer_text(len(cells)))
    return ("\n".join(out) + "\n" if out else "") + "\n"


def _print_expanded(names, types, cells, opts, title, footers, count_footer):
    out = []
    border = opts.border
    if title is None:
        title = opts.title
    if title and not opts.tuples_only:
        out.append(title)
    if not cells:
        if not opts.tuples_only and opts.footer and count_footer:
            return "(0 rows)\n\n"
        return "\n"
    hw = max(disp_width(ln) for n in names for ln in _cell_lines(n)) if names else 0
    dw = 0
    for row in cells:
        for s in row:
            for ln in _cell_lines(s):
                dw = max(dw, disp_width(ln))
    for i, row in enumerate(cells, 1):
        if border == 2:
            head = f"+-[ RECORD {i} ]"
            total = hw + dw + 7
            out.append(head + "-" * max(0, total - len(head) - 1) + "+")
        elif not opts.tuples_only:
            head = f"-[ RECORD {i} ]"
            total = hw + 3 + dw if border == 1 else hw + 1 + dw
            out.append(head + "-" * max(0, total - len(head)))
        elif i > 1:
            out.append("")
        for name, s in zip(names, row):
            dlines = _cell_lines(s)
            hlines = _cell_lines(name)
            n = max(len(dlines), len(hlines))
            for k in range(n):
                h = hlines[k] if k < len(hlines) else ""
                d = dlines[k] if k < len(dlines) else ""
                cont = k + 1 < len(dlines)
                hcell = h + " " * (hw - disp_width(h))
                if border == 2:
                    out.append(f"| {hcell} | {d}{' ' * (dw - disp_width(d))} {'+' if cont else ''}|"
                               if cont else f"| {hcell} | {d}{' ' * (dw - disp_width(d))} |")
                elif border == 1:
                    out.append(f"{hcell} | {d}" + ("+" if cont else "") if not cont else
                               f"{hcell} | {d}{' ' * (dw - disp_width(d))}+")
                else:
                    out.append(f"{hcell} {d}")
    if border == 2:
        out.append("+" + "-" * (hw + 2) + "+" + "-" * (dw + 2) + "+")
    if footers and not opts.tuples_only:
        out += footers
    return "\n".join(out) + "\n\n"


def _print_unaligned(names, cells, opts, rows, title, footers, expanded, count_footer):
    sep = opts.fieldsep
    out = []
    if title is None:
        title = opts.title
    if title and not opts.tuples_only:
        out.append(title)
    if expanded:
        for i, row in enumerate(cells):
            if i:
                out.append("")
            for n, s in zip(names, row):
                out.append(f"{n}{sep}{s}")
        return "\n".join(out) + ("\n" if out else "")
    if not opts.tuples_only:
        out.append(sep.join(names))
    for row in cells:
        out.append(sep.join(row))
    if not opts.tuples_only:
        if footers:
            out += footers
        if count_footer and opts.footer:
            out.append(_footer_text(len(cells)))
    return "\n".join(out) + "\n"


def _csv_cell(s, sep):
    if s == "" or any(ch in s for ch in (sep, '"', "\n", "\r")) or s.strip(" \t") != s:
        if s == "":
            return s
        return '"' + s.replace('"', '""') + '"'
    return s


def _print_csv(names, cells, opts, expanded):
    sep = opts.csv_fieldsep
    out = []
    if expanded:
        for row in cells:
            for n, s in zip(names, row):
                out.append(f"{_csv_cell(n, sep)}{sep}{_csv_cell(s, sep)}")
        return "\n".join(out) + ("\n" if out else "")
    if not opts.tuples_only:
        out.append(sep.join(_csv_cell(n, sep) for n in names))
    for row in cells:
        out.append(sep.join(_csv_cell(s, sep) for s in row))
    return "\n".join(out) + ("\n" if out else "")


# ---------------------------------------------------------------- errors

DISPLAY_SIZE = 60
MIN_RIGHT_CUT = 10


def error_position(query: str, loc: int) -> str:
    """fe-protocol3.c reportErrorPosition: LINE n: <the line, maybe cut with ...> and a caret under loc."""
    loc -= 1
    if loc < 0:
        return ""
    w = query.replace("\t", " ")
    qidx = []
    scr = []
    sc = 0
    line = 1
    ibeg = 0
    iend = -1
    cno = 0
    for cno, ch in enumerate(w):
        qidx.append(cno)
        scr.append(sc)
        if ch in "\r\n":
            if cno < loc:
                if ch == "\r" or cno == 0 or w[cno - 1] != "\r":
                    line += 1
                ibeg = cno + 1
            else:
                iend = cno
                break
        cw = disp_width(ch)
        sc += cw if cw > 0 else 1
    else:
        cno = len(w)
    if iend < 0:
        iend = cno
        while len(scr) <= iend:
            scr.append(sc)
    if loc > cno:
        return ""
    while len(scr) <= max(loc, iend):
        scr.append(sc)
    beg_trunc = end_trunc = False
    if scr[iend] - scr[ibeg] > DISPLAY_SIZE:
        if scr[ibeg] + DISPLAY_SIZE >= scr[loc] + MIN_RIGHT_CUT:
            while scr[iend] - scr[ibeg] > DISPLAY_SIZE:
                iend -= 1
            end_trunc = True
        else:
            while scr[loc] + MIN_RIGHT_CUT < scr[iend]:
                iend -= 1
                end_trunc = True
            while scr[iend] - scr[ibeg] > DISPLAY_SIZE:
                ibeg += 1
                beg_trunc = True
    prefix = f"LINE {line}: " + ("..." if beg_trunc else "")
    text = w[ibeg:iend]
    msg = prefix + text + ("..." if end_trunc else "") + "\n"
    caret = disp_width(prefix) + scr[loc] - scr[ibeg]
    return msg + " " * caret + "^\n"


def format_error(e: PgError, query: str | None, verbosity="default", prefix=""):
    sev = getattr(e, "severity", "ERROR")
    out = f"{prefix}{sev}:  {e.message}\n"
    if verbosity == "verbose":
        out = f"{prefix}{sev}:  {e.code}: {e.message}\n"
    if e.pos is not None and query is not None:
        out += error_position(query, e.pos)
    if e.detail:
        out += f"DETAIL:  {e.detail}\n"
    if e.hint:
        out += f"HINT:  {e.hint}\n"
    if e.context and verbosity != "terse":
        out += f"CONTEXT:  {e.context}\n"
    return out


# ---------------------------------------------------------------- psql

class Quit(Exception):
    pass


class Psql:
    def __init__(self, cluster: Cluster, dbname: str, user: str = "ada", env: Env | None = None, *,
                 interactive=True, quiet=False, echo_queries=False, single_tx=False, seed=0, variables=None):
        self.cluster = cluster
        self.env = env or Env()
        self.interactive = interactive
        self.quiet = quiet
        self.opts = PrintOpts()
        self.vars = {"AUTOCOMMIT": "on", "COMP_KEYWORD_CASE": "preserve-upper", "ECHO": "none",
                     "ECHO_HIDDEN": "off", "ENCODING": "UTF8", "FETCH_COUNT": "0", "HIDE_TABLEAM": "off",
                     "HIDE_TOAST_COMPRESSION": "off", "HISTCONTROL": "none", "HISTSIZE": "500",
                     "LASTOID": "0", "ON_ERROR_ROLLBACK": "off", "ON_ERROR_STOP": "off", "PROMPT1": "%/%R%x%# ",
                     "PROMPT2": "%/%R%x%# ", "PROMPT3": ">> ", "SHOW_ALL_RESULTS": "on",
                     "SHOW_CONTEXT": "errors", "VERBOSITY": "default", "VERSION": f"PostgreSQL {SERVER_VERSION}",
                     "VERSION_NAME": "16.4", "VERSION_NUM": "160004", "QUIET": "on" if quiet else "off",
                     "USER": user, "HOST": "/var/run/postgresql", "PORT": "5432", "SERVER_VERSION_NAME": "16.4",
                     "SERVER_VERSION_NUM": "160004", "IGNOREEOF": "0", "SINGLELINE": "off", "SINGLESTEP": "off"}
        if variables:
            self.vars.update(variables)
        self.timing = False
        self.out: list = []
        self.scan = Scanner()
        self.last_query = ""
        self.history: list[str] = []
        self.hist_buf: list[str] = []
        self.ofile = None          # \o target
        self.last_error = None
        self.copy_pending = None   # COPY ... FROM STDIN typed at the prompt: (prepared, lines)
        self.file_ctx = None       # (name, line number) while running \i / -f
        self.edit_request = None   # \e: (path, text, line) for the host to open an editor
        self.seed = seed
        self.session = None
        self.log = []              # every query sent: {sql, results, error} (the grader reads it)
        self.connect(dbname, user)

    # ------------------------------------------------------------ connection
    def connect(self, dbname, user):
        s = self.cluster.connect(dbname, user, clock=self.env.now_us, seed=self.seed)
        if self.session is not None:
            self.session.close()
        self.session = s
        self.vars["DBNAME"] = dbname
        self.vars["USER"] = user

    @property
    def superuser(self):
        r = self.cluster.roles.get(self.session.user)
        return r is not None and r.superuser

    # ------------------------------------------------------------ output
    def w(self, text):
        if text:
            self.out.append(("out", text))

    def e(self, text):
        if text:
            self.out.append(("err", text))

    def take_output(self):
        out = self.out
        self.out = []
        return out

    def result_out(self, text):
        """Query results go to \\o's file when one is set."""
        if self.ofile is not None:
            try:
                self.env.write_file(self.ofile, text, append=True)
            except OSError as err:
                self.e(f"{self.ofile}: {err}\n")
            return
        self.w(text)

    def banner(self):
        return f'psql ({PSQL_VERSION})\nType "help" for help.\n\n'

    def prompt(self):
        p = self.vars["PROMPT2"] if self.scan.buf or self.scan.state else self.vars["PROMPT1"]
        if self.copy_pending is not None:
            return ">> "
        return self.expand_prompt(p)

    def expand_prompt(self, p):
        out = []
        i = 0
        while i < len(p):
            ch = p[i]
            if ch == "%" and i + 1 < len(p):
                k = p[i + 1]
                i += 2
                if k == "/":
                    out.append(self.session.dbname)
                elif k == "~":
                    out.append("~" if self.session.dbname == self.session.user else self.session.dbname)
                elif k == "R":
                    if self.scan.buf or self.scan.state:
                        out.append(self.scan.prompt_char())
                    else:
                        out.append("=")
                elif k == "x":
                    st = self.session.status
                    out.append({"I": "", "T": "*", "E": "!"}[st])
                elif k == "#":
                    out.append("#" if self.superuser else ">")
                elif k == "n":
                    out.append(self.session.user)
                elif k == "m" or k == "M":
                    out.append("[local]")
                elif k == ">":
                    out.append("5432")
                elif k == "%":
                    out.append("%")
                elif k == "l":
                    out.append(str(len(self.scan.text.split("\n"))))
                elif k == "p":
                    out.append(str(self.session.pid))
                elif k == "w":
                    out.append(" " * len(self.expand_prompt(self.vars["PROMPT1"])))
                else:
                    out.append("")
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    # ------------------------------------------------------------ input
    def feed(self, line: str):
        """One line of input (without its newline), as typed at the prompt or read from a file."""
        if self.copy_pending is not None:
            return self._copy_line(line)
        if self.interactive and self.file_ctx is None and not self.scan.buf and not self.scan.state:
            stripped = line.strip()
            low = stripped.rstrip(";").strip().lower()
            if stripped.lower() in ("help",):
                self.w('You are using psql, the command-line interface to PostgreSQL.\n'
                       'Type:  \\copyright for distribution terms\n'
                       '       \\h for help with SQL commands\n'
                       '       \\? for help with psql commands\n'
                       '       \\g or terminate with semicolon to execute query\n'
                       '       \\q to quit\n')
                return
            if stripped.lower() in ("quit", "exit", "quit;", "exit;"):
                raise Quit()
        if self.interactive and self.file_ctx is None and (line.strip() or self.scan.buf):
            self.hist_buf.append(line)
        self.scan_line(line)
        if self.interactive and self.file_ctx is None and not self.scan.buf and not self.scan.state and \
                self.hist_buf:
            entry = "\n".join(self.hist_buf).strip()
            if entry and (not self.history or self.history[-1] != entry):
                self.history.append(entry)
            self.hist_buf = []
        elif self.interactive and self.file_ctx is None and not self.scan.buf and not self.scan.state:
            self.hist_buf = []
        if self.interactive and self.scan.text.strip() == "" and not self.scan.state:
            pass

    def scan_line(self, line: str):
        sc = self.scan
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            st = sc.state
            if st in ("'", "e"):
                sc.buf.append(ch)
                if st == "e" and ch == "\\" and i + 1 < n:
                    sc.buf.append(line[i + 1])
                    i += 2
                    continue
                if ch == "'":
                    if i + 1 < n and line[i + 1] == "'":
                        sc.buf.append("'")
                        i += 2
                        continue
                    sc.state = None
                i += 1
                continue
            if st == '"':
                sc.buf.append(ch)
                if ch == '"':
                    sc.state = None
                i += 1
                continue
            if st == "$":
                if line.startswith(sc.dollar, i):
                    sc.buf.append(sc.dollar)
                    i += len(sc.dollar)
                    sc.state = None
                    continue
                sc.buf.append(ch)
                i += 1
                continue
            if st == "*":
                if line.startswith("*/", i):
                    sc.buf.append("*/")
                    i += 2
                    sc.cdepth -= 1
                    if sc.cdepth == 0:
                        sc.state = None
                    continue
                if line.startswith("/*", i):
                    sc.buf.append("/*")
                    i += 2
                    sc.cdepth += 1
                    continue
                sc.buf.append(ch)
                i += 1
                continue
            # normal state
            if ch in " \t\r\f" and not sc.buf:
                i += 1
                continue
            if line.startswith("--", i):
                if sc.buf:
                    sc.buf.append(line[i:])
                break
            if line.startswith("/*", i):
                sc.state = "*"
                sc.cdepth = 1
                sc.buf.append("/*")
                i += 2
                continue
            if ch == "'":
                prev = sc.buf[-1] if sc.buf else ""
                prev2 = sc.buf[-2] if len(sc.buf) > 1 else ""
                if prev in ("E", "e") and not (prev2.isalnum() or prev2 == "_"):
                    sc.state = "e"
                else:
                    sc.state = "'"
                sc.buf.append(ch)
                i += 1
                continue
            if ch == '"':
                sc.state = '"'
                sc.buf.append(ch)
                i += 1
                continue
            if ch == "$":
                m = re.match(r"\$([A-Za-z_][A-Za-z_0-9]*)?\$", line[i:])
                prev = sc.buf[-1] if sc.buf else ""
                if m and not (prev.isalnum() or prev == "_"):
                    sc.state = "$"
                    sc.dollar = m.group(0)
                    sc.buf.append(m.group(0))
                    i += m.end()
                    continue
            if ch == "(":
                sc.depth += 1
            elif ch == ")":
                if sc.depth > 0:
                    sc.depth -= 1
            elif ch == ":" and i + 1 < n and line[i + 1] != ":" and (not sc.buf or sc.buf[-1] != ":"):
                m = re.match(r":(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\1", line[i:])
                if m and m.group(2) in self.vars:
                    val = self.vars[m.group(2)]
                    q = m.group(1)
                    if q == "'":
                        val = "'" + val.replace("'", "''") + "'"
                    elif q == '"':
                        val = '"' + val.replace('"', '""') + '"'
                    rest = line[i + m.end():]
                    # psql scans the substituted text again
                    line = line[:i] + val + rest
                    n = len(line)
                    if q:
                        sc.buf.append(val)
                        i += len(val)
                    continue
                if m and m.group(1):
                    self.e(f"unrecognized value \"{m.group(2)}\"\n" if False else "")
            elif ch == ";" and sc.depth == 0:
                sc.buf.append(";")
                i += 1
                self.send(sc.text)
                sc.reset()
                continue
            elif ch == "\\":
                end = self.meta(line[i + 1:])
                if end is None:
                    return
                i += 1 + end
                continue
            sc.buf.append(ch)
            i += 1
        if sc.buf or sc.state:
            sc.buf.append("\n")

    # ------------------------------------------------------------ running SQL
    def send(self, sql, expanded=None, target=None, gset=None):
        sql = sql.strip("\n") if sql.endswith("\n") else sql
        if not sql.strip():
            return
        self.last_query = sql
        if self.vars.get("ECHO") == "queries" or self.vars.get("ECHO") == "all" and self.file_ctx is not None:
            self.w(sql + "\n")
        sess = self.session
        start = self.env.now_us()
        rec = {"sql": sql, "db": sess.dbname, "results": [], "error": None}
        self.log.append(rec)
        try:
            results = sess.execute(sql)
        except PgError as err:
            rec["error"] = (err.code, err.message)
            self.report_error(err, sql)
            self._timing(start, sql, [])
            return
        except RecursionError:
            rec["error"] = ("54001", "stack depth limit exceeded")
            self.report_error(PgError("stack depth limit exceeded", "54001",
                                      hint='Increase the configuration parameter "max_stack_depth" (currently '
                                           '2048kB), after ensuring the platform\'s stack depth limit is adequate.'),
                              sql)
            return
        rec["results"] = [_record(r) for r in results]
        for r in results:
            self.show_result(r, sql, expanded, target, gset)
        self._timing(start, sql, results)

    def _timing(self, start, sql, results):
        rows = sum(len(r.rows or []) for r in results)
        cost = 40 + 6 * len(sql) // 10 + 3 * rows
        cost = cost + (self.env.now_us() - start)
        self.env.spend_us(cost - (self.env.now_us() - start))
        if self.timing:
            ms = cost / 1000
            s = f"Time: {ms:.3f} ms"
            if ms >= 1000:
                secs = ms / 1000
                m, sec = divmod(secs, 60)
                s += f" ({int(m):02d}:{sec:06.3f})"
            self.w(s + "\n")

    def show_result(self, r, sql, expanded=None, target=None, gset=None):
        for sev, msg, detail, hint in r.notices:
            if sev == "NOTICE" and self.session.settings.get("client_min_messages") in ("warning", "error"):
                continue
            text = self._prefix() + f"{sev}:  {msg}\n"
            if detail:
                text += f"DETAIL:  {detail}\n"
            if hint:
                text += f"HINT:  {hint}\n"
            self.e(text)
        if r.copy is not None:
            kind, payload = r.copy
            if kind == "out":
                self.result_out(payload)
                if not self.quiet:
                    self.result_out(r.tag + "\n") if False else None
                if not self.quiet:
                    self.w(r.tag + "\n")
                return
            # COPY FROM STDIN typed at the prompt: collect lines until \.
            self.copy_pending = (payload, [])
            if self.interactive and self.file_ctx is None:
                self.w("Enter data to be copied followed by a newline.\n"
                       "End with a backslash and a period on a line by itself, or an EOF signal.\n")
            return
        self.vars["ROW_COUNT"] = str(len(r.rows) if r.rows is not None else _tag_count(r.tag))
        self.vars["ERROR"] = "false"
        self.vars["SQLSTATE"] = "00000"
        if r.has_rows:
            if gset is not None:
                if len(r.rows) != 1:
                    self.e("no rows returned for \\gset\n" if not r.rows else "more than one row returned for \\gset\n")
                    return
                for n, v, t in zip(r.names, r.rows[0], r.types):
                    if v is not None:
                        self.vars[gset + n] = T.output(v, t)
                    else:
                        self.vars.pop(gset + n, None)
                return
            opts = self.opts
            if expanded:
                opts = opts.copy()
                opts.expanded = "on"
            text = print_table(r.names, r.types, r.rows, opts, columns=self.env.columns)
            if target is not None:
                try:
                    self.env.write_file(target, text)
                except OSError as err:
                    self.e(f"{target}: {err}\n")
            else:
                self.result_out(text)
            if r.tag.split()[0] in ("INSERT", "UPDATE", "DELETE", "MERGE") and not self.quiet:
                self.result_out(r.tag + "\n")
            return
        if not self.quiet and r.tag:
            self.result_out(r.tag + "\n")

    def _prefix(self):
        if self.file_ctx is not None and self.file_ctx[0] is not None:
            name, ln = self.file_ctx
            return f"psql:{name}:{ln}: "
        return ""

    def report_error(self, err: PgError, sql):
        self.last_error = (err, sql)
        self.vars["ERROR"] = "true"
        self.vars["SQLSTATE"] = err.code
        self.vars["LAST_ERROR_MESSAGE"] = err.message
        self.vars["LAST_ERROR_SQLSTATE"] = err.code
        self.e(format_error(err, sql, self.vars.get("VERBOSITY", "default"), self._prefix()))
        if self.vars.get("ON_ERROR_STOP") in ("on", "1", "true") and self.file_ctx is not None:
            raise _StopScript()

    # ------------------------------------------------------------ COPY FROM STDIN at the prompt
    def _copy_line(self, line):
        prep, lines = self.copy_pending
        if line == "\\.":
            self.copy_pending = None
            data = "".join(ln + "\n" for ln in lines)
            try:
                r = self.session.copy_in(prep, data)
            except PgError as err:
                self.report_error(err, None)
                return
            if not self.quiet:
                self.w(r.tag + "\n")
            return
        lines.append(line)

    def end_of_input(self):
        """Ctrl-D at the prompt / end of a file: an unfinished COPY ends, a pending query is dropped."""
        if self.copy_pending is not None:
            self._copy_line("\\.")
            return False
        return True

    # ------------------------------------------------------------ meta-commands
    def meta(self, rest: str):
        """A backslash command at `rest` (after the backslash). -> characters consumed, or None for the whole line."""
        m = re.match(r"([A-Za-z?!+]+[0-9A-Za-z+]*|!|\?|;)", rest)
        if not m:
            self.e(f"invalid command \\{rest.split()[0] if rest.split() else ''}\n"
                   f"Try \\? for help.\n")
            return None
        cmd = m.group(1)
        argtext = rest[m.end():]
        # \\ ends a command early
        end = None
        if "\\\\" in argtext and cmd not in ("!", "copy", "h", "help"):
            k = argtext.index("\\\\")
            argtext, end = argtext[:k], m.end() + k + 2
        if cmd in ("!", "copy", "h", "help", "echo", "qecho", "warn", "set", "prompt", "setenv", "sf", "sv"):
            pass
        try:
            self.run_meta(cmd, argtext)
        except Quit:
            raise
        except _StopScript:
            raise
        except PgError as err:
            self.report_error(err, None)
        return end

    def args(self, text):
        """Split meta-command arguments: 'quoted' (with '' and \\ escapes), "quoted", `shell`, :variables."""
        out = []
        i = 0
        n = len(text)
        while True:
            while i < n and text[i] in " \t":
                i += 1
            if i >= n:
                return out
            buf = []
            quoted_ident = False
            while i < n and text[i] not in " \t":
                ch = text[i]
                if ch == "'":
                    j = i + 1
                    while j < n:
                        if text[j] == "'" and j + 1 < n and text[j + 1] == "'":
                            buf.append("'")
                            j += 2
                            continue
                        if text[j] == "'":
                            break
                        if text[j] == "\\" and j + 1 < n:
                            nx = text[j + 1]
                            buf.append({"n": "\n", "t": "\t", "\\": "\\", "'": "'"}.get(nx, nx))
                            j += 2
                            continue
                        buf.append(text[j])
                        j += 1
                    i = j + 1
                    continue
                if ch == '"':
                    j = text.find('"', i + 1)
                    j = n if j < 0 else j
                    buf.append(text[i:j + 1])
                    i = j + 1
                    continue
                if ch == ":":
                    mm = re.match(r":(['\"]?)([A-Za-z_][A-Za-z_0-9]*)\1", text[i:])
                    if mm and mm.group(2) in self.vars:
                        v = self.vars[mm.group(2)]
                        if mm.group(1) == "'":
                            v = "'" + v.replace("'", "''") + "'"
                        elif mm.group(1) == '"':
                            v = '"' + v + '"'
                        buf.append(v)
                        i += mm.end()
                        continue
                if ch == "`":
                    j = text.find("`", i + 1)
                    j = n if j < 0 else j
                    out_text, _ = self.env.shell(text[i + 1:j])
                    buf.append(out_text.rstrip("\n"))
                    i = j + 1
                    continue
                buf.append(ch)
                i += 1
            out.append("".join(buf))

    def run_meta(self, cmd, argtext):
        a = self.args(argtext)
        if cmd == "q" or cmd == "quit":
            raise Quit()
        if cmd == "?":
            self.help_meta(a)
            return
        if cmd in ("h", "help"):
            self.help_sql(argtext.strip())
            return
        if cmd == "copyright":
            self.w("PostgreSQL Database Management System\n(formerly known as Postgres, then as Postgres95)\n\n"
                   "Portions Copyright (c) 1996-2023, PostgreSQL Global Development Group\n\n"
                   "Portions Copyright (c) 1994, The Regents of the University of California\n\n"
                   "Permission to use, copy, modify, and distribute this software and its\ndocumentation for any "
                   "purpose, without fee, and without a written agreement\nis hereby granted, provided that the "
                   "above copyright notice and this\nparagraph and the following two paragraphs appear in all "
                   "copies.\n")
            return
        if cmd in ("g", "gx", "G"):
            buf = self.scan.text.strip() or self.last_query
            self.scan.reset()
            target = a[0] if a and not a[0].startswith("(") else None
            if target and target.startswith("|"):
                self.e("\\g with a pipe is not supported by postgres-lite\n")
                target = None
            self.send(buf, expanded=cmd == "gx", target=target)
            return
        if cmd == "gset":
            buf = self.scan.text.strip() or self.last_query
            self.scan.reset()
            self.send(buf, gset=a[0] if a else "")
            return
        if cmd == "gdesc":
            buf = self.scan.text.strip() or self.last_query
            self.scan.reset()
            self.gdesc(buf)
            return
        if cmd == "p":
            text = self.scan.text.rstrip("\n") or self.last_query
            self.w((text + "\n") if text else "Query buffer is empty.\n")
            return
        if cmd == "r":
            self.scan.reset()
            self.w("Query buffer reset (cleared).\n")
            return
        if cmd == "w":
            if not a:
                self.e("\\w: missing required argument\n")
                return
            try:
                self.env.write_file(a[0], (self.scan.text or self.last_query).rstrip("\n") + "\n")
            except OSError as err:
                self.e(f"{a[0]}: {err}\n")
            return
        if cmd in ("e", "edit"):
            text = self.scan.text or self.last_query
            if a:
                try:
                    text = self.env.read_file(a[0])
                except OSError:
                    text = ""
            self.edit_request = {"path": a[0] if a else None, "text": text}
            return
        if cmd == "x":
            self.toggle_expanded(a[0] if a else None)
            return
        if cmd == "timing":
            v = a[0].lower() if a else None
            if v is None:
                self.timing = not self.timing
            else:
                self.timing = _bool(v, "\\timing")
            self.w(f"Timing is {'on' if self.timing else 'off'}.\n")
            return
        if cmd == "pset":
            self.pset(a)
            return
        if cmd == "a":
            self.opts.format = "aligned" if self.opts.format != "aligned" else "unaligned"
            self.w(f"Output format is {self.opts.format}.\n")
            return
        if cmd == "t":
            self.opts.tuples_only = (not self.opts.tuples_only) if not a else _bool(a[0], "\\t")
            self.w(f"Tuples only is {'on' if self.opts.tuples_only else 'off'}.\n")
            return
        if cmd == "f":
            if a:
                self.opts.fieldsep = a[0]
            self.w(f'Field separator is "{self.opts.fieldsep}".\n')
            return
        if cmd == "H":
            self.e("HTML output is not supported by postgres-lite\n")
            return
        if cmd == "C":
            self.opts.title = a[0] if a else None
            self.w(f'Title is "{self.opts.title}".\n' if self.opts.title else "Title is unset.\n")
            return
        if cmd in ("echo", "qecho", "warn"):
            nl = True
            if a and a[0] == "-n":
                nl = False
                a = a[1:]
            text = " ".join(a) + ("\n" if nl else "")
            if cmd == "warn":
                self.e(text)
            elif cmd == "qecho":
                self.result_out(text)
            else:
                self.w(text)
            return
        if cmd == "set":
            if not a:
                for k in sorted(self.vars):
                    self.w(f"{k} = '{self.vars[k]}'\n")
                return
            name = a[0]
            if not re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", name):
                self.e(f'invalid variable name: "{name}"\n')
                return
            self.vars[name] = "".join(a[1:])
            self._var_hook(name)
            return
        if cmd == "unset":
            if a:
                self.vars.pop(a[0], None)
            return
        if cmd in ("c", "connect"):
            self.reconnect(a)
            return
        if cmd == "conninfo":
            self.w(f'You are connected to database "{self.session.dbname}" as user "{self.session.user}" via socket '
                   f'in "/var/run/postgresql" at port "5432".\n')
            return
        if cmd == "encoding":
            self.w("UTF8\n")
            return
        if cmd == "errverbose":
            if self.last_error is None:
                self.w("There is no previous error.\n")
                return
            err, sql = self.last_error
            text = format_error(err, sql, "verbose")
            self.e(text)
            return
        if cmd == "i" or cmd == "ir" or cmd == "include":
            if not a:
                self.e("\\i: missing required argument\n")
                return
            self.include(a[0])
            return
        if cmd == "o" or cmd == "out":
            if a:
                if a[0].startswith("|"):
                    self.e("\\o with a pipe is not supported by postgres-lite\n")
                    return
                try:
                    self.env.write_file(a[0], "")
                except OSError as err:
                    self.e(f"{a[0]}: {err}\n")
                    return
                self.ofile = a[0]
            else:
                self.ofile = None
            return
        if cmd == "!":
            out, status = self.env.shell(argtext.strip() or "bash")
            self.w(out)
            return
        if cmd == "cd":
            path = a[0] if a else self.env.home
            try:
                self.env.chdir(path)
            except OSError as err:
                self.e(f"\\cd: could not change directory to \"{path}\": {err}\n")
            return
        if cmd == "s":
            text = "".join(h + "\n" for h in self.history)
            if a:
                try:
                    self.env.write_file(a[0], text)
                    self.w(f'Wrote history to file "{a[0]}".\n')
                except OSError as err:
                    self.e(f"{a[0]}: {err}\n")
            else:
                self.w(text)
            return
        if cmd == "copy":
            self.copy(argtext)
            return
        if cmd == "watch":
            self.e("\\watch is not supported by postgres-lite (it runs the query until you press Ctrl-C)\n")
            return
        if cmd in ("password", "crosstabview", "lo_import", "lo_export", "lo_list", "lo_unlink", "bind",
                   "if", "elif", "else", "endif", "gexec", "getenv", "setenv", "prompt", "sf", "sv", "ef", "ev",
                   "z", "dp", "drds", "dRp", "dRs", "dx", "dX", "dy", "dl", "dL", "dc", "dC", "dd", "dD", "des",
                   "deu", "dew", "det", "dE", "dF", "dFd", "dFp", "dFt", "dg", "do", "dO", "dP", "lo"):
            if cmd == "dx":
                from .describe import describe_extensions
                self.result_out(describe_extensions(self))
                return
            if cmd == "sv" and a:
                from .describe import show_view
                self.w(show_view(self, a[0]))
                return
            self.e(f"\\{cmd} is not supported by postgres-lite\n")
            return
        if cmd.startswith("d") or cmd in ("l", "list", "l+", "list+"):
            from .describe import describe
            text, err = describe(self, cmd, a)
            if err:
                self.e(self._prefix() + err if not self.interactive or self.file_ctx else err)
            if text:
                self.result_out(text)
            return
        self.e(f"invalid command \\{cmd}\nTry \\? for help.\n")

    def _var_hook(self, name):
        if name == "QUIET":
            self.quiet = self.vars[name].lower() in ("on", "1", "true", "")

    def toggle_expanded(self, v):
        o = self.opts
        if v is None:
            o.expanded = "off" if o.expanded == "on" else "on"
        elif v.lower() == "auto":
            o.expanded = "auto"
        else:
            o.expanded = "on" if _bool(v, "\\x") else "off"
        self.w({"on": "Expanded display is on.\n", "off": "Expanded display is off.\n",
                "auto": "Expanded display is used automatically.\n"}[o.expanded])

    def pset(self, a):
        o = self.opts
        if not a:
            items = [("border", str(o.border)), ("columns", str(o.columns)), ("csv_fieldsep", f"'{o.csv_fieldsep}'"),
                     ("expanded", o.expanded), ("fieldsep", f"'{o.fieldsep}'"), ("fieldsep_zero", "off"),
                     ("footer", "on" if o.footer else "off"), ("format", o.format), ("linestyle", o.linestyle),
                     ("null", f"'{o.null}'"), ("numericlocale", "off"), ("pager", "1" if o.pager == "on" else "0"),
                     ("pager_min_lines", "0"), ("recordsep", "'\\n'"), ("recordsep_zero", "off"),
                     ("tableattr", ""), ("title", ""), ("tuples_only", "on" if o.tuples_only else "off"),
                     ("unicode_border_linestyle", "single"), ("unicode_column_linestyle", "single"),
                     ("unicode_header_linestyle", "single"), ("xheader_width", "full")]
            for k, v in items:
                self.w(f"{k:<24} {v}\n")
            return
        name = a[0]
        val = a[1] if len(a) > 1 else None
        if name == "format":
            if val is not None:
                choices = ["aligned", "asciidoc", "csv", "html", "latex", "latex-longtable", "troff-ms",
                           "unaligned", "wrapped"]
                hits = [c for c in choices if c.startswith(val.lower())]
                if not hits:
                    self.e('\\pset: allowed formats are aligned, asciidoc, csv, html, latex, latex-longtable, '
                           'troff-ms, unaligned, wrapped\n')
                    return
                f = hits[0] if len(hits) == 1 or val.lower() in hits else hits[0]
                if f in ("asciidoc", "html", "latex", "latex-longtable", "troff-ms"):
                    self.e(f"\\pset: format {f} is not supported by postgres-lite\n")
                    return
                o.format = "aligned" if f == "wrapped" else f
            self.w(f"Output format is {o.format}.\n")
        elif name == "border":
            if val is not None:
                o.border = max(0, min(2, int(val) if val.isdigit() else 1))
            self.w(f"Border style is {o.border}.\n")
        elif name == "expanded" or name == "x":
            self.toggle_expanded(val)
        elif name == "null":
            if val is not None:
                o.null = val
            self.w(f'Null display is "{o.null}".\n')
        elif name == "fieldsep":
            if val is not None:
                o.fieldsep = val
            self.w(f'Field separator is "{o.fieldsep}".\n')
        elif name == "csv_fieldsep":
            if val is not None:
                o.csv_fieldsep = val
            self.w(f'Field separator for CSV is "{o.csv_fieldsep}".\n')
        elif name in ("tuples_only", "t"):
            o.tuples_only = (not o.tuples_only) if val is None else _bool(val, "\\pset")
            self.w(f"Tuples only is {'on' if o.tuples_only else 'off'}.\n")
        elif name == "footer":
            o.footer = (not o.footer) if val is None else _bool(val, "\\pset")
            self.w(f"Default footer is {'on' if o.footer else 'off'}.\n")
        elif name == "title":
            o.title = val
            self.w(f'Title is "{val}".\n' if val else "Title is unset.\n")
        elif name == "pager":
            if val is None:
                o.pager = "off" if o.pager == "on" else "on"
            elif val.lower() == "always":
                o.pager = "always"
            else:
                o.pager = "on" if _bool(val, "\\pset") else "off"
            self.w({"on": "Pager is used for long output.\n", "off": "Pager usage is off.\n",
                    "always": "Pager is always used.\n"}[o.pager])
        elif name == "linestyle":
            if val is not None:
                if val not in ("ascii", "old-ascii", "unicode"):
                    self.e("\\pset: allowed line styles are ascii, old-ascii, unicode\n")
                    return
                o.linestyle = val
            self.w(f'Line style is {o.linestyle}.\n')
        elif name == "columns":
            if val is not None:
                o.columns = int(val) if val.isdigit() else 0
            self.w(f"Target width is {o.columns if o.columns else 'unset'}.\n")
        elif name == "recordsep":
            self.w('Record separator is <newline>.\n')
        else:
            self.e(f"\\pset: unknown option: {name}\n")

    def reconnect(self, a):
        db = a[0] if a and a[0] != "-" else self.session.dbname
        user = a[1] if len(a) > 1 and a[1] != "-" else self.session.user
        if db not in self.cluster.databases:
            self.e(f'connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  database '
                   f'"{db}" does not exist\nPrevious connection kept\n')
            return
        if user not in self.cluster.roles:
            self.e(f'connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  role '
                   f'"{user}" does not exist\nPrevious connection kept\n')
            return
        self.connect(db, user)
        self.w(f'You are now connected to database "{db}" as user "{user}".\n')

    def include(self, path):
        try:
            text = self.env.read_file(path)
        except OSError as err:
            self.e(f"{path}: {err}\n")
            return
        self.run_script(text, path)

    def run_script(self, text, name):
        saved = (self.file_ctx, self.interactive, self.scan)
        self.scan = Scanner()
        try:
            lines = text.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            for k, line in enumerate(lines, 1):
                self.file_ctx = (name, k)
                self.feed(line)
            if self.copy_pending is not None:
                self.end_of_input()
            if self.scan.text.strip():
                self.send(self.scan.text)
        except _StopScript:
            raise
        finally:
            self.file_ctx, self.interactive, self.scan = saved[0], saved[1], saved[2]

    def gdesc(self, sql):
        from .executor import Ctx
        try:
            results = self.session.execute(sql)
        except PgError as err:
            self.report_error(err, sql)
            return
        r = results[-1]
        if not r.has_rows:
            self.w("The command has no result, or the result has no columns.\n")
            return
        rows = [(n, t.format()) for n, t in zip(r.names, r.types)]
        self.result_out(print_table(["Column", "Type"], [T.NAME, T.TEXT], rows, self.opts))

    def help_meta(self, a):
        path = os.path.join(os.path.dirname(__file__), "psql_help.txt")
        with open(path, encoding="utf-8") as f:
            self.w(f.read())

    def help_sql(self, topic):
        path = os.path.join(os.path.dirname(__file__), "psql_h.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not topic:
            self.w(data["list"])
            return
        t = " ".join(topic.upper().split()).rstrip(";")
        cmds = data["cmds"]
        if t == "*":
            self.w("\n".join(cmds[k] for k in sorted(cmds)))
            return
        hits = [k for k in cmds if k == t] or [k for k in cmds if k.startswith(t)]
        if not hits:
            words = t.split()
            while words and not hits:
                words = words[:-1]
                pref = " ".join(words)
                hits = [k for k in cmds if k == pref] if pref else []
        if not hits:
            self.w(f'No help available for "{topic}".\nTry \\h with no arguments to see available help.\n')
            return
        self.w("\n".join(cmds[k] for k in sorted(hits)))

    # ------------------------------------------------------------ \copy
    def copy(self, argtext):
        text = argtext.strip()
        m = re.match(r"(?is)^(\(.*\)|[^\s(]+(?:\s*\([^)]*\))?)\s+(from|to)\s+(.*)$", text)
        if not m:
            self.e("\\copy: parse error at end of line\n" if text else "\\copy: arguments required\n")
            return
        what, direction, rest = m.group(1), m.group(2).lower(), m.group(3)
        rest = rest.strip()
        fm = re.match(r"""^('(?:[^']|'')*'|"[^"]*"|\S+)\s*(.*)$""", rest, re.S)
        target, opts = fm.group(1), fm.group(2)
        if target.startswith("'"):
            fname = target[1:-1].replace("''", "'")
        elif target.startswith('"'):
            fname = target[1:-1]
        else:
            fname = target
        low = fname.lower()
        sql = f"COPY {what} {direction.upper()} {'STDOUT' if direction == 'to' else 'STDIN'} {opts}".rstrip()
        try:
            if direction == "to":
                results = self.session.execute(sql)
                r = results[0]
                data = r.copy[1] if r.copy else ""
                if low in ("stdout", "pstdout"):
                    self.w(data)
                else:
                    try:
                        self.env.write_file(fname, data)
                    except OSError as err:
                        self.e(f"{fname}: {err}\n")
                        return
                if not self.quiet:
                    self.w(r.tag + "\n")
            else:
                if low in ("stdin", "pstdin"):
                    results = self.session.execute(sql)
                    r = results[0]
                    self.copy_pending = (r.copy[1], [])
                    if self.interactive and self.file_ctx is None:
                        self.w("Enter data to be copied followed by a newline.\n"
                               "End with a backslash and a period on a line by itself, or an EOF signal.\n")
                    return
                try:
                    data = self.env.read_file(fname)
                except OSError as err:
                    self.e(f"{fname}: {err}\n")
                    return
                results = self.session.execute(sql)
                r = results[0]
                r2 = self.session.copy_in(r.copy[1], data)
                if not self.quiet:
                    self.w(r2.tag + "\n")
        except PgError as err:
            self.report_error(err, sql if err.pos is None else sql)


class _StopScript(Exception):
    pass


def _record(r):
    """What the grader keeps of a result: columns, types, rows, the command tag, and which output columns the
    query sorted by (so ties can be compared in any order)."""
    out = {"tag": r.tag, "names": r.names, "types": [t.name for t in r.types] if r.types else None,
           "rows": r.rows, "typeobjs": r.types, "order": None}
    q = getattr(r, "query", None)
    if q is not None and getattr(q, "order", None):
        out["order"] = [(i, desc, nf) for i, desc, nf, _ in q.order]
        out["order_visible"] = all(i < q.ntargets for i, *_ in q.order)
    return out


def _bool(v, where):
    v = v.lower()
    if v in ("on", "true", "1", "yes", "t", "y"):
        return True
    if v in ("off", "false", "0", "no", "f", "n"):
        return False
    raise PgError(f'unrecognized value "{v}" for "{where}": Boolean expected', "22023")


def _tag_count(tag):
    parts = (tag or "").split()
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return 0

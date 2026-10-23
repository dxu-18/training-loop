"""Shell builtins. Each is a generator `fn(sh, args) -> status`."""

from __future__ import annotations

import posixpath
import re

from . import pattern as P
from .fileio import open_path, read, readline
from .kernel import NS, SIG_DESC, SIGCONT, SIGHUP, SIGKILL, SIGNALS, SIGNUM, SIGTERM, Signal
from .parser import ASSIGN_RE, RESERVED, Parser, ParseError
from .printf import escapes, format_printf
from .quoting import declare_value, single_quote, xtrace_word
from .shell import (BREAK, BreakLoop, ContinueLoop, ExitShell, ReturnFunc, ShellAbort, Var, assoc_keys)
from .vfs import FSError

BUILTINS: dict = {}
SPECIAL = {":", ".", "break", "continue", "eval", "exec", "exit", "export", "readonly", "return", "set", "shift",
           "trap", "unset", "source"}


def builtin(*names):
    def reg(fn):
        for n in names:
            BUILTINS[n] = fn
        return fn
    return reg


def _err(sh, msg):
    yield from sh.error(msg)


def _usage(sh, name, usage):
    yield from sh.error(f"{name}: usage: {usage}")
    return 2


def _out(sh, text):
    try:
        yield from sh.out(text)
        return True
    except FSError as e:
        yield from sh.error(f"{'echo' if True else ''}: write error: {e.msg}")
        return False


# ---------------------------------------------------------------------------- trivial

@builtin(":", "true")
def b_true(sh, args):
    return 0
    yield


@builtin("false")
def b_false(sh, args):
    return 1
    yield


# ---------------------------------------------------------------------------- echo / printf

@builtin("echo")
def b_echo(sh, args):
    a = args[1:]
    nl, esc = True, "xpg_echo" in sh.shopts
    i = 0
    while i < len(a) and re.fullmatch(r"-[neE]+", a[i]):
        for c in a[i][1:]:
            if c == "n":
                nl = False
            elif c == "e":
                esc = True
            else:
                esc = False
        i += 1
    text = " ".join(a[i:])
    if esc:
        text, stop = escapes(text, echo=True)
        if stop:
            nl = False
    try:
        yield from sh.out(text + ("\n" if nl else ""))
    except FSError as e:
        yield from sh.error(f"echo: write error: {e.msg}")
        return 1
    return 0


@builtin("printf")
def b_printf(sh, args):
    a = args[1:]
    var = None
    if a and a[0] == "-v":
        if len(a) < 2:
            yield from sh.error("printf: -v: option requires an argument")
            return 2
        var = a[1]
        a = a[2:]
    if a and a[0] == "--":
        a = a[1:]
    if not a:
        yield from sh.error("printf: usage: printf [-v var] format [arguments]")
        return 2
    from .timeutil import strftime_epoch
    text, errors, _stop = format_printf(a[0], a[1:], lambda f, v: strftime_epoch(sh, f, v))
    for e in errors:
        yield from sh.error(e)
    if var is not None:
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(.*)\])?", var)
        if not m:
            yield from sh.error(f"printf: `{var}': not a valid identifier")
            return 2
        idx = m.group(2)
        if idx is not None and not sh.is_assoc(m.group(1)):
            idx = sh.arith(idx)
        sh.set_var(m.group(1), text, idx)
    else:
        try:
            yield from sh.out(text)
        except FSError as e:
            yield from sh.error(f"printf: write error: {e.msg}")
            return 1
    return 1 if errors else 0


# ---------------------------------------------------------------------------- directories

def _normalize(path: str) -> str:
    parts = []
    for c in path.split("/"):
        if c in ("", "."):
            continue
        if c == "..":
            if parts:
                parts.pop()
            continue
        parts.append(c)
    return "/" + "/".join(parts)


def change_dir(sh, target: str, physical=False, quiet=False, name="cd"):
    """-> error message or None. Updates PWD/OLDPWD and the process cwd."""
    cur = sh.get("PWD") or sh.proc.cwd
    new = target if target.startswith("/") else (cur.rstrip("/") + "/" + target)
    logical = _normalize(new)
    fs = sh.fs
    try:
        node = fs.lookup(new if physical else logical, "/", True, sh.proc.uid, sh.proc.gids)
    except FSError as e:
        return f"{name}: {target}: {e.msg}"
    if node.type != "d":
        return f"{name}: {target}: Not a directory"
    if not fs.can(node, 1, sh.proc.uid, sh.proc.gids):
        return f"{name}: {target}: Permission denied"
    if physical:
        logical = fs.path_of(node)
    sh.set_var("OLDPWD", cur, check=False)
    sh.proc.cwd = logical
    sh.set_var("PWD", logical, check=False)
    sh.dir_logical = logical
    return None


@builtin("cd")
def b_cd(sh, args):
    a = args[1:]
    physical = False
    while a and a[0].startswith("-") and a[0] not in ("-",) and len(a[0]) > 1:
        if a[0] == "--":
            a = a[1:]
            break
        for c in a[0][1:]:
            if c == "P":
                physical = True
            elif c == "L":
                physical = False
            elif c in "e@":
                pass
            else:
                yield from sh.error(f"cd: -{c}: invalid option")
                yield from sh.err("cd: usage: cd [-L|[-P [-e]] [-@]] [dir]\n")
                return 2
        a = a[1:]
    if len(a) > 1:
        yield from sh.error("cd: too many arguments")
        return 1
    show = False
    if not a:
        target = sh.get("HOME")
        if target is None:
            yield from sh.error("cd: HOME not set")
            return 1
    elif a[0] == "-":
        target = sh.get("OLDPWD")
        if target is None:
            yield from sh.error("cd: OLDPWD not set")
            return 1
        show = True
    else:
        target = a[0]
    if target == "":
        return 0
    cdpath = sh.get("CDPATH")
    if cdpath and not target.startswith(("/", "./", "../")) and target not in (".", ".."):
        for d in cdpath.split(":"):
            cand = (d or ".").rstrip("/") + "/" + target
            base = cand if cand.startswith("/") else (sh.get("PWD", sh.proc.cwd).rstrip("/") + "/" + cand)
            try:
                if sh.fs.lookup(_normalize(base)).type == "d":
                    msg = change_dir(sh, cand, physical)
                    if msg is None:
                        if d:
                            yield from sh.out(sh.get("PWD") + "\n")
                        return 0
            except FSError:
                continue
    msg = change_dir(sh, target, physical)
    if msg:
        yield from sh.error(msg)
        return 1
    if show:
        yield from sh.out(sh.get("PWD") + "\n")
    return 0


@builtin("pwd")
def b_pwd(sh, args):
    physical = "-P" in args[1:]
    for a in args[1:]:
        if a not in ("-P", "-L"):
            yield from sh.error(f"pwd: {a}: invalid option")
            yield from sh.err("pwd: usage: pwd [-LP]\n")
            return 2
    cwd = sh.get("PWD") or sh.proc.cwd
    if physical:
        try:
            cwd = sh.fs.realpath(cwd)
        except FSError:
            pass
    yield from sh.out(cwd + "\n")
    return 0


def _dirs_line(sh, full=False):
    stack = [sh.get("PWD", sh.proc.cwd)] + list(reversed(sh.dirstack))
    return " ".join(d if full else sh.tilde_short(d) for d in stack)


@builtin("pushd")
def b_pushd(sh, args):
    a = [x for x in args[1:] if x != "-n"]
    if not a:
        if not sh.dirstack:
            yield from sh.error("pushd: no other directory")
            return 1
        target = sh.dirstack.pop()
        cur = sh.get("PWD", sh.proc.cwd)
        msg = change_dir(sh, target, name="pushd")
        if msg:
            sh.dirstack.append(target)
            yield from sh.error(msg)
            return 1
        sh.dirstack.append(cur)
    elif re.fullmatch(r"[+-]\d+", a[0]):
        stack = [sh.get("PWD", sh.proc.cwd)] + list(reversed(sh.dirstack))
        n = int(a[0][1:])
        if n >= len(stack):
            yield from sh.error(f"pushd: {a[0]}: directory stack index out of range")
            return 1
        k = n if a[0][0] == "+" else len(stack) - 1 - n
        rot = stack[k:] + stack[:k]
        msg = change_dir(sh, rot[0], name="pushd")
        if msg:
            yield from sh.error(msg)
            return 1
        sh.dirstack = list(reversed(rot[1:]))
    else:
        cur = sh.get("PWD", sh.proc.cwd)
        msg = change_dir(sh, a[0], name="pushd")
        if msg:
            yield from sh.error(msg)
            return 1
        sh.dirstack.append(cur)
    yield from sh.out(_dirs_line(sh) + "\n")
    return 0


@builtin("popd")
def b_popd(sh, args):
    a = [x for x in args[1:] if x != "-n"]
    if not sh.dirstack:
        yield from sh.error("popd: directory stack empty")
        return 1
    if a and re.fullmatch(r"[+-]\d+", a[0]):
        stack = [sh.get("PWD", sh.proc.cwd)] + list(reversed(sh.dirstack))
        n = int(a[0][1:])
        k = n if a[0][0] == "+" else len(stack) - 1 - n
        if k == 0:
            msg = change_dir(sh, stack[1], name="popd")
            if msg:
                yield from sh.error(msg)
                return 1
            stack = stack[1:]
        else:
            del stack[k]
        sh.dirstack = list(reversed(stack[1:]))
    else:
        target = sh.dirstack.pop()
        msg = change_dir(sh, target, name="popd")
        if msg:
            yield from sh.error(msg)
            return 1
    yield from sh.out(_dirs_line(sh) + "\n")
    return 0


@builtin("dirs")
def b_dirs(sh, args):
    a = args[1:]
    if "-c" in a:
        sh.dirstack = []
        return 0
    stack = [sh.get("PWD", sh.proc.cwd)] + list(reversed(sh.dirstack))
    full = "-l" in a
    show = [d if full else sh.tilde_short(d) for d in stack]
    if "-v" in a:
        yield from sh.out("".join(f"{i:2d}  {d}\n" for i, d in enumerate(show)))
    elif "-p" in a:
        yield from sh.out("".join(d + "\n" for d in show))
    else:
        yield from sh.out(" ".join(show) + "\n")
    return 0


# ---------------------------------------------------------------------------- variables

ATTR_ORDER = "aAilnrtux"


def attr_flags(var) -> str:
    return "".join(c for c in ATTR_ORDER if c in var.attrs)


def declare_line(sh, name, var) -> str:
    flags = attr_flags(var) or "-"
    if var.value is None:
        return f"declare -{flags} {name}"
    if isinstance(var.value, dict):
        if "A" in var.attrs:
            items = " ".join(f"[{_assoc_key(k)}]={declare_value(var.value[k])}" for k in assoc_keys(var.value))
            return f"declare -{flags} {name}=({items} )" if items else f"declare -{flags} {name}=()"
        items = " ".join(f"[{k}]={declare_value(v)}" for k, v in sorted(var.value.items()))
        return f"declare -{flags} {name}=({items})"
    return f"declare -{flags} {name}={declare_value(var.value)}"


def _assoc_key(k: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.\-+@%]+", k):
        return k
    return declare_value(k)


def _all_vars(sh):
    names = {}
    for scope in [sh.globals] + sh.scopes:
        for k, v in scope.items():
            names[k] = v
    return names


def set_line(name, var) -> str:
    """`set` / `declare` (no args) style line."""
    if isinstance(var.value, dict):
        if "A" in var.attrs:
            items = " ".join(f"[{_assoc_key(k)}]={declare_value(var.value[k])}" for k in assoc_keys(var.value))
            return f"{name}=({items} )"
        items = " ".join(f"[{k}]={declare_value(v)}" for k, v in sorted(var.value.items()))
        return f"{name}=({items})"
    v = var.value
    if v == "":
        return f"{name}="
    from .quoting import ansic_quote, contains_metas, needs_ansic
    if needs_ansic(v):
        return f"{name}={ansic_quote(v)}"
    if contains_metas(v) or "~" in v:
        return f"{name}={single_quote(v)}"
    return f"{name}={v}"


def _declare(sh, args, mode):
    """declare / typeset / local / export / readonly."""
    name0 = args[0]
    a = list(args[1:])
    add, remove = set(), set()
    print_mode = funcs = funcnames = glob = False
    if mode == "export":
        add.add("x")
    if mode == "readonly":
        add.add("r")
    while a and isinstance(a[0], str) and a[0][:1] in "-+" and len(a[0]) > 1:
        opt = a.pop(0)
        if opt == "--":
            break
        on = opt[0] == "-"
        for c in opt[1:]:
            if c == "p":
                print_mode = True
            elif c == "f":
                funcs = True
            elif c == "F":
                funcnames = True
            elif c == "g":
                glob = True
            elif c == "n" and mode == "export":
                remove.add("x") if on else None
                add.discard("x")
            elif c in "aAilnrtux":
                (add if on else remove).add(c)
            else:
                yield from sh.error(f"{name0}: {opt[0]}{c}: invalid option")
                usage = {"export": "export [-fn] [name[=value] ...] or export -p",
                         "readonly": "readonly [-aAf] [name[=value] ...] or readonly -p",
                         "local": "local [option] name[=value] ..."}.get(mode, f"{name0} [-aAfFgiIlnrtux] [name[=value] ...] or {name0} -p [-aAfFilnrtux] [name ...]")
                yield from sh.err(f"{name0}: usage: {usage}\n")
                return 2
    if mode == "local" and not sh.funcnames:
        yield from sh.error("local: can only be used in a function")
        return 1
    local = (mode in ("local",) or (mode in ("declare", "typeset") and sh.funcnames and not glob))
    # functions
    if funcs or funcnames:
        from .unparse import function_text
        names = [x for x in a if isinstance(x, str)] or sorted(sh.funcs)
        status = 0
        for n in names:
            fn = sh.funcs.get(n)
            if fn is None:
                if a:
                    status = 1
                continue
            if mode == "export":
                fn_attrs = getattr(sh, "exported_funcs", set())
                fn_attrs.add(n)
                sh.exported_funcs = fn_attrs
                continue
            if mode == "readonly":
                continue
            if funcnames:
                yield from sh.out(f"declare -f {n}\n" if True else n)
            else:
                yield from sh.out(function_text(fn) + "\n")
        return status
    if not a:
        allv = _all_vars(sh)
        lines = []
        for n in sorted(allv):
            v = sh.lookup(n)
            if v is None:
                continue
            if mode == "export" or "x" in add and mode != "declare":
                if "x" not in v.attrs:
                    continue
                lines.append(declare_line(sh, n, v).replace("declare -", "declare -", 1))
            elif mode == "readonly":
                if "r" not in v.attrs:
                    continue
                lines.append(declare_line(sh, n, v))
            elif print_mode or add:
                if add and not (add <= v.attrs):
                    continue
                lines.append(declare_line(sh, n, v))
            else:
                if v.value is None:
                    continue
                lines.append(set_line(n, v))
        if mode in ("declare", "typeset") and not print_mode and not add:
            from .unparse import function_text
            for n in sorted(sh.funcs):
                lines.append(function_text(sh.funcs[n]))
        yield from sh.out("".join(line + "\n" for line in lines))
        return 0
    if print_mode:
        status = 0
        for n in a:
            v = sh.lookup(n)
            if v is None:
                yield from sh.error(f"{name0}: {n}: not found")
                status = 1
                continue
            yield from sh.out(declare_line(sh, n, v) + "\n")
        return status
    status = 0
    for item in a:
        if not isinstance(item, str):   # name=(...) array literal word
            m = ASSIGN_RE.match(item.raw)
            n = m.group(1)
            append = bool(m.group(3))
            if not _valid(n):
                yield from sh.error(f"{name0}: `{item.raw}': not a valid identifier")
                status = 1
                continue
            cont = sh.scopes[-1] if local and sh.scopes else sh.globals if glob or not local else sh._container(n)
            v = cont.get(n)
            if v is None:
                v = Var()
                cont[n] = v
            if "r" in v.attrs:
                yield from sh.error(f"{n}: readonly variable")
                status = 1
                continue
            for c in add - {"r"}:
                v.attrs.add(c)
            if "A" in add:
                v.attrs.discard("a")
            try:
                yield from sh.assign_array(n, item.array, append, local=local and not glob, assoc="A" in v.attrs)
            except ShellAbort as e:
                status = e.status
                continue
            v = sh.lookup(n)
            if "r" in add:
                v.attrs.add("r")
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?(\+?=)?(.*)\Z", item, re.S)
        if not m or (m.group(3) is None and m.group(4)):
            yield from sh.error(f"{name0}: `{item}': not a valid identifier")
            status = 1
            continue
        n, sub, eq, val = m.group(1), m.group(2), m.group(3), m.group(4)
        if local:
            cont = sh.scopes[-1]
        elif glob:
            cont = sh.globals
        else:
            cont = sh._container(n)
        v = cont.get(n)
        if v is None:
            outer = sh.lookup(n) if local else None
            v = Var()
            if outer is not None and mode == "local" and eq is None:
                pass
            cont[n] = v
            if "allexport" in sh.opts:
                v.attrs.add("x")
        if "r" in v.attrs and (eq is not None or add - {"r", "x"} or remove):
            yield from sh.error(f"{n}: readonly variable")
            status = 1
            continue
        for c in remove:
            v.attrs.discard(c)
        if "a" in add and not isinstance(v.value, dict):
            v.value = {} if v.value is None else {0: v.value}
        if "A" in add:
            if isinstance(v.value, dict) and "a" in v.attrs and v.value:
                yield from sh.error(f"{name0}: {n}: cannot convert indexed to associative array")
                status = 1
                continue
            if not isinstance(v.value, dict):
                v.value = {} if v.value is None else {"0": v.value}
        for c in add - {"r"}:
            v.attrs.add(c)
        if "l" in add:
            v.attrs.discard("u")
        if "u" in add:
            v.attrs.discard("l")
        if eq is not None:
            if val.startswith("(") and val.endswith(")") and ("a" in v.attrs or "A" in v.attrs):
                inner = val[1:-1]
                words = []
                p = Parser(inner)
                try:
                    while p.peek().kind == "word":
                        words.append(p.advance().word)
                except ParseError:
                    words = []
                yield from sh.assign_array(n, words, eq == "+=", local=local, assoc="A" in v.attrs)
            else:
                idx = None
                if sub:
                    idx = sub[1:-1] if "A" in v.attrs else sh.arith(sub[1:-1])
                v.attrs.discard("r")
                try:
                    sh.set_var(n, val, idx, append=eq == "+=", local=local)
                finally:
                    pass
        elif "x" in add and v.value is None and mode == "export":
            pass
        if "r" in add:
            v.attrs.add("r")
    return status


def _valid(name):
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None


@builtin("declare", "typeset")
def b_declare(sh, args):
    return (yield from _declare(sh, args, "declare"))


@builtin("local")
def b_local(sh, args):
    return (yield from _declare(sh, args, "local"))


@builtin("export")
def b_export(sh, args):
    if len(args) == 1 or args[1:] == ["-p"]:
        lines = []
        for n, v in sorted(_all_vars(sh).items()):
            v = sh.lookup(n)
            if v is not None and "x" in v.attrs:
                lines.append(declare_line(sh, n, v).replace("declare --", "declare -x").replace(
                    "declare -" + attr_flags(v), "declare -" + attr_flags(v)))
        yield from sh.out("".join(x + "\n" for x in lines))
        return 0
    return (yield from _declare(sh, args, "export"))


@builtin("readonly")
def b_readonly(sh, args):
    if len(args) == 1 or args[1:] == ["-p"]:
        lines = [declare_line(sh, n, sh.lookup(n)) for n in sorted(_all_vars(sh)) if sh.lookup(n) and "r" in sh.lookup(n).attrs]
        yield from sh.out("".join(x + "\n" for x in lines))
        return 0
    return (yield from _declare(sh, args, "readonly"))


@builtin("unset")
def b_unset(sh, args):
    a = args[1:]
    mode = "v?"
    while a and a[0].startswith("-") and len(a[0]) > 1:
        o = a.pop(0)
        if o == "--":
            break
        if o == "-f":
            mode = "f"
        elif o == "-v":
            mode = "v"
        elif o == "-n":
            mode = "v"
        else:
            yield from sh.error(f"unset: {o}: invalid option")
            yield from sh.err("unset: usage: unset [-f] [-v] [-n] [name ...]\n")
            return 2
    status = 0
    for n in a:
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]", n, re.S)
        if mode == "f":
            sh.funcs.pop(n, None)
            continue
        if m:
            name, sub = m.group(1), m.group(2)
            idx = sub if sh.is_assoc(name) else (sh.arith(sub) if sub not in ("@", "*") else None)
            if sub in ("@", "*"):
                if not sh.unset_var(name):
                    status = 1
            elif not sh.unset_var(name, idx):
                status = 1
            continue
        if not _valid(n):
            yield from sh.error(f"unset: `{n}': not a valid identifier")
            status = 1
            continue
        if sh.lookup(n) is None and mode == "v?" and n in sh.funcs:
            sh.funcs.pop(n)
            continue
        if not sh.unset_var(n):
            status = 1
    return status


@builtin("shift")
def b_shift(sh, args):
    n = 1
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            yield from sh.error(f"shift: {args[1]}: numeric argument required")
            return 1
    if n < 0:
        yield from sh.error(f"shift: {n}: shift count out of range")
        return 1
    if n > len(sh.positional):
        if "shift_verbose" in sh.shopts:
            yield from sh.error(f"shift: {n}: shift count out of range")
        return 1
    sh.positional = sh.positional[n:]
    return 0


@builtin("let")
def b_let(sh, args):
    if len(args) < 2:
        yield from sh.error("let: expression expected")
        return 1
    v = 0
    sh._arith_cmd = True
    sh._arith_ctx = "let: "
    try:
        for e in args[1:]:
            v = sh.arith(e)
    except ShellAbort as e:
        return e.status
    finally:
        sh._arith_cmd = False
        sh._arith_ctx = ""
    return 0 if v != 0 else 1


# ---------------------------------------------------------------------------- read / mapfile

@builtin("read")
def b_read(sh, args):
    a = args[1:]
    raw = silent = False
    arr = None
    delim = "\n"
    prompt = None
    nchars = None
    exact = False
    fd = 0
    timeout = None
    i = 0
    while i < len(a) and a[i].startswith("-") and len(a[i]) > 1:
        opt = a[i]
        if opt == "--":
            i += 1
            break
        j = 1
        while j < len(opt):
            c = opt[j]
            if c in "adpntuie":
                val = opt[j + 1:] if j + 1 < len(opt) else (a[i + 1] if i + 1 < len(a) else None)
                if c in "ie":
                    if c == "e":
                        j += 1
                        continue
                if val is None:
                    yield from sh.error(f"read: -{c}: option requires an argument")
                    yield from sh.err("read: usage: read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] "
                                      "[-p prompt] [-t timeout] [-u fd] [name ...]\n")
                    return 2
                if j + 1 >= len(opt):
                    i += 1
                if c == "a":
                    arr = val
                elif c == "d":
                    delim = val[:1] if val else "\0"
                elif c == "p":
                    prompt = val
                elif c == "n":
                    nchars = int(val)
                elif c == "t":
                    timeout = float(val)
                elif c == "u":
                    fd = int(val)
                break
            if c == "N":
                val = opt[j + 1:] if j + 1 < len(opt) else (a[i + 1] if i + 1 < len(a) else None)
                if j + 1 >= len(opt):
                    i += 1
                nchars = int(val)
                exact = True
                break
            if c == "r":
                raw = True
            elif c == "s":
                silent = True
            else:
                yield from sh.error(f"read: -{c}: invalid option")
                yield from sh.err("read: usage: read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] "
                                  "[-p prompt] [-t timeout] [-u fd] [name ...]\n")
                return 2
            j += 1
        i += 1
    names = a[i:]
    for n in names + ([arr] if arr else []):
        if not _valid(n) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\[.*\]", n):
            yield from sh.error(f"read: `{n}': not a valid identifier")
            return 1
    of = sh.proc.fds.get(fd)
    if of is None:
        yield from sh.error(f"read: {fd}: invalid file descriptor: Bad file descriptor")
        return 1
    if prompt is not None and of.kind == "tty":
        yield from sh.err(prompt)
    k = sh.kernel
    buf = []
    eof = False
    d = delim.encode() if delim != "\0" else b"\0"
    if timeout is not None and timeout == 0:
        return 0
    if nchars is not None:
        while len(buf) < nchars:
            ch = yield from read(k, of, 1)
            if not ch:
                eof = True
                break
            c = ch.decode("utf-8", "surrogateescape")
            if not exact and c == delim:
                break
            if not raw and not exact and c == "\\":
                nxt = yield from read(k, of, 1)
                if nxt and nxt != b"\n":
                    buf.append("\\" + nxt.decode("utf-8", "surrogateescape"))
                continue
            buf.append(c)
        line = "".join(buf)
    else:
        parts = []
        while True:
            data = yield from readline(k, of, d)
            if not data:
                eof = True
                break
            text = data.decode("utf-8", "surrogateescape")
            if text.endswith(delim):
                text = text[:-1]
                ended = True
            else:
                ended = False
                eof = True
            if not raw and text.endswith("\\") and ended and delim == "\n" and not (len(text) >= 2 and _odd_backslashes(text)):
                parts.append(text[:-1])
                continue
            if not raw and ended and _odd_backslashes(text) and delim == "\n":
                parts.append(text[:-1])
                continue
            parts.append(text)
            break
        line = "".join(parts)
    if not raw:
        # keep backslash-escaped characters from splitting: mark them
        pieces = []
        i2 = 0
        while i2 < len(line):
            c = line[i2]
            if c == "\\" and i2 + 1 < len(line):
                pieces.append(("\x00" + line[i2 + 1]))
                i2 += 2
            else:
                pieces.append(c)
                i2 += 1
        marked = pieces
    else:
        marked = list(line)
    ifs = sh.get("IFS")
    if ifs is None:
        ifs = " \t\n"
    if arr is not None:
        fields = _ifs_split(marked, ifs, None)
        sh.unset_var(arr)
        v = Var({i: _unmark(f) for i, f in enumerate(fields)}, "a")
        cont = sh._container(arr)
        if arr in cont and "r" in cont[arr].attrs:
            yield from sh.error(f"{arr}: readonly variable")
            return 1
        cont[arr] = v
    elif not names:
        sh.set_var("REPLY", _unmark(marked))
    else:
        fields = _ifs_split(marked, ifs, len(names))
        for idx, n in enumerate(names):
            val = _unmark(fields[idx]) if idx < len(fields) else ""
            m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\[(.*)\]", n)
            try:
                if m:
                    sh.set_var(m.group(1), val, m.group(2) if sh.is_assoc(m.group(1)) else sh.arith(m.group(2)))
                else:
                    sh.set_var(n, val)
            except ShellAbort as e:
                return e.status
    return 1 if eof else 0


def _odd_backslashes(text):
    n = len(text) - len(text.rstrip("\\"))
    return n % 2 == 1


def _unmark(chars) -> str:
    return "".join(c[1:] if c.startswith("\x00") else c for c in chars)


def _ifs_split(chars, ifs, nfields):
    """Split a list of characters (escaped ones are "\\x00c") by IFS; the last field keeps the rest."""
    ws = set(c for c in ifs if c in " \t\n")
    other = set(c for c in ifs if c not in " \t\n")
    n = len(chars)
    i = 0
    while i < n and chars[i] in ws:
        i += 1
    fields = []
    while i < n:
        if nfields is not None and len(fields) == nfields - 1:
            rest = chars[i:]
            # strip trailing IFS whitespace
            j = len(rest)
            while j > 0 and rest[j - 1] in ws:
                j -= 1
            rest = rest[:j]
            # a single trailing non-whitespace delimiter is removed too
            if rest and rest[-1] in other and len(rest) >= 1:
                k = len(rest) - 1
                prev = rest[:k]
                if not any(c in other for c in prev):
                    rest = prev
                    while rest and rest[-1] in ws:
                        rest = rest[:-1]
            fields.append(rest)
            return fields
        cur = []
        while i < n and chars[i] not in ws and chars[i] not in other:
            cur.append(chars[i])
            i += 1
        fields.append(cur)
        # skip delimiter: whitespace, at most one non-ws delimiter, whitespace
        while i < n and chars[i] in ws:
            i += 1
        if i < n and chars[i] in other:
            i += 1
            while i < n and chars[i] in ws:
                i += 1
            if i >= n and nfields is None:
                pass
    return fields


@builtin("mapfile", "readarray")
def b_mapfile(sh, args):
    a = args[1:]
    strip = False
    delim = "\n"
    count = 0
    origin = 0
    skip = 0
    fd = 0
    clear = True
    i = 0
    while i < len(a) and a[i].startswith("-") and len(a[i]) > 1:
        o = a[i]
        if o == "-t":
            strip = True
        elif o in ("-d", "-n", "-O", "-s", "-u", "-C", "-c"):
            v = a[i + 1] if i + 1 < len(a) else ""
            i += 1
            if o == "-d":
                delim = v[:1] or "\0"
            elif o == "-n":
                count = int(v)
            elif o == "-O":
                origin = int(v)
                clear = False
            elif o == "-s":
                skip = int(v)
            elif o == "-u":
                fd = int(v)
        else:
            yield from sh.error(f"{args[0]}: {o}: invalid option")
            return 2
        i += 1
    name = a[i] if i < len(a) else "MAPFILE"
    of = sh.proc.fds.get(fd)
    lines = []
    d = delim.encode() if delim != "\0" else b"\0"
    n = 0
    while True:
        if count and len(lines) >= count:
            break
        data = yield from readline(sh.kernel, of, d)
        if not data:
            break
        n += 1
        if n <= skip:
            continue
        text = data.decode("utf-8", "surrogateescape")
        if strip and text.endswith(delim):
            text = text[:-1]
        lines.append(text)
    cont = sh._container(name)
    v = cont.get(name)
    if v is None or clear:
        v = Var({}, "a") if v is None else v
        v.value = {} if clear or not isinstance(v.value, dict) else v.value
        v.attrs.add("a")
        cont[name] = v
    for k, line in enumerate(lines):
        v.value[origin + k] = line
    return 0


# ---------------------------------------------------------------------------- options

SET_OPTIONS = ["allexport", "braceexpand", "emacs", "errexit", "errtrace", "functrace", "hashall", "histexpand",
               "history", "ignoreeof", "interactive-comments", "keyword", "monitor", "noclobber", "noexec", "noglob",
               "nolog", "notify", "nounset", "onecmd", "physical", "pipefail", "posix", "privileged", "verbose", "vi",
               "xtrace"]
SET_LETTERS = {"a": "allexport", "b": "notify", "e": "errexit", "f": "noglob", "h": "hashall", "k": "keyword",
               "m": "monitor", "n": "noexec", "p": "privileged", "t": "onecmd", "u": "nounset", "v": "verbose",
               "x": "xtrace", "B": "braceexpand", "C": "noclobber", "E": "errtrace", "H": "histexpand",
               "P": "physical", "T": "functrace"}
SHOPT_OPTIONS = ["assoc_expand_once", "autocd", "cdable_vars", "cdspell", "checkhash", "checkjobs", "checkwinsize",
                 "cmdhist", "compat31", "compat32", "compat40", "compat41", "compat42", "compat43", "compat44",
                 "complete_fullquote", "direxpand", "dirspell", "dotglob", "execfail", "expand_aliases", "extdebug",
                 "extglob", "extquote", "failglob", "force_fignore", "globasciiranges", "globskipdots", "globstar",
                 "gnu_errfmt", "histappend", "histreedit", "histverify", "hostcomplete", "huponexit",
                 "inherit_errexit", "interactive_comments", "lastpipe", "lithist", "localvar_inherit",
                 "localvar_unset", "login_shell", "mailwarn", "no_empty_cmd_completion", "nocaseglob", "nocasematch",
                 "noexpand_translation", "nullglob", "patsub_replacement", "progcomp", "progcomp_alias", "promptvars",
                 "restricted_shell", "shift_verbose", "sourcepath", "varredir_close", "xpg_echo"]


@builtin("set")
def b_set(sh, args):
    a = args[1:]
    if not a:
        lines = [set_line(n, sh.lookup(n)) for n in sorted(_all_vars(sh)) if sh.lookup(n) and sh.lookup(n).value is not None]
        from .unparse import function_text
        lines += [function_text(sh.funcs[n]) for n in sorted(sh.funcs)]
        yield from sh.out("".join(x + "\n" for x in lines))
        return 0
    i = 0
    while i < len(a):
        arg = a[i]
        if arg == "--":
            sh.positional = a[i + 1:]
            return 0
        if arg == "-":
            sh.opts -= {"xtrace", "verbose"}
            sh.positional = a[i + 1:] if i + 1 < len(a) else sh.positional
            return 0
        if arg[:1] in "-+" and len(arg) > 1:
            on = arg[0] == "-"
            if arg[1:] == "o":
                if i + 1 >= len(a):
                    if on:
                        yield from sh.out("".join(f"{o:<15}\t{'on' if o in sh.opts else 'off'}\n" for o in SET_OPTIONS))
                    else:
                        yield from sh.out("".join(f"set {'-' if o in sh.opts else '+'}o {o}\n" for o in SET_OPTIONS))
                    return 0
                name = a[i + 1]
                if name not in SET_OPTIONS:
                    yield from sh.error(f"set: {name}: invalid option name")
                    return 1
                _setopt(sh, name, on)
                i += 2
                continue
            for c in arg[1:]:
                if c in SET_LETTERS:
                    _setopt(sh, SET_LETTERS[c], on)
                elif c == "o":
                    pass
                else:
                    yield from sh.error(f"set: {arg[0]}{c}: invalid option")
                    yield from sh.err("set: usage: set [-abefhkmnptuvxBCEHPT] [-o option-name] [--] [-] [arg ...]\n")
                    return 2
            i += 1
            continue
        sh.positional = a[i:]
        return 0
    return 0


def _setopt(sh, name, on):
    if on:
        sh.opts.add(name)
        if name == "vi":
            sh.opts.discard("emacs")
        if name == "emacs":
            sh.opts.discard("vi")
    else:
        sh.opts.discard(name)


@builtin("shopt")
def b_shopt(sh, args):
    a = args[1:]
    mode = None
    quiet = printable = setonly = False
    while a and a[0].startswith("-") and len(a[0]) > 1:
        for c in a.pop(0)[1:]:
            if c in "su":
                mode = c
            elif c == "q":
                quiet = True
            elif c == "p":
                printable = True
            elif c == "o":
                setonly = True
            else:
                yield from sh.error(f"shopt: -{c}: invalid option")
                yield from sh.err("shopt: usage: shopt [-pqsu] [-o] [optname ...]\n")
                return 2
    names = SET_OPTIONS if setonly else SHOPT_OPTIONS
    have = sh.opts if setonly else sh.shopts
    if not a:
        rows = [n for n in names if (mode is None) or (mode == "s") == (n in have)]
        if quiet:
            return 0
        if printable or mode:
            yield from sh.out("".join(f"{'set' if setonly else 'shopt'} {'-s' if n in have else '-u'}{'o' if setonly else ''} {n}\n"
                                      if not setonly else f"set {'-' if n in have else '+'}o {n}\n" for n in rows))
        else:
            yield from sh.out("".join(f"{n:<15}\t{'on' if n in have else 'off'}\n" for n in rows))
        return 0
    status = 0
    for n in a:
        if n not in names:
            yield from sh.error(f"shopt: {n}: invalid shell option name")
            status = 1
            continue
        if mode == "s":
            have.add(n)
        elif mode == "u":
            have.discard(n)
        else:
            if n not in have:
                status = 1
            if not quiet:
                if printable:
                    yield from sh.out(f"shopt {'-s' if n in have else '-u'} {n}\n")
                else:
                    yield from sh.out(f"{n:<15}\t{'on' if n in have else 'off'}\n")
    return status


# ---------------------------------------------------------------------------- source / eval / exec / exit

@builtin("source", ".")
def b_source(sh, args):
    if len(args) < 2:
        yield from sh.error(f"{args[0]}: filename argument required")
        yield from sh.err(f"{args[0]}: usage: {args[0]} filename [arguments]\n")
        return 2
    name = args[1]
    path = name
    if "/" not in name and "sourcepath" in sh.shopts:
        for d in (sh.get("PATH", "")).split(":"):
            cand = (d or ".").rstrip("/") + "/" + name
            try:
                if sh.fs.lookup(cand, sh.proc.cwd).type == "f":
                    path = cand
                    break
            except FSError:
                continue
    try:
        of = open_path(sh.proc, path, "r")
        if of.kind == "dir":
            yield from sh.error(f"{args[0]}: {name}: is a directory")
            return 1
        from .fileio import read_all
        data = yield from read_all(sh.kernel, of)
    except FSError as e:
        yield from sh.error(f"{name}: {e.msg}")
        return 1
    saved_pos = None
    if len(args) > 2:
        saved_pos = sh.positional
        sh.positional = args[2:]
    sh.sources.append(name)
    saved_script, saved_line = sh.script, sh.lineno
    if not sh.interactive:
        sh.script = name
    sh.source_depth = getattr(sh, "source_depth", 0) + 1
    try:
        status = yield from sh.run_script_text(data.decode("utf-8", "surrogateescape"))
    except ReturnFunc as r:
        status = r.status
    finally:
        sh.source_depth -= 1
        sh.sources.pop()
        sh.script, sh.lineno = saved_script, saved_line
        if saved_pos is not None:
            sh.positional = saved_pos
    return status


@builtin("eval")
def b_eval(sh, args):
    src = " ".join(args[1:])
    if not src.strip():
        return 0
    p = Parser(src, sh.aliases if "expand_aliases" in sh.shopts else None, line=sh.lineno or 1)
    status = 0
    while True:
        try:
            cmd = p.parse_next()
        except ParseError as e:
            yield from sh.error(e.msg)
            return 2
        if cmd is None:
            return status
        status = yield from sh.run_list(cmd)


@builtin("exec")
def b_exec(sh, args):
    a = args[1:]
    while a and a[0].startswith("-") and a[0] in ("-c", "-l", "-a"):
        a = a[1:] if a[0] != "-a" else a[2:]
    if not a:
        return 0
    path = sh.find_command(a[0])
    if path is None:
        yield from sh.error(f"exec: {a[0]}: not found")
        if not sh.interactive:
            raise ExitShell(127)
        return 127
    from .exec import exec_program
    sh.proc.argv = list(a)
    st = yield from exec_program(sh.proc, path, a, sh.env(), sh)
    raise ExitShell(st)


@builtin("exit", "logout")
def b_exit(sh, args):
    if args[0] == "logout" and sh.interactive and False:
        pass
    if len(args) > 2:
        yield from sh.error(f"{args[0]}: too many arguments")
        return 1
    status = sh.status
    if len(args) == 2:
        try:
            status = int(args[1]) & 0xFF
        except ValueError:
            yield from sh.error(f"{args[0]}: {args[1]}: numeric argument required")
            status = 2
    if sh.interactive:
        yield from sh.err("exit\n" if args[0] == "exit" else "logout\n")
    raise ExitShell(status)


@builtin("return")
def b_return(sh, args):
    if not sh.funcnames and not getattr(sh, "source_depth", 0):
        yield from sh.error("return: can only `return' from a function or sourced script")
        return 2
    status = sh.status
    if len(args) > 1:
        try:
            status = int(args[1]) & 0xFF
        except ValueError:
            yield from sh.error(f"return: {args[1]}: numeric argument required")
            status = 2
    raise ReturnFunc(status)


def _loop_arg(sh, args, name):
    n = 1
    if len(args) > 1:
        try:
            n = int(args[1])
        except ValueError:
            yield from sh.error(f"{name}: {args[1]}: numeric argument required")
            return None
        if n < 1:
            yield from sh.error(f"{name}: {n}: loop count out of range")
            return None
    return n


@builtin("break")
def b_break(sh, args):
    n = yield from _loop_arg(sh, args, "break")
    if n is None:
        return 1
    if sh.loop_depth == 0:
        yield from sh.error("break: only meaningful in a `for', `while', or `until' loop")
        return 0
    raise BreakLoop(min(n, sh.loop_depth))


@builtin("continue")
def b_continue(sh, args):
    n = yield from _loop_arg(sh, args, "continue")
    if n is None:
        return 1
    if sh.loop_depth == 0:
        yield from sh.error("continue: only meaningful in a `for', `while', or `until' loop")
        return 0
    raise ContinueLoop(min(n, sh.loop_depth))


# ---------------------------------------------------------------------------- test / [

def unary_test(sh, op, s) -> bool:
    if op == "-z":
        return s == ""
    if op == "-n":
        return s != ""
    if op == "-v":
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(.*)\])?", s)
        if not m:
            return False
        v = sh.lookup(m.group(1))
        if m.group(2) is None:
            return sh.is_set(m.group(1)) and (v is None or v.value is not None)
        if v is None or not isinstance(v.value, dict):
            return False
        key = m.group(2)
        if key in ("@", "*"):
            return bool(v.value)
        return (key if "A" in v.attrs else sh.arith(key)) in v.value
    if op == "-o":
        return s in sh.opts
    if op == "-t":
        try:
            of = sh.proc.fds.get(int(s))
        except ValueError:
            return False
        return of is not None and of.kind == "tty"
    if op == "-R":
        return False
    fs, proc = sh.fs, sh.proc
    if s == "":
        return False
    if s.startswith("/dev/fd/") or s in ("/dev/stdin", "/dev/stdout", "/dev/stderr"):
        return op in ("-e", "-r", "-w", "-c", "-p") if op != "-f" else False
    if op in ("-L", "-h"):
        try:
            return fs.lookup(s, proc.cwd, follow=False).type == "l"
        except FSError:
            return False
    try:
        node = fs.lookup(s, proc.cwd, True, proc.uid, proc.gids)
    except FSError:
        return False
    if op in ("-e", "-a"):
        return True
    if op == "-f":
        return node.type == "f"
    if op == "-d":
        return node.type == "d"
    if op == "-s":
        return node.size > 0 if node.type != "d" else True
    if op == "-r":
        return fs.can(node, 4, proc.uid, proc.gids)
    if op == "-w":
        return fs.can(node, 2, proc.uid, proc.gids)
    if op == "-x":
        return fs.can(node, 1, proc.uid, proc.gids) and (node.type == "d" or bool(node.mode & 0o111))
    if op == "-p":
        return node.type == "p"
    if op == "-S":
        return node.type == "s"
    if op == "-b":
        return False
    if op == "-c":
        return node.type == "c"
    if op == "-g":
        return bool(node.mode & 0o2000)
    if op == "-u":
        return bool(node.mode & 0o4000)
    if op == "-k":
        return bool(node.mode & 0o1000)
    if op == "-O":
        return node.uid == proc.uid
    if op == "-G":
        return node.gid == proc.gid
    if op == "-N":
        return node.mtime > node.atime
    return False


class TestError(Exception):
    pass


def _int_operand(sh, s, cond):
    if cond:
        return sh.arith(s) if s.strip() else 0
    t = s.strip()
    if re.fullmatch(r"[+-]?\d+", t):
        return int(t)
    raise TestError(f"{s}: integer expression expected")


def binary_test(sh, op, a, b, cond=False) -> bool:
    if op in ("=", "=="):
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a.encode() < b.encode()
    if op == ">":
        return a.encode() > b.encode()
    if op in ("-eq", "-ne", "-lt", "-le", "-gt", "-ge"):
        x, y = _int_operand(sh, a, cond), _int_operand(sh, b, cond)
        return {"-eq": x == y, "-ne": x != y, "-lt": x < y, "-le": x <= y, "-gt": x > y, "-ge": x >= y}[op]
    fs, proc = sh.fs, sh.proc
    try:
        na = fs.lookup(a, proc.cwd)
    except FSError:
        na = None
    try:
        nb = fs.lookup(b, proc.cwd)
    except FSError:
        nb = None
    if op == "-nt":
        return na is not None and (nb is None or na.mtime > nb.mtime)
    if op == "-ot":
        return nb is not None and (na is None or na.mtime < nb.mtime)
    if op == "-ef":
        return na is not None and na is nb
    raise TestError(f"{op}: binary operator expected")


UNARY = {"-a", "-b", "-c", "-d", "-e", "-f", "-g", "-h", "-k", "-p", "-r", "-s", "-t", "-u", "-w", "-x", "-G", "-L",
         "-N", "-O", "-S", "-z", "-n", "-o", "-v", "-R"}
BINARY = {"=", "==", "!=", "<", ">", "-eq", "-ne", "-lt", "-le", "-gt", "-ge", "-nt", "-ot", "-ef"}


def _test_expr(sh, a):
    """POSIX test with bash's argument-count rules."""
    n = len(a)
    if n == 0:
        return False
    if n == 1:
        return a[0] != ""
    if n == 2:
        if a[0] == "!":
            return a[1] == ""
        if a[0] in UNARY:
            return unary_test(sh, a[0], a[1])
        raise TestError(f"{a[0]}: unary operator expected")
    if n == 3:
        if a[1] in BINARY:
            return binary_test(sh, a[1], a[0], a[2])
        if a[0] == "!":
            return not _test_expr(sh, a[1:])
        if a[0] == "(" and a[2] == ")":
            return a[1] != ""
        if a[1] == "-a":
            return a[0] != "" and a[2] != ""
        if a[1] == "-o":
            return a[0] != "" or a[2] != ""
        raise TestError(f"{a[1]}: binary operator expected")
    if n == 4:
        if a[0] == "!":
            return not _test_expr(sh, a[1:])
        if a[0] == "(" and a[3] == ")":
            return _test_expr(sh, a[1:3])
    return _test_full(sh, a)


def _test_full(sh, a):
    pos = [0]

    def peek():
        return a[pos[0]] if pos[0] < len(a) else None

    def take():
        pos[0] += 1
        return a[pos[0] - 1]

    def or_():
        v = and_()
        while peek() == "-o":
            take()
            r = and_()
            v = v or r
        return v

    def and_():
        v = not_()
        while peek() == "-a":
            take()
            r = not_()
            v = v and r
        return v

    def not_():
        if peek() == "!":
            take()
            return not not_()
        return prim()

    def prim():
        t = peek()
        if t is None:
            raise TestError("argument expected")
        if t == "(":
            take()
            v = or_()
            if peek() != ")":
                raise TestError("`)' expected")
            take()
            return v
        if pos[0] + 2 < len(a) + 1 and pos[0] + 1 < len(a) and a[pos[0] + 1] in BINARY and pos[0] + 2 < len(a):
            x = take()
            op = take()
            y = take()
            return binary_test(sh, op, x, y)
        if t in UNARY and pos[0] + 1 < len(a):
            take()
            return unary_test(sh, t, take())
        take()
        return t != ""
    v = or_()
    if pos[0] != len(a):
        raise TestError("too many arguments")
    return v


@builtin("test", "[")
def b_test(sh, args):
    a = list(args[1:])
    name = args[0]
    if name == "[":
        if not a or a[-1] != "]":
            yield from sh.error("[: missing `]'")
            return 2
        a = a[:-1]
    try:
        return 0 if _test_expr(sh, a) else 1
    except TestError as e:
        yield from sh.error(f"{name}: {e}")
        return 2


# ---------------------------------------------------------------------------- getopts

@builtin("getopts")
def b_getopts(sh, args):
    if len(args) < 3:
        yield from sh.error("getopts: usage: getopts optstring name [arg ...]")
        return 2
    optstring, name = args[1], args[2]
    argv = args[3:] if len(args) > 3 else list(sh.positional)
    silent = optstring.startswith(":")
    if silent:
        optstring = optstring[1:]
    try:
        optind = int(sh.get("OPTIND", "1"))
    except ValueError:
        optind = 1
    if optind < 1:
        optind = 1
    if getattr(sh, "_optind_seen", None) != optind:
        sh.getopts_pos = 0
    pos = sh.getopts_pos
    if optind - 1 >= len(argv):
        sh.set_var(name, "?")
        sh.unset_var("OPTARG")
        return 1
    arg = argv[optind - 1]
    if pos == 0:
        if arg == "--":
            sh.set_var("OPTIND", str(optind + 1))
            sh._optind_seen = optind + 1
            sh.set_var(name, "?")
            sh.unset_var("OPTARG")
            return 1
        if not arg.startswith("-") or arg == "-":
            sh.set_var(name, "?")
            sh.unset_var("OPTARG")
            return 1
        pos = 1
    c = arg[pos]
    pos += 1
    nxt_optind = optind
    if pos >= len(arg):
        nxt_optind = optind + 1
        pos = 0
    idx = optstring.find(c)
    prog = sh.script or sh.arg0
    if idx < 0 or c == ":":
        if silent:
            sh.set_var("OPTARG", c)
        else:
            sh.unset_var("OPTARG")
            if sh.get("OPTERR", "1") != "0":
                yield from sh.err(f"{prog}: illegal option -- {c}\n")
        sh.set_var(name, "?")
    elif idx + 1 < len(optstring) and optstring[idx + 1] == ":":
        if pos:   # rest of this argument is the value
            sh.set_var("OPTARG", arg[pos:])
            nxt_optind = optind + 1
            pos = 0
            sh.set_var(name, c)
        elif nxt_optind - 1 < len(argv):
            sh.set_var("OPTARG", argv[nxt_optind - 1])
            nxt_optind += 1
            sh.set_var(name, c)
        else:
            if silent:
                sh.set_var(name, ":")
                sh.set_var("OPTARG", c)
            else:
                sh.set_var(name, "?")
                sh.unset_var("OPTARG")
                if sh.get("OPTERR", "1") != "0":
                    yield from sh.err(f"{prog}: option requires an argument -- {c}\n")
    else:
        sh.set_var(name, c)
        sh.unset_var("OPTARG")
    sh.getopts_pos = pos
    sh.set_var("OPTIND", str(nxt_optind))
    sh._optind_seen = nxt_optind
    return 0


# ---------------------------------------------------------------------------- traps & signals

def parse_signal(spec: str):
    s = spec.upper()
    if s.isdigit():
        n = int(s)
        return n if n == 0 or n in SIGNALS else None
    if s.startswith("SIG"):
        s = s[3:]
    if s in ("EXIT",):
        return 0
    return SIGNUM.get(s)


def signal_listing() -> str:
    items = [(n, "SIG" + name) for n, name in sorted(SIGNALS.items())]
    rt = [(34 + i, f"SIGRTMIN+{i}" if i else "SIGRTMIN") for i in range(16)] + \
         [(50 + i, f"SIGRTMAX-{14 - i}" if i < 14 else "SIGRTMAX") for i in range(15)]
    items += rt
    lines = []
    for i in range(0, len(items), 5):
        row = items[i:i + 5]
        lines.append("\t".join(f"{n:2d}) {name}" for n, name in row))
    return "\n".join(lines) + "\n"


@builtin("trap")
def b_trap(sh, args):
    a = args[1:]
    if a and a[0] == "--":
        a = a[1:]
    if not a or a == ["-p"]:
        lines = []
        for name in sorted(sh.traps, key=lambda n: (0 if n == "EXIT" else SIGNUM.get(n, 100), n)):
            lines.append(f"trap -- {single_quote(sh.traps[name])} {name if name in ('EXIT', 'ERR', 'DEBUG', 'RETURN') else 'SIG' + name}\n")
        yield from sh.out("".join(lines))
        return 0
    if a[0] == "-l":
        yield from sh.out(signal_listing())
        return 0
    if a[0] == "-p":
        lines = []
        for spec in a[1:]:
            sig = parse_signal(spec)
            name = "EXIT" if sig == 0 else SIGNALS.get(sig, spec.upper())
            if name in sh.traps:
                lines.append(f"trap -- {single_quote(sh.traps[name])} {name if name == 'EXIT' else 'SIG' + name}\n")
        yield from sh.out("".join(lines))
        return 0
    action = a[0]
    specs = a[1:]
    if not specs:
        # `trap 5` style: reset
        specs = [action]
        action = "-"
    if re.fullmatch(r"\d+", action) and specs:
        pass
    status = 0
    for spec in specs:
        up = spec.upper()
        if up in ("ERR", "DEBUG", "RETURN"):
            name = up
        else:
            sig = parse_signal(spec)
            if sig is None:
                yield from sh.error(f"trap: {spec}: invalid signal specification")
                status = 1
                continue
            name = "EXIT" if sig == 0 else SIGNALS[sig]
        if action == "-":
            sh.traps.pop(name, None)
            if name in SIGNUM:
                sh.ignored_sigs.discard(SIGNUM[name])
                sh.proc.ignored.discard(SIGNUM[name])
        else:
            sh.traps[name] = action
            if action == "" and name in SIGNUM:
                sh.ignored_sigs.add(SIGNUM[name])
                sh.proc.ignored.add(SIGNUM[name])
    return status


# ---------------------------------------------------------------------------- kill / jobs

def find_job(sh, spec: str):
    """%1 %+ %% %- %name %?text -> Job or None."""
    jobs = sh.jobs
    if not jobs:
        return None
    order = [n for n in sh.job_order if n in jobs]
    s = spec[1:] if spec.startswith("%") else spec
    if s in ("", "+", "%"):
        return jobs[order[-1]] if order else None
    if s == "-":
        return jobs[order[-2]] if len(order) > 1 else (jobs[order[-1]] if order else None)
    if s.isdigit():
        return jobs.get(int(s))
    if s.startswith("?"):
        cands = [j for j in jobs.values() if s[1:] in j.text]
    else:
        cands = [j for j in jobs.values() if j.text.startswith(s)]
    return cands[-1] if cands else None


def current_marks(sh):
    order = [n for n in sh.job_order if n in sh.jobs]
    cur = order[-1] if order else None
    prev = order[-2] if len(order) > 1 else None
    return cur, prev


def job_line(sh, job, long=False, pids_only=False):
    cur, prev = current_marks(sh)
    mark = "+" if job.num == cur else ("-" if job.num == prev else " ")
    st = job.status_text()
    text = job.text
    if st != "Running" and text.endswith(" &"):
        text = text[:-2]
    if long:
        return f"[{job.num}]{mark} {job.procs[0].pid} {st:<24}{text}"
    return f"[{job.num}]{mark}  {st:<24}{text}"


def notify_jobs(sh):
    """Report finished/stopped jobs before the next prompt (interactive shells)."""
    out = []
    for num in sorted(sh.jobs):
        job = sh.jobs[num]
        st = job.state()
        if st == "Done":
            if job.report_at is not None and job.report_at > sh.prompt_no:
                continue
            out.append(job_line(sh, job))
            del sh.jobs[num]
        elif st == "Stopped" and job.reported_state != "Stopped":
            job.reported_state = "Stopped"
            out.append(job_line(sh, job))
        elif st == "Running":
            job.reported_state = "Running"
    sh.job_order = [n for n in sh.job_order if n in sh.jobs]
    return "".join(line + "\n" for line in out)


@builtin("jobs")
def b_jobs(sh, args):
    a = args[1:]
    long = "-l" in a
    pids = "-p" in a
    running = "-r" in a
    stopped = "-s" in a
    specs = [x for x in a if not x.startswith("-")]
    jobs = [find_job(sh, s) for s in specs] if specs else [sh.jobs[n] for n in sorted(sh.jobs)]
    lines = []
    for spec, job in zip(specs or [None] * len(jobs), jobs):
        if job is None:
            yield from sh.error(f"jobs: {spec}: no such job")
            continue
        st = job.state()
        if running and st != "Running" or stopped and st != "Stopped":
            continue
        if pids:
            lines.append(str(job.procs[0].pid))
        else:
            lines.append(job_line(sh, job, long))
        if st == "Done":
            job.notified = True
    yield from sh.out("".join(x + "\n" for x in lines))
    for n in [n for n, j in sh.jobs.items() if j.state() == "Done" and j.notified]:
        del sh.jobs[n]
    return 0


@builtin("fg")
def b_fg(sh, args):
    if "monitor" not in sh.opts:
        yield from sh.error("fg: no job control")
        return 1
    spec = args[1] if len(args) > 1 else "%+"
    job = find_job(sh, spec)
    if job is None:
        yield from sh.error(f"fg: {spec if len(args) > 1 else 'current'}: no such job")
        return 1
    text = job.text[:-2] if job.text.endswith(" &") else job.text
    yield from sh.out(text + "\n")
    for p in job.procs:
        sh.kernel.signal(p, SIGCONT)
    job.text = text
    sh.job_order = [n for n in sh.job_order if n != job.num] + [job.num]
    tty = sh.world.tty_of(sh.proc)
    old = tty.fg_pgid if tty else None
    if tty:
        tty.fg_pgid = job.pgid
    yield from sh.wait_for(job.procs)
    if tty:
        tty.fg_pgid = old
    if job.state() == "Stopped":
        yield from sh.err(f"\n{job_line(sh, job)}\n")
        job.reported_state = "Stopped"
        return 128 + 20
    del sh.jobs[job.num]
    sh.job_order = [n for n in sh.job_order if n in sh.jobs]
    last = job.procs[-1]
    if last.term_sig and last.term_sig != 2:
        yield from sh.err(SIG_DESC.get(last.term_sig, "Killed") + "\n")
    return last.status


@builtin("bg")
def b_bg(sh, args):
    if "monitor" not in sh.opts:
        yield from sh.error("bg: no job control")
        return 1
    specs = args[1:] or ["%+"]
    status = 0
    for spec in specs:
        job = find_job(sh, spec)
        if job is None:
            yield from sh.error(f"bg: {spec if args[1:] else 'current'}: no such job")
            status = 1
            continue
        if job.state() == "Running":
            yield from sh.error(f"bg: job {job.num} already in background")
            continue
        if not job.text.endswith("&"):
            job.text += " &"
        cur, _ = current_marks(sh)
        mark = "+" if job.num == cur else "-"
        yield from sh.out(f"[{job.num}]{mark} {job.text}\n")
        job.reported_state = "Running"
        for p in job.procs:
            sh.kernel.signal(p, SIGCONT)
    return status


def _report_killed(sh):
    """An interactive `wait` reports a job killed by a signal as soon as it reaps it."""
    if not sh.interactive:
        return
    for num in sorted(sh.jobs):
        job = sh.jobs[num]
        if job.state() == "Done" and job.died_in_wait and job.procs[-1].term_sig not in (0, 2, 13):
            yield from sh.err(job_line(sh, job) + "\n")
            del sh.jobs[num]
    sh.job_order = [n for n in sh.job_order if n in sh.jobs]


@builtin("wait")
def b_wait(sh, args):
    a = args[1:]
    any_one = False
    if a and a[0] == "-n":
        any_one = True
        a = a[1:]
    if a and a[0] == "-f":
        a = a[1:]
    sh._in_wait_builtin = True
    for job in sh.jobs.values():
        if job.state() == "Done" and job.died_line is not None and job.died_line == sh.prompt_no:
            job.died_in_wait, job.report_at = True, sh.prompt_no + 1
    try:
        if not a:
            procs = [p for j in sh.jobs.values() for p in j.procs]
            kids = [p for p in sh.host.procs.values() if p.parent is sh.proc] + procs
            if any_one:
                live = [p for p in procs if p.state != "Z"]
                if not live:
                    return 127
                while all(p.state != "Z" for p in live):
                    if sh.pending_traps:
                        return 128 + sh.pending_traps[0]
                    yield ("wait", live)
                done = next(p for p in live if p.state == "Z")
                if done.job is not None and done.job.num in sh.jobs and done.job.state() == "Done" and \
                        not (done.job.died_in_wait and sh.interactive):
                    del sh.jobs[done.job.num]
                return done.status
            while any(p.state not in ("Z",) for p in set(kids)):
                live = [p for p in set(kids) if p.state not in ("Z",)]
                if all(p.state == "T" for p in live):
                    break
                if sh.pending_traps:
                    return 128 + sh.pending_traps[0]
                yield ("wait", live)
            yield from _report_killed(sh)
            for n in [n for n, j in sh.jobs.items() if j.state() == "Done" and not (j.died_in_wait and sh.interactive)]:
                del sh.jobs[n]
            sh.job_order = [n for n in sh.job_order if n in sh.jobs]
            return 0
        status = 0
        for spec in a:
            if spec.startswith("%"):
                job = find_job(sh, spec)
                if job is None:
                    yield from sh.error(f"wait: {spec}: no such job")
                    status = 127
                    continue
                procs = job.procs
            else:
                try:
                    pid = int(spec)
                except ValueError:
                    yield from sh.error(f"wait: `{spec}': not a pid or valid job spec")
                    status = 2
                    continue
                p = next((p for j in sh.jobs.values() for p in j.procs if p.pid == pid), None)
                if p is None:
                    p = getattr(sh, "reaped", {}).get(pid)
                    if p is None:
                        yield from sh.error(f"wait: pid {pid} is not a child of this shell")
                        status = 127
                        continue
                procs = [p]
                job = p.job
            while any(p.state not in ("Z", "T") for p in procs):
                if sh.pending_traps:
                    return 128 + sh.pending_traps[0]
                yield ("wait", procs)
            yield from _report_killed(sh)
            status = procs[-1].status
            if job is not None and job.num in sh.jobs and job.state() == "Done" and not (job.died_in_wait and sh.interactive):
                del sh.jobs[job.num]
        sh.job_order = [n for n in sh.job_order if n in sh.jobs]
        return status
    finally:
        sh._in_wait_builtin = False


@builtin("disown")
def b_disown(sh, args):
    a = args[1:]
    h = "-h" in a
    all_ = "-a" in a
    running = "-r" in a
    specs = [x for x in a if not x.startswith("-")]
    if all_ or running:
        jobs = [j for j in sh.jobs.values() if not running or j.state() == "Running"]
    elif specs:
        jobs = []
        for s in specs:
            j = find_job(sh, s)
            if j is None:
                yield from sh.error(f"disown: {s}: no such job")
                return 1
            jobs.append(j)
    else:
        j = find_job(sh, "%+")
        if j is None:
            yield from sh.error("disown: current: no such job")
            return 1
        jobs = [j]
    for j in jobs:
        if h:
            j.nohup = True
            for p in j.procs:
                p.ignored.add(SIGHUP)
        else:
            sh.jobs.pop(j.num, None)
            for p in j.procs:
                p.job = None
    sh.job_order = [n for n in sh.job_order if n in sh.jobs]
    return 0


def kill_targets(sh, spec):
    """-> list of procs, or an error string."""
    if spec.startswith("%"):
        job = find_job(sh, spec)
        if job is None:
            return f"{spec}: no such job"
        return [p for p in job.procs if p.state != "Z"] or f"{spec}: no such job"
    try:
        pid = int(spec)
    except ValueError:
        return f"{spec}: arguments must be process or job IDs"
    host = sh.host
    if pid < 0 or (pid == 0):
        pg = -pid if pid < 0 else sh.proc.pgid
        procs = [p for p in host.procs.values() if p.pgid == pg]
        return procs or f"({pid}) - No such process"
    p = host.procs.get(pid)
    if p is None or p.state == "Z":
        return f"({pid}) - No such process"
    return [p]


@builtin("kill")
def b_kill(sh, args):
    a = args[1:]
    sig = SIGTERM
    if not a:
        yield from sh.err("kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]\n")
        return 2
    if a[0] in ("-l", "-L"):
        if len(a) == 1:
            yield from sh.out(signal_listing())
            return 0
        out = []
        for x in a[1:]:
            if x.isdigit():
                n = int(x)
                if n > 128:
                    n -= 128
                out.append(SIGNALS.get(n, "?"))
            else:
                n = parse_signal(x)
                out.append(str(n) if n is not None else "?")
        yield from sh.out("".join(o + "\n" for o in out))
        return 0
    if a[0] in ("-s", "-n"):
        if len(a) < 2:
            yield from sh.error(f"kill: {a[0]}: option requires an argument")
            return 2
        s = parse_signal(a[1])
        if s is None:
            yield from sh.error(f"kill: {a[1]}: invalid signal specification")
            return 1
        sig = s
        a = a[2:]
    elif a[0].startswith("-") and len(a[0]) > 1 and a[0] != "--":
        s = parse_signal(a[0][1:])
        if s is None:
            yield from sh.error(f"kill: {a[0][1:]}: invalid signal specification")
            return 1
        sig = s
        a = a[1:]
    if a and a[0] == "--":
        a = a[1:]
    if not a:
        yield from sh.err("kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]\n")
        return 2
    status = 0
    for spec in a:
        targets = kill_targets(sh, spec)
        if isinstance(targets, str):
            yield from sh.error(f"kill: {targets}")
            status = 1
            continue
        for p in targets:
            if sh.proc.uid != 0 and p.uid != sh.proc.uid:
                yield from sh.error(f"kill: ({p.pid}) - Operation not permitted")
                status = 1
                continue
            if sig in (SIGTERM, SIGHUP, 2) and p.state == "T":
                sh.kernel.signal(p, sig)
                sh.kernel.signal(p, SIGCONT)
            else:
                sh.kernel.signal(p, sig)
    if sig not in (0,):
        yield ("yield",)   # let the signalled processes react before the next command
    return status


# ---------------------------------------------------------------------------- command lookup

def _keyword(name):
    return name in RESERVED


@builtin("type")
def b_type(sh, args):
    a = args[1:]
    flags = set()
    while a and a[0].startswith("-") and len(a[0]) > 1:
        for c in a.pop(0)[1:]:
            flags.add(c)
    status = 0
    from .unparse import function_text
    for name in a:
        found = False
        out = []
        if "f" not in flags and name in sh.aliases and "P" not in flags:
            found = True
            out.append(("alias", f"{name} is aliased to `{sh.aliases[name]}'"))
        if _keyword(name) and "P" not in flags:
            found = True
            out.append(("keyword", f"{name} is a shell keyword"))
        if name in sh.funcs and "f" not in flags and "P" not in flags:
            found = True
            out.append(("function", f"{name} is a function\n{function_text(sh.funcs[name])}"))
        if name in BUILTINS and "P" not in flags:
            found = True
            out.append(("builtin", f"{name} is a shell builtin"))
        paths = _all_paths(sh, name) if "a" in flags or not out or "P" in flags else []
        if not ("a" in flags) and out and "P" not in flags:
            paths = []
        for p in paths:
            found = True
            hashed = name in sh.hashed and sh.hashed[name] == p and "a" not in flags
            out.append(("file", f"{name} is hashed ({p})" if hashed else f"{name} is {p}", p))
        if not found:
            if "t" not in flags and "p" not in flags and "P" not in flags:
                yield from sh.error(f"type: {name}: not found")
            status = 1
            continue
        if "a" not in flags:
            out = out[:1]
        for item in out:
            kind, text = item[0], item[1]
            if "t" in flags:
                yield from sh.out(kind + "\n")
            elif "p" in flags or "P" in flags:
                if kind == "file":
                    yield from sh.out(item[2] + "\n")
            else:
                yield from sh.out(text + "\n")
    return status


def _all_paths(sh, name):
    if "/" in name:
        try:
            node = sh.fs.lookup(name, sh.proc.cwd)
            return [name] if node.type == "f" and node.mode & 0o111 else []
        except FSError:
            return []
    out = []
    for d in sh.get("PATH", "").split(":"):
        cand = (d or ".").rstrip("/") + "/" + name
        try:
            node = sh.fs.lookup(cand, sh.proc.cwd)
        except FSError:
            continue
        if node.type == "f" and node.mode & 0o111 and cand not in out:
            out.append(cand)
    return out


@builtin("command")
def b_command(sh, args):
    a = args[1:]
    flags = set()
    while a and a[0].startswith("-") and len(a[0]) > 1:
        o = a.pop(0)
        if o == "--":
            break
        flags |= set(o[1:])
    if not a:
        return 0
    name = a[0]
    if "v" in flags or "V" in flags:
        status = 0
        for name in a:
            if name in sh.aliases:
                yield from sh.out(f"alias {name}={single_quote(sh.aliases[name])}\n" if "v" in flags
                                  else f"{name} is aliased to `{sh.aliases[name]}'\n")
            elif _keyword(name) or name in sh.funcs or name in BUILTINS:
                if "v" in flags:
                    yield from sh.out(name + "\n")
                else:
                    kind = "a shell keyword" if _keyword(name) else ("a function" if name in sh.funcs else "a shell builtin")
                    yield from sh.out(f"{name} is {kind}\n")
            else:
                p = sh.find_command(name)
                if p is None:
                    if "V" in flags:
                        yield from sh.error(f"command: {name}: not found")
                    status = 1
                else:
                    yield from sh.out((p if "v" in flags else f"{name} is {p}") + "\n")
        return status
    b = BUILTINS.get(name)
    if b is not None:
        return (yield from b(sh, a))
    return (yield from sh.run_external(a, [], [], False))


@builtin("builtin")
def b_builtin(sh, args):
    if len(args) < 2:
        return 0
    b = BUILTINS.get(args[1])
    if b is None:
        yield from sh.error(f"builtin: {args[1]}: not a shell builtin")
        return 1
    return (yield from b(sh, args[1:]))


@builtin("hash")
def b_hash(sh, args):
    a = args[1:]
    if not a:
        if not sh.hashed:
            yield from sh.err("hash: hash table empty\n")
            return 0
        hits = getattr(sh, "hash_hits", {})
        lines = ["hits\tcommand"] + [f"{hits.get(n, 1):4d}\t{p}" for n, p in sh.hashed.items()]
        yield from sh.out("".join(x + "\n" for x in lines))
        return 0
    if a[0] == "-r":
        sh.hashed.clear()
        return 0
    if a[0] == "-d":
        for n in a[1:]:
            sh.hashed.pop(n, None)
        return 0
    if a[0] == "-t":
        status = 0
        for n in a[1:]:
            if n in sh.hashed:
                yield from sh.out(sh.hashed[n] + "\n")
            else:
                yield from sh.error(f"hash: {n}: not found")
                status = 1
        return status
    status = 0
    for n in a:
        p = sh.find_command(n)
        if p is None:
            yield from sh.error(f"hash: {n}: not found")
            status = 1
        else:
            sh.hashed[n] = p
    return status


@builtin("alias")
def b_alias(sh, args):
    a = [x for x in args[1:] if x != "--"]
    if not a or a == ["-p"]:
        yield from sh.out("".join(f"alias {n}={single_quote(v)}\n" for n, v in sorted(sh.aliases.items())))
        return 0
    status = 0
    for item in a:
        if "=" in item:
            n, v = item.split("=", 1)
            if not n or any(c in n for c in " \t\n/$`=|&;()<>'\""):
                yield from sh.error(f"alias: `{n}': invalid alias name")
                status = 1
                continue
            sh.aliases[n] = v
        else:
            if item in sh.aliases:
                yield from sh.out(f"alias {item}={single_quote(sh.aliases[item])}\n")
            else:
                yield from sh.error(f"alias: {item}: not found")
                status = 1
    return status


@builtin("unalias")
def b_unalias(sh, args):
    a = args[1:]
    if not a:
        yield from sh.err("unalias: usage: unalias [-a] name [name ...]\n")
        return 2
    if a[0] == "-a":
        sh.aliases.clear()
        return 0
    status = 0
    for n in a:
        if n in sh.aliases:
            del sh.aliases[n]
        else:
            yield from sh.error(f"unalias: {n}: not found")
            status = 1
    return status


# ---------------------------------------------------------------------------- history

@builtin("history")
def b_history(sh, args):
    a = args[1:]
    if a and a[0] == "-c":
        sh.history.clear()
        sh.hist_base = 1
        return 0
    if a and a[0] == "-d":
        try:
            n = int(a[1])
        except (IndexError, ValueError):
            yield from sh.error("history: -d: option requires an argument")
            return 2
        idx = n - sh.hist_base if n > 0 else len(sh.history) + n
        if not 0 <= idx < len(sh.history):
            yield from sh.error(f"history: {n}: history position out of range")
            return 1
        del sh.history[idx]
        return 0
    if a and a[0] == "-s":
        sh.history.append(" ".join(a[1:]))
        return 0
    if a and a[0] in ("-w", "-a", "-r", "-n"):
        hf = sh.get("HISTFILE")
        if hf and a[0] in ("-w", "-a"):
            try:
                of = open_path(sh.proc, hf, "w" if a[0] == "-w" else "a")
                from .fileio import write
                yield from write(sh.kernel, of, "".join(h + "\n" for h in sh.history).encode())
            except FSError as e:
                yield from sh.error(f"history: {hf}: {e.msg}")
                return 1
        return 0
    entries = list(enumerate(sh.history, sh.hist_base))
    if a:
        try:
            n = int(a[0])
        except ValueError:
            yield from sh.error(f"history: {a[0]}: numeric argument required")
            return 1
        entries = entries[-n:] if n > 0 else []
    yield from sh.out("".join(f"{i:5d}  {h}\n" for i, h in entries))
    return 0


@builtin("fc")
def b_fc(sh, args):
    a = args[1:]
    if a and a[0] == "-l":
        entries = list(enumerate(sh.history, sh.hist_base))[:-1][-16:]
        yield from sh.out("".join(f"{i}\t {h}\n" for i, h in entries))
        return 0
    if a and a[0] == "-s":
        rest = a[1:]
        subst = [x for x in rest if "=" in x]
        prefix = [x for x in rest if "=" not in x]
        hist = sh.history[:-1]
        cmd = None
        if prefix:
            cmd = next((h for h in reversed(hist) if h.startswith(prefix[0])), None)
        elif hist:
            cmd = hist[-1]
        if cmd is None:
            yield from sh.error("fc: no command found")
            return 1
        for s in subst:
            old, new = s.split("=", 1)
            cmd = cmd.replace(old, new, 1)
        yield from sh.err(cmd + "\n")
        if sh.history:
            sh.history[-1] = cmd
        return (yield from b_eval(sh, ["eval", cmd]))
    yield from sh.error("fc: the editor isn't available in this terminal; use `fc -s old=new` or `fc -l`")
    return 1


# ---------------------------------------------------------------------------- misc

@builtin("times")
def b_times(sh, args):
    yield from sh.out("0m0.012s 0m0.004s\n0m0.031s 0m0.009s\n")
    return 0


@builtin("umask")
def b_umask(sh, args):
    a = args[1:]
    sym = "-S" in a
    a = [x for x in a if x not in ("-S", "-p")]
    m = sh.proc.umask
    if not a:
        if sym:
            perm = 0o777 & ~m

            def part(shift):
                b = (perm >> shift) & 7
                return ("r" if b & 4 else "") + ("w" if b & 2 else "") + ("x" if b & 1 else "")
            yield from sh.out(f"u={part(6)},g={part(3)},o={part(0)}\n")
        else:
            yield from sh.out(f"{m:04o}\n")
        return 0
    v = a[0]
    if re.fullmatch(r"[0-7]{1,4}", v):
        sh.proc.umask = int(v, 8)
        return 0
    yield from sh.error(f"umask: {v}: octal number out of range" if v.isdigit() else f"umask: `{v[0]}': invalid symbolic mode operator")
    return 1


ULIMITS = [("c", "core file size", "blocks", "0"), ("d", "data seg size", "kbytes", "unlimited"),
           ("e", "scheduling priority", "", "0"), ("f", "file size", "blocks", "unlimited"),
           ("i", "pending signals", "", "127767"), ("l", "max locked memory", "kbytes", "8192"),
           ("m", "max memory size", "kbytes", "unlimited"), ("n", "open files", "", "1024"),
           ("p", "pipe size", "512 bytes", "8"), ("q", "POSIX message queues", "bytes", "819200"),
           ("r", "real-time priority", "", "0"), ("s", "stack size", "kbytes", "8192"),
           ("t", "cpu time", "seconds", "unlimited"), ("u", "max user processes", "", "127767"),
           ("v", "virtual memory", "kbytes", "unlimited"), ("x", "file locks", "", "unlimited")]


@builtin("ulimit")
def b_ulimit(sh, args):
    a = args[1:]
    if "-a" in a:
        lines = []
        for c, desc, unit, val in ULIMITS:
            label = f"{desc} ({unit + ', ' if unit else ''}-{c})"
            lines.append(f"{label:<32}{val}")
        yield from sh.out("".join(x + "\n" for x in lines))
        return 0
    letter = "f"
    for x in a:
        if x.startswith("-") and len(x) == 2:
            letter = x[1]
    val = next((v for c, _, _, v in ULIMITS if c == letter), "unlimited")
    if any(not x.startswith("-") for x in a):
        return 0
    yield from sh.out(val + "\n")
    return 0


@builtin("caller")
def b_caller(sh, args):
    if not sh.funcnames:
        return 1
    yield from sh.out(f"{sh.lineno} {sh.funcnames[-2] if len(sh.funcnames) > 1 else 'main'} {sh.script or 'main'}\n")
    return 0


@builtin("enable")
def b_enable(sh, args):
    if len(args) == 1 or args[1] in ("-a", "-p"):
        yield from sh.out("".join(f"enable {n}\n" for n in sorted(BUILTINS)))
    return 0


@builtin("complete", "compopt")
def b_complete(sh, args):
    if len(args) > 1 and args[1] == "-p":
        for n, spec in sorted(getattr(sh, "completions", {}).items()):
            yield from sh.out(f"complete {spec} {n}\n")
        return 0
    if len(args) >= 3:
        comp = getattr(sh, "completions", {})
        comp[args[-1]] = " ".join(args[1:-1])
        sh.completions = comp
    return 0


@builtin("bind")
def b_bind(sh, args):
    if len(args) > 1 and args[1] in ("-P", "-p", "-l"):
        yield from sh.out("".join(f"\"{k}\": {v}\n" for k, v in [
            ("\\C-a", "beginning-of-line"), ("\\C-e", "end-of-line"), ("\\C-k", "kill-line"),
            ("\\C-r", "reverse-search-history"), ("\\C-u", "unix-line-discard"), ("\\C-w", "unix-word-rubout"),
            ("\\C-y", "yank"), ("\\eb", "backward-word"), ("\\ef", "forward-word"), ("\\e.", "yank-last-arg")]))
    return 0


@builtin("suspend")
def b_suspend(sh, args):
    if sh.interactive and "-f" not in args:
        yield from sh.error("suspend: cannot suspend a login shell")
        return 1
    return 0


@builtin("compgen")
def b_compgen(sh, args):
    a = args[1:]
    word = ""
    actions = []
    wordlist = None
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-W":
            wordlist = a[i + 1]
            i += 2
            continue
        if x == "-A":
            actions.append(a[i + 1])
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            m = {"c": "command", "b": "builtin", "k": "keyword", "a": "alias", "f": "file", "d": "directory",
                 "v": "variable", "e": "export", "u": "user", "j": "job"}
            for c in x[1:]:
                if c in m:
                    actions.append(m[c])
            i += 1
            continue
        word = x
        i += 1
    cands = []
    for act in actions:
        if act in ("command", "builtin"):
            cands += sorted(BUILTINS)
        if act in ("command", "keyword"):
            cands += sorted(RESERVED)
        if act in ("command", "alias"):
            cands += sorted(sh.aliases)
        if act in ("command", "function"):
            cands += sorted(sh.funcs)
        if act == "command":
            for d in sh.get("PATH", "").split(":"):
                try:
                    node = sh.fs.lookup(d or ".", sh.proc.cwd)
                except FSError:
                    continue
                if node.type == "d":
                    cands += [n for n in sh.fs.listdir(node) if node.entries[n].type == "f" and node.entries[n].mode & 0o111]
        if act in ("file", "directory"):
            d, _, prefix = word.rpartition("/")
            base = d + "/" if d else ""
            try:
                node = sh.fs.lookup(d or ".", sh.proc.cwd)
                for n in sh.fs.listdir(node):
                    if act == "directory" and node.entries[n].type != "d":
                        continue
                    cands.append(base + n)
            except FSError:
                pass
        if act in ("variable",):
            cands += sorted(_all_vars(sh))
        if act == "export":
            cands += sorted(sh.env())
        if act == "user":
            cands += sorted(u.name for u in sh.host.users.values())
        if act == "job":
            cands += [j.text.split()[0] for j in sh.jobs.values()]
    if wordlist is not None:
        cands += wordlist.split()
    seen = []
    for c in cands:
        if c.startswith(word) and c not in seen:
            seen.append(c)
    if not seen:
        return 1
    yield from sh.out("".join(c + "\n" for c in seen))
    return 0


@builtin("help")
def b_help(sh, args):
    from .helptext import BUILTIN_HELP
    a = [x for x in args[1:] if not x.startswith("-")]
    flags = "".join(x[1:] for x in args[1:] if x.startswith("-"))
    if not a:
        names = sorted(BUILTIN_HELP)
        lines = ["GNU bash, version 5.2.21(1)-release (x86_64-pc-linux-gnu)",
                 "These shell commands are defined internally.  Type `help' to see this list.",
                 "Type `help name' to find out more about the function `name'.",
                 "Use `info bash' to find out more about the shell in general.",
                 "Use `man -k' or `info' to find out more about commands not in this list.", ""]
        lines += [f" {BUILTIN_HELP[n][0]}" for n in names]
        yield from sh.out("\n".join(lines) + "\n")
        return 0
    status = 0
    for name in a:
        hits = [n for n in BUILTIN_HELP if P.fnmatch(name, n)] or [n for n in BUILTIN_HELP if n.startswith(name)]
        if not hits:
            yield from sh.error(f"help: no help topics match `{name}'.  Try `help help' or `man -k {name}' or `info {name}'.")
            status = 1
            continue
        for n in hits:
            syn, short, body = BUILTIN_HELP[n]
            if "d" in flags:
                yield from sh.out(f"{n} - {short}\n")
            elif "s" in flags:
                yield from sh.out(f"{syn}\n")
            else:
                text = f"{syn}\n    {short}\n" + ("    \n" + "".join(f"    {ln}\n" if ln else "    \n" for ln in body.split("\n")) if body else "")
                yield from sh.out(text)
    return status

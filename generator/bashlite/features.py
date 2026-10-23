"""What a piece of bash uses: commands, their flags and syntax, for a problem's `require` / `forbid` lists.

features(src) -> set of strings:
  * every command run, by name: "grep", "sort" (also inside $(...), pipelines, `xargs CMD`, `find -exec CMD`,
    `sudo CMD`, `bash -c '...'`, `watch CMD` ...)
  * "<cmd> <flag>" for each option: "grep -c", "sort -n", "sort -k" (from -k2), "ls --all", "find -name";
    combined short options count separately ("ls -la" gives "ls -l" and "ls -a")
  * "<cmd> <subcommand>" for the first operand: "pip install", "tmux attach", "conda create"
  * syntax: "|", "|&", "&&", "||", "&", "!", "time", ">", ">>", "<", "2>", "2>>", "&>", "2>&1", ">&2", "<<",
    "<<-", "<<<", ">|", "$(", "`", "$((", "<(", ">(", "((", "[[", "{", "(", "if", "for", "for((", "while",
    "until", "case", "select", "function", "glob", "**", "{,}", "{..}", "~", "$name", "${", "array", "arr=(",
    "+=", "local", "[[ <op>" (e.g. "[[ -f", "[[ =~"), "param:<op>" (e.g. "param::-", "param:%%", "param:#",
    "param:len", "param:/", "param:^^", "param:sub", "param:!"), special parameters "$?", "$#", "$@", "$*",
    "$$", "$!", "$0", "$1".., history "!!", "!$", "!*", "!n", "!-n", "!str", "^^" (quick substitution).
"""

from __future__ import annotations

import re

from .parser import ParseError, Parser
from .syntax import (AndOr, ArithCmd, ArithFor, ArithSub, Assign, Brace, Case, CmdList, CmdSub, Cond, CondAnd,
                     CondBinary, CondNot, CondOr, CondUnary, DQ, For, FuncDef, If, Lit, Node, Param, Pipeline, ProcSub, Redir, Select, Simple,
                     Subshell, While, Word)

# commands that run another command given as their arguments: name -> options that take a value
WRAPPERS = {"sudo": {"-u", "-g", "-C"}, "command": set(), "builtin": set(), "exec": {"-a"}, "nohup": set(),
            "env": {"-u", "-C", "-S"}, "nice": {"-n"}, "timeout": {"-s", "-k"}, "watch": {"-n", "-d"},
            "time": {"-f", "-o"}, "xargs": {"-n", "-L", "-I", "-P", "-d", "-a", "-E", "-s", "-i"}, "stdbuf": set(),
            "setsid": set(), "taskset": set(), "srun": set(), "uv": set(), "torchrun": set()}
SHELLS = {"bash", "sh", "dash", "zsh"}


def _parse(src: str):
    try:
        return [Parser(src).parse_all()]
    except ParseError:
        out = []
        for line in src.split("\n"):
            try:
                out.append(Parser(line).parse_all())
            except ParseError:
                continue
        return out


def features(src: str) -> set[str]:
    out: set[str] = set()
    for tree in _parse(src):
        _walk(tree, out)
    _history(src, out)
    return out


def _history(src, out):
    for line in src.split("\n"):
        s = line.strip()
        if s.startswith("^") and s.count("^") >= 2:
            out.add("^^")
        for m in re.finditer(r"(?<![\\$\[{])!(!|\$|\*|-?\d+|\?[^?]*\??|[A-Za-z_][\w.-]*)", s):
            tok = m.group(1)
            if tok in ("!", "$", "*"):
                out.add("!" + tok)
            elif tok.lstrip("-").isdigit():
                out.add("!-n" if tok.startswith("-") else "!n")
            elif tok.startswith("?"):
                out.add("!?")
            else:
                out.add("!str")
        if re.search(r"!!:|!\$:|:s/|:gs/|:h\b|:t\b|:r\b|:e\b|:p\b", s):
            out.add("!:")


def _walk(node, out):
    if node is None:
        return
    if isinstance(node, list):
        for x in node:
            _walk(x, out)
        return
    if isinstance(node, tuple):
        for x in node:
            _walk(x, out)
        return
    if not isinstance(node, Node):
        return
    t = type(node)
    if t is CmdList:
        for ao, bg in node.items:
            if bg:
                out.add("&")
            _walk(ao, out)
        return
    if t is AndOr:
        for op, _ in node.rest:
            out.add(op)
        _walk(node.first, out)
        for _, p in node.rest:
            _walk(p, out)
        return
    if t is Pipeline:
        if len(node.cmds) > 1:
            out.add("|")
        if any(node.pipe_err):
            out.add("|&")
        if node.negate:
            out.add("!")
        if node.timed:
            out.add("time")
        _walk(node.cmds, out)
        return
    if t is Simple:
        _simple(node, out)
        return
    if t is Redir:
        _redir(node, out)
        return
    if t is FuncDef:
        out.add("function")
        _walk(node.body, out)
        return
    if t is Brace:
        out.add("{")
    elif t is Subshell:
        out.add("(")
    elif t is If:
        out.add("if")
    elif t is For:
        out.add("for")
    elif t is ArithFor:
        out.add("for((")
    elif t is Select:
        out.add("select")
    elif t is While:
        out.add("until" if node.until else "while")
    elif t is Case:
        out.add("case")
    elif t is Cond:
        out.add("[[")
    elif t is ArithCmd:
        out.add("((")
    elif t is CondNot:
        out.add("[[ !")
    elif t is CondAnd:
        out.add("[[ &&")
    elif t is CondOr:
        out.add("[[ ||")
    elif t is CondUnary:
        out.add(f"[[ {node.op}")
    elif t is CondBinary:
        out.add(f"[[ {node.op}")
    elif t is Word:
        _word(node, out)
        return
    elif t is Assign:
        if node.array is not None:
            out.add("arr=(")
            out.add("array")
        if node.index is not None:
            out.add("array")
        if node.append:
            out.add("+=")
    elif t is CmdSub:
        out.add("`" if node.backtick else "$(")
        _sub_body(node.body, node.src, out)
        return
    elif t is ProcSub:
        out.add(node.direction + "(")
        _sub_body(node.body, node.src, out)
        return
    elif t is ArithSub:
        out.add("$((")
    elif t is Param:
        _param(node, out)
        return
    if t is CondBinary:
        _walk(node.left, out)
        (_inner if node.op in ("=~", "==", "=", "!=") else _walk)(node.right, out)
        return
    for k in t.__slots__:
        v = getattr(node, k, None)
        if isinstance(v, (Node, list, tuple)):
            _walk(v, out)


def _sub_body(body, src, out):
    if isinstance(body, Node):
        _walk(body, out)
    elif isinstance(src, str):
        for tree in _parse(src):
            _walk(tree, out)


def _param(p: Param, out):
    name = p.name
    if name in ("?", "#", "@", "*", "$", "!", "0", "-") or (isinstance(name, str) and name.isdigit()):
        out.add("$" + name if not name.isdigit() or name == "0" else "$1")
    else:
        out.add("$name")
    if p.braced:
        out.add("${")
    if p.sub is not None:
        out.add("array")
        sub = p.sub.raw if isinstance(p.sub, Word) else str(p.sub)
        if sub in ("@", "*"):
            out.add("array[@]")
    if p.indirect:
        out.add("param:!")
    if p.op:
        out.add("param:" + {":": "sub"}.get(p.op, p.op))
    for arg in (p.arg, p.arg2, p.sub):
        _inner(arg, out)


WORD_SHAPES = {"glob", "**", "{,}", "{..}", "~"}


def _inner(node, out):
    """Walk a pattern or regex operand: what it runs counts, its glob-like characters do not."""
    if node is None:
        return
    tmp: set[str] = set()
    _walk(node, tmp)
    out |= tmp - WORD_SHAPES


def _word(w: Word, out):
    for part in w.parts:
        if type(part) is Lit:
            text = part.text
            if re.search(r"(?<!\\)[*?]|\[[^\]]+\]", text):
                out.add("glob")
            if "**" in text:
                out.add("**")
            if re.search(r"\{[^{}]*,[^{}]*\}", text):
                out.add("{,}")
            if re.search(r"\{-?\w+\.\.-?\w+(\.\.-?\d+)?\}", text):
                out.add("{..}")
            if text.startswith("~") and part is w.parts[0]:
                out.add("~")
        else:
            _walk(part, out)
    if w.array is not None:
        _walk(w.array, out)


def _redir(r: Redir, out):
    op = r.op
    fd = "" if r.fd is None else str(r.fd)
    if op in (">&", "<&"):
        tgt = r.target.literal() if r.target is not None else ""
        out.add(f"{fd or ('1' if op == '>&' else '0')}{op}{tgt}" if tgt and tgt != "-" else op)
        if op == ">&" and not fd and tgt == "2":
            out.add(">&2")
        if fd == "2" and tgt == "1":
            out.add("2>&1")
    elif op in ("<<", "<<-"):
        out.add(op)
    else:
        out.add(fd + op if fd not in ("", "1" if op in (">", ">>", ">|") else "0") else op)
    if r.target is not None:
        _walk(r.target, out)
    if r.heredoc is not None and r.heredoc.body is not None:
        _walk(r.heredoc.body, out)


def _simple(c: Simple, out):
    for a in c.assigns:
        _walk(a, out)
    for r in c.redirs:
        _redir(r, out)
    words = c.words
    for w in words:
        _word(w, out)
    lits = [w.literal() for w in words]
    _command(lits, out)


def _command(lits, out):
    """Record a command line given as literal words (None for words with expansions)."""
    i = 0
    while i < len(lits) and lits[i] is not None and re.match(r"^[A-Za-z_]\w*=", lits[i]) and i == 0:
        i += 1
    if i >= len(lits) or lits[i] is None:
        return
    name = lits[i].rsplit("/", 1)[-1] if not lits[i].startswith("./") else lits[i]
    if name in ("local", "declare", "typeset", "export", "readonly"):
        out.add(name)
    out.add(name)
    args = lits[i + 1:]
    _flags(name, _own_args(name, args), out)
    # commands that run commands
    if name in SHELLS:
        for k, a in enumerate(args):
            if a == "-c" and k + 1 < len(args) and args[k + 1] is not None:
                for tree in _parse(args[k + 1]):
                    _walk(tree, out)
                return
    if name in ("python", "python3", "python3.12") and len(args) > 1 and args[0] == "-m" and args[1] is not None:
        _command(args[1:], out)          # python -m pip install -U pip: the module runs like a command
    if name == "find":
        for k, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                rest = []
                for b in args[k + 1:]:
                    if b in (";", "+", "\\;"):
                        break
                    rest.append(b)
                _command(rest, out)
        return
    if name in ("ssh",):
        own = _own_args(name, args)
        remote = args[len(own):]
        if remote and all(a is not None for a in remote):
            for tree in _parse(" ".join(remote)):
                _walk(tree, out)
        return
    if name in WRAPPERS:
        takes = WRAPPERS[name]
        k = 0
        while k < len(args):
            a = args[k]
            if a is None:
                return
            if name == "env" and re.match(r"^[A-Za-z_]\w*=", a):
                k += 1
                continue
            if name == "timeout" and not a.startswith("-") and re.match(r"^\d+(\.\d+)?[smhd]?$", a):
                k += 1
                break
            if a == "--":
                k += 1
                break
            if a.startswith("-"):
                k += 2 if a in takes else 1
                continue
            break
        if name == "uv":
            if args[:1] == ["run"]:
                _command(args[1:], out)
            return
        if name == "torchrun":
            return
        if k < len(args):
            _command(args[k:], out)


def _own_args(name, args):
    """The arguments that belong to `name` itself (not to a command it runs)."""
    if name == "find":
        for k, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                return args[:k + 1]
        return args
    if name == "ssh":
        for k, a in enumerate(args):
            if a is not None and not a.startswith("-") and (k == 0 or args[k - 1] not in
                                                           ("-p", "-i", "-L", "-R", "-J", "-o", "-l", "-F", "-D")):
                return args[:k + 1]
        return args
    if name in WRAPPERS and name not in ("uv", "torchrun"):
        takes = WRAPPERS[name]
        k = 0
        while k < len(args):
            a = args[k]
            if a is None or a == "--":
                break
            if name == "env" and re.match(r"^[A-Za-z_]\w*=", a):
                k += 1
                continue
            if name == "timeout" and not a.startswith("-") and re.match(r"^\d+(\.\d+)?[smhd]?$", a):
                k += 1
                break
            if a.startswith("-"):
                k += 2 if a in takes else 1
                continue
            break
        return args[:k]
    return args


SUBCOMMAND_TOOLS = {"pip", "pip3", "uv", "conda", "tmux", "apt", "apt-get", "brew", "git", "docker", "jupyter", "systemctl",
                    "crontab", "ssh-keygen", "kubectl", "npm", "cargo", "go"}


def _flags(name, args, out):
    """Options of one command: "cmd -x" (clusters split), "cmd --long", "cmd -x VALUE", "cmd SUBCOMMAND" for the
    first operand, and for tools with subcommands also "cmd SUB -x" (`pip install -r`, `tmux new -s`)."""
    first_operand = True
    prev_flag = None
    sub = None

    def add(feature):
        out.add(f"{name} {feature}")
        if sub is not None:
            out.add(f"{name} {sub} {feature}")
    for a in args:
        if a is None:
            prev_flag = None
            first_operand = False
            continue
        if a == "--":
            break
        if a.startswith("--") and len(a) > 2:
            flag = a.split("=", 1)[0]
            add(flag)
            prev_flag = flag if "=" not in a else None
            continue
        if a.startswith("-") and len(a) > 1 and not re.match(r"^-\d+(\.\d+)?$", a) or (name == "head" or name == "tail") and re.match(r"^-\d+$", a):
            add(a)
            body = a[1:]
            if name != "find" and re.match(r"^[A-Za-z0-9]+$", body):
                for ch in body:          # a cluster of short options (-la, -1S, -k2)
                    if ch.isalpha():
                        add(f"-{ch}")
            elif name != "find" and body[:1].isalpha():
                add(f"-{body[0]}")       # -k3,3nr, -d, -v+1d: the option letter
            prev_flag = a
            continue
        if a.startswith("+") and name in ("tail", "set", "shopt"):
            out.add(f"{name} {a}")
        if prev_flag is not None:
            out.add(f"{name} {prev_flag} {a}")
            prev_flag = None
        if first_operand:
            out.add(f"{name} {a}")
            first_operand = False
            if name in SUBCOMMAND_TOOLS:
                sub = a
        elif sub is not None and name in ("uv", "conda", "jupyter") and len(sub.split()) == 1 and not a.startswith(("/", ".", "~")) \
                and sub in ("pip", "env", "python", "kernelspec"):
            sub = f"{sub} {a}"           # uv pip install, conda env create
            out.add(f"{name} {sub}")


def alternatives(req: str) -> list[str]:
    """ "grep|awk" -> ["grep", "awk"]; the pipe features "|", "||" and "|&" stay whole."""
    if req in ("|", "||", "|&"):
        return [req]
    parts = re.split(r"(?<=[^|])\|(?=[^|])", req)
    return [p for p in parts if p] or [req]


def missing(src: str, require: list[str]) -> str | None:
    """None if `src` uses every required feature ("a|b" = either), else a message for the first missing one."""
    have = features(src)
    for req in require or []:
        alts = alternatives(req)
        if not any(a in have for a in alts):
            return "This problem asks you to use " + " or ".join(f"`{_shown(a)}`" for a in alts) + "."
    return None


def forbidden(src: str, forbid: list[str]) -> str | None:
    have = features(src)
    for f in forbid or []:
        if f in have:
            return f"Solve this one without `{_shown(f)}`."
    return None


def _shown(feature: str) -> str:
    if feature.startswith("param:"):
        op = feature[6:]
        return {"len": "${#var}", "sub": "${var:offset:length}", "!": "${!var}"}.get(op, "${var" + op + "...}")
    if feature.startswith("[[ "):
        return feature + " ... ]]"
    return {"glob": "a glob pattern (*, ?, [...])", "{,}": "brace expansion {a,b}", "{..}": "a {start..end} sequence",
            "arr=(": "an array", "array": "an array", "array[@]": '"${array[@]}"', "$name": "a variable",
            "function": "a function", "^^": "^old^new", "!str": "!string", "!n": "!n", "!-n": "!-n"}.get(feature, feature)

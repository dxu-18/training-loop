"""Print commands back as bash does for `type f`, `declare -f` and `jobs`."""

from __future__ import annotations

from .syntax import (AndOr, ArithCmd, ArithFor, Brace, Case, CmdList, Cond, For, FuncDef, If, Pipeline, Select, Simple,
                     Subshell, While)

IND = "    "
_INLINE_GROUPS = [False]     # jobs print { ...; } groups on one line (loops and ifs still span lines)


def job_text(node) -> str:
    """A command as `jobs` and job notices show it."""
    _INLINE_GROUPS[0] = True
    try:
        return pipeline_text(node, 0) if isinstance(node, Pipeline) else andor_text(node, 0)
    finally:
        _INLINE_GROUPS[0] = False


def redirs_text(redirs) -> tuple[str, list[str]]:
    parts, heredocs = [], []
    for r in redirs:
        fd = "" if r.fd is None else str(r.fd)
        if r.op in ("<<", "<<-"):
            parts.append(f"{fd}{r.op}{r.target.raw}")
            heredocs.append(r.heredoc.text + r.heredoc.delim + "\n")
        elif r.op in (">&", "<&"):
            parts.append(f"{fd}{r.op}{r.target.raw}")
        else:
            parts.append(f"{fd}{r.op} {r.target.raw}")
    return " ".join(parts), heredocs


def simple_text(c: Simple) -> tuple[str, list[str]]:
    words = [a.raw for a in c.assigns] + [w.raw for w in c.words]
    rt, hd = redirs_text(c.redirs)
    return " ".join(words + ([rt] if rt else [])), hd


def pipeline_text(p: Pipeline, level: int) -> str:
    out = []
    for i, c in enumerate(p.cmds):
        out.append(command_text(c, level))
        if i < len(p.cmds) - 1:
            out.append(" |& " if p.pipe_err and p.pipe_err[i] else " | ")
    s = "".join(out)
    if p.negate:
        s = "! " + s
    if p.timed:
        s = "time " + s
    return s


def andor_text(a: AndOr, level: int) -> str:
    s = pipeline_text(a.first, level)
    for op, p in a.rest:
        s += f" {op} " + pipeline_text(p, level)
    return s


def inline_list(lst: CmdList, level: int) -> str:
    parts = []
    for ao, bg in lst.items:
        parts.append(andor_text(ao, level) + (" &" if bg else ""))
    return "; ".join(parts)


def body_lines(lst: CmdList, level: int, semi_last: bool) -> list[str]:
    out = []
    n = len(lst.items)
    for i, (ao, bg) in enumerate(lst.items):
        text = andor_text(ao, level)
        if bg:
            text += " &"
        elif i < n - 1 or semi_last:
            if not text.rstrip().endswith("EOF") and "\n" not in text:
                text += ";"
        out.append(IND * level + text)
    return out


def command_text(c, level: int) -> str:
    t = type(c)
    pad = IND * level
    if t is Simple:
        s, hd = simple_text(c)
        if hd:
            s += "\n" + "".join(hd)
        return s
    rt, hd = redirs_text(getattr(c, "redirs", []))
    suffix = (" " + rt if rt else "") + ("\n" + "".join(hd) if hd else "")
    if t is Brace and _INLINE_GROUPS[0]:
        body = inline_list(c.body, level)
        return "{ " + body + ("" if body.endswith("&") else ";") + " }" + suffix
    if t is Brace:
        lines = body_lines(c.body, level + 1, False)
        return "{ \n" + "\n".join(lines) + "\n" + pad + "}" + suffix
    if t is Subshell:
        return "( " + inline_list(c.body, level) + " )" + suffix
    if t is If:
        return _if_text(c.clauses, c.else_, level) + suffix
    if t is While:
        kw = "until" if c.until else "while"
        return (f"{kw} {inline_list(c.cond, level)}; do\n" + "\n".join(body_lines(c.body, level + 1, True)) +
                "\n" + pad + "done" + suffix)
    if t in (For, Select):
        kw = "for" if t is For else "select"
        words = "" if c.words is None else " in " + " ".join(w.raw for w in c.words)
        return (f"{kw} {c.var}{words};\n{pad}do\n" + "\n".join(body_lines(c.body, level + 1, True)) + "\n" + pad +
                "done" + suffix)
    if t is ArithFor:
        return (f"for (({c.init.raw}; {c.cond.raw.strip()}; {c.step.raw.strip()}))\n{pad}do\n" +
                "\n".join(body_lines(c.body, level + 1, True)) + "\n" + pad + "done" + suffix)
    if t is Case:
        out = [f"case {c.word.raw} in "]
        for pats, body, term in c.items:
            out.append(IND * (level + 1) + " | ".join(p.raw for p in pats) + ")")
            if body is not None:
                out += body_lines(body, level + 2, False)
            out.append(IND * (level + 1) + term)
        out.append(pad + "esac")
        return "\n".join(out) + suffix
    if t is Cond:
        return f"[[ {c.src} ]]" + suffix
    if t is ArithCmd:
        return f"(( {c.src.strip()} ))" + suffix
    if t is FuncDef:
        return function_text(c, level)
    if t is CmdList:
        return inline_list(c, level)
    return "?"


def _if_text(clauses, else_, level) -> str:
    pad = IND * level
    cond, body = clauses[0]
    out = [f"if {inline_list(cond, level)}; then"]
    out += body_lines(body, level + 1, True)
    if len(clauses) > 1:
        out.append(pad + "else")
        out.append(IND * (level + 1) + _if_text(clauses[1:], else_, level + 1) + ";")
    elif else_ is not None:
        out.append(pad + "else")
        out += body_lines(else_, level + 1, True)
    out.append(pad + "fi")
    return "\n".join(out)


def function_text(fn: FuncDef, level: int = 0) -> str:
    body = fn.body
    pad = IND * level
    if type(body) is Brace:
        lines = body_lines(body.body, level + 1, False)
        text = "\n".join(lines)
        rt, hd = redirs_text(body.redirs)
        return f"{fn.name} () \n{pad}{{ \n{text}\n{pad}}}" + (" " + rt if rt else "")
    return f"{fn.name} () \n{pad}" + command_text(body, level)

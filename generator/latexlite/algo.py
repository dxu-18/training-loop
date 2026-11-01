"""algorithmic (algpseudocode) and code listings (listings; minted needs -shell-escape)."""

from __future__ import annotations

import re

from .engine import cmd, env
from .model import PLAIN, text
from .tokens import Tok

BLOCKS = {"For": ("for", "do", "EndFor", "end for"), "ForAll": ("for all", "do", "EndFor", "end for"),
          "While": ("while", "do", "EndWhile", "end while"), "If": ("if", "then", "EndIf", "end if"),
          "Loop": ("loop", None, "EndLoop", "end loop"), "Repeat": ("repeat", None, "Until", None),
          "Function": ("function", None, "EndFunction", "end function"),
          "Procedure": ("procedure", None, "EndProcedure", "end procedure")}
ENDS = {"EndFor": "For", "EndWhile": "While", "EndIf": "If", "EndLoop": "Loop", "Until": "Repeat",
        "EndFunction": "Function", "EndProcedure": "Procedure"}
BOLD = PLAIN.with_(bold=True)


@env("algorithmic")
def algorithmic_env(e, name, t):
    step = e.read_opt()
    try:
        every = int("".join(x.text() for x in step).strip()) if step is not None else 0
    except ValueError:
        every = 0
    body = e.env_body(name, t)
    lines = []
    stack = []
    # split at the commands; each starts a line
    cmds = []
    cur = None
    for tok in body:
        if tok.k == "cs" and tok.v in ("State", "Statex", "Require", "Ensure", "Input", "Output", "Return",
                                       "Comment") + tuple(BLOCKS) + tuple(ENDS) + ("Else", "ElsIf"):
            e.used["\\" + tok.v] += 1
            if tok.v == "Comment" and cur is not None:
                cur["toks"].append(tok)
                continue
            if tok.v == "Return" and cur is not None and cur["cmd"] in ("State", "Statex") and \
                    not any(x.k not in ("sp",) for x in cur["toks"]):
                cur["toks"].append(tok)
                continue
            cur = {"cmd": tok.v, "tok": tok, "toks": []}
            cmds.append(cur)
            continue
        if cur is None:
            if tok.k not in ("sp", "par"):
                e.error("Undefined control sequence." if tok.k == "cs" else "LaTeX Error: Something's wrong--"
                        "perhaps a missing \\item.", tok, "Every line of an algorithmic starts with \\State, \\If, "
                                                          "\\For ...")
            continue
        cur["toks"].append(tok)
    n = 0
    for c in cmds:
        name_c = c["cmd"]
        toks = c["toks"]
        level = len(stack)
        if name_c in ENDS:
            want = ENDS[name_c]
            if not stack or stack[-1] != want and not (want == "For" and stack[-1] == "ForAll"):
                e.error("LaTeX Error: Some blocks are not closed!!!", c["tok"],
                        f"\\{name_c} doesn't match an open \\{want}.")
            else:
                stack.pop()
            level = len(stack)
            if name_c == "Until":
                arg, rest = _arg(toks)
                content = [text("until ", BOLD)] + e.capture(arg) + e.capture(rest)
            else:
                content = [text(BLOCKS[ENDS[name_c]][3], BOLD)]
        elif name_c in BLOCKS:
            kw, after, _, _ = BLOCKS[name_c]
            if name_c in ("Function", "Procedure"):
                fn, rest = _arg(toks)
                args, rest = _arg(rest)
                content = [text(kw + " ", BOLD), text("".join(x.text() for x in fn).strip(), PLAIN.with_(shape="sc"))]
                content += [text("(")] + e.capture(args) + [text(")")] + e.capture(rest)
            elif name_c in ("Loop", "Repeat"):
                content = [text(kw, BOLD)] + e.capture(toks)
            else:
                cond, rest = _arg(toks)
                content = [text(kw + " ", BOLD)] + e.capture(cond) + [text(" " + after, BOLD)] + _comment(e, rest)
            stack.append(name_c)
        elif name_c in ("Else", "ElsIf"):
            if not stack or stack[-1] != "If":
                e.error("LaTeX Error: Some blocks are not closed!!!", c["tok"], f"\\{name_c} outside an \\If.")
            level = max(0, len(stack) - 1)
            if name_c == "Else":
                content = [text("else", BOLD)] + _comment(e, toks)
            else:
                cond, rest = _arg(toks)
                content = [text("else if ", BOLD)] + e.capture(cond) + [text(" then", BOLD)] + _comment(e, rest)
        elif name_c in ("Require", "Ensure", "Input", "Output"):
            content = [text({"Require": "Require:", "Ensure": "Ensure:", "Input": "Input:",
                             "Output": "Output:"}[name_c] + " ", BOLD)] + e.capture(toks)
            lines.append({"n": None, "indent": 0, "c": content, "label": True})
            continue
        else:   # State / Statex / Return
            content = _comment(e, toks) if any(x.k == "cs" and x.v == "Comment" for x in toks) else e.capture(
                _returns(toks))
            if name_c == "Return":
                content = [text("return ", BOLD)] + content
        if name_c == "Statex":
            lines.append({"n": None, "indent": level, "c": content})
            continue
        n += 1
        show = every > 0 and n % every == 0
        lines.append({"n": n if show else None, "indent": level, "c": content})
    if stack:
        e.error("LaTeX Error: Some blocks are not closed!!!", t,
                f"\\{stack[-1]} is never closed: add \\End{stack[-1].replace('All', '')}.")
    e.add_block({"t": "algorithmic", "lines": lines})


def _arg(toks):
    i = 0
    while i < len(toks) and toks[i].k == "sp":
        i += 1
    if i < len(toks) and toks[i].k == "bg":
        depth = 0
        k = i
        while k < len(toks):
            if toks[k].k == "bg":
                depth += 1
            elif toks[k].k == "eg":
                depth -= 1
                if depth == 0:
                    return toks[i + 1:k], toks[k + 1:]
            k += 1
    return [], toks


def _returns(toks):
    """\\Return inside a \\State: the keyword in bold."""
    out = []
    for x in toks:
        if x.k == "cs" and x.v == "Return":
            out += [Tok("cs", "textbf", x.file, x.line), Tok("bg", "{"), *[Tok("ch", c) for c in "return"],
                    Tok("eg", "}"), Tok("sp", " ")]
        else:
            out.append(x)
    return out


def _comment(e, toks):
    """Text then \\Comment{...}: the comment goes to the right, after a triangle."""
    idx = next((i for i, x in enumerate(toks) if x.k == "cs" and x.v == "Comment"), None)
    if idx is None:
        return e.capture(_returns(toks))
    before = e.capture(_returns(toks[:idx]))
    arg, rest = _arg(toks[idx + 1:])
    return before + [{"t": "algcomment", "c": e.capture(arg)}]


@cmd("State", "Statex", "Require", "Ensure", "Return", "Comment", "For", "ForAll", "EndFor", "While", "EndWhile",
     "If", "ElsIf", "Else", "EndIf", "Loop", "EndLoop", "Repeat", "Until", "Function", "EndFunction", "Procedure",
     "EndProcedure", "Input", "Output")
def alg_outside(e, t):
    e.error("Undefined control sequence.", t, f"\\{t.v} only works inside an algorithmic environment.")


@cmd("Call")
def call(e, t):
    fn = e.arg_text("Call")
    args = e.read_arg("Call")
    e.add_inline({"t": "text", "s": fn, "st": e.style.with_(shape="sc")})
    e.emit_text("(")
    e.run_tokens(args)
    e.emit_text(")")


# ---------------------------------------------------------------- listings

KEYWORDS = {
    "python": set("""and as assert async await break class continue def del elif else except False finally for from
                    global if import in is lambda None nonlocal not or pass raise return True try while with yield
                    print""".split()),
    "c": set("""auto break case char const continue default do double else enum extern float for goto if int long
              register return short signed sizeof static struct switch typedef union unsigned void volatile while
              include define""".split()),
    "c++": set("""auto break case char class const continue default delete do double else enum for if int long
                namespace new private protected public return short sizeof static struct switch template this
                throw try typedef using virtual void while bool true false""".split()),
    "java": set("""abstract boolean break byte case catch char class continue default do double else extends final
                 finally float for if implements import int interface long new package private protected public
                 return short static super switch this throw throws try void while""".split()),
    "bash": set("""if then else elif fi for while do done case esac function in return export local echo""".split()),
}
KEYWORDS["sh"] = KEYWORDS["bash"]
COMMENT = {"python": "#", "bash": "#", "sh": "#", "c": "//", "c++": "//", "java": "//"}


def lst_options(e, opt: str) -> dict:
    from .tokens import tokenize
    o = dict(e.lst_settings)
    o.update(e.kv(tokenize(opt, "opt")) if opt else {})
    return o


@env("lstlisting")
def lstlisting_env(e, name, t):
    from .features import take_verbatim
    s, extra = take_verbatim(e, name)
    opt = extra[2] if isinstance(extra, tuple) else ""
    o = lst_options(e, opt)
    lang = str(o.get("language", "")).strip("{}").lower()
    if lang.startswith("[") and "]" in lang:
        lang = lang.split("]", 1)[1]
    if lang and lang not in KEYWORDS and lang not in ("tex", "latex", "sql", "r", "matlab", "html", "xml", "json",
                                                       "yaml", "make", "ruby", "go", "rust"):
        e.error(f"Package Listings Error: Couldn't load requested language.", t,
                f"listings doesn't know the language `{lang}` (Python, C, C++, Java, bash ...).")
    caption = None
    if isinstance(o.get("caption"), str):
        from .tokens import tokenize
        e.counters.step("lstlisting")
        num = e.counters.the("lstlisting")
        e.set_label_target("lstlisting", num)
        caption = {"label": f"Listing {num}:", "c": e.capture(tokenize(o["caption"], t.file))}
    if isinstance(o.get("label"), str):
        e.add_label(o["label"], t)
    lines = s.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if lines and lines[0] == "":
        lines = lines[1:]
    first = 1
    try:
        first = int(o.get("firstnumber", 1))
    except (TypeError, ValueError):
        pass
    basic = str(o.get("basicstyle", ""))
    e.add_block({"t": "listing", "lines": [_highlight(x, lang, o) for x in lines], "lang": lang,
                 "numbers": o.get("numbers") if isinstance(o.get("numbers"), str) else None,
                 "frame": o.get("frame") if isinstance(o.get("frame"), str) else None, "first": first,
                 "tt": "ttfamily" in basic, "size": next((s2 for s2 in ("small", "footnotesize", "scriptsize", "tiny")
                                                          if "\\" + s2 in basic), None), "caption": caption})


def _highlight(line, lang, o):
    """A source line as (kind, text) spans: kw (bold), comment (italic), string, plain."""
    kws = KEYWORDS.get(lang, set())
    com = COMMENT.get(lang)
    spans = []
    i = 0
    while i < len(line):
        if com and line.startswith(com, i):
            spans.append(("comment", line[i:]))
            break
        c = line[i]
        if c in "\"'" and lang:
            j = line.find(c, i + 1)
            j = len(line) - 1 if j < 0 else j
            spans.append(("string", line[i:j + 1]))
            i = j + 1
            continue
        m = re.match(r"[A-Za-z_]\w*", line[i:])
        if m:
            w = m.group(0)
            spans.append(("kw" if w in kws else "plain", w))
            i += len(w)
            continue
        spans.append(("plain", c))
        i += 1
    merged = []
    for k, s in spans:
        if merged and merged[-1][0] == k == "plain":
            merged[-1] = ("plain", merged[-1][1] + s)
        else:
            merged.append((k, s))
    return merged


@env("minted")
def minted_env(e, name, t):
    from .features import take_verbatim
    take_verbatim(e, name)
    e.error("Package minted Error: You must invoke LaTeX with the -shell-escape flag.", t,
            "minted runs Pygments, so it needs pdflatex -shell-escape (Overleaf allows it). Here, use listings.")


@cmd("lstinline")
def lstinline(e, t):
    e.read_opt()
    nt = e.next()
    if nt is None:
        return
    if nt.k == "bg":
        body = e._group_body(nt, "lstinline")
        e.add_inline({"t": "verb", "s": "".join(x.text() for x in body), "st": e.style})
        return
    delim = nt.v
    out = []
    while True:
        u = e.next()
        if u is None or (u.k != "sp" and u.text() == delim):
            break
        out.append(u.text())
    e.add_inline({"t": "verb", "s": "".join(out), "st": e.style})

"""printf(1)/bash printf: format reuse, %b %q %(fmt)T, C-style numeric conversions and escapes."""

from __future__ import annotations

import math
import re

from .quoting import backslash_quote

SPEC = re.compile(r"%([-+ #0']*)(\*|\d+)?(?:\.(\*|\d*))?(\([^)]*\)T|[diouxXeEfFgGaAcsbq%])")


def escapes(s: str, echo: bool = False) -> tuple[str, bool]:
    """Backslash escapes for printf formats (echo=False) or echo -e / %b (echo=True). -> (text, stop)."""
    out, i, n = [], 0, len(s)
    simple = {"a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
              "\\": "\\"}
    if not echo:
        simple.update({'"': '"', "'": "'", "?": "?"})
    while i < n:
        c = s[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        d = s[i + 1]
        if d in simple:
            out.append(simple[d])
            i += 2
        elif d == "c" and echo:
            return "".join(out), True
        elif echo and d == "0" or (not echo and d in "01234567"):
            j = i + 2 if echo else i + 1
            k = j
            while k < n and k < j + 3 and s[k] in "01234567":
                k += 1
            out.append(chr(int(s[j:k] or "0", 8) & 0xFF))
            i = k
        elif d == "x":
            k = i + 2
            while k < n and k < i + 4 and s[k] in "0123456789abcdefABCDEF":
                k += 1
            if k == i + 2:
                out.append("\\x")
                i += 2
            else:
                out.append(chr(int(s[i + 2:k], 16)))
                i = k
        elif d in "uU":
            lim = 4 if d == "u" else 8
            k = i + 2
            while k < n and k < i + 2 + lim and s[k] in "0123456789abcdefABCDEF":
                k += 1
            if k == i + 2:
                out.append("\\" + d)
                i += 2
            else:
                out.append(chr(int(s[i + 2:k], 16)))
                i = k
        else:
            out.append("\\" + d)
            i += 2
    return "".join(out), False


def parse_int(arg: str):
    """-> (value, ok). Accepts 'c, "c, 0x.., 0.., +/-, surrounding blanks."""
    s = arg.strip(" \t\n")
    if arg[:1] in ("'", '"'):
        return (ord(arg[1]) if len(arg) > 1 else 0), True
    m = re.match(r"[+-]?(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", s)
    if not m:
        return 0, s == "" and False
    t = m.group(0)
    sign = -1 if t.startswith("-") else 1
    body = t.lstrip("+-")
    if body[:2].lower() == "0x":
        v = int(body[2:], 16)
    elif len(body) > 1 and body[0] == "0":
        v = int(body, 8)
    else:
        v = int(body)
    return sign * v, m.end() == len(s) and arg.strip(" \t\n") == s and not arg.endswith((" ", "\t"))


def parse_float(arg: str):
    s = arg.strip(" \t\n")
    if arg[:1] in ("'", '"'):
        return float(ord(arg[1]) if len(arg) > 1 else 0), True
    m = re.match(r"[+-]?(inf(inity)?|nan|0[xX][0-9a-fA-F.]+(p[+-]?\d+)?|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)", s, re.I)
    if not m:
        return 0.0, False
    t = m.group(0)
    try:
        v = float.fromhex(t) if "x" in t.lower() else float(t)
    except ValueError:
        v = 0.0
    return v, m.end() == len(s)


def conv(flags: str, width, prec, ch: str, arg, errors: list, name="printf") -> str:
    flags = flags.replace("'", "")
    w = "" if width is None else str(width)
    p = "" if prec is None else "." + str(prec)
    if ch in "diouxX":
        if arg is None:
            v = 0
        else:
            v, ok = parse_int(arg)
            if not ok:
                errors.append(f"{name}: {arg}: invalid number")
        if ch == "u" and v < 0:
            v += 1 << 64
        if ch in "oxX" and v < 0:
            v += 1 << 64
        c = "d" if ch in "iu" else ch
        if ch == "o" and "#" in flags:
            body = "%" + flags.replace("#", "") + w + p + "o"
            s = body % v
            if not s.strip().startswith("0"):
                s = ("%" + flags.replace("#", "") + w + "s") % ("0" + (("%" + p + "o") % v if p else "%o" % v))
            return s
        return ("%" + flags + w + p + c) % v
    if ch in "eEfFgGaA":
        if arg is None:
            v = 0.0
        else:
            v, ok = parse_float(arg)
            if not ok:
                errors.append(f"{name}: {arg}: invalid number")
        if ch in "aA":
            s = float(v).hex()
            return s.upper() if ch == "A" else s
        if math.isinf(v) or math.isnan(v):
            word = ("inf" if math.isinf(v) else "nan")
            if ch.isupper():
                word = word.upper()
            sign = "-" if v < 0 else ("+" if "+" in flags else (" " if " " in flags else ""))
            return ("%" + flags.replace("0", "") + w + "s") % (sign + word)
        return ("%" + flags + w + p + ch) % v
    if ch == "c":
        s = (arg or "")[:1]
        return ("%" + flags.replace("0", "") + w + "s") % s
    if ch == "s":
        return ("%" + flags.replace("0", "") + w + p + "s") % (arg or "")
    raise ValueError(ch)


def format_printf(fmt: str, args: list[str], strftime=None, name="printf"):
    """-> (output, errors: list[str], stopped). Reuses the format while arguments remain."""
    out, errors = [], []
    args = list(args)
    first = True
    fmt_text, _ = fmt, None
    while first or args:
        consumed = False
        i, n = 0, len(fmt)
        while i < n:
            c = fmt[i]
            if c == "\\":
                # one escape at a time
                j = i + 2
                if i + 1 < n and fmt[i + 1] in "01234567":
                    j = i + 1
                    while j < n and j < i + 4 and fmt[j] in "01234567":
                        j += 1
                elif i + 1 < n and fmt[i + 1] == "x":
                    j = i + 2
                    while j < n and j < i + 4 and fmt[j] in "0123456789abcdefABCDEF":
                        j += 1
                elif i + 1 < n and fmt[i + 1] in "uU":
                    lim = 4 if fmt[i + 1] == "u" else 8
                    j = i + 2
                    while j < n and j < i + 2 + lim and fmt[j] in "0123456789abcdefABCDEF":
                        j += 1
                text, _stop = escapes(fmt[i:j])
                out.append(text)
                i = j
                continue
            if c != "%":
                out.append(c)
                i += 1
                continue
            m = SPEC.match(fmt, i)
            if not m:
                bad = fmt[i + 1:i + 2] or ""
                j = i + 1
                while j < n and fmt[j] in "-+ #0'0123456789.*":
                    j += 1
                bad = fmt[i:j + 1]
                errors.append(f"{name}: `{bad[-1:] if bad else '%'}': invalid format character")
                return "".join(out), errors, True
            flags, width, prec, ch = m.group(1), m.group(2), m.group(3), m.group(4)
            i = m.end()
            if ch == "%":
                out.append("%")
                continue
            if width == "*":
                a = args.pop(0) if args else None
                consumed = True
                width = parse_int(a)[0] if a is not None else 0
                if width < 0:
                    flags += "-"
                    width = -width
            elif width is not None:
                width = int(width)
            if prec == "*":
                a = args.pop(0) if args else None
                consumed = True
                prec = parse_int(a)[0] if a is not None else 0
            elif prec == "":
                prec = 0
            elif prec is not None:
                prec = int(prec)
            arg = args.pop(0) if args else None
            if arg is not None:
                consumed = True
            if ch == "b":
                text, stop = escapes(arg or "", echo=True)
                if prec is not None:
                    text = text[:prec]
                out.append(("%" + flags.replace("0", "") + (str(width) if width else "") + "s") % text)
                if stop:
                    return "".join(out), errors, True
                continue
            if ch == "q":
                q = backslash_quote(arg or "") if arg is not None else "''"
                out.append(("%" + flags.replace("0", "") + (str(width) if width else "") + "s") % q)
                continue
            if ch.endswith("T"):
                tfmt = ch[1:-2]
                v = parse_int(arg)[0] if arg not in (None, "") else -1
                text = strftime(tfmt, v) if strftime else ""
                if prec is not None:
                    text = text[:prec]
                out.append(("%" + flags.replace("0", "") + (str(width) if width else "") + "s") % text)
                continue
            out.append(conv(flags, width, prec, ch, arg, errors, name))
        first = False
        if not consumed:
            break
    return "".join(out), errors, False

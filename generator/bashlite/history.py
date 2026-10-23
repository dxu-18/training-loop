"""Interactive history expansion: !! !n !-n !str !?str? !$ !^ !* !:n-m, modifiers :h :t :r :e :p :s/a/b/ :gs, ^a^b."""

from __future__ import annotations

import re

META = set("|&;<>()")


def tokenize(line: str) -> list[str]:
    """Split a history entry into words like bash's history_tokenize (quotes kept, metacharacters separate)."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t\n":
            i += 1
            continue
        if c in META:
            j = i + 1
            while j < n and line[j] in META and line[i:j + 1] in ("&&", "||", ">>", "<<", ">&", "<&", "|&", ";;", "2>"):
                j += 1
            out.append(line[i:j])
            i = j
            continue
        j = i
        while j < n and line[j] not in " \t\n" and line[j] not in META:
            if line[j] in "'\"":
                q = line[j]
                k = line.find(q, j + 1)
                j = n if k < 0 else k + 1
                continue
            if line[j] == "\\":
                j += 2
                continue
            j += 1
        out.append(line[i:j])
        i = j
    return out


class HistError(Exception):
    pass


def _word_range(spec: str, nwords: int):
    """'$' '^' '*' 'n' 'n-m' 'n*' 'n-' '-m' -> (start, end) inclusive word indices."""
    last = nwords - 1
    if spec == "$":
        return last, last
    if spec == "^":
        return 1, 1
    if spec == "*":
        return 1, last
    m = re.fullmatch(r"(\d+|\^|\$)?(-|\*)?(\d+|\$)?", spec)
    if not m:
        raise HistError("bad word specifier")
    a = m.group(1)
    a = 1 if a == "^" else (last if a == "$" else (int(a) if a else 0))
    if m.group(2) == "*":
        return a, last
    if m.group(2) == "-":
        b = m.group(3)
        b = last if b == "$" else (int(b) if b else last - 1)
        return a, b
    return a, a


def expand(line: str, history: list[str], quick=True):
    """-> (new_line, changed, print_only). Raises HistError(message)."""
    if quick and line.startswith("^"):
        parts = line[1:].split("^")
        if len(parts) >= 2:
            old, new = parts[0], parts[1]
            if not history:
                raise HistError("!!: event not found")
            prev = history[-1]
            if old not in prev:
                raise HistError(f":s^{old}^{new}: substitution failed" if False else "^" + old + "^" + new + ": substitution failed")
            rest = "^".join(parts[2:])
            return prev.replace(old, new, 1) + rest, True, False
    out = []
    i, n = 0, len(line)
    changed = False
    print_only = False
    in_sq = False
    while i < n:
        c = line[i]
        if c == "'" and not in_sq:
            # single quotes protect from expansion (only outside double quotes)
            dq = line[:i].count('"') % 2 == 1
            if not dq:
                in_sq = True
                out.append(c)
                i += 1
                continue
        elif c == "'" and in_sq:
            in_sq = False
            out.append(c)
            i += 1
            continue
        if in_sq or c != "!":
            if c == "\\" and i + 1 < n and line[i + 1] == "!":
                out.append("\\!")   # bash keeps the backslash for !
                i += 2
                continue
            out.append(c)
            i += 1
            continue
        nxt = line[i + 1] if i + 1 < n else ""
        prev = line[i - 1] if i else ""
        # bash's inhibitions: $! (last background pid), ${!name} (indirection), [!...] (negated bracket)
        if prev == "$" or prev == "[" or (prev == "{" and i > 1 and line[i - 2] == "$" and "}" in line[i + 1:]):
            out.append("!")
            i += 1
            continue
        if nxt in ("", " ", "\t", "\n", "=", "("):
            out.append("!")
            i += 1
            continue
        if nxt == '"' and line[:i].count('"') % 2 == 1:
            out.append("!")
            i += 1
            continue
        # event designator
        j = i + 1
        event = None
        spec_text = ""
        if nxt == "!":
            event = history[-1] if history else None
            spec_text = "!!"
            j = i + 2
        elif nxt == "#":
            event = "".join(out)
            spec_text = "!#"
            j = i + 2
        elif nxt.isdigit() or (nxt == "-" and i + 2 < n and line[i + 2].isdigit()):
            m = re.match(r"-?\d+", line[i + 1:])
            num = int(m.group(0))
            j = i + 1 + m.end()
            spec_text = "!" + m.group(0)
            idx = num - 1 if num > 0 else len(history) + num
            event = history[idx] if 0 <= idx < len(history) else None
        elif nxt == "?":
            k = line.find("?", i + 2)
            sub = line[i + 2:k if k >= 0 else n]
            j = (k + 1) if k >= 0 else n
            spec_text = "!?" + sub + ("?" if k >= 0 else "")
            event = next((h for h in reversed(history) if sub in h), None)
        elif nxt in "$^*:":
            event = history[-1] if history else None
            spec_text = "!"
            j = i + 1
        else:
            m = re.match(r"[^\s:;&|<>()\"'!]+", line[i + 1:])
            prefix = m.group(0) if m else ""
            j = i + 1 + len(prefix)
            spec_text = "!" + prefix
            event = next((h for h in reversed(history) if h.startswith(prefix)), None)
        if event is None:
            raise HistError(f"{spec_text}: event not found")
        words = tokenize(event)
        text = event
        # word designator
        if j < n and (line[j] == ":" or (line[j] in "$^*" and spec_text != "!#")) or spec_text == "!":
            if j < n and line[j] == ":":
                m = re.match(r"(\d+(?:-\d*|\*)?|-\d+|\$|\^|\*|\d*-\$?|\^-\$?)", line[j + 1:])
                if m and m.group(0):
                    spec = m.group(0)
                    j = j + 1 + len(spec)
                else:
                    spec = None
            elif j < n and line[j] in "$^*":
                spec = line[j]
                j += 1
            else:
                spec = None
            if spec is not None:
                a, b = _word_range(spec, len(words))
                if a > len(words) - 1 or b > len(words) - 1 or a < 0:
                    if spec == "*" and len(words) == 1:
                        text = ""
                    else:
                        raise HistError(f"{spec_text}{':' if spec not in '$^*' else ''}{spec}: bad word specifier")
                else:
                    text = " ".join(words[a:b + 1])
        # modifiers
        while j < n and line[j] == ":":
            mod = line[j + 1:j + 2]
            if mod in ("h", "t", "r", "e", "p", "q", "x", "&", "u", "l"):
                j += 2
                if mod == "h":
                    text = text.rsplit("/", 1)[0] if "/" in text else text
                elif mod == "t":
                    text = text.rsplit("/", 1)[-1]
                elif mod == "r":
                    base = text.rsplit("/", 1)[-1]
                    if "." in base:
                        text = text[:len(text) - (len(base) - base.rindex("."))]
                elif mod == "e":
                    base = text.rsplit("/", 1)[-1]
                    text = base[base.rindex("."):] if "." in base else ""
                elif mod == "p":
                    print_only = True
                elif mod == "q":
                    text = "'" + text.replace("'", "'\\''") + "'"
                elif mod == "&":
                    pass
                continue
            if mod in ("s", "g", "G") or line[j + 1:j + 3] == "gs":
                glob = False
                k = j + 1
                if line[k] in "gG":
                    glob = True
                    k += 1
                if k >= n or line[k] != "s":
                    break
                delim = line[k + 1] if k + 1 < n else "/"
                parts = []
                p = k + 2
                cur = []
                while p < n and len(parts) < 2:
                    if line[p] == "\\" and p + 1 < n and line[p + 1] == delim:
                        cur.append(delim)
                        p += 2
                        continue
                    if line[p] == delim:
                        parts.append("".join(cur))
                        cur = []
                        p += 1
                        continue
                    cur.append(line[p])
                    p += 1
                if len(parts) < 2:
                    parts.append("".join(cur))
                old, new = parts[0], parts[1].replace("&", parts[0]) if len(parts) > 1 else ""
                if old not in text:
                    raise HistError(f":s{delim}{old}{delim}{new}{delim}: substitution failed")
                text = text.replace(old, new) if glob else text.replace(old, new, 1)
                j = p
                continue
            break
        out.append(text)
        changed = True
        i = j
    return "".join(out), changed, print_only


def join_lines(lines: list[str]) -> str:
    """One history entry for a multi-line command (cmdhist)."""
    out = lines[0]
    for ln in lines[1:]:
        prev = out.rstrip()
        if prev.endswith(("then", "do", "else", "{", "|", "&&", "||", "(", ";", "in", "&")) or not prev:
            out = prev + " " + ln.strip()
        else:
            out = prev + "; " + ln.strip()
    return out

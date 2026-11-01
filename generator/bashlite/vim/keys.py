"""Keys inside Vim: one token per key press.

A printable key is its character ("a", " ", "$"); every other key is Vim's notation: "<Esc>", "<CR>", "<BS>",
"<Tab>", "<C-a>", "<Up>" ... Registers hold keys as text (a macro is text), so each special key also has a character:
the control characters themselves (<Esc> is "\\x1b", <CR> is "\\r", <C-a> is "\\x01"), and private-use characters for
keys that have none (arrows, <Del>, <Home> ...), which :registers shows the way Vim does (<80>ku).
"""

from __future__ import annotations

import re

# keys with a control character of their own
CTRL_CHAR = {"<Esc>": "\x1b", "<CR>": "\r", "<NL>": "\n", "<Tab>": "\t", "<BS>": "\x08", "<Nul>": "\x00",
             "<C-\\>": "\x1c", "<C-]>": "\x1d", "<C-^>": "\x1e", "<C-_>": "\x1f"}
for _i in range(1, 27):
    _c = chr(96 + _i)
    _k = f"<C-{_c}>"
    if chr(_i) not in CTRL_CHAR.values():
        CTRL_CHAR[_k] = chr(_i)
# keys without one: private-use characters, shown as Vim's internal codes
SPECIAL_CHAR = {"<Up>": "", "<Down>": "", "<Left>": "", "<Right>": "", "<Home>": "",
                "<End>": "", "<PageUp>": "", "<PageDown>": "", "<Del>": "",
                "<S-Tab>": "", "<Insert>": ""}
SPECIAL_SHOWN = {"": "<80>ku", "": "<80>kd", "": "<80>kl", "": "<80>kr", "": "<80>kh",
                 "": "<80>@7", "": "<80>kP", "": "<80>kN", "": "<80>kD", "": "<80>kB",
                 "": "<80>kI"}
CHAR_KEY = {v: k for k, v in {**CTRL_CHAR, **SPECIAL_CHAR}.items()}

# other spellings of the same key
ALIASES = {"<esc>": "<Esc>", "<c-[>": "<Esc>", "<cr>": "<CR>", "<enter>": "<CR>", "<return>": "<CR>", "<c-m>": "<CR>",
           "<nl>": "<NL>", "<c-j>": "<NL>", "<tab>": "<Tab>", "<c-i>": "<Tab>", "<bs>": "<BS>", "<backspace>": "<BS>",
           "<c-h>": "<BS>", "<space>": " ", "<lt>": "<", "<bar>": "|", "<bslash>": "\\", "<del>": "<Del>",
           "<delete>": "<Del>", "<up>": "<Up>", "<down>": "<Down>", "<left>": "<Left>", "<right>": "<Right>",
           "<home>": "<Home>", "<end>": "<End>", "<pageup>": "<PageUp>", "<pagedown>": "<PageDown>",
           "<s-tab>": "<S-Tab>", "<insert>": "<Insert>", "<nul>": "<Nul>", "<c-@>": "<Nul>", "<c-space>": "<Nul>",
           "<c-6>": "<C-^>", "<c-^>": "<C-^>", "<c-]>": "<C-]>", "<c-\\>": "<C-\\>", "<c-_>": "<C-_>"}

# the page's key names (docs/terminal.js parseInput) -> tokens
PAGE = {"Esc": "<Esc>", "Enter": "<CR>", "C-m": "<CR>", "C-j": "<NL>", "BSpace": "<BS>", "C-h": "<BS>",
        "Tab": "<Tab>", "Space": " ", "Up": "<Up>", "Down": "<Down>", "Left": "<Left>", "Right": "<Right>",
        "Home": "<Home>", "End": "<End>", "PgUp": "<PageUp>", "PgDn": "<PageDown>", "Delete": "<Del>",
        "C-Space": "<Nul>", "C-[": "<Esc>", "C-^": "<C-^>", "C-]": "<C-]>", "C-\\": "<C-\\>", "C-_": "<C-_>",
        "S-Tab": "<S-Tab>", "C-i": "<Tab>"}

_NOTATION = re.compile(r"<[^<>\s]+>")


def norm(tok: str) -> str:
    """One spelling per key: "<c-a>" -> "<C-a>", "<esc>" -> "<Esc>"."""
    low = tok.lower()
    if low in ALIASES:
        return ALIASES[low]
    m = re.fullmatch(r"<(?:c|ctrl)-(.)>", tok, re.I)
    if m:
        c = m.group(1).lower()
        return ALIASES.get(f"<c-{c}>", f"<C-{c}>")
    m = re.fullmatch(r"<(?:m|a|alt|meta)-(.)>", tok, re.I)
    if m:
        return f"<M-{m.group(1)}>"
    for k in list(CTRL_CHAR) + list(SPECIAL_CHAR):
        if k.lower() == low:
            return k
    return tok


def parse(notation: str, leader: str = "\\") -> list[str]:
    """Vim notation -> tokens: "ciwfoo<Esc>" -> ["c", "i", "w", "f", "o", "o", "<Esc>"]. <leader> is expanded."""
    out = []
    i = 0
    while i < len(notation):
        if notation[i] == "<":
            m = _NOTATION.match(notation, i)
            if m:
                tok = m.group(0)
                if tok.lower() == "<leader>":
                    out.extend(parse(leader, "\\"))
                    i = m.end()
                    continue
                n = norm(tok)
                if n != tok or n in CTRL_CHAR or n in SPECIAL_CHAR or re.fullmatch(r"<M-.>", n):
                    out.extend(split_meta(n))
                    i = m.end()
                    continue
        out.append(notation[i])
        i += 1
    return out


def split_meta(tok: str) -> list[str]:
    """Alt-x in a terminal is Esc then x."""
    m = re.fullmatch(r"<M-(.)>", tok)
    return ["<Esc>", m.group(1)] if m else [tok]


def to_text(tokens) -> str:
    """Tokens -> register text (a macro)."""
    out = []
    for t in tokens:
        if len(t) == 1:
            out.append(t)
        elif t in CTRL_CHAR:
            out.append(CTRL_CHAR[t])
        elif t in SPECIAL_CHAR:
            out.append(SPECIAL_CHAR[t])
    return "".join(out)


def from_text(text: str) -> list[str]:
    """Register text -> tokens (playing a macro)."""
    return [CHAR_KEY.get(c, c) for c in text]


def shown(text: str) -> str:
    """How Vim shows special characters in :registers and on screen: ^[ ^M <80>ku."""
    out = []
    for c in text:
        if c in SPECIAL_SHOWN:
            out.append(SPECIAL_SHOWN[c])
        elif c == "\n":
            out.append("^J")
        elif ord(c) < 32:
            out.append("^" + chr(ord(c) + 64))
        elif c == "\x7f":
            out.append("^?")
        else:
            out.append(c)
    return "".join(out)


def to_notation(tokens) -> str:
    """Tokens -> notation (a solution shown on the page)."""
    return "".join(t if len(t) == 1 and t != "<" else ("<lt>" if t == "<" else t) for t in tokens)


def from_page(kind: str, val: str) -> list[str]:
    """One page event (("text", "abc") or ("key", "C-a")) -> tokens."""
    if kind == "text":
        return list(val)
    if val in PAGE:
        return [PAGE[val]]
    m = re.fullmatch(r"C-(.)", val)
    if m:
        return [norm(f"<C-{m.group(1)}>")]
    m = re.fullmatch(r"M-(.+)", val)
    if m:
        rest = m.group(1)
        return ["<Esc>"] + (list(rest) if len(rest) == 1 else from_page("key", rest))
    if len(val) == 1:
        return [val]
    return []


def to_page(tokens) -> list[tuple[str, str]]:
    """Tokens -> page events (a reference solution typed into the terminal)."""
    back = {"<Esc>": "Esc", "<CR>": "Enter", "<NL>": "C-j", "<BS>": "BSpace", "<Tab>": "Tab", "<Up>": "Up",
            "<Down>": "Down", "<Left>": "Left", "<Right>": "Right", "<Home>": "Home", "<End>": "End",
            "<PageUp>": "PgUp", "<PageDown>": "PgDn", "<Del>": "Delete", "<Nul>": "C-Space", "<C-^>": "C-^",
            "<C-]>": "C-]", "<C-\\>": "C-\\", "<C-_>": "C-_", "<S-Tab>": "S-Tab"}
    out: list[tuple[str, str]] = []
    for t in tokens:
        if len(t) == 1:
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1] + t)
            else:
                out.append(("text", t))
        elif t in back:
            out.append(("key", back[t]))
        else:
            m = re.fullmatch(r"<C-(.)>", t)
            if m:
                out.append(("key", f"C-{m.group(1)}"))
    return out


def count_keys(tokens) -> int:
    return len(tokens)

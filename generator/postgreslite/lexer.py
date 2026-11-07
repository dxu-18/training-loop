"""The SQL lexer, after PostgreSQL 16's scan.l: identifiers (folded to lower case), quoted identifiers, '' / E'' /
$tag$ strings, numbers (int4 / int8 / numeric, 0x / 0o / 0b, underscores), operators (with scan.l's rules for
trailing + and -), comments (nested /* */), parameters. Every token keeps its character offset for error positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import PgError

# kwlist.h (PostgreSQL 16): category letter (R reserved, C col_name, T type_func_name, U unreserved), then a for
# AS_LABEL (needs AS before it as a column alias), b for BARE_LABEL
_KW = {
    "Ra": """array as create except fetch for from grant group having intersect into limit offset on order returning to
union where window with""",
    "Rb": """all analyse analyze and any asc asymmetric both case cast check collate column constraint current_catalog
current_date current_role current_time current_timestamp current_user default deferrable desc distinct do else
end false foreign in initially lateral leading localtime localtimestamp not null only or placing primary
references select session_user some symmetric system_user table then trailing true unique user using variadic
when""",
    "Ca": """char character precision""",
    "Cb": """between bigint bit boolean coalesce dec decimal exists extract float greatest grouping inout int integer
interval json_array json_arrayagg json_object json_objectagg least national nchar none normalize nullif
numeric out overlay position real row setof smallint substring time timestamp treat trim values varchar
xmlattributes xmlconcat xmlelement xmlexists xmlforest xmlnamespaces xmlparse xmlpi xmlroot xmlserialize
xmltable""",
    "Ta": """isnull notnull overlaps""",
    "Tb": """authorization binary collation concurrently cross current_schema freeze full ilike inner is join left like
natural outer right similar tablesample verbose""",
    "Ua": """day filter hour minute month over second varying within without year""",
    "Ub": """abort absent absolute access action add admin after aggregate also alter always asensitive assertion
assignment at atomic attach attribute backward before begin breadth by cache call called cascade cascaded
catalog chain characteristics checkpoint class close cluster columns comment comments commit committed
compression configuration conflict connection constraints content continue conversion copy cost csv cube
current cursor cycle data database deallocate declare defaults deferred definer delete delimiter delimiters
depends depth detach dictionary disable discard document domain double drop each enable encoding encrypted
enum escape event exclude excluding exclusive execute explain expression extension external family finalize
first following force format forward function functions generated global granted groups handler header hold
identity if immediate immutable implicit import include including increment indent index indexes inherit
inherits inline input insensitive insert instead invoker isolation json key keys label language large last
leakproof level listen load local location lock locked logged mapping match matched materialized maxvalue
merge method minvalue mode move name names new next nfc nfd nfkc nfkd no normalized nothing notify nowait
nulls object of off oids old operator option options ordinality others overriding owned owner parallel
parameter parser partial partition passing password plans policy preceding prepare prepared preserve prior
privileges procedural procedure procedures program publication quote range read reassign recheck recursive ref
referencing refresh reindex relative release rename repeatable replace replica reset restart restrict return
returns revoke role rollback rollup routine routines rows rule savepoint scalar schema schemas scroll search
security sequence sequences serializable server session set sets share show simple skip snapshot sql stable
standalone start statement statistics stdin stdout storage stored strict strip subscription support sysid
system tables tablespace temp template temporary text ties transaction transform trigger truncate trusted type
types uescape unbounded uncommitted unencrypted unknown unlisten unlogged until update vacuum valid validate
validator value version view views volatile whitespace work wrapper write xml yes zone""",
}
KEYWORDS: dict[str, str] = {}      # word -> category letter
AS_LABEL: set[str] = set()         # keywords that need AS to be a column alias
for _k, _words in _KW.items():
    for _w in _words.split():
        KEYWORDS[_w] = _k[0]
        if _k[1] == "a":
            AS_LABEL.add(_w)

OP_CHARS = set("~!@#^&|`?+-*/%<>=")
SELF_CHARS = set(",()[].;:+-*/%^<>=")
_IDENT_START = re.compile(r"[A-Za-z_\u0080-\U0010ffff]")
_IDENT = re.compile(r"[A-Za-z_\u0080-\U0010ffff][A-Za-z_0-9$\u0080-\U0010ffff]*")
_DOLLAR = re.compile(r"\$([A-Za-z_\u0080-\U0010ffff][A-Za-z_0-9\u0080-\U0010ffff]*)?\$")
_PARAM = re.compile(r"\$(\d+)")
_DEC = r"\d(?:_?\d)*"
_NUMBER = re.compile(
    rf"(?P<hex>0[xX](?:_?[0-9A-Fa-f])+)|(?P<oct>0[oO](?:_?[0-7])+)|(?P<bin>0[bB](?:_?[01])+)"
    rf"|(?P<real>(?:{_DEC}(?:\.(?:{_DEC})?)?|\.{_DEC})[Ee][-+]?{_DEC})"
    rf"|(?P<num>{_DEC}\.(?:{_DEC})?|\.{_DEC})"
    rf"|(?P<int>{_DEC})")
_NAMEDATALEN = 63


@dataclass(slots=True)
class Token:
    kind: str        # IDENT KW QIDENT STR INT NUM PARAM OP SELF EOF
    value: object    # folded name / keyword / string contents / number text / operator
    pos: int         # offset of the first character
    text: str        # the source text of the token
    end: int = 0     # offset after the last character

    def is_kw(self, *words) -> bool:
        return self.kind == "KW" and self.value in words

    def is_self(self, ch) -> bool:
        return self.kind == "SELF" and self.value == ch


def downcase(name: str) -> str:
    """PostgreSQL folds unquoted identifiers: only ASCII letters change case."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in name)


def _near(sql: str, pos: int) -> str:
    return sql[pos:].split("\n")[0] if pos < len(sql) else ""


def _estring(body: str, pos: int) -> str:
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        i += 1
        if i >= len(body):
            break
        c = body[i]
        simple = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
        if c in simple:
            out.append(simple[c])
            i += 1
        elif c in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i:])
            out.append(chr(int(m.group(0), 8)))
            i += len(m.group(0))
        elif c == "x" and i + 1 < len(body) and body[i + 1] in "0123456789abcdefABCDEF":
            m = re.match(r"[0-9a-fA-F]{1,2}", body[i + 1:])
            out.append(chr(int(m.group(0), 16)))
            i += 1 + len(m.group(0))
        elif c in "uU":
            n = 4 if c == "u" else 8
            hexs = body[i + 1:i + 1 + n]
            if len(hexs) != n or not all(h in "0123456789abcdefABCDEF" for h in hexs):
                raise PgError("invalid Unicode escape", "22025",
                              hint="Unicode escapes must be \\uXXXX or \\UXXXXXXXX.", pos=pos + 1)
            out.append(chr(int(hexs, 16)))
            i += 1 + n
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(sql: str) -> list[Token]:
    """All tokens of `sql`, ending with an EOF token. Raises PgError for unterminated literals and the like."""
    toks: list[Token] = []
    n = len(sql)
    i = 0
    while True:
        # whitespace and comments
        while i < n:
            c = sql[i]
            if c in " \t\n\r\f\v":
                i += 1
            elif sql.startswith("--", i):
                j = sql.find("\n", i)
                i = n if j < 0 else j + 1
            elif sql.startswith("/*", i):
                depth, j = 1, i + 2
                while j < n and depth:
                    if sql.startswith("/*", j):
                        depth += 1
                        j += 2
                    elif sql.startswith("*/", j):
                        depth -= 1
                        j += 2
                    else:
                        j += 1
                if depth:
                    raise PgError(f'unterminated /* comment at or near "{_near(sql, i)}"', "42601", pos=i + 1)
                i = j
            else:
                break
        if i >= n:
            toks.append(Token("EOF", None, n, "", n))
            return toks
        c = sql[i]
        start = i
        # strings: '...', E'...', N'...', B'...', X'...'
        prefix = ""
        if c in "eEnNbBxX" and i + 1 < n and sql[i + 1] == "'":
            prefix = c.upper()
            i += 1
            c = "'"
        elif c in "uU" and sql.startswith("&'", i + 1):
            raise PgError("Unicode escape strings (U&'...') are not supported by postgres-lite", "0A000", pos=i + 1)
        if c == "'":
            parts = []
            while True:
                j = i + 1
                buf = []
                while True:
                    if j >= n:
                        raise PgError(f'unterminated quoted string at or near "{_near(sql, start)}"', "42601",
                                      pos=start + 1)
                    ch = sql[j]
                    if ch == "'":
                        if j + 1 < n and sql[j + 1] == "'":
                            buf.append("'")
                            j += 2
                            continue
                        break
                    if ch == "\\" and prefix == "E" and j + 1 < n:
                        buf.append(sql[j:j + 2])
                        j += 2
                        continue
                    buf.append(ch)
                    j += 1
                parts.append("".join(buf))
                i = j + 1
                # SQL: two literals separated by whitespace with a newline are one literal
                k = i
                saw_nl = False
                while k < n and sql[k] in " \t\n\r\f\v":
                    saw_nl |= sql[k] in "\n\r"
                    k += 1
                if saw_nl and k < n and sql[k] == "'":
                    i = k
                    continue
                break
            body = "".join(parts)
            if prefix == "E":
                body = _estring(body, start)
            kind = {"B": "BITSTR", "X": "BITSTR"}.get(prefix, "STR")
            if kind == "BITSTR":
                body = (prefix.lower()) + body
            toks.append(Token(kind, body, start, sql[start:i], i))
            continue
        # quoted identifier
        if c == '"':
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise PgError(f'unterminated quoted identifier at or near "{_near(sql, start)}"', "42601",
                                  pos=start + 1)
                if sql[j] == '"':
                    if j + 1 < n and sql[j + 1] == '"':
                        buf.append('"')
                        j += 2
                        continue
                    break
                buf.append(sql[j])
                j += 1
            name = "".join(buf)
            if not name:
                raise PgError(f'zero-length delimited identifier at or near """"', "42601", pos=start + 1)
            i = j + 1
            toks.append(Token("QIDENT", name[:_NAMEDATALEN], start, sql[start:i], i))
            continue
        # dollar-quoted strings and parameters
        if c == "$":
            m = _DOLLAR.match(sql, i)
            if m:
                delim = m.group(0)
                j = sql.find(delim, m.end())
                if j < 0:
                    raise PgError(f'unterminated dollar-quoted string at or near "{_near(sql, start)}"', "42601",
                                  pos=start + 1)
                i = j + len(delim)
                toks.append(Token("STR", sql[m.end():j], start, sql[start:i], i))
                continue
            m = _PARAM.match(sql, i)
            if m:
                i = m.end()
                toks.append(Token("PARAM", int(m.group(1)), start, m.group(0), i))
                continue
        # numbers
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            m = _NUMBER.match(sql, i)
            text = m.group(0)
            j = m.end()
            if j < n and _IDENT_START.match(sql[j]):            # 123abc: trailing junk (PG 15+)
                m2 = _IDENT.match(sql, j)
                raise PgError(f'trailing junk after numeric literal at or near "{sql[start:m2.end()]}"', "42601",
                              pos=start + 1)
            i = j
            clean = text.replace("_", "")
            if m.group("hex") or m.group("oct") or m.group("bin"):
                base = {"x": 16, "o": 8, "b": 2}[clean[1].lower()]
                toks.append(Token("INT", str(int(clean[2:], base)), start, text, i))
            elif m.group("real") or m.group("num"):
                toks.append(Token("NUM", clean, start, text, i))
            else:
                toks.append(Token("INT", clean, start, text, i))
            continue
        # identifiers and keywords
        if _IDENT_START.match(c):
            m = _IDENT.match(sql, i)
            i = m.end()
            word = downcase(m.group(0))
            if word in KEYWORDS:
                toks.append(Token("KW", word, start, m.group(0), i))
            else:
                toks.append(Token("IDENT", word[:_NAMEDATALEN], start, m.group(0), i))
            continue
        # :: .. :=
        if sql.startswith("::", i):
            i += 2
            toks.append(Token("OP", "::", start, "::", i))
            continue
        if sql.startswith("..", i):
            i += 2
            toks.append(Token("OP", "..", start, "..", i))
            continue
        if sql.startswith(":=", i):
            i += 2
            toks.append(Token("OP", ":=", start, ":=", i))
            continue
        # operators
        if c in OP_CHARS:
            j = i
            while j < n and sql[j] in OP_CHARS:
                j += 1
            op = sql[i:j]
            for cut in ("/*", "--"):
                k = op.find(cut, 1) if op.startswith(cut) is False else -1
                if k > 0:
                    op = op[:k]
            if len(op) > 1 and op[-1] in "+-":
                if not any(ch in "~!@#^&|`?%" for ch in op):
                    while len(op) > 1 and op[-1] in "+-":
                        op = op[:-1]
            i += len(op)
            if len(op) == 1 and op in SELF_CHARS:
                toks.append(Token("SELF", op, start, op, i))
            else:
                toks.append(Token("OP", "<>" if op == "!=" else op, start, op, i))
            continue
        if c in SELF_CHARS:
            i += 1
            toks.append(Token("SELF", c, start, c, i))
            continue
        raise PgError(f'syntax error at or near "{c}"', "42601", pos=start + 1)

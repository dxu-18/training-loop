"""BibTeX: .bib files, the standard styles (plain, unsrt, abbrv, alpha, plainnat, unsrtnat, abbrvnat) and
citation labels (numeric, or author-year with natbib)."""

from __future__ import annotations

import re

STYLES = {"plain", "unsrt", "abbrv", "alpha", "plainnat", "unsrtnat", "abbrvnat", "ieeetr", "acm", "siam",
          "apalike", "biblatex"}
ACCENT_MAP = {r"{\L}": "Ł", r"{\l}": "ł", r"\L ": "Ł", r"{\o}": "ø", r"{\O}": "Ø", r"{\ss}": "ß", r"{\ae}": "æ",
              r"{\aa}": "å", r"\'e": "é", r"\'{e}": "é", r'\"o': "ö", r'\"{o}': "ö", r'\"u': "ü", r'\"{u}': "ü",
              r'\"a': "ä", r'\"{a}': "ä", r"\'a": "á", r"\'{a}": "á", r"\`e": "è", r"\`{e}": "è", r"\c{c}": "ç",
              r"\~n": "ñ", r"\~{n}": "ñ", r"\'i": "í", r"\'{\i}": "í", r"\'o": "ó", r"\'{o}": "ó"}


def parse_bib(src: str):
    """-> ({key: entry}, [problems]). An entry: {"type", "key", fields...}."""
    entries, problems = {}, []
    i = 0
    n = len(src)
    while True:
        at = src.find("@", i)
        if at < 0:
            break
        m = re.match(r"@\s*(\w+)\s*([{(])", src[at:])
        if not m:
            i = at + 1
            continue
        typ = m.group(1).lower()
        j = at + m.end()
        # find the matching close
        depth = 1
        k = j
        while k < n and depth:
            if src[k] in "{(":
                depth += 1
            elif src[k] in "})":
                depth -= 1
            k += 1
        body = src[j:k - 1]
        i = k
        if typ in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        key = key.strip()
        if not key:
            problems.append("I was expecting a ',' or a '}'")
            continue
        entry = {"type": typ, "key": key}
        for fname, val in _fields(rest):
            entry[fname.lower()] = val
        if key in entries:
            problems.append(f"Repeated entry---line ? of file: {key}")
        entries[key] = entry
    return entries, problems


def _fields(s):
    i = 0
    n = len(s)
    while i < n:
        m = re.match(r"\s*,?\s*(\w[\w-]*)\s*=\s*", s[i:])
        if not m:
            break
        name = m.group(1)
        i += m.end()
        if i < n and s[i] == "{":
            depth = 0
            k = i
            while k < n:
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            yield name, s[i + 1:k]
            i = k + 1
        elif i < n and s[i] == '"':
            k = s.find('"', i + 1)
            yield name, s[i + 1:k]
            i = k + 1
        else:
            m2 = re.match(r"([\w.-]+)", s[i:])
            if not m2:
                break
            yield name, m2.group(1)
            i += m2.end()


def clean(s: str) -> str:
    for a, b in ACCENT_MAP.items():
        s = s.replace(a, b)
    s = re.sub(r"\\textit\{([^{}]*)\}|\\emph\{([^{}]*)\}", lambda m: m.group(1) or m.group(2), s)
    return s.replace("{", "").replace("}", "").replace("~", "\u00a0").replace("--", "\u2013")


def names(field: str):
    """BibTeX names: "Last, First and First Last" -> [(first, last)]."""
    out = []
    for part in re.split(r"\s+and\s+", field.strip()):
        part = part.strip()
        if not part:
            continue
        if part == "others":
            out.append(("", "others"))
            continue
        if "," in part:
            last, first = part.split(",", 1)
        else:
            bits = part.split()
            last, first = (bits[-1], " ".join(bits[:-1])) if len(bits) > 1 else (part, "")
        out.append((clean(first.strip()), clean(last.strip())))
    return out


def initials(first):
    return " ".join(w[0] + "." for w in re.split(r"[\s~]+", first) if w)


def sentence_case(title: str) -> str:
    """BibTeX's change.case$ "t": lowercase all but the first letter, keeping {braced} parts."""
    out = []
    depth = 0
    first = True
    for c in title:
        if c == "{":
            depth += 1
            continue
        if c == "}":
            depth -= 1
            continue
        if depth == 0 and c.isalpha():
            out.append(c if first else c.lower())
            first = False
        else:
            out.append(c)
            if c.isalpha():
                first = False
    return clean("".join(out))


def load(e, files, t):
    style = e.bib["style"]
    if style is None and e.bib.get("biblatex") is None and "biblatex" not in e.pkg_closure:
        e.warn("BibTeX: I found no \\bibliographystyle command---while reading file main.aux", t)
        e.bib["broken"] = True
        return
    if style not in STYLES and style != "biblatex":
        e.warn(f"BibTeX: I couldn't open style file {style}.bst", t)
        e.bib["broken"] = True
        return
    for f in files:
        fn = f if f.endswith(".bib") else f + ".bib"
        src = e.files.get(fn)
        if src is None:
            e.warn(f"BibTeX: I couldn't open database file {fn}", t)
            continue
        entries, problems = parse_bib(src)
        for p in problems:
            e.warn("BibTeX: " + p, t)
        e.bib["entries"].update(entries)
    e.bib["printed"] = True


def manual_bibliography(e, body, t):
    """\\begin{thebibliography}{9} ... \\bibitem{key} text ..."""
    from .features import split_items
    items = []
    cur = None
    for tok in body:
        if tok.k == "cs" and tok.v == "bibitem":
            cur = {"toks": []}
            items.append(cur)
            continue
        if cur is not None:
            cur["toks"].append(tok)
    out = []
    for it in items:
        toks = it["toks"]
        # [label]{key}
        j = 0
        while j < len(toks) and toks[j].k == "sp":
            j += 1
        if j < len(toks) and toks[j].k == "bg":
            k = j
            while k < len(toks) and toks[k].k != "eg":
                k += 1
            key = "".join(x.text() for x in toks[j + 1:k]).strip()
            toks = toks[k + 1:]
        else:
            key = "?"
        out.append({"key": key, "c": e.capture(toks)})
        e.bib["entries"].setdefault(key, {"key": key, "manual": True})
    e.bib["manual"] = [x["key"] for x in out]
    e.add_block({"t": "bibliography", "manual": out})


# ---------------------------------------------------------------- labels and entries

def author_short(entry):
    ns = names(entry.get("author", entry.get("editor", "")))
    if not ns:
        return clean(entry.get("key", "?"))
    if len(ns) == 1:
        return ns[0][1]
    if len(ns) == 2 and ns[1][1] != "others":
        return f"{ns[0][1]} and {ns[1][1]}"
    return f"{ns[0][1]} et al."


def year(entry):
    return clean(entry.get("year", "????"))


def order(e):
    """The entries the bibliography lists, in order, with their numbers."""
    style = e.bib["style"] or "plain"
    keys = [k for k in e.bib["cited"] if k in e.bib["entries"]]
    if "*" in e.bib["cited"]:
        keys = list(e.bib["entries"])
    if style in ("unsrt", "unsrtnat", "ieeetr"):
        return keys
    def sk(k):
        en = e.bib["entries"][k]
        ns = names(en.get("author", en.get("editor", "")))
        return (" ".join(n[1] for n in ns).lower(), year(en), clean(en.get("title", "")).lower())
    return sorted(keys, key=sk)


def alpha_label(entry):
    """alpha.bst: three letters of a single author's name, else the authors' initials (+ past four), and the year."""
    ns = names(entry.get("author", ""))
    y = year(entry)[-2:]
    if not ns:
        return entry["key"][:3] + y
    if len(ns) == 1:
        return ns[0][1][:3] + y
    if len(ns) > 4:
        return "".join(n[1][:1] for n in ns[:3]) + "+" + y
    return "".join(n[1][:1] for n in ns) + y


def is_authoryear(e):
    style = e.bib["style"] or "plain"
    opts = e.pkgs.get("natbib", [])
    return "natbib" in e.pkg_closure and style in ("plainnat", "unsrtnat", "abbrvnat", "apalike") and \
        "numbers" not in opts


def format_entry(e, entry):
    """The bibliography text of an entry (plain-like styles), as inline runs."""
    from .model import PLAIN, text
    style = e.bib["style"] or "plain"
    abbrv = style in ("abbrv", "abbrvnat", "ieeetr")
    ns = names(entry.get("author", ""))
    def nm(f, l):
        f2 = initials(f) if abbrv else f
        return f"{f2} {l}".strip() if l != "others" else "others"
    if len(ns) == 1:
        who = nm(*ns[0])
    elif len(ns) == 2:
        who = f"{nm(*ns[0])} and {nm(*ns[1])}"
    elif ns:
        who = ", ".join(nm(*x) for x in ns[:-1]) + ", and " + nm(*ns[-1])
    else:
        who = ""
    if ns and ns[-1][1] == "others":
        who = ", ".join(nm(*x) for x in ns[:-1]) + " et al."
    typ = entry.get("type", "misc")
    title = entry.get("title", "")
    title_s = sentence_case(title) if typ in ("article", "inproceedings", "incollection", "misc", "techreport",
                                              "conference", "phdthesis", "mastersthesis", "unpublished") else clean(title)
    it = PLAIN.with_(shape="it")
    out = []
    if who:
        out.append(text(who + ". "))
    if typ == "book":
        out.append(text(clean(title), it))
        out.append(text(". "))
    else:
        out.append(text(title_s + ". "))
    y = year(entry)
    if typ == "article":
        out.append(text(clean(entry.get("journal", "")), it))
        vol = entry.get("volume")
        if vol:
            out.append(text(", " + clean(vol)))
            if entry.get("number"):
                out.append(text(f"({clean(entry['number'])})"))
            if entry.get("pages"):
                out.append(text(":" + clean(entry["pages"])))
        out.append(text(f", {y}."))
    elif typ in ("inproceedings", "incollection", "conference"):
        out.append(text("In "))
        out.append(text(clean(entry.get("booktitle", "")), it))
        if entry.get("pages"):
            out.append(text(", pages " + clean(entry["pages"])))
        out.append(text(f", {y}."))
    elif typ == "book":
        pub = clean(entry.get("publisher", ""))
        out.append(text(f"{pub}, {y}." if pub else f"{y}."))
    else:
        extra = []
        if entry.get("howpublished"):
            extra.append(clean(entry["howpublished"]))
        if entry.get("eprint"):
            pre = entry.get("archiveprefix", entry.get("archivePrefix", "arXiv"))
            extra.append(f"{clean(pre)}:{clean(entry['eprint'])}" if style not in ("plainnat", "unsrtnat", "abbrvnat")
                         else f"{clean(pre)} preprint {clean(pre)}:{clean(entry['eprint'])}")
        if entry.get("institution"):
            extra.append(clean(entry["institution"]))
        extra.append(y)
        out.append(text(", ".join(extra) + "."))
    if entry.get("url") and style in ("plainnat", "unsrtnat", "abbrvnat"):
        out.append(text(" URL "))
        out.append({"t": "link", "url": entry["url"], "c": [text(entry["url"], PLAIN.with_(family="tt"))],
                    "kind": "url"})
        out.append(text("."))
    return out

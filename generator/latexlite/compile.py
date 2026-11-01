"""compile(): source -> {document, html, log, errors, warnings}. References and citations are resolved here, once
for both the page and the grader (all of them: like latexmk, which reruns LaTeX until they settle)."""

from __future__ import annotations

from . import algo, bib, commands, features, floats, tables  # noqa: F401  (they register their handlers)
from . import beamer, pgfplots, tikz  # noqa: F401
from .engine import Engine, Stop
from .tokens import Tok, tokenize

CREF = {"section": ("section", "sections", "Section", "Sections"),
        "subsection": ("section", "sections", "Section", "Sections"),
        "subsubsection": ("section", "sections", "Section", "Sections"),
        "paragraph": ("paragraph", "paragraphs", "Paragraph", "Paragraphs"),
        "chapter": ("chapter", "chapters", "Chapter", "Chapters"),
        "part": ("part", "parts", "Part", "Parts"),
        "appendix": ("appendix", "appendices", "Appendix", "Appendices"),
        "equation": ("eq.", "eqs.", "Equation", "Equations"),
        "figure": ("fig.", "figs.", "Figure", "Figures"), "subfigure": ("fig.", "figs.", "Figure", "Figures"),
        "table": ("table", "tables", "Table", "Tables"), "subtable": ("table", "tables", "Table", "Tables"),
        "item": ("item", "items", "Item", "Items"), "footnote": ("footnote", "footnotes", "Footnote", "Footnotes"),
        "algorithm": ("algorithm", "algorithms", "Algorithm", "Algorithms"),
        "lstlisting": ("listing", "listings", "Listing", "Listings"), "page": ("page", "pages", "Page", "Pages")}
AUTOREF = {"figure": "Figure", "subfigure": "Figure", "table": "Table", "subtable": "Table", "section": "section",
           "subsection": "subsection", "subsubsection": "subsubsection", "chapter": "chapter", "part": "Part",
           "appendix": "Appendix", "equation": "Equation", "item": "item", "footnote": "footnote",
           "algorithm": "Algorithm", "lstlisting": "Listing", "paragraph": "paragraph"}


def compile_doc(files: dict[str, str], main: str = "main.tex", preamble: str | None = None,
                preamble_file: str = "preamble.tex") -> dict:
    """Compile files[main]. With `preamble`, files[main] is only the body: the document is the preamble, then
    \\begin{document}, the body, \\end{document} (line numbers stay the body's own)."""
    e = Engine(files, main)
    if preamble is not None:
        pre = tokenize(preamble, preamble_file)
        body = tokenize(files.get(main, ""), main)
        mk = lambda k, v: Tok(k, v, "(wrapper)", 0, 0)  # noqa: E731
        begin = [mk("cs", "begin"), mk("bg", "{")] + [mk("ch", c) for c in "document"] + [mk("eg", "}"), mk("par", "")]
        end = [mk("sp", " "), mk("cs", "end"), mk("bg", "{")] + [mk("ch", c) for c in "document"] + [mk("eg", "}")]
        e.files = dict(files, **{preamble_file: preamble})
        e.toks = pre + begin + body + end
    else:
        e.toks = tokenize(files.get(main, ""), main)
    e.pos = 0
    try:
        e.run()
        if not e.ended:
            if e.in_document:
                e.error("Emergency stop.", e.last_tok, "*** (job aborted, no legal \\end found): add \\end{document} "
                                                       "at the end.")
            elif e.cls is not None:
                e.error("LaTeX Error: Missing \\begin{document}.", e.last_tok,
                        "The body of the document goes between \\begin{document} and \\end{document}.")
    except Stop:
        pass
    except RecursionError:
        e.error("TeX capacity exceeded, sorry [input stack size=10000].", e.last_tok, "A macro calls itself forever.")
    if e.in_document:
        while len(e.frames) > 2:
            e.pop_frame()
        e.end_par()
        body_blocks = e.frames[-1]["blocks"] if len(e.frames) > 1 else []
    else:
        body_blocks = []
    doc = {"class": e.cls or "article", "options": e.cls_opts, "packages": dict(e.pkgs), "blocks": body_blocks,
           "footnotes": e.footnotes, "page": e.page, "toc": e.toc, "title": None}
    resolve(e, doc)
    if e.beamer is not None:
        doc["frames"] = e.beamer.frames
        resolve_blocks(e, [b for f in e.beamer.frames for b in f.get("blocks", [])])
    return {"doc": doc, "engine": e, "errors": e.errors, "warnings": e.warnings, "used": e.used, "labels": e.labels}


# ---------------------------------------------------------------- references and citations

def resolve(e, doc):
    resolve_blocks(e, doc["blocks"])
    for f in doc["footnotes"]:
        resolve_blocks(e, f["c"])
    # the table of contents, lists of figures/tables: the entries are known now
    lof, lot = [], []
    _floats(doc["blocks"], lof, lot)
    doc["lof"], doc["lot"] = lof, lot
    if e.bib.get("printed") or e.bib.get("manual"):
        doc["bib"] = bibliography(e)


def _floats(blocks, lof, lot):
    for b in blocks:
        if b.get("t") == "caption" and b["kind"] in ("figure", "table"):
            (lof if b["kind"] == "figure" else lot).append(b)
        for key in ("c", "blocks"):
            v = b.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and "t" in v[0] and v[0]["t"] not in (
                    "text", "math"):
                _floats(v, lof, lot)
        if b.get("t") == "par":
            for x in b["c"]:
                if x.get("t") == "box":
                    _floats([x["b"]], lof, lot)


def walk_inlines(blocks, fn):
    """Call fn on every inline of these blocks (recursively)."""
    for b in blocks:
        t = b.get("t")
        if t == "par":
            _walk_inl(b["c"], fn)
        for key in ("c", "title", "head", "note", "date", "label"):
            if t == "par" and key == "c":
                continue
            v = b.get(key)
            if isinstance(v, list):
                if v and isinstance(v[0], dict) and v[0].get("t") in ("text", "math", "ref", "cite", "fn", "link",
                                                                       "verb", "space", "br", "box", "logo",
                                                                       "runin", "and", "qed", "algcomment", "rule",
                                                                       "bibitem", "vspace_after"):
                    _walk_inl(v, fn)
                elif key == "c":
                    walk_inlines(v, fn)
        if t == "list":
            for it in b["items"]:
                _walk_inl(it["label"], fn)
                walk_inlines(it["c"], fn)
        if t == "title":
            for a in b["authors"]:
                _walk_inl(a, fn)
        if t == "tabular":
            for r in b["rows"]:
                for c in r.get("cells") or []:
                    if "c" in c:
                        _walk_inl(c["c"], fn)
                    else:
                        walk_inlines(c.get("blocks", []), fn)
        if t == "display":
            for r in b["rows"]:
                if "intertext" in r:
                    _walk_inl(r["intertext"], fn)
        if t == "algorithmic":
            for ln in b["lines"]:
                _walk_inl(ln["c"], fn)
        if t == "tcolorbox" and b.get("title"):
            _walk_inl(b["title"], fn)


def _walk_inl(inls, fn):
    for x in inls:
        fn(x)
        if x.get("t") == "link":
            _walk_inl(x["c"], fn)
        elif x.get("t") == "box":
            b = x["b"]
            if isinstance(b.get("c"), list) and b["c"] and b["c"][0].get("t") in ("text", "math", "ref", "cite"):
                _walk_inl(b["c"], fn)
            else:
                walk_inlines([b], fn)
        elif x.get("t") in ("runin", "algcomment"):
            _walk_inl(x["c"], fn)


def resolve_blocks(e, blocks):
    def fn(x):
        if x.get("t") == "ref":
            x["txt"], x["ok"] = ref_text(e, x)
        elif x.get("t") == "cite":
            x["txt"], x["ok"] = cite_text(e, x)
    walk_inlines(blocks, fn)


def ref_text(e, node):
    how = node["how"]
    keys = node["keys"]
    found = []
    for k in keys:
        lab = e.labels.get(k)
        if lab is None:
            e.warn(f"LaTeX Warning: Reference `{k}' on page 1 undefined on input line {node.get('line', 0)}.",
                   Tok("cs", "ref", node.get("file", "main.tex"), node.get("line", 0), 0))
            return "??", False
        found.append(lab)
    if not found:
        return "??", False
    lab = found[0]
    num = lab["text"] or ""
    kind = lab["kind"] or ""
    if how == "pageref":
        return "1", True
    if how == "eqref":
        return f"({num})", True
    if how in ("autoref", "Autoref"):
        name = AUTOREF.get(kind) or (e.theorems.get(lab.get("env") or "", {}) and
                                     _thm_name(e, lab.get("env")) or "")
        if how == "Autoref":
            name = name[:1].upper() + name[1:]
        return (f"{name} {num}" if name else num), True
    if how == "nameref":
        return num, True
    if how in ("cref", "Cref", "labelcref", "crefrange", "Crefrange"):
        names = CREF.get(kind)
        if names is None and lab.get("env"):
            nm = _thm_name(e, lab["env"]).lower()
            names = (nm, nm + "s", nm.title(), nm.title() + "s")
        if names is None:
            names = (kind, kind + "s", kind.title(), kind.title() + "s")
        cap = how.startswith("C")
        nums = [(f"({x['text']})" if (x["kind"] == "equation") else x["text"]) for x in found]
        if how == "labelcref":
            return ", ".join(nums), True
        if how in ("crefrange", "Crefrange") and len(nums) == 2:
            return f"{names[3 if cap else 1]} {nums[0]} to {nums[1]}", True
        if len(nums) == 1:
            return f"{names[2 if cap else 0]} {nums[0]}", True
        joined = ", ".join(nums[:-1]) + " and " + nums[-1] if len(nums) > 2 else f"{nums[0]} and {nums[1]}"
        return f"{names[3 if cap else 1]} {joined}", True
    return num if num else "", True


def _thm_name(e, env_name):
    th = e.theorems.get(env_name)
    if not th:
        return env_name or ""
    return "".join(x.text() for x in th["title"]).strip()


def cite_text(e, node):
    keys = node["keys"]
    how = node["how"]
    ok = True
    entries = e.bib["entries"]
    if e.bib.get("broken") or not (e.bib.get("printed") or e.bib.get("manual")):
        for k in keys:
            e.warn(f"LaTeX Warning: Citation `{k}' on page 1 undefined on input line {node.get('line', 0)}.",
                   Tok("cs", "cite", node.get("file", "main.tex"), node.get("line", 0), 0))
        return ("(?)" if how in ("citep",) and "natbib" in e.pkg_closure else "[?]"), False
    order = e.bib.get("manual") or bib.order(e)
    labels = []
    missing = []
    for k in keys:
        if k not in entries:
            missing.append(k)
            e.warn(f"LaTeX Warning: Citation `{k}' on page 1 undefined on input line {node.get('line', 0)}.",
                   Tok("cs", "cite", node.get("file", "main.tex"), node.get("line", 0), 0))
            labels.append(None)
        else:
            labels.append(k)
    if missing:
        ok = False
    authoryear = bib.is_authoryear(e) and not e.bib.get("manual")
    style = e.bib["style"] or "plain"
    post = "".join(x.get("s", "") for x in node.get("post") or [] if x.get("t") == "text")
    pre = "".join(x.get("s", "") for x in node.get("pre") or [] if x.get("t") == "text")

    def num(k):
        if k is None:
            return "?"
        if style == "alpha":
            return bib.alpha_label(entries[k])
        return str(order.index(k) + 1) if k in order else "?"

    def auth(k):
        return bib.author_short(entries[k]) if k else "?"

    def yr(k):
        return bib.year(entries[k]) if k else "?"
    if authoryear:
        if how in ("citet", "cite", "Citet", "textcite", "Textcite"):
            parts = [f"{auth(k)} ({yr(k)}{', ' + post if post and i == len(labels) - 1 else ''})"
                     for i, k in enumerate(labels)]
            return ", ".join(parts), ok
        if how in ("citep", "Citep", "parencite", "autocite"):
            inner = "; ".join(f"{auth(k)}, {yr(k)}" for k in labels)
            return "(" + (pre + " " if pre else "") + inner + (", " + post if post else "") + ")", ok
        if how == "citealt":
            return "; ".join(f"{auth(k)} {yr(k)}" for k in labels), ok
        if how == "citealp":
            return "; ".join(f"{auth(k)}, {yr(k)}" for k in labels), ok
        if how == "citeauthor":
            return ", ".join(auth(k) for k in labels), ok
        if how == "citeyear":
            return ", ".join(yr(k) for k in labels), ok
    nums = ", ".join(num(k) for k in labels)
    br = "[" + (pre + " " if pre else "") + nums + (", " + post if post else "") + "]"
    if how in ("citet", "Citet", "textcite", "Textcite"):
        return ", ".join(f"{auth(k)} [{num(k)}]" for k in labels), ok
    if how == "citeauthor":
        return ", ".join(auth(k) for k in labels), ok
    if how == "citeyear":
        return ", ".join(yr(k) for k in labels), ok
    if how == "citenum":
        return nums, ok
    return br, ok


def bibliography(e):
    """The reference list: [(label or None, inlines)] in order."""
    if e.bib.get("manual"):
        return {"manual": True}
    if e.bib.get("broken"):
        return {"entries": []}
    order = bib.order(e)
    authoryear = bib.is_authoryear(e)
    style = e.bib["style"] or "plain"
    out = []
    for i, k in enumerate(order):
        en = e.bib["entries"][k]
        if authoryear:
            lab = None
        elif style == "alpha":
            lab = bib.alpha_label(en)
        else:
            lab = str(i + 1)
        out.append({"key": k, "label": lab, "c": bib.format_entry(e, en)})
    return {"entries": out, "authoryear": authoryear}

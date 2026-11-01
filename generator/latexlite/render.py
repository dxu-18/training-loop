"""The document model -> HTML that looks like the compiled PDF (styles in docs/style.css, `.tex-*`).

Sizes are in pt, as LaTeX's: the class's font size (10/11/12pt), the paper and its margins (geometry), the
article class's heading sizes and spacing. Math is left as `<span class="tex-m" data-tex="...">` for KaTeX to typeset
in the browser; equation numbers come from latex-lite (\\tag*).
"""

from __future__ import annotations

import html
import re

from .commands import dim_pt
from .model import SIZES_10, SIZES_11, SIZES_12

PAPER = {"letterpaper": (614.3, 794.97), "a4paper": (597.5, 845.0), "a5paper": (420.6, 597.5),
         "b5paper": (516.0, 729.0), "legalpaper": (614.3, 1012.0), "executivepaper": (522.0, 756.0)}
TEXTWIDTH = {10: 345.0, 11: 360.0, 12: 390.0}
SECTION_STYLE = {-1: ("Huge", True), 0: ("huge", True), 1: ("Large", True), 2: ("large", True), 3: ("normalsize", True),
                 4: ("normalsize", True), 5: ("normalsize", True)}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


class Renderer:
    def __init__(self, doc: dict, engine):
        self.doc = doc
        self.e = engine
        opts = doc.get("options", [])
        self.pt = 12 if "12pt" in opts else 11 if "11pt" in opts else 10
        self.sizes = {10: SIZES_10, 11: SIZES_11, 12: SIZES_12}[self.pt]
        self.hyperref = "hyperref" in engine.pkg_closure and doc["class"] != "beamer"
        self.report = doc["class"] in ("report", "book", "memoir")
        self.step = None            # beamer: the overlay step being drawn
        self.hidden = False

    # ------------------------------------------------------------ the page

    def page_box(self):
        g = self.doc["page"]["geometry"]
        opts = self.doc.get("options", [])
        paper = next((p for p in PAPER if p in opts or g.get(p) is True), "letterpaper")
        if isinstance(g.get("paper"), str) and g["paper"] in PAPER:
            paper = g["paper"]
        pw, ph = PAPER[paper]
        if "landscape" in opts or g.get("landscape") is True:
            pw, ph = ph, pw
        tw = 410.0 if "twocolumn" in opts else TEXTWIDTH[self.pt]    # the classes' two-column text block
        left = right = (pw - tw) / 2
        top, bottom = 100.0, 100.0
        def d(k):
            v = g.get(k)
            return dim_pt(v) if isinstance(v, str) else None
        m = d("margin")
        if m is not None:
            left = right = top = bottom = m
        for k, attr in (("hmargin", "lr"), ("vmargin", "tb")):
            v = d(k)
            if v is not None:
                if attr == "lr":
                    left = right = v
                else:
                    top = bottom = v
        left = d("left") if d("left") is not None else d("lmargin") if d("lmargin") is not None else left
        right = d("right") if d("right") is not None else d("rmargin") if d("rmargin") is not None else right
        top = d("top") if d("top") is not None else top
        bottom = d("bottom") if d("bottom") is not None else bottom
        if d("textwidth") is not None:
            tw = d("textwidth")
            left = right = (pw - tw) / 2
        return pw, ph, left, right, top, bottom

    def html(self) -> str:
        if self.doc["class"] == "beamer":
            from .beamer import render_slides
            return render_slides(self)
        pw, ph, left, right, top, bottom = self.page_box()
        self.textwidth = pw - left - right
        pages = [[]]
        for b in self.doc["blocks"]:
            if b.get("t") == "newpage":
                if pages[-1]:          # \newpage on a page with nothing on it does nothing
                    pages.append([])
            else:
                pages[-1].append(b)
        while len(pages) > 1 and not pages[-1]:
            pages.pop()
        spacing = self.doc["page"].get("spacing") or 1.0
        parskip = self.doc["page"].get("parskip")
        cls = "tex-doc tex-" + self.doc["class"]
        if "twocolumn" in self.doc.get("options", []):
            cls += " tex-twocolumn"
        style = (f"--tex-fs:{self.sizes['normalsize']}pt;--tex-lh:{1.2 * spacing:.3f};--tex-pw:{pw:.1f}pt;"
                 f"--tex-tw:{self.textwidth:.1f}pt;")
        if parskip:
            v = dim_pt(parskip)
            if v is not None:
                style += f"--tex-parskip:{v:.1f}pt;"
        pi = self.doc["page"].get("parindent")
        if pi:
            v = dim_pt(pi)
            if v is not None:
                style += f"--tex-parindent:{v:.1f}pt;"
        out = [f'<div class="{cls}" style="{style}">']
        fns = list(self.doc["footnotes"])
        for i, blocks in enumerate(pages):
            n = i + 1
            ps = self.doc["page"]["style"]
            if i == 0 and self.doc["page"].get("this"):
                ps = self.doc["page"]["this"]
            if i == 0 and any(b.get("t") == "title" for b in blocks) and ps == "headings":
                ps = "plain"
            out.append(f'<div class="tex-page" style="width:{pw:.1f}pt;min-height:{ph:.1f}pt;'
                       f'padding:{top:.1f}pt {right:.1f}pt {bottom:.1f}pt {left:.1f}pt">')
            out.append(self.header(ps, n))
            if "twocolumn" in self.doc.get("options", []):
                # LaTeX fills the left column first: give the columns the page's text height
                out.append(f'<div class="tex-body" style="height:{ph - top - bottom - 30:.1f}pt">')
            else:
                out.append('<div class="tex-body">')
            out.append(self.blocks(blocks))
            out.append("</div>")
            page_fns = [f for f in fns if f.get("page", 1) == n] if len(pages) > 1 else fns
            if page_fns:
                out.append('<div class="tex-footnotes"><hr>')
                for f in page_fns:
                    out.append(f'<div class="tex-fn"><sup>{esc(str(f["n"]))}</sup>{self.blocks(f["c"], inline=True)}'
                               f'</div>')
                out.append("</div>")
            out.append(self.footer(ps, n))
            out.append("</div>")
        out.append("</div>")
        return "".join(out)

    def header(self, ps, n):
        if ps == "fancy":
            h = self.doc["page"]["head"]
            rule = True
            m = self.e.macros.get("headrulewidth")
            if m is not None and "".join(x.text() for x in m["body"]).strip().startswith("0"):
                rule = False
            parts = [self.hf(h.get(p), n) for p in "lcr"]
            if not any(parts):
                return f'<div class="tex-head{" tex-headrule" if rule else ""}"></div>'
            return (f'<div class="tex-head{" tex-headrule" if rule else ""}"><span>{parts[0]}</span><span>{parts[1]}'
                    f'</span><span>{parts[2]}</span></div>')
        if ps == "headings":
            sec = next((b for b in self.doc["blocks"] if b.get("t") == "heading" and b["level"] == 1), None)
            label = ""
            if sec is not None:
                label = (f'{esc(sec["num"] or "")}&ensp;' if sec["num"] else "") + self.inl(sec["c"])
            return (f'<div class="tex-head"><span class="tex-sc">{label}</span><span></span><span>{n}</span></div>')
        return ""

    def footer(self, ps, n):
        if ps == "empty" or ps == "headings":
            return ""
        if ps == "fancy":
            f = self.doc["page"]["foot"]
            parts = [self.hf(f.get(p), n) for p in "lcr"]
            m = self.e.macros.get("footrulewidth")
            rule = m is not None and not "".join(x.text() for x in m["body"]).strip().startswith("0")
            return (f'<div class="tex-foot{" tex-footrule" if rule else ""}"><span>{parts[0]}</span><span>'
                    f'{parts[1]}</span><span>{parts[2]}</span></div>')
        return f'<div class="tex-foot"><span></span><span>{n}</span><span></span></div>'

    def hf(self, toks, n):
        if not toks:
            return ""
        e = self.e
        e.counters.v["page"] = n
        saved = e.in_document, e.ended
        e.in_document, e.ended = True, False      # the header runs after \end{document} has stopped the engine
        try:
            e.push_frame(kind="hf", align="l")
            e.run_tokens(toks)
            f = e.pop_frame()
        finally:
            e.in_document, e.ended = saved
        inl = [x for b in f["blocks"] if b["t"] == "par" for x in b["c"]]
        return self.inl(inl)

    # ------------------------------------------------------------ blocks

    def visible(self, spec) -> bool:
        if self.step is None or not spec:
            return True
        from .beamer import parse_spec
        neg = spec.startswith("!")
        ok, _ = parse_spec(spec.lstrip("!"))
        return ok(self.step) != neg

    def blocks(self, blocks, inline=False) -> str:
        out = []
        for b in blocks:
            t = b.get("t")
            if t == "pause":
                if self.step is not None and b["from"] > self.step:
                    self.hidden = True
                continue
            if t == "onslide":
                self.hidden = not self.visible(b["spec"])
                continue
            fn = getattr(self, "b_" + str(t), None)
            if fn is None:
                continue
            was_hidden = self.hidden
            html_ = fn(b, inline) if fn.__code__.co_argcount > 2 else fn(b)
            if was_hidden:
                html_ = f'<div class="tex-hidden">{html_}</div>'
            out.append(html_)
        return "".join(out)

    def b_par(self, b, inline=False):
        cls = ["tex-p"]
        if not b.get("indent", True):
            cls.append("tex-noindent")
        a = b.get("align", "j")
        if a != "j":
            cls.append("tex-align-" + a)
        if any(x.get("t") == "space" and x.get("w") in ("hfill", "fill") for x in b["c"]):
            cls.append("tex-flex")
        if inline:
            return f'<span class="tex-p-inline">{self.inl(b["c"])}</span> '
        # a flex line (\hfill between boxes) ignores text-indent: the paragraph's indent box goes in as a spacer
        lead = '<span class="tex-indent"></span>' if "tex-flex" in cls and b.get("indent", True) and a == "j" else ""
        return f'<div class="{" ".join(cls)}">{lead}{self.inl(b["c"])}</div>'

    def b_heading(self, b):
        lvl = b["level"]
        size, bold = SECTION_STYLE.get(lvl, ("normalsize", True))
        num = b.get("num")
        title = self.inl(b["c"])
        if lvl <= 0 and self.report:
            word = "Appendix" if b.get("appendix") and lvl == 0 else ("Part" if lvl < 0 else "Chapter")
            top = f'<div class="tex-chapnum">{word} {esc(num)}</div>' if num else ""
            return f'<div class="tex-h tex-chapter">{top}<div class="tex-chaptitle">{title}</div></div>'
        if lvl < 0:
            return (f'<div class="tex-h tex-part"><div class="tex-chapnum">Part {esc(num or "")}</div>'
                    f'<div class="tex-chaptitle">{title}</div></div>')
        n = f'<span class="tex-secnum">{esc(num)}</span>' if num else ""
        return f'<div class="tex-h tex-h{lvl} tex-sz-{size}">{n}{title}</div>'

    def b_title(self, b):
        authors = "".join(f'<div class="tex-author">{self.inl(a)}</div>' for a in b["authors"])
        return (f'<div class="tex-titleblock"><div class="tex-title">{self.inl(b["title"])}</div>'
                f'<div class="tex-authors">{authors}</div><div class="tex-date">{self.inl(b["date"])}</div></div>')

    def b_abstract(self, b):
        return (f'<div class="tex-abstract"><div class="tex-abstract-h">Abstract</div>'
                f'{self.blocks(b["c"])}</div>')

    def b_toc(self, b):
        rows = []
        for t in self.doc.get("toc", []):
            lvl = t["level"]
            num = f'<span class="tex-tocnum">{esc(t["num"])}</span>' if t["num"] else ""
            rows.append(f'<div class="tex-toc-row tex-toc{lvl}">{num}<span class="tex-toc-title">{self.inl(t["c"])}'
                        f'</span><span class="tex-dots"></span><span class="tex-tocpage">1</span></div>')
        name = "Contents"
        return f'<div class="tex-toc"><div class="tex-h tex-h1 tex-sz-Large">{name}</div>{"".join(rows)}</div>'

    def b_lof(self, b, kind="lof"):
        items = self.doc.get("lof" if kind == "lof" else "lot", [])
        rows = "".join(f'<div class="tex-toc-row tex-toc2"><span class="tex-tocnum">{esc(c["num"])}</span>'
                       f'<span class="tex-toc-title">{self.inl(c["c"])}</span><span class="tex-dots"></span>'
                       f'<span class="tex-tocpage">1</span></div>' for c in items)
        name = "List of Figures" if kind == "lof" else "List of Tables"
        return f'<div class="tex-toc"><div class="tex-h tex-h1 tex-sz-Large">{name}</div>{rows}</div>'

    def b_lot(self, b):
        return self.b_lof(b, "lot")

    def b_list(self, b):
        kind = b["kind"]
        cls = f"tex-list tex-{kind}" + (" tex-compact" if b.get("compact") else "") + \
            (" tex-flush" if b.get("leftmargin") == "*" else "")
        items = []
        for it in b["items"]:
            lab = self.inl(it["label"])
            ov = f' data-overlay="{esc(it["overlay"])}"' if it.get("overlay") else ""
            if it.get("overlay") and not self.visible(it["overlay"]):
                ov += ' style="visibility:hidden"' 
            if kind == "description":
                first, rest = self._split_first(it["c"])
                items.append(f'<div class="tex-item tex-desc"{ov}><div class="tex-p tex-noindent"><span class="tex-b">'
                             f'{lab}</span> {first}</div>{rest}</div>')
            else:
                items.append(f'<div class="tex-item"{ov}><span class="tex-label">{lab}</span>'
                             f'<div class="tex-item-body">{self.blocks(it["c"])}</div></div>')
        return f'<div class="{cls}">{"".join(items)}</div>'

    def _split_first(self, blocks):
        if blocks and blocks[0].get("t") == "par":
            return self.inl(blocks[0]["c"]), self.blocks(blocks[1:])
        return "", self.blocks(blocks)

    def b_display(self, b):
        env = b["env"].rstrip("*")
        rows = b["rows"]
        parts = []
        cur = []
        for r in rows:
            if "intertext" in r:
                if cur:
                    parts.append(("math", cur))
                    cur = []
                parts.append(("text", r["intertext"]))
            else:
                cur.append(r)
        if cur:
            parts.append(("math", cur))
        out = []
        for kind, content in parts:
            if kind == "text":
                out.append(f'<div class="tex-p tex-noindent">{self.inl(content)}</div>')
                continue
            if env == "multline":
                lines = []
                for i, r in enumerate(content):
                    pos = "l" if i == 0 else "r" if i == len(content) - 1 else "c"
                    tex = r["tex"] + (f"\\tag*{{{r['tag']}}}" if r.get("tag") else "")
                    lines.append(f'<div class="tex-dm tex-mline-{pos}" data-tex="{esc(tex)}" data-display="1"></div>')
                out.append(f'<div class="tex-display tex-multline">{"".join(lines)}</div>')
                continue
            if env in ("equation", "displaymath"):
                r = content[0]
                tex = r["tex"] + (f" \\tag*{{{r['tag']}}}" if r.get("tag") else "")
            else:
                wrap = {"align": "align*", "flalign": "align*", "alignat": "align*", "eqnarray": "align*",
                        "gather": "gather*"}.get(env, "align*")
                body = " \\\\ ".join(r["tex"] + (f" \\tag*{{{r['tag']}}}" if r.get("tag") else "") for r in content)
                tex = f"\\begin{{{wrap}}}{body}\\end{{{wrap}}}"
            out.append(f'<div class="tex-dm" data-tex="{esc(tex)}" data-display="1"></div>')
        return f'<div class="tex-display">{"".join(out)}</div>'

    def b_float(self, b):
        kind = b["kind"]
        if kind == "algorithm":
            inner = self.blocks(b["c"])
            return f'<div class="tex-float tex-algorithm">{inner}</div>'
        return f'<div class="tex-float tex-{kind}">{self.blocks(b["c"])}</div>'

    def b_caption(self, b):
        txt = self.inl(b["c"])
        lab = esc(b["label"])
        if b["kind"] == "algorithm":
            return f'<div class="tex-algcaption"><span class="tex-b">{lab}</span> {txt}</div>'
        if b["kind"].startswith("sub"):
            return f'<div class="tex-caption tex-subcaption">{lab} {txt}</div>'
        plain = re.sub(r"<[^>]+>", "", txt)
        cls = "tex-caption" + (" tex-caption-long" if len(plain) + len(b["label"]) > 85 else "")
        return f'<div class="{cls}">{lab} {txt}</div>'

    def b_group(self, b):
        a = {"center": "c", "flushleft": "l", "flushright": "r"}.get(b["align"], "c")
        return f'<div class="tex-group tex-align-{a}">{self.blocks(b["c"])}</div>'

    def b_quote(self, b):
        return f'<div class="tex-quote tex-{b["kind"]}">{self.blocks(b["c"])}</div>'

    def b_verbatim(self, b):
        return f'<pre class="tex-verbatim">{esc(b["s"])}</pre>'

    def b_listing(self, b):
        cls = "tex-listing" + (" tex-frame-" + b["frame"] if b.get("frame") else "") + \
            (" tex-tt" if b.get("tt") else "") + (f" tex-sz-{b['size']}" if b.get("size") else "")
        rows = []
        for i, spans in enumerate(b["lines"]):
            num = f'<span class="tex-lnum">{b["first"] + i}</span>' if b.get("numbers") == "left" else ""
            code = "".join(f'<span class="tex-lst-{k}">{esc(s)}</span>' if k != "plain" else esc(s) for k, s in spans)
            rows.append(f'<div class="tex-lline">{num}<span class="tex-lcode">{code or "&#8203;"}</span></div>')
        cap = ""
        if b.get("caption"):
            cap = f'<div class="tex-caption">{esc(b["caption"]["label"])} {self.inl(b["caption"]["c"])}</div>'
        return f'{cap}<div class="{cls}">{"".join(rows)}</div>'

    def b_algorithmic(self, b):
        rows = []
        for ln in b["lines"]:
            # \Require / \Ensure sit in the label column, flush left; a \Comment goes to the right
            num = "" if ln.get("label") else f'<span class="tex-alnum">{"" if ln["n"] is None else str(ln["n"]) + ":"}</span>'
            code = self.inl([x for x in ln["c"] if x.get("t") != "algcomment"])
            com = self.inl([x for x in ln["c"] if x.get("t") == "algcomment"])
            rows.append(f'<div class="tex-alline">{num}<span class="tex-albody" style="padding-left:'
                        f'{1.5 * ln["indent"]:.1f}em"><span class="tex-alcode">{code}</span>{com}</span></div>')
        return f'<div class="tex-algorithmic">{"".join(rows)}</div>'

    def b_theorem(self, b):
        head = self.inl(b["head"]) + (f" {esc(b['num'])}" if b.get("num") else "")
        note = ""
        if b.get("note") is not None:
            note = f' <span class="tex-thmnote">({self.inl(b["note"])})</span>'
        style = b["style"]
        if b.get("ams"):
            head_html = f'<span class="tex-thmhead tex-thm-{style}">{head}</span>{note}<span class="tex-thmhead ' \
                        f'tex-thm-{style}">.</span>'
        else:
            head_html = f'<span class="tex-thmhead tex-thm-plain">{head}{note}</span>'
        first, rest = self._split_first(b["c"])
        return (f'<div class="tex-theorem tex-thmstyle-{style}"><div class="tex-p tex-noindent">{head_html} {first}</div>'
                f'{rest}</div>')

    def b_proof(self, b):
        title = self.inl(b["title"]) if b.get("title") else "Proof"
        first, rest = self._split_first(b["c"])
        qed = '<span class="tex-qed">□</span>' if b.get("qed") else ""
        if rest:
            rest = rest[:-6] + qed + "</div>" if rest.endswith("</div>") else rest + qed
            return (f'<div class="tex-proof"><div class="tex-p tex-noindent"><span class="tex-it">{title}.</span> '
                    f'{first}</div>{rest}</div>')
        return (f'<div class="tex-proof"><div class="tex-p tex-noindent"><span class="tex-it">{title}.</span> {first}'
                f'{qed}</div></div>')

    def b_tcolorbox(self, b):
        title = ""
        if b.get("title"):
            title = f'<div class="tex-tcb-title" style="background:{b["frame"]};color:{b["coltitle"]}">' \
                    f'{self.inl(b["title"])}</div>'
        radius = "0" if b.get("sharp") else "4pt"
        return (f'<div class="tex-tcolorbox" style="border-color:{b["frame"]};background:{b["back"]};'
                f'border-radius:{radius}">{title}<div class="tex-tcb-body">{self.blocks(b["c"])}</div></div>')

    def b_multicols(self, b):
        return f'<div class="tex-multicols" style="column-count:{b["n"]}">{self.blocks(b["c"])}</div>'

    def b_columnbreak(self, b):
        return '<div class="tex-colbreak"></div>'

    def b_vspace(self, b):
        h = b["h"]
        if h == "fill":
            return '<div class="tex-vfill"></div>'
        return f'<div class="tex-vspace" style="height:{max(0.0, float(h)):.1f}pt"></div>'

    def b_bibliography(self, b):
        name = "Bibliography" if self.report else "References"
        head = f'<div class="tex-h tex-h1 tex-sz-Large">{name}</div>' if not self.report else \
            f'<div class="tex-h tex-chapter"><div class="tex-chaptitle">{name}</div></div>'
        if b.get("manual"):
            rows = "".join(f'<div class="tex-bibitem"><span class="tex-biblabel">[{i + 1}]</span><span>'
                           f'{self.inl(x["c"])}</span></div>' for i, x in enumerate(b["manual"]))
            return f'<div class="tex-bib">{head}{rows}</div>'
        bibd = self.doc.get("bib") or {"entries": []}
        rows = []
        for en in bibd.get("entries", []):
            if en["label"] is None:
                rows.append(f'<div class="tex-bibitem tex-bib-ay"><span>{self.inl(en["c"])}</span></div>')
            else:
                rows.append(f'<div class="tex-bibitem"><span class="tex-biblabel">[{esc(en["label"])}]</span>'
                            f'<span>{self.inl(en["c"])}</span></div>')
        return f'<div class="tex-bib">{head}{"".join(rows)}</div>'

    def b_frame(self, b):
        return ""

    def b_btitle(self, b):
        sub = f'<div class="tex-bsubtitle">{self.inl(b["subtitle"])}</div>' if b.get("subtitle") else ""
        authors = "".join(f'<span>{self.inl(a)}</span>' for a in b["authors"])
        inst = f'<div class="tex-binst">{self.inl(b["institute"])}</div>' if b.get("institute") else ""
        return (f'<div class="tex-btitlepage"><div class="tex-btitle">{self.inl(b["title"])}{sub}</div>'
                f'<div class="tex-bauthors">{authors}</div>{inst}<div class="tex-bdate">{self.inl(b["date"])}</div>'
                f'</div>')

    def b_btoc(self, b):
        secs = self.e.beamer.sections if self.e.beamer else []
        cur = (self.slide_frame or {}).get("section") if hasattr(self, "slide_frame") else None
        rows = []
        for i, sct in enumerate(secs, 1):
            rows.append(f'<div class="tex-btoc-row"><span class="tex-btoc-num">{i}</span>{self.inl(sct["title"])}'
                        f'</div>')
            for sub in sct["subs"]:
                rows.append(f'<div class="tex-btoc-sub">{self.inl(sub)}</div>')
        return f'<div class="tex-btoc">{"".join(rows)}</div>'

    def b_columns(self, b):
        cols = "".join(f'<div class="tex-bcol" style="width:{self.width_css(c["w"]) or "50%"}">{self.blocks(c["c"])}'
                       f'</div>' for c in b["cols"])
        return f'<div class="tex-bcolumns">{cols}</div>'

    def b_bblock(self, b):
        if b.get("spec") and not self.visible(b["spec"]):
            return f'<div class="tex-hidden">{self._bblock(b)}</div>'
        return self._bblock(b)

    def _bblock(self, b):
        return (f'<div class="tex-bblock tex-{b["kind"]}"><div class="tex-bblock-title">{self.inl(b["title"])}</div>'
                f'<div class="tex-bblock-body">{self.blocks(b["c"])}</div></div>')

    # ------------------------------------------------------------ inlines

    def st_class(self, st) -> str:
        if st is None:
            return ""
        c = []
        if st.bold:
            c.append("tex-b")
        if st.shape != "n":
            c.append("tex-" + st.shape)
        if st.family != "rm":
            c.append("tex-" + st.family)
        if st.size != "normalsize":
            c.append("tex-sz-" + st.size)
        if st.under:
            c.append("tex-u")
        return " ".join(c)

    def wrap(self, st, inner: str) -> str:
        cls = self.st_class(st)
        color = f' style="color:{st.color}"' if st is not None and st.color else ""
        if not cls and not color:
            return inner
        return f'<span class="{cls}"{color}>{inner}</span>' if cls else f'<span{color}>{inner}</span>'

    def inl(self, nodes) -> str:
        out = []
        for x in nodes or []:
            t = x.get("t")
            if t == "pause":
                if self.step is not None and x["from"] > self.step:
                    self.hidden = True
                continue
            if t == "onslide":
                self.hidden = not self.visible(x["spec"])
                continue
            if self.hidden:
                saved, self.hidden = self.hidden, False
                inner = self.inl([x])
                self.hidden = saved
                out.append(f'<span class="tex-hidden">{inner}</span>')
                continue
            if t == "only":
                if self.visible(x["spec"]):
                    out.append(self.inl(x["c"]))
                continue
            if t in ("uncover", "invisible"):
                vis = self.visible(x["spec"]) if t == "uncover" else not self.visible(x["spec"])
                out.append(self.inl(x["c"]) if vis else f'<span class="tex-hidden">{self.inl(x["c"])}</span>')
                continue
            if t == "alertspec":
                out.append(self.inl(x["c"] if self.visible(x["spec"]) else x["plain"]))
                continue
            if t == "text":
                out.append(self.wrap(x["st"], esc(x["s"])))
            elif t == "math":
                out.append(self.wrap(x.get("st"), f'<span class="tex-m" data-tex="{esc(x["tex"])}"></span>'))
            elif t == "ref":
                txt = esc(x.get("txt", "??"))
                if not x.get("ok", True):
                    txt = f'<span class="tex-b">{txt}</span>'
                cls = "tex-ref" + (" tex-link-int" if self.hyperref else "")
                out.append(self.wrap(x["st"], f'<span class="{cls}">{txt}</span>'))
            elif t == "cite":
                txt = esc(x.get("txt", "[?]"))
                if not x.get("ok", True):
                    txt = re.sub(r"\?", '<span class="tex-b">?</span>', txt)
                cls = "tex-cite" + (" tex-link-cite" if self.hyperref else "")
                out.append(self.wrap(x["st"], f'<span class="{cls}">{txt}</span>'))
            elif t == "fn":
                cls = "tex-fnmark" + (" tex-link-int" if self.hyperref else "")
                out.append(f'<sup class="{cls}">{esc(str(x["n"]))}</sup>')
            elif t == "link":
                cls = "tex-link" + (" tex-link-url" if self.hyperref else "")
                out.append(f'<a class="{cls}" href="{esc(x["url"])}" target="_blank" rel="noopener">'
                           f'{self.inl(x["c"])}</a>')
            elif t == "verb":
                out.append(self.wrap(x["st"], f'<span class="tex-tt tex-verb">{esc(x["s"])}</span>'))
            elif t == "space":
                w = x["w"]
                if w in ("hfill", "fill"):
                    out.append('<span class="tex-hfill"></span>')
                elif w == "dotfill":
                    out.append('<span class="tex-hfill tex-dotfill"></span>')
                elif w == "hrulefill":
                    out.append('<span class="tex-hfill tex-rulefill"></span>')
                elif isinstance(w, (int, float)):
                    out.append(f'<span class="tex-space" style="margin-left:{w:.2f}pt"></span>')
                else:
                    em = {"thin": 1 / 6, "med": 2 / 9, "thick": 5 / 18, "en": 0.5, "quad": 1, "qquad": 2,
                          "negthin": -1 / 6}.get(w, 0.25)
                    out.append(f'<span class="tex-space" style="margin-left:{em:.3f}em"></span>')
            elif t == "br":
                out.append("<br>")
            elif t == "box":
                out.append(self.box(x["b"]))
            elif t == "logo":
                logos = {"LaTeX": 'L<span class="tex-la">a</span>T<span class="tex-e">e</span>X',
                         "TeX": 'T<span class="tex-e">e</span>X',
                         "LaTeXe": 'L<span class="tex-la">a</span>T<span class="tex-e">e</span>X&thinsp;2<span '
                                   'class="tex-e">ε</span>'}
                out.append(self.wrap(x["st"], f'<span class="tex-logo">{logos[x["s"]]}</span>'))
            elif t == "runin":
                num = f'{esc(x["num"])}&ensp;' if x.get("num") else ""
                out.append(f'<span class="tex-b tex-runin">{num}{self.inl(x["c"])}</span>')
            elif t == "qed":
                out.append('<span class="tex-qed">□</span>')
            elif t == "algcomment":
                out.append(f'<span class="tex-algcomment">▷ {self.inl(x["c"])}</span>')
            elif t == "rule":
                out.append(f'<span class="tex-rule" style="width:{x["w"]:.1f}pt;height:{x["h"]:.1f}pt"></span>')
            elif t == "bibitem":
                pass
        return "".join(out)

    # ------------------------------------------------------------ boxes

    def width_css(self, w):
        """A width like 0.8\\linewidth -> CSS (a percentage of the line), or pt."""
        if not w:
            return None
        s = str(w).strip().replace(" ", "")
        m = re.match(r"^(-?[\d.]*)\\(linewidth|textwidth|columnwidth|hsize)$", s)
        if m:
            f = float(m.group(1)) if m.group(1) not in ("", ".") else 1.0
            return f"{f * 100:.2f}%"
        pt = dim_pt(s)
        return f"{pt:.1f}pt" if pt is not None else None

    def box(self, b) -> str:
        t = b.get("t")
        if t == "graphic":
            return self.graphic(b)
        if t == "tabular":
            from .render_table import tabular
            return tabular(self, b)
        if t in ("minipage", "subfloat"):
            w = self.width_css(b.get("w")) or "100%"
            cls = "tex-minipage" if t == "minipage" else "tex-subfloat"
            return f'<span class="{cls}" style="width:{w}">{self.blocks(b["c"])}</span>'
        if t == "colorbox":
            border = f"border:0.4pt solid {b['frame']};" if b.get("frame") else ""
            return f'<span class="tex-colorbox" style="background:{b["bg"] or "transparent"};{border}">' \
                   f'{self.inl(b["c"])}</span>'
        if t == "mbox":
            return f'<span class="tex-mbox{" tex-fbox" if b.get("frame") else ""}">{self.inl(b["c"])}</span>'
        if t == "transform":
            op, v = b["op"], b["v"]
            try:
                f = float(v)
            except ValueError:
                f = 0
            tr = f"rotate({-f}deg)" if op == "rotatebox" else f"scale({f})"
            return f'<span class="tex-transform" style="transform:{tr}">{self.inl(b["c"])}</span>'
        if t == "resized":
            return f'<span class="tex-resized">{self.inl(b["c"])}</span>'
        if t == "tikz":
            return f'<span class="tex-tikz">{b["svg"]}</span>'
        if t == "plot":
            return f'<span class="tex-plot">{b["svg"]}</span>'
        return ""

    def graphic(self, b) -> str:
        found = b.get("found")
        w = self.width_css(b.get("w"))
        h = self.width_css(b.get("h")) if b.get("h") else None
        style = []
        nat_w, nat_h = 320.0, 240.0
        if b.get("scale"):
            try:
                s = float(b["scale"])
                style.append(f"width:{nat_w * s:.1f}pt")
            except ValueError:
                pass
        if w:
            style.append(f"width:{w}")
        if h and not w:
            style.append(f"height:{h}")
        if not style:
            style.append(f"width:{nat_w:.1f}pt")
        if b.get("angle"):
            try:
                style.append(f"transform:rotate({-float(b['angle'])}deg)")
            except ValueError:
                pass
        st = ";".join(style)
        if found is None:
            return f'<span class="tex-graphic tex-missing" style="{st}"><span>{esc(b["file"])}</span></span>'
        if found.get("svg"):
            return f'<span class="tex-graphic" style="{st}">{found["svg"]}</span>'
        name = found.get("example") or found.get("file", "")
        letter = {"example-image-a": "A", "example-image-b": "B", "example-image-c": "C"}.get(name, "")
        label = "Image " + letter if letter else "Image"
        return (f'<span class="tex-graphic tex-example" style="{st}"><svg viewBox="0 0 320 240" '
                f'preserveAspectRatio="none"><rect x="0.5" y="0.5" width="319" height="239" fill="#e8e8e8" '
                f'stroke="#555"/><line x1="0" y1="0" x2="320" y2="240" stroke="#bbb"/><line x1="320" y1="0" x2="0" '
                f'y2="240" stroke="#bbb"/><text x="160" y="128" text-anchor="middle" font-size="30" '
                f'font-family="KaTeX_Main, serif" fill="#333">{esc(label)}</text></svg></span>')


def render(result: dict) -> str:
    return Renderer(result["doc"], result["engine"]).html()

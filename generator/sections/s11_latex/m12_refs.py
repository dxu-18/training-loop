"""11.12 Cross-references, links & citations: \\ref \\pageref \\eqref, cleveref's \\cref / \\Cref, hyperref,
\\cite with BibTeX styles, natbib's \\citet / \\citep, references.bib entries."""

from ._common import AMS, BASE, L, drill, fixed, template

M = "11.12"
BIB = """@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
@inproceedings{devlin2019bert,
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle = {Proceedings of NAACL},
  year      = {2019}
}
@article{kingma2014adam,
  title   = {Adam: A Method for Stochastic Optimization},
  author  = {Kingma, Diederik P. and Ba, Jimmy},
  journal = {arXiv preprint arXiv:1412.6980},
  year    = {2014}
}
@misc{he2016resnet,
  title         = {Deep Residual Learning for Image Recognition},
  author        = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year          = {2016},
  eprint        = {1512.03385},
  archivePrefix = {arXiv}
}
"""
KEYS = {"vaswani2017attention": "Transformers", "devlin2019bert": "BERT", "kingma2014adam": "Adam",
        "he2016resnet": "ResNets"}
HYPER = AMS + "\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{cleveref}\n"
NAT = BASE + "\\usepackage{natbib}\n"
FILES = {"references.bib": BIB}


def _fig(label="fig:loss"):
    return (f"\\begin{{figure}}[htbp]\n  \\centering\n  \\includegraphics[width=0.4\\linewidth]{{example-image}}\n"
            f"  \\caption{{Loss.}}\\label{{{label}}}\n\\end{{figure}}\n")


# ================================================================= lite

@template(M, "lite")
def refs(rng):
    a, b = rng.choice([("Introduction", "Method"), ("Data", "Results"), ("Background", "Experiments"),
                       ("Setup", "Analysis")])
    key = a.lower()[:5]
    eq, lab = rng.choice([("E = mc^2", "e"), ("y = Wx + b", "lin"), ("p = \\sigma(z)", "sig"), ("L = -\\log p", "nll")])
    items = [
        (f"Write a section {a} labelled sec:{key} (text: Hello.), then a section {b} saying \u201cAs in Section 1, on "
         f"page 1.\u201d with \\ref and \\pageref.",
         f"\\section{{{a}}}\\label{{sec:{key}}}\nHello.\n\\section{{{b}}}\nAs in Section~\\ref{{sec:{key}}}, on "
         f"page~\\pageref{{sec:{key}}}.", {"require": ["\\ref", "\\pageref"]}),
        (f"Write a numbered equation ${eq}$ labelled eq:{lab}, then \u201cBy (1), it holds.\u201d with \\eqref.",
         f"\\begin{{equation}}\\label{{eq:{lab}}}\n  {eq}\n\\end{{equation}}\nBy~\\eqref{{eq:{lab}}}, it holds.",
         {"require": ["\\eqref"]}),
    ]
    return drill(rng, items, preamble=HYPER)


@template(M, "lite")
def cleveref(rng):
    items = [
        ("Make the figure shown (labelled fig:loss), then write \u201cFigure 1 shows the loss.\u201d with \\Cref "
         "(it writes \u201cFigure\u201d for you).", _fig() + "\\Cref{fig:loss} shows the loss.", {"require": ["\\Cref"]}),
        ("Make the figure shown (labelled fig:loss), then write \u201cSee fig. 1.\u201d with \\cref (lower case, "
         "abbreviated).", _fig() + "See \\cref{fig:loss}.", {"require": ["\\cref"]}),
        ("Write a numbered equation $y = Wx$ labelled eq:lin, then \u201cSee eq. (1).\u201d with \\cref.",
         "\\begin{equation}\\label{eq:lin}\n  y = Wx\n\\end{equation}\nSee \\cref{eq:lin}.", {"require": ["\\cref"]}),
        ("Make the table shown (labelled tab:res), then write \u201cTable 1 lists the runs.\u201d with \\Cref.",
         "\\begin{table}[htbp]\n  \\centering\n  \\caption{Runs.}\\label{tab:res}\n  \\begin{tabular}{lc}\n"
         "    run & loss \\\\\n    a & 0.31 \\\\\n  \\end{tabular}\n\\end{table}\n\\Cref{tab:res} lists the runs.",
         {"require": ["\\Cref"]}),
    ]
    return drill(rng, items, preamble=HYPER)


# ================================================================= easy

@template(M, "easy")
def cite_numeric(rng):
    k = rng.choice(list(KEYS))
    k2 = rng.choice([x for x in KEYS if x != k])
    items = [
        (f"Cite {k} with \\cite in \u201c{KEYS[k]} changed the field [1].\u201d, then print the bibliography from "
         f"references.bib in the plain style.",
         f"{KEYS[k]} changed the field~\\cite{{{k}}}.\n\\bibliographystyle{{plain}}\n\\bibliography{{references}}",
         {"require": ["\\cite", "\\bibliography"]}),
        (f"Cite {k} and {k2} together (one \\cite with both keys) in \u201cSee [1, 2].\u201d, with the unsrt style "
         f"(numbered in citation order).", f"See~\\cite{{{k},{k2}}}.\n\\bibliographystyle{{unsrt}}\n"
                                           f"\\bibliography{{references}}", {"require": ["\\cite"]}),
    ]
    return drill(rng, items, preamble=BASE, files=FILES)


@template(M, "easy")
def natbib_cites(rng):
    k = rng.choice(list(KEYS))
    items = [
        (f"With natbib and the plainnat style, write \u201cAuthor et al. (year) introduced it.\u201d with \\citet for "
         f"{k}, then the bibliography.", f"\\citet{{{k}}} introduced it.\n\\bibliographystyle{{plainnat}}\n"
                                        f"\\bibliography{{references}}", {"require": ["\\citet"]}),
        (f"With natbib and plainnat, write \u201cIt works (Author et al., year).\u201d with \\citep for {k}, then the "
         f"bibliography.", f"It works~\\citep{{{k}}}.\n\\bibliographystyle{{plainnat}}\n\\bibliography{{references}}",
         {"require": ["\\citep"]}),
    ]
    return drill(rng, items, preamble=NAT, files=FILES)


# ================================================================= medium

@template(M, "medium")
def bib_entry(rng):
    year = rng.choice([2023, 2024, 2025])
    eprint = f"{year % 100}{rng.randint(1, 12):02d}.{rng.randint(10000, 99999)}"
    title = rng.choice(["Scaling Small Models", "Warmup Considered Helpful", "Tokens All the Way Down"])
    main = "As shown by \\citet{lastname2024arxiv}.\n\\bibliographystyle{plainnat}\n\\bibliography{references}"
    sol = (f"@misc{{lastname2024arxiv,\n  title         = {{{title}}},\n  author        = {{Lastname, Firstname and "
           f"Other, Author}},\n  year          = {{{year}}},\n  eprint        = {{{eprint}}},\n"
           f"  archivePrefix = {{arXiv}}\n}}\n")
    return L(f"main.tex (shown) cites lastname2024arxiv, which isn't in references.bib yet. Write references.bib with "
             f"that entry, like the guide's arXiv example: @misc, title \u201c{title}\u201d, authors Firstname Lastname and "
             f"Author Other, year {year}, eprint {eprint}, archivePrefix arXiv.", sol, mode="bib", preamble=NAT,
             body=main)


@template(M, "medium")
def autoref_names(rng):
    sec = rng.choice(["Data", "Model", "Training", "Evaluation"])
    items = [
        (f"Write a section {sec} labelled sec:data, a numbered equation $y = f(x)$ labelled eq:f, and the table "
         "shown (labelled tab:t), then: \u201cSection 1, Equation 1 and Table 1.\u201d using \\Cref three times.",
         f"\\section{{{sec}}}\\label{{sec:data}}\n\\begin{{equation}}\\label{{eq:f}}\n  y = f(x)\n\\end{{equation}}\n"
         "\\begin{table}[htbp]\n  \\centering\n  \\caption{T.}\\label{tab:t}\n  \\begin{tabular}{c}\n    1 \\\\\n"
         "  \\end{tabular}\n\\end{table}\n\\Cref{sec:data}, \\Cref{eq:f} and \\Cref{tab:t}.", {"require": ["\\Cref"]}),
        ("Two figures labelled fig:a and fig:b (as shown), then \u201cSee figs. 1 and 2.\u201d with ONE \\cref.",
         _fig("fig:a") + _fig("fig:b") + "See \\cref{fig:a,fig:b}.", {"require": ["\\cref"]}),
    ]
    return drill(rng, items, preamble=HYPER)


# ================================================================= hard

@template(M, "hard")
def undefined(rng):
    k = rng.choice(list(KEYS))
    items = [
        ("This prints ?? (Reference undefined): the label is spelled differently. Fix the \\ref.",
         "\\section{Results}\\label{sec:results}\nSee Section~\\ref{sec:results}.",
         {"starter": "\\section{Results}\\label{sec:results}\nSee Section~\\ref{sec:result}.", "preamble": BASE}),
        (f"This prints [?] (Citation undefined): there's no \\bibliographystyle. Add it (plain) so BibTeX runs.",
         f"Read~\\cite{{{k}}}.\n\\bibliographystyle{{plain}}\n\\bibliography{{references}}",
         {"starter": f"Read~\\cite{{{k}}}.\n\\bibliography{{references}}", "preamble": BASE, "files": FILES}),
        (f"The key is wrong (Citation undefined). The .bib file has {k}: fix the \\cite.",
         f"Read~\\cite{{{k}}}.\n\\bibliographystyle{{plain}}\n\\bibliography{{references}}",
         {"starter": f"Read~\\cite{{{k[:-2]}}}.\n\\bibliographystyle{{plain}}\n\\bibliography{{references}}",
          "preamble": BASE, "files": FILES}),
    ]
    return drill(rng, items, preamble=BASE)


@template(M, "hard")
def citation_forms(rng):
    a, b = rng.sample(list(KEYS), 2)
    sol = (f"\\citet{{{a}}} came before \\citet{{{b}}}; both are cited often~\\citep{{{a},{b}}}, see "
           f"\\citep[p.~3]{{{b}}}.\n\\bibliographystyle{{plainnat}}\n\\bibliography{{references}}")
    return L("**Citation forms.** Recreate the target with natbib: two textual citations, one parenthetical with two "
             "keys, and one with a page note (\\citep[p.~3]{...}).", sol, preamble=NAT, files=FILES,
             require=["\\citet", "\\citep"])


# ================================================================= challenge

@template(M, "challenge")
@fixed
def related_work(rng):
    ks = list(KEYS)
    sol = ("\\section{Related Work}\\label{sec:related}\n"
           f"\\citet{{{ks[0]}}} introduced attention-only models; \\citet{{{ks[1]}}} pretrained them. Optimizers "
           f"such as Adam~\\citep{{{ks[2]}}} train residual networks~\\citep{{{ks[3]}}}.\n"
           "\\section{Method}\nWe build on \\cref{sec:related}.\n\\bibliographystyle{plainnat}\n"
           "\\bibliography{references}")
    return L("**Related work.** Recreate the target: natbib citations of all four papers in references.bib, a \\cref "
             "back to the section, and the plainnat bibliography.", sol,
             preamble=BASE + "\\usepackage{natbib}\n\\usepackage{hyperref}\n\\usepackage{cleveref}\n", files=FILES,
             require=["\\citet", "\\citep", "\\cref", "\\bibliography"])


@template(M, "challenge")
def cross_refs(rng):
    eq = rng.choice(["y = \\sigma(Wx + b)", "h = \\operatorname{ReLU}(Wx)", "p = \\operatorname{softmax}(z)",
                     "z = W_2 \\tanh(W_1 x)"])
    nxt = rng.choice(["Training", "Experiments", "Results", "Ablations"])
    sol = ("\\section{Model}\\label{sec:model}\n\\begin{equation}\\label{eq:model}\n  " + eq + "\n"
           "\\end{equation}\n" + _fig("fig:arch") +
           f"\\section{{{nxt}}}\n" + "\\Cref{sec:model} defines \\cref{eq:model}, drawn in \\cref{fig:arch} on "
           "page~\\pageref{fig:arch}.")
    return L("**Everything cross-referenced.** Recreate the target: a section, an equation and a figure, then a "
             "sentence referring to all three with cleveref and to the figure's page.", sol, preamble=HYPER,
             require=["\\cref", "\\Cref", "\\pageref"])

"""11.13 Algorithms & code listings: algorithm floats with algpseudocode (\\State \\For \\If \\While \\Function
\\Comment \\Return, line numbers), lstlisting with its options, \\lstinline, referencing an algorithm."""

from ._common import AMS, L, drill, fixed, template

M = "11.13"
ALG = AMS + "\\usepackage{algorithm}\n\\usepackage{algpseudocode}\n"
LST = AMS + "\\usepackage{listings}\n"
BOTH = ALG + "\\usepackage{listings}\n\\usepackage{hyperref}\n\\usepackage{cleveref}\n"

SGD = """\\begin{algorithm}[t]
\\caption{Mini-batch SGD with early stopping}\\label{alg:sgd}
\\begin{algorithmic}[1]
  \\Require dataset $\\mathcal{D}$, learning rate $\\eta$, epochs $E$
  \\Ensure trained parameters $\\theta$
  \\State initialize $\\theta$
  \\For{$e = 1$ \\textbf{to} $E$}
    \\For{each mini-batch $B \\subset \\mathcal{D}$}
      \\State $g \\gets \\frac{1}{|B|} \\sum_{(x, y) \\in B} \\nabla_\\theta \\ell(f_\\theta(x), y)$
      \\State $\\theta \\gets \\theta - \\eta g$ \\Comment{gradient step}
    \\EndFor
    \\If{validation loss stopped improving}
      \\State \\textbf{break}
    \\EndIf
  \\EndFor
  \\State \\Return $\\theta$
\\end{algorithmic}
\\end{algorithm}"""

ADAM = """\\begin{algorithm}[t]
\\caption{Adam}\\label{alg:adam}
\\begin{algorithmic}[1]
  \\Require step size $\\eta$, decay rates $\\beta_1, \\beta_2$, $\\epsilon$
  \\State $m_0 \\gets 0$, $v_0 \\gets 0$, $t \\gets 0$
  \\While{$\\theta_t$ not converged}
    \\State $t \\gets t + 1$
    \\State $g_t \\gets \\nabla_\\theta f_t(\\theta_{t-1})$
    \\State $m_t \\gets \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$
    \\State $v_t \\gets \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$
    \\State $\\hat{m}_t \\gets m_t / (1 - \\beta_1^t)$, $\\hat{v}_t \\gets v_t / (1 - \\beta_2^t)$
    \\State $\\theta_t \\gets \\theta_{t-1} - \\eta \\hat{m}_t / (\\sqrt{\\hat{v}_t} + \\epsilon)$
  \\EndWhile
  \\State \\Return $\\theta_t$
\\end{algorithmic}
\\end{algorithm}"""

PY = ["loss = criterion(model(x), y)\nloss.backward()\noptimizer.step()",
      "for x, y in loader:\n    optimizer.zero_grad()\n    loss = criterion(model(x), y)\n    loss.backward()\n"
      "    optimizer.step()",
      "def accuracy(logits, y):\n    return (logits.argmax(1) == y).float().mean()"]


# ================================================================= lite

@template(M, "lite")
def statements(rng):
    v = rng.choice(["\\theta", "w", "x"])
    agg = rng.choice([("sum", "s", "s + v"), ("maximum", "m", "\\max(m, v)")])
    items = [
        (f"Write an algorithmic block with line numbers ([1]) of three statements: initialize ${v}$, compute "
         f"$g \\gets \\nabla L({v})$, and \\Return ${v}$ (a \\State before \\Return).",
         f"\\begin{{algorithmic}}[1]\n  \\State initialize ${v}$\n  \\State $g \\gets \\nabla L({v})$\n"
         f"  \\State \\Return ${v}$\n\\end{{algorithmic}}", {"require": ["env:algorithmic", "\\State", "\\Return"]}),
        (f"Write an algorithmic block with line numbers: \\Require a list $x$, \\Ensure its {agg[0]} ${agg[1]}$, then "
         f"${agg[1]} \\gets 0$, a \\ForAll{{$v \\in x$}} loop setting ${agg[1]} \\gets {agg[2]}$, and \\Return ${agg[1]}$.",
         f"\\begin{{algorithmic}}[1]\n  \\Require a list $x$\n  \\Ensure its {agg[0]} ${agg[1]}$\n"
         f"  \\State ${agg[1]} \\gets 0$\n  \\ForAll{{$v \\in x$}}\n    \\State ${agg[1]} \\gets {agg[2]}$\n  \\EndFor\n"
         f"  \\State \\Return ${agg[1]}$\n\\end{{algorithmic}}", {"require": ["\\Require", "\\Ensure", "\\EndFor"]}),
    ]
    return drill(rng, items, preamble=ALG)


@template(M, "lite")
def listing(rng):
    code = rng.choice(PY)
    items = [
        ("Show this PyTorch code in a Python listing (lstlisting with language=Python).",
         f"\\begin{{lstlisting}}[language=Python]\n{code}\n\\end{{lstlisting}}", {"require": ["env:lstlisting"]}),
        ("Show this PyTorch code in a Python listing with line numbers on the left and a frame "
         "(language=Python, numbers=left, frame=single).",
         f"\\begin{{lstlisting}}[language=Python, numbers=left, frame=single]\n{code}\n\\end{{lstlisting}}",
         {"require": ["env:lstlisting"]}),
    ]
    return drill(rng, items, preamble=LST)


# ================================================================= easy

@template(M, "easy")
def control_flow(rng):
    items = [
        ("Write a numbered algorithmic block: \\If{$x > 0$} return $x$, \\Else return $0$ (ReLU), closed with \\EndIf.",
         "\\begin{algorithmic}[1]\n  \\If{$x > 0$}\n    \\State \\Return $x$\n  \\Else\n    \\State \\Return $0$\n"
         "  \\EndIf\n\\end{algorithmic}", {"require": ["\\If", "\\EndIf"]}),
        ("Write a numbered algorithmic block: $t \\gets 0$, then \\While{loss $> \\epsilon$} take a step "
         "($\\theta \\gets \\theta - \\eta g$) and $t \\gets t + 1$, closed with \\EndWhile.",
         "\\begin{algorithmic}[1]\n  \\State $t \\gets 0$\n  \\While{loss $> \\epsilon$}\n"
         "    \\State $\\theta \\gets \\theta - \\eta g$\n    \\State $t \\gets t + 1$\n  \\EndWhile\n\\end{algorithmic}",
         {"require": ["\\While"]}),
        ("Write a numbered algorithmic block: a \\Function{ReLU}{$x$} that returns $\\max(0, x)$, closed with "
         "\\EndFunction.", "\\begin{algorithmic}[1]\n  \\Function{ReLU}{$x$}\n    \\State \\Return $\\max(0, x)$\n"
                           "  \\EndFunction\n\\end{algorithmic}", {"require": ["\\Function"]}),
        ("Write a numbered algorithmic block for the sign of $x$: \\If{$x > 0$} return 1, \\ElsIf{$x < 0$} return "
         "$-1$, \\Else return 0, closed with \\EndIf.",
         "\\begin{algorithmic}[1]\n  \\If{$x > 0$}\n    \\State \\Return $1$\n  \\ElsIf{$x < 0$}\n"
         "    \\State \\Return $-1$\n  \\Else\n    \\State \\Return $0$\n  \\EndIf\n\\end{algorithmic}",
         {"require": ["\\ElsIf"]}),
    ]
    return drill(rng, items, preamble=ALG)


@template(M, "easy")
def float_and_comment(rng):
    cap, com = rng.choice([("One SGD step", "update"), ("A gradient step", "descend"), ("Plain SGD", "step"),
                           ("Gradient descent", "move downhill")])
    items = [
        (f"Put a numbered algorithmic block in an algorithm float captioned \u201c{cap}\u201d: "
         f"$g \\gets \\nabla L(\\theta)$ and $\\theta \\gets \\theta - \\eta g$ with the comment \u201c{com}\u201d at "
         f"the right (\\Comment).",
         f"\\begin{{algorithm}}[t]\n\\caption{{{cap}}}\n\\begin{{algorithmic}}[1]\n  \\State $g \\gets \\nabla "
         f"L(\\theta)$\n  \\State $\\theta \\gets \\theta - \\eta g$ \\Comment{{{com}}}\n\\end{{algorithmic}}\n"
         f"\\end{{algorithm}}", {"require": ["env:algorithm", "\\Comment"]}),
    ]
    return drill(rng, items, preamble=ALG)


# ================================================================= medium

@template(M, "medium")
@fixed
def guide_sgd(rng):
    return L("**The guide's SGD.** Recreate the target: an algorithm float with \\Require / \\Ensure, nested \\For "
             "loops, a \\Comment, an \\If with break, and the \\Return.", SGD, preamble=ALG,
             require=["env:algorithm", "\\For", "\\If", "\\Comment"])


@template(M, "medium")
def listing_options(rng):
    code = rng.choice(PY)
    items = [
        ("Show the code in a listing captioned \u201cThe training step\u201d (caption=...), in a small typewriter "
         "font (basicstyle=\\ttfamily\\small), with line numbers on the left and a frame.",
         f"\\begin{{lstlisting}}[language=Python, basicstyle=\\ttfamily\\small, frame=single, numbers=left, "
         f"caption={{The training step}}]\n{code}\n\\end{{lstlisting}}", {"require": ["env:lstlisting"]}),
        ("Write the sentence \u201cCall optimizer.step() after loss.backward().\u201d with both calls as inline code "
         "(\\lstinline|...|).", "Call \\lstinline|optimizer.step()| after \\lstinline|loss.backward()|.",
         {"require": ["\\lstinline"]}),
    ]
    return drill(rng, items, preamble=LST)


# ================================================================= hard

@template(M, "hard")
@fixed
def adam_alg(rng):
    return L("**Adam as an algorithm** (the guide's drill). Recreate the target: an algorithm float with a \\While "
             "loop, the bias-corrected moments and the update.", ADAM, preamble=ALG, require=["\\While"])


@template(M, "hard")
def fix_algorithm(rng):
    good = ("\\begin{algorithmic}[1]\n  \\For{$i = 1$ \\textbf{to} $n$}\n    \\If{$x_i < 0$}\n"
            "      \\State $x_i \\gets 0$\n    \\EndIf\n  \\EndFor\n\\end{algorithmic}")
    items = [
        ("This doesn't compile (Some blocks are not closed!!!): the \\If is never closed. Add the missing \\EndIf.",
         good, {"starter": good.replace("    \\EndIf\n", "")}),
        ("This doesn't compile (Some blocks are not closed!!!): the \\For is never closed. Add the missing \\EndFor.",
         good, {"starter": good.replace("  \\EndFor\n", "")}),
        ("This doesn't compile: \\State is undefined because algpseudocode isn't loaded. You can't change the "
         "preamble, so write the same steps as a plain enumerate list instead: \u201cset $x_i \\gets 0$ for "
         "negative $x_i$\u201d as one item, and \u201creturn $x$\u201d as another.",
         "\\begin{enumerate}\n  \\item set $x_i \\gets 0$ for negative $x_i$\n  \\item return $x$\n\\end{enumerate}",
         {"preamble": AMS, "starter": "\\begin{algorithmic}[1]\n  \\State set $x_i \\gets 0$ for negative $x_i$\n"
                                      "  \\State \\Return $x$\n\\end{algorithmic}"}),
        ("This doesn't compile: the listing's language is misspelled (Package Listings Error: Couldn't load "
         "requested language). Fix it: Python.",
         "\\begin{lstlisting}[language=Python]\nprint(\"hi\")\n\\end{lstlisting}",
         {"preamble": LST, "starter": "\\begin{lstlisting}[language=Pyhton]\nprint(\"hi\")\n\\end{lstlisting}"}),
    ]
    return drill(rng, items, preamble=ALG)


# ================================================================= challenge

@template(M, "challenge")
@fixed
def alg_with_ref(rng):
    sol = SGD + "\n\\Cref{alg:sgd} stops early; line~5 is the update."
    return L("**Reference the algorithm** (the guide's drill). Recreate the target: the guide's SGD algorithm, then "
             "a sentence starting with \\Cref to it.", sol, preamble=BOTH, require=["\\Cref", "env:algorithm"])


@template(M, "challenge")
@fixed
def method_section(rng):
    code = PY[1]
    sol = ("\\section{Method}\nThe loop in \\cref{lst:train} implements \\cref{alg:step}.\n"
           "\\begin{algorithm}[t]\n\\caption{One epoch}\\label{alg:step}\n\\begin{algorithmic}[1]\n"
           "  \\ForAll{mini-batches $(x, y)$}\n    \\State zero the gradients\n"
           "    \\State $\\ell \\gets \\operatorname{CE}(f(x), y)$\n    \\State backpropagate $\\ell$\n"
           "    \\State step the optimizer \\Comment{$\\theta \\gets \\theta - \\eta g$}\n  \\EndFor\n"
           "\\end{algorithmic}\n\\end{algorithm}\n"
           f"\\begin{{lstlisting}}[language=Python, numbers=left, frame=single, caption={{The training loop}}, "
           f"label=lst:train]\n{code}\n\\end{{lstlisting}}")
    return L("**Pseudocode and code.** Recreate the target: a section, an algorithm, and its PyTorch version as a "
             "captioned, labelled listing, cross-referenced with \\cref.", sol, preamble=BOTH,
             require=["\\cref", "env:algorithm", "env:lstlisting"])

"""Grading Vim problems (COMP_GUIDE §9, generator/sections/s09_vim).

The session's keys are replayed on the problem's machine, the reference keys are typed into a fresh copy, and Vim's
state at the end of each is compared: what the problem's `vim["check"]` names (the text of the buffers by default).
`require` / `forbid` name commands as Vim.used records them ("ciw", "dd", "v_d", ":s_g", "i_<C-w>" ...: what the user
typed, macros and . included), and `vim["max_keys"]` is a par: the keys pressed in Vim, VimGolf style.
"""

from __future__ import annotations

import re

ARROWS = ("<Left>", "<Right>", "<Up>", "<Down>")
GROUPS = {   # forbid / require names that stand for several commands
    "arrows": [p + a for a in ARROWS for p in ("", "i_", "v_", "c_")],
    "hjkl": ["h", "j", "k", "l"],
    "insert_arrows": ["i_" + a for a in ARROWS],
    "mouse": ["<LeftMouse>"],
}
SHOWN = {"<CR>": "Enter", "<Esc>": "Esc", "<BS>": "Backspace", "<Tab>": "Tab", "<Del>": "Delete", "<Nul>": "Ctrl-Space"}
MODE_NAMES = {"v_": "in Visual mode", "i_": "in Insert mode", "c_": "on the command line"}


def nice(name: str) -> str:
    """A command as the user would type it, for messages: "v_d" -> "`d` in Visual mode", "<C-a>" -> "`Ctrl-A`"."""
    if name in GROUPS:
        return {"arrows": "the arrow keys", "hjkl": "h j k l", "insert_arrows": "the arrow keys in Insert mode",
                "mouse": "the mouse"}[name]
    if name.startswith("key:"):
        name = name[4:]
    where = ""
    for pre, label in MODE_NAMES.items():
        if name.startswith(pre) and len(name) > 2:
            name, where = name[2:], " " + label
            break
    if name.startswith(":s_"):
        return f"the `{name[3:]}` flag of `:s`"
    if name.startswith(":range"):
        name = ":10,20" + name[6:]

    def key(m):
        k = m.group(0)
        if k in SHOWN:
            return SHOWN[k]
        mm = re.fullmatch(r"<C-(.+)>", k)
        if mm:
            return "Ctrl-" + (mm.group(1).upper() if len(mm.group(1)) == 1 else mm.group(1))
        return k
    return "`" + re.sub(r"<[^<>]+>", key, name) + "`" + where


def _used(used, name):
    return any(used.get(n) for n in GROUPS.get(name, [name]))


def session_state(t) -> dict | None:
    """Vim's state at the end of a LiveTerm session (the last Vim it ran), plus the files in the home directory."""
    v = getattr(t.world, "vim_last", None)
    if v is None:
        return None
    st = v.state()
    st["keys"] = v.nkeys_total
    st["used"] = dict(v.used)
    st["open"] = t.vim() is not None
    st["files"] = home_files(t)
    st["bash_vi"] = "vi" in t.sh.opts
    return st


def home_files(t) -> dict:
    """{path relative to ~: text} for the regular files under the home directory (dot-directories skipped)."""
    fs = t.fs
    home = t.home if hasattr(t, "home") else "/home/ada"
    out = {}

    def walk(path, rel, depth):
        try:
            node = fs.lookup(path)
        except Exception:  # noqa: BLE001
            return
        if node.type == "f":
            out[rel] = bytes(node.data).decode("utf-8", "replace")
            return
        if node.type != "d" or depth > 6:
            return
        for name in sorted(fs.listdir(node)):
            if name.startswith(".") and name not in (".vimrc",):
                continue
            walk(path.rstrip("/") + "/" + name, f"{rel}/{name}" if rel else name, depth + 1)
    walk(home, "", 0)
    return out


def reference_state(problem: dict, setup: str | None = None) -> dict:
    from .live import vim_session
    t = vim_session(problem, setup=setup)
    st = session_state(t)
    if st is None:
        raise ValueError("the reference solution never opened Vim")
    return st


def _q(s: str, n: int = 70) -> str:
    s = s if len(s) <= n else s[:n - 1] + "…"
    return "`" + s + "`" if s else "(an empty line)"


def text_message(name: str, exp: list, got: list, what: str = "") -> str | None:
    if exp == got:
        return None
    label = f"`{name}`{what}" if name else "The buffer"
    for i, (e, g) in enumerate(zip(exp, got)):
        if e != g:
            return f"Line {i + 1} of {label} should be {_q(e)}, but it is {_q(g)}."
    if len(got) > len(exp):
        return (f"{label} has {len(got)} lines; it should have {len(exp)} (line {len(exp) + 1} is {_q(got[len(exp)])}, "
                f"which shouldn't be there).")
    return (f"{label} has {len(got)} lines; it should have {len(exp)} (line {len(got) + 1} should be "
            f"{_q(exp[len(got)])}).")


def _bufs(st):
    return {b["name"]: b for b in st["buffers"] if b["kind"] == "file" and b["name"]}


def _shape_text(node) -> str:
    if isinstance(node, str):
        return node
    kind, kids = node
    sep = " | " if kind == "row" else " / "
    return "(" + sep.join(_shape_text(k) for k in kids) + ")"


def _count_windows(node) -> int:
    return 1 if isinstance(node, str) else sum(_count_windows(k) for k in node[1])


def _fold_text(folds) -> str:
    if not folds:
        return "none"
    return ", ".join(f"{a + 1}-{b + 1} ({'closed' if c else 'open'})" for a, b, c in folds)


def _reg_text(r) -> str:
    if not r:
        return "nothing"
    text, kind = r[0], r[1]
    shown = text.replace("\n", "⏎")
    return _q(shown, 60) + (" (whole lines)" if kind == "V" else "")


def compare(spec: dict, exp: dict, got: dict) -> str | None:
    """The first difference that matters to the problem, as a message (None: they match)."""
    checks = spec.get("check") or ["text"]
    eb, gb = _bufs(exp), _bufs(got)
    for c in checks:
        if c == "text":
            for name, b in eb.items():
                if name not in gb:
                    return f"`{name}` isn't open in Vim (it should be)."
                msg = text_message(name, b["lines"], gb[name]["lines"])
                if msg:
                    return msg
        elif c == "files":
            names = spec.get("files") or sorted(set(exp["files"]) | set(got["files"]))
            for name in names:
                e, g = exp["files"].get(name), got["files"].get(name)
                if e == g:
                    continue
                if e is None:
                    return f"`{name}` shouldn't have been saved (written to disk)."
                if g is None:
                    return f"There's no `{name}` on disk: it should be saved (:w writes the file)."
                if name in gb and gb[name]["lines"] == e.split("\n")[:-1] and gb[name].get("modified"):
                    return f"`{name}` is right in Vim but not saved yet: write it with :w."
                return text_message(name, e.split("\n")[:-1] if e.endswith("\n") else e.split("\n"),
                                    g.split("\n")[:-1] if g.endswith("\n") else g.split("\n"), " on disk")
        elif c == "quit":
            if (exp["quit"] is not None) != (got["quit"] is not None):
                return "Vim should have exited at the end." if exp["quit"] is not None else "Vim should still be open."
        elif c == "cursor":
            if exp["buf"] != got["buf"]:
                return f"The cursor should end in `{exp['buf']}` (it's in `{got['buf'] or '[No Name]'}`)."
            (el, ec), (gl, gc) = exp["cursor"], got["cursor"]
            if (el, ec) != (gl, gc):
                return (f"The cursor should end on line {el + 1}, column {ec + 1} (the ruler shows {el + 1},{ec + 1}); "
                        f"it's on line {gl + 1}, column {gc + 1}.")
        elif c == "line":
            if exp["cursor"][0] != got["cursor"][0] or exp["buf"] != got["buf"]:
                return f"The cursor should end on line {exp['cursor'][0] + 1} (it's on line {got['cursor'][0] + 1})."
        elif c == "topline":
            if exp["topline"] != got["topline"]:
                return (f"The window should show line {exp['topline'] + 1} at the top; it shows line "
                        f"{got['topline'] + 1}.")
        elif c == "layout":
            if exp["tabs"] != got["tabs"]:
                return f"There should be {exp['tabs']} tab page(s); there are {got['tabs']}."
            for i, (e, g) in enumerate(zip(exp["layout"], got["layout"])):
                if e != g:
                    where = f" in tab {i + 1}" if exp["tabs"] > 1 else ""
                    n_e, n_g = _count_windows(e), _count_windows(g)
                    if n_e != n_g:
                        return f"There should be {n_e} window(s){where}; there are {n_g}."
                    return (f"The windows{where} should be {_shape_text(e)}, but they are {_shape_text(g)} "
                            f"(a | b: side by side, a / b: one above the other).")
            if exp["tab"] != got["tab"]:
                return f"Tab page {exp['tab'] + 1} should be the current one (it's {got['tab'] + 1})."
            ec = [w for w in exp["windows"] if w["current"]]
            gc = [w for w in got["windows"] if w["current"]]
            if ec and gc and (ec[0]["buf"], ec[0]["row"], ec[0]["col"]) != (gc[0]["buf"], gc[0]["row"], gc[0]["col"]):
                return f"The cursor should end in the window showing `{ec[0]['buf']}`" + (
                    " (the other one of the two)" if ec[0]["buf"] == gc[0]["buf"] else "") + "."
        elif c == "sizes":
            es = [(w["buf"], w["height"], w["width"]) for w in exp["windows"]]
            gs = [(w["buf"], w["height"], w["width"]) for w in got["windows"]]
            if es != gs:
                for e, g in zip(es, gs):
                    if e != g:
                        return (f"The window showing `{e[0]}` should be {e[1]} lines high and {e[2]} columns wide; "
                                f"it is {g[1]} × {g[2]}.")
                return "The windows aren't the right sizes."
        elif c == "buffers":
            e = sorted(b["name"] for b in exp["buffers"] if b["listed"] and b["kind"] == "file")
            g = sorted(b["name"] for b in got["buffers"] if b["listed"] and b["kind"] == "file")
            if e != g:
                return f"The buffer list (:ls) should hold {', '.join(e) or 'nothing'}; it holds {', '.join(g) or 'nothing'}."
        elif c == "regs":
            for r in spec.get("regs", ""):
                e, g = exp["regs"].get(r), got["regs"].get(r)
                if (e or None) != (g or None) and not (not e and not g):
                    return f"Register `{r}` should hold {_reg_text(e)}; it holds {_reg_text(g)}."
        elif c == "marks":
            for m in spec.get("marks", ""):
                src_e = exp["global_marks"] if m.isupper() else exp["marks"]
                src_g = got["global_marks"] if m.isupper() else got["marks"]
                e, g = src_e.get(m), src_g.get(m)
                if m.isupper():
                    e = e and (e[0], tuple(e[1]))
                    g = g and (g[0], tuple(g[1]))
                else:
                    e = e and tuple(e)
                    g = g and tuple(g)
                if e != g:
                    if g is None:
                        return f"Mark `{m}` isn't set (m{m} sets it)."
                    pos = e[1] if m.isupper() else e
                    return f"Mark `{m}` should be on line {pos[0] + 1}, column {pos[1] + 1}."
        elif c == "folds":
            if exp["folds"] != got["folds"]:
                return f"The folds should be: {_fold_text(exp['folds'])}; they are: {_fold_text(got['folds'])}."
        elif c == "options":
            for o in spec.get("options", []):
                e, g = exp["options"].get(o), got["options"].get(o)
                if e != g:
                    if isinstance(e, bool):
                        return f"The option '{o}' should be {'on' if e else 'off'} (:set {'' if e else 'no'}{o})."
                    return f"The option '{o}' should be {e!r}; it is {g!r}."
        elif c == "maps":
            for mode, maps in exp["maps"].items():
                for lhs, rhs in maps.items():
                    g = got["maps"].get(mode, {}).get(lhs)
                    if g != rhs:
                        what = {"n": "Normal", "i": "Insert", "v": "Visual", "x": "Visual", "c": "Command-line",
                                "o": "Operator-pending"}.get(mode, mode)
                        return (f"There should be a {what}-mode mapping {lhs} → {rhs}" +
                                (f" (it maps to {g})." if g else "."))
            if exp.get("leader") != got.get("leader"):
                return f"mapleader should be {exp.get('leader')!r}."
        elif c == "bash_vi":
            if exp["bash_vi"] != got["bash_vi"]:
                return "Bash's line editing should be in vi mode (`set -o vi` at the prompt)." if exp["bash_vi"] \
                    else "Bash's line editing should be back in emacs mode (`set -o emacs`)."
        elif c == "mode":
            if exp["mode"] != got["mode"]:
                names = {"n": "Normal", "i": "Insert", "v": "Visual", "V": "Visual Line", "\x16": "Visual Block",
                         "R": "Replace", "c": "Command-line"}
                return f"Vim should end in {names.get(exp['mode'], exp['mode'])} mode."
    return None


def matches(spec: dict, exp: dict, got: dict) -> bool:
    return compare(spec, exp, got) is None


def _result(**kw):
    out = {"correct": False, "compiled": True, "stdout": "", "value": None, "error": None, "message": "", "console": []}
    out.update(kw)
    return out


RESET = " Press Reset to start over."


def checker(problem: dict, log: list, end) -> dict:
    from .check import SetupError
    from .directives import DirectiveError
    from .live import LiveTerm
    spec = problem.get("vim") or {}
    try:
        t = LiveTerm(problem, problem.get("setup", ""))
        t.replay(log, end)
        got = session_state(t)
        if got is None:
            return _result(compiled=False, message="Vim hasn't been started yet: open it at the prompt (for example "
                                                   "`vim notes.py`), do the task, then Submit.")
        started = any(ln.startswith("#: vim ") or ln == "#: vim" for ln in problem.get("setup", "").split("\n"))
        if not got["keys"] and started:
            return _result(compiled=False, message="You haven't pressed any keys in Vim yet: do the task in the "
                                                   "terminal, then Submit.")
        used = got["used"]
        for req in problem.get("require") or []:
            alts = req.split("|")
            if not any(_used(used, a) for a in alts):
                return _result(compiled=False, message="This problem asks you to use " +
                               " or ".join(nice(a) for a in alts) + "." + RESET)
        for f in problem.get("forbid") or []:
            if _used(used, f):
                return _result(compiled=False, message=f"Solve this one without {nice(f)}." + RESET)
        exp = reference_state(problem)
        msg = compare(spec, exp, got)
        if msg:
            return _result(message=msg)
        par = spec.get("max_keys")
        if par and got["keys"] > par:
            return _result(compiled=False, message=f"Right result, but in {got['keys']} keystrokes: par for this one "
                                                   f"is {par}. Find a shorter way." + RESET)
        note = f"Done in {got['keys']} keystrokes" + (f" (par {par})." if par else ".")
        return _result(correct=True, message=note)
    except (SetupError, DirectiveError) as e:
        return _result(error=f"The problem's setup failed: {e}")

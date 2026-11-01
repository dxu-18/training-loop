""":terminal: a shell in a window. Each line typed runs on the machine (bash-lite) and its output is appended.
<C-w>N goes to Terminal-Normal mode (move and yank like in any buffer), i goes back; exit closes the window."""

from __future__ import annotations

from .engine import Window


def open_terminal(v, cmd="", vertical=False):
    b = v.new_buffer(name="!/bin/bash" if not cmd else f"!{cmd}", kind="terminal", listed=True)
    b.opts["modifiable"] = False
    prompt = v.io.prompt(v.cwd)
    b.extra.update({"running": True, "line": "", "prompt": prompt, "cwd": v.cwd})
    b.lines[:] = [prompt]
    v.split(vertical, buf=b)
    if not vertical:
        v.win.req_height = max(3, (v.height - 2) // 2)
        v.layout()
    v.term_mode = True
    v.used[":terminal"] += 1
    if cmd:
        b.extra["line"] = cmd
        run_line(v, b)
        b.extra["running"] = False
        b.lines.append("")
        b.name = f"!{cmd} [finished]"
        v.term_mode = False


def run_line(v, b):
    line = b.extra["line"]
    b.lines[-1] = b.extra["prompt"] + line
    b.extra["line"] = ""
    if line.strip() in ("exit", "logout"):
        b.extra["running"] = False
        v.term_mode = False
        w = next((w for w in v.all_windows() if w.buf is b), None)
        if w is not None and len(v.all_windows()) > 1:
            v.close_window(w, force=True)
        else:
            b.lines.append("[Process exited 0]")
        return
    out = v.io.run(line, cwd=b.extra["cwd"], keep_cwd=True)
    if out.get("cwd"):
        b.extra["cwd"] = out["cwd"]
    text = (out.get("stdout") or "") + (out.get("stderr") or "")
    if text:
        b.lines.extend(text.rstrip("\n").split("\n"))
    b.extra["prompt"] = v.io.prompt(b.extra["cwd"])
    b.lines.append(b.extra["prompt"])


def term_key(v, key):
    b = v.buf
    if b.kind != "terminal" or not b.extra.get("running"):
        v.term_mode = False
        return v.normal_key(key)
    if b.extra.get("ctrl_w"):
        b.extra["ctrl_w"] = False
        if key in ("N", "<C-n>"):
            v.term_mode = False
            v.used["<C-w>N"] += 1
            return
        v.term_mode = False
        v.window_cmd(key, None)
        if v.buf is b:
            v.term_mode = True
        return
    if key == "<C-w>":
        b.extra["ctrl_w"] = True
        return
    if key in ("<CR>", "<NL>"):
        run_line(v, b)
    elif key == "<BS>":
        b.extra["line"] = b.extra["line"][:-1]
    elif key == "<C-u>":
        b.extra["line"] = ""
    elif key == "<C-c>":
        b.lines[-1] = b.extra["prompt"] + b.extra["line"] + "^C"
        b.extra["line"] = ""
        b.lines.append(b.extra["prompt"])
    elif key == "<C-d>":
        if not b.extra["line"]:
            b.extra["line"] = "exit"
            run_line(v, b)
    elif len(key) == 1:
        b.extra["line"] += key
    if b.extra.get("running") and v.buf is b:
        b.lines[-1] = b.extra["prompt"] + b.extra["line"]
        v.win.cursor = [len(b.lines) - 1, len(b.lines[-1])]

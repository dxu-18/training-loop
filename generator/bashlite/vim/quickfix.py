"""The quickfix window (:copen) for :vimgrep results."""

from __future__ import annotations


def lines_of(v):
    out = []
    for e in v.qf:
        out.append(f"{e['file']}|{e['line'] + 1} col {e['col'] + 1}| {e['text'].strip()}")
    return out or [""]


def open_qf(v, height=10):
    for w in v.all_windows():
        if w.buf.kind == "quickfix":
            v.goto_win(w)
            refresh(v)
            return
    b = None
    for x in v.buffers.values():
        if x.kind == "quickfix":
            b = x
    if b is None:
        b = v.new_buffer(name="[Quickfix List]", kind="quickfix", listed=False)
    b.lines[:] = lines_of(v)
    b.opts["modifiable"] = False
    # at the bottom, full width
    t = v.tab
    from .engine import Window
    w = Window(b)
    w.req_height = height
    t["layout"] = ["col", [t["layout"], ["leaf", w]]]
    t["prev"] = v.win
    t["win"] = w
    w.cursor = [v.qf_idx, 0]
    v.layout()
    v.used[":copen"] += 1


def refresh(v):
    for w in v.all_windows(all_tabs=True):
        if w.buf.kind == "quickfix":
            w.buf.lines[:] = lines_of(v)
            w.cursor = [min(v.qf_idx, len(w.buf.lines) - 1), 0]


def qf_key(v, key):
    if key != "<CR>":
        return False
    from .ex import qf_jump
    qf_jump(v, v.win.cursor[0])
    return True

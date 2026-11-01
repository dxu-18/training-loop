""":help: a help window with the entry for a topic (the descriptions are the ones the problems' Instructions use)."""

from __future__ import annotations

from .engine import VimError

FILES = {"motion": "motion.txt", "change": "change.txt", "insert": "insert.txt", "visual": "visual.txt",
         "pattern": "pattern.txt", "undo": "undo.txt", "windows": "windows.txt", "tabpage": "tabpage.txt",
         "fold": "fold.txt", "options": "options.txt", "map": "map.txt", "autocmd": "autocmd.txt",
         "quickfix": "quickfix.txt", "terminal": "terminal.txt", "spell": "spell.txt", "various": "various.txt",
         "editing": "editing.txt", "starting": "starting.txt", "registers": "change.txt", "help": "help.txt"}


def open_help(v, topic, vertical=False):
    from .catalog import CATALOG, SECTIONS
    if not topic:
        name, lines = "help.txt", [
            "*help.txt*\tFor Vim version 9.1.  Last change: 2024 Jan 02", "",
            "\t\t\tVIM - main help file", "",
            "Move around:  Use the cursor keys, or \"h\" to go left,\t       h   l",
            "\t      \"j\" to go down, \"k\" to go up, \"l\" to go right.\t\t j",
            "Close this window:  Use \":q<Enter>\".",
            "Get out of Vim:  Use \":qa!<Enter>\" (careful, all changes are lost!).", "",
            "Jump to a subject:  Position the cursor on a tag (e.g. |bars|) and hit CTRL-].",
            "Get specific help:  It is possible to go directly to whatever you want help",
            "\t\t  on, by giving an argument to the |:help| command.", "",
            "\t      WHAT\t\t\tPREPEND    EXAMPLE\t~",
            "\t  Normal mode command\t\t\t   :help x",
            "\t  Visual mode command\t\t  v_\t   :help v_u",
            "\t  Insert mode command\t\t  i_\t   :help i_<Esc>",
            "\t  Command-line command\t\t  :\t   :help :quit",
            "\t  Option\t\t\t  '\t   :help 'textwidth'"]
    else:
        key = topic.strip("'")
        # a whole page: :help registers, :help windows, :help fold ...
        page = {"registers": "registers", "windows": "windows", "folding": "fold", "fold": "fold",
                "undo-tree": "undo", "spell": "spell", "quickfix": "quickfix", "mapping": "map", "autocmd": "autocmd",
                "tabpage": "tabpage", "terminal": "terminal", "pattern": "pattern", "visual-mode": "visual",
                "insert": "insert", "options": "options"}.get(key)
        if page is not None:
            first = next(k for k, e in CATALOG.items() if e.get("help") == page)
            topic = key = first
        entry = CATALOG.get(topic) or CATALOG.get(key) or CATALOG.get(":" + key.lstrip(":")) or \
            CATALOG.get(key.replace("CTRL-", "<C-").replace("ctrl-", "<C-") + (">" if key.lower().startswith("ctrl-") else ""))
        if entry is None:
            cands = [k for k in CATALOG if key in k]
            entry = CATALOG.get(cands[0]) if cands else None
            if entry is None:
                raise VimError(f"E149: Sorry, no help for {topic}")
            key = cands[0]
        sect = SECTIONS.get(entry.get("help", "various"), "various")
        name = FILES.get(entry.get("help", "various"), "various.txt")
        lines = [f"*{name}*\tFor Vim version 9.1.  Last change: 2024 Jan 02", "", sect, "",
                 f"{key:<24}*{key}*", f"\t\t\t{entry['desc']}", ""]
        for k, e in CATALOG.items():
            if e.get("help") == entry.get("help") and k != key:
                lines.append(f"{k:<24}*{k}*")
                lines.append(f"\t\t\t{e['desc']}")
                lines.append("")
    # an existing help window is reused
    for w in v.all_windows():
        if w.buf.kind == "help":
            v.goto_win(w)
            break
    else:
        b = v.new_buffer(name=name, kind="help", listed=False)
        if vertical:
            v.split(True, buf=b)
        else:
            from .engine import Window
            v.split(False, buf=b)
            v.win.req_height = max(5, (v.height - 3) // 2)
            v.layout()
    b = v.buf
    b.name = name
    b.lines[:] = lines
    b.undo = type(b.undo)(b.lines, v.now)          # the text is the help file, not a change to undo
    b.opts["modifiable"] = False
    b.readonly = True
    b.modified = False
    target = next((i for i, x in enumerate(lines) if x.startswith(f"{topic:<24}") or f"*{topic}*" in x), 0)
    v.set_cursor(target, 0)
    v.win.topline = target
    v.used[":help"] += 1

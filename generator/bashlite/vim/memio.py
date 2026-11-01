"""An in-memory machine for Vim without bash-lite (tests): files in a dict, no shell."""

from __future__ import annotations

import fnmatch


class MemIO:
    def __init__(self, files=None, cwd="/home/ada", readonly=()):
        self.files = dict(files or {})
        self.cwd = cwd
        self.ro = set(readonly)

    def _abs(self, p):
        return p if p.startswith("/") else self.cwd.rstrip("/") + "/" + p

    def read(self, path):
        return self.files.get(path)

    def write(self, path, text, force=False):
        if path in self.ro and not force:
            return "Permission denied"
        self.files[path] = text
        return None

    def writable(self, path):
        return path not in self.ro

    def exists(self, path):
        return path in self.files or self.isdir(path)

    def isdir(self, path):
        pre = path.rstrip("/") + "/"
        return any(f.startswith(pre) for f in self.files)

    def listdir(self, path):
        pre = path.rstrip("/") + "/"
        seen = {}
        for f in self.files:
            if f.startswith(pre):
                rest = f[len(pre):]
                name = rest.split("/", 1)[0]
                seen[name] = "/" in rest
        return [(n, d, False) for n, d in sorted(seen.items())]

    def glob(self, cwd, pattern):
        pat = self._abs(pattern) if not pattern.startswith("/") else pattern
        out = []
        for f in sorted(self.files):
            if fnmatch.fnmatch(f, pat) or ("**/" in pat and fnmatch.fnmatch(f, pat.replace("**/", ""))):
                out.append(f[len(cwd.rstrip("/")) + 1:] if f.startswith(cwd.rstrip("/") + "/") else f)
        return out

    def complete(self, cwd, part):
        names = [n + ("/" if d else "") for n, d, _ in self.listdir(cwd)]
        return [n for n in names if n.startswith(part)]

    def run(self, cmd, stdin=None, cwd=None, keep_cwd=False):
        """A few filters (sort, cat), enough for the tests; bash-lite runs the real thing."""
        import shlex
        argv = shlex.split(cmd)
        if argv and argv[0] in ("sort", "cat") and stdin is not None:
            lines = stdin.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if argv[0] == "sort":
                flags = "".join(a.lstrip("-") for a in argv[1:] if a.startswith("-"))
                lines = sorted(lines, key=(lambda x: float(x.split()[0]) if x.split() and
                               x.split()[0].lstrip("-").replace(".", "", 1).isdigit() else 0.0) if "n" in flags else None,
                               reverse="r" in flags)
                if "u" in flags:
                    lines = list(dict.fromkeys(lines))
            return {"stdout": "".join(x + "\n" for x in lines), "stderr": "", "status": 0}
        return {"stdout": "", "stderr": f"bash: {argv[0] if argv else ''}: command not found\n", "status": 127}

    def prompt(self, cwd):
        return "$ "

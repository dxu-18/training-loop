"""tar, gzip/gunzip/zcat, zip/unzip, xz/unxz/xzcat, bzip2/bunzip2."""

from __future__ import annotations

import gzip as _gzip
import io
import re
import tarfile
import zipfile
import zlib

from ..exec import util
from ..kernel import NS
from ..pattern import fnmatch
from ..timeutil import local_dt, strftime
from ..vfs import FSError, mode_string


def _lzma():
    try:
        import lzma
        return lzma
    except ImportError:
        return None


def _bz2():
    try:
        import bz2
        return bz2
    except ImportError:
        return None


def compress(data: bytes, kind: str, mtime: int = 0, name: str = "") -> bytes:
    if kind == "gz":
        buf = io.BytesIO()
        with _gzip.GzipFile(filename=name, mode="wb", fileobj=buf, mtime=mtime, compresslevel=6) as f:
            f.write(data)
        return buf.getvalue()
    if kind == "xz":
        lz = _lzma()
        if lz is None:
            raise FSError("EINVAL")
        return lz.compress(data)
    if kind == "bz2":
        bz = _bz2()
        if bz is None:
            raise FSError("EINVAL")
        return bz.compress(data)
    return data


def detect(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":
        return "gz"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:3] == b"BZh":
        return "bz2"
    return ""


def decompress(data: bytes, kind: str) -> bytes:
    if kind == "gz":
        return _gzip.decompress(data)
    if kind == "xz":
        lz = _lzma()
        return lz.decompress(data)
    if kind == "bz2":
        bz = _bz2()
        return bz.decompress(data)
    return data


# ---------------------------------------------------------------------------- tar

def _tar_args(argv):
    """Expand bundled old-style options (tar czvf x.tgz dir) into a normal list."""
    if not argv:
        return argv
    first = argv[0]
    if not first.startswith("-") and re.fullmatch(r"[A-Za-z]+", first):
        out = []
        rest = list(argv[1:])
        for c in first:
            out.append("-" + c)
            if c in "fCTXbN" and rest:
                out.append(rest.pop(0))
        return out + rest
    return argv


@util("tar")
def u_tar(ctx):
    a = _tar_args(ctx.argv[1:])
    mode = None
    archive = None
    comp = None
    verbose = 0
    chdir = None
    excludes = []
    strip = 0
    to_stdout = False
    keep_old = False
    members = []
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x.startswith("--"):
            k, _, v = x.partition("=")
            if k in ("--create",):
                mode = "c"
            elif k == "--extract" or k == "--get":
                mode = "x"
            elif k == "--list":
                mode = "t"
            elif k == "--append":
                mode = "r"
            elif k == "--file":
                archive = v or nxt
                if not v:
                    i += 1
            elif k == "--gzip" or k == "--gunzip":
                comp = "gz"
            elif k == "--xz":
                comp = "xz"
            elif k == "--bzip2":
                comp = "bz2"
            elif k == "--auto-compress":
                comp = "auto"
            elif k == "--verbose":
                verbose += 1
            elif k == "--directory":
                chdir = v or nxt
                if not v:
                    i += 1
            elif k == "--exclude":
                excludes.append(v or nxt)
                if not v:
                    i += 1
            elif k == "--strip-components":
                strip = int(v)
            elif k == "--to-stdout":
                to_stdout = True
            elif k == "--keep-old-files":
                keep_old = True
            elif k in ("--version",):
                ctx.w("tar (GNU tar) 1.35\n")
                return 0
            elif k == "--help":
                ctx.w("Usage: tar [OPTION...] [FILE]...\nGNU 'tar' saves many files together into a single tape or disk archive,\n"
                      "and can restore individual files from the archive.\n\nExamples:\n"
                      "  tar -cf archive.tar foo bar  # Create archive.tar from files foo and bar.\n"
                      "  tar -tvf archive.tar         # List all files in archive.tar verbosely.\n"
                      "  tar -xf archive.tar          # Extract all files from archive.tar.\n")
                return 0
            i += 1
            continue
        if x.startswith("-") and len(x) > 1:
            j = 1
            while j < len(x):
                c = x[j]
                if c in "cxtru":
                    mode = c if c in "cxtr" else "r"
                elif c == "z":
                    comp = "gz"
                elif c == "J":
                    comp = "xz"
                elif c == "j":
                    comp = "bz2"
                elif c == "a":
                    comp = "auto"
                elif c == "v":
                    verbose += 1
                elif c == "O":
                    to_stdout = True
                elif c == "k":
                    keep_old = True
                elif c in "pPhm":
                    pass
                elif c in "fC":
                    val = x[j + 1:] if j + 1 < len(x) else nxt
                    if j + 1 >= len(x):
                        i += 1
                    if c == "f":
                        archive = val
                    else:
                        chdir = val
                    break
                else:
                    ctx.err(f"invalid option -- '{c}'")
                    ctx.e("Try 'tar --help' or 'tar --usage' for more information.\n")
                    return 2
                j += 1
            i += 1
            continue
        members.append(x)
        i += 1
    if mode is None:
        ctx.err("You must specify one of the '-Acdtrux', '--delete' or '--test-label' options")
        ctx.e("Try 'tar --help' or 'tar --usage' for more information.\n")
        return 2
    if archive is None:
        archive = "-"
    if comp == "auto" or (comp is None and mode == "c"):
        comp = "gz" if archive.endswith((".tgz", ".tar.gz", ".gz")) else "xz" if archive.endswith((".txz", ".xz")) else \
            "bz2" if archive.endswith((".bz2", ".tbz2")) else ("" if comp is None else None)
        comp = comp or None
    fs = ctx.fs
    base = ctx.cwd if chdir is None else ctx.abspath(chdir)
    if mode == "c":
        if not members:
            ctx.err("Cowardly refusing to create an empty archive")
            ctx.e("Try 'tar --help' or 'tar --usage' for more information.\n")
            return 2
        buf = io.BytesIO()
        status = 0
        warned = False
        listing = []
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
            for m in members:
                try:
                    node = fs.lookup(m, base, False)
                except FSError as e:
                    ctx.err(f"{m}: Cannot stat: {e.msg}")
                    status = 2
                    continue
                arcroot = m
                if arcroot.startswith("/"):
                    if not warned:
                        ctx.err("Removing leading `/' from member names")
                        warned = True
                    arcroot = arcroot.lstrip("/")
                for path, n in _walk_tar(fs, m, node, base):
                    name = (arcroot + path[len(m):]) if path.startswith(m) else path
                    name = name.lstrip("/") if name.startswith("/") else name
                    if any(fnmatch(e, path.rstrip("/").rsplit("/", 1)[-1]) or fnmatch(e, path) for e in excludes):
                        continue
                    ti = tarfile.TarInfo(name + ("/" if n.type == "d" and not name.endswith("/") else ""))
                    ti.mtime = n.mtime // NS
                    ti.mode = n.mode & 0o7777
                    ti.uid, ti.gid = n.uid, n.gid
                    ti.uname, ti.gname = ctx.host.user_name(n.uid), ctx.host.group_name(n.gid)
                    if n.type == "d":
                        ti.type = tarfile.DIRTYPE
                        tf.addfile(ti)
                    elif n.type == "l":
                        ti.type = tarfile.SYMTYPE
                        ti.linkname = n.target
                        tf.addfile(ti)
                    elif n.type == "f":
                        ti.size = len(n.data)
                        tf.addfile(ti, io.BytesIO(bytes(n.data)))
                    else:
                        continue
                    listing.append(ti)
        data = buf.getvalue()
        if comp:
            try:
                data = compress(data, comp, ctx.now // NS)
            except FSError:
                ctx.err(f"Cannot use compressed or remote archives")
                return 2
        for ti in listing:
            if verbose:
                line = _tv_line(ctx, ti) if verbose > 1 else ti.name
                (ctx.e if archive == "-" else ctx.w)(line + "\n")
        if archive == "-":
            yield from ctx.out(data)
        else:
            try:
                of = ctx.open(archive, "w")
                of.inode.data[:] = data
            except FSError as e:
                ctx.err(f"{archive}: Cannot open: {e.msg}")
                ctx.err("Error is not recoverable: exiting now")
                return 2
        if status:
            ctx.err("Exiting with failure status due to previous errors")
        return status
    # read an archive
    try:
        data = (yield from ctx.read_fd(0)) if archive == "-" else (yield from ctx.read_path(archive))
    except FSError as e:
        ctx.err(f"{archive}: Cannot open: {e.msg}")
        ctx.err("Error is not recoverable: exiting now")
        return 2
    kind = detect(data)
    try:
        if kind:
            data = decompress(data, kind)
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        infos = tf.getmembers()
    except Exception:  # noqa: BLE001
        ctx.err("This does not look like a tar archive")
        ctx.err("Exiting with failure status due to previous errors")
        return 2
    want = members
    found = set()
    status = 0
    for ti in infos:
        name = ti.name
        if want and not any(name == w or name.rstrip("/") == w.rstrip("/") or name.startswith(w.rstrip("/") + "/")
                            or fnmatch(w, name) for w in want):
            continue
        for w in want:
            if name == w or name.rstrip("/") == w.rstrip("/") or name.startswith(w.rstrip("/") + "/") or fnmatch(w, name):
                found.add(w)
        if any(fnmatch(e, name.rstrip("/").rsplit("/", 1)[-1]) for e in excludes):
            continue
        if mode == "t":
            ctx.w((_tv_line(ctx, ti) if verbose else name) + "\n")
            continue
        parts = name.split("/")
        if strip:
            parts = parts[strip:]
            if not parts or parts == [""]:
                continue
        rel = "/".join(parts)
        if verbose:
            ctx.w((_tv_line(ctx, ti) if verbose > 1 else name) + "\n")
        if to_stdout:
            if ti.isfile():
                ctx.w(tf.extractfile(ti).read())
            continue
        dest = base.rstrip("/") + "/" + rel.rstrip("/")
        try:
            if ti.isdir():
                n = fs.mkdir(dest, "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
                n.mode = ti.mode & ~ctx.proc.umask if ctx.uid else ti.mode
                n.mtime = ti.mtime * NS
            elif ti.issym():
                fs.mkdir(dest.rsplit("/", 1)[0] or "/", "/", 0o755, ctx.uid, ctx.proc.gid, parents=True)
                if fs.exists(dest, follow=False):
                    fs.unlink(dest)
                fs.symlink(ti.linkname, dest, "/", ctx.uid, ctx.proc.gid)
            elif ti.isfile():
                if keep_old and fs.exists(dest):
                    ctx.err(f"{rel}: Cannot open: File exists")
                    status = 2
                    continue
                content = tf.extractfile(ti).read()
                n = fs.write_file(dest, content, "/", ti.mode & ~ctx.proc.umask if ctx.uid else ti.mode, ctx.uid,
                                  ctx.proc.gid, ti.mtime * NS)
                n.mode = (ti.mode & ~ctx.proc.umask) if ctx.uid else ti.mode
        except FSError as e:
            ctx.err(f"{rel}: Cannot open: {e.msg}")
            status = 2
    for w in want:
        if w not in found:
            ctx.err(f"{w}: Not found in archive")
            status = 2
    if status:
        ctx.err("Exiting with failure status due to previous errors")
    return status


def _walk_tar(fs, path, node, base):
    yield path, node
    if node.type == "d":
        for name in fs.listdir(node):
            child = node.entries[name]
            yield from _walk_tar(fs, path.rstrip("/") + "/" + name, child, base)


def _tv_line(ctx, ti):
    class N:
        pass
    n = N()
    n.type = "d" if ti.isdir() else ("l" if ti.issym() else "f")
    n.mode = ti.mode
    mode = mode_string(n)
    ug = f"{ti.uname or ti.uid}/{ti.gname or ti.gid}"
    size = str(ti.size)
    width = 19
    pad = max(width - len(ug) - len(size), 1)
    when = strftime("%Y-%m-%d %H:%M", local_dt(ti.mtime * NS, ctx.tz))
    name = ti.name + (f" -> {ti.linkname}" if ti.issym() else "")
    return f"{mode} {ug}{' ' * pad}{size} {when} {name}"


# ---------------------------------------------------------------------------- gzip family

def _gz_main(ctx, decomp=False, to_stdout=False, kind="gz", suffix=".gz", name="gzip"):
    a = ctx.argv[1:]
    keep = force = verbose = listing = test = recursive = False
    files = []
    for x in a:
        if x.startswith("--"):
            m = {"--decompress": "d", "--uncompress": "d", "--stdout": "c", "--to-stdout": "c", "--keep": "k",
                 "--force": "f", "--verbose": "v", "--list": "l", "--test": "t", "--recursive": "r", "--best": "9",
                 "--fast": "1", "--quiet": "q"}
            if x not in m:
                ctx.err(f"unrecognized option '{x}'")
                return 1
            flags = m[x]
        elif x.startswith("-") and len(x) > 1:
            flags = x[1:]
        else:
            files.append(x)
            continue
        for c in flags:
            if c == "d":
                decomp = True
            elif c == "c":
                to_stdout = True
            elif c == "k":
                keep = True
            elif c == "f":
                force = True
            elif c == "v":
                verbose = True
            elif c == "l":
                listing = True
            elif c == "t":
                test = True
            elif c == "r":
                recursive = True
            elif c.isdigit() or c in "qnN":
                pass
            else:
                ctx.err(f"invalid option -- '{c}'")
                return 1
    fs = ctx.fs
    if not files:
        data = yield from ctx.read_fd(0)
        if decomp or listing:
            try:
                out = decompress(data, kind if detect(data) == "" else detect(data))
            except Exception:  # noqa: BLE001
                ctx.err("stdin: not in gzip format")
                return 1
        else:
            out = compress(data, kind, 0)
        yield from ctx.out(out)
        return 0
    if recursive:
        expanded = []
        for f in files:
            try:
                node = fs.lookup(f, ctx.cwd)
            except FSError:
                expanded.append(f)
                continue
            if node.type == "d":
                expanded += [p for p, n in fs.walk(node, f) if n.type == "f"]
            else:
                expanded.append(f)
        files = expanded
    status = 0
    if listing:
        ctx.w("         compressed        uncompressed  ratio uncompressed_name\n")
        tc = tu = 0
        for f in files:
            try:
                raw = yield from ctx.read_path(f)
                plain = decompress(raw, "gz")
            except Exception:  # noqa: BLE001
                ctx.err(f"{f}: not in gzip format")
                status = 1
                continue
            un = f[:-3] if f.endswith(".gz") else f
            ratio = (1 - (len(raw) - 0) / len(plain)) * 100 if plain else 0.0
            ctx.w(f"{len(raw):>19} {len(plain):>19} {ratio:5.1f}% {un}\n")
            tc += len(raw)
            tu += len(plain)
        return status
    for f in files:
        try:
            node = fs.lookup(f, ctx.cwd)
        except FSError as e:
            if decomp and not f.endswith(suffix) and fs.exists(f + suffix, ctx.cwd):
                f = f + suffix
                node = fs.lookup(f, ctx.cwd)
            else:
                ctx.err(f"{f}: {e.msg}")
                status = 1
                continue
        if node.type == "d":
            ctx.err(f"{f} is a directory -- ignored")
            status = 2
            continue
        data = bytes(node.data)
        if decomp or test:
            if not f.endswith((suffix, ".tgz", ".z", ".Z")) and not to_stdout and not test:
                ctx.err(f"{f}: unknown suffix -- ignored")
                status = 2
                continue
            k = detect(data)
            try:
                plain = decompress(data, k or kind)
            except Exception:  # noqa: BLE001
                ctx.err(f"{f}: not in {name} format")
                status = 1
                continue
            if test:
                if verbose:
                    ctx.w(f"{f}:\t OK\n")
                continue
            if to_stdout:
                yield from ctx.out(plain)
                continue
            target = f[:-len(suffix)] if f.endswith(suffix) else (f[:-4] + ".tar" if f.endswith(".tgz") else f[:-2])
            if fs.exists(target, ctx.cwd) and not force:
                ctx.err(f"{target} already exists; not overwritten")
                status = 2
                continue
            n = fs.write_file(target, plain, ctx.cwd, node.mode, node.uid, node.gid, node.mtime)
            n.mode = node.mode
            if verbose:
                ratio = (1 - len(data) / len(plain)) * 100 if plain else 0.0
                ctx.e(f"{f}:\t{ratio:5.1f}% -- replaced with {target}\n" if not keep else f"{f}:\t{ratio:5.1f}% -- created {target}\n")
            if not keep:
                fs.unlink(f, ctx.cwd)
            continue
        if f.endswith(suffix) and not force:
            ctx.err(f"{f} already has {suffix} suffix -- unchanged")
            status = 2
            continue
        packed = compress(data, kind, node.mtime // NS, f.rsplit("/", 1)[-1])
        if to_stdout:
            yield from ctx.out(packed)
            continue
        target = f + suffix
        if fs.exists(target, ctx.cwd) and not force:
            ctx.err(f"{target} already exists; not overwritten")
            status = 2
            continue
        n = fs.write_file(target, packed, ctx.cwd, node.mode, node.uid, node.gid, node.mtime)
        n.mode = node.mode
        if verbose:
            ratio = (1 - len(packed) / len(data)) * 100 if data else 0.0
            ctx.e(f"{f}:\t{ratio:5.1f}% -- {'created' if keep else 'replaced with'} {target}\n")
        if not keep:
            fs.unlink(f, ctx.cwd)
    return status


@util("gzip")
def u_gzip(ctx):
    return (yield from _gz_main(ctx))


@util("gunzip")
def u_gunzip(ctx):
    return (yield from _gz_main(ctx, decomp=True))


@util("zcat")
def u_zcat(ctx):
    return (yield from _gz_main(ctx, decomp=True, to_stdout=True))


@util("xz")
def u_xz(ctx):
    return (yield from _gz_main(ctx, kind="xz", suffix=".xz", name="xz"))


@util("unxz")
def u_unxz(ctx):
    return (yield from _gz_main(ctx, decomp=True, kind="xz", suffix=".xz", name="xz"))


@util("xzcat")
def u_xzcat(ctx):
    return (yield from _gz_main(ctx, decomp=True, to_stdout=True, kind="xz", suffix=".xz", name="xz"))


@util("bzip2")
def u_bzip2(ctx):
    return (yield from _gz_main(ctx, kind="bz2", suffix=".bz2", name="bzip2"))


@util("bunzip2")
def u_bunzip2(ctx):
    return (yield from _gz_main(ctx, decomp=True, kind="bz2", suffix=".bz2", name="bzip2"))


@util("zless", "zmore")
def u_zless(ctx):
    return (yield from _gz_main(ctx, decomp=True, to_stdout=True))


# ---------------------------------------------------------------------------- zip / unzip

def _zip_time(ns, ctx):
    d = local_dt(ns, ctx.tz)
    return (max(d.year, 1980), d.month, d.day, d.hour, d.minute, d.second - d.second % 2)


@util("zip")
def u_zip(ctx):
    a = ctx.argv[1:]
    recursive = quiet = junk = False
    delete = False
    excludes = []
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-x", "--exclude"):
            i += 1
            while i < len(a) and not a[i].startswith("-"):
                excludes.append(a[i])
                i += 1
            continue
        if x.startswith("-") and len(x) > 1:
            for c in x[1:]:
                if c == "r":
                    recursive = True
                elif c == "q":
                    quiet = True
                elif c == "j":
                    junk = True
                elif c == "d":
                    delete = True
                elif c.isdigit() or c in "uyv":
                    pass
            i += 1
            continue
        ops.append(x)
        i += 1
    if not ops:
        ctx.e("zip error: Nothing to do! (try: zip -r archive.zip . -i ...)\n")
        return 12
    zname = ops[0] if ops[0].endswith(".zip") else ops[0] + ".zip"
    paths = ops[1:]
    fs = ctx.fs
    existing = {}
    exists = fs.exists(zname, ctx.cwd)
    if exists:
        try:
            with zipfile.ZipFile(io.BytesIO(bytes(fs.lookup(zname, ctx.cwd).data))) as zf:
                for info in zf.infolist():
                    existing[info.filename] = (info, zf.read(info))
        except zipfile.BadZipFile:
            ctx.e(f"zip error: Zip file structure invalid ({zname})\n")
            return 3
    if delete:
        for p in paths:
            for name in list(existing):
                if fnmatch(p, name):
                    del existing[name]
                    if not quiet:
                        ctx.w(f"deleting: {name}\n")
        entries = []
    else:
        entries = []
        for p in paths:
            try:
                node = fs.lookup(p, ctx.cwd)
            except FSError:
                ctx.e(f"\tzip warning: name not matched: {p}\n")
                continue
            if node.type == "d":
                if recursive:
                    for path, n in fs.walk(node, p.rstrip("/")):
                        entries.append((path, n))
                else:
                    entries.append((p.rstrip("/"), node))
            else:
                entries.append((p, node))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        added = []
        for path, n in entries:
            arc = path.rsplit("/", 1)[-1] if junk else path.lstrip("/")
            if any(fnmatch(e, arc) for e in excludes):
                continue
            if n.type == "d":
                if junk:
                    continue
                arc = arc.rstrip("/") + "/"
                info = zipfile.ZipInfo(arc, _zip_time(n.mtime, ctx))
                info.external_attr = (0o40000 | (n.mode & 0o7777)) << 16
                verb = "updating" if arc in existing else "adding"
                existing.pop(arc, None)
                added.append((info, b"", f"  {verb}: {arc} (stored 0%)"))
                continue
            data = bytes(n.data)
            info = zipfile.ZipInfo(arc, _zip_time(n.mtime, ctx))
            info.external_attr = (0o100000 | (n.mode & 0o7777)) << 16
            comp = zlib.compressobj(6, zlib.DEFLATED, -15)
            packed = comp.compress(data) + comp.flush()
            if data and len(packed) < len(data):
                info.compress_type = zipfile.ZIP_DEFLATED
                pct = round(100 * (1 - len(packed) / len(data)))
                msg = f"(deflated {pct}%)"
            else:
                info.compress_type = zipfile.ZIP_STORED
                msg = "(stored 0%)"
            verb = "updating" if arc in existing else "adding"
            existing.pop(arc, None)
            added.append((info, data, f"  {verb}: {arc} {msg}"))
        for name, (info, data) in existing.items():
            zf.writestr(info, data)
        for info, data, msg in added:
            zf.writestr(info, data)
            if not quiet:
                ctx.w(msg + "\n")
    fs.write_file(zname, out.getvalue(), ctx.cwd, 0o644 & ~ctx.proc.umask | 0o644 & 0, ctx.uid, ctx.proc.gid)
    return 0
    yield


@util("unzip")
def u_unzip(ctx):
    a = ctx.argv[1:]
    listing = quiet = overwrite = never = test = pipe = False
    dest = None
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-d" and i + 1 < len(a):
            dest = a[i + 1]
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            for c in x[1:]:
                if c == "l":
                    listing = True
                elif c == "q":
                    quiet = True
                elif c == "o":
                    overwrite = True
                elif c == "n":
                    never = True
                elif c == "t":
                    test = True
                elif c == "p":
                    pipe = True
            i += 1
            continue
        ops.append(x)
        i += 1
    if not ops:
        ctx.w("UnZip 6.00 of 20 April 2009, by Debian. Original by Info-ZIP.\n\n"
              "Usage: unzip [-Z] [-opts[modifiers]] file[.zip] [list] [-x xlist] [-d exdir]\n")
        return 0
    zname = ops[0]
    fs = ctx.fs
    if not fs.exists(zname, ctx.cwd) and fs.exists(zname + ".zip", ctx.cwd):
        zname += ".zip"
    try:
        raw = bytes(fs.lookup(zname, ctx.cwd).data)
    except FSError:
        ctx.w(f"unzip:  cannot find or open {ops[0]}, {ops[0]}.zip or {ops[0]}.ZIP.\n")
        return 9
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        ctx.w(f"Archive:  {zname}\n  End-of-central-directory signature not found.  Either this file is not\n"
              f"  a zipfile, or it constitutes one disk of a multi-part archive.\n")
        return 9
    wanted = ops[1:]
    infos = [x for x in zf.infolist() if not wanted or any(fnmatch(w, x.filename) for w in wanted)]
    if listing:
        ctx.w(f"Archive:  {zname}\n  Length      Date    Time    Name\n---------  ---------- -----   ----\n")
        total = 0
        for info in infos:
            y, mo, d, h, mi, _ = info.date_time
            ctx.w(f"{info.file_size:>9}  {y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}   {info.filename}\n")
            total += info.file_size
        ctx.w(f"---------                     -------\n{total:>9}                     {len(infos)} file{'s' if len(infos) != 1 else ''}\n")
        return 0
    if pipe:
        for info in infos:
            if not info.is_dir():
                ctx.w(zf.read(info))
        return 0
    if not quiet:
        ctx.w(f"Archive:  {zname}\n")
    if test:
        for info in infos:
            ctx.w(f"    testing: {info.filename:<24} OK\n")
        ctx.w(f"No errors detected in compressed data of {zname}.\n")
        return 0
    base = dest if dest else ""
    if dest:
        fs.mkdir(dest, ctx.cwd, 0o755, ctx.uid, ctx.proc.gid, parents=True)
    from datetime import datetime
    from ..timeutil import localize, to_ns
    status = 0
    for info in infos:
        target = (base.rstrip("/") + "/" if base else "") + info.filename
        when = to_ns(localize(datetime(*info.date_time), ctx.tz))
        mode = (info.external_attr >> 16) & 0o7777
        if info.is_dir():
            if not fs.exists(target, ctx.cwd):
                n = fs.mkdir(target.rstrip("/"), ctx.cwd, 0o755, ctx.uid, ctx.proc.gid, parents=True)
                if not quiet:
                    ctx.w(f"   creating: {target}\n")
            continue
        if fs.exists(target, ctx.cwd) and not overwrite:
            if never:
                continue
            ctx.w(f"replace {target}? [y]es, [n]o, [A]ll, [N]one, [r]ename: ")
            from ..fileio import readline
            ans = yield from readline(ctx.k, ctx.proc.fds.get(0))
            ans = ans.decode(errors="replace").strip()
            if not ans:
                ctx.w("NULL\n(EOF or read error, treating as \"[N]one\" ...)\n")
                return 1
            if ans in ("A",):
                overwrite = True
            elif ans in ("N",):
                return 0
            elif ans not in ("y", "Y"):
                continue
        parent = target.rsplit("/", 1)[0] if "/" in target else ""
        if parent and not fs.exists(parent, ctx.cwd):
            fs.mkdir(parent, ctx.cwd, 0o755, ctx.uid, ctx.proc.gid, parents=True)
            if not quiet:
                ctx.w(f"   creating: {parent}/\n")
        data = zf.read(info)
        n = fs.write_file(target, data, ctx.cwd, mode or 0o644, ctx.uid, ctx.proc.gid, when)
        n.mode = (mode or 0o644) & ~ctx.proc.umask | (mode & 0o111 if mode else 0)
        if not quiet:
            verb = "inflating" if info.compress_type == zipfile.ZIP_DEFLATED else "extracting"
            ctx.w(f"  {verb}: {target}" + ("  " if verb == "inflating" else " ") + "\n" if False else
                  (f"  inflating: {target}  \n" if verb == "inflating" else f" extracting: {target}  \n"))
    return status

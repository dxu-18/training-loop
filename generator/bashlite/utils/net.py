"""ssh ssh-keygen ssh-copy-id ssh-agent ssh-add scp rsync curl wget ping."""

from __future__ import annotations

import base64
import hashlib
import json
import re

from ..exec import util
from ..kernel import NS
from ..pattern import fnmatch
from ..timeutil import local_dt, strftime
from ..vfs import FSError

INTERNET = {"example.com": "93.184.215.14", "www.example.com": "93.184.215.14", "api.github.com": "140.82.112.6",
            "github.com": "140.82.112.4", "raw.githubusercontent.com": "185.199.108.133", "httpbin.org": "54.208.105.16",
            "pypi.org": "151.101.0.223", "files.pythonhosted.org": "151.101.0.223", "huggingface.co": "18.155.68.94",
            "download.pytorch.org": "18.160.201.104", "google.com": "142.250.80.46", "www.google.com": "142.250.80.36",
            "localhost": "127.0.0.1"}


# ---------------------------------------------------------------------------- keys

def key_material(seed: str) -> bytes:
    return hashlib.sha256(seed.encode()).digest()


def pub_blob(kind: str, raw: bytes) -> bytes:
    name = b"ssh-ed25519" if kind == "ed25519" else b"ssh-rsa"
    if kind == "ed25519":
        return len(name).to_bytes(4, "big") + name + (32).to_bytes(4, "big") + raw[:32]
    body = hashlib.sha512(raw).digest() * 6
    return len(name).to_bytes(4, "big") + name + b"\x00\x00\x00\x03\x01\x00\x01" + (len(body) + 1).to_bytes(4, "big") + b"\x00" + body


def fingerprint(blob: bytes) -> str:
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def randomart(blob: bytes, title: str) -> str:
    digest = hashlib.sha256(blob).digest()
    w, h = 17, 9
    field = [[0] * h for _ in range(w)]
    x, y = w // 2, h // 2
    chars = " .o+=*BOX@%&#/^SE"
    for byte in digest:
        for _ in range(4):
            x += 1 if byte & 1 else -1
            y += 1 if byte & 2 else -1
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            if field[x][y] < len(chars) - 3:
                field[x][y] += 1
            byte >>= 2
    field[w // 2][h // 2] = len(chars) - 2
    field[x][y] = len(chars) - 1
    t = f"[{title}]"
    left = (w - len(t)) // 2
    top = "+" + "-" * left + t + "-" * (w - len(t) - left) + "+"
    rows = ["|" + "".join(chars[field[i][j]] for i in range(w)) + "|" for j in range(h)]
    b = "[SHA256]"
    lb = (w - len(b)) // 2
    bottom = "+" + "-" * lb + b + "-" * (w - len(b) - lb) + "+"
    return "\n".join([top] + rows + [bottom])


def parse_pub(text: str):
    parts = text.strip().split()
    if len(parts) >= 2 and parts[0].startswith("ssh-"):
        try:
            return parts[0], base64.b64decode(parts[1]), " ".join(parts[2:])
        except Exception:  # noqa: BLE001
            return None
    return None


def identity_pubs(ctx, idfiles):
    """Public key lines for the client's identities (default ~/.ssh/id_ed25519, id_rsa, plus agent keys)."""
    home = ctx.env.get("HOME", ctx.host.home_of(ctx.uid))
    files = idfiles or [f"{home}/.ssh/id_ed25519", f"{home}/.ssh/id_rsa"]
    out = []
    for f in files:
        f = f.replace("~", home, 1) if f.startswith("~") else f
        for cand in (f + ".pub", f):
            try:
                text = bytes(ctx.fs.lookup(cand, ctx.cwd).data).decode()
            except FSError:
                continue
            if cand.endswith(".pub"):
                p = parse_pub(text)
                if p:
                    out.append(p)
                    break
            elif "BEGIN OPENSSH PRIVATE KEY" in text:
                m = re.search(r"pubkey:([A-Za-z0-9+/=]+)", text)
                if m:
                    out.append(("ssh-ed25519", base64.b64decode(m.group(1)), ""))
                break
    sock = ctx.env.get("SSH_AUTH_SOCK")
    agent = ctx.world.__dict__.get("agents", {}).get(sock) if sock else None
    if agent:
        out += agent["keys"]
    return out


# ---------------------------------------------------------------------------- ssh config

def ssh_config(ctx, alias):
    """Effective options for a destination alias (first value wins, like ssh -G)."""
    home = ctx.env.get("HOME", ctx.host.home_of(ctx.uid))
    opts = {}
    try:
        text = bytes(ctx.fs.lookup(f"{home}/.ssh/config", ctx.cwd).data).decode()
    except FSError:
        text = ""
    active = True
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\S+)\s*(?:=\s*|\s+)(.*)", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip().strip('"')
        if key == "host":
            pats = val.split()
            ok = False
            for pat in pats:
                if pat.startswith("!") and fnmatch(pat[1:], alias):
                    ok = False
                    break
                if fnmatch(pat, alias):
                    ok = True
            active = ok
            continue
        if key == "match":
            active = val.lower() == "all"
            continue
        if active:
            if key in ("localforward", "remoteforward", "identityfile"):
                opts.setdefault(key, []).append(val)
            else:
                opts.setdefault(key, val)
    return opts


def parse_dest(dest):
    user = None
    if "@" in dest:
        user, dest = dest.rsplit("@", 1)
    port = None
    m = re.match(r"ssh://(?:([^@]+)@)?([^:/]+)(?::(\d+))?", dest)
    if m:
        user = m.group(1) or user
        dest = m.group(2)
        port = int(m.group(3)) if m.group(3) else None
    return user, dest, port


class SshError(Exception):
    def __init__(self, msg, status=255):
        super().__init__(msg)
        self.msg, self.status = msg, status


def connect(ctx, dest, opts):
    """Resolve and authenticate. Returns (remote host, remote user, alias). Raises SshError."""
    user, alias, port = parse_dest(dest)
    cfg = ssh_config(ctx, alias)
    hostname = cfg.get("hostname", alias).replace("%h", alias)
    user = user or opts.get("user") or cfg.get("user") or ctx.host.user_name(ctx.uid)
    port = opts.get("port") or port or int(cfg.get("port", 22))
    idfiles = opts.get("identity") or cfg.get("identityfile")
    jump = opts.get("jump") or cfg.get("proxyjump")
    world = ctx.world
    host = world.resolve(hostname)
    if host is None:
        raise SshError(f"ssh: Could not resolve hostname {hostname}: Name or service not known")
    via = None
    if jump and jump != "none":
        via_host, via_user, _ = connect(ctx, jump, {})
        via = via_host
    if host.extra.get("private") and (via is None or via.name not in host.extra.get("reachable_from", [])):
        raise SshError(f"ssh: connect to host {hostname} port {port}: Connection timed out")
    if port != host.ssh_port:
        raise SshError(f"ssh: connect to host {hostname} port {port}: Connection refused")
    uid = host.uid_of(user)
    if uid is None or uid not in host.users:
        raise SshError(f"{user}@{hostname}: Permission denied (publickey).")
    rhome = host.users[uid].home
    try:
        auth = bytes(host.fs.lookup(f"{rhome}/.ssh/authorized_keys").data).decode()
    except FSError:
        auth = ""
    allowed = {parse_pub(line)[1] for line in auth.splitlines() if parse_pub(line)}
    keys = identity_pubs(ctx, idfiles if isinstance(idfiles, list) else ([idfiles] if idfiles else None))
    if not any(k[1] in allowed for k in keys) and host.authorized_keys_required:
        pw = host.extra.get("password")
        if pw is None:
            raise SshError(f"{user}@{hostname}: Permission denied (publickey).")
    return host, user, alias


def remote_proc(ctx, host, user, argv, fds):
    """A process for `user` on `host` that inherits our terminal fds."""
    k = ctx.k
    uid = host.uid_of(user)
    u = host.users[uid]
    p = k.spawn(host, argv, None, None, fds=fds, cwd=u.home, uid=uid, gid=u.gid)
    p.gids = host.gids_of(uid)
    p.ctty = ctx.proc.ctty
    p.tty = "pts/1"
    p.ppid = 1
    return p


@util("ssh")
def u_ssh(ctx):
    a = ctx.argv[1:]
    opts = {}
    forwards = []
    no_cmd = background = False
    tty_force = False
    show_config = False
    i = 0
    dest = None
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if dest is None and x.startswith("-") and len(x) > 1:
            j = 1
            while j < len(x):
                c = x[j]
                if c in "ipLlJoRDFEWbcmBS":
                    val = x[j + 1:] or nxt
                    if not x[j + 1:]:
                        i += 1
                    if c == "i":
                        opts.setdefault("identity", []).append(val)
                    elif c == "p":
                        opts["port"] = int(val)
                    elif c == "L":
                        forwards.append(val)
                    elif c == "l":
                        opts["user"] = val
                    elif c == "J":
                        opts["jump"] = val
                    elif c == "o":
                        k, _, v = val.partition("=")
                        if k.lower() == "port":
                            opts["port"] = int(v)
                        elif k.lower() == "user":
                            opts["user"] = v
                    break
                if c == "N":
                    no_cmd = True
                elif c == "f":
                    background = True
                elif c == "t":
                    tty_force = True
                elif c == "G":
                    show_config = True
                elif c == "V":
                    ctx.e("OpenSSH_9.6p1 Ubuntu-3ubuntu13.5, OpenSSL 3.0.13 30 Jan 2024\n")
                    return 0
                elif c in "TqvCAaXxYnkgK46":
                    pass
                else:
                    ctx.e(f"ssh: illegal option -- {c}\n")
                    return 255
                j += 1
            i += 1
            continue
        if dest is None:
            dest = x
            i += 1
            break
        i += 1
    command = a[i:]
    if dest is None:
        ctx.e("usage: ssh [-46AaCfGgKkMNnqsTtVvXxYy] [-B bind_interface] [-b bind_address]\n           [-c cipher_spec] [-D [bind_address:]port] [-E log_file]\n"
              "           [-e escape_char] [-F configfile] [-I pkcs11] [-i identity_file]\n           [-J destination] [-L address] [-l login_name] [-m mac_spec]\n"
              "           [-O ctl_cmd] [-o option] [-P tag] [-p port] [-R address]\n           [-S ctl_path] [-W host:port] [-w local_tun[:remote_tun]]\n"
              "           destination [command [argument ...]]\n")
        return 255
    if show_config:
        user, alias, port = parse_dest(dest)
        cfg = ssh_config(ctx, alias)
        ctx.w(f"host {alias}\nuser {user or opts.get('user') or cfg.get('user') or ctx.host.user_name(ctx.uid)}\n"
              f"hostname {cfg.get('hostname', alias)}\nport {opts.get('port') or cfg.get('port', 22)}\n")
        for f in cfg.get("identityfile", []):
            ctx.w(f"identityfile {f}\n")
        for f in cfg.get("localforward", []):
            lp, rest = f.split(None, 1) if " " in f else (f, "")
            ctx.w(f"localforward {lp} [{rest.split(':')[0]}]:{rest.split(':')[-1]}\n" if rest else f"localforward {f}\n")
        if cfg.get("proxyjump"):
            ctx.w(f"proxyjump {cfg['proxyjump']}\n")
        return 0
    try:
        host, user, alias = connect(ctx, dest, opts)
    except SshError as e:
        ctx.e(e.msg + "\n")
        return e.status
    cfg = ssh_config(ctx, alias)
    all_forwards = forwards + [f.replace(" ", ":", 1) for f in cfg.get("localforward", [])]
    for f in all_forwards:
        parts = f.split(":")
        if len(parts) == 3:
            lport, thost, tport = int(parts[0]), parts[1], int(parts[2])
        elif len(parts) == 4:
            lport, thost, tport = int(parts[1]), parts[2], int(parts[3])
        else:
            ctx.e(f"Bad local forwarding specification '{f}'\n")
            return 255
        if lport in ctx.host.services and ctx.host.services[lport].proc is not None and ctx.host.services[lport].proc.state != "Z":
            ctx.e(f"bind [127.0.0.1]:{lport}: Address already in use\nchannel_setup_fwd_listener_tcpip: cannot listen to port: {lport}\n"
                  f"Could not request local forwarding.\n")
            continue
        from ..world import Service
        ctx.host.services[lport] = Service(lport, ctx.proc, None, f"forward:{host.name}:{thost}:{tport}")
        ctx.proc.ports.append(lport)
    if no_cmd or (background and not command):
        if background:
            # ssh -f: go to the background (the parent returns, the tunnel stays up)
            from ..world import forever
            k = ctx.k
            bg = k.spawn(ctx.host, list(ctx.argv), None, ctx.proc, fds={}, cwd=ctx.cwd)
            bg.ppid = 1
            bg.ports = list(ctx.proc.ports)
            bg.daemon = True
            for port in bg.ports:
                ctx.host.services[port].proc = bg
            ctx.proc.ports = []
            k.start(bg, forever())
            return 0
        # -N in the foreground: hold the tunnel until Ctrl-C
        yield from ctx.flush()
        tty = ctx.world.tty_of(ctx.proc)
        fg = ctx.foreground()
        if fg:
            yield ("sleep", ctx.now + ctx.world.ctrl_c_after * NS)
            ctx.world.tty.write("echo", b"^C\n")
            return 130
        while True:
            yield ("forever",)
    from ..shell import Shell
    from ..world import init_shell, login_env
    fds = dict(ctx.proc.fds)
    if command:
        cmdline = " ".join(command)
        p = remote_proc(ctx, host, user, ["bash", "-c", cmdline], fds)
        env = login_env(host, user)
        env["SSH_CONNECTION"] = f"{ctx.host.ip} 51234 {host.ip} 22"
        env["SSH_CLIENT"] = f"{ctx.host.ip} 51234 22"
        from ..exec import exec_program
        ctx.k.start(p, exec_program(p, "/bin/bash", ["bash", "-c", cmdline], env, None))
        yield from ctx.wait([p])
        if tty_force:
            ctx.e(f"Connection to {alias} closed.\n")
        return p.status
    # interactive login: the remote shell reads the lines typed after `ssh host`
    yield from ctx.flush()
    rtty_name = "pts/1"
    p = remote_proc(ctx, host, user, ["-bash"], fds)
    sh = Shell(ctx.world, p, interactive=True)
    sh.arg0 = "bash"
    env = login_env(host, user)
    env["SSH_CONNECTION"] = f"{ctx.host.ip} 51234 {host.ip} 22"
    env["SSH_TTY"] = f"/dev/{rtty_name}"
    init_shell(sh, env, interactive=True)

    def session(shell):
        motd = host.motd or (f"Welcome to Ubuntu 24.04.1 LTS (GNU/Linux {host.kernel_release} x86_64)\n\n"
                             " * Documentation:  https://help.ubuntu.com\n * Management:     https://landscape.canonical.com\n"
                             " * Support:        https://ubuntu.com/pro\n\n")
        last = strftime("%a %b %e %H:%M:%S %Y", local_dt(ctx.now - 3600 * 26 * NS, None))
        yield from shell.out(motd + f"Last login: {last} from {ctx.host.ip}\n")
        rc = shell.get("HOME", "") + "/.bashrc"
        if host.fs.exists(rc):
            yield from shell.eval_string(f". {rc}")
        st = yield from shell.repl(idle=False)
        if not getattr(shell, "_exited", False):
            yield from shell.err("logout\n")
        return st
    ctx.k.start(p, session(sh))
    p.pgid = ctx.proc.pgid
    yield from ctx.wait([p])
    ctx.e(f"Connection to {alias} closed.\n")
    return p.status


@util("ssh-keygen")
def u_ssh_keygen(ctx):
    a = ctx.argv[1:]
    kind = "ed25519"
    comment = None
    path = None
    passphrase = None
    bits = None
    quiet = False
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x == "-t" and nxt:
            kind = nxt
            i += 2
            continue
        if x == "-C" and nxt is not None:
            comment = nxt
            i += 2
            continue
        if x == "-f" and nxt:
            path = nxt
            i += 2
            continue
        if x == "-N" and nxt is not None:
            passphrase = nxt
            i += 2
            continue
        if x == "-b" and nxt:
            bits = int(nxt)
            i += 2
            continue
        if x == "-q":
            quiet = True
        elif x == "-l" and nxt == "-f" and i + 2 < len(a):
            try:
                text = bytes(ctx.lookup(a[i + 2]).data).decode()
            except FSError as e:
                ctx.e(f"{a[i + 2]}: {e.msg}\n")
                return 255
            p = parse_pub(text)
            if not p:
                ctx.e(f"{a[i + 2]} is not a public key file.\n")
                return 255
            label = "ED25519" if p[0] == "ssh-ed25519" else "RSA"
            ctx.w(f"{256 if label == 'ED25519' else 3072} {fingerprint(p[1])} {p[2] or 'no comment'} ({label})\n")
            return 0
        i += 1
    if kind not in ("ed25519", "rsa", "ecdsa"):
        ctx.e(f"unknown key type {kind}\n")
        return 255
    home = ctx.env.get("HOME", ctx.host.home_of(ctx.uid))
    default = f"{home}/.ssh/id_{kind}"
    from ..fileio import readline
    if not quiet:
        ctx.w(f"Generating public/private {kind} key pair.\n")
    if path is None:
        ctx.w(f"Enter file in which to save the key ({default}): ")
        yield from ctx.flush()
        ans = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").strip()
        if not ans:
            ctx.w("\n") if False else None
        path = ans or default
    path = path.replace("~", home, 1) if path.startswith("~") else path
    d = path.rsplit("/", 1)[0] if "/" in path else ctx.cwd
    if not ctx.fs.exists(d, ctx.cwd):
        node = ctx.fs.mkdir(d, ctx.cwd, 0o700, ctx.uid, ctx.proc.gid, parents=True)
        node.mode = 0o700
        ctx.w(f"Created directory '{d}'.\n")
    if ctx.fs.exists(path, ctx.cwd):
        ctx.w(f"{path} already exists.\nOverwrite (y/n)? ")
        yield from ctx.flush()
        ans = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").strip()
        if not ans.lower().startswith("y"):
            return 1
    if passphrase is None:
        ctx.w(f'Enter passphrase for "{path}" (empty for no passphrase): ')
        yield from ctx.flush()
        p1 = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").rstrip("\n")
        ctx.w("\nEnter same passphrase again: ")
        yield from ctx.flush()
        p2 = (yield from readline(ctx.k, ctx.proc.fds.get(0))).decode(errors="replace").rstrip("\n")
        ctx.w("\n")
        if p1 != p2:
            ctx.w("Passphrases do not match.  Try again.\n")
            return 1
    if comment is None:
        comment = f"{ctx.host.user_name(ctx.uid)}@{ctx.host.name}"
    seed = f"{ctx.world.seed}:{path}:{comment}:{ctx.now}"
    raw = key_material(seed)
    blob = pub_blob("ed25519" if kind != "rsa" else "rsa", raw)
    pub_line = f"{'ssh-ed25519' if kind != 'rsa' else 'ssh-rsa'} {base64.b64encode(blob).decode()} {comment}\n"
    body = base64.b64encode(b"openssh-key-v1\x00" + raw * 8).decode()
    priv = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "\n".join(body[k:k + 70] for k in range(0, len(body), 70)) + \
           f"\n-----END OPENSSH PRIVATE KEY-----\n"
    priv_node = ctx.fs.write_file(path, priv + f"# pubkey:{base64.b64encode(blob).decode()}\n" if False else priv, ctx.cwd, 0o600, ctx.uid, ctx.proc.gid)
    priv_node.mode = 0o600
    priv_node.xattr = {"pub": base64.b64encode(blob).decode()}
    pub_node = ctx.fs.write_file(path + ".pub", pub_line, ctx.cwd, 0o644, ctx.uid, ctx.proc.gid)
    pub_node.mode = 0o644
    if not quiet:
        title = "ED25519 256" if kind != "rsa" else f"RSA {bits or 3072}"
        ctx.w(f"Your identification has been saved in {path}\nYour public key has been saved in {path}.pub\n"
              f"The key fingerprint is:\n{fingerprint(blob)} {comment}\nThe key's randomart image is:\n{randomart(blob, title)}\n")
    return 0


@util("ssh-copy-id")
def u_ssh_copy_id(ctx):
    a = ctx.argv[1:]
    idf = None
    port = None
    dest = None
    i = 0
    while i < len(a):
        if a[i] == "-i" and i + 1 < len(a):
            idf = a[i + 1]
            i += 2
            continue
        if a[i] == "-p" and i + 1 < len(a):
            port = int(a[i + 1])
            i += 2
            continue
        if a[i] in ("-f", "-n"):
            i += 1
            continue
        dest = a[i]
        i += 1
    home = ctx.env.get("HOME", ctx.host.home_of(ctx.uid))
    if dest is None:
        ctx.e("Usage: /usr/bin/ssh-copy-id [-h|-?|-f|-n|-s|-x] [-i [identity_file]] [-p port] [-F alternative ssh_config file] [-t target_path] [[-o <ssh -o options>] ...] [user@]hostname\n")
        return 1
    pubfile = (idf if idf and idf.endswith(".pub") else (idf + ".pub" if idf else None))
    if pubfile is None:
        for cand in (f"{home}/.ssh/id_ed25519.pub", f"{home}/.ssh/id_rsa.pub"):
            if ctx.fs.exists(cand):
                pubfile = cand
                break
    if pubfile is None or not ctx.fs.exists(pubfile, ctx.cwd):
        ctx.e("/usr/bin/ssh-copy-id: ERROR: No identities found\n")
        return 1
    pub_text = bytes(ctx.fs.lookup(pubfile, ctx.cwd).data).decode()
    user, alias, dport = parse_dest(dest)
    cfg = ssh_config(ctx, alias)
    hostname = cfg.get("hostname", alias)
    host = ctx.world.resolve(hostname)
    user = user or cfg.get("user") or ctx.host.user_name(ctx.uid)
    ctx.e(f"/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: \"{pubfile}\"\n")
    if host is None:
        ctx.e(f"ssh: Could not resolve hostname {hostname}: Name or service not known\n")
        return 1
    uid = host.uid_of(user)
    if uid is None:
        ctx.e(f"{user}@{hostname}: Permission denied (publickey,password).\n")
        return 1
    rhome = host.users[uid].home
    auth_path = f"{rhome}/.ssh/authorized_keys"
    try:
        existing = bytes(host.fs.lookup(auth_path).data).decode()
    except FSError:
        existing = ""
    ctx.e("/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed\n")
    if pub_text.split()[1] in existing:
        ctx.e("\n/usr/bin/ssh-copy-id: WARNING: All keys were skipped because they already exist on the remote system.\n"
              "\t\t(if you think this is a mistake, you may want to use -f option)\n\n")
        return 0
    ctx.e("/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys\n")
    ctx.e(f"{user}@{hostname}'s password: \n")
    d = host.fs.mkdir(f"{rhome}/.ssh", "/", 0o700, uid, host.users[uid].gid, parents=True)
    d.mode = 0o700
    node = host.fs.write_file(auth_path, existing + pub_text, "/", 0o600, uid, host.users[uid].gid)
    node.mode = 0o600
    shown = f"{user}@{alias}" if "@" in dest else alias
    ctx.w(f"\nNumber of key(s) added: 1\n\nNow try logging into the machine, with:   \"ssh '{shown}'\"\n"
          "and check to make sure that only the key(s) you wanted were added.\n\n")
    return 0
    yield


@util("ssh-agent")
def u_ssh_agent(ctx):
    a = ctx.argv[1:]
    from ..world import forever
    k = ctx.k
    agent = k.spawn(ctx.host, ["ssh-agent"] + ([x for x in a]), None, ctx.proc, fds={}, cwd="/")
    agent.ppid = 1
    agent.daemon = True
    k.start(agent, forever())
    rnd = "".join(ctx.world.rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(12))
    sock = f"/tmp/ssh-{rnd}/agent.{agent.pid - 1}"
    ctx.fs.mkdir(sock.rsplit("/", 1)[0], "/", 0o700, ctx.uid, ctx.proc.gid, parents=True)
    agents = ctx.world.__dict__.setdefault("agents", {})
    agents[sock] = {"pid": agent.pid, "keys": []}
    if "-c" in a:
        ctx.w(f"setenv SSH_AUTH_SOCK {sock};\nsetenv SSH_AGENT_PID {agent.pid};\necho Agent pid {agent.pid};\n")
    else:
        ctx.w(f"SSH_AUTH_SOCK={sock}; export SSH_AUTH_SOCK;\nSSH_AGENT_PID={agent.pid}; export SSH_AGENT_PID;\necho Agent pid {agent.pid};\n")
    return 0
    yield


@util("ssh-add")
def u_ssh_add(ctx):
    a = ctx.argv[1:]
    sock = ctx.env.get("SSH_AUTH_SOCK")
    agents = ctx.world.__dict__.get("agents", {})
    agent = agents.get(sock)
    if agent is None or ctx.host.procs.get(agent["pid"]) is None:
        ctx.e("Could not open a connection to your authentication agent.\n")
        return 2
    if "-l" in a or "-L" in a:
        if not agent["keys"]:
            ctx.w("The agent has no identities.\n")
            return 1
        for kind, blob, comment in agent["keys"]:
            if "-L" in a:
                ctx.w(f"{kind} {base64.b64encode(blob).decode()} {comment}\n")
            else:
                ctx.w(f"256 {fingerprint(blob)} {comment} (ED25519)\n")
        return 0
    if "-D" in a:
        agent["keys"] = []
        ctx.e("All identities removed.\n")
        return 0
    files = [x for x in a if not x.startswith("-")]
    home = ctx.env.get("HOME", ctx.host.home_of(ctx.uid))
    if not files:
        files = [f for f in (f"{home}/.ssh/id_rsa", f"{home}/.ssh/id_ed25519", f"{home}/.ssh/id_ecdsa") if ctx.fs.exists(f)]
    status = 0
    for f in files:
        f2 = f.replace("~", home, 1) if f.startswith("~") else f
        try:
            ctx.lookup(f2)
        except FSError as e:
            ctx.e(f"{f}: {e.msg}\n")
            status = 1
            continue
        try:
            p = parse_pub(bytes(ctx.lookup(f2 + ".pub").data).decode())
        except FSError:
            p = None
        if p is None:
            ctx.e(f"Error loading key \"{f}\": invalid format\n")
            status = 1
            continue
        agent["keys"].append(p)
        ctx.e(f"Identity added: {f} ({p[2]})\n")
    return status
    yield


# ---------------------------------------------------------------------------- scp / rsync

def split_remote(spec):
    """'host:path' / 'user@host:path' -> (dest, path) or (None, spec) for local paths."""
    m = re.match(r"^((?:[^@:/]+@)?[^:/]+):(.*)$", spec)
    if m and not spec.startswith("/") and not spec.startswith("./"):
        return m.group(1), m.group(2)
    return None, spec


def resolve_side(ctx, spec, opts):
    dest, path = split_remote(spec)
    if dest is None:
        return ctx.host, ctx.uid, ctx.proc.gid, ctx.cwd, path
    host, user, alias = connect(ctx, dest, opts)
    uid = host.uid_of(user)
    home = host.users[uid].home
    if path == "":
        path = home
    elif path.startswith("~/"):
        path = home + path[1:]
    elif path == "~":
        path = home
    return host, uid, host.users[uid].gid, home, path


@util("scp")
def u_scp(ctx):
    a = ctx.argv[1:]
    recursive = quiet = preserve = False
    opts = {}
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ("-P",) and i + 1 < len(a):
            opts["port"] = int(a[i + 1])
            i += 2
            continue
        if x == "-i" and i + 1 < len(a):
            opts.setdefault("identity", []).append(a[i + 1])
            i += 2
            continue
        if x == "-J" and i + 1 < len(a):
            opts["jump"] = a[i + 1]
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            recursive = recursive or "r" in x
            quiet = quiet or "q" in x
            preserve = preserve or "p" in x
            i += 1
            continue
        ops.append(x)
        i += 1
    if len(ops) < 2:
        ctx.e("usage: scp [-346ABCOpqRrsTv] [-c cipher] [-D sftp_server_path] [-F ssh_config]\n           [-i identity_file] [-J destination] [-l limit] [-o ssh_option]\n           [-P port] [-S program] [-X sftp_option] source ... target\n")
        return 1
    try:
        dst_host, duid, dgid, dcwd, dpath = resolve_side(ctx, ops[-1], opts)
    except SshError as e:
        ctx.e(e.msg + "\n")
        return 255
    status = 0
    for src in ops[:-1]:
        try:
            s_host, suid, sgid, scwd, spath = resolve_side(ctx, src, opts)
        except SshError as e:
            ctx.e(e.msg + "\n")
            return 255
        try:
            node = s_host.fs.lookup(spath, scwd)
        except FSError as e:
            ctx.e(f"scp: {spath}: {e.msg}\n")
            status = 1
            continue
        if node.type == "d" and not recursive:
            ctx.e(f"scp: {spath}: not a regular file\n")
            status = 1
            continue
        base = spath.rstrip("/").rsplit("/", 1)[-1]
        full_d = dpath if dpath.startswith("/") else dcwd.rstrip("/") + "/" + dpath
        try:
            dn = dst_host.fs.lookup(full_d)
            target = full_d.rstrip("/") + "/" + base if dn.type == "d" else full_d
        except FSError:
            parent = full_d.rstrip("/").rsplit("/", 1)[0] or "/"
            if not dst_host.fs.exists(parent):
                ctx.e(f"scp: dest open \"{dpath}\": No such file or directory\nscp: failed to upload file {spath} to {dpath}\n")
                status = 1
                continue
            target = full_d
        s_full = spath if spath.startswith("/") else scwd.rstrip("/") + "/" + spath
        s_host.fs.copy_tree(node, dst_host.fs, target, duid, dgid, keep=preserve)
        if not quiet and ctx.stdout_is_tty():
            for path, n in s_host.fs.walk(node, base):
                if n.type == "f":
                    size = len(n.data)
                    name = path.rsplit("/", 1)[-1]
                    ctx.w(f"{name:<47}100% {_human_bytes(size):>7} {_human_bytes(size * 40 + 1000) + '/s':>9}   00:00    \n")
    return status
    yield


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return str(n)


def _commas(n):
    return f"{n:,}"


@util("rsync")
def u_rsync(ctx):
    a = ctx.argv[1:]
    o = {"r": False, "v": False, "n": False, "delete": False, "times": False, "perms": False, "progress": False,
         "human": False, "update": False, "stats": False, "checksum": False, "links": False}
    excludes = []
    includes = []
    opts = {}
    ops = []
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x.startswith("--"):
            k, _, v = x.partition("=")
            if k == "--exclude":
                excludes.append(v if _ else nxt)
                if not _:
                    i += 1
            elif k == "--include":
                includes.append(v if _ else nxt)
                if not _:
                    i += 1
            elif k in ("--delete", "--delete-after", "--delete-before"):
                o["delete"] = True
            elif k == "--dry-run":
                o["n"] = True
            elif k == "--progress":
                o["progress"] = True
            elif k == "--archive":
                o.update(r=True, times=True, perms=True, links=True)
            elif k == "--recursive":
                o["r"] = True
            elif k == "--verbose":
                o["v"] = True
            elif k == "--stats":
                o["stats"] = True
            elif k == "--human-readable":
                o["human"] = True
            elif k in ("--partial", "--compress", "--times", "--checksum"):
                if k == "--times":
                    o["times"] = True
            elif k == "--version":
                ctx.w("rsync  version 3.2.7  protocol version 31\n")
                return 0
            i += 1
            continue
        if x == "-e" and nxt:
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            for c in x[1:]:
                if c == "a":
                    o.update(r=True, times=True, perms=True, links=True)
                elif c == "r":
                    o["r"] = True
                elif c == "v":
                    o["v"] = True
                elif c == "n":
                    o["n"] = True
                elif c == "P":
                    o["progress"] = True
                elif c == "h":
                    o["human"] = True
                elif c == "u":
                    o["update"] = True
                elif c == "t":
                    o["times"] = True
                elif c == "c":
                    o["checksum"] = True
                elif c in "zplgoDHAX":
                    pass
                else:
                    ctx.e(f"rsync: -{c}: unknown option\nrsync error: syntax or usage error (code 1) at main.c(1795) [client=3.2.7]\n")
                    return 1
            i += 1
            continue
        ops.append(x)
        i += 1
    if len(ops) < 2:
        ctx.e("rsync: need at least a source and a destination\n" if ops else
              "rsync  version 3.2.7  protocol version 31\nUsage: rsync [OPTION]... SRC [SRC]... DEST\n")
        return 1
    try:
        dst_host, duid, dgid, dcwd, dpath = resolve_side(ctx, ops[-1], opts)
    except SshError as e:
        ctx.e(e.msg + "\nrsync: connection unexpectedly closed (0 bytes received so far) [sender]\n"
              "rsync error: unexplained error (code 255) at io.c(232) [sender=3.2.7]\n")
        return 255
    lines = []
    total_size = 0
    sent = 0
    nfiles = 0
    status = 0
    for src in ops[:-1]:
        try:
            s_host, suid, sgid, scwd, spath = resolve_side(ctx, src, opts)
        except SshError as e:
            ctx.e(e.msg + "\n")
            return 255
        try:
            node = s_host.fs.lookup(spath, scwd)
        except FSError as e:
            ctx.e(f"rsync: [sender] link_stat \"{s_host.fs and (spath if spath.startswith('/') else scwd + '/' + spath)}\" failed: {e.msg} (2)\n")
            status = 23
            continue
        contents = spath.endswith("/") or spath in (".", "")
        if node.type == "d" and not o["r"]:
            lines.append(f"skipping directory {spath.rstrip('/') or '.'}")
            continue
        full_d = dpath if dpath.startswith("/") else dcwd.rstrip("/") + "/" + dpath
        dest_root = full_d.rstrip("/") if contents or node.type != "d" else full_d.rstrip("/") + "/" + spath.rstrip("/").rsplit("/", 1)[-1]
        if node.type != "d":
            try:
                dn = dst_host.fs.lookup(full_d)
                if dn.type == "d":
                    dest_root = full_d.rstrip("/") + "/" + spath.rsplit("/", 1)[-1]
            except FSError:
                pass
        prefix = "" if contents else (spath.rstrip("/").rsplit("/", 1)[-1] + ("/" if node.type == "d" else ""))
        entries = []
        if node.type == "d":
            if contents:
                entries.append(("./", node, dest_root))
            else:
                entries.append((prefix, node, dest_root))

            def walk(n, rel, droot):
                for name in s_host.fs.listdir(n):
                    child = n.entries[name]
                    r = rel + name
                    if any(fnmatch(e.rstrip("/"), name) or fnmatch(e.rstrip("/"), r.rstrip("/")) for e in excludes) and \
                            not any(fnmatch(inc, name) for inc in includes):
                        continue
                    if child.type == "d":
                        entries.append((prefix + r + "/", child, droot + "/" + r))
                        walk(child, r + "/", droot)
                    else:
                        entries.append((prefix + r, child, droot + "/" + r))
            walk(node, "", dest_root)
        else:
            entries.append((spath.rsplit("/", 1)[-1], node, dest_root))
        # deletions (dest entries missing from the source)
        if o["delete"] and node.type == "d":
            try:
                dnode = dst_host.fs.lookup(dest_root)
                want = {e[2] for e in entries}
                gone = []
                for path, n in dst_host.fs.walk(dnode, dest_root):
                    rel = path[len(dest_root):].lstrip("/")
                    if path == dest_root or path in want:
                        continue
                    if any(fnmatch(e.rstrip("/"), rel.rsplit("/", 1)[-1]) for e in excludes):
                        continue
                    if not any(path.startswith(g + "/") for g in gone):
                        gone.append(path)
                for path in sorted(gone, key=lambda p: -p.count("/")):
                    rel = path[len(dest_root):].lstrip("/")
                    n = dst_host.fs.lookup(path, follow=False)
                    lines.append(f"deleting {prefix if not contents else ''}{rel}{'/' if n.type == 'd' else ''}")
                    if not o["n"]:
                        if n.type == "d":
                            for p2, n2 in sorted(dst_host.fs.walk(n, path), key=lambda t: -t[0].count("/")):
                                if n2.type == "d":
                                    dst_host.fs.rmdir(p2)
                                else:
                                    dst_host.fs.unlink(p2)
                        else:
                            dst_host.fs.unlink(path)
            except FSError:
                pass
        for rel, n, dst in entries:
            if n.type == "f":
                total_size += len(n.data)
            try:
                existing = dst_host.fs.lookup(dst, follow=False)
            except FSError:
                existing = None
            if n.type == "d":
                if existing is None:
                    lines.append(rel)
                    if not o["n"]:
                        d = dst_host.fs.mkdir(dst, "/", n.mode, duid, dgid, parents=True)
                        d.mode = n.mode if o["perms"] else d.mode
                        if o["times"]:
                            d.mtime = n.mtime
                elif rel == "./" or o["times"] and existing.mtime != n.mtime:
                    lines.append(rel)
                    if not o["n"] and o["times"]:
                        existing.mtime = n.mtime
                continue
            same = existing is not None and existing.type == "f" and (
                (bytes(existing.data) == bytes(n.data)) if o["checksum"] else
                (len(existing.data) == len(n.data) and (existing.mtime == n.mtime if o["times"] else bytes(existing.data) == bytes(n.data))))
            if o["update"] and existing is not None and existing.mtime > n.mtime:
                continue
            if same:
                continue
            lines.append(rel)
            nfiles += 1
            sent += len(n.data)
            if o["progress"]:
                lines.append(f"{len(n.data):>15,} 100%    0.00kB/s    0:00:00 (xfr#{nfiles}, to-chk=0/{len(entries)})")
            if not o["n"]:
                parent = dst.rsplit("/", 1)[0] or "/"
                if not dst_host.fs.exists(parent):
                    dst_host.fs.mkdir(parent, "/", 0o755, duid, dgid, parents=True)
                new = dst_host.fs.write_file(dst, bytes(n.data), "/", n.mode, duid, dgid)
                new.mode = n.mode if o["perms"] else new.mode
                new.mtime = n.mtime if o["times"] else ctx.now
    if o["v"] or o["n"]:
        out = []
        if o["v"]:
            out.append("sending incremental file list")
            out += lines
            sent_b = sent + 70 * max(len(lines), 1) + 40
            recv_b = 19 * nfiles + 38
            rate = (sent_b + recv_b) / 0.5
            speed = total_size / (sent_b + recv_b) if sent_b + recv_b else 0
            out.append("")
            out.append(f"sent {_commas(sent_b)} bytes  received {_commas(recv_b)} bytes  {rate:,.2f} bytes/sec")
            out.append(f"total size is {_commas(total_size)}  speedup is {speed:.2f}" + (" (DRY RUN)" if o["n"] else ""))
        ctx.w("".join(x + "\n" for x in out))
    return status
    yield


# ---------------------------------------------------------------------------- HTTP

class HttpResult:
    def __init__(self, status=200, headers=None, body=b"", reason=None):
        self.status, self.headers, self.body = status, headers or {}, body
        self.reason = reason or {200: "OK", 201: "Created", 204: "No Content", 206: "Partial Content", 301: "Moved Permanently",
                                 302: "Found", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
                                 405: "Method Not Allowed", 422: "Unprocessable Entity", 500: "Internal Server Error"}.get(status, "")


class HttpError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


EXAMPLE_HTML = b"""<!doctype html>
<html>
<head>
    <title>Example Domain</title>
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents.</p>
</div>
</body>
</html>
"""


def default_web(url_host, path, method, headers, body):
    if url_host in ("example.com", "www.example.com"):
        if path in ("/", ""):
            return HttpResult(200, {"Content-Type": "text/html; charset=UTF-8"}, EXAMPLE_HTML)
        if path == "/download":
            return HttpResult(302, {"Location": "https://example.com/files/data.zip"}, b"")
        return HttpResult(404, {"Content-Type": "text/html"}, b"<h1>404 Not Found</h1>\n")
    if url_host == "api.github.com":
        m = re.match(r"/repos/([^/]+)/([^/]+)/?$", path)
        if m:
            stars = {"pytorch/pytorch": 89214, "numpy/numpy": 29810, "pandas-dev/pandas": 45102, "huggingface/transformers": 142877,
                     "scikit-learn/scikit-learn": 61733}.get(f"{m.group(1)}/{m.group(2)}", 1234)
            data = {"id": abs(hash(path)) % 10**9, "name": m.group(2), "full_name": f"{m.group(1)}/{m.group(2)}",
                    "private": False, "html_url": f"https://github.com/{m.group(1)}/{m.group(2)}",
                    "description": "", "fork": False, "stargazers_count": stars, "watchers_count": stars,
                    "language": "Python", "forks_count": stars // 4, "open_issues_count": stars // 15, "default_branch": "main"}
            return HttpResult(200, {"Content-Type": "application/json; charset=utf-8"}, (json.dumps(data, indent=2) + "\n").encode())
        return HttpResult(404, {"Content-Type": "application/json"}, b'{\n  "message": "Not Found",\n  "documentation_url": "https://docs.github.com/rest",\n  "status": "404"\n}\n')
    if url_host == "httpbin.org":
        if path.startswith("/status/"):
            return HttpResult(int(path.split("/")[2]), {}, b"")
        if path in ("/get", "/post", "/put", "/delete", "/anything"):
            data = {"args": {}, "headers": {k: v for k, v in headers.items()}, "url": f"https://httpbin.org{path}"}
            if body:
                data["data"] = body.decode("utf-8", "replace")
                try:
                    data["json"] = json.loads(body)
                except ValueError:
                    data["json"] = None
            return HttpResult(200, {"Content-Type": "application/json"}, (json.dumps(data, indent=2) + "\n").encode())
        if path == "/json":
            return HttpResult(200, {"Content-Type": "application/json"}, b'{"slideshow": {"author": "Yours Truly", "title": "Sample Slide Show"}}\n')
    return HttpResult(404, {"Content-Type": "text/html"}, b"<html><body><h1>404 Not Found</h1></body></html>\n")


def http_request(ctx, url, method="GET", headers=None, body=b""):
    """-> HttpResult. Raises HttpError(curl code, message)."""
    headers = headers or {}
    m = re.match(r"(https?)://([^/:]+)(?::(\d+))?(/[^?#]*)?(\?[^#]*)?", url)
    if not m:
        raise HttpError(3, f"URL rejected: Malformed input to a URL function")
    scheme, h, port, path, query = m.group(1), m.group(2), m.group(3), m.group(4) or "/", m.group(5) or ""
    port = int(port) if port else (443 if scheme == "https" else 80)
    world = ctx.world
    full = f"{scheme}://{h}{path}{query}"
    fixed = world.web.get(full) or world.web.get(f"{scheme}://{h}{path}") or world.web.get(f"{h}{path}")
    if fixed is not None:
        res = fixed(method, path, headers, body) if callable(fixed) else fixed
        return res if isinstance(res, HttpResult) else HttpResult(200, {}, res if isinstance(res, bytes) else str(res).encode())
    target = None
    if h in ("localhost", "127.0.0.1", "0.0.0.0", ctx.host.name):
        target = ctx.host
    else:
        target = world.resolve(h)
    if target is not None:
        svc = target.services.get(port)
        if svc is None or (svc.proc is not None and svc.proc.state == "Z"):
            raise HttpError(7, f"Failed to connect to {h} port {port} after 0 ms: Couldn't connect to server")
        if svc.name.startswith("forward:"):
            _, rhost, thost, tport = svc.name.split(":")
            remote = world.hosts[rhost]
            rsvc = remote.services.get(int(tport)) if thost in ("localhost", "127.0.0.1", remote.name) else None
            if rsvc is None or rsvc.handler is None:
                raise HttpError(52, "Empty reply from server")
            return rsvc.handler(method, path + query, headers, body)
        if svc.handler is None:
            raise HttpError(52, "Empty reply from server")
        return svc.handler(method, path + query, headers, body)
    if h not in INTERNET:
        raise HttpError(6, f"Could not resolve host: {h}")
    return default_web(h, path, method, headers, body)


def mime_of(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {"html": "text/html", "htm": "text/html", "txt": "text/plain", "csv": "text/csv", "json": "application/json",
            "png": "image/png", "jpg": "image/jpeg", "py": "text/x-python", "md": "text/markdown", "js": "text/javascript",
            "css": "text/css", "zip": "application/zip", "gz": "application/gzip", "tar": "application/x-tar"}.get(ext, "application/octet-stream")


def static_handler(host, root):
    """python -m http.server: serve a directory of the host's filesystem."""
    fs = host.fs

    def handle(method, path, headers, body):
        p = path.split("?")[0]
        from urllib.parse import unquote
        p = unquote(p)
        full = root.rstrip("/") + p
        try:
            node = fs.lookup(full)
        except FSError:
            return HttpResult(404, {"Content-Type": "text/html;charset=utf-8"},
                              b'<!DOCTYPE HTML>\n<html lang="en">\n    <head>\n        <meta charset="utf-8">\n        <title>Error response</title>\n'
                              b"    </head>\n    <body>\n        <h1>Error response</h1>\n        <p>Error code: 404</p>\n"
                              b"        <p>Message: File not found.</p>\n    </body>\n</html>\n")
        if node.type == "d":
            if "index.html" in node.entries:
                return HttpResult(200, {"Content-Type": "text/html"}, bytes(node.entries["index.html"].data))
            items = "".join(f'<li><a href="{n}{"/" if node.entries[n].type == "d" else ""}">{n}{"/" if node.entries[n].type == "d" else ""}</a></li>\n'
                            for n in fs.listdir(node))
            html = (f'<!DOCTYPE HTML>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>Directory listing for {p}</title>\n</head>\n'
                    f"<body>\n<h1>Directory listing for {p}</h1>\n<hr>\n<ul>\n{items}</ul>\n<hr>\n</body>\n</html>\n")
            return HttpResult(200, {"Content-Type": "text/html; charset=utf-8"}, html.encode())
        return HttpResult(200, {"Content-Type": mime_of(p)}, bytes(node.data))
    return handle


@util("curl")
def u_curl(ctx):
    a = ctx.argv[1:]
    o = {"silent": False, "show_err": False, "follow": False, "include": False, "head": False, "fail": False,
         "verbose": False, "remote_name": False, "write_out": None, "out": None, "method": None, "resume": False}
    headers = {}
    data = []
    urls = []
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        long = {"--silent": "s", "--show-error": "S", "--location": "L", "--include": "i", "--head": "I", "--fail": "f",
                "--verbose": "v", "--remote-name": "O", "--insecure": "k", "--compressed": "", "--progress-bar": "#"}
        if x in long:
            flags = long[x]
        elif x in ("-o", "--output") and nxt is not None:
            o["out"] = nxt
            i += 2
            continue
        elif x in ("-X", "--request") and nxt:
            o["method"] = nxt
            i += 2
            continue
        elif x in ("-H", "--header") and nxt:
            k, _, v = nxt.partition(":")
            headers[k.strip()] = v.strip()
            i += 2
            continue
        elif x in ("-d", "--data", "--data-raw", "--data-binary", "--data-urlencode") and nxt is not None:
            if nxt.startswith("@") and x != "--data-raw":
                try:
                    nxt = (yield from ctx.text(nxt[1:])).replace("\n", "" if x != "--data-binary" else "\n")
                except FSError:
                    ctx.e(f"Warning: Couldn't read data from file \"{nxt[1:]}\", this makes an empty POST.\n")
                    nxt = ""
            data.append(nxt)
            i += 2
            continue
        elif x == "--json" and nxt is not None:
            data.append(nxt)
            headers.setdefault("Content-Type", "application/json")
            headers.setdefault("Accept", "application/json")
            i += 2
            continue
        elif x in ("-w", "--write-out") and nxt is not None:
            o["write_out"] = nxt
            i += 2
            continue
        elif x in ("-u", "--user", "-A", "--user-agent", "--max-time", "-m", "--retry", "--connect-timeout", "-e", "--referer") and nxt is not None:
            if x in ("-A", "--user-agent"):
                headers["User-Agent"] = nxt
            i += 2
            continue
        elif x == "-C" and nxt is not None:
            o["resume"] = nxt == "-"
            i += 2
            continue
        elif x.startswith("-") and len(x) > 1 and not x.startswith("--"):
            flags = x[1:]
        elif x.startswith("--"):
            ctx.e(f"curl: option {x}: is unknown\ncurl: try 'curl --help' or 'curl --manual' for more information\n")
            return 2
        else:
            urls.append(x)
            i += 1
            continue
        for c in flags:
            if c == "s":
                o["silent"] = True
            elif c == "S":
                o["show_err"] = True
            elif c == "L":
                o["follow"] = True
            elif c == "i":
                o["include"] = True
            elif c == "I":
                o["head"] = True
            elif c == "f":
                o["fail"] = True
            elif c == "v":
                o["verbose"] = True
            elif c == "O":
                o["remote_name"] = True
            elif c in "k#":
                pass
        i += 1
    if not urls:
        ctx.e("curl: no URL specified!\ncurl: try 'curl --help' or 'curl --manual' for more information\n")
        return 2
    status = 0
    for url in urls:
        if not re.match(r"https?://", url):
            url = "http://" + url
        method = o["method"] or ("HEAD" if o["head"] else ("POST" if data else "GET"))
        body = "&".join(data).encode()
        if data and "Content-Type" not in headers:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        try:
            res = http_request(ctx, url, method, dict(headers, **{"Host": re.sub(r"https?://([^/]+).*", r"\1", url),
                                                                   "User-Agent": headers.get("User-Agent", "curl/8.5.0"),
                                                                   "Accept": headers.get("Accept", "*/*")}), body)
            hops = 0
            while o["follow"] and res.status in (301, 302, 303, 307, 308) and "Location" in res.headers and hops < 10:
                url = res.headers["Location"]
                res = http_request(ctx, url, "GET" if res.status in (301, 302, 303) else method, headers, body)
                hops += 1
        except HttpError as e:
            if not o["silent"] or o["show_err"]:
                ctx.e(f"curl: ({e.code}) {e.msg}\n")
            status = e.code
            continue
        if o["fail"] and res.status >= 400:
            if not o["silent"] or o["show_err"]:
                ctx.e(f"curl: (22) The requested URL returned error: {res.status}\n")
            status = 22
            continue
        payload = b""
        if o["include"] or o["head"]:
            hdrs = {"Content-Length": str(len(res.body)), **res.headers}
            payload += f"HTTP/1.1 {res.status} {res.reason}\r\n".encode()
            for k, v in hdrs.items():
                payload += f"{k}: {v}\r\n".encode()
            payload += b"\r\n"
        if not o["head"]:
            payload += res.body
        target = o["out"]
        if o["remote_name"]:
            target = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "index.html"
        if target and target != "-":
            try:
                of = ctx.open(target, "w")
                of.inode.data[:] = payload
            except FSError as e:
                ctx.e(f"curl: (23) Failure writing output to destination\n" if e.code != "ENOENT" else
                      f"Warning: Failed to open the file {target}: No such file or directory\ncurl: (23) Failure writing output to destination\n")
                status = 23
                continue
            if not o["silent"]:
                n = len(payload)
                ctx.e("  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n"
                      "                                 Dload  Upload   Total   Spent    Left  Speed\n"
                      f"100 {_cm(n)}  100 {_cm(n)}    0     0  {_cm(n * 10)}      0 --:--:-- --:--:-- --:--:-- {_cm(n * 10 + 120)}\n")
        else:
            ctx.w(payload)
        if o["write_out"]:
            w = o["write_out"].replace("%{http_code}", str(res.status)).replace("%{size_download}", str(len(res.body))) \
                .replace("%{content_type}", res.headers.get("Content-Type", "")).replace("%{url_effective}", url) \
                .replace("%{time_total}", "0.012345").replace("\\n", "\n")
            ctx.w(w)
    return status


def _cm(n):
    """curl progress meter number (5 chars, k/M suffix)."""
    if n < 100000:
        return f"{n:5d}"
    if n < 1024 * 1000:
        return f"{n / 1024:4.0f}k"
    return f"{n / 1048576:4.1f}M" if n < 1048576 * 10 else f"{n / 1048576:4.0f}M"


@util("wget")
def u_wget(ctx):
    a = ctx.argv[1:]
    quiet = cont = False
    out = None
    prefix = None
    urls = []
    i = 0
    while i < len(a):
        x = a[i]
        nxt = a[i + 1] if i + 1 < len(a) else None
        if x in ("-O", "--output-document") and nxt is not None:
            out = nxt
            i += 2
            continue
        if x in ("-P", "--directory-prefix") and nxt is not None:
            prefix = nxt
            i += 2
            continue
        if x.startswith("-") and len(x) > 1:
            if "q" in x or x == "--quiet":
                quiet = True
            if "c" in x[1:2] or x == "--continue":
                cont = True
            i += 1
            continue
        urls.append(x)
        i += 1
    if not urls:
        ctx.e("wget: missing URL\nUsage: wget [OPTION]... [URL]...\n\nTry `wget --help' for more options.\n")
        return 1
    status = 0
    for url in urls:
        if not re.match(r"https?://", url):
            url = "http://" + url
        host = re.sub(r"https?://([^/:]+).*", r"\1", url)
        when = strftime("%Y-%m-%d %H:%M:%S", local_dt(ctx.now, ctx.tz))
        if not quiet:
            ctx.e(f"--{when}--  {url}\n")
        ip = INTERNET.get(host) or (ctx.world.resolve(host).ip if ctx.world.resolve(host) else None)
        if ip is None:
            if not quiet:
                ctx.e(f"Resolving {host} ({host})... failed: Name or service not known.\nwget: unable to resolve host address ‘{host}’\n")
            status = 4
            continue
        try:
            res = http_request(ctx, url)
            while res.status in (301, 302) and "Location" in res.headers:
                url = res.headers["Location"]
                res = http_request(ctx, url)
        except HttpError as e:
            if not quiet:
                ctx.e(f"Resolving {host} ({host})... {ip}\nConnecting to {host} ({host})|{ip}|:443... failed: Connection refused.\n")
            status = 4
            continue
        name = out or (url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "index.html")
        if prefix and not out:
            name = prefix.rstrip("/") + "/" + name
            ctx.fs.mkdir(prefix, ctx.cwd, 0o755, ctx.uid, ctx.proc.gid, parents=True)
        if not quiet:
            ctx.e(f"Resolving {host} ({host})... {ip}\nConnecting to {host} ({host})|{ip}|:443... connected.\n"
                  f"HTTP request sent, awaiting response... {res.status} {res.reason}\n")
        if res.status >= 400:
            if not quiet:
                ctx.e(f"{when} ERROR {res.status}: {res.reason}.\n\n")
            status = 8
            continue
        body = res.body
        existing = None
        if ctx.fs.exists(name, ctx.cwd) and not out:
            existing = bytes(ctx.fs.lookup(name, ctx.cwd).data)
            if not cont:
                k = 1
                while ctx.fs.exists(f"{name}.{k}", ctx.cwd):
                    k += 1
                name = f"{name}.{k}"
                existing = None
        n = len(body)
        if not quiet:
            ctype = res.headers.get("Content-Type", "application/octet-stream").split(";")[0]
            if cont and existing is not None and len(existing) < n:
                ctx.e(f"Length: {n} ({_wh(n)}), {n - len(existing)} ({_wh(n - len(existing))}) remaining [{ctype}]\n")
            elif cont and existing is not None and len(existing) >= n:
                ctx.e(f"\n    The file is already fully retrieved; nothing to do.\n\n")
                continue
            else:
                ctx.e(f"Length: {n} ({_wh(n)}) [{ctype}]\n")
            ctx.e(f"Saving to: ‘{name}’\n\n{name.rsplit('/', 1)[-1][:18]:<18} 100%[===================>] {_wh(n):>7}  --.-KB/s    in 0s      \n\n"
                  f"{when} ({max(n, 1) / 1024:.1f} KB/s) - ‘{name}’ saved [{n}/{n}]\n\n")
        node = ctx.fs.write_file(name, body, ctx.cwd, 0o644, ctx.uid, ctx.proc.gid)
        node.mode = 0o664 & ~ctx.proc.umask
    return status
    yield


def _wh(n):
    for unit, size in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= size:
            return f"{n / size:.1f}{unit}" if n / size < 10 else f"{n / size:.0f}{unit}"
    return f"{n}"


@util("ping")
def u_ping(ctx):
    a = ctx.argv[1:]
    count = None
    interval = 1.0
    quiet = False
    target = None
    i = 0
    while i < len(a):
        x = a[i]
        if x == "-c" and i + 1 < len(a):
            count = int(a[i + 1])
            i += 2
            continue
        if x.startswith("-c") and x[2:].isdigit():
            count = int(x[2:])
        elif x == "-i" and i + 1 < len(a):
            interval = float(a[i + 1])
            i += 2
            continue
        elif x in ("-W", "-w", "-s") and i + 1 < len(a):
            i += 2
            continue
        elif x == "-q":
            quiet = True
        else:
            target = x
        i += 1
    if target is None:
        ctx.e("ping: usage error: Destination address required\n")
        return 1
    host = ctx.world.resolve(target)
    ip = host.ip if host else INTERNET.get(target)
    if ip is None and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", target):
        ip = target
    if ip is None:
        ctx.e(f"ping: {target}: Name or service not known\n")
        return 2
    ctx.w(f"PING {target} ({ip}) 56(84) bytes of data.\n")
    yield from ctx.flush()
    sent = recv = 0
    times = []
    start = ctx.now
    tty = ctx.world.tty_of(ctx.proc)
    fg = ctx.foreground()
    seq = 0
    while count is None or seq < count:
        seq += 1
        sent += 1
        up = host is None or ctx.now >= host.extra.get("up_after", 0) and not host.extra.get("down")
        if up:
            t = 11.0 + ((seq * 37) % 10) / 10 if host is None else 0.4 + ((seq * 13) % 9) / 10
            times.append(t)
            recv += 1
            if not quiet:
                ctx.w(f"64 bytes from {ip}: icmp_seq={seq} ttl={56 if host is None else 64} time={t:.{1 if t >= 10 else 3}f} ms\n")
        elif not quiet:
            ctx.w(f"From {ctx.host.ip} icmp_seq={seq} Destination Host Unreachable\n")
        yield from ctx.flush()
        if count is not None and seq >= count:
            break
        if count is None and fg and (ctx.now - start) / NS + interval > ctx.world.ctrl_c_after:
            ctx.world.tty.write("echo", b"^C\n")
            break
        yield from ctx.sleep(interval)
    elapsed = int((ctx.now - start) / 1_000_000)
    loss = 100 * (sent - recv) // sent if sent else 0
    ctx.w(f"\n--- {target} ping statistics ---\n{sent} packets transmitted, {recv} received, "
          f"{'+' + str(sent - recv) + ' errors, ' if recv < sent and False else ''}{loss}% packet loss, time {elapsed}ms\n")
    if times:
        avg = sum(times) / len(times)
        mdev = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        ctx.w(f"rtt min/avg/max/mdev = {min(times):.3f}/{avg:.3f}/{max(times):.3f}/{mdev:.3f} ms\n")
    return 0 if recv else 1

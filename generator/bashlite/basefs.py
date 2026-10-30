"""The filesystem every simulated host starts with (an Ubuntu 24.04-like layout)."""

from __future__ import annotations

BASHRC = """\
# ~/.bashrc: executed by bash(1) for non-login shells.

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
HISTCONTROL=ignoreboth
shopt -s histappend
HISTSIZE=1000
HISTFILESIZE=2000
shopt -s checkwinsize

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\\[\\033[01;32m\\]\\u@\\h\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\\u@\\h:\\w\\$ '
fi
unset color_prompt force_color_prompt

# enable color support of ls and also add handy aliases
alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
"""

PROFILE = """\
# ~/.profile: executed by the command interpreter for login shells.
if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc"
    fi
fi
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
"""

OS_RELEASE = """\
PRETTY_NAME="Ubuntu 24.04.1 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.1 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
UBUNTU_CODENAME=noble
"""


def build_base(host) -> None:
    from .exec import UTILS, stub_bytes
    from . import utils  # noqa: F401  (registers every command)
    fs = host.fs
    host.add_user("root", 0, 0, "root", "/root")
    for gid, name in ((4, "adm"), (24, "cdrom"), (27, "sudo"), (100, "users"), (999, "docker"), (1001, "ml"),
                      (33, "www-data"), (65534, "nogroup"), (101, "crontab")):
        host.groups[gid] = (name, set())
    host.add_user("ada", 1000, 1000, "Ada Lovelace,,,", groups=("adm", "sudo", "docker", "ml"))
    host.add_user("bob", 1002, 1002, "Bob,,,", groups=("ml",))
    from .world import User
    host.users[33] = User("www-data", 33, 33, "www-data", "/var/www", "/usr/sbin/nologin")
    host.users[65534] = User("nobody", 65534, 65534, "nobody", "/nonexistent", "/usr/sbin/nologin")

    def d(path, mode=0o755, uid=0, gid=0):
        node = fs.mkdir(path, "/", mode, uid, gid, parents=True)
        node.mode, node.uid, node.gid = mode, uid, gid
        return node

    def f(path, text, mode=0o644, uid=0, gid=0):
        node = fs.write_file(path, text, "/", mode, uid, gid)
        node.mode = mode
        return node
    for p in ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/local/bin", "/usr/local/sbin", "/usr/share/man", "/etc/ssh",
              "/var/log", "/var/lib", "/opt", "/mnt", "/srv", "/media", "/run", "/boot", "/snap"):
        d(p)
    for link, target in (("/bin", "usr/bin"), ("/sbin", "usr/sbin"), ("/lib", "usr/lib")):
        fs.symlink(target, link)
    d("/tmp", 0o1777)
    d("/var/tmp", 0o1777)
    d("/var/spool/cron/crontabs", 0o1730, 0, 101)
    d("/root", 0o700)
    d("/dev")
    d("/proc", 0o555)
    d("/data", 0o755, 1000, 1000)
    for name in ("null", "zero", "urandom", "random", "tty", "full"):
        fs.mknod("/dev/" + name, name if name != "full" else "null", 0o666)
    fs.symlink("/proc/self/fd", "/dev/fd")
    for i, n in enumerate(("stdin", "stdout", "stderr")):
        fs.symlink(f"/proc/self/fd/{i}", f"/dev/{n}")
    d("/dev/shm", 0o1777)
    # users
    users = sorted(host.users.values(), key=lambda u: u.uid)
    f("/etc/passwd", "".join(f"{u.name}:x:{u.uid}:{u.gid}:{u.gecos}:{u.home}:{u.shell}\n" for u in users))
    host.groups.setdefault(42, ("shadow", set()))
    f("/etc/group", "".join(f"{n}:x:{g}:{','.join(sorted(m))}\n" for g, (n, m) in sorted(host.groups.items())))
    shadow = f("/etc/shadow", "".join(f"{u.name}:{'$y$j9T$' + 'x' * 43 if u.uid >= 1000 or u.uid == 0 else '*'}:"
                                      f"19800:0:99999:7:::\n" for u in users), 0o640)
    shadow.gid = 42
    f("/etc/hostname", host.name + "\n")
    f("/etc/hosts", f"127.0.0.1\tlocalhost\n127.0.1.1\t{host.name}\n\n::1\tip6-localhost ip6-loopback\n")
    f("/etc/os-release", OS_RELEASE)
    f("/etc/lsb-release", "DISTRIB_ID=Ubuntu\nDISTRIB_RELEASE=24.04\nDISTRIB_CODENAME=noble\n"
                          "DISTRIB_DESCRIPTION=\"Ubuntu 24.04.1 LTS\"\n")
    f("/etc/shells", "# /etc/shells: valid login shells\n/bin/sh\n/bin/bash\n/usr/bin/bash\n/bin/dash\n/usr/bin/dash\n")
    f("/etc/timezone", "Etc/UTC\n")
    f("/etc/environment", 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\n')
    f("/etc/bash.bashrc", "# System-wide .bashrc file for interactive bash(1) shells.\n")
    f("/etc/profile", "# /etc/profile: system-wide .profile file for the Bourne shell (sh(1))\n")
    f("/etc/ssh/ssh_config", "Host *\n    SendEnv LANG LC_*\n    HashKnownHosts yes\n")
    f("/etc/crontab", "SHELL=/bin/sh\n# m h dom mon dow user\tcommand\n17 *\t* * *\troot\tcd / && run-parts --report /etc/cron.hourly\n")
    f("/proc/cpuinfo", "".join(f"processor\t: {i}\nvendor_id\t: GenuineIntel\nmodel name\t: {host.cpu_model}\n"
                               f"cpu MHz\t\t: 2000.000\ncpu cores\t: {host.cpus}\n\n" for i in range(host.cpus)), 0o444)
    total = host.mem_kib
    f("/proc/meminfo", f"MemTotal:       {total} kB\nMemFree:        {total * 61 // 100} kB\n"
                       f"MemAvailable:   {total * 78 // 100} kB\nBuffers:          {total // 90} kB\n"
                       f"Cached:          {total // 7} kB\nSwapTotal:       2097148 kB\nSwapFree:        2097148 kB\n", 0o444)
    f("/proc/version", f"Linux version {host.kernel_release} (buildd@lcy02-amd64-080) (x86_64-linux-gnu-gcc-13 (Ubuntu "
                       f"13.2.0-23ubuntu4) 13.2.0, GNU ld (GNU Binutils for Ubuntu) 2.42) #45-Ubuntu SMP PREEMPT_DYNAMIC "
                       f"Fri Aug 30 12:02:04 UTC 2024\n", 0o444)
    f("/proc/loadavg", " ".join(f"{x:.2f}" for x in host.load) + " 1/412 4242\n", 0o444)
    # programs
    from .utils import GNU_ALIASES, MACOS_ONLY, NOT_PREINSTALLED
    skip = GNU_ALIASES | NOT_PREINSTALLED | (MACOS_ONLY if host.os != "macos" else set())
    for name in sorted(UTILS):
        if name in skip:
            continue
        if name in ("[",):
            path = "/usr/bin/["
        elif "/" in name:
            continue
        else:
            path = "/usr/bin/" + name
        node = fs.write_file(path, stub_bytes(name), "/", 0o755, 0, 0)
        node.mode = 0o755
    for alias, target in (("python3", "python3.12"), ("python", "python3"), ("pip3", "pip"), ("sh", "dash"),
                          ("vi", "vim"), ("awk", "gawk"), ("rgrep", "grep"), ("pager", "less")):
        if fs.exists("/usr/bin/" + target) and not fs.exists("/usr/bin/" + alias, follow=False):
            fs.symlink(target, "/usr/bin/" + alias)
    # home directory
    home = "/home/ada"
    d("/home", 0o755)
    d(home, 0o750, 1000, 1000)
    f(home + "/.bashrc", BASHRC, 0o644, 1000, 1000)
    f(home + "/.profile", PROFILE, 0o644, 1000, 1000)
    f(home + "/.bash_logout", "# ~/.bash_logout: executed by bash(1) when login shell exits.\n", 0o644, 1000, 1000)
    d(home + "/.cache", 0o700, 1000, 1000)
    d(home + "/.local/bin", 0o755, 1000, 1000)
    d(home + "/.local/share", 0o755, 1000, 1000)
    d("/home/bob", 0o750, 1002, 1002)
    # everything above is "old": make the files a few months older than the session start
    old = host.world.kernel.now - 86400 * 1_000_000_000 * 120
    for _, node in fs.walk(fs.root, "/"):
        node.mtime = node.atime = node.ctime = old
    for p in (home, home + "/.bashrc", home + "/.profile", home + "/.cache", "/tmp"):
        n = fs.lookup(p)
        n.mtime = n.atime = n.ctime = host.world.kernel.now - 86400 * 1_000_000_000 * 3

"""Simulated commands. Importing this package registers every one in exec.UTILS."""

from . import shells, basic, files, text, grep, sed, findx, awk, bc, jq, archives, procs, misc, net, tmux, pyenv  # noqa: F401
from ..exec import UTILS

UTILS.setdefault("awk", UTILS["gawk"])
GNU_ALIASES: set[str] = set()       # installed by `brew install coreutils ...` on macOS, absent on Linux
# programs a fresh machine doesn't have: package managers' console scripts, GPU tools (added with the GPUs), macOS-only
NOT_PREINSTALLED = {"uv", "conda", "jupyter", "jupyter-lab", "tensorboard", "torchrun", "nvidia-smi", "nvtop"}
MACOS_ONLY = {"md5", "brew"}
# Homebrew's GNU tools on macOS (gls, gsed, ggrep, gfind, ...) are the GNU implementations
for _name in ("ls", "cat", "date", "sort", "head", "tail", "cut", "wc", "stat", "du", "cp", "mv", "rm", "readlink",
              "realpath", "sha256sum", "md5sum", "sed", "grep", "egrep", "fgrep", "find", "xargs", "touch", "mktemp",
              "basename", "dirname", "seq", "tr", "uniq", "tac", "nl", "paste", "join", "split", "comm", "chmod"):
    if _name in UTILS and "g" + _name not in UTILS:
        UTILS["g" + _name] = UTILS[_name]
        GNU_ALIASES.add("g" + _name)

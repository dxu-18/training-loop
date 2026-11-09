"""bash-lite's git: a git implementation on the simulated filesystem (objects with real SHA-1 ids, refs, the
index, reflogs, config) with the porcelain commands people use, plus GitHub-lite (github.py) and the gh CLI.
Importing this package registers the `git` program (cli.py) and every command module."""

from . import cli  # noqa: F401
from . import cmd_basic, switching, cmd_history, cmd_branch, cmd_plumbing, help  # noqa: F401,E402
from . import cmd_remote, cmd_clone, cmd_extra, gh_core, gh_pr, tools, gitstate  # noqa: F401,E402

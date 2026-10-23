"""`help` text for the builtins: (synopsis, one-line summary, details). Short paraphrases of bash's help."""

BUILTIN_HELP = {
    ":": (": [arguments]", "Null command.", "No effect; the command does nothing.\n\nExit Status:\nAlways succeeds."),
    ".": (". filename [arguments]", "Execute commands from a file in the current shell.",
          "Read and execute commands from FILENAME in the current shell.  The\nentries in $PATH are used to find the "
          "directory containing FILENAME.\nIf any ARGUMENTS are supplied, they become the positional parameters\n"
          "when FILENAME is executed.\n\nExit Status:\nReturns the status of the last command executed in FILENAME; "
          "fails if\nFILENAME cannot be read."),
    "alias": ("alias [-p] [name[=value] ... ]", "Define or display aliases.",
              "Without arguments, `alias' prints the list of aliases in the reusable\nform `alias NAME=VALUE' on "
              "standard output.\n\nOtherwise, an alias is defined for each NAME whose VALUE is given.\nA trailing space "
              "in VALUE causes the next word to be checked for\nalias substitution when the alias is expanded.\n\n"
              "Options:\n  -p\tprint all defined aliases in a reusable format\n\nExit Status:\nalias returns true unless "
              "a NAME is supplied for which no alias has been\ndefined."),
    "bg": ("bg [job_spec ...]", "Move jobs to the background.",
           "Place the jobs identified by each JOB_SPEC in the background, as if they\nhad been started with `&'.  If "
           "JOB_SPEC is not present, the shell's notion\nof the current job is used.\n\nExit Status:\nReturns success "
           "unless job control is not enabled or an error occurs."),
    "break": ("break [n]", "Exit for, while, or until loops.",
              "Exit a FOR, WHILE or UNTIL loop.  If N is specified, break N enclosing\nloops.\n\nExit Status:\nThe exit "
              "status is 0 unless N is not greater than or equal to 1."),
    "builtin": ("builtin [shell-builtin [arg ...]]", "Execute shell builtins.",
                "Execute SHELL-BUILTIN with arguments ARGs without performing command\nlookup.  This is useful when you "
                "wish to reimplement a shell builtin\nas a shell function, but need to execute the builtin within the "
                "function."),
    "cd": ("cd [-L|[-P [-e]] [-@]] [dir]", "Change the shell working directory.",
           "Change the current directory to DIR.  The default DIR is the value of the\nHOME shell variable. If DIR is "
           "\"-\", it is converted to $OLDPWD.\n\nThe variable CDPATH defines the search path for the directory "
           "containing\nDIR.  Alternative directory names in CDPATH are separated by a colon (:).\n\nOptions:\n  -L\t"
           "force symbolic links to be followed: resolve symbolic\n\t\tlinks in DIR after processing instances of `..'\n"
           "  -P\tuse the physical directory structure without following\n\t\tsymbolic links: resolve symbolic links in "
           "DIR before\n\t\tprocessing instances of `..'\n\nExit Status:\nReturns 0 if the directory is changed, and if "
           "$PWD is set successfully when\n-P is used; non-zero otherwise."),
    "command": ("command [-pVv] command [arg ...]", "Execute a simple command or display information about commands.",
                "Runs COMMAND with ARGS suppressing  shell function lookup, or display\ninformation about the specified "
                "COMMANDs.\n\nOptions:\n  -p    use a default value for PATH\n  -v    print a single word indicating "
                "the command or filename that\n        invokes COMMAND\n  -V    print a more verbose description of each "
                "COMMAND\n\nExit Status:\nReturns exit status of COMMAND, or failure if COMMAND is not found."),
    "continue": ("continue [n]", "Resume for, while, or until loops.",
                 "Resumes the next iteration of the enclosing FOR, WHILE or UNTIL loop.\nIf N is specified, resumes the "
                 "Nth enclosing loop."),
    "declare": ("declare [-aAfFgiIlnrtux] [name[=value] ...] or declare -p [-aAfFilnrtux] [name ...]",
                "Set variable values and attributes.",
                "Declare variables and give them attributes.  If no NAMEs are given,\ndisplay the attributes and values "
                "of all variables.\n\nOptions:\n  -f\trestrict action or display to function names and definitions\n"
                "  -F\trestrict display to function names only\n  -g\tcreate global variables when used in a shell "
                "function\n  -p\tdisplay the attributes and value of each NAME\n\nOptions which set attributes:\n  -a\tto "
                "make NAMEs indexed arrays (if supported)\n  -A\tto make NAMEs associative arrays (if supported)\n  -i\tto "
                "make NAMEs have the `integer' attribute\n  -l\tto convert the value of each NAME to lower case on "
                "assignment\n  -n\tmake NAME a reference to the variable named by its value\n  -r\tto make NAMEs readonly\n"
                "  -u\tto convert the value of each NAME to upper case on assignment\n  -x\tto make NAMEs export\n\n"
                "Using `+' instead of `-' turns off the given attribute."),
    "dirs": ("dirs [-clpv] [+N] [-N]", "Display directory stack.",
             "Display the list of currently remembered directories.  Directories\nfind their way onto the list with the "
             "`pushd' command; you can get\nback up through the list with the `popd' command.\n\nOptions:\n  -c\tclear "
             "the directory stack by deleting all of the elements\n  -l\tdo not print tilde-prefixed versions of "
             "directories relative\n\t\tto your home directory\n  -p\tprint the directory stack with one entry per line\n"
             "  -v\tprint the directory stack with one entry per line prefixed\n\t\twith its position in the stack"),
    "disown": ("disown [-h] [-ar] [jobspec ... | pid ...]", "Remove jobs from current shell.",
               "Removes each JOBSPEC argument from the table of active jobs.  Without\nany JOBSPECs, the shell uses its "
               "notion of the current job.\n\nOptions:\n  -a\tremove all jobs if JOBSPEC is not supplied\n  -h\tmark each "
               "JOBSPEC so that SIGHUP is not sent to the job if the\n\t\tshell receives a SIGHUP\n  -r\tremove only "
               "running jobs"),
    "echo": ("echo [-neE] [arg ...]", "Write arguments to the standard output.",
             "Display the ARGs, separated by a single space character and followed by a\nnewline, on the standard output."
             "\n\nOptions:\n  -n\tdo not append a newline\n  -e\tenable interpretation of the following backslash escapes"
             "\n  -E\texplicitly suppress interpretation of backslash escapes\n\n`echo' interprets the following "
             "backslash-escaped characters:\n  \\a\talert (bell)\n  \\b\tbackspace\n  \\c\tsuppress further output\n  "
             "\\e\tescape character\n  \\f\tform feed\n  \\n\tnew line\n  \\r\tcarriage return\n  \\t\thorizontal tab\n  "
             "\\v\tvertical tab\n  \\\\\tbackslash\n  \\0nnn\tthe character whose ASCII code is NNN (octal)\n  \\xHH\tthe "
             "eight-bit character whose value is HH (hexadecimal)"),
    "eval": ("eval [arg ...]", "Execute arguments as a shell command.",
             "Combine ARGs into a single string, use the result as input to the shell,\nand execute the resulting "
             "commands."),
    "exec": ("exec [-cl] [-a name] [command [argument ...]] [redirection ...]",
             "Replace the shell with the given command.",
             "Execute COMMAND, replacing this shell with the specified program.\nARGUMENTS become the arguments to "
             "COMMAND.  If COMMAND is not specified,\nany redirections take effect in the current shell."),
    "exit": ("exit [n]", "Exit the shell.",
             "Exits the shell with a status of N.  If N is omitted, the exit status\nis that of the last command "
             "executed."),
    "export": ("export [-fn] [name[=value] ...] or export -p", "Set export attribute for shell variables.",
               "Marks each NAME for automatic export to the environment of subsequently\nexecuted commands.  If VALUE is "
               "supplied, assign VALUE before exporting.\n\nOptions:\n  -f\trefer to shell functions\n  -n\tremove the "
               "export property from each NAME\n  -p\tdisplay a list of all exported variables and functions"),
    "false": ("false", "Return an unsuccessful result.", "Exit Status:\nAlways fails."),
    "fc": ("fc [-e ename] [-lnr] [first] [last] or fc -s [pat=rep] [command]", "Display or execute commands from the history list.",
           "fc is used to list or edit and re-execute commands from the history list.\n\nOptions:\n  -l\tlist lines "
           "instead of editing\n  -s\tre-execute a command after making the substitutions pat=rep"),
    "fg": ("fg [job_spec]", "Move job to the foreground.",
           "Place the job identified by JOB_SPEC in the foreground, making it the\ncurrent job.  If JOB_SPEC is not "
           "present, the shell's notion of the\ncurrent job is used.\n\nExit Status:\nStatus of command placed in "
           "foreground, or failure if an error occurs."),
    "getopts": ("getopts optstring name [arg ...]", "Parse option arguments.",
                "Getopts is used by shell procedures to parse positional parameters\nas options.\n\nOPTSTRING contains "
                "the option letters to be recognized; if a letter\nis followed by a colon, the option is expected to have "
                "an argument,\nwhich should be separated from it by white space.\n\nEach time it is invoked, getopts will "
                "place the next option in the\nshell variable $name, initializing name if it does not exist, and\nthe "
                "index of the next argument to be processed into the shell\nvariable OPTIND.  OPTIND is initialized to 1 "
                "each time the shell or\na shell script is invoked.  When an option requires an argument,\ngetopts places "
                "that argument into the shell variable OPTARG.\n\nIf the first character of OPTSTRING is a colon, getopts "
                "uses silent\nerror reporting."),
    "hash": ("hash [-lr] [-p pathname] [-dt] [name ...]", "Remember or display program locations.",
             "Determine and remember the full pathname of each command NAME.  If\nno arguments are given, information "
             "about remembered commands is displayed.\n\nOptions:\n  -d\tforget the remembered location of each NAME\n"
             "  -r\tforget all remembered locations\n  -t\tprint the remembered location of each NAME"),
    "help": ("help [-dms] [pattern ...]", "Display information about builtin commands.",
             "Displays brief summaries of builtin commands.  If PATTERN is\nspecified, gives detailed help on all "
             "commands matching PATTERN,\notherwise the list of help topics is printed.\n\nOptions:\n  -d\toutput short "
             "description for each topic\n  -s\toutput only a short usage synopsis for each topic matching\n\t\tPATTERN"),
    "history": ("history [-c] [-d offset] [n] or history -anrw [filename] or history -ps arg [arg...]",
                "Display or manipulate the history list.",
                "Display the history list with line numbers, prefixing each modified\nentry with a `*'.  An argument of N "
                "lists only the last N entries.\n\nOptions:\n  -c\tclear the history list by deleting all of the entries\n"
                "  -d offset\tdelete the history entry at position OFFSET.\n  -w\twrite the current history to the "
                "history file"),
    "jobs": ("jobs [-lnprs] [jobspec ...] or jobs -x command [args]", "Display status of jobs.",
             "Lists the active jobs.  JOBSPEC restricts output to that job.\nWithout options, the status of all active "
             "jobs is displayed.\n\nOptions:\n  -l\tlists process IDs in addition to the normal information\n  -p\tlists "
             "process IDs only\n  -r\trestrict output to running jobs\n  -s\trestrict output to stopped jobs"),
    "kill": ("kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]",
             "Send a signal to a job.",
             "Send the processes identified by PID or JOBSPEC the signal named by\nSIGSPEC or SIGNUM.  If neither SIGSPEC "
             "nor SIGNUM is present, then\nSIGTERM is assumed.\n\nOptions:\n  -s sig\tSIG is a signal name\n  -n sig\tSIG "
             "is a signal number\n  -l\tlist the signal names; if arguments follow `-l' they are\n\t\tassumed to be "
             "signal numbers for which names should be listed"),
    "let": ("let arg [arg ...]", "Evaluate arithmetic expressions.",
            "Evaluate each ARG as an arithmetic expression.  Evaluation is done in\nfixed-width integers with no check "
            "for overflow.\n\nExit Status:\nIf the last ARG evaluates to 0, let returns 1; let returns 0 otherwise."),
    "local": ("local [option] name[=value] ...", "Define local variables.",
              "Create a local variable called NAME, and give it VALUE.  OPTION can\nbe any option accepted by `declare'."
              "\n\nLocal variables can only be used within a function; they are visible\nonly to the function where "
              "they are defined and its children."),
    "logout": ("logout [n]", "Exit a login shell.", "Exits a login shell with exit status N."),
    "mapfile": ("mapfile [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quantum] [array]",
                "Read lines from the standard input into an indexed array variable.",
                "Read lines from the standard input into the indexed array variable ARRAY, or\nfrom file descriptor FD "
                "if the -u option is supplied.  The variable MAPFILE\nis the default ARRAY.\n\nOptions:\n  -d delim\tUse "
                "DELIM to terminate lines, instead of newline\n  -n count\tCopy at most COUNT lines\n  -O origin\tBegin "
                "assigning to ARRAY at index ORIGIN\n  -s count\tDiscard the first COUNT lines read\n  -t\tRemove a "
                "trailing DELIM from each line read (default newline)\n  -u fd\tRead lines from file descriptor FD"),
    "popd": ("popd [-n] [+N | -N]", "Remove directories from stack.",
             "Removes entries from the directory stack.  With no arguments, removes\nthe top directory from the stack, "
             "and changes to the new top directory."),
    "printf": ("printf [-v var] format [arguments]", "Formats and prints ARGUMENTS under control of the FORMAT.",
               "Options:\n  -v var\tassign the output to shell variable VAR rather than\n\t\tdisplay it on the standard "
               "output\n\nFORMAT is a character string which contains three types of objects: plain\ncharacters, which "
               "are simply copied to standard output; character escape\nsequences, which are converted and copied to the "
               "standard output; and\nformat specifications, each of which causes printing of the next successive\n"
               "argument.\n\nIn addition to the standard format characters csndiouxXeEfFgGaA described\nin printf(3), "
               "printf interprets:\n\n  %b\texpand backslash escape sequences in the corresponding argument\n  %q\tquote "
               "the argument in a way that can be reused as shell input\n  %(fmt)T\toutput the date-time string "
               "resulting from using FMT as a format\n\t        string for strftime(3)\n\nThe format is re-used as "
               "necessary to consume all of the arguments."),
    "pushd": ("pushd [-n] [+N | -N | dir]", "Add directories to stack.",
              "Adds a directory to the top of the directory stack, or rotates\nthe stack, making the new top of the "
              "stack the current working\ndirectory.  With no arguments, exchanges the top two directories."),
    "pwd": ("pwd [-LP]", "Print the name of the current working directory.",
            "Options:\n  -L\tprint the value of $PWD if it names the current working\n\t\tdirectory\n  -P\tprint the "
            "physical directory, without any symbolic links\n\nBy default, `pwd' behaves as if `-L' were specified."),
    "read": ("read [-ers] [-a array] [-d delim] [-i text] [-n nchars] [-N nchars] [-p prompt] [-t timeout] [-u fd] [name ...]",
             "Read a line from the standard input and split it into fields.",
             "Reads a single line from the standard input, or from file descriptor FD\nif the -u option is supplied.  "
             "The line is split into fields as with word\nsplitting, and the first word is assigned to the first NAME, "
             "the second\nword to the second NAME, and so on, with any leftover words assigned to\nthe last NAME.  Only "
             "the characters found in $IFS are recognized as word\ndelimiters.\n\nIf no NAMEs are supplied, the line "
             "read is stored in the REPLY variable.\n\nOptions:\n  -a array\tassign the words read to sequential indices "
             "of the array\n\t\tvariable ARRAY, starting at zero\n  -d delim\tcontinue until the first character of DELIM "
             "is read, rather\n\t\tthan newline\n  -n nchars\treturn after reading NCHARS characters\n  -p prompt\toutput "
             "the string PROMPT without a trailing newline before\n\t\tattempting to read\n  -r\tdo not allow "
             "backslashes to escape any characters\n  -s\tdo not echo input coming from a terminal\n  -t timeout\ttime "
             "out and return failure if a complete line of\n\t\tinput is not read within TIMEOUT seconds\n  -u fd\tread "
             "from file descriptor FD instead of the standard input"),
    "readonly": ("readonly [-aAf] [name[=value] ...] or readonly -p", "Mark shell variables as unchangeable.",
                 "Mark each NAME as read-only; the values of these NAMEs may not be\nchanged by subsequent assignment."),
    "return": ("return [n]", "Return from a shell function.",
               "Causes a function or sourced script to exit with the return value\nspecified by N.  If N is omitted, the "
               "return status is that of the\nlast command executed within the function or script."),
    "set": ("set [-abefhkmnptuvxBCEHPT] [-o option-name] [--] [-] [arg ...]",
            "Set or unset values of shell options and positional parameters.",
            "Change the value of shell attributes and positional parameters, or\ndisplay the names and values of shell "
            "variables.\n\nOptions:\n  -a  Mark variables which are modified or created for export.\n  -e  Exit "
            "immediately if a command exits with a non-zero status.\n  -f  Disable file name generation (globbing).\n  "
            "-m  Job control is enabled.\n  -u  Treat unset variables as an error when substituting.\n  -x  Print "
            "commands and their arguments as they are executed.\n  -C  If set, disallow existing regular files to be "
            "overwritten\n      by redirection of output.\n  -o option-name\n      Set the variable corresponding to "
            "option-name, e.g. pipefail:\n          the return value of a pipeline is the status of\n          the last "
            "command to exit with a non-zero status,\n          or zero if no command exited with a non-zero status\n"
            "  --  Assign any remaining arguments to the positional parameters.\n\nUsing + rather than - causes these "
            "flags to be turned off."),
    "shift": ("shift [n]", "Shift positional parameters.",
              "Rename the positional parameters $N+1,$N+2 ... to $1,$2 ...  If N is\nnot given, it is assumed to be 1."),
    "shopt": ("shopt [-pqsu] [-o] [optname ...]", "Set and unset shell options.",
              "Change the setting of each shell option OPTNAME.  Without any option\narguments, list each supplied "
              "OPTNAME, or all shell options if no\nOPTNAMEs are given, with an indication of whether or not each is "
              "set.\n\nOptions:\n  -o\trestrict OPTNAMEs to those defined for use with `set -o'\n  -p\tprint each shell "
              "option with an indication of its status\n  -q\tsuppress output\n  -s\tenable (set) each OPTNAME\n  -u\t"
              "disable (unset) each OPTNAME"),
    "source": ("source filename [arguments]", "Execute commands from a file in the current shell.",
               "Read and execute commands from FILENAME in the current shell.  The\nentries in $PATH are used to find the "
               "directory containing FILENAME."),
    "suspend": ("suspend [-f]", "Suspend shell execution.", "Suspend the execution of this shell until it receives a SIGCONT signal."),
    "test": ("test [expr]", "Evaluate conditional expression.",
             "Exits with a status of 0 (true) or 1 (false) depending on\nthe evaluation of EXPR.\n\nFile operators:\n\n"
             "  -d FILE        True if file is a directory.\n  -e FILE        True if file exists.\n  -f FILE        True "
             "if file exists and is a regular file.\n  -r FILE        True if file is readable by you.\n  -s FILE        "
             "True if file exists and is not empty.\n  -w FILE        True if the file is writable by you.\n  -x FILE     "
             "   True if the file is executable by you.\n  -L FILE        True if file is a symbolic link.\n\nString "
             "operators:\n\n  -z STRING      True if string is empty.\n  -n STRING      True if string is not empty.\n  "
             "STRING1 = STRING2\n                 True if the strings are equal.\n  STRING1 != STRING2\n                 "
             "True if the strings are not equal.\n\nOther operators:\n\n  ! EXPR         True if expr is false.\n  EXPR1 "
             "-a EXPR2  True if both expr1 AND expr2 are true.\n  EXPR1 -o EXPR2  True if either expr1 OR expr2 is true.\n"
             "\n  arg1 OP arg2   Arithmetic tests.  OP is one of -eq, -ne,\n                 -lt, -le, -gt, or -ge."),
    "[": ("[ arg... ]", "Evaluate conditional expression.",
          "This is a synonym for the \"test\" builtin, but the last argument must\nbe a literal `]', to match the opening "
          "`['."),
    "times": ("times", "Display process times.",
              "Prints the accumulated user and system times for the shell and all of its\nchild processes."),
    "trap": ("trap [-lp] [[arg] signal_spec ...]", "Trap signals and other events.",
             "Defines and activates handlers to be run when the shell receives signals\nor other conditions.\n\nARG is a "
             "command to be read and executed when the shell receives the\nsignal(s) SIGNAL_SPEC.  If ARG is absent (and "
             "a single SIGNAL_SPEC\nis supplied) or `-', each specified signal is reset to its original\nvalue.  If ARG is "
             "the null string each SIGNAL_SPEC is ignored by the\nshell and by the commands it invokes.\n\nIf a SIGNAL_SPEC "
             "is EXIT (0) ARG is executed on exit from the shell.\nIf a SIGNAL_SPEC is ERR, ARG is executed each time a "
             "command\nfails.\n\nOptions:\n  -l\tprint a list of signal names and their corresponding numbers\n  -p\t"
             "display the trap commands associated with each SIGNAL_SPEC"),
    "true": ("true", "Return a successful result.", "Exit Status:\nAlways succeeds."),
    "type": ("type [-afptP] name [name ...]", "Display information about command type.",
             "For each NAME, indicate how it would be interpreted if used as a\ncommand name.\n\nOptions:\n  -a\tdisplay "
             "all locations containing an executable named NAME;\n\t\tincludes aliases, builtins, and functions\n  -f\t"
             "suppress shell function lookup\n  -P\tforce a PATH search for each NAME\n  -p\treturns the name of the "
             "disk file that would be executed\n  -t\toutput a single word which is one of `alias', `keyword',\n\t\t"
             "`function', `builtin', `file' or `', if NAME is an alias,\n\t\tshell reserved word, shell function, shell "
             "builtin, disk file,\n\t\tor not found, respectively"),
    "typeset": ("typeset [-aAfFgiIlnrtux] name[=value] ... or typeset -p [-aAfFilnrtux] [name ...]",
                "Set variable values and attributes.", "A synonym for `declare'.  See `help declare'."),
    "ulimit": ("ulimit [-SHabcdefiklmnpqrstuvxPRT] [limit]", "Modify shell resource limits.",
               "Provides control over the resources available to the shell and processes\nit creates.\n\n  -a\tall "
               "current limits are reported\n  -n\tthe maximum number of open file descriptors"),
    "umask": ("umask [-p] [-S] [mode]", "Display or set file mode mask.",
              "Sets the user file-creation mask to MODE.  If MODE is omitted, prints\nthe current value of the mask.\n\n"
              "Options:\n  -S\tmakes the output symbolic; otherwise an octal number is output"),
    "unalias": ("unalias [-a] name [name ...]", "Remove each NAME from the list of defined aliases.",
                "Options:\n  -a\tremove all alias definitions"),
    "unset": ("unset [-f] [-v] [-n] [name ...]", "Unset values and attributes of shell variables and functions.",
              "For each NAME, remove the corresponding variable or function.\n\nOptions:\n  -f\ttreat each NAME as a shell "
              "function\n  -v\ttreat each NAME as a shell variable"),
    "wait": ("wait [-fn] [-p var] [id ...]", "Wait for job completion and return exit status.",
             "Waits for each process identified by an ID, which may be a process ID or a\njob specification, and reports "
             "its termination status.  If ID is not\ngiven, waits for all currently active child processes, and the "
             "return\nstatus is zero.\n\nIf the -n option is supplied, waits for a single job from the list of IDs,\nor, "
             "if no IDs are supplied, for the next job to complete and returns its\nexit status."),
    "readarray": ("readarray [-d delim] [-n count] [-O origin] [-s count] [-t] [-u fd] [-C callback] [-c quantum] [array]",
                  "Read lines from a file into an array variable.", "A synonym for `mapfile'."),
    "caller": ("caller [expr]", "Return the context of the current subroutine call.", ""),
    "compgen": ("compgen [-abcdefgjksuv] [-o option] [-A action] [-G globpat] [-W wordlist] [word]",
                "Display possible completions depending on the options.", ""),
    "complete": ("complete [-abcdefgjksuv] [-pr] [-DEI] [-o option] [-A action] [-W wordlist] [name ...]",
                 "Specify how arguments are to be completed by Readline.", ""),
    "enable": ("enable [-a] [-dnps] [-f filename] [name ...]", "Enable and disable shell builtins.", ""),
    "bind": ("bind [-lpsvPSVX] [-m keymap] [-f filename] [-q name] [-u name] [-r keyseq] [-x keyseq:shell-command] [keyseq:readline-function or readline-command]",
             "Set Readline key bindings and variables.", ""),
}

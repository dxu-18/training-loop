"""Manual pages, --help text and tldr pages for the simulated commands (our own wording, GNU layout)."""

# name: (section, short description, synopsis, description, [(flags, text)])
PAGES = {
    "ls": ("1", "list directory contents", "ls [OPTION]... [FILE]...",
           "List information about the FILEs (the current directory by default). Sort entries alphabetically if none of -cftuvSUX nor --sort is specified.",
           [("-a, --all", "do not ignore entries starting with ."), ("-A, --almost-all", "do not list implied . and .."),
            ("-d, --directory", "list directories themselves, not their contents"),
            ("-F, --classify[=WHEN]", "append indicator (one of */=>@|) to entries WHEN"),
            ("-h, --human-readable", "with -l and -s, print sizes like 1K 234M 2G etc."),
            ("-i, --inode", "print the index number of each file"), ("-l", "use a long listing format"),
            ("-r, --reverse", "reverse order while sorting"), ("-R, --recursive", "list subdirectories recursively"),
            ("-s, --size", "print the allocated size of each file, in blocks"),
            ("-S", "sort by file size, largest first"), ("-t", "sort by time, newest first; see --time"),
            ("-X", "sort alphabetically by entry extension"), ("-1", "list one file per line"),
            ("--color[=WHEN]", "color the output WHEN; more info below"),
            ("--group-directories-first", "group directories before files")]),
    "cp": ("1", "copy files and directories", "cp [OPTION]... [-T] SOURCE DEST\n       cp [OPTION]... SOURCE... DIRECTORY",
           "Copy SOURCE to DEST, or multiple SOURCE(s) to DIRECTORY.",
           [("-a, --archive", "same as -dR --preserve=all"), ("-i, --interactive", "prompt before overwrite"),
            ("-n, --no-clobber", "do not overwrite an existing file"),
            ("-p", "same as --preserve=mode,ownership,timestamps"), ("-r, -R, --recursive", "copy directories recursively"),
            ("-t, --target-directory=DIRECTORY", "copy all SOURCE arguments into DIRECTORY"),
            ("-u, --update", "copy only when the SOURCE file is newer than the destination file"),
            ("-v, --verbose", "explain what is being done")]),
    "mv": ("1", "move (rename) files", "mv [OPTION]... [-T] SOURCE DEST\n       mv [OPTION]... SOURCE... DIRECTORY",
           "Rename SOURCE to DEST, or move SOURCE(s) to DIRECTORY.",
           [("-f, --force", "do not prompt before overwriting"), ("-i, --interactive", "prompt before overwrite"),
            ("-n, --no-clobber", "do not overwrite an existing file"),
            ("-t, --target-directory=DIRECTORY", "move all SOURCE arguments into DIRECTORY"),
            ("-v, --verbose", "explain what is being done")]),
    "rm": ("1", "remove files or directories", "rm [OPTION]... [FILE]...",
           "This manual page documents the GNU version of rm. rm removes each specified file. By default, it does not remove directories.",
           [("-f, --force", "ignore nonexistent files and arguments, never prompt"), ("-i", "prompt before every removal"),
            ("-I", "prompt once before removing more than three files, or when removing recursively"),
            ("-r, -R, --recursive", "remove directories and their contents recursively"),
            ("-d, --dir", "remove empty directories"), ("-v, --verbose", "explain what is being done")]),
    "mkdir": ("1", "make directories", "mkdir [OPTION]... DIRECTORY...", "Create the DIRECTORY(ies), if they do not already exist.",
              [("-m, --mode=MODE", "set file mode (as in chmod), not a=rwx - umask"),
               ("-p, --parents", "no error if existing, make parent directories as needed"),
               ("-v, --verbose", "print a message for each created directory")]),
    "rmdir": ("1", "remove empty directories", "rmdir [OPTION]... DIRECTORY...", "Remove the DIRECTORY(ies), if they are empty.",
              [("-p, --parents", "remove DIRECTORY and its ancestors"), ("-v, --verbose", "output a diagnostic for every directory processed")]),
    "touch": ("1", "change file timestamps", "touch [OPTION]... FILE...",
              "Update the access and modification times of each FILE to the current time. A FILE argument that does not exist is created empty.",
              [("-a", "change only the access time"), ("-c, --no-create", "do not create any files"),
               ("-d, --date=STRING", "parse STRING and use it instead of current time"), ("-m", "change only the modification time"),
               ("-r, --reference=FILE", "use this file's times instead of current time"),
               ("-t STAMP", "use [[CC]YY]MMDDhhmm[.ss] instead of current time")]),
    "ln": ("1", "make links between files", "ln [OPTION]... [-T] TARGET LINK_NAME",
           "Create a link to TARGET with the name LINK_NAME. Create hard links by default, symbolic links with --symbolic.",
           [("-f, --force", "remove existing destination files"), ("-n, --no-dereference", "treat LINK_NAME as a normal file if it is a symbolic link to a directory"),
            ("-r, --relative", "with -s, create links relative to link location"), ("-s, --symbolic", "make symbolic links instead of hard links"),
            ("-v, --verbose", "print name of each linked file")]),
    "cat": ("1", "concatenate files and print on the standard output", "cat [OPTION]... [FILE]...",
            "Concatenate FILE(s) to standard output. With no FILE, or when FILE is -, read standard input.",
            [("-A, --show-all", "equivalent to -vET"), ("-b, --number-nonblank", "number nonempty output lines, overrides -n"),
             ("-E, --show-ends", "display $ at end of each line"), ("-n, --number", "number all output lines"),
             ("-s, --squeeze-blank", "suppress repeated empty output lines"), ("-T, --show-tabs", "display TAB characters as ^I")]),
    "tac": ("1", "concatenate and print files in reverse", "tac [OPTION]... [FILE]...", "Write each FILE to standard output, last line first.", []),
    "nl": ("1", "number lines of files", "nl [OPTION]... [FILE]...", "Write each FILE to standard output, with line numbers added.",
           [("-b, --body-numbering=STYLE", "use STYLE for numbering body lines (a: all, t: non-empty, n: none)"),
            ("-w, --number-width=NUMBER", "use NUMBER columns for line numbers")]),
    "head": ("1", "output the first part of files", "head [OPTION]... [FILE]...",
             "Print the first 10 lines of each FILE to standard output. With more than one FILE, precede each with a header giving the file name.",
             [("-c, --bytes=[-]NUM", "print the first NUM bytes of each file; with the leading '-', print all but the last NUM bytes"),
              ("-n, --lines=[-]NUM", "print the first NUM lines instead of the first 10; with the leading '-', print all but the last NUM lines"),
              ("-q, --quiet", "never print headers giving file names")]),
    "tail": ("1", "output the last part of files", "tail [OPTION]... [FILE]...",
             "Print the last 10 lines of each FILE to standard output.",
             [("-c, --bytes=[+]NUM", "output the last NUM bytes; or use -c +NUM to output starting with byte NUM of each file"),
              ("-f, --follow[={name|descriptor}]", "output appended data as the file grows"),
              ("-F", "same as --follow=name --retry"),
              ("-n, --lines=[+]NUM", "output the last NUM lines, instead of the last 10; or use -n +NUM to skip NUM-1 lines at the start"),
              ("--pid=PID", "with -f, terminate after process ID, PID dies")]),
    "less": ("1", "opposite of more", "less [options] [file...]",
             "Less is a program similar to more, but which allows backward movement in the file as well as forward movement. "
             "Commands: SPACE or f forward one window; b backward one window; /pattern search forward; n repeat search; "
             "N repeat search in reverse; g go to first line; G go to last line; F follow (like tail -f); q quit.",
             [("-N, --LINE-NUMBERS", "display line numbers"), ("-S, --chop-long-lines", "chop (truncate) long lines rather than wrapping them"),
              ("-R, --RAW-CONTROL-CHARS", "output ANSI color escape sequences in raw form"), ("+F", "start in follow mode")]),
    "diff": ("1", "compare files line by line", "diff [OPTION]... FILES", "Compare FILES line by line.",
             [("-q, --brief", "report only when files differ"), ("-s, --report-identical-files", "report when two files are the same"),
              ("-u, -U NUM, --unified[=NUM]", "output NUM (default 3) lines of unified context"),
              ("-y, --side-by-side", "output in two columns"), ("-r, --recursive", "recursively compare any subdirectories found"),
              ("-i, --ignore-case", "ignore case differences in file contents"), ("-w, --ignore-all-space", "ignore all white space")]),
    "cmp": ("1", "compare two files byte by byte", "cmp [OPTION]... FILE1 [FILE2 [SKIP1 [SKIP2]]]", "Compare two files byte by byte.",
            [("-l, --verbose", "output byte numbers and differing byte values"), ("-s, --quiet, --silent", "suppress all normal output")]),
    "comm": ("1", "compare two sorted files line by line", "comm [OPTION]... FILE1 FILE2",
             "Compare sorted files FILE1 and FILE2 line by line. With no options, produce three-column output: lines unique to FILE1, lines unique to FILE2, and lines common to both.",
             [("-1", "suppress column 1 (lines unique to FILE1)"), ("-2", "suppress column 2 (lines unique to FILE2)"),
              ("-3", "suppress column 3 (lines that appear in both files)")]),
    "md5sum": ("1", "compute and check MD5 message digest", "md5sum [OPTION]... [FILE]...", "Print or check MD5 (128-bit) checksums.",
               [("-c, --check", "read checksums from the FILEs and check them")]),
    "sha256sum": ("1", "compute and check SHA256 message digest", "sha256sum [OPTION]... [FILE]...", "Print or check SHA256 (256-bit) checksums.",
                  [("-c, --check", "read checksums from the FILEs and check them")]),
    "xxd": ("1", "make a hex dump or do the reverse", "xxd [options] [infile [outfile]]", "xxd creates a hex dump of a given file or standard input.",
            [("-l len", "stop after writing <len> octets"), ("-p", "output in plain hexdump style"), ("-r", "reverse operation: convert hexdump into binary")]),
    "strings": ("1", "print the sequences of printable characters in files", "strings [-n min-len] [file...]",
                "For each file given, strings prints the printable character sequences that are at least 4 characters long.", [("-n min-len", "print sequences of at least min-len characters")]),
    "column": ("1", "columnate lists", "column [options] [file ...]", "The column utility formats its input into multiple columns.",
               [("-t, --table", "determine the number of columns the input contains and create a table"),
                ("-s, --separator separators", "specify the possible input item delimiters"),
                ("-o, --output-separator string", "specify the columns delimiter for table output")]),
    "chmod": ("1", "change file mode bits", "chmod [OPTION]... MODE[,MODE]... FILE...\n       chmod [OPTION]... OCTAL-MODE FILE...",
              "chmod changes the file mode bits of each given file according to mode, which can be either a symbolic representation of changes to make, or an octal number representing the bit pattern for the new mode bits. The format of a symbolic mode is [ugoa...][[-+=][perms...]...], where perms is either zero or more letters from the set rwxXst.",
              [("-R, --recursive", "change files and directories recursively"), ("-v, --verbose", "output a diagnostic for every file processed")]),
    "chown": ("1", "change file owner and group", "chown [OPTION]... [OWNER][:[GROUP]] FILE...", "Change the owner and/or group of each FILE to OWNER and/or GROUP.",
              [("-R, --recursive", "operate on files and directories recursively")]),
    "tar": ("1", "an archiving utility", "tar [OPTION...] [FILE]...",
            "GNU tar saves many files together into a single tape or disk archive, and can restore individual files from the archive.",
            [("-c, --create", "create a new archive"), ("-x, --extract", "extract files from an archive"),
             ("-t, --list", "list the contents of an archive"), ("-f, --file=ARCHIVE", "use archive file or device ARCHIVE"),
             ("-z, --gzip", "filter the archive through gzip"), ("-j, --bzip2", "filter the archive through bzip2"),
             ("-J, --xz", "filter the archive through xz"), ("-v, --verbose", "verbosely list files processed"),
             ("-C, --directory=DIR", "change to DIR before performing any operations"),
             ("--exclude=PATTERN", "exclude files matching PATTERN")]),
    "gzip": ("1", "compress or expand files", "gzip [ -cdfhklLnNrtvV19 ] [name ...]",
             "Gzip reduces the size of the named files. Whenever possible, each file is replaced by one with the extension .gz.",
             [("-c --stdout", "write on standard output, keep original files unchanged"), ("-d --decompress", "decompress"),
              ("-k --keep", "keep (don't delete) input files"), ("-l --list", "list compressed file contents"),
              ("-r --recursive", "operate recursively on directories")]),
    "zip": ("1", "package and compress (archive) files", "zip [-options] zipfile list", "zip is a compression and file packaging utility.",
            [("-r", "travel the directory structure recursively"), ("-q", "quiet mode"), ("-j", "junk (don't record) directory names")]),
    "unzip": ("1", "list, test and extract compressed files in a ZIP archive", "unzip [-Z] [-cflptTuvz[abjnoqsCDKLMUVWX$/:^]] file[.zip] [file(s) ...] [-x xfile(s) ...] [-d exdir]",
              "unzip will list, test, or extract files from a ZIP archive.",
              [("-l", "list archive files (short format)"), ("-d exdir", "an optional directory to which to extract files"),
               ("-o", "overwrite existing files without prompting"), ("-q", "perform operations quietly")]),
    "du": ("1", "estimate file space usage", "du [OPTION]... [FILE]...", "Summarize device usage of the set of FILEs, recursively for directories.",
           [("-a, --all", "write counts for all files, not just directories"), ("-c, --total", "produce a grand total"),
            ("-d, --max-depth=N", "print the total for a directory only if it is N or fewer levels below the command line argument"),
            ("-h, --human-readable", "print sizes in human readable format (e.g., 1K 234M 2G)"),
            ("-s, --summarize", "display only a total for each argument")]),
    "df": ("1", "report file system space usage", "df [OPTION]... [FILE]...", "df displays the amount of space available on the file system containing each file name argument.",
           [("-h, --human-readable", "print sizes in powers of 1024 (e.g., 1023M)"), ("-T, --print-type", "print file system type")]),
    "find": ("1", "search for files in a directory hierarchy", "find [-H] [-L] [-P] [starting-point...] [expression]",
             "find searches the directory tree rooted at each given starting-point by evaluating the given expression from left to right.",
             [("-name pattern", "base of file name matches shell pattern pattern"), ("-iname pattern", "like -name, but the match is case insensitive"),
              ("-type c", "file is of type c: d directory, f regular file, l symbolic link"),
              ("-size n[cwbkMG]", "file uses n units of space, rounding up"), ("-mtime n", "file's data was last modified n*24 hours ago"),
              ("-maxdepth levels", "descend at most levels levels of directories below the starting-points"),
              ("-path pattern", "file name matches shell pattern pattern"), ("-not expr", "true if expr is false"),
              ("-delete", "delete files; true if removal succeeded"), ("-exec command ;", "execute command; true if 0 status is returned"),
              ("-exec command {} +", "run command with as many found file names appended as possible"),
              ("-print0", "print the full file name on the standard output, followed by a null character")]),
    "grep": ("1", "print lines that match patterns", "grep [OPTION...] PATTERNS [FILE...]",
             "grep searches for PATTERNS in each FILE. A FILE of \"-\" stands for standard input. grep prints each line that matches a pattern.",
             [("-E, --extended-regexp", "interpret PATTERNS as extended regular expressions"), ("-F, --fixed-strings", "interpret PATTERNS as fixed strings, not regular expressions"),
              ("-P, --perl-regexp", "interpret PATTERNS as Perl-compatible regular expressions"),
              ("-i, --ignore-case", "ignore case distinctions in patterns and input data"), ("-v, --invert-match", "invert the sense of matching, to select non-matching lines"),
              ("-w, --word-regexp", "select only those lines containing matches that form whole words"),
              ("-c, --count", "print a count of matching lines for each input file"), ("-l, --files-with-matches", "print the name of each input file that matches"),
              ("-o, --only-matching", "print only the matched (non-empty) parts of a matching line"), ("-n, --line-number", "prefix each line of output with its line number"),
              ("-r, --recursive", "read all files under each directory, recursively"),
              ("-A NUM, --after-context=NUM", "print NUM lines of trailing context after matching lines"),
              ("-B NUM, --before-context=NUM", "print NUM lines of leading context before matching lines"),
              ("--include=GLOB", "search only files whose base name matches GLOB"), ("--exclude-dir=GLOB", "skip any command-line directory with a name suffix that matches GLOB")]),
    "sed": ("1", "stream editor for filtering and transforming text", "sed [OPTION]... {script-only-if-no-other-script} [input-file]...",
            "Sed is a stream editor. A stream editor is used to perform basic text transformations on an input stream.",
            [("-n, --quiet, --silent", "suppress automatic printing of pattern space"), ("-e script", "add the script to the commands to be executed"),
             ("-i[SUFFIX], --in-place[=SUFFIX]", "edit files in place (makes backup if SUFFIX supplied)"),
             ("-E, -r, --regexp-extended", "use extended regular expressions in the script")]),
    "awk": ("1", "pattern scanning and processing language", "awk [ POSIX or GNU style options ] -f program-file [ -- ] file ...\n       awk [ POSIX or GNU style options ] [ -- ] program-text file ...",
            "Gawk is the GNU Project's implementation of the AWK programming language.",
            [("-F fs, --field-separator fs", "use fs for the input field separator"), ("-v var=val, --assign var=val", "assign the value val to the variable var"),
             ("-f program-file", "read the AWK program source from the file program-file")]),
    "sort": ("1", "sort lines of text files", "sort [OPTION]... [FILE]...", "Write sorted concatenation of all FILE(s) to standard output.",
             [("-f, --ignore-case", "fold lower case to upper case characters"), ("-h, --human-numeric-sort", "compare human readable numbers (e.g., 2K 1G)"),
              ("-n, --numeric-sort", "compare according to string numerical value"), ("-r, --reverse", "reverse the result of comparisons"),
              ("-k, --key=KEYDEF", "sort via a key; KEYDEF gives location and type"), ("-t, --field-separator=SEP", "use SEP instead of non-blank to blank transition"),
              ("-u, --unique", "output only the first of an equal run"), ("-V, --version-sort", "natural sort of (version) numbers within text")]),
    "uniq": ("1", "report or omit repeated lines", "uniq [OPTION]... [INPUT [OUTPUT]]", "Filter adjacent matching lines from INPUT (or standard input).",
             [("-c, --count", "prefix lines by the number of occurrences"), ("-d, --repeated", "only print duplicate lines, one for each group"),
              ("-u, --unique", "only print unique lines"), ("-i, --ignore-case", "ignore differences in case when comparing")]),
    "cut": ("1", "remove sections from each line of files", "cut OPTION... [FILE]...", "Print selected parts of lines from each FILE to standard output.",
            [("-c, --characters=LIST", "select only these characters"), ("-d, --delimiter=DELIM", "use DELIM instead of TAB for field delimiter"),
             ("-f, --fields=LIST", "select only these fields")]),
    "tr": ("1", "translate or delete characters", "tr [OPTION]... STRING1 [STRING2]", "Translate, squeeze, and/or delete characters from standard input, writing to standard output.",
           [("-c, -C, --complement", "use the complement of STRING1"), ("-d, --delete", "delete characters in STRING1, do not translate"),
            ("-s, --squeeze-repeats", "replace each sequence of a repeated character that is listed in the last specified STRING, with a single occurrence of that character")]),
    "wc": ("1", "print newline, word, and byte counts for each file", "wc [OPTION]... [FILE]...", "Print newline, word, and byte counts for each FILE.",
           [("-c, --bytes", "print the byte counts"), ("-m, --chars", "print the character counts"), ("-l, --lines", "print the newline counts"),
            ("-w, --words", "print the word counts")]),
    "paste": ("1", "merge lines of files", "paste [OPTION]... [FILE]...", "Write lines consisting of the sequentially corresponding lines from each FILE, separated by TABs.",
              [("-d, --delimiters=LIST", "reuse characters from LIST instead of TABs"), ("-s, --serial", "paste one file at a time instead of in parallel")]),
    "join": ("1", "join lines of two files on a common field", "join [OPTION]... FILE1 FILE2", "For each pair of input lines with identical join fields, write a line to standard output.",
             [("-t CHAR", "use CHAR as input and output field separator"), ("-a FILENUM", "also print unpairable lines from file FILENUM"),
              ("-1 FIELD", "join on this FIELD of file 1"), ("-2 FIELD", "join on this FIELD of file 2")]),
    "split": ("1", "split a file into pieces", "split [OPTION]... [FILE [PREFIX]]", "Output pieces of FILE to PREFIXaa, PREFIXab, ...; default size is 1000 lines, and default PREFIX is 'x'.",
              [("-l, --lines=NUMBER", "put NUMBER lines/records per output file"), ("-b, --bytes=SIZE", "put SIZE bytes per output file"),
               ("-d", "use numeric suffixes starting at 0, not alphabetic")]),
    "xargs": ("1", "build and execute command lines from standard input", "xargs [options] [command [initial-arguments]]",
              "xargs reads items from the standard input, delimited by blanks or newlines, and executes the command one or more times with the items as arguments.",
              [("-0, --null", "input items are terminated by a null character instead of by whitespace"), ("-I replace-str", "replace occurrences of replace-str in the initial-arguments with names read from standard input"),
               ("-n max-args", "use at most max-args arguments per command line"), ("-P max-procs", "run up to max-procs processes at a time"),
               ("-r, --no-run-if-empty", "if the standard input does not contain any nonblanks, do not run the command"),
               ("-t, --verbose", "print the command line on the standard error output before executing it")]),
    "jq": ("1", "Command-line JSON processor", "jq [options...] filter [files...]",
           "jq can transform JSON in various ways, by selecting, iterating, reducing and otherwise mangling JSON documents.",
           [("-r, --raw-output", "output strings directly rather than as JSON"), ("-c, --compact-output", "compact instead of pretty-printed output"),
            ("-s, --slurp", "read all inputs into an array"), ("-n, --null-input", "use null as the single input value"),
            ("--arg name value", "set $name to the string value")]),
    "bc": ("1", "An arbitrary precision calculator language", "bc [ -hlwsqv ] [long-options] [  file ... ]",
           "bc is a language that supports arbitrary precision numbers with interactive execution of statements. The variable scale sets the number of digits after the decimal point.",
           [("-l, --mathlib", "define the standard math library (scale=20, s c a l e j functions)")]),
    "seq": ("1", "print a sequence of numbers", "seq [OPTION]... LAST\n       seq [OPTION]... FIRST LAST\n       seq [OPTION]... FIRST INCREMENT LAST",
            "Print numbers from FIRST to LAST, in steps of INCREMENT.",
            [("-s, --separator=STRING", "use STRING to separate numbers (default: \\n)"), ("-w, --equal-width", "equalize width by padding with leading zeroes")]),
    "tee": ("1", "read from standard input and write to standard output and files", "tee [OPTION]... [FILE]...",
            "Copy standard input to each FILE, and also to standard output.", [("-a, --append", "append to the given FILEs, do not overwrite")]),
    "basename": ("1", "strip directory and suffix from filenames", "basename NAME [SUFFIX]\n       basename OPTION... NAME...",
                 "Print NAME with any leading directory components removed. If specified, also remove a trailing SUFFIX.", [("-a, --multiple", "support multiple arguments and treat each as a NAME"), ("-s, --suffix=SUFFIX", "remove a trailing SUFFIX; implies -a")]),
    "dirname": ("1", "strip last component from file name", "dirname [OPTION] NAME...", "Output each NAME with its last non-slash component and trailing slashes removed.", []),
    "readlink": ("1", "print resolved symbolic links or canonical file names", "readlink [OPTION]... FILE...", "Print value of a symbolic link or canonical file name.",
                 [("-f, --canonicalize", "canonicalize by following every symlink in every component of the given name recursively")]),
    "realpath": ("1", "print the resolved path", "realpath [OPTION]... FILE...", "Print the resolved absolute file name; all but the last component must exist.",
                 [("-e, --canonicalize-existing", "all components of the path must exist"), ("--relative-to=DIR", "print the resolved path relative to DIR")]),
    "stat": ("1", "display file or file system status", "stat [OPTION]... FILE...", "Display file or file system status.",
             [("-c --format=FORMAT", "use the specified FORMAT instead of the default"), ("-L, --dereference", "follow links")]),
    "file": ("1", "determine file type", "file [-bchiklLNnprsSvzZ0] file ...", "file tests each argument in an attempt to classify it.", [("-b, --brief", "do not prepend filenames to output lines")]),
    "tree": ("1", "list contents of directories in a tree-like format", "tree [-acdfghilnpqrstuvxACDFJQNSUX] [-L level [-R]] [directory ...]",
             "Tree is a recursive directory listing program that produces a depth indented listing of files.",
             [("-a", "all files are printed"), ("-d", "list directories only"), ("-L level", "max display depth of the directory tree"), ("-I pattern", "do not list those files that match the wild-card pattern")]),
    "ps": ("1", "report a snapshot of the current processes", "ps [options]", "ps displays information about a selection of the active processes.",
           [("aux", "every process, BSD syntax, with user-oriented columns"), ("-e", "select all processes"), ("-f", "do full-format listing"),
            ("-o format", "user-defined format"), ("-p pidlist", "select by process ID"), ("--sort spec", "specify sorting order")]),
    "pgrep": ("1", "look up, signal, or wait for processes based on name and other attributes", "pgrep [options] pattern",
              "pgrep looks through the currently running processes and lists the process IDs which match the selection criteria to stdout.",
              [("-a, --list-full", "list the full command line as well as the process ID"), ("-f, --full", "match against the full command line"),
               ("-l, --list-name", "list the process name as well as the process ID"), ("-u, --euid euid,...", "only match processes whose effective user ID is listed")]),
    "pkill": ("1", "look up, signal, or wait for processes based on name and other attributes", "pkill [options] pattern", "pkill sends the specified signal (by default SIGTERM) to each process instead of listing them on stdout.",
              [("-f, --full", "match against the full command line"), ("-signal", "defines the signal to send to each matched process")]),
    "kill": ("1", "send a signal to a process", "kill [options] <pid> [...]", "The default signal for kill is TERM. Use -l or -L to list available signals.",
             [("-s, --signal <signal>", "specify the signal to be sent"), ("-l, --list [signal]", "list signal names")]),
    "top": ("1", "display Linux processes", "top -hv|-bcEeHiOSs1 -d secs -n max -u|U user -p pids -o field -w [cols]",
            "The top program provides a dynamic real-time view of a running system. Press q to quit.", [("-b", "batch-mode operation"), ("-n", "number of iterations")]),
    "nohup": ("1", "run a command immune to hangups, with output to a non-tty", "nohup COMMAND [ARG]...",
              "Run COMMAND, ignoring hangup signals. If standard output is a terminal, append output to 'nohup.out'.", []),
    "timeout": ("1", "run a command with a time limit", "timeout [OPTION] DURATION COMMAND [ARG]...",
                "Start COMMAND, and kill it if still running after DURATION. If the command times out, exit with status 124.",
                [("-s, --signal=SIGNAL", "specify the signal to be sent on timeout"), ("-k, --kill-after=DURATION", "also send a KILL signal if COMMAND is still running this long after the initial signal")]),
    "watch": ("1", "execute a program periodically, showing output fullscreen", "watch [options] command", "watch runs command repeatedly, displaying its output.",
              [("-n, --interval seconds", "specify update interval")]),
    "lsof": ("8", "list open files", "lsof [ -t ] [ -i [i] ] [ -p s ] [ names ]", "Lsof lists information about files opened by processes.",
             [("-i [i]", "selects the listing of files any of whose Internet address matches the address specified in i, e.g. -i :8888"), ("-t", "produce terse output: process identifiers only")]),
    "ssh": ("1", "OpenSSH remote login client", "ssh [-46AaCfGgKkMNnqsTtVvXxYy] [-i identity_file] [-J destination] [-L address] [-p port] destination [command [argument ...]]",
            "ssh (SSH client) is a program for logging into a remote machine and for executing commands on a remote machine.",
            [("-i identity_file", "selects a file from which the identity (private key) is read"), ("-J destination", "connect to the target host by first making a ssh connection to the jump host"),
             ("-L [bind_address:]port:host:hostport", "specifies that connections to the given TCP port on the local host are to be forwarded to the given host and port on the remote side"),
             ("-N", "do not execute a remote command; useful for just forwarding ports"), ("-f", "requests ssh to go to background just before command execution"),
             ("-p port", "port to connect to on the remote host")]),
    "scp": ("1", "OpenSSH secure file copy", "scp [-346ABCOpqRrsTv] [-P port] source ... target", "scp copies files between hosts on a network.",
            [("-r", "recursively copy entire directories"), ("-P port", "specifies the port to connect to on the remote host")]),
    "rsync": ("1", "a fast, versatile, remote (and local) file-copying tool", "rsync [OPTION...] SRC... [DEST]",
              "Rsync is a fast and extraordinarily versatile file copying tool. A trailing slash on the source copies the contents of the directory.",
              [("-a, --archive", "archive mode is -rlptgoD (no -A,-X,-U,-N,-H)"), ("-v, --verbose", "increase verbosity"), ("-z, --compress", "compress file data during the transfer"),
               ("-P", "same as --partial --progress"), ("-n, --dry-run", "perform a trial run with no changes made"),
               ("--delete", "delete extraneous files from dest dirs"), ("--exclude=PATTERN", "exclude files matching PATTERN")]),
    "curl": ("1", "transfer a URL", "curl [options / URLs]", "curl is a tool for transferring data from or to a server using URLs.",
             [("-L, --location", "follow redirects"), ("-o, --output <file>", "write to file instead of stdout"), ("-O, --remote-name", "write output to a file named as the remote file"),
              ("-s, --silent", "silent or quiet mode"), ("-X, --request <method>", "specify request method to use"), ("-H, --header <header>", "pass custom header(s) to server"),
              ("-d, --data <data>", "HTTP POST data"), ("-i, --include", "include protocol response headers in the output")]),
    "wget": ("1", "The non-interactive network downloader", "wget [option]... [URL]...", "GNU Wget is a free utility for non-interactive download of files from the Web.",
             [("-c, --continue", "continue getting a partially-downloaded file"), ("-q, --quiet", "turn off Wget's output"), ("-O file", "write documents to file")]),
    "tmux": ("1", "terminal multiplexer", "tmux [-2CDluvV] [-c shell-command] [-f file] [-L socket-name] [-S socket-path] [-T features] [command [flags]]",
             "tmux is a terminal multiplexer: it enables a number of terminals to be created, accessed, and controlled from a single screen. "
             "The default command key bindings are prefix C-b followed by: d detach, c new window, n/p next/previous window, % split left/right, \" split top/bottom, z zoom pane, [ copy mode, x kill pane, , rename window.",
             [("new-session [-d] [-s session-name]", "create a new session"), ("attach-session [-t target-session]", "attach to an existing session"),
              ("list-sessions", "list all sessions"), ("kill-session [-t target-session]", "destroy the given session")]),
    "nvidia-smi": ("1", "NVIDIA System Management Interface program", "nvidia-smi [OPTION1 [ARG1]] [OPTION2 [ARG2]] ...",
                   "nvidia-smi provides monitoring and management capabilities for NVIDIA GPUs.",
                   [("-l, --loop=SEC", "probe until Ctrl+C at specified second interval"), ("--query-gpu=", "information about GPU"),
                    ("--query-compute-apps=", "list of active compute processes"), ("--format=", "comma separated list of format options: csv, noheader, nounits")]),
    "python3": ("1", "an interpreted, interactive, object-oriented programming language", "python3 [ -c command | script | - ] [ arguments ]",
                "Python is an interpreted, interactive, object-oriented programming language.",
                [("-c command", "specify the command to execute"), ("-m module", "run library module as a script")]),
    "pip": ("1", "A tool for installing and managing Python packages", "pip <command> [options]", "pip is the package installer for Python.",
            [("install", "install packages"), ("uninstall", "uninstall packages"), ("freeze", "output installed packages in requirements format"),
             ("list", "list installed packages"), ("show", "show information about installed packages")]),
    "man": ("1", "an interface to the system reference manuals", "man [man options] [[section] page ...] ...",
            "man is the system's manual pager. Each page argument given to man is normally the name of a program, utility or function.",
            [("-k, --apropos", "equivalent to apropos"), ("-f, --whatis", "equivalent to whatis")]),
    "date": ("1", "print or set the system date and time", "date [OPTION]... [+FORMAT]", "Display date and time in the given FORMAT.",
             [("-d, --date=STRING", "display time described by STRING, not 'now'"), ("-u, --utc, --universal", "print or set Coordinated Universal Time (UTC)"),
              ("-I[FMT], --iso-8601[=FMT]", "output date/time in ISO 8601 format"), ("-R, --rfc-email", "output date and time in RFC 5322 format")]),
    "echo": ("1", "display a line of text", "echo [SHORT-OPTION]... [STRING]...", "Echo the STRING(s) to standard output.",
             [("-n", "do not output the trailing newline"), ("-e", "enable interpretation of backslash escapes")]),
    "printf": ("1", "format and print data", "printf FORMAT [ARGUMENT]...", "Print ARGUMENT(s) according to FORMAT.", []),
    "sleep": ("1", "delay for a specified amount of time", "sleep NUMBER[SUFFIX]...", "Pause for NUMBER seconds. SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.", []),
    "which": ("1", "locate a command", "which [-a] filename ...", "which returns the pathnames of the files which would be executed in the current environment.",
              [("-a", "print all matching pathnames of each argument")]),
    "crontab": ("1", "maintain crontab files for individual users", "crontab [ -u user ] file\n       crontab [ -u user ] [ -i ] { -e | -l | -r }",
                "crontab is the program used to install a crontab table file. Each line is: minute hour day-of-month month day-of-week command.",
                [("-l", "the current crontab will be displayed on standard output"), ("-r", "the current crontab will be removed"),
                 ("-e", "edit the current crontab using the editor specified by VISUAL or EDITOR")]),
    "shellcheck": ("1", "Shell script analysis tool", "shellcheck [OPTIONS...] FILES...", "ShellCheck is a static analysis and linting tool for sh/bash scripts.", []),
    "bash": ("1", "GNU Bourne-Again SHell", "bash [options] [command_string | file]",
             "Bash is an sh-compatible command language interpreter that executes commands read from the standard input or from a file.",
             [("-c", "commands are read from the first non-option argument command_string"), ("-x", "print commands and their arguments as they are executed"),
              ("-e", "exit immediately if a pipeline returns a non-zero status"), ("-n", "read commands but do not execute them")]),
    "mktemp": ("1", "create a temporary file or directory", "mktemp [OPTION]... [TEMPLATE]", "Create a temporary file or directory, safely, and print its name.",
               [("-d, --directory", "create a directory, not a file"), ("-p DIR, --tmpdir[=DIR]", "interpret TEMPLATE relative to DIR")]),
    "env": ("1", "run a program in a modified environment", "env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]", "Set each NAME to VALUE in the environment and run COMMAND.",
            [("-i, --ignore-environment", "start with an empty environment"), ("-u, --unset=NAME", "remove variable from the environment")]),
    "uname": ("1", "print system information", "uname [OPTION]...", "Print certain system information.", [("-a, --all", "print all information"), ("-r, --kernel-release", "print the kernel release")]),
    "free": ("1", "Display amount of free and used memory in the system", "free [options]", "free displays the total amount of free and used physical and swap memory in the system.",
             [("-h, --human", "show all output fields automatically scaled to shortest three digit unit")]),
    "nproc": ("1", "print the number of processing units available", "nproc [OPTION]...", "Print the number of processing units available to the current process.", []),
    "uptime": ("1", "Tell how long the system has been running.", "uptime [options]", "uptime gives a one line display of the current time, how long the system has been running, how many users are currently logged on, and the system load averages.", []),
    "ssh-keygen": ("1", "OpenSSH authentication key utility", "ssh-keygen [-q] [-a rounds] [-b bits] [-C comment] [-f output_keyfile] [-t dsa | ecdsa | ed25519 | rsa]",
                   "ssh-keygen generates, manages and converts authentication keys for ssh.", [("-t type", "specifies the type of key to create"), ("-C comment", "provides a new comment"), ("-f filename", "specifies the filename of the key file")]),
    "conda": ("1", "package and environment manager", "conda [-h] [-v] command ...", "conda is a tool for managing and deploying applications, environments and packages.", []),
    "uv": ("1", "An extremely fast Python package manager", "uv [OPTIONS] <COMMAND>", "uv manages Python versions, virtual environments and packages.", []),
}

ALIASES = {"python": "python3", "pip3": "pip", "dir": "ls", "vdir": "ls", "egrep": "grep", "fgrep": "grep", "gawk": "awk",
           "gunzip": "gzip", "zcat": "gzip", "more": "less", "sha1sum": "sha256sum", "sha512sum": "sha256sum",
           "pkill": "pkill", "htop": "top"}

TLDR = {
    "tar": ("Archiving utility.\n  Often combined with a compression method, such as gzip or bzip2.",
            [("Create an archive and write it to a file", "tar cf path/to/target.tar path/to/file1 path/to/file2 ..."),
             ("Create a gzipped archive and write it to a file", "tar czf path/to/target.tar.gz path/to/file1 path/to/file2 ..."),
             ("Extract a (compressed) archive file into the current directory verbosely", "tar xvf path/to/source.tar[.gz|.bz2|.xz]"),
             ("Extract a (compressed) archive file into the target directory", "tar xf path/to/source.tar[.gz|.bz2|.xz] -C path/to/directory"),
             ("List the contents of a tar file verbosely", "tar tvf path/to/source.tar"),
             ("Extract files matching a pattern from an archive file", "tar xf path/to/source.tar --wildcards \"*.html\"")]),
    "find": ("Find files or directories under a directory tree, recursively.",
             [("Find files by extension", "find path/to/directory -name '*.ext'"),
              ("Find directories matching a given name, in case-insensitive mode", "find path/to/directory -type d -iname '*lib*'"),
              ("Find files matching a given pattern, excluding specific paths", "find path/to/directory -name '*.py' -not -path '*/site-packages/*'"),
              ("Find files matching a given size range, limiting the recursive depth to \"1\"", "find path/to/directory -maxdepth 1 -size +500k -size -10M"),
              ("Run a command for each file (use `{}` within the command to access the filename)", "find path/to/directory -name '*.ext' -exec wc -l {} \\;"),
              ("Find files modified in the last 7 days", "find path/to/directory -daystart -mtime -7"),
              ("Find empty (0 byte) files and delete them", "find path/to/directory -type f -empty -delete")]),
    "grep": ("Find patterns in files using regular expressions.",
             [("Search for a pattern within a file", "grep \"search_pattern\" path/to/file"),
              ("Search in a case-insensitive manner", "grep -i \"search_pattern\" path/to/file"),
              ("Search recursively in the current directory", "grep -r \"search_pattern\" ."),
              ("Print line numbers and context lines around each match", "grep -n -C 2 \"search_pattern\" path/to/file"),
              ("Print only the matching parts with an extended regular expression", "grep -oE \"[0-9]+\" path/to/file"),
              ("Print lines that do not match the pattern", "grep -v \"search_pattern\" path/to/file")]),
    "sed": ("Edit text in a scriptable manner.",
            [("Replace the first occurrence of a regex in each line and print the result", "sed 's/regex/replace/' path/to/file"),
             ("Replace all occurrences of a pattern in a file, in place", "sed -i 's/find/replace/g' path/to/file"),
             ("Delete lines matching a pattern", "sed '/regex/d' path/to/file"),
             ("Print only lines 10 to 20", "sed -n '10,20p' path/to/file")]),
    "awk": ("A versatile programming language for working on files.",
            [("Print the fifth column (a.k.a. field) in a space-separated file", "awk '{print $5}' path/to/file"),
             ("Print the second column of the lines containing \"foo\"", "awk '/foo/ {print $2}' path/to/file"),
             ("Print the last column of each line, using a comma as a field separator", "awk -F ',' '{print $NF}' path/to/file"),
             ("Sum the values in the first column and print the total", "awk '{s+=$1} END {print s}' path/to/file")]),
    "rsync": ("Transfer files either to or from a remote host (but not between two remote hosts), by default using SSH.",
              [("Transfer a file", "rsync path/to/source path/to/destination"),
               ("Use archive mode, verbose output and compression, showing progress", "rsync -avzP path/to/source path/to/destination"),
               ("Copy the contents of a directory (note the trailing slash)", "rsync -av path/to/source/ path/to/destination"),
               ("Delete files in the destination that are not in the source (dry run first)", "rsync -avn --delete path/to/source/ path/to/destination")]),
    "ssh": ("Secure Shell is a protocol used to securely log onto remote systems.",
            [("Connect to a remote server", "ssh username@remote_host"),
             ("Connect to a remote server with a specific identity (private key)", "ssh -i path/to/key_file username@remote_host"),
             ("Connect to a remote server using a specific port", "ssh username@remote_host -p 2222"),
             ("Run a command on a remote server", "ssh remote_host command -with -flags"),
             ("SSH tunneling: forward a local port to a remote one", "ssh -L 9999:localhost:8888 username@remote_host"),
             ("Connect through a jump host", "ssh -J username@jump_host username@remote_host")]),
    "tmux": ("Terminal multiplexer.\n  It allows multiple sessions with windows, panes, and more.",
             [("Start a new session", "tmux"), ("Start a new named session", "tmux new -s name"),
              ("List existing sessions", "tmux ls"), ("Attach to the most recently used session", "tmux attach"),
              ("Detach from the current session (inside a tmux session)", "<Ctrl>-B d"),
              ("Kill a session by name", "tmux kill-session -t name")]),
    "curl": ("Transfers data from or to a server.",
             [("Download the contents of a URL to a file", "curl http://example.com --output path/to/file"),
              ("Download a file, saving the output under the filename indicated by the URL", "curl --remote-name http://example.com/filename"),
              ("Download a file, following location redirects", "curl --location --remote-name http://example.com/filename"),
              ("Send JSON data to a server", "curl -X POST -H 'Content-Type: application/json' -d '{\"key\": \"value\"}' http://example.com/api")]),
    "chmod": ("Change the access permissions of a file or directory.",
              [("Give the [u]ser who owns a file the right to e[x]ecute it", "chmod u+x path/to/file"),
               ("Give the [u]ser rights to [r]ead and [w]rite to a file/directory", "chmod u+rw path/to/file_or_directory"),
               ("Remove e[x]ecutable rights from the [g]roup", "chmod g-x path/to/file"),
               ("Set permissions with octal notation", "chmod 755 path/to/file")]),
    "ls": ("List directory contents.",
           [("List files one per line", "ls -1"), ("List all files, including hidden files", "ls -a"),
            ("Long format list (permissions, ownership, size, and modification date) of all files", "ls -la"),
            ("Long format list with size displayed using human-readable units (KiB, MiB, GiB)", "ls -lh"),
            ("Long format list sorted by size (descending)", "ls -lS"), ("Long format list of all files, sorted by modification date (oldest first)", "ls -ltr")]),
    "xargs": ("Execute a command with piped arguments coming from another command, a file, etc.",
              [("Run a command using the input data as arguments", "arguments_source | xargs command"),
               ("Delete all files with a .backup extension (-0 handles names with spaces)", "find . -name '*.backup' -print0 | xargs -0 rm -v"),
               ("Execute the command once for each input line, replacing any occurrences of the placeholder", "arguments_source | xargs -I _ command _ optional_extra_arguments"),
               ("Parallel runs of up to max-procs processes at a time", "arguments_source | xargs -P max-procs command")]),
    "jq": ("A JSON processor that uses a domain-specific language (DSL).",
           [("Execute a specific expression (print a colored and formatted JSON output)", "cat path/to/file.json | jq '.'"),
            ("Output the value of a given key", "jq '.key_name' path/to/file.json"),
            ("Output elements from arrays", "jq '.[]' path/to/file.json"),
            ("Output raw strings", "jq -r '.[].name' path/to/file.json"),
            ("Filter an array", "jq 'map(select(.score > 0.9))' path/to/file.json")]),
    "nvidia-smi": ("Aid the management and monitoring of NVIDIA GPU devices.",
                   [("Display information on all available GPUs and processes using them", "nvidia-smi"),
                    ("Display more detailed GPU information", "nvidia-smi --query"),
                    ("Monitor overall GPU usage with 1-second update interval", "nvidia-smi dmon"),
                    ("Query chosen fields as CSV", "nvidia-smi --query-gpu=index,name,memory.used --format=csv")]),
}


def wrap(text, width=65, indent=7):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(" " * indent + cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(" " * indent + cur)
    return "\n".join(lines)


def man_text(name: str) -> str | None:
    key = ALIASES.get(name, name)
    page = PAGES.get(key)
    if page is None:
        return None
    sec, short, syn, desc, opts = page
    title = f"{key.upper()}({sec})"
    group = {"1": "User Commands", "8": "System Manager's Manual"}[sec]
    head = title + group.center(78 - 2 * len(title)) + title
    out = [head, "", "NAME", f"       {key} - {short}", "", "SYNOPSIS"]
    out += ["       " + line.strip() for line in syn.split("\n")]
    out += ["", "DESCRIPTION", wrap(desc), ""]
    for flags, text in opts:
        out.append("       " + flags)
        out.append(wrap(text, 58, 14))
        out.append("")
    out.append("SEE ALSO")
    out.append(f"       Full documentation <https://www.gnu.org/software/coreutils/{key}>")
    out.append("")
    out.append(f"bash-lite 2026                     October 2026" + " " * 18 + title)
    return "\n".join(out) + "\n"


def help_text(name: str) -> str | None:
    key = ALIASES.get(name, name)
    page = PAGES.get(key)
    if page is None:
        return None
    sec, short, syn, desc, opts = page
    first, *rest = syn.split("\n")
    lines = [f"Usage: {first}"] + ["  or:  " + r.strip() for r in rest]
    lines.append(desc)
    lines.append("")
    for flags, text in opts:
        if flags.startswith("-") and not flags.startswith("--"):
            f = "  " + flags
        else:
            f = "      " + flags if flags.startswith("--") else "  " + flags
        if len(f) < 30:
            lines.append(f.ljust(30) + text)
        else:
            lines.append(f)
            lines.append(" " * 30 + text)
    lines.append("      --help        display this help and exit")
    lines.append("      --version     output version information and exit")
    return "\n".join(lines) + "\n"


def whatis_line(name: str) -> str | None:
    key = ALIASES.get(name, name)
    page = PAGES.get(key)
    if page is None:
        return None
    return f"{name} ({page[0]})".ljust(21) + "- " + page[1]


def tldr_text(name: str) -> str | None:
    page = TLDR.get(name)
    if page is None:
        return None
    desc, examples = page
    out = ["", f"  {name}", "", f"  {desc}", f"  More information: <https://manned.org/{name}>.", ""]
    for title, cmd in examples:
        out.append(f"  - {title}:")
        out.append("")
        out.append(f"    {cmd}")
        out.append("")
    return "\n".join(out)

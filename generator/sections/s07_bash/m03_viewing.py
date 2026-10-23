"""7.03 Viewing & comparing files."""

from ._common import MODELS, NAMES, TEAMS, WORDS, P, catalog, challenge, csv_text, files, q, template, train_log

M = "7.03"


def _lines(rng, n, pool=WORDS):
    return [f"{rng.choice(pool)} {rng.choice(pool)} {rng.randint(1, 999)}" for _ in range(n)]


def _csv(rng, n):
    rows = [(i + 1, rng.choice(NAMES), rng.choice(TEAMS), round(rng.uniform(0.5, 0.99), 3)) for i in range(n)]
    return csv_text(("id", "name", "team", "score"), rows)


# ================================================================= lite

@template(M, "lite")
def view(rng):
    n = rng.randint(14, 30)
    f = rng.choice(["notes.txt", "words.txt", "list.txt", "todo.txt", "lines.txt"])
    data = rng.choice(["data.csv", "scores.csv", "people.csv"])
    setup = files({f: "\n".join(_lines(rng, n)) + "\n", data: _csv(rng, rng.randint(5, 12))})
    k = rng.randint(3, 8)
    c = rng.randint(5, 40)
    items = [
        ("cat", f"Print `{f}`.", f"cat {f}"),
        ("cat -n", f"Print `{f}` with line numbers (`cat`).", f"cat -n {f}"),
        ("nl", f"Number the lines of `{f}` with `nl`.", f"nl {f}"),
        ("tac", f"Print `{f}` last line first.", f"tac {f}"),
        ("head", f"Print the first 10 lines of `{f}`.", f"head {f}"),
        ("head -n", f"Print the first {k} lines of `{f}`.", f"head -n {k} {f}"),
        ("head -c", f"Print the first {c} bytes of `{f}`.", f"head -c {c} {f}"),
        ("tail -n", f"Print the last {k} lines of `{f}`.", f"tail -n {k} {f}"),
        ("tail -n", f"Print `{data}` without its header line.", f"tail -n +2 {data}"),
        ("less", f"Open `{f}` in the pager `less` (here it prints the whole file).", f"less {f}"),
        ("column -t", f"Show `{data}` as aligned columns (comma-separated input).", f"column -s, -t {data}",
         {"require": ["column -t", "column -s"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def compare(rng):
    base = _lines(rng, rng.randint(5, 9))
    other = list(base)
    i = rng.randrange(len(other))
    other[i] = other[i].replace(other[i].split()[0], rng.choice(WORDS).upper(), 1)
    if rng.random() < 0.7:
        other.insert(rng.randrange(len(other) + 1), f"{rng.choice(WORDS)} new {rng.randint(1, 99)}")
    a, b = rng.sample(["old.txt", "new.txt", "v1.txt", "v2.txt", "before.txt", "after.txt"], 2)
    setup = files({a: "\n".join(base) + "\n", b: "\n".join(other) + "\n"})
    items = [
        ("diff", f"Show how `{b}` differs from `{a}` (plain `diff`).", f"diff {a} {b}"),
        ("diff -u", f"Show the differences between `{a}` and `{b}` in unified format.", f"diff -u {a} {b}"),
        ("cmp", f"Find the first byte where `{a}` and `{b}` differ, with `cmp`.", f"cmp {a} {b}"),
        ("md5sum", f"Print the MD5 checksums of `{a}` and `{b}`.", f"md5sum {a} {b}"),
        ("sha256sum", f"Print the SHA-256 checksum of `{a}`.", f"sha256sum {a}"),
        ("comm", f"Print the lines that are in both `{a}` and `{b}` (sort them on the fly with process "
                 f"substitution and use `comm`).", f"comm -12 <(sort {a}) <(sort {b})",
         {"require": ["comm", "<("]}),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def slices(rng):
    n = rng.randint(30, 60)
    f = rng.choice(["big.log", "events.txt", "output.txt", "train.log"])
    setup = files({f: "".join(f"line {i}: {rng.choice(WORDS)}\n" for i in range(1, n + 1))})
    a = rng.randint(5, n // 2)
    b = a + rng.randint(3, 10)
    k = rng.randint(2, 6)
    kind = rng.choice(["range", "skip", "last", "range-numbered", "single"])
    if kind == "range":
        return P(f"Print lines {a} to {b} of `{f}` using `head` and `tail`.", setup, f"head -n {b} {f} | tail -n +{a}",
                 require=["head", "tail"])
    if kind == "skip":
        return P(f"Print `{f}` without its first {k} lines.", setup, f"tail -n +{k + 1} {f}", require=["tail"])
    if kind == "last":
        return P(f"Print only the last line of `{f}`.", setup, f"tail -n 1 {f}", require=["tail"])
    if kind == "range-numbered":
        return P(f"Print lines {a} to {b} of `{f}` with their line numbers (`cat -n`), using head and tail.", setup,
                 f"cat -n {f} | head -n {b} | tail -n +{a}", require=["cat -n", "head|tail"])
    return P(f"Print line {a} of `{f}` (just that line).", setup, f"head -n {a} {f} | tail -n 1", require=["head", "tail"])


@template(M, "easy")
def follow_and_inspect(rng):
    kind = rng.choice(["follow", "follow-F", "xxd", "strings", "less-S"])
    if kind in ("follow", "follow-F"):
        epochs = rng.randint(3, 6)
        gap = rng.randint(1, 2)
        writer = (f"(for e in $(seq 1 {epochs}); do sleep {gap}; echo \"epoch $e loss=0.$((90 - e * 7))\" "
                  f">> train.log; done) & disown")
        flag = "-f" if kind == "follow" else "-F"
        return P(f"Training appends a line to `train.log` every {gap} s. Follow the log as it grows with "
                 f"`tail {flag}` (it runs until you press Ctrl-C; here that happens after 10 s).",
                 "#: ctrlc 10\n: > train.log\n" + writer, f"tail {flag} train.log", require=[f"tail {flag}"])
    if kind == "xxd":
        data = "".join(rng.choice("ABCDEFGHIJKLMNOP") for _ in range(rng.randint(20, 40)))
        return P("Show a hex dump of the first two lines of `xxd` output for `blob.bin`.",
                 f"printf '%s' {q(data)} > blob.bin", "xxd blob.bin | head -2", require=["xxd", "head"])
    if kind == "strings":
        ws = rng.sample(WORDS, 3)
        blob = "\\x00\\x01" + "\\x00\\x02\\x03".join(f"{w}_{rng.randint(10, 99)}" for w in ws) + "\\x00"
        return P("`model.bin` is binary with a few readable labels inside. Print its printable strings (at least "
                 "4 characters long, `strings` default).", f"printf '{blob}' > model.bin", "strings model.bin",
                 require=["strings"])
    data = rng.choice(["data.csv", "scores.csv"])
    return P(f"View `{data}` as aligned columns in `less` without wrapping long lines (`-S`).",
             files({data: _csv(rng, rng.randint(4, 8))}), f"column -s, -t {data} | less -S",
             require=["column", "less -S"])


# ================================================================= medium

@template(M, "medium")
def checksums(rng):
    names = rng.sample(["model.pt", "weights.bin", "data.tar.gz", "vocab.txt", "config.json", "tokenizer.json"], 3)
    spec = {n: "".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9))) + "\n" for n in names}
    setup = files(spec) + "\nsha256sum " + " ".join(names) + " > SHA256SUMS"
    bad = rng.choice(names)
    kind = rng.choice(["verify", "tampered", "md5-list"])
    if kind == "verify":
        return P("`SHA256SUMS` lists the expected checksums of the downloaded files. Verify them all with one command.",
                 setup, "sha256sum -c SHA256SUMS", require=["sha256sum"])
    if kind == "tampered":
        return P(f"Someone modified a download after `SHA256SUMS` was written. Verify the files and print only the "
                 f"lines for files that FAILED.", setup + f"\necho tampered >> {bad}",
                 "sha256sum -c SHA256SUMS 2>/dev/null | grep FAILED", require=["sha256sum", "grep"])
    return P("Write the MD5 checksums of all three files into `MD5SUMS` (`md5sum` format), then check them with "
             "`md5sum -c`.", setup, f"md5sum {' '.join(names)} > MD5SUMS\nmd5sum -c MD5SUMS", require=["md5sum"])


@template(M, "medium")
def set_differences(rng):
    people = rng.sample(NAMES, 10)
    a = rng.sample(people, rng.randint(5, 7))
    b = rng.sample(people, rng.randint(5, 7))
    fa, fb = rng.choice([("monday.txt", "tuesday.txt"), ("team_a.txt", "team_b.txt"), ("before.txt", "after.txt")])
    setup = files({fa: "\n".join(a) + "\n", fb: "\n".join(b) + "\n"})
    kind = rng.choice(["only-a", "only-b", "both-count", "union"])
    if kind == "only-a":
        return P(f"Print the names in `{fa}` that are NOT in `{fb}` (the files aren't sorted).", setup,
                 f"comm -23 <(sort {fa}) <(sort {fb})", require=["comm"])
    if kind == "only-b":
        return P(f"Print the names that appear only in `{fb}`, not in `{fa}` (the files aren't sorted).", setup,
                 f"comm -13 <(sort {fa}) <(sort {fb})", require=["comm"])
    if kind == "both-count":
        return P(f"How many names are in both files? Print just the number.", setup,
                 f"comm -12 <(sort {fa}) <(sort {fb}) | wc -l", require=["comm", "wc"])
    return P(f"Print every name that's in either file, sorted, without duplicates.", setup,
             f"sort -u {fa} {fb}", require=["sort"])


# ================================================================= hard

@template(M, "hard")
def compare_dirs(rng):
    names = rng.sample(["a.py", "b.py", "c.md", "d.json", "e.yaml", "f.txt", "g.sh"], 5)
    old = {f"old/{n}": f"{rng.choice(WORDS)}\n{rng.choice(WORDS)}\n" for n in names}
    new = {k.replace("old/", "new/"): v for k, v in old.items()}
    changed = rng.choice(names)
    new[f"new/{changed}"] = new[f"new/{changed}"].replace("\n", f" {rng.choice(WORDS)}\n", 1)
    gone = rng.choice([n for n in names if n != changed])
    del new[f"new/{gone}"]
    new[f"new/extra_{rng.randint(1, 9)}.txt"] = "added\n"
    setup = files({**old, **new})
    kind = rng.choice(["brief", "full"])
    if kind == "brief":
        return P("Compare the directories `old/` and `new/` recursively; just report which files differ or exist on "
                 "one side only.", setup, "diff -rq old new", require=["diff -r"])
    return P("Show every difference between `old/` and `new/` recursively, as `diff -r` does.", setup,
             "diff -r old new", require=["diff -r"])


@template(M, "hard")
def first_difference(rng):
    n = rng.randint(40, 80)
    base = [f"{i},{rng.choice(NAMES)},{rng.randint(1, 100)}" for i in range(n)]
    other = list(base)
    k = rng.randrange(5, n)
    other[k] = other[k].rsplit(",", 1)[0] + f",{rng.randint(101, 200)}"
    setup = files({"export_a.csv": "\n".join(base) + "\n", "export_b.csv": "\n".join(other) + "\n"})
    kind = rng.choice(["cmp", "show"])
    if kind == "cmp":
        return P("Two exports should be identical. Print where they first differ (byte and line), with `cmp`.", setup,
                 "cmp export_a.csv export_b.csv", require=["cmp"])
    return P("Two exports should be identical but aren't. Print just the line number of the first difference.",
             setup, "cmp export_a.csv export_b.csv | awk '{print $NF}'", require=["cmp|diff"])


# ================================================================= challenge

def _snapshots(rng, n):
    old, new = {}, {}
    pool = [f"{d}/{rng.choice(WORDS)}_{i}.txt" for i, d in enumerate(rng.choices(["", "a", "a/b", "c"], k=n))]
    for p in pool:
        p = p.lstrip("/")
        body = f"{rng.choice(WORDS)} {rng.randint(1, 9)}\n"
        r = rng.random()
        if r < 0.15:
            old[f"old/{p}"] = body
        elif r < 0.3:
            new[f"new/{p}"] = body
        else:
            old[f"old/{p}"] = body
            new[f"new/{p}"] = body if rng.random() < 0.6 else body + "edit\n"
    return files({**old, **new}, mkdirs=["old", "new"])


@template(M, "challenge")
def snapshot_report(rng):
    setup = _snapshots(rng, 6)
    hidden = [_snapshots(rng, rng.randint(25, 40)) for _ in range(2)]
    sol = ("cd old && find . -type f | sort > ../old.list && cd ..\ncd new && find . -type f | sort > ../new.list && cd ..\n"
           "comm -23 old.list new.list | sed 's|^\\./|removed |'\ncomm -13 old.list new.list | sed 's|^\\./|added |'\n"
           "comm -12 old.list new.list | while read -r f; do cmp -s \"old/$f\" \"new/$f\" || echo \"changed ${f#./}\"; done")
    return challenge("Snapshot report", "`old/` and `new/` are two snapshots of a folder tree. Print `removed PATH` for "
                     "files only in old, then `added PATH` for files only in new, then `changed PATH` for files in both "
                     "whose contents differ. Paths are relative to the snapshot root; each group in sorted order.",
                     setup, sol, hidden, "up to 40 files in nested folders")


def _dupes(rng, n):
    pool = [f"{rng.choice(WORDS)} {rng.randint(1, 5)}\n" for _ in range(max(2, n // 3))]
    spec = {}
    while len(spec) < n:
        spec[f"photos/{rng.choice(WORDS)}_{rng.randint(100, 999)}.jpg"] = rng.choice(pool)
    return files(spec)


@template(M, "challenge")
def duplicate_groups(rng):
    setup = _dupes(rng, 7)
    hidden = [_dupes(rng, rng.randint(30, 50)) for _ in range(2)]
    sol = ("md5sum photos/* | sort -k1,1 -k2,2 | awk '{h[$1] = h[$1] ? h[$1] \" \" $2 : $2; n[$1]++} "
           "END {for (k in h) if (n[k] > 1) print h[k]}' | sort")
    return challenge("Duplicate photos", "Some files in `photos/` have identical contents. Print one line per group "
                     "of identical files (groups of 2 or more): the file paths in sorted order separated by spaces; "
                     "lines sorted.", setup, sol, hidden, "up to 50 files", require=["md5sum|sha256sum|cmp"])

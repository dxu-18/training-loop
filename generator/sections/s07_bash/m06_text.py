"""7.06 Text processing: cut, sort, uniq, tr, sed, awk, jq."""

import json

from ._common import DATASETS, MODELS, NAMES, TEAMS, WORDS, P, catalog, challenge, csv_text, files, q, template

M = "7.06"


def _runs_csv(rng, n, header=True):
    rows = []
    for i in range(n):
        rows.append((f"run{i + 1:02d}", rng.choice(MODELS[:8]), rng.choice(DATASETS[:5]),
                     rng.choice(["1e-3", "3e-4", "1e-4", "3e-3"]), rng.randint(5, 60), f"{rng.uniform(0.6, 0.98):.3f}"))
    return csv_text(("run", "model", "dataset", "lr", "epochs", "acc"), rows) if header else \
        "".join(",".join(map(str, r)) + "\n" for r in rows)


def _people_json(rng, n):
    return [{"name": nm, "team": rng.choice(TEAMS), "score": round(rng.uniform(0.5, 0.99), 2),
             "gpus": rng.randint(0, 8)} for nm in rng.sample(NAMES, n)]


# ================================================================= lite

@template(M, "lite")
def columns_and_sorting(rng):
    csv = _runs_csv(rng, rng.randint(6, 10))
    words = [rng.choice(WORDS[:12]) for _ in range(rng.randint(10, 16))]
    nums = [str(rng.randint(1, 500)) for _ in range(rng.randint(6, 10))]
    sizes = [f"{rng.randint(1, 900)}{rng.choice(['K', 'M', 'G'])}" for _ in range(rng.randint(5, 8))]
    text = " ".join(rng.choice(WORDS) for _ in range(6)) + "\n" + "  ".join(rng.choice(WORDS) for _ in range(4)) + "\n"
    setup = files({"runs.csv": csv, "names.txt": "\n".join(words) + "\n", "nums.txt": "\n".join(nums) + "\n",
                   "sizes.txt": "\n".join(sizes) + "\n", "notes.txt": text, "win.txt": "line one\r\nline two\r\n",
                   "dupes.txt": "\n".join(sorted(words)) + "\n"})
    c = rng.randint(4, 9)
    items = [
        ("cut -d", "Print the `run` and `dataset` columns (1st and 3rd) of `runs.csv`.", "cut -d, -f1,3 runs.csv",
         {"require": ["cut -d", "cut -f"]}),
        ("cut -c", f"Print the first {c} characters of each line of `notes.txt`.", f"cut -c1-{c} notes.txt"),
        ("sort", "Sort `names.txt` alphabetically.", "sort names.txt"),
        ("sort -n", "Sort the numbers in `nums.txt` numerically, smallest first.", "sort -n nums.txt"),
        ("sort -r", "Sort `names.txt` in reverse alphabetical order.", "sort -r names.txt"),
        ("sort -u", "Sort `names.txt` and drop duplicate lines (one command).", "sort -u names.txt"),
        ("sort -h", "Sort the human-readable sizes in `sizes.txt` (like `12K`, `3M`) from small to large.",
         "sort -h sizes.txt"),
        ("sort -t", "Sort the rows of `runs.csv` (without its header) by the `acc` column (6th), highest first.",
         "tail -n +2 runs.csv | sort -t, -k6,6nr", {"require": ["sort -t", "sort -k"]}),
        ("uniq -c", "Count how often each word occurs in `names.txt` (`sort | uniq -c`).", "sort names.txt | uniq -c"),
        ("uniq -d", "`dupes.txt` is sorted. Print each line that occurs more than once (once each).", "uniq -d dupes.txt"),
        ("tr", "Print `notes.txt` in UPPER CASE.", "tr 'a-z' 'A-Z' < notes.txt"),
        ("tr -d", "`win.txt` has Windows line endings. Strip the `\\r` characters and save the result as `unix.txt`.",
         "tr -d '\\r' < win.txt > unix.txt", {"post": "cat -A unix.txt"}),
        ("tr -s", "Squeeze runs of spaces in `notes.txt` into single spaces.", "tr -s ' ' < notes.txt"),
        ("tr", "Turn `runs.csv` into tab-separated values (print it).", "tr ',' '\\t' < runs.csv"),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def sed_basics(rng):
    old, new = rng.sample(WORDS, 2)
    lines = [" ".join(rng.choice([old, old, rng.choice(WORDS)]) for _ in range(4)) for _ in range(rng.randint(4, 7))]
    lr0, lr1 = rng.choice([("0.1", "0.01"), ("0.001", "0.0003"), ("3e-4", "1e-4")])
    bs0, bs1 = rng.choice([(32, 64), (16, 32), (64, 128)])
    cfg = f"# training config\nlr: {lr0}\nbatch_size: {bs0}\n# data\ndataset: {rng.choice(DATASETS)}\nepochs: {rng.randint(5, 90)}\n"
    numbered = "\n".join(f"line {i}: {rng.choice(WORDS)}" for i in range(1, rng.randint(25, 40))) + "\n"
    a = rng.randint(3, 8)
    b = a + rng.randint(2, 6)
    setup = files({"f.txt": "\n".join(lines) + "\n", "config.yaml": cfg, "big.txt": numbered,
                   "data.csv": _runs_csv(rng, 4), "ids.txt": " ".join(str(rng.randint(1, 999)) for _ in range(5)) + "\n"})
    items = [
        ("sed", f"Replace the first `{old}` on each line of `f.txt` with `{new}` (print the result).",
         f"sed 's/{old}/{new}/' f.txt"),
        ("sed", f"Replace every `{old}` in `f.txt` with `{new}` (print the result).", f"sed 's/{old}/{new}/g' f.txt"),
        ("sed -i", f"Change `lr: {lr0}` to `lr: {lr1}` in `config.yaml`, editing the file in place.",
         f"sed -i 's/lr: {lr0}/lr: {lr1}/' config.yaml", {"post": "cat config.yaml"}),
        ("sed -i", f"Change the batch size in `config.yaml` from {bs0} to {bs1}, in place.",
         f"sed -i 's/batch_size: {bs0}/batch_size: {bs1}/' config.yaml", {"post": "cat config.yaml"}),
        ("sed -n", f"Print lines {a} to {b} of `big.txt` with sed.", f"sed -n '{a},{b}p' big.txt"),
        ("sed", "Print `data.csv` without its first line (sed).", "sed '1d' data.csv"),
        ("sed", "Print `config.yaml` without its comment lines (lines starting with `#`).", "sed '/^#/d' config.yaml"),
        ("sed -E", "Wrap every number in `ids.txt` in angle brackets, like `<42>` (extended regex, a group and `\\1`).",
         "sed -E 's/([0-9]+)/<\\1>/g' ids.txt"),
    ]
    return catalog(rng, items, setup)


# ================================================================= easy

@template(M, "easy")
def awk_basics(rng):
    csv = _runs_csv(rng, rng.randint(6, 10))
    ws = "".join(f"{rng.choice(NAMES)} {rng.choice(TEAMS)} {rng.uniform(0.5, 0.99):.2f}\n" for _ in range(rng.randint(6, 10)))
    code = "".join(("x = " + "y" * rng.randint(70, 95) if rng.random() < 0.4 else f"z = {rng.randint(1, 9)}") + "\n"
                   for _ in range(6))
    thr = rng.choice([0.7, 0.8, 0.9])
    setup = files({"runs.csv": csv, "results.txt": ws, "f.py": code})
    items = [
        ("awk", "Print the first whitespace-separated field of every line of `results.txt`.", "awk '{print $1}' results.txt"),
        ("awk -F", "Print the `model` (2nd) and the last column of every row of `runs.csv`.",
         "awk -F, '{print $2, $NF}' runs.csv"),
        ("awk", "Print the mean of the `epochs` column (5th) of `runs.csv`, skipping the header.",
         "awk -F, 'NR > 1 {sum += $5; n++} END {print sum / n}' runs.csv"),
        ("awk", f"Print the rows of `results.txt` whose 3rd column is greater than {thr}.",
         f"awk '$3 > {thr}' results.txt"),
        ("awk", "Count how many rows each team (2nd column) has in `results.txt`; print `TEAM COUNT` lines sorted by team.",
         "awk '{count[$2]++} END {for (k in count) print k, count[k]}' results.txt | sort"),
        ("awk -v", "Print the first two columns of `results.txt` separated by a tab (set `OFS` with `-v`).",
         "awk -v OFS='\\t' '{print $1, $2}' results.txt"),
        ("awk", "Print the lines of `f.py` longer than 80 characters.", "awk 'length($0) > 80' f.py"),
        ("awk", "Print `results.txt` as a table: the name left-aligned in 10 characters, then the score with 3 decimals.",
         "awk '{printf \"%-10s %.3f\\n\", $1, $3}' results.txt"),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def jq_basics(rng):
    people = _people_json(rng, rng.randint(4, 6))
    thr = rng.choice([0.7, 0.8, 0.9])
    doc = {"project": rng.choice(WORDS), "items": people}
    rows = "".join(json.dumps(p) + "\n" for p in people[:3])
    setup = files({"data.json": json.dumps(doc), "people.json": json.dumps(people), "rows.jsonl": rows})
    items = [
        ("jq", "Pretty-print `data.json`.", "jq '.' data.json"),
        ("jq", "From `data.json`, print an object with just the `name` and `score` of each item.",
         "jq '.items[] | {name, score}' data.json"),
        ("jq -r", "Print every `name` in `people.json` as raw text (no quotes), one per line.", "jq -r '.[].name' people.json"),
        ("jq", f"Keep only the people in `people.json` with a score above {thr} (print the filtered array).",
         f"jq 'map(select(.score > {thr}))' people.json"),
        ("jq -s", "`rows.jsonl` has one JSON object per line. Slurp them into a single array.", "jq -s '.' rows.jsonl"),
        ("jq -c", "Print each person from `people.json` as compact JSON, one per line.", "jq -c '.[]' people.json"),
        ("jq", "How many people are in `people.json`? Print the number with jq.", "jq 'length' people.json"),
    ]
    return catalog(rng, items, setup)


# ================================================================= medium

@template(M, "medium")
def counting_pipelines(rng):
    csv = _runs_csv(rng, rng.randint(10, 16))
    log = "".join(f"{rng.choice(['GET', 'POST', 'GET', 'GET', 'PUT'])} /{rng.choice(['api', 'predict', 'health', 'docs'])} "
                  f"{rng.choice([200, 200, 200, 404, 500])}\n" for _ in range(rng.randint(15, 25)))
    k = rng.randint(2, 3)
    setup = files({"runs.csv": csv, "access.log": log})
    kind = rng.choice(["models", "datasets", "status", "paths", "unique"])
    if kind == "models":
        return P("Count the rows per model (2nd column) in `runs.csv` (skip the header), most frequent first, "
                 "with the classic `sort | uniq -c | sort -rn` pipeline.", setup,
                 "tail -n +2 runs.csv | cut -d, -f2 | sort | uniq -c | sort -rn", require=["uniq -c", "sort -n|sort -rn"])
    if kind == "datasets":
        return P(f"Print the {k} most common datasets (3rd column) in `runs.csv` with their counts, most frequent "
                 f"first.", setup, f"tail -n +2 runs.csv | cut -d, -f3 | sort | uniq -c | sort -rn | head -{k}",
                 require=["uniq -c", "head"])
    if kind == "status":
        return P("Count the requests per HTTP status code (3rd field) in `access.log`, most frequent first.", setup,
                 "awk '{print $3}' access.log | sort | uniq -c | sort -rn", require=["uniq -c"])
    if kind == "paths":
        return P("Which path (2nd field) in `access.log` was requested most often? Print just the path.", setup,
                 "awk '{print $2}' access.log | sort | uniq -c | sort -rn | head -1 | awk '{print $2}'",
                 require=["uniq -c"])
    return P("How many different models appear in `runs.csv`? Print just the number.", setup,
             "tail -n +2 runs.csv | cut -d, -f2 | sort -u | wc -l", require=["sort -u|uniq", "wc"])


@template(M, "medium")
def combine_files(rng):
    n = rng.randint(4, 7)
    ids = rng.sample(range(100, 999), n)
    names = rng.sample(NAMES, n)
    scores = [rng.randint(50, 100) for _ in range(n)]
    setup = files({"names.csv": "".join(f"{i},{nm}\n" for i, nm in sorted(zip(ids, names))),
                   "scores.csv": "".join(f"{i},{s}\n" for i, s in sorted(zip(ids, scores))),
                   "a.txt": "\n".join(names) + "\n", "b.txt": "\n".join(map(str, scores)) + "\n",
                   "nums.txt": "\n".join(map(str, scores)) + "\n",
                   "big.csv": "".join(f"{i},{rng.choice(WORDS)}\n" for i in range(rng.randint(20, 35)))})
    k = rng.choice([5, 8, 10])
    items = [
        ("join -t", "Join `names.csv` and `scores.csv` on their first column (both are sorted by id).",
         "join -t, names.csv scores.csv"),
        ("paste -d", "Put `a.txt` and `b.txt` side by side, separated by a comma.", "paste -d, a.txt b.txt"),
        ("paste -s", "Add up the numbers in `nums.txt` by joining them with `+` (`paste -sd+`) and feeding that to `bc`.",
         "paste -sd+ nums.txt | bc", {"require": ["paste -s", "bc"]}),
        ("split -l", f"Split `big.csv` into chunks of {k} lines named `chunk_aa`, `chunk_ab`, ...",
         f"split -l {k} big.csv chunk_", {"post": "wc -l chunk_*"}),
        ("seq", f"Print `run 1` to `run {n}`, one per line, using `seq` and `xargs -I{{}}`.",
         f'seq 1 {n} | xargs -I{{}} echo "run {{}}"', {"require": ["seq", "xargs -I{}"]}),
        ("xargs -P", "`urls.txt` lists files to download. Fetch them all quietly with `wget`, one URL per call and 4 "
                     "downloads at a time.", "cat urls.txt | xargs -n 1 -P 4 wget -q",
         {"require": ["xargs -n", "xargs -P", "wget"], "post": "ls *.csv; cat *.csv",
          "setup": "\n".join(f"#: url https://data.example.com/{d}.csv {d},{rng.randint(1, 99)}" for d in DATASETS[:n])
                   + "\n" + files({"urls.txt": "".join(f"https://data.example.com/{d}.csv\n" for d in DATASETS[:n])})}),
    ]
    return catalog(rng, items, setup)


# ================================================================= hard

@template(M, "hard")
def csv_report(rng):
    csv = _runs_csv(rng, rng.randint(10, 16))
    setup = files({"runs.csv": csv})
    kind = rng.choice(["mean-per-model", "best-per-dataset", "top-k"])
    if kind == "mean-per-model":
        return P("For each model in `runs.csv`, print the model and its mean `acc` (6th column) with 3 decimals, "
                 "sorted by model name.", setup,
                 "awk -F, 'NR > 1 {s[$2] += $6; n[$2]++} END {for (m in s) printf \"%s %.3f\\n\", m, s[m] / n[m]}' "
                 "runs.csv | sort", require=["awk"])
    if kind == "best-per-dataset":
        return P("For each dataset, print the dataset and the run with the highest `acc`, sorted by dataset "
                 "(ties: the first one in the file).", setup,
                 "awk -F, 'NR > 1 && (!($3 in b) || $6 > b[$3]) {b[$3] = $6; r[$3] = $1} END {for (d in b) print d, r[d]}' "
                 "runs.csv | sort", require=["awk"])
    k = rng.randint(2, 4)
    return P(f"Print the `run`, `model` and `acc` of the {k} best runs, best first, as an aligned table "
             f"(`column -t`).", setup,
             f"tail -n +2 runs.csv | sort -t, -k6,6nr | head -{k} | cut -d, -f1,2,6 | tr ',' ' ' | column -t",
             require=["sort", "head"])


@template(M, "hard")
def config_surgery(rng):
    keys = {"lr": rng.choice(["0.1", "0.001", "3e-4"]), "batch_size": rng.choice([16, 32, 64]),
            "epochs": rng.randint(5, 50), "dropout": rng.choice(["0.1", "0.2", "0.5"]), "seed": rng.randint(0, 9)}
    cfg = "".join(f"{k}: {v}\n" for k, v in keys.items())
    configs = {f"configs/{m}.yaml": cfg.replace(f"seed: {keys['seed']}", f"seed: {i}") for i, m in enumerate(rng.sample(MODELS, 3))}
    setup = files(configs)
    kind = rng.choice(["all-files", "comment", "jq-edit"])
    new_bs = keys["batch_size"] * 2
    if kind == "all-files":
        return P(f"Double the batch size in every YAML file in `configs/` (from {keys['batch_size']} to {new_bs}), in "
                 f"place, with one `sed` command.", setup, f"sed -i 's/^batch_size: {keys['batch_size']}$/batch_size: {new_bs}/' configs/*.yaml",
                 post="grep -H batch_size configs/*.yaml", require=["sed -i"])
    if kind == "comment":
        return P("Comment out the `dropout` line (put `# ` in front of it) in every YAML file in `configs/`, in place.",
                 setup, "sed -i 's/^dropout:/# dropout:/' configs/*.yaml", post="cat configs/*.yaml", require=["sed -i"])
    people = _people_json(rng, 4)
    return P("In `people.json`, give everyone on the `infra` team 2 more GPUs and write the result to `out.json` "
             "(compact, one line).", files({"people.json": json.dumps(people)}),
             "jq -c 'map(if .team == \"infra\" then .gpus += 2 else . end)' people.json > out.json",
             post="cat out.json", require=["jq"])


# ================================================================= challenge

def _corpus(rng, n_lines):
    vocab = rng.sample(WORDS, 14)
    out = []
    for _ in range(n_lines):
        ws = [rng.choice(vocab[:rng.randint(4, 14)]) for _ in range(rng.randint(3, 10))]
        ws = [w.capitalize() if rng.random() < 0.2 else w for w in ws]
        out.append(" ".join(ws) + rng.choice([".", ",", "!", "", "?"]))
    return files({"book.txt": "\n".join(out) + "\n"})


@template(M, "challenge")
def word_frequencies(rng):
    k = rng.randint(4, 6)
    setup = _corpus(rng, 5)
    hidden = [_corpus(rng, rng.randint(80, 150)) for _ in range(2)]
    sol = f"tr -cs 'A-Za-z' '\\n' < book.txt | tr 'A-Z' 'a-z' | grep -v '^$' | sort | uniq -c | sort -k1,1nr -k2,2 | head -{k}"
    return challenge("Word frequencies", f"Print the {k} most frequent words in `book.txt` as `uniq -c` prints them "
                     f"(count, then word). Words are runs of letters, compared case-insensitively (print them in lower "
                     f"case); ties broken alphabetically.", setup, sol, hidden, "up to 150 lines")


def _scores(rng, n):
    rows = [{"user": rng.choice(NAMES[:10]), "task": rng.choice(["mmlu", "gsm8k", "humaneval", "arc"]),
             "score": rng.randint(0, 100)} for _ in range(n)]
    return files({"scores.json": json.dumps(rows)})


@template(M, "challenge")
def leaderboard(rng):
    setup = _scores(rng, 8)
    hidden = [_scores(rng, rng.randint(60, 120)) for _ in range(2)]
    sol = ("jq -r 'group_by(.user) | map({user: .[0].user, best: (group_by(.task) | map(max_by(.score).score) | add)}) "
           "| sort_by(-.best, .user) | .[] | \"\\(.user) \\(.best)\"' scores.json")
    return challenge("Leaderboard", "`scores.json` is an array of submissions `{user, task, score}`; users may submit "
                     "a task several times. A user's total is the sum over tasks of their BEST score on that task. "
                     "Print `USER TOTAL` lines, highest total first, ties alphabetically by user.", setup, sol, hidden,
                     "up to 120 submissions", require=["jq"])

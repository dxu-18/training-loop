"""7.09 Arrays & arithmetic."""

from ._common import DATASETS, MODELS, NAMES, WORDS, P, catalog, challenge, files, q, template

M = "7.09"


def _arr(rng, n=None):
    return rng.sample(WORDS, n or rng.randint(4, 6))


# ================================================================= lite

@template(M, "lite")
def indexed_arrays(rng):
    xs = _arr(rng)
    extra = rng.choice([w for w in WORDS if w not in xs])
    i = rng.randint(1, len(xs) - 1)
    a = rng.randint(1, len(xs) - 2)
    setup = f"arr=({' '.join(xs)})"
    items = [
        ("arr=(", f"Create an array `arr` holding `{' '.join(xs)}` and print all its elements (`\"${{arr[@]}}\"`).",
         f'arr=({" ".join(xs)})\necho "${{arr[@]}}"', {"setup": "", "require": ["arr=(", "array[@]"]}),
        ("+=", f"Append `{extra}` to the array `arr`, then print all the elements.", f'arr+=({extra})\necho "${{arr[@]}}"'),
        ("array", "Print the first element of `arr`.", 'echo "${arr[0]}"'),
        ("array", "Print the last element of `arr` (negative index).", 'echo "${arr[-1]}"'),
        ("array", f"Print element number {i} of `arr` (indices start at 0).", f'echo "${{arr[{i}]}}"'),
        ("param:len", "Print how many elements `arr` has.", 'echo "${#arr[@]}"'),
        ("param:!", "Print the indices of `arr` (`${!arr[@]}`).", 'echo "${!arr[@]}"'),
        ("param:sub", f"Print 2 elements of `arr` starting at index {a} (array slice).", f'echo "${{arr[@]:{a}:2}}"'),
        ("for", "Print each element of `arr` on its own line with a `for` loop over `\"${arr[@]}\"`.",
         'for x in "${arr[@]}"; do echo "$x"; done', {"require": ["for", "array[@]"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "lite")
def arithmetic(rng):
    a, b, c = rng.randint(2, 20), rng.randint(2, 9), rng.randint(2, 9)
    y = rng.randint(3, 9)
    x = rng.randint(3, 11) * y + rng.randint(1, y - 1)      # never divisible: the remainder isn't 0
    p = rng.randint(5, 12)
    start, inc = rng.randint(0, 5), rng.randint(2, 9)
    items = [
        ("$((", f"Print {a} + {b} * {c} with arithmetic expansion.", f"echo $(({a} + {b} * {c}))"),
        ("$((", f"Print the integer quotient of {x} divided by {y} (bash arithmetic truncates).", f"echo $(({x} / {y}))"),
        ("$((", f"Print the remainder of {x} divided by {y}.", f"echo $(({x} % {y}))"),
        ("$((", f"Print 2 to the power {p}.", f"echo $((2 ** {p}))"),
        ("((", f"Set `i={start}`, increment it with `((i++))`, add {inc} with `((i += {inc}))`, then print it.",
         f"i={start}\n((i++))\n((i += {inc}))\necho \"$i\"", {"require": ["(("]}),
        ("bc", f"Print {x} / {y} with 4 decimals using `bc` (`scale=4`).", f'echo "scale=4; {x}/{y}" | bc'),
        ("awk", f"Print {x} / {y} as a float with `awk` (in a `BEGIN` block).", f"awk 'BEGIN {{print {x}/{y}}}'"),
        ("python3 -c", f"Print {x} / {y} with Python (`python3 -c`).", f'python3 -c "print({x}/{y})"'),
    ]
    return catalog(rng, items)


# ================================================================= easy

@template(M, "easy")
def assoc_arrays(rng):
    sizes = {m: rng.choice([8, 16, 32, 64, 128]) for m in rng.sample(MODELS, rng.randint(3, 4))}
    ks = list(sizes)
    body = " ".join(f"[{k}]={v}" for k, v in sizes.items())
    setup = f"declare -A bs=({body})"
    k = rng.choice(ks)
    items = [
        ("declare -A", f"Create an associative array `bs` mapping {', '.join(f'`{k}` to {v}' for k, v in sizes.items())} "
                       f"and print the batch size of `{k}`.", f'declare -A bs=({body})\necho "${{bs[{k}]}}"',
         {"setup": ""}),
        ("array", f"Print the batch size stored for `{k}` in the associative array `bs`.", f'echo "${{bs[{k}]}}"'),
        ("param:!", "Print every `MODEL=BATCH` pair of `bs`, one per line, sorted by model name.",
         'for k in "${!bs[@]}"; do echo "$k=${bs[$k]}"; done | sort', {"require": ["param:!", "for"]}),
        ("param:len", "Print how many models `bs` has.", 'echo "${#bs[@]}"'),
        ("array", f"Add a model `tiny` with batch size 256 to `bs`, then print all batch sizes sorted numerically, one "
                  f"per line.", 'bs[tiny]=256\nprintf "%s\\n" "${bs[@]}" | sort -n', {"require": ["array"]}),
    ]
    return catalog(rng, items, setup)


@template(M, "easy")
def mapfile_lines(rng):
    lines = [f"{rng.choice(NAMES)} {rng.randint(1, 99)}" for _ in range(rng.randint(4, 8))]
    setup = files({"people.txt": "\n".join(lines) + "\n"})
    kind = rng.choice(["count", "last", "reverse", "index"])
    if kind == "count":
        return P("Read the lines of `people.txt` into an array with `mapfile -t`, then print how many there are.",
                 setup, 'mapfile -t lines < people.txt\necho "${#lines[@]}"', require=["mapfile -t"])
    if kind == "last":
        return P("Read `people.txt` into an array (`mapfile -t`) and print its last line.", setup,
                 'mapfile -t lines < people.txt\necho "${lines[-1]}"', require=["mapfile"])
    if kind == "reverse":
        return P("Read `people.txt` into an array and print the lines in reverse order using the array (a C-style "
                 "or index loop).", setup,
                 'mapfile -t lines < people.txt\nfor ((i = ${#lines[@]} - 1; i >= 0; i--)); do echo "${lines[i]}"; done',
                 require=["mapfile", "for((|while"])
    k = rng.randint(1, len(lines) - 1)
    return P(f"Read `people.txt` into an array and print line number {k + 1} (counting from 1) via the array.", setup,
             f'mapfile -t lines < people.txt\necho "${{lines[{k}]}}"', require=["mapfile"])


# ================================================================= medium

@template(M, "medium")
def sweep_lists(rng):
    lrs = rng.sample(["1e-2", "3e-3", "1e-3", "3e-4", "1e-4"], rng.randint(2, 3))
    seeds = rng.sample(range(0, 9), rng.randint(2, 3))
    kind = rng.choice(["grid", "count", "zip"])
    if kind == "grid":
        return P(f"Put the learning rates `{' '.join(lrs)}` in an array `lrs` and the seeds `{' '.join(map(str, seeds))}` "
                 f"in `seeds`. Print `lr=LR seed=SEED` for every combination (learning rate in the outer loop).", "",
                 f"lrs=({' '.join(lrs)})\nseeds=({' '.join(map(str, seeds))})\n"
                 'for lr in "${lrs[@]}"; do for s in "${seeds[@]}"; do echo "lr=$lr seed=$s"; done; done',
                 require=["arr=(", "array[@]", "for"])
    if kind == "count":
        return P(f"With `lrs=({' '.join(lrs)})` and `seeds=({' '.join(map(str, seeds))})`, print how many runs a full "
                 f"grid needs, using array lengths and arithmetic.", f"lrs=({' '.join(lrs)})\nseeds=({' '.join(map(str, seeds))})",
                 'echo $(( ${#lrs[@]} * ${#seeds[@]} ))', require=["$((", "param:len"])
    ms = rng.sample(MODELS, len(lrs))
    return P(f"`models=({' '.join(ms)})` and `lrs=({' '.join(lrs)})` line up by position. Print `MODEL LR` pairs, one "
             f"per line (loop over the indices `${{!models[@]}}`).", f"models=({' '.join(ms)})\nlrs=({' '.join(lrs)})",
             'for i in "${!models[@]}"; do echo "${models[i]} ${lrs[i]}"; done', require=["param:!"])


@template(M, "medium")
def float_math(rng):
    nums = [round(rng.uniform(0.1, 9.9), 2) for _ in range(rng.randint(3, 6))]
    a, b = rng.randint(1, 99), rng.randint(3, 13)
    kind = rng.choice(["mean-awk", "ratio-bc", "sum-bc", "sqrt"])
    setup = files({"vals.txt": "\n".join(map(str, nums)) + "\n"})
    if kind == "mean-awk":
        return P("Print the mean of the numbers in `vals.txt` with awk.", setup,
                 "awk '{s += $1} END {print s / NR}' vals.txt", require=["awk"])
    if kind == "ratio-bc":
        return P(f"Print {a}/{b} to 3 decimals with `bc`.", "", f'echo "scale=3; {a}/{b}" | bc', require=["bc"])
    if kind == "sum-bc":
        return P("Add up the numbers in `vals.txt` with `paste -sd+` and `bc`.", setup, "paste -sd+ vals.txt | bc",
                 require=["bc"])
    return P(f"Print the square root of {a} with `bc -l`, to 4 decimals (`scale=4; sqrt(...)`).", "",
             f'echo "scale=4; sqrt({a})" | bc -l', require=["bc"])


# ================================================================= hard

@template(M, "hard")
def counters(rng):
    words = [rng.choice(WORDS[:8]) for _ in range(rng.randint(10, 18))]
    setup = files({"words.txt": "\n".join(words) + "\n"})
    kind = rng.choice(["assoc-count", "max"])
    if kind == "assoc-count":
        return P("Count how often each word of `words.txt` occurs using a bash associative array (no `uniq`/`awk`), "
                 "then print `WORD COUNT` lines sorted by word.", setup,
                 'declare -A c\nwhile read -r w; do ((c[$w]++)); done < words.txt\n'
                 'for w in "${!c[@]}"; do echo "$w ${c[$w]}"; done | sort', require=["declare -A", "(("])
    nums = [rng.randint(1, 999) for _ in range(rng.randint(5, 9))]
    return P(f"`nums=({' '.join(map(str, nums))})`. Find the largest number with a loop and arithmetic (no `sort`), "
             f"and print it.", f"nums=({' '.join(map(str, nums))})",
             'max=${nums[0]}\nfor n in "${nums[@]}"; do ((n > max)) && max=$n; done\necho "$max"', require=["(("])


@template(M, "hard")
def gpu_assign(rng):
    n = rng.randint(4, 8)
    jobs = [f"{rng.choice(MODELS)}-{i}" for i in range(n)]
    g = rng.randint(2, 4)
    return P(f"`jobs=({' '.join(jobs)})` must be spread round-robin over {g} GPUs (job 0 on GPU 0, job 1 on GPU 1, "
             f"..., wrapping around). Print `JOB -> GPU N` for each job, using the index and `%`.",
             f"jobs=({' '.join(jobs)})", f'for i in "${{!jobs[@]}}"; do echo "${{jobs[i]}} -> GPU $((i % {g}))"; done',
             require=["$((", "array"])


# ================================================================= challenge

def _matrix(rng, r, c):
    return files({"grid.txt": "\n".join(" ".join(str(rng.randint(0, 9)) for _ in range(c)) for _ in range(r)) + "\n"})


@template(M, "challenge")
def column_sums(rng):
    setup = _matrix(rng, 3, 4)
    hidden = [_matrix(rng, rng.randint(20, 40), rng.randint(5, 12)) for _ in range(2)]
    sol = ('declare -a sum\nwhile read -ra row; do\n  for i in "${!row[@]}"; do ((sum[i] += row[i])); done\n'
           'done < grid.txt\necho "${sum[@]}"\nmax=0; best=0\nfor i in "${!sum[@]}"; do ((sum[i] > max)) && max=${sum[i]} && best=$i; done\n'
           'echo "column $best"')
    return challenge("Column sums", "`grid.txt` is a matrix of digits separated by spaces. Print the sum of each "
                     "column on one line (space-separated), then `column K` for the (0-based) column with the largest "
                     "sum (the first one on ties). Use bash arrays and arithmetic.", setup, sol, hidden,
                     "up to 40 rows × 12 columns")


def _ports(rng, n):
    return files({"requests.txt": "\n".join(f"{rng.choice(['alice', 'bob', 'carol', 'dave', 'erin'])} "
                                            f"{rng.choice([8888, 6006, 8000, 5000, 8080])}" for _ in range(n)) + "\n"})


@template(M, "challenge")
def port_allocator(rng):
    base = rng.choice([9000, 10000, 20000])
    setup = _ports(rng, 5)
    hidden = [_ports(rng, rng.randint(30, 60)) for _ in range(2)]
    sol = (f'declare -A got\nnext={base}\nwhile read -r user port; do\n  key="$user:$port"\n'
           '  if [[ -z ${got[$key]} ]]; then got[$key]=$next; ((next++)); fi\n  echo "$user $port -> ${got[$key]}"\n'
           'done < requests.txt')
    return challenge("Tunnel ports", f"Each line of `requests.txt` is `USER PORT`: someone wants a tunnel to a remote "
                     f"port. Give every distinct (user, port) pair its own local port, counting up from {base} in order "
                     f"of first appearance, and print `USER PORT -> LOCAL` for every line (repeats reuse their port).",
                     setup, sol, hidden, "up to 60 requests", require=["declare -A"])

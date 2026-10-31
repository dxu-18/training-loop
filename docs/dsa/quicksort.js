// 13.03 Quicksort: partition A[lo, hi) around a pivot, then sort both sides recursively (a partition function plus
// quicksort), in Python, C++, Java or Rust. The pivot option shows why its choice matters.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const BS = ((root.DSATutorials || {})["binary-search"] || (typeof require !== "undefined" ? require("./binary-search.js") : null))._internals;
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type that can be ordered: a type parameter T (Comparable in Java, PartialOrd in Rust)." } },
    { id: "pivot", label: "Pivot", choices: [["middle", "middle element"], ["first", "first element"]],
      help: { middle: "Partition around A[lo + (hi - lo) / 2]. Sorted input splits evenly.",
        first: "Partition around A[lo]. Simple, but already-sorted input puts everything on one side every time: O(n²)." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", pivot: "middle" };
  const first = (o) => o.pivot === "first";
  const describe = (o) => `quicksort of ${o.type === "int" ? "int" : "generic"} elements with the ${first(o) ? "first" : "middle"} element as pivot, in ${DL.LANGS[o.lang].name}`;

  function complexity(o) {
    return [
      { id: "expected", label: "Time, typical input", big: "O(n log n)", note: "the pivot usually splits the segment into reasonable parts" },
      { id: "worst", label: "Worst case", big: "O(n²)", note: first(o) ? "already-sorted input: every partition is lopsided" : "rare inputs built against the middle pivot (a random pivot makes them unlikely)" },
      { id: "sorted", label: "Already sorted input", big: first(o) ? "O(n²)" : "O(n log n)", note: first(o) ? "the first element is the smallest every time" : "the middle element splits it evenly" },
      { id: "partition", label: "Partition", big: "O(n)", note: "one pass over the segment" },
      { id: "space", label: "Extra space", big: "O(log n)", note: "in place; the recursion stack (O(n) in the worst case)" },
      { id: "stable", label: "Stable", big: "no", note: "swaps can reorder equal elements", extra: true },
      { id: "merge", label: "Mergesort, for comparison", big: "O(n log n)", note: "every case, stable, but O(n) extra space", extra: true },
      { id: "is_sorted", label: "Contract is_sorted", big: "O(n)", note: "one pass over the segment" },
    ];
  }

  function diagram(o) {
    const v = o.type === "int" ? ["20", "10", "30", "40", "70", "50", "60"] : ["bee", "ant", "cat", "dog", "gnu", "eel", "fox"];
    const tag = { 0: "A[lo, mid)", 3: "mid", 4: "A[mid + 1, hi)" };
    return {
      rows: [v.map((x, i) => ({ box: x, tags: tag[i] ? [tag[i]] : [], cls: i === 3 ? "pivot" : "" }))],
      caption: "partition moves the pivot to A[mid], with every element at most the pivot before it and every bigger one after it. The pivot is then in its final place, and quicksort sorts the two sides.",
    };
  }

  // ---------------------------------------------------------------- specs
  const O = "Quicksort";
  const seg = [["lo", "int"], ["hi", "int"]];
  const FNS = {
    is_sorted: { params: [["A", "Arr"], ...seg], ret: "bool", owner: O },
    all_at_most: { params: [["A", "Arr"], ...seg, ["x", "Tref"]], ret: "bool", owner: O },
    all_bigger: { params: [["A", "Arr"], ...seg, ["x", "Tref"]], ret: "bool", owner: O },
    swap: { params: [["A", "ArrMut"], ["i", "int"], ["j", "int"]], ret: "void", owner: O },
    partition: { params: [["A", "ArrMut"], ["lo", "int"], ["pi", "int"], ["hi", "int"]], ret: "int", owner: O, bounds: ["PartialOrd"] },
    quicksort: { params: [["A", "ArrMut"], ...seg], ret: "void", owner: O, bounds: ["PartialOrd"] },
    print_arr: { params: [["A", "Arr"], ...seg], ret: "void", harness: true },
    print_sorted: { params: [["A", "Arr"], ...seg], ret: "void", harness: true },
  };

  // ---------------------------------------------------------------- the functions
  const allAtMost = BS.all(">", "Every element of `A[lo, hi)` must be at most `x`.");
  const allBigger = BS.all("<=", "Every element of `A[lo, hi)` must be bigger than `x`.");
  const swap = () => [
    { say: "Python swaps with one tuple assignment: both values on the right are read before either is written.", code: { py: ["A[i], A[j] = A[j], A[i]"] }, only: ["py"] },
    { say: "Rust can't move an element out of the slice into a temporary, so use the slice's own `swap`.", code: { rs: ["A.swap(i, j);"] }, only: ["rs"] },
    { ...S("Save `A[i]`…", "let t = A[i]"), only: ["cpp", "java"] },
    { ...S("…overwrite it with `A[j]`…", "A[i] = A[j]"), only: ["cpp", "java"] },
    { ...S("…and put the saved value in `A[j]`.", "A[j] = t"), only: ["cpp", "java"] },
  ];
  const partition = () => [
    S("**Requires**: the pivot's index `pi` is in the segment.", "assert lo <= pi and pi < hi"),
    S("Move the pivot to the front, `A[lo]`, out of the way.", "swap(A, lo, pi)"),
    S("`A[lo + 1, left)` will hold elements at most the pivot (empty so far)…", "let left = lo + 1"),
    S("…and `A[right, hi)` elements bigger than it. `A[left, right)` is still unknown.", "let right = hi"),
    S("Until nothing unknown is left…", "while left < right", [
      S("**Loop invariant**: the left side is at most the pivot…", "assert all_at_most(A, lo + 1, left, A[lo])"),
      S("…and the right side is bigger.", "assert all_bigger(A, right, hi, A[lo])"),
      S("The next unknown element is at most the pivot: it's already on the left side.", "if A[left] <= A[lo]", [S("", "left += 1")]),
      S("Otherwise swap it to the end of the unknown part, where it becomes the right side's first element.", "else", [S("", "swap(A, left, right - 1)"), S("", "right -= 1")]),
    ]),
    S("`left == right`. Put the pivot between the two sides: swap it with the last element of the left side.", "swap(A, lo, left - 1)"),
    S("**Ensures**: everything before the pivot is at most it, everything after is bigger.", "assert all_at_most(A, lo, left - 1, A[left - 1]) and all_bigger(A, left, hi, A[left - 1])"),
    S("Return the pivot's new index.", "return left - 1"),
  ];
  const quicksort = (o) => () => [
    S("**Requires**: a real segment.", "assert lo <= hi"),
    S("Base case: zero or one element is already sorted.", "if hi - lo <= 1", [S("", "return")]),
    first(o) ? S("The pivot: the first element.", "let pi = lo")
      : S("The pivot: the middle element (`lo + (hi - lo) / 2`).", "let pi = lo + (hi - lo) / 2"),
    S("Partition around it. The pivot lands in its final place, `mid`.", "let mid = partition(A, lo, pi, hi)"),
    S("Sort the left side…", "quicksort(A, lo, mid)"),
    S("…and the right side (the pivot itself stays put).", "quicksort(A, mid + 1, hi)"),
    S("**Ensures**: sorted.", "assert is_sorted(A, lo, hi)"),
  ];

  function setup(c) {
    return {
      py: [],
      cpp: [{ say: "At the top of the file: `<vector>` for the arrays, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <iostream>", "#include <cassert>"] }],
      java: [{ say: "Declare `class Quicksort`. Every function goes inside it, as a `static` method.", code: ["class Quicksort {"], close: ["}"], body: [], deferClose: true, cls: O }],
      rs: [{ say: "At the very top: allow capital variable names (the array is called `A`).", code: ["#![allow(non_snake_case)]"] }],
    }[c.lang];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    return [
      { id: "contracts", title: "Contracts",
        goal: "Quicksort sorts `A[lo, hi)` in place: pick a **pivot**, **partition** the segment so smaller elements come before the pivot and bigger ones after, then sort both sides. " +
          "First the contracts: {fn:is_sorted(A, lo, hi)}, and {fn:all_at_most} and {fn:all_bigger}, which say which side of a value a whole segment is on.",
        decl: setup,
        fns: [fn("is_sorted", BS.isSorted, "Write {fn:is_sorted}: is `A[lo, hi)` in order?"),
          fn("all_at_most", allAtMost, "Write {fn:all_at_most}."), fn("all_bigger", allBigger, "Write {fn:all_bigger}.")],
        contractNote: true, tests: ["contracts"] },
      { id: "swap", title: "Swap",
        goal: "Quicksort moves elements around with {fn:swap(A, i, j)}, which exchanges `A[i]` and `A[j]`.",
        fns: [fn("swap", swap, "Write {fn:swap}.")], tests: ["contracts", "swap"] },
      { id: "partition", title: "Partition",
        goal: "{fn:partition(A, lo, pi, hi)} rearranges `A[lo, hi)` around the pivot `A[pi]` and returns the pivot's new index `mid`: everything in `A[lo, mid)` is at most the pivot, everything in `A[mid + 1, hi)` is bigger. " +
          "One pass with two growing sides, `left` and `right`, and the unknown part between them: O(n).",
        fns: [fn("partition", partition, "Write the header: {fn:partition} takes the array, `lo`, the pivot's index and `hi`, and returns an index.")], tests: ["contracts", "swap", "partition"] },
      { id: "quicksort", title: "Quicksort",
        goal: `{fn:quicksort(A, lo, hi)}: zero or one element is sorted; otherwise pick the pivot (${first(o) ? "the first element" : "the middle element"}), partition, and sort both sides recursively. ` +
          (first(o) ? "Typical input: O(n log n). Already-sorted input: every partition is lopsided, O(n²)." : "Typical input, including sorted input: O(n log n)."),
        fns: [fn("quicksort", quicksort(o), "Write the header: {fn:quicksort} takes the array and the segment to sort.")], tests: ["contracts", "swap", "partition", "quicksort"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const printArr = () => [S("", 'print "["'), S("", "for k = lo to hi", [S("", "if k > lo", [S("", 'print ", "')]), S("", "print A[k]")]), S("", 'print "]"')];
  // a side of a partition, printed in sorted order: any correct partition prints the same
  const printSorted = () => [
    S("", "let B = vec()"), S("", "for k = lo to hi", [S("", "push B, A[k]")]),
    S("", "for i = 1 to hi - lo", [S("", "let j = i"), S("", "while j > 0 and B[j - 1] > B[j]", [S("", "swap B, j - 1, j"), S("", "j -= 1")])]),
    S("", 'print "["'), S("", "for k = 0 to hi - lo", [S("", "if k > 0", [S("", 'print ", "')]), S("", "print B[k]")]), S("", 'print "]"'),
  ];
  const sortJS = (xs) => [...xs].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const DATA = {
    int: { w: [10, 20, 30, 40, 50, 60, 70], mixed: [50, 20, 70, 10, 40, 30, 60], pool: [3, 8, 12, 17, 21, 26, 33, 38, 41, 47, 52, 58, 64, 69, 73] },
    generic: { w: ["ant", "bee", "cat", "dog", "eel", "fox", "gnu"], mixed: ["eel", "bee", "gnu", "ant", "dog", "cat", "fox"],
      pool: ["ant", "bat", "bee", "cat", "cod", "cow", "dog", "eel", "elk", "emu", "fox", "gnu", "hen", "jay", "owl"] },
  };

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const arr = (A) => `[${A.join(", ")}]`;
    const showA = (st) => [S("", `print_arr(A, 0, ${st.A.length})`)];
    return {
      arr: (A, st) => { st.A = [...A]; return { ir: [S("", `let A = [${A.map(lit).join(", ")}]`), say(`A = ${arr(A)}`), S("", "println")], out: `A = ${arr(A)}` }; },
      is_sorted: ([lo, hi], st) => C.ops.bool(o, `is_sorted(A, ${lo}, ${hi})`, `${nm("is_sorted")}(A, ${lo}, ${hi})`,
        st.A.slice(lo, hi).every((x, i, B) => i === 0 || !(x < B[i - 1]))),
      all_at_most: ([lo, hi, x], st) => C.ops.bool(o, `all_at_most(A, ${lo}, ${hi}, ${lit(x)})`, `${nm("all_at_most")}(A, ${lo}, ${hi}, ${x})`, st.A.slice(lo, hi).every((y) => !(x < y))),
      all_bigger: ([lo, hi, x], st) => C.ops.bool(o, `all_bigger(A, ${lo}, ${hi}, ${lit(x)})`, `${nm("all_bigger")}(A, ${lo}, ${hi}, ${x})`, st.A.slice(lo, hi).every((y) => x < y)),
      swap: ([i, j], st) => {
        [st.A[i], st.A[j]] = [st.A[j], st.A[i]];
        return { ir: [S("", `swap(A, ${i}, ${j})`), say(`swap(A, ${i}, ${j}): A = `), ...showA(st), S("", "println")], out: `swap(A, ${i}, ${j}): A = ${arr(st.A)}` };
      },
      partition: ([lo, pi, hi], st) => {
        const segA = st.A.slice(lo, hi), p = st.A[pi];
        const small = sortJS(segA.filter((y) => !(p < y)));
        small.splice(small.lastIndexOf(p), 1);   // the pivot itself sits between the sides
        const big = sortJS(segA.filter((y) => p < y));
        const mid = lo + small.length;
        const n = st.A.length;
        return { ir: [S("", `let mid = partition(A, ${lo}, ${pi}, ${hi})`), say(`${nm("partition")}(A, ${lo}, ${pi}, ${hi}) = `), S("", "print mid"),
          say(`: A[0, ${lo}) = `), S("", `print_arr(A, 0, ${lo})`), say(", left = "), S("", `print_sorted(A, ${lo}, mid)`), say(", pivot = "), S("", "print A[mid]"),
          say(", right = "), S("", `print_sorted(A, mid + 1, ${hi})`), say(`, A[${hi}, ${n}) = `), S("", `print_arr(A, ${hi}, ${n})`), S("", "println")],
          out: `${nm("partition")}(A, ${lo}, ${pi}, ${hi}) = ${mid}: A[0, ${lo}) = ${arr(st.A.slice(0, lo))}, left = ${arr(small)}, pivot = ${p}, right = ${arr(big)}, A[${hi}, ${n}) = ${arr(st.A.slice(hi))}` };
      },
      quicksort: ([lo, hi], st) => {
        st.A.splice(lo, hi - lo, ...sortJS(st.A.slice(lo, hi)));
        return { ir: [S("", `quicksort(A, ${lo}, ${hi})`), say(`quicksort(A, ${lo}, ${hi}): A = `), ...showA(st), S("", "println")], out: `quicksort(A, ${lo}, ${hi}): A = ${arr(st.A)}` };
      },
    };
  }

  function testGroups(names, o, seed) {
    const D = DATA[o.type];
    const [a, b, c, d, e, f, g] = D.w;
    const G = {
      contracts: [
        [["arr", D.w], ["is_sorted", [0, 7]], ["is_sorted", [3, 3]], ["all_at_most", [0, 3, c]], ["all_at_most", [0, 4, c]], ["all_at_most", [2, 2, a]],
          ["all_bigger", [4, 7, d]], ["all_bigger", [3, 7, d]], ["all_bigger", [7, 7, a]], ["all_at_most", [0, 7, g]], ["all_bigger", [0, 7, a]]],
        [["arr", D.mixed], ["is_sorted", [0, 7]], ["is_sorted", [1, 2]], ["is_sorted", [3, 5]]],
      ],
      swap: [[["arr", [a, b, c, d]], ["swap", [0, 2]], ["swap", [1, 1]], ["swap", [3, 0]]]],
      partition: [
        [["arr", D.mixed], ["partition", [0, 4, 7]]],
        [["arr", D.mixed], ["partition", [0, 3, 7]]],
        [["arr", D.mixed], ["partition", [0, 2, 7]]],
        [["arr", D.mixed], ["partition", [1, 5, 6]]],
        [["arr", [c, a, c, b, c]], ["partition", [0, 0, 5]]],
        [["arr", [d, b]], ["partition", [0, 0, 2]]],
        [["arr", [d, b, e]], ["partition", [1, 1, 2]]],
      ],
      quicksort: [
        [["arr", D.mixed], ["quicksort", [0, 7]]],
        [["arr", [...D.w].reverse()], ["quicksort", [0, 7]]],
        [["arr", D.w], ["quicksort", [0, 7]]],
        [["arr", []], ["quicksort", [0, 0]]],
        [["arr", [c]], ["quicksort", [0, 1]]],
        [["arr", [c, a, c, b, a, c]], ["quicksort", [0, 6]]],
        [["arr", D.mixed], ["quicksort", [2, 5]], ["quicksort", [0, 7]]],
      ],
    };
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      for (let k = 0; k < 4; k++) {
        const n = Math.floor(r() * 21);
        const A = Array.from({ length: n }, () => D.pool[Math.floor(r() * D.pool.length)]);
        groups.push([["arr", A], ["quicksort", [0, n]]]);
      }
    }
    return groups;
  }

  const refFns = (o, names) => [{ name: "print_arr", steps: printArr() }, ...(names.includes("partition") ? [{ name: "print_sorted", steps: printSorted() }] : [])];

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const what = {
      is_sorted: "true if `A[lo, hi)` is in order",
      all_at_most: "true if every element of `A[lo, hi)` is at most `x`",
      all_bigger: "true if every element of `A[lo, hi)` is bigger than `x`",
      swap: "exchange `A[i]` and `A[j]`",
      partition: "rearrange `A[lo, hi)` around the pivot `A[pi]`; return its new index `mid`, with `A[lo, mid)` at most the pivot and `A[mid + 1, hi)` bigger; O(hi - lo)",
      quicksort: `sort \`A[lo, hi)\` in place, with the ${first(o) ? "first" : "middle"} element as pivot`,
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check the contracts: partition's loop invariant and result, and \`${fname("is_sorted", L)}\` after sorting. ${L === "java" ? "Put the functions inside `class Quicksort` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), checks each partition's two sides, and sorts every array from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "quicksort", num: "13.03", title: "Quicksort", heading: "Build quicksort, step by step", diagramLabel: "Your partition",
    OPTIONS, DEFAULTS, FNS, STRUCTS: {}, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

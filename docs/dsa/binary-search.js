// 13.01 Binary search: search a sorted array A[0, n) by halving the segment [lo, hi), iteratively or recursively,
// in Python, C++, Java or Rust; then the lower bound (the first element >= x, or where x would be inserted).
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type that can be ordered: a type parameter T (Comparable in Java, PartialOrd in Rust)." } },
    { id: "style", label: "Style", choices: [["iterative", "iterative"], ["recursive", "recursive"]],
      help: { iterative: "A loop narrows [lo, hi) until it finds x or the segment is empty. O(1) extra space.",
        recursive: "Each call compares the middle element, then searches one half by calling itself. O(log n) stack space." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", style: "iterative" };
  const rec = (o) => o.style === "recursive";
  const describe = (o) => `${rec(o) ? "recursive" : "iterative"} binary search over ${o.type === "int" ? "int" : "generic"} elements, in ${DL.LANGS[o.lang].name}`;

  function complexity(o) {
    return [
      { id: "search", label: "Binary search", big: "O(log n)", note: "each round halves [lo, hi)" },
      { id: "lower", label: "Lower bound", big: "O(log n)", note: "the same halving" },
      { id: "space", label: "Extra space", big: rec(o) ? "O(log n)" : "O(1)", note: rec(o) ? "one stack frame per halving" : "just lo, hi and mid" },
      { id: "linear", label: "Linear search", big: "O(n)", note: "what `is_in` does: check every element", extra: true },
      { id: "minmax", label: "Minimum / maximum", big: "O(1)", note: "`A[0]` and `A[n - 1]` of a sorted array", extra: true },
      { id: "insert", label: "Insert, staying sorted", big: "O(n)", note: "finding the spot is O(log n); shifting the rest is O(n)", extra: true },
      { id: "is_sorted", label: "Contract is_sorted", big: "O(n)", note: "one pass over the segment" },
      { id: "contracts", label: "With contracts on", big: "O(n log n)", note: "the invariant checks look at the whole array every round" },
    ];
  }

  function diagram(o) {
    const v = o.type === "int" ? ["10", "20", "30", "40", "50", "60", "70"] : ["ant", "bee", "cat", "dog", "eel", "fox", "gnu"];
    const role = { 0: "lo", 3: "mid" };
    const cells = v.map((x, i) => ({ box: x, tags: [String(i), role[i] || ""] }));
    cells.push({ box: "end", tags: ["7", "hi = n"], cls: "dummy" });   // one past the last element
    return {
      rows: [cells],
      caption: "Searching A[lo, hi) for x: everything left of lo is smaller than x and everything from hi on is bigger, so x can only be in [lo, hi). Each round compares A[mid] with x and keeps one half.",
    };
  }

  // ---------------------------------------------------------------- specs
  const O = "BinarySearch";
  const ORD = ["PartialOrd"];
  const seg = [["A", "Arr"], ["lo", "int"], ["hi", "int"]];
  const FNS = {
    is_sorted: { params: seg, ret: "bool", owner: O, contract: true },
    is_in: { params: [...seg, ["x", "Tref"]], ret: "bool", owner: O, contract: true },
    all_smaller: { params: [...seg, ["x", "Tref"]], ret: "bool", owner: O, contract: true },
    all_bigger: { params: [...seg, ["x", "Tref"]], ret: "bool", owner: O, contract: true },
    all_at_least: { params: [...seg, ["x", "Tref"]], ret: "bool", owner: O, contract: true },
    binsearch: { params: [["A", "Arr"], ["n", "int"], ["x", "Tref"]], ret: "index", owner: O, bounds: ORD },
    binsearch_rec: { params: [...seg, ["x", "Tref"]], ret: "index", owner: O, bounds: ORD },
    lower_bound: { params: [["A", "Arr"], ["n", "int"], ["x", "Tref"]], ret: "int", owner: O, bounds: ORD },
    lower_bound_rec: { params: [...seg, ["x", "Tref"]], ret: "int", owner: O, bounds: ORD },
  };

  // ---------------------------------------------------------------- the functions
  const SORTED = S("**Requires**: the array is sorted.", "assert is_sorted(A, 0, n)");
  const MID = S("The middle of [`lo`, `hi`). `lo + (hi - lo) / 2` instead of `(lo + hi) / 2`: the same value, but it can't overflow when `lo + hi` is huge.", "let mid = lo + (hi - lo) / 2");
  const START = [S("The search segment starts as the whole array, [`0`, `n`).", "let lo = 0"), S("", "let hi = n")];

  const isSorted = () => [
    S("Every element must be at least the one before it. Compare each `A[i]` with `A[i - 1]` for `i` in [`lo + 1`, `hi`): starting at `lo + 1`, instead of stopping at `hi - 1`, means an empty segment can't make an index negative.",
      "for i = lo + 1 to hi", [S("A pair out of order: not sorted.", "if A[i - 1] > A[i]", [S("", "return false")])]),
    S("No pair out of order. (An empty or one-element segment is sorted.)", "return true"),
  ];
  const isIn = () => [
    S("Look at every element of `A[lo, hi)`: a plain linear search, slow but obviously right. Binary search must agree with it.", "for i = lo to hi", [
      S("Found it.", "if A[i] == x", [S("", "return true")])]),
    S("Not there.", "return false"),
  ];
  const all = (op, say) => () => [
    S(say, "for i = lo to hi", [S("One element on the wrong side: false.", `if A[i] ${op} x`, [S("", "return false")])]),
    S("Every element passed (and an empty segment passes).", "return true"),
  ];
  const allSmaller = all(">=", "Every element of `A[lo, hi)` must be smaller than `x`.");
  const allBigger = all("<=", "Every element of `A[lo, hi)` must be bigger than `x`.");
  const allAtLeast = all("<", "Every element of `A[lo, hi)` must be at least `x`.");

  const binsearchIter = () => [SORTED, ...START,
    S("While the segment isn't empty…", "while lo < hi", [
      S("**Loop invariant**: everything left of `lo` is smaller than `x`…", "assert all_smaller(A, 0, lo, x)"),
      S("…and everything from `hi` on is bigger.", "assert all_bigger(A, hi, n, x)"),
      MID,
      S("Found it.", "if A[mid] == x", [S("", "return mid")]),
      S("`A[mid]` is smaller, so `x` can only be to its right: [`lo`, `mid`] joins the smaller side.", "elif A[mid] < x", [S("", "lo = mid + 1")]),
      S("Otherwise `A[mid]` is bigger, so `x` can only be to its left: `mid` is the new end.", "else", [S("", "hi = mid")]),
    ]),
    S("The segment is empty, so by the invariant `x` isn't in `A`. **Ensures**: check that.", "assert not is_in(A, 0, n, x)"),
    S("Not found.", "return -1"),
  ];
  const binsearchRec = () => [
    S("**Requires**: the segment is sorted.", "assert is_sorted(A, lo, hi)"),
    S("Base case: an empty segment holds nothing.", "if lo == hi", [S("", "return -1")]),
    MID,
    S("Found it.", "if A[mid] == x", [S("", "return mid")]),
    S("`A[mid]` is smaller: `x` can only be in the right half, [`mid + 1`, `hi`).", "if A[mid] < x", [S("", "return binsearch_rec(A, mid + 1, hi, x)")]),
    S("Otherwise only the left half, [`lo`, `mid`), is left.", "return binsearch_rec(A, lo, mid, x)"),
  ];
  const binsearchWrap = () => [SORTED, S("Search the whole array, [`0`, `n`).", "return binsearch_rec(A, 0, n, x)")];

  const LB_ENSURES = S("**Ensures**: everything before the answer is smaller than `x`, and everything from it on is at least `x`.",
    "assert all_smaller(A, 0, lo, x) and all_at_least(A, lo, n, x)");
  const lowerIter = () => [SORTED, ...START,
    S("While the segment isn't empty…", "while lo < hi", [
      S("**Loop invariant**: everything left of `lo` is smaller than `x`…", "assert all_smaller(A, 0, lo, x)"),
      S("…and everything from `hi` on is at least `x`.", "assert all_at_least(A, hi, n, x)"),
      MID,
      S("`A[mid]` is smaller: the answer is to its right.", "if A[mid] < x", [S("", "lo = mid + 1")]),
      S("`A[mid]` is at least `x`: it may be the answer, so it becomes the end of the segment (kept on the \"at least\" side).", "else", [S("", "hi = mid")]),
    ]),
    LB_ENSURES,
    S("`lo == hi` now: the first index whose element is at least `x`.", "return lo"),
  ];
  const lowerRec = () => [
    S("**Requires**: the segment is sorted.", "assert is_sorted(A, lo, hi)"),
    S("Base case: an empty segment; the answer is `lo`.", "if lo == hi", [S("", "return lo")]),
    MID,
    S("`A[mid]` is smaller: the answer is in [`mid + 1`, `hi`).", "if A[mid] < x", [S("", "return lower_bound_rec(A, mid + 1, hi, x)")]),
    S("Otherwise `A[mid]` may be the answer: search [`lo`, `mid`), which returns `mid` if nothing earlier qualifies.", "return lower_bound_rec(A, lo, mid, x)"),
  ];
  const lowerWrap = () => [SORTED,
    S("Search the whole array.", "let lo = lower_bound_rec(A, 0, n, x)"),
    LB_ENSURES,
    S("Return it.", "return lo"),
  ];

  // ---------------------------------------------------------------- the file's setup, per language
  function setup(c) {
    return {
      py: [],
      cpp: [{ say: "At the top of the file: `<vector>` for the arrays, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <iostream>", "#include <cassert>"] }],
      java: [{ say: "Declare `class BinarySearch`. Every function goes inside it, as a `static` method.", code: ["class BinarySearch {"], close: ["}"], body: [], deferClose: true, cls: O }],
      rs: [{ say: "At the very top: allow capital variable names (the array is called `A`).", code: ["#![allow(non_snake_case)]"] }],
    }[c.lang];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const r = rec(o);
    const on = C.contractsOn(o);
    return [
      { id: "sorted", title: "Sorted arrays and their contracts",
        goal: "Binary search works on a **sorted** array `A`. A **segment** `A[lo, hi)` is the elements from index `lo` up to, not including, `hi`: " +
          "it has `hi - lo` elements, and it's empty when `lo == hi`. Before searching, write two contracts: {fn:is_sorted(A, lo, hi)} and {fn:is_in(A, lo, hi, x)}.",
        decl: setup,
        fns: [fn("is_sorted", isSorted, "Write {fn:is_sorted}: is `A[lo, hi)` in order?"), fn("is_in", isIn, "Write {fn:is_in}: does `x` occur in `A[lo, hi)`?")],
        contractNote: true, tests: ["sorted"] },
      { id: "invariant", title: "The loop invariant",
        goal: "Binary search keeps a segment [`lo`, `hi`) where `x` could still be: everything left of it is smaller than `x` and everything from `hi` on is bigger, " +
          "`A[0, lo) < x < A[hi, n)`. That's the **loop invariant**: true before the first round and after every round, so still true when the segment is empty. Write the two checks it needs.",
        fns: [fn("all_smaller", allSmaller, "Write {fn:all_smaller}."), fn("all_bigger", allBigger, "Write {fn:all_bigger}.")],
        tests: ["sorted", "invariant"] },
      { id: "binsearch", title: "Binary search",
        goal: (on ? "" : "Binary search works on a **sorted** array `A`; `A[lo, hi)` is the segment from `lo` up to, not including, `hi`. ") +
          "{fn:binsearch(A, n, x)} returns an index `i` with `A[i] == x`, or -1 if `x` isn't in `A[0, n)`. Compare the middle element with `x` and keep the half where `x` can be: " +
          "the segment halves every time, so O(log n)." + (r ? " Recursively: {fn:binsearch_rec(A, lo, hi, x)} searches [`lo`, `hi`) and calls itself on one half; {fn:binsearch} starts it on the whole array." : ""),
        fns: r ? [fn("binsearch_rec", binsearchRec, "Write {fn:binsearch_rec}, which searches the segment [`lo`, `hi`)."), fn("binsearch", binsearchWrap, "Write {fn:binsearch}, which searches the whole array.")]
          : [fn("binsearch", binsearchIter, "Write the header: {fn:binsearch} takes the array, its length and `x`, and returns an index.")],
        tests: ["sorted", "invariant", "binsearch"] },
      { id: "lower_bound", title: "Lower bound",
        goal: "{fn:lower_bound(A, n, x)} returns the first index whose element is at least `x`: where `x` is, or where it would be inserted to keep `A` sorted (`n` if every element is smaller). " +
          "With duplicates it finds the first copy." + (on ? " The invariant becomes `A[0, lo) < x <= A[hi, n)`, so write {fn:all_at_least} first." : " The segment keeps `A[0, lo) < x <= A[hi, n)`."),
        fns: [fn("all_at_least", allAtLeast, "Write {fn:all_at_least}."),
          ...(r ? [fn("lower_bound_rec", lowerRec, "Write {fn:lower_bound_rec}."), fn("lower_bound", lowerWrap, "Write {fn:lower_bound}.")]
            : [fn("lower_bound", lowerIter, "Write {fn:lower_bound}.")])],
        tests: ["sorted", "invariant", "binsearch", "lower_bound"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const DATA = {
    int: { S7: [10, 20, 30, 40, 50, 60, 70], absent: [5, 25, 75], unsorted: [30, 10, 20], dups: [10, 20, 20, 20, 40], probes: [5, 10, 15, 20, 30, 40, 45],
      pool: [3, 8, 12, 17, 21, 26, 33, 38, 41, 47, 52, 58, 64, 69, 73, 77, 85, 91, 96] },
    generic: { S7: ["ant", "bee", "cat", "dog", "eel", "fox", "gnu"], absent: ["aardvark", "cow", "yak"], unsorted: ["cat", "ant", "bee"],
      dups: ["ant", "bee", "bee", "bee", "dog"], probes: ["aardvark", "ant", "bat", "bee", "cat", "dog", "yak"],
      pool: ["ant", "bat", "bee", "cat", "cod", "cow", "dog", "eel", "elk", "emu", "fox", "gnu", "hen", "jay", "owl", "pig", "ram", "yak"] },
  };
  const lt = (a, b) => a < b;   // JavaScript orders numbers and lowercase ASCII words like the four languages do

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const arr = (A) => `[${A.join(", ")}]`;
    const segFn = (name, f) => ([lo, hi, x], st) => C.ops.bool(o, `${name}(A, ${lo}, ${hi}, ${lit(x)})`, `${nm(name)}(A, ${lo}, ${hi}, ${x})`, f(st.A.slice(lo, hi), x));
    return {
      arr: (A, st) => { st.A = A; return { ir: [S("", `let A = [${A.map(lit).join(", ")}]`), say(`A = ${arr(A)}`), S("", "println")], out: `A = ${arr(A)}` }; },
      is_sorted: ([lo, hi], st) => C.ops.bool(o, `is_sorted(A, ${lo}, ${hi})`, `${nm("is_sorted")}(A, ${lo}, ${hi})`,
        st.A.slice(lo, hi).every((x, i, B) => i === 0 || !lt(x, B[i - 1]))),
      is_in: segFn("is_in", (B, x) => B.includes(x)),
      all_smaller: segFn("all_smaller", (B, x) => B.every((y) => lt(y, x))),
      all_bigger: segFn("all_bigger", (B, x) => B.every((y) => lt(x, y))),
      all_at_least: segFn("all_at_least", (B, x) => B.every((y) => !lt(y, x))),
      binsearch: ([n, x], st) => C.ops.value(o, `binsearch(A, ${n}, ${lit(x)})`, `binsearch(A, ${n}, ${x})`, st.A.slice(0, n).indexOf(x)),
      lower_bound: ([n, x], st) => {
        const B = st.A.slice(0, n);
        const i = B.findIndex((y) => !lt(y, x));
        return C.ops.value(o, `lower_bound(A, ${n}, ${lit(x)})`, `${nm("lower_bound")}(A, ${n}, ${x})`, i < 0 ? n : i);
      },
    };
  }

  function testGroups(names, o, seed) {
    const D = DATA[o.type];
    const v = D.S7, [z0, z1, z2] = D.absent;
    const G = {
      sorted: [
        [["arr", v], ["is_sorted", [0, 7]], ["is_sorted", [2, 5]], ["is_sorted", [3, 3]], ["is_in", [0, 7, v[3]]], ["is_in", [0, 3, v[3]]], ["is_in", [3, 4, v[3]]],
          ["is_in", [4, 7, v[3]]], ["is_in", [0, 7, z0]], ["is_in", [2, 2, v[2]]], ["is_in", [6, 7, v[6]]]],
        [["arr", D.unsorted], ["is_sorted", [0, 3]], ["is_sorted", [1, 3]], ["is_sorted", [0, 2]], ["is_sorted", [0, 1]], ["is_sorted", [2, 3]]],
        [["arr", []], ["is_sorted", [0, 0]], ["is_in", [0, 0, v[0]]]],
        [["arr", D.dups], ["is_sorted", [0, 5]], ["is_in", [1, 4, D.dups[2]]]],
      ],
      invariant: [
        [["arr", v], ["all_smaller", [0, 3, v[3]]], ["all_smaller", [0, 4, v[3]]], ["all_smaller", [0, 0, v[0]]], ["all_smaller", [2, 5, v[6]]], ["all_smaller", [0, 7, z2]],
          ["all_smaller", [6, 7, v[6]]], ["all_bigger", [4, 7, v[3]]], ["all_bigger", [3, 7, v[3]]], ["all_bigger", [7, 7, v[0]]], ["all_bigger", [0, 7, z0]], ["all_bigger", [0, 1, v[0]]]],
      ],
      binsearch: [
        [["arr", v], ...v.map((x) => ["binsearch", [7, x]]), ...D.absent.map((x) => ["binsearch", [7, x]]), ["binsearch", [3, v[5]]], ["binsearch", [3, v[1]]]],
        [["arr", []], ["binsearch", [0, v[0]]]],
        [["arr", [v[3]]], ["binsearch", [1, v[3]]], ["binsearch", [1, v[0]]], ["binsearch", [1, v[6]]]],
        [["arr", [v[1], v[4]]], ["binsearch", [2, v[1]]], ["binsearch", [2, v[4]]], ["binsearch", [2, v[2]]], ["binsearch", [2, z0]], ["binsearch", [2, z2]]],
      ],
      lower_bound: [
        [["arr", D.dups], ["all_at_least", [1, 5, D.dups[1]]], ["all_at_least", [0, 5, D.dups[1]]], ["all_at_least", [5, 5, v[0]]], ["all_at_least", [4, 5, D.dups[4]]],
          ...D.probes.map((x) => ["lower_bound", [5, x]]), ["lower_bound", [3, D.dups[4]]]],
        [["arr", v], ["lower_bound", [7, z0]], ["lower_bound", [7, z1]], ["lower_bound", [7, z2]], ["lower_bound", [7, v[0]]], ["lower_bound", [7, v[6]]]],
        [["arr", []], ["lower_bound", [0, v[0]]]],
      ],
    };
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      const pick = (k) => { const P = [...D.pool]; const out = []; while (out.length < k && P.length) out.push(P.splice(Math.floor(r() * P.length), 1)[0]); return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)); };
      for (let g = 0; g < 3; g++) {
        const A = pick(Math.floor(r() * 14));
        const ops = [["arr", A]];
        for (let i = 0; i < 12; i++) ops.push(["binsearch", [A.length, D.pool[Math.floor(r() * D.pool.length)]]]);
        groups.push(ops);
      }
      const B = pick(6).flatMap((x) => (r() < 0.5 ? [x, x] : [x]));
      groups.push([["arr", B], ...D.pool.slice(0, 10).map((x) => ["lower_bound", [B.length, x]])]);
    }
    return groups;
  }

  const refFns = () => [];   // the tests only call the user's functions on arrays they build

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const r = rec(o);
    const what = {
      is_sorted: "true if `A[lo, hi)` is in order",
      is_in: "true if `x` occurs in `A[lo, hi)` (linear search)",
      all_smaller: "true if every element of `A[lo, hi)` is smaller than `x`",
      all_bigger: "true if every element of `A[lo, hi)` is bigger than `x`",
      all_at_least: "true if every element of `A[lo, hi)` is at least `x`",
      ...(r ? { binsearch_rec: "search the segment [lo, hi): an index of `x`, or -1" } : {}),
      binsearch: "an index `i` with `A[i] == x`, or -1; requires a sorted array; O(log n)",
      ...(r ? { lower_bound_rec: "the lower bound within [lo, hi)" } : {}),
      lower_bound: "the first index whose element is at least `x` (`n` if none); O(log n)",
    };
    return [
      `Build an **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check the contracts: sortedness on entry${r ? "" : ", the loop invariant every round"}, and the results on exit. ${L === "java" ? "Put the functions inside `class BinarySearch` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored) and runs every test from the tutorial plus random arrays and searches.",
    ].join("\n");
  }

  const DEF = {
    id: "binary-search", num: "13.01", title: "Binary search", heading: "Build binary search, step by step", diagramLabel: "Your search",
    OPTIONS, DEFAULTS, FNS, STRUCTS: {}, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_sorted", "is_in", "all_smaller", "all_bigger", "all_at_least"],
    _internals: { isSorted, all },   // (the sorting tutorials reuse them)
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

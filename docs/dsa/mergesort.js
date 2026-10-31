// 13.02 Mergesort: sort A[lo, hi) by sorting both halves recursively and merging them (a merge function plus a recursive
// mergesort), in Python, C++, Java or Rust.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const BS = ((root.DSATutorials || {})["binary-search"] || (typeof require !== "undefined" ? require("./binary-search.js") : null))._internals;
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type that can be ordered: a type parameter T (Comparable in Java, PartialOrd + Clone in Rust)." } },
  ];
  const DEFAULTS = { lang: "py", type: "int" };
  const describe = (o) => `mergesort of ${o.type === "int" ? "int" : "generic"} elements, in ${DL.LANGS[o.lang].name}`;

  function complexity() {
    return [
      { id: "time", label: "Time, every case", big: "O(n log n)", note: "log n levels of halving, O(n) merging per level" },
      { id: "merge", label: "Merge", big: "O(n)", note: "each element is copied into B and back once" },
      { id: "space", label: "Extra space", big: "O(n)", note: "the temporary array B (plus an O(log n) stack)" },
      { id: "sorted", label: "Already sorted input", big: "O(n log n)", note: "no better, no worse", extra: true },
      { id: "stable", label: "Stable", big: "yes", note: "on a tie, merge takes the left element first" },
      { id: "quick", label: "Quicksort, for comparison", big: "O(n log n)", note: "expected; in place, but O(n²) worst case and not stable", extra: true },
      { id: "is_sorted", label: "Contract is_sorted", big: "O(n)", note: "one pass over the segment" },
    ];
  }

  function diagram(o) {
    const v = o.type === "int" ? ["10", "20", "30", "40", "50", "60"] : ["ant", "bee", "cat", "dog", "eel", "fox"];
    const row = (label, xs, tag) => [{ label }, ...xs.map((x, i) => ({ box: x, tags: i === 0 && tag ? [tag] : [] }))];
    return {
      rows: [row("left", [v[0], v[3], v[4]], "A[lo, mid)"), row("right", [v[1], v[2], v[5]], "A[mid, hi)"), row("merged", v, "A[lo, hi)")],
      caption: "mergesort sorts A[lo, mid) and A[mid, hi) the same way (recursively), then merge walks the two sorted halves, always taking the smaller front element into B, and copies B back into A[lo, hi).",
    };
  }

  // ---------------------------------------------------------------- specs
  const O = "Mergesort";
  const seg = [["lo", "int"], ["hi", "int"]];
  const FNS = {
    is_sorted: { params: [["A", "Arr"], ...seg], ret: "bool", owner: O },
    merge: { params: [["A", "ArrMut"], ["lo", "int"], ["mid", "int"], ["hi", "int"]], ret: "void", owner: O, bounds: ["PartialOrd", "Clone"] },
    mergesort: { params: [["A", "ArrMut"], ...seg], ret: "void", owner: O },
    print_arr: { params: [["A", "Arr"], ...seg], ret: "void", harness: true },
  };

  // ---------------------------------------------------------------- the functions
  const merge = () => [
    S("**Requires**: both halves are sorted.", "assert is_sorted(A, lo, mid) and is_sorted(A, mid, hi)"),
    S("A temporary array for the merged result.", "let B = vec()"),
    S("`i` walks the left half, `A[lo, mid)`…", "let i = lo"),
    S("…and `j` the right half, `A[mid, hi)`.", "let j = mid"),
    S("While both halves have elements left…", "while i < mid and j < hi", [
      S("…take the smaller front element. On a tie take the left one, so equal elements keep their order: that makes mergesort **stable**.", "if A[i] <= A[j]", [
        S("", "push B, A[i]"), S("", "i += 1")]),
      S("Otherwise the right one is smaller.", "else", [S("", "push B, A[j]"), S("", "j += 1")]),
    ]),
    S("One half is used up: copy what's left of the other. First the left…", "while i < mid", [S("", "push B, A[i]"), S("", "i += 1")]),
    S("…then the right. (Only one of these two loops does anything.)", "while j < hi", [S("", "push B, A[j]"), S("", "j += 1")]),
    S("`B` holds `A[lo, hi)` in order: copy it back.", "for k = 0 to hi - lo", [S("", "A[lo + k] = B[k]")]),
    S("**Ensures**: the whole segment is sorted.", "assert is_sorted(A, lo, hi)"),
  ];
  const mergesort = () => [
    S("**Requires**: a real segment.", "assert lo <= hi"),
    S("Base case: zero or one element is already sorted.", "if hi - lo <= 1", [S("", "return")]),
    S("Split at the middle (`lo + (hi - lo) / 2`, which can't overflow).", "let mid = lo + (hi - lo) / 2"),
    S("Sort the left half…", "mergesort(A, lo, mid)"),
    S("…and the right half, the same way.", "mergesort(A, mid, hi)"),
    S("Merge the two sorted halves.", "merge(A, lo, mid, hi)"),
    S("**Ensures**: sorted.", "assert is_sorted(A, lo, hi)"),
  ];

  function setup(c) {
    return {
      py: [],
      cpp: [{ say: "At the top of the file: `<vector>` for the arrays, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <iostream>", "#include <cassert>"] }],
      java: [{ say: "Import `ArrayList` (merge's temporary array grows as it fills).", code: ["import java.util.ArrayList;"] },
        { say: "Declare `class Mergesort`. Every function goes inside it, as a `static` method.", code: ["class Mergesort {"], close: ["}"], body: [], deferClose: true, cls: O }],
      rs: [{ say: "At the very top: allow capital variable names (the array is called `A`).", code: ["#![allow(non_snake_case)]"] }],
    }[c.lang];
  }

  function steps() {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    return [
      { id: "sorted", title: "Sorted segments",
        goal: "Mergesort sorts a segment `A[lo, hi)` in place: split it in half, sort each half the same way, and **merge** the two sorted halves. " +
          "Start with the contract every step checks, {fn:is_sorted(A, lo, hi)}.",
        decl: setup, fns: [fn("is_sorted", BS.isSorted, "Write {fn:is_sorted}: is `A[lo, hi)` in order?")], contractNote: true, tests: ["sorted"] },
      { id: "merge", title: "Merge",
        goal: "{fn:merge(A, lo, mid, hi)} takes two sorted neighbours, `A[lo, mid)` and `A[mid, hi)`, and makes `A[lo, hi)` sorted. Walk both halves with `i` and `j`, " +
          "always taking the smaller front element into a temporary array `B`, then copy `B` back. O(n) for n = `hi - lo`.",
        fns: [fn("merge", merge, "Write the header: {fn:merge} takes the array and the three boundaries.")], tests: ["sorted", "merge"] },
      { id: "mergesort", title: "Mergesort",
        goal: "{fn:mergesort(A, lo, hi)}: a segment of 0 or 1 elements is already sorted; otherwise sort both halves recursively and merge them. " +
          "log n levels of halving with O(n) merging per level: O(n log n), whatever the input.",
        fns: [fn("mergesort", mergesort, "Write the header: {fn:mergesort} takes the array and the segment to sort.")], tests: ["sorted", "merge", "mergesort"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const printArr = () => [S("", 'print "["'), S("", "for k = lo to hi", [S("", "if k > lo", [S("", 'print ", "')]), S("", "print A[k]")]), S("", 'print "]"')];
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
    const after = (call, label, st, f) => {
      f(st);
      return { ir: [S("", call), say(`${label}: A = `), S("", `print_arr(A, 0, ${st.A.length})`), S("", "println")], out: `${label}: A = ${arr(st.A)}` };
    };
    return {
      arr: (A, st) => { st.A = [...A]; return { ir: [S("", `let A = [${A.map(lit).join(", ")}]`), say(`A = ${arr(A)}`), S("", "println")], out: `A = ${arr(A)}` }; },
      is_sorted: ([lo, hi], st) => C.ops.bool(o, `is_sorted(A, ${lo}, ${hi})`, `${nm("is_sorted")}(A, ${lo}, ${hi})`,
        st.A.slice(lo, hi).every((x, i, B) => i === 0 || !(x < B[i - 1]))),
      merge: ([lo, mid, hi], st) => after(`merge(A, ${lo}, ${mid}, ${hi})`, `merge(A, ${lo}, ${mid}, ${hi})`, st,
        (s) => { s.A.splice(lo, hi - lo, ...sortJS(s.A.slice(lo, hi))); }),
      mergesort: ([lo, hi], st) => after(`mergesort(A, ${lo}, ${hi})`, `mergesort(A, ${lo}, ${hi})`, st,
        (s) => { s.A.splice(lo, hi - lo, ...sortJS(s.A.slice(lo, hi))); }),
    };
  }

  function testGroups(names, o, seed) {
    const D = DATA[o.type];
    const [a, b, c, d, e, f, g] = D.w;
    const G = {
      sorted: [
        [["arr", D.w], ["is_sorted", [0, 7]], ["is_sorted", [2, 5]], ["is_sorted", [3, 3]]],
        [["arr", D.mixed], ["is_sorted", [0, 7]], ["is_sorted", [1, 3]], ["is_sorted", [0, 2]], ["is_sorted", [6, 7]]],
        [["arr", []], ["is_sorted", [0, 0]]],
        [["arr", [b, b, d]], ["is_sorted", [0, 3]]],
      ],
      merge: [
        [["arr", [a, d, e, b, c, f]], ["merge", [0, 3, 6]]],
        [["arr", [b, a]], ["merge", [0, 1, 2]]],
        [["arr", [g, d, a, b, e, c]], ["merge", [1, 2, 5]]],
        [["arr", [a, b, c]], ["merge", [0, 0, 3]], ["merge", [0, 3, 3]]],
        [["arr", [b, b, d, a, b, c]], ["merge", [0, 3, 6]]],
      ],
      mergesort: [
        [["arr", D.mixed], ["mergesort", [0, 7]]],
        [["arr", [...D.w].reverse()], ["mergesort", [0, 7]]],
        [["arr", D.w], ["mergesort", [0, 7]]],
        [["arr", []], ["mergesort", [0, 0]]],
        [["arr", [c]], ["mergesort", [0, 1]]],
        [["arr", [c, a, c, b, a]], ["mergesort", [0, 5]]],
        [["arr", D.mixed], ["mergesort", [2, 5]], ["mergesort", [0, 7]]],
      ],
    };
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      for (let k = 0; k < 4; k++) {
        const n = Math.floor(r() * 21);
        const A = Array.from({ length: n }, () => D.pool[Math.floor(r() * D.pool.length)]);
        groups.push([["arr", A], ["mergesort", [0, n]]]);
      }
    }
    return groups;
  }

  const refFns = () => [{ name: "print_arr", steps: printArr() }];

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const what = {
      is_sorted: "true if `A[lo, hi)` is in order",
      merge: "requires sorted `A[lo, mid)` and `A[mid, hi)`; makes `A[lo, hi)` sorted, stably, in O(hi - lo)",
      mergesort: "sorts `A[lo, hi)` in place, O(n log n)",
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_sorted", L)}\` in the contracts. ${L === "java" ? "Put the functions inside `class Mergesort` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored) and sorts every array from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "mergesort", num: "13.02", title: "Mergesort", heading: "Build mergesort, step by step", diagramLabel: "Your sort",
    OPTIONS, DEFAULTS, FNS, STRUCTS: {}, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

// 13.07 Unbounded arrays: an array that grows and shrinks by moving into a bigger or smaller fixed block (CMU 15-122's
// uba), with a choice of growth and shrink rules, in Python, C++, Java or Rust.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type: a type parameter T (in Rust it needs Clone + Default, to fill and copy the block)." } },
    { id: "grow", label: "Growth", choices: [["double", "×2"], ["half", "×1.5"], ["plus", "+4"]],
      help: { double: "When the block is full, move into one twice as big. Each element is copied O(1) times on average.",
        half: "Grow by half (rounding up). Less wasted space than ×2, slightly more copying; still O(1) amortized.",
        plus: "Grow by 4 slots each time. Resizes never get rarer, so n adds copy about n²/8 elements: O(n) amortized." } },
    { id: "shrink", label: "Shrinking", choices: [["quarter", "at ¼ full"], ["half", "below ½ full"], ["never", "never"]],
      help: { quarter: "Halve the block when only a quarter of it is used, so a shrink and the next grow are far apart.",
        half: "Halve the block as soon as less than half of it is used. Adding and removing at that boundary resizes every time: thrashing.",
        never: "Keep the block at its largest size. Fast, but memory never comes back." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", grow: "double", shrink: "quarter" };
  const GROW = { double: "A.limit * 2", half: "A.limit + (A.limit + 1) / 2", plus: "A.limit + 4" };
  const growJS = { double: (l) => l * 2, half: (l) => l + Math.floor((l + 1) / 2), plus: (l) => l + 4 };
  const GROW_SAY = { double: "Twice as many slots.", half: "Half as many again, rounding up (so a 1-slot block still grows).", plus: "Four more slots." };
  const describe = (o) => `unbounded array of ${o.type === "int" ? "int" : "generic"} elements (growing ${{ double: "×2", half: "×1.5", plus: "by 4" }[o.grow]}, ` +
    `${{ quarter: "shrinking at ¼ full", half: "shrinking below ½ full", never: "never shrinking" }[o.shrink]}), in ${DL.LANGS[o.lang].name}`;

  function complexity(o) {
    const g = o.grow, s = o.shrink;
    return [
      { id: "get", label: "Get / set", big: "O(1)", note: "index into `A.data`" },
      { id: "add", label: "Add (amortized)", big: g === "plus" ? "O(n)" : "O(1)",
        note: g === "plus" ? "a resize every 4 adds, each copying everything" : "resizes get rarer as the array grows" },
      { id: "add_worst", label: "One add, worst case", big: "O(n)", note: "when it resizes, every element is copied" },
      { id: "rem", label: "Remove (amortized)", big: s === "half" ? "O(n)" : "O(1)",
        note: s === "half" ? "add, remove, add, remove at the boundary resizes every time" : s === "never" ? "it never resizes" : "after a shrink the array is half full: far from both thresholds" },
      { id: "copies", label: "Copies for n adds", big: { double: "< 2n", half: "< 3n", plus: "≈ n²/8" }[g],
        note: { double: "1 + 2 + 4 + … + n", half: "a geometric sum again", plus: "1 + 5 + 9 + … : quadratic" }[g] },
      { id: "space", label: "Space", big: s === "never" ? "O(peak)" : "O(n)",
        note: s === "never" ? "the block stays at its largest" : s === "quarter" ? "the block is at most about 4× the elements" : "the block is at most about 2× the elements" },
      { id: "len", label: "Length", big: "O(1)", note: "kept in `A.size`" },
      { id: "insert", label: "Insert in the middle", big: "O(n)", note: "everything after it shifts", extra: true },
      { id: "is_uba", label: "Contract is_uba", big: "O(1)", note: "checks size, limit and the block's length" },
    ];
  }

  function diagram(o) {
    const v = o.type === "int" ? ["10", "20", "30"] : ["ant", "bee", "cat"];
    // two label lines per slot: the index, then a marker (so the markers line up)
    const cells = v.map((x, i) => ({ box: x, tags: [String(i), ""] }));
    const free = "\u00a0".repeat(String(v[0]).length);   // an empty slot as wide as a full one
    cells.push({ box: free, tags: ["3", "A.size"], cls: "dummy" }, { box: free, tags: ["4", ""], cls: "dummy" }, { text: "A.limit = 5" });
    return {
      rows: [cells],
      caption: `A.data has A.limit slots and the elements are A.data[0, A.size); at least one slot is always free. When an add fills the last one, the elements move to a bigger block (${{ double: "×2", half: "×1.5", plus: "+4" }[o.grow]}).`,
    };
  }

  // ---------------------------------------------------------------- specs
  const O = "UnboundedArray", M = "mut:UnboundedArray", R = "ref:UnboundedArray";
  const FNS = {
    is_uba: { params: [["A", R]], ret: "bool", owner: O },
    uba_len: { params: [["A", R]], ret: "int", owner: O },
    uba_get: { params: [["A", R], ["i", "int"]], ret: "Tpeek", owner: O },
    uba_set: { params: [["A", M], ["i", "int"], ["x", "T"]], ret: "void", owner: O },
    uba_resize: { params: [["A", M], ["new_limit", "int"]], ret: "void", owner: O, bounds: ["Clone", "Default"] },
    uba_add: { params: [["A", M], ["x", "T"]], ret: "void", owner: O },
    uba_rem: { params: [["A", M]], ret: "T", owner: O },
    show: { params: [["A", R]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
  };
  const STRUCTS = { UnboundedArray: { fields: { size: "int", limit: "int", data: "ArrV" } } };

  // ---------------------------------------------------------------- the functions
  const REQ = S("**Requires**: the array is valid when the function starts.", "assert is_uba(A)");
  const ENS = S("**Ensures**: still valid when it returns.", "assert is_uba(A)");
  const isUba = () => [
    S("One slot always stays free: `size < limit`.", "if A.size >= A.limit", [S("", "return false")]),
    { ...S("A size can't be negative. (Rust's `usize` can't be anyway, so Rust skips this.)", "if A.size < 0", [S("", "return false")]), only: ["py", "cpp", "java"] },
    S("And `data` really has `limit` slots.", "return len(A.data) == A.limit"),
  ];
  const len = () => [S("The number of elements is kept in the array.", "return A.size")];
  const inRange = { ...S("**Requires**: `i` is an element's index, in [`0`, `A.size`).", "assert 0 <= i and i < A.size"), only: ["py", "cpp", "java"] };
  const inRangeRs = { ...S("**Requires**: `i` is an element's index: below `A.size` (a `usize` is never negative).", "assert i < A.size"), only: ["rs"] };
  const get = () => [REQ, inRange, inRangeRs,
    S((c) => (c.lang === "rs" && c.generic ? "Return the element (a reference `&T`: Rust doesn't move it out of the block)." : "Return the element."), "return A.data[i]")];
  const set = () => [REQ, inRange, inRangeRs, S("Overwrite the element.", "A.data[i] = x"), ENS];
  const resize = () => [
    S("**Requires**: the block is consistent and the elements fit in it. Weaker than `is_uba`: `uba_add` calls this exactly when the block is full (`size == limit`).",
      "assert A.size <= A.limit and len(A.data) == A.limit"),
    S("**Requires**: the elements, plus one free slot, fit in the new block.", "assert A.size < new_limit"),
    S("A new block of `new_limit` slots.", "let B = alloc(new_limit)"),
    S("Copy the elements over, `A.data[0, A.size)`: O(size). This is the only expensive part of the whole structure.", "for i = 0 to A.size", [S("", "B[i] = A.data[i]")]),
    S("Use the new block. (C++ and Rust free the old one as it's replaced; Python and Java leave that to the garbage collector.)", "A.data = B"),
    S("Record its size.", "A.limit = new_limit"), ENS];
  const add = (o) => () => [REQ,
    S("Write `x` into the free slot at the end.", "A.data[A.size] = x"),
    S("One more element.", "A.size += 1"),
    S("If that used the last free slot, grow now, so the next add has room. " + GROW_SAY[o.grow], "if A.size == A.limit", [S("", `uba_resize(A, ${GROW[o.grow]})`)]),
    ENS];
  const rem = (o) => () => [REQ,
    S("**Requires**: there is an element to remove.", "assert A.size > 0"),
    S("One element fewer: the last one is `A.data[A.size]` now.", "A.size -= 1"),
    S("Save it.", "let x = A.data[A.size]"),
    ...(o.shrink === "quarter" ? [S("Only a quarter of the block is used: halve it. (Waiting until ¼ means the next resize, either way, is many operations away.)",
      "if A.limit >= 4 and A.size <= A.limit / 4", [S("", "uba_resize(A, A.limit / 2)")])]
      : o.shrink === "half" ? [S("Less than half of the block is used: halve it. (Right at that boundary, add and remove will resize every time.)",
        "if A.limit >= 2 and A.size < A.limit / 2", [S("", "uba_resize(A, A.limit / 2)")])]
        : []),
    ENS, S("Return it.", "return x")];

  // ---------------------------------------------------------------- the struct, per language
  function decl(c) {
    const g = c.generic, L = c.lang;
    const say = {
      size: "`size`: how many elements there are.",
      limit: "`limit`: how many slots the block has. It starts at 1: an empty array with one free slot.",
      data: "`data`: the block itself, `limit` slots.",
    };
    if (L === "py") {
      return [{ say: g ? "Declare the class `UnboundedArray[T]`." : "Declare the class `UnboundedArray`.", code: [g ? "class UnboundedArray[T]:" : "class UnboundedArray:"], body: [
        { say: "The constructor makes an empty array.", code: ["def __init__(self) -> None:"], body: [
          { say: say.size, code: ["self.size = 0"] }, { say: say.limit, code: ["self.limit = 1"] },
          { say: say.data + ` Each slot holds a placeholder (\`${g ? "None" : "0"}\`) until it's written.`, code: [`self.data = [${g ? "None" : "0"}] * self.limit`] }] }] }];
    }
    if (L === "cpp") {
      return [
        { say: "At the top of the file: `<vector>`, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <iostream>", "#include <cassert>"] },
        { say: g ? "Declare the struct, a template over `T`." : "Declare the struct `UnboundedArray`.", code: g ? ["template <typename T>", "struct UnboundedArray {"] : ["struct UnboundedArray {"], close: ["};"], body: [
          { say: say.size, code: ["int size = 0;"] }, { say: say.limit, code: ["int limit = 1;"] },
          { say: say.data + " A `std::vector` used only as a fixed block: we never `push_back`. Growing it is our job.", code: [`std::vector<${c.T}> data = std::vector<${c.T}>(1);`] }] }];
    }
    if (L === "java") {
      return [{ say: (g ? "Declare the class `UnboundedArray<T>`." : "Declare the class `UnboundedArray`.") + " Every function goes inside it, as a `static` method.",
        code: [g ? "class UnboundedArray<T> {" : "class UnboundedArray {"], close: ["}"], deferClose: true, cls: O, body: [
          { say: say.size, code: ["int size = 0;"] }, { say: say.limit, code: ["int limit = 1;"] },
          { say: say.data + (g ? " Java can't create a `T[]` directly: make an `Object[]` and cast it (the compiler warns the cast is unchecked; the array only ever holds `T`s)." : ""),
            code: [g ? "T[] data = (T[]) new Object[1];" : "int[] data = new int[1];"] }] }];
    }
    const T = c.T;
    return [
      { say: "At the very top: allow capital variable names (the array is called `A`).", code: ["#![allow(non_snake_case)]"] },
      { say: g ? "Declare the struct `UnboundedArray<T>`." : "Declare the struct `UnboundedArray`.", code: [g ? "struct UnboundedArray<T> {" : "struct UnboundedArray {"], close: ["}"], body: [
        { say: say.size, code: ["size: usize,"] }, { say: say.limit, code: ["limit: usize,"] },
        { say: say.data + " A `Vec` used only as a fixed block: we never `push` onto it. Growing it is our job.", code: [`data: Vec<${T}>,`] }] },
      { say: "A constructor `UnboundedArray::new()`: no elements, one free slot." + (g ? " Filling a block needs `T: Clone + Default`." : ""),
        code: [g ? "impl<T: Clone + Default> UnboundedArray<T> {" : "impl UnboundedArray {"], close: ["}"], body: [
          { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: [`UnboundedArray { size: 0, limit: 1, data: vec![${g ? "T::default()" : "0"}; 1] }`] }] }] },
    ];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const growText = { double: "twice as big", half: "half as big again (rounding up)", plus: "with 4 more slots" }[o.grow];
    return [
      { id: "uba", title: "The array and its contracts",
        goal: "An unbounded array `A` keeps its elements in a fixed block `A.data` of `A.limit` slots, using the first `A.size` of them: the segment `A.data[0, A.size)`. " +
          "One slot always stays free (`size < limit`). Write the struct, its contract {fn:is_uba(A)}, and {fn:uba_len(A)}.",
        decl: (c) => decl(c),
        fns: [fn("is_uba", isUba, "Write {fn:is_uba}, the representation invariant."), fn("uba_len", len, "Write {fn:uba_len}.")],
        contractNote: true, tests: ["uba"] },
      { id: "get_set", title: "Get and set",
        goal: "{fn:uba_get(A, i)} and {fn:uba_set(A, i, x)} read and write element `i`. They **require** `0 <= i < A.size`: the slots after `size` hold no elements. O(1).",
        fns: [fn("uba_get", get, "Write {fn:uba_get}."), fn("uba_set", set, "Write {fn:uba_set}.")], tests: ["uba", "get_set"] },
      { id: "resize", title: "Resize",
        goal: "{fn:uba_resize(A, new_limit)} moves the elements into a new block of `new_limit` slots: allocate it, copy `A.data[0, A.size)`, switch. O(size). " +
          "It's the only expensive operation, so how often it runs decides the cost of everything else.",
        fns: [fn("uba_resize", resize, "Write the header: {fn:uba_resize} takes the array and the new number of slots.")], tests: ["uba", "get_set", "resize"] },
      { id: "add", title: "Add",
        goal: `{fn:uba_add(A, x)} writes \`x\` into the free slot at the end. If that fills the block, it grows into one ${growText}. ` +
          (o.grow === "plus" ? "Growing by a constant means a resize every 4 adds, each copying everything: n adds cost O(n²), O(n) each." :
            "Each resize copies everything, but resizes get rarer as the array grows: n adds copy fewer than " + (o.grow === "double" ? "2n" : "3n") + " elements in total, **amortized** O(1) per add."),
        fns: [fn("uba_add", add(o), "Write the header: {fn:uba_add} takes the array and the new element.")], tests: ["uba", "get_set", "resize", "add"] },
      { id: "rem", title: "Remove",
        goal: "{fn:uba_rem(A)} removes and returns the last element. It **requires** a non-empty array. " + {
          quarter: "When only a quarter of the block is in use, halve it: memory comes back, and after shrinking the block is half full, so neither threshold is close. O(1) amortized.",
          half: "When less than half the block is in use, halve it. That shrinks early, and right at the boundary every add grows and every remove shrinks: O(n) each (thrashing).",
          never: "It never shrinks: O(1), but the block keeps the size of the largest the array ever was.",
        }[o.shrink],
        fns: [fn("uba_rem", rem(o), "Write the header: {fn:uba_rem} returns the removed element.")], tests: ["uba", "get_set", "resize", "add", "rem"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const VALS = { int: [10, 20, 30, 40, 50, 60, 70, 80, 90], generic: ["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "jay"] };

  function showFn() {
    const check = (c, msg) => S("", `if not (${c})`, [S("", `print " BROKEN: ${msg}"`), S("", "println"), S("", "return")]);
    return [S("", 'print "["'), S("", "let m = A.size"), S("", "if m > len(A.data)", [S("", "m = len(A.data)")]),
      S("", "for k = 0 to m", [S("", "if k > 0", [S("", 'print ", "')]), S("", "print A.data[k]")]),
      S("", 'print "] size="'), S("", "print A.size"), S("", 'print " limit="'), S("", "print A.limit"),
      check("A.size < A.limit", "size must stay below limit (one free slot)"), check("len(A.data) == A.limit", "data must have exactly limit slots"),
      S("", "println")];
  }

  function breaks(o) {
    return [["A.size = A.limit", "A.size equals A.limit"], ["A.limit += 1", "A.limit is not the block's length"],
      ["A.data = alloc(A.limit + 1)", "the block has one slot too many"], ...(o.lang === "rs" ? [] : [["A.size = 0 - 1", "A.size is negative"]])];
  }

  // a model of the chosen policies: the exact size, limit and elements after every operation
  function model(o) {
    const st = { size: 0, limit: 1, data: [undefined] };
    const resize = (nl) => { const B = new Array(nl); for (let i = 0; i < st.size; i++) B[i] = st.data[i]; st.data = B; st.limit = nl; };
    return {
      st,
      show: () => `[${st.data.slice(0, st.size).join(", ")}] size=${st.size} limit=${st.limit}`,
      add: (x) => { st.data[st.size] = x; st.size += 1; if (st.size === st.limit) resize(growJS[o.grow](st.limit)); },
      rem: () => {
        st.size -= 1;
        const x = st.data[st.size];
        if (o.shrink === "quarter" && st.limit >= 4 && st.size <= Math.floor(st.limit / 4)) resize(Math.floor(st.limit / 2));
        if (o.shrink === "half" && st.limit >= 2 && st.size < Math.floor(st.limit / 2)) resize(Math.floor(st.limit / 2));
        return x;
      },
      resize,
    };
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const B = breaks(o);
    const shown = (st, call, label) => ({ ir: [say(`${label}: `), S("", call), S("", "show(A)")], out: `${label}: ${st.m.show()}` });
    return {
      new: (x, st) => { st.m = model(o); return { ir: [S("", "make UnboundedArray A"), say("new array: "), S("", "show(A)")], out: `new array: ${st.m.show()}` }; },
      fill: (xs, st) => { xs.forEach((y) => st.m.add(y)); return { ir: xs.map((y) => S("", `ref_uba_add(A, ${lit(y)})`)) }; },
      is_uba: () => C.ops.bool(o, "is_uba(A)", `${nm("is_uba")}(A)`, true),
      uba_len: (x, st) => C.ops.value(o, "uba_len(A)", `${nm("uba_len")}(A)`, st.m.st.size),
      get: (i, st) => C.ops.value(o, `uba_get(A, ${i})`, `${nm("uba_get")}(A, ${i})`, st.m.st.data[i]),
      set: ([i, x], st) => { st.m.st.data[i] = x; return shown(st, `uba_set(A, ${i}, ${lit(x)})`, `${nm("uba_set")}(A, ${i}, ${x})`); },
      resize: (n, st) => { st.m.resize(n); return shown(st, `uba_resize(A, ${n})`, `${nm("uba_resize")}(A, ${n})`); },
      add: (x, st) => { st.m.add(x); return shown(st, `uba_add(A, ${lit(x)})`, `${nm("uba_add")}(A, ${x})`); },
      rem: (x, st) => {
        const y = st.m.rem();
        return { ir: [say(`${nm("uba_rem")}(A) = `), S("", "print uba_rem(A)"), say(": "), S("", "show(A)")], out: `${nm("uba_rem")}(A) = ${y}: ${st.m.show()}` };
      },
      // many adds (or removes) at once, then one look: the policy's long-run behaviour
      add_n: ([n, vals], st) => {
        for (let k = 0; k < n; k++) st.m.add(vals[k % vals.length]);
        return { ir: [...Array.from({ length: n }, (_, k) => S("", `uba_add(A, ${lit(vals[k % vals.length])})`)), say(`after ${n} adds: `), S("", "show(A)")],
          out: `after ${n} adds: ${st.m.show()}` };
      },
      rem_n: (n, st) => {
        for (let k = 0; k < n; k++) st.m.rem();
        return { ir: [...Array.from({ length: n }, () => S("", "uba_rem(A)")), say(`after ${n} removes: `), S("", "show(A)")], out: `after ${n} removes: ${st.m.show()}` };
      },
      break: (k) => ({ ir: [S("", B[k][0]), say(`after ${B[k][1]}: ${nm("is_uba")}(A) = `), S("", "printb is_uba(A)"), S("", "println")],
        out: `after ${B[k][1]}: ${nm("is_uba")}(A) = false` }),
    };
  }

  function testGroups(names, o, seed) {
    const v = VALS[o.type];
    const [a, b, c, d, e, f] = v;
    const G = {
      uba: [
        [["new"], ["is_uba"], ["uba_len"]],
        [["new"], ["fill", [a, b, c]], ["is_uba"], ["uba_len"]],
        ...breaks(o).map((_, k) => [["new"], ["fill", [a, b, c]], ["break", k]]),
      ],
      get_set: [[["new"], ["fill", [a, b, c]], ["get", 0], ["get", 2], ["set", [1, d]], ["get", 1], ["set", [0, e]], ["get", 0]]],
      resize: [[["new"], ["fill", [a, b]], ["resize", 8], ["resize", 3], ["resize", 5]]],
      add: [[["new"], ...v.slice(0, 7).map((x) => ["add", x])], [["new"], ["add_n", [20, v]]]],
      rem: [
        [["new"], ["add_n", [8, v]], ...Array.from({ length: 8 }, () => ["rem"]), ["add", a], ["add", b]],
        [["new"], ["add_n", [20, v]], ["rem_n", 15]],
      ],
    };
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      const ops = [["new"]];
      let n = 0;
      for (let i = 0; i < 70; i++) {
        const x = r();
        const pick = () => v[Math.floor(r() * v.length)];
        if (x < 0.5) { ops.push(["add", pick()]); n++; }
        else if (x < 0.75 && n) { ops.push(["rem"]); n--; }
        else if (x < 0.87 && n) ops.push(["get", Math.floor(r() * n)]);
        else if (n) ops.push(["set", [Math.floor(r() * n), pick()]]);
      }
      groups.push(ops, [["new"], ["add_n", [33, v]], ["rem_n", 30], ["add_n", [5, v]]]);
    }
    return groups;
  }

  const refFns = (o, names) => (names.includes("uba")
    ? [{ name: "uba_resize", steps: resize() }, { name: "uba_add", steps: add(o)() }, { name: "show", steps: showFn() }] : []);

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const what = {
      is_uba: "size < limit, and data has exactly limit slots",
      uba_len: "the number of elements, O(1)",
      uba_get: "element i (requires 0 <= i < size), O(1)",
      uba_set: "overwrite element i (requires 0 <= i < size), O(1)",
      uba_resize: "move the elements into a new block of new_limit slots (requires size < new_limit)",
      uba_add: `add x at the end; when that fills the block, grow it ${{ double: "×2", half: "to limit + (limit + 1) / 2", plus: "by 4 slots" }[o.grow]}`,
      uba_rem: `remove and return the last element (requires a non-empty array)${{ quarter: "; halve the block when limit >= 4 and size <= limit / 4",
        half: "; halve the block when limit >= 2 and size < limit / 2", never: "; never shrink" }[o.shrink]}`,
    };
    return [
      `Build an **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**UnboundedArray.** Fields \`size\`, \`limit\` and \`data\` (a block of \`limit\` slots). ${{ py: "`UnboundedArray()`", cpp: "`UnboundedArray A;`", java: "`new UnboundedArray()`", rs: "`UnboundedArray::new()`" }[L]} ` +
        "makes an empty array with `limit` 1.", "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_uba", L)}(A)\` at the start and end of every operation. ${L === "java" ? "Put the functions inside `class UnboundedArray` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints `size` and `limit` after every operation, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "unbounded-arrays", num: "13.07", title: "Unbounded arrays", heading: "Build an unbounded array, step by step", diagramLabel: "Your array",
    OPTIONS, DEFAULTS, FNS, STRUCTS, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

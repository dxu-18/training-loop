// 13.11 Heaps: a binary heap in an array (CMU 15-122's heaps), min or max, 0- or 1-indexed, in Python, C++, Java or
// Rust, ending with bottom-up heapify and heapsort.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type that can be ordered: a type parameter T (Comparable in Java, PartialOrd in Rust)." } },
    { id: "order", label: "Order", choices: [["min", "min-heap"], ["max", "max-heap"]],
      help: { min: "Every parent is ≤ its children, so the root is the smallest element.", max: "Every parent is ≥ its children, so the root is the largest element." } },
    { id: "index", label: "Indexing", choices: [["0", "0-indexed"], ["1", "1-indexed"]],
      help: { 0: "The root is H.data[0]; the children of i are 2i + 1 and 2i + 2, its parent (i - 1) / 2.",
        1: "The root is H.data[1] and slot 0 stays unused (as in 15-122 and CLRS): the children of i are 2i and 2i + 1, its parent i / 2." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", order: "min", index: "0" };
  const one = (o) => o.index === "1", min = (o) => o.order === "min";
  const R0 = (o) => (one(o) ? 1 : 0);                          // the root's index
  const END = (o) => (one(o) ? "H.size + 1" : "H.size");      // one past the last element
  const LAST = (o) => (one(o) ? "H.size" : "H.size - 1");     // the last element
  const describe = (o) => `${min(o) ? "min" : "max"}-heap of ${o.type === "int" ? "int" : "generic"} elements, ${o.index}-indexed, in ${DL.LANGS[o.lang].name}`;
  // (a generic Java block is made as Comparable[]: the functions that order its elements read it as one)
  const ctxExtra = () => ({});

  function complexity(o) {
    return [
      { id: "add", label: "Add", big: "O(log n)", note: "sift up: at most one swap per level" },
      { id: "rem", label: `Remove the ${min(o) ? "minimum" : "maximum"}`, big: "O(log n)", note: "the last element moves to the root, then sifts down" },
      { id: "peek", label: `Peek at the ${min(o) ? "minimum" : "maximum"}`, big: "O(1)", note: "it's always at the root" },
      { id: "heapify", label: "Heapify n elements", big: "O(n)", note: "bottom-up: most nodes are near the bottom and sift down only a little" },
      { id: "heapsort", label: "Heapsort", big: "O(n log n)", note: "heapify, then n removes" },
      { id: "height", label: "Height", big: "O(log n)", note: "a complete tree: every level full except the last" },
      { id: "space", label: "Space", big: "O(n)", note: "one array, no pointers: the shape is implied by the indices" },
      { id: "find", label: "Find any element", big: "O(n)", note: "a heap only knows where the extreme is", extra: true },
      { id: "is_heap", label: "Contract is_heap", big: "O(n)", note: "check every element against its parent" },
    ];
  }

  // ---------------------------------------------------------------- the model: the exact array every test expects
  function model(o) {
    const r = R0(o);
    const st = { size: 0, cap: 0, len: 0, d: [] };
    const above = (a, b) => (min(o) ? a <= b : a >= b);
    const par = (i) => (one(o) ? Math.floor(i / 2) : Math.floor((i - 1) / 2));
    const lc = (i) => (one(o) ? 2 * i : 2 * i + 1), rc = (i) => lc(i) + 1;
    const end = () => st.size + r;
    const swap = (i, j) => { const t = st.d[i]; st.d[i] = st.d[j]; st.d[j] = t; };
    const siftUp = (i) => { let c = i; while (c > r) { const p = par(c); if (above(st.d[p], st.d[c])) return; swap(p, c); c = p; } };
    const siftDown = (i) => {
      let c = i;
      while (lc(c) < end()) {
        let s = lc(c);
        if (rc(c) < end() && above(st.d[rc(c)], st.d[s])) s = rc(c);
        if (above(st.d[c], st.d[s])) return;
        swap(c, s); c = s;
      }
    };
    const ok = () => {
      if (st.size > st.cap || st.size < 0 || st.len !== st.cap + r) return false;
      for (let i = r + 1; i < end(); i++) if (!above(st.d[par(i)], st.d[i])) return false;
      return true;
    };
    return {
      st, above, par, lc, rc, siftUp, siftDown, swap, ok,
      init: (cap) => { st.size = 0; st.cap = cap; st.len = cap + r; st.d = new Array(cap + r).fill(null); },
      add: (x) => { if (one(o)) { st.size += 1; st.d[st.size] = x; siftUp(st.size); } else { st.d[st.size] = x; st.size += 1; siftUp(st.size - 1); } },
      rem: () => {
        const x = st.d[r];
        if (one(o)) { st.d[1] = st.d[st.size]; st.size -= 1; } else { st.size -= 1; st.d[0] = st.d[st.size]; }
        if (st.size > 0) siftDown(r);
        return x;
      },
      heapify: () => { let i = Math.floor(st.size / 2); if (one(o)) { while (i > 0) { siftDown(i); i -= 1; } } else { while (i > 0) { i -= 1; siftDown(i); } } },
      show: () => `[${st.d.slice(r, Math.max(r, Math.min(end(), st.len))).join(", ")}] size=${st.size} capacity=${st.cap}` + (ok() ? "" : " BROKEN: is_heap(H) is false"),
    };
  }

  function diagram(o) {
    const m = model(o);
    const v = o.type === "int" ? [50, 30, 70, 20, 40, 60, 10] : ["mole", "fox", "seal", "cat", "hen", "pig", "ant"];
    m.init(7);
    v.forEach((x) => m.add(x));
    const r = R0(o);
    // the tree the indices describe, drawn by in-order column and depth; then the array itself
    const order = [];
    const walk = (i, d) => { if (i < m.st.size + r) { walk(m.lc(i), d + 1); order.push([i, d]); walk(m.rc(i), d + 1); } };
    walk(r, 0);
    const rows = [];
    for (let d = 0; d <= 2; d++) {
      rows.push([{ label: d === 0 ? "root" : `depth ${d}` }, ...order.map(([i, id]) => ({ box: String(m.st.d[i]), tags: [], cls: id === d ? "" : "spacer" }))]);
    }
    const cells = [{ label: "H.data" }];
    if (one(o)) cells.push({ box: "_", tags: ["0"], cls: "dummy" });
    for (let i = r; i < m.st.size + r; i++) cells.push({ box: String(m.st.d[i]), tags: [String(i)] });
    rows.push(cells);
    return { rows, caption: `The tree is stored row by row in H.data: no pointers. ${one(o) ? "The children of i are 2i and 2i + 1, its parent i / 2." : "The children of i are 2i + 1 and 2i + 2, its parent (i - 1) / 2."} ` +
      `Every parent is ${min(o) ? "≤" : "≥"} its children, so the ${min(o) ? "smallest" : "largest"} element is at the root.` };
  }

  // ---------------------------------------------------------------- specs
  const O = "Heap", M = "mut:Heap", R = "ref:Heap", ORD = ["PartialOrd"];
  const FNS = {
    heap_size: { params: [["H", R]], ret: "int", owner: O },
    heap_full: { params: [["H", R]], ret: "bool", owner: O },
    parent: { params: [["i", "int"]], ret: "int", owner: O },
    left: { params: [["i", "int"]], ret: "int", owner: O },
    right: { params: [["i", "int"]], ret: "int", owner: O },
    ok_above: { params: [["H", R], ["i", "int"], ["j", "int"]], ret: "bool", owner: O, bounds: ORD },
    swap: { params: [["H", M], ["i", "int"], ["j", "int"]], ret: "void", owner: O },
    is_heap: { params: [["H", R]], ret: "bool", owner: O, bounds: ORD, contract: true },
    sift_up: { params: [["H", M], ["i", "int"]], ret: "void", owner: O, bounds: ORD },
    heap_add: { params: [["H", M], ["x", "T"]], ret: "void", owner: O, bounds: ORD },
    heap_peek: { params: [["H", R]], ret: "Tpeek", owner: O, bounds: ORD },
    sift_down: { params: [["H", M], ["i", "int"]], ret: "void", owner: O, bounds: ORD },
    heap_rem: { params: [["H", M]], ret: "T", owner: O, bounds: ORD },
    heapify: { params: [["H", M]], ret: "void", owner: O, bounds: ORD },
    heapsort: { params: [["A", "ArrMut"]], ret: "void", owner: O, bounds: ["PartialOrd", "Clone", "Default"] },
    show: { params: [["H", R]], ret: "void", harness: true, bounds: ["std::fmt::Display", "PartialOrd"] },
    put: { params: [["H", M], ["i", "int"], ["x", "T"]], ret: "void", harness: true },
  };
  const STRUCTS = { Heap: { fields: { size: "int", capacity: "int", data: "ArrV" } } };

  // ---------------------------------------------------------------- the functions
  const REQ = S("**Requires**: the heap is valid when the function starts.", "assert is_heap(H)");
  const ENS = S("**Ensures**: still a valid heap when it returns.", "assert is_heap(H)");
  const ret = (e) => [S("", `return ${e}`)];
  // **Requires** i is an element's index (Rust's usize is never negative, so 0-indexed Rust checks only the top)
  const inRange = (o) => (one(o) ? [S("**Requires**: `i` is an element's index, in [1, size + 1).", "assert 1 <= i and i < H.size + 1")]
    : [{ ...S("**Requires**: `i` is an element's index, in [0, size).", "assert 0 <= i and i < H.size"), only: ["py", "cpp", "java"] },
      { ...S("**Requires**: `i` is an element's index: below `size` (a `usize` is never negative).", "assert i < H.size"), only: ["rs"] }]);
  const parentFn = (o) => () => [S(one(o) ? "Integer division: the parent of 2i and 2i + 1 is i." : "Integer division: the parent of 2i + 1 and 2i + 2 is i.", one(o) ? "return i / 2" : "return (i - 1) / 2")];
  const leftFn = (o) => () => [S("", one(o) ? "return 2 * i" : "return 2 * i + 1")];
  const rightFn = (o) => () => [S("Right after the left child.", one(o) ? "return 2 * i + 1" : "return 2 * i + 2")];
  const okAbove = (o) => () => [S(min(o) ? "In a min-heap the smaller element goes on top: `H.data[i]` may sit above `H.data[j]` if it's ≤. (Equal elements may go either way.)"
    : "In a max-heap the bigger element goes on top: `H.data[i]` may sit above `H.data[j]` if it's ≥. (Equal elements may go either way.)", min(o) ? "return H.data[i] <= H.data[j]" : "return H.data[i] >= H.data[j]")];
  const swap = () => [
    { say: "Python swaps with one tuple assignment: both values on the right are read before either is written.", code: { py: ["H.data[i], H.data[j] = H.data[j], H.data[i]"] }, only: ["py"] },
    { say: "Rust can't move an element out of the vector into a temporary, so use the vector's own `swap`.", code: { rs: ["H.data.swap(i, j);"] }, only: ["rs"] },
    { ...S("Save `H.data[i]`…", "let t = H.data[i]"), only: ["cpp", "java"] },
    { ...S("…overwrite it with `H.data[j]`…", "H.data[i] = H.data[j]"), only: ["cpp", "java"] },
    { ...S("…and put the saved element in `H.data[j]`.", "H.data[j] = t"), only: ["cpp", "java"] }];
  const isHeap = (o) => () => [
    S("The **shape**: at most `capacity` elements…", "if H.size > H.capacity", ret("false")),
    { ...S("…and never fewer than none. (A `usize` can't be negative, so Rust skips this.)", "if H.size < 0", ret("false")), only: ["py", "cpp", "java"] },
    S(one(o) ? "`data` has `capacity + 1` slots (slot 0 is unused)." : "`data` really has `capacity` slots.", one(o) ? "if len(H.data) != H.capacity + 1" : "if len(H.data) != H.capacity", ret("false")),
    S("The **order**: every element except the root may sit below its parent.", `for i = ${R0(o) + 1} to ${END(o)}`, [S("", "if not ok_above(H, parent(i), i)", ret("false"))]),
    S("", "return true")];
  const siftUp = (o) => () => [...inRange(o),
    S("`c` is where the element is now; it moves up from `i`.", "let c = i"),
    S("While it isn't at the root…", `while c > ${R0(o)}`, [
      S("…look at its parent.", "let p = parent(c)"),
      S("If the parent may sit above it, the order is restored: stop.", "if ok_above(H, p, c)", [S("", "return")]),
      S("Otherwise swap the two…", "swap(H, p, c)"),
      S("…and carry on from the parent's position.", "c = p")])];
  const add = (o) => () => [REQ,
    S("**Requires**: there is room.", "assert not heap_full(H)"),
    ...(one(o) ? [S("One more element: the first free slot is `H.data[H.size + 1]`, the next spot in the bottom row of the tree.", "H.size += 1"),
      S("Put `x` there…", "H.data[H.size] = x"), S("…and sift it up to where it belongs.", "sift_up(H, H.size)")]
      : [S("Put `x` in the first free slot, `H.data[H.size]`: the next spot in the bottom row of the tree.", "H.data[H.size] = x"),
        S("One more element.", "H.size += 1"), S("Sift it up to where it belongs.", "sift_up(H, H.size - 1)")]),
    ENS];
  const peek = (o) => () => [REQ, S("**Requires**: a non-empty heap.", "assert H.size > 0"),
    S((c) => `The ${min(o) ? "smallest" : "largest"} element is always at the root${c.lang === "rs" && c.generic ? " (return a reference `&T`)" : ""}.`, `return H.data[${R0(o)}]`)];
  const siftDown = (o) => () => [...inRange(o),
    S("`c` is where the element is now; it moves down from `i`.", "let c = i"),
    S("While it has a left child (no left child means no children at all)…", `while left(c) < ${END(o)}`, [
      S("…compare it with that child…", "let s = left(c)"),
      S("…or with the right child, if there is one and it may sit above the left one.", `if right(c) < ${END(o)} and ok_above(H, right(c), s)`, [S("", "s = right(c)")]),
      S("If the element may sit above that child, the order is restored: stop.", "if ok_above(H, c, s)", [S("", "return")]),
      S("Otherwise swap them…", "swap(H, c, s)"),
      S("…and carry on from the child's position.", "c = s")])];
  const rem = (o) => () => [REQ, S("**Requires**: a non-empty heap.", "assert H.size > 0"),
    S(`Save the root, the ${min(o) ? "smallest" : "largest"} element.`, `let x = H.data[${R0(o)}]`),
    ...(one(o) ? [S("Move the last element to the root (the tree keeps its shape)…", "H.data[1] = H.data[H.size]"), S("…one element fewer…", "H.size -= 1")]
      : [S("One element fewer: the last one is `H.data[H.size]` now…", "H.size -= 1"), S("…move it to the root (the tree keeps its shape)…", "H.data[0] = H.data[H.size]")]),
    S("…and, if anything is left, sift it down to where it belongs.", "if H.size > 0", [S("", `sift_down(H, ${R0(o)})`)]),
    ENS, S("", "return x")];
  const heapify = (o) => () => [
    S("**Requires**: the elements fit.", "assert H.size <= H.capacity"),
    S(one(o) ? "The last element with a child is `H.size / 2` (the parent of the last element)." : "The elements with children are the first `H.size / 2`: indices [0, H.size / 2).", "let i = H.size / 2"),
    ...(one(o) ? [S("Sift each of them down, from the last one back to the root. Each subtree below is a heap by the time its root sifts down.", "while i > 0", [
      S("", "sift_down(H, i)"), S("", "i -= 1")])]
      : [S("Sift each of them down, from the last one back to the root. Each subtree below is a heap by the time its root sifts down.", "while i > 0", [
        S("", "i -= 1"), S("", "sift_down(H, i)")])]),
    ENS];
  const heapsort = (o) => () => [
    S("A heap with room for every element.", "make Heap H(len(A))"),
    S("Copy the elements in, in whatever order they are…", "for i = 0 to len(A)", [S("", one(o) ? "H.data[i + 1] = A[i]" : "H.data[i] = A[i]")]),
    S("", "H.size = len(A)"),
    S("…and turn them into a heap, all at once: O(n).", "heapify(H)"),
    min(o) ? S("Remove them one by one, smallest first, into `A` from the front: `A` ends up sorted.", "for i = 0 to len(A)", [S("", "A[i] = heap_rem(H)")])
      : S("Remove them one by one, largest first, into `A` from the back: `A` ends up sorted, smallest first.", "for i = 0 to len(A)", [S("", "A[len(A) - 1 - i] = heap_rem(H)")])];

  // ---------------------------------------------------------------- the struct, per language
  function decl(c, o) {
    const g = c.generic, L = c.lang, T = c.T, I = one(o);
    const slots = I ? "capacity + 1" : "capacity";
    const say = {
      size: "`size`: how many elements there are.",
      capacity: "`capacity`: how many elements fit. It's fixed: the constructor takes it.",
      data: `\`data\`: the elements, as a complete binary tree stored row by row${I ? ", from slot 1 (slot 0 stays unused, so it has `capacity + 1` slots)" : ""}.`,
    };
    if (L === "py") {
      return [{ say: g ? "Declare the class `Heap[T]`." : "Declare the class `Heap`.", code: [g ? "class Heap[T]:" : "class Heap:"], body: [
        { say: "The constructor makes an empty heap that can hold `capacity` elements.", code: ["def __init__(self, capacity: int) -> None:"], body: [
          { say: say.size, code: ["self.size = 0"] }, { say: say.capacity, code: ["self.capacity = capacity"] },
          { say: say.data + ` Each slot holds a placeholder (\`${g ? "None" : "0"}\`) until it's written.`, code: [`self.data = [${g ? "None" : "0"}] * ${I ? "(capacity + 1)" : "capacity"}`] }] }] }];
    }
    if (L === "cpp") {
      return [
        { say: "At the top of the file: `<vector>`, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <iostream>", "#include <cassert>"] },
        { say: g ? "Declare the struct, a template over `T`." : "Declare the struct `Heap`.", code: g ? ["template <typename T>", "struct Heap {"] : ["struct Heap {"], close: ["};"], body: [
          { say: say.size, code: ["int size = 0;"] }, { say: say.capacity, code: ["int capacity;"] }, { say: say.data, code: [`std::vector<${T}> data;`] },
          { say: "The constructor sets `capacity` and makes `data` that many slots.", code: [`Heap(int capacity) : capacity(capacity), data(${slots}) {}`] }] }];
    }
    if (L === "java") {
      return [{ say: (g ? "Declare the class `Heap<T>`." : "Declare the class `Heap`.") + " Every function goes inside it, as a `static` method.",
        code: [g ? "class Heap<T> {" : "class Heap {"], close: ["}"], deferClose: true, cls: O, body: [
          { say: say.size, code: ["int size;"] }, { say: say.capacity, code: ["int capacity;"] }, { say: say.data, code: [g ? "T[] data;" : "int[] data;"] },
          { say: "The constructor sets `capacity` and makes the array." + (g ? " Java can't create a `T[]`: make a `Comparable[]` (the elements are ordered with `compareTo`) and cast it; the compiler warns the cast is unchecked." : ""),
            code: ["Heap(int capacity) {"], close: ["}"], body: [{ say: "", code: ["this.capacity = capacity;"] },
              { say: "", code: [g ? `this.data = (T[]) new Comparable[${slots}];` : `this.data = new int[${slots}];`] }] }] }];
    }
    return [
      { say: "At the very top: allow capital variable names (the heap is called `H`).", code: ["#![allow(non_snake_case)]"] },
      { say: g ? "Declare the struct `Heap<T>`." : "Declare the struct `Heap`.", code: [g ? "struct Heap<T> {" : "struct Heap {"], close: ["}"], body: [
        { say: say.size, code: ["size: usize,"] }, { say: say.capacity, code: ["capacity: usize,"] }, { say: say.data + " A `Vec` used as a fixed block: we never `push`.", code: [`data: Vec<${T}>,`] }] },
      { say: "A constructor `Heap::new(capacity)`." + (g ? " Filling the block needs `T: Clone + Default`." : ""),
        code: [g ? "impl<T: Clone + Default> Heap<T> {" : "impl Heap {"], close: ["}"], body: [
          { say: "", code: ["fn new(capacity: usize) -> Self {"], close: ["}"], body: [{ say: "", code: [`Heap { size: 0, capacity, data: vec![${g ? "T::default()" : "0"}; ${slots}] }`] }] }] },
    ];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const on = C.contractsOn(o), I = one(o), top = min(o) ? "smallest" : "largest";
    const all = [
      { id: "heap", title: "The heap",
        goal: `A heap is a complete binary tree (every level full except the last, which fills from the left) stored row by row in an array: the root is \`H.data[${R0(o)}]\`` +
          (I ? ", slot 0 stays unused, and the children of `i` are `2i` and `2i + 1`." : ", and the children of `i` are `2i + 1` and `2i + 2`.") +
          " The shape needs no pointers: {fn:parent(i)}, {fn:left(i)} and {fn:right(i)} compute where to go. {fn:heap_size(H)} and {fn:heap_full(H)} report on the capacity.",
        decl: (c) => decl(c, o),
        fns: [fn("heap_size", () => ret("H.size"), "Write {fn:heap_size}."), fn("heap_full", () => [S("Full when every slot is used.", "return H.size == H.capacity")], "Write {fn:heap_full}."),
          fn("parent", parentFn(o), "Write {fn:parent}."), fn("left", leftFn(o), "Write {fn:left}."), fn("right", rightFn(o), "Write {fn:right}.")] },
      { id: "order", title: "Order and swap",
        goal: `The **ordering invariant**: every parent ${min(o) ? "≤" : "≥"} its children, so the ${top} element is at the root. {fn:ok_above(H, i, j)} says whether the element at \`i\` may sit above the one at \`j\`: ` +
          `it's the only place that knows this is a ${min(o) ? "min" : "max"}-heap (flip it and the same code builds the other kind). {fn:swap(H, i, j)} exchanges two elements.`,
        fns: [fn("ok_above", okAbove(o), "Write {fn:ok_above}."), fn("swap", swap, "Write {fn:swap}.")] },
      { id: "contracts", title: "Contracts",
        goal: `{fn:is_heap(H)} checks the shape (\`size\` within \`capacity\`, \`data\` the right length) and the order: every element from index ${R0(o) + 1} on may sit below its parent. Every operation checks it.`,
        fns: [fn("is_heap", isHeap(o), "Write {fn:is_heap}, the representation invariant.")], contractNote: true },
      { id: "add", title: "Sift up and add",
        goal: `{fn:heap_add(H, x)} puts \`x\` in the next free spot of the bottom row, which keeps the shape but may break the order. {fn:sift_up(H, i)} repairs it: while the element ` +
          `may not sit below its parent, swap them. One swap per level: O(log n). {fn:heap_peek(H)} returns the root, the ${top} element.`,
        fns: [fn("sift_up", siftUp(o), "Write {fn:sift_up}."), fn("heap_add", add(o), "Write {fn:heap_add}."), fn("heap_peek", peek(o), "Write {fn:heap_peek}.")] },
      { id: "rem", title: "Sift down and remove",
        goal: `{fn:heap_rem(H)} removes the root: the last element takes its place (keeping the shape), then {fn:sift_down(H, i)} repairs the order: while a child may sit above the element, ` +
          `swap it with the ${top === "smallest" ? "smaller" : "larger"} child. O(log n).`,
        fns: [fn("sift_down", siftDown(o), "Write {fn:sift_down}."), fn("heap_rem", rem(o), "Write {fn:heap_rem}.")] },
      { id: "heapify", title: "Heapify and heapsort",
        goal: "{fn:heapify(H)} turns elements in any order into a heap at once: sift down every element that has children, from the last one back to the root. That's O(n), not O(n log n): " +
          "half the elements are leaves and don't move, and the few near the top are the only ones that travel far. {fn:heapsort(A)} heapifies a copy of `A`, then removes the elements back into `A` in order: O(n log n).",
        fns: [fn("heapify", heapify(o), "Write {fn:heapify}."), fn("heapsort", heapsort(o), "Write {fn:heapsort}; it sorts `A`, smallest first.")] },
    ];
    const ids = all.map((st) => st.id);
    return all.map((st, i) => ({ ...st, tests: ids.slice(0, i + 1) }));
  }

  // ---------------------------------------------------------------- tests
  const VALS = { int: [50, 30, 70, 20, 40, 60, 80, 10, 35, 65, 25, 45], generic: ["mole", "fox", "seal", "cat", "hen", "pig", "yak", "ant", "gnu", "owl", "dog", "ibis"] };
  const BIG = { int: 99, generic: "zebu" }, SMALL = { int: 1, generic: "aaa" };

  function showFn(o) {
    return [S("", 'print "["'), S("", `let m = ${END(o)}`), S("", "if m > len(H.data)", [S("", "m = len(H.data)")]),
      S("", `for k = ${R0(o)} to m`, [S("", `if k > ${R0(o)}`, [S("", 'print ", "')]), S("", "print H.data[k]")]),
      S("", 'print "] size="'), S("", "print H.size"), S("", 'print " capacity="'), S("", "print H.capacity"),
      S("", "if not is_heap(H)", [S("", 'print " BROKEN: is_heap(H) is false"'), S("", "println"), S("", "return")]), S("", "println")];
  }
  const putFn = () => [S("", "H.data[i] = x")];

  // broken heaps for is_heap; the element values are written through the checker's put()
  function breaks(o, m) {
    const bad = (top) => C.lit(min(o) === top ? BIG[o.type] : SMALL[o.type]);
    const last = m.st.size - 1 + R0(o);
    return [[["H.size = H.capacity + 1"], "size is bigger than capacity"], [["H.capacity += 1"], "capacity doesn't match the array"],
      [[`put(H, ${R0(o)}, ${bad(true)})`], "the root is out of order"], [[`put(H, ${last}, ${bad(false)})`], "the last element is out of order with its parent"],
      ...(o.lang === "rs" ? [] : [[["H.size = 0 - 1"], "size is negative"]])];
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const shown = (st, call, label) => ({ ir: [say(`${label}: `), S("", call), S("", "show(H)")], out: `${label}: ${st.m.show()}` });
    return {
      new: (cap, st) => { st.m = model(o); st.m.init(cap); return { ir: [S("", `make Heap H(${cap})`), say(`new heap (capacity ${cap}): `), S("", "show(H)")], out: `new heap (capacity ${cap}): ${st.m.show()}` }; },
      fill: (xs, st) => { xs.forEach((y) => st.m.add(y)); return { ir: xs.map((y) => S("", `ref_heap_add(H, ${lit(y)})`)) }; },
      size: (x, st) => C.ops.value(o, "heap_size(H)", `${nm("heap_size")}(H)`, st.m.st.size),
      full: (x, st) => C.ops.bool(o, "heap_full(H)", `${nm("heap_full")}(H)`, st.m.st.size === st.m.st.cap),
      idx: ([f, i], st) => C.ops.value(o, `${f}(${i})`, `${nm(f)}(${i})`, { parent: st.m.par, left: st.m.lc, right: st.m.rc }[f](i)),
      ok_above: ([i, j], st) => C.ops.bool(o, `ok_above(H, ${i}, ${j})`, `${nm("ok_above")}(H, ${i}, ${j})`, st.m.above(st.m.st.d[i], st.m.st.d[j])),
      swap: ([i, j], st) => { st.m.swap(i, j); return shown(st, `swap(H, ${i}, ${j})`, `${nm("swap")}(H, ${i}, ${j})`); },
      is_heap: (x, st) => C.ops.bool(o, "is_heap(H)", `${nm("is_heap")}(H)`, st.m.ok()),
      // an element written past the end without sifting (the first half of an add), then sift_up on its own
      place: (x, st) => {
        const i = st.m.st.size + R0(o);
        st.m.st.d[i] = x; st.m.st.size += 1;
        return { ir: [S("", `put(H, ${i}, ${lit(x)})`), S("", "H.size += 1"), say(`${x} written at ${i} without sifting: `), S("", "show(H)")], out: `${x} written at ${i} without sifting: ${st.m.show()}` };
      },
      sift_up: (i, st) => { st.m.siftUp(i); return shown(st, `sift_up(H, ${i})`, `${nm("sift_up")}(H, ${i})`); },
      add: (x, st) => { st.m.add(x); return shown(st, `heap_add(H, ${lit(x)})`, `${nm("heap_add")}(H, ${x})`); },
      peek: (x, st) => C.ops.value(o, "heap_peek(H)", `${nm("heap_peek")}(H)`, st.m.st.d[R0(o)]),
      // the root overwritten (the first half of a remove), then sift_down on its own
      put_root: (x, st) => {
        st.m.st.d[R0(o)] = x;
        return { ir: [S("", `put(H, ${R0(o)}, ${lit(x)})`), say(`root overwritten with ${x}: `), S("", "show(H)")], out: `root overwritten with ${x}: ${st.m.show()}` };
      },
      sift_down: (i, st) => { st.m.siftDown(i); return shown(st, `sift_down(H, ${i})`, `${nm("sift_down")}(H, ${i})`); },
      rem: (x, st) => {
        const y = st.m.rem();
        return { ir: [say(`${nm("heap_rem")}(H) = `), S("", "print heap_rem(H)"), say(": "), S("", "show(H)")], out: `${nm("heap_rem")}(H) = ${y}: ${st.m.show()}` };
      },
      // elements written in the given order, not a heap yet
      raw: (xs, st) => {
        xs.forEach((y, k) => { st.m.st.d[k + R0(o)] = y; });
        st.m.st.size = xs.length;
        return { ir: [...xs.map((y, k) => S("", `put(H, ${k + R0(o)}, ${lit(y)})`)), S("", `H.size = ${xs.length}`), say("written in this order: "), S("", "show(H)")],
          out: `written in this order: ${st.m.show()}` };
      },
      heapify: (x, st) => { st.m.heapify(); return shown(st, "heapify(H)", `${nm("heapify")}(H)`); },
      sort: (xs) => {
        const sorted = [...xs].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
        return { ir: [S("", "block", [S("", `let A = [${xs.map(lit).join(", ")}]`), S("", "heapsort(A)"), say(`${nm("heapsort")}([${xs.join(", ")}]) = [`),
          S("", "for k = 0 to len(A)", [S("", "if k > 0", [S("", 'print ", "')]), S("", "print A[k]")]), S("", 'println "]"')])],
          out: `${nm("heapsort")}([${xs.join(", ")}]) = [${sorted.join(", ")}]` };
      },
      break: (k, st) => {
        const B = breaks(o, st.m);
        return { ir: [...B[k][0].map((x) => S("", x)), say(`after ${B[k][1]}: ${nm("is_heap")}(H) = `), S("", "printb is_heap(H)"), S("", "println")],
          out: `after ${B[k][1]}: ${nm("is_heap")}(H) = false` };
      },
    };
  }

  function testGroups(names, o, seed) {
    const v = VALS[o.type], r = R0(o);
    const filled = (xs, cap) => { const m = model(o); m.init(cap); xs.forEach((x) => m.add(x)); return m; };
    const G = {
      heap: [[["new", 7], ["size"], ["full"], ...[1, 2, 3, 4, 5, 6].map((i) => ["idx", ["parent", i + r]]), ...[0, 1, 2].map((i) => ["idx", ["left", i + r]]),
        ...[0, 1, 2].map((i) => ["idx", ["right", i + r]])], [["new", 3], ["fill", v.slice(0, 3)], ["size"], ["full"]]],
      order: [[["new", 7], ["fill", [v[3], v[0], v[3], v[1]]], ["ok_above", [r, r + 1]], ["ok_above", [r + 1, r]], ["ok_above", [r, r + 2]], ["ok_above", [r + 2, r]],
        ["ok_above", [r + 1, r + 3]], ["ok_above", [r + 3, r + 1]], ["swap", [r, r + 3]], ["swap", [r + 1, r + 2]], ["swap", [r + 2, r + 2]]]],
      contracts: [[["new", 5], ["is_heap"]], [["new", 7], ["fill", v.slice(0, 6)], ["is_heap"]],
        // (a full heap: size past capacity then reads past the array, rather than into a placeholder)
        ...breaks(o, filled(v.slice(0, 7), 7)).map((_, k) => [["new", 7], ["fill", v.slice(0, 7)], ["break", k]])],
      add: [
        [["new", 8], ["fill", v.slice(0, 5)], ["place", v[7]], ["sift_up", 5 + r], ["place", v[6]], ["sift_up", 6 + r], ["place", v[4]], ["sift_up", 7 + r]],
        [["new", 9], ...[v[0], v[1], v[2], v[3], v[1], v[7], v[6], v[5]].map((x) => ["add", x]), ["peek"], ["size"], ["full"], ["add", v[8]], ["full"], ["peek"]],
      ],
      rem: [
        [["new", 7], ["fill", v.slice(0, 7)], ["put_root", min(o) ? BIG[o.type] : SMALL[o.type]], ["sift_down", r], ["put_root", v[4]], ["sift_down", r]],
        [["new", 8], ["fill", [...v.slice(0, 6), v[3]]], ...Array.from({ length: 7 }, () => ["rem"]), ["add", v[9]], ["add", v[10]], ["peek"]],
      ],
      heapify: [
        [["new", 10], ["raw", v.slice(0, 10)], ["heapify"]],
        [["new", 4], ["raw", [v[7], v[3]]], ["heapify"], ["rem"], ["rem"]],
        [["new", 6], ["raw", [...v.slice(0, 5)].sort((a, b) => (a < b ? -1 : 1)).reverse()], ["heapify"]],
        [["sort", v.slice(0, 9)], ["sort", []], ["sort", [v[2]]], ["sort", [v[1], v[1], v[0], v[1]]], ["sort", [...v.slice(0, 6)].sort((a, b) => (a < b ? -1 : 1))]],
      ],
    };
    const groups = names.flatMap((n) => G[n] || []);
    if (seed) {
      const rnd = C.mulberry32(seed);
      const ops = [["new", 16]];
      let n = 0;
      for (let i = 0; i < 60; i++) {
        const x = rnd(), pick = v[Math.floor(rnd() * v.length)];
        if (x < 0.55 && n < 16) { ops.push(["add", pick]); n++; }
        else if (x < 0.85 && n) { ops.push(["rem"]); n--; }
        else if (n) ops.push(["peek"]);
      }
      groups.push(ops, [["sort", Array.from({ length: 11 }, () => v[Math.floor(rnd() * v.length)])]]);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's Heap
  function refFns(o, names) {
    if (!names.includes("heap")) return [];
    return [{ name: "parent", steps: parentFn(o)() }, { name: "left", steps: leftFn(o)() }, { name: "right", steps: rightFn(o)() },
      { name: "ok_above", steps: okAbove(o)() }, { name: "swap", steps: swap() }, { name: "is_heap", steps: isHeap(o)() },
      { name: "sift_up", steps: siftUp(o)() }, { name: "heap_full", steps: [S("", "return H.size == H.capacity")] }, { name: "heap_add", steps: add(o)() },
      { name: "show", steps: showFn(o) }, { name: "put", steps: putFn() }];
  }

  function problem(o) {
    const L = o.lang, I = one(o), top = min(o) ? "smallest" : "largest";
    const sig = C.headers(DEF, o);
    const what = {
      heap_size: "the number of elements", heap_full: "true when size == capacity",
      parent: I ? "i / 2" : "(i - 1) / 2", left: I ? "2i" : "2i + 1", right: I ? "2i + 1" : "2i + 2",
      ok_above: `true if H.data[i] may sit above H.data[j] (${min(o) ? "<=" : ">="})`, swap: "exchange H.data[i] and H.data[j]",
      is_heap: "the shape (size within capacity, data of the right length) and the order (every element below may sit below its parent)",
      sift_up: "move the element at i up while it may not sit below its parent",
      heap_add: "add x (requires room) and sift it up", heap_peek: `the ${top} element (requires a non-empty heap)`,
      sift_down: `move the element at i down while a child may sit above it, swapping with the ${top === "smallest" ? "smaller" : "larger"} child (the right one when it may sit above the left)`,
      heap_rem: `remove and return the ${top} element: the last element moves to the root and sifts down`,
      heapify: "turn H.data's elements into a heap: sift down every element with children, from the last back to the root",
      heapsort: `sort A smallest first: heapify a copy, then remove the elements back into A${min(o) ? "" : " from the back"}`,
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Heap.** Fields \`size\`, \`capacity\` and \`data\` (${I ? "`capacity + 1` slots, slot 0 unused" : "`capacity` slots"}); ` +
        `${{ py: "`Heap(capacity)`", cpp: "`Heap H(capacity);`", java: "`new Heap(capacity)`", rs: "`Heap::new(capacity)`" }[L]} makes an empty heap. The root is \`H.data[${R0(o)}]\`.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_heap", L)}(H)\` at the start and end of every operation that changes the heap. ${L === "java" ? "Put the functions inside `class Heap` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints the array after every change, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "heaps", num: "13.11", title: "Heaps", heading: "Build a heap, step by step", diagramLabel: "Your heap",
    OPTIONS, DEFAULTS, FNS, STRUCTS, ctxExtra, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_heap", "break"],
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

// 13.12 Priority queues: elements served by priority (smallest number first, first-in first-out among equals), built on
// a binary heap, a sorted linked list or an unsorted array, in Python, C++, Java or Rust.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type: a type parameter T. The queue never compares elements, only their priorities, so T needs nothing." } },
    { id: "impl", label: "Built on", choices: [["heap", "binary heap"], ["list", "sorted list"], ["array", "unsorted array"]],
      help: { heap: "A min-heap of entries in an array (13.11): adding and removing are both O(log n). The standard choice.",
        list: "A linked list kept in priority order: removing is O(1) (the front), but adding walks to the right spot, O(n).",
        array: "An array in whatever order entries arrive: adding is O(1), but finding the next one to serve scans everything, O(n)." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", impl: "heap" };
  const heap = (o) => o.impl === "heap", list = (o) => o.impl === "list", arr = (o) => o.impl !== "list";
  const describe = (o) => `priority queue of ${o.type === "int" ? "int" : "generic"} elements on ${{ heap: "a binary heap", list: "a sorted linked list", array: "an unsorted array" }[o.impl]}, in ${DL.LANGS[o.lang].name}`;
  const ctxExtra = () => ({ node: { name: "Entry", fields: { data: "T", prio: "V", seq: "int", next: "N" } } });

  function complexity(o) {
    const B = { heap: { add: ["O(log n)", "add at the end, sift up"], peek: ["O(1)", "the root of the heap"], rem: ["O(log n)", "the last entry moves to the root, sifts down"], all: ["O(n log n)", "heapsort, in effect"] },
      list: { add: ["O(n)", "walk to the entry's place in the order"], peek: ["O(1)", "the first entry"], rem: ["O(1)", "unlink the first entry"], all: ["O(n²)", "every add walks the list"] },
      array: { add: ["O(1)", "amortized: append, doubling when full"], peek: ["O(n)", "scan for the entry served first"], rem: ["O(n)", "scan, then move the last entry into the hole"], all: ["O(n²)", "every remove scans"] } }[o.impl];
    return [
      { id: "add", label: "Add", big: B.add[0], note: B.add[1] },
      { id: "peek", label: "Peek", big: B.peek[0], note: B.peek[1] },
      { id: "rem", label: "Remove", big: B.rem[0], note: B.rem[1] },
      { id: "all", label: "n adds, then n removes", big: B.all[0], note: B.all[1] },
      { id: "ties", label: "Equal priorities", big: "FIFO", note: "each entry's sequence number breaks ties: first in, first out" },
      { id: "space", label: "Space", big: "O(n)", note: list(o) ? "one entry per element, plus its link" : "one entry per element, the array at most 2× that" },
      { id: "change", label: "Change a priority", big: "O(n)", note: "find the entry first; a heap with an index map does it in O(log n) (Dijkstra)", extra: true },
      { id: "is_pq", label: "Contract is_pq", big: "O(n)", note: heap(o) ? "every entry after its parent" : list(o) ? "every entry after the one before it" : "every slot in use holds an entry" },
    ];
  }

  // ---------------------------------------------------------------- the model: the exact queue every test expects
  const before = (e, f) => (e.p !== f.p ? e.p < f.p : e.s < f.s);
  function model(o) {
    const st = { size: 0, cap: 4, count: 0, d: [null, null, null, null], l: [] };
    const swap = (i, j) => { const t = st.d[i]; st.d[i] = st.d[j]; st.d[j] = t; };
    const siftUp = (i) => { let c = i; while (c > 0) { const p = Math.floor((c - 1) / 2); if (before(st.d[p], st.d[c])) return; swap(p, c); c = p; } };
    const siftDown = (i) => {
      let c = i;
      while (2 * c + 1 < st.size) {
        let s = 2 * c + 1;
        if (s + 1 < st.size && before(st.d[s + 1], st.d[s])) s += 1;
        if (before(st.d[c], st.d[s])) return;
        swap(c, s); c = s;
      }
    };
    const best = () => { let b = 0; for (let i = 1; i < st.size; i++) if (before(st.d[i], st.d[b])) b = i; return b; };
    const entry = (x, p) => ({ x, p, s: st.count++ });
    const ok = () => {
      if (list(o)) return st.l.length === st.size && st.l.every((e, i) => e.s < st.count && (i === 0 || before(st.l[i - 1], e)));
      if (st.size > st.cap || st.d.length !== st.cap) return false;
      for (let i = 0; i < st.size; i++) {
        if (!st.d[i] || st.d[i].s >= st.count) return false;
        if (heap(o) && i > 0 && !before(st.d[Math.floor((i - 1) / 2)], st.d[i])) return false;
      }
      return true;
    };
    const lab = (e) => `${e.x}(${e.p})`;
    return {
      st, best, ok,
      add: (x, p) => {
        if (list(o)) { const e = entry(x, p); const j = st.l.findIndex((f) => before(e, f)); if (j < 0) st.l.push(e); else st.l.splice(j, 0, e); st.size += 1; return; }
        if (st.size === st.cap) { st.d = [...st.d, ...new Array(st.cap).fill(null)]; st.cap *= 2; }
        st.d[st.size] = entry(x, p); st.size += 1;
        if (heap(o)) siftUp(st.size - 1);
      },
      peek: () => (list(o) ? st.l[0].x : heap(o) ? st.d[0].x : st.d[best()].x),
      rem: () => {
        if (list(o)) { st.size -= 1; return st.l.shift().x; }
        const b = heap(o) ? 0 : best(), e = st.d[b];
        st.size -= 1; st.d[b] = st.d[st.size]; st.d[st.size] = null;
        if (heap(o) && st.size > 0) siftDown(0);
        return e.x;
      },
      show: () => `[${(list(o) ? st.l : st.d).map((e) => (e ? lab(e) : "_")).join(", ")}] size=${st.size}` + (ok() ? "" : " BROKEN: is_pq(P) is false"),
    };
  }

  function diagram(o) {
    const m = model(o);
    const v = o.type === "int" ? [10, 20, 30, 40, 50, 60] : ["ant", "bee", "cat", "dog", "eel", "fox"];
    [[v[0], 3], [v[1], 1], [v[2], 2], [v[3], 1], [v[4], 3]].forEach(([x, p]) => m.add(x, p));
    const lab = (e) => `${e.x}(${e.p})`;
    if (list(o)) {
      return { rows: [[{ label: "P.head" }, ...m.st.l.flatMap((e) => [{ box: lab(e), tags: [] }, { arrow: "→" }]), { text: "null" }]],
        caption: "Entries in the order they'll be served: by priority (smaller first), then by arrival. Removing takes the first; adding walks to its place." };
    }
    const cells = [{ label: "P.data" }, ...m.st.d.map((e, i) => ({ box: e ? lab(e) : "_", tags: [String(i)], cls: e ? "" : "dummy" }))];
    if (!heap(o)) return { rows: [cells], caption: "Entries in arrival order. Adding appends; removing scans for the smallest priority (the earliest arrival among equals) and moves the last entry into its slot." };
    const order = [];
    const walk = (i, d) => { if (i < m.st.size) { walk(2 * i + 1, d + 1); order.push([i, d]); walk(2 * i + 2, d + 1); } };
    walk(0, 0);
    const rows = [0, 1, 2].map((d) => [{ label: d === 0 ? "root" : `depth ${d}` }, ...order.map(([i, id]) => ({ box: lab(m.st.d[i]), tags: [], cls: id === d ? "" : "spacer" }))]);
    return { rows: [...rows, cells], caption: "A min-heap of entries (13.11): every entry is served before its children, so the root is served next. Ties go to the earlier arrival." };
  }

  // ---------------------------------------------------------------- specs
  const O = "PQ", M = "mut:PQ", R = "ref:PQ";
  const FNS = {
    before: { params: [["e", "N"], ["f", "N"]], ret: "bool", owner: O },
    pq_size: { params: [["P", R]], ret: "int", owner: O },
    pq_empty: { params: [["P", R]], ret: "bool", owner: O },
    is_pq: { params: [["P", R]], ret: "bool", owner: O, contract: true },
    pq_grow: { params: [["P", M]], ret: "void", owner: O },
    pq_swap: { params: [["P", M], ["i", "int"], ["j", "int"]], ret: "void", owner: O },
    sift_up: { params: [["P", M], ["i", "int"]], ret: "void", owner: O },
    sift_down: { params: [["P", M], ["i", "int"]], ret: "void", owner: O },
    pq_best: { params: [["P", R]], ret: "int", owner: O },
    pq_add: { params: [["P", M], ["x", "T"], ["prio", "V"]], ret: "void", owner: O },
    pq_peek: { params: [["P", R]], ret: "Tpeek", owner: O },
    pq_rem: { params: [["P", M]], ret: "T", owner: O },
    show: { params: [["P", R]], ret: "void", harness: true, bounds: ["std::fmt::Display"] },
  };
  const STRUCTS = (o) => ({ PQ: { fields: list(o) ? { size: "int", count: "int", head: "N" } : { size: "int", capacity: "int", count: "int", data: "NArrV" } } });

  // ---------------------------------------------------------------- the functions
  const REQ = S("**Requires**: the queue is valid when the function starts.", "assert is_pq(P)");
  const ENS = S("**Ensures**: still valid when it returns.", "assert is_pq(P)");
  const NONEMPTY = S("**Requires**: a non-empty queue.", "assert not pq_empty(P)");
  const ret = (e) => [S("", `return ${e}`)];
  const beforeFn = () => [
    S("A smaller priority number is served first (priority 1 before priority 2).", "if e.prio != f.prio", ret("e.prio < f.prio")),
    S("Equal priorities: the entry added first (the smaller sequence number) goes first. That makes the queue **stable**: first in, first out among equals.", "return e.seq < f.seq")];
  const isPq = (o) => () => (list(o) ? [
    S("Walk the list, counting the entries.", "let n = 0"), S("", "let p = P.head"),
    S("", "while p != null", [
      S("", "n += 1"),
      S("Every entry got its sequence number from `count`.", "if p.seq >= P.count", ret("false")),
      S("Each entry is served before the next one. (This also catches a list that loops back: somewhere it would return to an entry served earlier.)",
        "if p.next != null and not before(p, p.next)", ret("false")),
      S("", "p = p.next")]),
    S("The count matches `size`.", "return n == P.size")]
    : [
      S("The entries fit, and `data` really has `capacity` slots.", "if P.size > P.capacity or len(P.data) != P.capacity", ret("false")),
      S("Every slot in [0, size) holds an entry…", "for i = 0 to P.size", [
        S("…whose sequence number came from `count`.", "if P.data[i] == null or P.data[i].seq >= P.count", ret("false")),
        ...(heap(o) ? [S("The heap order: every entry but the root is served after its parent.", "if i > 0 and not before(P.data[(i - 1) / 2], P.data[i])", ret("false"))] : [])]),
      S("", "return true")]);
  const grow = () => [
    S("A block twice as big.", "let B = allocn(P.capacity * 2)"),
    S("Copy the entries over.", "for i = 0 to P.size", [S("", "B[i] = P.data[i]")]),
    S("Use the new block.", "P.data = B"), S("", "P.capacity = P.capacity * 2")];
  const pqSwap = () => [S("Exchange the two entries (just pointers: nothing is copied).", "let t = P.data[i]"), S("", "P.data[i] = P.data[j]"), S("", "P.data[j] = t")];
  const siftUp = () => [
    S("`c` is where the entry is now; it moves up from `i`.", "let c = i"),
    S("While it isn't the root…", "while c > 0", [
      S("…look at its parent.", "let p = (c - 1) / 2"),
      S("If the parent is served first, the order is restored: stop.", "if before(P.data[p], P.data[c])", [S("", "return")]),
      S("Otherwise swap them and carry on from the parent's position.", "pq_swap(P, p, c)"), S("", "c = p")])];
  const siftDown = () => [
    S("`c` is where the entry is now; it moves down from `i`.", "let c = i"),
    S("While it has a left child…", "while 2 * c + 1 < P.size", [
      S("…compare with that child…", "let s = 2 * c + 1"),
      S("…or with the right child, if there is one and it's served first.", "if s + 1 < P.size and before(P.data[s + 1], P.data[s])", [S("", "s = s + 1")]),
      S("If the entry is served before that child, the order is restored: stop.", "if before(P.data[c], P.data[s])", [S("", "return")]),
      S("Otherwise swap them and carry on from the child's position.", "pq_swap(P, c, s)"), S("", "c = s")])];
  const best = () => [
    S("Scan every entry, remembering the index of the one served first so far.", "let b = 0"),
    S("", "for i = 1 to P.size", [S("", "if before(P.data[i], P.data[b])", [S("", "b = i")])]),
    S("", "return b")];
  const add = (o) => () => (list(o) ? [REQ,
    S("A new entry, with the next sequence number.", "let e = new(x, prio, P.count)"), S("", "P.count += 1"),
    S("It goes first if the list is empty or it's served before the current first entry.", "if P.head == null or before(e, P.head)", [S("", "e.next = P.head"), S("", "P.head = e")]),
    S("Otherwise walk to the last entry that's served before it…", "else", [
      S("", "let p = P.head"), S("", "while p.next != null and before(p.next, e)", [S("", "p = p.next")]),
      S("…and link it in after that one.", "e.next = p.next"), S("", "p.next = e")]),
    S("One more entry.", "P.size += 1"), ENS]
    : [REQ,
      S("No free slot: grow first.", "if P.size == P.capacity", [S("", "pq_grow(P)")]),
      S(heap(o) ? "A new entry, with the next sequence number, in the first free slot: the next spot in the bottom row of the heap." : "A new entry, with the next sequence number, at the end.", "P.data[P.size] = new(x, prio, P.count)"),
      S("", "P.count += 1"), S("One more entry.", "P.size += 1"),
      ...(heap(o) ? [S("Sift it up to where it belongs.", "sift_up(P, P.size - 1)")] : []), ENS]);
  const peek = (o) => () => [REQ, NONEMPTY,
    S((c) => (list(o) ? "The first entry is served first" : heap(o) ? "The root is served first" : "Find the entry served first") + (c.lang === "rs" && c.generic ? "; return a reference to its element." : "; return its element."),
      list(o) ? "return P.head.data" : heap(o) ? "return P.data[0].data" : "return P.data[pq_best(P)].data")];
  const rem = (o) => () => (list(o) ? [REQ, NONEMPTY,
    S("The first entry is served first.", "let e = P.head"), S("Unlink it.", "P.head = e.next"), S("One entry fewer.", "P.size -= 1"), ENS,
    S("Take its element and free the entry.", "let x = take e"), S("", "return x")]
    : [REQ, NONEMPTY,
      ...(heap(o) ? [S("The root is served first.", "let e = P.data[0]")] : [S("Find the entry served first.", "let b = pq_best(P)"), S("", "let e = P.data[b]")]),
      S("One entry fewer: the last one is `P.data[P.size]` now…", "P.size -= 1"),
      S(heap(o) ? "…move it to the root…" : "…move it into the removed entry's slot (the array has no order to keep)…", heap(o) ? "P.data[0] = P.data[P.size]" : "P.data[b] = P.data[P.size]"),
      S("…and clear its old slot.", "P.data[P.size] = null"),
      ...(heap(o) ? [S("If anything is left, sift the new root down.", "if P.size > 0", [S("", "sift_down(P, 0)")])] : []),
      ENS, S("Take the element and free the entry.", "let x = take e"), S("", "return x")]);

  // ---------------------------------------------------------------- declarations, per language
  function decl(c, o) {
    const g = c.generic, L = c.lang, T = c.T, A = arr(o);
    const say = {
      entry: "An entry holds an element, its priority, and its sequence number: the order in which it was added." + (list(o) ? " And the next entry in the list." : ""),
      size: "`size`: how many entries there are.",
      capacity: "`capacity`: how many slots `data` has. It starts at 4 and doubles when full.",
      count: "`count`: how many entries were ever added. The next entry's sequence number.",
      data: heap(o) ? "`data`: the entries, as a heap: a complete binary tree stored row by row." : "`data`: the entries, in the order they arrived. Empty slots are null.",
      head: "`head`: the first entry, the one served next; null when empty.",
    };
    const fields = A ? ["size", "capacity", "count", "data"] : ["size", "count", "head"];
    if (L === "py") {
      const E = g ? "Entry[T]" : "Entry";
      const init = { size: "self.size = 0", capacity: "self.capacity = 4", count: "self.count = 0", data: `self.data: list[${E} | None] = [None] * self.capacity`, head: `self.head: ${E} | None = None` };
      return [
        { say: say.entry, code: [g ? "class Entry[T]:" : "class Entry:"], body: [{ say: "", code: [`def __init__(self, data: ${T}, prio: int, seq: int) -> None:`], body: [
          { say: "", code: ["self.data = data"] }, { say: "", code: ["self.prio = prio"] }, { say: "", code: ["self.seq = seq"] },
          ...(list(o) ? [{ say: "", code: [`self.next: ${E} | None = None`] }] : [])] }] },
        { say: "The queue. Its constructor makes an empty one.", code: [g ? "class PQ[T]:" : "class PQ:"], body: [{ say: "", code: ["def __init__(self) -> None:"],
          body: fields.map((f) => ({ say: say[f], code: [init[f]] })) }] },
      ];
    }
    if (L === "cpp") {
      const E = g ? "Entry<T>" : "Entry", tpl = g ? ["template <typename T>"] : [];
      const init = { size: "int size = 0;", capacity: "int capacity = 4;", count: "int count = 0;", data: `std::vector<${E}*> data = std::vector<${E}*>(4, nullptr);`, head: `${E}* head = nullptr;` };
      return [
        { say: `At the top of the file: ${A ? "`<vector>`, " : ""}\`<iostream>\` for printing, \`<cassert>\` for contracts.`, code: [...(A ? ["#include <vector>"] : []), "#include <iostream>", "#include <cassert>"] },
        { say: say.entry, code: [...tpl, "struct Entry {"], close: ["};"], body: [
          { say: "", code: [`${T} data;`] }, { say: "", code: ["int prio;"] }, { say: "", code: ["int seq;"] }, ...(list(o) ? [{ say: "", code: [`${E}* next = nullptr;`] }] : []),
          { say: "A constructor taking the element, its priority and its sequence number.", code: [`Entry(${T} data, int prio, int seq) : data(data), prio(prio), seq(seq) {}`] }] },
        { say: "The queue. The defaults make an empty one.", code: [...tpl, "struct PQ {"], close: ["};"], body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    if (L === "java") {
      const E = g ? "Entry<T>" : "Entry";
      const init = { size: "int size;", capacity: "int capacity = 4;", count: "int count;", data: `${E}[] data = new Entry[4];`, head: `${E} head;` };
      return [
        { say: say.entry, code: [g ? "class Entry<T> {" : "class Entry {"], close: ["}"], body: [
          { say: "", code: [`${T} data;`] }, { say: "", code: ["int prio;"] }, { say: "", code: ["int seq;"] }, ...(list(o) ? [{ say: "", code: [`${E} next;`] }] : []),
          { say: "A constructor taking the element, its priority and its sequence number.", code: [`Entry(${T} data, int prio, int seq) {`], close: ["}"], body: [
            { say: "", code: ["this.data = data;"] }, { say: "", code: ["this.prio = prio;"] }, { say: "", code: ["this.seq = seq;"] }] }] },
        { say: "The queue. Every function goes inside this class, as a `static` method." + (g && A ? " (`new Entry[4]` makes an array of the raw type; the compiler warns the assignment is unchecked.)" : ""),
          code: [g ? "class PQ<T> {" : "class PQ {"], close: ["}"], deferClose: true, cls: O, body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    const E = g ? "Entry<T>" : "Entry";
    const rtype = { size: "size: usize,", capacity: "capacity: usize,", count: "count: usize,", data: `data: Vec<*mut ${E}>,`, head: `head: *mut ${E},` };
    const rinit = { size: "size: 0", capacity: "capacity: 4", count: "count: 0", data: "data: vec![ptr::null_mut(); 4]", head: "head: ptr::null_mut()" };
    return [
      { say: "At the very top: allow capital variable names (the queue is called `P`), and import `std::ptr` for null pointers.", code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: say.entry + " Entries are reached through raw pointers, like the hash table's entries in 13.08.", code: [`struct ${E} {`], close: ["}"], body: [
        { say: "", code: [`data: ${T},`] }, { say: "", code: ["prio: i32,"] }, { say: "", code: ["seq: usize,"] }, ...(list(o) ? [{ say: "", code: [`next: *mut ${E},`] }] : [])] },
      { say: "`Entry::new` puts an entry on the heap and returns a raw pointer to it.", code: [g ? "impl<T> Entry<T> {" : "impl Entry {"], close: ["}"], body: [
        { say: "", code: [`fn new(data: ${T}, prio: i32, seq: usize) -> *mut ${E} {`], close: ["}"], body: [
          { say: "", code: [`Box::into_raw(Box::new(Entry { data, prio, seq${list(o) ? ", next: ptr::null_mut()" : ""} }))`] }] }] },
      { say: "The queue.", code: [g ? "struct PQ<T> {" : "struct PQ {"], close: ["}"], body: fields.map((f) => ({ say: say[f], code: [rtype[f]] })) },
      { say: "A constructor `PQ::new()`: an empty queue.", code: [g ? "impl<T> PQ<T> {" : "impl PQ {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: [`PQ { ${fields.map((f) => rinit[f]).join(", ")} }`] }] }] },
    ];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const H = heap(o), Li = list(o);
    const all = [
      { id: "entries", title: "Entries and the queue",
        goal: "A priority queue serves elements by priority, not by arrival: here the smallest priority number goes first (priority 1 before priority 2). Each element is stored in an **entry** with its priority and a **sequence number**, " +
          "taken from a counter as it's added. {fn:before(e, f)} decides which of two entries is served first: the smaller priority, and between equal priorities the smaller sequence number, so equals come out first in, first out. " +
          (Li ? "The queue `P` keeps its entries in a linked list, in the order they'll be served." : H ? "The queue `P` keeps its entries in an array, as a min-heap ordered by `before`." : "The queue `P` keeps its entries in an array, in arrival order."),
        decl: (c) => decl(c, o),
        fns: [fn("before", beforeFn, "Write {fn:before}."), fn("pq_size", () => ret("P.size"), "Write {fn:pq_size}."), fn("pq_empty", () => ret("P.size == 0"), "Write {fn:pq_empty}.")] },
      { id: "contracts", title: "Contracts",
        goal: "{fn:is_pq(P)} checks " + (Li ? "that every entry is served before the next one, that every sequence number came from the counter, and that there are exactly `size` entries."
          : "that the entries fit in `data`, that every slot in [0, size) holds an entry whose sequence number came from the counter" + (H ? ", and the heap order: every entry is served after its parent." : ".")) + " Every operation checks it.",
        fns: [fn("is_pq", isPq(o), "Write {fn:is_pq}, the representation invariant.")], contractNote: true },
      { id: "add", title: "Add",
        goal: Li ? "{fn:pq_add(P, x, prio)} makes an entry with the next sequence number and links it in at its place in the order: after every entry served before it. That walk is O(n)."
          : H ? "{fn:pq_add(P, x, prio)} puts a new entry in the next free slot and sifts it up, as in 13.11, but comparing entries with {fn:before}: O(log n). {fn:pq_grow(P)} doubles the array when it's full, and {fn:pq_swap(P, i, j)} swaps two entries."
            : "{fn:pq_add(P, x, prio)} appends a new entry: O(1). {fn:pq_grow(P)} doubles the array when it's full (O(n), but rare: O(1) amortized).",
        fns: Li ? [fn("pq_add", add(o), "Write {fn:pq_add}.")]
          : [fn("pq_grow", grow, "Write {fn:pq_grow}."), ...(H ? [fn("pq_swap", pqSwap, "Write {fn:pq_swap}."), fn("sift_up", siftUp, "Write {fn:sift_up}.")] : []), fn("pq_add", add(o), "Write {fn:pq_add}.")] },
      { id: "peek", title: "Peek",
        goal: Li ? "{fn:pq_peek(P)} returns the element served next without removing it: the first entry's. O(1)." : H ? "{fn:pq_peek(P)} returns the element served next without removing it: the root's. O(1)."
          : "{fn:pq_best(P)} scans the entries for the one served first and returns its index: O(n). {fn:pq_peek(P)} returns that entry's element without removing it.",
        fns: [...(H || Li ? [] : [fn("pq_best", best, "Write {fn:pq_best}.")]), fn("pq_peek", peek(o), "Write {fn:pq_peek}.")] },
      { id: "rem", title: "Remove",
        goal: Li ? "{fn:pq_rem(P)} unlinks the first entry and returns its element: O(1)."
          : H ? "{fn:pq_rem(P)} removes the root: the last entry takes its place and {fn:sift_down(P, i)} moves it down while a child is served before it. O(log n)."
            : "{fn:pq_rem(P)} removes the entry {fn:pq_best} finds; the last entry moves into its slot, since the array keeps no order. O(n).",
        fns: [...(H ? [fn("sift_down", siftDown, "Write {fn:sift_down}.")] : []), fn("pq_rem", rem(o), "Write {fn:pq_rem}.")] },
    ];
    const ids = all.map((st) => st.id);
    return all.map((st, i) => ({ ...st, tests: ids.slice(0, i + 1) }));
  }

  // ---------------------------------------------------------------- tests
  const VALS = { int: [10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31], generic: ["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "jay", "owl", "pig", "ram"] };

  function showFn(o) {
    const item = [S("", "print e.data"), S("", 'print "("'), S("", "print e.prio"), S("", 'print ")"')];
    const tail = [S("", 'print "] size="'), S("", "print P.size"), S("", "if not is_pq(P)", [S("", 'print " BROKEN: is_pq(P) is false"'), S("", "println"), S("", "return")]), S("", "println")];
    if (list(o)) {
      return [S("", 'print "["'), S("", "let e = P.head"), S("", "let c = 0"),
        S("", "while e != null and c <= P.size", [S("", "if c > 0", [S("", 'print ", "')]), ...item, S("", "e = e.next"), S("", "c += 1")]), ...tail];
    }
    return [S("", 'print "["'), S("", "for k = 0 to len(P.data)", [S("", "if k > 0", [S("", 'print ", "')]), S("", "let e = P.data[k]"),
      S("", "if e == null", [S("", 'print "_"')]), S("", "else", item)]), ...tail];
  }

  // broken queues for is_pq
  function breaks(o) {
    return [[["P.size += 1"], "size is one too big"], [[list(o) ? "P.head.next.seq = P.count" : "P.data[1].seq = P.count"], "a sequence number that didn't come from count"],
      ...(list(o) ? [[["P.head.prio = 99"], "the first entry is out of order"], [["P.head.next.next = P.head"], "the list loops back"]]
        : [[["P.capacity += 1"], "capacity doesn't match the array"], ...(heap(o) ? [[["P.data[0].prio = 99"], "the root is out of order"]] : [])])];
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    return {
      entry: ([x, p, s]) => ({
        ir: [S("", `let e = new(${lit(x)}, ${p}, ${s})`), say(`Entry(${x}, ${p}, ${s}): data=`), S("", "print e.data"), say(", prio="), S("", "print e.prio"), say(", seq="), S("", "print e.seq"),
          ...(list(o) ? [say(", next is null: "), S("", "printb e.next == null")] : []), S("", "println")],
        out: `Entry(${x}, ${p}, ${s}): data=${x}, prio=${p}, seq=${s}${list(o) ? ", next is null: true" : ""}` }),
      before: ([a, b]) => ({
        ir: [S("", "block", [S("", `let e = new(${lit(a[0])}, ${a[1]}, ${a[2]})`), S("", `let f = new(${lit(b[0])}, ${b[1]}, ${b[2]})`),
          say(`${nm("before")}(${a.join("/")}, ${b.join("/")}) = `), S("", "printb before(e, f)"), S("", "println")])],
        out: `${nm("before")}(${a.join("/")}, ${b.join("/")}) = ${before({ p: a[1], s: a[2] }, { p: b[1], s: b[2] })}` }),
      new: (x, st) => { st.m = model(o); return { ir: [S("", "make PQ P"), say("new queue: "), S("", "show(P)")], out: `new queue: ${st.m.show()}` }; },
      fill: (xps, st) => { xps.forEach(([x, p]) => st.m.add(x, p)); return { ir: xps.map(([x, p]) => S("", `ref_pq_add(P, ${lit(x)}, ${p})`)) }; },
      size: (x, st) => C.ops.value(o, "pq_size(P)", `${nm("pq_size")}(P)`, st.m.st.size),
      empty: (x, st) => C.ops.bool(o, "pq_empty(P)", `${nm("pq_empty")}(P)`, st.m.st.size === 0),
      is_pq: (x, st) => C.ops.bool(o, "is_pq(P)", `${nm("is_pq")}(P)`, st.m.ok()),
      add: ([x, p], st) => { st.m.add(x, p); return { ir: [say(`${nm("pq_add")}(P, ${x}, ${p}): `), S("", `pq_add(P, ${lit(x)}, ${p})`), S("", "show(P)")], out: `${nm("pq_add")}(P, ${x}, ${p}): ${st.m.show()}` }; },
      peek: (x, st) => C.ops.value(o, "pq_peek(P)", `${nm("pq_peek")}(P)`, st.m.peek()),
      best: (x, st) => C.ops.value(o, "pq_best(P)", `${nm("pq_best")}(P)`, st.m.best()),
      rem: (x, st) => {
        const y = st.m.rem();
        return { ir: [say(`${nm("pq_rem")}(P) = `), S("", "print pq_rem(P)"), say(": "), S("", "show(P)")], out: `${nm("pq_rem")}(P) = ${y}: ${st.m.show()}` };
      },
      break: (k) => {
        const B = breaks(o);
        return { ir: [...B[k][0].map((x) => S("", x)), say(`after ${B[k][1]}: ${nm("is_pq")}(P) = `), S("", "printb is_pq(P)"), S("", "println")],
          out: `after ${B[k][1]}: ${nm("is_pq")}(P) = false` };
      },
    };
  }

  function testGroups(names, o, seed) {
    const v = VALS[o.type];
    const five = [[v[0], 3], [v[1], 1], [v[2], 2], [v[3], 1], [v[4], 3]];
    const G = {
      entries: [
        [["entry", [v[0], 2, 0]]],
        [["before", [[v[0], 2, 0], [v[1], 1, 1]]], ["before", [[v[1], 1, 1], [v[0], 2, 0]]], ["before", [[v[2], 2, 3], [v[3], 2, 5]]], ["before", [[v[3], 2, 5], [v[2], 2, 3]]],
          ["before", [[v[4], -1, 9], [v[5], 0, 2]]]],
        [["new"], ["size"], ["empty"], ["fill", five.slice(0, 3)], ["size"], ["empty"]],
      ],
      contracts: [[["new"], ["is_pq"]], [["new"], ["fill", five], ["is_pq"]], ...breaks(o).map((_, k) => [["new"], ["fill", five], ["break", k]])],
      add: [
        [["new"], ...five.map((x) => ["add", x]), ["add", [v[5], 1]], ["add", [v[6], 0]], ["add", [v[7], 2]], ["add", [v[8], 1]], ["size"]],
        [["new"], ...[5, 4, 3, 2, 1].map((p, k) => ["add", [v[k], p]])],
      ],
      peek: [[["new"], ["add", [v[0], 2]], ["peek"], ...(arr(o) && !heap(o) ? [["best"]] : []), ["add", [v[1], 3]], ["peek"], ["add", [v[2], 1]], ["peek"],
        ...(arr(o) && !heap(o) ? [["best"]] : []), ["add", [v[3], 1]], ["peek"], ["add", [v[4], 0]], ["peek"], ...(arr(o) && !heap(o) ? [["best"]] : [])]],
      rem: [
        [["new"], ["fill", [...five, [v[5], 1], [v[6], 2]]], ...Array.from({ length: 7 }, () => ["rem"]), ["empty"], ["add", [v[7], 4]], ["add", [v[8], 4]], ["peek"], ["rem"], ["rem"]],
        [["new"], ...[1, 1, 1, 1, 1].map((p, k) => ["add", [v[k], p]]), ["rem"], ["add", [v[5], 1]], ["rem"], ["rem"], ["rem"], ["rem"], ["rem"]],
      ],
    };
    const groups = names.flatMap((n) => G[n] || []);
    if (seed) {
      const r = C.mulberry32(seed);
      const ops = [["new"]];
      let n = 0;
      for (let i = 0; i < 70; i++) {
        const x = r();
        if (x < 0.55) { ops.push(["add", [v[Math.floor(r() * v.length)], 1 + Math.floor(r() * 4)]]); n++; }
        else if (x < 0.85 && n) { ops.push(["rem"]); n--; }
        else if (n) ops.push(["peek"]);
      }
      groups.push(ops);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's Entry and PQ
  function refFns(o, names) {
    if (!names.includes("entries")) return [];
    return [{ name: "before", steps: beforeFn() }, { name: "is_pq", steps: isPq(o)() },
      ...(arr(o) ? [{ name: "pq_grow", steps: grow() }] : []), ...(heap(o) ? [{ name: "pq_swap", steps: pqSwap() }, { name: "sift_up", steps: siftUp() }] : []),
      { name: "pq_add", steps: add(o)() }, { name: "show", steps: showFn(o) }];
  }

  function problem(o) {
    const L = o.lang, H = heap(o), Li = list(o);
    const sig = C.headers(DEF, o);
    const what = {
      before: "true if e is served before f: the smaller prio, then the smaller seq",
      pq_size: "the number of entries", pq_empty: "true when there are none",
      is_pq: Li ? "every entry served before the next, every seq below count, exactly size entries" : "entries fit in data, every slot in [0, size) holds an entry with seq below count" + (H ? ", heap order" : ""),
      ...(Li ? {} : { pq_grow: "double the array, keeping the entries in their slots" }),
      ...(H ? { pq_swap: "exchange two entries", sift_up: "move the entry at i up while it's served before its parent" } : {}),
      pq_add: "add x with priority prio and the next sequence number" + (Li ? ", at its place in the list" : H ? "; double when full; sift up" : " at the end; double when full"),
      ...(H || Li ? {} : { pq_best: "the index of the entry served first" }),
      pq_peek: "the element served next (requires a non-empty queue)",
      ...(H ? { sift_down: "move the entry at i down while a child (the right one if it's served before the left) is served before it" } : {}),
      pq_rem: "remove and return the element served next" + (Li ? "" : H ? ": the last entry moves to the root and sifts down; clear its old slot" : ": the last entry moves into its slot; clear the last slot"),
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Entry.** Fields \`data\`, \`prio\`, \`seq\`${Li ? ", `next`" : ""}. **PQ.** Fields ${(Li ? ["size", "count", "head"] : ["size", "capacity", "count", "data"]).map((f) => `\`${f}\``).join(", ")}; ` +
        `an empty queue has ${Li ? "no entries" : "capacity 4"}. The smallest priority number is served first; equal priorities, first in, first out.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_pq", L)}(P)\` at the start and end of every operation that changes the queue. ${L === "java" ? "Put the functions inside `class PQ` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints the queue after every change, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "priority-queues", num: "13.12", title: "Priority queues", heading: "Build a priority queue, step by step", diagramLabel: "Your queue",
    OPTIONS, DEFAULTS, FNS, STRUCTS, ctxExtra, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_pq", "break"],
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

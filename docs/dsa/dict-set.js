// 13.10 Dictionaries and sets: a key -> value dictionary on a hash table (separate chaining) or a binary search tree,
// sets as dictionaries whose values don't matter, set operations, and a client that only uses the interface, in
// Python, C++, Java or Rust. Keys are int or string; values are ints.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "keys", label: "Keys", choices: [["int", "int"], ["string", "string"]],
      help: { int: "Integer keys (and integer values).", string: "String keys, like words (and integer values, like counts)." } },
    { id: "base", label: "Built on", choices: [["hash", "hash table"], ["bst", "binary search tree"]],
      help: { hash: "Separate chaining (13.08): expected O(1) per operation, but the keys come out in no particular order.",
        bst: "A binary search tree (13.09): O(height) per operation, and the keys come out sorted. On random keys the height is O(log n)." } },
  ];
  const DEFAULTS = { lang: "py", keys: "int", base: "hash" };
  const hash = (o) => o.base === "hash";
  const describe = (o) => `dictionary and set with ${o.keys} keys on ${hash(o) ? "a hash table" : "a binary search tree"}, in ${DL.LANGS[o.lang].name}`;
  const KEY_T = { int: { py: "int", cpp: "int", java: "int", rs: "i32" }, string: { py: "str", cpp: "std::string", java: "String", rs: "String" } };
  const ctxExtra = (o) => ({ node: hash(o) ? { name: "Entry", fields: { key: "T", value: "V", next: "N" } } : { name: "Node", fields: { key: "T", value: "V", left: "N", right: "N" } },
    strT: o.keys === "string", generic: false, T: KEY_T[o.keys][o.lang] });

  function complexity(o) {
    const H = hash(o);
    return [
      { id: "ops", label: "Insert / lookup / delete", big: H ? "O(1)" : "O(h)", note: H ? "expected: chains stay short (at most one entry per bucket on average)" : "one path from the root: O(log n) on random keys" },
      { id: "worst", label: "Worst case", big: "O(n)", note: H ? "every key in one bucket" : "sorted keys make the tree a chain (13.09's AVL trees prevent it)" },
      { id: "sorted", label: "Print in sorted order", big: H ? "O(n log n)" : "O(n)", note: H ? "a hash table has no order: collect the keys and sort them" : "an in-order walk is already sorted" },
      { id: "min", label: "Smallest key", big: H ? "O(n)" : "O(h)", note: H ? "look at every entry" : "keep going left" },
      { id: "setops", label: "Union / intersection / difference", big: H ? "O(n + m)" : "O((n + m) h)", note: H ? "expected: one lookup or add per element" : "one O(h) lookup or add per element" },
      { id: "tally", label: "Count n keys", big: H ? "O(n)" : "O(n h)", note: "one lookup and one insert per key" },
      { id: "space", label: "Space", big: "O(n)", note: H ? "n entries, plus fewer than 2n buckets" : "n nodes" },
      { id: "is_dict", label: "Contract is_dict", big: "O(n)", note: H ? "rehash every key to check its bucket" : "check every node against bounds from its ancestors" },
    ];
  }

  // ---------------------------------------------------------------- the models: the exact structure every test expects
  function hashJS(o, k, m) {
    if (o.keys === "int") return ((k % m) + m) % m;
    let h = 0;
    for (const c of k) h = (h * 31 + c.charCodeAt(0)) % m;
    return h;
  }
  const lt = (a, b) => a < b;
  function model(o) {
    if (hash(o)) {
      const st = { size: 0, cap: 8, b: Array.from({ length: 8 }, () => []) };
      const resize = (nc) => {
        const B = Array.from({ length: nc }, () => []);
        for (let i = 0; i < st.cap; i++) for (const e of st.b[i]) B[hashJS(o, e.k, nc)].unshift(e);
        st.b = B; st.cap = nc;
      };
      const find = (k) => st.b[hashJS(o, k, st.cap)].find((e) => e.k === k);
      return {
        st, find, resize,
        insert: (k, v) => { const e = find(k); if (e) { e.v = v; return; } st.b[hashJS(o, k, st.cap)].unshift({ k, v }); st.size += 1; if (st.size > st.cap) resize(st.cap * 2); },
        remove: (k) => { const b = st.b[hashJS(o, k, st.cap)]; const i = b.findIndex((e) => e.k === k); if (i < 0) return false; b.splice(i, 1); st.size -= 1; return true; },
        entries: () => st.b.flat(),   // iteration order: bucket by bucket, each chain from the front
        preorder: () => st.b.flat(),
        show: () => `{${st.b.map((b, i) => [b, i]).filter(([b]) => b.length).map(([b, i]) => `${i}: [${b.map((e) => `${e.k}=${e.v}`).join(", ")}]`).join(", ")}} size=${st.size} capacity=${st.cap}`,
      };
    }
    const st = { root: null, size: 0 };
    const findN = (n, k) => (!n ? null : k === n.k ? n : lt(k, n.k) ? findN(n.l, k) : findN(n.r, k));
    const ins = (n, k, v) => { if (!n) return { k, v, l: null, r: null }; if (lt(k, n.k)) n.l = ins(n.l, k, v); else n.r = ins(n.r, k, v); return n; };
    const minN = (n) => { while (n.l) n = n.l; return n; };
    const delMin = (n) => (!n.l ? n.r : ((n.l = delMin(n.l)), n));
    const del = (n, k) => {
      if (lt(k, n.k)) n.l = del(n.l, k);
      else if (lt(n.k, k)) n.r = del(n.r, k);
      else { if (!n.l) return n.r; if (!n.r) return n.l; const m = minN(n.r); n.k = m.k; n.v = m.v; n.r = delMin(n.r); }
      return n;
    };
    const inorder = (n, out = []) => { if (n) { inorder(n.l, out); out.push(n); inorder(n.r, out); } return out; };
    const pre = (n, out = []) => { if (n) { out.push(n); pre(n.l, out); pre(n.r, out); } return out; };
    const shape = (n) => (!n ? "_" : !n.l && !n.r ? `${n.k}=${n.v}` : `${n.k}=${n.v}(${shape(n.l)}, ${shape(n.r)})`);
    return {
      st, find: (k) => findN(st.root, k),
      insert: (k, v) => { const e = findN(st.root, k); if (e) { e.v = v; return; } st.root = ins(st.root, k, v); st.size += 1; },
      remove: (k) => { if (!findN(st.root, k)) return false; st.root = del(st.root, k); st.size -= 1; return true; },
      entries: () => inorder(st.root).map((n) => ({ k: n.k, v: n.v })),
      preorder: () => pre(st.root).map((n) => ({ k: n.k, v: n.v })),
      nodes: () => { const out = []; const walk = (n, path) => { if (n) { out.push([n, path]); walk(n.l, `${path}.left`); walk(n.r, `${path}.right`); } }; walk(st.root, "H.root"); return out; },
      show: () => `${shape(st.root)} size=${st.size}`,
    };
  }
  const printed = (m) => `{${m.entries().map((e) => `${e.k}=${e.v}`).join(", ")}}`;
  const setShow = (m) => `{${m.entries().map((e) => e.k).join(", ")}} size=${m.st.size}`;

  function diagram(o) {
    const m = model(o);
    const keys = o.keys === "int" ? [13, 5, 21, 3, 8, 19] : ["dog", "bee", "owl", "ant", "cat", "gnu"];
    keys.forEach((k, i) => m.insert(k, i + 1));
    if (hash(o)) {
      const rows = m.st.b.map((b, i) => [b, i]).filter(([b]) => b.length).map(([b, i]) =>
        [{ label: `H.table[${i}]` }, ...b.flatMap((e) => [{ box: `${e.k}=${e.v}`, tags: [] }, { arrow: "→" }]), { text: "null" }]);
      return { rows, caption: "Each bucket H.table[i] chains the entries whose key hashes to i. Walking the table visits the keys in bucket order, which isn't sorted." };
    }
    const order = [];
    const walk = (n, d) => { if (n) { walk(n.l, d + 1); order.push([n, d]); walk(n.r, d + 1); } };
    walk(m.st.root, 0);
    const depth = Math.max(...order.map(([, d]) => d));
    const rows = [];
    for (let d = 0; d <= depth; d++) rows.push([{ label: d === 0 ? "H.root" : `depth ${d}` }, ...order.map(([n, nd]) => ({ box: `${n.k}=${n.v}`, tags: [], cls: nd === d ? "" : "spacer" }))]);
    return { rows, caption: "Each node holds a key and its value, ordered by key: smaller keys to the left. An in-order walk visits the keys sorted." };
  }

  // ---------------------------------------------------------------- specs
  const O = "Dict", M = "mut:Dict", R = "ref:Dict", SO = "Set", SM = "mut:Set", SR = "ref:Set";
  function FNS(o) {
    const common = {
      dict_size: { params: [["H", R]], ret: "int", owner: O },
      is_dict: { params: [["H", R]], ret: "bool", owner: O, contract: true },
      dict_find: { params: [["H", R], ["k", "Tref"]], ret: "N", owner: O },
      dict_contains: { params: [["H", R], ["k", "Tref"]], ret: "bool", owner: O },
      dict_lookup: { params: [["H", R], ["k", "Tref"]], ret: "V", owner: O },
      dict_insert: { params: [["H", M], ["k", "T"], ["v", "V"]], ret: "void", owner: O },
      dict_delete: { params: [["H", M], ["k", "Tref"]], ret: "bool", owner: O },
      dict_print: { params: [["H", R]], ret: "void", owner: O },
      set_size: { params: [["S", SR]], ret: "int", owner: SO },
      set_add: { params: [["S", SM], ["x", "T"]], ret: "void", owner: SO },
      set_contains: { params: [["S", SR], ["x", "Tref"]], ret: "bool", owner: SO },
      set_remove: { params: [["S", SM], ["x", "Tref"]], ret: "bool", owner: SO },
      add_all: { params: [["R", SM], ["A", SR]], ret: "void", owner: SO },
      set_union: { params: [["R", SM], ["A", SR], ["B", SR]], ret: "void", owner: SO },
      set_intersection: { params: [["R", SM], ["A", SR], ["B", SR]], ret: "void", owner: SO },
      set_difference: { params: [["R", SM], ["A", SR], ["B", SR]], ret: "void", owner: SO },
      tally: { params: [["H", M], ["A", "Arr"]], ret: "void", owner: "Client" },
      show: { params: [["H", R]], ret: "void", harness: true },
      show_set: { params: [["S", SR]], ret: "void", harness: true },
    };
    if (hash(o)) {
      return { ...common, dict_hash: { params: [["k", "Tref"], ["m", "int"]], ret: "int", owner: O }, dict_resize: { params: [["H", M], ["new_capacity", "int"]], ret: "void", owner: O } };
    }
    return { ...common,
      is_ordered: { params: [["n", "N"], ["lo", "N"], ["hi", "N"]], ret: "bool", owner: O, contract: true },
      node_size: { params: [["n", "N"]], ret: "int", owner: O, contract: true },
      node_find: { params: [["n", "N"], ["k", "Tref"]], ret: "N", owner: O },
      node_insert: { params: [["n", "N"], ["k", "T"], ["v", "V"]], ret: "N", owner: O },
      node_min: { params: [["n", "N"]], ret: "N", owner: O },
      node_delete_min: { params: [["n", "N"]], ret: "N", owner: O },
      node_delete: { params: [["n", "N"], ["k", "Tref"]], ret: "N", owner: O },
      node_print: { params: [["n", "N"], ["c", "int"]], ret: "int", owner: O },
      node_add_all: { params: [["R", SM], ["n", "N"]], ret: "void", owner: SO },
      node_add_common: { params: [["R", SM], ["n", "N"], ["B", SR]], ret: "void", owner: SO },
      node_add_missing: { params: [["R", SM], ["n", "N"], ["B", SR]], ret: "void", owner: SO },
      show_node: { params: [["n", "N"], ["depth", "int"]], ret: "void", harness: true },
      show_keys: { params: [["n", "N"], ["c", "int"]], ret: "int", harness: true },
    };
  }
  const STRUCTS = (o) => ({ Dict: { fields: hash(o) ? { size: "int", capacity: "int", table: "NArrV" } : { root: "N", size: "int" } }, Set: { fields: { dict: "V:Dict" } } });

  // ---------------------------------------------------------------- the dictionary
  const REQ = S("**Requires**: the dictionary is valid when the function starts.", "assert is_dict(H)");
  const ENS = S("**Ensures**: still valid when it returns.", "assert is_dict(H)");
  const ret = (e) => [S("", `return ${e}`)];

  // the hash function, written per language (13.08's)
  function hashSteps(o) {
    const raw = (say, code, body, close) => ({ say, code, ...(body ? { body, close } : {}) });
    if (o.keys === "int") {
      return [raw((c) => ({ py: "The key itself, reduced into [0, m). Python's `%` is never negative for a positive `m`.",
        cpp: "The key itself, reduced into [0, m). C++'s `%` keeps the key's sign, so add `m` and take `%` again.",
        java: "The key itself, reduced into [0, m). `Math.floorMod` is never negative.",
        rs: "The key itself, reduced into [0, m) with `rem_euclid` (never negative), as a `usize` position." }[c.lang]),
      { py: ["return k % m"], cpp: ["return ((k % m) + m) % m;"], java: ["return Math.floorMod(k, m);"], rs: ["return k.rem_euclid(m as i32) as usize;"] })];
    }
    return [
      raw("Start at 0.", { py: ["h = 0"], cpp: ["int h = 0;"], java: ["int h = 0;"], rs: ["let mut h = 0;"] }),
      raw("For each character of the key…", { py: ["for c in k:"], cpp: ["for (char c : k) {"], java: ["for (int i = 0; i < k.length(); i++) {"], rs: ["for c in k.bytes() {"] },
        [raw("…multiply by 31, add the character's code, and reduce mod `m` right away.",
          { py: ["h = (h * 31 + ord(c)) % m"], cpp: ["h = (h * 31 + (unsigned char) c) % m;"], java: ["h = (h * 31 + k.charAt(i)) % m;"], rs: ["h = (h * 31 + c as usize) % m;"] })],
        { py: [], cpp: ["}"], java: ["}"], rs: ["}"] }),
      raw("`h` is in [0, m).", { py: ["return h"], cpp: ["return h;"], java: ["return h;"], rs: ["return h;"] })];
  }
  const HT = {
    isDict: () => [
      S("The table really has `capacity` buckets.", "if H.capacity < 1 or len(H.table) != H.capacity", ret("false")),
      S("Count the entries as you go.", "let count = 0"),
      S("Look at every bucket…", "for i = 0 to H.capacity", [
        S("…and walk its chain.", "let p = H.table[i]"),
        S("", "while p != null", [
          S("Every entry sits in the bucket its key hashes to.", "if dict_hash(p.key, H.capacity) != i", ret("false")),
          S("", "count += 1"),
          S("More entries than `size` (or a chain that loops back): stop.", "if count > H.size", ret("false")),
          S("", "p = p.next")])]),
      S("The count matches `size`.", "return count == H.size")],
    find: () => [
      S("Walk the chain of the key's bucket.", "let p = H.table[dict_hash(k, H.capacity)]"),
      S("", "while p != null", [S("Found the key's entry.", "if p.key == k", ret("p")), S("", "p = p.next")]),
      { ...S("Not in the dictionary.", "return null"), noMutate: "in Python, falling off the end returns None anyway" }],
    print: () => [
      S("", 'print "{"'), S("`first`: nothing printed yet (no comma before the first pair).", "let first = true"),
      S("Every bucket, in order…", "for i = 0 to H.capacity", [
        S("", "let p = H.table[i]"),
        S("…every entry in its chain.", "while p != null", [
          S("", "if not first", [S("", 'print ", "')]), S("", "first = false"),
          S("", "print p.key"), S("", 'print "="'), S("", "print p.value"), S("", "p = p.next")])]),
      S("", 'println "}"')],
    resize: () => [
      S("**Requires**: at least one bucket.", "assert new_capacity >= 1"),
      S("A new table of empty buckets.", "let B = allocn(new_capacity)"),
      S("Move every entry: for each old bucket…", "for i = 0 to H.capacity", [
        S("", "let p = H.table[i]"),
        S("…walk its chain:", "while p != null", [
          S("Remember the rest of the chain (`p` is about to be relinked).", "let q = p.next"),
          S("Rehash the key for the new capacity.", "let j = dict_hash(p.key, new_capacity)"),
          S("Put `p` at the front of its new bucket.", "p.next = B[j]"), S("", "B[j] = p"),
          S("On to the rest of the old chain.", "p = q")])]),
      S("Use the new table.", "H.table = B"), S("", "H.capacity = new_capacity"), ENS],
    insert: () => [REQ,
      S("The key may already be there: then just update its value.", "let p = dict_find(H, k)"), S("", "if p != null", [S("", "p.value = v"), S("", "return")]),
      S("Otherwise a new entry goes at the front of its bucket's chain.", "let i = dict_hash(k, H.capacity)"),
      S("", "let e = new(k, v)"), S("", "e.next = H.table[i]"), S("", "H.table[i] = e"),
      S("One more entry.", "H.size += 1"),
      S("More entries than buckets: double the table, so chains stay short.", "if H.size > H.capacity", [S("", "dict_resize(H, H.capacity * 2)")]), ENS],
    del: () => [REQ,
      S("The key's bucket.", "let i = dict_hash(k, H.capacity)"),
      S("Walk its chain with `prev` one step behind.", "let prev = null"), S("", "let p = H.table[i]"),
      S("", "while p != null and p.key != k", [S("", "prev = p"), S("", "p = p.next")]),
      S("Not there: nothing to delete.", "if p == null", ret("false")),
      S("Unlink `p`: the bucket or the entry before it skips it.", "if prev == null", [S("", "H.table[i] = p.next")]), S("", "else", [S("", "prev.next = p.next")]),
      S("Free it.", "free p"), S("One entry fewer.", "H.size -= 1"), ENS, S("", "return true")],
  };
  const BT = {
    isOrdered: () => [
      S("An empty subtree is ordered.", "if n == null", ret("true")),
      S("`lo` and `hi` bound this subtree from its ancestors ({null} = no bound): the key must be strictly between them.", "if lo != null and n.key <= lo.key", ret("false")),
      S("", "if hi != null and n.key >= hi.key", ret("false")),
      S("Everything on the left is smaller than `n`, everything on the right bigger.", "return is_ordered(n.left, lo, n) and is_ordered(n.right, n, hi)")],
    nodeSize: () => [S("", "if n == null", ret("0")), S("A node plus both subtrees.", "return 1 + node_size(n.left) + node_size(n.right)")],
    isDict: () => [S("Ordered by key, with exactly `size` nodes.", "return is_ordered(H.root, null, null) and node_size(H.root) == H.size")],
    nodeFind: () => [
      S("Fell off the tree: not there.", "if n == null", ret("null")),
      S("Found the key's node.", "if k == n.key", ret("n")),
      S("Smaller: it can only be on the left…", "if k < n.key", ret("node_find(n.left, k)")),
      S("…otherwise only on the right.", "return node_find(n.right, k)")],
    find: () => [S("Search from the root.", "return node_find(H.root, k)")],
    nodePrint: () => [
      S("`c` pairs were printed before this subtree.", "if n == null", ret("c")),
      S("In order: the left subtree first…", "let d = node_print(n.left, c)"),
      S("…then this node's pair (after a comma, unless it's the first)…", "if d > 0", [S("", 'print ", "')]),
      S("", "print n.key"), S("", 'print "="'), S("", "print n.value"),
      S("…then the right subtree. Return how many are printed now.", "return node_print(n.right, d + 1)")],
    print: () => [S("", 'print "{"'), S("The pairs in key order: sorted.", "node_print(H.root, 0)"), S("", 'println "}"')],
    nodeInsert: () => [
      S("An empty spot: the new node goes here.", "if n == null", ret("new(k, v)")),
      S("Smaller: insert into the left subtree (its root may change)…", "if k < n.key", [S("", "n.left = node_insert(n.left, k, v)")]),
      S("…otherwise into the right. (`dict_insert` only calls this for a key that isn't there yet.)", "else", [S("", "n.right = node_insert(n.right, k, v)")]),
      S("", "return n")],
    insert: () => [REQ,
      S("The key may already be there: then just update its value.", "let p = dict_find(H, k)"), S("", "if p != null", [S("", "p.value = v"), S("", "return")]),
      S("Otherwise insert a new node from the root (the root itself may change).", "H.root = node_insert(H.root, k, v)"),
      S("One more entry.", "H.size += 1"), ENS],
    nodeMin: () => [S("Keep going left.", "let p = n"), S("", "while p.left != null", [S("", "p = p.left")]), S("", "return p")],
    nodeDeleteMin: () => [
      S("No left child: this is the smallest key; its right subtree takes its place.", "if n.left == null", [S("", "let r = n.right"), S("", "free n"), S("", "return r")]),
      S("Otherwise delete the minimum from the left subtree.", "n.left = node_delete_min(n.left)"), S("", "return n")],
    nodeDelete: () => [
      S("Smaller: delete from the left subtree…", "if k < n.key", [S("", "n.left = node_delete(n.left, k)")]),
      S("…bigger: from the right.", "elif n.key < k", [S("", "n.right = node_delete(n.right, k)")]),
      S("Found the key's node:", "else", [
        S("no left child: its right subtree takes its place…", "if n.left == null", [S("", "let r = n.right"), S("", "free n"), S("", "return r")]),
        S("…no right child: its left subtree does…", "if n.right == null", [S("", "let l = n.left"), S("", "free n"), S("", "return l")]),
        S("…two children: copy in the next key in order (the right subtree's smallest) with its value, and delete that node instead.", "let m = node_min(n.right)"),
        S("", "n.key = m.key"), S("", "n.value = m.value"), S("", "n.right = node_delete_min(n.right)")]),
      S("", "return n")],
    del: () => [REQ,
      S("Not there: nothing to delete.", "if dict_find(H, k) == null", ret("false")),
      S("Delete from the root.", "H.root = node_delete(H.root, k)"), S("One entry fewer.", "H.size -= 1"), ENS, S("", "return true")],
  };
  // the same in both: built on dict_find, so it doesn't matter what's underneath
  const contains = () => [S("The key is there when `dict_find` finds it.", "return dict_find(H, k) != null")];
  const lookup = () => [REQ, S("**Requires**: the key is there.", "assert dict_contains(H, k)"), S("Its value.", "return dict_find(H, k).value")];

  // ---------------------------------------------------------------- sets, set operations, the client
  const setFns = {
    size: () => [S("A set is a dictionary whose values don't matter: its size is the dictionary's.", "return dict_size(S.dict)")],
    add: () => [S("Use the element as a key; the value is never read, so store 0. (Adding an element that's there changes nothing.)", "dict_insert(S.dict, x, 0)")],
    contains: () => [S("", "return dict_contains(S.dict, x)")],
    remove: () => [S("", "return dict_delete(S.dict, x)")],
  };
  // walk a hash-based set's keys, doing `act` (statements on p.key) for each
  const walkHash = (act) => [
    S("Every bucket of `A`'s table…", "for i = 0 to A.dict.capacity", [
      S("", "let p = A.dict.table[i]"),
      S("…every entry in its chain:", "while p != null", [...act, S("", "p = p.next")])])];
  const HOPS = {
    addAll: () => walkHash([S("add its key to `R`.", "set_add(R, p.key)")]),
    union: () => [S("Everything in `A`…", "add_all(R, A)"), S("…and everything in `B` (`set_add` ignores what's there already).", "add_all(R, B)")],
    intersection: () => [
      S("Every bucket of `A`'s table…", "for i = 0 to A.dict.capacity", [
        S("", "let p = A.dict.table[i]"),
        S("…every entry in its chain: its key goes into `R` if `B` has it too.", "while p != null", [S("", "if set_contains(B, p.key)", [S("", "set_add(R, p.key)")]), S("", "p = p.next")])])],
    difference: () => [
      S("Every bucket of `A`'s table…", "for i = 0 to A.dict.capacity", [
        S("", "let p = A.dict.table[i]"),
        S("…every entry in its chain: its key goes into `R` if `B` doesn't have it.", "while p != null", [S("", "if not set_contains(B, p.key)", [S("", "set_add(R, p.key)")]), S("", "p = p.next")])])],
  };
  const BOPS = {
    nodeAddAll: () => [S("", "if n == null", [S("", "return")]), S("This node's key, then both subtrees'.", "set_add(R, n.key)"), S("", "node_add_all(R, n.left)"), S("", "node_add_all(R, n.right)")],
    addAll: () => [S("Every key of `A`'s tree.", "node_add_all(R, A.dict.root)")],
    nodeAddCommon: () => [S("", "if n == null", [S("", "return")]), S("This node's key goes into `R` if `B` has it too…", "if set_contains(B, n.key)", [S("", "set_add(R, n.key)")]),
      S("…then the same for both subtrees.", "node_add_common(R, n.left, B)"), S("", "node_add_common(R, n.right, B)")],
    nodeAddMissing: () => [S("", "if n == null", [S("", "return")]), S("This node's key goes into `R` if `B` doesn't have it…", "if not set_contains(B, n.key)", [S("", "set_add(R, n.key)")]),
      S("…then the same for both subtrees.", "node_add_missing(R, n.left, B)"), S("", "node_add_missing(R, n.right, B)")],
    union: HOPS.union,
    intersection: () => [S("Walk `A`'s tree, keeping the keys `B` has.", "node_add_common(R, A.dict.root, B)")],
    difference: () => [S("Walk `A`'s tree, keeping the keys `B` doesn't have.", "node_add_missing(R, A.dict.root, B)")],
  };
  const tally = () => [
    S("For each key in `A`…", "for i = 0 to len(A)", [
      S("…seen before: one more than its count so far…", "if dict_contains(H, A[i])", [S("", "let c = dict_lookup(H, A[i])"), S("", "dict_insert(H, A[i], c + 1)")]),
      S("…seen for the first time: a count of 1.", "else", [S("", "dict_insert(H, A[i], 1)")])])];

  // ---------------------------------------------------------------- declarations, per language
  function decl(c, o) {
    const L = c.lang, K = c.T, H = hash(o);
    const say = {
      size: "`size`: how many entries there are.",
      capacity: "`capacity`: how many buckets the table has. It starts at 8.",
      table: "`table`: the buckets, each the first entry of a chain, or null.",
      root: "`root`: the root node, null when the dictionary is empty.",
    };
    const fields = H ? ["size", "capacity", "table"] : ["root", "size"];
    const N = H ? "Entry" : "Node";
    const nodeSay = H ? "An entry holds a key, its value, and the next entry in the same bucket." : "A node holds a key, its value, and two children.";
    const links = H ? ["next"] : ["left", "right"];
    if (L === "py") {
      const init = { size: "self.size = 0", capacity: "self.capacity = 8", table: "self.table: list[Entry | None] = [None] * self.capacity", root: "self.root: Node | None = None" };
      return [
        { say: nodeSay, code: [`class ${N}:`], body: [{ say: "", code: [`def __init__(self, key: ${K}, value: int) -> None:`], body: [
          { say: "", code: ["self.key = key"] }, { say: "", code: ["self.value = value"] }, ...links.map((f) => ({ say: "", code: [`self.${f}: ${N} | None = None`] }))] }] },
        { say: "The dictionary. Its constructor makes an empty one.", code: ["class Dict:"], body: [{ say: "", code: ["def __init__(self) -> None:"],
          body: fields.map((f) => ({ say: say[f], code: [init[f]] })) }] },
      ];
    }
    if (L === "cpp") {
      const init = { size: "int size = 0;", capacity: "int capacity = 8;", table: "std::vector<Entry*> table = std::vector<Entry*>(8, nullptr);", root: "Node* root = nullptr;" };
      return [
        { say: `At the top of the file: ${H ? "`<vector>`, " : ""}\`<string>\`, \`<iostream>\` for printing, \`<cassert>\` for contracts.`, code: [...(H ? ["#include <vector>"] : []), "#include <string>", "#include <iostream>", "#include <cassert>"] },
        { say: nodeSay, code: [`struct ${N} {`], close: ["};"], body: [
          { say: "", code: [`${K} key;`] }, { say: "", code: ["int value;"] }, ...links.map((f) => ({ say: "", code: [`${N}* ${f} = nullptr;`] })),
          { say: "A constructor taking the key and the value.", code: [`${N}(${K} key, int value) : key(key), value(value) {}`] }] },
        { say: "The dictionary. The defaults make an empty one.", code: ["struct Dict {"], close: ["};"], body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    if (L === "java") {
      const init = { size: "int size;", capacity: "int capacity = 8;", table: "Entry[] table = new Entry[8];", root: "Node root;" };
      return [
        { say: nodeSay, code: [`class ${N} {`], close: ["}"], body: [
          { say: "", code: [`${K} key;`] }, { say: "", code: ["int value;"] }, ...links.map((f) => ({ say: "", code: [`${N} ${f};`] })),
          { say: "A constructor taking the key and the value.", code: [`${N}(${K} key, int value) {`], close: ["}"], body: [{ say: "", code: ["this.key = key;"] }, { say: "", code: ["this.value = value;"] }] }] },
        { say: "The dictionary. Its functions go inside this class, as `static` methods.", code: ["class Dict {"], close: ["}"], deferClose: true, cls: O,
          body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    const rtype = { size: "size: usize,", capacity: "capacity: usize,", table: "table: Vec<*mut Entry>,", root: "root: *mut Node," };
    const rinit = { size: "size: 0", capacity: "capacity: 8", table: "table: vec![ptr::null_mut(); 8]", root: "root: ptr::null_mut()" };
    return [
      { say: "At the very top: allow capital variable names (the dictionary is called `H`), and import `std::ptr` for null pointers.", code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: nodeSay + " Reached through raw pointers, as in 13.08 and 13.09.", code: [`struct ${N} {`], close: ["}"], body: [
        { say: "", code: [`key: ${K},`] }, { say: "", code: ["value: i32,"] }, ...links.map((f) => ({ say: "", code: [`${f}: *mut ${N},`] }))] },
      { say: `\`${N}::new\` puts one on the heap and returns a raw pointer to it.`, code: [`impl ${N} {`], close: ["}"], body: [
        { say: "", code: [`fn new(key: ${K}, value: i32) -> *mut ${N} {`], close: ["}"], body: [
          { say: "", code: [`Box::into_raw(Box::new(${N} { key, value, ${links.map((f) => `${f}: ptr::null_mut()`).join(", ")} }))`] }] }] },
      { say: "The dictionary.", code: ["struct Dict {"], close: ["}"], body: fields.map((f) => ({ say: say[f], code: [rtype[f]] })) },
      { say: "`Dict::new()`: an empty dictionary.", code: ["impl Dict {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: [`Dict { ${fields.map((f) => rinit[f]).join(", ")} }`] }] }] },
    ];
  }
  function setDecl(c) {
    const say = "A set wraps a dictionary: its elements are the keys.";
    return {
      py: [{ say: "Declare the class `Set`.", code: ["class Set:"], body: [{ say: "", code: ["def __init__(self) -> None:"], body: [{ say, code: ["self.dict = Dict()"] }] }] }],
      cpp: [{ say: "Declare the struct `Set`.", code: ["struct Set {"], close: ["};"], body: [{ say, code: ["Dict dict;"] }] }],
      java: [{ say: "Declare the class `Set`; the set functions go inside it.", code: ["class Set {"], close: ["}"], deferClose: true, cls: SO, body: [{ say, code: ["Dict dict = new Dict();"] }] }],
      rs: [{ say: "Declare the struct `Set`.", code: ["struct Set {"], close: ["}"], body: [{ say, code: ["dict: Dict,"] }] },
        { say: "`Set::new()`: an empty set, around an empty dictionary.", code: ["impl Set {"], close: ["}"], body: [
          { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: ["Set { dict: Dict::new() }"] }] }] }],
    }[c.lang];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const Hs = hash(o), X = Hs ? HT : BT, OPS = Hs ? HOPS : BOPS;
    const all = [
      { id: "dict", title: Hs ? "The table" : "The tree",
        goal: "A **dictionary** maps keys to values: insert a pair, look a key up, delete it. Here it's built on " +
          (Hs ? "a hash table with separate chaining (13.08): an entry holds a key, its value and the next entry in its bucket, and {fn:dict_hash(k, m)} picks the bucket."
            : "a binary search tree (13.09), ordered by key: a node holds a key, its value and two children.") +
          " The dictionary `H` keeps its size; {fn:dict_size(H)} returns it.",
        decl: (c) => decl(c, o),
        fns: [...(Hs ? [{ name: "dict_hash", steps: hashSteps(o), headerSay: "Write {fn:dict_hash}." }] : []), fn("dict_size", () => ret("H.size"), "Write {fn:dict_size}.")] },
      { id: "contracts", title: "Contracts",
        goal: Hs ? "{fn:is_dict(H)}: every entry sits in the bucket its key hashes to, and there are exactly `size` of them."
          : "{fn:is_ordered(n, lo, hi)} checks the order with bounds from the ancestors, {fn:node_size(n)} counts nodes, and {fn:is_dict(H)} puts them together.",
        fns: Hs ? [fn("is_dict", X.isDict, "Write {fn:is_dict}, the representation invariant.")]
          : [fn("is_ordered", X.isOrdered, "Write {fn:is_ordered}."), fn("node_size", X.nodeSize, "Write {fn:node_size}."), fn("is_dict", X.isDict, "Write {fn:is_dict}, the representation invariant.")],
        contractNote: true },
      { id: "lookup", title: "Lookup and printing",
        goal: "{fn:dict_find(H, k)} returns the key's " + (Hs ? "entry (walking its bucket's chain)" : "node (one path down the tree, through {fn:node_find(n, k)})") + ", or null. " +
          "{fn:dict_contains} and {fn:dict_lookup} are built on it, and they're the same code whatever is underneath. {fn:dict_print(H)} prints every pair: " +
          (Hs ? "bucket by bucket, an order that has nothing to do with the keys." : "with an in-order walk, {fn:node_print(n, c)}, so the keys come out sorted."),
        fns: Hs ? [fn("dict_find", X.find, "Write {fn:dict_find}."), fn("dict_contains", contains, "Write {fn:dict_contains}."), fn("dict_lookup", lookup, "Write {fn:dict_lookup}, which requires the key."),
          fn("dict_print", X.print, "Write {fn:dict_print}.")]
          : [fn("node_find", X.nodeFind, "Write {fn:node_find}, recursively."), fn("dict_find", X.find, "Write {fn:dict_find}."), fn("dict_contains", contains, "Write {fn:dict_contains}."),
            fn("dict_lookup", lookup, "Write {fn:dict_lookup}, which requires the key."), fn("node_print", X.nodePrint, "Write {fn:node_print}."), fn("dict_print", X.print, "Write {fn:dict_print}.")] },
      { id: "insert", title: "Insert",
        goal: "{fn:dict_insert(H, k, v)} updates the value of a key that's there, or adds a new pair: " +
          (Hs ? "at the front of its bucket's chain, doubling the table with {fn:dict_resize(H, new_capacity)} (every key rehashed) when there are more entries than buckets."
            : "{fn:node_insert(n, k, v)} walks down to an empty spot and returns the subtree's root."),
        fns: Hs ? [fn("dict_resize", X.resize, "Write {fn:dict_resize}."), fn("dict_insert", X.insert, "Write {fn:dict_insert}.")]
          : [fn("node_insert", X.nodeInsert, "Write {fn:node_insert}, recursively."), fn("dict_insert", X.insert, "Write {fn:dict_insert}.")] },
      { id: "delete", title: "Delete",
        goal: Hs ? "{fn:dict_delete(H, k)} unlinks the key's entry from its chain and returns true, or returns false if the key isn't there."
          : "{fn:dict_delete(H, k)} removes the key's node: with at most one child, the child takes its place; with two, the node takes the next key in order (with its value) and that node is removed instead.",
        fns: Hs ? [fn("dict_delete", X.del, "Write {fn:dict_delete}.")]
          : [fn("node_min", X.nodeMin, "Write {fn:node_min}."), fn("node_delete_min", X.nodeDeleteMin, "Write {fn:node_delete_min}."), fn("node_delete", X.nodeDelete, "Write {fn:node_delete}."),
            fn("dict_delete", X.del, "Write {fn:dict_delete}.")] },
      { id: "sets", title: "Sets",
        goal: "A **set** is a dictionary whose values don't matter: `S` wraps one, and its elements are the keys. {fn:set_add(S, x)}, {fn:set_contains(S, x)}, {fn:set_remove(S, x)} and {fn:set_size(S)} " +
          "just call the dictionary's functions: this is the **abstraction boundary**. The set never looks inside the dictionary, so it works the same on a hash table or a tree.",
        decl: (c) => setDecl(c),
        fns: [fn("set_size", setFns.size, "Write {fn:set_size}."), fn("set_add", setFns.add, "Write {fn:set_add}."), fn("set_contains", setFns.contains, "Write {fn:set_contains}."),
          fn("set_remove", setFns.remove, "Write {fn:set_remove}.")] },
      { id: "setops", title: "Set operations",
        goal: "Union, intersection and difference fill a result set `R`: {fn:set_union(R, A, B)} gets everything in `A` or `B`, {fn:set_intersection(R, A, B)} what's in both, {fn:set_difference(R, A, B)} what's in `A` but not `B`. " +
          (Hs ? "Each walks `A`'s table (bucket by bucket, each chain) and asks `B` with {fn:set_contains}; {fn:add_all(R, A)} adds all of `A`."
            : "Each walks `A`'s tree recursively ({fn:node_add_all}, {fn:node_add_common}, {fn:node_add_missing}) and asks `B` with {fn:set_contains}."),
        fns: Hs ? [fn("add_all", OPS.addAll, "Write {fn:add_all}."), fn("set_union", OPS.union, "Write {fn:set_union}."), fn("set_intersection", OPS.intersection, "Write {fn:set_intersection}."),
          fn("set_difference", OPS.difference, "Write {fn:set_difference}.")]
          : [fn("node_add_all", OPS.nodeAddAll, "Write {fn:node_add_all}."), fn("add_all", OPS.addAll, "Write {fn:add_all}."), fn("set_union", OPS.union, "Write {fn:set_union}."),
            fn("node_add_common", OPS.nodeAddCommon, "Write {fn:node_add_common}."), fn("set_intersection", OPS.intersection, "Write {fn:set_intersection}."),
            fn("node_add_missing", OPS.nodeAddMissing, "Write {fn:node_add_missing}."), fn("set_difference", OPS.difference, "Write {fn:set_difference}.")] },
      { id: "client", title: "A client: counting",
        goal: `A **client** uses only the interface. {fn:tally(H, A)} counts how many times each key appears in the array \`A\` (with string keys: word counts): ` +
          "look the key up, add one, insert it back. It never mentions buckets or nodes, so it's the same code on either structure; only the speed and the order of {fn:dict_print} differ.",
        decl: (c) => (c.lang === "java" ? [{ say: "In Java, client code lives in a class of its own.", code: ["class Client {"], close: ["}"], deferClose: true, cls: "Client", body: [] }] : []),
        fns: [fn("tally", tally, "Write {fn:tally}.")] },
    ];
    const ids = all.map((st) => st.id);
    return all.map((st, i) => ({ ...st, tests: ids.slice(0, i + 1) }));
  }

  // ---------------------------------------------------------------- tests
  const KEYS = { int: [13, 5, 21, 3, 8, 19, 0, 30, 11, -5, 6, 14], string: ["dog", "bee", "owl", "ant", "cat", "gnu", "eel", "pig", "fox", "hen", "jay", "ram"] };
  const ABSENT = { int: [27, 4, -3], string: ["yak", "cow", "zebu"] };
  const BIG = { int: 99, string: "zebu" }, SMALL = { int: -99, string: "aaa" };

  function showFn(o) {
    const broken = [S("", "if not is_dict(H)", [S("", 'print " BROKEN: is_dict(H) is false"'), S("", "println"), S("", "return")]), S("", "println")];
    if (hash(o)) {
      return [S("", 'print "{"'), S("", "let first = true"),
        S("", "for i = 0 to len(H.table)", [S("", "let p = H.table[i]"), S("", "if p != null", [
          S("", "if not first", [S("", 'print ", "')]), S("", "first = false"), S("", "print i"), S("", 'print ": ["'), S("", "let c = 0"),
          S("", "while p != null and c <= H.size", [S("", "if c > 0", [S("", 'print ", "')]), S("", "print p.key"), S("", 'print "="'), S("", "print p.value"), S("", "p = p.next"), S("", "c += 1")]),
          S("", 'print "]"')])]),
        S("", 'print "} size="'), S("", "print H.size"), S("", 'print " capacity="'), S("", "print H.capacity"), ...broken];
    }
    return [S("", "show_node(H.root, 0)"), S("", 'print " size="'), S("", "print H.size"), ...broken];
  }
  const showNode = () => [
    S("", "if n == null", [S("", 'print "_"'), S("", "return")]), S("", "if depth > 64", [S("", 'print "..."'), S("", "return")]),
    S("", "print n.key"), S("", 'print "="'), S("", "print n.value"), S("", "if n.left == null and n.right == null", [S("", "return")]),
    S("", 'print "("'), S("", "show_node(n.left, depth + 1)"), S("", 'print ", "'), S("", "show_node(n.right, depth + 1)"), S("", 'print ")"')];
  const showKeys = () => [S("", "if n == null", [S("", "return c")]), S("", "let d = show_keys(n.left, c)"), S("", "if d > 0", [S("", 'print ", "')]), S("", "print n.key"),
    S("", "return show_keys(n.right, d + 1)")];
  function showSetFn(o) {
    const tail = [S("", 'print "} size="'), S("", "print S.dict.size"), S("", "if not is_dict(S.dict)", [S("", 'print " BROKEN: is_dict(S.dict) is false"'), S("", "println"), S("", "return")]), S("", "println")];
    if (!hash(o)) return [S("", 'print "{"'), S("", "show_keys(S.dict.root, 0)"), ...tail];
    return [S("", 'print "{"'), S("", "let first = true"),
      S("", "for i = 0 to len(S.dict.table)", [S("", "let p = S.dict.table[i]"), S("", "let c = 0"),
        S("", "while p != null and c <= S.dict.size", [S("", "if not first", [S("", 'print ", "')]), S("", "first = false"), S("", "print p.key"), S("", "p = p.next"), S("", "c += 1")])]),
      ...tail];
  }

  // broken dictionaries for is_dict, built from where the model put the keys
  function breaks(o, m) {
    const out = [[["H.size += 1"], "H.size is one too big"]];
    if (hash(o)) {
      const b = m.st.b.findIndex((x) => x.length), e = m.st.b.findIndex((x) => !x.length);
      out.push([["H.capacity = H.capacity * 2"], "H.capacity doesn't match the table"]);
      out.push([[`H.table[${e}] = H.table[${b}]`, `H.table[${b}] = null`], "a chain is in the wrong bucket"]);
      out.push([[`H.table[${b}].next = H.table[${b}]`], "a chain loops back on itself"]);
      return out;
    }
    const path = (start, dir) => { let n = m.st.root[start === "left" ? "l" : "r"], p = `H.root.${start}`; while (n && n[dir === "left" ? "l" : "r"]) { n = n[dir === "left" ? "l" : "r"]; p += `.${dir}`; } return n ? p : null; };
    const hiPath = path("left", "right"), loPath = path("right", "left");
    if (hiPath) out.push([[`${hiPath}.key = ${C.lit(BIG[o.keys])}`], "a key in the left subtree is bigger than the root's"]);
    if (loPath) out.push([[`${loPath}.key = ${C.lit(SMALL[o.keys])}`], "a key in the right subtree is smaller than the root's"]);
    return out;
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const sets = (st, name) => (st.sets[name] ||= model(o));
    return {
      hash: ([k, m]) => C.ops.value(o, `dict_hash(${lit(k)}, ${m})`, `${nm("dict_hash")}(${k}, ${m})`, hashJS(o, k, m)),
      node: ([k, v]) => ({
        ir: [S("", `let e = new(${lit(k)}, ${v})`), say(`${hash(o) ? "Entry" : "Node"}(${k}, ${v}): key=`), S("", "print e.key"), say(", value="), S("", "print e.value"),
          ...(hash(o) ? [say(", next is null: "), S("", "printb e.next == null")] : [say(", left is null: "), S("", "printb e.left == null"), say(", right is null: "), S("", "printb e.right == null")]),
          S("", "println")],
        out: `${hash(o) ? "Entry" : "Node"}(${k}, ${v}): key=${k}, value=${v}, ${hash(o) ? "next is null: true" : "left is null: true, right is null: true"}` }),
      new: (x, st) => { st.m = model(o); return { ir: [S("", "make Dict H"), say("new dictionary: "), S("", "show(H)")], out: `new dictionary: ${st.m.show()}` }; },
      fill: (kvs, st) => { kvs.forEach(([k, v]) => st.m.insert(k, v)); return { ir: kvs.map(([k, v]) => S("", `ref_dict_insert(H, ${lit(k)}, ${v})`)) }; },
      size: (x, st) => C.ops.value(o, "dict_size(H)", `${nm("dict_size")}(H)`, st.m.st.size),
      is_dict: (x, st) => C.ops.bool(o, "is_dict(H)", `${nm("is_dict")}(H)`, true),
      contains: (k, st) => C.ops.bool(o, `dict_contains(H, ${lit(k)})`, `${nm("dict_contains")}(H, ${k})`, !!st.m.find(k)),
      lookup: (k, st) => C.ops.value(o, `dict_lookup(H, ${lit(k)})`, `${nm("dict_lookup")}(H, ${k})`, st.m.find(k).v),
      find_null: (k, st) => C.ops.bool(o, `dict_find(H, ${lit(k)}) == null`, `${nm("dict_find")}(H, ${k}) is null`, !st.m.find(k)),
      print: (x, st) => ({ ir: [say(`${nm("dict_print")}(H) prints: `), S("", "dict_print(H)")], out: `${nm("dict_print")}(H) prints: ${printed(st.m)}` }),
      resize: (n, st) => { st.m.resize(n); return { ir: [say(`${nm("dict_resize")}(H, ${n}): `), S("", `dict_resize(H, ${n})`), S("", "show(H)")], out: `${nm("dict_resize")}(H, ${n}): ${st.m.show()}` }; },
      insert: ([k, v], st) => { st.m.insert(k, v); return { ir: [say(`${nm("dict_insert")}(H, ${k}, ${v}): `), S("", `dict_insert(H, ${lit(k)}, ${v})`), S("", "show(H)")], out: `${nm("dict_insert")}(H, ${k}, ${v}): ${st.m.show()}` }; },
      delete: (k, st) => {
        const r = st.m.remove(k);
        return { ir: [say(`${nm("dict_delete")}(H, ${k}) = `), S("", `printb dict_delete(H, ${lit(k)})`), say(": "), S("", "show(H)")], out: `${nm("dict_delete")}(H, ${k}) = ${r}: ${st.m.show()}` };
      },
      break_empty: () => ({ ir: [S("", "H.capacity = 16"), say(`after setting H.capacity to 16 on an empty table: ${nm("is_dict")}(H) = `), S("", "printb is_dict(H)"), S("", "println")],
        out: `after setting H.capacity to 16 on an empty table: ${nm("is_dict")}(H) = false` }),
      break: (k, st) => {
        const B = breaks(o, st.m);
        return { ir: [...B[k][0].map((x) => S("", x)), say(`after ${B[k][1]}: ${nm("is_dict")}(H) = `), S("", "printb is_dict(H)"), S("", "println")],
          out: `after ${B[k][1]}: ${nm("is_dict")}(H) = false` };
      },
      // sets: st.sets[name] models each one
      make_set: (name, st) => { st.sets = st.sets || {}; st.sets[name] = model(o); return { ir: [S("", `make Set ${name}`)] }; },
      set_fill: ([name, xs], st) => { xs.forEach((x) => sets(st, name).insert(x, 0)); return { ir: [...xs.map((x) => S("", `ref_set_add(${name}, ${lit(x)})`)), say(`${name} = `), S("", `show_set(${name})`)], out: `${name} = ${setShow(sets(st, name))}` }; },
      set_add: (x, st) => { st.sets.S.insert(x, 0); return { ir: [say(`${nm("set_add")}(S, ${x}): `), S("", `set_add(S, ${lit(x)})`), S("", "show_set(S)")], out: `${nm("set_add")}(S, ${x}): ${setShow(st.sets.S)}` }; },
      set_contains: (x, st) => C.ops.bool(o, `set_contains(S, ${lit(x)})`, `${nm("set_contains")}(S, ${x})`, !!st.sets.S.find(x)),
      set_remove: (x, st) => {
        const r = st.sets.S.remove(x);
        return { ir: [say(`${nm("set_remove")}(S, ${x}) = `), S("", `printb set_remove(S, ${lit(x)})`), say(": "), S("", "show_set(S)")], out: `${nm("set_remove")}(S, ${x}) = ${r}: ${setShow(st.sets.S)}` };
      },
      set_size: (x, st) => C.ops.value(o, "set_size(S)", `${nm("set_size")}(S)`, st.sets.S.st.size),
      // R = A op B, in a fresh R of its own
      setop: ([op, a, b], st) => {
        const A = st.sets[a], B = st.sets[b], R = model(o);
        const keep = { union: () => true, intersection: (k) => !!B.find(k), difference: (k) => !B.find(k) }[op];
        A.preorder().forEach((e) => { if (keep(e.k)) R.insert(e.k, 0); });
        if (op === "union") B.preorder().forEach((e) => R.insert(e.k, 0));
        return { ir: [S("", "block", [S("", "make Set R"), S("", `set_${op}(R, ${a}, ${b})`), say(`${nm(`set_${op}`)}(R, ${a}, ${b}): R = `), S("", "show_set(R)")])],
          out: `${nm(`set_${op}`)}(R, ${a}, ${b}): R = ${setShow(R)}` };
      },
      add_all: ([a], st) => {
        const R = model(o);
        st.sets[a].preorder().forEach((e) => R.insert(e.k, 0));
        return { ir: [S("", "block", [S("", "make Set R"), S("", `add_all(R, ${a})`), say(`${nm("add_all")}(R, ${a}): R = `), S("", "show_set(R)")])], out: `${nm("add_all")}(R, ${a}): R = ${setShow(R)}` };
      },
      tally: (keys, st) => {
        const m = model(o);
        keys.forEach((k) => { const e = m.find(k); m.insert(k, e ? e.v + 1 : 1); });
        return { ir: [S("", "block", [S("", `let A = [${keys.map(lit).join(", ")}]`), S("", "make Dict H"), S("", "tally(H, A)"), say(`${nm("tally")}(H, [${keys.join(", ")}]): `), S("", "dict_print(H)"),
          say("  "), S("", "show(H)")])], out: [`${nm("tally")}(H, [${keys.join(", ")}]): ${printed(m)}`, `  ${m.show()}`] };
      },
    };
  }

  function testGroups(names, o, seed) {
    const K = KEYS[o.keys], Z = ABSENT[o.keys];
    const kv = (n, from = 0) => K.slice(from, from + n).map((k, i) => [k, from + i + 1]);
    const filled = (n) => { const m = model(o); kv(n).forEach(([k, v]) => m.insert(k, v)); return m; };
    const G = {
      dict: [[["node", [K[0], 7]]], [["new"], ["size"]], ...(hash(o) ? [[...K.slice(0, 8).map((k) => ["hash", [k, 8]]), ...K.slice(0, 3).map((k) => ["hash", [k, 16]]), ["hash", [K[1], 1]]]] : [])],
      contracts: [[["new"], ["is_dict"]], [["new"], ["fill", kv(6)], ["is_dict"], ["size"]],
        ...breaks(o, filled(6)).map((_, k) => [["new"], ["fill", kv(6)], ["break", k]]),
        // (only the capacity check notices this one: there's no entry to land in the wrong bucket)
        ...(hash(o) ? [[["new"], ["break_empty"]]] : [])],
      lookup: [[["new"], ["print"], ["find_null", K[0]], ["fill", kv(6)], ...K.slice(0, 6).map((k) => ["contains", k]), ...Z.map((k) => ["contains", k]),
        ["lookup", K[0]], ["lookup", K[3]], ["lookup", K[5]], ["find_null", Z[0]], ["find_null", K[2]], ["print"]]],
      insert: [
        [["new"], ...kv(5).map((x) => ["insert", x]), ["insert", [K[1], 42]], ["insert", [K[0], 9]], ["size"]],
        [["new"], ...kv(10).map((x) => ["insert", x]), ["size"], ["lookup", K[7]], ["print"]],
        ...(hash(o) ? [[["new"], ["fill", kv(3)], ["resize", 16], ["resize", 2], ["contains", K[1]]]] : []),
      ],
      delete: [
        (() => {
          const m = filled(10), ops = [["new"], ["fill", kv(10)], ["delete", Z[1]]];
          if (hash(o)) return [...ops, ...[K[0], K[2], K[1], K[9], K[5]].map((k) => ["delete", k]), ["delete", K[0]], ["size"], ["insert", [K[0], 5]], ["print"]];
          const kinds = [(n) => !n.l && !n.r, (n) => n.l && !n.r, (n) => !n.l && n.r, (n) => n.l && n.r, (n) => n === m.st.root];
          let last = null;
          for (const kind of kinds) {
            const hit = m.nodes().find(([n]) => kind(n));
            if (!hit) continue;
            last = hit[0].k;
            ops.push(["delete", last]);
            m.remove(last);
          }
          return [...ops, ["delete", last], ["size"], ["insert", [K[0], 5]], ["print"]];
        })(),
        [["new"], ["fill", kv(3)], ["delete", K[0]], ["delete", K[1]], ["delete", K[2]], ["delete", K[2]], ["size"]],
        // only a right child, which has a left child (and the mirror image): the child's subtree must take the node's place whole
        ...(() => { const [a, b, c, d] = [...K.slice(0, 4)].sort((x, y) => (x < y ? -1 : 1)); return [[["new"], ["fill", [[a, 1], [c, 2], [b, 3]]], ["delete", a]], [["new"], ["fill", [[d, 1], [b, 2], [c, 3]]], ["delete", d]]]; })(),
      ],
      sets: [[["make_set", "S"], ["set_size", 0], ["set_add", K[0]], ["set_add", K[1]], ["set_add", K[2]], ["set_add", K[1]], ["set_size", 0], ["set_contains", K[1]], ["set_contains", Z[0]],
        ["set_remove", K[1]], ["set_remove", K[1]], ["set_contains", K[1]], ["set_size", 0], ...K.slice(3, 10).map((k) => ["set_add", k])]],
      setops: [
        [["make_set", "A"], ["make_set", "B"], ["set_fill", ["A", [K[0], K[1], K[2], K[3], K[4], K[5]]]], ["set_fill", ["B", [K[4], K[2], K[6], K[7], K[0]]]],
          ["add_all", ["A"]], ["setop", ["union", "A", "B"]], ["setop", ["intersection", "A", "B"]], ["setop", ["difference", "A", "B"]], ["setop", ["difference", "B", "A"]]],
        [["make_set", "A"], ["make_set", "B"], ["set_fill", ["A", [K[1], K[3]]]], ["set_fill", ["B", []]],
          ["setop", ["union", "A", "B"]], ["setop", ["union", "B", "A"]], ["setop", ["intersection", "A", "B"]], ["setop", ["difference", "A", "B"]], ["setop", ["difference", "B", "A"]]],
      ],
      client: [[["tally", [K[0], K[1], K[0], K[2], K[1], K[0], K[3], K[2], K[0]]]], [["tally", []]], [["tally", [...K.slice(0, 9), K[4], K[4]]]]],
    };
    const groups = names.flatMap((n) => G[n] || []);
    if (seed) {
      const r = C.mulberry32(seed);
      const pool = [...K, ...Z];
      const ops = [["new"]];
      const m = model(o);
      for (let i = 0; i < 60; i++) {
        const x = r(), k = pool[Math.floor(r() * pool.length)];
        if (x < 0.5) { const v = 1 + Math.floor(r() * 99); ops.push(["insert", [k, v]]); m.insert(k, v); }
        else if (x < 0.75) { ops.push(["delete", k]); m.remove(k); }
        else if (x < 0.9) ops.push(["contains", k]);
        else if (m.find(k)) ops.push(["lookup", k]);
        else ops.push(["size"]);
      }
      ops.push(["print"]);
      const pick = (n) => Array.from({ length: n }, () => pool[Math.floor(r() * pool.length)]);
      groups.push(ops, [["make_set", "A"], ["make_set", "B"], ["set_fill", ["A", pick(8)]], ["set_fill", ["B", pick(8)]],
        ["setop", ["union", "A", "B"]], ["setop", ["intersection", "A", "B"]], ["setop", ["difference", "A", "B"]]], [["tally", pick(14)]]);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's types
  function refFns(o, names) {
    if (!names.includes("dict")) return [];
    const base = hash(o) ? [{ name: "dict_hash", steps: hashSteps(o) }, { name: "is_dict", steps: HT.isDict() }, { name: "dict_find", steps: HT.find() }, { name: "dict_resize", steps: HT.resize() },
      { name: "dict_insert", steps: HT.insert() }]
      : [{ name: "is_ordered", steps: BT.isOrdered() }, { name: "node_size", steps: BT.nodeSize() }, { name: "is_dict", steps: BT.isDict() }, { name: "node_find", steps: BT.nodeFind() },
        { name: "dict_find", steps: BT.find() }, { name: "node_insert", steps: BT.nodeInsert() }, { name: "dict_insert", steps: BT.insert() },
        { name: "show_node", steps: showNode() }, { name: "show_keys", steps: showKeys() }];
    return [...base, { name: "show", steps: showFn(o) }, ...(names.includes("sets") ? [{ name: "set_add", steps: setFns.add() }, { name: "show_set", steps: showSetFn(o) }] : [])];
  }

  function problem(o) {
    const L = o.lang, Hs = hash(o);
    const sig = C.headers(DEF, o);
    const what = {
      ...(Hs ? { dict_hash: o.keys === "int" ? "the key reduced into [0, m), never negative" : "h = (h * 31 + character code) % m over the characters, starting at 0" } : {}),
      dict_size: "the number of entries",
      ...(Hs ? { is_dict: "every entry in its key's bucket, exactly size entries" }
        : { is_ordered: "every key strictly between the bounds lo and hi (nodes, or null for none)", node_size: "the number of nodes", is_dict: "ordered, with exactly size nodes" }),
      ...(Hs ? {} : { node_find: "the key's node in the subtree, or null" }),
      dict_find: "the key's " + (Hs ? "entry" : "node") + ", or null", dict_contains: "true if the key is there", dict_lookup: "the key's value (requires the key)",
      ...(Hs ? {} : { node_print: "print the subtree's pairs in order after c earlier ones; return the new count" }),
      dict_print: `print every pair as \`{k=v, k=v}\` and end the line: ${Hs ? "bucket by bucket, each chain from the front" : "in key order"}`,
      ...(Hs ? { dict_resize: "rehash every entry into new_capacity buckets (front of each chain)" } : { node_insert: "insert a new key below n (smaller left, otherwise right); return the subtree's root" }),
      dict_insert: "update the key's value, or add the pair" + (Hs ? " at the front of its chain; then if size > capacity, resize to capacity * 2" : ""),
      ...(Hs ? {} : { node_min: "the leftmost node", node_delete_min: "remove the leftmost node; return the subtree's root", node_delete: "remove the key's node (two children: copy in the right subtree's minimum key and value)" }),
      dict_delete: "remove the key; false if it isn't there",
      set_size: "the number of elements", set_add: "add x (value 0 in the dictionary)", set_contains: "true if x is in the set", set_remove: "remove x; false if it isn't there",
      ...(Hs ? {} : { node_add_all: "add every key of the subtree to R: the node's, then the left subtree's, then the right's" }),
      add_all: "add every element of A to R" + (Hs ? ", bucket by bucket" : ""),
      set_union: "R gets the elements of A, then of B",
      ...(Hs ? {} : { node_add_common: "add each key of the subtree that B has (node, left, right)" }),
      set_intersection: "R gets the elements of A that B has",
      ...(Hs ? {} : { node_add_missing: "add each key of the subtree that B doesn't have (node, left, right)" }),
      set_difference: "R gets the elements of A that B doesn't have",
      tally: "for each key of A: insert it with count 1, or add 1 to its count",
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**${Hs ? "Entry" : "Node"}.** Fields \`key\`, \`value\`, ${Hs ? "`next`" : "`left`, `right`"}. **Dict.** Fields ${(Hs ? ["size", "capacity", "table"] : ["root", "size"]).map((f) => `\`${f}\``).join(", ")}` +
        `${Hs ? "; an empty dictionary has capacity 8" : ""}. **Set.** One field, \`dict\`.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_dict", L)}(H)\` at the start and end of every dictionary operation that changes it. ${L === "java" ? "Put the dictionary functions in `class Dict`, the set functions in `class Set` and `tally` in `class Client`, as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints the structure after every change, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "dictionaries-sets", num: "13.10", title: "Dictionaries & sets", heading: "Build a dictionary and a set, step by step", diagramLabel: "Your dictionary",
    OPTIONS, DEFAULTS, FNS, STRUCTS, ctxExtra, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_dict", "break", "break_empty"],
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

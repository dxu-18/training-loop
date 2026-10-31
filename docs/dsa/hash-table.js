// 13.08 Hash tables: key -> value tables with separate chaining, or open addressing with linear or quadratic probing,
// int or string keys, in Python, C++, Java or Rust. The hash function is written by hand (the same numbers in every
// language), so the tests can print the exact layout of the table.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const { S, fname } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "keys", label: "Keys", choices: [["int", "int"], ["string", "string"]],
      help: { int: "Integer keys: the hash is the key, reduced into [0, m).", string: "String keys: a polynomial hash of the characters, reduced into [0, m) as it goes." } },
    { id: "method", label: "Collisions", choices: [["chain", "separate chaining"], ["linear", "linear probing"], ["quadratic", "quadratic probing"]],
      help: { chain: "Each slot (bucket) holds a linked list of the entries that hash there. The table doubles when there are more entries than buckets.",
        linear: "Open addressing: one entry per slot; on a collision try the next slot, then the next. The table doubles when more than half its slots are in use.",
        quadratic: "Open addressing: on a collision jump 1, then 2, then 3 slots further (steps t(t+1)/2), which breaks up clusters. The table doubles when more than half its slots are in use." } },
  ];
  const DEFAULTS = { lang: "py", keys: "int", method: "chain" };
  const chain = (o) => o.method === "chain";
  const describe = (o) => `hash table with ${o.keys} keys and ${{ chain: "separate chaining", linear: "linear probing", quadratic: "quadratic probing" }[o.method]}, in ${DL.LANGS[o.lang].name}`;
  const KEY_T = { int: { py: "int", cpp: "int", java: "int", rs: "i32" }, string: { py: "str", cpp: "std::string", java: "String", rs: "String" } };
  const ctxExtra = (o) => ({ node: { name: "Entry", fields: { key: "T", value: "V", next: "N" } }, strT: o.keys === "string", generic: false, T: KEY_T[o.keys][o.lang] });

  function complexity(o) {
    const ch = chain(o);
    return [
      { id: "ops", label: "Insert / lookup / delete", big: "O(1)", note: ch ? "expected: with at most one entry per bucket on average, chains are short" : "expected: at most half the slots are used, so probes are short" },
      { id: "worst", label: "Worst case", big: "O(n)", note: ch ? "every key in the same bucket" : "one long run of full slots" },
      { id: "resize", label: "Resize", big: "O(n)", note: "rehash every entry; doubling makes it O(1) amortized per insert" },
      { id: "cluster", label: ch ? "Collisions" : o.method === "linear" ? "Primary clustering" : "Clustering",
        big: ch ? "chains" : o.method === "linear" ? "yes" : "less", note: ch ? "colliding keys share a bucket's list" : o.method === "linear" ? "runs of full slots grow and merge, and every key landing in one walks to its end" : "jumps of 1, 2, 3… leave a run quickly" },
      { id: "delete", label: "Delete", big: "O(1)", note: ch ? "unlink the entry from its chain" : "leave a tombstone, so later probes keep going past the slot" },
      { id: "space", label: "Space", big: "O(n)", note: ch ? "n entries + capacity buckets (capacity ≥ n)" : "capacity ≥ 2 × the slots in use" },
      { id: "order", label: "Min / sorted order", big: "O(n)", note: "a hash table keeps no order", extra: true },
      { id: "is_ht", label: "Contract is_ht", big: "O(n)", note: ch ? "rehash every entry to check its bucket" : "re-probe every entry to check it's reachable" },
    ];
  }

  // ---------------------------------------------------------------- the model: the table every test expects
  function hashJS(o, k, m) {
    if (o.keys === "int") return ((k % m) + m) % m;
    let h = 0;
    for (const c of k) h = (h * 31 + c.charCodeAt(0)) % m;
    return h;
  }
  const probeJS = (o, h, t, m) => (o.method === "linear" ? (h + t) % m : (h + Math.floor(t * (t + 1) / 2)) % m);
  const TOMB = { tomb: true };
  function model(o) {
    const st = { size: 0, used: 0, cap: 8, slots: new Array(8).fill(null), buckets: Array.from({ length: 8 }, () => []) };
    const findSlot = (k) => { const h = hashJS(o, k, st.cap); for (let t = 0; t < st.cap; t++) { const j = probeJS(o, h, t, st.cap); const e = st.slots[j]; if (e === null) return -1; if (e !== TOMB && e.k === k) return j; } return -1; };
    const resize = (nc) => {
      if (chain(o)) {
        const B = Array.from({ length: nc }, () => []);
        for (let i = 0; i < st.cap; i++) for (const e of st.buckets[i]) B[hashJS(o, e.k, nc)].unshift(e);
        st.buckets = B;
      } else {
        const B = new Array(nc).fill(null);
        for (let i = 0; i < st.cap; i++) {
          const e = st.slots[i];
          if (e === null || e === TOMB) continue;
          const h = hashJS(o, e.k, nc);
          let t = 0, j = probeJS(o, h, 0, nc);
          while (B[j] !== null) { t += 1; j = probeJS(o, h, t, nc); }
          B[j] = e;
        }
        st.slots = B;
        st.used = st.size;
      }
      st.cap = nc;
    };
    const find = (k) => (chain(o) ? st.buckets[hashJS(o, k, st.cap)].find((e) => e.k === k) : (findSlot(k) >= 0 ? st.slots[findSlot(k)] : undefined));
    return {
      st, find, resize, findSlot,
      insert: (k, v) => {
        const e = find(k);
        if (e) { e.v = v; return; }
        if (chain(o)) {
          st.buckets[hashJS(o, k, st.cap)].unshift({ k, v });
          st.size += 1;
          if (st.size > st.cap) resize(st.cap * 2);
          return;
        }
        const h = hashJS(o, k, st.cap);
        let t = 0, i = probeJS(o, h, 0, st.cap);
        while (st.slots[i] !== null && st.slots[i] !== TOMB) { t += 1; i = probeJS(o, h, t, st.cap); }
        if (st.slots[i] === null) st.used += 1;
        st.slots[i] = { k, v };
        st.size += 1;
        if (st.used * 2 > st.cap) resize(st.cap * 2);
      },
      remove: (k) => {
        if (chain(o)) {
          const b = st.buckets[hashJS(o, k, st.cap)];
          const i = b.findIndex((e) => e.k === k);
          if (i < 0) return false;
          b.splice(i, 1); st.size -= 1; return true;
        }
        const j = findSlot(k);
        if (j < 0) return false;
        st.slots[j] = TOMB; st.size -= 1; return true;
      },
      show: () => (chain(o)
        ? `{${st.buckets.map((b, i) => [b, i]).filter(([b]) => b.length).map(([b, i]) => `${i}: [${b.map((e) => `${e.k}=${e.v}`).join(", ")}]`).join(", ")}} size=${st.size} capacity=${st.cap}`
        : `[${st.slots.map((e) => (e === null ? "_" : e === TOMB ? "X" : `${e.k}=${e.v}`)).join(", ")}] size=${st.size} used=${st.used} capacity=${st.cap}`),
    };
  }

  function diagram(o) {
    const m = model(o);
    // (four keys keep an open-addressing table at 8 slots; 3 and 11 collide)
    const keys = (o.keys === "int" ? [3, 11, 5, 8, 19] : ["ant", "bee", "cat", "dog", "eel"]).slice(0, chain(o) ? 5 : 4);
    keys.forEach((k, i) => m.insert(k, i + 1));
    if (chain(o)) {
      const rows = m.st.buckets.map((b, i) => [b, i]).filter(([b]) => b.length).map(([b, i]) =>
        [{ label: `H.table[${i}]` }, ...b.flatMap((e, j) => [{ box: `${e.k}=${e.v}`, tags: [] }, { arrow: "→" }]), { text: "null" }]);
      return { rows, caption: "Each bucket H.table[i] is a linked list of the entries whose key hashes to i. Empty buckets hold null. Lookup hashes the key, then walks one short chain." };
    }
    const cells = m.st.slots.map((e, i) => ({ box: e === null ? "_" : `${e.k}=${e.v}`, tags: [String(i)], cls: e === null ? "dummy" : "" }));
    return { rows: [cells], caption: `One entry per slot. A key that collides ${o.method === "linear" ? "moves to the next slot, then the next" : "jumps 1, then 2, then 3 slots further"} until it finds a free one; lookup follows the same steps and stops at an empty slot.` };
  }

  // ---------------------------------------------------------------- specs
  const O = "HashTable", M = "mut:HashTable", R = "ref:HashTable";
  function FNS(o) {
    const common = {
      ht_hash: { params: [["k", "Tref"], ["m", "int"]], ret: "int", owner: O },
      is_ht: { params: [["H", R]], ret: "bool", owner: O, contract: true },
      ht_size: { params: [["H", R]], ret: "int", owner: O },
      ht_contains: { params: [["H", R], ["k", "Tref"]], ret: "bool", owner: O },
      ht_get: { params: [["H", R], ["k", "Tref"]], ret: "V", owner: O },
      ht_resize: { params: [["H", M], ["new_capacity", "int"]], ret: "void", owner: O },
      ht_insert: { params: [["H", M], ["k", "T"], ["v", "V"]], ret: "void", owner: O },
      ht_delete: { params: [["H", M], ["k", "Tref"]], ret: "bool", owner: O },
      show: { params: [["H", R]], ret: "void", harness: true },
    };
    if (chain(o)) return { ...common, ht_find: { params: [["H", R], ["k", "Tref"]], ret: "N", owner: O } };
    return { ...common, ht_probe: { params: [["h", "int"], ["t", "int"], ["m", "int"]], ret: "int", owner: O },
      ht_find_slot: { params: [["H", R], ["k", "Tref"]], ret: "index", owner: O } };
  }
  const STRUCTS = (o) => ({ HashTable: { fields: chain(o) ? { size: "int", capacity: "int", table: "NArrV" } : { size: "int", used: "int", capacity: "int", table: "NArrV", tomb: "N" } } });

  // ---------------------------------------------------------------- the hash function (written per language)
  function hashSteps(o) {
    const raw = (say, code, body, close) => ({ say, code, ...(body ? { body, close } : {}) });
    if (o.keys === "int") {
      return [
        raw((c) => ({ py: "The key itself, reduced into [0, m). Python's `%` is never negative for a positive `m`, so negative keys land in range too.",
          cpp: "The key itself, reduced into [0, m). C++'s `%` keeps the key's sign, so add `m` and take `%` again: negative keys land in range too.",
          java: "The key itself, reduced into [0, m). `Math.floorMod` is never negative (plain `%` keeps the key's sign).",
          rs: "The key itself, reduced into [0, m). `rem_euclid` is never negative (plain `%` keeps the key's sign); then make it a `usize` position." }[c.lang]),
        { py: ["return k % m"], cpp: ["return ((k % m) + m) % m;"], java: ["return Math.floorMod(k, m);"], rs: ["return k.rem_euclid(m as i32) as usize;"] }),
      ];
    }
    return [
      raw("Start at 0.", { py: ["h = 0"], cpp: ["int h = 0;"], java: ["int h = 0;"], rs: ["let mut h = 0;"] }),
      raw("For each character of the key…", { py: ["for c in k:"], cpp: ["for (char c : k) {"], java: ["for (int i = 0; i < k.length(); i++) {"], rs: ["for c in k.bytes() {"] },
        [raw("…multiply by 31, add the character's code, and reduce mod `m` right away, so `h` stays small and never overflows.",
          { py: ["h = (h * 31 + ord(c)) % m"], cpp: ["h = (h * 31 + (unsigned char) c) % m;"], java: ["h = (h * 31 + k.charAt(i)) % m;"], rs: ["h = (h * 31 + c as usize) % m;"] })],
        { py: [], cpp: ["}"], java: ["}"], rs: ["}"] }),
      raw("`h` is in [0, m).", { py: ["return h"], cpp: ["return h;"], java: ["return h;"], rs: ["return h;"] }),
    ];
  }
  const probeSteps = (o) => [o.method === "linear"
    ? S("The `t`-th try is `t` slots after the home slot `h`, wrapping around.", "return (h + t) % m")
    : S("The `t`-th try jumps `t(t + 1) / 2` slots from `h` (0, 1, 3, 6, …): with a power-of-two `m` these reach every slot.", "return (h + t * (t + 1) / 2) % m")];

  // ---------------------------------------------------------------- separate chaining
  const REQ = S("**Requires**: the table is valid when the function starts.", "assert is_ht(H)");
  const ENS = S("**Ensures**: still valid when it returns.", "assert is_ht(H)");
  const CHN = {
    isHt: () => [
      S("The table really has `capacity` buckets.", "if H.capacity < 1 or len(H.table) != H.capacity", [S("", "return false")]),
      S("Count the entries as you go.", "let count = 0"),
      S("Look at every bucket…", "for i = 0 to H.capacity", [
        S("…and walk its chain.", "let p = H.table[i]"),
        S("", "while p != null", [
          S("Every entry sits in the bucket its key hashes to.", "if ht_hash(p.key, H.capacity) != i", [S("", "return false")]),
          S("", "count += 1"),
          S("More entries than `size` (or a chain that loops back): stop.", "if count > H.size", [S("", "return false")]),
          S("", "p = p.next")])]),
      S("The count matches `size`.", "return count == H.size")],
    find: () => [
      S("Walk the chain of the key's bucket.", "let p = H.table[ht_hash(k, H.capacity)]"),
      S("", "while p != null", [S("Found the key's entry.", "if p.key == k", [S("", "return p")]), S("", "p = p.next")]),
      { ...S("Not in the table.", "return null"), noMutate: "in Python, falling off the end returns None anyway" }],
    contains: () => [S("The key is there when `ht_find` finds its entry.", "return ht_find(H, k) != null")],
    get: () => [REQ, S("**Requires**: the key is in the table.", "assert ht_contains(H, k)"), S("Its entry's value.", "return ht_find(H, k).value")],
    resize: () => [
      S("**Requires**: at least one bucket.", "assert new_capacity >= 1"),
      S("A new table of empty buckets.", "let B = allocn(new_capacity)"),
      S("Move every entry: for each old bucket…", "for i = 0 to H.capacity", [
        S("", "let p = H.table[i]"),
        S("…walk its chain:", "while p != null", [
          S("Remember the rest of the chain (we're about to relink `p`).", "let q = p.next"),
          S("Rehash the key for the new capacity: its bucket can change.", "let j = ht_hash(p.key, new_capacity)"),
          S("Put `p` at the front of its new bucket.", "p.next = B[j]"), S("", "B[j] = p"),
          S("On to the rest of the old chain.", "p = q")])]),
      S("Use the new table.", "H.table = B"), S("", "H.capacity = new_capacity"), ENS],
    insert: () => [REQ,
      S("The key may already be there…", "let p = ht_find(H, k)"),
      S("…then just update its value.", "if p != null", [S("", "p.value = v"), S("", "return")]),
      S("Otherwise the new entry goes at the front of its bucket's chain.", "let i = ht_hash(k, H.capacity)"),
      S("", "let e = new(k, v)"), S("", "e.next = H.table[i]"), S("", "H.table[i] = e"),
      S("One more entry.", "H.size += 1"),
      S("More entries than buckets: double the table, so chains stay short on average.", "if H.size > H.capacity", [S("", "ht_resize(H, H.capacity * 2)")]),
      ENS],
    del: () => [REQ,
      S("The key's bucket.", "let i = ht_hash(k, H.capacity)"),
      S("Walk its chain with `prev` one step behind.", "let prev = null"), S("", "let p = H.table[i]"),
      S("", "while p != null and p.key != k", [S("", "prev = p"), S("", "p = p.next")]),
      S("Not there: nothing to delete.", "if p == null", [S("", "return false")]),
      S("Unlink `p`: the bucket or the entry before it skips it.", "if prev == null", [S("", "H.table[i] = p.next")]),
      S("", "else", [S("", "prev.next = p.next")]),
      S("Free it.", "free p"), S("One entry fewer.", "H.size -= 1"), ENS, S("", "return true")],
  };

  // ---------------------------------------------------------------- open addressing
  const OAD = {
    isHt: () => [
      S("The table really has `capacity` slots.", "if H.capacity < 1 or len(H.table) != H.capacity", [S("", "return false")]),
      S("Count entries (`full`) and slots in use, entries or tombstones (`nonempty`).", "let full = 0"), S("", "let nonempty = 0"),
      S("Look at every slot.", "for i = 0 to H.capacity", [
        S("", "let p = H.table[i]"),
        S("", "if p != null", [S("", "nonempty += 1"),
          S("A real entry (not a tombstone):", "if p != H.tomb", [S("", "full += 1"),
            S("searching for its key must find it right here: reachable from its home slot, with no copy of the key earlier on the way.", "if ht_find_slot(H, p.key) != i", [S("", "return false")])])])]),
      S("The counts match, and at most half the slots are in use.", "return full == H.size and nonempty == H.used and H.used * 2 <= H.capacity")],
    findSlot: () => [
      S("The key's home slot.", "let h = ht_hash(k, H.capacity)"),
      S("Try slot after slot, at most `capacity` times:", "for t = 0 to H.capacity", [
        S("", "let j = ht_probe(h, t, H.capacity)"),
        S("An empty slot ends the search: the key isn't in the table.", "if H.table[j] == null", [S("", "return -1")]),
        S("A real entry (not a tombstone) with the key: found.", "if H.table[j] != H.tomb and H.table[j].key == k", [S("", "return j")])]),
      { ...S("Not found.", "return -1"), noMutate: "unreachable while at most half the slots are in use: an empty slot always ends the loop" }],
    contains: () => [S("The key is there when `ht_find_slot` finds its slot.", "return ht_find_slot(H, k) >= 0")],
    get: () => [REQ, S("**Requires**: the key is in the table.", "assert ht_contains(H, k)"),
      S("", "let j = ht_find_slot(H, k)"), S("The value in its slot.", "return H.table[j].value")],
    resize: () => [
      S("**Requires**: the entries fill at most half of the new table.", "assert H.size * 2 <= new_capacity"),
      S("A new table of empty slots.", "let B = allocn(new_capacity)"),
      S("For every old slot…", "for i = 0 to H.capacity", [
        S("", "let p = H.table[i]"),
        S("…that holds a real entry (tombstones are simply dropped):", "if p != null and p != H.tomb", [
          S("probe the new table from the key's new home slot…", "let h = ht_hash(p.key, new_capacity)"), S("", "let t = 0"), S("", "let j = ht_probe(h, 0, new_capacity)"),
          S("…to the first empty slot…", "while B[j] != null", [S("", "t += 1"), S("", "j = ht_probe(h, t, new_capacity)")]),
          S("…and put the entry there.", "B[j] = p")])]),
      S("Use the new table. No tombstones left: the slots in use are exactly the entries.", "H.table = B"), S("", "H.capacity = new_capacity"), S("", "H.used = H.size"), ENS],
    insert: () => [REQ,
      S("The key may already be there…", "let j = ht_find_slot(H, k)"),
      S("…then just update its value.", "if j >= 0", [S("", "H.table[j].value = v"), S("", "return")]),
      S("Otherwise probe from the key's home slot…", "let h = ht_hash(k, H.capacity)"), S("", "let t = 0"), S("", "let i = ht_probe(h, 0, H.capacity)"),
      S("…past full slots: an empty slot or a tombstone can take the new entry.", "while H.table[i] != null and H.table[i] != H.tomb", [S("", "t += 1"), S("", "i = ht_probe(h, t, H.capacity)")]),
      S("A fresh slot (not a reused tombstone) is one more slot in use.", "if H.table[i] == null", [S("", "H.used += 1")]),
      S("Put the entry there.", "H.table[i] = new(k, v)"), S("One more entry.", "H.size += 1"),
      S("More than half the slots in use: double the table, so probes stay short (and an empty slot always exists).", "if H.used * 2 > H.capacity", [S("", "ht_resize(H, H.capacity * 2)")]),
      ENS],
    del: () => [REQ,
      S("Find the key's slot.", "let j = ht_find_slot(H, k)"),
      S("Not there: nothing to delete.", "if j < 0", [S("", "return false")]),
      S("Free the entry.", "free H.table[j]"),
      S("Leave a **tombstone**, not an empty slot: keys that probed past this slot must still be found.", "H.table[j] = H.tomb"),
      S("One entry fewer (the slot stays in use).", "H.size -= 1"), ENS, S("", "return true")],
  };

  // ---------------------------------------------------------------- declarations, per language
  function decl(c, o) {
    const L = c.lang, K = c.T, ch = chain(o);
    const dk = { py: o.keys === "int" ? "0" : '""', cpp: o.keys === "int" ? "0" : '""', java: o.keys === "int" ? "0" : '""', rs: o.keys === "int" ? "0" : "String::new()" }[L];
    const say = {
      entry: ch ? "An entry holds a key, its value, and the next entry in the same bucket." : "An entry holds a key and its value.",
      size: "`size`: how many entries there are.",
      capacity: `\`capacity\`: how many ${ch ? "buckets" : "slots"} the table has. It starts at 8${ch ? "" : ", a power of two (quadratic probing needs one to reach every slot)"}.`,
      table: ch ? "`table`: the buckets, each the first entry of a chain, or null." : "`table`: the slots, each an entry, a tombstone, or null (empty).",
      used: "`used`: slots that aren't empty: entries plus tombstones. It decides when to grow.",
      tomb: "`tomb`: one special entry that marks deleted slots (a **tombstone**).",
    };
    const fields = ch ? ["size", "capacity", "table"] : ["size", "used", "capacity", "table", "tomb"];
    if (L === "py") {
      const init = { size: "self.size = 0", used: "self.used = 0", capacity: "self.capacity = 8", table: "self.table = [None] * self.capacity", tomb: `self.tomb = Entry(${dk}, 0)` };
      return [
        { say: say.entry, code: ["class Entry:"], body: [{ say: "", code: [`def __init__(self, key: ${K}, value: int) -> None:`], body: [
          { say: "", code: ["self.key = key"] }, { say: "", code: ["self.value = value"] },
          ...(ch ? [{ say: "", code: ["self.next: Entry | None = None"] }] : [])] }] },
        { say: "The table.", code: ["class HashTable:"], body: [{ say: "The constructor makes an empty table.", code: ["def __init__(self) -> None:"],
          body: fields.map((f) => ({ say: say[f], code: [init[f]] })) }] },
      ];
    }
    if (L === "cpp") {
      const init = { size: "int size = 0;", used: "int used = 0;", capacity: "int capacity = 8;", table: "std::vector<Entry*> table = std::vector<Entry*>(8, nullptr);", tomb: `Entry* tomb = new Entry(${dk}, 0);` };
      return [
        { say: "At the top of the file: `<vector>`, `<string>`, `<iostream>` for printing, `<cassert>` for contracts.", code: ["#include <vector>", "#include <string>", "#include <iostream>", "#include <cassert>"] },
        { say: say.entry, code: ["struct Entry {"], close: ["};"], body: [
          { say: "", code: [`${K} key;`] }, { say: "", code: ["int value;"] }, ...(ch ? [{ say: "", code: ["Entry* next = nullptr;"] }] : []),
          { say: "A constructor taking the key and the value.", code: [`Entry(${K} key, int value) : key(key), value(value) {}`] }] },
        { say: "The table. The defaults make an empty one.", code: ["struct HashTable {"], close: ["};"], body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    if (L === "java") {
      const init = { size: "int size;", used: "int used;", capacity: "int capacity = 8;", table: "Entry[] table = new Entry[8];", tomb: `Entry tomb = new Entry(${dk}, 0);` };
      return [
        { say: say.entry, code: ["class Entry {"], close: ["}"], body: [
          { say: "", code: [`${K} key;`] }, { say: "", code: ["int value;"] }, ...(ch ? [{ say: "", code: ["Entry next;"] }] : []),
          { say: "A constructor taking the key and the value.", code: [`Entry(${K} key, int value) {`], close: ["}"], body: [{ say: "", code: ["this.key = key;"] }, { say: "", code: ["this.value = value;"] }] }] },
        { say: "The table. Every function goes inside this class, as a `static` method.", code: ["class HashTable {"], close: ["}"], deferClose: true, cls: O,
          body: fields.map((f) => ({ say: say[f], code: [init[f]] })) },
      ];
    }
    const rinit = { size: "size: 0", used: "used: 0", capacity: "capacity: 8", table: "table: vec![ptr::null_mut(); 8]", tomb: `tomb: Entry::new(${dk}, 0)` };
    const rtype = { size: "size: usize,", used: "used: usize,", capacity: "capacity: usize,", table: "table: Vec<*mut Entry>,", tomb: "tomb: *mut Entry," };
    return [
      { say: "At the very top: allow capital variable names (the table is called `H`), and import `std::ptr` for null pointers.", code: ["#![allow(non_snake_case)]", "use std::ptr;"] },
      { say: say.entry + " Entries are reached through raw pointers, like the linked-list nodes of 13.04.", code: ["struct Entry {"], close: ["}"], body: [
        { say: "", code: [`key: ${K},`] }, { say: "", code: ["value: i32,"] }, ...(ch ? [{ say: "", code: ["next: *mut Entry,"] }] : [])] },
      { say: "`Entry::new` puts an entry on the heap and returns a raw pointer to it.", code: ["impl Entry {"], close: ["}"], body: [
        { say: "", code: [`fn new(key: ${K}, value: i32) -> *mut Entry {`], close: ["}"], body: [
          { say: "", code: [`Box::into_raw(Box::new(Entry { key, value${ch ? ", next: ptr::null_mut()" : ""} }))`] }] }] },
      { say: "The table.", code: ["struct HashTable {"], close: ["}"], body: fields.map((f) => ({ say: say[f], code: [rtype[f]] })) },
      { say: "A constructor `HashTable::new()`: an empty table.", code: ["impl HashTable {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: [`HashTable { ${fields.map((f) => rinit[f]).join(", ")} }`] }] }] },
    ];
  }

  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const ch = chain(o), X = ch ? CHN : OAD;
    const on = C.contractsOn(o);
    const hashGoal = `{fn:ht_hash(k, m)} turns a key into a slot number in [0, m). ` + (o.keys === "int" ? "For integer keys the key itself works, made non-negative." :
      "For strings, a polynomial hash: `h = h * 31 + code`, character by character, reduced mod `m` at every step (the same numbers in every language, and no overflow).");
    return [
      { id: "hash", title: ch ? "The table and the hash function" : "The table, hashing and probing",
        goal: (ch ? "An entry holds a key, a value and the next entry in its bucket; the table `H` holds `capacity` buckets. " :
          "An entry holds a key and a value; the table `H` holds `capacity` slots, each an entry, a **tombstone** (a deleted entry) or null. ") +
          hashGoal + (ch ? "" : ` Open addressing also needs {fn:ht_probe(h, t, m)}: the \`t\`-th slot to try for a key whose home slot is \`h\` (${o.method === "linear" ? "linear: `h + t`" : "quadratic: `h + t(t + 1)/2`"}, mod \`m\`).`),
        decl: (c) => decl(c, o),
        fns: [{ name: "ht_hash", steps: hashSteps(o), headerSay: "Write {fn:ht_hash}." }, ...(ch ? [] : [fn("ht_probe", () => probeSteps(o), "Write {fn:ht_probe}.")])],
        tests: ["hash"] },
      { id: "table", title: on ? "The contract" : ch ? "Size" : "Finding a key's slot",
        goal: (ch ? "" : "{fn:ht_find_slot(H, k)} probes from `k`'s home slot until it finds `k` (its slot) or an empty slot (-1); tombstones don't stop it. ") +
          (on ? "Write {fn:is_ht(H)}: " + (ch ? "every entry sits in the bucket its key hashes to, and there are exactly `size` of them." :
            "searching for each entry's key finds it at its own slot, the counts match, and at most half the slots are in use.") + " Every operation checks it."
            : "And {fn:ht_size(H)}, the number of entries."),
        fns: [...(ch ? [] : [fn("ht_find_slot", X.findSlot, "First {fn:ht_find_slot}: probe for a key, stopping at an empty slot.")]),
          fn("is_ht", X.isHt, "Write {fn:is_ht}, the representation invariant."), fn("ht_size", () => [S("The number of entries.", "return H.size")], "Write {fn:ht_size}.")],
        contractNote: true, tests: ["hash", "table"] },
      { id: "lookup", title: "Lookup",
        goal: ch ? "{fn:ht_find(H, k)} walks the chain of `k`'s bucket and returns its entry, or null. {fn:ht_contains} and {fn:ht_get} are built on it. Expected O(1): chains are short."
          : "{fn:ht_contains} and {fn:ht_get} are built on {fn:ht_find_slot}. Expected O(1): with at most half the slots in use, probes are short.",
        fns: ch ? [fn("ht_find", X.find, "Write {fn:ht_find}."), fn("ht_contains", X.contains, "Write {fn:ht_contains}."), fn("ht_get", X.get, "Write {fn:ht_get}, which requires the key.")]
          : [fn("ht_contains", X.contains, "Write {fn:ht_contains}."), fn("ht_get", X.get, "Write {fn:ht_get}, which requires the key.")],
        tests: ["hash", "table", "lookup"] },
      { id: "resize", title: "Resize",
        goal: "{fn:ht_resize(H, new_capacity)} moves every entry into a new table. Each key is **rehashed**: its slot depends on the capacity. O(n), but doubling makes it rare: O(1) amortized per insert." +
          (ch ? "" : " Tombstones are dropped along the way."),
        fns: [fn("ht_resize", X.resize, "Write {fn:ht_resize}.")], tests: ["hash", "table", "lookup", "resize"] },
      { id: "insert", title: "Insert",
        goal: "{fn:ht_insert(H, k, v)} sets `k`'s value: it updates an existing entry, or adds a new one " +
          (ch ? "at the front of its bucket's chain, and doubles the table when there are more entries than buckets." : "in the first empty slot or tombstone along `k`'s probe sequence, and doubles the table when more than half its slots are in use."),
        fns: [fn("ht_insert", X.insert, "Write {fn:ht_insert}.")], tests: ["hash", "table", "lookup", "resize", "insert"] },
      { id: "delete", title: "Delete",
        goal: ch ? "{fn:ht_delete(H, k)} unlinks `k`'s entry from its chain and returns true, or returns false if `k` isn't there."
          : "{fn:ht_delete(H, k)} replaces `k`'s entry with a **tombstone** and returns true (false if `k` isn't there). An empty slot would cut the probe sequences of keys stored after it.",
        fns: [fn("ht_delete", X.del, "Write {fn:ht_delete}.")], tests: ["hash", "table", "lookup", "resize", "insert", "delete"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const KEYS = { int: [3, 11, 19, 5, 13, 0, 8, -5, 21, 6, 14, 30], string: ["ant", "bee", "cat", "dog", "eel", "fox", "gnu", "hen", "jay", "owl", "pig", "ram"] };
  const ABSENT = { int: [27, 4, -3], string: ["yak", "cow", "zebu"] };

  function showFn(o) {
    const tail = [S("", "if not is_ht(H)", [S("", 'print " BROKEN: is_ht(H) is false"'), S("", "println"), S("", "return")]), S("", "println")];
    if (chain(o)) {
      return [S("", 'print "{"'), S("", "let first = true"),
        S("", "for i = 0 to len(H.table)", [S("", "let p = H.table[i]"), S("", "if p != null", [
          S("", "if not first", [S("", 'print ", "')]), S("", "first = false"), S("", "print i"), S("", 'print ": ["'), S("", "let c = 0"),
          S("", "while p != null and c <= H.size", [S("", "if c > 0", [S("", 'print ", "')]), S("", "print p.key"), S("", 'print "="'), S("", "print p.value"), S("", "p = p.next"), S("", "c += 1")]),
          S("", 'print "]"')])]),
        S("", 'print "} size="'), S("", "print H.size"), S("", 'print " capacity="'), S("", "print H.capacity"), ...tail];
    }
    return [S("", 'print "["'),
      S("", "for i = 0 to len(H.table)", [S("", "if i > 0", [S("", 'print ", "')]), S("", "let p = H.table[i]"),
        S("", "if p == null", [S("", 'print "_"')]), S("", "elif p == H.tomb", [S("", 'print "X"')]),
        S("", "else", [S("", "print p.key"), S("", 'print "="'), S("", "print p.value")])]),
      S("", 'print "] size="'), S("", "print H.size"), S("", 'print " used="'), S("", "print H.used"), S("", 'print " capacity="'), S("", "print H.capacity"), ...tail];
  }

  // broken tables for is_ht, built from where the model put the keys
  function breaks(o, m) {
    const out = [[["H.size += 1"], "H.size is one too big"], [["H.capacity = H.capacity * 2"], "H.capacity doesn't match the table"]];
    const cap = m.st.cap;
    if (chain(o)) {
      const b = m.st.buckets.findIndex((x) => x.length), e = m.st.buckets.findIndex((x) => !x.length);
      out.push([[`H.table[${e}] = H.table[${b}]`, `H.table[${b}] = null`], "a chain is in the wrong bucket"]);
      out.push([[`H.table[${b}].next = H.table[${b}]`], "a chain loops back on itself"]);
    } else {
      out.push([["H.used += 1"], "H.used is one too big"]);
      // an entry moved out of its home slot into an empty one: its home slot is empty now, so a search stops there
      const empty = m.st.slots.findIndex((x) => x === null);
      const home = m.st.slots.findIndex((x, s) => x && x !== TOMB && hashJS(o, x.k, cap) === s);
      if (home >= 0) out.push([[`H.table[${empty}] = H.table[${home}]`, `H.table[${home}] = null`], "an entry is cut off from its home slot"]);
      out.push([[`H.table[${empty}] = H.tomb`], "a tombstone that used doesn't count"]);
    }
    return out;
  }

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const shown = (st, call, label) => ({ ir: [say(`${label}: `), S("", call), S("", "show(H)")], out: `${label}: ${st.m.show()}` });
    return {
      hash: ([k, m]) => C.ops.value(o, `ht_hash(${lit(k)}, ${m})`, `${nm("ht_hash")}(${k}, ${m})`, hashJS(o, k, m)),
      probe: ([h, t, m]) => C.ops.value(o, `ht_probe(${h}, ${t}, ${m})`, `${nm("ht_probe")}(${h}, ${t}, ${m})`, probeJS(o, h, t, m)),
      new: (x, st) => { st.m = model(o); return { ir: [S("", "make HashTable H"), say("new table: "), S("", "show(H)")], out: `new table: ${st.m.show()}` }; },
      fill: (kvs, st) => { kvs.forEach(([k, v]) => st.m.insert(k, v)); return { ir: kvs.map(([k, v]) => S("", `ref_ht_insert(H, ${lit(k)}, ${v})`)) }; },
      is_ht: () => C.ops.bool(o, "is_ht(H)", `${nm("is_ht")}(H)`, true),
      size: (x, st) => C.ops.value(o, "ht_size(H)", `${nm("ht_size")}(H)`, st.m.st.size),
      contains: (k, st) => C.ops.bool(o, `ht_contains(H, ${lit(k)})`, `${nm("ht_contains")}(H, ${k})`, !!st.m.find(k)),
      get: (k, st) => C.ops.value(o, `ht_get(H, ${lit(k)})`, `${nm("ht_get")}(H, ${k})`, st.m.find(k).v),
      resize: (n, st) => { st.m.resize(n); return shown(st, `ht_resize(H, ${n})`, `${nm("ht_resize")}(H, ${n})`); },
      insert: ([k, v], st) => { st.m.insert(k, v); return shown(st, `ht_insert(H, ${lit(k)}, ${v})`, `${nm("ht_insert")}(H, ${k}, ${v})`); },
      delete: (k, st) => {
        const r = st.m.remove(k);
        return { ir: [say(`${nm("ht_delete")}(H, ${k}) = `), S("", `printb ht_delete(H, ${lit(k)})`), say(": "), S("", "show(H)")], out: `${nm("ht_delete")}(H, ${k}) = ${r}: ${st.m.show()}` };
      },
      break_empty: () => ({ ir: [S("", "H.capacity = 16"), say(`after setting H.capacity to 16 on an empty table: ${nm("is_ht")}(H) = `), S("", "printb is_ht(H)"), S("", "println")],
        out: `after setting H.capacity to 16 on an empty table: ${nm("is_ht")}(H) = false` }),
      break: (k, st) => {
        const [stms, desc] = breaks(o, st.m)[k];
        return { ir: [...stms.map((x) => S("", x)), say(`after ${desc}: ${nm("is_ht")}(H) = `), S("", "printb is_ht(H)"), S("", "println")],
          out: `after ${desc}: ${nm("is_ht")}(H) = false` };
      },
    };
  }

  function testGroups(names, o, seed) {
    const K = KEYS[o.keys], Z = ABSENT[o.keys];
    const kv = (n, from = 0) => K.slice(from, from + n).map((k, i) => [k, from + i + 1]);
    const filled = (n) => { const m = model(o); kv(n).forEach(([k, v]) => m.insert(k, v)); return m; };
    const G = {
      hash: [[...K.slice(0, 8).map((k) => ["hash", [k, 8]]), ...K.slice(0, 4).map((k) => ["hash", [k, 16]]), ["hash", [K[0], 1]]],
        ...(chain(o) ? [] : [[...[0, 1, 2, 3, 4, 7].map((t) => ["probe", [5, t, 8]]), ["probe", [15, 3, 16]]]])],
      table: [
        [["new"], ["is_ht"], ["size"]],
        [["new"], ["fill", kv(4)], ["is_ht"], ["size"]],
        ...breaks(o, filled(4)).map((_, k) => [["new"], ["fill", kv(4)], ["break", k]]),
        [["new"], ["break_empty"]],
      ],
      lookup: [[["new"], ["fill", kv(4)], ...K.slice(0, 4).map((k) => ["contains", k]), ...Z.map((k) => ["contains", k]), ["get", K[0]], ["get", K[2]], ["get", K[3]]]],
      resize: [[["new"], ["fill", kv(3)], ["resize", 16], ["resize", chain(o) ? 2 : 8], ["contains", K[1]]]],
      insert: [
        [["new"], ...kv(4).map((x) => ["insert", x]), ["insert", [K[1], 42]], ["insert", [K[4], 5]]],
        [["new"], ...kv(10).map((x) => ["insert", x]), ["size"], ["get", K[7]]],
      ],
      delete: [
        [["new"], ["fill", kv(5)], ["delete", K[0]], ["contains", K[1]], ["contains", K[2]], ["delete", Z[0]], ["delete", K[3]], ["insert", [K[5], 6]],
          ["delete", K[0]], ["contains", K[4]], ["size"], ["insert", [K[0], 9]],
          ...kv(5, 6).map((x) => ["insert", x]), ["size"]],
      ],
    };
    const groups = names.flatMap((n) => G[n]);
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
        else if (m.find(k)) ops.push(["get", k]);
        else ops.push(["size"]);
      }
      groups.push(ops);
    }
    return groups;
  }

  function refFns(o, names) {
    if (!names.includes("table")) return [];
    const ch = chain(o), X = ch ? CHN : OAD;
    return [{ name: "ht_hash", steps: hashSteps(o) }, ...(ch ? [{ name: "ht_find", steps: X.find() }] : [{ name: "ht_probe", steps: probeSteps(o) }, { name: "ht_find_slot", steps: X.findSlot() }]),
      { name: "is_ht", steps: X.isHt() }, { name: "ht_resize", steps: X.resize() }, { name: "ht_insert", steps: X.insert() }, { name: "show", steps: showFn(o) }];
  }

  function problem(o) {
    const L = o.lang, ch = chain(o);
    const sig = C.headers(DEF, o);
    const what = {
      ht_hash: o.keys === "int" ? "the key reduced into [0, m), never negative" : "h = (h * 31 + character code) % m over the characters, starting at 0",
      ...(ch ? {} : { ht_probe: o.method === "linear" ? "(h + t) % m" : "(h + t(t + 1)/2) % m" }),
      is_ht: ch ? "every entry in its key's bucket, exactly size entries" : "every entry reachable by probing, counts match size and used, used * 2 <= capacity",
      ht_size: "the number of entries",
      ...(ch ? { ht_find: "the key's entry, or null" } : { ht_find_slot: "the key's slot, or -1; stops at an empty slot, skips tombstones" }),
      ht_contains: "true if the key is in the table",
      ht_get: "the key's value (requires the key)",
      ht_resize: "rehash every entry into a new table of new_capacity " + (ch ? "buckets (front of each chain)" : "slots (first empty slot along the probe sequence); used becomes size"),
      ht_insert: "update the value, or add an entry " + (ch ? "at the front of its chain; then if size > capacity, resize to capacity * 2" : "in the first empty slot or tombstone; then if used * 2 > capacity, resize to capacity * 2"),
      ht_delete: ch ? "unlink the key's entry; false if it isn't there" : "replace the key's entry with the tombstone; false if it isn't there",
    };
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Entry.** Fields \`key\`, \`value\`${ch ? ", `next`" : ""}. **HashTable.** Fields ${(ch ? ["size", "capacity", "table"] : ["size", "used", "capacity", "table", "tomb"]).map((f) => `\`${f}\``).join(", ")}; ` +
        `an empty table has capacity 8.`, "",
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_ht", L)}(H)\` at the start and end of every operation. ${L === "java" ? "Put the functions inside `class HashTable` as static methods. " : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored), prints the whole table after every change, and runs every test from the tutorial plus random ones.",
    ].join("\n");
  }

  const DEF = {
    id: "hash-tables", num: "13.08", title: "Hash tables", heading: "Build a hash table, step by step", diagramLabel: "Your table",
    OPTIONS, DEFAULTS, FNS, STRUCTS, ctxExtra, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
    CONTRACT_OPS: ["is_ht", "break", "break_empty"],
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

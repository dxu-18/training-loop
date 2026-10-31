// 13.03 Queues: on a linked list with a dummy node (CMU 15-122), or on two stacks (amortized O(1)), in Python, C++,
// Java or Rust. The two-stack version builds its stack with 13.02's steps.
(function (root) {
  "use strict";
  const DL = root.DSALang || (typeof require !== "undefined" ? require("./lang.js") : null);
  const C = root.DSACommon || (typeof require !== "undefined" ? require("./common.js") : null);
  const STK = ((root.DSATutorials || {}).stacks || (typeof require !== "undefined" ? require("./stack.js") : null))._internals;
  const { S, fname, VALUES } = C;

  const OPTIONS = [
    { id: "lang", label: "Language", choices: [["py", "Python"], ["cpp", "C++"], ["java", "Java"], ["rs", "Rust"]] },
    { id: "type", label: "Elements", choices: [["int", "int"], ["generic", "generic (T)"]],
      help: { int: "Every element is an integer.", generic: "Any element type: a type parameter T (templates in C++, generics in Java and Rust, type parameters in Python)." } },
    { id: "impl", label: "Built on", choices: [["list", "a linked list"], ["stacks", "two stacks"]],
      help: { list: "A linked list that ends in a dummy node: enq writes at the back, deq takes the front. Every operation O(1).",
        stacks: "Two stacks: enq pushes onto an inbox, deq pops from an outbox, refilled from the inbox when it runs empty. O(1) amortized." } },
  ];
  const DEFAULTS = { lang: "py", type: "int", impl: "list" };
  const onStacks = (o) => o.impl === "stacks";
  const describe = (o) => `queue of ${o.type === "int" ? "int" : "generic"} elements on ${onStacks(o) ? "two stacks" : "a linked list with a dummy node"}, in ${DL.LANGS[o.lang].name}`;

  function complexity(o) {
    const st = onStacks(o);
    return [
      { id: "space", label: "Space", big: "O(n)", note: st ? "n nodes in two stacks" : "n nodes × (element + 1 pointer) + 1 dummy" },
      { id: "enq", label: "Enqueue", big: "O(1)", note: st ? "push onto the inbox" : "write into the dummy, add a new one" },
      { id: "deq", label: st ? "Dequeue (amortized)" : "Dequeue", big: "O(1)", note: st ? "each element moves inbox → outbox at most once" : "advance `Q.front`" },
      { id: "deq_worst", label: "One dequeue, worst case", big: st ? "O(n)" : "O(1)", note: st ? "an empty outbox: all n elements move over" : "every call, not just on average", extra: true },
      { id: "peek", label: st ? "Peek (amortized)" : "Peek", big: "O(1)", note: st ? "may refill the outbox first" : "read the front's element" },
      { id: "size", label: "Size", big: "O(1)", note: st ? "`inbox.size + outbox.size`" : "kept in `Q.size`" },
      { id: "empty", label: "Is empty", big: "O(1)", note: st ? "both stacks empty" : "`Q.size == 0`" },
      { id: "search", label: "Search", big: "O(n)", note: "not a queue operation: look at every element", extra: true },
      { id: "is_queue", label: "Contract is_queue", big: "O(n)", note: "walks the queue, so contracts make every call O(n) while testing" },
    ];
  }

  function diagram(o) {
    const v = o.type === "int" ? ["10", "20", "30", "40"] : ["ant", "bee", "cat", "dog"];
    if (!onStacks(o)) {
      return {
        rows: [[{ box: v[0], tags: ["Q.front"] }, { arrow: "→" }, { box: v[1], tags: [] }, { arrow: "→" }, { box: v[2], tags: [] }, { arrow: "→" },
          { box: "dummy", tags: ["Q.back"], cls: "dummy" }, { arrow: "→" }, { text: "null" }]],
        caption: "The queue is the segment [Q.front, Q.back): Q.back is a dummy node. enq writes into the dummy and adds a new one after it; deq takes the front. Empty = Q.front is Q.back.",
      };
    }
    return {
      rows: [
        [{ label: "outbox" }, { box: v[0], tags: ["Q.outbox.top"] }, { arrow: "→" }, { box: v[1], tags: [] }, { arrow: "→" }, { text: "null" }],
        [{ label: "inbox" }, { box: v[3], tags: ["Q.inbox.top"] }, { arrow: "→" }, { box: v[2], tags: [] }, { arrow: "→" }, { text: "null" }],
      ],
      caption: `deq pops from Q.outbox (oldest on top); enq pushes onto Q.inbox (newest on top). When the outbox runs empty, every element moves over, which reverses them into queue order. This queue: ${v.join(", ")}.`,
    };
  }

  // ---------------------------------------------------------------- specs
  const MQ = "mut:Queue", RQ = "ref:Queue", MS = "mut:Stack", RS = "ref:Stack";
  function FNS(o) {
    const show = { harness: true, bounds: ["std::fmt::Display"], ret: "void" };
    const queue = (peekKind) => ({
      is_queue: { params: [["Q", RQ]], ret: "bool", owner: "Queue" },
      queue_empty: { params: [["Q", RQ]], ret: "bool", owner: "Queue" },
      enq: { params: [["Q", MQ], ["x", "T"]], ret: "void", owner: "Queue" },
      deq: { params: [["Q", MQ]], ret: "T", owner: "Queue" },
      queue_peek: { params: [["Q", peekKind]], ret: "Tpeek", owner: "Queue" },
      queue_size: { params: [["Q", RQ]], ret: "int", owner: "Queue" },
      show: { params: [["Q", RQ]], ...show },
    });
    if (!onStacks(o)) return { is_segment: { params: [["start", "N"], ["end", "N"], ["n", "int"]], ret: "bool", owner: "Queue" }, ...queue(RQ) };
    return {
      is_segment: { params: [["start", "N"], ["end", "N"], ["n", "int"]], ret: "bool", owner: "Stack" },
      is_stack: { params: [["S", RS]], ret: "bool", owner: "Stack" },
      stack_empty: { params: [["S", RS]], ret: "bool", owner: "Stack" },
      push: { params: [["S", MS], ["x", "T"]], ret: "void", owner: "Stack" },
      pop: { params: [["S", MS]], ret: "T", owner: "Stack" },
      peek: { params: [["S", RS]], ret: "Tpeek", owner: "Stack" },
      refill: { params: [["Q", MQ]], ret: "void", owner: "Queue" },
      ...queue(MQ),   // peeking may refill the outbox, so it changes Q
      show_items: { params: [["S", RS]], ...show },
      show_stack: { params: [["S", RS]], ...show },
    };
  }
  const STRUCTS = (o) => (onStacks(o)
    ? { Stack: { fields: { top: "N", size: "int" } }, Queue: { fields: { inbox: "V:Stack", outbox: "V:Stack" } } }
    : { Queue: { fields: { front: "N", back: "N", size: "int" } } });

  // ---------------------------------------------------------------- the functions: on a list
  const REQ = S("**Requires**: the queue is valid when the function starts.", "assert is_queue(Q)");
  const ENS = S("**Ensures**: the queue is still valid when it returns.", "assert is_queue(Q)");
  const NONEMPTY = S("**Requires**: the queue is not empty.", "assert not queue_empty(Q)");

  const list = {
    isQueue: () => [
      S("The dummy `Q.back` always exists and ends the queue.", "if Q.back == null or Q.back.next != null", [S("", "return false")]),
      S("The queue is the segment [`Q.front`, `Q.back`): exactly `Q.size` nodes.", "return is_segment(Q.front, Q.back, Q.size)")],
    queueEmpty: () => [S("Empty exactly when the size is 0 (then `Q.front` is the dummy).", "return Q.size == 0")],
    enq: () => [REQ,
      S("Write `x` into the dummy: it becomes the last element.", "Q.back.data = x"),
      S("Make a fresh dummy `d`.", "let d = new(default)"),
      S("The new last element points to it.", "Q.back.next = d"),
      S("`d` is the new end. No special case for an empty queue.", "Q.back = d"),
      S("One more element.", "Q.size += 1"), ENS],
    deq: () => [REQ, NONEMPTY,
      S("`n` is the front node.", "let n = Q.front"),
      S("The next node (the dummy, if `n` was the last element) is the new front.", "Q.front = n.next"),
      S("One element fewer.", "Q.size -= 1"),
      S("Save the element and free the node.", "let x = take n"), ENS,
      S("Return the element.", "return x")],
    peek: () => [REQ, NONEMPTY,
      S((c) => (c.lang === "rs" && c.generic ? "Return the front element without removing it (a reference `&T`: Rust can't move it out of the node)."
        : "Return the front element without removing it."), "return Q.front.data")],
    size: () => [S("The size is kept in the queue.", "return Q.size")],
  };

  // ---------------------------------------------------------------- the functions: on two stacks
  const two = {
    isQueue: () => [S("A queue is valid when both its stacks are.", "return is_stack(Q.inbox) and is_stack(Q.outbox)")],
    queueEmpty: () => [S("Empty when both stacks are.", "return stack_empty(Q.inbox) and stack_empty(Q.outbox)")],
    enq: () => [REQ, S("New elements go on top of the inbox.", "push(Q.inbox, x)"), ENS],
    refill: () => [REQ,
      S("Only when the outbox is empty…", "if stack_empty(Q.outbox)", [
        S("…move every element from the inbox to the outbox. Popping reverses them, so the oldest element ends up on top.", "while not stack_empty(Q.inbox)", [
          S("", "push(Q.outbox, pop(Q.inbox))")])]),
      ENS],
    deq: () => [REQ, NONEMPTY,
      S("Make sure the front element is on top of the outbox.", "refill(Q)"),
      S("Pop it.", "let x = pop(Q.outbox)"), ENS,
      S("Return it.", "return x")],
    peek: () => [REQ, NONEMPTY,
      S("Make sure the front element is on top of the outbox.", "refill(Q)"), ENS,
      S((c) => (c.lang === "rs" && c.generic ? "Peek at the outbox's top (a reference `&T`)." : "Peek at the outbox's top."), "return peek(Q.outbox)")],
    size: () => [S("The elements of both stacks.", "return Q.inbox.size + Q.outbox.size")],
  };

  // ---------------------------------------------------------------- declarations, per language
  function listDecl(c) {
    const g = c.generic, L = c.lang;
    const dflt = !g ? "0" : { py: "None", cpp: "T{}", java: "null", rs: "T::default()" }[L];
    const say = {
      front: "`front`: the first node, or the dummy itself when the queue is empty.",
      back: "`back`: the dummy node at the end. It holds no element; the queue is the segment [`Q.front`, `Q.back`).",
      size: "`size`: the number of elements.",
      ctor: "The constructor makes an empty queue: one dummy node, which is both the front and the back.",
    };
    if (L === "py") {
      return [{ say: g ? "Declare the class `Queue[T]`." : "Declare the class `Queue`.", code: [g ? "class Queue[T]:" : "class Queue:"], body: [
        { say: say.ctor, code: ["def __init__(self) -> None:"], body: [
          { say: say.back + " Its element is a placeholder that is never read.", code: [g ? "self.back: Node[T] = Node(None)" : "self.back = Node(0)"] },
          { say: say.front, code: ["self.front = self.back"] },
          { say: say.size, code: ["self.size = 0"] }] }] }];
    }
    if (L === "cpp") {
      const P = g ? "Node<T>*" : "Node*";
      return [{ say: g ? "Declare the queue struct, a template over `T`." : "Declare the queue struct `Queue`.",
        code: g ? ["template <typename T>", "struct Queue {"] : ["struct Queue {"], close: ["};"], body: [
          { say: say.front, code: [`${P} front;`] }, { say: say.back, code: [`${P} back;`] }, { say: say.size, code: ["int size = 0;"] },
          { say: say.ctor, code: ["Queue() {"], close: ["}"], body: [{ say: "", code: [`back = new ${g ? "Node<T>" : "Node"}(${dflt});`] }, { say: "", code: ["front = back;"] }] }] }];
    }
    if (L === "java") {
      return [{ say: (g ? "Declare the class `Queue<T>`." : "Declare the class `Queue`.") + " Every function from here on goes inside this class, as a `static` method that takes the queue `Q`.",
        code: [g ? "class Queue<T> {" : "class Queue {"], close: ["}"], deferClose: true, cls: "Queue", body: [
          { say: say.front, code: [`${g ? "Node<T>" : "Node"} front;`] }, { say: say.back, code: [`${g ? "Node<T>" : "Node"} back;`] }, { say: say.size, code: ["int size;"] },
          { say: say.ctor, code: ["Queue() {"], close: ["}"], body: [{ say: "", code: [`back = new ${g ? "Node<>" : "Node"}(${dflt});`] }, { say: "", code: ["front = back;"] }] }] }];
    }
    const N = g ? "Node<T>" : "Node";
    return [
      { say: g ? "Declare the struct `Queue<T>`." : "Declare the struct `Queue`.", code: [g ? "struct Queue<T> {" : "struct Queue {"], close: ["}"], body: [
        { say: say.front, code: [`front: *mut ${N},`] }, { say: say.back, code: [`back: *mut ${N},`] }, { say: say.size, code: ["size: usize,"] }] },
      { say: say.ctor + (g ? " The dummy needs some value of type `T`, hence `T: Default`." : ""),
        code: [g ? "impl<T: Default> Queue<T> {" : "impl Queue {"], close: ["}"], body: [
          { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [
            { say: "Make the dummy (its element is a placeholder that is never read).", code: [`let d = Node::new(${dflt});`] },
            { say: "", code: ["Queue { front: d, back: d, size: 0 }"] }] }] },
    ];
  }

  function twoDecl(c) {
    const g = c.generic, L = c.lang;
    const St = g ? "Stack<T>" : "Stack";
    const say = {
      inbox: "`inbox`: the stack that `enq` pushes onto (the back of the queue, newest on top).",
      outbox: "`outbox`: the stack that `deq` pops from (the front of the queue, oldest on top).",
    };
    if (L === "py") {
      return [{ say: g ? "Declare the class `Queue[T]`." : "Declare the class `Queue`.", code: [g ? "class Queue[T]:" : "class Queue:"], body: [
        { say: "The constructor makes an empty queue: two empty stacks.", code: ["def __init__(self) -> None:"], body: [
          { say: say.inbox, code: [g ? "self.inbox: Stack[T] = Stack()" : "self.inbox = Stack()"] },
          { say: say.outbox, code: [g ? "self.outbox: Stack[T] = Stack()" : "self.outbox = Stack()"] }] }] }];
    }
    if (L === "cpp") {
      return [{ say: (g ? "Declare the queue struct, a template over `T`." : "Declare the queue struct `Queue`.") + " Its two stacks start empty by themselves.",
        code: g ? ["template <typename T>", "struct Queue {"] : ["struct Queue {"], close: ["};"], body: [
          { say: say.inbox, code: [`${St} inbox;`] }, { say: say.outbox, code: [`${St} outbox;`] }] }];
    }
    if (L === "java") {
      return [{ say: (g ? "Declare the class `Queue<T>`." : "Declare the class `Queue`.") + " Its functions go inside it as `static` methods; they call the stack's as `Stack.push(...)`.",
        code: [g ? "class Queue<T> {" : "class Queue {"], close: ["}"], deferClose: true, cls: "Queue", body: [
          { say: say.inbox, code: [`${St} inbox = new ${g ? "Stack<>" : "Stack"}();`] },
          { say: say.outbox, code: [`${St} outbox = new ${g ? "Stack<>" : "Stack"}();`] }] }];
    }
    return [
      { say: g ? "Declare the struct `Queue<T>`." : "Declare the struct `Queue`.", code: [g ? "struct Queue<T> {" : "struct Queue {"], close: ["}"], body: [
        { say: say.inbox, code: [`inbox: ${St},`] }, { say: say.outbox, code: [`outbox: ${St},`] }] },
      { say: "A constructor `Queue::new()`: two empty stacks.", code: [g ? "impl<T> Queue<T> {" : "impl Queue {"], close: ["}"], body: [
        { say: "", code: ["fn new() -> Self {"], close: ["}"], body: [{ say: "", code: ["Queue { inbox: Stack::new(), outbox: Stack::new() }"] }] }] },
    ];
  }

  const RUST_WHY = "The queue's last node is pointed at from two places (the node before it and `Q.back`), which `Box` (one owner per node) can't express; raw pointers can. `std::collections::LinkedList` is built the same way.";

  // ---------------------------------------------------------------- steps
  function steps(o) {
    const fn = (name, body, headerSay) => ({ name, steps: body(), headerSay });
    const node = (goal) => ({ id: "node", title: "The node", goal, tests: ["node"],
      decl: (c) => C.nodeSteps(c, { varName: "Q", what: "queue", rustWhy: onStacks(o) ? STK.RUST_WHY : RUST_WHY }) });
    const contracts = "Every operation checks {fn:is_queue(Q)} when it starts and before it returns.";
    if (!onStacks(o)) {
      return [
        node("The queue is a linked list: elements leave at the front and join at the back. Each node holds one element (`data`) and a pointer to the next node. A pointer that points nowhere is null."),
        { id: "queue", title: "The queue and its contracts",
          goal: "The queue `Q` keeps its `front` node, a dummy `back` node and its `size`. Write the **contracts** first: {fn:is_segment(start, end, n)} " +
            "(exactly `n` nodes from `start` up to, not including, `end`) and {fn:is_queue(Q)}, true exactly when the segment [`Q.front`, `Q.back`) holds `Q.size` nodes. " + contracts,
          decl: (c) => listDecl(c),
          fns: [fn("is_segment", STK.isSegment, "Write {fn:is_segment}: does [`start`, `end`) hold exactly `n` nodes?"),
            fn("is_queue", list.isQueue, "Write {fn:is_queue}, the representation invariant."),
            fn("queue_empty", list.queueEmpty, "Write {fn:queue_empty}.")],
          contractNote: true, tests: ["node", "queue"] },
        { id: "enq", title: "Enqueue",
          goal: "{fn:enq(Q, x)} adds `x` at the back. The dummy makes it O(1) with no special cases: write `x` into the dummy, then add a fresh dummy after it.",
          fns: [fn("enq", list.enq, "Write the header: {fn:enq} takes the queue and the new element.")], tests: ["node", "queue", "enq"] },
        { id: "deq", title: "Dequeue",
          goal: "{fn:deq(Q)} removes the front element and returns it. It **requires** a non-empty queue. O(1).",
          fns: [fn("deq", list.deq, "Write the header: {fn:deq} returns the removed element.")], tests: ["node", "queue", "enq", "deq"] },
        { id: "peek_size", title: "Peek and size",
          goal: "{fn:queue_peek(Q)} returns the front element without removing it (it requires a non-empty queue), and {fn:queue_size(Q)} returns how many elements there are. Both O(1).",
          fns: [fn("queue_peek", list.peek, "Write {fn:queue_peek}."), fn("queue_size", list.size, "Write {fn:queue_size}.")],
          tests: ["node", "queue", "enq", "deq", "peek_size"] },
      ];
    }
    return [
      node("Two stacks make the queue, and each stack is a linked list whose first node is its top. Each node holds one element (`data`) and a pointer to the node below it. A pointer that points nowhere is null."),
      { id: "stack", title: "The stack and its contracts",
        goal: "First the stack (13.02 in brief): `S` keeps its `top` node and its `size`. Its contracts: {fn:is_segment(start, end, n)} and {fn:is_stack(S)}, true exactly when [`S.top`, null) holds `S.size` nodes.",
        decl: (c) => STK.stackDecl(c),
        fns: [fn("is_segment", STK.isSegment, "Write {fn:is_segment}: does [`start`, `end`) hold exactly `n` nodes?"),
          fn("is_stack", STK.isStack, "Write {fn:is_stack}."), fn("stack_empty", STK.stackEmpty, "Write {fn:stack_empty}.")],
        contractNote: true, tests: ["node", "stack"] },
      { id: "stack_ops", title: "Push, pop and peek",
        goal: "The stack's operations, all O(1): {fn:push(S, x)} puts `x` on top, {fn:pop(S)} removes and returns the top element, and {fn:peek(S)} returns it without removing it.",
        fns: [fn("push", STK.push, "Write {fn:push}."), fn("pop", STK.pop, "Write {fn:pop}."), fn("peek", STK.peek, "Write {fn:peek}.")],
        tests: ["node", "stack", "stack_ops"] },
      { id: "queue", title: "The queue and its contracts",
        goal: "The queue `Q` holds two stacks: `inbox`, where new elements arrive, and `outbox`, where the oldest wait on top. {fn:is_queue(Q)} holds when both stacks are valid. " + contracts,
        decl: (c) => twoDecl(c),
        fns: [fn("is_queue", two.isQueue, "Write {fn:is_queue}."), fn("queue_empty", two.queueEmpty, "Write {fn:queue_empty}.")],
        tests: ["node", "stack", "stack_ops", "queue"] },
      { id: "enq", title: "Enqueue",
        goal: "{fn:enq(Q, x)} pushes `x` onto the inbox. O(1).",
        fns: [fn("enq", two.enq, "Write the header: {fn:enq} takes the queue and the new element.")], tests: ["node", "stack", "stack_ops", "queue", "enq"] },
      { id: "deq", title: "Dequeue",
        goal: "{fn:deq(Q)} returns the oldest element, which is on top of the outbox, once the outbox has been refilled. {fn:refill(Q)} moves the whole inbox to the outbox, but **only when the outbox is empty**. " +
          "A single `deq` can then cost O(n), but each element moves at most once, so a sequence of operations costs O(1) per operation: **amortized** O(1).",
        fns: [fn("refill", two.refill, "First a helper, {fn:refill}."), fn("deq", two.deq, "Then {fn:deq}, which requires a non-empty queue.")],
        tests: ["node", "stack", "stack_ops", "queue", "enq", "deq"] },
      { id: "peek_size", title: "Peek and size",
        goal: "{fn:queue_peek(Q)} returns the front element without removing it (refilling first, like `deq`), and {fn:queue_size(Q)} adds up the two stacks' sizes.",
        fns: [fn("queue_peek", two.peek, "Write {fn:queue_peek}."), fn("queue_size", two.size, "Write {fn:queue_size}.")],
        tests: ["node", "stack", "stack_ops", "queue", "enq", "deq", "peek_size"] },
    ];
  }

  // ---------------------------------------------------------------- tests
  const LIST_BREAKS = [["Q.size += 1", "Q.size is one too big"], ["Q.size -= 1", "Q.size is one too small"], ["Q.back = Q.front", "Q.back points at the first node"],
    ["Q.back.next = Q.front", "the dummy's next is not null"], ["Q.back = null", "Q.back is null"]];
  const TWO_BREAKS = [["Q.inbox.size += 1", "Q.inbox.size is one too big"], ["Q.outbox.size += 1", "Q.outbox.size is one too big"], ["Q.inbox.top = null", "Q.inbox.top is null"]];

  // the checker's view of a queue: its elements and the first broken rule
  function showList() {
    const seg = "following next from Q.front does not reach Q.back after exactly Q.size nodes (check size and next pointers)";
    return [S("", 'print "["'), S("", "let p = Q.front"), S("", "let count = 0"),
      S("", "while p != null and p != Q.back and count < Q.size + 2", [S("", "if count > 0", [S("", 'print ", "')]), S("", "print p.data"), S("", "p = p.next"), S("", "count += 1")]),
      S("", 'print "] size="'), S("", "print Q.size"),
      ...[["Q.back != null", "Q.back is null"], ["Q.back.next == null", "the dummy's next must be null"], ["is_segment(Q.front, Q.back, Q.size)", seg]]
        .map(([c, m]) => S("", `if not (${c})`, [S("", `print " BROKEN: ${m}"`), S("", "println"), S("", "return")])),
      S("", "println")];
  }
  // two stacks: out=[top first] in=[top first]
  const showItems = () => [S("", 'print "["'), S("", "let p = S.top"), S("", "let count = 0"),
    S("", "while p != null and count < S.size + 2", [S("", "if count > 0", [S("", 'print ", "')]), S("", "print p.data"), S("", "p = p.next"), S("", "count += 1")]),
    S("", 'print "]"')];
  const showStack = () => [S("", "show_items(S)"), S("", 'print " size="'), S("", "print S.size"),
    S("", "if not is_stack(S)", [S("", 'print " BROKEN: following next from S.top does not reach null after exactly S.size nodes"'), S("", "println"), S("", "return")]),
    S("", "println")];
  const showTwo = () => [S("", 'print "out="'), S("", "show_items(Q.outbox)"), S("", 'print " in="'), S("", "show_items(Q.inbox)"),
    ...[["outbox", "Q.outbox"], ["inbox", "Q.inbox"]].map(([n, e]) => S("", `if not is_stack(${e})`, [S("", `print " BROKEN: ${e} is not a valid stack (check its size and next pointers)"`), S("", "println"), S("", "return")])),
    S("", "println")];

  function handlers(o) {
    const nm = (n) => fname(n, o.lang);
    const { lit, say } = C;
    const show = (A) => `[${A.join(", ")}] size=${A.length}`;
    const q = {
      node: C.ops.node(false),
      queue_empty: (x, st) => C.ops.bool(o, "queue_empty(Q)", `${nm("queue_empty")}(Q)`, st.size() === 0),
      queue_size: (x, st) => C.ops.value(o, "queue_size(Q)", `${nm("queue_size")}(Q)`, st.size()),
      is_queue: () => C.ops.bool(o, "is_queue(Q)", `${nm("is_queue")}(Q)`, true),
    };
    const brk = (B) => (k) => ({ ir: [S("", B[k][0]), say(`after ${B[k][1]}: ${nm("is_queue")}(Q) = `), S("", "printb is_queue(Q)"), S("", "println")],
      out: `after ${B[k][1]}: ${nm("is_queue")}(Q) = false` });
    if (!onStacks(o)) {
      return { ...q,
        new: (x, st) => { st.A = []; st.size = () => st.A.length; return { ir: [S("", "make Queue Q"), say("new queue: "), S("", "show(Q)")], out: `new queue: ${show(st.A)}` }; },
        fill: (xs, st) => { st.A.push(...xs); return { ir: xs.map((y) => S("", `ref_enq(Q, ${lit(y)})`)) }; },
        enq: (x, st) => { st.A.push(x); return { ir: [say(`enq(Q, ${x}): `), S("", `enq(Q, ${lit(x)})`), S("", "show(Q)")], out: `enq(Q, ${x}): ${show(st.A)}` }; },
        deq: (x, st) => { const y = st.A.shift(); return { ir: [say("deq(Q) = "), S("", "print deq(Q)"), say(": "), S("", "show(Q)")], out: `deq(Q) = ${y}: ${show(st.A)}` }; },
        queue_peek: (x, st) => C.ops.value(o, "queue_peek(Q)", `${nm("queue_peek")}(Q)`, st.A[0]),
        break: brk(LIST_BREAKS),
        segment: (d) => {
          const n = `Q.size${d === 0 ? "" : d > 0 ? ` + ${d}` : ` - ${-d}`}`;
          return C.ops.bool(o, `is_segment(Q.front, Q.back, ${n})`, `${nm("is_segment")}(Q.front, Q.back, ${n})`, d === 0);
        },
      };
    }
    // two stacks: the model keeps both stacks (top first), and moves the inbox over exactly when the code does
    const showQ = (st) => `out=[${st.out.join(", ")}] in=[${st.inb.join(", ")}]`;
    const refill = (st) => { if (!st.out.length) while (st.inb.length) st.out.unshift(st.inb.shift()); };
    const SB = STK.BREAKS;
    return { ...q,
      snew: (x, st) => { st.A = []; return { ir: [S("", "make Stack S"), say("new stack: "), S("", "show_stack(S)")], out: `new stack: ${show(st.A)}` }; },
      sfill: (xs, st) => { xs.forEach((y) => st.A.unshift(y)); return { ir: xs.map((y) => S("", `ref_push(S, ${lit(y)})`)) }; },
      push: (x, st) => { st.A.unshift(x); return { ir: [say(`push(S, ${x}): `), S("", `push(S, ${lit(x)})`), S("", "show_stack(S)")], out: `push(S, ${x}): ${show(st.A)}` }; },
      pop: (x, st) => { const y = st.A.shift(); return { ir: [say("pop(S) = "), S("", "print pop(S)"), say(": "), S("", "show_stack(S)")], out: `pop(S) = ${y}: ${show(st.A)}` }; },
      peek: (x, st) => C.ops.value(o, "peek(S)", "peek(S)", st.A[0]),
      stack_empty: (x, st) => C.ops.bool(o, "stack_empty(S)", `${nm("stack_empty")}(S)`, st.A.length === 0),
      is_stack: () => C.ops.bool(o, "is_stack(S)", `${nm("is_stack")}(S)`, true),
      sbreak: (k) => ({ ir: [S("", SB[k][0]), say(`after ${SB[k][1]}: ${nm("is_stack")}(S) = `), S("", "printb is_stack(S)"), S("", "println")],
        out: `after ${SB[k][1]}: ${nm("is_stack")}(S) = false` }),
      ssegment: (d) => {
        const n = `S.size${d === 0 ? "" : d > 0 ? ` + ${d}` : ` - ${-d}`}`;
        return C.ops.bool(o, `is_segment(S.top, null, ${n})`, `${nm("is_segment")}(S.top, null, ${n})`, d === 0);
      },
      new: (x, st) => { st.inb = []; st.out = []; st.size = () => st.inb.length + st.out.length;
        return { ir: [S("", "make Queue Q"), say("new queue: "), S("", "show(Q)")], out: `new queue: ${showQ(st)}` }; },
      qfill: (xs, st) => { xs.forEach((y) => st.inb.unshift(y)); return { ir: xs.map((y) => S("", `ref_push(Q.inbox, ${lit(y)})`)) }; },
      enq: (x, st) => { st.inb.unshift(x); return { ir: [say(`enq(Q, ${x}): `), S("", `enq(Q, ${lit(x)})`), S("", "show(Q)")], out: `enq(Q, ${x}): ${showQ(st)}` }; },
      deq: (x, st) => { refill(st); const y = st.out.shift(); return { ir: [say("deq(Q) = "), S("", "print deq(Q)"), say(": "), S("", "show(Q)")], out: `deq(Q) = ${y}: ${showQ(st)}` }; },
      queue_peek: (x, st) => { refill(st); return { ir: [say(`${nm("queue_peek")}(Q) = `), S("", "print queue_peek(Q)"), say(": "), S("", "show(Q)")], out: `${nm("queue_peek")}(Q) = ${st.out[0]}: ${showQ(st)}` }; },
      break: brk(TWO_BREAKS),
    };
  }

  function testGroups(names, o, seed) {
    const [a, b, c, d, e, f] = VALUES[o.type];
    const G = {
      node: [[["node", a]], [["node", f]]],
      enq: [[["new"], ["enq", a], ["enq", b], ["enq", c], ["is_queue"], ["queue_empty"]]],
      peek_size: [[["new"], ["queue_size"], ["enq", a], ["enq", b], ["queue_peek"], ["queue_size"], ["deq"], ["enq", c], ["queue_peek"], ["queue_size"],
        ["deq"], ["queue_peek"], ["queue_size"]]],
    };
    if (!onStacks(o)) {
      Object.assign(G, {
        queue: [
          [["new"], ["queue_empty"], ["is_queue"]],
          [["new"], ["fill", [a, b, c]], ["queue_empty"], ["is_queue"], ["segment", 0], ["segment", -1], ["segment", 1], ["segment", 2]],
          [["new"], ["fill", [d]], ["is_queue"], ["segment", 0]],
          ...LIST_BREAKS.map((_, k) => [["new"], ["fill", [a, b, c]], ["break", k]]),
        ],
        deq: [[["new"], ["enq", a], ["enq", b], ["deq"], ["enq", c], ["deq"], ["deq"], ["queue_empty"], ["enq", d], ["deq"], ["enq", e], ["enq", f], ["deq"]]],
      });
    } else {
      Object.assign(G, {
        stack: [
          [["snew"], ["stack_empty"], ["is_stack"]],
          [["snew"], ["sfill", [a, b, c]], ["stack_empty"], ["is_stack"], ["ssegment", 0], ["ssegment", -1], ["ssegment", 1], ["ssegment", 2]],
          [["snew"], ["sfill", [d]], ["is_stack"], ["ssegment", 0]],
          ...STK.BREAKS.map((_, k) => [["snew"], ["sfill", [a, b, c]], ["sbreak", k]]),
        ],
        stack_ops: [[["snew"], ["push", a], ["push", b], ["peek"], ["pop"], ["push", c], ["peek"], ["pop"], ["pop"], ["stack_empty"]]],
        queue: [
          [["new"], ["queue_empty"], ["is_queue"]],
          [["new"], ["qfill", [a, b, c]], ["queue_empty"], ["is_queue"]],
          ...TWO_BREAKS.map((_, k) => [["new"], ["qfill", [a, b, c]], ["break", k]]),
        ],
        deq: [[["new"], ["enq", a], ["enq", b], ["deq"], ["enq", c], ["deq"], ["deq"], ["queue_empty"], ["enq", d], ["enq", e], ["deq"], ["enq", f], ["deq"], ["deq"], ["queue_empty"]]],
      });
    }
    const groups = names.flatMap((n) => G[n]);
    if (seed) {
      const r = C.mulberry32(seed);
      const vals = VALUES[o.type];
      const ops = [["new"]];
      let n = 0;
      for (let i = 0; i < 60; i++) {
        const x = r();
        if (x < 0.4) { ops.push(["enq", vals[Math.floor(r() * vals.length)]]); n++; }
        else if (x < 0.65 && n) { ops.push(["deq"]); n--; }
        else if (x < 0.8 && n) ops.push(["queue_peek"]);
        else if (x < 0.9) ops.push(["queue_size"]);
        else if (x < 0.95) ops.push(["queue_empty"]);
        else ops.push(["is_queue"]);
      }
      groups.push(ops);
    }
    return groups;
  }

  // the checker's own helpers, built on the user's types
  function refFns(o, names) {
    if (!onStacks(o)) {
      return names.includes("queue") ? [{ name: "is_segment", steps: STK.isSegment() }, { name: "enq", steps: list.enq() }, { name: "show", steps: showList() }] : [];
    }
    if (!names.includes("stack")) return [];
    const fns = [{ name: "is_segment", steps: STK.isSegment() }, { name: "is_stack", steps: STK.isStack() }, { name: "push", steps: STK.push() },
      { name: "show_items", steps: showItems() }, { name: "show_stack", steps: showStack() }];
    if (names.includes("queue")) fns.push({ name: "show", steps: showTwo() });
    return fns;
  }

  function problem(o) {
    const L = o.lang;
    const sig = C.headers(DEF, o);
    const st = onStacks(o);
    const what = {
      is_segment: "true if following next from start reaches end after exactly n nodes",
      ...(st ? {
        is_stack: "true when [top, null) holds exactly size nodes",
        stack_empty: "true if the stack has no elements",
        push: "put x on top, O(1)", pop: "remove and return the top element, O(1)", peek: "the top element, O(1)",
      } : {}),
      is_queue: st ? "both stacks are valid" : "true when [front, back) holds exactly size nodes and back is the dummy",
      queue_empty: "true if the queue has no elements",
      enq: "add x at the back, O(1)",
      ...(st ? { refill: "if the outbox is empty, move the whole inbox onto it" } : {}),
      deq: `remove and return the front element (requires a non-empty queue), ${st ? "amortized " : ""}O(1)`,
      queue_peek: `the front element, without removing it (requires a non-empty queue), ${st ? "amortized " : ""}O(1)`,
      queue_size: "the number of elements, O(1)",
    };
    const make = { py: (n) => `\`${n}()\``, cpp: (n) => `\`${n} ${n[0]};\``, java: (n) => `\`new ${n}()\``, rs: (n) => `\`${n}::new()\`` }[L];
    return [
      `Build a **${describe(o)}** from a blank file, with no hints: everything the tutorial walked you through.`, "",
      `**Node.** Fields \`data\` and \`next\`; ${L === "rs" ? "`Node::new(data)` returns a raw pointer to a new heap node" : "a constructor taking the element"}, with a null \`next\`.`, "",
      ...(st ? [`**Stack.** Fields \`top\` and \`size\`; ${make("Stack")} makes an empty stack.`, "",
        `**Queue.** Fields \`inbox\` and \`outbox\`, two stacks; ${make("Queue")} makes an empty queue.`, ""]
        : [`**Queue.** Fields \`front\`, \`back\` (a dummy node) and \`size\`; ${make("Queue")} makes an empty queue: one dummy node, both front and back.`, ""]),
      "**Functions** (exact names and signatures):", "",
      ...Object.keys(what).map((n) => `* \`${sig(n)}\`: ${what[n]}`), "",
      `Check \`${fname("is_queue", L)}(Q)\` at the start and end of every queue operation. ${L === "java" ? `Put the functions inside ${st ? "`class Stack` and `class Queue`" : "`class Queue`"} as static methods. ` : ""}` +
        "You don't need a `main`: the checker adds its own (yours, if any, is ignored) and runs every test from the tutorial plus a random sequence of operations.",
    ].join("\n");
  }

  const DEF = {
    id: "queues", num: "13.03", title: "Queues", heading: "Build a queue, step by step", diagramLabel: "Your queue",
    OPTIONS, DEFAULTS, FNS, STRUCTS, describe, complexity, diagram, steps, handlers, testGroups, refFns, problem,
  };
  const tutorial = C.make(DEF);
  if (typeof module !== "undefined" && module.exports) module.exports = tutorial;
})(typeof window !== "undefined" ? window : globalThis);

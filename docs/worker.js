// Runs Pyodide off the main thread (a module worker). The page kills and restarts
// this worker when user code exceeds its time limit (e.g. an infinite loop).
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

let pyodide = null;
let api = null;
let booting = null; // one shared init, even if several "init" messages arrive
const loaded = new Set();

async function init() {
  pyodide = await loadPyodide();
  const build = await (await fetch("build.json", { cache: "no-cache" })).json();
  const zip = await (await fetch(`generator.zip?v=${build.hash}`)).arrayBuffer();
  pyodide.unpackArchive(zip, "zip");
  api = pyodide.pyimport("generator.api");
}

async function ensure(packages) {
  const missing = packages.filter((p) => !loaded.has(p));
  if (missing.length) {
    await pyodide.loadPackage(missing);
    missing.forEach((p) => loaded.add(p));
  }
}

const commands = {
  async init({ packages }) {
    await (booting ||= init());
    await ensure(packages || []);
    return JSON.parse(api.info());
  },
  generate: ({ section, module, n, seed, tier }) => JSON.parse(api.generate(section, module, n, seed, tier ?? null)),
  run: ({ problem, code }) => JSON.parse(api.run(JSON.stringify(problem), code)),
  grade: ({ problem, code }) => JSON.parse(api.grade(JSON.stringify(problem), code)),
  // grade every template's own solution inside Pyodide (used by scripts/e2e_check.py)
  selftest: ({ section, seeds }) => JSON.parse(api.self_test(section, seeds ?? 2)),
};

self.onmessage = async ({ data: { id, cmd, args } }) => {
  try {
    self.postMessage({ id, ok: true, data: await commands[cmd](args || {}) });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};

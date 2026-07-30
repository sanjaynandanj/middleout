const worker = new Worker("worker.js");
const pending = new Map();
let nextId = 1;

worker.onmessage = (e) => {
  const { id } = e.data;
  const cb = pending.get(id);
  if (cb) {
    pending.delete(id);
    cb(e.data);
  }
};

function callWorker(action, bytes, ai) {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    const copy = bytes.slice();
    worker.postMessage({ id, action, data: copy.buffer, ai }, [copy.buffer]);
  });
}

async function gzipBaseline(bytes) {
  const t0 = performance.now();
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  const size = (await new Response(stream).arrayBuffer()).byteLength;
  return { size, ms: performance.now() - t0 };
}

function weissman(r, tMs, rRef, tRefMs) {
  const t = Math.max(tMs, 1.05);
  const tr = Math.max(tRefMs, 1.05);
  return (r / rRef) * (Math.log(tr) / Math.log(t));
}

const fmt = (n) => n.toLocaleString("en-US");
const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const status = document.getElementById("status");
const results = document.getElementById("results");
const restable = document.getElementById("restable");
const wblock = document.getElementById("wblock");
const wscore = document.getElementById("wscore");
const download = document.getElementById("download");
let outputBlob = null;
let outputName = "";
let busy = false;

function setStatus(msg, cls = "") {
  status.textContent = msg;
  status.className = cls;
}

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
  fileInput.value = "";
});
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); })
);
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
download.addEventListener("click", () => {
  const url = URL.createObjectURL(outputBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = outputName;
  a.click();
  URL.revokeObjectURL(url);
});

async function handleFile(file) {
  if (busy) return;
  busy = true;
  results.classList.add("hidden");
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (file.name.endsWith(".mo")) {
      await doDecompress(file, bytes);
    } else {
      await doCompress(file, bytes);
    }
  } catch (err) {
    setStatus(String(err.message || err), "error");
  } finally {
    busy = false;
  }
}

async function doCompress(file, bytes) {
  const ai = document.querySelector('input[name="engine"]:checked').value === "ai";
  if (ai && bytes.length > 4 * 1024 * 1024) {
    setStatus("the Box caps out at 4 MB in the browser — use the CLI for bigger files", "error");
    return;
  }
  setStatus(ai ? "the Box is thinking (one bit at a time)…" : "compressing middle-out…", "blink");

  const [res, gz] = await Promise.all([
    callWorker("compress", bytes, ai),
    gzipBaseline(bytes),
  ]);
  if (!res.ok) { setStatus(res.error, "error"); return; }

  const out = new Uint8Array(res.out);
  const ratio = bytes.length / out.length;
  const gzRatio = bytes.length / gz.size;
  const w = weissman(ratio, res.ms, gzRatio, gz.ms);

  restable.innerHTML = `
    <table>
      <tr><th>codec</th><th>size</th><th>ratio</th><th>time</th></tr>
      <tr><td>original</td><td>${fmt(bytes.length)} B</td><td>—</td><td>—</td></tr>
      <tr><td>browser gzip</td><td>${fmt(gz.size)} B</td><td>${gzRatio.toFixed(3)}</td><td>${gz.ms.toFixed(0)} ms</td></tr>
      <tr><td>middleout${ai ? "-ai" : "-lz"}</td>
        <td class="${out.length < gz.size ? "win" : ""}">${fmt(out.length)} B</td>
        <td class="${ratio > gzRatio ? "win" : ""}">${ratio.toFixed(3)}</td>
        <td>${res.ms.toFixed(0)} ms</td></tr>
    </table>`;
  wscore.textContent = w.toFixed(3);
  wblock.classList.remove("hidden");

  outputBlob = new Blob([out], { type: "application/octet-stream" });
  outputName = file.name + ".mo";
  download.textContent = `download ${outputName}`;
  results.classList.remove("hidden");
  setStatus(ratio > gzRatio ? "smaller than gzip. Richard would be proud." : "done.");
}

async function doDecompress(file, bytes) {
  setStatus("decompressing…", "blink");
  const res = await callWorker("decompress", bytes, false);
  if (!res.ok) { setStatus(res.error, "error"); return; }

  const out = new Uint8Array(res.out);
  restable.innerHTML = `
    <table>
      <tr><th>&nbsp;</th><th>size</th><th>time</th></tr>
      <tr><td>${file.name}</td><td>${fmt(bytes.length)} B</td><td>—</td></tr>
      <tr><td>restored</td><td class="win">${fmt(out.length)} B</td><td>${res.ms.toFixed(0)} ms</td></tr>
    </table>`;
  wblock.classList.add("hidden");

  outputBlob = new Blob([out], { type: "application/octet-stream" });
  outputName = file.name.replace(/\.mo$/, "") || "restored.bin";
  download.textContent = `download ${outputName}`;
  results.classList.remove("hidden");
  setStatus("restored, bit for bit.");
}

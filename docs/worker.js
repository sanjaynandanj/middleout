let wasm = null;

const ready = (async () => {
  const res = await fetch("middleout.wasm");
  let instance;
  try {
    ({ instance } = await WebAssembly.instantiateStreaming(res));
  } catch {
    const bytes = await (await fetch("middleout.wasm")).arrayBuffer();
    ({ instance } = await WebAssembly.instantiate(bytes));
  }
  wasm = instance.exports;
})();

function copyIn(bytes) {
  const ptr = wasm.mo_alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}

function readOut(ptr, len) {
  return new Uint8Array(wasm.memory.buffer, ptr, len).slice();
}

self.onmessage = async (e) => {
  const { id, action, data, ai } = e.data;
  await ready;
  const input = new Uint8Array(data);
  try {
    const inPtr = copyIn(input);
    const lenPtr = wasm.mo_alloc(4);
    const t0 = performance.now();
    const outPtr =
      action === "compress"
        ? wasm.mo_compress(inPtr, input.length, ai ? 1 : 0, lenPtr)
        : wasm.mo_decompress(inPtr, input.length, lenPtr);
    const ms = performance.now() - t0;
    // memory may have grown during the call; re-read views afterwards
    const outLen = new DataView(wasm.memory.buffer).getUint32(lenPtr, true);
    if (action === "decompress" && outPtr === 0) {
      throw new Error("not a valid .mo file");
    }
    const out = readOut(outPtr, outLen);
    wasm.mo_free(inPtr, input.length);
    wasm.mo_free(lenPtr, 4);
    if (outPtr !== 0) wasm.mo_free(outPtr, outLen);
    self.postMessage({ id, ok: true, out: out.buffer, ms }, [out.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err.message || err) });
  }
};

# middleout

Middle-out compression, Pied Piper style. Optimal tip-to-tip efficiency.

A lossless compression CLI in pure Rust with two engines:

- **`middleout-lz`** (default) — a genuinely middle-out LZ compressor. Fast.
- **`middleout-ai`** (`--ai`) — "the Box": an online-learning, context-mixing
  compressor that beats `zstd -19` on compression ratio for text, JSON, HTML,
  and executables. Slow, dense, worth it.

Plus a `bench` command that scores everything with the actual **Weissman
score** formula from the show.

> This is a working compressor with verified roundtrips and tests, built for
> fun and for learning how modern compression works. The file format is not
> stable between versions — don't trust it as your only copy of anything.

## Quick start

Requires a [Rust toolchain](https://rustup.rs) (any recent stable).

```sh
git clone https://github.com/sanjaynandanj/middleout
cd middleout
cargo build --release
./target/release/middleout --help
```

Or install it onto your PATH:

```sh
cargo install --path .
```

## Usage

### Compress / decompress

```sh
# Fast LZ engine (default) -> writes notes.txt.mo
middleout compress notes.txt

# AI engine: much better ratio, ~100x slower
middleout compress notes.txt --ai

# Restore (engine is auto-detected from the file header)
middleout decompress notes.txt.mo

# Explicit output paths
middleout compress data.json -o data.mo
middleout decompress data.mo -o restored.json
```

Example output:

```
notes.txt -> notes.txt.mo | 164168 -> 10039 bytes | ratio 16.353 | 552.8 ms | engine: ai context-mixer
```

Decompression auto-detects which engine produced the file, so `decompress`
never needs a flag.

### Benchmark

```sh
middleout bench file1.json file2.html big.log
```

Compresses each file with gzip -6, zstd -3, zstd -19, and both middleout
engines, verifies a full roundtrip for every codec, and prints ratio, timing,
and Weissman score:

```
=== recipes.json (164168 bytes) ===
codec              compressed    ratio    comp ms  decomp ms   Weissman
gzip -6 (ref)           13738   11.950        5.1        0.5      1.000
zstd -3                 19558    8.394        3.0        1.6      1.035
zstd -19                11036   14.876      336.7        0.5      0.347
middleout-lz            17244    9.520       10.3        0.4      0.553
middleout-ai            10039   16.353      552.8      545.0      0.351
```

## Which engine should I use?

| | `middleout-lz` (default) | `middleout-ai` (`--ai`) |
|---|---|---|
| Speed | ~10-20 MB/s | ~0.2-0.3 MB/s |
| Ratio | below gzip | **beats zstd -19** on compressible data |
| Memory | ~2x input size | ~80 MB + input size |
| Use when | you want fast and fun | you want the smallest file and can wait |

## What kinds of files does it work on?

Measured on real files (ratios, higher is better):

| file type | gzip -6 | zstd -19 | middleout-ai |
|---|---:|---:|---:|
| JSON export | 11.95 | 14.88 | **16.35** |
| HTML page | 5.27 | 5.82 | **6.74** |
| Source code (1.6 MB) | 5.50 | 6.90 | **8.02** |
| Windows EXE | 2.46 | 2.82 | **2.95** |
| PDF | 1.30 | 1.31 | 1.31 |
| JPEG | 1.005 | 1.005 | 1.02 |

Rule of thumb: if a human or a program *wrote* the bytes (text, code, logs,
JSON, CSV, XML, binaries), the AI engine wins. If a compressor already
touched the bytes (JPEG, MP4, ZIP, most PDFs), nothing helps — the entropy
is already extracted. Incompressible input falls back to raw storage, so a
`.mo` file is never more than 13 bytes larger than the original.

## The Weissman score

```
W = alpha * (r / r_ref) * (log T_ref / log T)
```

where `r` is compression ratio and `T` is compression time. The bench uses
`alpha = 1`, times in milliseconds, and gzip -6 as the reference codec
(so gzip always scores exactly 1.000).

Fair warning: the score heavily rewards speed, so `zstd -3` tends to win
Weissman while `middleout-ai` wins ratio. The Box was never about speed.

## How it works

### The LZ engine (mode 1)

Actually middle-out: the input is split at the midpoint and both halves are
compressed working outward from the middle — the first half is processed in
reverse. Each half is LZSS with a 64 KB window, 4-byte hash-chain match
finding, lazy matching (gzip's trick: defer a match one byte if the next
position offers a longer one), and flag-byte token groups. Matches are 4-259
bytes at distances up to 65,535.

### The AI engine (mode 2)

A miniature PAQ-style context mixer that predicts the input **one bit at a
time** and codes each bit with a binary arithmetic coder:

1. **Seven predictors** run in parallel: context models of order 0-4
   (hashed into 4M-entry tables with count-based adaptive learning rates),
   a word model (hash of the current alphanumeric run), and a **match model**
   that finds the most recent occurrence of the current 4-byte context and
   predicts whatever byte followed it — confidence scales with match length.
2. **A tiny neural network mixes them**: predictions are converted to the
   logistic domain and combined with a weighted sum; 256 weight sets
   (selected by the partial-byte context) are trained by online gradient
   descent as the data streams through.
3. **Secondary estimation (SSE/APM)** refines the mixed probability against
   actually-observed outcomes, interpolated across 33 bins per byte context.
4. **Arithmetic coding** turns each probability + bit into a fraction of a
   bit of output — a bit predicted at 99% costs ~0.014 bits.

Decompression runs the identical model in lockstep: it makes the same
predictions, decodes each bit, and stays bit-for-bit synchronized. This is
the same architecture family as PAQ/lpaq and modern LLM+arithmetic-coding
compressors ("language modeling is compression") — minus a few billion
parameters.

### File format

```
"MOUT" | mode: u8 | original_len: u64 LE | payload
```

| mode | payload |
|---|---|
| 0 | raw bytes (incompressible fallback) |
| 1 | middle-out LZ: `len_first_half: u64 LE` + two LZSS streams |
| 2 | AI engine arithmetic-coded bitstream |

## Development

```sh
cargo test --release   # roundtrip tests: empty, tiny, repetitive, random, text
cargo build --release
```

Project layout:

```
src/main.rs    CLI (clap)
src/codec.rs   container format, engine dispatch, raw fallback, tests
src/lz.rs      middle-out LZSS engine
src/model.rs   AI engine: context models, match model, mixer, SSE
src/arith.rs   binary arithmetic coder (lpaq-style carry-free range coder)
src/bench.rs   gzip/zstd comparison + Weissman scoring
```

gzip and zstd only appear in `bench.rs` as comparison baselines — the two
middleout engines are dependency-free, hand-rolled compression.

## License

MIT. See [LICENSE](LICENSE).

# middleout

Middle-out compression, Pied Piper style. Optimal tip-to-tip efficiency.

Two engines, one CLI:

- **`middleout-lz`** (default) — genuine middle-out LZ: the input is split at
  the midpoint and both halves are compressed working outward from the middle
  (the first half is processed in reverse). LZSS, 64K window, hash chains,
  lazy matching. Fast.
- **`middleout-ai`** (`--ai`) — "the Box": an online-learning context-mixing
  compressor. Six context models (orders 0-4 + a word model) plus a match
  model each predict the next bit; a tiny neural mixer trained by online
  gradient descent blends them, an SSE stage refines the estimate, and a
  binary arithmetic coder writes the bits. The same principle behind PAQ and
  modern LLM+arithmetic-coding compressors, minus a few billion parameters.
  Slow, dense.

## Usage

```
middleout compress <file>          # -> file.mo (LZ engine)
middleout compress <file> --ai     # -> file.mo (AI engine)
middleout decompress <file.mo>
middleout bench <files...>         # vs gzip/zstd, with Weissman scores
```

## Weissman score

The bench computes the actual Weissman score from the show:

```
W = alpha * (r / r_ref) * (log T_ref / log T)
```

with alpha = 1, times in milliseconds, and gzip -6 as the reference codec.

## Numbers (1.6 MB of non-repetitive source text)

| codec        | ratio | comp ms |
|--------------|------:|--------:|
| gzip -6      | 5.50  | 82      |
| zstd -3      | 5.29  | 25      |
| zstd -19     | 6.90  | 2785    |
| middleout-lz | 4.49  | 163     |
| middleout-ai | **8.02** | 10161 |

The AI engine beats zstd -19 on ratio by ~14% and gzip by ~46%. It is also
roughly 100x slower, which is canon: the Box was never about speed.

Every bench run verifies a full roundtrip for every codec.

## Format

`MOUT | mode u8 | original_len u64 LE | payload`, modes: 0 = stored raw
(incompressible fallback), 1 = middle-out LZ, 2 = AI context mixer.

//! "AI mode": an online-learning neural context-mixing compressor.
//!
//! Eleven predictors feed a two-layer neural mixer network:
//!   - nine context models (orders 0-6, a word model, and a sparse/gap
//!     model that skips the most recent byte),
//!   - a match model that follows the last occurrence of the current
//!     context through history,
//!   - an online-trained LSTM that predicts whole bytes and is folded
//!     down to per-bit probabilities (see lstm.rs).
//!
//! Layer 1 is three mixer banks selected by different contexts (partial
//! byte, previous byte, match state); layer 2 mixes their outputs. All
//! weights train by gradient descent as the data streams through. The
//! final probability is refined by a two-stage SSE/APM chain before
//! hitting the arithmetic coder — the same architecture family as PAQ8
//! and cmix, minus a few billion parameters.

use crate::arith::{Decoder, Encoder};
use crate::lstm::Lstm;

const TABLE_BITS: usize = 20;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_SIZE - 1;
const NMODELS: usize = 9;
const NINPUTS: usize = NMODELS + 2; // context models + match model + LSTM
const NBANKS: usize = 3;
const MATCH_BITS: usize = 22;
const MATCH_MASK: usize = (1 << MATCH_BITS) - 1;
const BANK_LR: f32 = 0.02;
const L2_LR: f32 = 0.008;

/// Logistic squash: maps a stretch value in [-2047, 2047] to a 12-bit probability.
fn squash(d: i32) -> i32 {
    if d >= 2047 {
        return 4095;
    }
    if d <= -2047 {
        return 0;
    }
    const T: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
        3348, 3607, 3785, 3901, 3968, 4010, 4032, 4045, 4050, 4052, 4053, 4054, 4054, 4054,
    ];
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (T[i] * (128 - w) + T[i + 1] * w + 64) >> 7
}

fn build_stretch() -> Vec<i16> {
    let mut table = vec![0i16; 4096];
    let mut pi = 0usize;
    for x in -2047..=2047 {
        let v = squash(x) as usize;
        for entry in table.iter_mut().take(v + 1).skip(pi) {
            *entry = x as i16;
        }
        pi = v + 1;
    }
    for entry in table.iter_mut().skip(pi) {
        *entry = 2047;
    }
    table
}

fn hash(a: u32, b: u32) -> usize {
    let h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77);
    (h ^ (h >> 15)) as usize & TABLE_MASK
}

fn hash64(a: u64, b: u32) -> usize {
    let h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (b as u64).wrapping_mul(0x85EB_CA77);
    ((h ^ (h >> 29)) as usize) & TABLE_MASK
}

/// Adaptive bit model: a 12-bit probability packed with a 4-bit confidence
/// count in one u16, so each model touches a single cache line per bit.
/// The count slows the learning rate as evidence accumulates; the divide is
/// replaced by a reciprocal multiply.
struct CtxTable {
    t: Vec<u16>,
}

/// RECIP[c] = 65536 / (c + 2)
const RECIP: [i32; 16] = [
    32768, 21845, 16384, 13107, 10922, 9362, 8192, 7281, 6553, 5957, 5461, 5041, 4681, 4369,
    4096, 3855,
];

impl CtxTable {
    fn new(size: usize) -> Self {
        CtxTable { t: vec![2048 << 4; size] }
    }

    #[inline]
    fn prob(&self, idx: usize) -> usize {
        (self.t[idx] >> 4) as usize
    }

    #[inline]
    fn update(&mut self, idx: usize, bit: u32) {
        let v = self.t[idx];
        let p = (v >> 4) as i32;
        let c = (v & 15) as i32;
        let target = (bit << 12) as i32;
        let np = p + (((target - p) * RECIP[c as usize]) >> 16);
        let nc = (c + 1).min(15);
        self.t[idx] = ((np as u16) << 4) | nc as u16;
    }
}

/// One SSE/APM stage: a learned refinement curve per context, interpolated
/// across 33 bins in the stretch domain.
struct Apm {
    t: Vec<u16>,
    cell: usize,
}

impl Apm {
    fn new(nctx: usize) -> Self {
        let mut t = Vec::with_capacity(nctx * 33);
        for _ in 0..nctx {
            for i in 0..33 {
                t.push((squash((i - 16) * 128) << 4) as u16);
            }
        }
        Apm { t, cell: 0 }
    }

    #[inline]
    fn refine(&mut self, p: i32, ctx: usize, stretch: &[i16]) -> i32 {
        let s = stretch[p as usize] as i32 + 2048;
        let bin = (s >> 7) as usize;
        let w = s & 127;
        let base = ctx * 33 + bin;
        self.cell = base + if w >= 64 { 1 } else { 0 };
        (self.t[base] as i32 * (128 - w) + self.t[base + 1] as i32 * w) >> 11
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let cell = &mut self.t[self.cell];
        let v = *cell as i32;
        *cell = (v + (((bit << 16) as i32 - v) >> 5)) as u16;
    }
}

struct Mixer {
    tables: Vec<CtxTable>,
    banks: [Vec<[f32; NINPUTS]>; NBANKS], // layer-1 mixers, each bank selected by its own context
    l2: Vec<[f32; NBANKS]>,               // layer-2 mixer, selected by partial byte
    stretch: Vec<i16>,
    apm1: Apm, // SSE stage 1: partial-byte context
    apm2: Apm, // SSE stage 2: order-2 hashed context
    lstm: Lstm,
    c0: u32, // partial byte with leading 1 sentinel
    c4: u32, // last 4 whole bytes
    c8: u64, // last 8 whole bytes
    word: u32,
    nbits: u32, // bits consumed of the current byte
    // Match model: predicts the next bit by following the most recent
    // occurrence of the current 4-byte context through history.
    buf: Vec<u8>,
    match_table: Vec<u32>,
    match_ptr: usize,
    match_len: u32,
    match_valid: bool,
    expected_byte: u32,
    match_bit: u32,
    base: [usize; NMODELS], // per-byte context hashes, refreshed once per byte
    idx: [usize; NMODELS],
    st: [i32; NINPUTS],
    bank_ctx: [usize; NBANKS],
    bank_st: [f32; NBANKS], // layer-1 outputs in the stretch domain
    bank_p: [i32; NBANKS],  // layer-1 outputs squashed, for their own training
    pm: i32,                // layer-2 output (pre-SSE), used for weight training
    p: i32,                 // final probability used for coding
}

impl Mixer {
    fn new() -> Self {
        let mut tables = Vec::with_capacity(NMODELS);
        tables.push(CtxTable::new(256)); // order 0
        for _ in 1..NMODELS {
            tables.push(CtxTable::new(TABLE_SIZE));
        }
        let mut mixer = Mixer {
            tables,
            banks: [
                vec![[0.15; NINPUTS]; 256], // by partial byte
                vec![[0.15; NINPUTS]; 256], // by previous byte
                vec![[0.15; NINPUTS]; 16],  // by match state
            ],
            l2: vec![[0.34; NBANKS]; 256],
            stretch: build_stretch(),
            apm1: Apm::new(256),
            apm2: Apm::new(4096),
            lstm: Lstm::new(),
            c0: 1,
            c4: 0,
            c8: 0,
            word: 0,
            nbits: 0,
            buf: Vec::new(),
            match_table: vec![u32::MAX; 1 << MATCH_BITS],
            match_ptr: 0,
            match_len: 0,
            match_valid: false,
            expected_byte: 0,
            match_bit: 0,
            base: [0; NMODELS],
            idx: [0; NMODELS],
            st: [0; NINPUTS],
            bank_ctx: [0; NBANKS],
            bank_st: [0.0; NBANKS],
            bank_p: [2048; NBANKS],
            pm: 2048,
            p: 2048,
        };
        mixer.refresh_bases();
        mixer
    }

    fn refresh_bases(&mut self) {
        self.base[1] = hash(self.c4 & 0xFF, 0x01);
        self.base[2] = hash(self.c4 & 0xFFFF, 0x02);
        self.base[3] = hash(self.c4 & 0xFF_FFFF, 0x03);
        self.base[4] = hash(self.c4, 0x04);
        self.base[5] = hash64(self.c8 & 0xFF_FFFF_FFFF, 0x05);
        self.base[6] = hash64(self.c8 & 0xFFFF_FFFF_FFFF, 0x06);
        self.base[7] = hash(self.word, 0x07);
        self.base[8] = hash(((self.c8 >> 8) & 0xFFFF) as u32, 0x08); // gap: skips newest byte
    }

    fn predict(&mut self) -> u16 {
        let c0mix = self.c0 as usize * 0x02ED;
        self.idx[0] = self.c0 as usize;
        for i in 1..NMODELS {
            self.idx[i] = (self.base[i] + c0mix) & TABLE_MASK;
        }
        for i in 0..NMODELS {
            let pr = self.tables[i].prob(self.idx[i]);
            self.st[i] = self.stretch[pr] as i32;
        }

        self.st[NMODELS] = if self.match_valid {
            self.match_bit = (self.expected_byte >> (7 - self.nbits)) & 1;
            let confidence = (256 + 256 * self.match_len.min(28) as i32).min(2047);
            if self.match_bit == 1 { confidence } else { -confidence }
        } else {
            0
        };
        self.st[NMODELS + 1] = self.stretch[self.lstm.bit_p(self.c0 as usize) as usize] as i32;

        // Layer 1: three mixer banks, each with weights selected by a
        // different view of the current state.
        self.bank_ctx[0] = (self.c0 & 0xFF) as usize;
        self.bank_ctx[1] = (self.c4 & 0xFF) as usize;
        self.bank_ctx[2] =
            if self.match_valid { 8 + self.match_len.min(7) as usize } else { self.nbits.min(7) as usize };
        for k in 0..NBANKS {
            let w = &self.banks[k][self.bank_ctx[k]];
            let mut dot = 0.0f32;
            for i in 0..NINPUTS {
                dot += w[i] * self.st[i] as f32;
            }
            let dot = dot.clamp(-2047.0, 2047.0);
            self.bank_st[k] = dot;
            self.bank_p[k] = squash(dot as i32);
        }

        // Layer 2: mix the bank outputs in the stretch domain.
        let w2 = &self.l2[(self.c0 & 0xFF) as usize];
        let mut dot2 = 0.0f32;
        for k in 0..NBANKS {
            dot2 += w2[k] * self.bank_st[k];
        }
        self.pm = squash(dot2.clamp(-2047.0, 2047.0) as i32).clamp(1, 4094);

        // SSE chain: two learned refinement stages.
        let a1 = self.apm1.refine(self.pm, (self.c0 & 0xFF) as usize, &self.stretch);
        let p1 = ((self.pm + 3 * a1) / 4).clamp(1, 4094);
        let ctx2 = ((self.c4 & 0xFFFF).wrapping_mul(0x9E37_79B1) >> 20) as usize & 0xFFF;
        let a2 = self.apm2.refine(p1, ctx2, &self.stretch);
        self.p = ((p1 + 3 * a2) / 4).clamp(1, 4094);
        self.p as u16
    }

    fn update(&mut self, bit: u32) {
        self.apm1.update(bit);
        self.apm2.update(bit);

        let target = (bit << 12) as i32;
        let err2 = (target - self.pm) as f32 / 4096.0;
        let w2 = &mut self.l2[(self.c0 & 0xFF) as usize];
        for k in 0..NBANKS {
            w2[k] += L2_LR * err2 * (self.bank_st[k] / 2048.0);
        }
        for k in 0..NBANKS {
            let err = (target - self.bank_p[k]) as f32 / 4096.0;
            let w = &mut self.banks[k][self.bank_ctx[k]];
            for i in 0..NINPUTS {
                w[i] += BANK_LR * err * (self.st[i] as f32 / 2048.0);
            }
        }
        for i in 0..NMODELS {
            self.tables[i].update(self.idx[i], bit);
        }

        if self.match_valid && bit != self.match_bit {
            self.match_valid = false;
            self.match_len = 0;
        }

        self.c0 = (self.c0 << 1) | bit;
        self.nbits += 1;
        if self.c0 >= 256 {
            let byte = self.c0 - 256;
            self.c4 = (self.c4 << 8) | byte;
            self.c8 = (self.c8 << 8) | byte as u64;
            let b = byte as u8;
            let is_word = b.is_ascii_alphanumeric() || b == b'_';
            self.word = if is_word {
                self.word.wrapping_mul(0x0100_0193) ^ byte
            } else {
                0
            };
            self.c0 = 1;
            self.nbits = 0;
            self.buf.push(b);
            self.lstm.on_byte(b);

            if self.buf.len() >= 4 {
                let mh = (self.c4.wrapping_mul(0x9E37_79B1) >> 10) as usize & MATCH_MASK;
                if self.match_valid {
                    self.match_ptr += 1;
                    self.match_len += 1;
                } else {
                    let cand = self.match_table[mh];
                    if cand != u32::MAX {
                        self.match_ptr = cand as usize;
                        self.match_len = 1;
                    } else {
                        self.match_len = 0;
                    }
                }
                self.match_table[mh] = self.buf.len() as u32;
                self.match_valid = self.match_len > 0 && self.match_ptr < self.buf.len();
                if self.match_valid {
                    self.expected_byte = self.buf[self.match_ptr] as u32;
                }
            }
            self.refresh_bases();
        }
    }
}

/// Flush-to-zero + denormals-are-zero. The online LSTM inevitably drives
/// weights and probabilities toward the subnormal range, where x86 takes a
/// ~100x microcode penalty per operation. Both compress and decompress set
/// the same mode, so predictions remain bit-for-bit identical.
fn enable_ftz() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_getcsr, _mm_setcsr};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        #[allow(deprecated)]
        unsafe {
            _mm_setcsr(_mm_getcsr() | 0x8040);
        }
    }
}

pub fn compress(data: &[u8]) -> Vec<u8> {
    enable_ftz();
    let mut mixer = Mixer::new();
    let mut enc = Encoder::new();
    for &byte in data {
        for k in (0..8).rev() {
            let bit = ((byte >> k) & 1) as u32;
            let p = mixer.predict();
            enc.encode(bit, p);
            mixer.update(bit);
        }
    }
    enc.finish()
}

pub fn decompress(payload: &[u8], orig_len: usize) -> Vec<u8> {
    enable_ftz();
    let mut mixer = Mixer::new();
    let mut dec = Decoder::new(payload);
    let mut out = Vec::with_capacity(orig_len);
    for _ in 0..orig_len {
        let mut byte = 0u32;
        for _ in 0..8 {
            let p = mixer.predict();
            let bit = dec.decode(p);
            mixer.update(bit);
            byte = (byte << 1) | bit;
        }
        out.push(byte as u8);
    }
    out
}

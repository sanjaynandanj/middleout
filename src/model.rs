//! "AI mode": an online-learning context-mixing compressor.
//!
//! Six context models (orders 0-4 plus a word model) each predict the next
//! bit. Their predictions are mixed in the logistic domain by a tiny neural
//! layer (256 weight sets selected by the partial byte) trained online by
//! gradient descent as the data streams through — the same principle behind
//! PAQ and modern LLM+arithmetic-coding compressors, minus a few billion
//! parameters.

use crate::arith::{Decoder, Encoder};

const TABLE_BITS: usize = 20;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: usize = TABLE_SIZE - 1;
const NMODELS: usize = 6;
const NINPUTS: usize = NMODELS + 1; // context models + match model
const MATCH_BITS: usize = 22;
const MATCH_MASK: usize = (1 << MATCH_BITS) - 1;

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

struct Mixer {
    tables: Vec<CtxTable>,
    weights: Vec<[f32; NINPUTS]>, // one weight set per partial-byte context
    stretch: Vec<i16>,
    apm: Vec<u16>, // secondary estimation: 256 byte-contexts x 33 bins, 16-bit probs
    apm_cell: usize,
    c0: u32, // partial byte with leading 1 sentinel
    c4: u32, // last 4 whole bytes
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
    pm: i32, // mixer output (pre-APM), used for weight training
    p: i32,  // final probability used for coding
}

impl Mixer {
    fn new() -> Self {
        let mut tables = Vec::with_capacity(NMODELS);
        tables.push(CtxTable::new(256)); // order 0
        for _ in 1..NMODELS {
            tables.push(CtxTable::new(TABLE_SIZE));
        }
        let mut apm = Vec::with_capacity(256 * 33);
        for _ in 0..256 {
            for i in 0..33 {
                apm.push((squash((i - 16) * 128) << 4) as u16);
            }
        }
        let mut mixer = Mixer {
            tables,
            weights: vec![[0.35; NINPUTS]; 256],
            stretch: build_stretch(),
            apm,
            apm_cell: 0,
            c0: 1,
            c4: 0,
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
        self.base[5] = hash(self.word, 0x05);
    }

    fn predict(&mut self) -> u16 {
        let c0mix = self.c0 as usize * 0x02ED;
        self.idx[0] = self.c0 as usize;
        for i in 1..NMODELS {
            self.idx[i] = (self.base[i] + c0mix) & TABLE_MASK;
        }

        let w = &self.weights[self.c0 as usize & 0xFF];
        let mut dot = 0.0f32;
        for i in 0..NMODELS {
            let pr = self.tables[i].prob(self.idx[i]);
            self.st[i] = self.stretch[pr] as i32;
            dot += w[i] * self.st[i] as f32;
        }

        self.st[NMODELS] = if self.match_valid {
            self.match_bit = (self.expected_byte >> (7 - self.nbits)) & 1;
            let confidence = (256 + 256 * self.match_len.min(28) as i32).min(2047);
            if self.match_bit == 1 { confidence } else { -confidence }
        } else {
            0
        };
        dot += w[NMODELS] * self.st[NMODELS] as f32;

        self.pm = squash(dot.clamp(-2047.0, 2047.0) as i32).clamp(1, 4094);

        // Secondary estimation: refine the mixer output using a learned
        // curve per previous-byte context, interpolated across 33 bins.
        let ctx = (self.c4 & 0xFF) as usize;
        let s = self.stretch[self.pm as usize] as i32 + 2048;
        let bin = (s >> 7) as usize;
        let w = s & 127;
        let base = ctx * 33 + bin;
        let pa = (self.apm[base] as i32 * (128 - w) + self.apm[base + 1] as i32 * w) >> 11;
        self.apm_cell = base + if w >= 64 { 1 } else { 0 };

        self.p = ((self.pm + 3 * pa) / 4).clamp(1, 4094);
        self.p as u16
    }

    fn update(&mut self, bit: u32) {
        let cell = &mut self.apm[self.apm_cell];
        let v = *cell as i32;
        *cell = (v + (((bit << 16) as i32 - v) >> 5)) as u16;

        let err = ((bit << 12) as i32 - self.pm) as f32 / 4096.0;
        let w = &mut self.weights[self.c0 as usize & 0xFF];
        for i in 0..NINPUTS {
            w[i] += 0.02 * err * (self.st[i] as f32 / 2048.0);
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

pub fn compress(data: &[u8]) -> Vec<u8> {
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

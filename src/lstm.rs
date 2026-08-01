//! Online LSTM byte predictor — a real recurrent neural network trained by
//! backpropagation *while the data streams through*, the same trick cmix and
//! NNCP use. No pretraining, no stored weights: encoder and decoder both
//! start from identical initial weights and perform identical updates, so
//! their predictions stay bit-for-bit in lockstep.
//!
//! Once per byte: one forward pass (embedding -> LSTM cell -> 256-way
//! softmax) and one truncated-backprop training step on the byte that
//! actually arrived. The softmax over next-byte values is folded into a
//! binary tree of prefix sums so each of the 8 bit predictions is a single
//! divide.

const H: usize = 32; // hidden units
const G4: usize = 4 * H; // gate pre-activations: input | forget | output | cell
const LR: f32 = 0.03;

/// Rational tanh approximation (Pade 3/2), exact at 0, clamped past |x| = 3.
#[inline]
fn ftanh(x: f32) -> f32 {
    if x >= 3.0 {
        return 1.0;
    }
    if x <= -3.0 {
        return -1.0;
    }
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    0.5 * (ftanh(0.5 * x) + 1.0)
}

/// Fast exp via exponent-bit assembly: 2^z split into integer (IEEE exponent
/// field) and fractional (cubic with exact endpoints) parts. ~0.1% relative
/// error — plenty for a softmax.
#[inline]
fn fexp(x: f32) -> f32 {
    // Floor at 2^-60 (~8.7e-19 after softmax scaling) keeps every
    // probability a normal float — subnormals cost ~100x on x86.
    let z = (x * std::f32::consts::LOG2_E).max(-60.0);
    let zi = z.floor();
    let zf = z - zi;
    let p = 1.0 + zf * (0.695_178_7 + zf * (0.226_169_6 + zf * 0.078_651_7));
    f32::from_bits(((zi as i32 + 127) as u32) << 23) * p
}

pub struct Lstm {
    wx: Vec<f32>, // 256 x G4: per-byte embedding rows feeding all four gates
    wh: Vec<f32>, // G4 x H: recurrent weights
    b: Vec<f32>,  // G4
    wy: Vec<f32>, // 256 x H: output projection
    by: Vec<f32>, // 256
    h: [f32; H],
    c: [f32; H],
    // Activations saved from the last forward pass for the training step.
    h_in: [f32; H],
    c_in: [f32; H],
    gi: [f32; H],
    gf: [f32; H],
    go: [f32; H],
    gg: [f32; H],
    th: [f32; H],
    input: usize,
    p: [f32; 256],
    tree: [f32; 512], // subtree sums of p; node k covers bytes with prefix k
}

impl Lstm {
    pub fn new() -> Self {
        let mut rng = 0x2545_F491u32;
        let mut rand = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            ((rng >> 8) as f32 / 16_777_216.0 - 0.5) * 0.16
        };
        let wx = (0..256 * G4).map(|_| rand()).collect();
        let wh = (0..G4 * H).map(|_| rand()).collect();
        let mut b = vec![0.0; G4];
        b[H..2 * H].fill(1.0); // forget-gate bias: remember by default
        let mut lstm = Lstm {
            wx,
            wh,
            b,
            wy: vec![0.0; 256 * H],
            by: vec![0.0; 256],
            h: [0.0; H],
            c: [0.0; H],
            h_in: [0.0; H],
            c_in: [0.0; H],
            gi: [0.0; H],
            gf: [0.0; H],
            go: [0.0; H],
            gg: [0.0; H],
            th: [0.0; H],
            input: 0,
            p: [0.0; 256],
            tree: [0.0; 512],
        };
        lstm.forward(0);
        lstm
    }

    /// Train on the byte that actually arrived, then advance the recurrence
    /// so `bit_p` predicts the byte after it.
    pub fn on_byte(&mut self, byte: u8) {
        self.train(byte as usize);
        self.forward(byte as usize);
    }

    /// P(next bit == 1) for partial byte `c0` (leading-1 sentinel), 12-bit.
    #[inline]
    pub fn bit_p(&self, c0: usize) -> i32 {
        let den = self.tree[c0];
        if den <= 1e-9 {
            return 2048;
        }
        ((self.tree[2 * c0 + 1] / den * 4096.0) as i32).clamp(1, 4095)
    }

    fn forward(&mut self, input: usize) {
        self.h_in = self.h;
        self.c_in = self.c;
        self.input = input;

        let mut z = [0.0f32; G4];
        let xrow = &self.wx[input * G4..(input + 1) * G4];
        for g in 0..G4 {
            let wr = &self.wh[g * H..(g + 1) * H];
            let mut acc = self.b[g] + xrow[g];
            for j in 0..H {
                acc += wr[j] * self.h_in[j];
            }
            z[g] = acc;
        }
        for k in 0..H {
            self.gi[k] = sigmoid(z[k]);
            self.gf[k] = sigmoid(z[H + k]);
            self.go[k] = sigmoid(z[2 * H + k]);
            self.gg[k] = ftanh(z[3 * H + k]);
            self.c[k] = self.gf[k] * self.c_in[k] + self.gi[k] * self.gg[k];
            self.th[k] = ftanh(self.c[k]);
            self.h[k] = self.go[k] * self.th[k];
        }

        let mut logits = [0.0f32; 256];
        let mut maxl = f32::MIN;
        for j in 0..256 {
            let row = &self.wy[j * H..(j + 1) * H];
            let mut acc = self.by[j];
            for k in 0..H {
                acc += row[k] * self.h[k];
            }
            logits[j] = acc;
            if acc > maxl {
                maxl = acc;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..256 {
            let e = fexp(logits[j] - maxl);
            self.p[j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..256 {
            self.p[j] *= inv;
            self.tree[256 + j] = self.p[j];
        }
        for k in (1..256).rev() {
            self.tree[k] = self.tree[2 * k] + self.tree[2 * k + 1];
        }
    }

    /// One step of truncated backprop (gradients stop at the previous
    /// hidden/cell state — the online-compression compromise).
    fn train(&mut self, target: usize) {
        let mut dh = [0.0f32; H];
        for j in 0..256 {
            let d = self.p[j] - if j == target { 1.0 } else { 0.0 };
            self.by[j] -= LR * d;
            let row = &mut self.wy[j * H..(j + 1) * H];
            for k in 0..H {
                dh[k] += d * row[k];
                row[k] -= LR * d * self.h[k];
            }
        }

        let mut dz = [0.0f32; G4];
        for k in 0..H {
            let d_o = dh[k] * self.th[k];
            let dth = dh[k] * self.go[k];
            let dc = dth * (1.0 - self.th[k] * self.th[k]);
            let di = dc * self.gg[k];
            let dg = dc * self.gi[k];
            let df = dc * self.c_in[k];
            dz[k] = (di * self.gi[k] * (1.0 - self.gi[k])).clamp(-1.0, 1.0);
            dz[H + k] = (df * self.gf[k] * (1.0 - self.gf[k])).clamp(-1.0, 1.0);
            dz[2 * H + k] = (d_o * self.go[k] * (1.0 - self.go[k])).clamp(-1.0, 1.0);
            dz[3 * H + k] = (dg * (1.0 - self.gg[k] * self.gg[k])).clamp(-1.0, 1.0);
        }

        let xrow = &mut self.wx[self.input * G4..(self.input + 1) * G4];
        for g in 0..G4 {
            let d = dz[g];
            self.b[g] -= LR * d;
            xrow[g] -= LR * d;
            let wr = &mut self.wh[g * H..(g + 1) * H];
            for j in 0..H {
                wr[j] -= LR * d * self.h_in[j];
            }
        }
    }
}

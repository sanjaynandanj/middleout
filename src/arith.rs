//! Binary arithmetic coder (carry-free range coder, lpaq style).
//! Probabilities are 12-bit: p in 0..4096 is P(bit == 1).

pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder { x1: 0, x2: 0xFFFF_FFFF, out: Vec::new() }
    }

    pub fn encode(&mut self, bit: u32, p: u16) {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p as u32;
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..4 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
        self.out
    }
}

pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: 0xFFFF_FFFF, x: 0, input, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next_byte() as u32;
        }
        d
    }

    fn next_byte(&mut self) -> u8 {
        let b = self.input.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    pub fn decode(&mut self, p: u16) -> u32 {
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p as u32;
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.next_byte() as u32;
        }
        bit
    }
}

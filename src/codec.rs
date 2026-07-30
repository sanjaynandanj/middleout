//! Container format: "MOUT" | mode u8 | original_len u64 LE | payload.
//! Modes: 0 = stored raw, 1 = middle-out LZ, 2 = AI context mixer.

use crate::{lz, model};
use anyhow::{Result, bail, ensure};

pub const MAGIC: &[u8; 4] = b"MOUT";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Raw = 0,
    Lz = 1,
    Ai = 2,
}

pub fn compress(data: &[u8], ai: bool) -> Vec<u8> {
    let (mode, payload) = if ai {
        (Mode::Ai, model::compress(data))
    } else {
        (Mode::Lz, lz::compress(data))
    };
    let (mode, payload) = if payload.len() >= data.len() {
        (Mode::Raw, data.to_vec())
    } else {
        (mode, payload)
    };
    let mut out = Vec::with_capacity(13 + payload.len());
    out.extend_from_slice(MAGIC);
    out.push(mode as u8);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

pub fn decompress(container: &[u8]) -> Result<Vec<u8>> {
    ensure!(container.len() >= 13, "file too short to be a .mo container");
    ensure!(&container[..4] == MAGIC, "bad magic: not a middleout file");
    let mode = container[4];
    let orig_len = u64::from_le_bytes(container[5..13].try_into().unwrap()) as usize;
    let payload = &container[13..];
    let out = match mode {
        0 => {
            ensure!(payload.len() == orig_len, "raw payload length mismatch");
            payload.to_vec()
        }
        1 => lz::decompress(payload, orig_len)?,
        2 => model::decompress(payload, orig_len),
        m => bail!("unknown mode byte {m}"),
    };
    ensure!(out.len() == orig_len, "decompressed length mismatch");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift_bytes(n: usize) -> Vec<u8> {
        let mut state = 0x2545F491_4F6CDD1Du64;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect()
    }

    fn corpora() -> Vec<Vec<u8>> {
        vec![
            vec![],
            vec![42],
            vec![7; 100_000],
            b"tip to tip efficiency".repeat(500),
            xorshift_bytes(50_000),
            include_bytes!("model.rs").to_vec(),
            (0u32..2_000).flat_map(|i| (i % 251) as u8..=255).collect(),
        ]
    }

    #[test]
    fn roundtrip_lz() {
        for data in corpora() {
            let c = compress(&data, false);
            assert_eq!(decompress(&c).unwrap(), data, "lz failed on len {}", data.len());
        }
    }

    #[test]
    fn roundtrip_ai() {
        for data in corpora() {
            let c = compress(&data, true);
            assert_eq!(decompress(&c).unwrap(), data, "ai failed on len {}", data.len());
        }
    }

    #[test]
    fn incompressible_falls_back_to_raw() {
        let data = xorshift_bytes(10_000);
        let c = compress(&data, false);
        assert_eq!(c.len(), data.len() + 13);
        assert_eq!(c[4], Mode::Raw as u8);
    }

    #[test]
    fn rejects_garbage() {
        assert!(decompress(b"not a container at all").is_err());
        assert!(decompress(b"").is_err());
    }
}

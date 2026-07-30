//! Middle-out LZ: split the input at the midpoint and compress both halves
//! working outward from the middle (the first half is processed reversed).
//! Each half is LZSS with a 64K window, hash-chain matching, and
//! flag-byte token groups.

use anyhow::{Result, bail, ensure};

const MIN_MATCH: usize = 4;
const MAX_MATCH: usize = MIN_MATCH + 255;
const WINDOW: usize = 65_535;
const HASH_BITS: u32 = 16;
const MAX_CHAIN: usize = 64;

pub fn compress(data: &[u8]) -> Vec<u8> {
    let mid = data.len() / 2;
    let first_rev: Vec<u8> = data[..mid].iter().rev().copied().collect();
    let a = lzss(&first_rev);
    let b = lzss(&data[mid..]);
    let mut out = Vec::with_capacity(8 + a.len() + b.len());
    out.extend_from_slice(&(a.len() as u64).to_le_bytes());
    out.extend_from_slice(&a);
    out.extend_from_slice(&b);
    out
}

pub fn decompress(payload: &[u8], orig_len: usize) -> Result<Vec<u8>> {
    ensure!(payload.len() >= 8, "truncated LZ payload");
    let alen = u64::from_le_bytes(payload[..8].try_into().unwrap()) as usize;
    ensure!(payload.len() >= 8 + alen, "truncated LZ payload");
    let mid = orig_len / 2;
    let mut first = unlzss(&payload[8..8 + alen], mid)?;
    first.reverse();
    let second = unlzss(&payload[8 + alen..], orig_len - mid)?;
    first.extend_from_slice(&second);
    Ok(first)
}

fn hash4(bytes: &[u8]) -> usize {
    let v = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

enum Token {
    Literal(u8),
    Match { dist: u16, len: u8 }, // real length = len + MIN_MATCH
}

fn find_match(data: &[u8], head: &[usize], prev: &[usize], i: usize) -> (usize, usize) {
    if i + MIN_MATCH > data.len() {
        return (0, 0);
    }
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    let max_len = MAX_MATCH.min(data.len() - i);
    let mut j = head[hash4(&data[i..])];
    let mut chain = 0;
    while j != usize::MAX && chain < MAX_CHAIN {
        let dist = i - j;
        if dist > WINDOW {
            break;
        }
        let mut len = 0;
        while len < max_len && data[j + len] == data[i + len] {
            len += 1;
        }
        if len > best_len {
            best_len = len;
            best_dist = dist;
            if len == max_len {
                break;
            }
        }
        j = prev[j];
        chain += 1;
    }
    (best_len, best_dist)
}

fn lzss(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut prev = vec![usize::MAX; data.len()];
    let mut group: Vec<Token> = Vec::with_capacity(8);

    let insert = |head: &mut [usize], prev: &mut [usize], pos: usize| {
        let h = hash4(&data[pos..]);
        prev[pos] = head[h];
        head[h] = pos;
    };

    let mut i = 0;
    while i < data.len() {
        let (len, dist) = find_match(data, &head, &prev, i);
        if i + MIN_MATCH <= data.len() {
            insert(&mut head, &mut prev, i);
        }

        // Lazy matching: if the next position holds a strictly longer match,
        // emit a literal here and take the longer match next iteration.
        let take_match = len >= MIN_MATCH
            && !(len < MAX_MATCH
                && i + 1 + MIN_MATCH <= data.len()
                && find_match(data, &head, &prev, i + 1).0 > len);

        if take_match {
            group.push(Token::Match {
                dist: dist as u16,
                len: (len - MIN_MATCH) as u8,
            });
            let end = (i + len).min(data.len().saturating_sub(MIN_MATCH - 1));
            for pos in (i + 1)..end {
                insert(&mut head, &mut prev, pos);
            }
            i += len;
        } else {
            group.push(Token::Literal(data[i]));
            i += 1;
        }

        if group.len() == 8 {
            flush_group(&mut out, &group);
            group.clear();
        }
    }
    if !group.is_empty() {
        flush_group(&mut out, &group);
    }
    out
}

fn flush_group(out: &mut Vec<u8>, group: &[Token]) {
    let mut flags = 0u8;
    for (k, tok) in group.iter().enumerate() {
        if matches!(tok, Token::Match { .. }) {
            flags |= 1 << k;
        }
    }
    out.push(flags);
    for tok in group {
        match tok {
            Token::Literal(b) => out.push(*b),
            Token::Match { dist, len } => {
                out.extend_from_slice(&dist.to_le_bytes());
                out.push(*len);
            }
        }
    }
}

fn unlzss(input: &[u8], out_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(out_len);
    let mut pos = 0;
    while out.len() < out_len {
        ensure!(pos < input.len(), "truncated LZ stream");
        let flags = input[pos];
        pos += 1;
        for k in 0..8 {
            if out.len() == out_len {
                break;
            }
            if flags & (1 << k) != 0 {
                ensure!(pos + 3 <= input.len(), "truncated match token");
                let dist = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
                let len = input[pos + 2] as usize + MIN_MATCH;
                pos += 3;
                ensure!(dist != 0 && dist <= out.len(), "bad match distance");
                ensure!(out.len() + len <= out_len, "match overruns output");
                let start = out.len() - dist;
                for idx in 0..len {
                    out.push(out[start + idx]);
                }
            } else {
                ensure!(pos < input.len(), "truncated literal");
                out.push(input[pos]);
                pos += 1;
            }
        }
    }
    if pos < input.len() {
        bail!("trailing garbage in LZ stream");
    }
    Ok(out)
}

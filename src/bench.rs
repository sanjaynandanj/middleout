//! Benchmark: middleout (both engines) vs gzip and zstd, scored with the
//! Weissman score: W = alpha * (r / r_ref) * (log T_ref / log T),
//! alpha = 1, times in milliseconds, gzip -6 as the reference codec.

use anyhow::{Context, Result, ensure};
use middleout::codec;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

struct Entry {
    name: &'static str,
    compressed: usize,
    ratio: f64,
    comp_ms: f64,
    decomp_ms: f64,
}

fn weissman(ratio: f64, time_ms: f64, ref_ratio: f64, ref_time_ms: f64) -> f64 {
    let t = time_ms.max(1.05);
    let tr = ref_time_ms.max(1.05);
    (ratio / ref_ratio) * (tr.ln() / t.ln())
}

fn gzip_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn gzip_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = flate2::write::GzDecoder::new(Vec::new());
    dec.write_all(data).unwrap();
    dec.finish().unwrap()
}

fn measure(
    name: &'static str,
    data: &[u8],
    compress: impl Fn(&[u8]) -> Vec<u8>,
    decompress: impl Fn(&[u8]) -> Vec<u8>,
) -> Result<Entry> {
    let t0 = Instant::now();
    let compressed = compress(data);
    let comp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let restored = decompress(&compressed);
    let decomp_ms = t1.elapsed().as_secs_f64() * 1000.0;
    ensure!(restored == data, "{name}: roundtrip verification FAILED");

    Ok(Entry {
        name,
        compressed: compressed.len(),
        ratio: data.len() as f64 / compressed.len().max(1) as f64,
        comp_ms,
        decomp_ms,
    })
}

pub fn run(files: &[std::path::PathBuf]) -> Result<()> {
    for file in files {
        bench_file(file)?;
    }
    Ok(())
}

fn bench_file(path: &Path) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    println!("\n=== {} ({} bytes) ===", path.display(), data.len());

    let entries = vec![
        measure("gzip -6 (ref)", &data, gzip_compress, gzip_decompress)?,
        measure(
            "zstd -3",
            &data,
            |d| zstd::encode_all(d, 3).unwrap(),
            |d| zstd::decode_all(d).unwrap(),
        )?,
        measure(
            "zstd -19",
            &data,
            |d| zstd::encode_all(d, 19).unwrap(),
            |d| zstd::decode_all(d).unwrap(),
        )?,
        measure(
            "middleout-lz",
            &data,
            |d| codec::compress(d, false),
            |d| codec::decompress(d).unwrap(),
        )?,
        measure(
            "middleout-ai",
            &data,
            |d| codec::compress(d, true),
            |d| codec::decompress(d).unwrap(),
        )?,
    ];

    let (ref_ratio, ref_ms) = (entries[0].ratio, entries[0].comp_ms);
    println!(
        "{:<16} {:>12} {:>8} {:>10} {:>10} {:>10}",
        "codec", "compressed", "ratio", "comp ms", "decomp ms", "Weissman"
    );
    for e in &entries {
        println!(
            "{:<16} {:>12} {:>8.3} {:>10.1} {:>10.1} {:>10.3}",
            e.name,
            e.compressed,
            e.ratio,
            e.comp_ms,
            e.decomp_ms,
            weissman(e.ratio, e.comp_ms, ref_ratio, ref_ms)
        );
    }
    println!("(alpha = 1, times in ms, roundtrip verified for every codec)");
    Ok(())
}

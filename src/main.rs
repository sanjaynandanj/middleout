mod arith;
mod bench;
mod codec;
mod lz;
mod model;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "middleout",
    version,
    about = "Middle-out compression, Pied Piper style.",
    long_about = "Middle-out compression, Pied Piper style.\n\
                  Default engine: middle-out LZ (fast). --ai switches to an\n\
                  online-learning context-mixing model (slow, dense)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compress a file to <input>.mo
    Compress {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Use the AI context-mixing engine (better ratio, slower)
        #[arg(long)]
        ai: bool,
    },
    /// Decompress a .mo file
    Decompress {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Benchmark against gzip/zstd and print Weissman scores
    Bench { files: Vec<PathBuf> },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Compress { input, output, ai } => cmd_compress(&input, output, ai),
        Cmd::Decompress { input, output } => cmd_decompress(&input, output),
        Cmd::Bench { files } => bench::run(&files),
    }
}

fn cmd_compress(input: &Path, output: Option<PathBuf>, ai: bool) -> Result<()> {
    let data = std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let out_path = output.unwrap_or_else(|| {
        let mut p = input.as_os_str().to_owned();
        p.push(".mo");
        PathBuf::from(p)
    });

    let t0 = Instant::now();
    let container = codec::compress(&data, ai);
    let elapsed = t0.elapsed();

    std::fs::write(&out_path, &container)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let ratio = data.len() as f64 / container.len().max(1) as f64;
    println!(
        "{} -> {} | {} -> {} bytes | ratio {:.3} | {:.1} ms | engine: {}",
        input.display(),
        out_path.display(),
        data.len(),
        container.len(),
        ratio,
        elapsed.as_secs_f64() * 1000.0,
        if ai { "ai context-mixer" } else { "middle-out lz" },
    );
    Ok(())
}

fn cmd_decompress(input: &Path, output: Option<PathBuf>) -> Result<()> {
    let container =
        std::fs::read(input).with_context(|| format!("reading {}", input.display()))?;
    let out_path = output.unwrap_or_else(|| {
        match input.extension().and_then(|e| e.to_str()) {
            Some("mo") => input.with_extension(""),
            _ => {
                let mut p = input.as_os_str().to_owned();
                p.push(".out");
                PathBuf::from(p)
            }
        }
    });

    let t0 = Instant::now();
    let data = codec::decompress(&container)?;
    let elapsed = t0.elapsed();

    std::fs::write(&out_path, &data)
        .with_context(|| format!("writing {}", out_path.display()))?;
    println!(
        "{} -> {} | {} bytes restored | {:.1} ms",
        input.display(),
        out_path.display(),
        data.len(),
        elapsed.as_secs_f64() * 1000.0,
    );
    Ok(())
}

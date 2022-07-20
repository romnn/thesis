#![allow(warnings)]

use anyhow::Result;
use buildtools;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn build_nvbit(build_dir: &Path, version: &str) -> Result<()> {
    let compressed = build_dir.join(format!("nvbit-{}.tar.bz2", version));
    let _ = std::fs::remove_file(&compressed);
    buildtools::http::download_file(
        &format!(
            "https://github.com/NVlabs/NVBit/releases/download/{}/nvbit-Linux-x86_64-{}.tar.bz2",
            version, version
        ),
        &compressed,
    )?;
    let tarball = build_dir.join(format!("nvbit-{}", version));
    let _ = std::fs::remove_file(&tarball);
    buildtools::decompress::decompress_tar_bz2(&compressed, &tarball)?;
    Ok(())
}

fn main() -> Result<()> {
    let start = Instant::now();
    let output_base_path = buildtools::output();
    build_nvbit(
        &output_base_path,
        &env::var("NVBIT_VERSION").unwrap_or("1.5.5".to_string()),
    )?;
    Ok(())
}

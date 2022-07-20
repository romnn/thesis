use anyhow::Result;
use std::path::Path;

pub fn download_file(url: &str, dest: &Path) -> Result<()> {
    let resp = reqwest::blocking::get(url)?;
    let mut dest = std::fs::OpenOptions::new()
        .read(false)
        .truncate(true)
        .write(true)
        .create(true)
        .open(dest)?;
    let mut stream = std::io::Cursor::new(resp.bytes()?);
    std::io::copy(&mut stream, &mut dest)?;
    Ok(())
}

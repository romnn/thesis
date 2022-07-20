use anyhow::Result;
use bzip2::read::BzDecoder;
use std::fs::File;
use std::path::Path;
use tar::Archive;

pub fn decompress_tar_bz2(src: &Path, dest: &Path) -> Result<()> {
    let compressed = File::open(src)?;
    // let _ = std::fs::remove_file(&dest);
    // std::fs::create_dir(&dest)?;
    // let mut decompressed = std::fs::OpenOptions::new()
    //     .read(false)
    //     .truncate(true)
    //     .write(true)
    //     .create(true)
    //     .open(dest)?;
    let stream = BzDecoder::new(compressed);
    let mut archive = Archive::new(stream);
    archive.unpack(&dest)?;
    // decompressor.read_into
    // let mut stream = DecoderReader::new(compressed);
    // std::io::copy(&mut stream, &mut decompressed)?;
    Ok(())
}

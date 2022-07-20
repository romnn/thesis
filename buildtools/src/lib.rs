pub mod git;
pub mod http;
pub mod decompress;

use std::path::PathBuf;
use std::env;

pub fn output() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

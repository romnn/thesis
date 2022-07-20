// pub mod ptx;
use ptx;
use std::path::PathBuf;

pub fn main() {
    ptx::gpgpu_ptx_sim_load_ptx_from_filename(&PathBuf::from("../kernels/mm/mm.sm_75.ptx"))
        .unwrap();
}

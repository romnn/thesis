use anyhow::Result;
use git2 as git;
use regex::Regex;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn output() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

fn find_cuda_bin(name: &str) -> PathBuf {
    env::var("CUDA_INSTALL_PATH")
        .ok()
        .map(|cuda_install| [&cuda_install, "bin", name].iter().collect())
        .and_then(|bin: PathBuf| if bin.exists() { Some(bin) } else { None })
        .unwrap_or(PathBuf::from(name))
}

fn build_env() -> Result<()> {
    let repo = git::Repository::open(".").expect("couldn't open repository");
    let obj = repo.head()?.resolve()?.peel(git2::ObjectType::Commit)?;
    let commit = obj
        .into_commit()
        .map_err(|_| git::Error::from_str("couldn't find commit"))?;
    println!("cargo:warning=commit:{:?}", commit.id());
    // GPGPUSIM_BUILD := gpgpu-sim_git-commit-$(GIT_COMMIT)_modified_$(GIT_FILES_CHANGED)
    let nvcc_bin_path = find_cuda_bin("nvcc");

    lazy_static::lazy_static! {
        static ref CUDA_VERSION_REG: Regex = Regex::new(r"release\s*(?P<cuda_version>[\d\.]*)\s*,").unwrap();
    }

    let cuda_version_string = Command::new(nvcc_bin_path).arg("--version").output()?;
    let cuda_version_string = String::from_utf8(cuda_version_string.stdout)?;
    let cuda_version_string = CUDA_VERSION_REG
        .captures(&cuda_version_string)
        .and_then(|cap| cap.name("cuda_version"))
        .map(|cap| cap.as_str())
        .ok_or(anyhow::anyhow!("failed to get cuda version"))?;
    println!("cargo:warning=cuda_version:{:?}", cuda_version_string);

    lazy_static::lazy_static! {
        static ref CUDART_VERSION_REG: Regex = Regex::new(r"\s*(?P<major>\d*).(?P<minor>\d*)\s*").unwrap();
    }
    let cudart_version: u32 = CUDART_VERSION_REG
        .captures(&cuda_version_string)
        .and_then(|cap| {
            let major: Option<u32> = cap
                .name("major")
                .and_then(|major| major.as_str().parse().ok());
            let minor: Option<u32> = cap
                .name("minor")
                .and_then(|minor| minor.as_str().parse().ok());
            match (major, minor) {
                (Some(major), Some(minor)) => {
                    format!("{:02}{:02}", 10 * major, 10 * minor).parse().ok()
                }
                _ => None,
            }
        })
        .ok_or(anyhow::anyhow!("failed to get cudart version"))?;

    println!("cargo:warning=cudart_version:{:?}", cudart_version);
    // cargo:rustc-cfg=cudart_le_10010"
    // cargo:rustc-cfg=cudart_ge_10010"
    // cargo:rustc-cfg=cudart_eq_10010"
    // cargo:rustc-cfg=cudart_gt_10010"
    // cargo:rustc-cfg=cudart_lt_10010"

    // # Detect CUDA Runtime Version
    // CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
    // CUDART_VERSION:=$(shell echo $(CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

    // # Detect GCC Version
    // CC_VERSION := $(shell gcc --version | head -1 | awk '{for(i=1;i<=NF;i++){ if(match($$i,/^[0-9]\.[0-9]\.[0-9]$$/))  {print $$i; exit 0 }}}')

    // # Detect Support for C++11 (C++0x) from GCC Version
    // GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=include/cuda_runtime_api.h");
    build_env().unwrap();

    // let include_paths: Vec<PathBuf> = vec![];
    // let clang_includes = include_paths
    //     .iter()
    //     .map(|include| format!("-I{}", include.to_string_lossy()));

    // #Detect Git branch and commit #
    // let git_branch = Command::new("git")
    //     .arg("rev-parse")
    //     .arg("--abbrev-ref")
    //     .arg("HEAD")
    //     .output();

    // GIT_COMMIT := $(shell git log -n 1 | head -1 | sed -re 's/commit (.*)/\1/')
    // GIT_FILES_CHANGED_A:=$(shell git diff --numstat | wc | sed -re 's/^\s+([0-9]+).*/\1./')
    // GIT_FILES_CHANGED:= $(GIT_FILES_CHANGED_A)$(shell git diff --numstat --cached | wc | sed -re 's/^\s+([0-9]+).*/\1/')
    // GPGPUSIM_BUILD := gpgpu-sim_git-commit-$(GIT_COMMIT)_modified_$(GIT_FILES_CHANGED)
    // endif

    // # Detect CUDA Runtime Version
    // CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
    // CUDART_VERSION:=$(shell echo $(CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')

    // # Detect GCC Version
    // CC_VERSION := $(shell gcc --version | head -1 | awk '{for(i=1;i<=NF;i++){ if(match($$i,/^[0-9]\.[0-9]\.[0-9]$$/))  {print $$i; exit 0 }}}')

    // # Detect Support for C++11 (C++0x) from GCC Version
    // GNUC_CPP0X := $(shell gcc --version | perl -ne 'if (/gcc\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

    let builder = bindgen::Builder::default()
        .ctypes_prefix("libc")
        .allowlist_type("^cuda.*")
        .allowlist_type("^_cuda.*")
        .allowlist_type("^(u)int.*")
        .allowlist_type("^CU.*")
        .allowlist_var("^cuda.*")
        .generate_comments(false)
        .rustified_enum("*")
        // not sure about this one
        .size_t_is_usize(false)
        .prepend_enum_name(false)
        .derive_eq(true)
        .derive_ord(true)
        .derive_hash(true)
        .derive_default(true)
        // found using locate cuda_runtime_api.h
        // on my machine, found under /usr/include/cuda_runtime_api.h
        .header("include/cuda_runtime_api.h")
        .header("include/cuda_api.h");

    let bindings = builder
        .generate()
        .expect("Unable to generate bindings")
        .to_string();

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    println!("cargo:warning={:?}", output().join("bindings.rs"));
    fs::write(output().join("bindings.rs"), &bindings).expect("Couldn't write bindings!");
}

// bindgen \
//   --whitelist-type="^cuda.*" \
//   --whitelist-type="^surfaceReference" \
//   --whitelist-type="^textureReference" \
//   --whitelist-var="^cuda.*" \
//   --whitelist-function="^cuda.*" \
//   --default-enum-style=rust \
//   --no-doc-comments \
//   --with-derive-default \
//   --with-derive-eq \
//   --with-derive-hash \
//   --with-derive-ord \
//   /opt/cuda/include/cuda_runtime.h \
//   > src/cuda_runtime.rs

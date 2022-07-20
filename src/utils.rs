use super::context::GPGPUContext;
use anyhow::Result;
use base64ct::{Base64, Encoding};
use md5::{Digest, Md5};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::RwLock;
use std::{env, fs, io};

// Extract the code using cuobjdump and remove unnecessary sections

pub fn md5sum(path: &Path) -> Result<String> {
    let mut file = fs::File::open(&path)?;
    let mut hasher = Md5::new();
    io::copy(&mut file, &mut hasher)?;
    let hash = hasher.finalize();
    Ok(Base64::encode_string(&hash))
}

impl GPGPUContext {
    pub fn extract_code_using_cuobjdump(&self) -> Result<()> {
        // CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
        // // prevent the dumping by cuobjdump everytime we execute the code!
        // const char *override_cuobjdump = getenv("CUOBJDUMP_SIM_FILE");
        // char command[1000];
        // std::string app_binary = get_app_binary();
        // // Running cuobjdump using dynamic link to current process
        // snprintf(command, 1000, "md5sum %s ", app_binary.c_str());
        // printf("Running md5sum using \"%s\"\n", command);
        // if (system(command)) {
        // std::cout << "Failed to execute: " << command << std::endl;
        // exit(1);
        // }

        // check if the binary was modified
        let app_bin_path =
            process_path::get_executable_path().ok_or(anyhow::anyhow!("no executable path"))?;
        let digest = md5sum(&app_bin_path);
        println!("digest: {:?}", digest);

        // #if (CUDART_VERSION >= 6000)
        // snprintf(command, 1000,
        //        "$CUDA_INSTALL_PATH/bin/cuobjdump -lptx %s  | cut -d \":\" -f 2 | "
        //        "awk '{$1=$1}1' > %s",
        //        app_binary.c_str(), ptx_list_file_name);

        // .ok_or(anyhow::anyhow!("env CUDA_INSTALL_PATH not set"))?

        let cuobjdump_bin_path: PathBuf = env::var("CUDA_INSTALL_PATH")
            .ok()
            .map(|cuda_install| [&cuda_install, "bin", "cuobjdump"].iter().collect())
            .and_then(|bin: PathBuf| if bin.exists() { Some(bin) } else { None })
            .unwrap_or(PathBuf::from("cuobjdump"));
        println!("cuobjdump -lptx {:?}", &app_bin_path);
        let cuobjdump_result = Command::new(&cuobjdump_bin_path)
            .arg("-lptx")
            .arg(&app_bin_path)
            .output()?;
        let cuobjdump_out = String::from_utf8(cuobjdump_result.stdout)?;

        lazy_static::lazy_static! {
            static ref PTX_FILE_REG: Regex = Regex::new(r"PTX file\s*\d*:\s*(?P<ptx_file>\S*)\s*").unwrap();
        }
        // println!("test {:?}", &cuobjdump_out);

        // let ptx_files: Vec<&str> = PTX_FILE_REG
        let ptx_files = PTX_FILE_REG
            .captures_iter(&cuobjdump_out)
            .filter_map(|cap| cap.name("ptx_file"))
            .map(|cap| cap.as_str());

        let mut no_of_ptx = 0;
        // let mut version_filename: HashMap<u32, HashSet<String>> = HashMap::new();
        // let mut version_filename: HashMap<u32, HashSet<String>> = HashMap::new();
        for ptx_file in ptx_files {
            println!("Extracting specific PTX file named {}", ptx_file);
            match Command::new(&cuobjdump_bin_path)
                .arg("-xptx")
                .arg(&ptx_file)
                .arg(&app_bin_path)
                .output()
            {
                Ok(cuobjdump_ptx_file_result) => {
                    // todo: not needed
                    let cuobjdump_ptx_file_out =
                        String::from_utf8(cuobjdump_ptx_file_result.stdout)?;
                    println!("{:?}", cuobjdump_ptx_file_out);

                    // group ptx files based on sm architecture
                    lazy_static::lazy_static! {
                        static ref PTX_SM_VERSION_REG: Regex = Regex::new(r"sm_(?P<ptx_sm_version>\d+)\.ptx").unwrap();
                    };

                    let ptx_sm_version: u32 = PTX_SM_VERSION_REG
                        .captures(&ptx_file)
                        .and_then(|cap| cap.name("ptx_sm_version"))
                        .and_then(|cap| cap.as_str().parse().ok())
                        .ok_or(anyhow::anyhow!("PTX list is not in correct format"))?;

                    // cannot use .write()? here, because the Error (Poison Error) that would be
                    // returned is not Send
                    let mut version_filename = self.version_filename.write().unwrap();
                    let mut sm_ptx_files = version_filename
                        .entry(ptx_sm_version)
                        .or_insert(HashSet::new());
                    // .or_insert("tests".to_string());
                    sm_ptx_files.insert(ptx_file.to_string());
                    no_of_ptx += 1;
                }
                Err(err) => println!("{:?}", err),
            };
            // snprintf(command, 1000, "$CUDA_INSTALL_PATH/bin/cuobjdump -xptx %s %s",
            //                         ptx_file, app_binary.c_str());
            //       if (system(command) != 0) {
            //                   printf("ERROR: command: %s failed \n", command);
            //                           exit(0);
            //                                 }
        }

        if (no_of_ptx < 1) {
            println!(
                "WARNING: Number of ptx in the executable file are 0. One of the reasons might be"
            );
            println!("\t1. CDP is enabled");
            println!("\t2. When using PyTorch, PYTORCH_BIN is not set correctly");
        }
        // .collect();
        // println!("ptx files: {:?}", ptx_files);
        //
        // std::ifstream infile(ptx_list_file_name);
        //   std::string line;
        //   while (std::getline(infile, line)) {
        //     // int pos = line.find(std::string(get_app_binary_name(app_binary)));
        //     int pos1 = line.find("sm_");
        //     int pos2 = line.find_last_of(".");
        //     if (pos1 == std::string::npos && pos2 == std::string::npos) {
        //       printf("ERROR: PTX list is not in correct format");
        //       exit(0);
        //     }
        //     std::string vstr = line.substr(pos1 + 3, pos2 - pos1 - 3);
        //     int version = atoi(vstr.c_str());
        //     if (version_filename.find(version) == version_filename.end()) {
        //       version_filename[version] = std::set<std::string>();
        //     }
        //     version_filename[version].insert(line);
        //   }
        Ok(())
    }

    pub fn pruneSectionList(&self) -> Result<()> {
        // unsigned forced_max_capability = context->get_device()
        //                                        ->get_gpgpu()
        //                                        ->get_config()
        //                                        .get_forced_max_capability();
        Ok(())
    }

    pub fn cuobjdumpInit(&self) -> Result<()> {
        // CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
        // let ctx = GPGPUContext::new();
        // let ctx = &GLOBAL_CTX;
        self.extract_code_using_cuobjdump()?;

        // this can be used to speed up parsing and analyzing the PTX code
        if env::var("CUOBJDUMP_SIM_FILE").is_ok() {
            let cuobjdumpSectionList = self.pruneSectionList();
            // let cuobjdumpSectionList = mergeSections();
        };
        Ok(())
        // const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
        //   if (pre_load == NULL || strlen(pre_load) == 0) {
        //     cuobjdumpSectionList = pruneSectionList(context);
        //     cuobjdumpSectionList = mergeSections();
        //   }
    }
}

// void cuda_runtime_api::cuobjdumpInit() {
//   CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
//   extract_code_using_cuobjdump();  // extract all the output of cuobjdump to
//                                    // _cuobjdump_*.*
//   const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
//   if (pre_load == NULL || strlen(pre_load) == 0) {
//     cuobjdumpSectionList = pruneSectionList(context);
//     cuobjdumpSectionList = mergeSections();
//   }
// }

// pub fn get_app_binary() -> Result<&str> {
// char self_exe_path[1025];
// #ifdef __APPLE__
// uint32_t size = sizeof(self_exe_path);
// if (_NSGetExecutablePath(self_exe_path, &size) != 0) {
//   printf("GPGPU-Sim ** ERROR: _NSGetExecutablePath input buffer too small\n");
//   exit(1);
// }
// #else
// std::stringstream exec_link;
// exec_link << "/proc/self/exe";
// cfg_if::cfg_if! {
//     if #[cfg(target_os = macos)] {
//     } else if #[cfg(target_family = unix)] {
//         fs::read_link("/proc/self/exe")
//     } else {
//         anyhow::anyhow!("unsupported architecture")
//     }
// }
// let path = ;
// ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
// assert(path_length != -1);
// self_exe_path[path_length] = '\0';
// #endif

// printf("self exe links to: %s\n", self_exe_path);
// return self_exe_path;
// }

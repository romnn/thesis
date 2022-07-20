use super::cuda_runtime_api as rt;
use ptx;
use super::utils;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::path::PathBuf;
use std::sync;

// this is the factory function for the global gpgpu context static singleton
// gpgpu_context *GPGPU_Context() {
//   static gpgpu_context *gpgpu_ctx = NULL;
//   if (gpgpu_ctx == NULL) {
//     gpgpu_ctx = new gpgpu_context();
//   }
//   return gpgpu_ctx;
// }

lazy_static::lazy_static! {
    // pub static ref GLOBAL_CTX: sync::Mutex<GPGPUContext> = sync::Mutex::new(GPGPUContext::default());
    pub static ref GLOBAL_CTX: GPGPUContext = GPGPUContext::default();
}

pub struct CUctx_st {
    // /// m_code is fat binary handle => global symbol table
    // m_code: HashMap<u32, *SymbolTable>,
    m_last_fat_cubin_handle: u32,
    // /// m_kernel_lookup is unique id (CUDA app function address) => kernel entry point
    // m_kernel_lookup: HashMap<*const ffi::c_void, *mut function_info>,
    // /// todo
    // m_binary_info: gpgpu_ptx_sim_info,
}
// _cuda_device_id *m_gpu;  // selected gpu
//   std::map<unsigned, symbol_table *>
//       m_code;  // fat binary handle => global symbol table
//   unsigned m_last_fat_cubin_handle;
//   std::map<const void *, function_info *>
//       m_kernel_lookup;  // unique id (CUDA app function address) => kernel entry
//                         // point
//   struct gpgpu_ptx_sim_info m_binary_info;

struct GPGPU {}

fn to_char_slice(s: &str, len: usize) -> Result<&[libc::c_char]> {
    let cs = ffi::CString::new(s)?;
    unsafe {
        Ok(std::slice::from_raw_parts(
            cs.as_bytes_with_nul().as_ptr() as *const libc::c_char,
            std::cmp::min(s.len(), len),
        ))
    }
}

impl GPGPU {
    pub fn props(&self) -> Result<rt::cudaDeviceProp> {
        let mut name: [libc::c_char; 256usize] = [0; 256];
        name.copy_from_slice(to_char_slice("test", 256usize)?);
        Ok(rt::cudaDeviceProp {
            name,
            ..Default::default()
        })
    }
}

// #[derive(Send, Sync)]
pub struct GPGPUContext {
    // pub next_fat_bin_handle: sync::atomic::AtomicUsize,
    pub next_fat_cubin_handle: sync::Mutex<libc::c_ulonglong>, // c_uint
    pub version_filename: sync::RwLock<HashMap<u32, HashSet<String>>>,
    // pub version_filename: HashMap<u32, String>,
    // pub version_filename: sync::Arc<sync::RwLock<HashMap<u32, HashSet<String>>>>,
    // pub version_filename: sync::Arc<sync::RwLock<HashMap<u32, String>>>,
    // pub ctx: sync::Mutex<*mut rt::CUcontext>,
    // symbol_table *g_global_allfiles_symbol_table;
    // const char *g_filename;
    // unsigned sm_next_access_uid;
    // unsigned warp_inst_sm_next_uid;
    // unsigned operand_info_sm_next_uid;  // uid for operand_info
    // unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
    // unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
    // unsigned long long g_ptx_cta_info_uid;
    // unsigned symbol_sm_next_uid;  // uid for symbol
    // unsigned function_info_sm_next_uid;
    // std::vector<ptx_instruction *>
    //   s_g_pc_to_insn;  // a direct mapping from PC to instruction
    // bool debug_tensorcore;

    // // objects pointers for each file
    // cuda_runtime_api *api;
    // ptxinfo_data *ptxinfo;
    // ptx_recognizer *ptx_parser;
    // GPGPUsim_ctx *the_gpgpusim;
    // cuda_sim *func_sim;
    // cuda_device_runtime *device_runtime;
    // ptx_stats *stats;
}

impl GPGPUContext {
    // pub fn new() -> Self {
    //     Self {}
    // }

    // pub fn ctx(&self) -> rt::CUContext {
    //     // let _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();

    // }
    pub fn cudaRegisterFatBinary(&self, fatCubin: *mut ffi::c_void) -> *mut *mut ffi::c_void {
        // static mut next_fat_bin_handle: usize = 0; //AtomicUsize::new(0);
        // static next_fat_bin_handle: sync::atomic::AtomicUsize = sync::atomic::AtomicUsize::new(1);
        // static next_fat_bin_handle: sync::atomic::AtomicUsize = sync::atomic::AtomicUsize::new(1);
        // let next_fat_bin_handle: sync::atomic::AtomicUsize = sync::atomic::AtomicUsize::new(1);

        #[cfg(not(target_pointer_width = "64"))]
        println!(
          "GPGPU-Sim PTX: FatBin file name extraction has not been tested on "
          "32-bit system.");

        let app_binary_path = process_path::get_executable_path();
        println!("app binary: {:?}", app_binary_path);
        let filename = "default";

        // this is probably wrong
        // let fat_cubin_handle = 1;
        let mut next_fat_cubin_handle = self.next_fat_cubin_handle.lock().unwrap();
        // let mut fat_cubin_handle: Box<libc::c_ulonglong> = Box::new(*next_fat_cubin_handle);
        let mut fat_cubin_handle: libc::c_ulonglong = *next_fat_cubin_handle;
        *next_fat_cubin_handle += 1;
        // let fat_cubin_handle: libc::ulonglong = self.next_fat_bin_handle.lock().unwrap();
        // .load(sync::atomic::Ordering::SeqCst);
        // next_fat_bin_handle += 1;
        // self.next_fat_bin_handle
        //     .fetch_add(1, sync::atomic::Ordering::SeqCst);
        println!("fat_cubin_handle {}", fat_cubin_handle);

        if (fat_cubin_handle == 1) {
            self.cuobjdumpInit().unwrap();
        }
        // ctx->api->cuobjdumpInit();
        // ctx->api->cuobjdumpRegisterFatBinary(fat_cubin_handle, "default", context);
        // fat_cubin_handle.as_ptr()
        // &mut fat_cubin_handle as *mut *mut ffi::c_void // _ as *mut c_void;
        // &mut &mut *(&mut fat_cubin_handle as *mut _) as *mut *mut ffi::c_void
        // let
        // let state_ptr: *mut ffi::c_void = (&mut fat_cubin_handle as *mut _) as *mut ffi::c_void;
        // let state_ptr: *const *const ffi::c_void =

        // unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
        // let mut raw1: *mut ffi::c_void =
        //     (&mut fat_cubin_handle as *mut _) as *mut ffi::c_void;
        // let raw2: *mut *mut ffi::c_void = &mut raw1 as *mut *mut ffi::c_void;

        // let back: libc::c_ulonglong = unsafe { *(*(raw2 as *mut *mut libc::c_ulonglong)) };
        // assert!(*fat_cubin_handle == back);

        // let raw: *mut *mut ffi::c_void = unsafe { next_fat_cubin_handle as *mut *mut ffi::c_void };

        // let back: libc::c_ulonglong = unsafe { raw as libc::c_ulonglong };

        // std::mem::forget(fat_cubin_handle);
        // std::mem::forget(raw1);

        // raw2
        // void **fatCubinHandle = malloc(sizeof(void*));

        // pointer to value 1
        let raw1: *mut ffi::c_void = Box::into_raw(Box::new(fat_cubin_handle)) as *mut ffi::c_void;

        // pointer to pointer to value 1
        let raw2: *mut *mut ffi::c_void = Box::into_raw(Box::new(raw1)) as *mut *mut ffi::c_void;

        raw2
        // let raw1: Box<*mut ffi:c_void> = Box::into_raw(Box::new(1)) as *mut ffi::c_void;
        // Box::into_raw(raw2)
        // .into_raw()
        // fat_cubin_handle
        // 3
        // state_ptr
        // &mut &mut *(&mut fat_cubin_handle as *mut _) as *mut *mut ffi::c_void
        // *mut ffi::c_void // _ as *mut c_void;
        // ptr::null_mut()
    }

    pub fn cudaRegisterFunction(
        &self,
        fatCubinHandle: *mut *mut ffi::c_void,
        hostFun: *const libc::c_char,
        deviceFun: *mut libc::c_char,
        deviceName: *const libc::c_char,
        thread_limit: libc::c_int,
        tid: *mut rt::uint3,
        bid: *mut rt::uint3,
        bDim: *mut rt::dim3,
        gDim: *mut rt::dim3,
        wSize: *mut libc::c_int,
    ) {
        let fat_cubin_handle: libc::c_ulonglong =
            unsafe { *(*(fatCubinHandle as *mut *mut libc::c_ulonglong)) };
        // unsafe { *(*(fatCubinHandle as *mut *mut libc::c_ulonglong)) };
        // let back2: Box<libc::c_ulonglong> = unsafe {
        //     Box::from_raw(fatCubinHandle);
        // };
        // let fatCubinHandle: libc::c_ulonglong =
        //     unsafe { *(*(fatCubinHandle as *mut *mut libc::c_ulonglong)) };
        println!("handle: {}", fat_cubin_handle);
        self.cuobjdumpParseBinary(fat_cubin_handle);

        let hostFun = unsafe { ffi::CStr::from_ptr(hostFun) }
            .to_string_lossy()
            .into_owned();
        let deviceFun = unsafe { ffi::CStr::from_ptr(deviceFun) }
            .to_string_lossy()
            .into_owned();
        // let deviceName2 = unsafe { ffi::CStr::from_ptr(deviceName).to_str() };
        self.register_function(fat_cubin_handle, &hostFun, &deviceFun);

        // unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
        // this is where the ptx is loaded from the file
        // cuobjdumpParseBinary
        // gpgpu_ptx_sim_load_ptx_from_filename
        // will need a parser, grammar for cuobjdump?
        //
        // get the ptx files:
        println!("ptx files: {:?}", self.version_filename.read().unwrap());
    }

    pub fn cuobjdumpParseBinary(&self, fat_cubin_handle: libc::c_ulonglong) {
        // if (api->fatbin_registered[handle]) return;
        //   api->fatbin_registered[handle] = true;
        //     std::string fname = api->fatbinmap[handle];

        //     if (api->name_symtab.find(fname) != api->name_symtab.end()) {
        //     symbol_table *symtab = api->name_symtab[fname];
        //     context->add_binary(symtab, handle);
        //     return;
        //   }
        //   symbol_table *symtab;
        //
        // loops through all ptx files from smallest sm version to largest
        for (sm, ptx_files) in &*self.version_filename.read().unwrap() {
            for ptx_file in ptx_files {
                println!("GPGPU-Sim PTX: Parsing {}", ptx_file);
                let symtab = ptx::gpgpu_ptx_sim_load_ptx_from_filename(&PathBuf::from(ptx_file));
            }
        }
    }

    pub fn register_function(
        &self,
        fat_cubin_handle: libc::c_ulonglong,
        hostFun: &str,
        deviceFun: &str,
    ) -> () {
    }

    pub fn init() -> () {
        // let gpu = GPGPU {};
        // let props = gpu.props();
        // todo: set the properties for the gpu
        // struct _cuda_device_id *gpgpu_context::GPGPUSim_Init() {
        // _cuda_device_id *the_device = the_gpgpusim->the_cude_device;
        // if (!the_device) {
        //   gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

        //   cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
        //   snprintf(prop->name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
        //   prop->major = the_gpu->compute_capability_major();
        //   prop->minor = the_gpu->compute_capability_minor();
        //   prop->totalGlobalMem = 0x80000000 /* 2 GB */;
        //   prop->memPitch = 0;
        //   if (prop->major >= 2) {
        //     prop->maxThreadsPerBlock = 1024;
        //     prop->maxThreadsDim[0] = 1024;
        //     prop->maxThreadsDim[1] = 1024;
        //   } else {
        //     prop->maxThreadsPerBlock = 512;
        //     prop->maxThreadsDim[0] = 512;
        //     prop->maxThreadsDim[1] = 512;
        //   }

        //   prop->maxThreadsDim[2] = 64;
        //   prop->maxGridSize[0] = 0x40000000;
        //   prop->maxGridSize[1] = 0x40000000;
        //   prop->maxGridSize[2] = 0x40000000;
        //   prop->totalConstMem = 0x40000000;
        //   prop->textureAlignment = 0;
        //   //        * TODO: Update the .config and xml files of all GPU config files
        //   //        with new value of sharedMemPerBlock and regsPerBlock
        //   prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
        // #if (CUDART_VERSION > 5050)
        //   prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
        //   prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
        // #endif
        //   prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
        //   prop->regsPerBlock = the_gpu->num_registers_per_block();
        //   prop->warpSize = the_gpu->wrp_size();
        //   prop->clockRate = the_gpu->shader_clock();
        // #if (CUDART_VERSION >= 2010)
        //   prop->multiProcessorCount = the_gpu->get_config().num_shader();
        // #endif
        // #if (CUDART_VERSION >= 4000)
        //   prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
        // #endif
        //   the_gpu->set_prop(prop);
        //   the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
        //   the_device = the_gpgpusim->the_cude_device;
        // }
        // start_sim_thread(1);
        // return the_device;
        // }
    }
}

// CUcontext

// CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
//   // static CUctx_st *the_context = NULL;
//   CUctx_st *the_context = ctx->the_gpgpusim->the_context;
//   if (the_context == NULL) {
//     _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
//     ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
//     the_context = ctx->the_gpgpusim->the_context;
//   }
//   return the_context;
// }

impl std::default::Default for GPGPUContext {
    fn default() -> Self {
        // let gpu = ctx::init();
        // let gpu = GPGPU {};
        // let props = gpu.props();
        Self {
            // next_fat_bin_handle: sync::atomic::AtomicUsize::new(1),
            next_fat_cubin_handle: sync::Mutex::new(1),
            // version_filename: sync::RwLock::new(HashMap::new()),
            // version_filename: HashMap::new(),
            version_filename: sync::RwLock::new(HashMap::new()),
            // ctx: sync::Mutex::new(rt::CUctx_st { gpu }.mut_ptr()),
            // g_global_allfiles_symbol_table = NULL;
            // sm_next_access_uid = 0;
            // warp_inst_sm_next_uid = 0;
            // operand_info_sm_next_uid = 1;
            // kernel_info_m_next_uid = 1;
            // g_num_ptx_inst_uid = 0;
            // g_ptx_cta_info_uid = 1;
            // symbol_sm_next_uid = 1;
            // function_info_sm_next_uid = 1;
            // debug_tensorcore = 0;
            // api = new cuda_runtime_api(this);
            // ptxinfo = new ptxinfo_data(this);
            // ptx_parser = new ptx_recognizer(this);
            // the_gpgpusim = new GPGPUsim_ctx(this);
            // func_sim = new cuda_sim(this);
            // device_runtime = new cuda_device_runtime(this);
            // stats = new ptx_stats(this);
        }
    }
}

// unsafe impl Send for GPGPUContext {}
// unsafe impl Sync for GPGPUContext {}

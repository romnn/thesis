#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(warnings, dead_code)]

mod context;
mod cuda_runtime_api;
mod utils;

use context::GLOBAL_CTX;
use cuda_runtime_api as rt;
use libc;
use std::ffi;
use std::os::raw;
use std::ptr;
use std::sync;

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: *mut ffi::c_void) -> *mut *mut ffi::c_void {
    println!("__cudaRegisterFatBinary({:?})", fatCubin);
    // let ctx = &GLOBAL_CTX;
    // let GLOBAL_CTX = context::GPGPUContext::default();
    // let GLOBAL_CTX = sync::Mutex::new(context::GPGPUContext::default());
    // GLOBAL_CTX.lock().unwrap().cudaRegisterFatBinary(fatCubin)
    GLOBAL_CTX.cudaRegisterFatBinary(fatCubin)
}

// These are vector types derived from the basic integer and floating-point types. They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields x, y, z, and w, respectively. They all come with a constructor function of the form make_<type name>; for example,

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
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
    let hostFun1 = unsafe { ffi::CStr::from_ptr(hostFun).to_string_lossy() };
    let hostFun2 = unsafe { ffi::CStr::from_ptr(hostFun).to_bytes() };
    let deviceFun2 = unsafe { ffi::CStr::from_ptr(deviceFun).to_string_lossy() };
    let deviceName2 = unsafe { ffi::CStr::from_ptr(deviceName).to_string_lossy() };
    println!(
        "__cudaRegisterFunction({:?}, {:?}, {:02X?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?}, {:?})",
        fatCubinHandle,
        hostFun1,
        hostFun2,
        deviceFun2,
        deviceName2,
        thread_limit,
        tid,
        bid,
        bDim,
        gDim,
        wSize
    );
    // let GLOBAL_CTX = sync::Mutex::new(context::GPGPUContext::default());
    // GLOBAL_CTX.lock().unwrap().cudaRegisterFunction(
    GLOBAL_CTX.cudaRegisterFunction(
        fatCubinHandle,
        hostFun,
        deviceFun,
        deviceName,
        thread_limit,
        tid,
        bid,
        bDim,
        gDim,
        wSize,
    )
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(fatCubinHandle: *mut *mut ffi::c_void) {
    println!("__cudaRegisterFatBinaryEnd({:?})", fatCubinHandle);
}

#[no_mangle]
pub extern "C" fn cudaGetErrorString(error: rt::cudaError) {
    // https://michael-f-bryan.github.io/rust-ffi-guide/errors/return_types.html
    println!("cudaGetErrorString({:?})", error);
}

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: *mut ffi::c_void,
    src: *const ffi::c_void,
    count: libc::size_t,
    kind: rt::cudaMemcpyKind,
) -> rt::cudaError {
    println!("cudaMemcpy({:?}, {:?}, {:?}, {:?})", dst, src, count, kind);
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaGetLastError(err: *mut ffi::c_void) -> rt::cudaError {
    println!("cudaGetLastError({:?})", err);
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaFree(devPtr: *mut ffi::c_void) -> rt::cudaError {
    println!("cudaFree({:?})", devPtr);
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaMalloc(devPtr: *mut *mut ffi::c_void, size: libc::size_t) -> rt::cudaError {
    println!("cudaMalloc({:?}, {:?})", devPtr, size);
    rt::cudaError::cudaSuccess
}

#[no_mangle]
// pub extern "C" fn cudaDeviceSynchronize(todo: *mut ffi::c_void) -> rt::cudaError {
pub extern "C" fn cudaDeviceSynchronize() -> rt::cudaError {
    // println!("cudaDeviceSynchronize({:?})", todo);
    println!("cudaDeviceSynchronize()");
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: rt::dim3,
    blockDim: rt::dim3,
    sharedMem: libc::size_t,
    stream: *mut rt::CUstream_st,
) -> libc::c_uint {
    // todo unsigned
    println!(
        "__cudaPushCallConfiguration({:?}, {:?}, {:?}, {:?})",
        gridDim, blockDim, sharedMem, stream
    );
    0
    // rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut rt::dim3,
    blockDim: *mut rt::dim3,
    sharedMem: *mut libc::size_t,
    stream: *mut ffi::c_void,
) -> rt::cudaError {
    let gridDim = unsafe { *gridDim };
    let blockDim = unsafe { *blockDim };
    let sharedMem = unsafe { *sharedMem };
    println!(
        "__cudaPopCallConfiguration({:?}, {:?}, {:?}, {:?})",
        gridDim, blockDim, sharedMem, stream
    );
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    hostFun: *const u8,
    gridDim: rt::dim3,
    blockDim: rt::dim3,
    args: *const *const rt::dim3,
    sharedMem: libc::size_t,
    stream: rt::cudaStream_t,
) -> rt::cudaError {
    println!(
        "cudaLaunchKernel({:?}, {:?}, {:?}, {:?}, {:?}, {:?})",
        hostFun, gridDim, blockDim, args, sharedMem, stream
    );
    rt::cudaError::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: *mut *mut ffi::c_void) {
    println!("__cudaUnregisterFatBinary({:?})", fatCubinHandle);
}

// void __cudaUnregisterFatBinary(void **fatCubinHandle) {
//   if (g_debug_execution >= 3) {
//     announce_call(__my_func__);
//   }
// }

// __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const char *hostFun,
//                                                 dim3 gridDim, dim3 blockDim,
//                                                 const void **args,
//                                                 size_t sharedMem,
//                                                 cudaStream_t stream) {
//   return cudaLaunchKernelInternal(hostFun, gridDim, blockDim, args, sharedMem,
//                                   stream);
// }

// void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
//                                       const char *hostFun, char *deviceFun,
//                                       const char *deviceName, int thread_limit,
//                                       uint3 *tid, uint3 *bid, dim3 *bDim,
//                                       dim3 *gDim) {
//   cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
//                                thread_limit, tid, bid, bDim, gDim);
// }

// cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
//                                                  size_t *sharedMem,
//                                                  void *stream) {
//   if (g_debug_execution >= 3) {
//     announce_call(__my_func__);
//   }
//   return g_last_cudaError = cudaSuccess;
// }

// cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
//   return cudaDeviceSynchronizeInternal();
// }

// __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
//   return cudaMallocInternal(devPtr, size);
// }

// __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
//   if (g_debug_execution >= 3) {
//     announce_call(__my_func__);
//   }
//   // TODO...  manage g_global_mem space?
//   return g_last_cudaError = cudaSuccess;
// }

// __host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
//   if (g_debug_execution >= 3) {
//     announce_call(__my_func__);
//   }
//   return g_last_cudaError;
// }

// __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
//                                           size_t count,
//                                           enum cudaMemcpyKind kind) {
//   return cudaMemcpyInternal(dst, src, count, kind);
// }

// __host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
//   if (g_debug_execution >= 3) {
//     announce_call(__my_func__);
//   }
//   if (g_last_cudaError == cudaSuccess) return "no error";
//   char buf[1024];
//   snprintf(buf, 1024, "<<GPGPU-Sim PTX: there was an error (code = %d)>>",
//            g_last_cudaError);
//   return strdup(buf);
// }

/*
void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaRegisterFatBinaryInternal(fatCubin);
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim) {
  cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
                               thread_limit, tid, bid, bDim, gDim);
}
*/

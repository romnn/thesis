#include "mm.h"
#include <cuda_runtime.h>
/* #include <cudart.h> */
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T> std::vector<T> create_rand_vector(size_t n) {
  std::random_device r;
  std::default_random_engine e(r());
  /* std::uniform_int_distribution<int> uniform_dist(-256, 256); */
  std::uniform_int_distribution<int> uniform_dist(1, 5);

  std::vector<T> vec(n);
  for (size_t i{0}; i < n; ++i) {
    vec.at(i) = static_cast<T>(uniform_dist(e));
  }

  return vec;
}

void mm(size_t m, size_t n, size_t p) {
  std::vector<float> const mat_1_vec{create_rand_vector<float>(m * n)};
  std::vector<float> const mat_2_vec{create_rand_vector<float>(n * p)};
  std::vector<float> mat_3_vec(m * p);
  std::vector<float> mat_4_vec(m * p);

  float const *mat_1{mat_1_vec.data()};
  float const *mat_2{mat_2_vec.data()};
  float *mat_3{mat_3_vec.data()};
  float *mat_4{mat_4_vec.data()};

  /* for (size_t i = 0; i < m; i++) { */
  /*   for (size_t j = 0; j < n; j++) { */
  /*     std::cout << mat_1[i * n + j] << ", "; */
  /*   } */
  /*   std::cout << std::endl; */
  /* } */
  /* std::cout << std::endl; */

  /* for (size_t i = 0; i < n; i++) { */
  /*   for (size_t j = 0; j < p; j++) { */
  /*     std::cout << mat_2[i * p + j] << ", "; */
  /*   } */
  /*   std::cout << std::endl; */
  /* } */
  /* std::cout << std::endl; */

  /* for (size_t i = 0; i < m; i++) { */
  /*   for (size_t j = 0; j < p; j++) { */
  /*     std::cout << mat_3[i * p + j] << ", "; */
  /*   } */
  /*   std::cout << std::endl; */
  /* } */
  /* std::cout << std::endl; */

  /* mm(mat_1, mat_2, mat_3, m, n, p); */
  /* __cudaRegisterFunction(NULL, sizeof(float) * mat_1_vec.size()); */

  /* void **fatCubinHandle, const char *hostFun, */
  /*                                 char *deviceFun, const char *deviceName, */
  /*                                 int thread_limit, uint3 *tid, uint3 *bid, */
  /*                                 dim3 *bDim, dim3 *gDim */

  float *d_mat_1, *d_mat_2, *d_mat_4;

  // Allocate device buffer.
  checkCuda(cudaMalloc(&d_mat_1, sizeof(float) * mat_1_vec.size()));
  checkCuda(cudaMalloc(&d_mat_2, sizeof(float) * mat_2_vec.size()));
  checkCuda(cudaMalloc(&d_mat_4, sizeof(float) * mat_4_vec.size()));

  // Copy data from host to device.
  checkCuda(cudaMemcpy(d_mat_1, mat_1, sizeof(float) * mat_1_vec.size(),
                       cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_mat_2, mat_2, sizeof(float) * mat_2_vec.size(),
                       cudaMemcpyHostToDevice));

  // Run matrix multiplication on GPU.
  mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
              << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Copy data from device to host.
  checkCuda(cudaMemcpy(mat_4, d_mat_4, sizeof(float) * mat_4_vec.size(),
                       cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < p; j++) {
      std::cout << mat_4[i * p + j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Free device buffer.
  checkCuda(cudaFree(d_mat_1));
  checkCuda(cudaFree(d_mat_2));
  checkCuda(cudaFree(d_mat_4));
  /* return allclose<T>(mat_3_vec, mat_4_vec, 1e-4); */
}

int main() {
  size_t m = 3;
  size_t n = 3;
  size_t p = 3;
  mm(m, n, p);
}

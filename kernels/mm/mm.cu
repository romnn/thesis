
#define BLOCK_DIM 32

__global__ void mm_kernel(float const *mat_1, float const *mat_2, float *mat_3,
                          size_t m, size_t n, size_t p) {
  // 2D block and 2D thread
  // Each thread computes one cell in mat_3.
  size_t i{blockIdx.y * blockDim.y + threadIdx.y};
  size_t j{blockIdx.x * blockDim.x + threadIdx.x};

  // Do not process outside the matrix.
  // Do not forget the equal sign!
  if ((i >= m) || (j >= p)) {
    return;
  }

  float acc_sum{0};
  for (size_t k = 0; k < n; k++) {
    acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
  }
  mat_3[i * p + j] = acc_sum;
}

void mm_cuda(float const *mat_1, float const *mat_2, float *mat_3, size_t m,
             size_t n, size_t p) {
  dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                static_cast<double>(threads_per_block.y));
  mm_kernel<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n,
                                                    p);
}

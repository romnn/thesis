#pragma once
#include <cstddef>
/* __global__ void mm_kernel(float const *mat_1, float const *mat_2, float
 * *mat_3, */
/*                           size_t m, size_t n, size_t p); */
void mm_cuda(float const *mat_1, float const *mat_2, float *mat_3, size_t m,
             size_t n, size_t p);

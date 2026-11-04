// Eight stores per thread pass over a buffer inside the L2, the same words again each launch.
__global__ void __launch_bounds__(256) k_rewrite(unsigned* a) {
  const size_t base = (size_t)blockIdx.x * 1024 + threadIdx.x * 4;
  for (int k = 0; k < 32; ++k)
    for (int j = 0; j < 4; ++j) a[base + (size_t)k * 4096 + j] = k + j;
}

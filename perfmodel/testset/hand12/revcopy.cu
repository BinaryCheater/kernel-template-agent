// Copy reversed past the L2: lanes descend in the written buffer.
__global__ void __launch_bounds__(256) k_revcopy(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  const size_t n = (size_t)gridDim.x * 256 * 64;
  for (int k = 0; k < 64; ++k) {
    size_t i = base + (size_t)k * 256;
    a[n - 1 - i] = b[i];
  }
}

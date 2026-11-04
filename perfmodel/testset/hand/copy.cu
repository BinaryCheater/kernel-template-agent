// Copy one array to another past the L2.
__global__ void __launch_bounds__(256) k_copy(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) {
    size_t i = base + (size_t)k * 256;
    a[i] = b[i];
  }
}

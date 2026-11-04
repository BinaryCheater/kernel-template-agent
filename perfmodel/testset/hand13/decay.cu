// Scale 12 MiB in place, two passes: every line read, then written.
__global__ void __launch_bounds__(256) k_decay(float* a) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int p = 0; p < 2; ++p)
    for (int k = 0; k < 16; ++k) {
      size_t i = base + (size_t)k * 256;
      a[i] = a[i] * 0.99f;
    }
}

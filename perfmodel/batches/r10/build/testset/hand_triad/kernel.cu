// STREAM triad, one element per thread pass, grid over the array.
extern "C" __global__ void __launch_bounds__(256) k_triad(float* a, const float* b, const float* c) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) {
    size_t i = base + (size_t)k * 256;
    a[i] = b[i] + 0.5f * c[i];
  }
}

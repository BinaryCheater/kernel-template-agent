// STREAM scale over 64 MiB.
extern "C" __global__ void __launch_bounds__(256) k_scale(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) {
    size_t i = base + (size_t)k * 256;
    a[i] = 3.f * b[i];
  }
}

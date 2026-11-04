// Three fields of a 64-byte record per thread: strided stores into parts of sectors.
__global__ void __launch_bounds__(256) k_aos(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    size_t i = base + (size_t)k * 256;
    float v = b[i];
    a[i * 16] = v;
    a[i * 16 + 1] = v * 2.f;
    a[i * 16 + 2] = v + 1.f;
  }
}

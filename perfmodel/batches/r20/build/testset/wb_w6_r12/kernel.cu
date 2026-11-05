// round 20: 12 MiB read once, 6 MiB stored once, both in order, as many blocks as the larger
extern "C" __global__ void __launch_bounds__(256) k_wb(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 64; ++k) {
    const size_t i = base + (size_t)k * 256;
    if (i < 3145728) s += b[i];
    if (i < 1572864) a[i] = s;
  }
}

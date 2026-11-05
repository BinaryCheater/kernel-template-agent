// round 20: 9 MiB read once, 9 MiB stored once, both in order, as many blocks as the larger
extern "C" __global__ void __launch_bounds__(256) k_wb(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 4 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 4; ++k) {
    const size_t i = base + (size_t)k * 256;
    if (i < 2359296) s += b[i];
    if (i < 2359296) a[i] = s;
  }
}

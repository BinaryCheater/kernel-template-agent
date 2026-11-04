// Random reads past the L2 into a small contiguous output.
extern "C" __global__ void __launch_bounds__(256) k_gather(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 16; ++k) {
    unsigned h = (unsigned)(base + (size_t)k * 256) * 2654435761u;
    s += b[(h >> 7) & 8388607u];
    a[base + (size_t)k * 256] = s;
  }
}

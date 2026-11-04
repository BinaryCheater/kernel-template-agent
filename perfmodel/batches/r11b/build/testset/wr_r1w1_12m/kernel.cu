// round 11: 1 reads and 1 stores per pass over 12 MiB of elements
extern "C" __global__ void __launch_bounds__(256) k_wr(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 64; ++k) {
    size_t i = base + (size_t)k * 256;
    s += b[i * 1 + 0];
    a[i * 1 + 0] = s;
  }
}

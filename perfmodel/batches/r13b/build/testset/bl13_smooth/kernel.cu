// A 1-2-1 filter in place, four passes over 4 MiB in one launch.
extern "C" __global__ void __launch_bounds__(256) k_smooth(float* y) {
  const unsigned base = blockIdx.x * 256 * 8 + threadIdx.x;
  for (int p = 0; p < 4; ++p)
    for (int k = 0; k < 8; ++k) {
      unsigned i = base + k * 256;
      float l = y[(i + 1048575u) & 1048575u];
      float r = y[(i + 1u) & 1048575u];
      y[i] = 0.25f * l + 0.5f * y[i] + 0.25f * r;
    }
}

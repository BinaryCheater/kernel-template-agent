// Row ends clamped: min and max in the index.
extern "C" __global__ void __launch_bounds__(128) k_clampread(unsigned* o, const unsigned* a) {
  const int t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (int k = -8; k < 8; ++k) {
    const int i = t * 4 + k * 32;
    s += a[i < 0 ? 0 : (i > 1048575 ? 1048575 : i)];
  }
  o[t] = s;
}

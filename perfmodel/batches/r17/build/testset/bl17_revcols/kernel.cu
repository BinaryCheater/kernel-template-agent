// Rows in order, columns in reverse.
extern "C" __global__ void __launch_bounds__(128) k_revcols(float* o, const float* a) {
  const unsigned t = threadIdx.x, y = blockIdx.x;
  float s = 1;
  for (unsigned k = 0; k < 6; ++k) s *= a[y * 1024 + 1023 - t * 6 - k];
  o[y * 128 + t] = s;
}

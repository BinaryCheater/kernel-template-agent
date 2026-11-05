// Rows in reverse order, columns in order.
__global__ void __launch_bounds__(128) k_revrows(float* o, const float* a) {
  const unsigned t = threadIdx.x, y = blockIdx.x;
  float s = 0;
  for (unsigned k = 0; k < 8; ++k) s += a[(1023 - y) * 2048 + t * 8 + k];
  o[y * 128 + t] = s;
}

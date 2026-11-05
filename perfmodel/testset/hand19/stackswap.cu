// Six loads on the stack, read back rotated by a loaded word.
__global__ void __launch_bounds__(128) k_stackswap(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[8];
  for (unsigned k = 0; k < 8; ++k) w[k] = a[(t + k * 16) & 262143u];
  const unsigned r = w[0] & 7u;
  o[t] = w[r] + w[(r + 3) & 7u];
}

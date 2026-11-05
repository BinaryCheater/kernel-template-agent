// A table of indices, then the value they name.
__global__ void __launch_bounds__(256) k_twolevel(unsigned* o, const unsigned* a, const unsigned* b) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) s += b[a[(t + k) & 262143u] & 1048575u];
  o[t] = s;
}

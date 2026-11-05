// Loads at an index divided by a constant that is no power of two.
__global__ void __launch_bounds__(256) k_divseven(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) s += a[(t * 37 + k * 7) / 7];
  o[t & 65535u] = s;
}

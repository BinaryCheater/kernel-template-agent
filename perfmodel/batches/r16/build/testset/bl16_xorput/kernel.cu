// Stores at an xorshift of the thread.
extern "C" __global__ void __launch_bounds__(256) k_xorput(unsigned* o) {
  unsigned x = blockIdx.x * 256 + threadIdx.x + 1;
  for (unsigned k = 0; k < 4; ++k) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    o[x & 2097151u] = k;
  }
}

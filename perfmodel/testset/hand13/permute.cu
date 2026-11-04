// Apply a permutation read from memory: stores at loaded addresses.
__global__ void __launch_bounds__(256) k_permute(unsigned* o, const unsigned* p, const unsigned* a) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    size_t i = base + (size_t)k * 256;
    o[p[i] & 4194303u] = a[i];
  }
}

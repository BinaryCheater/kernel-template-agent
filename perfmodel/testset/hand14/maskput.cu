// As hashput, half of the lanes store.
__global__ void __launch_bounds__(256) k_maskput(unsigned* a, const unsigned* key) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = key[(t * 8 + k) & 1048575u];
    if (v & 1u) a[(v * 2654435761u >> 9) & 2097151u] = t;
  }
}

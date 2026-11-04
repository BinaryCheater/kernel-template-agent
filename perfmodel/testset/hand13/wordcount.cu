// Count hashed keys from an input stream in 1024 bins.
__global__ void __launch_bounds__(256) k_wordcount(unsigned* h, const unsigned* w) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = w[(t * 8 + k) & 1048575u];
    atomicAdd(&h[((v * 2654435761u) >> 12) & 1023u], 1u);
  }
}

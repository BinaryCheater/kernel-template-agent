// Each block keeps its own tally of 1024 bins, updated without atomics.
__global__ void __launch_bounds__(256) k_blockcount(unsigned* c, const unsigned* key) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = key[(t * 8 + k) & 1048575u];
    c[blockIdx.x * 1024u + (v & 1023u)] += 1u;
  }
}

// A running minimum per block group: all of a block's lanes meet on one word.
__global__ void __launch_bounds__(256) k_blockmin(unsigned* m, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = a[(t * 8 + k) & 1048575u];
    atomicMin(&m[blockIdx.x & 127u], v);
  }
}

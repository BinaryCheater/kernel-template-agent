// A running minimum per block group: all of a block's lanes meet on one word.
extern "C" __global__ void __launch_bounds__(256) k_blockmin(unsigned* m, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = (t + k) * 2654435761u;
    atomicMin(&m[blockIdx.x & 127u], v);
  }
}

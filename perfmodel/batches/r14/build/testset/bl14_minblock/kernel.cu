// All lanes of a block keep the smallest loaded word in their block's slot.
extern "C" __global__ void __launch_bounds__(256) k_minblock(unsigned* m, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) atomicMin(&m[blockIdx.x & 127u], a[(t * 8 + k) & 1048575u]);
}

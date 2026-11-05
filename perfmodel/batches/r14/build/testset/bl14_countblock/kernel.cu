// All lanes of a block add a loaded value to their block's word.
extern "C" __global__ void __launch_bounds__(256) k_countblock(unsigned* c, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&c[blockIdx.x & 63u], a[(t * 8 + k) & 1048575u] & 15u);
}

// round 13: all lanes of a block add to one word of 1, the same word every time
extern "C" __global__ void __launch_bounds__(256) k_atb(unsigned* h) {
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&h[blockIdx.x & 0u], 1u);
}

// round 13: all lanes of a block on one word of 1024, a new word every 2 atomics
extern "C" __global__ void __launch_bounds__(256) k_atk(unsigned* h) {
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&h[(blockIdx.x * 37u + k / 2u * 101u) & 1023u], 1u);
}

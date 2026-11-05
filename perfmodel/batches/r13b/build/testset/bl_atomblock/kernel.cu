// Every thread of a block adds to one of 64 words: contention by block, not by hash.
extern "C" __global__ void __launch_bounds__(256) k_atomblock(unsigned* h) {
  for (int k = 0; k < 8; ++k) atomicAdd(&h[blockIdx.x & 63], 1u);
}

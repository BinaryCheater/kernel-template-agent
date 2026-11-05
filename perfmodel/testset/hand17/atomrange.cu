// Counters in a range of 1024 words per block, four per thread.
__global__ void __launch_bounds__(128) k_atomrange(unsigned* o) {
  const unsigned t = threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) atomicAdd(&o[blockIdx.x * 1024 + (t * 37 + k * 11) % 1024u], 1u);
}

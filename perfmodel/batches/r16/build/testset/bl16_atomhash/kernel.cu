// Counters at a hashed word per lane.
extern "C" __global__ void __launch_bounds__(256) k_atomhash(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) atomicAdd(&o[(t * 2654435761u + k * 40503u) >> 12], 1u);
}

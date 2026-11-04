// Atomics walking a table one line apart: each lane its own line, no hash.
extern "C" __global__ void __launch_bounds__(256) k_atomline(unsigned* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k) atomicAdd(&h[((t + k * 7u) & 16383u) * 32u], 1u);
}

// round 12b: 32 lanes on one word of 64
extern "C" __global__ void __launch_bounds__(256) k_atw(unsigned* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k)
    atomicAdd(&h[((t / 32 + k * 4099u) * 2654435761u >> 7) & 63u], 1u);
}

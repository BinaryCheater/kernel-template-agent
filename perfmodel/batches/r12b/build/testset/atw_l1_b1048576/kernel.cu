// round 12b: 1 lanes on one word of 1048576
extern "C" __global__ void __launch_bounds__(256) k_atw(unsigned* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k)
    atomicAdd(&h[((t / 1 + k * 4099u) * 2654435761u >> 7) & 1048575u], 1u);
}

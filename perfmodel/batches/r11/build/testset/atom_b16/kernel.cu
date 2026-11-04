// round 11: 16 atomics per thread on 16 bins
extern "C" __global__ void __launch_bounds__(256) k_atom(unsigned* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k)
    atomicAdd(&h[((t + k * 1048576u) * 2654435761u >> 11) & 15u], 1u);
}

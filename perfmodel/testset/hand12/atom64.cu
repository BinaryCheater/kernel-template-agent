// 64-bit atomic adds on hashed bins.
__global__ void __launch_bounds__(256) k_atom64(unsigned long long* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k) atomicAdd(&h[((t + k * 65536u) * 2654435761u >> 12) & 8191u], 1ull);
}

// Running maximum over hashed bins.
__global__ void __launch_bounds__(256) k_atommax(unsigned* h) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k) atomicMax(&h[((t + k * 65536u) * 2654435761u >> 12) & 4095u], t + k);
}

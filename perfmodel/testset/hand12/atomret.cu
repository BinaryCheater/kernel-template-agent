// Fetch-add with the old value used: tickets into an output.
__global__ void __launch_bounds__(256) k_atomret(unsigned* h, unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (int k = 0; k < 8; ++k) s += atomicAdd(&h[(t * 2654435761u >> 18) & 1023u], k);
  a[t] = s;
}

// Stores in bit-reversed order over 8 MiB.
__global__ void __launch_bounds__(256) k_brevfill(unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) a[__brev(t * 8 + k) >> 11] = t;
}

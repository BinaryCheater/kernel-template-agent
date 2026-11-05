// Stores 128 bytes apart over a table past the dirty limit.
extern "C" __global__ void __launch_bounds__(256) k_stridefill(unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) a[((t * 8 + k) * 32u) & 8388607u] = t;
}

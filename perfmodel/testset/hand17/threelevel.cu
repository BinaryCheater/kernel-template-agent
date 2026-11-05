// Two tables of indices, then the value.
__global__ void __launch_bounds__(128) k_threelevel(unsigned* o, const unsigned* a, const unsigned* b, const unsigned* c) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = c[b[a[t & 262143u] & 262143u] & 1048575u];
}

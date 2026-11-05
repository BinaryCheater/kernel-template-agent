// A prefix sum of ten loads on the stack, read back at an index from the data.
__global__ void __launch_bounds__(128) k_prefixstack(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[10];
  unsigned s = 0;
  for (unsigned k = 0; k < 10; ++k) { s += a[(t + k) & 262143u]; w[k] = s; }
  o[t] = w[s & 7u];
}

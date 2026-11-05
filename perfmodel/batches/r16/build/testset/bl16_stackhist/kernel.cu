// A histogram of 32 bins on the stack, per thread.
extern "C" __global__ void __launch_bounds__(128) k_stackhist(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned h[32];
  for (unsigned k = 0; k < 32; ++k) h[k] = 0;
  for (unsigned k = 0; k < 32; ++k) h[a[(t * 3 + k) & 262143u] & 31u] += 1;
  o[t] = h[t & 31u];
}

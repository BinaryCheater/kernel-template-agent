// A running prefix over eight loads on the stack, one read back by a loaded word.
extern "C" __global__ void __launch_bounds__(128) k_stackfive(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[8];
  w[0] = a[t & 262143u];
  for (unsigned k = 1; k < 8; ++k) w[k] = w[k - 1] + a[(t + k * 32) & 262143u];
  o[t] = w[w[3] & 7u];
}

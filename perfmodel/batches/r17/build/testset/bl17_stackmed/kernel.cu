// Eight loads sorted on the stack by insertion, per thread.
extern "C" __global__ void __launch_bounds__(256) k_stackmed(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned w[8];
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = a[(t * 3 + k) & 262143u];
    int j = (int)k - 1;
    while (j >= 0 && w[j] > v) { w[j + 1] = w[j]; --j; }
    w[j + 1] = v;
  }
  o[t] = w[4];
}

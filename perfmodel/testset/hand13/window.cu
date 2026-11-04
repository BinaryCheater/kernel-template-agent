// A running hash over the last 32 loaded words, kept in a ring on the stack.
__global__ void __launch_bounds__(256) k_window(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[32];
  for (int i = 0; i < 32; ++i) r[i] = i;
  unsigned s = t;
  for (unsigned i = 0; i < 64; ++i) {
    unsigned v = a[(t * 64 + i) & 1048575u];
    r[(i * 5) & 31u] = v ^ s;
    s += r[(i * 3 + t) & 31u];
  }
  o[t] = s;
}

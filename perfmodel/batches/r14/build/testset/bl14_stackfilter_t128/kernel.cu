// A 16-word delay line on the stack, a different tap per lane.
extern "C" __global__ void __launch_bounds__(128) k_stackfilter(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned r[16];
  for (int i = 0; i < 16; ++i) r[i] = i;
  unsigned s = 0;
  for (unsigned i = 0; i < 64; ++i) {
    unsigned v = a[(t * 64 + i) & 1048575u];
    r[i & 15u] = v;
    s += r[(i + t) & 15u];
  }
  o[t] = s;
}

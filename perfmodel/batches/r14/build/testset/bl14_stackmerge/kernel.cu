// Two runs of loaded words on the stack, merged by index per lane.
extern "C" __global__ void __launch_bounds__(256) k_stackmerge(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned p[32], q[32];
  for (int i = 0; i < 32; ++i) {
    p[i] = a[(t * 64 + i) & 1048575u];
    q[i] = a[(t * 64 + 32 + i) & 1048575u];
  }
  int x = 0, y = 0;
  unsigned s = 0;
  for (int i = 0; i < 32; ++i) {
    if (p[x & 31] < q[y & 31]) s += p[x++ & 31];
    else s ^= q[y++ & 31];
  }
  o[t] = s + x;
}

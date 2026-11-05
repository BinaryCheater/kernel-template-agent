// 64 loaded words on the stack, read back at an index that depends on the data.
extern "C" __global__ void __launch_bounds__(128) k_stackscan(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned r[64];
  for (int i = 0; i < 64; ++i) r[i] = a[(t * 64 + i) & 1048575u];
  unsigned s = t;
  for (int i = 0; i < 64; ++i) s += r[(s + i) & 63u];
  o[t] = s;
}

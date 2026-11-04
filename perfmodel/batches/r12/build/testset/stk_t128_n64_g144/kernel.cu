// round 12: 64 words of stack per thread, stored and loaded four passes
extern "C" __global__ void __launch_bounds__(128) k_stk(unsigned* sink) {
  volatile unsigned a[64];
  unsigned s = threadIdx.x;
  for (int i = 0; i < 64; i++) a[i] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < 64; i++) {
      a[(i * 5 + threadIdx.x) & 63] = s + i;
      s ^= a[(i * 3 + r) & 63];
    }
  if (s == 12345) sink[0] = 1;
}

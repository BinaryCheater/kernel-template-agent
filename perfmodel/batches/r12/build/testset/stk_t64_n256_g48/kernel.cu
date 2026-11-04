// round 12: 256 words of stack per thread, stored and loaded four passes
extern "C" __global__ void __launch_bounds__(64) k_stk(unsigned* sink) {
  volatile unsigned a[256];
  unsigned s = threadIdx.x;
  for (int i = 0; i < 256; i++) a[i] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < 256; i++) {
      a[(i * 5 + threadIdx.x) & 255] = s + i;
      s ^= a[(i * 3 + r) & 255];
    }
  if (s == 12345) sink[0] = 1;
}

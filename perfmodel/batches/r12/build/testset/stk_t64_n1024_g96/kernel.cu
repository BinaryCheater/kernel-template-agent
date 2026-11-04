// round 12: 1024 words of stack per thread, stored and loaded four passes
extern "C" __global__ void __launch_bounds__(64) k_stk(unsigned* sink) {
  volatile unsigned a[1024];
  unsigned s = threadIdx.x;
  for (int i = 0; i < 1024; i++) a[i] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < 1024; i++) {
      a[(i * 5 + threadIdx.x) & 1023] = s + i;
      s ^= a[(i * 3 + r) & 1023];
    }
  if (s == 12345) sink[0] = 1;
}

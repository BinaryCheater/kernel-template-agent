// round 12b: 256 words of stack per thread, one word per pass, all lanes alike
extern "C" __global__ void __launch_bounds__(64) k_stu(unsigned* sink, const unsigned* cfg) {
  volatile unsigned a[256];
  unsigned s = threadIdx.x, z = cfg[0] & 0;
  for (int i = 0; i < 256; i++) a[(i + z) & 255] = s;
  for (int r = 0; r < 4; r++)
    for (int i = 0; i < 256; i++) {
      a[(i * 5 + z) & 255] = s + i;
      s ^= a[(i * 3 + r + z) & 255];
    }
  if (s == 12345) sink[0] = 1;
}

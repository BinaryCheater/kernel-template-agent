// round 16: dram_burst's kernel: 48 blocks in step, 8 MiB, cs
extern "C" __global__ void __launch_bounds__(1024) k_mem(int* o, const int4* a) {
  const int4* p = a + blockIdx.x * 1024 + threadIdx.x;
  int s = 0;
  for (int i = 0; i < 10; i++) {
    const int4 v = __ldcs(p + (size_t)i * 48 * 1024);
    s ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (s == 0x7fff1234) o[0] = s;
}

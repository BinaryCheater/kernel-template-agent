// round 17: block turnover: compute for about 2 us, 48 blocks of 128, many on an SM
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  unsigned s = threadIdx.x;
  for (unsigned k = 0; k < 400; ++k) { s = s * 3u + k; asm volatile("" : "+r"(s)); }
  if (s == 0x7fff1234u) o[0] = s;
}

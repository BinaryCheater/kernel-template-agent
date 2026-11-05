// Four neighbouring words summed, the range starting at a stride of 7 words.
__global__ void __launch_bounds__(128) k_rangesum(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) s += a[t * 7 + k];
  o[t & 65535u] = s;
}

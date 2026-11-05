// Bins by the count of bits and the position of the top bit.
extern "C" __global__ void __launch_bounds__(256) k_popindex(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const unsigned x = t * 2654435761u + k;
    s += a[(__popc(x) << 8) + (31 - __clz(x | 1)) * 16 + k];
  }
  o[t] = s;
}

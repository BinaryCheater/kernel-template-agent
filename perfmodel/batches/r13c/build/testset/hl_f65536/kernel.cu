// round 13: 2 M hashed loads over 65536 KiB, one sum per thread stored
extern "C" __global__ void __launch_bounds__(256) k_hl(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    s += a[(x >> 7) & 16777215u];
  }
  o[t] = s;
}

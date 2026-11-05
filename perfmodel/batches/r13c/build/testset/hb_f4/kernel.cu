// round 13: 2 M hashed stores, each block over its own 4 KiB
extern "C" __global__ void __launch_bounds__(256) k_hb(unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    a[blockIdx.x * 1024u + ((x >> 7) & 1023u)] = t;
  }
}

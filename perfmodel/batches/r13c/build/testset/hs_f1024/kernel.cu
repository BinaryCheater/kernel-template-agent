// round 13: 2 M hashed stores over 1024 KiB
extern "C" __global__ void __launch_bounds__(256) k_hs(unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    a[(x >> 7) & 262143u] = t;
  }
}

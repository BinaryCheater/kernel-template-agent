// round 12b: fetch-add on 1024 bins, the old value used
extern "C" __global__ void __launch_bounds__(256) k_atr(unsigned* h, unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (int k = 0; k < 8; ++k) s += atomicAdd(&h[((t + k * 4099u) * 2654435761u >> 7) & 1023u], k);
  a[t] = s;
}

// round 16: 7 MiB written, read once, by threads
extern "C" __global__ void __launch_bounds__(32) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 32 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    ((uint2*)o)[t * 8 + k] = a[t * 8 + k];
  }
}

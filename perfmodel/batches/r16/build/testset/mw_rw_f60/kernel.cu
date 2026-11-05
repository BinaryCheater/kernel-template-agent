// round 16: 6.0 MiB written, as much read once
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    ((uint2*)o)[t * 8 + k] = a[t * 8 + k];
  }
}

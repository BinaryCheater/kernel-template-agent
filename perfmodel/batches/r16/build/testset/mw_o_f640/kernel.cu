// round 16: 64 MiB written
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    ((uint2*)o)[t * 8 + k] = make_uint2(t, k);
  }
}

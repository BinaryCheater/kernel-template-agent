// round 16: 8-byte lanes 32 bytes apart, sectors filled
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) {
    ((uint2*)o)[t * 4 + k] = make_uint2(t, k);
  }
}

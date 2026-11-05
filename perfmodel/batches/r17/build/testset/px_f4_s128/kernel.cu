// round 16: 8-byte lanes 128 bytes apart read and written in place
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 16; ++k) {
    ((uint2*)o)[t * 16 + k].x += 1u;
  }
}

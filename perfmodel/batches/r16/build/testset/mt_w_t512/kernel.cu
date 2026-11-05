// round 16: rows warm, stores
extern "C" __global__ void __launch_bounds__(512) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 512 + threadIdx.x;
  for (unsigned k = 0; k < 1; ++k) {
    ((uint2*)o)[blockIdx.x * 512 + threadIdx.x * 1 + k] = make_uint2(t, k);
  }
}

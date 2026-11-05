// round 16: rows warm, loads and a barrier
extern "C" __global__ void __launch_bounds__(256) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 2; ++k) {
    const uint2 x = a[blockIdx.x * 512 + threadIdx.x * 2 + k];
    s += x.x ^ x.y;
  }
  __syncthreads();
  o[blockIdx.x & 65535u] = s;
}

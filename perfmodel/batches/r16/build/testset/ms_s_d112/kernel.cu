// round 16: shared memory, 16 bytes each thread, 112 bytes apart: stores, barrier, loads
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  __shared__ uint4 v[896];
  const unsigned t = threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    v[t * 7] = make_uint4(t, k, t, k);
    __syncthreads();
    const uint4 x = v[((t + 1) & 127u) * 7];
    s += x.x;
    __syncthreads();
  }
  o[blockIdx.x] = s;
}

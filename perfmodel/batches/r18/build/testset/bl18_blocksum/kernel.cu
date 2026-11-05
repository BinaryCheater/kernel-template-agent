// A block's words summed into one word per eight blocks.
extern "C" __global__ void __launch_bounds__(128) k_blocksum(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  atomicAdd(&o[blockIdx.x >> 3], a[t & 1048575u]);
}

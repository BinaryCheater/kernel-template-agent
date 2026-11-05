// The block's minimum into one word per block.
__global__ void __launch_bounds__(128) k_blockmin(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  atomicMin(&o[blockIdx.x], a[t & 1048575u]);
}

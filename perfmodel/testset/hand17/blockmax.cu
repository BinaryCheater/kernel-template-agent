// The block's maximum of a loaded word, one word per sixteen blocks.
__global__ void __launch_bounds__(256) k_blockmax(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  atomicMax(&o[blockIdx.x >> 4], a[(t * 5) & 1048575u]);
}

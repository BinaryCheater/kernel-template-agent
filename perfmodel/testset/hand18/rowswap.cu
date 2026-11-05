// Each row of 1024 words written to the row after it.
__global__ void __launch_bounds__(256) k_rowswap(unsigned* o, const unsigned* a) {
  const unsigned y = blockIdx.x / 4, x = (blockIdx.x % 4) * 256 + threadIdx.x;
  o[(y + 1) * 1024 + x] = a[y * 1024 + x];
}

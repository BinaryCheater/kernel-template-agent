// Rows of 2048 words read forwards and backwards in turn.
extern "C" __global__ void __launch_bounds__(128) k_zigzag(unsigned* o, const unsigned* a) {
  const unsigned y = blockIdx.x / 16, x = (blockIdx.x % 16) * 128 + threadIdx.x;
  const unsigned i = (y & 1u) ? 2047u - x : x;
  o[blockIdx.x & 65535u] = a[y * 2048 + i];
}

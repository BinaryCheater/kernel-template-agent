// Chunks of 12 words per thread, 7 threads to a row: the bytes per lane are not a power of two.
extern "C" __global__ void __launch_bounds__(128) k_chunked(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const unsigned row = t / 7, col = t % 7;
  for (unsigned k = 0; k < 6; ++k) o[row * 84 + col * 12 + k] = a[row * 84 + col * 12 + k] ^ t;
}

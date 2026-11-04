// Two arrays zipped into pairs: each lane stores two neighbouring words.
__global__ void __launch_bounds__(256) k_interleave(float* o, const float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    size_t i = base + (size_t)k * 256;
    o[i * 2] = a[i];
    o[i * 2 + 1] = b[i];
  }
}

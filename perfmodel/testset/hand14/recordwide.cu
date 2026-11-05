// Every thread writes the 32 words of its own 128-byte record.
__global__ void __launch_bounds__(256) k_recordwide(unsigned* a) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 32; ++k) a[r * 32 + k] = (unsigned)r ^ k;
}

// Every thread writes its own 32-byte record, field by field.
__global__ void __launch_bounds__(256) k_recordinit(unsigned* a) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k) a[r * 8 + k] = (unsigned)r + k;
}

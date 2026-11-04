// Every thread writes its own 64-byte record, field by field.
__global__ void __launch_bounds__(256) k_recordfill(unsigned* a) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 16; ++k) a[r * 16 + k] = (unsigned)r + k;
}

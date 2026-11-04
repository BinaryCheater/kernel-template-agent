// Out-of-place transpose of a 1024 x 1024 matrix: rows read in order, columns stored apart.
__global__ void __launch_bounds__(256) k_transpose(float* b, const float* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k) {
    unsigned i = t * 8 + k;
    b[(i & 1023u) * 1024u + (i >> 10)] = a[i];
  }
}

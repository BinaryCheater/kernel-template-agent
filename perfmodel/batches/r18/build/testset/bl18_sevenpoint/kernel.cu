// A seven-point stencil over planes of 64 by 64 words.
extern "C" __global__ void __launch_bounds__(64) k_sevenpoint(float* o, const float* a) {
  const unsigned z = blockIdx.z + 1, y = blockIdx.y + 1, x = threadIdx.x;
  const unsigned i = z * 4096 + y * 64 + x;
  o[i] = a[i] + a[i - 4096] + a[i + 4096] + a[i - 64] + a[i + 64] + a[i + (x < 63 ? 1 : 0)] + a[i - (x > 0 ? 1 : 0)];
}

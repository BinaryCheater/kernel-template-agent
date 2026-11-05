// A five-point stencil over rows of 4096 words.
extern "C" __global__ void __launch_bounds__(128) k_fivepoint(float* o, const float* a) {
  const unsigned W = 4096, y = blockIdx.x / 32 + 1, x = (blockIdx.x % 32) * 128 + threadIdx.x;
  o[y * W + x] = a[y * W + x] + a[(y - 1) * W + x] + a[(y + 1) * W + x] + a[y * W + x + 1] + a[y * W + x - 1];
}

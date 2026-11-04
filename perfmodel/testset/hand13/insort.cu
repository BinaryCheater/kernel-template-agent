// Median of sixteen loaded values by insertion sort on the thread's stack.
__global__ void __launch_bounds__(128) k_insort(float* o, const float* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  float w[16];
  for (int i = 0; i < 16; ++i) {
    float v = a[(t * 16 + i) & 1048575u];
    int j = i;
    while (j > 0 && w[j - 1] > v) {
      w[j] = w[j - 1];
      --j;
    }
    w[j] = v;
  }
  o[t] = w[8];
}

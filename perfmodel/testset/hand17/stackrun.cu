// Running sums in a frame of 40 words, read back in reverse.
__global__ void __launch_bounds__(128) k_stackrun(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[40];
  w[0] = a[t & 262143u];
  for (unsigned k = 1; k < 40; ++k) w[k] = w[k - 1] + k;
  unsigned s = 0;
  for (unsigned k = 0; k < 40; ++k) s ^= w[(39 - k + t) % 40u];
  o[t] = s;
}

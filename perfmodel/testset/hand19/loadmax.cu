// The largest of six loads 64 words apart.
__global__ void __launch_bounds__(128) k_loadmax(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 6; ++k) {
    const unsigned v = a[(t + k * 64) & 262143u];
    s = v > s ? v : s;
  }
  o[t] = s;
}

// Sixty-four-bit words read, added, stored.
__global__ void __launch_bounds__(128) k_doubleword(unsigned long long* o, const unsigned long long* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[t & 131071u] + a[(t + 1u) & 131071u];
}

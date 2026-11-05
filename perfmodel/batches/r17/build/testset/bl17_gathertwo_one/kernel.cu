// Two pairs of offsets from two loaded bases.
extern "C" __global__ void __launch_bounds__(128) k_gathertwo(unsigned* o, const unsigned* a, const unsigned* b) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const unsigned i = a[t & 262143u] & 524287u, j = a[(t + 512) & 262143u] & 524287u;
  o[t] = b[i] + b[j];
}

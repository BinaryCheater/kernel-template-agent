// Four offsets from one loaded base.
extern "C" __global__ void __launch_bounds__(256) k_gatherfour(unsigned* o, const unsigned* a, const unsigned* b) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  const unsigned i = a[t & 262143u] & 524287u;
  o[t] = b[i] + b[i + 1024];
}

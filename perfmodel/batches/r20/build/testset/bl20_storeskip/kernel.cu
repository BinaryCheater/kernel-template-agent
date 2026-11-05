// Stores eight bytes apart.
extern "C" __global__ void __launch_bounds__(128) k_storeskip(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t * 2] = a[t & 262143u] + 1u;
}

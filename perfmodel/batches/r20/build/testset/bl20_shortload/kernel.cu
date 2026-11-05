// Sixteen-bit words widened and stored.
extern "C" __global__ void __launch_bounds__(128) k_shortload(unsigned* o, const unsigned short* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = (unsigned)a[t & 524287u] * 7u;
}

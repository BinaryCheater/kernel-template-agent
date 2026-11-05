// Odd lanes load, even lanes keep the thread index.
extern "C" __global__ void __launch_bounds__(128) k_maskload(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = t;
  if (t & 1u) s = a[t];
  o[t] = s + 1u;
}

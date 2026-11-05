// One lane of sixteen loads, the others keep the thread index.
__global__ void __launch_bounds__(128) k_everysixteen(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = t;
  if ((t & 15u) == 0u) s = a[t >> 4];
  o[t] = s;
}

// Each pair of lanes reads the other lane's word of a pair.
extern "C" __global__ void __launch_bounds__(128) k_evenodd(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[(t & ~1u) & 262143u];
}

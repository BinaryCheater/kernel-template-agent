// Two tallies of 1024 bins, one by thread, one by a word loaded.
extern "C" __global__ void __launch_bounds__(128) k_atomtwo(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  atomicAdd(&o[t & 1023u], 1u);
  atomicAdd(&o[1024 + ((t >> 3) & 1023u)], a[t & 262143u]);
}

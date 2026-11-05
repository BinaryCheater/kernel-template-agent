// Two stores a thread, the thread index and a word loaded.
__global__ void __launch_bounds__(128) k_storepair(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t * 2 + 1] = a[t & 262143u];
  o[t * 2] = t;
}

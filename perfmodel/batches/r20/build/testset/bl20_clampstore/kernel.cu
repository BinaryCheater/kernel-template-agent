// A store only where the word loaded passes a limit.
extern "C" __global__ void __launch_bounds__(128) k_clampstore(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const unsigned v = a[t & 262143u];
  if (v & 256u) o[t] = v & 255u;
}

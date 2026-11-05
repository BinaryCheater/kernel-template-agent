// Three loads a row of a 4 MiB ring, eight words between lanes.
extern "C" __global__ void __launch_bounds__(128) k_ringload(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 3; ++k) s ^= a[(blockIdx.x * 1024 + threadIdx.x * 8 + k) & 1048575u];
  o[t] = s;
}

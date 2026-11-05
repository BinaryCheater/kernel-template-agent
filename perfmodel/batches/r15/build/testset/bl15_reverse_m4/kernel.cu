// A negative stride from the end of a table.
extern "C" __global__ void __launch_bounds__(256) k_reverse(unsigned* o, const unsigned* a) {
  const int t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (int k = 0; k < 16; ++k) s += a[1048575 - t * 16 - k];
  o[1048575 - t] = s;
}

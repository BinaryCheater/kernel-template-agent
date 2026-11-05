// A 64-bit product in the index: past 4 GiB of index space, wrapped to the table.
extern "C" __global__ void __launch_bounds__(256) k_wideindex(unsigned* o, const unsigned* a) {
  const unsigned long long t = blockIdx.x * 256ull + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) s += a[(t * 1000003ull * 4096ull + k * 16) >> 12 & 2097151ull];
  o[t] = s;
}

// Exchange two halves of an array element by element: two loads, two stores.
extern "C" __global__ void __launch_bounds__(256) k_swaprows(unsigned* a) {
  const unsigned base = blockIdx.x * 256 * 8 + threadIdx.x;
  for (int k = 0; k < 8; ++k) {
    unsigned i = base + k * 256;
    unsigned j = i + 1048576u;
    unsigned lo = a[i];
    unsigned hi = lo + 1u;
    a[i] = hi;
    a[j] = lo;
  }
}

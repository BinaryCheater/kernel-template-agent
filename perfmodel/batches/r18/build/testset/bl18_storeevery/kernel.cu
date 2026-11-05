// Stores to every ninth word, three per thread.
extern "C" __global__ void __launch_bounds__(128) k_storeevery(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 3; ++k) o[(t * 3 + k) * 9] = t + k;
}

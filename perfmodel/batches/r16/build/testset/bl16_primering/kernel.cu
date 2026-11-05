// A store ring of a prime count of words.
extern "C" __global__ void __launch_bounds__(256) k_primering(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) o[(t * 17 + k * 5) % 1048573u] = a[t & 262143u];
}

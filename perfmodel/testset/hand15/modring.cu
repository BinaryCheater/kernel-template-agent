// A ring of 13 times 256 KiB words, read by a remainder and written back beside it.
__global__ void __launch_bounds__(128) k_modring(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) o[(t * 3 + k) % 1703936u] = a[(t + k * 64) % 1703936u] + k;
}

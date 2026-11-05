// A hash table without probing: the slot comes from a loaded key.
extern "C" __global__ void __launch_bounds__(256) k_hashput(unsigned* a, const unsigned* key) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = key[(t * 8 + k) & 1048575u];
    a[((v + k) * 2654435761u >> 9) & 262143u] = t;
  }
}

// Hash-join probe: a key, its slot in a table, a result per key.
extern "C" __global__ void __launch_bounds__(256) k_probe(unsigned* out, const unsigned* keys, const unsigned* table) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned i = t * 8 + k;
    unsigned key = keys[i];
    unsigned v = table[((key * 2654435761u) >> 9) & 262143u];
    out[i] = v ^ key;
  }
}

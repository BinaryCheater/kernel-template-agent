// Cuckoo lookup: two hashed slots per key, the smaller one kept.
extern "C" __global__ void __launch_bounds__(256) k_cuckoo(unsigned* out, const unsigned* table) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    unsigned a = table[(x >> 6) & 8388607u];
    unsigned b = table[((x ^ (x >> 15)) * 2246822519u >> 6) & 8388607u];
    s ^= a < b ? a : b;
  }
  out[t] = s;
}

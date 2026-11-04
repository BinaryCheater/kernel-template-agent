// Mark hashed items in a bit set of 1 Mibit.
extern "C" __global__ void __launch_bounds__(256) k_bitset(unsigned* bits) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    atomicOr(&bits[(x >> 13) & 262143u], 1u << ((x >> 8) & 31u));
  }
}

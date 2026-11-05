// round 16: 64 MiB hashed
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    o[((t * 8 + k) * 2654435761u) >> 6 & 16777215u] = t;
  }
}

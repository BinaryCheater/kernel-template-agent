// Deposit loaded floats into 4096 hashed cells.
__global__ void __launch_bounds__(256) k_floatadd(float* s, const float* x) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned i = t * 8 + k;
    float v = x[i & 1048575u];
    atomicAdd(&s[((i * 2654435761u) >> 11) & 4095u], v);
  }
}

// An index through a float: a resampled row.
__global__ void __launch_bounds__(128) k_floatindex(unsigned* o, const float* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  float s = 0;
  for (unsigned k = 0; k < 8; ++k) s += a[(unsigned)((float)t * 0.37f + (float)k * 1.5f)];
  o[t] = (unsigned)s;
}

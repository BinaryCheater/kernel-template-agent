// generated, seed 1010, kernel 2
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(64) k_stress2(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 64 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 32; ++k) {
    s[t & 63] = f3;
    u2 = (u2 ^ (u1 >> 7)) + (u2 << 3);
    f3 = expf(f0 * 0.001f);
    f3 = (float)u1 * 1e-9f;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

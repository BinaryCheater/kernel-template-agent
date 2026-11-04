// generated, seed 1010, kernel 8
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(128) k_stress8(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 128 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 16; ++k) {
    f3 = f1 * 0.999f + f0;
    u2 ^= in[(base + (size_t)k * 1024) & MASK];
    f2 = sqrtf(fabsf(f3) + 1.f);
    f3 = expf(f1 * 0.001f);
    f3 = (float)u3 * 1e-9f;
    u1 = (unsigned)fabsf(f1) + u2;
    f2 = f1 + f3;
    s[t & 63] = f1;
    u3 ^= in[(base + (size_t)k * 1024) & MASK];
    f3 = sqrtf(fabsf(f1) + 1.f);
    s[t & 63] = f2;
    f2 = sqrtf(fabsf(f1) + 1.f);
    f2 = sqrtf(fabsf(f1) + 1.f);
    out[(base + (size_t)k * 256) & OMASK] = u1;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

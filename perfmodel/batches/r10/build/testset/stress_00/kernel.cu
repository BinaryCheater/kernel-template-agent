// generated, seed 1010, kernel 0
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(64) k_stress0(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 64 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 8; ++k) {
    u3 ^= in[(base + (size_t)k * 1024) & MASK];
    f0 = (u2 & 1) ? f2 : f1;
    u1 = (u0 ^ (u1 >> 7)) + (u1 << 3);
    f0 = (float)u0 * 1e-9f;
    f0 = (float)u2 * 1e-9f;
    f1 = f2 / (f2 + 1.5f);
    f1 = (u2 & 1) ? f2 : f1;
    out[(base + (size_t)k * 256) & OMASK] = u0;
    f2 = f1 + f0;
    f3 += s[(t + k) & 63];
    f3 += (float)in[(base * 3 + (size_t)k * 4099) & MASK];
    f0 = sqrtf(fabsf(f0) + 1.f);
    f0 = f2 * 0.999f + f0;
    f1 += s[(t + k) & 63];
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

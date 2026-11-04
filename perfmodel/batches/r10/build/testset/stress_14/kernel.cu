// generated, seed 1010, kernel 14
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(64) k_stress14(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 64 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 8; ++k) {
    out[(base + (size_t)k * 256) & OMASK] = u1;
    f3 += s[(t + k) & 63];
    f3 = (u3 & 1) ? f3 : f0;
    u0 ^= in[(base + (size_t)k * 256) & MASK];
    u0 = in[(base ^ u2) & MASK];
    f3 = f0 + f1;
    u0 ^= in[(base + (size_t)k * 8) & MASK];
    f0 = (u2 & 1) ? f2 : f1;
    u3 = (unsigned)fabsf(f1) + u1;
    s[t & 63] = f0;
    f1 = f3 / (f2 + 1.5f);
    f3 = sqrtf(fabsf(f3) + 1.f);
    f3 += s[(t + k) & 63];
    f2 = f1 * 0.999f + f0;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

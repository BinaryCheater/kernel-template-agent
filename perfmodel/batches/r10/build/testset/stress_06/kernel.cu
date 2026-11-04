// generated, seed 1010, kernel 6
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(128) k_stress6(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 128 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 32; ++k) {
    u1 = (unsigned)fabsf(f3) + u2;
    f0 = sqrtf(fabsf(f1) + 1.f);
    f2 += s[(t + k) & 63];
    u2 = (unsigned)fabsf(f3) + u3;
    f0 += (float)in[(base * 97 + (size_t)k * 4099) & MASK];
    out[(base + (size_t)k * 64) & OMASK] = u1;
    f3 = (float)u3 * 1e-9f;
    out[(base + (size_t)k * 256) & OMASK] = u1;
    f0 = expf(f3 * 0.001f);
    f1 += s[(t + k) & 63];
    f1 = f0 * 0.999f + f2;
    f2 = f0 / (f1 + 1.5f);
    f1 = sqrtf(fabsf(f0) + 1.f);
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

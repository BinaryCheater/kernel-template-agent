// generated, seed 1010, kernel 15
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(256) k_stress15(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 256 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 32; ++k) {
    f0 = sqrtf(fabsf(f0) + 1.f);
    out[(base + (size_t)k * 8) & OMASK] = u1;
    f1 += (float)in[(base * 17 + (size_t)k * 4099) & MASK];
    f1 = f1 / (f1 + 1.5f);
    f1 += s[(t + k) & 63];
    u1 ^= in[(base + (size_t)k * 64) & MASK];
    s[t & 63] = f3;
    u2 = u1 * 2654435761u + u3;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

// generated, seed 1010, kernel 3
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(128) k_stress3(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 128 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 16; ++k) {
    f2 += (float)in[(base * 97 + (size_t)k * 4099) & MASK];
    s[t & 63] = f1;
    f2 = f1 + f3;
    s[t & 63] = f2;
    f2 = expf(f3 * 0.001f);
    f2 = f1 / (f0 + 1.5f);
    s[t & 63] = f3;
    u3 = (unsigned)fabsf(f2) + u0;
    f0 = expf(f1 * 0.001f);
    u2 ^= in[(base + (size_t)k * 1) & MASK];
    u3 ^= in[(base + (size_t)k * 1) & MASK];
    u0 ^= in[(base + (size_t)k * 64) & MASK];
    f0 = f2 / (f2 + 1.5f);
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

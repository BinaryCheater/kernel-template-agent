// generated, seed 1010, kernel 13
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(256) k_stress13(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 256 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 8; ++k) {
    u1 = u2 * 2654435761u + u3;
    u3 ^= in[(base + (size_t)k * 8) & MASK];
    s[t & 63] = f1;
    f1 += s[(t + k) & 63];
    f0 = f0 * 0.999f + f1;
    u3 = in[(base ^ u2) & MASK];
    f2 = expf(f1 * 0.001f);
    f2 = f1 + f3;
    f2 = f2 + f3;
    f3 = f2 * 0.999f + f2;
    f0 = (float)u3 * 1e-9f;
    f3 += (float)in[(base * 3 + (size_t)k * 4099) & MASK];
    f3 = f0 * 0.999f + f0;
    u1 = (u0 ^ (u2 >> 7)) + (u1 << 3);
    u0 = u2 * 2654435761u + u1;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

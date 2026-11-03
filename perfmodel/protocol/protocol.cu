// Measurement protocol kernels (gb10-measure steps 2, 3, 6): clock calibration and two canaries.
//  clock: (a) one warp runs a dependent FFMA chain of CHAIN instructions, timed by events and by
//         clock64: GHz = clock64 cycles / event time; (b) 48 blocks spin on clock64 for a fixed
//         number of ticks and read %globaltimer at both ends (upstream microbench/clock.cu):
//         GHz = ticks / ns, min and max over the SMs.
//  fma:   compute canary, 48 x 1024 threads of 8 independent FFMA chains, event time.
//  mem:   memory canary, a 256 MB read+write copy (DRAM-bound), event time and GB/s.
// Usage: protocol <clock|fma|mem>
#include <cstring>
#include "../probes/common.cuh"

constexpr int CHAIN = 1 << 22;
__global__ void chain(float* out, unsigned long long* cyc) {
  float a = threadIdx.x * 1e-3f;
  long long c0 = clock64();
#pragma unroll 16
  for (int i = 0; i < CHAIN; i++) a = fmaf(a, 0.999f, 0.5f);
  long long c1 = clock64();
  if (threadIdx.x == 0) cyc[0] = c1 - c0;
  if (a == 0.f) out[0] = a;
}
__global__ void spin(long long ticks, unsigned long long* out, float* sink) {
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  long long c0 = clock64();
  float a = threadIdx.x;
  while (clock64() - c0 < ticks) a = a * 1.0001f + 0.5f;
  long long c1 = clock64();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (threadIdx.x == 0) {
    out[2 * blockIdx.x] = c1 - c0;
    out[2 * blockIdx.x + 1] = g1 - g0;
  }
  if (a == 0.f) sink[0] = a;
}
constexpr int FMA_IT = 1 << 14;
__global__ void fma_canary(float* out) {
  float a[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] = threadIdx.x + j;
#pragma unroll 4
  for (int i = 0; i < FMA_IT; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = fmaf(a[j], 0.9999f, 0.25f);
  float s = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) s += a[j];
  if (s == 0.f) out[0] = s;
}
__global__ void copy(const int4* __restrict__ src, int4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "clock";
  float* f;
  unsigned long long* c;
  CK(cudaMalloc(&f, 4));
  CK(cudaMalloc(&c, 96 * 8));
  if (!strcmp(mode, "clock")) {
    unsigned long long cyc = 0;
    Stat s = repeat([&] {
      double us = event_us([&] { chain<<<1, 32>>>(f, c); });
      CK(cudaMemcpy(&cyc, c, 8, cudaMemcpyDeviceToHost));
      return cyc / (us * 1e3);
    });
    char x[128];
    snprintf(x, sizeof x, "chain_cycles=%llu cycles_per_ffma=%.3f", cyc, (double)cyc / CHAIN);
    emit("clock", "chain_GHz", s, "GHz", x);
    unsigned long long h[96];
    double mn = 1e9, mx = 0;
    for (int r = 0; r < 3; r++) {
      spin<<<48, 128>>>(50000000LL, c, f);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h, c, 96 * 8, cudaMemcpyDeviceToHost));
      for (int b = 0; b < 48; b++) {
        double g = (double)h[2 * b] / h[2 * b + 1];
        mn = std::min(mn, g);
        mx = std::max(mx, g);
      }
    }
    printf("RESULT clock spin_GHz median=%.4f iqr=%.4f unit=GHz n=144 min=%.4f max=%.4f\n", (mn + mx) / 2, mx - mn, mn, mx);
  } else if (!strcmp(mode, "fma")) {
    Stat s = repeat([&] { return event_us([&] { fma_canary<<<48, 1024>>>(f); }); });
    emit("canary", "fma_us", s, "us");
  } else {
    size_t bytes = 256ull << 20, n = bytes / 16;
    int4 *a, *b;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMemset(a, 1, bytes));
    Stat s = repeat([&] { return event_us([&] { copy<<<48 * 8, 256>>>(a, b, n); }); });
    char x[64];
    snprintf(x, sizeof x, "GBs=%.1f", 2.0 * bytes / (s.median * 1e3));
    emit("canary", "mem_us", s, "us", x);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}

// DRAM rate of a short read burst (open questions 12, 24: rows with DRAM share 84-94 % replay +10 to
// +17 %; 17 MB of decode reads in 68 us is above the 237 GB/s plateau of a 67 MB stream). The
// dram_rotate pattern: 48 blocks of T threads, each element once, LDG.128, one load in flight per
// warp, at n x 786 KB per launch, a walk over 256 MB (every launch cold). Reported: event time per
// launch and GB/s.
#include "../common.cuh"
constexpr size_t UNIT = 48ull * 1024 * 16;
template <int T, int N>
__global__ void __launch_bounds__(T) stream(const int4* __restrict__ src, int* sink) {
  constexpr int STEPS = UNIT * N / 16 / (48 * T);
  int acc = 0;
  const int4* p = src + blockIdx.x * T + threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < STEPS; i++) {
    int4 v = __ldcs(p + (size_t)i * 48 * T);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x7fff1234) sink[0] = acc;
}
int main() {
  const size_t TOTAL = 256ull << 20;
  int4* buf;
  int* sink;
  CK(cudaMalloc(&buf, TOTAL));
  CK(cudaMemset(buf, 1, TOTAL));
  CK(cudaMalloc(&sink, 4));
  auto one = [&](auto kern, int t, int n) {
    size_t at = 0, chunk = UNIT * n;
    Stat s = repeat([&] {
      if (at + chunk > TOTAL) at = 0;
      const int4* src = buf + at / 16;
      at += chunk;
      return event_us([&] { kern<<<48, t>>>(src, sink); });
    });
    char p[32], x[64];
    snprintf(p, sizeof p, "t%d_n%d", t, n);
    snprintf(x, sizeof x, "GBs=%.1f bytes=%zu", chunk / (s.median * 1e3), chunk);
    emit("dram_burst", p, s, "us", x);
  };
  one(stream<256, 5>, 256, 5);
  one(stream<256, 10>, 256, 10);
  one(stream<256, 21>, 256, 21);
  one(stream<256, 42>, 256, 42);
  one(stream<256, 85>, 256, 85);
  one(stream<1024, 5>, 1024, 5);
  one(stream<1024, 10>, 1024, 10);
  one(stream<1024, 21>, 1024, 21);
  one(stream<1024, 42>, 1024, 42);
  one(stream<1024, 85>, 1024, 85);
}

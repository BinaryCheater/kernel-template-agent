// DRAM latency and its spread against load (LOOP contention group; inventory §5 DRAM latency). One
// lane of block 0 chases a random single cycle of lines through 256 MiB (past the L2) and stores
// each hop's cycles; G-1 other blocks read the same buffer, 32 lanes 128 B apart per warp, with Z
// FFMA passes between loads. Reported: hop p10, p50, p90, p99 in cycles, and the load in GB/s.
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 128, HOPS = 256;
constexpr size_t BUF = 256 << 20;
__device__ __forceinline__ unsigned ld(const void* p) {
  unsigned v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
template <int Z>
__global__ void __launch_bounds__(T, 1) k(unsigned* lat, const char* src, float* sink, unsigned* start) {
  const int t = threadIdx.x, b = blockIdx.x;
  float f = t;
  if (b == 0) {
    if (t == 0) {
      unsigned at = *start;
#pragma unroll 1
      for (int h = 0; h < HOPS; h++) {
        u64 c0 = clock64();
        at = ld(src + (size_t)at * 128);
        lat[h] = (unsigned)(clock64() - c0);
      }
      *start = at;
    }
  } else {
    constexpr int PASSES = 330000 / (1100 + Z);
    unsigned s0 = (unsigned)b * 104728u + t * 128u;
#pragma unroll 1
    for (int s = 0; s < PASSES; s++) {
      unsigned v = ld(src + (s0 + (unsigned)s * 40960u) % BUF);
      if (v == 0x12345u) sink[0] = 1.f;
#pragma unroll 1
      for (int z = 0; z < Z; z++) f = fmaf(f, 0.9999f, 0.25f);
    }
  }
  if (f == 1.2345f) sink[0] = f;
}
int main() {
  unsigned* lat;
  unsigned* start;
  char* src;
  float* sink;
  CK(cudaMalloc(&lat, HOPS * 4));
  CK(cudaMalloc(&start, 4));
  CK(cudaMalloc(&src, BUF));
  CK(cudaMalloc(&sink, 4));
  // one random cycle over the lines (Sattolo), the next line's index in each line's first word
  const size_t L = BUF / 128;
  std::vector<unsigned> nxt(L * 32, 0), perm(L);
  for (size_t i = 0; i < L; i++) perm[i] = i;
  u64 x = 88172645463325252ull;
  for (size_t i = L - 1; i > 0; i--) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    std::swap(perm[i], perm[x % i]);
  }
  for (size_t i = 0; i < L; i++) nxt[i * 32] = perm[i];
  CK(cudaMemcpy(src, nxt.data(), BUF, cudaMemcpyHostToDevice));
  unsigned zero = 0;
  CK(cudaMemcpy(start, &zero, 4, cudaMemcpyHostToDevice));
  auto point = [&](auto kern, int z, int g) {
    std::vector<double> v;
    double span = 0;
    for (int r = 0; r < 10; r++) {
      span = event_us([&] { kern<<<g, T>>>(lat, src, sink, start); });
      if (r < 2) continue;
      std::vector<unsigned> h(HOPS);
      CK(cudaMemcpy(h.data(), lat, HOPS * 4, cudaMemcpyDeviceToHost));
      for (unsigned c : h) v.push_back(c);
    }
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    for (auto [tag, q] : {std::pair{"p10", 10}, {"p50", 50}, {"p90", 90}, {"p99", 99}}) {
      char p[48];
      snprintf(p, sizeof p, "g%d_z%d_%s", g, z, tag);
      emit("dram_spread", p, Stat{v[n * q / 100], 0, (int)n}, "cycles");
    }
    constexpr int W = T / 32;
    char p[48];
    snprintf(p, sizeof p, "g%d_z%d_GBs", g, z);
    double gb = (g - 1) * W * 4096.0 * (330000 / (1100 + z)) / (span * 1e3);
    emit("dram_spread", p, Stat{gb, 0, 1}, "GB/s");
  };
#define PT(Z) point(k<Z>, Z, 48);
  point(k<0>, 0, 1);
  PT(0) PT(250) PT(500) PT(1000) PT(2000) PT(4000) PT(8000)
}

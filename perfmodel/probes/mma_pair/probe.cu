// Tensor pipe in the GEMM loop shape, alone and with two blocks per SM (STATE.md open questions 2
// and 3; inventory §2 "Tensor pipe per sub-partition" and "Warp scheduler ... greedy then oldest").
// Each warp runs the inner loop of TileLang's 128x128x32 / 4-warp HGEMM tile (warp tile 64x64): per
// k16 step 4 ldmatrix.x4 (A), 4 ldmatrix.x4.trans (B) and 32 independent HMMA.16816.F32, from
// shared memory only (no global traffic). 128 threads = one warp per sub-partition; 40 KB of dynamic
// shared memory so that at most two blocks share an SM, as in the test-set kernel.
// Points: b1 (one block), b48 (one per SM, the whole chip's tensor pipes), b96_s<S> (two per SM;
// blocks 48..95 first run a dependent FFMA chain of S iterations, a start skew). Per block thread 0
// records %smid, clock64 and %globaltimer at entry, after the skew and at exit (after
// __syncthreads). Reported per point: the SM's span (first entry to last exit) in cycles, for the
// pairs also the first block's finish (entry to the earlier exit); GHz = cycles / globaltimer ns over
// the spans, and the event time of the launch.
#include "../common.cuh"
constexpr int ITERS = 512;
constexpr int SMEM = 40 * 1024;
#define LDSM4(r, a) \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory")
#define LDSM4T(r, a) \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory")
__device__ __forceinline__ void mma(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
template <int SKEW>
__global__ void __launch_bounds__(128) k(unsigned zero, unsigned long long* out, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  // each ldmatrix.x4 reads 512 contiguous bytes (lane l gives row l): conflict-free
  const unsigned base = (unsigned)__cvta_generic_to_shared(buf) + warp * 8192 + lane * 16;
  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;
  __syncthreads();
  unsigned long long c0 = clock64(), g0 = gtime();
  float x = (float)(threadIdx.x & zero);
  if (blockIdx.x >= 48) {
#pragma unroll 1
    for (int i = 0; i < SKEW; i++) x = fmaf(x, 1.0001f, 1.0f);
  }
  __syncthreads();
  unsigned long long cs = clock64();
  unsigned off = (unsigned)x & zero;
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
    unsigned a[4][4], b[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) LDSM4(a[i], base + off + i * 512);
#pragma unroll
    for (int i = 0; i < 4; i++) LDSM4T(b[i], base + off + 2048 + i * 512);
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int j = 0; j < 8; j++) mma(acc[i][j], a[i], b[j / 2][2 * (j % 2)], b[j / 2][2 * (j % 2) + 1]);
    off = (a[0][0] ^ b[0][0]) & zero;
  }
  __syncthreads();
  unsigned long long c1 = clock64(), g1 = gtime();
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long* o = out + 6 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = cs, o[3] = c1, o[4] = g0, o[5] = g1;
  }
  float s = x;
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) s += acc[i][j][c];
  if (s == 12345.f) sink[0] = s;  // every accumulator live
}
struct Res {
  double span, first, ghz, pairs;
};
// per SM: span = last exit - first entry; first = earlier exit - first entry (two blocks on it)
static Res reduce(const std::vector<unsigned long long>& h, int nb) {
  struct S {
    std::vector<int> b;
  };
  std::vector<S> sm(256);
  for (int b = 0; b < nb; b++) sm[h[6 * b] % 256].b.push_back(b);
  std::vector<double> span, first, ghz;
  for (auto& s : sm) {
    if (s.b.empty()) continue;
    unsigned long long e0 = ~0ull, x0 = ~0ull, x1 = 0, g0 = ~0ull, g1 = 0;
    for (int b : s.b) {
      e0 = std::min(e0, h[6 * b + 1]), x0 = std::min(x0, h[6 * b + 3]), x1 = std::max(x1, h[6 * b + 3]);
      g0 = std::min(g0, h[6 * b + 4]), g1 = std::max(g1, h[6 * b + 5]);
    }
    span.push_back((double)(x1 - e0));
    if (s.b.size() == 2) first.push_back((double)(x0 - e0));
    ghz.push_back((double)(x1 - e0) / (double)(g1 - g0));
  }
  return {stat_of(span).median, first.empty() ? 0 : stat_of(first).median, stat_of(ghz).median, (double)first.size()};
}
int main() {
  unsigned long long* o;
  float* sink;
  CK(cudaMalloc(&o, 8 * 6 * 96));
  CK(cudaMalloc(&sink, 4));
  void (*fs[3])(unsigned, unsigned long long*, float*) = {k<0>, k<4096>, k<16384>};
  const int skews[3] = {0, 4096, 16384};
  for (auto f : fs) CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  auto point = [&](int fi, int nb, const char* name) {
    std::vector<unsigned long long> h(6 * nb);
    std::vector<double> ev, gz, fi_;
    Stat s = repeat([&] {
      ev.push_back(event_us([&] { fs[fi]<<<nb, 128, SMEM>>>(0u, o, sink); }));
      CK(cudaMemcpy(h.data(), o, 8 * 6 * nb, cudaMemcpyDeviceToHost));
      Res r = reduce(h, nb);
      gz.push_back(r.ghz);
      fi_.push_back(r.first);
      return r.span;
    });
    ev.erase(ev.begin(), ev.begin() + WARM);
    gz.erase(gz.begin(), gz.begin() + WARM);
    fi_.erase(fi_.begin(), fi_.begin() + WARM);
    Stat e = stat_of(ev), g = stat_of(gz), f = stat_of(fi_);
    char x[160];
    snprintf(x, sizeof x, "event_us=%.3f ghz=%.4f skew_iters=%d", e.median, g.median, skews[fi]);
    char p[48];
    snprintf(p, sizeof p, "%s_span", name);
    emit("mma_pair", p, s, "cycles", x);
    if (nb == 96) {
      snprintf(p, sizeof p, "%s_first", name);
      emit("mma_pair", p, f, "cycles", x);
    }
  };
  point(0, 1, "b1");
  point(0, 48, "b48");
  point(0, 96, "b96_s0");
  point(1, 96, "b96_s4096");
  point(2, 96, "b96_s16384");
}

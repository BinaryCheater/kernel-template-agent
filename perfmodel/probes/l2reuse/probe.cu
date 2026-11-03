// L2 residency of re-read data (STATE.md open question 4 / round 3; inventory §5 "L2 capacity and
// replacement"): what "warm" means. A buffer of F = 0.75 n MiB (n steps of 48 blocks x 1024 threads x
// 16 B) is read with 16-byte loads of one access kind:
//   kind 0 `cg` ld.global.cg (L2 only), kind 1 `ca` plain LDG (L1-allocating, as TileLang's GEMV and
//   RMSNorm load), kind 2 `cs` ld.global.cs (evict-first, as the dram_rotate probe).
// Modes per (n, kind), all timed by %globaltimer inside the kernel (first block entry to last block
// exit; event time beside it):
//   cold  a 96 MiB read of another buffer (a read flush, no dirty lines), then one pass;
//   warm  one pass after an untimed pass of the same kernel, back-to-back (across launches);
//   in4   a read flush, then four passes over the buffer in one launch (within one kernel).
// Re-read time per pass within a kernel = (in4 - cold) / 3. Profiler mode for counters:
//   bin ncu <n> <kind> <mode>   runs the mode once (the last `rd` launch is the one to profile).
#include "../common.cuh"
#include <cstring>
constexpr int TB = 1024, NB = 48;
constexpr size_t STEP = (size_t)NB * TB;  // int4 per step
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
template <int K>
__device__ __forceinline__ int4 ld(const int4* p) {
  int4 v;
  if (K == 0)
    asm volatile("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else if (K == 1)
    v = *p;
  else
    asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int N, int K, int P>
__global__ void __launch_bounds__(TB) rd(const int4* __restrict__ src, unsigned long long* t, int* sink) {
  unsigned long long g0 = gtime();
  int acc = 0;
  const int4* p = src + blockIdx.x * TB + threadIdx.x;
#pragma unroll 1
  for (int r = 0; r < P; r++) {
#pragma unroll 4
    for (int i = 0; i < N; i++) {
      int4 v = ld<K>(p + (size_t)i * STEP);
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x7fff1234) sink[0] = acc;
  __syncthreads();
  if (threadIdx.x == 0) t[2 * blockIdx.x] = g0, t[2 * blockIdx.x + 1] = gtime();
}
// read flush: 96 MiB of another buffer, plain loads
__global__ void flush(const int4* __restrict__ src, size_t n4, int* sink) {
  int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldcg(src + i);
    acc ^= v.x ^ v.w;
  }
  if (acc == 0x7fff1234) sink[0] = acc;
}
static const size_t FLUSH = 96ull << 20;
static int4 *buf, *fbuf;
static int* sink;
static unsigned long long* tdev;
static double span_us() {
  unsigned long long h[2 * NB];
  CK(cudaMemcpy(h, tdev, sizeof h, cudaMemcpyDeviceToHost));
  unsigned long long a = h[0], b = h[1];
  for (int i = 0; i < NB; i++) a = std::min(a, h[2 * i]), b = std::max(b, h[2 * i + 1]);
  return (b - a) * 1e-3;
}
static void do_flush() { flush<<<NB * 4, 1024>>>(fbuf, FLUSH / 16, sink); }
// one mode: returns (span us, event us) of the timed launch
template <int N, int K>
static void mode(const char* m, double* span, double* ev) {
  if (!strcmp(m, "warm")) {
    rd<N, K, 1><<<NB, TB>>>(buf, tdev, sink);
    *ev = event_us([&] { rd<N, K, 1><<<NB, TB>>>(buf, tdev, sink); });
  } else if (!strcmp(m, "cold")) {
    do_flush();
    *ev = event_us([&] { rd<N, K, 1><<<NB, TB>>>(buf, tdev, sink); });
  } else {
    do_flush();
    *ev = event_us([&] { rd<N, K, 4><<<NB, TB>>>(buf, tdev, sink); });
  }
  CK(cudaDeviceSynchronize());
  *span = span_us();
}
static const char* KIND[] = {"cg", "ca", "cs"};
template <int N, int K>
static void point(const char* only_mode) {
  for (const char* m : {"cold", "warm", "in4"}) {
    if (only_mode && strcmp(only_mode, m)) continue;
    if (only_mode) {  // profiler mode: one run
      double s, e;
      mode<N, K>(m, &s, &e);
      printf("RESULT l2reuse ncu_n%d_%s_%s once\n", N, KIND[K], m);
      continue;
    }
    std::vector<double> sp, evs;
    for (int i = 0; i < WARM + REPS; i++) {
      double s, e;
      mode<N, K>(m, &s, &e);
      if (i >= WARM) sp.push_back(s), evs.push_back(e);
    }
    Stat s = stat_of(sp), e = stat_of(evs);
    char p[48], x[96];
    snprintf(p, sizeof p, "n%d_%s_%s", N, KIND[K], m);
    double mib = 0.75 * N * (strcmp(m, "in4") ? 1 : 4);
    snprintf(x, sizeof x, "MiB=%.2f GBs=%.1f event_us=%.3f", 0.75 * N, mib * 1048576 / (s.median * 1e3), e.median);
    emit("l2reuse", p, s, "us", x);
  }
}
template <int N>
static void all_kinds(int kind, const char* m) {
  if (kind < 0 || kind == 0) point<N, 0>(m);
  if (kind < 0 || kind == 1) point<N, 1>(m);
  if (kind < 0 || kind == 2) point<N, 2>(m);
}
int main(int argc, char** argv) {
  CK(cudaMalloc(&buf, 64ull << 20));
  CK(cudaMalloc(&fbuf, FLUSH));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMalloc(&tdev, 2 * NB * sizeof(unsigned long long)));
  CK(cudaMemset(buf, 1, 64ull << 20));
  CK(cudaMemset(fbuf, 2, FLUSH));
  CK(cudaDeviceSynchronize());
  int n = -1, kind = -1;
  const char* m = nullptr;
  if (argc > 4 && !strcmp(argv[1], "ncu")) {
    n = atoi(argv[2]);
    for (int k = 0; k < 3; k++)
      if (!strcmp(argv[3], KIND[k])) kind = k;
    m = argv[4];
  }
#define PT(N) \
  if (n < 0 || n == N) all_kinds<N>(kind, m);
  PT(2) PT(8) PT(16) PT(20) PT(22) PT(24) PT(27) PT(32) PT(43)
  return 0;
}

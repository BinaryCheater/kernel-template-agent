// Register spills (inventory §3 local memory; r7: 128x128 tiles at 64 threads replay +82 %, measure
// +228 %; 256x256x32 s2 t256 870 us, replay 344; both carry the fragment on the stack, 255 registers
// per thread): N words of stack per thread, loaded and stored M times, T threads, one launch of G
// blocks with S bytes of dynamic shared memory (L1 left to the stack). G 48: one block per SM; 192:
// four. Reported: span on an SM (cycles).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int M = 4;
template <int T, int N>
__global__ void __launch_bounds__(T) k(u64* out, unsigned* sink) {
  volatile unsigned a[N];
  u64 c0 = clock64();
  unsigned s = threadIdx.x;
#pragma unroll 1
  for (int r = 0; r < M; r++) {
#pragma unroll 1
    for (int i = 0; i < N; i++) {
      a[i] = s + i;
      s ^= a[i];
    }
  }
  __syncthreads();
  u64 c1 = clock64();
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    u64* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
  if (s == 12345) sink[0] = 1;
}
static double reduce(const std::vector<u64>& h, int nb) {
  std::vector<u64> lo(256, ~0ull), hi(256, 0);
  for (int b = 0; b < nb; b++) {
    int m = h[3 * b] % 256;
    lo[m] = std::min(lo[m], h[3 * b + 1]), hi[m] = std::max(hi[m], h[3 * b + 2]);
  }
  std::vector<double> span;
  for (int m = 0; m < 256; m++)
    if (hi[m]) span.push_back((double)(hi[m] - lo[m]));
  return stat_of(span).median;
}
int main() {
  u64* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8 * 3 * 192));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, int t, int n) {
    for (int nb : {1, 48, 192})
      for (int sm : {0, 96 * 1024}) {
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
        std::vector<u64> h(3 * nb);
        Stat s = repeat([&] {
          CK(event_us([&] { kern<<<nb, t, sm>>>(o, sink); }) < 0);
          CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
          return reduce(h, nb);
        });
        char p[48];
        snprintf(p, sizeof p, "t%d_n%d_g%d_s%d", t, n, nb, sm / 1024);
        emit("spill", p, s, "cycles");
      }
  };
#define PT(T, N) point(k<T, N>, T, N)
  PT(64, 16); PT(64, 64); PT(64, 256); PT(64, 1024);
  PT(256, 16); PT(256, 64); PT(256, 256); PT(256, 1024);
}

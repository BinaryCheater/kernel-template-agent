// Which sub-partition a block's warps go to (question 29; inventory §1 warp scheduler; GEMV n_part 2
// and reduce 16: 64-thread blocks, replay +73 %, measured -12 %). T threads per block, n blocks per
// SM (grid 48 n), every warp runs one stream: E ex2 (the XU, 8 cycles per instruction per
// sub-partition) or an mma stream (the tensor pipe, 16). Reported: span on an SM (cycles).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int ITERS = 128;
#define OPS4 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ void mma(float* f, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " OPS4
               : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mark(u64* out, u64 c0, u64 c1) {
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    u64* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
}
template <int T>
__global__ void __launch_bounds__(T) x(u64* out, float* sink) {
  float e[8];
#pragma unroll
  for (int j = 0; j < 8; j++) e[j] = -1e-3f * (threadIdx.x + j);
  __syncthreads();
  u64 c0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int j = 0; j < 8; j++) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(e[j]));
  }
  __syncthreads();
  u64 c1 = clock64();
  mark(out, c0, c1);
  float s = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) s += e[j];
  if (s == 1.2345f) sink[0] = s;
}
template <int T>
__global__ void __launch_bounds__(T) m(u64* out, float* sink) {
  float f[8][4];
  unsigned a[4], b[2];
#pragma unroll
  for (int i = 0; i < 4; i++) a[i] = threadIdx.x * (i + 1);
#pragma unroll
  for (int i = 0; i < 2; i++) b[i] = threadIdx.x * (i + 5);
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  __syncthreads();
  u64 c0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int c = 0; c < 8; c++) mma(f[c], a, b);
  }
  __syncthreads();
  u64 c1 = clock64();
  mark(out, c0, c1);
  float s = 0;
#pragma unroll
  for (int c = 0; c < 8; c++) s += f[c][0] + f[c][1];
  if (s == 1.2345f) sink[0] = s;
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
  float* sink;
  CK(cudaMalloc(&o, 8 * 3 * 384));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, const char* name, int t, int n) {
    int nb = 48 * n;
    std::vector<u64> h(3 * nb);
    Stat s = repeat([&] {
      CK(event_us([&] { kern<<<nb, t>>>(o, sink); }) < 0);
      CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
      return reduce(h, nb);
    });
    char p[48];
    snprintf(p, sizeof p, "%s_t%d_n%d", name, t, n);
    emit("warp_map", p, s, "cycles");
  };
  for (int n : {1, 2, 4, 8}) point(x<32>, "ex2", 32, n);
  for (int n : {1, 2, 4}) point(x<64>, "ex2", 64, n);
  for (int n : {1, 2}) point(x<96>, "ex2", 96, n);
  for (int n : {1, 2}) point(x<128>, "ex2", 128, n);
  for (int n : {1, 2, 4}) point(m<32>, "mma", 32, n);
  for (int n : {1, 2, 4}) point(m<64>, "mma", 64, n);
  for (int n : {1, 2}) point(m<128>, "mma", 128, n);
}

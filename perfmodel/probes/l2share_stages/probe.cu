// P-2, second experiment (STATE.md open question 1): cp.async delivery from the L2 by stages and
// blocks. The loop is TileLang's pipelined copy loop (as in l2share_copy's kernel: prologue of S-1
// stages, then per k-tile: __syncthreads, copies of tile k+S-1, commit, wait_group<S-1>,
// __syncthreads), loads only, no compute, no store; tile 128x256x32 fp16 (A 128 rows x 64 B pieces,
// B 32 rows x 512 B) = 24 KB per stage, 256 threads (A 2, B 4 cp.async.cg 16 B per thread per tile),
// 32 k-tiles (K = 1024), linear conflict-free shared layout, L2-resident operands (warm).
// Stages S in {1, 2, 3, 4}; blocks in {1, 16, 48} (one per SM; grids as in l2share_copy). With S = 1
// every tile waits for its own copies: the per-tile time is the L2 round trip under the load the
// other blocks make. Reported: mean per-block %globaltimer span (entry to exit) in us, and the event
// time of the launch.
#include <cuda_fp16.h>
#include "../common.cuh"
constexpr int M = 1536, N = 1024, K = 1024, BM = 128, BN = 256, BK = 32, KT = K / BK;
constexpr int TILE = (BM + BN) * BK * 2;  // 24576 B per stage
__device__ unsigned long long ts[2 * 64];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
template <int S>
__device__ __forceinline__ void tile(unsigned sm, const __half* A, const __half* B, int kt, int st) {
  const int t = threadIdx.x;
  const unsigned s0 = sm + st * TILE;
#pragma unroll
  for (int i = 0; i < 2; i++) {  // A: 512 pieces of 16 B; 4 per 64-byte row
    int p = i * 256 + t;
    cp16(s0 + p * 16, A + (size_t)(p / 4) * K + kt * BK + (p % 4) * 8);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {  // B: 1024 pieces; 32 per 512-byte row
    int p = i * 256 + t;
    cp16(s0 + BM * BK * 2 + p * 16, B + (size_t)(kt * BK + p / 32) * N + (p % 32) * 8);
  }
}
template <int S>
__global__ void __launch_bounds__(256, 1) load(const __half* A, const __half* B) {
  extern __shared__ __align__(128) unsigned char buf[];
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) ts[2 * bid] = gtime();
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const __half* a = A + (size_t)blockIdx.y * BM * K;
  const __half* b = B + blockIdx.x * BN;
#pragma unroll
  for (int s = 0; s < S - 1; s++) {
    tile<S>(sm, a, b, s, s);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#pragma unroll 1
  for (int k = 0; k < KT - (S - 1); k++) {
    __syncthreads();
    tile<S>(sm, a, b, k + S - 1, (k + S - 1) % S);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group %0;" ::"n"(S - 1) : "memory");
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) ts[2 * bid + 1] = gtime();
}
__global__ void fill(__half* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    p[i] = __float2half((float)(x & 0xffff) / 32768.0f - 1.0f);
  }
}
template <int S>
static void run(const __half* A, const __half* B) {
  const int smem = S * TILE;
  CK(cudaFuncSetAttribute(load<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int grids[3][2] = {{1, 1}, {4, 4}, {4, 12}};
  static unsigned long long h[2 * 64];
  for (auto& g : grids) {
    dim3 grid(g[0], g[1]);
    int nb = g[0] * g[1];
    std::vector<double> ev;
    Stat s = repeat([&] {
      ev.push_back(event_us([&] { load<S><<<grid, 256, smem>>>(A, B); }));
      CK(cudaMemcpyFromSymbol(h, ts, 16 * nb));
      double m = 0;
      for (int b = 0; b < nb; b++) m += h[2 * b + 1] - h[2 * b];
      return m / nb / 1e3;
    });
    ev.erase(ev.begin(), ev.begin() + WARM);
    Stat e = stat_of(ev);
    char p[32], x[64];
    snprintf(p, sizeof p, "s%d_b%d", S, nb);
    snprintf(x, sizeof x, "event_us=%.3f event_iqr=%.3f", e.median, e.iqr);
    emit("l2share_stages", p, s, "us", x);
  }
}
int main() {
  __half *A, *B;
  CK(cudaMalloc(&A, (size_t)M * K * 2));
  CK(cudaMalloc(&B, (size_t)K * N * 2));
  fill<<<1024, 256>>>(A, (size_t)M * K, 0x9E3779B9u);
  fill<<<1024, 256>>>(B, (size_t)K * N, 0x7F4A7C15u);
  CK(cudaDeviceSynchronize());
  run<1>(A, B);
  run<2>(A, B);
  run<3>(A, B);
  run<4>(A, B);
}

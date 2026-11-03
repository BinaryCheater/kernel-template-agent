// P-2: cp.async chip-L2 delivery with one block per SM (STATE.md P-2). The kernel is upstream's
// TileLang-generated loads-only, store-free variant of c128x256x64_st2_th256 (calibration/spark-6951/
// ablation_cpasync/c128x256x64_st2_th256/nocompute_nostore.kernel.cu, copied unchanged): 16 k-tiles of
// 48 KB (A 128x64 + B 64x256 fp16), 2 stages, 256 threads; L2-resident operands (warm).
// Blocks in {1, 16, 32, 48} (grid 1x1, 4x4, 4x8, 4x12; A is allocated with 1536 rows so the 48-block
// grid stays in bounds). Reported: mean per-block %globaltimer span (kernel entry to exit) and the
// event time of the launch. Build: nvcc -O3 -arch=sm_121a -std=c++20 --expt-relaxed-constexpr -I<tilelang src>
#include <cuda.h>
#include "../common.cuh"
__device__ unsigned long long __tl_ts[2 * 65536];
#include "kernel.cu"
__global__ void fill(half_t* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    x ^= x >> 15;
    p[i] = half_t((float)(x & 0xffff) / 32768.0f - 1.0f);
  }
}
int main() {
  const int M = 1536, N = 1024, K = 1024, smem = 98304;
  half_t *A, *B, *C;
  CK(cudaMalloc(&A, (size_t)M * K * 2));
  CK(cudaMalloc(&B, (size_t)K * N * 2));
  CK(cudaMalloc(&C, (size_t)M * N * 2));
  fill<<<1024, 256>>>(A, (size_t)M * K, 0x9E3779B9u);
  fill<<<1024, 256>>>(B, (size_t)K * N, 0x7F4A7C15u);
  CK(cudaDeviceSynchronize());
  CK(cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int grids[4][2] = {{1, 1}, {4, 4}, {4, 8}, {4, 12}};
  static unsigned long long ts[2 * 48];
  for (auto& g : grids) {
    dim3 grid(g[0], g[1]);
    int nb = g[0] * g[1];
    std::vector<double> ev;
    Stat s = repeat([&] {
      ev.push_back(event_us([&] { matmul_kernel<<<grid, 256, smem>>>(A, B, C); }));
      CK(cudaMemcpyFromSymbol(ts, __tl_ts, 16 * nb));
      double m = 0;
      for (int b = 0; b < nb; b++) m += ts[2 * b + 1] - ts[2 * b];
      return m / nb / 1e3;
    });
    ev.erase(ev.begin(), ev.begin() + WARM);
    Stat e = stat_of(ev);
    char p[32], x[64];
    snprintf(p, sizeof p, "b%d", nb);
    snprintf(x, sizeof x, "event_us=%.3f event_iqr=%.3f", e.median, e.iqr);
    emit("l2share_copy", p, s, "us", x);
  }
}

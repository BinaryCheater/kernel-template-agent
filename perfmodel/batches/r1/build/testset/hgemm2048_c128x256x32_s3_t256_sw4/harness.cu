// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random fp16 inputs in [-1, 1); warm cache state: WARM untimed launches, then REPS launches
// timed one by one with CUDA events (no profiler). Usage: harness [reps] (reps 1 = one launch after
// warm-up, for the profiler).
#include <cuda.h>
#include "/o/common.cuh"
#include "kernel.cu"
__global__ void fill(half_t* p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = half_t((float)(x & 0xffff) / 32768.0f - 1.0f);
  }
}
int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 30;
  const size_t sizes[] = {4194304, 4194304, 4194304};
  half_t* buf[3];
  for (int i = 0; i < 3; i++) { CK(cudaMalloc(&buf[i], sizes[i] * 2)); fill<<<1024, 256>>>(buf[i], sizes[i], 0x9E3779B9u * (i + 1)); }
  CK(cudaDeviceSynchronize());
  const int smem = 73728;
  CK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(8, 16, 1), block(256);
  auto launch = [&] { gemm_kernel<<<grid, block, smem>>>(buf[0], buf[1], buf[2]); };
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT hgemm2048_c128x256x32_s3_t256_sw4 once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us(launch));
  Stat s = stat_of(v);
  printf("RESULT hgemm2048_c128x256x32_s3_t256_sw4 kernel median=%.4f iqr=%.4f unit=us n=%d samples=", s.median, s.iqr, s.n);
  for (double x : v) printf("%.3f,", x);
  printf("\n");
  CK(cudaDeviceSynchronize());
  return 0;
}

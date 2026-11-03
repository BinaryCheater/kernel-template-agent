// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random inputs (fp16 values in [-1, 1) written over every buffer's bytes); WARM untimed launches,
// then REPS launches timed one by one with CUDA events (no profiler). ROT input sets are used in
// turn (ROT 1: warm; ROT x footprint above the L2: rotating). Usage: harness [reps] (reps 1 = one
// launch after warm-up, for the profiler).
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
  const int ROT = 8;
  const size_t bytes[] = {4194304, 8192, 4194304};
  void* buf[ROT][3];
  for (int r = 0; r < ROT; r++)
    for (int i = 0; i < 3; i++) {
      CK(cudaMalloc(&buf[r][i], bytes[i]));
      fill<<<1024, 256>>>((half_t*)buf[r][i], bytes[i] / 2, 0x9E3779B9u * (i + 1) + 77u * r);
    }
  CK(cudaDeviceSynchronize());
  const int smem = 512;
  CK(cudaFuncSetAttribute(rms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(512), block(128);
  int next = 0;
  auto launch = [&] { void** b = buf[next++ % ROT]; rms_kernel<<<grid, block, smem>>>((const half_t*)b[1], (const half_t*)b[0], (half_t*)b[2]); };
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT rmsnorm_m512_n4096_rotating once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us(launch));
  Stat s = stat_of(v);
  printf("RESULT rmsnorm_m512_n4096_rotating kernel median=%.4f iqr=%.4f unit=us n=%d samples=", s.median, s.iqr, s.n);
  for (double x : v) printf("%.3f,", x);
  printf("\n");
  CK(cudaDeviceSynchronize());
  return 0;
}

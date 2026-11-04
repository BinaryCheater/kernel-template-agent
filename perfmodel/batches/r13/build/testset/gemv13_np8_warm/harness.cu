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
  const int ROT = 1;
  const size_t bytes[] = {12288, 9437184, 589824, 6144};
  void* buf[ROT][4];
  for (int r = 0; r < ROT; r++)
    for (int i = 0; i < 4; i++) {
      CK(cudaMalloc(&buf[r][i], bytes[i]));
      fill<<<1024, 256>>>((half_t*)buf[r][i], bytes[i] / 2, 0x9E3779B9u * (i + 1) + 77u * r);
    }
  CK(cudaDeviceSynchronize());
  const int smem = 0;
  CK(cudaFuncSetAttribute(gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  dim3 grid(384, 1), block(32, 8);
  int next = 0;
  auto launch = [&](cudaStream_t st = 0) {
    void** b = buf[next++ % ROT];
    gemv_kernel<<<grid, block, smem, st>>>((const half_t*)b[0], (const signed char*)b[1], (half_t*)b[3], (const half_t*)b[2]);
  };
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT gemv13_np8_warm once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us([&] { launch(); }));
  Stat s = stat_of(v);
  printf("RESULT gemv13_np8_warm kernel median=%.4f iqr=%.4f unit=us n=%d samples=", s.median, s.iqr, s.n);
  for (double x : v) printf("%.3f,", x);
  printf("\n");
  // round 4: the same kernel as inference runs it: B2B launches back-to-back between two events
  // (per launch), and the same B2B launches captured in one CUDA graph (per launch)
  const int B2B = 50;
  std::vector<double> vb, vg;
  for (int r = 0; r < 3 + reps / 2; r++) {
    double x = event_us([&] { for (int i = 0; i < B2B; i++) launch(); }) / B2B;
    if (r >= 3) vb.push_back(x);
  }
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < B2B; i++) launch(st);
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < 3 + reps / 2; r++) {
    CK(cudaEventRecord(e0, st));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r >= 3) vg.push_back(ms * 1e3 / B2B);
  }
  for (auto [tag, w] : {std::pair{"b2b", &vb}, std::pair{"graph", &vg}}) {
    Stat t = stat_of(*w);
    printf("RESULT gemv13_np8_warm %s median=%.4f iqr=%.4f unit=us n=%d per_launch_of=%d\n", tag, t.median, t.iqr, t.n, B2B);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}

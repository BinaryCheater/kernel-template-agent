// Test-set harness (perfmodel/testset/gen.py): TileLang's kernel, launched as TileLang launches it,
// on random inputs (fp16 values in [-1, 1) written over every buffer's bytes); WARM untimed launches,
// then REPS launches timed one by one with CUDA events (no profiler). ROT input sets are used in
// turn (ROT 1: warm; ROT x footprint above the L2: rotating). Usage: harness [reps] (reps 1 = one
// launch after warm-up, for the profiler).
#include <cuda.h>
#include "/o/common.cuh"
#include <tl_templates/cuda/common.h>
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
  const size_t bytes[] = {262144, 4194304};
  void* buf[ROT][2];
  // round 14: reps 0 is the preflight: one launch, each buffer followed by a marked guard, read back
  const size_t GUARD = reps == 0 ? 65536 : 0;
  for (int r = 0; r < ROT; r++)
    for (int i = 0; i < 2; i++) {
      CK(cudaMalloc(&buf[r][i], bytes[i] + GUARD));
      if (GUARD) CK(cudaMemset((char*)buf[r][i] + bytes[i], 0xA5, GUARD));
      fill<<<1024, 256>>>((half_t*)buf[r][i], bytes[i] / 2, 0x9E3779B9u * (i + 1) + 77u * r);
    }
  CK(cudaDeviceSynchronize());
  const int smem = 0;
  dim3 grid(240, 1, 1), block(256);
  int next = 0;
  auto launch = [&](cudaStream_t st = 0) {
    void** b = buf[next++ % ROT];
    k_occ<<<grid, block, smem, st>>>((unsigned*)b[0], ( const unsigned*)b[1]);
  };
  if (reps == 0) {
    launch();
    CK(cudaDeviceSynchronize());
    std::vector<unsigned char> g(GUARD);
    for (int i = 0; i < 2; i++) {
      CK(cudaMemcpy(g.data(), (char*)buf[0][i] + bytes[i], GUARD, cudaMemcpyDeviceToHost));
      for (unsigned char x : g)
        if (x != 0xA5) { printf("PREFLIGHT left buffer %d\n", i); return 3; }
    }
    printf("PREFLIGHT ok\n");
    return 0;
  }
  for (int i = 0; i < 10; i++) launch();
  CK(cudaDeviceSynchronize());
  if (reps == 1) { launch(); CK(cudaDeviceSynchronize()); printf("RESULT occ_t256_k5 once\n"); return 0; }
  std::vector<double> v;
  for (int r = 0; r < reps; r++) v.push_back(event_us([&] { launch(); }));
  Stat s = stat_of(v);
  printf("RESULT occ_t256_k5 kernel median=%.4f iqr=%.4f unit=us n=%d samples=", s.median, s.iqr, s.n);
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
    printf("RESULT occ_t256_k5 %s median=%.4f iqr=%.4f unit=us n=%d per_launch_of=%d\n", tag, t.median, t.iqr, t.n, B2B);
  }
  CK(cudaDeviceSynchronize());
  return 0;
}

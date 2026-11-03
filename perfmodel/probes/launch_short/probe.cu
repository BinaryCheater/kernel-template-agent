// The launch term of short kernels (round 6; STATE.md question 14: combine b2b +59.9 %): the launch
// probe's work kernel (48 blocks x 128 threads of a dependent FFMA chain, code copied) at spans
// 0.1 - 2.5 us, one launch between events, back-to-back and in a graph, per launch. Between spans
// 0.3 us (combine: 1.91 us per launch) and 0.74 us (3.69 us) the back-to-back time leaves the
// submission rate: where, and as a step (a launch that finds the previous kernel still running
// waits for it, then the whole gap) or as a ramp.
#include "../common.cuh"
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
template <int I>
__global__ void work(float* o, unsigned long long* t) {
  unsigned long long g0 = gtime();
  float f = threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < I; i++) f = fmaf(f, 1.0001f, 0.5f);
  if (f == 1.2345f) o[0] = f;
  __syncthreads();
  if (threadIdx.x == 0) t[2 * blockIdx.x] = g0, t[2 * blockIdx.x + 1] = gtime();
}
static unsigned long long* tdev;
template <int I>
static void work_point(float* o) {
  // as the harness: launches timed one by one, nothing in between; the spans stay on the device
  // until all repetitions are done
  const int R = WARM + REPS;
  std::vector<double> ev, sp;
  for (int r = 0; r < R; r++) {
    double e = event_us([&] { work<I><<<48, 128>>>(o, tdev + 96 * r); });
    if (r >= WARM) ev.push_back(e);
  }
  std::vector<unsigned long long> h(96 * R);
  CK(cudaMemcpy(h.data(), tdev, h.size() * sizeof(h[0]), cudaMemcpyDeviceToHost));
  for (int r = WARM; r < R; r++) {
    unsigned long long a = h[96 * r], b = h[96 * r + 1];
    for (int i = 0; i < 48; i++) a = std::min(a, h[96 * r + 2 * i]), b = std::max(b, h[96 * r + 2 * i + 1]);
    sp.push_back((b - a) * 1e-3);
  }
  Stat e = stat_of(ev), s = stat_of(sp);
  char n[32], x[64];
  snprintf(n, sizeof n, "work_i%d", I);
  snprintf(x, sizeof x, "span_us=%.3f event_minus_span=%.3f", s.median, e.median - s.median);
  emit("launch_short", n, e, "us", x);
}
// back-to-back (graph = false) or one graph of 50 launches; per launch, and the spans' median
template <int I>
static void seq_point(float* o, bool graph) {
  const int L = 50, R = 3 + 8;
  static cudaStream_t st = nullptr;
  if (!st) CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraphExec_t ge = nullptr;
  if (graph) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < L; i++) work<I><<<48, 128, 0, st>>>(o, tdev + 96 * i);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> ev, sp;
  for (int r = 0; r < R; r++) {
    CK(cudaEventRecord(e0, st));
    if (graph) CK(cudaGraphLaunch(ge, st));
    else
      for (int i = 0; i < L; i++) work<I><<<48, 128, 0, st>>>(o, tdev + 96 * i);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r >= 3) ev.push_back(ms * 1e3 / L);
  }
  std::vector<unsigned long long> h(96 * L);
  CK(cudaMemcpy(h.data(), tdev, h.size() * sizeof(h[0]), cudaMemcpyDeviceToHost));
  for (int k = 0; k < L; k++) {  // launch k of the last sequence wrote slot k
    unsigned long long a = h[96 * k], b = h[96 * k + 1];
    for (int i = 0; i < 48; i++) a = std::min(a, h[96 * k + 2 * i]), b = std::max(b, h[96 * k + 2 * i + 1]);
    sp.push_back((b - a) * 1e-3);
  }
  Stat e = stat_of(ev), s = stat_of(sp);
  char n[32], x[64];
  snprintf(n, sizeof n, "%s_work_i%d", graph ? "graph" : "b2b", I);
  snprintf(x, sizeof x, "span_us=%.3f per_launch_minus_span=%.3f", s.median, e.median - s.median);
  emit("launch_short", n, e, "us", x);
}
template <int... Is>
static void seq_points(float* o) {
  for (bool g : {false, true}) (seq_point<Is>(o, g), ...);
}
template <int... Is>
static void points(float* o) {
  (work_point<Is>(o), ...);
  seq_points<Is...>(o);
}
int main() {
  float* f;
  CK(cudaMalloc(&f, 4));
  CK(cudaMalloc(&tdev, 96 * std::max(WARM + REPS, 50) * sizeof(unsigned long long)));
  points<0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256>(f);
  return 0;
}

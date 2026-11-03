// Shared host helpers for the perfmodel probes (perfmodel/probes/*/probe.cu).
// Every probe prints one line per sweep point:
//   RESULT <probe> <point> median=<x> iqr=<y> unit=<u> n=<samples> [key=value ...]
// after WARM untimed repetitions, from REPS timed repetitions (medians and IQR, never means;
// gb10-measure). The loops inside the kernels have compile-time trip counts so that the CPU
// replay of the probe's SASS resolves them (sass/sm value tracking).
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x)                                                                    \
  do {                                                                           \
    auto e_ = (x);                                                               \
    if (e_) {                                                                    \
      fprintf(stderr, "cuda error %d (%s) at line %d\n", (int)e_,                \
              cudaGetErrorString((cudaError_t)e_), __LINE__);                    \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)
static const int WARM = 3, REPS = 15;
struct Stat {
  double median, iqr;
  int n;
};
static Stat stat_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return {v[n / 2], v[(3 * n) / 4] - v[n / 4], (int)n};
}
// `once()` runs the kernel and returns one sample (cycles, us, ...)
template <class F>
static Stat repeat(F once) {
  for (int i = 0; i < WARM; i++) once();
  std::vector<double> v;
  for (int i = 0; i < REPS; i++) v.push_back(once());
  return stat_of(v);
}
static void emit(const char* probe, const char* point, Stat s, const char* unit, const char* extra = "") {
  printf("RESULT %s %s median=%.4f iqr=%.4f unit=%s n=%d %s\n", probe, point, s.median, s.iqr, unit, s.n, extra);
  fflush(stdout);
}
// event-timed whole launch in microseconds
template <class L>
static double event_us(L launch) {
  static cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (!e0) {
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
  }
  CK(cudaEventRecord(e0));
  launch();
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  return ms * 1e3;
}

// Launch cost as the test-set harness sees it (round 3; inventory §1 "Launch"): the fitted
// launch_stream_us (2.6 us) is 50-78 % of the small test rows (combine, RMSNorm warm). Every
// test-set time is one launch between two events with a synchronize after it (common.cuh event_us),
// so the probe times exactly that, for kernels with (almost) no work:
//   empty  an empty kernel, grid G x T threads;
//   tiny   each thread stores one int (one STG per warp);
//   b2b    100 empty launches back-to-back between two events, per launch (the stream rate);
//   work   48 blocks x 128 threads of a dependent FFMA chain of I iterations (I = 0 .. 16384, spans 0.1 .. 150 us), its
//          own span from %globaltimer inside (first block entry to last exit) beside the event
//          time: whether the event time is launch + span (additive) or hides part of the span.
// Grids 1 .. 8192 at 128 threads; 48 .. 512 also at 1024 threads.
#include "../common.cuh"
__global__ void empty() {}
__global__ void tiny(int* o) { o[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x; }
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
  emit("launch", n, e, "us", x);
}
int main() {
  int* o;
  CK(cudaMalloc(&o, 4096 * 1024 * sizeof(int)));
  for (int i = 0; i < 100; i++) empty<<<1, 32>>>();
  CK(cudaDeviceSynchronize());
  struct P { int g, t; };
  CK(cudaMalloc(&tdev, 96 * (WARM + REPS) * sizeof(unsigned long long)));
  const P pts[] = {{1, 128},   {48, 128},   {96, 128},   {512, 128},  {1024, 128}, {2048, 128},
                   {4096, 128}, {8192, 128}, {48, 1024}, {128, 1024}, {256, 1024}, {512, 1024}};
  for (P p : pts) {
    char n[48];
    snprintf(n, sizeof n, "empty_g%d_t%d", p.g, p.t);
    emit("launch", n, repeat([&] { return event_us([&] { empty<<<p.g, p.t>>>(); }); }), "us");
    snprintf(n, sizeof n, "tiny_g%d_t%d", p.g, p.t);
    emit("launch", n, repeat([&] { return event_us([&] { tiny<<<p.g, p.t>>>(o); }); }), "us");
  }
  for (P p : {P{1, 128}, P{512, 128}, P{4096, 128}}) {
    char n[48];
    snprintf(n, sizeof n, "b2b_g%d_t%d", p.g, p.t);
    emit("launch", n, repeat([&] {
           return event_us([&] {
                    for (int i = 0; i < 100; i++) empty<<<p.g, p.t>>>();
                  }) / 100;
         }),
         "us");
  }
  float* f;
  CK(cudaMalloc(&f, 4));
  work_point<0>(f);
  work_point<64>(f);
  work_point<128>(f);
  work_point<256>(f);
  work_point<512>(f);
  work_point<1024>(f);
  work_point<4096>(f);
  work_point<16384>(f);
  return 0;
}

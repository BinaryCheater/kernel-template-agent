// Launch cost as the test-set harness sees it (round 3; inventory §1 "Launch"): the fitted
// launch_stream_us (2.6 us) is 50-78 % of the small test rows (combine, RMSNorm warm). Every
// test-set time is one launch between two events with a synchronize after it (common.cuh event_us),
// so the probe times exactly that, for kernels with (almost) no work:
//   empty  an empty kernel, grid G x T threads;
//   tiny   each thread stores one int (one STG per warp);
//   b2b    100 empty launches back-to-back between two events, per launch (the stream rate).
// Grids 1, 48, 96, 512, 1024, 4096 at 128 threads; 48 and 512 also at 1024 threads.
#include "../common.cuh"
__global__ void empty() {}
__global__ void tiny(int* o) { o[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x; }
int main() {
  int* o;
  CK(cudaMalloc(&o, 4096 * 1024 * sizeof(int)));
  for (int i = 0; i < 100; i++) empty<<<1, 32>>>();
  CK(cudaDeviceSynchronize());
  struct P { int g, t; };
  const P pts[] = {{1, 128}, {48, 128}, {96, 128}, {512, 128}, {1024, 128}, {4096, 128}, {48, 1024}, {512, 1024}};
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
  return 0;
}

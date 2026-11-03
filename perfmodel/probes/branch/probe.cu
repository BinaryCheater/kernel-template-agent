// Taken-branch cost (round 6; STATE.md question 4, inventory §1 "Branch and loop resolution"): a
// loop of N independent FFMA per pass, N = 4 .. 512, on 1, 4 (one per sub-partition) and 32 warps
// (8 per sub-partition); straight lines of 64 .. 4096 FFMA as the no-branch control (the smoke run
// gave 1152 cycles for 1024: instruction fetch, inventory §1); a forward branch over 8 FFMA inside
// a loop of 16, taken or not by a run-time mask (`it & z`, z all ones: skip).
// Reported: cycles per pass over the block (clock64 around the loop after a block-wide barrier).
#include "../common.cuh"
constexpr int ITERS = 512;
template <int N>
__device__ __forceinline__ void ops(float* x, float a, float b) {
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(x[i % 8]) : "f"(a), "f"(b));
}
template <int N>
__global__ void loop(unsigned long long* out, float* sink) {
  float x[8], a = threadIdx.x * 1.5f, b = threadIdx.x + 0.5f;
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) ops<N>(x, a, b);
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) s += x[i];
  if (s == 1.2345f) sink[0] = s;
}
template <int N>
__global__ void flat(unsigned long long* out, float* sink) {
  float x[8], a = threadIdx.x * 1.5f, b = threadIdx.x + 0.5f;
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll
  for (int i = 0; i < N / 64; i++) ops<64>(x, a, b);
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) s += x[i];
  if (s == 1.2345f) sink[0] = s;
}
__global__ void fwd(unsigned z, unsigned long long* out, float* sink) {
  float x[8], a = threadIdx.x * 1.5f, b = threadIdx.x + 0.5f;
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (unsigned it = 1; it <= ITERS; it++) {
    ops<8>(x, a, b);
    if ((it & z) == 0) ops<8>(x, a, b);
    ops<8>(x, a, b);
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) s += x[i];
  if (s == 1.2345f) sink[0] = s;
}
static unsigned long long* o;
static float* sink;
template <class F>
static void point(const char* name, int warps, double per, F launch) {
  unsigned long long h;
  Stat s = repeat([&] {
    launch(32 * warps);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
    return (double)h / per;
  });
  char p[48];
  snprintf(p, sizeof p, "%s_w%d", name, warps);
  emit("branch", p, s, "cycles");
}
template <int N>
static void sweep() {
  char n[16];
  snprintf(n, sizeof n, "loop%d", N);
  for (int w : {1, 4, 32}) point(n, w, ITERS, [&](int t) { loop<N><<<1, t>>>(o, sink); });
}
template <int N>
static void flats() {
  char n[16];
  snprintf(n, sizeof n, "flat%d", N);
  for (int w : {1, 2, 4, 8, 32}) point(n, w, N, [&](int t) { flat<N><<<1, t>>>(o, sink); });
}
int main() {
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  sweep<4>();
  sweep<8>();
  sweep<16>();
  sweep<32>();
  sweep<64>();
  sweep<128>();
  sweep<256>();
  sweep<512>();
  sweep<1024>();
  flats<64>();
  flats<256>();
  flats<1024>();
  flats<4096>();
  for (int w : {1, 4, 32}) {
    point("fwd_taken", w, ITERS, [&](int t) { fwd<<<1, t>>>(0xffffffffu, o, sink); });
    point("fwd_not", w, ITERS, [&](int t) { fwd<<<1, t>>>(0u, o, sink); });
  }
}

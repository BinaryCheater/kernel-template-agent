// Block barriers (inventory §7 "BAR.SYNC release 13 + 2 per warp": base assumed; "BAR.ARV, BAR.RED":
// absent, every BAR* treated as blocking). Two parts:
//  sync_wW:   W warps run ITERS back-to-back bar.sync 0: cycles per barrier.
//  arrive_LL: 8 warps, producer/consumer on named barriers (the FA-forward / warp-specialised
//             pattern). Per pass: consumers (warps 4-7) run a dependent chain of L FFMAs, then
//             bar.sync 1, 256, then bar.arrive 2, 256; producers (warps 0-3) read clock64, bar.arrive
//             1, 256, read clock64, then bar.sync 2, 256 (which keeps barrier generations apart).
//             Reported: producer warp 0's cycles from before to after its bar.arrive, per pass.
//             L in {0, 64, 256}.
#include "../common.cuh"
constexpr int ITERS = 128;
__global__ void sync_k(unsigned long long* out) {
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) asm volatile("bar.sync 0;" ::: "memory");
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
}
template <int L>
__global__ void arrive_k(unsigned long long* out, float* sink) {
  int warp = threadIdx.x / 32;
  float a = threadIdx.x;
  long long acc = 0;
  __syncthreads();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
    if (warp >= 4) {
#pragma unroll
      for (int j = 0; j < L; j++) a = fmaf(a, 0.999f, 0.5f);
      asm volatile("bar.sync 1, 256;" ::: "memory");
      asm volatile("bar.arrive 2, 256;" ::: "memory");
    } else {
      long long t0 = clock64();
      asm volatile("bar.arrive 1, 256;" ::: "memory");
      long long t1 = clock64();
      acc += t1 - t0;
      asm volatile("bar.sync 2, 256;" ::: "memory");
    }
  }
  if (threadIdx.x == 0) out[0] = acc;
  if (a == 0.f) sink[0] = a;
}
int main() {
  unsigned long long* o;
  float* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  unsigned long long h;
  for (int w : {2, 4, 8, 16, 32}) {
    Stat s = repeat([&] {
      sync_k<<<1, 32 * w>>>(o);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / ITERS;
    });
    char p[32];
    snprintf(p, sizeof p, "sync_w%d", w);
    emit("namedbar", p, s, "cycles/barrier");
  }
  auto arr = [&](auto kern, const char* p) {
    Stat s = repeat([&] {
      kern<<<1, 256>>>(o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / ITERS;
    });
    emit("namedbar", p, s, "cycles/arrive");
  };
  arr(arrive_k<0>, "arrive_L0");
  arr(arrive_k<64>, "arrive_L64");
  arr(arrive_k<256>, "arrive_L256");
}

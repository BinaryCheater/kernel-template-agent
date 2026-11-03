// MUFU.EX2 beside FFMA from the same warps (inventory §2 "ex2 with FFMAs between them": absent;
// the FA-forward softmax pattern). 32 warps on one SM (8 per sub-partition); per loop pass every
// thread runs E independent ex2.approx.ftz chains and F independent FFMA chains, interleaved.
// (E, F) in {(8,0), (8,8), (8,16), (8,32), (0,32), (4,32)}. Reported: cycles per loop pass per
// sub-partition (clock64 of warp 0 around the loop / ITERS, all 8 warps of a sub-partition share it).
#include "../common.cuh"
constexpr int ITERS = 256;
template <int E, int F>
__global__ void k(unsigned long long* out, float* sink) {
  float e[E > 0 ? E : 1], f[F > 0 ? F : 1];
#pragma unroll
  for (int j = 0; j < (E > 0 ? E : 1); j++) e[j] = -1e-3f * (threadIdx.x + j);
#pragma unroll
  for (int j = 0; j < (F > 0 ? F : 1); j++) f[j] = threadIdx.x + j;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int j = 0; j < (E > F ? E : F); j++) {
      if (j < E) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(e[j]));
      if (j < F) f[j] = fmaf(f[j], 0.9999f, 0.25f);
    }
  }
  __syncthreads();  // every warp done: the block's time, not warp 0's
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int j = 0; j < E; j++) s += e[j];
#pragma unroll
  for (int j = 0; j < F; j++) s += f[j];
  if (s == 1.2345f) sink[0] = s;
}
int main() {
  unsigned long long* o;
  float* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  unsigned long long h;
  auto one = [&](auto kern, const char* p) {
    Stat s = repeat([&] {
      kern<<<1, 1024>>>(o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / ITERS;
    });
    emit("ex2_ffma", p, s, "cycles/pass/smsp");
  };
  one(k<8, 0>, "E8_F0");
  one(k<8, 8>, "E8_F8");
  one(k<8, 16>, "E8_F16");
  one(k<8, 32>, "E8_F32");
  one(k<0, 32>, "E0_F32");
  one(k<4, 32>, "E4_F32");
}

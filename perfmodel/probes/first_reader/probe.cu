// A second reader of a line that is on its way (questions 8, 27, 30; inventory §5: L1, L2 and DRAM
// latency, no miss-status entry yet). One block of 4 warps. Per step, after a block barrier, warp 1
// (the lead) reads 4 lines, warp 0 (the follower) waits Z FFMA passes and reads the same 4 lines,
// timed by its own clock around the load. LEAD 0: the follower alone, the control at every Z.
// KIND 0 plain LDG (through the L1; `volatile` loads go past it, r9), KIND 1 cp.async (past the L1).
// rot: a 45 MB read between launches puts the lines out of the L2 (r9's walk wrapped and re-read
// them); warm: the same lines every launch.
// Reported: the follower's cycles per step.
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 128, STEPS = 64, STEP_BYTES = 512;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpwait() {
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
// a read of every third line of 45 MB between timed launches: what a rotating launch reads is out of the L2
__global__ void __launch_bounds__(1024, 1) flush(const char* p, unsigned* sink) {
  const size_t i = (size_t)(blockIdx.x * 1024 + threadIdx.x) * 384;
  unsigned s = 0;
  for (int k = 0; k < 3; k++) {
    unsigned v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p + i + k * 128) : "memory");
    s += v;
  }
  if (s == 0x12345) sink[0] = s;
}
__device__ __forceinline__ float ld(const char* p) {
  float v;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
template <int Z, int KIND, int LEAD>
__global__ void __launch_bounds__(T, 1) k(u64* out, const char* src, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31;
  float f = t;
  u64 acc = 0;
#pragma unroll 1
  for (int s = 0; s < STEPS; s++) {
    __syncthreads();
    const char* p = src + s * STEP_BYTES + lane * 16;
    if (w == 1 && LEAD) {
      if (KIND) {
        cp16(sm + 512 + lane * 16, p);
        cpwait();
      } else {
        f += ld(p);
      }
    }
    if (w == 0) {
#pragma unroll 1
      for (int z = 0; z < Z; z++) f = fmaf(f, 0.9999f, 0.25f);
      u64 c0 = clock64();
      if (KIND) {
        cp16(sm + lane * 16, p);
        cpwait();
      } else {
        float v = ld(p);
        if (v == 1.2345f) sink[0] = v;
      }
      acc += clock64() - c0;
    }
  }
  if (t == 0) out[0] = acc;
  if (f == 1.2345f) sink[0] = f;
}
int main() {
  u64* o;
  char* src;
  char* other;
  float* sink;
  const size_t W = 1 << 20, F = 48 << 20;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&src, W));
  CK(cudaMalloc(&other, F));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, W));
  CK(cudaMemset(other, 1, F));
  auto point = [&](auto kern, const char* kind, int z, int lead, bool rot) {
    Stat s = repeat([&] {
      if (rot) flush<<<112, 1024>>>(other, (unsigned*)sink);
      kern<<<1, T, 1024>>>(o, src, sink);
      CK(cudaDeviceSynchronize());
      u64 h;
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / STEPS;
    });
    char p[48];
    snprintf(p, sizeof p, "%s_%s_z%d_%s", rot ? "rot" : "warm", kind, z, lead ? "lead" : "alone");
    emit("first_reader", p, s, "cycles");
  };
  for (int rot : {0, 1}) {
#define PT(Z)                                 \
  point(k<Z, 0, 1>, "ldg", Z, 1, rot);        \
  point(k<Z, 0, 0>, "ldg", Z, 0, rot);        \
  point(k<Z, 1, 1>, "cp", Z, 1, rot);         \
  point(k<Z, 1, 0>, "cp", Z, 0, rot);
    PT(0) PT(4) PT(8) PT(16) PT(24) PT(32) PT(48) PT(64)
  }
}

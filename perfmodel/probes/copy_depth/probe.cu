// How many cp.async a warp and an SM may have in flight before issue waits (LOOP contention group;
// inventory §4 LSU queue depth). One block of W warps; each issues K 512-byte copies with no wait,
// then a commit and wait 0. Warm: 8 MiB tile; rot: a 48 MB read before the launch. Reported: warp
// 0's cycles to issue K copies, and to the last landing. In the replay issue never waits on what
// is in flight.
#include "../common.cuh"
typedef unsigned long long u64;
constexpr size_t TILE = 8 << 20;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
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
template <int K, int T>
__global__ void __launch_bounds__(T, 1) k(u64* out, const char* src) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31;
  __syncthreads();
  u64 c0 = clock64();
#pragma unroll 1
  for (int i = 0; i < K; i++)
    cp16(sm + (i & 15) * 512 + lane * 16, src + ((w * 1024 + i) * 512) % TILE + lane * 16);
  u64 c1 = clock64();
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  u64 c2 = clock64();
  if (t == 0) {
    out[0] = c1 - c0;
    out[1] = c2 - c0;
  }
}
int main() {
  u64* o;
  char* src;
  char* other;
  unsigned* sink;
  const size_t F = 48 << 20;
  CK(cudaMalloc(&o, 16));
  CK(cudaMalloc(&src, TILE));
  CK(cudaMalloc(&other, F));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, TILE));
  CK(cudaMemset(other, 1, F));
  auto point = [&](auto kern, int key, int t, bool rot) {
    double total = 0;
    Stat s = repeat([&] {
      if (rot) flush<<<112, 1024>>>(other, sink);
      kern<<<1, t, 8192>>>(o, src);
      CK(cudaDeviceSynchronize());
      u64 h[2];
      CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
      total = (double)h[1];
      return (double)h[0];
    });
    char p[48], x[48];
    snprintf(p, sizeof p, "%s_t%d_k%d", rot ? "rot" : "warm", t, key);
    snprintf(x, sizeof x, "landed=%.0f", total);
    emit("copy_depth", p, s, "cycles", x);
    snprintf(p, sizeof p, "%s_t%d_k%d_landed", rot ? "rot" : "warm", t, key);
    emit("copy_depth", p, Stat{total, 0, 1}, "cycles");
  };
  for (int rot : {0, 1}) {
#define PT(K)                             \
  point(k<K, 32>, K, 32, rot);            \
  point(k<K, 128>, K, 128, rot);
    PT(4) PT(8) PT(16) PT(32) PT(64) PT(128) PT(256)
  }
}

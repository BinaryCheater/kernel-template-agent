// How many cp.async an SM holds in flight with one and with two blocks (question 47, D864): the
// fit `copy_inflight_sm` says 56, with two blocks only. B blocks per SM (48 B blocks, 48 KiB of
// shared memory each), each of T threads; every warp issues K 512-byte copies from a warm 8 MiB
// tile with no wait, then a commit and wait, then a barrier; R rounds. Reported: the median over
// blocks and rounds of warp 0's cycles to issue K copies and to the last landing.
#include "../common.cuh"
#include <algorithm>
#include <vector>
typedef unsigned u32;
constexpr size_t TILE = 8 << 20;
constexpr int R = 16, SMEM = 48 << 10;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
template <int K, int T>
__global__ void __launch_bounds__(T, 2) k(u32* out, const char* src) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31;
  u32 a[R], b[R];
  for (int r = 0; r < R; r++) {
    __syncthreads();
    const unsigned long long c0 = clock64();
#pragma unroll 1
    for (int i = 0; i < K; i++) {
      const size_t at = ((size_t)(blockIdx.x * 4 + w) * 64 + r * K + i) * 512 + lane * 16;
      cp16(sm + (i & 63) * 512 + lane * 16, src + at % TILE);
    }
    const unsigned long long c1 = clock64();
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const unsigned long long c2 = clock64();
    a[r] = (u32)(c1 - c0);
    b[r] = (u32)(c2 - c0);
  }
  if (t == 0)
    for (int r = 0; r < R; r++) {
      out[(blockIdx.x * R + r) * 2] = a[r];
      out[(blockIdx.x * R + r) * 2 + 1] = b[r];
    }
}
int main() {
  u32* o;
  char* src;
  const int G = 96;
  CK(cudaMalloc(&o, G * R * 8));
  CK(cudaMalloc(&src, TILE));
  CK(cudaMemset(src, 1, TILE));
  auto point = [&](auto kern, int key, int t, int blocks) {
    std::vector<u32> h(G * R * 2);
    double landed = 0;
    Stat s = repeat([&] {
      kern<<<blocks * 48, t, SMEM>>>(o, src);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), o, h.size() * 4, cudaMemcpyDeviceToHost));
      std::vector<u32> iss, land;
      for (int i = 0; i < blocks * 48 * R; i++) {
        if (i % R == 0) continue;  // the first round of a block starts cold in the warp's state
        iss.push_back(h[i * 2]);
        land.push_back(h[i * 2 + 1]);
      }
      std::sort(iss.begin(), iss.end());
      std::sort(land.begin(), land.end());
      landed = land[land.size() / 2];
      return (double)iss[iss.size() / 2];
    });
    char p[48], x[48];
    snprintf(p, sizeof p, "b%d_t%d_k%d", blocks, t, key);
    snprintf(x, sizeof x, "landed=%.0f", landed);
    emit("copy_two", p, s, "cycles", x);
    snprintf(p, sizeof p, "b%d_t%d_k%d_landed", blocks, t, key);
    emit("copy_two", p, Stat{landed, 0, 1}, "cycles");
  };
#define P(K, T, B) point(k<K, T>, K, T, B)
  P(2, 128, 1); P(4, 128, 1); P(6, 128, 1); P(8, 128, 1); P(12, 128, 1);
  P(16, 128, 1); P(24, 128, 1); P(32, 128, 1); P(48, 128, 1); P(64, 128, 1);
  P(2, 128, 2); P(4, 128, 2); P(6, 128, 2); P(8, 128, 2); P(12, 128, 2);
  P(16, 128, 2); P(24, 128, 2); P(32, 128, 2); P(48, 128, 2); P(64, 128, 2);
  P(32, 160, 1); P(32, 192, 1); P(32, 224, 1);  // 5 to 7 warps: per SM or per sub-partition
  P(4, 32, 1); P(16, 32, 1); P(64, 32, 1); P(4, 32, 2); P(16, 32, 2); P(64, 32, 2);
  return 0;
}

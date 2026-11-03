// DRAM reads and writes together (round 4; inventory §5 "DRAM write rate and read/write mix"): the
// rotating rows that write as much as they read (RMSNorm rotating: X in, Y out) run at 207 GB/s on
// the GPU and above the DRAM plateau in the replay, which counts only the reads. A streaming kernel
// reads NR planes and writes NW planes of 30 MiB each, element for element, 16-byte accesses, one
// access per plane in flight per thread (#pragma unroll 1), 96 blocks x 512 threads; four sets of
// planes in turn (480 MiB in rotation > the L2: every read misses, every written line goes back).
// Mixes read:write 1:0 (2,0), 2:1 (2,1), 1:1 (1,1), 1:2 (1,2), 0:1 (0,2). Reported: event time per
// launch, total, read and write GB/s.
#include "../common.cuh"
constexpr int GRID = 96 * 512, STEPS = 40;
constexpr size_t UN = (size_t)GRID * STEPS;  // int4 per plane (30 MiB)
template <int NR, int NW>
__global__ void __launch_bounds__(512) rw(const int4* __restrict__ a, int4* __restrict__ b, int* sink) {
  const int t = blockIdx.x * 512 + threadIdx.x;
  int acc = 0;
#pragma unroll 1
  for (int i = 0; i < STEPS; i++) {
    const size_t k = (size_t)i * GRID + t;
    int4 v = make_int4(t, i, t ^ i, 7);
    if (NR >= 1) {
      int4 x = a[k];
      v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
    }
    if (NR >= 2) {
      int4 x = a[UN + k];
      v.x ^= x.x, v.y ^= x.y, v.z ^= x.z, v.w ^= x.w;
    }
    if (NW >= 1) b[k] = v;
    if (NW >= 2) b[UN + k] = v;
    acc ^= v.x;
  }
  if (NW == 0 && acc == 0x7fff1234) sink[0] = acc;
}
int main() {
  int4 *src[4], *dst[4];
  int* sink;
  for (int r = 0; r < 4; r++) {
    CK(cudaMalloc(&src[r], 2 * UN * 16));
    CK(cudaMalloc(&dst[r], 2 * UN * 16));
    CK(cudaMemset(src[r], 1, 2 * UN * 16));
    CK(cudaMemset(dst[r], 0, 2 * UN * 16));
  }
  CK(cudaMalloc(&sink, 4));
  auto one = [&](auto kern, int nr, int nw) {
    int r = 0;
    Stat s = repeat([&] {
      return event_us([&] {
        kern<<<96, 512>>>(src[r % 4], dst[r % 4], sink);
        r++;
      });
    });
    char p[32], x[96];
    snprintf(p, sizeof p, "r%d_w%d", nr, nw);
    double rb = nr * UN * 16.0, wb = nw * UN * 16.0;
    snprintf(x, sizeof x, "GBs=%.1f read_GBs=%.1f write_GBs=%.1f", (rb + wb) / (s.median * 1e3),
             rb / (s.median * 1e3), wb / (s.median * 1e3));
    emit("dram_rw", p, s, "us", x);
  };
  one(rw<2, 0>, 2, 0);
  one(rw<2, 1>, 2, 1);
  one(rw<1, 1>, 1, 1);
  one(rw<1, 2>, 1, 2);
  one(rw<0, 2>, 0, 2);
}

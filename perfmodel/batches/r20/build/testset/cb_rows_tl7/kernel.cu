// round 20: copies of 16 bytes into shared memory, lanes on a bank by the layout tl7
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const char* a) {
  __shared__ __align__(128) unsigned char buf[65536];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const unsigned t = threadIdx.x;
  unsigned s = 0;
  for (unsigned q = 0; q < 4 * 8; ++q) {
    const unsigned r = q % 4;
    for (unsigned k = 0; k < 7; ++k) {
      const unsigned x = k * 64 + (t >> 1);
      const char* p = a + (size_t)(blockIdx.x * 4 + r) * 65536 + ((x / 7) * 448 + ((k * 128 + t) % 14) * 8) * 2;
      const unsigned at = sm + (((x % 7) * 2048 + (x / 7) * 32 + ((((k * 16 + (t >> 3)) % 14) / 7 + (t & 1)) & 1) * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(at), "l"(p) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  o[blockIdx.x] = t ^ s;
}

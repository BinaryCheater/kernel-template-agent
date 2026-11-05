// round 16: copies of 16 bytes per thread, rows 48 bytes a thread
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const char* a) {
  __shared__ __align__(128) unsigned char buf[32768];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const unsigned t = threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    const char* p = a + ((size_t)(blockIdx.x * 128 + t) * 8 + k) * 48;
    const unsigned at = sm + (t * 16 + k * 2048);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(at), "l"(p) : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  o[blockIdx.x] = t;
}

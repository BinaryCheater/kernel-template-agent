#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float C_local[128];
  float t[1];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((i_1 * 1024) + (((int)threadIdx.x) * 8))])), (&(A[((((((int)blockIdx.y) * 262144) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((i_2 * 1024) + (((int)threadIdx.x) * 8))])), (&(B[((((i_2 * 12288) + ((((int)threadIdx.x) >> 4) * 1536)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((i_4 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(B[(((((i_4 * 12288) + ((((int)threadIdx.x) >> 4) * 1536)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 49152)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 62; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((ko + 2) % 3) * 4096) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))])), (&(A[((((((((int)blockIdx.y) * 262144) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((ko + 2) % 3) * 4096) + (i_6 * 1024)) + (((int)threadIdx.x) * 8))])), (&(B[((((((ko * 49152) + (i_6 * 12288)) + ((((int)threadIdx.x) >> 4) * 1536)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 98304)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    t[0] = (((float)((half_t*)A_shared)[((ko % 3) * 4096)]) + ((float)((half_t*)B_shared)[((ko % 3) * 4096)]));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  t[0] = (((float)((half_t*)A_shared)[8192]) + ((float)((half_t*)B_shared)[8192]));
  tl::cp_async_wait<0>();
  __syncthreads();
  t[0] = (((float)((half_t*)A_shared)[0]) + ((float)((half_t*)B_shared)[0]));
  #pragma unroll
  for (int i_7 = 0; i_7 < 32; ++i_7) {
    *(float4*)(C_local + (i_7 * 4)) = make_float4(t[0], t[0], t[0], t[0]);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 32; ++i_8) {
    half_t C_local_cast[4];
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i_8 * 4));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + (((((((int)blockIdx.y) * 196608) + (i_8 * 6144)) + ((((int)threadIdx.x) >> 5) * 1536)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}


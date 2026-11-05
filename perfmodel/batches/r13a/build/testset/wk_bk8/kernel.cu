#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 6144));
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((int)threadIdx.x) * 8)])), (&(A[((((int)blockIdx.y) * 262144) + (((int)threadIdx.x) * 2048))])));
  tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int)threadIdx.x) >> 4) * 2048) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((int)threadIdx.x) * 8) + 1024)])), (&(A[(((((int)blockIdx.y) * 262144) + (((int)threadIdx.x) * 2048)) + 8)])));
  tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 1024)])), (&(B[(((((((int)threadIdx.x) >> 4) * 2048) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 16384)])));
  tl::cp_async_commit();
  for (int ko = 0; ko < 254; ++ko) {
    __syncthreads();
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((ko + 2) % 3) * 1024) + (((int)threadIdx.x) * 8))])), (&(A[((((((int)blockIdx.y) * 262144) + (((int)threadIdx.x) * 2048)) + (ko * 8)) + 16)])));
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((ko + 2) % 3) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[(((((ko * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local[16];
      half_t B_local[16];
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((ko % 3) * 1024) + ((((((((((int)threadIdx.x) & 63) >> 5) * 64) + (i_1 * 16)) + ((((int)threadIdx.x) & 31) >> 4)) + (((int)threadIdx.x) & 15)) & 127) * 8))])), (&(A_local[(i_1 * 4)])));
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 4; ++i_2) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((ko % 3) * 1024) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_2 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 1023) >> 9) + (i_2 >> 1)) & 1) * 32)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_2 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 511) >> 8) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 4)])));
      }
      for (int i_3 = 0; i_3 < 4; ++i_3) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 4)), reinterpret_cast<const unsigned*>(B_local + (j * 4)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 4)), reinterpret_cast<const unsigned*>(B_local + ((j * 4) + 2)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[16];
    half_t B_local_1[16];
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 64) + (i_4 * 16)) + ((((int)threadIdx.x) & 31) >> 4)) + (((int)threadIdx.x) & 15)) & 127) * 8) + 2048)])), (&(A_local_1[(i_4 * 4)])));
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 64)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_5 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 1023) >> 9) + (i_5 >> 1)) & 1) * 32)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_5 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 511) >> 8) + (i_5 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(B_local_1[(i_5 * 4)])));
    }
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + ((i_6 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_6 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 4)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + (((i_6 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_6 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 4) + 2)));
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_2[16];
    half_t B_local_2[16];
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 64) + (i_7 * 16)) + ((((int)threadIdx.x) & 31) >> 4)) + (((int)threadIdx.x) & 15)) & 127) * 8)])), (&(A_local_2[(i_7 * 4)])));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((int)threadIdx.x) >> 6) * 512) + ((((int)threadIdx.x) & 7) * 64)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_8 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 1023) >> 9) + (i_8 >> 1)) & 1) * 32)) + ((((((((((((int)threadIdx.x) & 15) * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (i_8 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) & 511) >> 8) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_8 * 4)])));
    }
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + ((i_9 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_9 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 4)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 8, false, true>(reinterpret_cast<float*>(C_local + (((i_9 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_9 * 4)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 4) + 2)));
      }
    }
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 64; ++i_10) {
    half_t C_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_10 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 262144) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_10 >> 4) * 32768)) + ((i_10 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_10 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}


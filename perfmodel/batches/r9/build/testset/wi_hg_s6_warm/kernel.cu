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
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 262144) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((i_2 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B[(((((i_4 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_5 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[(((((i_6 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_7 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])));
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 4; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(B[(((((i_8 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 196608)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_9 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 128)])));
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 4; ++i_10) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_10 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B[(((((i_10 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 59; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((((ko + 5) % 6) * 4096) + (i_11 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 262144) + (i_11 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 160)])));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((((((ko + 5) % 6) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_12 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((ko * 65536) + (i_12 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 327680)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<5>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((ko % 6) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_13 * 8)])));
        }
        #pragma unroll
        for (int i_14 = 0; i_14 < 4; ++i_14) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((ko % 6) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_14 * 8)])));
        }
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_16 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 20480)])), (&(A_local_1[(i_16 * 8)])));
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_17 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20480)])), (&(B_local_1[(i_17 * 8)])));
      }
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_18 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_18 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_18 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_19 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_2[(i_19 * 8)])));
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_20 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_20 * 8)])));
      }
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_21 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_21 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t A_local_3[32];
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_22 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_3[(i_22 * 8)])));
      }
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_23 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_23 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(B_local_3[(i_23 * 8)])));
      }
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_24 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_24 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_24 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_4[32];
    half_t B_local_4[32];
    for (int ki_4 = 0; ki_4 < 2; ++ki_4) {
      #pragma unroll
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_25 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 8192)])), (&(A_local_4[(i_25 * 8)])));
      }
      #pragma unroll
      for (int i_26 = 0; i_26 < 4; ++i_26) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_4 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_26 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_4[(i_26 * 8)])));
      }
      for (int i_27 = 0; i_27 < 4; ++i_27) {
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_27 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_27 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_5[32];
    half_t B_local_5[32];
    for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
      #pragma unroll
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_28 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_5) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12288)])), (&(A_local_5[(i_28 * 8)])));
      }
      #pragma unroll
      for (int i_29 = 0; i_29 < 4; ++i_29) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 2048) + (ki_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_29 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(B_local_5[(i_29 * 8)])));
      }
      for (int i_30 = 0; i_30 < 4; ++i_30) {
        for (int j_5 = 0; j_5 < 4; ++j_5) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_30 * 32) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_5 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_30 * 32) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + (i_30 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 64; ++i_31) {
    half_t C_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_31 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 262144) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_31 >> 4) * 32768)) + ((i_31 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_31 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}


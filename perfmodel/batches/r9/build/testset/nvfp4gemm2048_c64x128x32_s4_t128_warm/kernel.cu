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

extern "C" __global__ void gemm_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Bq_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Sc_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 10240));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 26624));
  float C_local[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((int)threadIdx.x) * 16)])), (&(Bq[((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024))])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[(((int)threadIdx.x) * 2)])), (&(Sc[((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128))])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 131072) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 2048)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 16)])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[((((int)threadIdx.x) * 2) + 256)])), (&(Sc[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + 2)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 4096)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 32)])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[((((int)threadIdx.x) * 2) + 512)])), (&(Sc[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + 4)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 6144)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 48)])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[((((int)threadIdx.x) * 2) + 768)])), (&(Sc[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + 6)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_4 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6144)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_4 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 60; ++ko) {
    tl::cp_async_wait<7>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      signed char Bq_shared_local_cast_1[2];
      half_t Sc_shared_local_cast_2[2];
      half_t B_shared_local_cast[2];
      *(char2*)(Bq_shared_local_cast_1 + 0) = make_char2(((signed char*)Bq_shared)[((((ko & 3) * 2048) + (i_5 * 128)) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((((ko & 3) * 2048) + (i_5 * 128)) + ((int)threadIdx.x))]);
      *(uint1*)(Sc_shared_local_cast_2 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((((ko & 3) * 256) + (i_5 * 16)) + (((int)threadIdx.x) >> 3))], ((half_t*)Sc_shared)[((((ko & 3) * 256) + (i_5 * 16)) + (((int)threadIdx.x) >> 3))]));
      for (int vec = 0; vec < 2; ++vec) {
        half_t condval;
        if ((8 <= ((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15))) {
          int condval_1;
          if ((0 < ((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) >> 1) & 3))) {
            condval_1 = (((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) >> 1) & 3) - 1));
          } else {
            condval_1 = (((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) & 1);
          }
          condval = ((((half_t)condval_1) * half_t(0x1p-1f/*5.000000e-01*/)) * half_t(-0x1p+0f/*-1.000000e+00*/));
        } else {
          int condval_2;
          if ((0 < ((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) >> 1) & 3))) {
            condval_2 = (((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) >> 1) & 3) - 1));
          } else {
            condval_2 = (((((int)Bq_shared_local_cast_1[vec]) >> (vec * 4)) & 15) & 1);
          }
          condval = (((half_t)condval_2) * half_t(0x1p-1f/*5.000000e-01*/));
        }
        B_shared_local_cast[vec] = (condval * Sc_shared_local_cast_2[vec]);
      }
      *(uint1*)(((half_t*)B_shared) + (((((i_5 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast + 0);
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((ko & 3) * 2048) + (((int)threadIdx.x) * 16))])), (&(Bq[((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (ko * 16)) + 64)])));
    tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[(((ko & 3) * 256) + (((int)threadIdx.x) * 2))])), (&(Sc[((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + (ko * 2)) + 8)])));
    tl::cp_async_commit();
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      half_t A_local[16];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((ko & 3) * 2048) + (((((int)threadIdx.x) & 63) >> 5) * 1024)) + (i_6 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_6 * 8)])));
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_7 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
        }
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_8 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_8 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((((ko & 3) * 2048) + (i_9 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 131072) + (i_9 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 128)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  #pragma unroll
  for (int i_10 = 0; i_10 < 16; ++i_10) {
    signed char Bq_shared_local_cast_4[2];
    half_t Sc_shared_local_cast_5[2];
    half_t B_shared_local_cast_3[2];
    *(char2*)(Bq_shared_local_cast_4 + 0) = make_char2(((signed char*)Bq_shared)[((i_10 * 128) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((i_10 * 128) + ((int)threadIdx.x))]);
    *(uint1*)(Sc_shared_local_cast_5 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((i_10 * 16) + (((int)threadIdx.x) >> 3))], ((half_t*)Sc_shared)[((i_10 * 16) + (((int)threadIdx.x) >> 3))]));
    for (int vec_1 = 0; vec_1 < 2; ++vec_1) {
      half_t condval_3;
      if ((8 <= ((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15))) {
        int condval_4;
        if ((0 < ((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) >> 1) & 3))) {
          condval_4 = (((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_4 = (((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) & 1);
        }
        condval_3 = ((((half_t)condval_4) * half_t(0x1p-1f/*5.000000e-01*/)) * half_t(-0x1p+0f/*-1.000000e+00*/));
      } else {
        int condval_5;
        if ((0 < ((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) >> 1) & 3))) {
          condval_5 = (((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_5 = (((((int)Bq_shared_local_cast_4[vec_1]) >> (vec_1 * 4)) & 15) & 1);
        }
        condval_3 = (((half_t)condval_5) * half_t(0x1p-1f/*5.000000e-01*/));
      }
      B_shared_local_cast_3[vec_1] = (condval_3 * Sc_shared_local_cast_5[vec_1]);
    }
    *(uint1*)(((half_t*)B_shared) + (((((i_10 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_3 + 0);
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  {
    half_t A_local_1[16];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_11 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[(i_11 * 8)])));
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_12 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
      }
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_13 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_13 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_13 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 16; ++i_14) {
    signed char Bq_shared_local_cast_7[2];
    half_t Sc_shared_local_cast_8[2];
    half_t B_shared_local_cast_6[2];
    *(char2*)(Bq_shared_local_cast_7 + 0) = make_char2(((signed char*)Bq_shared)[(((i_14 * 128) + ((int)threadIdx.x)) + 2048)], ((signed char*)Bq_shared)[(((i_14 * 128) + ((int)threadIdx.x)) + 2048)]);
    *(uint1*)(Sc_shared_local_cast_8 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_14 * 16) + (((int)threadIdx.x) >> 3)) + 256)], ((half_t*)Sc_shared)[(((i_14 * 16) + (((int)threadIdx.x) >> 3)) + 256)]));
    for (int vec_2 = 0; vec_2 < 2; ++vec_2) {
      half_t condval_6;
      if ((8 <= ((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15))) {
        int condval_7;
        if ((0 < ((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) >> 1) & 3))) {
          condval_7 = (((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_7 = (((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) & 1);
        }
        condval_6 = ((((half_t)condval_7) * half_t(0x1p-1f/*5.000000e-01*/)) * half_t(-0x1p+0f/*-1.000000e+00*/));
      } else {
        int condval_8;
        if ((0 < ((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) >> 1) & 3))) {
          condval_8 = (((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_8 = (((((int)Bq_shared_local_cast_7[vec_2]) >> (vec_2 * 4)) & 15) & 1);
        }
        condval_6 = (((half_t)condval_8) * half_t(0x1p-1f/*5.000000e-01*/));
      }
      B_shared_local_cast_6[vec_2] = (condval_6 * Sc_shared_local_cast_8[vec_2]);
    }
    *(uint1*)(((half_t*)B_shared) + (((((i_14 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_6 + 0);
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t A_local_2[16];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 2; ++i_15) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_15 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 2048)])), (&(A_local_2[(i_15 * 8)])));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_16 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_2[(i_16 * 8)])));
      }
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_17 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_17 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_18 = 0; i_18 < 16; ++i_18) {
    signed char Bq_shared_local_cast_10[2];
    half_t Sc_shared_local_cast_11[2];
    half_t B_shared_local_cast_9[2];
    *(char2*)(Bq_shared_local_cast_10 + 0) = make_char2(((signed char*)Bq_shared)[(((i_18 * 128) + ((int)threadIdx.x)) + 4096)], ((signed char*)Bq_shared)[(((i_18 * 128) + ((int)threadIdx.x)) + 4096)]);
    *(uint1*)(Sc_shared_local_cast_11 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_18 * 16) + (((int)threadIdx.x) >> 3)) + 512)], ((half_t*)Sc_shared)[(((i_18 * 16) + (((int)threadIdx.x) >> 3)) + 512)]));
    for (int vec_3 = 0; vec_3 < 2; ++vec_3) {
      half_t condval_9;
      if ((8 <= ((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15))) {
        int condval_10;
        if ((0 < ((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) >> 1) & 3))) {
          condval_10 = (((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_10 = (((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) & 1);
        }
        condval_9 = ((((half_t)condval_10) * half_t(0x1p-1f/*5.000000e-01*/)) * half_t(-0x1p+0f/*-1.000000e+00*/));
      } else {
        int condval_11;
        if ((0 < ((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) >> 1) & 3))) {
          condval_11 = (((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_11 = (((((int)Bq_shared_local_cast_10[vec_3]) >> (vec_3 * 4)) & 15) & 1);
        }
        condval_9 = (((half_t)condval_11) * half_t(0x1p-1f/*5.000000e-01*/));
      }
      B_shared_local_cast_9[vec_3] = (condval_9 * Sc_shared_local_cast_11[vec_3]);
    }
    *(uint1*)(((half_t*)B_shared) + (((((i_18 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_9 + 0);
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t A_local_3[16];
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_19 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_3[(i_19 * 8)])));
      }
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_20 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_3[(i_20 * 8)])));
      }
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_21 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_21 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_21 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_22 = 0; i_22 < 16; ++i_22) {
    signed char Bq_shared_local_cast_13[2];
    half_t Sc_shared_local_cast_14[2];
    half_t B_shared_local_cast_12[2];
    *(char2*)(Bq_shared_local_cast_13 + 0) = make_char2(((signed char*)Bq_shared)[(((i_22 * 128) + ((int)threadIdx.x)) + 6144)], ((signed char*)Bq_shared)[(((i_22 * 128) + ((int)threadIdx.x)) + 6144)]);
    *(uint1*)(Sc_shared_local_cast_14 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_22 * 16) + (((int)threadIdx.x) >> 3)) + 768)], ((half_t*)Sc_shared)[(((i_22 * 16) + (((int)threadIdx.x) >> 3)) + 768)]));
    for (int vec_4 = 0; vec_4 < 2; ++vec_4) {
      half_t condval_12;
      if ((8 <= ((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15))) {
        int condval_13;
        if ((0 < ((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) >> 1) & 3))) {
          condval_13 = (((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_13 = (((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) & 1);
        }
        condval_12 = ((((half_t)condval_13) * half_t(0x1p-1f/*5.000000e-01*/)) * half_t(-0x1p+0f/*-1.000000e+00*/));
      } else {
        int condval_14;
        if ((0 < ((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) >> 1) & 3))) {
          condval_14 = (((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) & 1) + 2) << (((((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) >> 1) & 3) - 1));
        } else {
          condval_14 = (((((int)Bq_shared_local_cast_13[vec_4]) >> (vec_4 * 4)) & 15) & 1);
        }
        condval_12 = (((half_t)condval_14) * half_t(0x1p-1f/*5.000000e-01*/));
      }
      B_shared_local_cast_12[vec_4] = (condval_12 * Sc_shared_local_cast_14[vec_4]);
    }
    *(uint1*)(((half_t*)B_shared) + (((((i_22 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_12 + 0);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_4[16];
    half_t B_local_4[32];
    for (int ki_4 = 0; ki_4 < 2; ++ki_4) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 2; ++i_23) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + (i_23 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 6144)])), (&(A_local_4[(i_23 * 8)])));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 4; ++i_24) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_24 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_4) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_4[(i_24 * 8)])));
      }
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_25 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_25 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_25 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_25 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 32; ++i_26) {
    half_t C_local_cast_15[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_26 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast_15 + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((((int)threadIdx.x) & 63) >> 5) * 65536)) + ((i_26 >> 4) * 32768)) + ((i_26 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_26 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast_15 + 0);
  }
}


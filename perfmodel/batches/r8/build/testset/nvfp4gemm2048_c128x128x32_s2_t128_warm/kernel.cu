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
  void* Sc_shared = ((void*)((char*)buf_dyn_shmem + 4096));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 5120));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 21504));
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((int)threadIdx.x) * 16)])), (&(Bq[((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024))])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[(((int)threadIdx.x) * 2)])), (&(Sc[((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128))])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 262144) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 2048)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 16)])));
  tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[((((int)threadIdx.x) * 2) + 256)])), (&(Sc[(((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + 2)])));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 62; ++ko) {
    tl::cp_async_wait<3>();
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 16; ++i_3) {
      signed char Bq_shared_local_cast_1[2];
      half_t Sc_shared_local_cast_2[2];
      half_t B_shared_local_cast[2];
      *(char2*)(Bq_shared_local_cast_1 + 0) = make_char2(((signed char*)Bq_shared)[((((ko & 1) * 2048) + (i_3 * 128)) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((((ko & 1) * 2048) + (i_3 * 128)) + ((int)threadIdx.x))]);
      *(uint1*)(Sc_shared_local_cast_2 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((((ko & 1) * 256) + (i_3 * 16)) + (((int)threadIdx.x) >> 3))], ((half_t*)Sc_shared)[((((ko & 1) * 256) + (i_3 * 16)) + (((int)threadIdx.x) >> 3))]));
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
      *(uint1*)(((half_t*)B_shared) + (((((i_3 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast + 0);
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((ko & 1) * 2048) + (((int)threadIdx.x) * 16))])), (&(Bq[((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (ko * 16)) + 32)])));
    tl::cp_async_gs<4>((&(((half_t*)Sc_shared)[(((ko & 1) * 256) + (((int)threadIdx.x) * 2))])), (&(Sc[((((((int)blockIdx.x) * 16384) + (((int)threadIdx.x) * 128)) + (ko * 2)) + 4)])));
    tl::cp_async_commit();
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((ko & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_4 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_4 * 8)])));
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_5 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
        }
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_6 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_6 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((((ko & 1) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 262144) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    signed char Bq_shared_local_cast_4[2];
    half_t Sc_shared_local_cast_5[2];
    half_t B_shared_local_cast_3[2];
    *(char2*)(Bq_shared_local_cast_4 + 0) = make_char2(((signed char*)Bq_shared)[((i_8 * 128) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((i_8 * 128) + ((int)threadIdx.x))]);
    *(uint1*)(Sc_shared_local_cast_5 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((i_8 * 16) + (((int)threadIdx.x) >> 3))], ((half_t*)Sc_shared)[((i_8 * 16) + (((int)threadIdx.x) >> 3))]));
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
    *(uint1*)(((half_t*)B_shared) + (((((i_8 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_3 + 0);
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_9 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_1[(i_9 * 8)])));
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_10 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_1[(i_10 * 8)])));
      }
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_11 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_11 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_12 = 0; i_12 < 16; ++i_12) {
    signed char Bq_shared_local_cast_7[2];
    half_t Sc_shared_local_cast_8[2];
    half_t B_shared_local_cast_6[2];
    *(char2*)(Bq_shared_local_cast_7 + 0) = make_char2(((signed char*)Bq_shared)[(((i_12 * 128) + ((int)threadIdx.x)) + 2048)], ((signed char*)Bq_shared)[(((i_12 * 128) + ((int)threadIdx.x)) + 2048)]);
    *(uint1*)(Sc_shared_local_cast_8 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_12 * 16) + (((int)threadIdx.x) >> 3)) + 256)], ((half_t*)Sc_shared)[(((i_12 * 16) + (((int)threadIdx.x) >> 3)) + 256)]));
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
    *(uint1*)(((half_t*)B_shared) + (((((i_12 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast_6 + 0);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_2[(i_13 * 8)])));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((int)threadIdx.x) >> 6) * 2048) + (i_14 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(B_local_2[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    half_t C_local_cast_9[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_16 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast_9 + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 262144) + (((((int)threadIdx.x) & 63) >> 5) * 131072)) + ((i_16 >> 4) * 32768)) + ((i_16 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast_9 + 0);
  }
}


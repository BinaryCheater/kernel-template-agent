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
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 9216));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 25600));
  float C_local[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((int)threadIdx.x) * 16)])), (&(Bq[((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024))])));
  __syncthreads();
  ((half_t*)Sc_shared)[((int)threadIdx.x)] = Sc[((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64))];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 131072) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 2048)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 16)])));
  __syncthreads();
  ((half_t*)Sc_shared)[(((int)threadIdx.x) + 128)] = Sc[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + 1)];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 4096)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 32)])));
  __syncthreads();
  ((half_t*)Sc_shared)[(((int)threadIdx.x) + 256)] = Sc[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + 2)];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 131072) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 6144)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 48)])));
  __syncthreads();
  ((half_t*)Sc_shared)[(((int)threadIdx.x) + 384)] = Sc[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + 3)];
  tl::cp_async_commit();
  __syncthreads();
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
      *(uint1*)(Sc_shared_local_cast_2 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((((ko & 3) * 128) + (i_5 * 8)) + (((int)threadIdx.x) >> 4))], ((half_t*)Sc_shared)[((((ko & 3) * 128) + (i_5 * 8)) + (((int)threadIdx.x) >> 4))]));
      int broadcast_var_1 = 15;
      half_t broadcast_var_2 = half_t(0x1p+3f/*8.000000e+00*/);
      uint1 __1;
        uint1 __2;
          uint1 __3;
          int2 __4;
            int2 __5;
              int2 __6;
              char2 v_ = *(char2*)(Bq_shared_local_cast_1 + 0);
              __6.x = (int)(v_.x);
              __6.y = (int)(v_.y);
              int2 v__1 = make_int2((0)+(4*0), (0)+(4*1));
              __5.x = (__6.x >> v__1.x);
              __5.y = (__6.y >> v__1.y);
            int2 v__2 = make_int2(broadcast_var_1, broadcast_var_1);
            __4.x = (__5.x & v__2.x);
            __4.y = (__5.y & v__2.y);
          ((half2*)(&(__3.x)))->x = (half_t)(__4.x);
          ((half2*)(&(__3.x)))->y = (half_t)(__4.y);
          uint1 v__3 = make_uint1(__pack_half2(broadcast_var_2, broadcast_var_2));
          *(uint1*)(&(__2.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__3.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__3.x)))));
        uint1 v__4 = *(uint1*)(Sc_shared_local_cast_2 + 0);
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__2.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__4.x)))));
      *(uint1*)(B_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)B_shared) + (((((i_5 * 256) + ((((int)threadIdx.x) >> 4) * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(B_shared_local_cast + 0);
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((ko & 3) * 2048) + (((int)threadIdx.x) * 16))])), (&(Bq[((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (ko * 16)) + 64)])));
    __syncthreads();
    ((half_t*)Sc_shared)[(((ko & 3) * 128) + ((int)threadIdx.x))] = Sc[((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + ko) + 4)];
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
    *(uint1*)(Sc_shared_local_cast_5 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((i_10 * 8) + (((int)threadIdx.x) >> 4))], ((half_t*)Sc_shared)[((i_10 * 8) + (((int)threadIdx.x) >> 4))]));
    int broadcast_var_3 = 15;
    half_t broadcast_var_4 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __7;
      uint1 __8;
        uint1 __9;
        int2 __10;
          int2 __11;
            int2 __12;
            char2 v__5 = *(char2*)(Bq_shared_local_cast_4 + 0);
            __12.x = (int)(v__5.x);
            __12.y = (int)(v__5.y);
            int2 v__6 = make_int2((0)+(4*0), (0)+(4*1));
            __11.x = (__12.x >> v__6.x);
            __11.y = (__12.y >> v__6.y);
          int2 v__7 = make_int2(broadcast_var_3, broadcast_var_3);
          __10.x = (__11.x & v__7.x);
          __10.y = (__11.y & v__7.y);
        ((half2*)(&(__9.x)))->x = (half_t)(__10.x);
        ((half2*)(&(__9.x)))->y = (half_t)(__10.y);
        uint1 v__8 = make_uint1(__pack_half2(broadcast_var_4, broadcast_var_4));
        *(uint1*)(&(__8.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__9.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__8.x)))));
      uint1 v__9 = *(uint1*)(Sc_shared_local_cast_5 + 0);
      *(uint1*)(&(__7.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__8.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__9.x)))));
    *(uint1*)(B_shared_local_cast_3 + 0) = __7;
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
    *(uint1*)(Sc_shared_local_cast_8 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_14 * 8) + (((int)threadIdx.x) >> 4)) + 128)], ((half_t*)Sc_shared)[(((i_14 * 8) + (((int)threadIdx.x) >> 4)) + 128)]));
    int broadcast_var_5 = 15;
    half_t broadcast_var_6 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __13;
      uint1 __14;
        uint1 __15;
        int2 __16;
          int2 __17;
            int2 __18;
            char2 v__10 = *(char2*)(Bq_shared_local_cast_7 + 0);
            __18.x = (int)(v__10.x);
            __18.y = (int)(v__10.y);
            int2 v__11 = make_int2((0)+(4*0), (0)+(4*1));
            __17.x = (__18.x >> v__11.x);
            __17.y = (__18.y >> v__11.y);
          int2 v__12 = make_int2(broadcast_var_5, broadcast_var_5);
          __16.x = (__17.x & v__12.x);
          __16.y = (__17.y & v__12.y);
        ((half2*)(&(__15.x)))->x = (half_t)(__16.x);
        ((half2*)(&(__15.x)))->y = (half_t)(__16.y);
        uint1 v__13 = make_uint1(__pack_half2(broadcast_var_6, broadcast_var_6));
        *(uint1*)(&(__14.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__15.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__13.x)))));
      uint1 v__14 = *(uint1*)(Sc_shared_local_cast_8 + 0);
      *(uint1*)(&(__13.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__14.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__14.x)))));
    *(uint1*)(B_shared_local_cast_6 + 0) = __13;
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
    *(uint1*)(Sc_shared_local_cast_11 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_18 * 8) + (((int)threadIdx.x) >> 4)) + 256)], ((half_t*)Sc_shared)[(((i_18 * 8) + (((int)threadIdx.x) >> 4)) + 256)]));
    int broadcast_var_7 = 15;
    half_t broadcast_var_8 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __19;
      uint1 __20;
        uint1 __21;
        int2 __22;
          int2 __23;
            int2 __24;
            char2 v__15 = *(char2*)(Bq_shared_local_cast_10 + 0);
            __24.x = (int)(v__15.x);
            __24.y = (int)(v__15.y);
            int2 v__16 = make_int2((0)+(4*0), (0)+(4*1));
            __23.x = (__24.x >> v__16.x);
            __23.y = (__24.y >> v__16.y);
          int2 v__17 = make_int2(broadcast_var_7, broadcast_var_7);
          __22.x = (__23.x & v__17.x);
          __22.y = (__23.y & v__17.y);
        ((half2*)(&(__21.x)))->x = (half_t)(__22.x);
        ((half2*)(&(__21.x)))->y = (half_t)(__22.y);
        uint1 v__18 = make_uint1(__pack_half2(broadcast_var_8, broadcast_var_8));
        *(uint1*)(&(__20.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__21.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__18.x)))));
      uint1 v__19 = *(uint1*)(Sc_shared_local_cast_11 + 0);
      *(uint1*)(&(__19.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__20.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__19.x)))));
    *(uint1*)(B_shared_local_cast_9 + 0) = __19;
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
    *(uint1*)(Sc_shared_local_cast_14 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_22 * 8) + (((int)threadIdx.x) >> 4)) + 384)], ((half_t*)Sc_shared)[(((i_22 * 8) + (((int)threadIdx.x) >> 4)) + 384)]));
    int broadcast_var_9 = 15;
    half_t broadcast_var_10 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __25;
      uint1 __26;
        uint1 __27;
        int2 __28;
          int2 __29;
            int2 __30;
            char2 v__20 = *(char2*)(Bq_shared_local_cast_13 + 0);
            __30.x = (int)(v__20.x);
            __30.y = (int)(v__20.y);
            int2 v__21 = make_int2((0)+(4*0), (0)+(4*1));
            __29.x = (__30.x >> v__21.x);
            __29.y = (__30.y >> v__21.y);
          int2 v__22 = make_int2(broadcast_var_9, broadcast_var_9);
          __28.x = (__29.x & v__22.x);
          __28.y = (__29.y & v__22.y);
        ((half2*)(&(__27.x)))->x = (half_t)(__28.x);
        ((half2*)(&(__27.x)))->y = (half_t)(__28.y);
        uint1 v__23 = make_uint1(__pack_half2(broadcast_var_10, broadcast_var_10));
        *(uint1*)(&(__26.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__27.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__23.x)))));
      uint1 v__24 = *(uint1*)(Sc_shared_local_cast_14 + 0);
      *(uint1*)(&(__25.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__26.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__24.x)))));
    *(uint1*)(B_shared_local_cast_12 + 0) = __25;
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
    uint1 __31;
    float2 v__25 = *(float2*)(C_local + (i_26 * 2));
    ((half2*)(&__31))[0] = __float22half2_rn(((float2*)(&v__25))[0]);
    *(uint1*)(C_local_cast_15 + 0) = __31;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((((int)threadIdx.x) & 63) >> 5) * 65536)) + ((i_26 >> 4) * 32768)) + ((i_26 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_26 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast_15 + 0);
  }
}


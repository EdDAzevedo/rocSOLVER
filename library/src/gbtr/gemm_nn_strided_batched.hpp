/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
#ifndef GEMM_NN_STRIDED_BATCHED_H
#define GEMM_NN_STRIDED_BATCHED_H


#include "gbtr_common.h"
#include "gemm_nn.hpp"
#include "cal_index.h"

template< typename T>
GLOBAL_FUNCTION void
gemm_nn_strided_batched_kernel(
rocblas_int const m,
rocblas_int const n,
rocblas_int const k,
rocblas_int const batchCount,
T const alpha,
T const * const A_,
rocblas_int const ldA,
rocblas_long const strideA,

T const * const B_,
rocblas_int const ldB,
rocblas_long const strideB,

T const beta,
T  *C_,
rocblas_int const ldC,
rocblas_long const strideC
)
{


rocblas_long ci_A,cj_A,ck_A;
rocblas_long ci_B,cj_B,ck_B;
rocblas_long ci_C,cj_C,ck_C;

rocblas_int const nrowA = m; 
rocblas_int const ncolA = k;
rocblas_int const nrowB = k;
rocblas_int const ncolB = n;
rocblas_int const nrowC = m;
rocblas_int const ncolC = n;

CAL_INDEX(nrowA,ncolA, ldA, strideA, batchCount, ci_A, cj_A, ck_A );
CAL_INDEX(nrowB,ncolB, ldB, strideB, batchCount, ci_B, cj_B, ck_B );
CAL_INDEX(nrowC,ncolC, ldC, strideC, batchCount, ci_C, cj_C, ck_C );

bool const is_kij_A = is_kij(nrowA,ncolA, ldA,strideA,batchCount);
bool const is_kij_B = is_kij(nrowB,ncolB, ldB,strideB,batchCount);
bool const is_kij_C = is_kij(nrowC,ncolC, ldC,strideC,batchCount);
bool const is_all_kij = is_kij_A && is_kij_B && is_kij_C;


#define A(i,j,k) A_[ ci_A*i + cj_A*j + ck_A*k]
#define B(i,j,k) B_[ ci_B*i + cj_B*j + ck_B*k]
#define C(i,j,k) C_[ ci_C*i + cj_C*j + ck_C*k]

#ifdef USE_GPU
rocblas_int const iv_start = threadIdx.x + blockIdx.x * blockDim.x;
rocblas_int const iv_inc = gridDim.x * blockDim.x;
#else
rocblas_int const iv_start = 0;
rocblas_int const iv_inc = 1;
#endif

T const zero = 0;

bool is_beta_zero = (beta == zero);


if (is_all_kij) {
  for(rocblas_int jc=0; jc < ncolC; jc++) {
  for(rocblas_int ic=0; ic < nrowC; ic++) {

      #pragma omp parallel for SIMD
      for(rocblas_int iv=iv_start; iv < batchCount; iv += iv_inc) {
           T cij = zero;
           rocblas_int const ja = 0;
           T const * Ap = &(A(ic,ja,iv));
           T const * Bp = &(B(ja,jc,iv));
           auto const Ap_inc = &(A(ic,ja+1,iv)) - Ap;
           auto const Bp_inc = &(B(ja+1,jc,iv)) - Bp;
             for(rocblas_int ja=0; ja < ncolA; ja++) {
 //              cij += A(ic,ja,iv) * B(ja,jc,iv);
                 cij += (*Ap) * (*Bp);
                 Ap += Ap_inc;
                 Bp += Bp_inc;
               };
  

           if (is_beta_zero) {
              C(ic,jc,iv) = alpha * cij;
              }
           else
             C(ic,jc,iv) = beta*C(ic,jc,iv) + alpha * cij;
              };
     };
     };

  }
else {


 #pragma omp parallel for SIMD
 for(rocblas_int iv=iv_start; iv < batchCount; iv += iv_inc) {
     for(rocblas_int jc=0; jc < ncolC; jc++) {
     for(rocblas_int ic=0; ic < nrowC; ic++) {
       T cij = zero;
       for(rocblas_int ja=0; ja < ncolA; ja++) {
          cij += A(ic,ja,iv) * B(ja,jc,iv);
          };


       if (is_beta_zero) {
          C(ic,jc,iv) = alpha * cij;
          }
       else {
          C(ic,jc,iv) = beta * C(ic,jc,iv) + alpha * cij;
          };
      };
      };
   };
 
     

 };

}


#undef A
#undef B
#undef C

#undef CAL_INDEX
#undef is_ijk
#undef is_ikj
#undef is_kij


#endif

/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
#ifndef GETRF_NPVT_BF_HPP
#define GETRF_NPVT_BF_HPP

#include "gbtr_common.h"

template< typename T>
DEVICE_FUNCTION void
getrf_npvt_bf_device
( 
rocblas_int const batchCount,
rocblas_int const m,
rocblas_int const n,
T *A_,
rocblas_int const lda,
rocblas_int *pinfo
)
{
rocblas_int const min_mn = (m < n) ? m : n;
T const one = 1;
rocblas_int info = 0;

#ifdef USE_GPU
rocblas_int const iv_start = ( blockIdx.x * blockDim.x + threadIdx.x) + 1;
rocblas_int const iv_end = batchCount;
rocblas_int const iv_inc = (gridDim.x * blockDim.x);
#else
rocblas_int const iv_start = 1;
rocblas_int const iv_end = batchCount;
rocblas_int const iv_inc = 1;
#endif

#define A(iv,i,j) A_[ indx3f( iv,i,j, batchCount,lda) ]

T const zero = 0;

  for( rocblas_int j=1; j <= min_mn; j++) {
   rocblas_int const jp1 = j + 1;

   #pragma omp parallel for SIMD reduction(max:info)
   for(rocblas_int iv=iv_start; iv <= iv_end; iv += iv_inc) {
       bool const is_diag_zero = (std::abs(A(iv,j,j)) == zero );
       T const Ujj_iv = is_diag_zero ? one : A(iv,j,j);
       info = is_diag_zero && (info == 0) ? j : info;
  
       for(rocblas_int ia=jp1; ia <= m; ia++) {
             A(iv,ia,j) = A(iv,ia,j) / Ujj_iv;
             };
       };

    SYNCTHREADS;

   for(rocblas_int ja=jp1; ja <= n; ja++) {
   for(rocblas_int ia=jp1; ia <= m; ia++) {

     #pragma omp parallel for SIMD
     for(rocblas_int iv=iv_start; iv <= iv_end; iv += iv_inc) {
        A(iv,ia,ja) = A(iv,ia,ja) - A(iv,ia,j) * A(iv,j,ja);
        };

      };
      };

      SYNCTHREADS;
  };

*pinfo = info;
}
#undef A


#endif

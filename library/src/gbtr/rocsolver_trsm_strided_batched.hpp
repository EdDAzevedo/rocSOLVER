
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas/rocblas.h"
#include <assert.h>

#pragma once
template<typename T>
rocblas_status
rocsolver_trsm_strided_batched(
      rocblas_handle handle,
      rocblas_side side,
      rocblas_fill uplo,
      rocblas_operation transA,
      rocblas_diagonal diag,
      rocblas_int m,
      rocblas_int n,
      const T* alpha,
      const T* A,
      rocblas_int lda,
      rocblas_stride strideA,
      T* B,
      rocblas_int ldb,
      rocblas_stride strideB,
      rocblas_int batch_count )
{
    #pragma unused(handle,side,uplo,transA,diag,m,n,alpha)
    #pragma unused(A,lda,strideA,B,ldb,strideB,batch_count)
    assert( false );
    return( rocblas_status_not_implemented );
}
   

template<>
rocblas_status
rocsolver_trsm_strided_batched(
      rocblas_handle handle,
      rocblas_side side,
      rocblas_fill uplo,
      rocblas_operation transA,
      rocblas_diagonal diag,
      rocblas_int m,
      rocblas_int n,
      const double* alpha,
      const double* A,
      rocblas_int lda,
      rocblas_stride strideA,
      double* B,
      rocblas_int ldb,
      rocblas_stride strideB,
      rocblas_int batch_count )
{
  return( rocblas_dtrsm_strided_batched(
              handle,
              side, uplo, transA, diag,
              m,n, alpha,
              A, lda, strideA,
              B, ldb, strideB,
              batch_count )
         );
}



template<>
rocblas_status
rocsolver_trsm_strided_batched(
      rocblas_handle handle,
      rocblas_side side,
      rocblas_fill uplo,
      rocblas_operation transA,
      rocblas_diagonal diag,
      rocblas_int m,
      rocblas_int n,
      const float* alpha,
      const float* A,
      rocblas_int lda,
      rocblas_stride strideA,
      float* B,
      rocblas_int ldb,
      rocblas_stride strideB,
      rocblas_int batch_count )
{
  return( rocblas_strsm_strided_batched(
              handle,
              side, uplo, transA, diag,
              m,n, alpha,
              A, lda, strideA,
              B, ldb, strideB,
              batch_count )
         );
}



template<>
rocblas_status
rocsolver_trsm_strided_batched(
      rocblas_handle handle,
      rocblas_side side,
      rocblas_fill uplo,
      rocblas_operation transA,
      rocblas_diagonal diag,
      rocblas_int m,
      rocblas_int n,
      const rocblas_double_complex* alpha,
      const rocblas_double_complex* A,
      rocblas_int lda,
      rocblas_stride strideA,
      rocblas_double_complex* B,
      rocblas_int ldb,
      rocblas_stride strideB,
      rocblas_int batch_count )
{
  return( rocblas_ztrsm_strided_batched(
              handle,
              side, uplo, transA, diag,
              m,n, alpha,
              A, lda, strideA,
              B, ldb, strideB,
              batch_count )
         );
}



template<>
rocblas_status
rocsolver_trsm_strided_batched(
      rocblas_handle handle,
      rocblas_side side,
      rocblas_fill uplo,
      rocblas_operation transA,
      rocblas_diagonal diag,
      rocblas_int m,
      rocblas_int n,
      const rocblas_float_complex* alpha,
      const rocblas_float_complex* A,
      rocblas_int lda,
      rocblas_stride strideA,
      rocblas_float_complex* B,
      rocblas_int ldb,
      rocblas_stride strideB,
      rocblas_int batch_count )
{
  return( rocblas_ctrsm_strided_batched(
              handle,
              side, uplo, transA, diag,
              m,n, alpha,
              A, lda, strideA,
              B, ldb, strideB,
              batch_count )
         );
}

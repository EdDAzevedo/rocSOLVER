
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas/rocblas.h"
#include <assert.h>

template <typename T>
rocblas_status rocsolver_gemm_batched_impl(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const T* alpha,

                                      const T* const A_array[],
                                      rocblas_int offsetA, 
                                      rocblas_int lda,

                                      const T* const B_array[],
                                      rocblas_int offsetB,
                                      rocblas_int ldb,

                                      const T* beta,

                                      T* const C_array[],
                                      rocblas_int offsetC,
                                      rocblas_int ldc,

                                      rocblas_int batch_count)
{

  rocblas_status istat = rocblas_status_success;
  {
  // -------------------------
  // adjust the pointer arrays
  // -------------------------

  bool const is_add = true;
  istat = rocsolver_adjust_batch( handle, is_add, A_array, offsetA, batch_count);
  if (istat != rocblas_status_success) { return(istat); };

  istat = rocsolver_adjust_batch( handle, is_add, B_array, offsetB, batch_count);
  if (istat != rocblas_status_success) { return(istat); };

  istat = rocsolver_adjust_batch( handle, is_add, C_array, offsetC, batch_count);
  if (istat != rocblas_status_success) { return(istat); };
  };


  {
  // ----------------------------
  // call to rocblas GEMM batched
  // ----------------------------
  istat = rocblas_gemm_batched( handle, transA, transB, m,n,k,
                                        alpha, A_array, lda, B_array, ldb,
                                        beta,  C_array, ldc, 
                                               batch_count );

  if (istat != rocblas_status_success) { return(istat); };
  };


  {
  // --------------------------
  // restore the pointer arrays
  // --------------------------
  bool const is_add = false;
  istat = rocsolver_adjust_batch( handle, is_add, A_array, offsetA, batch_count);
  if (istat != rocblas_status_success) { return(istat); };

  istat = rocsolver_adjust_batch( handle, is_add, B_array, offsetB, batch_count);
  if (istat != rocblas_status_success) { return(istat); };

  istat = rocsolver_adjust_batch( handle, is_add, C_array, offsetC, batch_count);
  if (istat != rocblas_status_success) { return(istat); };
  };


  return( rocblas_status_success );

}

template<typename T>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const T* alpha,
                                      const T* const A[],
                                      rocblas_int lda,
                                      const T* const B[],
                                      rocblas_int ldb,
                                      const T* beta,
                                      T* const C[],
                                      rocblas_int ldc,
                                      rocblas_int batch_count)
{
#pragma unused(handle,transA,transB,m,n,k,alpha)
#pragma unused(A,lda,B,ldb,beta,C,ldc,batch_count)

  return( rocblas_status_not_implemented );
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const double* alpha,
                                      const double* const A[],
                                      rocblas_int lda,
                                      const double* const B[],
                                      rocblas_int ldb,
                                      const double* beta,
                                      double* const C[],
                                      rocblas_int ldc,
                                      rocblas_int batch_count)
{
    return (rocblas_dgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const float* alpha,
                                      const float* const A[],
                                      rocblas_int lda,
                                      const float* const B[],
                                      rocblas_int ldb,
                                      const float* beta,
                                      float* const C[],
                                      rocblas_int ldc,
                                      rocblas_int batch_count)
{
    return (rocblas_sgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const rocblas_double_complex* alpha,
                                      const rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      const rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      const rocblas_double_complex* beta,
                                      rocblas_double_complex* const C[],
                                      rocblas_int ldc,
                                      rocblas_int batch_count)
{
    return (rocblas_zgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int m,
                                      rocblas_int n,
                                      rocblas_int k,
                                      const rocblas_float_complex* alpha,
                                      const rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      const rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      const rocblas_float_complex* beta,
                                      rocblas_float_complex* const C[],
                                      rocblas_int ldc,
                                      rocblas_int batch_count)
{
    return (rocblas_cgemm_batched(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                  ldc, batch_count));
}

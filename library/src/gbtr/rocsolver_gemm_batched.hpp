
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas/rocblas.h"
#include <assert.h>

template <typename T>
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
#pragma unused(handle, transA, transB, m, n, k, alpha, beta, batch_count)
#pragma unused(A, lda)
#pragma unused(B, ldb)
#pragma unused(C, ldc)
    assert(false);
    return (rocblas_status_not_implemented);
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

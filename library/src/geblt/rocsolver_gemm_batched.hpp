
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef ROCSOLVER_GEMM_BATCHED_HPP
#define ROCSOLVER_GEMM_BATCHED_HPP

#include "rocblas/rocblas.h"
#include <assert.h>

template <typename T>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      const rocblas_operation transA,
                                      const rocblas_operation transB,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,
                                      T* alpha,
                                      T* A_array[],
                                      const rocblas_int lda,
                                      T* B_array[],
                                      const rocblas_int ldb,
                                      T* beta,
                                      T* C_array[],
                                      const rocblas_int ldc,
                                      const rocblas_int batch_count)
{
#pragma unused(handle, transA, transB, m, n, k, alpha)
#pragma unused(A_array, lda, B_array, ldb, beta, C_array, ldc, batch_count)

    return (rocblas_status_not_implemented);
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      const rocblas_operation transA,
                                      const rocblas_operation transB,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,

                                      double* alpha,

                                      double* A_array[],
                                      const rocblas_int lda,

                                      double* B_array[],
                                      const rocblas_int ldb,

                                      double* beta,
                                      double* C_array[],

                                      const rocblas_int ldc,
                                      const rocblas_int batch_count)
{
    return (rocblas_dgemm_batched(handle, transA, transB, m, n, k, alpha, A_array, lda, B_array,
                                  ldb, beta, C_array, ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      const rocblas_operation transA,
                                      const rocblas_operation transB,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,

                                      float* alpha,

                                      float* A_array[],
                                      const rocblas_int lda,

                                      float* B_array[],
                                      const rocblas_int ldb,

                                      float* beta,
                                      float* C_array[],

                                      const rocblas_int ldc,
                                      const rocblas_int batch_count)
{
    return (rocblas_sgemm_batched(handle, transA, transB, m, n, k, alpha, A_array, lda, B_array,
                                  ldb, beta, C_array, ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      const rocblas_operation transA,
                                      const rocblas_operation transB,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,

                                      rocblas_double_complex* alpha,

                                      rocblas_double_complex* A_array[],
                                      const rocblas_int lda,

                                      rocblas_double_complex* B_array[],
                                      const rocblas_int ldb,

                                      rocblas_double_complex* beta,
                                      rocblas_double_complex* C_array[],

                                      const rocblas_int ldc,
                                      const rocblas_int batch_count)
{
    return (rocblas_zgemm_batched(handle, transA, transB, m, n, k, alpha, A_array, lda, B_array,
                                  ldb, beta, C_array, ldc, batch_count));
}

template <>
rocblas_status rocsolver_gemm_batched(rocblas_handle handle,
                                      const rocblas_operation transA,
                                      const rocblas_operation transB,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      const rocblas_int k,

                                      rocblas_float_complex* alpha,

                                      rocblas_float_complex* A_array[],
                                      const rocblas_int lda,

                                      rocblas_float_complex* B_array[],
                                      const rocblas_int ldb,

                                      rocblas_float_complex* beta,
                                      rocblas_float_complex* C_array[],

                                      const rocblas_int ldc,
                                      const rocblas_int batch_count)
{
    return (rocblas_cgemm_batched(handle, transA, transB, m, n, k, alpha, A_array, lda, B_array,
                                  ldb, beta, C_array, ldc, batch_count));
}
#endif


/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "rocblas/rocblas.h"
#include <assert.h>

template <typename T>
rocblas_status rocsolver_gemm_strided_batched(rocblas_handle handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const T* alpha,
                                              const T* A,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              const T* B,
                                              rocblas_int ldb,
                                              rocblas_stride strideB,
                                              const T* beta,
                                              T* C,
                                              rocblas_int ldc,
                                              rocblas_stride strideC,
                                              rocblas_int batch_count)
{
#pragma unused(handle, transA, transB, m, n, k, alpha, beta, batch_count)
#pragma unused(A, lda, strideA)
#pragma unused(B, ldb, strideB)
#pragma unused(C, ldc, strideC)
    assert(false);
    return (rocblas_status_not_implemented);
}

template <>
rocblas_status rocsolver_gemm_strided_batched(rocblas_handle handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const double* alpha,
                                              const double* A,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              const double* B,
                                              rocblas_int ldb,
                                              rocblas_stride strideB,
                                              const double* beta,
                                              double* C,
                                              rocblas_int ldc,
                                              rocblas_stride strideC,
                                              rocblas_int batch_count)
{
    return (rocblas_dgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                          B, ldb, strideB, beta, C, ldc, strideC, batch_count));
}

template <>
rocblas_status rocsolver_gemm_strided_batched(rocblas_handle handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const float* alpha,
                                              const float* A,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              const float* B,
                                              rocblas_int ldb,
                                              rocblas_stride strideB,
                                              const float* beta,
                                              float* C,
                                              rocblas_int ldc,
                                              rocblas_stride strideC,
                                              rocblas_int batch_count)
{
    return (rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                          B, ldb, strideB, beta, C, ldc, strideC, batch_count));
}

template <>
rocblas_status rocsolver_gemm_strided_batched(rocblas_handle handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const rocblas_double_complex* alpha,
                                              const rocblas_double_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              const rocblas_double_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride strideB,
                                              const rocblas_double_complex* beta,
                                              rocblas_double_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride strideC,
                                              rocblas_int batch_count)
{
    return (rocblas_zgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                          B, ldb, strideB, beta, C, ldc, strideC, batch_count));
}

template <>
rocblas_status rocsolver_gemm_strided_batched(rocblas_handle handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int m,
                                              rocblas_int n,
                                              rocblas_int k,
                                              const rocblas_float_complex* alpha,
                                              const rocblas_float_complex* A,
                                              rocblas_int lda,
                                              rocblas_stride strideA,
                                              const rocblas_float_complex* B,
                                              rocblas_int ldb,
                                              rocblas_stride strideB,
                                              const rocblas_float_complex* beta,
                                              rocblas_float_complex* C,
                                              rocblas_int ldc,
                                              rocblas_stride strideC,
                                              rocblas_int batch_count)
{
    return (rocblas_cgemm_strided_batched(handle, transA, transB, m, n, k, alpha, A, lda, strideA,
                                          B, ldb, strideB, beta, C, ldc, strideC, batch_count));
}

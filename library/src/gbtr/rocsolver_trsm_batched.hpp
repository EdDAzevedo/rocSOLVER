
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas/rocblas.h"
#include <assert.h>

/*
rocblas_status rocblas_strsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float *alpha, const float *const A[], rocblas_int lda, float *const B[], rocblas_int ldb, rocblas_int batch_count)
*/

#pragma once
template <typename T>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      const T* alpha,
                                      const T* const A[],
                                      rocblas_int lda,
                                      T* const B[],
                                      rocblas_int ldb,
                                      rocblas_int batch_count)
{
#pragma unused(handle, side, uplo, transA, diag, m, n, alpha)
#pragma unused(A, lda, B, ldb, batch_count)
    assert(false);
    return (rocblas_status_not_implemented);
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      const double* alpha,
                                      const double* const A[],
                                      rocblas_int lda,
                                      double* const B[],
                                      rocblas_int ldb,
                                      rocblas_int batch_count)
{
    return (rocblas_dtrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      const float* alpha,
                                      const float* const A[],
                                      rocblas_int lda,
                                      float* const B[],
                                      rocblas_int ldb,
                                      rocblas_int batch_count)
{
    return (rocblas_strsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      const rocblas_double_complex* alpha,
                                      const rocblas_double_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_double_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_int batch_count)
{
    return (rocblas_ztrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      rocblas_side side,
                                      rocblas_fill uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal diag,
                                      rocblas_int m,
                                      rocblas_int n,
                                      const rocblas_float_complex* alpha,
                                      const rocblas_float_complex* const A[],
                                      rocblas_int lda,
                                      rocblas_float_complex* const B[],
                                      rocblas_int ldb,
                                      rocblas_int batch_count)
{
    return (rocblas_ctrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

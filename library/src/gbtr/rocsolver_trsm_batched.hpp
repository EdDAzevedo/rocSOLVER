
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas/rocblas.h"
#include <assert.h>

#pragma once
#ifndef ROCSOLVER_TRSM_BATCHED_HPP
#define ROCSOLVER_TRSM_BATCHED_HPP

template <typename T>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_fill uplo,
                                      const rocblas_operation transA,
                                      const rocblas_diagonal diag,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      T* alpha,
                                      T* A[],
                                      const rocblas_int lda,
                                      T* B[],
                                      const rocblas_int ldb,
                                      const rocblas_int batch_count)
{
#pragma unused(handle, side, uplo, transA, diag, m, n, alpha)
#pragma unused(A, lda, B, ldb, batch_count)
    assert(false);
    return (rocblas_status_not_implemented);
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_fill uplo,
                                      const rocblas_operation transA,
                                      const rocblas_diagonal diag,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      double* alpha,
                                      double* A[],
                                      const rocblas_int lda,
                                      double* B[],
                                      const rocblas_int ldb,
                                      const rocblas_int batch_count)
{
    return (rocblas_dtrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_fill uplo,
                                      const rocblas_operation transA,
                                      const rocblas_diagonal diag,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      float* alpha,
                                      float* A[],
                                      const rocblas_int lda,
                                      float* B[],
                                      const rocblas_int ldb,
                                      const rocblas_int batch_count)
{
    return (rocblas_strsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_fill uplo,
                                      const rocblas_operation transA,
                                      const rocblas_diagonal diag,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      rocblas_double_complex* alpha,
                                      rocblas_double_complex* A[],
                                      const rocblas_int lda,
                                      rocblas_double_complex* B[],
                                      const rocblas_int ldb,
                                      const rocblas_int batch_count)
{
    return (rocblas_ztrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

template <>
rocblas_status rocsolver_trsm_batched(rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_fill uplo,
                                      const rocblas_operation transA,
                                      const rocblas_diagonal diag,
                                      const rocblas_int m,
                                      const rocblas_int n,
                                      rocblas_float_complex* alpha,
                                      rocblas_float_complex*  A[],
                                      const rocblas_int lda,
                                      rocblas_float_complex*  B[],
                                      const rocblas_int ldb,
                                      const rocblas_int batch_count)
{
    return (rocblas_ctrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb,
                                  batch_count));
}

#endif

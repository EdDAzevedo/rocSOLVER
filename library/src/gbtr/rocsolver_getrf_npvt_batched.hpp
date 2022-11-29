/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "assert.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#pragma once

/*
rocblas_status rocsolver_zgetrf_npvt_batched(rocblas_handle handle, const rocblas_int m, const rocblas_int n, rocblas_double_complex *const A[], const rocblas_int lda, rocblas_int *info, const rocblas_int batch_count)
*/

template <typename T>
rocblas_status rocsolver_getrf_npvt_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            T* const A_[],
                                            const rocblas_int lda,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{
#pragma unused(handle, m, n, A_, lda, info, batch_count)
    assert(false);
    return (rocblas_status_not_implemented);
};

template <>
rocblas_status rocsolver_getrf_npvt_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            double* const A_[],
                                            const rocblas_int lda,
                                            rocblas_int* info,
                                            const rocblas_int batch_count)
{
    return (rocsolver_dgetrf_npvt_batched(handle, m, n, A_, lda, info, batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            float* const A_[],
                                            const rocblas_int lda,
                                            rocblas_int info[],
                                            const rocblas_int batch_count)
{
    return (rocsolver_sgetrf_npvt_batched(handle, m, n, A_, lda, info, batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_double_complex* const A_[],
                                            const rocblas_int lda,
                                            rocblas_int info[],
                                            const rocblas_int batch_count)
{
    return (rocsolver_zgetrf_npvt_batched(handle, m, n, A_, lda, info, batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_batched(rocblas_handle handle,
                                            const rocblas_int m,
                                            const rocblas_int n,
                                            rocblas_float_complex* const A_[],
                                            const rocblas_int lda,
                                            rocblas_int info[],
                                            const rocblas_int batch_count)
{
    return (rocsolver_cgetrf_npvt_batched(handle, m, n, A_, lda, info, batch_count));
}

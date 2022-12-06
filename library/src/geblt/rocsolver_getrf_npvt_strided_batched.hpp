/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "assert.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

#pragma once
template <typename T>
rocblas_status rocsolver_getrf_npvt_strided_batched(rocblas_handle handle,
                                                    rocblas_int const m,
                                                    rocblas_int const n,
                                                    T* const A_,
                                                    rocblas_int const lda,
                                                    rocblas_stride const strideA,
                                                    rocblas_int info_array[],
                                                    rocblas_int batch_count)
{
#pragma unused(handle, m, n, A_, lda, strideA, info_array, batch_count)
    assert(false);
    return (rocblas_status_not_implemented);
};

template <>
rocblas_status rocsolver_getrf_npvt_strided_batched(rocblas_handle handle,
                                                    rocblas_int const m,
                                                    rocblas_int const n,
                                                    double* const A_,
                                                    rocblas_int const lda,
                                                    rocblas_stride const strideA,
                                                    rocblas_int info_array[],
                                                    rocblas_int batch_count)
{
    return (rocsolver_dgetrf_npvt_strided_batched(handle, m, n, A_, lda, strideA, info_array,
                                                  batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_strided_batched(rocblas_handle handle,
                                                    rocblas_int const m,
                                                    rocblas_int const n,
                                                    float* const A_,
                                                    rocblas_int const lda,
                                                    rocblas_stride const strideA,
                                                    rocblas_int info_array[],
                                                    rocblas_int batch_count)
{
    return (rocsolver_sgetrf_npvt_strided_batched(handle, m, n, A_, lda, strideA, info_array,
                                                  batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_strided_batched(rocblas_handle handle,
                                                    rocblas_int const m,
                                                    rocblas_int const n,
                                                    rocblas_double_complex* const A_,
                                                    rocblas_int const lda,
                                                    rocblas_stride const strideA,
                                                    rocblas_int info_array[],
                                                    rocblas_int batch_count)
{
    return (rocsolver_zgetrf_npvt_strided_batched(handle, m, n, A_, lda, strideA, info_array,
                                                  batch_count));
}

template <>
rocblas_status rocsolver_getrf_npvt_strided_batched(rocblas_handle handle,
                                                    rocblas_int const m,
                                                    rocblas_int const n,
                                                    rocblas_float_complex* const A_,
                                                    rocblas_int const lda,
                                                    rocblas_stride const strideA,
                                                    rocblas_int info_array[],
                                                    rocblas_int batch_count)
{
    return (rocsolver_cgetrf_npvt_strided_batched(handle, m, n, A_, lda, strideA, info_array,
                                                  batch_count));
}

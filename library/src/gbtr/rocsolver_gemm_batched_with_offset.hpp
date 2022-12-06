
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#ifndef ROCSOLVER_GEMM_BATCHED_WITH_OFFSET_HPP
#define ROCSOLVER_GEMM_BATCHED_WITH_OFFSET_HPP

#include "rocblas/rocblas.h"
#include <assert.h>

#include "rocsolver_gemm_batched.hpp"

template <typename T, typename I>
rocblas_status rocsolver_gemm_batched_with_offset(rocblas_handle handle,
                                                  const rocblas_operation transA,
                                                  const rocblas_operation transB,
                                                  const I m,
                                                  const I n,
                                                  const I k,
                                                  T* alpha,

                                                  T* A_array[],
                                                  const I offsetA,
                                                  const I lda,

                                                  T* B_array[],
                                                  const I offsetB,
                                                  const I ldb,

                                                  T* beta,

                                                  T* C_array[],
                                                  const I offsetC,
                                                  const I ldc,
                                                  const I batch_count)
{
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;
    {
        // -------------------------
        // adjust the pointer arrays
        // -------------------------

        bool const is_add = true;
        istat = rocsolver_adjust_batch(handle, is_add, A_array, offsetA, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };

        istat = rocsolver_adjust_batch(handle, is_add, B_array, offsetB, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };

        istat = rocsolver_adjust_batch(handle, is_add, C_array, offsetC, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };
    };

    {
        // ----------------------------
        // call to rocblas GEMM batched
        // ----------------------------
        istat = rocsolver_gemm_batched(handle, transA, transB, m, n, k, alpha, A_array, lda,
                                       B_array, ldb, beta, C_array, ldc, batch_count);

        if(istat != rocblas_status_success)
        {
            goto L999;
        };
    };

    {
        // --------------------------
        // restore the pointer arrays
        // --------------------------
        bool const is_add = false;
        istat = rocsolver_adjust_batch(handle, is_add, A_array, offsetA, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };

        istat = rocsolver_adjust_batch(handle, is_add, B_array, offsetB, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };

        istat = rocsolver_adjust_batch(handle, is_add, C_array, offsetC, batch_count);
        if(istat != rocblas_status_success)
        {
            goto L999;
        };
    };

L999:

    rocblas_set_pointer_mode(handle, old_mode);
    return (istat);
}
#endif

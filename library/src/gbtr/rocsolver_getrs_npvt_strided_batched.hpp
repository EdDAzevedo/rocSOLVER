/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "rocblas/rocblas.h"

#include "rocsolver_trsm_strided_batched.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_getrs_npvt_strided_batched(rocblas_handle handle,
                                                    I const n,
                                                    I const nrhs,
                                                    T const* A_,
                                                    I const lda,
                                                    Istride const strideA,
                                                    T* const B_,
                                                    I const ldb,
                                                    Istride const strideB,
                                                    I const batch_count)
{
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;

    // -------------------------------------
    // solve L * X = B, overwriting B with X
    // -------------------------------------
    {
        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const trans = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        T alpha = 1;

        istat = rocsolver_trsm_strided_batched(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_,
                                               lda, strideA, B_, ldb, strideB, batch_count);
        if(istat != rocblas_status_success)
        {
            rocblas_set_pointer_mode(handle, old_mode);
            return (istat);
        };
    };

    // -----------------------------------
    // solve U*X = B, overwriting B with X
    // -----------------------------------

    {
        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const trans = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        T alpha = 1;

        istat = rocsolver_trsm_strided_batched(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_,
                                               lda, strideA, B_, ldb, strideB, batch_count);
        if(istat != rocblas_status_success)
        {
            rocblas_set_pointer_mode(handle, old_mode);
            return (istat);
        };
    };

    rocblas_set_pointer_mode(handle, old_mode);
    return (rocblas_status_success);
};

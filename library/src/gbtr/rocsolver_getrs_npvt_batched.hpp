
/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once

#include "rocblas/rocblas.h"

#include "rocsolver_trsm_batched.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_getrs_npvt_batched(rocblas_handle handle,
                                                    I const n,
                                                    I const nrhs,
                                                    T * A_array[],
                                                    I const offsetA,
                                                    I const lda,

                                                    T* const B_array[],
                                                    I const offsetB,
                                                    I const ldb,

                                                    I const batch_count)
{
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    rocblas_status istat = rocblas_status_success;

    {
    bool const is_add = true;

    istat = rocblas_adjust_batch(  handle, is_add, A_array, offsetA, batch_count );
    if (istat != rocblas_status_success) { goto L999; };

    istat = rocblas_adjust_batch(  handle, is_add, B_array, offsetB, batch_count );
    if (istat != rocblas_status_success) { goto L999; };
    };

    // -------------------------------------
    // solve L * X = B, overwriting B with X
    // -------------------------------------
    {
        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const trans = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        T alpha = 1;

        istat = rocsolver_trsm_batched(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_array,
                                               lda, B_array, ldb, batch_count);
        if(istat != rocblas_status_success) { goto L999; };
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

        istat = rocsolver_trsm_batched(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_array,
                                               lda, B_array, ldb,  batch_count);
        if(istat != rocblas_status_success) { goto L999; };
    };


    {
    bool const is_add = false;

    istat = rocblas_adjust_batch(  handle, is_add, A_array, offsetA, batch_count );
    if (istat != rocblas_status_success) { goto L999; };

    istat = rocblas_adjust_batch(  handle, is_add, B_array, offsetB, batch_count );
    if (istat != rocblas_status_success) { goto L999; };
    };

L999:
    rocblas_set_pointer_mode(handle, old_mode);
    return (istat);
};

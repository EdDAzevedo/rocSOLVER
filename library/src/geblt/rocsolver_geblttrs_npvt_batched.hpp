
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef ROCSOVLER_GEBLTTRS_NPVT_BATCHED_HPP
#define ROCSOVLER_GEBLTTRS_NPVT_BATCHED_HPP

#include "geblt_common.h"
#include "rocsolver_checkargs_geblt_npvt_batched.hpp"
#include "rocsolver_geblttrs_npvt_batched_large.hpp"
#include "rocsolver_geblttrs_npvt_batched_small.hpp"

template <typename T, typename I>
rocblas_status rocsolver_geblttrs_npvt_batched_impl(rocblas_handle handle,
                                                    I nb,
                                                    I nblocks,
                                                    I nrhs,
                                                    T* A_array[],
                                                    I lda,
                                                    T* B_array[],
                                                    I ldb,
                                                    T* C_array[],
                                                    I ldc,
                                                    T* X_array[],
                                                    I ldx,
                                                    I batch_count)
{
    rocblas_status istat
        = rocsolver_checkargs_geblt_npvt_batched(handle, nb, nblocks, nrhs, A_array, lda, B_array,
                                                 ldb, C_array, ldc, X_array, ldx, batch_count);
    if(istat != rocblas_status_continue)
    {
        return (istat);
    };

    if(nb < NB_SMALL)
    {
        istat = rocsolver_geblttrs_npvt_batched_small_template(handle, nb, nblocks, nrhs, A_array,
                                                               lda, B_array, ldb, C_array, ldc,
                                                               X_array, ldx, batch_count);
    }
    else
    {
        istat = rocsolver_geblttrs_npvt_batched_large_template(handle, nb, nblocks, nrhs, A_array,
                                                               lda, B_array, ldb, C_array, ldc,
                                                               X_array, ldx, batch_count);
    };

    return (istat);
};
#endif

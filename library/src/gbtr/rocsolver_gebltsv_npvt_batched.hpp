

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
#ifndef ROCSOLVER_GEBLTSV_NPVT_BATCHED_HPP
#define ROCSOLVER_GEBLTSV_NPVT_BATCHED_HPP

#include "geblt_common.h"
#include "rocsolver_geblttrf_npvt_batched.hpp"
#include "rocsolver_geblttrs_npvt_batched.hpp"

template <typename T, typename I>
rocblas_status rocsolver_gebltsv_npvt_batched_impl(rocblas_handle handle,
                                                   const I nb,
                                                   const I nblocks,
                                                   const I nrhs,

                                                   T* A_array[],
                                                   const I lda,

                                                   T* B_array[],
                                                   const I ldb,

                                                   T* C_array[],
                                                   const I ldc,

                                                   T* X_array[],
                                                   const I ldx,

                                                   I info_array[],
                                                   const I batch_count)
{
    rocblas_status istat = rocblas_status_success;

    istat = rocsolver_geblttrf_npvt_batched_impl(handle, nb, nblocks, A_array, lda, B_array, ldb,
                                                 C_array, ldc, info_array, batch_count);
    if(istat != rocblas_status_success)
    {
        return (istat);
    };

    istat = rocsolver_geblttrs_npvt_batched_impl(handle, nb, nblocks, nrhs, A_array, lda, B_array,
                                                 ldb, C_array, ldc, X_array, ldx, batch_count);

    return (istat);
}

#endif

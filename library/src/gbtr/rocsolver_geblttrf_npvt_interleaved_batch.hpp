
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
#ifndef ROCSOLVER_GEBLTTRF_NPVT_INTERLEAVED_BATCH_H
#define ROCSOLVER_GEBLTTRF_NPVT_INTERLEAVED_BATCH_H

#include "geblt_common.h"
#include "geblttrf_npvt_bf.hpp"
#include "rocsolver_checkargs_geblt_npvt_interleaved_batch.hpp"

template <typename T, typename I>
rocblas_status rocsolver_geblttrf_npvt_interleaved_batch_impl(rocblas_handle handle,
                                                              I nb,
                                                              I nblocks,
                                                              T* A_,
                                                              I lda,
                                                              T* B_,
                                                              I ldb,
                                                              T* C_,
                                                              I ldc,
                                                              I devinfo_array[],
                                                              I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_status istat = rocsolver_checkargs_geblt_npvt_interleaved_batch(
        handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batch_count);
    if(istat != rocblas_status_continue)
    {
        return (istat);
    };

    istat = geblttrf_npvt_bf_template(handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array,
                                      batch_count);

    return (istat);
}

#endif
